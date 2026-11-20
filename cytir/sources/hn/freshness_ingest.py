#!/usr/bin/env python3
"""hn_freshness_ingest.py — Hacker News as a LIVE FRESHNESS source for the
news-refresh cadence (task 294). HN is otherwise only an endorsement/discovery
signal (ADR-019 `ingest_hn.py` harvests candidates from CC); this makes it a
first-class fresh feed: new community-endorsed tech/AI links, dated for free.

Source = the OFFICIAL, free HN APIs, NO scraping (sidesteps consent/rate on HN
itself; the LINKED article is still consent-gated before we fetch it):
  * Algolia HN Search  `search_by_date?tags=(story|front_page)`  — time-windowed,
    carries points + created_at. (Server-side points filter 400s on the current
    index — same note as ingest_hn.py — so we filter points client-side.)
  * Firebase `topstories` + `beststories` + `item/<id>` — official fallback
    when Algolia is unavailable.  It is explicitly marked *best-effort-current-
    lists*, because Firebase cannot enumerate an exact historical time window.
    That makes safe additions possible without presenting the fallback as a
    complete HN refresh.  `--source auto` uses this only after Algolia fails.

For each qualifying story we emit ONE AW record = the **linked article URL** (the
authoritative primary source), carrying:
  * url            — the linked article (the AW result the agent fetches)
  * hn_url         — https://news.ycombinator.com/item?id=<objectID> (discussion)
  * hn_score       — points (endorsement/importance signal for the ranker)
  * title, snippet — title + (best-effort, consent-gated) extractive snippet
  * published      — the article's date (extract_published on the fetched HTML)
                     FALLING BACK to the HN submission date (created_at) — free,
                     and never later than the article for a freshly-posted link.
  * disc_class=news + source_type=reference + tier=C  -> rides the 12-month window
    and the delta refresh exactly like the other news sources.

Community score defaults to 40. A hash-bound metadata rule may admit only a
reviewed primary-source URL below it; no domain-wide score bypass is allowed.
MATURATION window (`--since-days`, default 7) — a story posted 3 days ago has had
time to cross the threshold; dedup vs prod by point_id (dump_netnew) means a story
that qualifies on a later day is a single ADD, never a duplicate.

Output: a staging JSONL of gated rows ready for embed_tier1.py -> aw_dhn_cov ->
dump_netnew -> the news-refresh delta wave. Text is NOT mirrored beyond the
snippet (ADR-039); title-only fallback when the linked article can't be fetched
(the URL itself is still the grounding value).

PRESERVATION (task 445). The linked-page GET is an acquisition, so it obeys the
`news_article` preservation profile: the ORIGINAL response bytes, the transport
headers, and the request identity we actually sent (method, exact URL, outbound
headers incl. the crawler UA) are sealed under `--preserve-dir` BEFORE anything
is extracted, exactly as `acquire_hn_firebase_preserved.py` binds the official
Firebase GET. Extraction then reads the *validated* artifact back, so a
linked-article-derived row cannot exist without preserved bytes; a fetch that
cannot be sealed falls back to the title-only path for that row. `--no-fetch`
keeps the title-only lane and needs no preservation root.

  python3 -m cytir.sources.hn.freshness_ingest --min-points 40 --since-days 7 \
      --out <work>/hn_freshness.rows.jsonl \
      --preserve-dir <work>/stage-<day>/hn_linked \
      [--limit N] [--no-fetch]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client
from cytir.acquire.artifact_io import atomic_publish, canonical_json
from cytir.news.extract_published import extract_published
from cytir.acquire import preserved_stage_io as PS
from cytir.core.user_agent import user_agent  # task 432: one home for the crawler identity
try:
    from cytir.core.common import extractive_snippet
    from cytir.extract.html_extract import reg_domain
except Exception:  # noqa: BLE001 — keep the module importable standalone
    def extractive_snippet(t, max_words=30):
        return " ".join((t or "").split()[:max_words])

    def reg_domain(host):
        return host

ALGOLIA = "https://hn.algolia.com/api/v1/search_by_date"
FIREBASE = "https://hacker-news.firebaseio.com/v0"
HN_ITEM = "https://news.ycombinator.com/item?id={}"
UA = {"User-Agent": user_agent()}
# Algolia accepts hitsPerPage=1000 but does not paginate beyond that result
# window.  In particular it can report ``nbPages == 1`` for a 7-day query with
# thousands of hits.  Treat 1,000 as a hard protocol cap, never as a complete
# page that happens to be convenient.
RESULT_CAP = 1000
MAX_TIME_PARTITIONS = 1024
CLASS_TRUST_C = 0.6          # Tier-C community endorsement, matches other news
DEFAULT_COMMUNITY_MIN_POINTS = 40
STRONG_ENDORSEMENT_POINTS = 100
PAGE_GAP = 0.3
FIREBASE_FEEDS = ("topstories", "beststories")
# Same transport schema the official Firebase acquisition seals (commit
# 4ff5fc84) — one shape of request identity for the whole HN lane.
LINKED_TRANSPORT_SCHEMA = "aw.http-api-response.v2"
LINKED_RECEIPT_SCHEMA = "aw.hn-linked-page-receipt.v1"


class HNSourceError(RuntimeError):
    """A source transport or protocol failure, never an empty HN window."""


# --------------------------- consent gate (linked article) ----------------- #
_robots: dict = {}


def consent_ok(url: str) -> bool:
    try:
        p = urlsplit(url)
    except Exception:  # noqa: BLE001
        return False
    if p.scheme not in ("http", "https") or not p.netloc:
        return False
    host_scheme = f"{p.scheme}://{p.netloc}"
    if host_scheme not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(host_scheme + "/robots.txt")
        try:
            with make_client(HttpSpec(timeout=8.0, follow_redirects=True)) as rc:
                r = rc.get(host_scheme + "/robots.txt")
                r.raise_for_status()
                rp.parse(r.content.decode("utf-8", "replace").splitlines())
            _robots[host_scheme] = rp
        except Exception:  # noqa: BLE001 — robots unavailable -> default-allow
            _robots[host_scheme] = None
    rp = _robots[host_scheme]
    return True if rp is None else rp.can_fetch(UA["User-Agent"], url)


# ------------------- linked-page preservation (task 445) ------------------- #
def _decode_html(body: bytes, content_type: str) -> str:
    """Decode the PRESERVED bytes; the bytes, not this text, are the record."""
    charset = ""
    for part in (content_type or "").split(";")[1:]:
        name, _, value = part.partition("=")
        if name.strip().lower() == "charset":
            charset = value.strip().strip('"')
    for candidate in (charset, "utf-8"):
        if candidate:
            try:
                return body.decode(candidate)
            except (LookupError, UnicodeDecodeError):
                continue
    return body.decode("utf-8", "replace")


def seal_linked_fetch(root: Path, url: str, object_id: str,
                      response, fetched_at: str) -> dict:
    """Seal one linked-page response as original bytes + transport + identity."""
    body = bytes(response.content)
    request = response.request
    transport = {
        "schema": LINKED_TRANSPORT_SCHEMA,
        "requested_url": url,
        "request_method": str(request.method),
        "request_url": str(request.url),
        "request_headers": sorted((str(k), str(v))
                                  for k, v in request.headers.multi_items()),
        "final_url": str(response.url),
        "http_status": response.status_code,
        "fetched_at": fetched_at,
        "response_headers": sorted((str(k), str(v))
                                   for k, v in response.headers.multi_items()),
        "body": {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()},
    }
    agents = [value for key, value in transport["request_headers"]
              if key.casefold() == "user-agent"]
    # Consent evidence is only worth something if it names the crawler that
    # actually made the request (task 432); an unbound identity is not sealable.
    if len(agents) != 1 or agents[0] != UA["User-Agent"]:
        raise ValueError(f"linked-page request identity is unbound for {url}")
    return {
        "schema": LINKED_RECEIPT_SCHEMA,
        "hn_object_id": object_id,
        "requested_url": url,
        "final_url": transport["final_url"],
        "http_status": response.status_code,
        "content_type": str(response.headers.get("content-type", "")),
        "fetched_at": fetched_at,
        "artifacts": {
            "body": PS.store_artifact(root, "original/linked_pages", "html", body),
            "transport": PS.store_artifact(
                root, "original/linked_transport", "json",
                (canonical_json(transport) + "\n").encode()),
        },
    }


def _valid_linked_receipt(root: Path, path: Path, url: str) -> dict | None:
    """Reuse an already-sealed fetch (same-day `--resume`) instead of re-GETting
    it; a receipt whose artifacts do not validate is not evidence."""
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (value.get("schema") != LINKED_RECEIPT_SCHEMA
                or value.get("requested_url") != url):
            return None
        for artifact in value["artifacts"].values():
            PS.validate_artifact(root, artifact)
        return value
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError,
            PS.PreservedArtifactError):
        return None


def fetch_article(client: httpx.Client, url: str, preserve_root: Path,
                  object_id: str):
    """Single GET of the linked article, PRESERVED BEFORE IT IS READ.

    The old implementation extracted a snippet in memory and threw the response
    away, so no daily HN slice could ever satisfy the `news_article` profile's
    `original` stage. Here the response bytes, its transport headers and the
    exact outbound request identity are sealed first, and the extraction input
    is read back from the validated artifact — a linked-article-derived row is
    therefore impossible without preserved bytes.

    -> (html, lang, receipt). A fetch that cannot be sealed returns
    (None, None, None): the caller falls back to the title-only path.
    """
    # The immutable HN item id is also the receipt's name; a non-canonical one
    # is a broken source boundary, not a row to degrade quietly.
    if not object_id.isdigit():
        raise HNSourceError(f"non-canonical HN objectID {object_id!r}")
    receipt_path = preserve_root / "receipts" / f"{object_id}.json"
    receipt = _valid_linked_receipt(preserve_root, receipt_path, url)
    if receipt is None:
        try:
            fetched_at = dt.datetime.now(dt.timezone.utc).isoformat().replace(
                "+00:00", "Z")
            response = client.get(url, headers=UA, follow_redirects=True,
                                  timeout=20)
            receipt = seal_linked_fetch(preserve_root, url, object_id, response,
                                        fetched_at)
            encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
            if receipt_path.exists():
                raise ValueError(f"linked-page receipt collision: {receipt_path}")
            atomic_publish(receipt_path, encoded)
        except Exception:  # noqa: BLE001 — unpreservable fetch -> title-only
            return None, None, None
    if receipt["http_status"] != 200 or "text/html" not in (
            receipt["content_type"] or "text/html"):
        return None, None, receipt
    try:
        body = PS.load_validated_artifact(preserve_root, receipt["artifacts"]["body"])
    except PS.PreservedArtifactError:
        return None, None, None
    html = _decode_html(body, receipt["content_type"])
    lang = None
    import re
    m = re.search(r'<html[^>]+lang\s*=\s*["\']([a-zA-Z]{2})', html or "")
    if m:
        lang = m.group(1).lower()
    return html, lang, receipt


# --------------------------- HN pull (Algolia) ----------------------------- #
def _algolia_slice(client, lo_i: int, hi_i: int, tags: str) -> dict:
    """Read one half-open time slice with the usual bounded retry contract."""
    params = {"tags": tags, "hitsPerPage": RESULT_CAP, "page": 0,
              "numericFilters": f"created_at_i>={lo_i},created_at_i<{hi_i}"}
    for attempt in range(5):
        try:
            r = client.get(ALGOLIA, params=params, headers=UA, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if not isinstance(data, dict):
                    raise HNSourceError("Algolia returned a non-object JSON payload")
                return data
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPError(f"HTTP {r.status_code}")
            raise HNSourceError(f"Algolia HTTP {r.status_code}: {r.text[:200]}")
        except HNSourceError:
            raise
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise HNSourceError(f"Algolia unreachable: {e!r} "
                                    "(NOT treating as empty — liveness lesson)") from e
            time.sleep(2 ** attempt)


def algolia_window(client, lo_i: int, hi_i: int, tags: str):
    """Enumerate every story in [lo_i, hi_i) without trusting pagination.

    Algolia's result window is capped at 1,000: a response for a denser range
    is *not* a complete source even when it says ``nbPages=1``.  Recursively
    partition by integer epoch until every leaf is below the cap.  The leaves
    are disjoint half-open ranges, so their union is precisely the requested
    interval.  Any unresolved cap fails closed rather than emitting a falsely
    complete receipt.
    """
    if hi_i <= lo_i:
        raise ValueError("Algolia window must be a non-empty half-open interval")

    leaves: list[dict] = []
    raw: list[dict] = []

    def visit(lo: int, hi: int) -> None:
        if len(leaves) >= MAX_TIME_PARTITIONS:
            raise HNSourceError(
                f"Algolia time partition limit {MAX_TIME_PARTITIONS} reached "
                f"before [{lo}, {hi}) could be enumerated")
        data = _algolia_slice(client, lo, hi, tags)
        nb_hits = data.get("nbHits")
        hits = data.get("hits")
        if not isinstance(nb_hits, int) or nb_hits < 0 or not isinstance(hits, list):
            raise HNSourceError("Algolia response lacks valid nbHits/hits fields")
        if nb_hits > RESULT_CAP:
            if hi - lo <= 1:
                raise HNSourceError(
                    f"Algolia result cap unresolved for one-second slice [{lo}, {hi})")
            mid = lo + (hi - lo) // 2
            visit(lo, mid)
            visit(mid, hi)
            return

        # With hitsPerPage == result cap, an uncapped leaf must be complete in
        # one response.  Do not silently revive the old pagination assumption.
        if len(hits) != nb_hits or data.get("nbPages", 1) != 1:
            raise HNSourceError(
                f"Algolia incomplete uncapped slice [{lo}, {hi}): "
                f"nbHits={nb_hits} returned={len(hits)} nbPages={data.get('nbPages')!r}")
        leaves.append({"start_epoch": lo, "end_epoch": hi,
                       "nb_hits": nb_hits, "returned_hits": len(hits)})
        raw.extend(hits)

    visit(lo_i, hi_i)
    # Defensive de-duplication by immutable HN ID makes an upstream overlap or
    # retry anomaly explicit in the receipt without changing source coverage.
    by_oid: dict[str, dict] = {}
    for hit in raw:
        oid = hit.get("objectID")
        if oid is None:
            raise HNSourceError("Algolia hit lacks immutable objectID")
        by_oid.setdefault(str(oid), hit)
    hits = sorted(by_oid.values(), key=lambda h: h.get("created_at_i") or 0,
                  reverse=True)
    return hits, {
        "algolia_result_cap": RESULT_CAP,
        "algolia_time_partition_count": len(leaves),
        "algolia_time_partitions": leaves,
        "algolia_raw_hits": len(raw),
        "algolia_unique_object_ids": len(hits),
    }


def _firebase_json(client, path: str):
    """Fetch an official Firebase endpoint with the same loud retry contract."""
    last = None
    for attempt in range(5):
        try:
            r = client.get(f"{FIREBASE}/{path}", headers=UA, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPError(f"HTTP {r.status_code}")
            raise HNSourceError(f"Firebase HTTP {r.status_code}: {r.text[:200]}")
        except HNSourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt == 4:
                raise HNSourceError(f"Firebase unreachable: {last!r} "
                                    "(NOT treating as empty — liveness lesson)") from exc
            time.sleep(2 ** attempt)


def firebase_window(client, lo_i: int, hi_i: int, tags: str,
                    item_cap: int = 1000) -> list[dict]:
    """Best-effort current-list fallback, explicitly not a historical census.

    Firebase exposes ranked current lists and individual items, rather than the
    Algolia time-window index.  It is therefore suitable only for clearly
    labelled safe additions during an Algolia outage; it must not erase or age
    out a source based on apparent absence.
    """
    if tags != "story":
        raise HNSourceError("Firebase fallback supports only tags=story")
    ids: list[int] = []
    seen: set[int] = set()
    for feed in FIREBASE_FEEDS:
        values = _firebase_json(client, f"{feed}.json")
        if not isinstance(values, list):
            raise HNSourceError(f"Firebase {feed} returned a non-list payload")
        for value in values:
            try:
                oid = int(value)
            except (TypeError, ValueError):
                continue
            if oid not in seen:
                seen.add(oid)
                ids.append(oid)
            if len(ids) >= item_cap:
                break
        if len(ids) >= item_cap:
            break
    out = []
    for oid in ids:
        item = _firebase_json(client, f"item/{oid}.json")
        if not isinstance(item, dict) or item.get("type") != "story":
            continue
        created = item.get("time")
        if not isinstance(created, int) or not lo_i <= created < hi_i:
            continue
        # Match Algolia's field names so the downstream row contract is one
        # implementation, irrespective of the official transport used.
        out.append({"objectID": str(oid), "url": item.get("url"),
                    "title": item.get("title"), "points": item.get("score"),
                    "created_at_i": created})
    return out


def qualifying_stories(raw: list[dict], min_points: int):
    seen: dict[str, dict] = {}
    for h in raw:
        url = (h.get("url") or "").split("#")[0]
        if not url:                      # Ask/Show-HN text posts: no primary URL
            continue
        pts = h.get("points") or 0
        admission = h.get("hn_admission")
        if pts < min_points and admission != "authoritative-primary":
            continue
        key = url.lower().rstrip("/")
        prior = seen.get(key)
        # HN can receive repeat submissions. Keep the strongest endorsement,
        # rather than whichever Algolia happened to return newest first.
        if prior is None or pts > (prior.get("points") or 0):
            seen[key] = h
    return sorted(seen.values(), key=lambda h: h.get("created_at_i") or 0,
                  reverse=True)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_authoritative_hn_admission(host_meta_path: Path | None) -> dict[str, dict]:
    """Load data-declared low-score admission rules, never host logic in code.

    Community score remains the broad HN spam control.  A host may additionally
    describe a URL owner segment and a hash-bound reviewed owner list, allowing
    primary-source URLs to enter below that threshold.  The only generic code
    primitive is ``path-owner``; host, path shape, owner set, and score floor
    remain policy data.  A malformed or hash-drifted policy aborts the source
    boundary rather than quietly broadening it.
    """
    if host_meta_path is None:
        return {}
    try:
        metadata = json.loads(host_meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HNSourceError(f"HN host metadata unavailable: {exc}") from exc
    if not isinstance(metadata, dict):
        raise HNSourceError("HN host metadata must be an object")
    out: dict[str, dict] = {}
    for host, raw in metadata.items():
        rule = raw.get("hn_low_score_admission") if isinstance(raw, dict) else None
        if rule is None:
            continue
        if not isinstance(rule, dict) or rule.get("kind") != "path-owner":
            raise HNSourceError(f"invalid HN authoritative admission rule for {host!r}")
        min_points, owner_segment = rule.get("min_points"), rule.get("owner_segment")
        owner_policy = rule.get("owner_policy")
        if (not isinstance(min_points, int) or min_points < 1 or
                not isinstance(owner_segment, int) or owner_segment < 0 or
                not isinstance(owner_policy, dict)):
            raise HNSourceError(f"invalid HN authoritative admission bounds for {host!r}")
        relative = owner_policy.get("path")
        expected = owner_policy.get("sha256")
        list_key = owner_policy.get("list_key")
        if not isinstance(relative, str) or not isinstance(expected, str) or not isinstance(list_key, str):
            raise HNSourceError(f"invalid HN owner policy pointer for {host!r}")
        policy_path = (host_meta_path.parent / relative).resolve()
        # The policy belongs to the same immutable release, not an arbitrary
        # filesystem target named in a JSON record.
        if not policy_path.is_relative_to(host_meta_path.parent.parent.resolve()):
            raise HNSourceError(f"HN owner policy escapes release tree for {host!r}")
        try:
            if _sha256_file(policy_path) != expected:
                raise HNSourceError(f"HN owner policy hash drift for {host!r}")
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HNSourceError(f"HN owner policy unreadable for {host!r}: {exc}") from exc
        owners = policy.get(list_key) if isinstance(policy, dict) else None
        if (not isinstance(owners, list) or not owners or
                any(not isinstance(owner, str) or not owner for owner in owners)):
            raise HNSourceError(f"HN owner policy list invalid for {host!r}")
        out[host.lower()] = {
            "min_points": min_points,
            "owner_segment": owner_segment,
            "owners": {owner.lower() for owner in owners},
            "policy_path": str(policy_path),
            "policy_sha256": expected,
            "list_key": list_key,
        }
    return out


def mark_authoritative_primary(raw: list[dict], rules: dict[str, dict]) -> None:
    """Attach an auditable admission reason to policy-scoped primary URLs."""
    for story in raw:
        url = (story.get("url") or "").split("#")[0]
        parsed = urlsplit(url)
        host = parsed.netloc.lower().split(":", 1)[0]
        rule = rules.get(host)
        if not rule or (story.get("points") or 0) < rule["min_points"]:
            continue
        segments = [segment for segment in parsed.path.split("/") if segment]
        index = rule["owner_segment"]
        if len(segments) > index and segments[index].lower() in rule["owners"]:
            story["hn_admission"] = "authoritative-primary"


def source_stories(client, since_days: int, min_points: int, now_i: int,
                   tags: str, source: str, firebase_item_cap: int,
                   host_meta_path: Path | None = None):
    """Return records plus the completeness receipt carried into each row."""
    lo_i = now_i - since_days * 86400
    if source == "algolia":
        raw, enumeration = algolia_window(client, lo_i, now_i, tags)
        rules = load_authoritative_hn_admission(host_meta_path)
        mark_authoritative_primary(raw, rules)
        selected = qualifying_stories(raw, min_points)
        authority = [story for story in selected
                     if story.get("hn_admission") == "authoritative-primary"
                     and (story.get("points") or 0) < min_points]
        rule_receipt = {
            host: {key: value for key, value in rule.items() if key != "owners"}
            for host, rule in rules.items()
        }
        return selected, {
            "transport": "algolia", "completeness": "time-window-complete",
            "hn_authoritative_low_score_rules": rule_receipt,
            "hn_authoritative_low_score_admitted": len(authority),
            **enumeration,
        }
    if source == "firebase":
        raw = firebase_window(client, lo_i, now_i, tags, firebase_item_cap)
        return qualifying_stories(raw, min_points), {
            "transport": "firebase", "completeness": "best-effort-current-lists",
            "feeds": list(FIREBASE_FEEDS), "item_cap": firebase_item_cap,
        }
    if source != "auto":
        raise ValueError(f"unknown HN source {source!r}")
    try:
        return source_stories(client, since_days, min_points, now_i, tags,
                              "algolia", firebase_item_cap, host_meta_path)
    except HNSourceError as algolia_error:
        print(f"WARN: {algolia_error}; using explicit Firebase best-effort fallback",
              file=sys.stderr, flush=True)
        return source_stories(client, since_days, min_points, now_i, tags,
                              "firebase", firebase_item_cap, host_meta_path)


# --------------------------- row build ------------------------------------- #
def build_row(h: dict, html, lang, now_ref: dt.date, source_receipt: dict,
              linked_fetch: dict | None = None) -> dict:
    url = (h.get("url") or "").split("#")[0]
    title = (h.get("title") or "").strip()
    oid = str(h.get("objectID"))
    created = h.get("created_at_i")
    hn_date = (dt.datetime.fromtimestamp(created, dt.timezone.utc).date().isoformat()
               if created else None)
    # `published` = the HN SUBMISSION date (created_at, from the API — free, and
    # what Tim asked for). For HN the *endorsement* date is the freshness key: a
    # classic paper resurfaced on the front page today should read as fresh, not
    # age out on its original date. The article's TRUE publish date (when we can
    # read it) is preserved separately as `article_published` metadata.
    published = hn_date
    article_published = extract_published(html, url=url, now_ref=now_ref) if html else None
    if html:
        import re
        # cheap text for embedding/snippet (no full mirror; ADR-039)
        txt = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html,
                     flags=re.I | re.S)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = " ".join(txt.split())[:1500]
        body = txt if len(txt) > 200 else title
    else:
        body = title
    text = f"{title}. {body}" if body != title else title
    dom = ""
    try:
        dom = reg_domain(urlsplit(url).netloc)
    except Exception:  # noqa: BLE001
        pass
    row = {
        "url": url,
        "hn_url": HN_ITEM.format(oid),
        "hn_score": h.get("points"),
        "hn_admission": h.get("hn_admission", "community-score"),
        "hn_strong_endorsement": (h.get("points") or 0) >= STRONG_ENDORSEMENT_POINTS,
        "hn_object_id": oid,
        "title": title,
        "snippet": extractive_snippet(text),
        "text": text,                       # embed input; stripped by embed_tier1
        "anchor": title,
        "lang": (lang or "en"),
        "domain": dom,
        "tier": "C",
        "trust_tier": CLASS_TRUST_C,
        "source_type": "reference",
        "disc_class": "news",               # -> 12-month window + delta refresh
        # Keep the long-standing Algolia provenance stable.  A fallback is
        # deliberately distinguishable in both provenance and payload rather
        # than silently looking like a complete window query.
        "provenance": ("hn-freshness" if source_receipt["transport"] == "algolia"
                       else "hn-freshness-firebase-best-effort"),
        "hn_source_receipt": source_receipt,
        # Which lane produced this row's content, so a downstream preservation
        # bridge never has to infer it from the text (task 445).
        "hn_content_mode": "linked_article_derived" if html else "title_only",
        "published": published,             # HN submission date (freshness key)
        "last_seen": now_ref.isoformat(),
        "search_ok": True,
    }
    if linked_fetch:      # sealed original bytes + transport + request identity
        row["hn_linked_fetch"] = linked_fetch
    if article_published:                   # article's true date, only when read
        row["article_published"] = article_published
    return row


def main(argv: list[str] | None = None, prog: str = "freshness_ingest") -> int:
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-points", type=int, default=DEFAULT_COMMUNITY_MIN_POINTS)
    ap.add_argument("--since-days", type=int, default=7)
    ap.add_argument("--tags", default="story",
                    help="Algolia tag scope: 'story' (all) or 'front_page'")
    ap.add_argument("--source", choices=("auto", "algolia", "firebase"), default="auto",
                    help="official HN transport; auto falls back to explicitly incomplete Firebase lists")
    ap.add_argument("--firebase-item-cap", type=int, default=1000,
                    help="bounded item reads for Firebase's explicitly incomplete fallback")
    ap.add_argument("--receipt", default=None,
                    help="write the immutable source/completeness receipt even for an empty window")
    ap.add_argument("--host-meta", type=Path, default=None,
                    help="host metadata containing hash-bound authoritative low-score rules")
    ap.add_argument("--now", default=None, help="reference today YYYY-MM-DD (test)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip the linked-article fetch (title-only + HN date)")
    ap.add_argument("--preserve-dir", type=Path, default=None,
                    help="root for the sealed linked-page originals + transport; "
                         "required whenever the linked article is fetched")
    args = ap.parse_args(argv)
    # Fail closed: fetching without a preservation root is exactly the defect
    # (bytes read, then dropped) that keeps every daily HN slice non-authorizing.
    if not args.no_fetch and args.preserve_dir is None:
        ap.error("--preserve-dir is required unless --no-fetch: a linked-page "
                 "fetch must be preserved as original bytes + transport")
    preserve_root = args.preserve_dir.resolve() if args.preserve_dir else None

    now_ref = (dt.date.fromisoformat(args.now) if args.now
               else dt.datetime.now(dt.timezone.utc).date())
    now_i = int(time.time())
    client = make_client(HttpSpec(timeout=5.0, follow_redirects=False))

    stories, source_receipt = source_stories(
        client, args.since_days, args.min_points, now_i, args.tags,
        args.source, args.firebase_item_cap, args.host_meta)
    object_ids = "\n".join(sorted(str(story.get("objectID")) for story in stories))
    source_receipt.update({
        "window_start_epoch": now_i - args.since_days * 86400,
        "window_end_epoch": now_i,
        "since_days": args.since_days,
        "tags": args.tags,
        "min_points": args.min_points,
        "qualifying_stories": len(stories),
        "qualifying_object_ids_sha256": hashlib.sha256(object_ids.encode()).hexdigest(),
    })
    if args.receipt:
        os.makedirs(os.path.dirname(args.receipt) or ".", exist_ok=True)
        with open(args.receipt, "w", encoding="utf-8") as receipt_file:
            json.dump(source_receipt, receipt_file, indent=2, sort_keys=True)
            receipt_file.write("\n")
    print(f"HN: {len(stories)} stories >= {args.min_points} pts in last "
          f"{args.since_days}d (tags={args.tags}, "
          f"source={source_receipt['transport']}, "
          f"completeness={source_receipt['completeness']})", flush=True)
    if args.limit:
        stories = stories[:args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = gated = fetched = titleonly = preserved = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for h in stories:
            url = (h.get("url") or "").split("#")[0]
            html = lang = linked_fetch = None
            if not args.no_fetch:
                if not consent_ok(url):
                    print(f"  [consent] DISALLOW {url[:80]} — skip", flush=True)
                    continue
                gated += 1
                html, lang, linked_fetch = fetch_article(
                    client, url, preserve_root, str(h.get("objectID")))
                if linked_fetch:
                    preserved += 1
                if html:
                    fetched += 1
                else:
                    titleonly += 1
                time.sleep(0.05)
            row = build_row(h, html, lang, now_ref, source_receipt, linked_fetch)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"DONE wrote {n} rows (gated {gated}, fetched {fetched}, "
          f"preserved {preserved}, title-only {titleonly}) -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
