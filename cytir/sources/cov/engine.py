#!/usr/bin/env python3
"""Task 351 — shared coverage-pack DOCUMENT ingester (PRs #47/#48 packs).

Generalizes the proven task-209 FINMA document-corpus pattern (enumerate a
regulator/IGO's complete document set via its machine route, fetch politely
with L0 retention, Docling PDFs to markdown, gate through the production
chunk@1 funnel) into ONE config-driven ingester: the bespoke-per-source part
is a ROUTES YAML entry (config is data, doctrine 2), not a copied script.

Route modes (per source):
  sitemap          sitemap.xml enumeration + include regex (HTML doc pages;
                   PDFs they link get harvested via pdf_include)
  listing          seed/hub/paginated listing pages scanned for doc links
                   (doc_re) — the admin.ch-family hub-page shape
  sitecore_widget  the finma.ch/swissmedic.ch Sitecore document widget:
                   POST <base><api_path> ds=<GUID>, ?Skip=N paginated
                   (generalizes ingest_finma_circulars.enumerate_source;
                   that script remains the 209 pack's frozen form)
  oai              OAI-PMH ListRecords harvest (resumptionToken loop),
                   oai_dc metadata; PDF link derivation per pdf_from
  urllist          explicit url list / generated url patterns (e.g. NORMLEX
                   instrument codes)

THREE STAGES (same venv split as task 209 — Docling needs numpy2 in
the operator venv; the fasttext LID gate 166 needs numpy<2 in the hygiene venv):

  enumerate  (.venv)    routes YAML -> inv/<source>.jsonl (url, kind, title,
                        description, date, lang). Listing/API/OAI responses
                        are L0-retained (kind=api).
  fetch      (.venv)    robots per host (fail-closed skip, ADR-073 posture:
                        a WAF/JS-shell robots is a DECLINE signal), per-host
                        politeness max(cfg delay, crawl-delay), L0 raw
                        retention (ADR-026, kind=pdf/html), Docling PDF ->
                        markdown (OCR off), HTML kept raw -> docs.jsonl.
                        PDFs linked from fetched HTML pages (pdf_include)
                        are harvested into the queue. Resumable (state.json).
  gate       (hygiene)  docs.jsonl -> the tracked production chunk profile:
                        PDFs via curated-head + aw_common's canonical chunker
                        dispatcher, HTML via aw_common.process_html
                        (gates 097/166/167);
                        apparatus filter + head-cap; per-source stamps
                        (provenance, trust, tier, license, theme, consent
                        basis, published) -> passages.jsonl for
                        stage_pack_b1.py. Candidate builds may override with
                        AW_CHUNKER; regular runs follow data/build_config.json.

Consent (POLICY §2 U1 / ADR-076): per-source `consent:` block in the routes
YAML. `basis: robots-live-resolve` sources are admitted by the live robots
verification this ingester performs at fetch time (fail-closed; checked_at =
fetch date). `basis: official-dump-license` sources (OAI/bulk routes with an
explicit open license) carry the dump grant per ADR-076 — the license, not a
robots probe, is the admission basis; robots is still honored when we touch
the host directly. Every passage carries its consent stamps; stage_pack_b1
(task-351 extension) passes them through per record.

  .venv/bin/python -m cytir.sources.cov.engine enumerate --routes data/cov351_routes.yaml \
      --pack regdocs --out-dir $COV351_WORK/regdocs
  .venv/bin/python -m cytir.sources.cov.engine fetch     --routes ... --pack regdocs \
      --out-dir ... [--only k1,k2] [--limit N]
  hygiene/bin/python -m cytir.sources.cov.engine gate     --routes ... --pack regdocs \
      --out-dir ...
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET

import yaml

from cytir.acquire import crawl as pack_fetch  # noqa: E402
from cytir.core.paths import operator_settings  # noqa: E402
from cytir.sources.family import HttpSpec  # noqa: E402
from cytir.sources.http import make_client  # noqa: E402
from cytir.chunk.generic import chunk_generic_records  # noqa: E402
import cytir.core.ids as ID  # noqa: E402
import cytir.core.common as AC  # noqa: E402
from cytir.acquire.crawl import (L0Store, fetch_robots, is_apparatus,  # noqa: E402
                        resolve_l0_root, sitemap_urls)

TODAY = dt.date.today().isoformat()

# href scan for pdf harvesting off fetched HTML pages (single/double/unquoted)
HREF_RE = pack_fetch.HREF_RE

canon = pack_fetch.make_canon()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# routes config
# --------------------------------------------------------------------------
def load_routes(path: str, pack: str) -> dict:
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    if pack not in cfg.get("packs", {}):
        raise SystemExit(f"pack {pack!r} not in {path} "
                         f"(have: {sorted(cfg.get('packs', {}))})")
    p = cfg["packs"][pack]
    p.setdefault("theme", [])
    for key, src in p.get("sources", {}).items():
        for req in ("host", "mode", "source_type", "trust", "license",
                    "consent"):
            if req not in src:
                raise SystemExit(f"source {key}: missing required key {req!r}")
        if src["consent"].get("basis") not in (
                "robots-live-resolve", "official-dump-license",
                "official-no-copyright"):
            raise SystemExit(f"source {key}: unknown consent basis "
                             f"{src['consent'].get('basis')!r}")
        if src["consent"]["basis"] != "robots-live-resolve" and \
                not src["consent"].get("source"):
            raise SystemExit(f"source {key}: dump/no-copyright consent needs "
                             f"an explicit consent.source string")
        src.setdefault("tier", "A")
        src.setdefault("delay", 2.0)
        src.setdefault("cap", 5000)
        src.setdefault("lang_default", "en")
        src.setdefault("theme", p["theme"])
        if src.get("dated_family_policy") not in (None, "latest_stable"):
            raise SystemExit(
                f"source {key}: unknown dated_family_policy "
                f"{src.get('dated_family_policy')!r}")
    return p


def fetch_origin_hosts(cfg: dict) -> list[str]:
    """Every origin a route may fetch, primary first and without duplicates.

    Cross-host document assets are a declared acquisition boundary. Keeping
    them in ``ok_hosts`` makes the preflight exhaustive instead of discovering
    a new origin only after an HTML page has already been fetched.
    """
    result: list[str] = []
    for host in [cfg["host"], *(cfg.get("ok_hosts") or [])]:
        host = str(host).lower().strip().strip(".")
        if not host or "://" in host or "/" in host:
            raise ValueError(f"invalid fetch origin host {host!r}")
        if host not in result:
            result.append(host)
    return result


def robots_allow_exact_url(robots: dict[str, object], url: str) -> bool:
    """Fail closed unless the exact URL passes its own origin's robots rules."""
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower().strip().strip(".")
    if parsed.scheme.lower() not in {"http", "https"}:
        return False
    rules = robots.get(host)
    if rules is None:
        return False
    target = parsed.path + (("?" + parsed.query) if parsed.query else "")
    return bool(rules.allowed(target))


def resolve_fetch_origin_robots(client, cfg: dict,
                                robots: dict[str, object], say) -> tuple[list[str], float]:
    """Resolve every declared fetch origin before a route becomes active."""
    origin_hosts = fetch_origin_hosts(cfg)
    for host in origin_hosts:
        if host in robots:
            continue
        rules = fetch_robots(client, host)
        robots[host] = rules
        if rules is None:
            say(f"robots UNAVAILABLE/JS-shell/403 on {host} "
                f"-> SOURCE SKIPPED (fail-closed)")
        else:
            say(f"robots ok {host} crawl-delay={rules.delay}")
        time.sleep(0.5)
    unavailable = [host for host in origin_hosts if robots[host] is None]
    delay = max([float(cfg["delay"])]
                + [robots[host].delay or 0.0
                   for host in origin_hosts if robots[host] is not None])
    return unavailable, delay


def _rx(pat: str | None) -> re.Pattern | None:
    return re.compile(pat) if pat else None


def _lang_for(url: str, cfg: dict) -> str:
    rx = cfg.get("lang_url_re") or r"/(de|fr|it|en)(?:/|$)"
    m = re.search(rx, urllib.parse.urlsplit(url).path)
    return m.group(1) if m else cfg["lang_default"]


_DATED_DOC_FAMILY = re.compile(
    r"^(?P<prefix>/(?:docs|specification)/)"
    r"(?P<version>draft|\d{4}-\d{2}-\d{2})"
    r"(?P<suffix>/.*)$")


def collapse_latest_dated_families(rows: list[dict]) -> list[dict]:
    """Collapse ISO-dated/draft documentation snapshots by logical path.

    Stable dates sort lexically in chronological order.  Draft is retained
    only when no stable edition of that logical page exists.  Unversioned
    pages and unrelated date-like path segments are untouched.
    """
    families: dict[tuple[str, str, str], set[str]] = {}
    parsed: list[tuple[dict, tuple[str, str, str] | None, str | None]] = []
    for row in rows:
        url = urllib.parse.urlsplit(row["url"])
        match = _DATED_DOC_FAMILY.match(url.path)
        if not match:
            parsed.append((row, None, None))
            continue
        family = (url.netloc, match.group("prefix"), match.group("suffix"))
        version = match.group("version")
        families.setdefault(family, set()).add(version)
        parsed.append((row, family, version))

    winners = {}
    for family, versions in families.items():
        stable = sorted(v for v in versions if v != "draft")
        winners[family] = stable[-1] if stable else "draft"
    return [row for row, family, version in parsed
            if family is None or winners[family] == version]


def url_in_route_scope(url: str, cfg: dict) -> bool:
    """Apply the route's reviewed page *and* harvested-PDF boundary.

    A sitemap/listing route may deliberately scope its navigable HTML pages
    with ``include`` while allowing linked document assets through the narrower
    ``pdf_include`` rule.  Resumed gates receive those already-fetched assets,
    so treating the page rule as the sole boundary silently drops a valid
    document class (eCH ``/sites/default/files/...pdf`` was the incident).
    ``pdf_include`` is an explicit reviewed allow-list, never a host-wide
    exception; ``exclude`` remains absolute.
    """
    inc, exc, pdf_inc = (_rx(cfg.get("include")), _rx(cfg.get("exclude")),
                         _rx(cfg.get("pdf_include")))
    if exc and exc.search(url):
        return False
    return bool(not inc or inc.search(url) or
                (pdf_inc and pdf_inc.search(url)))


def register_gate_record(seen: dict[str, str], rec: dict) -> bool:
    """Register one emitted record by the shared production point identity.

    The gate is the producer, so it must collapse byte-identical chunker
    duplicates before staging rather than relying on Qdrant's last-write
    behaviour.  A shared point ID with different emitted payload is a genuine
    identity collision and must stop the build for investigation.
    """
    digest = ID.content_digest(rec["text"])
    _tier, anchor_component = ID.classify_tier(
        rec.get("source_type"), rec.get("anchor"), True, digest,
        title=rec.get("title"))
    point_id = str(ID.mint_point_id(AC.canonical_url(rec["url"]),
                                    anchor_component))
    raw = json.dumps(rec, ensure_ascii=False, allow_nan=False,
                     sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(raw).hexdigest()
    previous = seen.get(point_id)
    if previous is None:
        seen[point_id] = fingerprint
        return True
    if previous == fingerprint:
        return False
    raise RuntimeError(
        f"FATAL: non-identical records share point identity {point_id}")


def gate_dated_family_winners(docs_path: str, pack: dict) -> dict[str, set[str]]:
    """Precompute URL-only winners without retaining fetched bodies in RAM."""
    candidates: dict[str, list[dict]] = {}
    for line in open(docs_path, encoding="utf-8"):
        if not line.strip():
            continue
        doc = json.loads(line)
        cfg = pack["sources"].get(doc.get("source"))
        if (cfg is None or
                cfg.get("dated_family_policy") != "latest_stable" or
                not url_in_route_scope(doc.get("url", ""), cfg)):
            continue
        candidates.setdefault(doc["source"], []).append({"url": doc["url"]})
    return {
        source: {row["url"] for row in collapse_latest_dated_families(rows)}
        for source, rows in candidates.items()
    }


# --------------------------------------------------------------------------
# stage: enumerate
# --------------------------------------------------------------------------
def enum_sitemap(client, cfg, l0, say) -> list[dict]:
    inc = _rx(cfg.get("include"))
    exc = _rx(cfg.get("exclude"))
    pdf_rx = re.compile(r"(?i)\.pdf(\?|$)")
    urls: list[str] = []
    for sm in cfg["sitemaps"]:
        urls += sitemap_urls(client, sm, delay=cfg["delay"])
        time.sleep(cfg["delay"])
    out, seen = [], set()
    for u in urls:
        cu = canon(u, keep_query=bool(cfg.get("keep_query")))
        if not cu or cu in seen:
            continue
        if urllib.parse.urlsplit(cu).netloc != cfg["host"]:
            continue
        if exc and exc.search(cu):
            continue
        if inc and not inc.search(cu):
            continue
        seen.add(cu)
        kind = "pdf" if pdf_rx.search(cu) else "html"
        out.append({"url": cu, "kind": kind, "lang": _lang_for(cu, cfg)})
    if cfg.get("dated_family_policy") == "latest_stable":
        before = len(out)
        out = collapse_latest_dated_families(out)
        say(f"  dated families: {before} -> {len(out)} latest stable")
    say(f"  sitemap: {len(urls)} locs -> {len(out)} in scope")
    return out


def enum_listing(client, cfg, l0, say) -> list[dict]:
    """Fetch listing/hub pages; harvest doc links via doc_re. Pagination via
    '{n}' template in a seed + n_start/n_max (stops early on an empty page)."""
    doc_re = re.compile(cfg["doc_re"])
    exc = _rx(cfg.get("exclude"))
    pdf_rx = re.compile(r"(?i)\.pdf(\?|$)")
    pages: list[str] = []
    for s in cfg.get("seeds", []):
        if "{n}" in s:
            for n in range(int(cfg.get("n_start", 0)),
                           int(cfg.get("n_max", 0)) + 1):
                pages.append(s.replace("{n}", str(n)))
        else:
            pages.append(s)
    out, seen = [], set()
    empty_streak = 0
    for pu in pages:
        try:
            r = client.get(pu)
        except Exception as e:  # noqa: BLE001
            say(f"  ! listing {pu}: {e}")
            continue
        if l0 is not None and r.status_code == 200:
            l0.put(pu, r.content, final_url=str(r.url), status=r.status_code,
                   content_type=r.headers.get("content-type"), kind="api")
        found = 0
        if r.status_code == 200:
            for m in doc_re.finditer(r.text):
                groups = m.groupdict()
                g = (groups.get("url") if "url" in groups else
                     m.group(m.lastindex or 0)).strip()
                g = g.replace("\\/", "/")     # JSON-embedded URLs (WP REST)
                u = urllib.parse.urljoin(str(r.url), g)
                cu = canon(u, keep_query=bool(cfg.get("keep_query")))
                if not cu or cu in seen or (exc and exc.search(cu)):
                    continue
                ok_hosts = set(cfg.get("ok_hosts", [cfg["host"]]))
                if urllib.parse.urlsplit(cu).netloc not in ok_hosts:
                    continue
                seen.add(cu)
                found += 1
                row = {"url": cu,
                       "kind": "pdf" if pdf_rx.search(cu) else "html",
                       "lang": _lang_for(cu, cfg)}
                # Some listing APIs expose an opaque transport URL alongside
                # authoritative record metadata.  Named regex groups make
                # that metadata part of the recurring inventory instead of
                # forcing the gate to invent a title from /content or UUID.
                for field in ("title", "description", "date", "lang"):
                    value = groups.get(field)
                    if value:
                        import html as _html
                        row[field] = _html.unescape(value).strip()
                out.append(row)
        empty_streak = empty_streak + 1 if not found else 0
        if "{n}" in str(cfg.get("seeds")) and empty_streak >= 2:
            break
        time.sleep(cfg["delay"])
    say(f"  listing: {len(pages)} pages -> {len(out)} docs")
    return out


def enum_sitecore_widget(client, cfg, l0, say) -> list[dict]:
    """finma/swissmedic Sitecore search widget (generalized task-209 route)."""
    base = cfg["base"]
    api = cfg["api_path"]
    host = cfg["host"]
    langword = re.compile(
        r"\s+(Deutsch|Französisch|Franz\.?|Italienisch|Englisch)\s*$", re.I)
    out, seen = [], set()
    for w in cfg["widgets"]:
        skip = 0
        n_before = len(out)
        while True:
            url = f"{base}{api}" + (f"?Skip={skip}" if skip else "")
            r = client.post(url, data={"ds": w["ds"]},
                            headers={"Content-Type":
                                     "application/x-www-form-urlencoded"})
            r.raise_for_status()
            if l0 is not None:
                l0.put(f"{api}?ds={w['ds']}&Skip={skip}", r.content,
                       status=r.status_code, content_type="application/json",
                       kind="api")
            j = r.json()
            items = j.get("Items") or []
            for it in items:
                ext = (it.get("Extension") or "").lower()
                # PDF items AND extension-less items (HTML detail pages —
                # finma Kasuistik/Gerichtsentscheide, recon p3b)
                if ext and ext != "pdf":
                    continue
                kind = "pdf" if ext == "pdf" else "html"
                title = (it.get("Title") or "").strip()
                desc = (it.get("Description") or "").strip()
                date = (it.get("Date") or "").strip()
                variants = []
                if w.get("multilang") and it.get("OtherLanguageLinks"):
                    for ol in it["OtherLanguageLinks"]:
                        variants.append(
                            (ol["Url"], (ol.get("Name") or "").lower()[:2]
                             or None))
                elif it.get("Link"):
                    variants.append((it["Link"], None))
                for u, lname in variants:
                    u = u if u.startswith("http") else \
                        urllib.parse.urljoin(base, u)
                    cu = canon(u, keep_query=True)
                    if not cu or cu in seen or \
                            urllib.parse.urlsplit(cu).netloc != host:
                        continue
                    seen.add(cu)
                    out.append({"url": cu, "kind": kind,
                                "title": langword.sub("", title).strip(),
                                "description": desc, "date": date,
                                "lang": lname or _lang_for(u, cfg),
                                "widget": w["key"]})
            count = j.get("Count") or 0
            skip += len(items)
            if not items or skip >= count:
                break
            time.sleep(cfg["delay"])
        say(f"  widget {w['key']}: {len(out) - n_before} pdf targets")
        time.sleep(cfg["delay"])
    return out


_OAI_NS = {"oai": "http://www.openarchives.org/OAI/2.0/",
           "dc": "http://purl.org/dc/elements/1.1/",
           "oaidc": "http://www.openarchives.org/OAI/2.0/oai_dc/"}


def _oai_get(client, endpoint, q, say, delay: float, attempts: int = 5):
    """One OAI request, with backoff on TRANSIENT transport failures.

    Incident (task 399, 2026-08-07): a bare `client.get` here killed a WHO-IRIS
    harvest at ~3,000 of ~195,000 records with
    `httpx.RemoteProtocolError: Server disconnected without sending a response`.
    Over a multi-hour ListRecords loop a transient disconnect is close to
    certain, and the endpoint was verified healthy immediately afterwards
    (Identify 200 in 1.0s, robots 200) — so this was never a block or a rate
    problem, just an unretried network blip.

    Returns the response, or None when the set should be abandoned. A 4xx/5xx
    is NOT retried here (the caller already handles status codes); only
    transport-level exceptions are.
    """
    for attempt in range(attempts):
        try:
            return client.get(endpoint, params=q)
        except Exception as e:  # noqa: BLE001 - transport-level only
            wait = delay * (2 ** attempt)
            say(f"  ! OAI transport error ({type(e).__name__}: {e}) — "
                f"retry {attempt + 1}/{attempts} in {wait:.0f}s")
            time.sleep(wait)
    say("  ! OAI transport error persisted — abandoning this set")
    return None


def enum_oai(client, cfg, l0, say) -> list[dict]:
    """OAI-PMH ListRecords harvest (oai_dc), resumptionToken loop, per set."""
    endpoint = cfg["endpoint"]
    prefix = cfg.get("metadata_prefix", "oai_dc")
    sets = cfg.get("sets") or [None]
    max_rec = int(cfg.get("max_records", 0))
    id_rx = _rx(cfg.get("identifier_re"))       # canonical-URL pick
    pdf_from = cfg.get("pdf_from", "identifier_pdf")
    out, seen = [], set()

    # CHECKPOINT (task 399): a ~195k-record harvest runs for hours, and the
    # pre-fix behaviour accumulated everything in memory and wrote the
    # inventory only on success — so one disconnect at record 3,000 threw away
    # the whole run and left an EMPTY inv/ directory. Records are now appended
    # as they arrive and the resumptionToken is checkpointed beside them, so a
    # re-run continues instead of restarting.
    ckpt_recs = cfg.get("_ckpt_path")
    ckpt_meta = (ckpt_recs + ".meta.json") if ckpt_recs else None
    resume_token, resume_set = None, None
    if ckpt_recs and os.path.exists(ckpt_recs) and os.path.exists(ckpt_meta):
        try:
            meta = json.load(open(ckpt_meta))
            resume_token, resume_set = meta.get("token"), meta.get("set")
            with open(ckpt_recs, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    if d["url"] not in seen:
                        seen.add(d["url"])
                        out.append(d)
            say(f"  resuming from checkpoint: {len(out)} records, "
                f"set={resume_set}, token={'yes' if resume_token else 'no'}")
        except Exception as e:  # noqa: BLE001
            say(f"  ! checkpoint unreadable ({e}) — restarting this source")
            out, seen, resume_token, resume_set = [], set(), None, None

    ck = open(ckpt_recs, "a", encoding="utf-8") if ckpt_recs else None

    def _save_token(st_, token_):
        if ck:
            ck.flush()
        if ckpt_meta:
            tmp = ckpt_meta + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"set": st_, "token": token_}, fh)
            os.replace(tmp, ckpt_meta)

    for st in sets:
        # when resuming, skip sets already drained before the checkpointed one
        if resume_set is not None and st != resume_set:
            continue
        token = resume_token
        resume_set = None          # only skip on the first loop entry
        pages = 0
        while True:
            if token:
                q = {"verb": "ListRecords", "resumptionToken": token}
            else:
                q = {"verb": "ListRecords", "metadataPrefix": prefix}
                if st:
                    q["set"] = st
            r = _oai_get(client, endpoint, q, say, float(cfg.get("delay", 2.0)))
            if r is None:
                break
            if r.status_code != 200:
                say(f"  ! OAI {st} HTTP {r.status_code} — stopping this set")
                break
            if l0 is not None:
                l0.put(str(r.url), r.content, status=r.status_code,
                       content_type="text/xml", kind="api")
            try:
                root = ET.fromstring(r.content)
            except ET.ParseError as e:
                say(f"  ! OAI parse error {e} — stopping this set")
                break
            err = root.find("oai:error", _OAI_NS)
            if err is not None:
                say(f"  ! OAI error {err.get('code')}: {(err.text or '')[:80]}")
                break
            recs = root.findall(".//oai:record", _OAI_NS)
            for rec in recs:
                hdr = rec.find("oai:header", _OAI_NS)
                if hdr is not None and hdr.get("status") == "deleted":
                    continue
                md = rec.find(".//oaidc:dc", _OAI_NS)
                if md is None:
                    continue

                def vals(tag):
                    return [e.text.strip() for e in
                            md.findall(f"dc:{tag}", _OAI_NS)
                            if e is not None and e.text and e.text.strip()]

                idents = vals("identifier")
                url = None
                for i in idents:
                    if i.startswith("http") and (not id_rx or id_rx.search(i)):
                        url = i
                        break
                if not url:
                    continue
                cu = canon(url, keep_query=True)
                if not cu or cu in seen:
                    continue
                seen.add(cu)
                pdf = None
                if pdf_from == "identifier_pdf":
                    pdf = next((i for i in idents
                                if i.startswith("http")
                                and i.lower().endswith(".pdf")), None)
                titles = vals("title")
                descs = vals("description")
                dates = vals("date")
                langs = vals("language")
                rights = vals("rights")
                lang = (langs[0][:2].lower() if langs
                        else cfg["lang_default"])
                # record_kind: meta = metadata-only minimal records (title+
                # abstract embedded, canonical URL surfaced, NOTHING fetched
                # from the host) — the WHO-IRIS minimal-record posture
                kind = cfg.get("record_kind") or ("pdf" if pdf else "html")
                row = {"url": cu, "kind": kind,
                            "fetch_url": pdf or cu,
                            "title": titles[0] if titles else None,
                            "description": (descs[0][:1200]
                                            if descs else None),
                            "date": dates[0] if dates else None,
                            "lang": lang, "set": st,
                            "subject": "; ".join(vals("subject")[:12]) or None,
                            "doc_type": "; ".join(vals("type")[:4]) or None,
                            "rights": "; ".join(rights)[:300] or None}
                out.append(row)
                if ck:
                    ck.write(json.dumps(row, ensure_ascii=False) + "\n")
                if max_rec and len(out) >= max_rec:
                    break
            tok_el = root.find(".//oai:resumptionToken", _OAI_NS)
            token = (tok_el.text or "").strip() if tok_el is not None else ""
            pages += 1
            if pages % 20 == 0:
                say(f"  OAI {st}: {len(out)} records so far")
                _save_token(st, token)
            if not token or (max_rec and len(out) >= max_rec):
                break
            time.sleep(cfg["delay"])
        _save_token(st, token if token else None)
        say(f"  OAI set {st}: cumulative {len(out)} records")
    if ck:
        ck.close()
    return out


def _expand_codes(spec: str) -> list[str]:
    """'C001-C190' -> C001..C190 ; plain codes pass through."""
    m = re.match(r"^([A-Z]+)(\d+)-\1(\d+)$", spec)
    if not m:
        return [spec]
    pre, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    width = len(m.group(2))
    return [f"{pre}{n:0{width}d}" for n in range(a, b + 1)]


def resolve_urls_file(path: str) -> str:
    """A relative `urls_file` in the routes file resolves under $COV351_WORK (default: `<work_root>/cov351`, the
    operator setting), so the shipped routes name no box path (task 825). Absolute paths pass."""
    if os.path.isabs(path):
        return path
    return os.path.join(os.environ.get("COV351_WORK") or os.path.join(operator_settings().work_root, "cov351"), path)


def enum_urllist(client, cfg, l0, say) -> list[dict]:
    out = []
    entries = list(cfg.get("urls", []))
    if cfg.get("urls_file"):
        with open(resolve_urls_file(cfg["urls_file"]), encoding="utf-8") as f:
            entries += [json.loads(l) for l in f if l.strip()]
    # code_patterns: [{template: ".../...{code}", codes: ["C001-C190", "P029"],
    #                  lang: en}] — a-priori enumerable instrument families
    # (NORMLEX conventions/recommendations/protocols, recon igo_routes)
    for pat in cfg.get("code_patterns", []):
        for spec in pat["codes"]:
            for code in _expand_codes(spec):
                entries.append({"url": pat["template"].format(code=code),
                                "lang": pat.get("lang"),
                                "title": pat.get("title_prefix", "").format(
                                    code=code) or None})
    for e in entries:
        if isinstance(e, str):
            e = {"url": e}
        cu = canon(e["url"], keep_query=bool(cfg.get("keep_query")))
        if not cu:
            continue
        out.append({"url": cu,
                    "kind": e.get("kind") or
                    ("pdf" if re.search(r"(?i)\.pdf(\?|$)", cu) else "html"),
                    "title": e.get("title"), "lang":
                    e.get("lang") or _lang_for(cu, cfg)})
    say(f"  urllist: {len(out)} docs")
    return out


ENUM = {"sitemap": enum_sitemap, "listing": enum_listing,
        "sitecore_widget": enum_sitecore_widget, "oai": enum_oai,
        "urllist": enum_urllist}



def _cov_client(timeout: float):
    """The consent-guarded package client (task 2420). Same as the old raw `httpx.Client`: the
    pack-fetch User-Agent (`pack_fetch.UA` is `pack_fetch_user_agent()`), redirects followed, the
    stage's timeout, no retries; the factory adds the request-layer decline check."""
    return make_client(HttpSpec(timeout=timeout, user_agent_kind="pack_fetch", follow_redirects=True))


def stage_enumerate(args):
    pack = load_routes(args.routes, args.pack)
    D = args.out_dir
    os.makedirs(os.path.join(D, "inv"), exist_ok=True)
    l0 = None if args.no_l0 else L0Store(
        resolve_l0_root(args.l0_root, D), f"cov351_{args.pack}")
    client = _cov_client(90)
    only = set(args.only.split(",")) if args.only else None

    def say(m):
        print(m, flush=True)

    robots_seen: dict[str, bool] = {}
    for key, cfg in pack["sources"].items():
        if only and key not in only:
            continue
        inv_path = os.path.join(D, "inv", f"{key}.jsonl")
        if os.path.exists(inv_path) and not args.re_enumerate:
            say(f"[{key}] inventory cached "
                f"({sum(1 for _ in open(inv_path))} rows)")
            continue
        # robots fail-closed BEFORE any enumeration request to the host
        # (ADR-073 posture; enumerate hits sitemaps/APIs/OAI on the host)
        host = cfg["host"]
        if host not in robots_seen:
            rb = fetch_robots(client, host)
            robots_seen[host] = rb is not None
            time.sleep(0.5)
        if not robots_seen[host]:
            say(f"[{key}] robots UNAVAILABLE/JS-shell/403 on {host} "
                f"-> SOURCE SKIPPED (fail-closed)")
            continue
        say(f"[{key}] enumerating via {cfg['mode']} …")
        # checkpoint sidecar (task 399): long OAI harvests resume instead of
        # restarting; ignored by every other enumerate mode.
        cfg["_ckpt_path"] = inv_path + ".ckpt.jsonl"
        docs = ENUM[cfg["mode"]](client, cfg, l0, say)
        cap = int(cfg["cap"])
        if len(docs) > cap:
            say(f"  cap {cap} < {len(docs)} — truncating inventory")
            docs = docs[:cap]
        tmp = inv_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for d in docs:
                d["source"] = key
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        os.replace(tmp, inv_path)
        # the inventory is now durable; drop the resume sidecars
        for stale in (cfg["_ckpt_path"], cfg["_ckpt_path"] + ".meta.json"):
            try:
                os.remove(stale)
            except OSError:
                pass
        say(f"[{key}] inventory: {len(docs)} docs -> {inv_path}")
    if l0 is not None:
        l0.close(complete=False)
    client.close()
    print("ENUMERATE DONE", flush=True)


# --------------------------------------------------------------------------
# stage: fetch
# --------------------------------------------------------------------------
def make_converter():
    from docling.document_converter import (DocumentConverter,
                                            PdfFormatOption)
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    po = PdfPipelineOptions()
    po.do_ocr = False
    po.do_table_structure = False
    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=po)})


def close_fetch_l0(l0, *, drained: bool) -> None:
    """Seal L0 exactly when the fetch frontier reached a terminal state.

    Task 416 found nine fully drained Task-351 waves still carrying only
    ``_meta.partial.json`` because this driver always passed ``False`` to
    ``L0Store.close``.  The fetch loop already computes the authoritative
    terminal condition; discarding it here forked runtime truth from the
    preservation receipt.
    """
    if l0 is not None:
        l0.close(complete=drained)


def pdf_to_markdown(conv, body: bytes, name: str) -> str:
    from docling.datamodel.base_models import DocumentStream
    return conv.convert(
        DocumentStream(name=name, stream=io.BytesIO(body))
    ).document.export_to_markdown()


_OPAQUE_PDF_TITLE = re.compile(
    r"(?:[0-9a-f]{8}[- ][0-9a-f]{4}[- ][0-9a-f]{4}[- ][0-9a-f]{4}[- ][0-9a-f]{12}|%[0-9a-f]{2})",
    re.IGNORECASE)
_OPAQUE_TRANSPORT_LEAF = re.compile(
    r"^(?:content|download|file|document|attachment|view|raw)$",
    re.IGNORECASE)


def preferred_pdf_title(supplied: str, markdown: str, url: str) -> str:
    """Prefer an extracted document heading when transport metadata is opaque.

    Backend file services commonly expose UUID filenames, while repository
    downloads retain URL escapes.  Those are neither a useful source title
    nor an honest dense-retrieval probe.  Explicit human titles remain
    authoritative; only opaque transport names yield to Docling's heading.
    """
    supplied = (supplied or "").strip()
    transport = supplied or re.sub(r"[-_]+", " ", os.path.splitext(
        os.path.basename(urllib.parse.urlsplit(url).path))[0]).strip()
    if (transport and not _OPAQUE_PDF_TITLE.search(transport) and
            not _OPAQUE_TRANSPORT_LEAF.fullmatch(transport)):
        return transport
    for line in markdown.splitlines():
        match = re.match(r"^#{1,6}\s+(.{3,200})$", line.strip())
        if match:
            candidate = match.group(1).strip()
            if not _OPAQUE_PDF_TITLE.search(candidate):
                return candidate
    return transport


def inventory_metadata(out_dir: str) -> dict[str, dict]:
    """Load small authoritative metadata fields from cached inventories.

    Fetch rows normally inherit these fields directly.  The join is also
    required for a resumable gate when route metadata extraction improves
    after the immutable body fetch completed: re-enumeration can then repair
    metadata without refetching or rewriting source bodies.
    """
    result: dict[str, dict] = {}
    inv_dir = os.path.join(out_dir, "inv")
    if not os.path.isdir(inv_dir):
        return result
    for path in sorted(glob.glob(os.path.join(inv_dir, "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            url = row.get("url")
            if not url:
                continue
            values = {key: row[key] for key in
                      ("title", "description", "date", "lang")
                      if row.get(key)}
            if values:
                result[url] = values
    return result


MAX_PDF_BYTES = 60 * 1024 * 1024      # sanity cap; larger is logged + skipped


def stage_fetch(args):
    pack = load_routes(args.routes, args.pack)
    D = args.out_dir
    docs_path = os.path.join(D, "docs.jsonl")
    state_path = os.path.join(D, "fetch_state.json")
    flog_path = os.path.join(D, "fetch_log.jsonl")

    state = json.load(open(state_path)) if os.path.exists(state_path) else {
        "done": [], "fetched": 0, "pdf_ok": 0, "html_ok": 0, "empty_md": 0,
        "http_err": 0, "docling_err": 0, "oversize": 0, "robots_skipped": {},
        "harvested_pdfs": 0, "per_source": {}, "started_at": _now()}
    done = set(state["done"])

    client = _cov_client(120)
    only = set(args.only.split(",")) if args.only else None
    df = open(docs_path, "a", encoding="utf-8")

    # queues per source from inventories (+ persisted harvested pdf queue)
    queues: dict[str, list[dict]] = {}
    qseen: dict[str, set] = {}
    active: dict[str, dict] = {}
    robots: dict[str, object] = {}
    for key, cfg in pack["sources"].items():
        if only and key not in only:
            continue
        inv_path = os.path.join(D, "inv", f"{key}.jsonl")
        if not os.path.exists(inv_path):
            print(f"[{key}] no inventory — run enumerate first; skipping",
                  flush=True)
            continue
        unavailable, effective_delay = resolve_fetch_origin_robots(
            client, cfg, robots,
            lambda message: print(f"[{key}] {message}", flush=True))
        if unavailable:
            state["robots_skipped"][key] = \
                state["robots_skipped"].get(key, 0) + 1
            continue
        cfg = dict(cfg, eff_delay=effective_delay)
        active[key] = cfg
        queues[key], qseen[key] = [], set()
        n_meta = 0
        for line in open(inv_path, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            u = d.get("fetch_url") or d["url"]
            if u in done or u in qseen[key]:
                continue
            qseen[key].add(u)
            if d.get("kind") == "meta":
                # metadata-only minimal record: nothing is fetched from the
                # host; the OAI metadata IS the record (IRIS posture)
                d.update({"fmt": "meta", "ok": True, "fetched_at": TODAY,
                          "source": key})
                df.write(json.dumps(d, ensure_ascii=False) + "\n")
                done.add(u)
                n_meta += 1
                continue
            queues[key].append(d)
        if n_meta:
            state["meta_rows"] = state.get("meta_rows", 0) + n_meta
            print(f"[{key}] {n_meta} metadata-only records -> docs.jsonl",
                  flush=True)
        for d in state.get("pdf_queue", {}).get(key, []):
            u = d.get("fetch_url") or d["url"]
            if u not in done and u not in qseen[key]:
                qseen[key].add(u)
                queues[key].append(d)
        print(f"[{key}] queue {len(queues[key])}", flush=True)

    l0 = None if args.no_l0 else L0Store(
        resolve_l0_root(args.l0_root, D), f"cov351_{args.pack}")
    if l0 is not None:
        print(f"[l0] -> {l0.dir} ({l0.records} carried over)", flush=True)

    conv = None
    flog = open(flog_path, "a", encoding="utf-8")
    next_ok: dict[str, float] = {k: 0.0 for k in active}
    new = 0
    t0 = time.time()
    last_out = [time.time()]

    def say(m):
        print(m, flush=True)
        last_out[0] = time.time()

    def heartbeat():
        if time.time() - last_out[0] > args.hb_secs:
            qs = {k: len(q) for k, q in queues.items() if q}
            say(f"  [hb] fetched={state['fetched']} pdf_ok={state['pdf_ok']} "
                f"html_ok={state['html_ok']} queues={qs}")

    def persist():
        state["done"] = list(done)
        state["pdf_queue"] = {
            k: [d for d in q if d.get("harvested")]
            for k, q in queues.items()}
        if l0 is not None:
            l0.persist()
        tmp = state_path + ".tmp"
        json.dump(state, open(tmp, "w"), ensure_ascii=False)
        os.replace(tmp, state_path)
        df.flush()

    def flog_row(**kw):
        kw["t"] = _now()
        flog.write(json.dumps(kw, ensure_ascii=False) + "\n")
        flog.flush()

    while any(queues.values()):
        if args.limit and new >= args.limit:
            say(f"--limit {args.limit} reached (resumable)")
            break
        if state["fetched"] >= args.budget:
            say("FETCH BUDGET reached — stopping")
            break
        now = time.time()
        ready = [k for k, q in queues.items() if q and next_ok[k] <= now]
        if not ready:
            heartbeat()
            nxt = min(next_ok[k] for k, q in queues.items() if q)
            time.sleep(max(0.05, min(nxt - now, 5.0)))
            continue
        key = min(ready, key=lambda k: next_ok[k])
        cfg = active[key]
        ps = state["per_source"].setdefault(key, {})
        if ps.get("fetched", 0) >= int(cfg["cap"]):
            say(f"[{key}] cap {cfg['cap']} reached — draining "
                f"({len(queues[key])} left)")
            queues[key] = []
            continue
        # PDF-first pop: harvested PDFs are the target corpus and must never
        # starve behind a long page tail when a source cap bites
        qi = next((i for i, x in enumerate(queues[key])
                   if x.get("kind") == "pdf"), 0)
        d = queues[key].pop(qi)
        u = d.get("fetch_url") or d["url"]
        if u in done:
            continue
        pq = urllib.parse.urlsplit(u)
        if not robots_allow_exact_url(robots, u):
            ps["robots_skipped"] = ps.get("robots_skipped", 0) + 1
            done.add(u)
            continue
        next_ok[key] = time.time() + cfg["eff_delay"]
        new += 1
        state["fetched"] += 1
        ps["fetched"] = ps.get("fetched", 0) + 1
        try:
            r = client.get(u)
            ctype = (r.headers.get("content-type") or "").lower()
            body = r.content
            is_pdf = ("pdf" in ctype) or body[:5] == b"%PDF-"
            flog_row(key=key, url=u, status=r.status_code,
                     bytes=len(body), ctype=ctype[:60],
                     final_url=(str(r.url) if str(r.url) != u else None))
            if r.status_code != 200:
                state["http_err"] += 1
                ps["http_err"] = ps.get("http_err", 0) + 1
                done.add(u)
            elif is_pdf:
                if len(body) > MAX_PDF_BYTES:
                    state["oversize"] += 1
                    done.add(u)
                else:
                    if l0 is not None:
                        l0.put(u, body, final_url=str(r.url),
                               status=r.status_code, content_type=ctype,
                               kind="pdf")
                    if conv is None:
                        say("[docling] loading converter…")
                        conv = make_converter()
                    name = os.path.basename(pq.path) or "doc.pdf"
                    try:
                        md = pdf_to_markdown(conv, body, name)
                    except Exception as e:  # noqa: BLE001
                        md = ""
                        state["docling_err"] += 1
                        say(f"  ! docling {u}: {str(e)[:100]}")
                    row = dict(d)
                    row.update({"fmt": "pdf", "ok": bool(md and len(md) >= 120),
                                "md": md if md and len(md) >= 120 else "",
                                "final_url": str(r.url), "fetched_at": TODAY,
                                "source": key})
                    if row["ok"]:
                        state["pdf_ok"] += 1
                        ps["pdf_ok"] = ps.get("pdf_ok", 0) + 1
                    else:
                        state["empty_md"] += 1
                        ps["empty_md"] = ps.get("empty_md", 0) + 1
                    df.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done.add(u)
            elif "html" in ctype:
                if l0 is not None:
                    l0.put(u, body, final_url=str(r.url),
                           status=r.status_code, content_type=ctype,
                           encoding=r.encoding, kind="html")
                row = dict(d)
                row.update({"fmt": "html", "ok": True, "html": r.text,
                            "final_url": str(r.url), "fetched_at": TODAY,
                            "source": key})
                df.write(json.dumps(row, ensure_ascii=False) + "\n")
                state["html_ok"] += 1
                ps["html_ok"] = ps.get("html_ok", 0) + 1
                done.add(u)
                # harvest linked PDFs (the admin.ch hub-page shape)
                prx = _rx(cfg.get("pdf_include"))
                if prx:
                    import html as _html
                    for m in HREF_RE.findall(r.text):
                        h = _html.unescape((m[0] or m[1] or m[2]).strip())
                        pu = urllib.parse.urljoin(str(r.url), h)
                        if not prx.search(pu):
                            continue
                        cu = canon(pu, keep_query=True)
                        if not cu or cu in done or cu in qseen[key]:
                            continue
                        qseen[key].add(cu)
                        state["harvested_pdfs"] += 1
                        queues[key].append(
                            {"url": cu, "kind": "pdf", "harvested": True,
                             "lang": d.get("lang"), "source": key})
            else:
                state["http_err"] += 1
                ps["http_err"] = ps.get("http_err", 0) + 1
                done.add(u)
        except Exception as e:  # noqa: BLE001
            ps["error"] = ps.get("error", 0) + 1
            flog_row(key=key, url=u, err=f"{type(e).__name__}: {e}")
            say(f"  ! {u}: {str(e)[:120]}")
            # transient -> NOT marked done; a resume retries it
        heartbeat()
        if new % 50 == 0:
            persist()
            rate = new / max(time.time() - t0, 1)
            qs = {k: len(q) for k, q in queues.items() if q}
            say(f"  {new} fetched  pdf_ok={state['pdf_ok']} "
                f"html_ok={state['html_ok']}  {rate*3600:.0f}/h  queues={qs}")

    drained = not any(queues.values())
    persist()
    df.close()
    flog.close()
    close_fetch_l0(l0, drained=drained)
    client.close()
    print("FETCH DONE" if drained else "FETCH PAUSED", flush=True)
    print(f"  {json.dumps({k: v for k, v in state.items() if k not in ('done', 'pdf_queue')}, ensure_ascii=False)}",
          flush=True)


# --------------------------------------------------------------------------
# stage: gate  (hygiene venv)
# --------------------------------------------------------------------------
def stage_gate(args):
    from cytir.core.common import (Passage, enable_outlink_capture, extraction_junk,
                           extractive_snippet, low_quality, process_html)
    from cytir.acquire import fetch_process as fp

    pack = load_routes(args.routes, args.pack)
    D = args.out_dir
    docs_path = os.path.join(D, "docs.jsonl")
    passages_path = os.path.join(D, "passages.jsonl")
    enable_outlink_capture(D)      # HTML pages feed the graph

    stats = {"docs": 0, "docs_with_passages": 0, "meta_fallback": 0,
             "dropped_no_meta": 0, "passages": 0, "duplicate_identity": 0,
             "capped": 0, "rejects": {},
             "apparatus": {}, "per_source": {}, "per_lang": {},
             "route_excluded": 0}

    def bump(d, k, n=1):
        d[k] = d.get(k, 0) + n

    def stamps(cfg, doc):
        base = {"provenance": doc["source"],
                "license": cfg["license"],
                "theme": list(cfg.get("theme") or []) or None,
                "published": doc.get("date") or None,
                "consent_basis": cfg["consent"]["basis"],
                "consent_state": cfg["consent"].get(
                    "state",
                    "live-ok" if cfg["consent"]["basis"] ==
                    "robots-live-resolve" else "dump"),
                "consent_source": cfg["consent"].get("source") or
                f"robots {cfg['host']} live-resolve (AuthoritativeWebBot)",
                "consent_checked_at": doc.get("fetched_at") or TODAY,
                "attribution": cfg.get("attribution"),
                "license_version": cfg.get("license_version"),
                # task 355 (ADR-082): explicit per-source train_ok override,
                # e.g. sources ADMITTED for search but not ML training.
                # Absent for a source -> unset here, same as before this
                # field existed (no behavior change for packs that don't
                # set it).
                "train_ok": cfg.get("train_ok")}
        # per-record rights (OAI) override the source default honestly:
        # an explicit CC statement becomes the record's license; anything
        # else is preserved verbatim in license_version (master-only)
        r = doc.get("rights")
        if r:
            if "creativecommons" in r.lower() or re.search(r"\bCC[ -]BY", r):
                base["license"] = r[:120]
            base["license_version"] = r[:200]
        return {k: v for k, v in base.items() if v is not None}

    def common(cfg, doc):
        dom = re.sub(r"^www\d*\.", "",
                     urllib.parse.urlsplit(doc["url"]).netloc.lower())
        return dict(domain=dom, tier=cfg.get("tier", "A"),
                    source_type=cfg["source_type"],
                    trust_tier=float(cfg["trust"]),
                    last_seen=doc.get("fetched_at") or TODAY)

    def pdf_passages(cfg, doc):
        md = doc.get("md") or ""
        title = preferred_pdf_title(doc.get("title") or "", md, doc["url"])
        if not title:
            m = re.match(r"^#\s+(.{2,200})$", md.lstrip().split("\n", 1)[0])
            if m:
                title = m.group(1).strip()
            else:
                title = re.sub(
                    r"[-_]+", " ",
                    os.path.splitext(os.path.basename(
                        urllib.parse.urlsplit(doc["url"]).path))[0]).strip()
        head = f"# {title}\n\n"
        if doc.get("description"):
            head += f"{doc['description']}\n\n"
        full = head + md
        if len(full) < 120 or low_quality(full) or extraction_junk(full):
            return [], title
        lang, conf = fp.detect_lang(md or full, doc["url"],
                                    doc.get("lang") or cfg["lang_default"])
        if conf and lang not in fp.LANG_SCOPE:
            bump(stats["rejects"], "lang_out_of_scope")
            return [], title
        out = []
        units = chunk_generic_records(full)
        for unit in units:
            ch = unit.text
            if low_quality(ch):
                continue
            if is_apparatus(ch):
                bump(stats["apparatus"], "bibliography")
                continue
            rr = extraction_junk(ch)
            if rr:
                bump(stats["rejects"], rr)
                continue
            out.append(AC.passage_from_chunk(
                unit, url=doc["url"], title=title or doc["url"],
                snippet=extractive_snippet(ch), lang=lang,
                **common(cfg, doc)))
        if len(out) > args.max_chunks:
            stats["capped"] += len(out) - args.max_chunks
            out = out[:args.max_chunks]
        return out, title

    def html_passages(cfg, doc):
        meta = {"content": True, "links": False, "want_stats": True,
                "lang_default": doc.get("lang") or cfg["lang_default"],
                "passage_common": common(cfg, doc)}
        passages, _l, st = process_html(doc["url"], doc.get("html") or "",
                                        meta)
        for rr, n in (st.get("extraction_rejects") or {}).items():
            bump(stats["rejects"], rr, n)
        kept = []
        for p in passages:
            if is_apparatus(p.text):
                bump(stats["apparatus"], "bibliography")
                continue
            kept.append(p)
        if len(kept) > args.max_chunks:
            stats["capped"] += len(kept) - args.max_chunks
            kept = kept[:args.max_chunks]
        return kept

    pf = open(passages_path, "w", encoding="utf-8")
    dated_winners = gate_dated_family_winners(docs_path, pack)
    inv_metadata = inventory_metadata(D)
    seen_urls = set()
    seen_point_ids: dict[str, str] = {}
    for line in open(docs_path, encoding="utf-8"):
        if not line.strip():
            continue
        doc = json.loads(line)
        for key, value in inv_metadata.get(doc.get("url"), {}).items():
            if not doc.get(key):
                doc[key] = value
        cfg = pack["sources"].get(doc["source"])
        if cfg is None:
            stats["route_excluded"] += 1
            continue
        u = doc["url"]
        if not url_in_route_scope(u, cfg):
            stats["route_excluded"] += 1
            continue
        if (cfg.get("dated_family_policy") == "latest_stable" and
                u not in dated_winners.get(doc["source"], set())):
            stats["route_excluded"] += 1
            continue
        if u in seen_urls:          # re-fetched rows (resume): last-write-wins
            continue                # is wrong here — first row wins, dups skip
        seen_urls.add(u)
        stats["docs"] += 1
        psrc = stats["per_source"].setdefault(doc["source"], {})
        title = doc.get("title")
        if doc.get("fmt") == "meta":
            # metadata-only minimal record (IRIS posture): title+abstract IS
            # the record; nothing was fetched from the host
            pl = []
            if title:
                text = title + (". " + doc["description"]
                                if doc.get("description") else "")
                if not low_quality(text):
                    lang = (doc.get("lang") or cfg["lang_default"])
                    if lang not in fp.LANG_SCOPE:
                        bump(stats["rejects"], "lang_out_of_scope")
                    else:
                        pl = [Passage(url=u, title=title,
                                      snippet=extractive_snippet(text),
                                      text=text, lang=lang,
                                      **common(cfg, doc))]
            if pl:
                stats["docs_with_passages"] += 1
                bump(psrc, "docs_ok")
            else:
                stats["dropped_no_meta"] += 1
                bump(psrc, "dropped")
                continue
        elif doc.get("fmt") == "pdf":
            pl, title = pdf_passages(cfg, doc) if doc.get("ok") \
                else ([], doc.get("title"))
        else:
            pl = html_passages(cfg, doc) if doc.get("ok") else []
        if doc.get("fmt") == "meta":
            pass                        # counted above
        elif pl:
            stats["docs_with_passages"] += 1
            bump(psrc, "docs_ok")
        elif doc.get("fmt") == "pdf" and (doc.get("title") or
                                          doc.get("description")):
            # 209 rule: curated metadata fallback so the authoritative URL
            # still surfaces; never for junk (no metadata -> drop + count)
            text = (doc.get("title") or title or "") + \
                (". " + doc["description"] if doc.get("description") else "")
            pl = [Passage(url=u, title=doc.get("title") or title or u,
                          snippet=extractive_snippet(text), text=text,
                          lang=doc.get("lang") or cfg["lang_default"],
                          **common(cfg, doc))]
            stats["meta_fallback"] += 1
            bump(psrc, "meta_fallback")
        else:
            stats["dropped_no_meta"] += 1
            bump(psrc, "dropped")
            continue
        sts = stamps(cfg, doc)
        for p in pl:
            rec = AC.passage_corpus_record(p)
            rec.pop("content_digest", None)
            rec.pop("point_id", None)
            rec.update(sts)
            if not register_gate_record(seen_point_ids, rec):
                # Identical chunker repeats are producer-side noise.  They
                # must not reach Qdrant, where upsert order would otherwise
                # hide the defect and make staged row counts ambiguous.
                stats["duplicate_identity"] += 1
                continue
            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            bump(stats["per_lang"], p.lang)
            stats["passages"] += 1
            bump(psrc, "passages")
    pf.close()
    json.dump(stats, open(os.path.join(D, "gate_stats.json"), "w"), indent=2)
    print("GATE DONE", flush=True)
    print(json.dumps({k: v for k, v in stats.items()
                      if k != "per_lang"}, ensure_ascii=False), flush=True)
    print(f"  per_lang={json.dumps(stats['per_lang'])}", flush=True)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def std(p):
        p.add_argument("--routes", required=True)
        p.add_argument("--pack", required=True)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--only", default="")
        p.add_argument("--l0-root", default="")
        p.add_argument("--no-l0", action="store_true")
        return p

    e = std(sub.add_parser("enumerate"))
    e.add_argument("--re-enumerate", action="store_true")
    e.set_defaults(func=stage_enumerate)

    f = std(sub.add_parser("fetch"))
    f.add_argument("--limit", type=int, default=0)
    f.add_argument("--budget", type=int, default=60000)
    f.add_argument("--hb-secs", type=float, default=240.0)
    f.set_defaults(func=stage_fetch)

    g = sub.add_parser("gate")
    g.add_argument("--routes", required=True)
    g.add_argument("--pack", required=True)
    g.add_argument("--out-dir", required=True)
    g.add_argument("--max-chunks", type=int, default=50)
    g.set_defaults(func=stage_gate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
