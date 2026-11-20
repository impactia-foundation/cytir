#!/usr/bin/env python3
"""Consent GATE + entity-card fallback for the discovery pipeline (post-pass).

Tim's decision (2026-07-03): *"for the discovery third-party domains:
absolutely check consent; if no consent, create an entity card for the
(sub)domain"* — the card lets agents route to the site's own search instead
(specs/expansion-strategy-2026-07.md §4, specs/adr-source-authority-layer.md).

This is a POST-PASS over the discovery fetch's outputs — it never touches the
running fetch (task 075). Three subcommands:

  gate         passages.jsonl + state.json → resolve consent LIVE per HOST
               (cytir/consent/resolver.py, ADR-009 two-flag rubric) and
               split the passages:
                 gated/passages_ok.jsonl        → the embed/merge input
                                                  (rows stamped search_ok /
                                                  train_ok / consent_checked_at)
                 gated/passages_declined.jsonl  → EXCLUDED from merge (audit)
                 gated/passages_unknown.jsonl   → HELD (retryable; re-run gate)
                 gated/gate_summary.json        → stats + declined/unknown host
                                                  lists (card queue)
               Idempotent: re-running re-reads everything and rewrites the
               splits (consent results are cached 30 days; "unknown" always
               re-resolves). Because the fetch keeps running, the gate is
               re-run at MERGE TIME over the final passages.jsonl — the merge
               step consumes gated/passages_ok.jsonl, never the raw file:
                 python discovery_ingest.py embed \
                     --in ~/aw-data/discovery/gated/passages_ok.jsonl ...

               CORPUS STORE (native writer, 2026-07-04): pass
               --corpus-wave <name> and the gate ALSO writes the ok rows
               (consent-stamped — this is the natural corpus-store write
               point) as a parquet wave via the retired `corpus_store` module, into
               --corpus-dest (default the orchestrator box ~/aw-data/corpus-staging/).
               The corpus is PRIMARY; collections are derived (task 085).
               The gate runs on the orchestrator box, the canonical store lives on
               the build box — writing locally then shipping is the simple
               reliable path. Ship a finished wave with:
                 rsync -a ~/aw-data/corpus-staging/wave=<name> \
                   <host>:<corpus-root>/
               Flag-gated + additive: without --corpus-wave nothing changes.
               Re-runs rewrite the wave in place (force=True) — same
               idempotence as the jsonl splits.

  cards        for every DECLINED (sub)domain, build a source card
               (schema of build_source_cards.py; ADR rubric row "robots/
               consent deny → card"): disposition="consent-declined" (a
               card-family disposition), jurisdiction/class/trust from the
               discovery-queue metadata, endorsement = the queue's
               endorse_score, a `consent` block (flags+signals+checked_at),
               and a `navigate` block via the T3 fallback chain:
                 (1) OpenSearch: IF robots (cached body from the resolver)
                     allows our agent a single homepage GET → fetch the
                     homepage, look for <link rel="search" type=
                     "application/opensearchdescription+xml">, fetch the
                     descriptor, extract the Url template ({searchTerms}→{q});
                     also probe /.well-known/opensearch.xml. Skipped whenever
                     robots disallows the path (the probe stays
                     consent-compatible — robots-respecting, ≤3 GETs/host).
                 (2) else the conventional  https://host/?q={q}  (alternative
                     /search?q={q}) flagged "confidence": "guess".
               Grouping: if a registrable domain's declined hosts are only
               {apex, www.apex} → ONE card for the apex; otherwise one card
               per (sub)domain (Tim: "entity card for the (sub)domain").

  embed-cards  embed new cards into the existing `source_cards` collection
               (the 2,617-card prototype, config B) EXACTLY the way
               build_source_cards.cmd_embed built them — same card_text
               surface, same bm25 transform, same point ids
               (uuid5 of "card:<domain>", so re-runs and rubric demotions
               upsert in place) — but with the dense leg served by the TEI
               arctic container (the orchestrator box:8088) instead of a local GPU
               (documents embed with NO prefix; slice to MRL-128 + L2 renorm;
               parity per task 070's TEI parity note).

USAGE
  python consent_gate.py gate \
      --passages ~/aw-data/discovery/passages.jsonl \
      --state    ~/aw-data/discovery/state.json \
      --queue    ~/aw-data/discovery_queue.json \
      --out-dir  ~/aw-data/discovery/gated

  python consent_gate.py cards \
      --summary ~/aw-data/discovery/gated/gate_summary.json \
      --queue   ~/aw-data/discovery_queue.json \
      --out     ~/aw-data/discovery/gated/cards_declined.jsonl

  python consent_gate.py embed-cards \
      --in ~/aw-data/discovery/gated/cards_declined.jsonl \
      --tei http://localhost:8088 --qdrant-url http://localhost:16333 \
      --collection source_cards
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.robotparser
from collections import Counter, defaultdict

from cytir.consent import resolver as cr
from cytir.consent.policy import gate_bucket as _policy_gate_bucket
from cytir.consent.resolver import bucket, resolve_hosts

# card-schema helpers from the 2,617-card prototype (keep schemas identical)
from cytir.corpus.source_cards import (
    CLASS_FRESHNESS, CLASS_TRUST, KNOWN_NAV, card_text, country_for_domain, derive_topics,
    navigation_for)
from cytir.extract.html_extract import reg_domain


def _host_of(url: str) -> str:
    """The row's host, lowercase, WITHOUT the FQDN trailing dot. The resolver keys its verdicts and its
    cache files by the dot-free host (`resolve_hosts`, `_cache_path`), so a dotted key here looked
    the verdict up under a name it never holds and routed the row to `unknown` (task 824)."""
    return (urllib.parse.urlsplit(url).hostname or "").lower().rstrip(".")


def gate_bucket(rec: dict, *, search_indexing: bool = False,
                accept_search_grant: bool = False,
                unclear_ok: bool = False) -> str:
    """Classify one receipt for the requested downstream use.

    Collection embedding is retrieval indexing, not model training. The normal
    gate retains its historical both-flags behavior; callers that explicitly
    declare ``search_indexing`` follow the current aw_b1 projection contract:
    resolved ``search_ok=true`` is admitted and ``train_ok`` remains provenance
    that bars later training/export use.
    """
    return _policy_gate_bucket(
        rec, search_indexing=search_indexing,
        accept_search_grant=accept_search_grant, unclear_ok=unclear_ok,
        legacy_bucket=bucket)


# ============================================================================
# GATE
# ============================================================================

def cmd_gate(args):
    passages = os.path.expanduser(args.passages)
    out_dir = pathlib.Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # hosts = every host that actually appears in the fetched passages, plus
    # (informationally) the state file's done-domains. The gate is host-level:
    # robots/tdmrep are host-scoped (consent_resolver docstring).
    hosts: set[str] = set()
    rows_by_host = Counter()
    n_rows = 0
    if os.path.exists(passages):
        with open(passages) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                h = _host_of(r.get("url", ""))
                if h:
                    hosts.add(h)
                    rows_by_host[h] += 1
                n_rows += 1
    state = {}
    if args.state and os.path.exists(os.path.expanduser(args.state)):
        state = json.load(open(os.path.expanduser(args.state))).get("done", {})
    print(f"gate: {n_rows} passage rows across {len(hosts)} hosts "
          f"(state: {len(state)} domains done)", flush=True)
    if not hosts:
        print("nothing fetched yet — nothing to gate.")
        return

    res = resolve_hosts(hosts, cache_dir=os.path.expanduser(args.cache_dir),
                        concurrency=args.concurrency,
                        direct_crawl=getattr(args, "direct_crawl", False))

    accept_grant = getattr(args, "accept_search_grant", False)
    register_path = getattr(args, "unclear_register", "") or ""
    unclear_ok = bool(register_path)
    search_indexing = getattr(args, "search_indexing", False)
    verdict = {h: gate_bucket(
                   res[h], search_indexing=search_indexing,
                   accept_search_grant=accept_grant, unclear_ok=unclear_ok)
               for h in res}

    # --- ADR-024 amendment (2026-07-13): unclear-consent register ----------
    # every host admitted via the unclear class is appended (never silently)
    if unclear_ok:
        import yaml
        register_path = os.path.expanduser(register_path)
        reg = {"hosts": []}
        if os.path.exists(register_path):
            reg = yaml.safe_load(open(register_path)) or {"hosts": []}
        reg.setdefault("hosts", [])
        reg_hosts = {e.get("host") for e in reg["hosts"]}
        appended = []
        for h, b in sorted(verdict.items()):
            if b != "ok_search_only" or not cr.unclear_search_ok(res[h]):
                continue
            if h in reg_hosts:
                print(f"  [unclear-register] {h}: already registered — "
                      f"included for search (train_ok=False)", flush=True)
                continue
            rb = res[h].get("signals", {}).get("robots", {})
            entry = {
                "host": h,
                "signals_observed": "; ".join(res[h].get("reasons", [])),
                # cr.unclear_search_tokens is the ONE place this list is
                # built (engineering-doctrine 1) -- it folds in "ccbot" when
                # apply_robots_flags's dedicated ccbot_root branch (ADR-024
                # amendment 5) is what flipped train_ok, since that branch
                # never touches train_tokens_disallowed itself. Reading
                # train_tokens_disallowed directly here, as before, silently
                # dropped every CCBot-only host from amendment 2's register.
                "robots_train_tokens_blocked":
                    sorted(cr.unclear_search_tokens(res[h])),
                "wildcard_root_allowed": bool(rb.get("wildcard_root", True)),
                "claude_tokens_blocked": [],
                "tdm_or_noai": False,
                "observed_at": cr._now_iso()[:10],
                "rationale": ("vendor-specific training-token blocks only; "
                              "no wildcard/general AI block; no explicit "
                              "search signal; included for search per "
                              "ADR-024 amendment 2026-07-13"),
                "provenance": getattr(args, "corpus_wave", "") or "unknown",
                "added_by": "consent_gate.py --unclear-register (automatic)",
            }
            reg["hosts"].append(entry)
            reg_hosts.add(h)
            appended.append(h)
            print(f"  [unclear-register] APPEND {h}: "
                  f"{entry['signals_observed']}", flush=True)
        if appended:
            tmp = register_path + ".tmp"
            with open(tmp, "w") as f:
                f.write("# Consent-unclear register — ADR-024 amendment "
                        "(Tim, 2026-07-13).\n# Maintained by consent_gate.py; "
                        "see the repo copy for the full policy header.\n")
                yaml.safe_dump(reg, f, sort_keys=False, allow_unicode=True)
            os.replace(tmp, register_path)
            print(f"  [unclear-register] {len(appended)} host(s) appended -> "
                  f"{register_path}", flush=True)
    files = {b: open(out_dir / f"passages_{b}.jsonl", "w")
             for b in ("ok", "declined", "unknown")}
    split_rows = Counter()
    try:
        with open(passages) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                h = _host_of(r.get("url", ""))
                b = verdict.get(h, "unknown")
                if b in ("ok", "ok_search_only"):
                    rec = res[h]
                    # stamp the RESOLVED flags (ok_search_only rows carry
                    # train_ok=False into the payload/index — the explicit
                    # Content-Signal search=yes grant covers the retrieval
                    # embedding + snippet; training use stays excluded)
                    r["search_ok"] = bool(rec["search_ok"])
                    r["train_ok"] = bool(rec["train_ok"])
                    r["consent_checked_at"] = rec["checked_at"]
                split_rows[b] += 1
                if b == "ok_search_only":
                    b = "ok"                    # merges via passages_ok.jsonl
                files[b].write(json.dumps(r, ensure_ascii=False) + "\n")
    finally:
        for fh in files.values():
            fh.close()

    host_ct = Counter(verdict.values())
    declined = sorted(h for h, b in verdict.items() if b == "declined")
    unknown = sorted(h for h, b in verdict.items() if b == "unknown")
    search_only = sorted(h for h, b in verdict.items() if b == "ok_search_only")
    ok_rows_total = split_rows.get("ok", 0) + split_rows.get("ok_search_only", 0)
    summary = {
        "gated_at": cr._now_iso(),
        "passages_file": passages,
        "state_domains_done": len(state),
        "hosts": {"total": len(hosts),
                  **{k: host_ct.get(k, 0)
                     for k in ("ok", "ok_search_only", "declined", "unknown")}},
        "rows": {"total": n_rows,
                 **{k: split_rows.get(k, 0)
                    for k in ("ok", "ok_search_only", "declined", "unknown")}},
        "search_only_hosts": [
            {"host": h, "rows": rows_by_host[h],
             "search_ok": res[h]["search_ok"], "train_ok": res[h]["train_ok"],
             "reasons": res[h]["reasons"], "checked_at": res[h]["checked_at"],
             "note": (("merged for retrieval indexing under search_ok=true; "
                       "train_ok=False stamped — EXCLUDE from any training "
                       "use/export")
                      if search_indexing else
                      (("merged via ADR-024 amendment 2026-07-13 unclear-"
                       "consent register (vendor training-token blocks only); "
                       "train_ok=False stamped — EXCLUDE from any training "
                       "use/export")
                      if unclear_ok and cr.unclear_search_ok(res[h]) else
                      ("merged via EXPLICIT Content-Signal search=yes grant; "
                       "train_ok=False stamped — EXCLUDE from any training "
                       "use/export")))}
            for h in search_only],
        "declined_hosts": [
            {"host": h, "rows_excluded": rows_by_host[h],
             "search_ok": res[h]["search_ok"], "train_ok": res[h]["train_ok"],
             "reasons": res[h]["reasons"], "checked_at": res[h]["checked_at"]}
            for h in declined],
        "unknown_hosts": [
            {"host": h, "rows_held": rows_by_host[h],
             "reasons": res[h]["reasons"]} for h in unknown],
        "note": ("re-run this gate over the final passages.jsonl at merge time; "
                 "the merge/embed step consumes passages_ok.jsonl ONLY"),
    }
    with open(out_dir / "gate_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nGATE done -> {out_dir}")
    print(f"  hosts: ok={host_ct.get('ok',0)} "
          f"ok_search_only={host_ct.get('ok_search_only',0)} "
          f"declined={host_ct.get('declined',0)} "
          f"unknown={host_ct.get('unknown',0)}")
    print(f"  rows : ok={split_rows.get('ok',0)} "
          f"ok_search_only(train_ok=False)={split_rows.get('ok_search_only',0)} "
          f"declined(EXCLUDED)={split_rows.get('declined',0)} "
          f"unknown(HELD)={split_rows.get('unknown',0)}")
    if declined:
        print(f"  declined hosts queued for cards: {declined[:10]}"
              f"{' …' if len(declined) > 10 else ''}")

    # ---- corpus store (flag-gated, additive): ok rows -> parquet wave ------
    if getattr(args, "corpus_wave", ""):
        from cytir.corpus import store as corpus_store  # local: gate itself needs no pyarrow

        ok_path = out_dir / "passages_ok.jsonl"
        try:
            ever = "trafilatura-unknown"
            try:
                import trafilatura
                ever = f"trafilatura-{trafilatura.__version__}"
            except Exception:  # noqa: BLE001
                pass
            wmeta = corpus_store.write_wave(
                corpus_store._iter_jsonl(str(ok_path)),
                args.corpus_wave,
                dest=os.path.expanduser(args.corpus_dest),
                extraction_version=ever,
                force=True,
                meta={
                    "source": str(ok_path),
                    "source_sha256": corpus_store._sha256(str(ok_path)),
                    "source_lines": ok_rows_total,
                    "gated_from": passages,
                    "gate_hosts_ok": host_ct.get("ok", 0)
                    + host_ct.get("ok_search_only", 0),
                    "notes": ("written by consent_gate.py gate --corpus-wave; "
                              "consent-stamped ok rows only. Ship with: "
                              "rsync -a <dest>/wave=" + args.corpus_wave +
                              " <host>:<corpus-root>/"),
                },
            )
            match = "OK" if wmeta["rows"] == ok_rows_total else "MISMATCH"
            print(f"  corpus wave={args.corpus_wave}: {wmeta['rows']} rows "
                  f"-> {corpus_store.wave_dir(args.corpus_dest, args.corpus_wave)} "
                  f"(vs ok rows: {match})")
        except Exception as e:  # noqa: BLE001 — never lose the gate outputs
            print(f"  !! corpus-store write FAILED (gate outputs intact): {e}",
                  file=sys.stderr)


# ============================================================================
# CARDS — consent-declined (sub)domains get the entity-card fallback
# ============================================================================

_OS_LINK = re.compile(
    r'<link[^>]+type=["\']application/opensearchdescription\+xml["\'][^>]*>',
    re.I)
_HREF = re.compile(r'href=["\']([^"\']+)["\']', re.I)
_OS_TEMPLATE = re.compile(r'<Url[^>]+template=["\']([^"\']+)["\']', re.I)


def _robots_allows(host: str, path: str, cache_dir: str) -> bool:
    """Consent-compatible probe check: use the resolver's CACHED robots body;
    never re-fetch robots here. Unknown/missing cache → do not probe."""
    p = cr._cache_path(cache_dir, host)
    if not p.exists():
        return False
    try:
        rec = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return False
    body = rec.get("_robots_body", "")
    if not body:
        # no robots published (404/empty) → a single homepage GET is fine
        sig = rec.get("signals", {}).get("robots", {})
        return bool(sig.get("fetched")) and sig.get("http_status") not in (401, 403)
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(body.splitlines())
    try:
        return rp.can_fetch(cr.ROBOT_AGENT, path) and rp.can_fetch("*", path)
    except Exception:  # noqa: BLE001
        return False


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_META_DESC_RE = re.compile(
    r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']{10,400})["\']'
    r'|<meta[^>]+content=["\']([^"\']{10,400})["\'][^>]+name=["\']description["\']',
    re.I)


def _clean_frag(s: str) -> str:
    import html
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = html.unescape(s)                      # &Auml; → Ä (keep, don't strip)
    return re.sub(r"\s+", " ", s).strip()[:300]


def _probe_opensearch(client, host: str, cache_dir: str):
    """T3 tier: OpenSearch descriptor probe. Robots-respecting, ≤3 GETs, polite.
    Returns (navigation-dict | None, page_meta) — page_meta carries the
    homepage <title> + meta-description harvested from the SAME single
    consent-compatible GET (task 091: card topic/description specificity is
    the known bottleneck; the site's own words beat class boilerplate, and
    they arrive in the site's own language)."""
    candidates = []
    page_meta: dict = {}
    # (a) homepage <link rel=search> — one GET, only if robots allows "/"
    if _robots_allows(host, "/", cache_dir):
        r, _ = cr._get(client, f"https://{host}/")
        if r is not None and r.status_code == 200 and "html" in \
                r.headers.get("content-type", "html"):
            head = r.text[:200_000]
            tm = _TITLE_RE.search(head)
            if tm:
                page_meta["title"] = _clean_frag(tm.group(1))
            dm = _META_DESC_RE.search(head)
            if dm:
                page_meta["description"] = _clean_frag(dm.group(1) or dm.group(2))
            m = _OS_LINK.search(head)
            if m:
                hm = _HREF.search(m.group(0))
                if hm:
                    candidates.append(urllib.parse.urljoin(str(r.url), hm.group(1)))
        time.sleep(cr.HOST_DELAY)
    # (b) the conventional well-known location
    if _robots_allows(host, "/.well-known/opensearch.xml", cache_dir):
        candidates.append(f"https://{host}/.well-known/opensearch.xml")
    for c_url in candidates[:2]:
        # (descriptor may live on a CDN host — it's still just a descriptor GET)
        r, _ = cr._get(client, c_url)
        time.sleep(cr.HOST_DELAY)
        if r is not None and r.status_code == 200:
            t = _OS_TEMPLATE.search(r.text[:100_000])
            if t and "{searchTerms}" in t.group(1):
                tpl = t.group(1).replace("{searchTerms}", "{q}")
                return {
                    "search_url_template": tpl,
                    "query_format": "OpenSearch descriptor template",
                    "method": "GET", "opensearch": c_url,
                    "sitemap": None, "api": None,
                    "derivation": "opensearch-probe",
                    "confidence": "verified",
                }, page_meta
    return None, page_meta


def _guess_nav(host: str) -> dict:
    return {
        "search_url_template": f"https://{host}/?q={{q}}",
        "alternatives": [f"https://{host}/search?q={{q}}"],
        "query_format": "?q=<urlencoded terms> (conventional; verify per site)",
        "method": "GET", "opensearch": None, "sitemap": None, "api": None,
        "derivation": "pattern-default",
        "confidence": "guess",
    }


def _group_declined(declined_hosts: list[dict]) -> dict[str, dict]:
    """{card_domain: representative-host-entry}. One card per registrable
    domain when only apex/www declined; else one card per (sub)domain."""
    by_reg = defaultdict(list)
    for e in declined_hosts:
        by_reg[reg_domain(e["host"])].append(e)
    out = {}
    for reg, entries in by_reg.items():
        hosts = {e["host"] for e in entries}
        if hosts <= {reg, "www." + reg}:
            best = max(entries, key=lambda e: e.get("rows_excluded", 0))
            out[reg] = best
        else:
            for e in entries:
                out[e["host"].removeprefix("www.")] = e
    return out


def cmd_cards(args):
    summary = json.load(open(os.path.expanduser(args.summary)))
    queue = {q["domain"]: q for q in json.load(open(os.path.expanduser(args.queue)))}
    declined = summary.get("declined_hosts", [])
    if not declined:
        print("no declined hosts — no cards to build.")
        pathlib.Path(os.path.expanduser(args.out)).write_text("")
        return
    grouped = _group_declined(declined)
    client = cr._client()
    cache_dir = os.path.expanduser(args.cache_dir)
    cards = []
    for card_domain, entry in sorted(grouped.items()):
        host = entry["host"]
        q = queue.get(reg_domain(host), {})
        cls = q.get("class", "other")
        # T3 nav chain: curated table → OpenSearch probe → conventional guess.
        # The probe also harvests the homepage title/meta-description (same
        # single robots-respecting GET) for card specificity (task 091).
        nav, page_meta = None, {}
        if not args.no_probe:
            try:
                nav, page_meta = _probe_opensearch(client, host, cache_dir)
            except Exception:  # noqa: BLE001 — probe must never crash the batch
                nav, page_meta = None, {}
        if card_domain in KNOWN_NAV:
            nav = navigation_for(card_domain, cls)
            nav["confidence"] = "curated"
        nav = nav or _guess_nav(host)
        desc_bits = [b for b in (page_meta.get("title"),
                                 page_meta.get("description")) if b]
        desc = (" — ".join(desc_bits) + ". " if desc_bits else "") + \
            "Source declined indexing consent; use its own search via the " \
            "navigation template."
        card = {
            "domain": card_domain,
            "host": host,
            "disc_class": cls,
            "authoritative_for": derive_topics(card_domain, cls,
                                               q.get("by_classes")),
            "endorsement": {
                "source": "discovery-graph",
                "score": round(q.get("score", 0.0), 2),
                "endorse_score": round(q.get("endorse_score", 0.0), 2),
                "by_classes": q.get("by_classes", []),
                "why": q.get("why", ""),
            },
            "jurisdiction": country_for_domain(card_domain)
                            or ("CH" if q.get("swiss") else None),
            "license": "unverified (consent-declined; not mirrored)",
            "freshness_class": CLASS_FRESHNESS.get(cls, "medium"),
            "trust_tier": round(CLASS_TRUST.get(cls, 0.45), 2),
            "set": "discovery-consent-declined",
            "description": desc,
            "navigation": nav,
            # rubric row: fetchable=False (robots/consent deny) → card
            "disposition": "consent-declined",
            "signals": {"open_license": False, "fetchable": False,
                        "stable": True, "passage_dense": True},
            "why": ("consent-declined: " + "; ".join(entry.get("reasons", []))
                    )[:500],
            "consent": {
                "search_ok": entry.get("search_ok", False),
                "train_ok": entry.get("train_ok", False),
                "checked_at": entry.get("checked_at"),
                "rows_excluded_at_gate": entry.get("rows_excluded", 0),
            },
        }
        cards.append(card)
        print(f"  card {card_domain}: nav={nav['derivation']} "
              f"({nav.get('confidence')})", flush=True)
    client.close()
    out = pathlib.Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    nav_ct = Counter(c["navigation"]["derivation"] for c in cards)
    print(f"\nCARDS done: {len(cards)} consent-declined cards -> {out}")
    print(f"  navigation derivation: {dict(nav_ct)}")


# ============================================================================
# EMBED-CARDS — into the existing source_cards collection, via TEI (no GPU)
# ============================================================================

def _tei_embed(tei_url: str, texts: list[str]) -> list[list[float]]:
    """Document embeddings via the TEI arctic container: NO prefix, normalize,
    slice to 128 + L2 renorm (parity: task 070's TEI parity note)."""
    import httpx
    import numpy as np
    out = []
    with httpx.Client(timeout=120) as c:
        B = 16                       # TEI max_client_batch_size is 32
        for i in range(0, len(texts), B):
            r = c.post(tei_url.rstrip("/") + "/embed",
                       json={"inputs": texts[i:i + B],
                             "normalize": True, "truncate": True})
            r.raise_for_status()
            out.extend(r.json())
    a = np.asarray(out, dtype="float32")[:, :128]
    a /= np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    return a.tolist()


def cmd_embed_cards(args):
    from qdrant_client import models

    from cytir.core.bm25_tokenizer import bm25_tokens
    from cytir.core.ids import record_point_id as point_id
    from cytir.core.qdrant import qdrant_client

    cards = [json.loads(l) for l in open(os.path.expanduser(args.input))
             if l.strip()]
    if not cards:
        print("no cards to embed.")
        return
    client = qdrant_client(args.qdrant_url, timeout=120)
    assert client.collection_exists(args.collection), \
        f"collection {args.collection} not found — this step UPSERTS into the " \
        f"existing prototype collection, it never creates one"

    texts = [card_text(c) for c in cards]
    dense = _tei_embed(args.tei, texts)

    def bm25_vec(text):
        # identical transform to build_source_cards.cmd_embed (_bm25_vector):
        # Counter over the {int-hash: tf} dict; hash(int)&0x7FFFFFFF == int fold
        c = Counter(bm25_tokens(text))
        if not c:
            return None
        return models.SparseVector(indices=[hash(t) & 0x7FFFFFFF for t in c],
                                   values=[float(v) for v in c.values()])

    points = []
    for c, vec in zip(cards, dense):
        payload = {k: v for k, v in c.items() if k != "text"}
        vecs = {"dense": vec}
        sv = bm25_vec(card_text(c))
        if sv is not None:
            vecs["bm25"] = sv
        points.append(models.PointStruct(
            id=point_id({"url": "card:" + c["domain"]}),
            vector=vecs, payload=payload))
    client.upsert(collection_name=args.collection, points=points, wait=True)
    cnt = client.count(args.collection, exact=True).count
    print(f"EMBED-CARDS done: upserted {len(points)} cards "
          f"(ids stable → demotions update in place). "
          f"{args.collection} now has {cnt} cards.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate", help="consent-split fetched passages (post-pass)")
    g.add_argument("--passages", required=True)
    g.add_argument("--state", default="")
    g.add_argument("--queue", default="~/aw-data/discovery_queue.json")
    g.add_argument("--out-dir", default="~/aw-data/discovery/gated")
    g.add_argument("--cache-dir", default=cr.CACHE_DIR)
    g.add_argument("--concurrency", type=int, default=cr.CONCURRENCY)
    g.add_argument("--corpus-wave", default="",
                   help="also write the ok rows as this corpus-store parquet "
                        "wave (cytir.corpus.store); empty = off")
    g.add_argument("--corpus-dest", default="~/aw-data/corpus-staging",
                   help="corpus-store root for --corpus-wave (local staging; "
                        "rsync wave dirs to the canonical corpus store)")
    g.add_argument("--direct-crawl", action="store_true",
                   help="wave provenance is OUR OWN polite fetch (not Common "
                        "Crawl); cached under <host>@direct.json (task 187). "
                        "Since ADR-024 am.5 this records provenance only — a "
                        "CCBot-disallow stamps train_ok=false and leaves "
                        "search_ok alone in BOTH modes.")
    g.add_argument("--accept-search-grant", action="store_true",
                   help="hosts with an EXPLICIT Content-Signal search=yes + "
                        "ai-train=no merge as ok_search_only with "
                        "train_ok=False stamped (task 187 / ohchr.org)")
    g.add_argument("--search-indexing", action="store_true",
                   help="apply aw_b1 retrieval-indexing policy: require resolved "
                        "search_ok=true; retain train_ok only as a provenance "
                        "restriction for training/export")
    g.add_argument("--unclear-register", default="",
                   help="path to consent-unclear-register.yaml (ADR-024 "
                        "amendment 2026-07-13): hosts blocking ONLY non-Claude "
                        "vendor training tokens merge as ok_search_only with "
                        "train_ok=False stamped, and every qualifying host is "
                        "APPENDED to this register (never silently). Empty = "
                        "strict both-flags rule")
    g.set_defaults(func=cmd_gate)

    c = sub.add_parser("cards", help="build cards for consent-declined hosts")
    c.add_argument("--summary", required=True, help="gate_summary.json")
    c.add_argument("--queue", default="~/aw-data/discovery_queue.json")
    c.add_argument("--out", required=True)
    c.add_argument("--cache-dir", default=cr.CACHE_DIR)
    c.add_argument("--no-probe", action="store_true",
                   help="skip the OpenSearch probe (pattern-guess only)")
    c.set_defaults(func=cmd_cards)

    e = sub.add_parser("embed-cards", help="upsert cards into source_cards via TEI")
    e.add_argument("--in", dest="input", required=True)
    e.add_argument("--tei", default="http://localhost:8088")
    e.add_argument("--qdrant-url", default="http://localhost:16333",
                   help="source_cards qdrant (an ssh tunnel to it by default)")
    e.add_argument("--collection", default="source_cards")
    e.set_defaults(func=cmd_embed_cards)

    args = ap.parse_args()
    # task 864 (review 869 S1): from an aw_e10 T0 to the flip only the drain writes the twin
    from cytir.index import twin_write_hold as _TWH
    _TWH.refuse_cli_write(args, tool='cytir/consent/gate.py')
    args.func(args)


if __name__ == "__main__":
    main()
