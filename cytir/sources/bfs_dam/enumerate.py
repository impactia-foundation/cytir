#!/usr/bin/env python3
"""BFS DAM asset-API ingester (task #30 / #116) — Swiss statistical depth.

Enumerate the Swiss Federal Statistical Office (BFS/OFS/UST) Digital Asset
Management catalog and build ONE authoritative-locator passage per asset:
a descriptive title + the asset's own machine-readable summary + statistical
theme + period covered + agency, pointing at the stable human/agent page
``https://www.bfs.admin.ch/asset/{lang}/{damId}``.

Source (policy-a, Open-use license): the DAM asset API at
``https://dam-api.bfs.admin.ch/hub/api/dam/assets`` — a paginated JSON catalog
of ~185,931 assets (charts, data tables, publications, web content, press
releases), each already in a single language. This is the compliant deep-index
route for BFS: the human ``/bfs/...`` and ``/asset/...`` pages are an AEM/Vue
SPA that Common Crawl captures as an empty shell (task 094 root-cause); the DAM
API is the structured replacement carrying the same content the SPA renders.

Two phases, separated so the slow / rate-limited network fetch runs ONCE
(detached + resumable) and the fast, local parse/gate can be re-run freely:

    # Phase A — enumerate (long, detached, resumable):
    python3 -m cytir.sources.bfs_dam.enumerate enumerate --raw assets_raw.jsonl \
        --state enum_state.json [--sleep 1.5]

    # Phase B — parse + text-quality gate -> passages (fast, local):
    python3 -m cytir.sources.bfs_dam.enumerate parse --raw assets_raw.jsonl --out bfs_dam.jsonl
    # (or, on the source-family path: cyt source run bfs_dam_exact --opt root=<dir>)

Pagination: the API pages in blocks of 20 via ``?skip=N&pageSize=20`` — only
``skip`` works (``page``/``offset``/``from``/``pageNumber`` are silently
ignored; ``pageSize`` is capped at 20). It is slow (seconds/req) and
soft-throttles bursts (HTTP 000 / connection-refused after rapid requests), so
``enumerate`` uses ONE keep-alive connection, a polite inter-request delay, and
exponential backoff on 000/429/5xx. Each page is checkpointed to the state file
so a killed run resumes from ``next_skip``.

Text-quality gate (the PubMed / number-soup lesson of task 097): we do NOT embed
pure-data assets with no descriptive prose. An asset is KEPT only if it carries
a machine-readable summary (abstractShort / shortSummary) OR a descriptive
title (>= 4 distinct alphabetic words). Future-embargoed, prose-less, and
extraction-junk assets are dropped and counted per reason (quality over count).

Tier A / gov-stat. Open use (attribution: source = BFS/FSO).

Moved here from the flat pipeline driver (task 1487, S7); the old path forwards to this module.
It sits beside the builder (`cytir.extract.bfs_dam`, task 1101) and is the family's DISCOVERY entry:
`enumerate` writes the `assets_raw.jsonl` that `bfs_dam_exact` reads (`--opt root=`). It is a
network phase of ~186k pages, detached and resumable, so it is not the family's `discover`
(which reads the finished catalogue) nor an `l1()` (idempotent, small). `parse` is kept for the
old CLI; `cyt source run bfs_dam_exact` is the same builder through the Protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import httpx
from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

from cytir.extract.bfs_dam import (AGENCY, ASSET_URL, DOMAIN, MAX_TEXT, MIN_TITLE_WORDS,  # noqa: F401  # re-exported via the ingest_bfs_dam alias shim (no in-repo attribute reader found 2026-10-30; kept for the shim contract)
                                PERIOD_LABEL, THEME_LABEL, VALID_LANGS, build_passage,
                                is_future_embargo, last_seen_of, norm_lang, strip_html,
                                summary_text, theme_path)

API = "https://dam-api.bfs.admin.ch/hub/api/dam/assets"
PAGE_SIZE = 20          # API-enforced cap


# --------------------------------------------------------------------------- #
# Phase A — enumerate the catalog (polite, resumable)
# --------------------------------------------------------------------------- #
def get_page(client: httpx.Client, skip: int, *, tries: int = 12):
    """Polite GET of one skip-page returning parsed JSON.

    Retries connection errors / 429 / 5xx (the throttle classes) with
    exponential backoff; RAISES only after exhausting `tries` so a transient
    outage never silently looks like 'end of catalog'."""
    delay = 5.0
    last = None
    url = f"{API}?skip={skip}&pageSize={PAGE_SIZE}"
    for attempt in range(tries):
        try:
            r = client.get(url)
        except httpx.HTTPError as e:            # incl. connect-refused (HTTP 000)
            last = e
            print(f"  net-err skip={skip} ({attempt+1}/{tries}): {e!r} "
                  f"-> backoff {delay:.0f}s", file=sys.stderr, flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 180)
            continue
        sc = r.status_code
        if sc == 200:
            try:
                return r.json()
            except ValueError as e:
                last = e
                time.sleep(delay)
                delay = min(delay * 2, 180)
                continue
        if sc == 429 or 500 <= sc < 600:
            ra = r.headers.get("retry-after")
            wait = float(ra) if (ra and ra.isdigit()) else delay
            print(f"  {sc} skip={skip} ({attempt+1}/{tries}) -> wait {wait:.0f}s",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 180)
            last = RuntimeError(f"{sc} on {url}")
            continue
        raise RuntimeError(f"unexpected {sc} on {url}")
    raise RuntimeError(f"BFS DAM API kept failing skip={skip}: {last}")


def cmd_enumerate(args):
    raw_path = os.path.expanduser(args.raw)
    state_path = os.path.expanduser(args.state)
    state = {"next_skip": args.start_skip, "count": 0, "total": None}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        print(f"[resume] next_skip={state['next_skip']} "
              f"count={state['count']} total={state.get('total')}", flush=True)
    end_skip = args.end_skip if args.end_skip > 0 else None  # None = to catalog end

    client = make_client(HttpSpec(
        timeout=60, follow_redirects=True, user_agent_component="bfs-dam-ingest"))

    skip = state["next_skip"]
    count = state["count"]
    total = state.get("total")
    t0 = time.time()
    pages = 0
    # append mode: resume adds to the existing raw file; parse dedups on
    # (damId, lang) so a re-fetched boundary page is harmless.
    with open(raw_path, "a", encoding="utf-8") as out:
        while True:
            data = get_page(client, skip)
            total = data.get("total", total)
            rows = data.get("data") or []
            if not rows:
                print(f"empty page at skip={skip} — catalog end "
                      f"(total={total}, fetched={count})", flush=True)
                break
            for a in rows:
                out.write(json.dumps(a, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
            count += len(rows)
            skip += PAGE_SIZE
            pages += 1
            state = {"next_skip": skip, "count": count, "total": total,
                     "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            tmp = state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, state_path)
            if pages % 25 == 0:
                rate = count / max(time.time() - t0, 1)
                pct = 100 * count / total if total else 0
                print(f"  skip={skip} fetched={count}/{total} ({pct:.1f}%) "
                      f"{rate:.1f} assets/s", flush=True)
            if total is not None and skip >= total:
                print(f"reached total={total} (fetched={count})", flush=True)
                break
            if end_skip is not None and skip >= end_skip:
                print(f"reached end_skip={end_skip} (fetched={count})", flush=True)
                break
            if args.limit and count >= args.limit:
                print(f"hit --limit {args.limit}", flush=True)
                break
            time.sleep(args.sleep)
    print(f"ENUMERATE done: {count} assets -> {raw_path} "
          f"({(time.time()-t0)/60:.1f} min this run)", flush=True)


def cmd_parse(args):
    raw_path = os.path.expanduser(args.raw)
    out_path = os.path.expanduser(args.out)
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    reasons: Counter = Counter()
    amg_hist: Counter = Counter()
    coll_hist: Counter = Counter()
    lang_hist: Counter = Counter()
    theme_hist: Counter = Counter()
    n_raw = kept = 0
    seen_ids: set[str] = set()

    with open(raw_path, encoding="utf-8") as f, \
            open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                asset = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            n_raw += 1
            desc = asset.get("description") or {}
            bfs = asset.get("bfs") or {}
            amg_hist[((bfs.get("articleModelGroup") or {}).get("name"))] += 1
            lang_hist[desc.get("language")] += 1
            for c in (desc.get("categorization") or {}).get("collection") or []:
                if c.get("level") == 0:
                    coll_hist[c.get("code") or c.get("name")] += 1
            rec = build_passage(asset, now_iso, reasons)
            if rec is None:
                continue
            pid = rec["point_id"]
            if pid in seen_ids:          # in-file dup (idempotent uuid5)
                reasons["dup_pointid"] += 1
                continue
            seen_ids.add(pid)
            for t in theme_path(desc):
                theme_hist[t] += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"parsed {n_raw} raw assets -> {kept} passages ({out_path})")
    print(f"DROP reasons: {dict(reasons.most_common())}")
    print(f"lang(raw): {dict(lang_hist.most_common())}")
    print(f"articleModelGroup(raw): {dict(amg_hist.most_common())}")
    print(f"collection level0(raw): {dict(coll_hist.most_common(15))}")
    print(f"top themes(kept): {dict(theme_hist.most_common(15))}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enumerate", help="fetch the DAM catalog (resumable)")
    e.add_argument("--raw", required=True, help="append raw assets jsonl")
    e.add_argument("--state", required=True, help="resume checkpoint json")
    e.add_argument("--sleep", type=float, default=1.5,
                   help="polite inter-request delay (s)")
    e.add_argument("--start-skip", type=int, default=0,
                   help="shard start offset (used only when no state file yet)")
    e.add_argument("--end-skip", type=int, default=0,
                   help="shard end offset (0 = to catalog end)")
    e.add_argument("--limit", type=int, default=0, help="stop after N assets")
    e.set_defaults(func=cmd_enumerate)

    p = sub.add_parser("parse", help="parse+gate raw assets into passages")
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_parse)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
