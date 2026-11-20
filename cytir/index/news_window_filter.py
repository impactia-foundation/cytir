#!/usr/bin/env python3
# Moved from the pipeline-era `news_window_filter` module (task 677, task 643 S5); the old path was retired.
"""news_window_filter.py — the rolling news-freshness window (Tim's news-freshness
policy, 2026-07-21) and the retention/purge horizon (Tim's ruling, 2026-08-11).

This module answers one question — "given a cutoff `months` back from `now_ref`,
is this article's `published` date in or out?" — and is deliberately agnostic
about WHICH cutoff a caller means. Two callers use it for two different
purposes, with two different constants (never conflate them — one constant,
one meaning, one home):

  * AW_NEWS_FRESH_MONTHS (default 12)  — freshness/discovery semantics: how far
    back `news_refresh_stage.sh` looks when discovering newly-published articles
    (`sitemap_ingest.py --window-months`), and the ranking-freshness tier. This
    is Tim's original 2026-07-21 policy and is UNCHANGED by the 2026-08-11 ruling.
  * AW_NEWS_RETAIN_MONTHS (default 36) — the purge/aged-out horizon used by
    `news_refresh_delta.sh` (via this module's `--months`) to compute the
    delete-id list from a dump of the twin's news points. Tim's 2026-08-11
    ruling, verbatim: "we're currently aiming for last 12 months which is good
    but we probably shouldn't purge before 24-36 months. the logic being that
    models are released 6-12 months old and may stay in service 1-2 years."
    Retention floor is 24 months; implemented at 36 (the top of Tim's range).
    An article can be well outside the 12-month FRESH tier (so it stops
    benefiting from the freshness ranking signal) while still being retained
    and searchable for up to 36 months — it is only deleted after that.

Two independent knobs, both passed in EXPLICITLY so the decision is deterministic
and testable (no hidden Date.now in library code — the CLI/caller supplies now):
  * `now_ref`  — the reference "today" (datetime.date).
  * `months`   — window length (default 12; callers pass their own constant).
The cutoff is `now_ref` minus `months` CALENDAR months (month-accurate, no
365-day approximation), so a run on the 21st keeps everything published on/after
the same day-of-month N months earlier.

NEWS ONLY. A record is subject to the window iff `is_news(payload)` — non-news
source_types (law, gov, docs, reference-that-isn't-journalism) are never dropped
by age. News is identified by `disc_class == "news"` (how the swissinfo/ch-media
ingests tag it) OR an explicit news domain/source_type override the caller adds.

  from cytir.index.news_window_filter import is_news, in_window, window_cutoff
  cutoff = window_cutoff(now_ref, months=12)          # a datetime.date
  keep = in_window(payload.get("published"), cutoff)  # True/False/None(unknown)
"""
from __future__ import annotations

import datetime as _dt
from typing import Optional

# how the news ingests tag journalism rows; extend via `extra_news_domains`
NEWS_DISC_CLASSES = {"news"}
# source_types that are inherently journalism (kept minimal; disc_class is primary)
NEWS_SOURCE_TYPES: set = set()


def window_cutoff(now_ref: _dt.date, months: int = 12) -> _dt.date:
    """The oldest published date still IN window = now_ref minus `months`
    calendar months. Articles published strictly before this are aged out."""
    if isinstance(now_ref, _dt.datetime):
        now_ref = now_ref.date()
    y = now_ref.year
    m = now_ref.month - months
    while m <= 0:
        m += 12
        y -= 1
    # clamp day for short months (e.g. cutoff landing on Feb 30 -> Feb 28/29)
    d = now_ref.day
    while True:
        try:
            return _dt.date(y, m, d)
        except ValueError:
            d -= 1


def is_news(payload: dict,
            extra_news_domains: Optional[set] = None) -> bool:
    """Is this payload a news/journalism record subject to the rolling window?"""
    if not payload:
        return False
    if payload.get("disc_class") in NEWS_DISC_CLASSES:
        return True
    if payload.get("source_type") in NEWS_SOURCE_TYPES:
        return True
    if extra_news_domains and payload.get("domain") in extra_news_domains:
        return True
    return False


def in_window(published_iso: Optional[str], cutoff: _dt.date) -> Optional[bool]:
    """True  -> published on/after cutoff (keep)
       False -> published before cutoff (aged out -> delete)
       None  -> no/unparseable date (caller decides: keep-but-flag, don't delete
                on an unknown date — deleting on missing data would be lossy)."""
    if not published_iso:
        return None
    try:
        y, m, d = (int(x) for x in str(published_iso)[:10].split("-"))
        pub = _dt.date(y, m, d)
    except (ValueError, TypeError):
        return None
    return pub >= cutoff


# --------------------------------------------------------------------------- #
# CLI: partition a jsonl(.gz) of {payload:{...}} (or a dump_netnew dump) into
# in-window / aged-out / unknown, and emit the aged-out point-id delete list.
# --------------------------------------------------------------------------- #
def _partition(records, cutoff, extra_domains):
    keep, aged, unknown, nonnews = [], [], [], 0
    for r in records:
        payload = r.get("payload", r)  # accept dump lines OR bare payloads
        if not is_news(payload, extra_domains):
            nonnews += 1
            keep.append(r)
            continue
        verdict = in_window(payload.get("published"), cutoff)
        if verdict is True:
            keep.append(r)
        elif verdict is False:
            aged.append(r)
        else:
            unknown.append(r)
    return keep, aged, unknown, nonnews


def main() -> int:
    import argparse
    import gzip
    import json
    import sys

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True,
                    help="jsonl(.gz): dump_netnew lines or bare payload objects")
    ap.add_argument("--now", default=None,
                    help="reference 'today' YYYY-MM-DD (default: real UTC today)")
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--delete-ids-out", default=None,
                    help="write aged-out point ids (one per line) for `python -m cytir.index.apply_delta` --delete-ids")
    ap.add_argument("--keep-out", default=None,
                    help="write the in-window records back out (jsonl.gz)")
    ap.add_argument("--news-domains", default="",
                    help="comma-separated extra domains to treat as news")
    args = ap.parse_args()

    now_ref = (_dt.date.fromisoformat(args.now) if args.now
               else _dt.datetime.utcnow().date())
    cutoff = window_cutoff(now_ref, args.months)
    extra = {d.strip() for d in args.news_domains.split(",") if d.strip()} or None

    opn = gzip.open if args.inp.endswith(".gz") else open
    with opn(args.inp, "rt", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    keep, aged, unknown, nonnews = _partition(records, cutoff, extra)
    print(f"now={now_ref} months={args.months} cutoff={cutoff}", file=sys.stderr)
    print(f"total={len(records)} news_in_window={len(keep) - nonnews} "
          f"aged_out={len(aged)} unknown_date={len(unknown)} non_news={nonnews}",
          file=sys.stderr)

    if args.delete_ids_out:
        with open(args.delete_ids_out, "w") as o:
            for r in aged:
                pid = r.get("id", (r.get("payload", r)).get("point_id"))
                if pid is not None:
                    o.write(f"{pid}\n")
        print(f"wrote {len(aged)} delete-ids -> {args.delete_ids_out}", file=sys.stderr)

    if args.keep_out:
        wopn = gzip.open if args.keep_out.endswith(".gz") else open
        with wopn(args.keep_out, "wt", encoding="utf-8") as o:
            for r in keep:
                o.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(keep)} in-window records -> {args.keep_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
