#!/usr/bin/env python3
"""Weekly aged-out tombstone pack for the news family (task 659x, 2026-09-25).

WHY THIS EXISTS
----------------
`news_backlog_composer.compute_aged_out_tombstones` scrolls the WHOLE twin
news population (`scroll_news_points`, is_news() filtered client-side) to
find rows past the 36-month retention horizon (ADR-085 amendment,
2026-08-10). Against a 35M+-point twin this scroll alone added 30+ minutes
to the 09-24 daily run -- for a trickle of rows: a 36-month horizon crosses
very few rows per day. The coordinator's ruling (2026-09-25): pull it off
the daily critical path entirely and run it as its OWN weekly pack, same
tombstone rules and cross-pack dedup as every other pack this queue drains.
`news_daily_pack_and_queue.py` and `hn_discussion_daily.py` no longer call
`compute_aged_out_tombstones` at all -- this script is the only caller now.

This pack carries ONLY tombstones (`kept=[]`) -- it deletes, it never adds
rows. Deletions lose nothing by waiting a week (the coordinator's own
words); a daily run staying fast is the entire point.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys

from cytir.core import twin as TWIN  # noqa: E402
from cytir.core.paths import SettingRefused  # noqa: E402
from cytir.news import compose as C  # noqa: E402
from cytir.news import daily as D  # noqa: E402

QUEUE_DIR = D.QUEUE_DIR
PACK_PREFIX = "news-aged-out-weekly-"


def run(*, twin_url: str, collection: str, retain_months: int,
       now_ref: _dt.date, week_tag: str) -> dict:
    from cytir.core.qdrant import qdrant_client  # noqa: PLC0415
    client = qdrant_client(twin_url, timeout=600)  # a full scroll is slow by
    # nature (module docstring) -- a generous timeout here is correct, not a
    # workaround; it is why this scroll no longer shares a budget with the
    # daily run's own timeout.

    aged_out_result = C.compute_aged_out_tombstones(
        client, collection, now_ref=now_ref, months=retain_months)
    tombstones = aged_out_result["tombstones"]
    age_unknown_kept = aged_out_result["age_unknown_kept"]
    print(f"[aged-out weekly] {now_ref.isoformat()} retain_months="
         f"{retain_months} aged_out={len(tombstones)} "
         f"age_unknown_kept={len(age_unknown_kept)}", file=sys.stderr)

    pack_id = f"{PACK_PREFIX}{week_tag}"
    already_claimed = C.other_queued_tombstone_ids(str(QUEUE_DIR), pack_id)
    tombstones = C.dedupe_tombstones_against_queue(
        tombstones, already_claimed,
        log=lambda msg: print(msg, file=sys.stderr))
    C.assert_no_cross_pack_tombstone_collision(tombstones, already_claimed)

    out_dir = QUEUE_DIR / pack_id
    manifest = C.write_pack(
        str(out_dir), kept=[], tombstones=tombstones, pack_id=pack_id,
        family="news", collection=collection, days=[now_ref.isoformat()],
        extra_manifest={"arm": "aged_out_weekly", "retain_months": retain_months,
                        "age_unknown_kept": len(age_unknown_kept),
                        "age_unknown_kept_rows": age_unknown_kept})
    print(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--collection", default=TWIN.collection())
    ap.add_argument("--twin-url", default=D.TWIN_URL)
    ap.add_argument("--retain-months", type=int, default=36)
    ap.add_argument("--now", default=None,
                    help="YYYY-MM-DD; defaults to today UTC")
    args = ap.parse_args(argv)

    # Fail at START, before the (slow) twin scroll: the tombstone pack goes to the operator
    # setting `queue_dir`, which an unconfigured box refuses by name.
    try:
        str(QUEUE_DIR)
    except SettingRefused as exc:
        print(f"[REFUSED] news aged-out weekly: {exc}", file=sys.stderr, flush=True)
        return 2

    now_ref = (_dt.date.fromisoformat(args.now) if args.now
              else _dt.datetime.utcnow().date())
    # ISO week tag, not a calendar day: this pack runs weekly, and its
    # promotion-queue directory name should say so rather than looking like
    # a daily pack with a coincidental day suffix.
    iso = now_ref.isocalendar()
    week_tag = f"{iso[0]}W{iso[1]:02d}"

    run(twin_url=args.twin_url, collection=args.collection,
       retain_months=args.retain_months, now_ref=now_ref, week_tag=week_tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
