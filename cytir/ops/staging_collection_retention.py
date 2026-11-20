#!/usr/bin/env python3
"""Drop LANDED-and-verified staging collections from aw-qdrant-tier1
(task 659x/661, 2026-09-26).

WHY THIS EXISTS
----------------
Root cause of the 2026-09-25 news outage (NEXT.md, this task's own
notes): the staging container mmaps many files per collection and
accumulates a new one roughly daily per arm (chmedia/swissinfo/hn/
hn_discussion), never removed -- ~130 collections was enough to exhaust
the kernel's vm.max_map_count. Raising the limit
(the build box's 99-aw-qdrant.conf sysctl drop-in) fixed the immediate crash; it does
NOT fix the growth, which will eventually exhaust any limit. This script
is the other half: once a day's pack has LANDED (twin == prod, verified
at landing time -- the receipt records both counts, never re-derived
live here) its staging collection has done its job and is safe to drop.
The last `keep_days` days are kept regardless, so a landed-but-recent
pack is still inspectable for debugging.

SCOPE. This ONLY drops collections this naming convention owns
(`aw_dnews(256)?_<YYYYMMDD>_<chmedia|swissinfo|hn|hn_discussion>`) --
never touches an unrecognized collection name (the staging store also
holds independently-gated packs unrelated to the daily news lane; see
the build box's staging-qdrant ensure script's own comments on why this store is
not a generic scratch space). A collection whose owning pack has no
LANDED receipt, or whose receipt shows twin != prod, is left alone --
retention is conservative by construction, never a forced cleanup.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import re

from cytir.news import daily as D

COLLECTION_RE = re.compile(
    r"^aw_dnews(?:256)?_(\d{8})_(chmedia|swissinfo|hn|hn_discussion)$")

DEFAULT_KEEP_DAYS = 7
DEFAULT_STAGING_URL = "http://localhost:6398"


def parse_collection_name(name: str) -> tuple[str, str] | None:
    """`(day_iso, arm)` for a collection this retention job owns, else
    `None` (never touched). `day_iso` is the collection's OWN stamped
    day, exactly as its name carries it -- for hn_discussion that is
    day+1 relative to the pack day (hn_discussion_stage.sh's own AW_NOW
    anchoring, see hn_discussion_daily.run_stage()'s docstring); callers
    needing the PACK day use owning_pack_id(), not this day directly."""
    m = COLLECTION_RE.match(name)
    if not m:
        return None
    dayc, arm = m.groups()
    return f"{dayc[0:4]}-{dayc[4:6]}-{dayc[6:8]}", arm


def owning_pack_id(day: str, arm: str) -> str:
    """The promotion-queue pack id whose receipt governs this
    collection's landed status. hn_discussion's collection day is day+1
    relative to its pack (see parse_collection_name's docstring); every
    other arm's pack day is the collection's own day, unshifted."""
    dayc = day.replace("-", "")
    if arm == "hn_discussion":
        pack_day = (_dt.date.fromisoformat(day) - _dt.timedelta(days=1))
        return f"hn-discussion-daily-{pack_day.strftime('%Y%m%d')}"
    return f"news-daily-{dayc}"


def is_pack_landed_and_verified(pack_id: str, queue_dir: pathlib.Path) -> bool:
    """LANDED, and the receipt's OWN recorded twin/prod point counts
    agree -- the same verification `promote_queue.py` already performed
    at landing time, read back, never re-derived live here (a stale
    reload of a receipt is cheap; a live twin/prod recount is not, and
    this script has no business touching either mirror)."""
    receipt_path = queue_dir / pack_id / "receipt.json"
    if not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError):
        return False
    if receipt.get("status") != "LANDED":
        return False
    twin_after = ((receipt.get("twin") or {}).get("layout_after") or {})
    prod_after = ((receipt.get("prod") or {}).get("layout_after") or {})
    twin_n = twin_after.get("points_count")
    prod_n = prod_after.get("points_count")
    return twin_n is not None and twin_n == prod_n


def collections_to_drop(collection_names: list[str], *, queue_dir: pathlib.Path,
                        today: _dt.date,
                        keep_days: int = DEFAULT_KEEP_DAYS) -> list[dict]:
    """Pure decision function: no network, no qdrant client. Returns
    `[{"collection", "day", "arm", "pack_id"}, ...]` for every collection
    safe to drop -- old enough (>= keep_days) AND its owning pack shows
    LANDED with twin == prod. A collection whose name this job does not
    recognize, or whose owning pack is not (yet, or ever) landed-and-
    verified, is silently kept -- retention never guesses."""
    out = []
    for name in collection_names:
        parsed = parse_collection_name(name)
        if parsed is None:
            continue
        day, arm = parsed
        age_days = (today - _dt.date.fromisoformat(day)).days
        if age_days < keep_days:
            continue
        pack_id = owning_pack_id(day, arm)
        if is_pack_landed_and_verified(pack_id, queue_dir):
            out.append({"collection": name, "day": day, "arm": arm, "pack_id": pack_id})
    return out


def run(*, staging_url: str, queue_dir: pathlib.Path, today: _dt.date,
       keep_days: int, dry_run: bool, log=print) -> dict:
    from cytir.core.qdrant import qdrant_client  # noqa: PLC0415
    client = qdrant_client(staging_url, timeout=60)
    names = [c.name for c in client.get_collections().collections]
    plan = collections_to_drop(names, queue_dir=queue_dir, today=today,
                               keep_days=keep_days)
    log(f"[retention] {len(names)} staging collection(s), "
       f"{len(plan)} eligible to drop (keep_days={keep_days})")
    dropped = []
    for entry in plan:
        log(f"[retention] {'WOULD DROP' if dry_run else 'DROP'} "
           f"{entry['collection']} (pack={entry['pack_id']} landed+verified)")
        if not dry_run:
            client.delete_collection(entry["collection"])
            dropped.append(entry["collection"])
    return {"schema": "aw.staging-collection-retention.v1",
           "checked_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
           "staging_url": staging_url, "total_collections": len(names),
           "eligible": plan, "dropped": dropped, "dry_run": dry_run}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--staging-url", default=DEFAULT_STAGING_URL)
    ap.add_argument("--queue-dir", default=str(D.QUEUE_DIR))
    ap.add_argument("--keep-days", type=int, default=DEFAULT_KEEP_DAYS)
    ap.add_argument("--now", default=None, help="YYYY-MM-DD override (tests only)")
    ap.add_argument("--apply", action="store_true",
                    help="actually drop collections (default: dry-run report only)")
    args = ap.parse_args(argv)

    today = (_dt.date.fromisoformat(args.now) if args.now
            else _dt.datetime.utcnow().date())
    report = run(staging_url=args.staging_url, queue_dir=pathlib.Path(args.queue_dir),
                today=today, keep_days=args.keep_days, dry_run=not args.apply)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
