"""Discover the admitted-story frontier for the HN-discussion ingester.

`cytir.sources.hn.discussion_ingest` (task 510/601) consumes a pre-built JSONL frontier
of HN item IDs; it does no discovery of its own by design (one ingester, one
job -- fetch/select/render, never also decide admission). This module is that
discovery step, reusing the SAME official Algolia `search_by_date` window
enumeration `cytir.sources.hn.freshness_ingest` already implements
(`algolia_window`/`ALGOLIA`) rather than a second implementation of its
exhaustive time-partitioning-under-the-1000-hit-cap logic.

Deliberately DIFFERENT from `cytir.sources.hn.freshness_ingest.qualifying_stories`: that
function drops every story with no external `url` (Ask HN / Show HN text
posts), because the LINKED-PAGE arm has nothing to embed without one. The
discussion arm's row IS the thread, so a self-post is a perfectly good
discussion candidate -- dropping it here would be silently narrower than the
PRD's actual selection basis (points/age only). Admission here is exactly the
PRD's stated trigger: community score >= --min-points within the
--since-days window, nothing more.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx

from cytir.acquire.artifact_io import atomic_publish, canonical_json
from cytir.sources.hn.freshness_ingest import ALGOLIA, UA, HNSourceError, algolia_window  # noqa: F401


SCHEMA = "aw.hn-discussion-frontier.v1"
DEFAULT_MIN_POINTS = 40
DEFAULT_SINCE_DAYS = 7


def discover(client: Any, *, since_days: int, min_points: int,
            now_i: int) -> tuple[list[int], dict[str, Any]]:
    if since_days <= 0:
        raise ValueError("--since-days must be positive")
    if min_points <= 0:
        raise ValueError("--min-points must be positive")
    lo_i = now_i - since_days * 86400
    hits, enumeration = algolia_window(client, lo_i, now_i, "story")
    admitted = sorted(
        {int(hit["objectID"]) for hit in hits
         if isinstance(hit.get("points"), int) and hit["points"] >= min_points})
    receipt = {
        "schema": SCHEMA, "source": "algolia", "tags": "story",
        "since_days": since_days, "min_points": min_points,
        "window": {"start_epoch": lo_i, "end_epoch": now_i},
        "completeness": "time-window-complete",
        "counts": {"stories_in_window": len(hits), "admitted": len(admitted)},
        "enumeration": enumeration,
    }
    return admitted, receipt


def write_frontier(item_ids: list[int], out_path: Path,
                   receipt: dict[str, Any], receipt_path: Path) -> None:
    body = "".join(canonical_json({"id": item_id}) + "\n" for item_id in item_ids)
    if not body:
        raise HNSourceError("HN discussion frontier discovery admitted zero stories")
    encoded = body.encode("utf-8")
    receipt = dict(receipt)
    receipt["frontier_sha256"] = hashlib.sha256(encoded).hexdigest()
    receipt_encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    for path, data in ((out_path, encoded), (receipt_path, receipt_encoded)):
        if path.exists():
            if path.read_bytes() != data:
                raise HNSourceError(f"immutable frontier artifact collision: {path}")
        else:
            atomic_publish(path, data)


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--since-days", type=int, default=DEFAULT_SINCE_DAYS)
    parser.add_argument("--min-points", type=int, default=DEFAULT_MIN_POINTS)
    parser.add_argument("--now", default=None,
                        help="UTC date YYYY-MM-DD; defaults to the real current time")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.now:
        now_i = int(dt.datetime.strptime(args.now, "%Y-%m-%d")
                   .replace(tzinfo=dt.timezone.utc).timestamp())
    else:
        now_i = int(dt.datetime.now(dt.timezone.utc).timestamp())
    with httpx.Client() as client:
        item_ids, receipt = discover(
            client, since_days=args.since_days, min_points=args.min_points,
            now_i=now_i)
    write_frontier(item_ids, args.out, receipt, args.receipt)
    print(json.dumps({"admitted": len(item_ids), "out": str(args.out),
                      "receipt": str(args.receipt)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
