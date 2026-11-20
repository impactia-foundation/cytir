"""`cyt ops receipt backfill-distinct-urls <pack-dir>` (task 801).

A promotion-queue receipt that a prod-only resume wrote before task 801 has no
`mechanical_plan.rows.distinct_urls`, the one number `cytir.ops.cc_tranche_delta` reads from a tranche's
first pack. This verb adds it, from the pack's own record, never by hand:

    1. the value is the receipt's own `prod.rows.distinct_urls` (or `twin.rows`): `plan()` counted the
       whole pack projection before the resume narrowed the write set;
    2. it is checked against an independent count of the distinct `url` values in the chunk shards the
       receipt names (`<side>.chunks_dir/*.texts.jsonl.gz`); a difference is a refusal, nothing is written;
    3. the receipt is rewritten atomically (`promote_queue.atomic_write_text`) with `mechanical_plan.rows.distinct_urls`
       and a `backfilled_by` note. Nothing else in it changes. A receipt that already has the value is a
       no-op; a pack that is not `LANDED` is refused.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
import time
from pathlib import Path

NOTE_KEY = "backfilled_by"


class BackfillError(RuntimeError):
    """Refused: the receipt was not touched and the message says why."""


def _side(rec: dict) -> tuple[str, dict]:
    for name in ("twin", "prod"):
        side = rec.get(name)
        if isinstance(side, dict) and isinstance(side.get("rows"), dict) \
                and side["rows"].get("distinct_urls") is not None:
            return name, side
    raise BackfillError("the receipt has no twin.rows / prod.rows with distinct_urls to take it from")


def chunk_distinct_urls(chunks_dir: str | Path) -> int:
    files = sorted(glob.glob(str(chunks_dir) + "/*.texts.jsonl.gz"))
    if not files:
        raise BackfillError(f"no *.texts.jsonl.gz under {chunks_dir}: cannot cross-check the count")
    urls: set[str] = set()
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    urls.add(json.loads(line).get("url") or "")
    return len(urls)


def backfill(pack_dir: str | Path, *, now=time.time, count_chunks=chunk_distinct_urls, log=print) -> dict:
    pack_dir = Path(pack_dir)
    path = pack_dir / "receipt.json"
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise BackfillError(f"cannot read {path}: {exc}") from exc
    if not isinstance(rec, dict):
        raise BackfillError(f"{path} is not a JSON object")
    if rec.get("status") != "LANDED":
        raise BackfillError(f"{rec.get('pack_id')}: status is {rec.get('status')!r}, not LANDED")
    plan = rec.get("mechanical_plan") if isinstance(rec.get("mechanical_plan"), dict) else {}
    have = (plan.get("rows") or {}).get("distinct_urls") if isinstance(plan.get("rows") or {}, dict) else None
    if isinstance(have, int):
        log(f"[receipt-backfill] {rec.get('pack_id')}: mechanical_plan.rows.distinct_urls already "
            f"{have}, nothing to do")
        return {"pack_id": rec.get("pack_id"), "status": "already_present", "distinct_urls": have}
    side_name, side = _side(rec)
    value = int(side["rows"]["distinct_urls"])
    chunks_dir = side.get("chunks_dir")
    if not chunks_dir:
        raise BackfillError(f"{side_name}.chunks_dir is not recorded: cannot cross-check the count")
    counted = count_chunks(chunks_dir)
    if counted != value:
        raise BackfillError(f"{rec.get('pack_id')}: the receipt says {value} distinct URLs "
                            f"({side_name}.rows) but {chunks_dir} holds {counted}; nothing written")
    rows = side["rows"]
    rec["mechanical_plan"] = {
        **plan,
        "rows": {**(plan.get("rows") or {}), "vectors": rows.get("vectors"),
                 "projected": rows.get("projected"), "distinct_urls": value},
        "derived_from": f"{side_name}.rows of this receipt, cross-checked against the chunk shards"}
    rec[NOTE_KEY] = {"by": "aw ops receipt backfill-distinct-urls (task 801)",  # rename-codemod: keep (persisted: provenance note written into the build receipt)
                     "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now())),
                     "distinct_urls": value, "source": f"{side_name}.rows.distinct_urls",
                     "chunks_dir": chunks_dir, "chunks_distinct_urls": counted}
    from cytir.index.promote_queue import atomic_write_text     # mode 0644 like the live receipts, dir fsync
    atomic_write_text(path, json.dumps(rec, indent=1) + "\n")
    log(f"[receipt-backfill] {rec.get('pack_id')}: mechanical_plan.rows.distinct_urls = {value} "
        f"(from {side_name}.rows, {counted} counted in the chunks)")
    return {"pack_id": rec.get("pack_id"), "status": "backfilled", "distinct_urls": value}


def main(argv=None, prog: str = "cyt ops receipt backfill-distinct-urls") -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("pack_dir", type=Path, help="the pack's promotion-queue directory (holds receipt.json)")
    args = ap.parse_args(argv)
    try:
        backfill(args.pack_dir)
    except BackfillError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
