"""Split the never-queued part of the CC authority-frontier roster into tranche queue files (task 701).

The roster (`queue_*_fixed.json`, a list of `{"domain", "class", "score", ...}` best-first) holds 106,903
host candidates; the tranche queues of tasks 634/664/694 cover 47,575 of them. This module writes the
rest as `tranche<N>_never_queued.json` queue files in the shape `discovery_ingest.py fetch --queue`
reads, best-first and contiguous, so tranche N+1 never holds a better host than tranche N. A host is
left out when it is already in an earlier queue, on an excluded-publisher list, or on the sealed
hold-out list of hijacked hosts (`data/spam-host-lists.v1.json`, `cytir.ops.spam_rows`).

    python -m cytir.ops.cc_tranche_queue --roster R --queued Q [Q ...] --exclude E [E ...] \\
        --host-lists H --out-dir D --first-number 7 --parts 6

Writes `D/tranche<N>_never_queued.json` and `D/tranche_queue_plan.json` (counts per step); refuses when
nothing is left or when a written host is also in an earlier queue.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cytir.ops import spam_rows
from cytir.ops.cc_tranche_roster_fix import fold


def _domains(path: Path) -> list[str]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    items = d if isinstance(d, list) else d.get("domains", d.get("hosts", []))
    return [(x["domain"] if isinstance(x, dict) else x) for x in items]


def split(roster: list[dict], queued: set[str], excluded: set[str], held_out: set[str], parts: int) -> tuple[list[list[dict]], dict]:
    counts = {"roster": len(roster), "already_queued": 0, "excluded_publisher": 0, "hijacked_holdout": 0}
    left = []
    seen: set[str] = set()
    for r in roster:
        d = fold(spam_rows.norm_domain(r["domain"]))     # `www.`, `www2.`, `www150.` fold onto the bare domain (task 719)
        if d in seen:
            continue
        seen.add(d)
        if d in queued:
            counts["already_queued"] += 1
        elif d in excluded:
            counts["excluded_publisher"] += 1
        elif d in held_out:
            counts["hijacked_holdout"] += 1
        else:
            left.append(r)
    if not left:
        raise SystemExit("REFUSED: every roster host is already queued, excluded or held out")
    if parts < 1:
        raise SystemExit("REFUSED: --parts must be >= 1")
    size = -(-len(left) // parts)
    chunks = [left[i:i + size] for i in range(0, len(left), size)]
    counts.update(never_queued=len(left), parts=len(chunks), hosts_per_part=[len(c) for c in chunks])
    return chunks, counts


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog or "cc_tranche_queue", description=__doc__.split("\n\n")[0])
    ap.add_argument("--roster", required=True, type=Path)
    ap.add_argument("--queued", nargs="+", required=True, type=Path)
    ap.add_argument("--exclude", nargs="*", default=[], type=Path)
    ap.add_argument("--host-lists", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--first-number", required=True, type=int)
    ap.add_argument("--parts", required=True, type=int)
    a = ap.parse_args(argv)
    roster = json.loads(a.roster.read_text(encoding="utf-8"))
    queued = {fold(d) for p in a.queued for d in _domains(p)}
    excluded = {fold(d) for p in a.exclude for d in _domains(p)}
    held = {fold(d) for d in spam_rows.load_host_lists(a.host_lists)["hold_out"]}
    chunks, counts = split(roster, queued, excluded, held, a.parts)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for i, chunk in enumerate(chunks):
        n = a.first_number + i
        dup = {fold(spam_rows.norm_domain(r["domain"])) for r in chunk} & queued
        if dup:
            raise SystemExit(f"REFUSED: {len(dup)} host(s) of part {n} are in an earlier queue")
        f = a.out_dir / f"tranche{n}_never_queued.json"
        f.write_text(json.dumps(chunk), encoding="utf-8")
        files[n] = {"file": f.name, "hosts": len(chunk), "best_rank": roster.index(chunk[0])}
    counts["files"] = files
    (a.out_dir / "tranche_queue_plan.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
