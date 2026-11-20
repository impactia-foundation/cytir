"""Compact canonical append-only facet waves into current-state parquet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cytir.corpus import facet_bundle as fb


def main(argv: list[str] | None = None, *, prog: str = "cyt corpus facet-compact") -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--events-root", type=Path,
                       help="canonical facet_events root; verifies every wave manifest")
    group.add_argument("--in", dest="inputs", action="append", type=Path,
                       help="explicit manifested event wave (repeatable)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    events = (fb.load_canonical_events(args.events_root) if args.events_root
              else [event for path in args.inputs for event in fb.load_event_wave(path)])
    rows = fb.compact(events)
    manifest = fb.write_parquet(rows, args.out, source_events=events)
    print(json.dumps({"events": len(events), "current": len(rows),
                      "out": str(args.out), "manifest": manifest}, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(prog="python -m cytir.corpus.facet_cli"))
