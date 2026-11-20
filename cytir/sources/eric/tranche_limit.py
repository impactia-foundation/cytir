"""The materialization limit of one ERIC PDF tranche, read from its fetch RUN.json.

The limit a tranche is materialized with is the count of PDFs its fetch admitted
(`counts.pdf_complete` in the acquisition RUN.json). The chain driver used to
hold a hand table of these; a new tranche then needed a code change before the
driver could name it. This helper derives the number from the fetch receipt.

    python3 -m cytir.sources.eric.tranche_limit --root ERIC_ROOT --offset 150000

prints the integer and exits 0 when RUN.json exists with status "complete";
exits 3 (prints nothing) when it is missing, unreadable or not complete, so the
caller waits instead of guessing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

NOT_READY = 3
TRANCHE_SIZE = 10000


def run_json_path(root: pathlib.Path, offset: int, size: int = TRANCHE_SIZE) -> pathlib.Path:
    name = f"official-pdf-tranche-{offset:06d}-{offset + size - 1:06d}-v1"
    return root / "acquisition" / name / "RUN.json"


def tranche_limit(root: pathlib.Path, offset: int, size: int = TRANCHE_SIZE) -> int | None:
    """`pdf_complete` of a complete fetch RUN.json, else None."""
    try:
        run = json.loads(run_json_path(root, offset, size).read_text())
        if run.get("status") != "complete":
            return None
        return int(run["counts"]["pdf_complete"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--offset", required=True, type=int)
    parser.add_argument("--size", type=int, default=TRANCHE_SIZE)
    args = parser.parse_args(argv)
    limit = tranche_limit(args.root, args.offset, args.size)
    if limit is None:
        return NOT_READY
    print(limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
