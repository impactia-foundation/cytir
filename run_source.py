#!/usr/bin/env python3
"""Run a source pack end to end -- shim for `cyt source run` (task 677, S5).

    python3 run_source.py sources/ch/fedlex.yaml --plan
    cyt source run sources/ch/fedlex.yaml --plan        # the same door

The runner moved to `cytir/sources/runner.py` (task 643 S5); this repo-root name
stays because docs, task briefs and shell drivers use it (task 429 Blocker A
named it). One implementation: this file only forwards.
"""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cytir.sources.runner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
