"""cytir.gates.stamp_harness_weights — stamp `harness_weights` (the SHELL
DRIVER's own resolved path + sha256) into an arm's output JSON.

MOVED HERE from task 551's weight-stamping script in task 646
S4 (task 643 §1.3/§6.2): that path is now a shim
(`from cytir.gates.stamp_harness_weights import *`), so the shell drivers that
call it (the task 533 auth and arm drivers, and others)
and the task 551 weights-guard test keep working unchanged.

`cytir/gates/authority.py` already records what its own process env
resolved (`fusion_weights_provenance`, added by task 544) into the 1,418
leg it writes, and task 533's `seal_arm_533.py` independently
cross-checks the 629 arm's weights against the live server's own `/health`.
Both are real and both are correct. Neither is stamped by the SHELL SCRIPT
that actually chose `$W` -- the value is only ever visible if the reader
knows to open the right JSON key inside the right sub-process's output. This
script closes that gap at the harness level itself: the driver script
resolves `$W`, and this is the one line that writes `$W`'s identity into the
arm's own output file, independent of whatever gate_authority.py's or
seal_arm_533.py's own logic does. Two independently-computed stamps that
must agree are worth more than one; if they ever disagree, THAT disagreement
is the finding, not a coincidence to explain away.

Fails loud (nonzero exit, no partial write) if the output file does not
exist yet or the weights file cannot be read -- a stamp that silently
no-ops is worse than no stamp (engineering-doctrine 3: fail loud or don't
fail).

Usage:
    stamp_harness_weights_551.py --out <arm.json> --weights <path> \\
        --side <twin|build> --leg <full|dense-only|sparse-only|full-8|full-60>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp(out_path: str, weights_path: str, side: str, leg: str) -> dict:
    if not os.path.isfile(out_path):
        raise FileNotFoundError(
            f"{out_path} does not exist -- nothing to stamp; run the scorer "
            "before calling this script")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"weights file {weights_path} does not exist")

    with open(out_path) as fh:
        doc = json.load(fh)

    entry = {
        "path": os.path.abspath(weights_path),
        "sha256": sha256_file(weights_path),
        "side": side,
        "leg": leg,
        "basis": "resolved by the shell driver ($W), stamped at the harness "
                 "level -- not re-derived from the scorer's own process env",
    }
    doc["harness_weights"] = entry

    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    return entry


def main(argv=None, prog=None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--side", required=True)
    ap.add_argument("--leg", required=True)
    args = ap.parse_args(argv)

    try:
        entry = stamp(args.out, args.weights, args.side, args.leg)
    except FileNotFoundError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    print(f"stamped harness_weights into {args.out}: "
          f"{entry['sha256'][:12]} ({os.path.basename(entry['path'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
