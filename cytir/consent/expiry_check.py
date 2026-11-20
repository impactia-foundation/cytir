"""Guard: fail BEFORE the sealed consent probes pass `SEALED_MAX_AGE_DAYS` (tasks 738, 2531, 2532).

Reports only r6 inputs: hosts of families whose contract makes them sealed-probe inputs
(`sealed_family_refusal` == ""), read from the evidence pointers the contracts carry.
Exit 0: every r6 record has more than `--within-days` left (or no sealed probe loads in this
tree, as in the public cut). Exit 1: one expires within the window or already expired.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

from .derive import SEALED_MAX_AGE_DAYS, _parse_utc, load_sealed_probes, sealed_family_refusal
from .stamp import CONTRACTS_PATH, REPO

_FMT = "%Y-%m-%dT%H:%M:%SZ"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cytir.consent.expiry_check", description=__doc__.splitlines()[0])
    ap.add_argument("--within-days", type=float, default=7)
    ap.add_argument("--as-of", default=None, help="ISO UTC (YYYY-MM-DDTHH:MM:SSZ or a date); default now")
    ap.add_argument("--contracts", default=str(CONTRACTS_PATH))
    ap.add_argument("--repo-root", default=str(REPO), help="root the evidence pointers resolve against")
    args = ap.parse_args(argv)

    if args.as_of:
        raw = args.as_of if "T" in args.as_of else args.as_of + "T00:00:00Z"
        now = _parse_utc(raw)
        if now is None:
            print(f"REFUSED: --as-of {args.as_of!r} is not ISO UTC", file=sys.stderr)
            return 2
    else:
        now = datetime.now(timezone.utc).replace(microsecond=0)
    with open(args.contracts) as fh:
        families = json.load(fh)["families"]
    r6 = [f for f in families if not sealed_family_refusal(f)]
    probes = load_sealed_probes(r6, args.repo_root, strict=False)
    rows = [(fid, host, rec) for fid, hosts in sorted(probes.items()) for host, rec in sorted(hosts.items())]
    if not rows:
        print("SKIP: no sealed probe files in this tree")
        return 0
    window = timedelta(days=args.within_days)
    earliest, failing = None, []
    for fid, host, rec in rows:
        seen = _parse_utc(rec.get("checked_at"))
        if seen is None:
            print(f"{fid} {host} checked_at={rec.get('checked_at')!r} UNPARSEABLE")
            failing.append((fid, host))
            continue
        end = seen + timedelta(days=SEALED_MAX_AGE_DAYS)
        left = (end - now).total_seconds() / 86400
        print(f"{fid} {host} checked_at={rec['checked_at']} expires_at={end.strftime(_FMT)} days_left={left:.1f}")
        earliest = end if earliest is None or end < earliest else earliest
        if end - now <= window:
            failing.append((fid, host))
    if earliest is not None:
        print(f"earliest r6 expiry: {earliest.strftime(_FMT)} ({(earliest - now).total_seconds() / 86400:.1f} days)")
    if failing:
        print(f"FAIL: {len(failing)} r6 sealed probe record(s) expire within {args.within_days:g} days "
              "or already expired - re-seal with the task 738/2531 recipe")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
