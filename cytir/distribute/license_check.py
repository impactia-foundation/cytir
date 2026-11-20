#!/usr/bin/env python3
"""cytir.distribute.license_check — license coverage assertion (task 650, S8).

MOVED HERE from the retired `assert_license` module (task 307), which enforced
ADR-039's "every record ships a `license` field" but had **no caller
anywhere in the pipeline** (task 643 §4/§1.2, task 645 follow-up #3) —
the retired pipeline dump script's `--compact` mode (now `cyt dump`, `python -m cytir.distribute.dump`) shipped whatever it scrolled,
license coverage unchecked. `cytir/distribute/dump.py` is now that caller:
every export calls `assert_export_license_coverage()` below before it
writes a shard.

the retired `assert_license` module is now a shim (`from cytir.distribute.license_check
import *`), so the existing `python3 assert_license.py --qdrant-url ...`
staging-preflight usage keeps working.

Per ADR-112 point 2, this checks **presence**, never content: a `license`
value of `"unknown"` still counts as present and ships (the licence field
describes the source; it is not a distribution gate — see
`specs/dump-format.md` §3.3). What this refuses is a MISSING field, which
ADR-039 never permitted in the first place.

Usage (standalone, staging preflight — unchanged from task 307):
    python3 assert_license.py --qdrant-url http://localhost:6398 \
        --out license-report.json

    # gate a single collection at a stricter threshold
    python3 assert_license.py --collection aw_dgithub_cov --threshold 1.0

Usage (the new export-path caller, `cytir/distribute/dump.py`):
    from cytir.distribute.license_check import assert_export_license_coverage
    assert_export_license_coverage(rows, threshold=0.95)  # rows = projected
                                                            # payload dicts
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import pathlib as _pathlib

# Started by path (`python aw/...py`), sys.path[0] is this file's own directory, not the repo
# root: put the root first so `import cytir` resolves (pinned by the script start-guard test).
_REPO_ROOT = str(_pathlib.Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_COLLECTIONS = [
    "aw_dgithub_cov", "aw_dhn_cov", "aw_dswissinfo_cov", "aw_dsitemap_cov",
    "aw_dchmedia_cov", "aw_dtier1_cov", "aw_dnewscards_cov",
    "aw_dretailcards_cov", "aw_dzefix_cov", "aw_dwave2_cov",
]

#: Shared with `assert_export_license_coverage` — one number, not two
#: independently-chosen defaults (engineering-doctrine #2, config is data).
DEFAULT_THRESHOLD = 0.95


def has_license(pl: dict) -> bool:
    v = pl.get("license")
    return v is not None and str(v).strip() != ""


class LicenseCoverageRefusal(ValueError):
    """Export-path refusal: coverage fell below the threshold."""


class CoverageTally:
    """Streaming accumulator: one row at a time, bounded memory regardless
    of corpus size — `cytir/distribute/dump.py` folds this into the same pass
    that writes shards, so a 44M-point dump never has to hold every row's
    payload in memory just to know its own license coverage."""

    def __init__(self, *, family_key: str = "source_type"):
        self._family_key = family_key
        self._total = 0
        self._licensed = 0
        self._by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    def add(self, payload: dict) -> None:
        self._total += 1
        fam = payload.get(self._family_key) or f"(no-{self._family_key})"
        bucket = self._by_family[fam]
        bucket[0] += 1
        if has_license(payload):
            self._licensed += 1
            bucket[1] += 1

    def report(self, *, threshold: float = DEFAULT_THRESHOLD) -> dict:
        pct = (self._licensed / self._total) if self._total else 1.0
        return {
            "total": self._total, "licensed": self._licensed, "pct": pct,
            "threshold": threshold,
            "families": {f: {"total": t, "licensed": lic,
                              "pct": (lic / t if t else 1.0)}
                         for f, (t, lic) in sorted(self._by_family.items(),
                                                   key=lambda kv: -kv[1][0])},
        }

    def assert_ok(self, *, threshold: float = DEFAULT_THRESHOLD) -> dict:
        report = self.report(threshold=threshold)
        if report["pct"] < threshold:
            offenders = [f for f, r in report["families"].items()
                        if r["pct"] < 1.0]
            raise LicenseCoverageRefusal(
                f"license coverage {report['pct']:.1%} < required "
                f"{threshold:.0%} ({report['total'] - report['licensed']} "
                f"of {report['total']} row(s) missing `license`); families "
                f"with a gap: {', '.join(offenders) or '(overall only)'}")
        return report


def assert_export_license_coverage(rows, *, threshold: float = DEFAULT_THRESHOLD,
                                    family_key: str = "source_type") -> dict:
    """Coverage check over already-projected dump rows (payload dicts).

    Unlike `scan_collection` (below), this never touches Qdrant. Convenience
    wrapper over `CoverageTally` for callers (tests, small exports) that
    already have every row in memory; `cyt dump` itself uses `CoverageTally`
    directly, one row at a time, in the same pass that writes shards.
    """
    tally = CoverageTally(family_key=family_key)
    for row in rows:
        tally.add(row)
    return tally.assert_ok(threshold=threshold)


# ---------------------------------------------------------- qdrant preflight
def scan_collection(client, name: str) -> dict:
    total = 0
    licensed = 0
    by_wave: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [total, licensed]
    offset = None
    while True:
        pts, offset = client.scroll(
            name, limit=1000, offset=offset,
            with_payload=["license", "provenance"], with_vectors=False)
        if not pts:
            break
        for p in pts:
            pl = p.payload or {}
            total += 1
            wave = pl.get("provenance") or "(no-provenance)"
            row = by_wave[wave]
            row[0] += 1
            if has_license(pl):
                licensed += 1
                row[1] += 1
        if offset is None:
            break
    pct = (licensed / total) if total else 1.0
    waves = {
        w: {"total": t, "licensed": lic, "pct": (lic / t if t else 1.0)}
        for w, (t, lic) in sorted(by_wave.items(), key=lambda kv: -kv[1][0])
    }
    return {"collection": name, "total": total, "licensed": licensed,
            "pct": pct, "waves": waves}


def main() -> int:
    from cytir.core.qdrant import qdrant_client  # noqa: E402  (lazy: CLI-only dep)

    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant-url", default="http://localhost:6398")
    ap.add_argument("--collection", action="append", default=None,
                     help="repeatable; default = the 10 known staging "
                          "collections (see DEFAULT_COLLECTIONS)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                     help="minimum acceptable overall license coverage per "
                          "collection before exit-non-zero")
    ap.add_argument("--out", default=None, help="write full JSON report here")
    args = ap.parse_args()

    collections = args.collection or DEFAULT_COLLECTIONS
    client = qdrant_client(args.qdrant_url, timeout=600,
                           check_compatibility=False)

    report = {"qdrant_url": args.qdrant_url, "threshold": args.threshold,
              "collections": []}
    failed = []
    for name in collections:
        try:
            r = scan_collection(client, name)
        except Exception as e:  # noqa: BLE001
            print(f"!! {name}: could not scan ({type(e).__name__}: {e})",
                  file=sys.stderr)
            report["collections"].append({"collection": name, "error": str(e)})
            failed.append(name)
            continue
        report["collections"].append(r)
        status = "OK" if r["pct"] >= args.threshold else "BELOW THRESHOLD"
        print(f"{name:24s} total={r['total']:>8} licensed={r['licensed']:>8} "
              f"pct={r['pct']:.1%}  [{status}]")
        for w, wr in r["waves"].items():
            if wr["pct"] < 1.0:
                print(f"    wave={w:28s} total={wr['total']:>8} "
                      f"licensed={wr['licensed']:>8} pct={wr['pct']:.1%}")
        if r["pct"] < args.threshold:
            failed.append(name)

    print()
    if failed:
        print(f"FAIL: {len(failed)} collection(s) below threshold "
              f"{args.threshold:.0%}: {', '.join(failed)}")
    else:
        print(f"PASS: all {len(collections)} collections >= "
              f"{args.threshold:.0%} license coverage")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"report written -> {args.out}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
