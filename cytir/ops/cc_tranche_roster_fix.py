"""Fix a never-queued tranche roster in place before launch (task 716, tranche 13).

The one-off roster maker of tranche 13 (task 714) wrote 7,516 rows, all `max_pages: 120`.
The Opus review of PR #648 found four faults in the ROSTER (not in the depth code). This step fixes them,
reproducibly, on the real file:

  (a) depth follows the class evidence: a row whose `class_basis` is `endorsed` (only "endorsed by edu/gov
      registers") loses `max_pages` and runs at the default; a `self-signal` row (TLD / registry) keeps its depth;
  (b) `www\\d*.` hosts fold onto their bare domain: one row per folded domain inside the tranche (the row
      with the most captures, `cap_n200`, then the best `score`, then the earlier row), and no row whose folded
      domain is already in a queue of an earlier tranche. Rows keep their own `domain`: only the test is folded;
  (c) no row under an EXCLUDED PUBLISHER's registrable domain (`excluded_publishers.json`, field `reg`; the
      check is by registrable domain, never the exact host);
  (d) counts per class and per depth, before and after, are printed.

    python -m cytir.ops.cc_tranche_roster_fix --roster R --queued Q [Q ...] --exclude E [E ...] --backup-suffix S

Overwrites R in place after copying it to `R<S>` (refuses when the backup exists), and prints both sha256.
The roster maker itself is `cytir.ops.cc_tranche_roster` (task 719), which calls `fix()` below.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

_WWW = re.compile(r"^www\d*\.")


def fold(domain: str) -> str:
    """Fold `www.`, `www2.`, `www150.` onto the bare domain."""
    return _WWW.sub("", (domain or "").strip().lower())


def _items(path: Path) -> list:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d if isinstance(d, list) else d.get("domains", d.get("hosts", []))


def queued_hosts(paths) -> set[str]:
    return {fold(x["domain"] if isinstance(x, dict) else x) for p in paths for x in _items(p)}


def excluded_regs(paths) -> set[str]:
    out = set()
    for p in paths:
        for e in _items(p):
            out.add((e.get("reg") or e["domain"]).lower())
    return out


def counts(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "by_class": dict(sorted(collections.Counter(r["class"] for r in rows).items())),
        "by_class_basis": dict(sorted(collections.Counter(f"{r['class']}/{r['class_basis']}" for r in rows).items())),
        "by_depth": dict(sorted(collections.Counter(str(r.get("max_pages", "default")) for r in rows).items())),
    }


def fix(rows: list[dict], queued: set[str], excluded: set[str]) -> tuple[list[dict], dict]:
    """Return (fixed rows, report). Row order is kept (best-first)."""
    for r in rows:
        if r.get("class_basis") not in ("endorsed", "self-signal"):
            raise SystemExit(f"REFUSED: {r.get('domain')!r} has class_basis {r.get('class_basis')!r}")
    report = {"before": counts(rows), "dropped_excluded_publisher": [], "dropped_in_earlier_queue": 0,
              "dropped_duplicate_fold": 0, "duplicate_groups": 0}
    kept = []
    for r in rows:
        if r["reg"].lower() in excluded:
            report["dropped_excluded_publisher"].append(r["domain"])
        elif fold(r["domain"]) in queued:
            report["dropped_in_earlier_queue"] += 1
        else:
            kept.append(r)
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(kept):
        groups[fold(r["domain"])].append(i)
    drop: set[int] = set()
    for idx in groups.values():
        if len(idx) > 1:
            report["duplicate_groups"] += 1
            best = min(idx, key=lambda i: (-kept[i].get("cap_n200", 0), -kept[i].get("score", 0), i))
            drop.update(set(idx) - {best})
    report["dropped_duplicate_fold"] = len(drop)
    out = []
    for i, r in enumerate(kept):
        if i in drop:
            continue
        r = dict(r)
        if r["class_basis"] == "endorsed":
            r.pop("max_pages", None)
        out.append(r)
    if not out:
        raise SystemExit("REFUSED: nothing left")
    seen = [fold(r["domain"]) for r in out]
    if len(set(seen)) != len(seen) or set(seen) & queued:
        raise SystemExit("REFUSED: a folded duplicate or a queued host survived the fix")
    report["after"] = counts(out)
    return out, report


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog or "cc_tranche_roster_fix", description=__doc__.split("\n\n")[0])
    ap.add_argument("--roster", required=True, type=Path)
    ap.add_argument("--queued", nargs="+", required=True, type=Path)
    ap.add_argument("--exclude", nargs="+", required=True, type=Path)
    ap.add_argument("--backup-suffix", required=True)
    a = ap.parse_args(argv)
    backup = a.roster.with_name(a.roster.name + a.backup_suffix)
    if backup.exists():
        raise SystemExit(f"REFUSED: {backup} exists; the roster is already fixed or the suffix is reused")
    rows = json.loads(a.roster.read_text(encoding="utf-8"))
    fixed, report = fix(rows, queued_hosts(a.queued), excluded_regs(a.exclude))
    shutil.copy2(a.roster, backup)
    a.roster.write_text(json.dumps(fixed), encoding="utf-8")
    report["sha256_before"] = sha256(backup)
    report["sha256_after"] = sha256(a.roster)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
