"""The roster maker of the CC outlink tranches (task 719; the one-off scripts of tranche 13 become this module).

One way to make a `tranche<N>_never_queued.json`: class + spam pre-screen of the outlink candidates,
the drops, the depth, and (when the ring is too big for the disk) the split. The rules are those of tranche 13
(PORTFOLIO 3h, ruling 27 T2), plus the classes of later rulings:

  * class: only the classes of `CLASS_DEPTH` (`edu`, `gov`, `ref`; `health` since ruling 56, `corporate` since
    ruling 59). A row of another class is dropped and counted. A new class is one entry in `CLASS_DEPTH`, not code;
  * spam pre-screen: a host on the sealed hold-out list (`data/spam-host-lists.v1.json`), a hostname with a spam
    word, a link farm (>= 50 % of its source hosts are held out) is dropped;
  * drops, in this order, by `cytir.ops.cc_tranche_roster_fix.fix`: under an EXCLUDED PUBLISHER's registrable domain;
    whose `www\\d*.`-folded host is in a queue of an earlier tranche; a duplicate `www\\d*.` fold inside the tranche;
  * depth: per class and `class_basis`, from `CLASS_DEPTH`. For `edu`, `gov`, `ref`, `health`: a `self-signal` row
    (the hostname or registry carries the credential) gets `max_pages: 120`; an `endorsed` row (only the link graph
    does) carries no `max_pages` and runs at the default 60. For `corporate`: every row gets 120 (the host is named by
    a board ruling, not found by the link graph; its `class_basis` is `self-signal` and the ruling is named in
    `admitted_by`);
  * order: best endorsed first: `n_src` descending, then the host name.

    python -m cytir.ops.cc_tranche_roster make --candidates C --queued Q [Q ...] --exclude E [E ...] \\
        --host-lists H --out OUT [--keep-share 0.5 --rest-out REST] [--report R]

`--candidates` is a file of classified candidate rows (the ring file: `domain, reg, class, class_basis, n_src, ...`).
The classifier itself and the outlink graph step stay in the private build tree for now. `--keep-share S` writes the first S of the hosts to OUT and the rest to
`--rest-out` (not launched). Refuses to overwrite OUT, and refuses when nothing is left.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
from pathlib import Path

from cytir.ops import spam_rows
from cytir.ops.cc_tranche_roster_fix import counts, excluded_regs, fix, fold, queued_hosts

DEPTH_DEEP = 120
# The declared allow-list of classes. Per class: {class_basis: max_pages}; the key "*" is the depth of every other
# basis. A value of None is "no `max_pages`" (the tranche runs at its default, 60). A class not named here is dropped.
_SELF_DEEP = {"self-signal": DEPTH_DEEP, "*": None}
CLASS_DEPTH: dict[str, dict[str, int | None]] = {
    "edu": _SELF_DEEP,
    "gov": _SELF_DEEP,
    "ref": _SELF_DEEP,
    "health": _SELF_DEEP,                  # ruling 47/56 (task 1057), POLICY 4.6
    "corporate": {"*": DEPTH_DEEP},        # ruling 59 (task 1065), PORTFOLIO 3i: a ruling-named corporate site
}
ADMITTED_CLASSES = tuple(CLASS_DEPTH)
LINK_FARM_SHARE = 0.5

HOST_SPAM_NAME = re.compile(
    r"casino|slots?\b|slot-|-slot|togel|judi|gacor|sbobet|poker|betting|bet365|1xbet|mostbet|escort|callgirl|"
    r"call-girl|porn|xxx|sex|viagra|cialis|kamagra|pharmacy|payday|loan|diploma|essay|bitcoin|crypto|"
    r"forex|vape|cbd|keto|dating|hookup|bingo|lotto|lottery|pokies", re.IGNORECASE)


def hostname_spam(host: str) -> bool:
    return bool(spam_rows.lexicon_hits(host.replace("-", " ").replace(".", " ")) or HOST_SPAM_NAME.search(host))


def prescreen(rows: list[dict], held: set[str]) -> tuple[list[dict], dict]:
    """Class + spam pre-screen of screened rows. Does not trust an input file: it checks again."""
    drop: collections.Counter = collections.Counter()
    keep = []
    for r in rows:
        if r.get("class") not in CLASS_DEPTH:
            drop["class_not_admitted"] += 1
        elif fold(r["domain"]) in held:
            drop["held_host"] += 1
        elif hostname_spam(r["domain"]):
            drop["hostname_spam_lexicon"] += 1
        elif r.get("n_src_spammy", 0) / max(r.get("n_src", 1), 1) >= LINK_FARM_SHARE:
            drop["link_farm"] += 1
        else:
            keep.append(r)
    return keep, dict(drop)


def assign_depth(rows: list[dict]) -> list[dict]:
    """`max_pages` per `CLASS_DEPTH[class][class_basis]`; None -> no `max_pages` (default 60)."""
    out = []
    for r in rows:
        r = dict(r)
        r.pop("max_pages", None)
        rule = CLASS_DEPTH[r["class"]]
        depth = rule.get(r["class_basis"], rule["*"])
        if depth is not None:
            r["max_pages"] = depth
        out.append(r)
    return out


def order_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (-r.get("n_src", 0), r["domain"]))


def make_roster(candidates: list[dict], *, queued: set[str], excluded: set[str], held: set[str]) -> tuple[list[dict], dict]:
    """Candidates (screened rows) -> (roster rows, report with a count per drop reason)."""
    screened, drops = prescreen(candidates, held)
    if not screened:
        raise SystemExit("REFUSED: nothing left after the class + spam pre-screen")
    fixed, rep = fix(screened, queued, excluded)
    rows = order_rows(assign_depth(fixed))
    report = {"candidates": len(candidates), "dropped_prescreen": drops,
              "dropped_excluded_publisher": len(rep["dropped_excluded_publisher"]),
              "dropped_excluded_publisher_hosts": rep["dropped_excluded_publisher"],
              "dropped_in_earlier_queue": rep["dropped_in_earlier_queue"],
              "dropped_duplicate_fold": rep["dropped_duplicate_fold"], "roster": counts(rows)}
    return rows, report


def split_share(rows: list[dict], share: float) -> tuple[list[dict], list[dict]]:
    """The first `share` of the ordered rows (n_src desc, host name) and the rest."""
    if not 0 < share <= 1:
        raise SystemExit("REFUSED: --keep-share must be in (0, 1]")
    rows = order_rows(rows)
    k = math.ceil(len(rows) * share)
    return rows[:k], rows[k:]


def _write_new(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise SystemExit(f"REFUSED: {path} exists")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(rows), encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog or "cc_tranche_roster", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--candidates", required=True, type=Path)
    mk.add_argument("--queued", nargs="+", required=True, type=Path)
    mk.add_argument("--exclude", nargs="+", required=True, type=Path)
    mk.add_argument("--host-lists", required=True, type=Path)
    mk.add_argument("--out", required=True, type=Path)
    mk.add_argument("--keep-share", type=float, default=1.0)
    mk.add_argument("--rest-out", type=Path)
    mk.add_argument("--report", type=Path)
    a = ap.parse_args(argv)
    if a.keep_share < 1 and not a.rest_out:
        raise SystemExit("REFUSED: --keep-share < 1 needs --rest-out")
    held = {fold(d) for d in spam_rows.load_host_lists(a.host_lists)["hold_out"]}
    cand = json.loads(a.candidates.read_text(encoding="utf-8"))
    rows, report = make_roster(cand, queued=queued_hosts(a.queued), excluded=excluded_regs(a.exclude), held=held)
    keep, rest = split_share(rows, a.keep_share)
    if a.keep_share < 1:
        report["kept"], report["rest"] = counts(keep), counts(rest) if rest else {"rows": 0}
    _write_new(a.out, keep)
    if rest:
        _write_new(a.rest_out, rest)
    if a.report:
        a.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
