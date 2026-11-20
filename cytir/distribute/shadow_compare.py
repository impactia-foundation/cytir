"""Shadow compare: the scroll delta against the change-log delta (task 2603; ruling 68 (2);
`specs/delta-change-log.md` 3.7; the operator runbook shadow-period-runbook.md).

Builds the delta twice from ONE base and ONE collection, with the two existing builders
(`dump.LogDeltaBuilder`, `dump.DeltaBuilder`), each into its own scratch directory, then compares
what each would ship. READ-ONLY against the collection (scroll, by-id fetch). It never touches the
fingerprint ledger's state (no advance, no lock) and never signs or publishes anything.

Equality means, per id the delta names:
  op           `add` (row, id not in the base), `replace` (row, id in the base) or `tombstone`;
  fingerprint  `pack.row_fingerprint` of the shipped row (adds and replaces);
  reason       the tombstone reason, compared only when `--tombstone-reasons` gave the scroll
               leg an independent reason (otherwise the scroll leg takes the log's reason, since
               a scroll cannot know why a row left; the id set is still compared).
No scroll fallback: a missing prerequisite or a refusing log is exit 2, never a silent scroll."""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
import time
from typing import Iterable, Optional

from cytir.distribute import dump, pack

CLASSES = ("only_in_log", "only_in_scroll", "fingerprint_differs", "op_differs", "reason_differs")
BOUND = 20
EXIT_EQUAL, EXIT_DIFFER, EXIT_REFUSED = 0, 1, 2

# A side is `{id: (op, fingerprint_or_None, reason_or_None)}`.
Side = dict


class ShadowRefusal(RuntimeError):
    pass


def compare(log: Side, scroll: Side, *, compare_reasons: bool) -> dict:
    """Pure diff of two sides. Returns `{class: [ids...]}` for the five classes (full lists, in
    point-id order); `bound()` trims them for display."""
    out: dict[str, list[str]] = {c: [] for c in CLASSES}
    for id_ in sorted(set(log) | set(scroll), key=pack.point_id_key):
        a, b = log.get(id_), scroll.get(id_)
        if b is None:
            out["only_in_log"].append(id_)
        elif a is None:
            out["only_in_scroll"].append(id_)
        elif a[0] != b[0]:
            out["op_differs"].append(id_)
        elif a[1] != b[1]:
            out["fingerprint_differs"].append(id_)
        elif compare_reasons and a[2] != b[2]:
            out["reason_differs"].append(id_)
    return out


def bound(diff: dict, n: int = BOUND) -> dict:
    return {c: {"count": len(ids), "first": ids[:n]} for c, ids in diff.items()}


class _Reasons(dict):
    """Tombstone reasons for the scroll leg: the operator's file first, then the log's, else a
    placeholder (such an id is, by construction, only-in-scroll: the log never named it)."""

    def __init__(self, given: dict, from_log: dict):
        super().__init__(given)
        self._log = from_log
        self.fallback: set[str] = set()

    def __contains__(self, k):
        return True

    def __getitem__(self, k):
        if dict.__contains__(self, k):
            return dict.__getitem__(self, k)
        if k in self._log:
            return self._log[k]
        self.fallback.add(k)
        return "superseded"


def _write_leg(rows: Iterable[dict], tombstones_fn, out_dir: pathlib.Path, *, shard_rows: int):
    """Write one leg's shards + tombstones; returns `(fingerprint by id, tombstones, counts)`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}

    def tap():
        for row in rows:
            seen[row["id"]] = pack.row_fingerprint(row)
            yield row

    _shards, total = dump.write_shards(tap(), out_dir, shard_rows=shard_rows, log=lambda _l: None)
    tombs = tombstones_fn()
    n_tomb, tomb_sha = pack.write_tombstones(tombs, str(out_dir / "tombstones.jsonl.gz"))
    return seen, tombs, {"rows": total, "tombstones": n_tomb, "tombstones_sha256": tomb_sha}


def _base_ids(sidecar: str, count: int, wanted: set[str]) -> set[str]:
    return {i for i, _fp, _k in pack.iter_fingerprints(sidecar, expected_count=count) if i in wanted}


def run(*, client, collection: str, base_manifest_path: str, log_dir: str, out_dir: pathlib.Path,
        base_dir: Optional[str] = None, ledger=None, tombstone_reasons: Optional[dict] = None,
        shard_rows: int = 50_000, stamp: Optional[str] = None) -> tuple[int, dict]:
    """Both legs plus the compare. Returns `(exit_code, report)`; the report is also written to
    `out_dir/shadow-compare.<stamp>.json`. Any refusal is raised as `ShadowRefusal`."""
    from cytir.index import change_log as CL
    stamp = stamp or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = pathlib.Path(out_dir)
    base_manifest = pack.validate_manifest(json.loads(pathlib.Path(base_manifest_path).read_text()))
    bdir = pathlib.Path(base_dir or pathlib.Path(base_manifest_path).parent)
    base_sha = pack.sha256_file(str(base_manifest_path))
    ledger_base = None
    if ledger is not None:
        from cytir.distribute.fingerprint_ledger import base_for_delta
        ledger_base = base_for_delta(ledger, expected_manifest_sha256=base_sha)
    after = base_manifest.get("change_log_segment")
    if not after:
        raise ShadowRefusal("the base manifest records no change_log_segment: there is no cut to "
                            "compare from (cut a FULL with the log on)")
    cut_before = CL.latest_cut(log_dir)
    legs: dict[str, dict] = {}

    try:
        lb, note = dump.choose_delta_source(base_manifest, bdir, log_dir=log_dir, client=client,
                                            collection=collection, use_log=True, ledger_base=ledger_base)
    except (dump.DeltaRefusal, CL.ChangeLogError) as e:
        raise ShadowRefusal(f"log leg: {e}") from e
    if lb is None:
        raise ShadowRefusal(f"log leg cannot run ({note}); a scroll fallback would hide the "
                            f"difference, so none is taken")
    t0 = time.monotonic()
    try:
        log_rows, log_tombs, log_meta = _write_leg(lb.changed_rows(), lb.tombstones,
                                                   out_dir / stamp / "log", shard_rows=shard_rows)
    except (dump.DeltaRefusal, CL.ChangeLogError) as e:
        raise ShadowRefusal(f"log leg: {e}") from e
    log_meta.update(seconds=round(time.monotonic() - t0, 1), segments=len(lb.segments),
                    no_ops_dropped=lb.noops, last_cut=lb.last_cut)
    legs["log"] = log_meta
    log_reasons = {t.id: t.reason for t in log_tombs}

    reasons = _Reasons(tombstone_reasons or {}, log_reasons)
    sb = dump.DeltaBuilder(base_manifest, bdir, ledger_base)
    t0 = time.monotonic()
    try:
        sc_rows, sc_tombs, sc_meta = _write_leg(
            sb.changed_rows(dump.scroll_rows(client, collection)),
            lambda: sb.tombstones(reasons), out_dir / stamp / "scroll", shard_rows=shard_rows)
    except dump.DeltaRefusal as e:
        raise ShadowRefusal(f"scroll leg: {e}") from e
    sc_meta["seconds"] = round(time.monotonic() - t0, 1)
    legs["scroll"] = sc_meta
    cut_after = CL.latest_cut(log_dir)

    ids = set(log_rows) | set(sc_rows) | {t.id for t in log_tombs} | {t.id for t in sc_tombs}
    sidecar, count = dump._base_sidecar(base_manifest, bdir, ledger_base)
    in_base = _base_ids(sidecar, count, ids)

    def side(rows: dict, tombs) -> Side:
        s: Side = {i: ("replace" if i in in_base else "add", fp, None) for i, fp in rows.items()}
        for t in tombs:
            s[t.id] = ("tombstone", None, t.reason)
        return s

    diff = compare(side(log_rows, log_tombs), side(sc_rows, sc_tombs),
                   compare_reasons=bool(tombstone_reasons))
    cut_moved = cut_before != cut_after
    equal = not any(diff.values())
    report = {
        "schema": "aw.shadow-compare.v1", "stamp": stamp, "collection": collection,
        "verdict": "EQUAL" if equal else "DIFFER",
        "base_manifest": str(base_manifest_path), "base_manifest_sha256": base_sha,
        "base_dump_id": base_manifest.get("dump_id"), "base_cut": after,
        "ledger_used": ledger is not None,
        "log_note": note, "segment_range": {"after": after, "last": lb.last_cut},
        "cut_before": cut_before, "cut_after": cut_after, "cut_moved_during_compare": cut_moved,
        "tombstone_reasons_independent": bool(tombstone_reasons),
        "scroll_reason_fallback_ids": sorted(reasons.fallback, key=pack.point_id_key)[:BOUND],
        "legs": legs, "diff": bound(diff), "diff_counts": {c: len(v) for c, v in diff.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    rpath = out_dir / f"shadow-compare.{stamp}.json"
    if not equal and cut_moved:
        report["verdict"] = "INCONCLUSIVE"
        report["why"] = ("the log gained a segment while the scroll leg ran; a write that landed "
                         "mid-compare shows as only-in-scroll. Re-run.")
        rpath.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        raise ShadowRefusal(report["why"])
    rpath.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return (EXIT_EQUAL if equal else EXIT_DIFFER), report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="cyt dump shadow-compare",
        description="ruling 68 (2) shadow period: build the delta from the change log AND by scroll "
                    "from the same base, compare them (spec delta-change-log.md 3.7). Read-only on "
                    "the collection; run it against the TWIN, never prod.",
        epilog="exit codes:\n  0  EQUAL\n  1  DIFFER (report names the ids; do not ship a log delta)\n"
               "  2  REFUSED (missing segments, no cut, log refusal, cut moved mid-run): no verdict",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--qdrant-url", default=dump.DEFAULT_TWIN_QDRANT_URL)
    ap.add_argument("--i-know-this-is-not-the-twin", action="store_true")
    ap.add_argument("--delta-base-manifest", required=True)
    ap.add_argument("--delta-base-dir", default=None)
    ap.add_argument("--from-change-log", default=None, metavar="DIR",
                    help="default: the configured change_log_dir")
    ap.add_argument("--tombstone-reasons", default=None,
                    help="JSON {id: reason}: gives the scroll leg independent reasons, so reasons are compared")
    ap.add_argument("--out", required=True, help="report + scratch legs directory")
    ap.add_argument("--shard-rows", type=int, default=50_000)
    args = ap.parse_args(argv)
    dump.assert_qdrant_url_is_not_prod(
        args.qdrant_url, i_know_this_is_not_the_twin=args.i_know_this_is_not_the_twin)
    from cytir.core.qdrant import qdrant_client
    from cytir.index import change_log as CL
    log_dir = args.from_change_log
    if log_dir is None:
        cfg = CL.configured_dir()
        log_dir = str(cfg) if cfg else None
    if not log_dir:
        print("REFUSED: no change-log directory (--from-change-log or the change_log_dir setting)",
              file=sys.stderr)
        return EXIT_REFUSED
    reasons = json.loads(pathlib.Path(args.tombstone_reasons).read_text()) if args.tombstone_reasons else None
    from cytir.distribute.fingerprint_ledger import LedgerError
    try:
        rc, rep = run(client=qdrant_client(args.qdrant_url, timeout=300), collection=args.collection,
                      base_manifest_path=args.delta_base_manifest, base_dir=args.delta_base_dir,
                      log_dir=log_dir, out_dir=pathlib.Path(args.out), ledger=dump._configured_ledger(),
                      tombstone_reasons=reasons, shard_rows=args.shard_rows)
    except (ShadowRefusal, LedgerError, CL.ChangeLogError, dump.DeltaRefusal) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    lg, sc = rep["legs"]["log"], rep["legs"]["scroll"]
    print(f"{rep['verdict']}: log {lg['rows']} row(s) / {lg['tombstones']} tombstone(s) in "
          f"{lg['seconds']}s; scroll {sc['rows']} / {sc['tombstones']} in {sc['seconds']}s")
    if rc:
        for c, v in rep["diff"].items():
            if v["count"]:
                print(f"  {c}: {v['count']} (first {len(v['first'])}: {v['first']})")
    print(f"report -> {pathlib.Path(args.out) / ('shadow-compare.' + rep['stamp'] + '.json')}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
