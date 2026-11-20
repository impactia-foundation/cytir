#!/usr/bin/env python3
# Moved from the retired `gate_convergence` module (task 677, task 643 S5); the old path is an alias shim.
"""A collection that has just been written is not scored once.

WHY THIS EXISTS (task 609, measured 2026-09-17 on the build-box twin `aw_e8`)
---------------------------------------------------------------------------
The ADR-043 content gate inside `apply_content_pack_shared` cuts its candidate
arm immediately after the applier's `wait_green` returns.  Measured on the
09-16 news pack, with the applier's OWN instrument, on the SAME collection:

    at wait_green + 0      core-639 h@1 206 -> 201, 18 core losses  -> FAIL
    +8 minutes             core-639 h@1 206 -> 206,  0 core losses  -> PASS

Nothing about the corpus differed between those two readings.  255 of 1,420
queries had lost a top-10 url in the first arm and **82.8% of the dropped urls
were back** in the second.  Two content packs 17x apart in size (14,819 news
rows, 883 HN rows) produced the SAME 24-gold loss list, which is what a
transient property of the write looks like and not what a property of the
content looks like.

WHY NOT A TELEMETRY PREDICATE
-----------------------------
the retired `qdrant_readiness` script (task 608) is the strict state predicate, and on
this transition it does not discriminate.  At the moment the gate scored, the
collection reported `status: green`, `optimizer_status: ok`,
`update_queue_length: 0` -- and it still carried two `plain` segments of 2,472
and 2,496 points with `num_indexed_vectors: 0`, which it ALSO still carried
eight minutes later when the answers had converged.  The predicate was false
at both readings; the answers differed between them.  So the thing to check is
the ANSWERS.

THE RULE (ADR-043 amendment A5)
------------------------------
Score the collection TWICE, separated by a settle interval, and require the two
readings to agree per gold on the bar population.  Disagreement is
NOT-CONVERGED: the verdict is unknown, never a FAIL.  A pack refused by a gate
that had not converged has been refused by the clock.

`converged_arms` is that rule as a protocol: assert the readiness predicate,
cut an arm, settle, cut another, compare -- and on disagreement settle and
re-score again, up to the bound in `data/content-gate-convergence.v1.json`.
When the bound runs out the run is UNADJUDICATED and stops; it never becomes a
FAIL.  Only a CONVERGED pair yields an ADR-043 verdict, computed on the LATER
of the two readings.  The settle interval and the bound are measured defaults
in that contract, with the timelines that justify them, not constants here.

Fails closed, as a guard must (verification-doctrine 12): a missing arm, an
unreadable arm, an arm with query-vector misses, or an arm scoring a different
population is `UNKNOWN` (exit 2), never a pass.  `main()` carries the verdict
in an exit code.

    python3 -m pipeline.gate_convergence --arm FIRST.json --arm SECOND.json
"""
from __future__ import annotations

from cytir.core.paths import data_path

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
from cytir.index import layout as collection_layout  # noqa: E402

#: ADR-043's gold core plus the flood tripwire -- the population the content
#: gate's verdict is computed on. Convergence is judged on exactly the golds the
#: verdict rests on, never on a wider or narrower set.
BAR_SETS = ("authoritative", "foundation", "legal", "names", "rel_vs_trust")

#: The measured defaults -- the settle interval, the round bound and the
#: readiness disposition -- live in data, with the timelines that justify them,
#: never as constants here (ADR-043 am.5).
CONTRACT_PATH = data_path("content-gate-convergence.v1.json")

CONVERGED = "CONVERGED"
NOT_CONVERGED = "NOT-CONVERGED"
UNKNOWN = "UNKNOWN"
UNADJUDICATED = "UNADJUDICATED"


class ConvergenceUnknown(RuntimeError):
    """The arms cannot be compared. Never a pass, never a fail."""


class ConvergenceUnadjudicated(RuntimeError):
    """The bounded rounds ran out with the readings still disagreeing.

    Not a FAIL: the state was still moving every time it was read, so nothing
    was measured about the pack.  The caller stops and reports; it does not
    ship, and it does not record a verdict about the content either.
    """

    def __init__(self, message: str, receipt: dict):
        super().__init__(message)
        self.receipt = receipt


def _per_q(arm: dict, name: str) -> dict:
    block = arm.get("sets", {}).get(name)
    if block is None:
        raise ConvergenceUnknown(f"arm does not score the set {name!r}")
    per_q = block.get("per_q")
    if not isinstance(per_q, dict):
        raise ConvergenceUnknown(f"{name}: arm has no per-query block")
    return per_q


def load_arm(path) -> dict:
    """Read an arm, refusing anything that is not a reading of one state."""
    path = pathlib.Path(path)
    try:
        arm = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ConvergenceUnknown(f"{path}: absent") from exc
    except (OSError, ValueError) as exc:
        raise ConvergenceUnknown(f"{path}: unreadable ({exc})") from exc
    if not isinstance(arm, dict) or "sets" not in arm:
        raise ConvergenceUnknown(f"{path}: not an arm (no `sets`)")
    # An arm with cache misses embedded a different query set; an arm of errors
    # is not an arm at all (task 600 wrote one and handed it on).
    if arm.get("cache_miss"):
        raise ConvergenceUnknown(
            f"{path}: cache_miss={arm['cache_miss']} -- the query vectors are "
            "not sealed, so this is not a reading of this state")
    return arm


def comparable_ids(arm: dict, name: str) -> list[str]:
    """Query ids this arm reports a RANK for -- the roster to compare on.

    Derived from the artifact being judged, never a hand-list and never a count
    copied from elsewhere (memory: census-roster-from-manifest).

    It keys on `rank` and NOT on gold metadata, and that is load-bearing.  The
    first version of this module selected on `gold_urls`/`gold_domains`; the
    Task370 arms the content gate actually produces carry only
    `{rank, urls, trace}`, so the population came back EMPTY, nothing was
    compared, and the guard returned CONVERGED for every pair it was handed.
    A guard that cannot fail is decoration (verification-doctrine 12), and that
    one could not fail on the exact artifacts it exists to judge.  `rank` is
    what the verdict is computed from, so `rank` is what convergence is judged
    on; a query with no rank key is not comparable, and an empty population is
    refused loudly below rather than passing quietly.
    """
    return sorted(qid for qid, q in _per_q(arm, name).items() if "rank" in q)


def disagreements(first: dict, second: dict,
                  sets: tuple = BAR_SETS) -> list[dict]:
    """Every gold whose rank differs between the two readings."""
    out = []
    total = 0
    for name in sets:
        a, b = _per_q(first, name), _per_q(second, name)
        ids_a, ids_b = comparable_ids(first, name), comparable_ids(second, name)
        if set(ids_a) != set(ids_b):
            missing = sorted(set(ids_a) ^ set(ids_b))[:5]
            raise ConvergenceUnknown(
                f"{name}: the two arms score different populations "
                f"({len(ids_a)} vs {len(ids_b)}, e.g. {missing}) -- they are "
                "not two readings of one state")
        if not ids_a:
            raise ConvergenceUnknown(
                f"{name}: neither arm reports a rank for any query -- there is "
                "nothing to compare, which is UNKNOWN and never agreement")
        total += len(ids_a)
        for qid in ids_a:
            ra, rb = a[qid].get("rank"), b[qid].get("rank")
            if ra != rb:
                out.append({"set": name, "qid": qid, "first": ra,
                            "second": rb})
    if not total:
        raise ConvergenceUnknown("no query was comparable across the two arms")
    return out


def verdict(first_path, second_path, tolerance: int = 0) -> dict:
    """CONVERGED / NOT-CONVERGED, or raise ConvergenceUnknown.

    `tolerance` is the number of disagreeing golds the caller will accept. It
    defaults to ZERO and a caller may not raise it to excuse a run: this is a
    ranking-class question (ADR-043's per-gold bar), and two readings of ONE
    unchanged state that disagree on even one gold have told you the state is
    still moving.
    """
    if tolerance < 0:
        raise ConvergenceUnknown("tolerance must not be negative")
    first, second = load_arm(first_path), load_arm(second_path)
    moved = disagreements(first, second)
    return {
        "schema": "aw.gate-convergence.v1",
        "arms": [str(first_path), str(second_path)],
        "sets": list(BAR_SETS),
        "queries_compared": sum(len(comparable_ids(first, n))
                                for n in BAR_SETS),
        "disagreements": moved,
        "tolerance": tolerance,
        "verdict": CONVERGED if len(moved) <= tolerance else NOT_CONVERGED,
    }


def load_contract(path=None) -> dict:
    """The measured defaults, read from data, refusing anything soft.

    Fails closed on every way the file can fail to be a contract: absent,
    unreadable, wrong schema, a settle interval or round bound out of range, a
    bar-set list that disagrees with this module's, or a non-zero tolerance --
    which is the one value a caller might be tempted to raise to get a run
    through, and which this refuses rather than honours.
    """
    path = pathlib.Path(CONTRACT_PATH if path is None else path)
    try:
        doc = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ConvergenceUnknown(f"{path}: absent") from exc
    except (OSError, ValueError) as exc:
        raise ConvergenceUnknown(f"{path}: unreadable ({exc})") from exc
    if not isinstance(doc, dict) or doc.get("schema") != "aw.content-gate-convergence.v1":
        raise ConvergenceUnknown(f"{path}: not an aw.content-gate-convergence.v1 contract")
    settle, rounds = doc.get("settle_seconds"), doc.get("max_rounds")
    if not isinstance(settle, int) or isinstance(settle, bool) or not 0 < settle <= 3600:
        raise ConvergenceUnknown(
            f"{path}: settle_seconds={settle!r} is not a whole number of "
            "seconds in (0, 3600]")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or not 1 <= rounds <= 10:
        raise ConvergenceUnknown(f"{path}: max_rounds={rounds!r} is not in [1, 10]")
    if doc.get("tolerance") != 0:
        raise ConvergenceUnknown(
            f"{path}: tolerance={doc.get('tolerance')!r}. Convergence is a "
            "per-gold agreement between two readings of ONE state; a tolerance "
            "here would be a waiver, and a waiver is Tim's, by PR.")
    if tuple(doc.get("bar_sets") or ()) != BAR_SETS:
        raise ConvergenceUnknown(
            f"{path}: bar_sets {doc.get('bar_sets')!r} disagree with the "
            f"population the verdict is computed on {list(BAR_SETS)}")
    return doc


#: am.9 (Tim, 2026-09-21, pending its own PR -- OFF unless
#: `AW_GATE_SETTLE=layout` is exported, default unset): two readings 60s
#: apart are convergent when the optimizer log is quiet between them AND the
#: #218 layout fingerprint (`pipeline.collection_layout`) is unchanged --
#: task 627's mechanism finding is that BM25-leg drift on unchanged rows is
#: driven BY layout movement (an optimizer pass rewriting a segment's sparse
#: header), so a provably quiet layout is mechanism-justified evidence of a
#: settled reading, not merely a shorter guess at the same 480s wait. It is
#: an ADDITIONAL route to CONVERGED, layered on top of the am.5 scored-
#: agreement rule, never a replacement: every round still computes the
#: classic per-gold score verdict too, and CONVERGED is declared the moment
#: EITHER rule agrees. Falls back to the full `settle_seconds` wait whenever
#: the optimizer ran, a fingerprint could not be read, or the flag is off.
AM9_ENV = "AW_GATE_SETTLE"
AM9_SETTLE_SECONDS = 60


def _am9_layout_quiet(before: dict | None, after: dict | None) -> tuple[bool, str]:
    """(quiet, reason) -- never raises; an unreadable fingerprint just fails
    the check (falls back to the classic rule) rather than aborting the run.
    One home: `collection_layout.quiescent` is the same predicate the daily
    door's prod-baseline reuse decision uses (task626 catch-up item 1)."""
    return collection_layout.quiescent(before, after)


def converged_arms(cut_arm, *, contract=None, readiness=None, sleep=time.sleep,
                   log=print, layout_fingerprint=None) -> dict:
    """Cut arms until two consecutive readings agree, or refuse.

    `cut_arm(round_index)` cuts ONE arm of the state as it is now and returns
    its path; `readiness()` returns one `qdrant_readiness.snapshot` and may
    raise, in which case the caller aborts (a readiness reading that cannot be
    TAKEN is never clean).  A `not ready` snapshot is recorded and does not by
    itself refuse: task 609 measured the predicate false at both the failing
    and the passing reading of this very transition, so it is evidence, not the
    discriminator.

    `layout_fingerprint()` (optional) returns one `collection_layout.build`-
    shaped dict for the state under test and is only consulted when
    `AW_GATE_SETTLE=layout` is exported -- see am.9 above. Any exception it
    raises is treated as "fingerprint unreadable" (fall back), never as an
    abort: a layout read that cannot be taken degrades this round to the
    classic rule, it does not fail the whole protocol.

    Returns the receipt of the CONVERGED pair -- whose `arm` is the LATER of
    the two, the most settled reading of the state, and the one the ADR-043
    comparison is then computed on.  Raises `ConvergenceUnadjudicated` when the
    bounded rounds run out, and `ConvergenceUnknown` when a pair cannot be
    compared at all.  It cannot return a FAIL, because it does not compute one.
    """
    contract = load_contract() if contract is None else contract
    settle, bound = contract["settle_seconds"], contract["max_rounds"]
    am9_on = os.environ.get(AM9_ENV) == "layout" and layout_fingerprint is not None
    readiness_snapshot = None
    if readiness is not None:
        readiness_snapshot = readiness()
        problems = (readiness_snapshot or {}).get("problems") or []
        log(f"[convergence] readiness: ready={(readiness_snapshot or {}).get('ready')}"
            + (f" problems={problems}" if problems else ""))

    def _fp():
        if not am9_on:
            return None
        try:
            return layout_fingerprint()
        except Exception as exc:  # noqa: BLE001 -- degrade, never abort
            log(f"[convergence] am.9 layout read failed ({exc}); this round "
                "falls back to the classic rule")
            return None

    arms = [pathlib.Path(cut_arm(0))]
    rounds = []
    fp_before = _fp()
    for index in range(1, bound + 1):
        wait = AM9_SETTLE_SECONDS if (am9_on and fp_before is not None) else settle
        log(f"[convergence] round {index}/{bound}: settling {wait}s before "
            "the next reading of the same state"
            + (" (am.9 layout fast path armed)"
               if wait == AM9_SETTLE_SECONDS else ""))
        sleep(wait)
        arms.append(pathlib.Path(cut_arm(index)))
        fp_after = _fp()
        am9_quiet, am9_reason = (
            _am9_layout_quiet(fp_before, fp_after) if am9_on else (False, "am.9 off"))
        result = verdict(arms[-2], arms[-1])
        result["am9"] = {"on": am9_on, "settle_seconds": wait,
                         "layout_quiet": am9_quiet, "reason": am9_reason}
        rounds.append(result)
        log(f"[convergence] round {index}/{bound}: {result['verdict']} -- "
            f"{len(result['disagreements'])} of {result['queries_compared']} "
            "queries differ between the two readings"
            + (f" | am.9: {am9_reason}" if am9_on else ""))
        if result["verdict"] == CONVERGED or am9_quiet:
            return {"schema": "aw.gate-convergence-protocol.v1",
                    "verdict": CONVERGED,
                    "arm": str(arms[-1]),
                    "arms": [str(item) for item in arms],
                    "rounds": index, "max_rounds": bound,
                    "settle_seconds": settle,
                    "readiness": readiness_snapshot,
                    "am9_layout_quiet": am9_quiet,
                    "receipts": rounds}
        fp_before = fp_after
    receipt = {"schema": "aw.gate-convergence-protocol.v1",
               "verdict": UNADJUDICATED,
               "arms": [str(item) for item in arms],
               "rounds": bound, "max_rounds": bound,
               "settle_seconds": settle,
               "readiness": readiness_snapshot,
               "receipts": rounds}
    raise ConvergenceUnadjudicated(
        f"the collection was still moving after {bound} readings "
        f"{settle}s apart (last pair: {len(rounds[-1]['disagreements'])} of "
        f"{rounds[-1]['queries_compared']} queries differ). Nothing was "
        "measured about this pack: UNADJUDICATED, which is not a FAIL.",
        receipt)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", action="append", required=True, metavar="ARM.json",
                    help="give it TWICE: the two readings of the same state")
    ap.add_argument("--tolerance", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args(argv)

    if len(args.arm) != 2:
        print(f"{UNKNOWN}: need exactly two --arm values, got {len(args.arm)}")
        return 2
    try:
        result = verdict(args.arm[0], args.arm[1], args.tolerance)
    except ConvergenceUnknown as exc:
        print(f"{UNKNOWN}: {exc}")
        return 2
    if args.out:
        args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"{result['verdict']}: {len(result['disagreements'])} of "
          f"{result['queries_compared']} queries differ between the readings "
          f"(tolerance {result['tolerance']})")
    for row in result["disagreements"][:25]:
        print(f"  MOVED {row['set']}/{row['qid']} "
              f"{row['first']} -> {row['second']}")
    if len(result["disagreements"]) > 25:
        print(f"  ... and {len(result['disagreements']) - 25} more")
    return 0 if result["verdict"] == CONVERGED else 1


if __name__ == "__main__":
    sys.exit(main())
