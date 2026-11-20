#!/usr/bin/env python3
# Moved from the retired rollback_restore_null_arm script (task 677, task 643 S5); the old path was removed (task 683 S9).
"""ADR-043 am.3/am.6 -- the `rollback-restore` null arm, as a procedure.

A null arm is the control that says whose a loss is: the pack's, or the
instrument's.  am.6 admits exactly two protocols for producing one and closes the
set, and it names the obvious cheap alternative and refuses it --

    "am.5's converged pair agrees per gold on the bar population by construction
    (tolerance 0 over legal and names), so it can attribute no tripwire loss to
    the instrument.  A null arm needs a state TRANSITION over identical content."

-- so a control that cannot move is decoration, and this module exists because
the honest version is a real sequence of writes that somebody has to run.

WHAT IT DOES, AND WHY IN THIS ORDER
-----------------------------------
Given a twin that has just been written with a pack, and the exact ids that were
written:

1. **verify the ids are all present.**  A rollback that "deletes" ids the
   collection does not hold is not a rollback; it is a no-op that produces a null
   arm of the state you already had, which then attributes nothing and looks fine.
2. **delete exactly those ids**, and nothing else -- a retrieve-checked id list,
   never a filter.  A filter is a predicate over payloads, and a payload field
   that a later rule epoch writes differently would silently widen the delete
   (memory `rule-vs-data-drift`).
3. **verify they are gone AND that points_count fell by exactly that many.**  Two
   independent readings, because either alone can be satisfied by the wrong thing:
   absent ids with an unchanged count means something else was written at the same
   time, and a correct count with a surviving id means the delete was partial.
4. **seal the membership digest** of the restored state.  This is the evidence
   that the null arm's state holds the BASELINE's content -- the claim the whole
   control rests on -- and it is a real scroll of the collection, not an assertion.
5. **cut the null arm under am.5 convergence.**  The state was produced BY A
   WRITE, so a single reading of it can land inside exactly the transient am.5
   exists for (task 609: 255 of 1,420 queries missing a top-10 url at
   `wait_green`+0, 82.8% back eight minutes later).  An unconverged null arm hands
   the tripwire rule a random set of "instrument" golds, which is worse than
   having none.
6. **re-apply the pack**, so the twin ends where the transaction needs it.
7. **write the basis receipt** binding all of it to THIS baseline arm (am.6 rule
   3: task 604's and task 613's baselines of the same content disagree on 32 bar
   golds and 5 h@1, and feeding one run's null arm to the other's comparison
   turns a FAIL into a PASS).

WHAT IT REFUSES
---------------
Everything that would let the procedure appear to succeed while producing a
control that cannot do its job: a written-ids file that is empty, duplicated, or
not a list; an id missing before the delete; an id surviving after it; a count
delta that disagrees with the id count; a non-CONVERGED null reading; and a
restore that does not put the collection back to the count it started from.

It defaults to a DRY RUN and writes nothing without `--apply`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
from typing import Callable, Iterable, Sequence

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]

from cytir.core import twin as TWIN  # noqa: E402
from cytir.index import layout as LAYOUT  # noqa: E402
from cytir.gates import content_gate_null_arm as NULL  # noqa: E402
from cytir.gates import convergence as CONV  # noqa: E402
from cytir.index.promote_sealed_pack import _api, already_present  # noqa: E402

SCHEMA = NULL.BASIS_SCHEMA
PROTOCOL = "rollback-restore"

#: Deletes go in id batches, like the presence check. Qdrant takes an id list
#: directly; the batch exists so one enormous request cannot time out halfway and
#: leave the caller unsure how much was applied.
DELETE_BATCH = 500


class NullArmProcedureError(RuntimeError):
    """A refusal. The procedure stops; it never proceeds with a weaker control."""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def digest_file(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------ the recorded ids
def load_written_ids(path: str | pathlib.Path) -> list[str]:
    """The ids `promote_sealed_pack --record-written-ids` wrote, checked.

    Every refusal here is a way the rollback could delete the wrong set while
    reporting success:

    * **not a list** -- a receipt or an object was passed instead of the id file,
      and iterating a dict yields its KEYS, which for a receipt are field names.
    * **empty** -- a rollback of nothing produces a null arm of the state that was
      already there. It attributes nothing and it looks fine.
    * **duplicated** -- the count the delete is checked against would be wrong, so
      a partial delete could still balance.
    * **not strings** -- point ids are compared as strings everywhere in this
      pipeline; a bare int here and a string there is how "absent" gets reported
      for a point that is present.
    """
    raw = json.loads(pathlib.Path(path).read_text())
    if not isinstance(raw, list):
        raise NullArmProcedureError(
            f"{path}: written-ids must be a JSON LIST of point ids, got "
            f"{type(raw).__name__}. Iterating anything else yields the wrong "
            "things silently -- a dict yields its keys.")
    if not raw:
        raise NullArmProcedureError(
            f"{path}: no ids. A rollback of nothing produces a null arm of the "
            "state that was already there: it attributes nothing, and it looks "
            "like it worked.")
    ids = [str(value) for value in raw]
    if any(not isinstance(value, (str, int)) for value in raw):
        raise NullArmProcedureError(
            f"{path}: every id must be a string or an int; got "
            f"{sorted({type(v).__name__ for v in raw})}")
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        raise NullArmProcedureError(
            f"{path}: {duplicates} duplicate id(s). The delete would be checked "
            "against a count that overstates what it removes, so a partial "
            "delete could still balance.")
    return ids


# --------------------------------------------------------------- the rollback
def assert_all_present(url: str, collection: str, ids: Sequence[str]) -> dict:
    """Every recorded id must BE in the collection before anything is deleted."""
    present = set(already_present(url, collection, list(ids)))
    missing = [pid for pid in ids if pid not in present]
    if missing:
        raise NullArmProcedureError(
            f"{len(missing)} of {len(ids)} recorded id(s) are NOT in "
            f"{collection} (e.g. {missing[:3]}). Deleting a set the collection "
            "does not hold is a no-op dressed as a rollback: the null arm would "
            "read the state you already had.")
    return {"checked": len(ids), "present": len(present)}


def delete_ids(url: str, collection: str, ids: Sequence[str], *,
               batch: int = DELETE_BATCH) -> int:
    """Delete exactly these ids. An id list, never a filter.

    A filter is a predicate over payloads, and a payload field a later rule epoch
    writes differently would widen the delete without changing the filter
    (memory `rule-vs-data-drift`). The ids are what was written; they are what is
    removed.
    """
    removed = 0
    for start in range(0, len(ids), batch):
        chunk = [str(pid) for pid in ids[start:start + batch]]
        _api(url, f"/collections/{collection}/points/delete?wait=true",
             {"points": chunk}, method="POST")
        removed += len(chunk)
    return removed


def assert_rolled_back(url: str, collection: str, ids: Sequence[str],
                       points_before: int) -> dict:
    """Two independent readings, because either alone can be satisfied wrongly.

    Absent ids with an unchanged count means something else was written while the
    delete ran. A correct count with a surviving id means the delete was partial
    and something else made up the difference. Only both together say the
    collection now holds the baseline's content.
    """
    survivors = already_present(url, collection, list(ids))
    after = LAYOUT.fingerprint(url, collection)
    points_after = after.get("points_count")
    problems = []
    if survivors:
        problems.append(f"{len(survivors)} id(s) survived the delete "
                        f"(e.g. {survivors[:3]})")
    expected = points_before - len(ids)
    if points_after != expected:
        problems.append(
            f"points_count is {points_after:,}, expected {expected:,} "
            f"({points_before:,} - {len(ids):,})")
    if problems:
        raise NullArmProcedureError(
            "the rollback did not restore the baseline's content: "
            + "; ".join(problems))
    return {"points_before": points_before, "points_after": points_after,
            "deleted": len(ids), "survivors": 0, "layout_after": after}


# ------------------------------------------------------------- the null basis
def build_basis(*, collection: str, points_count: int, membership_sha256: str,
                convergence: dict, baseline_identity: str,
                baseline_arm_sha256: str, evidence: list[dict],
                null_arm_path: str) -> dict:
    """The am.6 receipt, assembled from things that were MEASURED.

    Every field here is a reading this procedure took or an artifact it can cite.
    Nothing is asserted: `membership_sha256` comes from a real scroll,
    `convergence` from `gate_convergence.verdict`, and `adjudicates` from the
    baseline arm's own bytes.
    """
    doc = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "collection": collection,
        "points_count": points_count,
        "membership_sha256": membership_sha256,
        "evidence": evidence,
        "adjudicates": {"baseline_identity": baseline_identity,
                        "baseline_arm_sha256": baseline_arm_sha256},
        "convergence": convergence,
        "null_arm": null_arm_path,
        "sealed_at": now(),
        "sealed_by": "cytir/gates/rollback_restore_null_arm.py",
        "what_this_proves": (
            "The null arm was read on a state produced by DELETING exactly the "
            "ids this run had written and nothing else, verified both by id "
            "readback and by a points_count delta, whose membership digest is "
            "sealed above. Its deltas against the baseline are therefore the "
            "instrument's, not the pack's."),
    }
    # Validate against the ratified reader before anyone depends on it, with the
    # same arguments the comparator will pass. A receipt that this module writes
    # and only this module can read is not a receipt.
    NULL.validate_basis(doc, collection=collection, points_count=points_count,
                        baseline_identity=baseline_identity,
                        baseline_arm_sha256=baseline_arm_sha256)
    return doc


def converged_null_arm(cut_arm: Callable[[int], pathlib.Path], *,
                       log=print) -> dict:
    """am.5 over the restored state, through `gate_convergence`'s own protocol."""
    receipt = CONV.converged_arms(cut_arm, log=log)
    if receipt.get("verdict") != CONV.CONVERGED:
        raise NullArmProcedureError(
            f"the null arm did not converge ({receipt.get('verdict')!r}). A "
            "reading of a state that was still moving cannot attribute "
            "anything, and an unconverged null arm hands the tripwire rule a "
            "random set of 'instrument' golds.")
    return receipt


# ------------------------------------------------------------------- the CLI
def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--twin-url", required=True)
    ap.add_argument("--collection", default=TWIN.collection())
    ap.add_argument("--written-ids", required=True,
                    help="the file promote_sealed_pack --record-written-ids wrote")
    ap.add_argument("--baseline-arm", required=True,
                    help="the arm this null arm is the control FOR (am.6 rule 3)")
    ap.add_argument("--baseline-identity", required=True)
    ap.add_argument("--membership-receipt", required=True,
                    help="the sealed membership receipt of the ROLLED-BACK state "
                         "(the retired `seal_collection_membership.py` script)")
    ap.add_argument("--null-arm", required=True,
                    help="the converged null arm artifact")
    ap.add_argument("--convergence-receipt", required=True,
                    help="the aw.gate-convergence.v1 receipt for that arm")
    ap.add_argument("--out", required=True, type=pathlib.Path,
                    help="where to write the aw.content-gate-null-basis.v2 doc")
    args = ap.parse_args(list(argv) if argv is not None else None)
    # task 864 (review 869 S1): from an aw_e10 T0 to the flip only the drain writes the twin
    from cytir.index import twin_write_hold as _TWH
    _TWH.refuse_cli_write(args, tool='cytir/gates/rollback_restore_null_arm.py')

    ids = load_written_ids(args.written_ids)
    membership = json.loads(pathlib.Path(args.membership_receipt).read_text())
    convergence = json.loads(pathlib.Path(args.convergence_receipt).read_text())
    convergence = dict(convergence)
    convergence.setdefault("receipt_sha256",
                           digest_file(args.convergence_receipt))
    basis = build_basis(
        collection=args.collection,
        points_count=int(membership["points_count"]),
        membership_sha256=str(membership["membership_sha256"]),
        convergence=convergence,
        baseline_identity=args.baseline_identity,
        baseline_arm_sha256=digest_file(args.baseline_arm),
        null_arm_path=str(args.null_arm),
        evidence=[
            {"path": str(args.written_ids),
             "sha256": digest_file(args.written_ids),
             "what": f"the {len(ids):,} ids this run wrote and then deleted"},
            {"path": str(args.membership_receipt),
             "sha256": digest_file(args.membership_receipt),
             "what": "membership digest of the rolled-back state"},
            {"path": str(args.null_arm), "sha256": digest_file(args.null_arm),
             "what": "the converged null arm read on that state"},
            {"path": str(args.convergence_receipt),
             "sha256": digest_file(args.convergence_receipt),
             "what": "the am.5 convergence receipt for that reading"},
            {"path": str(args.baseline_arm),
             "sha256": digest_file(args.baseline_arm),
             "what": "the baseline arm this null arm adjudicates (am.6 rule 3)"},
        ])
    args.out.write_text(json.dumps(basis, indent=1, sort_keys=True) + "\n")
    print(f"[basis] {args.out}: {PROTOCOL}, {len(ids):,} ids, "
          f"{basis['points_count']:,} points, convergence="
          f"{convergence.get('verdict')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
