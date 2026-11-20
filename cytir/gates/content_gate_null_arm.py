#!/usr/bin/env python3
# Moved from the retired `content_gate_null_arm` module (task 677, task 643 S5); the old path was removed (task 683 S9).
"""ADR-043 amendments 3 and 6 -- the content gate subtracts its own null arm,
and a LEGAL/NAMES loss must survive it.

The problem this module exists for, in one measurement (task 603). Two
readings of the SAME twin holding the SAME content -- byte-equal membership
digest `5d4307e1...`, 33,723,480 points on both sides, one written and rolled
back between them -- disagree about **34 golds**: 19 rank lower, 15 rank
higher, and the 663-gold headline moves 212 -> 213. Nothing was added,
nothing was deleted, and no query, cache, weight or flag differed. The
instrument moves on its own, because an apply rewrites segments and the dense
leg is an approximate search over them.

Against that background the 09-16 news pack's 40 "strict per-gold losses" are
not 40 losses. 19 of them are golds the instrument moved by itself, and the
`-5` net sits next to a `+1` the instrument produced out of nothing. A gate
that reports the raw list as the pack's cost is reporting its own noise as a
defect -- and the record shows exactly that happening: task 600 sec.7 read
the list as "14,819 fresh news chunks out-compete authoritative law", which
task 603 falsified (no news row stands above any of the 40).

So: **a loss is the pack's only when the instrument did not produce it, and a
net delta is the pack's only after the instrument's own is taken off.** That
is the whole rule. Three properties keep it from becoming a licence:

1. **It never turns a PASS into a FAIL, and never a FAIL into a PASS on its
   own numbers.** `metric_ok` is `delta_raw - delta_null >= -abs(delta_null)`,
   which for a quiet instrument (`delta_null == 0`) is exactly ADR-043's
   `delta_raw >= 0`, and for any instrument is never STRICTER than that bar.
   A pack that passes the unamended gate passes this one -- proven in
   `test_the_correction_is_one_sided`, which is why a run whose raw verdict is
   PASS needs no null arm at all.
2. **The band is measured, never a constant.** `abs(delta_null)` is whatever
   this run's null arm did. Today it is 1. It is not written down anywhere as
   1, and a quieter instrument tomorrow tightens the bar by itself.
3. **Nothing is hidden and nothing is waived.** Every raw loss is still
   printed. The instrument-attributable ones are printed under their own
   label with the null arm's rank beside them, so a human can disagree with
   the classification of any single row. The LEGAL and NAMES tripwires are
   computed on the RAW list and are never corrected -- a destroyed legal gold
   is a defect whatever the instrument was doing. (A6, below, keeps the raw
   ranks and the zero tolerance and adds one question before the refusal:
   whose loss is it?)

ADR-043 AMENDMENT A6 -- THE TRIPWIRES ARE ADJUDICATED TOO
---------------------------------------------------------
A3 rule 5 left the LEGAL and NAMES tripwires on the RAW list, uncorrected, and
that was right while nothing showed the instrument moving one.  Task 613 showed
it.  The 09-16 news pack, read through a CONVERGED gate (am.5), lost exactly
two golds -- `names/n02` 3->4 and `slice_b/sb-ch-holidays` 2->5 -- and neither
is displaced by a pack row: for n02 the only change in either leg's pool is one
UZH person page that is not in the pack rising to dense position 7.  Task 609's
settled reading of the SAME pack moved the same gold **4->3**, a gain.  Two
converged readings, opposite directions, same pack: a zero-tolerance bar over a
gold that flips is sampling a coin, not measuring a pack.

So A6: a tripwire loss still refuses at ZERO tolerance, and it must be
ATTRIBUTABLE first.  `tripwire_verdict` is that rule.  Three things keep it from
becoming the waiver A3 rule 5 was protecting against:

* **An unadjudicated tripwire loss still refuses.**  No null arm means no
  attribution, and no attribution means the pack does not ship.  A6 buys a
  measurement, never a tolerance.
* **The null arm must be the same run's.**  `assert_basis_adjudicates` refuses
  a null arm whose basis does not name THIS baseline arm.  Measured, on the
  arms in the record: task 604's and task 613's readings of the same content
  (same collection at the same count, same cache, retrieval, qhead, pinned env
  and probe fixture) disagree on **32 bar golds and 5 h@1**, and both of 613's
  lost golds sit in that set -- so an operator free to reach for another run's
  null arm could excuse any loss in the record.
* **The null arm is itself read under am.5.**  The restore that produces its
  state is a write, so its own reading can sit in task 609's transient.  The
  basis carries the convergence receipt that says it did not.

What this module deliberately does NOT do: decide whether the instrument
SHOULD be this noisy. It should not be -- a fused order that inverts under a
0.07% perturbation of one leg is the real defect task 603 found, and
tie-robust fusion is the lever that fixes it. Until that exists, measuring
the noise is better than charging it to whatever happens to be applied that
day.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

#: The basis document's schema. It is the receipt that says the null arm's two
#: states really did hold the same content -- this module can check that the
#: claim was MADE, named and hashed, and that it is about the right collection
#: at the right count; it cannot re-hash 33.7M points from inside a comparator,
#: and does not pretend to (the digests are produced by
#: `seal_collection_membership.py` and the restore tool, off-repo).
#:
#: **v2 (ADR-043 am.6)** adds two required blocks, because am.6 lets the null
#: arm adjudicate the LEGAL/NAMES tripwires and that raises what a null arm has
#: to prove about itself: `adjudicates` (the baseline arm this control was cut
#: for, by identity and sha256 -- no cross-run null arms) and `convergence`
#: (the null arm is itself a CONVERGED reading under am.5, because the restore
#: that produces its state is a write like any other).
BASIS_SCHEMA = "aw.content-gate-null-basis.v2"

#: v1 receipts predate am.6 and are refused rather than reinterpreted: they
#: name no baseline and carry no convergence receipt, so neither am.6 rule can
#: be checked against them.
BASIS_SCHEMA_SUPERSEDED = {
    "aw.content-gate-null-basis.v1":
        "ADR-043 am.6 requires `adjudicates` (the baseline arm this null arm "
        "was cut for) and `convergence` (the null arm read under am.5); a v1 "
        "receipt carries neither, so re-seal the basis as v2",
}

BASIS_REQUIRED = ("schema", "protocol", "collection", "points_count",
                  "membership_sha256", "evidence", "adjudicates",
                  "convergence")

#: A CLOSED set. A null arm is only a control if its two states are known to
#: hold the same content, and only these two transitions have been shown to
#: produce such a pair on this system. Adding a third is an ADR change, not a
#: string a run may pass in -- the standard a run is judged by may not be
#: softened from inside that run (verification-doctrine 12).
#:
#: `rollback-restore` is the one task 603 measured and the one every content
#: pack already pays for: the transaction writes the twin, the gate refuses,
#: the ratified restore tool puts the twin back to production's state, and the
#: null arm is the same gate arm cut once more on the restored state.
#: `neutral-rewrite` is provided for a calibration taken WITHOUT a pack --
#: existing rows re-upserted byte-identically, so the segments churn and the
#: content does not. It has never been run here; a first use must measure it,
#: not assume it behaves like the other one.
BASIS_PROTOCOLS = ("rollback-restore", "neutral-rewrite")

#: Named, refused, and told WHY -- because the obvious cheap null pair is the
#: one am.5 already cuts, and it cannot work. am.5's convergence protocol
#: requires the two readings of a state to agree PER GOLD on
#: `gate_convergence.BAR_SETS`, which contains `legal` and `names`; so a
#: converged pair moves no tripwire gold by construction, and a tripwire rule
#: judged against it could never classify anything as the instrument's. It
#: would be a control that cannot move -- decoration, the exact defect the
#: closed protocol list above exists to prevent. Measured on task 613's own
#: pairs: preapply-c/d and candidate-r0/r1, 0 of 691 disagreements each.
REFUSED_PROTOCOLS = {
    "two-readings-converged":
        "am.5's converged pair agrees per gold on the bar population by "
        "construction (tolerance 0 over legal and names), so it can attribute "
        "no tripwire loss to the instrument. A null arm needs a state "
        "TRANSITION over identical content",
}

#: Metadata every arm in the comparison must agree on. This is the same set
#: the comparator already requires of its baseline/candidate pair: the null
#: arm is a third arm of the SAME experiment, not a reading from a neighbouring
#: one, so a null arm cut under different weights, a different cache or a
#: different release is refused rather than averaged in.
BOUND_META = ("cache_sha256", "cache_vectors",
              "qvec_query_source_manifest_sha256", "qvec_query_source",
              "retrieval_sha256", "qhead_sha256", "pinned_env",
              "probe_fixture_sha256")


class NullArmError(ValueError):
    """The null arm is not admissible as a control."""


def rank_key(rank: int | None) -> int:
    """Absent from the top 10 sorts below every present rank."""
    return 10 ** 9 if rank is None else rank


def moved(before_sets: Mapping, after_sets: Mapping) -> set[str]:
    """`set/qid` of every query whose rank differs, in EITHER direction.

    Direction-agnostic on purpose. A gold the instrument can push down is a
    gold the instrument can push up; both readings prove the same thing --
    that this query sits on a tie the instrument decides. Counting only the
    downward half would let a query that the null arm moved up by two and the
    pack moved down by one be charged to the pack.
    """
    out: set[str] = set()
    for name, before in before_sets.items():
        after = after_sets.get(name) or {}
        for qid, row in (before.get("per_q") or {}).items():
            other = ((after.get("per_q") or {}).get(qid) or {})
            if rank_key(other.get("rank")) != rank_key(row.get("rank")):
                out.add(f"{name}/{qid}")
    return out


def split_losses(raw_losses: Sequence[tuple[str, object, object]],
                 null_moved: Iterable[str]) -> tuple[list, list]:
    """Partition the raw loss rows into (attributable, instrument).

    `raw_losses` rows are `(key, before_rank, after_rank)`. The rule is the
    whole amendment: a loss on a query the null arm also moved is not
    attributable to the pack.
    """
    null_moved = set(null_moved)
    attributable = [row for row in raw_losses if row[0] not in null_moved]
    instrument = [row for row in raw_losses if row[0] in null_moved]
    return attributable, instrument


def metric(delta_raw: int, delta_null: int | None) -> dict:
    """One metric's null-corrected reading and whether it clears the bar.

    `attributable = delta_raw - delta_null` is the pack's own net; `band =
    abs(delta_null)` is how far the instrument demonstrated it can move this
    metric with no content change at all, and differences inside it are not
    resolvable by this instrument on this run.

    With no null arm (`delta_null is None`) the reading is the unamended
    ADR-043 one: bar 0, band 0, `attributable = delta_raw`.
    """
    if delta_null is None:
        return {"delta_raw": delta_raw, "delta_null": None,
                "attributable": delta_raw, "band": 0,
                "ok": delta_raw >= 0, "adjudicated": False}
    band = abs(delta_null)
    attributable = delta_raw - delta_null
    return {"delta_raw": delta_raw, "delta_null": delta_null,
            "attributable": attributable, "band": band,
            "ok": attributable >= -band, "adjudicated": True}


def validate_basis(doc: object, *, collection: str, points_count: int,
                   baseline_identity: str | None = None,
                   baseline_arm_sha256: str | None = None) -> dict:
    """Refuse a null arm whose content-equality claim is absent or unbound.

    `baseline_identity` / `baseline_arm_sha256` are ADR-043 am.6 rule 3: given
    them, the basis must name THIS baseline arm as the one it adjudicates. They
    are optional in the signature and mandatory in practice -- both callers
    pass them (`compare_corpus_arms.py`, `news_daily_gate.judge`) -- because a
    receipt can be validated for shape before the arms are known.
    """
    if not isinstance(doc, Mapping):
        raise NullArmError("null basis is not an object")
    schema = doc.get("schema") if isinstance(doc, Mapping) else None
    if schema in BASIS_SCHEMA_SUPERSEDED:
        raise NullArmError(f"null basis schema {schema!r} is superseded: "
                           f"{BASIS_SCHEMA_SUPERSEDED[schema]}")
    missing = [key for key in BASIS_REQUIRED if key not in doc]
    if missing:
        raise NullArmError(f"null basis missing {missing}")
    if doc["schema"] != BASIS_SCHEMA:
        raise NullArmError(f"null basis schema {doc['schema']!r} "
                           f"is not {BASIS_SCHEMA}")
    if doc["protocol"] in REFUSED_PROTOCOLS:
        raise NullArmError(
            f"null basis protocol {doc['protocol']!r} is refused (ADR-043 "
            f"am.6): {REFUSED_PROTOCOLS[doc['protocol']]}")
    if doc["protocol"] not in BASIS_PROTOCOLS:
        raise NullArmError(
            f"null basis protocol {doc['protocol']!r} is not one this "
            f"amendment admits {list(BASIS_PROTOCOLS)}")
    if doc["collection"] != collection or doc["points_count"] != points_count:
        raise NullArmError(
            f"null basis is about {doc['collection']}@{doc['points_count']}, "
            f"the arms are {collection}@{points_count}")
    digest = str(doc["membership_sha256"])
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise NullArmError("null basis membership_sha256 is not a sha256")
    evidence = doc["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise NullArmError("null basis cites no evidence")
    for item in evidence:
        if not isinstance(item, Mapping) or {"path", "sha256", "what"} - set(item):
            raise NullArmError("null basis evidence row needs path/sha256/what")
    assert_basis_converged(doc)
    assert_basis_adjudicates(doc, identity=baseline_identity,
                             arm_sha256=baseline_arm_sha256)
    return dict(doc)


def assert_basis_converged(doc: Mapping) -> None:
    """ADR-043 am.6 rule 2: the null arm is itself a CONVERGED reading.

    The state a `rollback-restore` null arm is read on was produced by a WRITE
    -- the restore deletes the pack's rows and re-inserts the superseded ones
    -- so a single reading of it can sit inside exactly the transient am.5
    exists for (task 609: 255 of 1,420 queries missing a top-10 url at
    `wait_green`+0, 82.8% of them back eight minutes later). An unconverged
    null arm would then hand the tripwire rule a random set of "instrument"
    golds, which is worse than having no null arm at all.
    """
    receipt = doc.get("convergence")
    if not isinstance(receipt, Mapping):
        raise NullArmError(
            "null basis carries no convergence receipt -- am.6 rule 2: the "
            "null arm is read under am.5 like any other arm")
    if receipt.get("verdict") != "CONVERGED":
        raise NullArmError(
            f"null arm convergence verdict {receipt.get('verdict')!r} is not "
            "CONVERGED -- a reading of a state that was still moving cannot "
            "attribute anything")
    digest = str(receipt.get("receipt_sha256"))
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise NullArmError("null arm convergence receipt_sha256 is not a "
                           "sha256 -- the receipt must be citable")
    compared = receipt.get("queries_compared")
    if not isinstance(compared, int) or isinstance(compared, bool) or compared <= 0:
        raise NullArmError("null arm convergence receipt compares no queries")


def assert_basis_adjudicates(doc: Mapping, *, identity: str | None,
                             arm_sha256: str | None) -> None:
    """ADR-043 am.6 rule 3: this null arm was cut for THIS baseline.

    Without it the bound-metadata checks in `assert_arm_bound` are satisfied by
    any arm of the same release on the same collection at the same count -- and
    those exist from other runs, days apart. Measured: task 604's and task
    613's baseline readings of the same content, every bound field equal,
    disagree on 32 bar golds and 5 h@1; feeding 604's null arm to 613's
    comparison turns its FAIL into a PASS. The null arm is a control for the
    run it was cut in, and a citation is what makes that checkable.
    """
    claim = doc.get("adjudicates")
    if not isinstance(claim, Mapping):
        raise NullArmError(
            "null basis does not say which baseline arm it adjudicates "
            "(am.6 rule 3: `adjudicates.baseline_identity` + "
            "`baseline_arm_sha256`)")
    missing = {"baseline_identity", "baseline_arm_sha256"} - set(claim)
    if missing:
        raise NullArmError(f"null basis `adjudicates` missing {sorted(missing)}")
    if identity is not None and claim["baseline_identity"] != identity:
        raise NullArmError(
            f"null basis adjudicates {claim['baseline_identity']!r}, the "
            f"baseline arm is {identity!r} -- a null arm from another run is "
            "not this run's control (am.6 rule 3)")
    if arm_sha256 is not None and claim["baseline_arm_sha256"] != arm_sha256:
        raise NullArmError(
            f"null basis adjudicates baseline arm "
            f"{str(claim['baseline_arm_sha256'])[:8]}…, this comparison's "
            f"baseline is {str(arm_sha256)[:8]}… (am.6 rule 3)")


def tripwire_verdict(trip_losses: Sequence[tuple[str, object, object]],
                     null_moved: Iterable[str], *, adjudicated: bool) -> dict:
    """ADR-043 am.6 rule 1: which tripwire losses refuse, and why.

    Zero tolerance is unchanged -- ONE attributable tripwire loss refuses. What
    changes is that a tripwire gold the null arm also moved, with no content
    difference at all, is the instrument's and is reported as such. And a
    tripwire loss with NO null arm is `unadjudicated`: it still refuses,
    because an unmeasured attribution is not an excuse.
    """
    attributable, instrument = split_losses(trip_losses, null_moved)
    unadjudicated = bool(trip_losses) and not adjudicated
    return {"raw": list(trip_losses), "attributable": attributable,
            "instrument": instrument, "unadjudicated": unadjudicated,
            "refuses": bool(attributable) or unadjudicated}


def assert_arm_bound(baseline_meta: Mapping, null_meta: Mapping) -> None:
    """The null arm must be the same experiment, one state apart.

    Same release, same cache, same environment, same collection at the same
    count -- and a DIFFERENT target identity, because two names for one
    reading is not a control (task 603's null B reproduced null A exactly; a
    re-read measures determinism, not noise).
    """
    for key in BOUND_META:
        if baseline_meta.get(key) != null_meta.get(key):
            raise NullArmError(f"null arm not bound to the baseline: {key}")
    base, null = baseline_meta["target_manifest"], null_meta["target_manifest"]
    for key in ("qdrant_url", "collection", "points_count"):
        if base.get(key) != null.get(key):
            raise NullArmError(
                f"null arm target {key} {null.get(key)!r} != baseline "
                f"{base.get(key)!r} -- a null arm is the same collection at "
                f"the same count, or it is measuring something else")
    if base.get("identity") == null.get("identity"):
        raise NullArmError("null arm carries the baseline's own identity")
    if (baseline_meta["corpus_manifest"].get("representation") !=
            null_meta["corpus_manifest"].get("representation")):
        raise NullArmError(
            "null arm representation differs from the baseline's -- the "
            "control measures the instrument, not a second candidate")


def null_ranks(null_sets: Mapping) -> dict[str, object]:
    """`set/qid` -> the rank the null arm read, for the instrument listing."""
    return {f"{name}/{qid}": (row or {}).get("rank")
            for name, result in null_sets.items()
            for qid, row in ((result or {}).get("per_q") or {}).items()}


def render(header: str, metrics: Mapping[str, Mapping],
           attributable: Sequence, instrument: Sequence,
           basis: Mapping | None, ranks: Mapping[str, object] | None = None,
           tripwire: Mapping | None = None) -> list[str]:
    """The verdict's null-arm block, in the order a reader needs it."""
    lines = []
    if basis is None:
        lines.append("NULL-ARM none -- ADR-043 am.3 unadjudicated "
                     "(a PASS needs none; a FAIL is not yet the pack's)")
        if tripwire and tripwire["unadjudicated"]:
            lines.append(
                f"  TRIPWIRE-UNADJUDICATED {len(tripwire['raw'])} legal/names "
                "loss(es) and no null arm -- am.6: this still REFUSES, and "
                "the null arm is what would say whose it is")
        return lines
    lines.append(
        f"NULL-ARM {header} protocol={basis['protocol']} "
        f"membership={basis['membership_sha256'][:8]}")
    for name, value in metrics.items():
        lines.append(
            f"  {name}: raw {value['delta_raw']:+d}; null "
            f"{value['delta_null']:+d}; attributable "
            f"{value['attributable']:+d}; band +-{value['band']}"
            f" -> {'ok' if value['ok'] else 'REFUSED'}")
    lines.append(f"  strict per-gold losses: attributable={len(attributable)} "
                 f"instrument={len(instrument)}")
    ranks = ranks or {}
    for key, before, after in instrument:
        lines.append(f"  INSTRUMENT {key} {before}->{after} "
                     f"(null arm read {ranks.get(key)})")
    if tripwire is not None and tripwire["raw"]:
        lines.append(
            f"  legal/names losses: attributable="
            f"{len(tripwire['attributable'])} "
            f"instrument={len(tripwire['instrument'])} (am.6; zero tolerance "
            f"on the attributable ones)")
        for key, before, after in tripwire["instrument"]:
            lines.append(f"  TRIPWIRE-INSTRUMENT {key} {before}->{after} "
                         f"(null arm read {ranks.get(key)}, no content "
                         f"difference)")
        for key, before, after in tripwire["attributable"]:
            lines.append(f"  TRIPWIRE-ATTRIBUTABLE {key} {before}->{after} "
                         f"(null arm read {ranks.get(key)}) -- REFUSED")
    return lines
