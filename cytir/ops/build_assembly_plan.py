#!/usr/bin/env python3
"""Derive the assembly plan from measured artifacts.

Pure function of committed inputs; no corpus walk, no network.

A note on why this does NOT read the collision ledger
-----------------------------------------------------
The gate checks the plan against the sha-pinned task-456 ledger: the
multi-claim population, the row split, the tombstone count, the identity
collapse.  If the plan were derived from the ledger, every one of those checks
would compare the ledger to itself -- a filter selecting on what it reports.
So this builder is allowed the claim-manifest index, the fold report, and the
census, and is deliberately not allowed the ledger.  The two derivations meet
for the first time inside the guard, which is the only place a disagreement can
still be caught.

Conservation, and what it means to be honest about it
-----------------------------------------------------
Points whose owner is genuinely undecided get their own bucket.  Four
unadjudicated double claims are live; folding those points into ``replaced``
would make the arithmetic close while asserting an ownership nobody has ruled
on.  ``contested`` says what is true: measured, named, and not yet decided.

The pending lanes are stated on census 453's bucket basis while ``pending`` is
measured off the census remainder, and those two disagreed by 121,462 before
this task touched anything.  Forcing them to agree with a balancing lane would
invent a population, so the gap is declared and the guard treats it as an open
accounting item rather than a passing sum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = "aw.assembly-plan.v1"

TASK427 = "task427_postconsent"

ADR101_TABLE = {
    "exact_epoch4": "this_set",
    "reembed_epoch3_in_place": "this_set",
    "named_slice_payload_rule": "this_set",
    "atomic_epoch1": TASK427,
    "adr092_remainder": TASK427,
    "tombstone": "nobody",
}


def require_input(root: Path, rel: str) -> Path:
    """The path of a per-build assembly input under `root`, or ONE named reason (exit 2) when it is absent.

    These `data/assembly/` records are produced by a master build for one collection and are not shipped in
    the public cut (task 2285): a stranger's tree lacks them and supplies its own with `--root`.
    """
    path = root / rel
    if not path.is_file():
        raise SystemExit(f"build-assembly-plan: per-build assembly input {rel} not found under {root}; "
                         "it is produced by the master's assembly build and not shipped, so supply "
                         "your own tree with --root")
    return path


def binding(root: Path, rel: str) -> dict:
    path = require_input(root, rel)
    return {"uri": rel, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}



# The `why` for the two gzipped id-list records, kept beside the producer
# that emits them rather than only in the artifact they describe: a reader
# who finds one of these in a plan needs to know what it does BEFORE
# deciding whether the plan is safe to run.
G5_ROW_SUPPRESSION_NOTE = (
'aw.assembly-row-suppression-ids.v1 -- one point_id string per line, gzipped. Task 511: the ADR-101b/c ratified 13-pair row-level loser suppression (adr101_row_suppression.py), 301,094 distinct row ids across 8 sets (task472_cc/f2, task477, task482_cc/f2, task492, task494, task495). task427_postconsent is excluded (its own ADR-101 Ruling 1 suppression is a separate, older mechanism, adjudication.adr101a_correction_2, not wired here). 91 rows further excluded from the naive per-set union: G7 convergent-identity rows (byte-identical content minted the SAME row id by two of these 8 sets) where one co-owner suppresses and the other legitimately keeps its row -- suppressing the id globally would drop the row from the collection ENTIRELY rather than loading it once from the keeping set (harmless idempotent duplicate staging, never data loss); see task 511 script build_g5_suppression_artifact.py.')

G7_COLLISION_WINNERS_NOTE = (
"aw.cross-set-collision-winners.v1 -- one row per staged point id that more than one contributing set stages. Names WHICH set's row loads, from ratified precedence (task 513): the enumerated supplement pair, else the ADR-101/b/c class rank in data/assembly/adr101-class-precedence.v1.json, else ADR-101b Ruling 3a's lexicographic label. Consumed by `assembly_binder.py load --collision-winners`, NOT at stage time: suppressing a shared id at stage would drop it from both sets (task 511 s3).")

PRECEDENCE_SUPPLEMENT = "data/r0-427-collision-ledger-adr101c-supplement.v1.json"


def adjudicated_pairs(root: Path) -> dict[frozenset[str], dict]:
    """The contested pairs that now have a RATIFIED winner, keyed by set pair.

    A double-claimed point with no ruling has no owner, and this plan puts it in
    ``contested`` precisely so nobody can quietly fold it into ``replaced`` and
    close the arithmetic on an ownership nobody granted.

    Once the ADR ratifies, the opposite becomes true: the point HAS an owner.
    Leaving it in ``contested`` then understates both dispositions and holds
    guard G3 open on a question that has been answered -- which is its own kind
    of dishonesty, and the one this plan was drifting into after PR #80 and
    PR #84 merged.

    The ratification predicate is IMPORTED from the guard rather than
    reimplemented.  Two copies of "is this ADR in force" is how a plan comes to
    believe something the gate does not, and the gate is the instrument of
    record.  If the gate does not resolve, this returns nothing and every
    contested point stays contested: the failure direction is closed.
    """
    path = root / PRECEDENCE_SUPPLEMENT
    if not path.is_file():
        return {}
    record = json.loads(path.read_text())

    from cytir.ops.assembly_guards import _draft_marker_cleared  # noqa: PLC0415

    gate = (record.get("ratification") or {}).get("gate") or {}
    if gate.get("kind") != "adr_draft_marker_cleared":
        raise SystemExit(
            f"precedence supplement declares a ratification gate this builder "
            f"cannot evaluate ({gate.get('kind')!r}); refusing to guess whether "
            "it is in force")
    ratified, reason = _draft_marker_cleared(root, gate)
    if not ratified:
        return {}

    out: dict[frozenset[str], dict] = {}
    for line in (record.get("lines") or []):
        left, separator, right = str(line.get("pair") or "").partition(" x ")
        if not separator:
            raise SystemExit(
                f"precedence line {line.get('id')!r} names no 'LEFT x RIGHT' pair")
        if line.get("winner") not in {left, right}:
            raise SystemExit(
                f"precedence line {line.get('id')!r} names a winner that is not "
                "one of the two sets it rules")
        out[frozenset({left, right})] = line
    return out


ADDITION_MODE = "net_new_addition"


def addition_witnesses(root: Path) -> dict[str, dict]:
    """Find each addition set's witness by what it SAYS, not what it is called.

    Deriving the filename from the set_id would make the binding a naming
    convention, and a convention is exactly what nobody notices breaking: a
    witness that quietly stops being found reads to the guard as a witness that
    was never minted, and the set fails for a reason that is not its own.  So
    every witness in the directory is opened and matched on its declared
    ``set_id``.

    A set with no witness is left unbound on purpose.  The gate is the
    instrument of record, and G7 refuses an addition whose disjointness is
    unmeasured; refusing here instead would move that judgement into the
    builder, where the plan is written rather than checked.
    """
    found: dict[str, dict] = {}
    for path in sorted((root / "data/assembly").glob("census-witness-*.json")):
        record = json.loads(path.read_text())
        if record.get("schema") != "aw.net-new-addition-census-witness.v1":
            continue
        set_id = str(record.get("set_id") or "")
        if not set_id:
            raise SystemExit(f"{path.name}: an addition witness names no set_id")
        if set_id in found:
            raise SystemExit(
                f"two addition witnesses claim {set_id}: {found[set_id]['path']} "
                f"and {path.name}. Refusing to guess which one the plan binds.")
        found[set_id] = {"path": path.name,
                         "rel": f"data/assembly/{path.name}"}
    return found


def measured_covered_union(fold: dict) -> int:
    """How much of the census remainder the folded sets actually cover.

    The fold measures this over the real ID sets and publishes it.  This used to
    be approximated in the plan as ``sum(per-set inside) - sum(pairwise claim
    overlaps)`` -- inclusion-exclusion truncated at pairs, over overlap counts
    that were never restricted to the remainder.  Both halves are wrong on the
    real corpus: three-way sharing over-subtracts, and the four pairs naming
    ``task427_postconsent`` subtracted 40,400 points that were never added,
    because none of task 427's claims are inside the remainder at all.

    Refuse rather than fall back to the estimate.  A plan built on the estimate
    and a plan built on the measurement are not comparable, and silently
    preferring whichever happens to be present is how that difference goes
    unnoticed for a build.
    """
    union = (fold.get("remainder_basis") or {}).get("covered_union")
    if union is None:
        raise SystemExit(
            "the fold report does not publish `remainder_basis.covered_union`. "
            "The plan used to approximate it as sum(inside) - sum(pairwise "
            "claim overlaps), which over-subtracts three-way sharing and "
            "subtracts pair overlaps that are not inside the remainder at all. "
            "Re-run the fold with a build that measures the union.")
    return int(union)


def pairwise_covered_estimate(covered: int, overlaps: dict,
                              folded_ids: set) -> int:
    """The retired approximation, kept only to report the delta against it."""
    return covered - sum(
        int(count) for pair, count in overlaps.items()
        if pair.partition(" x ")[0] in folded_ids
        and pair.partition(" x ")[2] in folded_ids)


def unattributed_folded_sets(folded: list, lane_subtractions: dict,
                             containment: dict) -> list:
    """Folded sets that draw from the remainder but from no named lane.

    A set with no lane-basis line is counted on the remainder basis and not on
    the bucket basis, so it inflates the reconciliation residue by exactly its
    own claim count and the reconciliation quietly stops reconciling.  An
    unattributed set is a missing line, and a missing line is the defect the
    reconciliation exists to make visible -- so it is a refusal, not a warning.
    """
    return sorted(
        row["set_id"] for row in folded
        if row.get("mode") != "net_new_addition"
        and row["set_id"] not in lane_subtractions
        and int(containment.get(row["set_id"], {}).get(
            "inside_adr092_remainder", 0)) > 0)


def _divergence_reason(set_id: str, subtracted: int, removed: int,
                       witness: int) -> str:
    if removed == subtracted:
        return "witness and lane subtraction agree; this set adds no divergence"
    if subtracted == 0:
        # This branch used to add "they sit in census bucket A". That sentence
        # was picked from the SHAPE of the numbers (`subtracted == 0`) and no
        # measurement was ever behind it. Task 502 sec.2.4 ran the per-point
        # placement for the largest line it produced -- task467's 338,857
        # in-remainder claims -- and found **0** in bucket A, with 80.5% in
        # lane-bearing buckets. Deleted rather than softened: a reconciliation
        # line that states where points sit must be measured or say nothing.
        return ("no lane-basis subtraction is recorded for this set, so the "
                "bucket basis credits it nothing while the remainder basis "
                "credits every point it covers. Where its points actually sit "
                "is a per-point measurement, published by census v34's "
                "`covered_by_set`, not inferable from this line.")
    if witness == 0:
        return ("subtracted whole from its lane, but its points are outside the "
                "census remainder entirely -- the census never counted them as "
                "remaining work, so removing them from the remainder removes "
                "nothing.")
    return ("the bucket basis subtracts every point the set claims from its "
            "lane; the census remainder only loses the points its witness "
            "carries. The difference is what the witness had to exclude -- "
            "contested with another set, or already carried by an earlier "
            "witness.")


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--census-run", dest="census_v32_run", type=Path,
                        required=True,
                        help="RUN.json of the newest remainder census")
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--apply-terms", type=Path, default=None,
                        help="task 514 script measure_apply_terms.py output: "
                             "the delete set measured as an ID set difference")
    parser.add_argument("--loaded-row-ids", type=Path, default=None,
                        help="an INDEPENDENT measurement of the distinct ids "
                             "the load writes: the binder's load-summary.json "
                             "(`distinct_ids`) or task 514 scripts/"
                             "load_preflight_514.py (`expected_points_count`)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    root: Path = args.root
    index = json.loads(require_input(root, "data/assembly/claim-manifests.v1.json").read_text())
    fold = json.loads(require_input(root, "data/assembly/fold-report.v1.json").read_text())
    census = json.loads(require_input(root, "data/r0-1024-coverage-census.v1.json").read_text())
    v32 = json.loads(args.census_v32_run.read_text())

    membership = int(census["buckets"]["sum_check"]["total"])
    tombstoned = int(census["buckets"]["E_no_vector_will_exist"]["points"])
    v32_remaining = int(v32["remaining_points"])
    v31_remaining = int(census["basis"]["remainder_census"]["points"])

    containment = fold["remainder_containment"]
    task477_claims = int(containment.get(
        "task477_direct_capture_epoch4_chunks", {}).get("claims", 0))

    # ---- which census-453 lane each later-folded set draws down -------------
    # The `lanes` list below is stated on census 453's BUCKET basis, and every
    # set folded after task 462 wrote it draws its claimed old points out of one
    # of those buckets.  Until a set has a line here, the bucket basis keeps
    # counting work that a landed set has already done, and the reconciliation
    # residue grows by exactly that set's claim count -- which is what task 497
    # sec.5 and task 501 sec.5 escalated rather than closed.
    #
    # The attribution is NOT a judgement call.  It is task 498 sec.4's measured
    # lane table, and it reproduces that table's own drawdown figures exactly:
    #
    #   F1 (cc-main-2026-30-never-acquired)  task492            428,444
    #   F2 (class1 retained bytes, PDF half) task471            379,439
    #   D  (consent held)                    the four consent
    #                                        supplements        103,129
    #                                        = 1,405 + 29,794 + 2,270 + 69,660
    #
    # 103,129 is 498 sec.4's published D drawdown to the point, derived here
    # from the fold's own containment rather than retyped -- two independent
    # derivations of the same number.  A held->admitted re-derivation is by
    # definition a drawdown of `cc-consent-held` / `direct-consent-held`, which
    # is what those four sets are; no lane goes negative under this attribution,
    # which is the arithmetic check on it.
    LANE_ATTRIBUTION = {
        "task492_crawl30_cc_chunks": "F1-cc-main-2026-30-never-acquired",
        "task471_f2_pdf_docling_epoch4_chunks": "F2-pdf-half",
        "task482_cc_consent_supplement_chunks": "D-blocked-needs-decision",
        "task482_f2_consent_supplement_chunks": "D-blocked-needs-decision",
        "task494_4crawl_supplement_cc_chunks": "D-blocked-needs-decision",
        "task495_r7_stale_supplement_cc_chunks": "D-blocked-needs-decision",
    }

    # One set does not draw from a single lane, and its split is MEASURED per
    # point rather than declared.  Task 497 sec.5 and task 501 sec.5 both left
    # `task500_news_1024_in_place` unattributed on the grounds that census 453
    # publishes bucket membership as counts, not point IDs.  That is true of the
    # census -- but each bucket MEMBER has a producing artifact that does
    # enumerate, and at 22 points the join is cheap.  Task 502 ran it: the 22
    # rows are 5 distinct URLs, and each was located in a sealed ledger.
    #
    #   11 pts  swissinfo.ch/eng/swiss-diaspora/how-swiss-schools-abroad-...
    #           direct-lane TERMINALS.jsonl, partition p00424-a1,
    #           status "oversized", point_count 11
    #           -> census bucket E member `direct-oversized`.
    #   10 pts  3x srf.ch/sendungen/kassensturz-espresso/...
    #           r0-adr092-cc-parents-1d22ba19-p0/p5/p8 declined.jsonl.gz,
    #           status "declined", with cc-plan capture counts 7 / 2 / 1
    #           matching the staged chunk counts per URL exactly
    #           -> census bucket E member `cc-consent-declined`.
    #    1 pt   rts.ch/rts-couleur3 -- found in NO probed ledger.  Recorded as
    #           unplaced rather than assigned to whichever lane would close the
    #           arithmetic.
    #
    # So 21 of 22 are bucket E, and bucket E sits in the census remainder
    # belonging to NO lane (see the tombstone line below).  This set therefore
    # draws down no lane at all: its subtraction is 0, not a positive number.
    #
    # An earlier revision of this block recorded 10 of these as an F1 drawdown.
    # That was wrong twice over, and both errors are worth naming because they
    # are easy to repeat.  `miss-frontier.jsonl.gz` is the CC MISS union, not
    # census member F1 (`cc-main-2026-30-index-hits-never-acquired`, the CAPTURE
    # side) -- reading a hit there as "F1" inverts the lane.  And the probe that
    # produced it excluded the URL it had already placed by hand, so it agreed
    # with a conclusion instead of reproducing it.  The producer named below now
    # probes every URL against every ledger.
    MEASURED_LANE_SPLIT = {
        "task500_news_1024_in_place": {
            "_bucket_E_direct_oversized": 11,
            "_bucket_E_cc_consent_declined": 10,
            "_unplaced": 1,
            "_measured_by": "task 502 script place_news_in_place_points.py",
        },
    }

    def lane_draw(lane_id: str) -> int:
        """What the later-folded sets take out of one census-453 lane."""
        whole = sum(int(containment.get(set_id, {}).get("claims", 0))
                    for set_id, lane in LANE_ATTRIBUTION.items()
                    if lane == lane_id)
        split = sum(int(row.get(lane_id, 0))
                    for row in MEASURED_LANE_SPLIT.values())
        return whole + split
    overlaps = fold["claim_overlaps"]
    quarantined = sorted(fold.get("quarantined") or {})

    # ---- what the folded sets cover, inside the census remainder ----------
    # ONE basis, checked rather than assumed: the fold must have measured
    # containment against the same remainder this plan reports.  Task 462 had
    # to correct for the difference (the fold ran on v31, the plan on v32) by
    # subtracting each set's overlap with task 427's witness by hand.  That
    # correction is a double subtraction the moment the two agree, so the fix
    # is to make them agree and refuse when they do not.
    basis = fold.get("remainder_basis") or {}
    if int(basis.get("points", -1)) != int(v32["remaining_points"]):
        raise SystemExit(
            f"basis mismatch: the fold measured containment against a "
            f"{int(basis.get('points', -1)):,}-point remainder, the census RUN "
            f"reports {int(v32['remaining_points']):,}. Re-run the fold with "
            f"--remainder pointing at this census's remaining-points.jsonl.gz.")
    folded = [row for row in index["sets"] if row.get("folded_from")]
    folded_ids = {row["set_id"] for row in folded}
    covered = 0
    per_set = {}
    for row in folded:
        set_id = row["set_id"]
        inside = int(containment.get(set_id, {}).get("inside_adr092_remainder", 0))
        per_set[set_id] = {"claims": int(containment.get(set_id, {}).get(
                               "claims", 0)),
                           "inside_census_remainder": inside}
        covered += inside
    # What the folded sets actually cover inside the remainder is a UNION, and
    # the fold now measures it over the real ID sets and publishes it.  Read the
    # measurement; do not re-derive it.
    #
    # The previous derivation here was `sum(inside) - sum(pairwise overlaps)` --
    # inclusion-exclusion truncated at pairs, over claim-overlap counts that were
    # never restricted to the remainder.  Both halves are wrong on this corpus:
    # three-way sharing over-subtracts, and the four pairs naming task427 (which
    # contributes 0 to the sum, because none of its claims are inside the
    # remainder) subtracted 40,400 points that were never added.  The error
    # propagated straight into `pending`, `replaced` and the reconciliation
    # residue.  Refuse rather than fall back: an old plan built on the estimate
    # and a new one built on the measurement are not comparable, and silently
    # preferring whichever is present is how that difference goes unnoticed.
    approximated = pairwise_covered_estimate(covered, overlaps, folded_ids)
    folded_pair_overlaps = sum(
        int(count) for pair, count in overlaps.items()
        if pair.partition(" x ")[0] in folded_ids
        and pair.partition(" x ")[2] in folded_ids)
    covered = measured_covered_union(fold)

    # Only a pair with NO ratified winner is contested.  A pair the precedence
    # supplement rules has an owner, and the disposition below hands each of its
    # points to that owner.
    rulings = adjudicated_pairs(root)
    # The enumeration rules first; the ratified CLASS rules rule what it never
    # reached. Same table and same reduction as the guard's `class_rule_rulings`
    # (imported, not re-implemented -- a second copy of "who wins this class
    # pair" is the mirror-contract defect this project keeps paying for), so a
    # pair the plan folds into `replaced` is a pair the guard also finds ruled.
    if rulings:  # class rules only apply while the ADR they extend is ratified
        from cytir.ops.assembly_guards import class_rule_rulings  # noqa: PLC0415

        class _Ctx:
            pass
        ctx = _Ctx()
        ctx.root = root
        ctx.manifests = index
        ctx.records = {"fold_report": fold}
        for pair, line in class_rule_rulings(ctx, set(overlaps)).items():
            rulings.setdefault(pair, line)
    unruled = {pair: int(count) for pair, count in overlaps.items()
               if frozenset(pair.split(" x ")) not in rulings}
    ruled = {pair: int(count) for pair, count in overlaps.items()
             if pair not in unruled}
    contested = sum(unruled.values())
    unlocated = int(containment.get("task467_entity_card_1024_reembed", {})
                    .get("outside", 0))

    pending = v32_remaining - covered
    # task462 (172cf99d) wrote this as a loop over `folded` PLUS an explicit
    # add-on for "the 413-A Wikipedia in-place re-embed", because at the time
    # `wikipedia_413a` was not yet itself a folded set with a containment
    # entry. It has been one since the same task462 commit added it to
    # data/assembly/claim-manifests.v1.json with `folded_from` set and
    # `mode: in_place_reembed` -- so the loop below has counted its 19,227,117
    # claimed points all along, and the add-on double-counted them a second
    # time. That drove `re_embedded_in_place` to 38,884,954 against a total
    # membership of 34,464,401, which made `replaced` (membership minus the
    # other buckets) negative in the sealed task516 plan
    # (data/assembly/assembly-plan.current.v1.json) -- a real, reproducible
    # defect task 532 found via the new schema-validation contract test, not
    # a masked one. Fixed here by removing the redundant add-on; the already-
    # sealed task516 artifact is left as-is (data ledger, not code) and
    # explicitly excluded below pending Tim's call on whether to reseal it.
    in_place = sum(
        int(containment.get(row["set_id"], {}).get("claims", 0))
        for row in folded if row.get("mode") == "in_place_reembed")
    # A contested point an in-place set claims is only re-embedded in place if
    # that set actually OWNS it.  Before ratification it owned none of them, so
    # every such overlap came off.  Now it owns the ones the supplement rules in
    # its favour, and only the rest come off: the unruled pairs, plus any pair
    # ruled against it.
    #
    # Membership is read off the index rather than matched on substrings of the
    # set name.  The literal `"task467" in pair or "rii" in pair` this replaces
    # happened to be right for the three sets that existed, and would silently
    # capture any future set whose id contained "rii".
    in_place_sets = {row["set_id"] for row in index["sets"]
                     if row.get("mode") == "in_place_reembed"}
    not_owned_by_in_place = 0
    for pair, count in overlaps.items():
        sides = set(pair.split(" x "))
        if not sides & in_place_sets:
            continue
        line = rulings.get(frozenset(sides))
        if line is None or line["winner"] not in in_place_sets:
            not_owned_by_in_place += int(count)
    in_place -= not_owned_by_in_place
    replaced = membership - tombstoned - pending - contested - in_place

    # ---- census v34: the lane table is MEASURED, not computed --------------
    # Up to census 453 the lane sizes were a formula -- `bucket count minus
    # every landed claim` -- over counts the census published without point
    # IDs, while `pending` was measured off the remainder.  Two bases, one
    # computed and one measured, and the difference could only be declared.
    # Census v34 (`schema aw.r0-1024-coverage-census.v2`) places every
    # remainder point in a bucket per point, so both bases read off ONE
    # measurement and the lane table is looked up rather than derived.  The
    # v1 formulas below are kept for a v1 census, unchanged.
    census_v34 = census.get("lanes") is not None
    gap427 = census["task427_reconciliation_gap"]
    by_bucket = gap427["inside_remainder_by_bucket"]
    if sum(int(v) for v in by_bucket.values()) != int(
            gap427["inside_adr092_remainder"]):
        raise SystemExit("census 453's per-bucket split of task 427's "
                         "in-remainder claims does not sum to its own total")
    if census_v34:
        f1_points = f2_points = b2_points = 0
    else:
        members = {member["id"]: member for member in
                   census["buckets"]["F_unaccounted_no_documented_lane"]["members"]}
        f1_points = int(
            members["cc-main-2026-30-index-hits-never-acquired"]["points"])
        f2_points = int(
            members["class1-retained-bytes-located-never-materialized"]["points"])
        b2_points = int(next(
            member["old_points"] for member
            in census["buckets"]["B_sealed_epoch4_chunks_embed_pending"]["members"]
            if member["id"] == "track1a-retained"))
    rii_claims = int(containment.get(
        "task472_rii_courts_1024_embed", {}).get("claims", 0))
    f2_html_claims = int(containment.get(
        "task472_f2_html_1024_embed", {}).get("claims", 0))

    # ---- the additions term ----------------------------------------------
    # These rows were never membership points, so not one of them belongs in
    # the disposition identity above -- and none of the arithmetic that built
    # it touches them: an addition's claim is the empty set, so `containment`
    # credits it nothing and `covered`, `in_place` and `replaced` are all
    # unchanged by its arrival.  What it needs is its own term, and the
    # collection size the disposition alone cannot state.
    witnesses = addition_witnesses(root)
    addition_entries = []
    for row in index["sets"]:
        if row.get("mode") != ADDITION_MODE:
            continue
        set_id = row["set_id"]
        minted = row.get("new_point_ids") or row.get("additions") or {}
        witness_rel = witnesses.get(set_id, {}).get("rel")
        entry = {
            "set_id": set_id,
            "rows": int((row.get("vector") or {}).get("rows") or 0),
            "claimed_production_points": int(
                (row.get("claims") or {}).get("count") or 0),
            "row_ids": {"uri": minted.get("uri"),
                        "sha256": minted.get("sha256"),
                        "bytes": minted.get("bytes"),
                        "count": int(minted.get("count") or 0)},
        }
        if witness_rel:
            # pinned by digest here as well as in `records`: the same file is
            # named twice and a reader who checks only one of them should not
            # be able to check a different file from the guard.
            entry["witness"] = binding(root, witness_rel)
        addition_entries.append(entry)
    addition_entries.sort(key=lambda entry: entry["set_id"])
    addition_rows = sum(int(entry["rows"]) for entry in addition_entries)

    lanes = census["lanes"] if census_v34 else [
        {"lane_id": "C-direct-acquisition", "census_bucket": "C",
         "points": (int(census["buckets"]["C_in_flight_acquisition"]["points"])
                    - task477_claims),
         "work_type": "acquire+chunk+embed",
         "state": "task 477 materialized, embedded and (task 481) RE-SEALED "
                  f"this lane; its {task477_claims:,} claimed old points are "
                  "subtracted here. What remains is the acquisition lane's "
                  "other terminals plus the conserved zero_chunk / "
                  "duplicate_id / quarantine captures, which task 477 "
                  "terminated without producing a vector -- a named "
                  "sub-population that still needs its own disposition, not "
                  "acquisition work",
         "blocks_guards": ["G1", "G3", "G5", "G7"],
         "will_publish_claims": True},
        {"lane_id": "D-blocked-needs-decision", "census_bucket": "D",
         "points": (int(census["buckets"]["D_blocked_needs_decision"]["points"])
                    - int(by_bucket["D5-frontier-nonmine"]) - rii_claims
                    - lane_draw("D-blocked-needs-decision")),
         "work_type": "blocked-needs-decision",
         "state": "consent held / frontier non-mine; rii's 38,159 have landed, "
                  "and the four consent supplements (tasks 482 cc/f2, 494, 495) "
                  f"have re-derived {lane_draw('D-blocked-needs-decision'):,} "
                  "held points to admitted",
         "blocks_guards": ["G3"], "will_publish_claims": True},
        {"lane_id": "F1-cc-main-2026-30-never-acquired", "census_bucket": "F",
         "points": (f1_points - int(by_bucket["F1"])
                    - lane_draw("F1-cc-main-2026-30-never-acquired")),
         "work_type": "acquire+chunk+embed",
         "state": "WARC offsets planned, capture side never fetched; task 492's "
                  f"crawl-30 pass claimed "
                  f"{lane_draw('F1-cc-main-2026-30-never-acquired'):,}",
         "blocks_guards": ["G1", "G3", "G5", "G7", "G8"],
         "will_publish_claims": True},
        {"lane_id": "F2-pdf-half", "census_bucket": "F",
         "points": (f2_points - int(by_bucket["F2b-class1-nolocator"])
                    - f2_html_claims - lane_draw("F2-pdf-half")),
         "work_type": "materialize+chunk+embed",
         "state": "the HTML half landed as task472_f2_html; the PDF half is "
                  f"task 471 (Docling), which claimed "
                  f"{lane_draw('F2-pdf-half'):,}",
         "blocks_guards": ["G1", "G3", "G5", "G7", "G8"],
         "will_publish_claims": True},
        {"lane_id": "B2-track1a-residual", "census_bucket": "B",
         "points": b2_points - int(by_bucket["B2"]),
         "work_type": "needs-rebucketing",
         "state": "producer CANCELLED (ADR-101 Ruling 3); the points task 427 "
                  "does not cover still need a lane",
         "blocks_guards": ["G3"], "will_publish_claims": True},
    ]
    # ---- the two accounting bases, reconciled line by line ---------------
    # The lanes are stated on census 453's BUCKET basis; `pending` is measured
    # off the census REMAINDER.  Task 462 declared the difference (119,288) and
    # explained it as census 453's own bucket-vs-remainder gap.  Neither half
    # of that was quite right, and both are now measured:
    #
    #   * the 121,462 gap is not "remainder points whose work is done, sitting
    #     in bucket A".  It is bucket E exactly: 152,587 points that will never
    #     get a vector, minus the 31,125 the census already subtracts as the
    #     ADR-091 natural-person policy-drop witness.  The other eight E members
    #     -- consent-declined, non-textual, oversized, zero-chunk, http/redirect
    #     /transport terminals -- carry no witness, so they sit in the remainder
    #     while belonging to no lane, because no lane will ever work on them.
    #     They are `tombstoned`, not `pending`.
    #   * the rest of the difference is per-set and enumerable.
    #
    # Every line below is a measured population with a named provenance, and
    # the builder refuses if they do not sum to the observed gap.  A
    # reconciliation that closes by residue is not a reconciliation.
    lane_total = sum(lane["points"] for lane in lanes)
    bucket_total = sum(int(census["buckets"][key]["points"])
                       for key in ("B_sealed_epoch4_chunks_embed_pending",
                                   "C_in_flight_acquisition",
                                   "D_blocked_needs_decision",
                                   "F_unaccounted_no_documented_lane"))
    tombstone_bucket = int(census["buckets"]["E_no_vector_will_exist"]["points"])
    tombstone_witnessed = int(next(
        member["points"]
        for member in census["buckets"]["E_no_vector_will_exist"]["members"]
        if member["id"] == "adr091-natural-person-policy-drop"))
    if census_v34:
        # v34 moves the tombstoned residue OUT of the remainder, so bucket E no
        # longer sits inside it -- which is what retires this line rather than
        # shrinking it. The census publishes the measurement; it is read, not
        # assumed, and the whole of census 453's declared bucket-vs-remainder
        # gap WAS this line.
        tombstone_in_remainder = int(
            census["buckets"]["E_no_vector_will_exist"]["points_in_remainder"])
        tombstone_unwitnessed = tombstone_in_remainder
    else:
        tombstone_unwitnessed = tombstone_bucket - tombstone_witnessed

    # Each side of the accounting removes populations from a different total.
    # Both maps are derived from the same measured artifacts, and the LANE
    # SUBTRACTIONS below are the very numbers the lane list above used -- one
    # derivation, read twice, so the reconciliation cannot drift from the thing
    # it reconciles.
    witnessed = {
        row["set_id"]: int((row.get("census_witness") or {}).get("records") or 0)
        for row in index["sets"]}
    if census_v34:
        # v34's lane table is a per-point measurement of the SAME population
        # the remainder basis measures, so "what this set removed from a lane"
        # and "what it removed from the remainder" are one number, not two
        # bases to reconcile. The per-set line is therefore not a declaration
        # but a CROSS-CHECK: the census derived `inside_remainder` while
        # placing every point in a bucket, the fold derived it while measuring
        # containment, and the two must agree set by set.
        covered_by_set = census.get("covered_by_set") or {}
        mismatches = []
        for row in folded:
            set_id = row["set_id"]
            line = covered_by_set.get(set_id)
            want = int(containment.get(set_id, {}).get(
                "inside_adr092_remainder", 0))
            if line is None:
                mismatches.append(
                    f"{set_id}: census v34 publishes no covered_by_set line, "
                    "so its points were never placed in a bucket")
                continue
            got = line.get("inside_remainder")
            if got is None or int(got) != want:
                mismatches.append(
                    f"{set_id}: the census places {got} of its claims inside "
                    f"the remainder, the fold measured {want:,}")
        if mismatches:
            raise SystemExit(
                "census v34 and the fold disagree about what the folded sets "
                "cover, so the two accounting bases are not one measurement:\n  "
                + "\n  ".join(mismatches))
        lane_subtractions = {
            set_id: int((covered_by_set.get(set_id) or {}).get(
                "inside_remainder") or 0)
            for set_id in sorted({row["set_id"] for row in folded}
                                 | set(covered_by_set))
        }
    else:
        lane_subtractions = {
            TASK427: int(gap427["inside_adr092_remainder"]),
            "task472_cc_materializer_1024_embed": int(containment.get(
                "task472_cc_materializer_1024_embed", {}).get("claims", 0)),
            "task472_rii_courts_1024_embed": rii_claims,
            "task472_f2_html_1024_embed": f2_html_claims,
            "task477_direct_capture_epoch4_chunks": task477_claims,
            "task467_entity_card_1024_reembed": 0,
            # the later-folded sets, read off the SAME attribution the lanes
            # above subtract -- one derivation, read twice, so the
            # reconciliation cannot drift from the thing it reconciles.
            **{set_id: int(containment.get(set_id, {}).get("claims", 0))
               for set_id in LANE_ATTRIBUTION},
            # the measured split: only the part that came out of a lane
            **{set_id: sum(int(count) for key, count in row.items()
                           if not key.startswith("_"))
               for set_id, row in MEASURED_LANE_SPLIT.items()},
        }
        unattributed = unattributed_folded_sets(folded, lane_subtractions,
                                                containment)
        if unattributed:
            raise SystemExit(
                "these folded set(s) draw points out of the census remainder "
                "but no lane-basis line says which census-453 bucket they came "
                "from, so the two accounting bases cannot be reconciled: "
                + ", ".join(unattributed))
    # What the census remainder actually lost to each set.  On a v1 census that
    # is the records its witness carries, or -- for a set with no witness --
    # what it covers inside the remainder.  On v34 the remainder is pruned by
    # the TOMBSTONE rather than by witnesses, so a witness records nothing about
    # this remainder and the only true answer is the measured containment.
    remainder_side = {
        set_id: (int(containment.get(set_id, {}).get(
                     "inside_adr092_remainder", 0))
                 if census_v34 else
                 (witnessed.get(set_id, 0) or int(containment.get(
                     set_id, {}).get("inside_adr092_remainder", 0))))
        for set_id in lane_subtractions}
    lines = []
    for set_id, subtracted in sorted(lane_subtractions.items()):
        removed = remainder_side[set_id]
        lines.append({
            "set_id": set_id,
            "claims": int(containment.get(set_id, {}).get("claims", 0)),
            "removed_from_a_lane_on_the_bucket_basis": subtracted,
            "removed_from_the_census_remainder": removed,
            "divergence": removed - subtracted,
            "why": _divergence_reason(set_id, subtracted, removed,
                                      witnessed.get(set_id, 0)),
        })
    if census_v34:
        why_tombstone = (
            f"bucket E is {tombstone_bucket:,} points that will never get a "
            f"vector, and census v34 moved every one of them OUT of the "
            f"remainder: {tombstone_in_remainder:,} of them sit inside it. "
            "That is what closes census 453's declared bucket-vs-remainder "
            "gap rather than shrinking it -- the gap WAS bucket E's "
            "unwitnessed part sitting in the remainder while belonging to no "
            "lane, and under Tim's 2026-09-02 re-bucketing those points are "
            "tombstoned, not pending, so the remainder no longer holds them.")
    else:
        why_tombstone = (
            f"bucket E is {tombstone_bucket:,} points that will never get a "
            f"vector. The census subtracts {tombstone_witnessed:,} of them "
            "as the ADR-091 policy-drop witness; the other "
            f"{tombstone_unwitnessed:,} carry no witness, so they sit in the "
            "remainder while belonging to no lane -- because no lane will "
            "ever work on them. They are tombstoned, not pending. This is "
            "the whole of census 453's declared bucket-vs-remainder gap, "
            f"exactly: {v31_remaining:,} - {bucket_total:,} = "
            f"{v31_remaining - bucket_total:,}.")
    tombstone_line = {
        "set_id": "bucket E, unwitnessed",
        "divergence": -tombstone_unwitnessed,
        "members": [member["id"] for member
                    in census["buckets"]["E_no_vector_will_exist"]["members"]
                    if member["id"] != "adr091-natural-person-policy-drop"],
        "why": why_tombstone,
    }
    reconciliation_lines = [row for row in lines if row["divergence"]]
    reconciliation_lines.append(tombstone_line)
    declared = sum(int(row["divergence"]) for row in reconciliation_lines)
    gap = lane_total - pending
    residue = gap - declared
    lane_basis = {
        "basis_a": {
            "name": ("census v34 measured lanes" if census_v34
                     else "census-453 buckets"),
            "total": lane_total,
            "derivation": (
                "the per-point bucket assignment: how many remainder points "
                "sit in a lane bucket AND are covered by no landed claim set. "
                "Not a formula over bucket counts."
                if census_v34 else
                f"buckets B+C+D+F ({bucket_total:,}) minus every landed claim "
                "set, per lane")},
        "basis_b": {"name": "census remainder", "total": pending,
                    "derivation": f"census remaining_points "
                                  f"({int(v32['remaining_points']):,}) minus "
                                  f"the folded sets' union inside it "
                                  f"({covered:,})"},
        "gap": gap,
        "declared": declared,
        "unexplained_residue": residue,
        "lines": reconciliation_lines,
        "pre_existing_basis_gap": v31_remaining - bucket_total,
        "pre_existing_basis_gap_is": "bucket E minus its witnessed part, exactly",
        "residue_note": (
            "census v34 publishes the bucket assignment PER POINT, so both "
            "bases are one measurement and a residue here would be a real "
            "disagreement between the census and the fold -- which the "
            "covered_by_set cross-check above refuses set by set before this "
            "arithmetic runs."
            if census_v34 else
            "census 453 publishes its bucket assignment as COUNTS, not as "
            "point IDs, so a residue at this level cannot be attributed to a "
            "named population without re-deriving the per-point bucket "
            "assignment. It is declared, not absorbed: no lane and no line is "
            "adjusted to make it vanish."),
    }
    if abs(residue) > 2000:
        # A refusal that does not show its working sends the next reader back to
        # re-derive the arithmetic by hand, which is how a wrong line gets
        # "fixed" by tuning a number until the residue vanishes.  Print both
        # bases and every line instead.
        report = [
            f"lane-basis reconciliation leaves {residue:+,} unexplained, which "
            "is too large to declare: a line is missing or wrong.",
            "",
            f"  basis A (census-453 buckets)   lane_total = {lane_total:,}",
            f"  basis B (census remainder)     pending    = {pending:,}"
            f"   (= remainder {v32_remaining:,} - covered union {covered:,})",
            f"  gap = A - B = {gap:+,}    declared = {declared:+,}"
            f"    residue = {residue:+,}",
            "",
            "  lanes (basis A):",
        ]
        for lane in lanes:
            report.append(f"    {lane['lane_id']:<40} {lane['points']:>12,}")
        report.append("")
        report.append("  reconciliation lines (declared divergences):")
        for row in reconciliation_lines:
            report.append(
                f"    {row['set_id']:<40} {int(row['divergence']):>+12,}")
        report.append("")
        report.append(
            f"  for reference, the retired pairwise approximation of `covered` "
            f"was {approximated:,}, i.e. {covered - approximated:+,} against the "
            "measured union.")
        raise SystemExit("\n".join(report))

    plan = {
        "schema": SCHEMA,
        "build_id": args.build_id,
        "task": 462,
        "target": {"dim": 1024, "dtype": "float16", "collection": "aw_e4"},
        "note": "Pre-flight plan for the current inventory. Not a plan to flip: "
                "lanes are unsealed, one contributing set is quarantined and "
                "four double claims are unadjudicated.",
        "membership": {
            "uri": "artifact://_preservation/aw_e3-membership-20260814/RECEIPT.json",
            "sha256": census["basis"]["membership_receipt"]["sha256"],
            "points": membership},
        "inputs": {
            "collision_ledger": binding(root, "data/r0-427-collision-ledger.v1.json"),
            "coverage_census": binding(root, "data/r0-1024-coverage-census.v1.json"),
            "manifest_index": binding(root, "data/assembly/claim-manifests.v1.json"),
        },
        "records": {
            "g4_tombstone_suppression":
                binding(root, "data/assembly/g4-tombstone-suppression.v1.json"),
            "g4_tombstone_url_suppression":
                binding(root, "data/assembly/g4-tombstone-url-suppression.v1.json"),
            "g6_mixed_url_resolution":
                binding(root, "data/assembly/g6-mixed-url-resolution.v1.json"),
            "fold_report":
                binding(root, "data/assembly/fold-report.v1.json"),
            "cancellation:track1a_retained_bind":
                binding(root,
                        "data/assembly/cancellation-track1a-retained-bind.v1.json"),
            "g4_task472_cc_url_suppression_supplement":
                binding(root,
                        "data/assembly/"
                        "g4-task472-cc-url-suppression-supplement.v1.json"),
            "reseal:task477_direct_capture_epoch4_chunks":
                binding(root,
                        "data/assembly/reseal-task477-direct-captures.v1.json"),
            # The ADR-101c supplement carries v1's five ADR-101b lines
            # byte-for-byte and adds the eight pairs folding the six post-481
            # fragments surfaces: four card pairs ADR-101b Ruling 1 governs by
            # construction, and the four ADR-101c was written for.  Its
            # ratification gate reads the ADR-101c heading -- the later
            # amendment, so no line can be admitted on an unratified ADR.
            "precedence_supplement":
                binding(root,
                        "data/r0-427-collision-ledger-adr101c-supplement.v1.json"),
            # v2 re-measures every row-ID collision the fold reports, not just
            # the 2 pairs / 17 rows that existed when task 481 wrote v1.  The
            # guard admits a pair only when the record covers every one of its
            # measured rows, so a record pinned to the August population would
            # silently fail six pairs that are in fact proven convergent.  v1
            # stays on disk and v2 names it by digest.
            "g7_convergent_identity":
                binding(root,
                        "data/assembly/g7-convergent-identity-rows.v2.json"),
            # ADR-101b/c row-level loser suppression (task 511 s3). This lived
            # only as a hand-edit of the committed plan until task 513, so the
            # next `build_assembly_plan.py` run would have silently dropped it
            # and staged 301,094 losing rows again -- the same defect as 511's
            # own "suppressed-ids never reached the stage dir", one layer up.
            # `role` is what `assembly_binder.cmd_plan` selects on; it used to
            # select on the `.gz` suffix, which made any new gzipped id list
            # registered here a silent deletion.
            "g5_adr101bc_row_suppression": {
                **binding(root,
                          "data/assembly/g5-adr101bc-row-suppression.v1.jsonl.gz"),
                "role": "row_suppression",
                "schema_note": G5_ROW_SUPPRESSION_NOTE,
            },
            # Which set's row survives when two sets stage the SAME id (task
            # 513). Consumed by `load --collision-winners`, never at stage time:
            # a shared id suppressed at stage would vanish from both sets.
            "g7_cross_set_collision_winners": {
                **binding(root,
                          "data/assembly/"
                          "g7-cross-set-collision-winners.v1.jsonl.gz"),
                "role": "collision_winners",
                "schema_note": G7_COLLISION_WINNERS_NOTE,
            },
        },
        "expected_sets": sorted(row["set_id"] for row in index["sets"]),
        "precedence": {
            "adr": "ADR-101", "ruling": "chunk-epoch-then-lineage",
            "table": dict(ADR101_TABLE),
            "amended_by": "ADR-101a (413-A keeps its 5,085; a URL bearing an "
                          "ADR-091 tombstone accepts no replacement rows)"},
        "disposition": {
            "replaced": replaced,
            "re_embedded_in_place": in_place,
            "tombstoned": tombstoned,
            "carried_legacy": 0,
            "pending": pending,
            "contested": contested,
        },
        "carried_legacy": {
            "count": 0,
            "note": "no point may keep a legacy 128-dim vector in the 1024 "
                    "build; every uncovered point is PENDING under a named "
                    "lane, not silently carried"},
        "contested": {
            "count": contested,
            "note": "points two contributing sets both claim with no "
                    "adjudicated winner. Folding them into `replaced` would "
                    "close the arithmetic while asserting an ownership nobody "
                    "has ruled on.",
            "pairs": unruled,
            "adjudicated": {
                "count": sum(ruled.values()),
                "pairs": ruled,
                "winners": {pair: rulings[frozenset(pair.split(" x "))]["winner"]
                            for pair in ruled},
                "note": "measured double claims that DO have a ratified winner. "
                        "They are not contested: each point is disposed to its "
                        "winner -- re_embedded_in_place where the winner is an "
                        "in-place set, replaced otherwise. Listed here so the "
                        "disposition can be audited pair by pair rather than "
                        "inferred from a total.",
                "supplement": PRECEDENCE_SUPPLEMENT,
            },
            "unlocated_rows": unlocated,
        },
        "quarantined_sets": quarantined,
        "pending_lanes": lanes,
        "pending_lane_reconciliation": lane_basis,
        "policy_escalations": [],
        "reconciliation": {
            "census_453_remaining": v31_remaining,
            "census_run": {"uri": str(args.census_v32_run),
                           "remaining_points": v32_remaining},
            "subtracted_by_witnesses_since_census_453":
                v31_remaining - v32_remaining,
            "folded_coverage_inside_census_remainder": covered,
            # Not "removed": the pairwise estimate that used to subtract this
            # is retired (`measured_covered_union` reads the fold's measured
            # union instead). It is reported because it is a measured fact
            # about the corpus, and because the delta against the retired
            # estimate is what tells a reader the two are not interchangeable.
            # NOTE: this line named an undefined `folded_pairs` until task 505.
            # It never fired, because the plan has refused on the lane-basis
            # residue every run since the estimate was retired -- a latent
            # NameError sitting behind a refusal is exactly the shape of a
            # defect that surfaces on the day the refusal clears.
            "folded_pair_overlap_points": folded_pair_overlaps,
            "retired_pairwise_estimate_of_covered": approximated,
            "per_set": per_set,
            "named_lane_total": lane_total,
            "lane_basis_gap": lane_total - pending,
            "lane_basis_unexplained_residue": lane_basis["unexplained_residue"],
        },
    }

    # the apply block carries forward, with the folded sets' collapse counts
    prior = json.loads(
        require_input(root, "data/assembly/assembly-plan.current.v1.json").read_text())
    plan["apply"] = prior["apply"]
    plan["apply"]["folded_identity_collapse"] = {
        row["set_id"]: int((row.get("identity_collapse") or {}).get("count", 0))
        for row in folded}

    # -- the two apply terms that were never computed ----------------------
    # Task 513b §5.5 named both: `apply.loaded_row_ids.distinct` and
    # `apply.delete_set.count` are read by guard G7 and NOTHING in the repo
    # wrote either -- the apply block was inherited verbatim, so G7 either
    # WAITED (no rows yet) or FAILED ("the plan does not declare the distinct
    # loaded-row-ID count"). They are computed here.
    #
    # `loaded_row_ids.distinct` is deliberately NOT the sum of the sets'
    # `rows_loaded`: G7 asserts those two are equal, so deriving one from the
    # other turns the guard into a mirror -- the round-number tell of a filter
    # that reports on what it selected. It is an INDEPENDENT measurement of the
    # ids the loader actually wrote (`load-summary.json`'s `distinct_ids`) or,
    # before the load, the preflight's `expected_points_count` over the staged
    # tree plus the suppression/exclusion artifacts. G7 then compares two
    # derivations that share no input, which is the only version of that check
    # worth running.
    if args.loaded_row_ids:
        doc = json.loads(Path(args.loaded_row_ids).read_text())
        # `.get(k, default)` returns the VALUE when the key exists, and a
        # resumed load now writes `distinct_ids: null` on purpose -- it did not
        # observe the ids, and null says so (task 515). Falling through on a
        # present-but-null key is the difference between "this artifact does
        # not measure it" and "this artifact measured nothing".
        distinct = doc.get("distinct_ids")
        if distinct is None:
            distinct = doc.get("expected_points_count")
        if distinct is None:
            raise SystemExit(
                f"{args.loaded_row_ids}: names neither `distinct_ids` (a load "
                "summary) nor `expected_points_count` (a load preflight)")
        plan["apply"]["loaded_row_ids"] = {
            "distinct": int(distinct),
            "measured_by": doc.get("schema") or "assembly_binder load",
            "source": str(args.loaded_row_ids),
            "basis": ("the ids the loader wrote"
                      if "distinct_ids" in doc else
                      "the preflight's expected count over the staged tree, "
                      "the exclusion artifact and the set-scoped row "
                      "suppression"),
        }

    # The delete set is every old production point a contributing set
    # supersedes, minus the ids that are also being LOADED -- those are upserted
    # in place (ADR-101 Ruling 2's G7 remedy) and deleting them would delete a
    # row this build is writing.
    #
    # It cannot be stated by adding per-set counts. The contributing sets' claims
    # OVERLAP -- 976,351 points under ADR-101 alone -- so the claim counts sum to
    # 35,281,002 against a measured union of 33,982,440, a 1,298,562 over-count,
    # and the identity-collapse counts have the same problem. So it is MEASURED
    # as a set difference by task 514 script measure_apply_terms.py, which
    # emits the ids as a digest-pinned artifact; this reads that measurement and
    # reconciles it against the plan's own membership-derived disposition, which
    # comes from a different input.
    if args.apply_terms:
        terms = json.loads(Path(args.apply_terms).read_text())
        disposition_superseded = (plan["disposition"]["replaced"]
                                  + plan["disposition"]["re_embedded_in_place"])
        excess = int(terms["claims_union"]) - disposition_superseded
        plan["apply"]["delete_set"] = {
            **{k: v for k, v in (prior["apply"].get("delete_set") or {}).items()
               if k != "pending_reason"},
            "count": int(terms["delete_set"]),
            "measured_by": terms["schema"],
            "artifact": terms["delete_set_artifact"],
            "basis": (
                f"measured set difference: the union of every contributing "
                f"replacement set's claims ({int(terms['claims_union']):,} "
                f"distinct old point ids) less the union of their "
                f"identity-collapse ids ({int(terms['identity_collapse_union']):,}"
                ", every one of which is inside the claims union)"),
            "claims_union": int(terms["claims_union"]),
            "identity_collapse_union": int(terms["identity_collapse_union"]),
            "reconciled_against_disposition": {
                "membership_minus_tombstoned": disposition_superseded,
                "claims_union_excess": excess,
                "note": ("the claims union exceeds `replaced + "
                         "re_embedded_in_place` because some claimed old points "
                         "are not in the membership this plan builds against, "
                         "and some are inside census bucket E; both are measured "
                         "by task 514 script explain_claims_union_excess.py "
                         "and reported in task 514 artifacts/"
                         "claims-union-excess.v1.json"),
            },
        }

    # The term is written only when there is something to put in it: an empty
    # `additions` block says exactly what its absence says, and a plan that
    # carries a bucket for a population it does not have invites the next
    # reader to look for one.
    if addition_entries:
        plan["additions"] = {
            "count": addition_rows,
            "collection_points_after_additions": membership + addition_rows,
            "sets": addition_entries,
            "note": "rows contributed by sets whose points were never in the "
                    "membership. The disposition above is an identity OVER the "
                    "membership, so none of its six buckets can hold them; "
                    "without this term the built collection holds "
                    f"{membership + addition_rows:,} points while the plan "
                    f"says {membership:,}.",
        }
    for entry in addition_entries:
        if entry.get("witness"):
            plan["records"][f"addition_witness:{entry['set_id']}"] = dict(
                entry["witness"])

    total = sum(int(value) for value in plan["disposition"].values())
    if total != membership:
        raise SystemExit(f"disposition sums to {total:,}, not {membership:,}")
    if addition_rows and any(entry["claimed_production_points"]
                             for entry in addition_entries):
        raise SystemExit(
            "an addition set claims a production point; the disposition and "
            "the additions term would both count it")
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    from cytir.ops import _provenance
    _provenance.write_provenance(args.out, "build-assembly-plan", _provenance.arg_inputs(args, args.out))
    print(json.dumps({"disposition": plan["disposition"],
                      "additions": {
                          "count": addition_rows,
                          "sets": [entry["set_id"]
                                   for entry in addition_entries],
                          "witnesses_bound": sum(
                              1 for entry in addition_entries
                              if entry.get("witness")),
                          "collection_points_after_additions":
                              membership + addition_rows},
                      "lanes": lane_total,
                      "reconciliation": plan["reconciliation"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
