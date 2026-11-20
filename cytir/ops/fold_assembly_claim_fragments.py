#!/usr/bin/env python3
"""Fold per-task claim-manifest fragments into the assembly claim index.

READ-ONLY over the corpus; writes only ``--out``.

Producers seal their own fragment (``data/assembly/claim-manifest-*.v1.json``)
and mark it ``pending_merge_into_claim-manifests.v1.json``.  This stage does the
merge -- but it is not a concatenation.  Three things have to happen first, and
each of them has already caught a real defect:

**1. Validate.**  A fragment that does not parse, or that omits a field the
ADR-101 contract requires, is refused by name.  It is not skipped.

**2. Normalize.**  The fragments use ``claims`` for the NEW point IDs a set
contributes.  ADR-101's contract uses ``claims`` for the OLD point IDs a set
binds as replaced (``claims: [old point ids | existing point ids]``).  Two
meanings for one field name is how an applier deletes the wrong 2.2M points.
Every fragment is rewritten to the contract with an explicit ``normalization``
record naming what moved, so the change is auditable rather than silent.

There are three ways a set can answer "which old points do you bind?", and all
three are branches here: an in-place re-embed binds its own IDs, a replacement
binds the points it supersedes, and a **net-new addition binds nothing**.  The
third is not the absence of an answer -- it is an answer, and until it had a
branch the fold reconstructed ``claims`` as the empty union and dropped the
addition's enumerated row IDs, after which G1 failed the set for publishing no
claim.  What an addition owes instead is the same enumeration under a digest,
over the IDs it actually owns, and the fold refuses one that arrives carrying
any old-point relationship at all.

**3. Measure, don't trust.**  Every count a fragment declares is re-derived
from the point-ID artifacts it pins: set sizes, the
``delete_set + identity_collapse = superseded`` conservation, pairwise
disjointness of both the old-point claims and the new row IDs across every
contributing set, and the ADR-091 tombstone intersection ADR-101a Ruling 2
turns on.  Counts that agree are reported as agreeing; counts that do not are
a hard failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

from cytir.ops.assembly_point_ids import (
    PointIdArtifactDefect, read_point_ids)

SCHEMA_MANIFESTS = "aw.assembly-claim-manifests.v1"
SCHEMA_MANIFEST = "aw.assembly-claim-manifest.v1"

#: A fragment must pin these, or it cannot be adjudicated.
REQUIRED = ("schema", "set_id", "mode", "contributes_vectors", "claims",
            "vector")

#: The lane this fold adjudicates: Track 1, the production-membership flip.
#: A fragment that declares no lane is a flip-lane fragment -- that was the
#: only kind when the field did not exist.
FLIP_LANE = "flip"

#: Lanes this fold knowingly does not adjudicate.  Track 2 (the 100M expansion
#: portfolio) writes `aw.assembly-claim-manifest.v1` fragments into the same
#: directory, but they claim no production point_id, supersede nothing, and are
#: tracked in their own ledger (`data/expansion-pack-admission-status.v1.json`).
#: Task 488's fragment says so in its own `lane_note`, in words -- and until
#: this gate existed the fold had no way to read it, so it tried to adjudicate
#: a set with no `new_point_ids` artifact and died on a KeyError.
#:
#: The gate is deliberately an allowlist of *known* lanes rather than a "skip
#: anything unfamiliar" rule: silently dropping a fragment is how a real claim
#: set goes missing from an assembly, which is exactly the failure the guards
#: exist to catch.  An unknown lane is a hard refusal.
NON_FLIP_LANES = {"expansion"}

#: A set whose rows were never membership points.  It supersedes nothing,
#: deletes nothing and claims no production point, so ADR-101's `claims` -- the
#: OLD point IDs a set binds as replaced -- is empty for it, and empty is the
#: TRUE answer rather than a missing one.  Before this mode had a branch here,
#: `normalize()` fell through to the reconstruct-from-delete-set case and
#: rebuilt `claims` as the empty union, dropping the fragment's enumerated row
#: IDs on the floor; guard G1 then failed the set for publishing no claim.
ADDITION_MODE = "net_new_addition"


def asserts_old_points(binding: dict | None) -> bool:
    """Does this binding assert a relationship to an existing production point?

    An explicit ``{"count": 0}`` does not -- a producer stating that its delete
    set is empty is saying the same thing as omitting it.  A uri, or any
    non-zero count, does.
    """
    binding = binding or {}
    return bool(binding.get("uri")) or int(binding.get("count") or 0) > 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ArtifactDefect(Exception):
    """A pinned artifact does not match what the producer published.

    Raised rather than exiting so the fold can quarantine one set and still
    report on the rest: a gate that dies on the first defect tells you about
    one problem per run, which is the slowest possible way to find four.
    """


def read_ids(root: Path, binding: dict, label: str) -> set[bytes]:
    """Read a pinned point-ID list, refusing digest drift.

    Delegates to `assembly_point_ids`, which is the ONE reader for both shapes
    the assembly's producers publish: `*.jsonl.gz`, and the bucket-witness
    `RECEIPT.json` that `gesetze` and `wikipedia_413a` use for their claims.
    Until task 513b this function knew only the first shape, so a fragment for
    either of those sets quarantined on a `BadGzipFile` -- and the reader for
    the second shape already existed in a task script. Two implementations of
    "what does this set claim?" is the mirror-contract defect, so there is one.
    """
    try:
        return read_point_ids(root, binding, label)
    except PointIdArtifactDefect as defect:
        raise ArtifactDefect(str(defect)) from defect


def load_fragment(path: Path) -> dict:
    raw = path.read_text()
    try:
        fragment = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path.name}: not valid JSON -- {exc}") from exc
    missing = [key for key in REQUIRED if key not in fragment]
    if missing:
        raise SystemExit(f"{path.name}: fragment omits {missing}")
    if fragment["schema"] != SCHEMA_MANIFEST:
        raise SystemExit(f"{path.name}: wrong schema {fragment['schema']!r}")
    return fragment


def _normalize_addition(fragment: dict, source_name: str,
                        new_ids_binding: dict, collapse: dict,
                        delete_set: dict, superseded_binding: dict | None,
                        notes: list[str]) -> tuple[dict, dict]:
    """Rewrite a net-new-addition fragment to the ADR-101 claim contract.

    The two older branches both answer the question "which OLD points does this
    set bind?".  An addition binds none, and that is a statement, not a gap --
    so this branch says it explicitly and keeps the fragment's enumerated row
    IDs, which the fall-through case silently discarded.

    What it will NOT do is let a set arrive here by mislabelling itself.  Every
    field below asserts a relationship to a point that already exists; a
    fragment carrying one and calling itself an addition is a replacement whose
    delete arithmetic nobody has checked, and the fold refuses it by name
    rather than normalizing it into something harmless-looking.
    """
    set_id = fragment["set_id"]
    for field, value in (("identity_collapse", collapse),
                         ("delete_set", delete_set),
                         ("old_point_ids_superseded", superseded_binding)):
        if asserts_old_points(value):
            raise SystemExit(
                f"{source_name}: {set_id} declares mode {ADDITION_MODE!r} and a "
                f"non-empty {field}. An addition's rows were never membership "
                "points, so it can supersede, collapse or delete nothing. This "
                "is a replacement that mislabelled itself, and the difference "
                "is whether the applier deletes live production points.")
    if fragment.get("contributes_production_points") is not False:
        raise SystemExit(
            f"{source_name}: {set_id} declares mode {ADDITION_MODE!r} but does "
            "not declare `contributes_production_points: false`. The claim that "
            "a set owns no production point is the whole basis on which G8 lets "
            "it publish no census witness; it must be stated, not inferred from "
            "an absent delete set.")
    if new_ids_binding.get("claims_are_production_points") is not False:
        raise SystemExit(
            f"{source_name}: {set_id} declares mode {ADDITION_MODE!r} but its "
            "`claims` binding does not declare "
            "`claims_are_production_points: false`. The fragments use `claims` "
            "for NEW point IDs and ADR-101 uses it for OLD ones; for an "
            "addition the two readings differ by the entire membership, so the "
            "fragment has to say which it means.")
    for field in ("uri", "sha256"):
        if not new_ids_binding.get(field):
            raise SystemExit(
                f"{source_name}: {set_id} declares mode {ADDITION_MODE!r} but "
                f"its row-ID artifact pins no {field}. An addition owes the "
                "same enumeration a replacement owes for its claim -- the same "
                "evidence, over the IDs it actually owns.")
    if int(new_ids_binding.get("count") or 0) <= 0:
        raise SystemExit(
            f"{source_name}: {set_id} declares mode {ADDITION_MODE!r} and zero "
            "row IDs. A set that adds nothing and claims nothing contributes "
            "nothing; cancel it rather than folding it.")

    notes.append(
        "claims := nothing (net-new addition): the set's rows were never "
        "membership points, so it binds no old point. The fragment's `claims` "
        "field holds its NEW row IDs and is kept as new_point_ids and as the "
        "`additions` term; it is NOT rebuilt from delete_set + "
        "identity_collapse, which for an addition is the empty set and would "
        "read as a set that claims nothing because it enumerated nothing.")

    manifest = {
        "schema": SCHEMA_MANIFEST,
        "set_id": set_id,
        "kind": fragment.get("kind"),
        "class": fragment.get("class"),
        "mode": ADDITION_MODE,
        "stage": fragment.get("stage"),
        "contributes_vectors": bool(fragment["contributes_vectors"]),
        "contributes_production_points": False,
        "vector": fragment["vector"],
        "new_point_ids": dict(new_ids_binding),
        "additions": {**new_ids_binding,
                      "rows": int((fragment["vector"] or {}).get("rows") or 0)},
        "membership_basis": fragment.get("membership_basis"),
        "identity_collapse": None,
        "delete_set": None,
        "claims": {
            "granularity": "point_id",
            "count": 0,
            "claims_are_production_points": False,
            "verified": True,
            "derived": "a net-new addition binds no old point: it supersedes "
                       "nothing, collapses nothing and deletes nothing",
        },
        "folded_from": source_name,
        "normalization": {
            "declared_mode": ADDITION_MODE,
            "effective_mode": ADDITION_MODE,
            "notes": notes,
        },
    }
    return manifest, {"set_id": set_id, "notes": notes}


def normalize(fragment: dict, source_name: str) -> tuple[dict, dict]:
    """Rewrite a producer fragment to the ADR-101 claim contract."""
    set_id = fragment["set_id"]
    notes: list[str] = []
    new_ids_binding = dict(fragment["claims"])
    collapse = fragment.get("identity_collapse") or {}
    delete_set = fragment.get("delete_set") or {}
    superseded_binding = fragment.get("old_point_ids_superseded")

    declared_mode = fragment["mode"]
    collapse_count = int(collapse.get("count") or 0)
    new_count = int(new_ids_binding.get("count") or 0)
    delete_count = delete_set.get("count")

    if declared_mode == ADDITION_MODE:
        return _normalize_addition(fragment, source_name, new_ids_binding,
                                   collapse, delete_set, superseded_binding,
                                   notes)

    # A set whose every new ID is also an old ID it "replaces", and which
    # deletes nothing, is an in-place re-embed however it labels itself.  The
    # distinction is the whole point of G7: an applier that reads
    # "replacement" may delete-then-insert and lose every one of them.
    effective_mode = declared_mode
    if (collapse_count == new_count and new_count > 0
            and (delete_count in (0, None))):
        if declared_mode != "in_place_reembed":
            effective_mode = "in_place_reembed"
            notes.append(
                f"mode corrected from {declared_mode!r} to 'in_place_reembed': "
                f"all {new_count:,} new point IDs equal the old ID they "
                "replace and the fragment declares no delete set. The "
                "fragment's own identity_collapse note says the same in words.")

    if effective_mode == "in_place_reembed" and superseded_binding:
        # An in-place re-embed is USUALLY claims == new ids, and when the
        # fragment says nothing else that is what we use. But "in place" does
        # not guarantee that every claimed point gets a row: wikipedia_413a
        # claims 19,227,117 and stages 19,227,106, because the projection
        # excluded 11. Taking claims := new ids there would silently shrink the
        # set's claim by exactly the points that are NOT being carried -- the
        # 11 would be neither loaded nor accounted as deleted, which is the
        # quiet-drop shape task 513 found for service_public_dila_exact at a
        # scale of 35,351. When the producer states its superseded set, that is
        # the claim, and the delete set is what the difference is for.
        claims_binding = dict(superseded_binding)
        notes.append(
            "claims := old_point_ids_superseded (in-place re-embed that states "
            "its own superseded set): the row ID and the old point ID are the "
            "same value for the rows it carries, and any claimed point it does "
            "NOT carry belongs in the delete set rather than vanishing from the "
            "claim")
    elif effective_mode == "in_place_reembed":
        claims_binding = dict(new_ids_binding)
        notes.append("claims := the set's own point IDs (in-place re-embed): "
                     "the row ID and the old point ID are the same value")
    elif superseded_binding:
        claims_binding = dict(superseded_binding)
        notes.append("claims := old_point_ids_superseded (the fragment's "
                     "`claims` field holds NEW point IDs, which the ADR-101 "
                     "contract calls the set's rows)")
    else:
        claims_binding = None
        notes.append("claims must be reconstructed as delete_set + "
                     "identity_collapse: the fragment publishes no "
                     "old_point_ids_superseded artifact")

    manifest = {
        "schema": SCHEMA_MANIFEST,
        "set_id": set_id,
        "kind": fragment.get("kind"),
        "class": fragment.get("class"),
        "mode": effective_mode,
        "stage": fragment.get("stage"),
        "contributes_vectors": bool(fragment["contributes_vectors"]),
        "vector": fragment["vector"],
        "new_point_ids": new_ids_binding,
        "identity_collapse": collapse or None,
        "delete_set": delete_set or None,
        "folded_from": source_name,
        "normalization": {
            "declared_mode": declared_mode,
            "effective_mode": effective_mode,
            "notes": notes,
        },
    }
    if claims_binding:
        # The producer's superseded-points artifact does not always carry a
        # granularity field; the contract requires one, and these are point IDs.
        claims_binding.setdefault("granularity", "point_id")
        claims_binding.setdefault("verified", True)
        manifest["claims"] = claims_binding
    return manifest, {"set_id": set_id, "notes": notes}



def partition_by_lane(paths: list[Path]) -> tuple[list[Path], list[dict]]:
    """Split discovered fragments into this fold's lane and the others.

    The fold discovers fragments by globbing ``claim-manifest-*.json``, and
    Track 2's expansion lane writes the same schema into the same directory.
    Those fragments claim no production point_id and pin no ``new_point_ids``
    artifact, so adjudicating one is not merely wrong, it is impossible.

    A fragment is excluded only when it *declares* a lane this fold knows it
    does not own.  An unknown lane value stops the run: a fold that skips what
    it does not recognise will one day skip a real claim set, and nothing
    downstream would show the gap -- the guards check the sets that are in the
    index, not the ones that should have been.
    """
    in_lane: list[Path] = []
    out_of_lane: list[dict] = []
    for path in paths:
        fragment = json.loads(path.read_text())
        lane = fragment.get("lane", FLIP_LANE)
        if lane == FLIP_LANE:
            in_lane.append(path)
            continue
        if lane not in NON_FLIP_LANES:
            raise SystemExit(
                f"{path.name}: declares lane {lane!r}, which this fold does "
                f"not recognise. Known lanes: {FLIP_LANE!r} (adjudicated here) "
                f"and {sorted(NON_FLIP_LANES)} (excluded by declaration). "
                "Refusing to guess -- a silently skipped fragment is a claim "
                "set missing from the assembly.")
        out_of_lane.append({
            "fragment": path.name,
            "set_id": fragment.get("set_id"),
            "lane": lane,
            "lane_note": fragment.get("lane_note"),
        })
    return in_lane, out_of_lane


def select_current_fragments(paths: list[Path]
                             ) -> tuple[list[Path], list[dict]]:
    """Pick the current fragment per set, honouring the re-seal convention.

    A broken seal is superseded, never overwritten: the producer's original
    stays on disk exactly as it left it and a later version sits beside it.
    So more than one fragment can name the same set, and the fold has to choose
    -- but not by trusting a filename.  The winner must SAY it supersedes the
    one it replaces, by name and by digest, or both are refused.  A silently
    higher version number is how the wrong claim set gets folded.
    """
    by_set: dict[str, list[tuple[int, Path, dict]]] = {}
    for path in paths:
        fragment = json.loads(path.read_text())
        version = int(str(path.name).rsplit(".v", 1)[-1].split(".")[0])
        by_set.setdefault(fragment["set_id"], []).append(
            (version, path, fragment))
    chosen: list[Path] = []
    resealed: list[dict] = []
    for set_id, entries in sorted(by_set.items()):
        entries.sort(key=lambda row: row[0])
        version, path, fragment = entries[-1]
        chosen.append(path)
        for _, older, _older_fragment in entries[:-1]:
            reseal = fragment.get("reseal") or {}
            if Path(str(reseal.get("supersedes") or "")).name != older.name:
                raise SystemExit(
                    f"{path.name} does not declare that it supersedes "
                    f"{older.name}; two fragments claim {set_id} and neither "
                    "names the other")
            actual = sha256_file(older)
            if reseal.get("supersedes_sha256") != actual:
                raise SystemExit(
                    f"{path.name} supersedes {older.name} at "
                    f"{reseal.get('supersedes_sha256')}, but that file is "
                    f"{actual}")
            resealed.append({"set_id": set_id, "superseded": older.name,
                             "superseded_sha256": actual,
                             "current": path.name,
                             "reason": reseal.get("reason")})
            print(f"[fold] {set_id:<40} {path.name} supersedes {older.name}",
                  flush=True)
    return chosen, resealed


def resolve_row_split(set_id: str, contested: int, rows_declared: int,
                      split: dict | None, wins_all_contested: bool = False) -> dict:
    """The G5 row-split fields for one set's `adjudication` (task 505 sec.6 /
    task 511 sec.2). Pure: no I/O, so directly testable without the full
    fold fixture tree.

    A row split is stated ONLY when `split` (this set's entry in a
    `--row-suppression` artifact, or None if the flag was omitted or the
    artifact is unratified) accounts for EVERY one of `contested`'s points --
    `int(split["covered_old_points"]) == contested`, an exact match, not a
    lower bound. `covered_old_points` is win-or-lose: a set contested in TWO
    pairs, winning one and losing the other (task492 wins vs task427 and
    loses to task467), is only fully resolved once BOTH pairs are ruled --
    `lost_old_points` alone would never reach `contested` for such a set,
    because it only ever counts the losing side. Partial coverage is treated
    exactly like no coverage: a partial split is a wrong split with extra
    steps.
    """
    if not contested:
        return {"rows_loaded": rows_declared, "rows_suppressed": 0}
    if wins_all_contested:
        # The set loses no contested pair, so its win is total: every declared
        # row loads and none is suppressed. This is the ratified precedence's
        # own consequence (ADR-101 Ruling 1 / ADR-101b/c as class rules), not a
        # waiver of the row split -- there is simply no losing row to list.
        return {
            "rows_loaded": rows_declared,
            "rows_suppressed": 0,
            "row_split_basis": (
                f"this set wins all {contested:,} of its contested old point(s) "
                "under ratified precedence (ADR-101 Ruling 1 / ADR-101b/c, "
                "enumerated pair or class rank), so it suppresses no row"),
        }
    if split and int(split["covered_old_points"]) == contested:
        if int(split["rows_declared"]) != rows_declared:
            raise ArtifactDefect(
                f"{set_id}: --row-suppression declares "
                f"{int(split['rows_declared']):,} rows, this fold measures "
                f"{rows_declared:,} -- stale row-suppression artifact")
        return {
            "rows_loaded": int(split["rows_loaded"]),
            "rows_suppressed": int(split["rows_suppressed"]),
            "row_split_basis": (
                "ADR-101b/c (data/r0-427-collision-ledger-adr101c-supplement"
                ".v1.json), row-level: adr101_row_suppression.py joined the "
                f"{contested:,} contested old point(s) to this set's own "
                f"rows by url ({split['lost_urls']:,} distinct lost url(s), "
                f"{split['pairs']} pair(s)); every contested point accounted "
                "for -- see --row-suppression for the per-row list"),
        }
    out = {"rows_pending_adjudication": rows_declared}
    if split:
        out["why_no_row_split"] = (
            f"--row-suppression resolves {int(split['covered_old_points']):,} "
            f"of this set's {contested:,} contested old point(s); partial "
            "coverage does not state a row split")
    else:
        out["why_no_row_split"] = (
            f"{contested:,} of this set's claimed old points are also "
            "claimed by another contributing set with no adjudicated "
            "winner; the row split cannot be stated until that is ruled")
    return out


def apply_measured_load_suppression(index: dict, path) -> list[dict]:
    """Charge every set the rows the LOAD will not write for it.

    A set's `rows_loaded` is a claim about the collection, and until task 514 it
    was derived only from the precedence mechanism that happened to be wired:
    the ADR-101b/c pairs (task 511 §2) and the atomic 1:1 rule
    (`bind_assembly_vector_rollups._derive_row_split`).  Two other mechanisms
    remove a staged row and neither reached the accounting -- the cross-set
    collision winner map (task 513 §2, which task 513 §5.5 flagged as owed) and
    ADR-101 Ruling 1's own row-level suppression, which task 511 §3 deferred for
    every set including task 427 itself.  The result was a plan that declared
    16,817 rows more loaded than any loader would write, and G7's identity
    (`distinct loaded ids == sum(rows_loaded)`) cannot be true and honest at the
    same time while that is so.

    The input is the measurement, per set, over the staged tree the load reads:
    `task 514 script reconcile_load_suppression.py`, which computes
    `rows_loaded = |staged(S)| - |collision_losers(S) ∪ suppression(S)|` as ID
    SETS.  It is not trusted blindly: every entry must agree with THIS fold
    about the set's current `rows_loaded` before its correction is applied, so a
    measurement taken against a different fold is refused rather than absorbed.
    """
    if not path:
        return []
    doc = json.loads(Path(path).read_text())
    by_id = {row["set_id"]: row for row in index["sets"]}
    out = []
    for row in doc["sets"]:
        set_id = row["set_id"]
        entry = by_id.get(set_id)
        if entry is None:
            raise ArtifactDefect(
                f"--measured-load-suppression names {set_id!r}, which this fold "
                "does not carry")
        adj = entry.setdefault("adjudication", {})
        was = int(adj.get("rows_loaded", -1))
        if was != int(row["rows_loaded_declared_by_fold"]):
            raise ArtifactDefect(
                f"{set_id}: the measurement was taken against a fold that "
                f"declared {int(row['rows_loaded_declared_by_fold']):,} rows "
                f"loaded; this fold declares {was:,}. Re-measure rather than "
                "applying a stale correction.")
        now = int(row["rows_loaded_computed"])
        if now == was:
            continue
        declared = int(adj["rows_declared"])
        adj["rows_loaded"] = now
        adj["rows_suppressed"] = declared - now
        adj["row_split_basis"] = (
            f"measured over the staged tree: {int(row['staged']):,} staged row(s) "
            f"less {int(row['not_loaded']):,} the load will not write "
            f"({int(row['collision_losers']):,} cross-set collision loser(s), "
            f"{int(row['adr101_r1']):,} forfeited under ADR-101 Ruling 1, "
            f"{int(row['adr101bc_still_staged']):,} under ADR-101b/c, as ID "
            "sets and not as a sum) -- "
            "task 514 script reconcile_load_suppression.py")
        out.append({"set_id": set_id, "rows_loaded_before": was,
                    "rows_loaded_after": now,
                    "rows_suppressed_after": declared - now})
        print(f"[fold] measured-load correction {set_id:<40} "
              f"{was:,} -> {now:,} rows loaded", flush=True)
    return out


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--fragments", type=Path, required=True)
    parser.add_argument("--named-slice-claims", type=Path,
                        help="output of produce_named_slice_claims.py")
    parser.add_argument("--task427-claims", required=True,
                        help="corpus-relative admitted-old-points.jsonl.gz")
    parser.add_argument("--task427-claims-sha256", required=True)
    parser.add_argument("--tombstone-claims", required=True,
                        help="corpus-relative ADR-091 tombstone point IDs")
    parser.add_argument("--task427-witness", type=Path,
                        help="RUN.json of the minted task-427 census witness")
    parser.add_argument("--fresh-witness", action="append", default=[],
                        metavar="SET_ID=/absolute/RUN.json",
                        help="a census point witness minted for a set the "
                             "census has not consumed yet. Repeatable; the "
                             "--task427-witness flag is the same thing with "
                             "the set_id hard-coded, kept because task 462's "
                             "reproduction line names it.")
    parser.add_argument("--remainder",
                        help="corpus-relative census remaining-points.jsonl.gz")
    parser.add_argument("--row-suppression", type=Path, default=None,
                        help="a g5-row-suppression.v1.json file (task 511: "
                             "the derived suppression, one entry per ruled "
                             "pair) -- the ADR-101b/c row-level split for "
                             "sets whose ownership IS ruled. Absent means "
                             "every contested set stays "
                             "rows_pending_adjudication, exactly as before "
                             "this flag existed.")
    parser.add_argument("--supersede", action="append", default=[],
                        metavar="OLD_SET_ID=NEW_SET_ID",
                        help="an index entry the fold replaces, stated "
                             "explicitly; guessing which pending entry a "
                             "fragment fulfils is how the wrong set gets "
                             "dropped")
    parser.add_argument("--measured-load-suppression", dest="measured_load",
                        type=Path, default=None,
                        help="task 514 script reconcile_load_suppression.py "
                             "output: the per-set rows the load will not write, "
                             "measured as ID sets over the staged tree")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    root: Path = args.corpus_root
    index = json.loads(args.index.read_text())
    if index.get("schema") != SCHEMA_MANIFESTS:
        raise SystemExit("index has the wrong schema")
    existing = {row["set_id"]: row for row in index["sets"]}

    # ---- load and normalize every fragment -------------------------------
    fragments = sorted(args.fragments.glob("claim-manifest-*.json"))
    if not fragments:
        raise SystemExit(f"no fragments under {args.fragments}")
    fragments, out_of_lane = partition_by_lane(fragments)
    for row in out_of_lane:
        print(f"[fold] OUT OF LANE {row['set_id']:<40} lane={row['lane']}"
              f"  ({row['fragment']})", flush=True)
    if not fragments:
        raise SystemExit(
            f"every fragment under {args.fragments} declares a non-flip lane")
    fragments, resealed = select_current_fragments(fragments)
    folded: list[dict] = []
    normalizations: list[dict] = []
    for path in fragments:
        fragment = load_fragment(path)
        manifest, note = normalize(fragment, path.name)
        folded.append(manifest)
        normalizations.append(note)
        print(f"[fold] {manifest['set_id']:<40} mode={manifest['mode']:<18}"
              f" rows={int(manifest['vector'].get('rows') or 0):>9,}",
              flush=True)

    # ---- read every point-ID set and re-derive the counts ----------------
    claims: dict[str, set[bytes]] = {}
    rows: dict[str, set[bytes]] = {}
    findings: list[str] = []
    conservation: dict[str, dict] = {}

    quarantined: dict[str, list[str]] = {}
    admitted: list[dict] = []
    for manifest in folded:
        set_id = manifest["set_id"]
        try:
            rows[set_id] = read_ids(root, manifest["new_point_ids"],
                                    f"{set_id}.new_point_ids")
            collapse = manifest.get("identity_collapse") or {}
            deletes = manifest.get("delete_set") or {}
            collapse_ids = (read_ids(root, collapse,
                                     f"{set_id}.identity_collapse")
                            if collapse.get("uri") else set())
            delete_ids = (read_ids(root, deletes, f"{set_id}.delete_set")
                          if deletes.get("uri") else set())
            if manifest.get("claims", {}).get("uri"):
                claims[set_id] = read_ids(root, manifest["claims"],
                                          f"{set_id}.claims")
        except ArtifactDefect as defect:
            quarantined.setdefault(set_id, []).append(str(defect))
            # Re-measure every artifact the set pins, so one run produces the
            # whole picture instead of one line of it.
            for field in ("new_point_ids", "claims", "identity_collapse",
                          "delete_set"):
                binding = manifest.get(field) or {}
                if not binding.get("uri") or not binding.get("sha256"):
                    continue
                path = root / str(binding["uri"]).removeprefix("artifact://")
                if not path.is_file():
                    continue
                actual = sha256_file(path)
                if actual != binding["sha256"]:
                    size = path.stat().st_size
                    declared = binding.get("bytes")
                    delta = (f", delta {size - int(declared):+d} bytes"
                             if declared else " (no byte count published)")
                    quarantined[set_id].append(
                        f"{field}: published {binding['sha256'][:16]}..., "
                        f"file is {actual[:16]}... ({size:,} bytes){delta}")
            rows.pop(set_id, None)
            claims.pop(set_id, None)
            findings.append(
                f"{set_id}: QUARANTINED -- its pinned artifacts do not match "
                "what the producer published; the set does not enter the "
                "assembly")
            print(f"[fold] {set_id:<40} QUARANTINED", flush=True)
            continue
        admitted.append(manifest)
        if manifest["mode"] == ADDITION_MODE:
            # An addition claims no old point.  Recording the empty set here
            # rather than skipping the set is what puts it into the pairwise
            # G2/G4 intersections below -- vacuously clean, but MEASURED clean.
            claims[set_id] = set()
            declared_rows = int((manifest.get("vector") or {}).get("rows") or 0)
            minted = len(rows[set_id])
            enumerated = int(manifest["new_point_ids"].get("count") or 0)
            conserves = declared_rows == minted == enumerated
            if not conserves:
                findings.append(
                    f"{set_id}: a net-new addition must satisfy rows == minted "
                    f"IDs == vectors; it declares {declared_rows:,} vector "
                    f"rows, enumerates {enumerated:,} row IDs and {minted:,} "
                    "are actually on disk")
            conservation[set_id] = {
                "superseded": 0, "delete_set": 0, "identity_collapse": 0,
                "addition_vector_rows": declared_rows,
                "addition_minted_ids": minted,
                "addition_enumerated_ids": enumerated,
                "conserves": conserves,
            }
            print(f"[fold] {set_id:<40} claims={0:>9,} "
                  f"rows={minted:>9,}  (net-new addition)", flush=True)
            continue
        if not manifest.get("claims", {}).get("uri"):
            claims[set_id] = collapse_ids | delete_ids
            manifest["claims"] = {
                "granularity": "point_id",
                "count": len(claims[set_id]),
                "derived": "delete_set UNION identity_collapse",
                "verified": True,
            }
        superseded = claims[set_id]
        # ONE conservation identity for both modes: every superseded point is
        # either reclaimed under its own id (identity_collapse) or removed
        # (delete_set), and never both. The in-place branch used to state this
        # as `conserves = not delete_ids` -- "in place means nothing is
        # deleted" -- which is the same identity only for a set that carries
        # ALL of its claims. wikipedia_413a carries 19,227,106 of 19,227,117
        # because the projection excluded 11, and under the old rule those 11
        # could never be accounted: the set was either non-conserving forever,
        # or its claim had to shrink to hide them. The union identity covers
        # both cases, and reduces to the old rule when the delete set is empty.
        if manifest["mode"] != "replacement" and collapse_ids and \
                collapse_ids != rows[set_id]:
            findings.append(
                f"{set_id}: an in-place re-embed must collapse every row "
                f"ID; {len(rows[set_id] - collapse_ids):,} do not")
        overlap = collapse_ids & delete_ids
        if overlap:
            findings.append(
                f"{set_id}: {len(overlap):,} IDs are in BOTH the delete set "
                "and the identity-collapse set; the applier would delete a "
                "point it is also loading")
        union = collapse_ids | delete_ids
        if union != superseded:
            findings.append(
                f"{set_id}: delete_set UNION identity_collapse "
                f"({len(union):,}) != superseded ({len(superseded):,}); "
                f"{len(superseded - union):,} superseded points are neither "
                "deleted nor reclaimed")
        conservation[set_id] = {
            "superseded": len(superseded),
            "delete_set": len(delete_ids),
            "identity_collapse": len(collapse_ids),
            "conserves": union == superseded and not overlap,
        }
        print(f"[fold] {set_id:<40} claims={len(superseded):>9,} "
              f"rows={len(rows[set_id]):>9,}", flush=True)

    # ---- task 427 and the tombstones -------------------------------------
    task427 = read_ids(root, {"uri": args.task427_claims,
                              "sha256": args.task427_claims_sha256},
                       "task427_postconsent.claims")
    claims["task427_postconsent"] = task427
    tombstones = read_ids(root, {"uri": args.tombstone_claims},
                          "natural_person_policy_drop.tombstones")
    print(f"[fold] task427 claims={len(task427):,} "
          f"tombstones={len(tombstones):,}", flush=True)

    # ---- G4: no winning claim may touch a tombstoned point ---------------
    tombstone_hits = {set_id: len(ids & tombstones)
                      for set_id, ids in claims.items()}
    for set_id, count in sorted(tombstone_hits.items()):
        if count and set_id != "task427_postconsent":
            findings.append(
                f"{set_id}: claims {count:,} ADR-091 tombstoned point(s); "
                "ADR-101 Ruling 1.3 gives a tombstoned point to nobody")

    # ---- G2: pairwise disjointness of old-point claims -------------------
    claim_overlaps = {}
    # Per set, the UNION of the points it shares with any other set -- not the
    # sum of its pair counts.  A point three sets claim is one contested point,
    # and summing pairs counts it twice for each of them.  Task 498 sec.7 named
    # this defect in the fold's `points_won` and it was still live here: with
    # summed pairs `task494_4crawl_supplement_cc_chunks` reported
    # **points_won = -883**, a negative population that would have been sealed
    # into a census witness scope.  Measured as a set union it cannot go
    # negative, because the contested set is a subset of the claims.
    contested_ids: dict[str, set] = {}
    for left, right in combinations(sorted(claims), 2):
        shared = claims[left] & claims[right]
        if shared:
            claim_overlaps[f"{left} x {right}"] = len(shared)
            contested_ids.setdefault(left, set()).update(shared)
            contested_ids.setdefault(right, set()).update(shared)
            findings.append(
                f"{left} and {right} both claim {len(shared):,} old point(s); "
                "every double claim needs an adjudicated winner")

    # ---- G7: row IDs must be globally distinct ---------------------------
    row_overlaps = {}
    for left, right in combinations(sorted(rows), 2):
        shared = rows[left] & rows[right]
        if shared:
            row_overlaps[f"{left} x {right}"] = len(shared)
            findings.append(
                f"{left} and {right} emit {len(shared):,} identical row ID(s)")

    # ---- an addition's minted IDs must not already belong to anybody ------
    # The pairwise loops above prove the minted IDs collide with no other
    # set's ROWS.  That is not the same question as whether one of them is
    # already a production point: an ID that equals an old point some other
    # set claims would be silently upserted over a live vector, and an ID that
    # equals a tombstone would resurrect a point ADR-091 dropped.  Both are
    # cheap to measure here, where every ID set is already in memory.
    addition_ids = {manifest["set_id"] for manifest in admitted
                    if manifest["mode"] == ADDITION_MODE}
    additions_report: dict[str, dict] = {}
    for set_id in sorted(addition_ids):
        minted = rows[set_id]
        claim_hits = {}
        for other, ids in sorted(claims.items()):
            if other == set_id or not ids:
                continue
            shared = minted & ids
            if shared:
                claim_hits[other] = len(shared)
                findings.append(
                    f"{set_id}: {len(shared):,} minted row ID(s) are old points "
                    f"{other} claims; a net-new addition's IDs must belong to "
                    "no production point, and an applier would upsert over a "
                    "live vector")
        tombstone_hits_here = len(minted & tombstones)
        if tombstone_hits_here:
            findings.append(
                f"{set_id}: {tombstone_hits_here:,} minted row ID(s) are "
                "ADR-091 tombstoned points; an addition may not resurrect a "
                "point the policy drop removed")
        additions_report[set_id] = {
            "rows": len(minted),
            "claims_production_points": 0,
            "minted_ids_in_another_sets_claims": claim_hits,
            "minted_ids_tombstoned": tombstone_hits_here,
            "minted_ids_in_census_remainder": None,
            "conserves": bool(conservation.get(set_id, {}).get("conserves")),
        }

    # ---- where the newly claimed points sat in the census ----------------
    remainder_containment = {}
    remainder_binding = None
    if args.remainder:
        remainder = read_ids(root, {"uri": args.remainder}, "census.remainder")
        remainder_binding = {
            "uri": args.remainder,
            "sha256": sha256_file(root / str(args.remainder).removeprefix(
                "artifact://")),
            "points": len(remainder),
        }
        print(f"[fold] census remainder {len(remainder):,}", flush=True)
        covered_union: set[str] = set()
        for set_id, ids in sorted(claims.items()):
            inside_ids = ids & remainder
            inside = len(inside_ids)
            covered_union |= inside_ids
            remainder_containment[set_id] = {
                "claims": len(ids), "inside_adr092_remainder": inside,
                "outside": len(ids) - inside}
        # The TRUE union, measured once, here, where every ID set is already in
        # memory.  Its consumer -- `build_assembly_plan.py` -- previously
        # approximated it as (sum of per-set `inside`) minus (sum of PAIRWISE
        # claim-overlap counts).  That approximation is wrong in two directions
        # at once and both are live on this corpus:
        #
        #   * inclusion-exclusion truncated at pairs over-subtracts as soon as
        #     three sets share a point, and
        #   * the pair counts are overlaps of CLAIMS, not of claims *inside the
        #     remainder*.  task427 contributes 0 to the sum (none of its claims
        #     are inside) yet four pairs name it, so 40,400 points get
        #     subtracted that were never added.
        #
        # This is the same defect task 498 sec.7 found in `points_won` -- summing
        # pair counts where the code means to union point sets -- in a second
        # consumer.  Publishing the measured union removes the estimate rather
        # than correcting it.
        remainder_binding["covered_union"] = len(covered_union)
        print(f"[fold] folded sets cover {len(covered_union):,} of the "
              f"remainder (measured union, not inclusion-exclusion)", flush=True)
        # The census remainder is a subset of the membership, so a minted ID
        # found inside it is proof the addition is not net-new.  This does not
        # prove the converse -- a full membership join is the addition
        # witness's job, and G7 refuses an addition that has no witness -- but
        # it is a real intersection over real IDs, and it is free here.
        for set_id in sorted(addition_ids):
            hits = len(rows[set_id] & remainder)
            additions_report[set_id]["minted_ids_in_census_remainder"] = hits
            if hits:
                findings.append(
                    f"{set_id}: {hits:,} minted row ID(s) are inside the census "
                    "remainder, so they are membership points and the set is "
                    "not a net-new addition")

    # ---- adjudicate what can be adjudicated ------------------------------
    # A set with no contested point owns everything it claims, so every row it
    # declares loads.  A set WITH a contested point gets a row split ONLY when
    # --row-suppression names a ratified, re-derived resolution that accounts
    # for EVERY one of its contested points (task 505 sec.6 / task 511 sec.2:
    # ADR-101b/c rule the WINNER at point-id level; the --row-suppression
    # file carries the ROW-level consequence (a url-join) and is cross-checked here
    # against this fold's OWN contested count, not trusted blind).  Partial
    # coverage -- the supplement resolves some but not all of a set's
    # contested points -- still declares no split: a partial split is a wrong
    # split with extra steps, and guard G5 is supposed to notice it cannot yet
    # be stated.
    contested_by_set = {set_id: len(ids) for set_id, ids in contested_ids.items()}
    row_suppression = {}
    if args.row_suppression and args.row_suppression.is_file():
        row_suppression_doc = json.loads(args.row_suppression.read_text())
        if row_suppression_doc.get("ratified"):
            row_suppression = row_suppression_doc.get("sets") or {}

    # A contested set that LOSES no pair suppresses no row: its win is total, so
    # rows_loaded = rows_declared and rows_suppressed = 0, no per-row list
    # needed. Task 505 §6 / 511 §5 both stated this for the 13 payload-route
    # sets ("all 13 win 100% of their claimed points already ... rows_suppressed
    # = 0 trivially"), but the fold could not act on it because it only ever
    # consumed the row-suppression artifact, which lists LOSERS. The winner of a
    # contested pair is decided by the SAME two artifacts the guard uses (the
    # enumerated supplement, then the ratified class ranks), imported rather
    # than re-implemented so a set the fold treats as a total winner is a set
    # the guard also finds ruled that way.
    from cytir.ops.assembly_guards import class_rule_rulings  # noqa: PLC0415
    from cytir.ops.build_assembly_plan import adjudicated_pairs  # noqa: PLC0415

    # The class-precedence table and the supplement live in the REPO (data/
    # assembly/), not under --corpus-root; the fold runs from the repo, so the
    # repo root is this file's grandparent.
    root_repo = Path(__file__).resolve().parents[2]

    class _Ctx:
        pass
    _ctx = _Ctx()
    _ctx.root = root_repo
    # The class of each set decides the class-rank rulings, and the sets being
    # folded THIS run are not yet merged into `existing` at this point -- they
    # are in `admitted`. Both are needed, or the newly-folded sets (which is
    # exactly the population this whole task binds) have no class and every
    # pair they are in reads as unruled.
    _class_sets = {**existing, **{m["set_id"]: m for m in admitted}}
    _ctx.manifests = {"sets": list(_class_sets.values())}
    _ctx.records = {"fold_report": {"claim_overlaps": claim_overlaps}}
    # Class ranks are ratified data by construction (ADR-101/b/c are merged), so
    # they apply unconditionally -- exactly as the guard's G2 applies them. The
    # ENUMERATED supplement lines carry their own draft-marker gate, evaluated
    # by `adjudicated_pairs` (the plan's own reader, reused so the three sites
    # agree). Both feed one winner map; enumerated lines win a tie.
    _rulings = dict(class_rule_rulings(_ctx, set(claim_overlaps)))
    _rulings.update(adjudicated_pairs(root_repo))
    loses_a_pair: set[str] = set()
    in_unruled_pair: set[str] = set()
    for pair in claim_overlaps:
        left, _, right = pair.partition(" x ")
        line = _rulings.get(frozenset({left, right}))
        if line is None:
            # neither set wins here -- both stay pending until this pair is ruled
            in_unruled_pair.update({left, right})
            continue
        loser = left if line["winner"] == right else right
        loses_a_pair.add(loser)
    for manifest in admitted:
        set_id = manifest["set_id"]
        contested = contested_by_set.get(set_id, 0)
        rows_declared = int((manifest.get("vector") or {}).get("rows") or 0)
        adjudication = {
            "claims": len(claims.get(set_id, ())),
            "contested_with_another_set": contested,
            "contested_with_task427": int(claim_overlaps.get(
                f"task427_postconsent x {set_id}",
                claim_overlaps.get(f"{set_id} x task427_postconsent", 0))),
            "winner_on_contested": None if contested else "this_set",
            "points_won": len(claims.get(set_id, ())) - contested,
            "contested_basis": "union of the point sets this set shares with "
                               "any other contributing set; NOT the sum of its "
                               "pair counts, which double-counts a point three "
                               "sets claim and can drive points_won negative",
            "points_lost_to_task427": 0,
            "rows_declared": rows_declared,
        }
        adjudication.update(resolve_row_split(
            set_id, contested, rows_declared, row_suppression.get(set_id),
            wins_all_contested=(contested > 0 and set_id not in loses_a_pair
                                and set_id not in in_unruled_pair
                                and set_id not in row_suppression)))
        manifest["adjudication"] = adjudication

    # ---- fold the named-slice claims into the existing sets --------------
    slice_bound = 0
    if args.named_slice_claims and args.named_slice_claims.is_file():
        slices = json.loads(args.named_slice_claims.read_text())
        by_set = {row["set_id"]: row for row in slices["claims"]}
        for row in index["sets"]:
            name = str(row["set_id"]).removeprefix("named_slice:")
            binding = by_set.get(name)
            if binding is None:
                continue
            row["claims"] = {
                "granularity": "point_id",
                "uri": binding["uri"], "sha256": binding["sha256"],
                "bytes": binding.get("bytes"), "count": binding["count"],
                "verified": True,
                "derived_from": binding.get("derived_from"),
            }
            slice_ids = read_ids(root, binding, f"named_slice.{name}")
            contested = len(slice_ids & task427)
            row["adjudication"] = {
                **(row.get("adjudication") or {}),
                "contested_with_task427": contested,
                "winner_on_contested": "this_set",
                "points_won": binding["count"],
                "points_lost_to_task427": 0,
            }
            row["census_witness"] = {
                "uri": binding["uri"], "sha256": binding["sha256"],
                "records": binding["count"],
                "shape": "membership/points.jsonl.gz",
                "consumed_by_census":
                    "r0-remainder-census-v31 (payload predicate, not a witness "
                    "artifact -- this enumeration selects the same points)",
                "scope": "all_claims",
                "note": "enumerated from the payload predicate the census "
                        "already applies; it selects exactly the same points, "
                        "so it subtracts exactly what the census subtracts",
            }
            slice_bound += 1
            print(f"[fold] named slice {name:<24} "
                  f"{binding['count']:>9,} ids bound, x427={contested:,}",
                  flush=True)

    # ---- merge -----------------------------------------------------------
    superseded_entries: dict[str, str] = {}
    for spec in args.supersede:
        old, separator, new = spec.partition("=")
        if not separator:
            raise SystemExit("--supersede needs OLD_SET_ID=NEW_SET_ID")
        if old not in existing:
            raise SystemExit(f"--supersede names an unknown index entry: {old}")
        if new not in {row["set_id"] for row in admitted}:
            raise SystemExit(f"--supersede names an unknown fragment: {new}")
        superseded_entries[old] = new
    for manifest in admitted:
        # A fragment REPLACES its index entry wholesale, which is right for
        # everything the fragment is authoritative about -- but the census
        # witness is not one of those things. It is bound either by
        # --fresh-witness on this run or by whoever produced the index entry
        # (`produce_assembly_artifacts.py`, for the sealed families the census
        # already consumed at v31). Dropping it turns G8 census-witness parity
        # from PASS to FAIL for a set whose only change was gaining row IDs --
        # exactly the "an incomplete fragment is worse than no fragment" trap
        # task 511 §5 named. Carry it forward, loudly, and never overwrite one
        # the fragment or a --fresh-witness supplies.
        prior = existing.get(manifest["set_id"])
        if (prior and prior.get("census_witness")
                and not manifest.get("census_witness")):
            manifest["census_witness"] = prior["census_witness"]
            print(f"[fold] {manifest['set_id']:<40} census witness carried "
                  f"forward from the index "
                  f"({prior['census_witness'].get('records')} records)",
                  flush=True)
        existing[manifest["set_id"]] = manifest
    for pending_id, replacement in superseded_entries.items():
        row = existing.pop(pending_id, None)
        if row is not None:
            existing[replacement]["supersedes"] = {
                "set_id": pending_id,
                "why": "the same population, now embedded and verified",
                "prior_claims": (row.get("claims") or {}).get("count"),
            }
            print(f"[fold] {replacement} supersedes pending {pending_id}",
                  flush=True)

    fresh_witnesses: list[tuple[str, Path]] = []
    if args.task427_witness and args.task427_witness.is_file():
        fresh_witnesses.append(("task427_postconsent", args.task427_witness))
    for spec in args.fresh_witness:
        set_id, separator, path = spec.partition("=")
        if not separator:
            raise SystemExit("--fresh-witness needs SET_ID=/absolute/RUN.json")
        fresh_witnesses.append((set_id, Path(path)))
    for set_id, path in fresh_witnesses:
        witness = json.loads(path.read_text())
        stage = witness["stages"]["identity"]
        # The stage block points at its receipt; the artifact inventory lives
        # inside the receipt, which is the thing the census actually reads.
        receipt = json.loads(
            (root / stage["receipt"]["uri"].removeprefix("artifact://")).read_text())
        artifact = receipt["artifacts"][0]
        # The witness must say it is about this set. Binding a witness to the
        # wrong set_id is how the census subtracts one population and the guard
        # credits another.
        if str(witness.get("source_id")) != set_id:
            raise SystemExit(
                f"--fresh-witness binds {set_id} to a witness whose source_id "
                f"is {witness.get('source_id')!r}")
        row = existing.get(set_id)
        if row is None:
            raise SystemExit(f"--fresh-witness names an unknown set: {set_id}")
        row["census_witness"] = {
            "uri": artifact["uri"],
            "sha256": artifact["sha256"],
            "records": int(stage["records"]),
            "shape": "membership/points.jsonl.gz",
            "scope": "unwitnessed_winning_subset",
            "winning_points_total": int(
                witness["counts"]["winning_points_total"]),
            "consumed_by_census": None,
            "note": witness["identity_scope"],
        }
        print(f"[fold] {set_id} census witness bound: "
              f"{stage['records']:,} points", flush=True)

    index["sets"] = [existing[key] for key in sorted(existing)]
    measured = apply_measured_load_suppression(index, args.measured_load)
    index["totals"] = {
        "sets": len(index["sets"]),
        "contributing": sum(1 for row in index["sets"]
                            if row.get("contributes_vectors")),
        "point_id_granularity": sum(
            1 for row in index["sets"]
            if (row.get("claims") or {}).get("granularity") == "point_id"),
        "cancelled": sum(1 for row in index["sets"]
                         if row.get("mode") == "cancelled"),
        "net_new_addition_sets": sum(1 for row in index["sets"]
                                     if row.get("mode") == ADDITION_MODE),
        "net_new_addition_rows": sum(
            int((row.get("vector") or {}).get("rows") or 0)
            for row in index["sets"] if row.get("mode") == ADDITION_MODE),
    }
    index["fold"] = {
        "producer": "aw ops fold-assembly-claim-fragments",
        "fragments": [path.name for path in fragments],
        "out_of_lane_fragments": out_of_lane,
        "named_slice_claims_bound": slice_bound,
        "superseded_entries": superseded_entries,
        "measured_load_corrections": measured,
    }
    args.out.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    report = {
        "schema": "aw.assembly-fold-report.v1",
        "fragments": [path.name for path in fragments],
        "out_of_lane_fragments": out_of_lane,
        "resealed": resealed,
        "normalizations": normalizations,
        "conservation": conservation,
        "additions": additions_report,
        "tombstone_intersection": tombstone_hits,
        "claim_overlaps": claim_overlaps,
        "row_id_overlaps": row_overlaps,
        "remainder_containment": remainder_containment,
        "remainder_basis": remainder_binding,
        "quarantined": quarantined,
        "findings": findings,
        "clean": not findings,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    from cytir.ops import _provenance
    _provenance.write_provenance(args.out, "fold-assembly-claim-fragments",
                                 _provenance.arg_inputs(args, args.out, args.report))
    print(f"[fold] sets={len(index['sets'])} findings={len(findings)}",
          flush=True)
    for line in findings:
        print(f"  ! {line}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
