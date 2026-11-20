"""Unit tests for the ADR-101 assembly-guard suite.

Every failure mode task 456 §guards names has a synthetic fixture here, plus a
happy-path fixture that proves the guard can actually reach PASS -- a gate that
can only fail is not a gate.  The real-ledger run lives at the bottom and is
skipped when the produced artifacts are not present in the clone.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

M = importlib.import_module("cytir.ops.assembly_guards")

PASS, FAIL, WAITING = M.PASS, M.FAIL, M.WAITING
TASK427 = M.TASK427

MEMBERSHIP = 34_464_401
TOMBSTONES = 152_587
MULTI_CLAIMED = 976_351
COLLAPSED = 56_074
MIXED_URLS, MIXED_POINTS, MIXED_ROWS = 94, 550, 110

POINT_A = "00000000-0000-0000-0000-00000000000a"
POINT_B = "00000000-0000-0000-0000-00000000000b"
POINT_C = "00000000-0000-0000-0000-00000000000c"
POINT_D = "00000000-0000-0000-0000-00000000000d"


# ---------------------------------------------------------------------------
# fixture construction
# ---------------------------------------------------------------------------


def _write(root: Path, name: str, payload: dict) -> dict:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(body)
    return {"uri": name,
            "sha256": hashlib.sha256(body.encode()).hexdigest()}


def _ledger() -> dict:
    """A minimal stand-in with the same shape as r0-427-collision-ledger.v1."""
    return {
        "schema": "aw.r0-427-collision-ledger.v1",
        "problem": {"outside_adr092_remainder": MULTI_CLAIMED,
                    "task427_admitted_old_points": 1_211_392},
        "by_class": {"tombstone": {"contested_with_task427": 4}},
        "precedence": {"recommendation": {"name": "chunk-epoch-then-lineage"}},
        "task427_disposition": {
            "lost_breakdown": {"nobody_adr091_tombstone": 4},
            "new_point_ids": {"distinct": 351_156,
                              "reused_as_an_old_point_it_replaces": COLLAPSED},
        },
        "other_collisions": {
            "cc_materializer_x_task427_postconsent": {"intersect_task427": 0},
            "wikipedia_413a_x_all_61_replacement_witnesses": {"intersect": 0},
            "gesetze_witness_x_all_61_replacement_witnesses": {"intersect": 0},
            "track1a_retained_bind_x_task427_postconsent": {
                "task427_old_points_on_shared_urls_inside_remainder": 115_652},
        },
        "assembly_guard": {"checks": [
            {"id": "G5", "measured_today": {
                "task427_rows_declared": 351_156,
                "task427_rows_loaded": 290_871,
                "task427_rows_suppressed": 60_175,
                "task427_rows_mixed": MIXED_ROWS}},
            {"id": "G6", "measured_today": {
                "mixed_urls": MIXED_URLS, "old_points": MIXED_POINTS,
                "new_rows": MIXED_ROWS}},
        ]},
    }


def _census() -> dict:
    return {"schema": "aw.r0-1024-coverage-census.v1",
            "buckets": {"E_no_vector_will_exist": {"points": TOMBSTONES},
                        "sum_check": {"total": MEMBERSHIP}}}


def _manifests() -> dict:
    """A green inventory: one winner, one loser, one tombstone, one cancelled."""
    return {
        "schema": "aw.assembly-claim-manifests.v1",
        "sets": [
            {"set_id": TASK427, "mode": "replacement",
             "contributes_vectors": True,
             "claims": {"granularity": "point_id", "uri": "artifact://x",
                        "sha256": "0" * 64, "count": 1_211_392,
                        "verified": True},
             "adjudication": {"points_won": 974_724, "points_lost": 236_668,
                              "rows_declared": 351_156, "rows_loaded": 290_871,
                              "rows_suppressed": 60_285,
                              "contested_with_task427": 0,
                              "winner_on_contested": "this_set"},
             "census_witness": {"records": 974_724, "scope": "winning_subset",
                                "shape": "membership/points.jsonl.gz"}},
            {"set_id": "atomic_loser", "mode": "replacement",
             "contributes_vectors": True,
             "claims": {"granularity": "point_id", "uri": "artifact://y",
                        "sha256": "1" * 64, "count": 1_294_205,
                        "verified": True},
             "adjudication": {"contested_with_task427": MULTI_CLAIMED,
                              "winner_on_contested": TASK427,
                              "points_won": 1_294_205 - MULTI_CLAIMED,
                              "points_lost_to_task427": MULTI_CLAIMED,
                              "rows_declared": 1_294_205,
                              "rows_loaded": 1_294_205 - MULTI_CLAIMED,
                              "rows_suppressed": MULTI_CLAIMED},
             "census_witness": {"records": 1_294_205,
                                "consumed_by_census": "v31",
                                "scope": "all_claims",
                                "shape": "membership/points.jsonl.gz"}},
            {"set_id": "natural_person_policy_drop", "mode": "tombstone",
             "contributes_vectors": False,
             "claims": {"granularity": "point_id", "uri": "artifact://z",
                        "sha256": "2" * 64, "count": 31_125, "verified": True},
             "tombstones": {"granularity": "point_id", "count": 31_125,
                            "reclaimed_by_a_winning_set": 4}},
            {"set_id": "track1a_retained_bind", "mode": "cancelled",
             "contributes_vectors": False,
             "claims": {"granularity": "url", "uri": "artifact://w",
                        "count": 41_849, "verified": False},
             "cancellation": {"uri": "cancellation.json"}},
        ],
    }


def _plan(bindings: dict) -> dict:
    return {
        "schema": "aw.assembly-plan.v1",
        "build_id": "test-build",
        "membership": {"points": MEMBERSHIP, "uri": "artifact://m",
                       "sha256": "3" * 64},
        "inputs": bindings["inputs"],
        "records": bindings["records"],
        "expected_sets": [TASK427, "atomic_loser",
                          "natural_person_policy_drop",
                          "track1a_retained_bind"],
        "precedence": {"adr": "ADR-101", "ruling": "chunk-epoch-then-lineage",
                       "table": dict(M.ADR101_PRECEDENCE)},
        "disposition": {"replaced": MEMBERSHIP - TOMBSTONES - 1_000_000,
                        "re_embedded_in_place": 1_000_000,
                        "tombstoned": TOMBSTONES,
                        "carried_legacy": 0,
                        "pending": 0,
                        "contested": 0},
        "carried_legacy": {"count": 0},
        "pending_lanes": [],
        "apply": {
            "mode": "upsert_in_place",
            "id_collision_policy": "upsert_in_place",
            "ordering": "upsert_before_delete",
            "delete_set_excludes_loaded_ids": True,
            "delete_set": {"count": 974_724 - COLLAPSED,
                           "excludes": "artifact://collapse-ids.jsonl.gz"},
            "identity_collapse": {"count": COLLAPSED,
                                  "uri": "artifact://collapse-ids.jsonl.gz",
                                  "sha256": "4" * 64},
            "loaded_row_ids": {"distinct": 290_871 + 1_294_205 - MULTI_CLAIMED},
        },
    }


@pytest.fixture()
def green(tmp_path: Path) -> tuple[Path, Path]:
    """A fully green fixture: every guard PASSes.  Mutate it to break one."""
    inputs = {
        "collision_ledger": _write(tmp_path, "ledger.json", _ledger()),
        "coverage_census": _write(tmp_path, "census.json", _census()),
        "manifest_index": _write(tmp_path, "manifests.json", _manifests()),
    }
    records = {
        "g4_tombstone_suppression": _write(tmp_path, "g4.json", {
            "schema": "aw.assembly-suppression-record.v1", "guard": "G4",
            "count": 4,
            "point_ids": [POINT_A, POINT_B, POINT_C, POINT_D]}),
        "g6_mixed_url_resolution": _write(tmp_path, "g6.json", {
            "schema": "aw.assembly-unit-resolution.v1", "guard": "G6",
            "urls": MIXED_URLS, "old_points": MIXED_POINTS,
            "task427_new_rows": MIXED_ROWS,
            "task427_rows_loaded": 0,
            "task427_rows_suppressed": MIXED_ROWS,
            "resolutions": [
                {"url": f"https://example.test/{index}", "old_points": 5,
                 "task427_new_rows": 1, "unit_winner": "atomic_loser",
                 "basis": "majority", "task427_rows_loaded": 0,
                 "task427_rows_suppressed": 1}
                for index in range(MIXED_URLS)]}),
        "cancellation:track1a_retained_bind": _write(tmp_path, "cancellation.json", {
            "schema": "aw.assembly-cancellation-record.v1",
            "authority": {"adr": "ADR-101", "ruling": 3}}),
        "g4_tombstone_url_suppression": _write(tmp_path, "g4url.json", {
            "schema": "aw.assembly-suppression-record.v1", "guard": "G4",
            "authority": {"adr": "ADR-101a", "correction": 2,
                          "ratified_by": "Tim, merged PR"},
            "urls": 0, "rows_suppressed": 0, "suppressed": [],
            "other_entries": [], "open_obligations": [],
            "other_sets": {"measured": True, "cohabiting": {},
                           "clean": ["atomic_loser"]}}),
        "fold_report": _write(tmp_path, "fold.json", {
            "schema": "aw.assembly-fold-report.v1",
            "conservation": {"atomic_loser": {
                "superseded": 10, "delete_set": 8, "identity_collapse": 2,
                "conserves": True}},
            "tombstone_intersection": {"atomic_loser": 0,
                                       "task427_postconsent": 4},
            "claim_overlaps": {}, "row_id_overlaps": {}, "findings": []}),
    }
    # 94 resolutions x 1 row = 94, but the ledger measured 110 rows; pad.
    payload = json.loads((tmp_path / "g6.json").read_text())
    payload["resolutions"][0]["task427_new_rows"] = MIXED_ROWS - MIXED_URLS + 1
    payload["resolutions"][0]["task427_rows_suppressed"] = (
        MIXED_ROWS - MIXED_URLS + 1)
    payload["resolutions"][0]["old_points"] = MIXED_POINTS - 5 * (MIXED_URLS - 1)
    records["g6_mixed_url_resolution"] = _write(tmp_path, "g6.json", payload)

    plan = _plan({"inputs": inputs, "records": records})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return tmp_path, plan_path


def run(tmp_path: Path, plan_path: Path) -> dict:
    return M.evaluate(tmp_path, plan_path)


def status_of(report: dict, guard_id: str) -> str:
    return next(row["status"] for row in report["guards"]
                if row["id"] == guard_id)


def mutate(tmp_path: Path, plan_path: Path, fn) -> dict:
    plan = json.loads(plan_path.read_text())
    fn(plan)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return run(tmp_path, plan_path)


def remanifest(tmp_path: Path, plan_path: Path, fn) -> dict:
    plan = json.loads(plan_path.read_text())
    manifests = json.loads(
        (tmp_path / plan["inputs"]["manifest_index"]["uri"]).read_text())
    fn(manifests)
    plan["inputs"]["manifest_index"] = _write(tmp_path, "manifests.json",
                                              manifests)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return run(tmp_path, plan_path)


def set_of(manifests: dict, set_id: str) -> dict:
    return next(row for row in manifests["sets"] if row["set_id"] == set_id)


# ---------------------------------------------------------------------------
# net-new additions -- fixture construction
# ---------------------------------------------------------------------------

ADDITION = "news_addition"
ADDITION_ROWS = 17_959
ADD_IDS = {"uri": "artifact://add-ids.jsonl.gz", "sha256": "5" * 64,
           "count": ADDITION_ROWS}


def _addition_set() -> dict:
    """A folded net-new-addition manifest, on task 500's real shape."""
    return {
        "set_id": ADDITION, "mode": "net_new_addition",
        "contributes_vectors": True, "contributes_production_points": False,
        "folded_from": "claim-manifest-news.v1.json",
        "claims": {"granularity": "point_id", "count": 0, "verified": True,
                   "claims_are_production_points": False},
        "new_point_ids": dict(ADD_IDS),
        "additions": {**ADD_IDS, "rows": ADDITION_ROWS},
        "membership_basis": {"receipt_sha256": "3" * 64, "rows": MEMBERSHIP},
        "vector": {"rows": ADDITION_ROWS, "dim": 1024, "dtype": "float16"},
        "adjudication": {"rows_declared": ADDITION_ROWS,
                         "rows_loaded": ADDITION_ROWS, "rows_suppressed": 0,
                         "contested_with_task427": 0, "points_won": 0,
                         "winner_on_contested": "this_set"},
    }


def _addition_witness() -> dict:
    return {
        "schema": "aw.net-new-addition-census-witness.v1",
        "set_id": ADDITION,
        "guard_1_enumeration_completeness": {
            "vector_rows": ADDITION_ROWS, "split_net_new": ADDITION_ROWS,
            "orphans_either_direction": 0, "verdict": "COMPLETE"},
        "guard_2_disjointness": {
            "membership_hits": 0,
            "cross_set": {"checked": True, "collisions": {},
                          "unchecked_artifacts": {}},
            "verdict": "DISJOINT"},
        "guard_3_conservation": {"holds": True, "verdict": "CONSERVED"},
    }


@pytest.fixture()
def with_addition(green):
    """The green fixture plus one clean net-new addition; still fully green."""
    tmp_path, plan_path = green
    plan = json.loads(plan_path.read_text())
    manifests = json.loads(
        (tmp_path / plan["inputs"]["manifest_index"]["uri"]).read_text())
    manifests["sets"].append(_addition_set())
    plan["inputs"]["manifest_index"] = _write(tmp_path, "manifests.json",
                                              manifests)

    fold = json.loads((tmp_path / "fold.json").read_text())
    fold["conservation"][ADDITION] = {
        "superseded": 0, "delete_set": 0, "identity_collapse": 0,
        "addition_vector_rows": ADDITION_ROWS,
        "addition_minted_ids": ADDITION_ROWS,
        "addition_enumerated_ids": ADDITION_ROWS, "conserves": True}
    fold["additions"] = {ADDITION: {
        "rows": ADDITION_ROWS, "claims_production_points": 0,
        "minted_ids_in_another_sets_claims": {},
        "minted_ids_tombstoned": 0, "minted_ids_in_census_remainder": 0,
        "conserves": True}}
    fold["tombstone_intersection"][ADDITION] = 0
    plan["records"]["fold_report"] = _write(tmp_path, "fold.json", fold)
    plan["records"][f"addition_witness:{ADDITION}"] = _write(
        tmp_path, "addition-witness.json", _addition_witness())

    plan["expected_sets"] = sorted(plan["expected_sets"] + [ADDITION])
    plan["additions"] = {
        "count": ADDITION_ROWS,
        "collection_points_after_additions": MEMBERSHIP + ADDITION_ROWS,
        "sets": [{"set_id": ADDITION, "rows": ADDITION_ROWS,
                  "claimed_production_points": 0, "row_ids": dict(ADD_IDS)}],
    }
    plan["apply"]["loaded_row_ids"]["distinct"] += ADDITION_ROWS
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return tmp_path, plan_path


def add_mutate(fixture, plan_fn=None, manifest_fn=None, witness_fn=None,
               fold_fn=None) -> dict:
    """Break exactly one thing about the clean addition."""
    tmp_path, plan_path = fixture
    plan = json.loads(plan_path.read_text())
    if plan_fn:
        plan_fn(plan)
    if manifest_fn:
        manifests = json.loads(
            (tmp_path / plan["inputs"]["manifest_index"]["uri"]).read_text())
        manifest_fn(set_of(manifests, ADDITION))
        plan["inputs"]["manifest_index"] = _write(tmp_path, "manifests.json",
                                                  manifests)
    if witness_fn:
        witness = _addition_witness()
        witness_fn(witness)
        plan["records"][f"addition_witness:{ADDITION}"] = _write(
            tmp_path, "addition-witness.json", witness)
    if fold_fn:
        fold = json.loads((tmp_path / "fold.json").read_text())
        fold_fn(fold)
        plan["records"]["fold_report"] = _write(tmp_path, "fold.json", fold)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return run(tmp_path, plan_path)


# ---------------------------------------------------------------------------
# the gate as a whole
# ---------------------------------------------------------------------------


def test_green_fixture_reaches_a_green_gate(green):
    report = run(*green)
    assert report["verdict"] == "GREEN", json.dumps(report["guards"], indent=1)
    assert report["flip_permitted"] is True
    assert report["summary"][PASS] == 8


def test_missing_plan_fails_every_guard_closed(tmp_path):
    report = M.evaluate(tmp_path, tmp_path / "absent.json")
    assert report["verdict"] == "BLOCKED"
    assert report["summary"][FAIL] == 8
    assert report["flip_permitted"] is False


def test_digest_drift_on_a_pinned_input_fails_closed(green):
    tmp_path, plan_path = green
    (tmp_path / "ledger.json").write_text('{"schema": "tampered"}\n')
    report = run(tmp_path, plan_path)
    assert report["verdict"] == "BLOCKED"
    assert "digest drift" in json.dumps(report)


def test_exit_codes_map_to_the_three_verdicts(green, tmp_path, monkeypatch):
    root, plan_path = green
    assert M.main(["--plan", str(plan_path), "--root", str(root)]) == 0
    report = mutate(root, plan_path, lambda plan: plan.update(
        {"pending_lanes": [{"lane_id": "F1", "points": 10,
                            "blocks_guards": ["G3"]}],
         "disposition": {**plan["disposition"], "pending": 10,
                         "replaced": plan["disposition"]["replaced"] - 10}}))
    assert report["verdict"] == "BURNING-DOWN"
    assert M.main(["--plan", str(plan_path), "--root", str(root)]) == 2
    mutate(root, plan_path, lambda plan: plan["apply"].update(
        {"ordering": "delete_then_insert"}))
    assert M.main(["--plan", str(plan_path), "--root", str(root)]) == 1


# ---------------------------------------------------------------------------
# G1 -- claim-manifest completeness
# ---------------------------------------------------------------------------


def test_g1_fails_a_set_that_publishes_urls_only(green):
    """The Track-1A failure: URL + point_count, no point IDs."""
    report = remanifest(*green, lambda manifests: set_of(
        manifests, "track1a_retained_bind").update(
            {"contributes_vectors": True, "mode": "replacement"}))
    assert status_of(report, "G1") == FAIL
    assert "counts or URLs only" in json.dumps(report)


def test_g1_fails_a_set_missing_from_the_index(green):
    report = remanifest(*green, lambda manifests: manifests["sets"].remove(
        set_of(manifests, "atomic_loser")))
    assert status_of(report, "G1") == FAIL
    assert "publish no manifest" in json.dumps(report)


def test_g1_fails_a_cancelled_set_with_no_cancellation_record(green):
    root, plan_path = green
    report = mutate(root, plan_path, lambda plan: plan["records"].pop(
        "cancellation:track1a_retained_bind"))
    assert status_of(report, "G1") == FAIL
    assert "cancellation record" in json.dumps(report)


def test_g1_fails_a_cancellation_that_names_no_adr(green):
    root, plan_path = green
    binding = _write(root, "cancellation.json", {"schema": "x", "authority": {}})
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"cancellation:track1a_retained_bind": binding}))
    assert status_of(report, "G1") == FAIL
    assert "names no ratified ADR" in json.dumps(report)


def test_g1_accepts_an_explicit_cancellation_but_not_an_absence(green):
    """ADR-101 Ruling 3: cancelled is a record, never a missing manifest."""
    report = run(*green)
    assert status_of(report, "G1") == PASS
    report = remanifest(*green, lambda manifests: manifests["sets"].remove(
        set_of(manifests, "track1a_retained_bind")))
    assert status_of(report, "G1") == FAIL


def test_g1_fails_closed_when_no_roster_is_declared(green):
    report = mutate(*green, lambda plan: plan.pop("expected_sets"))
    assert status_of(report, "G1") == FAIL
    assert "fails closed" in json.dumps(report)


def test_g1_waits_on_a_lane_that_will_publish_claims_later(green):
    report = mutate(*green, lambda plan: plan.update(
        {"pending_lanes": [{"lane_id": "F2", "points": 852_122,
                            "blocks_guards": ["G1", "G3"]}],
         "disposition": {**plan["disposition"], "pending": 852_122,
                         "replaced": plan["disposition"]["replaced"] - 852_122}}))
    assert status_of(report, "G1") == WAITING
    assert status_of(report, "G3") == WAITING


# ---------------------------------------------------------------------------
# G2 -- single-claim
# ---------------------------------------------------------------------------


def test_g2_fails_when_the_adjudicated_population_misses_the_ledger(green):
    report = remanifest(*green, lambda manifests: set_of(
        manifests, "atomic_loser")["adjudication"].update(
            {"contested_with_task427": MULTI_CLAIMED - 1}))
    assert status_of(report, "G2") == FAIL
    assert "double-claimed" in json.dumps(report)
    assert "+1 unaccounted" in json.dumps(report)


def test_g2_names_the_set_that_cannot_be_intersected(green):
    """The arxiv case: a payload predicate is not a claim."""
    def add_predicate_set(manifests):
        set_of(manifests, "atomic_loser")["adjudication"][
            "contested_with_task427"] = MULTI_CLAIMED - 176
        manifests["sets"].append(
            {"set_id": "named_slice:arxiv", "mode": "replacement",
             "contributes_vectors": True,
             "claims": {"granularity": "payload_predicate",
                        "uri": "artifact://rule", "count": 1_918_324,
                        "verified": False},
             "adjudication": {"contested_with_task427": None,
                              "winner_on_contested": "this_set",
                              "points_won": 1_918_324,
                              "points_lost_to_task427": 0}})
    report = remanifest(*green, add_predicate_set)
    assert status_of(report, "G2") == FAIL
    assert "+176 unaccounted" in json.dumps(report)
    assert "named_slice:arxiv" in json.dumps(report)
    # and G1 refuses the same set, for the same reason
    assert status_of(report, "G1") == FAIL


def test_g2_fails_an_unnamed_collision(green):
    report = remanifest(*green, lambda manifests: set_of(
        manifests, "atomic_loser")["adjudication"].pop("winner_on_contested"))
    assert status_of(report, "G2") == FAIL
    assert "name no winner" in json.dumps(report)


def test_g2_fails_an_unadjudicated_non_427_pair(green):
    root, plan_path = green
    ledger = _ledger()
    ledger["other_collisions"]["wikipedia_413a_x_all_61_replacement_witnesses"][
        "intersect"] = 17
    binding = _write(root, "ledger.json", ledger)
    report = mutate(root, plan_path, lambda plan: plan["inputs"].update(
        {"collision_ledger": binding}))
    assert status_of(report, "G2") == FAIL


def test_g2_fails_two_folded_sets_claiming_the_same_old_point(green):
    root, plan_path = green
    binding = _write(root, "fold.json", {
        "conservation": {}, "tombstone_intersection": {"a": 0, "b": 0},
        "claim_overlaps": {"a x b": 91}, "row_id_overlaps": {}})
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"fold_report": binding}))
    assert status_of(report, "G2") == FAIL
    assert "no adjudicated winner" in json.dumps(report)


def test_g2_waits_when_no_fold_report_is_bound(green):
    report = mutate(*green, lambda plan: plan["records"].pop("fold_report"))
    assert status_of(report, "G2") == WAITING


# ---------------------------------------------------------------------------
# G7 -- supersession conservation over folded sets
# ---------------------------------------------------------------------------


def test_g7_fails_a_set_that_orphans_superseded_points(green):
    """A point in neither the delete set nor the collapse set survives at 128."""
    root, plan_path = green
    binding = _write(root, "fold.json", {
        "conservation": {"task472_cc": {
            "superseded": 2_229_094, "delete_set": 2_213_529,
            "identity_collapse": 15_000, "conserves": False}},
        "tombstone_intersection": {"task472_cc": 0},
        "claim_overlaps": {}, "row_id_overlaps": {}})
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"fold_report": binding}))
    assert status_of(report, "G7") == FAIL
    assert "superseded 2,229,094" in json.dumps(report)


def test_g7_fails_two_folded_sets_emitting_the_same_row_id(green):
    root, plan_path = green
    binding = _write(root, "fold.json", {
        "conservation": {"a": {"superseded": 1, "delete_set": 1,
                               "identity_collapse": 0, "conserves": True}},
        "tombstone_intersection": {"a": 0},
        "claim_overlaps": {}, "row_id_overlaps": {"a x b": 12}})
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"fold_report": binding}))
    assert status_of(report, "G7") == FAIL
    assert "identical row ID" in json.dumps(report)


def test_g7_waits_when_no_fold_report_is_bound(green):
    report = mutate(*green, lambda plan: plan["records"].pop("fold_report"))
    assert status_of(report, "G7") == WAITING
    assert "no fold report is bound" in json.dumps(report)


# ---------------------------------------------------------------------------
# G3 -- conservation
# ---------------------------------------------------------------------------


def test_g3_fails_when_the_dispositions_do_not_sum(green):
    report = mutate(*green, lambda plan: plan["disposition"].update(
        {"replaced": plan["disposition"]["replaced"] - 1}))
    assert status_of(report, "G3") == FAIL
    assert "disposed !=" in json.dumps(report)


def test_g3_fails_on_a_silent_carried_legacy_population(green):
    """The census-453 bucket-F mode: 1.3M points every other gate passes."""
    carried = 1_309_473
    report = mutate(*green, lambda plan: (
        plan["disposition"].update(
            {"replaced": plan["disposition"]["replaced"] - carried,
             "carried_legacy": carried}),
        plan.update({"carried_legacy": {"count": carried}})))
    assert status_of(report, "G3") == FAIL
    assert "NOT enumerated" in json.dumps(report)


def test_g3_fails_an_enumerated_carried_legacy_set_with_no_waiver(green):
    carried = 1_309_473
    report = mutate(*green, lambda plan: (
        plan["disposition"].update(
            {"replaced": plan["disposition"]["replaced"] - carried,
             "carried_legacy": carried}),
        plan.update({"carried_legacy": {"count": carried,
                                        "enumeration_uri": "artifact://c"}})))
    assert status_of(report, "G3") == FAIL
    assert "no named waiver" in json.dumps(report)


def test_g3_refuses_a_self_issued_waiver(green):
    carried = 10
    report = mutate(*green, lambda plan: (
        plan["disposition"].update(
            {"replaced": plan["disposition"]["replaced"] - carried,
             "carried_legacy": carried}),
        plan.update({"carried_legacy": {
            "count": carried, "enumeration_uri": "artifact://c",
            "waiver": {"reason": "small"}}})))
    assert status_of(report, "G3") == FAIL
    assert "never self-issued" in json.dumps(report)


def test_g3_accepts_a_ratified_waiver(green):
    carried = 10
    report = mutate(*green, lambda plan: (
        plan["disposition"].update(
            {"replaced": plan["disposition"]["replaced"] - carried,
             "carried_legacy": carried}),
        plan.update({"carried_legacy": {
            "count": carried, "enumeration_uri": "artifact://c",
            "waiver": {"ratified_by": "Tim, PR #99"}}})))
    assert status_of(report, "G3") == PASS


def test_g3_fails_when_tombstones_miss_census_bucket_e(green):
    report = mutate(*green, lambda plan: plan["disposition"].update(
        {"tombstoned": TOMBSTONES - 4,
         "replaced": plan["disposition"]["replaced"] + 4}))
    assert status_of(report, "G3") == FAIL
    assert "census bucket E" in json.dumps(report)


def test_g3_waits_on_an_unadjudicated_double_claim(green):
    """A point two sets claim has no owner, so it has no disposition."""
    report = mutate(*green, lambda plan: (
        plan["disposition"].update(
            {"replaced": plan["disposition"]["replaced"] - 46_132,
             "contested": 46_132}),
        plan.update({"contested": {"count": 46_132, "pairs": {
            "task467 x task472_cc": 46_132}}})))
    assert status_of(report, "G3") == WAITING
    assert "no adjudicated winner" in json.dumps(report)
    assert report["flip_permitted"] is False


def test_g3_fails_when_pending_does_not_match_the_lanes(green):
    report = mutate(*green, lambda plan: plan.update(
        {"pending_lanes": [{"lane_id": "C", "points": 492_376}],
         "disposition": {**plan["disposition"], "pending": 1}}))
    assert status_of(report, "G3") == FAIL
    assert "!= the sum of" in json.dumps(report)


# ---------------------------------------------------------------------------
# G4 -- tombstone exclusivity
# ---------------------------------------------------------------------------


def test_g4_fails_a_reclaimed_tombstone_with_no_suppression_record(green):
    report = mutate(*green, lambda plan: plan["records"].pop(
        "g4_tombstone_suppression"))
    assert status_of(report, "G4") == FAIL
    assert "no suppression record" in json.dumps(report)


def test_g4_fails_a_short_suppression_record(green):
    root, plan_path = green
    binding = _write(root, "g4.json", {
        "guard": "G4", "count": 3, "point_ids": [POINT_A, POINT_B, POINT_C]})
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"g4_tombstone_suppression": binding}))
    assert status_of(report, "G4") == FAIL
    assert "3 IDs listed, 4 re-claimed" in json.dumps(report)


def test_g4_fails_a_suppression_record_with_a_bad_point_id(green):
    root, plan_path = green
    binding = _write(root, "g4.json", {
        "guard": "G4", "count": 4,
        "point_ids": [POINT_A, POINT_B, POINT_C, "not-a-uuid"]})
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"g4_tombstone_suppression": binding}))
    assert status_of(report, "G4") == FAIL
    assert "invalid point ID" in json.dumps(report)


def test_g4_fails_when_the_manifest_and_ledger_disagree_on_the_reclaim(green):
    report = remanifest(*green, lambda manifests: set_of(
        manifests, "natural_person_policy_drop")["tombstones"].update(
            {"reclaimed_by_a_winning_set": 9}))
    assert status_of(report, "G4") == FAIL
    assert "ledger measured 4" in json.dumps(report)


def test_g4_clears_a_tombstoned_url_the_suppression_record_covers(green):
    """The G6 measurement stands; ADR-101a's correction layers on top of it.
    Reading the measurement alone would make this guard fail forever."""
    root, plan_path = green
    payload = json.loads((root / "g6.json").read_text())
    url = payload["resolutions"][7]["url"]
    payload["resolutions"][7].update(
        {"point_winner_tally": {"nobody": 1, TASK427: 2},
         "unit_winner": TASK427, "basis": "majority",
         "task427_rows_loaded": 1, "task427_rows_suppressed": 0})
    g6 = _write(root, "g6.json", payload)
    g4url = _write(root, "g4url.json", {
        "guard": "G4", "urls": 1, "rows_suppressed": 1,
        "suppressed": [{"url": url, "rows_loaded_before": 1,
                        "rows_loaded_after": 0}],
        "other_sets": {"measured": True, "cohabiting": {}, "clean": ["x"]}})
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"g6_mixed_url_resolution": g6, "g4_tombstone_url_suppression": g4url}))
    assert status_of(report, "G4") == PASS


def test_g4_fails_a_row_loaded_on_a_tombstoned_url(green):
    """ADR-101a Correction 2: privacy beats coverage, and it is ratified."""
    root, plan_path = green
    payload = json.loads((root / "g6.json").read_text())
    payload["resolutions"][7].update(
        {"point_winner_tally": {"nobody": 1, TASK427: 2},
         "unit_winner": TASK427, "basis": "majority",
         "task427_rows_loaded": 1, "task427_rows_suppressed": 0})
    binding = _write(root, "g6.json", payload)
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"g6_mixed_url_resolution": binding}))
    assert status_of(report, "G4") == FAIL
    assert "tombstoned by the" in json.dumps(report)
    assert report["flip_permitted"] is False


def test_g4_fails_when_no_url_suppression_record_is_bound(green):
    report = mutate(*green, lambda plan: plan["records"].pop(
        "g4_tombstone_url_suppression"))
    assert status_of(report, "G4") == FAIL
    assert "binds no" in json.dumps(report)


def test_g4_fails_an_open_cohabitation_obligation_on_another_set(green):
    """A set other than 427 claims a point on a tombstoned URL and has not
    suppressed it. Task 462 cannot edit another set's row split, so it names
    the obligation and the gate refuses until the owner discharges it."""
    root, plan_path = green
    binding = _write(root, "g4url.json", {
        "guard": "G4", "urls": 0, "rows_suppressed": 0, "suppressed": [],
        "other_entries": [{"url": "https://x/", "set_id": "task472_cc",
                           "status": "open"}],
        "open_obligations": [{"url": "https://x/", "set_id": "task472_cc",
                              "status": "open"}],
        "other_sets": {"measured": True,
                       "cohabiting": {"task472_cc": ["https://x/"]},
                       "clean": []}})
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"g4_tombstone_url_suppression": binding}))
    assert status_of(report, "G4") == FAIL
    assert "has not suppressed it" in json.dumps(report)


def test_g4_accepts_a_cohabiting_url_that_loads_nothing(green):
    """task 427's fourth URL: it carries a tombstone but 427 wins none of its
    points, so the rule is already satisfied and nothing has to move."""
    root, plan_path = green
    binding = _write(root, "g4url.json", {
        "guard": "G4", "urls": 0, "rows_suppressed": 0, "suppressed": [],
        "other_entries": [{"url": "https://y/", "set_id": "task427_postconsent",
                           "status": "already-satisfied",
                           "rows_loaded_after": 0}],
        "open_obligations": [],
        "other_sets": {"measured": True,
                       "cohabiting": {"task427": ["https://y/"]},
                       "clean": ["a"]}})
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"g4_tombstone_url_suppression": binding}))
    assert status_of(report, "G4") == PASS


def test_g4_fails_a_suppression_record_that_still_loads_rows(green):
    root, plan_path = green
    binding = _write(root, "g4url.json", {
        "guard": "G4", "urls": 1, "rows_suppressed": 1,
        "suppressed": [{"url": "https://example.test/x",
                        "rows_loaded_after": 1}],
        "other_sets": {"measured": True, "cohabiting": {}, "clean": []}})
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"g4_tombstone_url_suppression": binding}))
    assert status_of(report, "G4") == FAIL
    assert "still declare loaded rows" in json.dumps(report)


def test_g4_waits_while_the_other_sets_are_unmeasured(green):
    root, plan_path = green
    binding = _write(root, "g4url.json", {
        "guard": "G4", "urls": 0, "rows_suppressed": 0, "suppressed": [],
        "other_sets": {"measured": False}})
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"g4_tombstone_url_suppression": binding}))
    assert status_of(report, "G4") == WAITING
    assert "unmeasured" in json.dumps(report)


def test_g4_fails_a_folded_set_that_claims_a_tombstoned_point(green):
    root, plan_path = green
    binding = _write(root, "fold.json", {
        "conservation": {}, "claim_overlaps": {}, "row_id_overlaps": {},
        "tombstone_intersection": {"task472_cc": 3}})
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"fold_report": binding}))
    assert status_of(report, "G4") == FAIL
    assert "claim a tombstoned point" in json.dumps(report)


def test_g4_waits_when_no_fold_report_is_bound(green):
    report = mutate(*green, lambda plan: plan["records"].pop("fold_report"))
    assert status_of(report, "G4") == WAITING
    assert "no fold report is bound" in json.dumps(report)


def test_g4_fails_when_no_tombstone_set_exists_at_all(green):
    report = remanifest(*green, lambda manifests: manifests["sets"].remove(
        set_of(manifests, "natural_person_policy_drop")))
    assert status_of(report, "G4") == FAIL
    assert "declares no tombstone set" in json.dumps(report)


# ---------------------------------------------------------------------------
# G5 -- row-level loser suppression
# ---------------------------------------------------------------------------


def test_g5_fails_when_rows_do_not_conserve(green):
    report = remanifest(*green, lambda manifests: set_of(
        manifests, TASK427)["adjudication"].update({"rows_loaded": 290_872}))
    assert status_of(report, "G5") == FAIL
    assert "!=" in json.dumps(report)


def test_g5_fails_a_plan_that_applies_a_different_precedence(green):
    report = mutate(*green, lambda plan: plan["precedence"]["table"].update(
        {"exact_epoch4": TASK427}))
    assert status_of(report, "G5") == FAIL
    assert "diverges from ADR-101" in json.dumps(report)


def test_g5_fails_a_plan_that_does_not_cite_adr101(green):
    report = mutate(*green, lambda plan: plan["precedence"].update(
        {"adr": "ADR-092"}))
    assert status_of(report, "G5") == FAIL


def test_g5_fails_loading_more_427_rows_than_the_ledger_allows(green):
    """Reading A smuggled in: 427 wins everywhere."""
    report = remanifest(*green, lambda manifests: set_of(
        manifests, TASK427)["adjudication"].update(
            {"rows_loaded": 351_156, "rows_suppressed": 0}))
    assert status_of(report, "G5") == FAIL
    assert "more than the ledger's ceiling" in json.dumps(report)


def test_g5_waits_on_a_set_with_no_vectors_yet(green):
    report = remanifest(*green, lambda manifests: manifests["sets"].append(
        {"set_id": "cc_materializer", "mode": "replacement",
         "contributes_vectors": True, "pending": {"reason": "embed-only"},
         "claims": {"granularity": "point_id", "uri": "artifact://cc",
                    "sha256": "5" * 64, "count": 2_229_094, "verified": True},
         "adjudication": {"points_won": 2_229_094,
                          "contested_with_task427": 0,
                          "winner_on_contested": "this_set"}}))
    assert status_of(report, "G5") == WAITING


# ---------------------------------------------------------------------------
# G6 -- URL-unit integrity
# ---------------------------------------------------------------------------


def test_g6_fails_with_no_unit_resolution(green):
    report = mutate(*green, lambda plan: plan["records"].pop(
        "g6_mixed_url_resolution"))
    assert status_of(report, "G6") == FAIL
    assert "no unit resolution is bound" in json.dumps(report)


def test_g6_fails_a_partial_resolution(green):
    root, plan_path = green
    payload = json.loads((root / "g6.json").read_text())
    payload["resolutions"] = payload["resolutions"][:-1]
    payload["urls"] = len(payload["resolutions"])
    binding = _write(root, "g6.json", payload)
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"g6_mixed_url_resolution": binding}))
    assert status_of(report, "G6") == FAIL
    assert "ledger measured 94" in json.dumps(report)


def test_g6_fails_an_unratified_tie_break(green):
    root, plan_path = green
    payload = json.loads((root / "g6.json").read_text())
    payload["resolutions"][3]["basis"] = "coin-flip"
    binding = _write(root, "g6.json", payload)
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"g6_mixed_url_resolution": binding}))
    assert status_of(report, "G6") == FAIL
    assert "not the ratified tie-break" in json.dumps(report)


def test_g6_fails_a_resolution_whose_rows_do_not_conserve(green):
    root, plan_path = green
    payload = json.loads((root / "g6.json").read_text())
    payload["resolutions"][3]["task427_rows_loaded"] = 5
    binding = _write(root, "g6.json", payload)
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"g6_mixed_url_resolution": binding}))
    assert status_of(report, "G6") == FAIL
    assert "row split does not conserve" in json.dumps(report)


def test_g6_fails_a_duplicate_url_in_the_resolution(green):
    root, plan_path = green
    payload = json.loads((root / "g6.json").read_text())
    payload["resolutions"][5]["url"] = payload["resolutions"][4]["url"]
    binding = _write(root, "g6.json", payload)
    report = mutate(root, plan_path, lambda plan: plan["records"].update(
        {"g6_mixed_url_resolution": binding}))
    assert status_of(report, "G6") == FAIL
    assert "duplicate URL" in json.dumps(report)


# ---------------------------------------------------------------------------
# G7 -- identity uniqueness and delete/insert ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ordering", sorted(M.FORBIDDEN_APPLY_ORDERINGS))
def test_g7_fails_the_two_orderings_that_lose_or_duplicate_points(green, ordering):
    report = mutate(*green, lambda plan: plan["apply"].update(
        {"ordering": ordering}))
    assert status_of(report, "G7") == FAIL
    assert "loses or duplicates" in json.dumps(report)


def test_g7_fails_a_plan_that_does_not_declare_upsert_in_place(green):
    report = mutate(*green, lambda plan: plan["apply"].update(
        {"mode": "replace", "id_collision_policy": "last_write_wins"}))
    assert status_of(report, "G7") == FAIL
    assert "not\\n" not in json.dumps(report)
    assert "upsert_in_place" in json.dumps(report)


def test_g7_fails_a_delete_set_that_does_not_exclude_loaded_ids(green):
    report = mutate(*green, lambda plan: plan["apply"].update(
        {"delete_set_excludes_loaded_ids": False}))
    assert status_of(report, "G7") == FAIL


def test_g7_fails_a_collapse_count_that_misses_the_ledger(green):
    report = mutate(*green, lambda plan: plan["apply"]["identity_collapse"].update(
        {"count": 0}))
    assert status_of(report, "G7") == FAIL
    assert "identity-collapsed IDs" in json.dumps(report)


def test_g7_fails_a_counted_but_unenumerated_collapse_set(green):
    report = mutate(*green, lambda plan: plan["apply"].update(
        {"identity_collapse": {"count": COLLAPSED}}))
    assert status_of(report, "G7") == FAIL
    assert "cannot exclude what it cannot name" in json.dumps(report)


def test_g7_fails_when_a_row_id_is_loaded_twice(green):
    report = mutate(*green, lambda plan: plan["apply"]["loaded_row_ids"].update(
        {"distinct": plan["apply"]["loaded_row_ids"]["distinct"] - 1}))
    assert status_of(report, "G7") == FAIL
    assert "loaded twice" in json.dumps(report)


def test_g7_fails_a_delete_set_that_names_the_wrong_exclusion_list(green):
    report = mutate(*green, lambda plan: plan["apply"]["delete_set"].update(
        {"excludes": "artifact://something-else"}))
    assert status_of(report, "G7") == FAIL
    assert "exclusion list" in json.dumps(report)


# ---------------------------------------------------------------------------
# G8 -- census witness parity
# ---------------------------------------------------------------------------


def test_g8_fails_a_set_with_no_census_witness(green):
    report = remanifest(*green, lambda manifests: set_of(
        manifests, TASK427).pop("census_witness"))
    assert status_of(report, "G8") == FAIL
    assert "publish no census point witness" in json.dumps(report)


def test_g8_fails_a_witness_that_covers_more_than_the_winning_subset(green):
    report = remanifest(*green, lambda manifests: set_of(
        manifests, TASK427)["census_witness"].update({"records": 1_211_392}))
    assert status_of(report, "G8") == FAIL
    assert "!= winning points" in json.dumps(report)


def test_g8_fails_a_fresh_witness_scoped_to_all_claims(green):
    """A new witness over every claim makes the next census raise, not subtract."""
    report = remanifest(*green, lambda manifests: set_of(
        manifests, TASK427)["census_witness"].update({"scope": "all_claims"}))
    assert status_of(report, "G8") == FAIL
    assert "scoped to its winning subset" in json.dumps(report)


def test_g8_accepts_an_unwitnessed_winning_subset_scope(green):
    """Task 427's real case: it wins 974,724 but only some are new to census."""
    report = remanifest(*green, lambda manifests: set_of(
        manifests, TASK427)["census_witness"].update(
            {"scope": "unwitnessed_winning_subset", "records": 235_041,
             "winning_points_total": 974_724}))
    assert status_of(report, "G8") == PASS


def test_g8_fails_a_subset_witness_that_exceeds_what_the_set_wins(green):
    report = remanifest(*green, lambda manifests: set_of(
        manifests, TASK427)["census_witness"].update(
            {"scope": "unwitnessed_winning_subset", "records": 1_000_000,
             "winning_points_total": 974_724}))
    assert status_of(report, "G8") == FAIL
    assert "more than the" in json.dumps(report)


def test_g8_accepts_a_consumed_witness_over_all_of_its_claims(green):
    """v31 completing IS the disjointness proof for a witness it consumed."""
    report = run(*green)
    assert status_of(report, "G8") == PASS


def test_g8_fails_a_witness_in_the_wrong_shape(green):
    report = remanifest(*green, lambda manifests: set_of(
        manifests, TASK427)["census_witness"].update({"shape": "counts.json"}))
    assert status_of(report, "G8") == FAIL
    assert "not in the shape" in json.dumps(report)


def test_g8_waits_on_a_set_that_is_not_embedded_yet(green):
    report = remanifest(*green, lambda manifests: manifests["sets"].append(
        {"set_id": "cc_materializer", "mode": "replacement",
         "contributes_vectors": True, "pending": {"reason": "embed-only"},
         "claims": {"granularity": "point_id", "uri": "artifact://cc",
                    "sha256": "5" * 64, "count": 2_229_094, "verified": True},
         "adjudication": {"points_won": 2_229_094,
                          "contested_with_task427": 0,
                          "winner_on_contested": "this_set",
                          "rows_declared": 0, "rows_loaded": 0,
                          "rows_suppressed": 0}}))
    assert status_of(report, "G8") == WAITING


# ---------------------------------------------------------------------------
# net-new additions -- the term the disposition identity cannot hold
# ---------------------------------------------------------------------------


def test_a_clean_addition_keeps_the_gate_green(with_addition):
    """The whole point: a set that adds rows nobody ever had is expressible.
    Before the additions term it was not -- G1 failed it for publishing an
    empty claim and G3 had no bucket that could hold its rows."""
    report = run(*with_addition)
    assert report["verdict"] == "GREEN", json.dumps(report["guards"], indent=1)
    assert report["summary"][PASS] == 8


def test_the_addition_is_counted_and_the_collection_size_is_stated(
        with_addition):
    report = run(*with_addition)
    g3 = next(row for row in report["guards"] if row["id"] == "G3")
    assert g3["counts"]["additions"] == ADDITION_ROWS
    # the membership identity is untouched: additions are not disposed
    assert g3["counts"]["membership"] == MEMBERSHIP
    detail = next(row for row in g3["checks"]
                  if row["name"] == "additions-declared")
    assert detail["counts"]["collection"] == MEMBERSHIP + ADDITION_ROWS


def test_two_addition_sets_are_summed_and_each_is_checked(with_addition):
    """The next daily news pack lands beside this one, so the term has to hold
    more than one set -- and a second set must not be able to ride the first
    one's witness."""
    tmp_path, plan_path = with_addition
    second, rows = "news_addition_2", 6_500
    plan = json.loads(plan_path.read_text())
    manifests = json.loads((tmp_path / "manifests.json").read_text())
    row = _addition_set()
    row["set_id"] = second
    for key in ("new_point_ids", "additions"):
        row[key] = {**row[key], "sha256": "6" * 64, "count": rows,
                    "uri": "artifact://add-ids-2.jsonl.gz"}
    row["additions"]["rows"] = rows
    row["vector"] = {**row["vector"], "rows": rows}
    row["adjudication"] = {**row["adjudication"], "rows_declared": rows,
                           "rows_loaded": rows}
    manifests["sets"].append(row)
    plan["inputs"]["manifest_index"] = _write(tmp_path, "manifests.json",
                                              manifests)

    fold = json.loads((tmp_path / "fold.json").read_text())
    fold["conservation"][second] = dict(fold["conservation"][ADDITION])
    fold["additions"][second] = {**fold["additions"][ADDITION], "rows": rows}
    fold["tombstone_intersection"][second] = 0
    plan["records"]["fold_report"] = _write(tmp_path, "fold.json", fold)

    witness = _addition_witness()
    witness["set_id"] = second
    witness["guard_1_enumeration_completeness"].update(
        {"vector_rows": rows, "split_net_new": rows})
    plan["records"][f"addition_witness:{second}"] = _write(
        tmp_path, "addition-witness-2.json", witness)

    plan["expected_sets"] = sorted(plan["expected_sets"] + [second])
    plan["additions"]["sets"].append(
        {"set_id": second, "rows": rows, "claimed_production_points": 0,
         "row_ids": {"uri": "artifact://add-ids-2.jsonl.gz",
                     "sha256": "6" * 64, "count": rows}})
    plan["additions"]["count"] = ADDITION_ROWS + rows
    plan["additions"]["collection_points_after_additions"] = (
        MEMBERSHIP + ADDITION_ROWS + rows)
    plan["apply"]["loaded_row_ids"]["distinct"] += rows
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    report = run(tmp_path, plan_path)
    assert report["verdict"] == "GREEN", json.dumps(report["guards"], indent=1)
    g3 = next(guard for guard in report["guards"] if guard["id"] == "G3")
    assert g3["counts"]["additions"] == ADDITION_ROWS + rows
    g8 = next(guard for guard in report["guards"] if guard["id"] == "G8")
    assert g8["counts"]["net_new_additions_out_of_scope"] == 2

    # and the second set cannot borrow the first one's proof
    report = mutate(tmp_path, plan_path,
                    lambda p: p["records"].pop(f"addition_witness:{second}"))
    assert status_of(report, "G7") == FAIL
    assert second in json.dumps(report)


# -- G1: the enumeration burden moves, it does not shrink --------------------


def test_g1_fails_an_addition_that_does_not_enumerate_its_row_ids(
        with_addition):
    """An empty claim because the set binds nothing, and an empty claim because
    the set enumerated nothing, are indistinguishable unless it publishes the
    IDs it does own."""
    report = add_mutate(with_addition,
                        manifest_fn=lambda row: row["new_point_ids"].pop("sha256"))
    assert status_of(report, "G1") == FAIL
    assert "does not enumerate its minted row IDs under a digest" in \
        json.dumps(report)


def test_g1_fails_an_addition_that_claims_a_production_point(with_addition):
    report = add_mutate(
        with_addition,
        manifest_fn=lambda row: row["claims"].update({"count": 1}))
    assert status_of(report, "G1") == FAIL
    assert "a net-new addition binds none" in json.dumps(report)


def test_g1_fails_an_addition_that_does_not_disclaim_production_points(
        with_addition):
    report = add_mutate(
        with_addition,
        manifest_fn=lambda row: row.pop("contributes_production_points"))
    assert status_of(report, "G1") == FAIL
    assert "contributes_production_points: false" in json.dumps(report)


def test_g1_fails_an_addition_that_declares_a_delete_set(with_addition):
    """A replacement wearing the wrong label: the applier would delete points
    nothing in the additions term accounts for."""
    report = add_mutate(
        with_addition,
        manifest_fn=lambda row: row.update(
            {"delete_set": {"count": 5, "uri": "artifact://d.jsonl.gz"}}))
    assert status_of(report, "G1") == FAIL
    assert "declares a non-empty delete_set" in json.dumps(report)


def test_g1_fails_an_addition_that_enumerates_nothing(with_addition):
    report = add_mutate(
        with_addition,
        manifest_fn=lambda row: row["new_point_ids"].update({"count": 0}))
    assert status_of(report, "G1") == FAIL
    assert "enumerates no minted row ID" in json.dumps(report)


# -- G3: the plan must say it, and say it right ------------------------------


def test_g3_fails_closed_when_an_addition_set_has_no_additions_term(
        with_addition):
    """The task-500 gap exactly: the collection holds membership + additions
    and the plan says membership."""
    report = add_mutate(with_addition, plan_fn=lambda plan: plan.pop("additions"))
    assert status_of(report, "G3") == FAIL
    assert "declares no `additions` term" in json.dumps(report)


def test_g3_fails_an_additions_term_for_a_population_that_does_not_exist(green):
    tmp_path, plan_path = green
    report = mutate(tmp_path, plan_path, lambda plan: plan.update(
        {"additions": {"count": 5, "collection_points_after_additions":
                       MEMBERSHIP + 5, "sets": []}}))
    assert status_of(report, "G3") == FAIL
    assert "describes a population that does not exist" in json.dumps(report)


def test_g3_reads_an_empty_additions_term_as_no_additions(green):
    """An empty term and no term say the same thing; failing the producer who
    writes it out is failing the honest one."""
    tmp_path, plan_path = green
    report = mutate(tmp_path, plan_path, lambda plan: plan.update(
        {"additions": {"count": 0, "sets": [],
                       "collection_points_after_additions": MEMBERSHIP}}))
    assert status_of(report, "G3") == PASS
    assert "no set adds net-new rows" in json.dumps(report)


def test_g3_fails_an_additions_count_that_misses_its_own_sets(with_addition):
    report = add_mutate(
        with_addition,
        plan_fn=lambda plan: plan["additions"].update({"count": 17_960}))
    assert status_of(report, "G3") == FAIL
    assert "its own sets sum to" in json.dumps(report)


def test_g3_fails_a_plan_that_disagrees_with_the_manifest_on_the_rows(
        with_addition):
    report = add_mutate(
        with_addition,
        plan_fn=lambda plan: plan["additions"]["sets"][0].update({"rows": 10}))
    assert status_of(report, "G3") == FAIL
    assert "plan says 10 rows, the manifest declares" in json.dumps(report)


def test_g3_fails_a_misstated_collection_size(with_addition):
    """The one sentence the disposition identity cannot say, said wrong."""
    report = add_mutate(
        with_addition,
        plan_fn=lambda plan: plan["additions"].update(
            {"collection_points_after_additions": MEMBERSHIP}))
    assert status_of(report, "G3") == FAIL
    assert "membership + " in json.dumps(report)


def test_g3_fails_an_addition_the_plan_forgot_to_list(with_addition):
    """Emptying the term is the same as omitting it, and omitting it while an
    addition set exists is the task-500 gap."""
    report = add_mutate(
        with_addition,
        plan_fn=lambda plan: plan["additions"].update(
            {"sets": [], "count": 0,
             "collection_points_after_additions": MEMBERSHIP}))
    assert status_of(report, "G3") == FAIL
    assert "declares no `additions` term" in json.dumps(report)


def test_g3_fails_an_additions_term_that_lists_the_wrong_set(with_addition):
    report = add_mutate(
        with_addition,
        plan_fn=lambda plan: plan["additions"]["sets"][0].update(
            {"set_id": "some_other_set"}))
    assert status_of(report, "G3") == FAIL
    assert "the manifest index declares" in json.dumps(report)


def test_g3_fails_an_addition_whose_row_ids_the_plan_does_not_pin(
        with_addition):
    report = add_mutate(
        with_addition,
        plan_fn=lambda plan: plan["additions"]["sets"][0]["row_ids"].pop(
            "sha256"))
    assert status_of(report, "G3") == FAIL
    assert "pins no digested row-ID artifact" in json.dumps(report)


def test_g3_still_sums_the_disposition_over_the_membership_alone(
        with_addition):
    """An addition must not be folded into a disposition bucket to make some
    other total close -- the identity is over the membership, and 17,959 rows
    that were never members would silently displace 17,959 that were."""
    report = add_mutate(
        with_addition,
        plan_fn=lambda plan: plan["disposition"].update(
            {"replaced": plan["disposition"]["replaced"] + ADDITION_ROWS}))
    assert status_of(report, "G3") == FAIL
    assert "disposed != " in json.dumps(report)


# -- G5: rows == minted IDs == vectors ---------------------------------------


def test_g5_fails_an_addition_whose_vectors_and_minted_ids_disagree(
        with_addition):
    report = add_mutate(
        with_addition,
        manifest_fn=lambda row: row["vector"].update({"rows": 17_958}))
    assert status_of(report, "G5") == FAIL
    assert "must satisfy rows == minted IDs" in json.dumps(report)


def test_g5_fails_an_addition_the_fold_measured_as_short(with_addition):
    """The plan and the manifest can agree with each other and both be wrong;
    the fold counted the IDs that are actually on disk."""
    report = add_mutate(
        with_addition,
        fold_fn=lambda fold: fold["additions"][ADDITION].update(
            {"conserves": False}))
    assert status_of(report, "G5") == FAIL
    assert "counted the minted IDs on disk" in json.dumps(report)


def test_g5_waits_when_the_fold_has_not_counted_the_minted_ids(with_addition):
    report = add_mutate(
        with_addition, fold_fn=lambda fold: fold.pop("additions"))
    assert status_of(report, "G5") == WAITING
    assert "have not been counted on disk" in json.dumps(report)


# -- G7: a minted ID must not already be somebody's point --------------------


def test_g7_fails_an_addition_with_no_disjointness_witness(with_addition):
    """Absence of a proof is not zero. The guard cannot walk 34.4M points
    itself, so it refuses the set rather than assuming the join was done."""
    report = add_mutate(
        with_addition,
        plan_fn=lambda plan: plan["records"].pop(
            f"addition_witness:{ADDITION}"))
    assert status_of(report, "G7") == FAIL
    assert "no addition witness is bound" in json.dumps(report)


def test_g7_fails_an_addition_whose_ids_are_already_membership_points(
        with_addition):
    report = add_mutate(
        with_addition,
        witness_fn=lambda w: w["guard_2_disjointness"].update(
            {"membership_hits": 3}))
    assert status_of(report, "G7") == FAIL
    assert "already membership points" in json.dumps(report)


def test_g7_fails_a_witness_proven_against_a_different_membership(
        with_addition):
    """A disjointness proof over some other membership dump proves nothing
    about the one this plan builds against."""
    report = add_mutate(
        with_addition,
        manifest_fn=lambda row: row["membership_basis"].update(
            {"receipt_sha256": "9" * 64}))
    assert status_of(report, "G7") == FAIL
    assert "the plan builds against" in json.dumps(report)


def test_g7_fails_an_addition_that_names_no_membership_basis(with_addition):
    report = add_mutate(with_addition,
                        manifest_fn=lambda row: row.pop("membership_basis"))
    assert status_of(report, "G7") == FAIL
    assert "names no membership basis" in json.dumps(report)


def test_g7_fails_a_witness_that_covers_part_of_the_population(with_addition):
    """The task-497 lesson in another lane: a record covering all but one row
    covers nothing -- the missing row is the one that loads."""
    report = add_mutate(
        with_addition,
        witness_fn=lambda w: w["guard_1_enumeration_completeness"].update(
            {"vector_rows": ADDITION_ROWS - 1}))
    assert status_of(report, "G7") == FAIL
    assert "proves nothing about the rest of it" in json.dumps(report)


def test_g7_fails_a_witness_bound_to_the_wrong_set(with_addition):
    report = add_mutate(with_addition,
                        witness_fn=lambda w: w.update({"set_id": "somebody_else"}))
    assert status_of(report, "G7") == FAIL
    assert "binding a witness to the wrong" in json.dumps(report)


def test_g7_fails_a_witness_with_the_wrong_schema(with_addition):
    report = add_mutate(with_addition,
                        witness_fn=lambda w: w.update({"schema": "something.else"}))
    assert status_of(report, "G7") == FAIL


def test_g7_waits_on_a_witness_that_could_not_read_every_artifact(
        with_addition):
    """Unreadable is neither clean nor broken; WAITING still prohibits a flip."""
    report = add_mutate(
        with_addition,
        witness_fn=lambda w: w["guard_2_disjointness"]["cross_set"].update(
            {"unchecked_artifacts": {"gesetze.claims": "enumerates nothing"}}))
    assert status_of(report, "G7") == WAITING
    assert "could not be read" in json.dumps(report)


def test_g7_fails_an_addition_whose_ids_the_fold_found_in_another_claim(
        with_addition):
    report = add_mutate(
        with_addition,
        fold_fn=lambda fold: fold["additions"][ADDITION].update(
            {"minted_ids_in_another_sets_claims": {"atomic_loser": 2}}))
    assert status_of(report, "G7") == FAIL
    assert "inside another set's claim" in json.dumps(report)


def test_g7_fails_an_addition_whose_ids_are_tombstoned(with_addition):
    report = add_mutate(
        with_addition,
        fold_fn=lambda fold: fold["additions"][ADDITION].update(
            {"minted_ids_tombstoned": 1}))
    assert status_of(report, "G7") == FAIL
    assert "ADR-091 tombstoned points" in json.dumps(report)


def test_g7_fails_an_addition_whose_ids_sit_in_the_census_remainder(
        with_addition):
    report = add_mutate(
        with_addition,
        fold_fn=lambda fold: fold["additions"][ADDITION].update(
            {"minted_ids_in_census_remainder": 4}))
    assert status_of(report, "G7") == FAIL
    assert "inside the census remainder" in json.dumps(report)


def test_g7_counts_the_additions_rows_in_the_loaded_id_total(with_addition):
    """The additions load real rows, so they belong in the distinct-ID sum --
    an addition excused from that sum is one the applier can double-load."""
    report = add_mutate(
        with_addition,
        plan_fn=lambda plan: plan["apply"]["loaded_row_ids"].update(
            {"distinct": plan["apply"]["loaded_row_ids"]["distinct"]
             - ADDITION_ROWS}))
    assert status_of(report, "G7") == FAIL
    assert "some row ID is loaded twice" in json.dumps(report)


# -- G8: the skip, stated rather than silent ---------------------------------


def test_g8_excuses_an_addition_from_the_replacement_witness_by_name(
        with_addition):
    """A replacement witness says which membership points to SUBTRACT from the
    census remainder. An addition has none to subtract, and
    census_r0_remainder.py raises on a point that is not in the remainder."""
    report = run(*with_addition)
    g8 = next(row for row in report["guards"] if row["id"] == "G8")
    assert g8["status"] == PASS
    assert g8["counts"]["net_new_additions_out_of_scope"] == 1
    named = next(row for row in g8["checks"]
                 if row["name"] == "additions-out-of-census-scope")
    assert ADDITION in named["detail"]


def test_g8_still_fails_a_replacement_set_with_no_witness(with_addition):
    """The skip is scoped to additions and to nothing else: a replacement
    standing beside one is checked exactly as before."""
    report = remanifest(*with_addition,
                        lambda manifests: set_of(
                            manifests, "atomic_loser").pop("census_witness"))
    assert status_of(report, "G8") == FAIL
    assert "publish no census point witness" in json.dumps(report)
    assert "atomic_loser" in json.dumps(report)


# ---------------------------------------------------------------------------
# a guard that throws is a guard that fails
# ---------------------------------------------------------------------------


def test_an_exception_inside_a_guard_is_a_failure_not_a_skip(green, monkeypatch):
    def boom(ctx):
        raise RuntimeError("instrument broke")
    monkeypatch.setattr(M, "GUARDS", [boom if fn is M.guard_g4 else fn
                                      for fn in M.GUARDS])
    monkeypatch.setitem(M._GUARD_NAMES, "boom", ("G4", "tombstone exclusivity"))
    report = run(*green)
    assert status_of(report, "G4") == FAIL
    assert "instrument broke" in json.dumps(report)


# ---------------------------------------------------------------------------
# the real ledger
# ---------------------------------------------------------------------------


REPO = Path(__file__).resolve().parents[1]
REAL_PLAN = REPO / "data/assembly/assembly-plan.current.v1.json"


@pytest.mark.skipif(not REAL_PLAN.is_file(),
                    reason="the produced assembly plan is not in this clone")
def test_real_ledger_run_is_reproducible_and_not_green_yet():
    report = M.evaluate(REPO, REAL_PLAN)
    assert report["setup_error"] is None, report["setup_error"]
    assert len(report["guards"]) == 8
    # Until task 514 this asserted `flip_permitted is False` -- correctly, while
    # lanes F1/F2/C were unsealed and G5/G7 had no rows to check.  All eight now
    # PASS, so pinning the old verdict would be pinning a stale fact rather than
    # a property.  What is asserted instead is the property that mattered: the
    # verdict is a FUNCTION of the eight guards, never a separate opinion, in
    # both directions.
    statuses = [row["status"] for row in report["guards"]]
    assert report["flip_permitted"] is all(s == PASS for s in statuses)
    if report["flip_permitted"]:
        assert set(statuses) == {PASS}
    else:
        assert report["verdict"] in {"BLOCKED", "BURNING-DOWN"}
    # and it must be deterministic
    again = M.evaluate(REPO, REAL_PLAN)
    assert [row["status"] for row in report["guards"]] == \
           [row["status"] for row in again["guards"]]


@pytest.mark.skipif(not REAL_PLAN.is_file(),
                    reason="the produced assembly plan is not in this clone")
def test_real_ledger_run_reproduces_the_task456_counts():
    report = M.evaluate(REPO, REAL_PLAN)
    guards = {row["id"]: row for row in report["guards"]}
    assert guards["G4"]["counts"]["reclaimed"] == 4
    assert guards["G6"]["counts"]["mixed_urls"] == MIXED_URLS
    assert guards["G7"]["counts"]["identity_collapsed"] == COLLAPSED
    assert guards["G3"]["counts"]["membership"] == MEMBERSHIP
    # The arxiv named slice is now enumerated, so the ledger's multi-claim
    # population reconciles exactly instead of falling 176 short.
    assert guards["G2"]["counts"]["multi_claimed_points"] == MULTI_CLAIMED
    multi = next(row for row in guards["G2"]["checks"]
                 if row["name"] == "multi-claim-population")
    assert multi["status"] == PASS
    # ...and G2 PASSES: every pair the fold measured is ruled, and the
    # supplement's gate -- the ADR heading in decisions.md carrying no draft
    # marker -- now resolves.  (Until PR #80 merged on 2026-08-30 this asserted
    # the opposite; the ratification is what moved it, not a code change.)
    # G2 is one of eight, and this test is about G2's own counts. It used to
    # also assert `flip_permitted is False`; that was a fact about the other
    # seven guards on the day it was written, not about G2, and it stopped being
    # true at task 514 when all eight passed. The verdict is asserted where it
    # belongs -- in the test named for it.
    assert guards["G2"]["status"] == PASS
    amendment = guards["G2"]["counts"]["precedence_amendment"]
    assert amendment["ratified"] is True
    # Pinned to the supplement the plan actually binds rather than to a literal,
    # so extending the supplement cannot pass this test while leaving a pair
    # unruled -- and cannot fail it merely for having grown.
    bound = json.loads(REAL_PLAN.read_text())["records"]["precedence_supplement"]
    supplement = json.loads((REPO / bound["uri"]).read_text())
    assert amendment["pairs_ruled"] == len(supplement["lines"])


@pytest.mark.skipif(not REAL_PLAN.is_file(),
                    reason="the produced assembly plan is not in this clone")
def test_real_atomic_families_really_are_one_row_per_point():
    """Task 456 §5's premise, re-asserted against the sealed vector rollups."""
    report = M.evaluate(REPO, REAL_PLAN)
    g5 = next(row for row in report["guards"] if row["id"] == "G5")
    premise = next(row for row in g5["checks"]
                   if row["name"] == "atomic-families-are-one-row-per-point")
    assert premise["status"] == PASS
    assert premise["counts"]["sets"] == 34


@pytest.mark.skipif(not REAL_PLAN.is_file(),
                    reason="the produced assembly plan is not in this clone")
def test_real_run_names_every_unsealed_lane():
    """Every lane is closed, and closed for a REASON the plan can show.

    This test used to pin five open lanes (C, D, F1, F2, B2) carrying ~404k
    points.  Tim closed them on 2026-09-02 -- the remainder re-buckets to
    census E -- and task 505's census v34 executed that per point.  So the
    assertion moves with the truth rather than being deleted, and it keeps the
    part that matters: a lane may be absent only because it holds nothing, and
    an empty lane list must be accompanied by a reconciled accounting, not by
    a missing one.  A plan that simply forgot to state its lanes would pass the
    old test's negative and fail this one.
    """
    report = M.evaluate(REPO, REAL_PLAN)
    accounting = report["pending_lane_accounting"]
    assert accounting["lanes"] == 0
    assert accounting["points"] == 0
    plan = json.loads(REAL_PLAN.read_text())
    assert plan["pending_lanes"] == []
    assert plan["disposition"]["pending"] == 0
    reconciliation = plan["pending_lane_reconciliation"]
    assert reconciliation["unexplained_residue"] == 0
    assert reconciliation["gap"] == 0


@pytest.mark.skipif(not REAL_PLAN.is_file(),
                    reason="the produced assembly plan is not in this clone")
def test_real_run_keeps_contested_points_out_of_replaced():
    """A point two sets claim has no owner, so it gets its own bucket rather
    than being folded into `replaced` to make the arithmetic close.

    This test used to pin 178,557 contested points -- 60,388 across the four
    pairs ADR-101b rules, plus the 118,169 that became visible when task 477's
    claim set left quarantine.  ADR-101c then ratified a winner for all 13
    measured pairs, so `contested` is legitimately 0 and the check PASSes.
    The property the test exists for is unchanged and still asserted: every
    contested point that the plan reports must have a RATIFIED winner named
    pair by pair, so the only way to reach 0 is adjudication, never absorption
    into `replaced`.  `test_an_unratified_gate_rules_nothing`
    (`test_assembly_plan_adjudication.py`) holds the other direction: with no
    ratified gate the builder rules nothing and every such point stays
    contested.
    """
    report = M.evaluate(REPO, REAL_PLAN)
    guards = {row["id"]: row for row in report["guards"]}
    assert guards["G3"]["counts"]["contested"] == 0
    sums = next(row for row in guards["G3"]["checks"]
                if row["name"] == "sums-to-membership")
    assert sums["status"] == PASS
    contested = next(row for row in guards["G3"]["checks"]
                     if row["name"] == "no-contested-points")
    assert contested["status"] == PASS
    plan = json.loads(REAL_PLAN.read_text())
    adjudicated = plan["contested"]["adjudicated"]
    assert plan["contested"]["pairs"] == {}
    assert adjudicated["count"] > 0
    # every adjudicated pair names a winner that is one of its two sets
    for pair, winner in adjudicated["winners"].items():
        assert winner in set(pair.split(" x ")), pair


# ---------------------------------------------------------------------------
# G2 -- the precedence amendment and its ratification gate
# ---------------------------------------------------------------------------
#
# An amendment rules pairs the sha-pinned ledger never measured.  The gate that
# decides whether the suite may read it is the ratified instrument itself --
# the ADR heading in decisions.md -- and not a status field the amendment
# writes about itself.  These tests pin both directions and the refusal.


def _amendment(root: Path, *, ratified_by=None, ratified_at=None,
               lines=(("a x b", "a", 91),), decisions="## ADR-101b (DRAFT "
               "— ratified by merging this PR) — test\n") -> dict:
    (root / "decisions.md").write_text(decisions)
    return _write(root, "adr101b.json", {
        "schema": M.SCHEMA_PRECEDENCE_SUPPLEMENT,
        "adr": "ADR-101b",
        "ratification": {
            "pull_request": 80,
            "ratified_by": ratified_by, "ratified_at": ratified_at,
            "gate": {"kind": "adr_draft_marker_cleared",
                     "document": "decisions.md",
                     "heading_prefix": "## ADR-101b",
                     "draft_marker": "(DRAFT — ratified by merging this PR)"},
        },
        "lines": [{"id": f"L{index}", "pair": pair, "winner": winner,
                   "points": points}
                  for index, (pair, winner, points) in enumerate(lines)],
    })


def _with_overlap(root: Path, plan_path: Path, overlaps: dict,
                  amendment: dict | None) -> dict:
    fold = _write(root, "fold.json", {
        "conservation": {}, "tombstone_intersection": {"a": 0, "b": 0},
        "claim_overlaps": overlaps, "row_id_overlaps": {}})

    def apply(plan):
        plan["records"]["fold_report"] = fold
        if amendment is not None:
            plan["records"]["precedence_supplement"] = amendment
    return mutate(root, plan_path, apply)


def test_g2_waits_while_the_amendment_is_still_a_draft(green):
    root, plan_path = green
    report = _with_overlap(root, plan_path, {"a x b": 91}, _amendment(root))
    assert status_of(report, "G2") == WAITING
    assert "not ratified" in json.dumps(report)
    # WAITING is not a pass: the flip stays prohibited either way.  What it
    # buys is the distinction between "a named ruling is before the ratifier"
    # and "nobody has ruled", which is what the burn-down needs.
    assert report["verdict"] == "BURNING-DOWN"
    assert report["flip_permitted"] is False


def test_g2_reads_the_amendment_once_the_draft_marker_is_gone(green):
    root, plan_path = green
    amendment = _amendment(
        root, decisions="## ADR-101b — Precedence for the new sets\n")
    report = _with_overlap(root, plan_path, {"a x b": 91}, amendment)
    assert status_of(report, "G2") == PASS
    assert "adjudicated by ADR-101b" in json.dumps(report)


def test_g2_refuses_an_amendment_that_ratifies_itself(green):
    """A record that names a ratifier the instrument does not show is refused."""
    root, plan_path = green
    amendment = _amendment(root, ratified_by="Tim", ratified_at="2026-08-30")
    report = _with_overlap(root, plan_path, {"a x b": 91}, amendment)
    assert status_of(report, "G2") == FAIL
    assert "does not ratify itself" in json.dumps(report)


def test_g2_still_fails_a_pair_the_amendment_does_not_rule(green):
    """Ratified or not, an amendment only covers what it enumerates."""
    root, plan_path = green
    amendment = _amendment(
        root, decisions="## ADR-101b — ratified\n")
    report = _with_overlap(root, plan_path,
                           {"a x b": 91, "c x d": 118_169}, amendment)
    assert status_of(report, "G2") == FAIL
    assert "'c x d': 118169" in json.dumps(report)


def test_g2_refuses_an_amendment_whose_winner_is_not_in_the_pair(green):
    root, plan_path = green
    amendment = _amendment(root, lines=(("a x b", "somebody_else", 91),),
                           decisions="## ADR-101b — ratified\n")
    report = _with_overlap(root, plan_path, {"a x b": 91}, amendment)
    assert status_of(report, "G2") == FAIL
    assert "names a winner that is not one of the two sets" in json.dumps(report)


def test_g2_refuses_a_gate_it_cannot_evaluate(green):
    root, plan_path = green
    _amendment(root)
    payload = json.loads((root / "adr101b.json").read_text())
    payload["ratification"]["gate"]["kind"] = "trust_me"
    amendment = _write(root, "adr101b.json", payload)
    report = _with_overlap(root, plan_path, {"a x b": 91}, amendment)
    assert status_of(report, "G2") == FAIL
    assert "unrecognised ratification gate" in json.dumps(report)


# ---------------------------------------------------------------------------
# G7 -- convergent row IDs (ADR-101b Ruling 3a), behind the same gate
# ---------------------------------------------------------------------------


def _convergent(root: Path, *, rows: int, identical: int,
                pair: str = "a x b", load: str = "single") -> dict:
    return _write(root, "convergent.json", {
        "schema": M.SCHEMA_CONVERGENT_IDENTITY,
        "guard": "G7", "load": load,
        "rows": [{"point_id": f"00000000-0000-0000-0000-{index:012d}",
                  "pair": pair,
                  "embedded_text_byte_identical": index < identical}
                 for index in range(rows)],
    })


def _with_row_overlap(root: Path, plan_path: Path, overlaps: dict,
                      convergent: dict | None, amendment: dict | None) -> dict:
    fold = _write(root, "fold.json", {
        "conservation": {"a": {"superseded": 1, "delete_set": 1,
                               "identity_collapse": 0, "conserves": True}},
        "tombstone_intersection": {"a": 0},
        "claim_overlaps": {}, "row_id_overlaps": overlaps})

    def apply(plan):
        plan["records"]["fold_report"] = fold
        if convergent is not None:
            plan["records"]["g7_convergent_identity"] = convergent
        if amendment is not None:
            plan["records"]["precedence_supplement"] = amendment
    return mutate(root, plan_path, apply)


def test_g7_fails_a_row_id_collision_with_no_convergence_record(green):
    root, plan_path = green
    report = _with_row_overlap(root, plan_path, {"a x b": 17}, None, None)
    assert status_of(report, "G7") == FAIL
    assert "no proven-convergent record" in json.dumps(report)


def test_g7_waits_on_proven_convergence_while_the_amendment_is_a_draft(green):
    root, plan_path = green
    report = _with_row_overlap(root, plan_path, {"a x b": 17},
                               _convergent(root, rows=17, identical=17),
                               _amendment(root))
    assert status_of(report, "G7") == WAITING
    assert "convergent identities" in json.dumps(report)
    assert report["flip_permitted"] is False


def test_g7_passes_proven_convergence_once_the_amendment_is_ratified(green):
    root, plan_path = green
    report = _with_row_overlap(
        root, plan_path, {"a x b": 17},
        _convergent(root, rows=17, identical=17),
        _amendment(root, decisions="## ADR-101b — ratified\n"))
    assert status_of(report, "G7") == PASS
    assert "identical row ID because they emit an identical row" in json.dumps(report)


def test_g7_refuses_a_record_that_covers_all_but_one_row(green):
    """16 of 17 covers no pair: the odd row is the one that loads a vector
    nobody chose."""
    root, plan_path = green
    report = _with_row_overlap(
        root, plan_path, {"a x b": 17},
        _convergent(root, rows=17, identical=16),
        _amendment(root, decisions="## ADR-101b — ratified\n"))
    assert status_of(report, "G7") == FAIL
    assert "no proven-convergent record" in json.dumps(report)


def test_g7_refuses_a_record_that_does_not_declare_a_single_load(green):
    root, plan_path = green
    report = _with_row_overlap(
        root, plan_path, {"a x b": 17},
        _convergent(root, rows=17, identical=17, load="both"),
        _amendment(root, decisions="## ADR-101b — ratified\n"))
    assert status_of(report, "G7") == FAIL
    assert "must declare a single load" in json.dumps(report)
