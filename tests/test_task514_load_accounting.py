#!/usr/bin/env python3
"""Tests for task 514's four new rule evaluations.

Each one exists because the rule it evaluates was ratified and then not
evaluated, which is the failure mode this whole assembly has hit repeatedly:
ADR-101 Ruling 1's row-level suppression (never wired for either direction), the
cross-set collision losers (never charged to their sets), a sealed
`EXCLUDED.json` (never read), and the two `apply` terms guard G7 reads that no
producer wrote.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cytir.index import assemble_inputs as AI  # noqa: E402
from cytir.index import assemble_plan as AP  # noqa: E402
from cytir.ops import fold_assembly_claim_fragments as F  # noqa: E402


# --------------------------------------------------------------------------
# 1. the set-scoped row-suppression reader
# --------------------------------------------------------------------------

from tests.helpers.repo_files import repo_file_or_skip

repo_file_or_skip("data/assembly/assembly-plan.current.v1.json")  # the public cut withholds this file; skip there (task 2382)
repo_file_or_skip("data/assembly/claim-manifests.v1.json")  # the public cut withholds this file; skip there (task 2382)


def _write(tmp_path, rows, name="s.jsonl.gz"):
    p = tmp_path / name
    with gzip.open(p, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def test_row_suppression_is_keyed_by_scope_not_by_id(tmp_path):
    """Two sets staging the same id must be separable, or suppressing one
    drops both -- the 91-id veto task 511 §3 had to add by hand."""
    p = _write(tmp_path, [
        {"point_id": "a", "scope": "setA", "set_id": "setA"},
        {"point_id": "a", "scope": "setB", "set_id": "setB"},
        {"point_id": "b", "scope": "setA", "set_id": "setA"},
    ])
    got = AI.load_row_suppression(p)
    assert got == {"setA": {"a", "b"}, "setB": {"a"}}


def test_row_suppression_refuses_an_unscoped_entry(tmp_path):
    p = _write(tmp_path, [{"point_id": "a", "set_id": "setA"}])
    with pytest.raises(SystemExit) as excinfo:
        AI.load_row_suppression(p)
    assert "scope" in str(excinfo.value)


def test_row_suppression_absent_file_is_no_suppression():
    assert AI.load_row_suppression(None) == {}


# --------------------------------------------------------------------------
# 2. the fold's measured-load correction
# --------------------------------------------------------------------------

def _index(rows_loaded, rows_declared=100, set_id="setA"):
    return {"sets": [{"set_id": set_id, "contributes_vectors": True,
                      "adjudication": {"rows_declared": rows_declared,
                                       "rows_loaded": rows_loaded,
                                       "rows_suppressed":
                                           rows_declared - rows_loaded}}]}


def _measurement(tmp_path, declared_by_fold, computed, set_id="setA"):
    doc = {"sets": [{"set_id": set_id, "staged": 100, "collision_losers": 3,
                     "adr101_r1": 5, "adr101bc_still_staged": 0,
                     "not_loaded": 100 - computed,
                     "rows_loaded_computed": computed,
                     "rows_loaded_declared_by_fold": declared_by_fold}]}
    p = tmp_path / "m.json"
    p.write_text(json.dumps(doc))
    return p


def test_measured_load_correction_charges_the_set(tmp_path):
    index = _index(rows_loaded=100)
    out = F.apply_measured_load_suppression(
        index, _measurement(tmp_path, declared_by_fold=100, computed=92))
    adj = index["sets"][0]["adjudication"]
    assert adj["rows_loaded"] == 92
    assert adj["rows_suppressed"] == 8
    assert adj["rows_loaded"] + adj["rows_suppressed"] == adj["rows_declared"]
    assert out == [{"set_id": "setA", "rows_loaded_before": 100,
                    "rows_loaded_after": 92, "rows_suppressed_after": 8}]


def test_measured_load_correction_refuses_a_stale_measurement(tmp_path):
    """A measurement taken against a different fold is a wrong number with a
    provenance trail, which is worse than no number."""
    index = _index(rows_loaded=100)
    with pytest.raises(F.ArtifactDefect) as excinfo:
        F.apply_measured_load_suppression(
            index, _measurement(tmp_path, declared_by_fold=99, computed=92))
    assert "Re-measure" in str(excinfo.value)


def test_measured_load_correction_refuses_an_unknown_set(tmp_path):
    index = _index(rows_loaded=100, set_id="setA")
    with pytest.raises(F.ArtifactDefect):
        F.apply_measured_load_suppression(
            index, _measurement(tmp_path, 100, 92, set_id="setZ"))


def test_measured_load_correction_is_a_no_op_without_the_flag():
    index = _index(rows_loaded=100)
    assert F.apply_measured_load_suppression(index, None) == []
    assert index["sets"][0]["adjudication"]["rows_loaded"] == 100


# --------------------------------------------------------------------------
# 3. the plan's two computed apply terms, against the real artifacts
# --------------------------------------------------------------------------

REPO = HERE.parent
PLAN = REPO / "data" / "assembly" / "assembly-plan.current.v1.json"
MANIFESTS = REPO / "data" / "assembly" / "claim-manifests.v1.json"


def test_plan_declares_a_measured_distinct_loaded_id_count():
    plan = json.loads(PLAN.read_text())
    loaded = plan["apply"].get("loaded_row_ids")
    assert loaded is not None, "guard G7 FAILS without this"
    assert isinstance(loaded["distinct"], int) and loaded["distinct"] > 0
    assert loaded.get("source"), "a count with no provenance is an assertion"


def test_distinct_loaded_ids_equal_the_sum_of_rows_loaded():
    """G7's identity, asserted here too so a producer change that breaks it
    fails in seconds rather than at the end of a six-hour load."""
    plan = json.loads(PLAN.read_text())
    manifests = json.loads(MANIFESTS.read_text())
    total = sum(int((row.get("adjudication") or {}).get("rows_loaded", 0))
                for row in manifests["sets"] if row.get("contributes_vectors"))
    assert plan["apply"]["loaded_row_ids"]["distinct"] == total


def test_delete_set_is_measured_not_summed():
    plan = json.loads(PLAN.read_text())
    manifests = json.loads(MANIFESTS.read_text())
    delete_set = plan["apply"]["delete_set"]
    assert delete_set["count"] is not None, "G7 WAITED on this for four tasks"
    claims_sum = sum(int((row.get("claims") or {}).get("count") or 0)
                     for row in manifests["sets"]
                     if row.get("contributes_vectors")
                     and row.get("mode") != "net_new_addition")
    # the union is strictly smaller than the sum -- the sets' claims overlap,
    # which is exactly why the delete set may not be added up
    assert delete_set["claims_union"] < claims_sum
    assert (delete_set["count"]
            == delete_set["claims_union"] - delete_set["identity_collapse_union"])
    assert delete_set["excludes"] == plan["apply"]["identity_collapse"]["uri"]


# --------------------------------------------------------------------------
# 4. a sealed EXCLUDED.json outranks a findable STAGE-RECEIPT
# --------------------------------------------------------------------------

def _source_dir(tmp_path, name, *, excluded, receipt):
    d = tmp_path / "vectors" / "release=r" / f"source={name}"
    (d / "rollup-evidence" / "stage").mkdir(parents=True)
    if receipt:
        (d / "rollup-evidence" / "stage" / "STAGE-RECEIPT.json").write_text(
            json.dumps({"artifacts": [], "records": 0}))
    if excluded:
        (d / "EXCLUDED.json").write_text(json.dumps(
            {"schema": "aw.duplicate-vector-rollup-exclusion.v1",
             "admission": False, "status": "excluded"}))
    return d


def test_excluded_marker_is_checked_before_the_stage_receipt(tmp_path, monkeypatch):
    """The defect: `source=technical-manufacturer-catalog-admitted-products-v2`
    carries `admission: false` AND a good STAGE-RECEIPT, and the receipt was
    found first, so 445 rows planned and staged."""
    _source_dir(tmp_path, "kept", excluded=False, receipt=True)
    _source_dir(tmp_path, "marked", excluded=True, receipt=True)
    monkeypatch.setattr(AP, "load_pack_route_extra_sources", lambda: [])
    monkeypatch.setattr(AP, "load_pack_stage_overrides", lambda: {})
    monkeypatch.setattr(AP, "load_pack_route_exclusions", lambda *a, **k: {})
    by_set: dict = {}
    pop = {"set_id": "task417_families", "mode": "replacement",
           "vectors": "artifact://vectors/release=r/source=kept/x"}
    AP._plan_pack_sources(pop, tmp_path, by_set)
    # neither source resolves a payload shard, but the MARKED one must be
    # refused for being marked, not for the receipt it happens to carry
    assert by_set["task417_families"]["sources"] == 2
    assert by_set["task417_families"]["unresolved"] == 2


# --------------------------------------------------------------------------
# 5. the two post-load coverage checks are properties of a FULL pass
# --------------------------------------------------------------------------

def test_post_load_checks_are_guarded_against_a_resume():
    """`winners_seen` is built while upserting, so a resumed partition
    contributes nothing to it. On this build's first resume that turned a
    correct 4-id finding into a spurious 11,474-id refusal, and the
    collision-winner check still guards against it by saying NOT EVALUATED
    rather than firing.

    `suppressed_by_set` had the same hole (a spurious 26-scope refusal) and
    was ALSO guarded this way originally -- but task 515 (commit f77df69f,
    "the load-suppression identity is not declared == skipped-at-load")
    retired that guard as a worse fix than the alternative: `per_scope` is
    now aggregated from the per-partition `*.loaded.json` receipts, which
    are a full-pass record regardless of how many processes it took, so the
    row-suppression check evaluates correctly on a resume instead of being
    skipped. Pinning both checks to the same guard string would re-fail the
    moment task 515's fix reproduces standalone -- test the current,
    stronger design instead."""
    src = (HERE.parent / "cytir" / "index" / "assemble_load.py").read_text()
    assert "if collision_winners and resumed_partitions:" in src
    assert "winner-map coverage NOT EVALUATED" in src
    # row_suppression is no longer skipped on a resume (task515 f77df69f) --
    # it is evaluated unconditionally from the per-partition receipts.
    assert "if row_suppression and resumed_partitions:" not in src
    assert "row-suppression coverage NOT EVALUATED" not in src
    assert "if row_suppression:" in src
    assert "Evaluated over the WHOLE build from the per-partition receipts" in src


def test_a_forfeited_winner_is_distinguished_from_a_map_build_disagreement():
    """An adjudicated id can fail to load from its winner because the winner
    never staged it (a real defect) or because a ratified precedence forfeited
    the winner's row (ADR-101b Ruling 3a's LABEL tie-break meeting ADR-101
    Ruling 1).  The second is reported by id; only the first refuses."""
    src = (HERE.parent / "cytir" / "index" / "assemble_load.py").read_text()
    assert "forfeited_winner" in src and "disagreement" in src
    assert "ratified precedence forfeited it" in src
    # only the disagreement branch refuses
    refusal = src.split("forfeited_winner = sorted(")[1]
    assert "REFUSING" in refusal.split("if disagreement:")[1]
    assert "REFUSING" not in refusal.split("if disagreement:")[0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
