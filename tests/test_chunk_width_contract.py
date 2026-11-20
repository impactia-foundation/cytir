#!/usr/bin/env python3
"""Task 541 — the guards that refuse a mis-declared row.

Two surfaces, both wired into the path they guard:

  * `payload_shape_contract.chunk_width` -- evaluated by `assembly_binder
    stage` AND `assembly_binder verify`, from the same receipts every other
    dimension uses.
  * `source_family_contract` -- `chunk_policy.policy_family` checked against
    `data/chunk-policy-registry.v1.json`, and a `declared` family refused when
    it selects a policy the registry does not mark `current`.

Each test states the defect it refuses. A guard with a test and no caller is
decoration (verification-doctrine 12), so `test_the_binder_calls_the_backfill`
asserts the wiring too.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cytir.chunk.registry as CREG
from cytir.index import payload_shape_contract as PSC
from cytir.sources import contract as SFC


REPO = pathlib.Path(__file__).resolve().parent.parent
REFERENCE_SHA = "73507461445b7a78602b0f3fc2f03f905f8fb59f74420d6e02b92b7df07de9b8"
GENERIC_SHA = "7dcb54c497e1855ad91ddafe80ff34b1da14f3d23b3af26dc7c7b53ab70ce6bf"
SET = "task541_chunk_reference_delta"


def base_receipt(**extra) -> dict:
    """A receipt shaped like the binder's own, passing every OTHER dimension,
    so a FAIL in these tests can only be the chunk-width arm."""
    return {"set_id": SET, "shard": "shard-0000", "staged": 1000,
            "serving_metadata": {"meta_row_missing_serving_signal": 0},
            "staged_without_sparse": 0,
            "provenance_head": {"rows": 1000, "rows_with_provenance_head": 0},
            **extra}


@pytest.fixture
def contract() -> dict:
    return PSC.load()


def width(**over) -> dict:
    base = {"rows": 1000, "rows_measured": 1000, "rows_over_hard_max": 0,
            "max_chunk_n_words": 288, "hard_max": 288,
            "policy_sha256": REFERENCE_SHA, "policy_family": "chunk.reference",
            "rows_selector_measured": 1000, "rows_mis_selected": 0,
            "declared_chunk_rules": {}, "partitions": 1}
    base.update(over)
    return base


def verdict_for(report: dict, set_id: str = SET) -> dict:
    return next(r for r in report["sets"] if r["set_id"] == set_id)


# ---------------------------------------------------------------------------
# the contract block itself
# ---------------------------------------------------------------------------

def test_the_delta_set_declares_every_required_field(contract):
    spec = PSC.chunk_width_limits(contract, SET)
    assert spec["policy_sha256"] == REFERENCE_SHA
    assert spec["max_rows_over_hard_max_frac"] == 0.0
    assert spec["min_chunk_n_words_presence"] == 1.0


def test_a_set_with_no_entry_is_not_checked(contract):
    assert PSC.chunk_width_limits(contract, "task417_families") is None


def test_a_partial_declaration_is_refused_not_defaulted(contract):
    doc = json.loads(json.dumps(contract))
    doc["chunk_width"]["half_declared"] = {"policy_sha256": REFERENCE_SHA}
    with pytest.raises(ValueError) as exc:
        PSC.chunk_width_limits(doc, "half_declared")
    assert "never inherit a default" in str(exc.value)


# ---------------------------------------------------------------------------
# the four arms
# ---------------------------------------------------------------------------

def test_a_conforming_set_passes(contract):
    r = PSC.evaluate([base_receipt(chunk_width=width())], contract)
    assert verdict_for(r)["verdict"] == "PASS", r["findings"]


def test_declared_but_unmeasured_is_a_fail(contract):
    """Task 535 section 2.3: the instrument did not exist, so the question
    could not be asked. Reporting that as PASS is how it stayed unasked."""
    r = PSC.evaluate([base_receipt()], contract)
    assert verdict_for(r)["verdict"] == "FAIL"
    assert any("no partition measured it" in f for f in r["findings"])


def test_a_different_policy_hash_is_a_fail(contract):
    r = PSC.evaluate(
        [base_receipt(chunk_width=width(policy_sha256=GENERIC_SHA))], contract)
    assert verdict_for(r)["verdict"] == "FAIL"
    assert any("different rules" in f for f in r["findings"])


def test_rows_that_cannot_state_their_width_are_a_fail(contract):
    r = PSC.evaluate(
        [base_receipt(chunk_width=width(rows_measured=800))], contract)
    assert verdict_for(r)["verdict"] == "FAIL"
    assert any("chunk_n_words" in f for f in r["findings"])


def test_a_row_wider_than_the_declared_policy_is_a_fail(contract):
    """The row was not cut by the policy the set names. This is the arm that
    catches a lane declaring chunk.reference and shipping epoch-4 widths."""
    r = PSC.evaluate(
        [base_receipt(chunk_width=width(rows_over_hard_max=1,
                                        max_chunk_n_words=1140))], contract)
    assert verdict_for(r)["verdict"] == "FAIL"
    assert any("word_guard.hard_max" in f for f in r["findings"])


def test_the_selector_conformance_arm_catches_task_535s_actual_defect(contract):
    """Every row correctly declared epoch-4; the PAGES in the wrong family.

    No count check, no hash check and no width check can see this -- which is
    exactly why 82 hosts moved 3.73x and the repo stayed green.
    """
    r = PSC.evaluate(
        [base_receipt(chunk_width=width(rows_mis_selected=1))], contract)
    assert verdict_for(r)["verdict"] == "FAIL"
    assert any("wrong family" in f for f in r["findings"])


def test_an_unstamped_lane_fails_the_selector_arm_rather_than_skipping_it(contract):
    r = PSC.evaluate(
        [base_receipt(chunk_width=width(rows_selector_measured=0))], contract)
    assert verdict_for(r)["verdict"] == "FAIL"
    assert any("chunk_selector" in f for f in r["findings"])


# ---------------------------------------------------------------------------
# the measurement, on a real staged-payload shape
# ---------------------------------------------------------------------------

def test_measure_reads_the_row_not_the_manifest(tmp_path):
    path = tmp_path / f"{SET}.shard-0000.payload.jsonl.gz"
    rows = [
        {"payload": {"chunk_n_words": 210, "chunk_rule": "ref@5",
                     "chunk_selector": {"policy_family": "chunk.reference"}}},
        {"payload": {"chunk_n_words": 900, "chunk_rule": "ref@5",
                     "chunk_selector": {"policy_family": "chunk.generic"}}},
        {"payload": {"chunk_rule": "ref@5"}},          # no width at all
    ]
    with gzip.open(path, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    got = PSC.measure_partition_chunk_width(
        [path], REFERENCE_SHA, 288, expect_policy_family="chunk.reference")
    assert got["rows"] == 3
    assert got["rows_measured"] == 2
    assert got["rows_over_hard_max"] == 1
    assert got["max_chunk_n_words"] == 900
    assert got["rows_selector_measured"] == 2
    assert got["rows_mis_selected"] == 1
    assert got["declared_chunk_rules"] == {"ref@5": 3}


def test_backfill_reads_the_hard_max_from_the_pinned_artifact(tmp_path, contract):
    """Not a literal in the guard. The policy's own `word_guard.hard_max` and
    the registry's own family are the source (engineering-doctrine 1)."""
    staged = tmp_path / "staged"
    staged.mkdir()
    path = staged / f"{SET}.shard-0000.payload.jsonl.gz"
    with gzip.open(path, "wt") as fh:
        fh.write(json.dumps({"payload": {"chunk_n_words": 288}}) + "\n")
    receipts = [base_receipt()]
    assert PSC.backfill_chunk_width(receipts, staged, contract, log=lambda *a: None) == 1
    assert receipts[0]["chunk_width"]["hard_max"] == 288
    assert receipts[0]["chunk_width"]["policy_family"] == "chunk.reference"
    assert receipts[0]["chunk_width"]["rows_over_hard_max"] == 0


def test_the_binder_calls_the_backfill_on_both_paths():
    """A guard nothing calls is decoration. Stage AND verify, or a tree staged
    before the dimension existed loads unchecked."""
    stage_src = (REPO / "cytir" / "index" / "assemble_stage.py").read_text()
    verify_src = (REPO / "cytir" / "index" / "assemble_verify.py").read_text()
    assert stage_src.count("PSC.backfill_chunk_width(") == 1
    assert verify_src.count("PSC.backfill_chunk_width(") == 1


# ---------------------------------------------------------------------------
# the family declaration
# ---------------------------------------------------------------------------

def test_the_shipped_declarations_still_pass():
    report = SFC.validate(SFC.load())
    assert report["problems"] == [], report["problems"]


def _generic_family(doc):
    """A family whose declared chunk_policy is `chunk.generic`.

    Task 565 re-cut the two families that used to declare it outright, so
    their declaration is now `chunk.reference` with a `chunk.generic`
    `unselected` half. These tests are about the VALIDATOR, not about which
    family happens to hold which policy today, so the fixture is BUILT from
    the ratified generic artifact rather than found in the data file -- which
    is what kept them from breaking every time the corpus moved.
    """
    fam = json.loads(json.dumps(doc["families"][0]))
    fam["chunk_policy"] = {
        "policy_id": "chunk-policy-struct-md-768w-o0-adjacent-sections-tail-1152-tok2560.v1",
        "policy_family": "chunk.generic",
        "sha256": hashlib.sha256(
            (REPO / "data" /
             "chunk-policy-struct-md-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json"
             ).read_bytes()).hexdigest(),
        "epoch": 4,
        "tokenizer_revision": "ac6544c8a46e00af67e330e85a9028c66b8cfd9a",
    }
    fam["unknowns"] = [u for u in fam.get("unknowns", [])
                       if not u.startswith("chunk_policy.")]
    return fam


def test_a_family_that_names_the_wrong_policy_family_is_refused():
    doc = SFC.load()
    fam = _generic_family(doc)
    fam["chunk_policy"]["policy_family"] = "chunk.reference"
    problems = SFC.validate_family(fam)
    assert any("registry puts" in p for p in problems), problems


def test_a_declared_family_may_now_select_the_ratified_reference_policy():
    """ADR-090 amendment 1 (PR #130) flipped chunk.reference to `current`.

    Until 2026-09-09 this test asserted the opposite -- that a declared family
    selecting it is refused because the policy was `proposed`. The REFUSAL is
    still tested, on a policy that is genuinely not selectable, by
    `test_a_declared_family_may_not_select_a_superseded_policy` below; what
    changed is this policy's status, not the guard.
    """
    doc = SFC.load()
    fam = _generic_family(doc)
    fam["status"] = "declared"
    fam["unknowns"] = []
    fam["chunk_policy"] = {
        "policy_id": "chunk-policy-struct-md-192w-o0-adjacent-sections-tail-288-tok2560.v1",
        "policy_family": "chunk.reference",
        "sha256": REFERENCE_SHA, "epoch": 5}
    problems = SFC.validate_family(fam)
    assert not any("registry status" in p for p in problems), problems


def test_a_declared_family_may_not_select_a_superseded_policy():
    """The registry-status refusal itself, on a policy that is not current."""
    doc = SFC.load()
    superseded = (REPO / "data" /
                  "chunk-policy-struct-md-768w-o0.v1.json")
    sha = hashlib.sha256(superseded.read_bytes()).hexdigest()
    fam = _generic_family(doc)
    fam["status"] = "declared"
    fam["unknowns"] = []
    fam["chunk_policy"] = {
        "policy_id": "chunk-policy-struct-md-768w-o0.v1",
        "policy_family": "chunk.generic",
        "sha256": sha, "epoch": 2}
    problems = SFC.validate_family(fam)
    assert any("registry status" in p and "superseded" in p
               for p in problems), problems


def test_a_policy_the_registry_does_not_know_is_refused(tmp_path):
    doc = SFC.load()
    fam = _generic_family(doc)
    # a real policy file the registry deliberately has no entry for
    stray = REPO / "data" / "chunk-policy-task541-stray-for-test.v1.json"
    stray.write_text(json.dumps({
        "schema": "aw.chunk-policy.v1", "policy_id": "stray",
        "policy_version": 1, "algorithm": "source-record-identity-v1",
        "epoch": 1, "rule_legend": "stray@1"}))
    try:
        fam["chunk_policy"] = {
            "policy_id": "chunk-policy-task541-stray-for-test.v1",
            "epoch": 1}
        problems = SFC.validate_family(fam)
        assert any("not in data/chunk-policy-registry" in p
                   for p in problems), problems
    finally:
        stray.unlink()


def test_the_ratified_table_names_the_ruling_for_the_new_rules():
    for key in ("chunk_policy.policy_family", "chunk_policy.selectable"):
        assert "ADR-090" in SFC.RATIFIED[key]
