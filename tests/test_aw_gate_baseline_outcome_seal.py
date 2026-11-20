"""The CI-fixture gate baseline cannot change a gold outcome without a policy change.

Two-step re-record. `AW_GATE_RECORD_BASELINE=1` (in the index-serve gate test) re-writes the
baseline, including its `golds_outcome_digest` (sha256 over the sorted id, hit_at_k, rank and
rank_exact of every gold). It never writes the contract file. A re-record may therefore move
plumbing fields (profile digests, timestamps, ports, `top` tails) and this test stays green. If a
gold's hit or rank moved, the digest moves, this test fails, and it stays red until the sealed
contract `data/ci-fixture-gate-contract.v1.json` changes in a POLICY PR that the orchestrator or
Tim reviews.
"""
import json
import pathlib

from cytir.gates.slice_gate import golds_outcome_digest

REPO = pathlib.Path(__file__).resolve().parents[1]
BASELINE = REPO / "data" / "gate-slices" / "ci-fixture.baseline.v1.json"
CONTRACT = REPO / "data" / "ci-fixture-gate-contract.v1.json"


def test_the_baseline_gold_outcomes_match_the_sealed_contract():
    baseline = json.loads(BASELINE.read_text())
    contract = json.loads(CONTRACT.read_text())
    golds = baseline["golds"]
    recomputed = golds_outcome_digest(golds)
    assert baseline["golds_outcome_digest"] == recomputed, "baseline digest is not its golds' digest"
    assert len(golds) == contract["golds_n"]
    assert recomputed == contract["golds_outcome_digest"], (
        "a gold's hit_at_k, rank or rank_exact differs from the sealed contract: changing it "
        "needs the contract file changed in a POLICY PR")


def test_the_digest_sees_outcomes_only():
    rows = json.loads(BASELINE.read_text())["golds"]
    base = golds_outcome_digest(rows)
    moved = [dict(r, top=["x"], query="y") for r in rows]
    assert golds_outcome_digest(moved) == base
    worse = [dict(r) for r in rows]
    worse[0]["rank"] = (worse[0]["rank"] or 0) + 1
    assert golds_outcome_digest(worse) != base
