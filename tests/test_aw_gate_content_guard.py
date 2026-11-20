"""task 626 (Tim's standing catch-up waiver, GO-626-standing-catchup
condition 3): every gold whose fused rank is worse in the candidate than
the concurrent production baseline must have an IDENTICAL dense leg (ids
and order) -- any difference means something besides the known BM25-IDF
instrument property (task 627, ADR-109) moved that gold, and the day must
STOP rather than proceed under the standing waiver.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from cytir.gates import content_guard as cgd
from cytir.gates import convergence as gc


def arm(ranks: dict, dense: dict | None = None, sets=gc.BAR_SETS) -> dict:
    """One arm scoring one gold per named set, at the given rank, with a
    `trace.dense` id list (defaulting to a fixed 3-id list unless overridden
    per set by `dense`)."""
    dense = dense or {}
    out = {"sets": {}}
    for name in sets:
        out["sets"][name] = {"per_q": {
            f"{name}-1": {
                "rank": ranks.get(name, 1),
                "trace": {"dense": dense.get(name, ["d1", "d2", "d3"]),
                          "bm25": ["b1", "b2", "b3"]}}}}
    return out


def test_no_loss_no_violation_possible():
    baseline = arm({})
    candidate = arm({})
    receipt = cgd.evaluate(baseline, candidate)
    assert receipt["passed"] is True
    assert receipt["lost_golds"] == 0
    assert receipt["violations"] == []


def test_a_loss_with_identical_dense_passes_the_guard():
    """The mechanism-consistent case: rank moved, dense leg untouched --
    exactly what task 627 found for the real 09-17 pack."""
    baseline = arm({"legal": 1})
    candidate = arm({"legal": 3})  # worse
    receipt = cgd.evaluate(baseline, candidate)
    assert receipt["passed"] is True
    assert receipt["lost_golds"] == 1
    assert receipt["checked"][0]["dense_identical"] is True


def test_a_loss_with_a_different_dense_leg_stops_the_guard():
    baseline = arm({"legal": 1})
    candidate = arm({"legal": 3}, dense={"legal": ["dX", "d2", "d3"]})
    receipt = cgd.evaluate(baseline, candidate)
    assert receipt["passed"] is False
    assert receipt["lost_golds"] == 1
    assert len(receipt["violations"]) == 1
    v = receipt["violations"][0]
    assert v["qid"] == "legal/legal-1"
    assert v["baseline_dense"] == ["d1", "d2", "d3"]
    assert v["candidate_dense"] == ["dX", "d2", "d3"]


def test_dense_order_counts_not_just_membership():
    """Same ids, different order -- still a violation. The gate's own
    convergence rule (`gate_convergence.comparable_ids`) is rank-based, not
    set-based, for exactly this reason; the guard holds the same line."""
    baseline = arm({"names": 1})
    candidate = arm({"names": 2}, dense={"names": ["d2", "d1", "d3"]})
    receipt = cgd.evaluate(baseline, candidate)
    assert receipt["passed"] is False
    assert receipt["violations"][0]["qid"] == "names/names-1"


def test_a_gold_that_moved_but_did_not_get_WORSE_is_never_checked():
    """A rank IMPROVEMENT is not a loss; the guard only exists to catch a
    real content defect hiding behind a REFUSE, and only losses can be
    that."""
    baseline = arm({"legal": 3})
    candidate = arm({"legal": 1}, dense={"legal": ["dX", "d2", "d3"]})
    receipt = cgd.evaluate(baseline, candidate)
    assert receipt["passed"] is True
    assert receipt["lost_golds"] == 0


def test_absent_from_top10_is_the_worst_rank_and_is_checked():
    baseline = arm({"foundation": 1})
    candidate = arm({"foundation": None})
    receipt = cgd.evaluate(baseline, candidate)
    assert receipt["lost_golds"] == 1
    assert receipt["checked"][0]["after_rank"] is None


def test_a_missing_trace_is_unknown_never_a_pass():
    baseline = arm({"legal": 1})
    candidate = arm({"legal": 3})
    del candidate["sets"]["legal"]["per_q"]["legal-1"]["trace"]
    with pytest.raises(cgd.ContentGuardUnknown, match="no trace recorded"):
        cgd.evaluate(baseline, candidate)


def test_a_missing_query_is_unknown_never_a_pass():
    baseline = arm({"legal": 1})
    candidate = arm({"legal": 3})
    del candidate["sets"]["legal"]["per_q"]["legal-1"]
    with pytest.raises(cgd.ContentGuardUnknown, match="absent from this arm"):
        cgd.evaluate(baseline, candidate)


def test_render_names_every_violation():
    baseline = arm({"legal": 1, "names": 1})
    candidate = arm({"legal": 3, "names": 2},
                    dense={"legal": ["dX"], "names": ["dY"]})
    receipt = cgd.evaluate(baseline, candidate)
    text = "\n".join(cgd.render(receipt))
    assert "STOP" in text
    assert "legal/legal-1" in text
    assert "names/names-1" in text


def test_render_says_pass_when_clean():
    receipt = cgd.evaluate(arm({}), arm({}))
    text = "\n".join(cgd.render(receipt))
    assert text.startswith("CONTENT GUARD (strict): PASS")


def test_the_default_population_is_the_gates_own_bar_sets():
    """One home: the guard checks exactly the population the daily gate's
    own verdict is computed on, never a wider or narrower set."""
    import inspect
    assert "names: Iterable[str] = gc.BAR_SETS" in inspect.getsource(cgd.evaluate)


# --------------------------------------------------------- pool mode ------
# task626/627 (Tim, 2026-09-21): the strict guard STOPPED 09-15 on two lost
# golds -- authoritative/de-fresh-005 (rank 10->None) and foundation/eu-001
# (rank 2->6) -- that Tim read by hand and found to be cross-mirror HNSW
# tail variation, not content from the pack: the dense leg's first
# disagreement was at position 56 and 66 respectively (both beyond the
# ADR-035 k-stability depth of 50), and NONE of the new dense/BM25 ids on
# either gold were pack rows (independently re-verified against the real
# pack's own 15,542 ids -- artifact + tombstoned predecessors -- before
# sealing this fixture). `task626_content_guard_pool_
# fixture.json` is the real projection of those two golds (plus one
# unchanged control gold per set) from the actual arms
# (`task626f-guard-20260915/{prod-baseline,twin-candidate}`).
POOL_FIXTURE = json.loads((pathlib.Path(__file__).resolve().parents[0] /
                           "testdata" /
                           "task626_content_guard_pool_fixture.json").read_text())


def _pool_arm(side: str) -> dict:
    """Reshape the fixture's {qid: {baseline, candidate}} projection into
    one arm-shaped dict (`sets[name].per_q[qid] = <that side's row>`)."""
    sets = {}
    for name, block in POOL_FIXTURE["sets"].items():
        sets[name] = {"per_q": {qid: row[side] for qid, row in block["per_q"].items()}}
    return {"sets": sets}


def test_the_fixture_is_the_real_09_15_arms():
    baseline, candidate = _pool_arm("baseline"), _pool_arm("candidate")
    assert baseline["sets"]["authoritative"]["per_q"]["de-fresh-005"]["rank"] == 10
    assert candidate["sets"]["authoritative"]["per_q"]["de-fresh-005"]["rank"] is None
    assert baseline["sets"]["foundation"]["per_q"]["eu-001"]["rank"] == 2
    assert candidate["sets"]["foundation"]["per_q"]["eu-001"]["rank"] == 6


def test_strict_mode_stops_on_the_real_09_15_pair():
    """Reproduces what actually happened: the strict guard (correctly, by
    its own contract) STOPPED this pack -- any dense difference on a lost
    gold is a violation, full stop, no attribution reasoning."""
    baseline, candidate = _pool_arm("baseline"), _pool_arm("candidate")
    receipt = cgd.evaluate(baseline, candidate, names=("authoritative", "foundation"))
    assert receipt["passed"] is False
    assert receipt["lost_golds"] == 2
    assert len(receipt["violations"]) == 2


def test_pool_mode_clears_the_real_09_15_pair_at_depth_50():
    """The refinement: neither lost gold's dense/BM25 movement is
    attributable to the pack (no pack rows among the new ids on either
    gold, confirmed against the real pack's actual id set), and both
    golds' first dense disagreement (56, 66) falls OUTSIDE the ADR-035
    k-stability depth of 50 -- so under `pool` mode this pack PASSES."""
    baseline, candidate = _pool_arm("baseline"), _pool_arm("candidate")
    pack_ids = set()  # real check already proved 0 pack-row hits either way
    receipt = cgd.evaluate_pool(baseline, candidate, pack_ids,
                                names=("authoritative", "foundation"), depth=50)
    assert receipt["passed"] is True
    assert receipt["lost_golds"] == 2
    assert receipt["violations"] == []
    by_qid = {row["qid"]: row for row in receipt["checked"]}
    fresh = by_qid["authoritative/de-fresh-005"]
    assert fresh["first_dense_diff_position"] == 56
    assert fresh["dense_pack_row_count"] == 0
    eu = by_qid["foundation/eu-001"]
    assert eu["first_dense_diff_position"] == 66
    assert eu["dense_pack_row_count"] == 0


def test_pool_mode_still_stops_if_a_new_dense_id_is_a_pack_row():
    """The refinement is not a blanket pass -- if the SAME pair had a pack
    row among the new dense ids, it would still stop, at any depth."""
    baseline, candidate = _pool_arm("baseline"), _pool_arm("candidate")
    b_dense = baseline["sets"]["authoritative"]["per_q"]["de-fresh-005"]["trace"]["dense"]
    c_dense = candidate["sets"]["authoritative"]["per_q"]["de-fresh-005"]["trace"]["dense"]
    new_ids = set(c_dense) - set(b_dense)
    assert new_ids, "the fixture's own new-dense-ids set must be non-empty for this test to mean anything"
    receipt = cgd.evaluate_pool(baseline, candidate, {next(iter(new_ids))},
                                names=("authoritative", "foundation"), depth=50)
    assert receipt["passed"] is False
    row = receipt["violations"][0]
    assert row["qid"] == "authoritative/de-fresh-005"
    assert row["reason"] == "dense_new_id_is_pack_row"
    assert row["dense_pack_row_count"] >= 1


def test_pool_mode_stops_when_the_diff_is_within_depth():
    """A depth wide enough to cover position 56 catches de-fresh-005 via
    criterion (b) even with no pack-row hit."""
    baseline, candidate = _pool_arm("baseline"), _pool_arm("candidate")
    receipt = cgd.evaluate_pool(baseline, candidate, set(),
                                names=("authoritative", "foundation"), depth=60)
    by_qid = {row["qid"]: row for row in receipt["checked"]}
    assert by_qid["authoritative/de-fresh-005"]["violation"] is True
    assert by_qid["authoritative/de-fresh-005"]["reason"] == \
        "first_dense_diff_within_pool_depth"
    # eu-001's first diff (66) is still beyond depth 60
    assert by_qid["foundation/eu-001"]["violation"] is False


def test_pool_depth_reads_the_env_with_the_adr035_default():
    assert cgd.DEFAULT_POOL_DEPTH == 50
    old = os.environ.pop(cgd.POOL_DEPTH_ENV, None)
    try:
        assert cgd.pool_depth() == 50
        os.environ[cgd.POOL_DEPTH_ENV] = "77"
        assert cgd.pool_depth() == 77
    finally:
        if old is None:
            os.environ.pop(cgd.POOL_DEPTH_ENV, None)
        else:
            os.environ[cgd.POOL_DEPTH_ENV] = old


def test_pool_mode_never_touched_by_the_default_strict_evaluate():
    """AW_GUARD_MODE=pool is opt-in -- `evaluate()` (used everywhere else,
    including the live door) has no idea `evaluate_pool` exists."""
    import inspect
    assert "evaluate_pool" not in inspect.getsource(cgd.evaluate)
    assert "pool_depth" not in inspect.getsource(cgd.evaluate)


def test_render_names_the_mode_and_pool_depth():
    baseline, candidate = _pool_arm("baseline"), _pool_arm("candidate")
    receipt = cgd.evaluate_pool(baseline, candidate, set(),
                                names=("authoritative", "foundation"), depth=50)
    text = "\n".join(cgd.render(receipt))
    assert text.startswith("CONTENT GUARD (pool, pool_depth=50): PASS")
