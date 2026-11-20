"""`cytir.gates.flip_cells` -- the two cell rules of Ruling 31 (Tim 2026-10-12), ADR-043 amendment A10.

The numbers are task 863's component arms (E2 = German compound parts, against the E1 control, with a
rebuild null and an id null), copied in as literals: the rule is tested on the readings that motivated
it. Gold units throughout (counts; MRR as the reciprocal-rank sum).
"""
from __future__ import annotations

import pytest

from cytir.gates import flip_cells as FC


def _deltas(**classes):
    """{class: {metric: d}} from class=(d_hit1, d_hitk, d_mrr_sum)."""
    return {c: dict(zip(FC.METRICS, v)) for c, v in classes.items()}


# task 863, held-out sparse-only: the two null pairs, every class that moved
REBUILD_HO_SPARSE = _deltas(authoritative=(0, 0, 0.023), dom_reference=(1, 0, 0.5),
                            freshness=(0, -1, -0.067), jurisdiction=(0, 0, -0.014))
ID_HO_SPARSE = _deltas(authoritative=(0, 0, -0.033), dom_reference=(1, 0, 0.5),
                       freshness=(0, 0, -0.05), jurisdiction=(0, 0, -0.014))
# task 863, held-out dense-only: the id null moved en14 rank 1 -> 2 in dom_entity
REBUILD_HO_DENSE = _deltas(authoritative=(0, 0, 0.011), entity_card_golds=(1, 0, 0.5))
ID_HO_DENSE = _deltas(authoritative=(0, 0, 0.011), dom_entity=(-1, -1, -1.0),
                      freshness=(0, 0, 0.166), rel_vs_trust=(0, 1, 0.1))


def _read(d_arm, klass, metric, nulls):
    cells = [n.get(klass, {}).get(metric, 0.0) for n in nulls]
    return FC.read_component_cell(d_arm, cells, FC.spread(nulls, metric))


# ------------------------------------------------------------------------------------------------ #
# rule 1: component cells against the null pairs
# ------------------------------------------------------------------------------------------------ #

def test_the_spread_is_the_largest_move_either_null_shows_on_the_leg():
    nulls = [REBUILD_HO_SPARSE, ID_HO_SPARSE]
    assert FC.spread(nulls, "hit1") == 1
    assert FC.spread(nulls, "hitk") == 1
    assert FC.spread(nulls, "mrr") == 0.5
    assert FC.spread([REBUILD_HO_DENSE, ID_HO_DENSE], "mrr") == 1.0


def test_a_component_cell_worse_than_both_nulls_by_more_than_their_spread_is_flagged():
    """E2 held-out sparse-only authoritative: -3 h@1 (de-law-001 BGB 622 out of the top 10, court
    decisions above the statute); both nulls 0 in that class, spread 1 -> bar -1."""
    r = _read(-3, "authoritative", "hit1", [REBUILD_HO_SPARSE, ID_HO_SPARSE])
    assert r["flagged"] is True
    assert r["bar"] == -1
    r = _read(-2.293, "authoritative", "mrr", [REBUILD_HO_SPARSE, ID_HO_SPARSE])
    assert r["flagged"] is True and r["bar"] == pytest.approx(-0.533)


def test_a_component_cell_inside_the_null_spread_is_not_flagged():
    """E2 held-out sparse-only jurisdiction: -1 h@1, the size of the nulls' own one-gold moves."""
    r = _read(-1, "jurisdiction", "hit1", [REBUILD_HO_SPARSE, ID_HO_SPARSE])
    assert r["flagged"] is False
    # E3's one dense-leg move (bb-028 1 -> 2) is inside: the id null itself moved en14 1 -> 2
    r = _read(-1, "authoritative", "hit1", [REBUILD_HO_DENSE, ID_HO_DENSE])
    assert r["flagged"] is False
    r = _read(-0.5, "authoritative", "mrr", [REBUILD_HO_DENSE, ID_HO_DENSE])
    assert r["flagged"] is False


def test_the_same_one_gold_loss_is_flagged_on_mrr_when_the_nulls_never_moved_that_much():
    """E2 jurisdiction MRR -1.00 (a gold from rank 1 out): the sparse nulls' largest MRR move is
    0.5 (one gold 2 -> 1), so the MRR cell is flagged while the h@1 cell is inside."""
    r = _read(-1.0, "jurisdiction", "mrr", [REBUILD_HO_SPARSE, ID_HO_SPARSE])
    assert r["flagged"] is True and r["bar"] == pytest.approx(-0.514)


def test_a_null_that_gained_does_not_raise_the_bar():
    """One-sided, as ADR-043 A3 rule 4: E3 held-out sparse-only dom_reference MRR -0.05 against
    nulls that both GAINED +0.5 is noise, not a loss beyond the nulls."""
    r = _read(-0.05, "dom_reference", "mrr", [REBUILD_HO_SPARSE, ID_HO_SPARSE])
    assert r["bar"] == -0.5
    assert r["flagged"] is False


def test_a_cell_with_fewer_than_two_null_readings_is_flagged_with_the_reason():
    r = FC.read_component_cell(-1, [0.0], 1.0)
    assert r["flagged"] is True and "1 usable null reading" in r["reason"]
    r = FC.read_component_cell(-0.1, [], 0.0)
    assert r["flagged"] is True


def test_mrr_in_gold_units_is_the_class_mean_times_its_gold_count():
    a = {"hit1": 10, "hitk": 20, "mrr": 0.5, "n_gold": 30}
    b = {"hit1": 9, "hitk": 20, "mrr": 0.4833, "n_gold": 30}
    assert FC.delta_gold_units(a, b, "hit1") == -1
    assert FC.delta_gold_units(a, b, "mrr") == pytest.approx(-0.501)
    arm_a = {"classes": {"legal": a, "empty": {"hit1": 0, "hitk": 0, "mrr": 0, "n_gold": 0}}}
    arm_b = {"classes": {"legal": b, "empty": {"hit1": 0, "hitk": 0, "mrr": 0, "n_gold": 0}}}
    d = FC.null_pair_deltas(arm_a, arm_b)
    assert set(d) == {"legal"}   # a class with no gold reads nothing


def _contract(**rf):
    return {"regression_free": rf}


COMPONENT = {"legs": ["dense-only", "sparse-only"], "bar": "informational",
             "null_pair_kinds": ["rebuild", "id", "reread"]}


def test_a_contract_without_strict_legs_is_the_contract_before_ruling_31():
    assert FC.component_block(_contract()) is None


def test_a_strict_split_without_a_component_block_is_malformed():
    with pytest.raises(FC.CellRuleError, match="component_arms is missing"):
        FC.component_block(_contract(strict_legs=["full-stack"]))
    with pytest.raises(FC.CellRuleError, match="informational"):
        FC.component_block(_contract(strict_legs=["full-stack"],
                                     component_arms={**COMPONENT, "bar": "net-non-negative"}))
    with pytest.raises(FC.CellRuleError, match="closed list"):
        FC.component_block(_contract(strict_legs=["full-stack"],
                                     component_arms={**COMPONENT, "null_pair_kinds": ["vibes"]}))
    legs, block = FC.component_block(_contract(strict_legs=["full-stack"], component_arms=COMPONENT))
    assert legs == ("full-stack",) and block["bar"] == "informational"


# ------------------------------------------------------------------------------------------------ #
# rule 2: accepted losses on the strict bar
# ------------------------------------------------------------------------------------------------ #

R31 = "Ruling 31 (Tim 2026-10-12)"
DE_LAW_017 = {"gold": "de-law-017", "instrument": "1418", "leg": "full-stack", "class": "authoritative",
              "ruling": R31, "candidate_collection": "aw_e10",
              "shown": {"incumbent_rank": 1, "candidate_rank": 2}, "cite": "the task doc"}


def _accepting(*entries):
    return _contract(strict_legs=["full-stack"], component_arms=COMPONENT,
                     accepted_losses=list(entries))


def _entries(contract, coll="aw_e10"):
    return FC.accepted_entries(contract, instrument="1418", leg="full-stack", klass="authoritative",
                               candidate_collection=coll)


def test_an_accepted_loss_no_longer_fails_the_strict_bar():
    """de-law-017: StGB 263 rank 1 -> 2 behind Swiss StGB Art. 146; nothing else in the class moves."""
    entries = _entries(_accepting(DE_LAW_017))
    inc = {"de-law-017": 1, "ch-law-001": 2, "de-law-001": None}
    cand = {"de-law-017": 2, "ch-law-001": 2, "de-law-001": None}
    for metric, d in (("hit1", -1.0), ("mrr", -0.5)):
        res = FC.residual_after_accepted(d, entries, inc, cand, metric)
        assert res["problems"] == []
        assert res["residual"] == pytest.approx(0.0)
        assert res["applied"][0]["gold"] == "de-law-017"
        assert res["applied"][0]["ruling"] == R31


def test_an_unaccepted_loss_still_fails_the_strict_bar():
    """The accepted gold is removed; a SECOND gold falling in the same class is still a loss."""
    entries = _entries(_accepting(DE_LAW_017))
    inc = {"de-law-017": 1, "de-law-001": 1}
    cand = {"de-law-017": 2, "de-law-001": 3}
    res = FC.residual_after_accepted(-2.0, entries, inc, cand, "hit1")
    assert res["problems"] == []
    assert res["residual"] == -1.0


def test_an_accepted_gold_that_falls_further_than_shown_is_a_fresh_refusal():
    entries = _entries(_accepting(DE_LAW_017))
    res = FC.residual_after_accepted(-1.0, entries, {"de-law-017": 1}, {"de-law-017": 3}, "hit1")
    assert res["applied"] == [] and "rank 2" in res["problems"][0]
    res = FC.residual_after_accepted(-1.0, entries, {"de-law-017": 1}, {"de-law-017": None}, "hitk")
    assert res["applied"] == [] and res["problems"]


def test_an_accepted_loss_cannot_be_checked_without_per_gold_ranks():
    entries = _entries(_accepting(DE_LAW_017))
    res = FC.residual_after_accepted(-1.0, entries, None, {"de-law-017": 2}, "hit1")
    assert res["residual"] == -1.0 and "no per-gold ranks" in res["problems"][0]


def test_an_accepted_loss_is_bound_to_its_candidate_collection_and_its_cell():
    c = _accepting(DE_LAW_017)
    assert _entries(c, "aw_e11") == []
    assert len(_entries(c, "aw_e10")) == 1
    assert FC.accepted_entries(c, instrument="1418", leg="full-stack", klass="legal",
                               candidate_collection="aw_e10") == []


def test_a_malformed_accepted_loss_is_refused_not_skipped():
    for broken, why in ((dict(DE_LAW_017, cite=""), "cite"),
                        (dict(DE_LAW_017, ruling=""), "ruling"),
                        (dict(DE_LAW_017, shown={}), "shown.candidate_rank"),
                        (dict(DE_LAW_017, leg="sparse-only"), "not a strict leg")):
        with pytest.raises(FC.CellRuleError, match=why):
            _entries(_accepting(broken))


def test_a_gold_contribution_counts_only_inside_the_cut():
    assert FC.gold_contribution(1, 2, "hit1") == -1
    assert FC.gold_contribution(1, 2, "hitk") == 0
    assert FC.gold_contribution(9, None, "hitk") == -1
    assert FC.gold_contribution(1, 2, "mrr") == pytest.approx(-0.5)
    assert FC.gold_contribution(None, 4, "mrr") == pytest.approx(0.25)


def test_per_gold_ranks_come_from_either_arm_shape():
    rows = [{"id": "t01", "type": "coding", "rank": 4, "error": None},
            {"id": "t02", "type": "coding", "rank": None, "error": None},
            {"id": "t03", "type": "legal", "rank": 1, "error": "Timeout"}]
    assert FC.per_gold_from_rows(rows) == {"coding": {"t01": 4, "t02": None}}
    sets = {"authoritative": {"per_q": {"de-law-017": {"rank": 1, "urls": []}}},
            "general_flood": {"h1": 0}}
    assert FC.per_gold_from_sets(sets) == {"authoritative": {"de-law-017": 1}}


# ------------------------------------------------------------------------------------------------ #
# rule 2b: an accepted loss whose gold fell OUTSIDE the top k (Ruling 35, task 947)
# ------------------------------------------------------------------------------------------------ #

R35 = "Ruling 35 (Tim 2026-10-15)"
GX46 = {"gold": "gx46", "instrument": "629", "leg": "full-stack", "class": "research", "ruling": R35,
        "candidate_collection": "aw_e10", "cite": "the task doc",
        "shown": {"incumbent_rank": 6, "candidate_rank": None, "candidate_outside_top_k": True}}


def _gx46(entry=GX46):
    return FC.accepted_entries(_accepting(entry), instrument="629", leg="full-stack", klass="research",
                               candidate_collection="aw_e10")


def test_a_null_rank_with_the_outside_top_k_flag_is_accepted():
    assert [e["gold"] for e in _gx46()] == ["gx46"]


def test_a_null_rank_without_the_flag_is_refused():
    for shown in ({"incumbent_rank": 6, "candidate_rank": None}, {"incumbent_rank": 6},
                  {"incumbent_rank": 6, "candidate_rank": None, "candidate_outside_top_k": False}):
        with pytest.raises(FC.CellRuleError, match="shown.candidate_rank"):
            _gx46(dict(GX46, shown=shown))


def test_the_flag_with_an_integer_rank_is_contradictory_and_refused():
    shown = {"incumbent_rank": 6, "candidate_rank": 12, "candidate_outside_top_k": True}
    with pytest.raises(FC.CellRuleError, match="contradictory"):
        _gx46(dict(GX46, shown=shown))


def test_the_gold_that_fell_out_of_the_top_k_is_removed_from_the_class_loss():
    """class h@10 -1 with gx46 (rank 6 -> beyond k): nothing else moves, so the class clears."""
    entries = _gx46()
    for cand in ({"gx46": None, "other": 3}, {"other": 3}):  # null rank, or absent from the arm
        res = FC.residual_after_accepted(-1.0, entries, {"gx46": 6, "other": 3}, cand, "hitk")
        assert res["problems"] == [] and res["residual"] == pytest.approx(0.0)
        assert res["applied"][0]["outside_top_k"] is True and res["applied"][0]["rank_candidate"] is None


def test_a_second_loss_in_the_class_is_still_a_loss_beside_the_outside_top_k_gold():
    res = FC.residual_after_accepted(-2.0, _gx46(), {"gx46": 6, "o": 2}, {"gx46": None, "o": None}, "hitk")
    assert res["problems"] == [] and res["residual"] == -1.0


def test_the_gold_back_inside_the_top_k_is_covered_and_the_residual_is_the_actual_delta():
    entries = _gx46()
    res = FC.residual_after_accepted(0.0, entries, {"gx46": 6}, {"gx46": 4}, "hitk")
    assert res["problems"] == [] and res["residual"] == pytest.approx(0.0)
    res = FC.residual_after_accepted(-1.0, entries, {"gx46": 6, "o": 2}, {"gx46": 4, "o": None}, "hitk")
    assert res["problems"] == [] and res["residual"] == -1.0


def test_an_integer_shown_gold_that_is_now_absent_is_still_refused():
    entries = _entries(_accepting(DE_LAW_017))
    res = FC.residual_after_accepted(-1.0, entries, {"de-law-017": 1}, {}, "hit1")
    assert res["applied"] == [] and "not scored in both arms" in res["problems"][0]
    res = FC.residual_after_accepted(-1.0, entries, {"de-law-017": 1}, {"de-law-017": None}, "hitk")
    assert res["applied"] == [] and "ranks None on the candidate" in res["problems"][0]


def test_an_absent_incumbent_gold_is_still_refused_for_an_outside_top_k_entry():
    res = FC.residual_after_accepted(-1.0, _gx46(), {}, {"gx46": None}, "hitk")
    assert res["applied"] == [] and "not scored in both arms" in res["problems"][0]
