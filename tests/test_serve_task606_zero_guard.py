"""Task 606 -- the sparse-leg zero guard, tested on real recorded pools.

The defect this guards is not hypothetical and not new: task 336 found that a
*filtered* sparse query against a plain (unindexed) segment returns
filter-matching points that do not contain the term, scored 0.0, and that a
90-point upsert on aw_b1 broke 4 of 663 gate golds through it. The per-token
sub-legs and the stage-2 coverage oracle were guarded then; the MAIN legs, whose
min-max sets the scale of `rel.bm25`, were not.

The fixture is not invented. `tests/testdata/task606_bm25_legs.json` holds
five BM25 stage-1 pools recorded off the twin at production's binding, three of
them SHORT (under the candidate window) -- the population a plain segment can
actually pad, measured at 38 of 1,394 legs on the 1,418-query battery.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from cytir.serve import retrieval

FIXTURE = (pathlib.Path(__file__).resolve().parents[1]
           / "tests/testdata/task606_bm25_legs.json")


def legs():
    doc = json.loads(FIXTURE.read_text())
    assert doc["schema"] == "aw.task606.recorded-bm25-legs.v1"
    return doc["legs"]


def pad_with_plain_segment(scores: dict, n: int) -> dict:
    """What a plain segment adds: filter-matching ids the term is absent from."""
    padded = dict(scores)
    for i in range(n):
        padded[f"__plain_segment_{i}__"] = 0.0
    return padded


@pytest.mark.parametrize("leg", legs(), ids=lambda l: f"{l['set']}/{l['qid']}")
def test_the_guard_is_inert_on_a_pool_with_no_non_positive_score(leg):
    """Every recorded pool is all-positive, so the guard must change nothing."""
    scores = {k: float(v) for k, v in leg["scores"].items()}
    assert min(scores.values()) > 0.0, "fixture drifted: a recorded score is <= 0"
    assert retrieval._minmax(scores, drop_nonpositive=True) \
        == retrieval._minmax(scores)


@pytest.mark.parametrize("leg", legs(), ids=lambda l: f"{l['set']}/{l['qid']}")
def test_a_padded_pool_rescales_the_whole_leg_without_the_guard(leg):
    """The defect, priced on the real pool: `lo` collapses and every row moves."""
    scores = {k: float(v) for k, v in leg["scores"].items()}
    clean = retrieval._minmax(scores)
    padded = retrieval._minmax(pad_with_plain_segment(scores, 20))
    # the pool's own maximum is pinned at 1.0 by both normalisations; every
    # OTHER row is re-scaled, which is what makes this a leg-wide event rather
    # than a near-tie.
    hi = max(scores.values())
    tops = {k for k, v in scores.items() if v == hi}   # ties are real in a pool
    moved = [k for k in scores if abs(padded[k] - clean[k]) > 1e-9]
    assert set(moved) == set(scores) - tops
    worst = max(abs(padded[k] - clean[k]) for k in scores)
    # This is a re-scale, not a tie-break. Measured on these five recorded
    # pools: 0.041 / 0.272 / 0.552 / 0.692 / 0.738 in bm25_norm, and bm25 has
    # weight 1.0 in candidate533 -- so on a tight pool (the two `authoritative`
    # legs, lo 33-42 against hi 45-60) it moves `rel_core` by ~0.7, against a
    # MEDIAN fused margin of 0.31 over the whole core and 0.082 over the losses
    # task 604 attributed to a content pack.
    assert worst > 0.01


@pytest.mark.parametrize("leg", legs(), ids=lambda l: f"{l['set']}/{l['qid']}")
def test_the_guard_makes_the_pool_immune_to_any_amount_of_padding(leg):
    scores = {k: float(v) for k, v in leg["scores"].items()}
    clean = retrieval._minmax(scores, drop_nonpositive=True)
    for n in (1, 20, 500):
        padded = retrieval._minmax(pad_with_plain_segment(scores, n),
                                   drop_nonpositive=True)
        assert {k: padded[k] for k in scores} == clean
        # a non-match is not admitted: absent from the norm map, which every
        # read site resolves through `norm.get(pid, 0.0)`.
        assert not [k for k in padded if k.startswith("__plain_segment_")]


def test_a_negative_score_is_dropped_too_not_merely_a_zero():
    scores = {"a": 10.0, "b": 5.0, "c": -0.5, "d": 0.0}
    assert retrieval._minmax(scores, drop_nonpositive=True) \
        == retrieval._minmax({"a": 10.0, "b": 5.0})


def test_an_all_non_positive_pool_is_empty_not_all_ones():
    """Without the guard `hi <= lo` returns 1.0 for every row -- a pool made
    entirely of non-matches would score as a perfect lexical match."""
    scores = {"a": 0.0, "b": 0.0}
    assert retrieval._minmax(scores) == {"a": 1.0, "b": 1.0}
    assert retrieval._minmax(scores, drop_nonpositive=True) == {}


def test_the_guard_is_off_by_default_and_named_in_the_effective_config():
    assert retrieval._BM25_ZERO_GUARD is False
    assert retrieval.effective_config()["bm25_zero_guard"] is False


def test_the_guard_never_reaches_the_dense_leg():
    """A cosine of 0.0 or below is a real score, not a non-match."""
    assert "dense" not in retrieval._SPARSE_SCORE_LEGS
    assert retrieval._SPARSE_SCORE_LEGS == {
        "bm25", "target_bm25", "translated_bm25", "lexical"}
