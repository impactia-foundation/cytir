#!/usr/bin/env python3
"""Guards for the dense-only and sealed-IDF BM25 gate arms (task 629 P4).

The real-record half runs off `tests/testdata/task626_content_guard_pool_fixture.json`
— the actual 09-15 content-guard arms, with real per-leg point-id lists for real
golds. That fixture **predates** `run_corpus_arm.trace()`'s `urls` key, which makes
it the perfect witness for the one refusal that matters here: an arm with full leg
id lists and no url map must be reported UNSCOREABLE, never scored as a leg that
lost every gold.

The arithmetic half (the sealed-IDF score, the ordering, the coverage) is tested on
small explicit inputs, because those functions are pure and a fixture would only
obscure what the numbers are supposed to be.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent

from cytir.gates import leg_arms as L  # noqa: E402

FIXTURE = HERE / "testdata" / "task626_content_guard_pool_fixture.json"

TABLE = {"idf": {"art": 3.0, "336": 5.0, "co": 0.5, "der": 0.1},
         "points_count": 33_736_546}


def write_table(tmp_path, **kw) -> str:
    doc = {"idf": TABLE["idf"], "points_count": TABLE["points_count"],
           "measured_at": "2026-09-22T00:00:00Z", **kw}
    path = tmp_path / "idf.json"
    path.write_text(json.dumps(doc))
    return str(path)


# ------------------------------------------------------------- the sealed table
def test_a_table_with_no_points_count_is_refused(tmp_path):
    """A sealed table is only meaningful against the collection it was measured
    on; without that number it cannot be placed on the history at all."""
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"idf": {"art": 1.0}}))
    with pytest.raises(L.LegArmError, match="points_count"):
        L.load_idf_table(path)


def test_a_table_sealed_from_an_unsettled_collection_is_refused(tmp_path):
    """Task 627 §6-R3 cost 3: sealing from a drifted layout bakes its stale
    postings in permanently AND ships them to every self-hoster. Worse than having
    no table, so it refuses rather than warns."""
    path = write_table(tmp_path, settled=False)
    with pytest.raises(L.LegArmError, match="not stamped"):
        L.load_idf_table(path)


def test_a_table_that_merely_OMITS_settled_is_also_refused(tmp_path):
    """The hole worth closing: `measure_bm25_idf.py` writes no `settled` key at
    all, so `is not False` would have accepted every table it produces and caught
    only one honest enough to admit the problem. The check is `is True`."""
    path = write_table(tmp_path)          # no `settled` key
    with pytest.raises(L.LegArmError, match="not stamped"):
        L.load_idf_table(path)


def test_a_table_sealed_from_a_settled_collection_loads(tmp_path):
    table = L.load_idf_table(write_table(tmp_path, settled=True))
    assert table["points_count"] == 33_736_546
    assert table["vocabulary"] == len(TABLE["idf"])
    assert table["settled"] is True


def test_an_empty_table_is_refused(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"idf": {}, "points_count": 1}))
    with pytest.raises(L.LegArmError, match="not a usable table"):
        L.load_idf_table(path)


# --------------------------------------------------------------- the arithmetic
def test_the_sealed_score_is_the_adr109_formula():
    """`sum_t idf(t) * q(t) * d(t)`, with the sealed idf replacing Qdrant's."""
    score = L.sealed_bm25_score({"art": 1.0, "336": 1.0},
                                {"art": 0.4, "336": 0.9}, TABLE)
    assert score == pytest.approx(3.0 * 1.0 * 0.4 + 5.0 * 1.0 * 0.9)


def test_an_unmeasured_token_contributes_nothing_rather_than_dominating():
    """Task 627 §6-R3 cost 2: the shipped loader treats an absent token as HIGH
    idf, which is right for a suppression threshold and wrong for a weight. A
    missing token must not silently become the most important term in the query."""
    assert L.ABSENT_TOKEN_IDF == 0.0
    with_absent = L.sealed_bm25_score({"art": 1.0, "zzz": 1.0},
                                      {"art": 0.4, "zzz": 0.9}, TABLE)
    without = L.sealed_bm25_score({"art": 1.0}, {"art": 0.4}, TABLE)
    assert with_absent == pytest.approx(without)


def test_a_token_the_document_does_not_carry_contributes_nothing():
    assert L.sealed_bm25_score({"art": 1.0}, {"co": 9.9}, TABLE) == 0.0


def test_the_order_is_total_and_reproducible():
    """Ties break on the point id. An arbitrary tie order would reintroduce, as a
    coin flip, the irreproducibility this arm exists to remove."""
    pool = {"zzz": {"art": 0.5}, "aaa": {"art": 0.5}, "mid": {"art": 0.9}}
    assert L.sealed_bm25_order({"art": 1.0}, pool, TABLE) == ["mid", "aaa", "zzz"]
    # and the same pool in a different insertion order gives the same answer
    reordered = {k: pool[k] for k in ("mid", "zzz", "aaa")}
    assert (L.sealed_bm25_order({"art": 1.0}, reordered, TABLE)
            == L.sealed_bm25_order({"art": 1.0}, pool, TABLE))


def test_coverage_reports_what_the_table_could_not_weigh():
    cov = L.coverage(["art", "336", "zzz"], TABLE)
    assert cov == {"tokens": 3, "present": 2, "absent": ["zzz"],
                   "share": pytest.approx(2 / 3)}


def test_implied_df_inverts_the_idf_for_reporting():
    """Reporting only, never scoring — but it has to actually invert."""
    points = 1_000_000
    df = L.implied_df(5.0, points)
    assert 0 < df < points


# --------------------------------------------------------------- the gold match
def test_rank_is_the_legs_own_position_not_the_fused_one():
    ids = ["p1", "p2", "p3"]
    urls = {"p1": "https://other.test/x", "p2": "https://fedlex.admin.ch/eli/y"}
    assert L.rank_in_order(ids, urls, ["https://fedlex.admin.ch/eli/y"]) == 2


def test_an_unresolvable_id_still_occupies_its_rank():
    """Closing the gap would report a better rank than the leg produced."""
    ids = ["unknown", "p2"]
    urls = {"p2": "https://fedlex.admin.ch/eli/y"}
    assert L.rank_in_order(ids, urls, ["https://fedlex.admin.ch/eli/y"]) == 2


def test_a_domain_gold_matches_on_the_domain():
    urls = {"p1": "https://www.fedlex.admin.ch/eli/cc/1/2"}
    assert L.rank_in_order(["p1"], urls, [], ["fedlex.admin.ch"]) == 1


def test_a_gold_the_leg_never_returned_is_none():
    assert L.rank_in_order(["p1"], {"p1": "https://x.test/a"},
                           ["https://y.test/b"]) is None


# ----------------------------------------- the real fixture, and THE refusal
def test_the_fixture_is_the_real_arms_with_real_leg_ids():
    doc = json.loads(FIXTURE.read_text())
    assert doc["schema"] == "aw.task626-content-guard-refine-fixture.v1"
    row = doc["sets"]["authoritative"]["per_q"]
    first = row[sorted(row)[0]]
    for side in ("baseline", "candidate"):
        trace = first[side]["trace"]
        assert trace["dense"] and trace["bm25"], "real leg id lists"
        # the witness property: this fixture predates TRACE_EXTRA_KEYS
        assert "urls" not in trace


def test_an_arm_without_the_url_map_is_UNSCOREABLE_not_a_total_loss():
    """The refusal this module exists for.

    A pre-`urls` arm has full leg id lists that resolve to nothing. Scoring it
    would return every rank as None — a -639 h@1 produced entirely by a missing
    input, which is the failure memory `verify-liveness-before-trusting-misses`
    names: never record an error as an empty.
    """
    doc = json.loads(FIXTURE.read_text())
    per_q = {qid: row["baseline"]
             for qid, row in doc["sets"]["authoritative"]["per_q"].items()}
    golds = {qid: {"urls": ["https://never.matches.test/x"]} for qid in per_q}
    verdict = L.score_leg(per_q, "dense", golds)
    assert verdict["status"] == "UNSCOREABLE"
    assert verdict["scored"] == 0
    assert verdict["h1"] == 0 and verdict["h10"] == 0
    assert sorted(verdict["queries_unresolvable"]) == sorted(per_q)
    assert "re-cut" in verdict["why"]


def test_a_delta_against_an_unscoreable_arm_is_refused():
    """Otherwise the number is about the instrument, not about the pack."""
    doc = json.loads(FIXTURE.read_text())
    per_q = {qid: row["baseline"]
             for qid, row in doc["sets"]["authoritative"]["per_q"].items()}
    golds = {qid: {"urls": ["https://never.matches.test/x"]} for qid in per_q}
    unscoreable = L.score_leg(per_q, "dense", golds)
    good = {"leg": "dense", "h1": 1, "h10": 1, "scored": 1, "ranks": {"q": 1},
            "status": "OK"}
    with pytest.raises(L.LegArmError, match="UNSCOREABLE"):
        L.compare_leg(unscoreable, good)
    with pytest.raises(L.LegArmError, match="UNSCOREABLE"):
        L.compare_leg(good, unscoreable)


def test_the_fixture_scores_once_its_own_urls_are_supplied():
    """Same real leg ids, with a url map built from the fixture's OWN recorded
    top-10 urls — so the ids are real and the mapping is declared, not invented.

    This is the positive control for the refusal above: the module scores real leg
    orders perfectly well, and refuses only when it cannot resolve them.
    """
    doc = json.loads(FIXTURE.read_text())
    raw = doc["sets"]["authoritative"]["per_q"]
    per_q, golds = {}, {}
    for qid, row in raw.items():
        side = row["baseline"]
        ids = side["trace"]["dense"]
        fused_urls = side.get("urls") or []
        # pair the leg's leading ids with the arm's own recorded urls, in order
        urls = {str(pid): url for pid, url in zip(ids, fused_urls)}
        if not urls:
            continue
        per_q[qid] = {**side, "trace": {**side["trace"], "urls": urls}}
        golds[qid] = {"urls": [fused_urls[0]]}
    assert per_q, "fixture must carry recorded urls"
    verdict = L.score_leg(per_q, "dense", golds)
    assert verdict["status"] == "OK"
    assert verdict["scored"] == len(per_q)
    # the gold was the leg's own first url, so it ranks 1 wherever it resolved
    assert verdict["h1"] == len(per_q)


# ------------------------------------------------------------------ comparison
def test_compare_reports_raw_losses_and_gains_uncorrected():
    """ADR-043 A3 rule 2 insists the raw loss list is printed in full; the null-arm
    correction is the caller's, over the same shape. A module that pre-corrected
    them would hide the list the ADR requires."""
    base = {"leg": "dense", "h1": 2, "h10": 2, "scored": 3, "status": "OK",
            "ranks": {"a": 1, "b": 1, "c": None}}
    cand = {"leg": "dense", "h1": 1, "h10": 2, "scored": 3, "status": "OK",
            "ranks": {"a": 1, "b": 4, "c": 2}}
    out = L.compare_leg(base, cand)
    assert out["h1_delta"] == -1
    assert [r["qid"] for r in out["losses"]] == ["b"]
    assert [r["qid"] for r in out["gains"]] == ["c"]
    assert {r["qid"] for r in out["moved"]} == {"b", "c"}


def test_comparing_two_different_legs_is_refused():
    a = {"leg": "dense", "h1": 0, "h10": 0, "ranks": {}, "status": "OK"}
    b = {"leg": "bm25", "h1": 0, "h10": 0, "ranks": {}, "status": "OK"}
    with pytest.raises(L.LegArmError, match="different legs"):
        L.compare_leg(a, b)


def test_a_leg_outside_the_scored_set_is_refused():
    with pytest.raises(L.LegArmError, match="not a scored leg"):
        L.score_leg({}, "country_dense", {})


# ---------------------------------------------------------------------------
# the SEALED leg -- A8 rule 4's third arm (task 629 P4)
# ---------------------------------------------------------------------------
SEALED_TABLE = {"idf": {"7": 5.0, "9": 0.5}, "points_count": 1_000,
                "settled": True, "vocabulary": 2, "source": "fixture"}


def _arm_with_sparse():
    """One query, two candidates, and a live order the sealed table disagrees with.

    Qdrant's live bm25 put `b` first. Under the sealed table token 7 is worth 10x
    token 9, and `a` is the candidate that carries token 7 -- so the sealed order
    is (a, b). The gold is `a`, so the sealed leg ranks it 1 and the live leg 2.
    That inversion is the whole point of the arm: same stored vectors, same query,
    different idf, different order.
    """
    return {"q1": {"rank": 2, "trace": {
        "dense": ["a", "b"], "bm25": ["b", "a"],
        "urls": {"a": "https://example.org/a", "b": "https://example.org/b"},
        "bm25_query": {"7": 1.0, "9": 1.0},
        "sparse": {"a": {"7": 1.0}, "b": {"9": 1.0}},
    }}}


GOLDS = {"q1": {"urls": ["https://example.org/a"], "domains": []}}


def test_the_sealed_leg_reorders_the_pool_under_the_table():
    arm = _arm_with_sparse()
    sealed = L.score_sealed_leg(arm, GOLDS, SEALED_TABLE)
    assert sealed["status"] == "OK"
    assert sealed["leg"] == L.SEALED_LEG
    assert sealed["ranks"]["q1"] == 1, "token 7 outweighs token 9 in the table"
    assert sealed["h1"] == 1
    # and the LIVE leg, read from the same arm, disagrees -- which is the
    # measurement that makes the sealed arm worth cutting at all
    live = L.score_leg(arm, "bm25", GOLDS)
    assert live["ranks"]["q1"] == 2
    assert live["h1"] == 0


def test_the_sealed_leg_records_which_table_it_used():
    sealed = L.score_sealed_leg(_arm_with_sparse(), GOLDS, SEALED_TABLE)
    assert sealed["idf_table"] == "fixture"
    assert sealed["idf_points_count"] == 1_000


def test_an_arm_without_the_sparse_capture_is_unscoreable_not_a_total_loss():
    """The -639 h@1 failure mode, in the sealed leg's own form."""
    arm = {"q1": {"rank": 2, "trace": {
        "dense": ["a"], "bm25": ["b", "a"],
        "urls": {"a": "https://example.org/a"}}}}
    sealed = L.score_sealed_leg(arm, GOLDS, SEALED_TABLE)
    assert sealed["status"] == "UNSCOREABLE"
    assert sealed["h1"] == 0 and sealed["scored"] == 0
    assert "capture-sparse" in sealed["why"]
    # and a delta against it must refuse rather than report -1
    with pytest.raises(L.LegArmError, match="UNSCOREABLE"):
        L.compare_leg(L.score_sealed_leg(_arm_with_sparse(), GOLDS,
                                            SEALED_TABLE), sealed)


def test_a_live_leg_with_candidates_but_an_empty_capture_is_unresolvable():
    """An empty pool under a non-empty live leg is a broken capture, not a miss."""
    arm = {"q1": {"rank": 2, "trace": {
        "dense": ["a"], "bm25": ["b", "a"],
        "urls": {"a": "https://example.org/a"},
        "bm25_query": {"7": 1.0}, "sparse": {}}}}
    sealed = L.score_sealed_leg(arm, GOLDS, SEALED_TABLE)
    assert sealed["status"] == "UNSCOREABLE"
    assert sealed["queries_unresolvable"] == ["q1"]


def test_the_sealed_leg_shape_matches_score_leg_so_compare_leg_works():
    """One shape, so the null-arm correction applies over one shape."""
    sealed = L.score_sealed_leg(_arm_with_sparse(), GOLDS, SEALED_TABLE)
    live = L.score_leg(_arm_with_sparse(), "bm25", GOLDS)
    assert set(sealed) >= set(live), sorted(set(live) - set(sealed))
