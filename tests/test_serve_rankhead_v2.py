#!/usr/bin/env python3
"""Pure contract checks for Task 377 candidate evidence features.

No Qdrant or model is needed: the test guards the train/serve feature schema
and, especially, that presentation fields cannot influence the new evidence.
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cytir.serve import rankhead_v2  # noqa: E402


def _features(payload):
    return rankhead_v2.doc_features(
        payload, authority=0.7, country="CH", eu_supra=False,
        scope="CH", scope_conf=1.0, title_centrality=0.25, rel_ratio=0.6,
        dense_norm=0.8, dense_raw=0.73, dense_rank=0.5,
        bm25_raw=0.9, bm25_norm=0.7, bm25_rank=1.0,
        sparse_coverage=0.5, coverage_available=True,
        today=dt.date(2026, 8, 2))


def test_retrieval_feature_contract_is_complete_and_bounded():
    p = {"source_type": "official", "trust_tier": 0.95,
         "published": "2026-08-01", "last_seen": "2026-08-01"}
    f = _features(p)
    assert len(f) == rankhead_v2.N_FEATURES == len(rankhead_v2.FEATURES)
    got = dict(zip(rankhead_v2.FEATURES, f))
    assert got["dense_raw"] == 0.73
    assert got["dense_rank"] == 0.5
    assert got["bm25_raw"] == 0.9
    assert got["bm25_norm"] == 0.7
    assert got["bm25_rank"] == 1.0
    assert got["leg_agreement"] == 0.7
    assert got["sparse_coverage"] == 0.5
    assert got["sparse_coverage_available"] == 1.0
    assert all(0.0 <= v <= 1.0 for name, v in got.items()
               if name != "scope_match")


def test_snippet_is_presentation_only_for_candidate_features():
    base = {"source_type": "official", "trust_tier": 0.95}
    verbose = {**base, "title": "different display title",
               "snippet": "Marc Maret exact words that must not enter the head"}
    # Title centrality is a separately supplied, already-established signal;
    # the payload title/snippet themselves must not change this feature contract.
    assert _features(base) == _features(verbose)


def test_coverage_unavailable_is_explicit_not_a_false_nonmatch():
    f = rankhead_v2.retrieval_features(
        dense_raw=2.0, dense_norm=-1.0, dense_rank=-1.0,
        bm25_raw=2.0, bm25_norm=0.4, bm25_rank=0.0,
        sparse_coverage=1.0, coverage_available=False)
    got = dict(zip(rankhead_v2.FEATURES[-8:], f))
    assert got == {"dense_raw": 1.0, "dense_rank": 0.0, "bm25_raw": 1.0,
                   "bm25_norm": 0.4, "bm25_rank": 0.0,
                   "leg_agreement": 0.0, "sparse_coverage": 1.0,
                   "sparse_coverage_available": 0.0}
