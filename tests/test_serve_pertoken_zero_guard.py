#!/usr/bin/env python3
"""Task 336 — regression test for the per-token BM25 sub-leg ZERO-GUARD.

The fault (found in task 335 §3, reproduced by task 335 `repro_zde335.py`
against the live aw_b1 collection): `retrieval.search()` normalized each rare-token
BM25 sub-leg by `min(sub.values())` with no zero guard. A **filtered** sparse query
against a **plain (unindexed) segment** — which is what any collection carries right
after a small live write, i.e. after every ADR-066 delta-promote — returns
filter-matching points that do NOT contain the token, scored `0.0`. One such point
made `mn == 0` and every query with a rare token raised ZeroDivisionError. A
90-point upsert on aw_b1 broke 4 of 663 gate golds.

This test does NOT need a qdrant: it drives the real `retrieval.search()` with a
stub client that reproduces exactly that response shape.

Three properties, which are the whole contract of the fix:

  1. CRASH FIX      — a 0.0-score point in a rare-token sub-leg no longer raises.
  2. INERT          — with no zero-score points (an *indexed* collection), results
                      and scores are bit-for-bit what the pre-guard code produced.
                      Asserted against `_pre_guard_norm`, a verbatim copy of the
                      old two lines, over randomized sub-legs.
  3. LEAK-EQUIVALENT — the leaked 0.0-score point must not become a candidate:
                      results with the leak == results without it.

Run: python3 tests/test_serve_pertoken_zero_guard.py   (or: pytest tests/test_serve_pertoken_zero_guard.py)
"""
import os
import random
import sys

# Flags must be set BEFORE importing retrieval — they are read at import time.
# Keep the surface minimal: two-stage + per-token sub-legs ON (the code path
# under test), every learned/side feature OFF so the assertions are about the
# normalizer and nothing else.
os.environ.update({
    "AW_TWO_STAGE": "1", "AW_BM25_PER_TOKEN": "1",
    "AW_RANKHEAD": "0", "AW_QHEAD": "0", "AW_AUTHLEG": "0", "AW_AUTHORITY": "0",
    "AW_COMMERCIAL": "0", "AW_DEDUP_V2": "0", "AW_RELFLOOR": "0",
    "AW_HUBFLOOR": "0", "AW_TRUSTFLOOR": "0", "AW_RERANK": "0",
    "AW_EMBEDDER": "arctic", "AW_COLLECTION": "test_zero_guard",
})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cytir.serve import models, retrieval, schema  # noqa: E402

QUERY = "kuendigungsfrist arbeitsvertrag schweiz"
DENSE = [0.1] * 128


class P:
    """Minimal qdrant ScoredPoint / Record stand-in."""

    def __init__(self, pid, score=0.0, payload=None):
        self.id, self.score, self.payload = pid, score, payload


class Resp:
    def __init__(self, points):
        self.points = points


def _payload(pid):
    return {"url": f"https://example.org/{pid}", "domain": "example.org",
            "title": f"doc {pid}", "snippet": "s", "lang": "de",
            "source_type": "content", "trust_tier": 0.5, "last_seen": None}


class StubClient:
    """Serves the three calls `search()` makes. `sub_scores` is the per-token
    sub-leg the guard has to survive: {point_id: bm25_score}."""

    def __init__(self, main_ids, sub_scores):
        self.main_ids, self.sub_scores = main_ids, sub_scores

    def query_points(self, collection_name, query, using, limit,
                     query_filter=None, search_params=None, with_payload=True):
        # dense + main-bm25 legs: a plain descending pool, identical in every arm.
        pts = [P(i, score=1.0 - 0.01 * n) for n, i in enumerate(self.main_ids)]
        if with_payload is True:
            for p in pts:
                p.payload = _payload(p.id)
        return Resp(pts)

    def query_batch_points(self, collection_name, requests):
        # One rare-token sub-leg (short window => `len(sub) < _PER_TOKEN_LIMIT`,
        # so it takes the rare-token branch), then empty windows for the rest.
        out = [Resp([P(i, score=s) for i, s in self.sub_scores.items()])]
        out += [Resp([]) for _ in requests[1:]]
        return out

    def retrieve(self, collection_name, ids, with_payload=True):
        return [P(i, payload=_payload(i)) for i in ids]


def run_search(main_ids, sub_scores):
    import pytest
    mp = pytest.MonkeyPatch()      # restored: a bare assignment leaked `qdrant_io.client` into later tests
    mp.setattr(retrieval.qdrant_io, "client", lambda: StubClient(main_ids, sub_scores))
    models.embed_query = lambda q: (DENSE, None)
    models.last_full_query_vec = lambda: None
    try:
        return retrieval.search(QUERY, k=8, lang="de")
    finally:
        mp.undo()


def _pre_guard_norm(sub, k1):
    """VERBATIM copy of the two pre-336 lines — the inertness oracle."""
    mn = min(sub.values())
    return {pid: (s / mn) / (s / mn + k1) for pid, s in sub.items()}


def _guarded_norm(sub, k1):
    """The shipped guard, isolated, for the same-value comparison."""
    pos = [s for s in sub.values() if s > 0.0]
    if not pos:
        return None
    mn = min(pos)
    return {pid: (s / mn) / (s / mn + k1) for pid, s in sub.items() if s > 0.0}


def test_1_crash_fix():
    """A 0.0-score point in the sub-leg used to raise ZeroDivisionError."""
    sub = {1: 3.0, 2: 2.0, 3: 1.5, 99: 0.0}   # 99 = the plain-segment leak
    # the test is only meaningful if this input really did break the old code:
    try:
        _pre_guard_norm(sub, retrieval._PER_TOKEN_K1)
        raise AssertionError("pre-336 normalizer did NOT raise — test is vacuous")
    except ZeroDivisionError:
        pass
    hits = run_search(list(range(1, 21)) + [99], sub)
    assert hits, "search returned no hits"
    print(f"  [1] crash fix: pre-336 code raises ZeroDivisionError on this "
          f"input, guarded code returns {len(hits)} hits  OK")


def test_2_inert_on_indexed_collection():
    """No zero scores (an indexed collection) => identical to the old code."""
    k1 = retrieval._PER_TOKEN_K1
    rnd = random.Random(336)
    for trial in range(200):
        sub = {i: rnd.uniform(1e-6, 50.0) for i in range(1, rnd.randint(2, 12))}
        old, new = _pre_guard_norm(sub, k1), _guarded_norm(sub, k1)
        assert old == new, f"guard is NOT inert on trial {trial}: {old} != {new}"
    print("  [2] inert: 200 randomized all-positive sub-legs, "
          "bit-for-bit identical to the pre-336 normalizer  OK")


def test_3_leak_is_not_a_candidate():
    """The leaked 0.0 point must not enter bnorm/ids: results with the leak
    must equal results without it (what the indexed collection returns)."""
    ids = list(range(1, 21))
    sub = {1: 3.0, 2: 2.0, 3: 1.5}
    leak = dict(sub)
    leak[99] = 0.0
    # identical MAIN legs in both arms — the leak is real only in the *sub*-leg
    # (a filter-match that does not contain the token), which is the whole point.
    clean = run_search(ids, dict(sub))
    leaked = run_search(ids, leak)
    assert [h["url"] for h in clean] == [h["url"] for h in leaked], \
        "leaked zero-score point changed the result list"
    assert [round(h.get("score", 0.0), 9) for h in clean] == \
           [round(h.get("score", 0.0), 9) for h in leaked], \
        "leaked zero-score point changed the scores"
    print(f"  [3] leak-equivalent: {len(clean)} hits identical with/without "
          "the 0.0-score point  OK")


def test_4_all_zero_window_is_df0():
    """A window where EVERY point scored 0.0 means the token is absent — the
    same state an indexed collection reports as an empty window (df=0). Must
    not raise, and must not be treated as a rare-token rescue."""
    ids = list(range(1, 21))
    assert _guarded_norm({7: 0.0, 8: 0.0}, retrieval._PER_TOKEN_K1) is None
    hits = run_search(ids, {7: 0.0, 8: 0.0})
    assert hits, "all-zero sub-leg window broke the search"
    print(f"  [4] all-zero window: classified df=0, {len(hits)} hits  OK")


if __name__ == "__main__":
    print(f"per-token zero-guard (retrieval.py, task 336) — "
          f"_PER_TOKEN_K1={retrieval._PER_TOKEN_K1} "
          f"_PER_TOKEN_LIMIT={retrieval._PER_TOKEN_LIMIT} "
          f"bm25 tokens={len(schema.bm25_tokens(QUERY, 'de'))}")
    fails = 0
    for fn in (test_1_crash_fix, test_2_inert_on_indexed_collection,
               test_3_leak_is_not_a_candidate, test_4_all_zero_window_is_df0):
        try:
            fn()
        except Exception as e:
            fails += 1
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print("PASS" if not fails else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)
