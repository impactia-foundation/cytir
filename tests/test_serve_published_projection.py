#!/usr/bin/env python3
"""Task 625 — `/search` must project the `published` date it already stores.

The news-freshness policy (memory `aw-news-freshness-policy`) captures a
genuine, never-inferred `published` ISO date at ingest time
(the published-people extractor -> the sitemap ingester's
`row["published"]`, passed through by `aw_b1_projection.BASE_PASSTHROUGH`).
A live read of the 2026-09-16 pack on the twin (the build box, port 6457, `aw_e8`)
found the field genuinely populated for the large majority of rows -- the
minority reading `None` are section/hub/dossier pages with no defensible
single publication date (correct, "never a guess" absence, not a bug).

The real gap task 622 found was narrower than it read: `/search` never
projected the field at all, on EITHER of the two result-building paths in
`cytir/serve/retrieval.py` (the identifier/entity-card fast path and the main
fused-ranking path). (Task 622's own "0 of 14,819" reading also used the
wrong payload key -- `published_date`, which does not exist on this family's
served points; the real key, `published`, was never sampled.)

This is a pure unit test of the two projection sites -- no qdrant needed,
per verification-doctrine 1 (unit tests are triage, not coverage; the ranking
claims belong to the gate). The module is reimported under a controlled,
minimal-features environment (as `test_pagecap.py` / `test_pertoken_zero_guard.py`
do) rather than mutating whatever `cytir.serve.retrieval` another test file in
the same pytest session already imported -- `retrieval.py` reads its feature
flags at import time, so sharing the module across test files with different
envs is exactly the kind of order-dependent state this avoids.
"""
from __future__ import annotations

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DENSE = [0.1] * 128

# Minimal surface: two-stage off, every learned/side feature off, so the only
# thing under test is "does the projection dict carry `published`" -- not any
# ranking behaviour (that belongs to the gate, not a unit test).
_ENV = {
    "AW_EMBEDDER": "arctic", "AW_COLLECTION": "test_published_projection",
    "AW_TWO_STAGE": "0", "AW_BM25_PER_TOKEN": "0",
    "AW_RANKHEAD": "0", "AW_QHEAD": "0", "AW_AUTHLEG": "0", "AW_AUTHORITY": "0",
    "AW_COMMERCIAL": "0", "AW_DEDUP_V2": "0", "AW_RELFLOOR": "0",
    "AW_HUBFLOOR": "0", "AW_TRUSTFLOOR": "0", "AW_RERANK": "0",
}


class _Pt:
    def __init__(self, pid, payload, score=1.0):
        self.id, self.payload, self.score = pid, payload, score


def _news_payload(published, url):
    return {"url": url, "anchor": "a", "title": "t", "snippet": "s",
            "source_type": "reference", "lang": "de", "trust_tier": 0.7,
            "last_seen": "2026-09-16", "published": published}


def _load():
    """(re)import `cytir.serve.retrieval`/`models` fresh, under `_ENV`."""
    old = {k: os.environ.get(k) for k in _ENV}
    os.environ.update(_ENV)
    try:
        for mod in [m for m in list(sys.modules) if m.startswith("cytir.serve")]:
            del sys.modules[mod]
        retrieval = importlib.import_module("cytir.serve.retrieval")
        models = importlib.import_module("cytir.serve.models")
        return retrieval, models
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --------------------------------------------------------------------- #
# 1. identifier/entity-card fast path (`_identifier_lookup`)
# --------------------------------------------------------------------- #
def test_identifier_lookup_projects_published():
    retrieval, _ = _load()

    class _Client:
        def scroll(self, collection_name, scroll_filter, limit, with_payload):
            return ([_Pt(1, _news_payload("2026-09-13", "https://swissinfo.ch/1")),
                      _Pt(2, _news_payload(None, "https://srf.ch/hub"))],
                    None)

    retrieval.qdrant_io.client = lambda: _Client()
    # task 637 §22.6: the fast path now refuses a field the served collection
    # has no payload index on, because an unindexed filter on a `scroll` is a
    # full scan. Declare the index this test's premise assumes.
    retrieval.qdrant_io._indexed_fields = frozenset({"register_id", "lei"})
    out = retrieval._identifier_lookup("register_id", "CHE-123.456.789", k=8)
    assert [r["published"] for r in out] == ["2026-09-13", None], out
    print("  [1] _identifier_lookup projects `published` (dated + genuinely-absent)  OK")


# --------------------------------------------------------------------- #
# 2. main fused-ranking path (`search()`)
# --------------------------------------------------------------------- #
def test_main_search_projects_published():
    retrieval, models = _load()

    class _Client:
        def query_points(self, collection_name, query, using, limit,
                          query_filter=None, search_params=None, with_payload=True):
            pts = [_Pt(1, _news_payload("2026-09-13", "https://swissinfo.ch/1"), score=1.0),
                   _Pt(2, _news_payload(None, "https://srf.ch/hub"), score=0.9)]
            return type("R", (), {"points": pts})()

        def query_batch_points(self, collection_name, requests):
            return [type("R", (), {"points": []})() for _ in requests]

        def retrieve(self, collection_name, ids, with_payload=True):
            byid = {1: _news_payload("2026-09-13", "https://swissinfo.ch/1"),
                    2: _news_payload(None, "https://srf.ch/hub")}
            return [_Pt(i, byid[i]) for i in ids]

    # Restored on exit: bare assignments leaked `qdrant_io.client` into later tests of the
    # shipped-tree run, where this file is collected too (task 942).
    import pytest
    mp = pytest.MonkeyPatch()
    mp.setattr(retrieval.qdrant_io, "client", lambda: _Client())
    mp.setattr(models, "embed_query", lambda q: (DENSE, None))
    mp.setattr(models, "last_full_query_vec", lambda: None)
    try:
        hits = retrieval.search("velofahren in der schweiz", k=8, lang="de")
    finally:
        mp.undo()
    assert hits, "search returned no hits"
    by_url = {h["url"]: h for h in hits}
    assert "https://swissinfo.ch/1" in by_url and "https://srf.ch/hub" in by_url, hits
    assert by_url["https://swissinfo.ch/1"]["published"] == "2026-09-13", hits
    assert by_url["https://srf.ch/hub"]["published"] is None, hits
    print(f"  [2] search() projects `published` on {len(hits)} hits "
          "(dated + genuinely-absent)  OK")


if __name__ == "__main__":
    fails = 0
    for fn in (test_identifier_lookup_projects_published,
               test_main_search_projects_published):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print("PASS" if not fails else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)
