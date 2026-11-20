#!/usr/bin/env python3
"""The identifier fast path must refuse a field the served collection has no
payload index on (task 637 §22).

WHY THIS TEST EXISTS. `_detect_identifier` routes any 20-character alphanumeric
query to `_identifier_lookup("lei", ...)`, which builds a `FieldCondition` and
`scroll`s on it. On `aw_e8` that is ~1 ms because `lei` carries a keyword index.
`aw_e9` drops that index -- it stands over 0 points and costs 16.9 B/pt of disk
-- and an unindexed filter on a `scroll` is a FULL SCAN of 33.7M points: 54 s
measured, and it saturates the API pool into 503s. It is a slow success, not an
exception, so the fast path's `except` cannot see it. The guard is the only
thing between a dropped index and a query shape that takes the serving box down.
"""
import os
import sys

os.environ.update({"AW_EMBEDDER": "arctic", "AW_COLLECTION": "test_id_guard"})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cytir.serve import retrieval  # noqa: E402


class _Boom:
    """A qdrant client that fails the test if anything scrolls."""

    def scroll(self, *a, **kw):  # pragma: no cover - must never be reached
        raise AssertionError("scrolled on an unindexed field")


def _set_indexed(fields):
    retrieval.qdrant_io._indexed_fields = None if fields is None else frozenset(fields)


def test_lei_query_routes_to_the_lei_field():
    # the premise: this really is the shape that reaches the dropped index
    assert retrieval._detect_identifier("5493001KJTIIGC8Y1R12") == (
        "lei", "5493001KJTIIGC8Y1R12")


def test_refuses_when_the_field_is_not_indexed(monkeypatch):
    _set_indexed(["url", "domain", "lang", "country", "source_type",
                  "trust_tier", "published", "register_id"])  # aw_e9's eight
    monkeypatch.setattr(retrieval.qdrant_io, "client", lambda: _Boom())
    assert retrieval._identifier_lookup("lei", "5493001KJTIIGC8Y1R12", 10) == []
    _set_indexed(None)


def test_runs_when_the_field_is_indexed(monkeypatch):
    _set_indexed(["register_id", "lei"])  # aw_e8's shape
    seen = {}

    class _Ok:
        def scroll(self, collection_name, scroll_filter, limit, with_payload):
            seen["limit"] = limit
            return [], None

    monkeypatch.setattr(retrieval.qdrant_io, "client", lambda: _Ok())
    assert retrieval._identifier_lookup("lei", "5493001KJTIIGC8Y1R12", 10) == []
    assert seen["limit"] == 64, "task 199's k-independent scroll window"
    _set_indexed(None)


def test_a_failed_introspection_is_not_cached(monkeypatch):
    """A transient qdrant error must cost one request's fast path, not the
    process's -- caching the empty set would silently disable the identifier
    lookup until the next deploy."""
    _set_indexed(None)
    calls = []

    class _Down:
        def get_collection(self, name):
            calls.append(name)
            raise RuntimeError("connection refused")

    monkeypatch.setattr(retrieval.qdrant_io, "health_client", lambda: _Down())
    assert retrieval.indexed_fields() == frozenset()
    assert retrieval.indexed_fields() == frozenset()
    assert len(calls) == 2, "the failure was cached"
    assert retrieval.qdrant_io._indexed_fields is None


def test_a_successful_introspection_is_cached(monkeypatch):
    _set_indexed(None)
    calls = []

    class _Up:
        def get_collection(self, name):
            calls.append(name)
            return type("C", (), {"payload_schema": {"lang": object(),
                                                     "register_id": object()}})()

    monkeypatch.setattr(retrieval.qdrant_io, "health_client", lambda: _Up())
    assert retrieval.indexed_fields() == frozenset({"lang", "register_id"})
    assert retrieval.indexed_fields() == frozenset({"lang", "register_id"})
    assert len(calls) == 1
    _set_indexed(None)
