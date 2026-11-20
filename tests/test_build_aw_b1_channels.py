"""Offline contract tests for full-rebuild two-channel corpus handling."""
from __future__ import annotations

import importlib
import pathlib
import sys
from types import ModuleType, SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def _load(module_name: str):
    qdrant = ModuleType("qdrant_client")
    qdrant.QdrantClient = object
    qdrant.models = SimpleNamespace(
        PayloadSchemaType=SimpleNamespace(KEYWORD="keyword", FLOAT="float"),
        SparseVector=lambda **kwargs: kwargs)
    # Needs no qdrant_client. Imported for real BEFORE the snapshot so the cleanup below never pops
    # it: a module popped and later re-imported is a second object, and `url_key` then differs
    # between importers (test_task515_serving_metadata's mirror-contract check).
    importlib.import_module("cytir.index.serving_metadata")
    old = sys.modules.get("qdrant_client")
    before = set(sys.modules)
    sys.modules["qdrant_client"] = qdrant
    try:
        return importlib.import_module(module_name)
    finally:
        # Any aw.* module first imported under the stub bound `qm` to the fake
        # models (e.g. cytir.index.b1_projection); drop them so later importers
        # get the real qdrant_client (task 2119: poisoned test_task515).
        for key in set(sys.modules) - before:
            if key == "cytir" or key.startswith("cytir."):
                sys.modules.pop(key, None)
        if old is None:
            sys.modules.pop("qdrant_client", None)
        else:
            sys.modules["qdrant_client"] = old


build = _load("cytir.index.b1_slice")
from cytir.index import b1_corpus_kv as KV  # noqa: E402
from cytir.index import b1_projection as BP  # noqa: E402


class Txn:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value


def test_full_rebuild_kv_preserves_atomic_two_channel_pair():
    txn = Txn()
    key = b"x" * 16
    KV._kv_put_channels(
        txn, key, "article-local", "heading. article-local", "a" * 64)
    assert KV._kv_get_channels(txn, key) == {
        "dense_text": "article-local",
        "sparse_text": "heading. article-local",
        "bundle_digest": "a" * 64,
    }


def test_full_rebuild_uses_the_shared_title_site_composition_contract():
    record = {"text": "Article body.", "title": "OR — Art. 257",
              "site_name": "Fedlex", "source_type": "official"}
    from cytir.core import common as aw_common
    assert BP.compose_index_text(record) == aw_common.compose_index_text(record)


def test_full_rebuild_refuses_conflicting_or_incomplete_channel_contract():
    txn = Txn()
    key = b"x" * 16
    KV._kv_put_channels(txn, key, "one", "heading. one", "a" * 64)
    import pytest
    with pytest.raises(ValueError, match="conflicting"):
        KV._kv_put_channels(txn, key, "two", "heading. two", "b" * 64)
    with pytest.raises(ValueError, match="incomplete"):
        BP._channel_surfaces({"dense_text": "one"}, "fallback")


def test_full_rebuild_uses_sparse_hierarchy_and_reembeds_dense_local_surface():
    source = SimpleNamespace(
        id="old", vector={"dense": [0.1, 0.2]}, payload={
            "url": "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_257",
            "anchor": "art_257", "title": "OR — Art. 257", "site_name": "Fedlex",
            "lang": "de", "domain": "fedlex.admin.ch", "source_type": "official",
            "trust_tier": 1.0,
            "bundle_digest": "a" * 64,
        })
    # `verdict` also receives the record payload (ADR-076 dump basis, task 433).
    gate = SimpleNamespace(
        verdict=lambda _host, _row, _payload=None: (True, "ok"))
    point, meta = BP.project_record(
        source, "fedlex", {
            "texts": ["Art. 257 Der Mietzins."],
            "dense_text": "Art. 257 Der Mietzins.",
            "sparse_text": "Achter Titel: Die Miete. Art. 257 Der Mietzins.",
        }, {}, {}, gate, build.defaultdict(int))
    assert meta == {}
    assert point["dense_reembed_text"] is not None
    assert "Die Miete" not in point["dense_reembed_text"]
    assert "Fedlex" in point["dense_reembed_text"]
    assert point["payload"]["content_digest"] == build.ID.content_digest(
        "Art. 257 Der Mietzins.")
    assert point["payload"]["bundle_digest"] == "a" * 64


def test_slice_upsert_uses_the_same_explicit_dense_reembed_hook(monkeypatch):
    called = []
    monkeypatch.setattr(build, "_reembed_explicit_dense",
                        lambda chunk: called.append(list(chunk)) or 1)
    monkeypatch.setattr(build.qm, "PointStruct", lambda **kwargs: kwargs,
                        raising=False)

    class Client:
        def __init__(self):
            self.calls = []

        def upsert(self, name, points, wait):
            self.calls.append((name, points, wait))

    point = {"id": "explicit", "vector": {"dense": [0.0]}, "payload": {},
             "dense_reembed_text": "article-local"}
    client = Client()
    build.upsert_points(client, "slice", [point], batch=1, log=lambda _msg: None)
    assert called == [[point]]
    assert len(client.calls) == 1


def test_dense_contract_sample_always_includes_explicit_channel_cohort():
    ordinary = [{"id": f"ordinary-{i}", "vector": {"dense": [0.0]}}
                for i in range(10_000)]
    explicit = [{"id": f"explicit-{i}", "vector": {"dense": [0.0]},
                 "dense_reembed_text": "article-local"}
                for i in range(3)]
    sample = build._dense_contract_sample(ordinary + explicit,
                                          generic_limit=2, explicit_limit=10)
    assert {p["id"] for p in explicit}.issubset({p["id"] for p in sample})
