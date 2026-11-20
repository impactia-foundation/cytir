"""cytir.embed.bakeoff (task 2329): the bake-off INDEX stage lives in the package; its collection
goes through the one creator with the request kwargs the old script sent."""
from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")
from qdrant_client import models as qm  # noqa: E402

from cytir.core.common import DENSE_VEC, SPARSE_BM25, SPARSE_LEXICAL  # noqa: E402
from cytir.embed import bakeoff as B  # noqa: E402


class FakeClient:
    def __init__(self, exists=False):
        self.exists, self.created, self.indexes, self.updates, self.deleted = exists, [], [], [], []

    def collection_exists(self, name):
        return self.exists

    def delete_collection(self, name):
        self.deleted.append(name)

    def create_collection(self, **kw):
        self.created.append(kw)

    def create_payload_index(self, name, field_name, field_schema):
        self.indexes.append((field_name, field_schema))

    def update_collection(self, name, **kw):
        self.updates.append((name, kw))


def test_config_b_request_is_what_the_old_script_sent():
    c = FakeClient()
    B.ensure_collection(c, "col", 128, {"bm25"}, "int8", recreate=True, on_disk=True)
    kw, = c.created
    assert kw == dict(
        collection_name="col",
        vectors_config={DENSE_VEC: qm.VectorParams(size=128, distance=qm.Distance.COSINE, on_disk=True)},
        sparse_vectors_config={SPARSE_BM25: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        quantization_config=qm.ScalarQuantization(
            scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8, always_ram=True)))
    assert [f for f, _ in c.indexes] == ["lang", "tier", "domain", "trust_tier"]


def test_binary_quant_and_lexical_leg():
    c = FakeClient()
    B.ensure_collection(c, "col", 1024, {"bm25", "lexical"}, "binary")
    q = c.created[0]["quantization_config"]
    assert isinstance(q, qm.BinaryQuantization) and q.binary.always_ram is True
    assert c.updates and SPARSE_LEXICAL in c.updates[0][1]["sparse_vectors_config"]


def test_existing_collection_is_kept_on_resume_and_rebuilt_otherwise():
    kept = FakeClient(exists=True)
    B.ensure_collection(kept, "col", 128, {"bm25"}, "int8", recreate=False)
    assert kept.created == [] and kept.deleted == []
    rebuilt = FakeClient(exists=True)
    B.ensure_collection(rebuilt, "col", 128, {"bm25"}, "int8", recreate=True)
    assert rebuilt.deleted == ["col"] and len(rebuilt.created) == 1


def test_unknown_quant_is_refused():
    with pytest.raises(ValueError):
        B.ensure_collection(FakeClient(), "col", 128, {"bm25"}, "pq")


def test_the_moved_names_are_the_packages_objects():
    from cytir.core import ids, serving_config
    from cytir.embed import collection_stage, model
    assert B.load_embedder is model.load_embedder and B.EMBEDDERS is model.EMBEDDERS
    assert B.point_id is ids.record_point_id and B.mrl_trunc is serving_config.mrl_trunc
    assert B.existing_ids is collection_stage.existing_ids
