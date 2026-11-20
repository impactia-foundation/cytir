#!/usr/bin/env python3
"""Task 682 (task 643 S6): ONE embed stage, parameterized by the collection layout.

`cytir.embed.collection_stage` owns layout creation, the two encoders, the MRL truncation, the
upsert loop and the green wait. the P2 config-B shim (retired, task 1315)
called it and kept its CLI. The real-embedder, real-Qdrant identity proof (old tree vs new tree,
bitwise) is task 682's identity_proof script, recorded in task 682's doc; this file needs
neither a model nor a server: a fake client captures exactly what would be sent.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("qdrant_client")
from qdrant_client import models  # noqa: E402

from cytir import cli  # noqa: E402
from cytir.core import serving_config as SC  # noqa: E402
from cytir.embed import collection_stage as S  # noqa: E402


class FakeClient:
    """Records create_collection / payload index / upsert; `exists` preset."""

    def __init__(self, exists=False):
        self.exists = exists
        self.created = None
        self.indexes = []
        self.upserts = []
        self.deleted = []

    def collection_exists(self, name):
        return self.exists

    def delete_collection(self, name):
        self.deleted.append(name)

    def create_collection(self, **kwargs):
        self.created = kwargs

    def create_payload_index(self, name, field_name, field_schema):
        self.indexes.append((field_name, field_schema))

    def upsert(self, collection, points, wait):
        self.upserts.append((collection, points, wait))


# ------------------------------------------------------------------ the layouts
def test_no_layout_named_is_refused_and_names_the_layouts(monkeypatch):
    monkeypatch.delenv("AW_EMBED_LAYOUT", raising=False)
    assert not hasattr(S, "DEFAULT_LAYOUT")
    with pytest.raises(ValueError, match="no default") as e:
        S.resolve_layout()
    assert all(n in str(e.value) for n in S.LAYOUT_NAMES)


def test_the_served_layout_is_selectable_by_name():
    lay = S.resolve_layout("served")
    assert lay.name == "served" and lay.serving is SC.get(SC.FLIP_SERVING_CONFIG)
    # 256-d float16, product quantization x8, rescore x3
    assert (lay.dense_dim, lay.serving.dense_datatype, lay.serving.quantizer) == (256, "float16", "pq_x8")
    assert lay.serving.rescore and lay.serving.oversampling == 1.0


def test_the_env_names_the_layout(monkeypatch):
    monkeypatch.setenv("AW_EMBED_LAYOUT", "pack-staging")
    assert S.resolve_layout(dense_dim=128).name == "pack-staging"


def test_an_unknown_layout_and_a_widthless_pack_staging_are_refused():
    with pytest.raises(ValueError, match="unknown embed layout"):
        S.resolve_layout("nope")
    with pytest.raises(ValueError, match="explicit dense_dim"):
        S.resolve_layout("pack-staging")


def test_the_served_collection_is_created_as_production_serves_it():
    c = FakeClient()
    # the read-back assertion needs a server; the creation arguments are what is tested
    lay = S.resolve_layout("served")
    object.__setattr__(lay, "verify_shape", False)
    assert S.create_collection(c, "x", lay, recreate=False, log=lambda m: None) is True
    k = c.created
    vec = k["vectors_config"]["dense"]
    assert (vec.size, vec.datatype, vec.on_disk) == (256, models.Datatype.FLOAT16, True)
    assert k["quantization_config"].product.compression == models.CompressionRatio.X8
    assert k["hnsw_config"].m == 16 and k["on_disk_payload"] is True
    sp = k["sparse_vectors_config"]["bm25"]
    assert sp.modifier == models.Modifier.IDF and sp.index.datatype == models.Datatype.FLOAT16
    assert {f for f, _ in c.indexes} == set(SC.PAYLOAD_INDEXES)


@pytest.mark.parametrize("dim", [128, 256, 1024])
def test_pack_staging_reproduces_the_old_stage_pack_b1_collection(dim):
    c = FakeClient()
    S.create_collection(c, "x", S.resolve_layout("pack-staging", dense_dim=dim), recreate=True,
                        log=lambda m: None)
    k = c.created
    assert k["vectors_config"]["dense"].size == dim
    assert k["quantization_config"].scalar.type == models.ScalarType.INT8
    assert k["hnsw_config"].m == 0                               # bulk: no graph
    assert [f for f, _ in c.indexes] == ["lang", "tier", "domain", "provenance", "trust_tier"]


def test_recreate_deletes_and_resume_keeps_an_existing_collection():
    lay = S.resolve_layout("pack-staging", dense_dim=128)
    c = FakeClient(exists=True)
    assert S.create_collection(c, "x", lay, recreate=False, log=lambda m: None) is False
    assert c.created is None and c.deleted == []
    assert S.create_collection(c, "x", lay, recreate=True, log=lambda m: None) is True
    assert c.deleted == ["x"] and c.created is not None


# ------------------------------------------------------------------ the loop
def _fake_encoder(dim=1024):
    def encode(texts, batch=None):
        rng = np.random.default_rng(len(texts))
        v = rng.normal(size=(len(texts), dim)).astype("float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True), None
    return encode


def test_embed_rows_truncates_renormalizes_and_omits_an_absent_sparse():
    c = FakeClient()
    lay = S.resolve_layout("pack-staging", dense_dim=128)
    rows = [S.Row("00000000-0000-4000-8000-00000000000%d" % i, "t%d" % i,
                  ([1, 2], [0.5, 1.5]) if i % 2 == 0 else None, {"lang": "de"}) for i in range(5)]
    res = S.embed_rows(c, "coll", rows, lambda chunk: list(chunk), lay, _fake_encoder(),
                       batch=2, log=lambda m: None)
    assert res["points"] == 5 and len(c.upserts) == 3
    pts = [p for _, ps, _ in c.upserts for p in ps]
    assert all(len(p.vector["dense"]) == 128 for p in pts)
    assert all(abs(np.linalg.norm(p.vector["dense"]) - 1.0) < 1e-5 for p in pts)
    assert ["bm25" in p.vector for p in pts] == [True, False, True, False, True]
    assert all(wait is False for _, _, wait in c.upserts)


def test_embed_rows_uses_the_one_mrl_trunc():
    c = FakeClient()
    lay = S.resolve_layout("served")
    enc = _fake_encoder()
    S.embed_rows(c, "coll", [S.Row("00000000-0000-4000-8000-000000000000", "t", None, {})],
                 lambda chunk: list(chunk), lay, enc, batch=1, log=lambda m: None)
    want = SC.mrl_trunc(enc(["t"])[0], 256)[0]
    got = c.upserts[0][1][0].vector["dense"]
    assert np.array_equal(np.asarray(got, dtype="float32"), np.asarray(want, dtype="float32"))


# ------------------------------------------------------------------ the doors
def test_the_retired_index_door_always_refuses_and_names_the_replacement(monkeypatch):
    monkeypatch.setattr("cytir.core.qdrant.qdrant_client",
                        lambda *a, **k: pytest.fail("connected before the refusal"))
    for extra in ([], ["--layout", "served"]):
        with pytest.raises(SystemExit, match="cyt index build"):
            S.main(["--in", "x.jsonl", "--collection", "c", *extra])
    assert not hasattr(S, "legacy_config_b_materializer")


def test_the_moved_names_are_the_packages_objects():
    from cytir.embed import bakeoff as ib
    from cytir.core import ids
    from cytir.embed import model
    from cytir.index import stage_pack_b1 as spb
    assert ib.load_embedder is model.load_embedder and ib.EMBEDDERS is model.EMBEDDERS
    assert ib.point_id is ids.record_point_id
    assert ib.existing_ids is S.existing_ids
    assert spb.tei_doc_encoder is S.tei_doc_encoder


def test_no_second_door_creates_a_collection_or_upserts():
    """One stage owns the verb: the two callers hold no `create_collection`/`upsert` call."""
    for name in ("cytir/index/stage_pack_b1.py",):
        tree = ast.parse((HERE.parent / name).read_text(encoding="utf-8"))
        # a call on the stage module (`STAGE.create_collection`) is the door; a call on
        # anything else (a qdrant client) would be a second implementation
        calls = {n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and not (isinstance(n.func.value, ast.Name)
                          and n.func.value.id in ("STAGE", "_stage"))}
        assert not calls & {"create_collection", "upsert", "create_payload_index"}, name



def test_aw_embed_dispatches_to_the_three_modules(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr("cytir.embed.collection_stage.main", lambda argv, **kw: calls.append(("index", argv, kw)) or 0)
    monkeypatch.setattr("cytir.embed.shards.main", lambda argv: calls.append(("shards", argv)))
    monkeypatch.setattr("cytir.embed.stage.main", lambda argv: calls.append(("stage", argv)) or 0)
    assert cli.main(["embed", "index", "--in", "a"]) == 0
    assert cli.main(["embed", "shards", "--in", "d"]) == 0
    assert cli.main(["embed", "stage", "d"]) == 0
    assert [c[0] for c in calls] == ["index", "shards", "stage"]
    assert calls[0][1] == ["--in", "a"] and calls[0][2]["prog"] == "cyt embed index"
    assert cli.main(["embed"]) == 2
    assert cli.main(["embed", "nope"]) == 2
