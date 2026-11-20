"""Task 2538: `embedder.max_seq_length` is part of the declared embedding surface.

Recorded always; refused only when the pack AND the collection's profile both declare a cap
and the two differ. Absent on either side records (or records nothing) and passes."""
from __future__ import annotations

import json

import pytest

from cytir.core import serving_profile as SP
from cytir.core.model_pin import EMBED_MODEL_ID, embed_model_revision
from cytir.index import build as B


def _pack(tmp_path, caps):
    """A vectors dir with one shard manifest per entry of `caps` (None = no cap declared)."""
    vec = tmp_path / "vectors"
    vec.mkdir()
    for i, cap in enumerate(caps):
        emb = {"model": EMBED_MODEL_ID, "revision_pinned": embed_model_revision()}
        if cap is not None:
            emb["max_seq_length"] = cap
        (vec / f"shard-{i:04d}.vectors.manifest.json").write_text(json.dumps({
            "schema": "aw.413a.vectors.v1", "n": 1, "dim": B.SEALED_DIM, "embedder": emb}))
    return B.StagePack(pack_id="p", family="f", chunks_dir=str(tmp_path / "chunks"),
                       vectors_dir=str(vec), source=str(tmp_path))


@pytest.fixture(autouse=True)
def _dim(monkeypatch):
    monkeypatch.setattr(B.COMPOSE, "read_vector_native_dim", lambda d: B.SEALED_DIM)


def test_present_and_equal_passes_and_is_recorded(tmp_path):
    assert B.check_vectors(_pack(tmp_path, [2560, 2560]), 2560) == [2560]


def test_present_and_different_refuses_with_the_reason(tmp_path):
    with pytest.raises(B.BuildRefusal, match=r"max_seq_length 1024 differs .*embedder_max_seq 2560"):
        B.check_vectors(_pack(tmp_path, [1024]), 2560)


def test_collection_declares_nothing_records_the_pack_value(tmp_path):
    assert B.check_vectors(_pack(tmp_path, [1024]), None) == [1024]


def test_pack_declares_nothing_passes_and_records_nothing(tmp_path):
    assert B.check_vectors(_pack(tmp_path, [None]), 2560) == []


def test_receipt_carries_the_recorded_value(tmp_path):
    pack = _pack(tmp_path, [2560])
    caps = B.check_vectors(pack, None)
    assert B._pack_receipt(pack, None, None, tmp_path / "state", caps)["embedder_max_seq"] == [2560]


def test_profile_digest_unchanged_while_undeclared_and_roundtrips_when_declared():
    assert "embedder_max_seq" not in SP.SERVED.manifest()
    declared = SP.SERVED.with_(embedder_max_seq=2560).with_(name="served-cap-test")
    assert declared.manifest()["embedder_max_seq"] == 2560
    assert declared.digest() != SP.SERVED.digest()
    assert SP.ServingProfile.from_block(declared.block()).embedder_max_seq == 2560


def test_none_none_passes_and_receipt_records_empty(tmp_path):
    pack = _pack(tmp_path, [None])
    caps = B.check_vectors(pack, None)
    assert caps == []
    assert B._pack_receipt(pack, None, None, tmp_path / "state", caps)["embedder_max_seq"] == []


def test_a_non_integer_cap_refuses(tmp_path):
    with pytest.raises(B.BuildRefusal, match="not an integer"):
        B.check_vectors(_pack(tmp_path, ["2560"]), None)


def test_run_passes_the_profile_cap_into_check_vectors(tmp_path, monkeypatch):
    """Through run(), not check_vectors called directly: reverting the call must fail this."""
    declared = SP.SERVED.with_(embedder_max_seq=2560)
    monkeypatch.setattr(B.SP, "get", lambda name: declared)
    monkeypatch.setattr(B, "check_run_and_target", lambda url, coll, out, *, allow_remote: (url, None))
    pack = _pack(tmp_path, [1024])
    with pytest.raises(B.BuildRefusal, match="differs from the collection's declared"):
        B.run(mode="build", packs=[pack], url="http://127.0.0.1:1", collection="aw_build_x",
              profile_name="served", out_dir=tmp_path / "out")
