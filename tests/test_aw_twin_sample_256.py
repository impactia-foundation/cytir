"""Stranger path vs served vectors (256 layout), five frozen rows.

The served layout stores the first 256 dims of the 1024-d arctic vector, L2
renormalised, as float16. Each frozen row carries the record, the served
vector and the vector the public CPU path produced. The composition check is
offline. The embedding check skips when the pinned model is not cached.
"""
import base64
import json
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "data" / "twin-sample-256.v1.json"
TOL = 0.999


def _load():
    return json.loads(FIXTURE.read_text())


def _vec(b64):
    return np.frombuffer(base64.b64decode(b64), dtype="<f2").astype("f4")


def _cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def _model_cached(revision):
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    hub = Path(os.environ.get("HF_HUB_CACHE") or home / "hub")
    return (hub / "models--Snowflake--snowflake-arctic-embed-l-v2.0" / "snapshots" / revision).is_dir()


def test_composition_reproduces_the_frozen_input_text_byte_for_byte():
    from cytir.core.common import record_text_channels
    data = _load()
    assert len(data["rows"]) == 5
    for row in data["rows"]:
        dense, _sparse = record_text_channels(row["record"])
        assert dense.encode("utf-8") == row["dense_text"].encode("utf-8")


def test_frozen_rows_include_dense_text_different_from_text():
    rows = _load()["rows"]
    assert sum(r["record"]["dense_text"] != r["record"]["text"] for r in rows) >= 2


def test_embed_verb_reads_the_dense_channel(tmp_path):
    import gzip
    from cytir.embed.shards import read_texts
    rows = [{"point_id": "a", "text": "plain", "dense_text": "dense surface"},
            {"point_id": "b", "text": "only text"}]
    path = tmp_path / "s.texts.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ids, texts = read_texts(str(path))
    assert ids == ["a", "b"]
    assert texts == ["dense surface", "only text"]


def test_embed_verb_input_matches_frozen_dense_text(tmp_path):
    import gzip
    from cytir.embed.shards import read_texts
    rows = _load()["rows"]
    path = tmp_path / "f.texts.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps(dict(r["record"], point_id=str(i))) + "\n")
    _ids, texts = read_texts(str(path))
    assert texts == [r["dense_text"] for r in rows]


def test_frozen_vectors_agree():
    data = _load()
    for row in data["rows"]:
        a, b = _vec(row["twin_f16_b64"]), _vec(row["stranger_f16_b64"])
        assert a.shape == b.shape == (data["width"],)
        assert _cos(a, b) >= TOL


def test_cpu_embed_reproduces_the_served_vector():
    for mod in ("torch", "sentence_transformers"):
        pytest.importorskip(mod)
    data = _load()
    if not _model_cached(data["revision"]):
        pytest.skip("pinned embedder %s is not in the Hugging Face cache" % data["revision"][:8])
    from sentence_transformers import SentenceTransformer
    from cytir.core.serving_config import mrl_trunc
    model = SentenceTransformer(data["model"], trust_remote_code=True, device="cpu",
                                revision=data["revision"], local_files_only=True)
    model.max_seq_length = 512
    texts = [r["dense_text"] for r in data["rows"]]
    full = model.encode(texts, normalize_embeddings=True).astype("f2").astype("f4")
    got = np.asarray(mrl_trunc(full, data["width"]))
    for g, row in zip(got, data["rows"]):
        assert _cos(g, _vec(row["twin_f16_b64"])) >= TOL
