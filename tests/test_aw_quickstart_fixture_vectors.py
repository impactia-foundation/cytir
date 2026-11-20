"""Sealed CPU vectors for the QUICKSTART fixture rows (task 915, follow-up D4).

The seal (`tests/data/index-fixture-dense-cpu.v1.json`) holds the vectors `cyt embed shards
--device cpu --compute-dtype float32` writes for the seven committed fixture stages, at the pinned
model revision. This test re-embeds the stages on CPU and asserts every row has cosine >= 0.999
against the seal. It skips when torch, sentence-transformers or the cached model is missing.
Measured CPU-vs-GPU(fp16) minimum cosine was 0.99986, so 0.999 leaves a 7x margin on the deviation.
"""
import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parent.parent
SEAL = ROOT / "tests" / "data" / "index-fixture-dense-cpu.v1.json"
TOL = 0.999


def _load_seal():
    return json.loads(SEAL.read_text())


def _model_cached(revision):
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    hub = Path(os.environ.get("HF_HUB_CACHE") or home / "hub")
    snap = hub / "models--Snowflake--snowflake-arctic-embed-l-v2.0" / "snapshots" / revision
    return snap.is_dir()


def test_seal_is_self_consistent():
    seal = _load_seal()
    raw = base64.b64decode(seal["vectors_b64"])
    assert hashlib.sha256(raw).hexdigest() == seal["vectors_sha256"]
    assert len(raw) == seal["n"] * seal["width"] * 2
    assert len(seal["row_ids"]) == len(set(seal["row_ids"])) == seal["n"]


def test_cpu_reembed_matches_seal(tmp_path):
    seal = _load_seal()
    for mod in ("torch", "sentence_transformers"):
        pytest.importorskip(mod)
    if not _model_cached(seal["revision"]):
        pytest.skip("pinned embedder %s is not in the Hugging Face cache; run `cyt embed shards` once"
                    % seal["revision"][:8])
    sealed = np.frombuffer(base64.b64decode(seal["vectors_b64"]), dtype="<f2").reshape(-1, seal["width"])
    env = dict(os.environ, PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1")
    got_ids, got = [], []
    for stage in seal["stages"]:
        out = tmp_path / stage
        subprocess.run([sys.executable, "-m", "cytir.cli", "embed", "shards",
                        "--in", str(ROOT / "data" / "index-fixture" / stage / "chunks"),
                        "--out", str(out), "--device", "cpu", "--compute-dtype", "float32"],
                       check=True, env=env, cwd=str(ROOT), capture_output=True)
        got_ids += [json.loads(l)["point_id"] for l in (out / "shard-0000.row_ids.jsonl").read_text().splitlines()]
        got.append(np.frombuffer((out / "shard-0000.vectors.f16").read_bytes(), dtype="<f2")
                   .reshape(-1, seal["width"]))
    assert got_ids == seal["row_ids"]
    a = np.concatenate(got).astype("f4")
    b = sealed.astype("f4")
    cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    assert cos.min() >= TOL, "min cosine %.6f < %s" % (cos.min(), TOL)
