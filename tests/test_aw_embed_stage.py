"""Task 657 — `cytir.embed.stage`: the GPU-lock door in front of `cytir.embed.shards`.

No GPU or sentence_transformers needed: `gpu_compute_pids`/`check_gpu_idle`
are tested against a fake `nvidia-smi` on PATH (and its absence), `gpu_lease`
is tested against the real flock/threading machinery (mutual exclusion,
priority-sentinel yield, legacy marker directory), and `embed_stage`'s own
tests stub `subprocess.run` to prove the argv/env CONTRACT it hands to
`python -m cytir.embed.shards` (that module's own tests cover its behaviour;
this file is not a second test of it).
"""
from __future__ import annotations

import os
import stat
import sys
import threading
import time

import pytest

import cytir.embed.stage as stage


def _fake_nvidia_smi(tmp_path, pids: list[int]):
    """A fake `nvidia-smi` script on PATH reporting `pids` as compute-app
    holders, in the exact CSV shape `gpu_compute_pids` parses."""
    script = tmp_path / "nvidia-smi"
    lines = "\n".join(str(p) for p in pids)
    script.write_text(f"#!/bin/sh\nprintf '%s\\n' '{lines}'\n" if pids else
                      "#!/bin/sh\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_gpu_compute_pids_empty_when_nvidia_smi_absent(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent-bin-only")
    assert stage.gpu_compute_pids() == []


def test_gpu_compute_pids_parses_real_shape(tmp_path, monkeypatch):
    _fake_nvidia_smi(tmp_path, [1234, 5678])
    monkeypatch.setenv("PATH", str(tmp_path))
    assert stage.gpu_compute_pids() == [1234, 5678]


def test_check_gpu_idle_passes_when_empty(tmp_path, monkeypatch):
    _fake_nvidia_smi(tmp_path, [])
    monkeypatch.setenv("PATH", str(tmp_path))
    stage.check_gpu_idle()  # must not raise


def test_check_gpu_idle_refuses_when_busy(tmp_path, monkeypatch):
    _fake_nvidia_smi(tmp_path, [42])
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(stage.GPUBusyError, match=r"\[42\]"):
        stage.check_gpu_idle()


def test_gpu_lease_is_mutually_exclusive(tmp_path):
    """Two lease attempts on the SAME lock file must never overlap --
    the exact property the flock exists to guarantee for two embed
    invocations racing on one GPU."""
    lock = str(tmp_path / "gpu0.lock")
    legacy = str(tmp_path / "GPU-LOCK")
    overlap = []
    order = []

    def hold(tag, seconds):
        with stage.gpu_lease(gpu_lock=lock, legacy_lock_dir=legacy,
                             priority_sentinel=str(tmp_path / "no-sentinel"),
                             log=lambda _m: None):
            order.append(("enter", tag))
            if order.count(("enter", "a")) and order.count(("enter", "b")) \
                    and len(order) == 2:
                overlap.append(True)
            time.sleep(seconds)
            order.append(("exit", tag))

    t1 = threading.Thread(target=hold, args=("a", 0.2))
    t2 = threading.Thread(target=hold, args=("b", 0.0))
    t1.start()
    time.sleep(0.05)  # let t1 acquire first
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    # t2 must not have entered while t1 was still inside its critical
    # section: exit(a) must precede enter(b).
    assert order.index(("exit", "a")) < order.index(("enter", "b"))


def test_gpu_lease_yields_to_priority_sentinel(tmp_path):
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("held by a higher-priority lane")
    entered = []

    def release_sentinel_soon():
        time.sleep(0.15)
        sentinel.unlink()

    threading.Thread(target=release_sentinel_soon, daemon=True).start()
    t0 = time.time()
    with stage.gpu_lease(gpu_lock=str(tmp_path / "gpu0.lock"),
                         legacy_lock_dir=str(tmp_path / "GPU-LOCK"),
                         priority_sentinel=str(sentinel),
                         poll_seconds=0.05,
                         log=lambda _m: entered.append(_m)):
        pass
    assert time.time() - t0 >= 0.1, (
        "gpu_lease must not acquire the lock while the priority sentinel "
        "file still exists")


def test_gpu_lease_creates_and_removes_legacy_marker(tmp_path):
    legacy = tmp_path / "GPU-LOCK"
    assert not legacy.exists()
    with stage.gpu_lease(gpu_lock=str(tmp_path / "gpu0.lock"),
                         legacy_lock_dir=str(legacy),
                         priority_sentinel=str(tmp_path / "no-sentinel"),
                         log=lambda _m: None):
        assert legacy.is_dir()
    assert not legacy.exists()


def _clean_stage(tmp_path):
    """A one-row stage that passes the embed-stage consent gate (task 677:
    `embed_stage` now reads its chunk dir and refuses a missing/empty one)."""
    import gzip
    import json
    chunks = tmp_path / "chunks"
    chunks.mkdir(exist_ok=True)
    with gzip.open(chunks / "shard-0000.texts.jsonl.gz", "wt") as fh:
        fh.write(json.dumps({"point_id": "p0", "text": "t",
                             "url": "https://www.fedlex.admin.ch/eli/x"}) + "\n")
    return chunks


def test_embed_stage_refuses_when_gpu_busy(tmp_path, monkeypatch):
    _clean_stage(tmp_path)
    _fake_nvidia_smi(tmp_path, [999])
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    with pytest.raises(stage.GPUBusyError):
        stage.embed_stage(str(tmp_path / "chunks"), str(tmp_path / "vectors"),
                          gpu_lock=str(tmp_path / "gpu0.lock"),
                          legacy_lock_dir=str(tmp_path / "GPU-LOCK"),
                          priority_sentinel=str(tmp_path / "no-sentinel"))


def test_embed_stage_invokes_aw_embed_shards_under_the_lease(tmp_path):
    """`skip_idle_check=True` (no real GPU here); proves embed_stage builds
    the expected `python -m cytir.embed.shards --in/--out/--model/--revision`
    argv, runs it inside the lease (offline env vars set), and returns its
    exit code. `subprocess.run` is stubbed (not shelled out for real) so
    this proves the CONTRACT `embed_stage` hands to the subprocess, which
    is the thing a caller (the compose module, the chain hook) depends on
    -- not a re-test of `cytir.embed.shards` itself (that module's own tests
    cover it)."""
    _clean_stage(tmp_path)
    import types

    captured = {}

    def fake_run(argv, env=None, **kw):
        captured["argv"] = argv
        captured["env"] = env
        return types.SimpleNamespace(returncode=0)

    orig = stage.subprocess.run
    stage.subprocess.run = fake_run
    try:
        rc = stage.embed_stage(
            str(tmp_path / "chunks"), str(tmp_path / "vectors"),
            model="fake-model", revision="a" * 40,
            gpu_lock=str(tmp_path / "gpu0.lock"),
            legacy_lock_dir=str(tmp_path / "GPU-LOCK"),
            priority_sentinel=str(tmp_path / "no-sentinel"),
            skip_idle_check=True, log=lambda _m: None)
    finally:
        stage.subprocess.run = orig

    assert rc == 0
    argv = captured["argv"]
    assert argv[:3] == [sys.executable, "-m", "cytir.embed.shards"]
    assert "--in" in argv and str(tmp_path / "chunks") in argv
    assert "--out" in argv and str(tmp_path / "vectors") in argv
    assert "--model" in argv and "fake-model" in argv
    assert "--revision" in argv and "a" * 40 in argv
    assert captured["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["env"]["TRANSFORMERS_OFFLINE"] == "1"


def test_embed_stage_defaults_model_and_revision_from_the_pin(tmp_path):
    """No --model/--revision passed -> the committed pin, imported from
    `cytir.core.model_pin` (the mirror contract), not a second literal."""
    _clean_stage(tmp_path)
    import types

    from cytir.core.model_pin import EMBED_MODEL_ID, embed_model_revision

    captured = {}

    def fake_run(argv, env=None, **kw):
        captured["argv"] = argv
        return types.SimpleNamespace(returncode=0)

    orig = stage.subprocess.run
    stage.subprocess.run = fake_run
    try:
        stage.embed_stage(
            str(tmp_path / "chunks"), str(tmp_path / "vectors"),
            gpu_lock=str(tmp_path / "gpu0.lock"),
            legacy_lock_dir=str(tmp_path / "GPU-LOCK"),
            priority_sentinel=str(tmp_path / "no-sentinel"),
            skip_idle_check=True, log=lambda _m: None)
    finally:
        stage.subprocess.run = orig

    argv = captured["argv"]
    assert EMBED_MODEL_ID in argv
    assert embed_model_revision() in argv
