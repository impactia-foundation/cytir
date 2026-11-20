"""Task 2370: the staged Qdrant hop walk behind `cyt ops qdrant-hop-verify chain` (fixtures, no docker, no Qdrant)."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from cytir import cli
from cytir.ops import qdrant_hop_chain as HC
from cytir.ops import qdrant_hop_verify as HV


def _args(tmp_path, *extra, mode="snapshot"):
    snap = tmp_path / "src.snapshot"
    snap.write_bytes(b"x" * 2048)
    argv = ["--collection", "aw_x", "--work-dir", str(tmp_path / "w"), "--mode", mode]
    if mode == "snapshot":
        argv += ["--source-snapshot", str(snap)]
    else:
        (tmp_path / "st").mkdir(exist_ok=True)
        argv += ["--source-storage", str(tmp_path / "st"), "--i-understand-inplace-is-irreversible"]
    return HC.build_parser("t").parse_args(argv + list(extra))


def test_container_name_and_run_args():
    assert HC.container_name("p", "v1.13.6") == "p-v1-13-6"
    a = HC.docker_run_args(name="n", ver="v1.16.3", port=6421, image_repo="qdrant/qdrant", storage=Path("/s"),
                           snaps=Path("/n"), input_dir=Path("/in"), input_subdir="_hopin", nofile=99)
    assert "127.0.0.1:6421:6333" in a and "nofile=99:99" in a
    assert "/in:/qdrant/snapshots/_hopin:ro" in a and a[-1] == "qdrant/qdrant:v1.16.3"
    assert not any(":ro" in x for x in HC.docker_run_args(
        name="n", ver="v", port=1, image_repo="r", storage=Path("/s"), snaps=Path("/n"),
        input_dir=None, input_subdir="_hopin", nofile=1))


def test_disk_gate():
    assert HC.disk_gate(100, 300, 2.5) == 250
    with pytest.raises(HC.HopError, match="insufficient"):
        HC.disk_gate(100, 200, 2.5)


def test_size_kb(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a").write_bytes(b"x" * 1500)
    assert HC.size_kb(tmp_path / "d") == 2 and HC.size_kb(tmp_path / "none") == 0


def test_recover_body_and_snapshot_name():
    assert HC.recover_body("/q/s") == {"location": "file:///q/s"}
    assert HC.snapshot_name({"result": {"name": "a.snapshot"}}) == "a.snapshot"
    with pytest.raises(HC.HopError):
        HC.snapshot_name({"result": {}})


def test_hop_copy_env_sets_and_restores(monkeypatch):
    monkeypatch.delenv(HV.HOP_COPY_ENV, raising=False)
    with HC.hop_copy_env():
        assert os.environ[HV.HOP_COPY_ENV] == HC.HOP_COPY_REASON
    assert HV.HOP_COPY_ENV not in os.environ
    monkeypatch.setenv(HV.HOP_COPY_ENV, "mine")
    with HC.hop_copy_env():
        pass
    assert os.environ[HV.HOP_COPY_ENV] == "mine"


def test_peak_sampler(tmp_path):
    (tmp_path / "f").write_bytes(b"x" * 4096)
    s = HC.PeakSampler(tmp_path, 0.01)
    s.start()
    assert s.stop() >= 4


def test_validate_refusals(tmp_path):
    ap = HC.build_parser("t")
    for extra, msg in [(["--port", "6333"], "serving port"), (["--chain", ","], "empty")]:
        a = _args(tmp_path, *extra, "--dry-run")
        with pytest.raises(SystemExit):
            HC.validate(a, ap)
    a = HC.build_parser("t").parse_args(["--collection", "c", "--work-dir", str(tmp_path), "--mode", "inplace",
                                         "--source-storage", str(tmp_path), "--dry-run"])
    with pytest.raises(SystemExit):   # no irreversible ack
        HC.validate(a, ap)
    a = HC.build_parser("t").parse_args(["--collection", "c", "--work-dir", str(tmp_path), "--dry-run"])
    with pytest.raises(SystemExit):   # snapshot mode without a snapshot
        HC.validate(a, ap)


class FakeSh:
    def __init__(self):
        self.calls = []

    def __call__(self, argv, check=True):
        self.calls.append(list(argv))
        out = "aw-qdrant-hop-v1-13-6\n" if argv[:2] == ["docker", "ps"] else ""
        return subprocess.CompletedProcess(argv, 0, out, "")


def test_dry_run_prints_docker_and_touches_nothing(tmp_path, capsys):
    a = _args(tmp_path, "--dry-run")
    sh = FakeSh()
    rc = HC.Chain(a, sh=sh, say=lambda m: None).go()
    out = capsys.readouterr().out
    assert rc == 0 and sh.calls == []
    assert out.count("DRY: docker pull") == 4 and "DRY: docker run" in out and "DRY: verify hop3:v1.18.3" in out
    assert not (tmp_path / "w").exists()


def test_wait_green_and_red(tmp_path):
    a = _args(tmp_path)
    seq = iter(["unreadable", "yellow", "green"])
    c = HC.Chain(a, sh=FakeSh(), http=lambda *x, **k: {"result": {"status": next(seq)}}, sleep=lambda s: None,
                 say=lambda m: None)
    c.wait_green()
    c = HC.Chain(a, sh=FakeSh(), http=lambda *x, **k: {"result": {"status": "red"}}, sleep=lambda s: None,
                 say=lambda m: None)
    c.cur = "aw-qdrant-hop-v1-13-6"
    with pytest.raises(HC.HopError, match="RED"):
        c.wait_green()


def test_wait_ready_fails_fast_when_container_died(tmp_path):
    a = _args(tmp_path)

    def down(*x, **k):
        raise HV.QdrantError("refused")
    c = HC.Chain(a, sh=FakeSh(), http=down, sleep=lambda s: None, say=lambda m: None)
    c.cur = "dead-one"
    with pytest.raises(HC.HopError, match="exited during startup"):
        c.wait_ready()


def test_restore_ok_and_failure(tmp_path):
    a = _args(tmp_path)
    seen = []
    ok = HC.Chain(a, sh=FakeSh(), say=lambda m: None,
                  http=lambda url, method="GET", body=None, timeout=1: seen.append((url, method, body)) or {"result": True})
    ok.restore("/qdrant/snapshots/_hopin/s.snapshot")
    assert seen[0][1] == "PUT" and seen[0][0].endswith("/collections/aw_x/snapshots/recover?wait=true")
    assert seen[0][2] == {"location": "file:///qdrant/snapshots/_hopin/s.snapshot"}
    bad = HC.Chain(a, sh=FakeSh(), say=lambda m: None, http=lambda *x, **k: {"result": False})
    with pytest.raises(HC.HopError, match="recover FAILED"):
        bad.restore("/x")


def test_create_snapshot(tmp_path):
    a = _args(tmp_path)
    c = HC.Chain(a, sh=FakeSh(), say=lambda m: None, http=lambda *x, **k: {"result": {"name": "n.snapshot"}})
    assert c.create_snapshot() == "n.snapshot"


def test_stop_is_graceful_and_logs(tmp_path):
    a = _args(tmp_path)
    (tmp_path / "w" / "logs").mkdir(parents=True)
    sh = FakeSh()
    c = HC.Chain(a, sh=sh, say=lambda m: None)
    c.cur = "aw-qdrant-hop-v1-13-6"
    c.stop()
    assert ["docker", "stop", "-t", "300", "aw-qdrant-hop-v1-13-6"] in sh.calls and c.cur is None


def test_ensure_baseline_copies_file_and_reuses(tmp_path):
    base = tmp_path / "b.json"
    base.write_text(json.dumps({"exact_count": 3, "server_version": "1", "queries": {}, "content": {"sha256": "ab" * 32}}))
    a = _args(tmp_path, "--baseline", str(base))
    c = HC.Chain(a, sh=FakeSh(), say=lambda m: None)
    c.work.mkdir(parents=True)
    c.ensure_baseline()
    assert (c.work / "baseline.json").read_text() == base.read_text()
    c.ensure_baseline()   # resume: reuse


def test_inplace_without_baseline_refused(tmp_path):
    a = _args(tmp_path, mode="inplace")
    c = HC.Chain(a, sh=FakeSh(), say=lambda m: None)
    with pytest.raises(HC.HopError, match="needs --baseline"):
        c.ensure_baseline()


def test_verify_hop_uses_hop_copy_env_and_fails_on_diff(tmp_path, monkeypatch):
    a = _args(tmp_path)
    c = HC.Chain(a, sh=FakeSh(), say=lambda m: None)
    c.work.mkdir(parents=True)
    (c.work / "baseline.json").write_text(json.dumps({"queries": {"q": 1}, "sample_ids": [1]}))
    envs = []
    monkeypatch.setattr(HV, "fingerprint", lambda *x, **k: envs.append(os.environ.get(HV.HOP_COPY_ENV)) or {})
    monkeypatch.setattr(HV, "compare", lambda *x, **k: 1)
    with pytest.raises(HC.HopError, match="VERIFY FAILED"):
        c.verify_hop("hop1:v")
    assert envs == [HC.HOP_COPY_REASON]
    monkeypatch.setattr(HV, "compare", lambda *x, **k: 0)
    c.verify_hop("hop1:v")


def test_snapshot_chain_resumes_from_markers(tmp_path):
    a = _args(tmp_path, "--chain", "v1.13.6")
    c = HC.Chain(a, sh=FakeSh(), say=lambda m: None)
    c.state.mkdir(parents=True)
    out = tmp_path / "out.snapshot"
    out.write_text("s")
    (c.state / "hop1-v1.13.6.done").write_text(json.dumps({"out_snapshot": str(out)}))
    assert c.chain_snapshot() == out
    assert (c.work / "FINAL_SNAPSHOT").read_text().strip() == str(out)
    out.unlink()
    with pytest.raises(HC.HopError, match="is gone"):
        c.chain_snapshot()


def test_door_routes_chain_with_prog(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["ops", "qdrant-hop-verify", "chain", "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("usage: cyt ops qdrant-hop-verify chain") and "Operator runbook" in out
