"""Task 2913: `cyt ops writer-locks --check` reads the registry lock records and never writes.

Every record here is made by the REAL `write_lock.acquire` (the conftest pins an isolated local registry),
so a change of the record format breaks these tests."""
from __future__ import annotations

import ast
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import types

import pytest

from cytir import cli
from cytir.ops import write_lock as WL
from cytir.ops import writer_locks as W

OLD = "2020-01-01T00:00:00Z"


def _at(delta: dt.timedelta) -> str:
    """A record time `delta` from now, in the registry's UTC format."""
    return (dt.datetime.now(dt.timezone.utc) + delta).strftime("%Y-%m-%dT%H:%M:%SZ")


def _take(monkeypatch, name="PROD-WRITE-LOCK", *, pid, at=None, host=None):
    if at:
        monkeypatch.setattr(WL, "now", lambda: at)
    if host:
        monkeypatch.setattr(WL, "socket", types.SimpleNamespace(gethostname=lambda: host))
    WL.acquire(name, "641", "pack-promotion-queue", pid=pid)


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def _run(capsys, *args):
    rc = cli._ops_main(["writer-locks", "--check", *args])
    return rc, capsys.readouterr().out


def test_no_lock_is_rc0_and_names_the_unchecked_token_lock(capsys):
    rc, out = _run(capsys)
    assert rc == 0
    assert "TWIN-WRITE-LOCK: free" in out and "PROD-WRITE-LOCK: free" in out
    assert "not checked" in out


def test_fresh_lock_with_a_live_pid_is_rc0(monkeypatch, capsys):
    _take(monkeypatch, pid=os.getpid())
    rc, out = _run(capsys)
    assert rc == 0
    assert "held and live" in out


def test_old_lock_with_a_dead_pid_is_stale_rc1_with_the_manual_steps(monkeypatch, capsys):
    _take(monkeypatch, pid=_dead_pid(), at=OLD)
    rc, out = _run(capsys)
    assert rc == 1
    assert "PROD-WRITE-LOCK: STALE" in out and "(pid dead)" in out
    assert "twin == prod" in out and "LOCK-RELEASES.log" in out
    assert len(W.STALE_STEPS) <= 4


def test_young_lock_with_a_dead_pid_is_not_stale_yet(monkeypatch, capsys):
    _take(monkeypatch, pid=_dead_pid())
    rc, out = _run(capsys)
    assert rc == 0 and "younger than the limit" in out


def test_old_lock_with_a_live_pid_is_rc0_held_and_live(monkeypatch, capsys):
    # The record must be AFTER this test process started (fix 2), and old under --max-age-min 0.
    _take(monkeypatch, pid=os.getpid(), at=_at(dt.timedelta(seconds=-1)))
    rc, out = _run(capsys, "--max-age-min", "0")
    assert rc == 0
    assert "held and live" in out and "STALE" not in out


def test_old_lock_from_another_host_is_stale_by_age_with_unknown_liveness(monkeypatch, capsys):
    _take(monkeypatch, pid=os.getpid(), at=OLD, host="some-other-box")
    rc, out = _run(capsys)
    assert rc == 1
    assert "(pid unknown)" in out and "host=some-other-box" in out


def test_max_age_is_honoured(monkeypatch, capsys):
    _take(monkeypatch, pid=_dead_pid(), at=WL.now())
    rc, _ = _run(capsys, "--max-age-min", "0")
    assert rc == 1


def test_alive_pid_that_is_not_python_or_aw_is_not_the_holder_so_stale(monkeypatch, capsys):
    monkeypatch.setattr(W, "_cmdline", lambda pid: "/usr/sbin/cron -f")
    _take(monkeypatch, pid=os.getpid(), at=_at(dt.timedelta(seconds=-1)))
    rc, out = _run(capsys, "--max-age-min", "0")
    assert rc == 1 and "STALE" in out and "pid reused (command line is not python or aw)" in out


def test_old_lock_whose_live_pid_is_not_python_or_aw_is_stale(monkeypatch, tmp_path, capsys):
    # argv0 `awk`: contains the substring "aw" but is not a holder (fix 1). bash is started under
    # that argv0 (a real executable, never a coreutils multicall binary that refuses a foreign
    # name, as uutils `sleep` does on hosted runners); /proc/<pid>/cmdline then reads "awk ...".
    bash = shutil.which("bash")
    child = subprocess.Popen(["awk", "-c", "read x"], executable=bash, stdin=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while (W._cmdline(child.pid) or "").strip() != "awk -c read x" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child.poll() is None, "the helper process died before the check"
        assert W._cmdline(child.pid).strip() == "awk -c read x"
        _take(monkeypatch, pid=child.pid, at=_at(dt.timedelta(seconds=-1)))
        rc, out = _run(capsys, "--max-age-min", "0")
    finally:
        child.kill()
        child.wait()
        child.stdin.close()
    assert rc == 1 and "PROD-WRITE-LOCK: STALE" in out and "pid reused" in out


def test_old_lock_whose_python_pid_started_after_the_record_is_stale(monkeypatch, capsys):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _take(monkeypatch, pid=child.pid, at=_at(dt.timedelta(hours=-3)))
        rc, out = _run(capsys)
    finally:
        child.kill()
        child.wait()
    assert rc == 1 and "process started after the lock was taken" in out


def test_record_in_the_future_is_unreadable_rc2(monkeypatch, capsys):
    _take(monkeypatch, pid=os.getpid(), at=_at(dt.timedelta(days=365)))
    rc, out = _run(capsys)
    assert rc == 2 and "at= is in the future" in out


def test_superscript_two_pid_is_non_numeric_not_a_traceback(monkeypatch, capsys):
    _take(monkeypatch, pid="²", at=OLD)
    rc, out = _run(capsys)
    assert rc == 1 and "(pid unknown)" in out and "Traceback" not in out


def test_non_numeric_pid_is_unknown_liveness(monkeypatch, capsys):
    _take(monkeypatch, pid="n/a-multiprocess", at=OLD)
    rc, out = _run(capsys)
    assert rc == 1 and "(pid unknown)" in out


@pytest.mark.parametrize("text", ["task=1 owner=x host=h at=2020-01-01T00:00:00Z",      # no pid
                                  "garbage with no fields",
                                  "task=1 host=h pid=1 at=yesterday"])
def test_corrupt_record_is_rc2(text, capsys):
    pathlib.Path(WL.effective_location()[1], "TWIN-WRITE-LOCK").write_text(text + "\n")
    rc, out = _run(capsys)
    assert rc == 2 and "ERROR" in out


def test_empty_record_and_missing_directory_are_rc2(monkeypatch, capsys):
    directory = pathlib.Path(WL.effective_location()[1])
    (directory / "PROD-WRITE-LOCK").write_text("")
    assert _run(capsys)[0] == 2
    monkeypatch.setitem(WL.__dict__, "LOCK_DIR", str(directory / "nope"))
    assert _run(capsys)[0] == 2


def test_a_registry_on_another_host_is_refused_not_dialled(monkeypatch, capsys):
    monkeypatch.setitem(WL.__dict__, "LOCK_HOST", "elsewhere")
    monkeypatch.setattr(WL, "ssh", lambda *a, **k: pytest.fail("the check must not open a connection"))
    rc, out = _run(capsys)
    assert rc == 2 and "elsewhere" in out


def test_json_is_one_object_with_the_same_facts(monkeypatch, capsys):
    _take(monkeypatch, pid=_dead_pid(), at=OLD)
    rc, out = _run(capsys, "--json")
    doc = json.loads(out)
    assert rc == doc["rc"] == 1
    prod = next(r for r in doc["locks"] if r["lock"] == "PROD-WRITE-LOCK")
    assert prod["status"] == "STALE" and prod["liveness"] == "dead" and prod["pid"].isdigit()
    assert doc["not_checked"]


def test_the_module_cannot_write_delete_rename_or_dial():
    """Read-only by construction: no write-mode open, no unlink / rename / remove, no ssh or socket call."""
    tree = ast.parse(pathlib.Path(W.__file__).read_text())
    banned = {"unlink", "rename", "remove", "rmdir", "rmtree", "write_text", "write_bytes", "touch",
              "mkdir", "chmod", "ssh", "acquire", "release", "run", "Popen", "system", "connect", "urlopen"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            assert name not in banned, f"read-only module calls {name} at line {node.lineno}"
            if name == "open":
                modes = [a.value for a in node.args[1:2] if isinstance(a, ast.Constant)]
                modes += [k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
                assert all(set(str(m)) <= set("rbt") for m in modes), f"write-mode open at line {node.lineno}"
    imports = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)}
    assert not imports & {"subprocess", "shutil", "urllib", "requests", "http"}


def test_the_verb_is_registered_and_help_works():
    assert cli._PRODUCER_VERBS["writer-locks"] == "cytir.ops.writer_locks"
    out = subprocess.run([sys.executable, "-m", "cytir.cli", "ops", "writer-locks", "--help"],
                         capture_output=True, text=True, timeout=60).stdout
    assert "usage: cyt ops writer-locks" in out
