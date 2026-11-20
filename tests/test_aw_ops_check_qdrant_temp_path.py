"""`cyt ops check-qdrant-temp-path` (the P18 monitor): checks the EFFECT, not the declaration.

The running qdrant does not echo its temp path, so the monitor checks what qdrant DID: the env
var, a mount with room behind it, and the absence of `<storage>/tmp`. Any one alone can pass while
the setting is ineffective, so each failure is exercised on its own. No docker: `docker()` and the
disk reading are stubbed.
"""
from __future__ import annotations

import inspect
import json
from collections import namedtuple

import pytest

from cytir.ops import check_qdrant_temp_path as P18

_Usage = namedtuple("_Usage", "total used free")
STORAGE = {"Destination": "/qdrant/storage", "Source": "/srv/qdrant"}
TEMP = {"Destination": "/qdrant/snapshot-tmp", "Source": "/srv/scratch"}


def _stub(monkeypatch, *, env, mounts, storage_tmp="absent", free_gb=500.0):
    monkeypatch.setattr(P18.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(P18.shutil, "disk_usage", lambda path: _Usage(0, 0, free_gb * 1e9))

    def docker(*args, timeout=30):
        if args[:2] == ("inspect", "-f") and "Env" in args[2]:
            return json.dumps(env)
        if args[:2] == ("inspect", "-f") and "Mounts" in args[2]:
            return json.dumps(mounts)
        if args[0] == "exec":
            return storage_tmp
        raise AssertionError(args)
    monkeypatch.setattr(P18, "docker", docker)


def _run(capsys, *extra):
    rc = P18.main(["--json", *extra])
    return rc, json.loads(capsys.readouterr().out)


def test_all_three_conditions_hold_passes(monkeypatch, capsys):
    _stub(monkeypatch, env=["QDRANT__STORAGE__TEMP_PATH=/qdrant/snapshot-tmp"], mounts=[STORAGE, TEMP])
    rc, rep = _run(capsys)
    assert rc == 0 and rep["ok"] and rep["temp_path_host"] == "/srv/scratch"


def test_unset_env_fails(monkeypatch, capsys):
    _stub(monkeypatch, env=["PATH=/bin"], mounts=[STORAGE, TEMP])
    rc, rep = _run(capsys)
    assert rc == 2 and any("is NOT set on" in p for p in rep["problems"])


def test_env_naming_an_unmounted_path_fails(monkeypatch, capsys):
    _stub(monkeypatch, env=["QDRANT__STORAGE__TEMP_PATH=/inside/image"], mounts=[STORAGE])
    rc, rep = _run(capsys)
    assert rc == 2 and any("not backed by any mount" in p for p in rep["problems"])


def test_a_mount_without_room_fails(monkeypatch, capsys):
    _stub(monkeypatch, env=["QDRANT__STORAGE__TEMP_PATH=/qdrant/snapshot-tmp"], mounts=[STORAGE, TEMP],
          free_gb=10.0)
    rc, rep = _run(capsys, "--min-free-gb", "150")
    assert rc == 2 and any("below the 150.0 GB" in p for p in rep["problems"])


def test_storage_tmp_present_fails_even_with_the_setting(monkeypatch, capsys):
    _stub(monkeypatch, env=["QDRANT__STORAGE__TEMP_PATH=/qdrant/snapshot-tmp"], mounts=[STORAGE, TEMP],
          storage_tmp="PRESENT")
    rc, rep = _run(capsys)
    assert rc == 2 and any("EXISTS" in p for p in rep["problems"])


def test_no_docker_cli_cannot_tell(monkeypatch, capsys):
    monkeypatch.setattr(P18.shutil, "which", lambda name: None)
    assert P18.main([]) == 3


def test_the_module_says_why_a_config_read_is_not_available():
    src = inspect.getsource(P18)
    assert "QDRANT__STORAGE__TEMP_PATH" in src and "/qdrant/storage/tmp" in src
    assert "does not echo" in src


@pytest.mark.parametrize("argv", [["ops", "check-qdrant-temp-path", "--help"]])
def test_the_door_reaches_the_monitor(argv, capsys):
    from cytir import cli
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 0 and "--min-free-gb" in capsys.readouterr().out
