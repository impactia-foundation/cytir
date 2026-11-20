"""`cyt source plan FAMILY` is `cyt source run FAMILY --dry-run` (task 1379)."""
import json
import socket

import pytest

from cytir import cli


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("plan FAMILY touched the network")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def _root(tmp_path):
    root = tmp_path / "dump"
    root.mkdir()
    (root / "Posts.xml").write_text("<posts/>")
    return root


def _run(capsys, argv):
    rc = cli.main(argv)
    return rc, capsys.readouterr()


def test_plan_family_prints_the_dry_run_plan_and_writes_nothing(tmp_path, capsys, no_network,
                                                                monkeypatch):
    root = _root(tmp_path)
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    rc, out = _run(capsys, ["source", "plan", "stackoverflow_qa", "--skip-consent",
                            "--opt", f"root={root}"])
    assert rc == 0, out.err
    plan = json.loads(out.out[out.out.index("{"):])
    assert plan["family_id"] == "stackoverflow_qa" and plan["dry_run"] is True
    assert plan["units"] == ["Posts.xml"]
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before    # nothing written


def test_plan_family_equals_run_family_dry_run(tmp_path, capsys, no_network):
    root = _root(tmp_path)
    args = ["stackoverflow_qa", "--skip-consent", "--opt", f"root={root}"]
    rc1, a = _run(capsys, ["source", "plan", *args])
    rc2, b = _run(capsys, ["source", "run", *args, "--dry-run", "--out-dir", str(tmp_path / "o")])
    assert rc1 == rc2 == 0 and a.out == b.out
    assert not (tmp_path / "o").exists()


def test_plan_unknown_id_still_refuses_with_the_pack_message(capsys):
    rc, out = _run(capsys, ["source", "plan", "no_such_pack_or_family_1379"])
    assert rc != 0
    assert "no pack with id" in (out.err + out.out)


def test_usage_says_pack_or_family_for_plan(capsys):
    assert cli.main(["source"]) == 2
    err = capsys.readouterr().err
    assert "{run,plan} PACK|FAMILY" in err and "plan FAMILY == run FAMILY --dry-run" in err
