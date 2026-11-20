"""The b1 builder's command line: hold first, defaults unchanged, exit codes passed through."""
from __future__ import annotations

import pytest

from cytir.index import b1_cli as CLI
from cytir.index import twin_write_hold as TWH


def _spy(monkeypatch, order, status="DONE"):
    monkeypatch.setattr(TWH, "refuse_cli_write", lambda args, tool: order.append(("hold", tool)))
    monkeypatch.setattr(CLI, "run_slice", lambda args, log, corpus_dir: order.append(("slice", corpus_dir)))
    monkeypatch.setattr(CLI, "assert_ignition", lambda log: order.append(("ignition",)))

    def full(args, log, corpus_dir, resume_prefix):
        order.append(("full", corpus_dir, resume_prefix))
        return {}, status
    monkeypatch.setattr(CLI, "run_full", full)


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "work"))
    for k in ("AW_B1_CORPUS_DIR", "AW_B1_BUILD_DIR", "AW_B1_RESUME_PREFIX", "AW_HDD_ROOT", "AW_CORPUS_ROOT"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path / "work"


@pytest.mark.parametrize("mode", ["slice", "full"])
def test_the_hold_is_called_before_every_mode_dispatches(monkeypatch, scratch, mode):
    order = []
    _spy(monkeypatch, order)
    assert CLI.main(["--mode", mode]) == 0
    assert order[0][0] == "hold" and order[-1][0] == mode
    assert order[0][1] == "cytir/index/b1_cli.py"


@pytest.mark.parametrize("mode", ["slice", "full"])
def test_a_held_collection_refuses_before_any_mode_runs(monkeypatch, scratch, mode):
    order = []
    _spy(monkeypatch, order)

    def refuse(args, tool):
        raise SystemExit("REFUSED: held")
    monkeypatch.setattr(TWH, "refuse_cli_write", refuse)
    with pytest.raises(SystemExit):
        CLI.main(["--mode", mode])
    assert order == []


def test_the_default_mode_is_slice(monkeypatch, scratch):
    order = []
    _spy(monkeypatch, order)
    assert CLI.main([]) == 0
    assert [o[0] for o in order] == ["hold", "slice"]


@pytest.mark.parametrize("status,rc", [("DONE", 0), ("GATE_PASS", 0), ("STOP_COVERAGE", 3), ("OTHER", 0)])
def test_the_full_exit_codes(monkeypatch, scratch, status, rc):
    order = []
    _spy(monkeypatch, order, status=status)
    assert CLI.main(["--mode", "full"]) == rc


def test_the_full_mode_asserts_ignition_before_the_run(monkeypatch, scratch):
    order = []
    _spy(monkeypatch, order)
    CLI.main(["--mode", "full"])
    assert [o[0] for o in order] == ["hold", "ignition", "full"]


def test_the_directory_defaults_come_from_the_settings(monkeypatch, scratch):
    seen = {}
    monkeypatch.setattr(TWH, "refuse_cli_write", lambda args, tool: None)
    monkeypatch.setattr(CLI, "run_slice", lambda args, log, corpus_dir: seen.update(
        out=args.out_dir, cache=args.consent_cache, corpus=corpus_dir))
    CLI.main([])
    assert seen["out"] == str(scratch / "aw_b1_build")
    assert seen["cache"] == str(scratch / "aw_b1_build" / "consent_cache.json")
    assert seen["corpus"] == str(scratch / "aw-corpus")


def test_an_explicit_directory_wins_over_the_settings(monkeypatch, scratch, tmp_path):
    seen = {}
    monkeypatch.setattr(TWH, "refuse_cli_write", lambda args, tool: None)
    monkeypatch.setattr(CLI, "run_slice", lambda args, log, corpus_dir: seen.update(
        out=args.out_dir, cache=args.consent_cache))
    CLI.main(["--out-dir", str(tmp_path / "o"), "--consent-cache", str(tmp_path / "c.json")])
    assert seen == {"out": str(tmp_path / "o"), "cache": str(tmp_path / "c.json")}


def test_the_resume_prefix_default_names_this_module(monkeypatch, scratch):
    seen = {}
    monkeypatch.setattr(TWH, "refuse_cli_write", lambda args, tool: None)
    monkeypatch.setattr(CLI, "assert_ignition", lambda log: None)
    monkeypatch.setattr(CLI, "run_full", lambda args, log, corpus_dir, resume_prefix: (
        seen.update(prefix=resume_prefix), ({}, "DONE"))[1])
    CLI.main(["--mode", "full"])
    assert seen["prefix"].startswith("AW_FOLD_ACCENTS=1 ")
    assert seen["prefix"].endswith(" -m cytir.index.b1_cli")


def test_help_reads_no_setting(monkeypatch, capsys):
    monkeypatch.delenv("AW_WORK_ROOT", raising=False)
    with pytest.raises(SystemExit) as e:
        CLI.main(["--help"])
    assert e.value.code == 0
    assert "--mode" in capsys.readouterr().out
