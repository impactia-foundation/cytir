#!/usr/bin/env python3
"""Task 677 (task 643 S5): `cyt source {run,plan}` is the one door; `run_source.py`
and the old source-runner script are forwarders onto `cytir.sources.runner`.

No network and no crawler identity: every check here is `--plan` or a refusal.
The end-to-end fedlex run (live, stranger-style) is recorded in
the task-677 doc, not here.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import subprocess
import sys

import pytest

from tests.helpers.repo_files import repo_file_or_skip

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cytir import cli  # noqa: E402

FEDLEX_PACK = repo_file_or_skip(pathlib.Path("sources") / "ch" / "fedlex.yaml")

ENV = {k: v for k, v in os.environ.items()
       if k not in ("AW_OPERATOR_URL", "AW_OPERATOR_CONTACT")}


def _run(*argv):
    return subprocess.run(list(argv), cwd=REPO, env=ENV, capture_output=True,
                          text=True, timeout=120)


def test_aw_source_plan_compiles_the_fedlex_pack_with_no_identity(capsys):
    assert cli.main(["source", "plan", "sources/ch/fedlex.yaml"]) == 0
    out = capsys.readouterr().out
    assert "ch-fedlex" in out and "route=source_owned_bulk_dump" in out
    assert "extractor=cytir/sources/families/fedlex.py" in out


def test_the_two_doors_print_the_same_plan():
    via_cli = _run(sys.executable, "-m", "cytir.cli", "source", "plan",
                   "sources/ch/fedlex.yaml")
    via_script = _run(sys.executable, "run_source.py", "sources/ch/fedlex.yaml",
                      "--plan")
    assert via_cli.returncode == via_script.returncode == 0
    assert via_cli.stdout == via_script.stdout


@pytest.mark.parametrize("argv", [[], ["frobnicate", "x"], ["--plan"]])
def test_aw_source_refuses_an_unknown_subverb(argv, capsys):
    assert cli.main(["source", *argv]) == 2
    assert "usage: cyt source {run,plan}" in capsys.readouterr().err


def test_the_runner_finds_the_repos_packs_and_data():
    from cytir.sources import pack_loader, runner
    assert pack_loader.REPO_ROOT == REPO == runner.REPO_ROOT
    assert pack_loader.PRESETS_PATH.is_file()
    assert (REPO / "sources" / "ch" / "fedlex.yaml").is_file()


def test_a_recorded_decline_is_refused_by_the_runner_before_any_fetch(monkeypatch):
    """The runner's consent preflight (step 1) still refuses through the moved
    modules: `consent_block.assert_crawlable` raises for a registry host.
    (Importing the resolver needs a crawler identity; CI sets one, a stranger's
    shell may not.)"""
    monkeypatch.setenv("AW_OPERATOR_URL", "https://example.org/about")
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "ops@example.org")
    from cytir.sources import runner
    with pytest.raises(Exception) as err:
        runner.resolve_consent("prestations.vd.ch", direct_crawl=True)
    assert "CONSENT-BLOCKED" in str(err.value)
