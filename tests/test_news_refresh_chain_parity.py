#!/usr/bin/env python3
"""Task 1989 -- live parity: `news_refresh_{stage,delta}.sh` vs `cytir.news.refresh_chain`, same layout.

The shell script runs first against the recorder interpreters, its receipts are read as bytes and
cleared, then the Python port runs on the SAME paths. Calls, stdout and every receipt byte must be
equal. (`tests/fixtures/news_refresh_chain/` holds the recorded 'before' for when the scripts go.)
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

from tests.helpers.repo_files import repo_file_or_skip

APP = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / "tests"))
FIX = APP / "tests" / "fixtures" / "news_refresh_chain"

import news_refresh_chain_harness as H  # noqa: E402


@pytest.mark.parametrize("kind", ["stage", "delta"])
def test_python_port_equals_the_shell_script_byte_for_byte(kind, tmp_path):
    script = repo_file_or_skip(f"deploy/scripts/news_refresh_{kind}.sh")
    twin = H.Twin() if kind == "delta" else None
    try:
        sc = H.Scenario(kind, tmp_path, twin)
        rc, sh_out = sc.run_script(script)
        assert rc == 0, sh_out
        sh_snap, sh_calls = sc.snapshot(), sc.comparable_calls()
        sc.reset()
        sc.prepare()
        rc, py_out = sc.run_python()
        assert rc == 0, py_out
        assert sc.snapshot() == sh_snap
        assert sc.comparable_calls() == sh_calls
        assert py_out.strip().splitlines()[-1] == sh_out.strip().splitlines()[-1]
    finally:
        if twin:
            twin.close()


def _before(kind):
    return json.loads((FIX / f"{kind}_before.json").read_text())


def _as_recorded(calls, before):
    """task 2600: the host table moved to data/; the recorded run (sealed bytes) names its old home, read from it."""
    old = next(a for c in before["calls"] for a in c["argv"] if a.endswith("/news_host_meta.json"))
    return json.loads(json.dumps(calls).replace("<REPO>/data/news-host-meta.v1.json", old))


def test_stage_equals_the_recorded_shell_run(tmp_path):
    sc = H.Scenario("stage", tmp_path)
    rc, text = sc.run_python()
    assert rc == 0, text
    before = _before("stage")
    assert _as_recorded(sc.comparable_calls(), before) == before["calls"]
    assert sc.digests(sc.snapshot()) == before["digests"]
    assert text.strip().splitlines()[-1] == before["final_line"]


def test_delta_equals_the_recorded_shell_run(tmp_path):
    twin = H.Twin()
    try:
        sc = H.Scenario("delta", tmp_path, twin)
        rc, text = sc.run_python()
        assert rc == 0, text
        before = _before("delta")
        assert sc.comparable_calls() == before["calls"]
        assert sc.digests(sc.snapshot()) == before["digests"]
        assert sc.normalize(text.strip().splitlines()[-1]) == before["final_line"]
    finally:
        twin.close()
