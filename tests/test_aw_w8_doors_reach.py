"""Task 2383 (S5 W8): five operator tools are reachable from `cyt ops` / `cyt gate` with `main(argv, prog)`."""
import importlib
import subprocess
import sys

import pytest

from cytir import cli

OPS = {"http-drift": "cytir.ops.http_drift", "prod-env-run": "cytir.ops.prod_env_run"}
GATE = {("stamp-harness-weights",): ("cytir.gates.stamp_harness_weights", "cyt gate stamp-harness-weights"),
        ("legs", "prod"): ("cytir.gates.prod_legs", "cyt gate legs prod"),
        ("legs", "scratch"): ("cytir.gates.scratch_legs", "cyt gate legs scratch")}


def _fake(seen):
    def fake_main(a=None, prog=None, **kw):
        seen.update(argv=a, prog=prog)
        return 7
    return fake_main


@pytest.mark.parametrize("verb", list(OPS))
def test_ops_verb_dispatches(monkeypatch, verb):
    seen = {}
    assert cli._OPS_TOOL_VERBS[verb] == OPS[verb]
    monkeypatch.setattr(importlib.import_module(OPS[verb]), "main", _fake(seen))
    assert cli._ops_main([verb, "--x"]) == 7
    assert seen == {"argv": ["--x"], "prog": f"cyt ops {verb}"}


@pytest.mark.parametrize("path", list(GATE))
def test_gate_verb_dispatches(monkeypatch, path):
    module, prog = GATE[path]
    seen = {}
    monkeypatch.setattr(importlib.import_module(module), "main", _fake(seen))
    assert cli._gate_main([*path, "--x"]) == 7
    assert seen == {"argv": ["--x"], "prog": prog}


def test_legs_without_subverb_is_a_usage_error(capsys):
    assert cli._gate_main(["legs"]) == 2
    assert "cyt gate legs {prod,scratch}" in capsys.readouterr().err


@pytest.mark.parametrize("argv,usage", [
    (["ops", "http-drift"], "usage: cyt ops http-drift"),
    (["gate", "stamp-harness-weights"], "usage: cyt gate stamp-harness-weights"),
    (["gate", "legs", "prod"], "usage: cyt gate legs prod"),
    (["gate", "legs", "scratch"], "usage: cyt gate legs scratch"),
    (["ops", "prod-env-run"], "usage: cyt ops prod-env-run"),
])
def test_help_is_the_modules_own_from_the_real_entry_point(argv, usage):
    out = subprocess.run([sys.executable, "-m", "cytir.cli", *argv, "--help"],
                         capture_output=True, text=True, timeout=60).stdout
    assert usage in out
