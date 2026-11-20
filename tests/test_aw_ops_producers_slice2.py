"""Doors of the producers moved into cytir.ops (task 1082): each verb resolves, parses `--help`, and a producer's
output gets its provenance sidecar. The modules' logic is covered by their own suites."""
from __future__ import annotations

import json

import pytest

from cytir import cli
from cytir.ops import _provenance

VERBS = ["assembly-guards", "build-assembly-plan", "fold-assembly-claim-fragments", "rii-dump-basis-backfill",
         "extract-agentic-eval"]


@pytest.mark.parametrize("verb", VERBS)
def test_the_door_resolves_and_prints_help(verb, capsys):
    assert verb in cli._PRODUCER_VERBS
    with pytest.raises(SystemExit) as exc:
        cli._ops_main([verb, "--help"])
    assert exc.value.code == 0
    assert f"cyt ops {verb}" in capsys.readouterr().out


def test_arg_inputs_names_only_existing_input_files(tmp_path):
    import argparse
    a, out = tmp_path / "in.json", tmp_path / "out.json"
    a.write_text("{}")
    out.write_text("{}")
    ns = argparse.Namespace(plan=str(a), out=str(out), missing=str(tmp_path / "nope"), n=3)
    assert _provenance.arg_inputs(ns, out) == {"plan": str(a)}
    side = _provenance.write_provenance(out, "build-assembly-plan", _provenance.arg_inputs(ns, out))
    doc = json.loads(side.read_text())
    assert doc["producer"] == "aw ops build-assembly-plan" and list(doc["inputs"]) == ["plan"]
