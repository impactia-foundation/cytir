"""Task 2357 (S5 W6): the operator tools are reachable from the `cyt ops` verb table (`_PRODUCER_VERBS`)."""
import importlib
import subprocess
import sys

import pytest

from cytir import cli

VERBS = {"docs-roster": "cytir.ops.docs_roster", "extras-smoke": "cytir.ops.extras_smoke",
         "family-census": "cytir.ops.family_census", "health-host-vet": "cytir.ops.health_host_vet",
         "tombstones-index-text": "cytir.ops.index_text_tombstones", "tombstones-spam": "cytir.ops.spam_tombstones"}
HELPFUL = [v for v in VERBS if v != "extras-smoke"]      # extras_smoke is hand-parsed: no argparse --help


def test_table_names_the_modules():
    for verb, module in VERBS.items():
        assert cli._PRODUCER_VERBS[verb] == module


@pytest.mark.parametrize("verb", list(VERBS))
def test_verb_dispatches_and_propagates_exit(monkeypatch, verb):
    seen = {}

    def fake_main(a=None, prog=None, **kw):
        seen.update(argv=a, prog=prog)
        return 7

    monkeypatch.setattr(importlib.import_module(VERBS[verb]), "main", fake_main)
    assert cli._ops_main([verb, "--x"]) == 7
    assert seen == {"argv": ["--x"], "prog": f"cyt ops {verb}"}


@pytest.mark.parametrize("verb", HELPFUL)
def test_help_is_the_modules_own_from_the_real_entry_point(verb):
    out = subprocess.run([sys.executable, "-m", "cytir.cli", "ops", verb, "--help"],
                         capture_output=True, text=True, timeout=60).stdout
    assert f"usage: cyt ops {verb}" in out
