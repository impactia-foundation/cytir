"""Task 2494: `cyt ops registry-census` is the door to cytir.ops.registry_census."""
import subprocess
import sys

from cytir import cli
from cytir.ops import registry_census


def test_registry_census_verb_dispatches(monkeypatch):
    seen = {}

    def fake_main(a=None, prog=None):
        seen.update(argv=a, prog=prog)
        return 3
    monkeypatch.setattr(registry_census, "main", fake_main)
    assert cli._ops_main(["registry-census", "merge", "--json", "x"]) == 3
    assert seen == {"argv": ["merge", "--json", "x"], "prog": "cyt ops registry-census"}


def test_registry_census_help_runs():
    r = subprocess.run([sys.executable, "-m", "cytir.cli", "ops", "registry-census", "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "cyt ops registry-census" in r.stdout and "merge" in r.stdout
