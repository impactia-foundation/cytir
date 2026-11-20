"""Task 2489: `cyt gate qhead` is the door to the ADR-062 byte-identical AW_QHEAD gate."""
import subprocess
import sys

from cytir import cli
from cytir.gates import gate_qhead


def test_gate_qhead_verb_dispatches(monkeypatch):
    seen = {}

    def fake_main(a=None, prog=None):
        seen.update(argv=a, prog=prog)
        return 7
    monkeypatch.setattr(gate_qhead, "main", fake_main)
    assert cli._gate_main(["qhead", "--out", "x"]) == 7
    assert seen == {"argv": ["--out", "x"], "prog": "cyt gate qhead"}


def test_gate_qhead_help_runs():
    r = subprocess.run([sys.executable, "-m", "cytir.cli", "gate", "qhead", "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "--compare" in r.stdout and "cyt gate qhead" in r.stdout

