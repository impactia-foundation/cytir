"""Task 1521 -- every `cytir.probe.*` module is importable, exposes `main`, and answers `--help`.

`deploy/scripts/probe-run.sh` runs each as `python3 -m cytir.probe.<name>`; this is the cheap guard that
the shell's six entry points still exist and start. No network, no prod.
"""
import importlib
import os
import pathlib
import subprocess
import sys

import pytest

import cytir.probe

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = sorted(p.stem for p in pathlib.Path(cytir.probe.__file__).parent.glob("*.py") if p.stem != "__init__")


def test_the_six_probe_modules_are_present():
    assert NAMES == ["empty_snippet_check", "latency_check", "run", "sample", "sample_sliceb", "score_sliceb"]


@pytest.mark.parametrize("name", NAMES)
def test_module_imports_and_exposes_main(name):
    assert callable(importlib.import_module(f"cytir.probe.{name}").main)


@pytest.mark.parametrize("name", NAMES)
def test_module_help_exits_zero(name):
    p = subprocess.run([sys.executable, "-m", f"cytir.probe.{name}", "--help"],
                       capture_output=True, text=True, cwd=REPO, timeout=60)
    assert p.returncode == 0, p.stderr
    assert "usage" in p.stdout.lower()
