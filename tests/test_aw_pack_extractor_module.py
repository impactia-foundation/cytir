"""A pack's `extractor:` that names a package file runs as a module (task 730, slice 7).

A pack's `extractor:` that names `cytir/sources/families/fedlex.py` (the adapter module itself) is
started by the runner as `python -m cytir.sources.families.fedlex` with the checkout on PYTHONPATH, so
the line works from any working directory and the script directory is not put first on
`sys.path`.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import types

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cytir.sources import runner  # noqa: E402


def test_package_module_maps_a_package_path_only():
    assert runner._package_module("cytir/sources/families/fedlex.py") == "cytir.sources.families.fedlex"
    assert runner._package_module("scripts/x.py") is None
    assert runner._package_module("cytir/sources/families") is None
    assert runner._package_module("other/aw/x.py") is None


def _plan_naming(extractor):
    """The fields `run_via_extractor` reads (no pack file: the cut tree holds no `pipeline/`)."""
    return types.SimpleNamespace(extractor=extractor, pack_id="p", route_id="r", config_ref=None)


def test_runner_starts_a_package_extractor_as_a_module():
    args = argparse.Namespace(extractor_args=["--help"], dry_run=True, out_dir="x", limit=0,
                              budget=0, no_l0=False)
    got = runner.run_via_extractor(_plan_naming("cytir/sources/families/fedlex.py"), args=args)
    assert got["cmd"][1:3] == ["-m", "cytir.sources.families.fedlex"]
    assert got["cmd"][3:] == ["--help"]


def test_the_module_starts_from_a_foreign_working_directory(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    done = subprocess.run([sys.executable, "-m", "cytir.sources.families.fedlex", "--help"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "--eli" in done.stdout
