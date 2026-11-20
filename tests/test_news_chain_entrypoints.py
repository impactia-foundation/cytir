"""The daily news chain's scripts start from a clean environment (task 676).

The pack step runs `python cytir/sources/families/news.py ...` as a SCRIPT from the release
directory, with no PYTHONPATH: `sys.path[0]` is the script's own directory, so every
`import cytir...` must be preceded by the script putting the repo root on the path. The r20
release cut (2026-09-30) found `news.py` failing `ModuleNotFoundError: No module named
'cytir'` in the pack step, after #437 moved its `upsert_supersede` import into the package.
A unit test that imports the module from pytest's own path cannot see that; this one starts
each script the way the chain does (cwd elsewhere, PYTHONPATH unset) and asks for `--help`.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
CHAIN = (
    "cytir/sources/families/news.py",
    "cytir/sources/families/news_tier2_apply.py",
)


@pytest.mark.parametrize("script", CHAIN)
def test_the_script_starts_with_no_pythonpath_from_another_directory(script, tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONDONTWRITEBYTECODE="1", AW_OPERATOR_URL="https://websear.ch",
               AW_OPERATOR_CONTACT="bender@impactia.org")
    out = subprocess.run([sys.executable, str(REPO / script), "--help"], cwd=tmp_path,
                         env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, (script, out.stderr[-400:])
    assert "ModuleNotFoundError" not in out.stderr



def test_the_source_evidence_sealer_starts_as_a_module_from_the_release_dir(tmp_path):
    """The adapter starts it `<langgate python> -m cytir.news.seal_source_evidence` with the release
    on PYTHONPATH (`Env.module_env`), from any working directory (task 1949)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO))
    out = subprocess.run([sys.executable, "-m", "cytir.news.seal_source_evidence", "--help"],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-400:]
    assert "ModuleNotFoundError" not in out.stderr
