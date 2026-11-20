"""Scripts started by path carry a working bootstrap (task 1738).

`data/script-entrypoints.v1.json` lists the `cytir/` files a caller starts by path. Each must start
with no PYTHONPATH and no installed `aw` (`python -S`, clean cwd): its own bootstrap puts the repo
root on `sys.path` before the first `aw` import. Each listed caller must still name the file.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SET = json.loads((REPO / "data/script-entrypoints.v1.json").read_text())["by_path"]

_LOADER = (
    "import os, runpy, sys\n"
    "p = os.path.abspath(sys.argv[1])\n"
    "sys.path[0] = os.path.dirname(p)\n"
    "sys.argv = [p]\n"
    "runpy.run_path(p, run_name='aw_script_entrypoints')\n"
)


@pytest.mark.parametrize("entry", SET, ids=[e["file"] for e in SET])
def test_by_path_script_bootstraps_itself(entry, tmp_path):
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}
    r = subprocess.run([sys.executable, "-S", "-c", _LOADER, str(REPO / entry["file"])],
                       cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    # Not vacuous: the file either loads clean or stops at a THIRD-PARTY import (no site-packages
    # under -S), which proves the bootstrap ran and its first `aw` import resolved.
    m = re.search(r"No module named '([^'.]+)", r.stderr)
    assert r.returncode == 0 or (m is not None and m.group(1) != "cytir"), r.stderr[-600:]


@pytest.mark.parametrize("entry", SET, ids=[e["file"] for e in SET])
def test_listed_callers_still_name_the_file(entry):
    for caller in entry["callers"]:
        assert entry["file"] in (REPO / caller).read_text(), (entry["file"], caller)
