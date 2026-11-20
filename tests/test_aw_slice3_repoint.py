#!/usr/bin/env python3
"""Task 696 slice 3 (2/2) -- `cytir/` stops naming `pipeline/` paths where it only repoints.

Characterization of what did NOT change, plus the two script-start defects:

  * `cytir/core/common.py` and `cytir/sources/registry.py` failed with `No module named 'cytir'` when
    started by path from a foreign cwd (they import `aw.*` at module level and never put the
    repo root on `sys.path`). Both now start clean.
  * The serving-binding refusal text names the package function, not a `pipeline/` path; the
    refusal conditions themselves are unchanged.
  * `content_pack_arm` loads the release's own `cytir/core/serving_config.py`: the path it joins
    is the same file as before, and the file exists.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

SCRIPT_START = ["cytir/core/common.py", "cytir/sources/registry.py"]


@pytest.mark.parametrize("rel", SCRIPT_START)
def test_started_by_path_from_a_foreign_cwd(rel, tmp_path):
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    r = subprocess.run([sys.executable, str(REPO / rel)], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    assert "No module named" not in r.stderr


def test_serving_binding_refusal_names_the_package_not_a_pipeline_path():
    from cytir.core import serving_config as SC

    name = SC.FLIP_SERVING_CONFIG
    binding = {"schema": SC.GATE_BINDING_SCHEMA, "dense_rescore": None,
               "dense_oversampling": None, "dense_ignore_quant": True,
               "fusion_weights_sha256": "0" * 64}
    problems = SC.gate_serving_binding_problems(binding)
    assert problems, "a binding that differs from production must be refused"
    text = " | ".join(problems)
    assert "pipeline/" not in text
    assert f"cytir.core.serving_config.flip_deploy_env, {name}" in text


def test_content_pack_arm_loads_the_releases_own_serving_config():
    src = (REPO / "cytir/gates/content_pack_arm.py").read_text()
    assert 'release_dir / "cytir/core/serving_config.py"' in src
    assert (REPO / "cytir/core/serving_config.py").is_file()


def test_news_census_hn_label_is_not_a_pipeline_path():
    from cytir.ops import news_census as NC

    roster = NC.declared_roster()
    labels = roster["news.ycombinator.com"]
    assert labels == ["HN arms (hn_freshness_ingest + hn_discussion_daily)"]
