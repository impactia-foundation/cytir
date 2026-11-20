"""The suite is hermetic against an ambient crawler identity under either spelling (task 2991b).

A reader who follows the README exports CYTIR_OPERATOR_*; many tests set AW_OPERATOR_* themselves, and the
env-name mirror refuses two different values (EnvNameConflict). The root conftest removes the ambient pair
before any test runs, and `tests/conftest.py` marks the tests that share the throwaway Qdrant."""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest
import yaml

APP = pathlib.Path(__file__).resolve().parents[1]


def test_a_session_started_with_cytir_identity_runs_aw_setting_tests_clean():
    env = {**os.environ, "CYTIR_OPERATOR_URL": "https://example.org/ambient",
           "CYTIR_OPERATOR_CONTACT": "https://example.org/ambient/issues"}
    for k in ("AW_OPERATOR_URL", "AW_OPERATOR_CONTACT", "AW_TEST_QDRANT_URL"):
        env.pop(k, None)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist",
         str(APP / "tests" / "test_user_agent.py"), str(APP / "tests" / "test_aw_operator_identity.py")],
        cwd=APP, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
    assert "EnvNameConflict" not in r.stdout + r.stderr


def test_the_qdrant_marker_follows_the_module_text():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_tconf", APP / "tests" / "conftest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._QDRANT_SOURCE.search("from throwaway_qdrant import x")
    assert mod._QDRANT_SOURCE.search("url = H.qdrant_url()")
    assert not mod._QDRANT_SOURCE.search("def test_plain(): pass")


def test_the_workflow_runs_the_qdrant_tests_in_one_process():
    wf = APP / ".github" / "workflows" / "ci.yml"
    if not wf.is_file():
        pytest.skip("no workflow beside the app")
    steps = yaml.safe_load(wf.read_text())["jobs"]["test"]["steps"]
    runs = [str(s.get("run", "")) for s in steps]
    parallel = [r for r in runs if "pytest" in r and "-n 4" in r]
    serial = [r for r in runs if "pytest" in r and "-m qdrant" in r]
    assert len(parallel) == 1 and '-m "not qdrant"' in parallel[0], parallel
    assert len(serial) == 1 and "-n " not in serial[0] and "no:xdist" in serial[0], serial
