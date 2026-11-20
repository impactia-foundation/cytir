"""The shipped CI workflow gives every job that fetches from a host a crawler identity (task 2991).

Cytir refuses to send an anonymous User-Agent, so a job that runs `fetch_models` (or any other
network fetch) without CYTIR_OPERATOR_URL and CYTIR_OPERATOR_CONTACT fails on a fresh public repo
or a fork. The values come from repository variables with a fallback derived from the repository."""
from __future__ import annotations

import pathlib

import pytest
import yaml

WORKFLOW = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
NAMES = ("CYTIR_OPERATOR_URL", "CYTIR_OPERATOR_CONTACT")
FETCHERS = ("fetch_models", "cyt source", "cyt embed models", "cyt consent")

pytestmark = pytest.mark.skipif(not WORKFLOW.is_file(), reason="no workflow beside the app")


def _doc() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _fetching_jobs(doc: dict) -> list[str]:
    out = []
    for name, job in doc["jobs"].items():
        runs = "\n".join(str(s.get("run", "")) for s in job.get("steps", []))
        if any(f in runs for f in FETCHERS):
            out.append(name)
    return out


def test_there_are_fetching_jobs():
    assert _fetching_jobs(_doc()), "the detector found no fetching job: update FETCHERS"


def test_every_fetching_job_has_both_operator_variables():
    doc = _doc()
    top = doc.get("env") or {}
    for name in _fetching_jobs(doc):
        env = {**top, **(doc["jobs"][name].get("env") or {})}
        for var in NAMES:
            assert env.get(var), f"job {name!r} fetches from a host but sets no {var}"


def test_the_values_fall_back_to_the_repository_and_never_to_a_mailbox():
    env = _doc()["env"]
    for var in NAMES:
        value = env[var]
        assert f"vars.{var}" in value and "github.repository" in value, value
        assert "@" not in value, "a fallback must not be a mailbox address"


def test_the_workflow_runs_only_commands_the_package_installs():
    """The first public CI ran the old command name and failed: the console script is `cyt`."""
    for job in _doc()["jobs"].values():
        for step in job.get("steps", []):
            for line in str(step.get("run", "")).splitlines():
                assert not line.strip().startswith("aw "), line
