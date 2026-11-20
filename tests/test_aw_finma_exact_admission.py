"""The FINMA exact retained-PDF admission chain (task 1396): the gates that were shell, now Python.

No GPU, no release, no observed run. Pins: the parent gate accepts only the complete 2,956-document
web-PDF run; a missing environment variable refuses; the chain is CPU only and never embeds; the
four seals run in order inside one observed step; an existing fresh-root refuses before any step.
"""
from __future__ import annotations

import json

import pytest

from cytir.ops import finma_exact_admission as A

GOOD = {"schema": A.PARENT_SCHEMA, "status": A.PARENT_STATUS, "source_profile": "web_pdf",
        "counts": {"source_records": 2956, "markdown_documents": 2956, "outlink_parents": 2956},
        "metadata_normalization": {"markdown_changed": False}}


def _env(tmp_path):
    rel = tmp_path / "release"
    rel.mkdir()
    env = {k: str(tmp_path / k.lower()) for k in A.ENV_KEYS}
    env.update(AW_RELEASE_DIR=str(rel), AW_FINMA_COMBINE_WAITER_PID="0")
    return env


def test_parent_gate_accepts_only_the_complete_run():
    A.parent_gate(GOOD)
    for patch in ({"schema": "x"}, {"status": "partial"}, {"source_profile": "html"},
                  {"counts": {**GOOD["counts"], "source_records": 2955}},
                  {"counts": {**GOOD["counts"], "outlink_parents": 0}},
                  {"metadata_normalization": {"markdown_changed": True}},
                  {"metadata_normalization": {}}):
        with pytest.raises(A.AdmissionRefused, match="parent gate"):
            A.parent_gate({**GOOD, **patch})


def test_a_missing_environment_variable_refuses():
    with pytest.raises(A.AdmissionRefused, match="AW_RELEASE_DIR"):
        A.read_env({k: "x" for k in A.ENV_KEYS if k != "AW_RELEASE_DIR"})


def test_seals_run_in_order_and_the_chain_is_cpu_only(tmp_path):
    env = _env(tmp_path)
    steps = A.seal_steps("py", env)
    modules = [s[s.index("-m") + 1] for s in steps]
    assert modules == ["cytir.index.seal_source_profile_compliance",
                       "cytir.index.seal_content_fidelity_audit",
                       "cytir.embed.portableize_pack", "cytir.embed.adapt_pack_to_stage"]
    assert "--source-id" in steps[0] and A.SOURCE_ID in steps[0]
    assert not any("embed_compose" in part or "run_embed" in part for s in steps for part in s)
    assert A.step_pythonpath(env) == env["AW_RELEASE_DIR"]


def test_existing_fresh_root_refuses_before_any_step(tmp_path, monkeypatch):
    env = _env(tmp_path)
    rel = tmp_path / "release"
    (rel / "SHA256SUMS").write_text("")
    parent = tmp_path / "aw_finma_parent_root"
    parent.mkdir()
    (parent / "RUN.json").write_text(json.dumps(GOOD))
    (tmp_path / "aw_finma_pack_root").mkdir()
    calls = []
    monkeypatch.setattr(A, "verify_release", lambda *a: None)
    monkeypatch.setattr(A, "wait_for_pid", lambda pid: None)
    monkeypatch.setattr(A, "run_observed", lambda *a, **k: calls.append(a))
    with pytest.raises(A.AdmissionRefused, match="AW_FINMA_PACK_ROOT already exists"):
        A.admit(env, "py")
    assert calls == []


def test_profile_step_argv_runs_the_package_module(tmp_path, monkeypatch):
    env = _env(tmp_path)
    rel = tmp_path / "release"
    (rel / "SHA256SUMS").write_text("")
    parent = tmp_path / "aw_finma_parent_root"
    parent.mkdir()
    (parent / "RUN.json").write_text(json.dumps(GOOD))
    steps = {}
    monkeypatch.setattr(A, "release_revision", lambda r: "rev")
    monkeypatch.setattr(A, "verify_release", lambda *a: None)
    monkeypatch.setattr(A, "wait_for_pid", lambda pid: None)
    monkeypatch.setattr(A, "run_observed", lambda name, cmd, **k: steps.__setitem__(name, cmd))
    A.admit(env, "py")
    cmd = steps["profile"]
    assert cmd[:3] == ["env", f"PYTHONPATH={env['AW_RELEASE_DIR']}", "py"]
    assert cmd[3:5] == ["-m", "cytir.ops.audits.finma_web_pdf_profile"]
    assert A.PROFILE_AUDIT_MODULE == cmd[4]
