"""The gate inputs the public cut does not ship: an absent file is a named refusal, never a traceback and
never a pass. The readers are driven here with the file pointed at a path that does not exist."""
from __future__ import annotations

import sys

import pytest


def test_scratch_legs_names_the_missing_629_chooser(tmp_path, monkeypatch):
    from cytir.core import paths
    from cytir.gates import scratch_legs as SL
    monkeypatch.setattr(paths, "data_path", lambda *parts: tmp_path.joinpath(*parts))
    with pytest.raises(SL.ScratchLegRefused, match="eval_tasks.v2.yaml"):
        SL.queries_of("629")


def test_flip_gate_names_the_missing_629_chooser(tmp_path, monkeypatch):
    from cytir.gates import flip_gate as F
    monkeypatch.setattr(F, "REPO", tmp_path)
    doc = {"schema": "aw.scratch-leg-read.v1", "rows": [{"status": "OK", "id": "a", "cls": "c", "rank": 1}]}
    with pytest.raises(F.FlipGateError, match="eval_tasks.v2.yaml"):
        F._leg_read_arm(doc, "629")


def test_compare_corpus_arms_refuses_without_the_probe_fixture(tmp_path, monkeypatch, capsys):
    from cytir.gates import compare_corpus_arms as CC

    def missing(*_a, **_k):
        raise CC.ProbeFixtureMissing(f"probe fixture missing: {tmp_path / 'acceptance_probes.v1.json'}")
    monkeypatch.setattr(CC, "probe_specs", missing)
    monkeypatch.setattr(sys, "argv", ["compare_corpus_arms", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert CC.main() == 3
    assert "probe fixture missing" in capsys.readouterr().out


def test_run_corpus_arm_refuses_without_the_probe_fixture(tmp_path, monkeypatch):
    from cytir.gates import run_corpus_arm as RA
    monkeypatch.setattr(RA, "PROBES", tmp_path / "acceptance_probes.v1.json")
    monkeypatch.setattr(sys, "argv", ["run_corpus_arm", "--cache", "x=y", "--qvec-query-source-manifest", "x=y",
                                      "--target-manifest", "x=y", "--corpus-manifest", "x=y",
                                      "--out-dir", str(tmp_path / "o"), "--arm-id", "a"])
    with pytest.raises(SystemExit) as exc:
        RA.main()
    assert "probe fixture missing" in str(exc.value)


def test_card_waivers_absent_file_is_no_waivers_and_a_pinned_absent_file_refuses(tmp_path):
    from cytir.index import card_waivers as CW
    assert CW.load_waivers(tmp_path / "none.jsonl") == {}
    with pytest.raises(CW.CardWaiverError, match="does not exist"):
        CW.load_waivers(tmp_path / "none.jsonl", expected_sha256="0" * 64)
