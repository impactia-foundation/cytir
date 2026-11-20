"""The SIRENE acquisition extractor (ported from bash by task 2176): exit codes and the hand-off."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cytir.sources.official_bulk import sirene_legal_units as S


@pytest.fixture(autouse=True)
def _identity(monkeypatch):
    monkeypatch.setattr(S.user_agent, "user_agent", lambda *a: "AW-test (+https://example.org)")


def test_root_is_required(capsys):
    assert S.main([], env={}) == 1
    assert "CYTIR_SIRENE_ROOT" in capsys.readouterr().err


def test_fetch_vantage_is_required_like_the_root(tmp_path, capsys):
    assert S.main([], env={"AW_SIRENE_ROOT": str(tmp_path)}) == 1
    assert "CYTIR_FETCH_VANTAGE" in capsys.readouterr().err


def test_partial_evidence_set_is_refused(tmp_path, capsys):
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "stock-unite-legale.parquet").write_bytes(b"x")
    env = {"AW_SIRENE_ROOT": str(tmp_path), "AW_FETCH_VANTAGE": "build-box"}
    assert S.main([], env=env) == 2
    assert "partial immutable" in capsys.readouterr().err


def test_complete_evidence_set_goes_straight_to_the_bulk_adapter(tmp_path, monkeypatch):
    inputs = tmp_path / "input"
    inputs.mkdir()
    body = inputs / "stock-unite-legale.parquet"
    for p in (body, inputs / (body.name + ".headers"), inputs / (body.name + ".transport.json")):
        p.write_bytes(b"x")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=5)

    monkeypatch.setattr(S.subprocess, "run", fake_run)
    env = {"AW_SIRENE_ROOT": str(tmp_path), "AW_FETCH_VANTAGE": "build-box", "AW_PYTHON_BIN": "py"}
    assert S.main([], env=env) == 5
    assert len(calls) == 1  # no curl: the evidence is immutable
    cmd = calls[0]
    assert cmd[:3] == ["py", "-m", "cytir.sources.official_bulk.bulk"]
    assert cmd[cmd.index("--source-id") + 1] == "fr-sirene-legal-units"
    assert cmd[cmd.index("--fetch-vantage") + 1] == "build-box"
    assert cmd[cmd.index("--requested-url") + 1] == S.URL
