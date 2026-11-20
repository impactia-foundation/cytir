"""Task 432 — serving must be able to say which embedder revision it answers with.

Production TEI was launched with no --revision and `GET /info` returned
`"model_sha": null`. The build side pinned the sha on the command line; the read
side could not even report one. That is the mirror contract broken in the half
that decides what users see.

The check LOGS and records rather than refusing: a pin drift must be visible,
but turning it into a 503 in the request path trades a quality problem for an
outage.
"""
from __future__ import annotations


import pytest

from cytir.serve import models
from cytir.core.model_pin import EMBED_MODEL_ID, EMBED_MODEL_REVISION


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(models, "TEI_URL", "http://tei:80")
    monkeypatch.setattr(models, "_tei_rev",
                        {"checked": False, "model_sha": None, "ok": None, "detail": None})
    yield


def _stub_info(monkeypatch, payload):
    """The TEI /info answer, given at the internal door (`cytir.core.internal_http`)."""
    monkeypatch.setattr(models.IH, "fetch_json", lambda url, **kw: payload)


def test_pinned_revision_is_recognised(monkeypatch):
    _stub_info(monkeypatch, {"model_id": EMBED_MODEL_ID,
                             "model_sha": EMBED_MODEL_REVISION})
    models._check_tei_revision()
    st = models.tei_revision_status()
    assert st["ok"] is True
    assert st["model_sha"] == EMBED_MODEL_REVISION


def test_null_model_sha_is_a_finding_not_a_pass(monkeypatch):
    """The exact production state on 2026-08-19. It must not read as healthy."""
    _stub_info(monkeypatch, {"model_id": EMBED_MODEL_ID, "model_sha": None})
    models._check_tei_revision()
    st = models.tei_revision_status()
    assert st["ok"] is False
    assert "model_sha=null" in st["detail"]


def test_wrong_revision_is_a_finding(monkeypatch):
    _stub_info(monkeypatch, {"model_id": EMBED_MODEL_ID, "model_sha": "e" * 40})
    models._check_tei_revision()
    st = models.tei_revision_status()
    assert st["ok"] is False
    assert "e" * 40 in st["detail"]


def test_wrong_model_is_a_finding(monkeypatch):
    _stub_info(monkeypatch, {"model_id": "BAAI/bge-m3", "model_sha": "e" * 40})
    models._check_tei_revision()
    assert models.tei_revision_status()["ok"] is False


def test_unreachable_tei_is_unknown_not_a_failure(monkeypatch):
    """An unreachable sidecar is a different problem with its own alarm; it must
    not be reported as a pin violation."""
    def _boom(*a, **kw):
        raise OSError("connection refused")

    monkeypatch.setattr(models.IH, "fetch_json", _boom)
    models._check_tei_revision()
    st = models.tei_revision_status()
    assert st["ok"] is None
    assert "unreachable" in st["detail"]


def test_check_runs_once_per_process(monkeypatch):
    calls = {"n": 0}

    def _once(url, **kw):
        calls["n"] += 1
        return {"model_id": EMBED_MODEL_ID, "model_sha": EMBED_MODEL_REVISION}

    monkeypatch.setattr(models.IH, "fetch_json", _once)
    models._check_tei_revision()
    models._check_tei_revision()
    models._check_tei_revision()
    assert calls["n"] == 1


def test_server_uses_the_shared_pin():
    assert models.EMBED_MODEL == EMBED_MODEL_ID
    assert models.EMBED_REVISION == EMBED_MODEL_REVISION
