"""write_baseline via qdrant_connect (task 2069). Fake client; never a real Qdrant."""
from __future__ import annotations

from enum import Enum

import pytest

import cytir.core.qdrant as Q
from cytir.news import refresh_chain as RC


class _FakeInfo:
    def __init__(self, status, points):
        self.status = status
        self.points_count = points


def _patch_connect(monkeypatch, info=None, error=None):
    class _Client:
        def get_collection(self, name):
            if error:
                raise error
            return info

    seen = {}

    def fake(url, **kw):
        seen["url"], seen["kw"] = url, kw
        return _Client()

    monkeypatch.setattr(Q, "qdrant_connect", fake)
    return seen


def test_green_receipt_bytes(monkeypatch, tmp_path):
    class _St(str, Enum):
        GREEN = "green"

    seen = _patch_connect(monkeypatch, _FakeInfo(_St.GREEN, 7))
    out = tmp_path / "b.json"
    RC.write_baseline("http://twin.invalid:1", "col", out)
    expected = ('{\n  "collection": "col",\n  "points_count": 7,\n  "schema": "'
                + RC.BASELINE_SCHEMA
                + '",\n  "status": "green",\n  "url": "http://twin.invalid:1"\n}\n')
    assert out.read_text() == expected
    assert seen["kw"] == {"timeout": 30, "check_compatibility": False}


@pytest.mark.parametrize("status,points,msg", [
    ("yellow", 5, "canonical twin is not green"),
    ("green", 0, "canonical twin has no valid point count"),
    ("green", None, "canonical twin has no valid point count"),
])
def test_refusals(monkeypatch, tmp_path, status, points, msg):
    _patch_connect(monkeypatch, _FakeInfo(status, points))
    with pytest.raises(SystemExit) as e:
        RC.write_baseline("http://twin.invalid:1", "col", tmp_path / "b.json")
    assert str(e.value) == msg
    assert not (tmp_path / "b.json").exists()


def test_unavailable(monkeypatch, tmp_path):
    _patch_connect(monkeypatch, error=RuntimeError("boom"))
    with pytest.raises(SystemExit) as e:
        RC.write_baseline("http://twin.invalid:1", "col", tmp_path / "b.json")
    assert str(e.value) == "canonical twin unavailable: boom"
