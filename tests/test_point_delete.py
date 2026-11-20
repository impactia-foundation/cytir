#!/usr/bin/env python3
"""Guards for `point_delete.py` (task 641) -- the one home for "delete these
point ids, verified gone", extracted from `upsert_supersede.Superseder.run()`
so `cytir/index/promote_queue.py`'s tombstone apply and that class share one
implementation rather than two.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cytir.index import point_delete as PD  # noqa: E402


def _stub_api(monkeypatch, *, present_after=None, held=None):
    """held: ids the collection holds before the delete (None = every id asked for).
    present_after: ids that MockDelete claims are still present on the
    verify-read after a delete -- empty means every delete verified clean."""
    present_after = present_after or set()
    state = {"deleted": False}
    calls = []

    def fake(url, path, body=None, method=None):
        calls.append((path, body, method))
        if "/points/delete" in path:
            state["deleted"] = True
            return None
        if path.endswith("/points"):
            ids = body["ids"]
            if not state["deleted"]:
                return [{"id": i} for i in ids if held is None or i in held]
            state["deleted"] = False
            return [{"id": i} for i in ids if i in present_after]
        raise AssertionError(f"unexpected call {path}")

    monkeypatch.setattr(PD, "_api", fake)
    return calls


def test_an_absent_id_is_not_found_never_deleted(monkeypatch):
    calls = _stub_api(monkeypatch, held={"a", "c"})
    n = PD.delete_points_verified("http://x", "aw_e10", ["a", "b", "c"], batch=10, log=lambda *_: None)
    assert n == 2
    sent = [c[1]["points"] for c in calls if "/points/delete" in c[0]]
    assert sent == [["a", "c"]]
    calls.clear()
    assert PD.delete_points_verified("http://x", "aw_e10", ["b"], log=lambda *_: None) == 0
    assert not [c for c in calls if "/points/delete" in c[0]]


def test_delete_points_verified_deletes_and_confirms_gone(monkeypatch):
    calls = _stub_api(monkeypatch)
    n = PD.delete_points_verified("http://x", "aw_e8", ["a", "b", "c"], batch=10)
    assert n == 3
    delete_calls = [c for c in calls if "/points/delete" in c[0]]
    assert delete_calls == [
        ("/collections/aw_e8/points/delete?wait=true",
         {"points": ["a", "b", "c"]}, "POST")]


def test_delete_points_verified_batches(monkeypatch):
    calls = _stub_api(monkeypatch)
    ids = [str(i) for i in range(25)]
    n = PD.delete_points_verified("http://x", "aw_e8", ids, batch=10)
    assert n == 25
    delete_calls = [c for c in calls if "/points/delete" in c[0]]
    assert len(delete_calls) == 3
    assert [len(c[1]["points"]) for c in delete_calls] == [10, 10, 5]


def test_delete_points_verified_refuses_a_survivor(monkeypatch):
    _stub_api(monkeypatch, present_after={"b"})
    with pytest.raises(PD.PointDeleteError, match="still present"):
        PD.delete_points_verified("http://x", "aw_e8", ["a", "b", "c"])


def test_fetch_payload_fields_returns_only_found_ids(monkeypatch):
    def fake(url, path, body=None, method=None):
        assert path.endswith("/points")
        return [{"id": "a", "payload": {"published": "2020-01-01"}}]
    monkeypatch.setattr(PD, "_api", fake)
    got = PD.fetch_payload_fields("http://x", "aw_e8", ["a", "b"], ["published"])
    assert got == {"a": {"published": "2020-01-01"}}
    assert "b" not in got


def test_fetch_payload_fields_accepts_true_for_the_full_payload(monkeypatch):
    """Task 673 review round 4, finding 5: `with_payload=True` (qdrant's
    own "give me everything" value), not just a field allowlist -- a
    caller comparing full payloads (`_row_content_signature`'s fallback)
    must be able to ask for it."""
    def fake(url, path, body=None, method=None):
        assert body["with_payload"] is True
        return [{"id": "a", "payload": {"published": "2020-01-01", "title": "x"}}]
    monkeypatch.setattr(PD, "_api", fake)
    got = PD.fetch_payload_fields("http://x", "aw_e8", ["a"], True)
    assert got == {"a": {"published": "2020-01-01", "title": "x"}}
