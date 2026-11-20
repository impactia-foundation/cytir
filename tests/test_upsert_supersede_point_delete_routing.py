#!/usr/bin/env python3
"""Guards that `Superseder.run()` routes its delete through the shared
`point_delete.delete_points_verified` when constructed with `url=`, and is
BYTE-IDENTICAL to its pre-task-641 behaviour when it is not (every existing
caller across tasks 348, 351 and 385 constructs without `url`).
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cytir.index.supersede as US  # noqa: E402


class _FakeClient:
    """Enough of a QdrantClient for Superseder's constructor/legacy path --
    never actually called when `url=` routes through point_delete instead."""

    def delete(self, *a, **k):
        raise AssertionError("legacy qdrant_client delete must not run "
                             "when url= is set")

    def retrieve(self, *a, **k):
        raise AssertionError("legacy qdrant_client retrieve must not run "
                             "when url= is set")


def _stub_scan(sup, ids, urls):
    sup._scan_stale = lambda group_batch=64: (list(ids), list(urls), {})


def test_run_with_url_calls_the_shared_delete_primitive(monkeypatch):
    calls = []
    monkeypatch.setattr(
        US.PD, "delete_points_verified",
        lambda url, coll, ids, **k: calls.append((url, coll, list(ids))) or len(ids))
    monkeypatch.setattr(US, "_gold_audit", lambda *_a, **_k: 0)

    sup = US.Superseder(_FakeClient(), "aw_e8", url="http://twin:6457")
    _stub_scan(sup, ["p1", "p2"], ["https://x/a", "https://x/b"])
    stats = sup.run()

    assert calls == [("http://twin:6457", "aw_e8", ["p1", "p2"])]
    assert stats["superseded"] == 2


def test_run_without_url_keeps_the_legacy_qdrant_client_path(monkeypatch):
    """No behaviour change for the 8+ existing callers that never pass url."""
    called = []
    monkeypatch.setattr(US.PD, "delete_points_verified",
                        lambda *a, **k: called.append(1) or 0)
    monkeypatch.setattr(US, "_gold_audit", lambda *_a, **_k: 0)

    class LegacyClient(_FakeClient):
        def delete(self, coll, points_selector, wait):
            self.deleted = list(points_selector.points)

        def retrieve(self, coll, ids, with_payload, with_vectors):
            return []  # nothing survives -- verified gone

    client = LegacyClient()
    sup = US.Superseder(client, "aw_e8")  # no url=
    _stub_scan(sup, ["p1"], ["https://x/a"])
    stats = sup.run()

    assert called == [], "the shared primitive must not be touched"
    assert client.deleted == ["p1"]
    assert stats["superseded"] == 1
