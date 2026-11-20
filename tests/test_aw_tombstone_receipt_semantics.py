"""Task 1467: what `tombstoned` in a drain receipt means, proven against a fake Qdrant.

Real case: pack refresh-direct-hf-models-20261015 declared 38 tombstones and the receipt said
`twin.tombstoned 19`. The other 19 rows were kept ON PURPOSE (3 urls under the word-coverage floor),
so nothing was lost; the receipt just did not say so. `tombstoned` = ids present and verified gone;
`tombstones_requested` and `tombstones_not_found` now sit beside it.
"""
from __future__ import annotations

import pytest

from cytir.index import point_delete as PD
from cytir.index import promote_queue as PQ


class FakeQdrant:
    def __init__(self, held, applies_delete=True):
        self.held = set(held)
        self.applies_delete = applies_delete

    def api(self, url, path, body=None, method="POST", **kw):
        if "/points/delete" in path:
            if self.applies_delete:
                self.held -= set(body["points"])
            return {}
        return [{"id": i} for i in body["ids"] if i in self.held]


@pytest.fixture
def fake(monkeypatch):
    f = FakeQdrant({"a", "b", "c"})
    monkeypatch.setattr(PD, "_api", f.api)
    monkeypatch.setattr(PQ.PSP, "upsert", lambda u, c, pts, b, **kw: len(pts))
    monkeypatch.setattr(PQ.LAYOUT, "fingerprint", lambda u, c: {"points_count": 0})
    return f


def test_receipt_separates_requested_deleted_and_not_found(fake):
    tombs = [{"id": i} for i in ("a", "b", "c", "x", "y")]     # x, y never existed
    r = PQ.apply_content_pack_side("twin", "http://t", "aw_e8", [], tombs, batch=2, pause_s=0)
    assert r["tombstones_requested"] == 5 and r["tombstoned"] == 3 and r["tombstones_not_found"] == 2
    assert fake.held == set(), "every counted id is verified gone"


def test_an_acknowledged_but_unapplied_delete_raises_never_counts(monkeypatch):
    f = FakeQdrant({"a"}, applies_delete=False)
    monkeypatch.setattr(PD, "_api", f.api)
    with pytest.raises(PD.PointDeleteError):
        PD.delete_points_verified("http://t", "aw_e8", ["a"], batch=1)
