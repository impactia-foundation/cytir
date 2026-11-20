"""Task 2240: `cytir.index.apply_delta` writes only into an offline clone; a live target is refused first.

No real Qdrant: `qdrant_connect` is patched and must never be reached on a refusal."""
from __future__ import annotations

import gzip
import json
import sys

import pytest

from cytir.index import apply_delta as AD
from cytir.index import live_target as LT
from cytir.index import prod_guard as PG

CLONE_PORT = PG.PROD_PORT + 1          # a port that is not the serving tunnel
LOOPBACK = "local" + "host"
CLONE_URL = f"http://{LOOPBACK}:{CLONE_PORT}"
PROD_URL = f"http://{LOOPBACK}:{PG.PROD_PORT}"
POINT_ID = 7


class Stub:
    def __init__(self):
        self.upserts = []
        self.n = 3

    def get_collections(self):
        class _C:
            name = "aw_clone"
        return type("R", (), {"collections": [_C()]})()

    def count(self, *_a, **_k):
        return type("R", (), {"count": self.n})()

    def upsert(self, _coll, points, wait=True):
        self.upserts.append(list(points))
        self.n += len(points)


def _dump(tmp_path):
    p = tmp_path / "d.jsonl.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(json.dumps({"id": POINT_ID, "vector": {"dense": [0.1]},
                             "payload": {"url": "https://x.invalid/a"}, "_delta": "add"}) + "\n")
    return p


def _arm(monkeypatch, tmp_path, url, coll, holdings):
    connects = []
    stub = Stub()
    monkeypatch.setattr(LT, "read_holdings", lambda u: holdings)
    monkeypatch.setattr(AD, "qdrant_connect", lambda *a, **k: connects.append(a) or stub)
    monkeypatch.setattr(sys, "argv", ["apply_delta.py", "--qdrant-url", url, "--collection", coll,
                                      "--in", str(_dump(tmp_path)), "--no-supersede"])
    return connects, stub


@pytest.mark.parametrize("holdings,coll", [
    (({"aw_prod": "aw_x"}, frozenset({"aw_x"})), "aw_clone"),   # the endpoint holds the serving alias
    (({}, frozenset()), "aw_prod"),                              # the serving alias as the target
])
def test_a_live_target_is_refused_before_any_client_call(monkeypatch, tmp_path, holdings, coll):
    connects, stub = _arm(monkeypatch, tmp_path, PROD_URL, coll, holdings)
    with pytest.raises(SystemExit) as ei:
        AD.main()
    assert "FATAL" in str(ei.value)
    assert connects == [] and stub.upserts == []


def test_an_unreadable_endpoint_is_refused_not_written(monkeypatch, tmp_path):
    connects, stub = _arm(monkeypatch, tmp_path, CLONE_URL, "aw_clone", None)

    def boom(_u):
        raise LT.LiveTargetRefused("cannot read what the endpoint holds")

    monkeypatch.setattr(LT, "read_holdings", boom)
    with pytest.raises(SystemExit):
        AD.main()
    assert connects == [] and stub.upserts == []


def test_an_endpoint_that_holds_the_serving_alias_is_refused_on_any_port(monkeypatch, tmp_path):
    connects, stub = _arm(monkeypatch, tmp_path, CLONE_URL, "aw_clone",
                          ({"aw_prod": "aw_x"}, frozenset({"aw_x"})))
    with pytest.raises(SystemExit) as ei:
        AD.main()
    assert "LIVE endpoint" in str(ei.value)
    assert connects == [] and stub.upserts == []


def test_a_url_with_no_port_is_checked_on_the_port_the_client_uses(monkeypatch, tmp_path):
    seen = []
    connects, stub = _arm(monkeypatch, tmp_path, f"http://{LOOPBACK}", "aw_clone", ({}, frozenset()))
    monkeypatch.setattr(LT, "read_holdings", lambda u: seen.append(u) or ({"aw_prod": "aw_x"}, frozenset()))
    with pytest.raises(SystemExit):
        AD.main()
    assert seen == [f"http://{LOOPBACK}:{LT.QDRANT_DEFAULT_PORT}"]
    assert connects == []


def test_an_offline_clone_is_written(monkeypatch, tmp_path):
    monkeypatch.setattr(AD.PI, "assert_pages_persisted",
                        lambda *a, **k: {"pages": 1, "persisted": 1, "l1b_reconstructible": 0})
    monkeypatch.setattr(AD, "guard_card_overwrites", lambda *a, **k: None)
    monkeypatch.setattr(AD, "client_url", lambda c: CLONE_URL)
    connects, stub = _arm(monkeypatch, tmp_path, CLONE_URL, "aw_clone", ({}, frozenset({"aw_clone"})))
    assert AD.main() == 0
    assert len(connects) == 1 and len(stub.upserts) == 1 and stub.upserts[0][0].id == POINT_ID
