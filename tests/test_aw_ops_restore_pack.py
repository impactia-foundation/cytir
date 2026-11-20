"""Task 826: the restore-pack composer writes a drainable content_pack and refuses a bad row."""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cytir.ops import restore_pack as RP  # noqa: E402


def _row(i: str = "80234a81-43fb-5497-88f8-2c9a841e3d02", **over) -> dict:
    r = {"id": i, "payload": {"url": "https://aktionpinguin.ch/", "entity_kind": "org", "register_id": "CHE253624766"},
         "vector": {"dense": [0.1, 0.2], "bm25": {"indices": [3, 9], "values": [1.0, 2.0]}}}
    r.update(over)
    return r


def test_write_pack_is_ready_and_hash_matches(tmp_path):
    man = RP.write_pack(str(tmp_path), "p1", [_row(), _row("ac56e2a2-dc0c-5bf2-ac3e-bf3e436c662c")],
                        collection="aw_e8", family="general_web", note="n", task=826)
    d = tmp_path / "p1"
    assert (d / "READY").read_text().strip() == hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest()
    assert man["kind"] == "content_pack" and man["rows"] == 2 and "tombstones" not in man
    body = gzip.open(man["pack_points"], "rb").read()
    assert hashlib.sha256(body).hexdigest() == man["pack_points_sha256"]
    assert json.loads(body.splitlines()[0])["id"].startswith("80234a81")


@pytest.mark.parametrize("bad", [
    _row(vector={"dense": [0.1]}),                                  # no sparse leg
    _row(vector={"bm25": {"indices": [1], "values": [1.0]}}),       # no dense leg
    _row(payload={"title": "x"}),                                   # no url
])
def test_bad_row_is_refused(tmp_path, bad):
    with pytest.raises(SystemExit):
        RP.write_pack(str(tmp_path), "p2", [bad], collection="c", family="f", note="n", task=1)
    assert not (tmp_path / "p2").exists()


def test_duplicate_id_and_existing_dir_are_refused(tmp_path):
    with pytest.raises(SystemExit):
        RP.write_pack(str(tmp_path), "p3", [_row(), _row()], collection="c", family="f", note="n", task=1)
    RP.write_pack(str(tmp_path), "p4", [_row()], collection="c", family="f", note="n", task=1)
    with pytest.raises(SystemExit):
        RP.write_pack(str(tmp_path), "p4", [_row()], collection="c", family="f", note="n", task=1)


def test_export_refuses_a_missing_id(monkeypatch):
    held = _row()

    def fake_post(url, body, timeout=0):
        return {"result": [{"id": held["id"], "payload": held["payload"], "vector": held["vector"]}]}

    monkeypatch.setattr(RP, "_post", fake_post)
    assert RP.export_rows("http://x", "c", [held["id"]])[0]["id"] == held["id"]
    with pytest.raises(SystemExit):
        RP.export_rows("http://x", "c", [held["id"], "00000000-0000-5000-8000-000000000000"])


# ---- task 844: a restore pack may carry `superseded` tombstones for the anonymous cards its cards replace

def _card(i, url, qid="Q616975"):
    return {"id": i, "payload": {"url": url, "source_type": "entity_card", "entity_kind": "org", "qid": qid},
            "vector": {"dense": [0.1], "bm25": {"indices": [1], "values": [1.0]}}}


_OLD = "1e98f41c-9b31-5209-947b-d6d9e35cb14e"
_NEW = "5e27e2e1-af31-5b88-86bf-e602ddd23f33"
_STORED = {_OLD: {"url": "https://edbangerrecords.com/", "source_type": "entity_card"}}


def test_build_tombstones_follows_card_replaces():
    card = _card(_NEW, "https://www.edbangerrecords.com/")
    t = RP.build_tombstones(_STORED, [card], detail="d")
    assert [(x["id"], x["reason"], x["replaced_by"]) for x in t] == [(_OLD, "superseded", [_NEW])]
    with pytest.raises(SystemExit):
        RP.build_tombstones(_STORED, [_card(_NEW, "https://www.other.example/")], detail="d")
    # a stored card that HAS an identity is not replaced by a card of another identity
    named = {_OLD: dict(_STORED[_OLD], qid="Q1", entity_kind="org")}
    with pytest.raises(SystemExit):
        RP.build_tombstones(named, [card], detail="d")


def test_write_pack_carries_tombstones_with_matching_hash(tmp_path):
    card = _card(_NEW, "https://www.edbangerrecords.com/")
    tombs = RP.build_tombstones(_STORED, [card], detail="d")
    man = RP.write_pack(str(tmp_path), "p5", [card], collection="c", family="f", note="n", task=844, tombstones=tombs)
    raw = pathlib.Path(man["tombstones"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == man["tombstones_sha256"] and man["tombstones_count"] == 1
    assert man["tombstones_by_reason"] == {"superseded": 1} and man["rows"] == 1
    assert json.loads(raw)["id"] == _OLD


def test_tombstone_checks_refuse_bad_shapes(tmp_path):
    card = _card(_NEW, "https://www.edbangerrecords.com/")
    good = {"id": _OLD, "url": "https://edbangerrecords.com/", "reason": "superseded"}
    for bad in ([dict(good, reason="takedown")], [dict(good, id=_NEW)], [good, good], [dict(good, url="")]):
        with pytest.raises(SystemExit):
            RP.write_pack(str(tmp_path), "pb", [card], collection="c", family="f", note="n", task=1, tombstones=bad)
        assert not (tmp_path / "pb").exists()
