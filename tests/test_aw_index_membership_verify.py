from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from cytir.index import seal_collection_membership as verify  # the verify half (task 1234)


def _write_export(tmp_path, rows):
    shard = tmp_path / "membership-00000.jsonl.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    receipt = {
        "schema": "aw.qdrant-membership-receipt.v1",
        "rows": len(rows),
        "shards": [{"file": shard.name, "rows": len(rows),
                    "bytes": shard.stat().st_size,
                    "sha256": verify.sha256_file(shard)}],
    }
    return receipt


def test_fingerprint_is_order_independent_and_payload_sensitive():
    rows = [{"id": "2", "payload": {"b": 2}},
            {"id": "1", "payload": {"a": 1}}]
    assert verify.fingerprint(rows) == verify.fingerprint(reversed(rows))
    changed = [{"id": "2", "payload": {"b": 3}}, rows[1]]
    assert verify.fingerprint(rows) != verify.fingerprint(changed)


def test_expected_rows_rehashes_and_conserves_shard(tmp_path):
    rows = [{"id": "1", "payload": {"x": "y"}}]
    receipt = _write_export(tmp_path, rows)
    assert list(verify.expected_rows(receipt, tmp_path)) == rows

    shard = tmp_path / receipt["shards"][0]["file"]
    body = shard.read_bytes()
    shard.write_bytes(body[:-1] + bytes([body[-1] ^ 1]))
    with pytest.raises(verify.MembershipError, match="SHA mismatch"):
        list(verify.expected_rows(receipt, tmp_path))


def test_membership_row_bytes_ignore_object_key_order():
    left = {"id": 1, "payload": {"a": 1, "b": 2}}
    right = {"payload": {"b": 2, "a": 1}, "id": "1"}
    assert hashlib.sha256(verify.membership_row_bytes(left)).digest() == \
        hashlib.sha256(verify.membership_row_bytes(right)).digest()


def _stub_qdrant(monkeypatch, live_rows, counts):
    counts = list(counts)

    def api_json(url, body=None, timeout=300):
        if url.endswith("/points/scroll"):
            return {"points": [{"id": r["id"], "payload": r["payload"]} for r in live_rows],
                    "next_page_offset": None}
        return {"points_count": counts.pop(0)}
    monkeypatch.setattr(verify, "api_json", api_json)


def test_verify_door_passes_on_the_same_multiset_and_writes_the_proof(tmp_path, monkeypatch, capsys):
    rows = [{"id": "1", "payload": {"a": 1}}, {"id": "2", "payload": {"b": 2}}]
    receipt = _write_export(tmp_path, rows)
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    _stub_qdrant(monkeypatch, list(reversed(rows)), [2, 2])
    proof = tmp_path / "proof.json"
    rc = verify.main(["verify", "--receipt", str(tmp_path / "receipt.json"),
                      "--collection", "c", "--qdrant-url", "http://q.invalid", "--proof", str(proof)])
    assert rc == 0 and json.loads(proof.read_text())["status"] == "PASS"


@pytest.mark.parametrize("live,counts,why", [
    ([{"id": "1", "payload": {"a": 1}}, {"id": "2", "payload": {"b": 3}}], [2, 2], "differs"),
    ([{"id": "1", "payload": {"a": 1}}, {"id": "2", "payload": {"b": 2}}], [2, 3], "changed during"),
    ([{"id": "1", "payload": {"a": 1}}], [1, 1], "before scan"),
])
def test_verify_door_refuses_a_changed_payload_a_moving_count_and_a_short_collection(
        tmp_path, monkeypatch, capsys, live, counts, why):
    rows = [{"id": "1", "payload": {"a": 1}}, {"id": "2", "payload": {"b": 2}}]
    (tmp_path / "receipt.json").write_text(json.dumps(_write_export(tmp_path, rows)))
    _stub_qdrant(monkeypatch, live, counts)
    rc = verify.main(["verify", "--receipt", str(tmp_path / "receipt.json"), "--collection", "c"])
    assert rc == 1 and why in capsys.readouterr().err
