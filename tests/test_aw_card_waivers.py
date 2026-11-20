"""Task 728: the sealed per-id card waiver file (ruling 26 item 19b)."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cytir.index import card_waivers as CW  # noqa: E402
from cytir.index import promote_queue as Q  # noqa: E402
from cytir.index import supersede as US  # noqa: E402

PID = "11111111-0000-5000-8000-00000000000a"
OLD = {"url": "https://a.ch/", "source_type": "entity_card", "entity_kind": "company",
       "register_id": "CHE-100.000.001", "register_scheme": "CHE-UID"}
SHOP = {"url": "https://a.ch/", "source_type": "official", "title": "Shop"}


def _row(**over):
    row = {"id": PID, "url": "https://a.ch/", "wrong_card_identity": sorted(US.card_identity(OLD)),
           "reason": "task 621 wrong entity", "evidence_sha256": "b" * 64,
           "approved_by": "tim", "approved_at": "2026-10-08", "pr": "#1"}
    row.update(over)
    return row


def _seal(path, rows):
    body = (json.dumps({"schema": CW.WAIVER_SCHEMA}) + "\n" + "".join(json.dumps(r) + "\n" for r in rows)).encode()
    path.write_bytes(body + CW.seal_line(body, len(rows)).encode())
    return path


def _guard(monkeypatch, waivers):
    monkeypatch.setattr(Q.PSP, "retrieve_payloads", lambda *a, **k: {PID: OLD})
    return Q.check_card_overwrites([{"id": PID, "payload": SHOP}], "http://x", "aw_e8", waivers=waivers)


def test_repo_waiver_file_is_sealed_and_loads():
    assert isinstance(CW.load_waivers(), dict)


def test_without_a_waiver_row_the_overwrite_stays_refused_with_one_it_is_admitted(monkeypatch, tmp_path):
    with pytest.raises(Q.QueueError, match="non-card"):
        _guard(monkeypatch, {})
    out = _guard(monkeypatch, CW.load_waivers(_seal(tmp_path / "w.jsonl", [_row()])))
    assert out["waived"] == [{"id": PID, "pr": "#1", "evidence_sha256": "b" * 64}] and out["held"] == []


@pytest.mark.parametrize("over", [{"url": "https://other.ch/"},
                                  {"wrong_card_identity": ["reg:CHE:CHE999999999"]}])
def test_a_waiver_for_another_url_or_identity_admits_nothing(monkeypatch, tmp_path, over):
    with pytest.raises(Q.QueueError, match="non-card"):
        _guard(monkeypatch, CW.load_waivers(_seal(tmp_path / "w.jsonl", [_row(**over)])))


def test_a_waiver_never_admits_a_card_of_another_identity(monkeypatch):
    other = {**OLD, "register_id": "CHE-200.000.002"}
    monkeypatch.setattr(Q.PSP, "retrieve_payloads", lambda *a, **k: {PID: OLD})
    with pytest.raises(Q.QueueError):
        Q.check_card_overwrites([{"id": PID, "payload": other}], "http://x", "aw_e8",
                                waivers={PID: _row()})


def test_missing_file_is_empty(tmp_path):
    assert CW.load_waivers(tmp_path / "none.jsonl") == {}


@pytest.mark.parametrize("mutate", [
    lambda t: t.replace('"tim"', '"eve"'),                       # edited, not re-sealed
    lambda t: "\n".join(t.splitlines()[:-1]) + "\n",             # seal line removed
    lambda t: t.rstrip("\n"),                                    # no final newline
    lambda t: t.replace(CW.WAIVER_SCHEMA, "aw.card-waivers.v0"),
])
def test_unsealed_or_edited_file_is_refused(tmp_path, mutate):
    p = _seal(tmp_path / "w.jsonl", [_row()])
    p.write_text(mutate(p.read_text()))
    with pytest.raises(CW.CardWaiverError):
        CW.load_waivers(p)


@pytest.mark.parametrize("bad", [_row(id="nope"), _row(evidence_sha256="xyz"), _row(reason=""),
                                 _row(wrong_card_identity=[]), {k: v for k, v in _row().items() if k != "pr"},
                                 {**_row(), "extra": 1}])
def test_malformed_rows_are_refused_even_when_sealed(tmp_path, bad):
    with pytest.raises(CW.CardWaiverError):
        CW.load_waivers(_seal(tmp_path / "w.jsonl", [bad]))


def test_duplicate_id_is_refused(tmp_path):
    with pytest.raises(CW.CardWaiverError, match="duplicate"):
        CW.load_waivers(_seal(tmp_path / "w.jsonl", [_row(), _row(id=PID.upper())]))


def test_a_waiver_needs_the_new_row_at_the_waivers_url(monkeypatch, tmp_path):
    """S4."""
    elsewhere = {**SHOP, "url": "https://other.ch/"}
    monkeypatch.setattr(Q.PSP, "retrieve_payloads", lambda *a, **k: {PID: OLD})
    with pytest.raises(Q.QueueError):
        Q.check_card_overwrites([{"id": PID, "payload": elsewhere}], "http://x", "aw_e8",
                                waivers=CW.load_waivers(_seal(tmp_path / "w.jsonl", [_row()])))


@pytest.mark.parametrize("token", ["reg:CHE-UID:CHE-105.974.401", "reg:che:che1", "QID:Q1", "qid:q1", "x"])
def test_a_non_canonical_identity_token_is_refused_at_load(tmp_path, token):
    """S5: it would never match; say so at load."""
    with pytest.raises(CW.CardWaiverError, match="canonical form"):
        CW.load_waivers(_seal(tmp_path / "w.jsonl", [_row(wrong_card_identity=[token])]))


def test_the_manifest_pin_is_checked_against_the_file_bytes(tmp_path):
    import hashlib
    p = _seal(tmp_path / "w.jsonl", [_row()])
    assert PID in CW.load_waivers(p, expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    with pytest.raises(CW.CardWaiverError, match="manifest"):
        CW.load_waivers(p, expected_sha256="0" * 64)
    with pytest.raises(CW.CardWaiverError, match="does not exist"):
        CW.load_waivers(tmp_path / "gone.jsonl", expected_sha256="0" * 64)
