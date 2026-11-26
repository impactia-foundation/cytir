"""Lookup pins of the eight facade names the first lookup file did not reach (task 3021).

The drain's bodies read these names off `cytir.index.promote_queue` at call time (`_Q.<name>`), so a test
(or an operator's patch) that replaces one on the facade is honoured. Each test REPLACES the name on the
facade with a recording fake and asserts the drain used it. A bound name (`from ... import X`, a local copy)
passes every other test and fails here. Three names (`TFA`, `WriteAttempt`, `write_pack_receipt`) have their
pins in `test_drain_facade_lookups.py`; this file adds the sealed- and content-pack sites of the same three
and the five that had none.
"""
from __future__ import annotations

import json

from cytir.index import promote_queue as Q
from test_drain_pins_2941 import (  # noqa: F401  (helpers and the autouse fixtures they bring)
    _drain_with, _fresh_content_world, _fresh_sealed_world, _pending_content_resume)
from test_aw_sparse_rule import _capture, _point, fx, rule_env  # noqa: F401
from test_promote_queue import (  # noqa: F401
    MANIFEST_BASE, _default_prod_alias_resolution, _drain_plain, _one_pack_queue, _stub_plan_deps, write_contract,
    write_pack)


def _sealed_refused_by_the_writer(tmp_path, monkeypatch):
    """A sealed pack whose writer fails before any write: one REFUSED / live-apply-preflight record."""
    _, contract = _fresh_sealed_world(tmp_path, monkeypatch, profile="mechanical", supersede=False)

    def fail(**k):
        raise RuntimeError("boom")
    monkeypatch.setattr(Q, "live_apply", fail)
    return contract, tmp_path / "queue" / "fresh-1"


def test_the_drain_stamps_its_receipts_with_the_schema_read_from_the_facade(monkeypatch, tmp_path):
    monkeypatch.setattr(Q, "SCHEMA", "fake.schema.v9")
    contract, _ = _sealed_refused_by_the_writer(tmp_path, monkeypatch)
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert res[0]["schema"] == "fake.schema.v9"


def test_supersede_pack_stamps_its_receipt_with_the_schema_read_from_the_facade(monkeypatch, tmp_path):
    monkeypatch.setattr(Q, "SCHEMA", "fake.schema.v9")
    (tmp_path / "queue").mkdir()
    pack = write_pack(tmp_path / "queue", {**MANIFEST_BASE, "pack_id": "dup-1"})
    receipt = Q.supersede_pack(pack, reason="duplicate", task="3021")
    assert receipt["schema"] == "fake.schema.v9"
    assert json.loads((pack / "receipt.json").read_text())["schema"] == "fake.schema.v9"


def test_the_drain_writes_its_sealed_receipt_through_the_facades_write_pack_receipt(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(Q, "write_pack_receipt", lambda pack_dir, record: seen.append((pack_dir.name, record["status"])))
    contract, pack = _sealed_refused_by_the_writer(tmp_path, monkeypatch)
    _drain_with(tmp_path, apply=True, contract_path=contract)
    assert seen == [("fresh-1", "REFUSED")]
    assert not (pack / "receipt.json").exists()                  # the fake replaced the real writer


def test_the_sealed_drain_takes_its_write_attempt_from_the_facade(monkeypatch, tmp_path):
    made = []

    class FakeAttempt:
        started = False

        def __init__(self):
            made.append(self)
    monkeypatch.setattr(Q, "WriteAttempt", FakeAttempt)
    got = []
    _, contract = _fresh_sealed_world(tmp_path, monkeypatch, profile="mechanical", supersede=False)

    def writer(**k):
        got.append(k["write_attempt"])
        return {"twin_receipt": {}, "prod_receipt": {}}
    monkeypatch.setattr(Q, "live_apply", writer)
    _drain_with(tmp_path, apply=True, contract_path=contract)
    assert len(made) == 1 and got == made


def test_the_content_drain_takes_its_write_attempt_from_the_facade(monkeypatch, tmp_path):
    made = []

    class FakeAttempt:
        started = False

        def __init__(self):
            made.append(self)
    monkeypatch.setattr(Q, "WriteAttempt", FakeAttempt)
    _, seen, contract = _fresh_content_world(tmp_path, monkeypatch)
    _drain_with(tmp_path, apply=True, contract_path=contract)
    assert len(made) == 1 and [k["write_attempt"] for k in seen["apply"]] == made


def test_the_content_drain_takes_the_consent_census_from_the_facade(monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(Q, "consent_flag_census", lambda payloads: asked.append(list(payloads)) or {"from": "fake"})
    _, seen, contract = _fresh_content_world(tmp_path, monkeypatch)
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert asked == [[]] and res[0]["consent_flags"] == {"from": "fake"}


def test_the_pending_content_resume_runs_through_the_facades_retry_resume(monkeypatch, tmp_path):
    _pending_content_resume(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(Q, "_retry_resume",
                        lambda fn, *, pack_id: calls.append(pack_id) or {"prod_receipt": {"from": "fake-retry"}})
    monkeypatch.setattr(Q, "resolve_prod_alias", lambda *_a, **_k: "aw_e8")
    res = _drain_with(tmp_path, apply=True, contract_path=write_contract(tmp_path, "mechanical"))
    assert calls == ["cp-pending"] and res[0]["prod"] == {"from": "fake-retry"}


def test_the_sealed_drain_refuses_the_exceptions_the_facade_lists_and_only_those(monkeypatch, tmp_path):
    class Custom(Exception):
        pass
    _, contract = _fresh_sealed_world(tmp_path, monkeypatch, profile="mechanical", supersede=False)

    def planning(*a, **k):
        raise Custom("nope")
    monkeypatch.setattr(Q, "dry_plan", planning)
    monkeypatch.setattr(Q, "REFUSAL_EXCEPTIONS", (Custom,))
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert res[0]["status"] == "REFUSED" and res[0]["stage"] == "plan" and res[0]["reason"] == "Custom: nope"


def test_the_skip_cache_reads_the_refusal_stages_from_the_facade(monkeypatch, tmp_path):
    contract, pack = _sealed_refused_by_the_writer(tmp_path, monkeypatch)
    first = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert first[0]["status"] == "REFUSED" and first[0]["stage"] == "live-apply-preflight"
    # the real set does not cache this stage: a second pass meets the pack again
    again = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert again[0]["status"] == "REFUSED"
    monkeypatch.setattr(Q, "_CACHEABLE_REFUSAL_STAGES", frozenset({"live-apply-preflight"}))
    cached = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert cached[0]["status"] == "SKIPPED-REFUSED-UNCHANGED"
