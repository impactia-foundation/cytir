"""Task 2954: the pins sample review 2950 found missing before `drain()` moves (drain split step 14).

Four write sites of `drain()` share one failure branch (`except BaseException`): a write that was already issued
is ABORTED and stops the drain, a failure before any write is a REFUSED receipt and the drain goes on. Each of the
four is pinned here, then `_settle_guard` (the same shape at the "already held" exits), and the receipt carry-over of a prod-only resume.
The seven DONE writes needed no new pin: each is red under an existing test (task doc). Fakes only: no service is contacted.
"""
from __future__ import annotations

import json

import pytest

from cytir.index import promote_queue as Q
from test_drain_pins_2941 import (  # noqa: F401  (helpers and the autouse fixtures they bring)
    _drain_with, _fresh_content_world, _fresh_sealed_world, _live_twin_check_world, _pending_content_resume)
from test_aw_sparse_rule import _capture, _point, fx, rule_env  # noqa: F401
from test_promote_queue import (  # noqa: F401
    MANIFEST_BASE, _default_prod_alias_resolution, _drain_plain, _one_pack_queue, _stub_plan_deps, write_contract,
    write_pack)


def _receipt(pack):
    return json.loads((pack / "receipt.json").read_text())


def _writer(started: bool):
    """A fake writer that issues (or does not issue) a write, then fails."""
    def fail(**k):
        if started:
            k["write_attempt"].started = True
        raise RuntimeError("boom")
    return fail


# ---------------------------------------------------------------- the four write sites: sealed, content, resume x2

def _sealed(tmp_path, monkeypatch):
    seen, contract = _fresh_sealed_world(tmp_path, monkeypatch, profile="mechanical", supersede=False)
    return tmp_path / "queue" / "fresh-1", "live_apply", contract


def _content(tmp_path, monkeypatch):
    pack, seen, contract = _fresh_content_world(tmp_path, monkeypatch)
    return pack, "live_apply_content_pack", contract


def _prod_only(tmp_path, monkeypatch):
    seen, contract = _live_twin_check_world(tmp_path, monkeypatch)
    return tmp_path / "queue" / "test-pack-001", "resume_prod_only", contract


SITES = pytest.mark.parametrize("site", [_sealed, _content, _prod_only], ids=["sealed", "content", "prod-only-resume"])


@SITES
def test_2954_a_write_that_failed_after_it_was_issued_is_an_aborted_receipt_and_stops_the_drain(
        tmp_path, monkeypatch, site):
    """E2/E3/E5/E6 (sealed), B2/B3/B5/B6 (content), C2/C3/C5/C6 (prod-only resume) and the reason line of each."""
    pack, writer, contract = site(tmp_path, monkeypatch)
    monkeypatch.setattr(Q, writer, _writer(True))
    with pytest.raises(RuntimeError, match="boom"):
        _drain_with(tmp_path, apply=True, contract_path=contract)
    got = _receipt(pack)
    assert got["status"] == "ABORTED" and got["reason"] == "RuntimeError: boom"
    assert "stage" not in got or got["stage"] != "live-apply-preflight"
    assert not (pack / "DONE").exists()


@SITES
def test_2954_a_write_that_failed_before_any_write_is_a_retried_refusal_and_the_drain_goes_on(
        tmp_path, monkeypatch, site):
    """E6..E12, B6..B12, C6..C12: REFUSED at `live-apply-preflight` (never cached), a receipt on disk and in the
    results, no DONE, no stop."""
    pack, writer, contract = site(tmp_path, monkeypatch)
    monkeypatch.setattr(Q, writer, _writer(False))
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert len(res) == 1 and res[0]["status"] == "REFUSED" and res[0]["stage"] == "live-apply-preflight"
    assert res[0]["reason"] == "RuntimeError: boom"
    got = _receipt(pack)
    assert got["status"] == "REFUSED" and got["stage"] == "live-apply-preflight"
    assert got["reason"] == "RuntimeError: boom" and not (pack / "DONE").exists()
    assert "live-apply-preflight" not in Q._CACHEABLE_REFUSAL_STAGES


def test_2954_a_refused_fresh_content_pack_drops_a_stale_held_out_sidecar(tmp_path, monkeypatch):
    """B13: a later refusal of the pack leaves no sidecar of an earlier attempt."""
    pack, writer, contract = _content(tmp_path, monkeypatch)
    sidecar = pack / "held-out-cards.jsonl"
    sidecar.write_text('{"kind": "tombstone", "id": "t1"}\n')
    monkeypatch.setattr(Q, writer, _writer(False))
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert res[0]["status"] == "REFUSED" and not sidecar.exists()


def test_2954_a_failed_content_resume_is_an_aborted_receipt_with_the_reason_and_stops_the_drain(tmp_path, monkeypatch):
    """A1..A5: the fourth site (a content pack resumed from its twin state). Any failure stops the drain."""
    pack = _pending_content_resume(tmp_path, monkeypatch)
    monkeypatch.setattr(Q, "resume_content_pack_prod_only", _writer(False))
    with pytest.raises(RuntimeError, match="boom"):
        _drain_with(tmp_path, apply=True)
    got = _receipt(pack)
    assert got["status"] == "ABORTED" and got["reason"] == "RuntimeError: boom" and not (pack / "DONE").exists()


# ---------------------------------------------------------------- _settle_guard

class _Rec:
    def __init__(self):
        self.calls = []


def _guard(tmp_path, monkeypatch, behaviour, *, apply):
    """Run `_settle_guard` once with `settle_supersede_deleting` replaced by `behaviour(kw)`."""
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    seen = []

    def fake(**kw):
        seen.append(kw)
        return behaviour(kw)
    monkeypatch.setattr(Q, "settle_supersede_deleting", fake)
    record, results = {"pack_id": "p-1"}, []
    box = {}
    try:
        box["ok"] = Q._settle_guard(record, pack_dir, results, apply=apply, manifest={"m": 1}, task="t", batch=7)
    except BaseException as exc:  # noqa: BLE001
        box["raised"] = exc
    return record, results, pack_dir, seen, box


@pytest.mark.parametrize("apply,key,other", [(True, "supersede_recovery", "would_supersede_recovery"),
                                              (False, "would_supersede_recovery", "supersede_recovery")])
def test_2954_a_settle_that_returns_a_result_records_it_under_the_key_of_its_mode(tmp_path, monkeypatch, apply, key,
                                                                                  other):
    """G1/G18/G19/G20/G21/G23: the mode and every pass-through argument reach the settle; its result is recorded."""
    record, results, pack_dir, seen, box = _guard(tmp_path, monkeypatch, lambda kw: {"deleted": 3}, apply=apply)
    assert box == {"ok": True} and record[key] == {"deleted": 3} and other not in record
    assert results == [] and not (pack_dir / "receipt.json").exists()
    assert len(seen) == 1 and seen[0]["apply"] is apply
    assert seen[0]["manifest"] == {"m": 1} and seen[0]["task"] == "t" and seen[0]["batch"] == 7
    assert type(seen[0]["write_attempt"]).__name__ == "WriteAttempt" and seen[0]["write_attempt"].started is False


def test_2954_a_settle_with_nothing_to_report_records_no_key(tmp_path, monkeypatch):
    record, results, _pd, _seen, box = _guard(tmp_path, monkeypatch, lambda kw: None, apply=True)
    assert box == {"ok": True} and record == {"pack_id": "p-1"}


def test_2954_a_settle_that_fails_before_any_delete_is_a_refused_receipt_and_the_drain_goes_on(tmp_path, monkeypatch):
    """G5/G7/G8/G9/G11/G12/G13/G14/G15/G17/G24."""
    def boom(kw):
        raise RuntimeError("alias check failed")
    record, results, pack_dir, _seen, box = _guard(tmp_path, monkeypatch, boom, apply=True)
    assert box == {"ok": False}
    assert record["status"] == "REFUSED" and record["stage"] == "supersede-settle"
    assert record["reason"] == "RuntimeError: alias check failed"
    assert record["note"] == "the supersede settle stopped this pack; no DONE was written; no delete was issued"
    assert results == [record] and json.loads((pack_dir / "receipt.json").read_text()) == record


def test_2954_a_settle_that_fails_after_a_delete_was_issued_is_an_aborted_receipt_and_stops_the_drain(
        tmp_path, monkeypatch):
    """G2/G3/G9/G12/G15/G16/G22: the attempt the settle marks is the one the guard reads."""
    def boom(kw):
        kw["write_attempt"].started = True
        raise RuntimeError("delete died")
    record, results, pack_dir, _seen, box = _guard(tmp_path, monkeypatch, boom, apply=True)
    assert isinstance(box["raised"], RuntimeError) and "ok" not in box
    assert record["status"] == "ABORTED" and record["stage"] == "supersede-settle"
    assert record["note"].endswith("; a prod delete had been issued") and record["reason"] == "RuntimeError: delete died"
    assert results == [record] and json.loads((pack_dir / "receipt.json").read_text()) == record


def test_2954_an_alias_that_moved_mid_settle_is_an_aborted_receipt_even_with_no_delete_issued(tmp_path, monkeypatch):
    """G1: `ApplyAbort` is a SystemExit; it stops the drain but gets its receipt first."""
    def moved(kw):
        raise Q.TFA.ApplyAbort("alias moved mid-drain")
    record, results, pack_dir, _seen, box = _guard(tmp_path, monkeypatch, moved, apply=True)
    assert isinstance(box["raised"], Q.TFA.ApplyAbort)
    assert record["status"] == "ABORTED" and record["note"].endswith("; no delete was issued")
    assert results == [record] and json.loads((pack_dir / "receipt.json").read_text())["status"] == "ABORTED"


def test_2954_a_keyboard_interrupt_before_any_delete_passes_through_without_a_receipt(tmp_path, monkeypatch):
    """G4: only an Exception, an ApplyAbort or a failure after a delete earns a receipt."""
    def interrupted(kw):
        raise KeyboardInterrupt()
    record, results, pack_dir, _seen, box = _guard(tmp_path, monkeypatch, interrupted, apply=True)
    assert isinstance(box["raised"], KeyboardInterrupt)
    assert record == {"pack_id": "p-1"} and results == [] and not (pack_dir / "receipt.json").exists()


# ---------------------------------------------------------------- S3: found unpinned while correcting the 2941 table

def test_2954_a_single_target_that_holds_part_of_a_pack_is_a_retried_refusal_never_a_cached_one(tmp_path, monkeypatch):
    """S3. The 2941 table named a test that never fails under this mutation. A half-built collection is REFUSED at
    `live-apply-preflight`, which the skip cache never holds, so the next pass looks again (a `plan` stage would park
    the pack until a reseal)."""
    from test_drain_pins_2941 import _single_drain
    from cytir.index import promote_sealed_pack as PSP
    _one_pack_queue(tmp_path, monkeypatch)
    calls = []

    def partly(*_a, **_k):
        calls.append(1)
        raise PSP.AlreadyLandedError("part", ids=frozenset({"a", "b"}), present=frozenset({"a"}))
    monkeypatch.setattr(Q, "dry_plan", partly)
    res = _single_drain(tmp_path, monkeypatch, apply=True)
    assert res[0]["status"] == "REFUSED" and res[0]["stage"] == "live-apply-preflight"
    assert "partly written" in res[0]["reason"]
    again = _single_drain(tmp_path, monkeypatch, apply=True)
    assert again[0]["status"] == "REFUSED" and len(calls) == 2


# ---------------------------------------------------------------- S36 and S78, strengthened

def test_2954_a_prod_only_resume_keeps_the_twin_first_plan_of_the_receipt_it_replaces(tmp_path, monkeypatch):
    """S36 (strengthened). The 2941 pin only watched the argument of a spy. This one reads the receipt that lands on
    disk: the plan of the prior receipt (same manifest digest) survives the resume, and a prior of another manifest
    carries nothing, so the plan is derived from the prod leg."""
    import hashlib
    seen, contract = _live_twin_check_world(tmp_path, monkeypatch)
    pack = tmp_path / "queue" / "test-pack-001"
    monkeypatch.setattr(Q, "resume_prod_only", lambda **k: {"prod_receipt": {"written": 1, "rows": {"projected": 2}}})
    sha = hashlib.sha256((pack / "manifest.json").read_bytes()).hexdigest()
    prior = {"status": "REFUSED", "manifest_sha": sha, "mechanical_plan": {"rows": {"projected": 7}}}
    (pack / "receipt.json").write_text(json.dumps(prior))
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert res[0]["status"] == "LANDED" and res[0]["manifest_sha"] == sha
    assert _receipt(pack)["mechanical_plan"] == {"rows": {"projected": 7}}
    (pack / "DONE").unlink()
    (pack / "receipt.json").write_text(json.dumps({**prior, "manifest_sha": "0" * 64}))
    _drain_with(tmp_path, apply=True, contract_path=contract)
    derived = _receipt(pack)["mechanical_plan"]
    assert derived["rows"] == {"projected": 2} and "derived_from" in derived


def test_2954_the_plan_scratch_is_released_before_the_writer_runs_and_never_reaches_a_receipt(tmp_path, monkeypatch):
    """S78 (strengthened). The 2941 pin read the plan dict after the drain. This one holds the scratch only by weak
    reference and asks, INSIDE the writer, whether it is still alive: the matrix and the id index are large and must be
    gone before the pack is applied, though the caller still holds the plan dict."""
    import gc
    import weakref

    class Big:
        pass
    seen, contract = _fresh_sealed_world(tmp_path, monkeypatch, profile="mechanical", supersede=False)
    plans, refs, alive = [], [], []
    inner = Q.dry_plan

    def planner(*a, **k):
        plan = inner(*a, **k)
        plan["_matrix"], plan["_id_index"] = Big(), Big()
        refs.extend([weakref.ref(plan["_matrix"]), weakref.ref(plan["_id_index"])])
        plans.append(plan)
        return plan
    monkeypatch.setattr(Q, "dry_plan", planner)

    def writer(**k):
        gc.collect()
        alive.extend(r() is not None for r in refs)
        return {"twin_receipt": {}, "prod_receipt": {"superseded": 0}}
    monkeypatch.setattr(Q, "live_apply", writer)
    res = _drain_with(tmp_path, apply=True, contract_path=contract)
    assert res[0]["status"] == "LANDED" and alive == [False, False] and plans
    assert "_matrix" not in plans[0] and "_id_index" not in plans[0]
    text = (tmp_path / "queue" / "fresh-1" / "receipt.json").read_text()
    assert "_matrix" not in text and "_id_index" not in text and "_joined" not in text
