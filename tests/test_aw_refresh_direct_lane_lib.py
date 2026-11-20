"""real-record tests for cytir.sources.refresh_direct.lane_lib (task 632).

Uses real cadence rows from the committed data/refresh-cadence.v1.json (not
synthetic fixtures) for the due_report tests, and a real tmp-file flock for
the lock tests (two real OS processes contending, not a mock).
"""
from __future__ import annotations

import json
import math
import multiprocessing
import time
from datetime import datetime, timedelta, timezone

import pytest

from cytir.core.paths import data_path
from cytir.sources.refresh_direct import lane_lib as lib


NOW = datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc)


def _require_cadence():
    """Skip when the public cut withholds data/refresh-cadence.v1.json (exact path only)."""
    path = data_path("refresh-cadence.v1.json")
    if not path.is_file():
        pytest.skip("data/refresh-cadence.v1.json is withheld from the public cut")
    return path


def test_cold_start_is_immediately_due_and_alarms():
    r = lib.compute_due("x", "periodic_refetch", 30, None, now=NOW)
    assert r.due is True
    assert r.alarm is True
    assert r.overdue_ratio == math.inf


def test_not_due_before_cadence_elapses():
    last = (NOW - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = lib.compute_due("x", "periodic_refetch", 30, last, now=NOW)
    assert r.due is False
    assert r.alarm is False


def test_due_but_not_alarmed_just_past_cadence():
    last = (NOW - timedelta(days=31)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = lib.compute_due("x", "periodic_refetch", 30, last, now=NOW)
    assert r.due is True
    assert r.alarm is False  # ratio ~1.03, below the 2x alarm band


def test_alarm_fires_past_2x_cadence():
    last = (NOW - timedelta(days=61)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = lib.compute_due("x", "periodic_refetch", 30, last, now=NOW)
    assert r.due is True
    assert r.alarm is True
    assert r.overdue_ratio > 2.0


def test_none_cadence_never_due_never_alarms():
    r = lib.compute_due("news-x", "periodic_refetch", None, None, now=NOW)
    assert r.due is False
    assert r.alarm is False
    assert r.overdue_ratio is None


def test_state_roundtrip(tmp_path):
    state_path = tmp_path / "state.json"
    state = lib.load_state(state_path)  # no file yet -> empty scaffold
    assert state["sources"] == {}
    state = lib.mark_run_complete(state, "ch-fedlex", "periodic_refetch", "2026-09-20T00:00:00Z")
    lib.save_state(state_path, state)
    reloaded = lib.load_state(state_path)
    assert reloaded["sources"]["ch-fedlex"]["periodic_refetch"] == "2026-09-20T00:00:00Z"


def test_due_report_uses_real_cadence_table_and_skips_ineligible_sources():
    _require_cadence()
    cadence = json.loads(data_path("refresh-cadence.v1.json").read_text())
    rows = cadence["sources"]
    state = {"sources": {}}  # nothing ever run -> everything eligible is due
    results = lib.due_report(rows, state, "periodic_refetch", now=NOW)
    result_ids = {r.source_id for r in results}
    eligible_ids = {r["id"] for r in rows if r["eligible_now"]}
    assert result_ids == eligible_ids
    # every eligible law-government/commercial/stats/docs source with a real
    # cadence is due on a cold start; sources with cadence None never are.
    for r in results:
        row = next(x for x in rows if x["id"] == r.source_id)
        if row["periodic_refetch_days"] is None:
            assert r.due is False
        else:
            assert r.due is True


def test_ledger_append_is_one_json_object_per_line(tmp_path):
    ledger_path = tmp_path / "ledger.ndjson"
    lib.append_ledger(ledger_path, {"run": "a", "n": 1})
    lib.append_ledger(ledger_path, {"run": "b", "n": 2})
    lines = ledger_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["run"] == "a"
    assert json.loads(lines[1])["run"] == "b"


def _hold_lock_then_signal(lock_path, hold_seconds, ready_event):
    with lib.acquire_prod_write_lock(lock_path, blocking=True):
        ready_event.set()
        time.sleep(hold_seconds)


def test_prod_write_lock_excludes_a_concurrent_holder(tmp_path):
    lock_path = tmp_path / "PROD-WRITE-LOCK"
    ready = multiprocessing.Event()
    proc = multiprocessing.Process(target=_hold_lock_then_signal, args=(str(lock_path), 1.5, ready))
    proc.start()
    try:
        assert ready.wait(timeout=5), "holder process never acquired the lock"
        try:
            with lib.acquire_prod_write_lock(lock_path, blocking=False):
                raise AssertionError("should not have acquired an already-held lock")
        except TimeoutError:
            pass  # expected: non-blocking acquire refuses while held
    finally:
        proc.join(timeout=5)

    # once the holder releases, a fresh non-blocking acquire succeeds.
    with lib.acquire_prod_write_lock(lock_path, blocking=False):
        pass


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
