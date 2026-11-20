"""Task 2850: the rehearsal builds a delta FROM THE CHANGE LOG and applies it
(`specs/delta-change-log.md` section 5, tests 1 and 2) on the sealed rehearsal fixture."""
from __future__ import annotations

import shutil

import pytest

from cytir.distribute import rehearsal, rehearsal_log_delta as L
from cytir.distribute import pack

pytestmark = pytest.mark.skipif(not shutil.which("minisign"), reason="minisign not on PATH")


def _stage(tmp_path):
    sec, pub = rehearsal._ephemeral_keys(tmp_path)
    return L.rehearse_log_delta(rehearsal.DEFAULT_FIXTURE_DIR, tmp_path, sec, pub)


def test_log_delta_equals_scroll_delta_and_applies_to_the_source_state(tmp_path):
    r = _stage(tmp_path)
    assert r["log_equals_scroll"] is True
    assert r["segments"] == 6
    assert r["delta_ops"] == {"add": L.LOG_ADDS, "replace": L.LOG_REPLACES + L.LOG_PATCHES}
    assert r["tombstones"] == L.LOG_TOMBS
    assert set(r["tombstone_reasons"]) == set(pack.TOMBSTONE_REASONS)
    assert r["net_noops_dropped"] >= L.LOG_REVERTS            # a reverted patch is not shipped
    assert r["points_after"] == r["base_points"] + L.LOG_ADDS - L.LOG_TOMBS
    assert r["redump"]["rows"] == r["points_after"] and r["refusals"] == 3


def test_a_pack_that_never_reached_the_log_makes_the_comparison_fail(tmp_path, monkeypatch):
    """The comparison can fail: all but one title replace reach the twin without a log line (a bypass)."""
    real = L._land

    def land(log_dir, pack_id, plan):
        if "p2-replaces" in pack_id:
            plan = dict(plan, upserts=plan["upserts"][:1])
        real(log_dir, pack_id, plan)

    monkeypatch.setattr(L, "_land", land)
    with pytest.raises(rehearsal.RehearsalFailure, match="differ in"):
        _stage(tmp_path)


def test_rehearse_reports_the_log_delta_stage(tmp_path):
    assert rehearsal.rehearse(rehearsal.DEFAULT_FIXTURE_DIR, tmp_path)["log_delta"]["log_equals_scroll"] is True


#: Written out by hand, not read from the code under test.
_QUEUE_TO_PUBLIC = {
    "superseded": "superseded", "aged_out": "aged_out", "takedown": "takedown",
    "superseded_generation": "superseded_generation",
    "canonical_duplicate": "superseded_generation", "binary_content": "superseded_generation",
    "soft_404": "superseded_generation", "restore_rollback": "superseded_generation",
}


def test_the_reason_table_both_sides_rely_on_is_the_hand_written_one():
    from cytir.index import change_log as CL
    assert L.EXPECTED_PUBLIC_REASON == _QUEUE_TO_PUBLIC
    assert CL.QUEUE_TO_PUBLIC_REASON == _QUEUE_TO_PUBLIC


def test_one_corrupted_logged_tombstone_reason_fails_the_comparison(tmp_path, monkeypatch):
    real = L._land
    hit = []

    def land(log_dir, pack_id, plan):
        if "p3-tombstones" in pack_id:
            ts = [dict(t) for t in plan["tombstones"]]
            old = ts[0]["reason"]
            ts[0]["reason"] = "aged_out" if _QUEUE_TO_PUBLIC[old] != "aged_out" else "takedown"
            hit.append(old)
            plan = dict(plan, tombstones=ts)
        real(log_dir, pack_id, plan)

    monkeypatch.setattr(L, "_land", land)
    with pytest.raises(rehearsal.RehearsalFailure, match=r"tombstones \(id, reason\) differ"):
        _stage(tmp_path)
    assert hit


def test_a_non_local_client_is_refused_before_any_write(tmp_path):
    class Stub:
        calls: list = []

        def __getattr__(self, name):
            Stub.calls.append(name)
            if name == "_client":
                raise AttributeError(name)
            raise AssertionError(f"write attempted: {name}")

    with pytest.raises(rehearsal.RehearsalFailure, match="local in-memory"):
        L._drain_sequence(Stub(), [], tmp_path)
    assert set(Stub.calls) <= {"_client"}
