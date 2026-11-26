"""Task 3011 (review 2309 r2, item 1): `replay-pending` and the preflight must not treat a spool that is
still IN FLIGHT as a lost line. A replay during a running writer records the line before the write landed;
a delta cut in between then ships the old row. Reproduction: the review's `race.py`."""
from __future__ import annotations

import json
import os

import pytest

from cytir.index import change_log as CL
from cytir.ops import write_lock as WL


@pytest.fixture()
def log(tmp_path, monkeypatch):
    d = tmp_path / "log"
    d.mkdir()
    monkeypatch.setattr(CL, "configured_dir", lambda: d)
    CL.record_direct(tool="live-upsert", write_id="seed", collection="c", verified_written=True, upserts=["1"])
    return d


def _seqs(d):
    return {p.name: CL._header_seq(p) for p in sorted(d.glob("*.jsonl"))}


def _fake_foreign_writer(path, **over):
    """Rewrite a spool as if a different process (pid/start/host) had written it."""
    rec = json.loads(path.read_text())
    rec["writer"] = {**rec["writer"], **over}
    path.write_text(json.dumps(rec))


def test_the_review_race_a_replay_during_a_live_writer_records_nothing(log):
    pend = CL.spool_pending("consent-backfill", "runX-000001", "twin", {"payload_patches": ["7", "8"]})
    before = _seqs(log)
    CL.preflight_dir()                                  # the writer is alive: a second writer is not refused
    with pytest.raises(CL.ChangeLogError, match="still running"):
        CL.replay_pending()                             # and an operator's replay is refused
    assert _seqs(log) == before and pend.exists()
    p = CL.record_direct(tool="consent-backfill", write_id="runX-000001", collection="twin",
                         verified_written=True, payload_patches=["7", "8"])
    assert CL._header_seq(p) > max(before.values())     # the writer's own line takes a NEW seq


def test_a_dead_writers_spool_is_a_lost_line_and_replays(log):
    pend = CL.spool_pending("consent-backfill", "runY-000001", "twin", {"payload_patches": ["7"]})
    _fake_foreign_writer(pend, start="1")               # same pid, other start time: a recycled pid
    with pytest.raises(CL.ChangeLogError, match="replay-pending"):
        CL.preflight_dir()
    assert CL.replay_pending() == ["runY-000001"] and not pend.exists()


def test_another_host_or_a_legacy_spool_without_a_writer_is_not_alive(log):
    a = CL.spool_pending("t", "w1", "c", {"payload_patches": ["1"]})
    _fake_foreign_writer(a, host="some-other-box")
    b = CL.spool_pending("t", "w2", "c", {"payload_patches": ["2"]})
    rec = json.loads(b.read_text())
    rec.pop("writer")
    b.write_text(json.dumps(rec))
    with pytest.raises(CL.ChangeLogError, match="never recorded"):
        CL.preflight_dir()
    assert sorted(CL.replay_pending()) == ["w1", "w2"]


@pytest.mark.parametrize("name", WL.LOCKS)
def test_replay_refuses_while_a_writer_lock_is_held(log, name):
    pend = CL.spool_pending("t", "w1", "c", {"payload_patches": ["1"]})
    _fake_foreign_writer(pend, start="1")               # a dead writer: only the lock stops the replay
    before = _seqs(log)
    WL.acquire(name, "999", "test", pid=os.getpid())
    try:
        with pytest.raises(CL.ChangeLogError, match="refused"):
            CL.replay_pending()
    finally:
        WL.release(name, "999")
    assert _seqs(log) == before and pend.exists()
    assert CL.replay_pending() == ["w1"]


def test_replay_survives_a_writer_that_cleared_its_spool_first(log, monkeypatch):
    pend = CL.spool_pending("t", "w1", "c", {"payload_patches": ["1"]})
    _fake_foreign_writer(pend, start="1")
    real = CL.record_direct

    def clear_then_record(**kw):
        pend.unlink(missing_ok=True)
        return real(**kw)
    monkeypatch.setattr(CL, "record_direct", clear_then_record)
    assert CL.replay_pending() == ["w1"]


def test_the_audit_reports_leftover_spools(log, tmp_path):
    q = tmp_path / "queue"
    q.mkdir()
    assert CL.audit(q, log)["pending_spools"] == 0 and CL.audit(q, log)["oldest_pending_spool_age_s"] is None
    pend = CL.spool_pending("t", "w1", "c", {"payload_patches": ["1"]})
    os.utime(pend, (1, 1))
    rep = CL.audit(q, log)
    assert rep["pending_spools"] == 1 and rep["oldest_pending_spool_age_s"] > 1000


def test_dry_run_keeps_the_pending_spool_and_a_normal_run_removes_it(tmp_path):
    from cytir.index import dry_run as DRY
    spool = tmp_path / "w.json"
    spool.write_text("{}")
    with pytest.raises(DRY.DryRunWriteRefused):
        with DRY.guard(True):
            CL.clear_pending(spool)
    assert spool.exists()
    CL.clear_pending(spool)
    assert not spool.exists()


def test_dry_run_replay_pending_deletes_nothing(log):
    from cytir.index import dry_run as DRY
    pend = log / CL.PENDING_DIR
    pend.mkdir(exist_ok=True)
    f = pend / "x.json"
    f.write_text("{}")
    with pytest.raises(DRY.DryRunWriteRefused):
        with DRY.guard(True):
            CL.replay_pending()
    assert f.exists()
