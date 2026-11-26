"""Task 3002 (ruling 75, S8 slice S3): the first three direct writers call `change_log.record_direct`.

One block per writer: live upsert (`cytir.index.upsert`), twin-patch-fields, consent backfill. Each pins
WHAT is recorded (op, ids, tool name), THAT it is recorded after the write, that a log failure is loud and
never drops the write, and that the log being off changes nothing. The upsert block runs the real `_run`
against two in-memory Qdrant clients (the throwaway of `tests/test_change_log_direct.py`) and compares the
log with a scroll of the collection through the shadow-compare tool: the log must equal what the scroll saw.
The twin-patch and consent blocks use those writers' own harnesses (a REST fake for the payload endpoints)."""
from __future__ import annotations

import fcntl
import json
import pathlib
from types import SimpleNamespace

import pytest
from qdrant_client.http import models as qmodels

from cytir.consent import apply as CA
from cytir.core.qdrant import qdrant_client
from cytir.distribute import shadow_compare as SC
from cytir.index import change_log as CL
from cytir.index import upsert as UP
from cytir.ops import twin_patch_fields as TP
from tests.test_aw_consent_apply import FakeQdrant, _ctx, _twin_rows
from tests.test_aw_dump_shadow_compare import COLL, DIM, _point, _vec
from tests.test_aw_ops_twin_patch_fields import (COLL as TCOLL, Locks, Store, env, licence_rows,  # noqa: F401
                                                 real, receipt, run, twin_of, write_rows)
from tests.test_change_log_direct import world  # noqa: F401  (fixture: prod collection + base dump + log dir)


def _seed(d):
    """A log that already exists (a drain landing): the preflight refuses a directory that holds none."""
    CL.record_direct(tool="seed", write_id="s0", collection="c", verified_written=True, upserts=["0"],
                     out_dir=d, landed_at="2026-11-01T00:00:00Z")
    return d


def _segments(log):
    out = {}
    for f in sorted(pathlib.Path(log).glob("*.jsonl")):
        head, *body = [json.loads(x) for x in f.read_text().splitlines()]
        out[head["pack_id"]] = (head, body)
    return out


# ============================================================== writer 1: live upsert
def _twin_client():
    t = qdrant_client(":memory:")
    t.create_collection("aw_twin", vectors_config={"dense": qmodels.VectorParams(
        size=DIM, distance=qmodels.Distance.COSINE)})
    t.upsert("aw_twin", points=[_point(i) for i in (1, 2, 3, 4)])
    return t


def _wave(tmp_path):
    """Add 7, refresh 2 in place, add 30 on the page of 3 (supersede deletes 3)."""
    rows = [{"id": 7, "vector": {"dense": _vec(0.7)}, "payload": _point(7).payload},
            {"id": 2, "vector": {"dense": _vec(0.77)}, "payload": _point(2).payload},
            {"id": 30, "vector": {"dense": _vec(0.3)}, "payload": _point(3).payload}]
    inp = tmp_path / "wave.jsonl"
    inp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return inp


def _run_upsert(world, monkeypatch, *, log_dir, batch=10, collection=COLL, twin=True):
    c, _, _, tmp = world
    twin = _twin_client() if twin else None
    monkeypatch.setattr(UP, "qdrant_connect", lambda *a, **k: c)
    monkeypatch.setattr(UP, "LOCK", str(tmp / "lock"))
    monkeypatch.setattr(UP, "ram_guard", lambda *_: None)
    monkeypatch.setattr(UP, "guard_card_overwrites_for_input", lambda *a, **k: {})   # the card guard is not under test
    monkeypatch.setattr(CL, "configured_dir", lambda: log_dir)
    args = SimpleNamespace(inp=str(_wave(tmp)), collection=collection, qdrant_url="mem", batch=batch,
                           min_avail_gib=0.0, health_url="", health_every=20, purpose="t", skip_counts=True,
                           break_stale_lock=False, no_supersede=False, create_url_index=True,
                           card_waivers=None, card_waivers_sha256=None)
    ledger = []
    try:
        UP._run(args, ledger=lambda line, **k: ledger.append(line), twin_client=twin, twin_coll="aw_twin" if twin else None)
    finally:
        UP.release_lock() if pathlib.Path(UP.LOCK).exists() else None
    return twin, ledger


def _scroll(c):
    pts, _ = c.scroll(COLL, limit=100, with_payload=True, with_vectors=False)
    return sorted((int(p.id), json.dumps(p.payload, sort_keys=True)) for p in pts)


def test_live_upsert_logs_every_prod_batch_and_the_supersede_deletes_and_the_log_equals_the_scroll(world, monkeypatch):
    c, _, log, _ = world
    _run_upsert(world, monkeypatch, log_dir=log, batch=2)          # 3 records, batch 2 -> two prod batches
    segs = _segments(log)
    mine = {k: v for k, v in segs.items() if k.startswith("direct-live-upsert-")}
    ups = sorted(k for k in mine if "-supersede-" not in k)
    assert len(ups) == 2 and ups[0].endswith("-000001") and ups[1].endswith("-000002")
    assert [x["id"] for k in ups for x in mine[k][1]] == ["7", "2", "30"]
    assert all(x["op"] == "upsert" for k in ups for x in mine[k][1])
    sup = [v for k, v in mine.items() if "-supersede-" in k]
    assert len(sup) == 1 and [(x["id"], x["op"], x["reason"]) for x in sup[0][1]] == [("3", "tombstone", "superseded")]
    assert {v[0]["collection"] if "collection" in v[0] else COLL for v in mine.values()} == {COLL}
    assert CL.assert_segments_complete(log) == 4                  # the seed segment + three of this wave
    # the delta built from the log equals the delta a scroll of the collection sees
    rc, rep = SC.run(client=c, collection=COLL, base_manifest_path=str(world[1] / "manifest.json"),
                     log_dir=str(log), out_dir=world[3] / "out", stamp="20261104T000000Z")
    assert rc == 0 and rep["verdict"] == "EQUAL", rep["verdict"]


def test_the_no_twin_hatch_logs_the_supersede_tombstones_from_prod(world, monkeypatch):
    """Task 3011 (review 2309 r2 item 3): registry mutant `on_deleted=dlog.superseded if False else None`
    passes the static check; with no twin the log follows PROD, so the delete must still be logged."""
    _, _, log, _ = world
    _run_upsert(world, monkeypatch, log_dir=log, batch=10, twin=False)
    mine = {k: v for k, v in _segments(log).items() if "-supersede-" in k}
    assert [(x["id"], x["op"]) for v in mine.values() for x in v[1]] == [("3", "tombstone")]


def test_live_upsert_writes_the_same_points_with_the_log_on_or_off(world, monkeypatch):
    c, _, log, _ = world
    _run_upsert(world, monkeypatch, log_dir=None)
    off = _scroll(c)
    assert not list(log.glob("*direct-live-upsert*"))            # log off: no segment
    # the same wave on a fresh prod with the log on
    c2 = qdrant_client(":memory:")
    c2.create_collection(COLL, vectors_config={"dense": qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE)})
    c2.upsert(COLL, points=[_point(i) for i in (1, 2, 3, 4)])
    w2 = (c2, world[1], log, world[3])
    _run_upsert(w2, monkeypatch, log_dir=log)
    assert _scroll(c2) == off and list(log.glob("*direct-live-upsert*"))


def test_live_upsert_logs_a_prod_batch_only_after_it_is_written(world, monkeypatch):
    c, _, log, _ = world
    seen = []
    real_record = CL.record_direct

    def spy(**kw):
        seen.append((kw["write_id"], [str(i) for i in kw.get("upserts", ())],
                     all(c.retrieve(COLL, ids=[int(i)]) for i in kw.get("upserts", ()))))
        return real_record(**kw)
    monkeypatch.setattr(CL, "record_direct", spy)
    _run_upsert(world, monkeypatch, log_dir=log)
    batches = [s for s in seen if "-supersede-" not in s[0]]
    assert batches and all(on_prod for _, _, on_prod in batches)   # every id was on prod when its line was written


def test_a_log_failure_is_ledgered_and_stops_the_run_but_the_write_stays(world, monkeypatch):
    c, _, log, _ = world

    def boom(**kw):
        raise CL.ChangeLogError("disk full")
    monkeypatch.setattr(CL, "record_direct", boom)
    with pytest.raises(SystemExit, match="change-log record failed"):
        _run_upsert(world, monkeypatch, log_dir=log)
    assert c.retrieve(COLL, ids=[7])                              # the write is not undone and not hidden
    assert not [s for s in _segments(log) if s.startswith("direct-live-upsert-")]


def test_a_prod_upsert_that_is_not_completed_is_not_logged_and_stops(world, monkeypatch):
    c, _, log, _ = world
    monkeypatch.setattr(c, "upsert", lambda *a, **k: SimpleNamespace(status="acknowledged"))
    with pytest.raises(SystemExit, match="did not return `completed`"):
        _run_upsert(world, monkeypatch, log_dir=log)
    assert not [s for s in _segments(log) if s.startswith("direct-live-upsert-")]


def test_live_upsert_refuses_before_the_first_write_when_the_log_dir_is_not_there(world, monkeypatch):
    c, _, _, tmp = world
    before = _scroll(c)
    with pytest.raises(SystemExit, match="does not exist"):
        _run_upsert(world, monkeypatch, log_dir=tmp / "no-such-log")
    assert _scroll(c) == before and not (tmp / "no-such-log").exists()


def test_a_death_after_delete_batch_one_of_two_leaves_batch_ones_tombstones_in_the_log(world, monkeypatch):
    """Review 3009 (1): the tombstones are logged per VERIFIED delete batch, from the TWIN's deletes (the
    delta's source), so a death between two batches loses nothing that was deleted."""
    c, _, log, tmp = world
    inp = tmp / "wave2.jsonl"
    rows = [{"id": 30, "vector": {"dense": _vec(0.3)}, "payload": _point(3).payload},
            {"id": 40, "vector": {"dense": _vec(0.4)}, "payload": _point(4).payload}]
    inp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    orig_run = UP.Superseder.run
    monkeypatch.setattr(UP.Superseder, "run", lambda self, **k: orig_run(self, **{**k, "delete_batch": 1}))
    twin = _twin_client()
    real_delete, calls = twin.delete, []

    def dying_delete(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("killed between delete batch 1 and 2")
        return real_delete(*a, **k)
    monkeypatch.setattr(twin, "delete", dying_delete)
    monkeypatch.setattr(UP, "qdrant_connect", lambda *a, **k: c)
    monkeypatch.setattr(UP, "LOCK", str(tmp / "lock2"))
    monkeypatch.setattr(UP, "ram_guard", lambda *_: None)
    monkeypatch.setattr(UP, "guard_card_overwrites_for_input", lambda *a, **k: {})
    monkeypatch.setattr(CL, "configured_dir", lambda: log)
    args = SimpleNamespace(inp=str(inp), collection=COLL, qdrant_url="mem", batch=10, min_avail_gib=0.0,
                           health_url="", health_every=20, purpose="t", skip_counts=True, break_stale_lock=False,
                           no_supersede=False, create_url_index=True, card_waivers=None, card_waivers_sha256=None)
    with pytest.raises(RuntimeError, match="killed between"):
        UP._run(args, ledger=lambda *a, **k: None, twin_client=twin, twin_coll="aw_twin")
    UP.release_lock()
    gone_on_twin = {3, 4} - {int(p.id) for p in twin.retrieve("aw_twin", ids=[3, 4])}
    assert len(gone_on_twin) == 1                                 # exactly one delete batch landed on the twin
    tombs = [x for k, (_, body) in _segments(log).items() if "-supersede-" in k for x in body]
    assert [int(x["id"]) for x in tombs] == sorted(gone_on_twin)  # and it is in the log, not lost


# ============================================================== writer 2: twin-patch-fields
def _patch_args(env, rows_file):
    return ["--from", str(rows_file), "--fields", "license", "--apply", "--export", str(env.tmp / "export.jsonl")]


def test_twin_patch_fields_logs_the_rows_it_wrote_and_read_back(real, env, monkeypatch):
    log = _seed(env.tmp / "log")
    monkeypatch.setattr(CL, "configured_dir", lambda: log)
    rows = licence_rows(real)
    store = Store(twin_of(real))
    code, _ = run(env, store, _patch_args(env, write_rows(env.tmp / "q1.jsonl", rows)))
    assert code == TP.EXIT_OK
    (pack, (head, body)), = [(k, v) for k, v in _segments(log).items() if k != "direct-seed-s0"]
    assert pack.startswith("direct-twin-patch-fields-") and head["n"] == len(rows)
    assert sorted(x["id"] for x in body) == sorted(r["id"] for r in rows)
    assert {x["op"] for x in body} == {"replace"}                 # an in-place patch lists every id as a replace
    assert receipt(env, "apply")["change_log_segment"].endswith(pack + ".jsonl")


def test_twin_patch_fields_dry_run_and_a_log_that_is_off_write_no_segment(real, env, monkeypatch):
    log = _seed(env.tmp / "log")
    monkeypatch.setattr(CL, "configured_dir", lambda: log)
    rows = write_rows(env.tmp / "q1.jsonl", licence_rows(real))
    run(env, Store(twin_of(real)), ["--from", str(rows), "--fields", "license"])      # dry run
    assert set(_segments(log)) == {"direct-seed-s0"}
    monkeypatch.setattr(CL, "configured_dir", lambda: None)
    code, _ = run(env, Store(twin_of(real)), _patch_args(env, rows))
    assert code == TP.EXIT_OK and set(_segments(log)) == {"direct-seed-s0"}
    assert "change_log_segment" not in receipt(env, "apply")


def test_twin_patch_fields_log_failure_is_a_refusal_that_says_the_segment_is_missing_and_the_spool_replays_it(real, env, monkeypatch):
    log = _seed(env.tmp / "log")
    monkeypatch.setattr(CL, "configured_dir", lambda: log)
    real_record = CL.record_direct

    def boom(**kw):
        raise CL.ChangeLogError("no space")
    monkeypatch.setattr(CL, "record_direct", boom)
    store = Store(twin_of(real))
    code, _ = run(env, store, _patch_args(env, write_rows(env.tmp / "q1.jsonl", licence_rows(real))))
    assert code == TP.EXIT_REFUSED
    assert "MISSING" in receipt(env, "apply")["refused"] and "replay-pending" in receipt(env, "apply")["refused"]
    assert store.writes                                           # the twin write itself stands
    assert len(list((log / CL.PENDING_DIR).glob("*.json"))) == 1  # the lost line is spooled
    monkeypatch.setattr(CL, "record_direct", real_record)         # the log works again: replay
    monkeypatch.setattr(CL, "_writer_alive", lambda w: False)    # the refused writer has exited (same pid here)
    with pytest.raises(CL.ChangeLogError, match="never recorded"):
        CL.preflight_dir()                                        # a new run refuses until the line is repaired
    ids = CL.replay_pending()
    assert len(ids) == 1 and not list((log / CL.PENDING_DIR).glob("*.json"))
    (pack, (head, body)), = [(k, v) for k, v in _segments(log).items() if k.startswith("direct-twin-patch")]
    assert sorted(x["id"] for x in body) == sorted(r["id"] for r in licence_rows(real)) and CL.replay_pending() == []
    CL.preflight_dir()                                            # and the next run may write again


def test_twin_patch_fields_logs_the_attempted_set_even_when_a_row_does_not_read_back(real, env, monkeypatch):
    """Review 3009 (3): a replace line for an attempted id is safe; a lost line is not."""
    log = _seed(env.tmp / "log")
    monkeypatch.setattr(CL, "configured_dir", lambda: log)
    rows = licence_rows(real)
    lost = rows[0]["id"]

    class Lossy(Store):
        def write(self, url, coll, op, what, ids):             # the first row never lands
            super().write(url, coll, op, what, [i for i in ids if i != lost])
    code, _ = run(env, Lossy(twin_of(real)), _patch_args(env, write_rows(env.tmp / "q1.jsonl", rows)))
    assert code == TP.EXIT_REFUSED                              # the verify step refuses, as before
    (_, (head, body)), = [(k, v) for k, v in _segments(log).items() if k.startswith("direct-twin-patch")]
    assert sorted(x["id"] for x in body) == sorted(r["id"] for r in rows)


def test_twin_patch_fields_refuses_before_the_first_write_when_the_log_dir_is_not_there(real, env, monkeypatch):
    monkeypatch.setattr(CL, "configured_dir", lambda: env.tmp / "no-such-log")
    store = Store(twin_of(real))
    code, _ = run(env, store, _patch_args(env, write_rows(env.tmp / "q1.jsonl", licence_rows(real))))
    assert code == TP.EXIT_REFUSED and store.writes == [] and "does not exist" in receipt(env, "apply")["refused"]
    assert not (env.tmp / "no-such-log").exists()                # and it did not create a second log


# ============================================================== writer 3: consent backfill
def _apply(q, **kw):
    return CA.apply_twin(q, "aw_e8", _ctx(), page=4, batch=3, workers=2, checkpoint_pages=1,
                         log=lambda m: None, **kw)


def test_consent_backfill_logs_every_written_batch_as_replace_after_the_write(monkeypatch, tmp_path):
    log = _seed(tmp_path / "log")
    q = FakeQdrant(_twin_rows())
    seen = []
    real_record = CL.record_direct

    def spy(**kw):
        seen.append(set(kw["payload_patches"]) <= q.written_ids())     # the ids were already written
        return real_record(**kw)
    monkeypatch.setattr(CL, "configured_dir", lambda: log)
    monkeypatch.setattr(CL, "record_direct", spy)
    out = _apply(q)
    segs = {k: v for k, v in _segments(log).items() if k != "direct-seed-s0"}
    assert segs and all(k.startswith("direct-consent-backfill-") for k in segs)
    assert not list((log / CL.PENDING_DIR).glob("*.json"))           # every spooled batch was recorded and cleared
    logged = [x["id"] for _, body in segs.values() for x in body]
    assert sorted(logged) == sorted(q.written_ids()) and len(logged) == len(set(logged))
    assert {x["op"] for _, body in segs.values() for x in body} == {"replace"}
    assert sum(out["written_by_rule"].values()) == len(logged)
    assert all(seen) and len(seen) == len(segs)


def test_consent_backfill_writes_the_same_rows_with_the_log_on_or_off(monkeypatch, tmp_path):
    monkeypatch.setattr(CL, "configured_dir", lambda: None)
    off = FakeQdrant(_twin_rows())
    _apply(off)
    monkeypatch.setattr(CL, "configured_dir", lambda: _seed(tmp_path / "log"))
    on = FakeQdrant(_twin_rows())
    _apply(on)
    assert on.rows == off.rows and on.written_ids() == off.written_ids()


def test_consent_backfill_log_failure_stops_the_run_naming_the_batch(monkeypatch, tmp_path):
    q = FakeQdrant(_twin_rows())
    log = _seed(tmp_path / "log")
    monkeypatch.setattr(CL, "configured_dir", lambda: log)
    real_record = CL.record_direct

    def boom(**kw):
        raise CL.ChangeLogError("read-only")
    monkeypatch.setattr(CL, "record_direct", boom)
    with pytest.raises(RuntimeError, match="written on aw_e8 but its change-log record failed"):
        _apply(q)
    assert q.written_ids()                                         # the write is real and is not hidden
    assert list((log / CL.PENDING_DIR).glob("*.json"))             # and its attempted ids are spooled
    monkeypatch.setattr(CL, "record_direct", real_record)
    monkeypatch.setattr(CL, "_writer_alive", lambda w: False)    # the stopped writer has exited (same pid here)
    assert CL.replay_pending() and not list((log / CL.PENDING_DIR).glob("*.json"))
    logged = {x["id"] for k, (_, body) in _segments(log).items() if k.startswith("direct-consent") for x in body}
    assert logged and logged <= set(_twin_rows())


def test_consent_backfill_refuses_before_the_first_write_when_the_log_dir_is_not_there(monkeypatch, tmp_path):
    q = FakeQdrant(_twin_rows())
    monkeypatch.setattr(CL, "configured_dir", lambda: tmp_path / "no-such-log")
    with pytest.raises(RuntimeError, match="REFUSED: .*does not exist"):
        _apply(q)
    assert q.writes == [] and not (tmp_path / "no-such-log").exists()


def test_consent_backfill_dry_run_logs_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(CL, "configured_dir", lambda: tmp_path / "log")
    _apply(FakeQdrant(_twin_rows()), dry=True)
    assert not (tmp_path / "log").exists()


# ============================================================== review 2853 (a): one lock for the check and the write
def test_record_direct_checks_the_held_segment_under_the_seq_lock(tmp_path, monkeypatch):
    CL.record_direct(tool="t", write_id="w1", collection="c", verified_written=True, upserts=["a"], out_dir=tmp_path)
    real_find = CL.segment_for
    state = {}

    def probe(out, pack_id):
        fd = __import__("os").open(str(pathlib.Path(out) / CL.SEQ_LOCK_FILE), __import__("os").O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)         # must FAIL: this process's lock is held
            state["free"] = True
        except BlockingIOError:
            state["free"] = False
        finally:
            __import__("os").close(fd)
        return real_find(out, pack_id)
    monkeypatch.setattr(CL, "segment_for", probe)
    CL.record_direct(tool="t", write_id="w1", collection="c", verified_written=True, upserts=["a"], out_dir=tmp_path)
    assert state == {"free": False}
