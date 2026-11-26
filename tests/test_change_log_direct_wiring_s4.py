"""Task 3020 (ruling 75, S8 slice S4): the next direct writers call `change_log.record_direct`.

One block per writer: `cytir.index.supersede` and `surface-delta --apply`, both logged only for a run against the twin collection (review 3024). Each pins
WHAT is recorded (op, ids, tool), that the log is preflighted BEFORE the first write, that only a VERIFIED
write is logged, that a log failure is loud, that a dry run writes nothing (no segment, no spool), and that the
log being off changes nothing. Pattern of `tests/test_change_log_direct_wiring.py`."""
from __future__ import annotations

import json
import pathlib

import pytest
from cytir.core import twin as TWIN
from cytir.index import change_log as CL
from cytir.index import supersede as SS
from cytir.index import surface_delta as SD
from tests import test_aw_index_surface_delta as T
from tests.test_aw_dump_shadow_compare import COLL, _point
from tests.test_change_log_direct import world  # noqa: F401  (fixture: prod collection + base dump + log dir)


def _segments(log):
    out = {}
    for f in sorted(pathlib.Path(log).glob("*.jsonl")):
        head, *body = [json.loads(x) for x in f.read_text().splitlines()]
        out[head["pack_id"]] = (head, body)
    return out


def _mine(log, tool):
    return {k: v for k, v in _segments(log).items() if k.startswith(f"direct-{tool}-")}


def _ids(c):
    pts, _ = c.scroll(COLL, limit=100, with_payload=False, with_vectors=False)
    return sorted(str(p.id) for p in pts)


# ============================================================== writer 1: cytir.index.supersede
def _supersede_run(world, monkeypatch, *, log_dir, dry_run=False, twin=COLL, twin_url="mem", extra=()):
    c, _, _, tmp = world
    monkeypatch.setattr(TWIN, "configured_collection", lambda: twin)
    monkeypatch.setattr(TWIN, "configured_url", lambda: twin_url)
    dump = tmp / "dump.jsonl"
    dump.write_text(json.dumps({"id": 30, "payload": _point(3).payload}) + "\n")      # page 3 now lives at id 30
    monkeypatch.setattr(SS, "qdrant_connect", lambda *a, **k: c)
    monkeypatch.setattr(CL, "configured_dir", lambda: log_dir)
    argv = ["supersede", "--in", str(dump), "--qdrant-url", "mem", "--collection", COLL,
            "--assume-complete", "--create-url-index"] + (["--dry-run"] if dry_run else []) + list(extra)
    monkeypatch.setattr("sys.argv", argv)
    return SS.main()


def test_supersede_cli_logs_each_verified_delete_batch_with_the_ids_deleted(world, monkeypatch):
    c, _, log, _ = world
    _supersede_run(world, monkeypatch, log_dir=log)
    assert _ids(c) == ["1", "2", "4"]
    (_, body), = _mine(log, "supersede-cli").values()
    assert [(x["id"], x["op"], x["reason"]) for x in body] == [("3", "tombstone", "superseded")]


def test_supersede_cli_refuses_before_the_delete_when_the_log_dir_is_missing(world, monkeypatch, tmp_path):
    c, *_ = world
    with pytest.raises(SystemExit, match="ABORT"):
        _supersede_run(world, monkeypatch, log_dir=tmp_path / "no-such-log")
    assert _ids(c) == ["1", "2", "3", "4"]


def test_supersede_cli_dry_run_deletes_and_logs_nothing(world, monkeypatch, tmp_path):
    c, _, log, _ = world
    _supersede_run(world, monkeypatch, log_dir=log, dry_run=True)
    assert _ids(c) == ["1", "2", "3", "4"] and _mine(log, "supersede-cli") == {}
    assert not (log / CL.PENDING_DIR).exists()


def test_supersede_cli_log_failure_is_loud_after_the_delete(world, monkeypatch):
    c, _, log, _ = world

    def boom(**k):
        raise CL.ChangeLogError("disk full")
    monkeypatch.setattr(CL, "record_direct", boom)
    with pytest.raises(SystemExit, match="MISSING from the log"):
        _supersede_run(world, monkeypatch, log_dir=log)


def test_supersede_cli_deletes_the_same_with_the_log_off(world, monkeypatch):
    c, _, log, _ = world
    _supersede_run(world, monkeypatch, log_dir=None)
    assert _ids(c) == ["1", "2", "4"] and _mine(log, "supersede-cli") == {}


# ============================================================== writer 2: surface-delta --apply
def _sd_world(tmp_path, monkeypatch, *, completed=True):
    rows = T.rows_for("clean", 6)
    stage, vectors = T.write_shard(tmp_path, "shard-0000", rows, T.random_unit_matrix(6, seed=1))
    T.install_persisted_index(monkeypatch, tmp_path, rows)
    coll = T.FakeCollection({r["point_id"]: {"payload": {"url": r["url"]}, "vector": {"dense": [0.0] * 64}}
                             for r in rows})
    T.install_fake_qdrant(monkeypatch, coll)
    inner = SD.api

    def api(base, path, payload=None, method=""):
        res = inner(base, path, payload, method)
        if "/points/vectors" in path:
            res = {"result": {"status": "completed" if completed else "acknowledged"}}
        return res
    monkeypatch.setattr(SD, "api", api)
    return rows, coll, ["--stage", stage, "--vectors", vectors, "--report", str(tmp_path / "r.json"),
                        "--mrl", "64", "--no-sparse"]


def _sd_run(monkeypatch, argv, log_dir, apply=True, twin="fake_coll", twin_url="http://fake-qdrant.invalid"):
    monkeypatch.setattr(TWIN, "configured_collection", lambda: twin)
    monkeypatch.setattr(TWIN, "configured_url", lambda: twin_url)
    monkeypatch.setattr(CL, "configured_dir", lambda: log_dir)
    argv = list(argv) + (["--apply"] if apply else [])
    return T.run_main(monkeypatch, argv)


def test_surface_delta_logs_the_verified_vector_update_as_replace_and_clears_the_spool(tmp_path, monkeypatch):
    rows, coll, argv = _sd_world(tmp_path, monkeypatch)
    log = tmp_path / "log"
    CL.record_direct(tool="seed", write_id="s0", collection="c", verified_written=True, upserts=["0"],
                     out_dir=log, landed_at="2026-11-01T00:00:00Z")
    assert _sd_run(monkeypatch, argv, log) == 0
    (_, body), = _mine(log, "surface-delta").values()
    assert {x["op"] for x in body} == {"replace"}
    assert sorted(x["id"] for x in body) == sorted(r["point_id"] for r in rows)
    assert list((log / CL.PENDING_DIR).glob("*.json")) == []


def test_surface_delta_refuses_before_the_write_when_the_log_dir_is_missing(tmp_path, monkeypatch):
    rows, coll, argv = _sd_world(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="ABORT"):
        _sd_run(monkeypatch, argv, tmp_path / "no-such-log")
    assert coll.write_calls == 0


def test_surface_delta_logs_nothing_for_an_update_that_did_not_complete(tmp_path, monkeypatch):
    rows, coll, argv = _sd_world(tmp_path, monkeypatch, completed=False)
    log = tmp_path / "log"
    CL.record_direct(tool="seed", write_id="s0", collection="c", verified_written=True, upserts=["0"],
                     out_dir=log, landed_at="2026-11-01T00:00:00Z")
    with pytest.raises(SystemExit, match="did not return `completed`"):
        _sd_run(monkeypatch, argv, log)
    assert _mine(log, "surface-delta") == {}
    assert len(list((log / CL.PENDING_DIR).glob("*.json"))) == 1       # the spool stays: replay-pending can log it


def test_surface_delta_dry_run_writes_no_segment_and_no_spool(tmp_path, monkeypatch):
    rows, coll, argv = _sd_world(tmp_path, monkeypatch)
    log = tmp_path / "no-such-log"                                      # a dry run does not even preflight
    assert _sd_run(monkeypatch, argv, log, apply=False) == 0
    assert coll.write_calls == 0 and not log.exists()


def test_surface_delta_writes_the_same_with_the_log_off(tmp_path, monkeypatch):
    rows, coll, argv = _sd_world(tmp_path, monkeypatch)
    assert _sd_run(monkeypatch, argv, None) == 0
    assert coll.write_calls >= 1


# ============================================================== not the delta source: no line, no spool, no abort
def test_supersede_cli_on_another_collection_touches_the_log_dir_not_at_all(world, monkeypatch, tmp_path):
    c, _, log, _ = world
    before = sorted(p.name for p in log.rglob("*"))
    _supersede_run(world, monkeypatch, log_dir=tmp_path / "no-such-log", twin="aw_the_real_twin")
    assert _ids(c) == ["1", "2", "4"]                       # the delete still ran, with no preflight abort
    assert sorted(p.name for p in log.rglob("*")) == before and not (tmp_path / "no-such-log").exists()


def test_surface_delta_on_another_collection_touches_the_log_dir_not_at_all(tmp_path, monkeypatch):
    rows, coll, argv = _sd_world(tmp_path, monkeypatch)
    log = tmp_path / "no-such-log"
    assert _sd_run(monkeypatch, argv, log, twin="aw_the_real_twin") == 0
    assert coll.write_calls >= 1 and not log.exists()


def test_surface_delta_uncertain_write_ledgers_the_failure_names_replay_pending_and_keeps_the_spool(tmp_path, monkeypatch, capsys):
    rows, coll, argv = _sd_world(tmp_path, monkeypatch, completed=False)
    log = tmp_path / "log"
    CL.record_direct(tool="seed", write_id="s0", collection="c", verified_written=True, upserts=["0"],
                     out_dir=log, landed_at="2026-11-01T00:00:00Z")
    with pytest.raises(SystemExit, match="replay-pending"):
        _sd_run(monkeypatch, argv, log)
    assert "CHANGE-LOG-FAILED surface-delta/" in capsys.readouterr().out
    assert len(list((log / CL.PENDING_DIR).glob("*.json"))) == 1


# ============================================================== what "the twin" is (same resolution as `cyt dump`)
def test_the_twin_is_decided_with_the_twin_env_override_removed(monkeypatch, tmp_path):
    monkeypatch.setattr(CL, "configured_dir", lambda: tmp_path)
    twin_c, twin_u = TWIN.configured_collection(), TWIN.configured_url()
    monkeypatch.setenv("AW_TWIN_COLLECTION", "some_clone")
    monkeypatch.setenv("AW_TWIN_URL", "http://scratch.invalid:1")
    assert CL.log_this_run(twin_c, twin_u) is True                    # the env override does not move the twin
    assert CL.log_this_run("some_clone", "http://scratch.invalid:1") is False


def test_the_twin_name_on_another_spelling_or_qdrant_is_refused_unless_not_the_twin(monkeypatch, tmp_path):
    monkeypatch.setattr(CL, "configured_dir", lambda: tmp_path)
    c = TWIN.configured_collection()
    for url in ("http://scratch.invalid:6333", "http://localhost:1", "127.0.0.1:1"):
        with pytest.raises(SystemExit, match="--not-the-twin"):
            CL.log_this_run(c, url)
        assert CL.log_this_run(c, url, not_the_twin=True) is False
    assert CL.log_this_run(c, TWIN.configured_url() + "/") is True    # trailing slash only
    monkeypatch.setattr(CL, "configured_dir", lambda: None)
    assert CL.log_this_run(c, "http://localhost:1") is False        # log off: nothing to protect


def test_supersede_cli_refuses_a_twin_named_collection_on_another_url_before_any_delete(world, monkeypatch, tmp_path):
    c, _, log, _ = world
    monkeypatch.setattr(TWIN, "configured_url", lambda: "http://the-real-twin.invalid:1")     # args say "mem"
    with pytest.raises(SystemExit, match="REFUSED"):
        _supersede_run(world, monkeypatch, log_dir=log, twin_url="http://the-real-twin.invalid:1")
    assert _ids(c) == ["1", "2", "3", "4"]


def test_supersede_cli_not_the_twin_proceeds_with_no_line_and_no_spool(world, monkeypatch):
    c, _, log, _ = world
    before = sorted(p.name for p in log.rglob("*"))
    _supersede_run(world, monkeypatch, log_dir=log, twin_url="http://the-real-twin.invalid:1", extra=["--not-the-twin"])
    assert _ids(c) == ["1", "2", "4"] and sorted(p.name for p in log.rglob("*")) == before


def test_surface_delta_refuses_a_twin_named_collection_on_another_url_before_any_write(tmp_path, monkeypatch):
    rows, coll, argv = _sd_world(tmp_path, monkeypatch)
    log = _seeded(tmp_path)
    with pytest.raises(SystemExit, match="REFUSED"):
        _sd_run(monkeypatch, argv, log, twin_url="http://the-real-twin.invalid:1")
    assert coll.write_calls == 0


def test_surface_delta_not_the_twin_proceeds_with_no_line_and_no_spool(tmp_path, monkeypatch):
    rows, coll, argv = _sd_world(tmp_path, monkeypatch)
    log = _seeded(tmp_path)
    before = sorted(p.name for p in log.rglob("*"))
    assert _sd_run(monkeypatch, argv + ["--not-the-twin"], log, twin_url="http://the-real-twin.invalid:1") == 0
    assert coll.write_calls >= 1 and sorted(p.name for p in log.rglob("*")) == before


def _seeded(tmp_path):
    log = tmp_path / "log"
    CL.record_direct(tool="seed", write_id="s0", collection="c", verified_written=True, upserts=["0"],
                     out_dir=log, landed_at="2026-11-01T00:00:00Z")
    return log


def test_not_the_twin_with_the_exact_twin_url_and_name_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(CL, "configured_dir", lambda: tmp_path)
    with pytest.raises(SystemExit, match="IS the configured twin"):
        CL.log_this_run(TWIN.configured_collection(), TWIN.configured_url(), not_the_twin=True)
