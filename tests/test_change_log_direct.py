"""`change_log.record_direct` (task 2849, ruling 75): the log call of a writer that is not the drain.

Same segment shape as the drain's (reader, verify, completeness check accept it); logged AFTER the
verified write (spec 3.3); a crash between write and log is a missed entry that spec 3.7's shadow
compare catches. Harness: the throwaway in-memory Qdrant of tests/test_aw_dump_shadow_compare.py."""
from __future__ import annotations

import json

import pytest
from qdrant_client.http import models as qmodels

from cytir.distribute import dump, shadow_compare as SC
from cytir.index import change_log as CL
from tests.test_aw_dump_shadow_compare import COLL, DIM, _point

KW = dict(tool="twin-patch", write_id="b0001", collection="aw_twin")


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_a_direct_write_lands_one_segment_in_the_drain_line_shape(tmp_path):
    path = CL.record_direct(**KW, verified_written=True, upserts=["u1"], replaces=["r1"],
                            payload_patches=["p1"], tombstones=[{"id": "t1", "reason": "superseded"}],
                            out_dir=tmp_path, landed_at="2026-11-16T10:00:00Z")
    assert path.name == "20261116T100000Z-direct-twin-patch-b0001.jsonl"
    head, *body = _lines(path)
    assert head["schema"] == CL.SCHEMA and head["pack_id"] == "direct-twin-patch-b0001"
    assert head["n"] == 4 and head["seq"] == 1 and len(head["pack_sha"]) == 64
    assert body == [{"id": "u1", "op": "upsert"}, {"id": "r1", "op": "replace"},
                    {"id": "p1", "op": "replace"},
                    {"id": "t1", "op": "tombstone", "reason": "superseded", "queue_reason": "superseded"}]
    h2, b2 = CL.read_segment(path)                       # the reader the delta builder uses
    assert h2 == head and b2 == body
    counts = CL.verify(path, {"pack_id": head["pack_id"], "prod": {"written": 3, "tombstones_requested": 1}})
    assert counts["upsert"] == 1 and counts["replace"] == 2 and counts["tombstone"] == 1
    assert CL.assert_segments_complete(tmp_path) == 1    # the completeness check treats it as complete


def test_seq_is_shared_with_the_drain_writer(tmp_path):
    CL.record_landing({"pack_id": "p1", "status": "LANDED", "pack_sha": "ab" * 32},
                      {"upserts": ["i1"]}, tmp_path, landed_at="2026-11-16T09:00:00Z")
    d = CL.record_direct(**KW, verified_written=True, upserts=["u1"], out_dir=tmp_path,
                         landed_at="2026-11-16T10:00:00Z")
    assert _lines(d)[0]["seq"] == 2 and CL.assert_segments_complete(tmp_path) == 2


def test_a_retry_of_the_same_write_relands_the_same_file_and_bytes(tmp_path):
    a = CL.record_direct(**KW, verified_written=True, upserts=["u2", "u1"], out_dir=tmp_path,
                         landed_at="2026-11-16T10:00:00Z")
    first = a.read_bytes()
    b = CL.record_direct(**KW, verified_written=True, upserts=["u1", "u2"], out_dir=tmp_path,
                         landed_at="2026-11-16T11:00:00Z")     # a later retry: name and landed_at are kept
    assert b == a and b.read_bytes().splitlines()[0] == first.splitlines()[0]
    assert len(list(tmp_path.glob("*.jsonl"))) == 1 and CL.next_seq(tmp_path) == 2


def test_a_write_not_verified_landed_is_refused_and_nothing_is_written(tmp_path):
    for bad in (False, None, 1):
        with pytest.raises(CL.ChangeLogError, match="not verified"):
            CL.record_direct(**KW, verified_written=bad, upserts=["u1"], out_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_an_unmapped_tombstone_reason_is_refused_with_no_segment(tmp_path):
    with pytest.raises(CL.ChangeLogError):
        CL.record_direct(**KW, verified_written=True, out_dir=tmp_path,
                         tombstones=[{"id": "t1", "reason": "because"}])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("tool,wid", [("a/b", "1"), ("t", "../x"), ("", "1"), ("t", "")])
def test_a_name_that_is_not_file_name_safe_is_refused(tmp_path, tool, wid):
    with pytest.raises(CL.ChangeLogError):
        CL.record_direct(tool=tool, write_id=wid, collection="c", verified_written=True,
                         upserts=["u1"], out_dir=tmp_path)


def test_log_off_and_empty_write_return_none(tmp_path, monkeypatch):
    from cytir.core import paths
    monkeypatch.delenv("AW_CHANGE_LOG_DIR", raising=False)
    paths.reset_settings_cache()
    assert CL.record_direct(**KW, verified_written=True, upserts=["u1"]) is None
    assert CL.record_direct(**KW, verified_written=True, out_dir=tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_the_dir_defaults_to_the_change_log_dir_setting(tmp_path, monkeypatch):
    from cytir.core import paths
    monkeypatch.setenv("AW_CHANGE_LOG_DIR", str(tmp_path / "log"))
    paths.reset_settings_cache()
    p = CL.record_direct(**KW, verified_written=True, upserts=["u1"])
    assert p is not None and p.parent == tmp_path / "log"


def test_a_dry_run_writes_no_segment_and_no_directory(tmp_path):
    from cytir.index import dry_run as DRY
    sub = tmp_path / "sub"
    with pytest.raises(DRY.DryRunWriteRefused):
        with DRY.guard(True):
            CL.record_direct(**KW, verified_written=True, upserts=["u1"], out_dir=sub)
    assert not sub.exists()


# ------------------------------------------------- the delta reader and the 3.7 bypass check
@pytest.fixture
def world(tmp_path):
    from cytir.core.qdrant import qdrant_client
    c = qdrant_client(":memory:")
    c.create_collection(COLL, vectors_config={"dense": qmodels.VectorParams(
        size=DIM, distance=qmodels.Distance.COSINE)})
    c.upsert(COLL, points=[_point(i) for i in (1, 2, 3, 4)])
    base = tmp_path / "base"
    base.mkdir()
    shards, total, fps = dump.write_full_shards(dump.scroll_rows(c, COLL), base, shard_rows=10)
    manifest = dump.build_manifest(
        dump_id="base", collection=COLL, dims=DIM, distance="cosine", shards=shards,
        total_points=total, any_row_has_sparse=False, fingerprints=fps, profile=None)
    manifest["change_log_segment"] = {"stamp": "20261101T000000Z", "seq": 1}
    (base / "manifest.json").write_text(json.dumps(manifest))
    log = tmp_path / "log"
    CL.record_direct(tool="seed", write_id="cut", collection=COLL, verified_written=True,
                     upserts=["1"], out_dir=log, landed_at="2026-11-01T00:00:00Z")   # seq 1 = the cut
    return c, base, log, tmp_path


def _direct_write(c):
    """A live writer's mutation: add 7, patch 2, delete 3."""
    c.upsert(COLL, points=[_point(7), _point(2, 0.77)])
    c.delete(COLL, points_selector=qmodels.PointIdsList(points=[3]))


def _compare(world):
    c, base, log, tmp = world
    return SC.run(client=c, collection=COLL, base_manifest_path=str(base / "manifest.json"),
                  log_dir=str(log), out_dir=tmp / "out", stamp="20261104T000000Z")


def test_a_direct_write_logged_after_the_write_gives_a_delta_equal_to_the_scroll_delta(world):
    c, _, log, _ = world
    _direct_write(c)
    CL.record_direct(tool="live-upsert", write_id="w1", collection=COLL, verified_written=True,
                     upserts=["7"], payload_patches=["2"], tombstones=[{"id": "3", "reason": "superseded"}],
                     out_dir=log, landed_at="2026-11-02T01:00:00Z")
    rc, rep = _compare(world)
    assert rc == 0 and rep["verdict"] == "EQUAL"
    assert rep["legs"]["log"]["rows"] == rep["legs"]["scroll"]["rows"] == 2
    assert rep["legs"]["log"]["tombstones"] == rep["legs"]["scroll"]["tombstones"] == 1


def test_a_crash_between_the_write_and_the_log_is_caught_by_the_shadow_compare(world):
    c, _, log, _ = world
    c.upsert(COLL, points=[_point(9)])            # an earlier write that did log
    CL.record_direct(tool="live-upsert", write_id="w0", collection=COLL, verified_written=True,
                     upserts=["9"], out_dir=log, landed_at="2026-11-02T00:00:00Z")
    _direct_write(c)                              # the process dies here: no record_direct call
    rc, rep = _compare(world)
    assert rc == 1 and rep["verdict"] != "EQUAL"
    assert {"7", "2"} <= set(rep["diff"]["only_in_scroll"]["first"])
    assert rep["diff_counts"]["only_in_log"] == 0


def test_a_crash_with_no_segment_at_all_after_the_cut_is_refused_not_passed(world):
    c, _, _, _ = world
    _direct_write(c)
    with pytest.raises(SC.ShadowRefusal, match="no change-log segment"):
        _compare(world)


def test_a_crash_after_the_log_before_a_retry_leaves_one_segment_and_an_equal_delta(world):
    c, _, log, _ = world
    _direct_write(c)
    for _ in range(2):                            # the writer is resumed and logs again
        CL.record_direct(tool="live-upsert", write_id="w1", collection=COLL, verified_written=True,
                         upserts=["7"], payload_patches=["2"],
                         tombstones=[{"id": "3", "reason": "superseded"}], out_dir=log,
                         landed_at="2026-11-02T01:00:00Z")
    assert len(list(log.glob("*live-upsert-w1.jsonl"))) == 1
    rc, rep = _compare(world)
    assert rc == 0 and rep["verdict"] == "EQUAL"


# ------------------------------------------------- review 2853: concurrency and write-id reuse
def _land(args):
    d, i = args
    if i % 2:
        CL.record_direct(tool="t", write_id=f"r{i}", collection="c", verified_written=True,
                         upserts=[str(i)], out_dir=d)
    else:
        CL.record_landing({"pack_id": f"p{i}", "status": "LANDED", "pack_sha": "ab" * 32},
                          {"upserts": [str(i)]}, d)


def test_concurrent_writers_get_distinct_gap_free_seqs(tmp_path):
    import multiprocessing as mp
    for trial in range(3):
        d = tmp_path / f"t{trial}"
        with mp.get_context("fork").Pool(8) as pool:
            pool.map(_land, [(d, i) for i in range(16)])
        seqs = sorted(json.loads(f.read_text().splitlines()[0])["seq"] for f in d.glob("*.jsonl"))
        assert seqs == list(range(1, 17)), seqs
        assert CL.assert_segments_complete(d) == 16


def test_the_same_write_id_with_different_content_is_refused_and_the_segment_kept(tmp_path):
    a = CL.record_direct(**KW, verified_written=True, upserts=["a"], out_dir=tmp_path)
    before = a.read_bytes()
    with pytest.raises(CL.ChangeLogError, match="different content"):
        CL.record_direct(**KW, verified_written=True, upserts=["b", "c"], out_dir=tmp_path)
    assert a.read_bytes() == before and len(list(tmp_path.glob("*.jsonl"))) == 1


def test_a_dry_run_creates_no_seq_lock_and_no_directory(tmp_path):
    from cytir.index import dry_run as DRY
    sub = tmp_path / "sub"
    calls = (lambda: CL.record_landing({"pack_id": "p", "status": "LANDED", "pack_sha": "ab"},
                                       {"upserts": ["i"]}, sub),
             lambda: CL.record_direct(**KW, verified_written=True, upserts=["u"], out_dir=sub))
    for call in calls:
        with pytest.raises(DRY.DryRunWriteRefused):
            with DRY.guard(True):
                call()
    assert not sub.exists()
