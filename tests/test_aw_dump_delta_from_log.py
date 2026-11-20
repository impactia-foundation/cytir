"""`cyt dump --delta` from change-log segments (task 2504, ruling 2497 option (c); spec `delta-change-log.md` 3.4).

A throwaway in-memory Qdrant stands in for the twin; no live service is touched."""
from __future__ import annotations

import gzip
import json
import pathlib

import pytest
from qdrant_client.http import models as qmodels

from cytir.core import serving_profile as SP
from cytir.distribute import dump, pack

DIM = SP.get(dump.DEFAULT_PROFILE).dense_dim
COLL = "aw_test"
SCHEMA = "aw.change-log.v1"


def _vec(seed: float) -> list[float]:
    v = [0.0] * DIM
    v[0], v[1] = 1.0, seed
    return v


def _point(i: int, seed: float | None = None):
    return qmodels.PointStruct(
        id=i, vector={"dense": _vec(seed if seed is not None else i / 10)},
        payload={"url": f"https://example.org/{i}", "title": f"T{i}", "lang": "en",
                 "source_type": "web_page", "license": "CC0-1.0", "trust_tier": 0.5,
                 "last_seen": "2026-09-20T00:00:00Z"})


def _client():
    from cytir.core.qdrant import qdrant_client
    c = qdrant_client(":memory:")
    c.create_collection(COLL, vectors_config={"dense": qmodels.VectorParams(
        size=DIM, distance=qmodels.Distance.COSINE)})
    return c


AUTO = object()
CUT = {"stamp": "20261101T000000Z", "seq": 1}


def _segment(log_dir: pathlib.Path, stamp: str, pack_id: str, lines: list[dict],
             seq=AUTO) -> pathlib.Path:
    """`seq=AUTO`: one above the highest seq in the directory; `seq=None`: a header without `seq`."""
    f = log_dir / f"{stamp}-{pack_id}.jsonl"
    if seq is AUTO:
        seq = 1 + max([json.loads(x.read_text().splitlines()[0]).get("seq", 0)
                       for x in log_dir.glob("*.jsonl")] or [0])
    header = {"schema": SCHEMA, "pack_id": pack_id, "pack_sha": "ab" * 32,
              "landed_at": "2026-11-01T00:00:00Z", "n": len(lines)}
    if seq is not None:
        header["seq"] = seq
    f.write_text("\n".join(json.dumps(x) for x in [header, *lines]) + "\n")
    if seq is not None:      # the writer's counter, so the log-path tests run with the 2588 check active
        c = log_dir / ".seq"
        c.write_text(f"{max(seq, int(c.read_text()) if c.is_file() else 0)}\n")
    return f


@pytest.fixture
def world(tmp_path):
    """Base dump of ids 1-4 (cut at stamp 0), then the twin moves on: 2 changed, 3 gone, 4 logged but unchanged
    identically (net no-op), 7 new, 8 added then deleted inside the window."""
    c = _client()
    c.upsert(COLL, points=[_point(i) for i in (1, 2, 3, 4)])
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    shards, total, fps = dump.write_full_shards(dump.scroll_rows(c, COLL), base_dir, shard_rows=10)
    manifest = dump.build_manifest(
        dump_id="base", collection=COLL, dims=DIM, distance="cosine", shards=shards,
        total_points=total, any_row_has_sparse=False, fingerprints=fps, profile=None)
    manifest["change_log_segment"] = dict(CUT)
    (base_dir / "manifest.json").write_text(json.dumps(manifest))
    c.upsert(COLL, points=[_point(2, 0.77), _point(7), _point(8)])   # 4 is logged but its stored row is unchanged
    c.delete(COLL, points_selector=qmodels.PointIdsList(points=[3, 8]))
    log = tmp_path / "log"
    log.mkdir()
    _segment(log, "20261101T000000Z", "old", [{"id": "1", "op": "replace"}])   # at the cut: not read
    _segment(log, "20261102T010000Z", "p1", [
        {"id": "7", "op": "add"}, {"id": "2", "op": "replace"}, {"id": "4", "op": "upsert"},
        {"id": "8", "op": "add"}])
    _segment(log, "20261103T010000Z", "p2", [
        {"id": "3", "op": "tombstone", "reason": "superseded", "queue_reason": "canonical_duplicate"},
        {"id": "8", "op": "tombstone", "reason": "aged_out"}])
    return c, manifest, base_dir, log, tmp_path


def _builder(world, **kw):
    c, manifest, base_dir, log, _ = world
    return dump.LogDeltaBuilder(manifest, base_dir, log, client=c, collection=COLL,
                                after=manifest["change_log_segment"], **kw)


def test_log_delta_equals_scroll_delta_bit_for_bit(world):
    c, manifest, base_dir, _, tmp = world
    lb = _builder(world)
    log_rows = list(lb.changed_rows())
    sb = dump.DeltaBuilder(manifest, base_dir)
    scroll_rows = list(sb.changed_rows(dump.scroll_rows(c, COLL)))
    assert [r["id"] for r in log_rows] == ["2", "7"]
    assert log_rows == scroll_rows
    log_t = [(t.id, t.reason) for t in lb.tombstones()]
    assert log_t == [(t.id, t.reason) for t in sb.tombstones({"3": "superseded"})] == [("3", "superseded")]
    for name, rows in (("a", log_rows), ("b", scroll_rows)):
        out = tmp / name
        out.mkdir()
        dump.write_shards(iter(rows), out, shard_rows=10)
    a = sorted(p for p in (tmp / "a").rglob("*") if p.is_file())
    b = sorted(p for p in (tmp / "b").rglob("*") if p.is_file())
    assert [pack.sha256_file(str(x)) for x in a] == [pack.sha256_file(str(x)) for x in b] and a


def test_net_noop_is_dropped_and_counted(world):
    lb = _builder(world)
    assert "4" not in [r["id"] for r in lb.changed_rows()]
    assert lb.noops == 1 and lb.fetched == 3   # 4 (logged, unchanged); the cut segment is skipped by seq


def test_id_added_and_deleted_in_window_is_not_a_tombstone(world):
    lb = _builder(world)
    list(lb.changed_rows())
    assert [t.id for t in lb.tombstones()] == ["3"]


def test_log_naming_an_id_the_source_lacks_is_refused(world):
    c, manifest, base_dir, log, tmp = world
    _segment(log, "20261104T010000Z", "p3", [{"id": "99", "op": "add"}])
    lb = _builder(world)
    with pytest.raises(dump.DeltaRefusal, match="does not hold"):
        list(lb.changed_rows())
    with pytest.raises(dump.DeltaRefusal):
        lb.tombstones()


def test_rebase_in_window_is_refused(world):
    _, _, _, log, _ = world
    _segment(log, "20261104T010000Z", "flip-x", [{"op": "rebase", "why": "flip"}])
    with pytest.raises(dump.DeltaRefusal, match="rebase"):
        _builder(world)


def test_choose_source_falls_back_with_a_reason(world):
    c, manifest, base_dir, log, tmp = world
    kw = dict(log_dir=log, client=c, collection=COLL)
    b, note = dump.choose_delta_source(manifest, base_dir, **kw)
    assert b is not None and "2 segment(s)" in note
    no_cut = {k: v for k, v in manifest.items() if k != "change_log_segment"}
    assert dump.choose_delta_source(no_cut, base_dir, **kw)[0] is None
    assert "change_log_segment" in dump.choose_delta_source(no_cut, base_dir, **kw)[1]
    late = dict(manifest, change_log_segment={"stamp": "20261201T000000Z", "seq": 99})
    assert "no change-log segment after" in dump.choose_delta_source(late, base_dir, **kw)[1]
    assert dump.choose_delta_source(manifest, base_dir, **{**kw, "log_dir": None})[0] is None
    assert dump.choose_delta_source(manifest, base_dir, use_log=False, **kw)[0] is None
    (base_dir / manifest["fingerprints"]["path"]).unlink()
    assert "sidecar is missing" in dump.choose_delta_source(manifest, base_dir, **kw)[1]


def _run(monkeypatch, world, out_name, *extra):
    c, manifest, base_dir, log, tmp = world
    import cytir.core.qdrant as Q
    monkeypatch.setattr(Q, "qdrant_client", lambda *a, **k: c)
    out = tmp / out_name
    rc = dump.main(["--collection", COLL, "--out", str(out), "--delta-base-manifest",
                    str(base_dir / "manifest.json"), "--tombstone-reasons", str(tmp / "r.json"),
                    "--from-change-log", str(log), *extra])
    return rc, out


def test_cli_log_path_writes_manifest_note_and_next_cut_point(monkeypatch, world):
    (world[4] / "r.json").write_text(json.dumps({"3": "superseded"}))
    rc, out = _run(monkeypatch, world, "d1")
    m = json.loads((out / "manifest.json").read_text())
    assert rc == 0 and m["delta_source"]["path"] == "change-log" and m["delta_source"]["no_ops_dropped"] == 1
    assert m["change_log_segment"] == {"stamp": "20261103T010000Z", "seq": 3}
    # T5: the public tombstone file carries `reason` only; `queue_reason` is master-only
    text = gzip.open(out / "tombstones.jsonl.gz", "rt").read()
    assert '"superseded"' in text and "queue_reason" not in text and "canonical_duplicate" not in text


def test_cli_scroll_path_when_forced_says_so(monkeypatch, world):
    (world[4] / "r.json").write_text(json.dumps({"3": "superseded"}))
    rc, out = _run(monkeypatch, world, "d2", "--no-change-log")
    m = json.loads((out / "manifest.json").read_text())
    assert rc == 0 and m["delta_source"]["path"] == "scroll"
    assert m["change_log_segment"] == {"stamp": "20261103T010000Z", "seq": 3}   # the cut point still advances


def test_cli_missing_id_refusal_writes_no_manifest(monkeypatch, world):
    _segment(world[3], "20261104T010000Z", "p3", [{"id": "99", "op": "add"}])
    (world[4] / "r.json").write_text(json.dumps({"3": "superseded"}))
    with pytest.raises(dump.DeltaRefusal):
        _run(monkeypatch, world, "d4")
    assert not (world[4] / "d4" / "manifest.json").exists()


def test_full_dump_records_the_cut_point(monkeypatch, world):
    c, _, _, log, tmp = world
    import cytir.core.qdrant as Q
    monkeypatch.setattr(Q, "qdrant_client", lambda *a, **k: c)
    out = tmp / "full"
    dump.main(["--collection", COLL, "--out", str(out), "--from-change-log", str(log)])
    assert json.loads((out / "manifest.json").read_text())["change_log_segment"] == {"stamp": "20261103T010000Z", "seq": 3}


# ------------------------------------------------------------------ task 2508 (review 2507)
def test_rebase_segment_at_the_cut_is_skipped_by_seq(world):
    c, manifest, base_dir, log, _ = world
    (log / "20261101T000000Z-old.jsonl").unlink()
    _segment(log, "20261101T000000Z", "flip", [{"op": "rebase", "why": "flip"}], seq=1)
    lb = _builder(world)      # no refusal: the rebase has seq == cut.seq
    assert [r["id"] for r in lb.changed_rows()] == ["2", "7"]
    assert lb.segments == ["20261102T010000Z-p1.jsonl", "20261103T010000Z-p2.jsonl"]


def test_same_second_segments_replay_in_seq_order_not_name_order(world):
    _, _, _, log, _ = world
    # A (seq 4) tombstones 1, B (seq 5) re-adds it, same second; B sorts BEFORE A by name
    _segment(log, "20261104T010000Z", "zz-A", [{"id": "1", "op": "tombstone", "reason": "superseded"}])
    _segment(log, "20261104T010000Z", "aa-B", [{"id": "1", "op": "add"}])
    lb = _builder(world)
    list(lb.changed_rows())
    assert [t.id for t in lb.tombstones()] == ["3"]       # no tombstone for the live row 1


def test_tombstone_for_a_row_still_in_the_source_is_refused(world):
    _, _, _, log, _ = world
    _segment(log, "20261104T010000Z", "p3", [{"id": "1", "op": "tombstone", "reason": "superseded"}])
    with pytest.raises(dump.DeltaRefusal, match="still in the source"):
        list(_builder(world).changed_rows())


def test_segment_without_seq_is_refused_naming_it(world):
    _, _, _, log, _ = world
    _segment(log, "20261104T010000Z", "noseq", [{"id": "7", "op": "add"}], seq=None)
    with pytest.raises(dump.DeltaRefusal, match="20261104T010000Z-noseq.jsonl.*no `seq`"):
        _builder(world)


def test_seq_gap_is_refused_naming_the_segment(world):
    _, _, _, log, _ = world
    _segment(log, "20261104T010000Z", "gap", [{"id": "7", "op": "add"}], seq=5)   # 4 is missing
    with pytest.raises(dump.DeltaRefusal, match="not contiguous.*20261104T010000Z-gap.jsonl"):
        _builder(world)


def test_log_reset_under_a_live_cut_is_refused_not_silently_skipped(world):
    """Review 2509 A: cut at seq 50, log wiped, new segments seq 2..51 must not read as only 51."""
    c, manifest, base_dir, log, _ = world
    for f in log.glob("*.jsonl"):
        f.unlink()
    manifest = dict(manifest, change_log_segment={"stamp": "20261101T000000Z", "seq": 50})
    kw = dict(log_dir=log, client=c, collection=COLL)
    _segment(log, "20261105T010000Z", "n01", [{"id": "1", "op": "add"}], seq=1)
    with pytest.raises(dump.DeltaRefusal, match="counter went back"):
        dump.choose_delta_source(manifest, base_dir, **kw)
    # an empty log still falls back to scroll
    (log / "20261105T010000Z-n01.jsonl").unlink()
    assert dump.choose_delta_source(manifest, base_dir, **kw)[0] is None
    _segment(log, "20261105T010000Z", "n01", [{"id": "1", "op": "add"}], seq=1)
    for i in range(2, 52):
        _segment(log, "20261105T010000Z", f"n{i:02d}", [{"id": "7", "op": "add"}], seq=i)
    with pytest.raises(dump.DeltaRefusal, match="20261105T010000Z-n\\d+.jsonl is stamped after the cut.*counter went back"):
        dump.choose_delta_source(manifest, base_dir, **kw)


def test_change_log_error_is_a_refused_line_not_a_traceback(monkeypatch, world, capsys):
    c, manifest, base_dir, log, tmp = world
    import cytir.core.qdrant as Q
    monkeypatch.setattr(Q, "qdrant_client", lambda *a, **k: c)
    (log / "20261104T010000Z-bad.jsonl").write_text("not json\n")
    rc = dump.cli(["--collection", COLL, "--out", str(tmp / "o"), "--delta-base-manifest",
                   str(base_dir / "manifest.json"), "--from-change-log", str(log)])
    assert rc == 1 and "REFUSED:" in capsys.readouterr().err


def test_mixed_dense_width_is_refused_on_any_path():
    seen: list[int] = []
    dump._check_width(seen, {"dense_dims": 256, "id": "1"})
    with pytest.raises(ValueError, match="mixed dense width"):
        dump._check_width(seen, {"dense_dims": 128, "id": "2"})


def test_cli_missing_newest_segment_refuses_without_scroll_fallback(monkeypatch, world, capsys):
    """Task 2588: the writer's counter is ahead of the newest segment read: REFUSED, nothing written,
    and the scroll path is never taken."""
    c, manifest, base_dir, log, tmp = world
    (log / ".seq").write_text("4\n")          # segments hold seq 1-3; seq 4 landed but is not here
    scrolled = []
    real = c.scroll          # spy on the client: the scroll generator is built but never iterated
    monkeypatch.setattr(c, "scroll", lambda *a, **k: scrolled.append(1) or real(*a, **k))
    import cytir.core.qdrant as Q
    monkeypatch.setattr(Q, "qdrant_client", lambda *a, **k: c)
    out = tmp / "d5"
    rc = dump.cli(["--collection", COLL, "--out", str(out), "--delta-base-manifest",
                   str(base_dir / "manifest.json"), "--from-change-log", str(log)])
    assert rc == 1 and "newest segment is missing" in capsys.readouterr().err
    assert not scrolled
    assert not out.exists() or not (out / "manifest.json").exists()
