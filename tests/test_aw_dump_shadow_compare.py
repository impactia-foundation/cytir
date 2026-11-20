"""`cyt dump shadow-compare` (task 2603; ruling 68 (2); spec delta-change-log.md 3.7).

Same harness as tests/test_aw_dump_delta_from_log.py: a throwaway in-memory Qdrant, a FULL base
cut at a change-log stamp, then segments that land changes."""
from __future__ import annotations

import json

import pytest
from qdrant_client.http import models as qmodels

from cytir.core import serving_profile as SP
from cytir.distribute import dump, shadow_compare as SC

DIM = SP.get(dump.DEFAULT_PROFILE).dense_dim
COLL = "aw_test"
CUT = {"stamp": "20261101T000000Z", "seq": 1}


def _vec(seed):
    v = [0.0] * DIM
    v[0], v[1] = 1.0, seed
    return v


def _point(i, seed=None):
    return qmodels.PointStruct(
        id=i, vector={"dense": _vec(seed if seed is not None else i / 10)},
        payload={"url": f"https://example.org/{i}", "title": f"T{i}", "lang": "en",
                 "source_type": "web_page", "license": "CC0-1.0", "trust_tier": 0.5,
                 "last_seen": "2026-09-20T00:00:00Z"})


def _segment(log_dir, stamp, pack_id, lines):
    seq = 1 + max([json.loads(x.read_text().splitlines()[0]).get("seq", 0)
                   for x in log_dir.glob("*.jsonl")] or [0])
    header = {"schema": "aw.change-log.v1", "pack_id": pack_id, "pack_sha": "ab" * 32,
              "landed_at": "2026-11-01T00:00:00Z", "n": len(lines), "seq": seq}
    f = log_dir / f"{stamp}-{pack_id}.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in [header, *lines]) + "\n")
    (log_dir / ".seq").write_text(f"{seq}\n")
    return f


def _rewrite(f, keep=lambda line: True, extra=()):
    """A writer that missed (or doubled) a line wrote a consistent header: fix `n`, as it would."""
    head, *body = f.read_text().splitlines()
    body = [x for x in body if keep(x)] + [json.dumps(x) for x in extra]
    h = json.loads(head)
    h["n"] = len(body)
    f.write_text("\n".join([json.dumps(h), *body]) + "\n")


@pytest.fixture
def world(tmp_path):
    from cytir.core.qdrant import qdrant_client
    c = qdrant_client(":memory:")
    c.create_collection(COLL, vectors_config={"dense": qmodels.VectorParams(
        size=DIM, distance=qmodels.Distance.COSINE)})
    c.upsert(COLL, points=[_point(i) for i in (1, 2, 3, 4)])
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    shards, total, fps = dump.write_full_shards(dump.scroll_rows(c, COLL), base_dir, shard_rows=10)
    manifest = dump.build_manifest(
        dump_id="base", collection=COLL, dims=DIM, distance="cosine", shards=shards,
        total_points=total, any_row_has_sparse=False, fingerprints=fps, profile=None)
    manifest["change_log_segment"] = dict(CUT)
    (base_dir / "manifest.json").write_text(json.dumps(manifest))
    c.upsert(COLL, points=[_point(2, 0.77), _point(7), _point(8)])
    c.delete(COLL, points_selector=qmodels.PointIdsList(points=[3, 8]))
    log = tmp_path / "log"
    log.mkdir()
    _segment(log, "20261101T000000Z", "old", [{"id": "1", "op": "replace"}])
    p1 = _segment(log, "20261102T010000Z", "p1", [
        {"id": "7", "op": "add"}, {"id": "2", "op": "replace"}, {"id": "4", "op": "upsert"},
        {"id": "8", "op": "add"}])
    _segment(log, "20261103T010000Z", "p2", [
        {"id": "3", "op": "tombstone", "reason": "superseded"},
        {"id": "8", "op": "tombstone", "reason": "aged_out"}])
    return c, base_dir, log, tmp_path, p1


def _run(world, **kw):
    c, base_dir, log, tmp, _ = world
    return SC.run(client=c, collection=COLL, base_manifest_path=str(base_dir / "manifest.json"),
                  log_dir=str(log), out_dir=tmp / "out", stamp="20261104T000000Z", **kw)


def test_equal_and_report_written(world):
    rc, rep = _run(world)
    assert rc == 0 and rep["verdict"] == "EQUAL"
    assert rep["legs"]["log"]["rows"] == rep["legs"]["scroll"]["rows"] == 2
    assert rep["legs"]["log"]["tombstones"] == rep["legs"]["scroll"]["tombstones"] == 1
    assert rep["legs"]["log"]["no_ops_dropped"] == 1 and rep["segment_range"]["last"]["seq"] == 3
    saved = json.loads((world[3] / "out" / "shadow-compare.20261104T000000Z.json").read_text())
    assert saved["base_manifest_sha256"] and "seconds" in saved["legs"]["scroll"]
    assert (world[3] / "out" / "20261104T000000Z" / "log").is_dir()


def test_planted_missed_entry_is_only_in_scroll(world):
    p1 = world[4]
    _rewrite(p1, keep=lambda x: '"id": "7"' not in x)
    rc, rep = _run(world)
    assert rc == 1 and rep["diff"]["only_in_scroll"]["first"] == ["7"]
    assert rep["diff_counts"]["only_in_log"] == 0


def test_missed_tombstone_is_only_in_scroll(world):
    p2 = next(world[2].glob("*-p2.jsonl"))
    _rewrite(p2, keep=lambda x: '"id": "3"' not in x)
    rc, rep = _run(world)
    assert rc == 1 and rep["diff"]["only_in_scroll"]["first"] == ["3"]
    assert rep["scroll_reason_fallback_ids"] == ["3"]


def test_double_entry_is_harmless_and_equal(world):
    p1 = world[4]
    _rewrite(p1, extra=[{"id": "7", "op": "add"}])
    rc, rep = _run(world)
    assert rc == 0 and rep["verdict"] == "EQUAL"


def test_stale_fingerprint_is_fingerprint_differs(world):
    c = world[0]
    real = c.retrieve

    def stale(*a, **k):                       # the log leg reads a different row than the scroll sees
        pts = real(*a, **k)
        for p in pts:
            if p.payload and str(p.id) == "7":
                p.payload = {**p.payload, "title": "STALE"}
        return pts
    c.retrieve = stale
    rc, rep = _run(world)
    assert rc == 1 and rep["diff"]["fingerprint_differs"]["first"] == ["7"]


def test_independent_reason_is_compared(world):
    rc, rep = _run(world, tombstone_reasons={"3": "takedown"})
    assert rc == 1 and rep["diff"]["reason_differs"]["first"] == ["3"]


def test_pure_compare_classes_and_bound():
    log = {"1": ("add", "a", None), "2": ("replace", "x", None), "3": ("tombstone", None, "aged_out")}
    scroll = {"2": ("replace", "y", None), "3": ("replace", "z", None), "4": ("add", "b", None)}
    d = SC.compare(log, scroll, compare_reasons=False)
    assert d["only_in_log"] == ["1"] and d["only_in_scroll"] == ["4"]
    assert d["fingerprint_differs"] == ["2"] and d["op_differs"] == ["3"]
    many = {str(i): ("add", "a", None) for i in range(1, 60)}
    b = SC.bound(SC.compare(many, {}, compare_reasons=False))
    assert b["only_in_log"]["count"] == 59 and len(b["only_in_log"]["first"]) == 20


def test_missing_newest_segment_is_refused_without_scroll(world):
    c = world[0]
    (world[2] / ".seq").write_text("4\n")
    scrolled = []
    real = c.scroll
    c.scroll = lambda *a, **k: scrolled.append(1) or real(*a, **k)
    with pytest.raises(SC.ShadowRefusal, match="newest segment is missing"):
        _run(world)
    assert not scrolled


def test_cli_exit_codes(monkeypatch, world, capsys):
    c, base_dir, log, tmp, _ = world
    import cytir.core.qdrant as Q
    monkeypatch.setattr(Q, "qdrant_client", lambda *a, **k: c)
    argv = ["--collection", COLL, "--delta-base-manifest", str(base_dir / "manifest.json"),
            "--from-change-log", str(log), "--out", str(tmp / "o")]
    assert SC.main(argv) == 0
    (log / ".seq").write_text("9\n")
    assert SC.main(argv) == 2 and "REFUSED" in capsys.readouterr().err


def test_dump_verb_routes_shadow_compare(monkeypatch):
    import cytir.cli as cli
    seen = []
    monkeypatch.setattr(SC, "main", lambda a: seen.append(a) or 0)
    assert cli._dump_main(["shadow-compare", "--x"]) == 0 and seen == [["--x"]]
