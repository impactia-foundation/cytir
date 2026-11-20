# Task 650 (migration S8): moved to tests/ (2026-10-04) -- where the public cut ships and re-runs it.
"""Tests for cytir.distribute.dump — `cyt dump` (task 650, S8).

No live Qdrant/twin is used: `scroll_rows` is tested against a small fake
client, and `write_shards`/`build_manifest`/`DeltaBuilder` are tested as
pure functions over synthetic rows.
"""
from __future__ import annotations

import json

import pytest
from qdrant_client.http import models as qmodels

from cytir.distribute import dump, pack

# --------------------------------------------------------------- twin guard
def test_twin_default_is_accepted():
    dump.assert_qdrant_url_is_not_prod(
        dump.DEFAULT_TWIN_QDRANT_URL, i_know_this_is_not_the_twin=False)


@pytest.mark.parametrize("url", ["http://127.0.0.1:16333", "http://localhost:16333"])
def test_known_prod_endpoint_always_refused(url):
    with pytest.raises(dump.ProdRefusal):
        dump.assert_qdrant_url_is_not_prod(url, i_know_this_is_not_the_twin=True)
    with pytest.raises(dump.ProdRefusal):
        dump.assert_qdrant_url_is_not_prod(url, i_know_this_is_not_the_twin=False)


def test_non_twin_url_needs_the_explicit_flag():
    url = "http://127.0.0.1:9999"
    with pytest.raises(dump.ProdRefusal):
        dump.assert_qdrant_url_is_not_prod(url, i_know_this_is_not_the_twin=False)
    dump.assert_qdrant_url_is_not_prod(url, i_know_this_is_not_the_twin=True)


# ------------------------------------------------------------------- scroll
class _FakePoint:
    def __init__(self, id_, payload, vector):
        self.id = id_
        self.payload = payload
        self.vector = vector


class _FakeClient:
    def __init__(self, points, batch_boundaries=None):
        self._points = points

    def scroll(self, collection, limit, offset, with_payload, with_vectors):
        start = offset or 0
        batch = self._points[start:start + limit]
        next_offset = start + limit if start + limit < len(self._points) else None
        return batch, next_offset


def _general_point(i):
    return _FakePoint(
        f"id-{i}",
        {"url": f"https://example.org/{i}", "title": f"T{i}", "lang": "en",
         "source_type": "web_page", "license": "CC0-1.0", "trust_tier": 0.5,
         "last_seen": "2026-09-20T00:00:00Z"},
        {"dense": [0.1, 0.2], "bm25": qmodels.SparseVector(indices=[1, 2], values=[0.5, 0.5])})


def _entity_card_point(i):
    return _FakePoint(
        f"card-{i}",
        {"url": f"https://example.ch/entity/{i}", "title": f"Firma {i} AG",
         "snippet": "leaked-name-in-live-snippet Hans Muster",
         "dense_text": "Firma AG — Beratung. Registered "
                       "signatories/officers (commercial register): "
                       "Hans Muster.",
         "lang": "de", "source_type": "entity_card", "license": "unknown",
         "trust_tier": 0.85, "last_seen": "2026-09-20T00:00:00Z",
         "register_id": "CHE-1", "register_scheme": "zefix",
         "resolve_confidence": 0.9},
        {"dense": [0.3, 0.4], "bm25": qmodels.SparseVector(indices=[9], values=[1.0])})


def test_scroll_rows_drops_sparse_and_names_for_entity_cards():
    client = _FakeClient([_general_point(1), _entity_card_point(1)])
    rows = list(dump.scroll_rows(client, "aw_e8"))
    assert len(rows) == 2
    general, card = rows
    assert general["sparse_indices"] == [1, 2]
    assert card["sparse_indices"] is None
    assert card["sparse_values"] is None
    assert "Hans Muster" not in card["payload"]


def test_scroll_rows_refuses_point_with_no_dense_vector():
    bad = _FakePoint("id-x", {"url": "https://x"}, {})
    client = _FakeClient([bad])
    with pytest.raises(pack.ProjectionRefusal):
        list(dump.scroll_rows(client, "aw_e8"))


# ------------------------------------------------------------------- shards
def test_write_shards_batches_and_hashes(tmp_path):
    rows = [pack.row_to_record(f"id-{i}", [0.1, 0.2],
                               {"url": f"https://x/{i}", "source_type": "a",
                                "license": "MIT"}, None)
            for i in range(5)]
    shards, total, fps = dump.write_full_shards(iter(rows), tmp_path, shard_rows=2)
    assert total == 5
    assert len(shards) == 3  # 2 + 2 + 1
    assert shards[0]["rows"] == 2
    assert shards[-1]["rows"] == 1
    for s in shards:
        got_sha = pack.sha256_file(str(tmp_path / s["path"]))
        assert got_sha == s["sha256"]


# ----------------------------------------------------------------- manifest
def test_build_manifest_shape_is_valid():
    manifest = dump.build_manifest(
        dump_id="test-1", collection="aw_e8", dims=128, distance="cosine",
        shards=[{"path": "shards/shard-00000.parquet", "rows": 1,
                 "sha256": "a" * 64, "bytes": 10}],
        total_points=1, any_row_has_sparse=True, profile=None)
    pack.validate_manifest(manifest)
    assert manifest["embedder"]["dims"] == 128
    assert manifest["license"]["database"] == "ODC-By-1.0"
    assert manifest["sparse"]["vector_name"] == "bm25"


def test_build_manifest_no_sparse_block_when_no_row_carries_one():
    manifest = dump.build_manifest(
        dump_id="test-1", collection="aw_e8", dims=128, distance="cosine",
        shards=[], total_points=0, any_row_has_sparse=False, profile=None)
    assert manifest["sparse"] is None


# -------------------------------------------------------------- delta diff
def _row(id_, url, *, dense=(0.1, 0.2), license_="MIT"):
    return pack.row_to_record(id_, list(dense),
                              {"url": url, "source_type": "a", "license": license_},
                              None)


def test_delta_builder_finds_new_changed_and_gone(tmp_path):
    base_rows = [_row("id-1", "https://a"), _row("id-2", "https://b")]
    shards, _, fps = dump.write_full_shards(iter(base_rows), tmp_path, shard_rows=10)
    base_manifest = dump.build_manifest(
        dump_id="base", collection="aw_e8", dims=2, distance="cosine",
        shards=shards, total_points=2, any_row_has_sparse=False, fingerprints=fps, profile=None)

    # current snapshot: id-1 unchanged, id-2 changed (new dense), id-3 new,
    # id-2's sibling... (id-2 stays present so nothing is "gone" here)
    current = [_row("id-1", "https://a"),
               _row("id-2", "https://b", dense=(0.9, 0.9)),
               _row("id-3", "https://c")]
    builder = dump.DeltaBuilder(base_manifest, tmp_path)
    changed = list(builder.changed_rows(iter(current)))
    assert {r["id"] for r in changed} == {"id-2", "id-3"}
    assert builder.tombstones({}) == []  # nothing gone, no reasons needed


def test_delta_builder_refuses_gone_id_with_no_reason(tmp_path):
    base_rows = [_row("id-1", "https://a"), _row("id-2", "https://b")]
    shards, _, fps = dump.write_full_shards(iter(base_rows), tmp_path, shard_rows=10)
    base_manifest = dump.build_manifest(
        dump_id="base", collection="aw_e8", dims=2, distance="cosine",
        shards=shards, total_points=2, any_row_has_sparse=False, fingerprints=fps, profile=None)

    current = [_row("id-1", "https://a")]  # id-2 vanished
    builder = dump.DeltaBuilder(base_manifest, tmp_path)
    list(builder.changed_rows(iter(current)))
    with pytest.raises(dump.DeltaRefusal):
        builder.tombstones({})


def test_delta_builder_accepts_declared_reason(tmp_path):
    base_rows = [_row("id-1", "https://a"), _row("id-2", "https://b")]
    shards, _, fps = dump.write_full_shards(iter(base_rows), tmp_path, shard_rows=10)
    base_manifest = dump.build_manifest(
        dump_id="base", collection="aw_e8", dims=2, distance="cosine",
        shards=shards, total_points=2, any_row_has_sparse=False, fingerprints=fps, profile=None)

    current = [_row("id-1", "https://a")]
    builder = dump.DeltaBuilder(base_manifest, tmp_path)
    list(builder.changed_rows(iter(current)))
    tombstones = builder.tombstones({"id-2": "takedown"})
    assert len(tombstones) == 1
    assert tombstones[0].id == "id-2"
    assert tombstones[0].reason == "takedown"


# ------------------------------------------- delta cost (task 687, S8 step B)
def _full_base(tmp_path, rows):
    shards, total, fps = dump.write_full_shards(iter(rows), tmp_path, shard_rows=10)
    return dump.build_manifest(
        dump_id="base", collection="aw_e8", dims=2, distance="cosine",
        shards=shards, total_points=total, any_row_has_sparse=False,
        fingerprints=fps, profile=None)


def test_delta_builder_never_reads_base_shards(tmp_path, monkeypatch):
    """The delta's cost is the sidecar read plus the rows it hashes from the
    current snapshot -- the unchanged base shards are never opened."""
    base_rows = [_row(f"id-{i:02d}", f"https://a/{i}") for i in range(30)]
    base_manifest = _full_base(tmp_path, base_rows)

    def boom(*a, **k):
        raise AssertionError("DeltaBuilder read a base shard")

    monkeypatch.setattr(pack, "read_parquet_shard", boom)
    current = [_row(f"id-{i:02d}", f"https://a/{i}",
                    dense=(0.9, 0.9) if i in (3, 7) else (0.1, 0.2))
               for i in range(30)]
    builder = dump.DeltaBuilder(base_manifest, tmp_path)
    changed = list(builder.changed_rows(iter(current)))
    assert {r["id"] for r in changed} == {"id-03", "id-07"}
    assert builder.tombstones({}) == []


def test_delta_builder_refuses_base_without_fingerprints(tmp_path):
    base_manifest = _full_base(tmp_path, [_row("id-1", "https://a")])
    del base_manifest["fingerprints"]
    with pytest.raises(dump.DeltaRefusal, match="fingerprints"):
        dump.DeltaBuilder(base_manifest, tmp_path)


def test_delta_builder_refuses_tampered_fingerprint_sidecar(tmp_path):
    base_manifest = _full_base(tmp_path, [_row("id-1", "https://a")])
    with open(tmp_path / pack.FINGERPRINTS_FILENAME, "ab") as fh:
        fh.write(b"x")
    with pytest.raises(ValueError, match="sha256"):
        dump.DeltaBuilder(base_manifest, tmp_path)


# ------------------- fingerprint exactness + bounded memory (task 687 refactor)
def _old_4dp_rule(row):
    """The pre-#484 / #484 rule, verbatim: sha256 of payload + dense rounded to 4 dp."""
    import hashlib
    return hashlib.sha256(json.dumps(
        {"payload": row["payload"], "dense": [round(float(x), 4) for x in row["dense"]]},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _as_rest(dense):
    """What Qdrant's REST scroll hands Python: the float32's shortest decimal text, parsed."""
    import numpy as np
    return [float(str(np.float32(x))) for x in dense]


def _as_exact(dense):
    """What Parquet or gRPC hands Python: the exact double of the float32."""
    import numpy as np
    return [float(np.float32(x)) for x in dense]


def test_fingerprint_is_independent_of_transport_on_a_rounding_tie():
    # -0.06305 is a real value from the twin (task 687 refactor, id 00005b2c-...):
    # as REST text it parses to a double on the 4 dp tie, as the float32 it sits just past it.
    rest = _row("id-1", "https://a", dense=_as_rest([-0.06305, 0.5]))
    exact = _row("id-1", "https://a", dense=_as_exact([-0.06305, 0.5]))
    assert rest["dense"] != exact["dense"]
    assert pack.row_fingerprint(rest) == pack.row_fingerprint(exact)
    # the old rule disagrees on this very pair: the cause of the 5,564 false rows at 1M
    assert _old_4dp_rule(rest) != _old_4dp_rule(exact)


def test_fingerprint_changes_when_the_vector_changes_by_one_ulp():
    import numpy as np
    a = _row("id-1", "https://a", dense=[0.5, 0.25])
    nxt = float(np.nextafter(np.float32(0.5), np.float32(1)))
    b = _row("id-1", "https://a", dense=[nxt, 0.25])
    assert pack.row_fingerprint(a) != pack.row_fingerprint(b)


def test_fingerprint_covers_the_sparse_leg():
    pay = {"url": "https://a", "license": "MIT"}
    a = pack.row_to_record("id-1", [0.1, 0.2], pay, ([1, 2], [0.5, 0.25]))
    b = pack.row_to_record("id-1", [0.1, 0.2], pay, ([1, 3], [0.5, 0.25]))
    c = pack.row_to_record("id-1", [0.1, 0.2], pay, ([1, 2], [0.5, 0.75]))
    d = pack.row_to_record("id-1", [0.1, 0.2], pay, None)
    assert len({pack.row_fingerprint(r) for r in (a, b, c, d)}) == 4


def test_fingerprint_writer_refuses_unsorted_and_repeated_ids(tmp_path):
    w = pack.FingerprintWriter(str(tmp_path / "f.gz"))
    w.add("id-2", "0" * 16)
    with pytest.raises(pack.FingerprintOrderError):
        w.add("id-1", "0" * 16)
    w2 = pack.FingerprintWriter(str(tmp_path / "g.gz"))
    w2.add("id-2", "0" * 16)
    with pytest.raises(pack.FingerprintOrderError):
        w2.add("id-2", "0" * 16)


def test_point_id_key_is_qdrants_scroll_order():
    import uuid

    from cytir.core.qdrant import qdrant_client
    c = qdrant_client(":memory:")
    c.create_collection("x", vectors_config={"dense": qmodels.VectorParams(
        size=2, distance=qmodels.Distance.COSINE)})
    ids = [5, 300, 20, 7, 1000000] + [str(uuid.uuid4()) for _ in range(40)]
    c.upsert("x", points=[qmodels.PointStruct(id=i, vector={"dense": [1.0, 0.0]})
                          for i in ids])
    scrolled = [str(p.id) for p in c.scroll("x", limit=100)[0]]
    assert scrolled == sorted(scrolled, key=pack.point_id_key)
    assert len(scrolled) == len(ids)


def test_delta_builder_refuses_a_snapshot_out_of_point_id_order(tmp_path):
    base = _full_base(tmp_path, [_row("id-01", "https://a"), _row("id-02", "https://b")])
    builder = dump.DeltaBuilder(base, tmp_path)
    with pytest.raises(dump.DeltaRefusal, match="order"):
        list(builder.changed_rows(iter([_row("id-02", "https://b"), _row("id-01", "https://a")])))


def test_delta_builder_refuses_a_repeated_snapshot_id(tmp_path):
    base = _full_base(tmp_path, [_row("id-01", "https://a")])
    builder = dump.DeltaBuilder(base, tmp_path)
    with pytest.raises(dump.DeltaRefusal):
        list(builder.changed_rows(iter([_row("id-01", "https://a"), _row("id-01", "https://a")])))


def test_delta_builder_refuses_a_base_under_another_fingerprint_algorithm(tmp_path):
    base = _full_base(tmp_path, [_row("id-1", "https://a")])
    base["fingerprints"]["algo"] = "aw-row-fp-v1"
    with pytest.raises(dump.DeltaRefusal, match="algorithm"):
        dump.DeltaBuilder(base, tmp_path)


def test_delta_builder_tombstones_need_the_whole_snapshot(tmp_path):
    base = _full_base(tmp_path, [_row("id-1", "https://a")])
    with pytest.raises(dump.DeltaRefusal):
        dump.DeltaBuilder(base, tmp_path).tombstones({})


def test_merge_join_finds_adds_changes_and_gone_at_every_position(tmp_path):
    ids = [f"id-{i:03d}" for i in range(0, 60, 2)]            # base: even ids
    base = _full_base(tmp_path, [_row(i, f"https://a/{i}") for i in ids])
    cur = {i: _row(i, f"https://a/{i}") for i in ids}
    gone = {ids[0], ids[7], ids[-1]}                          # first, middle, last
    for g in gone:
        del cur[g]
    changed_ids = {ids[1], ids[8]}
    for c in changed_ids:
        cur[c] = _row(c, f"https://a/{c}", dense=(0.9, 0.9))
    added = {"id-000x", "id-031", "id-999"}                   # before, between, after
    for a in added:
        cur[a] = _row(a, f"https://a/{a}")
    snapshot = [cur[k] for k in sorted(cur, key=pack.point_id_key)]
    builder = dump.DeltaBuilder(base, tmp_path)
    out = {r["id"] for r in builder.changed_rows(iter(snapshot))}
    assert out == changed_ids | added
    assert {t.id for t in builder.tombstones({g: "aged_out" for g in gone})} == gone


def test_delta_builder_peak_memory_does_not_grow_with_the_corpus(tmp_path):
    """The old `{id: fp}` map was ~0.66 GB per 1M ids. The merge-join holds one
    sidecar line and one row: 8x the corpus must not cost 2x the peak."""
    import tracemalloc

    def peak(n, sub):
        d = tmp_path / sub
        d.mkdir()
        w = pack.FingerprintWriter(str(d / pack.FINGERPRINTS_FILENAME))
        for i in range(n):
            w.add(f"{i:012d}", pack.row_fingerprint(_row(f"{i:012d}", "https://a")))
        base = {"fingerprints": w.close()}
        builder = dump.DeltaBuilder(base, d)
        tracemalloc.start()
        n_changed = sum(1 for _ in builder.changed_rows(
            _row(f"{i:012d}", "https://a") for i in range(n)))
        _, p = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert n_changed == 0 and builder.tombstones({}) == []
        return p

    small, big = peak(3_000, "s"), peak(24_000, "b")
    assert big < 2 * small + 100_000, (small, big)


def test_sidecar_gzip_bytes_do_not_depend_on_the_clock(tmp_path, monkeypatch):
    """Task 2339: two dumps of the same rows must have the same sidecar bytes. `gzip.open`
    stamps the wall clock in the header; the server-side rehearsal saw two re-dumps of one
    collection differ only there."""
    import time

    ids = ["00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"]
    shas = []
    for i, t in enumerate((1_000_000_000, 1_700_000_000)):
        monkeypatch.setattr(time, "time", lambda t=t: t)
        w = pack.FingerprintWriter(str(tmp_path / f"f{i}.gz"))
        for id_ in ids:
            w.add(id_, "ab" * 16)
        w.close()
        n, tomb = pack.write_tombstones([], str(tmp_path / f"t{i}.gz"))
        shas.append((pack.sha256_file(str(tmp_path / f"f{i}.gz")), tomb))
    assert shas[0] == shas[1]
    for name in ("f0.gz", "t0.gz"):
        assert (tmp_path / name).read_bytes()[4:8] == b"\0\0\0\0"   # header mtime


def test_write_shards_reports_resident_structures_without_changing_the_shards(tmp_path):
    from cytir.distribute.idf import IdfAccumulator
    rows = [_row(f"{i}", f"https://x.ch/{i}") for i in range(1, 6)]
    for r in rows:
        r["sparse_indices"], r["sparse_values"] = [1, 2, i_term(r)], [1.0, 1.0, 1.0]
    lines: list[str] = []
    quiet = dump.write_shards(iter(rows), tmp_path / "a", shard_rows=2)
    loud = dump.write_shards(iter(rows), tmp_path / "b", shard_rows=2,
                             idf=IdfAccumulator(), log=lines.append)
    assert [s["sha256"] for s in quiet[0]] == [s["sha256"] for s in loud[0]]
    assert lines and "ids_held=0" in lines[-1] and "idf_terms=" in lines[-1]


def i_term(row):
    return int(row["id"]) + 10
