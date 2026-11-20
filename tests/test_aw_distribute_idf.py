"""The pinned BM25 IDF snapshot of a full dump (task 687, S8): declared in the manifest,
sealed by its sha256, verified by `cyt apply` before Qdrant is touched."""
from __future__ import annotations

import gzip
import json
import math
import time

import pytest

from cytir.core.qdrant import qdrant_client
from cytir.distribute import apply, dump, idf, pack


def _row(id_, sparse):
    return pack.row_to_record(
        id_, [0.1, 0.2, 0.3],
        {"url": f"https://example.org/{id_}", "source_type": "a", "license": "MIT",
         "title": id_}, sparse)


ROWS = [
    _row("00000000-0000-0000-0000-000000000001", ([5, 9], [1.0, 2.0])),
    _row("00000000-0000-0000-0000-000000000002", ([5], [3.0])),
    _row("00000000-0000-0000-0000-000000000003", None),          # no sparse leg
    _row("00000000-0000-0000-0000-000000000004", ([9, 11], [1.0, 1.0])),
]


def _dump(tmp_path, rows=ROWS, name="dump"):
    out = tmp_path / name
    acc = idf.IdfAccumulator()
    shards, total, fps = dump.write_full_shards(iter(rows), out, shard_rows=2, idf=acc)
    manifest = dump.build_manifest(
        dump_id=name, collection="src", dims=3, distance="cosine", shards=shards,
        total_points=total, any_row_has_sparse=True, idf_snapshot=acc.write(out),
        fingerprints=fps, profile=None)
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out, manifest


def test_the_accumulator_counts_rows_with_a_sparse_leg_and_each_term_once_per_row():
    acc = idf.IdfAccumulator()
    for r in ROWS:
        acc.add(r)
    acc.add({"sparse_indices": [5, 5, 5]})       # a repeat inside one row is one posting
    assert acc.n == 4
    assert dict(acc.df) == {5: 3, 9: 2, 11: 1}


def test_the_snapshot_holds_integers_sorted_and_the_manifest_declares_it(tmp_path):
    out, manifest = _dump(tmp_path)
    block = manifest["sparse"]["idf_snapshot"]
    assert block["n"] == 3 and block["vocabulary"] == 3
    assert block["formula"] == idf.FORMULA and block["schema"] == idf.SCHEMA
    with gzip.open(out / idf.FILENAME, "rt") as fh:
        assert fh.read() == "5\t2\n9\t2\n11\t1\n"
    assert manifest["sparse"]["modifier"] == "idf"
    assert idf.verify_snapshot(manifest, out) == block


def test_the_same_corpus_gives_the_same_bytes_later(tmp_path):
    _, first = _dump(tmp_path, name="a")
    time.sleep(1.1)                              # gzip's header mtime must not leak in
    _, second = _dump(tmp_path, name="b")
    assert first["sparse"]["idf_snapshot"]["sha256"] == second["sparse"]["idf_snapshot"]["sha256"]


def test_the_loaded_table_is_the_scoring_formula_over_the_integers(tmp_path):
    out, manifest = _dump(tmp_path)
    table = idf.load_snapshot(manifest, out)
    assert table[5] == pytest.approx(math.log((3 + 1) / (2 + 0.5)))
    assert table[11] == pytest.approx(math.log(4 / 1.5))


def test_a_dump_with_no_sparse_rows_declares_no_snapshot(tmp_path):
    out = tmp_path / "d"
    acc = idf.IdfAccumulator()
    dump.write_full_shards(iter([_row("00000000-0000-0000-0000-000000000001", None)]),
                           out, idf=acc)
    assert acc.write(out) is None
    assert idf.verify_snapshot({"sparse": None, "counts": {"points": 1}}, out) is None


@pytest.mark.parametrize("damage", ["append", "missing", "vocabulary", "schema",
                                    "formula", "n_too_big", "df_over_n"])
def test_apply_refuses_a_snapshot_that_is_not_what_the_manifest_declared(tmp_path, damage):
    out, manifest = _dump(tmp_path)
    block = manifest["sparse"]["idf_snapshot"]
    path = out / idf.FILENAME
    if damage == "append":
        path.write_bytes(path.read_bytes() + b"\0")
    elif damage == "missing":
        path.unlink()
    elif damage == "vocabulary":
        block["vocabulary"] += 1
    elif damage == "schema":
        block["schema"] = "aw.bm25-idf-snapshot.v0"
    elif damage == "formula":
        block["formula"] = "ln(n / df)"
    elif damage == "n_too_big":
        block["n"] = 99
    else:  # a self-consistent file whose df exceeds n: re-sealed so only the range check can catch it
        with gzip.GzipFile(path, "wb", mtime=0) as gz:
            gz.write(b"5\t9\n")
        block.update(sha256=pack.sha256_file(str(path)), vocabulary=1)
    with pytest.raises(apply.ApplyRefusal, match="REFUSED"):
        apply.verify_idf_snapshot(manifest, out)


def test_an_unsorted_snapshot_is_refused(tmp_path):
    out, manifest = _dump(tmp_path)
    path = out / idf.FILENAME
    with gzip.GzipFile(path, "wb", mtime=0) as gz:
        gz.write(b"9\t2\n5\t2\n11\t1\n")
    manifest["sparse"]["idf_snapshot"]["sha256"] = pack.sha256_file(str(path))
    with pytest.raises(apply.ApplyRefusal, match="ascending"):
        apply.verify_idf_snapshot(manifest, out)


def test_apply_creates_the_collection_with_the_manifests_idf_modifier(tmp_path):
    _, manifest = _dump(tmp_path)
    client = qdrant_client(":memory:")
    apply.create_collection(client, "t", manifest, legacy_layout=True)
    params = client.get_collection("t").config.params.sparse_vectors["bm25"]
    assert str(params.modifier).lower().endswith("idf")


def test_a_manifest_that_names_no_modifier_keeps_the_old_unweighted_collection(tmp_path):
    _, manifest = _dump(tmp_path)
    del manifest["sparse"]["modifier"]
    client = qdrant_client(":memory:")
    apply.create_collection(client, "t", manifest, legacy_layout=True)
    assert client.get_collection("t").config.params.sparse_vectors["bm25"].modifier is None


def test_the_apply_state_records_the_snapshot_it_verified(tmp_path):
    out, manifest = _dump(tmp_path)
    state = tmp_path / "state.json"
    assert apply.main(["--legacy-layout", "--manifest", str(out), "--qdrant-url", ":memory:",
                       "--collection", "t", "--allow-unsigned",
                       "--state-file", str(state)]) == 0
    assert json.loads(state.read_text())["idf_snapshot_sha256"] == \
        manifest["sparse"]["idf_snapshot"]["sha256"]
