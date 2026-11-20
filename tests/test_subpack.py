"""Tests for cytir/index/subpack.py (task 680): cutting a sealed pack into
drain-sized sub-packs with symlinks, never copies."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

from cytir.index import subpack as SP


def _make_pack(root: Path, rows_per_shard: list[int]) -> tuple[Path, Path]:
    chunks, vectors = root / "chunks", root / "vectors"
    chunks.mkdir(parents=True)
    vectors.mkdir(parents=True)
    for i, n in enumerate(rows_per_shard):
        tag = f"shard-{i:04d}"
        with gzip.open(chunks / f"{tag}.texts.jsonl.gz", "wt") as fh:
            for j in range(n):
                fh.write(json.dumps({"point_id": f"{tag}-{j}"}) + "\n")
        (vectors / f"{tag}.vectors.f16").write_bytes(b"\0" * (n * 2048))
        (vectors / f"{tag}.vectors.manifest.json").write_text(json.dumps({"n": n}))
        (vectors / f"{tag}.row_ids.jsonl").write_text(
            "".join(json.dumps({"point_id": f"{tag}-{j}"}) + "\n" for j in range(n)))
    return chunks, vectors


def test_plan_subpacks_groups_consecutive_shards_and_counts_rows(tmp_path):
    chunks, vectors = _make_pack(tmp_path, [5, 5, 5, 5, 2])
    subs = SP.plan_subpacks(str(chunks), str(vectors), 2)
    assert [s.tags for s in subs] == [("shard-0000", "shard-0001"),
                                      ("shard-0002", "shard-0003"),
                                      ("shard-0004",)]
    assert [s.rows for s in subs] == [10, 10, 2]
    assert [s.index for s in subs] == [0, 1, 2]


def test_plan_subpacks_covers_every_row_exactly_once(tmp_path):
    chunks, vectors = _make_pack(tmp_path, [7, 7, 7, 3])
    subs = SP.plan_subpacks(str(chunks), str(vectors), 3)
    assert sum(s.rows for s in subs) == 24
    assert [t for s in subs for t in s.tags] == [
        "shard-0000", "shard-0001", "shard-0002", "shard-0003"]


def test_plan_subpacks_refuses_unpaired_shards(tmp_path):
    chunks, vectors = _make_pack(tmp_path, [4, 4])
    (vectors / "shard-0001.vectors.manifest.json").unlink()
    with pytest.raises(SP.SubpackError, match="not the same set"):
        SP.plan_subpacks(str(chunks), str(vectors), 1)


def test_plan_subpacks_refuses_an_empty_pack_and_a_zero_width(tmp_path):
    chunks, vectors = _make_pack(tmp_path, [])
    with pytest.raises(SP.SubpackError, match="no shard"):
        SP.plan_subpacks(str(chunks), str(vectors), 2)
    chunks2, vectors2 = _make_pack(tmp_path / "b", [3])
    with pytest.raises(SP.SubpackError, match=">= 1"):
        SP.plan_subpacks(str(chunks2), str(vectors2), 0)


def test_materialize_subpack_links_the_shards_and_copies_nothing(tmp_path):
    chunks, vectors = _make_pack(tmp_path / "src", [3, 3, 3])
    sub = SP.plan_subpacks(str(chunks), str(vectors), 2)[0]
    out_c, out_v = SP.materialize_subpack(str(chunks), str(vectors), sub,
                                          str(tmp_path / "out/p0"))
    linked = sorted(os.listdir(out_c))
    assert linked == ["shard-0000.texts.jsonl.gz", "shard-0001.texts.jsonl.gz"]
    assert all((Path(out_c) / n).is_symlink() for n in linked)
    assert len(os.listdir(out_v)) == 2 * len(SP.VECTOR_SUFFIXES)
    assert (Path(out_v) / "shard-0000.vectors.f16").resolve() == \
        (vectors / "shard-0000.vectors.f16").resolve()


def test_a_materialized_subpack_reads_back_through_the_drain_readers(tmp_path):
    """The real readers (`promote_sealed_pack.read_records/read_vectors`) see
    exactly the sub-pack's rows and ids, no more."""
    from cytir.index import promote_sealed_pack as PSP

    chunks, vectors = _make_pack(tmp_path / "src", [4, 4, 4])
    sub = SP.plan_subpacks(str(chunks), str(vectors), 2)[0]
    out_c, out_v = SP.materialize_subpack(str(chunks), str(vectors), sub,
                                          str(tmp_path / "out/p0"))
    ids = [r["point_id"] for r in PSP.read_records(out_c)]
    assert len(ids) == 8
    _matrix, vec_ids = PSP.read_vectors(out_v, 256)
    assert sorted(vec_ids) == sorted(ids)


def test_materialize_subpack_is_idempotent(tmp_path):
    chunks, vectors = _make_pack(tmp_path / "src", [2, 2])
    sub = SP.plan_subpacks(str(chunks), str(vectors), 2)[0]
    a = SP.materialize_subpack(str(chunks), str(vectors), sub, str(tmp_path / "o"))
    b = SP.materialize_subpack(str(chunks), str(vectors), sub, str(tmp_path / "o"))
    assert a == b


def test_materialize_subpack_never_overwrites_real_data_with_a_link(tmp_path):
    chunks, vectors = _make_pack(tmp_path / "src", [2])
    sub = SP.plan_subpacks(str(chunks), str(vectors), 1)[0]
    out = tmp_path / "o"
    (out / "chunks").mkdir(parents=True)
    (out / "chunks" / "shard-0000.texts.jsonl.gz").write_bytes(b"real")
    with pytest.raises(SP.SubpackError, match="not a symlink"):
        SP.materialize_subpack(str(chunks), str(vectors), sub, str(out))
