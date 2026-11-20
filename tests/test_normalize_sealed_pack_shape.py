#!/usr/bin/env python3
"""Real-record guards for `normalize_sealed_pack_shape` (task 641).

Fixture `tests/testdata/task641_cc_tranche1/` is TWO ACTUAL rows of the
`cc_authority_frontier_w7` tranche-1 pack (`ch_law_courts-pack-0001`), pulled
from the build box 2026-09-23: the real `text/` shard is the re-embed stage's own
{point_id, text} rows (exactly what was embedded), `meta/` is the same two
rows' full metadata from the ORIGINAL flat pack, and `vectors/` is a real
2-row slice of the actual sealed 1024-d fp16 vectors for those exact ids
(same bytes, sliced, sha256 recomputed over the slice). Not a mock: this is
the artefact this project really produced, at 2-row scale.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import shutil
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent

from cytir.index import normalize_sealed_pack_shape as N  # noqa: E402

FIXTURE = HERE / "testdata" / "task641_cc_tranche1"
TEXT = str(FIXTURE / "text")
META = str(FIXTURE / "meta")
VECTORS = str(FIXTURE / "vectors")

REAL_IDS = ["00001183-f9d7-5a9a-8034-d72feb6d10a4",
           "0000155d-0c85-57ad-8ab5-d5d8b7c3fa0c"]


# --------------------------------------------------------------- the CC case
def test_target_ids_reads_the_real_vector_slice():
    ids = N.target_ids(VECTORS, mrl=256)
    assert ids == REAL_IDS


def test_load_id_map_reads_real_text_only_rows():
    m = N.load_id_map(TEXT)
    assert set(m) == set(REAL_IDS)
    for row in m.values():
        assert set(row.keys()) == {"point_id", "text"}, (
            "the re-embed stage shape carries nothing else -- that is the "
            "whole defect this module exists to fix")


def test_load_id_map_reads_real_metadata_rows():
    m = N.load_id_map(META)
    row = m[REAL_IDS[0]]
    assert row["url"]
    assert row["domain"]
    assert row["source_type"] == "official"
    assert "license" not in row or row["license"] is None


def test_merge_combines_real_text_and_real_metadata_by_id():
    merged, report = N.merge(metadata_from=[META], text_from=TEXT,
                             vectors_dir=VECTORS, mrl=256)
    assert report["merged_rows"] == 2
    assert report["metadata_rows_unused"] == 0
    ids = {r["point_id"] for r in merged}
    assert ids == set(REAL_IDS)
    for row in merged:
        # metadata survived
        assert row["url"] and row["domain"]
        # text is the RE-EMBED shard's text (what was actually embedded),
        # never the original pack's -- even though in this real pair they
        # happen to be identical, the merge must be provably TEXT-FROM-
        # priority, so assert against the text source directly rather than
        # trusting equality.
        text_row = N.load_id_map(TEXT)[row["point_id"]]
        assert row["text"] == text_row["text"]


def test_merge_refuses_a_vector_id_with_no_metadata(tmp_path):
    # Drop one row from the metadata source.
    thin_meta = tmp_path / "meta"
    thin_meta.mkdir()
    rows = list(N._shard_rows(META))
    with gzip.open(thin_meta / "shard-0000.texts.jsonl.gz", "wt") as fh:
        fh.write(json.dumps(rows[0]) + "\n")  # only one of the two ids
    with pytest.raises(N.NormalizeError, match="no metadata row"):
        N.merge(metadata_from=[str(thin_meta)], text_from=TEXT,
               vectors_dir=VECTORS, mrl=256)


def test_merge_refuses_a_vector_id_with_no_text(tmp_path):
    thin_text = tmp_path / "text"
    thin_text.mkdir()
    rows = list(N._shard_rows(TEXT))
    with gzip.open(thin_text / "shard-0000.texts.jsonl.gz", "wt") as fh:
        fh.write(json.dumps(rows[0]) + "\n")
    with pytest.raises(N.NormalizeError, match="no row in --text-from"):
        N.merge(metadata_from=[META], text_from=str(thin_text),
               vectors_dir=VECTORS, mrl=256)


def test_write_shard_round_trips_through_read_records(tmp_path):
    merged, _report = N.merge(metadata_from=[META], text_from=TEXT,
                              vectors_dir=VECTORS, mrl=256)
    out_path = N.write_shard(merged, tmp_path / "out")
    assert out_path.exists()
    reread = list(N._shard_rows(str(tmp_path / "out")))
    assert {r["point_id"] for r in reread} == set(REAL_IDS)


# ------------------------------------------------------- the OLD (select) case
def test_merge_with_no_text_from_selects_from_metadata_alone(tmp_path):
    """The OLD shape: one over-wide source has both text and metadata; the
    vectors dir defines a NARROWER id subset than the source covers. No
    --text-from needed -- the source's own text is used."""
    wide = tmp_path / "wide"
    wide.mkdir()
    rows = list(N._shard_rows(META))  # real metadata rows carry no `text`...
    # ...so build a wide source that DOES (metadata + a text field), matching
    # OLD's real single-shard shape (full metadata AND text in one row).
    wide_rows = [{**r, "text": "some real-shaped body text"} for r in rows]
    # Add an extra row the vectors set will NOT reference, to prove the
    # "wider source than the target" case this docstring describes.
    wide_rows.append({**wide_rows[0], "point_id": "extra-id-not-in-vectors"})
    with gzip.open(wide / "shard-0000.texts.jsonl.gz", "wt") as fh:
        for row in wide_rows:
            fh.write(json.dumps(row) + "\n")

    merged, report = N.merge(metadata_from=[str(wide)], text_from=None,
                             vectors_dir=VECTORS, mrl=256)
    assert report["merged_rows"] == 2
    assert report["metadata_rows_unused"] == 1
    assert report["metadata_rows_unused_sample"] == ["extra-id-not-in-vectors"]
    assert {r["point_id"] for r in merged} == set(REAL_IDS)
    assert all(r["text"] == "some real-shaped body text" for r in merged)


def test_merge_refuses_duplicate_id_within_one_source(tmp_path):
    dup = tmp_path / "dup"
    dup.mkdir()
    rows = list(N._shard_rows(META))
    with gzip.open(dup / "shard-0000.texts.jsonl.gz", "wt") as fh:
        fh.write(json.dumps(rows[0]) + "\n")
        fh.write(json.dumps(rows[0]) + "\n")  # same point_id twice
    with pytest.raises(N.NormalizeError, match="duplicate point_id"):
        N.load_id_map(str(dup))


def test_load_id_map_reads_plain_uncompressed_jsonl(tmp_path):
    """CC tranche 1's REAL original packs are plain `<cell>-pack-NNNN.jsonl`
    -- never gzipped, never named `shard-*`. This is the shape that made an
    earlier version of this module (which delegated to `promote_sealed_pack.
    read_records`, gz-shard-only) raise `no shard-*.texts.jsonl.gz under
    ...` on the real artifact at task 641's first live rehearsal."""
    plain = tmp_path / "plain"
    plain.mkdir()
    rows = list(N._shard_rows(META))  # real rows, re-written uncompressed
    (plain / "ch_law_courts-pack-0001.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    m = N.load_id_map(str(plain))
    assert set(m) == set(REAL_IDS)


def test_merge_refuses_overlapping_metadata_sources(tmp_path):
    copy_dir = tmp_path / "meta-copy"
    shutil.copytree(META, copy_dir)
    with pytest.raises(N.NormalizeError, match="already seen"):
        N.merge(metadata_from=[META, str(copy_dir)], text_from=TEXT,
               vectors_dir=VECTORS, mrl=256)


# --------------------------------- lang backfill (orchestrator ruling,
# --------------------------------- 2026-09-23, CC tranche 1, task 652)
def test_merge_without_backfill_lang_leaves_an_empty_lang_row_empty(tmp_path):
    """Default OFF: a real fixture row with `lang` stripped comes out of
    `merge()` exactly as it went in -- normalizing shape and backfilling a
    payload field are two different operations."""
    thin_meta = tmp_path / "meta-nolang"
    thin_meta.mkdir()
    rows = list(N._shard_rows(META))
    rows[1] = {**rows[1], "lang": None}  # the real schweizermonat.ch/de row
    with gzip.open(thin_meta / "shard-0000.texts.jsonl.gz", "wt") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    merged, report = N.merge(metadata_from=[str(thin_meta)], text_from=TEXT,
                             vectors_dir=VECTORS, mrl=256)
    assert "lang_backfill" not in report
    by_id = {r["point_id"]: r for r in merged}
    assert by_id[REAL_IDS[1]]["lang"] is None


def test_merge_with_backfill_lang_fills_the_real_empty_row(tmp_path, monkeypatch):
    """The real fixture's second row is the actual `schweizermonat.ch`
    German-language page (task 641 fixture docstring) -- strip its declared
    `lang` and prove `--backfill-lang` fills it back in from the SAME
    detector the language gate uses, while leaving the already-declared
    English row (`lfi.ch`, index 0) untouched."""
    from cytir.acquire import fetch_process as FP  # noqa: PLC0415

    def fake_detect(text, url, default):
        if "schweizermonat.ch" in (url or ""):
            return ("de", True)
        raise AssertionError(
            f"detector called for a row that should not need it: {url}")
    monkeypatch.setattr(FP, "detect_lang", fake_detect)

    thin_meta = tmp_path / "meta-nolang"
    thin_meta.mkdir()
    rows = list(N._shard_rows(META))
    assert rows[1]["url"].endswith("schweizermonat.ch/kryptos-kaufen")
    rows[1] = {**rows[1], "lang": None}
    with gzip.open(thin_meta / "shard-0000.texts.jsonl.gz", "wt") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    merged, report = N.merge(metadata_from=[str(thin_meta)], text_from=TEXT,
                             vectors_dir=VECTORS, mrl=256,
                             backfill_lang=True)
    assert report["lang_backfill"]["filled"] == 1
    assert report["lang_backfill"]["already_declared"] == 1
    by_id = {r["point_id"]: r for r in merged}
    assert by_id[REAL_IDS[0]]["lang"] == "en"
    assert "lang_source" not in by_id[REAL_IDS[0]]  # was already declared
    assert by_id[REAL_IDS[1]]["lang"] == "de"
    from cytir.gates import mechanical_content_check as MCC  # noqa: E402
    assert by_id[REAL_IDS[1]]["lang_source"] == MCC.lang_source_tag()
