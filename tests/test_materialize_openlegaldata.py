"""Real-record tests for the OpenLegalData materializer (task 633).

Fixture rows (`tests/fixtures/openlegaldata/task633_openlegaldata_sample_rows.json`) are
REAL rows pulled live from the sealed `dump-20260520` acquisition on the build box
(`.../de-openlegaldata-v1/acquisition/original/parquet/dump-20260520/
train-00000-of-00054.parquet`, task 489) -- the first 25 rows by id, plus 3
real rows with an empty `markdown_content` (a genuine, pre-existing
OpenLegalData extraction gap; see the module docstring in
`materialize_openlegaldata.py`). Nothing here is synthesized.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

import cytir.extract.openlegaldata as subject
from cytir.corpus import persist_l1_records as L1

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "openlegaldata" / "task633_openlegaldata_sample_rows.json"


def load_fixture() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_has_both_populated_and_empty_rows():
    rows = load_fixture()
    assert len(rows) >= 20
    populated = [r for r in rows if (r.get("markdown_content") or "").strip()]
    empty = [r for r in rows if not (r.get("markdown_content") or "").strip()]
    assert populated and empty, "fixture must exercise both branches"


def test_content_is_html_markdown_content_is_markdown():
    row = load_fixture()[0]
    assert row["content"].lstrip().startswith("<")
    assert "##" in row["markdown_content"] or ":   " in row["markdown_content"]


def test_build_url_matches_live_verified_template():
    row = load_fixture()[0]
    assert row["slug"] == "bgh-2018-09-05-2-str-45417"
    url = subject.build_url(row["slug"])
    assert url == "https://de.openlegaldata.io/case/bgh-2018-09-05-2-str-45417"


def test_build_url_refuses_missing_slug():
    with pytest.raises(subject.MaterializeError):
        subject.build_url("")
    with pytest.raises(subject.MaterializeError):
        subject.build_url(None)


def test_build_markdown_real_court_date_ecli_file_number():
    row = load_fixture()[0]
    assert row["ecli"] == "ECLI:DE:BGH:2018:050918U2STR454.17.0"
    assert row["file_number"] == "2 StR 454/17"
    assert row["date"] == "2018-09-05"
    md = subject.build_markdown(row)
    assert md is not None
    first_line = md.splitlines()[0]
    assert first_line.startswith("# ")
    assert "Bundesgerichtshof" in first_line
    assert row["file_number"] in first_line
    assert row["date"] in first_line
    assert row["ecli"] in first_line
    # the source markdown_content body is preserved, not rewritten
    assert row["markdown_content"].strip() in md


def test_build_markdown_returns_none_for_empty_body():
    empty_rows = [r for r in load_fixture()
                  if not (r.get("markdown_content") or "").strip()]
    assert empty_rows
    for row in empty_rows:
        assert subject.build_markdown(row) is None


def test_build_markdown_row_license_is_exact_urhg5_never_odbl():
    row = load_fixture()[0]
    out = subject.build_markdown_row(row, parent_archive={"relative_path": "x", "sha256": "0" * 64})
    assert out["license"] == "public_domain_official_work (UrhG §5)"
    assert "ODbL" not in out["license"]
    assert out["license"] != "ODbL"


def test_build_markdown_row_identity_and_anchor_fields():
    row = load_fixture()[0]
    parent_archive = {
        "relative_path": "original/parquet/dump-20260520/train-00000-of-00054.parquet",
        "sha256": "7cb627748e5249d702c5292cd612b9e2f0991acc0ea48b0b5eef0c3530d3399b",
    }
    out = subject.build_markdown_row(row, parent_archive=parent_archive)
    assert out["url"] == "https://de.openlegaldata.io/case/bgh-2018-09-05-2-str-45417"
    assert out["parent_archive"] == parent_archive
    assert out["source_record_id"] == row["id"]
    assert out["markdown_sha256"] == hashlib.sha256(out["markdown"].encode()).hexdigest()
    assert out["lang"] == "de"
    assert out["source_metadata"]["ecli"] == row["ecli"]


def test_build_markdown_row_raises_for_empty_body():
    empty_row = next(r for r in load_fixture()
                     if not (r.get("markdown_content") or "").strip())
    with pytest.raises(subject.MaterializeError):
        subject.build_markdown_row(empty_row, parent_archive={"relative_path": "x", "sha256": "0" * 64})


def test_identity_line_survives_as_h1_for_breadcrumb_prefix():
    """The identity line must be the document's H1 so chunk.generic's
    breadcrumb prefix carries it into every chunk of a long decision."""
    row = load_fixture()[0]
    line = subject.build_identity_line(row)
    assert line.startswith("# ")
    assert line.count("\n") == 0


def test_build_l1_record_round_trips_through_persist_l1_records(tmp_path):
    row = load_fixture()[0]
    parent_archive = {"relative_path": "x", "sha256": "0" * 64}
    md_row = subject.build_markdown_row(row, parent_archive=parent_archive)
    l1_record = subject.build_l1_record(md_row)

    written = L1.write_l1_records(
        [l1_record], release="test-task633", source="de-openlegaldata-test",
        origin="test-fixture:task633_openlegaldata_sample_rows.json",
        root=str(tmp_path))
    assert len(written) == 1
    canonical_url = next(iter(written))
    file_path = written[canonical_url]
    lines = list(L1.read_jsonl(file_path))
    assert len(lines) == 1
    persisted = lines[0]
    assert persisted["schema"] == "aw.l1-markdown.v1"
    assert persisted["markdown"] == md_row["markdown"]
    assert persisted["markdown_sha256"] == md_row["markdown_sha256"]
    # re-hash on read: the store's own sha must match the text it carries
    assert persisted["markdown_sha256"] == hashlib.sha256(
        persisted["markdown"].encode()).hexdigest()
    assert persisted["license"] == subject.LICENSE


def test_read_shard_rows_matches_json_fixture_shape(tmp_path):
    """DuckDB-vs-plain-JSON parity: the parquet reader must not silently
    reshape a column (e.g. flattening the `court` struct)."""
    import duckdb

    rows = load_fixture()[:5]
    # Build a real parquet file from the fixture rows via DuckDB's own
    # struct/JSON handling, then confirm the module's reader recovers the
    # same court sub-fields -- exercised on real values, not a mock schema.
    con = duckdb.connect()
    court = rows[0]["court"]
    parquet_path = tmp_path / "shard.parquet"
    con.execute(
        f"COPY (SELECT * FROM read_json_auto('{FIXTURE}')) "
        f"TO '{parquet_path}' (FORMAT PARQUET)")
    con.close()
    read_back = subject.read_shard_rows(parquet_path)
    assert len(read_back) == len(load_fixture())
    first = next(r for r in read_back if r["id"] == rows[0]["id"])
    assert first["court"]["name"] == court["name"]
    assert first["court"]["jurisdiction"] == court["jurisdiction"]
    assert first["ecli"] == rows[0]["ecli"]


def test_materialize_end_to_end_seals_a_real_run(tmp_path):
    """Full materialize() over a synthetic-but-real acquisition layout built
    from the real fixture rows -- proves the stage/RUN.json sealing chain and
    the quarantine path both work against genuine content."""
    import duckdb
    from cytir.acquire.artifact_io import atomic_publish, sha256_file

    acquisition_root = tmp_path / "acquisition"
    parquet_dir = acquisition_root / "original" / "parquet" / "dump-20260520"
    parquet_dir.mkdir(parents=True)
    shard_path = parquet_dir / "train-00000-of-00054.parquet"
    con = duckdb.connect()
    con.execute(
        f"COPY (SELECT * FROM read_json_auto('{FIXTURE}')) "
        f"TO '{shard_path}' (FORMAT PARQUET)")
    con.close()

    receipt = {
        "schema": "aw.stage-receipt.v1", "records": 1,
        "bytes": shard_path.stat().st_size,
        "format": "exact-openlegaldata-hf-parquet+dataset-info+HF-HEAD+range-transport",
        "logical_sha256": sha256_file(shard_path),
        "deterministic_encoding": True,
        "producer": {"id": "acquire_openlegaldata_hf_dump",
                     "version": "openlegaldata-hf-dump-acquisition.v1"},
        "artifacts": [{
            "uri": "artifact://original/parquet/dump-20260520/train-00000-of-00054.parquet",
            "bytes": shard_path.stat().st_size, "sha256": sha256_file(shard_path)}],
    }
    receipt_path = acquisition_root / "stages" / "original" / "receipt.json"
    atomic_publish(receipt_path, (json.dumps(receipt) + "\n").encode())
    original_stage = {
        "bytes": receipt["bytes"], "deterministic_encoding": True,
        "format": receipt["format"], "logical_sha256": receipt["logical_sha256"],
        "producer": receipt["producer"],
        "receipt": {"uri": "artifact://stages/original/receipt.json",
                    "bytes": receipt_path.stat().st_size,
                    "sha256": sha256_file(receipt_path)},
        "records": 1,
    }
    acq_run = {
        "schema": subject.ACQUISITION_SCHEMA, "status": "complete",
        "stages": {"original": original_stage},
    }
    acq_run_path = acquisition_root / "RUN.json"
    atomic_publish(acq_run_path, (json.dumps(acq_run) + "\n").encode())

    artifact_root = tmp_path / "corpus"
    output_root = artifact_root / "stages" / "openlegaldata-test"
    result = subject.materialize(
        artifact_root=artifact_root, acquisition_root=acquisition_root,
        acquisition_run_path=acq_run_path, shard_indices=[0],
        output_root=output_root)

    assert result["schema"] == subject.RUN_SCHEMA
    assert result["status"] == "complete"
    rows = load_fixture()
    populated = [r for r in rows if (r.get("markdown_content") or "").strip()]
    empty = [r for r in rows if not (r.get("markdown_content") or "").strip()]
    assert result["counts"]["markdown_documents"] == len(populated)
    assert result["counts"]["quarantined_empty_body"] == len(empty)
    assert result["counts"]["source_rows"] == len(rows)

    markdown_stage = result["stages"]["markdown"]
    assert markdown_stage["records"] == len(populated)
    shard_uri = markdown_stage["receipt"]["uri"]
    # every artifact URI in the sealed output must resolve under artifact_root
    stage_receipt_path = artifact_root / shard_uri.removeprefix("artifact://")
    assert stage_receipt_path.is_file()
    stage_receipt = json.loads(stage_receipt_path.read_text())
    data_shard = artifact_root / stage_receipt["artifacts"][0]["uri"].removeprefix("artifact://")
    with gzip.open(data_shard, "rt", encoding="utf-8") as fh:
        emitted = [json.loads(line) for line in fh]
    assert len(emitted) == len(populated)
    urls = {row["url"] for row in emitted}
    assert all(url.startswith("https://de.openlegaldata.io/case/") for url in urls)
    licenses = {row["license"] for row in emitted}
    assert licenses == {subject.LICENSE}

    # quarantine is real and non-empty given the fixture's 3 empty rows
    assert result["quarantine"]["empty_body"]["records"] == len(empty)


def test_materialize_refuses_a_second_run_into_nonempty_output(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "marker").write_text("x")
    with pytest.raises(subject.MaterializeError):
        subject.materialize(
            artifact_root=tmp_path, acquisition_root=tmp_path,
            acquisition_run_path=tmp_path / "does-not-exist.json",
            shard_indices=[0], output_root=tmp_path / "out")
