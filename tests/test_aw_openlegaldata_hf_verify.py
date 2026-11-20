import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cytir.acquire import openlegaldata_hf as ACQ
from cytir.acquire import stage_receipt as SS
from cytir.acquire import openlegaldata_hf as subject


COURT = pa.struct([
    ("id", pa.int64()), ("name", pa.string()), ("slug", pa.string()),
    ("city", pa.int64()), ("state", pa.int64()), ("jurisdiction", pa.string()),
    ("level_of_appeal", pa.string()),
])


def _row(i: int, *, content="Urteil " + "x" * 50, ecli="ECLI:DE:X:2026:1"):
    return {
        "id": i, "slug": f"case-{i}",
        "court": {"id": 1, "name": "Amtsgericht Test", "slug": "ag-test",
                  "city": 1, "state": 1, "jurisdiction": "Zivilrecht",
                  "level_of_appeal": "AG"},
        "file_number": f"1 C {i}/26", "date": "2026-01-0" + str(1 + i % 9),
        "created_date": "2026-01-01T00:00:00", "updated_date": "2026-01-01T00:00:00",
        "type": "urteil", "ecli": ecli, "content": content,
        "markdown_content": content, "reference_markers": "",
    }


def _write_shard(path: Path, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows, schema=pa.schema([
        ("id", pa.int64()), ("slug", pa.string()), ("court", COURT),
        ("file_number", pa.string()), ("date", pa.string()),
        ("created_date", pa.string()), ("updated_date", pa.string()),
        ("type", pa.string()), ("ecli", pa.string()), ("content", pa.string()),
        ("markdown_content", pa.string()), ("reference_markers", pa.string()),
    ]))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _build_fixture_run(root: Path, *, rows_per_shard: int, shard_count: int,
                       corrupt_shard: dict | None = None,
                       corrupt_every_shard=None) -> None:
    artifact_paths = []
    for index in range(shard_count):
        rel = f"dump-20260520/train-{index:05d}-of-{shard_count:05d}.parquet"
        path = root / "original/parquet" / rel
        rows = [_row(index * rows_per_shard + j) for j in range(rows_per_shard)]
        if corrupt_shard and corrupt_shard["shard"] == index:
            rows = corrupt_shard["rows"](rows)
        if corrupt_every_shard is not None:
            rows = corrupt_every_shard(rows)
        _write_shard(path, rows)
        artifact_paths.append(path)
    logical = hashlib.sha256(b"fixture").hexdigest()
    stage = SS.seal(artifact_root=root, artifact_paths=artifact_paths,
                    receipt_path=root / "stages/original/receipt.json",
                    records=shard_count, fmt="test-fixture",
                    logical_sha256=logical, producer_id="test", producer_version="v1")
    run = {"schema": "aw.openlegaldata-hf-dump-acquisition-run.v1",
          "version": "v1", "status": "complete", "config": ACQ.CONFIG,
          "stages": {"original": stage}}
    (root / "RUN.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")


def test_check_sample_row_flags_empty_content_and_bad_date():
    good = subject._check_sample_row(_row(0))
    assert good["content_nonempty"] and good["court_name_nonempty"] and good["date_sane"]
    bad = subject._check_sample_row({**_row(1), "content": "  ", "date": "not-a-date"})
    assert not bad["content_nonempty"] and not bad["date_sane"]


def test_verify_end_to_end_matches_exact_row_count(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ACQ, "SHARD_COUNT", 3)
    monkeypatch.setattr(ACQ, "EXPECTED_ROWS_HINT", 15)
    _build_fixture_run(tmp_path, rows_per_shard=5, shard_count=3)
    result = subject.verify(root=tmp_path, out_path=tmp_path / "VERIFY.json")
    assert result["counts"]["measured_total_rows"] == 15
    assert result["counts"]["row_count_drift"] == 0
    assert result["verdict"].startswith("PASS:")
    assert result["sample_verification"]["sample_size"] == 6  # 2/shard x 3 shards
    assert (tmp_path / "VERIFY.json").exists()
    # Re-running is idempotent against the immutable output.
    again = subject.verify(root=tmp_path, out_path=tmp_path / "VERIFY.json")
    assert again == result


def test_verify_reports_drift_without_failing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ACQ, "SHARD_COUNT", 2)
    monkeypatch.setattr(ACQ, "EXPECTED_ROWS_HINT", 999)
    _build_fixture_run(tmp_path, rows_per_shard=4, shard_count=2)
    result = subject.verify(root=tmp_path, out_path=tmp_path / "VERIFY.json")
    assert result["counts"]["measured_total_rows"] == 8
    assert result["counts"]["row_count_drift"] == 8 - 999
    assert result["verdict"].startswith("PASS_WITH_DRIFT:")


def test_verify_raises_when_content_fill_rate_is_below_floor(tmp_path: Path, monkeypatch):
    # 10 shards x 2 sampled rows = 20-row sample; empty out every shard's
    # first row (10/20 = 50% empty), well under the 90% floor.
    monkeypatch.setattr(ACQ, "SHARD_COUNT", 10)
    monkeypatch.setattr(ACQ, "EXPECTED_ROWS_HINT", 40)
    _build_fixture_run(
        tmp_path, rows_per_shard=4, shard_count=10,
        corrupt_every_shard=lambda rows: [{**rows[0], "content": ""}] + rows[1:])
    with pytest.raises(ValueError, match="decision-text fill rate"):
        subject.verify(root=tmp_path, out_path=tmp_path / "VERIFY.json")


def test_verify_tolerates_a_small_empty_content_tail(tmp_path: Path, monkeypatch):
    """Real source data (task 489, live 2026-08-31): a small fraction of
    genuine OpenLegalData decisions carry empty `content` alongside populated
    neighbours from the same court/shard. One empty row in a 100-row sample
    must PASS, not abort an otherwise-successful acquisition."""
    monkeypatch.setattr(ACQ, "SHARD_COUNT", 50)
    monkeypatch.setattr(ACQ, "EXPECTED_ROWS_HINT", 100)
    _build_fixture_run(
        tmp_path, rows_per_shard=2, shard_count=50,
        corrupt_shard={"shard": 0, "rows": lambda rows: (
            [{**rows[0], "content": ""}] + rows[1:])})
    result = subject.verify(root=tmp_path, out_path=tmp_path / "VERIFY.json")
    assert result["sample_verification"]["content_nonempty_count"] == 99
    assert result["verdict"].startswith("PASS")


def test_verify_tolerates_null_jurisdiction_for_federal_courts(tmp_path: Path, monkeypatch):
    """`jurisdiction` is legitimately null for federal courts (Bundesfinanzhof
    etc., observed live task 489) -- must never gate the run."""
    monkeypatch.setattr(ACQ, "SHARD_COUNT", 1)
    monkeypatch.setattr(ACQ, "EXPECTED_ROWS_HINT", 2)
    _build_fixture_run(
        tmp_path, rows_per_shard=2, shard_count=1,
        corrupt_shard={"shard": 0, "rows": lambda rows: [
            {**row, "court": {**row["court"], "jurisdiction": None}} for row in rows]})
    result = subject.verify(root=tmp_path, out_path=tmp_path / "VERIFY.json")
    assert result["sample_verification"]["jurisdiction_nonempty_rate"] == 0.0
    assert result["verdict"].startswith("PASS")


def test_verify_still_raises_on_missing_court_name(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ACQ, "SHARD_COUNT", 2)
    monkeypatch.setattr(ACQ, "EXPECTED_ROWS_HINT", 8)
    _build_fixture_run(
        tmp_path, rows_per_shard=4, shard_count=2,
        corrupt_shard={"shard": 0, "rows": lambda rows: (
            [{**rows[0], "court": {**rows[0]["court"], "name": ""}}] + rows[1:])})
    with pytest.raises(ValueError, match="no court name"):
        subject.verify(root=tmp_path, out_path=tmp_path / "VERIFY.json")


def test_verify_raises_on_schema_drift(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ACQ, "SHARD_COUNT", 1)
    monkeypatch.setattr(ACQ, "EXPECTED_ROWS_HINT", 4)
    rel = "dump-20260520/train-00000-of-00001.parquet"
    path = tmp_path / "original/parquet" / rel
    table = pa.table({"id": [1, 2], "content": ["a", "b"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    logical = hashlib.sha256(b"x").hexdigest()
    stage = SS.seal(artifact_root=tmp_path, artifact_paths=[path],
                    receipt_path=tmp_path / "stages/original/receipt.json",
                    records=1, fmt="test", logical_sha256=logical,
                    producer_id="test", producer_version="v1")
    run = {"schema": "aw.openlegaldata-hf-dump-acquisition-run.v1",
          "version": "v1", "status": "complete", "config": ACQ.CONFIG,
          "stages": {"original": stage}}
    (tmp_path / "RUN.json").write_text(json.dumps(run) + "\n")
    with pytest.raises(ValueError, match="schema drift"):
        subject.verify(root=tmp_path, out_path=tmp_path / "VERIFY.json")
