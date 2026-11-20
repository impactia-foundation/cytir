import gzip
import json
from pathlib import Path

from cytir.acquire.deterministic_gzip import DeterministicGzipWriter
import cytir.sources.jorf.filter_full_parents as subject
from cytir.acquire import stage_receipt as SS


def _write(path: Path, rows: list[dict]) -> None:
    writer = DeterministicGzipWriter(path)
    for row in rows:
        writer.write(row)
    writer.close()


def _read(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def test_filter_conserves_full_article_parents_and_explicit_residuals(
        tmp_path: Path, monkeypatch):
    root = tmp_path / "root"; source = root / "source"; source.mkdir(parents=True)
    markdown = source / "markdown/parents.jsonl.gz"
    outlinks = source / "outlinks/parents.jsonl.gz"
    full = {"source_record_id": "JORFTEXT1", "article_bodies": 2,
            "empty_article_records": 1, "baseline_production_point_ids": ["p1"],
            "markdown_sha256": "a" * 64, "parent_identity": {"source_id": "JORFTEXT1"}}
    residual = {"source_record_id": "JORFTEXT2", "article_bodies": 0,
                "body_sections": 1, "source_record_count": 2,
                "baseline_production_point_ids": ["p2"], "markdown_sha256": "b" * 64,
                "parent_identity": {"source_id": "JORFTEXT2"}}
    _write(markdown, [full, residual])
    _write(outlinks, [
        {"source_record_id": "JORFTEXT1", "links": [{"target": "u"}],
         "unresolved": [{"reason": "x"}], "explicit_empty": False},
        {"source_record_id": "JORFTEXT2", "links": [], "unresolved": [],
         "explicit_empty": True},
    ])
    stage = SS.seal(
        artifact_root=root, artifact_paths=[markdown, outlinks],
        receipt_path=source / "stages/materialization/receipt.json", records=2,
        fmt="test", logical_sha256="c" * 64, producer_id="test", producer_version="1")
    run_path = source / "RUN.json"
    run_path.write_text(json.dumps({
        "schema": "aw.jorf-exact-full-markdown-run.v1",
        "version": "jorf-exact-full-markdown.v2",
        "status": "materialization-complete-profile-fidelity-chunking-pending",
        "counts": {"documents": 2}, "stages": {"materialization": stage}}))
    monkeypatch.setattr(subject, "EXPECTED_SOURCE", 2)
    monkeypatch.setattr(subject, "EXPECTED_FULL", 1)
    monkeypatch.setattr(subject, "EXPECTED_RESIDUAL", 1)

    output = root / "filtered"
    result = subject.filter_full(
        artifact_root=root, parent_run=run_path, output=output)

    assert result["counts"] == {
        "source_documents": 2, "full_article_documents": 1,
        "source_content_residuals": 1, "full_article_bodies": 2,
        "empty_article_records_retained": 1, "links": 1,
        "unresolved_links": 1, "explicit_empty_outlink_parents": 0}
    assert [row["source_record_id"] for row in
            _read(output / "markdown/parents.jsonl.gz")] == ["JORFTEXT1"]
    excluded = _read(output / "residual/source-content.jsonl.gz")
    assert excluded[0]["source_record_id"] == "JORFTEXT2"
    assert excluded[0]["baseline_production_point_ids"] == ["p2"]
    assert excluded[0]["reason"] == (
        "no_full_article_body_in_terminal_official_linked_record_state")

