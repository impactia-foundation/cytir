import json
from pathlib import Path
import sys

from cytir.sources.eric import materialize as subject
import cytir.extract.pdf as EXTRACT


class FakeExtractor:
    """Stands in for `cytir.extract.pdf.extract_pdf` (task 656): same
    `(pdf_path) -> ExtractResult` signature the old `converter.convert()`
    duck-type used to provide."""

    def __init__(self, markdown: str, *, page_count: int = 2):
        self.markdown = markdown
        self.page_count = page_count
        self.calls = 0

    def __call__(self, _pdf_path):
        self.calls += 1
        return EXTRACT.ExtractResult(
            markdown=self.markdown, extractor="docling",
            extractor_version="test-docling", route_reason="test fixture",
            page_count=self.page_count, is_complex=False, table_dense_runs=0)


def test_full_pdf_markdown_outlinks_and_resume_without_truncation(tmp_path: Path,
                                                                  monkeypatch):
    acquisition = tmp_path / "acquisition"
    acquisition.mkdir()
    pdf = acquisition / "original/pdf/ED12/ED123456.bin"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.7\nbody")
    headers = acquisition / "transport/ED12/ED123456.headers"
    headers.parent.mkdir(parents=True)
    headers.write_text("HTTP/1.1 200\r\nContent-Type: application/pdf\r\n")
    transport = acquisition / "transport/ED12/ED123456.json"
    transport.write_text("{}\n")
    receipt = acquisition / "receipts/ED123456.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({
        "schema": "aw.eric-official-pdf-receipt.v1", "status": "complete",
        "eric_id": "ED123456",
        "source_url": "https://files.eric.ed.gov/fulltext/ED123456.pdf",
        "artifacts": {
            "response_body": subject._artifact(pdf, tmp_path),
            "response_headers": subject._artifact(headers, tmp_path),
            "transport": subject._artifact(transport, tmp_path),
        },
    }))
    (acquisition / "RUN.json").write_text(json.dumps({
        "schema": "aw.eric-official-pdf-acquisition-run.v1",
        "status": "complete", "counts": {"pdf_complete": 1},
        "stages": {"original": {}},
    }))
    monkeypatch.setattr(subject, "_validate_stage", lambda *args, **kwargs: {})
    long_tail = "z" * 20000
    extract_fn = FakeExtractor(
        "# Full report\n\nSee https://example.edu/manual?q=1#page.\n\n" + long_tail)
    root = tmp_path / "markdown"
    run = subject.materialize(
        acquisition_root=acquisition, root=root, artifact_root=tmp_path,
        extract_fn=extract_fn)
    assert run["counts"] == {"markdown_documents": 1, "docling_quarantine": 0,
                              "outlink_parents_including_empty": 1}
    parent = json.loads((root / "parents/ED12/ED123456.json").read_text())
    assert parent["markdown"].endswith(long_tail)
    assert parent["source_links"] == [{
        "target_url": "https://example.edu/manual?q=1#page",
        "relation": "pdf-uri-annotation",
        "context": "URI extracted from full Docling Markdown",
    }]
    diagnostics = json.loads(
        (root / "diagnostics/ED12/ED123456.json").read_text())
    assert diagnostics["page_count"] == 2
    assert diagnostics["do_ocr"] is False
    assert diagnostics["extractor"] == "docling"
    assert diagnostics["extractor_version"] == "test-docling"
    assert diagnostics["docling_version"] == "test-docling"
    assert run["extraction"]["content_cap"] is None

    subject.materialize(
        acquisition_root=acquisition, root=root, artifact_root=tmp_path,
        extract_fn=extract_fn)
    assert extract_fn.calls == 1


def test_empty_docling_result_is_explicit_quarantine_with_empty_stages(
        tmp_path: Path, monkeypatch):
    acquisition = tmp_path / "acquisition"; acquisition.mkdir()
    pdf = acquisition / "original/pdf/TD63/TD638294.bin"
    pdf.parent.mkdir(parents=True); pdf.write_bytes(b"%PDF-1.4\nscan")
    receipt = acquisition / "receipts/TD638294.json"; receipt.parent.mkdir()
    receipt.write_text(json.dumps({
        "schema": "aw.eric-official-pdf-receipt.v1", "status": "complete",
        "eric_id": "TD638294",
        "source_url": "https://files.eric.ed.gov/fulltext/TD638294.pdf",
        "artifacts": {"response_body": subject._artifact(pdf, tmp_path)},
    }))
    (acquisition / "RUN.json").write_text(json.dumps({
        "schema": "aw.eric-official-pdf-acquisition-run.v1", "status": "complete",
        "counts": {"pdf_complete": 1}, "stages": {"original": {}},
    }))
    monkeypatch.setattr(subject, "_validate_stage", lambda *args, **kwargs: {})
    run = subject.materialize(
        acquisition_root=acquisition, root=tmp_path / "markdown",
        artifact_root=tmp_path, extract_fn=FakeExtractor(""))
    assert run["counts"]["markdown_documents"] == 0
    assert run["counts"]["docling_quarantine"] == 1
    assert run["stages"]["markdown"]["records"] == 0
    assert run["stages"]["outlinks"]["records"] == 0


def test_bounded_process_emits_append_only_heartbeats_and_times_out(tmp_path: Path):
    heartbeat_root = tmp_path / "heartbeats/ED00/ED000001"
    try:
        subject._run_bounded_process(
            argv=[sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds=1, heartbeat_seconds=1,
            heartbeat_root=heartbeat_root,
            identity={"eric_id": "ED000001", "source_pdf_bytes": 123})
    except subject.DocumentExtractionTimeout as exc:
        assert exc.heartbeat_path.is_file()
    else:
        raise AssertionError("bounded process did not time out")
    values = [json.loads(path.read_text())
              for path in sorted(heartbeat_root.glob("*.json"))]
    assert values[0]["status"] == "running"
    assert values[-1]["status"] == "timeout"
    assert values[-1]["timeout_seconds"] == 1
    assert values[-1]["source_pdf_bytes"] == 123
    assert len({path.name for path in heartbeat_root.glob("*.json")}) == len(values)


def test_timeout_quarantine_preserves_source_and_page_diagnostics(
        tmp_path: Path, monkeypatch):
    acquisition = tmp_path / "acquisition"; acquisition.mkdir()
    pdf = acquisition / "original/pdf/ED12/ED123457.bin"
    pdf.parent.mkdir(parents=True); pdf.write_bytes(b"%PDF-1.4\nlarge")
    receipt = acquisition / "receipts/ED123457.json"; receipt.parent.mkdir()
    receipt.write_text(json.dumps({
        "schema": "aw.eric-official-pdf-receipt.v1", "status": "complete",
        "eric_id": "ED123457",
        "source_url": "https://files.eric.ed.gov/fulltext/ED123457.pdf",
        "artifacts": {"response_body": subject._artifact(pdf, tmp_path)},
    }))
    (acquisition / "RUN.json").write_text(json.dumps({
        "schema": "aw.eric-official-pdf-acquisition-run.v1", "status": "complete",
        "counts": {"pdf_complete": 1}, "stages": {"original": {}},
    }))
    monkeypatch.setattr(subject, "_validate_stage", lambda *args, **kwargs: {})

    def timeout(**kwargs):
        heartbeat = kwargs["heartbeat_root"] / "000000-timeout.json"
        subject._publish(heartbeat, subject._line({"status": "timeout"}))
        raise subject.DocumentExtractionTimeout("test timeout", heartbeat)

    monkeypatch.setattr(subject, "_extract_bounded", timeout)
    root = tmp_path / "markdown"
    run = subject.materialize(
        acquisition_root=acquisition, root=root, artifact_root=tmp_path,
        per_document_timeout_seconds=1, heartbeat_seconds=1)
    terminal = json.loads((root / "receipts/ED123457.json").read_text())
    assert terminal["reason"] == "extraction_wall_time_exceeded"
    assert set(terminal["artifacts"]) == {
        "failure", "page_diagnostics", "extraction_heartbeat"}
    diagnostics = json.loads(
        (root / "diagnostics/ED12/ED123457.json").read_text())
    assert diagnostics["status"] == "wall_time_timeout"
    assert diagnostics["pdf_bytes"] == len(b"%PDF-1.4\nlarge")
    assert diagnostics["page_count"] is None
    assert run["stages"]["diagnostics"]["records"] == 1
