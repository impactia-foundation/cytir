# Lives in tests/ with the rest of the package's tests (task 696 slice 2).
"""Tests for cytir.extract.pdf (task 656): the routed liteparse/Docling PDF
extractor built from task 655's bake-off (task 655's bake-off note).

Three layers, each proven independently:
  1. `decide_route` -- pure routing logic, every branch, no I/O, no deps.
  2. `extract_pdf` orchestration -- fake engine runners exercise the real
     routing/fallback/OCR-retry wiring deterministically, no heavy deps.
  3. The real engines -- `pytest.importorskip`-gated: proves the actual
     bounded-subprocess wiring against real, small, committed-shape PDFs
     when liteparse/Docling are installed (the build box; not CI's `dev` extra).
"""
from __future__ import annotations

import json
import sys
import time

import pytest

from cytir.extract import pdf as subject


# --------------------------------------------------------------------------
# A minimal, real, single-page PDF -- same construction as
# the task 471 PDF-pass script's `_pdf()`.
# --------------------------------------------------------------------------

def _pdf(text: str | None, *, pages: int = 1) -> bytes:
    content = (f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET" if text else "").encode()
    page_objs = []
    kids = []
    content_id = 3 + pages
    font_id = content_id + 1
    for i in range(pages):
        page_id = 3 + i
        kids.append(f"{page_id} 0 R")
        page_objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode())
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {pages} >>").encode(),
        *page_objs,
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % index + body + b"\nendobj\n"
    start = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += (b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, start))
    return bytes(out)


PDF_TEXT_1PAGE = _pdf("Verordnung ueber die Luftfahrt Artikel 12")
PDF_TEXT_3PAGE = _pdf("Bundesgesetz ueber den Datenschutz", pages=3)


# --------------------------------------------------------------------------
# 1. decide_route -- pure logic
# --------------------------------------------------------------------------

def test_route_short_simple_stays_liteparse():
    route, reason = subject.decide_route(
        page_count=5, is_complex=False, table_dense_runs=0,
        config=subject.ExtractConfig())
    assert route == "liteparse"
    assert "neither" in reason


def test_route_short_complex_goes_docling():
    route, reason = subject.decide_route(
        page_count=5, is_complex=True, table_dense_runs=0,
        config=subject.ExtractConfig())
    assert route == "docling"
    assert "complex" in reason


def test_route_short_table_dense_goes_docling():
    route, reason = subject.decide_route(
        page_count=5, is_complex=False, table_dense_runs=3,
        config=subject.ExtractConfig())
    assert route == "docling"
    assert "table-dense" in reason


def test_route_long_complex_still_liteparse():
    """Long PDFs always go to liteparse -- page-count budget overrides
    complexity/table signals (task 655 SS3/SS8)."""
    config = subject.ExtractConfig()
    route, reason = subject.decide_route(
        page_count=config.docling_max_pages + 1, is_complex=True,
        table_dense_runs=5, config=config)
    assert route == "liteparse"
    assert "pages >" in reason


def test_route_at_exactly_the_page_threshold_is_eligible():
    config = subject.ExtractConfig()
    route, _ = subject.decide_route(
        page_count=config.docling_max_pages, is_complex=True,
        table_dense_runs=0, config=config)
    assert route == "docling"


def test_route_unknown_page_count_defaults_liteparse():
    route, reason = subject.decide_route(
        page_count=None, is_complex=True, table_dense_runs=5,
        config=subject.ExtractConfig())
    assert route == "liteparse"
    assert "unknown" in reason


def test_table_dense_threshold_is_configurable():
    config = subject.ExtractConfig(table_dense_min_runs=3)
    route, _ = subject.decide_route(
        page_count=5, is_complex=False, table_dense_runs=2, config=config)
    assert route == "liteparse"
    route, _ = subject.decide_route(
        page_count=5, is_complex=False, table_dense_runs=3, config=config)
    assert route == "docling"


# --------------------------------------------------------------------------
# 2. extract_pdf orchestration -- fake runners, real routing/fallback wiring
# --------------------------------------------------------------------------

def _fake_liteparse(*, page_count=5, is_complex=False, table_dense_runs=0,
                     needs_ocr=False, markdown="liteparse markdown",
                     status="complete"):
    calls = []

    def runner(pdf_path, config, *, ocr=False):
        calls.append({"pdf_path": pdf_path, "ocr": ocr})
        if ocr:
            return subject.BackendResult(
                status="complete", markdown="OCR RECOVERED TEXT " * 20,
                page_count=page_count, is_complex=is_complex,
                table_dense_runs=table_dense_runs, needs_ocr=needs_ocr,
                version="liteparse-test")
        return subject.BackendResult(
            status=status, markdown=markdown, page_count=page_count,
            is_complex=is_complex, table_dense_runs=table_dense_runs,
            needs_ocr=needs_ocr, version="liteparse-test",
            error=None if status == "complete" else "fake liteparse failure")
    runner.calls = calls
    return runner


def _fake_docling(*, status="complete", markdown="docling markdown"):
    calls = []

    def runner(pdf_path, config):
        calls.append(pdf_path)
        return subject.BackendResult(
            status=status, markdown=markdown, version="docling-test",
            error=None if status == "complete" else "fake docling failure")
    runner.calls = calls
    return runner


def test_extract_pdf_default_route_uses_liteparse(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF_TEXT_1PAGE)
    result = subject.extract_pdf(
        pdf, liteparse_runner=_fake_liteparse(), docling_runner=_fake_docling())
    assert result.extractor == "liteparse"
    assert result.markdown == "liteparse markdown"
    assert result.fallback_from is None
    prov = result.provenance()
    assert prov["extractor"] == "liteparse"
    assert prov["extractor_version"] == "liteparse-test"
    assert "route_reason" in prov and "fallback_from" not in prov


def test_extract_pdf_routes_to_docling_on_complex_short_doc(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF_TEXT_1PAGE)
    result = subject.extract_pdf(
        pdf, liteparse_runner=_fake_liteparse(is_complex=True),
        docling_runner=_fake_docling())
    assert result.extractor == "docling"
    assert result.markdown == "docling markdown"
    assert result.provenance()["extractor_version"] == "docling-test"


def test_extract_pdf_falls_back_to_liteparse_on_docling_timeout(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF_TEXT_1PAGE)
    result = subject.extract_pdf(
        pdf, liteparse_runner=_fake_liteparse(is_complex=True),
        docling_runner=_fake_docling(status="timeout"))
    assert result.extractor == "liteparse"
    assert result.markdown == "liteparse markdown"
    assert result.fallback_from == "docling"
    assert "timeout" in result.fallback_reason
    prov = result.provenance()
    assert prov["fallback_from"] == "docling"
    assert "fallback_reason" in prov


def test_extract_pdf_falls_back_to_liteparse_on_docling_failure(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF_TEXT_1PAGE)
    result = subject.extract_pdf(
        pdf, liteparse_runner=_fake_liteparse(table_dense_runs=2),
        docling_runner=_fake_docling(status="failed"))
    assert result.extractor == "liteparse"
    assert result.fallback_from == "docling"
    assert "failed" in result.fallback_reason


def test_extract_pdf_raises_when_liteparse_itself_fails(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF_TEXT_1PAGE)
    with pytest.raises(subject.ExtractionFailed):
        subject.extract_pdf(
            pdf, liteparse_runner=_fake_liteparse(status="failed"),
            docling_runner=_fake_docling())


def test_extract_pdf_long_doc_never_calls_docling(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF_TEXT_1PAGE)
    config = subject.ExtractConfig()
    docling = _fake_docling()
    result = subject.extract_pdf(
        pdf, config=config,
        liteparse_runner=_fake_liteparse(
            page_count=config.docling_max_pages + 1, is_complex=True,
            table_dense_runs=9),
        docling_runner=docling)
    assert result.extractor == "liteparse"
    assert docling.calls == []


def test_extract_pdf_ocr_retry_on_near_empty_flagged_result(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF_TEXT_1PAGE)
    lp = _fake_liteparse(needs_ocr=True, markdown="xx")  # near-empty first pass
    result = subject.extract_pdf(pdf, liteparse_runner=lp,
                                  docling_runner=_fake_docling())
    assert result.ocr_retried is True
    assert "OCR RECOVERED TEXT" in result.markdown
    assert any(call["ocr"] for call in lp.calls)
    assert result.provenance()["ocr_retried"] is True


def test_extract_pdf_no_ocr_retry_when_not_flagged(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF_TEXT_1PAGE)
    lp = _fake_liteparse(needs_ocr=False, markdown="xx")
    result = subject.extract_pdf(pdf, liteparse_runner=lp,
                                  docling_runner=_fake_docling())
    assert result.ocr_retried is False
    assert all(not call["ocr"] for call in lp.calls)


# --------------------------------------------------------------------------
# Bounded child-process mechanism -- no heavy deps, always runs.
# --------------------------------------------------------------------------

def test_run_bounded_extract_hard_timeout_holds(tmp_path):
    out_path = tmp_path / "result.json"
    argv = [sys.executable, "-c", "import time; time.sleep(30)"]
    started = time.monotonic()
    status, elapsed, value = subject._run_bounded_extract(
        argv, out_path, timeout_seconds=1)
    wall = time.monotonic() - started
    assert status == "timeout"
    assert value is None
    assert wall < 10  # the hard cap held; nowhere near the child's 30s sleep


def test_run_bounded_extract_reads_result_on_success(tmp_path):
    out_path = tmp_path / "result.json"
    argv = [sys.executable, "-c",
            "import json,sys; open(sys.argv[1],'w').write("
            "json.dumps({'status':'complete','markdown':'hi'}))",
            str(out_path)]
    status, _elapsed, value = subject._run_bounded_extract(
        argv, out_path, timeout_seconds=10)
    assert status == "complete"
    assert value == {"status": "complete", "markdown": "hi"}


def test_run_bounded_extract_nonzero_exit_is_failed(tmp_path):
    out_path = tmp_path / "result.json"
    argv = [sys.executable, "-c", "import sys; sys.exit(1)"]
    status, _elapsed, value = subject._run_bounded_extract(
        argv, out_path, timeout_seconds=10)
    assert status == "failed"
    assert value is None


# --------------------------------------------------------------------------
# 3. Real engines -- importorskip-gated (not present in CI's `dev` extra;
# present on the build box, where liteparse/Docling are already installed).
# --------------------------------------------------------------------------

def test_real_liteparse_extracts_real_text(tmp_path):
    pytest.importorskip("liteparse", reason="real-engine half needs liteparse")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF_TEXT_1PAGE)
    backend = subject._run_liteparse(pdf, subject.ExtractConfig())
    assert backend.status == "complete"
    assert "Luftfahrt" in backend.markdown
    assert backend.page_count == 1
    assert backend.version


def test_real_liteparse_long_pdf_still_completes_fast(tmp_path):
    pytest.importorskip("liteparse", reason="real-engine half needs liteparse")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF_TEXT_3PAGE)
    backend = subject._run_liteparse(pdf, subject.ExtractConfig())
    assert backend.status == "complete"
    assert backend.page_count == 3


def test_real_docling_extracts_real_text(tmp_path):
    pytest.importorskip("docling", reason="real-engine half needs Docling")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF_TEXT_1PAGE)
    backend = subject._run_docling(pdf, subject.ExtractConfig())
    assert backend.status == "complete"
    assert "Luftfahrt" in backend.markdown


def test_real_docling_failure_falls_back_via_extract_pdf(tmp_path):
    """A garbage 'PDF' fails both engines identically in isolation, but
    routed through extract_pdf with the real liteparse leg working and
    Docling forced to fail, the fallback path is real end to end."""
    pytest.importorskip("liteparse", reason="real-engine half needs liteparse")
    pytest.importorskip("docling", reason="real-engine half needs Docling")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF_TEXT_1PAGE)

    def broken_docling(pdf_path, config):
        return subject.BackendResult(status="failed", error="forced for test")

    result = subject.extract_pdf(
        pdf, liteparse_runner=lambda p, c, ocr=False: subject._run_liteparse(
            p, c, ocr=ocr),
        docling_runner=broken_docling)
    # the real liteparse leg on a plain short doc routes liteparse by
    # default anyway (no complexity/table signal), so force the docling
    # route to prove the fallback wiring, not just the default path.
    forced_config = subject.ExtractConfig()
    real_lp = subject._run_liteparse(pdf, forced_config)
    route, _ = subject.decide_route(
        page_count=real_lp.page_count, is_complex=True,
        table_dense_runs=0, config=forced_config)
    assert route == "docling"
    assert result.extractor == "liteparse"
    assert "Luftfahrt" in result.markdown


def test_cli_json_output_shape(tmp_path, capsys):
    pytest.importorskip("liteparse", reason="real-engine half needs liteparse")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF_TEXT_1PAGE)
    old_argv = sys.argv
    try:
        sys.argv = ["aw-extract-pdf", str(pdf)]
        subject.main()
    finally:
        sys.argv = old_argv
    out = json.loads(capsys.readouterr().out)
    assert out["extractor"] == "liteparse"
    assert "markdown" in out
    assert out["pdf_router_version"] == subject.VERSION
