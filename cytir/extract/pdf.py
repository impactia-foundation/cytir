#!/usr/bin/env python3
"""cytir.extract.pdf — the ONE PDF-to-markdown extraction entry point (task 656).

Router born from task 655's liteparse-vs-Docling bake-off (301 real PDFs,
CPU-only, prod Docling config vs liteparse; task 655
+ task 655's artifacts). Headline findings that set this module's defaults:

  - liteparse completed 301/301 (100%). Docling (prod config: `do_ocr=False`,
    `do_table_structure=True`) completed 176/301 (58.5%) and timed out on
    125/301 (41.5%) at a 120s per-document budget -- 53.5% of ERIC reports.
  - liteparse uses ~735-805x less CPU per page than Docling (two independent
    derivations: mean-of-ratios and ratio-of-totals).
  - Docling keeps a real structural edge on the documents it *does* finish:
    ~5x more headings, ~2x more markdown tables on ERIC reports. Word-level
    retention is ~96% symmetric either way -- this is a structure story, not
    a "liteparse drops text" story.
  - Timeouts track page count, not a flat per-document cost (bake-off §3):
    Docling-complete docs averaged 16.2 pages (median 13); Docling-timeout
    docs averaged 113.8 pages (median 79). Per-document completion rate
    (task 655's `per_document.csv`, recomputed task 656): docs
    <=40 pages, 166/193 (86.0%) Docling-complete; docs >40 pages, 10/108
    (9.3%) -- the cliff task 655 §8 names as "~40-80 pages", lower bound.
  - liteparse's own `is_complex` flag alone is a weak router (bake-off §6:
    precision 0.37, recall 0.60 against a strict "Docling wins" label) --
    it must be combined with the page-count budget, never used alone.

Routing rule (task 655 §8, "ROUTE, don't switch outright and don't stay"):
  liteparse by default, for every document, always. Docling ONLY when the
  document is (a) short enough to plausibly finish (<= DOCLING_MAX_PAGES)
  AND (b) liteparse's own complexity flag fired, OR the document is
  table-dense by a measurable proxy (liteparse's own per-page ruled/text
  table-run counts, summed over the document). Long documents always go to
  liteparse -- that is exactly the class where Docling's completion rate
  collapses. If Docling fails or times out on a routed document, fall back
  to liteparse's already-computed result and record the fallback in
  provenance; the document is never lost.

"Worker-pool mode": production callers (the ERIC materializer's partitioned
workers, and future onboarded PDF families per task 646 S7) fan a tranche
out across N worker *processes*, each handling its share of documents
serially. Within a worker, each engine call for a single document is itself
a bounded, one-shot child process (`_run_bounded_extract`) with a hard wall
-time cap -- this is what lets a hung native call (liteparse's PDFium/
Tesseract, Docling's layout/table models) never stall the worker past its
declared timeout, regardless of what's happening inside the C/Rust
extension.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

VERSION = "aw.extract.pdf.v1"

# task 655 §3 / §8, recomputed on task 655's `per_document.csv` (task
# 656): the page count below which Docling can plausibly finish. 40 is the
# conservative (lower) bound of §8's "~40-80 pages" cliff -- <=40 pages:
# 86.0% Docling-complete; >40 pages: 9.3%.
DEFAULT_DOCLING_MAX_PAGES = 40

# liteparse's own per-page `ruled_table_count` + `text_table_run_count`,
# summed over the document, is the "measurable proxy" for table density --
# it is already computed by the same liteparse call that produces the
# routing signal, so no second pass over the document is needed. >=1 means
# "found at least one table-shaped structure".
DEFAULT_TABLE_DENSE_MIN_RUNS = 1

# task 655 ran its full 301-document roster (100% liteparse completion,
# 0 timeouts) inside a 120s hard cap per document -- that is the proven
# ceiling for the worker-pool's hard per-document timeout on the fast leg.
DEFAULT_LITEPARSE_TIMEOUT_SECONDS = 120

# Unchanged from the existing ERIC production ceiling
# (`cytir.sources.eric.materialize --per-document-timeout-seconds`,
# default 7200) -- a document only reaches Docling at all when it is
# already <= DOCLING_MAX_PAGES, so in practice this ceiling is a backstop,
# not the operating point (bake-off: Docling-complete docs averaged 48.9s).
DEFAULT_DOCLING_TIMEOUT_SECONDS = 7200

# task 655 §5: liteparse's OCR retry is a real recovery path for the rare
# scanned/image-only stratum (1/581 audited ERIC receipts) that Docling's
# production config cannot recover at all. Auto-retry only when the first
# pass both flags `needs_ocr` and returns near-empty text -- never spend the
# ~1,200x-slower OCR path on a normal born-digital document.
DEFAULT_OCR_RETRY_MIN_CHARS = 200

_THIS_FILE = Path(__file__).resolve()


@dataclass(frozen=True)
class ExtractConfig:
    docling_max_pages: int = DEFAULT_DOCLING_MAX_PAGES
    table_dense_min_runs: int = DEFAULT_TABLE_DENSE_MIN_RUNS
    liteparse_timeout_seconds: int = DEFAULT_LITEPARSE_TIMEOUT_SECONDS
    docling_timeout_seconds: int = DEFAULT_DOCLING_TIMEOUT_SECONDS
    ocr_retry_min_chars: int = DEFAULT_OCR_RETRY_MIN_CHARS
    ocr_retry_enabled: bool = True


@dataclass
class BackendResult:
    status: str  # "complete" | "timeout" | "failed"
    markdown: str = ""
    page_count: int | None = None
    is_complex: bool | None = None
    table_dense_runs: int | None = None
    needs_ocr: bool | None = None
    version: str | None = None
    error: str | None = None
    wall_time_seconds: float = 0.0


@dataclass
class ExtractResult:
    markdown: str
    extractor: str  # "liteparse" | "docling"
    extractor_version: str
    route_reason: str
    page_count: int | None
    is_complex: bool | None
    table_dense_runs: int | None
    fallback_from: str | None = None
    fallback_reason: str | None = None
    ocr_retried: bool = False
    wall_time_seconds: float = 0.0

    def provenance(self) -> dict:
        """The record fields this module promises every caller: which engine
        actually produced the markdown, its version, and why it was chosen."""
        value = {
            "pdf_router_version": VERSION,
            "extractor": self.extractor,
            "extractor_version": self.extractor_version,
            "route_reason": self.route_reason,
            "page_count": self.page_count,
        }
        if self.fallback_from:
            value["fallback_from"] = self.fallback_from
            value["fallback_reason"] = self.fallback_reason
        if self.ocr_retried:
            value["ocr_retried"] = True
        return value


class ExtractionFailed(RuntimeError):
    """The routed engine failed, and no usable fallback markdown exists."""


def decide_route(*, page_count: int | None, is_complex: bool | None,
                  table_dense_runs: int | None,
                  config: ExtractConfig) -> tuple[str, str]:
    """Pure routing decision -- no I/O, exhaustively unit-testable. See the
    module docstring for the evidence behind each branch."""
    if page_count is None:
        return "liteparse", "page count unknown; default to liteparse"
    if page_count > config.docling_max_pages:
        return "liteparse", (
            f"{page_count} pages > docling_max_pages={config.docling_max_pages} "
            "(task 655 SS3/SS8: Docling completion collapses past this page "
            "count -- long PDFs always go to liteparse)")
    if is_complex:
        return "docling", (
            f"<= {config.docling_max_pages} pages and liteparse flagged the "
            "document complex")
    runs = table_dense_runs or 0
    if runs >= config.table_dense_min_runs:
        return "docling", (
            f"<= {config.docling_max_pages} pages and table-dense "
            f"({runs} ruled/text table runs >= {config.table_dense_min_runs})")
    return "liteparse", (
        f"<= {config.docling_max_pages} pages but neither flagged complex "
        "nor table-dense")


# --------------------------------------------------------------------------
# Bounded one-shot child processes -- the hard-timeout mechanism shared by
# both engines.
# --------------------------------------------------------------------------

def _run_bounded_extract(argv: list[str], out_path: Path, *,
                          timeout_seconds: int) -> tuple[str, float, dict | None]:
    started = time.monotonic()
    try:
        completed = subprocess.run(argv, timeout=timeout_seconds,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        elapsed = time.monotonic() - started
    except subprocess.TimeoutExpired:
        return "timeout", float(timeout_seconds), None
    value = None
    if out_path.is_file():
        try:
            value = json.loads(out_path.read_text())
        except (OSError, ValueError):
            value = None
    if completed.returncode == 0 and value and value.get("status") == "complete":
        return "complete", elapsed, value
    return "failed", elapsed, value


def _liteparse_worker_main(pdf_path: Path, out_path: Path, *,
                            ocr_enabled: bool) -> None:
    import importlib.metadata
    from liteparse import LiteParse

    lp = LiteParse(output_format="markdown", ocr_enabled=ocr_enabled,
                    include_complexity=True, num_workers=1, quiet=True)
    try:
        result = lp.parse(pdf_path.read_bytes())
    finally:
        lp.close()
    pages = []
    for page in result.pages:
        complexity = page.complexity
        pages.append({
            "markdown": page.markdown or "",
            "is_complex": bool(complexity.layout.is_complex) if complexity else False,
            "needs_ocr": bool(complexity.needs_ocr) if complexity else False,
            "ruled_table_count": (
                (complexity.layout.ruled_table_count or 0) if complexity else 0),
            "text_table_run_count": (
                (complexity.layout.text_table_run_count or 0) if complexity else 0),
        })
    markdown = "\n\n".join(p["markdown"] for p in pages)
    value = {
        "status": "complete",
        "markdown": markdown,
        "page_count": result.total_pages,
        "is_complex": any(p["is_complex"] for p in pages),
        "needs_ocr": any(p["needs_ocr"] for p in pages),
        "table_dense_runs": sum(
            p["ruled_table_count"] + p["text_table_run_count"] for p in pages),
        "version": importlib.metadata.version("liteparse"),
    }
    out_path.write_text(json.dumps(value))


def _docling_worker_main(pdf_path: Path, out_path: Path) -> None:
    # task 655's resource note: Docling's table/layout models spawn BLAS/
    # ONNXRuntime thread pools that ignore AcceleratorOptions.num_threads
    # unless these are capped explicitly -- left uncapped they starve every
    # other lane sharing the box.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "ORT_NUM_THREADS"):
        os.environ.setdefault(var, "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only, ever
    import importlib.metadata
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    with tempfile.TemporaryDirectory(prefix="aw-extract-docling-") as directory:
        source = Path(directory) / "source.pdf"
        os.symlink(pdf_path.resolve(), source)
        result = converter.convert(source)
        markdown = result.document.export_to_markdown()
        page_count = len(getattr(result.document, "pages", {}))
    value = {
        "status": "complete", "markdown": markdown, "page_count": page_count,
        "version": importlib.metadata.version("docling"),
    }
    out_path.write_text(json.dumps(value))


def _worker_one_main(argv: list[str], *, mode: str) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("out", type=Path)
    if mode == "liteparse":
        parser.add_argument("--ocr", action="store_true")
    args = parser.parse_args(argv)
    try:
        if mode == "liteparse":
            _liteparse_worker_main(args.pdf.resolve(), args.out,
                                    ocr_enabled=args.ocr)
        else:
            _docling_worker_main(args.pdf.resolve(), args.out)
        return 0
    except Exception as exc:  # noqa: BLE001 -- reported to the parent, not raised
        args.out.write_text(json.dumps(
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        return 1


# --------------------------------------------------------------------------
# The two engine runners -- each a bounded child process, invokable directly
# or overridable for tests.
# --------------------------------------------------------------------------

def _run_liteparse(pdf_path: Path, config: ExtractConfig, *,
                    ocr: bool = False) -> BackendResult:
    with tempfile.TemporaryDirectory(prefix="aw-extract-lp-") as directory:
        out_path = Path(directory) / "result.json"
        argv = [sys.executable, str(_THIS_FILE), "--liteparse-one",
                str(pdf_path.resolve()), str(out_path)]
        if ocr:
            argv.append("--ocr")
        status, elapsed, value = _run_bounded_extract(
            argv, out_path, timeout_seconds=config.liteparse_timeout_seconds)
        if status == "complete":
            return BackendResult(
                status="complete", markdown=value["markdown"],
                page_count=value["page_count"], is_complex=value["is_complex"],
                table_dense_runs=value["table_dense_runs"],
                needs_ocr=value.get("needs_ocr"), version=value.get("version"),
                wall_time_seconds=elapsed)
        return BackendResult(
            status=status, wall_time_seconds=elapsed,
            error=(value or {}).get("error") or f"liteparse {status}")


def _run_docling(pdf_path: Path, config: ExtractConfig) -> BackendResult:
    with tempfile.TemporaryDirectory(prefix="aw-extract-dl-") as directory:
        out_path = Path(directory) / "result.json"
        argv = [sys.executable, str(_THIS_FILE), "--docling-one",
                str(pdf_path.resolve()), str(out_path)]
        status, elapsed, value = _run_bounded_extract(
            argv, out_path, timeout_seconds=config.docling_timeout_seconds)
        if status == "complete" and (value or {}).get("markdown", "").strip():
            return BackendResult(
                status="complete", markdown=value["markdown"],
                page_count=value.get("page_count"), version=value.get("version"),
                wall_time_seconds=elapsed)
        if status == "complete":
            return BackendResult(status="failed", wall_time_seconds=elapsed,
                                  error="docling emitted empty markdown")
        return BackendResult(
            status=status, wall_time_seconds=elapsed,
            error=(value or {}).get("error") or f"docling {status}")


# --------------------------------------------------------------------------
# The one entry point.
# --------------------------------------------------------------------------

def extract_pdf(pdf_path: Path, *, config: ExtractConfig | None = None,
                 liteparse_runner=_run_liteparse,
                 docling_runner=_run_docling) -> ExtractResult:
    """Extract `pdf_path` to markdown, routing between liteparse (default)
    and Docling per `decide_route`. Raises `ExtractionFailed` only when
    liteparse itself -- the always-available default -- fails; a Docling
    failure on a routed document is recovered via fallback to liteparse's
    already-computed result, never raised.
    """
    config = config or ExtractConfig()
    pdf_path = Path(pdf_path)
    lp = liteparse_runner(pdf_path, config)
    ocr_retried = False
    if (lp.status == "complete" and config.ocr_retry_enabled and lp.needs_ocr
            and len((lp.markdown or "").strip()) < config.ocr_retry_min_chars):
        # task 655 §5: a near-empty first pass that liteparse itself flags as
        # OCR-needed is exactly the rare scanned/image-only stratum where the
        # OCR path recovers real text Docling's production config cannot get
        # at all. Retry once; keep the first-pass result if the retry fails.
        ocr_lp = liteparse_runner(pdf_path, config, ocr=True)
        if ocr_lp.status == "complete" and len(
                (ocr_lp.markdown or "").strip()) > len((lp.markdown or "").strip()):
            lp = ocr_lp
            ocr_retried = True
    if lp.status != "complete":
        raise ExtractionFailed(
            f"liteparse failed on {pdf_path}: status={lp.status} error={lp.error}")

    route, reason = decide_route(page_count=lp.page_count, is_complex=lp.is_complex,
                                  table_dense_runs=lp.table_dense_runs, config=config)
    if route == "liteparse":
        return ExtractResult(
            markdown=lp.markdown, extractor="liteparse",
            extractor_version=lp.version or "unknown", route_reason=reason,
            page_count=lp.page_count, is_complex=lp.is_complex,
            table_dense_runs=lp.table_dense_runs, ocr_retried=ocr_retried,
            wall_time_seconds=lp.wall_time_seconds)

    docling = docling_runner(pdf_path, config)
    if docling.status == "complete":
        return ExtractResult(
            markdown=docling.markdown, extractor="docling",
            extractor_version=docling.version or "unknown", route_reason=reason,
            page_count=lp.page_count, is_complex=lp.is_complex,
            table_dense_runs=lp.table_dense_runs, ocr_retried=ocr_retried,
            wall_time_seconds=lp.wall_time_seconds + docling.wall_time_seconds)

    fallback_reason = f"docling status={docling.status}"
    if docling.error:
        fallback_reason += f" error={docling.error}"
    return ExtractResult(
        markdown=lp.markdown, extractor="liteparse",
        extractor_version=lp.version or "unknown", route_reason=reason,
        page_count=lp.page_count, is_complex=lp.is_complex,
        table_dense_runs=lp.table_dense_runs, fallback_from="docling",
        fallback_reason=fallback_reason, ocr_retried=ocr_retried,
        wall_time_seconds=lp.wall_time_seconds + docling.wall_time_seconds)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--liteparse-one":
        raise SystemExit(_worker_one_main(sys.argv[2:], mode="liteparse"))
    if len(sys.argv) > 1 and sys.argv[1] == "--docling-one":
        raise SystemExit(_worker_one_main(sys.argv[2:], mode="docling"))
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--docling-max-pages", type=int,
                         default=DEFAULT_DOCLING_MAX_PAGES)
    parser.add_argument("--table-dense-min-runs", type=int,
                         default=DEFAULT_TABLE_DENSE_MIN_RUNS)
    parser.add_argument("--liteparse-timeout-seconds", type=int,
                         default=DEFAULT_LITEPARSE_TIMEOUT_SECONDS)
    parser.add_argument("--docling-timeout-seconds", type=int,
                         default=DEFAULT_DOCLING_TIMEOUT_SECONDS)
    args = parser.parse_args()
    config = ExtractConfig(
        docling_max_pages=args.docling_max_pages,
        table_dense_min_runs=args.table_dense_min_runs,
        liteparse_timeout_seconds=args.liteparse_timeout_seconds,
        docling_timeout_seconds=args.docling_timeout_seconds)
    result = extract_pdf(args.pdf, config=config)
    print(json.dumps({"markdown": result.markdown, **result.provenance()},
                      indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
