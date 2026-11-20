#!/usr/bin/env python3
"""Materialize terminal official ERIC PDFs to full Markdown and outlinks.

task 656: extraction is delegated to `cytir.extract.pdf.extract_pdf` -- the
routed liteparse/Docling extractor from task 655's bake-off. liteparse is
the default engine for every document; Docling is used only for the narrow
short+complex/table-dense class (see `cytir/extract/pdf.py` for the routing
rule and its evidence). This module keeps its own receipt/sealing/
resumability and the bounded-child-process + heartbeat wrapper for
observability during a tranche run; the wrapped child now runs the router,
not Docling directly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit, urlunsplit

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire.deterministic_gzip import deterministic_gzip_text
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as SS

import cytir.extract.pdf as EXTRACT  # noqa: E402


VERSION = "eric-pdf-routed-markdown.v4-liteparse-default-docling-routed"
PRODUCER = "pipeline/materialize_eric_pdf_markdown.py"
URL_RE = re.compile(r"https?://[^\s<>{}\[\]`\"']+")
PREFLIGHT_SCHEMA = "aw.eric-pdf-partition-preflight.v1"


def _line(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode()


def _artifact(path: Path, artifact_root: Path) -> dict:
    return {"uri": "artifact://" + path.resolve().relative_to(
        artifact_root.resolve()).as_posix(), "bytes": path.stat().st_size,
        "sha256": sha256_file(path)}


def _publish(path: Path, body: bytes) -> None:
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError(f"immutable ERIC materialization collision: {path}")
        return
    atomic_publish(path, body)


def _canonical_link(raw: str) -> str | None:
    raw = raw.rstrip(".,;:!?)\"")
    try:
        split = urlsplit(raw)
    except ValueError:
        return None
    if split.scheme not in {"http", "https"} or not split.netloc:
        return None
    return urlunsplit((split.scheme.lower(), split.netloc.lower(),
                       split.path or "/", split.query, split.fragment))


def _links(markdown: str) -> list[dict]:
    values = sorted({link for raw in URL_RE.findall(markdown)
                     if (link := _canonical_link(raw))})
    return [{"target_url": value, "relation": "pdf-uri-annotation",
             "context": "URI extracted from full Docling Markdown"}
            for value in values]


def _diagnostics_from_result(result: "EXTRACT.ExtractResult") -> dict:
    """Shared shape between the in-process and bounded-child extraction
    paths -- what `_materialize_one` folds into the per-document diagnostics
    record, task 656's provenance contract (extractor/version/route_reason,
    plus fallback/OCR-retry when they happened)."""
    diagnostics = {"page_count": result.page_count,
                    "extracted_characters": len(result.markdown)}
    diagnostics.update(result.provenance())
    if result.extractor == "docling":
        diagnostics["docling_version"] = result.extractor_version
        diagnostics["do_table_structure"] = True
    return diagnostics


def _extract_inprocess(extract_fn, pdf: Path) -> tuple[str, dict]:
    """`extract_fn` defaults to `cytir.extract.pdf.extract_pdf`; tests inject a
    fake with the same `(pdf_path) -> ExtractResult` signature in place of
    the old `converter.convert()` duck-type."""
    result = extract_fn(pdf)
    return result.markdown, _diagnostics_from_result(result)


class DocumentExtractionTimeout(TimeoutError):
    """One source PDF exceeded its declared full-document wall-time."""

    def __init__(self, message: str, heartbeat_path: Path):
        super().__init__(message)
        self.heartbeat_path = heartbeat_path


def _child_env() -> dict:
    """The child runs `python -m cytir.sources.eric.materialize`; this checkout's root leads
    PYTHONPATH so a foreign editable `aw` install can never shadow it (task 646)."""
    root = str(Path(__file__).resolve().parents[3])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (root, env.get("PYTHONPATH", "")) if p)
    return env


def _run_bounded_process(*, argv: list[str], timeout_seconds: int,
                         heartbeat_seconds: int, heartbeat_root: Path,
                         identity: dict) -> tuple[int, float, Path]:
    if timeout_seconds <= 0 or heartbeat_seconds <= 0:
        raise ValueError("ERIC timeout and heartbeat intervals must be positive")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    process = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, env=_child_env())
    next_heartbeat = started
    sequence = 0

    def publish(status: str, elapsed: float, *, terminal: bool) -> Path:
        nonlocal sequence
        path = heartbeat_root / f"{sequence:06d}-{status}.json"
        value = {
            "schema": "aw.eric-document-extraction-heartbeat.v1",
            "status": status, "started_at": started_at,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 6),
            "timeout_seconds": timeout_seconds, "child_pid": process.pid,
            **identity,
        }
        if terminal:
            value["finished_at"] = value["observed_at"]
        _publish(path, _line(value))
        sequence += 1
        return path

    while True:
        returncode = process.poll()
        now = time.monotonic()
        elapsed = now - started
        if returncode is not None:
            terminal_path = publish(
                "complete" if returncode == 0 else "failed", elapsed,
                terminal=True)
            return returncode, elapsed, terminal_path
        if elapsed >= timeout_seconds:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            terminal_path = publish("timeout", elapsed, terminal=True)
            raise DocumentExtractionTimeout(
                f"full-document extraction exceeded {timeout_seconds}s",
                terminal_path)
        if now >= next_heartbeat:
            publish("running", elapsed, terminal=False)
            next_heartbeat = now + heartbeat_seconds
        time.sleep(min(0.25, heartbeat_seconds))


def _extract_bounded(*, pdf: Path, work_root: Path, heartbeat_root: Path,
                     eric_id: str, timeout_seconds: int,
                     heartbeat_seconds: int) -> tuple[str, dict, Path]:
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{eric_id}-", dir=work_root) as directory:
        result_path = Path(directory) / "result.json"
        argv = [sys.executable, "-m", "cytir.sources.eric.materialize", "--extract-one",
                str(pdf), "--extract-result", str(result_path)]
        returncode, elapsed, terminal_heartbeat = _run_bounded_process(
            argv=argv, timeout_seconds=timeout_seconds,
            heartbeat_seconds=heartbeat_seconds, heartbeat_root=heartbeat_root,
            identity={"eric_id": eric_id, "source_pdf_bytes": pdf.stat().st_size})
        if returncode != 0 or not result_path.is_file():
            raise RuntimeError(f"bounded extraction child failed rc={returncode}")
        value = json.loads(result_path.read_text())
        if value.get("status") != "complete":
            raise RuntimeError(str(value.get("error") or "bounded extraction child failed"))
        diagnostics = value["diagnostics"]
        diagnostics["wall_time_seconds"] = round(elapsed, 6)
        diagnostics["wall_time_limit_seconds"] = timeout_seconds
        return value["markdown"], diagnostics, terminal_heartbeat


def _extract_one_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--extract-one", type=Path, required=True)
    parser.add_argument("--extract-result", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = EXTRACT.extract_pdf(args.extract_one.resolve())
        value = {"status": "complete", "markdown": result.markdown,
                 "diagnostics": _diagnostics_from_result(result)}
    except Exception as exc:
        value = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    atomic_publish(args.extract_result, _line(value))
    return 0 if value["status"] == "complete" else 1


def _validate_parent_receipt(receipt: dict, artifact_root: Path) -> Path:
    if (receipt.get("schema") != "aw.eric-official-pdf-receipt.v1"
            or receipt.get("status") != "complete"):
        raise ValueError("ERIC PDF parent receipt is not complete")
    body = receipt.get("artifacts", {}).get("response_body") or {}
    uri = str(body.get("uri") or "")
    if not uri.startswith("artifact://"):
        raise ValueError("ERIC PDF parent body is not globally portable")
    path = artifact_root / uri.removeprefix("artifact://")
    if (not path.is_file() or path.stat().st_size != body.get("bytes")
            or sha256_file(path) != body.get("sha256")):
        raise ValueError("ERIC PDF parent body artifact mismatch")
    return path


def _validate_terminal(receipt: dict, *, source_receipt: dict,
                       artifact_root: Path) -> None:
    if (receipt.get("schema") != "aw.eric-pdf-materialization-receipt.v1"
            or receipt.get("status") not in {"complete", "quarantine"}
            or receipt.get("eric_id") != source_receipt.get("eric_id")):
        raise ValueError("existing ERIC Markdown receipt identity drift")
    for item in (receipt.get("artifacts") or {}).values():
        path = artifact_root / item["uri"].removeprefix("artifact://")
        if (not path.is_file() or path.stat().st_size != item["bytes"]
                or sha256_file(path) != item["sha256"]):
            raise ValueError("existing ERIC Markdown artifact mismatch")


def _selection_inventory(source_receipts: list[Path]) -> list[dict]:
    return [{"eric_id": json.loads(path.read_text())["eric_id"],
             "receipt_bytes": path.stat().st_size,
             "receipt_sha256": sha256_file(path)}
            for path in source_receipts]


def _selection_logical_sha256(inventory: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in inventory:
        digest.update((canonical_json(item) + "\n").encode())
    return digest.hexdigest()


def _validate_shared_preflight(*, path: Path, acquisition_run_path: Path,
                               acquisition_run: dict,
                               source_receipts: list[Path],
                               artifact_root: Path) -> dict:
    """Validate the one deep L0 witness shared by disjoint workers."""
    path = path.resolve()
    try:
        path.relative_to(artifact_root.resolve())
    except ValueError as exc:
        raise ValueError("ERIC shared preflight is outside artifact root") from exc
    value = json.loads(path.read_text())
    expected_stage_sha = hashlib.sha256(canonical_json(
        acquisition_run["stages"]["original"]).encode()).hexdigest()
    if (value.get("schema") != PREFLIGHT_SCHEMA
            or value.get("status") != "PASS"
            or value.get("deep_verification", {}).get(
                "selected_response_bodies_rehashed") != len(source_receipts)
            or value.get("acquisition_run_sha256") != sha256_file(acquisition_run_path)
            or value.get("original_stage_metadata_sha256") != expected_stage_sha
            or value.get("selected_complete_pdfs") != len(source_receipts)
            or value.get("selection_logical_sha256")
            != _selection_logical_sha256(_selection_inventory(source_receipts))):
        raise ValueError("ERIC shared partition preflight does not bind acquisition")
    return _artifact(path, artifact_root)


def build_record(source_receipt: dict, markdown: str) -> tuple[dict, dict]:
    """The terminal parent record and the parent-bound outlinks of one extracted ERIC PDF.

    The ONE place the record shape lives: the batch materializer and the `eric_official_pdf`
    adapter's `extract` both call it, so the two paths cannot drift."""
    eric_id = source_receipt["eric_id"]
    parent_body = source_receipt["artifacts"]["response_body"]
    links = _links(markdown)
    document_id = hashlib.sha256(f"eric\0{eric_id}".encode()).hexdigest()
    record = {
        "document_id": document_id, "source_record_id": eric_id,
        "url": source_receipt["source_url"], "title": f"ERIC {eric_id}",
        "markdown": markdown, "markdown_sha256": hashlib.sha256(
            markdown.encode()).hexdigest(), "lang": None,
        "parent_identity": {"eric_id": eric_id,
                            "pdf_sha256": parent_body["sha256"]},
        "parent_response": source_receipt["artifacts"],
        "source_version": "official ERIC full-text PDF response",
        "source_links": links,
    }
    outlinks = {"parent_url": source_receipt["source_url"],
                "document_id": document_id, "links": links,
                "explicit_empty": not bool(links)}
    return record, outlinks


def _materialize_one(*, source_path: Path, root: Path, artifact_root: Path,
                     extract_fn,
                     per_document_timeout_seconds: int | None = None,
                     heartbeat_seconds: int = 30) -> dict:
    source_receipt = json.loads(source_path.read_text())
    eric_id = source_receipt["eric_id"]
    terminal = root / "receipts" / f"{eric_id}.json"
    if terminal.exists():
        prior = json.loads(terminal.read_text())
        _validate_terminal(prior, source_receipt=source_receipt,
                           artifact_root=artifact_root)
        return prior
    pdf = _validate_parent_receipt(source_receipt, artifact_root)
    parent_body = source_receipt["artifacts"]["response_body"]
    heartbeat_root = root / "heartbeats" / eric_id[:4] / eric_id
    heartbeat_path = None
    try:
        if per_document_timeout_seconds is None:
            markdown, diagnostics = _extract_inprocess(extract_fn, pdf)
        else:
            markdown, diagnostics, heartbeat_path = _extract_bounded(
                pdf=pdf, work_root=root / "work", heartbeat_root=heartbeat_root,
                eric_id=eric_id, timeout_seconds=per_document_timeout_seconds,
                heartbeat_seconds=heartbeat_seconds)
        if not markdown.strip():
            raise ValueError("extraction emitted empty Markdown; OCR-required PDF excluded")
        record, outlinks = build_record(source_receipt, markdown)
        record["source_links"]  # precondition: build_record emits source_links (KeyError otherwise)
        diagnostics.update({
            "eric_id": eric_id, "pdf_sha256": parent_body["sha256"],
            "do_ocr": False,  # the primary pass; see ocr_retried for the rare recovery path
            "empty_markdown": False,
        })
        markdown_path = root / "parents" / eric_id[:4] / f"{eric_id}.json"
        outlinks_path = root / "outlinks" / eric_id[:4] / f"{eric_id}.json"
        diagnostics_path = root / "diagnostics" / eric_id[:4] / f"{eric_id}.json"
        _publish(markdown_path, _line(record)); _publish(outlinks_path, _line(outlinks))
        _publish(diagnostics_path, _line(diagnostics))
        receipt = {"schema": "aw.eric-pdf-materialization-receipt.v1",
                   "version": VERSION, "status": "complete", "eric_id": eric_id,
                   "source_url": source_receipt["source_url"],
                   "source_pdf": parent_body,
                   "source_receipt": _artifact(source_path, artifact_root),
                   "artifacts": {"markdown_parent": _artifact(markdown_path, artifact_root),
                                 "parent_bound_outlinks": _artifact(outlinks_path, artifact_root),
                                 "page_diagnostics": _artifact(diagnostics_path, artifact_root)}}
        if heartbeat_path is not None:
            receipt["artifacts"]["extraction_heartbeat"] = _artifact(
                heartbeat_path, artifact_root)
    except Exception as exc:
        if isinstance(exc, DocumentExtractionTimeout):
            heartbeat_path = exc.heartbeat_path
        elif per_document_timeout_seconds is not None:
            heartbeat_paths = sorted(heartbeat_root.glob("*.json"))
            heartbeat_path = heartbeat_paths[-1] if heartbeat_paths else None
        failure = root / "quarantine" / eric_id[:4] / f"{eric_id}.json"
        diagnostics_path = root / "diagnostics" / eric_id[:4] / f"{eric_id}.json"
        timeout = isinstance(exc, DocumentExtractionTimeout)
        diagnostics = {"eric_id": eric_id, "pdf_sha256": parent_body["sha256"],
            "pdf_bytes": parent_body["bytes"], "page_count": None,
            "status": "wall_time_timeout" if timeout else "extraction_failure",
            "timeout_seconds": per_document_timeout_seconds,
            "do_ocr": False, "do_table_structure": True}
        _publish(diagnostics_path, _line(diagnostics))
        _publish(failure, _line({"eric_id": eric_id,
            "source_url": source_receipt["source_url"],
            "source_pdf_sha256": parent_body["sha256"],
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}"}))
        receipt = {"schema": "aw.eric-pdf-materialization-receipt.v1",
                   "version": VERSION, "status": "quarantine", "eric_id": eric_id,
                   "source_url": source_receipt["source_url"],
                   "source_pdf": parent_body,
                   "source_receipt": _artifact(source_path, artifact_root),
                   "reason": ("extraction_wall_time_exceeded" if timeout
                              else "extraction_failure_or_empty"),
                   "artifacts": {"failure": _artifact(failure, artifact_root),
                                 "page_diagnostics": _artifact(diagnostics_path, artifact_root)}}
        if heartbeat_path is not None and heartbeat_path.is_file():
            receipt["artifacts"]["extraction_heartbeat"] = _artifact(
                heartbeat_path, artifact_root)
    _publish(terminal, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    return receipt


def materialize(*, acquisition_root: Path, root: Path, artifact_root: Path,
                limit: int | None = None, extract_fn=None,
                partition_index: int | None = None,
                partition_count: int | None = None,
                shared_preflight: Path | None = None,
                per_document_timeout_seconds: int | None = None,
                heartbeat_seconds: int = 30) -> dict:
    acquisition_run_path = acquisition_root / "RUN.json"
    acquisition_run = json.loads(acquisition_run_path.read_text())
    if (acquisition_run.get("schema") != "aw.eric-official-pdf-acquisition-run.v1"
            or acquisition_run.get("status") != "complete"):
        raise ValueError("terminal ERIC PDF acquisition RUN required")
    if (partition_index is None) != (partition_count is None):
        raise ValueError("partition index and count must be declared together")
    if partition_count is not None:
        if partition_count <= 1 or partition_index < 0 or partition_index >= partition_count:
            raise ValueError("invalid ERIC materialization partition")
        if shared_preflight is None:
            raise ValueError("partitioned ERIC worker requires shared deep preflight")
    else:
        if shared_preflight is not None:
            raise ValueError("shared ERIC preflight is only valid for partitioned workers")
        _validate_stage(acquisition_run["stages"]["original"], "eric.pdf-original",
                        artifact_root=artifact_root, deep=True)
        preflight_artifact = None
    source_receipts = []
    for path in sorted((acquisition_root / "receipts").glob("*.json")):
        value = json.loads(path.read_text())
        if value.get("status") == "complete":
            source_receipts.append(path)
    if len(source_receipts) != acquisition_run["counts"]["pdf_complete"]:
        raise ValueError("complete ERIC PDF receipt conservation failed")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        source_receipts = source_receipts[:limit]
    full_selection_count = len(source_receipts)
    if partition_count is not None:
        preflight_artifact = _validate_shared_preflight(
            path=shared_preflight, acquisition_run_path=acquisition_run_path,
            acquisition_run=acquisition_run, source_receipts=source_receipts,
            artifact_root=artifact_root)
    if partition_count is not None:
        source_receipts = [path for ordinal, path in enumerate(source_receipts)
                           if ordinal % partition_count == partition_index]
    identity = {"schema": "aw.eric-pdf-materialization-identity.v1",
                "version": VERSION,
                "acquisition_run_sha256": sha256_file(acquisition_run_path),
                "selected_complete_pdfs": len(source_receipts),
                "full_selection_complete_pdfs": full_selection_count,
                "limit": limit,
                "partition_index": partition_index,
                "partition_count": partition_count,
                "shared_deep_preflight": preflight_artifact}
    if per_document_timeout_seconds is not None:
        identity["per_document_timeout_seconds"] = per_document_timeout_seconds
        identity["heartbeat_seconds"] = heartbeat_seconds
    root.mkdir(parents=True, exist_ok=True)
    _publish(root / "IDENTITY.json", _line(identity))
    terminal_run = root / "RUN.json"
    if terminal_run.exists():
        prior = json.loads(terminal_run.read_text())
        if (prior.get("schema") != "aw.eric-pdf-materialization-run.v1"
                or prior.get("parent", {}).get("run_sha256")
                != identity["acquisition_run_sha256"]
                or prior.get("selection", {}).get("selected_complete_pdfs")
                != len(source_receipts)):
            raise ValueError("existing ERIC materialization RUN identity drift")
        for name in ("original", "markdown", "outlinks", "diagnostics", "quarantine"):
            _validate_stage(prior["stages"][name], f"eric.{name}",
                            artifact_root=artifact_root, deep=True)
        return prior
    if extract_fn is None and per_document_timeout_seconds is None:
        extract_fn = EXTRACT.extract_pdf
    receipts = []
    for index, source_path in enumerate(source_receipts, 1):
        receipts.append(_materialize_one(
            source_path=source_path, root=root, artifact_root=artifact_root,
            extract_fn=extract_fn,
            per_document_timeout_seconds=per_document_timeout_seconds,
            heartbeat_seconds=heartbeat_seconds))
        if index % 25 == 0:
            print(f"PROGRESS terminal={index}/{len(source_receipts)}", flush=True)
    complete = [row for row in receipts if row["status"] == "complete"]
    quarantined = [row for row in receipts if row["status"] == "quarantine"]
    markdown_path = root / "markdown/documents.jsonl.gz"
    outlinks_path = root / "outlinks/parent-bound.jsonl.gz"
    diagnostics_path = root / "diagnostics/pages.jsonl.gz"
    quarantine_path = root / "quarantine/records.jsonl.gz"
    for path in (markdown_path, outlinks_path, diagnostics_path, quarantine_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with deterministic_gzip_text(markdown_path) as stream:
        for row in complete:
            path = artifact_root / row["artifacts"]["markdown_parent"]["uri"].removeprefix("artifact://")
            stream.write(path.read_text())
    with deterministic_gzip_text(outlinks_path) as stream:
        for row in complete:
            path = artifact_root / row["artifacts"]["parent_bound_outlinks"]["uri"].removeprefix("artifact://")
            stream.write(path.read_text())
    with deterministic_gzip_text(diagnostics_path) as stream:
        for row in receipts:
            path = artifact_root / row["artifacts"]["page_diagnostics"]["uri"].removeprefix("artifact://")
            stream.write(path.read_text())
    with deterministic_gzip_text(quarantine_path) as stream:
        for row in quarantined:
            stream.write(canonical_json(row) + "\n")
    receipt_paths = sorted((root / "receipts").glob("*.json"))
    common = [root / "IDENTITY.json", *receipt_paths]
    markdown_stage = SS.seal(artifact_root=artifact_root,
        artifact_paths=[markdown_path, *common],
        receipt_path=root / "stages/markdown/receipt.json", records=len(complete),
        fmt="canonical-jsonl+deterministic-gzip+docling-full-markdown",
        logical_sha256=sha256_file(markdown_path), producer_id=PRODUCER,
        producer_version=VERSION)
    outlinks_stage = SS.seal(artifact_root=artifact_root,
        artifact_paths=[outlinks_path],
        receipt_path=root / "stages/outlinks/receipt.json", records=len(complete),
        fmt="canonical-jsonl+deterministic-gzip+parent-bound-links-including-empty",
        logical_sha256=sha256_file(outlinks_path), producer_id=PRODUCER,
        producer_version=VERSION)
    diagnostics_stage = SS.seal(artifact_root=artifact_root,
        artifact_paths=[diagnostics_path],
        receipt_path=root / "stages/diagnostics/receipt.json", records=len(receipts),
        fmt="canonical-jsonl+deterministic-gzip+page-level-docling-diagnostics",
        logical_sha256=sha256_file(diagnostics_path), producer_id=PRODUCER,
        producer_version=VERSION)
    quarantine_stage = SS.seal(artifact_root=artifact_root,
        artifact_paths=[quarantine_path],
        receipt_path=root / "stages/quarantine/receipt.json", records=len(quarantined),
        fmt="canonical-jsonl+deterministic-gzip",
        logical_sha256=sha256_file(quarantine_path), producer_id=PRODUCER,
        producer_version=VERSION)
    # task 656: the run-level block declares the ROUTER's fixed policy (every
    # worker in a partitioned run must declare the identical policy -- see
    # finalize_eric_pdf_materialization_workers.py's cross-worker check).
    # Which engine any one document actually used lives per-document in its
    # own diagnostics row (`extractor`/`extractor_version`/`route_reason`),
    # not here -- a routed run is not homogeneous by design.
    extraction = {"engine": "aw.extract.pdf", "router_version": EXTRACT.VERSION,
                  "docling_do_ocr": False, "docling_do_table_structure": True,
                  "content_cap": None}
    if per_document_timeout_seconds is not None:
        extraction["per_document_timeout_seconds"] = per_document_timeout_seconds
        extraction["heartbeat_seconds"] = heartbeat_seconds
    run = {"schema": "aw.eric-pdf-materialization-run.v1", "version": VERSION,
           "status": "ready_for_web_pdf_profile_fidelity_and_current_chunking",
           "source_profile": "web_pdf",
           "parent": {"run_sha256": sha256_file(acquisition_run_path),
                      "pdf_complete": acquisition_run["counts"]["pdf_complete"]},
           "selection": {"limit": limit,
                         "full_selection_complete_pdfs": full_selection_count,
                         "selected_complete_pdfs": len(source_receipts),
                         "partition_index": partition_index,
                         "partition_count": partition_count},
           "counts": {"markdown_documents": len(complete),
                      "docling_quarantine": len(quarantined),
                      "outlink_parents_including_empty": len(complete)},
           "extraction": extraction,
           "stages": {"original": acquisition_run["stages"]["original"],
                      "markdown": markdown_stage, "outlinks": outlinks_stage,
                      "diagnostics": diagnostics_stage,
                      "quarantine": quarantine_stage},
           "next": "web_pdf structure/fidelity audit, measured current-policy chunks, UUID/token gates; OCR-required quarantines remain excluded"}
    _publish(root / "RUN.json",
             (json.dumps(run, indent=2, sort_keys=True) + "\n").encode())
    for name in ("original", "markdown", "outlinks", "diagnostics", "quarantine"):
        _validate_stage(run["stages"][name], f"eric.{name}",
                        artifact_root=artifact_root, deep=True)
    return run


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--extract-one":
        raise SystemExit(_extract_one_main(sys.argv[1:]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--partition-index", type=int)
    parser.add_argument("--partition-count", type=int)
    parser.add_argument("--shared-preflight", type=Path)
    parser.add_argument("--per-document-timeout-seconds", type=int, default=7200)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(materialize(
        acquisition_root=args.acquisition_root.resolve(), root=args.root.resolve(),
        artifact_root=args.artifact_root.resolve(), limit=args.limit,
        partition_index=args.partition_index, partition_count=args.partition_count,
        shared_preflight=(args.shared_preflight.resolve()
                          if args.shared_preflight else None),
        per_document_timeout_seconds=args.per_document_timeout_seconds,
        heartbeat_seconds=args.heartbeat_seconds),
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
