#!/usr/bin/env python3
"""Materialize a bounded, preservation-first FINMA PDF Markdown pilot.

This is deliberately a pilot, not a production admission.  It decodes exact
PDF bodies from the sealed pack-fetch L0, verifies their declared digests,
converts the complete documents with pinned Docling settings, records per-page
geometry and conversion errors, and extracts explicit PDF/Markdown URI links.
No metadata-only fallback and no content or chunk cap is permitted.
"""
from __future__ import annotations

import argparse
import base64
import gzip
from cytir.acquire.gzip_pinned import deterministic_gzip_open
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import urllib.parse

from cytir.acquire.artifact_io import canonical_json, sha256_file
from cytir.acquire import crawl as pack_fetch
from cytir.acquire import stage_receipt as SS


VERSION = "finma-pdf-markdown-pilot.v6-contextual-outlinks"
URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"']+")
BASE = "https://www.finma.ch"


def _iter_l0(receipt: Path, artifact_root: Path):
    value = json.loads(receipt.read_text(encoding="utf-8"))
    for artifact in value.get("artifacts") or []:
        uri = artifact.get("uri", "")
        if not uri.endswith(".jsonl.gz"):
            continue
        path = artifact_root / uri.removeprefix("artifact://")
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"L0 shard binding mismatch: {uri}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                yield uri, line_number, json.loads(line)


def _decode(row: dict, location: str) -> bytes:
    body = base64.b64decode(row["body_b64"], validate=True)
    if len(body) != row.get("bytes") or hashlib.sha256(body).hexdigest() != row.get("sha256"):
        raise ValueError(f"body binding mismatch: {location}")
    return body


def _source_record_id(row: dict) -> str:
    """Use the source-owned URL identity, never a deduplicating body digest.

    FINMA legitimately serves the same PDF bytes at distinct language URLs.
    Those are separate source records even though preservation stores their
    shared body only once.  Conflating the URL identity with the content hash
    made stage inventories contain duplicate artifact URIs and silently made
    two official records share one parent ID.
    """
    value = pack_fetch.make_canon()(str(row.get("url") or ""), keep_query=True)
    if not value:
        raise ValueError("FINMA PDF row lacks a canonical source URL")
    return value


def _metadata(api_rows: list[tuple[str, int, dict]]) -> dict[str, dict]:
    canon = pack_fetch.make_canon(); result: dict[str, dict] = {}
    for uri, line_number, row in api_rows:
        document = json.loads(_decode(row, f"{uri}:{line_number}"))
        for item in document.get("Items") or []:
            if str(item.get("Extension") or "").casefold() != "pdf":
                continue
            variants = item.get("OtherLanguageLinks") or []
            if not variants and item.get("Link"):
                variants = [{"Url": item["Link"], "Name": ""}]
            for variant in variants:
                raw = urllib.parse.urljoin(BASE, variant.get("Url") or "")
                url = canon(raw, keep_query=True)
                if not url:
                    continue
                lang = str(variant.get("Name") or "").casefold()[:2]
                if not lang:
                    match = re.search(r"[?&]sc_lang=(de|fr|it|en)\b", raw)
                    lang = match.group(1) if match else "und"
                value = {
                    "title": str(item.get("Title") or "").strip(),
                    "description": str(item.get("Description") or "").strip(),
                    "release_date": str(item.get("Date") or "").strip(),
                    "language": lang,
                    "api_parent": {"uri": uri, "line": line_number,
                                   "sha256": row["sha256"]},
                }
                if url in result and result[url] != value:
                    raise ValueError(f"conflicting API metadata: {url}")
                result[url] = value
    return result


def _pdf_annotation_links(body: bytes) -> list[dict]:
    from pypdf import PdfReader
    links = []
    reader = PdfReader(io.BytesIO(body), strict=False)
    for page_number, page in enumerate(reader.pages, 1):
        for reference in page.get("/Annots") or []:
            annotation = reference.get_object()
            action = annotation.get("/A") or {}
            target = action.get("/URI")
            if target:
                links.append({
                    "target_url": str(target).strip(),
                    "page": page_number,
                    "method": "pdf-annotation-uri",
                    "relation": "citation-or-external-uri",
                    "context": f"PDF URI annotation on page {page_number}",
                })
    return links


def _markdown_links(markdown: str) -> list[dict]:
    """Retain each target with deterministic source-local citation context."""
    links = []
    for line_number, line in enumerate(markdown.splitlines(), 1):
        context = " ".join(line.split())
        for value in URL_RE.findall(line):
            links.append({
                "target_url": value.rstrip(".,;:"),
                "page": None,
                "method": "markdown-uri",
                "relation": "citation-or-external-uri",
                "context": f"Docling Markdown line {line_number}: {context}",
            })
    return links


def _dedupe_links(links: list[dict]) -> list[dict]:
    """Dedupe and sort links without comparing nullable page values."""
    unique = {
        (row["target_url"], row["page"], row["method"], row["context"]): row
        for row in links
        if row.get("target_url") and row.get("context")
    }
    return [unique[key] for key in sorted(
        unique, key=lambda value: (
            value[0], value[1] is None,
            -1 if value[1] is None else int(value[1]), value[2], value[3]))]


def _write_gzip(path: Path, rows: list[dict]) -> tuple[int, str]:
    logical = hashlib.sha256()
    with path.open("wb") as raw:
        with deterministic_gzip_open(raw, compresslevel=9) as out:
            for row in rows:
                encoded = (canonical_json(row) + "\n").encode("utf-8")
                out.write(encoded); logical.update(encoded)
        raw.flush(); os.fsync(raw.fileno())
    return len(rows), logical.hexdigest()


def run(*, l0_receipt: Path, artifact_root: Path, root: Path, limit: int,
        start: int = 0) -> dict:
    artifact_root = artifact_root.resolve(); root = root.resolve()
    root.relative_to(artifact_root)
    if root.exists():
        raise ValueError(f"immutable pilot root exists: {root}")
    api_rows, selected = [], []
    pdf_count = 0
    for uri, line_number, row in _iter_l0(l0_receipt.resolve(), artifact_root):
        kind = row.get("kind")
        if kind == "api":
            api_rows.append((uri, line_number, row))
        elif kind == "pdf":
            ordinal = pdf_count
            pdf_count += 1
            if start <= ordinal < start + limit:
                selected.append((uri, line_number, row))
    metadata = _metadata(api_rows)
    if len(selected) != limit:
        raise ValueError("pilot limit exceeds retained PDFs")

    partial = root.with_name(root.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import DocumentStream, InputFormat
        from docling.datamodel.accelerator_options import (
            AcceleratorDevice, AcceleratorOptions)
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        import importlib.metadata

        options = PdfPipelineOptions(); options.do_ocr = False
        options.do_table_structure = True
        options.accelerator_options = AcceleratorOptions(
            num_threads=2, device=AcceleratorDevice.CPU)
        converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
        originals_by_sha: dict[str, Path] = {}
        source_record_ids: set[str] = set()
        markdown_rows, outlink_rows = [], []
        original_logical = hashlib.sha256()
        for local_ordinal, (l0_uri, line_number, source) in enumerate(selected):
            ordinal = start + local_ordinal
            body = _decode(source, f"{l0_uri}:{line_number}")
            original = partial / "original" / f"{source['sha256']}.pdf"
            original.parent.mkdir(parents=True, exist_ok=True)
            if source["sha256"] not in originals_by_sha:
                original.write_bytes(body)
                originals_by_sha[source["sha256"]] = original
            canonical_url = _source_record_id(source)
            if canonical_url in source_record_ids:
                raise ValueError(f"duplicate canonical FINMA PDF URL: {canonical_url}")
            source_record_ids.add(canonical_url)
            original_logical.update((canonical_json({
                "source_record_id": canonical_url,
                "body_sha256": source["sha256"],
                "bytes": len(body),
            }) + "\n").encode("utf-8"))
            result = converter.convert(DocumentStream(
                name=original.name, stream=io.BytesIO(body)))
            if str(result.status).casefold().endswith("failure") or result.errors:
                raise ValueError(f"Docling conversion failed: {source['url']}: {result.errors}")
            markdown = result.document.export_to_markdown()
            if not markdown.strip():
                raise ValueError(f"empty Docling Markdown: {source['url']}")
            meta = metadata.get(canonical_url or source["url"], {})
            parent_id = canonical_url
            pages = [{"page_number": int(number),
                      "width": float(page.size.width), "height": float(page.size.height)}
                     for number, page in sorted(result.document.pages.items())]
            markdown_rows.append({
                "source_record_id": parent_id, "ordinal": ordinal,
                "url": source["url"], "canonical_url": canonical_url,
                "title": meta.get("title", ""),
                "description": meta.get("description", ""),
                "release_date": meta.get("release_date", ""),
                "language": meta.get("language", "und"),
                "markdown": markdown,
                "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
                "parent_pdf": {"sha256": source["sha256"], "bytes": len(body),
                               "l0_uri": l0_uri, "l0_line": line_number},
                "api_parent": meta.get("api_parent"),
                "extraction": {"adapter": "docling", "version":
                               importlib.metadata.version("docling"),
                               "ocr": False, "table_structure": True,
                               "accelerator_device": "cpu", "threads": 2,
                               "pages": pages, "errors": []},
            })
            links = _pdf_annotation_links(body) + _markdown_links(markdown)
            links = _dedupe_links(links)
            outlink_rows.append({
                "source_record_id": parent_id,
                "parent_url": canonical_url,
                "parent_markdown_sha256": markdown_rows[-1]["markdown_sha256"],
                "links": links,
                "explicit_empty": not bool(links),
            })

        markdown_path = partial / "markdown.jsonl.gz"
        markdown_count, markdown_logical = _write_gzip(markdown_path, markdown_rows)
        outlinks_path = partial / "outlinks.jsonl.gz"
        outlinks_count, outlinks_logical = _write_gzip(outlinks_path, outlink_rows)
        original_stage = SS.seal(
            artifact_root=artifact_root,
            artifact_paths=list(originals_by_sha.values()),
            receipt_path=partial / "ORIGINAL-RECEIPT.json", records=limit,
            fmt="content-addressed-exact-application-pdf-bodies+logical-source-url-map",
            logical_sha256=original_logical.hexdigest(),
            producer_id="pipeline/materialize_finma_pdf_pilot.py",
            producer_version=VERSION, publish_from=partial, publish_to=root)
        markdown_stage = SS.seal(
            artifact_root=artifact_root, artifact_paths=[markdown_path],
            receipt_path=partial / "MARKDOWN-RECEIPT.json", records=markdown_count,
            fmt="deterministic-gzip-jsonl-full-docling-markdown+page-diagnostics",
            logical_sha256=markdown_logical,
            producer_id="pipeline/materialize_finma_pdf_pilot.py",
            producer_version=VERSION, publish_from=partial, publish_to=root)
        outlink_stage = SS.seal(
            artifact_root=artifact_root, artifact_paths=[outlinks_path],
            receipt_path=partial / "OUTLINKS-RECEIPT.json", records=outlinks_count,
            fmt="deterministic-gzip-jsonl-contextual-pdf-annotation-and-markdown-uri-links-including-explicit-empty-parents",
            logical_sha256=outlinks_logical,
            producer_id="pipeline/materialize_finma_pdf_pilot.py",
            producer_version=VERSION, publish_from=partial, publish_to=root)
        value = {"schema": "aw.finma-pdf-markdown-pilot-run.v1", "version": VERSION,
                 "status": "bounded-pilot-complete-not-production-admission",
                 "selection": "contiguous [start,start+limit) PDFs in the hash-bound L0 artifact/row order",
                 "start": start, "limit": limit, "retained_pdf_population": pdf_count,
                 "l0_receipt": {"uri": f"artifact://{l0_receipt.resolve().relative_to(artifact_root).as_posix()}",
                                "sha256": sha256_file(l0_receipt)},
                 "stages": {"original": original_stage, "markdown": markdown_stage,
                            "outlinks": outlink_stage},
                 "gate": {"full_markdown_documents": markdown_count,
                          "explicit_outlink_parents": outlinks_count,
                          "metadata_fallbacks": 0, "content_caps": 0}}
        (partial / "RUN.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.rename(partial, root)
        return value
    except Exception:
        if partial.exists(): shutil.rmtree(partial)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l0-receipt", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    if args.limit <= 0: parser.error("--limit must be positive")
    if args.start < 0: parser.error("--start must be non-negative")
    print(json.dumps(run(l0_receipt=args.l0_receipt, artifact_root=args.artifact_root,
                         root=args.root, limit=args.limit, start=args.start),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
