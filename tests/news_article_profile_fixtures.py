"""Shared news_article profile audit fixtures (task 1570; moved beside its test, task 2250): build and run_audit recovered from the test module deleted by merge dcee56b04. Holds no tests."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


# task 1743 (CI #1374 importers): this checkout's repo root first -- `python path.py` puts only this file's own directory on sys.path
import pathlib as _pathlib_root  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, str(_pathlib_root.Path(__file__).resolve().parents[1]))
from cytir.core.ids import mint_point_id
import cytir.ops.audits.news_article_profile as A
from cytir.acquire import stage_receipt as SS


REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data" / "source-preservation-profiles.v1.json"
SOURCE_ID = "task450-fixture-news-v1"
URLS = ("https://publisher.test/a", "https://publisher.test/b")


def document_id(url: str) -> str:
    return hashlib.sha256(f"news-document\0{url}".encode()).hexdigest()


def envelope(headline, byline, publisher, fetched_at, url, body):
    return (f"# {headline}\n\nByline: {byline}\n\n"
            f"Publisher: {publisher}\n\nFetched: {fetched_at}\n\n"
            f"Source: {url}\n\n{body}\n")


def _seal(root: Path, name: str, rows: list[dict], fmt: str) -> dict:
    path = root / f"{name}.jsonl.gz"
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows)
    with gzip.GzipFile(filename="", mode="wb", mtime=0,
                       fileobj=path.open("wb")) as handle:
        handle.write(payload.encode("utf-8"))
    return SS.seal(
        artifact_root=root, artifact_paths=[path],
        receipt_path=root / f"{name}.receipt.json", records=len(rows),
        fmt=fmt, logical_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        producer_id="test_audit_news_article_profile_generalized.py",
        producer_version="task450.v1")


def build(tmp_path: Path, *, chunks_per_doc=2, counts=None,
          break_chunk_digest=False, break_outlinks=False,
          break_markdown_body=False) -> tuple[Path, Path, Path]:
    root = tmp_path / "artifacts"
    root.mkdir()
    markdown, outlinks, chunks = [], [], []
    for index, url in enumerate(URLS):
        did = document_id(url)
        fetched_at = f"2026-08-28T0{index}:00:00Z"
        headline = f"Headline {index}"
        byline = "A Reporter" if index == 0 else None
        body = f"Body of article {index}. " * 20
        text = envelope(headline, byline or A_ABSENT, "publisher.test",
                        fetched_at, url,
                        "" if break_markdown_body else body)
        markdown.append({
            "document_id": did, "url": url, "markdown": text,
            "headline": headline, "byline": byline,
            "byline_status": ("source_declared" if byline
                              else "not_declared_by_publisher"),
            "publication_time": "2026-08-27" if index == 0 else None,
            "publication_time_status": ("source_declared" if index == 0
                                        else "not_declared_by_publisher"),
            "publisher": "publisher.test", "fetched_at": fetched_at,
            "lineage": {"fetched_at": fetched_at, "original_sha256": "0" * 64},
        })
        if index == 0 and not break_outlinks:
            outlinks.append({"document_id": did, "publisher": "publisher.test",
                             "is_empty": False,
                             "target_url": "https://ref.test/law",
                             "anchor_or_context": "Bundesgesetz"})
        elif not break_outlinks:
            outlinks.append({"document_id": did, "publisher": "publisher.test",
                             "is_empty": True, "target_url": None,
                             "anchor_or_context": None})
        parent_sha = hashlib.sha256(text.encode()).hexdigest()
        for ordinal in range(chunks_per_doc):
            chunks.append({
                "point_id": mint_point_id(url, str(ordinal)),
                "document_id": did, "chunk_ordinal": ordinal,
                "parent_markdown_sha256": ("f" * 64 if break_chunk_digest
                                           else parent_sha),
                "source_metadata": {key: markdown[-1].get(key)
                                    for key in A.NEWS_KEYS},
            })
    stages = {
        "original": _seal(root, "original",
                          [{"document_id": document_id(u), "url": u}
                           for u in URLS], "news-original"),
        "markdown": _seal(root, "markdown", markdown, "news-markdown"),
        "outlinks": _seal(root, "outlinks", outlinks, "news-outlinks"),
    }
    run = tmp_path / "RUN.json"
    run.write_text(json.dumps({
        "schema": "aw.news-lane-run.v1",
        "status": "ready_for_current_policy_chunking",
        "counts": counts or {"article_parents": len(URLS),
                             "markdown_documents": len(URLS),
                             "quarantine_records": 0,
                             "chunk_rows": len(chunks)},
        "stages": stages}, indent=2, sort_keys=True) + "\n")

    pack_root = tmp_path / "pack"
    pack_root.mkdir()
    chunk_stage = _seal(pack_root, "chunks", chunks, "news-chunks")
    policy_path = REPO / A.NEWS_POLICY_PATH
    (pack_root / "PACK.json").write_text(json.dumps({
        "source_profile": "news_article",
        "chunk_policy": {"path": A.NEWS_POLICY_PATH,
                         "policy_id": A.NEWS_POLICY_ID,
                         "sha256": hashlib.sha256(
                             policy_path.read_bytes()).hexdigest()},
        "chunks": chunk_stage}, indent=2, sort_keys=True) + "\n")
    return root, run, pack_root


A_ABSENT = "not declared by publisher"


def run_audit(tmp_path, root, run, pack_root, **kwargs):
    return A.audit(parent_root=root, parent_run=run, pack_root=pack_root,
                   output_root=tmp_path / "audit", registry_path=REGISTRY,
                   source_id=kwargs.pop("source_id", SOURCE_ID), **kwargs)
