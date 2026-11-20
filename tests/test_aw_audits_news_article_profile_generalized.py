"""Task 450: the news_article auditor certifies the pack it is given.

It used to be pinned to the Aug-14 pack by four literal counts and one literal
`source_id`, so a fresh delta could never obtain a profile-compliance receipt.
These tests fix the generalization in place: the pack under audit is an input,
upstream-known counts stay pinnable exactly, and every semantic and
conservation check still fails closed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from cytir.core.ids import mint_point_id

import pytest

from cytir.ops.audits import news_article_profile as A
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


def test_an_arbitrary_pack_can_be_certified_and_names_itself(tmp_path):
    root, run, pack_root = build(tmp_path)
    evidence_path, checks_path = run_audit(tmp_path, root, run, pack_root)

    evidence = json.loads(evidence_path.read_text())
    assert evidence["source_id"] == SOURCE_ID
    assert evidence["counts"]["parents"] == 2
    assert evidence["counts"]["chunks"] == 4
    assert evidence["counts"]["unique_point_ids"] == 4
    assert evidence["counts"]["empty_outlink_parents"] == 1
    assert evidence["counts"]["source_declared_bylines"] == 1
    assert evidence["declared_counts"]["markdown_documents"] == 2

    checks = json.loads(checks_path.read_text())
    profile = json.loads(REGISTRY.read_text())["profiles"]["news_article"]
    assert set(checks) == set(profile["required_stages"])
    for stage in profile["required_stages"]:
        assert [row["requirement"] for row in checks[stage]] == \
            profile.get(stage, {}).get("requirements", [])
        assert all(row["status"] == "PASS" for row in checks[stage])
        assert all(row["verification_method"].strip() for row in checks[stage])
    # No literal from the Aug-14 pack survives in the emitted methods.
    assert "4,993" not in json.dumps(checks)


def test_upstream_known_counts_stay_pinnable_exactly(tmp_path):
    root, run, pack_root = build(tmp_path)
    run_audit(tmp_path, root, run, pack_root,
              expect_counts={"article_parents": 2, "chunk_rows": 4})
    with pytest.raises(ValueError, match="counts.article_parents"):
        run_audit(tmp_path, root, run, pack_root,
                  expect_counts={"article_parents": 4993})


def test_a_run_that_does_not_declare_its_population_is_refused(tmp_path):
    root, run, pack_root = build(
        tmp_path, counts={"article_parents": 2, "quarantine_records": 0})
    with pytest.raises(ValueError, match="markdown_documents"):
        run_audit(tmp_path, root, run, pack_root)


def test_declared_parent_count_must_match_the_sealed_markdown(tmp_path):
    root, run, pack_root = build(
        tmp_path, counts={"article_parents": 3, "markdown_documents": 3,
                          "quarantine_records": 0})
    with pytest.raises(ValueError, match="markdown parents=2"):
        run_audit(tmp_path, root, run, pack_root)


def test_a_wrong_run_status_is_refused(tmp_path):
    root, run, pack_root = build(tmp_path)
    with pytest.raises(ValueError, match="run status"):
        run_audit(tmp_path, root, run, pack_root, run_status="something-else")


def test_chunks_must_bind_their_parent_markdown_digest(tmp_path):
    root, run, pack_root = build(tmp_path, break_chunk_digest=True)
    with pytest.raises(ValueError, match="parent digest mismatch"):
        run_audit(tmp_path, root, run, pack_root)


def test_every_parent_must_have_an_outlink_representation(tmp_path):
    root, run, pack_root = build(tmp_path, break_outlinks=True)
    with pytest.raises(ValueError, match="outlink parent conservation"):
        run_audit(tmp_path, root, run, pack_root)


def test_markdown_without_a_body_is_refused(tmp_path):
    root, run, pack_root = build(tmp_path, break_markdown_body=True)
    with pytest.raises(ValueError, match="full body"):
        run_audit(tmp_path, root, run, pack_root)


def test_the_pinned_news_specific_chunk_policy_is_still_mandatory(tmp_path):
    root, run, pack_root = build(tmp_path)
    pack_path = pack_root / "PACK.json"
    pack = json.loads(pack_path.read_text())
    pack["chunk_policy"] = {"path": "data/legacy.json",
                            "policy_id": "legacy-180w-o30",
                            "sha256": "0" * 64}
    pack_path.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="pinned news-specific policy"):
        run_audit(tmp_path, root, run, pack_root)


def test_the_audit_output_is_immutable(tmp_path):
    root, run, pack_root = build(tmp_path)
    first = run_audit(tmp_path, root, run, pack_root)
    again = run_audit(tmp_path, root, run, pack_root)
    assert first == again
    tampered = first[0]
    body = json.loads(tampered.read_text())
    body["counts"]["parents"] = 99
    tampered.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="immutable news audit collision"):
        run_audit(tmp_path, root, run, pack_root)
