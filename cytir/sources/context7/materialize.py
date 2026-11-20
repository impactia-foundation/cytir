#!/usr/bin/env python3
"""Materialize preserved Context7 GitHub paths as versioned Markdown/outlinks.

Moved from the retired pipeline driver `materialize_context7_exact_paths` (task 1411, S7).
`receipt_rows` is the per-receipt step, shared with the `context7_exact` family adapter
(`cytir.sources.families.context7_exact`); `materialize` is the sealed-stage run, unchanged.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from cytir.extract.github_agent_resources import (compose_document_title, extract_outlinks,
                                      source_provenance, source_to_markdown)
from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire import preserved_stage_io as PS
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as SS


# v2 (task 517), and the bump is deliberate because it changes what the
# collection contains: the L0 provenance leaves the Markdown body for the
# record's own `official_source_identity`, and every document carries the
# `title` production gives it. v1 shipped 126,427 rows with no title at all and
# 72,330 whose snippet -- and whose embedding's head -- was the YAML front
# matter. See `acquire_github_repo_docs.source_provenance` /
# `compose_document_title`.
VERSION = "context7-exact-path-materialization.v2"
PRODUCER = "pipeline/materialize_context7_exact_paths.py"
RUN_SCHEMAS = {
    "aw.context7-exact-path-acquisition-run.v1",
    "aw.context7-missing-git-source-run.v1",
}
RECEIPT_SCHEMAS = {
    "aw.context7-exact-path-repository-receipt.v1",
    "aw.context7-missing-git-source-receipt.v1",
}


def _repository_pair_count(counts: dict) -> int | None:
    return counts.get("tranche_repository_ref_pairs",
                      counts.get("repository_ref_pairs"))


def _artifact(path: Path, artifact_root: Path) -> dict:
    relative = path.resolve().relative_to(artifact_root.resolve())
    return {"uri": f"artifact://{relative.as_posix()}",
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _bytes(artifact_root: Path, artifact: dict) -> bytes:
    PS.validate_artifact(artifact_root, artifact)
    return gzip.decompress(PS.artifact_path(artifact_root, artifact).read_bytes())


def _stage(*, artifact_root: Path, output_root: Path,
           name: str, rows: list[dict]) -> dict:
    body = b"".join((canonical_json(row) + "\n").encode() for row in rows)
    artifact = PS.store_artifact(output_root, f"stages/{name}/data", "jsonl", body)
    return SS.seal(
        artifact_root=artifact_root,
        artifact_paths=[PS.artifact_path(output_root, artifact)],
        receipt_path=output_root / "stages" / name / "receipt.json",
        records=len(rows), fmt="canonical-jsonl+deterministic-gzip",
        logical_sha256=hashlib.sha256(body).hexdigest(),
        producer_id=PRODUCER, producer_version=VERSION,
    )


def _upstream_evidence(*, source_root: Path, source_run: Path,
                       artifact_root: Path, expected_complete: int,
                       expected_recovery: int) -> tuple[dict, list[Path]]:
    run = json.loads(source_run.read_text())
    counts = run.get("counts") or {}
    statuses = counts.get("document_statuses") or {}
    if (run.get("schema") not in RUN_SCHEMAS
            or run.get("status") != "complete"
            or statuses.get("complete", 0) != expected_complete
            or statuses.get("quarantine", 0) != expected_recovery):
        raise ValueError("Context7 exact-path source RUN is not the sealed tranche")
    stage = run["stages"]["original"]
    _validate_stage(stage, "upstream.original", artifact_root=artifact_root,
                    deep=True)
    receipt_path = PS.artifact_path(artifact_root, stage["receipt"])
    receipt = json.loads(receipt_path.read_text())
    evidence = [source_run, receipt_path]
    evidence.extend(PS.artifact_path(artifact_root, row)
                    for row in receipt["artifacts"])
    return run, sorted(set(evidence))


def receipt_rows(receipt_path: Path, artifact_root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """One terminal receipt -> (markdown rows, outlink rows, recovery rows), unsorted. The body of
    the run loop, unchanged, so the family adapter and `materialize` cannot drift."""
    markdown_rows: list[dict] = []
    outlink_rows: list[dict] = []
    recovery_rows: list[dict] = []
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") not in RECEIPT_SCHEMAS
            or receipt.get("status") != "terminal"):
        raise ValueError("Context7 terminal receipt schema/status drift")
    repository = str(receipt["owner_repo"])
    revision_value = receipt.get("resolved_revision")
    revision = str(revision_value) if revision_value is not None else None
    if (any(document.get("status") == "complete"
            for document in receipt["documents"])
            and (revision is None or len(revision) != 40
                 or any(char not in "0123456789abcdef" for char in revision))):
        raise ValueError("Context7 resolved revision is not a pinned commit")
    receipt_artifact = _artifact(receipt_path, artifact_root)
    for document in receipt["documents"]:
        source_path = str(document["source_path"])
        if document["status"] != "complete":
            recovery_rows.append({
                "repository": repository,
                "requested_ref": receipt["requested_ref"],
                "resolved_revision": revision,
                "source_path": source_path,
                "reason": document.get("reason"),
                "terminal_receipt": receipt_artifact,
                "evidence": document.get("artifacts") or {},
                "status": "recovery_required",
            })
            continue
        raw_artifact = document["artifacts"]["response_body"]
        raw = _bytes(artifact_root, raw_artifact)
        if not raw:
            raise ValueError("complete Context7 source blob is empty")
        source_url = (
            f"https://github.com/{repository}/blob/{revision}/"
            + quote(source_path, safe="/")
        )
        markdown = source_to_markdown(
            raw, source_path=source_path).decode("utf-8")
        identity = source_provenance(
            owner_repo=repository, revision=revision,
            source_path=source_path, parent_sha256=raw_artifact["sha256"])
        title = compose_document_title(markdown, source_path)
        document_id = hashlib.sha256(
            f"context7-github-document\0{repository}\0{revision}\0{source_path}"
            .encode()).hexdigest()
        markdown_rows.append({
            "document_id": document_id, "url": source_url, "lang": None,
            "repository": repository, "revision": revision,
            "requested_ref": receipt["requested_ref"],
            "source_path": source_path,
            "git_blob_sha1": document["git_blob_sha1"],
            "parent_original_sha256": raw_artifact["sha256"],
            # `title` is the field `stage_structured_markdown_for_embedding`
            # copies into `source_metadata`, which is where
            # `aw_serving_metadata._pack_title` looks. Without it the
            # projection has nothing record-native to read and the row ships
            # untitled -- 126,427 of them did.
            "title": title,
            # The provenance that used to be the Markdown's first six lines.
            "official_source_identity": identity,
            "markdown": markdown,
            "lineage": {"terminal_receipt": receipt_artifact,
                        "response_body": raw_artifact},
            "full_prechunk_content": True,
        })
        links = extract_outlinks(markdown, source_url)
        if links:
            for ordinal, link in enumerate(links):
                outlink_rows.append({
                    "document_id": document_id, "source_url": source_url,
                    "repository": repository, "revision": revision,
                    "source_path": source_path, "ordinal": ordinal,
                    "target_url": link["target_url"],
                    "anchor_or_context": link["anchor_or_context"],
                    "is_empty": False,
                })
        else:
            outlink_rows.append({
                "document_id": document_id, "source_url": source_url,
                "repository": repository, "revision": revision,
                "source_path": source_path, "ordinal": 0,
                "target_url": None, "anchor_or_context": None,
                "is_empty": True,
            })
    return markdown_rows, outlink_rows, recovery_rows


def materialize(*, source_root: Path, source_run: Path, output_root: Path,
                artifact_root: Path, expected_complete: int,
                expected_recovery: int) -> dict:
    if output_root.exists():
        raise ValueError("Context7 materialization output root must be fresh")
    upstream, evidence_paths = _upstream_evidence(
        source_root=source_root, source_run=source_run,
        artifact_root=artifact_root, expected_complete=expected_complete,
        expected_recovery=expected_recovery)
    terminal_paths = sorted((source_root / "receipts").glob("*.json"))
    upstream_counts = upstream["counts"]
    expected_pairs = _repository_pair_count(upstream_counts)
    if not isinstance(expected_pairs, int) or len(terminal_paths) != expected_pairs:
        raise ValueError("Context7 repository/ref receipt conservation failed")

    markdown_rows: list[dict] = []
    outlink_rows: list[dict] = []
    recovery_rows: list[dict] = []
    for receipt_path in terminal_paths:
        md, ol, rec = receipt_rows(receipt_path, artifact_root)
        markdown_rows.extend(md)
        outlink_rows.extend(ol)
        recovery_rows.extend(rec)

    markdown_rows.sort(key=lambda row: (row["repository"].casefold(),
                                         row["revision"], row["source_path"]))
    outlink_rows.sort(key=lambda row: (row["source_url"], row["ordinal"]))
    recovery_rows.sort(key=lambda row: (row["repository"].casefold(),
                                        row["source_path"]))
    if (len(markdown_rows) != expected_complete
            or len(recovery_rows) != expected_recovery):
        raise ValueError("Context7 source document conservation failed")
    if len({row["document_id"] for row in markdown_rows}) != len(markdown_rows):
        raise ValueError("Context7 document identities are not unique")

    output_root.mkdir(parents=True)
    inventory = [_artifact(path, artifact_root) for path in evidence_paths]
    original_logical = hashlib.sha256(b"".join(
        (canonical_json(row) + "\n").encode() for row in inventory)).hexdigest()
    stages = {
        "original": SS.seal(
            artifact_root=artifact_root, artifact_paths=evidence_paths,
            receipt_path=output_root / "stages/original/receipt.json",
            records=expected_complete,
            fmt="pinned-git-tree+exact-raw-blob+transport+terminal-receipts",
            logical_sha256=original_logical, producer_id=PRODUCER,
            producer_version=VERSION),
        "markdown": _stage(artifact_root=artifact_root, output_root=output_root,
                           name="markdown", rows=markdown_rows),
        "outlinks": _stage(artifact_root=artifact_root, output_root=output_root,
                           name="outlinks", rows=outlink_rows),
        "recovery": _stage(artifact_root=artifact_root, output_root=output_root,
                           name="recovery", rows=recovery_rows),
    }
    result = {
        "schema": "aw.context7-exact-path-materialization-run.v1",
        "version": VERSION, "status": "ready_for_current_policy_chunking",
        "source_profile": "software_docs",
        "counts": {
            "source_frontier_documents": expected_complete + expected_recovery,
            "markdown_documents": len(markdown_rows),
            "recovery_documents": len(recovery_rows),
            "outlink_records_including_empty": len(outlink_rows),
            "explicit_empty_outlink_parents": sum(
                row["is_empty"] for row in outlink_rows),
        },
        "stages": stages,
        "index_admission":
            "blocked_until_current_chunks_profile_fidelity_and_vectors",
    }
    atomic_publish(output_root / "RUN.json",
                   (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    for name, stage in stages.items():
        _validate_stage(stage, name, artifact_root=artifact_root, deep=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "source-run", "output-root", "artifact-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-complete", type=int, required=True)
    parser.add_argument("--expected-recovery", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(
        source_root=args.source_root.resolve(),
        source_run=args.source_run.resolve(),
        output_root=args.output_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        expected_complete=args.expected_complete,
        expected_recovery=args.expected_recovery), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
