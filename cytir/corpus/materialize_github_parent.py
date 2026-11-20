#!/usr/bin/env python3
"""Close a terminal GitHub acquisition as portable Markdown/outlink stages.

Stage module (task 1594): a whole-root post-acquisition pass over the family
`github_agent_resources` (receipt validation, four sealed stages under
`derived-parent-v1/`, conservation check, immutable RUN.json). No HTTP.
Run: `python3 -m cytir.corpus.materialize_github_parent --root <terminal root>`.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from cytir.extract.github_agent_resources import validate_receipt
from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire import preserved_stage_io as PS
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as SS


VERSION = "github-acquisition-parent.v1"
# Frozen receipt label (task 1594): sealed receipts carry this string; it names the
# pre-move driver path and never changes, whatever the code home is.
PRODUCER = "pipeline/materialize_github_acquisition_parent.py"
DERIVED = "derived-parent-v1"


def _logical(rows: list[dict]) -> tuple[bytes, str]:
    body = b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)
    return body, hashlib.sha256(body).hexdigest()


def _stage(root: Path, name: str, rows: list[dict], *, fmt: str) -> dict:
    body, logical = _logical(rows)
    artifact = PS.store_artifact(
        root, f"{DERIVED}/stages/{name}/data", "jsonl", body)
    return SS.seal(
        artifact_root=root, artifact_paths=[PS.artifact_path(root, artifact)],
        receipt_path=root / DERIVED / "stages" / name / "receipt.json",
        records=len(rows), fmt=fmt, logical_sha256=logical,
        producer_id=PRODUCER, producer_version=VERSION)


def materialize(*, root: Path) -> dict:
    acquisition_run_path = root / "RUN.json"
    acquisition = json.loads(acquisition_run_path.read_text(encoding="utf-8"))
    counts = acquisition.get("counts") or {}
    if (acquisition.get("schema") != "aw.github-doc-acquisition-run.v1" or
            acquisition.get("status") != "complete" or
            counts.get("queue_repositories") != counts.get("terminal_repositories") or
            counts.get("remaining_repositories") != 0 or
            counts.get("attempt_failures") != 0):
        raise ValueError("GitHub acquisition is not exact-terminal")

    receipt_paths = sorted((root / "receipts").glob("*.json"))
    if len(receipt_paths) != counts["queue_repositories"]:
        raise ValueError("receipt-file conservation disagrees with acquisition RUN")
    markdown_rows: list[dict] = []
    outlink_rows: list[dict] = []
    quarantine_rows: list[dict] = []
    complete = quarantined = 0
    for receipt_path in receipt_paths:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        repository = str(receipt.get("repository") or "")
        validate_receipt(root, receipt, repository)
        if receipt["status"] == "quarantined":
            quarantined += 1
            quarantine = receipt["quarantine"]
            quarantine_rows.append({
                "repository": repository,
                "reason": quarantine["reason"],
                "detail": quarantine.get("detail"),
                "receipt_sha256": sha256_file(receipt_path),
                "evidence": quarantine["evidence"],
            })
            continue
        complete += 1
        revision = str(receipt["revision"])
        for document in receipt["documents"]:
            markdown_artifact = document["markdown"]
            markdown = gzip.decompress(
                PS.artifact_path(root, markdown_artifact).read_bytes()).decode("utf-8")
            source_path = str(document["source_path"])
            source_url = str(document["source_url"])
            document_id = hashlib.sha256(
                f"github-doc\0{repository}\0{revision}\0{source_path}".encode()).hexdigest()
            markdown_rows.append({
                "document_id": document_id, "url": source_url, "lang": None,
                "repository": repository, "revision": revision,
                "source_path": source_path,
                "parent_original_sha256": document["original"]["body_sha256"],
                "parent_markdown_sha256": markdown_artifact["body_sha256"],
                "markdown": markdown,
            })
            outlink_payload = json.loads(gzip.decompress(
                PS.artifact_path(root, document["outlinks"]).read_bytes()))
            links = outlink_payload.get("links") or []
            if len(links) != document["outlinks"]["records"]:
                raise ValueError("outlink artifact count disagrees with receipt")
            for ordinal, link in enumerate(links):
                outlink_rows.append({
                    "document_id": document_id, "source_url": source_url,
                    "target_url": link.get("target_url"),
                    "anchor": link.get("anchor_or_context"),
                    "ordinal": ordinal,
                })
    markdown_rows.sort(key=lambda row: (row["repository"].casefold(), row["source_path"]))
    outlink_rows.sort(key=lambda row: (row["document_id"], row["ordinal"]))
    quarantine_rows.sort(key=lambda row: row["repository"].casefold())
    if (complete != counts["complete_repositories"] or
            quarantined != counts["quarantined_repositories"] or
            len(markdown_rows) != counts["markdown_documents"] or
            len(outlink_rows) != counts["outlinks"]):
        raise ValueError("derived stages do not conserve terminal RUN counts")

    # The original/evidence stage is exhaustive over the immutable acquisition
    # tree that existed before this adapter's namespaced derived outputs.
    original_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and
        DERIVED not in path.relative_to(root).parts)
    inventory_rows = [{
        "uri": f"artifact://{path.relative_to(root).as_posix()}",
        "bytes": path.stat().st_size, "sha256": sha256_file(path),
    } for path in original_paths]
    _, original_logical = _logical(inventory_rows)
    original = SS.seal(
        artifact_root=root, artifact_paths=original_paths,
        receipt_path=root / DERIVED / "stages/original/receipt.json",
        records=len(receipt_paths), fmt="terminal-github-api-source-evidence-tree",
        logical_sha256=original_logical, producer_id=PRODUCER,
        producer_version=VERSION)
    stages = {
        "original": original,
        "markdown": _stage(root, "markdown", markdown_rows,
                           fmt="canonical-jsonl+deterministic-gzip"),
        "outlinks": _stage(root, "outlinks", outlink_rows,
                           fmt="canonical-jsonl+deterministic-gzip"),
        "quarantine": _stage(root, "quarantine", quarantine_rows,
                             fmt="canonical-jsonl+deterministic-gzip"),
    }
    result = {
        "schema": "aw.github-derived-parent-run.v1", "version": VERSION,
        "status": "ready_for_current_policy_chunking",
        "source_profile": "software_docs", "acquisition_run": {
            "uri": "artifact://RUN.json", "sha256": sha256_file(acquisition_run_path)},
        "counts": {"queue_repositories": len(receipt_paths),
                   "complete_repositories": complete,
                   "quarantined_repositories": quarantined,
                   "markdown_documents": len(markdown_rows),
                   "outlinks": len(outlink_rows)},
        "stages": stages,
        "index_admission": "blocked_until_current_policy_chunks_and_vectors_seal",
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    run_path = root / DERIVED / "RUN.json"
    if run_path.exists():
        if sha256_file(run_path) != hashlib.sha256(encoded).hexdigest():
            raise ValueError("immutable derived RUN collision")
    else:
        atomic_publish(run_path, encoded)
    for name, stage in stages.items():
        _validate_stage(stage, name, artifact_root=root, deep=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    result = materialize(root=parser.parse_args(argv).root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
