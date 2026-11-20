"""Combine the sealed KALI global projection with its increment recovery."""
from __future__ import annotations
import gzip
import hashlib
import json
import os
from pathlib import Path

from cytir.acquire.artifact_io import canonical_json, sha256_file
from cytir.acquire.deterministic_gzip import deterministic_gzip_text
from cytir.acquire import stage_receipt as SS


VERSION = "kali-exact-production-combine.v1"
PRODUCER = "pipeline/combine_kali_production_markdown.py"


def _binding(root: Path, receipt: dict, suffix: str) -> Path:
    rows = [row for row in receipt["artifacts"] if row["uri"].endswith(suffix)]
    if len(rows) != 1:
        raise ValueError(f"expected one {suffix!r} artifact")
    row = rows[0]
    path = root / row["uri"].removeprefix("artifact://")
    if (not path.is_file() or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]):
        raise ValueError(f"artifact differs from receipt: {path}")
    return path


def _write(handle, logical, value: dict) -> None:
    encoded = canonical_json(value) + "\n"
    handle.write(encoded); logical.update(encoded.encode())


def combine(*, artifact_root: Path, base_root: Path, recovery_root: Path,
            output: Path, expected: int = 128_990,
            supersedes_run: Path | None = None) -> dict:
    artifact_root = artifact_root.resolve(); output = output.resolve()
    output.relative_to(artifact_root)
    if output.exists():
        raise ValueError(f"immutable output exists: {output}")
    base_run_path = base_root / "RUN.json"
    recovery_run_path = recovery_root / "RUN.json"
    base = json.loads(base_run_path.read_text())
    recovery = json.loads(recovery_run_path.read_text())
    if (base.get("status") != "recovery-frontier"
            or recovery.get("status") != "complete"
            or recovery["counts"]["unresolved_source_ids"] != 0
            or base["counts"]["markdown_documents"]
            + recovery["counts"]["recovered_source_ids"] != expected):
        raise ValueError("KALI exact-production recovery is not complete")
    base_l0 = base.get("l0") or {}
    base_l0_receipt = artifact_root / str(base_l0.get("uri") or "").removeprefix("artifact://")
    base_archive_binding = base_l0.get("archive") or {}
    base_archive = artifact_root / str(
        base_archive_binding.get("uri") or "").removeprefix("artifact://")
    increment_receipt_binding = recovery.get("increments") or {}
    increment_receipt_path = artifact_root / str(
        increment_receipt_binding.get("uri") or "").removeprefix("artifact://")
    for path, binding in ((base_l0_receipt, base_l0),
                          (base_archive, base_archive_binding),
                          (increment_receipt_path, increment_receipt_binding)):
        if (not path.is_file() or path.stat().st_size != binding.get("bytes")
                or sha256_file(path) != binding.get("sha256")):
            raise ValueError(f"KALI original binding failure: {path}")
    increment_receipt = json.loads(increment_receipt_path.read_text())
    original_paths = [base_l0_receipt, base_archive, increment_receipt_path]
    for binding in increment_receipt.get("artifacts") or []:
        path = artifact_root / binding["uri"].removeprefix("artifact://")
        if (not path.is_file() or path.stat().st_size != binding["bytes"]
                or sha256_file(path) != binding["sha256"]):
            raise ValueError(f"KALI increment original binding failure: {path}")
        original_paths.append(path)
    base_markdown_receipt = json.loads((base_root / "MARKDOWN-RECEIPT.json").read_text())
    base_outlinks_receipt = json.loads((base_root / "OUTLINKS-RECEIPT.json").read_text())
    recovery_markdown_receipt = json.loads((recovery_root / "MARKDOWN-RECEIPT.json").read_text())
    recovery_outlinks_receipt = json.loads((recovery_root / "OUTLINKS-RECEIPT.json").read_text())
    markdown_inputs = (
        _binding(artifact_root, base_markdown_receipt, "markdown.jsonl.gz"),
        _binding(artifact_root, recovery_markdown_receipt, "markdown-supplement.jsonl.gz"))
    outlink_inputs = (
        _binding(artifact_root, base_outlinks_receipt, "outlinks.jsonl.gz"),
        _binding(artifact_root, recovery_outlinks_receipt, "outlinks-supplement.jsonl.gz"))

    partial = output.with_name(output.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    try:
        markdown_path = partial / "markdown/parents.jsonl.gz"
        outlinks_path = partial / "outlinks/parent-source-links.jsonl.gz"
        markdown_path.parent.mkdir(); outlinks_path.parent.mkdir()
        markdown_logical = hashlib.sha256(); outlinks_logical = hashlib.sha256()
        parents: dict[str, tuple[str, dict, dict, str]] = {}
        links = unresolved = empty = 0
        with deterministic_gzip_text(markdown_path) as sink:
            for path in markdown_inputs:
                with gzip.open(path, "rt", encoding="utf-8") as source:
                    for line in source:
                        row = json.loads(line); identifier = row.get("source_record_id")
                        markdown = row.get("markdown") or ""
                        archive = row.get("source_archive") or {}
                        if (not identifier or identifier in parents
                                or row.get("profile") != "anchored_law"
                                or not markdown or not archive.get("member")
                                or len(str(archive.get("sha256") or "")) != 64
                                or hashlib.sha256(markdown.encode()).hexdigest()
                                != row.get("markdown_sha256")):
                            raise ValueError("invalid/duplicate KALI parent")
                        parents[identifier] = (row["markdown_sha256"],
                                               row["source_version"], archive, row["url"])
                        _write(sink, markdown_logical, row)
        linked = set()
        with deterministic_gzip_text(outlinks_path) as sink:
            for path in outlink_inputs:
                with gzip.open(path, "rt", encoding="utf-8") as source:
                    for line in source:
                        row = json.loads(line); identifier = row.get("source_record_id")
                        expected_parent = parents.get(identifier)
                        if (identifier in linked or expected_parent is None
                                or (row.get("parent_markdown_sha256"), row.get("source_version"),
                                    row.get("parent_identity"), row.get("url")) != expected_parent
                                or not isinstance(row.get("links"), list)
                                or not isinstance(row.get("unresolved_relations"), list)):
                            raise ValueError("KALI outlink parent binding failure")
                        for link in row["links"]:
                            if not link.get("target") or not link.get("context"):
                                raise ValueError("KALI navigable link lacks target/context")
                        for relation in row["unresolved_relations"]:
                            reasons = relation.get("unresolved_reasons") or []
                            expected_reasons = [reason for missing, reason in (
                                (not relation.get("target"), "target_absent"),
                                (not relation.get("context"), "context_absent")) if missing]
                            if reasons != expected_reasons or not reasons:
                                raise ValueError("KALI unresolved relation classification drift")
                        linked.add(identifier); links += len(row["links"])
                        unresolved += len(row["unresolved_relations"])
                        empty += not row["links"]
                        _write(sink, outlinks_logical, row)
        if len(parents) != expected or linked != set(parents):
            raise ValueError("KALI exact parent/outlink conservation failed")
        unique_originals = sorted(set(original_paths))
        original_logical = hashlib.sha256("".join(
            f"{path.relative_to(artifact_root)}\0{sha256_file(path)}\n"
            for path in unique_originals).encode()).hexdigest()
        original_stage = SS.seal(
            artifact_root=artifact_root, artifact_paths=unique_originals,
            receipt_path=partial / "original/receipt.json",
            records=1 + int(increment_receipt.get("records") or 0),
            fmt="dila-kali-global-plus-ordered-increments+transport-evidence",
            logical_sha256=original_logical, producer_id=PRODUCER,
            producer_version=VERSION, publish_from=partial, publish_to=output,
            allow_existing_published=True)
        markdown_stage = SS.seal(
            artifact_root=artifact_root, artifact_paths=[markdown_path],
            receipt_path=partial / "markdown/receipt.json", records=expected,
            fmt="anchored-law-kali-markdown+canonical-jsonl+deterministic-gzip",
            logical_sha256=markdown_logical.hexdigest(), producer_id=PRODUCER,
            producer_version=VERSION, publish_from=partial, publish_to=output)
        outlinks_stage = SS.seal(
            artifact_root=artifact_root, artifact_paths=[outlinks_path],
            receipt_path=partial / "outlinks/receipt.json", records=expected,
            fmt="parent-bound-kali-legal-relations+deterministic-gzip",
            logical_sha256=outlinks_logical.hexdigest(), producer_id=PRODUCER,
            producer_version=VERSION, publish_from=partial, publish_to=output)
        result = {
            "schema": "aw.kali-exact-production-combined-run.v1", "status": "complete",
            "source_id": "dila-kali-exact-production-v2", "source_profile": "anchored_law",
            "counts": {"production_parents": expected, "source_links": links,
                       "unresolved_relations": unresolved,
                       "explicit_empty_outlink_parents": empty},
            "inputs": {
                "base_run": {"uri": f"artifact://{base_run_path.relative_to(artifact_root)}",
                             "bytes": base_run_path.stat().st_size,
                             "sha256": sha256_file(base_run_path)},
                "increment_recovery_run": {
                    "uri": f"artifact://{recovery_run_path.relative_to(artifact_root)}",
                    "bytes": recovery_run_path.stat().st_size,
                    "sha256": sha256_file(recovery_run_path)}},
            "stages": {"original": original_stage, "markdown": markdown_stage,
                       "outlinks": outlinks_stage},
            "next_gate": "anchored-law profile/fidelity audit and token-safe parent-local chunks",
        }
        if supersedes_run is not None:
            prior = supersedes_run.resolve(); prior.relative_to(artifact_root)
            result["supersedes"] = {
                "uri": f"artifact://{prior.relative_to(artifact_root)}",
                "bytes": prior.stat().st_size, "sha256": sha256_file(prior),
                "reason": "v1 combined run omitted exhaustive original L0 stage inventory"}
        (partial / "RUN.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.rename(partial, output)
        return result
    except BaseException:
        if partial.exists():
            failed = output.with_name(output.name + f".failed.{os.getpid()}")
            os.rename(partial, failed)
        raise
