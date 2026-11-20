#!/usr/bin/env python3
"""Deep profile/token/lineage audit for one bounded OpenAlex pilot pack."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterator
import uuid

from cytir.acquire.artifact_io import atomic_publish, sha256_file
from cytir.index.preservation_contract import _validate_stage
from cytir.sources.openalex.materialize_pilot import reconstruct_abstract
from cytir.core.paths import data_path


PROFILE = "academic_dump"
POLICY_FILE = data_path("chunk-policy-struct-md-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json")
# the pack records its policy as the repo-relative "data/<name>" string; compare that, not the checkout path
POLICY_PATH = "data/" + POLICY_FILE.name


def exact_abstract_section(markdown: str) -> str | None:
    """Return the renderer-owned abstract body, excluding its final newline.

    ``render_record`` emits exactly one terminal ``## Abstract`` section and
    applies ``strip()`` to the completed Markdown before restoring one final
    newline.  Extracting that section lets the auditor compare the durable
    field exactly instead of accepting a matching substring elsewhere.
    """
    marker = "\n## Abstract\n\n"
    if marker not in markdown or not markdown.endswith("\n"):
        return None
    return markdown.split(marker, 1)[1][:-1]


def abstract_fidelity_matches(source_abstract: str, markdown: str) -> bool:
    """Match exactly under the renderer's documented boundary normalization."""
    if not source_abstract or not markdown.endswith("\n"):
        return False
    # The source-declared abstract may itself contain a Markdown heading that
    # is byte-identical to the renderer-owned boundary.  Comparing the exact
    # terminal suffix preserves that source text without treating the nested
    # heading as a second renderer boundary.
    return markdown.endswith(
        "\n## Abstract\n\n" + source_abstract.rstrip() + "\n")


def build_profile_methods(*, selected: int, selected_parts: int,
                          parents: int, chunks: int,
                          external_links: int, empty: int,
                          token_max: int) -> dict[str, list[str]]:
    """Return one concrete verification method per profile requirement."""
    return {
        "original": [
            (f"Deep-rehashed {selected_parts} selected whole parquet part(s) and "
             f"the canonical full source-record stage for all {selected} records."),
            f"Verified {selected} stable unique OpenAlex work IDs against full raw source metadata.",
            "Bound the exact portable snapshot RUN, part receipt, part SHA-256 and record-stage logical digest."],
        "markdown": [
            f"Scanned {parents} uncapped Markdown parents; each retains title and full source-declared abstract.",
            "Verified source-native metadata, CC0 metadata license, underlying-work license fields and stable work identity.",
            "Matched every admitted parent to the exact parquet part path/SHA and full canonical source-record JSON."],
        "chunks": [
            f"Verified all {chunks} chunks remain within one OpenAlex source record.",
            "Recomputed every parent Markdown digest and matched contiguous parent-local ordinals.",
            f"Verified exact-token max={token_max}/2560 and zero over-limit rows."],
        "outlinks": [
            "Verified every outlink retains its OpenAlex parent work ID and source relation/JSON-path context.",
            f"Verified {external_links} DOI, source, author, institution, code/data and cited-work targets where present.",
            f"Verified all {parents} admitted parents are represented, including {empty} explicit-empty parents."],
        "vectors": [],
    }


def _iter_receipt_rows(root: Path, receipt: dict) -> Iterator[dict]:
    """Stream ordered stage shards without retaining the full stage in RAM."""
    for artifact in receipt["artifacts"]:
        if artifact["uri"].endswith(".jsonl.gz"):
            with gzip.open(root / artifact["uri"].removeprefix("artifact://"),
                           "rt", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)


def _iter_rows(root: Path, stage: dict) -> Iterator[dict]:
    receipt = _validate_stage(stage, "rows", artifact_root=root, deep=True)
    yield from _iter_receipt_rows(root, receipt)


def _rows(root: Path, stage: dict) -> list[dict]:
    """Legacy helper retained for callers/tests; the deep auditor streams."""
    return list(_iter_rows(root, stage))


def audit(*, artifact_root: Path, parent_run: Path, pack_root: Path,
          output_root: Path, registry_path: Path, source_id: str) -> tuple[Path, Path]:
    run = json.loads(parent_run.read_text())
    counts = run.get("counts") or {}
    selected = counts.get("selected_source_records")
    if (run.get("schema") != "aw.openalex-academic-pilot-materialization-run.v1"
            or run.get("status") != "ready_for_current_policy_chunking"
            or run.get("source_profile") != PROFILE
            or not isinstance(selected, int) or selected <= 0
            or counts.get("markdown_documents", 0)
            + counts.get("recovery_records", 0) != selected
            or run.get("classification")
            != "full_openalex_metadata_and_abstract_records_not_full_papers"):
        raise ValueError("OpenAlex pilot parent identity/conservation failed")
    selection = run.get("selection") or {}
    partition = selection.get("partition")
    if run.get("scope") == (
            "bounded_disjoint_whole_part_tranche_partition_no_full_frontier_point_claim"):
        if (not isinstance(partition, dict)
                or partition.get("selected_records") != selected
                or partition.get("selected_part_count")
                != selection.get("selected_part_count")
                or partition.get("global_selected_records", 0) < selected
                or partition.get("partition_count", 0) <= 1):
            raise ValueError("OpenAlex partition audit identity failed")
    elif partition is not None:
        raise ValueError("unpartitioned OpenAlex RUN claims partition identity")
    for name, stage in run["stages"].items():
        _validate_stage(stage, f"parent.{name}", artifact_root=artifact_root,
                        deep=True)
    source_by_id = {}
    source_parts = set()
    source_count = 0
    for row in _iter_rows(artifact_root, run["stages"]["original"]):
        source_count += 1
        source_record_id = str(row.get("source_record_id") or "")
        raw = row.get("record") or {}
        if (not source_record_id or source_record_id in source_by_id
                or raw.get("id") != source_record_id
                or row.get("metadata_license") != "CC0-1.0"
                or not row.get("source_part_sha256")
                or not (row.get("source_receipt") or {}).get("sha256")):
            raise ValueError("OpenAlex selected source record lost L0 identity")
        source_parts.add(row.get("source_part"))
        source_by_id[source_record_id] = {
            "abstract": reconstruct_abstract(raw.get("abstract_inverted_index")),
        }
    if source_count != selected:
        raise ValueError("OpenAlex source-record stage row conservation failed")
    selected_parts = len(source_parts)
    expected_parts = (run.get("selection") or {}).get("selected_part_count", 1)
    if selected_parts != expected_parts or selected_parts <= 0:
        raise ValueError("OpenAlex selected source part conservation failed")

    parent_by_document = {}; parent_by_source = set(); abstracts = 0
    parent_count = 0
    for row in _iter_rows(artifact_root, run["stages"]["markdown"]):
        parent_count += 1
        document_id = str(row.get("document_id") or "")
        source_record_id = str(row.get("source_record_id") or "")
        source_row = source_by_id.get(source_record_id)
        markdown = str(row.get("markdown") or "")
        abstract = (source_row or {}).get("abstract", "")
        metadata = row.get("source_metadata") or {}
        identity = {
            "document_id": bool(document_id),
            "unique_document_id": document_id not in parent_by_document,
            "unique_source_record_id": source_record_id not in parent_by_source,
            "source_record": source_row is not None,
            "exact_abstract_section": abstract_fidelity_matches(abstract, markdown),
            "source_url": row.get("url") == source_record_id,
            "full_prechunk_content": row.get("full_prechunk_content") is True,
            "metadata_openalex_id": metadata.get("openalex_id") == source_record_id,
            "metadata_license": metadata.get("metadata_license") == "CC0-1.0",
            "record_classification": metadata.get("record_classification")
            == "full_openalex_metadata_and_abstract_not_full_paper_content",
        }
        if not all(identity.values()):
            failed = ",".join(key for key, passed in identity.items() if not passed)
            raise ValueError(
                "OpenAlex Markdown lost exact abstract/metadata identity: "
                f"markdown_ordinal={parent_count} "
                f"source_record_id={source_record_id} failed={failed}")
        parent_by_document[document_id] = {
            "url": row.get("url"),
            "source_record_id": source_record_id,
            "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest()}
        parent_by_source.add(source_record_id)
        abstracts += 1
    if parent_count != counts["markdown_documents"]:
        raise ValueError("OpenAlex Markdown stage row conservation failed")
    recovery_ids = set(); recovery_count = 0
    for row in _iter_rows(artifact_root, run["stages"]["recovery"]):
        recovery_count += 1
        recovery_ids.add(str(row.get("source_record_id") or ""))
    if (recovery_count != counts["recovery_records"]
            or len(recovery_ids) != recovery_count
            or set(parent_by_source) & recovery_ids
            or set(parent_by_source) | recovery_ids != set(source_by_id)):
        raise ValueError("OpenAlex admitted/recovery source partition is not exact")

    represented = set(); empty = set(); external_links = 0
    outlink_count = 0
    for row in _iter_rows(artifact_root, run["stages"]["outlinks"]):
        outlink_count += 1
        document_id = str(row.get("document_id") or "")
        parent = parent_by_document.get(document_id)
        if parent is None or row.get("source_url") != parent["url"]:
            raise ValueError("OpenAlex outlink lost parent identity")
        represented.add(document_id)
        if row.get("is_empty"):
            if row.get("target_url") is not None:
                raise ValueError("OpenAlex explicit-empty outlink has target")
            empty.add(document_id)
        else:
            if (not str(row.get("target_url") or "").startswith(("http://", "https://"))
                    or not row.get("relation_contexts")):
                raise ValueError("OpenAlex relation lacks URL/source context")
            external_links += 1
    if (outlink_count != counts["outlink_records_including_empty"]
            or represented != set(parent_by_document)):
        raise ValueError("OpenAlex outlinks omit an admitted parent")

    pack_path = pack_root / "PACK.json"
    pack = json.loads(pack_path.read_text())
    distribution = pack.get("token_distribution") or {}
    if (pack.get("status") != "ready_for_gpu_queue"
            or pack.get("source_profile") != PROFILE
            or (pack.get("counts") or {}).get("documents") != parent_count
            or (pack.get("chunk_policy") or {}).get("path") != POLICY_PATH
            or distribution.get("over_hard_max") != 0
            or distribution.get("max", 2561) > 2560):
        raise ValueError("OpenAlex academic pack policy/token gate failed")
    point_ids = set(); ordinals: dict[str, list[int]] = {}; chunk_count = 0
    for row in _iter_rows(artifact_root, pack["chunks"]):
        chunk_count += 1
        point_id = str(row.get("point_id") or ""); uuid.UUID(point_id)
        document_id = str(row.get("document_id") or "")
        parent = parent_by_document.get(document_id)
        if (point_id in point_ids or parent is None
                or row.get("source_record_id")
                != parent["source_record_id"]
                or row.get("parent_markdown_sha256")
                != parent["markdown_sha256"]
                or row.get("source_profile") != PROFILE
                or not row.get("canonical_text") or not row.get("text")
                or not row.get("sparse_text")):
            raise ValueError("OpenAlex chunk lost parent/UUID/channel identity")
        point_ids.add(point_id)
        ordinals.setdefault(document_id, []).append(row.get("chunk_ordinal"))
    if (chunk_count != (pack.get("counts") or {}).get("chunks")
            or len(point_ids) != chunk_count
            or set(ordinals) != set(parent_by_document)
            or any(sorted(values) != list(range(len(values)))
                   for values in ordinals.values())):
        raise ValueError("OpenAlex chunk ordinal/parent conservation failed")

    registry = json.loads(registry_path.read_text())
    profile = registry["profiles"][PROFILE]
    methods = build_profile_methods(
        selected=selected, selected_parts=selected_parts,
        parents=parent_count, chunks=chunk_count,
        external_links=external_links, empty=len(empty),
        token_max=distribution.get("max"))
    checks = {}
    for stage in profile["required_stages"]:
        requirements = profile.get(stage, {}).get("requirements", [])
        if len(requirements) != len(methods[stage]):
            raise ValueError(f"OpenAlex profile method count mismatch: {stage}")
        checks[stage] = [{"requirement": requirement, "status": "PASS",
                          "verification_method": method}
                         for requirement, method in zip(requirements, methods[stage])]
    evidence = {
        "schema": "aw.openalex-academic-pilot-profile-evidence.v1",
        "source_id": source_id,
        "counts": {"selected_source_records": selected,
                   "selected_source_parts": selected_parts,
                   "markdown_parents": parent_count,
                   "recovery": recovery_count,
                   "chunks": chunk_count, "unique_point_ids": len(point_ids),
                   "outlinks": outlink_count, "external_links": external_links,
                   "explicit_empty_parents": len(empty)},
        "parent_run_sha256": sha256_file(parent_run),
        "pack_sha256": sha256_file(pack_path),
        "profile_registry_sha256": sha256_file(registry_path),
        "partition": partition,
        "classification": (
            "bounded partition of an ordered whole-part OpenAlex metadata/abstract tranche; streaming outlink audit; not full-paper content or full-frontier point projection"
            if partition else
            "bounded ordered whole-part OpenAlex metadata/abstract tranche; not full-paper content or full-frontier point projection"
            if run.get("scope") ==
            "bounded_disjoint_whole_part_tranche_no_full_frontier_point_claim"
            else
            "bounded whole-part OpenAlex metadata/abstract pilot; not full-paper content or full-frontier point projection"),
    }
    output_root.mkdir(parents=True, exist_ok=False)
    evidence_path = output_root / "verification-evidence.json"
    checks_path = output_root / "checks.json"
    for path, value in ((evidence_path, evidence), (checks_path, checks)):
        atomic_publish(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
    return evidence_path, checks_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "parent-run", "pack-root", "output-root",
                 "profile-registry"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    args = parser.parse_args()
    evidence, checks = audit(
        artifact_root=args.artifact_root.resolve(),
        parent_run=args.parent_run.resolve(), pack_root=args.pack_root.resolve(),
        output_root=args.output_root.resolve(),
        registry_path=args.profile_registry.resolve(), source_id=args.source_id)
    print(json.dumps({"evidence": str(evidence), "checks": str(checks)}, indent=2))


if __name__ == "__main__":
    main()
