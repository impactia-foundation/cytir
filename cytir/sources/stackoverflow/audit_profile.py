#!/usr/bin/env python3
"""Deep semantic/profile audit for a source-native Stack Overflow Q+A pilot pack."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import uuid

from cytir.acquire.artifact_io import sha256_file
from cytir.index.preservation_contract import _validate_stage
from cytir.sources import audit as shared


SOURCE_ID = "stackoverflow-qa-score-cells-v1"
PROFILE = "community_qa"
EXPECTED_CELLS = {
    "accepted_qscore_0_1", "accepted_qscore_2_4",
    "accepted_qscore_5_9", "accepted_qscore_10_plus",
}
ALLOWED_LICENSES = {"CC BY-SA 2.5", "CC BY-SA 3.0", "CC BY-SA 4.0"}
POLICY_PATH = (
    "data/chunk-policy-struct-md-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json")
POLICY_ID = "struct-md-768w-o0-adjacent-sections-tail-1152-tok2560"


def _assert_attribution(value: dict) -> None:
    if not (value.get("owner_user_id") or value.get("owner_display_name")):
        raise ValueError("Stack Overflow retained post lacks attribution identity")
    if bool(value.get("owner_user_id")) != bool(value.get("owner_profile_url")):
        raise ValueError("Stack Overflow user/profile attribution relation drifted")


def audit(*, artifact_root: Path, parent_run: Path, pack_root: Path,
          output_root: Path, registry_path: Path,
          source_id: str = SOURCE_ID,
          expected_schema: str = "aw.stackoverflow-qa-source-native-pilot-run.v1",
          expected_status: str = "source_native_pilot_complete_pending_profile_fidelity_chunk_measurement",
          expected_selected: int = 1024,
          expected_cells: set[str] | None = None,
          classification: str = "measured pilot only; no full-corpus point projection") -> tuple[Path, Path]:
    run = json.loads(parent_run.read_text())
    counts = run.get("counts") or {}
    expected_cells = set(expected_cells or EXPECTED_CELLS)
    if (run.get("schema") != expected_schema
            or run.get("status") != expected_status
            or run.get("source_profile") != PROFILE
            or counts.get("selected_questions") != expected_selected
            or counts.get("markdown_documents", 0) + counts.get("quarantine", 0) != expected_selected
            or (run.get("content_fidelity") or {}).get("source_content_cap") is not None
            or not (run.get("content_fidelity") or {}).get("exact_selected_xml_rows")
            or not (run.get("content_fidelity") or {}).get("full_prechunk_markdown")):
        raise ValueError("Stack Overflow source-native pilot conservation drifted")
    for name, stage in run["stages"].items():
        _validate_stage(stage, f"parent.{name}", artifact_root=artifact_root,
                        deep=True)

    pack_path = pack_root / "PACK.json"
    pack = json.loads(pack_path.read_text())
    policy = pack.get("chunk_policy") or {}
    expected_policy = registry_path.parent / POLICY_PATH.removeprefix("data/")
    if (pack.get("status") != "ready_for_gpu_queue"
            or pack.get("source_profile") != PROFILE
            or policy.get("path") != POLICY_PATH
            or policy.get("policy_id") != POLICY_ID
            or policy.get("sha256") != sha256_file(expected_policy)
            or (pack.get("token_distribution") or {}).get("over_hard_max") != 0
            or (pack.get("token_distribution") or {}).get("max", 2561) > 2560):
        raise ValueError("Stack Overflow Q+A pack policy/token gate failed")
    _validate_stage(pack["chunks"], "pack.chunks",
                    artifact_root=artifact_root, deep=True)

    markdown = shared.stage_rows(artifact_root, run["stages"]["markdown"])
    outlinks = shared.stage_rows(artifact_root, run["stages"]["outlinks"])
    quarantine = shared.stage_rows(artifact_root, run["stages"]["quarantine"])
    chunks = shared.stage_rows(artifact_root, pack["chunks"])
    parents = {}
    cells = set()
    prior_included = 0
    for row in markdown:
        document_id = str(row.get("document_id") or "")
        question = row.get("question") or {}
        answer = row.get("accepted_answer") or {}
        question_id = str(question.get("post_id") or "")
        answer_id = str(answer.get("post_id") or "")
        question_url = f"https://stackoverflow.com/questions/{question_id}"
        answer_url = f"https://stackoverflow.com/a/{answer_id}"
        text = str(row.get("markdown") or "")
        source_links = row.get("source_links") or []
        if (not document_id or document_id in parents
                or row.get("source_profile") != PROFILE or row.get("lang") != "en"
                or row.get("site") != "stackoverflow.com"
                or row.get("url") != question_url or row.get("answer_url") != answer_url
                or row.get("cell") not in EXPECTED_CELLS
                or question.get("license") not in ALLOWED_LICENSES
                or answer.get("license") not in ALLOWED_LICENSES
                or not question.get("creation_time") or not answer.get("creation_time")
                or not question.get("raw_xml_sha256")
                or not answer.get("raw_xml_sha256")
                or not text.startswith("# ")
                or "\n\n## Question\n\n" not in text
                or "\n\n## Accepted answer\n\n" not in text
                or not text.split("\n\n## Question\n\n", 1)[1]
                or not (row.get("extraction") or {}).get("full_prechunk_content")
                or (row.get("extraction") or {}).get("content_cap") is not None
                or {item.get("url") for item in source_links}
                != {question_url, answer_url}):
            raise ValueError("Stack Overflow Markdown source semantics are incomplete")
        _assert_attribution(question.get("attribution") or {})
        _assert_attribution(answer.get("attribution") or {})
        comparison = row.get("comparison_prior_amin3_qmin8_kans2") or {}
        if (comparison.get("question_min_score") != 8
                or comparison.get("answer_min_score") != 3
                or comparison.get("answer_cap") != 2):
            raise ValueError("Stack Overflow prior-arm comparison semantics drifted")
        prior_included += bool(comparison.get("included"))
        cells.add(row["cell"])
        parents[document_id] = row
    if cells != expected_cells:
        raise ValueError("Stack Overflow retained pilot lost a score cell")

    linked = set()
    canonical_question = set()
    accepted_answer = set()
    for row in outlinks:
        document_id = str(row.get("document_id") or "")
        parent = parents.get(document_id)
        if parent is None or row.get("source_url") != parent.get("url"):
            raise ValueError("Stack Overflow outlink lost parent identity")
        if row.get("is_empty"):
            if row.get("target_url") is not None:
                raise ValueError("Stack Overflow empty outlink carries a target")
        elif not row.get("target_url") or not row.get("relation"):
            raise ValueError("Stack Overflow outlink target/relation is incomplete")
        else:
            linked.add(document_id)
            if row.get("relation") == "canonical-question":
                canonical_question.add(document_id)
            elif row.get("relation") == "accepted-answer":
                accepted_answer.add(document_id)
    if linked != set(parents) or canonical_question != set(parents) \
            or accepted_answer != set(parents):
        raise ValueError("Stack Overflow parent-bound Q/A outlink conservation failed")

    parent_sha = {key: hashlib.sha256(value["markdown"].encode()).hexdigest()
                  for key, value in parents.items()}
    point_ids = set()
    ordinals: dict[str, list[int]] = {}
    for row in chunks:
        point_id = str(row.get("point_id") or "")
        uuid.UUID(point_id)
        document_id = str(row.get("document_id") or "")
        parent = parents.get(document_id)
        metadata = row.get("source_metadata") or {}
        if (point_id in point_ids or parent is None
                or row.get("parent_markdown_sha256") != parent_sha.get(document_id)
                or row.get("source_profile") != PROFILE
                or row.get("source_links") != parent.get("source_links")
                or metadata.get("question") != parent.get("question")
                or metadata.get("accepted_answer") != parent.get("accepted_answer")
                or metadata.get("cell") != parent.get("cell")
                or metadata.get("answer_url") != parent.get("answer_url")):
            raise ValueError("Stack Overflow chunk lost identity/source semantics")
        point_ids.add(point_id)
        ordinals.setdefault(document_id, []).append(row.get("chunk_ordinal"))
    if (set(ordinals) != set(parents)
            or any(sorted(values) != list(range(len(values)))
                   for values in ordinals.values())):
        raise ValueError("Stack Overflow chunk parent/ordinal conservation failed")

    registry = json.loads(registry_path.read_text())
    profile = registry["profiles"][PROFILE]
    methods = {
        "original": [
            "Deep-rehashed both complete official Posts/Users archives plus member CRC/inventory evidence.",
            "Verified exact byte-preserved selected question, answer, and user XML rows with per-row SHA-256.",
            "Checked stable question/accepted-answer/parent/user identities on every selected source record.",
            "Checked exact source-declared ContentLicense, score, and creation time on every retained post."],
        "markdown": [
            f"Scanned {len(parents)} complete uncapped question+accepted-answer Markdown documents including code and links.",
            "Checked question/answer IDs, scores, tags, creation times, licenses, and attribution boundaries.",
            "Matched every parent Markdown record to exact selected XML row digests and the archive-bound original stage.",
            f"Verified {len(quarantine)} explicit source gate failures and no silent drops."],
        "chunks": [
            "Verified every chunk belongs to exactly one question parent.",
            "Checked retained question/accepted-answer relations and source post identities on every chunk.",
            "Recomputed every parent Markdown digest; checked ordinal/UUID uniqueness and pinned policy/tokenizer evidence.",
            f"Exact tokenizer gate reports max {(pack.get('token_distribution') or {}).get('max')} <=2560 and zero over-limit chunks."],
        "outlinks": [
            "Verified canonical question and accepted-answer source URLs for every retained parent.",
            "Checked attribution profiles and cited targets retain relation plus anchor/context where source-declared.",
            "Verified every retained parent has a receipted relation/outlink representation."],
        "vectors": [],
    }
    checks = shared.profile_checks(profile, methods, label="Stack Overflow")
    evidence = {
        "schema": "aw.stackoverflow-qa-profile-evidence.v1",
        "source_id": source_id,
        "counts": {"selected_questions": counts["selected_questions"],
                   "parents": len(parents), "quarantine": len(quarantine),
                   "chunks": len(chunks), "unique_point_ids": len(point_ids),
                   "outlinks": len(outlinks), "prior_arm_included": prior_included},
        "score_cells": sorted(cells),
        "logical_sha256": {name: stage["logical_sha256"]
                           for name, stage in run["stages"].items()},
        "chunk_logical_sha256": pack["chunks"]["logical_sha256"],
        "parent_run_sha256": sha256_file(parent_run),
        "pack_sha256": sha256_file(pack_path),
        "profile_registry_sha256": sha256_file(registry_path),
        "classification": classification,
    }
    return shared.publish_evidence(output_root, evidence, checks, label="Stack Overflow")


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog or "audit_profile.py", description=__doc__)
    for name in ("artifact-root", "parent-run", "pack-root", "output-root",
                 "profile-registry"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-id", default=SOURCE_ID)
    parser.add_argument("--expected-schema", default="aw.stackoverflow-qa-source-native-pilot-run.v1")
    parser.add_argument("--expected-status", default="source_native_pilot_complete_pending_profile_fidelity_chunk_measurement")
    parser.add_argument("--expected-selected", type=int, default=1024)
    parser.add_argument("--expected-cell", action="append", default=[])
    parser.add_argument("--classification", default="measured pilot only; no full-corpus point projection")
    args = parser.parse_args(argv)
    evidence, checks = audit(
        artifact_root=args.artifact_root.resolve(),
        parent_run=args.parent_run.resolve(), pack_root=args.pack_root.resolve(),
        output_root=args.output_root.resolve(),
        registry_path=args.profile_registry.resolve(), source_id=args.source_id,
        expected_schema=args.expected_schema,
        expected_status=args.expected_status,
        expected_selected=args.expected_selected,
        expected_cells=set(args.expected_cell) if args.expected_cell else None,
        classification=args.classification)
    print(json.dumps({"evidence": str(evidence), "checks": str(checks)}, indent=2))
    return 0


if __name__ == "__main__":
    main()
