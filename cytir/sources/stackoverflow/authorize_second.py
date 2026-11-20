#!/usr/bin/env python3
"""Authorize one exact disjoint Stack Overflow production tranche.

This successor consumes the sealed full score>=10 frontier and the complete
first-tranche source/profile/fidelity/token/vector evidence.  It emits only
the explicitly authorized rank interval; it neither authorizes nor projects
the rest of the 551,326-pair frontier.
"""
from __future__ import annotations

import argparse
import gzip
from cytir.acquire.gzip_pinned import deterministic_gzip_open
import hashlib
import heapq
import json
import os
from pathlib import Path

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.index.preservation_contract import _validate_stage
import cytir.acquire.stage_receipt as SS


VERSION = "stackoverflow-qa-bounded-tranche-authorization.v1"
PRODUCER = "pipeline/authorize_stackoverflow_qa_second_tranche.py"


def _binding(path: Path, artifact_root: Path) -> dict:
    path = path.resolve()
    relative = path.relative_to(artifact_root.resolve()).as_posix()
    return {"uri": f"artifact://{relative}", "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def _stream_gzip(path: Path, rows) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    logical = hashlib.sha256()
    count = 0
    with partial.open("xb") as raw:
        with deterministic_gzip_open(raw, compresslevel=6) as target:
            for row in rows:
                body = (canonical_json(row) + "\n").encode()
                logical.update(body)
                target.write(body)
                count += 1
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(partial, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return count, logical.hexdigest()


def _smallest(frontier: Path, count: int) -> tuple[list[dict], int]:
    heap: list[tuple[int, int, dict]] = []
    records = 0
    with gzip.open(frontier, "rt", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("selection_key")
            if not isinstance(key, str) or len(key) != 64:
                raise ValueError("Stack Overflow frontier selection key is invalid")
            rank = int(key, 16)
            qid = int(row["question_id"])
            item = (-rank, qid, row)
            if len(heap) < count:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            records += 1
    rows = [row for _, _, row in sorted(heap, key=lambda item: -item[0])]
    rows.sort(key=lambda row: (row["selection_key"], int(row["question_id"])))
    return rows, records


def _load_keys(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line)["selection_key"] for line in source if line.strip()]


def _validate_first(*, parent: dict, pack: dict, profile: dict, fidelity: dict,
                    stage: dict, vector: dict) -> None:
    counts = parent.get("counts", {})
    if (parent.get("schema") != "aw.stackoverflow-qa-production-tranche-run.v1"
            or parent.get("status") != "source_native_tranche_complete_pending_profile_fidelity_chunk_measurement"
            or counts.get("selected_questions") != 10_000
            or counts.get("markdown_documents") != 10_000
            or counts.get("quarantine") != 0):
        raise ValueError("first Stack Overflow source tranche is not terminal")
    token = pack.get("token_distribution", {})
    chunks = pack.get("counts", {}).get("chunks")
    if (pack.get("status") != "ready_for_gpu_queue"
            or pack.get("counts", {}).get("documents") != 10_000
            or not isinstance(chunks, int) or chunks <= 0
            or token.get("hard_max") != 2560 or token.get("max", 2561) > 2560
            or token.get("over_hard_max") != 0):
        raise ValueError("first Stack Overflow chunk/token evidence is not terminal")
    checks = [entry for group in profile.get("checks", {}).values() for entry in group]
    if (profile.get("schema") != "aw.source-profile-compliance-receipt.v1"
            or profile.get("profile") != "community_qa" or not checks
            or any(entry.get("status") != "PASS" for entry in checks)):
        raise ValueError("first Stack Overflow source profile is not terminal")
    if (fidelity.get("schema") != "aw.content-fidelity-receipt.v1"
            or fidelity.get("original_bytes_complete") is not True
            or fidelity.get("markdown_content_complete") is not True
            or fidelity.get("source_text_truncation_applied") is not False):
        raise ValueError("first Stack Overflow fidelity evidence is not terminal")
    if (stage.get("schema") != "aw.stage-receipt.v1"
            or stage.get("status") != "complete" or stage.get("rows") != chunks):
        raise ValueError("first Stack Overflow portable stage is not terminal")
    storage = vector.get("storage", {})
    if (vector.get("schema") != "aw.vector-rollup.v1"
            or vector.get("status") != "complete" or vector.get("rows") != chunks
            or storage.get("dimensions") != 1024 or storage.get("dtype") != "float16"
            or storage.get("binary_transfer_ready") is not True
            or vector.get("verification", {}).get("failed") != 0):
        raise ValueError("first Stack Overflow vectors are not terminal")


def build(*, frontier_run: Path, frontier: Path, first_tranche: Path,
          first_parent: Path, first_pack: Path, first_profile: Path,
          first_fidelity: Path, first_stage: Path, first_vector: Path,
          root: Path, artifact_root: Path, offset: int, limit: int) -> dict:
    if root.exists():
        raise ValueError("Stack Overflow bounded authorization root must be fresh")
    if offset != 10_000 or limit != 10_000:
        raise ValueError("only the explicitly authorized second 10k tranche is allowed")
    frontier_seal = json.loads(frontier_run.read_text())
    if (frontier_seal.get("schema") != "aw.stackoverflow-qa-production-frontier-run.v1"
            or frontier_seal.get("status") != "exact_frontier_complete_first_tranche_authorized"):
        raise ValueError("terminal Stack Overflow full selection frontier required")
    _validate_stage(frontier_seal["stages"]["frontier"],
                    "stackoverflow.production_frontier",
                    artifact_root=artifact_root, deep=True)
    evidence_paths = [first_parent, first_pack, first_profile, first_fidelity,
                      first_stage, first_vector]
    parent, pack, profile, fidelity, stage, vector = [
        json.loads(path.read_text()) for path in evidence_paths]
    _validate_first(parent=parent, pack=pack, profile=profile,
                    fidelity=fidelity, stage=stage, vector=vector)

    ranked, full_records = _smallest(frontier, offset + limit)
    if full_records != frontier_seal["stages"]["frontier"]["records"]:
        raise ValueError("Stack Overflow full frontier count drift")
    first_keys = _load_keys(first_tranche)
    if first_keys != [row["selection_key"] for row in ranked[:offset]]:
        raise ValueError("first Stack Overflow tranche no longer matches rank prefix")
    selected = ranked[offset:offset + limit]
    if len(selected) != limit or set(first_keys) & {row["selection_key"] for row in selected}:
        raise ValueError("second Stack Overflow tranche is incomplete or overlaps first")
    user_ids = {int(value) for row in selected for value in
                (row.get("question_owner_user_id"), row.get("answer_owner_user_id"))
                if value is not None}
    expected_source_rows = 2 * limit + len(user_ids)

    root.mkdir(parents=True)
    tranche = root / "frontier/second-010000-019999.jsonl.gz"
    records, logical = _stream_gzip(tranche, selected)
    evidence = {path.stem: _binding(path, artifact_root) for path in evidence_paths}
    decision = {
        "schema": "aw.stackoverflow-qa-bounded-tranche-decision.v1",
        "version": VERSION,
        "status": "exact_second_tranche_authorized_only",
        "classification": "bounded measured continuation; no full-frontier document, chunk, vector, cost, or point extrapolation",
        "selector": {"ordering": "ascending selection_key then question_id",
                     "offset": offset, "limit": limit,
                     "full_frontier_records": full_records,
                     "disjoint_from_first_tranche": True,
                     "selection_logical_sha256": logical},
        "counts": {"questions": records, "unique_user_rows": len(user_ids),
                   "expected_source_rows_question_answer_user": expected_source_rows},
        "first_tranche_terminal_evidence": evidence,
        "next": "materialize only rank interval 10000:20000 with complete official XML rows, attribution, licenses, uncapped Markdown, and profile/fidelity/token/UUID gates",
    }
    decision_path = root / "DECISION.json"
    atomic_publish(decision_path,
                   (json.dumps(decision, indent=2, sort_keys=True) + "\n").encode())
    artifacts = [decision_path, tranche, frontier_run, frontier, first_tranche,
                 *evidence_paths]
    sealed = SS.seal(
        artifact_root=artifact_root, artifact_paths=artifacts,
        receipt_path=root / "stages/frontier/receipt.json", records=records,
        fmt="exact-disjoint-source-native-Q+accepted-A+user-selection-tranche",
        logical_sha256=hashlib.sha256((logical + sha256_file(decision_path)).encode()).hexdigest(),
        producer_id=PRODUCER, producer_version=VERSION)
    run = {
        "schema": "aw.stackoverflow-qa-production-frontier-run.v1",
        "version": VERSION, "status": "exact_second_tranche_authorized_only",
        "decision": decision, "stages": {"frontier": sealed},
    }
    atomic_publish(root / "RUN.json",
                   (json.dumps(run, indent=2, sort_keys=True) + "\n").encode())
    return run


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("frontier-run", "frontier", "first-tranche", "first-parent",
                 "first-pack", "first-profile", "first-fidelity", "first-stage",
                 "first-vector", "root", "artifact-root"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--offset", type=int, default=10_000)
    p.add_argument("--limit", type=int, default=10_000)
    a = p.parse_args()
    print(json.dumps(build(
        frontier_run=a.frontier_run.resolve(), frontier=a.frontier.resolve(),
        first_tranche=a.first_tranche.resolve(), first_parent=a.first_parent.resolve(),
        first_pack=a.first_pack.resolve(), first_profile=a.first_profile.resolve(),
        first_fidelity=a.first_fidelity.resolve(), first_stage=a.first_stage.resolve(),
        first_vector=a.first_vector.resolve(), root=a.root.resolve(),
        artifact_root=a.artifact_root.resolve(), offset=a.offset, limit=a.limit),
        indent=2, sort_keys=True))
