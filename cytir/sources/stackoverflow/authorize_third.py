#!/usr/bin/env python3
"""Authorize one exact disjoint Stack Overflow production tranche (the third).

This successor consumes the sealed full score>=10 frontier and the complete
first- and second-tranche source/profile/fidelity/token/vector evidence.  It
emits only the explicitly authorized rank interval [20000, 30000); it neither
authorizes nor projects the rest of the 551,326-pair frontier.

Structurally this is `authorize_stackoverflow_qa_second_tranche.py` advanced
by one 10k window: the frontier is immutable and deterministically ranked, so
the only things that change per tranche are (a) the offset, (b) the set of
prior tranches whose terminal evidence must be re-validated before the next
slice is cut, and (c) the prior tranches' own selection keys, checked as an
exact disjoint prefix.  A fresh `..._fourth_tranche.py` (offset 30000) is the
same shape again with three prior tranches to validate.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from cytir.acquire.artifact_io import atomic_publish, sha256_file
from cytir.sources.stackoverflow.authorize_second import _smallest, _stream_gzip, _binding
from cytir.index.preservation_contract import _validate_stage
import cytir.acquire.stage_receipt as SS


VERSION = "stackoverflow-qa-bounded-tranche-authorization.v1"
PRODUCER = "pipeline/authorize_stackoverflow_qa_third_tranche.py"
OFFSET = 20_000
LIMIT = 10_000


def _validate_terminal_tranche(*, parent: dict, pack: dict, profile: dict,
                                fidelity: dict, stage: dict, vector: dict | None,
                                label: str) -> None:
    """Re-validate one prior tranche's full terminal evidence chain.

    Identical gate to `authorize_stackoverflow_qa_second_tranche._validate_first`
    -- the schema/status/count shape a terminal tranche must satisfy does not
    depend on which ordinal tranche produced it, so this is deliberately not
    forked per-tranche.  `label` only decorates the raised error so a failure
    names which predecessor broke.

    `vector=None` (task 691) means "no vector rollup receipt exists yet": the
    embed is the unattended `growth_pending_compose` timer's job and it seals raw
    vector shards, not a rollup. The chunk/stage/profile/fidelity chain is still
    fully required; only the embed stage is skipped, and the caller records that.
    """
    counts = parent.get("counts", {})
    if (parent.get("schema") != "aw.stackoverflow-qa-production-tranche-run.v1"
            or parent.get("status") != "source_native_tranche_complete_pending_profile_fidelity_chunk_measurement"
            or counts.get("selected_questions") != 10_000
            or not isinstance(counts.get("markdown_documents"), int)
            or not isinstance(counts.get("quarantine"), int)
            or counts["markdown_documents"] <= 0 or counts["quarantine"] < 0
            # Every selected question is accounted for: admitted or quarantined with a
            # reason (task 691: ordinal 21 quarantined 1 question with no answer content).
            or counts["markdown_documents"] + counts["quarantine"] != 10_000):
        raise ValueError(f"{label} Stack Overflow source tranche is not terminal")
    token = pack.get("token_distribution", {})
    chunks = pack.get("counts", {}).get("chunks")
    if (pack.get("status") != "ready_for_gpu_queue"
            or pack.get("counts", {}).get("documents") != counts["markdown_documents"]
            or not isinstance(chunks, int) or chunks <= 0
            or token.get("hard_max") != 2560 or token.get("max", 2561) > 2560
            or token.get("over_hard_max") != 0):
        raise ValueError(f"{label} Stack Overflow chunk/token evidence is not terminal")
    checks = [entry for group in profile.get("checks", {}).values() for entry in group]
    if (profile.get("schema") != "aw.source-profile-compliance-receipt.v1"
            or profile.get("profile") != "community_qa" or not checks
            or any(entry.get("status") != "PASS" for entry in checks)):
        raise ValueError(f"{label} Stack Overflow source profile is not terminal")
    if (fidelity.get("schema") != "aw.content-fidelity-receipt.v1"
            or fidelity.get("original_bytes_complete") is not True
            or fidelity.get("markdown_content_complete") is not True
            or fidelity.get("source_text_truncation_applied") is not False):
        raise ValueError(f"{label} Stack Overflow fidelity evidence is not terminal")
    if (stage.get("schema") != "aw.stage-receipt.v1"
            or stage.get("status") != "complete" or stage.get("rows") != chunks):
        raise ValueError(f"{label} Stack Overflow portable stage is not terminal")
    if vector is None:
        return
    storage = vector.get("storage", {})
    if (vector.get("schema") != "aw.vector-rollup.v1"
            or vector.get("status") != "complete" or vector.get("rows") != chunks
            or storage.get("dimensions") != 1024 or storage.get("dtype") != "float16"
            or storage.get("binary_transfer_ready") is not True
            or vector.get("verification", {}).get("failed") != 0):
        raise ValueError(f"{label} Stack Overflow vectors are not terminal")


def _load_keys(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line)["selection_key"] for line in source if line.strip()]


def build(*, frontier_run: Path, frontier: Path,
          first_tranche: Path, first_parent: Path, first_pack: Path,
          first_profile: Path, first_fidelity: Path, first_stage: Path,
          first_vector: Path,
          second_tranche: Path, second_parent: Path, second_pack: Path,
          second_profile: Path, second_fidelity: Path, second_stage: Path,
          second_vector: Path,
          root: Path, artifact_root: Path, offset: int, limit: int) -> dict:
    if root.exists():
        raise ValueError("Stack Overflow bounded authorization root must be fresh")
    if offset != OFFSET or limit != LIMIT:
        raise ValueError("only the explicitly authorized third 10k tranche is allowed")
    frontier_seal = json.loads(frontier_run.read_text())
    if (frontier_seal.get("schema") != "aw.stackoverflow-qa-production-frontier-run.v1"
            or frontier_seal.get("status") != "exact_frontier_complete_first_tranche_authorized"):
        raise ValueError("terminal Stack Overflow full selection frontier required")
    _validate_stage(frontier_seal["stages"]["frontier"],
                    "stackoverflow.production_frontier",
                    artifact_root=artifact_root, deep=True)

    first_paths = [first_parent, first_pack, first_profile, first_fidelity,
                   first_stage, first_vector]
    first_parent_j, first_pack_j, first_profile_j, first_fidelity_j, \
        first_stage_j, first_vector_j = [json.loads(p.read_text()) for p in first_paths]
    _validate_terminal_tranche(parent=first_parent_j, pack=first_pack_j,
                               profile=first_profile_j, fidelity=first_fidelity_j,
                               stage=first_stage_j, vector=first_vector_j,
                               label="first")
    second_paths = [second_parent, second_pack, second_profile, second_fidelity,
                    second_stage, second_vector]
    second_parent_j, second_pack_j, second_profile_j, second_fidelity_j, \
        second_stage_j, second_vector_j = [json.loads(p.read_text()) for p in second_paths]
    _validate_terminal_tranche(parent=second_parent_j, pack=second_pack_j,
                               profile=second_profile_j, fidelity=second_fidelity_j,
                               stage=second_stage_j, vector=second_vector_j,
                               label="second")

    ranked, full_records = _smallest(frontier, offset + limit)
    if full_records != frontier_seal["stages"]["frontier"]["records"]:
        raise ValueError("Stack Overflow full frontier count drift")
    first_keys = _load_keys(first_tranche)
    second_keys = _load_keys(second_tranche)
    prior_keys = first_keys + second_keys
    if prior_keys != [row["selection_key"] for row in ranked[:offset]]:
        raise ValueError("prior Stack Overflow tranches no longer match rank prefix")
    selected = ranked[offset:offset + limit]
    if len(selected) != limit or (set(prior_keys) & {row["selection_key"] for row in selected}):
        raise ValueError("third Stack Overflow tranche is incomplete or overlaps a prior tranche")
    user_ids = {int(value) for row in selected for value in
                (row.get("question_owner_user_id"), row.get("answer_owner_user_id"))
                if value is not None}
    expected_source_rows = 2 * limit + len(user_ids)

    root.mkdir(parents=True)
    tranche = root / "frontier/third-020000-029999.jsonl.gz"
    records, logical = _stream_gzip(tranche, selected)
    evidence = {
        "first_parent": _binding(first_parent, artifact_root),
        "first_pack": _binding(first_pack, artifact_root),
        "first_profile": _binding(first_profile, artifact_root),
        "first_fidelity": _binding(first_fidelity, artifact_root),
        "first_stage": _binding(first_stage, artifact_root),
        "first_vector": _binding(first_vector, artifact_root),
        "second_parent": _binding(second_parent, artifact_root),
        "second_pack": _binding(second_pack, artifact_root),
        "second_profile": _binding(second_profile, artifact_root),
        "second_fidelity": _binding(second_fidelity, artifact_root),
        "second_stage": _binding(second_stage, artifact_root),
        "second_vector": _binding(second_vector, artifact_root),
    }
    decision = {
        "schema": "aw.stackoverflow-qa-bounded-tranche-decision.v1",
        "version": VERSION,
        "status": "exact_third_tranche_authorized_only",
        "classification": "bounded measured continuation; no full-frontier document, chunk, vector, cost, or point extrapolation; Track-2 expansion lane, not the Sep-8 flip",
        "selector": {"ordering": "ascending selection_key then question_id",
                     "offset": offset, "limit": limit,
                     "full_frontier_records": full_records,
                     "disjoint_from_prior_tranches": True,
                     "selection_logical_sha256": logical},
        "counts": {"questions": records, "unique_user_rows": len(user_ids),
                   "expected_source_rows_question_answer_user": expected_source_rows},
        "prior_tranches_terminal_evidence": evidence,
        "next": "materialize only rank interval 20000:30000 with complete official XML rows, attribution, licenses, uncapped Markdown, and profile/fidelity/token/UUID gates",
    }
    decision_path = root / "DECISION.json"
    atomic_publish(decision_path,
                   (json.dumps(decision, indent=2, sort_keys=True) + "\n").encode())
    artifacts = [decision_path, tranche, frontier_run, frontier,
                 first_tranche, second_tranche, *first_paths, *second_paths]
    sealed = SS.seal(
        artifact_root=artifact_root, artifact_paths=artifacts,
        receipt_path=root / "stages/frontier/receipt.json", records=records,
        fmt="exact-disjoint-source-native-Q+accepted-A+user-selection-tranche",
        logical_sha256=hashlib.sha256((logical + sha256_file(decision_path)).encode()).hexdigest(),
        producer_id=PRODUCER, producer_version=VERSION)
    run = {
        "schema": "aw.stackoverflow-qa-production-frontier-run.v1",
        "version": VERSION, "status": "exact_third_tranche_authorized_only",
        "decision": decision, "stages": {"frontier": sealed},
    }
    atomic_publish(root / "RUN.json",
                   (json.dumps(run, indent=2, sort_keys=True) + "\n").encode())
    return run


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("frontier-run", "frontier",
                 "first-tranche", "first-parent", "first-pack", "first-profile",
                 "first-fidelity", "first-stage", "first-vector",
                 "second-tranche", "second-parent", "second-pack", "second-profile",
                 "second-fidelity", "second-stage", "second-vector",
                 "root", "artifact-root"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--offset", type=int, default=OFFSET)
    p.add_argument("--limit", type=int, default=LIMIT)
    a = p.parse_args()
    print(json.dumps(build(
        frontier_run=a.frontier_run.resolve(), frontier=a.frontier.resolve(),
        first_tranche=a.first_tranche.resolve(), first_parent=a.first_parent.resolve(),
        first_pack=a.first_pack.resolve(), first_profile=a.first_profile.resolve(),
        first_fidelity=a.first_fidelity.resolve(), first_stage=a.first_stage.resolve(),
        first_vector=a.first_vector.resolve(),
        second_tranche=a.second_tranche.resolve(), second_parent=a.second_parent.resolve(),
        second_pack=a.second_pack.resolve(), second_profile=a.second_profile.resolve(),
        second_fidelity=a.second_fidelity.resolve(), second_stage=a.second_stage.resolve(),
        second_vector=a.second_vector.resolve(),
        root=a.root.resolve(), artifact_root=a.artifact_root.resolve(),
        offset=a.offset, limit=a.limit),
        indent=2, sort_keys=True))
