import gzip
import hashlib
import json
from pathlib import Path

import pytest

from cytir.acquire.artifact_io import canonical_json, sha256_file
from cytir.sources.stackoverflow.authorize_second import build, _smallest, _validate_first
from cytir.acquire import stage_receipt as SS


def _rows(path: Path, count: int) -> list[dict]:
    rows = []
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as target:
        for qid in range(count):
            key = hashlib.sha256(f"row-{qid}".encode()).hexdigest()
            row = {"question_id": qid, "accepted_answer_id": 1000 + qid,
                   "question_owner_user_id": qid % 3,
                   "answer_owner_user_id": qid % 5, "selection_key": key}
            target.write((canonical_json(row) + "\n").encode())
            rows.append(row)
    return rows


def test_ranked_second_slice_is_exact_and_disjoint(tmp_path: Path):
    frontier = tmp_path / "frontier.jsonl.gz"
    source = _rows(frontier, 40)
    ranked, records = _smallest(frontier, 20)
    expected = sorted(source, key=lambda row: (row["selection_key"], row["question_id"]))[:20]
    assert records == 40
    assert ranked == expected
    assert not ({row["selection_key"] for row in ranked[:10]}
                & {row["selection_key"] for row in ranked[10:20]})


def test_first_tranche_gate_requires_complete_fidelity_and_exact_tokens():
    parent = {"schema": "aw.stackoverflow-qa-production-tranche-run.v1",
              "status": "source_native_tranche_complete_pending_profile_fidelity_chunk_measurement",
              "counts": {"selected_questions": 10000, "markdown_documents": 10000,
                         "quarantine": 0}}
    pack = {"status": "ready_for_gpu_queue",
            "counts": {"documents": 10000, "chunks": 13112},
            "token_distribution": {"hard_max": 2560, "max": 2560,
                                   "over_hard_max": 0}}
    profile = {"schema": "aw.source-profile-compliance-receipt.v1",
               "profile": "community_qa",
               "checks": {"markdown": [{"status": "PASS"}]}}
    fidelity = {"schema": "aw.content-fidelity-receipt.v1",
                "original_bytes_complete": True, "markdown_content_complete": True,
                "source_text_truncation_applied": False}
    stage = {"schema": "aw.stage-receipt.v1", "status": "complete", "rows": 13112}
    vector = {"schema": "aw.vector-rollup.v1", "status": "complete", "rows": 13112,
              "storage": {"dimensions": 1024, "dtype": "float16",
                          "binary_transfer_ready": True},
              "verification": {"failed": 0}}
    _validate_first(parent=parent, pack=pack, profile=profile, fidelity=fidelity,
                    stage=stage, vector=vector)
    fidelity["source_text_truncation_applied"] = True
    with pytest.raises(ValueError, match="fidelity"):
        _validate_first(parent=parent, pack=pack, profile=profile, fidelity=fidelity,
                        stage=stage, vector=vector)


def test_build_seals_exact_second_10k_and_prior_terminal_evidence(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    source_root = artifacts / "source"
    source_root.mkdir(parents=True)
    frontier = source_root / "eligible.jsonl.gz"
    _rows(frontier, 20_001)
    ranked, _ = _smallest(frontier, 20_000)
    first = source_root / "first.jsonl.gz"
    with gzip.GzipFile(filename=str(first), mode="wb", mtime=0) as target:
        for row in ranked[:10_000]:
            target.write((canonical_json(row) + "\n").encode())
    stage = SS.seal(
        artifact_root=artifacts, artifact_paths=[frontier, first],
        receipt_path=source_root / "receipt.json", records=20_001,
        fmt="fixture", logical_sha256=sha256_file(frontier),
        producer_id="fixture", producer_version="v1")
    frontier_run = source_root / "RUN.json"
    frontier_run.write_text(json.dumps({
        "schema": "aw.stackoverflow-qa-production-frontier-run.v1",
        "status": "exact_frontier_complete_first_tranche_authorized",
        "stages": {"frontier": stage}}, sort_keys=True))
    parent = {"schema": "aw.stackoverflow-qa-production-tranche-run.v1",
              "status": "source_native_tranche_complete_pending_profile_fidelity_chunk_measurement",
              "counts": {"selected_questions": 10000, "markdown_documents": 10000,
                         "quarantine": 0}}
    pack = {"status": "ready_for_gpu_queue",
            "counts": {"documents": 10000, "chunks": 13112},
            "token_distribution": {"hard_max": 2560, "max": 2560,
                                   "over_hard_max": 0}}
    profile = {"schema": "aw.source-profile-compliance-receipt.v1",
               "profile": "community_qa", "checks": {"original": [{"status": "PASS"}]}}
    fidelity = {"schema": "aw.content-fidelity-receipt.v1",
                "original_bytes_complete": True, "markdown_content_complete": True,
                "source_text_truncation_applied": False}
    stage_evidence = {"schema": "aw.stage-receipt.v1", "status": "complete",
                      "rows": 13112}
    vector = {"schema": "aw.vector-rollup.v1", "status": "complete", "rows": 13112,
              "storage": {"dimensions": 1024, "dtype": "float16",
                          "binary_transfer_ready": True}, "verification": {"failed": 0}}
    paths = []
    for name, value in (("parent", parent), ("pack", pack), ("profile", profile),
                        ("fidelity", fidelity), ("stage", stage_evidence),
                        ("vector", vector)):
        path = source_root / f"{name}.json"
        path.write_text(json.dumps(value, sort_keys=True))
        paths.append(path)
    result = build(
        frontier_run=frontier_run, frontier=frontier, first_tranche=first,
        first_parent=paths[0], first_pack=paths[1], first_profile=paths[2],
        first_fidelity=paths[3], first_stage=paths[4], first_vector=paths[5],
        root=artifacts / "decision", artifact_root=artifacts,
        offset=10_000, limit=10_000)
    assert result["status"] == "exact_second_tranche_authorized_only"
    decision = result["decision"]
    assert decision["counts"] == {
        "questions": 10_000, "unique_user_rows": 5,
        "expected_source_rows_question_answer_user": 20_005}
    assert decision["selector"]["disjoint_from_first_tranche"] is True
    assert result["stages"]["frontier"]["records"] == 10_000
