import gzip
import hashlib
import json
from pathlib import Path

import pytest

from cytir.acquire.artifact_io import canonical_json, sha256_file
from cytir.sources.stackoverflow.authorize_third import build, _validate_terminal_tranche
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


def _terminal_tranche_evidence(chunks: int) -> dict:
    return {
        "parent": {"schema": "aw.stackoverflow-qa-production-tranche-run.v1",
                   "status": "source_native_tranche_complete_pending_profile_fidelity_chunk_measurement",
                   "counts": {"selected_questions": 10000, "markdown_documents": 10000,
                              "quarantine": 0}},
        "pack": {"status": "ready_for_gpu_queue",
                 "counts": {"documents": 10000, "chunks": chunks},
                 "token_distribution": {"hard_max": 2560, "max": 2560,
                                        "over_hard_max": 0}},
        "profile": {"schema": "aw.source-profile-compliance-receipt.v1",
                    "profile": "community_qa",
                    "checks": {"markdown": [{"status": "PASS"}]}},
        "fidelity": {"schema": "aw.content-fidelity-receipt.v1",
                     "original_bytes_complete": True, "markdown_content_complete": True,
                     "source_text_truncation_applied": False},
        "stage": {"schema": "aw.stage-receipt.v1", "status": "complete", "rows": chunks},
        "vector": {"schema": "aw.vector-rollup.v1", "status": "complete", "rows": chunks,
                   "storage": {"dimensions": 1024, "dtype": "float16",
                               "binary_transfer_ready": True},
                   "verification": {"failed": 0}},
    }


def test_terminal_tranche_gate_is_identical_regardless_of_ordinal():
    evidence = _terminal_tranche_evidence(13112)
    _validate_terminal_tranche(**evidence, label="first")
    _validate_terminal_tranche(**evidence, label="second")
    evidence["fidelity"]["source_text_truncation_applied"] = True
    with pytest.raises(ValueError, match="second.*fidelity"):
        _validate_terminal_tranche(**evidence, label="second")


def test_build_seals_exact_third_10k_and_both_prior_terminal_evidences(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    source_root = artifacts / "source"
    source_root.mkdir(parents=True)
    frontier = source_root / "eligible.jsonl.gz"
    _rows(frontier, 30_001)
    from cytir.sources.stackoverflow.authorize_second import _smallest
    ranked, _ = _smallest(frontier, 30_000)

    def _dump(path: Path, rows: list[dict]) -> Path:
        with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as target:
            for row in rows:
                target.write((canonical_json(row) + "\n").encode())
        return path

    first = _dump(source_root / "first.jsonl.gz", ranked[:10_000])
    second = _dump(source_root / "second.jsonl.gz", ranked[10_000:20_000])

    stage = SS.seal(
        artifact_root=artifacts, artifact_paths=[frontier, first, second],
        receipt_path=source_root / "receipt.json", records=30_001,
        fmt="fixture", logical_sha256=sha256_file(frontier),
        producer_id="fixture", producer_version="v1")
    frontier_run = source_root / "RUN.json"
    frontier_run.write_text(json.dumps({
        "schema": "aw.stackoverflow-qa-production-frontier-run.v1",
        "status": "exact_frontier_complete_first_tranche_authorized",
        "stages": {"frontier": stage}}, sort_keys=True))

    def _write_evidence(prefix: str, chunks: int) -> dict:
        evidence = _terminal_tranche_evidence(chunks)
        paths = {}
        for name, value in evidence.items():
            path = source_root / f"{prefix}-{name}.json"
            path.write_text(json.dumps(value, sort_keys=True))
            paths[name] = path
        return paths

    first_paths = _write_evidence("first", 13112)
    second_paths = _write_evidence("second", 13063)

    result = build(
        frontier_run=frontier_run, frontier=frontier,
        first_tranche=first, first_parent=first_paths["parent"],
        first_pack=first_paths["pack"], first_profile=first_paths["profile"],
        first_fidelity=first_paths["fidelity"], first_stage=first_paths["stage"],
        first_vector=first_paths["vector"],
        second_tranche=second, second_parent=second_paths["parent"],
        second_pack=second_paths["pack"], second_profile=second_paths["profile"],
        second_fidelity=second_paths["fidelity"], second_stage=second_paths["stage"],
        second_vector=second_paths["vector"],
        root=artifacts / "decision", artifact_root=artifacts,
        offset=20_000, limit=10_000)

    assert result["status"] == "exact_third_tranche_authorized_only"
    decision = result["decision"]
    assert decision["counts"]["questions"] == 10_000
    assert decision["selector"]["disjoint_from_prior_tranches"] is True
    assert result["stages"]["frontier"]["records"] == 10_000
    selected_keys = {row["selection_key"] for row in ranked[20_000:30_000]}
    with gzip.open(artifacts / "decision/frontier/third-020000-029999.jsonl.gz",
                   "rt", encoding="utf-8") as source:
        emitted_keys = {json.loads(line)["selection_key"] for line in source}
    assert emitted_keys == selected_keys


def test_build_refuses_when_offset_is_not_the_authorized_third_window(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="only the explicitly authorized third"):
        build(
            frontier_run=Path("/nonexistent"), frontier=Path("/nonexistent"),
            first_tranche=Path("/nonexistent"), first_parent=Path("/nonexistent"),
            first_pack=Path("/nonexistent"), first_profile=Path("/nonexistent"),
            first_fidelity=Path("/nonexistent"), first_stage=Path("/nonexistent"),
            first_vector=Path("/nonexistent"),
            second_tranche=Path("/nonexistent"), second_parent=Path("/nonexistent"),
            second_pack=Path("/nonexistent"), second_profile=Path("/nonexistent"),
            second_fidelity=Path("/nonexistent"), second_stage=Path("/nonexistent"),
            second_vector=Path("/nonexistent"),
            root=artifacts / "decision", artifact_root=artifacts,
            offset=10_000, limit=10_000)
