import gzip
import hashlib
import json
from pathlib import Path

import pytest

from cytir.acquire.artifact_io import canonical_json, sha256_file
from cytir.sources.stackoverflow.authorize_nth import build
from cytir.sources.stackoverflow.authorize_third import _validate_terminal_tranche
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


def _dump(path: Path, rows: list[dict]) -> Path:
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as target:
        for row in rows:
            target.write((canonical_json(row) + "\n").encode())
    return path


def _write_evidence(source_root: Path, prefix: str, chunks: int) -> dict:
    evidence = _terminal_tranche_evidence(chunks)
    paths = {}
    for name, value in evidence.items():
        path = source_root / f"{prefix}-{name}.json"
        path.write_text(json.dumps(value, sort_keys=True))
        paths[name] = path
    return paths


def _build_seven_tranche_fixture(tmp_path: Path):
    """Same fixture shape as test_authorize_stackoverflow_qa_seventh_tranche.py:
    one immutable ranked frontier, 6 sealed prior tranches (chunk counts taken
    from the real admitted ledger rows for tranches 1-6), ready to authorize
    the 7th (or, generically, ordinal=7 with 6 priors)."""
    artifacts = tmp_path / "artifacts"
    source_root = artifacts / "source"
    source_root.mkdir(parents=True)
    frontier = source_root / "eligible.jsonl.gz"
    _rows(frontier, 70001)
    from cytir.sources.stackoverflow.authorize_second import _smallest
    ranked, _ = _smallest(frontier, 70000)

    windows = [ranked[i * 10000:(i + 1) * 10000] for i in range(6)]
    names = ["first", "second", "third", "fourth", "fifth", "sixth"]
    real_chunk_counts = [13112, 13063, 12820, 13038, 13012, 12919]  # from data/expansion-pack-admission-status.v1.json
    tranche_paths = [_dump(source_root / f"{name}.jsonl.gz", window)
                     for name, window in zip(names, windows)]

    stage = SS.seal(
        artifact_root=artifacts, artifact_paths=[frontier, *tranche_paths],
        receipt_path=source_root / "receipt.json", records=70001,
        fmt="fixture", logical_sha256=sha256_file(frontier),
        producer_id="fixture", producer_version="v1")
    frontier_run = source_root / "RUN.json"
    frontier_run.write_text(json.dumps({
        "schema": "aw.stackoverflow-qa-production-frontier-run.v1",
        "status": "exact_frontier_complete_first_tranche_authorized",
        "stages": {"frontier": stage}}, sort_keys=True))

    evidence_paths = [_write_evidence(source_root, name, chunks)
                      for name, chunks in zip(names, real_chunk_counts)]
    priors = [{"tranche": tranche_paths[i], **evidence_paths[i]} for i in range(6)]
    return artifacts, frontier_run, frontier, ranked, priors


def test_build_selects_exactly_the_seventh_10k_rank_window(tmp_path: Path):
    """Ordinal 7 (offset 60000) emits exactly rank interval [60000, 70000) of the
    deterministic frontier: the emitted rows are the ranked slice, byte for byte,
    and the decision's selection hash is the hash of that same content."""
    artifacts, frontier_run, frontier, ranked, priors = _build_seven_tranche_fixture(tmp_path / "a")
    generic = build(
        frontier_run=frontier_run, frontier=frontier, priors=priors,
        ordinal=7, root=artifacts / "decision", artifact_root=artifacts,
        offset=60000, limit=10000)

    expected = b"".join((canonical_json(row) + "\n").encode() for row in ranked[60000:70000])
    with gzip.open(artifacts / "decision/frontier/tranche-060000-069999.jsonl.gz", "rb") as f:
        assert f.read() == expected
    assert (generic["decision"]["selector"]["selection_logical_sha256"]
            == hashlib.sha256(expected).hexdigest())
    assert generic["decision"]["counts"]["questions"] == 10000
    assert generic["decision"]["selector"]["offset"] == 60000
    assert generic["decision"]["selector"]["limit"] == 10000
    assert generic["decision"]["selector"]["disjoint_from_prior_tranches"] is True
    assert generic["decision"]["ordinal"] == 7
    assert generic["status"] == "exact_tranche_07_authorized_only"


def test_build_refuses_when_stated_offset_overlaps_a_prior_tranche(tmp_path: Path):
    """3 valid, mutually-consistent priors (offsets 0/10000/20000, true derived
    offset for tranche 4 = 30000) but the caller asks for offset=25000 -- which
    would start the new window 5000 rows INSIDE the third prior's already-
    claimed [20000, 30000) range. Must refuse, not silently authorize an
    overlapping slice."""
    artifacts = tmp_path / "artifacts"
    source_root = artifacts / "source"
    source_root.mkdir(parents=True)
    frontier = source_root / "eligible.jsonl.gz"
    _rows(frontier, 40001)
    from cytir.sources.stackoverflow.authorize_second import _smallest
    ranked, _ = _smallest(frontier, 40000)

    names = ["first", "second", "third"]
    windows = [ranked[0:10000], ranked[10000:20000], ranked[20000:30000]]
    chunk_counts = [13112, 13063, 12820]
    tranche_paths = [_dump(source_root / f"{n}.jsonl.gz", w) for n, w in zip(names, windows)]
    stage = SS.seal(
        artifact_root=artifacts, artifact_paths=[frontier, *tranche_paths],
        receipt_path=source_root / "receipt.json", records=40001,
        fmt="fixture", logical_sha256=sha256_file(frontier),
        producer_id="fixture", producer_version="v1")
    frontier_run = source_root / "RUN.json"
    frontier_run.write_text(json.dumps({
        "schema": "aw.stackoverflow-qa-production-frontier-run.v1",
        "status": "exact_frontier_complete_first_tranche_authorized",
        "stages": {"frontier": stage}}, sort_keys=True))
    evidence_paths = [_write_evidence(source_root, n, c) for n, c in zip(names, chunk_counts)]
    priors = [{"tranche": tranche_paths[i], **evidence_paths[i]} for i in range(3)]

    with pytest.raises(ValueError, match="does not match the offset derived"):
        build(frontier_run=frontier_run, frontier=frontier, priors=priors,
              ordinal=4, root=artifacts / "decision", artifact_root=artifacts,
              offset=25000, limit=10000)
    assert not (artifacts / "decision").exists()


def test_build_refuses_when_a_prior_is_duplicated(tmp_path: Path):
    """Passing the same prior tranche twice (a plausible caller mistake when
    assembling a long --prior-* list by hand) creates a real key overlap
    between "priors" even though each individually validates -- must refuse."""
    artifacts = tmp_path / "artifacts"
    source_root = artifacts / "source"
    source_root.mkdir(parents=True)
    frontier = source_root / "eligible.jsonl.gz"
    _rows(frontier, 20001)
    from cytir.sources.stackoverflow.authorize_second import _smallest
    ranked, _ = _smallest(frontier, 20000)
    tranche = _dump(source_root / "first.jsonl.gz", ranked[0:10000])
    stage = SS.seal(
        artifact_root=artifacts, artifact_paths=[frontier, tranche],
        receipt_path=source_root / "receipt.json", records=20001,
        fmt="fixture", logical_sha256=sha256_file(frontier),
        producer_id="fixture", producer_version="v1")
    frontier_run = source_root / "RUN.json"
    frontier_run.write_text(json.dumps({
        "schema": "aw.stackoverflow-qa-production-frontier-run.v1",
        "status": "exact_frontier_complete_first_tranche_authorized",
        "stages": {"frontier": stage}}, sort_keys=True))
    evidence = _write_evidence(source_root, "first", 13112)
    prior = {"tranche": tranche, **evidence}

    with pytest.raises(ValueError, match="overlap each other"):
        build(frontier_run=frontier_run, frontier=frontier, priors=[prior, prior],
              ordinal=3, root=artifacts / "decision", artifact_root=artifacts)


def test_build_derives_offset_when_not_stated(tmp_path: Path):
    """No --offset given: the module must derive it from the priors' own
    receipts (a single prior of 10000 rows -> ordinal 2 starts at 10000) and
    succeed, matching what authorize_stackoverflow_qa_second_tranche.py's own
    hardcoded OFFSET=10000 encodes."""
    artifacts = tmp_path / "artifacts"
    source_root = artifacts / "source"
    source_root.mkdir(parents=True)
    frontier = source_root / "eligible.jsonl.gz"
    _rows(frontier, 20001)
    from cytir.sources.stackoverflow.authorize_second import _smallest
    ranked, _ = _smallest(frontier, 20000)
    tranche = _dump(source_root / "first.jsonl.gz", ranked[0:10000])
    stage = SS.seal(
        artifact_root=artifacts, artifact_paths=[frontier, tranche],
        receipt_path=source_root / "receipt.json", records=20001,
        fmt="fixture", logical_sha256=sha256_file(frontier),
        producer_id="fixture", producer_version="v1")
    frontier_run = source_root / "RUN.json"
    frontier_run.write_text(json.dumps({
        "schema": "aw.stackoverflow-qa-production-frontier-run.v1",
        "status": "exact_frontier_complete_first_tranche_authorized",
        "stages": {"frontier": stage}}, sort_keys=True))
    evidence = _write_evidence(source_root, "first", 13112)
    prior = {"tranche": tranche, **evidence}

    result = build(frontier_run=frontier_run, frontier=frontier, priors=[prior],
                   ordinal=2, root=artifacts / "decision", artifact_root=artifacts)
    assert result["decision"]["selector"]["offset"] == 10000
    selected_keys = {row["selection_key"] for row in ranked[10000:20000]}
    with gzip.open(artifacts / "decision/frontier/tranche-010000-019999.jsonl.gz",
                   "rt", encoding="utf-8") as f:
        emitted = {json.loads(line)["selection_key"] for line in f}
    assert emitted == selected_keys


def test_build_refuses_wrong_prior_count_for_ordinal(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="requires exactly 6 prior"):
        build(frontier_run=Path("/nonexistent"), frontier=Path("/nonexistent"),
              priors=[{}, {}], ordinal=7, root=artifacts / "decision",
              artifact_root=artifacts)


def test_terminal_tranche_gate_reused_unforked_from_third_module():
    evidence = _terminal_tranche_evidence(12919)
    _validate_terminal_tranche(**evidence, label="sixth")
    evidence["fidelity"]["source_text_truncation_applied"] = True
    with pytest.raises(ValueError, match="sixth.*fidelity"):
        _validate_terminal_tranche(**evidence, label="sixth")
