#!/usr/bin/env python3
"""Materialize one sealed Stack Overflow production Q+A tranche.

Consumes the deterministic exact-selection tranche, preserves the exact source
XML rows for questions, accepted answers, and Users attribution, and then uses
the reviewed source-native community-QA renderer.  It makes no full-frontier
document or point projection.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import BinaryIO

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.extract.stackoverflow_pilot import (
    PROFILE, _json_bytes, _jsonl_bytes, _owner_ids, _publish_gzip,
    _raw_record, _run_7z, _xml_rows, materialize, select_users,
)
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as SS


VERSION = "stackoverflow-qa-production-tranche.v1"
PRODUCER = "pipeline/materialize_stackoverflow_qa_production_tranche.py"
#: Ordinals 1-7 were each authorized by their own hand-written
#: `authorize_stackoverflow_qa_<ordinal>_tranche.py`, and each minted its
#: own one-off status string -- a literal allowlist was the only option.
#: Ordinal 8+ is authorized by the generic, any-ordinal
#: `authorize_stackoverflow_qa_nth_tranche.py` (task 631, PR #295), whose
#: status is always `f"exact_tranche_{ordinal:02d}_authorized_only"` -- a
#: PATTERN, not a new literal to hand-add per tranche (that would just
#: repeat this exact bug at tranche 9, 10, ... every time). Found live,
#: task 663, 2026-09-25: tranche 8's real authorize run produced
#: `exact_tranche_08_authorized_only` and this allowlist, never updated
#: when #295 shipped the generic script, refused it with a bare
#: "frontier is not terminal" that named neither the status it got nor
#: the ones it would have accepted.
LEGACY_AUTHORIZED_FRONTIER_STATUSES = {
    "exact_frontier_complete_first_tranche_authorized",
    "exact_second_tranche_authorized_only",
    "exact_third_tranche_authorized_only",
    "exact_fourth_tranche_authorized_only",
    "exact_fifth_tranche_authorized_only",
    "exact_sixth_tranche_authorized_only",
    "exact_seventh_tranche_authorized_only",
}
_GENERIC_TRANCHE_STATUS_RE = re.compile(r"^exact_tranche_\d{2,}_authorized_only$")


def _require_authorized_frontier(frontier_seal: dict) -> None:
    status = frontier_seal.get("status")
    if (status not in LEGACY_AUTHORIZED_FRONTIER_STATUSES
            and not (isinstance(status, str)
                    and _GENERIC_TRANCHE_STATUS_RE.match(status))):
        raise ValueError(
            f"Stack Overflow exact frontier is not terminal: status "
            f"{status!r} is neither a legacy per-ordinal status "
            f"({sorted(LEGACY_AUTHORIZED_FRONTIER_STATUSES)}) nor matches "
            f"the generic-authorizer pattern {_GENERIC_TRANCHE_STATUS_RE.pattern!r}")


def _stage(*, artifact_root: Path, root: Path, name: str,
           paths: list[Path], records: int, fmt: str, logical: bytes) -> dict:
    return SS.seal(
        artifact_root=artifact_root, artifact_paths=paths,
        receipt_path=root / "stages" / name / "receipt.json",
        records=records, fmt=fmt,
        logical_sha256=hashlib.sha256(logical).hexdigest(),
        producer_id=PRODUCER, producer_version=VERSION)


def _load_frontier(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    identities = {(int(row["question_id"]), int(row["accepted_answer_id"]))
                  for row in rows}
    if len(identities) != len(rows):
        raise ValueError("production tranche contains duplicate Q+A identity")
    return rows


def select_exact_posts(stream: BinaryIO, *, frontier: list[dict]) -> dict:
    questions = {str(row["question_id"]): row for row in frontier}
    answers = {str(row["accepted_answer_id"]): row for row in frontier}
    question_records: dict[str, dict] = {}
    answer_records: dict[str, dict] = {}
    for raw, attrs in _xml_rows(stream):
        post_id = attrs.get("Id")
        if post_id in questions:
            row = questions[post_id]
            if attrs.get("PostTypeId") != "1":
                raise ValueError("selected question is not PostTypeId=1")
            if attrs.get("AcceptedAnswerId") != str(row["accepted_answer_id"]):
                raise ValueError("selected question accepted-answer identity drift")
            if hashlib.sha256(canonical_json(attrs).encode()).hexdigest() != row["question_attrs_sha256"]:
                raise ValueError("selected question attribute witness drift")
            question_records[post_id] = _raw_record(raw, attrs)
        if post_id in answers:
            row = answers[post_id]
            if attrs.get("PostTypeId") != "2":
                raise ValueError("selected accepted answer is not PostTypeId=2")
            if attrs.get("ParentId") != str(row["question_id"]):
                raise ValueError("selected answer parent identity drift")
            if hashlib.sha256(canonical_json(attrs).encode()).hexdigest() != row["answer_attrs_sha256"]:
                raise ValueError("selected answer attribute witness drift")
            answer_records[post_id] = _raw_record(raw, attrs)
    if set(question_records) != set(questions) or set(answer_records) != set(answers):
        raise ValueError("selected Stack Overflow Q+A source rows are incomplete")
    rows = []
    for row in frontier:
        qid, aid = str(row["question_id"]), str(row["accepted_answer_id"])
        rows.append({
            "cell": "accepted_qscore_10_plus",
            "selection_key": row["selection_key"],
            "question": question_records[qid],
            "accepted_answer": answer_records[aid],
            "top_answers_score_ge_3": [],
            "comparison_prior_amin3_qmin8_kans2": {
                "included": True, "question_min_score": 8,
                "answer_min_score": 3, "answer_cap": 2,
                "selected_answer_ids": [],
            },
        })
    return {
        "schema": "aw.stackoverflow-qa-exact-post-selection.v1",
        "version": VERSION,
        "counts": {"selected_questions": len(rows),
                   "selected_answers": len(rows)},
        "rows": rows,
    }


def _publish_outputs(*, root: Path, result: dict) -> tuple[dict, dict]:
    bodies = {
        "markdown": _jsonl_bytes(result["markdown"]),
        "outlinks": _jsonl_bytes(result["outlinks"]),
        "quarantine": _jsonl_bytes(result["quarantine"]),
    }
    paths = {name: root / "data" / f"{name}.jsonl.gz" for name in bodies}
    for name, body in bodies.items():
        _publish_gzip(paths[name], body)
    return paths, bodies


def build(*, posts_archive: Path, users_archive: Path, inventory_run: Path,
          frontier_run: Path, tranche: Path, root: Path,
          artifact_root: Path, expected_questions: int,
          expected_source_rows: int) -> dict:
    if (root / "RUN.json").exists():
        raise ValueError("Stack Overflow production tranche RUN already exists")
    root.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(inventory_run.read_text())
    if inventory.get("status") != "source_archives_crc_and_members_complete_pending_projection":
        raise ValueError("Stack Overflow archive inventory is not terminal")
    _validate_stage(inventory["stages"]["archive_inventory"],
                    "stackoverflow.archive_inventory", artifact_root=artifact_root,
                    deep=True)
    frontier_seal = json.loads(frontier_run.read_text())
    _require_authorized_frontier(frontier_seal)
    _validate_stage(frontier_seal["stages"]["frontier"],
                    "stackoverflow.production_frontier", artifact_root=artifact_root,
                    deep=True)
    frontier = _load_frontier(tranche)
    if len(frontier) != expected_questions:
        raise ValueError("Stack Overflow tranche question conservation failed")

    tranche_sha256 = sha256_file(tranche)
    posts_checkpoint = root / "checkpoints/posts-selection.json.gz"
    if posts_checkpoint.exists():
        selection = json.loads(gzip.decompress(posts_checkpoint.read_bytes()))
        if (selection.get("tranche_sha256") != tranche_sha256
                or selection.get("counts", {}).get("selected_questions") != expected_questions):
            raise ValueError("Stack Overflow posts checkpoint configuration drift")
    else:
        selection = _run_7z(posts_archive, "Posts.xml",
                            lambda stream: select_exact_posts(stream, frontier=frontier))
        selection["tranche_sha256"] = tranche_sha256
        _publish_gzip(posts_checkpoint, _json_bytes(selection))
    owner_ids = _owner_ids(selection)
    users_checkpoint = root / "checkpoints/users-selection.json.gz"
    if users_checkpoint.exists():
        user_selection = json.loads(gzip.decompress(users_checkpoint.read_bytes()))
        if user_selection.get("counts", {}).get("owner_ids_requested") != len(owner_ids):
            raise ValueError("Stack Overflow Users checkpoint configuration drift")
    else:
        user_selection = _run_7z(
            users_archive, "Users.xml",
            lambda stream: select_users(stream, owner_ids=owner_ids))
        _publish_gzip(users_checkpoint, _json_bytes(user_selection))
    source_rows = (selection["counts"]["selected_questions"]
                   + selection["counts"]["selected_answers"]
                   + user_selection["counts"]["owner_ids_found"])
    if (user_selection["counts"]["owner_ids_missing"] != 0
            or source_rows != expected_source_rows):
        raise ValueError("Stack Overflow exact selected source-row conservation failed")

    result = materialize(selection, user_selection)
    for row in result["markdown"]:
        row["extraction"]["producer"] = PRODUCER
        row["extraction"]["version"] = VERSION
    paths, bodies = _publish_outputs(root=root, result=result)
    metrics = {
        "schema": "aw.stackoverflow-qa-production-tranche-metrics.v1",
        "version": VERSION,
        "classification": "exact bounded source-native production tranche; no full-frontier point projection",
        "source_rows": source_rows,
        "selected_questions": expected_questions,
        "materialization": result["metrics"],
    }
    metrics_path = root / "data/metrics.json"
    metrics_body = _json_bytes(metrics)
    atomic_publish(metrics_path, metrics_body)
    original_logical = (gzip.decompress(posts_checkpoint.read_bytes())
                        + gzip.decompress(users_checkpoint.read_bytes())
                        + tranche.read_bytes() + frontier_run.read_bytes()
                        + inventory_run.read_bytes())
    stages = {
        "original": _stage(
            artifact_root=artifact_root, root=root, name="original",
            paths=[posts_checkpoint, users_checkpoint, tranche, frontier_run,
                   inventory_run], records=source_rows,
            fmt="official-StackOverflow-exact-Q+A+Users-XML-rows",
            logical=original_logical),
        "markdown": _stage(
            artifact_root=artifact_root, root=root, name="markdown",
            paths=[paths["markdown"]], records=len(result["markdown"]),
            fmt="canonical-jsonl+deterministic-gzip+full-source-native-QA-Markdown",
            logical=bodies["markdown"]),
        "outlinks": _stage(
            artifact_root=artifact_root, root=root, name="outlinks",
            paths=[paths["outlinks"]], records=len(result["outlinks"]),
            fmt="canonical-jsonl+deterministic-gzip+parent-bound-QA-relations",
            logical=bodies["outlinks"]),
        "quarantine": _stage(
            artifact_root=artifact_root, root=root, name="quarantine",
            paths=[paths["quarantine"]], records=len(result["quarantine"]),
            fmt="canonical-jsonl+deterministic-gzip+explicit-source-gate-failures",
            logical=bodies["quarantine"]),
        "metrics": _stage(
            artifact_root=artifact_root, root=root, name="metrics",
            paths=[metrics_path], records=1,
            fmt="canonical-json+exact-bounded-yield-no-extrapolation",
            logical=metrics_body),
    }
    run = {
        "schema": "aw.stackoverflow-qa-production-tranche-run.v1",
        "version": VERSION,
        "status": "source_native_tranche_complete_pending_profile_fidelity_chunk_measurement",
        "source_profile": PROFILE,
        "counts": {"selected_source_rows": source_rows,
                   "selected_questions": expected_questions,
                   "markdown_documents": len(result["markdown"]),
                   "outlink_rows": len(result["outlinks"]),
                   "quarantine": len(result["quarantine"])},
        "frontier_run_sha256": sha256_file(frontier_run),
        "archive_inventory_sha256": sha256_file(inventory_run),
        "content_fidelity": {"exact_selected_xml_rows": True,
                             "full_prechunk_markdown": True,
                             "source_content_cap": None},
        "stages": stages,
        "downstream": "community_qa profile/fidelity audit and measured current chunking; full frontier remains selection-only",
    }
    atomic_publish(root / "RUN.json", _json_bytes(run))
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("posts-archive", "users-archive", "inventory-run",
                 "frontier-run", "tranche", "root", "artifact-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, required=True)
    parser.add_argument("--expected-source-rows", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(build(
        posts_archive=args.posts_archive.resolve(),
        users_archive=args.users_archive.resolve(),
        inventory_run=args.inventory_run.resolve(),
        frontier_run=args.frontier_run.resolve(), tranche=args.tranche.resolve(),
        root=args.root.resolve(), artifact_root=args.artifact_root.resolve(),
        expected_questions=args.expected_questions,
        expected_source_rows=args.expected_source_rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
