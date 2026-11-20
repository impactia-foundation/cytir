"""Build an exact, source-native Stack Overflow Q+accepted-A frontier.

This is a selection stage, not a document or point projection.  It scans the
sealed Posts and Users members, conserves the terminal census score>=10 cell,
and emits exact eligible question/answer/user identities plus a deterministic
first production tranche.  Full bodies remain in the preserved archives and
must be materialized without caps by the downstream source-profile adapter.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import sqlite3
import subprocess
from typing import BinaryIO

from lxml import etree

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.extract import stackoverflow_qa as RULE
from cytir.extract.stackoverflow_qa import ALLOWED_LICENSES, MIN_SCORE
from cytir.sources.stackoverflow.authorize_second import _stream_gzip
from cytir.acquire import stage_receipt as SS


VERSION = "stackoverflow-qa-production-frontier.v1"
PRODUCER = "aw/sources/stackoverflow/frontier.py"


def _attrs_hash(element) -> str:
    return hashlib.sha256(canonical_json(dict(element.attrib)).encode()).hexdigest()


def _integer(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


def _identity_fields(element) -> tuple[int | None, str | None]:
    return _integer(element.get("OwnerUserId")), element.get("OwnerDisplayName")


def consume_posts(stream: BinaryIO, connection: sqlite3.Connection) -> dict:
    """Inventory exact score>=10 question and referenced-answer identities."""
    connection.executescript("""
      CREATE TABLE questions(
        question_id INTEGER PRIMARY KEY, accepted_answer_id INTEGER UNIQUE NOT NULL,
        score INTEGER NOT NULL, license TEXT, owner_user_id INTEGER,
        owner_display_name TEXT, creation_time TEXT, attrs_sha256 TEXT NOT NULL);
      CREATE TABLE answers(
        answer_id INTEGER PRIMARY KEY, parent_question_id INTEGER, score INTEGER,
        license TEXT, owner_user_id INTEGER, owner_display_name TEXT,
        creation_time TEXT, attrs_sha256 TEXT NOT NULL);
    """)
    wanted_answers: set[int] = set()
    counts = Counter()
    questions_batch, answers_batch = [], []
    context = etree.iterparse(stream, events=("end",), tag="row", huge_tree=True)
    for _, element in context:
        post_type = element.get("PostTypeId")
        if post_type == "1":
            accepted_id = _integer(element.get("AcceptedAnswerId"))
            score = _integer(element.get("Score")) or 0
            if accepted_id is not None and RULE.score_ok(score):
                question_id = _integer(element.get("Id"))
                if question_id is None:
                    raise ValueError("eligible question lacks source ID")
                owner_id, owner_display = _identity_fields(element)
                questions_batch.append((
                    question_id, accepted_id, score, element.get("ContentLicense"),
                    owner_id, owner_display, element.get("CreationDate"),
                    _attrs_hash(element)))
                wanted_answers.add(accepted_id)
                counts["questions_score10_with_accepted_id"] += 1
        elif post_type == "2":
            answer_id = _integer(element.get("Id"))
            if answer_id is not None and answer_id in wanted_answers:
                owner_id, owner_display = _identity_fields(element)
                answers_batch.append((
                    answer_id, _integer(element.get("ParentId")),
                    _integer(element.get("Score")) or 0,
                    element.get("ContentLicense"), owner_id, owner_display,
                    element.get("CreationDate"), _attrs_hash(element)))
                counts["referenced_answers_present"] += 1
        if len(questions_batch) >= 10000:
            connection.executemany("INSERT INTO questions VALUES(?,?,?,?,?,?,?,?)",
                                   questions_batch)
            questions_batch.clear()
        if len(answers_batch) >= 10000:
            connection.executemany("INSERT INTO answers VALUES(?,?,?,?,?,?,?,?)",
                                   answers_batch)
            answers_batch.clear()
        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]
    if questions_batch:
        connection.executemany("INSERT INTO questions VALUES(?,?,?,?,?,?,?,?)",
                               questions_batch)
    if answers_batch:
        connection.executemany("INSERT INTO answers VALUES(?,?,?,?,?,?,?,?)",
                               answers_batch)
    connection.commit()
    return dict(counts)


def requested_user_ids(connection: sqlite3.Connection) -> set[int]:
    rows = connection.execute("""
      SELECT owner_user_id FROM questions WHERE owner_user_id IS NOT NULL
      UNION SELECT owner_user_id FROM answers WHERE owner_user_id IS NOT NULL
    """)
    return {int(row[0]) for row in rows}


def consume_users(stream: BinaryIO, connection: sqlite3.Connection,
                  wanted: set[int]) -> dict:
    connection.execute("CREATE TABLE users(user_id INTEGER PRIMARY KEY, display_name TEXT)")
    batch, seen = [], 0
    context = etree.iterparse(stream, events=("end",), tag="row", huge_tree=True)
    for _, element in context:
        user_id = _integer(element.get("Id"))
        if user_id is not None and user_id in wanted:
            batch.append((user_id, element.get("DisplayName")))
            seen += 1
        if len(batch) >= 10000:
            connection.executemany("INSERT INTO users VALUES(?,?)", batch)
            batch.clear()
        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]
    if batch:
        connection.executemany("INSERT INTO users VALUES(?,?)", batch)
    connection.commit()
    return {"requested_user_ids": len(wanted), "found_user_rows": seen,
            "missing_user_rows": len(wanted) - seen}


def _eligible_rows(connection: sqlite3.Connection):
    condition, params = RULE.eligible_sql()
    query = f"""
      SELECT q.question_id,q.accepted_answer_id,q.score,q.license,
             q.owner_user_id,q.owner_display_name,q.creation_time,q.attrs_sha256,
             a.score,a.license,a.owner_user_id,a.owner_display_name,
             a.creation_time,a.attrs_sha256
      FROM questions q JOIN answers a ON a.answer_id=q.accepted_answer_id
      LEFT JOIN users qu ON qu.user_id=q.owner_user_id
      LEFT JOIN users au ON au.user_id=a.owner_user_id
      WHERE {condition}
      ORDER BY q.question_id
    """
    return connection.execute(query, params)


def _row(columns) -> dict:
    (qid, aid, qscore, qlicense, quid, qdisplay, qtime, qhash,
     ascore, alicense, auid, adisplay, atime, ahash) = columns
    return {
        "question_id": int(qid), "accepted_answer_id": int(aid),
        "question_score": int(qscore), "accepted_answer_score": int(ascore),
        "question_license": qlicense, "accepted_answer_license": alicense,
        "question_owner_user_id": quid, "question_owner_display_name": qdisplay,
        "answer_owner_user_id": auid, "answer_owner_display_name": adisplay,
        "question_creation_time": qtime, "answer_creation_time": atime,
        "question_attrs_sha256": qhash, "answer_attrs_sha256": ahash,
        "selection_key": hashlib.sha256(
            f"stackoverflow-production-q\0{qid}\0{aid}".encode()).hexdigest(),
    }


def finalize(connection: sqlite3.Connection, *, root: Path,
             tranche_size: int) -> dict:
    # Stream the full exact frontier and keep only the deterministic bottom-k
    # first tranche in memory.
    frontier_path = root / "frontier/eligible-score10.jsonl.gz"
    heap: list[tuple[int, int, dict]] = []
    eligible_count = 0
    def rows():
        nonlocal eligible_count
        for columns in _eligible_rows(connection):
            value = _row(columns); eligible_count += 1
            rank = int(value["selection_key"], 16)
            item = (-rank, int(value["question_id"]), value)
            if len(heap) < tranche_size:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            yield value
    full_count, full_logical = _stream_gzip(frontier_path, rows())
    tranche = [value for _, _, value in sorted(heap, key=lambda item: -item[0])]
    tranche.sort(key=lambda value: value["selection_key"])
    tranche_path = root / f"frontier/first-{tranche_size:06d}.jsonl.gz"
    tranche_count, tranche_logical = _stream_gzip(tranche_path, tranche)
    if full_count != eligible_count or tranche_count != min(tranche_size, full_count):
        raise ValueError("Stack Overflow frontier/tranche conservation failed")
    full_user_ids = set()
    for row in _eligible_rows(connection):
        for value in (row[4], row[10]):
            if value is not None:
                full_user_ids.add(int(value))
    tranche_user_ids = {int(value) for row in tranche for value in
                        (row["question_owner_user_id"], row["answer_owner_user_id"])
                        if value is not None}
    return {
        "frontier_path": frontier_path, "frontier_records": full_count,
        "frontier_logical_sha256": full_logical,
        "tranche_path": tranche_path, "tranche_records": tranche_count,
        "tranche_logical_sha256": tranche_logical,
        "full_expected_source_rows": 2 * full_count + len(full_user_ids),
        "full_unique_user_rows": len(full_user_ids),
        "tranche_expected_source_rows": 2 * tranche_count + len(tranche_user_ids),
        "tranche_unique_user_rows": len(tranche_user_ids),
    }


def _run_7z(archive: Path, member: str, consumer):
    process = subprocess.Popen(["7z", "x", "-so", str(archive), member],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    try:
        value = consumer(process.stdout)
    finally:
        process.stdout.close()
    stderr = process.stderr.read() if process.stderr else b""
    rc = process.wait()
    if rc:
        raise ValueError(f"7z {member} failed rc={rc}: {stderr[-1000:]!r}")
    return value


def build(*, posts_archive: Path, users_archive: Path, inventory_run: Path,
          census_run: Path, pilot_run: Path, pilot_pack: Path,
          vector_receipt: Path, root: Path, artifact_root: Path,
          tranche_size: int) -> dict:
    if root.exists():
        raise ValueError("Stack Overflow production frontier root must be fresh")
    root.mkdir(parents=True)
    inventory = json.loads(inventory_run.read_text())
    census = json.loads(census_run.read_text())
    census_score10 = int(census["counts"]["questions_with_accepted_answer_by_min_score"]["10"])
    if (inventory.get("status") !=
            "source_archives_crc_and_members_complete_pending_projection"
            or census.get("status") != "complete" or census_score10 <= 0):
        raise ValueError("terminal Stack Overflow census required")
    pilot = json.loads(pilot_run.read_text()); pack = json.loads(pilot_pack.read_text())
    vector = json.loads(vector_receipt.read_text())
    if (pilot.get("status") != "source_native_pilot_complete_pending_profile_fidelity_chunk_measurement"
            or pack.get("status") != "ready_for_gpu_queue"
            or vector.get("status") != "complete"):
        raise ValueError("terminal measured pilot evidence required")
    database = root / "work/selection.sqlite"
    database.parent.mkdir()
    connection = sqlite3.connect(database)
    try:
        post_counts = _run_7z(posts_archive, "Posts.xml",
                              lambda stream: consume_posts(stream, connection))
        if post_counts["questions_score10_with_accepted_id"] != census_score10:
            raise ValueError("score>=10 question count disagrees with terminal census")
        wanted = requested_user_ids(connection)
        _run_7z(users_archive, "Users.xml",
                              lambda stream: consume_users(stream, connection, wanted))
        measured = finalize(connection, root=root, tranche_size=tranche_size)
        answer_present = connection.execute("SELECT count(*) FROM questions q JOIN answers a "
            "ON a.answer_id=q.accepted_answer_id WHERE a.parent_question_id=q.question_id").fetchone()[0]
        marks = ",".join("?" for _ in ALLOWED_LICENSES)
        allowed_pairs = connection.execute("SELECT count(*) FROM questions q JOIN answers a "
            "ON a.answer_id=q.accepted_answer_id WHERE a.parent_question_id=q.question_id "
            f"AND q.license IN ({marks}) AND a.license IN ({marks})",
            (*ALLOWED_LICENSES, *ALLOWED_LICENSES)).fetchone()[0]
        attribution_pairs = connection.execute("""
            SELECT count(*) FROM questions q JOIN answers a
              ON a.answer_id=q.accepted_answer_id
            LEFT JOIN users qu ON qu.user_id=q.owner_user_id
            LEFT JOIN users au ON au.user_id=a.owner_user_id
            WHERE a.parent_question_id=q.question_id
              AND ((q.owner_user_id IS NOT NULL AND qu.user_id IS NOT NULL)
                   OR coalesce(q.owner_display_name,'')!='')
              AND ((a.owner_user_id IS NOT NULL AND au.user_id IS NOT NULL)
                   OR coalesce(a.owner_display_name,'')!='')
            """).fetchone()[0]
        license_pair_counts = {
            f"{question_license or 'MISSING'} -> {answer_license or 'MISSING'}": count
            for question_license, answer_license, count in connection.execute("""
              SELECT q.license,a.license,count(*) FROM questions q JOIN answers a
                ON a.answer_id=q.accepted_answer_id
              WHERE a.parent_question_id=q.question_id
              GROUP BY q.license,a.license ORDER BY q.license,a.license
            """)
        }
    finally:
        connection.close()
    decision = {
        "schema": "aw.stackoverflow-qa-production-selection-decision.v1",
        "version": VERSION, "status": "exact_frontier_complete",
        "classification": "exact source selection; no extrapolated document or point count",
        "quality_arm": {"question_min_score": MIN_SCORE,
                        "accepted_answer_required": True,
                        "licenses": list(ALLOWED_LICENSES),
                        "attribution_identity_required": True},
        "rationale": "ADR-019 score>=10 accepted-answer arm retained; measured pilot score>=10 cell retained 218/256 before current chunking",
        "exact_counts": {
            "census_score10_with_accepted_id": census_score10,
            "accepted_answer_rows_present": answer_present,
            "accepted_answer_rows_missing": census_score10 - answer_present,
            "license_pair_counts": license_pair_counts,
            "allowed_license_pairs": allowed_pairs,
            "attribution_reconstructible_pairs": attribution_pairs,
            "eligible_question_answer_pairs": measured["frontier_records"],
            "full_unique_user_rows": measured["full_unique_user_rows"],
            "full_expected_source_rows_question_answer_user": measured["full_expected_source_rows"],
            "first_tranche_questions": measured["tranche_records"],
            "first_tranche_unique_user_rows": measured["tranche_unique_user_rows"],
            "first_tranche_expected_source_rows_question_answer_user": measured["tranche_expected_source_rows"],
        },
        "pilot": {"documents": pack["counts"]["documents"],
                  "chunks": pack["counts"]["chunks"],
                  "measured_chunks_per_document": pack["counts"]["measured_chunks_per_document"],
                  "vector_rows": vector["rows"]},
        "admission": "first tranche only; full frontier requires separately sealed source-native Markdown/outlinks/profile/fidelity/token/UUID gates",
    }
    decision_path = root / "SELECTION.json"
    atomic_publish(decision_path, (json.dumps(decision, indent=2, sort_keys=True)+"\n").encode())
    artifacts = [decision_path, measured["frontier_path"], measured["tranche_path"],
                 inventory_run, census_run, pilot_run, pilot_pack, vector_receipt]
    stage = SS.seal(artifact_root=artifact_root, artifact_paths=artifacts,
                    receipt_path=root / "stages/frontier/receipt.json",
                    records=measured["frontier_records"],
                    fmt="exact-source-native-Q+accepted-A+user-selection-frontier",
                    logical_sha256=hashlib.sha256((
                        measured["frontier_logical_sha256"] +
                        measured["tranche_logical_sha256"] + sha256_file(decision_path)
                    ).encode()).hexdigest(), producer_id=PRODUCER,
                    producer_version=VERSION)
    run = {"schema": "aw.stackoverflow-qa-production-frontier-run.v1",
           "version": VERSION, "status": "exact_frontier_complete_first_tranche_authorized",
           "decision": decision, "stages": {"frontier": stage},
           "next": "materialize exact first-tranche source XML rows, full uncapped Q+A Markdown, Users attribution, and parent-bound outlinks"}
    atomic_publish(root / "RUN.json", (json.dumps(run, indent=2, sort_keys=True)+"\n").encode())
    return run


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    p = argparse.ArgumentParser(prog=prog or "frontier", description=__doc__)
    for name in ("posts-archive", "users-archive", "inventory-run", "census-run",
                 "pilot-run", "pilot-pack", "vector-receipt", "root", "artifact-root"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--tranche-size", type=int, default=10000)
    a = p.parse_args(argv)
    print(json.dumps(build(posts_archive=a.posts_archive.resolve(),
        users_archive=a.users_archive.resolve(), inventory_run=a.inventory_run.resolve(),
        census_run=a.census_run.resolve(), pilot_run=a.pilot_run.resolve(),
        pilot_pack=a.pilot_pack.resolve(), vector_receipt=a.vector_receipt.resolve(),
        root=a.root.resolve(), artifact_root=a.artifact_root.resolve(),
        tranche_size=a.tranche_size), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
