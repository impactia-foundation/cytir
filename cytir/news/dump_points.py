#!/usr/bin/env python3
"""Reconstruct current NEWS points from bounded, hash-pinned candidates.

Historical canonical collections do not index ``disc_class``. A filtered
scroll over tens of millions of points can therefore time out inside Qdrant.
This tool instead consumes the scheduler-owned NEWS candidate registry, unions
the freshly manifested staging arms, and re-reads those exact IDs from the
read-only twin. The current subset feeds aged-out deletion; the expanded
candidate registry is the successor that may be pinned only after a terminally
verified canonical apply.

If ``disc_class`` becomes indexed, the current dump may use it directly, but
the candidate successor is still advanced so a later collection without that
index retains a complete bounded path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable

from qdrant_client import models as qm

from cytir.core.qdrant import qdrant_client


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
PAYLOAD_FIELDS = ["url", "published", "domain", "disc_class", "source_type"]


class DumpError(RuntimeError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _key(point_id: Any) -> str:
    return json.dumps(point_id, sort_keys=True)


def _payload_schema(client: Any, collection: str) -> dict[str, Any]:
    info = client.get_collection(collection)
    schema = getattr(info, "payload_schema", None)
    if not isinstance(schema, dict):
        raise DumpError("collection payload schema is absent or malformed")
    return schema


def _scroll(client: Any, collection: str, flt: qm.Filter | None,
            batch: int) -> Iterable[Any]:
    offset = None
    while True:
        points, offset = client.scroll(
            collection, scroll_filter=flt, limit=batch, offset=offset,
            with_payload=PAYLOAD_FIELDS, with_vectors=False)
        yield from points
        if offset is None:
            return


def _load_candidates(path: Path, expected_sha256: str,
                     disc_class: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise DumpError("candidate registry is absent or not a regular file")
    if not SHA256.fullmatch(expected_sha256):
        raise DumpError("expected candidate-registry SHA-256 is malformed")
    actual_sha256 = digest(path)
    if actual_sha256 != expected_sha256:
        raise DumpError("candidate-registry SHA-256 mismatch")
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as source:
        for line_no, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DumpError(
                    f"candidate registry has invalid JSON at line {line_no}"
                ) from exc
            point_id = row.get("id")
            payload = row.get("payload")
            if point_id is None or not isinstance(payload, dict):
                raise DumpError(f"candidate row {line_no} is malformed")
            if payload.get("disc_class") != disc_class:
                raise DumpError(
                    f"candidate row {line_no} is not disc_class={disc_class}"
                )
            identity = _key(point_id)
            if identity in rows:
                raise DumpError(f"candidate registry repeats ID at line {line_no}")
            rows[identity] = {"id": point_id, "payload": payload}
    if not rows:
        raise DumpError("candidate registry is empty")
    return rows, {"path": str(path), "sha256": actual_sha256,
                  "points": len(rows)}


def _current_candidates(client: Any, collection: str,
                        candidates: dict[str, dict[str, Any]], disc_class: str,
                        batch: int) -> Iterable[Any]:
    ids = [row["id"] for _, row in sorted(candidates.items())]
    for start in range(0, len(ids), batch):
        points = client.retrieve(
            collection, ids=ids[start:start + batch],
            with_payload=PAYLOAD_FIELDS, with_vectors=False)
        for point in points:
            if (point.payload or {}).get("disc_class") == disc_class:
                yield point


def collect_news_points(client: Any, collection: str, disc_class: str,
                        batch: int, candidate_registry: Path,
                        candidate_registry_sha256: str,
                        staged_points: Iterable[Any]
                        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                   dict[str, Any]]:
    candidates, predecessor = _load_candidates(
        candidate_registry, candidate_registry_sha256, disc_class)
    staged_added = 0
    for point in staged_points:
        payload = point.payload or {}
        if payload.get("disc_class") != disc_class:
            raise DumpError("manifested staging arm contains a non-NEWS point")
        identity = _key(point.id)
        if identity not in candidates:
            staged_added += 1
        candidates[identity] = {"id": point.id, "payload": payload}

    schema = _payload_schema(client, collection)
    current: dict[str, dict[str, Any]] = {}

    def keep(point: Any) -> None:
        payload = point.payload or {}
        if payload.get("disc_class") == disc_class:
            current[_key(point.id)] = {"id": point.id, "payload": payload}

    if "disc_class" in schema:
        direct = qm.Filter(must=[qm.FieldCondition(
            key="disc_class", match=qm.MatchValue(value=disc_class))])
        for point in _scroll(client, collection, direct, batch):
            keep(point)
        mode = "indexed-disc-class"
    else:
        for point in _current_candidates(
                client, collection, candidates, disc_class, batch):
            keep(point)
        mode = "candidate-registry-reread"

    if not current:
        raise DumpError("current NEWS point reconstruction is empty")
    current_rows = [current[key] for key in sorted(current)]
    candidate_rows = [candidates[key] for key in sorted(candidates)]
    receipt = {
        "schema": "aw.news-point-dump.v3",
        "collection": collection,
        "disc_class": disc_class,
        "mode": mode,
        "predecessor_candidates": predecessor,
        "staged_candidates_added": staged_added,
        "successor_candidates": len(candidate_rows),
        "current_points": len(current_rows),
    }
    return current_rows, candidate_rows, receipt


def _write_temp(rows: list[dict[str, Any]], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "w", dir=out.parent, prefix=out.name + ".", delete=False) as f:
        temp = Path(f.name)
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return temp


def atomic_write(current_rows: list[dict[str, Any]], current_out: Path,
                 candidate_rows: list[dict[str, Any]], candidate_out: Path,
                 receipt: dict[str, Any], receipt_out: Path) -> None:
    resolved = {path.resolve() for path in (current_out, candidate_out, receipt_out)}
    if len(resolved) != 3:
        raise DumpError("current, successor-candidate, and receipt paths must differ")
    temps: list[Path] = []
    try:
        current_temp = _write_temp(current_rows, current_out)
        temps.append(current_temp)
        candidate_temp = _write_temp(candidate_rows, candidate_out)
        temps.append(candidate_temp)
        full_receipt = {
            **receipt,
            "current_output_sha256": digest(current_temp),
            "successor_candidates_sha256": digest(candidate_temp),
        }
        receipt_out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                "w", dir=receipt_out.parent, prefix=receipt_out.name + ".",
                delete=False) as f:
            receipt_temp = Path(f.name)
            temps.append(receipt_temp)
            f.write(json.dumps(full_receipt, indent=2, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(current_temp, current_out); temps.remove(current_temp)
        os.replace(candidate_temp, candidate_out); temps.remove(candidate_temp)
        os.replace(receipt_temp, receipt_out); temps.remove(receipt_temp)
    finally:
        for path in temps:
            path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="aw_v4_sat")
    parser.add_argument("--staging-url", required=True)
    parser.add_argument("--staging-collection", action="append", required=True)
    parser.add_argument("--candidate-registry", type=Path, required=True)
    parser.add_argument("--expected-candidate-registry-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--successor-candidates-out", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument("--disc-class", default="news")
    parser.add_argument("--batch", type=int, default=4000)
    args = parser.parse_args()
    if args.batch < 1:
        raise SystemExit("FATAL news point dump: --batch must be positive")
    try:
        client = qdrant_client(
            args.qdrant_url, timeout=600, check_compatibility=False)
        staging = qdrant_client(
            args.staging_url, timeout=600, check_compatibility=False)
        staged_points = (
            point
            for name in args.staging_collection
            for point in _scroll(staging, name, None, args.batch)
        )
        current_rows, candidate_rows, receipt = collect_news_points(
            client, args.collection, args.disc_class, args.batch,
            args.candidate_registry, args.expected_candidate_registry_sha256,
            staged_points)
        atomic_write(
            current_rows, args.out, candidate_rows,
            args.successor_candidates_out, receipt, args.receipt_out)
    except DumpError as exc:
        raise SystemExit(f"FATAL news point dump: {exc}") from exc
    print(
        f"DONE {len(current_rows):,} current news points; "
        f"{len(candidate_rows):,} successor candidates -> {args.out}",
        file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
