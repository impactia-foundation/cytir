"""Deterministic, streaming JSONL artifact transport helpers.

Artifact ``sha256`` values deliberately remain hashes of the bytes copied
between machines.  ``.gz`` is an optional transport representation, never a
new record-identity scheme: callers keep point/content hashes exactly as they
were and bind both compressed bytes and the decompressed byte stream in their
manifest when compression is selected.
"""
from __future__ import annotations

import gzip
from cytir.acquire.gzip_pinned import deterministic_gzip_open
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, TextIO

# One canonical-JSON encoding in the package (byte-identical to the former local
# copy: sort_keys, compact separators, ensure_ascii=False). Re-exported: callers
# of `artifact_io.canonical_json` keep working.
from cytir.chunk.provenance import canonical_json  # noqa: F401


GZIP = {"algorithm": "gzip", "mtime": 0, "filename": ""}


def is_gzip(path: Path) -> bool:
    return path.name.endswith(".gz")


def open_text_read(path: Path) -> TextIO:
    return gzip.open(path, "rt", encoding="utf-8", newline="") if is_gzip(path) \
        else path.open("r", encoding="utf-8", newline="")


def open_text_write(path: Path, *, gzip_transport: bool) -> TextIO:
    """Open a new UTF-8 artifact; gzip headers are reproducible across runs."""
    raw = path.open("xb")
    if not gzip_transport:
        return io.TextIOWrapper(raw, encoding="utf-8", newline="\n")
    compressed = deterministic_gzip_open(raw, compresslevel=9)
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_publish(path: Path, data: bytes) -> None:
    """Publish a new immutable artifact without replacing an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_jsonl_or_verify(path: Path, rows: Iterable[dict]) -> tuple[str, int]:
    """Stream canonical JSONL and reject an immutable-path content collision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest, count = hashlib.sha256(), 0
    with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            for row in rows:
                data = (canonical_json(row) + "\n").encode("utf-8")
                handle.write(data)
                digest.update(data)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        hexdigest = digest.hexdigest()
        if path.exists():
            if sha256_file(path) != hexdigest:
                raise ValueError(f"immutable artifact collision: {path}")
        else:
            os.link(temporary, path)
        return hexdigest, count
    finally:
        temporary.unlink(missing_ok=True)


def uncompressed_info(path: Path) -> dict[str, int | str]:
    """Hash/count logical JSONL bytes without materializing decompression."""
    digest = hashlib.sha256()
    total = 0
    opener = gzip.open if is_gzip(path) else open
    with opener(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
            total += len(block)
    return {"sha256": digest.hexdigest(), "bytes": total}


def close_attempt(*, prefix: Path, successor_prefix: Path, url: str) -> dict:
    """Immutably close a signal-interrupted range attempt from its successor boundary.

    Ported from the former `close_interrupted_http_attempt` script (1235).
    """
    started = Path(str(prefix) + ".started"); start_offset = Path(str(prefix) + ".start_offset")
    headers = Path(str(prefix) + ".headers"); metrics = Path(str(prefix) + ".metrics")
    successor_started = Path(str(successor_prefix) + ".started")
    successor_offset = Path(str(successor_prefix) + ".start_offset")
    if not all(path.is_file() for path in (started, start_offset, headers, metrics,
                                            successor_started, successor_offset)):
        raise ValueError("interrupted/successor boundary evidence is incomplete")
    if metrics.stat().st_size != 0:
        raise ValueError("interrupted-attempt recovery only accepts exact empty curl metrics")
    start = int(start_offset.read_text()); end = int(successor_offset.read_text())
    if end <= start:
        raise ValueError("successor boundary does not advance interrupted attempt")
    completed = Path(str(prefix) + ".completed"); rc = Path(str(prefix) + ".rc")
    end_offset = Path(str(prefix) + ".end_offset"); recovery = Path(str(prefix) + ".recovery.json")
    if any(path.exists() for path in (completed, rc, end_offset, recovery)):
        raise ValueError("interrupted-attempt recovery output already exists")
    completed_at = successor_started.read_text().strip()
    value = {"schema": "aw.interrupted-http-attempt-recovery.v1",
        "status": "closed_from_successor_resume_boundary", "return_code": 143,
        "http_status": 200, "final_url": url, "start_offset": start, "end_offset": end,
        "started_at": started.read_text().strip(), "completed_at": completed_at,
        "method": "successor exact start_offset proves prior partial end; original empty metrics retained",
        "evidence": {"headers_sha256": sha256_file(headers),
                     "empty_metrics_sha256": sha256_file(metrics),
                     "successor_started_sha256": sha256_file(successor_started),
                     "successor_start_offset_sha256": sha256_file(successor_offset)}}
    atomic_publish(completed, (completed_at + "\n").encode())
    atomic_publish(rc, b"143\n"); atomic_publish(end_offset, f"{end}\n".encode())
    atomic_publish(recovery, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
    return value
