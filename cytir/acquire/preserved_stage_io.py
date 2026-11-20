"""Shared content-addressed I/O for preserved non-L0 pipeline stages."""
from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

from cytir.acquire.gzip_pinned import (  # noqa: F401  re-exports
    GZIP_OS_UNKNOWN, deterministic_gzip, deterministic_gzip_open)


class PreservedArtifactError(RuntimeError):
    """A preserved artifact is missing, unsafe, or divergent."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as target:
        target.write(value)
        target.flush()
        os.fsync(target.fileno())
        temporary = Path(target.name)
    os.replace(temporary, path)




def store_artifact(root: Path, family: str, suffix: str,
                   value: bytes) -> dict[str, Any]:
    """Store deterministic gzip bytes at a path addressed by the raw digest."""
    body_sha = sha256_bytes(value)
    relative = PurePosixPath(family) / body_sha[:2] / f"{body_sha}.{suffix}.gz"
    path = root.joinpath(*relative.parts)
    compressed = deterministic_gzip(value)
    expected_sha = sha256_bytes(compressed)
    if path.exists():
        existing = path.read_bytes()
        if len(existing) != len(compressed) or sha256_bytes(existing) != expected_sha:
            raise PreservedArtifactError(
                f"existing artifact disagrees with content: {path}")
    else:
        atomic_write(path, compressed)
    return {
        "uri": f"artifact://{relative.as_posix()}",
        "compression": "gzip", "bytes": len(compressed),
        "sha256": expected_sha, "body_bytes": len(value),
        "body_sha256": body_sha,
    }


def artifact_path(root: Path, artifact: dict[str, Any]) -> Path:
    uri = str(artifact.get("uri", ""))
    if not uri.startswith("artifact://"):
        raise PreservedArtifactError(f"invalid artifact URI: {uri!r}")
    relative = PurePosixPath(uri.removeprefix("artifact://"))
    if relative.is_absolute() or ".." in relative.parts:
        raise PreservedArtifactError(f"unsafe artifact URI: {uri!r}")
    return root.joinpath(*relative.parts)


def load_validated_artifact(root: Path, artifact: dict[str, Any]) -> bytes:
    """Return raw bytes only after validating compressed and body identity."""
    path = artifact_path(root, artifact)
    if not path.is_file():
        raise PreservedArtifactError(f"missing artifact: {path}")
    compressed = path.read_bytes()
    if len(compressed) != artifact.get("bytes"):
        raise PreservedArtifactError(f"artifact byte mismatch: {path}")
    if sha256_bytes(compressed) != artifact.get("sha256"):
        raise PreservedArtifactError(f"artifact digest mismatch: {path}")
    try:
        raw = gzip.decompress(compressed)
    except (OSError, gzip.BadGzipFile) as exc:
        raise PreservedArtifactError(f"artifact compression invalid: {path}") from exc
    if (len(raw) != artifact.get("body_bytes") or
            sha256_bytes(raw) != artifact.get("body_sha256")):
        raise PreservedArtifactError(f"artifact body mismatch: {path}")
    return raw


def validate_artifact(root: Path, artifact: dict[str, Any]) -> None:
    load_validated_artifact(root, artifact)
