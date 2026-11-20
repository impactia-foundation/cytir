#!/usr/bin/env python3
"""Seal an exhaustive portable ``aw.stage-receipt.v1`` data inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable

from cytir.acquire.artifact_io import atomic_publish, sha256_file


SCHEMA = "aw.stage-receipt.v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StageSealError(ValueError):
    """A stage cannot be represented as a portable exhaustive inventory."""


def _inside(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise StageSealError(f"artifact is outside portable root: {path}") from exc


def _published_path(path: Path, publish_from: Path | None,
                    publish_to: Path | None,
                    allow_existing_published: bool = False) -> Path:
    if publish_from is None and publish_to is None:
        return path.resolve()
    if publish_from is None or publish_to is None:
        raise StageSealError("publish_from and publish_to must be provided together")
    try:
        relative = path.resolve().relative_to(publish_from.resolve())
    except ValueError as exc:
        if allow_existing_published:
            return path.resolve()
        raise StageSealError(
            f"staged artifact is outside publish_from: {path}") from exc
    return publish_to.resolve() / relative


def seal(*, artifact_root: Path, artifact_paths: list[Path], receipt_path: Path,
         records: int, fmt: str, logical_sha256: str,
         producer_id: str, producer_version: str,
         publish_from: Path | None = None,
         publish_to: Path | None = None,
         allow_existing_published: bool = False) -> dict:
    if records < 0 or not fmt or not producer_id or not producer_version:
        raise StageSealError("records/format/producer values are invalid")
    if not SHA256.fullmatch(logical_sha256):
        raise StageSealError("logical SHA-256 must be lowercase 64-hex")
    if not artifact_paths:
        raise StageSealError("stage requires at least one explicit data artifact")
    root = artifact_root.resolve()
    receipt_relative = _inside(
        root, _published_path(receipt_path, publish_from, publish_to,
                              allow_existing_published))
    artifacts = []
    seen = set()
    for path in artifact_paths:
        resolved = path.resolve()
        relative = _inside(
            root, _published_path(resolved, publish_from, publish_to,
                                  allow_existing_published))
        if relative == receipt_relative:
            raise StageSealError("stage receipt cannot inventory itself")
        uri = f"artifact://{relative.as_posix()}"
        if uri in seen:
            raise StageSealError(f"duplicate artifact: {uri}")
        seen.add(uri)
        if not resolved.is_file():
            raise StageSealError(f"missing artifact: {resolved}")
        artifacts.append({"uri": uri, "bytes": resolved.stat().st_size,
                          "sha256": sha256_file(resolved)})
    artifacts.sort(key=lambda row: row["uri"])
    receipt = {
        "schema": SCHEMA,
        "records": records,
        "bytes": sum(row["bytes"] for row in artifacts),
        "format": fmt,
        "logical_sha256": logical_sha256,
        "deterministic_encoding": True,
        "producer": {"id": producer_id, "version": producer_version},
        "artifacts": artifacts,
    }
    data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    if receipt_path.exists():
        if hashlib.sha256(data).hexdigest() != sha256_file(receipt_path):
            raise StageSealError(f"immutable receipt collision: {receipt_path}")
    else:
        atomic_publish(receipt_path, data)
    receipt_artifact = {
        "uri": f"artifact://{receipt_relative.as_posix()}",
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
    }
    return {
        "records": records, "bytes": receipt["bytes"], "format": fmt,
        "producer": receipt["producer"], "logical_sha256": logical_sha256,
        "deterministic_encoding": True, "receipt": receipt_artifact,
    }


def seal_prevalidated(*, artifact_root: Path,
                      artifact_inventory: Iterable[dict], receipt_path: Path,
                      records: int, fmt: str, logical_sha256: str,
                      producer_id: str, producer_version: str) -> dict:
    """Seal a sorted inventory whose bytes were already fully validated.

    This entry point avoids hashing large artifacts a second time.  It remains
    fail-closed about URI portability, ordering, uniqueness, metadata shape,
    file presence, and byte size.  Callers must supply each row only after
    independently checking the file digest represented by that row.
    """
    if records < 0 or not fmt or not producer_id or not producer_version:
        raise StageSealError("records/format/producer values are invalid")
    if not SHA256.fullmatch(logical_sha256):
        raise StageSealError("logical SHA-256 must be lowercase 64-hex")
    root = artifact_root.resolve()
    receipt_relative = _inside(root, receipt_path.resolve())
    receipt_uri = f"artifact://{receipt_relative.as_posix()}"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    total_bytes = artifact_count = 0
    previous_uri: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
                dir=receipt_path.parent, delete=False) as target:
            temporary = Path(target.name)
            target.write(b'{\n  "artifacts": [')
            for artifact in artifact_inventory:
                if not isinstance(artifact, dict) or set(artifact) != {
                        "uri", "bytes", "sha256", "_stat"}:
                    raise StageSealError("prevalidated artifact row is invalid")
                uri = artifact["uri"]
                size = artifact["bytes"]
                digest = artifact["sha256"]
                fingerprint = artifact["_stat"]
                if (not isinstance(uri, str)
                        or not uri.startswith("artifact://")
                        or not isinstance(size, int) or isinstance(size, bool)
                        or size < 0 or not isinstance(digest, str)
                        or not SHA256.fullmatch(digest)
                        or not isinstance(fingerprint, list)
                        or len(fingerprint) != 5
                        or not all(isinstance(value, int)
                                   for value in fingerprint)
                        or fingerprint[2] != size):
                    raise StageSealError("prevalidated artifact metadata is invalid")
                relative = Path(uri.removeprefix("artifact://"))
                resolved = (root / relative).resolve()
                if (_inside(root, resolved).as_posix() != relative.as_posix()
                        or uri == receipt_uri):
                    raise StageSealError(f"unsafe prevalidated artifact URI: {uri}")
                if previous_uri is not None and uri <= previous_uri:
                    raise StageSealError(
                        "prevalidated inventory must be unique and URI-sorted")
                stat = resolved.stat()
                observed = [stat.st_dev, stat.st_ino, stat.st_size,
                            stat.st_mtime_ns, stat.st_ctime_ns]
                if not resolved.is_file() or observed != fingerprint:
                    raise StageSealError(
                        f"prevalidated artifact identity drift: {resolved}")
                previous_uri = uri
                total_bytes += size
                artifact_count += 1
                public = {key: artifact[key]
                          for key in ("uri", "bytes", "sha256")}
                encoded = json.dumps(
                    public, indent=2, sort_keys=True).replace("\n", "\n    ")
                target.write((",\n    " if artifact_count > 1 else "\n    ").encode())
                target.write(encoded.encode())
            if not artifact_count:
                raise StageSealError("stage requires at least one explicit data artifact")
            suffix = {
                "bytes": total_bytes,
                "deterministic_encoding": True,
                "format": fmt,
                "logical_sha256": logical_sha256,
                "producer": {"id": producer_id, "version": producer_version},
                "records": records,
                "schema": SCHEMA,
            }
            target.write(b"\n  ],\n")
            fields = json.dumps(suffix, indent=2, sort_keys=True)[2:-2]
            target.write((fields + "\n}\n").encode())
            target.flush()
            os.fsync(target.fileno())
        receipt_sha = sha256_file(temporary)
        receipt_bytes = temporary.stat().st_size
        try:
            os.link(temporary, receipt_path)
        except FileExistsError:
            if (receipt_path.stat().st_size != receipt_bytes
                    or sha256_file(receipt_path) != receipt_sha):
                raise StageSealError(
                    f"immutable receipt collision: {receipt_path}")
        temporary.unlink()
        temporary = None
        receipt_artifact = {
            "uri": receipt_uri, "bytes": receipt_bytes,
            "sha256": receipt_sha,
        }
        return {
            "records": records, "bytes": total_bytes, "format": fmt,
            "producer": {"id": producer_id, "version": producer_version},
            "logical_sha256": logical_sha256,
            "deterministic_encoding": True, "receipt": receipt_artifact,
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--format", required=True)
    parser.add_argument("--logical-sha256", required=True)
    parser.add_argument("--producer-id", required=True)
    parser.add_argument("--producer-version", required=True)
    args = parser.parse_args()
    try:
        result = seal(
            artifact_root=args.artifact_root, artifact_paths=args.artifact,
            receipt_path=args.receipt, records=args.records, fmt=args.format,
            logical_sha256=args.logical_sha256, producer_id=args.producer_id,
            producer_version=args.producer_version)
    except (OSError, StageSealError) as exc:
        print(f"FATAL: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
