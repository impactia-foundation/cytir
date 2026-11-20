#!/usr/bin/env python3
"""Publish immutable global-root metadata wrappers for a sealed embedding pack."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


from cytir.acquire.artifact_io import atomic_publish


VERSION = "embedding-pack-global-lineage.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def global_ref(root: Path, path: Path) -> dict:
    relative = path.resolve().relative_to(root.resolve())
    return {"uri": "artifact://" + relative.as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def relative_uri(uri: str) -> Path:
    if not uri.startswith("artifact://"):
        raise ValueError("input artifact URI is invalid")
    relative = Path(uri.removeprefix("artifact://"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("input artifact URI is unsafe")
    return relative


def referenced_path(*, artifact_root: Path, pack: Path, ref: dict) -> Path:
    """Resolve legacy pack-local or current global-root refs by exact bytes/SHA."""
    relative = relative_uri(str(ref.get("uri") or ""))
    candidates = []
    for candidate in (artifact_root / relative, pack / relative):
        candidate = candidate.resolve()
        if candidate in candidates:
            continue
        if (candidate.is_file() and candidate.stat().st_size == ref.get("bytes")
                and sha256_file(candidate) == ref.get("sha256")):
            candidates.append(candidate)
    if len(candidates) != 1:
        raise ValueError(
            f"artifact reference must resolve exactly once by bytes/SHA: {ref.get('uri')}")
    return candidates[0]


def publish(*, artifact_root: Path, pack: Path) -> tuple[Path, Path]:
    artifact_root, pack = artifact_root.resolve(), pack.resolve()
    pack.relative_to(artifact_root)
    source_pack_path = pack / "PACK.json"
    source_pack = json.loads(source_pack_path.read_text())
    source_receipt_ref = (source_pack.get("chunks") or {}).get("receipt") or {}
    source_receipt_path = referenced_path(
        artifact_root=artifact_root, pack=pack, ref=source_receipt_ref)
    source_receipt = json.loads(source_receipt_path.read_text())
    if source_receipt.get("schema") != "aw.stage-receipt.v1":
        raise ValueError("source chunks do not use aw.stage-receipt.v1")
    portable_receipt = copy.deepcopy(source_receipt)
    byte_sum = 0
    seen = set()
    portable_artifacts = []
    for index, row in enumerate(source_receipt.get("artifacts") or []):
        path = referenced_path(artifact_root=artifact_root, pack=pack, ref=row)
        ref = global_ref(artifact_root, path)
        if ref["uri"] in seen:
            raise ValueError("duplicate portable chunk artifact URI")
        seen.add(ref["uri"]); byte_sum += ref["bytes"]; portable_artifacts.append(ref)
    if byte_sum != source_receipt.get("bytes"):
        raise ValueError("source chunk artifact byte sum mismatch")
    portable_receipt["artifacts"] = portable_artifacts
    portable_receipt["lineage_wrapper"] = {
        "producer": {"id": "tasks/417-scripts/portableize_embedding_pack.py",
                     "version": VERSION},
        "supersedes": global_ref(artifact_root, source_receipt_path),
    }
    portable_receipt_path = pack / "chunks/receipt.portable.v2.json"
    receipt_bytes = (json.dumps(portable_receipt, indent=2, sort_keys=True) + "\n").encode()
    if portable_receipt_path.exists() and portable_receipt_path.read_bytes() != receipt_bytes:
        raise ValueError("immutable portable chunk receipt collision")
    if not portable_receipt_path.exists():
        atomic_publish(portable_receipt_path, receipt_bytes)

    portable_pack = copy.deepcopy(source_pack)
    portable_pack["chunks"]["receipt"] = global_ref(
        artifact_root, portable_receipt_path)
    portable_pack["portable_lineage"] = {
        "schema": "aw.embedding-pack-portable-lineage.v1",
        "producer": {"id": "tasks/417-scripts/portableize_embedding_pack.py",
                     "version": VERSION},
        "supersedes": global_ref(artifact_root, source_pack_path),
        "chunk_receipt_supersedes": global_ref(artifact_root, source_receipt_path),
    }
    portable_pack_path = pack / "PACK.portable.v2.json"
    pack_bytes = (json.dumps(portable_pack, indent=2, sort_keys=True) + "\n").encode()
    if portable_pack_path.exists() and portable_pack_path.read_bytes() != pack_bytes:
        raise ValueError("immutable portable PACK collision")
    if not portable_pack_path.exists():
        atomic_publish(portable_pack_path, pack_bytes)
    return portable_pack_path, portable_receipt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    args = parser.parse_args()
    pack, receipt = publish(artifact_root=args.artifact_root, pack=args.pack)
    print(json.dumps({"pack": str(pack), "pack_sha256": sha256_file(pack),
                      "chunk_receipt": str(receipt),
                      "chunk_receipt_sha256": sha256_file(receipt)},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
