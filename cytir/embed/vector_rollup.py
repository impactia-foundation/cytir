#!/usr/bin/env python3
"""Seal verified FP16 shard artifacts into a terminal portable receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _artifact_binding(path: Path, artifact_root: Path) -> dict:
    try:
        relative = path.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError(f"artifact is outside --artifact-root: {path}") from exc
    return {
        "uri": f"artifact://{relative.as_posix()}",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _publish_bound_file(source: Path, vectors: Path, relative: Path,
                        artifact_root: Path) -> dict:
    """Publish one immutable receipt dependency below the vector root.

    ``artifact://`` is deliberately global-artifact-root relative.  A rollup
    must not point at a stage/work directory that will be absent after copy.
    Existing bytes are accepted only when they are identical.
    """
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe portable artifact path: {relative}")
    payload_sha = sha256_file(source)
    payload_bytes = source.stat().st_size
    destination = vectors / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (destination.stat().st_size != payload_bytes or
                sha256_file(destination) != payload_sha):
            raise ValueError(f"portable artifact collision: {destination}")
    else:
        partial = destination.with_name(
            destination.name + f".partial.{os.getpid()}")
        shutil.copyfile(source, partial)
        if (partial.stat().st_size != payload_bytes or
                sha256_file(partial) != payload_sha):
            partial.unlink(missing_ok=True)
            raise ValueError(f"portable artifact copy drift: {source}")
        os.rename(partial, destination)
    return _artifact_binding(destination, artifact_root)


def seal(args) -> dict:
    vectors = Path(args.vectors).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    try:
        vectors.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError("vectors must be below --artifact-root") from exc
    stage = Path(args.stage).resolve()
    output_name = getattr(args, "output_name", "RECEIPT.json")
    if (Path(output_name).name != output_name or
            output_name in {"", ".", ".."}):
        raise ValueError("output name must be one safe vector-root filename")
    output = vectors / output_name
    if output.exists():
        raise ValueError(f"refusing to overwrite terminal receipt: {output}")
    verification_path = Path(args.verification).resolve()
    verification = json.loads(verification_path.read_bytes())
    if verification.get("failed") != 0:
        raise ValueError("verification report contains failures")
    stage_receipt_path = stage / "STAGE-RECEIPT.json"
    stage_receipt = json.loads(stage_receipt_path.read_bytes())
    if stage_receipt.get("status") != "complete":
        raise ValueError("stage receipt is not complete")

    artifacts = []
    rows = 0
    vector_bytes = 0
    tags = sorted(path.name.removesuffix(".vectors.manifest.json")
                  for path in vectors.glob("shard-*.vectors.manifest.json"))
    if not tags:
        raise ValueError("no vector manifests found")
    for tag in tags:
        manifest_path = vectors / f"{tag}.vectors.manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        vector_path = vectors / f"{tag}.vectors.f16"
        ids_path = vectors / f"{tag}.row_ids.jsonl"
        n = manifest.get("n")
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"invalid row count in {manifest_path}")
        expected_bytes = n * args.dim * 2
        if vector_path.stat().st_size != expected_bytes:
            raise ValueError(f"vector byte-size mismatch: {vector_path}")
        vector_sha = sha256_file(vector_path)
        if vector_sha != manifest.get("sha256"):
            raise ValueError(f"vector SHA mismatch: {vector_path}")
        if sha256_file(ids_path) != manifest.get("row_ids_sha256"):
            raise ValueError(f"row-ids SHA mismatch: {ids_path}")
        artifacts.append({
            "shard": tag,
            "rows": n,
            "manifest": _artifact_binding(manifest_path, artifact_root),
            "row_ids": _artifact_binding(ids_path, artifact_root),
            "vectors": _artifact_binding(vector_path, artifact_root),
        })
        rows += n
        vector_bytes += expected_bytes
    if rows != stage_receipt.get("rows") or rows != verification.get("rows"):
        raise ValueError(
            f"row drift: shards={rows} stage={stage_receipt.get('rows')} "
            f"verification={verification.get('rows')}")
    if len(tags) != verification.get("shards"):
        raise ValueError("shard-count drift against verification")

    evidence_prefix = Path(getattr(args, "evidence_prefix", "rollup-evidence"))
    if (evidence_prefix.is_absolute() or ".." in evidence_prefix.parts
            or not evidence_prefix.parts):
        raise ValueError("evidence prefix must be a safe vector-root-relative path")
    stage_binding = _publish_bound_file(
        stage_receipt_path, vectors, evidence_prefix / "stage/STAGE-RECEIPT.json",
        artifact_root)
    verification_binding = _publish_bound_file(
        verification_path, vectors,
        evidence_prefix / f"verification/{verification_path.name}", artifact_root)

    receipt = {
        "schema": "aw.vector-rollup.v1",
        "status": "complete",
        "storage": {"dtype": "float16", "dimensions": args.dim,
                    "binary_transfer_ready": True},
        "rows": rows,
        "shards": len(tags),
        "vector_bytes": vector_bytes,
        "stage_receipt": stage_binding,
        "verification": {**verification_binding, "failed": 0},
        "artifacts": artifacts,
    }
    supersedes = getattr(args, "supersedes", "")
    if supersedes:
        previous = vectors / supersedes
        if (Path(supersedes).name != supersedes or not previous.is_file()):
            raise ValueError("superseded receipt must be a vector-root filename")
        receipt["supersedes"] = {
            "uri": _artifact_binding(previous, artifact_root)["uri"],
            "bytes": previous.stat().st_size,
            "sha256": sha256_file(previous),
        }
    partial = output.with_name(output.name + f".partial.{os.getpid()}")
    partial.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.rename(partial, output)
    return receipt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--artifact-root", required=True,
                    help="Global root against which every artifact:// URI resolves")
    ap.add_argument("--verification", required=True)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--output-name", default="RECEIPT.json")
    ap.add_argument("--supersedes", default="",
                    help="Existing vector-root receipt bound as superseded evidence")
    ap.add_argument("--evidence-prefix", default="rollup-evidence",
                    help="Fresh vector-root-relative namespace for bound evidence")
    args = ap.parse_args(argv)
    print(json.dumps(seal(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
