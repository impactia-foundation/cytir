#!/usr/bin/env python3
"""Adapt a sealed atomic embedding pack to the verified 413A GPU contract.

The adapter does not rewrite embedding text.  It deep-verifies the pack and
its exhaustive chunk receipt, hard-links the already sealed text shards into
an immutable stage directory, and writes the per-shard manifests required by
the 413A verifier.  This makes composition compatibility explicit while
keeping one byte-identical text artifact.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path


MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
REVISION = "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def artifact_path(pack: Path, uri: str) -> Path:
    if not uri.startswith("artifact://"):
        raise ValueError(f"non-portable artifact URI: {uri!r}")
    relative = Path(uri.removeprefix("artifact://"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact URI: {uri!r}")
    return pack / relative


def count_text_rows(path: Path, structured_legend: dict | None = None,
                    seen_point_ids: set[str] | None = None) -> tuple[int, set[str]]:
    rows = 0
    seen = seen_point_ids if seen_point_ids is not None else set()
    provenance_ids: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            row = json.loads(line)
            point_id = str(row.get("point_id") or "")
            text = str(row.get("text") or "")
            if not point_id or not text.strip():
                raise ValueError(f"{path.name}:{line_no} lacks point_id or text")
            try:
                uuid.UUID(point_id)
                valid_point_id = True
            except ValueError:
                try:
                    numeric = int(point_id)
                    valid_point_id = str(numeric) == point_id and 0 <= numeric < 2 ** 64
                except ValueError:
                    valid_point_id = False
            if not valid_point_id:
                raise ValueError(
                    f"{path.name}:{line_no} point_id is not a Qdrant UUID/uint64: "
                    f"{point_id!r}")
            if point_id in seen:
                raise ValueError(f"duplicate point_id in {path.name}: {point_id}")
            if structured_legend is not None:
                prov_id = str(row.get("prov_id") or "")
                if prov_id not in structured_legend:
                    raise ValueError(
                        f"{path.name}:{line_no} has unbound structured provenance: {prov_id!r}")
                if not all(isinstance(row.get(key), str) and row[key].strip()
                           for key in ("canonical_text", "sparse_text", "text")):
                    raise ValueError(
                        f"{path.name}:{line_no} lacks structured text channels")
                provenance_ids.add(prov_id)
            seen.add(point_id)
            rows += 1
    return rows, provenance_ids


def adapt(pack_arg: str, output_arg: str, artifact_root_arg: str,
          pack_manifest_arg: str = "PACK.json") -> dict:
    artifact_root = Path(artifact_root_arg).resolve()
    pack = Path(pack_arg).resolve()
    output = Path(output_arg).resolve()
    if not pack.is_relative_to(artifact_root) or not output.is_relative_to(artifact_root):
        raise ValueError("pack and output must be below the global artifact root")

    def global_uri(path: Path) -> str:
        return "artifact://" + path.resolve().relative_to(artifact_root).as_posix()

    pack_path = pack / pack_manifest_arg
    input_is_global = pack_path.name != "PACK.json"
    pack_doc = load_json(pack_path)
    pack_schema = pack_doc.get("schema")
    if (pack_schema not in {"aw.atomic-embedding-pack.v1",
                            "aw.structured-embedding-pack.v1"}
            or pack_doc.get("status") != "ready_for_gpu_queue"):
        raise ValueError("pack is not a terminal supported embedding pack")
    embedding = pack_doc.get("embedding") or {}
    if (embedding.get("model"), embedding.get("revision"),
            embedding.get("dimensions"), embedding.get("dtype"),
            embedding.get("encoding")) != (
                MODEL, REVISION, 1024, "float16", "row-major little-endian binary"):
        raise ValueError("pack embedding contract is incompatible with 413A")

    chunks = pack_doc.get("chunks") or {}
    receipt_ref = chunks.get("receipt") or {}
    receipt_path = artifact_path(
        artifact_root if input_is_global else pack,
        str(receipt_ref.get("uri") or ""))
    if (receipt_path.stat().st_size != receipt_ref.get("bytes")
            or sha256_file(receipt_path) != receipt_ref.get("sha256")):
        raise ValueError("pack chunk receipt bytes/SHA mismatch")
    receipt = load_json(receipt_path)
    if receipt.get("schema") != "aw.stage-receipt.v1":
        raise ValueError("chunk receipt is not aw.stage-receipt.v1")
    for key in ("records", "bytes", "format", "logical_sha256",
                "deterministic_encoding", "producer"):
        if receipt.get(key) != chunks.get(key):
            raise ValueError(f"PACK/chunk receipt mismatch for {key}")
    if receipt.get("deterministic_encoding") is not True:
        raise ValueError("chunk receipt is not deterministic")

    all_artifacts = receipt.get("artifacts")
    if not isinstance(all_artifacts, list) or not all_artifacts:
        raise ValueError("chunk receipt has no artifact inventory")
    seen_uris: set[str] = set()
    byte_sum = 0
    text_artifacts = []
    for row in all_artifacts:
        uri = str(row.get("uri") or "")
        if uri in seen_uris:
            raise ValueError(f"duplicate artifact URI: {uri}")
        seen_uris.add(uri)
        path = artifact_path(artifact_root if input_is_global else pack, uri)
        if path.stat().st_size != row.get("bytes") or sha256_file(path) != row.get("sha256"):
            raise ValueError(f"chunk artifact bytes/SHA mismatch: {uri}")
        byte_sum += path.stat().st_size
        if path.name.startswith("shard-") and path.name.endswith(".texts.jsonl.gz"):
            text_artifacts.append((path, row))
    if byte_sum != receipt.get("bytes"):
        raise ValueError("chunk artifact byte sum differs from receipt")
    if not text_artifacts:
        raise ValueError("pack contains no embedding text shards")

    chunk_policy = pack_doc.get("chunk_policy") or {}
    policy_path_value = str(chunk_policy.get("path") or "")
    if policy_path_value.startswith("artifact://"):
        policy_path_value = policy_path_value.removeprefix("artifact://")
    policy_path = Path(policy_path_value)
    if (not policy_path_value or policy_path.is_absolute()
            or ".." in policy_path.parts):
        raise ValueError(
            "chunk_policy.path must be a safe repo-relative or artifact URI")
    structured_legend = None
    composition = "precomposed-atomic-markdown-v1"
    if pack_schema == "aw.structured-embedding-pack.v1":
        legend_rows = [row for row in all_artifacts
                       if str(row.get("uri") or "").endswith("/provenance-legend.json")]
        if len(legend_rows) != 1:
            raise ValueError("structured pack must bind exactly one provenance legend")
        if (not chunk_policy.get("tokenizer_class")
                or chunk_policy.get("tokenizer_is_fast") is not True
                or not chunk_policy.get("tokenizer_revision")):
            raise ValueError("structured pack tokenizer provenance is incomplete")
        legend_path = artifact_path(
            artifact_root if input_is_global else pack, legend_rows[0]["uri"])
        structured_legend = load_json(legend_path)
        if not structured_legend:
            raise ValueError("structured provenance legend is empty")
        composition = "precomposed-structured-dense-channel-v1"

    if output.exists():
        raise ValueError(f"refusing existing output: {output}")
    partial = output.with_name(output.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    try:
        rows = 0
        seen_point_ids: set[str] = set()
        stage_artifacts = []
        for shard_index, (source_path, artifact) in enumerate(
                sorted(text_artifacts)):
            # Global portable PACKs may conserve shards from several immutable
            # worker roots.  Their local basenames legitimately repeat (each
            # worker starts at shard-0000), so flattening by source basename
            # causes collisions.  Canonical cross-root ordering plus fresh
            # contiguous stage names preserves legacy one-root semantics and
            # gives every source shard exactly one destination.
            destination = partial / f"shard-{shard_index:04d}.texts.jsonl.gz"
            os.link(source_path, destination)
            shard_rows, provenance_ids = count_text_rows(
                destination, structured_legend, seen_point_ids)
            rows += shard_rows
            tag = destination.name.removesuffix(".texts.jsonl.gz")
            stage_manifest = {
                "schema": "aw.413a.stage.v1",
                "shard": tag,
                "rows": shard_rows,
                "source_rows": shard_rows,
                "quarantined_rows": 0,
                "quarantine": [],
                "texts_sha256": artifact["sha256"],
                "records_sha256": receipt["logical_sha256"],
                "records_are_the_source": True,
                "source": {"path": global_uri(receipt_path),
                           "sha256": receipt_ref["sha256"]},
                "source_text_artifact": {
                    "uri": global_uri(source_path),
                    "bytes": artifact["bytes"],
                    "sha256": artifact["sha256"],
                },
                "chunk_policy": {
                    "id": chunk_policy.get("policy_id"),
                    "sha256": chunk_policy.get("sha256"),
                },
                "composition": "precomposed-atomic-markdown-v1",
            }
            stage_manifest["composition"] = composition
            stage_manifest["embedding_channel"] = "text"
            if structured_legend is not None:
                for prov_id in provenance_ids:
                    provenance = structured_legend[prov_id]
                    channels = provenance.get("channels") or {}
                    chunker = provenance.get("chunker") or {}
                    if (provenance.get("family") != "generic-structured"
                            or channels.get("channels_split") is not True
                            or not channels.get("dense")
                            or chunker.get("policy_sha256") != chunk_policy.get("sha256")
                            or chunker.get("policy_path") != chunk_policy.get("path")
                            or chunker.get("chunk_tokenizer_revision")
                            != chunk_policy.get("tokenizer_revision")):
                        raise ValueError(
                            f"structured provenance/policy mismatch: {prov_id}")
                stage_manifest["channel_provenance"] = {
                    "legend_uri": global_uri(legend_path),
                    "legend_sha256": legend_rows[0]["sha256"],
                    "prov_ids": sorted(provenance_ids),
                    "dense_field": "text",
                    "sparse_field": "sparse_text",
                    "canonical_field": "canonical_text",
                    "tokenizer_class": chunk_policy.get("tokenizer_class"),
                    "tokenizer_is_fast": chunk_policy.get("tokenizer_is_fast"),
                    "tokenizer_revision": chunk_policy.get("tokenizer_revision"),
                }
            manifest_path = partial / f"{tag}.stage.json"
            manifest_path.write_text(
                json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n")
            stage_artifacts.append({
                "uri": global_uri(output / destination.name),
                "bytes": destination.stat().st_size,
                "sha256": artifact["sha256"],
            })
        if rows != receipt.get("records"):
            raise ValueError(
                f"text row count={rows} differs from receipt={receipt.get('records')}")
        terminal = {
            "schema": "aw.stage-receipt.v1",
            "status": "complete",
            "records": rows,
            "rows": rows,
            "bytes": sum(row["bytes"] for row in stage_artifacts),
            "format": "413a-point-text-jsonl+deterministic-gzip",
            "logical_sha256": receipt["logical_sha256"],
            "deterministic_encoding": True,
            "producer": {"id": "tasks/417-scripts/adapt_pack_to_413a_stage.py",
                         "version": "v3-recursive-global-artifact-root"},
            "pack": {"uri": global_uri(pack_path), "bytes": pack_path.stat().st_size,
                     "sha256": sha256_file(pack_path)},
            "chunk_receipt": {"uri": global_uri(receipt_path),
                              "bytes": receipt_ref["bytes"],
                              "sha256": receipt_ref["sha256"]},
            "artifacts": stage_artifacts,
            "pack_schema": pack_schema,
            "embedding_channel": "text",
            "composition": composition,
        }
        (partial / "STAGE-RECEIPT.json").write_text(
            json.dumps(terminal, indent=2, sort_keys=True) + "\n")
        os.rename(partial, output)
        return terminal
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--pack-manifest", default="PACK.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(adapt(args.pack, args.output, args.artifact_root,
                           args.pack_manifest),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
