"""Shared pieces of the per-family profile audits.

A profile audit replays one sealed pack against its family contract. Every audit
reads stage rows through `preservation_contract._validate_stage`, checks the
chunks against their parents, turns the profile's requirements into PASS rows
and publishes two immutable evidence files. Those four steps live here; what a
family's parent rows must say stays in the family's own audit module.
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
import uuid

from cytir.acquire.artifact_io import atomic_publish
from cytir.index.preservation_contract import _validate_stage

MAX_EMBED_TOKENS = 2560


def stage_rows(root: Path, stage: dict, *, single_shard: bool = False) -> list[dict]:
    """Validate a stage deeply, then return every JSONL row of its artifacts.

    `single_shard` refuses a stage that holds anything but exactly one JSONL shard.
    """
    receipt = _validate_stage(stage, "rows", artifact_root=root, deep=True)
    shards = [artifact["uri"] for artifact in receipt["artifacts"]
              if artifact["uri"].endswith(".jsonl.gz")]
    if single_shard and len(shards) != 1:
        raise ValueError("expected exactly one canonical JSONL shard")
    rows: list[dict] = []
    for uri in shards:
        with gzip.open(root / uri.removeprefix("artifact://"),
                       "rt", encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    return rows


def validate_run_stages(run: dict, artifact_root: Path) -> None:
    """Deep-validate every stage receipt a parent RUN record lists."""
    for name, stage in run["stages"].items():
        _validate_stage(stage, f"parent.{name}", artifact_root=artifact_root, deep=True)


def contiguous(ordinals: Iterable[list]) -> bool:
    """True when every list is exactly 0..n-1 once sorted."""
    return all(sorted(values) == list(range(len(values))) for values in ordinals)


def pack_gate(pack: dict, *, profile: str, documents: int, label: str) -> dict:
    """Require a ready pack of `documents` parents whose tokens fit the embed limit."""
    distribution = pack.get("token_distribution") or {}
    if (pack.get("status") != "ready_for_gpu_queue"
            or pack.get("source_profile") != profile
            or (pack.get("counts") or {}).get("documents") != documents
            or distribution.get("over_hard_max") != 0
            or distribution.get("max", 10**9) > MAX_EMBED_TOKENS):
        raise ValueError(f"{label} PACK gate failed")
    return distribution


def check_outlinks(outlinks: list[dict], by_id: dict[str, dict], *, label: str,
                   bind_source_url: bool, exclusive: bool) -> tuple[set, set]:
    """Each outlink row keeps its parent's identity; return (linked, empty) parent ids.

    A non-empty row needs a target and context; an empty row carries no relation
    data. Every parent must appear, with contiguous row ordinals.
    """
    linked: set = set()
    empty: set = set()
    ordinals: dict[str, list] = defaultdict(list)
    for row in outlinks:
        parent = by_id.get(str(row.get("document_id") or ""))
        if (parent is None or row.get("schema") != "aw.parent-source-link.v1"
                or row.get("source_record_id") != parent["source_record_id"]
                or (bind_source_url and row.get("source_url") != parent["url"])
                or row.get("parent_markdown_sha256") != parent["markdown_sha256"]
                or row.get("parent_identity") != parent["parent_identity"]
                or row.get("source_version") != parent["source_version"]):
            raise ValueError(f"{label} outlink lost parent identity")
        ordinals[parent["document_id"]].append(row.get("ordinal"))
        if row.get("is_empty"):
            if any(row.get(key) is not None
                   for key in ("target_url", "relation", "anchor_or_context")):
                raise ValueError(f"{label} empty outlink has relation data")
            empty.add(parent["document_id"])
        else:
            if not row.get("target_url") or not row.get("anchor_or_context"):
                raise ValueError(f"{label} outlink lacks target/context")
            linked.add(parent["document_id"])
    if (linked | empty != set(by_id) or (exclusive and linked & empty)
            or not contiguous(ordinals.values())):
        raise ValueError(f"{label} outlink conservation failed")
    return linked, empty


def check_chunks(chunks: list[dict], by_id: dict[str, dict], *, profile: str,
                 label: str, bind_url: bool) -> set:
    """Each chunk has a unique UUID point id and lineage to one parent; return the ids."""
    point_ids: set = set()
    ordinals: dict[str, list] = defaultdict(list)
    for row in chunks:
        point_id = str(row.get("point_id") or "")
        uuid.UUID(point_id)
        parent = by_id.get(str(row.get("document_id") or ""))
        if (point_id in point_ids or parent is None
                or row.get("source_profile") != profile
                or (bind_url and row.get("url") != parent["url"])
                or row.get("source_record_id") != parent["source_record_id"]
                or row.get("parent_markdown_sha256") != parent["markdown_sha256"]
                or row.get("parent_identity") != parent["parent_identity"]):
            raise ValueError(f"{label} chunk lineage failed")
        point_ids.add(point_id)
        ordinals[parent["document_id"]].append(row.get("chunk_ordinal"))
    if set(ordinals) != set(by_id) or not contiguous(ordinals.values()):
        raise ValueError(f"{label} chunk ordinal conservation failed")
    return point_ids


def profile_checks(profile: dict, methods: dict[str, list[str]], *,
                   label: str) -> dict[str, list[dict[str, Any]]]:
    """Pair each profile requirement with the method that verified it (all PASS)."""
    checks: dict[str, list[dict[str, Any]]] = {}
    for stage in profile["required_stages"]:
        requirements = profile.get(stage, {}).get("requirements", [])
        if len(requirements) != len(methods[stage]):
            raise ValueError(f"{label} profile method count mismatch: {stage}")
        checks[stage] = [{"requirement": requirement, "status": "PASS",
                          "verification_method": method}
                         for requirement, method in zip(requirements, methods[stage])]
    return checks


def publish_evidence(output_root: Path, evidence: dict, checks: dict, *,
                     label: str) -> tuple[Path, Path]:
    """Write the evidence and checks files once; an existing different file is a refusal."""
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_path = output_root / "verification-evidence.json"
    checks_path = output_root / "checks.json"
    for path, value in ((evidence_path, evidence), (checks_path, checks)):
        encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        if path.exists() and path.read_bytes() != encoded:
            raise ValueError(f"immutable {label} audit collision: {path}")
        if not path.exists():
            atomic_publish(path, encoded)
    return evidence_path, checks_path
