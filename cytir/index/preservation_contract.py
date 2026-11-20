#!/usr/bin/env python3
# Moved from the pipeline-era `preservation_contract` module (task 677, task 643 S5); the old path was an alias shim, deleted in S9.
"""Fail-closed source preservation and reproducible-build contract.

A publishable AW release must bind every source to a committed source profile
and to immutable receipts for:

    original bytes -> extracted Markdown -> chunks -> vectors -> collection

The profile owns source-specific semantics (for example, legal article
boundaries or entity-card evidence composition).  This module owns the shared
mechanical contract: stage presence, content-addressed receipts, count
conservation, off-box verification, and the collection membership receipt.

`artifact://path` is a portable logical URI.  A clone resolves it below an
explicit artifact root; sealed manifests may not depend on machine-local paths.
The normal publish gate validates the receipt chain cheaply.  `--deep` also
reads and hashes every referenced receipt from the artifact root.

STATUS `ruled` (ADR-098 amendment 1)
------------------------------------
A manifest is `sealed` when every clause is proven.  `ruled` is the one other
status this module will verify, and it exists because the founder can decide to
promote while named proofs are still owed -- as he did for the `aw_e8` flip on
2026-09-16.  It is deliberately narrow:

  * a `ruling` block names who decided, when, how, what ratified it, what the
    ruling does NOT cover, and ONE collection;
  * every carried gap names the manifest it comes from, the obligation, an
    owner, the vehicle that discharges it, and the contract clauses it
    suspends -- which may only be drawn from `SUSPENDABLE_CLAUSES`;
  * a `rollback` block names the incumbent that stays restorable, reconciles
    its membership receipt, and names the destructive step withheld;
  * everything the ruling does not carry is asserted exactly as under a seal,
    and a carried clause that is nevertheless supplied is validated in full;
  * the carried gaps print to stderr on EVERY verify, because `alias_swap.sh`
    sends the proof to /dev/null and a ruled PASS must never pass silently.

The effect is a status that can be granted only by writing down what is still
owed, and that says so out loud every time it is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "aw.preservation-manifest.v1"
PROFILE_SCHEMA = "aw.source-preservation-profiles.v1"
STAGE_RECEIPT_SCHEMA = "aw.stage-receipt.v1"
CONTENT_FIDELITY_RECEIPT_SCHEMA = "aw.content-fidelity-receipt.v1"
PROFILE_COMPLIANCE_RECEIPT_SCHEMA = "aw.source-profile-compliance-receipt.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
STAGE_EDGES = (
    ("original_to_markdown", "original", "markdown"),
    ("markdown_to_chunks", "markdown", "chunks"),
    ("chunks_to_vectors", "chunks", "vectors"),
)
POPULATION_BASIS = "same-source-document-population"
REFRESH_MODES = {"pinned-originals", "allow-newer"}
CONTENT_SCOPE = "admitted-original-and-principal-content"

SEALED = "sealed"
RULED = "ruled"
MEMBERSHIP_RECEIPT_SCHEMAS = (
    "aw.task416.qdrant-membership-receipt.v1",
    "aw.qdrant-membership-receipt.v1",
)
# The ONLY clauses of this contract a named ruling may carry as an open
# obligation instead of a proof.  Each is a clause whose evidence is exactly
# what one of `data/production-rebuild-lock.v1.json`'s blocking gaps says is
# missing.  A gap that suspends nothing is legal and normal: it is carried as
# an obligation that costs this contract nothing (G8 is such a gap).
#
# Everything NOT listed here is asserted under a ruling exactly as it is under
# a seal -- and a suspended clause that is nevertheless PRESENT is still
# validated in full.  A ruling lowers what must be supplied; it never lowers
# what a supplied block must prove.
SUSPENDABLE_CLAUSES = (
    "sources",                   # the per-source original->markdown->chunks->
                                 # vectors receipt chain, with it the
                                 # content-fidelity and profile-compliance
                                 # audits and the summed-vector reconciliation
    "population",                # the baseline source/document resolution
    "rebuild",                   # the ordered, runnable rebuild command list
    "collection.build_receipt",  # the build that produced the collection
)
RULING_FIELDS = ("id", "collection", "ruled_by", "ruled_on", "ruled_via",
                 "ratified_by", "adr", "cite", "does_not_cover")


class ContractError(ValueError):
    """The release is not publishable or reproducible."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} {path} must be a JSON object")
    return value


def _require(obj: dict[str, Any], key: str, typ: type, where: str) -> Any:
    value = obj.get(key)
    if not isinstance(value, typ):
        raise ContractError(f"{where}.{key} must be {typ.__name__}")
    return value


def _nonnegative(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{where} must be a non-negative integer")
    return value


def _portable_uri(uri: str, where: str) -> PurePosixPath:
    if not uri.startswith("artifact://"):
        raise ContractError(
            f"{where}.uri must use artifact://, not a host-local path: {uri!r}")
    rel = PurePosixPath(uri.removeprefix("artifact://"))
    if not rel.parts or rel.is_absolute() or ".." in rel.parts:
        raise ContractError(f"{where}.uri is not a safe artifact path: {uri!r}")
    return rel


def _validate_artifact(value: Any, where: str, *, artifact_root: Path | None,
                       deep: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be an artifact object")
    uri = _require(value, "uri", str, where)
    rel = _portable_uri(uri, where)
    size = _nonnegative(value.get("bytes"), f"{where}.bytes")
    digest = _require(value, "sha256", str, where)
    if not SHA256_RE.fullmatch(digest):
        raise ContractError(f"{where}.sha256 must be lowercase 64-hex")
    if deep:
        if artifact_root is None:
            raise ContractError("deep verification requires an artifact root")
        path = artifact_root.joinpath(*rel.parts)
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            raise ContractError(f"missing artifact {uri} at {path}: {exc}") from exc
        if actual_size != size:
            raise ContractError(
                f"artifact size mismatch for {uri}: manifest={size}, actual={actual_size}")
        actual_sha = sha256_file(path)
        if actual_sha != digest:
            raise ContractError(
                f"artifact SHA mismatch for {uri}: manifest={digest}, actual={actual_sha}")
    return {"uri": uri, "bytes": size, "sha256": digest}


def load_profiles(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 is not None:
        if not SHA256_RE.fullmatch(expected_sha256):
            raise ContractError("profile_registry.sha256 must be lowercase 64-hex")
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise ContractError(
                f"source-profile registry SHA mismatch: manifest={expected_sha256}, actual={actual}")
    registry = _load_object(path, "source-profile registry")
    if registry.get("schema") != PROFILE_SCHEMA:
        raise ContractError(
            f"source-profile registry schema must be {PROFILE_SCHEMA!r}")
    profiles = _require(registry, "profiles", dict, "profile_registry")
    if not profiles:
        raise ContractError("source-profile registry has no profiles")
    return profiles


def _validate_stage(stage: Any, where: str, *, artifact_root: Path | None,
                    deep: bool) -> dict[str, Any]:
    if not isinstance(stage, dict):
        raise ContractError(f"{where} must be an object")
    records = _nonnegative(stage.get("records"), f"{where}.records")
    size = _nonnegative(stage.get("bytes"), f"{where}.bytes")
    fmt = _require(stage, "format", str, where).strip()
    if not fmt:
        raise ContractError(f"{where}.format must not be empty")
    producer = _require(stage, "producer", dict, where)
    producer_id = _require(producer, "id", str, f"{where}.producer").strip()
    version = _require(producer, "version", str, f"{where}.producer").strip()
    if not producer_id or not version:
        raise ContractError(f"{where}.producer id/version must not be empty")
    logical_sha256 = _require(stage, "logical_sha256", str, where)
    if not SHA256_RE.fullmatch(logical_sha256):
        raise ContractError(f"{where}.logical_sha256 must be lowercase 64-hex")
    if stage.get("deterministic_encoding") is not True:
        raise ContractError(f"{where}.deterministic_encoding must be true")
    receipt = _validate_artifact(stage.get("receipt"), f"{where}.receipt",
                                 artifact_root=artifact_root, deep=deep)
    artifacts = None
    if deep:
        assert artifact_root is not None
        receipt_rel = _portable_uri(receipt["uri"], f"{where}.receipt")
        receipt_path = artifact_root.joinpath(*receipt_rel.parts)
        inventory = _load_object(receipt_path, f"{where} stage receipt")
        if inventory.get("schema") != STAGE_RECEIPT_SCHEMA:
            raise ContractError(
                f"{where}.receipt schema must be {STAGE_RECEIPT_SCHEMA!r}")
        for key, expected in (
                ("records", records), ("bytes", size), ("format", fmt),
                ("logical_sha256", logical_sha256),
                ("deterministic_encoding", True)):
            if inventory.get(key) != expected:
                raise ContractError(
                    f"{where}.receipt {key} mismatch: "
                    f"stage={expected!r}, receipt={inventory.get(key)!r}")
        receipt_producer = inventory.get("producer")
        expected_producer = {"id": producer_id, "version": version}
        if receipt_producer != expected_producer:
            raise ContractError(
                f"{where}.receipt producer mismatch: "
                f"stage={expected_producer!r}, receipt={receipt_producer!r}")
        raw_artifacts = inventory.get("artifacts")
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise ContractError(
                f"{where}.receipt artifacts must be a non-empty list")
        artifacts = []
        seen_uris: set[str] = set()
        artifact_bytes = 0
        for i, artifact in enumerate(raw_artifacts):
            validated = _validate_artifact(
                artifact, f"{where}.receipt.artifacts[{i}]",
                artifact_root=artifact_root, deep=True)
            if validated["uri"] in seen_uris:
                raise ContractError(
                    f"{where}.receipt has duplicate artifact URI "
                    f"{validated['uri']!r}")
            seen_uris.add(validated["uri"])
            artifact_bytes += validated["bytes"]
            artifacts.append(validated)
        if artifact_bytes != size:
            raise ContractError(
                f"{where}.receipt artifact bytes={artifact_bytes} "
                f"!= stage bytes={size}")
    return {"records": records, "bytes": size, "format": fmt,
            "producer": {"id": producer_id, "version": version},
            "logical_sha256": logical_sha256,
            "deterministic_encoding": True, "receipt": receipt,
            "artifacts": artifacts}


def _validate_edge(edge: Any, where: str, input_rows: int, output_rows: int,
                   *, artifact_root: Path | None, deep: bool) -> dict[str, Any]:
    if not isinstance(edge, dict):
        raise ContractError(f"{where} must be an object")
    inputs = _nonnegative(edge.get("input_records"), f"{where}.input_records")
    parents = _nonnegative(edge.get("emitted_parent_records"),
                           f"{where}.emitted_parent_records")
    quarantined = _nonnegative(edge.get("quarantined_records"),
                               f"{where}.quarantined_records")
    outputs = _nonnegative(edge.get("output_records"), f"{where}.output_records")
    if inputs != input_rows:
        raise ContractError(f"{where}: input_records={inputs} != input stage records={input_rows}")
    if outputs != output_rows:
        raise ContractError(f"{where}: output_records={outputs} != output stage records={output_rows}")
    if inputs != parents + quarantined:
        raise ContractError(
            f"{where}: conservation failed: {inputs} != {parents} emitted parents + "
            f"{quarantined} quarantined")
    quarantine_receipt = edge.get("quarantine_receipt")
    if quarantined and quarantine_receipt is None:
        raise ContractError(f"{where}: non-zero quarantine requires quarantine_receipt")
    if not quarantined and quarantine_receipt is not None:
        raise ContractError(f"{where}: zero quarantine must not claim quarantine_receipt")
    if quarantine_receipt is not None:
        _validate_artifact(quarantine_receipt, f"{where}.quarantine_receipt",
                           artifact_root=artifact_root, deep=deep)
    return {"input_records": inputs, "emitted_parent_records": parents,
            "quarantined_records": quarantined, "output_records": outputs}


def _validate_content_fidelity(value: Any, where: str, *, source_id: str,
                               stages: dict[str, dict[str, Any]],
                               artifact_root: Path | None,
                               deep: bool) -> dict[str, Any]:
    """Reject source stages that can hide a pre-chunk content cap.

    Stage hashes prove that bytes are immutable, but a perfectly hashed stage
    can still be a lossy prefix (for example ``body[:6000]``).  This separate
    receipt binds an adapter audit to the exact original and Markdown logical
    digests admitted by the manifest.  Chunk-size limits remain legal at L2;
    truncating L0 or L1 before the chunker does not.
    """
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be an object")
    scope = _require(value, "scope", str, where)
    if scope != CONTENT_SCOPE:
        raise ContractError(f"{where}.scope must be {CONTENT_SCOPE!r}")
    if value.get("original_bytes_complete") is not True:
        raise ContractError(f"{where}.original_bytes_complete must be true")
    if value.get("markdown_content_complete") is not True:
        raise ContractError(f"{where}.markdown_content_complete must be true")
    if value.get("source_text_truncation_applied") is not False:
        raise ContractError(
            f"{where}.source_text_truncation_applied must be false")
    receipt = _validate_artifact(
        value.get("receipt"), f"{where}.receipt",
        artifact_root=artifact_root, deep=deep)
    if deep:
        assert artifact_root is not None
        rel = _portable_uri(receipt["uri"], f"{where}.receipt")
        audit = _load_object(
            artifact_root.joinpath(*rel.parts), f"{where} receipt")
        if audit.get("schema") != CONTENT_FIDELITY_RECEIPT_SCHEMA:
            raise ContractError(
                f"{where}.receipt schema must be "
                f"{CONTENT_FIDELITY_RECEIPT_SCHEMA!r}")
        expected = {
            "source_id": source_id,
            "scope": scope,
            "original_bytes_complete": True,
            "markdown_content_complete": True,
            "source_text_truncation_applied": False,
            "original_logical_sha256": stages["original"]["logical_sha256"],
            "markdown_logical_sha256": stages["markdown"]["logical_sha256"],
        }
        for key, expected_value in expected.items():
            if audit.get(key) != expected_value:
                raise ContractError(
                    f"{where}.receipt {key} mismatch: "
                    f"manifest/stage={expected_value!r}, "
                    f"receipt={audit.get(key)!r}")
        adapter_sha = audit.get("adapter_source_sha256")
        if not isinstance(adapter_sha, str) or not SHA256_RE.fullmatch(adapter_sha):
            raise ContractError(
                f"{where}.receipt adapter_source_sha256 must be lowercase 64-hex")
        method = audit.get("verification_method")
        if not isinstance(method, str) or not method.strip():
            raise ContractError(
                f"{where}.receipt verification_method must not be empty")
        _validate_artifact(
            audit.get("verification_evidence"),
            f"{where}.receipt.verification_evidence",
            artifact_root=artifact_root, deep=True)
    return {
        "scope": scope,
        "original_bytes_complete": True,
        "markdown_content_complete": True,
        "source_text_truncation_applied": False,
        "receipt": receipt,
    }


def _validate_profile_compliance(value: Any, where: str, *, source_id: str,
                                 profile_id: str, profile: dict[str, Any],
                                 registry_sha256: str,
                                 artifact_root: Path | None,
                                 deep: bool) -> dict[str, Any]:
    """Bind every source-profile requirement to explicit verification evidence."""
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be an object")
    receipt = _validate_artifact(
        value.get("receipt"), f"{where}.receipt",
        artifact_root=artifact_root, deep=deep)
    if deep:
        assert artifact_root is not None
        rel = _portable_uri(receipt["uri"], f"{where}.receipt")
        audit = _load_object(
            artifact_root.joinpath(*rel.parts), f"{where} receipt")
        if audit.get("schema") != PROFILE_COMPLIANCE_RECEIPT_SCHEMA:
            raise ContractError(
                f"{where}.receipt schema must be "
                f"{PROFILE_COMPLIANCE_RECEIPT_SCHEMA!r}")
        for key, expected in (
                ("source_id", source_id), ("profile", profile_id),
                ("profile_registry_sha256", registry_sha256)):
            if audit.get(key) != expected:
                raise ContractError(
                    f"{where}.receipt {key} mismatch: expected={expected!r}, "
                    f"actual={audit.get(key)!r}")
        _validate_artifact(
            audit.get("verification_evidence"),
            f"{where}.receipt.verification_evidence",
            artifact_root=artifact_root, deep=True)
        checks = audit.get("checks")
        if not isinstance(checks, dict):
            raise ContractError(f"{where}.receipt checks must be an object")
        expected_stages = profile.get("required_stages")
        if set(checks) != set(expected_stages):
            raise ContractError(
                f"{where}.receipt check stages must exactly match profile: "
                f"expected={sorted(expected_stages)!r}, actual={sorted(checks)!r}")
        for stage_name in expected_stages:
            requirements = profile.get(stage_name, {}).get("requirements", [])
            rows = checks.get(stage_name)
            if not isinstance(requirements, list) or not all(
                    isinstance(item, str) and item for item in requirements):
                raise ContractError(
                    f"profile {profile_id!r} stage {stage_name!r} has invalid requirements")
            if not isinstance(rows, list):
                raise ContractError(
                    f"{where}.receipt checks.{stage_name} must be a list")
            actual_requirements = [
                row.get("requirement") if isinstance(row, dict) else None
                for row in rows]
            if actual_requirements != requirements:
                raise ContractError(
                    f"{where}.receipt checks.{stage_name} must preserve every "
                    "profile requirement exactly and in registry order")
            for index, row in enumerate(rows):
                assert isinstance(row, dict)
                if row.get("status") != "PASS":
                    raise ContractError(
                        f"{where}.receipt checks.{stage_name}[{index}] is not PASS")
                method = row.get("verification_method")
                if not isinstance(method, str) or not method.strip():
                    raise ContractError(
                        f"{where}.receipt checks.{stage_name}[{index}]."
                        "verification_method must not be empty")
    return {"receipt": receipt}


def _validate_rebuild_commands(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ContractError("rebuild.commands must be a non-empty list")
    seen: set[str] = set()
    out = []
    for i, row in enumerate(value):
        where = f"rebuild.commands[{i}]"
        if not isinstance(row, dict):
            raise ContractError(f"{where} must be an object")
        phase = _require(row, "phase", str, where)
        argv = _require(row, "argv", list, where)
        if phase in seen or not phase:
            raise ContractError(f"{where}.phase must be non-empty and unique")
        seen.add(phase)
        if not argv or not all(isinstance(x, str) and x for x in argv):
            raise ContractError(f"{where}.argv must be a non-empty string array")
        out.append({"phase": phase, "argv": argv})
    if out[-1]["phase"] != "verify-membership":
        raise ContractError(
            "rebuild.commands must end with phase 'verify-membership'")
    return out


def _validate_population(value: Any, *, artifact_root: Path | None,
                         deep: bool) -> dict[str, Any]:
    """Bind a rechunked release to the source/document population it replaces.

    Point identity and point count are expected to change under a new chunking
    policy.  This block therefore proves one-to-one resolution of the baseline
    source/document inventory independently of final collection membership.
    """
    if not isinstance(value, dict):
        raise ContractError("population must be an object")
    basis = _require(value, "basis", str, "population")
    if basis != POPULATION_BASIS:
        raise ContractError(
            f"population.basis must be {POPULATION_BASIS!r}")
    baseline_collection = _require(
        value, "baseline_collection", str, "population").strip()
    if not baseline_collection:
        raise ContractError("population.baseline_collection must not be empty")
    baseline_points = _nonnegative(
        value.get("baseline_points"), "population.baseline_points")
    documents = _nonnegative(
        value.get("baseline_source_documents"),
        "population.baseline_source_documents")
    resolved = _nonnegative(
        value.get("resolved_source_documents"),
        "population.resolved_source_documents")
    if resolved != documents:
        raise ContractError(
            "population source/document conservation failed: "
            f"resolved={resolved} != baseline={documents}")
    for name in (
            "baseline_point_membership_receipt",
            "baseline_source_inventory_receipt",
            "resolution_receipt"):
        _validate_artifact(value.get(name), f"population.{name}",
                           artifact_root=artifact_root, deep=deep)

    refresh = _require(value, "refresh_policy", dict, "population")
    mode = _require(refresh, "mode", str, "population.refresh_policy")
    if mode not in REFRESH_MODES:
        raise ContractError(
            "population.refresh_policy.mode must be one of "
            f"{sorted(REFRESH_MODES)!r}")
    for field in (
            "retrieval_metadata_required",
            "retrieved_original_bytes_preserved",
            "content_change_classification_required"):
        if refresh.get(field) is not True:
            raise ContractError(
                f"population.refresh_policy.{field} must be true")
    return {
        "basis": basis,
        "baseline_collection": baseline_collection,
        "baseline_points": baseline_points,
        "baseline_source_documents": documents,
        "resolved_source_documents": resolved,
        "refresh_mode": mode,
    }


def _read_membership_receipt(artifact: dict[str, Any], where: str, *,
                             collection: str, points: int,
                             artifact_root: Path | None,
                             deep: bool) -> dict[str, Any] | None:
    """Reconcile a collection's point count against its membership receipt.

    Under a seal, `collection.points` is reconciled against the summed vector
    records of every source.  A ruling that carries the source chain has no
    such sum, so the count would be an unchecked number in a file -- exactly
    the shape of claim this contract exists to refuse.  The membership receipt
    is the substitute, and a stronger one for the fact at issue: it is produced
    by scrolling the live collection, it enumerates every shard it wrote with
    that shard's row count, byte count and sha256, and it refuses to finish
    unless the rows it streamed equal the count the collection reported.

    Read here (cheap: the receipt is a small JSON file), reconciled three ways,
    and never confused with a re-hash of the shards themselves -- which live
    off-box and are digest-verified where they live.
    """
    if not deep:
        return None
    assert artifact_root is not None
    rel = _portable_uri(artifact["uri"], where)
    receipt = _load_object(artifact_root.joinpath(*rel.parts), f"{where} receipt")
    schema = receipt.get("schema")
    if schema not in MEMBERSHIP_RECEIPT_SCHEMAS:
        raise ContractError(
            f"{where} schema is {schema!r}, not one of "
            f"{list(MEMBERSHIP_RECEIPT_SCHEMAS)!r}")
    if receipt.get("collection") != collection:
        raise ContractError(
            f"{where} is for collection {receipt.get('collection')!r}, "
            f"not {collection!r}")
    rows = _nonnegative(receipt.get("rows"), f"{where}.rows")
    if rows != points:
        raise ContractError(
            f"{where} rows={rows} != declared points={points}")
    shards = receipt.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ContractError(f"{where}.shards must be a non-empty list")
    shard_rows = shard_bytes = 0
    names: set[str] = set()
    for i, shard in enumerate(shards):
        at = f"{where}.shards[{i}]"
        if not isinstance(shard, dict):
            raise ContractError(f"{at} must be an object")
        name = _require(shard, "file", str, at).strip()
        if not name or name in names:
            raise ContractError(f"{at}.file must be non-empty and unique")
        names.add(name)
        digest = _require(shard, "sha256", str, at)
        if not SHA256_RE.fullmatch(digest):
            raise ContractError(f"{at}.sha256 must be lowercase 64-hex")
        shard_rows += _nonnegative(shard.get("rows"), f"{at}.rows")
        shard_bytes += _nonnegative(shard.get("bytes"), f"{at}.bytes")
    if shard_rows != rows:
        raise ContractError(
            f"{where} conservation failed: shards sum to {shard_rows} rows, "
            f"the receipt claims {rows}")
    total = receipt.get("total_bytes")
    if total is not None and total != shard_bytes:
        raise ContractError(
            f"{where}.total_bytes={total} != summed shard bytes={shard_bytes}")
    return {"schema": schema, "rows": rows, "shards": len(shards),
            "bytes": shard_bytes}


def _validate_ruling(value: Any, *, collection_name: str,
                     expect_collection: str | None) -> dict[str, Any]:
    """A named ruling: who decided, over what, and what stays owed.

    The ruling is the whole of the difference between `ruled` and `blocked`.
    It must therefore be unambiguous about all three, and it must be bound to
    ONE collection: a ruling for `aw_e8` is not a ruling for anything else, and
    a ruled manifest presented for an unnamed apply is refused outright.
    """
    if not isinstance(value, dict):
        raise ContractError(
            "a manifest with status 'ruled' must carry a `ruling` object")
    fields: dict[str, str] = {}
    for key in RULING_FIELDS:
        text = value.get(key)
        if not isinstance(text, str) or not text.strip():
            raise ContractError(f"ruling.{key} must be a non-empty string")
        fields[key] = text.strip()
    # "PENDING Tim's merge" is how this project writes a ruling BEFORE it is
    # ratified (ADR-107's named rulings do the same). A ruling that still says
    # so has not been ratified, and an unratified ruling is a waiver.
    if fields["ratified_by"].upper().startswith(("PENDING", "TBD", "TODO")):
        raise ContractError(
            f"ruling.ratified_by is {fields['ratified_by']!r}: a ruling that "
            "still says it is pending has not been ratified, and an "
            "unratified ruling is a waiver")
    ruled_collection = fields["collection"]
    if ruled_collection != collection_name:
        raise ContractError(
            f"ruling.collection {ruled_collection!r} != manifest collection "
            f"{collection_name!r}: a ruling is bound to one collection")
    if not expect_collection:
        raise ContractError(
            "a ruled manifest may only be verified against a named "
            "collection: pass --expect-collection (or set "
            "AW_PRESERVATION_COLLECTION). An unbound ruling is a waiver.")
    if expect_collection != ruled_collection:
        raise ContractError(
            f"ruling covers {ruled_collection!r}, and this apply is for "
            f"{expect_collection!r}")

    gaps = value.get("carried_gaps")
    if not isinstance(gaps, list) or not gaps:
        raise ContractError(
            "ruling.carried_gaps must be a non-empty list: a ruling that "
            "carries nothing is a seal, and should say so")
    carried: list[dict[str, Any]] = []
    suspended: set[str] = set()
    seen: set[str] = set()
    for i, gap in enumerate(gaps):
        where = f"ruling.carried_gaps[{i}]"
        if not isinstance(gap, dict):
            raise ContractError(f"{where} must be an object")
        gap_id = _require(gap, "id", str, where).strip()
        if not gap_id or gap_id in seen:
            raise ContractError(f"{where}.id must be non-empty and unique")
        seen.add(gap_id)
        for key in ("source_manifest", "obligation", "owner", "vehicle"):
            text = gap.get(key)
            if not isinstance(text, str) or not text.strip():
                raise ContractError(f"{where}.{key} must be a non-empty string")
        clauses = gap.get("suspends")
        if not isinstance(clauses, list):
            raise ContractError(
                f"{where}.suspends must be a list (empty means: carried as an "
                "obligation, suspending no clause of this contract)")
        for clause in clauses:
            if clause not in SUSPENDABLE_CLAUSES:
                raise ContractError(
                    f"{where}.suspends names {clause!r}, which is not a "
                    f"suspendable clause. Suspendable: "
                    f"{list(SUSPENDABLE_CLAUSES)!r}")
            suspended.add(clause)
        carried.append({"id": gap_id, "suspends": sorted(set(clauses)),
                        "owner": gap["owner"].strip(),
                        "vehicle": gap["vehicle"].strip()})
    return {**fields, "carried_gaps": carried, "suspended": sorted(suspended)}


def _validate_rollback(value: Any, *, ruled_collection: str,
                       artifact_root: Path | None,
                       deep: bool) -> dict[str, Any]:
    """The incumbent a ruled promotion can be put back to.

    A ruling carries open proofs forward across a promotion.  That is only
    honest while the thing being promoted over is still there: the difference
    between "we carried five proofs" and "we lost what they were about" is
    whether the incumbent can be restored.  So a ruled manifest states the
    incumbent by name, reconciles its own membership receipt, and names the
    destructive step that must NOT be run while the obligations are open.
    """
    if not isinstance(value, dict):
        raise ContractError(
            "a manifest with status 'ruled' must carry a `rollback` object: a "
            "promotion that carries open proofs must be reversible")
    name = _require(value, "collection", str, "rollback").strip()
    if not name:
        raise ContractError("rollback.collection must not be empty")
    if name == ruled_collection:
        raise ContractError(
            f"rollback.collection is {name!r}, the same collection the ruling "
            "promotes: that is not a rollback target")
    points = _nonnegative(value.get("points"), "rollback.points")
    for key in ("mechanism", "destructive_step_withheld"):
        text = value.get(key)
        if not isinstance(text, str) or not text.strip():
            raise ContractError(f"rollback.{key} must be a non-empty string")
    receipt = _validate_artifact(
        value.get("membership_receipt"), "rollback.membership_receipt",
        artifact_root=artifact_root, deep=deep)
    reconciled = _read_membership_receipt(
        receipt, "rollback.membership_receipt", collection=name,
        points=points, artifact_root=artifact_root, deep=deep)
    offbox = _validate_artifact(
        value.get("offbox_receipt"), "rollback.offbox_receipt",
        artifact_root=artifact_root, deep=deep)
    return {"collection": name, "points": points,
            "membership": reconciled, "offbox_receipt": offbox["uri"],
            "destructive_step_withheld":
                value["destructive_step_withheld"].strip()}


def _print_ruling(ruling: dict[str, Any], stream) -> None:
    """Say, on every verify, what this PASS rests on and what stays owed.

    Printed to stderr on purpose: `alias_swap.sh` sends the proof on stdout to
    /dev/null, and a verify that passes under a ruling must not be able to go
    by silently in anybody's terminal.
    """
    print(f"PRESERVATION RULING {ruling['id']} -- {ruling['adr']}", file=stream)
    print(f"  collection : {ruling['collection']}", file=stream)
    print(f"  ruled by   : {ruling['ruled_by']} on {ruling['ruled_on']} "
          f"({ruling['ruled_via']})", file=stream)
    print(f"  ratified   : {ruling['ratified_by']}", file=stream)
    print(f"  cite       : {ruling['cite']}", file=stream)
    print(f"  NOT covered: {ruling['does_not_cover']}", file=stream)
    print(f"  carried gaps ({len(ruling['carried_gaps'])}) -- these are open "
          "obligations, not closed proofs:", file=stream)
    for gap in ruling["carried_gaps"]:
        clauses = ", ".join(gap["suspends"]) or "no clause of this contract"
        print(f"    - {gap['id']}: suspends {clauses}; "
              f"owner {gap['owner']}; vehicle {gap['vehicle']}", file=stream)


def verify_manifest(manifest_path: str | Path, *, registry_path: str | Path | None = None,
                    artifact_root: str | Path | None = None, deep: bool = False,
                    expect_collection: str | None = None) -> dict[str, Any]:
    """Validate and return a compact publish/rebuild proof.

    Validation is deliberately independent of Qdrant.  The collection gate
    consumes this proof in addition to its data/readback and quality gates.
    """
    mp = Path(manifest_path).expanduser().resolve()
    manifest = _load_object(mp, "preservation manifest")
    if manifest.get("schema") != SCHEMA:
        raise ContractError(f"manifest schema must be {SCHEMA!r}")
    status = manifest.get("status")
    if status not in (SEALED, RULED):
        gaps = manifest.get("blocking_gaps") or []
        detail = "; ".join(str(x) for x in gaps[:5])
        raise ContractError(f"manifest is {status!r}, not sealed" +
                            (f": {detail}" if detail else ""))
    release_id = _require(manifest, "release_id", str, "manifest").strip()
    if not release_id:
        raise ContractError("manifest.release_id must not be empty")
    git_sha = _require(manifest, "git_sha", str, "manifest")
    if not GIT_SHA_RE.fullmatch(git_sha):
        raise ContractError("manifest.git_sha must be lowercase 40-hex")

    registry_ref = _require(manifest, "profile_registry", dict, "manifest")
    registry_sha = _require(registry_ref, "sha256", str, "profile_registry")
    if registry_path is None:
        rel = _require(registry_ref, "path", str, "profile_registry")
        rp = (mp.parent / rel).resolve()
    else:
        rp = Path(registry_path).expanduser().resolve()
    profiles = load_profiles(rp, registry_sha)
    root = Path(artifact_root).expanduser().resolve() if artifact_root else None

    # A ruling is read BEFORE any clause it might carry, so that every refusal
    # below is decided against the right requirement set -- and so that a
    # manifest claiming `ruled` without a ruling fails here rather than
    # somewhere downstream with a confusing message.
    collection = _require(manifest, "collection", dict, "manifest")
    collection_name = _require(collection, "name", str, "collection")
    if expect_collection and collection_name != expect_collection:
        raise ContractError(
            f"manifest collection {collection_name!r} != requested {expect_collection!r}")
    ruling: dict[str, Any] | None = None
    suspended: set[str] = set()
    rollback: dict[str, Any] | None = None
    if status == RULED:
        ruling = _validate_ruling(manifest.get("ruling"),
                                  collection_name=collection_name,
                                  expect_collection=expect_collection)
        suspended = set(ruling["suspended"])
        rollback = _validate_rollback(manifest.get("rollback"),
                                      ruled_collection=collection_name,
                                      artifact_root=root, deep=deep)
    elif manifest.get("ruling") is not None:
        raise ContractError(
            "a sealed manifest must not carry a `ruling` block: a seal that "
            "carries a ruling is a ruling that calls itself a seal")

    # A suspended clause may be ABSENT.  Present, it is validated in full --
    # a ruling lowers what must be supplied, never what a supplied block proves.
    def _carried(clause: str, value: Any) -> bool:
        return clause in suspended and value is None

    population = None
    if not _carried("population", manifest.get("population")):
        population = _validate_population(
            manifest.get("population"), artifact_root=root, deep=deep)

    sources = manifest.get("sources")
    source_ids: set[str] = set()
    total_chunks = total_vectors = 0
    profile_counts: dict[str, int] = {}
    if _carried("sources", sources):
        sources = []
    else:
        sources = _require(manifest, "sources", list, "manifest")
        if not sources:
            raise ContractError("manifest.sources must not be empty")
    for i, source in enumerate(sources):
        where = f"sources[{i}]"
        if not isinstance(source, dict):
            raise ContractError(f"{where} must be an object")
        source_id = _require(source, "source_id", str, where).strip()
        if not source_id or source_id in source_ids:
            raise ContractError(f"{where}.source_id must be non-empty and unique")
        source_ids.add(source_id)
        profile_id = _require(source, "profile", str, where)
        profile = profiles.get(profile_id)
        if not isinstance(profile, dict):
            raise ContractError(f"{where}.profile names unknown profile {profile_id!r}")
        required = profile.get("required_stages")
        if not isinstance(required, list) or not required:
            raise ContractError(f"profile {profile_id!r} has no required_stages")
        # The profile selects a family of semantics; the source configuration
        # pins the exact inventory, filters, versions, and adapter parameters.
        # Without both, a clone cannot know which member of that family to run.
        _validate_artifact(source.get("configuration"), f"{where}.configuration",
                           artifact_root=root, deep=deep)
        stages_raw = _require(source, "stages", dict, where)
        missing = [name for name in required if name not in stages_raw]
        if missing:
            raise ContractError(f"{where} missing required stages: {', '.join(missing)}")
        stages = {
            name: _validate_stage(stages_raw[name], f"{where}.stages.{name}",
                                  artifact_root=root, deep=deep)
            for name in required
        }
        if "original" in stages and "markdown" in stages:
            _validate_content_fidelity(
                source.get("content_fidelity"), f"{where}.content_fidelity",
                source_id=source_id, stages=stages,
                artifact_root=root, deep=deep)
        _validate_profile_compliance(
            source.get("profile_compliance"), f"{where}.profile_compliance",
            source_id=source_id, profile_id=profile_id, profile=profile,
            registry_sha256=registry_sha,
            artifact_root=root, deep=deep)
        lineage = _require(source, "lineage", dict, where)
        for edge_name, input_name, output_name in STAGE_EDGES:
            if input_name not in stages or output_name not in stages:
                continue
            _validate_edge(lineage.get(edge_name), f"{where}.lineage.{edge_name}",
                           stages[input_name]["records"], stages[output_name]["records"],
                           artifact_root=root, deep=deep)
        durability = _require(source, "durability", dict, where)
        if durability.get("status") != "verified_offbox":
            raise ContractError(f"{where}.durability.status must be 'verified_offbox'")
        _validate_artifact(durability.get("receipt"), f"{where}.durability.receipt",
                           artifact_root=root, deep=deep)
        total_chunks += stages.get("chunks", {}).get("records", 0)
        total_vectors += stages.get("vectors", {}).get("records", 0)
        profile_counts[profile_id] = profile_counts.get(profile_id, 0) + 1

    points = _nonnegative(collection.get("points"), "collection.points")
    membership = _validate_artifact(
        collection.get("membership_receipt"), "collection.membership_receipt",
        artifact_root=root, deep=deep)
    if "sources" in suspended and not sources:
        # No summed vector records exist to reconcile the count against, so the
        # membership receipt carries that burden instead -- read, not trusted.
        reconciled = _read_membership_receipt(
            membership, "collection.membership_receipt",
            collection=collection_name, points=points,
            artifact_root=root, deep=deep)
        # ADR-026's loss incident was a corpus that only ever lived on a build
        # box. With the per-source durability receipts carried, this receipt is
        # the whole of what fixes what the collection held at promotion time,
        # so it does not get to live on one disk on one machine either.
        durability = _require(collection, "durability", dict, "collection")
        if durability.get("status") != "verified_offbox":
            raise ContractError(
                "collection.durability.status must be 'verified_offbox': the "
                "membership receipt is the only proof of this collection's "
                "contents that the ruling does not carry, and a proof that "
                "exists once is not preserved")
        _validate_artifact(durability.get("receipt"),
                           "collection.durability.receipt",
                           artifact_root=root, deep=deep)
    else:
        reconciled = None
        if points != total_vectors:
            raise ContractError(
                f"collection.points={points} != summed source vector records={total_vectors}")
    if not _carried("collection.build_receipt", collection.get("build_receipt")):
        _validate_artifact(collection.get("build_receipt"),
                           "collection.build_receipt", artifact_root=root, deep=deep)
    commands: list[dict[str, Any]] = []
    if not _carried("rebuild", manifest.get("rebuild")):
        rebuild = _require(manifest, "rebuild", dict, "manifest")
        commands = _validate_rebuild_commands(rebuild.get("commands"))
    if ruling is not None:
        _print_ruling(ruling, sys.stderr)
    proof = {
        "status": "PASS",
        "schema": SCHEMA,
        "manifest_status": status,
        "release_id": release_id,
        "git_sha": git_sha,
        "manifest": str(mp),
        "manifest_sha256": sha256_file(mp),
        "profile_registry_sha256": registry_sha,
        "sources": len(sources),
        "population": population,
        "profile_counts": profile_counts,
        "chunks": total_chunks,
        "vectors": total_vectors,
        "collection": collection_name,
        "points": points,
        "commands": len(commands),
        "deep": deep,
    }
    if ruling is not None:
        proof["ruling_id"] = ruling["id"]
        proof["ruling"] = ruling
        proof["carried_gaps"] = [gap["id"] for gap in ruling["carried_gaps"]]
        proof["suspended_clauses"] = ruling["suspended"]
        proof["rollback"] = rollback
        proof["membership"] = reconciled
    return proof


def verify_publish_environment(*, expect_collection: str | None = None) -> dict[str, Any]:
    """Resolve the fail-closed publish gate from tracked runtime config."""
    manifest = os.environ.get("AW_PRESERVATION_MANIFEST", "").strip()
    if not manifest:
        raise ContractError(
            "AW_PRESERVATION_MANIFEST is required for every armed publish/apply")
    registry = os.environ.get("AW_PRESERVATION_PROFILE_REGISTRY") or None
    root = os.environ.get("AW_ARTIFACT_ROOT") or None
    deep = os.environ.get("AW_PRESERVATION_DEEP", "0") == "1"
    expected = expect_collection or os.environ.get("AW_PRESERVATION_COLLECTION", "").strip()
    if not expected:
        raise ContractError(
            "AW_PRESERVATION_COLLECTION is required to bind the manifest to this apply")
    proof = verify_manifest(manifest, registry_path=registry, artifact_root=root,
                            deep=deep, expect_collection=expected)
    # This is the ENV-driven gate the live/incremental write paths take
    # (`twin_first_apply`, `upsert_wave_prod`).  ADR-098 amendment 1's ruling
    # covers ONE alias move and says so in `does_not_cover`; the alias swap
    # reaches this module through the CLI, not through here.  So a ruled
    # manifest does not satisfy this gate, and the ruling cannot leak into an
    # operation class nobody ruled on by pointing an env var at it.
    if proof.get("manifest_status") == RULED:
        raise ContractError(
            f"{manifest} is `ruled` ({proof.get('ruling_id')}), not sealed. A "
            "named ruling covers the promotion it names and nothing else; the "
            "write paths that take this gate still require a sealed manifest, "
            "or the scoped content-pack contract (ADR-098 Ruling 1).")
    return proof


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="verify a sealed preservation manifest")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--registry")
    verify.add_argument("--artifact-root")
    verify.add_argument("--expect-collection")
    verify.add_argument("--deep", action="store_true")
    args = ap.parse_args(argv)
    try:
        proof = verify_manifest(
            args.manifest, registry_path=args.registry,
            artifact_root=args.artifact_root, deep=args.deep,
            expect_collection=args.expect_collection)
    except ContractError as exc:
        print(f"PRESERVATION CONTRACT FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(proof, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
