#!/usr/bin/env python3
# Moved from cytir.index.content_pack_preservation (task 677, task 643 S5); the old path is an alias shim.
"""Fail-closed preservation authority for one incremental content pack.

This contract is deliberately distinct from ``aw.preservation-manifest.v1``.
It authorizes only an exact, pre-planned content-pack mutation and explicitly
does not claim that the surrounding production collection is rebuildable.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
from cytir.core import internal_http as IH
from pathlib import Path
from typing import Any, Callable

from cytir.core.digest import canonical_sha256  # noqa: F401 -- re-exported; callers use CP.canonical_sha256
from cytir.index import preservation_contract as full


SCHEMA = "aw.content-pack-preservation-intent.v1"
SCOPE = "incremental-content-pack-only"
VECTOR_MEMBERSHIP_SCHEMA = "aw.content-pack-vector-membership-receipt.v1"
TOUCHED_GROUPS_SCHEMA = "aw.content-pack-touched-groups-receipt.v1"
BASELINE_MEMBERSHIP_SCHEMA = "aw.content-pack-baseline-membership-receipt.v1"
TERMINAL_REQUIREMENTS = {
    "exact_artifact_membership": True,
    "exact_artifact_content": True,
    "tombstones_absent": True,
    "twin_prod_plan_equivalence": True,
    "twin_prod_terminal_equivalence": True,
    "terminal_ledger_status": "TWIN-PROD-VERIFIED-OK",
}

ContractError = full.ContractError


# Task 447 / ADR-098: a textual ``twin.url != prod.url`` compare is defeated
# by an SSH tunnel that makes the twin URL resolve to production (task 444 —
# ``<build-box>:6334`` was silently a prod tunnel while scripts believed it was
# the twin). This is the fail-closed *physical* check layered on top of the
# textual one below: both endpoints are asked, live, for an instance-level
# identity, and the identities must differ.
#
# Empirically probed on the build box 2026-08-27 (task 447 evidence, both
# instances qdrant 1.18.3):
#   - ``GET /cluster`` returns ``{"status": "disabled"}`` on every endpoint in
#     this deployment (single-node, clustering off) — no peer_id is ever
#     exposed. Not usable.
#   - ``GET /telemetry`` -> ``result.id`` is a UUID minted fresh in-memory the
#     moment the qdrant process starts (no telemetry-id file exists anywhere
#     under its storage dir) and stays stable for the life of that process.
#     Real twin (``localhost:6444``): ``9084824f-ec91-4f60-b825-96401c2cf5d8``
#     (startup ``2026-08-27T19:25:14.990Z``). Prod-through-tunnel
#     (``localhost:6334``): ``442ddb62-bb8d-4837-91a0-467a506ec194`` (startup
#     ``2026-08-13T17:41:00.840Z``) — confirmed distinct, and stable across
#     repeated calls to the same endpoint.
# ``result.id`` is therefore the identity field; ``result.app.startup`` is
# bound alongside it as corroborating evidence (both must be present).
PHYSICAL_IDENTITY_TIMEOUT_S = 10

IdentityFetch = Callable[[str], dict[str, Any]]


def _fetch_identity_default(url: str) -> dict[str, Any]:
    """Live ``GET {url}/telemetry`` -> ``{"id": ..., "startup": ...}``.

    Raises :class:`ContractError` on any transport failure or malformed body
    — this is a fail-closed safety probe, not a best-effort lookup.
    """
    endpoint = url.rstrip("/") + "/telemetry"
    try:
        body = IH.fetch_json(endpoint, method="GET", timeout=PHYSICAL_IDENTITY_TIMEOUT_S)
    except (urllib.error.URLError, OSError, ValueError,
            json.JSONDecodeError) as exc:
        raise ContractError(
            f"physical identity probe failed for {endpoint}: {exc}") from exc
    result = body.get("result") if isinstance(body, dict) else None
    node_id = result.get("id") if isinstance(result, dict) else None
    app = result.get("app") if isinstance(result, dict) else None
    startup = app.get("startup") if isinstance(app, dict) else None
    if not isinstance(node_id, str) or not node_id.strip():
        raise ContractError(
            f"physical identity probe for {endpoint} did not return a "
            "non-empty result.id")
    if not isinstance(startup, str) or not startup.strip():
        raise ContractError(
            f"physical identity probe for {endpoint} did not return a "
            "non-empty result.app.startup")
    return {"id": node_id, "startup": startup}


def _require_physically_distinct_targets(
        targets: dict[str, dict[str, Any]], *,
        identity_fetch: IdentityFetch | None = None) -> dict[str, dict[str, str]]:
    """Prove twin and prod are different running Qdrant processes.

    Fail-closed: a refusing endpoint, a malformed identity, or matching
    identities all raise. There is no environment-variable escape hatch —
    this check runs on every armed content-pack apply, unconditionally.
    """
    fetch = identity_fetch or _fetch_identity_default
    identities: dict[str, dict[str, str]] = {}
    for role in ("twin", "prod"):
        identity = fetch(targets[role]["url"])
        if (not isinstance(identity, dict) or
                not isinstance(identity.get("id"), str) or
                not identity["id"].strip() or
                not isinstance(identity.get("startup"), str) or
                not identity["startup"].strip()):
            raise ContractError(
                f"targets.{role} physical identity probe returned an "
                f"incomplete identity: {identity!r}")
        identities[role] = {"id": identity["id"], "startup": identity["startup"]}
    if identities["twin"]["id"] == identities["prod"]["id"]:
        raise ContractError(
            "targets.twin and targets.prod resolve to the SAME physical "
            "Qdrant instance (both report /telemetry result.id "
            f"{identities['twin']['id']!r}) — refusing an apply that would "
            "silently write production twice; a tunnel or a repointed URL "
            "is the usual cause (task 444)")
    return identities


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be an object")
    return value


def _positive(value: Any, where: str) -> int:
    number = full._nonnegative(value, where)
    if not number:
        raise ContractError(f"{where} must be positive")
    return number


def _sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or not full.SHA256_RE.fullmatch(value):
        raise ContractError(f"{where} must be lowercase 64-hex")
    return value


def _artifact(value: Any, where: str, *, root: Path | None,
              deep: bool) -> dict[str, Any]:
    return full._validate_artifact(
        value, where, artifact_root=root, deep=deep)


def _same_artifact_identity(sealed: dict[str, Any], observed: Any) -> bool:
    return (isinstance(observed, dict) and
            sealed["bytes"] == observed.get("bytes") and
            sealed["sha256"] == observed.get("sha256"))


def _receipt_body(ref: dict[str, Any], where: str, *, root: Path | None,
                  deep: bool) -> dict[str, Any] | None:
    if not deep:
        return None
    assert root is not None
    rel = full._portable_uri(ref["uri"], where)
    return full._load_object(root.joinpath(*rel.parts), where)


def _validate_mutation_plan(value: Any, *, rows: int,
                            tombstone_sha256: str | None) -> dict[str, Any]:
    plan = _object(value, "mutation_plan")
    before = _object(plan.get("before"), "mutation_plan.before")
    if (before.get("status") != "green" or
            before.get("optimizer_status") != "ok" or
            before.get("update_queue_length") != 0):
        raise ContractError("mutation_plan.before must be quiet-green")
    _positive(before.get("points_count"), "mutation_plan.before.points_count")

    comparison = _object(plan.get("comparison"), "mutation_plan.comparison")
    for field in (
            "artifact_points_sha256", "existing_points_sha256",
            "identical_points_sha256", "write_points_sha256",
            "decisions_sha256"):
        _sha(comparison.get(field), f"mutation_plan.comparison.{field}")

    expected = {
        "rows": rows,
        "existing": plan.get("existing_ids"),
        "missing": plan.get("new_ids"),
        "identical": plan.get("point_identical"),
        "changed": plan.get("point_changed"),
    }
    for field, wanted in expected.items():
        actual = full._nonnegative(
            comparison.get(field), f"mutation_plan.comparison.{field}")
        if actual != wanted:
            raise ContractError(
                f"mutation_plan comparison {field} mismatch: "
                f"plan={wanted!r}, comparison={actual!r}")
    existing = full._nonnegative(
        plan.get("existing_ids"), "mutation_plan.existing_ids")
    missing = full._nonnegative(plan.get("new_ids"), "mutation_plan.new_ids")
    identical = full._nonnegative(
        plan.get("point_identical"), "mutation_plan.point_identical")
    changed = full._nonnegative(
        plan.get("point_changed"), "mutation_plan.point_changed")
    writes = full._nonnegative(
        plan.get("point_writes"), "mutation_plan.point_writes")
    noops = full._nonnegative(
        plan.get("point_noops"), "mutation_plan.point_noops")
    superseded = full._nonnegative(
        plan.get("superseded"), "mutation_plan.superseded")
    aged = full._nonnegative(
        plan.get("aged_tombstones"), "mutation_plan.aged_tombstones")
    topology = full._nonnegative(
        plan.get("planned_topology_mutations"),
        "mutation_plan.planned_topology_mutations")
    delta = plan.get("expected_delta")
    if not isinstance(delta, int) or isinstance(delta, bool):
        raise ContractError("mutation_plan.expected_delta must be an integer")
    if not (existing + missing == rows and identical + changed == existing and
            writes == changed + missing and noops == identical and
            topology == writes + superseded + aged and
            delta == missing - superseded - aged):
        raise ContractError("mutation_plan count conservation failed")
    if plan.get("tombstone_pointer_sha256") != tombstone_sha256:
        raise ContractError("mutation_plan tombstone SHA mismatch")
    predecessor = _object(
        plan.get("tombstone_predecessor_proof"),
        "mutation_plan.tombstone_predecessor_proof")
    if predecessor.get("rows") != superseded + aged:
        raise ContractError("mutation_plan predecessor count mismatch")
    _sha(predecessor.get("fingerprint_digest"),
         "mutation_plan.tombstone_predecessor_proof.fingerprint_digest")
    return plan


def _validate_targets(value: Any, *, plan: dict[str, Any],
                      root: Path | None, deep: bool,
                      identity_fetch: IdentityFetch | None = None) -> dict[str, Any]:
    targets = _object(value, "targets")
    if set(targets) != {"twin", "prod"}:
        raise ContractError("targets must contain exactly twin and prod")
    out: dict[str, Any] = {}
    logical_memberships = set()
    before_points = plan["before"]["points_count"]
    after_points = before_points + plan["expected_delta"]
    for role in ("twin", "prod"):
        target = _object(targets[role], f"targets.{role}")
        if target.get("role") != role:
            raise ContractError(f"targets.{role}.role mismatch")
        for field in ("url", "collection"):
            if not isinstance(target.get(field), str) or not target[field].strip():
                raise ContractError(f"targets.{role}.{field} must not be empty")
        if target["url"] != target["url"].rstrip("/"):
            raise ContractError(f"targets.{role}.url must be normalized")
        baseline = _object(target.get("baseline"), f"targets.{role}.baseline")
        if baseline.get("points") != before_points:
            raise ContractError(f"targets.{role} baseline point count mismatch")
        membership = _artifact(
            baseline.get("membership_receipt"),
            f"targets.{role}.baseline.membership_receipt", root=root, deep=deep)
        if baseline.get("membership_receipt", {}).get("rows") != before_points:
            raise ContractError(
                f"targets.{role} baseline membership rows mismatch")
        logical = _sha(
            baseline.get("logical_membership_sha256"),
            f"targets.{role}.baseline.logical_membership_sha256")
        membership_body = _receipt_body(
            membership, f"targets.{role}.baseline.membership_receipt",
            root=root, deep=deep)
        if membership_body is not None:
            if (membership_body.get("schema") != BASELINE_MEMBERSHIP_SCHEMA or
                    membership_body.get("status") != "PASS" or
                    membership_body.get("role") != role or
                    membership_body.get("collection") != target["collection"] or
                    membership_body.get("points") != before_points or
                    membership_body.get("logical_membership_sha256") != logical):
                raise ContractError(
                    f"targets.{role} baseline membership receipt mismatch")
            _artifact(
                membership_body.get("verification_evidence"),
                f"targets.{role}.baseline.membership_receipt."
                "verification_evidence", root=root, deep=True)
        logical_memberships.add(logical)
        if target.get("expected_after_points") != after_points:
            raise ContractError(f"targets.{role} expected-after count mismatch")
        out[role] = {
            "role": role, "url": target["url"],
            "collection": target["collection"],
            "baseline": {"points": before_points,
                         "membership_receipt": membership,
                         "logical_membership_sha256": logical},
            "expected_after_points": after_points,
        }
    if targets["twin"]["url"] == targets["prod"]["url"]:
        raise ContractError("targets must use distinct endpoints")
    # Textual check above is the cheap first fail (ADR-098); it is defeated
    # by an SSH tunnel that makes two distinct URLs resolve to the same
    # running instance (task 444). Physical check below is authoritative.
    physical_identity = _require_physically_distinct_targets(
        out, identity_fetch=identity_fetch)
    for role in ("twin", "prod"):
        out[role]["physical_identity"] = physical_identity[role]
    if len(logical_memberships) != 1:
        raise ContractError("target baseline memberships differ")
    return out


def _validate_sources(sources: Any, *, registry: dict[str, Any],
                      registry_sha256: str, root: Path | None,
                      deep: bool) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(sources, list) or not sources:
        raise ContractError("sources must be a non-empty list")
    seen: set[str] = set()
    total_vectors = 0
    validated = []
    for index, source in enumerate(sources):
        where = f"sources[{index}]"
        source = _object(source, where)
        source_id = full._require(source, "source_id", str, where).strip()
        if not source_id or source_id in seen:
            raise ContractError(f"{where}.source_id must be non-empty and unique")
        seen.add(source_id)
        profile_id = full._require(source, "profile", str, where)
        profile = registry.get(profile_id)
        if not isinstance(profile, dict):
            raise ContractError(f"{where}.profile names unknown profile {profile_id!r}")
        required = profile.get("required_stages")
        stages_raw = full._require(source, "stages", dict, where)
        if not isinstance(required, list) or not required:
            raise ContractError(f"profile {profile_id!r} has no required_stages")
        missing = [name for name in required if name not in stages_raw]
        if missing:
            raise ContractError(f"{where} missing required stages: {', '.join(missing)}")
        _artifact(
            source.get("configuration"), f"{where}.configuration",
            root=root, deep=deep)
        stages = {
            name: full._validate_stage(
                stages_raw[name], f"{where}.stages.{name}",
                artifact_root=root, deep=deep)
            for name in required
        }
        if "original" in stages and "markdown" in stages:
            full._validate_content_fidelity(
                source.get("content_fidelity"), f"{where}.content_fidelity",
                source_id=source_id, stages=stages,
                artifact_root=root, deep=deep)
        full._validate_profile_compliance(
            source.get("profile_compliance"), f"{where}.profile_compliance",
            source_id=source_id, profile_id=profile_id, profile=profile,
            registry_sha256=registry_sha256, artifact_root=root, deep=deep)
        lineage = full._require(source, "lineage", dict, where)
        for edge_name, input_name, output_name in full.STAGE_EDGES:
            if input_name in stages and output_name in stages:
                full._validate_edge(
                    lineage.get(edge_name), f"{where}.lineage.{edge_name}",
                    stages[input_name]["records"], stages[output_name]["records"],
                    artifact_root=root, deep=deep)
        durability = full._require(source, "durability", dict, where)
        if durability.get("status") != "verified_offbox":
            raise ContractError(f"{where}.durability.status must be 'verified_offbox'")
        _artifact(durability.get("receipt"), f"{where}.durability.receipt",
                  root=root, deep=deep)
        total_vectors += stages.get("vectors", {}).get("records", 0)
        validated.append({"source_id": source_id, "profile": profile_id})
    return validated, total_vectors


def verify_intent(intent_path: str | Path, *, runtime: dict[str, Any],
                  registry_path: str | Path | None = None,
                  artifact_root: str | Path | None = None,
                  deep: bool = False,
                  identity_fetch: IdentityFetch | None = None) -> dict[str, Any]:
    path = Path(intent_path).expanduser().resolve()
    intent = full._load_object(path, "content-pack preservation intent")
    if intent.get("schema") != SCHEMA:
        raise ContractError(f"intent schema must be {SCHEMA!r}")
    if intent.get("status") != "sealed":
        raise ContractError("content-pack preservation intent is not sealed")
    if intent.get("scope") != SCOPE:
        raise ContractError(f"intent scope must be {SCOPE!r}")
    if intent.get("does_not_supersede_global_rebuild_manifest") is not True:
        raise ContractError("intent must not supersede the global rebuild manifest")
    intent_id = full._require(intent, "intent_id", str, "intent").strip()
    if not intent_id:
        raise ContractError("intent.intent_id must not be empty")
    git_sha = full._require(intent, "git_sha", str, "intent")
    if not full.GIT_SHA_RE.fullmatch(git_sha):
        raise ContractError("intent.git_sha must be lowercase 40-hex")
    runtime_git_sha = runtime.get("git_sha")
    if (not isinstance(runtime_git_sha, str) or
            not full.GIT_SHA_RE.fullmatch(runtime_git_sha) or
            git_sha != runtime_git_sha):
        raise ContractError(
            "intent.git_sha does not match the running immutable release")

    root = Path(artifact_root).expanduser().resolve() if artifact_root else None
    global_manifest = _object(
        intent.get("global_rebuild_manifest"), "global_rebuild_manifest")
    if (global_manifest.get("schema") != full.SCHEMA or
            global_manifest.get("status") != "blocked"):
        raise ContractError("global rebuild manifest must remain explicitly blocked")
    global_ref = _artifact(
        global_manifest.get("artifact"), "global_rebuild_manifest.artifact",
        root=root, deep=deep)
    if not _same_artifact_identity(
            global_ref, runtime.get("global_rebuild_manifest")):
        raise ContractError(
            "global rebuild manifest does not match the running release")
    global_body = _receipt_body(
        global_ref, "global_rebuild_manifest.artifact", root=root, deep=deep)
    if global_body is not None and (
            global_body.get("schema") != full.SCHEMA or
            global_body.get("status") != "blocked"):
        raise ContractError("referenced global rebuild manifest is not blocked")

    registry_ref = _object(intent.get("profile_registry"), "profile_registry")
    registry_sha = _sha(registry_ref.get("sha256"), "profile_registry.sha256")
    if registry_path is None:
        relative = full._require(registry_ref, "path", str, "profile_registry")
        registry_file = (path.parent / relative).resolve()
    else:
        registry_file = Path(registry_path).expanduser().resolve()
    registry = full.load_profiles(registry_file, registry_sha)

    pack = _object(intent.get("pack"), "pack")
    runtime_pack = _object(runtime.get("pack"), "runtime.pack")
    name = full._require(pack, "name", str, "pack")
    if name != runtime_pack.get("name"):
        raise ContractError("pack name does not match the armed transaction")
    rows = _positive(pack.get("rows"), "pack.rows")
    if rows != runtime_pack.get("rows"):
        raise ContractError("pack row count does not match the armed transaction")
    artifact = _artifact(pack.get("artifact"), "pack.artifact", root=root, deep=deep)
    staged = _artifact(
        pack.get("staged_verification"), "pack.staged_verification",
        root=root, deep=deep)
    for label, sealed, observed in (
            ("artifact", artifact, runtime_pack.get("artifact")),
            ("staged verification", staged,
             runtime_pack.get("staged_verification"))):
        if not _same_artifact_identity(sealed, observed):
            raise ContractError(f"pack {label} does not match the armed transaction")

    tombstones = _object(pack.get("tombstones"), "pack.tombstones")
    mode = tombstones.get("mode")
    tombstone_sha: str | None
    if mode == "explicit":
        tombstone_ref = _artifact(
            tombstones.get("artifact"), "pack.tombstones.artifact",
            root=root, deep=deep)
        tombstone_sha = tombstone_ref["sha256"]
        if not _same_artifact_identity(
                tombstone_ref, runtime_pack.get("tombstone_artifact")):
            raise ContractError("tombstones do not match the armed transaction")
    elif mode == "none":
        tombstone_sha = None
        if tombstones.get("artifact") is not None or \
                runtime_pack.get("tombstone_artifact") is not None:
            raise ContractError("no-tombstone intent cannot bind a tombstone artifact")
    else:
        raise ContractError("pack.tombstones.mode must be 'explicit' or 'none'")
    source_group = full._nonnegative(
        tombstones.get("source_group"), "pack.tombstones.source_group")
    aged_out = full._nonnegative(
        tombstones.get("aged_out"), "pack.tombstones.aged_out")

    plan = _validate_mutation_plan(
        intent.get("mutation_plan"), rows=rows,
        tombstone_sha256=tombstone_sha)
    if plan != runtime.get("mutation_plan"):
        raise ContractError("sealed mutation plan does not match live preflight")
    if (source_group != plan["superseded"] or
            aged_out != plan["aged_tombstones"]):
        raise ContractError("tombstone class counts do not match mutation plan")
    targets = _validate_targets(
        intent.get("targets"), plan=plan, root=root, deep=deep,
        identity_fetch=identity_fetch)
    if {role: {key: target[key] for key in ("role", "url", "collection")}
            for role, target in targets.items()} != runtime.get("targets"):
        raise ContractError("intent targets do not match the armed transaction")

    sources, vector_records = _validate_sources(
        intent.get("sources"), registry=registry,
        registry_sha256=registry_sha, root=root, deep=deep)
    if vector_records != rows:
        raise ContractError(
            f"source vector records={vector_records} != pack rows={rows}")

    membership_ref = _artifact(
        intent.get("pack_vector_membership_receipt"),
        "pack_vector_membership_receipt", root=root, deep=deep)
    membership = _receipt_body(
        membership_ref, "pack_vector_membership_receipt", root=root, deep=deep)
    if membership is not None:
        artifact_ids = membership.get("artifact_point_ids_sha256")
        source_ids = membership.get("source_vector_point_ids_sha256")
        if (membership.get("schema") != VECTOR_MEMBERSHIP_SCHEMA or
                membership.get("status") != "PASS" or
                membership.get("artifact_sha256") != artifact["sha256"] or
                membership.get("staged_verification_sha256") != staged["sha256"] or
                membership.get("rows") != rows or
                membership.get("source_vector_records") != vector_records or
                not isinstance(artifact_ids, str) or
                not full.SHA256_RE.fullmatch(artifact_ids) or
                source_ids != artifact_ids):
            raise ContractError(
                "pack vector membership receipt does not bind the pack")
        _artifact(
            membership.get("verification_evidence"),
            "pack_vector_membership_receipt.verification_evidence",
            root=root, deep=True)

    touched_ref = _artifact(
        intent.get("touched_groups_receipt"), "touched_groups_receipt",
        root=root, deep=deep)
    touched = _receipt_body(
        touched_ref, "touched_groups_receipt", root=root, deep=deep)
    if touched is not None:
        if (touched.get("schema") != TOUCHED_GROUPS_SCHEMA or
                touched.get("status") != "PASS" or
                touched.get("artifact_sha256") != artifact["sha256"] or
                touched.get("tombstone_sha256") != tombstone_sha or
                touched.get("mutation_plan_sha256") != canonical_sha256(plan) or
                not isinstance(touched.get("touched_groups"), int) or
                isinstance(touched.get("touched_groups"), bool) or
                touched["touched_groups"] <= 0 or
                touched.get("predecessor_rows") !=
                plan["superseded"] + plan["aged_tombstones"]):
            raise ContractError(
                "touched-groups receipt does not bind the mutation")
        _artifact(
            touched.get("verification_evidence"),
            "touched_groups_receipt.verification_evidence",
            root=root, deep=True)
    if intent.get("terminal_verification") != TERMINAL_REQUIREMENTS:
        raise ContractError("terminal verification requirements were weakened")

    return {
        "status": "PASS", "schema": SCHEMA, "scope": SCOPE,
        "intent_id": intent_id, "intent": str(path),
        "intent_sha256": full.sha256_file(path), "git_sha": git_sha,
        "global_rebuild_status": "blocked",
        "global_rebuild_manifest_sha256": global_ref["sha256"],
        "pack": name, "artifact_sha256": artifact["sha256"],
        "rows": rows, "sources": len(sources),
        "mutation_plan_sha256": canonical_sha256(plan),
        "targets": runtime["targets"], "deep": deep,
        "targets_physical_identity": {
            role: targets[role]["physical_identity"] for role in ("twin", "prod")
        },
    }


def verify_publish_environment(
        *, runtime: dict[str, Any],
        identity_fetch: IdentityFetch | None = None) -> dict[str, Any]:
    intent = os.environ.get("AW_CONTENT_PACK_PRESERVATION_INTENT", "").strip()
    if not intent:
        raise ContractError(
            "AW_CONTENT_PACK_PRESERVATION_INTENT is required for every armed "
            "content-pack apply")
    registry = os.environ.get(
        "AW_CONTENT_PACK_PRESERVATION_PROFILE_REGISTRY") or None
    root = os.environ.get("AW_ARTIFACT_ROOT") or None
    if os.environ.get("AW_CONTENT_PACK_PRESERVATION_DEEP") != "1":
        raise ContractError(
            "AW_CONTENT_PACK_PRESERVATION_DEEP=1 is required for every "
            "armed content-pack apply")
    return verify_intent(
        intent, runtime=runtime, registry_path=registry,
        artifact_root=root, deep=True, identity_fetch=identity_fetch)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--runtime", required=True,
                        help="JSON file containing the exact armed runtime binding")
    parser.add_argument("--registry")
    parser.add_argument("--artifact-root")
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args(argv)
    try:
        proof = verify_intent(
            args.intent, runtime=full._load_object(Path(args.runtime), "runtime"),
            registry_path=args.registry, artifact_root=args.artifact_root,
            deep=args.deep)
    except ContractError as exc:
        print(f"CONTENT-PACK PRESERVATION FAIL: {exc}", file=__import__("sys").stderr)
        return 1
    print(json.dumps(proof, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
