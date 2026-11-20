#!/usr/bin/env python3
"""Seal the tier-2 content-pack preservation intent (ADR-098 Ruling 1).

`content_pack_preservation.verify_intent` is the authority an armed
incremental apply must satisfy, and `apply_content_pack_shared.py` already
injects it as its preservation verifier. What never existed was a producer for
the document it verifies (task 449 §6.4): the sealed intent itself, the pack
vector-membership receipt and the touched-groups receipt. This module is that
producer.

It manufactures nothing. Every number it writes is read back out of an
artifact that already exists:

* the pack's rows and point ids come from the pack artifact;
* the sources' vector records come from each arm's sealed `vectors` stage —
  a *different* artifact, produced by a different reader, so the membership
  receipt's `source_vector_point_ids_sha256 == artifact_point_ids_sha256`
  is a real cross-check between the staged collection read-back and the
  materialized pack rather than a restatement of one of them;
* the mutation plan is the live read-only preflight's own plan, verbatim, and
  the armed run re-derives it and refuses on any drift;
* the baseline membership receipts are the mirrors' own enumerations;
* the global rebuild manifest is bound by hash and must still say `blocked`.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

from cytir.acquire.artifact_io import atomic_publish, sha256_file
from cytir.index import content_pack_preservation as CP
from cytir.index import preservation_contract as full
from cytir.index import seal_collection_membership as MEMBERSHIP
from cytir.index.supersede import payload_group_key


class IntentError(RuntimeError):
    """The pack cannot be presented as an authorized incremental mutation."""


def _artifact(root: Path, path: Path) -> dict:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise IntentError(
            f"{path} is outside the portable artifact root {root}") from exc
    return {"uri": f"artifact://{relative.as_posix()}",
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _publish(path: Path, value: dict) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise IntentError(f"immutable collision: {path}")
        return
    atomic_publish(path, encoded)


def _rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def id_set_digest(ids) -> tuple[str, int]:
    """The membership digest rule, shared with `seal_collection_membership`."""
    ordered = sorted(ids)
    if len(ordered) != len(set(ordered)):
        raise IntentError("point ids are not a set")
    digest = hashlib.sha256()
    for point_id in ordered:
        digest.update(f"{point_id}\n".encode())
    return digest.hexdigest(), len(ordered)


def read_pack(artifact: Path) -> dict:
    """Point ids and touched supersession groups, straight from the pack."""
    ids, groups = [], set()
    for row in _rows(artifact):
        payload = row.get("payload") or {}
        group = payload_group_key(payload)
        if not group:
            raise IntentError(f"pack point {row.get('id')} has no groupable URL")
        groups.add(group)
        ids.append(str(row["id"]))
    digest, count = id_set_digest(ids)
    if not count:
        raise IntentError("pack artifact is empty")
    return {"rows": count, "point_ids_sha256": digest,
            "touched_groups": len(groups)}


def read_source_vectors(artifact_root: Path, sources: list[dict]) -> dict:
    """Point ids the sources' sealed `vectors` stages actually carry."""
    ids: list[str] = []
    per_source = {}
    for source in sources:
        stage = full._validate_stage(
            source["stages"]["vectors"], f"{source['source_id']}.vectors",
            artifact_root=artifact_root, deep=True)
        receipt = json.loads((artifact_root / stage["receipt"]["uri"].removeprefix(
            "artifact://")).read_text())
        rows = 0
        for entry in receipt["artifacts"]:
            path = artifact_root / entry["uri"].removeprefix("artifact://")
            if path.name.endswith(".jsonl.gz"):
                for row in _rows(path):
                    ids.append(str(row["point_id"]))
                    rows += 1
        if rows != stage["records"]:
            raise IntentError(
                f"{source['source_id']} vectors stage declares {stage['records']} "
                f"records but its artifacts carry {rows}")
        per_source[source["source_id"]] = rows
    digest, count = id_set_digest(ids)
    return {"rows": count, "point_ids_sha256": digest, "per_source": per_source}


def seal(*, artifact_root: Path, out_root: Path, intent_id: str,
         release_dir: Path, registry_path: Path, pack_name: str,
         pack_artifact: Path, staged_verification: Path,
         tombstone_pointers: Path | None, mutation_plan: dict,
         targets: dict, sources: list[dict], intent_path: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    git_sha = (release_dir / "GIT_SHA").read_text(encoding="utf-8").strip()
    if not full.GIT_SHA_RE.fullmatch(git_sha):
        raise IntentError("running release GIT_SHA must be lowercase 40-hex")

    # The blocked global manifest and the profile registry are copied into the
    # portable tree: a clone must be able to resolve every artifact:// the
    # intent names without also having the release checked out.
    global_source = release_dir / "data" / "production-rebuild-lock.v1.json"
    global_copy = out_root / "production-rebuild-lock.v1.json"
    registry_copy = out_root / "source-preservation-profiles.v1.json"
    for source_path, copy in ((global_source, global_copy),
                              (registry_path, registry_copy)):
        if copy.exists():
            if sha256_file(copy) != sha256_file(source_path):
                raise IntentError(f"immutable collision: {copy}")
        else:
            shutil.copyfile(source_path, copy)
    global_body = json.loads(global_copy.read_text())
    if (global_body.get("schema") != full.SCHEMA or
            global_body.get("status") != "blocked"):
        raise IntentError("global rebuild manifest must remain explicitly blocked")

    pack = read_pack(pack_artifact)
    vectors = read_source_vectors(artifact_root, sources)
    if pack["point_ids_sha256"] != vectors["point_ids_sha256"]:
        raise IntentError(
            "the pack's point ids and the sources' sealed vector stages "
            f"disagree ({pack['rows']} pack rows vs {vectors['rows']} vector "
            "records); the pack is not exactly what the lane staged")
    rows = pack["rows"]
    if mutation_plan.get("rows") != rows:
        raise IntentError(
            f"live mutation plan is for {mutation_plan.get('rows')} rows, "
            f"pack carries {rows}")

    artifact_ref = _artifact(artifact_root, pack_artifact)
    staged_ref = _artifact(artifact_root, staged_verification)
    tombstone_ref = (_artifact(artifact_root, tombstone_pointers)
                     if tombstone_pointers else None)
    if tombstone_ref is not None:
        pointers = json.loads(tombstone_pointers.read_text())
        if pointers.get("artifact_sha256") != artifact_ref["sha256"]:
            raise IntentError("tombstone pointers bind a different pack artifact")
        counts = pointers.get("counts") or {}
        source_group = int(counts.get("source_group", -1))
        aged_out = int(counts.get("aged_out", -1))
    else:
        source_group = aged_out = 0

    membership_evidence = out_root / "pack-vector-membership.evidence.json"
    _publish(membership_evidence, {
        "schema": "aw.content-pack-vector-membership-evidence.v1",
        "digest_rule": MEMBERSHIP.DIGEST_RULE,
        "pack_artifact_sha256": artifact_ref["sha256"],
        "pack_rows": pack["rows"],
        "source_vector_records": vectors["rows"],
        "per_source_vector_records": vectors["per_source"],
        "point_ids_sha256": pack["point_ids_sha256"]})
    membership_path = out_root / "pack-vector-membership.json"
    _publish(membership_path, {
        "schema": CP.VECTOR_MEMBERSHIP_SCHEMA, "status": "PASS",
        "artifact_sha256": artifact_ref["sha256"],
        "staged_verification_sha256": staged_ref["sha256"],
        "rows": rows, "source_vector_records": vectors["rows"],
        "artifact_point_ids_sha256": pack["point_ids_sha256"],
        "source_vector_point_ids_sha256": vectors["point_ids_sha256"],
        "verification_evidence": _artifact(artifact_root, membership_evidence)})

    predecessor_rows = mutation_plan["superseded"] + mutation_plan["aged_tombstones"]
    touched_evidence = out_root / "touched-groups.evidence.json"
    _publish(touched_evidence, {
        "schema": "aw.content-pack-touched-groups-evidence.v1",
        # A frozen description written into the sealed intent (its bytes are hashed), not a path.
        # Declared in `cytir.ops.public_cut.FROZEN_RECEIPT_LABELS`.
        "group_key": "pipeline/upsert_supersede.group_key over the pack payloads",
        "pack_artifact_sha256": artifact_ref["sha256"],
        "touched_groups": pack["touched_groups"],
        "superseded": mutation_plan["superseded"],
        "aged_tombstones": mutation_plan["aged_tombstones"],
        "predecessor_rows": predecessor_rows})
    touched_path = out_root / "touched-groups.json"
    _publish(touched_path, {
        "schema": CP.TOUCHED_GROUPS_SCHEMA, "status": "PASS",
        "artifact_sha256": artifact_ref["sha256"],
        "tombstone_sha256": tombstone_ref["sha256"] if tombstone_ref else None,
        "mutation_plan_sha256": CP.canonical_sha256(mutation_plan),
        "touched_groups": pack["touched_groups"],
        "predecessor_rows": predecessor_rows,
        "verification_evidence": _artifact(artifact_root, touched_evidence)})

    before = mutation_plan["before"]["points_count"]
    after = before + mutation_plan["expected_delta"]
    intent_targets = {}
    for role in ("twin", "prod"):
        target = targets[role]
        receipt = target["baseline"]
        if receipt["points"] != before:
            raise IntentError(
                f"{role} baseline membership was sealed at {receipt['points']} "
                f"points but the live plan starts at {before}")
        intent_targets[role] = {
            "role": role, "url": target["url"].rstrip("/"),
            "collection": target["collection"],
            "baseline": {
                "points": before,
                "membership_receipt": {
                    "uri": receipt["uri"], "bytes": receipt["bytes"],
                    "sha256": receipt["sha256"], "rows": receipt["rows"]},
                "logical_membership_sha256":
                    receipt["logical_membership_sha256"]},
            "expected_after_points": after}

    intent = {
        "schema": CP.SCHEMA, "status": "sealed", "scope": CP.SCOPE,
        "does_not_supersede_global_rebuild_manifest": True,
        "intent_id": intent_id, "git_sha": git_sha,
        "global_rebuild_manifest": {
            "schema": full.SCHEMA, "status": "blocked",
            "artifact": _artifact(artifact_root, global_copy)},
        "profile_registry": {
            "path": _relative_to(registry_copy, intent_path.parent),
            "sha256": sha256_file(registry_copy)},
        "pack": {
            "name": pack_name, "rows": rows, "artifact": artifact_ref,
            "staged_verification": staged_ref,
            "tombstones": ({"mode": "explicit", "artifact": tombstone_ref,
                            "source_group": source_group, "aged_out": aged_out}
                           if tombstone_ref else
                           {"mode": "none", "source_group": 0, "aged_out": 0})},
        "mutation_plan": mutation_plan,
        "targets": intent_targets,
        "sources": sources,
        "pack_vector_membership_receipt": _artifact(artifact_root, membership_path),
        "touched_groups_receipt": _artifact(artifact_root, touched_path),
        "terminal_verification": CP.TERMINAL_REQUIREMENTS,
    }
    _publish(intent_path, intent)
    return {"intent": str(intent_path), "intent_sha256": sha256_file(intent_path),
            "rows": rows, "touched_groups": pack["touched_groups"],
            "point_ids_sha256": pack["point_ids_sha256"],
            "expected_after_points": after}


def _relative_to(path: Path, base: Path) -> str:
    import os
    return os.path.relpath(path.resolve(), base.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="directory for the receipts this producer writes")
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--intent-id", required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--profile-registry", type=Path, required=True)
    parser.add_argument("--pack-name", required=True)
    parser.add_argument("--pack-artifact", type=Path, required=True)
    parser.add_argument("--staged-verification", type=Path, required=True)
    parser.add_argument("--tombstone-pointers", type=Path)
    parser.add_argument("--mutation-plan", type=Path, required=True,
                        help="the live read-only preflight's own plan")
    parser.add_argument("--target", action="append", required=True,
                        metavar="ROLE=URL,COLLECTION,BASELINE_JSON")
    parser.add_argument("--source", action="append", required=True,
                        metavar="SOURCE.json")
    args = parser.parse_args()
    targets = {}
    for item in args.target:
        role, _, rest = item.partition("=")
        url, collection, baseline = rest.split(",", 2)
        targets[role] = {"url": url, "collection": collection,
                         "baseline": json.loads(Path(baseline).read_text())}
    sources = [json.loads(Path(item).read_text())["source"]
               for item in args.source]
    try:
        result = seal(
            artifact_root=args.artifact_root.resolve(),
            out_root=args.out.resolve(), intent_id=args.intent_id,
            release_dir=args.release_dir.resolve(),
            registry_path=args.profile_registry.resolve(),
            pack_name=args.pack_name,
            pack_artifact=args.pack_artifact.resolve(),
            staged_verification=args.staged_verification.resolve(),
            tombstone_pointers=(args.tombstone_pointers.resolve()
                                if args.tombstone_pointers else None),
            mutation_plan=json.loads(args.mutation_plan.read_text()),
            targets=targets, sources=sources,
            intent_path=args.intent.resolve())
    except (IntentError, full.ContractError, OSError, KeyError,
            ValueError) as exc:
        print(f"CONTENT-PACK INTENT SEAL FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
