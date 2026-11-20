#!/usr/bin/env python3
"""Seal explicit PASS evidence for every requirement in a source profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cytir.acquire.artifact_io import atomic_publish, sha256_file
from cytir.index.preservation_contract import (PROFILE_COMPLIANCE_RECEIPT_SCHEMA,
                                   PROFILE_SCHEMA, _validate_profile_compliance)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _artifact(root: Path, path: Path) -> dict:
    relative = path.resolve().relative_to(root.resolve())
    return {"uri": f"artifact://{relative.as_posix()}",
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def seal(*, artifact_root: Path, registry_path: Path, profile_id: str,
         source_id: str, checks_path: Path, evidence_path: Path,
         receipt_path: Path) -> dict:
    registry = _load(registry_path)
    if registry.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"profile registry schema must be {PROFILE_SCHEMA!r}")
    profile = (registry.get("profiles") or {}).get(profile_id)
    if not isinstance(profile, dict):
        raise ValueError(f"unknown source profile: {profile_id}")
    checks = _load(checks_path)
    receipt = {
        "schema": PROFILE_COMPLIANCE_RECEIPT_SCHEMA,
        "source_id": source_id,
        "profile": profile_id,
        "profile_registry_sha256": sha256_file(registry_path),
        "verification_evidence": _artifact(artifact_root, evidence_path),
        "checks": checks,
    }
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    if receipt_path.exists():
        if receipt_path.read_bytes() != encoded:
            raise ValueError("immutable profile-compliance receipt collision")
    else:
        atomic_publish(receipt_path, encoded)
    envelope = {"receipt": _artifact(artifact_root, receipt_path)}
    _validate_profile_compliance(
        envelope, "profile_compliance", source_id=source_id,
        profile_id=profile_id, profile=profile,
        registry_sha256=sha256_file(registry_path),
        artifact_root=artifact_root, deep=True)
    return envelope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--profile-registry", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--verification-evidence", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    value = seal(
        artifact_root=args.artifact_root.resolve(),
        registry_path=args.profile_registry.resolve(), profile_id=args.profile,
        source_id=args.source_id, checks_path=args.checks.resolve(),
        evidence_path=args.verification_evidence.resolve(),
        receipt_path=args.receipt.resolve())
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
