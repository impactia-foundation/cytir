from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import cytir.index.preservation_contract as pc


REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data" / "source-preservation-profiles.v1.json"


def _artifact(root: Path, name: str, body: bytes | None = None) -> dict:
    body = body if body is not None else (name + "\n").encode()
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {"uri": f"artifact://{name}", "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest()}


def valid_manifest(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "artifacts"
    stage_names = ("original", "markdown", "chunks", "outlinks", "vectors")
    rows = {"original": 2, "markdown": 2, "chunks": 3,
            "outlinks": 4, "vectors": 3}
    stages = {}
    for name in stage_names:
        data_artifact = _artifact(
            root, f"source/{name}/data.jsonl",
            body=(f"{name}-canonical-stage-bytes\n").encode())
        logical_sha256 = data_artifact["sha256"]
        receipt_body = json.dumps({
            "schema": pc.STAGE_RECEIPT_SCHEMA,
            "records": rows[name],
            "bytes": data_artifact["bytes"],
            "format": "jsonl",
            "logical_sha256": logical_sha256,
            "deterministic_encoding": True,
            "producer": {"id": f"example-producer/{name}", "version": "v1"},
            "artifacts": [data_artifact],
        }, sort_keys=True).encode() + b"\n"
        receipt = _artifact(
            root, f"source/{name}/receipt.json", body=receipt_body)
        stages[name] = {
            "records": rows[name], "bytes": data_artifact["bytes"],
            "format": "jsonl", "logical_sha256": logical_sha256,
            "deterministic_encoding": True,
            "producer": {"id": f"example-producer/{name}", "version": "v1"},
            "receipt": receipt,
        }
    fidelity_evidence = _artifact(
        root, "source/content-fidelity-evidence.json",
        body=b'{"status":"PASS","checked":"full-source-fixture"}\n')
    fidelity_body = json.dumps({
        "schema": pc.CONTENT_FIDELITY_RECEIPT_SCHEMA,
        "source_id": "test-html",
        "scope": pc.CONTENT_SCOPE,
        "original_bytes_complete": True,
        "markdown_content_complete": True,
        "source_text_truncation_applied": False,
        "original_logical_sha256": stages["original"]["logical_sha256"],
        "markdown_logical_sha256": stages["markdown"]["logical_sha256"],
        "adapter_source_sha256": "b" * 64,
        "verification_method": "fixture byte-conservation audit",
        "verification_evidence": fidelity_evidence,
    }, sort_keys=True).encode() + b"\n"
    fidelity_receipt = _artifact(
        root, "source/content-fidelity-receipt.json", body=fidelity_body)
    registry = json.loads(REGISTRY.read_text())
    profile = registry["profiles"]["web_html"]
    profile_evidence = _artifact(
        root, "source/profile-compliance-evidence.json",
        body=b'{"status":"PASS","source":"fixture"}\n')
    profile_compliance_body = json.dumps({
        "schema": pc.PROFILE_COMPLIANCE_RECEIPT_SCHEMA,
        "source_id": "test-html",
        "profile": "web_html",
        "profile_registry_sha256": pc.sha256_file(REGISTRY),
        "verification_evidence": profile_evidence,
        "checks": {
            stage: [{
                "requirement": requirement,
                "status": "PASS",
                "verification_method": "fixture assertion",
            } for requirement in profile.get(stage, {}).get("requirements", [])]
            for stage in profile["required_stages"]
        },
    }, sort_keys=True).encode() + b"\n"
    profile_compliance_receipt = _artifact(
        root, "source/profile-compliance-receipt.json",
        body=profile_compliance_body)
    manifest = {
        "schema": pc.SCHEMA,
        "status": "sealed",
        "release_id": "test-release",
        "git_sha": "a" * 40,
        "profile_registry": {
            "path": "unused-with-explicit-registry.json",
            "sha256": pc.sha256_file(REGISTRY),
        },
        "population": {
            "basis": "same-source-document-population",
            "baseline_collection": "aw_e3",
            "baseline_points": 34464401,
            "baseline_source_documents": 2,
            "resolved_source_documents": 2,
            "baseline_point_membership_receipt": _artifact(
                root, "population/baseline-points.json"),
            "baseline_source_inventory_receipt": _artifact(
                root, "population/baseline-sources.json"),
            "resolution_receipt": _artifact(
                root, "population/resolution.json"),
            "refresh_policy": {
                "mode": "allow-newer",
                "retrieval_metadata_required": True,
                "retrieved_original_bytes_preserved": True,
                "content_change_classification_required": True,
            },
        },
        "sources": [{
            "source_id": "test-html",
            "profile": "web_html",
            "configuration": _artifact(root, "source/configuration.yaml"),
            "stages": stages,
            "content_fidelity": {
                "scope": pc.CONTENT_SCOPE,
                "original_bytes_complete": True,
                "markdown_content_complete": True,
                "source_text_truncation_applied": False,
                "receipt": fidelity_receipt,
            },
            "profile_compliance": {"receipt": profile_compliance_receipt},
            "lineage": {
                "original_to_markdown": {
                    "input_records": 2, "emitted_parent_records": 2,
                    "quarantined_records": 0, "output_records": 2,
                },
                "markdown_to_chunks": {
                    "input_records": 2, "emitted_parent_records": 2,
                    "quarantined_records": 0, "output_records": 3,
                },
                "chunks_to_vectors": {
                    "input_records": 3, "emitted_parent_records": 3,
                    "quarantined_records": 0, "output_records": 3,
                },
            },
            "durability": {
                "status": "verified_offbox",
                "receipt": _artifact(root, "source/offbox-receipt.json"),
            },
        }],
        "collection": {
            "name": "aw_test", "points": 3,
            "membership_receipt": _artifact(root, "collection/membership.json"),
            "build_receipt": _artifact(root, "collection/build.json"),
        },
        "rebuild": {"commands": [
            {"phase": "rebuild", "argv": [
                "python3", "{repo}/tools/example-rebuild.py",
                "--root", "{artifact_root}"]},
            {"phase": "verify-membership", "argv": [
                "python3", "-m", "cytir.index.seal_collection_membership", "verify",
                "--collection", "{collection}"]},
        ]},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path, root


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def test_valid_manifest_passes_shallow_and_deep(tmp_path):
    manifest, root = valid_manifest(tmp_path)
    shallow = pc.verify_manifest(manifest, registry_path=REGISTRY,
                                 expect_collection="aw_test")
    deep = pc.verify_manifest(manifest, registry_path=REGISTRY,
                              artifact_root=root, deep=True)
    assert shallow["status"] == deep["status"] == "PASS"
    assert deep["sources"] == 1 and deep["points"] == 3
    assert deep["chunks"] == deep["vectors"] == 3


def test_missing_required_markdown_stage_fails(tmp_path):
    manifest, _ = valid_manifest(tmp_path)
    data = _load(manifest)
    del data["sources"][0]["stages"]["markdown"]
    _save(manifest, data)
    with pytest.raises(pc.ContractError, match="missing required stages: markdown"):
        pc.verify_manifest(manifest, registry_path=REGISTRY)


def test_missing_content_fidelity_fails(tmp_path):
    manifest, _ = valid_manifest(tmp_path)
    data = _load(manifest)
    del data["sources"][0]["content_fidelity"]
    _save(manifest, data)
    with pytest.raises(pc.ContractError, match="content_fidelity must be an object"):
        pc.verify_manifest(manifest, registry_path=REGISTRY)


def test_declared_source_text_truncation_fails(tmp_path):
    manifest, _ = valid_manifest(tmp_path)
    data = _load(manifest)
    data["sources"][0]["content_fidelity"]["source_text_truncation_applied"] = True
    _save(manifest, data)
    with pytest.raises(pc.ContractError, match="source_text_truncation_applied must be false"):
        pc.verify_manifest(manifest, registry_path=REGISTRY)


def test_missing_profile_compliance_fails(tmp_path):
    manifest, _ = valid_manifest(tmp_path)
    data = _load(manifest)
    del data["sources"][0]["profile_compliance"]
    _save(manifest, data)
    with pytest.raises(pc.ContractError, match="profile_compliance must be an object"):
        pc.verify_manifest(manifest, registry_path=REGISTRY)


def test_deep_profile_compliance_requires_every_requirement(tmp_path):
    manifest, root = valid_manifest(tmp_path)
    data = _load(manifest)
    receipt_ref = data["sources"][0]["profile_compliance"]["receipt"]
    receipt = root / receipt_ref["uri"].removeprefix("artifact://")
    audit = json.loads(receipt.read_text())
    audit["checks"]["markdown"].pop()
    body = json.dumps(audit, sort_keys=True).encode() + b"\n"
    receipt.write_bytes(body)
    receipt_ref["bytes"] = len(body)
    receipt_ref["sha256"] = hashlib.sha256(body).hexdigest()
    _save(manifest, data)
    with pytest.raises(pc.ContractError, match="preserve every profile requirement"):
        pc.verify_manifest(manifest, registry_path=REGISTRY,
                           artifact_root=root, deep=True)


def test_deep_content_fidelity_must_bind_markdown_digest(tmp_path):
    manifest, root = valid_manifest(tmp_path)
    data = _load(manifest)
    receipt_ref = data["sources"][0]["content_fidelity"]["receipt"]
    receipt = root / receipt_ref["uri"].removeprefix("artifact://")
    audit = json.loads(receipt.read_text())
    audit["markdown_logical_sha256"] = "c" * 64
    body = json.dumps(audit, sort_keys=True).encode() + b"\n"
    receipt.write_bytes(body)
    receipt_ref["bytes"] = len(body)
    receipt_ref["sha256"] = hashlib.sha256(body).hexdigest()
    _save(manifest, data)
    with pytest.raises(pc.ContractError, match="markdown_logical_sha256 mismatch"):
        pc.verify_manifest(manifest, registry_path=REGISTRY,
                           artifact_root=root, deep=True)


def test_lineage_conservation_failure_is_fatal(tmp_path):
    manifest, _ = valid_manifest(tmp_path)
    data = _load(manifest)
    data["sources"][0]["lineage"]["markdown_to_chunks"]["quarantined_records"] = 1
    _save(manifest, data)
    with pytest.raises(pc.ContractError, match="conservation failed"):
        pc.verify_manifest(manifest, registry_path=REGISTRY)


def test_quarantine_requires_receipt(tmp_path):
    manifest, _ = valid_manifest(tmp_path)
    data = _load(manifest)
    edge = data["sources"][0]["lineage"]["original_to_markdown"]
    edge["emitted_parent_records"] = 1
    edge["quarantined_records"] = 1
    data["sources"][0]["stages"]["markdown"]["records"] = 1
    edge["output_records"] = 1
    _save(manifest, data)
    with pytest.raises(pc.ContractError, match="requires quarantine_receipt"):
        pc.verify_manifest(manifest, registry_path=REGISTRY)


def test_host_local_artifact_path_is_forbidden(tmp_path):
    manifest, _ = valid_manifest(tmp_path)
    data = _load(manifest)
    data["sources"][0]["stages"]["original"]["receipt"]["uri"] = \
        "file:///var/lib/example/receipt.json"
    _save(manifest, data)
    with pytest.raises(pc.ContractError, match="must use artifact://"):
        pc.verify_manifest(manifest, registry_path=REGISTRY)


def test_deep_verification_rehashes_receipts(tmp_path):
    manifest, root = valid_manifest(tmp_path)
    receipt = root / "source" / "chunks" / "receipt.json"
    body = receipt.read_bytes()
    receipt.write_bytes(bytes([body[0] ^ 1]) + body[1:])
    with pytest.raises(pc.ContractError, match="artifact SHA mismatch"):
        pc.verify_manifest(manifest, registry_path=REGISTRY,
                           artifact_root=root, deep=True)


def test_deep_verification_rehashes_stage_data_behind_intact_receipt(tmp_path):
    manifest, root = valid_manifest(tmp_path)
    artifact = root / "source" / "chunks" / "data.jsonl"
    body = artifact.read_bytes()
    artifact.write_bytes(bytes([body[0] ^ 1]) + body[1:])
    with pytest.raises(pc.ContractError, match="artifact SHA mismatch"):
        pc.verify_manifest(manifest, registry_path=REGISTRY,
                           artifact_root=root, deep=True)


def test_deep_verification_binds_stage_producer_to_receipt(tmp_path):
    manifest, root = valid_manifest(tmp_path)
    data = _load(manifest)
    receipt_ref = data["sources"][0]["stages"]["chunks"]["receipt"]
    receipt_path = root / receipt_ref["uri"].removeprefix("artifact://")
    receipt = _load(receipt_path)
    receipt["producer"] = {"id": "example-producer/other", "version": "v1"}
    _save(receipt_path, receipt)
    body = receipt_path.read_bytes()
    receipt_ref["bytes"] = len(body)
    receipt_ref["sha256"] = hashlib.sha256(body).hexdigest()
    _save(manifest, data)
    with pytest.raises(pc.ContractError, match="receipt producer mismatch"):
        pc.verify_manifest(manifest, registry_path=REGISTRY,
                           artifact_root=root, deep=True)


def test_stage_requires_logical_digest_and_deterministic_encoding(tmp_path):
    manifest, _ = valid_manifest(tmp_path)
    data = _load(manifest)
    del data["sources"][0]["stages"]["markdown"]["logical_sha256"]
    _save(manifest, data)
    with pytest.raises(pc.ContractError, match="logical_sha256"):
        pc.verify_manifest(manifest, registry_path=REGISTRY)


def test_blocked_manifest_reports_named_gaps(tmp_path):
    manifest, _ = valid_manifest(tmp_path)
    data = _load(manifest)
    data["status"] = "blocked"
    data["blocking_gaps"] = ["missing L1"]
    _save(manifest, data)
    with pytest.raises(pc.ContractError, match="missing L1"):
        pc.verify_manifest(manifest, registry_path=REGISTRY)


def test_publish_environment_fails_closed_without_manifest(monkeypatch):
    monkeypatch.delenv("AW_PRESERVATION_MANIFEST", raising=False)
    with pytest.raises(pc.ContractError, match="required"):
        pc.verify_publish_environment()


def test_rebuild_recipe_must_end_in_exact_membership_verification(tmp_path):
    manifest, _ = valid_manifest(tmp_path)
    data = _load(manifest)
    data["rebuild"]["commands"].pop()
    _save(manifest, data)
    with pytest.raises(pc.ContractError, match="must end with phase"):
        pc.verify_manifest(manifest, registry_path=REGISTRY)


def test_rechunked_release_requires_complete_baseline_population(tmp_path):
    manifest, _ = valid_manifest(tmp_path)
    data = _load(manifest)
    data["population"]["resolved_source_documents"] = 1
    _save(manifest, data)
    with pytest.raises(pc.ContractError, match="population source/document conservation"):
        pc.verify_manifest(manifest, registry_path=REGISTRY)


def test_allow_newer_requires_auditable_preserved_retrieval(tmp_path):
    manifest, _ = valid_manifest(tmp_path)
    data = _load(manifest)
    data["population"]["refresh_policy"]["retrieved_original_bytes_preserved"] = False
    _save(manifest, data)
    with pytest.raises(pc.ContractError, match="retrieved_original_bytes_preserved"):
        pc.verify_manifest(manifest, registry_path=REGISTRY)
