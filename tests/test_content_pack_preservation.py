from __future__ import annotations

import urllib.request
import hashlib
import json
from pathlib import Path

import pytest

import cytir.index.content_pack_preservation as cp
from cytir.index import preservation_contract as full
from test_preservation_contract import valid_manifest


REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data" / "source-preservation-profiles.v1.json"

# Fixture-standard target URLs (see intent_fixture) mapped to distinct fake
# physical identities, so every existing test exercises the task-447
# physical-distinctness check without touching the network.
FIXTURE_IDENTITIES = {
    "http://twin:6333": {"id": "fixture-twin-id",
                         "startup": "2020-01-01T00:00:00.000Z"},
    "http://prod:6333": {"id": "fixture-prod-id",
                         "startup": "2020-01-01T00:00:01.000Z"},
}


def _fixture_identity_fetch(url: str) -> dict:
    try:
        return dict(FIXTURE_IDENTITIES[url])
    except KeyError:
        raise cp.ContractError(f"no fixture identity registered for {url!r}")


# Captured before the autouse fixture below patches cp._fetch_identity_default,
# so the two tests that exercise the REAL default implementation still can.
REAL_FETCH_IDENTITY_DEFAULT = cp._fetch_identity_default


@pytest.fixture(autouse=True)
def _fake_qdrant_identity(monkeypatch):
    """Never hit the network from this suite: stand in for the live
    ``GET /telemetry`` probe with fixed, distinct identities for the two
    fixture target URLs."""
    monkeypatch.setattr(cp, "_fetch_identity_default", _fixture_identity_fetch)
    yield


def _put(root: Path, name: str, body: bytes) -> dict:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {"uri": f"artifact://{name}", "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest()}


def _json_artifact(root: Path, name: str, value: dict) -> dict:
    return _put(
        root, name,
        json.dumps(value, sort_keys=True).encode() + b"\n")


def _plan(tombstone_sha256: str) -> dict:
    return {
        "before": {"status": "green", "optimizer_status": "ok",
                   "points_count": 10, "update_queue_length": 0},
        "rows": 3, "existing_ids": 2, "new_ids": 1,
        "point_identical": 1, "point_noops": 1, "point_changed": 1,
        "point_writes": 2,
        "comparison": {
            "rows": 3, "existing": 2, "missing": 1,
            "identical": 1, "changed": 1,
            "artifact_points_sha256": "1" * 64,
            "existing_points_sha256": "2" * 64,
            "identical_points_sha256": "3" * 64,
            "write_points_sha256": "4" * 64,
            "decisions_sha256": "5" * 64,
        },
        "superseded": 1, "aged_tombstones": 0,
        "tombstone_pointer_sha256": tombstone_sha256,
        "tombstone_predecessor_proof": {
            "rows": 1, "fingerprint_digest": "6" * 64},
        "planned_topology_mutations": 3,
        "expected_delta": 0,
    }


def intent_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    full_manifest, root = valid_manifest(tmp_path)
    full_value = json.loads(full_manifest.read_text())
    global_ref = _json_artifact(root, "global/rebuild.json", {
        "schema": "aw.preservation-manifest.v1", "status": "blocked"})
    artifact = _put(root, "pack/points.jsonl", b"three sealed vectors\n")
    staged = _json_artifact(root, "pack/staged.json", {
        "status": "green", "points_count": 3})
    tombstones = _json_artifact(root, "pack/tombstones.json", {
        "schema": "aw.explicit-tombstone-pointers.v1",
        "artifact_sha256": artifact["sha256"],
        "counts": {"source_group": 1, "aged_out": 0}})
    plan = _plan(tombstones["sha256"])
    membership_evidence = _put(
        root, "pack/vector-membership.evidence.jsonl",
        b"three exact point ids agree\n")
    membership = _json_artifact(root, "pack/vector-membership.json", {
        "schema": cp.VECTOR_MEMBERSHIP_SCHEMA, "status": "PASS",
        "artifact_sha256": artifact["sha256"],
        "staged_verification_sha256": staged["sha256"],
        "rows": 3, "source_vector_records": 3,
        "artifact_point_ids_sha256": "7" * 64,
        "source_vector_point_ids_sha256": "7" * 64,
        "verification_evidence": membership_evidence,
    })
    touched_evidence = _put(
        root, "pack/touched-groups.evidence.jsonl", b"one exact group\n")
    touched = _json_artifact(root, "pack/touched-groups.json", {
        "schema": cp.TOUCHED_GROUPS_SCHEMA, "status": "PASS",
        "artifact_sha256": artifact["sha256"],
        "tombstone_sha256": tombstones["sha256"],
        "mutation_plan_sha256": cp.canonical_sha256(plan),
        "touched_groups": 1, "predecessor_rows": 1,
        "verification_evidence": touched_evidence,
    })
    baseline_evidence = _put(
        root, "baseline/exact-membership.evidence.jsonl",
        b"ten identical logical point ids\n")
    baseline_twin = _json_artifact(root, "baseline/twin.json", {
        "schema": cp.BASELINE_MEMBERSHIP_SCHEMA, "status": "PASS",
        "role": "twin", "collection": "aw_e3", "points": 10,
        "logical_membership_sha256": "8" * 64,
        "verification_evidence": baseline_evidence,
    })
    baseline_prod = _json_artifact(root, "baseline/prod.json", {
        "schema": cp.BASELINE_MEMBERSHIP_SCHEMA, "status": "PASS",
        "role": "prod", "collection": "aw_prod", "points": 10,
        "logical_membership_sha256": "8" * 64,
        "verification_evidence": baseline_evidence,
    })
    baseline_twin["rows"] = baseline_prod["rows"] = 10
    targets = {
        "twin": {"role": "twin", "url": "http://twin:6333",
                 "collection": "aw_e3"},
        "prod": {"role": "prod", "url": "http://prod:6333",
                 "collection": "aw_prod"},
    }
    intent_targets = {
        role: {
            **binding,
            "baseline": {
                "points": 10,
                "membership_receipt": (baseline_twin if role == "twin"
                                       else baseline_prod),
                "logical_membership_sha256": "8" * 64,
            },
            "expected_after_points": 10,
        }
        for role, binding in targets.items()
    }
    intent = {
        "schema": cp.SCHEMA, "status": "sealed", "scope": cp.SCOPE,
        "does_not_supersede_global_rebuild_manifest": True,
        "intent_id": "test-pack-intent", "git_sha": "a" * 40,
        "global_rebuild_manifest": {
            "schema": "aw.preservation-manifest.v1", "status": "blocked",
            "artifact": global_ref,
        },
        "profile_registry": {
            "path": "unused.json",
            "sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
        },
        "pack": {
            "name": "test-pack", "rows": 3, "artifact": artifact,
            "staged_verification": staged,
            "tombstones": {"mode": "explicit", "artifact": tombstones,
                           "source_group": 1, "aged_out": 0},
        },
        "mutation_plan": plan,
        "targets": intent_targets,
        "sources": full_value["sources"],
        "pack_vector_membership_receipt": membership,
        "touched_groups_receipt": touched,
        "terminal_verification": cp.TERMINAL_REQUIREMENTS,
    }
    path = tmp_path / "intent.json"
    path.write_text(json.dumps(intent, indent=2) + "\n")
    runtime = {
        "git_sha": "a" * 40,
        "global_rebuild_manifest": {
            "uri": "runtime://data/production-rebuild-lock.v1.json",
            "bytes": global_ref["bytes"],
            "sha256": global_ref["sha256"],
        },
        "pack": {
            "name": "test-pack", "rows": 3,
            "artifact": {"uri": "runtime://artifact",
                         "bytes": artifact["bytes"],
                         "sha256": artifact["sha256"]},
            "staged_verification": {
                "uri": "runtime://staged", "bytes": staged["bytes"],
                "sha256": staged["sha256"]},
            "tombstone_artifact": {
                "uri": "runtime://tombstones", "bytes": tombstones["bytes"],
                "sha256": tombstones["sha256"]},
        },
        "targets": targets,
        "mutation_plan": plan,
    }
    return path, root, runtime


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def test_valid_intent_passes_shallow_and_deep(tmp_path):
    intent, root, runtime = intent_fixture(tmp_path)
    shallow = cp.verify_intent(
        intent, runtime=runtime, registry_path=REGISTRY)
    deep = cp.verify_intent(
        intent, runtime=runtime, registry_path=REGISTRY,
        artifact_root=root, deep=True)
    assert shallow["status"] == deep["status"] == "PASS"
    assert deep["scope"] == cp.SCOPE
    assert deep["global_rebuild_status"] == "blocked"
    assert deep["targets"]["twin"]["collection"] == "aw_e3"
    assert deep["targets"]["prod"]["collection"] == "aw_prod"


@pytest.mark.parametrize(("field", "value", "match"), [
    ("schema", "aw.preservation-manifest.v1", "intent schema"),
    ("scope", "full-release", "intent scope"),
    ("does_not_supersede_global_rebuild_manifest", False, "must not supersede"),
])
def test_full_release_or_weakened_scope_cannot_be_used_as_content_intent(
        tmp_path, field, value, match):
    intent, _, runtime = intent_fixture(tmp_path)
    body = _load(intent); body[field] = value; _save(intent, body)
    with pytest.raises(cp.ContractError, match=match):
        cp.verify_intent(intent, runtime=runtime, registry_path=REGISTRY)


def test_full_release_verifier_rejects_incremental_intent(tmp_path):
    intent, _, _ = intent_fixture(tmp_path)
    with pytest.raises(full.ContractError, match="manifest schema"):
        full.verify_manifest(intent, registry_path=REGISTRY)


def test_global_rebuild_manifest_must_remain_blocked(tmp_path):
    intent, _, runtime = intent_fixture(tmp_path)
    body = _load(intent)
    body["global_rebuild_manifest"]["status"] = "sealed"
    _save(intent, body)
    with pytest.raises(cp.ContractError, match="remain explicitly blocked"):
        cp.verify_intent(intent, runtime=runtime, registry_path=REGISTRY)


def test_intent_git_sha_must_match_running_release(tmp_path):
    intent, _, runtime = intent_fixture(tmp_path)
    runtime["git_sha"] = "b" * 40
    with pytest.raises(cp.ContractError, match="running immutable release"):
        cp.verify_intent(intent, runtime=runtime, registry_path=REGISTRY)


def test_global_manifest_hash_must_match_running_release(tmp_path):
    intent, _, runtime = intent_fixture(tmp_path)
    runtime["global_rebuild_manifest"] = dict(
        runtime["global_rebuild_manifest"])
    runtime["global_rebuild_manifest"]["sha256"] = "f" * 64
    with pytest.raises(cp.ContractError, match="running release"):
        cp.verify_intent(intent, runtime=runtime, registry_path=REGISTRY)


def test_deep_verification_rejects_a_sealed_global_manifest_behind_blocked_label(
        tmp_path):
    intent, root, runtime = intent_fixture(tmp_path)
    body = _load(intent)
    ref = body["global_rebuild_manifest"]["artifact"]
    global_path = root / ref["uri"].removeprefix("artifact://")
    changed = json.dumps({
        "schema": "aw.preservation-manifest.v1", "status": "sealed"
    }, sort_keys=True).encode() + b"\n"
    global_path.write_bytes(changed)
    ref["bytes"] = len(changed)
    ref["sha256"] = hashlib.sha256(changed).hexdigest()
    runtime["global_rebuild_manifest"]["bytes"] = ref["bytes"]
    runtime["global_rebuild_manifest"]["sha256"] = ref["sha256"]
    _save(intent, body)
    with pytest.raises(cp.ContractError, match="is not blocked"):
        cp.verify_intent(
            intent, runtime=runtime, registry_path=REGISTRY,
            artifact_root=root, deep=True)


def test_live_plan_drift_fails_even_when_counts_still_balance(tmp_path):
    intent, _, runtime = intent_fixture(tmp_path)
    runtime["mutation_plan"] = dict(runtime["mutation_plan"])
    runtime["mutation_plan"]["comparison"] = dict(
        runtime["mutation_plan"]["comparison"])
    runtime["mutation_plan"]["comparison"]["decisions_sha256"] = "9" * 64
    with pytest.raises(cp.ContractError, match="live preflight"):
        cp.verify_intent(intent, runtime=runtime, registry_path=REGISTRY)


def test_target_collection_drift_fails(tmp_path):
    intent, _, runtime = intent_fixture(tmp_path)
    runtime["targets"] = {role: dict(value)
                          for role, value in runtime["targets"].items()}
    runtime["targets"]["prod"]["collection"] = "aw_e3"
    with pytest.raises(cp.ContractError, match="targets do not match"):
        cp.verify_intent(intent, runtime=runtime, registry_path=REGISTRY)


def test_target_baseline_membership_drift_fails(tmp_path):
    intent, _, runtime = intent_fixture(tmp_path)
    body = _load(intent)
    body["targets"]["prod"]["baseline"][
        "logical_membership_sha256"] = "9" * 64
    _save(intent, body)
    with pytest.raises(cp.ContractError, match="memberships differ"):
        cp.verify_intent(intent, runtime=runtime, registry_path=REGISTRY)


def test_source_vectors_must_conserve_to_pack_rows(tmp_path):
    intent, _, runtime = intent_fixture(tmp_path)
    body = _load(intent)
    body["sources"][0]["stages"]["vectors"]["records"] = 2
    body["sources"][0]["lineage"]["chunks_to_vectors"]["output_records"] = 2
    _save(intent, body)
    with pytest.raises(cp.ContractError, match="source vector records=2"):
        cp.verify_intent(intent, runtime=runtime, registry_path=REGISTRY)


def test_terminal_requirements_cannot_be_weakened(tmp_path):
    intent, _, runtime = intent_fixture(tmp_path)
    body = _load(intent)
    body["terminal_verification"]["tombstones_absent"] = False
    _save(intent, body)
    with pytest.raises(cp.ContractError, match="requirements were weakened"):
        cp.verify_intent(intent, runtime=runtime, registry_path=REGISTRY)


def test_deep_verification_rejects_touched_group_receipt_tamper(tmp_path):
    intent, root, runtime = intent_fixture(tmp_path)
    body = _load(intent)
    ref = body["touched_groups_receipt"]
    receipt_path = root / ref["uri"].removeprefix("artifact://")
    receipt = json.loads(receipt_path.read_text())
    receipt["mutation_plan_sha256"] = "0" * 64
    changed = json.dumps(receipt, sort_keys=True).encode() + b"\n"
    receipt_path.write_bytes(changed)
    ref["bytes"] = len(changed)
    ref["sha256"] = hashlib.sha256(changed).hexdigest()
    _save(intent, body)
    with pytest.raises(cp.ContractError, match="does not bind the mutation"):
        cp.verify_intent(
            intent, runtime=runtime, registry_path=REGISTRY,
            artifact_root=root, deep=True)


def test_armed_environment_requires_distinct_intent_variable(
        monkeypatch):
    monkeypatch.delenv("AW_CONTENT_PACK_PRESERVATION_INTENT", raising=False)
    monkeypatch.setenv("AW_PRESERVATION_MANIFEST", "full-release.json")
    with pytest.raises(cp.ContractError, match="CONTENT_PACK_PRESERVATION_INTENT"):
        cp.verify_publish_environment(runtime={})


def test_armed_environment_refuses_shallow_verification(monkeypatch):
    monkeypatch.setenv("AW_CONTENT_PACK_PRESERVATION_INTENT", "intent.json")
    monkeypatch.delenv("AW_CONTENT_PACK_PRESERVATION_DEEP", raising=False)
    with pytest.raises(cp.ContractError, match="PRESERVATION_DEEP=1 is required"):
        cp.verify_publish_environment(runtime={})


def test_armed_environment_refuses_false_deep_value(monkeypatch):
    monkeypatch.setenv("AW_CONTENT_PACK_PRESERVATION_INTENT", "intent.json")
    monkeypatch.setenv("AW_CONTENT_PACK_PRESERVATION_DEEP", "0")
    with pytest.raises(cp.ContractError, match="PRESERVATION_DEEP=1 is required"):
        cp.verify_publish_environment(runtime={})


# --- Task 447 / ADR-098: physical (not textual) twin/prod distinctness ----
#
# task 444: port 6334 on the build box was an SSH tunnel to production while scripts
# believed it was the twin. `targets.twin.url != targets.prod.url` (the
# textual check just above this block, content_pack_preservation.py ~line
# 209) is defeated by a tunnel — both URLs are still textually distinct
# strings. These tests exercise the live-identity probe layered on top.

TWIN_TARGET = {"role": "twin", "url": "http://twin:6333", "collection": "aw_e3"}
PROD_TARGET = {"role": "prod", "url": "http://prod:6333", "collection": "aw_prod"}


def test_physically_distinct_targets_pass_through_identities():
    identities = cp._require_physically_distinct_targets(
        {"twin": TWIN_TARGET, "prod": PROD_TARGET},
        identity_fetch=_fixture_identity_fetch)
    assert identities["twin"]["id"] != identities["prod"]["id"]
    assert identities == {
        "twin": FIXTURE_IDENTITIES["http://twin:6333"],
        "prod": FIXTURE_IDENTITIES["http://prod:6333"],
    }


def test_physically_distinct_targets_refuses_matching_ids():
    same_id = {"id": "same-process-id", "startup": "2026-08-27T00:00:00Z"}

    def fetch(url):
        return dict(same_id)

    with pytest.raises(cp.ContractError, match="SAME physical"):
        cp._require_physically_distinct_targets(
            {"twin": TWIN_TARGET, "prod": PROD_TARGET}, identity_fetch=fetch)


@pytest.mark.parametrize("broken_identity", [
    {},
    {"id": "only-an-id"},
    {"startup": "only-a-startup"},
    {"id": "", "startup": "2026-08-27T00:00:00Z"},
    {"id": "an-id", "startup": ""},
    {"id": None, "startup": "2026-08-27T00:00:00Z"},
    "not-a-dict",
])
def test_physically_distinct_targets_refuses_incomplete_identity(broken_identity):
    def fetch(url):
        return {"id": "prod-id", "startup": "2026-08-27T00:00:01Z"} \
            if url == PROD_TARGET["url"] else broken_identity

    with pytest.raises(cp.ContractError, match="incomplete identity"):
        cp._require_physically_distinct_targets(
            {"twin": TWIN_TARGET, "prod": PROD_TARGET}, identity_fetch=fetch)


def test_physically_distinct_targets_refuses_when_endpoint_probe_raises():
    def fetch(url):
        if url == TWIN_TARGET["url"]:
            raise cp.ContractError(f"physical identity probe failed for {url}")
        return {"id": "prod-id", "startup": "2026-08-27T00:00:01Z"}

    with pytest.raises(cp.ContractError, match="probe failed"):
        cp._require_physically_distinct_targets(
            {"twin": TWIN_TARGET, "prod": PROD_TARGET}, identity_fetch=fetch)


def test_default_identity_fetch_raises_on_unreachable_endpoint():
    # No mock here — this is the one test of the real network path. It must
    # fail closed (ContractError), never silently pass or hang.
    with pytest.raises(cp.ContractError, match="physical identity probe failed"):
        REAL_FETCH_IDENTITY_DEFAULT("http://127.0.0.1:1")


def test_default_identity_fetch_parses_telemetry_shape(monkeypatch):
    import io
    import json as _json

    body = _json.dumps({"result": {
        "id": "abc-123", "app": {"startup": "2026-08-27T19:25:14.990Z"}}}).encode()

    class _FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        assert request.full_url.endswith("/telemetry")
        return _FakeResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    identity = REAL_FETCH_IDENTITY_DEFAULT("http://twin:6444/")
    assert identity == {"id": "abc-123", "startup": "2026-08-27T19:25:14.990Z"}


def test_validate_targets_wires_in_physical_check_and_refuses_same_id(tmp_path):
    intent, _, runtime = intent_fixture(tmp_path)

    def same_id_fetch(url):
        return {"id": "collapsed-tunnel-id", "startup": "2026-08-27T00:00:00Z"}

    with pytest.raises(cp.ContractError, match="SAME physical"):
        cp.verify_intent(
            intent, runtime=runtime, registry_path=REGISTRY,
            identity_fetch=same_id_fetch)


def test_valid_intent_carries_physical_identity_evidence(tmp_path):
    intent, _, runtime = intent_fixture(tmp_path)
    proof = cp.verify_intent(intent, runtime=runtime, registry_path=REGISTRY)
    evidence = proof["targets_physical_identity"]
    assert evidence["twin"]["id"] != evidence["prod"]["id"]
    assert evidence["twin"] == FIXTURE_IDENTITIES["http://twin:6333"]
    assert evidence["prod"] == FIXTURE_IDENTITIES["http://prod:6333"]


def test_verify_publish_environment_threads_identity_fetch(
        tmp_path, monkeypatch):
    intent, root, runtime = intent_fixture(tmp_path)
    monkeypatch.setenv("AW_CONTENT_PACK_PRESERVATION_INTENT", str(intent))
    monkeypatch.setenv("AW_CONTENT_PACK_PRESERVATION_DEEP", "1")
    # verify_publish_environment always verifies deep, which needs the real
    # artifact root and a resolvable registry (the fixture intent's
    # profile_registry.path, "unused.json", is not a real file on disk).
    monkeypatch.setenv("AW_ARTIFACT_ROOT", str(root))
    monkeypatch.setenv("AW_CONTENT_PACK_PRESERVATION_PROFILE_REGISTRY",
                       str(REGISTRY))

    def same_id_fetch(url):
        return {"id": "collapsed-tunnel-id", "startup": "2026-08-27T00:00:00Z"}

    with pytest.raises(cp.ContractError, match="SAME physical"):
        cp.verify_publish_environment(runtime=runtime, identity_fetch=same_id_fetch)
