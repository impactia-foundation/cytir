"""ADR-098 amendment 1 — `status: ruled`, and what it may not excuse.

Two populations of test, deliberately:

  * synthetic fixtures, so every refusal path is exercised with one field
    wrong at a time;
  * the REAL tracked manifests, so the thing the alias swap will actually
    read is the thing under test — `data/production-flip-ruling-aw_e8.v1.json`
    (the ruling) and `data/production-rebuild-lock.v1.json` (which must stay
    blocked, and must keep refusing).

The bar these tests hold: a ruling may lower what must be SUPPLIED, and may
never lower what a supplied block PROVES.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers.repo_files import repo_file_or_skip  # noqa: E402

from cytir.gates import flip_gate
from cytir.index import preservation_contract as pc
from test_preservation_contract import _artifact, valid_manifest


REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data" / "source-preservation-profiles.v1.json"
RULED_MANIFEST = REPO / "data" / "production-flip-ruling-aw_e8.v1.json"
RULED_MANIFEST_E9 = REPO / "data" / "production-flip-ruling-aw_e9.v1.json"
MANIFEST_E10 = REPO / "data" / "production-flip-preservation-aw_e10.v1.json"
BLOCKED_MANIFEST = REPO / "data" / "production-rebuild-lock.v1.json"
CONTRACT = REPO / "data" / "flip-gate-contract.v1.json"

#: the four production ruling/lock files the public cut withholds (release policy);
#: a test that reads one skips in the cut and runs in full in the operator repo (task 2462 precedent)
WITHHELD_RULING_FILES = (
    "data/production-flip-ruling-aw_e8.v1.json",
    "data/production-flip-ruling-aw_e9.v1.json",
    "data/production-rebuild-lock.v1.json",
    "data/production-flip-preservation-aw_e10.v1.json",
)


@pytest.fixture
def withheld_ruling_files():
    for rel in WITHHELD_RULING_FILES:
        repo_file_or_skip(rel)  # skips the requesting test when absent (the public cut)


CANDIDATE = "aw_e8"
CANDIDATE_POINTS = 33_723_480
INCUMBENT = "aw_e3"
INCUMBENT_POINTS = 34_464_401

# aw_e9's own ruled manifest (task 651): aw_e9 is the candidate, aw_e8 -- the
# LIVE serving collection, which stays resident on the box -- is its rollback
# target. Both counts are read from the manifest itself in the tests below
# rather than hand-typed here, because aw_e8 keeps growing under ordinary
# growth packs (626/629) and a hand-typed constant would go stale the way the
# 09-16 membership receipt already has.
CANDIDATE_E9 = "aw_e9"
INCUMBENT_E9 = "aw_e8"


# --------------------------------------------------------------------------
# synthetic fixture: the smallest manifest that `ruled` accepts
# --------------------------------------------------------------------------
def _membership_receipt(root: Path, name: str, *, collection: str,
                        rows: int, shards: int = 2) -> dict:
    per = rows // shards
    entries = []
    for i in range(shards):
        count = per if i < shards - 1 else rows - per * (shards - 1)
        entries.append({"file": f"membership-{i:05d}.jsonl.gz", "rows": count,
                        "bytes": 100 + i, "sha256": f"{i:064d}"})
    body = json.dumps({
        "schema": "aw.task416.qdrant-membership-receipt.v1",
        "collection": collection,
        "rows": rows,
        "with_payload": True,
        "with_vectors": False,
        "shards": entries,
        "total_bytes": sum(e["bytes"] for e in entries),
    }, sort_keys=True).encode() + b"\n"
    return _artifact(root, name, body=body)


def ruled_manifest(tmp_path: Path) -> tuple[Path, Path]:
    """A ruled manifest carrying the same five gaps the real one carries."""
    path, root = valid_manifest(tmp_path)
    doc = json.loads(path.read_text())
    doc["status"] = "ruled"
    doc["collection"] = {
        "name": CANDIDATE,
        "points": 6,
        "membership_receipt": _membership_receipt(
            root, "collection/e8-membership.json",
            collection=CANDIDATE, rows=6),
        "durability": {
            "status": "verified_offbox",
            "receipt": _artifact(root, "collection/e8-offbox-done"),
        },
    }
    doc["rollback"] = {
        "collection": INCUMBENT,
        "points": 4,
        "mechanism": "deploy/scripts/flip-rollback.sh --to aw_e3",
        "destructive_step_withheld": "promote/finalize_promote.sh",
        "membership_receipt": _membership_receipt(
            root, "collection/e3-membership.json",
            collection=INCUMBENT, rows=4),
        "offbox_receipt": _artifact(root, "collection/e3-offbox-done"),
    }
    doc["ruling"] = {
        "id": "TEST-RULING-aw_e8",
        "adr": "ADR-098 amendment 1",
        "collection": CANDIDATE,
        "ruled_by": "Tim",
        "ruled_on": "2026-09-16",
        "ruled_via": "chat",
        "ratified_by": "merge of PR #999",
        "cite": "the ADR-098 rebuild-tier ruling (task 593)",
        "does_not_cover": "anything not listed in carried_gaps",
        "carried_gaps": [
            {"id": "G2-source-attribution", "source_manifest": "x.json",
             "suspends": ["sources", "population"],
             "obligation": "o", "owner": "orchestrator", "vehicle": "v"},
            {"id": "G3-l1-markdown", "source_manifest": "x.json",
             "suspends": ["sources"],
             "obligation": "o", "owner": "orchestrator", "vehicle": "v"},
            {"id": "G6-build-revision", "source_manifest": "x.json",
             "suspends": ["rebuild", "collection.build_receipt"],
             "obligation": "o", "owner": "orchestrator", "vehicle": "v"},
            {"id": "G8-legacy-dump-landing", "source_manifest": "x.json",
             "suspends": [],
             "obligation": "o", "owner": "orchestrator", "vehicle": "v"},
            {"id": "G9-content-fidelity-and-profile-compliance",
             "source_manifest": "x.json", "suspends": ["sources"],
             "obligation": "o", "owner": "orchestrator", "vehicle": "v"},
        ],
    }
    for clause in ("sources", "population", "rebuild"):
        doc.pop(clause, None)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path, root


def _verify(path: Path, root: Path, **kw):
    kw.setdefault("expect_collection", CANDIDATE)
    return pc.verify_manifest(path, registry_path=REGISTRY,
                              artifact_root=root, deep=True, **kw)


def _mutate(path: Path, fn) -> None:
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc, indent=2) + "\n")


# --------------------------------------------------------------------------
# it passes, and it says what it rests on
# --------------------------------------------------------------------------
def test_ruled_manifest_passes_deep_and_reports_the_ruling(tmp_path, capsys):
    path, root = ruled_manifest(tmp_path)
    proof = _verify(path, root)
    assert proof["status"] == "PASS"
    assert proof["manifest_status"] == "ruled"
    assert proof["ruling_id"] == "TEST-RULING-aw_e8"
    assert proof["carried_gaps"] == [
        "G2-source-attribution", "G3-l1-markdown", "G6-build-revision",
        "G8-legacy-dump-landing",
        "G9-content-fidelity-and-profile-compliance"]
    assert proof["suspended_clauses"] == [
        "collection.build_receipt", "population", "rebuild", "sources"]
    assert proof["rollback"]["collection"] == INCUMBENT
    assert proof["points"] == 6 and proof["sources"] == 0


def test_carried_gaps_print_to_stderr_on_every_verify(tmp_path, capsys):
    """alias_swap.sh sends stdout to /dev/null; the banner must survive that."""
    path, root = ruled_manifest(tmp_path)
    _verify(path, root)
    err = capsys.readouterr().err
    assert "PRESERVATION RULING TEST-RULING-aw_e8" in err
    assert "open obligations, not closed proofs" in err
    for gap in ("G2-source-attribution", "G8-legacy-dump-landing"):
        assert gap in err
    assert "no clause of this contract" in err  # G8 suspends nothing


def test_a_sealed_manifest_prints_no_ruling(tmp_path, capsys):
    path, root = valid_manifest(tmp_path)
    pc.verify_manifest(path, registry_path=REGISTRY, artifact_root=root,
                       deep=True, expect_collection="aw_test")
    assert "PRESERVATION RULING" not in capsys.readouterr().err


# --------------------------------------------------------------------------
# what `ruled` may not excuse
# --------------------------------------------------------------------------
def test_ruled_without_a_ruling_block_is_refused(tmp_path):
    path, root = ruled_manifest(tmp_path)
    _mutate(path, lambda d: d.pop("ruling"))
    with pytest.raises(pc.ContractError, match="must carry a `ruling` object"):
        _verify(path, root)


def test_a_sealed_manifest_may_not_carry_a_ruling(tmp_path):
    """A seal that carries a ruling is a ruling calling itself a seal."""
    path, root = ruled_manifest(tmp_path)
    _mutate(path, lambda d: d.__setitem__("status", "sealed"))
    with pytest.raises(pc.ContractError, match="must not carry a `ruling`"):
        pc.verify_manifest(path, registry_path=REGISTRY, artifact_root=root,
                           deep=True, expect_collection=CANDIDATE)


@pytest.mark.parametrize("field", pc.RULING_FIELDS)
def test_every_ruling_field_is_mandatory(tmp_path, field):
    path, root = ruled_manifest(tmp_path)
    _mutate(path, lambda d: d["ruling"].__setitem__(field, "  "))
    with pytest.raises(pc.ContractError, match=f"ruling.{field}"):
        _verify(path, root)


@pytest.mark.parametrize("pending", ("PENDING Tim's merge", "TBD", "todo: ask"))
def test_a_ruling_that_still_says_pending_is_not_ratified(tmp_path, pending):
    path, root = ruled_manifest(tmp_path)
    _mutate(path, lambda d: d["ruling"].__setitem__("ratified_by", pending))
    with pytest.raises(pc.ContractError, match="unratified ruling is a waiver"):
        _verify(path, root)


def test_the_flip_gate_also_refuses_a_pending_ratification(tmp_path, withheld_ruling_files):
    contract = json.loads(CONTRACT.read_text())
    doc = json.loads(RULED_MANIFEST.read_text())
    doc["ruling"]["ratified_by"] = "PENDING Tim's merge"
    local = tmp_path / "ruled.json"
    local.write_text(json.dumps(doc))
    contract["rebuild_tier"]["candidate_manifests"][CANDIDATE] = str(local)
    problems = flip_gate._preservation_problems(contract, CANDIDATE, REPO)
    assert any("has not been ratified" in p for p in problems)


def test_an_unbound_ruling_is_a_waiver_and_is_refused(tmp_path):
    path, root = ruled_manifest(tmp_path)
    with pytest.raises(pc.ContractError, match="An unbound ruling is a waiver"):
        pc.verify_manifest(path, registry_path=REGISTRY, artifact_root=root,
                           deep=True)


def test_a_ruling_does_not_cover_another_collection(tmp_path):
    path, root = ruled_manifest(tmp_path)
    with pytest.raises(pc.ContractError, match="!= requested 'aw_e9'"):
        _verify(path, root, expect_collection="aw_e9")


def test_a_ruling_whose_collection_disagrees_with_the_manifest_is_refused(tmp_path):
    path, root = ruled_manifest(tmp_path)
    _mutate(path, lambda d: d["ruling"].__setitem__("collection", "aw_e9"))
    with pytest.raises(pc.ContractError, match="bound to one collection"):
        _verify(path, root)


def test_carried_gaps_may_not_be_empty(tmp_path):
    path, root = ruled_manifest(tmp_path)
    _mutate(path, lambda d: d["ruling"].__setitem__("carried_gaps", []))
    with pytest.raises(pc.ContractError, match="carried_gaps must be a non-empty"):
        _verify(path, root)


@pytest.mark.parametrize("field", ("source_manifest", "obligation", "owner",
                                   "vehicle"))
def test_a_carried_gap_names_its_obligation_owner_and_vehicle(tmp_path, field):
    path, root = ruled_manifest(tmp_path)
    _mutate(path, lambda d: d["ruling"]["carried_gaps"][0].pop(field))
    with pytest.raises(pc.ContractError, match=f"carried_gaps\\[0\\].{field}"):
        _verify(path, root)


def test_only_listed_clauses_may_be_suspended(tmp_path):
    path, root = ruled_manifest(tmp_path)
    _mutate(path, lambda d: d["ruling"]["carried_gaps"][0]["suspends"].append(
        "collection.membership_receipt"))
    with pytest.raises(pc.ContractError, match="not a suspendable clause"):
        _verify(path, root)


def test_a_clause_nobody_carried_is_still_required(tmp_path):
    """Drop G6 and BOTH clauses it suspended are owed again."""
    path, root = ruled_manifest(tmp_path)
    _mutate(path, lambda d: d["ruling"].__setitem__(
        "carried_gaps",
        [g for g in d["ruling"]["carried_gaps"]
         if g["id"] != "G6-build-revision"]))
    with pytest.raises(pc.ContractError,
                       match="collection.build_receipt must be an artifact"):
        _verify(path, root)
    # supply the build receipt and the rebuild command list is owed next
    _mutate(path, lambda d: d["collection"].__setitem__(
        "build_receipt", _artifact(root, "collection/e8-build.json")))
    with pytest.raises(pc.ContractError, match="manifest.rebuild must be dict"):
        _verify(path, root)


def test_a_carried_clause_that_is_supplied_anyway_is_still_validated(tmp_path):
    """A ruling lowers what must be supplied, never what a block proves."""
    path, root = ruled_manifest(tmp_path)
    _mutate(path, lambda d: d.__setitem__(
        "rebuild", {"commands": [{"phase": "only", "argv": ["true"]}]}))
    with pytest.raises(pc.ContractError,
                       match="must end with phase 'verify-membership'"):
        _verify(path, root)


# --------------------------------------------------------------------------
# the count is reconciled, not asserted
# --------------------------------------------------------------------------
def test_points_are_reconciled_against_the_membership_receipt(tmp_path):
    path, root = ruled_manifest(tmp_path)
    _mutate(path, lambda d: d["collection"].__setitem__("points", 7))
    with pytest.raises(pc.ContractError, match="rows=6 != declared points=7"):
        _verify(path, root)


def test_a_membership_receipt_for_another_collection_is_refused(tmp_path):
    path, root = ruled_manifest(tmp_path)
    receipt = json.loads((root / "collection/e8-membership.json").read_text())
    receipt["collection"] = "aw_somewhere_else"
    body = json.dumps(receipt, sort_keys=True).encode() + b"\n"
    artifact = _artifact(root, "collection/e8-membership.json", body=body)
    _mutate(path, lambda d: d["collection"].__setitem__(
        "membership_receipt", artifact))
    with pytest.raises(pc.ContractError, match="is for collection"):
        _verify(path, root)


def test_shard_rows_must_sum_to_the_receipt_total(tmp_path):
    path, root = ruled_manifest(tmp_path)
    receipt = json.loads((root / "collection/e8-membership.json").read_text())
    receipt["shards"][0]["rows"] += 1
    body = json.dumps(receipt, sort_keys=True).encode() + b"\n"
    artifact = _artifact(root, "collection/e8-membership.json", body=body)
    _mutate(path, lambda d: d["collection"].__setitem__(
        "membership_receipt", artifact))
    with pytest.raises(pc.ContractError, match="conservation failed"):
        _verify(path, root)


def test_the_membership_receipt_must_be_off_box(tmp_path):
    path, root = ruled_manifest(tmp_path)
    _mutate(path, lambda d: d["collection"]["durability"].__setitem__(
        "status", "on_one_disk"))
    with pytest.raises(pc.ContractError, match="verified_offbox"):
        _verify(path, root)


# --------------------------------------------------------------------------
# the ruling does not leak into another operation class
# --------------------------------------------------------------------------
def test_a_ruled_manifest_does_not_satisfy_the_live_apply_gate(tmp_path, monkeypatch):
    """`verify_publish_environment` is what the WRITE paths take.

    ADR-098 am.1's ruling covers one alias move. The alias swap reaches this
    module through the CLI; the live/incremental appliers reach it here. A
    ruled manifest must not satisfy this one, or pointing an env var at it
    would carry the ruling into an operation class nobody ruled on.
    """
    path, root = ruled_manifest(tmp_path)
    monkeypatch.setenv("AW_PRESERVATION_MANIFEST", str(path))
    monkeypatch.setenv("AW_PRESERVATION_PROFILE_REGISTRY", str(REGISTRY))
    monkeypatch.setenv("AW_ARTIFACT_ROOT", str(root))
    monkeypatch.setenv("AW_PRESERVATION_DEEP", "1")
    monkeypatch.setenv("AW_PRESERVATION_COLLECTION", CANDIDATE)
    with pytest.raises(pc.ContractError,
                       match="covers the promotion it names and nothing else"):
        pc.verify_publish_environment()


def test_a_sealed_manifest_still_satisfies_the_live_apply_gate(tmp_path, monkeypatch):
    """The same gate, unchanged for a seal: nothing else was relaxed."""
    path, root = valid_manifest(tmp_path)
    monkeypatch.setenv("AW_PRESERVATION_MANIFEST", str(path))
    monkeypatch.setenv("AW_PRESERVATION_PROFILE_REGISTRY", str(REGISTRY))
    monkeypatch.setenv("AW_ARTIFACT_ROOT", str(root))
    monkeypatch.setenv("AW_PRESERVATION_DEEP", "1")
    monkeypatch.setenv("AW_PRESERVATION_COLLECTION", "aw_test")
    assert pc.verify_publish_environment()["status"] == "PASS"


# --------------------------------------------------------------------------
# the rollback the ruling rests on
# --------------------------------------------------------------------------
def test_a_ruled_manifest_without_a_rollback_is_refused(tmp_path):
    path, root = ruled_manifest(tmp_path)
    _mutate(path, lambda d: d.pop("rollback"))
    with pytest.raises(pc.ContractError,
                       match="must carry a `rollback` object"):
        _verify(path, root)


def test_the_rollback_target_may_not_be_the_promoted_collection(tmp_path):
    path, root = ruled_manifest(tmp_path)
    _mutate(path, lambda d: d["rollback"].__setitem__("collection", CANDIDATE))
    with pytest.raises(pc.ContractError, match="not a rollback target"):
        _verify(path, root)


def test_the_rollback_names_the_destructive_step_it_withholds(tmp_path):
    path, root = ruled_manifest(tmp_path)
    _mutate(path, lambda d: d["rollback"].__setitem__(
        "destructive_step_withheld", ""))
    with pytest.raises(pc.ContractError,
                       match="rollback.destructive_step_withheld"):
        _verify(path, root)


def test_the_rollback_membership_receipt_is_reconciled_too(tmp_path):
    path, root = ruled_manifest(tmp_path)
    _mutate(path, lambda d: d["rollback"].__setitem__("points", 5))
    with pytest.raises(pc.ContractError, match="rows=4 != declared points=5"):
        _verify(path, root)


# --------------------------------------------------------------------------
# the real manifests
# --------------------------------------------------------------------------
def test_the_blocked_rebuild_manifest_is_untouched_and_still_refuses(withheld_ruling_files):
    doc = json.loads(BLOCKED_MANIFEST.read_text())
    assert doc["status"] == "blocked"
    assert doc["collection"]["name"] == INCUMBENT
    assert [g["id"] for g in doc["blocking_gaps"]] == [
        "G2-source-attribution", "G3-l1-markdown", "G6-build-revision",
        "G8-legacy-dump-landing",
        "G9-content-fidelity-and-profile-compliance"]
    with pytest.raises(pc.ContractError, match="not sealed"):
        pc.verify_manifest(BLOCKED_MANIFEST, expect_collection=INCUMBENT)


def test_the_ruled_manifest_is_shaped_as_this_contract_requires(withheld_ruling_files):
    doc = json.loads(RULED_MANIFEST.read_text())
    assert doc["schema"] == pc.SCHEMA
    assert doc["status"] == "ruled"
    assert doc["collection"]["name"] == CANDIDATE
    assert doc["collection"]["points"] == CANDIDATE_POINTS
    assert doc["rollback"]["collection"] == INCUMBENT
    assert doc["rollback"]["points"] == INCUMBENT_POINTS
    assert doc["rollback"]["destructive_step_withheld"].split()[0].endswith(
        "promote/finalize_promote.sh")
    # nothing outside the five gaps of the blocked manifest may be carried
    blocked = json.loads(BLOCKED_MANIFEST.read_text())
    carried = [g["id"] for g in doc["ruling"]["carried_gaps"]]
    assert carried == [g["id"] for g in blocked["blocking_gaps"]]
    for gap in doc["ruling"]["carried_gaps"]:
        assert gap["source_manifest"] == "data/production-rebuild-lock.v1.json"
        for clause in gap["suspends"]:
            assert clause in pc.SUSPENDABLE_CLAUSES
    # it carries no source chain, so it must not claim one
    assert "sources" not in doc and "population" not in doc


def test_the_real_ruled_manifest_passes_the_shallow_contract(withheld_ruling_files):
    """Deep verification needs the artifact root, which is off-repo. Shallow
    verification is every structural clause, and it must hold in CI."""
    proof = pc.verify_manifest(RULED_MANIFEST, expect_collection=CANDIDATE)
    assert proof["status"] == "PASS"
    assert proof["points"] == CANDIDATE_POINTS
    assert proof["rollback"]["collection"] == INCUMBENT
    assert len(proof["carried_gaps"]) == 5


def test_the_real_ruled_manifest_refuses_any_other_collection(withheld_ruling_files):
    # named explicitly so this can never pass because the file is absent
    assert RULED_MANIFEST.is_file()
    for other in (INCUMBENT, "aw_e9", "aw_prod"):
        with pytest.raises(pc.ContractError):
            pc.verify_manifest(RULED_MANIFEST, expect_collection=other)


# --------------------------------------------------------------------------
# aw_e9's own ruled manifest (task 651)
# --------------------------------------------------------------------------
def test_the_e9_ruled_manifest_is_shaped_as_this_contract_requires(withheld_ruling_files):
    doc = json.loads(RULED_MANIFEST_E9.read_text())
    assert doc["schema"] == pc.SCHEMA
    assert doc["status"] == "ruled"
    assert doc["collection"]["name"] == CANDIDATE_E9
    assert doc["collection"]["points"] == 33_737_287
    assert doc["rollback"]["collection"] == INCUMBENT_E9
    assert doc["rollback"]["destructive_step_withheld"].split()[0].endswith(
        "promote/finalize_promote.sh")
    # the SAME five gaps as aw_e8's manifest -- aw_e9 is a rebuild of aw_e8's
    # own population, not a new corpus, so it closes none of them
    blocked = json.loads(BLOCKED_MANIFEST.read_text())
    carried = [g["id"] for g in doc["ruling"]["carried_gaps"]]
    assert carried == [g["id"] for g in blocked["blocking_gaps"]]
    for gap in doc["ruling"]["carried_gaps"]:
        for clause in gap["suspends"]:
            assert clause in pc.SUSPENDABLE_CLAUSES
    assert "sources" not in doc and "population" not in doc


def test_the_real_e9_ruled_manifest_passes_the_shallow_contract(withheld_ruling_files):
    """Deep verification needs the artifact root, which is off-repo. Shallow
    verification is every structural clause, and it must hold in CI."""
    proof = pc.verify_manifest(RULED_MANIFEST_E9, expect_collection=CANDIDATE_E9)
    assert proof["status"] == "PASS"
    assert proof["points"] == 33_737_287
    assert proof["rollback"]["collection"] == INCUMBENT_E9
    assert len(proof["carried_gaps"]) == 5


def test_the_real_e9_ruled_manifest_refuses_any_other_collection(withheld_ruling_files):
    assert RULED_MANIFEST_E9.is_file()
    for other in (INCUMBENT_E9, "aw_e3", "aw_prod"):
        with pytest.raises(pc.ContractError):
            pc.verify_manifest(RULED_MANIFEST_E9, expect_collection=other)


def test_the_e9_ruling_is_ratified_not_pending(withheld_ruling_files):
    doc = json.loads(RULED_MANIFEST_E9.read_text())
    ratified = doc["ruling"]["ratified_by"]
    assert not ratified.upper().startswith(("PENDING", "TBD", "TODO"))
    assert doc["ruling"]["ruled_by"] == "Tim"
    assert doc["ruling"]["collection"] == CANDIDATE_E9


def test_the_flip_gate_takes_the_ruling_for_named_candidates_only(withheld_ruling_files):
    """aw_e8 and aw_e9 each have their own named manifest (task 651); every
    OTHER candidate -- never named in `candidate_manifests` -- still meets
    the blocked default."""
    contract = json.loads(CONTRACT.read_text())
    rb = contract["rebuild_tier"]
    assert rb["required"] is True
    assert rb["manifest"] == "data/production-rebuild-lock.v1.json"
    assert rb["candidate_manifests"][CANDIDATE] == str(
        RULED_MANIFEST.relative_to(REPO))
    assert flip_gate._preservation_problems(contract, CANDIDATE, REPO) == []
    assert rb["candidate_manifests"]["aw_e9"] == str(
        RULED_MANIFEST_E9.relative_to(REPO))
    assert flip_gate._preservation_problems(contract, "aw_e9", REPO) == []
    # anything else still meets the blocked manifest
    problems = flip_gate._preservation_problems(contract, "aw_e11", REPO)
    assert problems and "not `sealed`" in problems[0]
    assert "production-rebuild-lock.v1.json" in problems[0]


def test_the_flip_gate_refuses_a_ruling_nobody_ratified(tmp_path, withheld_ruling_files):
    contract = json.loads(CONTRACT.read_text())
    doc = json.loads(RULED_MANIFEST.read_text())
    doc["ruling"]["ratified_by"] = ""
    local = tmp_path / "ruled.json"
    local.write_text(json.dumps(doc))
    contract["rebuild_tier"]["candidate_manifests"][CANDIDATE] = str(local)
    problems = flip_gate._preservation_problems(contract, CANDIDATE, REPO)
    assert any("ruling nobody merged is a waiver" in p for p in problems)


def test_the_flip_gate_refuses_a_ruled_manifest_with_no_ruling(tmp_path, withheld_ruling_files):
    contract = json.loads(CONTRACT.read_text())
    doc = json.loads(RULED_MANIFEST.read_text())
    doc.pop("ruling")
    local = tmp_path / "ruled.json"
    local.write_text(json.dumps(doc))
    contract["rebuild_tier"]["candidate_manifests"][CANDIDATE] = str(local)
    problems = flip_gate._preservation_problems(contract, CANDIDATE, REPO)
    assert any("carries no `ruling` block" in p for p in problems)


# --------------------------------------------------------------------------
# aw_e10 (task 865, P9): its OWN rebuild-tier manifest, blocked until the build
# --------------------------------------------------------------------------
def test_flip_gate_resolves_aw_e10_to_its_own_manifest_not_the_aw_e3_lock(withheld_ruling_files):
    contract = json.loads(CONTRACT.read_text())
    rb = contract["rebuild_tier"]
    rel = flip_gate._rebuild_tier_manifest(rb, "aw_e10")
    assert rel == str(MANIFEST_E10.relative_to(REPO))
    assert rel != rb["manifest"]
    doc = json.loads(MANIFEST_E10.read_text())
    assert doc["schema"] == rb["manifest_schema"]
    assert doc["collection"]["name"] == "aw_e10"


def test_flip_gate_accepts_the_ruled_aw_e10_manifest_on_its_own_ruling(withheld_ruling_files):
    """Task 950 (T20, RULING 36): the manifest is `ruled`; the shallow gate
    accepts it and records the ruling, not the aw_e3 lock's gaps."""
    contract = json.loads(CONTRACT.read_text())
    assert flip_gate._preservation_problems(contract, "aw_e10", REPO) == []
    rec = flip_gate._preservation_ruling(contract, "aw_e10", REPO)
    assert rec["collection"] == "aw_e10"
    assert rec["ruled_by"] == "Tim"
    assert rec["ruled_on"] == "2026-10-15"
    assert rec["carried_gaps"] == [
        "G2-source-attribution", "G3-l1-markdown", "G6-build-revision",
        "G8-legacy-dump-landing", "G9-content-fidelity-and-profile-compliance"]


def test_the_aw_e10_manifest_carries_the_five_lock_gaps_by_name(withheld_ruling_files):
    lock = json.loads(BLOCKED_MANIFEST.read_text())
    doc = json.loads(MANIFEST_E10.read_text())
    assert doc["status"] == "ruled"
    ruling = doc["ruling"]
    assert ruling["collection"] == "aw_e10"
    assert ruling["adr"] == "ADR-098 amendment 1"
    carried = {g["id"] for g in ruling["carried_gaps"]}
    assert {g["id"] for g in lock["blocking_gaps"]} <= carried
    for gid in carried:
        assert gid in ruling["does_not_cover"]
    # E10-ruling is closed by the block (task 950); E10-membership by the
    # task 955 receipt; E10-rollback by the rollback block (task 960).
    assert doc["blocking_gaps"] == []
    assert [g["id"] for g in doc["closed_gaps"]] == [
        "E10-ruling", "E10-membership", "E10-rollback"]
    assert doc["collection"]["points"] == 61718090
    assert doc["collection"]["durability"]["status"] == "verified_offbox"
    assert doc["rollback_planned"]["collection"] == "aw_e9"


def test_the_aw_e10_rollback_block_names_aw_e9_and_reconciles_its_own_receipt(withheld_ruling_files):
    """Task 960: the rollback block names aw_e9 with aw_e9's own receipt,
    declared at the receipt's own row count (the contract demands rows ==
    points), and withholds the destructive step."""
    doc = json.loads(MANIFEST_E10.read_text())
    e9 = json.loads(RULED_MANIFEST_E9.read_text())
    rb = doc["rollback"]
    assert rb["collection"] == "aw_e9"
    assert rb["points"] == e9["collection"]["points"] == 33737287
    assert rb["membership_receipt"] == e9["collection"]["membership_receipt"]
    assert rb["offbox_receipt"] == e9["collection"]["durability"]["receipt"]
    assert "finalize_promote.sh" in rb["destructive_step_withheld"]
    assert "61,730,003" in rb["mechanism"] and "STALENESS" in rb["mechanism"]
    assert "--rollback" in rb["mechanism"]
    # The shallow contract accepts the block as declared.
    pc.verify_manifest(MANIFEST_E10, registry_path=REGISTRY,
                       expect_collection="aw_e10")


def test_the_lock_is_untouched_and_still_aw_e3(withheld_ruling_files):
    lock = json.loads(BLOCKED_MANIFEST.read_text())
    assert lock["status"] == "blocked"
    assert lock["collection"]["name"] == "aw_e3"
