#!/usr/bin/env python3
"""ADR-113 — a `proposed` chunk policy selectable for ONE named family only.

`chunk_policy_registry.assert_selectable` used to know two answers: `current`
(any new materialization may select it) and `proposed` admitted only under
`experiment=True` (a measurement arm that writes to no collection). ADR-113
needed a third: ERIC's 1024w `chunk.generic` arm (E631-1, task 631) is real
production evidence for ERIC's own extraction, but promoting it to `current`
would flip the shared `chunk.generic` default for every other family that
uses it (docs, agent-resources, ...) on the strength of one family's
structural measurement alone.

`family_id=` + the registry's `admitted_for_families` list is the narrow
answer: a `proposed` policy stays `proposed` (refused for everyone else,
still refused for a plain `assert_selectable(policy)` call with no
`family_id`), and is selectable for a real (non-experiment) run ONLY by the
family an ADR actually named.

Run: python3 -m pytest tests/test_chunk_policy_registry.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cytir.chunk.registry as CREG  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent

#: The ADR-113 arm: chunk.generic epoch 8, status `proposed`,
#: admitted_for_families=["eric_official_pdf"].
ERIC_ARM_SHA = "3f3e24823cf2e0851b149690b6c43d579e3b7d1066a66617b97fd86c5cc7e8d4"
#: chunk.generic epoch 4, status `current` -- the shared default, unaffected.
GENERIC_CURRENT_SHA = "7dcb54c497e1855ad91ddafe80ff34b1da14f3d23b3af26dc7c7b53ab70ce6bf"
#: Any other `proposed` chunk.reference arm, to prove the mechanism doesn't
#: leak selectability onto policies nobody named.
REFERENCE_PROPOSED_SHA = "219487eb3f0c61c236a4fb62648dbd82dfca3012f2696a3e8bafa9545fae24bb"


def test_the_registry_still_marks_the_eric_arm_proposed():
    # ADR-113 scopes selectability; it does not ratify the arm to `current`.
    # A drive-by `status: current` edit would silently widen it to every
    # chunk.generic family, which is exactly what this ADR refused to do.
    doc = CREG.load()
    assert doc["policies"][ERIC_ARM_SHA]["status"] == "proposed"
    assert doc["policies"][ERIC_ARM_SHA]["admitted_for_families"] == [
        "eric_official_pdf"]


def test_plain_selection_still_refuses_the_eric_arm():
    with pytest.raises(CREG.ChunkPolicyRegistryError):
        CREG.assert_selectable(ERIC_ARM_SHA)


def test_a_different_family_id_is_still_refused():
    # The scoping is per-family, not per-policy-existing: naming a family the
    # ADR did NOT list must not become selectable by accident.
    with pytest.raises(CREG.ChunkPolicyRegistryError):
        CREG.assert_selectable(ERIC_ARM_SHA, family_id="stackoverflow_qa")


def test_eric_official_pdf_selects_the_arm_under_adr113():
    fam = CREG.assert_selectable(ERIC_ARM_SHA, family_id="eric_official_pdf")
    assert fam == "chunk.generic"


def test_experiment_true_is_unaffected_by_family_scoping():
    # The pre-existing measurement-arm path still works exactly as before;
    # family_id is an ADDITIONAL door, not a replacement for it.
    fam = CREG.assert_selectable(ERIC_ARM_SHA, experiment=True)
    assert fam == "chunk.generic"


def test_family_scoping_is_a_no_op_on_an_already_current_policy():
    # `admitted_for_families` on a `current` policy would be a no-op (already
    # selectable by everyone). Prove the GENERIC default still needs no
    # family_id at all.
    fam = CREG.assert_selectable(GENERIC_CURRENT_SHA)
    assert fam == "chunk.generic"


def test_family_scoping_never_admits_a_superseded_policy():
    # code-review finding (2026-09-25): `admitted_for_families` must excuse
    # ONLY a `proposed` policy. A hand-built doc where the ERIC arm's status
    # is `superseded` (the arm loses a future held-out eval to a wider one)
    # but `admitted_for_families` was never stripped must still refuse --
    # otherwise a stale scope entry silently re-selects a policy the
    # registry itself says may not be picked up for a new materialization,
    # exactly the class of drift this module exists to prevent.
    doc = CREG.load()
    stale = json.loads(json.dumps(doc))  # deep copy, no live-doc mutation
    stale["policies"][ERIC_ARM_SHA] = dict(
        stale["policies"][ERIC_ARM_SHA],
        status="superseded",
        successor_sha256=GENERIC_CURRENT_SHA)
    with pytest.raises(CREG.ChunkPolicyRegistryError):
        CREG.assert_selectable(ERIC_ARM_SHA, family_id="eric_official_pdf",
                               doc=stale)


def test_family_scoping_never_admits_a_special_purpose_policy():
    # Same defect class, the other non-selectable status this registry uses.
    doc = CREG.load()
    stale = json.loads(json.dumps(doc))
    stale["policies"][ERIC_ARM_SHA] = dict(
        stale["policies"][ERIC_ARM_SHA], status="special-purpose")
    with pytest.raises(CREG.ChunkPolicyRegistryError):
        CREG.assert_selectable(ERIC_ARM_SHA, family_id="eric_official_pdf",
                               doc=stale)


def test_source_family_contract_scoping_never_admits_a_superseded_policy():
    # The sibling fix in source_family_contract._check_chunk_policy_registry.
    from cytir.sources import contract as SFC

    doc = CREG.load()
    stale = json.loads(json.dumps(doc))
    stale["policies"][ERIC_ARM_SHA] = dict(
        stale["policies"][ERIC_ARM_SHA], status="superseded")
    policy_file = REPO / "data" / (
        "chunk-policy-struct-md-1024w-o0-adjacent-sections-tail-1536-"
        "tok2560-generic-arm.v1.json")

    def fake_entry(sha, d=None):
        return stale["policies"][sha]

    original_entry = CREG.entry
    CREG.entry = fake_entry
    try:
        problems = SFC._check_chunk_policy_registry(
            "eric_official_pdf", {"policy_family": "chunk.generic"},
            policy_file, "declared", REPO)
    finally:
        CREG.entry = original_entry
    assert any("registry status" in p for p in problems)


def test_family_scoping_does_not_leak_to_an_unrelated_proposed_policy():
    # A family_id that happens to be a real family, passed against a
    # DIFFERENT proposed policy that never named it, is still refused.
    with pytest.raises(CREG.ChunkPolicyRegistryError):
        CREG.assert_selectable(REFERENCE_PROPOSED_SHA,
                               family_id="eric_official_pdf")


def test_source_family_contract_honours_the_same_scoping():
    # The contract validator's own chunk-policy gate (only exercised for a
    # BRAND NEW `declared` family) must read the identical field, so a
    # future `declared` family scoped this way is not blocked either.
    from cytir.sources import contract as SFC

    doc = CREG.load()
    policy_file = REPO / "data" / (
        "chunk-policy-struct-md-1024w-o0-adjacent-sections-tail-1536-"
        "tok2560-generic-arm.v1.json")
    assert policy_file.exists()
    problems = SFC._check_chunk_policy_registry(
        "eric_official_pdf", {"policy_family": "chunk.generic"},
        policy_file, "declared", REPO)
    assert problems == []
    problems_unscoped = SFC._check_chunk_policy_registry(
        "some_other_family", {"policy_family": "chunk.generic"},
        policy_file, "declared", REPO)
    assert any("registry status" in p for p in problems_unscoped)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
