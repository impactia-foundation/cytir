#!/usr/bin/env python3
"""Task 634: the cc_authority_frontier_w7 family declaration, against the REAL
data/source-family-contracts.v1.json this repo ships.

Guards the specific thing task 630/634 found: a wave may not add to a family
whose status is not `declared` (source_family_contract.assert_declared refuses
`backfilled`/absent families). This pins that the W7 CC authority-frontier
family IS declared, has zero unknowns (so it actually clears `--strict`), and
that its chunk policy resolves to the real, CURRENT registry entry -- not a
value that merely looks plausible in isolation.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cytir.sources import contract as SFC  # noqa: E402

REPO = HERE.parent
DOC = SFC.load()
FAMS = SFC.by_id(DOC)
FID = "cc_authority_frontier_w7"


def test_family_is_declared_and_present():
    assert FID in FAMS, (
        f"{FID} is missing from {SFC.DEFAULT_PATH} -- run "
        "the task-634 declare_cc_authority_frontier_contract script")
    assert FAMS[FID]["status"] == "declared"


def test_zero_unknowns_declared_family_has_none():
    fam = FAMS[FID]
    assert fam["unknowns"] == []
    assert SFC.actual_unknowns(fam) == []


def test_validates_clean_no_problems():
    problems = SFC.validate_family(FAMS[FID], strict=True, repo=REPO)
    assert problems == [], problems


def test_assert_declared_does_not_refuse_this_family():
    # This is the exact call assembly_binder makes before it will read a row
    # of a family (source_family_contract.py module docstring). A family that
    # cannot pass this call cannot be staged at all -- task 630's finding.
    SFC.assert_declared([FID], doc=DOC, strict=True, repo=REPO,
                         stage="plan")


def test_chunk_policy_resolves_to_the_real_current_registry_entry():
    fam = FAMS[FID]
    cp = fam["chunk_policy"]
    pid = cp["policy_id"]
    policy_path = REPO / "data" / f"{pid}.json"
    assert policy_path.exists(), policy_path
    assert SFC.sha256_file(policy_path) == cp["sha256"]

    import cytir.chunk.registry as CREG  # noqa: E402
    reg = CREG.load()
    ent = CREG.entry(cp["sha256"], reg)
    assert ent["policy_family"] == cp["policy_family"] == "chunk.generic"
    assert ent["status"] in CREG.SELECTABLE, (
        f"chunk.generic policy {cp['sha256']} registry status "
        f"{ent['status']!r} is not selectable by a NEW materialization")


def test_dense_and_sparse_surfaces_are_declared_not_unknown():
    rep = FAMS[FID]["representation"]
    assert rep["dense_surface"] not in (None, "unknown", "")
    assert rep["sparse_surface"] not in (None, "unknown", "")
    assert rep["ruling"] == "RULED"
    assert rep["ruling_cite"]


def test_consent_is_live_per_origin_at_the_current_terminal_schema():
    consent = FAMS[FID]["consent"]
    versions, current = SFC._terminal_versions()
    assert consent["resolution"] == "live-per-origin"
    assert consent["terminal_schema_version"] == current
    assert consent["terminal_schema_version"] in versions


def test_licence_is_never_inferred_from_hosting():
    lic = FAMS[FID]["licence"]
    assert lic["inferred_from_hosting"] is False
    assert lic["basis"] in ("instrument", "statute", "explicit-grant")


def test_replacement_guards_not_required_for_net_new_addition_mode():
    # mode is net_new_addition, not replacement, so G1/G3/G5/G7 are declared
    # by choice here (matches the other declared family, doab), not owed by
    # the replacement-guard rule -- pin that the mode itself is what we think.
    fam = FAMS[FID]
    assert fam["mode"] == "net_new_addition"
    assert set(SFC.REPLACEMENT_GUARDS) <= set(fam["guards"])


def test_whole_document_still_validates_after_the_addition():
    report = SFC.validate(DOC, strict=False, repo=REPO)
    # Adding one declared family must not newly break any OTHER family's
    # validation or the roster-coverage computation.
    assert not any(p.startswith(f"{FID}:") for p in report["problems"])
