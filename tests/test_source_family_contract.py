#!/usr/bin/env python3
"""The source-family onboarding contract, against the REAL backfilled roster.

Every test below is a defect this contract exists to have refused, stated as
the thing that actually shipped:

  * a family reaching `assembly_binder plan` with nothing ever declared about
    it (the class task 526/527 found: fedlex's dense surface and the entity
    cards' sparse surface changed and no check could notice);
  * a licence inferred from where the document is hosted (2026-08-20);
  * consent inherited from Common Crawl (CLAUDE.md durable constraint);
  * a new family acquiring at an old terminal schema version (task 521:
    28,477 of 129,120 lost points sealed `declined` under v5);
  * the binder re-minting a sealed point id (ADR-101 Ruling 2);
  * a replacement that skips G1/G3/G5/G7;
  * a chunk policy id that resolves to no policy file;
  * an `unknown` value hidden by leaving it out of the `unknowns` list.

Run: python3 -m pytest tests/test_source_family_contract.py
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cytir.sources import contract as SFC  # noqa: E402
from tests.helpers.repo_files import repo_file_or_skip  # noqa: E402

REPO = HERE.parent
DOC = SFC.load()
FAMS = SFC.by_id(DOC)

#: Three real families, chosen because each is a different evidence shape:
#: fedlex is the one family with a declared title contract, task472_f2_html is
#: the one with a declared chunk policy AND the extraction-collapse ceiling,
#: and astra_html_exact is a plain pooled member with nothing declared.
FEDLEX = "fedlex_exact"
F2HTML = "task472_f2_html_1024_embed"
ASTRA = "astra_html_exact"


def _fam(name):
    return copy.deepcopy(FAMS[name])


# ---------------------------------------------------------------- the roster
def test_the_backfilled_roster_validates():
    report = SFC.validate(DOC)
    assert report["verdict"] == "PASS", report["problems"][:5]


#: task 558: the first family declared "declared" from birth rather than
#: backfilled. It is a standalone net-new pack (DOAB/OAPEN monograph pointer
#: cards) staged and embedded outside the ADR-101 assembly-week rebuild this
#: roster enumerates -- so it is correctly ABSENT from the claim-manifest
#: roster (`coverage.not_in_roster`, never `undeclared`) until a later task
#: folds it into that registry, if it ever does.
#: Task 629 P2 adds twelve more of exactly this shape: the Pool A expansion packs
#: are standalone, sealed outside the ADR-101 assembly-week rebuild this roster
#: enumerates, so they are correctly `coverage.not_in_roster` rather than
#: `undeclared`. Listed rather than pattern-matched, so a thirteenth cannot join by
#: accident.
POOL_A_NOT_IN_ROSTER = {
    "technical_manufacturer_catalog", "news_current_articles",
    "canada_official_news", "github_agent_resources", "service_public_fiche",
    "statcan_wds", "context7_pinned_docs", "stackoverflow_qa",
    "rfc_editor_net_new", "eric_official_pdf", "openalex_academic_dump",
    "hn_firebase_pointer",
}
DECLARED_NOT_IN_ROSTER = {"doab_monographs_mitpress_p0", "stackexchange_nonso_qa_p0",
                          "cc_authority_frontier_w7",
                          "de_openlegaldata_v1"} | POOL_A_NOT_IN_ROSTER | {
    # task 695: the DILA + official API/bulk families declared by one POLICY PR.
    "dila_juri", "dila_kali", "dila_jorf", "govuk_content",
    "us_federal_register_ecfr", "uk_legislation", "it_senato_gazzetta"} | {
    # POLICY PR of 2026-10-03 (ruling 23.10): the live daily news lane, a backfilled row.
    "news_daily"} | {
    # POLICY PR of task 737 (2026-10-08): the families whose packs still ran a legacy script extractor.
    "biomed_authority_core", "energy_climate_intl", "education_research", "urban_governance",
    "fr_insee_sirene_cards", "ch_zefix_cards", "ch_retailer_cards"} | {
    # POLICY PR of task 1245: finma_circulars, declared from a fresh consent probe (rows already in prod).
    "finma_circulars"}


def test_roster_covers_every_claim_manifest_set():
    """The roster is derived from the manifest, never a hand-list.

    memory: census-roster-from-manifest — three blind spots in one week.
    """
    repo_file_or_skip("data/assembly/claim-manifests.v1.json")
    names = SFC.roster(DOC)
    assert names == set(FAMS) - DECLARED_NOT_IN_ROSTER
    assert len(names) == 86


def test_roster_digest_drift_is_refused():
    repo_file_or_skip("data/assembly/claim-manifests.v1.json")
    doc = copy.deepcopy(DOC)
    doc["roster_source"]["sha256"] = "0" * 64
    with pytest.raises(SFC.FamilyContractError, match="moved under"):
        SFC.roster(doc)


def test_every_backfilled_family_declares_its_unknowns():
    for fid, fam in FAMS.items():
        assert sorted(fam["unknowns"]) == SFC.actual_unknowns(fam), fid


def test_strict_refuses_the_backfilled_roster():
    """The grandfathering is real, and `--strict` still sees through it."""
    report = SFC.validate(DOC, strict=True)
    assert report["verdict"] == "FAIL"
    assert len(report["problems"]) >= 85


# ------------------------------------------------- the refusal the binder gets
def test_an_undeclared_set_is_refused_at_plan():
    with pytest.raises(SFC.FamilyContractError) as exc:
        SFC.assert_declared([FEDLEX, "some_new_lane_nobody_declared"], DOC,
                            stage="plan")
    assert "some_new_lane_nobody_declared" in str(exc.value)
    assert "plan" in str(exc.value)


def test_a_declared_set_passes_at_stage():
    SFC.assert_declared([FEDLEX, F2HTML, ASTRA], DOC, stage="stage")


def test_strict_refuses_a_backfilled_set_at_stage():
    with pytest.raises(SFC.FamilyContractError, match="unknown"):
        SFC.assert_declared([ASTRA], DOC, strict=True, stage="stage")


# ------------------------------------------------------- the ratified rulings
def test_licence_inferred_from_hosting_is_refused():
    fam = _fam(FEDLEX)
    fam["licence"]["inferred_from_hosting"] = True
    problems = SFC.validate_family(fam)
    assert any("inferred_from_hosting" in p for p in problems)
    assert any("hosted" in p for p in problems), \
        "the refusal must carry the ruling, not just the field name"


def test_consent_inherited_from_common_crawl_is_refused():
    fam = _fam(FEDLEX)
    fam["consent"]["resolution"] = "inherited-from-common-crawl"
    fam["unknowns"] = SFC.actual_unknowns(fam)
    problems = SFC.validate_family(fam)
    assert any("consent.resolution" in p for p in problems)
    assert any("Common Crawl" in p for p in problems)


def test_a_new_family_must_acquire_at_the_current_terminal_version():
    """Task 521's 28,477 points. A NEW family may not adopt the old reading."""
    fam = _fam(FEDLEX)
    fam["status"] = "declared"
    fam["consent"]["terminal_schema_version"] = "adr092-direct-terminal.v5"
    problems = SFC.validate_family(fam)
    assert any("adr092-direct-terminal.v8" in p for p in problems)


def test_an_unknown_terminal_version_is_refused_for_a_new_family():
    fam = _fam(FEDLEX)
    fam["status"] = "declared"
    fam["identity"]["minted_by"] = SFC.UNKNOWN     # fedlex has no unknown since ruling 27: make one
    fam["unknowns"] = SFC.actual_unknowns(fam)
    problems = SFC.validate_family(fam)
    assert any("unknown value" in p for p in problems), \
        "status 'declared' with unknowns must be refused outright"


def test_rebinding_identity_at_assembly_is_refused():
    fam = _fam(FEDLEX)
    fam["identity"]["rebinds_at_assembly"] = True
    problems = SFC.validate_family(fam)
    assert any("rebinds_at_assembly" in p for p in problems)
    assert any("ADR-101" in p for p in problems)


def test_a_replacement_that_skips_a_guard_is_refused():
    fam = _fam(FEDLEX)
    assert fam["mode"] == "replacement"
    fam["guards"] = ["G1", "G3"]
    problems = SFC.validate_family(fam)
    assert any("G5" in p and "G7" in p for p in problems)


def test_payload_minimum_must_carry_license():
    """ADR-039: the compact dump carries a per-record `license`."""
    fam = _fam(FEDLEX)
    fam["payload_minimum"]["fields"] = [
        f for f in fam["payload_minimum"]["fields"] if f != "license"]
    problems = SFC.validate_family(fam)
    assert any("license" in p and "omits" in p for p in problems)


def test_master_only_fields_may_not_ship_in_the_dump_minimum():
    fam = _fam(FEDLEX)
    fam["payload_minimum"]["fields"].append("attribution")
    problems = SFC.validate_family(fam)
    assert any("master-only" in p for p in problems)


def test_gate_class_outside_the_ratified_taxonomy_is_refused():
    fam = _fam(FEDLEX)
    fam["gate_classes"] = ["swiss_law"]
    fam["unknowns"] = SFC.actual_unknowns(fam)
    problems = SFC.validate_family(fam)
    assert any("swiss_law" in p for p in problems)


# ------------------------------------------------------------ representation
def test_a_missing_sparse_surface_declaration_is_refused():
    """The entity-card defect, stated as a declaration."""
    fam = _fam(FEDLEX)
    del fam["representation"]["sparse_surface"]
    fam["unknowns"] = SFC.actual_unknowns(fam)
    problems = SFC.validate_family(fam)
    assert any("sparse_surface" in p and "not declared" in p
               for p in problems)


def test_sparse_none_requires_a_named_ruling():
    fam = _fam(FEDLEX)
    fam["representation"]["sparse_surface"] = "none"
    fam["representation"]["ruling"] = "RULED"
    fam["representation"]["ruling_cite"] = None    # fedlex_exact now carries a cite (ruling 26 item 20b)
    fam["unknowns"] = SFC.actual_unknowns(fam)
    problems = SFC.validate_family(fam)
    assert any("ruling_cite" in p for p in problems)


def test_ruled_without_a_cite_is_refused():
    fam = _fam(FEDLEX)
    fam["representation"]["ruling"] = "RULED"
    fam["representation"]["ruling_cite"] = None
    fam["unknowns"] = SFC.actual_unknowns(fam)
    problems = SFC.validate_family(fam)
    assert any("ruling nobody can find" in p for p in problems)


# --------------------------------------------------------------- chunk policy
def test_the_families_that_declare_a_chunk_policy_resolve_to_a_real_file():
    # Was "the two families", then three when task 558 added
    # doab_monographs_mitpress_p0, and 13 since task 565 declared
    # chunk.reference on the 12 families it re-cut under ADR-090 amendment 1,
    # 14 since task 631 declared stackexchange_nonso_qa_p0 (chunk.generic),
    # 15 since task 634 declared cc_authority_frontier_w7 (chunk.generic),
    # 16 since task 633 declared de_openlegaldata_v1 (chunk.generic),
    # and 28 since task 629 P2 declared a chunk policy on the 12 Pool A
    # families (including this file's own stackoverflow_qa entry),
    # 35 since task 695 declared the seven DILA + official API/bulk families,
    # and 36 since the POLICY PR of 2026-10-03 declared the news_daily row's chunk policy,
    # and 38 since ruling 26 item 20c declared `native` on fedlex_exact and loda_exact (both now
    # `native-chunked`, ruling 27; dila_kali replaced chunk.atomic by `native-chunked`, so the count
    # stays 38),
    # and 47 since task 737 declared a chunk policy on 4 authority-pack rows (chunk.generic),
    # the 3 card rows (chunk.atomic), bfs_dam_exact and scratch_entscheidsuche_exact (chunk.atomic).
    # and 53 since task 852 (ruling 29 T9 / 30) declared chunk.generic on 5 regulator rows
    # (astra_html, weko, bav, bazl, elcom) and `native-chunked` on the cantonal-law row,
    # and 54 since task 1245 declared finma_circulars (chunk.generic).
    # and 61 since tasks 2134 (context7_exact), 2135 (ietf_drafts_exact, routec_generic_complete_exact),
    # 2136 (rfc_exact) and 2139 (w3c_exact) declared their epoch-4 stage chunk.generic policy
    # (sealed-legend evidence, task 2133 / 2139); task 2137 (cochrane_direct_exact, cochrane_preserved_exact)
    # added two; task 2138 added none; and 65 since task 2260: bfs-pxweb, pubmed, swisstopo-v2 (atomic),
    # service_public_dila_exact (declared generic).
    # `native` and `native-chunked` are explicit values with no policy file
    # (cytir.sources.contract.NATIVE_CHUNK_POLICY, NATIVE_CHUNKED_CHUNK_POLICY).
    # Re-derived, not preserved: this count is a fact about the data file,
    # not a fixed constant, and it is asserted so that a family acquiring or
    # losing a declaration is a visible diff rather than a silent drift.
    declared = [f for f in FAMS.values()
                if f["chunk_policy"]["policy_id"] != SFC.UNKNOWN]
    assert len(declared) == 65, [f["family_id"] for f in declared]
    for fam in declared:
        pid = fam["chunk_policy"]["policy_id"]
        if pid in (SFC.NATIVE_CHUNK_POLICY, SFC.NATIVE_CHUNKED_CHUNK_POLICY):
            assert not (REPO / "data" / f"{pid}.json").exists()   # explicit value, never a file lookup
            continue
        assert (REPO / "data" / f"{pid}.json").exists()
        assert SFC.validate_family(fam) == [] or all(
            "chunk_policy" not in p for p in SFC.validate_family(fam))


def test_a_chunk_policy_id_with_no_file_is_refused():
    fam = _fam(F2HTML)
    fam["chunk_policy"]["policy_id"] = "chunk-policy-that-nobody-wrote.v1"
    fam["unknowns"] = SFC.actual_unknowns(fam)
    problems = SFC.validate_family(fam)
    assert any("resolves to no" in p for p in problems)


def test_a_drifted_chunk_policy_digest_is_refused():
    fam = _fam(F2HTML)
    fam["chunk_policy"]["sha256"] = "1" * 64
    fam["unknowns"] = SFC.actual_unknowns(fam)
    problems = SFC.validate_family(fam)
    assert any("sha256" in p for p in problems)


def test_a_declared_chunk_epoch_must_match_the_policy_file():
    fam = _fam(F2HTML)
    fam["chunk_policy"]["epoch"] = 99
    fam["unknowns"] = SFC.actual_unknowns(fam)
    problems = SFC.validate_family(fam)
    assert any("epoch" in p for p in problems)


# ----------------------------------------------------------------- title shape
def test_fedlex_is_the_one_family_with_a_declared_title_contract():
    declared = [f["family_id"] for f in FAMS.values()
                if f["title_shape"]["mode"] == "declared"]
    assert declared == [FEDLEX]
    row = FAMS[FEDLEX]["title_shape"]
    psc = json.loads(repo_file_or_skip("data/payload-shape-contract.v1.json").read_text())
    assert row["pattern"] == psc["title_shape"]["fedlex-r0-exact-current-v1"]["pattern"], \
        "the backfill must carry the pattern verbatim, not a paraphrase"


def test_opting_out_of_the_title_contract_needs_a_reason():
    fam = _fam(ASTRA)
    fam["title_shape"]["mode"] = "opted_out"
    fam["title_shape"]["why"] = None
    fam["unknowns"] = SFC.actual_unknowns(fam)
    problems = SFC.validate_family(fam)
    assert any("opted_out with no `why`" in p for p in problems)


# ------------------------------------------------------------- the unknowns list
def test_an_unknown_cannot_be_hidden_by_omitting_it():
    fam = _fam(ASTRA)
    fam["unknowns"] = [u for u in fam["unknowns"] if u != "identity.minted_by"]
    problems = SFC.validate_family(fam)
    assert any("actually holds the sentinel" in p for p in problems)


def test_an_unknown_cannot_be_invented_to_look_thorough():
    fam = _fam(ASTRA)
    fam["unknowns"] = sorted(fam["unknowns"] + ["representation.prov_id"])
    problems = SFC.validate_family(fam)
    assert any("actually holds the sentinel" in p for p in problems)


def test_evidence_saying_unknown_is_not_a_declared_unknown():
    """`evidence` is bookkeeping ABOUT the declaration, not part of it."""
    fam = _fam(ASTRA)
    fam["licence"]["evidence"] = ["unknown"]
    assert SFC.actual_unknowns(fam) == sorted(fam["unknowns"])


# ------------------------------------------------------- the mirror contract
def test_terminal_versions_are_read_from_the_one_home_not_copied():
    """No copy of the version taxonomy lives in source_family_contract.py.

    The reader parses `direct_terminal_contract.py` instead of importing it
    (that module refuses at import on a clone with no operator.json, and an
    `except ImportError` fallback to a local copy is exactly the drift this
    contract exists to prevent). This test pins both halves: the values, and
    the absence of a second definition.
    """
    versions, current = SFC._terminal_versions()
    assert current in versions
    assert current == "adr092-direct-terminal.v8"
    assert versions[0] == "adr092-direct-terminal.v5"

    src = (HERE.parent / "cytir" / "sources" / "contract.py").read_text()
    assert "adr092-direct-terminal.v6" not in src, \
        "a literal version string reappeared in source_family_contract.py"


def test_a_terminal_module_that_stopped_declaring_the_versions_is_refused(tmp_path):
    stub = tmp_path / "direct_terminal_contract.py"
    stub.write_text("SCHEMA_VERSIONS = 'not a tuple'\n")
    with pytest.raises(SFC.FamilyContractError, match="no longer defines"):
        SFC._terminal_versions(stub)


def test_a_terminal_module_whose_current_is_not_in_the_tuple_is_refused(tmp_path):
    stub = tmp_path / "direct_terminal_contract.py"
    stub.write_text(
        'A = "adr092-direct-terminal.v5"\n'
        'SCHEMA_VERSIONS = (A,)\n'
        'CURRENT_SCHEMA_VERSION = "adr092-direct-terminal.v9"\n')
    with pytest.raises(SFC.FamilyContractError, match="is not in"):
        SFC._terminal_versions(stub)


# ------------------------------------------------------------ binder wiring
def test_the_binder_calls_the_contract_at_plan_and_at_stage():
    """The refusal has to be WIRED, not merely available.

    `assert_payload_shape` runs in both stage and verify for the same reason:
    a tree planned before a contract existed must not become stageable by
    being old. Pinned textually because importing assembly_binder pulls the
    qdrant client, which a cold clone does not have.
    """
    src = (HERE.parent / "cytir" / "index" / "assemble_stage.py").read_text()
    plan_src = (HERE.parent / "cytir" / "index" / "assemble_plan.py").read_text()
    assert plan_src.count('assert_family_contract(') == 2, \
        "expected the helper definition plus one call in cmd_plan"
    assert src.count('assert_family_contract(') == 1, \
        "expected one call in cmd_stage"
    plan_body = plan_src.split("def cmd_plan(args):", 1)[1].split("\ndef ", 1)[0]
    stage_body = src.split("def cmd_stage(args):", 1)[1].split("\ndef ", 1)[0]
    assert 'assert_family_contract(' in plan_body
    assert 'assert_family_contract(' in stage_body


def test_the_current_assembly_plan_is_fully_declared():
    """The wiring must not break the build that is running right now.

    If this fails, a set in `data/assembly/assembly-plan.current.v1.json` has
    no declaration — which is the refusal working, not the test being wrong.
    """
    plan = json.loads(repo_file_or_skip("data/assembly/assembly-plan.current.v1.json").read_text())
    sets = plan.get("expected_sets") or []
    assert sets, "the current plan declares no expected_sets"
    SFC.assert_declared(sets, DOC, stage="plan")


def test_a_duplicate_declaration_is_refused():
    doc = copy.deepcopy(DOC)
    doc["families"].append(copy.deepcopy(doc["families"][0]))
    with pytest.raises(SFC.FamilyContractError, match="declared twice"):
        SFC.by_id(doc)
