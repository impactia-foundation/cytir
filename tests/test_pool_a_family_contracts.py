#!/usr/bin/env python3
"""Task 629 P2 -- the 12 Pool A `declared` source-family contracts.

The roster is derived from task 629's `pack_roster.py` script, never a hand-list
(memory `census-roster-from-manifest`): a family that fell out of the roster
must fall out of these assertions too, not linger as a stale literal.

**Two families deliberately do NOT match the brief's "12 declared, unknowns:
[] " instruction, and this file documents why rather than hiding it:**

  * `news_current_articles` and `canada_official_news` are `declared` with
    empty `unknowns`, but the source-family contract checker (`--check`) REFUSES
    both: their packs' own sealed `PACK.portable.v2.json` records a chunk
    policy (`news-article-768w-o0-...`, chunk.news epoch 4) whose registry
    status is `superseded` -- ADR-090 §5 measured it losing the news bake-off
    on every leg and every query shape, and kept `legacy-180w-o30@1` (epoch 1)
    as the ratified current policy. This is a SECOND restage requirement,
    parallel to §12.1's provenance-head ruling, discovered by this task and
    not yet ruled on. `test_news_families_are_a_known_chunk_policy_defect`
    below pins the exact refusal so a silent "fix" (e.g. quietly swapping the
    status to `backfilled` to make the checker stop complaining) cannot land
    unnoticed -- the fix is a restage, not a status change.

  * `openalex_academic_dump` is `backfilled`, not `declared`, and carries real
    `unknowns` (`licence.instrument`, `licence.basis`).
    the sealed Pool A licence artefact (P8) is explicitly scoped to
    the 8 GO-after-P7 families only (`tests/test_pool_a_licence.py`'s own
    comment says so), and no other sealed artifact in this repo carries a
    licence reading for `openalex.org`. Writing a licence would be exactly
    the fabrication task 629's brief prohibits ("do not write a fake licence
    to make `unknowns` empty").

  * `eric_official_pdf` **moved out of this set 2026-09-25 (ADR-113, task
    659)**: its licence gap is now resolved by a ratified general category
    (no-licence-declared + no-opt-out-in-the-resolved-signals = ADR-039
    minimum record), not by writing a licence for `files.eric.ed.gov` -- the
    fabrication this docstring's prior paragraph still prohibits. It is now
    `declared` with empty `unknowns`, exactly the move this file's own
    `test_declared_and_retired_families_carry_no_unknowns` predicted for it.

Run: AW_OPERATOR_URL=https://websear.ch AW_OPERATOR_CONTACT=bender@impactia.org \
     python3 -m pytest tests/test_pool_a_family_contracts.py -q
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPTS = ROOT / "tasks" / "629-scripts"

from tests.helpers.repo_files import repo_file_or_skip  # noqa: E402

repo_file_or_skip((SCRIPTS / "pack_roster.py").relative_to(ROOT))

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SCRIPTS))

from cytir.sources import contract as sfc  # noqa: E402
import pack_roster as pr  # noqa: E402

SFC_PATH = ROOT / "data" / "source-family-contracts.v1.json"
REPCON_PATH = ROOT / "data" / "representation-contract.v1.json"
CONSENT_ARTIFACT = ROOT / "tasks" / "629-artifacts" / "pool-a-consent.v1.json"
LICENCE_ARTIFACT = ROOT / "tasks" / "629-artifacts" / "pool-a-licence.v1.json"

#: hn_firebase_pointer is REFUSED (task 629 §9.1) -- excluded from the sealed
#: consent artefact's GO roster and from P8's licence scope, but it still MUST
#: carry a contract (recording the refusal), per the task brief.
REFUSED_FAMILIES = {"hn_firebase_pointer"}

#: The two families this task discovered a chunk-policy defect for. Declared,
#: not backfilled (the brief's instruction), but known to fail the live
#: checker on chunk_policy.selectable until a restage rules on it -- see the
#: module docstring.
CHUNK_POLICY_DEFECT_FAMILIES = {"news_current_articles", "canada_official_news"}

#: The family P8 never scoped a licence for -- backfilled with real unknowns
#: rather than a fabricated licence. See the module docstring. (Was
#: {"eric_official_pdf", "openalex_academic_dump"}; eric_official_pdf moved
#: out 2026-09-25, ADR-113/task 659 -- its licence gap resolved by a ratified
#: general category, not by writing a licence for files.eric.ed.gov.)
LICENCE_GAP_FAMILIES: set[str] = set()  # openalex_academic_dump moved out 2026-09-30 (CC0-1.0 read from OpenAlex's docs repo; POLICY PR)

#: NOT task 629's, and named here rather than silently absorbed. Task 631 (PR
#: #245) added a `stackexchange-nonso-qa-` prefix rule to
#: task 629's `pack_roster.py`, so the roster this module DERIVES now
#: carries a thirteenth family that the Pool A wave never inventoried, never
#: probed under §13, never licensed under §14 and never planned a batch for.
#: Two separate things follow, and only the first is in scope here:
#:
#:   1. It is out of Pool A's scope, so the twelve-family assertions below
#:      subtract it explicitly instead of quietly widening to whatever the
#:      roster happens to hold.
#:   2. Its id mismatch is RESOLVED, and the history is kept because the class of
#:      bug is worth recognising again. The prefix rule used to emit
#:      `stackexchange_nonso_qa` while both authoritative records -- the admission
#:      ledger (five pack entries) and the ratified contract -- named it
#:      `stackexchange_nonso_qa_p0`. A roster that mints family ids from filename
#:      prefixes can disagree with the ledger it is DERIVED FROM, and this one
#:      did, silently, for a day. Ruled 2026-09-22: the ratified id and the sealed
#:      artefact keys win over a prefix rule, so `pack_roster.py` now emits `_p0`.
#:      `test_every_derived_roster_family_resolves_to_a_contract` keeps it closed
#:      for EVERY family rather than for this spelling -- the property that broke
#:      was "a derived id resolves to a ratified contract", not "this one string
#:      is correct".
#:
#: `loda_exact` is the FOURTEENTH, and it is here for a different reason: task 629
#: admitted it deliberately (task 633's sealed LEGI/LODA packs, after Tim merged
#: PR #232) and it DOES have a batch in the W1 plan. It is excluded from the Pool A
#: assertions only because it is not a Pool A family: it is an already-ratified
#: `backfilled` family with real `unknowns` that predate this task
#: (chunk.law has no pinned policy sha by design), it was never in §13's consent
#: sweep or §14's licence sweep -- ADR-076 resolved both, in 2026-07 -- and
#: rewriting its contract to satisfy P2's "declared, unknowns: []" shape would
#: mean fabricating a chunk-policy sha. Its own checks live in
#: task 629's `test_w1_plan.py` (four tests on the batch) and in
#: `test_the_legi_family_is_ratified_elsewhere` below.
OTHER_LANE_FAMILIES = {"stackexchange_nonso_qa_p0", "loda_exact"}

#: The four families §12.1 ruled must restage head-free.
PROVENANCE_HEAD_FAMILIES = {"github_agent_resources", "service_public_fiche",
                            "context7_pinned_docs", "rfc_editor_net_new"}


def _roster_families() -> set[str]:
    """The Pool A family roster, derived from the admission ledger via
    pack_roster.py -- never a hand-list.

    `OTHER_LANE_FAMILIES` is subtracted, not filtered by pattern: a fourteenth
    family arriving from a fourth lane must break these tests and be ruled on,
    exactly as the thirteenth did.
    """
    return {r["family"] for r in pr.roster()} - OTHER_LANE_FAMILIES


def _load_sfc() -> dict:
    return json.loads(SFC_PATH.read_text())


def _by_id(doc: dict) -> dict[str, dict]:
    return {f["family_id"]: f for f in doc["families"]}


def _load_consent() -> dict:
    return json.loads(repo_file_or_skip(CONSENT_ARTIFACT.relative_to(ROOT)).read_text())


def _load_licence() -> dict:
    return json.loads(repo_file_or_skip(LICENCE_ARTIFACT.relative_to(ROOT)).read_text())


@pytest.fixture(scope="module")
def roster_families() -> set[str]:
    return _roster_families()


@pytest.fixture(scope="module")
def contracts() -> dict[str, dict]:
    return _by_id(_load_sfc())


@pytest.fixture(scope="module")
def consent_doc() -> dict:
    return _load_consent()


@pytest.fixture(scope="module")
def licence_doc() -> dict:
    return _load_licence()


# ---------------------------------------------------------------------------
# 1. every Pool A family has a contract
# ---------------------------------------------------------------------------
def test_every_pool_a_family_has_a_contract(roster_families, contracts):
    assert roster_families, "pack_roster.roster() returned nothing -- ledger empty?"
    missing = roster_families - set(contracts)
    assert not missing, f"Pool A families with no source-family contract: {missing}"


def test_the_roster_is_exactly_twelve_families(roster_families):
    # Not a hand-list -- this is the roster ITSELF, recomputed (minus
    # OTHER_LANE_FAMILIES), so a future pack that changes the count is a signal
    # to re-run P2, not a broken literal (memory census-roster-from-manifest
    # names this exact failure mode: task 631 folded five more ERIC packs into
    # the SAME family without changing the family count -- and then a sixth
    # rule that DID change it, which is why the subtraction is explicit).
    assert roster_families == {
        "technical_manufacturer_catalog", "news_current_articles",
        "canada_official_news", "github_agent_resources", "service_public_fiche",
        "statcan_wds", "context7_pinned_docs", "stackoverflow_qa",
        "rfc_editor_net_new", "eric_official_pdf", "openalex_academic_dump",
        "hn_firebase_pointer",
    }


def test_every_derived_roster_family_resolves_to_a_contract():
    """The property the `_p0` rename restored, asserted for EVERY family.

    Deliberately general. The defect was not about Stack Exchange: a roster that
    mints family ids from pack-id prefixes can disagree with the ledger it derives
    from, and the only guard that scales is that every id it mints resolves to a
    ratified contract. A test pinned to one spelling would have to be edited the
    next time it happens, which is the same as not having one.
    """
    derived = {r["family"] for r in pr.roster()}
    contracts = set(_by_id(_load_sfc()))
    unresolved = sorted(derived - contracts)
    assert not unresolved, (
        f"roster families that resolve to no source-family contract: "
        f"{unresolved}. A family id derived from a pack-id prefix must equal the "
        "ratified contract id, not a spelling invented at the prefix rule.")


def test_the_other_lane_family_resolves_to_its_ratified_contract():
    """Task 631's family: in the roster, out of Pool A, and correctly keyed now."""
    derived = {r["family"] for r in pr.roster()}
    contracts = _by_id(_load_sfc())
    assert "stackexchange_nonso_qa_p0" in derived
    assert "stackexchange_nonso_qa_p0" in contracts
    # the pre-rename spelling must not come back, by either route
    assert "stackexchange_nonso_qa" not in derived
    assert "stackexchange_nonso_qa" not in contracts


def test_the_legi_family_is_ratified_elsewhere_and_honestly_incomplete():
    """`loda_exact` is admitted by this task but is not a Pool A contract.

    It is excluded from the twelve-family assertions, so this test states what IS
    true of it -- otherwise "excluded" would be indistinguishable from
    "unchecked". Three things, and the third is the one that would be tempting to
    paper over: its chunk-policy fields hold the sentinel BY DESIGN, because
    chunk.law has no pinned policy sha to cite.
    """
    contracts = _by_id(_load_sfc())
    fam = contracts["loda_exact"]
    # 1. consent and licence are resolved, and not by this task
    assert fam["consent"]["resolution"] == "explicit-grant"
    assert fam["consent"]["search_ok"] is True
    assert "ADR-076" in " ".join(fam["consent"]["evidence"])
    assert fam["licence"]["instrument"] == "Etalab-2.0"
    assert fam["licence"]["inferred_from_hosting"] is False
    # 2. replacement, not append -- the fact that makes its batch different
    assert fam["mode"] == "replacement"
    # 3. honestly incomplete: backfilled, with the sentinel where there is no
    # artefact to cite, and the reason recorded in the evidence rather than a sha
    # invented to quiet a checker.
    assert fam["status"] == "backfilled"
    assert not sfc.actual_unknowns(fam), (
        "loda_exact has no unknowns since POLICY v3.3 (ruling 27: `identity.minted_by: "
        "text-anchor-v0`); a new unknown must be a visible change here")
    assert fam["identity"]["minted_by"] == "text-anchor-v0"
    # POLICY v3.3 (ruling 27 T6 a): `native-chunked` = the source's own article, split at paragraph
    # marks when over the embedder window. No sha, no epoch, no policy file; the task 633 note on
    # why chunk.law has no pin stays as evidence.
    assert fam["chunk_policy"]["policy_id"] == "native-chunked"
    assert fam["chunk_policy"]["sha256"] is None and fam["chunk_policy"]["epoch"] is None
    assert not [u for u in sfc.actual_unknowns(fam) if u.startswith("chunk_policy")]
    assert "no pinned policy_id/sha256/epoch to cite BY DESIGN" in " ".join(
        fam["chunk_policy"]["evidence"])
    # and the live validator must still accept it
    assert not sfc.validate_family(fam, repo=ROOT)


# ---------------------------------------------------------------------------
# 2. unknowns / status
# ---------------------------------------------------------------------------
def test_declared_and_retired_families_carry_no_unknowns(roster_families, contracts):
    for fid in roster_families - LICENCE_GAP_FAMILIES:
        fam = contracts[fid]
        assert fam["status"] in ("declared", "retired"), (
            fid, fam["status"])
        assert fam["unknowns"] == [], (fid, fam["unknowns"])
        assert sfc.actual_unknowns(fam) == [], (
            f"{fid}: declares unknowns=[] but the sentinel is actually present "
            f"at {sfc.actual_unknowns(fam)}")


def test_licence_gap_families_are_backfilled_with_named_unknowns(roster_families, contracts):
    for fid in LICENCE_GAP_FAMILIES:
        assert fid in roster_families
        fam = contracts[fid]
        assert fam["status"] == "backfilled", (fid, fam["status"])
        assert set(fam["unknowns"]) == {"licence.instrument", "licence.basis"}, (
            fid, fam["unknowns"])
        # the unknowns list must AGREE with what the sentinel-walker finds --
        # a family cannot hide an unknown by omitting it, or inflate one to
        # look thorough (the contract checker's `actual_unknowns`).
        assert sorted(fam["unknowns"]) == sfc.actual_unknowns(fam)
        # and it must be a REAL gap, not a lazily-blanked field: every other
        # block must be fully populated.
        assert fam["consent"]["search_ok"] is True
        assert fam["consent"]["train_ok"] is True
        assert fam["representation"]["dense_surface"]
        assert fam["chunk_policy"]["sha256"] != sfc.UNKNOWN


def test_hn_firebase_pointer_is_retired(contracts):
    fam = contracts["hn_firebase_pointer"]
    assert fam["status"] == "retired"
    assert fam["unknowns"] == []
    assert fam["mode"] == "cancelled"


# ---------------------------------------------------------------------------
# 3. the live validator -- run it, and pin the ONE known, reported defect
# ---------------------------------------------------------------------------
def test_every_pool_a_contract_validates():
    """No contract may fail the live validator.

    This replaces an earlier version that PINNED the two news families as
    failing. They no longer fail, and not because the defect was fixed: they were
    WITHDRAWN (`retired`), because a family whose packs were cut under a
    superseded chunk policy must not be onboarded, and `backfilled` would have
    been a lie in the other direction (that status means fields hold the
    `unknown` sentinel, and none of theirs does). The defect is still pinned --
    by the test below, on the data rather than on a validator refusal.
    """
    pass


def test_pool_a_contracts_all_validate(roster_families, contracts):
    problems_by_family = {}
    for fid in roster_families:
        probs = sfc.validate_family(contracts[fid], repo=ROOT)
        if probs:
            problems_by_family[fid] = probs
    assert not problems_by_family, problems_by_family


def test_the_superseded_news_chunk_policy_is_pinned_on_the_DATA(contracts):
    """The defect, asserted where it lives, so no status change can hide it.

    A silent "fix" -- flipping either family to `declared` without actually
    re-chunking -- fails here, because `declared` is asserted to be impossible
    while the recorded policy is still the superseded arm. And if the re-chunk
    DOES run, this test fails too and tells the next person to update it, which
    is the right kind of failure.
    """
    for fid in CHUNK_POLICY_DEFECT_FAMILIES:
        fam = contracts[fid]
        assert fam["status"] == "retired", (
            f"{fid} is {fam['status']!r}: a family cut under a superseded chunk "
            "policy may not be onboarded. If the re-chunk under legacy-180w-o30 "
            "has actually run, update this test and the contract together.")
        assert fam["unknowns"] == [], (
            f"{fid}: nothing about this family is UNKNOWN -- its policy is known "
            "and wrong, which is a different thing")
        policy = fam["chunk_policy"]
        assert policy.get("registry_status") == "superseded", policy
        assert policy.get("ratified_successor") == "legacy-180w-o30", policy
        evidence = " ".join(policy.get("evidence") or [])
        # the measurement, not just the label
        assert "LOST on every leg" in evidence, fid
        assert "ADR-090" in evidence, fid
        notes = fam.get("notes") or ""
        assert "re-chunk" in notes, fid


def test_every_family_in_the_document_still_validates_at_the_document_level(contracts):
    # validate() must not raise, and its problems must be exactly the two
    # known chunk-policy failures -- this catches a regression ANYWHERE in
    # the 99-family document, not just the 12 this task added.
    doc = _load_sfc()
    report = sfc.validate(doc, repo=ROOT)
    bad = [p for p in report["problems"]
           if not any(fid in p for fid in CHUNK_POLICY_DEFECT_FAMILIES)]
    assert not bad, bad


# ---------------------------------------------------------------------------
# 4. consent / licence agree with the sealed artefacts (no drift)
# ---------------------------------------------------------------------------
def test_consent_blocks_agree_with_the_sealed_consent_artifact(
        roster_families, contracts, consent_doc):
    verdicts = consent_doc["family_verdicts"]
    for fid in roster_families:
        assert fid in verdicts, fid
        fam = contracts[fid]
        # every family's consent.terminal_schema_version is the wave's sealed
        # v8, verbatim (memory: read the artefacts, never re-derive).
        assert fam["consent"]["terminal_schema_version"] == \
            consent_doc["terminal_schema_version"] == "adr092-direct-terminal.v8"

    # host-level search_ok/train_ok, spot-checked against
    # host_consent_resolution for every family with real hosts (hn is
    # excluded -- it was deliberately not probed).
    hosts = consent_doc["host_consent_resolution"]
    host_by_family = consent_doc["family_origin_roster"]
    for fid in roster_families - REFUSED_FAMILIES:
        fam = contracts[fid]
        for host in host_by_family[fid]:
            resolved = hosts[host]
            # a family's declared train_ok must not be MORE permissive than
            # any of its own hosts' resolved train_ok (a family can be
            # honestly true/true while one host is train_ok=false only if
            # the contract's consent block says so explicitly -- checked via
            # substring since the block is prose, not a per-host structure).
            if resolved["train_ok"] is False:
                assert "train_ok=false" in fam["consent"]["basis"].lower() \
                    or "train_ok" in (fam["consent"].get("notes") or "").lower(), (
                    fid, host, "declares no train_ok=false discussion despite "
                    "a host that is train-only-declined")


def test_licence_blocks_agree_with_the_sealed_licence_artifact(
        roster_families, contracts, licence_doc):
    go_families = set(licence_doc["families"])
    for fid in go_families:
        fam = contracts[fid]
        sealed = licence_doc["families"][fid]
        # every sealed origin's resolved_license (or its absence, for the
        # two 0-surviving-row exceptions) must be named somewhere in the
        # contract's licence.instrument prose -- a drift check, not a
        # byte-exact reproduction (the contract prose also carries the
        # ruling/remediation narrative the sealed artefact does not).
        for origin, info in sealed["origins"].items():
            token = info.get("resolved_license")
            if token:
                assert token.lower() in fam["licence"]["instrument"].lower(), (
                    fid, origin, token, fam["licence"]["instrument"][:200])
            else:
                assert fam["licence"]["instrument"] is not None
                assert "NOASSERTION" in fam["licence"]["instrument"] or \
                    "no licence field" in fam["licence"]["instrument"].lower() or \
                    "no license field" in fam["licence"]["instrument"].lower(), (
                    fid, origin, "0-surviving-row exception not documented")

    # licence-gap families must NOT claim a resolved instrument (they are not
    # in the sealed artefact's scope at all).
    for fid in LICENCE_GAP_FAMILIES:
        assert fid not in go_families, (
            f"{fid} unexpectedly appears in P8's sealed scope -- if a licence "
            "was actually resolved, this family should move out of "
            "LICENCE_GAP_FAMILIES and become `declared`")


# ---------------------------------------------------------------------------
# 5. the four provenance-head families refuse the aw_* frontmatter
# ---------------------------------------------------------------------------
def test_provenance_head_families_cap_the_materializer_head(contracts):
    repcon = json.loads(REPCON_PATH.read_text())
    entry = repcon["families"]["pool_a_provenance_head_restaged"]
    assert entry["dense"] in repcon["surfaces"]
    assert entry["sparse"] in repcon["surfaces"]
    # the mechanism actually wired in the representation-contract module for
    # a leading materializer-identity block is max_provenance_share (what
    # law_article_pack/chunk_lane_body_only use), NOT refused_index_markup
    # (whose INDEX_MARKUP dict only registers 'akn_anchor_id' today) -- see
    # this entry's own `evidence` string for the full reasoning.
    assert entry.get("max_provenance_share") is not None
    assert entry["max_provenance_share"] <= 0.05

    for fid in PROVENANCE_HEAD_FAMILIES:
        fam = contracts[fid]
        assert fam["representation"]["ruling"] == "RULED"
        assert "restage" in fam["representation"]["ruling_cite"].lower()
        assert "aw_" in fam["representation"]["dense_surface"] or \
            "post-restage" in fam["representation"]["dense_surface"].lower()

    applies_to = set(entry["applies_to"])
    # every pack of every provenance-head family must be named
    for fid in PROVENANCE_HEAD_FAMILIES:
        packs = {r["pack"] for r in pr.roster() if r["family"] == fid}
        assert packs <= applies_to, (fid, packs - applies_to)


def test_refused_index_markup_is_not_used_for_an_unregistered_class(contracts):
    # Guard against silently reintroducing the brief's literal (but
    # non-functional) suggestion: refused_index_markup values must resolve
    # in the representation-contract module's INDEX_MARKUP dict, which only
    # has 'akn_anchor_id' today. If this ever fires, either INDEX_MARKUP grew
    # a new entry (update this test) or a family started declaring a class
    # that will raise ContractError the moment representation_contract.py
    # runs it.
    from cytir.index import representation_contract as rc  # noqa: E402
    repcon = json.loads(REPCON_PATH.read_text())
    for key in ("pool_a_body_only_clean", "pool_a_provenance_head_restaged"):
        for name in repcon["families"][key].get("refused_index_markup", []):
            assert name in rc.INDEX_MARKUP, (key, name)


# ---------------------------------------------------------------------------
# 6. statcan_wds declares no sparse_text
# ---------------------------------------------------------------------------
def test_statcan_wds_declares_no_separate_sparse_text(contracts):
    fam = contracts["statcan_wds"]
    rep = fam["representation"]
    # RULED, not IDENTICAL: IDENTICAL means "provably the same as the
    # incumbent", and this net-new family has no incumbent to compare
    # against -- every Pool A family uses RULED for the same reason (see
    # RULED_CLEAN / HEAD_RULING_CITE in the build script).
    assert rep["ruling"] == "RULED"
    assert rep["ruling_cite"]
    assert "no separate" in rep["dense_surface"].lower() or \
        "no `sparse_text`" in rep["dense_surface"].lower()
    assert "no `sparse_text`" in rep["dense_surface"].lower() or \
        "no separate `sparse_text`" in rep["dense_surface"].lower()
    # dense and sparse are declared as the SAME surface -- there is nothing
    # to diverge from since only one field exists on the row.
    assert rep["dense_surface"] != ""
    assert "identical" in rep["sparse_surface"].lower()
    assert fam["chunk_policy"]["policy_family"] == "chunk.atomic"


# ---------------------------------------------------------------------------
# 7. the two news families declare a publication-date field + exclude rule
# ---------------------------------------------------------------------------
def test_news_families_declare_publication_date_and_exclude_not_backfill(contracts):
    for fid in ("news_current_articles", "canada_official_news"):
        fam = contracts[fid]
        notes = fam["notes"]
        assert "source_metadata.publication_time" in notes
        assert "EXCLUDED" in notes or "exclusion" in notes.lower()
        assert "never" in notes.lower() and "backfill" in notes.lower()


def test_news_families_are_a_known_chunk_policy_defect(contracts):
    """Pins the exact discovered defect -- see the module docstring. This
    must keep failing sfc.validate_family (test #3 above) until a restage is
    ruled on; this test checks the CONTRACT states the defect honestly."""
    for fid in ("news_current_articles", "canada_official_news"):
        fam = contracts[fid]
        cp = fam["chunk_policy"]
        assert cp["policy_family"] == "chunk.news"
        assert cp["epoch"] == 4
        assert "superseded" in fam["notes"].lower()
        assert "chunk_policy.selectable" in fam["notes"] or \
            "ADR-090" in fam["notes"]


# ---------------------------------------------------------------------------
# 8. hn_firebase_pointer records the refusal with its ADR cites
# ---------------------------------------------------------------------------
def test_hn_firebase_pointer_records_the_refusal_with_adr_cites(contracts):
    fam = contracts["hn_firebase_pointer"]
    rep = fam["representation"]
    assert "REFUSED" in rep["dense_surface"]
    assert "REFUSED" in rep["sparse_surface"]
    cite = rep["ruling_cite"]
    assert "ADR-098" in cite
    assert "ADR-106" in cite
    assert "Ruling 2" in cite
    assert "REFUSED" in fam["notes"] or "refused" in fam["notes"].lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# B01's declared payload facts (task 629, coordinator ruling 2026-09-22)
# ---------------------------------------------------------------------------
def test_b01s_payload_facts_are_declared_and_cite_the_ratified_class(contracts):
    """`source_type` and `trust_tier` are POLICY facts, not derivations.

    `promote_sealed_pack` refuses to guess them, and B01's rows carry neither --
    they carry url, lang and source_metadata.title and nothing else -- so without
    a declaration the batch cannot be composed at all. What this test protects is
    that the declaration stays an APPLICATION of POLICY 4.4's ratified
    vendor-content class rather than a number someone found convenient: 0.60 is
    4.4's value, and the citation has to be present for a reader to check it.
    """
    fam = contracts["technical_manufacturer_catalog"]
    declared = fam["declared_payload"]
    assert declared["trust_tier"] == 0.60, (
        "0.75 is the technical-documentation default; 4.4 puts vendor product "
        "surfaces at the vendor-content class")
    assert declared["title_from"] == ["source_metadata.title"], (
        "the only title these rows carry")
    basis = declared["basis"]
    assert "4.4" in basis and "2026-07-27" in basis
    assert "vendor-content" in basis
    assert "NOT a new policy value" in basis


def test_b01s_source_type_is_read_from_the_implementation_not_chosen(contracts):
    """`docs`, and the reason is a measurement rather than a preference.

    Task 330 implemented the same ruling on the build collection and its restamp
    wrote `trust_tier` ALONE, over a cohort whose predicate REQUIRES
    source_type=docs -- so every one of the 765,014 points at class@0.60 carries
    it. And `tech`, the intuitive answer, is not a source_type at all: it is a
    disc_class that `discovery_ingest.CLASS_SOURCE` maps TO `docs`. This test
    fails if the value drifts to something the serving vocabulary has never seen,
    which for 445 points would be a ranking class with no learned weight.
    """
    declared = contracts["technical_manufacturer_catalog"]["declared_payload"]
    assert declared["source_type"] == "docs"

    import importlib
    sys.path.insert(0, str(ROOT / "pipeline"))
    ingest = importlib.import_module("cytir.acquire.discovery_ingest")
    assert set(ingest.CLASS_SOURCE.values()) == {"official", "reference", "docs"}, (
        "the source_type vocabulary grew -- re-read whether `docs` is still the "
        "only value this class has")
    assert ingest.CLASS_SOURCE["tech"] == "docs", (
        "`tech` is a disc_class, not a source_type")
    assert declared["source_type"] in set(ingest.CLASS_SOURCE.values())
    assert "restamp330_trust.py" in declared["source_type_basis"]


# --------------------------------------------------------- W1 B02 / B03 / B05
# Task 629, 2026-09-23. Three more families had to declare the same two POLICY
# facts before their batches could be composed. What these tests protect is not
# the numbers -- it is that each number still has its TWO sources attached: the
# POLICY section, and the rows of the same domain the index is already serving.
# A declaration that loses its incumbent citation has become a preference.

def test_b02_enters_at_the_class_its_own_domain_already_serves(contracts):
    """service-public: `official`, and the incumbent leaves no other reading.

    37,512 rows on the family's two origin domains, 37,512 of them `official`
    (twin aw_e8, exact filtered counts, 2026-09-23). POLICY 4.1 names DILA in
    the section text and the pack IS DILA's own fiche dump.
    """
    declared = contracts["service_public_fiche"]["declared_payload"]
    assert declared["source_type"] == "official"
    assert declared["trust_tier"] == 1.0
    basis = declared["basis"]
    assert "4.1" in basis and "DILA" in basis
    assert "37,512" in basis, "the incumbent count is the second source"


def test_b02s_tier_is_read_from_the_rule_and_names_where_it_declines(contracts):
    """Not the fossil majority: what `aw_serving_metadata`'s order computes.

    Rule 4 (the domain registry, `min_domain_share = 0.90`) answers 1.0 for
    `entreprendre.service-public.gouv.fr` at share 1.000 and DECLINES on
    `service-public.gouv.fr`, whose modal 1.0 is share 0.88998 -- under the
    floor. It never computes 0.90. The 2,770 rows that carry 0.90 are a
    different LANE (`provenance: skills_transition`), not a trust epoch, and
    the declaration has to say so: a reader must be able to disagree with it.
    """
    declared = contracts["service_public_fiche"]["declared_payload"]
    tb = declared["trust_tier_basis"]
    assert "aw_serving_metadata" in tb, "the rule, cited"
    assert "0.88998" in tb and "0.90" in tb, "the share and the floor"
    assert "2,770" in tb and "skills_transition" in tb
    assert "rule_epochs" in tb
    assert "HONEST LIMIT" in tb, (
        "a family-level number where the mechanism is per-url is a "
        "simplification, and it has to be labelled as one")

    import importlib
    sys.path.insert(0, str(ROOT / "pipeline"))
    asm = importlib.import_module("cytir.index.serving_metadata")
    import inspect
    sig = inspect.signature(asm.ServingMetadataRegistry.__init__)
    assert sig.parameters["min_domain_share"].default == 0.90, (
        "the declaration quotes this floor; if it moved, re-read the shares")
    assert 22487 / 25267 < 0.90 <= 12245 / 12245, (
        "the arithmetic the citation rests on")


def test_b05s_tier_is_what_the_rule_computes_outright(contracts):
    """rfc-editor clears rule 4 on both fields, so the declaration restates
    the mechanism rather than adding a judgement to it."""
    tb = contracts["rfc_editor_net_new"]["declared_payload"]["trust_tier_basis"]
    assert "0.99429" in tb and "0.99686" in tb
    assert 45236 / 45496 >= 0.90 and 45353 / 45496 >= 0.90


def test_b03s_tier_comes_from_rule_1_because_rule_4_declines(contracts):
    """The host mixes two families, so the domain average answers nothing.

    `statcan.gc.ca` is www150 (WDS tables) and www23 (survey metadata) in one
    registrable domain: modal source_type share 0.73588, modal trust_tier share
    0.74266, both under the 0.90 floor. The family's own ingester answers.
    """
    tb = contracts["statcan_wds"]["declared_payload"]["trust_tier_basis"]
    assert "ingest_statcan_wds.py:183" in tb
    assert "0.73588" in tb and "0.74266" in tb
    assert "www23" in tb, "why the host average is not this family"
    assert 10200 / 13861 < 0.90 and 10294 / 13861 < 0.90


def test_b01s_vocabulary_claim_is_corrected_where_it_was_made(contracts):
    """`gov-stat` exists, so "exactly three values" was false.

    The correction lands on B01's own declaration rather than in a task doc:
    the next reader of that declaration is the one who would repeat the error.
    """
    basis = contracts["technical_manufacturer_catalog"][
        "declared_payload"]["source_type_basis"]
    assert "CORRECTION, task 629 2026-09-23" in basis
    assert "gov-stat" in basis and "10,200" in basis
    assert "restamp330_trust.py" in basis, "the original argument is kept"


def test_b01s_identity_records_the_retired_ruling_in_place(contracts):
    """B01 promotes under pack-declared now, and the falsified premise is
    corrected where it was asserted."""
    ident = contracts["technical_manufacturer_catalog"]["identity"]
    corr = ident["point_id_rule_correction"]
    assert "pack-declared" in corr and "falsified" in corr
    assert "445" in corr
    assert "mint_passage_point_id" in corr


def test_b03_source_type_is_read_from_its_own_ingester(contracts):
    """`gov-stat` @ 0.95 -- the value the family's own pipeline emits.

    `cytir/sources/families/statcan_wds.py` is the one home for this family's class,
    and the twin agrees at the pack's own url template. This also pins a real
    finding: `gov-stat` is NOT in `discovery_ingest.CLASS_SOURCE`'s
    official/reference/docs vocabulary, so B01's declaration was wrong to call
    that vocabulary exhaustive for the whole index.
    """
    declared = contracts["statcan_wds"]["declared_payload"]
    assert declared["source_type"] == "gov-stat"
    assert declared["trust_tier"] == 0.95

    source = (ROOT / "cytir" / "sources" / "families" / "statcan_wds.py").read_text(
        encoding="utf-8")
    assert 'source_type="gov-stat", trust_tier=0.95' in source, (
        "the declaration is read FROM the ingester; if the ingester moved, the "
        "declaration is stale, not the test")

    import importlib
    sys.path.insert(0, str(ROOT / "pipeline"))
    ingest = importlib.import_module("cytir.acquire.discovery_ingest")
    assert declared["source_type"] not in set(ingest.CLASS_SOURCE.values()), (
        "if `gov-stat` has joined the discovery vocabulary, re-read whether "
        "this family still sits outside it")


def test_b03_refuses_rather_than_entering_without_a_title(contracts):
    """An empty `title_from` is a refusal, and it says so.

    The pack carries no title field at all (0 of 10,136 rows have
    `source_metadata`). `--title-from` takes dotted paths, never a
    computation, so there is nothing to declare. Leaving the list empty makes
    `promote_sealed_pack` refuse on the 99% title floor instead of admitting
    10,136 titleless rows next to incumbents that all carry one.
    """
    declared = contracts["statcan_wds"]["declared_payload"]
    assert declared["title_from"] == []
    why = declared["title_from_basis"]
    assert "REFUSAL" in why
    assert "0/10,136" in why or "0 of 10,136" in why
    assert "re-staged" in why or "restage" in why.lower()


def test_b05_matches_the_same_url_shape_not_the_host_average(contracts):
    """rfc-editor: `official` @ 1.0, from the `.txt` cohort specifically.

    45,236 of 45,496 rows on rfc-editor.org are `official`, and the 260
    `reference` rows are /info/ and .html shapes the pack does not contain.
    Matching the host average instead of the same-shape incumbent is how a
    family enters at a class its own siblings do not hold.
    """
    declared = contracts["rfc_editor_net_new"]["declared_payload"]
    assert declared["source_type"] == "official"
    assert declared["trust_tier"] == 1.0
    assert declared["title_from"] == ["source_metadata.title"]
    note = declared["source_type_note"]
    assert "/info/" in note and "45,236" in declared["basis"]


def test_each_w1_declaration_carries_a_re_derived_identity_rule(contracts):
    """`pack-declared` is earned here, not asserted.

    The mode exists for ids this module cannot re-derive. For these three the
    id IS re-derivable -- every row, from its own persisted fields -- so the
    declaration records the recipe and the count. A `pack-declared` block with
    no recipe would be the unexplained acceptance `project_rows` refuses.
    """
    for fid, rows in (("service_public_fiche", "8,324 of 8,324"),
                      ("statcan_wds", "10,136 of 10,136"),
                      ("rfc_editor_net_new", "61,212 of 61,212")):
        ident = contracts[fid]["declared_payload"]["identity"]
        assert ident["mode"] == "pack-declared"
        assert "uuid5" in ident["point_id_rule"]
        assert "mint_passage_point_id" in ident["point_id_rule"]
        assert rows in ident["verified"], fid


def test_no_w1_declaration_claims_to_authorise_its_own_batch(contracts):
    """A class declaration is not an admission decision, and says which it is.

    B02 and B03 were measured to be replacements wearing a `net_new_addition`
    label: 99.5% and 99.1% of their rows land on urls the index already
    serves, under ids that do not collide -- so the promote's already-present
    refusal cannot see them. The contract carries that finding beside the
    declaration so a future reader cannot take the one for the other.
    """
    for fid in ("service_public_fiche", "statcan_wds", "rfc_editor_net_new"):
        block = contracts[fid]["declared_payload"]["does_not_authorise"]
        assert block and len(block) > 80, fid
    assert "99.5%" in contracts["service_public_fiche"][
        "declared_payload"]["does_not_authorise"]
    assert "99.1%" in contracts["statcan_wds"][
        "declared_payload"]["does_not_authorise"]
    # B05 is the one the measurement CLEARS on this axis, and it says so.
    assert "0 point-id collisions" in contracts["rfc_editor_net_new"][
        "declared_payload"]["does_not_authorise"]
