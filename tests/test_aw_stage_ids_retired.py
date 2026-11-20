"""Task 1349 (Q5(a)): the 19 assembly / pipeline stage ids are retired rows, not sources.

A stage id stays declared (the claim-manifest roster names it and served points may carry it),
so every one must still resolve by id; none may be a runnable family in the registry.
"""
import pytest

from cytir.sources import contract, registry

STAGE_IDS = (
    "natural_person_policy_drop", "task427_postconsent", "task467_entity_card_1024_reembed",
    "task471_f2_pdf_docling_epoch4_chunks", "task472_cc_materializer_1024_embed",
    "task472_f2_html_1024_embed", "task472_rii_courts_1024_embed",
    "task477_direct_capture_epoch4_chunks", "task482_cc_consent_supplement_chunks",
    "task482_f2_consent_supplement_chunks", "task492_crawl30_cc_chunks",
    "task494_4crawl_supplement_cc_chunks", "task495_r7_stale_supplement_cc_chunks",
    "task500_news_1024_add", "task500_news_1024_in_place", "task501_news_1024_add",
    "task505_narrow_b2_track1a_chunks", "task505_narrow_d_1024_reembed", "track1a_retained_bind",
)


def test_stage_ids_are_retired_but_still_resolve_by_id():
    doc = contract.load()
    rows = contract.by_id(doc)
    assert len(STAGE_IDS) == 19
    for fid in STAGE_IDS:
        assert rows[fid]["status"] == "retired", fid


def test_registry_roster_holds_no_stage_id():
    assert not set(STAGE_IDS) & set(registry.registered())


def test_contract_validates_and_roster_still_covers_every_stage_id():
    doc = contract.load()
    try:
        roster = contract.roster(doc)
    except contract.FamilyContractError as exc:
        # The sha-pinned claim-manifest roster is not part of the public cut;
        # inside that tree validate() can only report the missing source.
        pytest.skip(f"roster source not in this tree: {exc}")
    report = contract.validate(doc)
    assert report["verdict"] == "PASS", report["problems"][:3]
    assert set(STAGE_IDS) <= roster
