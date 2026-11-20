#!/usr/bin/env python3
"""The register-first card engine, against what production actually holds.

Every test is a defect this module exists to have refused, stated as the thing
that actually shipped:

  * `resolve_confidence` reaching production on 0 of 337,702 live entity cards,
    against ADR-041 §5's ratified "store the surviving score in each card payload
    so we can also re-gate live if needed" (measured 2026-09-19, `aw_e8`);
  * `legal_form` non-null on 5,696 rows of the July generation and 0 rows of
    today's, with no contract line able to see it go;
  * a card whose point id is neither the ratified `aw_b1_identity` id nor the
    legacy one, so applying it ADDS a second point at a URL production already
    holds (task 620's staged impactIA card, id `3b907382-…`);
  * a card admitted on a name match alone — the foreign-homonym class
    (`comodo.com` -> "Comodo Sàrl", caught in task 224's own 25-card sample);
  * a `register_id ==` join across two stores where production carries three
    spellings of the same scheme (`CHE` 92,187 / `CHE-UID` 88,801 / `CH-UID` 587);
  * a card emitted with `source_type` other than `entity_card`, the state 876 CH
    points in production are in today.

Run: python3 -m pytest tests/test_aw_corpus_register_cards.py
"""
from __future__ import annotations

import json

import pytest

from cytir.core import ids as aw_b1_identity
import cytir.corpus.register_cards as RF


def seed(**over):
    s = dict(
        register_id="CHE-396.275.781", register_scheme="CHE-UID",
        legal_name="impactIA",
        purpose="créer une entité dotée de ressources afin d'anticiper les "
                "bouleversements majeurs engendrés par l'Intelligence Artificielle",
        officers=["Timothy Shane O'Hear (président)"],
        legal_form="Fondation", legal_form_code="0110",
        canton="GE", locality="Genève", postcode="1201",
        street="rue de Chantepoulet 10",
        country="CH", lang="fr", domain="impactia.org",
        url="https://impactia.org/",
        license="Swiss public commercial register (Zefix/LINDAS, OGD)",
        provenance="zefix_register_first_task621",
        last_seen="2026-09-19",
    )
    s.update(over)
    return s


# ------------------------------------------------------------------ the join key
def test_normalize_register_id_folds_all_three_live_spellings():
    """Production carries the Swiss scheme three ways. Any count that crosses a
    store boundary must normalise first, or it misses half the population."""
    assert (RF.normalize_register_id("CHE-396.275.781")
            == RF.normalize_register_id("CHE396275781")
            == RF.normalize_register_id("che 396 275 781")
            == "CHE396275781")


def test_normalize_register_id_on_nothing():
    assert RF.normalize_register_id(None) is None
    assert RF.normalize_register_id("") is None
    assert RF.normalize_register_id("---") is None


def test_normalize_is_scheme_agnostic_so_it_works_for_siren_and_company_number():
    assert RF.normalize_register_id("552 032 534") == "552032534"
    assert RF.normalize_register_id("SC-123456") == "SC123456"


def test_uid_checksum_rejects_a_plausible_but_invalid_uid():
    assert RF.uid_checksum_ok("396275781") is True
    assert RF.uid_checksum_ok("396275782") is False
    assert RF.uid_checksum_ok("39627578") is False
    assert RF.uid_checksum_ok("abcdefghi") is False


# -------------------------------------------------------------- the verify contract
def test_name_alone_does_not_satisfy_the_contract():
    """The foreign-homonym class. A global `.com` carrying the same trade name is
    not the Swiss entity, and this is the only check that says so."""
    ev = RF.verify_pair(register_id="CHE-109.123.456", legal_name="Comodo Sàrl",
                        domain="comodo.com",
                        page_text="Comodo — cybersecurity for the enterprise. "
                                  "Roseland, New Jersey.",
                        http_status=200, final_host="comodo.com",
                        street="Route de Lausanne 3", postcode="1030", city="Bussigny")
    assert ev["name_match"] is True
    assert ev["address_tokens"] == []
    assert ev["contract_satisfied"] is False
    assert RF.grade_pair(ev, legal_name="Comodo Sàrl", domain="comodo.com") < RF.DEFAULT_THRESHOLD


def test_name_plus_one_address_token_satisfies_the_contract():
    ev = RF.verify_pair(register_id="CHE-396.275.781", legal_name="impactIA",
                        domain="impactia.org",
                        page_text="impactIA — fondation. 1201 Genève.",
                        http_status=200, final_host="impactia.org",
                        street="rue de Chantepoulet 10", postcode="1201", city="Genève")
    assert ev["contract_satisfied"] is True
    assert "postcode:1201" in ev["address_tokens"]


def test_valid_uid_on_page_is_a_certainty_override_on_any_tld():
    ev = RF.verify_pair(register_id="CHE-396.275.781", legal_name="Totally Other Name",
                        domain="example.com",
                        page_text="Impressum CHE-396.275.781",
                        http_status=200, final_host="example.com")
    assert ev["uid_on_page"] is True
    assert ev["contract_satisfied"] is True
    assert RF.grade_pair(ev, legal_name="Totally Other Name", domain="example.com") == 1.0


def test_a_different_valid_uid_on_the_page_is_not_this_entity():
    """ADR-042's precision gate: a brand-token domain whose on-page UID belongs to
    a DIFFERENT company is rejected, not admitted at a lower score."""
    ev = RF.verify_pair(register_id="CHE-396.275.781", legal_name="impactIA",
                        domain="service.ch",
                        page_text="Impressum CHE-105.933.557",
                        http_status=200, final_host="service.ch")
    assert ev["uid_on_page"] is False


def test_the_address_token_must_survive_past_the_head_of_the_page():
    """Task 224's BUG-2: a `html[:150000]` clip discarded the footer, where the
    imprint lives. impactIA's address tokens start at offset 161,507."""
    page = ("x" * 161_000) + " impactIA rue de Chantepoulet 10, 1201 Genève"
    ev = RF.verify_pair(register_id="CHE-396.275.781", legal_name="impactIA",
                        domain="impactia.org", page_text=page,
                        http_status=200, final_host="impactia.org",
                        street="rue de Chantepoulet 10", postcode="1201", city="Genève")
    assert ev["contract_satisfied"] is True
    assert len(ev["address_tokens"]) == 3


def test_street_number_mismatch_does_not_lose_the_street_token():
    ev = RF.verify_pair(register_id="CHE-396.275.781", legal_name="impactIA",
                        domain="impactia.org",
                        page_text="impactIA, rue de Chantepoulet 10bis, Genève",
                        http_status=200, final_host="impactia.org",
                        street="rue de Chantepoulet 10", city="Genève")
    assert any(a.startswith("street:") for a in ev["address_tokens"])


def test_postcode_token_does_not_match_inside_a_longer_number():
    ev = RF.verify_pair(register_id="CHE-396.275.781", legal_name="impactIA",
                        domain="impactia.org",
                        page_text="impactIA order no 812015566",
                        http_status=200, final_host="impactia.org", postcode="1201")
    assert ev["address_tokens"] == []


def test_a_redirect_into_an_aggregator_is_rejected_outright():
    """Being redirected INTO moneyhouse means the candidate is not the entity's
    own site — and harvesting those directories is the ADR-040 violation."""
    ev = RF.verify_pair(register_id="CHE-396.275.781", legal_name="impactIA",
                        domain="impactia.ch",
                        page_text="impactIA 1201 Genève rue de Chantepoulet",
                        http_status=200, final_host="www.moneyhouse.ch",
                        postcode="1201", city="Genève")
    assert ev["redirect_aggregator"] is True
    assert ev["contract_satisfied"] is False
    assert RF.grade_pair(ev, legal_name="impactIA", domain="impactia.ch") == 0.0


def test_a_parked_page_is_rejected():
    ev = RF.verify_pair(register_id="CHE-396.275.781", legal_name="impactIA",
                        domain="impactia.ch",
                        page_text="impactIA — this domain is parked. 1201 Genève",
                        http_status=200, final_host="impactia.ch",
                        postcode="1201")
    assert ev["parked"] is True
    assert ev["contract_satisfied"] is False


# ---------------------------------------------------------------- graded confidence
def test_generic_names_score_below_distinctive_ones():
    """ADR-041's fix for the generic-name flood that sank task 220 to 55%."""
    assert (RF.name_distinctiveness("Swiss Solutions AG")
            < RF.name_distinctiveness("Galaxus AG"))
    assert RF.name_distinctiveness("impactIA") > 0.3
    assert RF.name_distinctiveness("") == 0.0


def test_a_locked_tld_needs_less_corroboration_than_a_gtld():
    """The score subsumes the per-TLD rule, so there is no if/else chain."""
    ev = {"name_match": True, "name_full": True, "address_tokens": ["city:Bulle"],
          "uid_on_page": False, "parked": False, "redirect_aggregator": False}
    swiss = RF.grade_pair(ev, legal_name="Sonart Sàrl", domain="sonart.swiss")
    com = RF.grade_pair(ev, legal_name="Sonart Sàrl", domain="sonart.com")
    assert swiss > com


def test_a_stronger_address_token_scores_higher():
    base = {"name_match": True, "name_full": False, "uid_on_page": False,
            "parked": False, "redirect_aggregator": False}
    NAME, DOM = "Swiss Trading Services AG", "swisstrading.com"
    s_street = RF.grade_pair({**base, "address_tokens": ["street:Bahnhofstrasse"]},
                             legal_name=NAME, domain=DOM)
    s_pc = RF.grade_pair({**base, "address_tokens": ["postcode:8001"]},
                         legal_name=NAME, domain=DOM)
    s_city = RF.grade_pair({**base, "address_tokens": ["city:Zürich"]},
                           legal_name=NAME, domain=DOM)
    s_none = RF.grade_pair({**base, "address_tokens": []},
                           legal_name=NAME, domain=DOM)
    assert s_street > s_pc > s_city > s_none
    # a generic name on a gTLD with only a city token must NOT reach admission
    assert s_city < RF.DEFAULT_THRESHOLD


def test_confidence_is_always_in_range():
    ev = {"name_match": True, "name_full": True, "uid_on_page": False,
          "parked": False, "redirect_aggregator": False,
          "address_tokens": ["street:a", "postcode:b", "city:c"]}
    c = RF.grade_pair(ev, legal_name="Distinctive Coinedword", domain="x.swiss")
    assert 0.0 <= c <= 1.0


# ------------------------------------------------------------------- the identity
def test_point_id_is_the_ratified_aw_b1_id_production_already_holds():
    """This is the difference between REPAIRING the row production holds at a URL
    and ADDING a second point beside it. `https://impactia.org/` is one point in
    `aw_e8`, id acf1277c-dc45-579e-a894-1a28232e0f7a."""
    rec = RF.compose_register_card(seed(), evidence={"uid_on_page": False,
                                                    "address_tokens": []},
                                   confidence=1.0, env={})
    assert rec["point_id"] == "acf1277c-dc45-579e-a894-1a28232e0f7a"
    assert rec["point_id"] == aw_b1_identity.mint_point_id("https://impactia.org/", "")
    assert rec["tier"] == "C"


def test_point_id_is_not_task_620s_uid_addressed_id():
    rec = RF.compose_register_card(seed(), evidence={}, confidence=1.0, env={})
    # the staged pack's id, a literal (it is uuid5(NAMESPACE_URL, "che:CHE396275781"));
    # a uuid5( call here would make this file a new point-id-minting offender
    task620 = "3b907382-8281-5c66-854a-1d3550cba6ea"
    assert rec["point_id"] != task620


# ----------------------------------------------------------- the register surface
def test_dense_surface_is_the_register_composition_head_first():
    rec = RF.compose_register_card(seed(), evidence={}, confidence=1.0, env={})
    assert rec["dense_text"].startswith("impactIA — créer une entité")
    assert "Registered signatories/officers (commercial register): " in rec["dense_text"]
    assert "Timothy Shane O'Hear (président)" in rec["dense_text"]


def test_officers_come_from_the_seed_and_the_seed_only():
    """`card-officers-from-register-only`: there is no website input to this
    function, so there is no path by which a scraped name can become an officer."""
    import ast
    import inspect
    import textwrap
    fn = ast.parse(textwrap.dedent(inspect.getsource(RF.compose_register_card))).body[0]
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)):
        fn.body = fn.body[1:]           # drop the docstring, keep the code
    body = ast.unparse(fn)
    assert "website" not in body.lower(), (
        "the composer has no website input, so no scraped name can become an officer")
    assert "officers" in body
    rec = RF.compose_register_card(seed(officers=[]), evidence={}, confidence=1.0,
                                   env={})
    assert "signatories/officers" not in rec["dense_text"]


def test_a_card_with_no_purpose_falls_back_to_the_REGISTER_seat_not_the_title():
    """78 of this task's own 715 candidates have no Zefix Zweck. A dense surface
    that is only the card's own title is no surface at all — and the fallback must
    still come from the register, never from the website."""
    rec = RF.compose_register_card(seed(purpose=None, officers=[],
                                        legal_form="Fondation"), evidence={},
                                   confidence=0.9, env={})
    assert rec["dense_text"] == ("impactIA — Fondation. Genève, GE. "
                                 "Registered in the Swiss commercial register.")
    assert rec["dense_text"] != rec["title"] + "."


def test_the_no_purpose_fallback_degrades_to_the_name_when_the_register_gives_nothing():
    rec = RF.compose_register_card(
        seed(purpose=None, officers=[], legal_form=None, locality=None, canton=None),
        evidence={}, confidence=0.9, env={})
    assert rec["dense_text"] == "impactIA. Registered in the Swiss commercial register."


def test_sparse_channel_carries_both_register_id_spellings_as_hooks():
    rec = RF.compose_register_card(seed(), evidence={}, confidence=1.0, env={})
    assert "CHE396275781" in rec["sparse_text"]
    assert "Genève" in rec["sparse_text"]


def test_sparse_channel_is_segmented_from_the_dense_chunk():
    from cytir.core import projection as aw_b1_projection
    rec = RF.compose_register_card(seed(), evidence={}, confidence=1.0, env={})
    assert aw_b1_projection.SPARSE_SEGMENT_SEPARATOR.strip() in rec["sparse_text"]


# ------------------------------------------------------------- the payload contract
def test_resolve_confidence_is_on_every_card():
    """ADR-041 §5, unmet on 337,702 of 337,702 live cards."""
    rec = RF.compose_register_card(seed(), evidence={"uid_on_page": True},
                                   confidence=1.0, env={})
    assert rec["resolve_confidence"] == 1.0
    assert rec["resolve_uid_on_page"] is True
    assert rec["resolve_evidence_class"] == "uid_on_page"
    RF.assert_payload_contract(rec, env={})


@pytest.mark.parametrize("field", list(RF.REQUIRED_PAYLOAD_FIELDS))
def test_contract_fails_closed_on_each_missing_required_field(field):
    rec = RF.compose_register_card(seed(), evidence={}, confidence=1.0, env={})
    rec.pop(field)
    with pytest.raises(RF.PayloadContractRefusal) as e:
        RF.assert_payload_contract(rec, env={})
    assert field in str(e.value)


@pytest.mark.parametrize("field", ["register_id", "country", "register_scheme",
                                   "license", "provenance"])
def test_contract_fails_on_a_field_that_is_present_but_empty(field):
    """The defect a key-presence check cannot see: `legal_form_code` was a key on
    all 6,172 July rows with 0 non-null values."""
    rec = RF.compose_register_card(seed(), evidence={}, confidence=1.0, env={})
    rec[field] = ""
    with pytest.raises(RF.PayloadContractRefusal) as e:
        RF.assert_payload_contract(rec, env={})
    assert field in str(e.value)


def test_contract_refuses_a_non_card_source_type():
    """The state 876 CH points in production are in: a register_id and a
    zefix provenance under source_type=official, the register surface gone."""
    rec = RF.compose_register_card(seed(), evidence={}, confidence=1.0, env={})
    rec["source_type"] = "official"
    with pytest.raises(RF.PayloadContractRefusal) as e:
        RF.assert_payload_contract(rec, env={})
    assert "876" in str(e.value)


def test_contract_refuses_a_confidence_outside_range():
    rec = RF.compose_register_card(seed(), evidence={}, confidence=1.0, env={})
    rec["resolve_confidence"] = 1.4
    with pytest.raises(RF.PayloadContractRefusal):
        RF.assert_payload_contract(rec, env={})


# ------------------------------------------------------- legal_form behind the flag
def test_legal_form_is_absent_until_the_contract_declares_it(tmp_path):
    """A payload key the contract has never been told about is exactly the silent
    change the contract exists to refuse, so the engine leaves it out."""
    undeclared = tmp_path / "c.json"
    undeclared.write_text(json.dumps({"families": {"entity_card": {
        "identity_fields": ["title", "register_purpose"]}}}), encoding="utf-8")
    assert RF.legal_form_enabled({}, path=undeclared) is False
    old = RF.CONTRACT_PATH
    RF.CONTRACT_PATH = undeclared
    try:
        rec = RF.compose_register_card(seed(), evidence={}, confidence=1.0, env={})
    finally:
        RF.CONTRACT_PATH = old
    assert "legal_form" not in rec
    assert "legal_form_code" not in rec


def test_the_ratification_is_the_switch_not_a_hand_flipped_flag(tmp_path):
    """When the contract PR merges, the producer starts writing the field with no
    second change anywhere. The alternative leaves a window in which the contract says
    one thing and the producer does another -- the exact disagreement that let
    legal_form reach 0 non-null on all 337,702 live cards unnoticed."""
    declared = tmp_path / "d.json"
    declared.write_text(json.dumps({"families": {"entity_card": {
        "declared_payload_fields": ["legal_form", "legal_form_code"]}}}),
        encoding="utf-8")
    assert RF.legal_form_enabled({}, path=declared) is True


def test_an_unreadable_contract_fails_closed(tmp_path):
    bad = tmp_path / "broken.json"
    bad.write_text("{ not json", encoding="utf-8")
    assert RF.legal_form_enabled({}, path=bad) is False
    assert RF.legal_form_enabled({}, path=tmp_path / "absent.json") is False


def test_legal_form_is_written_and_required_once_the_flag_is_set():
    env = {"AW_CARD_LEGAL_FORM": "1"}
    rec = RF.compose_register_card(seed(), evidence={}, confidence=1.0, env=env)
    assert rec["legal_form"] == "Fondation"
    assert rec["legal_form_code"] == "0110"
    RF.assert_payload_contract(rec, env=env)
    rec.pop("legal_form")
    with pytest.raises(RF.PayloadContractRefusal) as e:
        RF.assert_payload_contract(rec, env=env)
    assert "legal_form" in str(e.value)


def test_the_override_works_in_both_directions(tmp_path):
    declared = tmp_path / "d.json"
    declared.write_text(json.dumps({"families": {"entity_card": {
        "declared_payload_fields": ["legal_form"]}}}), encoding="utf-8")
    undeclared = tmp_path / "u.json"
    undeclared.write_text(json.dumps({"families": {"entity_card": {}}}),
                          encoding="utf-8")
    for v in ("0", "false", "no"):
        assert RF.legal_form_enabled({"AW_CARD_LEGAL_FORM": v}, path=declared) is False
    for v in ("1", "true", "TRUE", "yes"):
        assert RF.legal_form_enabled({"AW_CARD_LEGAL_FORM": v}, path=undeclared) is True
    for v in ("", "  ", "maybe"):
        assert RF.legal_form_enabled({"AW_CARD_LEGAL_FORM": v}, path=undeclared) is False
        assert RF.legal_form_enabled({"AW_CARD_LEGAL_FORM": v}, path=declared) is True


def test_the_engine_reads_the_REAL_contract_in_the_repo():
    """Whatever the repo's own contract says today, the producer agrees with it. The
    coupling is checked against the live file rather than a fixture, so the answer
    changes when and only when the ratification lands."""
    if not RF.CONTRACT_PATH.is_file():
        pytest.skip(f"{RF.CONTRACT_PATH} is withheld from this cut")
    doc = json.loads(RF.CONTRACT_PATH.read_text(encoding="utf-8"))
    fam = doc["families"]["entity_card"]
    declared = (set(fam.get("declared_payload_fields") or [])
                | set(fam.get("identity_fields") or [])
                | set(fam.get("identity_fields_each") or [])
                | set(fam.get("sparse_identity_fields") or []))
    assert RF.contract_declares("legal_form") == ("legal_form" in declared)
    assert RF.legal_form_enabled({}) == ("legal_form" in declared)
    # register_id and resolve_confidence are required unconditionally: they are
    # ratified already (specs/entity-cards-phase2-companies.md; ADR-041 SS5), so the
    # engine does not consult the file for them.
    assert "register_id" in RF.REQUIRED_PAYLOAD_FIELDS
    assert "resolve_confidence" in RF.REQUIRED_PAYLOAD_FIELDS


# ------------------------------------------- any present-but-empty field is refused
def test_contract_refuses_ANY_present_but_empty_field_not_only_required_ones():
    """The defect that bit this task's own pack driver: it read a consent field the
    resolver does not emit (`consent_state`; the receipt carries `status`) and wrote it
    as null on 506 of 507 staged rows, past a check that only inspected its own
    required list. A key that is present and empty is never informative."""
    rec = RF.compose_register_card(seed(), evidence={"address_tokens": ["city:Gen\u00e8ve"]},
                                   confidence=1.0, env={})
    RF.assert_payload_contract(rec, env={})          # clean to start with
    rec["consent_state"] = None
    with pytest.raises(RF.PayloadContractRefusal) as e:
        RF.assert_payload_contract(rec, env={})
    assert "consent_state" in str(e.value)


def test_an_empty_address_token_list_is_omitted_not_written():
    """An absent field says 'not applicable'; an empty list says 'we looked and found
    none'. A UID-on-page admission needed no address token, so the field is left out."""
    uid_only = RF.compose_register_card(seed(), evidence={"uid_on_page": True,
                                                          "address_tokens": []},
                                        confidence=1.0, env={})
    assert "resolve_address_tokens" not in uid_only
    assert uid_only["resolve_evidence_class"] == "uid_on_page"
    RF.assert_payload_contract(uid_only, env={})
    with_addr = RF.compose_register_card(
        seed(), evidence={"address_tokens": ["postcode:1201"]}, confidence=0.95, env={})
    assert with_addr["resolve_address_tokens"] == ["postcode:1201"]


def test_a_false_boolean_is_not_treated_as_empty():
    """`resolve_uid_on_page: False` and `non_refreshable: False` are real answers."""
    rec = RF.compose_register_card(seed(), evidence={"address_tokens": ["city:Gen\u00e8ve"]},
                                   confidence=0.95, env={})
    assert rec["resolve_uid_on_page"] is False
    rec["non_refreshable"] = False
    RF.assert_payload_contract(rec, env={})
