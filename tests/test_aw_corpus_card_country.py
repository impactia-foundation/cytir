"""Task 2008: the card payload `country` is stamped from the register source at
the one composition point (`finish`), payload only: the composition digest and
the dense/sparse text do not change."""
from __future__ import annotations

import pytest

from cytir.corpus import card_recipe as recipe


def _card(**kw):
    rec = {"url": "https://example.test/", "title": "Example", "snippet": "s",
           "country": None, "register_id": None, "register_scheme": None,
           "provenance": "retailer_card",
           "dense_text": "Example dense prose.", "sparse_text": "example sparse"}
    rec.update(kw)
    return rec


@pytest.mark.parametrize("family,fields,expected", [
    ("ch_zefix_uid", {"register_id": "CHE-123.456.789", "register_scheme": "CHE-UID"}, "CH"),
    ("ch_bare_uid", {"register_id": "CHE-123.456.789", "register_scheme": "UID"}, "CH"),
    ("fr_sirene", {"register_id": "552032534", "register_scheme": "SIREN"}, "FR"),
    ("de_register", {"register_id": "HRB 1", "register_scheme": "DE-Registernummer"}, "DE"),
    ("ca_business_number", {"register_id": "123", "register_scheme": "CA-BusinessNumber"}, "CA"),
    ("fr_commune_head", {"provenance": "wikidata_fr"}, "FR"),
    ("de_head_postbank", {"provenance": "wikidata_de"}, "DE"),
])
def test_country_stamped_from_register_source(family, fields, expected):
    rec = recipe.finish(_card(**fields), None)
    assert rec["country"] == expected


def test_digest_and_channels_identical_with_and_without_stamp():
    base = _card(register_id="552032534", register_scheme="SIREN")
    stamped = recipe.finish(dict(base), None)
    explicit = recipe.finish(dict(base, country="FR"), None)
    unstampable = recipe.finish(_card(), None)
    assert stamped["country"] == "FR" and unstampable["country"] is None
    assert stamped["digest"] == explicit["digest"] == unstampable["digest"]
    for k in ("dense_text", "sparse_text", "text", "snippet", "title"):
        assert stamped.get(k) == unstampable.get(k)


def test_explicit_country_wins_and_existing_is_kept():
    rec = recipe.finish(_card(country="CH", register_scheme="SIREN",
                              register_id="1"), None)
    assert rec["country"] == "CH"
    existing = {"point_id": "p", "url": "https://example.test/", "country": "FR"}
    assert recipe.finish(_card(), existing)["country"] == "FR"


def test_register_without_id_or_unknown_scheme_stays_null():
    assert recipe.register_country(_card(register_scheme="SIREN")) is None
    assert recipe.register_country(_card(register_id="1", register_scheme="LEI")) is None


def test_p17_fallback_only_when_single_and_mapped(monkeypatch):
    def ent(*qids):
        return {"claims": {"P17": [
            {"mainsnak": {"datavalue": {"value": {"id": q}}}} for q in qids]}}
    assert recipe.p17_country(ent("Q142")) == "FR"
    assert recipe.p17_country(ent("Q142", "Q183")) is None
    assert recipe.p17_country(ent("Q99999")) is None
    assert recipe.p17_country(None) is None


def test_p17_does_not_beat_register_scheme_in_two_channel(monkeypatch):
    monkeypatch.setattr(recipe, "multilang_labels", lambda qids, aliases=False: {})
    # Review 2016: a CHE-UID card whose entity says DE (Q183) stays CH.
    seed = {"domain": "example-ag.ch", "url": "https://example-ag.ch/", "country": None,
            "lang": "de", "kind": "org", "cats": [], "name": "Example AG",
            "register_id": "CHE-123.456.789", "register_scheme": "CHE-UID"}
    ent = {"claims": {"P17": [{"mainsnak": {"datavalue": {"value": {"id": "Q183"}}}}]}}
    try:
        a, _ = recipe.build_two_channel(seed, None, ent=ent, qid="Q1",
                                        zweck=None, phase="t")
    except RuntimeError as exc:
        if "tokenizer" in str(exc):
            pytest.skip("the pinned Arctic tokenizer is not cached on this machine")
        raise
    assert a["country"] == "CH"


def test_p17_fills_registerless_wikidata_card_after_provenance():
    ent = {"claims": {"P17": [{"mainsnak": {"datavalue": {"value": {"id": "Q142"}}}}]}}
    rec = recipe.finish(_card(), None, ent=ent)
    assert rec["country"] == "FR"
    # ruled order: P17 comes before the existing-stamp fallback
    assert recipe.finish(_card(), {"point_id": "p1", "url": "https://example.test/",
                                "country": "BE"}, ent=ent)["country"] == "FR"


def test_two_channel_builder_stamps_without_changing_digest():
    seed = {"domain": "example-ag.ch", "url": "https://example-ag.ch/", "country": None,
            "lang": "de", "kind": "org", "cats": [], "name": "Example AG",
            "register_id": "CHE-123.456.789", "register_scheme": "CHE-UID"}
    try:
        a, _ = recipe.build_two_channel(seed, None, ent=None, qid=None,
                                        zweck=None, phase="t")
    except RuntimeError as exc:
        if "tokenizer" in str(exc):
            pytest.skip("the pinned Arctic tokenizer is not cached on this machine")
        raise
    assert a["country"] == "CH"
    assert a["digest"] == recipe.finish(
        {**a, "country": None, "register_id": None}, None)["digest"]
