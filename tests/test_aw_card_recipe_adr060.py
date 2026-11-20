"""Regression guards for the durable task-365 entity-card recipe fix."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from cytir.core.common import record_text_channels
from cytir.corpus import card_recipe as recipe
from tests.helpers import wikidata_cards as country_cards


def claim(value):
    return {"mainsnak": {"datavalue": {"value": value}}}


def test_shared_bridge_includes_instance_of():
    """Every caller of the shared ADR-060 bridge must see P31."""
    assert recipe.BRIDGE_PROPS == ("P31", "P452", "P1056", "P106")


def test_shared_finish_digest_tracks_channel_content():
    base = {
        "point_id": None, "url": "https://example.test/",
        "dense_text": "A regional bank.", "sparse_text": "bank regional",
    }
    first = recipe.finish(dict(base), None)
    changed = recipe.finish(
        {**base, "sparse_text": "bank regional mortgage"}, None)
    assert first["digest"] != changed["digest"]
    assert first["point_id"] == changed["point_id"]


def test_truncated_predecessor_inherits_only_leading_category():
    assert recipe.predecessor_category_terms(
        "Example — organisation, sells and refurbishes electronic devices"
    ) == ["organisation"]
    assert recipe.predecessor_category_terms(
        "Example — organisation, regional bank. Official website."
    ) == ["organisation", "regional bank"]


def test_generic_embedder_honors_dual_channels_and_legacy_fallback():
    assert record_text_channels({
        "text": "legacy", "dense_text": "dense prose",
        "sparse_text": "sparse hooks",
    }) == ("dense prose", "sparse hooks")
    assert record_text_channels({"text": "legacy"}) == ("legacy", "legacy")


def test_category_evidence_rejects_unknown_provenance():
    with __import__("pytest").raises(ValueError):
        recipe.retained_category_evidence({
            "category_evidence": {
                "schema": recipe.CATEGORY_EVIDENCE_SCHEMA,
                "sources": ["ungraded_sparse_bag"],
                "category_terms": ["personal loan"],
            },
        })
    with __import__("pytest").raises(ValueError):
        recipe.category_evidence_payload(
            sources=["ungraded_sparse_bag"],
            retained_terms=["personal loan"])


def test_shared_commercial_builder_puts_p31_in_dense(monkeypatch):
    # task 530: this recipe's website channel refuses to approximate the
    # token budget when the pinned Arctic tokenizer is unavailable (fail
    # loud, entity_card_website.py:_website_tokenizer) — deterministic in
    # any environment without `transformers` installed (or without network
    # access to fetch the pinned revision).
    pytest.importorskip("transformers")
    ent = {
        "claims": {"P31": [claim({"id": "Q_REGIONAL_BANK"})]},
        "labels": {"en": {"value": "Example Bank"}},
        "aliases": {},
        "descriptions": {"en": {"value": "A Swiss bank."}},
        "sitelinks": {},
    }

    def labels(qids, aliases=False):
        return {
            qid: {"en": "regional bank", "de": "Regionalbank",
                  "fr": "banque régionale", "it": "banca regionale",
                  **({"aliases": []} if aliases else {})}
            for qid in qids
        }

    monkeypatch.setattr(recipe, "multilang_labels", labels)
    rec, _audit = recipe.build_two_channel(
        {
            "domain": "bank.example", "url": "https://bank.example/",
            "country": "CH", "lang": "en", "kind": "org", "cats": [],
        },
        None, ent=ent, qid="Q_BANK", zweck=None, phase="test")
    assert "Industry: regional bank." in rec["dense_text"]


def test_country_head_generator_uses_two_channel_p31_recipe(tmp_path):
    enriched = {
        "country": "CH",
        "langs": ["de", "fr", "it", "en"],
        "class_labels": {
            "Q_REGIONAL_BANK": {
                "de": "Regionalbank", "fr": "banque régionale",
                "it": "banca regionale", "en": "regional bank",
            },
            "Q_LOAN": {
                "de": "Kredit", "fr": "crédit",
                "it": "credito", "en": "loan",
            },
        },
        "entities": {
            "Q_BANK": {
                "kind": "org",
                "sites": ["https://bank.example/"],
                "label_en": "Example Bank",
                "desc_en": "A Swiss bank.",
                "type": ["Q_REGIONAL_BANK"],
                "industry": [],
                "product": ["Q_LOAN"],
            },
        },
    }
    source = tmp_path / "enriched.json"
    target = tmp_path / "cards.jsonl"
    source.write_text(json.dumps(enriched))
    country_cards.cmd_cards(SimpleNamespace(
        enriched=str(source), out=str(target), existing_qids=None))

    rec = json.loads(target.read_text())
    assert rec["fx351_recipe"] == recipe.RECIPE
    assert "regional" in rec["sparse_text"]
    assert "Industry: Regionalbank." in rec["dense_text"]
    assert "loan" in rec["sparse_text"].lower()
    assert "Products/services: Kredit." in rec["dense_text"]
    assert len(rec["digest"]) == 64


# --- ADR-060 amendment (Tim 2026-10-07, ruling 27 T6): the dense text is never cut at 1,600 ----
_OFFICERS = os.path.join(os.path.dirname(__file__), "fixtures",
                         "card_officers_madeup.json")


def _madeup_officers():
    with open(_OFFICERS, encoding="utf-8") as fh:
        return json.load(fh)


def test_a_card_with_many_madeup_officers_keeps_every_name_whole():
    """42 made-up officers (3,850 chars of sentence): the old `[:1600]` cut the
    sentence mid-name; now the register composition stays (purpose first) and no name is cut."""
    fx = _madeup_officers()
    dense, _sparse = recipe.add_people_facet(fx["base_dense"], "", fx["people"])
    assert len(dense) > 1600
    assert dense.startswith(fx["base_dense"])
    for display in fx["people"].values():
        assert display in dense
    assert dense.endswith(list(fx["people"].values())[-1] + ".")


def test_the_card_recipe_has_no_1600_cut_left():
    src = open(recipe.__file__, encoding="utf-8").read()
    assert "[:1600]" not in src


def test_the_two_channel_builder_keeps_every_register_officer_whole():
    fx = _madeup_officers()
    seed = {"domain": "example-ag.ch", "url": "https://example-ag.ch/", "country": "CH",
            "lang": "de", "kind": "org", "cats": [], "name": "Example AG",
            "register_id": "CHE-123.456.789", "register_people": fx["people"]}
    try:
        rec, _audit = recipe.build_two_channel(seed, None, ent=None, qid=None,
                                               zweck=fx["base_dense"], phase="t")
    except RuntimeError as exc:
        if "tokenizer" in str(exc):
            pytest.skip("the pinned Arctic tokenizer is not cached on this machine")
        raise
    assert len(rec["dense_text"]) > 1600
    for display in fx["people"].values():
        assert display in rec["dense_text"]
