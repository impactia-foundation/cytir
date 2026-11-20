"""`ch_zefix_cards` on the family Protocol (task 2485): registered, PARTIAL, offline.

No network. One hand-written register seed; the card is the output of the moved composer
(`cytir.corpus.register_cards.compose_register_card`), called through the adapter's stages.
"""
from __future__ import annotations

import json

import pytest

from cytir.corpus import register_cards as RC
from cytir.sources import registry
from cytir.sources.families.ch_zefix_cards import ChZefixCardsFamily
from cytir.sources.family import RunContext, SourceFamily

SEED = {"register_id": "CHE-123.456.789", "register_scheme": "CH-UID", "legal_name": "Muster Beispiel AG",
        "purpose": "Entwicklung von Software.", "officers": ["Anna Muster"], "legal_form": "AG",
        "legal_form_code": "0106", "canton": "ZH", "locality": "Zuerich", "country": "CH", "lang": "de",
        "domain": "muster-beispiel.ch", "url": "https://muster-beispiel.ch/",
        "license": "opendata.swiss open use (terms_by)", "provenance": "zefix-lindas",
        "register_url": "https://www.zefix.ch/", "last_seen": "2026-11-05"}
ROW = {"seed": SEED, "evidence": {"uid_on_page": True}, "confidence": 0.97}


def _ctx(tmp_path, seeds):
    p = tmp_path / "seeds.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in seeds), encoding="utf-8")
    row = registry.admit("ch_zefix_cards")[1]
    return RunContext(out_dir=tmp_path / "out", today="2026-11-05", skip_consent=True,
                      options={"seeds": str(p)}, contract=row, log=lambda m: None)


def test_registered_conformant_and_partial():
    assert "ch_zefix_cards" in registry.registered()
    adapter, row, spec = registry.admit("ch_zefix_cards")
    assert isinstance(adapter, SourceFamily)
    assert row["route"] == adapter.route == "api"
    assert spec.declared == "chunk-policy-atomic-record.v1"
    assert adapter.PARTIAL == ("dialect:zefix-api",)


def test_stages_equal_the_moved_composer(tmp_path):
    fam, ctx = ChZefixCardsFamily(), _ctx(tmp_path, [ROW])
    (unit,) = fam.discover(ctx)
    assert unit.key == "CHE-123.456.789" and unit.consent_hosts == ()
    (cap,) = fam.acquire(unit, ctx)
    (doc,) = fam.extract(cap, ctx)
    (rec,) = fam.chunk(doc, ctx)
    assert rec == RC.compose_register_card(SEED, evidence=ROW["evidence"], confidence=0.97)
    assert rec["dense_text"].startswith("Muster Beispiel AG") and "Anna Muster" in rec["dense_text"]
    assert rec["resolve_confidence"] == 0.97


def test_missing_or_empty_seeds_refuse(tmp_path):
    fam = ChZefixCardsFamily()
    with pytest.raises(ValueError):
        fam.discover(RunContext(out_dir=tmp_path, today="2026-11-05", options={}))
    with pytest.raises(RuntimeError):
        fam.discover(_ctx(tmp_path, []))


def test_payload_contract_refuses_an_empty_field(tmp_path):
    fam = ChZefixCardsFamily()
    bad = {"seed": dict(SEED, license=""), "evidence": {"uid_on_page": True}, "confidence": 0.9}
    ctx = _ctx(tmp_path, [bad])
    (unit,) = fam.discover(ctx)
    (cap,) = fam.acquire(unit, ctx)
    with pytest.raises(RC.PayloadContractRefusal):
        fam.extract(cap, ctx)
