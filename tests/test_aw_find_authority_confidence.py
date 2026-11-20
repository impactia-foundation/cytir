"""task 2656: /find_authority score honesty (`confidence`, `weak_match`) and declined-card
disposition. Offline: the 12 probe queries are frozen as a card subset (raw dense cosines
observed on the served cards); Qdrant and the embedder are stubbed. No prod call."""
import importlib
import json
import pathlib
from types import SimpleNamespace as NS

import pytest

FIX = json.loads((pathlib.Path(__file__).parent / "data" / "find_authority_probe12.json").read_text())
CASES = FIX["queries"]


@pytest.fixture
def authority(monkeypatch):
    return importlib.import_module("cytir.serve.authority")


def _run(authority, monkeypatch, case):
    pts = [NS(id=i, score=c["similarity"],
              payload={"domain": c["domain"], "trust_tier": c["trust_tier"],
                       "disposition": c["disposition"]})
           for i, c in enumerate(case["top5"])]

    class _Client:
        def query_points(self, **kw):
            return NS(points=pts if kw.get("using") == authority.DENSE_VEC else [])

    monkeypatch.setattr(authority.models, "embed_query", lambda q: ([0.0], None))
    monkeypatch.setattr(authority.qdrant_io, "client", lambda: _Client())
    return authority.find_authorities(case["q"], k=5)


def test_fixture_has_the_twelve_probe_queries():
    assert len(CASES) == 12 and FIX["floor"] == 0.45


@pytest.mark.parametrize("case", CASES, ids=[c["q"] for c in CASES])
def test_probe_query_top1_and_weak_flags(authority, monkeypatch, case):
    cards = _run(authority, monkeypatch, case)
    top = cards[0]
    assert top["authority"] == case["expect_top1"]           # ranking unchanged
    assert top["confidence"] == top["similarity"] == case["top5"][0]["similarity"]
    assert top["weak_match"] is case["expect_top1_weak"]
    assert authority.annotate(cards)["weak_match"] is case["expect_response_weak"]


def test_response_weak_only_when_all_weak_and_has_note(authority):
    weak = {"weak_match": True}
    ok = {"weak_match": False}
    assert authority.annotate([weak, weak])["note"] == authority.WEAK_NOTE
    assert authority.annotate([weak, ok])["weak_match"] is False
    assert "note" not in authority.annotate([weak, ok])
    assert authority.annotate([])["weak_match"] is False


def test_floor_is_strict_and_by_name_is_never_weak(authority):
    pl = {"domain": "nzz.ch", "disposition": "card"}
    assert authority._card(pl, "q", 0.7, 0.45)["weak_match"] is False
    assert authority._card(pl, "q", 0.7, 0.4499)["weak_match"] is True
    assert authority._card(pl, "nzz", 0.7, 0.2, named=True)["weak_match"] is False


def test_declined_card_stays_rankable_but_is_not_fetchable(authority, monkeypatch):
    case = next(c for c in CASES if c["q"] == "Code civil suisse")
    cards = _run(authority, monkeypatch, case)
    assert cards[0]["authority"] == "rsn.ne.ch"              # still named first
    assert cards[0]["disposition"] == "consent-declined"
    assert cards[0]["fetchable"] is False
    assert all(c["fetchable"] for c in cards if c["disposition"] != "consent-declined")
    assert "declined_note" in authority.annotate(cards)
