"""Task 2753: /find_authority zero-card fallback (retry without country, `filter_relaxed`) and EU->INTL.
Offline: a fake card table honours the jurisdiction filter; Qdrant and the embedder are stubbed."""
from types import SimpleNamespace as NS

import pytest

from cytir.serve import authority

CARDS = [
    ("eur-lex.example", "INTL"), ("fedlex.example", "CH"),
    ("vd.example", "CH-VD"), ("us.example", "US"),
]


@pytest.fixture
def table(monkeypatch):
    seen = []

    class _Client:
        def query_points(self, **kw):
            flt = kw.get("query_filter")
            want = flt.must[0].match.value if flt else None
            seen.append(want)
            if kw.get("using") != authority.DENSE_VEC:
                return NS(points=[])
            return NS(points=[NS(id=i, score=0.9 - i / 100,
                                 payload={"domain": d, "jurisdiction": j, "trust_tier": "A"})
                              for i, (d, j) in enumerate(CARDS) if want in (None, j)])

    monkeypatch.setattr(authority.models, "embed_query", lambda q: ([0.0], None))
    monkeypatch.setattr(authority.qdrant_io, "client", lambda: _Client())
    return seen


def test_zero_under_country_retries_without_and_marks(table):
    cards, relaxed = authority.find_authorities_relaxed("q", k=5, country="FR")
    assert relaxed == "country" and len(cards) == 4
    assert table == ["FR", None]


def test_eu_maps_to_intl_without_relaxing(table):
    cards, relaxed = authority.find_authorities_relaxed("q", k=5, country="eu")
    assert relaxed is None and [c["jurisdiction"] for c in cards] == ["INTL"]


def test_legacy_ch_vd_is_exact(table):
    cards, relaxed = authority.find_authorities_relaxed("q", k=5, country="CH-VD")
    assert relaxed is None and [c["jurisdiction"] for c in cards] == ["CH-VD"]


def test_nonzero_country_match_unchanged(table):
    cards, relaxed = authority.find_authorities_relaxed("q", k=5, country="CH")
    assert relaxed is None and [c["jurisdiction"] for c in cards] == ["CH"]
    assert table == ["CH"]


def test_no_country_never_relaxed(table):
    cards, relaxed = authority.find_authorities_relaxed("q", k=5)
    assert relaxed is None and len(cards) == 4 and table == [None]


def test_empty_both_ways_stays_empty_unmarked(monkeypatch, table):
    monkeypatch.setattr(authority, "CARDS_COLLECTION", authority.CARDS_COLLECTION)
    monkeypatch.setattr(authority, "MIN_SIM", 2.0)
    cards, relaxed = authority.find_authorities_relaxed("q", k=5, country="FR")
    assert cards == [] and relaxed is None


def test_route_adds_field_only_when_relaxed(monkeypatch):
    from cytir.serve import app
    monkeypatch.setattr(app.authority, "find_authorities_relaxed",
                        lambda q, k, country: ([{"weak_match": False}], "country"))
    r = app.find_authority(q="x", k=3, country="FR")
    assert r["filter_relaxed"] == "country" and r["count"] == 1
    monkeypatch.setattr(app.authority, "find_authorities_relaxed",
                        lambda q, k, country: ([{"weak_match": False}], None))
    assert "filter_relaxed" not in app.find_authority(q="x", k=3, country="CH")
