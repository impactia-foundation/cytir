"""The builder projection in its package home (`cytir.index.b1_projection`).

Values are copied from the builder's channel and dump-basis tests; the full
matrix stays there. These cases pin the move."""
from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

from cytir.index import b1_projection as BP

FEDLEX_URL = "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_257"


def _gate():
    return SimpleNamespace(verdict=lambda _h, _r, _p=None: (True, "ok"))


def test_channel_surfaces_pair_is_atomic():
    assert BP._channel_surfaces({"text": "body"}, "body") == ("body", "body", False)
    assert BP._channel_surfaces(
        {"text": "prose", "dense_text": "prose", "sparse_text": "hooks"},
        "prose") == ("prose", "hooks", True)
    with pytest.raises(ValueError):
        BP._channel_surfaces({"dense_text": "one"}, "fallback")


def test_project_record_fedlex_keeps_payload_and_separate_dense_surface():
    source = SimpleNamespace(
        id="old", vector={"dense": [0.1, 0.2]}, payload={
            "url": FEDLEX_URL, "anchor": "art_257", "title": "OR — Art. 257",
            "site_name": "Fedlex", "lang": "de", "domain": "fedlex.admin.ch",
            "source_type": "official", "trust_tier": 1.0})
    point, meta = BP.project_record(
        source, "fedlex", {
            "texts": ["Art. 257 Der Mietzins."],
            "dense_text": "Art. 257 Der Mietzins.",
            "sparse_text": "Achter Titel: Die Miete. Art. 257 Der Mietzins.",
        }, {}, {}, _gate(), defaultdict(int))
    assert meta == {}
    pl = point["payload"]
    assert pl["build_id"] == "aw_b1" and pl["domain"] == "fedlex.admin.ch"
    assert pl["tier"] == "A" and pl["anchor"] == "art_257"
    assert "Die Miete" not in point["dense_reembed_text"]
    assert point["vector"]["dense"] == [0.1, 0.2]


def test_project_record_without_url_is_excluded_and_counted():
    stats = defaultdict(int)
    point, meta = BP.project_record(
        SimpleNamespace(id="x", payload={}, vector={"dense": [0.0]}),
        "o", None, {}, {}, _gate(), stats)
    assert point is None and meta == {} and stats["no_url"] == 1


def test_match_chunk_text_picks_the_prefix_sibling():
    stats = defaultdict(int)
    texts = ["Alpha beta gamma delta epsilon.", "Zeta eta theta iota kappa."]
    got = BP._match_chunk_text(texts, "Zeta eta theta iota", stats)
    assert got == texts[1] and stats["ambig_matched_prefix"] == 1
    assert BP._match_chunk_text(texts, "short", stats) is None


def test_to_sparse_and_payload_contract():
    assert BP._to_sparse({"indices": [1], "values": [0.5]}) == ([1], [0.5])
    assert BP._to_sparse(None) is None
    pts = [{"payload": {"source_type": "entity_card"}} for _ in range(25)]
    ok, detail = BP.evaluate_payload_contract(pts)
    assert not ok and detail["names_on_entity_card"]["present"] == 0
