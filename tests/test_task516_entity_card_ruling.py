#!/usr/bin/env python3
"""ADR-060's single-surface card variant (task 516's ruling).

Task 515 sec.9.1 left this open: `source_type == "entity_card"` activates a
contract in `project_record` that demands paired dense/sparse corpus channels
plus a `bundle_digest`, and task467's 336,254 sealed re-embed records carry one
text surface and no digest. Refusing the label shipped those rows with
`source_type` blank -- a ranking signal the collection they replace carries on
all but 10 of its points.

The ruling, with the two numbers that decide it:

  * the record is not INHERITING the label. It carries `card_recipe`,
    `card_recipe_source`, `card_recipe_tier` and `card_recipe_join_basis` --
    the ADR-060 recipe that produced the card -- so the producer, not a URL
    join, is what says it is a card.
  * production itself does not satisfy the digest half: 361,468 of its 370,432
    serving cards carry no `bundle_digest`. Stamping without one is parity,
    not a new liberty. The digest half of ADR-060 stays open and recorded.

The refusal it must NOT weaken is a re-chunked passage that merely shares a URL
with a card; `test_a_passage_sharing_a_url_with_a_card_is_still_refused` pins
that, and it is the same assertion task 515 shipped.

Run: python3 -m pytest tests/test_task516_entity_card_ruling.py
"""
from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault("AW_FOLD_ACCENTS", "1")

import cytir.core.common as AC  # noqa: E402
from cytir.index import serving_metadata as SM  # noqa: E402
from test_task515_serving_metadata import (  # noqa: E402
    FIXTURES, _mk, _project, _registry)

CARD = FIXTURES["task467_entity_card"]
PASSAGE = FIXTURES["task472_materializer"]


def _card_registry(tmp_path, url):
    return _registry(_mk(tmp_path), {AC.canonical_url(url): {
        "title": "Kairos Holding SA", "source_type": "entity_card",
        "trust_tier": 0.6, "lang": "fr", "country": "CH"}})


def test_a_recipe_owned_card_is_recognised_from_its_own_record():
    assert SM.is_recipe_owned_card(CARD) is True
    assert SM.is_recipe_owned_card(PASSAGE) is False
    assert SM.is_recipe_owned_card({}) is False
    assert SM.is_recipe_owned_card({"card_recipe": ""}) is False


def test_the_card_reaches_the_collection_stamped_entity_card(tmp_path):
    """This is the row task 514/515 shipped with no `source_type` at all."""
    reg = _card_registry(tmp_path, CARD["url"])
    point, _meta, stats = _project(CARD, registry=reg)
    assert point is not None
    payload = point["payload"]
    assert payload["source_type"] == "entity_card"
    assert payload["title"] == "Kairos Holding SA"
    assert payload["trust_tier"] == 0.6
    assert stats["meta_entity_card_stamped_from_recipe"] == 1
    assert stats["meta_refused_source_type_entity_card"] == 0


def test_the_open_half_of_adr060_is_counted_not_hidden(tmp_path):
    reg = _card_registry(tmp_path, CARD["url"])
    _point, _meta, stats = _project(CARD, registry=reg)
    assert stats["card_single_surface_no_bundle_digest"] == 1


def test_the_stamped_card_still_carries_a_bm25_posting(tmp_path):
    """A card with no posting is invisible to the lexical leg -- task 515's
    other defect, 339,307 rows of it. The recipe-owned surface regenerates
    one."""
    reg = _card_registry(tmp_path, CARD["url"])
    point, _meta, _stats = _project(CARD, registry=reg)
    sparse = (point.get("vector") or {}).get("bm25")
    assert sparse is not None and len(sparse.indices) > 0


def test_no_bundle_digest_is_invented(tmp_path):
    reg = _card_registry(tmp_path, CARD["url"])
    point, _meta, _stats = _project(CARD, registry=reg)
    assert "bundle_digest" not in point["payload"]


def test_a_passage_sharing_a_url_with_a_card_is_still_refused(tmp_path):
    """The refusal the ruling must not weaken."""
    reg = _registry(_mk(tmp_path), {AC.canonical_url(PASSAGE["url"]): {
        "title": "Some Entity", "source_type": "entity_card",
        "trust_tier": 0.9, "lang": "en"}})
    point, _meta, stats = _project(PASSAGE, registry=reg)
    assert "source_type" not in point["payload"]
    assert stats["meta_refused_source_type_entity_card"] == 1
    assert stats["meta_entity_card_stamped_from_recipe"] == 0
