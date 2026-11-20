#!/usr/bin/env python3
"""Composer and drain agree on what a card is: every supersede call site passes
the row's own entity markers (`entity_kind` / `qid`).

Fixture `fixtures/supersede-entity-marker-rows.json`: the 11 REAL rows the twin
holds under `srf.ch/sendungen/archiv/`, `srf.ch/radio-srf-virus/`,
`rts.ch/archives/` and `rsi.ch`; 4 are wikidata entity rows typed `reference`
that carry `entity_kind`/`qid`, 7 are page rows.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

pytest.importorskip("qdrant_client")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qdrant_client.http import models as qm  # noqa: E402

from cytir.core.qdrant import qdrant_client  # noqa: E402
from cytir.index import promote_queue as Q  # noqa: E402
from cytir.index import supersede as US  # noqa: E402
from cytir.news import compose as NC  # noqa: E402
from cytir.sources.families import news as NF  # noqa: E402

ROWS = json.loads((HERE / "fixtures" / "supersede-entity-marker-rows.json").read_text())
ENTITY_IDS = {r["id"] for r in ROWS if r["payload"].get("qid")}
COLL = "aw_e8"


def _store():
    client = qdrant_client(":memory:")
    client.create_collection(
        COLL, vectors_config={"dense": qm.VectorParams(size=3, distance=qm.Distance.COSINE)})
    client.upsert(COLL, points=[
        qm.PointStruct(id=r["id"], vector={"dense": [0.1, 0.2, 0.3]}, payload=r["payload"])
        for r in ROWS])
    return client


def _by_key(entity: bool):
    out: dict[str, list] = {}
    for r in ROWS:
        if (r["id"] in ENTITY_IDS) == entity:
            out.setdefault(Q.document_url_key(r["payload"]["url"]), []).append(r)
    return out


def _pack_row(pid, payload):
    return {"id": pid, "payload": payload}


def test_fixture_shape():
    assert len(ROWS) == 11 and len(ENTITY_IDS) == 4


def test_payload_group_key_is_group_key_with_the_row_own_markers():
    for r in ROWS:
        pl = r["payload"]
        assert US.payload_group_key(pl) == US.group_key(
            pl["url"], pl["source_type"], entity_kind=pl.get("entity_kind"),
            qid=pl.get("qid"))
        assert US.payload_group_key(pl).endswith(US.CARD_SUFFIX) == (r["id"] in ENTITY_IDS)
    assert US.payload_group_key(None) == US.group_key("")


def test_observe_payload_groups_a_marker_row_apart_from_its_page_row():
    sup = US.Superseder(_store(), COLL, dry_run=True)
    ent = next(r for r in ROWS if r["id"] in ENTITY_IDS)
    sup.observe_payload("new-entity", ent["payload"])
    assert list(sup.keep) == [US.payload_group_key(ent["payload"])]
    assert next(iter(sup.keep)).endswith(US.CARD_SUFFIX)


# ----------------------------------------------------------- composer: tombstones
@pytest.mark.parametrize("key", sorted(_by_key(False)))
def test_page_recapture_composes_no_tombstone_on_the_entity_row_and_the_drain_accepts(
        monkeypatch, key):
    page = _by_key(False)[key]
    ent = _by_key(True)[key][0]
    new = _pack_row("new-page-" + key, {**page[0]["payload"], "content_digest": "d2"})
    tombs = NC.compute_superseded_tombstones(_store(), COLL, [new], log=lambda *a, **k: None)
    ids = {t["id"] for t in tombs}
    assert ent["id"] not in ids
    # the old page rows go (url spellings the variant lookup cannot enumerate,
    # e.g. a trailing slash, are outside the selector's reach: not asserted)
    assert ids and ids <= {r["id"] for r in page}
    # the drain's validation, as a function (no drain): stored rows read back
    stored = {r["id"]: {k: v for k, v in r["payload"].items()
                        if k in ("url", "source_type", "entity_kind", "qid")} for r in ROWS}
    monkeypatch.setattr(Q.PSP, "already_present",
                        lambda u, c, i, batch=500: [x for x in i if x in stored])
    monkeypatch.setattr(Q.PD, "fetch_payload_fields",
                        lambda u, c, ids_, fields, batch=1000: {
                            i: dict(stored[i]) for i in ids_ if i in stored})
    pairs = {Q.document_url_key(new["payload"]["url"]): {
        (Q._norm_source_type(new["payload"]["source_type"]), Q._is_card_row(new["payload"]))}}
    report = Q.validate_tombstones(
        tombs, pack_urls={new["payload"]["url"]}, target_url="http://x",
        collection=COLL, pack_pairs=pairs)
    assert report["count"] == len(tombs)


def test_entity_refresh_composes_a_tombstone_only_on_the_entity_row():
    ent = next(r for r in ROWS if r["id"] == "3de105d5-6b9f-5bc1-8bb6-a3deefe556c5")
    new = _pack_row("new-entity", {**ent["payload"], "title": "changed"})
    tombs = NC.compute_superseded_tombstones(_store(), COLL, [new], log=lambda *a, **k: None)
    assert {t["id"] for t in tombs} == {ent["id"]}


def test_cross_family_filter_keeps_the_marker_in_the_folded_group():
    ent = next(r for r in ROWS if r["id"] in ENTITY_IDS)
    kept = [_pack_row("n", ent["payload"])]
    args = (["old"], [ent["payload"]["url"]], {"old": "reference"}, kept)
    quiet = lambda *a, **k: None  # noqa: E731
    # without the stored row's markers the old row would read as a page row and be dropped
    assert NC.filter_cross_family_stale(*args, log=quiet) == []
    got = NC.filter_cross_family_stale(
        *args, stale_markers={"old": {"entity_kind": "org", "qid": "Q1"}}, log=quiet)
    assert [t["id"] for t in got] == ["old"]


# ------------------------------------------------------ news family materializer
def test_existing_group_points_never_returns_a_marker_row_for_a_page_group():
    page = [r for r in ROWS if r["id"] not in ENTITY_IDS][0]["payload"]
    group = US.payload_group_key(page)
    keep = {group: {"new-page"}}
    variants = {group: US.url_variants(page["url"])}
    found = NF.existing_group_points(_store(), COLL, keep, variants)
    assert found and not (set(found) & ENTITY_IDS)

