#!/usr/bin/env python3
"""One definition of "this incoming card replaces that stored card": the entity
identity (register id > LEI > QID), not the URL alone.

Fixture `fixtures/card-identity-shared-url-rows.json`: REAL stored card rows of
the 56 URLs that hold two or more distinct entities, reduced to the fields the
functions read (no vectors, no titles, no people). `alpiq.com` (a QID-only
archive and the register card), `batitec.ch` (two register cards),
`oberriet.ch`, `autoneum.com`, `careweather.com` (two anonymous directory
cards) and one `official` row that carries card markers.
"""
from __future__ import annotations

import copy
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

from cytir.core import common as AC  # noqa: E402
from cytir.core.qdrant import qdrant_client  # noqa: E402
from cytir.index import promote_queue as Q  # noqa: E402
from cytir.index import supersede as US  # noqa: E402

FX = json.loads((HERE / "fixtures" / "card-identity-shared-url-rows.json").read_text())
PAGES = json.loads((HERE / "fixtures" / "supersede-entity-marker-rows.json").read_text())
COLL = "aw_e8"

ALPIQ_ARCHIVE, ALPIQ_REGISTER = (r for r in FX["alpiq.com"])
if ALPIQ_ARCHIVE["id"].startswith("f161"):
    ALPIQ_ARCHIVE, ALPIQ_REGISTER = ALPIQ_REGISTER, ALPIQ_ARCHIVE
BATITEC_A = next(r for r in FX["batitec.ch"] if r["id"].startswith("25ba5779"))
BATITEC_B = next(r for r in FX["batitec.ch"] if r["id"].startswith("2688c7ec"))
OBERRIET_QID, OBERRIET_REG = sorted(FX["oberriet.ch"], key=lambda r: "register_id" in r["payload"])
CARE_A, CARE_B = FX["careweather.com"]
AUTONEUM_QID, AUTONEUM_REG = sorted(FX["autoneum.com"], key=lambda r: "register_id" in r["payload"])
COLLAPSED_OFFICIAL = FX["billens-hennens.ch"][0]
PAGE_ROW = next(r for r in PAGES if "qid" not in r["payload"])


def pl(row, **over):
    out = copy.deepcopy(row["payload"])
    out.update(over)
    return out


# ------------------------------------------------------------------ identity
def test_fixture_shape():
    assert ALPIQ_ARCHIVE["payload"]["qid"] == "Q27483377"
    assert ALPIQ_REGISTER["payload"]["register_id"] == "CHE-105.974.401"
    assert "register_id" not in OBERRIET_QID["payload"]
    assert "register_id" not in CARE_A["payload"] and "qid" not in CARE_A["payload"]


def test_three_swiss_spellings_fold_to_one_token():
    base = {"register_id": "CHE-105.974.401"}
    spellings = [
        {**base, "register_scheme": "CHE-UID"},
        {"register_id": "CHE105974401", "register_scheme": "CHE"},
        {**base, "register_scheme": "CH-UID"},
        base,                                   # no scheme: a CHE-prefixed id is Swiss
    ]
    assert len({US.card_identity(s) for s in spellings}) == 1
    assert US.card_identity(spellings[0]) == frozenset({"reg:CHE:CHE105974401"})


def test_other_schemes_stay_apart_and_anonymous_is_empty():
    siren = US.card_identity({"register_id": "200036614", "register_scheme": "SIREN"})
    de = US.card_identity({"register_id": "200036614", "register_scheme": "DE-Registernummer"})
    assert siren and de and not (siren & de)
    assert US.card_identity(pl(CARE_A)) == frozenset()
    assert US.card_identity(None) == frozenset()


def test_identity_never_reads_the_payload_point_id():
    row = pl(ALPIQ_REGISTER, point_id="deadbeef-0000-0000-0000-000000000000")
    assert US.card_identity(row) == US.card_identity(pl(ALPIQ_REGISTER))
    # an anonymous card stays anonymous, whatever point_id it carries
    assert US.card_identity(pl(CARE_A, point_id="x")) == frozenset()


def test_primary_identity_precedence_register_then_lei_then_qid():
    reg = pl(ALPIQ_REGISTER)               # register id AND qid
    assert US.primary_identity(reg) == "reg:CHE:CHE105974401"
    assert US.primary_identity({**reg, "register_id": None, "lei": "5493001KJTIIGC8Y1R12"}) \
        == "lei:5493001KJTIIGC8Y1R12"
    assert US.primary_identity(pl(ALPIQ_ARCHIVE)) == "qid:Q27483377"
    assert US.primary_identity(pl(CARE_A)) is None


# ------------------------------------------------------------- card_replaces
def test_two_distinct_entities_at_one_url_replace_neither_the_other():
    for a, b in ((ALPIQ_ARCHIVE, ALPIQ_REGISTER), (BATITEC_A, BATITEC_B),
                 (OBERRIET_QID, OBERRIET_REG), (AUTONEUM_QID, AUTONEUM_REG)):
        assert not US.card_replaces(pl(a), pl(b)), (a["id"], b["id"])
        assert not US.card_replaces(pl(b), pl(a)), (b["id"], a["id"])


def test_an_entity_replaces_its_own_old_row_across_url_spellings():
    new = pl(ALPIQ_REGISTER, url="https://alpiq.com/")
    assert US.card_replaces(new, pl(ALPIQ_REGISTER))
    assert US.card_replaces(pl(OBERRIET_REG), pl(OBERRIET_REG, url="http://oberriet.ch/"))


def test_same_entity_under_two_id_spellings_replaces():
    old = pl(ALPIQ_REGISTER)                                  # CHE-UID, CHE-105.974.401
    new = pl(ALPIQ_REGISTER, register_id="CHE105974401", register_scheme="CHE", qid=None)
    assert US.card_replaces(new, old) and US.card_replaces(old, new)


def test_a_card_that_gains_a_register_id_still_replaces_its_qid_only_predecessor():
    enriched = pl(OBERRIET_REG)                               # same QID, now with a register id
    assert US.card_replaces(enriched, pl(OBERRIET_QID, qid=enriched["qid"]))


def test_two_register_ids_sharing_a_qid_are_two_entities():
    """CONSTRUCTED pair (the twin has none to query: `qid` is not indexed): two
    register cards of one URL that carry the same QID. The register ids decide;
    a shared QID or LEI never makes them one entity."""
    a = pl(BATITEC_A, qid="Q1")
    b = pl(BATITEC_B, qid="Q1")
    assert US.card_identity(a) & US.card_identity(b) == {"qid:Q1"}
    assert not US.card_replaces(a, b) and not US.card_replaces(b, a)
    a2, b2 = dict(a, lei="LEI0000000000000001"), dict(b, lei="LEI0000000000000001")
    assert not US.card_replaces(a2, b2) and not US.card_replaces(b2, a2)


def test_a_lower_precedence_overlap_counts_only_when_a_side_lacks_the_higher_id():
    reg = pl(BATITEC_A, qid="Q1")
    qid_only = {k: v for k, v in reg.items() if k not in ("register_id", "register_scheme")}
    assert US.card_replaces(reg, qid_only) and US.card_replaces(qid_only, reg)
    other_qid = pl(BATITEC_B, qid="Q2")
    assert not US.card_replaces(other_qid, reg)                    # no overlap at all
    lei_a = dict(qid_only, lei="L1", qid="Q9")
    lei_b = dict(qid_only, lei="L2", qid="Q9")
    assert not US.card_replaces(lei_a, lei_b)                       # both carry a LEI: it decides


def test_today_s_rule_for_a_card_with_no_identity_is_pinned():
    """Today's rule (group `canonical_url|card`): ANY incoming card at the
    canonical url replaces ANY stored card there. Kept for a STORED card with no
    identity (6,509 directory cards): any card of the same raw type replaces it.
    An incoming card with no identity never replaces a stored card that has one."""
    a, b = pl(CARE_A), pl(CARE_B)
    assert US.card_replaces(a, b) and US.card_replaces(b, a)       # anonymous both ways
    assert US.card_replaces(pl(ALPIQ_REGISTER, url="https://careweather.com/"), a)
    assert not US.card_replaces(a, pl(ALPIQ_REGISTER, url="https://careweather.com/"))


def test_card_and_non_card_row_at_one_url_never_replace_each_other():
    page = copy.deepcopy(PAGE_ROW["payload"])
    card = pl(OBERRIET_REG, url=page["url"])
    assert not US.card_replaces(card, page)
    assert not US.card_replaces(page, card)
    assert not US.payload_is_card(page)
    # unchanged from #598: the group key still keeps them apart
    assert US.payload_group_key(card).endswith(US.CARD_SUFFIX)
    assert not US.payload_group_key(page).endswith(US.CARD_SUFFIX)


def test_raw_source_type_must_match():
    """The collapsed `official` row carries card markers: an `entity_card` never
    replaces it, nor the reverse (same identity, different raw type)."""
    off = pl(COLLAPSED_OFFICIAL, url="https://billens-hennens.ch/")
    card = pl(COLLAPSED_OFFICIAL, url="https://billens-hennens.ch/", source_type="entity_card")
    assert US.payload_is_card(off) and US.payload_is_card(card)
    assert not US.card_replaces(card, off)
    assert not US.card_replaces(off, card)
    assert US.card_replaces(card, copy.deepcopy(card))


def test_different_canonical_url_never_replaces():
    assert not US.card_replaces(pl(ALPIQ_REGISTER), pl(ALPIQ_REGISTER, url="https://other.ch/"))
    assert not US.card_replaces(pl(ALPIQ_REGISTER, url=""), pl(ALPIQ_REGISTER, url=""))


# ----------------------------------------------------------------- Superseder
def _store(rows):
    # `url_variants` (pre-existing) enumerates slash-form spellings of a root
    # url; store the rows under that form so these tests exercise the SELECTION
    rows = [{"id": r["id"], "payload": {**r["payload"], "url": AC.canonical_url(r["payload"]["url"])}}
            for r in rows]
    client = qdrant_client(":memory:")
    client.create_collection(
        COLL, vectors_config={"dense": qm.VectorParams(size=3, distance=qm.Distance.COSINE)})
    client.upsert(COLL, points=[
        qm.PointStruct(id=r["id"], vector={"dense": [0.1, 0.2, 0.3]}, payload=r["payload"])
        for r in rows])
    return client


def _stale(stored_rows, incoming):
    sup = US.Superseder(_store(stored_rows), COLL, dry_run=True)
    for pid, payload in incoming:
        sup.observe_payload(pid, payload)
    stale_ids, _, _ = sup._scan_stale()
    return {str(i) for i in stale_ids}, sup


NEW_ID = "7a1240bb-0000-5000-8000-000000000001"


def test_a_pack_re_emitting_one_entity_does_not_tombstone_the_other():
    """alpiq.com: re-emitting the register card (CHE-105.974.401) must stale ONLY
    its own old row, never the QID-only archive card of another entity."""
    stale, sup = _stale([ALPIQ_ARCHIVE, ALPIQ_REGISTER],
                        [(NEW_ID, pl(ALPIQ_REGISTER, url="https://alpiq.com/"))])
    assert stale == {ALPIQ_REGISTER["id"]}
    assert sup.stats["unmatched_hits"] >= 1


def test_batitec_two_register_cards_are_independent():
    stale, _ = _stale([BATITEC_A, BATITEC_B],
                      [(NEW_ID, pl(BATITEC_B, url="https://batitec.ch/"))])
    assert stale == {BATITEC_B["id"]}


def test_anonymous_pair_still_collapses_to_the_new_card():
    stale, _ = _stale([CARE_A, CARE_B], [(NEW_ID, pl(CARE_A, url="https://careweather.com/x"))])
    assert stale == set()                                       # other url: untouched
    stale, _ = _stale([CARE_A, CARE_B], [(NEW_ID, pl(CARE_A, url="https://careweather.com/"))])
    assert stale == {CARE_A["id"], CARE_B["id"]}                # today's rule kept


def test_an_anonymous_incoming_card_never_stales_a_card_with_identity():
    stale, _ = _stale([ALPIQ_ARCHIVE, ALPIQ_REGISTER],
                      [(NEW_ID, pl(CARE_A, url="https://alpiq.com/"))])
    assert stale == set()


def test_a_collapsed_official_row_is_not_stale_for_an_entity_card_pack_and_vice_versa():
    off = {"id": COLLAPSED_OFFICIAL["id"], "payload": pl(COLLAPSED_OFFICIAL)}
    card_in = pl(COLLAPSED_OFFICIAL, url="https://billens-hennens.ch/", source_type="entity_card")
    stale, _ = _stale([off], [(NEW_ID, card_in)])
    assert stale == set()
    stored_card = {"id": "11111111-0000-5000-8000-000000000001", "payload": card_in}
    stale, _ = _stale([stored_card], [(NEW_ID, pl(COLLAPSED_OFFICIAL, url="https://billens-hennens.ch/"))])
    assert stale == set()


def test_a_page_row_is_still_grouped_apart_from_cards():
    page_url = PAGE_ROW["payload"]["url"]
    stored = [{"id": PAGE_ROW["id"], "payload": PAGE_ROW["payload"]},
              {"id": "22222222-0000-5000-8000-000000000002",
               "payload": pl(OBERRIET_REG, url=page_url)}]
    stale, _ = _stale(stored, [(NEW_ID, pl(OBERRIET_REG, url=page_url))])
    assert stale == {"22222222-0000-5000-8000-000000000002"}


# ---------------------------------------------------------------- the drain
def _tomb(row, url=None):
    return {"id": row["id"], "url": url or row["payload"]["url"], "reason": "superseded"}


def _validate(monkeypatch, stored_rows, tombstones, pack_rows):
    stored = {r["id"]: r["payload"] for r in stored_rows}
    monkeypatch.setattr(Q.PSP, "already_present", lambda url, coll, ids, **k: [i for i in ids if i in stored])
    monkeypatch.setattr(Q.PD, "fetch_payload_fields",
                        lambda url, coll, ids, fields, **k: {i: stored[i] for i in ids if i in stored})
    pairs, cards = {}, {}
    for pid, p in pack_rows:
        k = Q.document_url_key(p["url"])
        pairs.setdefault(k, set()).add((Q._norm_source_type(p.get("source_type")), Q._is_card_row(p)))
        if Q._is_card_row(p):
            cards.setdefault(k, []).append(p)
    return Q.validate_tombstones(tombstones, pack_urls={p["url"] for _, p in pack_rows},
                                 target_url="http://x", collection=COLL,
                                 pack_pairs=pairs, pack_cards=cards)


def test_validate_tombstones_refuses_the_other_entity_s_card(monkeypatch):
    pack = [(NEW_ID, pl(ALPIQ_REGISTER, url="https://alpiq.com/"))]
    with pytest.raises(Q.QueueError) as exc:
        _validate(monkeypatch, [ALPIQ_ARCHIVE, ALPIQ_REGISTER], [_tomb(ALPIQ_ARCHIVE)], pack)
    assert ALPIQ_ARCHIVE["id"] in str(exc.value)
    assert "qid:Q27483377" in str(exc.value)


def test_validate_tombstones_accepts_the_entity_s_own_old_card(monkeypatch):
    pack = [(NEW_ID, pl(ALPIQ_REGISTER, url="https://alpiq.com/"))]
    rep = _validate(monkeypatch, [ALPIQ_ARCHIVE, ALPIQ_REGISTER], [_tomb(ALPIQ_REGISTER)], pack)
    assert rep["count"] == 1 and rep["by_reason"] == {"superseded": 1}


def test_validate_tombstones_non_card_acceptance_is_unchanged(monkeypatch):
    page = PAGE_ROW
    pack = [(NEW_ID, pl(page, source_type=page["payload"]["source_type"]))]
    rep = _validate(monkeypatch, [page], [_tomb(page)], pack)
    assert rep["count"] == 1


def _guard(monkeypatch, stored_rows, pack_rows):
    stored = {r["id"]: r["payload"] for r in stored_rows}
    monkeypatch.setattr(Q.PSP, "retrieve_payloads",
                        lambda url, coll, ids, fields, **k: {i: stored[i] for i in ids if i in stored})
    return Q.check_card_overwrites([{"id": i, "payload": p} for i, p in pack_rows], "http://x", COLL)


def test_guard_refuses_the_batitec_re_emit_over_another_entity(monkeypatch):
    # a producer mints the URL-only Tier C id of A for entity B
    with pytest.raises(Q.QueueError) as exc:
        _guard(monkeypatch, [BATITEC_A, BATITEC_B],
               [(BATITEC_A["id"], pl(BATITEC_B, url="https://batitec.ch/"))])
    msg = str(exc.value)
    assert BATITEC_A["id"] in msg
    assert "CHE103883756" in msg and "CHE130449374" in msg      # both identities named


def test_guard_refuses_a_shop_row_written_over_a_card_with_identity(monkeypatch):
    """The L1-backfill failure: a page row at the card's id."""
    shop = {"url": "https://alpiq.com/", "source_type": "official", "title": "Shop", "snippet": "x"}
    with pytest.raises(Q.QueueError, match="non-card"):
        _guard(monkeypatch, [ALPIQ_REGISTER], [(ALPIQ_REGISTER["id"], shop)])


def test_guard_allows_the_same_entity_and_anonymous_stored_cards_and_new_ids(monkeypatch):
    out = _guard(monkeypatch, [ALPIQ_REGISTER, CARE_A],
                 [(ALPIQ_REGISTER["id"], pl(ALPIQ_REGISTER, snippet="enriched")),   # same entity, in place
                  (CARE_A["id"], pl(CARE_A, snippet="refreshed")),                  # anonymous stored, a card
                  ("33333333-0000-5000-8000-000000000003", pl(BATITEC_A))])          # a new id
    assert out["stored_cards_with_identity"] == 1 and out["stored_cards_anonymous"] == 1
    assert out["ids_read"] == 2


def test_guard_refuses_a_page_row_over_an_anonymous_stored_card(monkeypatch):
    """S4: a stored card with no identity may be overwritten by id only by a card."""
    page = {"url": "https://www.careweather.com", "source_type": "official", "title": "Shop"}
    assert not US.card_identity(CARE_A["payload"])
    with pytest.raises(Q.QueueError, match="non-card"):
        _guard(monkeypatch, [CARE_A], [(CARE_A["id"], page)])


def test_guard_lets_any_card_of_the_type_replace_an_anonymous_stored_card(monkeypatch):
    out = _guard(monkeypatch, [CARE_A],
                 [(CARE_A["id"], pl(ALPIQ_REGISTER, url="https://careweather.com/"))])
    assert out["stored_cards_anonymous"] == 1


@pytest.mark.parametrize("spelling", [lambda i: i.upper(), lambda i: i.replace("-", "")])
def test_guard_matches_ids_across_uuid_spellings(monkeypatch, spelling):
    """S1: Qdrant answers in canonical form whatever the pack spells; a pack row
    with the uppercase / no-hyphen id is still checked (and refused)."""
    stored = {ALPIQ_REGISTER["id"]: ALPIQ_REGISTER["payload"]}
    monkeypatch.setattr(Q.PSP, "retrieve_payloads",
                        lambda url, coll, ids, fields, **k: dict(stored))      # canonical keys
    shop = {"url": "https://alpiq.com/", "source_type": "official", "title": "Shop"}
    with pytest.raises(Q.QueueError, match="non-card"):
        Q.check_card_overwrites([{"id": spelling(ALPIQ_REGISTER["id"]), "payload": shop}],
                                "http://x", COLL)
    ok = Q.check_card_overwrites(
        [{"id": spelling(ALPIQ_REGISTER["id"]), "payload": pl(ALPIQ_REGISTER)}], "http://x", COLL)
    assert ok["stored_cards_with_identity"] == 1


def test_guard_refuses_an_unmatched_returned_id_instead_of_raising_keyerror(monkeypatch):
    monkeypatch.setattr(Q.PSP, "retrieve_payloads",
                        lambda url, coll, ids, fields, **k: {ALPIQ_REGISTER["id"]: ALPIQ_REGISTER["payload"]})
    with pytest.raises(Q.QueueError, match="matches no id of this pack"):
        Q.check_card_overwrites([{"id": "33333333-0000-5000-8000-000000000003",
                                  "payload": pl(ALPIQ_REGISTER)}], "http://x", COLL)


def test_guard_ignores_stored_non_card_rows(monkeypatch):
    out = _guard(monkeypatch, [PAGE_ROW], [(PAGE_ROW["id"], {"url": "https://x.ch/", "source_type": "news"})])
    assert out["stored_cards"] == 0


def test_dry_check_runs_the_guard_before_any_apply(tmp_path, monkeypatch):
    shop = {"url": "https://alpiq.com/", "source_type": "official", "title": "Shop", "snippet": "x"}
    pack_points = tmp_path / "points.jsonl"
    pack_points.write_text(json.dumps({"id": ALPIQ_REGISTER["id"], "payload": shop, "vector": {}}) + "\n")
    manifest = {"kind": "content_pack", "pack_id": "t", "family": "l1", "collection": COLL,
                "rows": 1, "pack_points": str(pack_points)}
    monkeypatch.setattr(Q, "fetch_collection_vector_config", lambda *a, **k: {})
    monkeypatch.setattr(Q.PSP, "retrieve_payloads",
                        lambda url, coll, ids, fields, **k: {ALPIQ_REGISTER["id"]: ALPIQ_REGISTER["payload"]})
    with pytest.raises(Q.QueueError, match="card overwrite guard refused"):
        Q.content_pack_dry_check(manifest, "http://x")
