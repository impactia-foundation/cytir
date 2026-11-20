"""Task 2218: the compose-time already-served check asks for the ids the drain WRITES.

A pack's declared `point_id` is not the id an I1 collection stores; the drain mints the I1 id of
the projected row. The composer used to ask the twin about the declared ids, so it reported
"0 served" for a pack the drain then refused as a partial overlap.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cytir.core import ids as ID  # noqa: E402
from cytir.core.canon import canonical_url  # noqa: E402
from cytir.core.ids import mint_point_id  # noqa: E402
from cytir.index import compose  # noqa: E402
from cytir.index import id_rule as IR  # noqa: E402
from cytir.index import promote_sealed_pack as PSP  # noqa: E402
from cytir.index import served_exclusion as SE  # noqa: E402


def _page(url, text):
    rec = {"url": url, "text": text, "title": "T", "lang": "en", "source_type": "reference",
           "trust_tier": 0.5, "license": "test-only-not-a-licence-finding"}
    rec["point_id"] = mint_point_id(canonical_url(url), ID.content_digest(text))
    return rec


def _card(url="https://batitec.ch/"):
    """An entity card: its declared id is url-only, its I1 id is `canon#card:<identity>`."""
    rec = {"url": url, "text": "Batitec AG, a Swiss company in the commercial register.",
           "title": "Batitec AG", "lang": "de", "source_type": "entity_card", "trust_tier": 0.85,
           "license": "test-only-not-a-licence-finding", "register_id": "CHE-103.883.756",
           "register_scheme": "CHE-UID", "entity_kind": "org", "country": "CH"}
    rec["point_id"] = mint_point_id(canonical_url(url), "")
    return rec


def _records():
    return [_card(), _page("https://example.org/b", "Beta page text.")]


def _drain_ids(recs):
    out = PSP.project_rows(iter(recs), consent=None, attribution=None, license_version=None,
                           id_rule="I1")[0]
    return list(out)


def _patch(monkeypatch, recs):
    monkeypatch.setattr(compose.PSP, "read_records", lambda d: iter(recs))


def test_declared_id_differs_from_minted_and_the_check_sees_the_minted_one(monkeypatch):
    recs = _records()
    _patch(monkeypatch, recs)
    ids = [r["point_id"] for r in recs]
    mint_map = compose._minted_id_map("x", ids, {}, IR.ID_RULE_I1)
    assert set(mint_map) == set(ids)
    assert any(mint_map[i] != i for i in ids), "fixture must have declared != minted ids"
    held = {mint_map[ids[0]]: {"url": "u"}}   # a twin holding ONE row, in minted form only

    def twin_rows(asked):
        return {i: held[i] for i in asked if i in held}

    assert twin_rows(ids) == {}   # what the composer asked before: 0 served
    got = compose._ask_by_minted(twin_rows, mint_map)(ids)
    assert list(got) == [ids[0]]


def test_prod_presence_list_is_translated_back_to_pack_ids(monkeypatch):
    recs = _records()
    _patch(monkeypatch, recs)
    ids = [r["point_id"] for r in recs]
    mint_map = compose._minted_id_map("x", ids, {}, IR.ID_RULE_I1)
    prod = {mint_map[ids[1]]}
    got = compose._ask_by_minted(lambda asked: [i for i in asked if i in prod], mint_map)(ids)
    assert got == [ids[1]]


def test_minted_map_uses_the_drains_own_function(monkeypatch):
    recs = _records()
    _patch(monkeypatch, recs)
    ids = [r["point_id"] for r in recs]
    mint_map = compose._minted_id_map("x", ids, {}, "I1")
    assert sorted(mint_map.values()) == sorted(_drain_ids(recs))
    # no id rule declared: the declared id stands (every collection before aw_e10)
    assert compose._minted_id_map("x", ids, {}, None) == {i: i for i in ids}


def test_full_overlap_in_minted_form_declares_all_served(tmp_path, monkeypatch):
    recs = _records()
    _patch(monkeypatch, recs)
    ids = [r["point_id"] for r in recs]
    mint_map = compose._minted_id_map("x", ids, {}, "I1")
    held = {mint_map[i]: {} for i in ids}
    decl = SE.declare_already_served(
        pack_dir=tmp_path / "p", pack_id="p", ids=ids,
        targets={"twin": ("t", "c"), "prod": ("p", "c")},
        twin_rows=compose._ask_by_minted(lambda a: {i: held[i] for i in a if i in held}, mint_map),
        prod_present=compose._ask_by_minted(lambda a: [i for i in a if i in held], mint_map),
        pack_payloads=lambda a: {})
    assert decl["all_served"] is True
    assert sorted(decl["excluded_ids"]) == sorted(ids)


# ---- task 2220: rows the projection refuses are excluded by declared id ------------------------

SEARCH_URL = "https://example.org/search?Search=Adult+Basic+Education"
SEARCH_REASON = "projection-rejected: search-result-url"


def _declare(tmp_path, recs, monkeypatch, held=()):
    _patch(monkeypatch, recs)
    ids = [r["point_id"] for r in recs]
    mint_map, rejected = compose._minted_and_rejected("x", ids, {}, "I1")
    served = {mint_map[i] for i in held}
    decl = SE.declare_already_served(
        pack_dir=tmp_path / "p", pack_id="p", ids=ids,
        targets={"twin": ("t", "c"), "prod": ("p", "c")},
        twin_rows=compose._ask_by_minted(lambda a: {i: {} for i in a if i in served}, mint_map),
        prod_present=compose._ask_by_minted(lambda a: [i for i in a if i in served], mint_map),
        pack_payloads=lambda a: {}, projection_rejected=rejected)
    return ids, mint_map, rejected, decl


def _file_rows(tmp_path):
    import json
    path = tmp_path / "p" / SE.SERVED_FILE
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_search_result_row_is_excluded_by_declared_id_with_its_reason(tmp_path, monkeypatch):
    recs = _records() + [_page(SEARCH_URL, "A site search result page.")]
    ids, mint_map, rejected, decl = _declare(tmp_path, recs, monkeypatch)
    bad = recs[-1]["point_id"]
    assert rejected == {bad: SEARCH_REASON} and bad not in mint_map
    assert _file_rows(tmp_path) == [{"point_id": bad, "reason": SEARCH_REASON}]
    assert decl["fields"]["projection_rejected"] == {SEARCH_REASON: 1}
    assert decl["fields"]["exclude_ids_file"].endswith(SE.SERVED_FILE)
    assert bad in decl["explicit_in_pack"]


def test_drain_join_is_total_over_composed_rows_and_vectors(tmp_path, monkeypatch):
    recs = _records() + [_page(SEARCH_URL, "A site search result page.")]
    ids, mint_map, _rejected, _decl = _declare(tmp_path, recs, monkeypatch,
                                               held=[recs[1]["point_id"]])
    from cytir.index import exclude_file as XF
    path = str(tmp_path / "p" / SE.SERVED_FILE)
    excl = XF.load_exclude_ids(path, sha256=XF.sha256_file(path), reason="x")
    rows, _skipped, _m, vector_key, _u = PSP.project_rows(
        iter(recs), consent=None, attribution=None, license_version=None, id_rule="I1")
    row_vectors = {str(vector_key.get(m, m)) for m in rows}
    served_minted = {mint_map[recs[1]["point_id"]]}
    # plan(): the vector side, less the excluded declared ids and the twin-served minted ones
    vec_side = {v for v in ids if v not in excl and mint_map.get(v) not in served_minted}
    row_side = {v for v in row_vectors if mint_map.get(v) not in served_minted}
    assert vec_side == row_side and vec_side, "vector/text join must be total"
    assert recs[-1]["point_id"] in excl


def test_pack_without_rejected_rows_writes_no_projection_field_or_file(tmp_path, monkeypatch):
    _ids, _mm, rejected, decl = _declare(tmp_path, _records(), monkeypatch)
    assert rejected == {}
    assert "projection_rejected" not in decl["fields"]
    assert not (tmp_path / "p" / SE.SERVED_FILE).exists()
    assert "exclude_ids_file" not in decl["fields"]


def test_other_projection_refusal_gets_the_other_reason(monkeypatch):
    rec = _page("https://example.org/ok", "Some text.")
    _patch(monkeypatch, [rec])
    monkeypatch.setattr(compose.PSP, "project_rows", lambda it, **kw: ({}, 1, {}, {}, []))
    _mm, rejected = compose._minted_and_rejected("x", [rec["point_id"]], {}, "I1")
    assert rejected == {rec["point_id"]: "projection-rejected: other"}


# ---- task 2222: review follow-ups on #1659 ------------------------------------------------------


def test_shared_declared_id_accepted_and_rejected_is_refused(monkeypatch):
    good = _page("https://example.org/ok", "Some text.")
    bad = dict(good)                                  # same declared point_id, other record
    _patch(monkeypatch, [good, bad])
    real = compose.PSP.project_rows

    def fake(it, **kw):
        rec = list(it)[0]
        if rec is bad:
            return {}, 1, {}, {}, []
        return real(iter([rec]), **kw)
    monkeypatch.setattr(compose.PSP, "project_rows", fake)
    with pytest.raises(compose.ComposeError) as exc:
        compose._minted_and_rejected("x", [good["point_id"]], {}, "I1")
    msg = str(exc.value)
    assert good["point_id"] in msg and "1 declared point_id" in msg and "no READY" in msg


def test_all_served_record_lists_the_projection_rejected_counts_and_ids(tmp_path, monkeypatch):
    import json
    recs = [_page("https://example.org/b", "Beta page text."), _page(SEARCH_URL, "Hits.")]
    ids, mint_map, rejected, _d = _declare(tmp_path, recs, monkeypatch)
    held = {i: {} for i in mint_map.values()}
    decl = SE.declare_already_served(
        pack_dir=tmp_path / "q", pack_id="q", ids=ids,
        targets={"twin": ("t", "c"), "prod": ("p", "c")},
        twin_rows=compose._ask_by_minted(lambda a: {i: held[i] for i in a if i in held}, mint_map),
        prod_present=compose._ask_by_minted(lambda a: [i for i in a if i in held], mint_map),
        pack_payloads=lambda a: {}, projection_rejected=rejected)
    assert decl["all_served"] is True
    path = SE.write_composed_to_nothing(
        tmp_path, pack_id="q", rows=2, fields=decl["fields"], excluded_ids=decl["excluded_ids"],
        payload_differs=decl["payload_differs"], family="f", pack_chunks="c", vectors="v",
        projection_rejected=rejected)
    rec = json.loads(path.read_text())
    assert rec["projection_rejected"] == {SEARCH_REASON: 1}
    side = path.parent / rec["projection_rejected_file"]
    assert [json.loads(x)["point_id"] for x in side.read_text().splitlines()] == list(rejected)
    assert rec["projection_rejected_sha256"] == SE.XF.sha256_file(side)
