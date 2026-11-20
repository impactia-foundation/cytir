# Task 650 (migration S8): lives in tests/ with the rest of the package's tests (task 696 slice 2).
"""Tests for cytir.distribute.pack — the declared dump payload projection
(task 650, S8; specs/dump-format.md)."""
from __future__ import annotations

import gzip
import json

import pytest

from cytir.distribute import pack

# ------------------------------------------------------------- fixtures
ENTITY_CARD_PAYLOAD = {
    "url": "https://example.ch/entity/1",
    "title": "Muster AG",
    "snippet": "Muster AG — Beratung. Registered signatories/officers "
               "(commercial register): Hans Muster, Anna Beispiel.",
    "dense_text": "Muster AG — Beratung. Registered signatories/officers "
                  "(commercial register): Hans Muster, Anna Beispiel.",
    "sparse_text": "muster ag beratung hans muster anna beispiel",
    "facet_terms": ["Hans Muster", "Anna Beispiel", "CHE-123.456.789"],
    "lang": "de",
    "trust_tier": 0.85,
    "last_seen": "2026-09-20T00:00:00Z",
    "country": "CH",
    "license": "unknown",
    "register_id": "CHE-123.456.789",
    "register_scheme": "zefix",
    "resolve_confidence": 0.92,
    "source_type": "entity_card",
    "domain": "example.ch",
}

DEFAULT_PAYLOAD = {
    "url": "https://en.wikipedia.org/wiki/Example",
    "title": "Example",
    "snippet": "Example is a placeholder domain name.",
    "lang": "en",
    "source_type": "wiki_article",
    "trust_tier": 0.9,
    "last_seen": "2026-09-20T00:00:00Z",
    "country": None,
    "license": "CC-BY-SA-4.0",
    "domain": "en.wikipedia.org",
    "tier": "A",
    "content_digest": "ab" * 32,
    "attribution": "Wikipedia contributors",
    "license_version": "4.0",
    "outlinks": ["https://example.com"],
}


# ---------------------------------------------------------------- projection
def test_default_projection_keeps_only_whitelisted_fields():
    row = pack.project_payload(DEFAULT_PAYLOAD)
    assert set(row) <= set(pack.DEFAULT_PROJECTION_FIELDS)
    assert row["license"] == "CC-BY-SA-4.0"
    assert "attribution" not in row
    assert "license_version" not in row
    assert "outlinks" not in row
    # a None value is dropped, not shipped as null
    assert "country" not in row


def test_entity_card_projection_never_ships_officer_names_as_text():
    row = pack.project_payload(ENTITY_CARD_PAYLOAD)
    assert set(row) <= set(pack.ENTITY_CARD_PROJECTION_FIELDS)
    blob = json.dumps(row)
    assert "Hans Muster" not in blob
    assert "Anna Beispiel" not in blob
    assert "dense_text" not in row
    assert "sparse_text" not in row
    assert "facet_terms" not in row
    assert row["source_type"] == "entity_card"


def test_entity_card_snippet_recomposed_when_officer_clause_cut_short():
    # A shorter dense_text where the officer clause is still the tail.
    payload = dict(ENTITY_CARD_PAYLOAD)
    payload["dense_text"] = ("Kurz AG — Beratung. Registered "
                              "signatories/officers (commercial register): "
                              "Peter Vertreter.")
    row = pack.project_payload(payload)
    assert "Peter Vertreter" not in row["snippet"]
    assert row["snippet"].startswith("Kurz AG")


def test_entity_card_snippet_strips_no_parenthetical_officer_phrasing():
    # the ADR-060 card recipe's wikidata-phase-3 builder appends
    # "Registered signatories/officers: <names>." with NO "(commercial
    # register)" parenthetical -- a different literal from
    # compose_register_card's. task 654: found by reading that builder,
    # not by a failing test; regression-guards the invariant-marker regex.
    payload = dict(ENTITY_CARD_PAYLOAD)
    payload["dense_text"] = ("Vector8 AG — Beratung. Registered "
                              "signatories/officers: Marc Maret; Nina Keller.")
    row = pack.project_payload(payload)
    assert "Marc Maret" not in row["snippet"]
    assert "Nina Keller" not in row["snippet"]
    assert row["snippet"].startswith("Vector8 AG")


def test_entity_card_snippet_falls_back_to_title_if_stripped_to_empty():
    payload = dict(ENTITY_CARD_PAYLOAD)
    payload["dense_text"] = ("Registered signatories/officers (commercial "
                              "register): Only Officer.")
    row = pack.project_payload(payload)
    assert row["snippet"] == payload["title"]


def test_project_payload_refuses_row_with_no_url():
    with pytest.raises(pack.ProjectionRefusal):
        pack.project_payload({"title": "no url here", "source_type": "x"})


def test_ships_sparse():
    assert pack.ships_sparse(DEFAULT_PAYLOAD) is True
    assert pack.ships_sparse(ENTITY_CARD_PAYLOAD) is False


# ----------------------------------------------------------------- tombstones
def test_tombstone_rejects_unknown_reason():
    with pytest.raises(ValueError):
        pack.Tombstone(id="x", reason="because", at="2026-09-25T00:00:00Z")


def test_tombstone_roundtrip(tmp_path):
    tombstones = [
        pack.Tombstone(id="a", reason="superseded", at="2026-09-25T00:00:00Z"),
        pack.Tombstone(id="b", reason="takedown", at="2026-09-25T00:00:01Z"),
    ]
    path = str(tmp_path / "tombstones.jsonl.gz")
    n, sha = pack.write_tombstones(tombstones, path)
    assert n == 2
    assert sha == pack.sha256_file(path)
    got = list(pack.read_tombstones(path))
    assert [t.id for t in got] == ["a", "b"]
    assert [t.reason for t in got] == ["superseded", "takedown"]
    with gzip.open(path, "rt") as fh:
        lines = fh.readlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["reason"] == "superseded"


# ------------------------------------------------------------------- parquet
def test_parquet_shard_roundtrip(tmp_path):
    rows = [
        pack.row_to_record("id-1", [0.1, 0.2, 0.3],
                           {"url": "https://a", "title": "A"},
                           ([1, 5], [0.4, 0.6])),
        pack.row_to_record("id-2", [0.4, 0.5, 0.6],
                           {"url": "https://b", "title": "B"}, None),
    ]
    path = str(tmp_path / "shard-00000.parquet")
    pack.write_parquet_shard(rows, path)
    got = list(pack.read_parquet_shard(path))
    assert len(got) == 2
    assert got[0]["id"] == "id-1"
    assert got[0]["payload"] == {"url": "https://a", "title": "A"}
    assert got[0]["sparse_indices"] == [1, 5]
    assert pytest.approx(got[0]["dense"], abs=1e-6) == [0.1, 0.2, 0.3]
    assert got[1]["sparse_indices"] is None
    assert got[1]["sparse_values"] is None


def test_parquet_shard_handles_full_uint32_bm25_hash_range(tmp_path):
    # task 654: the legacy `aw_common` `_hash` is `int(md5(term)[:8], 16)` --
    # the full UNSIGNED 32-bit range. The first real-twin dump shard (a
    # signed-int32 sparse_indices column) raised "Value 2167715904 too large
    # to fit in C integer type" -- a value comfortably above INT32_MAX
    # (2147483647) but a completely ordinary BM25 term hash. A synthetic
    # fixture with small indices like [1, 5] never exercises this.
    big = 2167715904  # > INT32_MAX, < UINT32_MAX; the real value that broke it
    row = pack.row_to_record(
        "id-real", [0.1, 0.2], {"url": "https://c", "title": "C"},
        ([big, 4294967295, 0], [0.7, 0.3, 0.1]))
    path = str(tmp_path / "shard-00001.parquet")
    pack.write_parquet_shard([row], path)
    got = list(pack.read_parquet_shard(path))
    assert got[0]["sparse_indices"] == [big, 4294967295, 0]


# ------------------------------------------------------------------ manifest
def test_validate_manifest_requires_fields():
    with pytest.raises(ValueError):
        pack.validate_manifest({"schema": pack.DUMP_MANIFEST_SCHEMA})


def test_validate_manifest_accepts_minimal_dump_manifest():
    m = {
        "schema": pack.DUMP_MANIFEST_SCHEMA, "dump_id": "x",
        "created_at": "now", "source": {}, "embedder": {}, "counts": {},
        "shards": [], "license": {},
    }
    assert pack.validate_manifest(m) is m


def test_validate_manifest_rejects_unknown_schema():
    with pytest.raises(ValueError):
        pack.validate_manifest({"schema": "not-a-real-schema"})


# ------------------------------------------- task 768: digest + consent flags + privacy
def test_digest_ships_as_32_hex_prefix_of_the_64_hex_sha256():
    full = "0123456789abcdef" * 4
    assert pack.project_payload({**DEFAULT_PAYLOAD, "content_digest": full})["content_digest"] == full[:32]
    assert pack.project_payload({**DEFAULT_PAYLOAD, "content_digest": full[:32].upper()})[
        "content_digest"] == full[:32]


@pytest.mark.parametrize("bad", ["deadbeef", "a" * 40, "z" * 64, 12345, ""])
def test_a_digest_that_is_not_the_recipe_is_not_shipped(bad):
    assert "content_digest" not in pack.project_payload({**DEFAULT_PAYLOAD, "content_digest": bad})


def test_consent_fields_ship_on_every_row_and_unknown_is_null_never_true():
    bare = pack.project_payload(DEFAULT_PAYLOAD)
    assert all(k in bare and bare[k] is None for k in pack.CONSENT_FIELDS)
    card = pack.project_payload(ENTITY_CARD_PAYLOAD)
    assert all(card[k] is None for k in pack.CONSENT_FIELDS)
    full = pack.project_payload({**DEFAULT_PAYLOAD, "search_ok": True, "train_ok": False,
                                 "consent_basis": "robots-search-ok", "consent_state": "resolved"})
    assert (full["search_ok"], full["train_ok"]) == (True, False)
    assert full["consent_basis"] == "robots-search-ok" and full["consent_state"] == "resolved"
    assert pack.project_payload({**DEFAULT_PAYLOAD, "search_ok": False})["search_ok"] is False


def test_a_coded_consent_value_is_decoded_through_the_server_table():
    from cytir.index import payload_profile as PP
    tables = PP.load_tables()
    code = tables["consent_basis"]['"robots-search-ok"']
    scode = tables["consent_state"][next(iter(tables["consent_state"]))]
    state = json.loads(next(v for v, c in tables["consent_state"].items() if c == scode))
    row = pack.project_payload({**DEFAULT_PAYLOAD, "consent_basis": code, "consent_state": scode,
                                "search_ok": True, "train_ok": True})
    assert row["consent_basis"] == "robots-search-ok" and row["consent_state"] == state


def test_an_unknown_consent_code_or_a_non_bool_flag_is_refused():
    with pytest.raises(pack.ProjectionRefusal):
        pack.project_payload({**DEFAULT_PAYLOAD, "consent_basis": 99999})
    with pytest.raises(pack.ProjectionRefusal):
        pack.project_payload({**DEFAULT_PAYLOAD, "train_ok": "yes"})


def test_master_only_fields_never_ship_ADR_039():
    row = pack.project_payload({**DEFAULT_PAYLOAD, "owner": "Jane Doe", "consent_checked_at": "x",
                                "consent_source": "host robots.txt sentence"})
    for k in ("owner", "attribution", "license_version", "author", "outlinks",
              "consent_checked_at", "consent_source"):
        assert k not in row


def test_no_register_person_name_is_readable_in_a_card_record_memo_645():
    names = ["Hans Muster", "Anna Beispiel"]
    payload = {**ENTITY_CARD_PAYLOAD, "search_ok": True, "train_ok": True,
               "consent_basis": "register-open-data", "consent_state": "resolved",
               "owner": "Hans Muster", "officers": names}
    row = pack.project_payload(payload)
    text = json.dumps(row, ensure_ascii=False)
    for name in names + ["Muster, Hans", "HANS MUSTER"]:
        assert name.lower() not in text.lower()
    assert "registered signatories" not in text.lower()
    assert pack.ships_sparse(payload) is False
