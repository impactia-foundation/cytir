"""`cytir.index.row_decode`: one canonical row for a stored payload of any profile (task 855).

The property: for every representative row (20 real twin rows read by id, the six mirror-guard
rows on both sides, the news daily sample, the consent family rows) and every CODED profile P,

    decode(project(raw, P), P) == canonical(raw, P) == decode(raw, P)

where `canonical` is written out HERE, independently of the module: the enum values as they are,
`content_digest` cut to its first 32 hex, `published` in its contract form, P's dropped keys
removed. And the raw twin row read as a raw twin (`prod`) equals the coded row read as P except for
the keys P does not store, which the coded row names in `absent`.
"""
from __future__ import annotations

import copy
import json
import os
import pickle

import pytest

import coded_twin_rows as CT  # noqa: E402  (tests/)
from cytir.core.published import normalize_payload_published
from cytir.index import payload_profile as PP
from cytir.index import row_decode as RD

CODED_PROFILES = [p for p, rules in PP.PROFILE_RULES.items() if rules]
ROWS = CT.representative_rows()


def expected_row_by_hand(raw: dict, profile: str) -> dict:
    drop = {k for k, r in PP.PROFILE_RULES[profile].items() if r == "drop"}
    out = {}
    for k, v in normalize_payload_published(raw).items():
        if k in drop:
            continue
        out[k] = v[:32] if (k == "content_digest" and isinstance(v, str)) else v
    return out


def test_the_rows_cover_every_family_and_the_awkward_values():
    labels = [l for l, _ in ROWS]
    assert len([l for l in labels if l.startswith("twin:")]) == 20
    payloads = [p for _, p in ROWS]
    provs = {p.get("provenance") for p in payloads}
    assert {"loda", "weko_docs", "sitemap-ingest", "wikidata_ch", "l1_backfill_refetch"} <= provs
    assert any(p.get("source_type") == "entity_card" for p in payloads)
    assert any(len(p.get("content_digest") or "") == 64 for p in payloads)
    # an open-key value with no code (written raw by every writer) and one with a code
    t = CT.tables()
    cs = [p["consent_source"] for p in payloads if "consent_source" in p]
    assert any(PP._canon(v) not in t["consent_source"] for v in cs)
    assert any(PP._canon(v) in t["consent_source"] for v in cs)
    assert t["provenance"][PP._canon("weko_docs")] == 0          # code 0: "no provenance" to `0 or ""`


@pytest.mark.parametrize("profile", CODED_PROFILES)
@pytest.mark.parametrize("label,raw", ROWS, ids=[l for l, _ in ROWS])
def test_decode_of_the_projection_is_the_canonical_row(label, raw, profile):
    stored = CT.coded(raw, profile)
    assert PP.form_of(stored, profile, frozenset({"consent_source"})) in ("projected", "n/a")
    want = expected_row_by_hand(raw, profile)
    got = RD.decode_row(stored, profile=profile)
    assert got == want
    assert RD.decode_row(raw, profile=profile) == want
    assert got.absent == frozenset(k for k, r in PP.PROFILE_RULES[profile].items() if r == "drop")
    # the raw twin row read as a RAW twin differs from the coded one only by the absent keys
    as_raw_twin = RD.decode_row(raw, profile="prod")
    assert {k: v for k, v in as_raw_twin.items() if k not in got.absent} == dict(got)
    assert as_raw_twin.absent == frozenset()
    # idempotent, and undeclared (profile None) decodes the values the same way
    assert RD.decode_row(got, profile=profile) == got
    assert {k: v for k, v in RD.decode_row(stored, profile=None).items()} == dict(got)


def test_digest_key_is_the_projection_prefix_and_same_digest_ignores_the_form():
    raw = "30c320257eee2f8787532a432c1039f3e933421fe7723bed0000000000000000"
    assert RD.digest_key(raw) == raw[:32] == PP.project_payload({"content_digest": raw}, "normalized",
                                                                 {})["content_digest"]
    assert RD.digest_key(raw[:32]) == raw[:32]
    assert RD.digest_key(None) is None
    assert RD.same_digest(raw, raw[:32]) and not RD.same_digest(raw, None)
    assert not RD.same_digest(raw, "f" * 64)


def test_a_code_with_no_value_is_loud_and_an_unknown_open_value_is_kept():
    with pytest.raises(RD.RowDecodeError):
        RD.decode_row({"provenance": 999_999}, profile="rebuild_e10")
    assert RD.decode_row({"consent_source": "a sentence with no code"},
                         profile="rebuild_e10")["consent_source"] == "a sentence with no code"
    assert RD.decode_row({"provenance": True}, profile=None)["provenance"] is True   # bool is no code


def test_profile_of_reads_the_declaration_never_the_row():
    assert RD.profile_of("aw_e8") == "prod"
    assert RD.profile_of("aw_e9") == "normalized"
    assert RD.profile_of("aw_prod") is None            # an alias is not declared
    assert RD.profile_of("no_such_collection") is None
    assert RD.reader("aw_e8").absent == frozenset()
    assert RD.reader("aw_e9").absent == frozenset({"stamps"})


def test_the_flip_day_switch_is_the_declaration_and_needs_no_restart(monkeypatch, tmp_path):
    """The twin's profile flips when the declaration names it coded: a decoder resolved before
    the data commit is not reused after it (cached on the file's mtime and size)."""
    path = CT.declare(monkeypatch, tmp_path, profile="prod")         # the new twin, declared raw
    before = RD.reader(CT.CODED_TWIN)
    assert before.profile == "prod" and before.absent == frozenset()
    doc = json.loads(path.read_text())
    doc["collections"][CT.CODED_TWIN]["profile"] = "rebuild_e10"   # the data commit
    path.write_text(json.dumps(doc, indent=2))
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    after = RD.reader(CT.CODED_TWIN)
    assert after.profile == "rebuild_e10"
    assert after.absent == frozenset({"stamps", "point_id", "tier", "build_id"})
    raw = CT.real_rows()[0]["payload"]
    assert after(CT.coded(raw)) == after(raw)


def test_a_canonical_row_keeps_its_markers_through_copy_and_pickle():
    row = RD.decode_row(CT.coded(CT.real_rows()[1]["payload"]), profile="rebuild_e10")
    for other in (copy.copy(row), copy.deepcopy(row), pickle.loads(pickle.dumps(row))):
        assert other == row and other.absent == row.absent and other.profile == "rebuild_e10"
    assert RD.as_canonical_row(row) is row
    assert RD.as_canonical_row({"provenance": 0})["provenance"] == "weko_docs"


def test_an_undeclared_collection_reads_raw_rows_but_refuses_a_code():
    """An index arm built with `grow=True` numbers values its own way: a code of an undeclared
    collection is never read with the committed table (review of #791, S1)."""
    raw = CT.real_rows()[0]["payload"]
    dec = RD.reader("aw_scratch_not_declared")
    assert dec.profile is None and not dec.declared
    assert dec(raw) == RD.decode_row(raw, profile=None)
    with pytest.raises(RD.RowDecodeError, match="not declared"):
        dec(CT.coded(raw, "normalized"))
    with pytest.raises(RD.RowDecodeError):
        RD.reader("aw_prod")({"provenance": 27})          # an alias: resolve it first
    assert RD.as_canonical_row({"provenance": 27})["provenance"] == "loda"   # a handed-in row: committed table


def test_no_table_value_is_an_integer_so_an_int_is_always_a_code():
    for key, table in CT.tables().items():
        assert key in RD.ENUM_KEYS
        for value_json in table:
            v = json.loads(value_json)
            assert not (isinstance(v, int) and not isinstance(v, bool)), (key, value_json)
