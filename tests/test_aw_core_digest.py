"""Byte-identity of the shared digest home against the old per-site bodies (task 1181)."""
from __future__ import annotations

import hashlib
import itertools
import json

from cytir.core import digest

VALUES = [
    {}, [], None, 0, -7, 3.5, 1e300, True, "plain", "Zürich ü ß 日本   \U0001F600",
    {"b": 1, "a": 2}, {"a": 2, "b": 1}, {"z": None, "ä": [1, 2.0, "x"], "k": {"n": {"d": [None]}}},
    [1, [2, [3, {"x": "é"}]]], {"1": 1, "10": 10, "2": 2}, {"é": "è", "e": "e"},
    [0.1, 0.2, 1 / 3, 2 ** 70], {"": ""}, {"u": "é́"}, ["a\nb\t\"q\"\\"],
    {"vec": [0.0, -0.0, 1.5e-8], "t": True, "f": False},
    {"nested": {"k%d" % i: [i, str(i) * 3] for i in range(12)}},
    {"name": "Müller & Söhne AG", "city": "Genève"}, ["Ǆ", "ﬁ", "\u2028", "\x00"],
    {"k": {"é": {"ü": [1.25, -2.5, 1e-7]}}}, {"emoji": "\U0001F4A9", "n": 10 ** 20},
    {"a": [{"b": [{"c": "ç"}]}]}, {"x": 1.0, "y": 100.0, "z": 1e21}, "Ärger", "日本語",
    {"sorted": ["z", "a", "é", "e"]}, {"é": 1, "z": 2, "a": 3}, [None, True, False, 0, ""],
    {"deep": [[[[["ö"]]]]]},
]
VALUES += [dict(p) for p in itertools.permutations([("a", 1), ("b", [2, "ö"]), ("c", None)])]


def _old_news(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _old_apply(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _old_run_corpus_arm(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _old_content_pack_preservation(value) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def test_enough_values():
    assert len(VALUES) >= 20


def test_line_bytes_identical_to_old_news_and_apply():
    for v in VALUES:
        assert digest.canonical_line_bytes(v) == _old_news(v) == _old_apply(v)


def test_sha256_identical_to_old_arm_and_preservation():
    for v in VALUES:
        assert digest.canonical_sha256(v) == _old_run_corpus_arm(v) == _old_content_pack_preservation(v)


def test_key_order_does_not_matter():
    assert digest.canonical_sha256({"a": 1, "b": 2}) == digest.canonical_sha256({"b": 2, "a": 1})


def test_callers_use_the_home():
    from cytir.gates import run_corpus_arm
    from cytir.index import apply, content_pack_preservation
    from cytir.sources.families import news
    assert apply.canonical is digest.canonical_line_bytes
    assert news.canonical is digest.canonical_line_bytes
    assert content_pack_preservation.canonical_sha256 is digest.canonical_sha256
    assert run_corpus_arm.canonical_sha256 is digest.canonical_sha256


# --- task 1191: the ensure_ascii=False (utf-8) family ---

def _old_provenance_canonical_json(value) -> str:  # verbatim pre-1191 body
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _old_facet_bundle_canonical(value) -> str:  # verbatim pre-1191 body
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _old_site_associations_canonical_json(value) -> str:  # verbatim pre-1191 body
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _old_chunk2_delta_journal_canonical_bytes(value) -> bytes:  # verbatim pre-1191 body
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def _old_serving_profile_canonical_digest(stored) -> str:  # verbatim pre-1191 body
    blob = json.dumps(stored, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _old_collection_membership_canonical_row(row) -> bytes:  # verbatim pre-1191 body (dumps part)
    payload = row.get("payload") or {}
    return json.dumps(
        {"id": str(row["id"]), "payload": payload}, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode("utf-8")


def _old_verify_collection_membership_canonical(row) -> bytes:  # verbatim pre-1196 body, minus the two validation lines
    payload = row.get("payload") or {}
    return json.dumps(
        {"id": str(row["id"]), "payload": payload}, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode("utf-8")


def test_utf8_family_has_enough_values():
    assert len(VALUES) >= 40


def test_utf8_family_identical_to_old_bodies():
    for v in VALUES:
        s = digest.canonical_json_utf8(v)
        assert s == _old_provenance_canonical_json(v) == _old_facet_bundle_canonical(v) \
            == _old_site_associations_canonical_json(v)
        assert digest.canonical_utf8_bytes(v) == _old_chunk2_delta_journal_canonical_bytes(v)
        assert digest.canonical_utf8_sha256(v) == _old_serving_profile_canonical_digest(v)
        row = {"id": 7, "payload": v if isinstance(v, dict) else {"v": v}}
        assert digest.canonical_utf8_bytes(
            {"id": str(row["id"]), "payload": row["payload"]}) == _old_collection_membership_canonical_row(row)
        assert digest.canonical_utf8_bytes(
            {"id": str(row["id"]), "payload": row["payload"]}) == _old_verify_collection_membership_canonical(row)


def test_utf8_family_keeps_non_ascii():
    assert digest.canonical_json_utf8({"k": "Zürich"}) == '{"k":"Zürich"}'
    assert digest.canonical_line_bytes({"k": "Zürich"}) != digest.canonical_utf8_bytes({"k": "Zürich"})


def test_utf8_callers_use_the_home():
    from cytir.chunk import provenance
    assert provenance.canonical_json is digest.canonical_json_utf8
