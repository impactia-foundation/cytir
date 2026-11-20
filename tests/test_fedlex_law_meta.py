#!/usr/bin/env python3
"""The fedlex law-metadata contract (task 516).

The abbreviation and the SR number are METADATA -- fedlex SPARQL's
`jolux:titleShort` and `jolux:historicalLegalId` -- and the short common name is
a curated table. The epoch-4 re-materializer resolved none of them and shipped
22,928 statutes whose own abbreviation appears nowhere in their title or in the
BM25 surface composed from it.

These tests hold the three things that must not drift again:
the composition, the curated table, and the sealed artifact's digest binding.

Run: python3 -m pytest tests/test_fedlex_law_meta.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from cytir.sources.families import fedlex_law_meta as LM  # noqa: E402

REAL = json.loads((pathlib.Path(__file__).resolve().parent.parent
                   / "data/fedlex-law-names.v1.json").read_text())


# --------------------------------------------------------------- composition
def test_compose_title_prefix_matches_the_production_formula():
    """`fedlex_dump.parse_akn` as it stood before task 516, verbatim."""
    def production(name, abbrev, sr):
        paren = "; ".join(x for x in (abbrev, ("SR %s" % sr) if sr else None) if x)
        return "%s (%s)" % (name, paren) if paren else name

    for name, abbrev, sr in [
            ("Obligationenrecht", "OR", "220"),
            ("Alkoholgrenzwerte im Strassenverkehr", None, "741.13"),
            ("Datenschutzgesetz", "DSG", None),
            ("Some Act", None, None),
            ("Loi sur la TVA", "LTVA", "641.20")]:
        assert LM.compose_title_prefix(name, abbrev, sr) == production(name, abbrev, sr)


def test_the_prefix_is_the_string_the_twin_serves():
    assert LM.compose_title_prefix("Obligationenrecht", "OR", "220") == \
        "Obligationenrecht (OR; SR 220)"
    assert LM.compose_title_prefix(
        "Alkoholgrenzwerte im Strassenverkehr", None, "741.13") == \
        "Alkoholgrenzwerte im Strassenverkehr (SR 741.13)"


def test_a_law_with_neither_is_its_bare_name_not_empty_parentheses():
    assert LM.compose_title_prefix("Verordnung X", None, None) == "Verordnung X"


# -------------------------------------------------------------- curated table
def test_the_curated_table_is_data_not_code():
    """It moved out of `fedlex_dump.py` the moment a second producer read it."""
    source = (pathlib.Path(__file__).resolve().parents[1] / "cytir" / "sources" / "families"
              / "fedlex.py").read_text()        # moved into the package (task 683)
    assert "LAW_META = {" not in source
    assert "fedlex_law_meta as LM" in source


def test_the_curated_table_still_carries_the_18_pinned_elis():
    names = LM.load_curated_names()
    assert len(names) == 18
    assert names["cc/27/317_321_377"]["de"] == "Obligationenrecht"
    assert names["cc/1993/1798_1798_1798"]["fr"] == "Loi sur le droit d'auteur"


def test_the_curated_names_are_short_common_names_not_doc_titles():
    """The whole point of the table: distinct from both the bare abbreviation
    and the formal docTitle. A 'name' that is 90 characters of formal title is
    the defect wearing the fix's clothes."""
    for eli, langs in LM.load_curated_names().items():
        for lang, name in langs.items():
            assert 3 <= len(name) <= 90, (eli, lang, name)
            assert not name.lower().startswith("bundesgesetz vom"), (eli, lang)


def test_short_name_falls_back_loudly_and_in_order():
    warnings = []
    curated = {"cc/1/1": {"de": "Kurzname"}}
    assert LM.short_name(curated, "cc/1/1", "de", "ABK", "Formal", warnings.append) \
        == "Kurzname"
    assert not warnings
    assert LM.short_name(curated, "cc/9/9", "de", "ABK", "Formal", warnings.append) \
        == "ABK"
    assert len(warnings) == 1 and "no curated short name" in warnings[0]
    assert LM.short_name(curated, "cc/9/9", "de", None,
                         "Formal docTitle " * 10, warnings.append).startswith("Formal")
    assert len(warnings) == 2


# ------------------------------------------------------------ sealed artifact
def _seal(tmp_path, elis):
    doc = {"schema": LM.SCHEMA, "resolved_utc": "2026-09-05T02:09:54Z",
           "endpoint": LM.SPARQL_ENDPOINT, "elis": elis}
    body = (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode()
    p = tmp_path / "law-meta.json"
    p.write_bytes(body)
    return p, hashlib.sha256(body).hexdigest()


def test_the_sealed_artifact_is_bound_by_digest(tmp_path):
    p, sha = _seal(tmp_path, {"cc/1/1": {"de": {"title_prefix": "X (Y; SR 1)"}}})
    assert LM.load_sealed(p, sha)["elis"]
    with pytest.raises(ValueError, match="sha256"):
        LM.load_sealed(p, "0" * 64)


def test_a_sealed_artifact_of_the_wrong_schema_is_refused(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"schema": "something.else", "elis": {"a": {}}}))
    with pytest.raises(ValueError, match="not aw.fedlex-law-meta.v1"):
        LM.load_sealed(p)


def test_an_empty_sealed_artifact_is_refused(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"schema": LM.SCHEMA, "elis": {}}))
    with pytest.raises(ValueError, match="carries no ELIs"):
        LM.load_sealed(p)


def test_sealed_title_prefix_reads_only_a_real_prefix(tmp_path):
    sealed = {"elis": {"cc/1/1": {"de": {"title_prefix": "X (Y; SR 1)"},
                                  "fr": {"title_prefix": "  "},
                                  "it": {"abbrev": "Z"}}}}
    assert LM.sealed_title_prefix(sealed, "cc/1/1", "de") == "X (Y; SR 1)"
    assert LM.sealed_title_prefix(sealed, "cc/1/1", "fr") is None
    assert LM.sealed_title_prefix(sealed, "cc/1/1", "it") is None
    assert LM.sealed_title_prefix(sealed, "cc/9/9", "de") is None


# ------------------------------------------------------------- the resolution
class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


class _FakeHttp:
    """Records the queries so the politeness and the backfill are checkable."""

    def __init__(self, per_lang):
        self.per_lang = per_lang
        self.calls = []

    def post(self, url, data=None, headers=None):
        query = data["query"]
        lang = next(l for l in ("de", "fr", "it", "en") if f"/{l}>" in query)
        self.calls.append(lang)
        row = self.per_lang.get(lang)
        bindings = [] if row is None else [
            {k: {"value": v} for k, v in row.items() if v is not None}]
        return _FakeResponse({"results": {"bindings": bindings}})


def test_de_fr_it_are_always_probed_so_the_sr_backfill_has_data():
    """The SR number is language-invariant but fedlex answers it for only some
    languages -- SchKG/en and DSG/de-fr-it each omit `historicalLegalId` while a
    sibling carries it (verified 2026-07-30, task 358b)."""
    http = _FakeHttp({"de": {"title": "T", "short": "OR", "sr": "220"},
                      "fr": {"title": "T", "short": "CO"},
                      "it": {"title": "T", "short": "CO"},
                      "en": {"title": "T", "short": "CO"}})
    out = LM.resolve_law_meta("cc/1/1", ["en"], http, min_interval_s=0)
    assert sorted(http.calls) == ["de", "en", "fr", "it"]
    assert out["en"]["sr"] == "220" and out["fr"]["sr"] == "220"


def test_a_language_the_endpoint_does_not_answer_is_absent_not_invented():
    http = _FakeHttp({"de": {"title": "T", "short": "OR", "sr": "220"}})
    out = LM.resolve_law_meta("cc/1/1", ["de", "en"], http, min_interval_s=0)
    assert "en" not in out
