"""The fedlex article-title mirror contract (task 516).

the old `fedlex_dump` loader (the live loader, which wrote the 23,872 fedlex rows
production serves) and task 417's `materialize_r0_fedlex_anchored` script
(the pinned epoch-4 re-materialization) must compose the SAME title from the
same inputs. They did not: the materializer composed `docTitle - <num>` and
shipped 0 of 22,928 titles carrying the SR number or the law's abbreviation,
against 98.4% / 97.7% in production (task 515 section 7.3). Every legal gold
the 1024 build lost is abbreviation-led.

`fedlex_akn.compose_article_title` is now the one implementation. This file
asserts it against production's ORIGINAL formula, transcribed here verbatim as
the reference oracle, so a future edit to either side has to break a test
rather than a build.

Run: python3 -m pytest tests/test_fedlex_akn.py
"""
from __future__ import annotations

import pathlib
import sys

import pytest
from lxml import etree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cytir.sources.families import fedlex_akn as F  # noqa: E402

NS = F.AKN


def production_formula(title_prefix: str, art_cite: str, marginal: str) -> str:
    """`fedlex_dump.parse_akn` as it stood at commit 8757cf47, verbatim."""
    title = title_prefix
    title = "%s — %s" % (title, art_cite)
    if marginal:
        title = "%s — %s" % (title, marginal)
    return title


CASES = [
    ("Obligationenrecht (OR; SR 220)", "Art. 411", "IV. Kreditnehmer und Auftraggeber"),
    ("Urheberrechtsgesetz (URG; SR 231.1)", "Art. 21", ""),
    ("Loi sur la TVA (LTVA; SR 641.20)", "Art. 25", "Chapitre 2 Taux de l'impôt"),
    ("Alkoholgrenzwerte im Strassenverkehr (SR 741.13)", "Art. 1", "1. Abschnitt"),
    # A `<num>` that exists but is empty: production kept the separator, and so
    # must the shared implementation, or 1024 titles differ from production's
    # for no reason anybody chose.
    ("Datenschutzgesetz (DSG; SR 235.1)", "", "2. Abschnitt"),
    ("Zivilgesetzbuch (ZGB; SR 210)", "art_730", ""),
]


@pytest.mark.parametrize("prefix,cite,marginal", CASES)
def test_compose_article_title_matches_production_byte_for_byte(prefix, cite, marginal):
    assert F.compose_article_title(prefix, cite, marginal) == \
        production_formula(prefix, cite, marginal)


def test_the_composed_title_is_the_one_the_twin_serves():
    """The exact string task 515 read off the production twin."""
    assert F.compose_article_title(
        "Obligationenrecht (OR; SR 220)", "Art. 411",
        "IV. Kreditnehmer und Auftraggeber") == \
        "Obligationenrecht (OR; SR 220) — Art. 411 — IV. Kreditnehmer und Auftraggeber"


def _article(xml: str):
    return next(etree.fromstring(xml.encode()).iter("{%s}article" % NS))


def test_ancestor_heading_path_joins_num_and_heading_with_a_space():
    article = _article(f'''<akomaNtoso xmlns="{NS}"><act><body>
      <title eId="t"><num>Achter Titel:</num><heading>Die Miete</heading>
        <chapter eId="c"><num>IV.</num><heading>Kreditnehmer und Auftraggeber</heading>
          <article eId="art_1"><num>Art. 1</num></article>
        </chapter></title></body></act></akomaNtoso>''')
    path, marginal = F.ancestor_heading_path(article)
    assert path == ["Achter Titel: Die Miete", "IV. Kreditnehmer und Auftraggeber"]
    assert marginal == "IV. Kreditnehmer und Auftraggeber"


def test_ancestor_heading_path_sees_ancestors_without_an_eid():
    """Production's marginal does not require the ancestor to be anchored."""
    article = _article(f'''<akomaNtoso xmlns="{NS}"><act><body>
      <chapter><num>IV.</num><heading>Ohne eId</heading>
        <article eId="art_1"><num>Art. 1</num></article></chapter>
      </body></act></akomaNtoso>''')
    assert F.ancestor_heading_path(article)[1] == "IV. Ohne eId"


def test_article_citation_falls_back_to_the_anchor_only_when_num_is_absent():
    with_num = _article(f'<article xmlns="{NS}" eId="art_1"><num>Art. 1</num></article>')
    without = _article(f'<article xmlns="{NS}" eId="art_1"><paragraph>x</paragraph></article>')
    empty = _article(f'<article xmlns="{NS}" eId="art_1"><num/></article>')
    assert F.article_citation(with_num, "art_1") == "Art. 1"
    assert F.article_citation(without, "art_1") == "art_1"
    assert F.article_citation(empty, "art_1") == ""


def test_fedlex_dump_imports_the_shared_contract_rather_than_forking_it():
    """One home. The two copies of these helpers are how the fork happened."""
    # the loader moved into the package (task 683); the old `fedlex_dump` module is a shim now
    source = (pathlib.Path(__file__).resolve().parents[1] / "cytir" / "sources" / "families"
              / "fedlex.py").read_text()
    assert "from cytir.sources.families.fedlex_akn import" in source
    for forked in ("def ancestor_heading_path", "def _clean_text_parts",
                   "def source_note_pointer", "def _text_without"):
        assert forked not in source, f"{forked} is defined twice again"
