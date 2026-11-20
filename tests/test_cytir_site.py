"""`cytir.ops.cytir_site`: the four static pages of cytir.org are the render of one template + four strings
files, and every language carries every string (task 2654)."""
from __future__ import annotations

import re

import pytest

from cytir.ops import cytir_site as m

HEADLINE = {"en": "common land", "fr": "terre commune", "de": "Allmend", "it": "terra comune"}


def _strings():
    return {lg: m.load_strings(lg) for lg in m.LANGS}


def test_the_committed_pages_equal_the_render():
    for path, text in m.render_all().items():
        assert path.is_file(), f"{path}: run python -m cytir.ops.cytir_site --write"
        assert path.read_text(encoding="utf-8") == text, f"{path}: run python -m cytir.ops.cytir_site --write"
    assert m.stale() == []
    assert m.main(["--check"]) == 0


def test_every_english_key_exists_in_the_other_languages_and_is_not_empty():
    s = _strings()
    assert len(s["en"]) > 60
    for lg in m.LANGS[1:]:
        assert set(s[lg]) == set(s["en"]), lg
        assert all(isinstance(v, str) and v.strip() for v in s[lg].values()), lg
    assert m.problems(m.TEMPLATE.read_text(encoding="utf-8"), s) == []


def test_each_page_has_its_lang_the_four_hreflangs_and_the_switcher():
    for lg in m.LANGS:
        html = m.page_path(lg).read_text(encoding="utf-8")
        assert f'<html lang="{lg}">' in html
        for other in (*m.LANGS, "x-default"):
            assert re.search(rf'<link rel="alternate" hreflang="{other}" href="https://cytir\.org/[a-z/]*">', html), (lg, other)
        assert f'<link rel="canonical" href="{m.page_url(lg)}">' in html
        assert len(re.findall(r'<a [^>]*aria-current="page"', html)) == 1
        assert f'hreflang="{lg}" aria-current="page"' in html
        assert HEADLINE[lg] in re.search(r"<h1>(.*?)</h1>", html).group(1)
        # one asset tree for all four pages: no path relative to the page
        assert not re.search(r'(?:src|href)="(?:\.\./)?assets/', html)


def test_the_lead_names_cytir_and_the_welsh_gloss_in_every_language():
    gloss = {"en": "Welsh", "fr": "en gallois", "de": "walisisch", "it": "in gallese"}
    for lg in m.LANGS:
        lead = m.load_strings(lg)["hero_lead"]
        assert "<strong>Cytir</strong>" in lead and gloss[lg] in lead, lg


def test_swiss_orthography_and_the_stable_code_block():
    for lg in m.LANGS:
        html = m.page_path(lg).read_text(encoding="utf-8")
        assert "ß" not in html
        assert '"cytir-websearch": { "url": "https://mcp.websear.ch" }' in html


def test_a_missing_or_stray_key_is_reported():
    s = _strings()
    t = m.TEMPLATE.read_text(encoding="utf-8")
    del s["fr"]["nav_why"]
    s["de"]["extra"] = "x"
    s["it"]["nav_try"] = " "
    bad = m.problems(t, s)
    assert "fr: missing key nav_why" in bad
    assert "de: key extra is not in en.json" in bad
    assert "it: empty value for nav_try" in bad
    with pytest.raises(ValueError):
        m.render("fr", t, s)


def test_an_attribute_key_may_not_hold_a_quote():
    s = _strings()
    s["en"]["nav_label_attr"] = 'Sec"tions'
    assert any("nav_label_attr" in p for p in m.problems(m.TEMPLATE.read_text(encoding="utf-8"), s))
