"""The websear.ch page carries one i18n dictionary per language; every language has the same keys."""
import re
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "webapp" / "index.html").read_text(encoding="utf-8")


def _keys(lang: str) -> set[str]:
    start = re.search(r"^  %s: \{$" % lang, HTML, re.M).end()
    end = re.search(r"^  \},?$", HTML[start:], re.M).start()
    return set(re.findall(r"^    ([A-Za-z0-9_]+):", HTML[start:start + end], re.M))


def test_every_language_has_the_same_i18n_keys():
    en = _keys("en")
    assert "footer_collab" in en and len(en) > 50
    for lang in ("fr", "de", "it"):
        assert _keys(lang) == en, lang


def test_page_carries_cytir_brand_and_partner_links():
    assert 'class="cytir-lockup"' in HTML
    for url in ("https://cytir.org", "https://impactia.org", "https://vector8.com"):
        assert url in HTML
