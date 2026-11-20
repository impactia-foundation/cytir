"""Task 2009: site-search result URLs are rejected at the projection choke point."""
import pytest

from cytir.core import projection
from cytir.core import search_result_url as sru

REJECTED = [
    "https://fintechnews.ch/?s=postfinance",
    "https://example.ch/blog/archive?s=zins",
    "https://example.ch/search?q=mietrecht",
    "https://example.ch/suche?query=kuendigung",
    "https://example.ch/recherche?recherche=loi",
    "https://example.ch/de/search/?k=steuern",
    "https://example.ch/?search=abc",
    "https://de.wikipedia.org/w/index.php?search=Zürich",
    "https://example.ch/index.php?search=steuern",
    "https://x.ch/de/suche/?q=a",
    "https://x.ch/suche/?query=miete",
]

ADMITTED = [
    "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_336",
    "https://www.rfc-editor.org/rfc/rfc9110",
    "https://example.ch/",
    "https://example.ch/artikel?page=2",
    "https://api.example.ch/v1/items?q=abc",
    "https://example.ch/news/suche-nach-wahrheit",
    "https://example.ch/?lang=de",
    "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_257",
    "https://www.rfc-editor.org/rfc/rfc7519.txt",
    "https://api.github.com/search/repositories?q=x",
    "https://www.admin.ch/gov/de/start/dokumentation/medienmitteilungen.msg-id-1.html?k=foo",
    "https://example.ch/index.php?id=12",
    "https://example.ch/de/suche",
    "https://example.ch/de/suchmaschine/?q=x",
]


@pytest.mark.parametrize("url", REJECTED)
def test_rejected(url):
    assert sru.is_search_result_url(url)


@pytest.mark.parametrize("url", ADMITTED)
def test_admitted(url):
    assert not sru.is_search_result_url(url)


def test_degenerate_inputs():
    assert not sru.is_search_result_url(None)
    assert not sru.is_search_result_url("")


def _row(url):
    return {"url": url, "text": "Ein Absatz mit genug Text.", "title": "T",
            "source_type": "web", "domain": "example.ch"}


def test_projection_rejects_and_counts():
    sru.reset_search_result_reject_count()
    assert projection.project_record(_row("https://fintechnews.ch/?s=postfinance")) is None
    assert sru.search_result_reject_count() == 1


def test_projection_admits_normal_url():
    sru.reset_search_result_reject_count()
    out = projection.project_record(_row("https://example.ch/artikel"))
    assert out is not None
    assert sru.search_result_reject_count() == 0
