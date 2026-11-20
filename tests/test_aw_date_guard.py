"""Task 760: a page's date exception must never fail the page's DOMAIN.

Each fixture under tests/fixtures/t760_date_pages/ is the offending region of a
REAL page from CC tranches 7-12 (read from the build box L0 store, trimmed; the
prose paragraphs are made up). Before the fix `trafilatura.extract_metadata`
raised out of `process_html` and the domain was marked `error`.
"""
import pathlib

import pytest

import cytir.core.common as AC  # noqa

@pytest.fixture(autouse=True)
def _langgate_optional(monkeypatch):
    # language detection is not under test (no fastText model needed in CI)
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")


FIX = pathlib.Path(__file__).parent / "fixtures" / "t760_date_pages"

# fixture -> (the real URL, the exception htmldate raised on the real page)
DATE_PAGES = {
    "surf": ("https://www.surf.nl/agenda?page=3", "ValueError"),
    "nibsc": ("https://nibsc.org/science_and_research/publications.aspx", "ValueError"),
    "ucsd": ("https://pages.ucsd.edu/~aronatas/soc20602.html", "ValueError"),
    "hnb": ("https://www.hnb.hr/pretraga?q=act+on+Croatian&modifiedFrom=2026-07-10&modifiedTo=2026-07-11", "ValueError"),
    "skillscompetencescanada": ("https://www.skillscompetencescanada.com/en/calendars/", "ValueError"),
    "erichorvitz": ("https://www.erichorvitz.com/abstracts.htm", "ValueError"),
    "sgk": ("https://www.sgk.gov.tr/Arama/Index/sut", "KeyError"),
    "ieu": ("https://www.ieu.edu.tr/tr/news/type/read/id/140", "KeyError"),
    "karatekin": ("https://www.karatekin.edu.tr/tum-etkinlikler", "KeyError"),
}
CTX = {"content": True, "links": False, "want_stats": True,
       "passage_common": {"domain": "example.org", **AC.tier_stamps("B")}, "lang_default": "en"}


def _html(name):
    return (FIX / f"{name}.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", sorted(DATE_PAGES))
def test_fixture_still_trips_the_library(name):
    """Pins the cause: the raw library call raises. If a trafilatura/htmldate
    upgrade fixes it, this skips (the guard stays correct either way)."""
    import trafilatura
    try:
        trafilatura.extract_metadata(_html(name))
    except Exception as e:  # noqa: BLE001
        assert type(e).__name__ == DATE_PAGES[name][1]
    else:
        pytest.skip("installed htmldate no longer raises on this page")


@pytest.mark.parametrize("name", sorted(DATE_PAGES))
def test_dateguard_returns_metadata_without_date(name):
    meta, err = AC.extract_metadata_dateguard(_html(name))
    if err is None:
        pytest.skip("installed htmldate no longer raises on this page")
    assert meta is not None and meta.date is None
    assert err.startswith(DATE_PAGES[name][1])
    assert meta.title            # the rest of the metadata survives


@pytest.mark.parametrize("name", sorted(DATE_PAGES))
def test_process_html_survives_and_counts(name):
    url = DATE_PAGES[name][0]
    passages, _links, stats = AC.process_html(url, _html(name), dict(CTX))
    rej = stats["extraction_rejects"]
    assert passages, f"{name}: page must still yield rows"
    assert rej.get(AC.DATE_PARSE_FAILED, 0) in (0, 1)   # 0 only if upstream fixed it


def test_dateguard_is_identity_on_a_clean_page():
    import trafilatura
    html = ("<html><head><title>Plain page</title></head><body><p>"
            + "A plain sentence with no date at all. " * 20 + "</p></body></html>")
    meta, err = AC.extract_metadata_dateguard(html)
    direct = trafilatura.extract_metadata(html)
    assert err is None
    keep = ("title", "author", "sitename", "date", "description", "url", "hostname")
    assert {k: getattr(meta, k) for k in keep} == {k: getattr(direct, k) for k in keep}


def test_dateguard_restores_find_date():
    from trafilatura import metadata as tm
    real = tm.find_date
    AC.extract_metadata_dateguard(_html("surf"))
    assert tm.find_date is real


def test_clean_page_process_html_is_unchanged_and_uncounted():
    html = ("<html><head><title>Plain page</title></head><body><main><p>"
            + "A plain sentence about nothing in particular, written for a test. " * 12
            + "</p></main></body></html>")
    _p, _l, stats = AC.process_html("https://example.org/a", html, dict(CTX))
    assert AC.DATE_PARSE_FAILED not in stats["extraction_rejects"]


# --- dict-valued JSON-LD title (fibe-berlin.com) ---------------------------

def test_coerce_title_str_is_untouched():
    assert AC.coerce_title("  Keep  me ") == "  Keep  me "


@pytest.mark.parametrize("value,expected", [
    ({"title": "Favourite Spots in Berlin  "}, "Favourite Spots in Berlin"),
    ({"name": {"text": "A  B"}}, "A B"),
    ([{"headline": "First"}, "Second"], "First"),
    ({"author": {"name": None}}, None),
    (None, None), (7, None), ({}, None),
])
def test_coerce_title_structured(value, expected):
    assert AC.coerce_title(value) == expected


def test_fibe_dict_title_page_yields_rows():
    url = "https://www.fibe-berlin.com/en/newsroom/blog-detail_2432.html"
    passages, _l, _s = AC.process_html(url, _html("fibe"), dict(CTX))
    assert passages
    assert all(isinstance(p.title, str) for p in passages)
    assert passages[0].title.startswith("Favourite Spots in Berlin")


# --- the DOMAIN guard: one page's date exception never fails the domain -----

def test_a_date_page_does_not_fail_its_domain():
    from contextlib import redirect_stdout
    import io
    from unittest import mock
    from cytir.acquire import discovery_ingest as di

    clean = ("<html><head><title>Plain page</title></head><body><main><p>"
             + "A plain sentence about nothing in particular, written for a test. " * 12
             + "</p></main></body></html>")
    pages = {"https://x.example/agenda": _html("surf"), "https://x.example/plain": clean,
             "https://x.example/tr": _html("ieu")}
    recs = [{"url": u, "offset": 0, "length": 1, "warc_filename": "f", "digest": None}
            for u in pages]

    def inline_extract(url, html, meta):      # the pool worker, in-process
        p, links, st = AC.process_html(url, html, {**meta, "want_stats": True})
        return p, links, st["extraction_rejects"], False

    with mock.patch.object(di, "fetch_warc", side_effect=lambda c, r: pages[r["url"]]), \
            mock.patch.object(di, "_extract_in_pool", side_effect=inline_extract), \
            mock.patch.object(di.l0_store, "l0_enabled", return_value=False), \
            redirect_stdout(io.StringIO()):
        dom, rows, _links, n_pages, status, rejects = di.process_domain(
            None, {"domain": "x.example"}, di.DEFAULT_CRAWL, 10, preresolved=recs)
    assert status == "ok" and n_pages == 3
    assert {r["url"] for r in rows} == set(pages)           # every page produced rows
    assert rejects.get(AC.DATE_PARSE_FAILED) == 2           # counted, not fatal


def test_news_publisher_metadata_keeps_headline_when_the_date_stage_raises():
    """sitemap_ingest._publisher_metadata used to drop headline AND byline with it (meta=None)."""
    from cytir.acquire import sitemap_ingest as si
    headline, _byline, published = si._publisher_metadata(_html("surf"))
    assert headline == "Agenda" and published is None


def test_news_publisher_metadata_reads_a_dict_title():
    from cytir.acquire import sitemap_ingest as si
    headline, _b, _p = si._publisher_metadata(_html("fibe"))
    assert headline == "Favourite Spots in Berlin"
