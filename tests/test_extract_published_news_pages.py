"""Task 448 — published-date extraction for the srf.ch / rts.ch source feed.

850 of the 2026-08-27 news pack's 6,357 rows (every srf.ch and rts.ch row) had
no `published`. Neither publisher's sitemap carries `<lastmod>`, and the sitemap
date was the only signal `sitemap_ingest --freshness` ever read — the page's own
HTML carried the date the whole time.

Every HTML fragment below is COPIED VERBATIM out of
`stage-2026-08-27/raw/*.raw.jsonl.gz` of the news-freshness work tree on the
operator build box (the preserved raw store for that exact pack), trimmed to the date
signals. The URLs are named so the sample can be re-pulled and re-checked.
"""
from __future__ import annotations

import datetime as dt

import pytest

from cytir.news.extract_published import extract_published, published_for_page
from tests.helpers.repo_files import repo_file_or_skip

NOW = dt.date(2026, 8, 27)

# --- verbatim raw-store fragments --------------------------------------------

# https://www.srf.ch/news/international/sturzflut-in-nepal-und-tibet-forscher-
#   wahrscheinlich-ein-historisch-grosser-felssturz
# SRF articles carry schema.org microdata in <head>: an itemprop datePublished
# next to a dateModified. This is the shape 162 of the 259 undated srf.ch URLs
# in the pack have.
SRF_ARTICLE = """<html><head>
<meta name="i18n:date:today" content="Heute">
<meta name="i18n:status:updated" content="Aktualisiert">
<meta itemprop="datePublished" content="2026-08-27T07:13:03+02:00">
<meta itemprop="dateModified" content="2026-08-27T07:13:03+02:00">
<meta itemprop="uploadDate" content="2026-08-27T06:00:00+02:00">
</head><body><p>Sturzflut in Nepal und Tibet.</p></body></html>"""

# https://www.srf.ch/news/schweiz/zukunftsforschung-fuer-natur-gegen-ki-schweizer-
#   volk-ist-sich-bei-vielem-einig  — same shape, a different day/time.
SRF_ARTICLE_2 = """<html><head>
<meta itemprop="datePublished" content="2026-08-27T06:47:28+02:00">
<meta itemprop="dateModified" content="2026-08-27T06:47:28+02:00">
</head><body><p>Zukunftsforschung.</p></body></html>"""

# https://www.rts.ch/education/a-propos/ — RTS pages carry a schema.org
# NewsArticle JSON-LD block AND an OpenGraph article:published_time. Note the
# JSON-LD dateModified is eight years later than datePublished: the extractor
# must take the PUBLICATION date, which is what the news window is about.
RTS_ARTICLE = """<html><head>
<meta property="article:published_time" content="2017-09-11T11:38:13Z">
<script type="application/ld+json">
{"@context":"http://schema.org","@type":"NewsArticle",
 "mainEntityOfPage":"https://www.rts.ch/education/2017/article/a-propos-27341261.html",
 "headline":"\\u00c0 propos",
 "datePublished":"2017-09-11T13:38:13+02:00",
 "dateModified":"2025-09-01T13:56:22+02:00",
 "inLanguage":"fr","articleSection":"D\\u00e9couverte"}
</script></head><body><p>A propos.</p></body></html>"""

# https://www.rts.ch/archives/ — a section hub. 107 of rts.ch's 109 undated
# URLs in the pack look like this: no article date anywhere, because there is
# no article. These MUST stay dateless.
RTS_HUB = """<html><head>
<meta property="og:title" content="Archives">
<meta property="og:type" content="website">
</head><body><h1>Archives</h1><p>Les archives de la RTS.</p></body></html>"""

# https://www.srf.ch/meteo/warnungen — same class on the srf.ch side: a utility
# page with no publication event. 97 of srf.ch's 259 undated URLs are these.
SRF_HUB = """<html><head>
<meta name="i18n:date:now" content="Jetzt">
<meta name="i18n:date:today" content="Heute">
</head><body><h1>Warnungen</h1></body></html>"""


# --- extraction --------------------------------------------------------------

def test_srf_article_microdata_date_published_is_read():
    assert extract_published(SRF_ARTICLE, url="https://www.srf.ch/news/international/x",
                             now_ref=NOW, use_htmldate=False) == "2026-08-27"
    assert extract_published(SRF_ARTICLE_2, now_ref=NOW,
                             use_htmldate=False) == "2026-08-27"


def test_rts_jsonld_publication_date_wins_over_its_modification_date():
    assert extract_published(RTS_ARTICLE, url="https://www.rts.ch/education/a-propos",
                             now_ref=NOW, use_htmldate=False) == "2017-09-11"


@pytest.mark.parametrize("html,label", [(RTS_HUB, "rts hub"), (SRF_HUB, "srf hub")])
def test_a_page_with_no_publication_event_stays_dateless(html, label):
    """Absent beats invented: a section/utility page must not acquire a date."""
    assert extract_published(html, now_ref=NOW, use_htmldate=False) is None


def test_the_fetch_date_is_never_substituted_for_a_missing_publication_date():
    assert published_for_page(RTS_HUB, url="https://www.rts.ch/archives",
                              now_ref=NOW) is None
    assert published_for_page("", url="https://www.srf.ch/x", now_ref=NOW) is None


# --- sanity bounds -----------------------------------------------------------

def test_a_far_future_date_is_rejected_rather_than_stamped():
    html = '<meta itemprop="datePublished" content="2031-04-02T09:00:00+02:00">'
    assert extract_published(html, now_ref=NOW, use_htmldate=False) is None


def test_tomorrow_is_allowed_for_timezone_skew_but_not_the_day_after():
    ok = '<meta itemprop="datePublished" content="2026-08-28T23:30:00+02:00">'
    too_far = '<meta itemprop="datePublished" content="2026-08-29T00:30:00+02:00">'
    assert extract_published(ok, now_ref=NOW, use_htmldate=False) == "2026-08-28"
    assert extract_published(too_far, now_ref=NOW, use_htmldate=False) is None


def test_a_pre_corpus_date_is_rejected_as_a_misparse():
    html = '<meta itemprop="datePublished" content="1970-01-01T00:00:00Z">'
    assert extract_published(html, now_ref=NOW, use_htmldate=False) is None


def test_an_unparseable_date_string_yields_nothing_not_a_partial_guess():
    html = '<meta itemprop="datePublished" content="letzten Donnerstag">'
    assert extract_published(html, now_ref=NOW, use_htmldate=False) is None


# --- the page-level precedence the ingest applies ----------------------------

def test_the_pages_own_date_wins_over_the_sitemap_lastmod():
    """Tim 2026-09-30 (plan.md ruling 18): `<lastmod>` is a modification date; the
    page's own date wins where both exist."""
    assert published_for_page(SRF_ARTICLE, url="https://www.srf.ch/news/x",
                              sitemap_published="2026-08-21",
                              now_ref=NOW) == "2026-08-27"


def test_the_sitemap_date_is_the_fallback_when_the_page_names_no_date():
    assert published_for_page(RTS_HUB, url="https://www.rts.ch/archives",
                              sitemap_published="2026-08-21",
                              now_ref=NOW) == "2026-08-21"
    assert published_for_page("", url="https://www.swissinfo.ch/eng/x",
                              sitemap_published="2026-08-21T10:00:00Z",
                              now_ref=NOW) == "2026-08-21"


def test_the_page_html_alone_still_dates_a_page_with_no_sitemap_date():
    assert published_for_page(SRF_ARTICLE, url="https://www.srf.ch/news/x",
                              sitemap_published=None,
                              now_ref=NOW) == "2026-08-27"
    assert published_for_page(RTS_ARTICLE, url="https://www.rts.ch/x",
                              sitemap_published="",
                              now_ref=NOW) == "2017-09-11"


def test_an_out_of_range_sitemap_date_falls_through_to_the_page():
    """A junk <lastmod> is not a reason to refuse the page's own good date."""
    assert published_for_page(SRF_ARTICLE, sitemap_published="0001-01-01",
                              now_ref=NOW) == "2026-08-27"


def test_page_resolution_does_not_depend_on_an_optional_package():
    """`published_for_page` pins use_htmldate=False: the same bytes must yield
    the same date on a box with htmldate installed and one without."""
    import inspect
    sig = inspect.signature(published_for_page)
    assert sig.parameters["use_htmldate"].default is False


# --- real swissinfo / RSI / SRF pages (task 676, plan.md ruling 18) -------------
# 13 pages read from the sealed originals of the 2026-09-28 daily stage on the operator build box
# (`stage-2026-09-28/pages/original`), with the `published` the pipeline stamped on
# their rows at the time (the sitemap `lastmod`, for swissinfo and RSI). Each page is
# stored as its `<head>` plus the `<time>`/`<meta>`/JSON-LD tags of its body: the only
# signals the extractor reads, proved equal to the full page's answer at capture.

import gzip
import json
import pathlib

_FIXTURE = (pathlib.Path(__file__).resolve().parent.parent / "tasks" / "676-artifacts"
            / "page-date-rows-fixture.jsonl.gz")
repo_file_or_skip(_FIXTURE)


def _real_pages():
    with gzip.open(_FIXTURE, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_the_real_pages_cover_the_cases_the_ruling_distinguishes():
    kinds = {}
    for r in _real_pages():
        kinds.setdefault((r["domain"], r["kind"]), 0)
        kinds[(r["domain"], r["kind"])] += 1
    assert kinds == {("swissinfo.ch", "same"): 2, ("swissinfo.ch", "pos"): 3,
                     ("swissinfo.ch", "neg"): 2, ("rsi.ch", "pos"): 2,
                     ("srf.ch", "same"): 2, ("srf.ch", "none"): 2}


def test_on_real_pages_the_page_date_replaces_the_sitemap_lastmod():
    changed = 0
    for r in _real_pages():
        now = dt.date.fromisoformat(r["as_of"])
        got = published_for_page(r["html_signals"], url=r["url"],
                                 sitemap_published=r["row_published"], now_ref=now)
        if r["kind"] == "none":
            # the page names no date: the sitemap date (or nothing) stays
            assert got == (r["row_published"] or None), r["url"]
        else:
            assert got == r["page_date"], r["url"]
            changed += got != r["row_published"]
    assert changed == 7, "3+2 swissinfo and 2 rsi rows move; 'same' rows and 'none' rows do not"


def test_the_old_precedence_would_have_kept_every_lastmod_on_those_pages():
    """The defect this ruling removes, on the same real pages: with the sitemap date
    winning, all seven pages whose own date differs keep the modification date."""
    for r in _real_pages():
        if r["kind"] in ("pos", "neg"):
            assert r["row_published"] != r["page_date"], r["url"]


def test_the_page_date_ignores_the_sitemap_date_even_when_both_are_readable():
    rows = [r for r in _real_pages() if r["kind"] in ("pos", "neg")]
    assert len(rows) == 7
    for r in rows:
        now = dt.date.fromisoformat(r["as_of"])
        with_sitemap = published_for_page(r["html_signals"], url=r["url"],
                                          sitemap_published=r["row_published"], now_ref=now)
        without = published_for_page(r["html_signals"], url=r["url"],
                                     sitemap_published=None, now_ref=now)
        assert with_sitemap == without == r["page_date"]


# --- page-date clamp is the real day, not the as-of day (task 676, 2026-09-30) ----

def test_page_date_clamp_uses_the_real_day_so_backfill_keeps_recent_page_dates():
    import datetime as dt
    from cytir.acquire import sitemap_ingest as si
    asof = dt.date(2026, 7, 18)           # a free past as-of day of a backfill run
    today = dt.date(2026, 9, 30)
    assert si.page_date_clamp_day(asof, today=today) == today
    # the old clamp (as-of + 1 day) dropped a real page date of 2026-09-25 ...
    assert published_for_page(SRF_ARTICLE, now_ref=asof) is None
    # ... the real day keeps it, and a far-future mis-parse is still refused
    html = '<meta itemprop="datePublished" content="2026-09-25T08:00:00+02:00">'
    assert published_for_page(html, now_ref=si.page_date_clamp_day(asof, today=today)) == "2026-09-25"
    html = '<meta itemprop="datePublished" content="2027-01-05">'
    assert published_for_page(html, now_ref=si.page_date_clamp_day(asof, today=today)) is None
    # an as-of day in the future of the real day is never lowered
    assert si.page_date_clamp_day(dt.date(2026, 10, 5), today=today) == dt.date(2026, 10, 5)
