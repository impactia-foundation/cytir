"""Sitemap discovery for a freshness window (task 676).

The three daily-lane defects measured on 2026-09-30 (stage-2026-09-25/28):

* swissinfo: a global child budget of 200 was spent on the English seed's 272
  per-day children, so ger/fre/ita were never discovered;
* rts.ch: `www.rts.ch` never matched HOST_RULES' `rts.ch`, so its exclusion rules
  were skipped and all 609 sitemap URLs were fetched daily at Crawl-delay 10;
* SRF: monthly sitemaps named `YYYY_MM.xml` were all walked (357,399 URLs) for a
  one-month window.

The fixtures are the publishers' real index shapes (child URLs copied from the
live indexes fetched 2026-09-30).
"""
from __future__ import annotations

import datetime as dt
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("AW_OPERATOR_URL", "https://websear.ch")
os.environ.setdefault("AW_OPERATOR_CONTACT", "bender@impactia.org")

from cytir.acquire import sitemap_ingest as S  # noqa: E402

NOW = dt.date(2026, 9, 29)


def _swiss_index(lang: str, year: int, months_days) -> str:
    kids = "".join(
        f"<sitemap><loc>https://www.swissinfo.ch/{lang}/sitemap-{year}.xml?mm={m:02d}&amp;dd={d:02d}</loc></sitemap>"
        for m, d in months_days)
    return f'<?xml version="1.0"?><sitemapindex xmlns="x">{kids}</sitemapindex>'


def _urlset(*locs) -> str:
    body = "".join(f"<url><loc>{u}</loc><lastmod>2026-09-28</lastmod></url>" for u in locs)
    return f'<?xml version="1.0"?><urlset xmlns="x">{body}</urlset>'


def _days(year, start, end):
    d, out = start, []
    while d <= end:
        out.append((d.month, d.day))
        d += dt.timedelta(days=1)
    return out


def _fake_world(monkeypatch):
    """swissinfo as measured: each language has a 2026 index (Jan 1 .. Sep 29 = 272
    children, newest first) and a 2025 index (365 children)."""
    bodies, fetched = {}, []
    for lang in ("eng", "ger", "fre", "ita"):
        d26 = list(reversed(_days(2026, dt.date(2026, 1, 1), dt.date(2026, 9, 29))))
        d25 = list(reversed(_days(2025, dt.date(2025, 1, 1), dt.date(2025, 12, 31))))
        assert len(d26) == 272 and len(d25) == 365
        bodies[f"https://www.swissinfo.ch/{lang}/sitemap.xml"] = _swiss_index(lang, 2026, d26)
        bodies[f"https://www.swissinfo.ch/{lang}/sitemap-2025.xml"] = _swiss_index(lang, 2025, d25)
        for year, dl in ((2026, d26), (2025, d25)):
            for m, d in dl:
                bodies[f"https://www.swissinfo.ch/{lang}/sitemap-{year}.xml?mm={m:02d}&dd={d:02d}"] = \
                    _urlset(f"https://www.swissinfo.ch/{lang}/a-{year}{m:02d}{d:02d}")

    def fake_fetch(client, url, gbucket, hbucket, tries=3):
        fetched.append(url)
        return bodies.get(url)

    monkeypatch.setattr(S, "_fetch_body", fake_fetch)
    return fetched


SWISS_CFG = {"host": "www.swissinfo.ch", "sitemaps": [
    f"https://www.swissinfo.ch/{lang}/{name}" for lang in ("eng", "ger", "fre", "ita")
    for name in ("sitemap.xml", "sitemap-2025.xml")]}


def test_child_sitemap_end_reads_the_period_from_the_url():
    assert S.child_sitemap_end(
        "https://www.swissinfo.ch/eng/sitemap-2026.xml?mm=09&dd=29") == dt.date(2026, 9, 29)
    assert S.child_sitemap_end(
        "https://www.swissinfo.ch/eng/sitemap-2026.xml?mm=09&amp;dd=29") == dt.date(2026, 9, 29)
    assert S.child_sitemap_end(
        "https://www.srf.ch/sitemaps/aron/articles/2026_02.xml") == dt.date(2026, 2, 28)
    assert S.child_sitemap_end(
        "https://www.srf.ch/sitemaps/aron/articles/2025_12.xml") == dt.date(2025, 12, 31)
    assert S.child_sitemap_end("https://sitemap.lrc.swisstxt.ch/date/2026_09_29.xml") == dt.date(2026, 9, 29)
    assert S.child_sitemap_end("https://www.srf.ch/sitemaps/aron/landingpages.xml") is None
    assert S.child_sitemap_end("https://www.rsi.ch/generic-static/sitemap/archive/story/259.xml") is None
    assert S.child_sitemap_end("https://x.ch/sitemap-2026.xml?mm=13&dd=40") is None


def test_the_old_budget_finds_english_only(monkeypatch):
    """The defect, reproduced on the real shape: no window, budget 200."""
    _fake_world(monkeypatch)
    urls = S.discover_sitemap_urls(None, SWISS_CFG, "/nonexistent", None, None, max_children=200)
    langs = {u["url"].split("/")[3] for u in urls}
    assert langs == {"eng"}


def test_a_one_month_window_reaches_all_four_languages(monkeypatch):
    fetched = _fake_world(monkeypatch)
    start = S.window_cutoff_date(NOW, 1)
    assert start == dt.date(2026, 8, 29)
    urls = S.discover_sitemap_urls(None, SWISS_CFG, "/nonexistent", None, None,
                                   max_children=200, window_start=start)
    langs = {u["url"].split("/")[3] for u in urls}
    assert langs == {"eng", "ger", "fre", "ita"}
    # 32 days x 4 languages, nothing older, and no 2025 child was fetched
    days = {u["url"].rsplit("-", 1)[1] for u in urls}
    assert min(days) == "20260829" and max(days) == "20260929" and len(urls) == 32 * 4
    assert not [f for f in fetched if "sitemap-2025.xml?" in f]
    assert len(fetched) == 8 + 32 * 4, "8 seeds plus the in-window children only"


def test_a_pruned_child_does_not_spend_the_budget(monkeypatch):
    _fake_world(monkeypatch)
    urls = S.discover_sitemap_urls(None, SWISS_CFG, "/nonexistent", None, None,
                                   max_children=40, window_start=dt.date(2026, 8, 29))
    # 40 = 8 seeds' worth of nothing + 40 children: the budget bounds kept children only
    assert len(urls) == 40


def test_window_start_none_keeps_the_old_behaviour(monkeypatch):
    _fake_world(monkeypatch)
    a = S.discover_sitemap_urls(None, SWISS_CFG, "/nonexistent", None, None, max_children=60)
    b = S.discover_sitemap_urls(None, SWISS_CFG, "/nonexistent", None, None,
                                max_children=60, window_start=None)
    assert a == b


def test_select_recent_uses_the_shared_cutoff_arithmetic():
    assert S.window_cutoff_date(dt.date(2026, 3, 31), 1) == dt.date(2026, 2, 28)
    assert S.window_cutoff_date(dt.datetime(2026, 1, 15, 3), 3) == dt.date(2025, 10, 15)
    rows = [{"url": "https://www.swissinfo.ch/eng/x", "lastmod": "2026-08-30"},
            {"url": "https://www.swissinfo.ch/eng/y", "lastmod": "2026-08-01"}]
    got = S.select_recent(rows, {"host": "www.swissinfo.ch", "include": ()}, 10, None,
                          NOW, months=1)
    assert [r["url"] for r in got] == ["https://www.swissinfo.ch/eng/x"]


def test_rule_for_host_folds_www_so_rts_rules_apply():
    assert S.rule_for_host("www.rts.ch") is S.HOST_RULES["rts.ch"]
    assert S.rule_for_host("rts.ch") is S.HOST_RULES["rts.ch"]
    assert S.rule_for_host("www.srf.ch") == {}
    assert S.rule_for_host("WWW.RTS.CH") is S.HOST_RULES["rts.ch"]


def test_rts_exclusion_now_removes_the_real_hub_urls():
    """Real done_urls of www.rts.ch on 2026-09-28 (609 fetched, 0 survive)."""
    cfg = dict(S.rule_for_host("www.rts.ch"))
    cfg.setdefault("host", "www.rts.ch")
    real = [
        "https://www.rts.ch/education/", "https://www.rts.ch/info/monde/",
        "https://www.rts.ch/info/", "https://www.rts.ch/dossiers/elections/",
        "https://www.rts.ch/sport/football/", "https://www.rts.ch/archives/",
        "https://www.rts.ch/emissions/cqfd/", "https://www.rts.ch/meteo/"]
    article = "https://www.rts.ch/info/suisse/2026/article/une-loi-14812345.html"
    kept = S._filtered_urls([{"url": u, "lastmod": None} for u in real + [article]], cfg, None)
    assert [r["url"] for r in kept] == [article]


def test_srf_monthly_children_are_pruned_to_the_window(monkeypatch):
    months = [f"{y}_{m:02d}" for y in range(2012, 2027) for m in range(1, 13)][:-3]
    idx = "".join(f"<sitemap><loc>https://www.srf.ch/sitemaps/aron/articles/{m}.xml</loc></sitemap>"
                  for m in months)
    idx += "<sitemap><loc>https://www.srf.ch/sitemaps/aron/landingpages.xml</loc></sitemap>"
    bodies = {"https://www.srf.ch/sitemap/index/srf": f"<sitemapindex>{idx}</sitemapindex>"}
    fetched = []

    def fake(client, url, gb, hb, tries=3):
        fetched.append(url)
        return bodies.get(url, "<urlset><url><loc>https://www.srf.ch/news/x-" + url[-11:-4] + "</loc></url></urlset>")

    monkeypatch.setattr(S, "_fetch_body", fake)
    urls = S.discover_sitemap_urls(None, {"host": "www.srf.ch", "sitemaps": ["https://www.srf.ch/sitemap/index/srf"]},
                                   "/x", None, None, max_children=200,
                                   window_start=dt.date(2026, 8, 29))
    kids = [f for f in fetched if "articles/" in f]
    assert all(k.endswith(("2026_08.xml", "2026_09.xml")) for k in kids) or kids == []
    assert len(fetched) < 6, "the run must not walk 174 monthly files for a one-month window"
    assert any(u["url"].startswith("https://www.srf.ch/news/x-") for u in urls)
