"""watson.ch 12-month backfill: weekly-sitemap discovery, monthly runs, real-page extraction.

Fixtures (tests/fixtures/watson, fetched from watson.ch from the AW crawl host on 2026-10-03, robots allow, Crawl-delay
1 s): the weekly urlsets of ISO week 2026-W36 (DE: 590 urls, FR: 395; the files hold the first 30 / 15 entries plus one
`videos`, `vermarktung`, `commercialisation` entry each) and three article pages (two DE, one FR).
"""
import datetime as dt
import gzip
import json
import pathlib

import pytest

from cytir.news import coverage_backfill as B
from cytir.news import publishers as REG
from cytir.news import watson_backfill as W

FX = pathlib.Path(__file__).resolve().parent / "fixtures" / "watson"
TODAY = dt.date(2026, 10, 3)
SITEMAP_DE = (FX / "sitemap-de-2026w36-trimmed.xml").read_text(encoding="utf-8")
SITEMAP_FR = (FX / "sitemap-fr-2026w36-trimmed.xml").read_text(encoding="utf-8")


def _weekly(week: str, lang: str, day: dt.date) -> str:
    """A tiny synthetic urlset for a week other than the fixture's: three articles dated `day`."""
    pre = "/fr" if lang == "fr" else ""
    body = "".join(f"<url><loc>https://www.watson.ch{pre}/schweiz/{week}{1 if lang == 'de' else 2}{i}00-a-{i}</loc>"
                   f"<lastmod>{day.isoformat()}T10:00:00+02:00</lastmod></url>" for i in range(3))
    return f'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{body}</urlset>'


def _fake_fetch(unclean=None):
    calls = []

    def fetch(url):
        calls.append(url)
        if unclean and unclean in url:
            return 503, ""
        lang = "fr" if "/fr/" in url else "de"
        week = url.split("date=")[1]
        if week == "202636":
            return 200, SITEMAP_FR if lang == "fr" else SITEMAP_DE
        y, w = int(week[:4]), int(week[4:])
        return 200, _weekly(week, lang, dt.date.fromisocalendar(y, w, 3))
    return fetch, calls


# ------------------------------------------------------------------ sitemap shape
def test_the_real_weekly_urlsets_parse_and_classify():
    de = W.parse_urlset(SITEMAP_DE)
    assert len(de) >= 30 and all(m.startswith("2026-0") for _, m in de)
    kinds = {}
    for url, _ in de:
        kinds[W.classify(url)] = kinds.get(W.classify(url), 0) + 1
    assert kinds["article"] >= 25
    assert any(k.startswith("section_") for k in kinds), "a videos or marketing entry is left out, by name"
    fr = W.parse_urlset(SITEMAP_FR)
    assert all(u.startswith("https://www.watson.ch/fr/") for u, _ in fr)
    assert W.classify("https://www.watson.ch/fr/suisse/economie/928328363-la-patronne") == "article"
    assert W.classify("https://www.watson.ch/fr/commercialisation/123456789-x") == "section_commercialisation"
    assert W.classify("https://www.watson.ch/videos/deutschland/838953387-x") == "section_videos"
    assert W.classify("https://www.watson.ch/schweiz/") == "not_article_shaped"
    assert W.classify("https://example.com/a/123456789-x") == "foreign_host"


def test_iso_weeks_of_the_window_cross_the_year_boundary():
    weeks = W.weeks(dt.date(2025, 10, 3), TODAY)
    assert weeks[0] == "202540" and weeks[-1] == "202640" and len(weeks) == 53
    assert "202601" in weeks and "202552" in weeks
    assert len(W.seeds(TODAY)) == 106
    assert W.seeds(TODAY)[0][2] == "https://www.watson.ch/sitemap.xml?date=202540"
    assert W.seeds(TODAY)[1][2] == "https://www.watson.ch/fr/sitemap.xml?date=202540"


# ------------------------------------------------------------------ discovery
def test_discovery_reads_every_week_once_and_resumes(tmp_path):
    fetch, calls = _fake_fetch()
    acct = W.discover(tmp_path, fetch, TODAY)
    assert acct["clean"] and acct["seeds"] == 106 == len(calls) and acct["terminals"] == {"ok": 106}
    assert acct["left_out"], "left-out urls are counted by reason, not dropped silently"
    rows = [json.loads(x) for x in (tmp_path / "urls.jsonl").read_text().splitlines()]
    assert len({r["url"] for r in rows}) == len(rows) > 300
    assert {r["lang"] for r in rows} == {"de", "fr"} and all(r["day"] for r in rows)
    fetch2, calls2 = _fake_fetch()
    assert W.discover(tmp_path, fetch2, TODAY)["clean"] and calls2 == [], "ok seeds are not fetched again"


def test_an_unreadable_week_is_an_error_terminal_not_an_empty_week(tmp_path):
    fetch, _ = _fake_fetch(unclean="date=202601")
    acct = W.discover(tmp_path, fetch, TODAY)
    assert not acct["clean"] and acct["terminals"]["http_503"] == 2
    d = tmp_path.parent / "state-unclean"
    (d / "discovery").mkdir(parents=True)
    for name in ("state.json", "urls.jsonl"):
        (d / "discovery" / name).write_text((tmp_path / name).read_text())
    with pytest.raises(SystemExit, match="not clean"):
        W.runs(d, smoke=False, make_run=lambda **k: k)


# ------------------------------------------------------------------ the monthly runs
def _discovered(tmp_path):
    fetch, _ = _fake_fetch()
    W.discover(tmp_path / "discovery", fetch, TODAY)
    return tmp_path


def test_twelve_monthly_runs_on_their_own_free_days(tmp_path):
    state_dir = _discovered(tmp_path)
    runs = B.plan(["watson"], today=TODAY, state_dir=state_dir)
    assert [r.month for r in runs] == [f"2025-{m:02d}" for m in (10, 11, 12)] + [f"2026-{m:02d}" for m in range(1, 10)]
    assert len(runs) == 12 and len({r.as_of for r in runs}) == 12
    assert runs[0].as_of == "2026-06-16" and runs[-1].as_of == "2026-06-27"
    forbidden = {"2026-06-15", "2026-06-29", "2026-06-30"}  # SRF 15th, the smoke day, swissinfo's month end
    assert not forbidden & {r.as_of for r in runs}
    assert all(r.as_of < "2026-07-21" for r in runs), "never a day of the daily timer's territory"
    for r in runs:
        assert r.publisher == "watson" and r.cap == 8000 and r.host_rate == 0.5 and r.urls and not r.seeds
        assert 1.0 / r.host_rate >= 2.0, "Crawl-delay at least 2 s"
        row = r.queue()[0]
        assert row["domain"] == "www.watson.ch" and row["urls"] == r.urls and row["cap"] == 8000 and "sitemaps" not in row
    assert runs[0].urls[0].startswith("https://www.watson.ch")
    # the as-of day belongs to the MONTH, so it does not move when discovery is repeated later
    assert W.as_of_for("2026-03") == "2026-06-21"


def test_the_current_month_joins_the_last_full_month(tmp_path):
    rows = [{"url": "u1", "day": "2026-09-30"}, {"url": "u2", "day": "2026-10-02"}, {"url": "u3", "day": ""},
            {"url": "u4", "day": "2025-09-01"}, {"url": "u5", "day": "2025-10-03"}]
    buckets, left = W.month_buckets(rows, "2025-10-03", TODAY)
    assert buckets == {"2025-10": ["u5"], "2026-09": ["u1", "u2"]}
    assert left == {"urls": 5, "no_day": 1, "before_window": 1, "bucketed": 3}


def test_a_month_over_the_row_cap_refuses_rather_than_truncates(tmp_path, monkeypatch):
    state_dir = _discovered(tmp_path)
    monkeypatch.setattr(W, "CAP", 5)
    with pytest.raises(SystemExit, match="exceed the run cap"):
        B.plan(["watson"], today=TODAY, state_dir=state_dir)


def test_the_smoke_run_is_tiny_and_on_a_day_no_month_uses(tmp_path):
    state_dir = _discovered(tmp_path)
    (run,) = B.plan(["watson"], today=TODAY, smoke=True, state_dir=state_dir)
    assert run.smoke and run.as_of == W.SMOKE_ASOF == "2026-06-29" and len(run.urls) <= 25 and run.cap == 25
    assert run.as_of not in {W.as_of_for(f"2025-{m}") for m in (10, 11, 12)}


def test_an_executed_run_names_only_watson_as_the_expected_host(tmp_path, monkeypatch):
    # the box paths (spool, queue) are settings a stranger configures; this test needs none of them
    monkeypatch.setattr(B, "spool_payload_bytes", lambda *a, **k: 0)
    monkeypatch.setattr(B, "pack_manifest", lambda *a, **k: None)
    monkeypatch.setattr(B.D, "staged_points", lambda day: {"total": 0})
    state_dir = _discovered(tmp_path)
    (run,) = B.plan(["watson"], today=TODAY, smoke=True, state_dir=state_dir)
    seen = {}

    def runner(day, args):
        import os
        seen["env"] = os.environ.get(B.EXPECTED_HOSTS_ENV)
        seen["cap"] = os.environ.get("AW_RECENT_CAP")
        seen["host_rate"] = args.host_rate
    entry = B.execute(run, state_dir=state_dir, runner=runner, iface_bytes=lambda iface: 0)
    assert entry["status"] == "done" and seen == {"env": "www.watson.ch", "cap": "25", "host_rate": 0.5}
    assert "www.watson.ch" in REG.hosts() and "www.watson.ch" not in REG.hosts(daily=True)


# ------------------------------------------------------------------ real pages
def _page(name):
    with gzip.open(FX / f"article-{name}.html.gz", "rt", encoding="utf-8", errors="replace") as fh:
        return fh.read(), (FX / f"article-{name}.url").read_text().strip()


@pytest.mark.parametrize("name,lang,title_start", [
    ("de1", "de", "Seco-Chefin warnt"), ("de2", "de", "Swiss Re sieht KI-Rechenzentren"),
    ("fr1", "fr", "La patronne du Seco")])
def test_real_articles_give_title_page_date_language_and_passages(monkeypatch, name, lang, title_start):
    pytest.importorskip("trafilatura")
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")
    monkeypatch.setenv("AW_ALLOW_UNIDENTIFIED_CRAWL", "1")
    from cytir.core.common import process_html
    from cytir.news.extract_published import published_for_page
    html, url = _page(name)
    assert published_for_page(html, url) == "2026-09-06", "the page's own date, with no sitemap date given"
    passages, _ = process_html(url, html, {"content": True, "links": False, "lang_default": "de",
                                           "passage_common": {"domain": "www.watson.ch",
                                                              "source_type": "news_article",
                                                              "tier": "C", "trust_tier": 0.6}})
    assert len(passages) >= 1
    p = passages[0]
    assert p.title.startswith(title_start) and p.lang == lang and p.url == url
    assert len(p.text) > 200
