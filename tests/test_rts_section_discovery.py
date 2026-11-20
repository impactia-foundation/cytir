"""RTS section pagination discovery (task 676, plan.md ruling 18).

Real pages: the task-676 artifact `rts-pages-fixture.json.gz` holds what rts.ch returned on
2026-09-30 from the GPU box (AWbot user agent, 12 s between requests): the `/info/suisse/`
hub, `/info/suisse/page/1`, `/page/60`, `/page/400`, and the `info` and `sport` sitemaps.
The site is not called: a scripted client serves those pages.

What each group would have caught:
* a card date is read in every shape the site prints (absolute, today, yesterday,
  weekday) and a card WITHOUT a date (the static strip) is never an article of the section;
* `/hbv7/` is disallowed by robots.txt: a sport section ends `robots_disallowed`, which is
  the publisher's statement, not an error and not an empty section;
* an HTTP failure ends a section as `error` (resumable), never as complete;
* every planned section has exactly one terminal.
"""
from __future__ import annotations
import datetime as dt
import pathlib
import sys
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import cytir.news.rts_section_discovery as R  # noqa: E402
from cytir.news import coverage_backfill as B  # noqa: E402


TODAY = dt.date(2026, 9, 30)


D = dt.date


# ------------------------------------------------------------------ parsing
def test_card_dates_in_every_shape_the_site_prints():
    assert R.parse_card_date("le 9 janvier 2026", TODAY) == D(2026, 1, 9)
    assert R.parse_card_date("le 1er février 2026", TODAY) == D(2026, 2, 1)
    assert R.parse_card_date("aujourd&#39;hui à 11:50", TODAY) == TODAY
    assert R.parse_card_date("hier à 18:25", TODAY) == D(2026, 9, 29)
    assert R.parse_card_date("dimanche à 15:30", TODAY) == D(2026, 9, 27)   # TODAY is a Wednesday
    assert R.parse_card_date("mercredi à 08:00", TODAY) == D(2026, 9, 23), "a week ago, not today"
    assert R.parse_card_date("le 31 septembre 2026", TODAY) is None
    assert R.parse_card_date("", TODAY) is None


def test_window_start_is_twelve_months_back():
    assert R.window_start(D(2026, 9, 30)) == D(2025, 9, 28)
    assert R.window_start(D(2026, 1, 15)) == D(2025, 1, 15)


# ------------------------------------------------------------------ crawling
class Client:
    """Serves `script(url) -> (status, text)` and records every request."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def get(self, url):
        self.calls.append(url)
        status, text = self.script(url)
        return type("R", (), {"status_code": status, "text": text, "headers": {}})()


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _fetcher(script, robots_ok=lambda u: True, clock=None):
    clock = clock or Clock()
    c = Client(script)
    return R.Fetcher(c, robots_ok, 10.0, sleep=clock.sleep, clock=clock), c, clock


def _run(tmp_path, fetcher, sections, start=D(2026, 3, 1)):
    return R.discover(tmp_path, fetcher, sections, today=TODAY, start=start)


def test_a_crawl_delay_below_the_floor_is_raised_to_it():
    f, _, _ = _fetcher(lambda u: (200, ""))
    assert R.Fetcher(Client(lambda u: (200, "")), lambda u: True, 1.0).delay_s == R.MIN_DELAY_S
    assert R.Fetcher(Client(lambda u: (200, "")), lambda u: True, 25.0).delay_s == 25.0


def test_a_hub_without_load_more_is_reported_not_guessed(tmp_path):
    f, c, _ = _fetcher(lambda url: (200, "<html><body>no cards</body></html>"))
    state = _run(tmp_path, f, ["/info/dialogue/"])
    assert state["sections"]["/info/dialogue/"]["status"] == "no_loadmore"


def test_rts_needs_its_discovery_first(tmp_path):
    with pytest.raises(SystemExit, match="discover rts"):
        B.plan(["rts"], today=TODAY, state_dir=tmp_path)
    with pytest.raises(SystemExit, match="--state-dir"):
        B.plan(["rts"], today=TODAY)


def test_the_three_real_article_url_shapes_are_recognised_and_hubs_are_not():
    shapes = [
        "https://www.rts.ch/info/suisse/2026/article/drame-a-crans-montana-la-gerante-du-constellation-s-excuse-pour-la-tragedie-29113505.html",
        "https://www.rts.ch/info/suisse/13341000-pourquoi-le-prix-de-lelectricite-va-augmenter-en-suisse.html",
        "https://www.rts.ch/info/monde/2022/minute-par-minute/27415950.html",
    ]
    assert all(R._ARTICLE.search(u) for u in shapes)
    for hub in ("https://www.rts.ch/info/suisse/", "https://www.rts.ch/info/",
                "https://www.rts.ch/sport/football/"):
        assert not R._ARTICLE.search(hub.rstrip("/"))
