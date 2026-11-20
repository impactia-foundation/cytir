"""A section stops on its OWN evidence, never because its cards are already known (BAU shift 5).

The defect (plan.md ruling 18, found 2026-10-03): `STALE_PAGES=3` counted pages with no NEW
url, and the first pages of `/info/suisse/`, `/info/monde/`, `/info/culture/`, `/info/regions/`
overlap the `/info/` front feed processed first, so those four hubs ended `stale` after 3 pages
in September. A live probe of `/info/suisse/page/30` (late May 2026) showed 4 of 24 articles
known. The rule now: a stall is `STALE_PAGES` consecutive pages with no article card, or a
successor that is the page just read; the window end is the page's own cursor day (or, with none,
every dated card) older than the window start.

Real pages: `testdata/rts-hubs-20261003.json.gz` (hubs, pages 1 to 3 of `/info/monde/` and
`/info/culture/` as rts.ch returned them from the build box on 2026-10-03, AWbot UA, 20 s apart) and
the 2026-09-30 fixture (`/info/suisse/` page 60 = January 2026, page 400 = August 2022).
"""
from __future__ import annotations
import datetime as dt
import gzip
import json
import pathlib
import pytest
import cytir.news.rts_section_discovery as R  # noqa: E402


HERE = pathlib.Path(__file__).resolve().parent


TODAY = dt.date(2026, 10, 3)


D = dt.date


START = D(2026, 3, 1)


@pytest.fixture(scope="module")
def hubs():
    with gzip.open(HERE / "testdata" / "rts-hubs-20261003.json.gz", "rt", encoding="utf-8") as fh:
        return json.load(fh)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Client:
    def __init__(self, script):
        self.script, self.calls = script, []

    def get(self, url):
        self.calls.append(url.replace(R.ORIGIN, ""))
        status, text = self.script(url.replace(R.ORIGIN, ""))
        return type("R", (), {"status_code": status, "text": text, "headers": {}})()


def _fetcher(script):
    clock = Clock()
    client = Client(script)
    return R.Fetcher(client, lambda u: True, 10.0, sleep=clock.sleep, clock=clock), client


def test_with_no_cursor_day_every_dated_card_before_the_window_ends_the_section():
    page = {"last_date": None, "cards": [{"url": "u1", "date": D(2025, 1, 1)},
                                         {"url": "u2", "date": D(2024, 12, 1)}]}
    assert R._before_window(page, START) is True
    page["cards"] += [{"url": "u3", "date": D(2026, 5, 1)}, {"url": "u4", "date": D(2026, 5, 2)}]
    assert R._before_window(page, START) is False, "the median day decides, not the oldest card"
    assert R._before_window({"last_date": None, "cards": []}, START) is False, \
        "no evidence is not evidence of the window start"


def test_pages_without_any_article_card_are_a_stall_after_stale_pages(tmp_path, hubs):
    n = {"i": 0}

    def script(path):
        if path == "/info/monde/":
            return 200, hubs["hub_monde"]
        n["i"] += 1     # the site keeps giving a successor, with changing cursors, and no cards
        return 200, f'<html><a data-loadmore-src="/info/monde/page/{n["i"] + 1}?lastDate=2026-09-01"></a></html>'
    f, _ = _fetcher(script)
    state = R.discover(tmp_path, f, ["/info/monde/"], today=TODAY, start=START)
    e = state["sections"]["/info/monde/"]
    assert e["status"] == "stale" and e["stale_run"] == R.STALE_PAGES and e["pages"] == R.STALE_PAGES
