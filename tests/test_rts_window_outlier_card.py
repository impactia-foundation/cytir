"""A section's window end is judged on the page's own cards, not on the site's `lastDate` link.

Found 2026-10-03 (plan ruling 18): rts.ch fills the successor link's `lastDate` from ONE outlier card per
page (`/info/suisse/page/8` holds cards of 2-5 September 2026 and a link `lastDate=2024-02-06`), and ignores the
parameter when it serves the next page. Judging by it ended 24 hubs `window_reached` after 9 pages, so the
12-month backfill saw about one month of every hub.
"""
import datetime as dt
import json

from cytir.news import rts_section_discovery as R

TODAY = dt.date(2026, 10, 3)
START = dt.date(2025, 10, 3)
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
          "octobre", "novembre", "décembre"]


def _page(days, nxt, sec="suisse", n0=0):
    cards = "".join(
        f'<a href="/info/{sec}/2026/article/t{n0 + i}-{29300000 + n0 + i}.html"><p class="card-time">'
        f'le {d.day} {MONTHS[d.month - 1]} {d.year}</p></a>' for i, d in enumerate(days))
    return f'<div>{cards}<div data-loadmore-src="{nxt}"></div></div>'


class _Client:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get(self, url):
        path = url.replace(R.ORIGIN, "").split("?")[0]
        self.calls.append(path)
        body = self.pages.get(path)
        return type("R", (), {"status_code": 200 if body else 404, "text": body or "", "headers": {}})()


def _fetcher(pages):
    c = _Client(pages)
    return R.Fetcher(c, lambda u: True, 10.0, sleep=lambda s: None, clock=lambda: 0.0), c


def _site():
    """Hub + pages 1..4 are September 2026 .. December 2025; every page ends in an outlier card and the
    successor link names the outlier's day, as the live site does. Page 5 is June 2025."""
    out = dt.date(2024, 2, 6)
    sept = [dt.date(2026, 9, 5)] * 10 + [dt.date(2026, 9, 3)] * 10
    pages = {"/info/suisse/": _page(sept, f"/info/suisse/page/1?lastDate={out}")}
    for n, day in ((1, dt.date(2026, 9, 1)), (2, dt.date(2026, 6, 1)), (3, dt.date(2026, 3, 1)),
                   (4, dt.date(2025, 12, 1))):
        pages[f"/info/suisse/page/{n}"] = _page([day] * 20 + [out], f"/info/suisse/page/{n + 1}?lastDate={out}",
                                                n0=100 * n)
    pages["/info/suisse/page/5"] = _page([dt.date(2025, 6, 1)] * 20, "/info/suisse/page/6?lastDate=2025-06-01",
                                         n0=900)
    return pages


def test_an_outlier_card_does_not_end_the_section(tmp_path):
    f, c = _fetcher(_site())
    state = R.discover(tmp_path, f, ["/info/suisse/"], today=TODAY, start=START)
    e = state["sections"]["/info/suisse/"]
    assert e["status"] == "window_reached", e
    assert c.calls[-1] == "/info/suisse/page/5", "the section ran down to the page whose own days are older"
    assert e["last_date"] == "2025-06-01"
    assert state["urls_total"] >= 80


def test_the_pages_median_day_not_its_link_decides():
    old_days = [dt.date(2025, 1, 1)] * 20 + [dt.date(2026, 9, 1)]
    page = {"cards": [{"date": d} for d in old_days], "last_date": dt.date(2026, 9, 1)}
    assert R._before_window(page, START) is True, "an outlier NEWER card does not extend the section either"
    page = {"cards": [{"date": dt.date(2026, 9, 1)}] * 20 + [{"date": dt.date(2024, 2, 6)}],
            "last_date": dt.date(2024, 2, 6)}
    assert R._before_window(page, START) is False


def test_a_v2_state_that_ended_on_an_outlier_is_reopened(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"schema": R.STATE_SCHEMA_V2, "sections": {
        "/info/suisse/": {"status": "window_reached", "cursor": "/info/suisse/page/2?lastDate=2024-02-06",
                          "pages": 3, "last_date": "2024-02-06", "stale_run": 0, "urls_new": 5},
        "/info/dialogue/": {"status": "complete", "pages": 4}}}))
    state = R.load_state(tmp_path / "state.json")
    assert state["schema"] == R.STATE_SCHEMA
    assert "status" not in state["sections"]["/info/suisse/"]
    assert state["sections"]["/info/suisse/"]["cursor"].startswith("/info/suisse/page/2")
    assert state["sections"]["/info/dialogue/"]["status"] == "complete"
    # a v3 state is left as it is
    state["sections"]["/info/suisse/"]["status"] = "window_reached"
    R.save_state(tmp_path / "state.json", state)
    assert R.load_state(tmp_path / "state.json")["sections"]["/info/suisse/"]["status"] == "window_reached"
