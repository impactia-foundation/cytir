"""Tests for news_window_filter.py — Tim's 2026-08-11 retention ruling.

The ruling splits one constant into two (engineering-doctrine: one constant,
one meaning, one home):
  * AW_NEWS_FRESH_MONTHS  = 12 (unchanged) — discovery/freshness-ranking tier.
  * AW_NEWS_RETAIN_MONTHS = 36 (was effectively 12) — purge/aged-out horizon.

news_window_filter.py itself stays generic (it just answers "in/out of a
`months`-back cutoff"); these tests exercise it the way each real caller does:
news_refresh_stage.sh with months=12, news_refresh_delta.sh with months=36.
"""
from __future__ import annotations

import datetime as dt

from cytir.index import news_window_filter as subject

NOW = dt.date(2026, 8, 11)


def _published(months_ago: int) -> str:
    """An ISO date exactly `months_ago` calendar months before NOW."""
    y, m = NOW.year, NOW.month - months_ago
    while m <= 0:
        m += 12
        y -= 1
    return dt.date(y, m, NOW.day).isoformat()


def test_fresh_tier_boundary_unchanged_at_12_months():
    """The FRESH/discovery cutoff (AW_NEWS_FRESH_MONTHS=12) is untouched by
    the retention ruling: an article exactly 12mo old is in, 12mo+1day out."""
    cutoff = subject.window_cutoff(NOW, months=12)
    assert cutoff == dt.date(2025, 8, 11)
    assert subject.in_window(_published(12), cutoff) is True
    just_before = (cutoff - dt.timedelta(days=1)).isoformat()
    assert subject.in_window(just_before, cutoff) is False


def test_20_month_old_article_not_in_delete_list_at_36_month_retain():
    """The core acceptance test from the ruling: 20mo old survives the new
    36mo purge horizon (it would NOT have survived the old 12mo horizon)."""
    cutoff_retain = subject.window_cutoff(NOW, months=36)
    published = _published(20)
    assert subject.in_window(published, cutoff_retain) is True  # keep

    # Demonstrate this is exactly the case the old 12mo purge would have
    # deleted -- the ruling's whole point.
    cutoff_old_fresh = subject.window_cutoff(NOW, months=12)
    assert subject.in_window(published, cutoff_old_fresh) is False


def test_40_month_old_article_is_in_delete_list_at_36_month_retain():
    cutoff_retain = subject.window_cutoff(NOW, months=36)
    published = _published(40)
    assert subject.in_window(published, cutoff_retain) is False  # delete


def test_36_month_boundary_exact():
    cutoff_retain = subject.window_cutoff(NOW, months=36)
    assert cutoff_retain == dt.date(2023, 8, 11)
    assert subject.in_window(_published(36), cutoff_retain) is True
    just_before = (cutoff_retain - dt.timedelta(days=1)).isoformat()
    assert subject.in_window(just_before, cutoff_retain) is False


def test_24_month_old_article_survives_the_retention_floor_too():
    """Tim's floor was 24mo, implemented at 36mo -- a 24mo article must
    survive under either reading."""
    for months in (24, 36):
        cutoff = subject.window_cutoff(NOW, months=months)
        assert subject.in_window(_published(24), cutoff) is True


def test_partition_zero_deletes_below_36_months(monkeypatch=None):
    """End-to-end: _partition against a realistic mixed-age news set with the
    RETAIN horizon (36) must delete nothing younger than 36mo, and must delete
    everything older."""
    ages_months = [0, 3, 11, 12, 13, 20, 23, 24, 25, 35, 36, 37, 48, 60]
    records = [
        {"id": f"p{age}", "payload": {"disc_class": "news",
                                       "published": _published(age)}}
        for age in ages_months
    ]
    cutoff = subject.window_cutoff(NOW, months=36)
    keep, aged, unknown, nonnews = subject._partition(records, cutoff, None)
    aged_ages = sorted(int(r["id"][1:]) for r in aged)
    assert aged_ages == [37, 48, 60]
    assert all(age < 36 or age == 36 for age in
               sorted(int(r["id"][1:]) for r in keep))
    assert unknown == []
    assert nonnews == 0
    # explicit: nothing below 36 months is ever in the delete list
    assert all(age >= 36 for age in aged_ages)


def test_non_news_never_aged_out_regardless_of_months():
    """A non-news record (law/gov/reference) is never subject to either
    window, however old — only disc_class=='news' is."""
    record = {"id": "gov1", "payload": {"disc_class": "gov",
                                         "published": _published(120)}}
    cutoff = subject.window_cutoff(NOW, months=36)
    keep, aged, unknown, nonnews = subject._partition([record], cutoff, None)
    assert aged == []
    assert nonnews == 1
    assert keep == [record]


def test_unknown_date_is_never_deleted():
    record = {"id": "nodate", "payload": {"disc_class": "news"}}
    cutoff = subject.window_cutoff(NOW, months=36)
    keep, aged, unknown, nonnews = subject._partition([record], cutoff, None)
    assert aged == []
    assert len(unknown) == 1
