from datetime import date
import gzip

import pytest

from cytir.acquire import sitemap_ingest as subject


class AllowAll:
    def can_fetch(self, _agent, _url):
        return True


def test_select_recent_is_windowed_newest_first_then_undated():
    rows = [
        {"url": "https://example.test/old", "lastmod": "2025-07-31"},
        {"url": "https://example.test/current", "lastmod": "2026-08-02"},
        {"url": "https://example.test/middle", "lastmod": "2026-01-03"},
        {"url": "https://example.test/undated", "lastmod": None},
        {"url": "https://example.test/future", "lastmod": "2026-08-04"},
    ]

    selected = subject.select_recent(
        rows, {"host": "example.test"}, 3, AllowAll(), date(2026, 8, 3), 12)

    assert [row["url"] for row in selected] == [
        "https://example.test/current",
        "https://example.test/middle",
        "https://example.test/undated",
    ]
    assert selected[0]["published"] == "2026-08-02"
    assert "published" not in selected[-1]


def test_select_recent_clamps_leap_day_cutoff():
    rows = [
        {"url": "https://example.test/keep", "lastmod": "2023-02-28"},
        {"url": "https://example.test/drop", "lastmod": "2023-02-27"},
    ]

    selected = subject.select_recent(
        rows, {"host": "example.test"}, 10, AllowAll(), date(2024, 2, 29), 12)

    assert [row["url"] for row in selected] == ["https://example.test/keep"]


def test_consent_stamp_is_explicit_and_fail_closed():
    assert subject._consent_stamp({
        "search_ok": True, "train_ok": False, "checked_at": "2026-08-03T00:00:00Z",
    }) == {
        "search_ok": True,
        "train_ok": False,
        "consent_state": "resolved",
        "consent_basis": "robots-live-resolve",
        "consent_source": "consent_resolver",
        "consent_checked_at": "2026-08-03T00:00:00Z",
    }
    with pytest.raises(ValueError):
        subject._consent_stamp({"search_ok": True, "train_ok": True})


def test_search_only_publisher_is_admitted_for_indexing_not_training():
    receipt = {
        "status": "ok", "search_ok": True, "train_ok": False,
        "checked_at": "2026-08-26T00:00:00Z",
    }
    assert subject._search_index_bucket(receipt) == "ok_search_only"


def test_search_decline_remains_excluded_from_indexing():
    receipt = {
        "status": "ok", "search_ok": False, "train_ok": False,
        "checked_at": "2026-08-26T00:00:00Z",
    }
    assert subject._search_index_bucket(receipt) == "declined"


def test_discover_sitemap_unescapes_child_query_urls(monkeypatch):
    index = """<sitemapindex><sitemap><loc>https://example.test/day.xml?mm=08&amp;dd=05</loc></sitemap></sitemapindex>"""
    child = """<urlset><url><loc>https://example.test/article?a=1&amp;b=2</loc><lastmod>2026-08-05</lastmod></url></urlset>"""
    seen = []

    def fetch(_client, url, *_args, **_kwargs):
        seen.append(url)
        return index if url == "https://example.test/sitemap.xml" else child

    monkeypatch.setattr(subject, "_fetch_body", fetch)
    rows = subject.discover_sitemap_urls(None, {"host": "example.test",
        "sitemaps": ["https://example.test/sitemap.xml"]}, "unused", None, None)

    assert seen == ["https://example.test/sitemap.xml", "https://example.test/day.xml?mm=08&dd=05"]
    assert rows == [{"url": "https://example.test/article?a=1&b=2", "lastmod": "2026-08-05"}]


def test_decode_sitemap_bytes_unwraps_bounded_nested_gzip():
    expected = b'<?xml version="1.0"?><urlset></urlset>'
    nested = expected
    for _ in range(4):
        nested = gzip.compress(nested)
    assert subject._decode_sitemap_bytes(nested) == expected.decode()


def test_decode_sitemap_bytes_rejects_excessive_nesting(monkeypatch):
    nested = b"<urlset></urlset>"
    for _ in range(3):
        nested = gzip.compress(nested)
    monkeypatch.setattr(subject, "_SITEMAP_GZIP_MAX_LAYERS", 2)
    assert subject._decode_sitemap_bytes(nested) is None
