"""task 661 F5 follow-up (2026-09-26): the vendor-blog RSS new-refs
discovery adapter (OpenAI news, Google AI blog).

Fixtures are REAL feed responses, not hand-built:
`tests/data/refresh_direct/openai-news-rss.xml` (fetched live from
openai.com/news/rss.xml, 2026-09-26, truncated to the first 10 of 1230
real items -- the XML text of each kept item is byte-identical to the
live response). `tests/data/refresh_direct/google-ai-blog-rss.xml` (fetched live
from blog.google/technology/ai/rss/, 2026-09-26, kept whole -- 20 items,
31 KB).
"""
from __future__ import annotations

import pathlib

import pytest

from cytir.sources.refresh_direct import vendor_blog_rss as V

OPENAI_FIXTURE = pathlib.Path(__file__).parent / "data" / "refresh_direct" / "openai-news-rss.xml"
GOOGLE_FIXTURE = pathlib.Path(__file__).parent / "data" / "refresh_direct" / "google-ai-blog-rss.xml"


def test_parse_rss_reads_the_real_openai_feed():
    items = V.parse_rss(OPENAI_FIXTURE.read_text(), "openai-news")
    assert len(items) == 10
    first = items[0]
    assert first["url"] == "https://openai.com/index/proaction"
    assert first["title"] == "Proaction boosts sales 60% and saves 75+ hours with Codex"
    assert first["description"] == (
        "With Codex, GPT-Live-1, and GPT-6 Astra, Proaction builds, "
        "operates, and sells modern fleet management faster.")
    assert first["published"] == "Fri, 25 Sep 2026 19:00:00 GMT"
    assert first["blog"] == "openai-news"


def test_parse_rss_reads_the_real_google_feed():
    items = V.parse_rss(GOOGLE_FIXTURE.read_text(), "google-ai-blog")
    assert len(items) == 20
    first = items[0]
    assert first["url"] == (
        "https://blog.google/innovation-and-ai/technology/research/google-beam-expansion/")
    assert first["title"] == "Google Beam expands with new regions, partners, and customers"
    # Google's <description> carries a leading <img ...> tag before the
    # real sentence -- stripped, no HTML in the output.
    assert first["description"] == (
        "We’re expanding Google Beam to five new countries, and "
        "partnering with Industrious for an extended network.")
    assert "<img" not in first["description"]
    assert first["published"] == "Wed, 23 Sep 2026 18:00:00 +0000"
    assert first["blog"] == "google-ai-blog"


def test_parse_rss_description_is_empty_string_not_none_when_absent():
    xml = """<rss><channel>
        <item><title>no description</title>
              <link>https://example.com/a</link><pubDate>y</pubDate></item>
    </channel></rss>"""
    items = V.parse_rss(xml, "test")
    assert items[0]["description"] == ""


def test_parse_rss_ignores_google_only_namespaced_elements():
    """Google's feed carries per-item media:content/og:*/author:* the
    parser must not choke on or accidentally read as the title/link."""
    items = V.parse_rss(GOOGLE_FIXTURE.read_text(), "google-ai-blog")
    for it in items:
        assert it["url"].startswith("https://blog.google/")
        assert "<img" not in it["title"]  # never leaks description/og markup into title


def test_parse_rss_skips_an_item_with_no_link():
    xml = """<rss><channel>
        <item><title>no link here</title><pubDate>x</pubDate></item>
        <item><title>real</title><link>https://example.com/a</link><pubDate>y</pubDate></item>
    </channel></rss>"""
    items = V.parse_rss(xml, "test")
    assert len(items) == 1
    assert items[0]["url"] == "https://example.com/a"


def test_discover_refuses_when_no_blogs_are_configured():
    with pytest.raises(ValueError):
        V.discover({"acquire": {"params": {"blogs": []}}})


def test_discover_uses_the_injected_fetch_never_the_network():
    calls = []

    def fake_fetch(url):
        calls.append(url)
        if "openai" in url:
            return OPENAI_FIXTURE.read_text()
        return GOOGLE_FIXTURE.read_text()

    result = V.discover(
        {"acquire": {"params": {"blogs": [
            {"label": "openai-news", "url": "https://openai.com/news/rss.xml"},
            {"label": "google-ai-blog", "url": "https://blog.google/technology/ai/rss/"},
        ]}}},
        fetch=fake_fetch)
    assert calls == ["https://openai.com/news/rss.xml", "https://blog.google/technology/ai/rss/"]
    assert result["blogs_checked"] == 2
    assert result["blogs_errored"] == 0
    assert result["candidates_found"] == 30  # 10 openai + 20 google


def test_discover_names_a_failed_blog_error_without_dropping_the_others():
    def flaky_fetch(url):
        if "bad-blog" in url:
            raise RuntimeError("HTTP 503")
        return OPENAI_FIXTURE.read_text()

    result = V.discover(
        {"acquire": {"params": {"blogs": [
            {"label": "openai-news", "url": "https://openai.com/news/rss.xml"},
            {"label": "bad-blog", "url": "https://example.com/bad-blog/rss.xml"},
        ]}}},
        fetch=flaky_fetch)
    assert result["blogs_checked"] == 2
    assert result["blogs_errored"] == 1
    assert "bad-blog" in result["errors"]
    assert "503" in result["errors"]["bad-blog"]
    assert result["candidates_found"] == 10  # the good blog's posts still counted


def test_blogs_from_pack_path_reads_the_real_pack():
    """The pack's root `acquire.params` cannot hold a flat blog list
    (schema-invalid, same reason as the GitHub/HF siblings), so the
    tracked blogs live in `routes[]`. This proves the real committed
    pack yields the same two-blog list."""
    blogs = V.blogs_from_pack_path("sources/us/vendor-ai-blogs-tracked.yaml")
    assert blogs[:2] == [
        {"label": "openai.com/news", "url": "https://openai.com/news/rss.xml"},
        {"label": "blog.google/technology/ai",
         "url": "https://blog.google/technology/ai/rss/"},
    ]
    # task 715: the Google section feeds that carry the Gemini model posts
    assert len(blogs) == 7
    assert {b["label"] for b in blogs[2:]} >= {
        "blog.google/innovation-and-ai/models-and-research/gemini-models"}


def test_discover_falls_back_to_pack_path_when_acquire_is_absent():
    """The real `new_refs.py` path: cadence-table rows never
    carry `acquire` (only `pack_path`)."""
    calls = []

    def fake_fetch(url):
        calls.append(url)
        if "openai" in url:
            return OPENAI_FIXTURE.read_text()
        return GOOGLE_FIXTURE.read_text()

    result = V.discover(
        {"id": "us-vendor-ai-blogs-tracked",
         "pack_path": "sources/us/vendor-ai-blogs-tracked.yaml"},
        fetch=fake_fetch)
    assert calls[:2] == ["https://openai.com/news/rss.xml", "https://blog.google/technology/ai/rss/"]
    assert result["blogs_checked"] == len(calls) == 7
    # the fake serves the same 20 Google items for every Google feed: dedup keeps 20
    assert result["candidates_found"] == 30


def test_discover_refuses_when_pack_path_has_no_tracked_blogs(tmp_path):
    empty_pack = tmp_path / "empty.yaml"
    empty_pack.write_text("id: x\nroutes: []\n")
    with pytest.raises(ValueError):
        V.discover({"pack_path": str(empty_pack)})


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))


# ---- task 715 (2026-10-07): section feeds of the same host ----------------

def test_discover_fetches_a_post_listed_in_two_feeds_once():
    item = ("<item><title>t</title><link>{u}</link>"
            "<pubDate>Wed, 30 Sep 2026 20:00:00 +0000</pubDate></item>")
    feeds = {
        "https://h.example/a/rss/": "<rss><channel>" + item.format(u="https://h.example/p/x/")
                                    + item.format(u="https://h.example/p/y/") + "</channel></rss>",
        "https://h.example/b/rss/": "<rss><channel>" + item.format(u="https://h.example/p/x")
                                    + "</channel></rss>",
    }
    row = {"acquire": {"params": {"blogs": [
        {"label": "a", "url": "https://h.example/a/rss/"},
        {"label": "b", "url": "https://h.example/b/rss/"}]}}}
    out = V.discover(row, fetch=feeds.__getitem__)
    assert [c["url"] for c in out["candidates"]] == [
        "https://h.example/p/x/", "https://h.example/p/y/"]
    assert out["candidates"][0]["blog"] == "a"


def test_roster_feeds_are_well_formed_and_belong_to_their_canonical_domain():
    import urllib.parse as up
    import yaml
    doc = yaml.safe_load((V.REPO_ROOT / "sources/us/vendor-ai-blogs-tracked.yaml").read_text())
    routes = doc["routes"]
    assert routes, "roster has no routes"
    urls = []
    for r in routes:
        params = r["acquire"]["params"]
        url = params["url"]
        parts = up.urlsplit(url)
        assert parts.scheme == "https" and parts.netloc, url
        assert params["kind"] == "rss", url
        canon = r["canonical_domain"]
        assert parts.netloc == canon or parts.netloc.endswith("." + canon), (url, canon)
        assert r["host"].startswith(canon), (r["host"], canon)
        assert r.get("evidence"), f"route {r['host']} carries no discovery evidence"
        urls.append(url)
    assert len(urls) == len(set(urls)), "a feed is listed twice"
    labels = [r["host"] for r in routes]
    assert len(labels) == len(set(labels)), "a route label is listed twice"
    assert {r["canonical_domain"] for r in routes} == {"openai.com", "blog.google"}, \
        "a new host is a legislative change, not a roster addition"
    assert "https://blog.google/innovation-and-ai/models-and-research/gemini-models/rss/" in urls
