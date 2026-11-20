"""`crawl.sitemap_urls`: an error is raised, an empty sitemap is `[]` (task 683, S7 part 2).

Before, a 500, a timeout or a bad gzip all returned `[]`, which a caller cannot tell from a real
empty sitemap. Callers today: `cytir.acquire.crawl` (the pack_fetch inventory build),
`cytir.sources.cov.engine.enum_sitemap`, and the sitemap family; none is on a live timer or chain.
"""
import gzip
import pathlib
import sys

import pytest

httpx = pytest.importorskip("httpx")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cytir.acquire import crawl  # noqa: E402

XML = ('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
       '<url><loc>https://x.test/a</loc></url><url><loc>https://x.test/b?x=1&amp;y=2</loc></url>'
       '</urlset>')
INDEX = ('<sitemapindex><sitemap><loc>https://x.test/s1.xml</loc></sitemap>'
         '<sitemap><loc>https://x.test/s2.xml</loc></sitemap></sitemapindex>')


def _client(routes):
    def handler(req):
        r = routes.get(str(req.url))
        if isinstance(r, Exception):
            raise r
        status, body = r if r else (404, b"")
        return httpx.Response(status, content=body)
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


def test_a_readable_sitemap_returns_its_locs():
    c = _client({"https://x.test/s.xml": (200, XML.encode())})
    assert crawl.sitemap_urls(c, "https://x.test/s.xml") == [
        "https://x.test/a", "https://x.test/b?x=1&y=2"]


def test_a_gzip_sitemap_is_read():
    c = _client({"https://x.test/s.xml.gz": (200, gzip.compress(XML.encode()))})
    assert len(crawl.sitemap_urls(c, "https://x.test/s.xml.gz")) == 2


def test_a_read_empty_sitemap_is_the_only_empty():
    c = _client({"https://x.test/s.xml": (200, b"<urlset></urlset>")})
    assert crawl.sitemap_urls(c, "https://x.test/s.xml") == []


@pytest.mark.parametrize("route", [
    (500, b"boom"), (404, b"nope"), (403, b"blocked"), (200, b"\x1f\x8bnot-gzip"),
    httpx.ConnectTimeout("slow")])
def test_an_error_raises_and_names_the_url(route):
    c = _client({"https://x.test/s.xml": route})
    with pytest.raises(crawl.SitemapError, match="x.test/s.xml"):
        crawl.sitemap_urls(c, "https://x.test/s.xml")


def test_a_failing_child_of_an_index_is_not_dropped_silently():
    c = _client({"https://x.test/i.xml": (200, INDEX.encode()),
                 "https://x.test/s1.xml": (200, XML.encode()),
                 "https://x.test/s2.xml": (503, b"busy")})
    with pytest.raises(crawl.SitemapError, match="s2.xml"):
        crawl.sitemap_urls(c, "https://x.test/i.xml")


def test_the_old_helper_and_its_shim_are_one_object():
    from cytir.acquire import crawl as pack_fetch  # the pipeline shim that live-era callers import
    assert pack_fetch.sitemap_urls is crawl.sitemap_urls
    assert pack_fetch.SitemapError is crawl.SitemapError
