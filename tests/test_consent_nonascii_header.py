"""Task 791: a non-ASCII response header never crashes the consent resolver.

Real records (the seal watchdog log, 2026-10-08 consent pass of the five
delta packs, task 783): hosts held as ``resolver crash: UnicodeEncodeError('ascii',
...)``, e.g.

    a Colombian senate host '404 Categoría no encontrada'
    an Australian NGO host      '“SAMEORIGIN”'
    a WordPress host             '⛄'
    an Indian institute host       "default-src \\x91self\\x92"

Root cause: `_fetch_bounded` rebuilt the response from `headers.multi_items()`
(decoded str pairs); `httpx.Response(headers=<str pairs>)` re-encodes every value
as ASCII. The crash is a HEADER VALUE, not the reason phrase (httpx already
decodes that with errors="ignore"). The fix carries the raw header bytes.

Contract here: the status CODE is kept, the host is not crashed, and the
classification is the one the policy bodies alone give. No network: httpx
MockTransport serves the exact bytes seen on the wire.
"""
import httpx
import pytest

import cytir.consent.resolver as cr

# (header name, header value bytes as sent on the wire, reason phrase bytes)
RECORDED = [
    (b"X-Status", "404 Categoría no encontrada".encode("latin-1"), b"Not Found"),
    (b"X-Frame-Options", "“SAMEORIGIN”".encode("cp1252"), b"Not Found"),
    (b"X-Emoji", "⛄".encode("utf-8"), b"Not Found"),
    (b"Content-Security-Policy", b"default-src \x91self\x92", b"Not Found"),
    # the brief's wording: a non-ASCII reason phrase itself
    (b"X-Plain", b"1", "Categoría no encontrada".encode("latin-1")),
]


def _client(robots_status, robots_body, header, value, reason):
    def handler(request):
        if request.url.path == "/robots.txt":
            status, body = robots_status, robots_body
        else:
            status, body = 404, b""
        return httpx.Response(
            status, headers=[(header, value)], content=body,
            extensions={"reason_phrase": reason})
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(cr.time, "sleep", lambda _s: None)


@pytest.mark.parametrize("header,value,reason", RECORDED)
def test_fetch_bounded_keeps_code_and_bytes(header, value, reason):
    client = _client(404, b"", header, value, reason)
    resp = cr._fetch_bounded(client, "https://example.test/robots.txt",
                             deadline=cr.time.monotonic() + 30)
    assert resp.status_code == 404
    assert (header.lower(), value) in [(k.lower(), v) for k, v in resp.headers.raw]


@pytest.mark.parametrize("header,value,reason", RECORDED)
def test_host_with_nonascii_header_is_not_crashed_and_stays_open(header, value, reason):
    client = _client(404, b"", header, value, reason)
    rec = cr.resolve_host(client, "example.test",
                          direct_crawl=True, origin_scheme="https")
    assert rec["status"] == "ok"
    assert rec["signals"]["robots"]["http_status"] == 404
    assert rec["search_ok"] is True and rec["train_ok"] is True
    assert not any("resolver crash" in r for r in rec["reasons"])


@pytest.mark.parametrize("header,value,reason", RECORDED)
def test_classification_is_unchanged_by_the_header(header, value, reason):
    """A real disallow still declines: the header only used to hide it."""
    body = b"User-agent: *\nDisallow: /\n"
    with_header = cr.resolve_host(_client(200, body, header, value, reason),
                                  "example.test", direct_crawl=True,
                                  origin_scheme="https")
    plain = cr.resolve_host(_client(200, body, b"X-Plain", b"1", b"OK"),
                            "example.test", direct_crawl=True,
                            origin_scheme="https")
    keys = ("status", "search_ok", "train_ok")
    assert {k: with_header[k] for k in keys} == {k: plain[k] for k in keys}
    assert with_header["search_ok"] is False
    assert with_header["signals"]["robots"]["http_status"] == 200
