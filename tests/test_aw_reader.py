"""Task 2710: the local reader (`cyt read`, `cytir-reader`). Offline: fixtures and mocked fetch/render only."""
import asyncio
import json
import shutil
from pathlib import Path

import pytest

from cytir.core.search_engines import is_search_engine
from cytir.reader import config, fetch as F, read as R
from cytir.reader.extract import extract
from cytir.reader.render import InstalledBrowser, Renderer, RenderError, find_renderer
from cytir.reader.shell import is_shell

FX = Path(__file__).parent / "fixtures" / "reader"
SHELL = (FX / "fedlex_shell.html").read_text(encoding="utf-8")
RENDERED = (FX / "fedlex_rendered_art336c.html").read_text(encoding="utf-8")
URL = "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_336_c"
ART = "Nach Ablauf der Probezeit darf der Arbeitgeber das Arbeitsverhältnis nicht kündigen"
PLAIN = "<html><body><article>" + ("Real text about a topic. " * 60) + "</article></body></html>"


@pytest.fixture(autouse=True)
def _offline_dns(monkeypatch):
    """No test resolves a name for real: numeric shorthand is loopback, everything else is a public address."""
    import socket
    loopback = {"127.1", "2130706433", "0x7f.0.0.1"}

    def fake(host, *a, **k):
        addr = "127.0.0.1" if host in loopback else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, 0))]
    monkeypatch.setattr("cytir.reader.fetch.socket.getaddrinfo", fake)


def test_shell_detection_true_and_false():
    assert is_shell(SHELL, extract(SHELL, URL)[0])                      # the real fedlex shell
    thin = "<html><body><noscript>Please enable JavaScript.</noscript><p>x</p></body></html>"
    assert is_shell(thin, "x")                                           # noscript message
    assert is_shell("<noscript>Diese Seite ist nur mit einem Javascript-fähigen Browser verfügbar</noscript>", "short")
    assert not is_shell(PLAIN, extract(PLAIN, "https://x.org/")[0])     # real page
    assert not is_shell(RENDERED, extract(RENDERED, URL)[0])             # rendered DOM
    rich = "<html><noscript>enable javascript</noscript></html>"
    assert not is_shell(rich, "word " * 600)                             # a JS footer on a rich page is not a shell


@pytest.mark.parametrize("url", [
    "https://www.bing.com/search?q=x", "https://duckduckgo.com/?q=x", "https://www.google.com/search?q=x",
    "https://yandex.ru/search/?text=x", "https://search.yahoo.com/search?p=x", "https://uk.search.yahoo.com/x",
    "https://www.yahoo.com/search?p=x", "https://search.brave.com/search?q=x", "https://brave.com/search/?q=x",
    "https://kagi.com/search?q=x"])
def test_search_engines_blocked(url):
    assert is_search_engine(url)


@pytest.mark.parametrize("url", ["https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de", "https://brave.com/about",
                                 "https://www.yahoo.com/news", "https://docs.python.org/3/", "", None])
def test_normal_urls_not_blocked(url):
    assert not is_search_engine(url)


def test_fetch_refuses_search_engines_and_non_http():
    for bad in ("https://www.bing.com/search?q=x", "ftp://example.org/x", "file:///etc/passwd", "javascript:alert(1)"):
        with pytest.raises(F.FetchRefused):
            F.check_url(bad)
    with pytest.raises(F.FetchRefused):
        R.read("https://duckduckgo.com/?q=x", fetcher=lambda u: pytest.fail("must not fetch"))


def test_anchor_trim_on_the_rendered_fixture():
    text, anchored = extract(RENDERED, URL)
    assert anchored and ART in text
    assert "Art. 336" in text
    full, anchored_full = extract(RENDERED, "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de")
    assert not anchored_full and full.strip()             # no fragment: the whole page, untrimmed
    other, _ = extract(RENDERED, "https://x.org/p#art_336_b")
    assert ART not in other                                 # stops at the next art_ sibling
    missing, anchored_missing = extract(RENDERED, "https://x.org/p#nope")
    assert not anchored_missing and ART in missing          # absent id: falls back to the whole text


def test_anchor_trim_heading_siblings():
    html = ("<html><body><h2 id='s1'>One</h2><p>alpha text</p><h3>Sub</h3><p>beta text</p>"
            "<h2 id='s2'>Two</h2><p>gamma text</p></body></html>")
    text, anchored = extract(html, "https://x.org/#s1")
    assert anchored and "alpha" in text and "beta" in text and "gamma" not in text


class FakeFetch:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def __call__(self, url):
        self.calls.append(url)
        key = url.partition("#")[0]
        if key not in self.pages:
            raise F.FetchError("HTTP 404")
        return F.Fetched(url, 200, "text/html", self.pages[key])


class FakeRenderer(Renderer):
    def __init__(self, html=RENDERED):
        super().__init__("fake", "/bin/fake")
        self.html, self.rendered = html, []

    def render(self, url):
        self.rendered.append(url)
        if isinstance(self.html, Exception):
            raise self.html
        return self.html


def test_fallback_plain_fetch_wins():
    f = FakeFetch({"https://x.org/p": PLAIN})
    out = R.read("https://x.org/p", read_url="https://x.org/twin", fetcher=f, renderer=FakeRenderer())
    assert out["source_used"] == "url" and f.calls == ["https://x.org/p"] and out["renderer"] is None


def test_fallback_read_url_before_render():
    f = FakeFetch({"https://x.org/p": SHELL, "https://x.org/twin": RENDERED})
    r = FakeRenderer()
    out = R.read("https://x.org/p#art_336_c", read_url="https://x.org/twin", fetcher=f, renderer=r)
    assert out["source_used"] == "read_url" and ART in out["text"] and not r.rendered
    assert f.calls[1] == "https://x.org/twin#art_336_c"     # the twin keeps the anchor


def test_fallback_render_when_twin_also_a_shell_or_missing():
    for pages in ({"https://x.org/p": SHELL, "https://x.org/twin": SHELL}, {"https://x.org/p": SHELL}):
        r = FakeRenderer()
        out = R.read("https://x.org/p#art_336_c", read_url="https://x.org/twin", fetcher=FakeFetch(pages), renderer=r)
        assert out["source_used"] == "rendered" and out["renderer"] == "fake" and ART in out["text"]
        assert r.rendered == ["https://x.org/p"]            # fragment not sent to the renderer


def test_no_renderer_returns_shell_text_with_install_hint():
    out = R.read("https://x.org/p", fetcher=FakeFetch({"https://x.org/p": SHELL}), renderer=None)
    assert out["source_used"] == "url" and config.INSTALL_HINT in out["notes"]


def test_render_failure_is_a_note_not_a_crash():
    out = R.read("https://x.org/p", fetcher=FakeFetch({"https://x.org/p": SHELL}),
                 renderer=FakeRenderer(RenderError("boom")))
    assert any("render failed" in n for n in out["notes"]) and out["source_used"] == "url"


def test_max_chars_truncates():
    out = R.read("https://x.org/p", max_chars=100, fetcher=FakeFetch({"https://x.org/p": PLAIN}), renderer=None)
    assert out["truncated"] and len(out["text"]) == 100
    assert set(out) == {"url", "final_url", "source_used", "renderer", "text", "truncated", "elapsed_ms", "notes"}


def test_renderer_selection_and_root_sandbox(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/chromium" if n == "chromium" else None)
    r = find_renderer("auto")
    assert isinstance(r, InstalledBrowser) and r.name == "chromium"
    monkeypatch.setattr("cytir.reader.render._is_root", lambda: True)
    assert "--no-sandbox" in r.argv("https://x.org/") and "--dump-dom" in r.argv("https://x.org/")
    monkeypatch.setattr("cytir.reader.render._is_root", lambda: False)
    assert "--no-sandbox" not in r.argv("https://x.org/")
    assert find_renderer("obscura") is None and find_renderer("none") is None
    monkeypatch.setenv("AW_READER_RENDERER", "none")
    assert find_renderer() is None


@pytest.mark.skipif(not any(shutil.which(b) for b in config.BROWSER_BINARIES), reason="no installed browser")
def test_installed_browser_renders_a_local_page(tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<html><body><div id=a></div><script>document.getElementById('a').textContent='JS-DRAWN'"
                    "</script></body></html>")
    assert "JS-DRAWN" in find_renderer("auto").render(page.as_uri())


def test_cli_json_shape(monkeypatch, capsys):
    from cytir import cli
    from cytir.reader import cli as rcli
    real = R.read
    monkeypatch.setattr(rcli.R, "read", lambda url, **kw: real(
        url, fetcher=FakeFetch({"https://x.org/p": PLAIN}), renderer=None, max_chars=kw["max_chars"]))
    assert cli.main(["read", "https://x.org/p", "--max-chars", "50", "--renderer", "none"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["source_used"] == "url" and len(doc["text"]) == 50 and doc["truncated"] is True


def test_cli_refusal_is_json_and_nonzero(capsys):
    from cytir import cli
    assert cli.main(["read", "https://www.bing.com/search?q=x"]) == 1
    assert "search-engine" in json.loads(capsys.readouterr().out)["error"]


def test_mcp_tool_registration():
    pytest.importorskip("mcp", reason="mcp is in requirements-ci.txt")
    from cytir.reader import mcp as rmcp
    tools = {t.name: t for t in asyncio.run(rmcp.mcp.list_tools())}
    assert set(tools) == {"read"} and rmcp.MCP_NAME == "cytir-reader"
    props = tools["read"].input_schema["properties"]
    assert {"url", "read_url", "max_chars"} <= set(props)
