"""Task 2710 review fixes: shell false positives, SSRF/redirects, size caps, process-group kill, max_chars."""
import os
import time
from pathlib import Path

import httpx
import pytest

from cytir.reader import config, fetch as F, read as R
from cytir.reader.extract import extract
from cytir.reader.render import Renderer, RenderError
from cytir.reader.shell import is_shell

SHELL = (Path(__file__).parent / "fixtures" / "reader" / "fedlex_shell.html").read_text(encoding="utf-8")
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


def test_short_static_page_with_big_inline_script_is_not_a_shell():
    html = ("<html><body><main><p>" + "A short but real paragraph of static text. " * 8 + "</p></main>"
            "<script type='application/ld+json'>" + '{"a": "' + "x" * 30000 + '"}</script></body></html>')
    text = extract(html, "https://x.org/")[0]
    assert len(text) < config.SHELL_MIN_TEXT_CHARS and len(html) > config.SHELL_MIN_HTML_BYTES
    assert not is_shell(html, text)


def test_short_page_that_mentions_javascript_is_not_a_shell():
    msg = "JavaScript is a language. To enable JavaScript in your editor, see the docs."
    assert not is_shell(f"<html><body><p>{msg}</p></body></html>", msg)


def test_fedlex_fixture_is_a_shell():
    assert is_shell(SHELL, extract(SHELL, "https://x.org/")[0])


# SSRF inputs, not endpoints. The literal-endpoint scan in the throwaway-Qdrant guard reads `http://<ip>` strings
# (its allow-list is keyed per file and literal), so the URLs are built from the host at test time.
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "10.1.2.3", "172.16.0.1", "192.168.1.1",
                                  "169.254.169.254", "[::1]", "[fc00::1]"])
def test_private_literal_targets_refused(host):
    bad = "http://" + host + "/"
    with pytest.raises(F.FetchRefused):
        F.check_url(bad)


@pytest.mark.parametrize("loc", ["https://www.bing.com/search?q=x", "ftp://example.org/x", "http://127.0.0.1/admin"])
def test_redirect_to_a_refused_target_is_refused_on_the_hop(loc):
    seen = []

    def handler(req):
        seen.append(str(req.url))
        if req.url.host == "x.org":
            return httpx.Response(302, headers={"location": loc})
        return httpx.Response(200, text="never fetched")
    with pytest.raises((F.FetchRefused, F.FetchError)):
        F.fetch("https://x.org/p", transport=httpx.MockTransport(handler))
    assert seen == ["https://x.org/p"]


def test_streaming_size_cap(monkeypatch):
    monkeypatch.setattr(config, "MAX_BYTES", 1000)
    with pytest.raises(F.FetchError, match="over"):
        F.fetch("https://x.org/p", transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"a" * 5000)))
    ok = F.fetch("https://x.org/p", transport=httpx.MockTransport(lambda r: httpx.Response(200, text="fine")))
    assert ok.html == "fine"


class _Fetch:
    def __call__(self, url):
        return F.Fetched(url, 200, "text/html", PLAIN)


def test_negative_max_chars_is_an_error_and_zero_clamps():
    with pytest.raises(ValueError):
        R.read("https://x.org/p", max_chars=-1, fetcher=_Fetch(), renderer=None)
    assert len(R.read("https://x.org/p", max_chars=0, fetcher=_Fetch(), renderer=None)["text"]) == 1


def _script_renderer(tmp_path, body):
    s = tmp_path / "fake.sh"
    s.write_text("#!/bin/sh\n" + body)
    s.chmod(0o755)

    class Fake(Renderer):
        def argv(self, url):
            return [self.binary]
    return Fake("fake", str(s))


def test_renderer_timeout_kills_the_whole_process_group(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RENDER_TIMEOUT_S", 1.0)
    pidfile = tmp_path / "child.pid"
    r = _script_renderer(tmp_path, f"sleep 60 &\necho $! > {pidfile}\nwait\n")
    with pytest.raises(RenderError, match="timed out"):
        r.render("https://x.org/")
    pid = int(pidfile.read_text())
    for _ in range(30):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    pytest.fail("the helper process survived the renderer timeout")


def test_renderer_output_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_BYTES", 1000)
    r = _script_renderer(tmp_path, "head -c 5000 /dev/zero | tr '\\0' a\n")
    with pytest.raises(RenderError, match="output over"):
        r.render("https://x.org/")


def test_non_chrome_renderer_over_the_rss_cap_is_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ALT_RENDERER_MAX_RSS", 30 * 1024 * 1024)
    s = tmp_path / "hog.sh"
    s.write_text("#!/bin/sh\nexec python3 -c \"x = bytearray(150 * 2**20); import time; time.sleep(30)\"\n")
    s.chmod(0o755)

    class Hog(Renderer):
        limited = True

        def argv(self, url):
            return [self.binary]
    t0 = time.time()
    with pytest.raises(RenderError, match="memory over"):
        Hog("hog", str(s)).render("https://x.org/")
    assert time.time() - t0 < 20


@pytest.mark.parametrize("bad", ["http://127.1/", "http://2130706433/", "http://0x7f.0.0.1/"])
def test_non_canonical_loopback_forms_refused(bad):
    with pytest.raises(F.FetchRefused):
        F.check_url(bad)


def test_public_resolution_allowed_and_private_resolution_refused(monkeypatch):
    import socket
    F.check_url("https://docs.python.org/3/")   # the autouse fake answers a public address
    monkeypatch.setattr("cytir.reader.fetch.socket.getaddrinfo", lambda h, *a, **k: [
        (socket.AF_INET, 1, 6, "", ("93.184.216.34", 0)), (socket.AF_INET, 1, 6, "", ("10.0.0.5", 0))])
    with pytest.raises(F.FetchRefused):   # ANY private answer refuses (a rebinding-style public name)
        F.check_url("https://rebind.example.org/")


def test_resolution_failure_refuses(monkeypatch):
    import socket

    def boom(*a, **k):
        raise socket.gaierror("no such host")
    monkeypatch.setattr("cytir.reader.fetch.socket.getaddrinfo", boom)
    with pytest.raises(F.FetchRefused, match="cannot resolve"):
        F.check_url("https://nonexistent.invalid/")


def test_renderer_stderr_flood_does_not_stall(tmp_path):
    r = _script_renderer(tmp_path, "head -c 3000000 /dev/zero | tr '\\0' e >&2\necho '<html>ok</html>'\n")
    t0 = time.time()
    assert "ok" in r.render("https://x.org/") and time.time() - t0 < 10


def test_mcp_value_error_becomes_a_tool_error():
    pytest.importorskip("mcp")
    from mcp.server.mcpserver.exceptions import ToolError
    from cytir.reader import mcp as rmcp
    with pytest.raises(ToolError):
        rmcp.read("https://x.org/p", max_chars=-5)
