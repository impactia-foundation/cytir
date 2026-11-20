"""`cytir.core.internal_http.public_self_urlopen`: the operator's own public endpoint (task 2162)."""
from __future__ import annotations

import http.server
import threading
import urllib.error

import pytest

from cytir.core import internal_http as ih


class _H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/away":
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{self.server.server_port}/ok")
            self.end_headers()
        elif self.path == "/same":
            self.send_response(302)
            self.send_header("Location", "/ok")
            self.end_headers()
        elif self.path == "/gone":
            self.send_response(404)
            self.end_headers()
        else:
            body = b'{"ua": "%s"}' % self.headers.get("User-Agent", "").encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


@pytest.fixture
def server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


def test_loopback_http_ok_with_fixed_user_agent(server):
    with ih.public_self_urlopen(server + "/ok", timeout=5, caller="t") as r:
        assert b"aw-t/1" in r.read()


def test_timeout_is_required():
    with pytest.raises(TypeError):
        ih.public_self_urlopen("https://example.org/")  # type: ignore[call-arg]


@pytest.mark.parametrize("url", ["http://api.selfhost.example/health", "http://10.0.0.5/health",
                                 "ftp://example.org/x", "file:///etc/passwd", "https:///nohost"])
def test_non_https_non_loopback_refused(url):
    with pytest.raises(ih.PublicSelfRefused):
        ih.public_self_urlopen(url, timeout=1)


def test_https_passes_the_scheme_check_for_any_host():
    assert ih._check_public_self("https://api.selfhost.example/x") == "api.selfhost.example"


def test_cross_host_redirect_refused(server):
    # 127.0.0.1 -> localhost is another host name: refused
    with pytest.raises(ih.PublicSelfRefused):
        ih.public_self_urlopen(server + "/away", timeout=5)


def test_same_host_redirect_followed(server):
    with ih.public_self_urlopen(server + "/same", timeout=5) as r:
        assert r.status == 200


def test_non_2xx_raises_http_error(server):
    with pytest.raises(urllib.error.HTTPError):
        ih.public_self_urlopen(server + "/gone", timeout=5)
