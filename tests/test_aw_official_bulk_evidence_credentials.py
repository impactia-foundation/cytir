"""Task 2192: a credentialed fetch leaves no secret in any sealed evidence file, log or error."""
import base64
import gzip
import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import cytir.sources.official_bulk.bulk as bulk
import cytir.sources.official_bulk.direct as direct
import cytir.sources.official_bulk.http_evidence as subject
from cytir.acquire.artifact_io import sha256_file

SECRET = "s3cr3t-2192"
COOKIE = "sess-" + SECRET


class _Stub(BaseHTTPRequestHandler):
    def do_GET(self):
        want = "Basic " + base64.b64encode(f"alice:{SECRET}".encode()).decode()
        if self.headers.get("Authorization") != want:
            self.send_response(401)
            self.end_headers()
            return
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Set-Cookie", f"session={COOKIE}; HttpOnly")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture()
def stub():
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://alice:{SECRET}@127.0.0.1:{server.server_port}/dump.json"
    server.shutdown()


def _tree_text(root: Path) -> str:
    """Every byte under root, plus the gunzipped content of each file, as latin-1 text."""
    out = []
    for p in root.rglob("*"):
        if p.is_file():
            raw = p.read_bytes()
            out.append(raw.decode("latin-1"))
            try:
                out.append(gzip.decompress(raw).decode("latin-1"))
            except OSError:
                pass
    return "\n".join(out)


def _fetch(url: str, tmp: Path):
    body, headers, metrics = tmp / "body", tmp / "headers", tmp / "metrics"
    with open(metrics, "wb") as out:
        subprocess.run(["curl", "--fail-with-body", "--silent", "--show-error", "--dump-header",
                        str(headers), "--write-out", "%{json}", url, "--output", str(body)],
                       stdout=out, check=True)
    return body, headers, metrics


@pytest.mark.skipif(shutil.which("curl") is None, reason="curl missing")
@pytest.mark.parametrize("seal", [bulk, direct])
def test_credentialed_fetch_leaves_no_secret_on_disk(stub, tmp_path, capsys, seal):
    body, headers, metrics = _fetch(stub, tmp_path)
    assert COOKIE in headers.read_text()  # the stub really sent the cookie
    meta = tmp_path / "meta.json"
    rc = subject.main(["--body", str(body), "--headers", str(headers), "--curl-metrics", str(metrics),
                       "--requested-url", stub, "--fetched-at", "2026-11-01T00:00:00Z",
                       "--out", str(meta)])
    assert rc == 0
    extra = {"country": "ch"} if seal is bulk else {"artifact_root": tmp_path, "license_name": "x"}
    seal.acquire(body=body, headers=headers, metadata=meta, root=tmp_path / "root", source_id="s",
                 source_name="n", requested_url=stub, content_scope="c", next_projection="p",
                 fetch_vantage="v", **extra)
    captured = capsys.readouterr()
    everything = _tree_text(tmp_path) + captured.out + captured.err
    assert SECRET not in everything and COOKIE not in everything
    assert "set-cookie: <redacted>" in headers.read_text().lower()
    assert json.loads(meta.read_text())["requested_url"].startswith("http://<redacted>@127.0.0.1:")


def test_redact_helpers():
    assert subject.redact_url("https://u:p@h.test/x?a=1") == "https://<redacted>@h.test/x?a=1"
    assert subject.redact_url("https://h.test/a@b") == "https://h.test/a@b"
    raw = (b"HTTP/1.1 200 OK\r\nAuthorization: Basic abc\r\nPROXY-AUTHORIZATION: x\r\n"
           b"Cookie: a=b\r\nSet-Cookie: c=d\r\nContent-Type: text/plain\r\n\r\n")
    out = subject.redact_header_dump(raw)
    assert b"abc" not in out and b"a=b" not in out and b"c=d" not in out
    assert b"PROXY-AUTHORIZATION: <redacted>\r\n" in out and b"Content-Type: text/plain\r\n" in out
    assert subject.redact_header_dump(out) == out


def test_error_text_hides_userinfo(tmp_path):
    (tmp_path / "b").write_bytes(b"x")
    (tmp_path / "h").write_bytes(b"x")
    metrics = tmp_path / "m"
    metrics.write_text(json.dumps({"http_code": 500, "url_effective": f"https://u:{SECRET}@h.test/"}))
    with pytest.raises(ValueError) as err:
        subject.record(body=tmp_path / "b", headers=tmp_path / "h", curl_metrics=metrics,
                       requested_url=f"https://u:{SECRET}@h.test/", fetched_at="t",
                       output=tmp_path / "o")
    assert SECRET not in str(err.value)
    assert SECRET not in metrics.read_text()


REDIRECT = f"https://bob:{SECRET}@elsewhere.test/x"


class _Redirect(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(302)
        self.send_header("Location", REDIRECT)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


@pytest.mark.skipif(shutil.which("curl") is None, reason="curl missing")
def test_redirect_location_userinfo_is_redacted(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), _Redirect)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/r"
        body, headers, metrics = tmp_path / "body", tmp_path / "headers", tmp_path / "metrics"
        with open(metrics, "wb") as out:
            subprocess.run(["curl", "--silent", "--dump-header", str(headers), "--write-out",
                            "%{json}", url, "--output", str(body)], stdout=out, check=True)
    finally:
        server.shutdown()
    assert SECRET in headers.read_text()
    with pytest.raises(ValueError):  # a 302 is not a success, but redaction runs first
        subject.record(body=body, headers=headers, curl_metrics=metrics, requested_url=url,
                       fetched_at="t", output=tmp_path / "o")
    text = headers.read_text()
    assert SECRET not in text
    assert "location: https://<redacted>@elsewhere.test/x" in text.lower()


def test_header_dump_folds_and_location():
    raw = (b"HTTP/1.1 200 OK\r\nSet-Cookie: a=1;\r\n b=" + SECRET.encode() + b"\r\n"
           b"\tmore=" + SECRET.encode() + b"\r\nX-Other: ok\r\n"
           b"Content-Location: https://u:" + SECRET.encode() + b"@h.test/y\r\n"
           b"Location: https://h.test/plain\r\n\r\n")
    out = subject.redact_header_dump(raw)
    assert SECRET.encode() not in out
    assert b"Set-Cookie: <redacted>\r\nX-Other: ok\r\n" in out
    assert b"Content-Location: https://<redacted>@h.test/y\r\n" in out
    assert b"Location: https://h.test/plain\r\n" in out
    assert subject.redact_header_dump(out) == out


def test_credential_free_fixture_bytes_are_unchanged(tmp_path):
    headers_raw = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nLocation: https://h.test/a\r\n\r\n"
    metrics_raw = (json.dumps({"http_code": 200, "url_effective": "https://h.test/d"},
                              indent=2, sort_keys=True) + "\n").encode()
    body, headers, metrics = tmp_path / "b", tmp_path / "h", tmp_path / "m"
    body.write_bytes(b"xyz")
    headers.write_bytes(headers_raw)
    metrics.write_bytes(metrics_raw)
    out = tmp_path / "o"
    subject.record(body=body, headers=headers, curl_metrics=metrics,
                   requested_url="https://h.test/d", fetched_at="t", output=out)
    assert headers.read_bytes() == headers_raw and metrics.read_bytes() == metrics_raw
    expected = {
        "schema": "aw.http-fetch-metadata.v1", "requested_url": "https://h.test/d",
        "final_url": "https://h.test/d", "http_status": 200, "fetched_at": "t",
        "body": {"bytes": 3, "sha256": sha256_file(body)},
        "response_headers": {"encoding": "curl-dump-header-raw", "bytes": len(headers_raw),
                             "sha256": sha256_file(headers)}}
    assert out.read_bytes() == (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode()


@pytest.mark.parametrize("seal", [bulk, direct])
def test_reverify_of_existing_raw_headers_never_refuses_or_rewrites(tmp_path, seal):
    body, headers, meta = tmp_path / "b", tmp_path / "h", tmp_path / "m"
    body.write_bytes(b"x")
    headers.write_bytes(b"HTTP/1.1 200 OK\r\nSet-Cookie: k=public\r\n\r\n")
    before = headers.read_bytes()
    meta.write_text(json.dumps({
        "schema": "aw.http-fetch-metadata.v1", "requested_url": "u", "http_status": 200,
        "final_url": "u", "fetched_at": "t",
        "body": {"bytes": 1, "sha256": sha256_file(body)},
        "response_headers": {"bytes": headers.stat().st_size, "sha256": sha256_file(headers)}}))
    meta_before = meta.read_bytes()
    seal.check_transport(body=body, headers=headers, metadata=meta, requested_url="u")
    assert headers.read_bytes() == before and meta.read_bytes() == meta_before
