"""Ordered mirror list for a pinned model file (task 2776): fall back on TLS/HTTP failure,
refuse on a sha256 mismatch, say which source served. The network is mocked."""
from __future__ import annotations

import hashlib
import ssl
from pathlib import Path

import httpx
import pytest

from cytir.embed import fetch_models
from cytir.embed.model_assets import LID_176, ModelAsset

FULL = b"authoritative-web-fake-model-bytes" * 4
WRONG = bytes(len(FULL))            # same size, other bytes
U1, U2 = "https://primary.invalid/m.bin", "https://mirror.invalid/m.bin"


class _Resp:
    def __init__(self, body: bytes, status: int = 200):
        self.body, self.status = body, status

    def raise_for_status(self):
        if self.status >= 400:
            req = httpx.Request("GET", "https://x.invalid")
            raise httpx.HTTPStatusError("bad", request=req, response=httpx.Response(self.status, request=req))

    def iter_bytes(self, _n):
        yield self.body


class _Ctx:
    def __init__(self, v):
        self.v = v

    def __enter__(self):
        return self.v

    def __exit__(self, *e):
        return False


class _Client:
    def __init__(self, script, seen):
        self.script, self.seen = script, seen

    def stream(self, _m, url):
        self.seen.append(url)
        out = self.script[url]
        if isinstance(out, Exception):
            raise out
        return _Ctx(out)


def _asset(dest: Path) -> ModelAsset:
    a = ModelAsset(name="mirror-test", filename="m.bin", url=U1, mirrors=(U2,),
                   sha256=hashlib.sha256(FULL).hexdigest(), size_bytes=len(FULL),
                   env_var="AW_TEST_M", why="fixture")
    a.resolve = lambda: (str(dest), False)
    a.default_dest = lambda: dest
    return a


def _run(monkeypatch, tmp_path, script):
    seen: list[str] = []
    monkeypatch.setattr(fetch_models, "make_client", lambda _s: _Ctx(_Client(script, seen)))
    dest = tmp_path / "m.bin"
    ok = fetch_models.handle(_asset(dest), check_only=False, force=False)
    return ok, dest, seen


def _tls():
    return httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate has expired")


def test_tls_failure_falls_back_to_mirror_and_names_it(tmp_path, monkeypatch, capsys):
    ok, dest, seen = _run(monkeypatch, tmp_path, {U1: _tls(), U2: _Resp(FULL)})
    assert ok and dest.read_bytes() == FULL
    assert seen == [U1, U2]                      # TLS is deterministic: one try, then the mirror
    cap = capsys.readouterr()
    assert "TLS error" in cap.err and "certificate has expired" in cap.err
    assert f"served by {U2}" in cap.out


def test_http_status_falls_back_and_is_named(tmp_path, monkeypatch, capsys):
    ok, dest, seen = _run(monkeypatch, tmp_path, {U1: _Resp(b"", 503), U2: _Resp(FULL)})
    assert ok and seen == [U1, U2]
    assert "HTTP 503" in capsys.readouterr().err


def test_ssl_error_in_cause_chain_counts_as_tls(tmp_path, monkeypatch):
    err = httpx.ConnectError("connect failed")
    err.__cause__ = ssl.SSLCertVerificationError("expired")
    ok, _d, seen = _run(monkeypatch, tmp_path, {U1: err, U2: _Resp(FULL)})
    assert ok and seen == [U1, U2]


def test_sha_mismatch_is_a_refusal_not_a_silent_next_mirror(tmp_path, monkeypatch, capsys):
    ok, dest, seen = _run(monkeypatch, tmp_path, {U1: _Resp(WRONG), U2: _Resp(FULL)})
    assert ok is False
    assert seen == [U1]                          # the mirror is never asked
    assert not dest.exists() and not (tmp_path / "m.bin.part").exists()
    assert "sha256 mismatch" in capsys.readouterr().err


def test_all_sources_failing_reports_every_cause(tmp_path, monkeypatch, capsys):
    ok, dest, _seen = _run(monkeypatch, tmp_path, {U1: _tls(), U2: _Resp(b"", 404)})
    assert ok is False and not dest.exists()
    err = capsys.readouterr().err
    assert "every source failed" in err and "TLS error" in err and "HTTP 404" in err


def test_howto_lists_every_mirror_in_order():
    h = _asset(Path("/tmp/m.bin")).howto()
    assert h.index(U1) < h.index(U2)


def test_lid176_has_a_pinned_hf_mirror_and_url_is_first():
    assert LID_176.urls[0] == LID_176.url and len(LID_176.urls) >= 2
    assert "dl.fbaipublicfiles.com" in LID_176.urls[0]
    assert any(u.startswith("https://huggingface.co/") for u in LID_176.urls[1:])
    for u in LID_176.urls:
        assert u in LID_176.howto()


def test_consent_refusal_propagates_and_skips_the_next_mirror(tmp_path, monkeypatch):
    from cytir.consent.block import ConsentBlocked
    seen: list[str] = []
    script = {U1: ConsentBlocked("declined host"), U2: _Resp(FULL)}
    monkeypatch.setattr(fetch_models, "make_client", lambda _s: _Ctx(_Client(script, seen)))
    with pytest.raises(ConsentBlocked):
        fetch_models._download(_asset(tmp_path / "m.bin"), tmp_path / "m.bin")
    assert seen == [U1] and not (tmp_path / "m.bin").exists()
