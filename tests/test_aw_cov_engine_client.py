"""The COV engine builds its HTTP clients through the package factory (task 2420)."""
from __future__ import annotations

import httpx

import cytir.sources.cov.engine as eng


def test_cov_client_request_shape(monkeypatch):
    monkeypatch.setenv("AW_OPERATOR_URL", "https://example.org/bot")
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "ops@example.org")
    seen = []
    monkeypatch.setattr(httpx, "HTTPTransport", lambda *a, **k: httpx.MockTransport(
        lambda r: seen.append(r) or httpx.Response(200)))
    monkeypatch.setattr("cytir.consent.guard.assert_crawlable", lambda h: None)
    for t in (90, 120):
        c = eng._cov_client(t)
        assert c.timeout == httpx.Timeout(t) and c.follow_redirects is True
        assert c.headers["user-agent"].startswith("AuthoritativeWebBot")
        assert c.get("https://example.org/x").status_code == 200
        c.close()
    assert [r.method for r in seen] == ["GET", "GET"]
    assert all(r.headers["user-agent"].startswith("AuthoritativeWebBot") for r in seen)
