"""make_internal_client: unguarded, loopback or configured-service hosts only (task 1623, spec section 4)."""
import httpx
import pytest

from cytir.consent.guard import is_guarded
from cytir.sources import http as H


def _mock():
    return httpx.MockTransport(lambda req: httpx.Response(200, json={"host": req.url.host}))


def test_loopback_accepted():
    for base in ("http://localhost:6333", "http://127.0.0.1:8080", "http://[::1]:8081"):
        c = H.make_internal_client(base, transport=_mock())
        assert H.is_internal(c)


def test_configured_qdrant_host_accepted(monkeypatch):
    from cytir.serve import settings
    real = settings.current
    monkeypatch.setattr(settings, "current",
                        lambda name, env=None: "http://qdrant.internal:6333" if name == "AW_QDRANT_URL" else real(name, env))
    c = H.make_internal_client("http://qdrant.internal:6333", transport=_mock())
    assert c.get("/x").json() == {"host": "qdrant.internal"}


def test_origin_host_refused():
    with pytest.raises(H.InternalHostRefused):
        H.make_internal_client("https://example.org", transport=_mock())


def test_client_is_not_guarded_and_reaches_transport(monkeypatch):
    from cytir.consent import guard
    c = H.make_internal_client("http://localhost:6333", transport=_mock())
    assert not is_guarded(c)
    def boom(host):
        raise AssertionError("guard consulted")
    monkeypatch.setattr(guard, "assert_crawlable", boom)
    assert c.get("/collections").status_code == 200


def test_operator_user_agent_sent():
    seen = {}
    def h(req):
        seen["ua"] = req.headers["user-agent"]
        return httpx.Response(200)
    H.make_internal_client("http://localhost:1", transport=httpx.MockTransport(h)).get("/")
    assert seen["ua"] and "httpx" not in seen["ua"].lower().split("/")[0]


def test_internal_call_needs_no_crawler_identity(monkeypatch):
    """Task 1782: loopback calls send `aw-internal/<v>` and never consult the crawler identity; an
    origin host is still refused."""
    from cytir.core import user_agent
    monkeypatch.delenv("AW_ALLOW_UNIDENTIFIED_CRAWL", raising=False)
    monkeypatch.setattr(user_agent, "operator", lambda: ("", "", False))
    seen = {}
    def h(req):
        seen["ua"] = req.headers["user-agent"]
        return httpx.Response(200)
    with H.make_internal_client("http://127.0.0.1:1", transport=httpx.MockTransport(h)) as c:
        c.get("/")
    assert seen["ua"].startswith("aw-internal/")
    with pytest.raises(user_agent.UnidentifiedCrawler):
        user_agent.user_agent()
    with pytest.raises(user_agent.UnidentifiedCrawler):  # a crawl client still needs the identity
        H.make_client(transport=_mock())
    with pytest.raises(H.InternalHostRefused):
        H.make_internal_client("https://example.org", transport=_mock())


def test_unspecified_address_is_a_loopback_spelling():
    for base in ("http://0.0.0.0:6333", "http://[::]:6333"):
        assert H.is_internal(H.make_internal_client(base, transport=_mock()))
