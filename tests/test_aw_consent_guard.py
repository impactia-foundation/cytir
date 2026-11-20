"""Consent request choke point, step 1 (task 1590): guard transport + factory.

Fake transport only: every inner transport counts its calls; no socket is ever opened.
"""
import asyncio
import json

import httpx
import pytest

from cytir.consent import block
from cytir.consent.block import ConsentBlocked, ConsentRegistryError
from cytir.consent.guard import (ConsentGuardAsyncTransport, ConsentGuardTransport,
                              guard_client, is_guarded)
from cytir.core import user_agent
from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client


class Inner(httpx.BaseTransport):
    def __init__(self, handler=None):
        self.calls = 0
        self.handler = handler or (lambda req: httpx.Response(200, text="ok"))

    def handle_request(self, request):
        self.calls += 1
        return self.handler(request)


@pytest.fixture(autouse=True)
def registry(tmp_path, monkeypatch):
    path = tmp_path / "reg.json"
    path.write_text(json.dumps({"hosts": {
        "declined.example": {"state": "declined"},
        "open.example": {"state": "cleared"},
    }}))
    monkeypatch.setenv("AW_CONSENT_REGISTRY", str(path))
    monkeypatch.setattr(user_agent, "user_agent", lambda component=None: "test-agent")
    monkeypatch.setattr(user_agent, "pack_fetch_user_agent", lambda: "test-pack-agent")
    return path


def test_refuses_declined_host_without_touching_inner():
    inner = Inner()
    client = make_client(transport=inner)
    with pytest.raises(ConsentBlocked):
        client.get("https://declined.example/x")
    with pytest.raises(ConsentBlocked):          # a parent-domain decline covers subdomains
        client.get("https://www.declined.example/x")
    assert inner.calls == 0


def test_passes_cleared_and_unlisted_hosts():
    inner = Inner()
    client = make_client(transport=inner)
    assert client.get("https://open.example/x").status_code == 200
    assert client.get("https://unlisted.example/x").status_code == 200
    assert inner.calls == 2


def test_redirect_onto_declined_host_refused_on_second_hop():          # spec test 4
    def handler(req):
        if req.url.host == "open.example":
            return httpx.Response(302, headers={"location": "https://declined.example/y"})
        return httpx.Response(200, text="should not be reached")
    inner = Inner(handler)
    client = make_client(HttpSpec(follow_redirects=True), transport=inner)
    with pytest.raises(ConsentBlocked):
        client.get("https://open.example/x")
    assert inner.calls == 1


def test_cleared_host_one_lookup_for_n_requests(monkeypatch):          # spec test 5
    lookups = []
    real = block.lookup
    monkeypatch.setattr(block, "lookup", lambda h, p=None: (lookups.append(h), real(h, p))[1])
    client = make_client(transport=Inner())
    for _ in range(5):
        client.get("https://open.example/x")
    assert len(lookups) == 1


def test_unreadable_registry_fails_closed(tmp_path, monkeypatch):      # spec test 6
    monkeypatch.setenv("AW_CONSENT_REGISTRY", str(tmp_path / "missing.json"))
    inner = Inner()
    client = make_client(transport=inner)
    with pytest.raises(ConsentRegistryError):
        client.get("https://open.example/x")
    assert inner.calls == 0


def test_blocked_host_is_not_cached_as_cleared():
    inner = Inner()
    client = make_client(transport=inner)
    for _ in range(2):
        with pytest.raises(ConsentBlocked):
            client.get("https://declined.example/x")
    assert inner.calls == 0


def test_factory_wraps_transport_and_applies_spec():
    client = make_client(HttpSpec(headers={"x-a": "b"}, follow_redirects=False,
                                  user_agent_kind="pack_fetch"), transport=Inner())
    assert is_guarded(client)
    assert client.headers["user-agent"] == "test-pack-agent"
    assert client.headers["x-a"] == "b"
    assert client.follow_redirects is False
    assert make_client(transport=Inner()).headers["user-agent"] == "test-agent"


def test_factory_default_transport_is_guarded_without_network():
    client = make_client()
    assert isinstance(client._transport, ConsentGuardTransport)
    with pytest.raises(ConsentBlocked):          # refused before any socket is opened
        client.get("https://declined.example/x")


def test_factory_rejects_curl_kind_until_step_two():
    with pytest.raises(ValueError):
        make_client(HttpSpec(kind="curl"))


def test_guard_client_wraps_plain_client_once():
    inner = Inner()
    plain = httpx.Client(transport=inner)
    guard_client(plain)
    guard_client(plain)
    assert is_guarded(plain)
    assert not isinstance(plain._transport._inner, ConsentGuardTransport)
    with pytest.raises(ConsentBlocked):
        plain.get("https://declined.example/x")
    assert inner.calls == 0
    assert plain.get("https://open.example/").status_code == 200


def test_factory_client_not_double_wrapped_by_guard_client():
    client = make_client(transport=Inner())
    first = client._transport
    guard_client(client)
    assert client._transport is first


def test_async_guard_refuses_before_inner():
    class AInner(httpx.AsyncBaseTransport):
        calls = 0

        async def handle_async_request(self, request):
            AInner.calls += 1
            return httpx.Response(200)

    async def run():
        async with httpx.AsyncClient(transport=ConsentGuardAsyncTransport(AInner())) as c:
            with pytest.raises(ConsentBlocked):
                await c.get("https://declined.example/x")
            assert (await c.get("https://open.example/x")).status_code == 200
    asyncio.run(run())
    assert AInner.calls == 1
