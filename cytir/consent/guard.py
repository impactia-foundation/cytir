"""The request-layer consent guard (task 1590, spec `consent-request-choke-point` section 1).

`ConsentGuardTransport(inner)` wraps any `httpx.BaseTransport`. Before the inner transport
sees a request, the request host is checked against the recorded-decline registry
(`cytir.consent.block.assert_crawlable`). A blocked host raises `ConsentBlocked`; an unreadable
registry raises `ConsentRegistryError`. Both fail closed, before any socket opens.

httpx re-enters `handle_request` for every redirect hop, so a redirect onto a declined host is
refused too. A host that passed is remembered per transport instance (one lookup per host).
This module changes no consent semantics: it only moves the existing check to the request.
"""
from __future__ import annotations

import httpx

from cytir.consent.block import assert_crawlable

__all__ = ["ConsentGuardTransport", "ConsentGuardAsyncTransport", "guard_client", "is_guarded"]


class ConsentGuardTransport(httpx.BaseTransport):
    def __init__(self, inner: httpx.BaseTransport):
        self._inner = inner
        self._cleared: set[str] = set()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = (request.url.host or "").lower()
        if host not in self._cleared:
            assert_crawlable(host)          # raises ConsentBlocked / ConsentRegistryError
            self._cleared.add(host)
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


class ConsentGuardAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport):
        self._inner = inner
        self._cleared: set[str] = set()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = (request.url.host or "").lower()
        if host not in self._cleared:
            assert_crawlable(host)
            self._cleared.add(host)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def is_guarded(client: httpx.Client) -> bool:
    return isinstance(getattr(client, "_transport", None), ConsentGuardTransport)


def guard_client(client: httpx.Client) -> httpx.Client:
    """Re-wrap an existing client's transport in the guard (idempotent). Returns the client."""
    if not is_guarded(client):
        client._transport = ConsentGuardTransport(client._transport)
    return client
