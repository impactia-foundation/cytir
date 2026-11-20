"""The ONE factory for package HTTP clients to origin hosts (task 1590, spec section 2).

`make_client(spec)` builds an `httpx.Client` from an `HttpSpec` (task 1556). Its transport is
ALWAYS wrapped in `cytir.consent.guard.ConsentGuardTransport`, so a declined host cannot be reached
through a factory client, redirects included. Tests pass `httpx.MockTransport` as `transport`;
it is wrapped too, so the guard is provable with no socket.

`--skip-consent` skips live resolution only (spec section 3); it has no hook here and the guard
cannot be turned off. The curl kind is added with the run_family migration (spec step 2).
"""
from __future__ import annotations

import httpx

from cytir.consent.guard import ConsentGuardTransport
from cytir.sources.family import HttpSpec

__all__ = ["make_client", "make_internal_client", "InternalHostRefused", "is_internal"]


def make_client(spec: HttpSpec | None = None, *,
                transport: httpx.BaseTransport | None = None) -> httpx.Client:
    spec = spec or HttpSpec()
    if spec.kind != "httpx":
        raise ValueError(f"make_client builds httpx clients only; kind={spec.kind!r} "
                         "arrives with the curl guard (spec migration step 2)")
    from cytir.core import user_agent
    ua = (user_agent.pack_fetch_user_agent() if spec.user_agent_kind == "pack_fetch"
          else user_agent.user_agent(spec.user_agent_component))
    guarded = ConsentGuardTransport(transport if transport is not None else httpx.HTTPTransport())
    return httpx.Client(headers={"user-agent": ua, **spec.headers}, timeout=spec.timeout,
                        follow_redirects=spec.follow_redirects, transport=guarded)


class InternalHostRefused(ValueError):
    """`make_internal_client` was given a host that is neither loopback nor a configured service."""


# `0.0.0.0` / `::` (the unspecified address) connect to THIS box on Linux: a spelling of loopback, so the
# drain's live-target review can read what such an endpoint holds and refuse it as live (task 1782).
_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "::"})
_SERVICE_KNOBS = ("AW_QDRANT_URL", "AW_TEI_URL", "AW_API_URL")


def _host(url: str) -> str:
    return (httpx.URL(url).host or "").lower().strip("[]")


def _internal_hosts() -> set[str]:
    """Loopback plus the hosts of the Qdrant, TEI and API URLs, read from settings at call time."""
    from cytir.serve import settings
    hosts = set(_LOOPBACK)
    for knob in _SERVICE_KNOBS:
        url = settings.current(knob)
        if url:
            hosts.add(_host(str(url)))
    return hosts


def make_internal_client(base_url: str, *, timeout: float = 10.0,
                         transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """The ONE client for the operator's own services (spec section 4, internal category).

    NO consent guard: Qdrant, TEI and the API are not origin hosts. It refuses any host that is
    not loopback and not the host of a configured service URL, so an origin URL cannot slip through
    the unguarded door. It sends the internal User-Agent, not the crawler identity (task 1782).
    """
    host = _host(base_url)
    if host not in _internal_hosts():
        raise InternalHostRefused(f"{host!r} is not loopback nor a configured internal service; "
                                  "use cytir.sources.http.make_client for origin hosts")
    from cytir.core import user_agent
    # An internal call is not crawling: it sends an internal UA and never consults the crawler identity
    # (which fails closed with `UnidentifiedCrawler` when unset). Task 1782.
    client = httpx.Client(base_url=base_url, headers={"user-agent": f"aw-internal/{user_agent.VERSION}"},
                          timeout=timeout, transport=transport if transport is not None else httpx.HTTPTransport())
    client._aw_internal = True
    return client


def is_internal(client: httpx.Client) -> bool:
    return getattr(client, "_aw_internal", False) is True
