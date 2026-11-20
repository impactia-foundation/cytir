"""Plain HTTP fetch for the local reader (task 2710).

robots.txt is NOT consulted. A reader that fetches one URL for one user, on the user's own machine, is a user
agent, not a crawler: the user could open the same URL in a browser. Crawl consent (ADR-024) governs the index
build, not this call. The User-Agent is honest (config.USER_AGENT) and names the reader.

Every request hop is checked (`check_url` runs in an httpx request hook, so a redirect to a search engine, a
non-http scheme or a private address is refused). The host is resolved (getaddrinfo + inet_aton shorthand) and refused when any answer is private; see
_private_target for the remaining rebinding race.
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from cytir.core.search_engines import is_search_engine

from . import config


class FetchRefused(ValueError):
    """The URL is not one the reader fetches (scheme, search engine, private target)."""


class FetchError(RuntimeError):
    """The fetch failed (network, HTTP status, size)."""


@dataclass
class Fetched:
    final_url: str
    status: int
    content_type: str
    html: str


def _blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    return (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_reserved
            or ip.is_multicast)


def _addresses(host: str) -> set[str]:
    """Every address `host` can mean: the resolver's answers plus the dotted/decimal/hex shorthand `inet_aton` reads
    (`127.1`, `2130706433`, `0x7f.0.0.1` are loopback to the OS but not valid to `ipaddress`)."""
    out: set[str] = set()
    try:
        out.add(str(ipaddress.ip_address(host)))   # a canonical literal needs no resolver
    except ValueError:
        pass
    try:
        out.add(socket.inet_ntoa(socket.inet_aton(host)))
    except OSError:
        pass
    try:
        out.update(i[4][0].split("%")[0] for i in socket.getaddrinfo(host, None))
    except (socket.gaierror, UnicodeError, OSError) as e:
        if not out:
            raise FetchRefused(f"cannot resolve host {host!r} ({e}); refusing to read it") from e
    return out


def _private_target(host: str) -> bool:
    """True when `host` is `localhost` or ANY address it resolves to is private/loopback/link-local/reserved/
    multicast. Resolving here also catches a public name that points at a private address. Remaining gap: the HTTP
    client resolves again when it connects, so a resolver that answers differently the second time (a rebinding
    race) is narrowed, not closed."""
    if host == "localhost" or host.endswith(".localhost"):
        return True
    return any(_blocked_ip(ipaddress.ip_address(a)) for a in _addresses(host.strip("[]")))


def check_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise FetchRefused(f"only http(s) URLs are read, got {url!r}")
    if is_search_engine(url):
        raise FetchRefused("search-engine result pages are not read: use Cytir `search`, then read the source URL")
    if _private_target((parts.hostname or "").lower()):
        raise FetchRefused("private, loopback and link-local addresses are not read")


def _check_request(request: httpx.Request) -> None:
    check_url(str(request.url))


def fetch(url: str, transport: httpx.BaseTransport | None = None) -> Fetched:
    """GET `url` (fragment dropped on the wire), following redirects; refuse >MAX_BYTES."""
    check_url(url)
    try:
        with httpx.Client(follow_redirects=True, timeout=config.FETCH_TIMEOUT_S, transport=transport,
                          event_hooks={"request": [_check_request]},
                          headers={"User-Agent": config.USER_AGENT, "Accept": "text/html,*/*;q=0.5"}) as c:
            with c.stream("GET", url) as r:
                buf = bytearray()
                for chunk in r.iter_bytes():
                    buf += chunk
                    if len(buf) > config.MAX_BYTES:
                        raise FetchError(f"response over {config.MAX_BYTES} bytes")
                enc = r.encoding or "utf-8"
                ctype = r.headers.get("content-type", "")
                status, final = r.status_code, str(r.url)
    except httpx.HTTPError as e:
        raise FetchError(f"{type(e).__name__}: {e}") from e
    if status >= 400:
        raise FetchError(f"HTTP {status}")
    return Fetched(final, status, ctype, bytes(buf).decode(enc, errors="replace"))
