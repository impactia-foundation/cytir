"""One stdlib-shaped GET/POST door from the operator tools (index, ops, gates, serve, ...) to the operator's own services.

Lives in `cytir.core` (task 1701, ruling 60(c)): one door, one name, import-light (`httpx` is imported
lazily). It started in `cytir/index/` (task 1648) and moved to `cytir/sources/` (task 1673); both old
names are gone and no alias remains.

The operator tools talk to Qdrant, the AW API and TEI, never to an origin web host (spec
`consent-request-choke-point` step 5, task 1648). They used the stdlib opener; this module
sends the same requests through `cytir.sources.http.make_internal_client`, which refuses any host that
is neither loopback nor a configured service URL (`InternalHostRefused`, a `ValueError`).

Error shape is kept so the callers' `except` clauses stay as they were: a non-2xx answer raises
`urllib.error.HTTPError` (body readable with `.read()`), a transport failure raises
`urllib.error.URLError`. Redirects are NOT followed (an internal service does not redirect, and a
followed redirect could leave the internal host); a 3xx therefore raises `HTTPError`.

`public_self_urlopen` (task 2162) is the sibling door for the operator's OWN PUBLIC endpoint (the host
is the operator's flag, e.g. `--url`, `AW_HEALTH_HOST`, `--api`; never a constant). It is stdlib-only.
"""
from __future__ import annotations

import io
import json
import urllib.error

__all__ = ["send", "fetch", "fetch_json", "urlopen", "public_self_urlopen", "PublicSelfRefused"]


def make_internal_client(base_url: str, *, timeout: float = 10.0):
    """`cytir.sources.http.make_internal_client`, imported on first use.

    Importing this door must not pull `httpx`: stdlib-only entry points (the drain, the settings
    probe) import modules that now route through it, and they run where only the stdlib exists
    until a request is actually sent."""
    from cytir.sources import http
    return http.make_internal_client(base_url, timeout=timeout)


def send(url: str, *, body: dict | None = None, method: str | None = None,
         timeout: float = 30.0, headers: dict[str, str] | None = None,
         data: bytes | None = None) -> tuple[int, bytes]:
    """(status, body) for one request, whatever the status. Transport failure -> `URLError`.

    `data` is a pre-encoded request body (as `urllib.request.Request(data=...)` carried it)."""
    import httpx
    verb = method or ("POST" if (body is not None or data is not None) else "GET")
    parsed = httpx.URL(url)
    origin = f"{parsed.scheme}://{parsed.netloc.decode('ascii')}"
    hdrs = dict(headers or {})
    content = data
    if body is not None:
        content = json.dumps(body).encode()
        hdrs.setdefault("content-type", "application/json")
    with make_internal_client(origin, timeout=timeout) as client:
        try:
            r = client.request(verb, url, content=content, headers=hdrs)
        except httpx.TimeoutException as exc:
            # urllib surfaced a timeout as URLError(reason=TimeoutError); callers retry on that shape
            raise urllib.error.URLError(TimeoutError(str(exc))) from exc
        except httpx.HTTPError as exc:
            raise urllib.error.URLError(str(exc)) from exc
    return r.status_code, r.content


def fetch(url: str, *, body: dict | None = None, method: str | None = None,
          timeout: float = 30.0, headers: dict[str, str] | None = None,
          data: bytes | None = None) -> bytes:
    """The response body of a 2xx answer; any other status raises `urllib.error.HTTPError`."""
    status, content = send(url, body=body, method=method, timeout=timeout, headers=headers, data=data)
    if not 200 <= status < 300:
        raise urllib.error.HTTPError(url, status, f"HTTP {status}", {}, io.BytesIO(content))
    return content


def fetch_json(url: str, *, body: dict | None = None, method: str | None = None,
               timeout: float = 30.0, headers: dict[str, str] | None = None):
    return json.loads(fetch(url, body=body, method=method, timeout=timeout, headers=headers))


def urlopen(target, timeout: float = 30.0) -> io.BytesIO:
    """A stdlib-urlopen-shaped read through the internal client.

    `target` is a URL or a `urllib.request.Request` (its method, data and headers are kept). The
    result is a `BytesIO`: a context manager with `.read()`, so `json.load(r)` and `with ... as r`
    callers stay as they were. Non-2xx -> `HTTPError`, transport failure -> `URLError`; a host that
    is neither loopback nor a configured service -> `InternalHostRefused`."""
    if isinstance(target, str):
        return io.BytesIO(fetch(target, timeout=timeout))
    return io.BytesIO(fetch(target.full_url, data=target.data, method=target.get_method(),
                            timeout=timeout, headers=dict(target.header_items())))


# ---------------------------------------------------------------------------------------------
# The operator's own PUBLIC endpoint (task 2162). The host is chosen by the CALLER (the operator's
# flag or env), so a self-hoster runs the gates against their own domain. The helper does not pin a
# host (that is `cytir.probe.run`'s job: the probe must never leave websear.ch); it fixes the rest:
# https only (loopback may use http, for local stacks and tests), no cross-host redirect, a fixed
# User-Agent, and a required timeout.
# ---------------------------------------------------------------------------------------------

_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


class PublicSelfRefused(ValueError):
    """The URL is not https and not loopback, or a redirect left the host."""


def _check_public_self(url: str) -> str:
    import urllib.parse
    u = urllib.parse.urlsplit(url)
    host = (u.hostname or "").lower()
    if not host:
        raise PublicSelfRefused(f"no host in {url!r}")
    if u.scheme == "https" or (u.scheme == "http" and host in _LOOPBACK):
        return host
    raise PublicSelfRefused(f"{u.scheme}://{host}: public-self needs https (http only on loopback)")


def public_self_urlopen(url: str, *, timeout: float, data: bytes | None = None,
                        headers: dict[str, str] | None = None, caller: str = "aw"):
    """A stdlib-urlopen-shaped GET (POST when `data` is given) to the operator's own public endpoint.

    Returns the `http.client.HTTPResponse` (a context manager with `.read()`), as `urllib.request.urlopen`
    did, so callers parse as before. Non-2xx -> `urllib.error.HTTPError`, transport failure -> `URLError`.
    A redirect to another host (or to a non-https non-loopback URL) -> `PublicSelfRefused`."""
    import urllib.request
    host = _check_public_self(url)

    class _SameHost(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, hdrs, newurl):
            if _check_public_self(newurl) != host:
                raise PublicSelfRefused(f"redirect from {host} to another host refused")
            return super().redirect_request(req, fp, code, msg, hdrs, newurl)

    hdrs = {"User-Agent": f"aw-{caller}/1 (operator self-check)"}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=hdrs)
    return urllib.request.build_opener(_SameHost).open(req, timeout=timeout)  # noqa: S310 (scheme+host checked above)
