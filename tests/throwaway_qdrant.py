"""The ONE gate every Qdrant-backed test passes before it sends its first write or delete.

Review of #553 (round 4, F1, HIGH): the drain-test harness refused a live endpoint by a list of
PORTS. A tunnel to prod on a port outside the list passed, and the next `reset_collection(url,
"aw_e9")` was a `DELETE` of prod's physical collection. A port list is a list of spellings; the
fix is the one the production code already took (`cytir.index.live_target`): ask the endpoint what it
HOLDS.

An endpoint is a throwaway only when ALL of this holds, read before any write:

  1. its listing can be read (an unreadable endpoint is not proven harmless: fail closed);
  2. it holds NO alias (a prod alias, or any alias at all, means somebody serves from it);
  3. it holds NO collection with a declared live name (`live_target.live_collection_names()`);
  4. it carries a MARKER: the sentinel collection `SENTINEL`. The marker is put there by a test
     session, and ONLY on an EMPTY instance (an instance that holds anything and has no marker is
     refused). A fresh CI Qdrant is empty, so its first test session marks it; a local instance a
     developer reuses is marked from then on; a prod, a twin or any instance that already holds
     data never is.

`require_throwaway(url)` does the check once per session and URL and returns the URL; it is the
only way the shared helpers (the drain-test harness, the serving-profile tests,
the `cyt serve` tests) obtain one. `assert_proven(url)` is the second belt inside every helper that
creates, recreates, clears or deletes a collection. The port list stays as a third check only.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pytest

#: The collection that marks an instance as one a test session made. Not a live name.
SENTINEL = "aw_test_throwaway_sentinel"
#: Third belt only: ports of the live boxes (the live boxes' services and tunnels).
#: 6486 is the twin index's port on the build box (task 2940: it was missing from this list).
LIVE_PORTS = {6333, 6334, 6398, 6400, 6486, 16333, *range(6457, 6473)}

#: Task 2949: the denylist above misses every live port nobody listed. On loopback these two
#: ranges (the repo's live Qdrant services and tunnels sit in them) are refused for EVERY connect
#: unless the endpoint is the session's own throwaway (`allowed`). The ONE definition of the ranges.
QDRANT_RANGES = (range(6300, 6500), range(16300, 16500))
_LOOPBACK_NAMES = {"localhost", "::1", "0.0.0.0", "::", "0:0:0:0:0:0:0:1"}


def is_loopback(host) -> bool:
    h = str(host).lower()
    return h in _LOOPBACK_NAMES or h.startswith("127.") or h.startswith("::ffff:127.")


def in_ranges(port) -> bool:
    return any(port in r for r in QDRANT_RANGES)


def endpoint_of(url: str | None):
    """`(host, port)` of a throwaway URL, or None."""
    if not url:
        return None
    u = urllib.parse.urlparse(url)
    if u.hostname is None or u.port is None:
        return None
    return (u.hostname, u.port)


def allowed_from_env() -> frozenset:
    ep = endpoint_of(os.environ.get("AW_TEST_QDRANT_URL"))
    return frozenset([ep]) if ep else frozenset()


def refused(host, port, allowed=frozenset()) -> bool:
    """True when a connect to `host:port` must be refused before any socket opens: a listed live
    port on ANY host, or a loopback port in `QDRANT_RANGES` that is not an `allowed` endpoint
    (loopback spellings match each other: the throwaway `localhost:N` allows `127.0.0.1:N`)."""
    if port in LIVE_PORTS:
        return True
    if is_loopback(host) and in_ranges(port):
        return not any(is_loopback(h) and p == port for h, p in allowed)
    return False


_PROVEN: set[str] = set()


class ThrowawayRefused(AssertionError):
    """The endpoint is not provably a throwaway Qdrant: nothing was written."""


def _norm(url: str) -> str:
    return url.rstrip("/")


def _read_holdings(url: str):
    from cytir.index import live_target as LT
    try:
        return LT.read_holdings(url), LT
    except LT.LiveTargetRefused as exc:
        raise ThrowawayRefused(f"{url}: not provably a throwaway Qdrant, nothing written: {exc}") from exc


def prove_throwaway(url: str) -> str:
    """Read what `url` holds and refuse unless it is provably a throwaway; mark an EMPTY instance.
    Sends GETs, and ONE PUT (the sentinel) only on an instance that holds nothing at all."""
    url = _norm(url)
    (aliases, held), LT = _read_holdings(url)
    if aliases:
        raise ThrowawayRefused(f"{url} holds aliases {sorted(aliases)}: somebody serves from it, "
                               "it is not a throwaway. Nothing written")
    live = sorted(set(held) & LT.live_collection_names())
    if live:
        raise ThrowawayRefused(f"{url} holds the declared live collection(s) {live}: it is not a "
                               "throwaway. Nothing written")
    if SENTINEL not in held:
        if held:
            raise ThrowawayRefused(
                f"{url} holds {sorted(held)} and no `{SENTINEL}` marker: a non-empty, unmarked "
                "instance is refused. Use a fresh, empty Qdrant. Nothing written")
        req = urllib.request.Request(
            f"{url}/collections/{SENTINEL}", method="PUT",
            data=json.dumps({"vectors": {"size": 2, "distance": "Cosine"}}).encode(),
            headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=60):
            pass
    _PROVEN.add(url)
    # the prod gate must not read this endpoint's `aw_e9` / `aw_prod` as the serving box
    from cytir.index import prod_guard as PG
    PG.mark_throwaway(url)
    return url


def assert_proven(url: str) -> None:
    """The second belt: a helper that writes calls this first. Raises (sends nothing) for an URL
    that `require_throwaway` has not proven in this session."""
    if _norm(url) not in _PROVEN:
        raise ThrowawayRefused(f"{url} was not proven a throwaway in this session: refusing to "
                               "write. Obtain the URL from throwaway_qdrant.require_throwaway()")


def require_throwaway(url: str | None = None) -> str:
    """The URL of the session's throwaway Qdrant (`AW_TEST_QDRANT_URL`), proven. Skips without one,
    and FAILS (never skips) under `AW_REQUIRE_QDRANT=1` (the CI job)."""
    url = url if url is not None else os.environ.get("AW_TEST_QDRANT_URL")
    if not url:
        if os.environ.get("AW_REQUIRE_QDRANT") == "1":
            pytest.fail("AW_REQUIRE_QDRANT=1 but AW_TEST_QDRANT_URL is not set")
        pytest.skip("no AW_TEST_QDRANT_URL: these tests need a throwaway Qdrant (CI job / local binary)")
    u = urllib.parse.urlparse(url)
    assert u.hostname in ("127.0.0.1", "localhost"), f"{url}: only a local throwaway Qdrant"
    assert u.port not in LIVE_PORTS, f"{url}: that port is a live Qdrant"
    assert not in_ranges(u.port), (f"{url}: a throwaway never listens in {QDRANT_RANGES} (the ports of "
                                   "real Qdrant services); use a free ephemeral port")
    if _norm(url) in _PROVEN:
        return _norm(url)
    return prove_throwaway(url)
