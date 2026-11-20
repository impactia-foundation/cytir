"""The one gate every Qdrant call to the PROD target goes through (incident 2026-10-07).

THE INCIDENT. 2026-10-07 04:55Z-07:50Z the serving box's qdrant answered 187,000+ `POST
/collections/aw_e9/points/scroll` requests (hours 04-07: 4,423 / 61,890 / 62,779 / 43,547) from
the build box's system python, and search answered 503 for ~30 minutes. The client was a review
replay: it ran the real `promote_sealed_pack.plan()` with PROD as the target over packs that had
already landed. `already_served_urls` asks a filtered COUNT per URL batch and then SCROLLS every
batch that has a hit; a landed pack hits on every batch, so one 102k-URL tranche is thousands of
scrolls on a 30 GB box that also serves. Nothing refused it: a scroll is a "read" to the dry-run
guard.

THE RULES (this module is the only place they live)

  1. A SCROLL against a prod target is refused, unless the caller passes `allow_prod_scroll=True`
     AND a reason; the allowance is logged. Counts and id retrieves stay allowed: they are indexed.
  2. A REPLAY (`replay()` block, or `AW_REPLAY=1` in the environment) never queries prod at all:
     every `POST` to a prod target (retrieve, count, scroll, search) is refused, with no override.
     A replay takes the twin as both sides. Metadata `GET`s (`/aliases`, `/collections/x`) pass.
  3. Prod reads are paced: `PROD_LOOKUP_WORKERS` wide at most, `PROD_BATCH_SLEEP_S` between
     batches (`pace()`); the monitor saw 503s at 8 s of unpaced reads.

  4. A BUDGET. Outside a replay, one process sends at most `PROD_READ_BUDGET` (400) POST calls to
     prod (~200k ids at 500 per retrieve), with a loud line every 100 calls; the next call is
     refused. `AW_PROD_READ_BUDGET=unlimited` lifts it, and only the drain and compose unit
     drop-ins (the operator unit files under `deploy/systemd/`) set it.

WHAT "PROD" IS. A target is prod when ANY of these holds:
  * its collection is the serving alias (`aw_prod`), or a collection the payload-profile
    declaration marks `serving: true` (`data/collection-payload-profiles.v1.json`: today `aw_e9`),
    on ANY URL; or a name in `AW_PROD_COLLECTIONS` (comma list);
  * its URL is the prod tunnel (port 6334) or in `AW_PROD_URL`;
  * the endpoint's own `GET /aliases` lists `aw_prod` (asked once per URL and cached, the way
    `cytir.index.live_target` asks; a listing that cannot be read proves nothing and is retried after
    `ALIAS_RETRY_S`).
The profile NAME "prod" is not a signal: the twin's profile (`aw_e8`) is called "prod". CI never
sets `AW_PROD_COLLECTIONS` (the drain-golden harness names its throwaway prod `aw_e9`). A spelling
nobody listed is not caught here: `cytir.index.live_target` stays the authority for writes. This guard
is for reads.
"""
from __future__ import annotations

import contextlib
import functools
import logging
import os
import re
import threading
import time
from cytir.core import internal_http as IH
from urllib.parse import urlparse

log = logging.getLogger("aw.prod_guard")

DEFAULT_PROD_ALIAS = "aw_prod"
PROD_PORT = 6334
#: Prod-side lookups run at most this wide, with a pause after each batch (see `pace`).
PROD_LOOKUP_WORKERS = 2
PROD_BATCH_SLEEP_S = 0.25
#: A BIG PACK paces its prod WRITES harder (task 786, 2026-10-08). A pack of at least this many rows
#: (tranche 14: 1,409,761; tranche 13: 1,124,041) sleeps at least `PROD_BIG_PACK_PAUSE_S` between prod
#: upsert batches, whatever smaller `--pause-s` the unit passes. Smaller packs and the twin leg keep
#: the caller's value untouched. 09-28 lesson: big packs stress the 30 GB box (memory
#: `aw-heavy-rewrite-degrades-prod`).
PROD_BIG_PACK_ROWS = 1_000_000
PROD_BIG_PACK_PAUSE_S = 1.0
REPLAY_ENV = "AW_REPLAY"
#: POST calls one process may send to prod outside a replay (500 ids each: ~200k ids).
PROD_READ_BUDGET = 400
BUDGET_ENV = "AW_PROD_READ_BUDGET"
BUDGET_LOUD_EVERY = 100
#: how long an unreadable `/aliases` answer is remembered before the endpoint is asked again
ALIAS_RETRY_S = 60
ALIAS_TIMEOUT_S = 5

_SCROLL = re.compile(r"^/collections/[^/]+/points/scroll$")
_COLLECTION = re.compile(r"^/collections/([^/?]+)")
_POST_QUERY = re.compile(r"^/collections/[^/]+/(?:points(?:/.*)?|facet)$")

_STATE = {"replay": 0, "calls": 0}
_THROWAWAY: set[tuple[str, int | None]] = set()
_ALIASES: dict[str, tuple[bool | None, float]] = {}      # url -> (holds the prod alias?, asked at)
_LOCK = threading.Lock()
#: every allowed prod scroll of this process, for the receipt of whoever asked for it
ALLOWED: list[dict] = []


class ProdReadRefused(RuntimeError):
    """A prod call the rules forbid. Not retried, not swallowed: the caller used the wrong target."""


@functools.lru_cache(maxsize=1)
def _serving_collections() -> frozenset[str]:
    """Collections the payload-profile declaration marks `serving: true`. NOT the profile name."""
    from cytir.index import payload_profile as PP
    return frozenset(n for n, e in PP.load_profiles().items() if e.get("serving") is True)


def _collections() -> frozenset[str]:
    extra = {c.strip() for c in os.environ.get("AW_PROD_COLLECTIONS", "").split(",") if c.strip()}
    return frozenset({DEFAULT_PROD_ALIAS} | _serving_collections() | extra)


def _fetch_aliases(url: str) -> list[str] | None:
    """The alias names `GET <url>/aliases` lists; None when it cannot be read."""
    try:
        doc = IH.fetch_json(url.rstrip("/") + "/aliases", timeout=ALIAS_TIMEOUT_S)
        return [e["alias_name"] for e in doc["result"]["aliases"]]
    except Exception:   # noqa: BLE001 -- unreadable proves nothing; the caller decides
        return None


def _holds_prod_alias(url: str) -> bool:
    """True when the endpoint's own alias listing names `aw_prod`. Cached per URL."""
    key = url.rstrip("/").lower()
    now = time.monotonic()
    with _LOCK:
        hit = _ALIASES.get(key)
    if hit is not None and (hit[0] is not None or now - hit[1] < ALIAS_RETRY_S):
        return bool(hit[0])
    names = _fetch_aliases(url)
    with _LOCK:
        _ALIASES[key] = (None if names is None else DEFAULT_PROD_ALIAS in names, now)
    return bool(names) and DEFAULT_PROD_ALIAS in names


def _hostport(url: str) -> tuple[str, int | None]:
    p = urlparse(url if "//" in url else "//" + url)
    return (p.hostname or "").lower(), p.port


def mark_throwaway(url: str) -> None:
    """Declare `url` a proven throwaway Qdrant: nothing on it is prod, whatever its collections or
    aliases are called. ONLY `tests/throwaway_qdrant.prove_throwaway` calls this, after it read the
    endpoint's holdings (no alias, no declared live name, the session's marker, a non-live port). The
    drain-golden harness builds its own `aw_prod` alias over a collection named `aw_e9` there."""
    with _LOCK:
        _THROWAWAY.add(_hostport(url if "//" in url else "//" + url))


def is_prod(url: str | None, collection: str | None = None) -> bool:
    """True when (url, collection) names the serving box or its collection."""
    if url and _hostport(url if "//" in url else "//" + url) in _THROWAWAY:
        return False
    if collection and collection in _collections():
        return True
    if not url:
        return False
    if "//" not in url:
        url = "http://" + url
    host, port = _hostport(url)
    if port == PROD_PORT:
        return True
    for u in os.environ.get("AW_PROD_URL", "").split(","):
        if u.strip() and _hostport(u.strip()) == (host, port):
            return True
    return _holds_prod_alias(url)


def reset() -> None:
    """Forget the call count and the cached alias answers (tests; a long-lived process never needs it)."""
    with _LOCK:
        _STATE["calls"] = 0
        _ALIASES.clear()
    _serving_collections.cache_clear()


def replay_active() -> bool:
    return _STATE["replay"] > 0 or os.environ.get(REPLAY_ENV) == "1"


@contextlib.contextmanager
def replay():
    """Mark the process a replay: no query reaches prod while the block runs."""
    with _LOCK:
        _STATE["replay"] += 1
    try:
        yield
    finally:
        with _LOCK:
            _STATE["replay"] -= 1


def _budget() -> int | None:
    raw = os.environ.get(BUDGET_ENV, "").strip().lower()
    if raw == "unlimited":
        return None
    try:
        return int(raw) if raw else PROD_READ_BUDGET
    except ValueError:
        return PROD_READ_BUDGET


def _spend(url: str, coll: str | None, bare: str) -> None:
    """Count one prod POST against the process budget; refuse above it (outside a replay)."""
    limit = _budget()
    with _LOCK:
        _STATE["calls"] += 1
        n = _STATE["calls"]
    if limit is None:
        return
    if n > limit:
        raise ProdReadRefused(
            f"REFUSED POST {bare} on prod ({url}, {coll}): this process already sent {n - 1} POST "
            f"calls to prod (budget {limit}, ~{limit * 500} ids). A tool that needs more is a "
            f"drain or compose unit: set {BUDGET_ENV}=unlimited in ITS drop-in, nowhere else.")
    if n % BUDGET_LOUD_EVERY == 0:
        log.warning("PROD READ BUDGET: %d of %d POST calls to prod used by this process (%s)",
                    n, limit, bare)


def check(url: str, method: str, path: str, *, collection: str | None = None,
          allow_prod_scroll: bool = False, reason: str | None = None) -> bool:
    """Raise `ProdReadRefused` for a forbidden call; return True when the target is prod.

    `collection` defaults to the one in `path`. Call FIRST in every HTTP primitive."""
    bare = path.split("?", 1)[0].rstrip("/")
    coll = collection or (_COLLECTION.match(bare).group(1) if _COLLECTION.match(bare) else None)
    if not is_prod(url, coll):
        return False
    m = (method or "GET").upper()
    if replay_active() and m == "POST" and _POST_QUERY.match(bare):
        raise ProdReadRefused(
            f"REFUSED {m} {bare} on prod ({url}, {coll}): a replay takes the TWIN as both sides "
            f"({REPLAY_ENV}=1 or a replay() block is active). No override exists.")
    if m == "POST" and _SCROLL.match(bare):
        if not (allow_prod_scroll and (reason or "").strip()):
            raise ProdReadRefused(
                f"REFUSED scroll on prod ({url}, {coll}): a scroll scans the serving box "
                "(incident 2026-10-07, 187k scrolls, search 503 for ~30 min). Ask the TWIN "
                "(`checked_against: twin`) or use an id retrieve / a filtered count. "
                "An operator who must scroll passes allow_prod_scroll=True and a reason.")
        entry = {"url": url, "collection": coll, "path": bare, "reason": reason.strip(),
                 "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with _LOCK:
            ALLOWED.append(entry)
        log.warning("ALLOWED prod scroll: %s", entry)
    if m == "POST" and _POST_QUERY.match(bare):
        _spend(url, coll, bare)
    return True


def check_url(full_url: str, method: str = "POST", **kw) -> bool:
    """`check` for a caller that holds one full URL (`http://host:port/collections/c/points/scroll`)."""
    p = urlparse(full_url)
    return check(f"{p.scheme}://{p.netloc}", method, p.path, **kw)


def pace(url: str | None, collection: str | None = None, *, sleep=None) -> None:
    """The pause after one batch of prod lookups; a no-op for any other target."""
    if is_prod(url, collection):
        (sleep or time.sleep)(PROD_BATCH_SLEEP_S)


def lookup_workers(url: str | None, collection: str | None, wanted: int) -> int:
    """`wanted`, capped at `PROD_LOOKUP_WORKERS` for a prod target."""
    return min(wanted, PROD_LOOKUP_WORKERS) if is_prod(url, collection) else wanted


def write_pause(role: str, rows: int, pause_s: float) -> float:
    """The sleep between upsert batches: `pause_s`, raised to `PROD_BIG_PACK_PAUSE_S` for the PROD leg
    of a pack of `PROD_BIG_PACK_ROWS` rows or more. The twin leg is never changed."""
    if role == "prod" and rows >= PROD_BIG_PACK_ROWS:
        return max(pause_s, PROD_BIG_PACK_PAUSE_S)
    return pause_s


def is_big_pack(rows: int) -> bool:
    return rows >= PROD_BIG_PACK_ROWS
