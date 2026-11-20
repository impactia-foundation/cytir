"""Is this Qdrant endpoint, or this collection on it, LIVE?  (slice 8 step 2, round 3 of #553)

A build transaction (`promote_queue.Single`, `cyt index build`) writes a collection under a lock of
its own and WITHOUT the twin-first gate. It must therefore never be pointed at what serves search.
Rounds 1 and 2 of the review proved that "what serves" cannot be recognised by how a URL is
SPELLED: `0.0.0.0:6334`, `[::ffff:127.0.0.1]:6334`, a LAN IP, a tunnel or a hostname all reach the
same server, and a wrong list of spellings let a build write 12,571 points into the live
`source_cards`. So the question is put to the ENDPOINT:

    A target endpoint is LIVE when it holds the configured prod alias, or any collection a live
    box declares (the payload-profile declaration, the sealed serving-profile fixtures, and the
    collection the live server's authority sidecar queries). A live endpoint is refused for every
    collection name on it, whatever name the build asked for.

It reads `GET /aliases` and `GET /collections` and FAILS CLOSED: a listing that cannot be read, or
that is not the shape Qdrant answers, proves nothing. Nothing here writes. The declared names are
READ from the repository's own declarations; the one fact that no declaration carries (the cards
collection's default name) is a single constant that a test pins to the server's own default.
"""
from __future__ import annotations

import os
import pathlib
from typing import Iterable

from cytir.index import payload_profile as PP
from cytir.index import promote_sealed_pack as PSP

ROOT = pathlib.Path(__file__).resolve().parents[2]
SERVING_PROFILE_FIXTURES = ROOT / "data" / "serving-profiles"

#: The default name of the collection the live server's authority sidecar queries. The server
#: reads it from `AW_CARDS_COLLECTION` (`cytir/serve/authority.py`); a test pins this constant
#: to that default, so the two cannot drift apart.
CARDS_COLLECTION_DEFAULT = "source_cards"
CARDS_COLLECTION_ENV = "AW_CARDS_COLLECTION"

DEFAULT_PROD_ALIAS = "aw_prod"


class LiveTargetRefused(RuntimeError):
    """The target is, or cannot be proven not to be, a live endpoint or collection."""


def live_collection_names() -> frozenset[str]:
    """The physical collection names a live box declares, derived from the repository's own
    declarations (nothing hand-listed here): every collection the payload-profile declaration
    names (except a `staged` one, task 847), every sealed serving-profile fixture (`<name>.config.v1.json`), and the cards
    collection the live server queries."""
    declared = PP.load_profiles()
    # task 847: a `staged` declaration names a collection that does not serve yet (aw_e10 during its
    # build). It is the BUILD's target, so it is not a live name; the flip commit sets it `live`.
    names = set(declared) - PP.staged_collections(declared)
    # task 2254: a `twin-fixture` is the CI job's throwaway collection (`aw_gate_fixture`); `cyt index
    # build` must be able to create it. Only a non-serving, role-declared fixture is exempt.
    names -= {n for n, e in declared.items()
              if str(e.get("role", "")).lower().startswith("twin-fixture") and e.get("serving") is not True}
    if SERVING_PROFILE_FIXTURES.is_dir():
        names |= {p.name[: -len(".config.v1.json")]
                  for p in SERVING_PROFILE_FIXTURES.glob("*.config.v1.json")}
    # BOTH: an env override ADDS a live name; it never hides the default one (round 4, F3)
    names.add(CARDS_COLLECTION_DEFAULT)
    if os.environ.get(CARDS_COLLECTION_ENV):
        names.add(os.environ[CARDS_COLLECTION_ENV])
    return frozenset(names)


#: The port `qdrant_client.QdrantClient` uses for an http(s) URL that names none (it ignores the
#: scheme's own default). `urllib` would use 80 / 443, so a URL checked with one and written with the
#: other is two different endpoints (round 4, B3 of #559).
QDRANT_DEFAULT_PORT = 6333


def normalize_endpoint(url: str) -> str:
    """`scheme://host:port` with the port the CLIENT uses (`QDRANT_DEFAULT_PORT` when the URL has
    none), no trailing slash. The ONE string a caller hands to the liveness check AND to the client
    that writes. Refuses what it cannot read as a plain endpoint (no host, a scheme other than
    http/https, userinfo, a path, a query)."""
    import urllib.parse
    u = urllib.parse.urlsplit(url.strip())
    if u.scheme.lower() not in ("http", "https") or not u.hostname:
        raise LiveTargetRefused(f"{url!r} is not an http(s) endpoint with a host")
    if u.username or u.password or u.path not in ("", "/") or u.query or u.fragment:
        raise LiveTargetRefused(f"{url!r}: an endpoint is scheme://host[:port] only (no credentials, "
                                "path, query or fragment)")
    try:
        port = u.port
    except ValueError as exc:
        raise LiveTargetRefused(f"{url!r}: {exc}") from exc
    host = f"[{u.hostname}]" if ":" in u.hostname else u.hostname
    return f"{u.scheme.lower()}://{host}:{port or QDRANT_DEFAULT_PORT}"


def read_holdings(url: str) -> tuple[dict[str, str], frozenset[str]]:
    """(aliases `{alias: collection}`, collection names) the endpoint holds. Raises
    `LiveTargetRefused` when either listing cannot be read or is not Qdrant's shape: an
    unreadable endpoint is not proven harmless (fail closed)."""
    try:
        listed_aliases = PSP._api(url, "/aliases", retry_status=False)
        listed_collections = PSP._api(url, "/collections", retry_status=False)
        aliases = {e["alias_name"]: e["collection_name"]
                   for e in listed_aliases["aliases"]}
        collections = frozenset(e["name"] for e in listed_collections["collections"])
    except Exception as exc:   # noqa: BLE001 -- ANY failure to read the listing refuses
        raise LiveTargetRefused(
            f"cannot read what {url} holds (aliases, collections), so it cannot be proven "
            f"not live: {type(exc).__name__}: {exc}") from exc
    return aliases, collections


def refuse_live_target(url: str, collection: str, *, prod_aliases: Iterable[str] = (),
                       extra_names: Iterable[str] = ()) -> None:
    """Refuse `collection` on `url` when the endpoint is live, or the collection is an alias or a
    declared live name. `prod_aliases`: the prod alias the caller is configured with (the drain's
    `--prod-alias`), added to the default. `extra_names`: collections the caller knows are live
    (a sealed pack's own twin collection)."""
    aliases, held = read_holdings(url)
    prod = {DEFAULT_PROD_ALIAS, *prod_aliases}
    live = live_collection_names() | frozenset(extra_names)
    live_alias_targets = {aliases[a] for a in prod if a in aliases}
    on_endpoint = sorted((prod & set(aliases)) | (live & held))
    if on_endpoint:
        raise LiveTargetRefused(
            f"{url} is a LIVE endpoint: it holds {on_endpoint}. A build never writes there, "
            f"whatever the collection is called ({collection!r}); `cyt index promote` is the door "
            "to the real drain")
    if collection in aliases:
        raise LiveTargetRefused(f"{collection!r} is an alias (of {aliases[collection]!r}): a build "
                                "writes a physical collection only")
    if collection in live or collection in live_alias_targets:
        raise LiveTargetRefused(f"{collection!r} is a live collection (declared, or named by a "
                                "prod alias): `cyt index promote` is the door to the real drain")
