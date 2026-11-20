"""S10 step 6 (task 1104): the serving path's one external call, the Qdrant I/O seam.

Every Qdrant client the server uses is built and cached HERE, through `cytir.core.qdrant.qdrant_client`
(the one legal construction site; it is in the serving image because the image copies `cytir/`,
verified by task 1083). Tests patch `qdrant_io.client`; no code assigns `retrieval.client`
(`tests/test_serve_s10_pure_modules.py` enforces it). The lazy clients and the timeouts are state in
this one module (R4): mutated here, never re-bound anywhere else. No server module is imported
but `respond`, `schema` and `settings`; `retrieval` never.
"""
from __future__ import annotations

from qdrant_client import models as qm

from cytir.core.qdrant import qdrant_client

from . import settings as _settings
from .respond import _canonical_key
from .schema import COLLECTION
from .settings import SETTINGS as _S

_client = None
_health_client = None
_indexed_fields: frozenset[str] | None = None

# Qdrant call timeouts (S4). Search calls block FastAPI's sync-route threadpool
# (~40 threads); at the old 30s a hung qdrant parked every thread for 30s x 3
# calls and took /health down with it. Measured on the live 44M collection
# (2026-07-03): fully-cold dense leg (random 128-dim vector, rescore=off)
# max 0.41s, bm25 leg ~5ms — 5s is >10x headroom per call. The 2.5-3.5s
# "cold new query" total is TEI query-embed time, which is NOT under this
# timeout (only qdrant HTTP calls are). Env-overridable for ops rollback.
_QDRANT_TIMEOUT = _S.runtime.qdrant_timeout
_QDRANT_HEALTH_TIMEOUT = _S.runtime.qdrant_health_timeout


def indexed_fields() -> frozenset[str]:
    """The payload fields the SERVED collection actually has an index on.

    Read once per process from the collection's own `payload_schema`, because
    the set is a property of the build and changes under the server at a flip:
    `aw_e8` carries ten payload indexes, `aw_e9` carries eight (task 637 drops
    the dead `tier` and the 0-point `lei`). Any code path that builds a
    `FieldCondition` on a field that MIGHT not be indexed has to ask first.

    A failed read is NOT cached, so a transient qdrant hiccup disables the
    caller's fast path for one request instead of for the process's life.
    """
    global _indexed_fields
    if _indexed_fields is None:
        try:
            schema_ = health_client().get_collection(COLLECTION).payload_schema
        except Exception:  # noqa: BLE001 — introspection must never 500 a search
            return frozenset()
        _indexed_fields = frozenset(schema_ or {})
    return _indexed_fields


def _identifier_lookup(field: str, value: str, k: int, extra_filter=None) -> list[dict]:
    """Exact payload-filter lookup for an entity card by register_id/lei.
    Requires a keyword index on `field` (task 170 §B.2). Returns formatted
    results (entity cards first) or [] to fall through to normal search.

    REFUSES on an unindexed field (task 637 §22). The filter below is a
    `scroll`, and a scroll whose filter names a field with no payload index is
    a full scan of the collection: measured at 54 s on 33.7M against 3 s for
    the indexed form, and it saturates the API pool into 503s (memory
    `prod-serving-box-fragile-scrolls`). That failure is a slow SUCCESS, so the
    `except` below cannot catch it -- an identifier-shaped query would simply
    take the box down. `aw_e9` drops the `lei` index, and `_detect_identifier`
    still routes any 20-character alphanumeric query to `field="lei"`, so on the
    flip this stops being hypothetical.
    """
    if field not in indexed_fields():
        return []
    flt = qm.Filter(must=[qm.FieldCondition(key=field,
                                            match=qm.MatchValue(value=value))])
    if extra_filter is not None:     # task 2640: the operator filter still applies
        flt = qm.Filter(must=[flt, extra_filter])
    try:
        # task 199: limit floored at 64 (was max(k, 8)) so the scroll window —
        # and therefore the cards-first sort input — does not depend on k for
        # k <= 64: identifier results are a strict k-prefix too. An identifier
        # matches a handful of points; the indexed-filter scroll cost is ~0.
        pts, _ = client().scroll(collection_name=COLLECTION, scroll_filter=flt,
                                 limit=max(k, 64), with_payload=True)
    except Exception:  # noqa: BLE001 — never let the fast-path break search
        return []
    # a UID may also appear on a few content pages; entity cards rank first
    pts.sort(key=lambda p: 0 if (p.payload or {}).get("source_type")
             == "entity_card" else 1)
    out, seen = [], set()
    for p in pts:
        pl = p.payload or {}
        key = _canonical_key(pl.get("url"))
        if key in seen:
            continue
        seen.add(key)
        out.append({"url": pl.get("url"), "anchor": pl.get("anchor"),
                    "title": pl.get("title"), "snippet": pl.get("snippet"),
                    "source_type": pl.get("source_type"), "lang": pl.get("lang"),
                    "trust_tier": pl.get("trust_tier"),
                    "last_seen": pl.get("last_seen"),
                    # task 625: the news-freshness policy captures a genuine,
                    # never-inferred `published` date at ingest (aw_b1_projection's
                    # BASE_PASSTHROUGH) but the serving projection never surfaced
                    # it. `None` where the source itself carries no defensible
                    # date (e.g. a section/hub page) -- absent is correct there,
                    # not a gap. Additive key; parity checks compare urls only.
                    "published": pl.get("published"),
                    "score": round(float(pl.get("trust_tier") or 1.0), 4)})
        if len(out) >= k:
            break
    return out


def _url() -> str:
    """The Qdrant URL, refused unless it is http(s). `cytir.core.qdrant.qdrant_client` opens a LOCAL store
    for any other string (`:memory:`, a path); the server must fail loudly instead (review 1109)."""
    url = _settings.current("AW_QDRANT_URL")
    if not str(url).startswith(("http://", "https://")):
        raise ValueError(f"CYTIR_QDRANT_URL must start with http:// or https://, got {url!r}")
    return url


def client():
    global _client
    if _client is None:
        _client = qdrant_client(_url(), timeout=_QDRANT_TIMEOUT)
    return _client


def health_client():
    """Separate client with a 2s timeout so /health degrades FIRST when qdrant
    is slow/hung, instead of queueing behind 5s search calls (S4)."""
    global _health_client
    if _health_client is None:
        _health_client = qdrant_client(_url(),
                                       timeout=_QDRANT_HEALTH_TIMEOUT)
    return _health_client


def _leg(qvec, using, candidates, qfilter, search_params=None, with_payload=True):
    return client().query_points(collection_name=COLLECTION, query=qvec, using=using,
                                 limit=candidates, query_filter=qfilter,
                                 search_params=search_params,
                                 with_payload=with_payload).points
