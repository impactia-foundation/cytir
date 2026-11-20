"""AW search-api: JSON search API + the human search page (spec §8 surfaces).
Loads the models. MCP is served by a separate thin process (mcp_server.py)."""
from __future__ import annotations

import logging
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import authority, heads, operators, qdrant_io, retrieval, settings, tables
from . import legs as _legs
from .query import parse_operators
from .read_url import attach_read_urls
from .settings import SETTINGS as _S


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Fail at container start, not on the first query, when German de-compounding is
    unavailable (ADR-121). The default query has no `lang`, so every 12+ character token is
    split: a missing splitter or table would otherwise 500 nearly every query, and each
    request would retry the 40 MB load. `DecompoundUnavailable` stops the start-up."""
    from cytir.core import decompound

    decompound.ensure_ready()
    yield


api = FastAPI(title="authoritative-web", version="0.1.0", lifespan=_lifespan)

log = logging.getLogger("aw_server")

WEBAPP = pathlib.Path(_S.runtime.webapp)

# Privacy notice (task 390) — a sibling static page next to index.html, served
# at /privacy. Required on the serving surface before the register-published-
# people facet (POLICY §4.9.1) is exposed to public traffic; it is a launch
# condition of that section, not a merge condition.
PRIVACY = WEBAPP.parent / "privacy.html"

# Both HTML pages are read from disk per request (no restart needed to publish
# a copy change) and must revalidate, so a corrected notice or a redeployed
# page is never served stale. Caddy sets this for `/` and `/index.html`; it
# does NOT match /privacy, so the header is set here — one knob, in the repo,
# rather than an edit to the shared reverse proxy.
NO_CACHE = {"Cache-Control": "no-cache, must-revalidate"}

# Serve the landing-page assets (logos, favicon, apple-touch-icon, og:image).
# index.html references them at /assets/... — mount the sibling assets/ dir.
# Guarded so the app still boots if assets/ is absent.
ASSETS = WEBAPP.parent / "assets"
if ASSETS.is_dir():
    api.mount("/assets", StaticFiles(directory=ASSETS), name="assets")


@api.get("/health")
def health():
    # Uses a DEDICATED short-timeout qdrant client (S4): when qdrant hangs,
    # /health must degrade fast (2s) and legibly — not queue behind search
    # calls or block a threadpool thread for the full search timeout.
    try:
        n = qdrant_io.health_client().get_collection(retrieval.COLLECTION).points_count
        # task 522: echo the load-bearing serving knobs (collection, MRL
        # width, dense rescore/oversampling/ignore-quant, weights file +
        # sha256) so a flip is verified by `curl /health`, not by trusting
        # the compose file was applied (deployment-doctrine "No Invisible
        # State"; task 519 found exactly this gap unfindable from the API).
        return {"status": "ok", "points": n, "config": retrieval.effective_config()}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"status": "degraded", "error": str(e)}, status_code=503)


@api.get("/config")
def config():
    """Every setting the process parsed, its source and the artifacts it loaded (S10 step 2,
    `aw.serve-config.v1`). Answers without Qdrant or the models: drift is a `curl | diff`."""
    return settings.config_document(_S, artifacts=tables.loaded_artifacts(heads._head_artifacts()))


@api.get("/search")
def search(q: str = Query(..., min_length=1), k: int = Query(10, ge=1, le=100),
           lang: str | None = None, country: str | None = None,
           jurisdiction: str | None = None,
           authorities: int = Query(0, ge=0, le=1),
           site: list[str] | None = Query(None), exclude_site: list[str] | None = Query(None),
           source: list[str] | None = Query(None)):
    # `country` is an OPTIONAL soft-boost facet (CH/FR/DE/CA/US). Unset/none =>
    # ranking AND response byte-identical to before; a value floats that
    # jurisdiction's authoritative sources up without filtering anything out.
    # `jurisdiction` is accepted as an alias for `country` (the name callers
    # intuitively reach for; task 257) — `country` wins if both are given.
    country = country or jurisdiction
    # Search operators (task 2640): `site:` `-site:` `source:` `lang:` `country:` in the query text
    # and the named parameters `site` `exclude_site` `source`, merged. No operator => q, lang,
    # country pass through untouched and the response has no `filters` key (byte-identical).
    q, found = parse_operators(q)
    ops = found.merged(site=site or (), exclude_site=exclude_site or (), source=source or ())
    op_filter, filters, boosts = None, {}, {}
    if ops.any():
        try:
            op_filter, filters = operators.build(ops)
        except operators.OperatorError as e:
            return JSONResponse({"error": str(e), "status": "bad_request"}, status_code=400)
        for s in filters.pop("unknown_source", ()):   # an unknown source: value is query text
            q = f"{q} source:{s}".strip()
        if not q.strip():
            return JSONResponse({"error": "query has no terms besides the operators",
                                 "status": "bad_request"}, status_code=400)
        notes = [filters["note"]] if filters.get("note") else []
        for name, op_val, par_val in (("lang", found.lang, lang), ("country", found.country, country)):
            if op_val:
                if name == "lang":
                    filters[name] = op_val
                else:   # country is a soft boost, not a filter: echoed under `boosts` (task 2655)
                    boosts[name] = op_val
                if par_val and par_val.lower() != op_val.lower():
                    notes.append(f"{name}:{op_val} in the query overrides {name}={par_val}")
        if notes:
            filters["note"] = "; ".join(notes)
        lang = found.lang or lang
        country = found.country or country
    extra = {"op_filter": op_filter} if op_filter is not None else {}
    #
    # Graceful failure (S4): any unexpected backend error (qdrant down/slow,
    # TEI + fallback both failing, ...) returns a small JSON 503 instead of a
    # raw 500 HTML page — clients (webapp, MCP, monitors) always get JSON.
    # Full traceback still lands in the container logs.
    _legs.AUTHLEG_DEGRADED.set(None)   # task 2665: this request's own authority-leg outcome
    try:
        results = retrieval.search(q, k=k, lang=lang, country=country, **extra)
    except Exception:  # noqa: BLE001
        log.exception("search failed for q=%r", q)
        return JSONResponse(
            {"error": "search backend unavailable — try again shortly",
             "status": "degraded"},
            status_code=503)
    attach_read_urls(results)   # task 2709: additive `read_url` where a no-JavaScript twin exists
    resp = {"query": q, "lang": lang, "count": len(results), "results": results}
    # task 2665 (review 2671 F1): a filtered search whose bounded authority leg did not answer
    # says so; absent otherwise (byte-identical response).
    if _legs.AUTHLEG_DEGRADED.get() is not None:
        resp["degraded"] = {"authority_leg": _legs.AUTHLEG_DEGRADED.get()}
    if filters:
        resp["filters"] = filters
    if country and "country" not in boosts:
        boosts["country"] = country.strip().upper()   # the named param echoes like the operator (task 2663)
    if boosts:
        resp["boosts"] = boosts
    if country:
        resp["country"] = country.strip().upper()
    # Opt-in authorities SIDECAR (source-authority layer / ADR-021 guard):
    # a SEPARATE top-level block from a SEPARATE tiny collection — never
    # interleaved with or scored against results[]. Default (authorities
    # unset/0) leaves the response byte-identical. Failure-isolated: any
    # source_cards error is logged and the sidecar silently omitted.
    if authorities:
        try:
            host_ok = operators.host_predicate(ops) if ops.any() else None
            resp["authorities"] = authority.find_authorities(
                q, k=3, country=country, floor=authority.SIDECAR_FLOOR,
                **({"host_ok": host_ok} if host_ok else {}))
        except Exception:  # noqa: BLE001
            log.exception("authorities sidecar failed for q=%r (search unaffected)", q)
    return resp


@api.get("/find_authority")
def find_authority(q: str = Query(..., min_length=1),
                   k: int = Query(3, ge=1, le=10),
                   country: str | None = None):
    # Dedicated source-card lookup (the MCP `find_authority` tool calls this).
    # Queries ONLY the source_cards collection — never the passage pool.
    # Failure-isolated: backend errors return a clear JSON 503, never a 500,
    # and can never affect /search.
    try:
        cards, relaxed = authority.find_authorities_relaxed(q, k=k, country=country)
    except Exception as e:  # noqa: BLE001
        log.exception("find_authority failed for q=%r", q)
        return JSONResponse(
            {"error": f"authority lookup unavailable ({type(e).__name__}) — "
                      "try again shortly; /search is unaffected",
             "status": "degraded", "authorities": []},
            status_code=503)
    return {"query": q, "count": len(cards), **authority.annotate(cards),
            **({"filter_relaxed": relaxed} if relaxed else {}),
            "authorities": cards}


@api.get("/", response_class=HTMLResponse)
def home():
    if WEBAPP.exists():
        return WEBAPP.read_text(encoding="utf-8")
    return "<h1>authoritative-web</h1><p>search API at <code>/search?q=…</code></p>"


@api.get("/privacy", response_class=HTMLResponse)
def privacy():
    # Fail LOUD: a missing notice is a governance problem (§4.9.1 launch
    # condition), so it 404s visibly rather than falling back to a stub that
    # would let an operator believe the notice is live when it is not.
    if not PRIVACY.exists():
        log.error("privacy notice missing at %s — /privacy is serving 404", PRIVACY)
        return JSONResponse(
            {"error": "privacy notice unavailable — please contact "
                      "bender@impactia.org", "status": "missing"},
            status_code=404)
    return HTMLResponse(PRIVACY.read_text(encoding="utf-8"), headers=NO_CACHE)
