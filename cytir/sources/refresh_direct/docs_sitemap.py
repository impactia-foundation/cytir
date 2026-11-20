#!/usr/bin/env python3
"""Docs-site discovery adapter `docs_sitemap` for the refresh-cadence lane (task 972).

DISCOVERS the page URLs of one docs host from its own sitemap (nested sitemap
indexes included), applies the pack's scope (`select.path_deny_regex` for the
locale mirrors, `cap`, robots) and returns candidates with `lastmod`. It does
not fetch page bodies, embed or write: `direct_compose.run_stage`
does that, and does the change detection (`unchanged_against_twin`, below).

One way to enumerate a sitemap: this module calls
`cytir.acquire.sitemap_ingest.discover_sitemap_urls` / `select_urls`, the code the
bulk docs campaign already runs. The pack is the source of truth for scope
(`sources/us/opencode-ai-docs.yaml`), read through `cytir.sources.pack_loader`.

Two route shapes in the same pack:
  * `source_owned_sitemap_fetch` -- `params.urls` are sitemap files.
  * `direct_origin_crawl` with `params.seeds` -- ONE level of same-host links
    from each seed page, filtered by the route's `select`. For a docs tree the
    sitemap does not list (opencode.ai/v2/docs is such a tree: 0 of the 684
    sitemap URLs). Depth is capped at 1; no deeper crawl.

Out of scope, said once: tombstones for pages that left the sitemap.
"""
from __future__ import annotations

import html as _html
import re
import urllib.parse
import urllib.robotparser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

SITEMAP_ROUTE = "source_owned_sitemap_fetch"
CRAWL_ROUTE = "direct_origin_crawl"
_HREF_RE = re.compile(r"""href\s*=\s*["']([^"'#]+)""", re.I)
_SKIP_EXT = (".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js", ".ico",
             ".webp", ".xml", ".json", ".txt", ".zip", ".woff", ".woff2")


class _Resp:
    """The three attributes `sitemap_ingest._fetch_body` reads."""
    def __init__(self, status: int, body: bytes):
        self.status_code, self.content, self.headers = status, body, {}


class _Client:
    def __init__(self, fetch_bytes):
        self._fetch = fetch_bytes

    def get(self, url):
        status, body = self._fetch(url)
        return _Resp(status, body)


class _NoWait:
    def wait(self):
        return None


def default_fetch_bytes(url: str) -> tuple[int, bytes]:
    """ONE plain GET, honest UA. Politeness (delay) is the caller's `bucket`."""
    from cytir.sources.family import HttpSpec  # noqa: PLC0415
    from cytir.sources.http import make_client  # noqa: PLC0415
    with make_client(HttpSpec(timeout=20.0, follow_redirects=True)) as client:
        r = client.get(url)
    return r.status_code, r.content


def load_plans(pack_path: str) -> list:
    from cytir.sources import pack_loader  # noqa: PLC0415
    doc = pack_loader.load_pack(REPO_ROOT / pack_path)
    return [p for p in pack_loader.plans(doc) if p.route in (SITEMAP_ROUTE, CRAWL_ROUTE)]


def load_robots(host: str, fetch_bytes):
    """Live robots.txt for the host; None (= no rules) when unreadable. The consent gate
    (`consent_gate.py --direct-crawl`) still resolves robots per row afterwards."""
    try:
        status, body = fetch_bytes(f"https://{host}/robots.txt")
    except Exception:  # noqa: BLE001
        return None
    if status != 200:
        return None
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(body.decode("utf-8", "ignore").splitlines())
    return rp


def _deny_re(plan):
    return (plan.select or {}).get("path_deny_regex")


def seed_links(page_html: str, base_url: str, plan) -> list[dict]:
    """Same-host links on a seed page that pass the route's `select`. Trailing-slash twins
    (`/x` and `/x/`) count once; the first spelling wins."""
    host = urllib.parse.urlsplit(base_url).hostname
    pred = plan.predicate()
    seen, out = set(), []
    for href in _HREF_RE.findall(page_html or ""):
        u = urllib.parse.urljoin(base_url, _html.unescape(href.strip()))
        parts = urllib.parse.urlsplit(u)
        if parts.scheme not in ("http", "https") or parts.hostname != host or parts.query:
            continue
        if parts.path.lower().endswith(_SKIP_EXT):
            continue
        key = parts.path.rstrip("/") or "/"
        if key in seen or not pred(u):
            continue
        seen.add(key)
        out.append({"url": f"{parts.scheme}://{parts.netloc}{parts.path}", "lastmod": None})
    return out


def route_candidates(plan, fetch_bytes, bucket=None) -> list[dict]:
    from cytir.acquire import sitemap_ingest as SI  # noqa: PLC0415
    host = plan.host.split("/")[0]
    rp = load_robots(host, fetch_bytes)
    cap = int(plan.cap or 400)
    delay = float(plan.delay_s or 1.5)
    if plan.route == SITEMAP_ROUTE:
        bucket = bucket or SI.TokenBucket(1.0 / delay)
        cfg = {"host": host, "sitemaps": list(plan.params["urls"]),
               "exclude_re": _deny_re(plan)}
        urls = SI.discover_sitemap_urls(_Client(fetch_bytes), cfg, "", bucket, _NoWait())
        kept = SI.select_urls(urls, cfg, cap, rp)
    else:
        urls = []
        for seed in plan.params.get("seeds") or []:
            status, body = fetch_bytes(seed)
            if status != 200:
                raise RuntimeError(f"seed {seed} answered HTTP {status}")
            urls.extend(seed_links(body.decode("utf-8", "ignore"), seed, plan))
        pred = plan.predicate()
        kept, seen = [], set()
        for u in urls:
            if u["url"] in seen or not pred(u["url"]) or not SI.robots_ok(rp, u["url"]):
                continue
            seen.add(u["url"])
            kept.append(u)
        kept = kept[:cap]
    return [{"url": u["url"], "lastmod": u.get("lastmod"), "delay_s": delay,
             "route": plan.route, "host": plan.host} for u in kept]


def discover(source_row: dict, *, fetch_bytes=None, plans=None, bucket=None) -> dict:
    """`ADAPTER_REGISTRY` entry point: callable(source_row) -> dict. `fetch_bytes(url) ->
    (status, bytes)` is injected in tests. An empty route list is a refusal, a failed route is
    named in `errors` (an error is not an empty -- `discovery_refusal` reads `orgs_errored`)."""
    fetch_bytes = fetch_bytes or default_fetch_bytes
    plans = plans if plans is not None else load_plans(source_row["pack_path"])
    if not plans:
        raise ValueError(f"docs_sitemap: no sitemap/crawl routes in {source_row.get('pack_path')!r}")
    cands, errors, seen = [], {}, set()
    for plan in plans:
        try:
            for c in route_candidates(plan, fetch_bytes, bucket):
                if c["url"] not in seen:
                    seen.add(c["url"])
                    cands.append(c)
        except Exception as exc:  # noqa: BLE001 -- one route must not hide the others
            errors[plan.route_id] = f"{type(exc).__name__}: {exc}"
    return {"adapter": "docs_sitemap", "orgs_checked": len(plans), "orgs_errored": len(errors),
            "errors": errors, "candidates_found": len(cands), "candidates": cands}


def twin_digests(client, collection: str, url: str) -> set[str]:
    """Canonical `content_digest` key (32 hex, `row_decode.digest_key`) of every twin point whose
    `url` equals `url` (exact match on an indexed field); the rows are decoded first."""
    from qdrant_client import models as qm  # noqa: PLC0415
    from cytir.index import row_decode  # noqa: PLC0415
    decode = row_decode.reader(collection)
    out, offset = set(), None
    while True:
        pts, offset = client.scroll(
            collection, scroll_filter=qm.Filter(must=[qm.FieldCondition(
                key="url", match=qm.MatchValue(value=url))]),
            limit=256, offset=offset, with_payload=["content_digest"], with_vectors=False)
        out.update(str(decode(p.payload).get("content_digest") or "") for p in pts)  # canonical row
        if offset is None:
            return out


def unchanged_against_twin(rows: list[dict], twin_set: set[str]) -> bool:
    """True when the page's freshly cut chunks are exactly the twin's chunks for that URL (same
    set of canonical digest keys: `row_decode.digest_key(cytir.core.ids.content_digest(text))`, the
    32 hex the coded twin stores). A 64-hex and a 32-hex spelling of one digest are equal.
    No rows, or no twin rows, is never 'unchanged'."""
    from cytir.core import ids  # noqa: PLC0415
    from cytir.index import row_decode  # noqa: PLC0415
    if not rows or not twin_set:
        return False
    return {row_decode.digest_key(ids.content_digest(r["text"])) for r in rows} == \
        {row_decode.digest_key(d) for d in twin_set}
