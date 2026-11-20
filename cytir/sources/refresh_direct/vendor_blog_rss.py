#!/usr/bin/env python3
"""Vendor-blog RSS discovery adapter for the refresh-cadence new-refs lane
(task 661 F5, follow-up: industry-site refresh, "major industry sites like
Hugging Face" -> vendor blogs -- OpenAI, Google -- scoped 2026-09-26).

Same shape and scope as `cytir/sources/refresh_direct/github_releases.py`/`cytir.sources.refresh_direct.hf_org_models`:
DISCOVERS candidate post URLs from a source-owned, unauthenticated RSS/Atom
feed -- never scrapes the blog's HTML index, per
`data/acquisition-policy.v1.json`'s route precedence. Does not fetch post
bodies, embed, or write anywhere.

Both tracked blogs publish a standard RSS 2.0 feed (verified live
2026-09-26):
  - OpenAI news: https://openai.com/news/rss.xml (1230 items at verification
    time -- `tests/data/refresh_direct/openai-news-rss.xml` keeps the first 10, a
    REAL truncation of the real response, not hand-built).
  - Google AI blog: https://blog.google/technology/ai/rss/ (redirects live
    to .../innovation-and-ai/technology/ai/rss/ -- kept in full,
    `tests/data/refresh_direct/google-ai-blog-rss.xml`, 20 items).
Confirmed fields common to both: `<item><title>`, `<link>`, `<pubDate>`
(RFC 822), `<guid>`. OpenAI wraps text nodes in CDATA and sets
`guid isPermaLink="true"`; Google does not use CDATA and its `<guid>`
carries no `isPermaLink` attribute (defaults true per the RSS 2.0 spec
when absent) -- this parser reads `.text` either way, which handles CDATA
transparently (ElementTree exposes CDATA content as plain text), and does
not special-case `isPermaLink`. Google's feed also carries extra
non-standard namespaced elements per item (`media:content`, `og:*`,
`author:*`) that this parser simply ignores -- it only ever looks for the
four bare (namespace-less) fields above, which is exactly the plain RSS
2.0 subset the format guarantees.

Registered as `ADAPTER_REGISTRY["us-vendor-ai-blogs-tracked"]` in
`new_refs.py` (pack-id-keyed, same reason as the GitHub/HF
siblings: shares the coarse `adapter: api`/`rss` family with unrelated
packs), activates once `sources/us/vendor-ai-blogs-tracked.yaml` lands as
`ingested` via its own POLICY PR (a follow-up to #383, not part of it).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Plain text from an RSS `<description>` -- Google's carries a
    leading `<img ...>` tag before the real sentence (verified live,
    2026-09-26); OpenAI's is already plain text. No HTML in the output
    either way (this becomes an indexed field, never re-rendered)."""
    return " ".join(_HTML_TAG_RE.sub(" ", text).split())


def parse_rss(xml_text: str, blog: str) -> list[dict[str, Any]]:
    """Pure: no I/O. Each RSS `<item>` is one blog post. Returns
    `[{"url", "title", "description", "published", "blog"}, ...]`, in the
    feed's own order (both real feeds are newest-first). `blog` is a
    caller-supplied label (e.g. "openai-news", "google-ai-blog") stamped
    on every candidate so a mixed multi-feed run can tell them apart --
    neither feed's own XML names the publisher machine-readably.
    `description` is the item's own openly-published summary sentence
    (HTML-stripped) -- task 661 F5 (2026-09-28): this is what a
    WAF/bot-challenged page falls back to as its minimum record
    (`direct_compose.build_feed_item_row`), so it is
    captured for every item, not only the ones that turn out
    challenged."""
    root = ET.fromstring(xml_text)
    out = []
    for item in root.findall(".//item"):
        link_el = item.find("link")
        title_el = item.find("title")
        desc_el = item.find("description")
        pubdate_el = item.find("pubDate")
        url = (link_el.text or "").strip() if link_el is not None else None
        if not url:
            continue
        out.append({
            "url": url,
            "title": (title_el.text or "").strip() if title_el is not None else "",
            "description": (_strip_html(desc_el.text) if desc_el is not None
                            and desc_el.text else ""),
            "published": pubdate_el.text if pubdate_el is not None else None,
            "blog": blog,
        })
    return out


def blogs_from_pack_path(pack_path: str) -> list[dict[str, str]]:
    """Read the tracked blog list from the pack yaml's own `routes[]`
    (same discipline as `github_releases_adapter.repos_from_pack_path`):
    the pack schema's root `acquire.params` has a fixed
    `source_owned_incremental_feed` shape (`url` singular) with no room
    for a flat list, so each tracked blog is its own `routes[]` entry,
    `acquire.params.url` the feed, `host` the label. Returns
    `[{"label": ..., "url": ...}, ...]`. `pack_path` is the field
    `cyt source refresh-cadence` already stamps on every
    cadence-table row (repo-root-relative), so this needs no new plumbing
    through `new_refs.py`."""
    import yaml  # noqa: PLC0415 -- only needed on this path, not in unit tests
    doc = yaml.safe_load((REPO_ROOT / pack_path).read_text(encoding="utf-8"))
    blogs = []
    for route in (doc or {}).get("routes") or []:
        url = ((route or {}).get("acquire") or {}).get("params", {}).get("url")
        host = (route or {}).get("host", "")
        if url:
            blogs.append({"label": host, "url": url})
    return blogs


def discover(source_row: dict, *, fetch=None) -> dict:
    """The `ADAPTER_REGISTRY` entry point: `callable(source_row) -> dict`.
    `fetch` is injected (defaults to a real `httpx` GET) so tests never
    touch the network -- pass a fake that returns fixture RSS XML.
    The tracked-blog list comes from
    `source_row["acquire"]["params"]["blogs"]` (a list of
    `{"label", "url"}`) when the caller provides it directly (unit
    tests), else -- the real `new_refs.py` path, whose
    cadence-table rows never carry `acquire` -- from
    `source_row["pack_path"]`'s own `routes[]` (see
    `blogs_from_pack_path`). An empty/missing list either way is a
    refusal, never a silent zero-candidate "check" (same discipline as
    the GitHub/HF siblings)."""
    blogs = ((source_row.get("acquire") or {}).get("params") or {}).get("blogs") or []
    if not blogs and source_row.get("pack_path"):
        blogs = blogs_from_pack_path(source_row["pack_path"])
    if not blogs:
        raise ValueError(
            "vendor_blog_rss adapter: no tracked blogs found in "
            "source_row.acquire.params.blogs nor in the pack at "
            f"source_row['pack_path']={source_row.get('pack_path')!r} -- "
            "nothing to track (a real gap, not a quiet day)")

    if fetch is None:
        from cytir.sources.family import HttpSpec  # noqa: PLC0415
        from cytir.sources.http import make_client  # noqa: PLC0415
        def fetch(url: str) -> str:
            with make_client(HttpSpec(timeout=15.0, follow_redirects=True)) as client:
                r = client.get(url)
            r.raise_for_status()
            return r.text

    candidates: list[dict] = []
    seen_urls: set[str] = set()
    errors: dict[str, str] = {}
    for blog in blogs:
        label = blog["label"]
        url = blog["url"]
        try:
            xml_text = fetch(url)
            # One post often sits in several section feeds of the same host
            # (task 715): first feed wins, so it is fetched and packed once.
            for cand in parse_rss(xml_text, label):
                key = cand["url"].rstrip("/")
                if key not in seen_urls:
                    seen_urls.add(key)
                    candidates.append(cand)
        except Exception as exc:  # noqa: BLE001 -- one blog's feed failing
            # must not silently drop the others; it's named, not swallowed.
            errors[label] = f"{type(exc).__name__}: {exc}"

    return {
        "adapter": "vendor_blog_rss",
        "blogs_checked": len(blogs),
        "blogs_errored": len(errors),
        "errors": errors,
        "candidates_found": len(candidates),
        "candidates": candidates,
    }
