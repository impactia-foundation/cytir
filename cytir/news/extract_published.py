#!/usr/bin/env python3
"""extract_published.py — extract an article's PUBLISHED date from raw HTML.

News/reference sources need a `published` (ISO-8601 date) stamped in the payload
so (a) the fusion `recency` signal (ADR-037) can fire and (b) the rolling news
window (`news_window_filter.py`, Tim's 2026-07-21 news-freshness policy +
2026-08-11 retention ruling) can drop aged-out articles — 12mo fresh/discovery
tier, 36mo purge horizon. This module is the single extractor the
ingest/enrich path calls; it does NOT mirror article text (ADR-039 minimum
content) — it reads only the date signals out of the HTML and returns a date.

Priority chain (highest-confidence first — matches the policy note ordering):

  1. schema.org / JSON-LD  `datePublished`  (Article/NewsArticle/BlogPosting/…,
     incl. @graph and list-wrapped docs; `dateCreated` as a same-tier sibling)
  2. OpenGraph            `article:published_time`  (<meta property=...>)
  3. <meta> date tags     name/itemprop/property in _META_DATE_KEYS, plus
     <time datetime=... pubdate> and itemprop=datePublished
  4. sitemap  <lastmod>   (passed in by the caller as `sitemap_lastmod`)
  5. htmldate / trafilatura date extractor  (original-date preference) — only
     if installed; the module degrades gracefully without it.

Everything through step 4 is pure-python (regex + json), so the common path has
no third-party dependency and is fully unit-testable offline. Step 5 is a lazy,
optional import.

Returns an ISO date string `YYYY-MM-DD`, or None when no defensible date is
found. A parsed date is sanity-clamped: rejected if before `MIN_YEAR` or more
than one day after `now_ref` (a mis-parsed far-future date is worse than none).

  from extract_published import extract_published
  d = extract_published(html, url=url, sitemap_lastmod=lastmod, now_ref=now)
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from typing import Optional

MIN_YEAR = 1995  # nothing in the news corpus predates this; older => misparse

# ---------------------------------------------------------------------------
# date normalisation
# ---------------------------------------------------------------------------
_ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
# a few common non-ISO shapes seen in meta tags / lastmod
_SLASH_RE = re.compile(r"\b(\d{4})/(\d{2})/(\d{2})\b")
_DOT_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")  # dd.mm.yyyy (DE/CH)


def _clamp(d: _dt.date, now_ref: _dt.date) -> Optional[str]:
    if d.year < MIN_YEAR:
        return None
    if d > now_ref + _dt.timedelta(days=1):
        return None
    return d.isoformat()


def normalize_date(raw, now_ref: _dt.date) -> Optional[str]:
    """Best-effort parse of a raw date string/number to an ISO `YYYY-MM-DD`,
    clamped to a sane range. Returns None if nothing parseable."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        for x in raw:  # JSON-LD sometimes lists dates; take the first that parses
            got = normalize_date(x, now_ref)
            if got:
                return got
        return None
    s = str(raw).strip()
    if not s:
        return None
    # ISO / RFC3339 (2025-11-04, 2025-11-04T09:12:00+01:00) — take the date part
    m = _ISO_RE.search(s)
    if m:
        try:
            return _clamp(_dt.date(int(m[1]), int(m[2]), int(m[3])), now_ref)
        except ValueError:
            pass
    m = _SLASH_RE.search(s)
    if m:
        try:
            return _clamp(_dt.date(int(m[1]), int(m[2]), int(m[3])), now_ref)
        except ValueError:
            pass
    m = _DOT_RE.search(s)  # dd.mm.yyyy
    if m:
        try:
            return _clamp(_dt.date(int(m[3]), int(m[2]), int(m[1])), now_ref)
        except ValueError:
            pass
    # epoch seconds (some parsely/sailthru feeds)
    if s.isdigit() and len(s) in (10, 13):
        try:
            ts = int(s[:10])
            return _clamp(_dt.datetime.utcfromtimestamp(ts).date(), now_ref)
        except (ValueError, OSError):
            pass
    return None


# ---------------------------------------------------------------------------
# 1. JSON-LD schema.org
# ---------------------------------------------------------------------------
_LD_RE = re.compile(
    r'<script[^>]+type\s*=\s*["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.I | re.S,
)
# fields we accept, in preference order
_LD_DATE_KEYS = ("datePublished", "dateCreated", "uploadDate")


def _walk_ld(node, keys):
    """Depth-first search of a parsed JSON-LD node for the first date key hit."""
    if isinstance(node, dict):
        for k in keys:
            if node.get(k):
                return node[k]
        for v in node.values():
            got = _walk_ld(v, keys)
            if got:
                return got
    elif isinstance(node, list):
        for v in node:
            got = _walk_ld(v, keys)
            if got:
                return got
    return None


def from_jsonld(html: str):
    for blk in _LD_RE.findall(html):
        blk = blk.strip()
        if not blk:
            continue
        # some pages emit multiple concatenated JSON objects or trailing junk;
        # try strict first, then a lenient object-by-object scan.
        candidates = []
        try:
            candidates.append(json.loads(blk))
        except json.JSONDecodeError:
            for mo in re.finditer(r"\{.*?\}", blk, re.S):
                try:
                    candidates.append(json.loads(mo.group(0)))
                except json.JSONDecodeError:
                    continue
        for c in candidates:
            got = _walk_ld(c, _LD_DATE_KEYS)
            if got:
                return got
    return None


# ---------------------------------------------------------------------------
# 2. OpenGraph article:published_time
# ---------------------------------------------------------------------------
def _meta_content(html: str, attr: str, value: str):
    """Return the content= of a <meta {attr}="{value}"> tag (attr order-agnostic)."""
    # content before the key
    pat1 = re.compile(
        r'<meta[^>]+content\s*=\s*["\']([^"\']+)["\'][^>]*\b' + re.escape(attr) +
        r'\s*=\s*["\']' + re.escape(value) + r'["\']', re.I)
    # key before content
    pat2 = re.compile(
        r'<meta[^>]+\b' + re.escape(attr) + r'\s*=\s*["\']' + re.escape(value) +
        r'["\'][^>]+content\s*=\s*["\']([^"\']+)["\']', re.I)
    m = pat1.search(html) or pat2.search(html)
    return m.group(1) if m else None


def from_opengraph(html: str):
    return _meta_content(html, "property", "article:published_time")


# ---------------------------------------------------------------------------
# 3. generic <meta> date tags + <time>
# ---------------------------------------------------------------------------
# (attr, value) pairs, in descending confidence
_META_DATE_KEYS = [
    ("property", "article:published_time"),
    ("itemprop", "datePublished"),
    ("name", "parsely-pub-date"),
    ("name", "article:published_time"),
    ("name", "publish-date"),
    ("name", "publishdate"),
    ("name", "pubdate"),
    ("name", "date"),
    ("name", "sailthru.date"),
    ("name", "dc.date.issued"),
    ("name", "DC.date.issued"),
    ("name", "dcterms.date"),
    ("name", "dc.date"),
    ("name", "og:updated_time"),
    ("property", "og:updated_time"),
    ("itemprop", "dateModified"),
]

_TIME_RE = re.compile(
    r'<time[^>]+datetime\s*=\s*["\']([^"\']+)["\']', re.I)


def from_meta(html: str):
    for attr, value in _META_DATE_KEYS:
        got = _meta_content(html, attr, value)
        if got:
            return got
    m = _TIME_RE.search(html)  # <time datetime="...">
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# 5. htmldate / trafilatura (optional, lazy)
# ---------------------------------------------------------------------------
_HTMLDATE = None  # tri-state: None=unknown, False=absent, callable=present


def _htmldate(html: str, url: Optional[str]):
    global _HTMLDATE
    if _HTMLDATE is None:
        try:
            from htmldate import find_date  # trafilatura's date engine
            _HTMLDATE = find_date
        except Exception:  # noqa: BLE001
            _HTMLDATE = False
    if not _HTMLDATE:
        return None
    try:
        return _HTMLDATE(html, url=url, original_date=True,
                         outputformat="%Y-%m-%d", extensive_search=False)
    except Exception:  # noqa: BLE001  (htmldate can raise on pathological pages)
        return None


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def extract_published(html: str,
                      url: Optional[str] = None,
                      sitemap_lastmod: Optional[str] = None,
                      now_ref: Optional[_dt.date] = None,
                      use_htmldate: bool = True) -> Optional[str]:
    """Return the article's published date as ISO `YYYY-MM-DD`, or None.

    `now_ref` is the reference "now" (a datetime.date); passed in explicitly so
    the extractor is deterministic/testable. Defaults to today (UTC) when None.
    """
    if now_ref is None:
        now_ref = _dt.datetime.utcnow().date()
    elif isinstance(now_ref, _dt.datetime):
        now_ref = now_ref.date()

    html = html or ""
    # ordered signal sources; first that normalises to a sane date wins
    for src in (from_jsonld, from_opengraph, from_meta):
        got = normalize_date(src(html), now_ref)
        if got:
            return got
    got = normalize_date(sitemap_lastmod, now_ref)
    if got:
        return got
    if use_htmldate:
        got = normalize_date(_htmldate(html, url), now_ref)
        if got:
            return got
    return None


def published_for_page(html: str,
                       url: Optional[str] = None,
                       sitemap_published: Optional[str] = None,
                       now_ref: Optional[_dt.date] = None,
                       use_htmldate: bool = False) -> Optional[str]:
    """The date the SOURCE-FEED ingest stamps on every passage of one page.

    Task 448. `sitemap_ingest --freshness` used to take the sitemap `<lastmod>`
    and nothing else, so a publisher whose sitemap carries no dates produced
    dateless news rows: 850 of the 2026-08-27 pack's 6,357 rows (every srf.ch
    and rts.ch row) had no `published`, which is inadmissible under POLICY.md@v1
    §4.7 and invisible to the rolling news window. The page's own HTML carried
    the date the whole time — it was simply never read.

    Precedence (Tim, 2026-09-30, plan.md ruling 18): the page's OWN date wins
    where both exist; the sitemap `<lastmod>` is only the fallback. `<lastmod>` is a
    modification date. Measured on the real 2026-09-28 swissinfo stage: of 784 urls
    the page date equals the sitemap date for 424 and differs for 360 (340 of them
    the sitemap a day or more LATER than the page, 20 earlier). The chain is the
    extractor's own: JSON-LD `datePublished`, OpenGraph, meta/`<time>` tags, then
    the sitemap date, then (off by default, see below) htmldate.
    Before this ruling the sitemap date won and the page was read only where the
    sitemap gave nothing (task 448).

    `use_htmldate` defaults to False here (it is True in `extract_published`):
    an ingest must not produce different dates on two boxes depending on whether
    an optional package happens to be installed. The pure-python chain is the
    contract; enabling htmldate would need it pinned as a real dependency first.

    Returns an ISO `YYYY-MM-DD`, or None — a page with no defensible date keeps
    an ABSENT field. Never a guess, never the fetch date.
    """
    if now_ref is None:
        now_ref = _dt.datetime.utcnow().date()
    elif isinstance(now_ref, _dt.datetime):
        now_ref = now_ref.date()
    return extract_published(html, url=url, sitemap_lastmod=sitemap_published,
                             now_ref=now_ref, use_htmldate=use_htmldate)


if __name__ == "__main__":  # tiny CLI: read HTML on stdin, print the date
    import sys
    data = sys.stdin.read()
    print(extract_published(data, url=(sys.argv[1] if len(sys.argv) > 1 else None))
          or "")
