#!/usr/bin/env python3
# Moved into the package (task 683 S7 tranche 4); the old path is an alias shim.
"""Sitemap INGEST — direct polite fetch of curated docs hosts (wave 1b, task #27).

ADR-013 sanctions sitemaps as the tier-2 anchor route ("software-doc sitemaps";
the HUG sitemap deep-crawl is precedent). This ingester is the sanctioned
sitemap path for the wave-1b LLM-vendor + framework docs hosts (task 080 §2):
given curated queue entries (route sitemap / sitemap-probe / cc-columnar,
wave=1b) it, per host:

  1. CONSENT + ROBOTS FIRST. The host is consent-resolved live
     (cytir/consent/resolver.py, ADR-009). declined → the host is skipped
     ENTIRELY (nothing fetched; card candidate). unknown → skipped, retryable.
     robots.txt (the resolver's cached body) is obeyed per URL for BOTH our
     token (`authoritative-web`) and `*`; Crawl-delay lowers the per-host rate.
  2. Discovers URLs via the host's sitemap(s): robots `Sitemap:` lines, else
     the explicit list in HOST_RULES, else /sitemap.xml | /sitemap_index.xml |
     /sitemap.txt probes. Index sitemaps recurse (depth 2, child cap,
     optional child-name filter). .txt and .gz variants supported.
  3. Filters to the docs path-scope (HOST_RULES include/exclude, seeded from
     the queue's `path_hint`) + per-entry `cap` (≤ --max-per-domain),
     importance-ordered (shallow/short URLs first, discovery_ingest.dedup_rank).
  4. DIRECT polite fetch: ≤ --host-rate req/s per host (default 2; robots
     Crawl-delay lowers it), ≤ --rate aggregate across hosts (default 4),
     the discovery UA, retries/backoff (Retry-After honoured), hard socket +
     httpx deadlines, extraction in the shared process-pool sandbox with its
     hard deadline (discovery_ingest._extract_in_pool — regex-bomb-safe).
  5. One-pass aw_common.process_html → config-B passage rows WITH snippet+text
     (provenance="sitemap-ingest") + the links sidecar (one {url, domain,
     links} row per page — the endorsement graph keeps compounding). Rows
     also carry an `outlinks` column (task 312, AW_OUTLINKS_COLUMN=1 default
     on) — the raw unfiltered per-page link inventory, distinct from the
     endorsement-graph links sidecar above.
  6. L0 originals (specs/corpus-layers.md): these are DIRECT fetches — we are
     the archive — so each page's raw HTML is appended (gzip member) to
     --raw-dir/<host>.raw.jsonl.gz  ({url, final_url, http_status, fetched_at,
     lastmod, sha1, html}). CC-fallback pages store WARC POINTERS instead
     (<host>.ccpointers.jsonl) — Common Crawl is their durable L0.
     --ship-raw rsyncs each finished host's raw file to the build-box corpus
     store and removes the local copy (the orchestrator box disk is tight). ADDITIONALLY
     (task 312, AW_L0=1 default on): every fetched page (both routes) is
     archived to cytir/acquire/l0_store.py's content-addressed store — see that
     module's docstring for why this is a separate layout from the raw-dir
     file above, and specs/architecture.md §4.2 for why CC-fallback pages
     now also get bytes on disk (not pointer-only, superseding the older
     corpus-layers.md guidance for this new store).
  7. Resumable: per-host state with a done-URL set (saved every SAVE_EVERY
     pages); finished hosts are skipped on re-run, unfinished hosts skip
     already-fetched URLs. A stall-kill supervisor can wrap it exactly like
     run_fetch_supervised.sh.

CC-COLUMNAR FALLBACK (route cc-columnar, or sitemap probe finds nothing):
resolve the host's captures via the columnar Parquet cc-index
(cc_columnar.resolve_domains) and range-GET WARC bodies through
discovery_ingest.fetch_warc — throttled by discovery_ingest's global bucket at
--cc-rate (≤4; keep ≤2 if another CC job is running — shared ADR-014 budget).

VERSIONED-DOCS CANONICALIZATION (per-host, documented in HOST_RULES `note`):
prefer the one current/stable version path where the sitemap exposes
duplicates (django /en/5.2/, postgres /docs/current/, rust /stable/, python
/3.14/, node /api/ not /docs/<ver>/, php /manual/en/). docs.anthropic.com
301s to platform.claude.com — fetched there, consent-resolved there.

USAGE
  # probe only (consent + robots + sitemap discovery + selection counts):
  python sitemap_ingest.py fetch --queue ~/aw-data/wave1_queue.json --wave 1b \
      --out ~/aw-data/wave1b/passages.jsonl --state ~/aw-data/wave1b/state.json \
      --raw-dir ~/aw-data/wave1b/raw --discover-only

  # the real run (then: consent_gate.py gate → the retired config-B embed → merge):
  python sitemap_ingest.py fetch --queue ~/aw-data/wave1_queue.json --wave 1b \
      --out ~/aw-data/wave1b/passages.jsonl \
      --links-out ~/aw-data/wave1b/outlinks.jsonl \
      --state ~/aw-data/wave1b/state.json --raw-dir ~/aw-data/wave1b/raw \
      --ship-raw '<host>:<corpus-root>/l0_direct/wave=wave1b_raw/'
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import io
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.robotparser
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

# Started by path (the release's clamp check names this file): the repo root first, so `aw` imports.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from cytir.acquire import cc_columnar as ccx  # noqa: E402
from cytir.consent import resolver as cr  # noqa: E402
from cytir.consent import policy as consent_policy  # noqa: E402
from cytir.acquire import discovery_ingest as di  # noqa: E402  (reuse: sandbox, WARC fetch, throttle)
from cytir.acquire import fetch_process as fp  # noqa: F401,E402 (lazy trafilatura import resolution)
from cytir.consent.resolver import resolve_hosts  # noqa: E402
from cytir.acquire.discovery_ingest import CLASS_SOURCE, CLASS_TRUST, _HTML_CAP, _extract_in_pool, fetch_warc
from cytir.extract.html_extract import outlinks_column_enabled
from cytir.extract.html_extract import extract_content_markdown, extract_outlink_context_records, parse_html_soup
from cytir.extract.html_extract import process_html as _process_html_inproc  # noqa: E402
from cytir.news.extract_published import published_for_page  # noqa: E402,F401  (task 448)
from cytir.news.page_rows import finish_page_rows  # noqa: E402  (task 2783)
from cytir.acquire import preserved_page_fetch as PF  # noqa: E402  (task 450: sealed originals)
from cytir.news import recapture_skip as NRS  # noqa: E402  (ruling 8, fetch-time half)
from cytir.acquire import l0_store  # noqa: E402  (task 312: L0 raw-HTML archive + outlinks column)
from cytir.acquire import gz_shard_store  # noqa: E402  (task 428: crash-safe gzip stores)

# Extraction path. `_extract_in_pool` runs the extractor in discovery_ingest's
# multiprocessing-SPAWN sandbox (regex-bomb safe). That sandbox re-imports the
# __main__ module in each worker — which is `discovery_ingest` when THAT is the
# entrypoint, but `sitemap_ingest` here; re-importing this module (heavy: duckdb
# via cc_columnar) blows the per-page deadline and every extraction silently
# returns empty (task 287: raw HTML fetched, zero passages). Set
# AW_INPROC_EXTRACT=1 to extract in-process via process_html directly — the same
# call cc_ingest.py makes, no sandbox — the correct mode when sitemap_ingest is
# the entrypoint. Default keeps the pooled path for the wave-1b invocation.
_INPROC_EXTRACT = bool(os.environ.get("AW_INPROC_EXTRACT"))
# ground_class / class -> extraction language fallback (process_html still
# detects per page; this only seeds when detection is unsure).
_GC_LANG = {"fr": "fr", "de": "de", "ch": "de", "be": "fr", "ca": "en",
            "eu": "en", "ai": "en", "context7": "en", "tech": "en"}


def _extract_lang(item: dict) -> str:
    gc = (item.get("ground_class") or "").lower()
    for pref, lang in _GC_LANG.items():
        if gc.startswith(pref):
            return lang
    return "en"

UA = di.UA                        # the discovery UA (custom, identifies us)
ROBOT_AGENT = cr.ROBOT_AGENT      # "authoritative-web"
SAVE_EVERY = 25
MIN_FREE_BYTES = 700 * 1024 * 1024   # the orchestrator box / is 97% full — hard floor

# ---------------------------------------------------------------------------
# Per-host rules — canonicalization choices for versioned docs (task 080 §2).
# `host`   : canonical content host actually fetched (queue key may differ)
# `sitemaps`: explicit sitemap URLs (else robots Sitemap: lines, else probes)
# `include`/`exclude`: path-prefix scope filters (include=() → whole host)
# `child_re`: regex a child sitemap URL must match when recursing an index
# ---------------------------------------------------------------------------
HOST_RULES: dict[str, dict] = {
    "docs.anthropic.com": dict(
        host="platform.claude.com",
        sitemaps=["https://platform.claude.com/sitemap.xml"],
        include=("/docs",),
        note="docs.anthropic.com 301→platform.claude.com (robots Sitemap points "
             "there); fetch the canonical host, /docs paths only"),
    "platform.openai.com": dict(
        route="cc-columnar",
        include=("/docs",),
        note="probed 2026-07-04: sitemap.xml lists ONE url (/tokenizer) and "
             "CC has ZERO /docs captures (client-rendered SPA; robots allows "
             "/docs) → no policy-(a) enumeration exists; expected outcome "
             "'empty' — entity-card candidate instead"),
    "modelcontextprotocol.io": dict(
        include=(),
        note="whole host (spec + docs; small)"),
    "docs.djangoproject.com": dict(
        sitemaps=["https://docs.djangoproject.com/sitemap-en.xml"],
        include=("/en/5.2/",),
        note="sitemap index is per-language → sitemap-en.xml only; current "
             "stable /en/5.2/ only (robots itself disallows stale partial "
             "translations; /en/5.2/ is allowed)"),
    "nodejs.org": dict(
        route="cc-columnar",
        include=("/api/", "/learn/"),
        note="probed 2026-07-04: sitemap has /learn/ only, /api/ (the "
             "canonical unversioned latest API docs) is NOT in it → "
             "cc-columnar over both prefixes; versioned /docs/<ver>/api "
             "duplicates excluded by the prefix scope"),
    "postgresql.org": dict(
        host="www.postgresql.org",
        include=("/docs/current/",),
        note="/docs/current/ only — versioned /docs/<N>/ are duplicates"),
    "doc.rust-lang.org": dict(
        route="cc-columnar",
        include=("/stable/",),
        note="probed 2026-07-04: sitemap.txt lists only the 3 channel roots → "
             "cc-columnar, /stable/ scope (3,560 captures in 2026-25; the "
             "queue's path_hint choice — /nightly/, /beta/, /1.N.N/ are "
             "duplicates and out of scope)"),
    "typescriptlang.org": dict(
        host="www.typescriptlang.org",
        route="cc-columnar",
        include=("/docs/",),
        note="no sitemap (robots empty, /sitemap.xml 404) → cc-columnar; "
             "/docs/ scope"),
    "kubernetes.io": dict(
        include=("/docs/",),
        note="/docs/ scope = English docs (translations live under "
             "/<lang>/docs/, excluded by the prefix)"),
    "php.net": dict(
        host="www.php.net",
        route="cc-columnar",
        include=("/manual/en/",),
        note="no sitemap → cc-columnar (task 080); /manual/en/ only — other "
             "languages are translations of the same manual"),
    "huggingface.co": dict(
        route="cc-columnar",
        include=("/docs/", "/learn/"),
        exclude_re=r"^/docs/[^/]+/(?:v\d[\d.]*|main)(?:/|$)"
                   r"|^/docs/[^/]+/(?:de|es|fr|it|ja|ko|pt|te|tr|zh|zh-hans|"
                   r"zh-cn|hi|ar|ru|fa|id|ms|th|vi)(?:/|$)",
        note="probed 2026-07-04: sitemap-doc.xml lists only ~50 package "
             "LANDING pages (per-package sitemaps exist but are version-"
             "pinned) → cc-columnar over /docs/ + /learn/; versioned "
             "(/vN.N/, /main/) and non-EN language dirs excluded — "
             "unversioned EN paths are the canonical alias of stable"),
    "docs.vllm.ai": dict(
        include=(),
        note="whole host; if both /en/stable/ and /en/latest/ appear the "
             "selection dedups by preferring stable (see _prefer_stable)"),
    "docs.python.org": dict(
        route="cc-columnar",
        include=("/3/",),
        note="probed 2026-07-04: sitemap.xml lists only the ~8 VERSION ROOTS "
             "→ cc-columnar. Canonicalization: /3/ (the current-stable alias "
             "— now serving 3.14) NOT /3.14/: the live index's python URLs "
             "are all /3/… and CC has 9× more /3/ captures (922 vs 101). "
             "Same-url+text passages upsert onto the same uuid5 point ids; "
             "changed pages add new ids and STALE 3.13-era ids remain until "
             "the refresh/expiry mechanism (task 080 §2 caveat)"),
    "rts.ch": dict(
        host="www.rts.ch",
        route="sitemap",
        sitemaps=["https://www.rts.ch/sitemaps/pages.xml"],
        exclude_re=(
            r"^/education(?:/|$)"                # evergreen explainer hub, no dates
            r"|(?:^|/)dossiers(?:/|$)"            # topic-tracker dossiers, no article date
            r"|^/emissions(?:/|$)"                # show landing pages
            r"|^/archives(?:/|$)"                 # theme archive index
            r"|^/entreprise(?:/|$)"               # corporate/about pages
            r"|^/religion(?:/|$)"                 # religion section hub
            r"|^/audio-podcast(?:/|$)"            # podcast hub/theme index (not an episode)
            r"|^/rts-(?:espace2|premiere|couleur3|option-musique)(?:/|$)"  # radio-brand pages
            r"|^/jeux-et-quiz"                    # quiz/games hub
            r"|^/newsletters"                     # newsletter signup
            r"|^/play/"                           # media-player chrome
            r"|^/fiction(?:/|$)"                  # drama/entertainment section, not journalism
            r"|^/sport(?:/[a-z0-9_-]+)*/?$"       # sport section/tournament hubs (no numeric-id leaf)
            r"|^/info(?:/(?:[a-z-]+/)*[a-z-]*)?/?$"  # topic-category hubs (real articles carry a
                                                   # numeric id in the leaf segment, so they never
                                                   # match this letters/hyphens-only pattern)
            r"|^/meteo(?:/|$)"                    # weather utility page
            r"|^/programme-tv(?:/|$)"             # TV listings utility page
        ),
        note="task 303/307 root-cause fix: rts.ch has no CC coverage + a "
             "non-conventional sitemap (/sitemaps/pages.xml, injected by "
             "the task-293 rts.ch runner), so nothing scoped its dossier/"
             "section/show-landing pages — 637 of them were classified "
             "disc_class='news' with no extractable published date "
             "(deleted from aw_dchmedia_cov, task 303). This exclude_re is "
             "the recurrence guard: validated 740/740 against "
             "newsfresh_303/chmedia_nodate.jsonl (the historical undated-junk "
             "census) while still passing numeric-id article-shaped paths "
             "(spot-checked, task 307)."),
}

_SKIP_EXT = (".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js",
             ".zip", ".gz", ".tgz", ".bz2", ".epub", ".xml", ".json", ".ico",
             ".webp", ".mp4", ".woff", ".woff2", ".rss", ".atom")

_LOC_RE = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.I | re.S)
_SMAP_BLOCK = re.compile(r"<sitemap[\s>].*?</sitemap>", re.I | re.S)
_URL_BLOCK = re.compile(r"<url[\s>].*?</url>", re.I | re.S)
_LASTMOD_RE = re.compile(r"<lastmod>\s*(.*?)\s*</lastmod>", re.I | re.S)


# --- politeness: aggregate + per-host token buckets --------------------------
class TokenBucket:
    def __init__(self, rate: float):
        self.interval = 1.0 / rate if rate > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def _disk_guard():
    free = shutil.disk_usage(os.path.expanduser("~")).free
    if free < MIN_FREE_BYTES:
        raise SystemExit(f"DISK LOW: {free/1e6:.0f} MB free < "
                         f"{MIN_FREE_BYTES/1e6:.0f} MB floor — aborting "
                         f"(ship/clean ~/aw-data and re-run; state resumes)")


def _host_of(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def _display_domain(host: str) -> str:
    return re.sub(r"^www\d*\.", "", host)


# --- robots ------------------------------------------------------------------
def load_robots(host: str, cache_dir: str):
    """RobotFileParser from the consent resolver's CACHED robots body (the
    resolver ran first — no extra request). (None, delay) when the host
    publishes no rules."""
    p = cr._cache_path(cache_dir, host)
    body = ""
    if p.exists():
        try:
            body = json.loads(p.read_text()).get("_robots_body", "") or ""
        except Exception:  # noqa: BLE001
            body = ""
    if not body.strip():
        return None, 0.0
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(body.splitlines())
    delay = 0.0
    try:
        delay = float(rp.crawl_delay(ROBOT_AGENT) or rp.crawl_delay("*") or 0)
    except Exception:  # noqa: BLE001
        delay = 0.0
    return rp, delay


def robots_ok(rp, url: str) -> bool:
    if rp is None:
        return True
    try:
        return rp.can_fetch(ROBOT_AGENT, url) and rp.can_fetch("*", url)
    except Exception:  # noqa: BLE001
        return False


# --- sitemap discovery ---------------------------------------------------
_SITEMAP_DECOMPRESSED_CAP = 64 * 1024 * 1024
_SITEMAP_GZIP_MAX_LAYERS = 8


def _decode_sitemap_bytes(raw: bytes) -> str | None:
    """Decode a bounded, possibly nested-gzip sitemap response.

    Some publisher CDNs apply gzip repeatedly while retaining the same XML
    content type (RTS was four layers deep on 2026-08-26).  HTTP clients remove
    at most the transport layer, so unwrap remaining magic-marked layers while
    bounding both nesting and expanded bytes.
    """
    layers = 0
    while raw[:2] == b"\x1f\x8b":
        if layers >= _SITEMAP_GZIP_MAX_LAYERS:
            return None
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as source:
                expanded = source.read(_SITEMAP_DECOMPRESSED_CAP + 1)
        except (OSError, EOFError):
            return None
        if len(expanded) > _SITEMAP_DECOMPRESSED_CAP:
            return None
        raw = expanded
        layers += 1
    return raw.decode("utf-8", "ignore")


def _fetch_body(client, url, gbucket, hbucket, tries=3):
    """Polite GET of a sitemap file; transparent gunzip. Returns text or None."""
    for attempt in range(tries):
        gbucket.wait(); hbucket.wait()
        try:
            r = client.get(url)
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (attempt + 1)); continue
        if r.status_code == 200:
            return _decode_sitemap_bytes(r.content)
        if r.status_code in (429, 503):
            ra = r.headers.get("retry-after", "")
            time.sleep(min(float(ra) if ra.isdigit() else 5 * (attempt + 1), 60))
            continue
        return None
    return None


def rule_for_host(qdom: str) -> dict:
    """HOST_RULES entry for a queue domain, `www.`-folded (task 676).

    The news queue names hosts `www.rts.ch`; HOST_RULES is keyed `rts.ch`.
    A plain `.get(qdom)` never matched, so rts.ch's own exclusion rules (hubs,
    dossiers, show landing pages) were never applied: the daily run fetched all
    609 sitemap URLs of rts.ch every day at the host's `Crawl-delay: 10` (about
    100 minutes of a 2 hour run) and every one of them is excluded by the rule
    (measured on stage-2026-09-28: 609 fetched, 0 survive `exclude_re`)."""
    q = (qdom or "").lower()
    if q in HOST_RULES:
        return HOST_RULES[q]
    bare = q[4:] if q.startswith("www.") else q
    return HOST_RULES.get(bare, {})


# Child sitemaps that name the period they cover in their own URL. A freshness
# run needs only the children that can hold entries inside its window; a global
# child budget spent in discovery order starves later seeds (swissinfo: the
# `/eng/` seed's 272 per-day children consumed the whole 200-child budget, so
# the daily run fetched English only -- 798 of 798 URLs on 2026-09-25).
_CHILD_DAY_QS = re.compile(r"sitemap-(\d{4})\.xml\?mm=(\d{2})&(?:amp;)?dd=(\d{2})")
_CHILD_YMD = re.compile(r"/(\d{4})_(\d{2})_(\d{2})\.xml(?:$|\?)")
_CHILD_YM = re.compile(r"/(\d{4})_(\d{2})\.xml(?:$|\?)")


def child_sitemap_end(url: str):
    """Latest date a child sitemap can hold, from its own URL; None if the URL
    names no period (then it is never pruned)."""
    from datetime import date, timedelta
    try:
        m = _CHILD_DAY_QS.search(url) or _CHILD_YMD.search(url)
        if m:
            return date(int(m[1]), int(m[2]), int(m[3]))
        m = _CHILD_YM.search(url)
        if m:
            y, mo = int(m[1]), int(m[2])
            first_next = date(y + (mo == 12), 1 if mo == 12 else mo + 1, 1)
            return first_next - timedelta(days=1)
    except ValueError:
        return None
    return None


def window_cutoff_date(now_ref, months: int):
    """The oldest date still in a `months`-month window ending at `now_ref`
    (the same arithmetic `select_recent` uses)."""
    if isinstance(now_ref, datetime):
        now_ref = now_ref.date()
    year, month = now_ref.year, now_ref.month - months
    while month <= 0:
        month += 12
        year -= 1
    day = now_ref.day
    while True:
        try:
            return datetime(year, month, day).date()
        except ValueError:
            day -= 1


def discover_sitemap_urls(client, cfg, cache_dir, gbucket, hbucket,
                          max_children=60, window_start=None):
    """Sitemap(s) → [{url, lastmod}]. Recurses index sitemaps (depth 2).

    `window_start` (a date, freshness runs only): a child whose URL names a
    period that ended before it is skipped and does NOT count against
    `max_children`."""
    host = cfg["host"]
    seeds = list(cfg.get("sitemaps") or [])
    if not seeds:                       # robots Sitemap: lines (cached body)
        p = cr._cache_path(cache_dir, host)
        if p.exists():
            try:
                body = json.loads(p.read_text()).get("_robots_body", "") or ""
                seeds = [m.strip() for m in re.findall(
                    r"(?im)^\s*sitemap\s*:\s*(\S+)", body)]
            except Exception:  # noqa: BLE001
                seeds = []
    if not seeds:                       # conventional probes
        seeds = [f"https://{host}/sitemap.xml",
                 f"https://{host}/sitemap_index.xml",
                 f"https://{host}/sitemap.txt"]
    child_re = re.compile(cfg["child_re"]) if cfg.get("child_re") else None
    out, seen_sm, seen_url = [], set(), set()
    queue = [(u, 0) for u in seeds]
    n_children = 0
    while queue:
        sm_url, depth = queue.pop(0)
        if sm_url in seen_sm:
            continue
        seen_sm.add(sm_url)
        body = _fetch_body(client, sm_url, gbucket, hbucket)
        if body is None:
            continue
        stripped = body.lstrip()
        if stripped.startswith("<"):
            if "<sitemapindex" in stripped[:2000].lower():
                kids = []
                for blk in _SMAP_BLOCK.findall(body):
                    m = _LOC_RE.search(blk)
                    if m:
                        # XML escapes query separators (`&amp;`) in sitemap
                        # child URLs.  HTTP clients must receive the decoded
                        # URL; otherwise a valid index returns no child
                        # entries and freshness misreports a source outage
                        # (Swissinfo, 2026-08-05).
                        kids.append(html.unescape(m.group(1).strip()))
                if not kids:            # lenient: some indexes skip <sitemap>
                    kids = [html.unescape(u.strip()) for u in _LOC_RE.findall(body)]
                for k in kids:
                    if depth >= 2 or n_children >= max_children:
                        break
                    if child_re and not child_re.search(k):
                        continue
                    if window_start is not None:
                        end = child_sitemap_end(k)
                        if end is not None and end < window_start:
                            continue
                    queue.append((k, depth + 1)); n_children += 1
            else:
                blocks = _URL_BLOCK.findall(body)
                if blocks:
                    for blk in blocks:
                        m = _LOC_RE.search(blk)
                        if not m:
                            continue
                        u = html.unescape(m.group(1).strip())
                        if u not in seen_url:
                            seen_url.add(u)
                            lm = _LASTMOD_RE.search(blk)
                            out.append({"url": u,
                                        "lastmod": lm.group(1).strip() if lm else None})
                else:                   # lenient urlset without <url> blocks
                    for u in _LOC_RE.findall(body):
                        u = html.unescape(u.strip())
                        if u not in seen_url:
                            seen_url.add(u)
                            out.append({"url": u, "lastmod": None})
        else:                           # plain-text sitemap (rust sitemap.txt)
            for line in body.splitlines():
                u = html.unescape(line.strip())
                if u.startswith("http") and u not in seen_url:
                    seen_url.add(u)
                    out.append({"url": u, "lastmod": None})
        if out and not queue:
            break
    return out


# --- path-scoped columnar CC resolution ---------------------------------
def resolve_cc_host(crawl: str, host: str, include: tuple, limit: int,
                    con) -> list[dict]:
    """Columnar cc-index resolve, PATH-SCOPED: a surtkey is `tld,dom)/path`,
    so `co,huggingface)/docs/` prunes to exactly the docs subtree — without
    this, per_domain shallow-first fills the cap with off-scope pages on big
    hosts. No include prefixes → whole-host via ccx.resolve_domains.
    Raises on query failure (transient → caller surfaces 'error', never
    'empty' — memory cc-cdx-outage-silent-swallow)."""
    dom = host.removeprefix("www.")
    if not include:
        got = ccx.resolve_domains(crawl, [dom], con=con, per_domain=limit)
        return got.get(dom, [])
    prefixes = []
    for inc in include:
        k = ccx.url_surtkey(f"http://{dom}{inc}")
        if k:
            # the surtkey drops a trailing slash — restore it or '/3/' scope would
            # range-match '/3.15/', '/3.16/' … (verified 2026-07-04)
            if inc.endswith("/") and not k.endswith("/"):
                k += "/"
            prefixes.append(k)
    if not prefixes:
        return []
    bounds = ccx.file_bounds(crawl, con)
    by_file: dict[str, list[str]] = {}
    for p in prefixes:
        lo, hi = p, p + "￿"
        for fn, mn, mx in bounds:
            if not (mx < lo or mn > hi):
                by_file.setdefault(fn, []).append(p)
    out: list[dict] = []
    for fn, pps in by_file.items():
        ors = " OR ".join(
            f"(p.url_surtkey >= '{pp}' AND p.url_surtkey < '{pp}￿')"
            for pp in pps)
        q = f"""
        WITH hits AS (
          SELECT p.url_surtkey, p.url, p.warc_filename,
                 p.warc_record_offset, p.warc_record_length,
                 p.content_digest,
                 row_number() OVER (PARTITION BY p.url_surtkey
                            ORDER BY p.warc_record_length DESC) AS rn
          FROM read_parquet('{fn}') p
          WHERE ({ors}) AND p.fetch_status = 200
            AND p.content_mime_type LIKE 'text/html%'
        )
        SELECT url, warc_filename, warc_record_offset, warc_record_length,
               content_digest
        FROM hits WHERE rn = 1
        ORDER BY length(url) LIMIT {int(limit)}"""
        for url, wfn, off, ln, dig in con.execute(q).fetchall():
            out.append({"url": url, "warc_filename": wfn, "offset": off,
                        "length": ln, "digest": dig})
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break
    return out


def _prefer_stable(urls: list[dict]) -> list[dict]:
    """When BOTH /en/stable/ and /en/latest/ (readthedocs-style) versions of
    the same page are discovered, keep the stable one."""
    paths = {urllib.parse.urlsplit(u["url"]).path for u in urls}
    has_stable = any("/stable/" in p for p in paths)
    if not has_stable:
        return urls
    out = []
    for u in urls:
        p = urllib.parse.urlsplit(u["url"]).path
        if "/latest/" in p and p.replace("/latest/", "/stable/") in paths:
            continue
        out.append(u)
    return out


def _filtered_urls(urls: list[dict], cfg, rp) -> list[dict]:
    """Apply the shared scope, hygiene, and robots filters before ordering."""
    host = cfg["host"]
    allowed_hosts = {host, "www." + host, host.removeprefix("www.")}
    include = tuple(cfg.get("include") or ())
    exclude = tuple(cfg.get("exclude") or ())
    exclude_re = re.compile(cfg["exclude_re"]) if cfg.get("exclude_re") else None
    alias = cfg.get("alias_prefix")           # e.g. "/stable": /stable/x ≡ /x
    if alias:
        paths = {urllib.parse.urlsplit(u["url"]).path for u in urls}
        urls = [u for u in urls
                if not (urllib.parse.urlsplit(u["url"]).path.startswith(alias + "/")
                        and urllib.parse.urlsplit(u["url"]).path[len(alias):]
                        in paths)]
    seen, kept = set(), []
    for u in _prefer_stable(urls):
        raw = u["url"].split("#")[0].strip()
        try:
            parts = urllib.parse.urlsplit(raw)
        except Exception:  # noqa: BLE001
            continue
        if parts.scheme not in ("http", "https"):
            continue
        if (parts.hostname or "").lower() not in allowed_hosts:
            continue
        if parts.query:                      # docs pages with queries = chrome
            continue
        path = parts.path or "/"
        if path.lower().endswith(_SKIP_EXT):
            continue
        if include and not any(path.startswith(p) for p in include):
            continue
        if exclude and any(path.startswith(p) for p in exclude):
            continue
        if exclude_re and exclude_re.search(path):
            continue
        if not robots_ok(rp, raw):
            continue
        if raw in seen:
            continue
        seen.add(raw)
        kept.append({"url": raw, "lastmod": u.get("lastmod")})
    return kept


def select_urls(urls: list[dict], cfg, cap: int, rp) -> list[dict]:
    """Scope filter + hygiene + robots + importance order + cap."""
    kept = _filtered_urls(urls, cfg, rp)
    kept.sort(key=lambda r: (r["url"].count("/"), len(r["url"])))
    return kept[:cap]


def _parse_lastmod(lastmod):
    if not lastmod:
        return None
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(lastmod))
    if not match:
        return None
    try:
        return datetime(*(int(match[index]) for index in range(1, 4))).date()
    except ValueError:
        return None


def page_date_clamp_day(now_ref, today=None):
    """The day a PAGE date is clamped against: the real current day, never earlier
    than the run's as-of day.

    Task 676, orchestrator ruling 2026-09-30. `now_ref` is the run's as-of day; a
    backfill run uses a free past as-of day (the daily timer owns the recent ones),
    so clamping page dates to `now_ref + 1 day` threw away every page date after it
    (a mis-parsed far-future date is worse than none, but a real date of the last
    ten days is not one). Selection (`select_recent`) still uses the as-of day.
    """
    if isinstance(now_ref, datetime):
        now_ref = now_ref.date()
    today = today or datetime.utcnow().date()
    return max(today, now_ref) if now_ref else today


def select_recent(urls: list[dict], cfg, cap: int, rp, now_ref,
                  months: int = 12) -> list[dict]:
    """Select newest in-window sitemap entries, then undated backfill rows."""
    if months < 1:
        raise ValueError("months must be positive")
    if isinstance(now_ref, datetime):
        now_ref = now_ref.date()
    cutoff = window_cutoff_date(now_ref, months)

    dated, undated = [], []
    for row in _filtered_urls(urls, cfg, rp):
        value = _parse_lastmod(row.get("lastmod"))
        if value is None:
            undated.append(row)
        elif cutoff <= value <= now_ref:
            dated.append({**row, "published": value.isoformat(), "_date": value})
    dated.sort(key=lambda row: row["_date"], reverse=True)
    for row in dated:
        row.pop("_date")
    return (dated + undated)[:cap]


def _consent_stamp(consent: dict) -> dict:
    """Persist the exact host decision that admitted a direct source fetch."""
    checked_at = consent.get("checked_at")
    if consent.get("search_ok") is not True or not checked_at:
        raise ValueError("resolved indexing consent requires search_ok and checked_at")
    return {
        "search_ok": True,
        "train_ok": consent.get("train_ok") is True,
        "consent_state": "resolved",
        "consent_basis": "robots-live-resolve",
        "consent_source": "consent_resolver",
        "consent_checked_at": checked_at,
    }


def _search_index_bucket(consent: dict) -> str:
    """Classify a resolved receipt for search indexing, not model training."""
    return consent_policy.gate_bucket(consent, search_indexing=True)


# --- one host -----------------------------------------------------------------
def fetch_page(client, url, gbucket, hbucket, allowed_hosts, tries=3,
               preserve_root=None):
    """Polite direct GET. -> (final_url, status, html|None, note, receipt|None).

    Without `preserve_root` the behaviour is exactly what it always was and the
    receipt is None.

    With `preserve_root` (task 450) the response is **sealed before it is
    read**: exact bytes, transport headers and the outbound request identity go
    into the content-addressed preserved store, and the text returned here is
    decoded from the *validated* artifact. A publisher-derived row therefore
    cannot exist without preserved bytes. A fetch that cannot be sealed yields
    no text at all — it is never downgraded to unevidenced content.
    """
    if preserve_root is not None:
        return _fetch_page_preserved(
            client, url, gbucket, hbucket, allowed_hosts, tries, preserve_root)
    for attempt in range(tries):
        gbucket.wait(); hbucket.wait()
        try:
            r = client.get(url)
        except Exception as e:  # noqa: BLE001
            if attempt == tries - 1:
                return url, 0, None, f"conn:{type(e).__name__}", None
            time.sleep(1.5 * (attempt + 1)); continue
        if r.status_code in (429, 503):
            ra = r.headers.get("retry-after", "")
            time.sleep(min(float(ra) if ra.isdigit() else 5 * (attempt + 1), 90))
            continue
        if 500 <= r.status_code <= 599:
            time.sleep(2 * (attempt + 1)); continue
        if r.status_code != 200:
            return str(r.url), r.status_code, None, f"http {r.status_code}", None
        if "html" not in r.headers.get("content-type", "text/html"):
            return str(r.url), 200, None, "non-html", None
        fhost = _host_of(str(r.url))
        if fhost not in allowed_hosts:
            return str(r.url), 200, None, f"off-host redirect → {fhost}", None
        return str(r.url), 200, r.text, None, None
    return url, 0, None, "retries exhausted", None


def _fetch_page_preserved(client, url, gbucket, hbucket, allowed_hosts, tries,
                          preserve_root):
    """The preserving twin of `fetch_page`; see its docstring."""
    root = pathlib.Path(preserve_root)
    # Resume-safe: an already-sealed fetch for this exact requested URL is
    # immutable evidence, so reuse it instead of re-GETting the publisher.
    receipt = PF.valid_receipt(root, PF.receipt_path(root, url), url)
    if receipt is None:
        response = None
        for attempt in range(tries):
            gbucket.wait(); hbucket.wait()
            try:
                r = client.get(url)
            except Exception as e:  # noqa: BLE001
                if attempt == tries - 1:
                    return url, 0, None, f"conn:{type(e).__name__}", None
                time.sleep(1.5 * (attempt + 1)); continue
            if r.status_code in (429, 503):
                ra = r.headers.get("retry-after", "")
                time.sleep(min(float(ra) if ra.isdigit() else 5 * (attempt + 1), 90))
                continue
            if 500 <= r.status_code <= 599:
                time.sleep(2 * (attempt + 1)); continue
            response = r
            break
        if response is None:
            return url, 0, None, "retries exhausted", None
        fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            receipt = PF.seal_page_fetch(root, url, response, fetched_at,
                                         UA["User-Agent"])
            PF.publish_receipt(root, url, receipt)
        except Exception as e:  # noqa: BLE001 — unpreservable fetch -> no row
            return (str(response.url), response.status_code, None,
                    f"unpreserved:{type(e).__name__}", None)
    final_url = receipt["final_url"]
    status = receipt["http_status"]
    if status != 200:
        return final_url, status, None, f"http {status}", receipt
    if not PF.is_admissible_article(receipt):
        return final_url, status, None, "non-html", receipt
    fhost = _host_of(final_url)
    if fhost not in allowed_hosts:
        return final_url, status, None, f"off-host redirect → {fhost}", receipt
    try:
        body = PF.load_body(root, receipt)
    except Exception as e:  # noqa: BLE001 — evidence that will not validate
        return final_url, status, None, f"unreadable:{type(e).__name__}", receipt
    return (final_url, status,
            PF.decode_body(body, receipt.get("content_type")), None, receipt)


# --- task 450: the news_article document record ------------------------------
# One row per preserved page that yielded principal content. It is the
# `markdown` + `outlinks` evidence the `news_article` profile requires, written
# by the run that actually produced the chunks so the lineage is real rather
# than reconstructed afterwards. The Markdown envelope is deliberately the SAME
# shape the already-admitted news sources use (see
# `cytir.ops.audits.news_article_profile`), so one auditor reads every news lane.
NEWS_DOCUMENT_SCHEMA = "aw.news-lane-document.v1"
NEWS_BYLINE_ABSENT = "not declared by publisher"


def news_document_id(url: str) -> str:
    """Stable document identity shared with the admitted news materializers."""
    return hashlib.sha256(f"news-document\0{url}".encode()).hexdigest()


def _publisher_metadata(html: str):
    """(headline, byline, publication_time) as the publisher declared them."""
    headline = byline = published = None
    try:
        # task 760: a date-stage exception (htmldate: +25:00 offsets, Turkish
        # month names) leaves headline/byline intact and `published` None
        # instead of discarding the whole publisher metadata.
        from cytir.extract.html_extract import coerce_title, extract_metadata_dateguard
        meta, _date_error = extract_metadata_dateguard(html)
    except Exception:  # noqa: BLE001 — absence stays explicit, never a guess
        meta = None
    if meta is not None:
        headline = (coerce_title(getattr(meta, "title", None)) or "").strip() or None
        byline = (getattr(meta, "author", None) or "").strip() or None
        published = (getattr(meta, "date", None) or "").strip() or None
    return headline, byline, published


def _news_document_record(*, url, document_id, html, markdown, receipt,
                          publisher, published, outlinks, point_ids) -> dict:
    headline, byline, meta_published = _publisher_metadata(html)
    headline = headline or url
    publication_time = published or meta_published
    fetched_at = receipt["fetched_at"]
    envelope = (
        f"# {headline}\n\n"
        f"Byline: {byline or NEWS_BYLINE_ABSENT}\n\n"
        f"Publisher: {publisher}\n\n"
        f"Fetched: {fetched_at}\n\n"
        f"Source: {url}\n\n"
        f"{markdown}\n")
    return {
        "schema": NEWS_DOCUMENT_SCHEMA,
        "document_id": document_id,
        "url": url,
        "headline": headline,
        "byline": byline,
        "byline_status": ("source_declared" if byline
                          else "not_declared_by_publisher"),
        "publication_time": publication_time,
        "publication_time_status": ("source_declared" if publication_time
                                    else "not_declared_by_publisher"),
        "publisher": publisher,
        "fetched_at": fetched_at,
        "markdown": envelope,
        "markdown_sha256": hashlib.sha256(envelope.encode("utf-8")).hexdigest(),
        "body_markdown_sha256": hashlib.sha256(
            markdown.encode("utf-8")).hexdigest(),
        "lineage": {
            "fetched_at": fetched_at,
            "original_sha256": receipt["body_sha256"],
            "original_bytes": receipt["body_bytes"],
            "http_status": receipt["http_status"],
            "final_url": receipt["final_url"],
            "receipt_object_id": receipt["object_id"],
            "transport_sha256":
                receipt["artifacts"]["transport"]["body_sha256"],
        },
        "chunk_point_ids": list(point_ids),
        "outlinks": [
            {"ordinal": ordinal, "target_url": row["url"],
             "anchor_or_context": row["anchor_or_context"],
             "nofollow": row["nofollow"]}
            for ordinal, row in enumerate(outlinks)],
    }


class HostResult:
    def __init__(self, qdom):
        self.qdom = qdom
        self.status = "pending"      # ok | empty | error | skipped-consent-*
        self.route = ""
        self.discovered = 0
        self.selected = 0
        self.fetched = 0
        self.failed = 0
        self.rows = 0
        self.links = 0
        self.note = ""


def _record_recapture_skips(args, ctx, qdom: str, skipped: list[dict]) -> None:
    """Append this host's fetch-time skips to `recapture_skipped.jsonl` next to the
    state file (one line per url), so a skipped url has a terminal and a reason."""
    if not skipped:
        return
    path = pathlib.Path(os.path.expanduser(args.state)).parent / "recapture_skipped.jsonl"
    with ctx["write_lock"], path.open("a", encoding="utf-8") as fh:
        for entry in skipped:
            fh.write(json.dumps({"domain": qdom, **entry}, sort_keys=True) + "\n")


def process_host(item, args, ctx) -> HostResult:
    qdom = item["domain"]
    cfg = dict(rule_for_host(qdom))
    cfg.setdefault("host", qdom)
    if "include" not in cfg and item.get("path_hint"):
        cfg["include"] = (item["path_hint"],)   # explicit include=() = whole host
    if item.get("sitemaps"):
        cfg["sitemaps"] = item["sitemaps"]
    if item.get("include") is not None:
        cfg["include"] = tuple(item["include"])
    host = cfg["host"]
    cls = item.get("class", "tech")
    cap = min(int(item.get("cap") or args.max_per_domain), args.max_per_domain)
    res = HostResult(qdom)
    res.route = cfg.get("route") or item.get("route") or "sitemap"
    if ctx.get("no_cc_fallback") and res.route == "cc-columnar":
        # sitemap-only mode (task 287): a HOST_RULES cc-columnar route is moot
        # when CC does not hold the host — try its sitemap instead of CC.
        res.route = "sitemap"
    st = ctx["state"]["hosts"].setdefault(qdom, {})
    done_urls = set(st.get("done_urls", []))

    # ---- 1. consent gate: skip the host entirely unless resolved ok --------
    # direct_crawl: our copy is a DIRECT origin fetch (not a CC snapshot). It
    # is recorded as provenance (mode + cache key). Task 287's CC-invisible
    # route used to DEPEND on it, because a CCBot-disallow declined search in
    # CC mode and would have wrongly excluded CCBot-blocking gov/law hosts;
    # ADR-024 am.5 removed that dependency — a CCBot-disallow now stamps
    # train_ok=false in both modes and never declines search.
    cres = resolve_hosts([host], cache_dir=ctx["cache_dir"], progress=False,
                         direct_crawl=ctx.get("direct_crawl", False))
    consent = cres[host]
    verdict = _search_index_bucket(consent)
    if verdict not in {"ok", "ok_search_only"}:
        res.status = f"skipped-consent-{verdict}"
        res.note = "; ".join(cres[host].get("reasons", [])[:2])
        st.update(status=res.status, note=res.note)
        return res
    consent_stamp = _consent_stamp(consent)
    rp, crawl_delay = load_robots(host, ctx["cache_dir"])
    host_rate = args.host_rate
    if crawl_delay > 0:
        host_rate = min(host_rate, 1.0 / crawl_delay)
    hbucket = TokenBucket(host_rate)

    # ---- 2. discover URLs ----------------------------------------------------
    client = ctx["client"]
    selected: list[dict] = []
    if res.route in ("sitemap", "sitemap-probe"):
        if item.get("urls"):
            # An explicit url list (the RTS section-pagination backfill: the site has
            # no article sitemap). Entries are urls or {url, lastmod}; they go through
            # the same scope, robots, selection and fetch path as sitemap urls.
            urls = [{"url": (u["url"] if isinstance(u, dict) else u),
                     "lastmod": (u.get("lastmod") if isinstance(u, dict) else None)}
                    for u in item["urls"]]
        else:
            urls = discover_sitemap_urls(
                client, cfg, ctx["cache_dir"], ctx["gbucket"], hbucket,
                max_children=args.max_child_sitemaps,
                window_start=(window_cutoff_date(ctx["now_ref"], args.window_months)
                              if getattr(args, "freshness", False) else None))
        res.discovered = len(urls)
        if urls:
            res.route = "sitemap"
            if getattr(args, "freshness", False):
                selected = select_recent(
                    urls, cfg, cap, rp, ctx["now_ref"], args.window_months)
                res.note += (f"freshness: {len(selected)} in "
                             f"{args.window_months}mo window "
                             f"(of {len(urls)} discovered); ")
            else:
                selected = select_urls(urls, cfg, cap, rp)
        elif getattr(args, "freshness", False):
            res.status = "error"
            res.note = ("freshness: no sitemap (egress?) — no Common Crawl "
                        "fallback; retry the source")
            st.update(status="error", note=res.note)
            return res
        elif ctx.get("no_cc_fallback"):
            # pure-sitemap route (task 287): CC does not hold this host at all,
            # so there is nothing to fall back TO — record 'empty', not 'error'.
            res.status = "empty"
            res.note = "no sitemap urls (no-cc-fallback)"
            st.update(status="empty", note=res.note)
            return res
        else:
            res.route = "cc-columnar"       # probe found nothing → fallback
            res.note = "no sitemap found → cc fallback; "
    if res.route == "cc-columnar":
        recs = ctx["cc_resolved"].get(qdom)
        if recs is None:
            res.status = "error"
            res.note += "cc resolve errored (transient) — re-run retries"
            return res
        urls = [{"url": r["url"], "lastmod": None, "_cc": r} for r in recs]
        res.discovered = len(urls)
        by_url = {r["url"].split("#")[0].strip(): r.get("_cc") for r in urls}
        selected = select_urls(urls, cfg, cap, rp)
        for s in selected:
            s["_cc"] = by_url.get(s["url"])
    recapture = ctx.get("recapture")
    if recapture and res.route == "sitemap" and getattr(args, "freshness", False):
        # Ruling 8, fetch-time half: an already-ingested url older than the
        # overhang is not fetched. Every skip is written to
        # recapture_skipped.jsonl beside the state file, never just dropped.
        try:
            selected, skipped = NRS.skip_served(
                selected, now_ref=ctx["now_ref"], client=ctx["recapture_client"],
                **recapture)
        except NRS.RecaptureLookupError as exc:
            res.status = "error"
            res.note += f"recapture lookup failed ({exc}) -- retry the source"
            st.update(status="error", note=res.note)
            return res
        _record_recapture_skips(args, ctx, qdom, skipped)
        st.update(recapture_skipped=len(skipped), recapture_overhang_days=recapture[
            "overhang_days"])
        res.note += (f"recapture-overhang: skipped {len(skipped)} already "
                     f"ingested >{recapture['overhang_days']}d old; ")
    res.selected = len(selected)
    if not selected:
        res.status = "empty"
        st.update(status="empty", note=res.note)
        return res
    if args.discover_only:
        res.status = "discover-only"
        return res

    # ---- 3/4/5/6. fetch → L0 raw → extract → rows/links ----------------------
    common = dict(domain=_display_domain(host), tier="C",
                  trust_tier=CLASS_TRUST.get(cls, 0.75),
                  source_type=CLASS_SOURCE.get(cls, "docs"),
                  last_seen=time.strftime("%Y-%m-%d"))
    allowed_hosts = {host, "www." + host, host.removeprefix("www.")}
    raw_dir = pathlib.Path(os.path.expanduser(args.raw_dir))
    raw_dir.mkdir(parents=True, exist_ok=True)
    is_cc = res.route == "cc-columnar"
    preserve_root = ctx.get("preserve_root")
    if preserve_root is not None and is_cc:
        # A CC capture is not a response we made; it cannot carry our request
        # identity, so it can never satisfy `news_article.original`. Refuse the
        # host rather than emit unpreservable rows beside sealed ones.
        res.status = "error"
        res.note += ("preserved fetch requested but this host routes to "
                     "cc-columnar; no request identity is sealable")
        st.update(status="error", note=res.note)
        return res
    raw_path = raw_dir / (f"{host}.ccpointers.jsonl" if is_cc
                          else f"{host}.raw.jsonl.gz")
    # Task 428: the gzip branch must NOT append into an existing file — a
    # run killed mid-write leaves a torn gzip member, and a later member
    # appended behind it makes the whole file unreadable from byte 0 (the
    # incident that cost three L0 waves). Rotate to `<host>.rN.raw.jsonl.gz`
    # instead; `glob("*.raw.jsonl.gz")` readers still match it. The CC branch
    # is plain-text JSONL — an append there has no framing to tear.
    if is_cc:
        raw_f = open(raw_path, "a", encoding="utf-8")
    else:
        raw_f, raw_path = gz_shard_store.create_rotated_gzip_text(
            str(raw_path), suffix=".raw.jsonl.gz")
    n_since_save = 0
    try:
        for i, s in enumerate(selected, 1):
            url = s["url"]
            if url in done_urls:
                continue
            _disk_guard()
            if is_cc:
                rec = s.get("_cc")
                if not rec:
                    continue
                html = fetch_warc(client, {
                    "url": url, "offset": rec["offset"],
                    "length": rec["length"],
                    "filename": rec["warc_filename"]})
                final_url, status_code, note = url, 200, None
                page_receipt = None
                if html:
                    raw_f.write(json.dumps({
                        "url": url, "crawl": args.crawl,
                        "warc_filename": rec["warc_filename"],
                        "offset": rec["offset"], "length": rec["length"],
                        "digest": rec.get("digest")},
                        ensure_ascii=False) + "\n")
            else:
                final_url, status_code, html, note, page_receipt = fetch_page(
                    client, url, ctx["gbucket"], hbucket, allowed_hosts,
                    preserve_root=preserve_root)
                if html:
                    # Compatibility sidecar, byte-for-byte the store it always
                    # was, so every existing `glob("*.raw.jsonl.gz")` reader
                    # keeps its exact current meaning. It is NOT a preservation
                    # stage: the sealed originals under --preserve-dir are.
                    raw_f.write(json.dumps({
                        "url": url, "final_url": final_url,
                        "http_status": status_code,
                        "fetched_at": (page_receipt["fetched_at"] if page_receipt
                                       else time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                          time.gmtime())),
                        "lastmod": s.get("lastmod"),
                        "sha1": hashlib.sha1(
                            html.encode("utf-8", "ignore")).hexdigest(),
                        "ua": UA["User-Agent"],
                        "html": html[:_HTML_CAP]}, ensure_ascii=False) + "\n")
            if not html:
                res.failed += 1
                done_urls.add(url)      # 4xx/non-html: don't refetch on resume
                continue
            res.fetched += 1
            purl = (final_url or url).split("#")[0]
            # task 312: L0 raw-HTML archive — additive alongside the existing
            # --raw-dir per-host file above (unchanged), non-fatal on failure,
            # AW_L0=1 default on. Written for BOTH routes (direct AND
            # cc-columnar): architecture.md §4.2 supersedes corpus-layers.md's
            # older "CC content = pointers only" stance — with bytes on disk,
            # extract@N+1 is a local read instead of a re-fetch either way.
            if l0_store.l0_enabled():
                try:
                    l0_store.write(
                        purl, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        html.encode("utf-8", "ignore"),
                        extra={"host": host, "route": res.route})
                except Exception as e:  # noqa: BLE001 — archival side-effect only
                    print(f"  ! L0 write failed {purl}: {e}", flush=True)
            _emeta = {
                "content": True, "links": True,
                "passage_common": common, "lang_default": _extract_lang(item),
                "digest": (s.get("_cc") or {}).get("digest") if is_cc else None,
                "outlinks_column": outlinks_column_enabled(),
                "sitemap_lastmod": s.get("lastmod"),
                "now_ref": ctx.get("now_ref"),
                "published_htmldate": getattr(args, "freshness", False),
            }
            # Task 450: in preserved mode the chunk input is the Markdown this
            # run seals as the `markdown` stage — extracted ONCE from the
            # untruncated preserved bytes and handed to the chunker through
            # `content_markdown`, so `markdown -> chunks` is a real lineage
            # edge and not an assumption that a second extraction reproduced
            # the retained document. `_HTML_CAP` is deliberately NOT applied
            # there: a pre-chunk prefix cap is exactly what
            # `preservation_contract._validate_content_fidelity` forbids.
            document_markdown = None
            page_outlinks: list[dict] = []
            if preserve_root is not None:
                document_markdown, _md_mode = extract_content_markdown(html, purl)
                if document_markdown:
                    _emeta["content_markdown"] = document_markdown
                    _emeta["content_markdown_mode"] = "preserved"
                page_outlinks = extract_outlink_context_records(
                    parse_html_soup(html), purl)
            document_id = news_document_id(purl)
            extract_input = html if preserve_root is not None else html[:_HTML_CAP]
            if _INPROC_EXTRACT:
                # in-process (no sandbox): process_html without want_stats -> 2-tuple
                passages, links = _process_html_inproc(purl, extract_input, _emeta)
            else:
                # discovery_ingest sandbox: 4-tuple (passages, links,
                # reject-stats, timed_out) since task 634's circuit breaker
                passages, links, _rej, _timed_out = _extract_in_pool(
                    purl, extract_input, _emeta)
            # Tasks 448/2783: the per-page date decision and row finishing live in
            # `cytir.news.page_rows.finish_page_rows` (one function, also the news_daily stages).
            rows, page_published = finish_page_rows(
                passages, extract_input=extract_input, url=purl, cls=cls,
                consent_stamp=consent_stamp,
                now_ref=page_date_clamp_day(ctx.get("now_ref")),
                freshness=_emeta["published_htmldate"],
                sitemap_published=s.get("published"),
                page_receipt=page_receipt, document_id=document_id)
            document = None
            if page_receipt is not None and document_markdown:
                document = _news_document_record(
                    url=purl, document_id=document_id, html=html,
                    markdown=document_markdown, receipt=page_receipt,
                    publisher=common["domain"], published=page_published,
                    outlinks=page_outlinks,
                    point_ids=[row["point_id"] for row in rows])
            with ctx["write_lock"]:
                for row in rows:
                    ctx["out_f"].write(json.dumps(row, ensure_ascii=False) + "\n")
                ctx["links_f"].write(json.dumps(
                    {"url": purl, "domain": common["domain"],
                     "links": sorted(links)}, ensure_ascii=False) + "\n")
                if document is not None and ctx.get("docs_f") is not None:
                    ctx["docs_f"].write(
                        json.dumps(document, ensure_ascii=False) + "\n")
                    ctx["docs_f"].flush()
                ctx["out_f"].flush(); ctx["links_f"].flush()
            res.rows += len(rows)
            res.links += len(links)
            done_urls.add(url)
            n_since_save += 1
            if n_since_save >= SAVE_EVERY:
                raw_f.flush()
                with ctx["write_lock"]:
                    st.update(status="in-progress", done_urls=sorted(done_urls),
                              fetched=res.fetched, rows=res.rows)
                    save_state(args.state, ctx["state"])
                n_since_save = 0
                print(f"  [{qdom}] {res.fetched}/{res.selected} pages "
                      f"{res.rows}r", flush=True)
    finally:
        raw_f.close()

    res.status = "ok" if res.fetched else "empty"
    with ctx["write_lock"]:
        st.update(status=res.status, route=res.route,
                  discovered=res.discovered, selected=res.selected,
                  done_urls=sorted(done_urls), fetched=res.fetched,
                  failed=res.failed, rows=res.rows, links=res.links,
                  note=res.note, raw_file=str(raw_path))
        save_state(args.state, ctx["state"])

    # ---- ship the finished host's raw L0 to the corpus store -----------------
    if args.ship_raw and res.status == "ok" and not is_cc:
        rc = subprocess.call(["rsync", "-a", str(raw_path), args.ship_raw])
        if rc == 0:
            os.remove(raw_path)
            st["raw_shipped"] = args.ship_raw
            with ctx["write_lock"]:
                save_state(args.state, ctx["state"])
            print(f"  [{qdom}] raw shipped -> {args.ship_raw}", flush=True)
        else:
            print(f"  [{qdom}] !! raw ship failed rc={rc} (kept locally)",
                  flush=True)
    return res


# --- state --------------------------------------------------------------------
def load_state(path):
    if path and os.path.exists(os.path.expanduser(path)):
        try:
            return json.load(open(os.path.expanduser(path)))
        except Exception:  # noqa: BLE001
            pass
    return {"hosts": {}}


def save_state(path, state):
    if not path:
        return
    p = pathlib.Path(os.path.expanduser(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    json.dump(state, open(tmp, "w"))
    os.replace(tmp, str(p))


# --- main ----------------------------------------------------------------------
def _preservation_outputs(args):
    """Resolve --preserve-dir/--docs-out, refusing an incoherent invocation.

    Checked before any queue, state, socket or client work: a run that cannot
    produce the evidence it was asked for must fail at the command line, not
    after it has already fetched from a publisher.
    """
    preserve_root = (pathlib.Path(os.path.expanduser(args.preserve_dir))
                     if getattr(args, "preserve_dir", "") else None)
    docs_out = (pathlib.Path(os.path.expanduser(args.docs_out))
                if getattr(args, "docs_out", "") else None)
    if docs_out is not None and preserve_root is None:
        raise SystemExit(
            "--docs-out requires --preserve-dir: a document record whose "
            "response was not sealed is not evidence")
    return preserve_root, docs_out


def cmd_fetch(args):
    _preservation_outputs(args)
    queue = json.load(open(os.path.expanduser(args.queue)))
    items = [q for q in queue if str(q.get("wave", "")) == args.wave]
    if args.only:
        keep = {d.strip() for d in args.only.split(",")}
        items = [q for q in items if q["domain"] in keep]
    if not items:
        raise SystemExit(f"no queue entries for wave={args.wave}")

    state = load_state(args.state)
    todo = [q for q in items
            if state["hosts"].get(q["domain"], {}).get("status")
            not in ("ok", "empty")] if args.resume else items
    # big caps first so one long host doesn't own the tail
    todo.sort(key=lambda q: -int(q.get("cap") or args.max_per_domain))
    print(f"wave={args.wave} hosts={len(items)} resume-skip="
          f"{len(items) - len(todo)} to-fetch={len(todo)} | "
          f"host-rate≤{args.host_rate}/s global≤{args.rate}/s "
          f"cc-rate≤{args.cc_rate}/s workers={args.workers}", flush=True)
    if not todo:
        print("nothing to do."); return

    # granular timeouts + socket default (the 2026-07-03 ssl.py-hang lesson)
    socket.setdefaulttimeout(90)
    client = make_client(
        HttpSpec(timeout=httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=120.0),
                 headers={k.lower(): v for k, v in UA.items()}, follow_redirects=True),
        transport=httpx.HTTPTransport(http2=False, limits=httpx.Limits(
            max_connections=args.workers * 2, max_keepalive_connections=args.workers)))

    # cc fallback: pre-resolve the cc-route hosts via the columnar index; set
    # discovery_ingest's global WARC throttle to --cc-rate (shared ADR-014
    # budget — keep ≤2 if another CC job is running).
    di._RATE = args.cc_rate
    cc_items = []
    for q in todo:
        cfg = HOST_RULES.get(q["domain"], {})
        if (cfg.get("route") or q.get("route")) == "cc-columnar":
            cc_items.append(q)
    if args.no_cc_fallback:
        cc_items = []   # sitemap-only mode: never touch the columnar CC index
    cc_resolved: dict[str, list | None] = {}
    if cc_items:   # resolve in --discover-only too (index queries, no WARC GETs)
        con = ccx._con(threads=2)
        for q in cc_items:
            cfg = HOST_RULES.get(q["domain"], {})
            host = cfg.get("host", q["domain"])
            include = tuple(cfg["include"]) if "include" in cfg else \
                ((q["path_hint"],) if q.get("path_hint") else ())
            cap = min(int(q.get("cap") or args.max_per_domain),
                      args.max_per_domain)
            try:
                cc_resolved[q["domain"]] = resolve_cc_host(
                    args.crawl, host, include, cap * 6, con)
                print(f"  [columnar] {q['domain']}: "
                      f"{len(cc_resolved[q['domain']])} captures "
                      f"(scope={list(include) or 'whole-host'})", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  [columnar] {q['domain']} ERRORED ({e!r}) → 'error'",
                      file=sys.stderr, flush=True)
                cc_resolved[q["domain"]] = None
        con.close()

    out = pathlib.Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    links_out = pathlib.Path(os.path.expanduser(args.links_out)) \
        if args.links_out else out.parent / "outlinks.jsonl"
    preserve_root, docs_out = _preservation_outputs(args)
    if preserve_root is not None:
        preserve_root.mkdir(parents=True, exist_ok=True)
    if docs_out is not None:
        docs_out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume and out.exists()) else "w"
    now_ref = (datetime.strptime(args.now, "%Y-%m-%d").date()
               if getattr(args, "now", "") else datetime.utcnow().date())
    recapture = recapture_client = None
    if getattr(args, "recapture_twin_url", "") and getattr(args, "recapture_collection", ""):
        if not getattr(args, "freshness", False):
            raise SystemExit("--recapture-* applies to --freshness runs only")
        recapture = {"twin_url": args.recapture_twin_url,
                     "collection": args.recapture_collection,
                     "overhang_days": args.recapture_overhang_days}
        recapture_client = NRS.open_client(args.recapture_twin_url)
        print(f"recapture-overhang: skip already-ingested urls older than "
              f"{args.recapture_overhang_days}d (twin {args.recapture_collection})",
              flush=True)
    ctx = {
        "state": state, "cache_dir": os.path.expanduser(args.cache_dir),
        "client": client, "gbucket": TokenBucket(args.rate),
        "write_lock": threading.Lock(), "cc_resolved": cc_resolved,
        "out_f": (open(os.devnull, "w") if args.discover_only
                  else out.open(mode)),
        "links_f": (open(os.devnull, "w") if args.discover_only
                    else links_out.open(mode)),
        "direct_crawl": args.direct_crawl,
        "no_cc_fallback": args.no_cc_fallback,
        "now_ref": now_ref,
        "recapture": recapture, "recapture_client": recapture_client,
        "preserve_root": preserve_root,
        "docs_f": (None if (docs_out is None or args.discover_only)
                   else docs_out.open(mode)),
    }

    results: list[HostResult] = []
    try:
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(process_host, q, args, ctx): q for q in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                results.append(r)
                print(f"[{i}/{len(todo)}] {r.qdom}: {r.status} "
                      f"route={r.route} disc={r.discovered} sel={r.selected} "
                      f"fetched={r.fetched} fail={r.failed} rows={r.rows} "
                      f"links={r.links} {r.note}", flush=True)
    finally:
        ctx["out_f"].close(); ctx["links_f"].close()
        if ctx.get("docs_f") is not None:
            ctx["docs_f"].close()
        save_state(args.state, state)

    ok = sum(1 for r in results if r.status == "ok")
    n_rows = sum(r.rows for r in results)
    print(f"\nFETCH done. hosts ok={ok} of {len(results)} | rows={n_rows} "
          f"-> {out}", flush=True)
    errs = [r.qdom for r in results if r.status == "error"]
    if errs:
        print(f"⚠ {len(errs)} hosts ERRORED (transient, NOT empty) — re-run "
              f"retries only these: {errs}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="sitemap discover + direct polite fetch")
    f.add_argument("--queue", required=True)
    f.add_argument("--wave", default="1b")
    f.add_argument("--only", default="", help="comma list of queue domains")
    f.add_argument("--out", required=True)
    f.add_argument("--links-out", default="")
    f.add_argument("--state", default="")
    f.add_argument("--raw-dir", required=True,
                   help="L0 originals dir (direct fetches: raw HTML gz; "
                        "cc fallback: WARC pointer jsonl)")
    f.add_argument("--preserve-dir", default="",
                   help="task 450: seal every publisher response (exact bytes "
                        "+ transport headers + our outbound request identity) "
                        "under this root BEFORE reading it, and extract from "
                        "the validated artifact. Required for any lane whose "
                        "rows must satisfy a preservation profile; a fetch "
                        "that cannot be sealed then yields no row at all.")
    f.add_argument("--docs-out", default="",
                   help="task 450: per-document news_article evidence "
                        "(Markdown parent + anchor-bearing outlinks + response "
                        "lineage + the chunk point ids it produced). Requires "
                        "--preserve-dir.")
    f.add_argument("--ship-raw", default="",
                   help="rsync dest (e.g. host:/…/aw-corpus/l0_direct/"
                        "wave=wave1b_raw/) — each finished host's raw gz is "
                        "shipped then removed locally")
    f.add_argument("--host-rate", type=float, default=2.0,
                   help="per-host req/s ceiling (robots Crawl-delay lowers it)")
    f.add_argument("--rate", type=float, default=4.0,
                   help="aggregate direct-fetch req/s across all hosts")
    f.add_argument("--cc-rate", type=float, default=3.0,
                   help="aggregate WARC req/s for the cc fallback (ADR-014 "
                        "shared budget; use 2 if another CC job runs)")
    f.add_argument("--workers", type=int, default=4, help="concurrent hosts")
    f.add_argument("--max-per-domain", type=int, default=4000)
    f.add_argument("--max-child-sitemaps", type=int, default=60)
    f.add_argument("--crawl", default=di.DEFAULT_CRAWL)
    f.add_argument("--cache-dir", default=cr.CACHE_DIR)
    f.add_argument("--resume", action="store_true", default=True)
    f.add_argument("--no-resume", dest="resume", action="store_false")
    f.add_argument("--freshness", action="store_true",
                   help="source-only newest-first selection in a rolling window")
    f.add_argument("--window-months", type=int, default=12)
    f.add_argument("--recapture-twin-url", default="",
                   help="serving twin Qdrant url; with --recapture-collection, "
                        "an already-ingested url older than the overhang is not "
                        "fetched (--freshness runs only; ruling 8)")
    f.add_argument("--recapture-collection", default="")
    f.add_argument("--recapture-overhang-days", type=int,
                   default=NRS.NSC.DEFAULT_NEWS_RECAPTURE_OVERHANG_DAYS)
    f.add_argument("--now", default="",
                   help="freshness reference day in YYYY-MM-DD")
    f.add_argument("--discover-only", action="store_true",
                   help="consent+robots+sitemap discovery + selection counts "
                        "only; fetches sitemap files, never pages")
    f.add_argument("--direct-crawl", action="store_true",
                   help="provenance is OUR OWN direct origin fetch, not a CC "
                        "copy (consent_resolver direct_crawl: records mode + "
                        "cache key; since ADR-024 am.5 it changes no flag). "
                        "Use for CC-invisible hosts fetched straight from "
                        "their sitemap (task 287).")
    f.add_argument("--no-cc-fallback", action="store_true",
                   help="pure-sitemap route: if sitemap discovery yields no "
                        "urls, mark the host 'empty' rather than falling back "
                        "to the columnar CC index (correct for hosts CC does "
                        "not hold at all — task 287).")
    f.set_defaults(func=cmd_fetch)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
