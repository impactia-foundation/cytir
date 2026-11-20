#!/usr/bin/env python3
"""Task 195 — the SHARED pack-ingestion fetch layer (the L0 contract, ADR-026).

WHY: ADR-026/specs/corpus-layers.md is the standard — L0 direct fetches keep
raw HTML/XML permanently (we are the sole holder); rebuild-from-scratch is
replay L0→L1→L2→embed, never a re-fetch. But the ~52 bespoke ingest_*.py
scripts only shared the DOWNSTREAM standard (gates 097/166/167, consent,
dump schema); the fetch layer was copy-pasted per pack and raw HTML was
never retained (task 195 audit: depth175-193 all extract-in-flight and
discard). This module is the one fetch layer every future pack uses.

WHAT a pack gets by calling `run_pack(...)`:
  * robots verification per host (wildcard-capable, Google-style longest
    match, AuthoritativeWebBot UA; JS-shell / 5xx / 403-challenge robots
    ⇒ host skipped) + per-host politeness (max(cfg delay, crawl-delay))
  * **L0 retention by default**: every 200 HTML/API body is written raw
    (exact bytes, base64 in jsonl.gz shards + sha256) to the corpus store's
    L0 layout on the build-box HDD —
        <l0-root>/wave=<pack>/shard-NNNN.jsonl.gz + _meta.json
    (default l0-root: <corpus-root>/l0;
    `_meta.json` is written only when the crawl DRAINS — wave-complete
    marker per corpus-layers.md; interim runs keep `_meta.partial.json`).
  * outlink capture via the task 194 sink (aw_common.enable_outlink_capture
    — one {url, domain, links, anchors} row per parsed page to
    <out-dir>/outlinks.jsonl, schema-compatible with
    aggregate_harvested_links.py). run_pack additionally routes NON-indexed
    fetched pages (hubs, out-of-scope) through the sink, so link-rich hub
    pages feed the endorsement graph too.
  * fetch-log (<out-dir>/fetch_log.jsonl, one row per fetch attempt) +
    resumability (state.json done-set/frontier — the 175 pattern) + budget /
    per-host caps / --limit smoke runs
  * heartbeat: a flushed stdout line at least every --hb-secs (default 240s)
    so a slow-host tail (e.g. one 30s-crawl-delay host draining alone) never
    trips the task 100 supervisors' 20-min log-mtime stall guard (the
    intra-host heartbeat lesson, task 111 / discovery_ingest.py).

ADAPTER PATTERN: the bespoke-per-source part of a pack shrinks to an
inventory adapter — a HOSTS dict (same shape the 175/176/180-pattern packs
already use) + per-source metadata + optional custom mode handlers:

    HOSTS = {"eeoc": dict(host="www.eeoc.gov", mode="sitemap",
                          sitemaps=[...], include=fn, pre=fn,
                          lang=fn, delay=2.0, cap=3600,
                          source_type="official", tier="A", trust=0.9), ...}
    canon = pack_fetch.make_canon(ALIAS)
    ap = argparse.ArgumentParser(); pack_fetch.add_standard_args(ap, ...)
    pack_fetch.run_pack(pack="dei_pack", hosts=HOSTS, canon=canon,
                        args=ap.parse_args(), title_noise=_TITLE_NOISE)

Host cfg keys: host, mode ("sitemap" | "bfs" | a key in `mode_handlers`),
sitemaps/seeds, include (sitemap-inventory filter), allow (indexability),
hub + hub_depth (BFS frontier), expand (sitemap host also BFS-expands),
ok_hosts (redirect targets accepted), lang(url)->str, delay, cap,
source_type, trust, tier (else default_tier), license, pre (html pre-hook).

Extraction runs through aw_common.process_html ⇒ gates 097 (extraction
junk), 166 (language scope), 167 (listing shell) all live; the 175-pattern
apparatus (citation-list) filter + over-chunk head-cap + H1 title cleanup
are applied here, shared. Output records: asdict(Passage) minus
content_digest, + provenance (+ license when the host cfg carries one) —
byte-identical to the 175→193 pack dump schema.

API-sourced packs (ctgov/eurostat-class) don't use run_pack's crawl loop but
MUST use L0Store directly: store the raw API responses as their L0
(`L0Store(...).put(url=..., body=resp.content, kind="api")`).
CC-sourced content is NOT this module's path — it keeps WARC pointers
(discovery_ingest.py / corpus-layers.md §L0).

CLI (L0 maintenance):
  python pack_fetch.py l0-stats  --wave-dir <...>/l0/wave=<pack>
  python pack_fetch.py l0-verify --wave-dir <...>/l0/wave=<pack>  # sha256 all bodies
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import functools
import glob
import gzip
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
import urllib.parse
from dataclasses import asdict

import httpx
from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

from cytir.acquire import gz_shard_store
from cytir.core.canon import https_frontier_url
from cytir.extract.html_extract import enable_outlink_capture, extract_content_markdown, process_html
from cytir.core.paths import operator_settings
from cytir.core.user_agent import pack_fetch_user_agent

PACK_FETCH_VERSION = "pack_fetch.py v1 (task 195)"

UA_TOKEN = "authoritativewebbot"


def _ua() -> str:
    """The wire identity: built by `cytir.core.user_agent` from the operator's URL and contact
    (AW_OPERATOR_URL / AW_OPERATOR_CONTACT); refuses when nobody configured them."""
    return pack_fetch_user_agent()


def default_l0_root() -> str:
    """L0 fetch waves under the bulk-data disk (operator setting `hdd_root`)."""
    return os.path.join(operator_settings().hdd_root, "aw-corpus", "l0")


def default_l1_root() -> str:
    """Extracted Markdown waves under the bulk-data disk."""
    return os.path.join(operator_settings().hdd_root, "aw-corpus", "l1")


def __getattr__(name):
    # `crawl.UA` / `HEADERS` / `HDD_ROOT` / `DEFAULT_L0_ROOT` / `DEFAULT_L1_ROOT` stay readable
    # (callers use them) but resolve on first ACCESS, so importing this module never needs a
    # configured identity or work root; an unconfigured one refuses by name where it is used.
    if name == "UA":
        return _ua()
    if name == "HEADERS":
        return {"User-Agent": _ua()}
    if name == "HDD_ROOT":
        return operator_settings().hdd_root
    if name == "DEFAULT_L0_ROOT":
        return default_l0_root()
    if name == "DEFAULT_L1_ROOT":
        return default_l1_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

BINARY_RE = re.compile(
    r"(?i)\.(pdf|jpe?g|png|gif|svg|zip|docx?|xlsx?|pptx?|mp[34]|avi|mov|ico|"
    r"css|js|xml|rss|epub|webp|csv|ods)(\?|$)")

_WWW_RE = re.compile(r"^www\d*\.")


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- 175-pattern shared extraction hygiene -----------------------------------
_CITE_RE = re.compile(
    r'(?:\b(?:19|20)\d{2}[a-z]?[,)\].]|doi:|\b\d{1,4}[–-]\d{1,4}\b|ISBN|, pp?\.)')
_ITEM_RE = re.compile(r'^\s*(?:[-*]\s|–––)')


def is_apparatus(text: str) -> str | None:
    """Citation-list detector (175 pattern, precision-first): a chunk is
    dropped as bibliography ONLY if it starts with a list marker AND has >=3
    list items AND >=3 citation signatures. Real prose never starts with '- '."""
    t = text.strip()
    if not _ITEM_RE.match(t):
        return None
    lines = [l for l in t.split("\n") if l.strip()]
    items = sum(1 for l in lines if _ITEM_RE.match(l))
    if items >= 3 and len(_CITE_RE.findall(t)) >= 3:
        return "bibliography"
    return None


def clean_title(passages, fallback: str, noise_re: re.Pattern | None = None) -> str:
    """H1-from-first-chunk title (175 pattern); else strip the pack's
    site-suffix noise regex from the trafilatura title."""
    if passages:
        first = passages[0].text.lstrip()
        if first.startswith("# "):
            h1 = first.split("\n", 1)[0][2:].strip()
            if 2 <= len(h1) <= 200:
                return h1
    t = (fallback or "").strip()
    if noise_re is not None:
        t = noise_re.sub("", t)
    return t or fallback


# --- robots (wildcard-capable, Google-style longest-match) --------------------
class Robots:
    def __init__(self, text: str | None):
        self.rules: list[tuple[bool, re.Pattern, int]] = []
        self.delay: float | None = None
        self.broken = False           # robots endpoint served HTML (JS shell)
        if text is None:
            return
        if text.lstrip()[:1] == "<":
            self.broken = True
            return
        groups, agents, rules, delay, last_ua = [], [], [], None, False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            m = re.match(r"(?i)(user-agent|disallow|allow|crawl-delay)\s*:\s*(.*)",
                         line)
            if not m:
                continue
            key, val = m.group(1).lower(), m.group(2).strip()
            if key == "user-agent":
                if not last_ua and agents:
                    groups.append((agents, rules, delay))
                    agents, rules, delay = [], [], None
                agents.append(val.lower())
                last_ua = True
            else:
                last_ua = False
                if key == "disallow" and val:
                    rules.append((False, val))
                elif key == "allow" and val:
                    rules.append((True, val))
                elif key == "crawl-delay":
                    try:
                        delay = float(val)
                    except ValueError:
                        pass
        if agents:
            groups.append((agents, rules, delay))
        chosen = None
        for ags, rs, d in groups:      # specific UA record wins over *
            if any(a != "*" and a in UA_TOKEN for a in ags):
                chosen = (rs, d)
                break
        if chosen is None:
            for ags, rs, d in groups:
                if "*" in ags:
                    chosen = (rs, d)
                    break
        if chosen:
            self.delay = chosen[1]
            for allow, pat in chosen[0]:
                rx = re.escape(pat).replace(r"\*", ".*")
                if rx.endswith(r"\$"):
                    rx = rx[:-2] + "$"
                self.rules.append((allow, re.compile("^" + rx), len(pat)))

    def allowed(self, path_q: str) -> bool:
        best, best_len = None, -1
        for allow, rx, plen in self.rules:
            if rx.match(path_q):
                if plen > best_len or (plen == best_len and allow):
                    best, best_len = allow, plen
        return True if best is None else best


def fetch_robots(client: httpx.Client, host: str) -> Robots | None:
    """None => host must be skipped (5xx / JS shell / WAF 403 challenge on
    robots itself — a bot-targeted block is a DECLINE signal, we never spoof
    UAs). Plain 4xx => no robots => allow-all.

    4 attempts on transient failure (192 lesson: the build box egress randomly
    refuses ~15% of connects — a single-shot fetch mis-skips healthy hosts)."""
    for _attempt in range(4):
        try:
            r = client.get(f"https://{host}/robots.txt")
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(8)
                continue
            if r.status_code == 403:   # WAF challenge on robots itself
                return None
            if r.status_code >= 400:
                return Robots("")      # no robots -> allow-all
            rb = Robots(r.text)
            return None if rb.broken else rb
        except Exception:  # noqa: BLE001
            time.sleep(8)
    return None


# --- sitemap walker ------------------------------------------------------------
LOC_RE = re.compile(r"<loc>\s*([^<\s]+?)\s*</loc>", re.I)


class SitemapError(RuntimeError):
    """A sitemap could not be read: our transport failed, the origin answered non-200, or the
    body was not a readable (gzip) XML. Never the same thing as an empty sitemap: an error
    returned as `[]` looks like a clean, empty inventory (task 683 finding 3)."""


def sitemap_urls(client: httpx.Client, url: str, delay: float = 1.0,
                 depth: int = 0) -> list[str]:
    """Every `<loc>` of the sitemap at `url` (an index is followed three levels deep).

    Raises `SitemapError` when any sitemap in the walk cannot be read. A sitemap that was read
    and lists nothing returns `[]`: that is the only way `[]` comes back."""
    try:
        r = client.get(url)
        if r.status_code != 200:
            raise SitemapError(f"{url}: http {r.status_code}")
        body = r.content
        if url.endswith(".gz") or body[:2] == b"\x1f\x8b":
            body = gzip.decompress(body)
        text = body.decode("utf-8", "replace")
    except SitemapError:
        raise
    except Exception as exc:  # noqa: BLE001 -- transport or gzip failure: loud, with the url
        raise SitemapError(f"{url}: {type(exc).__name__}: {exc}") from exc
    locs = [l.replace("&amp;", "&") for l in LOC_RE.findall(text)]
    if "<sitemapindex" in text[:2000]:
        out: list[str] = []
        if depth < 3:
            for l in locs:
                time.sleep(delay)          # crawl-delay applies to sitemaps too
                out += sitemap_urls(client, l, delay, depth + 1)
        return out
    return locs


# --- URL helpers ----------------------------------------------------------------
# single-quoted (tbray.org) and unquoted hrefs too, not just double-quoted
# (192 lesson — martinfowler/tbray markup broke BFS discovery otherwise)
HREF_RE = re.compile(r'''href\s*=\s*(?:"([^"#]+)|'([^'#]+)|([^"'\s>#]+))''')


def make_canon(alias: dict[str, str] | None = None):
    """Build the pack's frontier-link function: https, host aliased+lowered,
    query dropped unless keep_query (`cytir.core.canon.https_frontier_url`)."""
    return functools.partial(https_frontier_url, alias=alias or {})


def page_links(base_url: str, html: str, host: str, canon,
               keep_query_re: re.Pattern | None = None,
               binary_re: re.Pattern = BINARY_RE) -> list[str]:
    """Same-host frontier links (BFS/expand crawling). NOT the endorsement
    outlinks — those are captured by the task 194 sink in process_html."""
    out = []
    for m in HREF_RE.findall(html):
        h = m[0] or m[1] or m[2]
        u = urllib.parse.urljoin(base_url, h.strip())
        if not u.startswith("http"):
            continue
        keep_q = bool(keep_query_re and keep_query_re.search(u))
        cu = canon(u, keep_query=keep_q)
        if not cu or urllib.parse.urlsplit(cu).netloc != host:
            continue
        if binary_re.search(cu):
            continue
        out.append(cu)
    return out


def _path(u: str) -> str:
    return urllib.parse.urlsplit(u).path or "/"


# --- L0 store (ADR-026: raw originals, we are the sole holder) -----------------
class L0Store:
    """Append raw fetched bodies (EXACT bytes) into the corpus store's L0
    layout:  <root>/wave=<pack>/shard-NNNN.jsonl.gz.

    One JSON row per fetch: {url, final_url, status, content_type, encoding,
    kind, fetched_at, sha256, bytes, tool, body_b64}. Base64 keeps the bytes
    exact (sha256-verifiable — a lossy text decode is NOT L0); gzip absorbs
    most of the inflation. Appending re-opens the newest shard as a fresh
    gzip member (valid concatenated gzip), so resumed crawls keep appending.
    Rows are sync-flushed per put — a killed run loses at most the last row.

    `_meta.json` is written by close(complete=True) ONLY (wave-complete
    marker per specs/corpus-layers.md); interim persists write
    `_meta.partial.json`.
    """

    def __init__(self, root: str, pack: str,
                 shard_max_bytes: int = 512 * 1024 * 1024):
        self.dir = os.path.join(os.path.expanduser(root), f"wave={pack}")
        os.makedirs(self.dir, exist_ok=True)
        self.pack = pack
        self.shard_max = shard_max_bytes
        self.records = 0
        self.body_bytes = 0
        prev = os.path.join(self.dir, "_meta.partial.json")
        if os.path.exists(prev):
            try:
                m = json.load(open(prev))
                self.records = int(m.get("records", 0))
                self.body_bytes = int(m.get("body_bytes", 0))
            except Exception:  # noqa: BLE001
                pass
        # Task 428: a resumed run starts a NEW shard, never re-opens the
        # previous run's. Appending into an existing shard is what put a
        # complete gzip member behind a killed run's torn one, making the
        # whole file unreadable from byte 0 (see gz_shard_store's docstring).
        self._idx = gz_shard_store.next_shard_index(self.dir)
        self._gz = None
        self._raw = None

    def _open(self):
        # O_EXCL create via gz_shard_store: never opens an existing shard, so
        # a crashed writer's torn member is always the LAST member of its own
        # file and no complete member can hide behind it (task 428).
        _path, self._raw, self._idx = gz_shard_store.create_shard(
            self.dir, self._idx)
        self._gz = gzip.GzipFile(fileobj=self._raw, mode="wb")

    def put(self, url: str, body: bytes, *, final_url: str | None = None,
            status: int = 200, content_type: str | None = None,
            encoding: str | None = None, kind: str = "html",
            extra: dict | None = None) -> str:
        """Store one raw body; returns its sha256 hex."""
        if self._gz is None:
            self._open()
        sha = hashlib.sha256(body).hexdigest()
        row = {"url": url, "final_url": final_url or url, "status": status,
               "content_type": content_type, "encoding": encoding,
               "kind": kind, "fetched_at": _now_iso(), "sha256": sha,
               "bytes": len(body), "tool": PACK_FETCH_VERSION,
               "body_b64": base64.b64encode(body).decode("ascii")}
        if extra:
            row["extra"] = extra
        self._gz.write((json.dumps(row, ensure_ascii=False) + "\n")
                       .encode("utf-8"))
        self._gz.flush()               # Z_SYNC_FLUSH: recoverable on kill
        self.records += 1
        self.body_bytes += len(body)
        if self._raw.tell() >= self.shard_max:
            self._gz.close(); self._raw.close()
            self._gz = self._raw = None
            self._idx += 1
        return sha

    def _meta(self, complete: bool) -> dict:
        files = sorted(os.path.basename(p) for p in
                       glob.glob(os.path.join(self.dir, "shard-*.jsonl.gz")))
        return {"layer": "L0", "wave": self.pack, "kind": "direct-fetch",
                "records": self.records, "body_bytes": self.body_bytes,
                "files": files,
                "bytes": sum(os.path.getsize(os.path.join(self.dir, f))
                             for f in files),
                "tool": PACK_FETCH_VERSION, "ua": _ua(),
                "complete": complete, "updated_at": _now_iso(),
                "written_by": "aw/acquire/crawl.py"}

    def persist(self):
        """Interim marker (crash-visible progress). NOT the complete marker."""
        if self._gz is not None:
            self._gz.flush()
        tmp = os.path.join(self.dir, "_meta.partial.json.tmp")
        json.dump(self._meta(False), open(tmp, "w"), indent=2)
        os.replace(tmp, os.path.join(self.dir, "_meta.partial.json"))

    def close(self, complete: bool):
        if self._gz is not None:
            self._gz.close(); self._raw.close()
            self._gz = self._raw = None
        if complete:
            meta = self._meta(True)
            meta["sha256"] = {
                f: _file_sha256(os.path.join(self.dir, f))
                for f in meta["files"]}
            tmp = os.path.join(self.dir, "_meta.json.tmp")
            json.dump(meta, open(tmp, "w"), indent=2)
            os.replace(tmp, os.path.join(self.dir, "_meta.json"))
            part = os.path.join(self.dir, "_meta.partial.json")
            if os.path.exists(part):
                os.remove(part)
        else:
            self.persist()


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_l0(wave_dir: str):
    """Yield L0 rows (body as bytes under 'body').

    Reads through gz_shard_store's member walker (task 428): tolerates a
    truncated member ANYWHERE in a shard, not just at the end, and resyncs
    onto the members behind it. The old reader stopped at the first tear and
    abandoned the rest of the file, which is how 22,862 rows in three waves
    looked lost for weeks while complete gzip members sat behind the tear.
    """
    for path in sorted(glob.glob(os.path.join(wave_dir, "shard-*.jsonl.gz"))):
        for row in gz_shard_store.iter_jsonl_rows(path):
            if "body_b64" not in row:
                continue
            try:
                row["body"] = base64.b64decode(row.pop("body_b64"))
            except (ValueError, TypeError):
                continue               # row torn mid-base64
            yield row


class MarkdownStore:
    """Append-only L1 store whose emitted Markdown is the exact chunk input.

    Every indexable L0 parent gets one transformation row, including explicit
    quarantines.  Emitted rows retain the full Markdown plus both parent and
    child digests; chunk policy rejections remain visible instead of silently
    disappearing.  Complete waves carry per-shard hashes just like L0.
    """

    def __init__(self, root: str, pack: str,
                 shard_max_bytes: int = 512 * 1024 * 1024):
        self.dir = os.path.join(os.path.expanduser(root), f"wave={pack}")
        os.makedirs(self.dir, exist_ok=True)
        self.pack = pack
        self.shard_max = shard_max_bytes
        self.records = self.emitted = self.quarantined = 0
        self.chunk_parents_emitted = self.chunk_parents_quarantined = 0
        prev = os.path.join(self.dir, "_meta.partial.json")
        if os.path.exists(prev):
            try:
                meta = json.load(open(prev))
                for key in ("records", "emitted", "quarantined",
                            "chunk_parents_emitted", "chunk_parents_quarantined"):
                    setattr(self, key, int(meta.get(key, 0)))
            except Exception:  # noqa: BLE001
                pass
        # Task 428: a resumed run starts a NEW shard, never re-opens the
        # previous run's. Appending into an existing shard is what put a
        # complete gzip member behind a killed run's torn one, making the
        # whole file unreadable from byte 0 (see gz_shard_store's docstring).
        self._idx = gz_shard_store.next_shard_index(self.dir)
        self._gz = None
        self._raw = None

    def _open(self):
        # O_EXCL create via gz_shard_store: never opens an existing shard, so
        # a crashed writer's torn member is always the LAST member of its own
        # file and no complete member can hide behind it (task 428).
        _path, self._raw, self._idx = gz_shard_store.create_shard(
            self.dir, self._idx)
        self._gz = gzip.GzipFile(fileobj=self._raw, mode="wb")

    def put(self, *, url: str, original_sha256: str | None,
            markdown: str | None, extraction_mode: str | None,
            rejection: str | None, chunk_count: int,
            chunk_rejections: dict[str, int], source_profile: str) -> str | None:
        if self._gz is None:
            self._open()
        markdown_bytes = markdown.encode("utf-8") if markdown is not None else b""
        markdown_sha = hashlib.sha256(markdown_bytes).hexdigest() \
            if markdown is not None else None
        status = "emitted" if markdown is not None else "quarantined"
        chunk_status = "emitted" if chunk_count else "quarantined"
        row = {
            "url": url,
            "source_profile": source_profile,
            "status": status,
            "parent_original_sha256": original_sha256,
            "markdown_sha256": markdown_sha,
            "markdown_bytes": len(markdown_bytes),
            "markdown": markdown,
            "extractor": {
                "id": "aw_common.extract_content_markdown",
                "version": _package_version("trafilatura"),
                "mode": extraction_mode,
            },
            "rejection": rejection,
            "chunk_status": chunk_status if markdown is not None else "not_run",
            "chunk_count": chunk_count,
            "chunk_rejections": chunk_rejections,
        }
        self._gz.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        self._gz.flush()
        self.records += 1
        if markdown is None:
            self.quarantined += 1
        else:
            self.emitted += 1
            if chunk_count:
                self.chunk_parents_emitted += 1
            else:
                self.chunk_parents_quarantined += 1
        if self._raw.tell() >= self.shard_max:
            self._gz.close(); self._raw.close()
            self._gz = self._raw = None
            self._idx += 1
        return markdown_sha

    def _meta(self, complete: bool) -> dict:
        files = sorted(os.path.basename(path) for path in
                       glob.glob(os.path.join(self.dir, "shard-*.jsonl.gz")))
        return {
            "layer": "L1", "wave": self.pack,
            "kind": "extracted-markdown", "records": self.records,
            "emitted": self.emitted, "quarantined": self.quarantined,
            "chunk_parents_emitted": self.chunk_parents_emitted,
            "chunk_parents_quarantined": self.chunk_parents_quarantined,
            "files": files,
            "bytes": sum(os.path.getsize(os.path.join(self.dir, name))
                         for name in files),
            "tool": PACK_FETCH_VERSION, "complete": complete,
            "updated_at": _now_iso(), "written_by": "aw/acquire/crawl.py",
        }

    def persist(self):
        if self._gz is not None:
            self._gz.flush()
        tmp = os.path.join(self.dir, "_meta.partial.json.tmp")
        with open(tmp, "w", encoding="utf-8") as dst:
            json.dump(self._meta(False), dst, indent=2)
        os.replace(tmp, os.path.join(self.dir, "_meta.partial.json"))

    def close(self, complete: bool):
        if self._gz is not None:
            self._gz.close(); self._raw.close()
            self._gz = self._raw = None
        if complete:
            meta = self._meta(True)
            meta["sha256"] = {
                name: _file_sha256(os.path.join(self.dir, name))
                for name in meta["files"]}
            tmp = os.path.join(self.dir, "_meta.json.tmp")
            with open(tmp, "w", encoding="utf-8") as dst:
                json.dump(meta, dst, indent=2)
            os.replace(tmp, os.path.join(self.dir, "_meta.json"))
            partial = os.path.join(self.dir, "_meta.partial.json")
            if os.path.exists(partial):
                os.remove(partial)
        else:
            self.persist()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def iter_l1(wave_dir: str):
    """Yield every readable retained Markdown/quarantine transformation row.

    Same member-walking reader as iter_l0 (task 428) — a torn member no
    longer hides the members behind it."""
    yield from gz_shard_store.iter_store_rows(wave_dir)


def resolve_l0_root(explicit: str | None, out_dir: str) -> str:
    """HDD placement rule: on the build box, L0 goes under aw-corpus/l0 on the
    data HDD, never the boot drive. Elsewhere (the orchestrator box smoke runs / tests),
    fall back to <out-dir>/l0 with a warning."""
    if explicit:
        return explicit
    default = default_l0_root()
    if os.path.isdir(os.path.dirname(default)):
        return default
    fallback = os.path.join(out_dir, "l0")
    print(f"[l0] {default} not present on this box -> L0 falls back "
          f"to {fallback} (smoke/dev only; production packs run on the operator box)",
          flush=True)
    return fallback


def resolve_l1_root(explicit: str | None, out_dir: str) -> str:
    if explicit:
        return explicit
    default = default_l1_root()
    if os.path.isdir(os.path.dirname(default)):
        return default
    fallback = os.path.join(out_dir, "l1")
    print(f"[l1] {default} not present on this box -> L1 falls back "
          f"to {fallback} (smoke/dev only; production packs run on the operator box)",
          flush=True)
    return fallback


# --- state (175 pattern) ---------------------------------------------------------
def load_state(path: str) -> dict:
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:  # noqa: BLE001
            pass
    return {"done": [], "queued": {}, "started_at": _now_iso(),
            "hosts": {}, "passages": 0, "apparatus": {}, "capped": 0,
            "rejects": {}, "fetches": 0, "l0_records": 0}


def save_state(path: str, state: dict):
    tmp = path + ".tmp"
    json.dump(state, open(tmp, "w"), ensure_ascii=False)
    os.replace(tmp, path)


def write_chunk_receipt(path: str, *, pack: str, records: int,
                        markdown_store: MarkdownStore, complete: bool) -> str:
    """Write the atomic L2 terminal/partial receipt for passages.jsonl."""
    data_path = os.path.join(path, "passages.jsonl")
    meta = {
        "layer": "L2", "wave": pack, "kind": "chunks",
        "records": records,
        "input_markdown_records": markdown_store.emitted,
        "emitted_parent_records": markdown_store.chunk_parents_emitted,
        "quarantined_parent_records": markdown_store.chunk_parents_quarantined,
        "file": "passages.jsonl",
        "bytes": os.path.getsize(data_path) if os.path.exists(data_path) else 0,
        "tool": PACK_FETCH_VERSION, "complete": complete,
        "updated_at": _now_iso(), "written_by": "aw/acquire/crawl.py",
    }
    if complete:
        meta["sha256"] = _file_sha256(data_path) if os.path.exists(data_path) \
            else hashlib.sha256(b"").hexdigest()
    name = "_chunks_meta.json" if complete else "_chunks_meta.partial.json"
    target = os.path.join(path, name)
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as dst:
        json.dump(meta, dst, indent=2)
    os.replace(tmp, target)
    if complete:
        partial = os.path.join(path, "_chunks_meta.partial.json")
        if os.path.exists(partial):
            os.remove(partial)
    return target


def make_hstat(notscope_key: str = "not_scope"):
    def hstat(state: dict, key: str) -> dict:
        return state["hosts"].setdefault(key, {
            "pages_ok": 0, "empty": 0, "error": 0, "robots_skipped": 0,
            notscope_key: 0, "passages": 0, "fetched": 0})
    return hstat


def add_standard_args(ap: argparse.ArgumentParser, *, out_dir: str,
                      budget: int = 45000):
    """Uniform pack CLI: every adapter gets the same knobs."""
    ap.add_argument("--out-dir", default=out_dir)
    ap.add_argument("--max-chunks", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N NEW fetches this run (smoke test)")
    ap.add_argument("--only", default="",
                    help="comma-separated host keys (smoke test)")
    ap.add_argument("--budget", type=int, default=budget,
                    help="hard total-fetch stop (safety)")
    ap.add_argument("--l0-root", default="",
                    help=f"L0 wave root (default <hdd_root>/aw-corpus/l0 when its parent "
                         f"exists, else <out-dir>/l0)")
    ap.add_argument("--no-l0", action="store_true",
                    help="disable raw-HTML retention (tests/smoke ONLY — "
                         "production packs keep L0, ADR-026)")
    ap.add_argument("--l1-root", default="",
                    help=f"L1 Markdown wave root (default <hdd_root>/aw-corpus/l1 when its "
                         f"parent exists, else <out-dir>/l1)")
    ap.add_argument("--hb-secs", type=float, default=240.0,
                    help="max stdout silence before a flushed [hb] line "
                         "(supervisor stall-guard reset; task 111)")
    return ap


# --- the crawl loop ---------------------------------------------------------------
KIND_PRIO_DEFAULT = {"content": 0, "cand": 1, "list": 2, "hub": 3}


class PackCtx:
    """Handle passed to custom mode handlers (e.g. mh's NICE list walker):
    exposes enq / index_page / page_links / per-host stats."""

    def __init__(self):
        self.enq = None          # enq(key, url, kind, depth=0)
        self.index_page = None   # index_page(key, cfg, url, html) -> int
        self.hstat = None        # hstat(key) -> dict
        self.state = None
        self.canon = None
        self.client = None


def run_pack(*, pack: str, hosts: dict[str, dict], args, canon,
             alias: dict[str, str] | None = None,
             title_noise: re.Pattern | None = None,
             default_tier: str = "B",
             notscope_key: str = "not_scope",
             kind_prio: dict[str, int] | None = None,
             mode_handlers: dict | None = None,
             client: httpx.Client | None = None,
             apparatus_fn=is_apparatus) -> dict:
    """The shared multi-host polite crawl (extracted from the 175→193 pack
    ingesters, unchanged semantics) + the L0/outlink/fetch-log layer.

    `args` comes from add_standard_args. `alias` is the pack's host-alias
    dict (redirect-target folding, e.g. "eeoc.gov" -> "www.eeoc.gov").
    `mode_handlers` maps a custom queue KIND (e.g. mh's NICE "list"/"cand")
    to  handler(ctx, key, cfg, url, kind, depth, html)  called instead of
    the default index+BFS step. Returns the final state.
    """
    alias = alias or {}
    kind_prio = kind_prio or KIND_PRIO_DEFAULT
    mode_handlers = mode_handlers or {}

    D = args.out_dir
    os.makedirs(os.path.join(D, "inv"), exist_ok=True)
    passages_path = os.path.join(D, "passages.jsonl")
    state_path = os.path.join(D, "state.json")
    fetchlog_path = os.path.join(D, "fetch_log.jsonl")

    # task 194: harvest every parsed page's external links for the
    # endorsement graph (streamed to <out-dir>/outlinks.jsonl).
    enable_outlink_capture(D)

    l0: L0Store | None = None
    if not getattr(args, "no_l0", False):
        l0 = L0Store(resolve_l0_root(getattr(args, "l0_root", ""), D), pack)
        print(f"[l0] retaining raw bodies -> {l0.dir} "
              f"({l0.records} records carried over)", flush=True)
    else:
        print("[l0] DISABLED (--no-l0) — smoke/tests only, never production "
              "(ADR-026)", flush=True)
    l1 = MarkdownStore(
        resolve_l1_root(getattr(args, "l1_root", ""), D), pack)
    print(f"[l1] retaining extracted Markdown -> {l1.dir} "
          f"({l1.records} transformations carried over)", flush=True)

    own_client = client is None
    if own_client:
        client = make_client(HttpSpec(timeout=45, user_agent_kind="pack_fetch",
                                      headers={"user-agent": _ua()}, follow_redirects=True))

    last_seen = dt.date.today().isoformat()
    active_req = {k: v for k, v in hosts.items()
                  if not args.only or k in args.only.split(",")}

    state = load_state(state_path)
    done: set[str] = set(state["done"])
    hstat_ = make_hstat(notscope_key)

    # --- robots per host (live, before any content fetch) --------------------
    robots: dict[str, Robots] = {}
    active: dict[str, dict] = {}
    for key, cfg in active_req.items():
        rb = fetch_robots(client, cfg["host"])
        if rb is None:
            print(f"[{key}] robots UNAVAILABLE/JS-shell/403 -> HOST SKIPPED",
                  flush=True)
            hstat_(state, key)["robots_skipped"] += 1
            continue
        eff = max(cfg["delay"], rb.delay or 0.0)
        robots[key] = rb
        cfg = dict(cfg, eff_delay=eff)
        active[key] = cfg
        print(f"[{key}] robots ok  crawl-delay={rb.delay}  eff_delay={eff}s",
              flush=True)
        time.sleep(0.5)

    # --- queues: rebuilt from inventories/seeds + persisted BFS finds ---------
    queues: dict[str, list[list]] = {}
    qseen: dict[str, set[str]] = {}

    def enq(key: str, url: str, kind: str, depth: int = 0):
        if not url or url in done or url in qseen[key]:
            return
        if BINARY_RE.search(url):
            return
        qseen[key].add(url)
        queues[key].append([url, kind, depth])

    for key, cfg in active.items():
        queues[key], qseen[key] = [], set()
        if cfg["mode"] == "sitemap":
            inv_path = os.path.join(D, "inv", f"{key}.txt")
            if os.path.exists(inv_path):
                urls = [l.strip() for l in open(inv_path) if l.strip()]
            else:
                raw = []
                for sm in cfg["sitemaps"]:
                    raw += sitemap_urls(client, sm, delay=cfg["eff_delay"])
                urls = sorted({canon(u) for u in raw
                               if u and cfg["include"](u)} - {""})
                # keep only our host (sitemaps can cross-reference)
                urls = [u for u in urls
                        if urllib.parse.urlsplit(u).netloc == cfg["host"]]
                with open(inv_path, "w") as f:
                    f.write("\n".join(urls))
                print(f"[{key}] inventory built: {len(urls)} urls "
                      f"(from {len(raw)} sitemap locs)", flush=True)
            for u in urls[:cfg["cap"]]:
                enq(key, u, "content")
            for s in cfg.get("seeds", []):
                enq(key, canon(s), "hub")
        elif cfg["mode"] == "bfs":
            for s in cfg["seeds"]:
                enq(key, canon(s), "hub")
        elif cfg["mode"] in mode_handlers:
            for s in cfg.get("seeds", []):
                enq(key, canon(s, keep_query=True), cfg.get("seed_kind", "list"))
        # persisted frontier from a previous run
        for entry in state["queued"].get(key, []):
            enq(key, entry[0], entry[1], entry[2] if len(entry) > 2 else 0)
        # a seed may sit in `done` from a smoke run whose frontier was not
        # persisted — if the queue came up empty, re-crawl from the seeds so
        # BFS cannot silently stall
        if not queues[key] and cfg["mode"] != "sitemap":
            for s in cfg.get("seeds", []):
                u = canon(s, keep_query=(cfg["mode"] in mode_handlers))
                done.discard(u)
                enq(key, u,
                    cfg.get("seed_kind", "list") if cfg["mode"] in mode_handlers
                    else "hub")
            print(f"[{key}] empty queue -> re-seeded ({len(queues[key])})",
                  flush=True)

    pf = open(passages_path, "a", encoding="utf-8")
    flog = open(fetchlog_path, "a", encoding="utf-8")
    next_ok = {k: 0.0 for k in active}
    new_fetches = 0
    t0 = time.time()
    last_out = [time.time()]
    hb_secs = getattr(args, "hb_secs", 240.0)

    def say(msg: str):
        print(msg, flush=True)
        last_out[0] = time.time()

    def heartbeat():
        # a slow-host tail (30s crawl-delay draining alone) is otherwise
        # silent for 50×30s > the supervisors' 20-min stall guard
        if hb_secs and time.time() - last_out[0] > hb_secs:
            qs = {k: len(q) for k, q in queues.items() if q}
            say(f"  [hb] fetches={state['fetches']} "
                f"passages={state['passages']} queues={qs}")

    def persist():
        state["done"] = list(done)
        # merge, don't clobber: an --only run must not wipe the persisted
        # frontier of hosts it did not activate
        merged = dict(state.get("queued", {}))
        for k in active:
            merged[k] = queues[k]
        state["queued"] = {k: q for k, q in merged.items() if q}
        if l0 is not None:
            state["l0_records"] = l0.records
            l0.persist()
        state["l1_records"] = l1.records
        l1.persist()
        save_state(state_path, state)

    def flog_row(**kw):
        kw["t"] = _now_iso()
        flog.write(json.dumps(kw, ensure_ascii=False) + "\n")
        flog.flush()

    current_original_sha = [None]

    def index_page(key: str, cfg: dict, url: str, html: str,
                   original_sha256: str | None = None) -> int:
        st = hstat_(state, key)
        original_sha256 = original_sha256 or current_original_sha[0]
        dom = _WWW_RE.sub("", urllib.parse.urlsplit(url).netloc.lower()) or \
            _WWW_RE.sub("", cfg["host"])
        meta = {"content": True, "links": False, "want_stats": True,
                "lang_default": cfg["lang"](url),
                "passage_common": {
                    "domain": dom,
                    "tier": cfg.get("tier", default_tier),
                    "source_type": cfg["source_type"],
                    "trust_tier": cfg["trust"], "last_seen": last_seen}}
        pre = cfg.get("pre")
        if pre:
            html = pre(html)
        markdown, extract_mode = extract_content_markdown(html, url)
        if markdown is None:
            l1.put(url=url, original_sha256=original_sha256, markdown=None,
                   extraction_mode=None, rejection="extract_empty_or_low_quality",
                   chunk_count=0, chunk_rejections={},
                   source_profile=cfg.get("preservation_profile", "web_html"))
            st["empty"] += 1
            return 0
        meta["content_markdown"] = markdown
        meta["content_markdown_mode"] = extract_mode
        passages, _links, stats = process_html(url, html, meta)
        for r, n in stats.get("extraction_rejects", {}).items():
            state["rejects"][r] = state["rejects"].get(r, 0) + n
        kept = []
        for p in passages:
            reason = apparatus_fn(p.text) if apparatus_fn else None
            if reason:
                state["apparatus"][reason] = \
                    state["apparatus"].get(reason, 0) + 1
            else:
                kept.append(p)
        if len(kept) > args.max_chunks:
            state["capped"] += len(kept) - args.max_chunks
            kept = kept[:args.max_chunks]
        markdown_sha = l1.put(
            url=url, original_sha256=original_sha256, markdown=markdown,
            extraction_mode=extract_mode, rejection=None, chunk_count=len(kept),
            chunk_rejections=stats.get("extraction_rejects", {}),
            source_profile=cfg.get("preservation_profile", "web_html"))
        if not kept:
            st["empty"] += 1
            return 0
        title = clean_title(kept, kept[0].title, title_noise)
        for ordinal, p in enumerate(kept):
            p.title = title
            rec = asdict(p)
            rec.pop("content_digest", None)
            rec["provenance"] = pack
            rec["parent_original_sha256"] = original_sha256
            rec["parent_markdown_sha256"] = markdown_sha
            rec["chunk_ordinal"] = ordinal
            rec["chunk_count"] = len(kept)
            rec["source_profile"] = cfg.get("preservation_profile", "web_html")
            if cfg.get("license"):
                rec["license"] = cfg["license"]
            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pf.flush()
        st["pages_ok"] += 1
        st["passages"] += len(kept)
        state["passages"] += len(kept)
        return len(kept)

    ctx = PackCtx()
    ctx.enq, ctx.index_page, ctx.state = enq, index_page, state
    ctx.hstat = lambda key: hstat_(state, key)
    ctx.canon, ctx.client = canon, client

    while any(queues.values()):
        if args.limit and new_fetches >= args.limit:
            say(f"--limit {args.limit} reached, stopping (resumable)")
            break
        if state["fetches"] >= args.budget:
            say("FETCH BUDGET reached — stopping")
            break
        now = time.time()
        ready = [k for k, q in queues.items() if q and next_ok[k] <= now]
        if not ready:
            heartbeat()
            nxt = min(next_ok[k] for k, q in queues.items() if q)
            time.sleep(max(0.05, min(nxt - now, 5.0)))
            continue
        # oldest-eligible first keeps all hosts moving
        key = min(ready, key=lambda k: next_ok[k])
        cfg = active[key]
        st = hstat_(state, key)
        if st["fetched"] >= cfg["cap"]:
            say(f"[{key}] cap {cfg['cap']} reached — draining queue "
                f"({len(queues[key])} left)")
            queues[key] = []
            continue
        # content-first pop (hubs/lists only when no content is pending) so
        # hub exploration can never crowd out real pages within the cap
        qi = min(range(len(queues[key])),
                 key=lambda i: (kind_prio.get(queues[key][i][1], 0), i))
        url, kind, depth = queues[key].pop(qi)
        if url in done:
            continue
        pq = urllib.parse.urlsplit(url)
        if not robots[key].allowed(
                pq.path + (("?" + pq.query) if pq.query else "")):
            st["robots_skipped"] += 1
            done.add(url)
            continue
        next_ok[key] = time.time() + cfg["eff_delay"]
        new_fetches += 1
        state["fetches"] += 1
        try:
            r = client.get(url)
            final_host = urllib.parse.urlsplit(str(r.url)).netloc.lower()
            ctype = r.headers.get("content-type", "").lower()
            ok_hosts = cfg.get("ok_hosts", {cfg["host"]})
            l0_sha = None
            if r.status_code == 200 and "html" in ctype and l0 is not None:
                # L0 BEFORE extraction: retention survives extraction crashes
                l0_sha = l0.put(url, r.content, final_url=str(r.url),
                                status=r.status_code, content_type=ctype,
                                encoding=r.encoding, kind="html")
            current_original_sha[0] = l0_sha
            flog_row(key=key, url=url, status=r.status_code,
                     final_url=(str(r.url) if str(r.url) != url else None),
                     bytes=len(r.content), l0=l0_sha)
            if r.status_code != 200 or "html" not in ctype or \
                    alias.get(final_host, final_host) not in ok_hosts:
                st["error"] += 1
            else:
                html = r.text
                handler = mode_handlers.get(kind)
                if handler:
                    handler(ctx, key, cfg, url, kind, depth, html)
                else:                                   # content / hub / bfs
                    allow = cfg.get("allow")
                    indexable = allow is None or allow(url)
                    if indexable:
                        index_page(key, cfg, str(r.url).split("#")[0], html)
                    else:
                        st[notscope_key] += 1
                        # task 194: non-indexed pages (hubs, out-of-scope)
                        # still feed the endorsement graph via the sink
                        process_html(url, html, {"content": False,
                                                 "links": False})
                    if cfg["mode"] == "bfs" or cfg.get("expand"):
                        hub = cfg.get("hub")
                        for l in page_links(url, html, cfg["host"], canon):
                            if allow(l):
                                enq(key, l, "content", depth + 1)
                            elif hub and hub(l) and \
                                    depth + 1 <= cfg.get("hub_depth", 0):
                                enq(key, l, "hub", depth + 1)
            st["fetched"] += 1
        except Exception as e:  # noqa: BLE001
            st["error"] += 1
            flog_row(key=key, url=url, err=f"{type(e).__name__}: {e}")
            print(f"  ! {url}: {e}", file=sys.stderr, flush=True)
        done.add(url)
        heartbeat()
        if new_fetches % 50 == 0:
            persist()
            rate = new_fetches / max(time.time() - t0, 1)
            qs = {k: len(q) for k, q in queues.items() if q}
            say(f"  {new_fetches} fetched this run  passages={state['passages']}"
                f"  {rate*3600:.0f}/h  queues={qs}")

    drained = not any(queues.values())
    persist()
    pf.close()
    flog.close()
    chunk_receipt = write_chunk_receipt(
        D, pack=pack, records=state["passages"], markdown_store=l1,
        complete=drained)
    if l0 is not None:
        l0.close(complete=drained)
        print(f"[l0] {'COMPLETE' if drained else 'partial'} — "
              f"{l0.records} records, {l0.body_bytes/1e9:.2f} GB raw -> "
              f"{l0.dir}", flush=True)
    l1.close(complete=drained)
    print(f"[l1] {'COMPLETE' if drained else 'partial'} — "
          f"{l1.emitted} Markdown documents, {l1.quarantined} extraction "
          f"quarantines -> {l1.dir}", flush=True)
    print(f"[l2] {'COMPLETE' if drained else 'partial'} — "
          f"{state['passages']} chunks -> {chunk_receipt}", flush=True)
    if own_client:
        client.close()
    print("CRAWL DONE" if drained else "CRAWL PAUSED", flush=True)
    for key in sorted(state["hosts"]):
        print(f"  {key}: {json.dumps(state['hosts'][key])}", flush=True)
    print(f"  total passages={state['passages']} rejects={state['rejects']} "
          f"apparatus={state['apparatus']} capped={state['capped']}",
          flush=True)
    return state


# --- CLI: L0 maintenance -----------------------------------------------------------
def _cmd_l0_stats(a):
    n = b = 0
    kinds: dict[str, int] = {}
    for row in iter_l0(a.wave_dir):
        n += 1
        b += row["bytes"]
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    meta_p = os.path.join(a.wave_dir, "_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else \
        {"complete": False}
    print(f"{a.wave_dir}: {n} records, {b/1e9:.2f} GB raw, kinds={kinds}, "
          f"complete={meta.get('complete')}")


def _cmd_l0_verify(a):
    n = bad = 0
    for row in iter_l0(a.wave_dir):
        n += 1
        if hashlib.sha256(row["body"]).hexdigest() != row["sha256"]:
            bad += 1
            print(f"  SHA MISMATCH: {row['url']}")
    print(f"{a.wave_dir}: {n} records verified, {bad} mismatches")
    if bad:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("l0-stats", help="record/byte counts of an L0 wave")
    s.add_argument("--wave-dir", required=True)
    s.set_defaults(func=_cmd_l0_stats)
    v = sub.add_parser("l0-verify", help="sha256-verify every stored body")
    v.add_argument("--wave-dir", required=True)
    v.set_defaults(func=_cmd_l0_verify)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
