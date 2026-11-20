#!/usr/bin/env python3
# Moved from `discovery_ingest.py` (task 683 S7 tranche 4); the old path is an alias shim.
"""Discovery ENGINE — ingest endorsed-but-not-indexed domains (ADR-020 step 5/§Process).

The repeatable loop's *ingest* stage. Input is a prioritized `discovery_queue.json`
(produced by prep_discovery_queue.py from the endorsement run); for each domain
this:

  1. CDX-looks-up its authoritative pages on Common Crawl  (policy (a): dumps/CC,
     NEVER live-scrape) — using the HARDENED CDX discipline from extract_outlinks:
     transient CDX errors (000/429/5xx) are retried with backoff and surfaced as
     `error`, NEVER silently treated as `empty` (memory cc-cdx-outage-silent-swallow).
  2. Range-GETs the WARC bodies, throttled to an AGGREGATE ≤ RATE req/s across all
     workers (CC bulk-fetch IP-bans otherwise, ADR-014).
  3. Parses each page ONCE via aw_common.process_html → content Passage rows
     AND the page's outbound endorsement links (the compounding graph for the
     next endorsement wave — spec discovery-engine.md).
  4. Writes passages.jsonl + a sidecar outlinks jsonl (--links-out; one
     {url, domain, links} row per fetched page). Resumable: per-domain
     done-marker state file, so a re-run skips finished domains and the WARC
     fetch is crash-safe.

Then the EMBED stage (config B) is delegated to cytir.embed.bakeoff with the LOCKED
aw_v3 config:  arctic-l-v2 · MRL-128 · int8 · bm25 leg · no snippet · on_disk →
staging collection `aw_v3_p2_discovery` (the live index is never touched; merged
into aw_v3 at the migration's P2). See notes/discovery-engine.md + aw-v3-migration.md.

USAGE
  # FETCH (CC) — politeness-capped, resumable:
  python discovery_ingest.py fetch \
      --queue ~/aw-data/discovery_queue.json \
      --out   ~/aw-data/discovery/passages.jsonl \
      --state ~/aw-data/discovery/state.json \
      --max-per-domain 60 --rate 4 --workers 2 --crawl CC-MAIN-2026-25
      # add  --limit-domains N  to cap, or  --only a.ch,b.ch  for a dry-run subset

  # EMBED (config B) — on a GPU box, after FETCH. --intent is required (task
  # 634): config B is NOT a promotable artifact -- measurement-only staging
  # collections only. A pack destined for the admission ledger goes through
  # the 1024-d fp16 chain instead (task 413a embed_shards.py).
  python discovery_ingest.py embed \
      --in ~/aw-data/discovery/passages.jsonl --intent measurement-only \
      --collection aw_v3_p2_discovery --qdrant-url http://localhost:6333

GATING: the full FETCH is gated on the HN CC fetch finishing (shared ≤5 req/s CC
budget). Dry-run on 2-3 domains only to validate, then STOP until triggered.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import os
from cytir.core import envnames as _envnames
import pathlib
import socket
import subprocess
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

from cytir.extract.html_extract import outlinks_column_enabled, process_html, reg_domain
from cytir.acquire import fetch_process as fp  # noqa: F401,E402  (so process_html's lazy trafilatura import resolves)
from cytir.acquire import cc_columnar as ccx  # noqa: E402  — domain resolver (columnar index, no CDX)
from cytir.acquire import l0_store  # noqa: E402  (task 312: L0 raw-HTML archive + outlinks column)

CDX = "https://index.commoncrawl.org/{crawl}-index"
WARC = "https://data.commoncrawl.org/"
UA = {"User-Agent": "authoritative-web/0.1 (+https://websear.ch; grounding index for AI agents)"}
DEFAULT_CRAWL = "CC-MAIN-2026-25"

# Discovery is endorsement-driven, not a curated allowlist → tier C ("docs"
# /endorsed). trust_tier scaled by the candidate's class authority at stage time.
CLASS_TRUST = {"gov": 0.85, "health": 0.82, "edu": 0.8, "ref": 0.78,
               "tech": 0.75, "news": 0.6, "other": 0.55, "commercial": 0.5}
# task-263 fix (2026-07-19): `health` -> `reference`, NOT `official`. The `health`
# classifier bucket is broad-substring / endorsement-graph driven (HEALTH_KW +
# health-seed inheritance) and pulls in global medical publishers, research, and
# classifier misfires (journals.plos.org, rust-lang.github.io, cmake.org, …).
# `official` is is_location_bound()==True, so routing health there made globally-
# authoritative content country-stampable and demotable on geo-scoped queries
# (the ADR-054 error class). `reference` is non-location-bound -> INTL, which is
# correct for cross-jurisdiction authority. Genuine gov health bodies still land
# in `official` via disc_class=gov (e.g. bag.admin.ch). See task 263
CLASS_SOURCE = {"gov": "official", "health": "reference", "edu": "reference",
                "ref": "reference", "tech": "docs", "news": "reference",
                "other": "docs", "commercial": "docs"}


# --- earned vs inherited trust (task 440) -----------------------------------
# CLASS_TRUST above prices a class reached by a DOMAIN SELF-SIGNAL: a verifiable
# credential in the hostname (.gov/.edu TLD, admin.ch, a university keyword). A
# class reached through the endorsement graph is hearsay, not a credential, and
# until task 440 it was priced identically. The rule, its policy basis and the
# measurements behind it live in trust_model.py.
from cytir.acquire.trust_model import ENDORSED, SELF_SIGNAL, resolve_trust  # noqa: E402,F401  # re-exported: read as di.ENDORSED/di.SELF_SIGNAL by the discovery_ingest characterization test (test_aw_news_t4_discovery_characterization.py:35)


def class_trust_for(cls: str, class_basis: str | None,
                    n_endorser_regs: int | None) -> float:
    """CLASS_TRUST lookup + the earned/inherited rule."""
    return resolve_trust(CLASS_TRUST.get(cls, 0.55), class_basis, n_endorser_regs)


class CdxError(RuntimeError):
    """Transient CDX failure (outage / 429 / 5xx). DISTINCT from a genuine empty
    result so an outage is never silently counted as 'this domain has no pages'.
    (memory cc-cdx-outage-silent-swallow — 2026-06-28 cost 117/120 seeds.)"""


# --- aggregate request pacing (≤ RATE req/s across ALL workers) -------------
_RATE = 4.0
_RL_LOCK = threading.Lock()
_RL_NEXT = [0.0]


def _throttle():
    if _RATE <= 0:
        return
    interval = 1.0 / _RATE
    with _RL_LOCK:
        now = time.monotonic()
        slot = max(now, _RL_NEXT[0])
        _RL_NEXT[0] = slot + interval
    wait = slot - time.monotonic()
    if wait > 0:
        time.sleep(wait)


# --- extraction sandbox -------------------------------------------------
# trafilatura/htmldate regex on pathological pages can spin a CPU thread
# forever (2026-07-04 livelock: 6 resume attempts stuck on the same poison
# domain). Threads can't be killed, so extraction runs in a small process
# pool with a hard deadline; on timeout the pool is terminated and the page
# is skipped. HTML is size-capped first — the bombs are megabyte-scale.
import multiprocessing as _mp

_EXTRACT_TIMEOUT = float(os.environ.get("AW_EXTRACT_TIMEOUT", "90"))
# task 634: a domain whose captured pages are individually extraction-hostile
# (a dynamic per-record database with no static prose — apps.who.int's
# food-additives-contaminants-jecfa-database was the case that surfaced this)
# had no bound OTHER than --max-per-domain on how much of _EXTRACT_TIMEOUT it
# could burn: max-per-domain=60 x 90s = up to 90 minutes stalled on ONE domain,
# with nothing downstream distinguishing "slow domain" from "hung run". After
# this many CONSECUTIVE per-page extraction timeouts within one domain,
# process_domain stops issuing further pages for it (recorded, not silent) —
# a single non-timing-out page resets the counter, so a domain that is merely
# slow-but-working is never cut short.
_DOMAIN_TIMEOUT_BREAKER = int(os.environ.get("AW_DOMAIN_TIMEOUT_BREAKER", "5"))
_HTML_CAP = int(os.environ.get("AW_HTML_CAP", "1500000"))
# Extract-pool size (was hardcoded 2). With many fetch workers, a 2-worker pool
# queues extractions past the per-submit timeout and cascade-terminates; size it
# to the host's cores so extraction keeps pace with fetch. AW_EXTRACT_POOL.
_XPOOL_SIZE = int(os.environ.get("AW_EXTRACT_POOL", "2"))
_XPOOL = None
_XPOOL_LOCK = threading.Lock()


def _extract_in_pool(url, html, meta):
    """process_html with a hard per-page deadline. Returns ([], set(), {}, False)
    on TIMEOUT (page skipped, never blocks the run) — the trailing bool is
    `timed_out`, read by process_domain's circuit breaker (task 634) so a
    genuinely-empty-but-fast extraction is distinguishable from one that ate the
    whole _EXTRACT_TIMEOUT. meta carries want_stats=True so the extraction_junk
    gate's per-reason reject counts (task 097) come back across the process
    boundary.

    task 634, second regression this closes: a REAL exception raised inside
    the worker (e.g. LangGateInactive, or the earlier class_basis TypeError)
    used to be caught by the SAME broad `except Exception` as a genuine
    mp.TimeoutError, printed as the identical, misleading
    "extract-timeout/skip", and swallowed into a clean-looking empty result
    -- `status='ok', rows=0` reads as success. A real (non-timeout)
    exception is now RE-RAISED here, not swallowed: it propagates through
    process_domain (uncaught there, by design) to cmd_fetch's per-domain
    error boundary (an "error" status, loud and retryable -- never a silent
    "ok"), and through smoke_check() as the real exception type and message,
    not a vague zero-rows report. Only a genuine mp.TimeoutError -- the
    worker legitimately still running, unknown state -- terminates the pool
    and feeds the circuit breaker; a worker that returned cleanly with an
    exception is not corrupted and the pool is not torn down for it."""
    global _XPOOL
    meta = {**meta, "want_stats": True}
    with _XPOOL_LOCK:
        if _XPOOL is None:
            _XPOOL = _mp.get_context("spawn").Pool(_XPOOL_SIZE)
        pool = _XPOOL
    res = pool.apply_async(process_html, (url, html, meta))
    try:
        passages, links, stats = res.get(_EXTRACT_TIMEOUT)
        return passages, links, stats.get("extraction_rejects", {}), False
    except _mp.TimeoutError:
        with _XPOOL_LOCK:
            if _XPOOL is pool:
                try:
                    pool.terminate()
                except Exception:  # noqa: BLE001
                    pass
                _XPOOL = None
        print(f"  extract-timeout/skip: {url[:100]}", flush=True)
        return [], set(), {}, True
    except Exception as e:  # noqa: BLE001 — a real worker-side exception: LOUD, never masked as a timeout
        print(f"  ! extraction FAILED (not a timeout) {url[:100]}: "
              f"{type(e).__name__}: {e}", flush=True)
        raise


def reset_extraction_pool() -> None:
    """Force the NEXT _extract_in_pool call to spawn a fresh worker pool.

    task 634: the pool is a process-global singleton, and 'spawn' workers
    capture os.environ at the moment they are FORKED -- not at the moment
    _extract_in_pool is later called. A pool created earlier in the same
    long-lived process (a prior real extraction, in a test or an
    interactive session) keeps running with whatever environment it
    started with, even after os.environ changes later in that same
    process. Confirmed live: a pool spawned before AW_LANGGATE_OPTIONAL
    was set kept raising LangGateInactive in its (stale) workers long
    after the env var was set, in the SAME process. smoke_check() calls
    this first so it always verifies against a pool that matches the
    CURRENT environment, not a leftover from something earlier in this
    process."""
    global _XPOOL
    with _XPOOL_LOCK:
        if _XPOOL is not None:
            try:
                _XPOOL.terminate()
            except Exception:  # noqa: BLE001
                pass
            _XPOOL = None


def _retryable(status: int) -> bool:
    return status == 0 or status == 429 or 500 <= status <= 599


def cdx_query(client, domain, crawl, limit, retries=5):
    """CDX status-200 HTML captures for a domain. Returns [] only for a GENUINE
    empty/404; raises CdxError after exhausting retries on transient failures."""
    last = None
    for attempt in range(retries):
        _throttle()
        try:
            r = client.get(CDX.format(crawl=crawl),
                           params={"url": domain, "matchType": "domain",
                                   "output": "json", "filter": "status:200",
                                   "limit": str(limit)})
        except Exception as e:  # noqa: BLE001 — conn/timeout = transient
            last = f"conn:{type(e).__name__}"
        else:
            if r.status_code == 200:
                recs = []
                for line in r.text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        j = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue   # malformed CDX line — skip, do NOT abort
                    if "text/html" in j.get("mime", "text/html"):
                        recs.append(j)
                return recs                      # genuine result (may be empty)
            if r.status_code == 404:
                return []                        # CC: domain not captured here
            if not _retryable(r.status_code):
                raise CdxError(f"{domain}: CDX HTTP {r.status_code}")
            last = f"HTTP {r.status_code}"
        time.sleep(min(2 ** attempt, 30))        # 1,2,4,8,16s backoff
    raise CdxError(f"{domain}: CDX unreachable after {retries} tries ({last})")


# --- signed-S3 WARC fast path (task 124) -----------------------------------
# The WARC bodies the mirror serves live at the identical key under
# `s3://commoncrawl/<rec['filename']>`. Signed ranged GETs skip the mirror's
# bulk throttle (the ≤RATE req/s pacing exists for CC's shared HTTP infra, which
# signing lifts), so the S3 path is NOT `_throttle()`-paced — instead a
# configurable AGGREGATE concurrency cap (a semaphore, default 32) bounds it so a
# bug can't run away. Transport is chosen by ccx.use_s3() — the SAME decision the
# columnar resolver used, so resolve + fetch always agree per host.
_S3_MAX_CONC = int(os.environ.get("AW_CC_S3_CONCURRENCY", "32"))
_S3_SEM = threading.BoundedSemaphore(_S3_MAX_CONC)
_S3_CLIENT = None
_S3_CLIENT_LOCK = threading.Lock()


class WarcRangeFetchError(RuntimeError):
    """One bounded signed-S3 WARC range attempt failed with its identity."""


# --- shared egress byte-rate limiter (task 634, 2026-09-23) -----------------
# 2026-09-22/23: raising --workers (which only widens the ABOVE concurrency
# semaphore) drove the build box's aggregate inbound past the ZyWALL rule (<=~8-10
# MB/s bulk, aggregate, at all times -- memory zywall-protections-stay, a
# founder-ruled protection, not a derived caution) -- 26 MB/s observed with
# --workers 24, and dropping to --workers 10 alone only fell to ~21 MB/s
# because concurrency was never the thing gating bytes/sec: `_S3_SEM` bounds
# how many range-GETs run at once, not how many bytes/sec they move, and each
# WARC record's length is known BEFORE the GET (it's in the columnar/CDX
# record), so it can be metered precisely rather than guessed from worker
# count. This is a cross-process token bucket (multiple `fetch` invocations —
# one per tranche — run concurrently and must share ONE aggregate cap, not
# each get their own): state lives in a small JSON file under an OS
# `flock`, refilled at `AW_EGRESS_MBPS` MB/s with a 2s burst allowance.
# `--egress-mbps 0` (default) disables it — existing/no-network callers are
# unaffected.
class SharedByteRateLimiter:
    """Cross-process, cross-thread rate limiter over a shared, flock-guarded
    state file — a virtual-scheduling design (the SAME pattern `_throttle()`
    above already uses for req/s: a shared "next available slot" clock,
    advanced under the lock, slept on AFTER releasing it), not a token
    battery.

    task 634 (2026-09-23), bug found under real concurrent load: an EARLIER
    version of this class stored a token count and had each caller
    independently compute "how long would I need to wait to afford this",
    then release the lock and sleep. Under concurrency that is wrong: N
    callers arriving together all see ~the same deficit, each computes ~the
    same wait, and all N sleep the same duration and then proceed TOGETHER —
    a 10-thread stress test at a 1 MB/s target measured 11 MB/s actual (a
    10-thread production run at 6 MB/s measured ~24 MB/s — the same failure
    mode, same order of magnitude). A token count cannot represent "N other
    callers have already reserved this capacity but not yet consumed it"; a
    scheduled slot can. Each acquire() reserves a slice of a shared virtual
    timeline (`next_avail`) UNDER THE LOCK — nbytes/rate seconds long,
    starting no earlier than whichever is later of "now" or the previous
    reservation's end — so the SECOND concurrent caller sees the FIRST
    one's reservation already extended into the future and is forced to
    queue behind it, not just behind its instantaneously-observed token
    count. Sleeping still happens only after the lock is released, so
    reservation (cheap, serialized) and waiting (the actual cost) stay
    decoupled, same as `_throttle()`.

    acquire(n) blocks the CALLING thread/process until its reserved slot
    arrives, then returns — the debit already happened at reservation time."""

    def __init__(self, path: str, rate_bytes_per_sec: float,
                 burst_seconds: float = 2.0):
        self.path = path
        self.rate = float(rate_bytes_per_sec)
        # burst_seconds bounds how far next_avail may lag behind "now" after
        # an idle period -- i.e. how much idle credit can be spent as a burst.
        self.burst_seconds = max(burst_seconds, 0.0)
        if self.rate <= 0:
            return
        if not os.path.exists(self.path):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    json.dump({"next_avail": time.monotonic(),
                               "rate_bytes_per_sec": self.rate}, f)
            except FileExistsError:
                pass  # another process created it first — fine, we share it

    def acquire(self, nbytes: int) -> None:
        if self.rate <= 0 or nbytes <= 0:
            return
        import fcntl
        cost = nbytes / self.rate
        with open(self.path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                raw = f.read().strip()
                # Bug found live, task 664, 2026-09-26: four CC tranche
                # fetches sharing this ONE bucket file (up to 40 worker
                # threads across processes) drove the fetch's success rate
                # to ~0% -- every acquire() raised an UNCAUGHT
                # JSONDecodeError ("Extra data") on a file that read back as
                # a complete, valid JSON object immediately followed by more
                # bytes (e.g. a stray "0}"). Never re-raise on a malformed
                # read: this file holds nothing but a disposable rate-limit
                # cursor, so a corrupt read is exactly as safe to treat as
                # "no state yet" (start the bucket fresh from `now`) as a
                # missing file already is -- and the very next line
                # overwrites it with a clean, valid object regardless.
                try:
                    st = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    st = {}
                now = time.monotonic()
                next_avail = float(st.get("next_avail", now))
                # bounded idle credit: don't let a long-idle bucket grant an
                # unlimited burst just because next_avail is far in the past.
                next_avail = max(next_avail, now - self.burst_seconds)
                slot = max(now, next_avail)
                f.seek(0)
                f.truncate()
                json.dump({"next_avail": slot + cost,
                           "rate_bytes_per_sec": self.rate}, f)
                # THE ACTUAL ROOT CAUSE: `json.dump` above only writes to
                # this file OBJECT's internal buffer. The lock was being
                # released (below) before that buffer was ever flushed to
                # the shared file, so a second process could open the same
                # path, acquire the now-free lock, and read/truncate/write
                # while this process's own buffered write was still
                # in-flight -- two processes' writes interleaving at the OS
                # level is exactly how one complete JSON object ends up
                # followed by stray trailing bytes from a delayed, later
                # flush. Flushing (and fsync-ing -- this is a small,
                # frequently-written file, not a place to trade durability
                # for a syscall) INSIDE the lock, before it is released, is
                # the actual fix; the JSONDecodeError guard above is
                # defense in depth for any state a system already left
                # corrupted before this landed.
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        wait = slot - now
        if wait > 0:
            time.sleep(wait)


_EGRESS_LIMITER = None
_EGRESS_LIMITER_LOCK = threading.Lock()


def _egress_limiter():
    """Lazy singleton, built from env vars set by cmd_fetch() from --egress-*
    args before any worker starts (mirrors _S3_SEM's env-var construction
    style). AW_EGRESS_MBPS=0 (default) means "no limiter" (acquire() no-ops)."""
    global _EGRESS_LIMITER
    if _EGRESS_LIMITER is None:
        with _EGRESS_LIMITER_LOCK:
            if _EGRESS_LIMITER is None:
                mbps = float(os.environ.get("AW_EGRESS_MBPS", "0"))
                state_path = os.environ.get(
                    "AW_EGRESS_STATE", "/tmp/aw_egress_bucket.json")
                _EGRESS_LIMITER = SharedByteRateLimiter(
                    state_path, mbps * 1_000_000)
    return _EGRESS_LIMITER


def _warc_s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        with _S3_CLIENT_LOCK:
            if _S3_CLIENT is None:
                _S3_CLIENT = ccx._s3_client(
                    connect_timeout=float(os.environ.get(
                        "AW_CC_S3_CONNECT_TIMEOUT", "15")),
                    read_timeout=float(os.environ.get(
                        "AW_CC_S3_READ_TIMEOUT", "60")),
                    # task 634: was "1" (= zero retries) with no comment
                    # explaining why, unlike cc_columnar._s3_client's own
                    # documented default of 6 attempts / adaptive backoff.
                    # A transient S3 SlowDown (observed live, task 634: "An
                    # error occurred (SlowDown) ... Please reduce your
                    # request rate") crashed the WHOLE fetch process instead
                    # of retrying one range GET -- at this module's own
                    # documented "unthrottled, ≤32 concurrent" signed-S3
                    # concurrency, an occasional SlowDown is expected
                    # traffic shape, not a fatal condition. Matches
                    # cc_columnar's already-ratified default.
                    total_max_attempts=int(os.environ.get(
                        "AW_CC_S3_TOTAL_MAX_ATTEMPTS", "6")))
    return _S3_CLIENT


_CHARSET_RE = re.compile(rb"charset\s*=\s*[\"']?\s*([A-Za-z0-9_.\-]+)", re.I)
# common alias → python codec (only the ones we actually meet on CH gov hosts)
_ENC_ALIASES = {
    "windows-1252": "cp1252", "win-1252": "cp1252", "cp-1252": "cp1252",
    "iso-8859-1": "latin-1", "latin1": "latin-1", "iso8859-1": "latin-1",
    "iso-8859-15": "iso-8859-15", "utf8": "utf-8", "utf_8": "utf-8",
    "us-ascii": "utf-8", "ascii": "utf-8",
}


def _warc_response_parts(content: bytes) -> tuple[bytes, bytes] | None:
    """Return the exact HTTP header and response-body bytes from one WARC range.

    The returned body is byte-for-byte what the archived HTTP response carried;
    it is not decoded, normalised, or re-encoded.  L0 callers must hash/store
    this value, not a decoded string.  ``None`` means the ranged object was not
    a complete gzip WARC response record; an empty body remains ``b\"\"`` and is
    therefore distinct from an error.
    """
    try:
        raw = gzip.GzipFile(fileobj=io.BytesIO(content)).read()
    except (EOFError, OSError, zlib.error):
        return None
    warc_end = raw.find(b"\r\n\r\n")
    if warc_end < 0:
        return None
    warc_headers = raw[:warc_end]
    lengths = [value.strip() for name, separator, value in
               (line.partition(b":") for line in warc_headers.split(b"\r\n"))
               if separator and name.strip().lower() == b"content-length"]
    if len(lengths) != 1 or not lengths[0].isdigit():
        return None
    block_start = warc_end + 4
    block_end = block_start + int(lengths[0])
    if block_end > len(raw) or raw[block_end:] not in {b"", b"\r\n\r\n"}:
        return None
    http_block = raw[block_start:block_end]
    http_end = http_block.find(b"\r\n\r\n")
    if http_end < 0:
        return None
    return http_block[:http_end], http_block[http_end + 4:]


def _decode_http_body(http_hdrs: bytes, body: bytes) -> str:
    """Charset-aware decode of an already separated HTTP response body.

    The old body.decode('utf-8','ignore') silently DROPPED high bytes on
    legacy windows-1252 / latin-1 gov hosts (e.g. rsGE .htm) — accents were
    lost irrecoverably ('générale'→'gnrale'). Now: honour the declared charset
    (HTTP Content-Type header → HTML <meta>), else utf-8-strict with a cp1252
    fallback (single-byte, lossless for the CH-gov legacy pages). UTF-8 hosts
    (sz.ch/ti.ch) are byte-identical to before (strict utf-8 succeeds first)."""
    enc = None
    m = _CHARSET_RE.search(http_hdrs)               # 1) HTTP Content-Type header
    if not m:
        m = _CHARSET_RE.search(body[:4096])         # 2) HTML <meta charset>
    if m:
        enc = m.group(1).decode("ascii", "ignore").strip().lower()
        enc = _ENC_ALIASES.get(enc, enc)
    if enc and enc not in ("utf-8",):
        try:
            return body.decode(enc)                 # declared legacy charset
        except (LookupError, UnicodeDecodeError):
            pass                                    # bad label → sniff below
    try:
        return body.decode("utf-8")                 # 3) strict utf-8
    except UnicodeDecodeError:
        return body.decode("cp1252", "replace")     # 4) legacy fallback (lossless)


def _decode_warc_body(content: bytes):
    """Backward-compatible decoded-text view of a compressed WARC range."""
    parts = _warc_response_parts(content)
    if parts is None:
        return None
    return _decode_http_body(*parts)


def _record_filename(rec: dict) -> str:
    """Accept both CDX and columnar resolver record field names."""
    filename = rec.get("filename") or rec.get("warc_filename")
    if not filename:
        raise KeyError("WARC record requires filename or warc_filename")
    return str(filename)


def _fetch_warc_record_s3(rec):
    """Ranged get_object one compressed WARC record via signed S3."""
    off, ln = int(rec["offset"]), int(rec["length"])
    rng = f"bytes={off}-{off + ln - 1}"
    filename = _record_filename(rec)
    # task 634 (2026-09-23): the record's byte length is known BEFORE the
    # GET -- meter it against the shared egress bucket up front, so a burst
    # of concurrent workers paces itself to the configured aggregate rate
    # instead of each independently racing to fill the concurrency semaphore.
    _egress_limiter().acquire(ln)
    with _S3_SEM:
        try:
            o = _warc_s3_client().get_object(
                Bucket=ccx.S3_BUCKET, Key=filename, Range=rng)
            body = o["Body"].read()
        except Exception as exc:
            raise WarcRangeFetchError(
                f"signed-s3 range failed key={filename} range={rng} "
                f"error={type(exc).__name__}:{exc}") from exc
    if len(body) != ln:
        raise WarcRangeFetchError(
            f"signed-s3 range length mismatch key={filename} range={rng} "
            f"expected={ln} observed={len(body)}")
    return body


def _fetch_warc_record_mirror(client, rec):
    """Range-GET one compressed WARC record from the HTTPS mirror."""
    off, ln = int(rec["offset"]), int(rec["length"])
    filename = _record_filename(rec)
    url = WARC + filename
    hdr = {**UA, "Range": f"bytes={off}-{off + ln - 1}"}
    for attempt in range(4):
        _throttle()
        try:
            r = client.get(url, headers=hdr)
            if r.status_code in (200, 206):
                return r.content
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5 * (attempt + 1))
    return None


def fetch_warc_record(client, rec) -> bytes | None:
    """Fetch one compressed WARC record through the selected shared transport."""
    if ccx.use_s3():
        return _fetch_warc_record_s3(rec)
    return _fetch_warc_record_mirror(client, rec)


def fetch_warc_response_body(client, rec) -> bytes | None:
    """Fetch the exact archived HTTP response-body bytes for L0 hashing.

    Transport is signed S3 when selected by :mod:`cc_columnar`, otherwise the
    Common Crawl HTTPS mirror.  ``None`` is a fetch/parse error; ``b\"\"`` is a
    successfully archived empty response.  This distinction prevents an
    unavailable transport from being counted as an empty capture.
    """
    content = fetch_warc_record(client, rec)
    if content is None:
        return None
    parts = _warc_response_parts(content)
    return None if parts is None else parts[1]


# Short alias for callers that make the L0 byte contract explicit in their API.
fetch_warc_bytes = fetch_warc_response_body


def _fetch_warc_s3(rec):
    """Backward-compatible decoded-text signed-S3 helper."""
    content = _fetch_warc_record_s3(rec)
    return None if content is None else _decode_warc_body(content)


def fetch_warc(client, rec):
    """Backward-compatible decoded-text WARC fetch.

    New L0 writers should call :func:`fetch_warc_response_body` and hash the
    returned bytes before any decoding.  Existing extraction callers retain
    their charset-aware string result through this wrapper.
    """
    content = fetch_warc_record(client, rec)
    return None if content is None else _decode_warc_body(content)


def crawl_label(crawl):
    return crawl.replace("CC-MAIN-", "")


def dedup_rank(caps, domain):
    """Dedup by TRACKING-STRIPPED URL; importance-order (shallow/short URLs
    first) so a per-domain cap keeps high-value authoritative pages, not an
    arbitrary first-N. Stripping (aw_common.strip_tracking_url: `?utm_`/`?ref=`
    query params + `;jsessionid` path tokens) at DEDUP time matters as much as
    at store time: task 111 found 14,987/14,998 statistik.arbeitsagentur.de CC
    captures were session-id variants of 82 pages — without stripping here they
    consume the whole per-domain page budget re-fetching the same pages."""
    from cytir.core.common import strip_tracking_url
    seen, out = set(), []
    for rec in caps:
        u = rec.get("url", "").split("#")[0]
        if not u:
            continue
        key = strip_tracking_url(u)
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    out.sort(key=lambda r: (r["url"].count("/"), len(r["url"])))
    return out


# --- startup smoke check (task 634) -----------------------------------------
# A known-good, self-contained page run through the REAL pipeline -- fetch
# stubbed (no network), everything from there on live: process_html,
# trafilatura, chunking, Passage construction, _extract_in_pool's real
# process pool. Catches the exact class of defect that let task 440's
# regression ship: a field added to `common`/passage_common that Passage's
# constructor does not accept crashes every extraction, is silently
# swallowed and mislabeled "extract-timeout/skip", and the run finishes
# hours later having produced nothing -- with a healthy-looking exit code.
_SMOKE_URL = "https://smoke-check.example/aw-discovery-self-test"
_SMOKE_HTML = (
    "<html><head><title>AW discovery smoke check</title></head><body>"
    "<main><h1>Self-test article</h1>"
    "<p>This paragraph exists only to give the extractor and chunker real "
    "prose to work with at startup, long enough to clear the minimum length "
    "and quality gates the pipeline applies before any real domain is "
    "fetched, so a startup regression is caught before it burns fetch "
    "budget on a real run of many hours.</p>"
    "<p>A second paragraph of the same synthetic content, present so the "
    "chunker has enough material to emit at least one real chunk for this "
    "self-check to inspect and assert on.</p>"
    "</main></body></html>")


class SmokeCheckFailed(RuntimeError):
    """The startup self-check did not produce a real passage -- refuse to
    start a run that would otherwise silently produce nothing for hours."""


def smoke_check() -> None:
    """Refuse to start `fetch` if the real pipeline cannot turn a known-good
    page into >=1 passage. Raises SmokeCheckFailed with the real traceback
    context (never swallowed) on any failure -- exactly what task 440's
    regression needed and did not have.

    Calls aw_common.process_html DIRECTLY (in this process, no
    multiprocessing pool) rather than going through process_domain's
    _extract_in_pool. Two real, DIFFERENT failure modes were reproduced
    going through the pool in a long-lived process (a full pytest session,
    400+ modules): (1) a pool spawned earlier with a different os.environ
    keeps stale workers (fixed by reset_extraction_pool(), still called by
    callers that need the pool boundary itself verified); (2) something
    elsewhere in the same process reassigns a `process_html` binding, and
    the STALE reference submitted to a 'spawn' pool fails to pickle
    ("not the same object as aw_common.process_html") -- a real defect in
    the calling process, not in the extraction path this check exists to
    guard. The regression this check guards against (task 440's
    class_basis/n_endorser_regs crashing Passage's constructor) lives
    entirely inside process_html/Passage construction, never in the pool
    boundary -- calling it directly still fully exercises that path, and
    removes an entire class of unrelated, environment-sharing fragility
    from what is meant to be a fast, reliable startup check. The pool's OWN
    health (can it spawn, does a trivial job round-trip) is a narrower,
    separate concern from this one."""
    from cytir.extract.html_extract import process_html
    common = dict(domain="smoke-check.example", tier="C",
                  trust_tier=class_trust_for("docs", "self-signal", 0),
                  source_type=CLASS_SOURCE.get("docs", "docs"),
                  last_seen=time.strftime("%Y-%m-%d"))
    meta = {"content": True, "links": True, "want_stats": True,
            "passage_common": common, "lang_default": "en", "digest": "smoke0"}
    try:
        passages, links, stats = process_html(_SMOKE_URL, _SMOKE_HTML, meta)
    except Exception as e:  # noqa: BLE001 — surface it, never swallow
        raise SmokeCheckFailed(
            f"startup smoke check raised {type(e).__name__}: {e} -- the "
            "real extraction path cannot process a known-good page; "
            "refusing to start a run that would silently yield nothing "
            "(task 634)") from e
    if not passages:
        raise SmokeCheckFailed(
            f"startup smoke check produced 0 passages "
            f"rejects={stats.get('extraction_rejects', {})} on a known-good "
            "page -- refusing to start (task 634)")


# A queue row may name its own page depth (`max_pages`); nothing above this is accepted (tranche 6's
# docs layer, the deepest queue, runs at 1000 through --max-per-domain).
MAX_PAGES_ROW_CEILING = 1000


def host_page_limit(item: dict, default: int) -> int:
    """Pages to fetch for one queue row. `default` is --max-per-domain (60 in every watchdog launch).

    * `max_pages` on the row (task 714: tranche 13 asks 120 for endorsed edu/gov/ref hosts) REPLACES
      the default, up or down. A value that is not an integer in 1..MAX_PAGES_ROW_CEILING is refused
      loudly: a roster that cannot say how deep it wants to go must not fetch at a guessed depth.
    * `cap` on the row (curated queues, e.g. wave1_queue.json) only LOWERS the result.
    * a row with neither gets `default`, so every queue written before task 714 behaves as before."""
    limit = default
    if item.get("max_pages") is not None:
        v = item["max_pages"]
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= MAX_PAGES_ROW_CEILING:
            raise ValueError(f"queue row {item.get('domain')!r}: max_pages={v!r} is not an integer "
                             f"in 1..{MAX_PAGES_ROW_CEILING}")
        limit = v
    if item.get("cap"):
        limit = min(int(item["cap"]), limit)
    return limit


def process_domain(client, item, crawl, max_pages, preresolved=None,
                   heartbeat_pages=0):
    """Fetch + extract one domain. Returns (domain, rows, link_rows, n_pages,
    status, rejects) where status ∈ 'ok' | 'empty' | 'error' (error = resolve
    failed, NOT empty) and rejects = {reason: count} from the extraction_junk
    gate (task 097). `link_rows` is one {url, domain, links: [...]} row per fetched
    page — the outbound-endorsement side of the one-pass parse (the discovery
    loop's compounding graph; spec discovery-engine.md). Rows are emitted for
    every fetched page (links may be []) so "parsed, no links" is
    distinguishable from "never fetched".

    `preresolved` (columnar path, default): a list of WARC records already
    resolved from the columnar index — {url, warc_filename, offset, length,
    digest}; `None` distinct from `[]` (None = resolve errored → re-queue;
    [] = genuinely no captures). When `preresolved` is omitted we fall back to
    the (rate-limited) per-domain CDX HTTP query (`--resolver cdx`)."""
    dom = item["domain"]
    cls = item.get("class", "other")
    # per-host depth: the queue row's own `max_pages` (task 714) replaces the default, a curated
    # queue's `cap` lowers it; --max-per-domain is the default for a row that names neither.
    max_pages = host_page_limit(item, max_pages)
    # task 440: trust depends on HOW the class was reached, not only on which
    # class it is. prep_discovery_queue emits class_basis + n_endorser_regs;
    # both are stamped onto the ROW (below, alongside disc_class/provenance —
    # never into `common`/passage_common) so an existing point's basis is
    # auditable later without re-deriving the endorsement graph.
    #
    # task 634 fix: class_basis/n_endorser_regs were stamped into `common`
    # here, which is spread as **fields straight into Passage(**fields)
    # (aw_common.py:_content_passages -> passage_from_chunk); Passage has no
    # such constructor arguments, so EVERY extraction since task 440 shipped
    # (2026-08-20) raised TypeError inside the extraction pool worker.
    # _extract_in_pool's broad `except Exception` (needed to also catch a
    # genuine pool timeout) swallowed it and printed the indistinguishable
    # "extract-timeout/skip" -- this pipeline has produced ZERO passages via
    # any real fetch since the day task 440 merged, and nothing caught it:
    # task 440's own test suite mocks below the Passage-construction layer,
    # and the module docstring's "READY to run on trigger" claim (2026-06-29,
    # predates the regression) was never re-verified end-to-end after. Fixed
    # by stamping these two fields onto `row` post-construction, exactly like
    # disc_class/provenance already are (see the `for p in passages:` loop
    # below) -- never through the Passage constructor's field set.
    class_basis = item.get("class_basis")
    n_endorser_regs = item.get("n_endorser_regs")
    common = dict(domain=dom, tier="C",
                  trust_tier=class_trust_for(cls, class_basis, n_endorser_regs),
                  source_type=CLASS_SOURCE.get(cls, "docs"),
                  last_seen=time.strftime("%Y-%m-%d"))
    # Language default when content-detection is low-confidence (task 274, bug C
    # root). Switzerland is MULTILINGUAL — hardcoding ".ch → de" mis-stamped ~25%
    # of CH (GE/VD/VS/FR/NE/JU communes are French, TI Italian) as German, which
    # a hard `lang` filter then EXCLUDES from lang=fr/it queries AND wrongly
    # SURFACES in lang=de. A wrong guess is worse than none: for .ch we pass
    # lang_default=None → detect_lang uses content first, then the URL /fr//it/
    # marker, and only if BOTH are silent leaves lang unset (honest; excluded
    # from scoped queries but still present unscoped, never mis-filed). .de/.at/
    # .li are genuinely German-majority, so their "de" default stays.
    lang_default = None if dom.endswith(".ch") else \
                   "de" if dom.endswith((".de", ".at", ".li")) else \
                   "fr" if ".fr" in dom else "en"
    if preresolved is not None:
        # columnar path: records already resolved (offset/length use ccx keys).
        raw = [{"url": r["url"], "offset": r["offset"], "length": r["length"],
                "filename": r["warc_filename"], "digest": r.get("digest")}
               for r in preresolved]
    else:
        try:
            raw = cdx_query(client, dom, crawl, max_pages * 3)
        except CdxError:
            return dom, [], [], 0, "error", {}
    if not raw:
        return dom, [], [], 0, "empty", {}
    caps = dedup_rank(raw, dom)[:max_pages]
    rows, link_rows, n_pages = [], [], 0
    rejects: dict[str, int] = {}
    consecutive_timeouts = 0
    for cap_index, rec in enumerate(caps):
        html = fetch_warc(client, rec)
        if not html:
            continue
        n_pages += 1
        url = rec["url"].split("#")[0]
        # task 312: L0 raw-HTML archive (additive, non-fatal, AW_L0=1 default
        # on) — the WARC body is already the "raw fetch" this domain got, so
        # archiving it here means extract@N+1 / outlinks-backfill for this
        # page is a disk read instead of a re-fetch (architecture.md §4.2).
        if l0_store.l0_enabled():
            try:
                l0_store.write(url, time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                   time.gmtime()),
                               html.encode("utf-8", "ignore"),
                               extra={"domain": dom, "provenance":
                                      "endorsement-discovery"})
            except Exception as e:  # noqa: BLE001 — archival side-effect only
                print(f"  ! L0 write failed {url}: {e}", flush=True)
        passages, links, page_rejects, timed_out = _extract_in_pool(
            url, html[:_HTML_CAP], {
                "content": True, "links": True,
                "passage_common": common, "lang_default": lang_default,
                "digest": rec.get("digest"),
                "outlinks_column": outlinks_column_enabled(),
            })
        for reason, n in page_rejects.items():
            rejects[reason] = rejects.get(reason, 0) + n
        for p in passages:
            row = p.payload(); row["text"] = p.text; row["point_id"] = p.point_id
            row["provenance"] = "endorsement-discovery"
            row["disc_class"] = cls
            # task 634: class_basis/n_endorser_regs stamped onto the ROW, not
            # into common/passage_common -- see the comment above `common =`.
            row["class_basis"] = class_basis or "endorsed"
            row["n_endorser_regs"] = int(n_endorser_regs or 0)
            rows.append(row)
        link_rows.append({"url": url, "domain": dom, "links": sorted(links)})
        # circuit breaker (task 634): a page that did NOT time out resets the
        # counter even if it yielded zero passages — this only fires on the
        # specific failure mode (the extraction pool itself hanging), never on
        # a domain that is legitimately low-yield but responsive.
        consecutive_timeouts = consecutive_timeouts + 1 if timed_out else 0
        if consecutive_timeouts >= _DOMAIN_TIMEOUT_BREAKER:
            skipped = len(caps) - (cap_index + 1)
            rejects["circuit_breaker_early_stop"] = skipped
            print(f"  ! {dom}: {_DOMAIN_TIMEOUT_BREAKER} consecutive "
                  f"extract-timeouts, stopping early ({skipped} of "
                  f"{len(caps)} remaining pages skipped)", flush=True)
            break
        # intra-host heartbeat: a big-cap host (e.g. a 19k-page insee) is
        # otherwise SILENT for >20 min between its start and the outer loop's
        # `[i/N] <host>: <status>` completion line — which trips the task 100
        # supervisors' 20-min log-mtime stall guard into a kill/resume loop
        # (task 111, task 145/145b). A flushed stdout line every N fetched
        # pages resets that guard (it keys purely on the log's mtime, so any
        # flushed write suffices). Host-labelled so --workers>1 interleaving
        # stays readable. Does NOT touch fetch semantics/ordering/resume/dedup.
        if heartbeat_pages and n_pages % heartbeat_pages == 0:
            print(f"  [hb] {dom}: {n_pages}/{len(caps)} pages, {len(rows)} rows",
                  flush=True)
    if n_pages == 0:
        return dom, [], [], 0, "empty", rejects
    return dom, rows, link_rows, n_pages, "ok", rejects


# --- resume state -----------------------------------------------------------
def load_state(path):
    if path and os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:  # noqa: BLE001
            pass
    return {"done": {}}


def save_state(path, state):
    if not path:
        return
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    json.dump(state, open(tmp, "w"))
    os.replace(tmp, path)


def _free_gb(path: str) -> float:
    """Free space (GiB) on the filesystem holding `path` (or its nearest parent
    that exists)."""
    p = pathlib.Path(path).resolve()
    while not p.exists():
        p = p.parent
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize / (1 << 30)


def cmd_fetch(args):
    global _RATE
    # task 634 (2026-09-23): set the shared egress bucket's env vars BEFORE
    # any worker thread starts, so _egress_limiter()'s lazy singleton (first
    # call wins) always sees this run's configuration, never a stale one left
    # by an earlier import in the same interpreter (tests, notebooks).
    os.environ["AW_EGRESS_MBPS"] = str(getattr(args, "egress_mbps", 0) or 0)
    if getattr(args, "egress_state", None):
        os.environ["AW_EGRESS_STATE"] = args.egress_state
    # task634 (2026-09-23): the columnar RESOLVE phase's DuckDB httpfs reads
    # cannot be metered from Python at all (see --require-local-index's own
    # help text) -- when a local mirror dir is given, refuse every remote
    # parquet read instead of trying to meter it.
    require_local = getattr(args, "require_local_index", "")
    if require_local:
        os.environ["AW_CC_LOCAL_INDEX_DIR"] = require_local
        os.environ["AW_CC_REQUIRE_LOCAL_INDEX"] = "1"
    else:
        _envnames.pop_knob(os.environ, "AW_CC_REQUIRE_LOCAL_INDEX")
    if not getattr(args, "skip_smoke_check", False):
        try:
            smoke_check()
        except SmokeCheckFailed as e:
            raise SystemExit(
                f"REFUSING to start: {e}\n"
                "(pass --skip-smoke-check only if you have independently "
                "verified the extraction path some other way)") from e
        print("  smoke check: real pipeline yields >=1 passage on a known-"
              "good page — proceeding.", flush=True)
    _RATE = args.rate
    queue = json.load(open(args.queue))
    if args.only:
        keep = {d.strip() for d in args.only.split(",")}
        queue = [q for q in queue if q["domain"] in keep]
    if args.limit_domains:
        queue = queue[: args.limit_domains]

    state = load_state(args.state)
    done = state.setdefault("done", {})
    todo = [q for q in queue if q["domain"] not in done] if args.resume else queue
    egress_mbps = float(os.environ.get("AW_EGRESS_MBPS", "0"))
    transport = ("S3-signed (%s, ≤%d concurrent)" %
                 (f"egress-capped {egress_mbps:g} MB/s shared"
                  if egress_mbps > 0 else "unthrottled", _S3_MAX_CONC)) \
        if ccx.use_s3() else ("HTTPS-mirror (≤%g req/s)" % args.rate)
    # per-host depth (task 714): validated for every row BEFORE the run starts (a bad row refuses the
    # whole fetch, not one domain hours in); the default is --max-per-domain.
    depth = {q["domain"]: host_page_limit(q, args.max_per_domain) for q in todo}
    deeper = sum(1 for v in depth.values() if v > args.max_per_domain)
    print(f"queue={len(queue)} | resume-skip={len(queue) - len(todo)} | "
          f"to-fetch={len(todo)} | crawl={args.crawl} | transport={transport} "
          f"workers={args.workers} max/dom={args.max_per_domain}"
          + (f" | depth: {deeper} host(s) above the default, max {max(depth.values())}"
             if deeper else ""), flush=True)
    # echo the egress cap into the state file too (order: "echoed in the
    # state file") -- a operator inspecting state.json (not just the log)
    # can see what this run is bound by without cross-referencing the CLI.
    state["_egress"] = {
        "mbps": egress_mbps,
        "state_path": os.environ.get("AW_EGRESS_STATE", "/tmp/aw_egress_bucket.json"),
    }
    if not todo:
        print("nothing to do."); return

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Split-mode guard (task 124): when the fetch worker runs on the SERVE box
    # (the serving box, S3-capable) it must never fill the disk out from under the
    # live index. Hard-stop before AND periodically during the run if the spool
    # filesystem drops below --min-free-gb. State is per-domain crash-safe, so a
    # stop here just means "resume later / on another box".
    free0 = _free_gb(str(out.parent))
    if free0 < args.min_free_gb:
        raise SystemExit(f"ABORT: only {free0:.1f} GiB free on {out.parent} "
                         f"(< --min-free-gb {args.min_free_gb}). Not starting a fetch.")
    links_out = pathlib.Path(args.links_out) if args.links_out else \
        out.parent / "outlinks.jsonl"
    links_out.parent.mkdir(parents=True, exist_ok=True)
    # Granular timeouts + a process-level socket default: a WARC read hung 3h in
    # ssl.py past httpx's flat timeout=60 (2026-07-03 stall, PID 1546067) — the
    # socket default is the belt that catches reads slipping past httpx's layer.
    socket.setdefaulttimeout(90)
    client = make_client(
        HttpSpec(timeout=httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=120.0),
                 headers={k.lower(): v for k, v in UA.items()}, follow_redirects=True),
        transport=httpx.HTTPTransport(limits=httpx.Limits(
            max_connections=args.workers * 2, max_keepalive_connections=args.workers)))
    status_ct = {"ok": 0, "empty": 0, "error": 0}
    errors, n_rows = [], 0
    mode = "a" if (args.resume and out.exists()) else "w"

    # --- columnar pre-resolution (default; ADR-006/014) ----------------------
    # Resolve EVERY queued domain → its WARC records up front via the columnar
    # Parquet index on the unblocked WARC mirror (DuckDB, surtkey-prefix pruning:
    # ~1 of 300 files per domain). This replaces the per-domain CDX HTTP API
    # (`index.commoncrawl.org`), which rate-limits/IP-bans bulk callers. Domains
    # whose resolution errors are surfaced as `error` (re-queued), never `empty`.
    preresolved: dict[str, list | None] = {}
    if args.resolver == "columnar":
        con = ccx._con(threads=args.columnar_threads)
        # chunk the domain list so one transient parquet read can't poison the
        # whole batch; an errored chunk leaves its domains unresolved (None).
        todo_doms = [q["domain"] for q in todo]
        CH = args.columnar_chunk
        for i in range(0, len(todo_doms), CH):
            chunk = todo_doms[i:i + CH]
            try:
                got = ccx.resolve_domains(
                    args.crawl, chunk, con=con,
                    per_domain=max(args.max_per_domain,
                                   max(depth[d] for d in chunk)) * 3)
            except Exception as e:  # noqa: BLE001 — transient parquet/HTTP error
                print(f"  [columnar] chunk {i//CH} ERRORED ({e!r}); its "
                      f"{len(chunk)} domains stay unresolved → 'error' → re-run",
                      file=sys.stderr, flush=True)
                for d in chunk:
                    preresolved[d] = None        # None = error, NOT empty
                continue
            for d in chunk:
                preresolved[d] = got.get(d, [])  # [] = genuine no-capture
            nres = sum(1 for d in chunk if preresolved.get(d))
            print(f"  [columnar {i//CH + 1}/{-(-len(todo_doms)//CH)}] "
                  f"{crawl_label(args.crawl)}: {nres}/{len(chunk)} domains have "
                  f"captures", flush=True)
        con.close()

    def work(item):
        dom = item["domain"]
        if args.resolver == "columnar":
            # task634 (2026-09-23): a columnar resolve error is recorded as
            # preresolved[dom] = None (see the chunk loop above) -- but
            # process_domain's OWN contract treats `preresolved is None` as
            # "the caller omitted it, fall back to CDX" (its docstring says
            # "None distinct from [] ... re-queue", yet nothing upstream of
            # it enforced that: this call site used to pass the errored
            # domain's None straight through, silently substituting a live
            # CDX lookup for what was supposed to be a deferred retry -- a
            # real behavior gap, found while making --require-local-index's
            # "refuse the read outright, never touch the network" claim
            # actually true; it existed independent of that flag). Return
            # 'error' directly here for a columnar-error domain, WITHOUT
            # calling process_domain at all -- no CDX call, no WARC fetch,
            # genuinely zero network for it this run.
            if dom in preresolved and preresolved[dom] is None:
                return dom, [], [], 0, "error", {}
            pre = preresolved.get(dom)
        else:
            pre = None
        return process_domain(client, item, args.crawl, args.max_per_domain,
                              preresolved=pre,
                              heartbeat_pages=args.heartbeat_pages)

    n_links = 0
    all_rejects: dict[str, int] = {}
    with out.open(mode) as f, links_out.open(mode) as lf, \
            ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(work, q): q for q in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                dom, rows, link_rows, n_pages, status, rejects = fut.result()
            except Exception as e:  # noqa: BLE001 — one domain's uncaught
                # fetch/extract failure (task 634: an S3 SlowDown that
                # exhausted retries was OBSERVED to crash the entire
                # multi-hour run here, losing every other domain's
                # already-durable progress). One bad domain is an "error"
                # (retryable on the next --resume), exactly like a resolve
                # failure a few lines up in this same function — never a
                # reason to abort domains that already succeeded.
                dom = futs[fut]["domain"]
                print(f"  ! {dom}: uncaught {type(e).__name__}: {e} "
                      f"(marked error, will re-run on --resume)", flush=True)
                dom, rows, link_rows, n_pages, status, rejects = (
                    dom, [], [], 0, "error", {})
            status_ct[status] += 1
            for reason, n in rejects.items():
                all_rejects[reason] = all_rejects.get(reason, 0) + n
            if status == "error":
                errors.append(dom)
            else:
                done[dom] = {"status": status, "pages": n_pages, "rows": len(rows)}
                if rejects:
                    done[dom]["extraction_rejects"] = rejects
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
            for lr in link_rows:
                lf.write(json.dumps(lr, ensure_ascii=False) + "\n")
                n_links += len(lr["links"])
            if status != "error":
                f.flush(); lf.flush()           # rows durable BEFORE the marker
                save_state(args.state, state)   # crash-safe per-domain marker
            if i % 16 == 0 and _free_gb(str(out.parent)) < args.min_free_gb:
                print(f"⚠ ABORT: free space on {out.parent} < "
                      f"{args.min_free_gb} GiB — stopping cleanly (resume later). "
                      f"Done {i}/{len(todo)} domains.", flush=True)
                break
            rej = f" rej={rejects}" if rejects else ""
            print(f"  [{i}/{len(todo)}] {dom}: {status} {n_pages}p {len(rows)}r "
                  f"{sum(len(lr['links']) for lr in link_rows)}l{rej}", flush=True)

    print(f"\nFETCH done. status={status_ct} rows={n_rows} "
          f"extraction_rejects={all_rejects or {}} -> {out} | "
          f"outlinks={n_links} -> {links_out}")
    if errors:
        print(f"⚠ {len(errors)} domains ERRORED (CDX transient, NOT empty) — re-run "
              f"to retry only these (they are NOT marked done). e.g. {errors[:5]}")


#: task 634 (2026-09-22/23): tranche 1's 204,737-row sealed pack was embedded
#: through THIS command (config B: arctic-embed-l-v2, int8, MRL-128) and then
#: scrolled straight into an admission-ledger pack (seal_cell_packs.py) — a
#: shape PR #279 could only enter with `admission: false`, because config B's
#: truncated/quantized vectors are not the durable artifact task 413 requires
#: (embed once at full 1024-d fp16, store it, MRL-truncate later — truncation
#: is free, widening is a full re-embed). Config B remains correct for its
#: ORIGINAL purpose — a quick, cheap staging collection for measurement (a
#: quality funnel, a ranking smoke test) that is never scrolled into a sealed
#: pack. `--intent` makes every caller say which one this run is, so the
#: mistake can't recur silently: `admission-pack` refuses here, at the top,
#: before any GPU time is spent, and points at the ratified chain
#: (`task 413a embed_shards.py`, `task 633 run_embed_lease.sh`).
CONFIG_B_NOT_PROMOTABLE = (
    "REFUSING to embed: --intent admission-pack was requested, and config B "
    "(arctic-embed-l-v2, int8, MRL-128 -- what this command runs) is NOT a "
    "promotable artifact (task 413/634 ruling: embed once at full 1024-d "
    "fp16, store it, MRL-truncate later -- truncation is free, "
    "widening is a full re-embed). Use the ratified 1024-d fp16 chain "
    "instead: task 413a embed_shards.py (called via "
    "task 633 run_embed_lease.sh for the GPU-lock discipline), "
    "over shard-*.texts.jsonl.gz built from this same gated "
    "passages_ok.jsonl. Pass --intent measurement-only for a throwaway "
    "staging collection (funnel/ranking checks) that will never be sealed "
    "into an admission-ledger pack.")


def cmd_embed(args):
    """Delegate to cytir.embed.bakeoff with the LOCKED config-B flags (aw_v3).

    CONSENT GATE (ADR-009 / Tim 2026-07-03): discovery passages must pass the
    live per-host consent gate (consent_gate.py) BEFORE embed/merge —
    embed consumes gated/passages_ok.jsonl (rows stamped search_ok/train_ok),
    never the raw fetch output. Declined hosts become source cards instead;
    unknown hosts are held for retry.

    PROMOTABILITY FENCE (task 634): --intent is required. `admission-pack`
    refuses unconditionally -- see CONFIG_B_NOT_PROMOTABLE. This is a run-
    start refusal, before the consent-stamp check and before any subprocess
    is spawned, so a caller cannot burn GPU time before finding out."""
    if args.intent == "admission-pack":
        raise SystemExit(CONFIG_B_NOT_PROMOTABLE)
    # refuse ungated input: gated rows carry the stamped consent flags.
    with open(args.inp) as f:
        first = next((l for l in f if l.strip()), None)
    if first is not None and "search_ok" not in json.loads(first) \
            and not args.allow_ungated:
        raise SystemExit(
            "REFUSING to embed: input rows carry no consent stamp "
            "(search_ok/train_ok). Run  consent_gate.py gate  over the fetch "
            "output and embed gated/passages_ok.jsonl instead "
            "(--allow-ungated to override for non-discovery corpora).")
    # the EMBED stage is `cytir.embed.bakeoff` (task 2329); PYTHONPATH carries the repo root so `-m` finds `aw` without moving the caller's cwd (relative --in paths).
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cmd = [sys.executable, "-m", "cytir.embed.bakeoff",
           "--in", args.inp, "--embedder", "arctic-l-v2",
           "--collection", args.collection, "--qdrant-url", args.qdrant_url,
           "--quant", "int8", "--mrl-dim", "128", "--legs", "bm25",
           "--max-seq", "512", "--batch", str(args.batch), "--on-disk", "--resume"]
    print("EMBED (config B):", " ".join(cmd), flush=True)
    # task 307 (preflight A5/T29 resolution, specs/aw_b1-preflight.md §4): this
    # used to drop the `snippet` field so the staged payload matched aw_v3's
    # shape. aw_b1 is a FRESH collection — that "match aw_v3" reason is void —
    # and the ≤25-word extractive snippet is required by ADR-027 for
    # ranking/display and is not the anon-RAM/novel-query latency culprit (that
    # was the sparse BM25 index residency, fixed by on_disk + keep-warm timer,
    # memory `aw-write-churn-anon-bloat`). `process_html` -> `extractive_snippet`
    # already caps every row at 25 words (aw_common.py default, task 307), so
    # every record embedded here ships its snippet unmodified.
    raise SystemExit(subprocess.call(cmd, env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH", "")]))}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="CC fetch + extract → passages.jsonl (resumable)")
    f.add_argument("--queue", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--links-out", default="",
                   help="sidecar outbound-links jsonl (one {url, domain, links} "
                        "row per fetched page — the endorsement-graph side of "
                        "the one-pass parse). Default: <out dir>/outlinks.jsonl")
    f.add_argument("--state", default="")
    f.add_argument("--crawl", default=DEFAULT_CRAWL)
    f.add_argument("--max-per-domain", type=int, default=60,
                   help="default per-domain page depth. A queue row's own "
                        "'max_pages' replaces it for that host (up to 1000; "
                        "task 714), its 'cap' field (curated queues) lowers it")
    f.add_argument("--rate", type=float, default=4.0,
                   help="aggregate req/s for WARC fetch across ALL workers "
                        "(≤5; ADR-014). Columnar resolve is NOT rate-paced — "
                        "DuckDB's own gentle single-stream httpfs handles it.")
    f.add_argument("--workers", type=int, default=2)
    f.add_argument("--egress-mbps", type=float, default=0.0,
                   help="aggregate inbound MB/s cap across ALL WARC range-GETs "
                        "(signed-S3 path only) -- shared, cross-process, via a "
                        "flock-guarded token bucket at --egress-state (task 634: "
                        "raising --workers alone does not bound bytes/sec, only "
                        "concurrency; the ZyWALL rule is an aggregate, "
                        "always-on cap -- memory zywall-protections-stay). "
                        "0 (default) = no cap, old behaviour. When two or more "
                        "tranches run concurrently, pass the SAME --egress-state "
                        "path AND the SAME --egress-mbps value (the intended "
                        "AGGREGATE cap, not a per-process share of it) to every "
                        "one of them -- the refill math uses THIS process's own "
                        "rate against the shared file's token count each time it "
                        "acquires, so mismatched rates across callers of the "
                        "same bucket produce an inconsistent effective cap. "
                        "Passing 0 to some callers does NOT reserve headroom for "
                        "others -- it makes that caller's own draws unmetered.")
    f.add_argument("--egress-state", default="/tmp/aw_egress_bucket.json",
                   help="path to the shared egress token-bucket state file "
                        "(flock-guarded JSON). Every concurrently-running "
                        "fetch that should share one aggregate cap must pass "
                        "the SAME path.")
    f.add_argument("--heartbeat-pages", type=int, default=200,
                   help="print a flushed '  [hb] <host>: <p>/<budget> pages, "
                        "<r> rows' line every N pages fetched WITHIN a host, so "
                        "a big-cap host (e.g. a 19k-page insee) keeps resetting "
                        "the supervisors' 20-min log-mtime stall guard "
                        "(task 100) instead of tripping a kill/resume loop "
                        "(task 111/145). 0 = off. Does not affect fetch "
                        "semantics/ordering/resume/dedup.")
    f.add_argument("--resolver", choices=["columnar", "cdx"], default="columnar",
                   help="how to resolve domain→WARC records. 'columnar' (default):"
                        " DuckDB over the Parquet cc-index on the unblocked WARC "
                        "mirror, surtkey-pruned per domain (ADR-006/014, the "
                        "at-scale path). 'cdx': the rate-limited CDX HTTP API "
                        "(index.commoncrawl.org) — small batches only.")
    f.add_argument("--columnar-threads", type=int, default=8,
                   help="DuckDB httpfs threads for the columnar RESOLVE query "
                        "(signed-S3 metadata/row-group reads, NOT the WARC "
                        "range-GET fetch -- that transport auto-detects "
                        "signed-S3 and is already unthrottled up to ~32 "
                        "concurrent). Was 2 (a leftover from the anonymous-"
                        "HTTPS-mirror caution in cc_columnar._con's docstring, "
                        "which does not apply once signed-S3 credentials are "
                        "in play). Raised to 8 (task 634, tranche 2: resolve "
                        "measured ~20 domains/minute at threads=2, the run's "
                        "own bottleneck, 88 min for 10/59 batches) -- a live "
                        "tranche already running at the old default keeps its "
                        "own args; this changes only future launches.")
    f.add_argument("--columnar-chunk", type=int, default=200,
                   help="domains per columnar resolve query-group (bounds the "
                        "blast radius of a transient parquet read error)")
    f.add_argument("--require-local-index", default="",
                   help="task634 (2026-09-23): a local CC columnar-index "
                        "mirror dir (mirror_cc_index.py's --out). "
                        "When set, the columnar resolve phase REFUSES any "
                        "parquet read that would touch the network -- "
                        "DuckDB's own httpfs does its S3 I/O in native code "
                        "with no Python-level byte hook (confirmed: it "
                        "ignores HTTP_PROXY/HTTPS_PROXY), so this is the "
                        "only way to guarantee it never bypasses "
                        "--egress-mbps. A chunk needing a not-yet-mirrored "
                        "file is deferred (retried later), never marked "
                        "'empty' -- see RemoteIndexReadBlocked. Leave unset "
                        "only when the mirror is complete for --crawl, or "
                        "when running WITHOUT an egress cap to respect.")
    f.add_argument("--resume", action="store_true", default=True)
    f.add_argument("--no-resume", dest="resume", action="store_false")
    f.add_argument("--limit-domains", type=int, default=0, help="cap #domains")
    f.add_argument("--min-free-gb", type=float, default=10.0,
                   help="hard-stop the fetch if the spool filesystem drops below "
                        "this many GiB free (split-mode safety on the SERVE box; "
                        "state is per-domain resumable). Default 10.")
    f.add_argument("--only", default="", help="comma list of domains (dry-run subset)")
    f.add_argument("--skip-smoke-check", action="store_true",
                   help="skip the startup self-check that refuses to start "
                        "if the real pipeline yields 0 passages on a known-"
                        "good page (task 634); only for a caller that has "
                        "independently verified the path some other way")
    f.set_defaults(func=cmd_fetch)

    e = sub.add_parser("embed", help="config-B embed → staging collection")
    e.add_argument("--in", dest="inp", required=True)
    e.add_argument("--collection", default="aw_v3_p2_discovery")
    e.add_argument("--qdrant-url", default="http://localhost:6333",
                   help="a non-loopback Qdrant/TEI/API host must also be set in AW_QDRANT_URL / AW_TEI_URL / AW_API_URL, or the internal client refuses it")
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--allow-ungated", action="store_true",
                   help="bypass the consent-stamp check (non-discovery corpora "
                        "whose consent is carried by their own dump license)")
    e.add_argument("--intent", required=True,
                   choices=["measurement-only", "admission-pack"],
                   help="measurement-only: a throwaway staging collection "
                        "(funnel/ranking checks) that is never sealed into an "
                        "admission-ledger pack. admission-pack: REFUSES -- "
                        "config B is not a promotable artifact (task 413/634); "
                        "use the 1024-d fp16 chain instead (see "
                        "CONFIG_B_NOT_PROMOTABLE). No default: every caller "
                        "states which one this run is.")
    e.set_defaults(func=cmd_embed)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
