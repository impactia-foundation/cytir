#!/usr/bin/env python3
"""Common-Crawl **columnar index** resolver — batch URL → WARC-record lookup over
the public Parquet index via signed S3 by default (HTTPS mirror fallback), via DuckDB.

Why columnar, not the CDX HTTP API
----------------------------------
The CDX HTTP API (`index.commoncrawl.org/<crawl>-index`) rate-limits aggressively
and IP-bans bulk callers (ADR-014; observed: the orchestrator box's IP gets `connection refused`
after a burst). The columnar index is the ADR-recommended at-scale path: it lives
on the SAME unblocked mirror that serves WARCs (`data.commoncrawl.org`), is sorted
by `url_surtkey`, and carries Parquet row-group min/max stats so DuckDB pushes a
surtkey predicate down and reads only the relevant byte ranges. A whole crawl's
`subset=warc` partition is ~300 sorted parquet files; one batched query resolves
thousands of exact URLs against it.

This module does NOT fetch page content (that's the WARC range-GET, kept in the
caller) and NEVER hits an origin server — it is pure CC-index access.

API
---
    resolve(crawl_id, urls) -> dict[url -> record]
        record = {url, warc_filename, offset, length, digest, fetch_time,
                  content_truncated, truncation_reason}  (status-200 HTML)
      Batched DuckDB query of the crawl's warc-subset parquet files for the SURT
      keys of `urls`. Missing urls are simply absent from the returned dict.

    crawl_paths(crawl_id) -> list[str]   # the 300 warc-subset parquet HTTPS urls
"""
from __future__ import annotations

import gzip
import io
import os
from datetime import datetime, timezone
from functools import lru_cache

from typing import TYPE_CHECKING

from cytir.acquire import cc_digest as CC_DIGEST
import httpx

from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client
from cytir.core.surtkey import assert_key_parity, surt_key_or_none

if TYPE_CHECKING:  # duckdb is a heavy import that only the CC-index path needs: load it where it is used,
    import duckdb  # so the news discovery path (sitemap_ingest -> this module) runs without it installed


def __getattr__(name: str):  # `cc_columnar.duckdb` keeps working for callers that reach it as an attribute
    if name == "duckdb":
        import duckdb
        return duckdb
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


MIRROR = "https://data.commoncrawl.org/"
# Signed-S3 fast path (ADR/task-124): the SAME objects the HTTPS mirror serves
# live in the AWS Open-Data bucket `s3://commoncrawl/` under the identical key
# suffix (a manifest line, or a WARC record's `filename`, is the S3 key verbatim).
# Signed GETs bypass the mirror's bulk throttling (~26 MB/s vs ~2 pages/s/worker);
# the bucket is NOT requester-pays (open data) but anonymous GET = 403, so a
# credential in the botocore chain (~/.aws) is required to sign. Same interface,
# new transport — every read simply swaps the base prefix.
S3_BUCKET = "commoncrawl"
S3_PREFIX = "s3://commoncrawl/"
S3_REGION = os.environ.get("AW_CC_S3_REGION", "us-east-1")
_PATHS_CACHE: dict[str, list[str]] = {}
_S3_DECISION: bool | None = None
_SPEC = HttpSpec(timeout=60, follow_redirects=True,
                 headers={"user-agent": "authoritative-web/0.1 (+https://websear.ch)"})
_CLIENT: httpx.Client | None = None


def _h() -> httpx.Client:
    """The consent-guarded CC data-host client, built on first use (spec step 4)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = make_client(_SPEC)
    return _CLIENT


# --- transport selection: signed S3 vs HTTPS mirror -------------------------
def _creds_available() -> bool:
    """True iff the botocore credential chain (env / ~/.aws / instance role)
    resolves a key — needed to SIGN S3 GETs (anonymous = 403). Never reads or
    logs the secret itself."""
    try:
        import boto3
        return boto3.Session().get_credentials() is not None
    except Exception:  # noqa: BLE001 — boto3 absent or misconfigured → no S3
        return False


def _probe_s3(timeout: float = 5.0) -> bool:
    """A <=`timeout`s signed reachability probe from this host.

    S3 is the normal Common Crawl route on Heimdall as well as other workers.
    The mirror is retained only as a fail-closed transport fallback when the
    credential chain or current network route genuinely fails this live probe.
    No retries: a broken route must not stall planner startup.
    """
    try:
        import boto3
        from botocore.config import Config
        c = boto3.client("s3", region_name=S3_REGION,
                         config=Config(connect_timeout=timeout, read_timeout=timeout,
                                       retries={"max_attempts": 0}))
        c.list_objects_v2(Bucket=S3_BUCKET,
                          Prefix="cc-index/table/cc-main/warc/", MaxKeys=1)
        return True
    except Exception:  # noqa: BLE001 — timeout / blackhole / auth → use mirror
        return False


def use_s3() -> bool:
    """Whether to read the columnar index (and WARC bodies) from signed S3 rather
    than the HTTPS mirror. Decision cached per process.

      AW_CC_S3=1 → force S3 (still requires creds to sign; falls back if absent)
      AW_CC_S3=0 → force the mirror (exactly today's behaviour)
      unset      → AUTO: creds present AND a <=5s probe from this host succeeds.

    The auto rule keeps the pipeline portable while making signed S3 the normal
    route on Heimdall; HTTPS is evidence-backed fallback, never a standing
    per-host policy."""
    global _S3_DECISION
    if _S3_DECISION is not None:
        return _S3_DECISION
    ov = os.environ.get("AW_CC_S3")
    if ov == "0":
        _S3_DECISION = False
    elif ov == "1":
        _S3_DECISION = _creds_available()      # forced, but signing needs creds
    else:
        _S3_DECISION = _creds_available() and \
            _probe_s3(float(os.environ.get("AW_CC_S3_PROBE", "5")))
    return _S3_DECISION


def _reset_s3_decision():
    """Test hook: clear the cached transport decision (so env changes re-apply)."""
    global _S3_DECISION
    _S3_DECISION = None


@lru_cache(maxsize=8)
def _s3_client(*, connect_timeout: float = 15, read_timeout: float = 60,
               total_max_attempts: int = 6):
    """A process-shared, connection-pooled boto3 client for bulk S3 reads."""
    import boto3
    from botocore.config import Config
    return boto3.client("s3", region_name=S3_REGION,
                        config=Config(connect_timeout=connect_timeout,
                                      read_timeout=read_timeout,
                                      retries={"total_max_attempts": total_max_attempts,
                                               "mode": "adaptive"},
                                      max_pool_connections=64))


def url_surtkey(url: str) -> str | None:
    """CC `url_surtkey` for an exact URL (e.g. com,bbc)/news/x). None on failure.

    Every lookup key of this module is made here. The first call in a process
    proves the key function still gives Common Crawl's own key for a handful of
    real URLs (`assert_key_parity`); a wrong key would make every lookup miss,
    and a miss looks like "no capture", so the run refuses instead."""
    assert_key_parity()  # raises RuntimeError on a mismatch; outside the try on purpose
    return surt_key_or_none(url)


def _fetch_time_iso(value) -> str | None:
    """JSON-stable UTC representation of the columnar ``fetch_time`` value."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        value = value.astimezone(timezone.utc)
        return value.isoformat().replace("+00:00", "Z")
    text = str(value).strip()
    if not text:
        return None
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1)
    if text.endswith("+00:00"):
        text = text[:-6] + "Z"
    elif not text.endswith("Z") and "+" not in text[10:] and "-" not in text[10:]:
        text += "Z"
    return text


def _truncation_metadata(value) -> tuple[bool, str | None]:
    """Normalise CC's STRING truncation field into a stable boolean + reason.

    The URL Index schema uses NULL/empty for a complete response and a reason
    string (for example ``"length"``) for a truncated one.  Treating the raw
    value as a boolean would make the string ``"false"`` truthy and would also
    leak a schema-specific tri-state into downstream receipts.
    """
    if value is None:
        return False, None
    reason = str(value).strip()
    return (bool(reason), reason or None)


def _manifest_key(crawl_id: str) -> str:
    return f"crawl-data/{crawl_id}/cc-index-table.paths.gz"


def _read_manifest_lines(crawl_id: str) -> list[str]:
    """Raw lines of the crawl's index-table manifest, via S3 (signed) or the
    HTTPS mirror — whichever `use_s3()` selects. Retried generously: the manifest
    is the single dependency for the whole batch and the mirror can return a
    transient 403/429 right after a request burst (observed 2026-06-30)."""
    import time
    last = None
    if use_s3():
        key = _manifest_key(crawl_id)
        c = _s3_client()
        for attempt in range(8):
            try:
                body = c.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                return gzip.open(io.BytesIO(body), "rt").read().splitlines()
            except Exception as e:  # noqa: BLE001 — transient S3 blip → back off
                last = type(e).__name__
            time.sleep(min(4.0 * (attempt + 1), 30))
        raise RuntimeError(f"cannot fetch columnar manifest (s3) for {crawl_id}: {last}")
    man = MIRROR + _manifest_key(crawl_id)
    for attempt in range(8):
        try:
            r = _h().get(man)
            if r.status_code == 200:
                return gzip.open(io.BytesIO(r.content), "rt").read().splitlines()
            last = f"HTTP {r.status_code}"      # 403/429 = mirror throttling → back off
        except httpx.HTTPError as e:
            last = repr(e)
        time.sleep(min(4.0 * (attempt + 1), 30))
    raise RuntimeError(f"cannot fetch columnar manifest for {crawl_id}: {last}")


def _local_mirror_dir(crawl_id: str) -> str | None:
    """task634 (2026-09-23, Order 2): AW_CC_LOCAL_INDEX_DIR, if set, points at
    a local mirror of ONE crawl's subset=warc parquet files (built by
    the retired `mirror_cc_index` script) -- e.g. .../cc-index-mirror/<crawl_id>. Only
    honoured for the EXACT crawl it names (a mirror for one crawl must never
    silently serve as a stand-in for another)."""
    root = os.environ.get("AW_CC_LOCAL_INDEX_DIR")
    if not root:
        return None
    d = os.path.join(root, crawl_id) if os.path.basename(root) != crawl_id else root
    return d if os.path.isdir(d) else None


def _local_mirror_manifest(local_dir: str) -> dict:
    man_path = os.path.join(local_dir, "manifest.json")
    if os.path.exists(man_path):
        import json
        with open(man_path) as f:
            return json.load(f).get("files", {})
    return {}


def crawl_paths(crawl_id: str) -> list[str]:
    """Paths of the crawl's `subset=warc` parquet index files (cached).

    task634 (2026-09-23, Order 2): per-file, prefers a LOCAL mirrored copy
    (AW_CC_LOCAL_INDEX_DIR) over the network when one exists and the mirror's
    own manifest confirms it downloaded clean (status="done", size matches
    the file on disk) -- this works correctly with a PARTIAL mirror (falls
    back to the network per-file, not all-or-nothing), so the resolve phase
    speeds up incrementally as the mirror download progresses, and a caller
    never has to wait for the full ~183 GB before seeing any benefit.
    Otherwise: the base prefix (`s3://commoncrawl/` vs the HTTPS mirror) is
    chosen by `use_s3()`; the manifest line suffix is identical on both
    remote transports."""
    if crawl_id in _PATHS_CACHE:
        return _PATHS_CACHE[crawl_id]
    lines = [ln.strip() for ln in _read_manifest_lines(crawl_id)
             if "subset=warc/" in ln]
    local_dir = _local_mirror_dir(crawl_id)
    local_files = _local_mirror_manifest(local_dir) if local_dir else {}
    remote_base = S3_PREFIX if use_s3() else MIRROR
    paths = []
    for key in lines:
        rec = local_files.get(key)
        local_path = os.path.join(local_dir or "", os.path.basename(key))
        if (rec and rec.get("status") == "done" and os.path.exists(local_path)
                and os.path.getsize(local_path) == rec.get("size")):
            paths.append(local_path)
        else:
            paths.append(remote_base + key)
    _PATHS_CACHE[crawl_id] = paths
    return paths


_BOUNDS_CACHE: dict[str, list[tuple[str, str, str]]] = {}


def file_bounds(crawl_id: str, con) -> list[tuple[str, str, str]]:
    """Per-file (path, surtkey_min, surtkey_max) for the crawl's warc-subset
    parquet files, read from Parquet FOOTERS only (fast: ~6 s for all ~300 files)
    and cached. The files are non-overlapping surtkey ranges, so each wanted key
    maps to exactly one file → we can prune to just the files we need instead of
    scanning all 300. Sorted by min."""
    if crawl_id in _BOUNDS_CACHE:
        return _BOUNDS_CACHE[crawl_id]
    paths = crawl_paths(crawl_id)
    flist = "['" + "','".join(paths) + "']"
    rows = con.execute(f"""
        SELECT file_name, min(stats_min_value), max(stats_max_value)
        FROM parquet_metadata({flist}) WHERE path_in_schema='url_surtkey'
        GROUP BY file_name""").fetchall()
    bounds = [(fn, mn, mx) for fn, mn, mx in rows if mn is not None and mx is not None]
    bounds.sort(key=lambda b: b[1])
    _BOUNDS_CACHE[crawl_id] = bounds
    return bounds


def index_local_work(crawl_id: str, urls: list[str], domains: list[str], con) -> list[dict]:
    """Map all exact SURTs and host/domain prefix ranges to physical Parquet files.

    This is the shared-planner topology primitive.  Input order and logical
    site chunking cannot affect the returned work: one row names one physical
    index file and all wanted exact keys/prefixes that can occur in that file.
    A caller can therefore checkpoint per file and keep total remote queries at
    O(files), rather than O(site-batches × files).
    """
    import bisect
    bounds = file_bounds(crawl_id, con)
    mins = [row[1] for row in bounds]
    work: dict[str, dict] = {
        filename: {"file": filename, "exact": {}, "prefixes": {}}
        for filename, _minimum, _maximum in bounds}
    exact = {}
    for url in sorted(set(urls)):
        key = url_surtkey(url)
        if key:
            exact.setdefault(key, url)
    for key, url in exact.items():
        index = bisect.bisect_right(mins, key) - 1
        if (0 <= index < len(bounds) and
                bounds[index][1] <= key <= bounds[index][2]):
            work[bounds[index][0]]["exact"][key] = url
    prefixes = {}
    for domain in sorted(set(domains)):
        prefix = domain_surtkey_prefix(domain)
        if prefix:
            prefixes.setdefault(prefix, domain)
    for prefix, domain in prefixes.items():
        low, high = prefix + "/", prefix + "/￿"
        for filename, minimum, maximum in bounds:
            if not (maximum < low or minimum > high):
                work[filename]["prefixes"][prefix] = domain
    return [
        {"file": value["file"],
         "exact": sorted(value["exact"].items()),
         "prefixes": sorted(value["prefixes"].items())}
        for value in work.values() if value["exact"] or value["prefixes"]
    ]


def _record(url, filename, offset, length, digest, fetched, truncated) -> dict:
    is_truncated, reason = _truncation_metadata(truncated)
    return {"url": url, "warc_filename": filename, "offset": int(offset),
            "length": int(length), "digest": digest,
            "fetch_time": _fetch_time_iso(fetched),
            "content_truncated": is_truncated,
            "truncation_reason": reason}


def _mime_predicate(prefixes: tuple[str, ...], *, column: str) -> str:
    if (not prefixes or any(
            not prefix or any(character not in
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.+-/"
                for character in prefix)
            for prefix in prefixes)):
        raise ValueError("unsafe or empty CC MIME-prefix policy")
    return " OR ".join(
        f"{column} LIKE '{prefix}%'" for prefix in prefixes)


def _nonempty_digest_predicate(*, column: str) -> str:
    """SQL predicate excluding Common Crawl's zero-byte payload digest."""
    return (
        f"upper(trim(coalesce({column}, ''))) NOT IN ("
        f"'{CC_DIGEST.EMPTY_SHA1_BASE32}', "
        f"'SHA1:{CC_DIGEST.EMPTY_SHA1_BASE32}')")


def resolve_exact_file(
        con, filename: str, exact: list[tuple[str, str]],
        allowed_mime_prefixes: tuple[str, ...] = ("text/html",),
) -> dict[str, dict[str, dict]]:
    """Resolve all exact keys assigned to one physical file in one SQL query."""
    if not exact:
        return {"complete": {}, "any": {}}
    con.execute("CREATE OR REPLACE TEMP TABLE aw_cc_exact(k VARCHAR, requested VARCHAR)")
    con.executemany("INSERT INTO aw_cc_exact VALUES (?, ?)", exact)
    escaped = filename.replace("'", "''")
    mime_predicate = _mime_predicate(
        allowed_mime_prefixes, column="p.content_mime_type")
    digest_predicate = _nonempty_digest_predicate(column="p.content_digest")
    rows = con.execute(f"""
        WITH hits AS (
          SELECT w.requested, p.url, p.warc_filename,
                 p.warc_record_offset, p.warc_record_length,
                 p.content_digest, p.fetch_time, p.content_truncated,
                 row_number() OVER (
                   PARTITION BY p.url_surtkey ORDER BY p.fetch_time DESC,
                     p.warc_record_length DESC, p.warc_filename ASC,
                     p.warc_record_offset ASC) AS rn_any,
                 row_number() OVER (
                   PARTITION BY p.url_surtkey
                   ORDER BY CASE WHEN p.content_truncated IS NULL OR
                                      trim(p.content_truncated) = '' THEN 0 ELSE 1 END,
                     p.fetch_time DESC, p.warc_record_length DESC,
                     p.warc_filename ASC, p.warc_record_offset ASC) AS rn_complete
          FROM read_parquet('{escaped}') p
          JOIN aw_cc_exact w ON p.url_surtkey = w.k
          WHERE p.fetch_status = 200
            AND ({mime_predicate})
            AND ({digest_predicate})
        )
        SELECT requested, url, warc_filename, warc_record_offset,
               warc_record_length, content_digest, fetch_time,
               content_truncated, rn_any, rn_complete
        FROM hits WHERE rn_any = 1 OR rn_complete = 1
    """).fetchall()
    out = {"complete": {}, "any": {}}
    for requested, url, wfn, off, length, digest, fetched, truncated, rn_any, rn_complete in rows:
        record = _record(url or requested, wfn, off, length, digest, fetched, truncated)
        if rn_any == 1:
            out["any"][requested] = record
        if rn_complete == 1 and not record["content_truncated"]:
            out["complete"][requested] = record
    return out


def resolve_prefix_file(con, filename: str,
                        prefixes: list[tuple[str, str]],
                        per_domain: int = 100) -> dict[str, dict[str, list[dict]]]:
    """Resolve shallow complete/any inventories for one physical file/query."""
    if not prefixes:
        return {"complete": {}, "any": {}}
    if per_domain <= 0:
        raise ValueError("per_domain must be positive")
    con.execute("CREATE OR REPLACE TEMP TABLE aw_cc_prefix("
                "p VARCHAR, domain VARCHAR, lo VARCHAR, hi VARCHAR)")
    con.executemany(
        "INSERT INTO aw_cc_prefix VALUES (?, ?, ?, ?)",
        [(prefix, domain, prefix + "/", prefix + "/￿")
         for prefix, domain in prefixes])
    escaped = filename.replace("'", "''")
    digest_predicate = _nonempty_digest_predicate(column="p.content_digest")
    rows = con.execute(f"""
        WITH hits AS (
          SELECT w.p, w.domain, p.url_surtkey, p.url, p.url_path,
                 p.warc_filename, p.warc_record_offset, p.warc_record_length,
                 p.content_digest, p.fetch_time, p.content_truncated,
                 row_number() OVER (
                   PARTITION BY w.p, p.url_surtkey ORDER BY p.fetch_time DESC,
                     p.warc_record_length DESC, p.warc_filename ASC,
                     p.warc_record_offset ASC) AS rn_any,
                 row_number() OVER (
                   PARTITION BY w.p, p.url_surtkey
                   ORDER BY CASE WHEN p.content_truncated IS NULL OR
                                      trim(p.content_truncated) = '' THEN 0 ELSE 1 END,
                     p.fetch_time DESC, p.warc_record_length DESC,
                     p.warc_filename ASC, p.warc_record_offset ASC) AS rn_complete
          FROM read_parquet('{escaped}') p
          JOIN aw_cc_prefix w ON p.url_surtkey >= w.lo AND p.url_surtkey < w.hi
          WHERE p.fetch_status = 200
            AND p.content_mime_type LIKE 'text/html%'
            AND ({digest_predicate})
        ), candidates AS (
          SELECT *, 'any' AS kind FROM hits WHERE rn_any = 1
          UNION ALL
          SELECT *, 'complete' AS kind FROM hits
          WHERE rn_complete = 1 AND
                (content_truncated IS NULL OR trim(content_truncated) = '')
        ), ranked AS (
          SELECT *, row_number() OVER (
            PARTITION BY p, kind
            ORDER BY length(url_path), length(url), url) AS shallow_rank
          FROM candidates
        )
        SELECT domain, kind, url, warc_filename, warc_record_offset,
               warc_record_length, content_digest, fetch_time, content_truncated
        FROM ranked WHERE shallow_rank <= {int(per_domain)}
        ORDER BY domain, kind, shallow_rank
    """).fetchall()
    out: dict[str, dict[str, list[dict]]] = {
        "complete": {}, "any": {}}
    for domain, kind, url, wfn, off, length, digest, fetched, truncated in rows:
        out[kind].setdefault(domain, []).append(
            _record(url, wfn, off, length, digest, fetched, truncated))
    return out


def _con(threads: int = 1) -> duckdb.DuckDBPyConnection:
    """DuckDB connection tuned for GENTLE access to the CC HTTPS mirror: single
    HTTP thread (no 300-file concurrent blast — that 403s the mirror), metadata
    cache + keep-alive on, modest retries. The CC mirror bulk-throttles bursts.

    MEMORY: an in-memory DuckDB defaults to ~80% of RAM with NO temp spill —
    on a small box a large resolve chunk gets the whole process OOM-killed
    (observed 2026-07-03 on the orchestrator box, 7.6GB RAM: 200-domain discovery chunk →
    SIGKILL at ~3GB RSS). Cap it (AW_DUCKDB_MEM, default 2GB) and give it a
    temp_directory so big scans spill to disk instead of dying."""
    import tempfile
    import duckdb
    con = duckdb.connect()
    # Detached/systemd services may intentionally have no HOME. DuckDB 1.5+
    # still needs an explicit home directory before INSTALL/LOAD can resolve
    # its extension cache. Keep this task-scoped and independently overrideable
    # instead of manufacturing a process HOME (which would also blur AWS
    # credential provenance).
    duckdb_home = os.path.abspath(os.path.expanduser(os.environ.get(
        "AW_DUCKDB_HOME", os.path.join(tempfile.gettempdir(), "aw-duckdb-home"))))
    os.makedirs(duckdb_home, exist_ok=True)
    escaped_home = duckdb_home.replace("'", "''")
    con.execute(f"SET home_directory='{escaped_home}';")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    if use_s3():
        # Signed-S3 read path: region + the botocore credential chain (httpfs
        # reads ~/.aws itself via the secret — no key ever touches our code/logs).
        con.execute(f"SET s3_region='{S3_REGION}';")
        try:
            con.execute("CREATE SECRET IF NOT EXISTS aw_cc_s3 "
                        "(TYPE s3, PROVIDER credential_chain);")
        except Exception:  # noqa: BLE001 — older duckdb lacks CREATE SECRET
            try:                                     # fall back: resolve chain → SET
                import boto3
                cr = boto3.Session().get_credentials().get_frozen_credentials()
                con.execute(f"SET s3_access_key_id='{cr.access_key}';")
                con.execute(f"SET s3_secret_access_key='{cr.secret_key}';")
                if cr.token:
                    con.execute(f"SET s3_session_token='{cr.token}';")
            except Exception:  # noqa: BLE001
                pass
    con.execute("SET enable_http_metadata_cache=true;")
    con.execute("SET http_keep_alive=true;")
    con.execute(f"SET threads={max(1, threads)};")   # serialize HTTP reads
    mem = os.environ.get("AW_DUCKDB_MEM", "2GB")
    spill = os.path.join(tempfile.gettempdir(), "aw-duckdb-spill")
    os.makedirs(spill, exist_ok=True)
    con.execute(f"SET memory_limit='{mem}';")
    con.execute(f"SET temp_directory='{spill}';")
    con.execute("SET preserve_insertion_order=false;")  # lower-memory large scans
    try:
        con.execute("SET http_retries=5;")
        con.execute("SET http_retry_wait_ms=2000;")
        con.execute("SET http_timeout=60000;")
    except Exception:  # noqa: BLE001 — older duckdb may lack some knobs
        pass
    return con


def resolve(crawl_id: str, urls: list[str], con=None,
            require_complete: bool = False,
            allowed_mime_prefixes: tuple[str, ...] = ("text/html",)) -> dict[str, dict]:
    """Resolve a batch of exact URLs to their status-200 HTML WARC records in
    `crawl_id` via one columnar query. Returns {url: {warc_filename, offset,
    length, digest, fetch_time, content_truncated, truncation_reason}}. URLs
    with no capture in this crawl are omitted. ``require_complete=True``
    excludes rows whose STRING ``content_truncated`` field carries a reason;
    this lets strict L0/card callers select an older complete capture rather
    than reject the newest truncated one. The default preserves existing
    discovery callers' behaviour.

    Robustness: on a query failure we raise — the caller must treat that as a
    transient error for the *whole batch* (re-queue), never as 'no captures'."""
    if not urls:
        return {}
    mime_predicate = _mime_predicate(
        allowed_mime_prefixes, column="p.content_mime_type")
    digest_predicate = _nonempty_digest_predicate(column="p.content_digest")
    own = con is None
    con = con or _con()
    try:
        # map surtkey -> url
        key2url: dict[str, str] = {}
        for u in urls:
            k = url_surtkey(u)
            if k:
                key2url.setdefault(k, u)
        if not key2url:
            return {}
        # bucket keys to the ONE file whose surtkey range contains each (the
        # files are non-overlapping ranges). Only those files get queried → we
        # never scan the whole 300-file index, and never touch the rate-limited
        # CDX host (this is all on the WARC mirror).
        import bisect
        bounds = file_bounds(crawl_id, con)
        mins = [b[1] for b in bounds]
        needed: set[str] = set()
        for k in key2url:
            i = bisect.bisect_right(mins, k) - 1   # rightmost file with min <= k
            if 0 <= i < len(bounds) and bounds[i][1] <= k <= bounds[i][2]:
                needed.add(bounds[i][0])
        if not needed:
            return {}
        # ONE batched query over only the needed files (not all ~300). DuckDB
        # parallelizes the file reads (threads) and skips row groups whose
        # min/max can't contain a wanted key via the semi-join predicate.
        con.execute("CREATE OR REPLACE TEMP TABLE want(k VARCHAR)")
        con.executemany("INSERT INTO want VALUES (?)", [(k,) for k in key2url])
        flist = "['" + "','".join(sorted(needed)) + "']"
        complete = "AND (p.content_truncated IS NULL OR trim(p.content_truncated) = '')" \
            if require_complete else ""
        q = f"""
        SELECT p.url_surtkey, p.url, p.warc_filename,
               p.warc_record_offset, p.warc_record_length, p.content_digest,
               p.fetch_time, p.content_truncated
        FROM read_parquet({flist}) p
        SEMI JOIN want w ON p.url_surtkey = w.k
        WHERE p.fetch_status = 200 AND ({mime_predicate})
              AND ({digest_predicate})
              {complete}
        QUALIFY row_number() OVER (PARTITION BY p.url_surtkey
                                   ORDER BY p.fetch_time DESC,
                                            p.warc_record_length DESC,
                                            p.warc_filename ASC,
                                            p.warc_record_offset ASC) = 1
        """
        out: dict[str, dict] = {}
        for k, url, wfn, off, ln, dig, fetched, truncated in con.execute(q).fetchall():
            u = key2url.get(k)
            if u is not None:
                is_truncated, truncation_reason = _truncation_metadata(truncated)
                out[u] = {"url": url or u, "warc_filename": wfn,
                          "offset": int(off), "length": int(ln), "digest": dig,
                          "fetch_time": _fetch_time_iso(fetched),
                          "content_truncated": is_truncated,
                          "truncation_reason": truncation_reason}
        return out
    finally:
        if own:
            con.close()


# ---------------------------------------------------------------------------
# DOMAIN resolution — the discovery-engine path (validated 2026-06-30).
#
# Why this is fast where arbitrary-URL `resolve()` was "too slow"
# ---------------------------------------------------------------
# The index is sorted by `url_surtkey`, and a SURT key is `tld,domain)/path`.
# So EVERY url of one registered domain shares the contiguous prefix
# `tld,domain)/` — it lands in (almost always) a SINGLE one of the ~300 parquet
# files, and within that file in one/two row-groups. A *domain* query therefore
# prunes to ~1 file + a sub-range; an *arbitrary URL set* (HN) scatters its keys
# across all 300 files, which is what made the URL path scan-heavy. Measured on
# CC-MAIN-2026-25 from the orchestrator box (no AWS): 300-file footer read ~1.7 s (cached
# after), then ~0.3–1.1 s per domain, <6 s for a 5-domain batch touching 4/300
# files. This is the ADR-006/014 "columnar via DuckDB for domain filtering" path.


def domain_surtkey_prefix(domain: str) -> str | None:
    """The `url_surtkey` prefix for a domain OR host, e.g.
    'lib4ri.ch' -> 'ch,lib4ri)', 'estv.admin.ch' -> 'ch,admin,estv)'.

    Granularity (corrected 2026-07-03 — the old docstring claimed subdomains
    were folded in; they are NOT): surt canonicalisation folds only a leading
    `www\\d*.` — every other subdomain gets its own key (`ch,admin,estv)/…`),
    which sorts OUTSIDE the registrable range [`ch,admin)/`, `ch,admin)/￿`]
    because ',' > ')'. So a registrable-domain entry resolves apex+www pages
    only, and reaching e.g. admin.ch agency sites requires per-HOST queue
    entries (see prep_discovery_queue.py host granularity)."""
    if not domain:                             # NULL url_host_registered_domain
        return None                            # rows exist in the cc-index
    k = url_surtkey("http://" + domain.strip().lstrip(".") + "/")
    if not k or ")" not in k:
        return None
    return k.split(")", 1)[0] + ")"            # drop the path, keep `tld,dom)`


def match_prefix(surtkey: str, prefixes_sorted: list[str]) -> str | None:
    """Which of the (sorted, ')'-terminated) wanted prefixes contains this
    surtkey? Prefix ranges are disjoint (no wanted prefix's range nests inside
    another's — `ch,admin,estv)/…` does not start with `ch,admin)/`), so at
    most one matches. Pure helper so resolve_domains' host↔record mapping is
    unit-testable offline."""
    import bisect
    i = bisect.bisect_right(prefixes_sorted, surtkey) - 1
    if 0 <= i < len(prefixes_sorted) and surtkey.startswith(prefixes_sorted[i] + "/"):
        return prefixes_sorted[i]
    return None


class RemoteIndexReadBlocked(RuntimeError):
    """task634 (2026-09-23): raised instead of touching the network when
    AW_CC_REQUIRE_LOCAL_INDEX=1 and this batch needs a parquet file that
    isn't (yet) locally mirrored. discovery_ingest.py's cmd_fetch already
    catches per-chunk resolve exceptions and defers the WHOLE chunk as
    `error` (None -> re-queued, NEVER 'empty') -- this class rides that
    existing, already-correct error path, so a domain whose file simply
    hasn't downloaded yet is retried later, not silently recorded as having
    zero captures. See egress-cap-and-local-mirror task doc: DuckDB's own
    httpfs does its S3 I/O in native code with no Python-level byte hook
    (confirmed empirically -- it ignores HTTP_PROXY/HTTPS_PROXY entirely),
    so the ONLY way to guarantee zero network egress from the resolve phase
    is to refuse the read outright, not meter it."""


def _require_local_index() -> bool:
    return os.environ.get("AW_CC_REQUIRE_LOCAL_INDEX", "") not in ("", "0")


def resolve_domains(crawl_id: str, domains: list[str], con=None,
                    per_domain: int | None = None,
                    require_complete: bool = False) -> dict[str, list[dict]]:
    """Resolve a batch of DOMAINS or HOSTS to their status-200 HTML WARC
    records in `crawl_id`, via surtkey-prefix partition-pruning (one query per
    parquet file actually touched, not all ~300). Entries may be registrable
    domains ('lib4ri.ch' → apex+www pages) or hosts ('estv.admin.ch' → that
    host's pages) — see domain_surtkey_prefix on granularity. Returns
    {domain: [ {url, warc_filename, offset, length, digest, fetch_time,
    content_truncated, truncation_reason}, ... ]} with at most `per_domain`
    shallowest URLs each (None = all). Domains with no capture map to an empty
    list. ``require_complete=True`` excludes captures carrying a non-empty CC
    truncation reason before capture selection. Raises on query failure —
    caller must treat the whole affected batch as a transient error (re-queue),
    never as 'no captures'.

    This is the discovery-engine resolver: it replaces the rate-limited CDX HTTP
    API (`index.commoncrawl.org`) with the unblocked columnar index on the WARC
    mirror, so the 4,811-domain discovery queue resolves without IP-ban risk."""
    if not domains:
        return {}
    own = con is None
    con = con or _con()
    try:
        # prefix -> the canonical domain string the caller passed
        pre2dom: dict[str, str] = {}
        for d in domains:
            p = domain_surtkey_prefix(d)
            if p:
                pre2dom.setdefault(p, d)
        if not pre2dom:
            return {}
        # map each wanted prefix to the parquet file(s) whose surtkey range can
        # contain it, then group prefixes BY FILE so we issue one query per file.
        bounds = file_bounds(crawl_id, con)          # sorted by min, cached
        import collections
        by_file: dict[str, list[str]] = collections.defaultdict(list)
        for p in pre2dom:
            lo, hi = p + "/", p + "/￿"
            for fn, mn, mx in bounds:
                if not (mx < lo or mn > hi):         # range overlap
                    by_file[fn].append(p)
        out: dict[str, list[dict]] = {pre2dom[p]: [] for p in pre2dom}
        # per_domain is a PER-DOMAIN cap, but a per-file SQL `LIMIT` applies
        # across ALL prefixes OR'd into one file-query, so domains that share a
        # parquet file STARVE each other: a domain with many short-path URLs
        # (ge.ch/eda.admin.ch `?page=N`) hogs the whole budget under
        # `ORDER BY length(url_path)`, and its file-mates resolve to ~0
        # (measured 2026-07-19: lausanne.ch 7927 standalone -> 73 in a 5-domain
        # chunk; stadt-zuerich.ch -> 0). This silently under-yields every
        # multi-domain columnar resolve. Fix: NO per-file SQL LIMIT — the Python
        # appender below already enforces the per-domain cap (`len(lst) <
        # per_domain`), which is per-domain-correct. The file-query materialises
        # all rn=1 captures for its domains (bounded by the corpus in that file);
        # keep the resolve batch (columnar-chunk) modest so that stays cheap.
        rank = "ORDER BY length(url_path), length(url)"
        complete = "AND (p.content_truncated IS NULL OR trim(p.content_truncated) = '')" \
            if require_complete else ""
        digest_predicate = _nonempty_digest_predicate(column="p.content_digest")
        if _require_local_index():
            remote_needed = [fn for fn in by_file
                             if not (fn.startswith("/") or fn.startswith("file://"))]
            if remote_needed:
                raise RemoteIndexReadBlocked(
                    f"AW_CC_REQUIRE_LOCAL_INDEX=1: this batch needs "
                    f"{len(remote_needed)} parquet file(s) not yet locally "
                    f"mirrored (e.g. {remote_needed[0]!r}) -- refusing the "
                    f"remote read. Deferred, not empty; retry once the "
                    f"CC index mirror covers "
                    f"{'it' if len(remote_needed) == 1 else 'them'}.")
        for fn, prefixes in by_file.items():
            # one query per file, all that file's wanted prefixes OR'd together,
            # De-dup to the newest capture per url_surtkey, then the largest at
            # an identical fetch time (stable WARC/offset tie-breakers below).
            ors = " OR ".join(
                f"(p.url_surtkey >= '{pp}/' AND p.url_surtkey < '{pp}/￿')"
                for pp in prefixes)
            q = f"""
            WITH hits AS (
              SELECT p.url_surtkey, p.url,
                     p.warc_filename, p.warc_record_offset, p.warc_record_length,
                     p.content_digest, p.fetch_time, p.content_truncated,
                     p.url_path,
                     row_number() OVER (PARTITION BY p.url_surtkey
                                ORDER BY p.fetch_time DESC,
                                         p.warc_record_length DESC,
                                         p.warc_filename ASC,
                                         p.warc_record_offset ASC) AS rn
              FROM read_parquet('{fn}') p
              WHERE ({ors})
                AND p.fetch_status = 200
                AND p.content_mime_type LIKE 'text/html%'
                AND ({digest_predicate})
                {complete}
            )
            SELECT url_surtkey, url, warc_filename, warc_record_offset,
                   warc_record_length, content_digest, fetch_time,
                   content_truncated, url_path
            FROM hits WHERE rn = 1 {rank}"""
            pref_sorted = sorted(prefixes)
            for (skey, url, wfn, off, ln, dig, fetched, truncated,
                 _path) in con.execute(q).fetchall():
                # Map the record back to the caller's queue key via ITS OWN
                # surtkey against the wanted prefixes. (The old code keyed on
                # url_host_registered_domain, which mis-filed HOST-level queue
                # entries: estv.admin.ch records came back keyed 'admin.ch'
                # and the host looked falsely empty — fixed 2026-07-03.)
                pre = match_prefix(skey, pref_sorted)
                if pre is None:               # can't happen given the ORs; be safe
                    continue
                key = pre2dom[pre]
                lst = out.setdefault(key, [])
                if per_domain is None or len(lst) < per_domain:
                    is_truncated, truncation_reason = _truncation_metadata(truncated)
                    lst.append({"url": url, "warc_filename": wfn,
                                "offset": int(off), "length": int(ln),
                                "digest": dig, "fetch_time": _fetch_time_iso(fetched),
                                "content_truncated": is_truncated,
                                "truncation_reason": truncation_reason})
        return out
    finally:
        if own:
            con.close()
