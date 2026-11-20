#!/usr/bin/env python3
# Moved from the retired `l0_store` module (task 683 S7 tranche 4); the old path is an alias shim.
"""L0 raw-HTML content-addressed archive (task 312; specs/architecture.md §4.2).

WHY (architecture.md §4.2, Tim 2026-07-24): the epochs DAG (specs/epochs.md)
makes an `extract` rule bump a full re-fetch of the web *unless the raw bytes
are already on disk*. With L0 retained for every new fetch, extract@N+1,
outlinks backfill, and future extraction needs become disk reads instead of
re-crawls. Cost is trivial (~200-300 GB compressed for the full page universe
vs 6.5 TB free on the corpus HDD).

This module is the ONE shared, content-addressed L0 sink every NEW ingest
chokepoint writes through (task 312 wires it into sitemap_ingest.py,
discovery_ingest.py, and cc_ingest.py — the fetch/extract funnel's three call
sites). It intentionally does NOT touch or replace the existing ad hoc L0
stores already in the repo:

  - `pack_fetch.L0Store` — per-pack gzip-member-append jsonl shards
    (`<HDD>/aw-corpus/l0/wave=<pack>/shard-*.jsonl.gz`, base64 body). Owned by
    `pack_fetch.run_pack`; untouched by this module.
  - `sitemap_ingest.py --raw-dir` — per-host `<host>.raw.jsonl.gz` /
    `<host>.ccpointers.jsonl` files. Still written by sitemap_ingest
    unconditionally; this module's write() calls are ADDITIVE alongside it
    (see the wiring in sitemap_ingest.py's fetch loop).

Both predate the content-addressed design this module implements and are out
of scope for task 312 to migrate — they keep working exactly as before.

Layout
------
    <root>/<shard>/<sha256-of-bytes>.html.zst   the zstd-compressed raw bytes
    <root>/<shard>/index.jsonl                  one line per write: {url,
                                                 sha256, fetched_at, path, size}
    <root>/<shard>/.seen/<key>                  empty idempotence markers

`shard` = `sha256(url)[:2]` — a 256-way fanout keyed on the URL (not content):
every fetch of one URL, across every digest it has ever had, lands in the
SAME shard. That makes the digest-less branch of `read()` (and any manual
"what did we archive for this url" audit) a single-shard scan instead of a
full-root one — content-hash sharding would scatter one URL's history across
many shards since a page's digest changes when it changes. The actual blob
FILENAME is still the content sha256, so bytes are content-addressed
regardless of which shard directory holds them (two different urls with
byte-identical bodies still share one file within whichever shard each url
mapped to — one copy per shard the content was written into, not one global
copy; a global-dedup layout is a possible future refinement, not needed at
today's scale per the architecture.md §4.2 cost estimate).

Default root: `AW_L0_ROOT` env, else `<hdd-root>/aw-corpus-l0`
(the build box corpus HDD — a SIBLING of `aw-corpus/`, deliberately not nested
under it, so this new content-addressed layout never collides with the
pre-existing `aw-corpus/l0/` pack_fetch.L0Store path).

Idempotent: `write(url, fetched_at, raw_bytes)` is a no-op (no duplicate bytes,
no duplicate index row) when the same (url, sha256-of-raw_bytes) pair has
already been written — safe to call on EVERY fetch, including refetches of an
unchanged page, with no dedup pass required upstream.

Env gate: `AW_L0=1` (default ON) — `l0_enabled()` is the single flag every
wired chokepoint checks before calling write(); flip to `0` process-wide (a
disk-pressure incident, a test run) with no code change.

CLI (spot checks / manual repair, mirrors corpus_store.py's CLI shape):
  python l0_store.py stat  [--root ROOT]
  python l0_store.py read --url URL [--digest SHA256] [--root ROOT] [--out FILE]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

from cytir.core.paths import operator_settings

_SUFFIX = ".html.zst"
_ZSTD_LEVEL = int(os.environ.get("AW_L0_ZSTD_LEVEL", "9"))


def l0_enabled() -> bool:
    """AW_L0=1 default ON (task 312)."""
    return os.environ.get("AW_L0", "1") not in ("0", "false", "False", "")


def default_root() -> str:
    """The L0 store root: operator setting `l0_store_root` (env `AW_L0_STORE_ROOT`; the older
    `AW_L0_ROOT` is an accepted alias), refused by name when unconfigured."""
    return os.path.expanduser(operator_settings().l0_store_root)


def __getattr__(name):
    # `DEFAULT_ROOT` stays readable but resolves on first ACCESS, never at import.
    if name == "DEFAULT_ROOT":
        return operator_settings().l0_store_root
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- zstd backend: python `zstandard` lib if installed, else the `zstd` CLI -
# Lazy + graceful, matching the codebase's convention for optional heavy deps
# (trafilatura/fasttext lazy-imports elsewhere in aw_common.py/fetch_process.py)
# so importing this module never hard-requires either at import time.
_ZSTD = {"tried": False, "lib": None, "cli": None}


def _zstd_backend() -> dict:
    if not _ZSTD["tried"]:
        _ZSTD["tried"] = True
        try:
            import zstandard  # noqa: F401
            _ZSTD["lib"] = zstandard
        except Exception:  # noqa: BLE001
            _ZSTD["cli"] = shutil.which("zstd")
    return _ZSTD


class ZstdUnavailable(RuntimeError):
    """Neither the `zstandard` python package nor a `zstd` CLI binary is
    available. Install one: `pip install zstandard` (the build box's venvs) or
    a system `zstd` package (already present on the orchestrator box via linuxbrew)."""


def _compress(raw: bytes) -> bytes:
    b = _zstd_backend()
    if b["lib"] is not None:
        return b["lib"].ZstdCompressor(level=_ZSTD_LEVEL).compress(raw)
    if b["cli"]:
        p = subprocess.run([b["cli"], f"-{_ZSTD_LEVEL}", "-c"], input=raw,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           check=True)
        return p.stdout
    raise ZstdUnavailable(str(ZstdUnavailable.__doc__))


def _decompress(blob: bytes) -> bytes:
    b = _zstd_backend()
    if b["lib"] is not None:
        # stream_reader (not the single-shot .decompress()) — a blob written
        # by the `zstd` CLI via stdin (the subprocess fallback in _compress)
        # doesn't embed a content-size header (piped input has unknown
        # length), which makes ZstdDecompressor().decompress() raise "could
        # not determine content size in frame header". Streaming works
        # either way, so one code path reads blobs from EITHER backend.
        import io
        with b["lib"].ZstdDecompressor().stream_reader(io.BytesIO(blob)) as r:
            return r.read()
    if b["cli"]:
        p = subprocess.run([b["cli"], "-d", "-c"], input=blob,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           check=True)
        return p.stdout
    raise ZstdUnavailable(str(ZstdUnavailable.__doc__))


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _shard_for_url(url: str) -> str:
    return hashlib.sha256((url or "").encode("utf-8")).hexdigest()[:2]


def _seen_key(url: str, digest: str) -> str:
    return hashlib.sha256(f"{url}\n{digest}".encode("utf-8")).hexdigest()


def write(url: str, fetched_at: str, raw_bytes: bytes, *,
         root: str | None = None, extra: dict | None = None) -> dict:
    """Content-addressed write. Idempotent on (url, sha256(raw_bytes)): a
    second write of the same page's same bytes is a no-op (returns the
    existing record with `deduped: True`) — no duplicate compressed blob on
    disk, no duplicate index.jsonl row.

    Returns `{url, sha256, fetched_at, path, size, deduped}` — `path` is
    ROOT-RELATIVE (portable across a root move/rsync). `fetched_at` is stored
    as GIVEN on a fresh write (first-seen timestamp is what matters for an
    archival record); a dedup hit returns the CURRENT call's fetched_at in the
    return value for caller convenience, but does not rewrite the index.

    `extra`: optional dict of additional fields merged into the index.jsonl
    row only (not the return value's core keys) — e.g. {"host": ..., "wave": ...}
    for a caller that wants richer forensics in the shard index.
    """
    root = os.path.expanduser(root or default_root())
    digest = hashlib.sha256(raw_bytes).hexdigest()
    shard = _shard_for_url(url)
    shard_dir = os.path.join(root, shard)
    rel_path = os.path.join(shard, digest + _SUFFIX)
    abs_path = os.path.join(root, rel_path)
    rec = {"url": url, "sha256": digest, "fetched_at": fetched_at,
          "path": rel_path, "size": len(raw_bytes)}

    marker = os.path.join(shard_dir, ".seen", _seen_key(url, digest))
    if os.path.exists(marker):
        return {**rec, "deduped": True}

    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(os.path.join(shard_dir, ".seen"), exist_ok=True)

    if not os.path.exists(abs_path):
        blob = _compress(raw_bytes)
        tmp = f"{abs_path}.tmp.{os.getpid()}.{time.time_ns()}"
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, abs_path)          # atomic within the shard dir

    row = dict(rec)
    if extra:
        row.update(extra)
    with open(os.path.join(shard_dir, "index.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Marker written LAST: a crash between the content write and here just
    # re-runs write() next time, which finds abs_path already present (skips
    # recompression) and appends one more index row — a harmless extra log
    # line, not a correctness break (index.jsonl is an append log, not a set;
    # dedup-on-disk is what the marker + content-file-exists check guarantee).
    open(marker, "w").close()
    return {**rec, "deduped": False}


def _find_by_url_in_shard(url: str, shard_dir: str) -> dict | None:
    """Scan ONE shard's index.jsonl (every write for `url` lives here — see
    module docstring) for the most recent row matching `url`."""
    idx_path = os.path.join(shard_dir, "index.jsonl")
    if not os.path.exists(idx_path):
        return None
    best = None
    with open(idx_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("url") == url:
                best = row              # last match in file order wins (latest)
    return best


def read(url: str, digest: str | None = None, *,
        root: str | None = None) -> bytes | None:
    """Return the raw (decompressed) bytes for `url`, or None if not found.

    `digest`: when known (e.g. carried alongside `url` in a corpus row), an
    O(1) direct lookup — the fast path for future re-extraction, which is
    exactly what L0 exists for (architecture.md §4.2). Omit `digest` to fall
    back to a scan of just `url`'s shard's index (every write for one url
    lands in the same shard by construction — see module docstring; this
    still touches disk O(1) shards, not O(all shards))."""
    root = os.path.expanduser(root or default_root())
    shard_dir = os.path.join(root, _shard_for_url(url))
    if digest:
        path = os.path.join(shard_dir, digest + _SUFFIX)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return _decompress(f.read())
    row = _find_by_url_in_shard(url, shard_dir)
    if row is None:
        return None
    path = os.path.join(root, row["path"])
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return _decompress(f.read())


# ------------------------------------------------------------------- CLI ---

def _cmd_stat(args):
    root = os.path.expanduser(args.root or default_root())
    if not os.path.isdir(root):
        print(f"L0 root does not exist yet: {root}")
        return
    shards = 0
    blobs = 0
    blob_bytes = 0
    index_rows = 0
    for shard_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(shard_dir) or len(os.path.basename(shard_dir)) != 2:
            continue
        shards += 1
        for p in glob.glob(os.path.join(shard_dir, f"*{_SUFFIX}")):
            blobs += 1
            blob_bytes += os.path.getsize(p)
        idx = os.path.join(shard_dir, "index.jsonl")
        if os.path.exists(idx):
            with open(idx, encoding="utf-8") as f:
                index_rows += sum(1 for _ in f)
    print(f"root={root}")
    print(f"shards={shards} blobs={blobs} blob_bytes={blob_bytes:,} "
         f"({blob_bytes / 1e9:.2f} GB) index_rows={index_rows}")


def _cmd_read(args):
    data = read(args.url, digest=args.digest, root=args.root)
    if data is None:
        print(f"not found: url={args.url} digest={args.digest}", file=sys.stderr)
        sys.exit(1)
    if args.out:
        with open(args.out, "wb") as f:
            f.write(data)
        print(f"wrote {len(data):,} bytes -> {args.out}")
    else:
        sys.stdout.buffer.write(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("stat", help="shard/blob/index-row counts under root")
    st.add_argument("--root", default=None)
    st.set_defaults(func=_cmd_stat)

    rd = sub.add_parser("read", help="fetch raw bytes for a url (+ optional digest)")
    rd.add_argument("--url", required=True)
    rd.add_argument("--digest", default=None)
    rd.add_argument("--root", default=None)
    rd.add_argument("--out", default=None, help="write bytes here (else stdout)")
    rd.set_defaults(func=_cmd_read)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
