#!/usr/bin/env python3
"""Crash-safe gzip shard stores: the ONE home for how AW appends to, and
reads back, `shard-NNNN.jsonl.gz` wave stores (task 428).

WHY THIS MODULE EXISTS (the incident it closes)
-----------------------------------------------
Three L0 wave stores were unreadable for weeks and 22,862 rows were written
off as lost. The mechanism, proven by byte forensics on 2026-08-19 (see
task 424 §"L0 salvage"):

  1. A fetch writer was killed mid-write. gzip members are flushed with
     Z_SYNC_FLUSH per row, so the shard file ended in a TORN member — a
     gzip stream with no terminating CRC32/ISIZE trailer.
  2. A resumed run re-opened that SAME file with `open(path, "ab")` and
     appended a brand-new, perfectly complete gzip member behind the tear.
  3. Every reader starts at byte 0, hits the torn member, raises
     `zlib.error`/`EOFError`, and abandons the whole file — including the
     complete member sitting behind the tear.

The torn prefix POISONS good data behind it. 17,744 rows were recovered by
locating the intact members; 5,141 are permanently lost because their shards
were never rewritten. The defect is not "gzip is fragile" — it is that a
resumed run appended into a file a crashed run had left torn.

THE TWO HALVES OF THE FIX
-------------------------
Write side (`create_shard`): a writer NEVER appends into a pre-existing
shard. Each writer instance claims a fresh shard index and creates the file
with `O_EXCL`. The invariant this buys:

    a torn gzip member is always the LAST member of its own file, and no
    complete member is ever behind a tear.

A crashed writer therefore leaves a file that every reader already tolerates
(read-until-tear), or no file at all — never a torn prefix in front of good
data. `O_EXCL` also makes two concurrent writers on one wave dir claim
different shards instead of interleaving members into one file, which the
old `"ab"` path did silently.

Read side (`iter_jsonl_rows`, `iter_gzip_members`): a member-walking reader
that resynchronises on the next gzip member after a tear instead of giving
up on the file. This is what makes the ~28 wave stores ALREADY on disk
readable — the write-side fix prevents new damage, it cannot repair old
bytes. Both `pack_fetch.iter_l0` and `pack_fetch.iter_l1` read through here.

Detector (`scan_file`, `scan_store`): the same member walk, reported as a
schema instead of rows, so any store can be checked for the signature. It
reports `poisoned` — a torn member with further members behind it, i.e. the
exact incident — and `rows_hidden`, the number of rows sitting strictly
behind a tear that a sequential reader can never reach. This is the promoted form of the one-off salvage scan; it lives
here as shared code with a schema, not as a copied snippet.

WHAT THIS MODULE IS NOT
-----------------------
It is not a merge with `l0_store.py`, and deliberately so. The two have
genuinely different contracts:

  - `l0_store.py` (task 312) is a CONTENT-ADDRESSED blob store: one file per
    (url, sha256) under a 256-way url fanout, written atomically with
    tmp+`os.replace`. It has no shards, no wave grouping, no append. It is
    already crash-safe by construction and needs no change.
  - `pack_fetch`'s stores are WAVE-SHARDED APPEND LOGS: rows carrying
    fetch metadata (status, content_type, kind, tool, extra) grouped by
    wave, with `_meta.json` wave-complete markers and per-shard sha256.
    Downstream readers (chunking, the task-427 re-chunk lane, wave
    inventory) address data by wave+shard.

Folding one into the other would rewrite the on-disk layout every downstream
reader addresses. The correct shared seam is not "where L0 bytes live" — it
is "how a shard file is opened and read", which is what both pack_fetch
stores got wrong in identical ways. That seam is this module, and both
`L0Store` and `MarkdownStore` use it.

CLI
---
  python gz_shard_store.py scan --dir  WAVE_DIR   [--json]
  python gz_shard_store.py scan --root L0_ROOT    [--json]   # every wave=*
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import gzip
import json
import mmap
import os
import re
import sys
import zlib

SHARD_GLOB = "shard-*.jsonl.gz"
SHARD_FMT = "shard-{:04d}.jsonl.gz"
_SHARD_RE = re.compile(r"shard-(\d+)\.jsonl\.gz$")

# A gzip member header starts with magic 1f 8b then CM=08 (deflate). AW writes
# no other compression method, so this 3-byte prefix is the resync anchor.
GZIP_MAGIC = b"\x1f\x8b\x08"
_CHUNK = 1 << 20


# ------------------------------------------------------------- write side ---

def shard_indices(wave_dir: str) -> list[int]:
    """Existing shard indices in `wave_dir`, ascending."""
    out = []
    for p in glob.glob(os.path.join(wave_dir, SHARD_GLOB)):
        m = _SHARD_RE.search(os.path.basename(p))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def next_shard_index(wave_dir: str) -> int:
    """The index a NEW writer must start at: one past the highest existing
    shard, or 0 in a fresh wave dir.

    This is the write-side fix in one line. The old code resumed AT the
    highest existing index and appended into it, which is what put a
    complete gzip member behind a crashed run's torn one.
    """
    idx = shard_indices(wave_dir)
    return idx[-1] + 1 if idx else 0


def create_shard(wave_dir: str, index: int):
    """Create and open a NEW shard file for writing. Never opens an existing
    file, so a previous run's torn tail can never gain a member behind it.

    Returns `(path, fileobj, index)` — `index` is the index actually claimed,
    which may be higher than requested if another writer holds it (O_EXCL
    loses the race and we advance rather than corrupt a shared file).
    """
    os.makedirs(wave_dir, exist_ok=True)
    while True:
        path = os.path.join(wave_dir, SHARD_FMT.format(index))
        try:
            fh = open(path, "xb")          # O_CREAT|O_EXCL: never an append
        except FileExistsError:
            index += 1
            continue
        return path, fh, index


def next_free_rotation(path: str, suffix: str) -> str:
    """A path that does not exist yet, derived from `path` by inserting a
    rotation counter BEFORE `suffix`.

    For stores whose filename is fixed by contract (sitemap_ingest's
    per-host `<host>.raw.jsonl.gz`, ingest_cc198's `pointers-198.jsonl.gz`)
    a new writer cannot simply pick the next index — but it must still not
    append into a file a crashed run may have left torn. Rotating to
    `<host>.r1.raw.jsonl.gz` keeps every existing `glob("*" + suffix)`
    reader matching, so the fix costs no reader change.
    """
    if not path.endswith(suffix):
        raise ValueError(f"{path!r} does not end with {suffix!r}")
    base = path[:-len(suffix)]
    if not os.path.exists(path):
        return path
    n = 1
    while os.path.exists(f"{base}.r{n}{suffix}"):
        n += 1
    return f"{base}.r{n}{suffix}"


def create_rotated_gzip_text(path: str, *, suffix: str,
                             encoding: str = "utf-8"):
    """Open a NEW gzip text stream, rotating past any existing file (O_EXCL).
    Returns `(stream, actual_path)`. Never appends, so a torn member is
    always last in its own file."""
    while True:
        cand = next_free_rotation(path, suffix)
        try:
            fd = os.open(cand, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue                       # lost the race; try the next one
        return gzip.open(os.fdopen(fd, "wb"), "wt", encoding=encoding), cand


# -------------------------------------------------------------- read side ---

def _decode_member(mm, start: int, limit: int, *, step: int = _CHUNK):
    """Inflate ONE gzip member beginning at byte `start`, feeding `step`-sized
    blocks. Returns `(data, consumed, torn, error)` — `consumed` is the
    member's compressed byte length when it terminated cleanly, else None.

    A zlib error costs us the whole failing block's output, so this is the
    FAST pass only; `_decode_member_precise` re-runs a failing member to
    recover the rows inside that block.
    """
    d = zlib.decompressobj(wbits=31)       # 31 = gzip wrapper, no auto-detect
    out: list[bytes] = []
    fed = 0
    err = None
    try:
        while start + fed < limit and not d.eof:
            piece = mm[start + fed:min(start + fed + step, limit)]
            fed += len(piece)
            out.append(d.decompress(piece))
        if not d.eof:
            out.append(d.flush())
    except zlib.error as e:                # noqa: PERF203 - one member, one try
        err = str(e)
    data = b"".join(out)
    if d.eof and err is None:
        return data, fed - len(d.unused_data), False, None
    return data, None, True, err or "member ends without gzip trailer"


def _decode_member_precise(mm, start: int, limit: int):
    """Inflate a TORN member, keeping every row the writer sync-flushed
    before the cut.

    Technique promoted from the former l0_repair script (task 192, the committed
    salvager this module supersedes): step in small blocks holding a
    `decompressobj.copy()` checkpoint, and when a block raises, rewind to the
    checkpoint and replay it ONE BYTE AT A TIME. Because `L0Store.put` calls
    `flush()` (Z_SYNC_FLUSH) after every row, the rows before the cut are
    individually decodable — feeding byte-wise extracts all of them instead
    of discarding the entire block that happens to contain the tear.
    """
    d = zlib.decompressobj(wbits=31)
    out: list[bytes] = []
    fed = 0
    err = None
    block = 4096
    while start + fed < limit and not d.eof:
        ckpt = d.copy()
        piece = mm[start + fed:min(start + fed + block, limit)]
        try:
            out.append(d.decompress(piece))
        except zlib.error as e:
            err = str(e)
            d = ckpt                       # rewind, then salvage byte-wise
            for off in range(len(piece)):
                try:
                    out.append(d.decompress(piece[off:off + 1]))
                except zlib.error:
                    break
            break
        fed += len(piece)
    data = b"".join(out)
    if d.eof and err is None:
        return data, fed - len(d.unused_data), False, None
    return data, None, True, err or "member ends without gzip trailer"


def _read_member(mm, start: int, limit: int):
    """Fast decode, escalating to the byte-wise salvage path only when the
    member is actually damaged — so a healthy store scans at full speed and
    a damaged one still gives up every recoverable row."""
    data, consumed, torn, err = _decode_member(mm, start, limit)
    if not torn:
        return data, consumed, torn, err
    return _decode_member_precise(mm, start, limit)


def _resync(mm, start: int, limit: int) -> int | None:
    """Find the next plausible gzip member start at or after `start`.

    After a tear we cannot trust byte accounting — the torn member's
    decompressor will happily consume the NEXT member's bytes as garbage
    deflate input — so the only way to find what follows is to scan for the
    member magic. Candidates are validated by actually inflating them:
    the gzip FLG byte must have its reserved bits clear, and the candidate
    must yield at least one byte of output. Magic bytes occurring inside
    compressed data fail both tests in practice.
    """
    pos = start
    while True:
        pos = mm.find(GZIP_MAGIC, pos, limit)
        if pos < 0:
            return None
        flg = mm[pos + 3] if pos + 3 < limit else 0xFF
        if not (flg & 0xE0):               # reserved bits 5..7 must be zero
            # Cheap validation: inflate at most a 64 KiB window. A real member
            # yields bytes immediately; magic bytes that merely occur inside
            # compressed data do not.
            probe = min(limit, pos + (1 << 16))
            data, _, _, _ = _decode_member(mm, pos, probe, step=4096)
            if data:
                return pos
        pos += 1


def iter_gzip_members(path: str):
    """Walk a (possibly multi-member, possibly torn) gzip file.

    Yields one dict per member:
      `{index, offset, compressed_bytes, data, torn, error}`
    Resynchronises past a torn member onto the next real member, which is
    exactly what a naive `gzip.open()` cannot do and why the incident's
    complete members were invisible for weeks.
    """
    size = os.path.getsize(path)
    if size == 0:
        return
    with open(path, "rb") as fh:
        with contextlib.closing(mmap.mmap(fh.fileno(), 0,
                                          access=mmap.ACCESS_READ)) as mm:
            pos = 0
            index = 0
            while pos is not None and pos < size:
                data, consumed, torn, err = _read_member(mm, pos, size)
                yield {"index": index, "offset": pos,
                       "compressed_bytes": consumed, "data": data,
                       "torn": torn, "error": err}
                index += 1
                if torn or not consumed:
                    nxt = _resync(mm, pos + 3, size)
                    if nxt is None or nxt <= pos:
                        return
                    pos = nxt
                else:
                    pos += consumed


def iter_jsonl_rows(path: str):
    """Yield every parseable JSON row from a shard, across all members and
    past any tear. A partial trailing line (the row the writer was mid-way
    through) is dropped; a line that does not parse is skipped rather than
    ending the file.
    """
    for member in iter_gzip_members(path):
        data = member["data"]
        if not data:
            continue
        text = data.decode("utf-8", "replace")
        complete_tail = text.endswith("\n")
        lines = text.split("\n")
        if not complete_tail:
            lines = lines[:-1]             # drop the row cut off by the tear
        for line in lines:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_store_rows(wave_dir: str):
    """Yield rows from every shard of a wave dir, in shard order."""
    for path in sorted(glob.glob(os.path.join(wave_dir, SHARD_GLOB))):
        yield from iter_jsonl_rows(path)


# --------------------------------------------------------------- detector ---

def _naive_rows(path: str) -> int:
    """Rows a plain `gzip.open()` reader sees — i.e. what the pipeline
    believed was in this file before the resilient reader existed. The
    difference against the member walk is the data the incident hid.
    """
    n = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            while True:
                try:
                    line = f.readline()
                except (EOFError, OSError, zlib.error):
                    return n
                if not line or not line.endswith("\n"):
                    return n
                n += 1
    except (EOFError, OSError, zlib.error):
        return n
    return n


def scan_file(path: str) -> dict:
    """Scan ONE shard. Schema:

      path, size_bytes, members, torn_members, complete_members,
      first_member_torn, poisoned, rows_readable, rows_reachable,
      rows_hidden, errors[]

    `poisoned` is the incident signature: a torn member with at least one
    further member behind it, so a sequential reader writes off good data.

    `rows_readable`  — rows the member walker recovers (all members).
    `rows_reachable` — rows up to and including the first tear; an UPPER
                       bound on what a sequential `gzip.open()` reader sees.
    `rows_hidden`    — `readable - reachable`, i.e. rows sitting strictly
                       behind a tear that a sequential reader can NEVER
                       reach. A lower bound on the true write-off.
    """
    members = 0
    torn = 0
    first_torn = False
    poisoned = False
    torn_seen = False
    rows = 0
    naive = 0
    errors: list[str] = []
    for m in iter_gzip_members(path):
        if torn_seen:
            poisoned = True                # a member exists BEHIND a tear
        members += 1
        if m["torn"]:
            torn += 1
            if m["index"] == 0:
                first_torn = True
            if m["error"]:
                errors.append(f"member {m['index']} @{m['offset']}: {m['error']}")
        data = m["data"]
        n = data.decode("utf-8", "replace").count("\n") if data else 0
        rows += n
        # A sequential gzip reader walks complete members in order and stops
        # at the FIRST torn one. Everything up to and including that member
        # is REACHABLE (it may get fewer rows than this — Python's gzip can
        # raise before yielding a small torn member's buffered rows — so this
        # is an upper bound on what it sees, and `rows_hidden` below is a
        # LOWER bound on the true write-off). Derived from the walk rather
        # than re-read from disk: that halves the I/O of a corpus-wide scan.
        if not torn_seen:
            naive += n
        if m["torn"]:
            torn_seen = True
    return {"path": path, "size_bytes": os.path.getsize(path),
            "members": members, "torn_members": torn,
            "complete_members": members - torn,
            "first_member_torn": first_torn, "poisoned": poisoned,
            "rows_readable": rows, "rows_reachable": naive,
            "rows_hidden": max(0, rows - naive), "errors": errors}


def scan_store(wave_dir: str) -> dict:
    """Scan every shard of a wave dir. Same schema, aggregated, plus
    `files` (per-shard scans) and `poisoned_files`.
    """
    files = [scan_file(p) for p in
             sorted(glob.glob(os.path.join(wave_dir, SHARD_GLOB)))]
    return {"dir": wave_dir, "shards": len(files),
            "members": sum(f["members"] for f in files),
            "torn_members": sum(f["torn_members"] for f in files),
            "poisoned_files": sum(1 for f in files if f["poisoned"]),
            "poisoned": any(f["poisoned"] for f in files),
            "rows_readable": sum(f["rows_readable"] for f in files),
            "rows_reachable": sum(f["rows_reachable"] for f in files),
            "rows_hidden": sum(f["rows_hidden"] for f in files),
            "files": files}


# -------------------------------------------------------------------- CLI ---

def _print_store(s: dict, verbose: bool):
    flag = "POISONED" if s["poisoned"] else ("torn" if s["torn_members"]
                                             else "ok")
    print(f"{flag:9s} {os.path.basename(s['dir']):40s} "
          f"shards={s['shards']:4d} members={s['members']:5d} "
          f"torn={s['torn_members']:3d} rows={s['rows_readable']:9d} "
          f"reach={s['rows_reachable']:9d} hidden={s['rows_hidden']:8d}")
    if verbose:
        for f in s["files"]:
            if f["poisoned"] or f["torn_members"]:
                print(f"    {os.path.basename(f['path'])}: members="
                      f"{f['members']} torn={f['torn_members']} "
                      f"first_torn={f['first_member_torn']} "
                      f"hidden={f['rows_hidden']}")
                for e in f["errors"]:
                    print(f"      {e}")


def repair_wave(wave_dir: str) -> None:
    """Salvage a wave dir whose shards hold a trailer-less (torn) gzip member.

    Ported from the former `l0_repair` script (task 192; 1235). Reads every
    shard through the shared member walker, dedupes by (url, sha256), verifies
    each body sha, rewrites clean single-member shards (<=512MB each), keeps
    the originals as `.corrupt` and writes a fresh `_meta.partial.json`.
    """
    import base64
    import datetime
    import hashlib

    rows = []
    bad_lines = 0
    for path in sorted(glob.glob(os.path.join(wave_dir, "shard-*.jsonl.gz"))):
        scan = scan_file(path)
        before = len(rows)
        for row in iter_jsonl_rows(path):
            rows.append(row)
        bad_lines += max(0, scan["rows_readable"] - (len(rows) - before))
        print(f"{path}: {scan['members']} members scanned "
              f"({scan['torn_members']} torn, poisoned={scan['poisoned']}), "
              f"rows so far {len(rows)}, dropped partial lines {bad_lines}")

    # dedup by (url, sha256) keeping first
    seen = set()
    clean = []
    for r in rows:
        k = (r.get("url"), r.get("sha256"))
        if k in seen:
            continue
        seen.add(k)
        clean.append(r)
    print(f"salvaged {len(clean)} unique rows ({len(rows)} raw, "
          f"{bad_lines} partial dropped)")

    # verify each body sha
    badsha = 0
    body_bytes = 0
    for r in clean:
        b = base64.b64decode(r["body_b64"])
        body_bytes += len(b)
        if hashlib.sha256(b).hexdigest() != r["sha256"]:
            badsha += 1
    print(f"sha256 verified: {len(clean)-badsha} ok, {badsha} bad")

    # rewrite as clean shards (<=512MB compressed each, single complete member)
    tmp_paths = []

    def open_shard(i):
        p = os.path.join(wave_dir, f"shard-{i:04d}.jsonl.gz.new")
        return p, open(p, "wb")

    idx = 0
    p, raw_f = open_shard(idx)
    tmp_paths.append(p)
    gzf = gzip.GzipFile(fileobj=raw_f, mode="wb")
    for r in clean:
        if hashlib.sha256(base64.b64decode(r["body_b64"])).hexdigest() != r["sha256"]:
            continue
        gzf.write((json.dumps(r, ensure_ascii=False) + "\n").encode())
        if raw_f.tell() >= 512 * 1024 * 1024:
            gzf.close()
            raw_f.close()
            idx += 1
            p, raw_f = open_shard(idx)
            tmp_paths.append(p)
            gzf = gzip.GzipFile(fileobj=raw_f, mode="wb")
    gzf.close()
    raw_f.close()

    # swap in
    for old in glob.glob(os.path.join(wave_dir, "shard-*.jsonl.gz")):
        os.rename(old, old + ".corrupt")
    for tp in tmp_paths:
        os.rename(tp, tp[:-4])

    files = sorted(os.path.basename(x) for x in
                   glob.glob(os.path.join(wave_dir, "shard-*.jsonl.gz")))
    meta = {"layer": "L0", "wave": os.path.basename(wave_dir).split("=", 1)[1],
            "kind": "direct-fetch", "records": len(clean) - badsha,
            "body_bytes": body_bytes,
            "files": files,
            "bytes": sum(os.path.getsize(os.path.join(wave_dir, f)) for f in files),
            "tool": "l0_repair (task 192 salvage of pack_fetch v1 kill-resume flaw)",
            "complete": False,
            "note": "salvaged after mid-shard trailer-less member; "
                    "budget-stopped crawl => wave partial by design",
            "updated_at": datetime.datetime.utcnow().isoformat() + "Z"}
    tmp = os.path.join(wave_dir, "_meta.partial.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp, os.path.join(wave_dir, "_meta.partial.json"))
    print("rewritten:", files)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scan", help="scan wave dir(s) for the torn/multi-"
                                     "member signature")
    g = sc.add_mutually_exclusive_group(required=True)
    g.add_argument("--dir", help="one wave=<pack> dir")
    g.add_argument("--root", help="an L0/L1 root; scans every wave=* under it")
    sc.add_argument("--json", action="store_true")
    sc.add_argument("--verbose", "-v", action="store_true")
    rp = sub.add_parser("repair", help="salvage a wave dir with a torn gzip "
                                       "member (rewrites clean shards)")
    rp.add_argument("wave_dir")
    args = ap.parse_args()

    if args.cmd == "repair":
        repair_wave(args.wave_dir)
        return

    dirs = ([args.dir] if args.dir else
            sorted(glob.glob(os.path.join(os.path.expanduser(args.root),
                                          "wave=*"))))
    # Streamed, not batched: a corpus-wide scan reads tens of GB off an HDD
    # and can be interrupted. Printing each store as it completes means a
    # killed scan still leaves every result it got to, instead of nothing.
    scans = []
    for d in dirs:
        s = scan_store(d)
        scans.append(s)
        if not args.json:
            _print_store(s, args.verbose)
            sys.stdout.flush()
    if args.json:
        json.dump(scans, sys.stdout, indent=2)
        print()
    else:
        bad = [s for s in scans if s["poisoned"]]
        print(f"\n{len(dirs)} store(s) scanned; {len(bad)} POISONED "
              f"(torn member with data behind it); "
              f"{sum(s['rows_hidden'] for s in scans)} rows sit behind a "
              f"tear, unreachable to a sequential reader.")
    return 1 if any(s["poisoned"] for s in scans) else 0


if __name__ == "__main__":
    sys.exit(main())
