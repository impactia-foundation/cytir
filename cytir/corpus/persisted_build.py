"""cytir.corpus.persisted_build -- `cyt ops build-persisted-index`: the ONE-TIME, out-of-band bulk build of the sqlite persisted-input index
(task 664, 2026-09-27 incident).

Run this ONCE (or whenever a genuine from-scratch rebuild is wanted --
never automatically, and never as part of a drain) before `persisted_
inputs.update_index_sqlite()`'s incremental top-up can do anything: that
function refuses fast if the db does not exist yet, precisely so the
first ~88.8M-record population is never attempted inline inside a
drain's timeout again.

THE INCIDENT: every drain since 2026-09-26 17:31 CEST attempted that
first population inline, inside `update_index_sqlite`'s per-file loop
(fine for the steady-state "a handful of files changed" case, the wrong
tool at this scale), and was killed at systemd's 6h start timeout still
IO-bound (process state D, ~13min actual CPU burned across 6h wall) --
restarting from zero every single time, landing nothing, for 315+
waiting packs.

BULK-LOAD DISCIPLINE (why this script is fast where the inline path was
not):
  * `synchronous=OFF` + `journal_mode=MEMORY` for the load (never for the
    incremental path, which stays WAL -- a crash mid-build only strands
    this script's TEMP file, never the live db).
  * ONE (or a few, batched every `--batch-rows`) `executemany()` calls,
    not one Python-level `execute()` per file -- 157k tiny transactions
    was most of the wall-clock cost.
  * The unique `(layer, key)` index is built AFTER the bulk load, via one
    in-db `INSERT OR REPLACE ... SELECT ... ORDER BY rowid` (last file
    wins on a duplicate key, the SAME semantics the incremental updater's
    own `INSERT OR REPLACE` already uses) -- never maintained row-by-row
    during the load.
  * The finished db is built at a TEMP path and atomically `os.replace()`d
    onto the real one: a killed build leaves the previous db (or none)
    untouched, never a half-populated "live" one a query could read mid-
    build.
  * The db's DIRECTORY is `persisted_inputs.sqlite_index_dir()` --
    `AW_PERSISTED_INDEX_SQLITE_DIR` when set, which MUST point at NVMe
    (fast local NVMe), not the HDD `corpus_root()` normally resolves to.
    An IO-bound 88.8M-record load against spinning disk is the other half
    of the incident this script exists to close.

Moved from the pipeline tree in task 1234, behaviour unchanged.

Deploy: `nice -n15 ionice -c3 cyt ops build-persisted-index --index-dir DIR`, detached (`nohup ... & disown` or a
systemd oneshot with no `TimeoutStartSec`), never inside a drain's own
process or timeout budget.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import time
from typing import Optional

import cytir.corpus.persisted as PI


def _log(*a):
    print(*a, flush=True)


def build(index_dir: str, *, registry_path: Optional[str] = None,
         root: Optional[str] = None, batch_rows: int = 200_000,
         log=_log) -> dict:
    t0 = time.time()
    final_path = PI.sqlite_index_path(index_dir)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    tmp_path = f"{final_path}.building-{os.getpid()}"
    for p in (tmp_path, tmp_path + "-wal", tmp_path + "-shm"):
        if os.path.exists(p):
            os.remove(p)

    conn = sqlite3.connect(tmp_path)
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-500000")  # ~500MB page cache
    # NO unique constraint yet, and named `_raw` -- a plain heap table
    # during the load, kept separate from the real (constrained) table
    # names `_SQLITE_DDL` creates below, so the later `INSERT OR REPLACE
    # INTO pages SELECT ... FROM pages_raw` reads one table and writes a
    # DIFFERENT one, never itself.
    conn.executescript("""
        CREATE TABLE pages_raw (layer TEXT, key INTEGER, store_id TEXT, file_path TEXT);
        CREATE TABLE shas_raw (layer TEXT, key INTEGER, file_path TEXT);
        CREATE TABLE scanned_files_raw (layer TEXT, path TEXT, size INTEGER,
                                        mtime_ns INTEGER, rows INTEGER, err TEXT);
    """)

    registry = PI.load_registry(registry_path)
    croot = PI.corpus_root(root)
    files_scanned = rows_indexed = 0
    page_buf: list[tuple] = []
    sha_buf: list[tuple] = []
    scanned_buf: list[tuple] = []

    def flush() -> None:
        nonlocal page_buf, sha_buf, scanned_buf
        if page_buf:
            conn.executemany("INSERT INTO pages_raw VALUES (?,?,?,?)", page_buf)
        if sha_buf:
            conn.executemany("INSERT INTO shas_raw VALUES (?,?,?)", sha_buf)
        if scanned_buf:
            conn.executemany(
                "INSERT INTO scanned_files_raw VALUES (?,?,?,?,?,?)", scanned_buf)
        conn.commit()
        page_buf, sha_buf, scanned_buf = [], [], []

    for store in PI.stores(registry):
        layer = store.layer
        for path in store.files(croot):
            files_scanned += 1
            try:
                st = os.stat(path)
            except OSError:
                continue
            rows = 0
            err = None
            try:
                for url, sha in PI._iter_store_file(store, path):
                    rows += 1
                    rows_indexed += 1
                    page_buf.append(
                        (layer, PI._to_signed64(PI.page_key(url)),
                        store.store_id, path))
                    if sha:
                        sha_buf.append(
                            (layer, PI._to_signed64(PI.pair_key(url, sha)),
                            path))
                    # Task 664, 2026-09-28: this check MUST be inside the
                    # per-row loop, not after it -- one huge file (the
                    # first live run hit a 1.9M-row file) otherwise
                    # buffers its ENTIRE row set in memory before ever
                    # flushing, defeating the whole point of batching by
                    # `--batch-rows`, not `--batch-files`.
                    if len(page_buf) >= batch_rows:
                        flush()
                        log(f"[bulk-build] {files_scanned} file(s), "
                           f"{rows_indexed:,} row(s) so far "
                           f"({round(time.time() - t0, 1)}s)...")
            except Exception as exc:  # noqa: BLE001 -- a hole in the
                # instrument, not an empty store (same bias as
                # persisted_inputs.py's own scan task: refuse, never
                # silently treat an unreadable file as "page never
                # stored").
                err = f"{type(exc).__name__}: {exc}"[:200]
            scanned_buf.append((layer, path, st.st_size, st.st_mtime_ns,
                               rows, err))
    flush()

    log(f"[bulk-build] load done: {files_scanned} file(s), "
       f"{rows_indexed:,} row(s) ({round(time.time() - t0, 1)}s) -- "
       "deduping + indexing...")

    # The real indexed shape (`pages`/`shas`/`scanned_files`/`meta`, the
    # SAME schema `update_index_sqlite`'s incremental path reads and
    # writes -- `_SQLITE_DDL` is the ONE declaration, never re-derived
    # here), built in ONE in-db statement per table from the raw load --
    # never pulling rows back into Python. `INSERT OR REPLACE ... ORDER BY
    # rowid` gives "last file wins" on a duplicate (layer, key), the SAME
    # semantics `update_index_sqlite`'s own INSERT OR REPLACE already uses.
    conn.executescript(PI._SQLITE_DDL)
    conn.execute(
        "INSERT OR REPLACE INTO pages (layer, key, store_id, file_path) "
        "SELECT layer, key, store_id, file_path FROM pages_raw ORDER BY rowid")
    conn.execute(
        "INSERT OR REPLACE INTO shas (layer, key, file_path) "
        "SELECT layer, key, file_path FROM shas_raw ORDER BY rowid")
    conn.execute(
        "INSERT OR REPLACE INTO scanned_files "
        "(layer, path, size, mtime_ns, rows, err) "
        "SELECT layer, path, size, mtime_ns, rows, err "
        "FROM scanned_files_raw ORDER BY rowid")
    conn.commit()
    log("[bulk-build] indexed. dropping the raw load tables + finalizing "
       "meta...")
    conn.executescript("""
        DROP TABLE pages_raw;
        DROP TABLE shas_raw;
        DROP TABLE scanned_files_raw;
    """)
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('built_at', ?)",
        (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),))
    unreadable_total = conn.execute(
        "SELECT COUNT(*) FROM scanned_files WHERE err IS NOT NULL"
    ).fetchone()[0]
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES "
        "('unreadable_files', ?)", (str(unreadable_total),))
    conn.execute("PRAGMA journal_mode=WAL")  # the LIVE mode from here on
    conn.commit()
    conn.execute("VACUUM")  # reclaim the raw tables' space before this
                            # becomes the live db, not after
    conn.close()

    os.replace(tmp_path, final_path)
    for suffix in ("-wal", "-shm"):
        stale = f"{final_path}{suffix}"
        if os.path.exists(stale):
            os.remove(stale)

    seconds = round(time.time() - t0, 1)
    log(f"[bulk-build] DONE: {files_scanned} file(s), {rows_indexed:,} "
       f"row(s), {unreadable_total} unreadable ({seconds}s) -> {final_path}")
    return {"files_scanned": files_scanned, "rows_indexed": rows_indexed,
           "unreadable_files": unreadable_total, "seconds": seconds,
           "index_path": final_path}


def build_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.strip().splitlines()[0])
    ap.add_argument("--index-dir", required=True,
                    help="the index directory (persisted_inputs.default_"
                        "index_dir() for the live index) -- the db file "
                        "itself lands under AW_PERSISTED_INDEX_SQLITE_DIR "
                        "when set, else here")
    ap.add_argument("--registry-path", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--batch-rows", type=int, default=200_000)
    return ap


def main(argv=None, prog: Optional[str] = None) -> int:
    args = build_parser(prog).parse_args(argv)
    build(args.index_dir, registry_path=args.registry_path, root=args.root,
         batch_rows=args.batch_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
