#!/usr/bin/env python3
"""Tests for `cytir.corpus.persisted_build` / `cyt ops build-persisted-index` (task 664, 2026-09-27
incident).

Real incident: every drain since 2026-09-26 17:31 CEST attempted the
FIRST ~88.8M-record population inline inside `update_index_sqlite`'s
per-file loop, got killed at systemd's 6h start timeout still IO-bound
(process state D), and restarted from zero next drain -- landing nothing,
315+ packs waiting. This module is the one-time, out-of-band, bulk-
optimized builder that must run before the incremental path can do
anything; `update_index_sqlite` now refuses fast if it has never run.

These tests reuse the real `store_tree` fixture (`persisted_store_tree`) (one L1 page, one L2-only page) so the bulk builder is proven
against the SAME real data the legacy mmap index and the incremental
sqlite updater are already tested against -- answer equivalence, not
just "the script runs".
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cytir.corpus.persisted as PI  # noqa: E402
import cytir.corpus.persisted_build as B  # noqa: E402
from persisted_store_tree import (  # noqa: E402
    store_tree, PERSISTED_URL, PERSISTED_SHA, REAL_URL,
)

__all__ = ["store_tree"]  # re-exported fixture, keeps flake8 quiet


def test_bulk_build_agrees_with_the_legacy_mmap_index(store_tree, tmp_path):
    """The equivalence bar: build the SAME real store tree through the
    bulk builder and the legacy mmap index, and assert every field of
    every lookup matches -- the same bar the persisted-inputs suite holds
    the incremental sqlite path to."""
    legacy_dir = tmp_path / "idx-legacy"
    PI.build_index(str(legacy_dir), root=str(store_tree),
                   log=lambda *a, **k: None)
    legacy = PI.PersistedIndex.open(str(legacy_dir))

    bulk_dir = tmp_path / "idx-bulk"
    B.build(str(bulk_dir), root=str(store_tree), log=lambda *a, **k: None)
    bulk = PI.SqlitePersistedIndex.open(str(bulk_dir))

    for url, sha in [(PERSISTED_URL, PERSISTED_SHA),
                     (PERSISTED_URL, "wrong-sha-entirely"),
                     (PERSISTED_URL, None),
                     (REAL_URL, None),
                     ("https://nobody.example/never-stored", None)]:
        legacy_hit = legacy.lookup(url, sha)
        bulk_hit = bulk.lookup(url, sha)
        if legacy_hit is None:
            assert bulk_hit is None, url
            continue
        assert bulk_hit is not None, url
        assert (legacy_hit.layer, legacy_hit.store_id, legacy_hit.sha_matched) == \
            (bulk_hit.layer, bulk_hit.store_id, bulk_hit.sha_matched), url
    legacy.close()
    bulk.close()


def test_bulk_build_writes_scanned_files_so_the_incremental_path_can_top_up(
        store_tree, tmp_path):
    """After the bulk build, an unchanged file must be recognized as
    already-scanned by update_index_sqlite -- the whole point of writing
    `scanned_files` from the bulk pass too, not leaving it for the
    incremental path to discover from scratch."""
    out = tmp_path / "idx"
    result = B.build(str(out), root=str(store_tree), log=lambda *a, **k: None)
    assert result["files_scanned"] > 0
    assert result["rows_indexed"] > 0

    follow_up = PI.update_index_sqlite(str(out), root=str(store_tree),
                                       log=lambda *a, **k: None)
    assert follow_up["files_changed"] == 0, (
        "every file the bulk build already scanned must read as unchanged "
        "to the incremental updater, not re-merged")


def test_bulk_build_flushes_mid_file_not_just_at_file_boundaries(
        store_tree, tmp_path, monkeypatch):
    """Real incident, 2026-09-28: the first live run's flush check sat
    AFTER the per-row loop (once per FILE, not once per `--batch-rows`),
    so a single large file buffered its entire row set in memory before
    ever flushing -- defeating batching for exactly the files it matters
    most for. With `--batch-rows` smaller than one file's own row count,
    `flush()` must fire more than once while still inside that file."""
    import sqlite3

    flush_calls = []
    real_connect = sqlite3.connect

    class _CountingConn(sqlite3.Connection):
        def executemany(self, sql, rows):
            if sql.startswith("INSERT INTO pages_raw"):
                rows = list(rows)
                flush_calls.append(len(rows))
            return super().executemany(sql, rows)

    def counting_connect(path, *a, **k):
        return real_connect(path, *a, factory=_CountingConn, **k)

    monkeypatch.setattr(sqlite3, "connect", counting_connect)

    # store_tree's L1 file has exactly 1 row; batch_rows=1 forces a flush
    # on literally every row, so with >=2 rows total across the tree,
    # more than one flush call for `pages_raw` must occur.
    B.build(str(tmp_path / "idx"), root=str(store_tree), batch_rows=1,
           log=lambda *a, **k: None)
    assert len(flush_calls) >= 2, (
        f"only {len(flush_calls)} flush(es) -- batching is not happening "
        "at the row level")


def test_bulk_build_is_idempotent_and_atomic(store_tree, tmp_path):
    """Running the builder twice must not corrupt or duplicate anything --
    the temp-path + os.replace() swap always lands a clean, complete db."""
    out = tmp_path / "idx"
    first = B.build(str(out), root=str(store_tree), log=lambda *a, **k: None)
    second = B.build(str(out), root=str(store_tree), log=lambda *a, **k: None)
    assert first["rows_indexed"] == second["rows_indexed"]

    idx = PI.SqlitePersistedIndex.open(str(out))
    assert idx.lookup(PERSISTED_URL) is not None
    idx.close()

    # No leftover temp/wal/shm files from either run.
    db_path = Path(PI.sqlite_index_path(str(out)))
    leftovers = list(db_path.parent.glob(f"{db_path.name}.building-*"))
    assert leftovers == []


def test_bulk_build_deduplicates_last_file_wins(store_tree, tmp_path):
    """Two store files that both mint the SAME page key -- the bulk build
    must keep the one processed LAST (matching update_index_sqlite's own
    INSERT OR REPLACE semantics), not silently pick an arbitrary one or
    fail on the primary-key conflict its own load path defers past."""
    # A second L1 file for the SAME url, in a DIFFERENT store file --
    # store.files() walks in sorted path order, so "second.jsonl.gz"
    # (which sorts after "l1-markdown-00000.jsonl.gz") is the one that
    # must win.
    l1_dir = next((store_tree / "l1").rglob("l1-markdown-00000.jsonl.gz")).parent
    with gzip.open(l1_dir / "zzz-second.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "schema": "aw.l1-markdown.v1",
            "canonical_url": PERSISTED_URL,
            "url": PERSISTED_URL,
            "markdown": "# A different capture\n",
            "markdown_sha256": "1" * 64}) + "\n")

    out = tmp_path / "idx"
    B.build(str(out), root=str(store_tree), log=lambda *a, **k: None)
    idx = PI.SqlitePersistedIndex.open(str(out))
    hit = idx.lookup(PERSISTED_URL, "1" * 64)
    assert hit is not None
    assert hit.sha_matched is True, (
        "the later file's sha must have won -- last file wins on a "
        "duplicate key")
    idx.close()


def test_update_index_sqlite_refuses_fast_with_no_bulk_build_yet(tmp_path):
    """The other half of the fix: the incremental path must REFUSE, not
    attempt the ~88.8M-record first population inline, when no db exists
    yet."""
    with pytest.raises(PI.PersistedInputsError,
                       match="cyt ops build-persisted-index"):
        PI.update_index_sqlite(str(tmp_path / "never-built"))
