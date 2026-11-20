#!/usr/bin/env python3
"""Offline unit tests for the crash-safe gzip shard layer (task 428).

These tests exist because of a real incident: three L0 wave stores were
unreadable for weeks and 22,862 rows written off, because a killed writer
left a TORN gzip member and a resumed run appended a COMPLETE member behind
it with `open(path, "ab")`. Readers start at byte 0, hit the tear, and
abandon the file — including the good data behind it.

The suite is built around a `legacy_append_writer` fixture that reproduces
the OLD writer byte-for-byte. Every "defect" test asserts the old artifact is
poisoned and that the new reader/detector see through it; every "fix" test
asserts the new writer cannot produce that artifact in the first place.

Run: python3 -m pytest tests/test_aw_gz_shard_store.py
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import zlib

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cytir.acquire.gz_shard_store as gz  # noqa: E402
from cytir.acquire.crawl import L0Store, MarkdownStore, iter_l0, iter_l1  # noqa: E402


# --- helpers ------------------------------------------------------------------

def legacy_append_writer(path: str, rows: list[dict], *, tear: bool = False):
    """The OLD `_open()`, verbatim: `open(path, "ab")` +
    `GzipFile(mode="ab")`. Appending to an existing file starts a NEW gzip
    member behind whatever is already there — the defect. `tear=True`
    truncates the member's tail so it never gets its gzip trailer, which is
    what a killed writer leaves behind.
    """
    before = os.path.getsize(path) if os.path.exists(path) else 0
    raw = open(path, "ab")
    gzf = gzip.GzipFile(fileobj=raw, mode="ab")
    for r in rows:
        gzf.write((json.dumps(r) + "\n").encode("utf-8"))
        gzf.flush()                       # Z_SYNC_FLUSH, as L0Store.put does
    gzf.close()
    raw.close()
    if tear:
        # Drop the trailing bytes of the member we just wrote (its deflate
        # end-block + CRC32 + ISIZE), leaving a member with no trailer.
        size = os.path.getsize(path)
        with open(path, "r+b") as f:
            f.truncate(before + max(1, (size - before) - 12))


def naive_rows(path: str) -> int:
    """What a plain `gzip.open()` reader sees — the pipeline's old view."""
    return gz._naive_rows(path)


def naive_urls(path: str) -> list[str]:
    """The URLs a plain `gzip.open()` reader recovers. Z_SYNC_FLUSH means it
    gets SOME rows before a tear; it can never get anything behind one."""
    out = []
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            while True:
                try:
                    line = f.readline()
                except (EOFError, OSError, zlib.error):
                    return out
                if not line or not line.endswith("\n"):
                    return out
                try:
                    out.append(json.loads(line)["url"])
                except Exception:  # noqa: BLE001
                    return out
    except (EOFError, OSError, zlib.error):
        return out
    return out


def rows_of(n: int, prefix: str = "u") -> list[dict]:
    return [{"url": f"https://x.ch/{prefix}{i}", "n": i, "pad": "y" * 200}
            for i in range(n)]


# --- the defect, reproduced against the OLD behavior ---------------------------

def test_legacy_append_after_tear_poisons_good_data(tmp_path):
    """The incident, in miniature: torn member first, complete member behind
    it. A naive reader sees NOTHING; the member walker sees everything."""
    shard = str(tmp_path / "shard-0000.jsonl.gz")
    legacy_append_writer(shard, rows_of(20, "crashed"), tear=True)
    legacy_append_writer(shard, rows_of(15, "resumed"))   # the fatal append

    scan = gz.scan_file(shard)
    assert scan["members"] == 2
    assert scan["first_member_torn"] is True
    assert scan["poisoned"] is True, "torn member with data behind it"

    # The naive reader can never see PAST the tear: the intact member behind
    # it is invisible, which is precisely how 22,862 rows were written off.
    assert not [u for u in naive_urls(shard) if "resumed" in u]
    assert naive_rows(shard) < scan["rows_readable"]
    # 15 rows sit strictly behind the tear; that is the guaranteed write-off.
    assert scan["rows_hidden"] == 15
    assert scan["rows_reachable"] >= naive_rows(shard), \
        "reachable must bound what a real gzip reader gets"
    # The member walker recovers the complete member AND the sync-flushed
    # rows from the torn one.
    urls = [r["url"] for r in gz.iter_jsonl_rows(shard)]
    assert len([u for u in urls if "resumed" in u]) == 15, "intact member"
    assert len([u for u in urls if "crashed" in u]) >= 19, "pre-tear rows"


def test_detector_reports_clean_store_as_clean(tmp_path):
    """The detector must not cry wolf: a healthy multi-shard store, and a
    store whose ONLY tear is a final member, are both un-poisoned."""
    clean = str(tmp_path / "shard-0000.jsonl.gz")
    legacy_append_writer(clean, rows_of(10))
    assert gz.scan_file(clean)["poisoned"] is False
    assert gz.scan_file(clean)["torn_members"] == 0

    tail_torn = str(tmp_path / "shard-0001.jsonl.gz")
    legacy_append_writer(tail_torn, rows_of(10), tear=True)
    s = gz.scan_file(tail_torn)
    assert s["torn_members"] == 1
    assert s["poisoned"] is False, "a tear at EOF hides nothing behind it"

    store = gz.scan_store(str(tmp_path))
    assert store["shards"] == 2 and store["poisoned"] is False


def test_torn_member_keeps_sync_flushed_rows(tmp_path):
    """The byte-wise salvage path (promoted from l0_repair) must recover
    essentially every row flushed before the cut, not just up to the last
    4 KiB block boundary."""
    shard = str(tmp_path / "shard-0000.jsonl.gz")
    legacy_append_writer(shard, rows_of(200), tear=True)
    got = list(gz.iter_jsonl_rows(shard))
    assert len(got) >= 199, f"recovered only {len(got)}/200 sync-flushed rows"
    assert got[0]["n"] == 0 and got[150]["n"] == 150


# --- the fix, on the write side ------------------------------------------------

def test_new_writer_never_appends_into_an_existing_shard(tmp_path):
    """(b) A resume after a crash must not produce a multi-member file whose
    first member is torn — the new writer starts a fresh shard instead."""
    st = L0Store(str(tmp_path), "w")
    for i in range(10):
        st.put(f"https://x.ch/crashed{i}", f"body-{i}".encode())
    st.persist()                          # progress marker, no clean close
    wdir = tmp_path / "wave=w"
    shard0 = sorted(wdir.glob("shard-*.jsonl.gz"))[0]
    del st                                # the process dies here

    # Simulate the kill: the open member never got its trailer.
    with open(shard0, "r+b") as f:
        f.truncate(os.path.getsize(shard0) - 8)

    st2 = L0Store(str(tmp_path), "w")     # resumed run
    for i in range(5):
        st2.put(f"https://x.ch/resumed{i}", f"tail-{i}".encode())
    st2.close(complete=True)

    shards = sorted(wdir.glob("shard-*.jsonl.gz"))
    assert len(shards) == 2, "resume must claim a NEW shard, not append"
    assert gz.scan_file(str(shard0))["members"] == 1, "no member behind tear"

    store = gz.scan_store(str(wdir))
    assert store["poisoned"] is False, "the incident signature is gone"
    assert store["poisoned_files"] == 0

    # (c) everything the resumed run wrote round-trips, and the crashed run's
    # flushed rows survive too.
    urls = [r["url"] for r in iter_l0(str(wdir))]
    assert [u for u in urls if "resumed" in u] == [
        f"https://x.ch/resumed{i}" for i in range(5)]
    assert len([u for u in urls if "crashed" in u]) >= 9


def test_crash_leaves_no_unreadable_file(tmp_path):
    """(a) A simulated crash at every byte offset of the tail must leave a
    file that still reads — never an unreadable-file state."""
    st = L0Store(str(tmp_path), "c")
    for i in range(30):
        st.put(f"https://x.ch/{i}", f"body-{i}".encode())
    st.persist()
    wdir = tmp_path / "wave=c"
    shard = sorted(wdir.glob("shard-*.jsonl.gz"))[0]
    full = open(shard, "rb").read()

    for cut in range(1, 40):              # 39 distinct kill points
        with open(shard, "wb") as f:
            f.write(full[:len(full) - cut])
        scan = gz.scan_file(str(shard))
        assert scan["poisoned"] is False
        rows = list(iter_l0(str(wdir)))   # must not raise
        assert len(rows) >= 25, f"cut={cut} lost too much: {len(rows)}"


def test_concurrent_writers_claim_separate_shards(tmp_path):
    """O_EXCL side effect worth having: two writers on one wave dir get
    different files instead of interleaving members into one."""
    a = L0Store(str(tmp_path), "p")
    b = L0Store(str(tmp_path), "p")
    a.put("https://x.ch/a", b"aaa")
    b.put("https://x.ch/b", b"bbb")
    a.close(complete=False)
    b.close(complete=False)
    wdir = tmp_path / "wave=p"
    assert len(sorted(wdir.glob("shard-*.jsonl.gz"))) == 2
    assert gz.scan_store(str(wdir))["poisoned"] is False
    assert {r["url"] for r in iter_l0(str(wdir))} == {
        "https://x.ch/a", "https://x.ch/b"}


def test_l1_markdown_store_gets_the_same_fix(tmp_path):
    """MarkdownStore had the identical `_open`; the L1 chunker input is as
    load-bearing as L0, so it must carry the same invariant."""
    st = MarkdownStore(str(tmp_path), "m")
    st.put(url="https://x.ch/1", original_sha256="a" * 64, markdown="# One",
           extraction_mode="precision", rejection=None, chunk_count=1,
           chunk_rejections={}, source_profile="web_html")
    st.persist()
    wdir = tmp_path / "wave=m"
    shard = sorted(wdir.glob("shard-*.jsonl.gz"))[0]
    with open(shard, "r+b") as f:
        f.truncate(os.path.getsize(shard) - 6)
    del st

    st2 = MarkdownStore(str(tmp_path), "m")
    st2.put(url="https://x.ch/2", original_sha256="b" * 64, markdown="# Two",
            extraction_mode="precision", rejection=None, chunk_count=1,
            chunk_rejections={}, source_profile="web_html")
    st2.close(complete=True)

    assert len(sorted(wdir.glob("shard-*.jsonl.gz"))) == 2
    assert gz.scan_store(str(wdir))["poisoned"] is False
    assert "https://x.ch/2" in {r["url"] for r in iter_l1(str(wdir))}


# --- round-trip ----------------------------------------------------------------

def test_roundtrip_everything_written_is_readable(tmp_path):
    """(c) Across shard rolls and resumes, every row written comes back."""
    written = []
    for run in range(4):                  # four separate "processes"
        st = L0Store(str(tmp_path), "rt", shard_max_bytes=400)
        for i in range(12):
            u = f"https://x.ch/r{run}-{i}"
            st.put(u, (f"body-{run}-{i}" * 5).encode())
            written.append(u)
        st.close(complete=(run == 3))
    wdir = str(tmp_path / "wave=rt")
    assert [r["url"] for r in iter_l0(wdir)] == written
    scan = gz.scan_store(wdir)
    assert scan["poisoned"] is False and scan["torn_members"] == 0
    assert scan["rows_readable"] == len(written) == scan["rows_reachable"]
    assert scan["rows_reachable"] == sum(
        naive_rows(str(p)) for p in sorted(
            __import__("pathlib").Path(wdir).glob("shard-*.jsonl.gz"))), \
        "on an undamaged store the derived count matches a real reader exactly"


def test_next_shard_index_advances_past_existing(tmp_path):
    d = str(tmp_path / "wave=x")
    os.makedirs(d)
    assert gz.next_shard_index(d) == 0
    for name in ("shard-0000.jsonl.gz", "shard-0007.jsonl.gz"):
        open(os.path.join(d, name), "wb").close()
    assert gz.next_shard_index(d) == 8
    path, fh, idx = gz.create_shard(d, gz.next_shard_index(d))
    fh.close()
    assert idx == 8 and path.endswith("shard-0008.jsonl.gz")


def test_create_shard_never_truncates_an_existing_file(tmp_path):
    d = str(tmp_path / "wave=y")
    os.makedirs(d)
    with open(os.path.join(d, "shard-0000.jsonl.gz"), "wb") as f:
        f.write(b"precious")
    path, fh, idx = gz.create_shard(d, 0)   # asked for a taken index
    fh.close()
    assert idx == 1
    assert open(os.path.join(d, "shard-0000.jsonl.gz"), "rb").read() == b"precious"


# --- fixed-name stores (sitemap_ingest, ingest_cc198) --------------------------

def test_rotation_keeps_existing_globs_matching(tmp_path):
    """Fixed-name gzip stores rotate instead of appending, and the rotated
    name must still be found by the readers' `glob("*.raw.jsonl.gz")`."""
    canonical = str(tmp_path / "example.ch.raw.jsonl.gz")
    seen = []
    for run in range(3):
        f, path = gz.create_rotated_gzip_text(canonical,
                                              suffix=".raw.jsonl.gz")
        f.write(json.dumps({"url": f"https://example.ch/{run}"}) + "\n")
        f.close()
        seen.append(path)

    assert seen[0] == canonical
    assert len(set(seen)) == 3, "each run must claim its own file"
    found = sorted(tmp_path.glob("*.raw.jsonl.gz"))
    assert len(found) == 3, "rotated names still match the readers' glob"
    urls = []
    for p in found:
        urls += [r["url"] for r in gz.iter_jsonl_rows(str(p))]
    assert sorted(urls) == [f"https://example.ch/{i}" for i in range(3)]


def test_rotation_never_appends_behind_a_tear(tmp_path):
    """The whole point: a crashed run's torn file gains no member behind it."""
    canonical = str(tmp_path / "example.ch.raw.jsonl.gz")
    f, path = gz.create_rotated_gzip_text(canonical, suffix=".raw.jsonl.gz")
    for i in range(50):
        f.write(json.dumps({"url": f"https://example.ch/a{i}"}) + "\n")
        f.flush()
    del f                                  # killed: no clean close
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - 4)

    f2, path2 = gz.create_rotated_gzip_text(canonical,
                                            suffix=".raw.jsonl.gz")
    f2.write(json.dumps({"url": "https://example.ch/resumed"}) + "\n")
    f2.close()

    assert path2 != path
    assert gz.scan_file(path)["poisoned"] is False
    assert gz.scan_file(path2)["torn_members"] == 0
    all_urls = set()
    for p in sorted(tmp_path.glob("*.raw.jsonl.gz")):
        all_urls |= {r["url"] for r in gz.iter_jsonl_rows(str(p))}
    assert "https://example.ch/resumed" in all_urls
    assert len(all_urls) >= 50


def test_rotation_rejects_a_mismatched_suffix(tmp_path):
    with pytest.raises(ValueError):
        gz.next_free_rotation(str(tmp_path / "a.txt"), ".raw.jsonl.gz")


# --- the repair path, end to end -----------------------------------------------

def test_l0_repair_makes_a_poisoned_wave_naively_readable(tmp_path):
    """`repair_wave` (the former l0_repair script) is what turns an already-damaged store back into one a
    NAIVE gzip reader can read. It now reads through the shared member
    walker; this proves the converged version still salvages the incident
    artifact end to end."""
    import base64
    import contextlib
    import io

    wdir = tmp_path / "wave=broken"
    wdir.mkdir()
    shard = str(wdir / "shard-0000.jsonl.gz")

    def l0_rows(n, prefix):
        out = []
        for i in range(n):
            body = f"<html>{prefix}-{i}</html>".encode()
            out.append({"url": f"https://x.ch/{prefix}{i}",
                        "final_url": f"https://x.ch/{prefix}{i}",
                        "status": 200, "content_type": "text/html",
                        "encoding": "utf-8", "kind": "html",
                        "fetched_at": "2026-08-19T00:00:00Z",
                        "sha256": __import__("hashlib").sha256(body).hexdigest(),
                        "bytes": len(body), "tool": "test",
                        "body_b64": base64.b64encode(body).decode("ascii")})
        return out

    legacy_append_writer(shard, l0_rows(30, "crashed"), tear=True)
    legacy_append_writer(shard, l0_rows(20, "resumed"))   # the fatal append
    assert gz.scan_file(shard)["poisoned"] is True
    assert not [u for u in naive_urls(shard) if "resumed" in u], \
        "the intact member behind the tear is invisible before repair"

    with contextlib.redirect_stdout(io.StringIO()):
        gz.repair_wave(str(wdir))

    repaired = sorted(wdir.glob("shard-*.jsonl.gz"))
    assert repaired, "repair must leave readable shards"
    assert sorted(wdir.glob("*.corrupt")), "originals preserved as .corrupt"
    total_naive = sum(naive_rows(str(p)) for p in repaired)
    assert total_naive >= 49, f"only {total_naive} rows naively readable"
    urls = set()
    for p in repaired:
        urls |= {r["url"] for r in gz.iter_jsonl_rows(str(p))}
    assert len([u for u in urls if "resumed" in u]) == 20
    for p in repaired:
        assert gz.scan_file(str(p))["poisoned"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
