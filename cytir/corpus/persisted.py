#!/usr/bin/env python3
# Moved from the flat module `persisted_inputs` (task 677, task 643 S5); the old path is an alias shim.
"""Persisted inputs are a precondition to load (ADR-105, task 581).

WHY THIS EXISTS
---------------
ADR-026 (2026-07-04) made the corpus primary and every qdrant collection a
DERIVED artifact: "every collection must be rebuildable from the store --
rebuild-from-scratch is replay L0->L1->L2->embed, no re-fetching."  That rule
was written down and never given an enforcement surface.  Between then and
2026-08, live campaigns wrote rows straight into prod and the twin without
sealing the page they had just read: task 426 found 1,636,193 points whose only
surviving input was their OWN legacy production passage, and task 579 found
492,211 deficit pages with no L0, no L1 and no L2 text anywhere on disk.  For
those rows ADR-026 is simply false -- they cannot be rebuilt, only re-fetched,
and a re-fetch is a fresh consent question and a fresh capture, not a replay.

The chain that produced that debt has four links and only one of them was ever
a decision (see task 581).  This module is
the guard for link (c): a loader that required nothing of its inputs.

WHAT IT REFUSES
---------------
Every path that writes rows into a collection asks, BEFORE the first write,
whether each row's page has a persisted L1 record (markdown + sha) in a
REGISTERED store.  A page that has none is not loadable: the refusal names the
pages (capped) and the count, and carries a non-zero exit through
``UnpersistedInputs``.

It is deliberately not softenable from inside a run (verification-doctrine 12).
There is no ``--allow`` flag, no ``report`` mode and no env bypass.  Two
requirement levels exist and both are declared by the CALLER in code, recorded
in the verdict, and named in an ADR:

  ``require="l1"``      the ADR-105 bar: a persisted L1 markdown record.
  ``require="l1b"``     additionally admits a page whose text survives only as
                        L2 passages -- the ADR-093 overlap-chaining route.  The
                        verdict reports ``l1b`` pages separately so a lane can
                        never report a chained reconstruction as a replay.

INDEX, NOT A SCAN
-----------------
There is no corpus-wide URL->text lookup in this repo (the L0/L1 iterators in
``pack_fetch.py`` are sequential wave readers).  Resolving one page by scanning
167 GB of stores is not a guard anyone will leave switched on, so the stores are
indexed once into sorted, memory-mapped, content-hashed arrays and the lookup is
a binary search.

  ``python3 -m cytir.corpus.persisted build-index --out-dir DIR``

The store roster is GLOBBED from the storage tree at build time and printed with
the result (corpus-doctrine 8, memory ``census-roster-from-manifest``); the
registry in ``data/persisted-input-stores.v1.json`` names the globs, never the
files.  The index manifest records the roster, the per-store counts, the
canonicalizer, the hash and the sha256 of every array, so a stale or truncated
index is a refusal rather than a silent miss.

FAIL-CLOSED
-----------
A missing index, an unreadable index, a manifest whose arrays do not match their
recorded sha256, an unknown ``require`` level, or an empty page list where rows
were expected -- every one of those raises.  Skipping on absent input is
reporting green over unchecked ground.

64-BIT KEYS
-----------
A page key is ``blake2b(canonical_url, digest_size=8)``.  Over 50M indexed pages
the chance that one un-persisted page collides with a persisted one -- the only
direction that can produce a FALSE PASS -- is 50e6/2**64 ~= 2.7e-12 per lookup.
That is recorded here rather than left for someone to re-derive.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import mmap
import multiprocessing
import os
import pathlib
import re
import sqlite3
import struct
import sys
import urllib.parse
import time
from typing import Iterable, Iterator, Optional

from cytir.core import common as AC  # noqa: E402  (mirror contract: ONE canonicalizer)
from cytir.core import paths as aw_paths  # noqa: E402

SCHEMA = "aw.persisted-inputs-index.v1"
REGISTRY_SCHEMA = "aw.persisted-input-stores.v1"
REGISTER_SCHEMA = "aw.unpersisted-legacy-register.v1"

#: record = big-endian (u64 page key, u16 store ordinal, u32 file ordinal).
#: Big-endian so a plain bytes comparison IS the numeric comparison, which is
#: what makes the binary search below a `bytes` bisect with no unpacking.
_REC = struct.Struct(">QHI")
RECORD_BYTES = _REC.size          # 14
_SHA_REC = struct.Struct(">Q")

LAYERS = ("l0", "l1", "l2")

#: the levels a caller may require.  A level is a code-level declaration, never
#: a runtime flag -- see the module docstring.
REQUIRE_LEVELS = {
    "l1": ("l1",),
    "l1b": ("l1", "l2"),
}

DEFAULT_REGISTRY = str(aw_paths.data_path("persisted-input-stores.v1.json"))
DEFAULT_REGISTER = str(aw_paths.data_path("unpersisted-legacy-register.v1.json"))

#: how many missing pages a refusal names before it says "and N more".  A
#: refusal that prints 400,000 URLs is a refusal nobody reads.
NAME_LIMIT = 20

#: task 641/662, 2026-09-25: the scan cache that makes `build_index`
#: incremental. Lives INSIDE the index dir (same lifecycle: a `--out-dir`
#: that is deleted or copied elsewhere carries its own cache with it).
CACHE_DIRNAME = ".scan-cache"
#: marks which registry content the cache was built against. A registry
#: edit (a store's glob, kind, or field list -- the EXTRACTION config, not
#: the file content) can make an OLD cached scan wrong for a file whose
#: bytes never changed, so any change to the registry file invalidates the
#: whole cache for that build (still incremental afterwards -- this is the
#: "full rebuild only on a schema change" the auto-refresh promises, and it
#: is cheap to check: one file hash, not a parse).
_REGISTRY_MARKER = "REGISTRY_SHA256"


class PersistedInputsError(RuntimeError):
    """Configuration, index or input the guard cannot evaluate. Fail closed."""


class UnpersistedInputs(PersistedInputsError):
    """Rows whose page has no persisted input. The refusal itself."""

    def __init__(self, message: str, verdict: dict):
        super().__init__(message)
        self.verdict = verdict


# --------------------------------------------------------------------------
# keys
# --------------------------------------------------------------------------

def page_key(url: str) -> int:
    """The index key of a page. ONE canonicalizer: ``aw_common.canonical_url``."""
    canon = AC.canonical_url(url or "")
    return int.from_bytes(
        hashlib.blake2b(canon.encode("utf-8"), digest_size=8).digest(), "big")


#: Address-spelling equivalences: one page, two spellings of the same address,
#: written by two different producers. NOT a canonicalizer change --
#: ``aw_common.canonical_url`` is a mirror contract that mints point ids and
#: must not move (engineering-doctrine 6). This is a LOOKUP-side widening: a
#: page is found under any spelling the corpus is known to have written it in,
#: and a hit is still a hit on the same stored text.
#:
#: The one rule, measured rather than guessed: the dump-derived Wikipedia waves
#: (`wave=l1_wiki_{en,de,fr,it}_20260601`, written by `corpus_store.py` from
#: `_dumps/derived/wiki_*_l1.jsonl`) spell a title's spaces as `%20`
#: -- `https://en.wikipedia.org/wiki/Imperial%20War%20Museum%20Duxford` -- while
#: production spells them with underscores. The first run of task 581's census
#: read that as 6,215,311 en.wikipedia pages with no stored text; an exhaustive
#: scan of all 7,189,653 rows of the en wave found the pages present under the
#: other spelling. A single-word title (`.../wiki/Rychvald`) matched either way,
#: which is exactly why a head sample did not show it (corpus-doctrine 5).
#:
#: Scoped to the host suffix it was measured on. Elsewhere `_` and `%20` are
#: different resources and folding them would manufacture a false PASS, which
#: is the unsafe direction for this guard.
_WIKI_HOSTS = (".wikipedia.org",)


def _wiki_title_spellings(url: str):
    """The other spelling(s) of a Wikipedia article address, or ()."""
    try:
        p = urllib.parse.urlsplit(url)
    except ValueError:
        return ()
    host = (p.netloc or "").lower()
    if not any(host == h.lstrip(".") or host.endswith(h) for h in _WIKI_HOSTS):
        return ()
    path = p.path or ""
    out = []
    decoded = urllib.parse.unquote(path)
    for candidate in (decoded.replace(" ", "_"), decoded.replace("_", " ")):
        if candidate == path:
            continue
        for rendered in (candidate, urllib.parse.quote(candidate, safe="/:@")):
            if rendered != path:
                out.append(urllib.parse.urlunsplit(
                    (p.scheme, p.netloc, rendered, p.query, p.fragment)))
    return tuple(dict.fromkeys(out))


def page_keys(url: str) -> tuple:
    """Every key this page may be indexed under. The canonical one comes first.

    Used by lookups only; the index is written under ``page_key`` alone, so
    adding a spelling rule never requires a rebuild.
    """
    keys = [page_key(url)]
    for alt in _wiki_title_spellings(url or ""):
        k = page_key(alt)
        if k not in keys:
            keys.append(k)
    return tuple(keys)


def pair_key(url: str, sha: str) -> int:
    """The index key of a (page, capture-sha) pair."""
    canon = AC.canonical_url(url or "")
    payload = canon.encode("utf-8") + b"\x00" + (sha or "").encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "big")


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def _glob_regex(pattern: str):
    """Compile a store glob to a regex with POSIX glob semantics.

    ``**`` crosses directory separators, a single ``*`` and ``?`` do not.
    Written out rather than delegated to ``fnmatch``, whose ``*`` matches
    ``/`` and would let ``l0/wave=*/shard-*.jsonl.gz`` admit files nested
    arbitrarily deeper than the store actually reaches -- an over-match here
    means declaring a page persisted that is not.
    """
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                out.append(re.escape(pattern[i]))
                i += 1
            else:
                out.append(pattern[i:j + 1])
                i = j + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


@dataclasses.dataclass(frozen=True)
class Store:
    store_id: str
    layer: str
    kind: str                       # jsonl_gz | jsonl | parquet
    glob: str
    url_fields: tuple
    sha_fields: tuple
    #: a record counts as persistence only if one of these is non-empty.
    #: Without it, `unresolved.jsonl.gz` -- a list of URLs the lane FAILED to
    #: retrieve -- indexes as proof that those pages are on disk.
    text_fields: tuple = ()
    note: str = ""

    def files(self, corpus_root: str) -> list:
        """The store's files, WALKED from the tree (never a hand list).

        Not ``glob.glob``. The stdlib glob module silently refuses to descend
        into or match any path component beginning with a dot, and this
        corpus keeps 156,522 of its 156,653 uncompressed L1 documents under
        ``wikipedia-markdown-v3/lang=*/.multistream-fragments/range-*/``. A
        glob-based roster found 131 of them, and the guard built on it would
        have reported every Wikipedia page in production as un-persisted --
        the same class of blind spot as the eight-file draft of this registry,
        one layer deeper (corpus-doctrine 8, memory
        ``census-roster-from-manifest``).

        ANCHORED per path SEGMENT of ``self.glob``, not a single top-level
        ``os.walk`` filtered afterwards (task 663, 2026-09-25, found live by
        py-spy: a drain stuck 3h21m with zero writes, deep inside
        ``_preservation/scratch-landing-v1/.../legi/legi/global/code_et...``
        -- millions of directories no store glob names). The OLD "strip
        trailing glob segments to find a literal head" approach anchors
        correctly ONLY when every wildcard segment sits at the END of the
        pattern; the moment the FIRST segment after ``corpus_root`` is a
        wildcard (``wave=*/*.parquet`` -- a real registered store, `l2-
        legacy-corpus`), stripping walks back to ``corpus_root`` itself and
        ``os.walk`` descends into EVERY sibling directory, including scratch
        trees no store glob ever names. Descending one segment at a time and
        matching EACH segment against only that level's directory listing
        (never a subtree scan) means a directory that does not match the
        pattern at ITS depth -- ``_preservation``, ``_work``, anything not
        named ``wave=*`` at the corpus root -- is never entered at all.
        """
        pattern = os.path.join(corpus_root, self.glob)
        rx = _glob_regex(pattern)

        def _dedup_by_realpath(paths: list[str]) -> list[str]:
            """Two different (possibly symlinked) routes landing on the SAME
            real directory must be descended into once, not twice -- but
            this is scoped to ONE step's own candidates; it must never
            reject a directory just because an EARLIER segment already
            anchored there (that is not a cycle, it is the next segment
            correctly continuing from where the last one left off)."""
            seen: set[str] = set()
            out_paths = []
            for p in paths:
                real = os.path.realpath(p)
                if real in seen:
                    continue
                seen.add(real)
                out_paths.append(p)
            return out_paths

        # `self.glob` alone drives descent -- `corpus_root`'s OWN path
        # components (e.g. `/mnt/.../aw-corpus`) are never re-matched
        # against a glob segment; they are simply where segment 0 starts.
        glob_segments = self.glob.split("/")
        current_dirs = [corpus_root]
        for seg in glob_segments[:-1]:
            next_dirs: list[str] = []
            if seg == "**":
                # Zero or more directory levels, at any depth, below each
                # currently-anchored dir -- still SCOPED to those dirs only
                # (e.g. corpus_root/l1 for `l1/**/*.jsonl.gz`), never a
                # sibling. `os.walk` here is intentional and correct: `**`
                # genuinely means "descend arbitrarily", and by this point
                # every literal/wildcard segment before it has already
                # pruned to a small, real subtree. `followlinks=True`: a
                # store tree assembled with a symlinked partition is
                # invisible to a plain, non-following walk, which is the
                # dot-directory blind spot again in another costume --
                # found when a symlinked `l1/release=news-daily` indexed as
                # zero pages. Cycles are broken by remembering real paths
                # already descended into, FRESH per `**` occurrence (a
                # symlink loop within THIS subtree, not cross-segment
                # bookkeeping that would wrongly block legitimate descent).
                seen_in_recursion: set[str] = set()
                for d in current_dirs:
                    for dirpath, dirnames, _files in os.walk(d, followlinks=True):
                        real = os.path.realpath(dirpath)
                        if real in seen_in_recursion:
                            dirnames[:] = []
                            continue
                        seen_in_recursion.add(real)
                        next_dirs.append(dirpath)
            elif any(ch in seg for ch in "*?["):
                seg_rx = _glob_regex(seg)
                for d in current_dirs:
                    try:
                        entries = list(os.scandir(d))
                    except OSError:
                        continue
                    for entry in entries:
                        if not seg_rx.match(entry.name):
                            continue
                        try:
                            is_dir = entry.is_dir(follow_symlinks=True)
                        except OSError:
                            continue
                        if not is_dir:
                            continue
                        next_dirs.append(entry.path)
                next_dirs = _dedup_by_realpath(next_dirs)
            else:
                # A literal segment needs no directory listing at all --
                # one os.path.isdir check per currently-anchored dir.
                for d in current_dirs:
                    cand = os.path.join(d, seg)
                    if os.path.isdir(cand):
                        next_dirs.append(cand)
                next_dirs = _dedup_by_realpath(next_dirs)
            current_dirs = next_dirs

        out = []
        for d in current_dirs:
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            for entry in entries:
                p = entry.path
                if not rx.match(p):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                # A symlinked RELEASE DIRECTORY is fine (followlinks
                # semantics above exist for exactly that -- the news-daily
                # incident this docstring's neighbour describes). A
                # symlinked LEAF FILE is a different failure: it makes
                # "persisted" depend on whatever scratch/work dir the
                # link's target lives in surviving untouched, which is
                # exactly what ADR-105 exists to rule out (task 641,
                # 2026-09-23 -- caught when `register_l1_release.py`'s
                # first cut symlinked task634/depth226 scratch files
                # straight into l1/). Refuse rather than index a page
                # whose only claim to persistence is a link.
                if os.path.islink(p):
                    raise PersistedInputsError(
                        f"{p}: is a symlink to {os.path.realpath(p)} -- "
                        "a store's LEAF FILE must be a real, persisted "
                        "copy (a symlinked release DIRECTORY is fine; a "
                        "symlinked file is not). Re-file it with "
                        "register_l1_release.py (default copy mode, no "
                        "--link) or remove it from the registered tree.")
                out.append(p)
        return sorted(out)


def load_registry(path: Optional[str] = None) -> dict:
    p = pathlib.Path(path or os.environ.get("AW_PERSISTED_REGISTRY")
                     or DEFAULT_REGISTRY)
    if not p.exists():
        raise PersistedInputsError(
            f"store registry {p} is missing -- refusing to treat an unknown "
            "set of stores as 'no stores'")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("schema") != REGISTRY_SCHEMA:
        raise PersistedInputsError(
            f"{p}: schema {doc.get('schema')!r} != {REGISTRY_SCHEMA!r}")
    return doc


def stores(registry: Optional[dict] = None) -> list:
    doc = registry if registry is not None else load_registry()
    out = []
    for raw in doc.get("stores") or []:
        layer = raw.get("layer")
        if layer not in LAYERS:
            raise PersistedInputsError(
                f"store {raw.get('store_id')!r}: layer {layer!r} not in {LAYERS}")
        out.append(Store(
            store_id=raw["store_id"], layer=layer, kind=raw["kind"],
            glob=raw["glob"], url_fields=tuple(raw.get("url_fields") or ()),
            sha_fields=tuple(raw.get("sha_fields") or ()),
            text_fields=tuple(raw.get("text_fields") or ()),
            note=raw.get("note", "")))
    if not out:
        raise PersistedInputsError("store registry declares no stores")
    return out


def corpus_root(explicit: Optional[str] = None) -> str:
    return str(explicit or aw_paths.CORPUS_ROOT)


# --------------------------------------------------------------------------
# index: read side
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Hit:
    url: str
    layer: str
    store_id: str
    store_path: str
    #: True/False when the caller supplied a sha the store can speak to;
    #: None when it supplied none, or the store records no capture sha.
    sha_matched: Optional[bool]


def default_index_dir() -> str:
    return (os.environ.get("AW_PERSISTED_INDEX")
            or os.path.join(corpus_root(), "_persisted-index"))


#: Opened indexes, keyed by directory. `assembly_binder load` calls the guard
#: once per partition (so a refusal lands on a resumable boundary), and the
#: file table alone is ~156k paths -- re-reading it per partition would turn a
#: free check into a real cost. Cleared by `reset_index_cache()`, which the
#: tests call so a monkeypatched AW_PERSISTED_INDEX is never served a stale
#: handle.
_OPEN_INDEXES: dict = {}


def reset_index_cache() -> None:
    for idx in list(_OPEN_INDEXES.values()):
        idx.close()
    _OPEN_INDEXES.clear()


def shared_index(directory: Optional[str] = None):
    """The open index for `directory` -- the sqlite-backed store (task 664,
    2026-09-26) when `<directory>/index.sqlite3` exists, else the legacy
    mmap `PersistedIndex` (`<directory>/INDEX.json`), so an index directory
    built before this task's migration keeps working unchanged until it is
    rebuilt. Both expose the identical `.lookup()`/`.directory`/`.manifest`
    surface `audit_pages` needs."""
    d = directory or default_index_dir()
    idx = _OPEN_INDEXES.get(d)
    if idx is None:
        if os.path.exists(sqlite_index_path(d)):
            idx = SqlitePersistedIndex.open(d)
        else:
            idx = PersistedIndex.open(d)
        _OPEN_INDEXES[d] = idx
    return idx


class PersistedIndex:
    """Memory-mapped, sorted, sha-verified page index over the registered stores."""

    def __init__(self, directory: str, manifest: dict, arrays: dict,
                 sha_arrays: dict, files: dict):
        self.directory = directory
        self.manifest = manifest
        self._arrays = arrays           # layer -> mmap
        self._sha_arrays = sha_arrays   # layer -> mmap
        self._files = files             # layer -> (store_ids, file_paths)

    # -- construction -----------------------------------------------------
    @classmethod
    def open(cls, directory: Optional[str] = None, *,
             verify_sha: bool = False) -> "PersistedIndex":
        d = directory or default_index_dir()
        man_path = os.path.join(d, "INDEX.json")
        if not os.path.exists(man_path):
            raise PersistedInputsError(
                f"no persisted-input index at {d} (INDEX.json missing). Build "
                "it with `python3 -m cytir.corpus.persisted build-index "
                f"--out-dir {d}`, or point AW_PERSISTED_INDEX at one. Refusing "
                "to load rows against an index that does not exist.")
        man = json.loads(pathlib.Path(man_path).read_text(encoding="utf-8"))
        if man.get("schema") != SCHEMA:
            raise PersistedInputsError(
                f"{man_path}: schema {man.get('schema')!r} != {SCHEMA!r}")
        arrays, sha_arrays, files = {}, {}, {}
        for layer, entry in (man.get("layers") or {}).items():
            path = os.path.join(d, entry["pages_file"])
            if not os.path.exists(path):
                raise PersistedInputsError(
                    f"{man_path} declares {entry['pages_file']} for layer "
                    f"{layer}, which is not on disk")
            size = os.path.getsize(path)
            if size != entry["pages_bytes"]:
                raise PersistedInputsError(
                    f"{path}: {size} bytes, manifest declares "
                    f"{entry['pages_bytes']} -- a truncated index misses "
                    "pages silently")
            if verify_sha and _sha256_file(path) != entry["pages_sha256"]:
                raise PersistedInputsError(f"{path}: sha256 mismatch")
            arrays[layer] = _mmap_ro(path)
            sp = entry.get("shas_file")
            if sp:
                spath = os.path.join(d, sp)
                if os.path.exists(spath):
                    sha_arrays[layer] = _mmap_ro(spath)
            files[layer] = (entry.get("store_ids") or [],
                            _read_file_table(os.path.join(d, entry["files_file"])))
        if not arrays:
            raise PersistedInputsError(
                f"{man_path} declares no layers -- an index over nothing "
                "reports every page missing, which is not a finding")
        return cls(d, man, arrays, sha_arrays, files)

    # -- lookup -----------------------------------------------------------
    def _find(self, layer: str, key: int):
        buf = self._arrays.get(layer)
        if buf is None:
            return None
        n = len(buf) // RECORD_BYTES
        target = key.to_bytes(8, "big")
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            off = mid * RECORD_BYTES
            if buf[off:off + 8] < target:
                lo = mid + 1
            else:
                hi = mid
        if lo >= n:
            return None
        off = lo * RECORD_BYTES
        k, store_ord, file_ord = _REC.unpack(buf[off:off + RECORD_BYTES])
        if k != key:
            return None
        return store_ord, file_ord

    def _sha_present(self, layer: str, key: int) -> Optional[bool]:
        buf = self._sha_arrays.get(layer)
        if buf is None or len(buf) == 0:
            return None
        n = len(buf) // 8
        target = key.to_bytes(8, "big")
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if buf[mid * 8:mid * 8 + 8] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo < n and buf[lo * 8:lo * 8 + 8] == target

    def layers_for(self, url: str) -> frozenset:
        keys = page_keys(url)
        return frozenset(l for l in self._arrays
                         if any(self._find(l, k) for k in keys))

    def lookup(self, url: str, sha: Optional[str] = None,
               order: Iterable = ("l1", "l0", "l2")) -> Optional[Hit]:
        keys = page_keys(url)
        for layer in order:
            found = None
            for key in keys:
                found = self._find(layer, key)
                if found:
                    break
            if not found:
                continue
            store_ord, file_ord = found
            store_ids, paths = self._files.get(layer, ([], []))
            sha_matched = None
            if sha:
                sha_matched = self._sha_present(layer, pair_key(url, sha))
            return Hit(url=url, layer=layer,
                       store_id=(store_ids[store_ord]
                                 if store_ord < len(store_ids) else "?"),
                       store_path=(paths[file_ord]
                                   if file_ord < len(paths) else ""),
                       sha_matched=sha_matched)
        return None

    def close(self) -> None:
        for buf in list(self._arrays.values()) + list(self._sha_arrays.values()):
            try:
                buf.close()
            except Exception:
                pass


def _mmap_ro(path: str):
    fh = open(path, "rb")
    if os.path.getsize(path) == 0:
        fh.close()
        return memoryview(b"")
    return mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_file_table(path: str) -> list:
    if not os.path.exists(path):
        raise PersistedInputsError(f"file table {path} is missing")
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------

def _normalize(pages) -> Iterator:
    for p in pages:
        if isinstance(p, str):
            yield p, None
        elif isinstance(p, dict):
            url = (p.get("url") or p.get("canonical_url")
                   or (p.get("payload") or {}).get("url")
                   or (p.get("payload") or {}).get("canonical_url"))
            sha = (p.get("sha") or p.get("markdown_sha256")
                   or p.get("content_digest")
                   or (p.get("payload") or {}).get("content_digest"))
            yield url, sha
        else:
            url, sha = (list(p) + [None])[:2]
            yield url, sha


def overlay_from_paths(url_to_path: dict, *, registry: Optional[dict] = None,
                       root: Optional[str] = None) -> dict:
    """Pages persisted in THIS run, keyed for the guard before the next rebuild.

    The index is a built artifact; a lane that files its L1 into the store and
    then asks the guard would be refused for minutes-old pages that are, in
    fact, on disk in a registered store. This closes that gap without opening a
    bypass: every entry must be a file that EXISTS and whose path matches a
    registered store's glob. You cannot hand the guard an overlay for a page
    you did not actually write into a registered store -- which is the whole
    difference between this and a `--allow` flag.
    """
    reg = stores(registry if registry is not None else load_registry())
    croot = corpus_root(root)
    patterns = [(s_.store_id, s_.layer, _glob_regex(os.path.join(croot, s_.glob)))
                for s_ in reg]
    out = {}
    for url, path in (url_to_path or {}).items():
        if not path or not os.path.exists(path):
            raise PersistedInputsError(
                f"overlay claims {url} is persisted at {path!r}, which does "
                "not exist. An overlay is evidence, not an assertion.")
        real = os.path.abspath(path)
        match = next(((sid, layer) for sid, layer, rx in patterns
                      if rx.match(real)), None)
        if match is None:
            raise PersistedInputsError(
                f"overlay claims {url} is persisted at {real}, which is not "
                "inside any store registered in "
                f"{DEFAULT_REGISTRY}. Filing text somewhere no registry can "
                "see it is the defect this guard exists for (lesson L581-2).")
        out[AC.canonical_url(url)] = Hit(url=url, layer=match[1],
                                         store_id=match[0], store_path=real,
                                         sha_matched=None)
    return out


def page_text_persisted(url: str, sha: Optional[str] = None, *,
                        index: Optional[PersistedIndex] = None,
                        order: Iterable = ("l1", "l0", "l2")) -> Optional[Hit]:
    """Resolve one page to the store file that persists its text, or None."""
    idx = index or shared_index()
    return idx.lookup(url, sha, order=order)


def audit_pages(pages, *, index: Optional[PersistedIndex] = None,
                require: str = "l1", label: str = "",
                check_sha: bool = False,
                overlay: Optional[dict] = None) -> dict:
    """Route every page to exactly one terminal. Never raises on a MISSING page.

    ``check_sha`` is OFF by default and that is deliberate. A row's
    ``content_digest`` is not the L1 store's ``markdown_sha256``: comparing them
    would turn a field-name mismatch of OURS into a verdict about the record
    (corpus-doctrine 3), and would refuse most of the corpus for a reason that
    is not true. A caller turns it on only when it knows its rows cite the same
    capture sha the store records, and the verdict says which it did.

    Terminals (corpus-doctrine 2), and they conserve:
    ``persisted + l1b_reconstructible + missing == pages``, with
    ``rows_without_url`` counted against ``rows``.

    ``l1b_reconstructible`` is reported whether or not the requested level
    admits it, because "this page's text survives only as L2 passages" is a
    different fact from "this page is gone", and a lane that cannot tell them
    apart will price a re-fetch campaign wrong.
    """
    if require not in REQUIRE_LEVELS:
        raise PersistedInputsError(
            f"require={require!r} is not a declared level {sorted(REQUIRE_LEVELS)}")
    accept = REQUIRE_LEVELS[require]
    # The shared, corpus-wide index is loaded LAZILY -- only the first time a
    # page misses the overlay -- rather than unconditionally up front. A
    # caller whose overlay covers every page it hands in (the ADR-105
    # write-then-check pattern: file the L1 record, then check it in the same
    # run) must not be refused by a shared index that is mid-rebuild or
    # temporarily absent; it never asks the shared index anything. Every
    # existing caller (overlay=None, or a page that isn't in the overlay)
    # sees IDENTICAL behavior to before: the first page still triggers
    # `shared_index()` and still raises the same error, just one loop
    # iteration later than the old eager call -- no observable difference,
    # since nothing else runs before that first lookup either way.
    _idx_box: list = [index]
    def _idx():
        if _idx_box[0] is None:
            _idx_box[0] = shared_index()
        return _idx_box[0]

    seen = set()
    from_overlay = 0
    persisted = l1b = missing = no_url = 0
    sha_mismatch = 0
    sha_unverified = 0
    missing_names, l1b_names, mismatch_names = [], [], []
    rows = 0
    for url, sha in _normalize(pages):
        rows += 1
        if not url:
            no_url += 1
            continue
        canon = AC.canonical_url(url)
        if canon in seen:
            continue
        seen.add(canon)
        hit = (overlay or {}).get(canon)
        if hit is not None:
            from_overlay += 1
        else:
            hit = _idx().lookup(url, sha if check_sha else None,
                                order=("l1", "l0", "l2"))
        if hit is None:
            missing += 1
            if len(missing_names) < NAME_LIMIT:
                missing_names.append(canon)
            continue
        if hit.layer == "l2":
            l1b += 1
            if len(l1b_names) < NAME_LIMIT:
                l1b_names.append(canon)
        else:
            persisted += 1
        if hit.sha_matched is False:
            sha_mismatch += 1
            if len(mismatch_names) < NAME_LIMIT:
                mismatch_names.append(canon)
        elif hit.sha_matched is None and sha:
            sha_unverified += 1

    pages_seen = len(seen)
    unaccepted_l1b = l1b if "l2" not in accept else 0
    verdict = "PASS" if (missing == 0 and no_url == 0 and sha_mismatch == 0
                         and unaccepted_l1b == 0) else "FAIL"
    # Report the shared index's identity only if this call actually touched
    # it -- a fully overlay-covered audit (every page resolved without ever
    # calling `_idx()`) must not force a load here just to fill in a summary
    # field; that would silently reintroduce the eager dependency this lazy
    # load exists to remove.
    loaded_idx = _idx_box[0]
    return {
        "schema": "aw.persisted-inputs-audit.v1",
        "label": label,
        "require": require,
        "sha_checked": bool(check_sha),
        "rows": rows,
        "pages": pages_seen,
        "persisted": persisted,
        "persisted_in_this_run": from_overlay,
        "l1b_reconstructible": l1b,
        "l1b_not_accepted": unaccepted_l1b,
        "missing": missing,
        "rows_without_url": no_url,
        "sha_mismatch": sha_mismatch,
        "sha_unverified": sha_unverified,
        "missing_sample": missing_names,
        "l1b_sample": l1b_names,
        "sha_mismatch_sample": mismatch_names,
        "index_dir": loaded_idx.directory if loaded_idx else None,
        "index_built_at": loaded_idx.manifest.get("built_at") if loaded_idx else None,
        # A hole in the instrument travels with every verdict built on it, so
        # a refusal caused by an unreadable store file can never be read as a
        # page that was never stored (verification-doctrine 6).
        "index_unreadable_files": (loaded_idx.manifest.get("unreadable_files", 0)
                                   if loaded_idx else 0),
        "index_not_consulted": loaded_idx is None,
        "verdict": verdict,
    }


def assert_pages_persisted(pages, *, index: Optional[PersistedIndex] = None,
                           require: str = "l1", label: str = "",
                           check_sha: bool = False,
                           overlay: Optional[dict] = None) -> dict:
    """The refusal. Returns the verdict on PASS, raises ``UnpersistedInputs``.

    Empty input is a refusal too: a loader that reaches this with no pages has
    either lost its rows or is asking the wrong question, and both are worse
    than the write it wanted to do.
    """
    v = audit_pages(pages, index=index, require=require, label=label,
                    check_sha=check_sha, overlay=overlay)
    if v["pages"] == 0:
        raise UnpersistedInputs(
            f"[persisted-inputs] REFUSED {label or '(unlabelled)'}: zero pages "
            "to check. An applier with no pages has lost its rows; it has not "
            "passed a check.", v)
    if v["verdict"] == "PASS":
        return v
    parts = [f"[persisted-inputs] REFUSED {label or '(unlabelled)'} "
             f"(ADR-105, require={require})"]
    if v["missing"]:
        parts.append(
            f"  {v['missing']:,} of {v['pages']:,} page(s) have no persisted "
            f"L1 record in any registered store:")
        parts += [f"    {u}" for u in v["missing_sample"]]
        if v["missing"] > len(v["missing_sample"]):
            parts.append(f"    ... and {v['missing'] - len(v['missing_sample']):,} more")
    if v["l1b_not_accepted"]:
        parts.append(
            f"  {v['l1b_not_accepted']:,} page(s) survive ONLY as L2 passages. "
            "They are reconstructible under ADR-093 overlap-chaining, which is "
            "a named route with a measured 0.135% material-loss rate -- not a "
            "replay. A caller that means to take it declares require='l1b'.")
        parts += [f"    {u}" for u in v["l1b_sample"]]
    if v["rows_without_url"]:
        parts.append(f"  {v['rows_without_url']:,} row(s) carry no url at all")
    if v["sha_mismatch"]:
        parts.append(
            f"  {v['sha_mismatch']:,} page(s) are persisted under a DIFFERENT "
            "capture than the row cites:")
        parts += [f"    {u}" for u in v["sha_mismatch_sample"]]
    if v.get("index_unreadable_files"):
        parts.append(
            f"  NOTE: {v['index_unreadable_files']} store file(s) were "
            "unreadable when this index was built (see INDEX.json "
            "`unreadable`). A page missing here may be inside one of them: "
            "this refusal is safe, but it is not proof the page was never "
            "stored.")
    parts.append(
        "  A row whose page is not on disk cannot be rebuilt (ADR-026); it can "
        "only be re-fetched, which is a fresh consent question. Seal the input "
        "first, or register the family in "
        "data/unpersisted-legacy-register.v1.json and take it to an ADR.")
    raise UnpersistedInputs("\n".join(parts), v)


# --------------------------------------------------------------------------
# the debt register
# --------------------------------------------------------------------------

def load_register(path: Optional[str] = None) -> dict:
    p = pathlib.Path(path or os.environ.get("AW_UNPERSISTED_REGISTER")
                     or DEFAULT_REGISTER)
    if not p.exists():
        raise PersistedInputsError(
            f"unpersisted-legacy register {p} is missing -- refusing to treat "
            "an unknown debt as no debt")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("schema") != REGISTER_SCHEMA:
        raise PersistedInputsError(
            f"{p}: schema {doc.get('schema')!r} != {REGISTER_SCHEMA!r}")
    return doc


def registered_families(doc: Optional[dict] = None) -> dict:
    d = doc if doc is not None else load_register()
    return {e["provenance"]: e for e in (d.get("families") or [])}


# --------------------------------------------------------------------------
# index: build side
# --------------------------------------------------------------------------

def _iter_jsonl(path: str, url_fields, sha_fields, text_fields):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            url = next((rec[f] for f in url_fields if rec.get(f)), None)
            if not url:
                continue
            if text_fields and not any(rec.get(f) for f in text_fields):
                # addressable but not text-bearing: a residual/unresolved/link
                # record. Indexing it would make a page the lane FAILED to
                # retrieve look persisted.
                continue
            sha = next((rec[f] for f in sha_fields if rec.get(f)), None)
            yield str(url), (str(sha) if sha else None)


def _iter_parquet(path: str, url_fields, sha_fields, text_fields):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    ucol = next((f for f in url_fields if f in names), None)
    if ucol is None:
        return
    scol = next((f for f in sha_fields if f in names), None)
    tcol = next((f for f in text_fields if f in names), None)
    cols = [c for c in (ucol, scol, tcol) if c]
    for batch in pf.iter_batches(batch_size=65536, columns=cols):
        d = batch.to_pydict()
        urls = d[ucol]
        shas = d[scol] if scol else [None] * len(urls)
        texts = d[tcol] if tcol else [None] * len(urls)
        for u, s, t in zip(urls, shas, texts):
            if not u:
                continue
            if text_fields and tcol and not t:
                continue
            yield str(u), (str(s) if s else None)


def _iter_store_file(store: Store, path: str):
    if store.kind in ("jsonl_gz", "jsonl"):
        return _iter_jsonl(path, store.url_fields, store.sha_fields,
                           store.text_fields)
    if store.kind == "parquet":
        return _iter_parquet(path, store.url_fields, store.sha_fields,
                             store.text_fields)
    raise PersistedInputsError(f"store {store.store_id}: unknown kind "
                               f"{store.kind!r}")


def _sort_dedup_pages(chunks) -> bytes:
    """Sort packed 14-byte page records and keep the FIRST record per page key.

    Big-endian packing makes byte order numeric order, so the sort is a plain
    lexicographic one. "First" is by (store, file) because the chunks are
    concatenated in registry order -- which is what makes the registry's order
    the precedence order, rather than whichever worker happened to finish first.
    numpy when present; the stdlib path is correct and slower, which matters on
    a build box and not at all in a test.
    """
    blob = b"".join(chunks)
    n = len(blob) // RECORD_BYTES
    if n == 0:
        return b""
    try:
        import numpy as np
    except ImportError:
        recs = [blob[i * RECORD_BYTES:(i + 1) * RECORD_BYTES] for i in range(n)]
        recs.sort(key=lambda r: r[:8])           # stable -> first wins
        out, last = [], None
        for r in recs:
            if r[:8] != last:
                out.append(r)
                last = r[:8]
        return b"".join(out)
    arr = np.frombuffer(blob, dtype=np.uint8).reshape(n, RECORD_BYTES)
    keys = arr[:, :8].copy().view(np.dtype(">u8")).ravel()
    order = np.argsort(keys, kind="stable")      # stable -> first wins
    k = keys[order]
    first = np.empty(len(k), dtype=bool)
    first[0] = True
    np.not_equal(k[1:], k[:-1], out=first[1:])
    return arr[order][first].tobytes()


def _sort_dedup_shas(chunks) -> bytes:
    blob = b"".join(chunks)
    n = len(blob) // 8
    if n == 0:
        return b""
    try:
        import numpy as np
    except ImportError:
        return b"".join(sorted({blob[i * 8:(i + 1) * 8] for i in range(n)}))
    return np.unique(np.frombuffer(blob, dtype=np.dtype(">u8"))).tobytes()


# --------------------------------------------------------------------------
# incremental scan cache (task 641/662, 2026-09-25)
# --------------------------------------------------------------------------
#
# The index build is CPU-bound on gzip inflation and json parsing of the
# STORE FILES, not on the final sort/dedup of the small packed records those
# files produce (`_sort_dedup_pages`/`_sort_dedup_shas` run over ~14 bytes
# per row, already binary, no parsing). Task 641, 2026-09-25: the news and
# re-fetch lanes add new L1 files daily, and every `drain()` rebuilt the
# WHOLE index from scratch on that news -- 157,032 L1 files, 41M+ rows,
# 35-120 minutes PER DRAIN, becoming the lane's actual throughput ceiling.
#
# The fix: cache each STORE FILE's own scanned bytes (`pb`/`sb` from
# `_scan_task`), keyed by its path, signed by (size, mtime_ns). A file whose
# signature is unchanged since it was last scanned contributes its CACHED
# bytes instead of being re-opened/re-parsed; a new or changed file is
# scanned and the cache is updated. The expensive-vs-cheap split in
# `build_index` below still runs `_sort_dedup_pages`/`_sort_dedup_shas` over
# the FULL current set of per-file bytes every time, in the SAME
# (store-order, then per-store sorted-path order) precedence order a full
# rebuild uses -- so the byte content that goes into the merge is always
# identical to what a full rebuild would have produced for that file; only
# the SOURCE of those bytes (cache read vs. fresh scan) differs. This is
# what makes "incremental" and "full" provably produce the same INDEX.json
# (see test_persisted_inputs.py's full-vs-incremental fixture test).


def _cache_entry_key(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()


def _cache_entry_paths(cache_dir: str, layer: str, path: str) -> tuple:
    key = _cache_entry_key(path)
    d = os.path.join(cache_dir, layer)
    return os.path.join(d, f"{key}.meta.json"), os.path.join(d, f"{key}.bin")


def _read_cache_entry(cache_dir: str, layer: str, path: str, size: int,
                      mtime_ns: int):
    """`(pb, sb, err, rows)` if a fresh cache entry exists for this exact
    (path, size, mtime_ns), else `None`. Never raises: a corrupt or partial
    cache entry (e.g. from a killed build) is just a cache MISS -- it costs
    one re-scan, never a wrong answer (mirrors verification-doctrine: a
    cache is an optimization, and a broken one must fail toward doing the
    real work, not toward a fabricated result)."""
    meta_p, bin_p = _cache_entry_paths(cache_dir, layer, path)
    try:
        meta = json.loads(pathlib.Path(meta_p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (meta.get("path") != path or meta.get("size") != size
            or meta.get("mtime_ns") != mtime_ns):
        return None
    try:
        blob = pathlib.Path(bin_p).read_bytes()
    except OSError:
        return None
    pages_len = meta.get("pages_len", 0)
    if len(blob) < pages_len:
        return None
    return blob[:pages_len], blob[pages_len:], meta.get("err"), meta.get("rows", 0)


def _write_cache_entry(cache_dir: str, layer: str, path: str, size: int,
                       mtime_ns: int, pb: bytes, sb: bytes, err, rows: int) -> None:
    """Atomic write (temp file + `os.replace`): a build killed mid-write
    must never leave a torn cache entry that a later `_read_cache_entry`
    could mistake for a valid one."""
    meta_p, bin_p = _cache_entry_paths(cache_dir, layer, path)
    os.makedirs(os.path.dirname(bin_p), exist_ok=True)
    tag = f".tmp{os.getpid()}"
    tmp_bin = bin_p + tag
    with open(tmp_bin, "wb") as fh:
        fh.write(pb)
        fh.write(sb)
    os.replace(tmp_bin, bin_p)
    meta = {"path": path, "size": size, "mtime_ns": mtime_ns, "rows": rows,
           "err": err, "pages_len": len(pb)}
    tmp_meta = meta_p + tag
    pathlib.Path(tmp_meta).write_text(json.dumps(meta), encoding="utf-8")
    os.replace(tmp_meta, meta_p)


def _cache_is_current(cache_dir: str, registry_path: str) -> bool:
    """False if the registry's own bytes changed since the cache was last
    built for it -- a store's glob/kind/field-list is EXTRACTION CONFIG, not
    file content, and the (size, mtime_ns) signature has no way to see a
    config edit. Also false (harmlessly) if the cache has never been built."""
    marker = os.path.join(cache_dir, _REGISTRY_MARKER)
    try:
        current = hashlib.sha256(
            pathlib.Path(registry_path).read_bytes()).hexdigest()
        prior = pathlib.Path(marker).read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return current == prior


def _stamp_cache_registry(cache_dir: str, registry_path: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.sha256(pathlib.Path(registry_path).read_bytes()).hexdigest()
    marker = os.path.join(cache_dir, _REGISTRY_MARKER)
    tmp = marker + f".tmp{os.getpid()}"
    pathlib.Path(tmp).write_text(digest, encoding="utf-8")
    os.replace(tmp, marker)


def _scan_task(task):
    """One store file -> its packed records. Runs in a worker process.

    The index build is CPU-bound on gzip inflation and json parsing -- one
    32 GB pubmed shard alone held a single core for 40 minutes -- so files are
    scanned in parallel and each worker returns two flat `bytes` blobs rather
    than a list of objects, which is what keeps the pickling cost negligible.
    """
    (store_id, kind, url_fields, sha_fields, text_fields,
     path, store_ord, file_ord) = task
    s = Store(store_id=store_id, layer="", kind=kind, glob="",
              url_fields=url_fields, sha_fields=sha_fields,
              text_fields=text_fields)
    recs = bytearray()
    shas = bytearray()
    rows = 0
    err = None
    try:
        for url, sha in _iter_store_file(s, path):
            rows += 1
            recs += _REC.pack(page_key(url), store_ord, file_ord)
            if sha:
                shas += _SHA_REC.pack(pair_key(url, sha))
    except Exception as exc:
        # A store file we cannot read is a HOLE IN THE INSTRUMENT, not an empty
        # store, and not a page that is absent (verification-doctrine 6).
        # ADR-093 section 6 already recorded that three l0 waves are
        # gzip-corrupt at seal and 16 of 28 carry `sealed_by: None`. The rows
        # read before the error are real and are kept; the file is recorded
        # with its error, the count travels in the manifest, and every verdict
        # built on this index carries it -- so a refusal caused by a hole can
        # never be read as a page that was never stored. The bias is toward
        # REFUSING, which is the safe direction for this guard.
        err = f"{type(exc).__name__}: {exc}"[:200]
    return file_ord, bytes(recs), bytes(shas), err, rows


def _log(*a):
    """Default build log. FLUSHED: a multi-hour index build whose progress sits
    in a pipe buffer is a build nobody can supervise (deployment-doctrine 2)."""
    print(*a, flush=True)


def build_index(out_dir: str, *, registry_path: Optional[str] = None,
                root: Optional[str] = None, workers: Optional[int] = None,
                log=_log, incremental: bool = True) -> dict:
    """Build (or refresh) the persisted-input index.

    ``incremental`` (default True, task 641/662, 2026-09-25): a STORE FILE
    whose (size, mtime_ns) is unchanged since it was last scanned into the
    on-disk cache under ``<out_dir>/.scan-cache/`` contributes its cached
    bytes instead of being re-opened and re-parsed. The merge step
    (``_sort_dedup_pages``/``_sort_dedup_shas``) still runs over the FULL
    current file set every time, in the same precedence order a full build
    uses, so the result is byte-identical to ``incremental=False`` -- only
    which files get re-scanned differs. Pass ``incremental=False`` (or edit
    ``registry_path``'s own file, which this function detects and treats as
    a forced full rescan -- see ``_cache_is_current``) to force a real full
    rebuild "on request" or "on a schema change"."""
    reg_doc = load_registry(registry_path)
    reg = stores(reg_doc)
    croot = corpus_root(root)
    if not os.path.isdir(croot):
        raise PersistedInputsError(
            f"corpus root {croot} is not a directory -- set CYTIR_CORPUS_ROOT")
    os.makedirs(out_dir, exist_ok=True)
    workers = int(workers or os.environ.get("AW_PERSISTED_INDEX_WORKERS")
                  or min(12, os.cpu_count() or 1))
    t0 = time.time()
    manifest_layers = {}
    roster_report = []
    unreadable: list = []
    total_cache_hits = 0
    total_scanned = 0

    registry_path_str = str(registry_path or DEFAULT_REGISTRY)
    cache_dir = os.path.join(out_dir, CACHE_DIRNAME)
    use_cache_reads = incremental and _cache_is_current(cache_dir, registry_path_str)
    if incremental and not use_cache_reads:
        log(f"[persisted-inputs] scan cache at {cache_dir} is absent or the "
           f"registry ({registry_path_str}) changed since it was built -- "
           "this build re-scans every file (a real full rebuild, driven by "
           "a schema change or a first build) and refreshes the cache for "
           "the NEXT incremental run")
    elif not incremental:
        log("[persisted-inputs] incremental=False -- forced full rebuild, "
           "cache reads skipped (cache is still refreshed for next time)")

    for layer in LAYERS:
        layer_stores = [s for s in reg if s.layer == layer]
        if not layer_stores:
            continue
        store_ids, file_paths, file_sigs, file_store_id, tasks = [], [], [], [], []
        page_chunks: list = []
        sha_chunks: list = []
        cache_hits = 0
        for s in layer_stores:
            files = s.files(croot)
            store_ord = len(store_ids)
            store_ids.append(s.store_id)
            roster_report.append(
                {"store_id": s.store_id, "layer": layer, "glob": s.glob,
                 "files": len(files)})
            log(f"[persisted-inputs] {layer}/{s.store_id}: {len(files)} file(s) "
                f"globbed from {os.path.join(croot, s.glob)}")
            # An empty store is REPORTED, never silently skipped: "no files"
            # and "store not configured" must not look the same.
            for path in files:
                file_ord = len(file_paths)
                file_paths.append(path)
                file_store_id.append(s.store_id)
                page_chunks.append(b"")
                sha_chunks.append(b"")
                st = os.stat(path)
                sig = (st.st_size, st.st_mtime_ns)
                file_sigs.append(sig)
                cached = (_read_cache_entry(cache_dir, layer, path, *sig)
                         if use_cache_reads else None)
                if cached is not None:
                    pb, sb, err, rows = cached
                    page_chunks[file_ord] = pb
                    sha_chunks[file_ord] = sb
                    cache_hits += 1
                    if err:
                        unreadable.append({
                            "store_id": s.store_id, "path": path,
                            "error": err, "rows_before_error": rows})
                else:
                    tasks.append((s.store_id, s.kind, s.url_fields, s.sha_fields,
                                  s.text_fields, path, store_ord, file_ord))

        rows_read = sum(len(pb) // RECORD_BYTES for pb in page_chunks)
        done = 0
        if tasks:
            log(f"[persisted-inputs] {layer}: {cache_hits}/{len(file_paths)} "
               f"file(s) served from the scan cache unchanged; scanning "
               f"{len(tasks)} new/changed file(s)")
            with multiprocessing.Pool(workers) as pool:
                for file_ord, pb, sb, err, rows in pool.imap_unordered(
                        _scan_task, tasks, chunksize=1):
                    page_chunks[file_ord] = pb
                    sha_chunks[file_ord] = sb
                    rows_read += rows
                    done += 1
                    if incremental:
                        size, mtime_ns = file_sigs[file_ord]
                        _write_cache_entry(cache_dir, layer, file_paths[file_ord],
                                          size, mtime_ns, pb, sb, err, rows)
                    if err:
                        unreadable.append({
                            "store_id": file_store_id[file_ord],
                            "path": file_paths[file_ord],
                            "error": err,
                            "rows_before_error": rows})
                        log(f"[persisted-inputs] UNREADABLE "
                            f"{file_paths[file_ord]}: {err} ({rows:,} row(s) "
                            "kept)")
                    if done % 2000 == 0 or done == len(tasks):
                        log(f"    {done}/{len(tasks)} file(s), "
                            f"{rows_read:,} row(s), {time.time()-t0:.0f}s")
        elif cache_hits:
            log(f"[persisted-inputs] {layer}: all {cache_hits} file(s) served "
               f"from the scan cache unchanged, nothing to scan")

        log(f"[persisted-inputs] {layer}: sorting {rows_read:,} record(s) "
           f"({cache_hits} cached, {len(tasks)} freshly scanned)")
        pages_blob = _sort_dedup_pages(page_chunks)
        shas_blob = _sort_dedup_shas(sha_chunks)
        del page_chunks, sha_chunks
        pages_file = f"{layer}.pages.bin"
        shas_file = f"{layer}.shas.bin"
        files_file = f"{layer}.files.json.gz"
        pp = os.path.join(out_dir, pages_file)
        pathlib.Path(pp).write_bytes(pages_blob)
        sp = os.path.join(out_dir, shas_file)
        pathlib.Path(sp).write_bytes(shas_blob)
        with gzip.open(os.path.join(out_dir, files_file), "wt",
                       encoding="utf-8") as fh:
            json.dump(file_paths, fh)
        manifest_layers[layer] = {
            "pages_file": pages_file,
            "pages_bytes": len(pages_blob),
            "pages_sha256": _sha256_file(pp),
            "pages": len(pages_blob) // RECORD_BYTES,
            "shas_file": shas_file,
            "shas": len(shas_blob) // 8,
            "shas_sha256": _sha256_file(sp),
            "files_file": files_file,
            "files": len(file_paths),
            "rows_read": rows_read,
            "store_ids": store_ids,
        }
        log(f"[persisted-inputs] {layer}: "
            f"{len(pages_blob)//RECORD_BYTES:,} distinct page(s), "
            f"{len(shas_blob)//8:,} capture sha(s), {time.time()-t0:.0f}s")
        del pages_blob, shas_blob
        total_cache_hits += cache_hits
        total_scanned += len(tasks)

    if incremental:
        # Stamped only after every layer scanned successfully -- a build
        # killed partway must not mark the cache "current" for a registry
        # it never finished indexing under.
        _stamp_cache_registry(cache_dir, registry_path_str)

    manifest = {
        "schema": SCHEMA,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_root": croot,
        "registry": str(registry_path or DEFAULT_REGISTRY),
        "registry_version": reg_doc.get("version"),
        "canonicalizer": "aw_common.canonical_url",
        "page_hash": "blake2b-64(canonical_url)",
        "pair_hash": "blake2b-64(canonical_url + NUL + sha)",
        "record_bytes": RECORD_BYTES,
        "workers": workers,
        "roster": roster_report,
        "layers": manifest_layers,
        "unreadable_files": len(unreadable),
        "unreadable": unreadable[:200],
        "elapsed_s": round(time.time() - t0, 1),
        "incremental": incremental,
        "cache_reads_used": use_cache_reads,
        "cache_hit_files": total_cache_hits,
        "scanned_files": total_scanned,
    }
    pathlib.Path(os.path.join(out_dir, "INDEX.json")).write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    log(json.dumps({"layers": {k: {kk: v[kk] for kk in
                                   ("pages", "shas", "files", "rows_read")}
                               for k, v in manifest_layers.items()}}, indent=1))
    return manifest


# --------------------------------------------------------------------------
# sqlite-backed index (task 664, 2026-09-26): a PERSISTENT store, queried
# per-page-url (O(pack), indexed), updated incrementally per changed store
# file -- never a full in-memory sort of the whole corpus.
#
# WHY THIS EXISTS: the #358 scan cache (above) already makes per-FILE
# scanning incremental -- a store file whose (size, mtime_ns) is unchanged
# contributes its cached bytes without being re-parsed. Measured live, task
# 664, 2026-09-26: that cache worked (157,057 cached, 14 freshly scanned)
# and `build_index` STILL took 15:56 -> 16:43 CEST (py-spy: stuck inside
# `_sort_dedup_pages`) just to reach the sort step -- because the FINAL
# assembly always re-gathers and Python-sorts every cached record (88.8M)
# into one big binary blob, on every drain, no matter how few files
# actually changed. `assert_pages_persisted` only ever asks "does url X
# have an L1 record" -- an O(pack) question a real index answers without
# ever touching the other 88.8M rows.
#
# A SQLite table with `(layer, key)` as its PRIMARY KEY (an implicit B-tree)
# gives that directly: `update_index_sqlite` deletes and re-inserts ONLY
# the rows a changed file contributed (an unchanged file is skipped via the
# SAME (path, size, mtime_ns) signature `_read_cache_entry` already uses);
# no global sort ever runs again. `SqlitePersistedIndex` is API-compatible
# with `PersistedIndex` (`.lookup()`, `.directory`, `.manifest`) so
# `audit_pages`/`assert_pages_persisted` need NOT change at all --
# `shared_index()` picks whichever backend is on disk.

SQLITE_INDEX_FILE = "index.sqlite3"

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS pages (
    layer TEXT NOT NULL,
    key INTEGER NOT NULL,
    store_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    PRIMARY KEY (layer, key)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS pages_file_idx ON pages(layer, file_path);
CREATE TABLE IF NOT EXISTS shas (
    layer TEXT NOT NULL,
    key INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    PRIMARY KEY (layer, key)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS shas_file_idx ON shas(layer, file_path);
CREATE TABLE IF NOT EXISTS scanned_files (
    layer TEXT NOT NULL,
    path TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    rows INTEGER NOT NULL,
    err TEXT,
    PRIMARY KEY (layer, path)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def _to_signed64(key: int) -> int:
    """SQLite's INTEGER column is signed 64-bit; `page_key()`/`pair_key()`
    return UNSIGNED 64-bit blake2b digests (half of them >= 2**63, which
    overflows a signed column outright). A bijective two's-complement
    reinterpretation -- never a hash, never a collision risk, the same
    input always maps to the same output and back -- keeps `key` a real
    INTEGER (indexed, fast) instead of falling back to TEXT."""
    return key - (1 << 64) if key >= (1 << 63) else key


def sqlite_index_dir(index_dir: str) -> str:
    """The DIRECTORY the sqlite db actually lives in -- `AW_PERSISTED_INDEX_
    SQLITE_DIR` when set, else `index_dir` unchanged (task 664, 2026-09-27
    incident: the default `index_dir` is under `corpus_root()`, which is
    the HDD (`<hdd-root>`, `df`: 7.4T). A from-scratch 88.8M-record
    build against an HDD-resident db is IO-bound into uselessness --
    measured live, every drain since 2026-09-26 17:31 CEST killed at
    systemd's 6h timeout, process state D, ~13min CPU burned in 6h wall.
    Splitting the db onto NVMe (`<nvme-root>`, `df`: 3.6T) from the
    legacy INDEX.json/scan-cache -- which stay wherever they already are,
    untouched -- needs no migration of anything else."""
    return os.environ.get("AW_PERSISTED_INDEX_SQLITE_DIR") or index_dir


def sqlite_index_path(index_dir: str) -> str:
    return os.path.join(sqlite_index_dir(index_dir), SQLITE_INDEX_FILE)


def sqlite_busy_timeout_s() -> float:
    """How long a writer waits for the sqlite index's single write lock.
    Task 676, 2026-09-30: two ERIC L1 seals refreshed the index at the same
    moment and one died at Python's 5 s default with `database is locked`
    (tranche 070000's chain, rc=1 with no other symptom). One merge is
    seconds to minutes, so waiting is right; the wait is bounded and
    overridable, and a genuinely stuck holder still surfaces as an error
    after the bound instead of hanging forever."""
    return float(os.environ.get("AW_PERSISTED_INDEX_BUSY_TIMEOUT_S", "900"))


def open_sqlite_db(index_dir: str, *, bulk: bool = False) -> sqlite3.Connection:
    """`bulk=True` (task 664, 2026-09-27): PRAGMAs for a large one-time
    LOAD, never for the incremental updater or a reader -- `synchronous=
    OFF` means a crash mid-build can corrupt the db (acceptable: the
    builder writes to a temp path and atomically replaces the real one
    only on a clean finish, so a crash just leaves the temp file, not a
    corrupted live index) and `journal_mode=MEMORY` skips the WAL/rollback
    journal entirely."""
    d = sqlite_index_dir(index_dir)
    os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(sqlite_index_path(index_dir),
                           timeout=sqlite_busy_timeout_s())
    if bulk:
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-200000")  # ~200MB page cache
    else:
        conn.executescript(_SQLITE_DDL)
        conn.execute("PRAGMA journal_mode=WAL")
    return conn


def update_index_sqlite(index_dir: str, *, registry_path: Optional[str] = None,
                        root: Optional[str] = None, log=_log) -> dict:
    """Merge every new-or-changed store file into the sqlite index. Never a
    full rebuild: an unchanged file (same path/size/mtime_ns as its
    `scanned_files` row) costs one indexed lookup, not a re-scan or a
    re-sort. Returns `{"files_scanned": n, "files_changed": n,
    "rows_indexed": n, "seconds": f}` -- `files_changed == 0` is the common,
    cheap "nothing new since last drain" case this replaces the expensive
    `persisted_index_is_stale()` pre-check for.

    REFUSES FAST if the db does not exist yet (task 664, 2026-09-27
    incident): the FIRST population is ~88.8M records, and this function's
    per-file loop -- fine for the steady-state "a handful of files changed"
    case -- is the wrong tool for that scale: every drain since 2026-09-26
    17:31 CEST attempted it inline, got killed at systemd's 6h start
    timeout still IO-bound in this loop (process state D, ~13min actual
    CPU burned in 6h wall), and restarted from zero next drain, forever.
    The first population is `cyt ops build-persisted-index`'s job (`cytir.corpus.persisted_build`) --
    run ONCE, out of band, with no timeout, bulk-insert PRAGMAs, and the
    unique index built AFTER the load, not this function's.
    """
    db_path = sqlite_index_path(index_dir)
    if not os.path.exists(db_path):
        raise PersistedInputsError(
            f"[persisted-inputs] REFUSED: no sqlite index at {db_path} yet. "
            "The first ~88.8M-record population must be built ONCE, out of "
            "band (no drain timeout), with `cyt ops build-persisted-index "
            f"--index-dir {index_dir}` -- bulk-"
            "insert PRAGMAs, index built after the load, minutes not hours. "
            "This function only ever does the cheap incremental top-up "
            "after that exists; it refuses rather than attempting the full "
            "population inline and getting killed by a drain's timeout.")
    t0 = time.time()
    conn = open_sqlite_db(index_dir)
    registry = load_registry(registry_path)
    croot = corpus_root(root)
    files_scanned = files_changed = rows_indexed = unreadable = 0
    try:
        for store in stores(registry):
            layer = store.layer
            for path in store.files(croot):
                files_scanned += 1
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                size, mtime_ns = st.st_size, st.st_mtime_ns
                cur = conn.execute(
                    "SELECT size, mtime_ns FROM scanned_files "
                    "WHERE layer=? AND path=?", (layer, path)).fetchone()
                if cur is not None and cur[0] == size and cur[1] == mtime_ns:
                    continue  # unchanged -- the common case, no re-scan
                page_rows: list[tuple] = []
                sha_rows: list[tuple] = []
                rows = 0
                err = None
                try:
                    for url, sha in _iter_store_file(store, path):
                        rows += 1
                        page_rows.append((layer, _to_signed64(page_key(url)),
                                         store.store_id, path))
                        if sha:
                            sha_rows.append((layer,
                                            _to_signed64(pair_key(url, sha)),
                                            path))
                except Exception as exc:  # noqa: BLE001 -- a hole in the
                    # instrument, not an empty store (same bias as
                    # `_scan_task`'s own docstring: refuse, never silently
                    # treat an unreadable file as "page never stored").
                    err = f"{type(exc).__name__}: {exc}"[:200]
                    unreadable += 1
                with conn:
                    conn.execute("DELETE FROM pages WHERE layer=? AND "
                                "file_path=?", (layer, path))
                    conn.execute("DELETE FROM shas WHERE layer=? AND "
                                "file_path=?", (layer, path))
                    conn.executemany(
                        "INSERT OR REPLACE INTO pages "
                        "(layer,key,store_id,file_path) VALUES (?,?,?,?)",
                        page_rows)
                    conn.executemany(
                        "INSERT OR REPLACE INTO shas "
                        "(layer,key,file_path) VALUES (?,?,?)", sha_rows)
                    conn.execute(
                        "INSERT OR REPLACE INTO scanned_files "
                        "(layer,path,size,mtime_ns,rows,err) VALUES "
                        "(?,?,?,?,?,?)", (layer, path, size, mtime_ns, rows, err))
                files_changed += 1
                rows_indexed += rows
                if files_changed % 200 == 0:
                    log(f"[sqlite-index] {files_changed} file(s) merged so far "
                       f"({rows_indexed:,} rows)...")
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES "
                "('built_at', ?)",
                (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),))
            unreadable_total = conn.execute(
                "SELECT COUNT(*) FROM scanned_files WHERE err IS NOT NULL"
            ).fetchone()[0]
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES "
                "('unreadable_files', ?)", (str(unreadable_total),))
    finally:
        conn.close()
    seconds = round(time.time() - t0, 3)
    log(f"[sqlite-index] {files_scanned} file(s) scanned, {files_changed} "
       f"changed, {rows_indexed:,} row(s) indexed ({seconds}s)")
    return {"files_scanned": files_scanned, "files_changed": files_changed,
           "rows_indexed": rows_indexed, "seconds": seconds}


class SqlitePersistedIndex:
    """API-compatible with `PersistedIndex` (`.lookup()`, `.directory`,
    `.manifest`) so `audit_pages`/`assert_pages_persisted` run UNCHANGED
    against either backend."""

    def __init__(self, directory: str, conn: sqlite3.Connection):
        self.directory = directory
        self.conn = conn
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        self.manifest = {
            "schema": SCHEMA, "built_at": meta.get("built_at"),
            "unreadable_files": int(meta.get("unreadable_files", 0) or 0),
        }

    @classmethod
    def open(cls, directory: Optional[str] = None) -> "SqlitePersistedIndex":
        d = directory or default_index_dir()
        p = sqlite_index_path(d)
        if not os.path.exists(p):
            raise PersistedInputsError(
                f"no sqlite persisted-input index at {p}. Build it with "
                "`persisted_inputs.update_index_sqlite(index_dir)`.")
        return cls(d, sqlite3.connect(p))

    def lookup(self, url: str, sha: Optional[str] = None,
              order: Iterable = ("l1", "l0", "l2")) -> Optional[Hit]:
        keys = [_to_signed64(k) for k in page_keys(url)]
        for layer in order:
            row = None
            for key in keys:
                row = self.conn.execute(
                    "SELECT store_id, file_path FROM pages WHERE layer=? "
                    "AND key=?", (layer, key)).fetchone()
                if row:
                    break
            if not row:
                continue
            store_id, file_path = row
            sha_matched = None
            if sha:
                pk = _to_signed64(pair_key(url, sha))
                sha_matched = self.conn.execute(
                    "SELECT 1 FROM shas WHERE layer=? AND key=?",
                    (layer, pk)).fetchone() is not None
            return Hit(url=url, layer=layer, store_id=store_id,
                      store_path=file_path, sha_matched=sha_matched)
        return None

    def close(self) -> None:
        self.conn.close()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="index the registered stores")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--registry", default=None)
    b.add_argument("--corpus-root", default=None)
    b.add_argument("--workers", type=int, default=None,
                   help="parallel file scanners; the build is CPU-bound "
                        "on gzip inflation and json parsing, not on the "
                        "disk. Default min(12, cpu_count), overridable "
                        "with AW_PERSISTED_INDEX_WORKERS.")
    b.add_argument("--full", action="store_true",
                   help="force a full rebuild: ignore the on-disk scan "
                        "cache entirely (still refreshes it for the next "
                        "incremental run). Default is incremental -- only "
                        "new/changed files are re-scanned; a registry edit "
                        "already forces this automatically.")

    c = sub.add_parser("check", help="audit a page list against the index")
    c.add_argument("--pages", required=True,
                   help="JSONL/TSV/text of urls (one per line, or {url,sha})")
    c.add_argument("--index-dir", default=None)
    c.add_argument("--require", default="l1", choices=sorted(REQUIRE_LEVELS))
    c.add_argument("--label", default="")
    c.add_argument("--check-sha", action="store_true",
                   help="also require the capture sha the page list cites to "
                        "be the one the store records (see audit_pages)")

    r = sub.add_parser("resolve", help="resolve ONE url to its store file")
    r.add_argument("--url", required=True)
    r.add_argument("--sha", default=None)
    r.add_argument("--index-dir", default=None)

    args = ap.parse_args(argv)
    try:
        if args.cmd == "build-index":
            build_index(args.out_dir, registry_path=args.registry,
                        root=args.corpus_root, workers=args.workers,
                        incremental=not args.full)
            return 0
        if args.cmd == "resolve":
            idx = PersistedIndex.open(args.index_dir)
            hit = idx.lookup(args.url, args.sha)
            print(json.dumps(dataclasses.asdict(hit) if hit else None, indent=1))
            return 0 if hit else 2
        idx = PersistedIndex.open(args.index_dir)
        pages = _read_page_list(args.pages)
        v = audit_pages(pages, index=idx, require=args.require,
                        label=args.label or args.pages,
                        check_sha=args.check_sha)
        print(json.dumps(v, indent=1))
        return 0 if v["verdict"] == "PASS" else 2
    except PersistedInputsError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _read_page_list(path: str) -> list:
    op = gzip.open if path.endswith(".gz") else open
    out = []
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if line.lstrip().startswith("{"):
                out.append(json.loads(line))
            elif "\t" in line:
                f = line.split("\t")
                out.append({"url": f[0], "sha": f[1] if len(f) > 1 else None})
            else:
                out.append({"url": line})
    return out


if __name__ == "__main__":
    raise SystemExit(main())
