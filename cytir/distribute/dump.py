#!/usr/bin/env python3
"""cytir.distribute.dump — `cyt dump` (task 650, S8; specs/dump-format.md).

MOVED + EXTENDED from the retired pipeline dump script (task 092's scroll-to-
jsonl.gz staging tool), which stays at that path as a shim. The original
tool's `--compact` flag (a deny-list strip via `aw_common.compact_payload`)
is REPLACED here, not reused: `cytir/distribute/pack.py`'s two whitelists
(`project_payload`) are the declared-projection form release-doctrine
requires for anything that leaves our control — a new payload key is
absent from a public dump until `pack.py` names it, never present by
default. the retired pipeline dump script's plain (non-compact) internal
staging->prod jsonl.gz path is UNCHANGED; that ship stays a deny-list
because both sides of it are us.

Reads ONLY from the twin (the build box, `twin_url` setting) — never prod. This is a
hard, unconditional refusal (see `PROD_QDRANT_DENYLIST`), not a convention:
no flag lifts it.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import sys
from typing import Iterator, Optional
from urllib.parse import urlparse

import pathlib as _pathlib

# Started by path (`python aw/...py`), sys.path[0] is this file's own directory, not the repo
# root: put the root first so `import cytir` resolves (pinned by the script start-guard test).
_REPO_ROOT = str(_pathlib.Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cytir.core import serving_profile as _SP
from cytir.core import twin as TWIN
from cytir.core.model_pin import EMBED_MODEL_ID, embed_model_revision
from cytir.distribute import pack
from cytir.distribute.idf import IdfAccumulator
from cytir.distribute.license_check import (
    DEFAULT_THRESHOLD,
    CoverageTally,
    LicenseCoverageRefusal,
)
from cytir.distribute.sign import sign_manifest
from cytir.distribute.torrent import TorrentRefusal, make_torrent

# ---------------------------------------------------------------- twin guard
#: The twin, per `cytir/index/promote_queue.py --twin-url`'s own default.
DEFAULT_TWIN_QDRANT_URL = TWIN.configured_url()   # overlay/default; an env value never redefines the twin here

#: Known prod-tunnel endpoints (`cytir/index/promote_queue.py --prod-url`'s own
#: default). Hard-refused unconditionally — no flag lifts this (release-
#: doctrine #1/#6: a public export never touches prod, full stop).
PROD_QDRANT_DENYLIST = frozenset({"127.0.0.1:16333", "localhost:16333", "127.0.0.1:6334", "localhost:6334"})

#: The serving profile a dump declares when `--profile` is not given. Its width is also
#: the fallback when `--dims` is absent and no point was scrolled (it was a literal 128,
#: the retired width, until slice 8 step 1).
DEFAULT_PROFILE = _SP.PROD_E9.name     # task 1271 (R1): `served` is the pre-e9 shape, still selectable by name

#: specs/dump-format.md §6 — the sparse leg's identity, when one is shipped.
SPARSE_VECTOR_NAME = "bm25"
SPARSE_TOKENIZER = "aw-bm25-v1"
#: The collection's sparse modifier: scores are `stored_weight * idf` (`cytir.distribute.idf`).
SPARSE_MODIFIER = "idf"


class ProdRefusal(RuntimeError):
    pass


class ProfileRefusal(RuntimeError):
    """The scrolled vectors are not the width the declared serving profile creates."""


def assert_qdrant_url_is_not_prod(qdrant_url: str, *,
                                   i_know_this_is_not_the_twin: bool) -> None:
    netloc = urlparse(qdrant_url).netloc.lower()
    if netloc in PROD_QDRANT_DENYLIST:
        raise ProdRefusal(
            f"REFUSED: {qdrant_url!r} is a known prod-tunnel endpoint "
            f"({sorted(PROD_QDRANT_DENYLIST)}). `cyt dump` reads from the "
            f"twin only; this refusal cannot be overridden by any flag.")
    if qdrant_url != DEFAULT_TWIN_QDRANT_URL and not i_know_this_is_not_the_twin:
        raise ProdRefusal(
            f"REFUSED: --qdrant-url {qdrant_url!r} is not the twin default "
            f"({DEFAULT_TWIN_QDRANT_URL!r}). Pass "
            f"--i-know-this-is-not-the-twin to confirm this is a scratch/CI "
            f"Qdrant instance, never prod.")


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


# -------------------------------------------------------------------- scroll
def point_to_row(p) -> dict:
    """One Qdrant point -> one projected, vector-carrying dump row. Refuses
    (via `pack.ProjectionRefusal`) a point with no `dense` vector, rather
    than emitting a row a self-hoster cannot query. The ONE projection door:
    `scroll_rows` and the rehearsal-fixture sampler both go through here."""
    payload = pack.project_payload(p.payload or {})
    vectors = p.vector or {}
    dense = vectors.get("dense")
    if dense is None:
        raise pack.ProjectionRefusal(
            f"point {p.id} has no 'dense' named vector — cannot "
            f"dump a row with no dense embedding")
    sparse = None
    if pack.ships_sparse(payload):
        sv = vectors.get(SPARSE_VECTOR_NAME)
        if sv is not None and hasattr(sv, "indices"):
            sparse = (list(sv.indices), list(sv.values))
    return pack.row_to_record(str(p.id), list(dense), payload, sparse)


def scroll_rows(client, collection: str, *, batch: int = 1000) -> Iterator[dict]:
    """Yield projected + vector-carrying rows, ready for `pack.write_parquet_shard`
    (after batching), from a live Qdrant collection."""
    offset = None
    while True:
        points, offset = client.scroll(collection, limit=batch, offset=offset,
                                       with_payload=True, with_vectors=True)
        if not points:
            break
        for p in points:
            yield point_to_row(p)
        if offset is None:
            break


# -------------------------------------------------------------------- shards
def _row_bytes(row: dict) -> int:
    """Python-object bytes of ONE buffered row (lists of boxed floats/ints
    included). A sampled estimate, computed by the code itself (no tracemalloc)."""
    total = sys.getsizeof(row)
    for v in row.values():
        total += sys.getsizeof(v)
        if isinstance(v, list):
            total += sum(sys.getsizeof(x) for x in v[:1]) * len(v)
    return total


def resident_line(total: int, buf: list[dict], idf: Optional[IdfAccumulator]) -> str:
    """One log line naming the structures `cyt dump` holds right now and how each
    scales: shard buffer (constant, `--shard-rows`), IDF Counter (terms), ids held
    (none: fingerprints stream to a sidecar). Sizes are estimates, not RSS."""
    buf_mb = (_row_bytes(buf[0]) * len(buf) / 1e6) if buf else 0.0
    out = [f"resident after {total:,} rows: shard_buffer={len(buf):,} rows ~{buf_mb:,.0f} MB (constant)",
           "ids_held=0"]
    if idf is not None:
        n = len(idf.df)
        est = (sys.getsizeof(idf.df) + n * 56) / 1e6  # dict table + two boxed ints per term
        out.append(f"idf_terms={n:,} ~{est:,.0f} MB (grows with terms)")
    return "  " + "; ".join(out)


def write_shards(rows: Iterator[dict], out_dir: pathlib.Path, *,
                  shard_rows: int = 50_000,
                  fp_writer: Optional["pack.FingerprintWriter"] = None,
                  idf: Optional[IdfAccumulator] = None,
                  log=None,
                  ) -> tuple[list[dict], int]:
    """Writes `out_dir/shards/shard-NNNNN.parquet`; returns
    (shard manifest entries, total row count). With `fp_writer` (a FULL
    dump), also records every row's fingerprint so the next delta does not
    have to re-read these shards. With `idf`, counts every row into the BM25 IDF
    snapshot accumulator (a FULL dump only; `cytir.distribute.idf`)."""
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict] = []
    buf: list[dict] = []
    total = 0
    shard_idx = 0

    def flush():
        nonlocal shard_idx
        if not buf:
            return
        path = shard_dir / f"shard-{shard_idx:05d}.parquet"
        pack.write_parquet_shard(buf, str(path))
        shards.append({
            "path": f"shards/{path.name}",
            "rows": len(buf),
            "sha256": pack.sha256_file(str(path)),
            "bytes": path.stat().st_size,
        })
        shard_idx += 1
        buf.clear()

    for row in rows:
        buf.append(row)
        total += 1
        if fp_writer is not None:
            fp_writer.add(row["id"], pack.row_fingerprint(row))
        if idf is not None:
            idf.add(row)
        if len(buf) >= shard_rows:
            flush()
        if total % 100_000 == 0:
            print(f"  dumped {total:,} rows...", flush=True)
            if log is not None:
                log(resident_line(total, buf, idf))
    if log is not None:
        log(resident_line(total, buf, idf))
    flush()
    return shards, total


def write_full_shards(rows: Iterator[dict], out_dir: pathlib.Path, *,
                      shard_rows: int = 50_000,
                      idf: Optional[IdfAccumulator] = None,
                      log=None,
                      ) -> tuple[list[dict], int, dict]:
    """The one way to write a FULL dump's rows: shards plus the fingerprint
    sidecar the next delta diffs against. Returns (shards, total,
    manifest `fingerprints` block). Pass an `IdfAccumulator` and, after this
    returns, `idf.write(out_dir)` for the manifest's `sparse.idf_snapshot`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_writer = pack.FingerprintWriter(
        str(out_dir / pack.FINGERPRINTS_FILENAME))
    shards, total = write_shards(rows, out_dir, shard_rows=shard_rows,
                                 fp_writer=fp_writer, idf=idf, log=log)
    return shards, total, fp_writer.close()


# ------------------------------------------------------------------ manifest
def _sparse_block(any_row_has_sparse: bool, idf_snapshot: Optional[dict]) -> Optional[dict]:
    if not any_row_has_sparse:
        return None
    return {
        "vector_name": SPARSE_VECTOR_NAME,
        "tokenizer": SPARSE_TOKENIZER,
        "modifier": SPARSE_MODIFIER,
        "idf_snapshot": idf_snapshot,
    }


def build_manifest(*, dump_id: str, collection: str, dims: int, distance: str,
                    shards: list[dict], total_points: int,
                    any_row_has_sparse: bool,
                    idf_snapshot: Optional[dict] = None,
                    fingerprints: Optional[dict] = None,
                    profile) -> dict:
    """`profile` is REQUIRED (keyword-only, no default): a `ServingProfile`, the layout
    `cyt apply` creates for these rows. Its name, digest and full declaration go in the
    manifest's `profile` block. Pass an explicit `None` only to write a legacy manifest
    (no block); `cyt apply` refuses that one without `--legacy-layout`."""
    manifest = {
        "schema": pack.DUMP_MANIFEST_SCHEMA,
        "dump_id": dump_id,
        "created_at": now_iso(),
        "source": {"collection": collection, "qdrant_role": "twin"},
        "embedder": {
            "model_id": EMBED_MODEL_ID,
            "revision": embed_model_revision(),
            "dims": dims,
            "quantization": "float32",
            "distance": distance,
        },
        "sparse": _sparse_block(any_row_has_sparse, idf_snapshot),
        "counts": {"points": total_points, "shards": len(shards)},
        "shards": shards,
        "license": {
            "database": "ODC-By-1.0",
            "database_url": "https://opendatacommons.org/licenses/by/1-0/",
            "notice": (
                "authoritative-web index compilation (c) impactIA. "
                "Attribution required for the compilation; each record also "
                "carries its own `license` field, which governs the source "
                "content at that record's `url` and is NOT a distribution "
                "gate (ADR-112)."),
        },
    }
    if fingerprints is not None:
        manifest["fingerprints"] = fingerprints
    if profile is not None:
        if profile.dense_dim != dims:
            raise ProfileRefusal(
                f"profile {profile.name!r} is {profile.dense_dim}-d, the dump is {dims}-d")
        manifest["profile"] = profile.block()
    return manifest


# ----------------------------------------------------------------- delta diff
class DeltaRefusal(RuntimeError):
    pass


def _base_sidecar(base_manifest: dict, base_dir: pathlib.Path,
                  ledger_base=None) -> tuple[str, int]:
    """The base's checked fingerprint sidecar `(path, count)`; refuses a base without one.
    THE one place both delta builders get their base. With a ledger (`ledger_base`, a
    `fingerprint_ledger.Base` already verified by `base_for_delta`) the base is the ledger's
    sidecar: the cumulative state after the last published pack, not the full's own."""
    if ledger_base is not None:
        return ledger_base.path, ledger_base.count
    block = base_manifest.get("fingerprints")
    if not block:
        raise DeltaRefusal(
            "the base manifest has no `fingerprints` block, so a delta "
            "cannot be diffed without re-reading every base shard. "
            "Re-run `cyt dump` for the base (task 687).")
    if block.get("algo") != pack.FINGERPRINT_ALGO:
        raise DeltaRefusal(
            f"the base fingerprints use algorithm {block.get('algo')!r}, "
            f"this build uses {pack.FINGERPRINT_ALGO!r}; the two are not "
            f"comparable. Re-run `cyt dump` for the base (task 687).")
    path = str(base_dir / block["path"])
    pack.check_fingerprints_sidecar(path, expected_sha256=block["sha256"])
    return path, block["count"]


class DeltaBuilder:
    """Delta against a full dump's declared per-row fingerprints
    (`manifest["fingerprints"]`, written by `write_shards(fp_writer=...)`).
    The base shards are never opened: `__init__` reads one sidecar, and the
    only rows hashed afterwards are the ones the current snapshot streams
    past it. A base manifest with no `fingerprints` block (pre-task-687) is
    refused, not re-hashed: one door, and the fix is to re-dump the base.

    MERGE-JOIN, bounded memory (task 687 refactor). The sidecar and the
    scrolled snapshot are both in point-id order (`pack.point_id_key`), so the
    two streams are walked together: peak memory does not grow with the
    corpus (it holds one sidecar line, one row, and the ids that vanished,
    which are the tombstones the pack must carry anyway). An input that is
    out of order or repeats an id is refused, never mislabelled.
    No write-time change log exists yet, so the current snapshot is still
    scrolled in full (that scroll, not the base, scales with the corpus)."""

    def __init__(self, base_manifest: dict, base_dir: pathlib.Path, ledger_base=None):
        self._path, self._count = _base_sidecar(base_manifest, base_dir, ledger_base)
        self._gone: list[str] = []
        self._joined = False

    def changed_rows(self, rows: Iterator[dict]) -> Iterator[dict]:
        """Yields only new-or-changed rows. Base ids the snapshot passes
        without meeting are collected for `tombstones()`, which may be called
        once this iterator is FULLY consumed."""
        base = pack.iter_fingerprints(self._path, expected_count=self._count)
        cur = next(base, None)
        last = None
        for row in rows:
            id_ = row["id"]
            key = pack.point_id_key(id_)
            if last is not None and not key > last:
                raise DeltaRefusal(
                    f"the current snapshot is not in strictly increasing "
                    f"point-id order at {id_!r} (repeat or out-of-order "
                    f"scroll); the streamed diff would mislabel rows.")
            last = key
            while cur is not None and cur[2] < key:
                self._gone.append(cur[0])  # in the base, absent now
                cur = next(base, None)
            if cur is not None and cur[2] == key:
                unchanged = cur[1] == pack.row_fingerprint(row)
                cur = next(base, None)
                if unchanged:
                    continue  # unchanged — not part of the delta
            yield row
        while cur is not None:
            self._gone.append(cur[0])
            cur = next(base, None)
        self._joined = True

    def tombstones(self, reasons: dict[str, str]) -> list["pack.Tombstone"]:
        """Call only after `changed_rows`'s iterator has been FULLY
        consumed. Refuses (does not guess) a vanished id with no declared
        reason."""
        if not self._joined:
            raise DeltaRefusal(
                "tombstones() called before changed_rows() was fully consumed")
        gone = self._gone
        missing_reasons = sorted(i for i in gone if i not in reasons)
        if missing_reasons:
            raise DeltaRefusal(
                f"{len(missing_reasons)} id(s) present in the base dump are "
                f"absent from the current twin with no declared tombstone "
                f"reason (first 5: {missing_reasons[:5]}). Pass "
                f"--tombstone-reasons mapping every vanished id to "
                f"superseded|aged_out|takedown|superseded_generation; `cyt dump` never guesses why "
                f"a point disappeared.")
        return [pack.Tombstone(id=i, reason=reasons[i])
                for i in gone]


# ------------------------------------------------- delta from the change log
#: Where a delta's rows came from, recorded in its manifest as `delta_source.path`.
SOURCE_CHANGE_LOG = "change-log"
SOURCE_SCROLL = "scroll"
def _segment_stamp(name: str) -> str | None:
    from cytir.index import change_log as CL   # one name pattern for builder and completeness check
    return CL.segment_stamp(name)


def latest_segment_stamp(log_dir: pathlib.Path | str | None) -> str | None:
    """The landing stamp of the newest segment in `log_dir`, or None (log off or empty). The cut
    point a dump records as `change_log_segment`: a stamp only, never the pack id in the file name."""
    if not log_dir or not pathlib.Path(log_dir).is_dir():
        return None
    stamps = [x for x in map(_segment_stamp, (f.name for f in pathlib.Path(log_dir).iterdir())) if x]
    return max(stamps) if stamps else None


def _check_width(dims_measured: list, row: dict) -> None:
    """One embedder width per dump (specs/dump-format.md 1); shared by the scroll and log paths."""
    if not dims_measured:
        dims_measured.append(row["dense_dims"])
    elif row["dense_dims"] != dims_measured[0]:
        raise ValueError(
            f"mixed dense width in one dump: {dims_measured[0]} vs "
            f"{row['dense_dims']} (point {row['id']}) — a dump ships "
            f"one embedder layout, specs/dump-format.md §1")


def _point_id(id_: str):
    """Qdrant wants an int for a numeric point id and the text for a UUID."""
    return int(id_) if id_.isascii() and id_.isdigit() else id_


class LogDeltaBuilder:
    """Delta from the drain's change-log segments (task 2504, ruling 2497 option (c);
    `specs/delta-change-log.md` 3.4). Cost is O(changes), not O(corpus).

    1. Read the segments with `seq` above the cut (`after` = the base's `change_log_segment`,
       `{stamp, seq}`); net them per id in `seq` order, last line wins (`add`/`replace`/`upsert` =
       present, `tombstone` = gone). A segment without `seq`, a duplicate or a gap in `seq` is
       REFUSED, naming the segment. A `rebase` line above the cut is REFUSED (spec 3.5): the base
       is no ancestor. A `rebase` at or below the cut is skipped by `seq`.
    2. Fetch the net-present ids from the source collection by id, project with `point_to_row`.
       An id the log says is present but the source lacks is REFUSED: a bypass or a twin loss.
    3. `pack.row_fingerprint` each fetched row and compare with the base sidecar (one streamed
       pass, memory O(changes)): equal = net no-op, dropped and counted.
    4. Net tombstones: only ids in the base ship (an id added and tombstoned inside the window
       never reached a mirror). Output carries the public `reason` only: `queue_reason`, if the
       segment holds one, is master-only and never read here.

    The log carries no fingerprints on purpose: a point as the drain built it is not the stored
    row (float16 storage, cosine normalisation), so only the fetched row hashes comparably."""

    def __init__(self, base_manifest: dict, base_dir: pathlib.Path, log_dir: pathlib.Path | str, *,
                 client, collection: str, after: dict, batch: int = 500, ledger_base=None):
        from cytir.index import change_log as CL
        self._sidecar, self._count = _base_sidecar(base_manifest, base_dir, ledger_base)
        self._client, self._collection, self._batch = client, collection, batch
        self.after = after
        present: dict[str, None] = {}
        gone: dict[str, str] = {}
        self.segments: list[str] = []
        cut_stamp, cut_seq = after["stamp"], after["seq"]
        found: list[tuple[int, str, pathlib.Path]] = []
        for f in pathlib.Path(log_dir).iterdir():
            stamp = _segment_stamp(f.name)
            if not stamp or stamp < cut_stamp:   # older than the cut by name: never read
                continue
            seq = CL._header_seq(f)
            if seq is None:
                raise DeltaRefusal(
                    f"segment {f.name} has no `seq` in its header: the order of the log cannot be "
                    f"proven (a segment from before task 2508, or a rewritten file). Cut a full dump.")
            if seq > cut_seq:
                found.append((seq, f.name, f))
            elif stamp > cut_stamp:
                raise DeltaRefusal(
                    f"segment {f.name} is stamped after the cut ({cut_stamp}) but holds seq {seq} <= "
                    f"the cut seq {cut_seq}: the change-log counter went back (a log reset or wipe "
                    f"under a live cut). Its changes cannot be placed; cut a full dump.")
        found.sort()
        self.last_cut = dict(after)
        want = cut_seq + 1
        for seq, name, f in found:
            if seq != want:
                raise DeltaRefusal(
                    f"change-log seq is not contiguous at segment {name}: expected {want}, found "
                    f"{seq} ({'duplicate' if seq < want else 'gap'}). A segment is missing or was "
                    f"compacted away; the delta cannot be proven complete. Cut a full dump.")
            want += 1
            _header, body = CL.read_segment(f)
            self.segments.append(name)
            self.last_cut = {"stamp": _segment_stamp(name), "seq": seq}
            for line in body:
                op = line["op"]
                if op == "rebase":
                    raise DeltaRefusal(
                        f"segment {name} holds a `rebase` line ({line.get('why')!r}): the base is "
                        f"no ancestor of the twin. Cut a full dump (spec 3.5); a delta across a "
                        f"rebase is refused, never approximated.")
                id_ = line["id"]
                if op == "tombstone":
                    present.pop(id_, None)
                    gone[id_] = line.get("reason", "")
                else:
                    gone.pop(id_, None)
                    present[id_] = None
        self._present = sorted(present, key=pack.point_id_key)
        self._gone = sorted(gone, key=pack.point_id_key)
        self._gone_reason = gone
        self._base_fp: dict[str, str] | None = None
        self.noops = 0
        self.fetched = 0
        self._done = False

    def _load_base(self) -> None:
        """One sequential pass over the sidecar, keeping only the ids the log names."""
        want = {*self._present, *self._gone}
        self._base_fp = {id_: fp for id_, fp, _ in
                         pack.iter_fingerprints(self._sidecar, expected_count=self._count)
                         if id_ in want}

    def _present_in_source(self, ids: list[str]) -> set[str]:
        got: set[str] = set()
        for i in range(0, len(ids), self._batch):
            part = ids[i:i + self._batch]
            got.update(str(p.id) for p in self._client.retrieve(
                self._collection, ids=[_point_id(x) for x in part],
                with_payload=False, with_vectors=False))
        return got

    def preflight(self) -> None:
        """Before any shard is written: (1) every net-present id exists in the source, (2) no
        net-deleted base id is still there. One id lookup per id, bounded by the log. It makes
        "no shard written" true for these refusals, and (2) catches a wrong net order (a delete
        replayed after its re-add) that would ship a tombstone for a live row."""
        if self._base_fp is None:
            self._load_base()
        lost = sorted(set(self._present) - self._present_in_source(self._present),
                      key=pack.point_id_key)
        if lost:
            raise DeltaRefusal(
                f"the change log names id(s) as present that the source collection "
                f"{self._collection!r} does not hold (first 5: {lost[:5]}). That is a write "
                f"that bypassed the drain or a twin loss. No shard was written. Fix the twin, or "
                f"cut a full dump.")
        shipped = [i for i in self._gone if i in self._base_fp]
        alive = sorted(self._present_in_source(shipped), key=pack.point_id_key)
        if alive:
            raise DeltaRefusal(
                f"{len(alive)} id(s) the log nets to deleted are still in the source collection "
                f"{self._collection!r} (first 5: {alive[:5]}). A tombstone would delete a live row. "
                f"No shard was written. Cut a full dump.")

    def changed_rows(self) -> Iterator[dict]:
        if self._base_fp is None:
            self._load_base()
        self.preflight()
        missing: list[str] = []
        for i in range(0, len(self._present), self._batch):
            ids = self._present[i:i + self._batch]
            got = {str(p.id): p for p in self._client.retrieve(
                self._collection, ids=[_point_id(x) for x in ids],
                with_payload=True, with_vectors=True)}
            for id_ in ids:
                p = got.get(id_)
                if p is None:
                    missing.append(id_)
                    continue
                row = point_to_row(p)
                self.fetched += 1
                if self._base_fp.get(id_) == pack.row_fingerprint(row):
                    self.noops += 1       # net no-op: same row as the base holds
                    continue
                yield row
            if missing:                   # refuse at the first batch that shows a gap
                break
        if missing:
            raise DeltaRefusal(
                f"the change log names id(s) as present that the source collection "
                f"{self._collection!r} does not hold (first 5: {missing[:5]}). That is a write "
                f"that bypassed the drain or a twin loss (a write landed after the preflight); shard "
                f"files already written stay in the output directory and NO manifest was written. "
                f"Fix the twin, or cut a full dump.")
        self._done = True

    def tombstones(self) -> list["pack.Tombstone"]:
        """After `changed_rows` is fully consumed. Public `reason` only; a reason outside
        `pack.TOMBSTONE_REASONS` is refused; there is no override on the log path (task 2512)."""
        if not self._done:
            raise DeltaRefusal("tombstones() called before changed_rows() was fully consumed")
        out, bad = [], []
        for id_ in self._gone:
            if id_ not in self._base_fp:
                continue                      # never in the mirror
            reason = self._gone_reason[id_]
            if reason not in pack.TOMBSTONE_REASONS:
                # `read_segment` already refuses a non-public reason; kept as a second lock.
                bad.append((id_, reason))
                continue
            out.append(pack.Tombstone(id=id_, reason=reason))
        if bad:
            raise DeltaRefusal(
                f"{len(bad)} tombstone line(s) carry a reason that is not public "
                f"(first 3: {bad[:3]}); the change log holds public reasons only. Cut a full dump.")
        return out


def choose_delta_source(base_manifest: dict, base_dir: pathlib.Path, *, log_dir, client,
                        collection: str,
                        use_log: bool = True, ledger_base=None) -> tuple["LogDeltaBuilder | None", str]:
    """`(builder, note)`: the log reader, or `(None, why)` when the scroll path must run. The note is
    printed and recorded in the manifest. A scroll fallback is chosen only for a missing prerequisite
    (no log dir, no cut point, no sidecar, no new segment); a log that is present and wrong REFUSES."""
    if not use_log:
        return None, "change log disabled (--no-change-log)"
    if not log_dir or not pathlib.Path(log_dir).is_dir():
        return None, "no change-log directory"
    after = base_manifest.get("change_log_segment")
    if not after:
        return None, "the base manifest records no change_log_segment (first delta after the log was switched on)"
    if not (isinstance(after, dict) and isinstance(after.get("seq"), int) and after.get("stamp")):
        return None, f"the base change_log_segment {after!r} is not {{stamp, seq}} (a dump from before task 2508)"
    block = base_manifest.get("fingerprints")
    if ledger_base is None and (not block or not (base_dir / block.get("path", "")).is_file()):
        return None, "the base fingerprint sidecar is missing"
    # Task 2588: a log whose newest segment is missing REFUSES (MissingSegmentError is a
    # ChangeLogError, printed by cli()); it is never a reason for the silent scroll fallback.
    from cytir.index import change_log as _CL
    _CL.assert_segments_complete(log_dir, after["seq"], after["stamp"])
    b = LogDeltaBuilder(base_manifest, base_dir, log_dir, client=client, collection=collection,
                        after=after, ledger_base=ledger_base)
    if not b.segments:
        return None, f"no change-log segment after seq {after['seq']}"
    return b, f"{len(b.segments)} segment(s) after seq {after['seq']}"


# ----------------------------------------------------------------------- CLI
def _configured_ledger():
    """The fingerprint ledger named by the `fingerprint_ledger_root` setting, or None (off)."""
    from cytir.core.paths import SettingRefused, operator_settings
    try:
        root = operator_settings().fingerprint_ledger_root
    except SettingRefused:
        return None
    if not root:
        return None
    from cytir.distribute.fingerprint_ledger import Ledger
    return Ledger(root)


def _lock_ledger(ledger, stack) -> None:
    """Single-flight on the ledger root (review 2591 R1): an exclusive non-blocking flock on
    `<root>/.lock`, held by `stack` until the dump is done (after the advance). A second dump
    REFUSES (LedgerError) instead of reading a sidecar the first one is replacing."""
    import fcntl
    from cytir.distribute.fingerprint_ledger import LedgerError
    ledger.root.mkdir(parents=True, exist_ok=True)
    fh = stack.enter_context(open(ledger.root / ".lock", "a"))
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise LedgerError(f"ledger {ledger.root} is held by another dump (single-flight lock "
                          f"{ledger.root / '.lock'}); nothing was written") from None


def main(argv: list[str] | None = None) -> int:
    import contextlib
    with contextlib.ExitStack() as stack:
        return _main(argv, stack)


def _main(argv, stack) -> int:
    ap = argparse.ArgumentParser(
        prog="cyt dump",
        description="export a Qdrant collection (twin only) into the "
                     "public dump format — specs/dump-format.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit codes:\n"
               "  0  done\n"
               "  1  REFUSED: nothing written (bad input, ledger missing/stale/locked, delta refusal)\n"
               "  3  LEDGER LAGGING: pack signed and kept; repair with rebuild_by_fold before the next delta\n"
               "  4  LEDGER CONFLICT: the ledger moved during the build; do NOT publish this pack")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--profile", default=DEFAULT_PROFILE,
                     help="serving profile this dump declares (name + digest go in the "
                          "manifest; `cyt apply` creates its collection from it). "
                          "Default: %(default)s (what prod runs; `served` is the pre-e9 shape)")
    ap.add_argument("--qdrant-url", default=DEFAULT_TWIN_QDRANT_URL)
    ap.add_argument("--i-know-this-is-not-the-twin", action="store_true",
                     help="required to use any --qdrant-url other than the "
                          "twin default; a known prod endpoint is refused "
                          "regardless of this flag")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--dump-id", default=None,
                     help="default: UTC timestamp + '-full' or '-delta'")
    ap.add_argument("--shard-rows", type=int, default=50_000)
    ap.add_argument("--sign-key", default=None,
                     help="minisign secret key path; omit to ship unsigned")
    ap.add_argument("--pubkey-id", default=None,
                     help="the minisign public key id (informational, "
                          "recorded in the manifest's signature block); the "
                          "key itself is published out of band, never "
                          "inside the dump — specs/dump-format.md §6")
    ap.add_argument("--torrent", action="store_true",
                     help="after signing, also write <out>.torrent for the "
                          "finished directory (`cyt torrent make`)")
    ap.add_argument("--license-threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--dims", type=int, default=None,
                     help="default: measured from the first scrolled point")
    # --- delta mode ---
    ap.add_argument("--delta-base-manifest", default=None,
                     help="path to a previous manifest.json; presence of "
                          "this flag switches to delta-pack mode")
    ap.add_argument("--delta-base-dir", default=None,
                     help="directory the base manifest's shard paths are "
                          "relative to (default: dirname of "
                          "--delta-base-manifest)")
    ap.add_argument("--tombstone-reasons", default=None,
                     help="JSON file: {id: 'superseded'|'aged_out'|"
                          "'takedown'|'superseded_generation'} for every id the delta finds gone "
                          "from the twin; required in delta mode if any id "
                          "vanished")
    ap.add_argument("--from-change-log", default=None, metavar="DIR",
                     help="delta mode: read the drain's change-log segments in DIR instead of "
                          "scrolling the collection (task 2504). Default: AW_CHANGE_LOG_DIR, when "
                          "set. The base manifest must record `change_log_segment`; otherwise, or "
                          "with no segment after it, the scroll path runs and the manifest says so")
    ap.add_argument("--no-change-log", action="store_true",
                     help="delta mode: force the scroll path (the audit of the log path, spec 3.7)")
    ap.add_argument("--sequence", type=int, default=1,
                     help="delta mode: this pack's monotonic sequence number")
    args = ap.parse_args(argv)

    from cytir.core import serving_profile as SP
    try:
        profile = SP.get(args.profile)
    except KeyError as e:
        ap.error(str(e.args[0]))

    from cytir.core.qdrant import qdrant_client  # lazy: CLI-only dependency

    assert_qdrant_url_is_not_prod(
        args.qdrant_url,
        i_know_this_is_not_the_twin=args.i_know_this_is_not_the_twin)

    if args.dims is not None and args.dims != profile.dense_dim:
        raise ProfileRefusal(
            f"--dims {args.dims} but profile {profile.name!r} is {profile.dense_dim}-d "
            f"(pick the profile of that width; nothing was exported)")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = qdrant_client(args.qdrant_url, timeout=300)
    info = client.get_collection(args.collection)
    try:
        coll_dim = info.config.params.vectors["dense"].size
    except Exception:  # noqa: BLE001 -- unreadable: the first-row check below still holds
        coll_dim = None
    if coll_dim is not None and coll_dim != profile.dense_dim:
        raise ProfileRefusal(
            f"collection {args.collection!r} holds {coll_dim}-d vectors but profile "
            f"{profile.name!r} is {profile.dense_dim}-d (pick the profile of that "
            f"width; nothing was exported)")
    try:
        distance = str(info.config.params.vectors["dense"].distance).lower()
        if "." in distance:
            distance = distance.split(".")[-1]
    except Exception:  # noqa: BLE001 — best-effort; the spec default stands
        distance = "cosine"

    print(f"{args.collection} @ {args.qdrant_url} -> {out_dir}", flush=True)
    rows_iter = scroll_rows(client, args.collection)

    delta_mode = bool(args.delta_base_manifest)
    ledger = _configured_ledger()   # None = off: a delta diffs against the base manifest, as before
    ledger_base = None
    if ledger is not None:
        _lock_ledger(ledger, stack)   # before the base is read (delta) / the first shard (full)
    dims_measured: list[int] = []
    sparse_seen = [False]

    def instrumented():
        for row in rows_iter:
            _check_width(dims_measured, row)
            if row["sparse_indices"] is not None:
                sparse_seen[0] = True
            yield row

    tally = CoverageTally()

    log_dir = args.from_change_log
    if log_dir is None:
        from cytir.index import change_log as _CL
        cfg = _CL.configured_dir()
        log_dir = str(cfg) if cfg else None
    # The cut point is read BEFORE any row is read: a segment that lands during the build is
    # replayed by the next delta (a double entry is harmless, spec 3.7), never skipped.
    from cytir.index import change_log as _CL0
    cut = _CL0.latest_cut(log_dir)

    if delta_mode:
        base_manifest = pack.validate_manifest(
            json.loads(pathlib.Path(args.delta_base_manifest).read_text()))
        base_dir = pathlib.Path(args.delta_base_dir or
                                pathlib.Path(args.delta_base_manifest).parent)
        if ledger is not None:
            # a stale / corrupt / rebased / missing ledger REFUSES the dump (LedgerError): no
            # fall back to the base manifest's sidecar, which is the full's, not the last pack's
            from cytir.distribute.fingerprint_ledger import LedgerMissing, base_for_delta
            try:
                ledger_base = base_for_delta(
                    ledger, expected_manifest_sha256=pack.sha256_file(args.delta_base_manifest),
                    since_seq=args.sequence - 1)
            except LedgerMissing as e:
                raise LedgerMissing(
                    f"{e}. Seed it: cytir.distribute.fingerprint_ledger.rebuild_by_fold(<packs dir>, "
                    f"Ledger(<root>)), or publish the next FULL with the ledger on (that seeds it)"
                ) from None
            print(f"delta base: fingerprint ledger (seq {ledger_base.sequence}, "
                  f"{ledger_base.count:,} rows)", flush=True)
        reasons = {}
        if args.tombstone_reasons:
            reasons = json.loads(pathlib.Path(args.tombstone_reasons).read_text())
        log_builder, source_note = choose_delta_source(
            base_manifest, base_dir, log_dir=log_dir, client=client,
            collection=args.collection, use_log=not args.no_change_log,
            ledger_base=ledger_base)
        source_path = SOURCE_CHANGE_LOG if log_builder else SOURCE_SCROLL
        print(f"delta source: {source_path} ({source_note})", flush=True)
        builder = log_builder or DeltaBuilder(base_manifest, base_dir, ledger_base)
        if log_builder:
            cut = log_builder.last_cut

        def tallied_delta_rows():
            rows = (log_builder.changed_rows() if log_builder
                    else builder.changed_rows(instrumented()))
            for row in rows:
                _check_width(dims_measured, row)
                if row["sparse_indices"] is not None:
                    sparse_seen[0] = True
                tally.add(json.loads(row["payload"]))
                yield row

        # write_shards fully drains the generator (streaming, bounded to
        # `--shard-rows` in memory at a time) before either check below runs
        # — both need the FULL pass to have happened first: the coverage
        # tally over every yielded row, and `builder.seen_ids` over every
        # base id the current snapshot did or didn't re-encounter.
        shards, total = write_shards(tallied_delta_rows(), out_dir,
                                     shard_rows=args.shard_rows,
                                     log=lambda line: print(line, flush=True))
        coverage = tally.assert_ok(threshold=args.license_threshold)
        print(f"license coverage (new/changed rows): {coverage['pct']:.1%}")
        tombstones = (builder.tombstones() if log_builder else builder.tombstones(reasons))
        n_tomb, tomb_sha = pack.write_tombstones(
            tombstones, str(out_dir / "tombstones.jsonl.gz"))
        dims = args.dims or (dims_measured[0] if dims_measured else profile.dense_dim)
        dump_id = args.dump_id or (now_iso().replace(":", "-") + "-delta")
        manifest = build_manifest(
            dump_id=dump_id, collection=args.collection, dims=dims,
            distance=distance, shards=shards, total_points=total,
            any_row_has_sparse=sparse_seen[0], profile=profile)
        manifest["schema"] = pack.DELTA_PACK_SCHEMA
        manifest["pack_id"] = dump_id
        manifest["sequence"] = args.sequence
        manifest["base_dump_id"] = base_manifest["dump_id"]
        manifest["base_manifest_sha256"] = pack.sha256_file(args.delta_base_manifest)
        manifest["tombstones"] = {
            "path": "tombstones.jsonl.gz", "count": n_tomb, "sha256": tomb_sha}
        manifest["delta_source"] = {"path": source_path, "note": source_note}
        if log_builder:
            manifest["delta_source"]["no_ops_dropped"] = log_builder.noops
        print(f"delta pack: {total:,} new/changed row(s), {n_tomb:,} tombstone(s)")
    else:
        def tallied_rows():
            for row in instrumented():
                tally.add(json.loads(row["payload"]))
                yield row

        idf_acc = IdfAccumulator()
        shards, total, fingerprints = write_full_shards(
            tallied_rows(), out_dir, shard_rows=args.shard_rows, idf=idf_acc,
            log=lambda line: print(line, flush=True))
        idf_snapshot = idf_acc.write(out_dir)
        coverage = tally.assert_ok(threshold=args.license_threshold)
        print(f"license coverage: {coverage['pct']:.1%}")
        dims = args.dims or (dims_measured[0] if dims_measured else profile.dense_dim)
        dump_id = args.dump_id or (now_iso().replace(":", "-") + "-full")
        manifest = build_manifest(
            dump_id=dump_id, collection=args.collection, dims=dims,
            distance=distance, shards=shards, total_points=total,
            any_row_has_sparse=sparse_seen[0], idf_snapshot=idf_snapshot,
            fingerprints=fingerprints, profile=profile)
        print(f"full dump: {total:,} row(s), {len(shards)} shard(s)")

    if cut:
        manifest["change_log_segment"] = cut   # {stamp, seq}: the NEXT delta replays seq > cut.seq
    # The signature block (if any) is written INTO manifest.json BEFORE the
    # manifest is signed — its own content (algo/sig_file name/pubkey_id) is
    # knowable up front, deterministically, with no chicken-and-egg. Mutating
    # manifest.json AFTER computing the signature would silently invalidate
    # it (the bytes minisign signed would no longer be the bytes on disk),
    # so this order is load-bearing, not stylistic.
    manifest_path = out_dir / "manifest.json"
    if args.sign_key:
        manifest["signature"] = {
            "algo": "minisign",
            "sig_file": manifest_path.name + ".minisig",
            **({"pubkey_id": args.pubkey_id} if args.pubkey_id else {}),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    if args.sign_key:
        sig_path = sign_manifest(str(manifest_path), args.sign_key)
        print(f"signed -> {sig_path}")
    else:
        print("UNSIGNED dump (no --sign-key) — `cyt apply` requires "
              "--allow-unsigned to accept it", file=sys.stderr)

    lag = conflict = None
    if ledger is not None:
        # AFTER the signature: the pack is final. A failure here never un-signs or deletes it;
        # the ledger lags (the next delta refuses as stale) and the repair is `rebuild_by_fold`.
        from cytir.distribute.fingerprint_ledger import (
            LedgerChainError, LedgerError, advance_from_pack_dir)
        try:
            st = advance_from_pack_dir(ledger, out_dir)["state"]
            print(f"fingerprint ledger -> seq {st['sequence']}, {st['count']:,} rows")
        except LedgerChainError as e:
            conflict = f"{e}"   # the ledger moved during the build: this pack is a FORK, not a lag
        except (LedgerError, OSError, ValueError) as e:
            lag = f"{type(e).__name__}: {e}"

    if args.torrent:
        data, report = make_torrent(out_dir)  # after signing: the .minisig is in the file list
        torrent_path = pathlib.Path(str(out_dir.resolve()) + ".torrent")
        torrent_path.write_bytes(data)
        print(f"torrent -> {torrent_path} (infohash {report['infohash']})")

    print(f"DONE -> {manifest_path}")
    if conflict:
        print(f"LEDGER CONFLICT: do not publish {out_dir}; it conflicts with the ledger "
              f"({conflict}). The pack is a fork of the published chain; discard it and rebuild. "
              f"rebuild_by_fold would refuse it.", file=sys.stderr)
        return 4
    if lag:
        print(f"LEDGER LAGGING: the pack above is signed and complete, but the fingerprint ledger "
              f"did not advance ({lag}). The next delta will refuse as stale. Repair: "
              f"cytir.distribute.fingerprint_ledger.rebuild_by_fold(<packs dir>, Ledger(<root>)).",
              file=sys.stderr)
        return 3
    return 0


def cli(argv: list[str] | None = None) -> int:
    """`main` with every refusal printed as `REFUSED: ...` (never a traceback)."""
    from cytir.distribute.fingerprint_ledger import LedgerError
    from cytir.index.change_log import ChangeLogError
    try:
        return main(argv)
    except (ProdRefusal, ProfileRefusal, LicenseCoverageRefusal, DeltaRefusal,
            TorrentRefusal, ChangeLogError, LedgerError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        print("note: --out may hold partial shard files from before this "
              "refusal fired; `manifest.json` was never written, so `cyt "
              "apply` cannot treat this directory as a valid dump — remove "
              "it (or use a fresh --out) before retrying.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
