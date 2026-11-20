"""cytir.index.build -- `cyt index build` and `cyt index apply` (slice 8 step 2,
specs/aw-index-serve-gate.md section 2).

A stranger builds an index with the SAME code our hourly drain runs, not a second path:

    compose_pack (verify every id under the family's declared scheme, stamp consent, write the
    manifest)  ->  promote_queue.drain(transaction=Single(target))  (plan, project, upsert)

The two paths differ only in the transaction, at the four places `promote_queue.Single` names.
This module adds what is NOT a transaction: refuse bad input before anything is written, create the
collection from ONE declared `ServingProfile` through the one creator, finalize it, and write the
receipt (`build-manifest.json`, schema `aw.index-build.v1`) with the row-fingerprint sidecar.

INPUT is the staged chunk stage `cytir.sources.family_stage` writes, with the vectors `cyt embed shards`
wrote beside it. The build NEVER embeds: one embed door feeds both paths.

    <stage>/PACK.json        {"family_id": ..., "pack_id": ..., ...}
    <stage>/chunks/shard-NNNN.texts.jsonl.gz
    <stage>/vectors/shard-NNNN.{vectors.f16,vectors.manifest.json,row_ids.jsonl}

    cyt source run FAMILY ...                       # -> pack
    python -m cytir.sources.family_stage stage ...    # -> <stage>/chunks (+ L1 filing)
    cyt embed shards --in <stage>/chunks --out <stage>/vectors
    cyt index build --pack <stage> --qdrant http://127.0.0.1:6333 --collection mine --out RUN_DIR
    cyt index apply --pack <stage2> --qdrant ... --collection mine --out RUN_DIR2   # a later delta

A second INPUT to the same door, an existing collection streamed through the payload projection and
the prune cut into a new collection of profile PROD_E9 (how prod's collection was built; task 1256):

    cyt index build --from-collection URL:COLLECTION --qdrant ... --collection mine --out RUN_DIR

Exit codes: 0 done; 1 a pack was refused by the mechanical gate (named in the receipt); 2 usage;
3 an input refusal (nothing written to the collection).

BESIDE THE RECEIPT a build leaves the row-fingerprint sidecar, the BM25 IDF snapshot over the rows it
wrote (`bm25-idf.tsv.gz`) and the qstop IDF TABLE measured on the built collection (`bm25_idf.json`,
design question Q4: a stranger measures IDF per build). `cyt serve --idf-table RUN_DIR/bm25_idf.json`
takes it; prod's own table is never touched.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass

from cytir.core import serving_profile as SP
from cytir.core.model_pin import assert_pin
from cytir.core.qdrant import qdrant_client
from cytir.index import compose as COMPOSE
from cytir.index import layout as LAYOUT
from cytir.index import live_target as LT
from cytir.index import promote_queue as PQ
from cytir.index import twin_write_hold as TWH

SCHEMA = "aw.index-build.v1"
EXIT_OK, EXIT_GATE, EXIT_USAGE, EXIT_REFUSED = 0, 1, 2, 3

#: The only profile a stranger may name. The legacy layouts stay private names for their shims.
PUBLIC_PROFILES = ("prod-e9", "served")
#: What `--pack` builds when `--profile` is not given: PROD_E9, the shape prod serves (task 1271, ruling
#: R1 from the 1253 map: no code path defaults to the arm shape -- `served` is the pre-e9 shape, m=16 and
#: float16 sparse -- again). `served` stays selectable by name.
DEFAULT_PACK_PROFILE = SP.PROD_E9.name
#: The DECLARED collection (`data/collection-payload-profiles.v1.json`) whose payload form a
#: collection built from a profile stores: what prod stores, so the built rows equal prod's rows.
PAYLOAD_PROFILE_OF = {"served": "aw_e9", "prod-e9": "aw_e9"}
#: Names the live boxes serve under: a build never targets one without `--allow-remote`.
_SERVED_NAME = re.compile(r"^aw_(prod|[ve]\d+)$")
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
IDF_TABLE_FILENAME = "bm25_idf.json"    # the qstop table measured on THIS build (RUN_DIR)
SEALED_DIM = 1024                    # the width `cyt embed shards` writes; any serving width is a truncation


class BuildRefusal(RuntimeError):
    """An input this verb refuses before it writes anything to the collection (exit 3)."""


@dataclass(frozen=True)
class StagePack:
    pack_id: str
    family: str
    chunks_dir: str
    vectors_dir: str
    source: str


# ----------------------------------------------------------------------------- input checks
def load_stage_pack(spec: str) -> StagePack:
    d = pathlib.Path(spec)
    meta_path = d / "PACK.json"
    if not meta_path.is_file():
        raise BuildRefusal(f"{d}: no PACK.json -- a --pack is a stage dir written by "
                           "`cytir.sources.family_stage` (or one that declares its family_id and pack_id)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    family, pack_id = meta.get("family_id"), meta.get("pack_id")
    if not family or not pack_id:
        raise BuildRefusal(f"{meta_path}: needs family_id and pack_id")
    chunks, vectors = d / "chunks", d / "vectors"
    if not list(chunks.glob("shard-*.texts.jsonl.gz")):
        raise BuildRefusal(f"{chunks}: no shard-*.texts.jsonl.gz")
    if not list(vectors.glob("*.vectors.manifest.json")):
        raise BuildRefusal(f"{vectors}: no vectors -- run `cyt embed shards --in {chunks} --out "
                           f"{vectors}` first; the build never embeds")
    return StagePack(pack_id=pack_id, family=family, chunks_dir=str(chunks),
                     vectors_dir=str(vectors), source=str(d))


def check_vectors(pack: StagePack, declared_max_seq: int | None = None) -> list[int]:
    """Width 1024 (the sealed artefact) and the pinned embedder, on every vector shard.

    The token cap (`embedder.max_seq_length`) is part of the declared embedding surface
    (task 2538). It is RECORDED always -- the sorted distinct values the pack's shards
    declare are returned for the receipt -- and REFUSED only when the collection's profile
    declares a cap (`declared_max_seq`) AND a shard declares a different one. A shard that
    declares none, or a profile that declares none, is recorded and passes."""
    try:
        dim = COMPOSE.read_vector_native_dim(pack.vectors_dir)
    except COMPOSE.ComposeError as exc:
        raise BuildRefusal(str(exc)) from exc
    if dim != SEALED_DIM:
        raise BuildRefusal(f"{pack.vectors_dir}: vectors are {dim}-dim, not the sealed {SEALED_DIM}")
    seen: set[int] = set()
    for man_path in sorted(pathlib.Path(pack.vectors_dir).glob("*.vectors.manifest.json")):
        embedder = (json.loads(man_path.read_text(encoding="utf-8")).get("embedder") or {})
        if not embedder:
            raise BuildRefusal(f"{man_path}: no `embedder` record -- cannot prove the pinned model")
        try:
            assert_pin(embedder.get("model"), embedder.get("revision_pinned"),
                       where=f"cyt index build: {man_path.name}")
        except ValueError as exc:
            raise BuildRefusal(str(exc)) from exc
        cap = embedder.get("max_seq_length")
        if cap is not None and (not isinstance(cap, int) or isinstance(cap, bool)):
            raise BuildRefusal(f"{man_path}: embedder max_seq_length {cap!r} is not an integer")
        if cap is not None:
            seen.add(cap)
            if declared_max_seq is not None and cap != declared_max_seq:
                raise BuildRefusal(
                    f"{man_path}: embedder max_seq_length {cap} differs from the collection's "
                    f"declared embedder_max_seq {declared_max_seq} (two declarations of the "
                    "embedding surface disagree; re-embed the pack or change the declaration "
                    "by ruling)")
    return sorted(seen)


def check_target(url: str, collection: str, *, allow_remote: bool) -> None:
    host = urllib.parse.urlparse(url).hostname
    if (host not in _LOCAL_HOSTS or _SERVED_NAME.match(collection)) and not allow_remote:
        raise BuildRefusal(
            f"{collection!r} at {url}: a build never targets a non-local Qdrant or a name the live "
            "boxes serve (aw_prod, aw_vN, aw_eN) without --allow-remote")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _shard_digests(directory: str, pattern: str) -> dict[str, str]:
    return {p.name: _sha256_file(str(p)) for p in sorted(pathlib.Path(directory).glob(pattern))}


# ------------------------------------------------------------------------------ RUN_DIR
#: Written into RUN_DIR when a build first uses it: the proof that the `queue/` and `state/` trees
#: in it are this verb's own (a resume, or an `apply` on the same dir, must be allowed to reuse them).
RUN_MARKER = ".aw-index-run.json"
RUN_MARKER_SCHEMA = "aw.index-run.v2"
_OWN_TREES = ("queue", "state", "artifacts")
_SCAN_DEPTH, _SCAN_BUDGET = 3, 50_000


def _is_queue_shaped(d: pathlib.Path) -> bool:
    """A directory with a pack dir in it: a child (not `_`-prefixed) that carries a READY or DONE
    marker. This is what the twin-first drain lists, and what a stray file would hide a pack in."""
    try:
        with os.scandir(d) as it:
            for e in it:
                if (not e.name.startswith("_") and e.is_dir(follow_symlinks=False)
                        and ((pathlib.Path(e.path) / "READY").is_file()
                             or (pathlib.Path(e.path) / "DONE").is_file())):
                    return True
    except OSError:
        return False
    return False


def check_run_dir(out_dir: pathlib.Path) -> None:
    """Refuse a RUN_DIR that overlaps ANY queue-shaped directory the build did not make, whatever
    the settings say about where the live queue is (round 4, B2 of #559: a box without the overlay,
    or with `AW_WORK_ROOT` exported, wrote `queue/`, `artifacts/` and `build-manifest.json` into
    the live queue, and the hourly drain then listed them as pack dirs).

    Refused: RUN_DIR itself, any ancestor of it, or any directory within a few levels below it that
    holds pack dirs with READY / DONE markers. The one exemption is the trees the build itself
    created: `RUN_MARKER` records their ABSOLUTE paths and a creation time, and only a path in that
    record is exempt (round 5, B2 residual: exempting `queue/`, `state/`, `artifacts/` by NAME let a
    marker in the parent of a live queue called `queue` through). The scan is bounded:
    a RUN_DIR with more than a few tens of thousands of entries cannot be proven free of a queue
    and is refused too (pick an empty one). Read-only."""
    here = pathlib.Path(out_dir).resolve()
    own = _own_trees(here)
    for d in (here, *here.parents):
        if _is_queue_shaped(d):
            raise BuildRefusal(
                f"--out {out_dir}: {d} is a queue (it holds pack dirs with READY/DONE markers "
                "this build did not create). A build keeps its queue, state and artifacts in a "
                "RUN_DIR of its own, never in or beside a promotion queue")
    seen = 0
    todo = [(here, 0)]
    while todo:
        d, depth = todo.pop()
        if depth >= _SCAN_DEPTH or not d.is_dir():
            continue
        try:
            children = sorted((e for e in os.scandir(d) if e.is_dir(follow_symlinks=False)),
                              key=lambda e: e.name)
        except OSError:
            continue
        for e in children:
            seen += 1
            if seen > _SCAN_BUDGET:
                raise BuildRefusal(f"--out {out_dir} holds too many directories to prove it holds "
                                   "no promotion queue: pick an empty RUN_DIR")
            c = pathlib.Path(e.path)
            if str(c) in own:
                continue
            if _is_queue_shaped(c):
                raise BuildRefusal(
                    f"--out {out_dir} contains {c}, a queue (pack dirs with READY/DONE markers "
                    "this build did not create): a RUN_DIR must not hold or sit above a queue")
            todo.append((c, depth + 1))


def _own_trees(here: pathlib.Path) -> frozenset[str]:
    """The absolute paths RUN_DIR's marker says THIS build created (empty without a readable marker
    of this schema for exactly this directory)."""
    try:
        rec = json.loads((here / RUN_MARKER).read_text(encoding="utf-8"))
        if rec.get("schema") != RUN_MARKER_SCHEMA or rec.get("run_dir") != str(here):
            return frozenset()
        return frozenset(str(x) for x in rec.get("created_dirs", ()) if isinstance(x, str))
    except (OSError, ValueError, AttributeError):
        return frozenset()


def _mark_run_dir(out_dir: pathlib.Path) -> None:
    """First use of RUN_DIR: record the absolute paths of the trees this build is about to create.
    A tree that already exists and holds anything is not the build's, so it is not recorded (and
    stays subject to the queue scan)."""
    here = pathlib.Path(out_dir).resolve()
    here.mkdir(parents=True, exist_ok=True)
    marker = here / RUN_MARKER
    if marker.exists():
        return
    created = [str(here / n) for n in _OWN_TREES
               if not (here / n).exists() or ((here / n).is_dir() and not any((here / n).iterdir()))]
    marker.write_text(json.dumps({
        "schema": RUN_MARKER_SCHEMA, "run_dir": str(here), "created_dirs": created,
        "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, indent=1) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------ the run
def check_run_and_target(url: str, collection: str, out_dir: pathlib.Path, *,
                         allow_remote: bool) -> tuple[str, "PQ.Single"]:
    """Every refusal a build makes before it writes, for BOTH inputs (`--pack`, `--from-collection`):
    the endpoint string, RUN_DIR (never in, above or around a queue), the target name (never prod,
    never a live name) and the endpoint's own holdings. Returns (normalized url, the Single target)."""
    try:    # ONE endpoint string, with the port the client uses, for the check AND the writes (B3)
        url = LT.normalize_endpoint(url)
    except LT.LiveTargetRefused as exc:
        raise BuildRefusal(str(exc)) from exc
    live_queue = PQ.live_queue_dir()          # None on a box that has no promotion queue (a stranger's)
    here = pathlib.Path(out_dir).resolve()
    if live_queue is not None and (here == live_queue or live_queue in here.parents
                                   or here in live_queue.parents):
        # compose writes the build's manifests under RUN_DIR before the drain reads them: RUN_DIR
        # must not be the live queue, inside it, or above it (the drain's own refusal comes later)
        raise BuildRefusal(f"--out {out_dir} is the live promotion queue ({live_queue}), inside it "
                           "or above it: a build keeps its queue, state and artifacts in a RUN_DIR "
                           "of its own")
    check_run_dir(here)             # whatever the settings say: no queue-shaped dir in, above or around it
    check_target(url, collection, allow_remote=allow_remote)
    target = PQ.Single(url=url, collection=collection)
    try:        # --allow-remote lifts the name check above, never this one: BEFORE --recreate deletes
        PQ.assert_single_target_not_live(target)
    except PQ.QueueError as exc:
        raise BuildRefusal(str(exc)) from exc
    return url, target


def run(*, mode: str, packs: list[StagePack], url: str, collection: str, profile_name: str,
        out_dir: pathlib.Path, resume: bool = False, recreate: bool = False,
        allow_remote: bool = False, batch: int = 256, as_of: str | None = None,
        consent_cache_dir: str | None = None, settle_timeout_s: float = 0.0,
        log=print) -> tuple[int, dict]:
    """`mode` is "build" (create the collection, apply, finalize) or "apply" (a delta onto an
    existing collection, after the shape check). Returns (exit code, the receipt)."""
    if profile_name not in PUBLIC_PROFILES:
        raise BuildRefusal(f"unknown profile {profile_name!r} (public: {', '.join(PUBLIC_PROFILES)})")
    profile = SP.get(profile_name)
    url, target = check_run_and_target(url, collection, out_dir, allow_remote=allow_remote)
    if not packs:
        raise BuildRefusal("no --pack given")
    seen: set[str] = set()
    recorded_caps: dict[str, list[int]] = {}
    for pack in packs:
        if pack.pack_id in seen:
            raise BuildRefusal(f"pack id {pack.pack_id!r} given twice")
        seen.add(pack.pack_id)
        recorded_caps[pack.pack_id] = check_vectors(pack, profile.embedder_max_seq)

    client = qdrant_client(url)
    exists = client.collection_exists(collection)
    if mode == "build" and exists and not (resume or recreate):
        raise BuildRefusal(f"{collection} already exists: --resume continues a build, --recreate "
                           "deletes it first")
    if mode == "apply":
        if not exists:
            raise BuildRefusal(f"{collection} does not exist: `cyt index build` creates it")
        try:
            SP.verify_created(client, collection, profile, log=log)
        except SP.ShapeMismatch as exc:
            raise BuildRefusal(f"the collection is not the {profile_name!r} shape: {exc}") from exc

    # compose every pack first: an id that does not reproduce refuses the run before any write
    from cytir.consent import stamp as CONSENT_STAMP
    _mark_run_dir(out_dir)
    queue = out_dir / "queue"
    state = out_dir / "state"       # the Single transaction's own receipts and DONE markers
    try:
        consent_ctx = CONSENT_STAMP.load_context(as_of=as_of, cache_dir=consent_cache_dir)
    except CONSENT_STAMP.ConsentStampRefused as exc:
        raise BuildRefusal(f"{exc} (pass --consent-cache-dir DIR: an existing directory, empty is fine "
                           "for a dump-route family)") from exc
    if mode == "build" and (recreate or not exists):
        # The collection is (re)created EMPTY: a DONE marker this run dir kept from an earlier build
        # of the same pack describes rows that are gone, and the drain would pass the pack over with
        # no result (stranger QUICKSTART F6: an empty collection, rc 0). The markers are this
        # verb's own (check_run_dir proved the trees); clear them for the packs of THIS run.
        # Nothing is deleted before the drain's own refusals have run, and only inside --out.
        root = out_dir.resolve()
        try:
            PQ.refuse_live_single(PQ.Single(url=url, collection=collection, state_dir=state),
                                  queue, out_dir / "artifacts")
            PQ.refuse_state_entries(state, [p.pack_id for p in packs], queue)
        except PQ.QueueError as exc:
            raise BuildRefusal(str(exc)) from exc
        markers = [m for pack in packs
                   for m in (state / pack.pack_id / "DONE", queue / pack.pack_id / "DONE",
                             queue / pack.pack_id / "SUPERSEDED")]
        for d in (queue, state, *(m.parent for m in markers)):
            if d.is_symlink() or (d.exists() and not d.resolve().is_relative_to(root)):
                raise BuildRefusal(f"--out {out_dir}: {d} is a symlink or lies outside --out; "
                                   "the build will not clear markers there (use a fresh --out)")
        for marker in markers:
            marker.unlink(missing_ok=True)
    composed: dict[str, dict] = {}
    for pack in packs:
        if (state / pack.pack_id / "DONE").exists() and resume:
            log(f"[{pack.pack_id}] already landed (DONE): skipped")
            continue
        try:
            res = COMPOSE.compose_pack(
                pack_id=pack.pack_id, family=pack.family, chunks_dir=pack.chunks_dir,
                vectors_dir=pack.vectors_dir, collection=collection, mrl=profile.dense_dim,
                qdrant_url=url if exists else None, queue_dir=str(queue), consent_ctx=consent_ctx)
        except (COMPOSE.ComposeError, CONSENT_STAMP.ConsentStampRefused) as exc:
            raise BuildRefusal(f"{pack.pack_id}: {type(exc).__name__}: {exc}") from exc
        composed[pack.pack_id] = res

    created = False
    if mode == "build":
        # The plan-time check above is minutes old (compose ran since): the alias can have moved
        # onto the target. Under the target's lock, right before the delete/create, ask the endpoint
        # again (round 4, B1: the reviewer's probe deleted a collection `aw_prod` had been pointed at).
        try:
            with PQ.single_target_lock(target, "index-build", f"{collection}-create"):
                PQ.assert_single_target_not_live(target)
                created = SP.create_collection(
                    client, collection, profile, on_exists="recreate" if recreate else "keep",
                    bulk=True, verify=True, log=log)
        except PQ.QueueError as exc:
            raise BuildRefusal(str(exc)) from exc
        except RuntimeError as exc:
            from cytir.ops import write_lock          # the lock registry cannot be read, or it is held
            if isinstance(exc, write_lock.BatchError):
                raise BuildRefusal(f"{collection}: the target lock could not be taken, nothing was "
                                   f"created or deleted: {exc}") from exc
            raise
        if profile.dense_dim != PQ.PSP.target_width(url, collection):
            raise BuildRefusal(f"{collection}: dense width is not the profile's {profile.dense_dim}")

    results = PQ.drain(
        queue, task="index-" + mode, apply=True, batch=batch, pause_s=0,
        artifact_root=out_dir / "artifacts", refresh_index=False,
        only=[p.pack_id for p in packs if p.pack_id in composed],
        transaction=PQ.Single(url=url, collection=collection,
                              payload_profile_of=PAYLOAD_PROFILE_OF[profile_name],
                              state_dir=state))
    by_id = {r["pack_id"]: r for r in results}
    refused = {pid: r for pid, r in by_id.items() if r.get("status") != "LANDED"}

    receipt: dict = {
        "schema": SCHEMA, "mode": mode, "collection": collection, "qdrant_url": url,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "as_of": as_of,
        "profile": profile.block(), "payload_profile_of": PAYLOAD_PROFILE_OF[profile_name],
        "collection_created": created,
        "declared_embedder_max_seq": profile.embedder_max_seq,
        "packs": [_pack_receipt(p, composed.get(p.pack_id), by_id.get(p.pack_id), state,
                                recorded_caps.get(p.pack_id, []))
                  for p in packs]}
    if refused:
        receipt["refused"] = {pid: {"status": r.get("status"), "stage": r.get("stage"),
                                    "reason": r.get("reason")} for pid, r in refused.items()}
        _write_receipt(out_dir, mode, receipt)
        for pid, r in refused.items():
            log(f"[{pid}] {r.get('status')} {r.get('stage') or ''}: {r.get('reason') or r.get('blocked_on')}")
        return EXIT_GATE, receipt

    if mode == "build":
        n_points = client.count(collection, exact=True).count
        if n_points == 0:
            # never a silent empty success: nothing landed, or every pack was passed over
            raise BuildRefusal(
                f"{collection} holds 0 points after the build ({sum(1 for r in results if r.get('status') == 'LANDED')} "
                f"of {len(packs)} pack(s) landed). Use a fresh --out, or remove the old one; "
                "a build never reports success on an empty collection")
        SP.finalize_collection(client, collection, profile, log=log)
    fp = _settle(url, collection, settle_timeout_s)
    receipt["layout"] = {k: fp[k] for k in ("points_count", "n_per_vector", "segment_sizes",
                                            "segments_count", "stale_postings", "stale_ratio")
                         if k in fp}
    receipt["layout_settled"] = LAYOUT.settled(fp)           # [] == settled; reported, never hidden
    receipt["fingerprints"], receipt["idf_snapshot"], receipt["idf_table"] = _write_row_sidecars(
        client, url, collection, out_dir, as_of=as_of, log=log)
    _write_receipt(out_dir, mode, receipt)
    return EXIT_OK, receipt


def _pack_receipt(pack: StagePack, composed: dict | None, landed: dict | None,
                  state: pathlib.Path, embedder_max_seq: list[int] | None = None) -> dict:
    out = {"pack_id": pack.pack_id, "family": pack.family, "source": pack.source,
           "embedder_max_seq": embedder_max_seq or [],
           "id_scheme": COMPOSE.id_scheme_for_family(pack.family),
           "chunks_sha256": _shard_digests(pack.chunks_dir, "shard-*.texts.jsonl.gz"),
           "vectors_sha256": _shard_digests(pack.vectors_dir, "*.vectors.f16")}
    if composed is not None:
        out["manifest_sha256"] = composed["ready_sha"]
        out["records"] = composed["identity"]["checked"]
    if landed is not None:
        rows = (landed.get("target") or {}).get("rows") or {}
        out["status"] = landed.get("status")
        out["admitted"] = rows.get("projected")
        out["written"] = (landed.get("target") or {}).get("written")
    elif (state / pack.pack_id / "DONE").exists():
        out["status"] = "DONE-EARLIER"
    return out


def _settle(url: str, collection: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    fp = LAYOUT.fingerprint(url, collection)
    while LAYOUT.settled(fp) and time.monotonic() < deadline:
        time.sleep(min(5.0, max(0.5, timeout_s / 20)))
        fp = LAYOUT.fingerprint(url, collection)
    return fp


def _write_row_sidecars(client, url: str, collection: str, out_dir: pathlib.Path, *,
                        as_of: str | None, log=print) -> tuple[dict, dict | None, dict | None]:
    """ONE pass over the built rows writes the three sidecars a build keeps beside its receipt.

    1. The row-fingerprint sidecar (`cytir.distribute.pack.row_fingerprint`, the dump's own key): one
       sorted `id<TAB>fingerprint` line per row. Two builds of the same packs have the same
       sidecar, and a test proves it.
    2. The BM25 IDF snapshot (`cytir.distribute.idf`, the dump's own `bm25-idf.tsv.gz`): `n` and the
       `df` of every term hash, counted over exactly the rows the build wrote.
    3. The qstop IDF TABLE (`cytir.index.idf_table.measure_built`, `bm25_idf.json`): the per-term IDF
       MEASURED on this collection, for `cyt serve --idf-table` (design question Q4). The shipped
       `cytir/serve/bm25_idf.json` is prod's table, measured on prod's collection; a build
       never reads or writes it."""
    from cytir.distribute import dump as DUMP
    from cytir.distribute import idf as SNAP
    from cytir.distribute import pack as PACK
    from cytir.index import idf_table as IDF_TABLE
    path = out_dir / PACK.FINGERPRINTS_FILENAME
    writer = PACK.FingerprintWriter(str(path))
    acc = SNAP.IdfAccumulator()
    for row in DUMP.scroll_rows(client, collection):
        writer.add(row["id"], PACK.row_fingerprint(row))
        acc.add(row)
    block = writer.close()
    # `gzip.open` stamps the wall-clock second into the header, so the same rows would hash
    # differently a second apart: re-encode with mtime 0, so the file's own sha256 is a function of
    # the rows only (the dump's sidecar has the same latent property; not changed here).
    import gzip

    from cytir.acquire.gzip_pinned import deterministic_gzip_open
    raw = gzip.open(path, "rb").read()
    with open(path, "wb") as out, deterministic_gzip_open(out, compresslevel=9) as gz:
        gz.write(raw)
    block["sha256"] = _sha256_file(str(path))
    snapshot = acc.write(out_dir)
    if snapshot is None:
        return block, None, None
    table = IDF_TABLE.measure_built(url, collection, acc.df, acc.n, as_of=as_of, batch=64, log=log)
    table_path = out_dir / IDF_TABLE_FILENAME
    table_path.write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return block, snapshot, {
        "path": IDF_TABLE_FILENAME, "sha256": _sha256_file(str(table_path)),
        "collection": table["collection"], "points_count": table["points_count"],
        "n_measured": table["n_measured"], "terms_in_table": len(table["idf"]),
        "pass_it_to": f"aw serve --collection {collection} --idf-table {table_path}"}  # rename-codemod: keep (persisted: summary field written into the build receipt)


def _write_receipt(out_dir: pathlib.Path, mode: str, receipt: dict) -> None:
    name = "build-manifest.json" if mode == "build" else "apply-manifest.json"
    (out_dir / name).write_text(json.dumps(receipt, indent=1, sort_keys=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------- the CLI
def _parser(mode: str, prog: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=prog, description=(
            "create a collection from a serving profile (default prod-e9) and apply sealed packs to it" if mode == "build"
            else "apply further sealed packs to an existing collection (after the shape check)"))
    ap.add_argument("--pack", action="append", default=[], metavar="STAGE_DIR",
                    help="a stage dir (PACK.json + chunks/ + vectors/); repeat for more packs")
    ap.add_argument("--qdrant", required=True, metavar="URL")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--profile", default=None,
                    help=f"the collection shape, one of {', '.join(PUBLIC_PROFILES)} (default {DEFAULT_PACK_PROFILE}: what prod runs); "
                         "`served` is the pre-e9 shape (hnsw m=16, float16 sparse index)")
    ap.add_argument("--out", required=True, metavar="RUN_DIR")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--as-of", default=None, metavar="ISO",
                    help="the UTC stamp of the consent derivation (default: now). Pin it for a "
                         "reproducible rebuild: it lands in every payload's consent_checked_at")
    ap.add_argument("--consent-cache-dir", default=None)
    ap.add_argument("--allow-remote", action="store_true")
    ap.add_argument("--settle-timeout", type=float, default=0.0, metavar="SECONDS",
                    help="wait this long for the collection's layout to settle before the receipt "
                         "(a collection under ~10,000 points per segment never settles)")
    if mode == "build":
        ap.add_argument("--resume", action="store_true",
                        help="continue a build: a pack already DONE in RUN_DIR/state is skipped")
        ap.add_argument("--recreate", action="store_true", help="delete the collection first")
        from cytir.index import from_collection as FC     # a second INPUT to the one build door
        FC.add_arguments(ap)
    return ap


def main(argv: list[str] | None = None, *, mode: str = "build", prog: str | None = None) -> int:
    ap = _parser(mode, prog or f"cyt index {mode}")
    args = ap.parse_args(argv)
    TWH.refuse_cli_write(args, tool=f"cyt index {mode}")   # while an aw_e10 build holds the twin, only the drain writes it
    if getattr(args, "from_collection", None):
        if args.pack:
            ap.error("--from-collection and --pack are two inputs to one build: give one")
        from cytir.index import from_collection as FC
        try:
            code, _ = FC.run(args)
        except BuildRefusal as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return EXIT_REFUSED
        return code
    if args.profile is not None and args.profile not in PUBLIC_PROFILES:
        # not argparse's `choices=` (rc 2): a private name is a refusal, rc 3, as `run()` gives it
        print(f"REFUSED: unknown profile {args.profile!r} (public: {', '.join(PUBLIC_PROFILES)})",
              file=sys.stderr)
        return EXIT_REFUSED
    try:
        packs = [load_stage_pack(s) for s in args.pack]
        code, receipt = run(
            mode=mode, packs=packs, url=args.qdrant, collection=args.collection,
            profile_name=args.profile or DEFAULT_PACK_PROFILE, out_dir=pathlib.Path(args.out),
            resume=getattr(args, "resume", False), recreate=getattr(args, "recreate", False),
            allow_remote=args.allow_remote, batch=args.batch, as_of=args.as_of,
            consent_cache_dir=args.consent_cache_dir, settle_timeout_s=args.settle_timeout)
    except BuildRefusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if code == EXIT_OK:
        print(f"[{mode}] done: {len(packs)} pack(s) -> {args.collection}; receipt "
              f"{pathlib.Path(args.out) / ('build-manifest.json' if mode == 'build' else 'apply-manifest.json')}")
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], mode="build", prog="python -m cytir.index.build"))
