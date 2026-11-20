"""cytir.index.from_collection -- `cyt index build --from-collection URL:COLLECTION` (task 1256).

The second INPUT to the one build door (`cytir.index.build`): instead of sealed stage packs, the rows
of an existing collection, STREAMED source -> levers -> target and never written to disk (at 33.7M
rows a row file would be ~280 GB of JSONL read once). This is how aw_e9 was built (task 637 S6,
by the streaming builder this module replaces, task 1256); the collection is created from `PROD_E9`, the
profile prod serves, through the one creator (`serving_profile.create_collection`). Every lever is
the package's own: the payload projection (`payload_profile.project_payload`), the prune cut
(`sparse_rule.build_prune_set` / `prune_postings`).

    cyt index build --from-collection TWIN_URL:TWIN_COLLECTION \\
        --qdrant BUILD_URL --collection aw_rebuilt --out RUN_DIR \\
        --prune-df-gt 0.02 --prune-df-lt 2 --guard-path /data:120

TWO PASSES, AND WHY THE FIRST ONE CANNOT BE SKIPPED. Pass 1 counts document frequency over the
SOURCE's own sparse vectors (`with_vector=["bm25"]`, no payload). It cannot be replaced by a
sample: a df > 2% cut could be estimated from one, but a hapax floor cannot -- a term of true df 1
and one of true df 1,000 both appear ~once in a 3.6% sample, so a sampled hapax set would delete
high-IDF content terms (task 637 sec 3.2 refused that claim for exactly this reason).

THE SHAPE IS PROD_E9. `--hnsw-m`, `--sparse-datatype`, `--dense-datatype` override one field each
(`ServingProfile.with_`); none defaults to the pre-e9 arm shape (m=16, float16 sparse, 4 segments,
10 in-RAM payload indexes, task 1253 sec 3). The profile is created at m=0 and PATCHed to its m
after the load (`finalize_collection`), the way every package build loads.

GUARDS. The source is read through `prod_guard` (a scroll of prod is refused, no override here),
and the source is only ever read. The target passes `build.check_run_and_target` (never prod, never
a live name, the endpoint's holdings) and the guard again before every write. A disk floor aborts
the run (`--guard-path PATH:GB`): a Qdrant that fills its disk mid-build leaves a collection nobody
can settle. Pass 2 is resumable by scroll offset (`--resume`); resume is by point id, so a re-run is
idempotent. This mode writes no row-fingerprint sidecar and no `bm25_idf.json` for
`cyt serve --idf-table` (each is a full scroll of the target). `--only-ids FILE` catches an EXISTING target up by id through the SAME transform (a
copier that "just copies the missing points" would write f32 sparse vectors and un-normalised
payloads into a uint8, enum-encoded collection, and nothing would report it).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import time
import uuid

from cytir.core import serving_profile as SP
from cytir.core.qdrant import qdrant_client
from cytir.index import build as B
from cytir.index import layout as LAYOUT
from cytir.index import payload_profile as PP
from cytir.index import prod_guard as PG
from cytir.index import promote_queue as PQ
from cytir.index import promote_sealed_pack as PSP
from cytir.index import sparse_rule as SR

BASE_PROFILE = SP.PROD_E9
STATE_FILE = "from-collection.state.json"
DF_CACHE_FILE = "df.json"
#: profile field <- CLI flag; a flag left unset keeps PROD_E9's value
OVERRIDES = {"hnsw_m": "hnsw_m", "sparse_index_datatype": "sparse_datatype",
             "dense_datatype": "dense_datatype"}
_SOURCE_READS = ("/points/scroll", "/points", "/points/count")


def add_arguments(ap) -> None:
    g = ap.add_argument_group("--from-collection (stream an existing collection; profile PROD_E9)")
    g.add_argument("--from-collection", metavar="URL:COLLECTION", default=None,
                   help="the source collection, read only (the operator's twin: TWIN_URL:TWIN_COLLECTION)")
    g.add_argument("--scroll", type=int, default=2000, help="source page size")
    g.add_argument("--prune-df-gt", type=float, default=None, help="drop terms in more than this share of rows")
    g.add_argument("--prune-df-lt", type=int, default=None, help="drop terms in fewer than this many rows")
    g.add_argument("--prune-idf-lt", type=float, default=None)
    g.add_argument("--payload-profile", default="prod", choices=sorted(PP.PROFILE_RULES),
                   help="default prod: every payload written as read (identity). aw_e9 was built "
                        "with normalized")
    g.add_argument("--hnsw-m", type=int, default=None, help=f"default {BASE_PROFILE.hnsw_m}")
    g.add_argument("--sparse-datatype", default=None, choices=("float32", "float16", "uint8"),
                   help=f"the sparse INDEX's weight encoding; default {BASE_PROFILE.sparse_index_datatype}")
    g.add_argument("--dense-datatype", default=None, choices=("float32", "float16", "uint8"),
                   help=f"default {BASE_PROFILE.dense_datatype}")
    g.add_argument("--df-cache", default=None, help=f"default RUN_DIR/{DF_CACHE_FILE}")
    g.add_argument("--df-only", action="store_true", help="run pass 1 and stop")
    g.add_argument("--only-ids", default=None, metavar="FILE",
                   help="DELTA CATCH-UP into an existing target built in the same RUN_DIR: one point "
                        "id per line (`python3 -m cytir.index.id_digest --diff` writes it)")
    g.add_argument("--guard-path", action="append", default=[], metavar="PATH:GB",
                   help="repeatable; the build aborts when PATH has less than GB free")


def parse_source(spec: str) -> tuple[str, str]:
    url, sep, coll = spec.rpartition(":")
    if not sep or not coll or "/" in coll or "//" not in url:
        raise B.BuildRefusal(f"--from-collection {spec!r}: expected URL:COLLECTION")
    return url.rstrip("/"), coll


def profile_for(args) -> SP.ServingProfile:
    """PROD_E9, with exactly the fields whose flag was given changed (and named in the profile)."""
    kw = {f: getattr(args, a) for f, a in OVERRIDES.items() if getattr(args, a, None) is not None}
    if not kw:
        return BASE_PROFILE
    name = BASE_PROFILE.name + "".join(f"+{k}={v}" for k, v in sorted(kw.items()))
    return BASE_PROFILE.with_(name=name, **kw)


# ------------------------------------------------------------------------ the guarded transport
def _source_api(url: str, path: str, body: dict | None = None):
    """The ONLY call against the source: a read, through the prod guard (a prod scroll is refused)."""
    bare = path.split("?", 1)[0]
    if body is not None and not bare.endswith(_SOURCE_READS):
        raise B.BuildRefusal(f"{path}: the source is read only")
    try:
        PG.check(url, "POST" if body is not None else "GET", path)
    except PG.ProdReadRefused as exc:
        raise B.BuildRefusal(str(exc)) from exc
    return PSP._api(url, path, body)


def _target_api(url: str, path: str, body: dict | None = None, method: str | None = None):
    """Every HTTP write to the target: refused when the guard says the target is prod."""
    if PG.check(url, method or ("POST" if body is not None else "GET"), path):
        raise B.BuildRefusal(f"{url}{path}: the target is prod; a build never writes it")
    return PSP._api(url, path, body, method)


def _refuse_prod_source(src_url: str, src: str) -> None:
    """Argument validation: a prod SOURCE is refused before any target write (task 1289). `_source_api`
    refuses at the first scroll, which came after the create and left an empty collection on the target
    box. No override flag: a prod scroll is never allowed here."""
    try:
        PG.check(src_url, "POST", f"/collections/{src}/points/scroll")
    except PG.ProdReadRefused as exc:
        raise B.BuildRefusal(str(exc)) from exc


def _finalize_target(client, url: str, target: str, profile, log=print) -> None:
    """`finalize_collection` PATCHes the target: a write, so the guard is asked first."""
    if PG.check(url, "PATCH", f"/collections/{target}"):
        raise B.BuildRefusal(f"{target} at {url} is prod; a build never patches it")
    SP.finalize_collection(client, target, profile, log=log)


def _create_target(client, single, target: str, profile, *, on_exists: str, log=print) -> bool:
    """The create (and, with `on_exists="recreate"`, the delete of an existing target) under the lock;
    the guard is asked again right before the first write."""
    with _create_lock(single):
        if PG.check(single.url, "DELETE" if on_exists == "recreate" else "PUT", f"/collections/{target}"):
            raise B.BuildRefusal(f"{target} at {single.url} is prod; a build never creates or deletes it")
        return SP.create_collection(client, target, profile, bulk=True, verify=False, log=log,
                                    on_exists=on_exists)


@contextlib.contextmanager
def _create_lock(single):
    """The Single transaction's target lock, with the target asked again under it right before the
    delete/create (the build door's round 4, B1: the alias can move onto the target meanwhile)."""
    if PG.check(single.url, "PUT", f"/collections/{single.collection}"):
        raise B.BuildRefusal(f"{single.collection} at {single.url} is prod; a build never creates it")
    try:
        with PQ.single_target_lock(single, "index-build", f"{single.collection}-create"):
            PQ.assert_single_target_not_live(single)
            yield
    except PQ.QueueError as exc:
        raise B.BuildRefusal(str(exc)) from exc


def scroll_pages(url, collection, *, limit, with_payload, with_vector, offset=None,
                 id_before=None, id_through=None):
    """Yield (points, next_offset) pages of a read-only scroll, in Qdrant's ascending id order.

    `id_before` / `id_through` (uuid ints, both optional): stop at the first id >= `id_before`
    (exclusive end of a uuid-space slice) or > `id_through` (a limited plan's last row). The page
    that meets the bound is cut there and yields next_offset None. Both None: the plain scroll."""
    while True:
        body = {"limit": limit, "with_payload": with_payload, "with_vector": with_vector}
        if offset is not None:
            body["offset"] = offset
        res = _source_api(url, f"/collections/{collection}/points/scroll", body)
        pts, offset = res.get("points") or [], res.get("next_page_offset")
        if id_before is not None or id_through is not None:
            keep = []
            for p in pts:
                i = uuid.UUID(str(p["id"])).int
                if (id_before is not None and i >= id_before) or (id_through is not None and i > id_through):
                    offset = None
                    break
                keep.append(p)
            pts = keep
        yield pts, offset
        if offset is None:
            return


def id_pages(url, collection, ids, *, limit, start=0):
    """Yield (points, next_index) for an EXPLICIT id list: a second INPUT to pass 2, not a second
    pass 2 -- its rows go through the identical transform. `next_index` is the resume position."""
    i = start
    while i < len(ids):
        chunk = ids[i:i + limit]
        res = _source_api(url, f"/collections/{collection}/points",
                          {"ids": chunk, "with_payload": True, "with_vector": True}) or []
        i += len(chunk)
        yield res, i


class DiskGuard:
    """Abort the build before a filesystem fills, not after."""

    def __init__(self, specs):
        self.floors = [(p, float(gb or 50)) for p, _, gb in (s.partition(":") for s in specs or [])]
        self.readings: dict[str, float] = {}

    def check(self, where: str) -> None:
        for path, floor in self.floors:
            free = shutil.disk_usage(path).free / 1e9
            self.readings[path] = min(self.readings.get(path, free), free)
            if free < floor:
                raise B.BuildRefusal(f"ABORT ({where}): {path} has {free:.1f} GB free, floor "
                                     f"{floor:.1f} GB; the build stops before the disk a settle needs fills")

    def report(self) -> dict:
        return {p: {"floor_gb": fl, "min_free_gb_observed": round(self.readings.get(p, 0.0), 1)}
                for p, fl in self.floors}


def pass1_df(url, collection, *, scroll, cache, guard, log):
    """Document frequency over the SOURCE's own sparse vectors (cached per source, complete only)."""
    if cache and os.path.exists(cache):
        blob = json.load(open(cache))
        if blob.get("source") == collection and blob.get("complete"):
            log(f"# df cache hit: n={blob['n']} postings={blob['postings']} terms={len(blob['df'])}")
            return {int(k): v for k, v in blob["df"].items()}, blob["n"], blob["postings"]
    df: dict[int, int] = {}
    n = postings = 0
    for pts, _ in scroll_pages(url, collection, limit=scroll, with_payload=False, with_vector=["bm25"]):
        for p in pts:
            n += 1
            for t in ((p.get("vector") or {}).get("bm25") or {}).get("indices", ()):
                df[t] = df.get(t, 0) + 1
                postings += 1
        if n % (scroll * 50) == 0:
            guard.check("pass1")
            log(f"  pass1 {n} points, {postings} postings, {len(df)} terms")
    if cache:
        json.dump({"source": collection, "complete": True, "n": n, "postings": postings,
                   "df": {str(k): v for k, v in df.items()}}, open(cache, "w"))
    return df, n, postings


def _seed_tables(profile: str) -> tuple[dict, dict | None]:
    """A fresh build starts from the declared enum tables of aw_e9 (`build.PAYLOAD_PROFILE_OF`, the
    declared collection a profile build stores the payload form of), so a value gets the integer
    aw_e9 gives it; a value the table lacks is appended (grow)."""
    if not PP.PROFILE_RULES[profile]:
        return {}, None
    entry = PP.load_profiles()[B.PAYLOAD_PROFILE_OF["served"]]
    tables = PP.load_tables(expect_sha256=entry.get("tables_sha256"))
    return tables, {"from": B.PAYLOAD_PROFILE_OF["served"], "sha256": entry.get("tables_sha256")}


# --------------------------------------------------------------------------------------- the run
def run(args, log=print) -> tuple[int, dict]:
    from cytir.index import twin_write_hold as TWH
    TWH.refuse_cli_write(args, tool="cyt index build --from-collection")
    if args.profile not in (None, SP.PROD_E9.name):
        raise B.BuildRefusal("--from-collection builds PROD_E9: change a field with --hnsw-m / "
                             "--sparse-datatype / --dense-datatype, not --profile")
    src_url, src = parse_source(args.from_collection)
    _refuse_prod_source(src_url, src)
    target, out_dir = args.collection, pathlib.Path(args.out)
    if target == src:
        raise B.BuildRefusal("target and source are the same collection")
    url, single = B.check_run_and_target(args.qdrant, target, out_dir, allow_remote=args.allow_remote)
    profile = profile_for(args)
    pprofile = args.payload_profile
    prune_flags = (args.prune_df_gt, args.prune_df_lt, args.prune_idf_lt)
    delta = bool(args.only_ids)
    B._mark_run_dir(out_dir)
    state_path = out_dir / (STATE_FILE.replace(".json", ".delta.json") if delta else STATE_FILE)
    df_cache = args.df_cache or str(out_dir / DF_CACHE_FILE)
    guard = DiskGuard(args.guard_path)
    guard.check("startup")
    client = qdrant_client(url)
    exists = client.collection_exists(target)
    src_info = _source_api(src_url, f"/collections/{src}")
    log(f"# source {src}: {src_info.get('points_count')} points, status {src_info.get('status')}")

    state = json.load(open(out_dir / STATE_FILE)) if (out_dir / STATE_FILE).exists() else {}
    only_ids: list[str] = []
    if delta:   # each refusal is a way a delta row would silently differ from the rows it joins
        if args.recreate:
            raise B.BuildRefusal("--only-ids catches an EXISTING target up; --recreate would delete it")
        if not exists:
            raise B.BuildRefusal(f"--only-ids needs an existing target; {target} is absent")
        if PP.PROFILE_RULES[pprofile] and not state.get("enum_tables_forward"):
            raise B.BuildRefusal(f"--payload-profile {pprofile} encodes enums and {out_dir / STATE_FILE} "
                                 "carries no enum_tables_forward: a fresh table would give a value a "
                                 f"different integer than the rows already in {target}")
        if any(f is not None for f in prune_flags) and not os.path.exists(df_cache):
            raise B.BuildRefusal("--only-ids with a prune needs the build's df cache: a df pass over "
                                 "the delta alone prunes against a different vocabulary")
        only_ids = [ln.strip() for ln in open(args.only_ids) if ln.strip()]
        if not only_ids:
            raise B.BuildRefusal(f"{args.only_ids} holds no ids: an empty catch-up is an argument "
                                 "error, not a no-op -- re-run the digest --diff")
        if state_path.exists():
            state = json.load(open(state_path))

    df, df_n, df_postings, prune, rule = {}, None, None, set(), {}
    if any(f is not None for f in prune_flags):
        df, df_n, df_postings = pass1_df(src_url, src, scroll=args.scroll, cache=df_cache, guard=guard, log=log)
        prune, rule = SR.build_prune_set(df, df_n, df_gt=args.prune_df_gt, idf_lt=args.prune_idf_lt,
                                         df_lt=args.prune_df_lt)
        log(f"# prune set: {len(prune)} terms of {len(df)}; rule={rule}")
    receipt: dict = {
        "schema": B.SCHEMA, "mode": "from-collection-delta" if delta else "from-collection",
        "collection": target, "qdrant_url": url,
        "source": {"url": src_url, "collection": src, "points_count": src_info.get("points_count"),
                   "status": src_info.get("status")},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile": profile.block(), "payload_profile": pprofile,
        "prune": {"rule": rule, "terms": len(prune), "df_n": df_n, "df_postings": df_postings}}
    if args.df_only:
        (out_dir / "df-manifest.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
        return B.EXIT_OK, receipt
    if delta:   # a delta joins rows of THIS shape: a target of another shape is refused before a write
        try:
            receipt["verify_before"] = SP.verify_created(client, target, profile, log=log)
        except SP.ShapeMismatch as exc:
            raise B.BuildRefusal(f"--only-ids: {exc}") from exc

    # `phase: loading` is written right after the create and after every complete page; a state
    # of the old builder carries only an offset
    resuming = (state.get("phase") == "loading" or bool(state.get("offset"))) and exists and not delta
    if args.resume and not delta and not resuming:
        raise B.BuildRefusal(f"--resume needs {state_path} of an UNFINISHED build that created "
                             f"{target} (phase loading); without it the run would re-scroll from the start "
                             "with fresh tables into a collection it did not build")
    if exists and not (resuming or args.recreate or delta or args.resume):
        raise B.BuildRefusal(f"{target} already exists: --resume continues a build, --recreate deletes it")
    created = False
    if not exists or (args.recreate and not resuming):
        created = _create_target(client, single, target, profile, log=log,
                                 on_exists="recreate" if args.recreate else "refuse")
        state = {}
    tables, seed = ((state.get("enum_tables_forward"), state.get("enum_tables_seed")) if state
                    else _seed_tables(pprofile))
    tables = tables or {}
    c = {k: state.get(k, 0) for k in ("sent", "postings_kept", "postings_dropped", "rows_no_sparse")}
    keys_seen, keys_written = set(state.get("payload_keys_seen") or ()), set(state.get("payload_keys_written") or ())
    # a delta resumes its position only in the SAME id list; a new list starts at 0 (the tables and
    # counts of the earlier catch-up still carry over: its enum tables may have grown)
    ids_sha = hashlib.sha256("\n".join(only_ids).encode()).hexdigest() if delta else None
    offset = state.get("offset")
    done = state.get("delta_done", 0) if delta and state.get("ids_sha256") == ids_sha else 0
    batch: list[dict] = []
    seen_delta = 0

    def send():
        nonlocal batch
        if batch:
            _target_api(url, f"/collections/{target}/points?wait=true", {"points": batch}, "PUT")
            batch = []

    def checkpoint(next_pos, finished=False):
        """State is written ONLY once a whole page is in the target (review 1268 B1: a mid-page
        checkpoint of the NEXT page's offset lost the page's unsent rows on resume, silently). A
        crash mid-page leaves the previous page's state: the resume re-reads this page from its
        start, re-projects it from the persisted tables (same rows in the same order give the same
        codes, so the rows already upserted are rewritten identically -- this holds while the source
        does not change, the twin-write hold's job), re-upserts by id (idempotent) and counts each
        row once, because the counters on disk are the previous page's."""
        blob = dict(c, offset=offset if delta else next_pos, enum_tables_forward=tables,
                    enum_tables_seed=seed, phase="loaded" if finished else "loading",
                    payload_keys_seen=sorted(keys_seen), payload_keys_written=sorted(keys_written))
        if delta:
            blob.update(delta_done=next_pos, ids_sha256=ids_sha)
        state_path.write_text(json.dumps(blob), encoding="utf-8")

    if created:
        checkpoint(None)        # a crash before the first page ends is resumable from the start
    pages = (id_pages(src_url, src, only_ids, limit=args.scroll, start=done) if delta else
             scroll_pages(src_url, src, limit=args.scroll, with_payload=True, with_vector=True, offset=offset))
    for pts, next_pos in pages:
        seen_delta += len(pts)
        for p in pts:
            pl = p.get("payload") or {}
            keys_seen |= set(pl)
            newpl = PP.project_payload(pl, pprofile, tables, grow=True)
            keys_written |= set(newpl)
            vec = {"dense": (p.get("vector") or {}).get("dense")}
            if vec["dense"] is None:
                raise B.BuildRefusal(f"source point {p['id']} has no dense vector ({c['sent']} rows sent)")
            sp = (p.get("vector") or {}).get("bm25")
            if sp:
                idx, val = sp.get("indices", []), sp.get("values", [])
                if prune:
                    before = len(idx)
                    idx, val = SR.prune_postings(idx, val, prune)
                    c["postings_dropped"] += before - len(idx)
                c["postings_kept"] += len(idx)
                if idx:
                    vec["bm25"] = {"indices": idx, "values": val}
                else:
                    c["rows_no_sparse"] += 1
            batch.append({"id": p["id"], "vector": vec, "payload": newpl})
            c["sent"] += 1
            if len(batch) >= args.batch:
                send()
        send()
        checkpoint(next_pos, finished=next_pos is None or (delta and next_pos >= len(only_ids)))
        if c["sent"] % (args.scroll * 25) < args.scroll:
            guard.check("pass2")
            log(f"  pass2 {c['sent']}/{src_info.get('points_count')} points")
    if delta and seen_delta != len(only_ids) - done:
        raise B.BuildRefusal(f"{done + seen_delta} of the {len(only_ids)} requested ids came back from {src}: "
                             "the delta was computed against a state the source no longer holds. Re-run "
                             "`python3 -m cytir.index.id_digest --diff` (the upserts applied are idempotent)")

    if not delta:
        _finalize_target(client, url, target, profile, log=log)
    exact = _target_api(url, f"/collections/{target}/points/count", {"exact": True})["count"]
    fp = B._settle(url, target, args.settle_timeout)
    receipt.update(
        collection_created=created, enum_tables_seed=seed, points_sent=c["sent"], points_exact=exact,
        lever=dict(c, delta_ids_requested=len(only_ids) or None,
                   payload_keys_seen=sorted(keys_seen), payload_keys_written=sorted(keys_written),
                   payload_keys_dropped=sorted(keys_seen - keys_written),
                   payload_enum_tables={k: {v: kk for kk, v in t.items()} for k, t in tables.items()}),
        layout={k: fp[k] for k in ("points_count", "segments_count", "stale_ratio") if k in fp},
        layout_settled=LAYOUT.settled(fp), disk=guard.report(),
        idf_table=None, idf_table_note="this mode writes no bm25_idf.json: measure one for `aw serve "  # rename-codemod: keep (persisted: idf_table_note in the build receipt)
        "--idf-table` with cytir.index.idf_table on the built collection")
    code = B.EXIT_OK
    if not delta:
        try:
            receipt["verify"] = SP.verify_created(client, target, profile, log=log)
        except SP.ShapeMismatch as exc:     # written, but not the profile: said in the receipt and the rc
            receipt["verify"], code = {"verified": False, "mismatch": str(exc)}, B.EXIT_GATE
    B._write_receipt(out_dir, "apply" if delta else "build", receipt)
    if exact != c["sent"]:
        raise B.BuildRefusal(f"sent {c['sent']} rows, {target} holds {exact} (receipt written)")
    log(f"[from-collection] done: {c['sent']} rows {src} -> {target}; receipt in {out_dir}")
    return code, receipt

