#!/usr/bin/env python3
# Moved from the retired `bm25_saturate` module (task 677, task 643 S5); the old path is an alias shim.
"""BM25 tf-saturation transform (task #22, rebuild-bundle centerpiece).

The stored `bm25` sparse weights across aw_v3* are RAW term frequencies
(`aw_common.bm25_tokens` emits {term-hash: tf}; qdrant's `modifier=idf`
supplies idf query-side). Raw tf means a "Laura" list page with tf=36
out-scores every rare-token page — the task 083 entity-query failure.

This tool rewrites every point's bm25 vector to BM25-saturated weights:

    w' = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

with k1=1.2, b=0.75, dl = sum(tf over the doc's bm25 vector) and avgdl
computed over the source collection in one streaming pass. idf stays with
qdrant's modifier (df is preserved: w' > 0 iff tf > 0). Saturated weights
live in (0, k1+1) — max ~2.2; tf=1 at dl=avgdl maps to exactly 1.0.

No re-tokenization: the raw tf IS the stored weight. Dense vectors and
payloads are copied through unchanged. Per ADR-022 this is a config-change
class rebuild: write into a FRESH target collection (bulk hnsw m=0, then
`finalize` -> m=16 + payload indexes + optional alias swap done manually).

Subcommands
-----------
  stats     pass 1: scroll src (bm25 vectors only) -> avgdl + df stats json/npz
  build     pass 2: scroll src (dense+bm25+payload) -> transformed points
            into dst. --raw copies weights untouched (control collection).
            --max-points / --ids-file / --start-id / --end-id for slices
            and id-range parallelism (UUID ids scroll in ascending order).
  finalize  set hnsw m=16 on dst + replicate src payload indexes
  probe     bm25-leg-only query against a collection (debug/validation)

Source collections are NEVER written (aw_v3_b is the prod rollback).

Edge cases handled: point with no bm25 vector (copied dense-only, counted),
empty bm25 vector (kept empty), non-positive weights (dropped, counted).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# The saturation math is a write/query mirror contract used by file-to-file
# workers too.  Keep the Qdrant dependency lazy so importing the pure transform
# does not smuggle a database client into an offline image.
try:
    from qdrant_client import QdrantClient
    from qdrant_client import models as m
except ImportError:  # pragma: no cover - exercised by offline worker tests
    QdrantClient = None
    m = None

K1 = 1.2
B = 0.75

# ADR-028 base build avgdl (aw_v3_b -> aw_v4_sat, task 096/098 pass-1 stats,
# /media/disk2/aw/bm25sat/bm25_stats.json). EVERY live upsert into the shared
# aw_v4_sat/aw_prod collection MUST saturate against THIS constant so new points
# are consistent with the saturated base (task 211 root cause, task 212 fix).
BASE_AVGDL = 199.71502927412075

# Saturated BM25 weights live in (0, k1+1) = (0, 2.2). A stored weight strictly
# above this can only be a RAW term frequency. SAT_CEIL is the raw-vs-saturated
# discriminator (small epsilon over k1+1). Used both as the idempotency guard on
# the write path (never double-transform an already-saturated vector) and as the
# raw-subset detector for the re-saturation pass (task 212). Proof it never
# misclassifies a saturated point as raw: saturate() maps every input to
# v*(k1+1)/(v+denom_len) < k1+1 = 2.2 < 2.25 for all v,denom_len>0.
SAT_CEIL = 2.25


def client(url: str) -> QdrantClient:
    if QdrantClient is None:
        raise RuntimeError("qdrant-client is required for bm25_saturate runtime commands")
    return QdrantClient(url=url, timeout=300, check_compatibility=False)


def is_saturated_values(values) -> bool:
    """True if a stored bm25 weight list is already saturated (or empty).

    Saturated weights are bounded by k1+1=2.2; a raw tf vector has at least one
    weight > SAT_CEIL as soon as any token repeats enough to matter (the whole
    flood class). A raw vector whose every weight is <= SAT_CEIL (short doc, all
    tf small) is already bounded and harmless — treating it as saturated is safe
    (leaving it is equivalent to saturating it for ranking; both stay <2.25)."""
    if not values:
        return True
    return max(values) <= SAT_CEIL


def saturate_sparse(indices, values, avgdl: float = BASE_AVGDL,
                    k1: float = K1, b: float = B):
    """Return (indices, values) with values BM25-tf-saturated, dropping
    non-positive weights. dl = sum of the (positive) raw weights, exactly as the
    ADR-028 rebuild pass computes it. Identical math to saturate()/transform_points."""
    fidx, fvals = [], []
    for i, v in zip(indices, values):
        if v > 0:
            fidx.append(i)
            fvals.append(v)
    dl = float(sum(fvals))
    return fidx, saturate(fvals, dl, avgdl, k1, b)


def saturate_point_bm25(vec: dict, avgdl: float = BASE_AVGDL) -> bool:
    """Idempotently saturate the 'bm25' sparse vector of a point's vector dict
    IN PLACE. No-op if there is no bm25 vector or it is already saturated.
    Returns True iff a raw vector was transformed. This is THE shared live-upsert
    write-path guard (task 212 Part 1) — same transform as the ADR-028 rebuild,
    so a live-upserted point is byte-consistent with the saturated base build."""
    bm = vec.get("bm25")
    if bm is None:
        return False
    values = getattr(bm, "values", None)
    indices = getattr(bm, "indices", None)
    if not values or is_saturated_values(values):
        return False
    fidx, fvals = saturate_sparse(indices, values, avgdl)
    bm.indices, bm.values = fidx, fvals
    return True


def tok_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def saturate(values: list[float], dl: float, avgdl: float,
             k1: float = K1, b: float = B) -> list[float]:
    denom_len = k1 * (1.0 - b + b * dl / avgdl)
    return [v * (k1 + 1.0) / (v + denom_len) for v in values]


# --------------------------------------------------------------------------
# pass 1 — stats
# --------------------------------------------------------------------------

def cmd_stats(a: argparse.Namespace) -> None:
    import numpy as np

    c = client(a.url)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = None if a.no_df else np.zeros(2 ** 32, dtype=np.uint32)  # ~17 GB RAM
    n_points = 0
    n_bm25 = 0
    n_no_bm25 = 0
    sum_dl = 0.0
    sum_nnz = 0
    max_w = 0.0
    dl_hist = np.zeros(a.dl_hist_max + 1, dtype=np.int64)
    n_int_ids = 0

    offset = None
    t0 = time.time()
    while True:
        pts, offset = c.scroll(
            a.src, limit=a.batch, offset=offset,
            with_payload=False, with_vectors=["bm25"])
        if not pts:
            break
        batch_idx: list[np.ndarray] = []
        for p in pts:
            n_points += 1
            if isinstance(p.id, int):
                n_int_ids += 1
            vec = (p.vector or {}).get("bm25")
            if vec is None or not vec.values:
                n_no_bm25 += 1
                continue
            n_bm25 += 1
            dl = float(sum(vec.values))
            sum_dl += dl
            sum_nnz += len(vec.values)
            mw = max(vec.values)
            if mw > max_w:
                max_w = mw
            dl_hist[min(int(dl), a.dl_hist_max)] += 1
            if df is not None:
                batch_idx.append(np.asarray(vec.indices, dtype=np.int64))
        if df is not None and batch_idx:
            np.add.at(df, np.concatenate(batch_idx), 1)
        if n_points % a.report_every < a.batch:
            el = time.time() - t0
            print(f"[stats] {n_points:,} pts  {n_points/max(el,1e-9):,.0f} pts/s  "
                  f"avgdl_so_far={sum_dl/max(n_bm25,1):.1f}  elapsed={el/60:.1f}m",
                  flush=True)
        if a.sleep_ms:
            time.sleep(a.sleep_ms / 1000.0)
        if offset is None:
            break

    avgdl = sum_dl / max(n_bm25, 1)
    stats = {
        "src": a.src, "n_points": n_points, "n_bm25": n_bm25,
        "n_no_bm25": n_no_bm25, "avgdl": avgdl,
        "avg_nnz": sum_nnz / max(n_bm25, 1), "max_weight": max_w,
        "n_int_ids": n_int_ids, "k1": K1, "b": B,
        "elapsed_s": time.time() - t0,
    }
    (out / "bm25_stats.json").write_text(json.dumps(stats, indent=2))
    np.save(out / "dl_hist.npy", dl_hist)
    if df is not None:
        nz = np.nonzero(df)[0]
        np.savez_compressed(out / "df.npz", ids=nz.astype(np.uint32),
                            counts=df[nz])
        stats["vocab_size"] = int(nz.size)
        (out / "bm25_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


# --------------------------------------------------------------------------
# pass 2 — build target
# --------------------------------------------------------------------------

def ensure_dst(c: QdrantClient, src: str, dst: str, hnsw_m: int) -> None:
    if c.collection_exists(dst):
        print(f"[build] dst {dst} exists — appending")
        return
    info = c.get_collection(src)
    dense = info.config.params.vectors["dense"]
    from cytir.core import serving_profile as SP
    SP.create_collection(
        c, dst, SP.mirror_int8(dense.size, str(getattr(dense.distance, "value", dense.distance)).lower(),
                               hnsw_m),
        on_exists="keep", log=lambda *a: None)
    print(f"[build] created {dst} (hnsw m={hnsw_m}, config mirrors {src})")


def transform_points(pts, avgdl: float, raw: bool, counters: dict) -> list:
    out = []
    for p in pts:
        vecs = dict(p.vector or {})
        bm = vecs.get("bm25")
        if bm is None or not bm.values:
            counters["no_bm25"] += 1
            if bm is not None:  # empty vector — keep it empty
                vecs["bm25"] = m.SparseVector(indices=[], values=[])
            else:
                vecs.pop("bm25", None)
        elif raw:
            pass
        else:
            idx, vals = [], []
            for i, v in zip(bm.indices, bm.values):
                if v > 0:
                    idx.append(i)
                    vals.append(v)
                else:
                    counters["nonpos_weights"] += 1
            dl = float(sum(vals))
            vecs["bm25"] = m.SparseVector(
                indices=idx, values=saturate(vals, dl, avgdl))
        out.append(m.PointStruct(id=p.id, vector=vecs, payload=p.payload))
    return out


def _uuid_ge(pid, bound: str) -> bool:
    import uuid
    try:
        return uuid.UUID(str(pid)).int >= uuid.UUID(bound).int
    except ValueError:
        return False  # int ids sort before uuids in qdrant scroll order


def cmd_build(a: argparse.Namespace) -> None:
    c = client(a.url)
    if a.dst in (a.src, "aw_v3_b", "aw_v3_clean", "aw_v3", "aw_v4_sat", "aw_prod"):
        sys.exit(f"refusing to write to {a.dst}")
    avgdl = 0.0
    if not a.raw or True:  # raw control still records avgdl provenance
        stats = json.loads(Path(a.stats).read_text())
        avgdl = float(stats["avgdl"])
        if stats.get("src") != a.src:
            print(f"[build] WARNING stats src={stats.get('src')} != {a.src}")
    ensure_dst(c, a.src, a.dst, a.hnsw_m)

    counters = {"points": 0, "no_bm25": 0, "nonpos_weights": 0}
    t0 = time.time()

    def upsert(pts) -> None:
        batch = transform_points(pts, avgdl, a.raw, counters)
        c.upsert(a.dst, points=batch, wait=True)
        counters["points"] += len(batch)

    if a.ids_file:
        ids = [ln.strip() for ln in Path(a.ids_file).read_text().splitlines()
               if ln.strip()]
        for i in range(0, len(ids), a.batch):
            pts = c.retrieve(a.src, ids=ids[i:i + a.batch],
                             with_payload=True, with_vectors=True)
            upsert(pts)
        print(f"[build] ids-file: {counters['points']:,} points "
              f"({len(ids):,} requested)")
    else:
        offset = a.start_id
        done = False
        while not done:
            pts, offset = c.scroll(
                a.src, limit=a.batch, offset=offset,
                with_payload=True, with_vectors=True)
            if not pts:
                break
            if a.end_id:
                pts = [p for p in pts if not _uuid_ge(p.id, a.end_id)]
                if len(pts) < a.batch:
                    done = True
            if a.max_points and counters["points"] + len(pts) >= a.max_points:
                pts = pts[: a.max_points - counters["points"]]
                done = True
            if pts:
                upsert(pts)
            if counters["points"] % a.report_every < a.batch:
                el = time.time() - t0
                print(f"[build] {counters['points']:,} pts  "
                      f"{counters['points']/max(el,1e-9):,.0f} pts/s  "
                      f"elapsed={el/60:.1f}m  next={offset}", flush=True)
            if a.sleep_ms:
                time.sleep(a.sleep_ms / 1000.0)
            if offset is None:
                break

    counters["elapsed_s"] = round(time.time() - t0, 1)
    counters["rate_pts_s"] = round(counters["points"] / max(counters["elapsed_s"], 1e-9), 1)
    print(f"[build] DONE {json.dumps(counters)}")


# --------------------------------------------------------------------------
# finalize — hnsw m=16 + payload indexes
# --------------------------------------------------------------------------

def cmd_finalize(a: argparse.Namespace) -> None:
    c = client(a.url)
    src_schema = c.get_collection(a.src).payload_schema or {}
    for field, info in src_schema.items():
        try:
            c.create_payload_index(a.dst, field_name=field,
                                   field_schema=info.data_type, wait=False)
            print(f"[finalize] payload index {field} ({info.data_type})")
        except Exception as e:  # noqa: BLE001 — already exists is fine
            print(f"[finalize] index {field}: {e}")
    c.update_collection(a.dst, hnsw_config=m.HnswConfigDiff(m=16))
    print(f"[finalize] {a.dst}: hnsw m=16 requested — wait for green "
          f"(optimizer builds the graph in background)")


# --------------------------------------------------------------------------
# probe — bm25-leg-only query
# --------------------------------------------------------------------------

def cmd_probe(a: argparse.Namespace) -> None:
    c = client(a.url)
    toks = a.query.lower().split()
    qv = m.SparseVector(indices=[tok_hash(t) for t in toks],
                        values=[1.0] * len(toks))
    r = c.query_points(a.coll, query=qv, using="bm25", limit=a.k,
                       with_payload=["url", "title"])
    print(f"=== '{a.query}' on {a.coll} (bm25 leg only, k={a.k}) ===")
    for i, p in enumerate(r.points, 1):
        print(f"{i:3d}. {p.score:8.3f}  {p.payload.get('url','')[:95]}"
              f"  | {str(p.payload.get('title'))[:55]}")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://localhost:6333")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stats")
    s.add_argument("--src", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--batch", type=int, default=2048)
    s.add_argument("--no-df", action="store_true")
    s.add_argument("--dl-hist-max", type=int, default=20000)
    s.add_argument("--report-every", type=int, default=500_000)
    s.add_argument("--sleep-ms", type=int, default=0)

    b = sub.add_parser("build")
    b.add_argument("--src", required=True)
    b.add_argument("--dst", required=True)
    b.add_argument("--stats", required=True, help="bm25_stats.json from pass 1")
    b.add_argument("--batch", type=int, default=512)
    b.add_argument("--max-points", type=int, default=0)
    b.add_argument("--ids-file", default=None)
    b.add_argument("--start-id", default=None)
    b.add_argument("--end-id", default=None)
    b.add_argument("--raw", action="store_true", help="copy weights untouched")
    b.add_argument("--hnsw-m", type=int, default=0)
    b.add_argument("--report-every", type=int, default=50_000)
    b.add_argument("--sleep-ms", type=int, default=0)

    f = sub.add_parser("finalize")
    f.add_argument("--src", required=True)
    f.add_argument("--dst", required=True)

    p = sub.add_parser("probe")
    p.add_argument("--coll", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("-k", type=int, default=10)

    a = ap.parse_args()
    # task 864 (review 869 S1): from an aw_e10 T0 to the flip only the drain writes the twin
    from cytir.index import twin_write_hold as _TWH
    _TWH.refuse_cli_write(a, tool='cytir/core/bm25_saturate.py')
    {"stats": cmd_stats, "build": cmd_build,
     "finalize": cmd_finalize, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    main()
