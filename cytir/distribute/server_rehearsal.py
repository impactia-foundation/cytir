"""Server-side release rehearsal (task 2339): the C.2 and D.2 checks that CI cannot run.

CI rehearses on `qdrant-client` local mode (`cyt rehearse run`). Local mode renormalises
cosine vectors in numpy and counts BM25 `n` over every point, so two claims were only ever
shown on a few hundred rows: (C.2) the dump of an applied copy is bit-identical to the dump
it came from, and (D.2) the IDF snapshot's `n` is what a real server uses in its scores.
This module proves both on a real Qdrant server at a real size:

  1. read the first N points of a source collection (read-only scroll, a limit, no filter)
     into a scratch collection built from the serving profile;
  2. `cyt dump` the scratch slice;
  3. `cyt apply` that dump into a second scratch collection;
  4. `cyt dump` the applied collection with the same options;
  5. compare the two dumps file by file (sha256) and the manifests field by field;
  6. D.2: the applied count equals the dump's declared `n`; the snapshot's `n` equals the
     number of rows with a sparse leg; real BM25 scores equal `stored_weight * idf` from the
     pinned snapshot.

Opt-in `--upgrade` (task 2404) adds the upgrade stage (task 2308) on the same scratch server:
`rehearse_upgrade` installs N-1 derived from the `slice` collection into a new `upgrade`
collection, applies delta N and checks the copy equals N. It holds the whole slice in memory,
so size `--rows` for the box.

Run it with `release-rehearsal-server.sh` under `deploy/` (it starts the scratch server)."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Optional

from cytir.distribute import apply, dump, idf, pack

#: Manifest fields that differ between two dumps of the same rows by construction.
VOLATILE = ("dump_id", "created_at")


def _du(path: pathlib.Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def copy_slice(src, src_name: str, dst, dst_name: str, rows: int, *, log=print) -> int:
    """Copy the first `rows` scrolled points (vectors and payload, untouched) from `src`
    (read-only) into a new `dst_name` collection built from the prod-e9 profile."""
    from qdrant_client.http import models as qm

    from cytir.core import serving_profile as SP

    SP.create_collection(dst, dst_name, SP.PROD_E9, on_exists="refuse", bulk=True,
                         log=lambda m: None)
    n, off, t0 = 0, None, time.monotonic()
    while n < rows:
        pts, off = src.scroll(src_name, limit=min(1000, rows - n), offset=off,
                              with_payload=True, with_vectors=True)
        if not pts:
            break
        dst.upsert(dst_name, points=[qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                                     for p in pts])
        n += len(pts)
        if n % 100_000 < 1000:
            log(f"copied {n} in {time.monotonic() - t0:.0f} s", flush=True)
        if off is None:
            break
    return n


def compare_dumps(a: pathlib.Path, b: pathlib.Path) -> dict:
    """Bit-exactness as C.2 defines it: every file the manifest lists (shards, fingerprint
    sidecar, IDF snapshot) has the same sha256 in both dumps (same bytes means same rows in
    the same order), and the manifests agree on every field but `VOLATILE` ones and the
    source collection's name. On a difference, names the first shard and row that differ."""
    ma = json.loads((a / "manifest.json").read_text())
    mb = json.loads((b / "manifest.json").read_text())
    out: dict = {"equal": True, "files": 0, "first_difference": None}

    def strip(m: dict) -> dict:
        m = {k: v for k, v in m.items() if k not in VOLATILE}
        m["source"] = {k: v for k, v in m["source"].items() if k != "collection"}
        return m

    if strip(ma) != strip(mb):
        keys = sorted(k for k in set(ma) | set(mb)
                      if k not in VOLATILE and k != "source" and ma.get(k) != mb.get(k))
        out["equal"] = False
        out["first_difference"] = {"manifest_fields": keys}
    paths = [s["path"] for s in ma["shards"]]
    fp = (ma.get("fingerprints") or {}).get("path")
    snap = ((ma.get("sparse") or {}).get("idf_snapshot") or {}).get("path")
    paths += [p for p in (fp, snap) if p]
    for p in paths:
        out["files"] += 1
        if (b / p).exists() and pack.sha256_file(str(a / p)) == pack.sha256_file(str(b / p)):
            continue
        out["equal"] = False
        if out["first_difference"]:
            continue
        if p.endswith(".parquet") and (b / p).exists():
            ra, rb = (list(pack.read_parquet_shard(str(d / p))) for d in (a, b))
            diff = {"shard": p, "rows": [len(ra), len(rb)]}
            for i, (x, y) in enumerate(zip(ra, rb)):
                if x != y:
                    diff = {"shard": p, "row": i, "id": x["id"],
                            "field": next((k for k in x if x[k] != y.get(k)), "?")}
                    break
            out["first_difference"] = diff
        else:
            out["first_difference"] = {"file": p}
    return out


def idf_server_check(client, collection: str, dump_dir: pathlib.Path, *, probes: int = 6) -> dict:
    """D.2: on a real server `n` in `ln((n+1)/(df+0.5))` is the count of points that carry a
    sparse vector. Checks the snapshot's `n` against the dump's sparse rows and the server's
    own scores against `stored_weight * idf(snapshot n, df)` for single-term queries."""
    from qdrant_client.http import models as qm

    manifest = json.loads((dump_dir / "manifest.json").read_text())
    blk = manifest["sparse"]["idf_snapshot"]
    df = dict(idf.iter_snapshot(dump_dir / blk["path"]))
    sparse_rows = 0
    for s in manifest["shards"]:
        sparse_rows += sum(1 for r in pack.read_parquet_shard(str(dump_dir / s["path"]))
                           if r["sparse_indices"] is not None)
    terms = sorted(df, key=lambda t: (df[t], t))
    chosen = terms[::max(1, len(terms) // probes)][:probes]
    worst = 0.0
    for term in chosen:
        hit = client.query_points(collection, query=qm.SparseVector(indices=[term], values=[1.0]),
                                  using=dump.SPARSE_VECTOR_NAME, limit=1,
                                  with_vectors=[dump.SPARSE_VECTOR_NAME]).points[0]
        sv = hit.vector[dump.SPARSE_VECTOR_NAME]
        w = dict(zip(sv.indices, sv.values))[term]
        want = w * idf.idf(blk["n"], df[term])
        worst = max(worst, abs(hit.score - want) / max(1.0, abs(want)))
    return {"snapshot_n": blk["n"], "dump_sparse_rows": sparse_rows,
            "n_equals_sparse_rows": blk["n"] == sparse_rows, "terms_scored": len(chosen),
            "worst_relative_score_error": worst, "scores_match": worst <= 1e-4}


SERVING_PORTS = (6333, 6486, 6485)
SCRATCH_COLLECTIONS = ("slice", "applied", "upgrade")


class ScratchRefusal(RuntimeError):
    pass


def _hostport(url: str) -> tuple[str, int]:
    from urllib.parse import urlparse

    u = urlparse(url)
    return (u.hostname or "").lower(), u.port or (443 if u.scheme == "https" else 80)


def _is_loopback(host: str) -> bool:
    import ipaddress

    import socket

    try:
        return ipaddress.ip_address(socket.gethostbyname(host)).is_loopback
    except (OSError, ValueError):
        return False


def check_scratch(twin_url: str, scratch_url: str, scratch) -> None:
    """Refuse before any write: the scratch server must be neither the source nor a serving
    port, and must hold none of the collections this run creates (nor an alias to one)."""
    t, s = _hostport(twin_url), _hostport(scratch_url)
    if t == s or (t[1] == s[1] and _is_loopback(t[0]) and _is_loopback(s[0])):
        raise ScratchRefusal(f"REFUSED: --scratch-url {scratch_url} is the twin")
    if s[1] in SERVING_PORTS:
        raise ScratchRefusal(f"REFUSED: --scratch-url port {s[1]} is a serving port")
    for name in SCRATCH_COLLECTIONS:
        if scratch.collection_exists(name):
            raise ScratchRefusal(f"REFUSED: scratch server already has a collection {name!r}")
    for a in scratch.get_aliases().aliases:
        if a.alias_name in SCRATCH_COLLECTIONS or a.collection_name in SCRATCH_COLLECTIONS:
            raise ScratchRefusal(
                f"REFUSED: scratch server has alias {a.alias_name!r} -> {a.collection_name!r}")


def upgrade_stage(scratch, work: pathlib.Path, dump_manifest: dict) -> dict:
    """The upgrade rehearsal against the scratch server (task 2404). N is the `slice`
    collection (read), N-1 is built into `upgrade` (written). The caller has already passed
    `check_scratch` and `refuse_cli_write`; this checks again that the client is not a serving
    port and that `upgrade` is absent, because it is the only function here that is public."""
    from cytir.distribute import rehearsal as R

    if scratch.collection_exists("upgrade"):
        raise ScratchRefusal("REFUSED: scratch server already has a collection 'upgrade'")
    sec, pub = R._ephemeral_keys(work)
    fx = {"embedder": dump_manifest["embedder"], "sparse": dump_manifest.get("sparse")}
    # No query vectors in the server slice: the top-1 comparison runs on zero queries; the
    # row-level equality and the re-dump check carry the proof.
    return R.rehearse_upgrade(scratch, fx, {"queries": []}, work, sec, pub, [],
                              client=scratch, src_name="slice", target="upgrade")


def run(args) -> dict:
    from cytir.core.qdrant import qdrant_client

    work = pathlib.Path(args.work)
    scratch = qdrant_client(args.scratch_url, timeout=300)
    check_scratch(args.twin_url, args.scratch_url, scratch)
    from cytir.index import twin_write_hold as _TWH
    # args not passed: twin_collection is the READ source, only the scratch write target is checked
    _TWH.refuse_cli_write(tool='cytir/distribute/server_rehearsal.py', extra=('slice', 'upgrade') if getattr(args, 'upgrade', False) else ('slice',))
    work.mkdir(parents=True, exist_ok=True)
    twin = qdrant_client(args.twin_url, timeout=300)
    rep: dict = {"requested_rows": args.rows, "shard_rows": args.shard_rows}
    common = ["--qdrant-url", args.scratch_url, "--i-know-this-is-not-the-twin",
              "--license-threshold", str(args.license_threshold),
              "--shard-rows", str(args.shard_rows)]
    t = time.monotonic()
    rep["copied"] = copy_slice(twin, args.twin_collection, scratch, "slice", args.rows)
    rep["copy_s"] = round(time.monotonic() - t)

    d1, d2 = work / "dump-1", work / "dump-2"
    t = time.monotonic()
    dump.main(["--collection", "slice", "--out", str(d1), "--dump-id", "rehearsal"] + common)
    rep["dump_s"] = round(time.monotonic() - t)
    m1 = json.loads((d1 / "manifest.json").read_text())
    rep["n"] = m1["counts"]["points"]
    rep["dump_bytes"] = _du(d1)
    rep["shards"] = m1["counts"]["shards"]

    t = time.monotonic()
    rep["apply_rc"] = apply.main(
        ["--manifest", str(d1), "--qdrant-url", args.scratch_url, "--collection", "applied",
         "--allow-unsigned", "--pause-s", "0", "--state-file", str(work / "apply-state.json")])
    rep["apply_s"] = round(time.monotonic() - t)

    t = time.monotonic()
    dump.main(["--collection", "applied", "--out", str(d2), "--dump-id", "rehearsal"] + common)
    rep["redump_s"] = round(time.monotonic() - t)
    rep["redump_bytes"] = _du(d2)

    rep["applied_count"] = scratch.count("applied", exact=True).count
    rep["count_ok"] = rep["applied_count"] == rep["n"] == rep["copied"]
    rep["manifest_sha256"] = [pack.sha256_file(str(d / "manifest.json")) for d in (d1, d2)]
    rep["compare"] = compare_dumps(d1, d2)
    rep["idf_d2"] = idf_server_check(scratch, "applied", d1)
    if getattr(args, "upgrade", False):
        t = time.monotonic()
        rep["upgrade"] = upgrade_stage(scratch, work, m1)
        rep["upgrade_s"] = round(time.monotonic() - t)
    return rep


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="cyt rehearse server", description=__doc__.split("\n")[0])
    ap.add_argument("--twin-url", required=True, help="source Qdrant, read only")
    ap.add_argument("--twin-collection", required=True)
    ap.add_argument("--scratch-url", required=True, help="empty scratch Qdrant (written to)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--shard-rows", type=int, default=50_000)
    ap.add_argument("--license-threshold", type=float, default=0.8)
    ap.add_argument("--upgrade", action="store_true",
                    help="also run the upgrade rehearsal (N-1 then delta N) on the scratch server")
    ap.add_argument("--report", default=None, help="write the JSON report here")
    args = ap.parse_args(argv)
    rep = run(args)
    text = json.dumps(rep, indent=2, sort_keys=True)
    print(text)
    if args.report:
        pathlib.Path(args.report).write_text(text + "\n")
        if "upgrade" in rep:
            up = pathlib.Path(args.report)
            up.with_name(up.stem + ".upgrade" + up.suffix).write_text(
                json.dumps(rep["upgrade"], indent=2, sort_keys=True) + "\n")
    ok = (rep["compare"]["equal"] and rep["count_ok"] and rep["idf_d2"]["n_equals_sparse_rows"]
          and rep["idf_d2"]["scores_match"] and rep["apply_rc"] in (0, None))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
