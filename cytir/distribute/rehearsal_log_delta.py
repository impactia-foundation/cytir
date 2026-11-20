"""The log-delta stage of `cyt rehearse` (task 2850; `specs/delta-change-log.md` section 5, tests 1, 2).

A base is dumped; a drain-shaped sequence of packs lands with the change log ON; a delta is built from
the LOG and, separately, by the scroll route; the two are compared row for row; the log delta is applied
onto a fresh install of the base; the copy equals the changed source (`redump_check`, which includes the
0-row 0-tombstone self-delta). The three refusals of section 3.4 (rebase, gap, missing newest segment)
run on copies of the same log.

What is independent between the two sides: the id set and the tombstone ids (and their reasons, which
the scroll side takes from the hard-coded `EXPECTED_PUBLIC_REASON`, not from `change_log`). What is NOT
independent: the row bytes (both sides read the same source store) and the add/replace kind (one
function computes it for both).

Spec section 5 test 1's "through the real drain code path" clause is only PARTLY met. The drain's
write legs talk HTTP to a twin and a prod and cannot run on an in-memory collection. The stage issues
the same kinds of Qdrant write (upsert, delete, set_payload) itself and lands each pack's record through
`promote_queue._log_landing` with HAND-BUILT plans. The drain's own plan construction (callers of
`_log_landing` in `promote_queue.py`) and its write and verify legs are NOT exercised. Follow-up: run the
real plan builders against a stubbed in-memory write leg. The stage touches no twin, no prod, no network,
and `_drain_sequence` refuses any client that is not a local in-memory one."""
from __future__ import annotations

import hashlib
import json
import os
from cytir.core import envnames as _envnames
import pathlib
import shutil
import time

from cytir.core import paths
from cytir.core import serving_profile as SP
from cytir.core.ids import mint_point_id
from cytir.core.qdrant import qdrant_client
from cytir.distribute import apply, dump, idf, pack
from cytir.distribute import rehearsal as R
from cytir.index import change_log as CL

LOG_ADDS, LOG_REPLACES, LOG_TOMBS, LOG_PATCHES, LOG_REVERTS, LOG_EPHEMERAL = 20, 40, 60, 10, 3, 3
#: The base dump's cut point while the log is still empty: older than any segment stamp, seq 0.
ZERO_CUT = {"stamp": "19700101T000000Z", "seq": 0}


def _point_of(row: dict):
    from qdrant_client.http import models as qm
    vector: dict = {"dense": row["dense"]}
    if row["sparse_indices"] is not None:
        vector[dump.SPARSE_VECTOR_NAME] = qm.SparseVector(
            indices=row["sparse_indices"], values=row["sparse_values"])
    return qm.PointStruct(id=row["id"], vector=vector, payload=json.loads(row["payload"]))


def _land(log_dir: pathlib.Path, pack_id: str, plan: dict) -> None:
    """One landed pack's segment, through the drain's own `_log_landing` seam."""
    from cytir.index import promote_queue as Q
    old = os.environ.get("AW_CHANGE_LOG_DIR")
    os.environ["AW_CHANGE_LOG_DIR"] = str(log_dir)
    paths.reset_settings_cache()
    try:
        Q._log_landing({"pack_id": pack_id, "status": "LANDED"}, plan,
                       manifest_sha=hashlib.sha256(pack_id.encode()).hexdigest())
    finally:
        if old is None:
            _envnames.pop_knob(os.environ, "AW_CHANGE_LOG_DIR")
        else:
            os.environ["AW_CHANGE_LOG_DIR"] = old
        paths.reset_settings_cache()


#: Hard-coded on purpose: the queue reason -> public reason table the planner expects. It must not be
#: read from `change_log`, or a wrong entry there would pass on both sides.
EXPECTED_PUBLIC_REASON = {
    "superseded": "superseded", "aged_out": "aged_out", "takedown": "takedown",
    "superseded_generation": "superseded_generation",
    "canonical_duplicate": "superseded_generation", "binary_content": "superseded_generation",
    "soft_404": "superseded_generation", "restore_rollback": "superseded_generation",
}


def _require_local_client(client) -> None:
    """Refuse (before any write) a client that is not qdrant-client's local in-memory engine."""
    from qdrant_client.local.qdrant_local import QdrantLocal
    inner = getattr(client, "_client", None)
    if not isinstance(inner, QdrantLocal) or getattr(inner, "location", None) != ":memory:":
        raise R.RehearsalFailure(
            f"the log-delta stage writes only to a local in-memory collection, got {type(client).__name__}")


def _drain_sequence(src, rows: list[dict], log_dir: pathlib.Path) -> dict:
    """Apply six packs to `src`, logging each: adds; title replaces; tombstones over every queue
    reason; a payload patch plus a patch that is reverted in the next pack; rows added and then
    tombstoned inside the window. Returns what a planner knows independently of the log."""
    _require_local_client(src)
    from qdrant_client.http import models as qm
    n = len(rows)
    R._check(n >= 160, "the fixture is too small for the log-delta stage")
    step = n // 160
    take = iter([rows[i * step] for i in range(160)])
    tombs = [next(take) for _ in range(LOG_TOMBS)]
    repl = [next(take) for _ in range(LOG_REPLACES)]
    patch = [next(take) for _ in range(LOG_PATCHES)]
    revert = [next(take) for _ in range(LOG_REVERTS)]
    templates = [next(take) for _ in range(LOG_ADDS + LOG_EPHEMERAL)]

    def clone(i: int, r: dict, tag: str) -> dict:
        p = json.loads(r["payload"])
        p["url"] = p["url"] + f"#{tag}-{i}"
        return R._with_payload(r, p, mint_point_id(f"https://rehearsal.invalid/{tag}/{i}/{r['id']}", ""))

    adds = [clone(i, r, "logadd") for i, r in enumerate(templates[:LOG_ADDS])]
    eph = [clone(i, r, "logeph") for i, r in enumerate(templates[LOG_ADDS:])]
    # pack 1: adds
    src.upsert(R.SOURCE, points=[_point_of(r) for r in adds])
    _land(log_dir, "rehearsal-p1-adds", {"upserts": [r["id"] for r in adds], "op": "add"})
    # pack 2: title replaces (a content pack re-writes the row)
    pts = []
    for r in repl:
        p = json.loads(r["payload"])
        p["title"] = f"REPLACED {p.get('title') or ''}"
        pts.append(_point_of(R._with_payload(r, p)))
    src.upsert(R.SOURCE, points=pts)
    _land(log_dir, "rehearsal-p2-replaces", {"upserts": [r["id"] for r in repl], "op": "upsert"})
    # pack 3: real deletes, every queue reason (each maps to a public one)
    qreasons = sorted(EXPECTED_PUBLIC_REASON)
    reasons_q = {r["id"]: qreasons[i % len(qreasons)] for i, r in enumerate(tombs)}
    src.delete(R.SOURCE, points_selector=qm.PointIdsList(points=[r["id"] for r in tombs]))
    _land(log_dir, "rehearsal-p3-tombstones",
          {"tombstones": [{"id": i, "reason": q} for i, q in reasons_q.items()]})
    # pack 4: a payload patch, and a patch the next pack reverts
    for r in patch:
        src.set_payload(R.SOURCE, payload={"snippet": "PATCHED " + (json.loads(r["payload"]).get("snippet") or "")},
                        points=[r["id"]])
    for r in revert:
        src.set_payload(R.SOURCE, payload={"snippet": "TEMPORARY"}, points=[r["id"]])
    _land(log_dir, "rehearsal-p4-patch", {"upserts": [r["id"] for r in patch + revert], "op": "replace"})
    # pack 5: the reverted rows go back; ephemeral rows are added. pack 6: the ephemeral rows go.
    for r in revert:
        src.set_payload(R.SOURCE, payload={"snippet": json.loads(r["payload"]).get("snippet")},
                        points=[r["id"]])
    src.upsert(R.SOURCE, points=[_point_of(r) for r in eph])
    _land(log_dir, "rehearsal-p5-revert-and-ephemeral",
          {"upserts": [r["id"] for r in revert + eph], "op": "upsert"})
    src.delete(R.SOURCE, points_selector=qm.PointIdsList(points=[r["id"] for r in eph]))
    _land(log_dir, "rehearsal-p6-ephemeral-gone",
          {"tombstones": [{"id": r["id"], "reason": "superseded"} for r in eph]})
    return {"adds": {r["id"] for r in adds}, "replaces": {r["id"] for r in repl},
            "patches": {r["id"] for r in patch}, "reverts": {r["id"] for r in revert},
            "ephemeral": {r["id"] for r in eph},
            "tombstones": {i: EXPECTED_PUBLIC_REASON[q] for i, q in reasons_q.items()}}


def rehearse_log_delta(fixture_dir: pathlib.Path, work: pathlib.Path, sec: str, pub: str) -> dict:
    """Raises `RehearsalFailure`. Compared, log delta vs scroll delta: the (id, op) set of the changed
    rows (add = id absent from the base, replace = present); per shared id the whole row (id, dense
    floats, sparse leg, payload text) by equality, by `pack.row_fingerprint` and by payload bytes; the
    tombstones as (id, reason), the scroll side taking its reasons from the planner, not from the log.
    NOT comparable: shard file bytes (two writers, no promised row order) and the manifests (they
    name different sources)."""
    t0 = time.monotonic()
    src, fx_manifest = R._load_into_memory(fixture_dir, R.SOURCE)   # own copy: this stage mutates it
    rows = sorted(dump.scroll_rows(src, R.SOURCE), key=lambda r: pack.point_id_key(r["id"]))
    log_dir = work / "log-delta-changelog"
    log_dir.mkdir()

    # 1. release N-1: a signed full dump; the log is empty, so its cut is the zero cut
    prev_dir = work / "log-delta-base"
    idf_acc = idf.IdfAccumulator()
    shards, total, fps = dump.write_full_shards(iter(rows), prev_dir, shard_rows=200, idf=idf_acc)
    prev = dump.build_manifest(
        dump_id="rehearsal-log-base", collection=R.SOURCE, dims=fx_manifest["embedder"]["dims"],
        distance="cosine", shards=shards, total_points=total,
        any_row_has_sparse=fx_manifest.get("sparse") is not None,
        idf_snapshot=idf_acc.write(prev_dir), fingerprints=fps, profile=SP.PROD_E9)
    prev["change_log_segment"] = dict(ZERO_CUT)
    prev_path = R._write_signed(prev, prev_dir, sec)
    base_ids = {r["id"] for r in rows}

    # 2. the drain-shaped sequence, change log on
    plan = _drain_sequence(src, rows, log_dir)
    segments = sorted(p.name for p in log_dir.glob("*.jsonl"))
    R._check(len(segments) == 6, f"expected 6 segments, found {len(segments)}")

    # 3a. the delta from the LOG
    builder, note = dump.choose_delta_source(prev, prev_dir, log_dir=log_dir, client=src,
                                             collection=R.SOURCE)
    R._check(builder is not None, f"the log delta was not chosen: {note}")
    dlt_dir = work / "log-delta-pack"
    log_rows: dict[str, dict] = {}
    d_shards, d_total = dump.write_shards(
        (log_rows.setdefault(r["id"], r) for r in builder.changed_rows()), dlt_dir, shard_rows=200)
    log_tombs = builder.tombstones()

    # 3b. the same delta by the scroll route
    sb = dump.DeltaBuilder(prev, prev_dir)
    scroll_rows = {r["id"]: r for r in sb.changed_rows(dump.scroll_rows(src, R.SOURCE))}
    scroll_tombs = sb.tombstones(dict(plan["tombstones"]))

    # 4. row for row
    def kind(i: str) -> str:
        return "add" if i not in base_ids else "replace"

    got_l = {(i, kind(i)) for i in log_rows}
    got_s = {(i, kind(i)) for i in scroll_rows}
    R._check(got_l == got_s, f"log delta and scroll delta differ in (id, op): "
             f"only log {sorted(got_l - got_s)[:3]}, only scroll {sorted(got_s - got_l)[:3]}")
    for i, r in log_rows.items():
        o = scroll_rows[i]
        R._check(r == o, f"row {i} differs between the log delta and the scroll delta")
        R._check(pack.row_fingerprint(r) == pack.row_fingerprint(o), f"fingerprint of {i} differs")
        R._check(r["payload"].encode() == o["payload"].encode(), f"payload bytes of {i} differ")
    tl = sorted((t.id, t.reason) for t in log_tombs)
    ts = sorted((t.id, t.reason) for t in scroll_tombs)
    R._check(tl == ts, "tombstones (id, reason) differ between the log delta and the scroll delta")
    ops = {"add": sum(k == "add" for _, k in got_l), "replace": sum(k == "replace" for _, k in got_l)}
    want = {"add": len(plan["adds"]), "replace": len(plan["replaces"]) + len(plan["patches"])}
    R._check(ops == want, f"delta holds {ops}, the planner expected {want}")
    R._check(len(tl) == LOG_TOMBS and {x for _, x in tl} == set(CL.PUBLIC_TOMBSTONE_REASONS),
             "the tombstones do not carry every public reason")
    R._check(not (plan["ephemeral"] | plan["reverts"]) & log_rows.keys()
             and builder.noops >= len(plan["reverts"]),
             "an add-then-delete row or a reverted patch reached the delta")
    R._check(not {i for i, _ in tl} & plan["ephemeral"], "a never-shipped id got a tombstone")

    # 5. the pack, signed; the stranger installs the base and applies the log delta
    n_tomb, tomb_sha = pack.write_tombstones(log_tombs, str(dlt_dir / "tombstones.jsonl.gz"))
    delta = dump.build_manifest(
        dump_id="rehearsal-log-delta", collection=R.SOURCE, dims=fx_manifest["embedder"]["dims"],
        distance="cosine", shards=d_shards, total_points=d_total,
        any_row_has_sparse=fx_manifest.get("sparse") is not None, profile=SP.PROD_E9)
    delta.update({"schema": pack.DELTA_PACK_SCHEMA, "pack_id": "rehearsal-log-delta", "sequence": 1,
                  "base_dump_id": prev["dump_id"],
                  "base_manifest_sha256": pack.sha256_file(str(prev_path)),
                  "change_log_segment": dict(builder.last_cut),
                  "delta_source": {"path": dump.SOURCE_CHANGE_LOG, "note": note},
                  "tombstones": {"path": "tombstones.jsonl.gz", "count": n_tomb, "sha256": tomb_sha}})
    R._write_signed(delta, dlt_dir, sec)
    apply.verify_dump_signature(prev, prev_dir, pubkey=pub, allow_unsigned=False)
    cp = qdrant_client(":memory:")
    apply.create_collection(cp, R.TARGET, prev, log=lambda m: None)
    apply.upsert_shards(cp, R.TARGET, prev, prev_dir)
    apply.verify_dump_signature(delta, dlt_dir, pubkey=pub, allow_unsigned=False)
    apply.verify_shard_hashes(delta, dlt_dir)
    R._check(apply.verify_tombstones(delta, dlt_dir) == n_tomb, "tombstone count differs")
    n_up = apply.upsert_shards(cp, R.TARGET, delta, dlt_dir)
    n_del = apply.apply_tombstones(cp, R.TARGET, delta, dlt_dir)
    R._check(n_up == d_total and n_del == n_tomb, f"applied {n_up} rows / {n_del} tombstones")
    count = cp.count(R.TARGET, exact=True).count
    R._check(count == src.count(R.SOURCE, exact=True).count == len(rows) + LOG_ADDS - LOG_TOMBS,
             f"the copy holds {count} rows")
    redump = R.redump_check(src, R.SOURCE, cp, R.TARGET, work / "log-delta-redump")

    # 6. the three refusals of section 3.4, on copies of this log
    def logcopy(name: str) -> pathlib.Path:
        d = work / name
        shutil.copytree(log_dir, d)
        return d

    def refused(label: str, exc, fn) -> None:
        try:
            fn()
        except exc:
            return
        raise R.RehearsalFailure(f"{label} was NOT refused")

    def choose(d):
        return lambda: dump.choose_delta_source(prev, prev_dir, log_dir=d, client=src,
                                                collection=R.SOURCE)

    rb = logcopy("log-delta-rebase")
    CL.record_rebase("rehearsal-flip", {"why": "rehearsal"}, rb)
    refused("a rebase inside the window", dump.DeltaRefusal, choose(rb))
    gp = logcopy("log-delta-gap")
    sorted(gp.glob("*.jsonl"))[2].unlink()
    refused("a seq gap", dump.DeltaRefusal, lambda: dump.LogDeltaBuilder(
        prev, prev_dir, gp, client=src, collection=R.SOURCE, after=dict(ZERO_CUT)))
    mn = logcopy("log-delta-missing-newest")
    sorted(mn.glob("*.jsonl"))[-1].unlink()
    refused("a missing newest segment", CL.MissingSegmentError, choose(mn))
    return {"base_points": total, "segments": len(segments), "delta_rows": d_total,
            "delta_ops": ops, "tombstones": n_tomb,
            "tombstone_reasons": sorted({x for _, x in tl}), "net_noops_dropped": builder.noops,
            "log_equals_scroll": True, "points_after": count, "redump": redump, "refusals": 3,
            "seconds": round(time.monotonic() - t0, 2)}
