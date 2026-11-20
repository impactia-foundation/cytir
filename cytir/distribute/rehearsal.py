#!/usr/bin/env python3
"""cytir.distribute.rehearsal — the release rehearsal (task 687; 643 §6.2 S8).

Release-doctrine #4: a release is applied with the published tool before
anyone else sees it, and the result is judged by the resulting state. This
module is that rehearsal as code, so CI and a stranger run the same thing:

    cyt rehearse run [--fixture DIR]      (CI runs exactly this verb)

`run` takes the SEALED fixture (a few hundred real twin points, already
projected through `pack.project_payload`, committed under
`data/release-rehearsal-fixture/`), loads it as the "source" collection,
then does exactly what a release does: dump -> sign -> verify -> apply into
a second, empty collection -> answer a fixed query set. It fails unless

  * the signature verifies and a tampered manifest, a tampered shard and a
    wrong key are each REFUSED,
  * the applied collection returns the same top-1 id as the source for every
    query, and the sealed id recorded when the fixture was built,
  * no readable text in the applied collection carries the register-officer
    marker phrase or any key `pack.py` does not name.

`build-fixture` (run on the box that has the twin; READ-ONLY) draws the
fixture from a collection by Qdrant's random sample, projects it, drops rows
with no `license` (the real dump would refuse that coverage; the count is
recorded, not hidden) and seals the expected top-1 per query.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cytir.core import serving_profile as SP
from cytir.core.ids import mint_point_id
from cytir.core.paths import data_path
from cytir.core.qdrant import qdrant_client
from cytir.distribute import apply, dump, idf, pack, torrent
from cytir.distribute.sign import minisign_available, sign_manifest

FIXTURE_SCHEMA = "aw.rehearsal-fixture.v1"
DEFAULT_FIXTURE_DIR = data_path("release-rehearsal-fixture")
SOURCE = "rehearsal_source"
TARGET = "rehearsal_target"
#: The invariant officer-clause marker (pack.py `_OFFICER_CLAUSE_RE`).
OFFICER_MARKER = "registered signatories/officers"
FORBIDDEN_KEYS = ("dense_text", "sparse_text", "facet_terms", "names")


class RehearsalFailure(AssertionError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise RehearsalFailure(msg)


#: Fields that must never reach a stranger (ADR-039: owner/attribution/licence_version are
#: master-only; memo 645: no register person names as text; `dense_text` etc.).
MASTER_ONLY = frozenset({"owner", "attribution", "license_version", "author", "outlinks",
                         "consent_checked_at", "consent_source", "_master", "family_id"})


def _scroll_all(client, collection: str):
    offset = None
    while True:
        pts, offset = client.scroll(collection, limit=500, offset=offset, with_payload=True)
        yield from pts
        if offset is None:
            return


def _top1(client, collection: str, vector: list[float]) -> Optional[str]:
    r = client.query_points(collection, query=vector, using="dense", limit=1)
    return str(r.points[0].id) if r.points else None


def _load_into_memory(fixture_dir: pathlib.Path, collection: str):
    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    pack.validate_manifest(manifest)
    apply.verify_shard_hashes(manifest, fixture_dir)
    client = qdrant_client(":memory:")
    # The committed fixture predates the profile block, and an in-memory source store
    # cannot hold a layout anyway: the old bare layout is named on purpose here.
    apply.create_collection(client, collection, manifest, legacy_layout=True,
                            log=lambda m: None)
    apply.upsert_shards(client, collection, manifest, fixture_dir)
    return client, manifest


# ---------------------------------------------------------------- build
def build_fixture(client, collection: str, out_dir: pathlib.Path, *,
                  n_points: int, n_queries: int) -> dict:
    """READ-ONLY on `client`. Writes the fixture dump + fixture.json."""
    from qdrant_client.http import models as qm

    def sample(n: int):
        return client.query_points(
            collection, query=qm.SampleQuery(sample=qm.Sample.RANDOM),
            limit=n, with_payload=True, with_vectors=True).points

    drawn = sample(n_points * 2)
    rows, dropped = [], 0
    for p in drawn:
        row = dump.point_to_row(p)
        if "license" not in json.loads(row["payload"]):
            dropped += 1
            continue
        rows.append(row)
        if len(rows) == n_points:
            break
    _check(len(rows) == n_points, f"only {len(rows)} licensed rows drawn")
    rows.sort(key=lambda r: pack.point_id_key(r["id"]))  # the sidecar is id-sorted
    in_slice = {r["id"] for r in rows}
    queries = [{"vector": [float(x) for x in p.vector["dense"]]}
               for p in sample(n_queries * 3) if str(p.id) not in in_slice][:n_queries]
    _check(len(queries) == n_queries, "could not draw enough query vectors")

    out_dir.mkdir(parents=True, exist_ok=True)
    shards, total, fps = dump.write_full_shards(iter(rows), out_dir,
                                                shard_rows=n_points)
    dims = len(rows[0]["dense"])
    manifest = dump.build_manifest(
        dump_id="rehearsal-fixture", collection=collection, dims=dims,
        distance="cosine", shards=shards, total_points=total,
        any_row_has_sparse=any(r["sparse_indices"] is not None for r in rows),
        fingerprints=fps, profile=SP.PROD_E9)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    src, _ = _load_into_memory(out_dir, SOURCE)
    for q in queries:
        q["expected_top1"] = _top1(src, SOURCE, q["vector"])
    meta = {"schema": FIXTURE_SCHEMA, "source_collection": collection,
            "sampled_at": dump.now_iso(), "points": total,
            "dropped_no_license": dropped, "queries": queries}
    (out_dir / "fixture.json").write_text(json.dumps(meta))
    return meta


def refresh_fixture(client, collection: str, fixture_dir: pathlib.Path, *,
                    batch: int = 100) -> dict:
    """Re-project the SEALED fixture's rows through the CURRENT `pack.project_payload`, reading
    each row's payload from the twin by id (READ-ONLY `retrieve`, no scroll). Vectors, ids and
    the query vectors do not change: only the payload form does. Used when the declared projection
    gains a field (task 768: digest + consent flags). A row the twin no longer holds (retired
    since the fixture was drawn) is dropped and COUNTED in `fixture.json`, and every query's
    `expected_top1` is re-sealed on the surviving rows."""
    meta = json.loads((fixture_dir / "fixture.json").read_text())
    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    old = [r for s_ in manifest["shards"]
           for r in pack.read_parquet_shard(str(fixture_dir / s_["path"]))]
    rows, missing = [], []
    for i in range(0, len(old), batch):
        chunk = old[i:i + batch]
        got = {str(p.id): p for p in client.retrieve(
            collection, ids=[r["id"] for r in chunk], with_payload=True, with_vectors=False)}
        for r in chunk:
            p = got.get(r["id"])
            if p is None:
                missing.append(r["id"])
                continue
            rows.append(pack.row_to_record(
                r["id"], r["dense"], pack.project_payload(p.payload or {}),
                (r["sparse_indices"], r["sparse_values"])
                if r["sparse_indices"] is not None else None))
    _check(len(rows) >= 300, f"only {len(rows)} fixture rows survive in {collection}")
    rows.sort(key=lambda r: pack.point_id_key(r["id"]))
    for s_ in manifest["shards"]:
        (fixture_dir / s_["path"]).unlink()
    shards, total, fps = dump.write_full_shards(iter(rows), fixture_dir, shard_rows=len(rows))
    fresh = dump.build_manifest(
        dump_id="rehearsal-fixture", collection=collection, dims=manifest["embedder"]["dims"],
        distance="cosine", shards=shards, total_points=total,
        any_row_has_sparse=any(r["sparse_indices"] is not None for r in rows),
        fingerprints=fps, profile=SP.PROD_E9)
    (fixture_dir / "manifest.json").write_text(json.dumps(fresh, indent=2))
    src, _ = _load_into_memory(fixture_dir, SOURCE)
    for q in meta["queries"]:
        q["expected_top1"] = _top1(src, SOURCE, q["vector"])
    meta["refreshed_at"] = dump.now_iso()
    meta["points"] = total
    meta["dropped_since_sampled"] = meta.get("dropped_since_sampled", 0) + len(missing)
    (fixture_dir / "fixture.json").write_text(json.dumps(meta))
    return {"points": total, "dropped_since_sampled": len(missing),
            "refreshed_at": meta["refreshed_at"]}


# ------------------------------------------------------------------ run
def _ephemeral_keys(work: pathlib.Path) -> tuple[str, str]:
    sec, pub = work / "rehearsal.sec", work / "rehearsal.pub"
    subprocess.run(["minisign", "-G", "-s", str(sec), "-p", str(pub), "-W", "-f"],
                   input="\n\n", text=True, capture_output=True, check=True)
    return str(sec), pub.read_text().splitlines()[-1].strip()


def _expect_refusal(label: str, fn) -> None:
    try:
        fn()
    except apply.ApplyRefusal:
        return
    raise RehearsalFailure(f"{label} was NOT refused")


#: A cosine collection renormalises every vector on upsert. qdrant-client's
#: local (:memory:) mode does that in numpy float32 and the result depends on
#: the upsert batch it arrived in and is not idempotent: identical input moved
#: by 1-2 ULP on 24 of 400 rows, and a unit vector re-stored moved on 166 of
#: 400 (task 687 refactor). A real Qdrant server (v1.18.3) is bit-stable: 0 of
#: 400 on the same rows, 0 of 60,000 twin rows across dump -> apply -> re-dump.
#: CI runs local mode, so this check states the bound instead of hiding it.
RENORM_MAX_ULP = 2


def _ulp_gap(a: list[float], b: list[float]) -> int:
    from array import array

    ia, ib = array("i", array("f", a).tobytes()), array("i", array("f", b).tobytes())
    return max((abs(x - y) for x, y in zip(ia, ib)), default=0)


def redump_check(src, src_name: str, target, target_name: str,
                 work: pathlib.Path) -> dict:
    """A copy applied from a release, scrolled again, against the source.

    Exact: ids, payload text and the sparse leg of every row. Dense: within
    `RENORM_MAX_ULP` float32 ULP per element (renormalisation, see above);
    the number of rows that moved at all is reported. And the copy, used as
    the base of its own delta, is a 0-row 0-tombstone delta against itself:
    the fingerprint rule does not invent changes in an unchanged store.

    `dense_rows_ulp_drift` counts rows whose dense float32 differs by at least 1 ULP between
    source and copy (not rows that "moved"). It depends on the upsert batch in local mode only
    (renormalisation on upsert, see `RENORM_MAX_ULP`)."""
    s = {r["id"]: r for r in dump.scroll_rows(src, src_name)}
    t = {r["id"]: r for r in dump.scroll_rows(target, target_name)}
    _check(s.keys() == t.keys(), "re-dump has different ids than the release")
    moved, worst = 0, 0
    for id_, r in s.items():
        o = t[id_]
        _check(r["payload"] == o["payload"], f"payload of {id_} changed in the copy")
        _check((r["sparse_indices"], r["sparse_values"]) ==
               (o["sparse_indices"], o["sparse_values"]), f"sparse leg of {id_} changed")
        gap = _ulp_gap(r["dense"], o["dense"])
        worst = max(worst, gap)
        moved += gap > 0
    _check(worst <= RENORM_MAX_ULP,
           f"dense vector moved {worst} ULP in the copy (bound {RENORM_MAX_ULP})")
    shards, total, fps = dump.write_full_shards(
        dump.scroll_rows(target, target_name), work, shard_rows=200)
    base = dump.build_manifest(
        dump_id="redump", collection=target_name, dims=len(next(iter(t.values()))["dense"]),
        distance="cosine", shards=shards, total_points=total,
        any_row_has_sparse=any(r["sparse_indices"] is not None for r in t.values()),
        fingerprints=fps, profile=None)  # throwaway base for DeltaBuilder, never applied
    builder = dump.DeltaBuilder(base, work)
    changed = list(builder.changed_rows(dump.scroll_rows(target, target_name)))
    gone = builder.tombstones({})
    _check(not changed and not gone,
           f"an unchanged store re-dumped is not a 0-row delta: {len(changed)} "
           f"changed, {len(gone)} gone")
    return {"rows": len(s), "dense_rows_ulp_drift": moved, "max_ulp": worst}


#: The upgrade stage's deterministic subsets (task 2308). Of the fixture's id-sorted rows,
#: UPGRADE_N evenly spaced rows are held back from N-1 (so the delta ADDS them), the row after
#: each is stored in N-1 with an older snippet (so the delta REPLACES it), and UPGRADE_N
#: synthetic rows exist only in N-1 (so the delta carries one tombstone each).
UPGRADE_N = 20
UPGRADE_STALE_PREFIX = "OLD VERSION: "
#: One reason per stale row, cycled, so the pack carries every real reason code.
UPGRADE_REASONS = ("superseded", "aged_out", "takedown", "superseded_generation")


def _write_signed(manifest: dict, out: pathlib.Path, sec: str) -> pathlib.Path:
    manifest["signature"] = {"algo": "minisign", "sig_file": "manifest.json.minisig"}
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    sign_manifest(str(path), sec)
    return path


def _with_payload(r: dict, payload: dict, new_id: Optional[str] = None) -> dict:
    sparse = ((r["sparse_indices"], r["sparse_values"])
              if r["sparse_indices"] is not None else None)
    return pack.row_to_record(new_id or r["id"], r["dense"], payload, sparse)


def rehearse_upgrade(src, fx_manifest: dict, meta: dict, work: pathlib.Path,
                     sec: str, pub: str, fresh_top1: list, *, client=None,
                     src_name: str = SOURCE, target: str = TARGET) -> dict:
    """Install N-1, apply delta pack N, and the copy is N (release-doctrine #4: the upgrade
    from the previous release, not only the fresh install). `src` holds N (the fixture).
    N-1 is derived from it deterministically: UPGRADE_N rows missing (delta adds), UPGRADE_N
    rows with an older snippet (delta replaces), UPGRADE_N extra rows (delta tombstones).

    Defaults reproduce the CI rehearsal: `client=None` builds the local-mode N-1 copy, and N is
    the collection `src_name` on `src`. The server-side rehearsal (task 2404) passes a scratch
    server `client` plus a NEW, empty `target` collection name (created here, never reused) and
    names its scratch slice as `src_name`; `src` is only ever read. The server stage passes no
    queries, so `queries_equal_fresh_install` is 0 there and is not exercised on a server."""
    import gzip

    n_rows = sorted(dump.scroll_rows(src, src_name), key=lambda r: pack.point_id_key(r["id"]))
    _check(len(n_rows) >= 4 * UPGRADE_N, "the fixture is too small for the upgrade stage")
    step = len(n_rows) // (2 * UPGRADE_N)
    add_ids = {n_rows[i * step]["id"] for i in range(UPGRADE_N)}
    rep_ids = {n_rows[i * step + 1]["id"] for i in range(UPGRADE_N)}
    _check(len(add_ids) == len(rep_ids) == UPGRADE_N and not add_ids & rep_ids,
           "upgrade subsets overlap")
    prev_rows, stale_ids = [], []
    for r in n_rows:
        if r["id"] in add_ids:
            continue
        if r["id"] in rep_ids:
            p = json.loads(r["payload"])
            p["snippet"] = UPGRADE_STALE_PREFIX + (p.get("snippet") or "")
            r = _with_payload(r, p)
        prev_rows.append(r)
    for i, r in enumerate(n_rows[:UPGRADE_N]):          # rows N does not have
        sid = mint_point_id(f"https://rehearsal.invalid/stale/{i}/{r['id']}", "")
        p = json.loads(r["payload"])
        p["url"] = p["url"] + f"#stale-{i}"
        stale_ids.append(sid)
        prev_rows.append(_with_payload(r, p, sid))
    prev_rows.sort(key=lambda r: pack.point_id_key(r["id"]))

    # 1. release N-1: a signed full dump
    prev_dir = work / "upgrade-prev"
    idf_acc = idf.IdfAccumulator()
    shards, total, fps = dump.write_full_shards(iter(prev_rows), prev_dir, shard_rows=200,
                                                idf=idf_acc)
    prev = dump.build_manifest(
        dump_id="rehearsal-prev", collection=src_name, dims=fx_manifest["embedder"]["dims"],
        distance="cosine", shards=shards, total_points=total,
        any_row_has_sparse=fx_manifest.get("sparse") is not None,
        idf_snapshot=idf_acc.write(prev_dir), fingerprints=fps, profile=SP.PROD_E9)
    prev_path = _write_signed(prev, prev_dir, sec)

    # 2. the stranger installs N-1
    apply.verify_dump_signature(prev, prev_dir, pubkey=pub, allow_unsigned=False)
    apply.verify_shard_hashes(prev, prev_dir)
    cp = client if client is not None else qdrant_client(":memory:")
    apply.create_collection(cp, target, prev, log=lambda m: None)
    apply.upsert_shards(cp, target, prev, prev_dir)
    _check(cp.count(target, exact=True).count == total == len(prev_rows),
           "N-1 install count differs from the N-1 dump")

    # 3. delta pack N from N-1, built as `cyt dump --delta-base-manifest` builds it
    dlt_dir = work / "upgrade-delta"
    builder = dump.DeltaBuilder(prev, prev_dir)
    d_shards, d_total = dump.write_shards(builder.changed_rows(dump.scroll_rows(src, src_name)),
                                          dlt_dir, shard_rows=200)
    reasons = {sid: UPGRADE_REASONS[i % len(UPGRADE_REASONS)] for i, sid in enumerate(stale_ids)}
    tombs = builder.tombstones(reasons)
    n_tomb, tomb_sha = pack.write_tombstones(tombs, str(dlt_dir / "tombstones.jsonl.gz"))
    _check(d_total == 2 * UPGRADE_N and n_tomb == UPGRADE_N,
           f"delta holds {d_total} rows / {n_tomb} tombstones, expected "
           f"{2 * UPGRADE_N} / {UPGRADE_N}")
    delta = dump.build_manifest(
        dump_id="rehearsal-delta", collection=src_name, dims=fx_manifest["embedder"]["dims"],
        distance="cosine", shards=d_shards, total_points=d_total,
        any_row_has_sparse=fx_manifest.get("sparse") is not None, profile=SP.PROD_E9)
    delta.update({"schema": pack.DELTA_PACK_SCHEMA, "pack_id": "rehearsal-delta", "sequence": 1,
                  "base_dump_id": prev["dump_id"],
                  "base_manifest_sha256": pack.sha256_file(str(prev_path)),
                  "tombstones": {"path": "tombstones.jsonl.gz", "count": n_tomb,
                                 "sha256": tomb_sha}})
    _write_signed(delta, dlt_dir, sec)

    # 4. a broken delta is refused before any write
    before = cp.count(target, exact=True).count
    bad = work / "upgrade-delta-tampered"
    shutil.copytree(dlt_dir, bad)
    broken = copy.deepcopy(delta)
    broken["sequence"] = 2
    (bad / "manifest.json").write_text(json.dumps(broken, indent=2, ensure_ascii=False))
    _expect_refusal("delta with a broken manifest signature", lambda: apply.verify_dump_signature(
        broken, bad, pubkey=pub, allow_unsigned=False))
    _check(cp.count(target, exact=True).count == before, "a refused delta wrote to the copy")
    badt = work / "upgrade-delta-badreason"
    shutil.copytree(dlt_dir, badt)
    lines = [t.to_json() for t in tombs]
    lines[-1]["reason"] = "because"
    with gzip.open(badt / "tombstones.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(x) + "\n" for x in lines))
    bt = copy.deepcopy(delta)
    bt["tombstones"]["sha256"] = pack.sha256_file(str(badt / "tombstones.jsonl.gz"))
    _expect_refusal("tombstone with an unknown reason",
                    lambda: apply.verify_tombstones(bt, badt))

    # 5. the stranger applies delta N, in the order `cyt apply` checks
    apply.verify_dump_signature(delta, dlt_dir, pubkey=pub, allow_unsigned=False)
    apply.verify_shard_hashes(delta, dlt_dir)
    _check(delta["base_manifest_sha256"] == pack.sha256_file(str(prev_path)),
           "delta base is not the installed N-1")
    _check(apply.verify_tombstones(delta, dlt_dir) == n_tomb, "tombstone count differs")
    n_up = apply.upsert_shards(cp, target, delta, dlt_dir)
    n_del = apply.apply_tombstones(cp, target, delta, dlt_dir)
    _check(n_up == d_total and n_del == n_tomb, f"applied {n_up} rows / {n_del} tombstones")

    # 6. the copy equals N
    count = cp.count(target, exact=True).count
    _check(count == len(n_rows), f"upgraded copy holds {count}, N holds {len(n_rows)}")
    held = {str(p.id): p.payload for p in _scroll_all(cp, target)}
    _check(not set(stale_ids) & held.keys(), "a tombstoned id survived the upgrade")
    want = {r["id"]: json.loads(r["payload"]) for r in n_rows}
    for i in rep_ids | add_ids:
        _check(held.get(i) == want[i], f"point {i} does not carry the N payload")
    _check(not any(UPGRADE_STALE_PREFIX in (p.get("snippet") or "") for p in held.values()),
           "an old snippet survived the upgrade")
    for i, q in enumerate(meta["queries"]):
        got = _top1(cp, target, q["vector"])
        _check(got == fresh_top1[i] == q["expected_top1"],
               f"query {i}: upgraded {got} / fresh install {fresh_top1[i]}")
    redump = redump_check(src, src_name, cp, target, work / "upgrade-redump")
    return {"n_minus_1_points": total, "delta_rows": d_total, "delta_adds": len(add_ids),
            "delta_replaces": len(rep_ids), "tombstones": n_tomb,
            "tombstone_reasons": sorted(set(reasons.values())), "points_after": count,
            "queries_equal_fresh_install": len(meta["queries"]), "redump": redump,
            "refusals": 2}


def rehearse(fixture_dir: pathlib.Path, work: pathlib.Path) -> dict:
    """Dump -> sign -> verify -> apply -> query. Raises RehearsalFailure."""
    _check(minisign_available(),
           "minisign is not on PATH: the rehearsal needs it (a release that "
           "was never signature-verified is not rehearsed)")
    meta = json.loads((fixture_dir / "fixture.json").read_text())
    _check(meta["schema"] == FIXTURE_SCHEMA, "unknown fixture schema")
    src, fx_manifest = _load_into_memory(fixture_dir, SOURCE)

    # 1. dump the source exactly as `cyt dump` does (shared functions)
    out = work / "release"
    idf_acc = idf.IdfAccumulator()
    shards, total, fps = dump.write_full_shards(
        dump.scroll_rows(src, SOURCE), out, shard_rows=200, idf=idf_acc)
    idf_block = idf_acc.write(out)
    tally = dump.CoverageTally()
    for row in dump.scroll_rows(src, SOURCE):
        tally.add(json.loads(row["payload"]))
    cov = tally.assert_ok(threshold=dump.DEFAULT_THRESHOLD)
    manifest = dump.build_manifest(
        dump_id="rehearsal-full", collection=SOURCE,
        dims=fx_manifest["embedder"]["dims"], distance="cosine", shards=shards,
        total_points=total, any_row_has_sparse=fx_manifest.get("sparse") is not None,
        idf_snapshot=idf_block, fingerprints=fps, profile=SP.PROD_E9)
    manifest["signature"] = {"algo": "minisign", "sig_file": "manifest.json.minisig"}
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # 2. sign + verify
    sec, pub = _ephemeral_keys(work)
    sign_manifest(str(manifest_path), sec)
    apply.verify_dump_signature(manifest, out, pubkey=pub, allow_unsigned=False)

    # 3. tampering is refused
    tampered = copy.deepcopy(manifest)
    tampered["counts"]["points"] += 1
    tdir = work / "tampered"
    shutil.copytree(out, tdir)
    (tdir / "manifest.json").write_text(json.dumps(tampered, indent=2, ensure_ascii=False))
    _expect_refusal("tampered manifest", lambda: apply.verify_dump_signature(
        tampered, tdir, pubkey=pub, allow_unsigned=False))
    (work / "other").mkdir()
    _, other_pub = _ephemeral_keys(work / "other")
    _expect_refusal("wrong public key", lambda: apply.verify_dump_signature(
        manifest, out, pubkey=other_pub, allow_unsigned=False))
    shard0 = out / manifest["shards"][0]["path"]
    shard0.write_bytes(shard0.read_bytes() + b"\0")
    _expect_refusal("tampered shard", lambda: apply.verify_shard_hashes(manifest, out))
    shard0.write_bytes(shard0.read_bytes()[:-1])  # restore for the apply below
    apply.verify_shard_hashes(manifest, out)

    # 3b. the pinned IDF snapshot: declared, verified, and a tampered or missing one refused
    _check(idf_block is not None, "the fixture has no sparse rows: no IDF snapshot to pin")
    apply.verify_idf_snapshot(manifest, out)
    snap = out / idf_block["path"]
    good = snap.read_bytes()
    snap.write_bytes(good + b"\0")
    _expect_refusal("tampered IDF snapshot", lambda: apply.verify_idf_snapshot(manifest, out))
    snap.unlink()
    _expect_refusal("missing IDF snapshot", lambda: apply.verify_idf_snapshot(manifest, out))
    snap.write_bytes(good)
    apply.verify_idf_snapshot(manifest, out)

    # 3c. the torrent: deterministic, exactly the manifest's files, and it describes the dir
    (out / "STRAY.txt").write_text("not in the manifest")
    t1, t_report = torrent.make_torrent(out)
    t2, _ = torrent.make_torrent(out)
    _check(t1 == t2, "two torrent runs on one directory differ")
    listed = sorted("/".join(f[b"path"][i].decode() for i in range(len(f[b"path"])))
                    for f in torrent.bdecode(t1)[b"info"][b"files"])
    named = sorted(p for p, _, _ in torrent.declared_files(manifest))
    _check(listed == named and "STRAY.txt" not in listed
           and "manifest.json.minisig" in listed and idf_block["path"] in listed,
           f"torrent files {listed} are not the manifest's {named}")
    (work / "release.torrent").write_bytes(t1)
    torrent.verify_torrent(work / "release.torrent", out)
    (out / "STRAY.txt").unlink()
    torrent.verify_torrent(work / "release.torrent", out)  # a stray file never mattered
    shard0.write_bytes(shard0.read_bytes() + b"\0")
    try:
        torrent.verify_torrent(work / "release.torrent", out)
        raise RehearsalFailure("a torrent of a tampered shard was NOT refused")
    except torrent.TorrentRefusal:
        pass
    shard0.write_bytes(shard0.read_bytes()[:-1])

    # 4. apply into an empty collection
    target = qdrant_client(":memory:")
    apply_receipt = apply.create_collection(target, TARGET, manifest, log=lambda m: None)
    _check(apply_receipt["verified"] is False,
           "an in-memory target cannot hold a layout: its receipt must say NOT verified")
    n = apply.upsert_shards(target, TARGET, manifest, out)
    count = target.count(TARGET, exact=True).count
    _check(n == count == total == meta["points"],
           f"counts differ: upserted {n}, in target {count}, dumped {total}, "
           f"fixture {meta['points']}")

    # 5. same top-1 as the source, and as the sealed ids
    for i, q in enumerate(meta["queries"]):
        s, t = _top1(src, SOURCE, q["vector"]), _top1(target, TARGET, q["vector"])
        _check(s == t == q["expected_top1"],
               f"query {i}: source {s} / applied {t} / sealed {q['expected_top1']}")

    # 5a. BM25 scores on the applied copy come from the pinned snapshot's integers:
    #     stored weight * ln((n + 1) / (df + 0.5)). `df` is the snapshot's own. `n` is the
    #     snapshot's `n` (rows with a sparse vector, which is what a Qdrant SERVER counts:
    #     `cytir/index/layout.py`); qdrant-client's local mode, which CI runs, counts every
    #     point instead, so the check feeds it that and states the quirk rather than hide it.
    df = dict(idf.iter_snapshot(out / idf_block["path"]))
    stored = {r["id"]: dict(zip(r["sparse_indices"], r["sparse_values"]))
              for r in dump.scroll_rows(src, SOURCE) if r["sparse_indices"] is not None}
    _check(idf_block["n"] == len(stored), f"snapshot n {idf_block['n']} != {len(stored)} sparse rows")
    from qdrant_client.http import models as qm
    probes = sorted(df, key=lambda t: (df[t], t))[::max(1, len(df) // 6)][:6]
    for term in probes:
        hit = target.query_points(TARGET, query=qm.SparseVector(indices=[term], values=[1.0]),
                                  using=dump.SPARSE_VECTOR_NAME, limit=1).points[0]
        want = stored[str(hit.id)][term] * idf.idf(count, df[term])
        _check(abs(hit.score - want) <= 1e-4 * max(1.0, abs(want)),
               f"BM25 score for term {term} is {hit.score}, the pinned snapshot gives {want}")

    # 5b. the applied copy re-dumped is the release it came from
    redump = redump_check(src, SOURCE, target, TARGET, work / "redump")

    # 6. no officer names, no unnamed keys in what a stranger holds
    allowed = set(pack.DEFAULT_PROJECTION_FIELDS) | set(pack.ENTITY_CARD_PROJECTION_FIELDS)
    src_payload = {str(p.id): p.payload for p in _scroll_all(src, SOURCE)}
    applied = {str(p.id): p.payload for p in _scroll_all(target, TARGET)}
    _check(set(applied) == set(src_payload), "applied ids differ from the source ids")
    consent = {"search_ok_false": 0, "train_ok_false": 0, "flags_null": 0, "train_ok_false_survived": 0}
    digests = 0
    for pid, payload in applied.items():
        _check(not (set(payload) - allowed), f"unnamed key(s) {set(payload) - allowed}")
        _check(OFFICER_MARKER not in json.dumps(payload, ensure_ascii=False).lower(),
               f"officer marker in point {pid}")
        _check(not (set(payload) & MASTER_ONLY), f"master-only key in point {pid}")
        # 7. consent flags: on EVERY row, equal to the source's, null when unknown (never true)
        _check(all(k in payload for k in pack.CONSENT_FIELDS), f"point {pid}: a consent field is missing")
        _check(all(payload[k] == src_payload[pid][k] for k in pack.CONSENT_FIELDS),
               f"point {pid}: consent fields changed through dump -> apply")
        consent["search_ok_false"] += payload["search_ok"] is False
        consent["flags_null"] += payload["search_ok"] is None and payload["train_ok"] is None
        if payload["train_ok"] is False:
            consent["train_ok_false"] += 1
            consent["train_ok_false_survived"] += src_payload[pid]["train_ok"] is False
        # 8. the digest: 32 lowercase hex or absent, nothing else
        d = payload.get(pack.DIGEST_FIELD)
        _check(d is None or (len(d) == pack.DIGEST_HEX and pack._HEX.fullmatch(d)),
               f"point {pid}: content_digest {d!r} is not {pack.DIGEST_HEX}-hex")
        digests += d is not None
    _check(consent["train_ok_false"] > 0,
           "the fixture holds no train_ok=false row: the rehearsal cannot prove it survives apply")
    _check(digests > 0, "no applied row carries a content_digest")
    # 9. the upgrade: install N-1, apply delta pack N, the copy equals N (task 2308)
    fresh_top1 = [_top1(target, TARGET, q["vector"]) for q in meta["queries"]]
    upgrade = rehearse_upgrade(src, fx_manifest, meta, work, sec, pub, fresh_top1)
    from cytir.distribute import rehearsal_log_delta    # task 2850: tests 1 and 2 of delta-change-log
    log_delta = rehearsal_log_delta.rehearse_log_delta(fixture_dir, work, sec, pub)
    return {"points": total, "queries": len(meta["queries"]), "upgrade": upgrade,
            "log_delta": log_delta,
            "license_coverage": cov["pct"], "signature": "verified",
            "tamper_refusals": 3, "redump": redump,
            "consent_flags": consent, "content_digest_rows": digests,
            "apply_layout": {k: apply_receipt[k] for k in ("layout", "verified", "reason")},
            "idf_snapshot": {"n": idf_block["n"], "vocabulary": idf_block["vocabulary"],
                             "scores_checked": len(probes), "tamper_refusals": 2},
            "torrent": {"infohash": t_report["infohash"], "files": t_report["files"],
                        "deterministic": True}}


def main(argv: list[str] | None = None, prog: str = "cyt rehearse") -> int:
    if argv and argv[0] == "server":    # task 2339: needs a real server, not the fixture
        from cytir.distribute import server_rehearsal
        return server_rehearsal.main(argv[1:])
    ap = argparse.ArgumentParser(prog=prog)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="rehearse the release on the sealed fixture")
    r.add_argument("--fixture", default=str(DEFAULT_FIXTURE_DIR))
    b = sub.add_parser("build-fixture",
                       help="draw the fixture from a twin collection (read-only)")
    b.add_argument("--collection", required=True)
    b.add_argument("--qdrant-url", default=dump.DEFAULT_TWIN_QDRANT_URL)
    b.add_argument("--i-know-this-is-not-the-twin", action="store_true")
    b.add_argument("--out", default=str(DEFAULT_FIXTURE_DIR))
    b.add_argument("--points", type=int, default=400)
    b.add_argument("--queries", type=int, default=40)
    f = sub.add_parser("refresh-fixture",
                       help="re-project the sealed fixture rows from the twin, by id (read-only)")
    f.add_argument("--collection", required=True)
    f.add_argument("--qdrant-url", default=dump.DEFAULT_TWIN_QDRANT_URL)
    f.add_argument("--i-know-this-is-not-the-twin", action="store_true")
    f.add_argument("--fixture", default=str(DEFAULT_FIXTURE_DIR))
    args = ap.parse_args(argv)
    if args.cmd == "refresh-fixture":
        dump.assert_qdrant_url_is_not_prod(
            args.qdrant_url, i_know_this_is_not_the_twin=args.i_know_this_is_not_the_twin)
        print(json.dumps(refresh_fixture(qdrant_client(args.qdrant_url, timeout=300),
                                         args.collection, pathlib.Path(args.fixture))))
        return 0
    if args.cmd == "run":
        with tempfile.TemporaryDirectory() as td:
            report = rehearse(pathlib.Path(args.fixture), pathlib.Path(td))
        print(json.dumps(report))
        return 0
    dump.assert_qdrant_url_is_not_prod(
        args.qdrant_url, i_know_this_is_not_the_twin=args.i_know_this_is_not_the_twin)
    meta = build_fixture(qdrant_client(args.qdrant_url, timeout=300), args.collection,
                         pathlib.Path(args.out), n_points=args.points,
                         n_queries=args.queries)
    print(json.dumps({k: v for k, v in meta.items() if k != "queries"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
