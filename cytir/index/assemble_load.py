"""Load step of the assembly binder (task 2112, binder slice A4).

Moved verbatim from the binder: `cmd_load`, which loads the staged rows into a
Qdrant collection. Every write call, batch, refusal and receipt is unchanged;
the only edits are that the Qdrant client is built with
`cytir.core.qdrant.qdrant_connect` (same timeout) and that a comment no longer
names a task-script path.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import time

import numpy as np

from cytir.core.qdrant import qdrant_connect
from cytir.index.assemble_inputs import (
    load_collision_winners, load_pack_route_exclusions, load_row_suppression,
    read_jsonl_gz_ids, sha256_file)
from cytir.index.assemble_plan import FULL_DIM, log
from cytir.index.assemble_stage import assert_persisted_inputs


def cmd_load(args):
    from qdrant_client import models as qm
    from cytir.core import serving_config as sc


    work = pathlib.Path(args.work)
    plan = json.loads((work / "binder-plan.json").read_text())
    cfg = sc.get(args.config)
    coll = args.collection or plan["target"]["collection"]
    client = qdrant_connect(args.qdrant, timeout=900)

    if plan["target"]["dim"] != FULL_DIM:
        raise SystemExit(f"plan target dim {plan['target']['dim']} != {FULL_DIM}")
    log(f"target collection {coll} at serving config {cfg.name} "
        f"(dense {cfg.dense_dim}, {cfg.quantizer}, {cfg.code_bytes_per_vector()} B/vec)")

    if not client.collection_exists(coll):
        sc.create_collection(client, coll, cfg, bulk=True, log=log)
    else:
        log(f"{coll} exists — resuming (upsert is idempotent by id)")
        sc.assert_collection_matches(client, coll, cfg, check_indexed=False)

    outdir = work / "staged"
    receipts = sorted(outdir.glob("*.receipt.json"))
    if not receipts:
        raise SystemExit("nothing staged")

    seen: dict[str, str] = {}
    collapse_expected = read_jsonl_gz_ids(
        pathlib.Path(args.collapse_ids)) if args.collapse_ids else None
    # Cross-set collision adjudication (task 513). An id staged by two sets is
    # upserted twice; whichever row lands last wins, and "last" is the order
    # `sorted(outdir.glob("*.receipt.json"))` happens to produce. That is only
    # harmless while the colliding rows are byte-identical, and task 513
    # measured that 9,347 of 11,478 are not (the I1 identity agrees -- same
    # canonical_url, same normalized content digest -- but the lanes' encoder
    # inputs differ by whitespace/rendering, so the vectors differ). This map
    # names the winner per id from ratified precedence, so the survivor is a
    # decision on the record rather than a filename accident.
    collision_winners = load_collision_winners(
        pathlib.Path(args.collision_winners)) if args.collision_winners else {}
    if collision_winners:
        log(f"collision winners: {len(collision_winners):,} adjudicated ids")
    collision_losers_skipped = 0
    losers_by_set: dict[str, int] = {}
    winners_seen: set[str] = set()
    collisions = []
    loaded = 0

    # A build staged before the pack-route exclusions existed still holds their
    # partitions on disk. Skip them here too, so this build does not have to be
    # re-staged to stop shipping 6.5M expansion-lane rows.
    pack_exclusions = load_pack_route_exclusions(args.pack_route_exclusions)
    excluded_rows = 0
    excluded_partitions = 0

    # Rows a ratified precedence forfeits, scoped to the set that forfeits them.
    # ADR-101 Ruling 1 has never been applied at row level for either direction
    # (task 511 §3 wired the ADR-101b/c pairs only, and excluded task 427 from
    # even those), so without this the build ships 816,871 vectors the record
    # says are never loaded -- the gap task 513b named.
    row_suppression = load_row_suppression(args.row_suppression)
    if row_suppression:
        log(f"row suppression: {sum(len(v) for v in row_suppression.values()):,} "
            f"row(s) across {len(row_suppression)} set(s)")
    suppressed_rows_skipped = 0
    suppressed_by_set: dict[str, int] = {}
    resumed_partitions = 0
    pending_since_check = 0

    for rp in receipts:
        r = json.loads(rp.read_text())
        sid, stem = r["set_id"], r["shard"]
        src = stem.split(".")[0].removeprefix("source=") if stem.startswith("source=") else None
        if src and src in pack_exclusions:
            excluded_rows += int(r["staged"])
            excluded_partitions += 1
            continue
        lp = outdir / f"{sid}.{stem}.loaded.json"
        if lp.exists() and not args.force:
            loaded += json.loads(lp.read_text())["loaded"]
            resumed_partitions += 1
            log(f"skip {sid}/{stem} (already loaded)")
            continue
        scope = src if src else sid
        suppress_here = row_suppression.get(scope, frozenset())
        vp = outdir / f"{sid}.{stem}.vectors.f16"
        pp = outdir / f"{sid}.{stem}.payload.jsonl.gz"
        if sha256_file(vp) != r["vectors_sha256"]:
            raise SystemExit(f"{vp}: digest drift since staging")
        n = r["staged"]
        vecs = np.fromfile(vp, dtype="<f2").reshape(n, FULL_DIM)
        with gzip.open(pp, "rt") as fh:
            rows = [json.loads(l) for l in fh]
        assert len(rows) == n

        # ADR-105 (task 581), per partition and BEFORE this partition's first
        # upsert. Checked here rather than once up front so a refusal always
        # lands on a partition boundary -- the same unit `.loaded.json` makes
        # resumable -- and so a work tree staged before this guard existed
        # cannot be loaded unchecked. The rows are already parsed; the check
        # costs no extra I/O.
        assert_persisted_inputs(
            ({"url": (r.get("payload") or {}).get("url"),
              "sha": (r.get("payload") or {}).get("content_digest")}
             for r in rows), args, "load", label=f"{sid}/{stem}")

        t0 = time.time()
        B = args.batch
        upserted_here = 0
        sup_here = 0
        col_here = 0
        for lo in range(0, n, B):
            hi = min(lo + B, n)
            dense = sc.mrl_trunc(np.asarray(vecs[lo:hi], dtype=np.float32),
                                 cfg.dense_dim)
            dense = np.asarray(dense, dtype=np.float32)
            pts = []
            for j in range(hi - lo):
                row = rows[lo + j]
                pid, payload = row["id"], row["payload"]
                if pid in suppress_here:
                    # a ratified precedence gave this row's old point to another
                    # set; the row is sealed evidence, never a loaded vector
                    suppressed_rows_skipped += 1
                    sup_here += 1
                    suppressed_by_set[scope] = suppressed_by_set.get(scope, 0) + 1
                    continue
                win = collision_winners.get(pid)
                if win is not None and win != sid:
                    # this set lost the id; its row is never upserted, so the
                    # winner's row cannot be overwritten by load order
                    collision_losers_skipped += 1
                    col_here += 1
                    losers_by_set[sid] = losers_by_set.get(sid, 0) + 1
                    continue
                if win is not None:
                    winners_seen.add(pid)
                prev = seen.get(pid)
                if prev is not None and prev != sid:
                    collisions.append((pid, prev, sid))
                seen[pid] = sid
                vec = {sc.DENSE: dense[j].tolist()}
                # ADR-099 Ruling 1 / ADR-028 / ADR-037: the BM25 leg is stored
                # SATURATED. This build is what finally makes the whole index
                # saturated in one pass -- the ~4-5% of prod points still holding
                # raw term frequencies (ADR-037's prod re-saturation, HELD since
                # 2026-07-14 pending a fusion re-tune that never deployed) do not
                # survive a fresh build, because nothing is copied.
                #
                # The weights come from `stage`, where `project_record` produced
                # them via PROJ.saturated_bm25. The loader does NOT re-derive
                # them: a second tokenizer here would be a forked mirror contract
                # against the query side, and it would silently drop
                # compose_index_text's title/site_name prefix and the entity-card
                # segment boundaries.
                sp = rows[lo + j].get("bm25")
                if sp and sp.get("indices"):
                    vec[sc.SPARSE_BM25] = qm.SparseVector(
                        indices=sp["indices"], values=sp["values"])
                pts.append(qm.PointStruct(id=pid, vector=vec, payload=payload))
            # RULE 1: upsert in place. Never delete-then-insert.
            if pts:
                client.upsert(coll, points=pts, wait=False)
                pending_since_check += len(pts)
            upserted_here += len(pts)
            # BACK-PRESSURE. `wait=False` is what makes a bulk load fast, and it
            # is also what let this build write 30,897,557 rows in 5.5 hours
            # while the collection applied 3.1M of them -- the writer outran the
            # applier by 10x, 62 GiB of operations sat unapplied in the WAL, and
            # nothing said so until someone counted the points. A fire-and-forget
            # writer that never looks back is not fast, it is unmeasured. Every
            # `lag_every` upserted rows, ask the collection how many points it
            # actually holds and refuse to keep piling on while it falls behind.
            if args.lag_every and pending_since_check >= args.lag_every:
                pending_since_check = 0
                waited = 0.0
                while True:
                    have = client.count(coll, exact=False).count
                    lag = (loaded + upserted_here) - have
                    if lag <= args.max_lag or waited >= args.lag_timeout:
                        break
                    if waited == 0.0:
                        log(f"  back-pressure: {lag:,} row(s) written but not yet "
                            f"applied (limit {args.max_lag:,}); waiting")
                    time.sleep(5)
                    waited += 5
                if lag > args.max_lag:
                    raise SystemExit(
                        f"REFUSING: the collection is {lag:,} rows behind the "
                        f"loader after {waited:.0f}s and is not catching up. A "
                        "store that cannot apply what it is sent will not become "
                        "correct by being sent more -- check the storage "
                        "filesystem (task 514: ntfs-3g/FUSE capped the apply path "
                        "at ~8k points/min) before resuming.")
            if (lo // B) % 100 == 0:
                el = max(time.time() - t0, 1e-9)
                log(f"  {sid}/{stem} {hi:,}/{n:,}  {hi/el:,.0f} pt/s")
        if upserted_here + sup_here + col_here != n:
            raise SystemExit(
                f"{sid}/{stem}: {n:,} staged but {upserted_here:,} upserted + "
                f"{sup_here:,} suppressed + {col_here:,} collision losers do not "
                "account for it")
        lp.write_text(json.dumps({"loaded": upserted_here, "staged": n,
                                  "collision_losers_skipped": col_here,
                                  "rows_suppressed": sup_here,
                                  "scope": scope,
                                  "set_id": sid, "shard": stem}))
        loaded += upserted_here
        log(f"loaded {sid}/{stem}: {upserted_here:,} of {n:,} staged"
            + (f" ({col_here:,} lost the id to another set)" if col_here else "")
            + (f" ({sup_here:,} forfeited by ratified precedence)" if sup_here else "")
            + f" (running total {loaded:,})")

    # -- cross-set collision adjudication, if a winner map was supplied ------
    if collision_winners and resumed_partitions:
        # `winners_seen` is built while upserting, so a resumed partition
        # contributes nothing to it and every id it holds looks missing. On the
        # first resume of this build that turned a correct 4-id finding into a
        # spurious 11,474-id refusal. The check is a property of a FULL pass;
        # say so rather than evaluating it against a partial one.
        log(f"winner-map coverage NOT EVALUATED: {resumed_partitions} "
            "partition(s) were already loaded and were skipped, so the ids they "
            "hold were never re-observed. Re-read the log of the pass that "
            "loaded them.")
    elif collision_winners:
        missing = set(collision_winners) - winners_seen
        # An adjudicated id can fail to load from its declared winner for two
        # very different reasons, and conflating them either hides a real
        # map/build disagreement or blocks a build for a rule working as ruled.
        #
        #   (a) the winner never staged it -- the map and the build disagree
        #       about who staged what. That is the defect this check exists for.
        #   (b) the winner staged it and a DIFFERENT ratified precedence
        #       forfeited that row (ADR-101 Ruling 1 / 101b / 101c). The map is
        #       right about who staged it; the row is simply not loaded by
        #       anyone, because the winner is the only set whose copy was
        #       eligible and its copy is forfeit.
        #
        # (b) is reported by id, never absorbed: the row vanishes from the
        # collection, which is exactly the shape task 511 §3 vetoed 91 ids to
        # avoid, and whether ADR-101b Ruling 3a's LABEL tie-break should fall
        # through to the next co-owner is a ruling nobody has made.
        forfeited_winner = sorted(
            pid for pid in missing
            if any(pid in ids for ids in row_suppression.values()))
        disagreement = sorted(set(missing) - set(forfeited_winner))
        if forfeited_winner:
            log(f"NOTE: {len(forfeited_winner)} adjudicated collision id(s) load "
                "from nobody -- the declared winner staged the row and a "
                "ratified precedence forfeited it: "
                + ", ".join(forfeited_winner))
        if disagreement:
            raise SystemExit(
                f"REFUSING: {len(disagreement):,} adjudicated collision id(s) "
                "never loaded from their declared winner set, and no ratified "
                "row suppression names them -- the winner map and the staged "
                f"build disagree about who staged what. "
                f"Examples: {disagreement[:5]}")
        log(f"collision adjudication: {collision_losers_skipped:,} losing row(s) "
            f"skipped across {len(losers_by_set)} set(s): "
            + ", ".join(f"{k} {v:,}" for k, v in sorted(losers_by_set.items())))
        if collisions:
            raise SystemExit(
                f"REFUSING: {len({c[0] for c in collisions}):,} id(s) were still "
                "upserted by two sets despite the winner map -- the map does not "
                "cover every collision in the staged build.")

    # -- intra-build collision adjudication ---------------------------------
    if collisions:
        cids = {c[0] for c in collisions}
        log(f"{len(cids):,} cross-set id collisions during load")
        if collapse_expected is not None:
            rogue = cids - collapse_expected
            if rogue:
                raise SystemExit(
                    f"REFUSING: {len(rogue):,} id collisions are NOT in the "
                    f"digest-pinned G7 identity-collapse set. An unenumerated "
                    f"collision means two sets disagree about what a point is. "
                    f"Examples: {sorted(rogue)[:5]}")
            log("all collisions are in the enumerated G7 collapse set — OK")
        else:
            log("WARNING: no --collapse-ids given, so collisions could not be "
                "checked against the enumerated set")

    # A resumed run upserts nothing, so its in-memory counters are all zero --
    # and a summary of zeros is a lie about the collection, not a description of
    # this run. The per-partition receipts are the durable record; aggregate
    # them so `load-summary.json` describes the BUILD however many passes it
    # took. (The first pass of this build wrote 30,897,557 rows and then
    # refused on a post-load check; the resume that completed it would otherwise
    # have overwritten that with `rows_upserted: 30,897,557, distinct_ids: 0`.)
    from_receipts = {"loaded": 0, "rows_suppressed": 0,
                     "collision_losers_skipped": 0, "staged": 0}
    per_scope: dict[str, int] = {}
    for lp in sorted(outdir.glob("*.loaded.json")):
        rec = json.loads(lp.read_text())
        for key in from_receipts:
            from_receipts[key] += int(rec.get(key) or 0)
        if rec.get("rows_suppressed"):
            scope = rec.get("scope") or rec["set_id"]
            per_scope[scope] = per_scope.get(scope, 0) + int(rec["rows_suppressed"])

    (work / "load-summary.json").write_text(json.dumps({
        "collection": coll, "config": cfg.name,
        # `rows_upserted` comes from the receipts, which cover the whole build.
        # `distinct_ids` cannot: `seen` only holds the ids THIS process
        # observed, so on a resume it is 0 -- and 0 is not "no distinct ids",
        # it is "not measured here". Task 514's class fix stopped the resume
        # writing a summary of zeros over a finished build; this is the same
        # bug one field further along. The measurement lives in the collection
        # itself (points_count) and in
        # task 514's load preflight script, which derives it from the
        # staged tree without a Qdrant at all.
        # `loaded` already credits a resumed partition from its own receipt
        # (line ~1522), so it covers the whole build either way. `seen` does
        # NOT: it holds only the ids THIS process observed, so on a resume it
        # is 0 -- and 0 is not "no distinct ids", it is "not measured here".
        # Task 514's class fix stopped a resumed run writing a summary of zeros
        # over a finished build; this is the same bug one field further along,
        # and it made `verify` compare a real points_count against a counter
        # that never ran.
        "rows_upserted": loaded,
        "distinct_ids": (len(seen) if not resumed_partitions else None),
        "distinct_ids_note": (None if not resumed_partitions else
                              f"{resumed_partitions} partition(s) resumed; "
                              "distinct ids not observed by this process -- "
                              "read the collection's points_count, or "
                              "load_preflight_514.py over the staged tree"),
        "build_totals_from_receipts": from_receipts,
        "build_rows_suppressed_by_scope": dict(sorted(per_scope.items())),
        "partitions_resumed_this_run": resumed_partitions,
        "cross_set_collisions": len({c[0] for c in collisions}),
        "collision_winners_file": args.collision_winners,
        "collision_winners_adjudicated": len(collision_winners),
        "collision_losers_skipped": collision_losers_skipped,
        "collision_losers_by_set": dict(sorted(losers_by_set.items())),
        "pack_route_exclusions": len(pack_exclusions),
        "pack_route_excluded_partitions": excluded_partitions,
        "pack_route_excluded_rows": excluded_rows,
        "row_suppression_file": args.row_suppression,
        "rows_suppressed_skipped": suppressed_rows_skipped,
        "rows_suppressed_by_scope": dict(sorted(suppressed_by_set.items())),
        "loaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=1))
    # Every declared suppression must have found its row. A suppression that
    # matched nothing is not "nothing to do" -- it is a scope key that does not
    # address the partition it was meant for, and it would ship the rows.
    if row_suppression:
        # Evaluated over the WHOLE build from the per-partition receipts, not
        # over this process's own run. A resumed run skips nothing, and the
        # previous wording ("NOT EVALUATED") turned a resume into a hole in the
        # instrument -- the receipts are a full-pass record and answer the
        # question directly.
        #
        # THE IDENTITY IS NOT declared == skipped-at-load. A row can also have
        # been removed one stage earlier: the plan's stage-time suppression set
        # takes it out of the staged tree, so `load` never sees it and cannot
        # skip it. Task 515 measured exactly this -- task427 declares 65,685
        # ADR-101 forfeits, 189 of which the stage had already removed, so the
        # load skipped 65,496 and the check refused a CORRECT build. The
        # credit is not taken on trust: an unmet declared id must be PRESENT in
        # this work dir's own `suppressed-ids.jsonl.gz`, or it is still a
        # scope key that addresses no partition and would have shipped the row.
        stage_supp_path = work / "suppressed-ids.jsonl.gz"
        stage_supp = (read_jsonl_gz_ids(stage_supp_path)
                      if stage_supp_path.exists() else set())
        missing, credited = {}, {}
        for scope, ids in row_suppression.items():
            declared = len(ids)
            skipped = per_scope.get(scope, suppressed_by_set.get(scope, 0))
            if skipped == declared:
                continue
            pre = len(set(ids) & stage_supp)
            if skipped + pre == declared:
                credited[scope] = pre
            else:
                missing[scope] = {"declared": declared, "skipped_at_load": skipped,
                                  "suppressed_at_stage": pre,
                                  "unaccounted": declared - skipped - pre}
        for scope, pre in sorted(credited.items()):
            log(f"row suppression: {scope} declares {len(row_suppression[scope]):,}; "
                f"{pre:,} of them were already suppressed at STAGE and never "
                f"reached the load -- accounted, not missing")
        if missing:
            raise SystemExit(
                f"REFUSING: {len(missing)} suppression scope(s) did not suppress "
                f"what they declare, and the difference is NOT accounted for by "
                f"stage-time suppression: {dict(list(missing.items())[:6])}")
        log(f"row suppression: {sum(per_scope.values()) or suppressed_rows_skipped:,} "
            f"row(s) skipped across {len(per_scope) or len(suppressed_by_set)} "
            f"scope(s), each exactly as declared once stage-time suppression is "
            f"credited")
    log(f"LOAD done: {loaded:,} rows upserted, {len(seen):,} distinct ids"
        + (f", {collision_losers_skipped:,} collision losers skipped"
           if collision_losers_skipped else "")
        + (f", {suppressed_rows_skipped:,} rows forfeited by ratified precedence"
           if suppressed_rows_skipped else ""))

    if args.finalize:
        sc.finalize_collection(client, coll, log=log)
        log("HNSW m=16 build started; use `verify` once it is green.")
