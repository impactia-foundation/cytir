"""Verify step of the assembly binder (task 2118, binder slice A5).

Moved verbatim from the binder: `cmd_verify`, which re-checks a loaded collection
against the plan, the stage receipts and the declared payload-shape contract. Every
check, refusal, receipt key and exit code is unchanged; the only edit is that the
Qdrant client is built with `cytir.core.qdrant.qdrant_connect` (same 900 s timeout).
"""
from __future__ import annotations

import json
import pathlib
import time

from cytir.core.qdrant import qdrant_connect
from cytir.index.assemble_inputs import load_pack_route_exclusions, read_jsonl_gz_ids
from cytir.index.assemble_plan import log


def cmd_verify(args):
    from cytir.core import serving_config as sc


    work = pathlib.Path(args.work)
    plan = json.loads((work / "binder-plan.json").read_text())
    stage = json.loads((work / "stage-summary.json").read_text())
    load = json.loads((work / "load-summary.json").read_text())
    cfg = sc.get(args.config or load["config"])
    coll = args.collection or load["collection"]
    client = qdrant_connect(args.qdrant, timeout=900)

    info = client.get_collection(coll)
    pts = info.points_count or 0
    problems = []

    # (0) THE PAYLOAD SHAPE (task 516). Checked before the arithmetic, because a
    #     collection can satisfy every conservation identity and still be
    #     unrankable: task 514's `aw_e4` held exactly the number of points its
    #     plan declared, with `trust_tier` and `source_type` blank on 36.5% of
    #     them and no BM25 posting on 63.5%, and `verify` passed. Re-evaluated
    #     here from the stage receipts against the DECLARED contract, so a tree
    #     staged before the contract existed cannot be loaded and shipped
    #     unchecked, and so the same numbers are asserted twice from one file.
    from cytir.index import payload_shape_contract as PSC
    shape_contract = PSC.load(getattr(args, "payload_shape_contract", None))
    excluded_sources = set(load_pack_route_exclusions())
    shape_receipts = []
    for rp in sorted((work / "staged").glob("*.receipt.json")):
        r = json.loads(rp.read_text())
        if r["shard"].split(".shard-")[0].removeprefix("source=") in excluded_sources:
            continue
        shape_receipts.append(r)
    # A tree staged before the contract existed carries no title-shape counter.
    # Measuring it here from the staged payloads -- loudly, and recorded as
    # measured at verify -- is the difference between "unchecked" and "checked
    # late"; reporting it as unchecked is how 22,928 unfindable statutes loaded.
    PSC.backfill_title_shape(shape_receipts, work / "staged", shape_contract,
                             log=log)
    PSC.backfill_provenance_head(shape_receipts, work / "staged", shape_contract,
                                 log=log)
    PSC.backfill_extraction_collapse(shape_receipts, work / "staged",
                                     shape_contract, log=log)
    PSC.backfill_chunk_width(shape_receipts, work / "staged",
                             shape_contract, log=log)
    shape_report = PSC.evaluate(shape_receipts, shape_contract)
    PSC.format_report(shape_report, log=log)
    (work / "payload-shape-report.verify.json").write_text(
        json.dumps(shape_report, indent=1))
    problems.extend(f"payload shape: {f}" for f in shape_report["findings"])

    # (1) per-set: staged == input - suppressed - excluded  (already enforced
    #     per partition; re-checked here at set level so the summary is a proof,
    #     not a restatement)
    for sid, v in stage.items():
        if v["staged"] + v["suppressed"] + v["excluded"] != v["input"]:
            problems.append(f"{sid}: per-set conservation broken")

    # (2) staged total == upserted + everything the LOAD is entitled to drop.
    #
    #     `staged == upserted` was true when `load` only upserted. It has not
    #     been true since three drop mechanisms landed -- pack-route exclusions
    #     (task 513), cross-set collision losers (task 513), and ADR-101 row
    #     suppression (task 514) -- and this check went on comparing the two
    #     numbers, so the first honest run of it after a correct load reported
    #     `staged 38,246,800 != upserted 30,897,557` as a PROBLEM. A check that
    #     is wrong about a correct build teaches an operator to ignore it.
    #
    #     The identity is now the whole arithmetic, and every term comes from
    #     the load's own receipts rather than from a number typed here.
    staged_total = sum(v["staged"] for v in stage.values())
    receipts = load.get("build_totals_from_receipts") or {}
    excluded = int(load.get("pack_route_excluded_rows") or 0)
    suppressed = int(receipts.get("rows_suppressed",
                                  load.get("rows_suppressed_skipped") or 0))
    losers = int(receipts.get("collision_losers_skipped",
                              load.get("collision_losers_skipped") or 0))
    accounted = load["rows_upserted"] + excluded + suppressed + losers
    if staged_total != accounted:
        problems.append(
            f"staged {staged_total:,} != upserted {load['rows_upserted']:,} + "
            f"excluded {excluded:,} + suppressed {suppressed:,} + collision "
            f"losers {losers:,} = {accounted:,}")

    # (3) distinct ids == upserted - enumerated collisions. Not `staged`: an
    #     excluded, suppressed or lost row is never upserted, so it can never
    #     be one of the distinct ids either.
    exp_distinct = load["rows_upserted"] - load["cross_set_collisions"]
    distinct = load.get("distinct_ids")
    if distinct is None:
        # A resumed load did not observe the ids, and `null` says so. The
        # collection is then the measurement: `points_count` equal to
        # `upserted - collisions` is the same identity, evaluated on the
        # artifact instead of on a counter. What is NOT allowed is treating an
        # unmeasured value as zero and calling the check green.
        log("NOTE: distinct_ids not observed by the load (resumed run); "
            "checks (3) and (4) are evaluated against the collection's own "
            "points_count, which is the stronger of the two anyway")
        if pts != exp_distinct:
            problems.append(f"collection points_count {pts:,} != upserted "
                            f"{load['rows_upserted']:,} - collisions "
                            f"{load['cross_set_collisions']:,} = "
                            f"{exp_distinct:,}")
    else:
        if exp_distinct != distinct:
            problems.append(f"distinct ids {distinct:,} != upserted "
                            f"{load['rows_upserted']:,} - collisions "
                            f"{load['cross_set_collisions']:,}")

        # (4) the collection holds exactly the distinct ids
        if pts != distinct:
            problems.append(f"collection points_count {pts:,} != distinct ids "
                            f"{distinct:,}")

    # (5) the shape is what was asked for, and it is fully indexed
    try:
        sc.assert_collection_matches(client, coll, cfg, check_indexed=True)
    except RuntimeError as e:
        problems.append(str(e))

    # (6) RULE 3: the not-carried sets must be ABSENT
    absent_checked = 0
    for name, path in (("tombstones", args.tombstone_ids),
                       ("delete-set", args.delete_ids)):
        if not path:
            continue
        ids = sorted(read_jsonl_gz_ids(pathlib.Path(path)))
        sample = ids[:args.absence_sample]
        got = client.retrieve(coll, ids=sample, with_payload=False)
        if got:
            problems.append(f"{name}: {len(got)} of {len(sample)} sampled ids "
                            f"ARE PRESENT in {coll} and must not be")
        absent_checked += len(sample)

    out = {
        "schema": "aw.binder-verify.v1", "task": 478,
        "collection": coll, "config": cfg.name,
        "points_count": pts, "indexed_vectors_count": info.indexed_vectors_count,
        "status": str(info.status),
        "staged_total": staged_total,
        "upserted": load["rows_upserted"],
        "distinct_ids": load["distinct_ids"],
        "cross_set_collisions": load["cross_set_collisions"],
        "absence_ids_checked": absent_checked,
        "payload_shape": {"verdict": shape_report["verdict"],
                          "contract": shape_report["contract"],
                          "families": shape_report["families"],
                          "published_families": shape_report.get(
                              "published_families", [])},
        "per_set": stage,
        "verdict": "PASS" if not problems else "FAIL",
        "problems": problems,
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (work / "verify-report.json").write_text(json.dumps(out, indent=1))
    for p in problems:
        log("PROBLEM:", p)
    log(f"VERIFY {out['verdict']}: {pts:,} points in {coll}")
    return 0 if not problems else 1
