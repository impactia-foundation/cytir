"""Guards and plan for the assembly binder (task 2099, binder slice A2).

Moved verbatim from the binder: the ADR-101 guard precondition (`cmd_guards`),
the pack-source planner, the source-family contract check and `cmd_plan`, which
fixes every expected count before a row is read. These decide what the assembly
admits; the only changes are that the repo root and the sealed-populations file
are located with `cytir.core.paths` instead of the binder's own `REPO`, and that the
constants and `log` they share with the binder now live here (the binder imports
them).
"""
from __future__ import annotations

import gzip
import json
import pathlib
import re
import subprocess
import sys
import time

from cytir.core.paths import REPO_ROOT, data_path
from cytir.index.assemble_inputs import (
    load_pack_route_exclusions, load_pack_route_extra_sources,
    load_pack_stage_overrides, read_jsonl_gz_ids, sha256_file)

FULL_DIM = 1024

# The projection revision the assembly is pinned to. Pinned rather than
# "current" because a projection change between the bake and the flip would
# silently change what the collection contains -- which is exactly what a pin
# is for and exactly what happened anyway, one layer down, when the INPUT
# shape changed under an unchanged projection.
#
# ec97ef9c (task 468, "consent-durable"): emits `consent_state` /
# `consent_basis` / `consent_source` from `dump_basis_evidence(payload)`
# instead of widening the passthrough tuple, so the SAME closed-set predicate
# governs admission and emission and a dump-basis record cannot be admitted
# while its evidence is dropped from the payload.
#
# 4dec6d22 (task 515) BUMPS the pin deliberately, for two changes:
#
#   1. `project_record` gains an optional `metadata_registry`. With it None the
#      projection is byte-identical to ec97ef9c (pinned by
#      `test_task515_serving_metadata`); with a registry it fills the
#      serving-signal fields a sealed embedding-stage record never carried.
#      Task 514 loaded 30,897,557 rows under ec97ef9c and 11,273,954 of them
#      (36.5%) reached the collection with no `trust_tier` and no
#      `source_type`; the ADR-043 gate fell from h@1 276 to 106 on 629 queries
#      and the fusion re-tune could not move a weight, because there was no
#      signal to weight.
#   2. the no-text tail stamps the tok epoch the CARRIED posting declares
#      instead of a hardcoded 1, now that the binder actually carries one.
#
# The pin held the projection constant while the defect walked in through the
# INPUT: the same code, fed a sealed embedding-stage record instead of a
# payload. A pin on the projection is not a pin on what the projection is
# given, which is why the stage receipt now records the shape of both.
#
# 63bc86f4 (task 516) BUMPS the pin deliberately, for one change:
#
#   3. ADR-060's card contract gains a SINGLE-SURFACE variant. `is_card` is
#      unchanged, and the paired dense/sparse + `bundle_digest` requirement is
#      unchanged for every input that has them; a card that carries the ADR-060
#      recipe that MADE it (`card_recipe`, `card_recipe_source`,
#      `card_recipe_tier`) and one sealed text surface is now projected instead
#      of refused, and counted as `card_single_surface_no_bundle_digest`.
#      Task 515 shipped 336,254 register-derived rows with no `source_type` at
#      all against a collection that carries it on all but 10 points, while
#      production itself ships 361,468 of its 370,432 cards with no
#      `bundle_digest` -- so the refusal was not enforcing ADR-060, it was
#      removing a ranking signal. A passage that merely shares a URL with a
#      card carries no recipe and is still refused (task 516 tests).
PINNED_PROJECTION_COMMIT = "63bc86f4"

# WAITING is not FAIL, but it is not a pass either: the guard suite reports
# WAITING for lanes that have not sealed, and the binder may only run once every
# lane it is loading has sealed. This is the list the operator must shrink to
# empty; naming it here means "the known WAITING set" is a checkable artifact
# rather than a sentence in a task brief.
KNOWN_WAITING_LANES = {
    "B1-cc-materializer", "B2-track1a-residual", "C-direct-acquisition",
    "D-blocked-needs-decision", "F1-cc-main-2026-30-never-acquired",
    "F2-class1-never-materialized",
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)



# ===========================================================================
# stage 0 -- guards (the precondition)
# ===========================================================================
def cmd_guards(args):
    """Run the ratified guard suite and refuse on anything worse than WAITING.

    `assembly_guards.py` exits 0 all-green, 1 on any FAIL, 2 on WAITING with no
    FAIL. The binder's precondition is stricter than "exit 0 eventually": a FAIL
    is terminal and a WAITING lane must be one of the enumerated unsealed lanes,
    never a surprise.
    """
    out = args.report or (pathlib.Path(args.work) / "guard-report.json"
                          if args.work else None)
    cmd = [sys.executable, "-m", "cytir.ops.assembly_guards",
           "--plan", args.plan, "--root", str(REPO_ROOT), "--format", "json"]
    if out:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--out", str(out)]
    log("running the ADR-101 guard suite:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    sys.stdout.write(r.stdout[-4000:])
    if r.stderr.strip():
        sys.stderr.write(r.stderr[-2000:])

    rep = json.loads(pathlib.Path(out).read_text()) if out and \
        pathlib.Path(out).exists() else json.loads(r.stdout)
    fails = [g["id"] for g in rep["guards"] if g["status"] == "FAIL"]
    waits = [g["id"] for g in rep["guards"] if g["status"] == "WAITING"]
    plan = json.loads(pathlib.Path(args.plan).read_text())
    unsealed = {l["lane_id"] for l in plan.get("pending_lanes", [])}
    unknown = unsealed - KNOWN_WAITING_LANES

    log(f"guards: FAIL={fails or '-'}  WAITING={waits or '-'}  "
        f"unsealed lanes={sorted(unsealed) or '-'}")
    if unknown:
        raise SystemExit(f"REFUSING: unsealed lane(s) not in the known WAITING "
                         f"set: {sorted(unknown)}. A new pending lane is a "
                         f"planning change, not something the binder may absorb.")
    if fails:
        raise SystemExit(
            f"REFUSING: guard(s) {fails} FAIL. The flip is prohibited while any "
            f"guard fails (ADR-101 Ruling 2) and there is no waiver flag. "
            f"Remedies are in the report's per-guard `remedy` field.")
    if waits:
        msg = (f"guards WAITING: {waits}; unsealed lanes {sorted(unsealed)}. "
               f"The binder may PLAN and STAGE sealed sets, but LOAD is refused "
               f"until every contributing lane has sealed.")
        if args.require_green:
            raise SystemExit("REFUSING (--require-green): " + msg)
        log("NOTE: " + msg)
        return 2
    log("ALL GUARDS GREEN — the flip gate is open.")
    return 0


# ===========================================================================
# stage 1 -- plan
# ===========================================================================
def _plan_pack_sources(pop, root, by_set):
    """Plan the 417-family release, whose payload lives one hop away.

    This looked like 125 bespoke joins and is not. Every source under
    `vectors/release=.../source=<name>/` carries the SAME shard triplet as
    everything else; what differs is only where the payload file sits and what it
    is called. The rollup dir holds receipts, `RECEIPT.portable.v5.json` and
    `rollup-evidence/`, but no payload -- and these ids do NOT join to the twin
    (verified 0/5 on fedlex: they are new epoch-4 identities, not the ones they
    replace). The payload is in the source's own STAGE dir, named
    `shard-NNNN.texts.jsonl.gz`, and it is a full record, not just text:
    `point_id, url, lang, text, sparse_text, source_profile, source_metadata,
    parent_identity, jurisdiction, ...`.

    The hop is recorded in `rollup-evidence/stage/STAGE-RECEIPT.json` ->
    `artifacts[].uri`, so the stage dir is DISCOVERED per source rather than
    guessed from a naming convention.

    Verified on fedlex, 2026-08-30: the stage file is positionally aligned with
    the vectors -- same order, same count (22,928 == 22,928), first five
    point_ids identical -- so the shard triplet's binding contract holds across
    the hop and no id-keyed join is needed at all.
    """
    sid = pop["set_id"]
    rel = pathlib.Path(root) / pop["vectors"].removeprefix(
        "artifact://").split("/source=")[0]
    srcs = sorted(d for d in rel.glob("source=*") if d.is_dir())
    if not srcs:
        raise SystemExit(f"{sid}: no source=* dirs under {rel}")
    # A glob cannot express "and also this one". `service_public_dila_exact` is
    # a flip-lane contributing set whose vectors were written under a DIFFERENT
    # release, so the glob never saw it, it staged nothing, and -- because this
    # is a fresh collection with `carried_legacy: 0` -- its 35,351 claimed
    # production points would not have been deleted, just never written (task
    # 513 §5.3; 500/500 of a sample are live in production today). Named
    # additions, never patterns, and each one checked against the row count two
    # independent artifacts already agree on.
    for extra in load_pack_route_extra_sources():
        d = (pathlib.Path(root) /
             extra["vectors_uri"].removeprefix("artifact://"))
        if not d.is_dir():
            raise SystemExit(
                f"{sid}: declared extra pack source {extra['source']!r} is not a "
                f"directory at {d} (data/assembly/pack-route-extra-sources.v1.json)")
        if d in srcs:
            raise SystemExit(
                f"{sid}: extra pack source {extra['source']!r} is already in the "
                "globbed release; remove the declaration rather than adding it twice")
        # A SUPERSESSION, not an addition (task 516). The re-materialized fedlex
        # family keeps its source name and its set, and lives under a new
        # release; the exclusion list is keyed by source NAME, so it cannot tell
        # the two apart and would drop both. The declaration names the release
        # it replaces, and the replacement only counts if the superseded
        # directory was actually in the glob -- a supersession that silently
        # became a no-op would ship the defect it was written to remove.
        superseded_release = extra.get("supersedes_release")
        if superseded_release:
            victim = next(
                (x for x in srcs
                 if x.name == f"source={extra['source']}"
                 and f"release={superseded_release}" in x.as_posix()), None)
            if victim is None:
                raise SystemExit(
                    f"{sid}: {extra['source']!r} declares it supersedes release "
                    f"{superseded_release!r}, but no such source directory is in "
                    f"the globbed release under {rel}")
            srcs.remove(victim)
            log(f"{sid}: - superseded {victim.as_posix().split('/vectors/')[-1]}")
        srcs.append(d)
        log(f"{sid}: + declared extra source {extra['source']} "
            f"(release {extra['release']}, expect {extra['expected_rows']:,} rows) "
            f"for set {extra['set_id']}")
    overrides = load_pack_stage_overrides()
    exclusions = load_pack_route_exclusions()
    out, tot, skipped = [], 0, []
    excluded_sources = []
    for sd in srcs:
        src_name = sd.name.removeprefix("source=")
        # A source whose own claim fragment declares a non-flip lane is not
        # part of this build. Skipping it here is the narrowing task 478 §2.5
        # assumed `claimed(S)` would perform and that the pack route never got:
        # without it, 6,521,047 expansion-lane rows plan, stage and load.
        if src_name in exclusions:
            excluded_sources.append(src_name)
            continue
        # The rollup-evidence directory is NOT one fixed name. Four sources
        # publish theirs as `rollup-evidence-v5/` or
        # `rollup-evidence-portable-v5/`, and hardcoding `rollup-evidence/`
        # reported them as having no receipt at all -- which is how task 478
        # sec.2.5 came to list nine "missing" receipts when only five are
        # genuinely absent (and those five carry a sealed EXCLUDED.json or
        # INVALIDATED.json, i.e. they are deliberately not in the build).
        # Discover it; refuse if more than one candidate disagrees.
        # The canonical path always wins where it exists, so this discovery
        # cannot change the resolution of any source that already resolved.
        # Variants are consulted ONLY to fill a genuine absence.
        # A sealed exclusion marker is checked FIRST, not as the explanation for
        # a missing receipt. Task 514 measured the difference: five directories
        # under this release carry an `EXCLUDED.json` (`admission: false`,
        # "byte-identical duplicate embedding; authoritative rollup already
        # terminal"), and one of them --
        # `source=technical-manufacturer-catalog-admitted-products-v2` -- also
        # has a perfectly good STAGE-RECEIPT, so it was found first and its 445
        # rows planned, staged and would have loaded. The marker is the
        # producer's own ruling about the directory; nothing downstream may
        # outvote it by being easier to find.
        sealed = next((m.name for m in
                       (sd / "EXCLUDED.json", sd / "INVALIDATED.json")
                       if m.exists()), None)
        if sealed:
            skipped.append((sd.name, f"sealed {sealed}, not in the build"))
            continue
        canonical = sd / "rollup-evidence" / "stage" / "STAGE-RECEIPT.json"
        cands = ([canonical] if canonical.exists() else
                 sorted(sd.glob("rollup-evidence*/stage/STAGE-RECEIPT.json")))
        if not cands:
            skipped.append((sd.name, "no STAGE-RECEIPT.json"))
            continue
        if len(cands) > 1:
            digests = {sha256_file(c) for c in cands}
            if len(digests) > 1:
                raise SystemExit(
                    f"{sd.name}: {len(cands)} stage receipts that are not "
                    "byte-identical; which one describes the vectors is not "
                    "decidable from the tree")
        rc = cands[0]
        rcd = json.loads(rc.read_text())
        arts = rcd.get("artifacts") or []
        prod = (rcd.get("producer") or {}).get("version", "?")
        texts, bare, used_override = {}, False, False
        for a in arts:
            uri = a["uri"].removeprefix("artifact://")
            # PRODUCER-VERSION DRIFT, not missing data. `adapt_pack_to_413a_stage.py`
            # v3-recursive-global-artifact-root writes a full artifact-root-relative
            # path ("_work/task417/r0-fedlex-portable-stage-v1/shard-0000.texts.jsonl.gz");
            # v1 writes a BARE filename ("shard-0000.texts.jsonl.gz") that is
            # relative to a stage dir the receipt never names. Resolving a bare
            # URI against the artifact root silently points at a file that does
            # not exist, so it is refused by name instead of guessed at.
            if "/" not in uri:
                bare = True
                # ...unless the stage dir it means has been declared and
                # verified in `pack-stage-overrides.v1.json`.  The override is
                # applied ONLY here, i.e. only to a receipt that is actually
                # bare, and every one of its own preconditions is re-checked
                # below.  A declared hop is checkable; a guessed one is not.
                over = overrides.get(src_name)
                if over:
                    p = (pathlib.Path(root)
                         / over["stage_dir"].removeprefix("artifact://") / uri)
                    m = re.search(r"(shard-\d+)\.texts\.jsonl\.gz$", p.name)
                    if m and p.exists():
                        texts[m.group(1)] = p
                        used_override = True
                continue
            p = pathlib.Path(root) / uri
            m = re.search(r"(shard-\d+)\.texts\.jsonl\.gz$", p.name)
            if m:
                texts[m.group(1)] = p
        if used_override:
            over = overrides[src_name]
            named = sum(1 for a in arts
                        if str(a["uri"]).endswith(".texts.jsonl.gz"))
            if len(texts) != named or len(texts) != int(over["shards"]):
                raise SystemExit(
                    f"{sd.name}: override resolves {len(texts)} texts shard(s), "
                    f"the receipt names {named} and the override declares "
                    f"{over['shards']}. An override that does not cover the "
                    "whole receipt would silently drop rows.")
            declared = int(rcd.get("records") or rcd.get("rows") or -1)
            if declared != int(over["receipt_records"]):
                raise SystemExit(
                    f"{sd.name}: the receipt declares {declared:,} records, the "
                    f"override was verified against "
                    f"{int(over['receipt_records']):,}. The receipt has moved "
                    "since the override was checked; re-verify it.")
        if not texts:
            skipped.append((sd.name, f"STAGE-RECEIPT (producer {prod}) names "
                            + ("BARE artifact URIs with no directory -- the stage "
                               "dir it means is not recorded anywhere in the receipt"
                               if bare else "no texts shard")))
            continue
        for v in sorted(sd.glob("shard-*.vectors.f16")):
            stem = v.name[:-len(".vectors.f16")]
            tp = texts.get(stem)
            if tp is None or not tp.exists():
                skipped.append((f"{sd.name}/{stem}", "payload shard missing"))
                continue
            man = json.loads((sd / f"{stem}.vectors.manifest.json").read_text())
            n = int(man["n"])
            if v.stat().st_size != n * FULL_DIM * 2:
                raise SystemExit(f"{v}: size != n*dim*2 -- truncated shard")
            out.append({
                "set_id": sid, "shard": f"{sd.name}.{stem}",
                "payload_route": "pack", "source": sd.name,
                "vectors": str(v), "row_ids": str(sd / f"{stem}.row_ids.jsonl"),
                "records": str(tp), "rows": n, "mode": pop["mode"],
            })
            tot += n
    # A declared extra source is only worth declaring if it brings the rows it
    # says it brings. Two independent artifacts already agree on the number
    # (the route ledger's replacement_chunks and the expansion ledger's chunks);
    # this is the third, from the shards themselves, and it refuses on a
    # mismatch rather than quietly staging a different population.
    for extra in load_pack_route_extra_sources():
        got = sum(p["rows"] for p in out
                  if p.get("source") == f"source={extra['source']}")
        if got != int(extra["expected_rows"]):
            raise SystemExit(
                f"{sid}: declared extra source {extra['source']!r} resolves to "
                f"{got:,} rows, but pack-route-extra-sources.v1.json declares "
                f"{int(extra['expected_rows']):,}. One of the two is wrong; "
                "refusing to guess.")
        log(f"{sid}: extra source {extra['source']} resolved at {got:,} rows "
            "== declared")

    by_set[sid] = {"declared_rows": None, "shard_rows": tot,
                   "shards": len(out), "mode": pop["mode"],
                   "sources": len(srcs), "unresolved": len(skipped),
                   "excluded_non_flip_lane": len(excluded_sources)}
    log(f"{sid}: {len(srcs)} sources -> {len(out)} partitions, {tot:,} rows"
        + (f"  [!] {len(skipped)} unresolved" if skipped else "")
        + (f"  [-] {len(excluded_sources)} excluded (non-flip lane)"
           if excluded_sources else ""))
    for name in excluded_sources:
        cite = (exclusions[name].get("named_by") or [{}])[0]
        where = cite.get("path") or cite.get("pack") or "?"
        log(f"    EXCLUDED {name}: {exclusions[name]['lane']} lane, "
            f"{exclusions[name].get('evidence_tier')} ({where})")
    for name, why in skipped[:10]:
        log(f"    UNRESOLVED {name}: {why}")
    if len(skipped) > 10:
        log(f"    ... and {len(skipped) - 10} more")
    return out



def assert_family_contract(set_ids, args, stage: str):
    """Refuse a set with no source-family declaration, before a row is read.

    Task 529. The class this catches is the one task 526/527 found by
    accident: the rebuild changed WHAT TEXT two families embed -- fedlex lost
    the law name/abbreviation/SR from its dense surface, entity cards lost
    their sparse surface entirely -- and every count check, every guard and
    the full-stack gate passed, because nothing in the pipeline had ever been
    told what those families were supposed to embed.

    A cheap check at the FRONT of an expensive process (verification doctrine,
    project interpretation): a `plan` that dies in a second beats a defect
    found after a day of staging.
    """
    from cytir.sources import contract as SFC

    doc = SFC.load(getattr(args, "family_contracts", None))
    SFC.assert_declared(set_ids, doc,
                        strict=bool(getattr(args, "strict_families", False)),
                        stage=stage)
    log(f"source-family contract: {len(set(set_ids))} set(s) declared"
        + (" (--strict-families)" if getattr(args, "strict_families", False)
           else ""))



def cmd_plan(args):
    """Resolve the assembly plan + registry into a per-partition work manifest.

    Every expected count is written down HERE, before a single row is read, so
    that `stage` and `load` are checking against a number that was fixed in
    advance rather than against whatever they happen to produce.
    """
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    plan = json.loads(pathlib.Path(args.plan).read_text())
    reg = json.loads(data_path(
        "assembly", "sealed-populations.v1.json").read_text())
    # Task 529: every set this plan intends to load must have a declared
    # source-family contract. Checked here, on the plan's own roster, so an
    # undeclared family cannot reach `stage` at all.
    assert_family_contract(
        sorted(set(args.sets.split(",")) if args.sets
               else (plan.get("expected_sets") or [])),
        args, "plan")
    root = args.artifact_root or reg["artifact_root"]["heimdall_path"]

    # -- the suppression universe -------------------------------------------
    # Loser suppression (ADR-101 Ruling 1) + G4 tombstone suppression
    # (ADR-101a Correction 2: a URL bearing an ADR-091 natural-person tombstone
    # accepts NO replacement rows -- privacy beats coverage).
    #
    # This loop used to select purely on the `.gz` suffix, which made "is this
    # a suppression list?" a question about a filename. Task 511 s3 found the
    # first half of that trap (no `.gz` record existed, so every plan run
    # reported "0 suppressed" while reporting the per-set counts as consistent);
    # task 513 found the second half, which is worse: registering ANY new
    # `.gz` id-list under `records` -- a collision-winner map, an audit dump --
    # would silently delete every id it names from the build. The role is now
    # declared, and an undeclared `.gz` record is refused rather than consumed.
    SUPPRESSION_ROLE = "row_suppression"
    suppressed: set[str] = set()
    for key, rec in (plan.get("records") or {}).items():
        uri = rec.get("uri")
        if not uri:
            continue
        p = REPO_ROOT / uri.removeprefix("artifact://")
        # The role is DECLARED. An earlier cut of this fell back to
        # `key.endswith("_suppression")`, which swept in `g4_tombstone_suppression`
        # and `g4_tombstone_url_suppression` -- two .json records the original
        # suffix-based code never consumed -- and `plan` died on
        # `BadGzipFile: b'{\n'`. Naming a record is not the same as declaring
        # what to do with it.
        role = rec.get("role")
        if role == SUPPRESSION_ROLE:
            ids = read_jsonl_gz_ids(p)
            suppressed |= ids
            log(f"suppression {key}: +{len(ids):,}")
        elif role:
            # a declared role this stage does not act on -- e.g.
            # `collision_winners`, which `load` consumes and `plan` must not
            log(f"record {key}: role {role!r}, not a stage-time suppression")
        elif p.suffix == ".gz":
            raise SystemExit(
                f"plan record {key!r} ({uri}) is a .gz id list with no declared "
                f"`role`. Refusing to guess: a .gz record used to be consumed as "
                f"a suppression list on its suffix alone, so an unrelated id dump "
                f"registered here would silently drop those ids from the build. "
                f"Declare \"role\": \"{SUPPRESSION_ROLE}\" if that is what it is, "
                f"or any other role string if it is not.")

    collapse_uri = plan["apply"]["identity_collapse"]["uri"]
    collapse_n = plan["apply"]["identity_collapse"]["count"]

    partitions = []
    by_set = {}
    wanted = set(args.sets.split(",")) if args.sets else None
    for pop in reg["populations"]:
        sid = pop["set_id"]
        if wanted and sid not in wanted:
            continue
        route = pop.get("payload_route", "stage" if pop.get("self_contained")
                        else None)
        if route == "pack":
            partitions.extend(_plan_pack_sources(pop, root, by_set))
            continue
        if route not in ("stage", "twin"):
            log(f"SKIP {sid}: no payload route "
                f"({pop.get('payload_source', 'none recorded')}).")
            continue
        vdir = pathlib.Path(root) / pop["vectors"].removeprefix("artifact://")
        sdir = pathlib.Path(root) / pop["stage"].removeprefix("artifact://")
        shards = sorted(vdir.glob("shard-*.vectors.f16"))
        if not shards:
            raise SystemExit(
                f"{sid}: no shards under {vdir}. See the per-population "
                f"`caution` in sealed-populations.v1.json (427 hides its shards "
                f"one level deeper than its manifest declares).")
        tot = 0
        for v in shards:
            stem = v.name[:-len(".vectors.f16")]
            man = json.loads((vdir / f"{stem}.vectors.manifest.json").read_text())
            n = int(man["n"])
            if v.stat().st_size != n * FULL_DIM * 2:
                raise SystemExit(f"{v}: size != n*dim*2 -- truncated shard")
            partitions.append({
                "set_id": sid, "shard": stem, "payload_route": route,
                "vectors": str(v), "row_ids": str(vdir / f"{stem}.row_ids.jsonl"),
                "records": (str(sdir / f"{stem}.records.jsonl.gz")
                            if route == "stage" else None),
                "rows": n, "mode": pop["mode"],
            })
            tot += n
        by_set[sid] = {"declared_rows": pop["rows"], "shard_rows": tot,
                       "shards": len(shards), "mode": pop["mode"]}
        if pop.get("rows") and tot != pop["rows"]:
            raise SystemExit(
                f"{sid}: shards sum to {tot:,} but the registry declares "
                f"{pop['rows']:,}. One of the two is wrong; refusing to guess.")
        log(f"{sid}: {len(shards)} shards, {tot:,} rows")

    man = {
        "schema": "aw.binder-plan.v1", "task": 478,
        "build_id": plan["build_id"],
        "assembly_plan": args.plan,
        "assembly_plan_sha256": sha256_file(args.plan),
        "target": plan["target"],
        "apply": plan["apply"],
        "pinned_projection_commit": PINNED_PROJECTION_COMMIT,
        "artifact_root": root,
        "partitions": partitions,
        "by_set": by_set,
        "expected_staged_total": sum(p["rows"] for p in partitions),
        "suppressed_ids": len(suppressed),
        "identity_collapse": {"uri": collapse_uri, "count": collapse_n},
        "disposition": plan["disposition"],
        "membership": plan["membership"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (work / "binder-plan.json").write_text(json.dumps(man, indent=1))
    if suppressed:
        with gzip.open(work / "suppressed-ids.jsonl.gz", "wt") as fh:
            for i in sorted(suppressed):
                fh.write(i + "\n")
    log(f"PLAN {len(partitions)} partitions, "
        f"{man['expected_staged_total']:,} rows before suppression, "
        f"{len(suppressed):,} suppressed ids")

    # The conservation identity from the plan, checked now rather than at the end.
    d = plan["disposition"]
    tot = sum(d.values())
    memb = plan["membership"]["points"]
    if tot != memb:
        raise SystemExit(f"dispositions sum to {tot:,} != membership {memb:,}")
    log(f"conservation precheck OK: dispositions sum to membership {memb:,}")
    log(f"  build population = re_embedded_in_place {d['re_embedded_in_place']:,}"
        f" + replaced {d['replaced']:,} = "
        f"{d['re_embedded_in_place'] + d['replaced']:,}")
    log(f"  excluded: tombstoned {d['tombstoned']:,}, pending {d['pending']:,}, "
        f"carried_legacy {d['carried_legacy']:,}")

