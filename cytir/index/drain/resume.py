"""Drain resume write path: `resume_prod_only` (the prod leg of a sealed pack whose twin leg already landed;
also the reconstruction resume, `twin_ids_override`), `resume_content_pack_prod_only` (the same for a content
pack: upsert and tombstone-delete) and `settle_supersede_deleting` (a url-group supersede state stuck at
`deleting`, decided by prod lookup).

Task 2929 (RULING 80, drain split step 13): moved verbatim out of `cytir/index/promote_queue.py`, which
re-exports the three names. A moved body that calls a name a test patches on the facade (or a name that
stays there) reads it at call time through `_Q` (the facade module).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib

from cytir.index import change_log as CL
from cytir.index import layout as LAYOUT
from cytir.index import prod_guard as PG
from cytir.index import promote_sealed_pack as PSP
from cytir.index import sparse_rule as SR
from cytir.index import url_group_supersede as UGS
from cytir.index.drain.common import QueueError, WriteAttempt
from cytir.index.drain.locks import PROD_LOCK_PATH
from cytir.index.drain.receipts import (
    load_content_pack_ids_from_state, prod_call_receipt, prod_chunk_checkpoint, read_inplace_ids,
    twin_written_ids_path)


def settle_supersede_deleting(manifest: dict, manifest_sha: str | None, artifact_root: pathlib.Path, *,
                              twin_url: str, prod_url: str, prod_collection: str, prod_alias: str,
                              batch: int, task: str, apply: bool,
                              write_attempt: "WriteAttempt | None" = None) -> dict | None:
    """Task 2364 (F1 + F2 of the #1753 review): decide a url-group supersede state stuck at
    `deleting` BY PROD LOOKUP, for the two "already held" exits and the prod-only resume (one
    helper, three callers). `None` = nothing to settle (not a supersede pack, no state, or a stage
    other than `deleting`). Otherwise it reads, BY ID (a retrieve, never a scroll), which of the
    state's `prod.to_delete` ids (all `ids` when prod's part never started) prod still holds:
      none  -> `marked-applied`: the delete had finished; `mark_applied` so the next drain is clean;
      some  -> `deleted-present`: the existing `UGS.delete_superseded("prod", ...)` runs for exactly
               those ids (twin export re-verified first, prod export verified, each batch verified
               gone), under the writer locks, then `mark_applied`.
    `apply=False` reads only and returns the decision as `would-mark-applied` / `would-delete`: no
    write, no lock. An unreadable state, or the state of another manifest, raises
    `CL.ChangeLogError` (the caller refuses; no DONE). Returns the receipt for the pack record."""
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    if not manifest.get("supersede"):
        return None
    pack_id = manifest["pack_id"]
    try:
        st = UGS.read_state(artifact_root, pack_id, manifest_sha)
    except QueueError as exc:
        raise CL.ChangeLogError(str(exc)) from exc
    if st is None or st.get("stage") != "deleting":
        return None
    # Review 1768 finding 2: `deleting` with no prod part also means the TWIN delete died halfway. The
    # twin is checked first; prod is never deleted while the twin still holds a superseded row.
    on_twin = UGS.present_on(twin_url, manifest["collection"], list(st["ids"]))
    if on_twin:
        raise QueueError(
            f"{pack_id}: the url-group supersede is at `deleting` and the twin still holds "
            f"{len(on_twin)} superseded id(s) (e.g. {on_twin[:3]}): its twin delete is unfinished. "
            "The plan path must finish the twin first; nothing was changed. "
            f"state {UGS.state_path(artifact_root, pack_id)}")
    # the alias the "none present" decision rests on must be the live one (a permanent `applied`)
    _Q._revalidate_prod_collection(pack_id, prod_url, prod_alias, prod_collection)
    prod_part = st.get("prod") or None
    owed = list(prod_part["to_delete"]) if prod_part else list(st["ids"])
    present = UGS.present_on(prod_url, prod_collection, owed)
    receipt = {"state_before": "deleting", "checked": len(owed), "present_on_prod": len(present),
               "present_on_twin": 0, "prod_collection": prod_collection,
               "state_path": str(UGS.state_path(artifact_root, pack_id))}
    if not apply:
        receipt["decision"] = "would-delete" if present else "would-mark-applied"
        return receipt
    plan = {"pack_id": pack_id, "manifest_sha": manifest_sha, "ids": st["ids"], "urls": st["urls"],
            "export": st["export"], "prod": prod_part, "summary": st.get("summary") or {},
            "skipped_pack_ids": st.get("skipped_pack_ids", []), "artifact_root": str(artifact_root)}
    if present:
        runb = _Q._load_run_batch()
        lock = _Q.TFA.ProductionWriterLock(host=_Q.prod_lock_host(), lock_path=PROD_LOCK_PATH,
                                        purpose=f"task{task}-{pack_id}-supersede-settle")
        runb.assert_locks_free(task)
        taken: list[str] = []
        try:
            for name in runb.LOCKS:
                runb.acquire(name, task, f"{pack_id}-supersede-settle", pid=os.getpid())
                taken.append(name)
            lock.acquire()
            try:
                _Q._revalidate_prod_collection(pack_id, prod_url, prod_alias, prod_collection)
                if write_attempt is not None:
                    write_attempt.started = True       # from here a failure is ABORTED, not REFUSED
                deleted = UGS.delete_superseded("prod", prod_url, prod_collection, plan,
                                                twin_url=twin_url, twin_collection=manifest["collection"],
                                                batch=batch)
            finally:
                lock.release()
        finally:
            for name in reversed(taken):
                runb.release(name, task)
        if prod_part:      # prepare_prod re-measured `to_delete` as what prod held now; keep the full set
            plan["prod"]["to_delete"] = sorted(set(owed) | set(plan["prod"]["to_delete"]))
        # `deleted` is what prepare_prod re-measured under the lock; it can exceed `present_on_prod`
        receipt.update(decision="deleted-present", deleted=deleted)
    else:
        receipt["decision"] = "marked-applied"
    UGS.mark_applied(artifact_root, plan)
    receipt["state_after"] = "applied"
    return receipt

def resume_prod_only(*, pack_id: str, manifest: dict, twin_url: str,
                     prod_url: str, task: str, batch: int, pause_s: float,
                     artifact_root: pathlib.Path,
                     prod_collection: str,
                     write_attempt: WriteAttempt | None = None,
                     twin_ids_override: list[str] | None = None,
                     prod_alias: str = "aw_prod") -> dict:
    """Resume ONLY the prod leg of a pack whose twin leg already landed and
    verified, but whose prod leg never completed.

    Task 664, 2026-09-26 incident: prod's qdrant restarted mid-write during
    `cc-authority-frontier-w7-tranche2-d`'s PROD leg -- twin had already
    written and verified 185,184 ids. Every subsequent drain re-ran
    `dry_plan()` against TWIN (as it always does), which now raised
    `PSP.AlreadyLandedError` -- a "plan" stage refusal, cached forever, so
    the gap could never close on its own. `drain()` catches that
    specifically and, if PROD is genuinely still missing the ids twin
    reports landed, calls this instead of treating it as a refusal.

    `twin_ids_override` (task 664, 2026-09-26, incident 2 --
    `cc-authority-frontier-w7-tranche2`): the caller's OWN live twin-vs-prod
    membership diff, used in place of reading `{pack_id}-twin-written-ids.
    json` when that per-pack artifact does not exist -- because twin has
    these ids from a DIFFERENT pack_id's earlier apply (e.g. an earlier
    split-pack landing this pack duplicates), not from a write under THIS
    pack_id. `drain()` computes it, not this function: deciding "is this
    genuinely a resume, or does some of this pack's content exist on
    neither twin nor prod" is a judgment about the WHOLE pack that belongs
    where `prod_missing` is already known, not re-derived here.

    Uses `twin_first_apply.resume_after_verified_twin_gate` -- the existing
    primitive built for exactly this shape (already used by
    `task 351 apply_content_pack_shared.py` and
    `task 385 apply_vector8_content_pack.py`) -- rather than a
    second recovery protocol. Twin is never re-written: `preflight`
    re-confirms, UNDER THE LOCK, that twin still holds every one of this
    pack's ids (the resume premise); only `apply_prod`/`verify_mirrors`
    touch the network.

    `prod_collection` (task 673 review round 3, finding 6/FLIP BLOCKER;
    made REQUIRED, no fallback, round 4 finding 3): the collection for the
    PROD leg, resolved by the caller through the live `aw_prod` alias --
    never `manifest["collection"]` (twin's fixed name, unaffected by a
    flip). No fallback: a caller that forgot to resolve the alias must
    fail loudly.

    `prod_alias` (round 4 finding 2): re-resolved and compared against
    `prod_collection` immediately before the PROD write, while the
    production lock is held (see `_revalidate_prod_collection`).

    PARTIAL PROD WRITE (task 788, F1 of task 786). A prod leg that died
    after some batches landed left prod holding a part of the pack, and
    `plan()`'s first-time rule ("any id present refuses") turned every later
    resume into `REFUSED live-apply-preflight`, every hour, until a manual
    reconcile. The twin-written ids now ride into `plan()`
    (`resume_twin_ids`): prod's present ids must be twin-written ids
    (skipped, never re-sent), exactly the rest are written, and an id prod
    holds that the twin leg did not write refuses as a collision
    (`PSP.ResumeCollisionError`). A prod that already holds every id writes
    nothing and closes. The prod receipt records `resumed_partial`, the
    counts (`present_before`, `written`, `present_after`,
    `dropped_by_twin_decision`) and the sha256 of the twin-written ids file.

    THE WRITE SET IS THE TWIN'S, NEVER THE PACK'S (task 788, tranche 14,
    2026-10-09): this resume used to re-project the whole pack with no drop
    set and wrote 1,409,761 rows where the twin leg had written 1,409,548 --
    213 gate-rejected rows reached prod only. Now ids outside `twin_ids` are
    never written, and `verify_mirrors` fails the leg (no DONE) unless the
    prod written-ids artifact holds exactly the twin's id set.
    """
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    runb = _Q._load_run_batch()
    artifact_root.mkdir(parents=True, exist_ok=True)
    ledger = artifact_root / f"{pack_id}-resume-ledger.jsonl"
    lock = _Q.TFA.ProductionWriterLock(host=_Q.prod_lock_host(), lock_path=PROD_LOCK_PATH,
                                    purpose=f"task{task}-{pack_id}-resume")
    runb.assert_locks_free(task)
    twin_collection = manifest["collection"]
    resolved_prod_collection = prod_collection
    if twin_ids_override is not None:
        twin_ids = list(twin_ids_override)
        twin_ids_source = "live twin check (twin_ids_override; no per-pack artifact)"
        twin_ids_sha = hashlib.sha256(json.dumps(sorted(twin_ids)).encode()).hexdigest()
    elif (twin_ids_path := twin_written_ids_path(artifact_root, pack_id)).exists():
        twin_ids_bytes = twin_ids_path.read_bytes()
        twin_ids = json.loads(twin_ids_bytes)
        twin_ids_source = str(twin_ids_path)
        twin_ids_sha = hashlib.sha256(twin_ids_bytes).hexdigest()
    else:
        raise QueueError(
            f"{pack_id}: resume requested but {twin_ids_path} (the twin "
            "leg's own written-ids artifact) does not exist -- refusing to "
            "guess which ids twin already landed.")

    def preflight() -> dict:
        present_on_twin = set(PSP.already_present(twin_url, twin_collection, twin_ids))
        missing_on_twin = set(twin_ids) - present_on_twin
        if missing_on_twin:
            raise QueueError(
                f"{pack_id}: resume premise failed -- {len(missing_on_twin)} "
                f"of {len(twin_ids)} id(s) the twin leg reported written "
                f"are no longer present on twin (e.g. "
                f"{sorted(missing_on_twin)[:5]}). Not a prod-only resume.")
        return {"twin_ids_count": len(twin_ids)}

    def post_twin_gate(st: dict) -> dict:
        return {"status": "PASS", "mechanical": True, "pack_id": pack_id,
               "note": "twin already verified by the ORIGINAL apply; "
                       "re-confirmed present by this resume's own preflight"}

    def apply_prod(st: dict) -> None:
        # Round 4 finding 2: re-confirm, under the lock, immediately
        # before this pack's prod write.
        _Q._revalidate_prod_collection(pack_id, prod_url, prod_alias,
                                    resolved_prod_collection)
        before = LAYOUT.fingerprint(prod_url, resolved_prod_collection)
        # Task 788: the KNOWN twin-written set rides into plan(), so a PARTIAL prod write (an
        # earlier attempt that died after k of n batches) resumes: prod's present ids must all be
        # twin-written (skipped), the rest of the twin-written ids are written, and an id prod holds
        # that the twin leg did not write refuses as a collision. Before this, plan() applied the
        # first-time rule here and refused every hour forever (task 786 F1).
        receipt, ids_path = _Q.apply_side("prod", prod_url, manifest, resolved_prod_collection,
                                       batch=batch, pause_s=pause_s, parallel=1,
                                       ids_dir=artifact_root,
                                       write_attempt=write_attempt,
                                       resume_twin_ids=frozenset(twin_ids))
        st["resume_dropped_ids"] = receipt.pop("_resume_dropped_ids", [])    # never in a receipt
        st["prod_before"] = before
        st["prod_receipt"] = receipt
        st["prod_ids_path"] = str(ids_path)

    def verify_mirrors(st: dict) -> dict:
        present_after = set(PSP.already_present(prod_url, resolved_prod_collection, twin_ids))
        missing = set(twin_ids) - present_after
        if missing:
            raise QueueError(
                f"{pack_id}/prod: {len(missing)} id(s) missing after resume "
                f"upsert (membership check failed): {sorted(missing)[:20]}")
        receipt = st["prod_receipt"]
        planned = receipt.get("resume") or {}
        # Task 788, tranche 14 (2026-10-09): the resume wrote the whole pack (1,409,761) while the
        # twin leg had written 1,409,548 -- 213 gate-rejected rows reached prod only. The leg's
        # write set must BE the twin's: compare the two artifacts' id sets, and fail loudly (no
        # DONE) on any difference.
        prod_artifact = set(json.loads(pathlib.Path(st["prod_ids_path"]).read_text()))
        twin_set = set(twin_ids)
        if prod_artifact != twin_set:
            raise QueueError(
                f"{pack_id}/prod: the prod written-ids artifact differs from the twin's "
                f"({len(prod_artifact - twin_set):,} prod-only, e.g. "
                f"{sorted(prod_artifact - twin_set)[:5]}; {len(twin_set - prod_artifact):,} "
                f"twin-only, e.g. {sorted(twin_set - prod_artifact)[:5]}). The prod leg's write set "
                "must be exactly the twin's; NOT marking DONE.")
        # Review 797 S1: the artifact says what this leg MEANT; prod says what IS there. One by-id
        # check (retrieve, never a scroll) that no id the twin leg dropped is on prod (213 for
        # tranche 14: one request).
        dropped = list(st.get("resume_dropped_ids") or ())
        dropped_on_prod = sorted(PSP.already_present(prod_url, resolved_prod_collection, dropped)) \
            if dropped else []
        if dropped_on_prod:
            raise QueueError(
                f"{pack_id}/prod: {len(dropped_on_prod):,} id(s) the twin leg dropped are on prod "
                f"(e.g. {dropped_on_prod[:5]}): prod diverges from twin; NOT marking DONE.")
        present_before = planned.get("present_before", 0)
        if present_before + receipt["written"] != len(set(twin_ids)):
            raise QueueError(
                f"{pack_id}/prod: resume arithmetic does not close: {present_before:,} present "
                f"before + {receipt['written']:,} written != {len(set(twin_ids)):,} twin-written ids")
        resume = {"resumed_partial": present_before > 0,
                  "present_before": present_before,
                  "written": receipt["written"],
                  "present_after": len(present_after),
                  "twin_written_ids": len(set(twin_ids)),
                  "twin_written_ids_sha256": twin_ids_sha,
                  "twin_written_ids_source": twin_ids_source,
                  "dropped_by_twin_decision": planned.get("dropped_by_twin_decision", 0),
                  "dropped_present_after": 0,
                  "prod_written_ids_equal_twin": True}
        receipt["resumed_partial"] = resume["resumed_partial"]
        receipt["resume"] = {**planned, **resume}
        # #414 review follow-up 1: flip scripts take no writer lock, so the
        # alias can move after the write; re-resolve as the LAST prod check
        # before this verify returns (and the pack goes DONE).
        _Q._revalidate_prod_collection(pack_id, prod_url, prod_alias,
                                    resolved_prod_collection)
        return {"status": "PASS", "role": "prod",
               "written": receipt["written"], "resume": resume}

    taken: list[str] = []
    try:
        for name in runb.LOCKS:
            runb.acquire(name, task, f"{pack_id}-promotion-queue-resume",
                        pid=os.getpid())
            taken.append(name)
        state = _Q.TFA.resume_after_verified_twin_gate(
            ledger=ledger, lock=lock, preflight=preflight,
            post_twin_gate=post_twin_gate, apply_prod=apply_prod,
            verify_mirrors=verify_mirrors, armed=True,
            preservation_verifier=lambda: {
                "status": "PASS",
                "scope": "narrow: RESUME of a sealed pack's prod leg only; "
                         "twin was already written and verified by a prior "
                         "run and is never rewritten here",
                "pack_id": pack_id})
    finally:
        for name in reversed(taken):
            runb.release(name, task)
    return state

def resume_content_pack_prod_only(*, pack_id: str, manifest: dict, twin_url: str,
                                  prod_url: str, task: str, batch: int,
                                  pause_s: float, artifact_root: pathlib.Path,
                                  state_record: dict,
                                  prod_collection: str,
                                  write_attempt: WriteAttempt | None = None,
                                  prod_alias: str = "aw_prod") -> dict:
    """Resume ONLY the prod leg of a CONTENT PACK (`kind: content_pack`,
    e.g. `l1-backfill-*`, news) whose twin leg already landed and verified
    -- the content-pack counterpart of `resume_prod_only()` (task 673).

    Unlike a sealed pack, a content pack's own apply does BOTH an upsert
    AND a tombstone-delete in one call (`apply_content_pack_side`), so the
    resume must finish both, not just the upsert. `state_record` must be
    the `"twin-write-complete"` stage (its `pack_ids`/`tomb_ids` are the
    EXACT ids the original apply verified on twin, post word-coverage-skip
    filtering) -- a `"twin-write-started"` record belongs to `drain()`'s
    full-replay path instead, since this function's `preflight` assumes
    twin already has every id and never touches it again.

    `prod_collection` (round 3 finding 6/FLIP BLOCKER; made REQUIRED, no
    fallback, round 4 finding 3): the collection to use for the PROD leg
    -- resolved by the caller through the live `aw_prod` alias, never
    `manifest["collection"]` (twin's own fixed name, `aw_e8`, unaffected
    by a flip). No fallback: a caller that forgot to resolve the alias
    must fail loudly.

    `prod_alias` (round 4 finding 2): re-resolved and compared against
    `prod_collection` immediately before the PROD write, while the
    production lock is held (see `_revalidate_prod_collection`).
    """
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    if state_record.get("stage") != "twin-write-complete":
        raise QueueError(
            f"{pack_id}: resume_content_pack_prod_only requires a "
            "'twin-write-complete' state record (got "
            f"{state_record.get('stage')!r}) -- a 'twin-write-started' "
            "record belongs to drain()'s full-replay path instead.")
    runb = _Q._load_run_batch()
    artifact_root.mkdir(parents=True, exist_ok=True)
    ledger = artifact_root / f"{pack_id}-resume-ledger.jsonl"
    lock = _Q.TFA.ProductionWriterLock(host=_Q.prod_lock_host(), lock_path=PROD_LOCK_PATH,
                                    purpose=f"task{task}-{pack_id}-resume")
    runb.assert_locks_free(task)
    twin_collection = manifest["collection"]
    resolved_prod_collection = prod_collection
    pack_ids: list[str] = state_record.get("pack_ids") or []
    tomb_ids: list[str] = state_record.get("tomb_ids") or []
    pack_id_set = set(pack_ids)
    points, tombstones = load_content_pack_ids_from_state(manifest, state_record)

    def preflight() -> dict:
        present_on_twin = set(PSP.already_present(twin_url, twin_collection, pack_ids)) \
            if pack_ids else set()
        missing_on_twin = pack_id_set - present_on_twin
        if missing_on_twin:
            raise QueueError(
                f"{pack_id}: resume premise failed -- {len(missing_on_twin)} "
                f"pack id(s) the twin leg reported written are no longer "
                f"present on twin (e.g. {sorted(missing_on_twin)[:5]}). "
                "Not a prod-only resume.")
        still_on_twin = set(PSP.already_present(twin_url, twin_collection, tomb_ids)) \
            if tomb_ids else set()
        if still_on_twin:
            raise QueueError(
                f"{pack_id}: resume premise failed -- {len(still_on_twin)} "
                f"tombstone id(s) the twin leg reported deleted are still "
                f"present on twin (e.g. {sorted(still_on_twin)[:5]}). Not a "
                "prod-only resume.")
        return {"pack_ids_count": len(pack_ids), "tomb_ids_count": len(tomb_ids)}

    def post_twin_gate(st: dict) -> dict:
        return {"status": "PASS", "mechanical": True, "pack_id": pack_id,
               "note": "twin already verified by the ORIGINAL apply; "
                       "re-confirmed present/absent by this resume's own "
                       "preflight"}

    @PSP.retry_leg()
    def apply_prod(st: dict) -> None:
        # Round 4 finding 2: re-confirm, under the lock, immediately
        # before this pack's prod write.
        _Q._revalidate_prod_collection(pack_id, prod_url, prod_alias,
                                    resolved_prod_collection)
        # Task 712: the export of a `superseded_generation` pack is re-verified (sha256 and
        # every row) before this prod leg; a mismatch stops the drain, the twin state stays.
        UGS.reverify_generation_export(manifest, tombstones, twin_url=twin_url,
                                       twin_collection=twin_collection)
        before = LAYOUT.fingerprint(prod_url, resolved_prod_collection)
        prod_before = PSP.PROD_STATS.mark()
        # Task 2942, a DEFENCE (no live fault today: `apply_content_pack_side` always upserts, and the rule's
        # upsert records a fresh readback). The readback is kept per collection for the whole process; forget the
        # earlier pack's, so a future apply that upserts nothing reports none (no `sparse_rule_readback` key).
        SR.LAST_READBACK.pop(resolved_prod_collection, None)
        receipt = _Q.apply_content_pack_side(
            "prod", prod_url, resolved_prod_collection, points, tombstones,
            batch=batch, pause_s=pause_s, write_attempt=write_attempt,
            checkpoint=(prod_chunk_checkpoint(artifact_root, pack_id,
                                              state_record.get("manifest_sha"), "write")
                        if PG.is_big_pack(len(points)) else None))
        calls = prod_call_receipt(len(points), prod_before)
        if calls is not None:
            receipt["prod_calls"] = calls
        st["prod_before"] = before
        st["prod_receipt"] = receipt

    def verify_mirrors(st: dict) -> dict:
        present_after = set(PSP.already_present(prod_url, resolved_prod_collection, pack_ids)) \
            if pack_ids else set()
        missing = pack_id_set - present_after
        if missing:
            raise QueueError(
                f"{pack_id}/prod: {len(missing)} pack id(s) missing after "
                f"resume upsert (membership check failed): "
                f"{sorted(missing)[:20]}")
        still_present = set(PSP.already_present(prod_url, resolved_prod_collection, tomb_ids)) \
            if tomb_ids else set()
        if still_present:
            raise QueueError(
                f"{pack_id}/prod: {len(still_present)} tombstoned id(s) "
                f"still present after resume delete (membership check "
                f"failed): {sorted(still_present)[:20]}")
        # Task 833 (G2): membership is not content. Read back the in-place rows first (this
        # resume made no pre-write lookup on prod: the ids the ORIGINAL legs recorded, prod's
        # when its lookup ran, else the twin's), every key + dense, pack vs prod AND twin vs prod.
        inplace, inplace_source = read_inplace_ids(
            artifact_root, pack_id, manifest_sha=state_record.get("manifest_sha"))
        payload_form = _Q.verify_written_form(
            prod_url, resolved_prod_collection, points, inplace_ids=inplace,
            twin=(twin_url, twin_collection))
        payload_form["inplace_source"] = inplace_source
        # #414 review follow-up 1: flip scripts take no writer lock, so the
        # alias can move after the write; re-resolve as the LAST prod check
        # before this verify returns (and the pack goes DONE).
        _Q._revalidate_prod_collection(pack_id, prod_url, prod_alias,
                                    resolved_prod_collection)
        return {"status": "PASS", "role": "prod",
               "written": st["prod_receipt"]["written"],
               "tombstoned": st["prod_receipt"]["tombstoned"],
               "payload_form": payload_form}

    taken: list[str] = []
    try:
        for name in runb.LOCKS:
            runb.acquire(name, task, f"{pack_id}-promotion-queue-resume",
                        pid=os.getpid())
            taken.append(name)
        state = _Q.TFA.resume_after_verified_twin_gate(
            ledger=ledger, lock=lock, preflight=preflight,
            post_twin_gate=post_twin_gate, apply_prod=apply_prod,
            verify_mirrors=verify_mirrors, armed=True,
            preservation_verifier=lambda: {
                "status": "PASS",
                "scope": "narrow: RESUME of a content pack's prod leg "
                         "only (upsert + tombstone-delete); twin was "
                         "already written and verified by a prior run and "
                         "is never rewritten here",
                "pack_id": pack_id})
    finally:
        for name in reversed(taken):
            runb.release(name, task)
    return state
