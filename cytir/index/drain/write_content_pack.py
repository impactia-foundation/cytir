"""Drain content-pack write path: `apply_content_pack_side` (upsert the points, then delete the tombstones
on one side) and `live_apply_content_pack` (the twin-first transaction around it).

Task 2911 (RULING 80, drain split step 11): moved verbatim out of `cytir/index/promote_queue.py`, which
re-exports both names. A moved body that calls a name a test patches on the facade (or a name that
stays there) reads it at call time through `_Q` (the facade module).
"""
from __future__ import annotations

import os
import pathlib
import time

from cytir.index import chunk_checkpoint as CK
from cytir.index import id_rule as IR
from cytir.index import layout as LAYOUT
from cytir.index import point_delete as PD
from cytir.index import prod_guard as PG
from cytir.index import promote_sealed_pack as PSP
from cytir.index import sparse_rule as SR
from cytir.index import twin_first as TFA
from cytir.index import url_group_supersede as UGS
from cytir.index.drain.common import QueueError, WriteAttempt
from cytir.index.drain.locks import PROD_LOCK_PATH
from cytir.index.drain.receipts import (
    inplace_ids_path, lookup_record_scope, prod_call_receipt, prod_chunk_checkpoint, record_inplace_ids,
    write_content_pack_twin_state
)


# ------------------------------------------------------------- apply a side
@PSP.retry_leg()
def apply_content_pack_side(role: str, url: str, collection: str, points: list[dict],
                           tombstones: list[dict], *, batch: int,
                           pause_s: float,
                           write_attempt: WriteAttempt | None = None,
                           checkpoint: "CK.ChunkCheckpoint | None" = None) -> dict:
    """Upsert the points, then delete the tombstones -- ONE transaction for
    this side. Order (upsert before delete) means the replacement content
    is on the target before the superseded rows are removed.

    `collection` is explicit, not derived from a manifest (task 673 review
    round 3, finding 6/FLIP BLOCKER): the TWIN and PROD sides can need
    DIFFERENT collection names -- twin is a fixed, configured collection
    (the build box's `aw_e8` stays the twin forever), but prod is served
    through the `aw_prod` alias, which the `aw_e9` flip repoints to a
    DIFFERENT physical collection; `aw_e8` on prod is deleted once that
    happens. A caller that derived `collection` from `manifest["collection"]`
    for BOTH sides would keep writing prod to a collection that no longer
    exists after the flip. The caller (`live_apply_content_pack`/`resume_
    content_pack_prod_only`) resolves the correct value per role and
    passes it in here."""
    # Task 864 (review 869 S4): a content pack carries its own ids, so the I1 guard can refuse it.
    # Checked HERE, before `write_attempt.started`, so a refusal is a clean REFUSED (nothing
    # written, the drain goes on), not an ABORTED mid-write that stops the drain. `upsert` checks
    # again (the door). Off for every collection without `id_rule`.
    IR.refuse_unless_i1(collection, points)
    before = LAYOUT.fingerprint(url, collection)
    t0 = time.time()
    if write_attempt is not None:
        write_attempt.started = True
    # task 786: a big pack's PROD leg sleeps longer between batches; the twin leg is never changed.
    written = PSP.upsert(url, collection, points, batch,
                         pause_s=PG.write_pause(role, len(points), pause_s),
                         **({} if checkpoint is None else {"checkpoint": checkpoint}))
    if checkpoint is not None:
        # Spent the moment every batch is sent: a record that outlived a FAILED membership proof would
        # make the next attempt skip the very batch that went missing.
        checkpoint.clear()
    tombstoned = 0
    if tombstones:
        ids = [t["id"] for t in tombstones]
        tombstoned = PD.delete_points_verified(url, collection, ids,
                                              batch=batch)
    after = LAYOUT.fingerprint(url, collection)
    # Task 1467: `tombstoned` = ids that were present and are now verified gone. `tombstones_requested`
    # is what this side was asked to delete; the difference is `tombstones_not_found` (absent before).
    return {"role": role, "written": written, "tombstoned": tombstoned,
           "tombstones_requested": len(tombstones),
           "tombstones_not_found": len(tombstones) - tombstoned,
           "write_seconds": round(time.time() - t0, 3),
           "layout_before": before, "layout_after": after}


def live_apply_content_pack(*, pack_id: str, manifest: dict, twin_url: str,
                           prod_url: str, task: str, batch: int, pause_s: float,
                           artifact_root: pathlib.Path, points: list[dict],
                           tombstones: list[dict],
                           prod_collection: str,
                           write_attempt: WriteAttempt | None = None,
                           manifest_sha: str | None = None,
                           prod_alias: str = "aw_prod") -> dict:
    """Twin -> verify -> prod -> verify, under the same `twin_first_apply`
    protocol every other pack uses -- the ONLY difference from `live_apply`
    is what `apply_target`/`verify_target` do (upsert+tombstone instead of
    plan()+build_points()).

    `prod_collection` (task 673 review round 3, finding 6/FLIP BLOCKER;
    made REQUIRED, no fallback, round 4 finding 3): the collection to use
    for the PROD leg specifically -- resolved by the caller (`drain()`)
    through the live `aw_prod` alias, NEVER `manifest["collection"]` (that
    name is `aw_e8`, which the `aw_e9` flip deletes from prod; twin keeps
    using `manifest["collection"]`, its own fixed, configured name,
    unaffected by any flip). A silent `manifest["collection"]` fallback
    here would let a caller that forgot to resolve the alias write the
    wrong, possibly-deleted collection without ever finding out -- every
    prod-facing caller must resolve it and pass it explicitly.

    `prod_alias` (round 4 finding 2): re-resolved and compared against
    `prod_collection` immediately before the PROD write, while the
    production lock is held (see `_revalidate_prod_collection`) -- an
    hourly drain can run for hours, and a flip mid-run must not silently
    write to a collection prod no longer serves."""
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    runb = _Q._load_run_batch()
    artifact_root.mkdir(parents=True, exist_ok=True)
    ledger = artifact_root / f"{pack_id}-ledger.jsonl"
    lock = TFA.ProductionWriterLock(host=_Q.prod_lock_host(), lock_path=PROD_LOCK_PATH,
                                    purpose=f"task{task}-{pack_id}")
    runb.assert_locks_free(task)
    state: dict = {}
    twin_collection = manifest["collection"]
    resolved_prod_collection = prod_collection

    def preflight():
        return state

    pack_ids = [p["id"] for p in points]
    tomb_ids = [t["id"] for t in tombstones]

    @PSP.retry_leg()
    def apply_target(role: str, st: dict) -> None:
        if role == "twin":
            # Written BEFORE any network call for this leg -- task 673: a
            # SIGKILL any time from here through `verify_target("twin",
            # ...)` must leave a resumable record, not silence.
            write_content_pack_twin_state(
                artifact_root, pack_id, stage="twin-write-started",
                manifest_sha=manifest_sha, pack_ids=pack_ids, tomb_ids=tomb_ids)
        else:
            # Round 4 finding 2: re-confirm, under the lock, immediately
            # before this pack's prod write.
            _Q._revalidate_prod_collection(pack_id, prod_url, prod_alias,
                                           resolved_prod_collection)
            # Task 712: a pack that deletes under `superseded_generation` re-verifies its export
            # before the prod leg (a replay from "twin-write-started" never ran the dry check).
            UGS.reverify_generation_export(manifest, tombstones, twin_url=twin_url,
                                           twin_collection=twin_collection)
        url = twin_url if role == "twin" else prod_url
        collection = twin_collection if role == "twin" else resolved_prod_collection
        # The expectation is computed BEFORE the write, per target --
        # coordinator's ruling, task 641, 2026-09-23: `written - tombstoned`
        # assumes every written row is net-new, which is false for a
        # recapture pack (some point ids already exist and get updated in
        # place, net delta 0 for those). The real expectation is "how many
        # of THIS target's rows are genuinely new" and "how many of THIS
        # target's tombstones are genuinely present to remove" -- twin and
        # prod can differ here (a pack can be new-to-twin but a recapture
        # on prod, or vice versa), so this is per-role, not shared.
        # task 786: the PROD leg checkpoints its two long steps (the pre-write lookup, the upsert
        # batches) per chunk, so a retry after an abort re-enters at the first unfinished chunk.
        prod_before = PSP.PROD_STATS.mark()
        lookup_ck = write_ck = None
        if role == "prod" and PG.is_big_pack(len(pack_ids)):     # an ordinary pack: the call as before
            lookup_ck = prod_chunk_checkpoint(artifact_root, pack_id, manifest_sha, "lookup")
            write_ck = prod_chunk_checkpoint(artifact_root, pack_id, manifest_sha, "write")
        with lookup_record_scope(lookup_ck):
            pre_present_pack = set(PSP.already_present(
                url, collection, pack_ids, **({} if lookup_ck is None else {"checkpoint": lookup_ck}))) \
                if pack_ids else set()
        pre_present_tomb = set(PSP.already_present(url, collection, tomb_ids)) \
            if tomb_ids else set()
        st[f"{role}_expect"] = {
            "new": len(pack_ids) - len(pre_present_pack),
            "removed": len(pre_present_tomb),
            "expected_delta": (len(pack_ids) - len(pre_present_pack))
                             - len(pre_present_tomb)}
        # Task 833 (G2): the ids this write OVERWRITES on this target, kept per leg (the first
        # lookup of this manifest wins) for the verify's sample and for a later audit.
        st[f"{role}_inplace"] = record_inplace_ids(
            artifact_root, pack_id, role=role, collection=collection,
            manifest_sha=manifest_sha, ids=pre_present_pack)
        # Task 2942, a DEFENCE (no live fault today: `apply_content_pack_side` always upserts, and the rule's
        # upsert records a fresh readback). The readback is kept per collection for the whole process; forget the
        # earlier pack's, so a future apply that upserts nothing reports none (no `sparse_rule_readback` key).
        SR.LAST_READBACK.pop(collection, None)
        st[f"{role}_receipt"] = _Q.apply_content_pack_side(
            role, url, collection, points, tombstones, batch=batch, pause_s=pause_s,
            write_attempt=write_attempt, checkpoint=write_ck)
        if role == "prod":
            calls = prod_call_receipt(len(points), prod_before)
            if calls is not None:
                st["prod_receipt"]["prod_calls"] = calls

    def verify_target(role: str, st: dict) -> dict:
        url = twin_url if role == "twin" else prod_url
        collection = twin_collection if role == "twin" else resolved_prod_collection
        receipt = st[f"{role}_receipt"]
        expect = st[f"{role}_expect"]
        # MEMBERSHIP IS THE PROOF, not points_count (coordinator's ruling):
        # every pack id must now be present, every tombstoned id must now
        # be absent. points_count can lag mid-optimization (a known
        # false-mismatch, memory `aw-heavy-rewrite-degrades-prod`-adjacent)
        # and is checked only as a secondary, non-fatal signal.
        present_after_pack = set(PSP.already_present(url, collection, pack_ids)) \
            if pack_ids else set()
        missing_pack = set(pack_ids) - present_after_pack
        if missing_pack:
            raise QueueError(
                f"{pack_id}/{role}: {len(missing_pack)} pack id(s) missing "
                f"after upsert (membership check failed): "
                f"{sorted(missing_pack)[:20]}")
        still_present_tomb = set(PSP.already_present(url, collection, tomb_ids)) \
            if tomb_ids else set()
        if still_present_tomb:
            raise QueueError(
                f"{pack_id}/{role}: {len(still_present_tomb)} tombstoned "
                f"id(s) still present after delete (membership check "
                f"failed): {sorted(still_present_tomb)[:20]}")
        # Task 675: membership is not representation. Read back a sample in the
        # TARGET's declared payload form (aw_e9 normalized, aw_e8 raw). Task 833 (G2):
        # in-place rows first, every key + dense, and twin vs prod on the prod leg.
        payload_form = _Q.verify_written_form(
            url, collection, points, inplace_ids=st.get(f"{role}_inplace") or (),
            twin=(twin_url, twin_collection) if role == "prod" else None)
        payload_form["inplace_ids_artifact"] = str(inplace_ids_path(artifact_root, pack_id))
        if role == "twin":
            # Membership just proved for real -- mark complete so a
            # SIGKILL before the prod leg starts leaves a resumable record
            # instead of an orphan (task 673).
            write_content_pack_twin_state(
                artifact_root, pack_id, stage="twin-write-complete",
                manifest_sha=manifest_sha, pack_ids=pack_ids, tomb_ids=tomb_ids)
        before_n = receipt["layout_before"].get("points_count") or 0
        after_n = receipt["layout_after"].get("points_count") or 0
        delta = after_n - before_n
        count_note = None
        if delta != expect["expected_delta"]:
            count_note = (
                f"points_count moved by {delta:+,}, expected "
                f"{expect['expected_delta']:+,} (new={expect['new']:,}, "
                f"removed={expect['removed']:,}) -- membership check PASSED "
                "so this is logged, not fatal (points_count can lag during "
                "optimization)")
            print(f"[{pack_id}/{role}] {count_note}")
        # #414 review follow-up 1: flip scripts take no writer lock, so the
        # alias can move after the write; re-resolve as the LAST prod check
        # before this verify returns (and the pack goes DONE).
        if role == "prod":
            _Q._revalidate_prod_collection(pack_id, prod_url, prod_alias,
                                           resolved_prod_collection)
        return {"role": role, "written": receipt["written"],
               "tombstoned": receipt["tombstoned"], "points_delta": delta,
               "expected_delta": expect["expected_delta"],
               "count_note": count_note, "payload_form": payload_form,
               "status": "PASS"}

    def post_twin_gate(st: dict, twin_proof: dict) -> dict:
        return {"status": "PASS", "mechanical": True, "pack_id": pack_id}

    taken: list[str] = []
    try:
        for name in runb.LOCKS:
            runb.acquire(name, task, f"{pack_id}-promotion-queue", pid=os.getpid())
            taken.append(name)
        TFA.twin_first(
            ledger=ledger, lock=lock, preflight=preflight,
            apply_target=apply_target, verify_target=verify_target,
            post_twin_gate=post_twin_gate, armed=True,
            preservation_verifier=lambda: {
                "status": "PASS",
                "scope": "narrow: content-pack upsert + declared-tombstone "
                         "delete; no release is published by this "
                         "transaction",
                "pack_id": pack_id})
    finally:
        for name in reversed(taken):
            runb.release(name, task)
    return state
