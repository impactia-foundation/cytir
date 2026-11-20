"""Drain sealed write path: `apply_side` (plan + write one target), `live_apply` (the twin-first transaction),
`single_apply` and `rehearse`.

Task 2902 (RULING 80, drain split step 10): moved verbatim out of `cytir/index/promote_queue.py`, which
re-exports every name here. A moved body that calls a name a test patches on the facade (or a name that
stays there) reads it at call time through `_Q` (the facade module).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time

from cytir.gates import rollback_restore_null_arm as RB
from cytir.index import dry_run as DRY
from cytir.index import layout as LAYOUT
from cytir.index import manifest_rows as MR
from cytir.index import prod_guard as PG
from cytir.index import promote_sealed_pack as PSP
from cytir.index import twin_first as TFA
from cytir.index import url_group_supersede as UGS
from cytir.index.drain.common import QueueError, Single, WriteAttempt, atomic_write_text, now, prod_lock_host
from cytir.index.drain.locks import PROD_LOCK_PATH
from cytir.index.drain.plan import LOOKUP_WORKERS
from cytir.index.drain.queue_scope import assert_real_path, read_manifest
from cytir.index.drain.receipts import lookup_record_scope, prod_call_receipt, prod_chunk_checkpoint
from cytir.index.drain.verify import _revalidate_prod_collection


# ------------------------------------------------------------- apply a side
@PSP.retry_leg()
def apply_side(role: str, url: str, manifest: dict, collection: str, *, batch: int,
              pause_s: float, ids_dir: pathlib.Path,
              drop_ids: frozenset[str] = frozenset(),
              parallel: int = 1,
              precomputed: tuple | None = None,
              write_attempt: WriteAttempt | None = None,
              persisted_check: bool = True,
              profile_collection: str | None = None,
              resume_twin_ids: frozenset[str] | None = None) -> tuple[dict, pathlib.Path]:
    """Plan + write one target. Returns (receipt, written-ids path).

    `resume_twin_ids` (task 788): passed ONLY by `resume_prod_only`, the prod leg of a sealed pack
    whose twin leg landed. `plan()` then writes exactly the twin-written ids prod lacks and skips
    the ones prod already holds (see `PSP.resume_partition`). Before any write the set is checked
    again (present + sent == the twin's set, nothing sent twice), else `QueueError` with no write
    started. A resume keeps NO batch record of its own (its fresh by-id lookup of prod is its
    record), and the first attempt's `write` record is deleted as soon as the resume starts, so no
    exit, a refusal included, leaves it behind. An empty remainder writes nothing. None = unchanged.

    Mirrors `promote_sealed_pack.main()`'s apply path exactly (same plan(),
    same build_points(), same upsert()) rather than re-deriving any of it;
    the only additions are the throttle, `parallel` (task 641, 2026-09-23:
    fires that many upsert batches concurrently -- pass this for the TWIN
    leg only, never prod), `precomputed` (skip re-deriving `joined`/
    `matrix`/`id_index` via a second full `plan()` when the caller already
    has them from its own dry_plan() against THIS SAME url -- `drain()`
    always dry-plans against the twin right before calling this for the
    twin leg, so that second plan() was a byte-for-byte repeat of the
    first: same already_present/already_served_urls/persisted-inputs/
    payload-carry cost, paid twice per pack for no reason), `drop_ids`
    (rows the mechanical-content-v1 check failed and this pack's failing
    share was low enough to drop rather than refuse -- computed once in
    `drain()`'s dry-run pass, applied here at write time), and this
    module's receipt shape.

    `collection` is explicit, not `manifest["collection"]` (task 673
    review round 3, finding 6/FLIP BLOCKER): twin uses a fixed, configured
    collection name; prod is served through the `aw_prod` alias, which the
    `aw_e9` flip repoints -- `manifest["collection"]` (`aw_e8`) is deleted
    from prod once that happens. The caller (`live_apply`/`resume_prod_
    only`) resolves the correct value per role."""
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    # task 786: the PROD leg of a big sealed pack re-runs plan() against prod (2,820 id retrieves for
    # tranche 14) and then sends ~1,400 PUT batches. Both steps keep a chunk-level record, so an aborted
    # attempt is resumed at the first unfinished chunk or batch, not from row 0. An ordinary pack and
    # the twin leg pass nothing new.
    if resume_twin_ids is not None and (role != "prod" or precomputed is not None or drop_ids):
        raise QueueError(
            f"{manifest['pack_id']}: resume_twin_ids is for the prod leg's resume only "
            "(role prod, its own plan(), no drop_ids)")
    prod_before = PSP.PROD_STATS.mark()
    lookup_ck = write_ck = resume_present = resume_dropped = None
    if role == "prod" and PG.is_big_pack(MR.rows_to_write_max(manifest) or 0):
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()
        lookup_ck = prod_chunk_checkpoint(ids_dir, manifest["pack_id"], digest, "lookup")
        if resume_twin_ids is None:
            write_ck = prod_chunk_checkpoint(ids_dir, manifest["pack_id"], digest, "write")
        else:
            # task 788: a resume's own record IS its fresh by-id lookup of prod (what prod lacks is
            # written, nothing else), so it keeps no batch record. The first attempt's batch record
            # is superseded the moment a resume starts and is deleted now, so no exit (review 797
            # S3: a refusal inside plan() used to leave it) can leave it for a later fresh apply.
            prod_chunk_checkpoint(ids_dir, manifest["pack_id"], digest, "write").clear()
    if precomputed is not None:
        joined, matrix, id_index, fingerprint_before = precomputed
        receipt: dict = {"layout_before": fingerprint_before}
    else:
        # `plan_kwargs_with_scope`, not `plan_kwargs` -- this call must
        # exclude exactly the ids `dry_plan()` already excluded (task 660,
        # see `language_scope_and_waiver_exclusions`'s docstring). Using the
        # bare `plan_kwargs(manifest)` here reproduced the pack's UNSCOPED
        # population at write time and refused a pack the dry-run had just
        # proved clean, inside the locked twin transaction.
        kwargs, _, _ = _Q.plan_kwargs_with_scope(manifest)
        if persisted_check is not True:   # only a Single transaction ever sets this
            kwargs["persisted_check"] = persisted_check
        if lookup_ck is not None:
            kwargs["lookup_checkpoint"] = lookup_ck
        if resume_twin_ids is not None:
            kwargs["resume_twin_ids"] = resume_twin_ids
        with lookup_record_scope(lookup_ck):
            receipt = PSP.plan(chunks_dir=manifest["pack_chunks"],
                               vectors_dir=manifest["vectors"], qdrant=url,
                               collection=collection,
                               mrl=int(manifest["mrl"]),
                               lookup_workers=LOOKUP_WORKERS.get(role, 1), **kwargs)
        joined = receipt.pop("_joined")
        matrix = receipt.pop("_matrix")
        id_index = receipt.pop("_id_index")
        receipt.pop("_already_served_pids", None)
        resume_present = receipt.pop("_resume_present", None)
        resume_dropped = receipt.pop("_resume_dropped", None)
        receipt["layout_before"] = LAYOUT.fingerprint(url, collection)
    if drop_ids:
        joined = {pid: v for pid, v in joined.items() if pid not in drop_ids}
        receipt["mechanical_content_dropped"] = len(drop_ids)
    points = PSP.build_points(joined, matrix, id_index, int(manifest["mrl"]))
    if resume_twin_ids is not None:
        # Review 797 M2: the write set is checked BEFORE any byte is sent, not only after: what prod
        # held plus what this leg sends must be exactly the twin's set, and nothing is sent twice.
        sent = [p["id"] for p in points]
        have, twin = set(resume_present or ()), set(resume_twin_ids)
        extra, short, twice = (set(sent) | have) - twin, twin - (set(sent) | have), set(sent) & have
        if extra or short or twice:
            raise QueueError(
                f"{manifest['pack_id']}/prod: the resume's write set would make prod differ from the "
                f"twin's written ids ({len(extra):,} not twin-written, e.g. {sorted(extra)[:5]}; "
                f"{len(short):,} twin ids neither present nor sent, e.g. {sorted(short)[:5]}; "
                f"{len(twice):,} sent though already present). Refusing before any write.")
        # S1: the ids the twin leg dropped, for one by-id check after the write (popped by the caller)
        receipt["_resume_dropped_ids"] = list(resume_dropped or ())
    ids_path = ids_dir / f"{manifest['pack_id']}-{role}-written-ids.json"
    if resume_twin_ids is None:
        atomic_write_text(ids_path, json.dumps([p["id"] for p in points], indent=1) + "\n")
    else:
        # task 788: on a resume the artifact names every id of this pack prod holds once the leg is
        # done -- what an earlier attempt landed plus what this one sends -- so it compares 1:1 with
        # the twin's artifact (`resume_prod_only` asserts the two sets are equal).
        atomic_write_text(ids_path, json.dumps(
            sorted(set(resume_present or ()) | {p["id"] for p in points}), indent=1) + "\n")
    t0 = time.time()
    if resume_twin_ids is not None and not points:
        # task 788: prod already holds every twin-written id -- nothing to send, no write started.
        written = 0
    else:
        if write_attempt is not None:
            write_attempt.started = True
        upsert_kwargs = ({} if profile_collection is None
                         else {"profile_collection": profile_collection})
        if write_ck is not None:
            upsert_kwargs["checkpoint"] = write_ck
        written = PSP.upsert(url, collection, points, batch,
                             pause_s=PG.write_pause(role, len(points), pause_s),
                             parallel=parallel, **upsert_kwargs)
    if write_ck is not None:
        write_ck.clear()                # every batch sent; see apply_content_pack_side
    receipt["written"] = written
    receipt["write_seconds"] = round(time.time() - t0, 3)
    receipt["layout_after"] = LAYOUT.fingerprint(url, collection)
    if role == "prod":
        calls = prod_call_receipt(len(points), prod_before)
        if calls is not None:                    # a big pack, or a leg that had to retry
            receipt["prod_calls"] = calls
    return receipt, ids_path


def live_apply(*, pack_id: str, manifest: dict, twin_url: str, prod_url: str,
              task: str, batch: int, pause_s: float,
              artifact_root: pathlib.Path,
              prod_collection: str,
              drop_ids: frozenset[str] = frozenset(),
              twin_parallel: int = 1,
              write_attempt: WriteAttempt | None = None,
              prod_alias: str = "aw_prod",
              supersede: dict | None = None) -> dict:
    """Twin -> verify -> gate (mechanical, already proved by the caller's dry
    run + mechanical-content-v1 check) -> prod -> verify, under
    `twin_first_apply.twin_first` -- the same ratified transaction protocol
    `run_batch.py` uses for W1, not a second one.

    `twin_parallel` (task 641, 2026-09-23, SE throughput): fires that many
    upsert batches concurrently, TWIN ONLY -- the build box is not RAM-limited
    (memory `aw-heavy-rewrite-degrades-prod` is specifically about the 30 GB
    prod box). The prod leg always stays sequential at `pause_s`, per the
    coordinator's ruling: a measured throttle only on prod.

    `prod_collection` (task 673 review round 3, finding 6/FLIP BLOCKER;
    made REQUIRED, no fallback, round 4 finding 3): the collection for the
    PROD leg, resolved by the caller through the live `aw_prod` alias --
    never `manifest["collection"]` (twin's own fixed name, unaffected by a
    flip; prod's alias target changes when one happens). No fallback: a
    caller that forgot to resolve the alias must fail loudly, not silently
    write the wrong, possibly-deleted collection.

    `prod_alias` (round 4 finding 2): re-resolved and compared against
    `prod_collection` immediately before the PROD write, while the
    production lock is held (see `_revalidate_prod_collection`).

    `supersede` (task 712): the plan of `url_group_supersede.plan_for_drain` -- the served rows
    of the pack's URL groups the pack does not carry, with their verified export. After each
    side's upsert those rows are deleted (verified gone), the export re-verified first. `None`
    (default) = this function exactly as before."""
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

    def apply_target(role: str, st: dict) -> None:
        if role == "prod":
            # Round 4 finding 2: re-confirm, under the lock, immediately
            # before this pack's prod write.
            _Q._revalidate_prod_collection(pack_id, prod_url, prod_alias,
                                        resolved_prod_collection)
            if supersede is not None:      # task 712: no prod write without the export intact
                UGS.reverify_before_write(supersede, twin_url=twin_url,
                                          twin_collection=twin_collection)
        url = twin_url if role == "twin" else prod_url
        collection = twin_collection if role == "twin" else resolved_prod_collection
        before = LAYOUT.fingerprint(url, collection)
        receipt, ids_path = _Q.apply_side(role, url, manifest, collection, batch=batch,
                                       pause_s=(0 if role == "twin" else pause_s),
                                       parallel=(twin_parallel if role == "twin" else 1),
                                       ids_dir=artifact_root,
                                       drop_ids=drop_ids,
                                       write_attempt=write_attempt)
        if supersede is not None:
            receipt["superseded"] = UGS.delete_superseded(
                role, url, collection, supersede, twin_url=twin_url,
                twin_collection=twin_collection, batch=batch)
            receipt["layout_after"] = LAYOUT.fingerprint(url, collection)
        st[f"{role}_before"] = before
        st[f"{role}_receipt"] = receipt
        st[f"{role}_ids_path"] = str(ids_path)

    def verify_target(role: str, st: dict) -> dict:
        receipt = st[f"{role}_receipt"]
        before = st[f"{role}_before"]
        after = receipt["layout_after"]
        written = receipt["written"]
        delta = (after.get("points_count") or 0) - (before.get("points_count") or 0)
        if delta > written:
            raise QueueError(
                f"{pack_id}/{role}: points_count grew by {delta:,} while this "
                f"run wrote {written:,}. Something else wrote inside the "
                "locked transaction; refusing before the next leg.")
        # url-group supersede: the lower bound is what THIS side's delete measured (prod: the
        # ids prod held, not the ids the twin held)
        if delta < -(receipt.get("superseded") or 0):
            raise QueueError(f"{pack_id}/{role}: points_count FELL by {-delta:,}")
        # #414 review follow-up 1: flip scripts take no writer lock, so the
        # alias can move after the write; re-resolve as the LAST prod check
        # before this verify returns (and the pack goes DONE).
        if role == "prod":
            _Q._revalidate_prod_collection(pack_id, prod_url, prod_alias,
                                        resolved_prod_collection)
        return {"role": role, "written": written, "points_delta": delta,
               "status": "PASS"}

    def post_twin_gate(st: dict, twin_proof: dict) -> dict:
        # The gate itself already ran, read-only, before any lock was taken
        # (the caller's dry_plan() + mechanical_content_status() check in
        # drain()) -- this callback exists only because twin_first_apply
        # requires one, and it re-states that PASS rather than re-deriving
        # it under the lock.
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
                "scope": ("narrow: one sealed pack, its rows appended and the rows of its URL "
                          "groups superseded (url-group), export-backed; no release is "
                          "published by this transaction" if supersede is not None else
                          "narrow: net-new append of one sealed pack; no "
                          "release is published by this transaction"),
                "pack_id": pack_id})
        if supersede is not None:
            UGS.mark_applied(artifact_root, supersede)
    finally:
        for name in reversed(taken):
            runb.release(name, task)
    return state


def _single_persisted_check() -> bool:
    """Single transaction knob (4): the ADR-105 persisted-input check runs only where an index
    exists. `AW_PERSISTED_INDEX_SQLITE_DIR` unset = the check is RECORDED as skipped in the plan
    receipt (`persisted_inputs.verdict == "SKIPPED"`), never as a pass."""
    return bool(os.environ.get("AW_PERSISTED_INDEX_SQLITE_DIR"))


def single_apply(*, pack_id: str, manifest: dict, single: Single, task: str, batch: int,
                 artifact_root: pathlib.Path,
                 drop_ids: frozenset[str] = frozenset(),
                 write_attempt: WriteAttempt | None = None) -> dict:
    """The `Single` leg of the drain: one target, no twin, ONE lock of its own, no ledger -- the same
    `apply_side` (plan, build_points, upsert) `live_apply` runs for each of its legs.

    ADR-110 check 5 reads "target == the pack's counts" here, where `live_apply` reads twin ==
    prod: (a) the target's `points_count` grew by exactly what this run wrote (nothing else writes
    into a collection a build owns, so `<` and `>` are both refusals); (b) what was written is
    exactly what the plan projected, less the rows the content check dropped; (c) when the pack
    had no skipped, excluded or dropped row at all, what was written is the manifest's `rows`."""
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    DRY.refuse("single apply")      # backstop: a dry run never reaches the write (#588)
    artifact_root.mkdir(parents=True, exist_ok=True)
    # ONE writer at a time per target, and the live-target check repeated under the lock, right
    # before the write (the alias may have moved since plan time).
    lock = f"SINGLE-WRITE-LOCK-{single.collection}"
    runb = _Q._load_run_batch()
    runb.acquire(lock, task, f"{pack_id}-single", pid=os.getpid())
    try:
        _Q.assert_single_target_not_live(single)
        before = LAYOUT.fingerprint(single.url, single.collection)
        receipt, ids_path = _Q.apply_side(
            "single", single.url, manifest, single.collection, batch=batch,
            pause_s=single.pause_s, parallel=single.parallel, ids_dir=artifact_root,
            drop_ids=drop_ids, write_attempt=write_attempt,
            persisted_check=_Q._single_persisted_check(),
            profile_collection=single.payload_profile_of)
    finally:
        runb.release(lock, task)
    after = receipt["layout_after"]
    written = receipt["written"]
    delta = (after.get("points_count") or 0) - (before.get("points_count") or 0)
    if delta != written:
        raise QueueError(
            f"{pack_id}/single: points_count moved by {delta:,} while this run wrote "
            f"{written:,}; a collection a build owns must grow by exactly what was written")
    rows = receipt.get("rows") or {}
    projected = rows.get("projected")
    if projected is not None and written != projected - len(drop_ids):
        raise QueueError(
            f"{pack_id}/single: wrote {written:,} but the plan projected {projected:,} "
            f"(less {len(drop_ids):,} dropped by the content check)")
    unaccounted = {k: rows.get(k, 0) for k in
                   ("skipped_by_projection", "excluded_undated", "excluded_out_of_scope_language")}
    # the most the drain writes is `rows_to_write_max` (an upper bound); a legacy pack has only `rows` and answers with it
    manifest_rows = MR.rows_to_write_max(manifest)
    if (manifest_rows is not None and not drop_ids and not any(unaccounted.values())
            and written != manifest_rows):
        raise QueueError(
            f"{pack_id}/single: wrote {written:,} but the manifest declares {manifest_rows:,} rows "
            "and nothing was skipped, excluded or dropped")
    return {"single_before": before, "single_receipt": receipt, "single_ids_path": str(ids_path),
            "check5": {"status": "PASS", "rule": "target == pack counts", "written": written,
                       "points_delta": delta, "manifest_rows": manifest_rows,
                       "dropped_by_content_check": len(drop_ids), **unaccounted}}


# ------------------------------------------------------ twin-only rehearsal
def rehearse(pack_dir: pathlib.Path, *, twin_url: str, task: str, batch: int,
            pause_s: float, artifact_root: pathlib.Path) -> dict:
    """Apply ONE pack to the TWIN ONLY, seal membership, roll back, re-verify.

    Never touches prod, never acquires the prod lock -- only TWIN-WRITE-LOCK,
    so it cannot collide with a live PROD-only transaction. Deliberately
    bypasses the gate-contract switch: its purpose is to prove the
    plumbing works, not to promote anything durable, and everything it
    writes is undone before it returns.
    """
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    runb = _Q._load_run_batch()
    manifest = read_manifest(pack_dir)
    assert_real_path(manifest)
    artifact_root.mkdir(parents=True, exist_ok=True)
    pack_id = manifest["pack_id"]

    runb.assert_locks_free(task)
    runb.acquire("TWIN-WRITE-LOCK", task, f"{pack_id}-rehearsal", pid=os.getpid())
    result: dict = {"pack_id": pack_id, "started_at": now()}
    try:
        before = LAYOUT.fingerprint(twin_url, manifest["collection"])
        points_before = before.get("points_count")
        receipt, ids_path = _Q.apply_side("twin", twin_url, manifest, manifest["collection"],
                                       batch=batch, pause_s=pause_s, ids_dir=artifact_root)
        result["apply"] = {"written": receipt["written"],
                           "write_seconds": receipt["write_seconds"]}
        ids = RB.load_written_ids(str(ids_path))
        result["present_after_write"] = RB.assert_all_present(
            twin_url, manifest["collection"], ids)
        RB.delete_ids(twin_url, manifest["collection"], ids)
        result["rollback"] = RB.assert_rolled_back(
            twin_url, manifest["collection"], ids, points_before)
        result["status"] = "REHEARSED-AND-ROLLED-BACK"
    finally:
        result["finished_at"] = now()
        runb.release("TWIN-WRITE-LOCK", task)
    return result
