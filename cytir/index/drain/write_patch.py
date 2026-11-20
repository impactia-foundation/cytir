"""Drain payload-patch write path: `live_apply_payload_patch` (the twin-first transaction around the
payload-only leg of `cytir.index.payload_patch`) and `_drain_payload_patch` (the per-pack driver the drain calls:
refusals, receipts, the change-log segment before DONE).

Task 2922 (RULING 80, drain split step 12): moved verbatim out of `cytir/index/promote_queue.py`, which
re-exports both names. A moved body that calls a name a test patches on the facade (or a name that
stays there) reads it at call time through `_Q` (the facade module).
"""
from __future__ import annotations

import os
import pathlib
import time

from cytir.index import payload_patch as PPK
from cytir.index import prod_guard as PG
from cytir.index import promote_sealed_pack as PSP
from cytir.index import twin_first as TFA
from cytir.index.drain.changelog import _log_landing
from cytir.index.drain.common import WriteAttempt, atomic_write_text, now
from cytir.index.drain.locks import PROD_LOCK_PATH
from cytir.index.drain.receipts import _write_receipt_never_masking


def live_apply_payload_patch(*, pack_id: str, manifest: dict, pack_dir: pathlib.Path,
                             twin_url: str, prod_url: str, task: str, batch: int, pause_s: float,
                             artifact_root: pathlib.Path, prod_collection: str,
                             write_attempt: WriteAttempt | None = None,
                             prod_alias: str = "aw_prod", twin_client=None, prod_client=None,
                             sleep=time.sleep) -> dict:
    """Task 2748: a `payload_patch` pack under the same `twin_first` protocol, locks and prod-alias
    revalidation as `live_apply_content_pack`; what differs is the leg (`payload_patch.apply_leg`:
    payload-only `set_payload`, conservation receipt, count delta 0). Idempotent, so a pack killed
    mid-leg is simply re-run by the next drain (every row already written reads `already_applied`).
    The inverse pack is written BEFORE the first write."""
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    runb = _Q._load_run_batch()
    artifact_root.mkdir(parents=True, exist_ok=True)
    ledger = artifact_root / f"{pack_id}-ledger.jsonl"
    lock = TFA.ProductionWriterLock(host=_Q.prod_lock_host(), lock_path=PROD_LOCK_PATH,
                                    purpose=f"task{task}-{pack_id}")
    runb.assert_locks_free(task)
    rows = PPK.load_rows(manifest, pack_dir)
    field = manifest["field"]
    twin_collection = manifest["collection"]
    clients = {"twin": twin_client or PPK.QdrantPatchClient(twin_url),
               "prod": prod_client or PPK.QdrantPatchClient(prod_url)}
    state: dict = {"rows": len(rows)}
    state["inverse_pack"] = PPK.write_inverse_pack(artifact_root, manifest, rows)

    def mark_started() -> None:
        if write_attempt is not None:
            write_attempt.started = True

    expect_missing = manifest.get("expect_missing", 0)

    def leg_pauses(role: str) -> tuple[float, float]:
        """(write pause, read pause): the prod leg takes the drain's big-pack floor and paces its
        read-only retrieves with the same pause; the twin reads unpaced."""
        if role == "prod":
            w = PG.write_pause("prod", len(rows), pause_s)
            return w, max(w, PG.PROD_BATCH_SLEEP_S)
        return pause_s, 0.0

    @PSP.retry_leg()
    def apply_target(role: str, st: dict) -> None:
        if role == "prod":
            _Q._revalidate_prod_collection(pack_id, prod_url, prod_alias, prod_collection)
        collection = twin_collection if role == "twin" else prod_collection
        w, r = leg_pauses(role)
        st[f"{role}_receipt"] = PPK.apply_leg(
            role, clients[role], collection, rows, field, batch=min(batch, PPK.MAX_BATCH),
            pause_s=w, read_pause_s=r, expect_missing=expect_missing,
            reference=st.get("twin_receipt") if role == "prod" else None,
            sleep=sleep, on_write=mark_started,
            # task 2777: the alias is read once before the classification (above) and once more after
            # it, right before the first write: a flip in between ABORTS with nothing written.
            before_write=(lambda: _Q._revalidate_prod_collection(pack_id, prod_url, prod_alias, prod_collection))
            if role == "prod" else None)

    def verify_target(role: str, st: dict) -> dict:
        collection = twin_collection if role == "twin" else prod_collection
        receipt = st[f"{role}_receipt"]
        PPK.verify_leg(clients[role], collection, rows, receipt, read_pause_s=leg_pauses(role)[1], sleep=sleep)
        if role == "prod":
            _Q._revalidate_prod_collection(pack_id, prod_url, prod_alias, prod_collection)
        PPK.write_leg_ids(artifact_root, pack_id, receipt)
        return {"role": role, "status": "PASS", **{k: receipt[k] for k in ("selected", *PPK.TERMS)},
                "verify_sample": receipt["verify_sample"],
                "points_delta": receipt["points_after"] - receipt["points_before"]}

    taken: list[str] = []
    try:
        for name in runb.LOCKS:
            runb.acquire(name, task, f"{pack_id}-promotion-queue", pid=os.getpid())
            taken.append(name)
        TFA.twin_first(
            ledger=ledger, lock=lock, preflight=lambda: state,
            apply_target=apply_target, verify_target=verify_target,
            post_twin_gate=lambda st, twin_proof: {"status": "PASS", "mechanical": True, "pack_id": pack_id},
            armed=True,
            preservation_verifier=lambda: {
                "status": "PASS",
                "scope": "narrow: payload-only set_payload of one allow-listed field; no ids added or "
                         "removed, no vector written; no release is published by this transaction",
                "pack_id": pack_id})
    finally:
        for name in reversed(taken):
            runb.release(name, task)
    return state


def _drain_payload_patch(*, pack_dir, manifest, record, gate, apply, twin_url, prod_url, task, batch,
                         pause_s, artifact_root, prod_collection, prod_alias, results) -> None:
    """One `payload_patch` pack of the drain loop (task 2748). Appends exactly one result, except when a
    failed receipt write in the plain-return refusal branch raises `RuntimeError` (no result, drain stops).
    A leg's pre-write classification (a moved row, missing > expect_missing, prod patching another id set than the twin) REFUSES the
    pack: ABORTED receipt, nothing DONE, the drain stops. `writes_issued` says whether ANY leg issued a write
    (`write_attempt.started`): false for a twin-leg refusal; true for a prod-leg refusal after a twin that wrote, false
    after a twin that already held the new values (it issued none). Any failure after a write also stops it.
    A no-write `ApplyAbort` (alias moved between classification and the first write) marks the pack REFUSED and STOPS the drain (task 2786); a flip after the twin wrote marks ABORTED and stops too."""
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    pack_id = pack_dir.name
    try:
        rows = PPK.load_rows(manifest, pack_dir)
    except (PPK.PayloadPatchError, OSError, ValueError) as exc:
        print(f"[{pack_id}] REFUSED (payload_patch): {exc}")
        record.update(status="REFUSED", stage="payload_patch_check", reason=f"{type(exc).__name__}: {exc}")
        _Q.write_pack_receipt(pack_dir, record)
        results.append(record)
        return
    record["payload_patch"] = {"field": manifest["field"], "rows": len(rows),
                               "version": manifest["payload_patch_version"]}
    record["gate"] = gate
    if gate["profile"] != "mechanical" or not apply:
        blocked = [] if gate["profile"] == "mechanical" else [
            f"gate: {gate['contract_path']} profile is {gate['profile']!r} "
            f"(flips to 'mechanical' once PR #{gate['pr']} merges)"]
        record["status"] = "PROVED-BLOCKED" if blocked else "PROVED-DRY-RUN"
        record["blocked_on"] = blocked
        print(f"[{pack_id}] {record['status']}: {'; '.join(blocked) if blocked else '--apply not set'}")
        _Q.write_pack_receipt(pack_dir, record)
        results.append(record)
        return
    write_attempt = _Q.WriteAttempt()
    try:
        state = _Q.live_apply_payload_patch(
            pack_id=pack_id, manifest=manifest, pack_dir=pack_dir, twin_url=twin_url,
            prod_url=prod_url, task=task, batch=batch, pause_s=pause_s, artifact_root=artifact_root,
            prod_collection=prod_collection(), write_attempt=write_attempt, prod_alias=prod_alias)
    except BaseException as exc:  # noqa: BLE001
        record["reason"] = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, (Exception, TFA.ApplyAbort)):
            # An interrupt (KeyboardInterrupt, SystemExit, ...; task 2800) is never a "pack REFUSED"
            # verdict. Only a pack that already wrote gets its ABORTED receipt, and that write cannot
            # mask the interrupt; with no write the pack stays READY with no receipt. It always propagates.
            if write_attempt.started:
                record.update(status="ABORTED", writes_issued=True)
                print(f"[{pack_id}] ABORTED (interrupted; drain stops; writes issued: True): {exc!r}")
                _write_receipt_never_masking(pack_dir, record, pack_id, "ABORTED")
                results.append(record)
            raise
        if write_attempt.started or isinstance(exc, PPK.PayloadPatchRefusal):
            # A write was issued, or a leg's pre-write classification refused the pack: the drain
            # stops either way. The next tick re-runs the idempotent pack and stops again until
            # someone acts (nothing is DONE).
            record["status"] = "ABORTED"
            record["writes_issued"] = write_attempt.started
            print(f"[{pack_id}] ABORTED (drain stops; writes issued: {write_attempt.started}): {exc}")
            _write_receipt_never_masking(pack_dir, record, pack_id, "ABORTED")
            results.append(record)
            raise
        record.update(status="REFUSED", stage="live-apply-preflight")
        print(f"[{pack_id}] REFUSED (payload_patch, no write issued): {exc}")
        if not _write_receipt_never_masking(pack_dir, record, pack_id, "REFUSED") \
                and not isinstance(exc, TFA.ApplyAbort):
            # A plain-return verdict with nothing on disk recording it: keep the pre-2817 loud stop
            # (the next pack must not land on a disk that cannot take a receipt).
            raise RuntimeError(f"[{pack_id}] REFUSED receipt write failed ({record['receipt_write_error']}); "
                               f"original reason: {record['reason']}") from exc
        if isinstance(exc, TFA.ApplyAbort):
            # The prod alias moved mid-drain (task 2786): the run-start collection is stale, so the
            # next pack would write its twin leg and only then abort. Stop the drain like the delete leg.
            results.append(record)
            raise
        results.append(record)
        return
    legs = {}
    for role in ("twin", "prod"):
        leg = dict(state[f"{role}_receipt"])
        leg["ids_artifact"] = str(artifact_root / f"{pack_id}-{role}-ids.json")
        legs[role] = {k: v for k, v in leg.items() if k != "ids"}
    record["status"] = "LANDED"
    record["twin"], record["prod"] = legs["twin"], legs["prod"]
    record["inverse_pack"] = state["inverse_pack"]
    # Task 2748 review: the change-log segment, BEFORE DONE (the weekly audit lists a DONE pack
    # without one). Every prod row now holding the new value is a `replace` (a payload-only change).
    prod_ids = state["prod_receipt"]["ids"]
    _log_landing(record, {"upserts": prod_ids["applied"] + prod_ids["already_applied"], "op": "replace"},
                 manifest_sha=record.get("manifest_sha"))
    atomic_write_text(pack_dir / "DONE", now() + "\n")
    _Q.write_pack_receipt(pack_dir, record)
    results.append(record)
