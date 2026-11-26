"""Drain: the hourly queue pass (`drain`) and the guard that settles an "already held" exit
(`_settle_guard`).

Task 2957 (RULING 80, drain split step 14): moved verbatim out of `cytir/index/promote_queue.py`, which
re-exports both names. A moved body that calls a name a test patches on the facade (or a facade-resident
constant, or the `TFA` alias a test replaces) reads it at call time through `_Q` (the facade module).

THREE REFUSAL POLICIES for a pack that cannot finish (task 3004; trace task 2912 finding F2). They are
a choice, not an accident. A persistent cause under policy 2 blocks every later pack until someone acts.
1. REFUSE AND CONTINUE: a sealed url-group pack whose re-plan fails (`REFUSED supersede_check`, or
   `live-apply-preflight`). The receipt is not cached, the pass goes on to the next pack, the pack is retried
   next hour. Visible as: the daily digest REFUSED list, and the monitor check `pack_deleting` when its state
   file stays at `deleting` for 6 h.
2. STOP THE DRAIN: a content pack whose prod-only resume fails (receipt ABORTED), and a payload patch whose
   pre-write check refuses (P1). Writing more packs on top of a half-applied one is not safe, so the pass ends
   and runs again next hour, "until someone acts". Visible as: the digest, and the 3 h "prod unchanged" watchdog.
3. STOP ON ABORT: `ApplyAbort` (a SystemExit: the alias moved mid-drain) and any failure after the shared
   `WriteAttempt` started stop the pass at once (`_settle_guard`, `stops = attempt.started or ...`).
A held writer lock (`cyt ops writer-locks --check`, monitor check `writer_locks`) makes every pack refuse at
`live-apply-preflight` while the unit exits 0: that is policy 1 for all packs together.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import socket
import time
import urllib.error

from cytir.index import change_log as CL
from cytir.index import dry_run as DRY
from cytir.index import german_parts_guard as GPG
from cytir.index import payload_patch as PPK
from cytir.index import prod_guard as PG
from cytir.index import promote_sealed_pack as PSP
from cytir.index import url_group_supersede as UGS
from cytir.index.drain.common import (
    READ_FAILURE_LOUD_AFTER,
    QueueError,
    Single,
    Transaction,
    TransientReadError,
    TwinFirst)
from cytir.index.drain.queue_scope import (
    GATE_CONTRACT_PATH,
    _dry_run_never_writes)


@_dry_run_never_writes
def drain(queue_dir: pathlib.Path, *, twin_url: str | None = None,
         prod_url: str | None = None, task: str,
         apply: bool, batch: int, pause_s: float, artifact_root: pathlib.Path,
         contract_path: pathlib.Path = GATE_CONTRACT_PATH,
         refresh_index: bool = True, full_persisted_index: bool = False,
         only: list[str] | None = None,
         twin_parallel: int = 1,
         budget_seconds: float | None = None,
         started_at_epoch: float | None = None,
         stop_margin_s: float = 5400.0,
         prod_alias: str = "aw_prod",
         transaction: Transaction | None = None) -> list[dict]:
    """Drain every READY pack in `queue_dir`, or, with `only`, exactly the
    named packs IN THE GIVEN ORDER (not resorted -- task 641, 2026-09-23:
    the coordinator asked for three news packs run in a specific sequence,
    ahead of and separate from whatever else sits in the queue, e.g. a
    from-scratch family whose input isn't safe to touch yet). A name in
    `only` that isn't a directory under `queue_dir` is a QueueError, not a
    silent skip -- an operator typo must not look like "ran everything
    requested".

    `budget_seconds` (task 673, review finding 5): the caller's own
    systemd `TimeoutStartSec` (or `TimeoutStartUSec`) budget -- so this
    drain never starts work it can't plausibly finish before a SIGKILL.
    `started_at_epoch` is when THAT budget's clock actually started
    (`time.time()`-scale; the wrapper script captures this before its own
    `git fetch`/`reset`, matching systemd's clock as closely as possible)
    -- default `None` means "now", i.e. this call's own start. The wall-
    clock deadline (`started_at_epoch + budget_seconds`) is translated
    into a `time.monotonic()` deadline exactly ONCE, here, so an NTP
    adjustment or a wall-clock jump mid-run can never move the goalposts.
    Three things are guarded, in order: (1) the persisted-input index
    refresh, which can itself take close to an hour -- skipped entirely,
    and the whole drain stops, if there isn't enough budget left to
    plausibly attempt it; (2) each new pack, using a duration ESTIMATE
    from its own op count (content packs only -- `rows` is a required
    manifest field, so this needs no file read) against
    `ESTIMATED_CONTENT_PACK_OPS_PER_SECOND` -- a pack that wouldn't
    plausibly fit is SKIPPED (stays READY, tried again next run), not
    refused; (3) the plain "is there any budget left at all" check before
    taking ANY pack, `stop_margin_s` of slack.

    This is what the resumable twin-write-state artifact (see
    `write_content_pack_twin_state`) is the safety net FOR, not a
    replacement for it: guarding (1)-(3) avoids most mid-pack kills, the
    artifact makes the ones that still happen (an estimate that runs low,
    a kill during this function's own non-pack work, a `-9` that skips
    even these checks) resumable instead of silent. `budget_seconds=None`
    (default): no deadline at all, unchanged behaviour for every existing
    caller and every test.

    `transaction` (slice 8 step 2): `None` (default) = `TwinFirst(twin_url, prod_url, prod_alias)`,
    this function's behaviour since task 641, byte for byte (pinned by
    `test_drain_transaction`). `Single(url, collection)` applies the same
    sealed packs to one target -- see `Single`. Content packs (supersede/tombstone deltas) are
    twin-first only: a `Single` drain REFUSES one, it does not guess."""
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    if transaction is None:
        if twin_url is None or prod_url is None:
            raise QueueError("drain: twin_url and prod_url are required unless transaction= is given")
        transaction = TwinFirst(twin_url, prod_url, prod_alias)
    elif isinstance(transaction, TwinFirst):
        if (twin_url, prod_url) not in ((None, None), (transaction.twin_url, transaction.prod_url)):
            raise QueueError("drain: twin_url/prod_url disagree with transaction=TwinFirst(...)")
        twin_url, prod_url, prod_alias = (transaction.twin_url, transaction.prod_url,
                                          transaction.prod_alias)
    elif isinstance(transaction, Single):
        if twin_url is not None or prod_url is not None:
            raise QueueError("drain: a Single transaction has no twin or prod url")
    else:
        raise QueueError(f"drain: unknown transaction {transaction!r}")
    single = transaction if isinstance(transaction, Single) else None
    single_state_root: pathlib.Path | None = None
    if single is not None:                # refuses BEFORE any read of the queue or write anywhere
        single_state_root = _Q.refuse_live_single(single, queue_dir, artifact_root)
    mono_deadline: float | None = None
    if budget_seconds is not None:
        wall_now = time.time()
        wall_deadline = (started_at_epoch if started_at_epoch is not None
                         else wall_now) + budget_seconds
        mono_deadline = time.monotonic() + (wall_deadline - wall_now)

    def remaining_s() -> float:
        return (mono_deadline - time.monotonic()
               if mono_deadline is not None else float("inf"))

    if only is not None:
        pack_dirs = []
        for name in only:
            p = queue_dir / name
            if not p.is_dir():
                raise QueueError(f"only={name!r}: no such pack dir under {queue_dir}")
            pack_dirs.append(p)
    else:
        # Family-grouped, resumable-and-largest-family-first (review round
        # 3 finding 1, superseding round 2's per-pack version) -- see
        # `order_pack_dirs_for_drain`. STRICT name order is preserved
        # WITHIN a family.
        pack_dirs = _Q.order_pack_dirs_for_drain(
            [p for p in queue_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_")],
            artifact_root)
    if single is not None:
        # The Single transaction READS the queue's manifests and writes ONLY to its own state dir
        # (F1): no copy of a pack dir is made (a dry run writes nothing, #588). A superseded pack
        # is not built.
        pack_dirs = [p for p in pack_dirs if not (p / "SUPERSEDED").exists()]
        _Q.refuse_state_entries(single_state_root, [p.name for p in pack_dirs], queue_dir)   # R1
    results: list[dict] = []
    if not pack_dirs:
        print(f"[queue] {queue_dir}: empty")
        return results

    if single is not None:
        print("[persisted-index] skipped: the single transaction does not refresh the corpus-wide "
              "persisted-input index (the ADR-105 check runs only where "
              "AW_PERSISTED_INDEX_SQLITE_DIR names one; the receipt says which)")
    if refresh_index and single is None:
        # Review finding 5: this step alone can take close to an hour (the
        # weekend-stall root cause -- log 2026-09-28 14:05Z). Guard it the
        # same as a pack: don't even attempt it without plausible room.
        if remaining_s() < stop_margin_s:
            print(f"[drain] stopping before the persisted-index refresh: "
                 f"only {remaining_s():.0f}s of budget left (< "
                 f"{stop_margin_s:.0f}s margin) -- nothing attempted this run")
            return results
        refreshed = _Q.refresh_persisted_index(full=full_persisted_index)
        if refreshed["rebuilt"]:
            print(f"[persisted-index] REBUILT in {refreshed['seconds']}s "
                 f"({refreshed['reason']})")
        else:
            print(f"[persisted-index] fresh, skipped rebuild "
                 f"({refreshed['reason']})")

    gate = _Q.gate_profile(contract_path)
    print(f"[gate] {gate['profile']} ({gate['adr']}, PR #{gate['pr']}, "
         f"{contract_path})")

    # Review round 3, finding 6/FLIP BLOCKER: resolved and threaded into
    # every prod-side write/read this run makes -- never `manifest
    # ["collection"]` for prod. `manifest["collection"]` (`aw_e8`) is a
    # FIXED, configured name every pack manifest declares; it is correct
    # for the TWIN (which stays `aw_e8` forever) but prod is served
    # through the `aw_prod` alias, which the `aw_e9` flip repoints to a
    # different physical collection -- prod's own `aw_e8` is deleted once
    # that happens. A drain that kept using `manifest["collection"]` for
    # prod would silently keep writing (or fail writing) against a
    # collection nothing serves the moment the flip lands.
    #
    # Round 4 finding 4: resolved LAZILY, not here unconditionally -- a
    # pure twin-only/dry-run drain (every pack REFUSED/PROVED-DRY-RUN/
    # PROVED-BLOCKED before ever touching prod) must not need a live prod
    # tunnel for nothing. `_prod_collection_cache` is a one-element cache:
    # the FIRST successful resolution, whenever/wherever it happens,
    # becomes this run's baseline and is reused for every pack after it;
    # each write function independently re-resolves and compares against
    # the value it's passed immediately before its own prod write, while
    # holding the production lock (round 4 finding 2,
    # `_revalidate_prod_collection`) -- this cache only avoids resolving
    # twice for the same still-current value. A prod outage surfaces
    # exactly where it's first needed: a read site already wrapped in the
    # existing connectivity-error handling gets a clean REFUSED (retries
    # next drain); a write site's failure propagates into the same mid-
    # write `except BaseException` handling as any other prod-write error
    # (ABORTED, drain stops).
    _prod_collection_cache: list[str] = []

    def get_prod_collection() -> str:
        if not _prod_collection_cache:
            resolved = _Q.resolve_prod_alias(prod_url, prod_alias)
            print(f"[drain] prod target: alias {prod_alias!r} -> collection "
                 f"{resolved!r} (twin stays each pack's own configured "
                 "manifest[\"collection\"])")
            _prod_collection_cache.append(resolved)
        return _prod_collection_cache[0]

    retired_markers: dict[str, int] = {}   # task 2286: why dirs were passed over without a result
    for pack_dir in pack_dirs:
        pack_id = pack_dir.name
        src_dir = pack_dir          # where the sealed pack's manifest.json / READY live
        if single is not None:
            # `pack_dir` is, from here on, the transaction's OWN state entry: receipts, the refusal
            # cache and DONE. The queue's pack dir is only read (`src_dir`).
            pack_dir = single_state_root / pack_id
            if not DRY.active():
                _Q.claim_state_entry(pack_dir)
        # NOTE (task 673 review round 3, finding 5): the budget check used
        # to live HERE, unconditionally, before even knowing whether this
        # pack is a content-pack resume -- which meant it always fired
        # first and pre-empted the more nuanced resume-aware logic below
        # (fits? BUDGET-EXCEEDS-TIMEOUT if it can never fit ANY run?
        # simply not enough left right now?). The generic version now
        # lives just below, scoped to non-content_pack (sealed) packs,
        # which have no per-pack size estimate to reason about; content
        # packs get their own full treatment once `manifest`/`twin_state`
        # are known.
        if (pack_dir / "DONE").exists():
            retired_markers["DONE"] = retired_markers.get("DONE", 0) + 1
            continue
        if (pack_dir / "SUPERSEDED").exists():
            # Retired via `supersede_pack()` -- proven wrong, never landed,
            # never retried (task 669). Distinct from DONE (landed on both
            # sides): this pack's rows are not live anywhere.
            retired_markers["SUPERSEDED"] = retired_markers.get("SUPERSEDED", 0) + 1
            continue
        if (pack_dir / "CARRIED-BY-FLIP").exists():
            # Retired via `mark_carried_by_flip()` (task 673) -- twin leg
            # is real, verified, applied content; prod leg is deliberately
            # withheld because the upcoming flip makes prod equal the twin
            # without a direct several-hundred-thousand-op write. Never
            # retried by this drain; a fresh manifest or an explicit
            # `--only` invocation is the only way to touch it again.
            retired_markers["CARRIED-BY-FLIP"] = retired_markers.get("CARRIED-BY-FLIP", 0) + 1
            continue

        manifest_path = src_dir / "manifest.json"
        manifest_sha = (hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                        if manifest_path.exists() else None)

        # A pack that REFUSED last run keeps its READY -- it is not dropped
        # from the queue -- but is skipped here, without re-running plan()/
        # mechanical-content/live-apply, as long as manifest.json's sha256
        # has not changed since that refusal AND the refusal's stage is one
        # of `_CACHEABLE_REFUSAL_STAGES` -- a deterministic content
        # judgment, not a transient/environmental failure. Task 660,
        # 2026-09-25: before this, a refused pack (e.g. CC tranche 1)
        # re-ran its full ~5-minute check EVERY hourly drain, for no new
        # information -- the only way to legitimately retry a genuine
        # content refusal is to change what was sealed (a restage), which
        # mints a new sha and clears this cache automatically.
        #
        # WHITELIST, not blacklist (task 662, 2026-09-25 -- see
        # `_CACHEABLE_REFUSAL_STAGES`'s comment for the full incident: task
        # 660's `stage != "intake"` blacklist let a lock-acquire failure
        # with no `stage` set fall through and get cached forever, same as
        # a content refusal). "intake" (no READY file / bad manifest
        # schema) and any pre-write live-apply preflight failure (lock
        # contention, an exhausted-retry API timeout, any other transient
        # cause) always retry -- only `plan` and `content_pack_check` are
        # cached.
        prior_receipt_path = pack_dir / "receipt.json"
        # The refusal is cached per (manifest, transaction, target): a receipt another transaction
        # or target wrote never skips this one. `None` = the twin-first drain, as it always was.
        transaction_key = _Q._single_transaction_key(single)
        if manifest_sha is not None and prior_receipt_path.exists():
            try:
                prior = json.loads(prior_receipt_path.read_text())
            except (OSError, json.JSONDecodeError):
                prior = None
            if (prior and prior.get("status") == "REFUSED"
                    and prior.get("stage") in _Q._CACHEABLE_REFUSAL_STAGES
                    and prior.get("manifest_sha") == manifest_sha
                    and prior.get("transaction_key") == transaction_key):
                print(f"[{pack_id}] SKIPPED -- REFUSED last run "
                     f"({prior.get('checked_at')}) at manifest sha "
                     f"{manifest_sha[:12]} (unchanged since); reseal with a "
                     "new manifest to retry")
                results.append({
                    "schema": _Q.SCHEMA, "pack_id": pack_id, "checked_at": _Q.now(),
                    "status": "SKIPPED-REFUSED-UNCHANGED",
                    "manifest_sha": manifest_sha,
                    "previous_reason": prior.get("reason")})
                continue

        record: dict = {"schema": _Q.SCHEMA, "pack_id": pack_id, "checked_at": _Q.now(),
                        "manifest_sha": manifest_sha}
        if single is not None:
            record["transaction_key"] = transaction_key
        try:
            manifest = _Q.read_manifest(src_dir)
            _Q.assert_real_path(manifest)
        except QueueError as exc:
            print(f"[{pack_id}] REFUSED at intake: {exc}")
            record.update(status="REFUSED", stage="intake", reason=str(exc))
            _Q.write_pack_receipt(pack_dir, record)
            results.append(record)
            continue

        record["family"] = manifest.get("family")

        if single is not None and manifest.get("kind") in ("content_pack", PPK.KIND):
            reason = ("a Single transaction applies sealed packs only; a content_pack carries "
                      "supersede/tombstone deltas, and a payload_patch rewrites rows in place, "
                      "both under the twin-first protocol (slice 8 step 2)")
            print(f"[{pack_id}] REFUSED (single transaction): {reason}")
            record.update(status="REFUSED", stage="single-transaction", reason=reason)
            _Q.write_pack_receipt(pack_dir, record)
            results.append(record)
            continue

        if manifest.get("kind") == PPK.KIND:
            _Q._drain_payload_patch(
                pack_dir=pack_dir, manifest=manifest, record=record, gate=gate, apply=apply,
                twin_url=twin_url, prod_url=prod_url, task=task, batch=batch, pause_s=pause_s,
                artifact_root=artifact_root, prod_collection=get_prod_collection,
                prod_alias=prod_alias, results=results)
            continue

        if manifest.get("kind") != "content_pack" and mono_deadline is not None:
            # Sealed packs have no cheap per-pack size estimate (no
            # required `rows` field) -- the same plain "is there any
            # budget left at all" guard the whole loop used to run
            # unconditionally, now scoped to just this kind (task 673
            # review round 3, finding 5).
            if remaining_s() < stop_margin_s:
                print(f"[drain] stopping before {pack_id}: only "
                     f"{remaining_s():.0f}s of budget left (< "
                     f"{stop_margin_s:.0f}s margin) -- remaining packs "
                     "stay READY for the next run")
                break

        # RESUME STATE FIRST (task 673), before any budget decision --
        # review round 2 finding 1: a pack with a pending resume must
        # NEVER be skipped by the budget estimate. Re-running
        # `content_pack_dry_check` on a resume would re-validate
        # tombstones against twin and either silently drop the ones a
        # completed/partial twin leg already deleted, or refuse the whole
        # pack outright (round 1 finding 2) -- so the ids to estimate from
        # are read here regardless of budget.
        twin_state = None
        if manifest.get("kind") == "content_pack":
            twin_state = _Q.read_content_pack_twin_state(
                artifact_root, pack_id, manifest_sha=manifest_sha)

        if manifest.get("kind") == "content_pack" and mono_deadline is not None:
            # Estimate BEFORE taking the pack, from its own declared op
            # count -- `rows` is a required content_pack manifest field,
            # so this needs no file read. A resume with stage
            # "twin-write-complete" only has the PROD leg left (1x); a
            # fresh pack or a "twin-write-started" replay still needs
            # both legs (2x).
            rows = manifest.get("rows") or 0
            tomb = manifest.get("tombstones_count") or 0
            if twin_state is not None:
                est_ops = (rows + tomb) if twin_state.get("stage") == "twin-write-complete" \
                    else 2 * (rows + tomb)
            else:
                est_ops = 2 * (rows + tomb)
            est_seconds = est_ops / _Q.ESTIMATED_CONTENT_PACK_OPS_PER_SECOND

            if twin_state is not None:
                if est_seconds > remaining_s() - stop_margin_s:
                    # Round 2 finding 1: refuse to skip a pending resume
                    # and move on -- a later pack (a recapture reusing
                    # these same ids, or a tombstone pack) could then land
                    # over a KNOWN, unresolved twin!=prod gap, rewriting
                    # rows this resume would have fixed or resurrecting
                    # ones it would have deleted. Stop the whole drain
                    # instead; the resumable state artifact is exactly
                    # what makes that safe to pick back up next run.
                    #
                    # Round 3 finding 5: if this resume can NEVER fit even
                    # a full, freshly-started budget window, "stop the
                    # drain" would otherwise repeat SILENTLY every single
                    # run -- just a print line nobody reviews. Persist a
                    # loud BUDGET-EXCEEDS-TIMEOUT receipt too (the
                    # watchdog/daily digest already surface that status
                    # for fresh packs, round 2 finding 2) before stopping,
                    # same as the fresh-pack case below.
                    if (budget_seconds is not None
                            and est_seconds > budget_seconds - stop_margin_s):
                        print(f"[{pack_id}] BUDGET-EXCEEDS-TIMEOUT (pending "
                             f"resume): estimated {est_seconds:.0f}s "
                             f"({est_ops} ops @ "
                             f"{_Q.ESTIMATED_CONTENT_PACK_OPS_PER_SECOND:.0f}/s) "
                             f"can never fit a {budget_seconds:.0f}s budget "
                             f"(- {stop_margin_s:.0f}s margin) -- this is a "
                             "KNOWN twin!=prod gap that no amount of "
                             "waiting will close; needs a manual resume "
                             "(smaller batch) or operator attention")
                        record.update(status="BUDGET-EXCEEDS-TIMEOUT",
                                      estimated_seconds=est_seconds,
                                      budget_seconds=budget_seconds,
                                      resume_stage=twin_state.get("stage"))
                        _Q.write_pack_receipt(pack_dir, record)
                        results.append(record)
                    print(f"[drain] stopping before {pack_id}: it has a "
                         f"pending resume (stage {twin_state.get('stage')!r}) "
                         f"estimated at {est_seconds:.0f}s, which does not "
                         f"fit in {remaining_s():.0f}s remaining (- "
                         f"{stop_margin_s:.0f}s margin) -- refusing to skip "
                         "a known twin!=prod gap and move on; stopping "
                         "the whole drain instead")
                    break
                # else: fits -- fall through to the resume branch below.
            else:
                if (budget_seconds is not None
                        and est_seconds > budget_seconds - stop_margin_s):
                    # A pack this big can never fit even a FULL, freshly-
                    # started budget window -- SKIPPED-BUDGET every single
                    # run forever is a silent, permanent failure. Say so
                    # loudly and persist a receipt the watchdog/daily
                    # digest can find (round 2 finding 2).
                    print(f"[{pack_id}] BUDGET-EXCEEDS-TIMEOUT: estimated "
                         f"{est_seconds:.0f}s ({est_ops} ops @ "
                         f"{_Q.ESTIMATED_CONTENT_PACK_OPS_PER_SECOND:.0f}/s) "
                         f"can never fit a {budget_seconds:.0f}s budget (- "
                         f"{stop_margin_s:.0f}s margin) -- needs a smaller "
                         "pack (reseal/split it), not another run")
                    record.update(status="BUDGET-EXCEEDS-TIMEOUT",
                                  estimated_seconds=est_seconds,
                                  budget_seconds=budget_seconds)
                    _Q.write_pack_receipt(pack_dir, record)
                    results.append(record)
                    continue
                if est_seconds > remaining_s() - stop_margin_s:
                    print(f"[{pack_id}] SKIPPED -- estimated {est_seconds:.0f}s "
                         f"({est_ops} ops @ {_Q.ESTIMATED_CONTENT_PACK_OPS_PER_SECOND:.0f}"
                         f"/s) would not fit in {remaining_s():.0f}s remaining "
                         f"(- {stop_margin_s:.0f}s margin); stays READY")
                    results.append({
                        "schema": _Q.SCHEMA, "pack_id": pack_id, "checked_at": _Q.now(),
                        "status": "SKIPPED-BUDGET", "manifest_sha": manifest_sha,
                        "estimated_seconds": est_seconds})
                    continue

        if manifest.get("kind") == "content_pack":
            # ADR-112/#310 shape: packs declare their own deletes. Its own
            # code path -- no sealed-pack plan()/mechanical-content-v1 (the
            # producing lane already resolved identity/licence/embedding;
            # see the module docstring at `apply_content_pack_side`).
            #
            # RESUME FIRST (task 673): if a prior cycle's twin leg touched
            # this EXACT manifest and never reached a DONE receipt (the
            # weekend-stall SIGKILL shape that orphaned
            # l1-backfill-batch-00001), replay from the recorded ids
            # instead of re-deriving anything -- re-running
            # `content_pack_dry_check` here would re-validate tombstones
            # against twin and either silently drop the ones a completed
            # or partial twin leg already deleted, or refuse the whole
            # pack outright (review finding 2). `twin_state` was already
            # read above, before the budget decision.
            if twin_state is not None:
                record["gate"] = gate
                stage = twin_state.get("stage")
                # Review finding 1: a resume is still a WRITE and must obey
                # --apply and the gate profile exactly like a normal land --
                # the prior version of this branch called the resume
                # unconditionally, so a dry run (or a battery-required gate)
                # would write prod and mark DONE.
                if gate["profile"] != "mechanical" or not apply:
                    blocked = [] if gate["profile"] == "mechanical" else [
                        f"gate: {gate['contract_path']} profile is "
                        f"{gate['profile']!r} (flips to 'mechanical' once "
                        f"PR #{gate['pr']} merges)"]
                    record["status"] = "PROVED-BLOCKED" if blocked else "PROVED-DRY-RUN"
                    record["blocked_on"] = blocked
                    record["note"] = (f"resume pending: twin leg recorded "
                                      f"{stage!r} for this manifest at "
                                      f"{twin_state.get('at')}")
                    print(f"[{pack_id}] {record['status']} (resume pending): "
                         f"{'; '.join(blocked) if blocked else '--apply not set'}")
                    _Q.write_pack_receipt(pack_dir, record)
                    results.append(record)
                    continue
                print(f"[{pack_id}] RESUMING (content_pack, stage={stage!r}): "
                     f"twin leg recorded at {twin_state.get('at')} for this "
                     f"manifest -- {len(twin_state.get('pack_ids', []))} pack "
                     f"id(s), {len(twin_state.get('tomb_ids', []))} tombstone "
                     "id(s)")
                write_attempt = _Q.WriteAttempt()
                record.update(_Q.read_held_out_receipt_fields(pack_dir))     # S2: the count survives a resume
                try:
                    if stage == "twin-write-complete":
                        state = _Q._retry_resume(
                            lambda: _Q.resume_content_pack_prod_only(
                                pack_id=pack_id, manifest=manifest, twin_url=twin_url,
                                prod_url=prod_url, task=task, batch=batch,
                                pause_s=pause_s, artifact_root=artifact_root,
                                state_record=twin_state, write_attempt=write_attempt,
                                prod_collection=get_prod_collection(),
                                prod_alias=prod_alias),
                            pack_id=pack_id)
                        record["prod"] = state.get("prod_receipt", {})
                        record["note"] = ("resumed prod-only from a "
                                         "twin-write-complete state record")
                    else:
                        # "twin-write-started": a kill landed somewhere
                        # INSIDE the twin leg (upsert, delete, or the state
                        # write itself) -- replay the FULL twin_first
                        # protocol from the recorded ids. Both
                        # apply_content_pack_side's upsert and
                        # delete_points_verified are idempotent (an
                        # already-present id upserts to the same content;
                        # an already-absent id's delete is a same-result
                        # no-op), so replaying is always safe, never a
                        # double-apply.
                        points, tombstones = _Q.load_content_pack_ids_from_state(
                            manifest, twin_state)
                        # task 2583 (Opus 2506 A, 2584): this replay skips the dry check; map every state
                        # tombstone reason BEFORE any write. A miss ABORTS through the handler below (a
                        # REFUSED receipt would be cached and strand the half-written pack).
                        for t in tombstones:
                            CL.public_reason(t.get("reason"))
                        state = _Q._retry_resume(
                            lambda: _Q.live_apply_content_pack(
                                pack_id=pack_id, manifest=manifest, twin_url=twin_url,
                                prod_url=prod_url, task=task, batch=batch,
                                pause_s=pause_s, artifact_root=artifact_root,
                                points=points, tombstones=tombstones,
                                write_attempt=write_attempt, manifest_sha=manifest_sha,
                                prod_collection=get_prod_collection(),
                                prod_alias=prod_alias),
                            pack_id=pack_id)
                        record["twin"] = state.get("twin_receipt", {})
                        record["prod"] = state.get("prod_receipt", {})
                        record["note"] = ("replayed the twin leg from a "
                                         "twin-write-started record, then "
                                         "landed prod")
                    record["status"] = "LANDED"
                    if CL.configured_dir() is not None:
                        # the state record names the ids the prod leg was sent (twin-first: prod == twin)
                        r_points, r_tombs = _Q.load_content_pack_ids_from_state(manifest, twin_state)
                        _Q._log_resumed_landing(record, [x["id"] for x in r_points], manifest=manifest,
                                             manifest_sha=manifest_sha, tombstones=r_tombs, op="upsert")
                    _Q.atomic_write_text(pack_dir / "DONE", _Q.now() + "\n")
                except BaseException as exc:  # noqa: BLE001
                    # Review finding 4: ANY resume failure stops the drain,
                    # unconditionally -- unlike a fresh pack's refusal
                    # (where "nothing written yet" really does mean safe to
                    # skip), a resume's failure means a KNOWN twin!=prod
                    # gap for this pack already exists and this run could
                    # not close it; continuing to other packs risks
                    # compounding or hiding that gap rather than surfacing
                    # it for a human.
                    record["reason"] = f"{type(exc).__name__}: {exc}"
                    record["status"] = "ABORTED"
                    print(f"[{pack_id}] ABORTED (resume failed, drain "
                         f"stops): {exc}")
                    _Q.write_pack_receipt(pack_dir, record)
                    results.append(record)
                    raise
                _Q.write_pack_receipt(pack_dir, record)
                results.append(record)
                continue
            try:
                check = _Q.content_pack_dry_check(manifest, twin_url, contract_path)
            except TransientReadError as exc:
                streak = _Q.read_failure_streak(pack_dir, manifest_sha)
                print(f"[drain] {pack_id}: twin read failed in the dry check, retried next "
                     f"run (failure {streak} in a row): {exc}")
                if streak >= READ_FAILURE_LOUD_AFTER:
                    print(f"[drain] LOUD: {pack_id} has failed its twin read {streak} runs "
                         "in a row (not a content refusal; check the twin)")
                record.update(status="REFUSED", stage="content_pack_read",
                             reason=str(exc), consecutive_read_failures=streak)
                _Q.write_pack_receipt(pack_dir, record)
                results.append(record)
                continue
            except (QueueError, *_Q.REFUSAL_EXCEPTIONS) as exc:
                print(f"[{pack_id}] REFUSED (content_pack): "
                     f"{type(exc).__name__}: {exc}")
                record.update(status="REFUSED", stage="content_pack_check",
                             reason=f"{type(exc).__name__}: {exc}")
                if not DRY.active():
                    _Q.drop_held_out_cards(pack_dir)
                _Q.write_pack_receipt(pack_dir, record)
                results.append(record)
                continue
            points = check.pop("_points_data")
            tombstones = check.pop("_tombstones_data")
            held_cards = check.pop("_held_out_cards", {"rows": [], "tombstone_ids": []})
            record.update(_Q.held_out_receipt_fields(held_cards))
            record["content_pack"] = check
            record["german_parts"], gp_reason = GPG.guarded(GPG.check_points, pack_id, points)
            if gp_reason:
                # NOT in `_CACHEABLE_REFUSAL_STAGES`: the verdict is a fact about the writer
                # interpreter of THIS run, not about the pack; it retries every drain.
                record.update(status="REFUSED", stage="german_parts", reason=gp_reason)
                _Q.write_pack_receipt(pack_dir, record)
                results.append(record)
                continue
            record["gate"] = gate
            if gate["profile"] != "mechanical" or not apply:
                blocked = [] if gate["profile"] == "mechanical" else [
                    f"gate: {gate['contract_path']} profile is "
                    f"{gate['profile']!r} (flips to 'mechanical' once PR "
                    f"#{gate['pr']} merges)"]
                record["status"] = "PROVED-BLOCKED" if blocked else "PROVED-DRY-RUN"
                record["blocked_on"] = blocked
                print(f"[{pack_id}] {record['status']}: "
                     f"{'; '.join(blocked) if blocked else '--apply not set'}")
                _Q.write_pack_receipt(pack_dir, record)
                results.append(record)
                continue
            write_attempt = _Q.WriteAttempt()
            try:
                # S1: inside the try -- a disk error is a retried preflight refusal, not a dead run
                if held_cards["rows"]:
                    _Q.write_held_out_cards(pack_dir, held_cards)
                # Task 675: refuse BEFORE the twin leg if any row cannot be
                # written in either target's declared payload form. Inside this
                # try on purpose: `write_attempt.started` is still False, so a
                # refusal lands as REFUSED / live-apply-preflight (retried next
                # drain, never cached), exactly like any other pre-write refusal.
                record["consent_flags"] = _Q.consent_flag_census(
                    p.get("payload") or {} for p in points)
                record["payload_profile"] = _Q.payload_profile_preflight_both(
                    manifest["collection"], get_prod_collection(),
                    [p.get("payload") or {} for p in points])
                state = _Q.live_apply_content_pack(
                    pack_id=pack_id, manifest=manifest, twin_url=twin_url,
                    prod_url=prod_url, task=task, batch=batch, pause_s=pause_s,
                    artifact_root=artifact_root, points=points,
                    tombstones=tombstones, write_attempt=write_attempt,
                    manifest_sha=manifest_sha, prod_collection=get_prod_collection(),
                    prod_alias=prod_alias)
                record["status"] = "LANDED"
                record["twin"] = state.get("twin_receipt", {})
                record["prod"] = state.get("prod_receipt", {})
                _Q._log_landing(record, {"upserts": [x["id"] for x in points], "op": "upsert",
                                      "tombstones": [{"id": t["id"], "reason": t.get("reason")}
                                                     for t in tombstones]},
                             manifest_sha=manifest_sha)
                _Q.atomic_write_text(pack_dir / "DONE", _Q.now() + "\n")
            except BaseException as exc:  # noqa: BLE001
                record["reason"] = f"{type(exc).__name__}: {exc}"
                if write_attempt.started:
                    # A write was already issued (twin and/or prod) -- this
                    # is a possible partial-write divergence, not a clean
                    # refusal. Stop the whole drain rather than risk landing
                    # more packs on top of an unverified state (task 660).
                    record["status"] = "ABORTED"
                    print(f"[{pack_id}] ABORTED (mid-write, drain stops): {exc}")
                    _Q.write_pack_receipt(pack_dir, record)
                    results.append(record)
                    raise
                # Nothing was written yet -- could be a clean re-derived
                # content refusal (e.g. the payload-carry population check
                # re-run at write time, task 660's head-of-line-blocking
                # fix) OR a transient one (lock contention, an
                # exhausted-retry API timeout). Either way this call site
                # cannot tell which, so it is NOT in
                # `_CACHEABLE_REFUSAL_STAGES` -- stage
                # "live-apply-preflight" always retries next drain (task
                # 662, 2026-09-25: a lock-acquire failure here used to get
                # cached forever as `stage=None`, same treatment as a real
                # content REFUSE).
                record["status"] = "REFUSED"
                record["stage"] = "live-apply-preflight"
                print(f"[{pack_id}] REFUSED (content_pack, no write issued): {exc}")
                if not DRY.active():
                    _Q.drop_held_out_cards(pack_dir)
                _Q.write_pack_receipt(pack_dir, record)
                results.append(record)
                continue
            _Q.write_pack_receipt(pack_dir, record)
            results.append(record)
            continue

        try:
            if single is None:
                plan = _Q.dry_plan(manifest, twin_url)
            else:
                plan = _Q.dry_plan(manifest, single.url, collection=single.collection,
                                persisted_check=_Q._single_persisted_check())
        except PSP.AlreadyLandedError as exc:
            if single is not None:
                # No twin to reconcile against: the target either holds every id (this pack
                # landed already -- a rerun after a kill between the write and DONE) or it holds a
                # part of them (a pack killed mid-write). The first is DONE; the second is a
                # half-built collection, which this transaction does not guess about.
                if set(exc.present) == set(exc.ids) and not apply:
                    record.update(status="PROVED-DRY-RUN", blocked_on=[],
                                  would="mark DONE: the target already holds every id")
                    record["transaction"] = _Q._single_record(single)
                    print(f"[{pack_id}] PROVED-DRY-RUN: would mark DONE (the target already "
                          "holds every id; --apply not set)")
                elif set(exc.present) == set(exc.ids):
                    record["status"] = "LANDED"
                    record["note"] = ("the target already held every id -- marking DONE, no "
                                      "write issued")
                    record["transaction"] = _Q._single_record(single)
                    _Q.atomic_write_text(pack_dir / "DONE", _Q.now() + "\n")
                    print(f"[{pack_id}] DONE (already fully landed on the target, no write issued)")
                else:
                    reason = (f"{len(exc.present)} of {len(exc.ids)} ids are already on "
                              f"{single.collection} -- a partly written pack; rebuild the "
                              "collection (--recreate) rather than append to it")
                    print(f"[{pack_id}] REFUSED: {reason}")
                    record.update(status="REFUSED", stage="live-apply-preflight", reason=reason)
                _Q.write_pack_receipt(pack_dir, record)
                results.append(record)
                continue
            # Task 664, 2026-09-26 incident: twin already has every one of
            # this pack's ids -- NOT necessarily "nothing left to do". If a
            # prior run's TWIN leg landed and verified but its PROD leg
            # then failed (a mid-write qdrant restart, a dead --prod-url,
            # any partial-write divergence), the old code treated "already
            # in twin" as a permanent content refusal: cached forever by
            # the "plan"-stage skip-cache, so the twin/prod gap could never
            # close by itself (measured live: tranche2-a and tranche2-d
            # both stuck this way, 385,800 points of twin/prod drift
            # between them). Check PROD membership on the SAME id set
            # before deciding what this is.
            try:
                prod_missing = sorted(
                    exc.ids - set(PSP.already_present(
                        prod_url, get_prod_collection(), sorted(exc.ids))))
            except (QueueError, *_Q.REFUSAL_EXCEPTIONS, TimeoutError,
                   socket.timeout, urllib.error.URLError, PG.ProdReadRefused) as prod_exc:
                # #414 review follow-up 4: `get_prod_collection()` above
                # raises QueueError when the `aw_prod` alias is missing
                # (mid-flip window); that is the same retry-next-drain
                # shape as a prod outage, not a crash of the whole drain.
                # NOT stage="plan" -- a prod connectivity hiccup while
                # deciding whether this is a resume is exactly the
                # "prod is fragile right now" shape this recovery exists
                # for, and caching it as a deterministic content refusal
                # would recreate the identical bug on a different exception
                # type. Always retries next drain (task 662's
                # `_CACHEABLE_REFUSAL_STAGES` whitelist).
                print(f"[{pack_id}] could not check prod membership during "
                     f"an already-landed resume check (retrying next "
                     f"drain): {type(prod_exc).__name__}: {prod_exc}")
                record.update(status="REFUSED", stage="live-apply-preflight",
                             reason=f"{type(prod_exc).__name__}: {prod_exc}")
                _Q.write_pack_receipt(pack_dir, record)
                results.append(record)
                continue
            if not prod_missing and not _Q.twin_set_matches_prod_pack_ids(
                    exc.ids, prod_missing, _Q.twin_written_ids_path(artifact_root, pack_id)):
                # Task 808 (review 797 S4 / 811 S1): prod holds pack ids the twin leg dropped. Refuse
                # HERE, with the same ResumeCollisionError the resume path raises, instead of falling
                # through to `resume_prod_only` (writer lock + a second full prod lookup every drain).
                # Nothing written; stage is not cached, so the next drain retries. (A replay cannot
                # reach this with a real divergence: in a replay "prod" is the twin.)
                exc3 = PSP.ResumeCollisionError(
                    _Q.outside_twin_set_reason(exc.ids, prod_missing,
                                            _Q.twin_written_ids_path(artifact_root, pack_id)),
                    ids=exc.ids, present=exc.ids)
                record.update(status="REFUSED", stage="live-apply-preflight",
                              reason=f"{type(exc3).__name__}: {exc3}")
                print(f"[{pack_id}] REFUSED (prod holds ids the twin leg did not write, no write "
                      f"issued): {exc3}")
                _Q.write_pack_receipt(pack_dir, record)
                results.append(record)
                continue
            if not prod_missing and not apply:
                if not _settle_guard(
                        record, pack_dir, results, apply=False, manifest=manifest, manifest_sha=manifest_sha,
                        artifact_root=artifact_root, twin_url=twin_url, prod_url=prod_url,
                        prod_collection=get_prod_collection(), prod_alias=prod_alias, batch=batch, task=task):
                    continue
                would_more = _Q._would_settle_text(record)
                record.update(status="PROVED-DRY-RUN", blocked_on=[],
                              would="mark DONE: twin and prod already hold every id" + would_more)
                print(f"[{pack_id}] PROVED-DRY-RUN: would mark DONE{would_more} (twin and prod "
                     "already hold every id; --apply not set)")
                _Q.write_pack_receipt(pack_dir, record)
                results.append(record)
                continue
            if not prod_missing:
                # Both legs already have every id (e.g. a receipt was lost
                # or overwritten after a genuinely completed pack) -- mark
                # DONE rather than refuse forever with nothing left to fix.
                record["status"] = "LANDED"
                record["note"] = ("twin and prod both already had every id "
                                  "-- marking DONE, no write issued")
                try:
                    if not _settle_guard(
                            record, pack_dir, results, apply=True, manifest=manifest, manifest_sha=manifest_sha,
                            artifact_root=artifact_root, twin_url=twin_url, prod_url=prod_url,
                            prod_collection=get_prod_collection(), prod_alias=prod_alias, batch=batch, task=task):
                        continue
                    _tombs = _Q._supersede_tombstones_on_resume(manifest, manifest_sha, artifact_root)
                    _Q._log_resumed_landing(record, exc.ids, manifest=manifest, manifest_sha=manifest_sha,
                                         tombstones=_tombs)
                except (CL.ChangeLogError, OSError) as log_exc:   # no DONE without its segment; retry next drain
                    record.update(status="REFUSED", stage="change-log",
                                  reason=f"{type(log_exc).__name__}: {log_exc}")
                    print(f"[{pack_id}] REFUSED (change-log / supersede state, no DONE): {log_exc}")
                    _Q.write_pack_receipt(pack_dir, record)
                    results.append(record)
                    continue
                _Q.atomic_write_text(pack_dir / "DONE", _Q.now() + "\n")
                _Q.write_pack_receipt(pack_dir, record)
                results.append(record)
                print(f"[{pack_id}] DONE (already fully landed on both "
                     "twin and prod, no write issued)")
                continue
            # Task 788: every id prod lacks is one the twin leg's gate dropped (the pack's own
            # artifact says so) -- the pack landed; say that instead of "resume N ids".
            landed = _Q.landed_with_twin_drops(
                exc.ids, prod_missing, _Q.twin_written_ids_path(artifact_root, pack_id))
            if landed is not None:
                record["landed_with_twin_drops"] = landed
                if not apply:
                    if not _settle_guard(
                            record, pack_dir, results, apply=False, manifest=manifest, manifest_sha=manifest_sha,
                            artifact_root=artifact_root, twin_url=twin_url, prod_url=prod_url,
                            prod_collection=get_prod_collection(), prod_alias=prod_alias, batch=batch, task=task):
                        continue
                    would_more = _Q._would_settle_text(record)
                    record.update(status="PROVED-DRY-RUN", blocked_on=[],
                                  would=f"mark DONE: {landed['note']}{would_more}")
                    print(f"[{pack_id}] PROVED-DRY-RUN: would mark DONE{would_more} ({landed['note']}; "
                          "--apply not set)")
                else:
                    record["status"] = "LANDED"
                    record["note"] = f"{landed['note']} -- marking DONE, no write issued"
                    try:
                        if not _settle_guard(
                                record, pack_dir, results, apply=True, manifest=manifest, manifest_sha=manifest_sha,
                                artifact_root=artifact_root, twin_url=twin_url, prod_url=prod_url,
                                prod_collection=get_prod_collection(), prod_alias=prod_alias, batch=batch, task=task):
                            continue
                        _tombs = _Q._supersede_tombstones_on_resume(manifest, manifest_sha, artifact_root)
                        _Q._log_resumed_landing(record, set(exc.ids) - set(prod_missing), manifest=manifest,
                                             manifest_sha=manifest_sha, tombstones=_tombs)
                    except (CL.ChangeLogError, OSError) as log_exc:   # no DONE without its segment
                        record.update(status="REFUSED", stage="change-log",
                                      reason=f"{type(log_exc).__name__}: {log_exc}")
                        print(f"[{pack_id}] REFUSED (change-log / supersede state, no DONE): {log_exc}")
                        _Q.write_pack_receipt(pack_dir, record)
                        results.append(record)
                        continue
                    _Q.atomic_write_text(pack_dir / "DONE", _Q.now() + "\n")
                    print(f"[{pack_id}] DONE ({landed['note']}; no write issued)")
                _Q.write_pack_receipt(pack_dir, record)
                results.append(record)
                continue
            # Genuine resume candidate: twin has SOME of this pack's ids
            # that prod does not (yet). Twin is NEVER re-written here --
            # only the prod leg runs, via `resume_prod_only()`. Which twin-
            # write it resumes FROM depends on how twin came to have them:
            #
            # * A per-pack artifact (`{pack_id}-twin-written-ids.json`)
            #   means THIS pack_id's own apply wrote twin and then its prod
            #   leg failed (the original 2026-09-26 incident, task 664) --
            #   `resume_prod_only()` reads it, unchanged.
            # * NO such artifact (task 664, 2026-09-26, incident 2 --
            #   `cc-authority-frontier-w7-tranche2`): twin has these ids
            #   because a DIFFERENT pack_id's apply wrote them (here,
            #   tranche2-a..d's earlier split-pack landing) -- there was
            #   never a twin WRITE under THIS pack_id to leave an artifact
            #   behind. Derive the same answer live instead of refusing: a
            #   fresh, direct twin membership check over exactly the ids
            #   `prod_missing` names, restricted to what is CONFIRMED on
            #   twin right now. If any of `prod_missing` is on NEITHER twin
            #   nor prod, that is NOT a resume -- it is genuinely new
            #   content this pack alone introduces, and writing it straight
            #   to prod without ever landing it on twin first would break
            #   the twin-mirror rule; refuse clearly instead of guessing or
            #   silently dropping it.
            twin_ids_path = _Q.twin_written_ids_path(artifact_root, pack_id)
            twin_ids_override = None
            row_check_ids = None
            if not twin_ids_path.exists():
                try:
                    twin_confirmed = set(PSP.already_present(
                        twin_url, manifest["collection"], prod_missing))
                except (*_Q.REFUSAL_EXCEPTIONS, TimeoutError, socket.timeout,
                       urllib.error.URLError) as twin_exc:
                    print(f"[{pack_id}] could not check twin membership "
                         "during a no-artifact resume derivation (retrying "
                         f"next drain): {type(twin_exc).__name__}: {twin_exc}")
                    record.update(status="REFUSED",
                                 stage="live-apply-preflight",
                                 reason=f"{type(twin_exc).__name__}: {twin_exc}")
                    _Q.write_pack_receipt(pack_dir, record)
                    results.append(record)
                    continue
                never_written = sorted(set(prod_missing) - twin_confirmed)
                if never_written:
                    reason = (
                        f"{pack_id}: {len(never_written)} of "
                        f"{len(prod_missing)} id(s) missing from prod are "
                        "on NEITHER twin nor prod, and no per-pack twin-"
                        "write artifact exists (e.g. "
                        f"{never_written[:3]}) -- not a resume; this pack "
                        "introduces genuinely new content that needs a "
                        "normal twin-first write, or is a duplicate stage "
                        "that should be excluded at compose time instead.")
                    reason += _Q._already_landed_overlap_note(
                        exc, prod_missing, never_written)
                    print(f"[{pack_id}] REFUSED: {reason}")
                    record.update(status="REFUSED", stage="live-apply-preflight",
                                 reason=reason)
                    pre_landed = sorted(getattr(exc, "present", ()) or ())
                    if pre_landed:
                        # Task 690: the ids that TRIGGERED this branch -- the
                        # refusal text above names the ids it is NOT about.
                        record["already_on_twin"] = {
                            "count": len(pre_landed),
                            "also_on_prod": len(set(pre_landed)
                                                - set(prod_missing)),
                            "ids": pre_landed[:50]}
                    _Q.write_pack_receipt(pack_dir, record)
                    results.append(record)
                    continue
                # Review 2900 finding 2: a prod leg that died part-way left some of THIS pack's ids on
                # prod. They are twin-written ids too, when the twin holds them: confirm that live, so
                # the resume skips them. One the twin does NOT hold stays out of the set and plan()
                # refuses it as a collision (prod holds an id the twin leg did not write).
                on_prod_already = sorted(set(exc.ids) - set(prod_missing))
                # An id is not a row: the ones accepted HERE are compared row by row (prod = twin =
                # the pack's row) before any write; a difference refuses as a collision.
                if on_prod_already:
                    try:
                        twin_confirmed |= set(PSP.already_present(
                            twin_url, manifest["collection"], on_prod_already))
                    except (*_Q.REFUSAL_EXCEPTIONS, TimeoutError, socket.timeout,
                           urllib.error.URLError) as twin_exc:
                        print(f"[{pack_id}] could not check twin membership of the ids already on "
                             f"prod (retrying next drain): {type(twin_exc).__name__}: {twin_exc}")
                        record.update(status="REFUSED", stage="live-apply-preflight",
                                     reason=f"{type(twin_exc).__name__}: {twin_exc}")
                        _Q.write_pack_receipt(pack_dir, record)
                        results.append(record)
                        continue
                twin_ids_override = sorted(twin_confirmed)
                row_check_ids = frozenset(twin_confirmed & set(on_prod_already))
                print(f"[{pack_id}] {'RESUMING' if apply else 'WOULD RESUME'} "
                     f"prod-only: {len(prod_missing)} "
                     f"of {len(exc.ids)} id(s) still missing from prod, "
                     "confirmed on twin by a live check (no per-pack "
                     "artifact -- these ids are on twin because a different "
                     "pack_id landed them there (e.g. an earlier split-pack "
                     "landing this is a duplicate of; task 664 incident 2) "
                     "or because this pack's own prod leg died part-way and "
                     "its written-ids artifact is lost)")
            else:
                print(f"[{pack_id}] {'RESUMING' if apply else 'WOULD RESUME'} "
                     f"prod-only: {len(prod_missing)} "
                     f"of {len(exc.ids)} id(s) still missing from prod, "
                     "twin already has all of them per this pack's own "
                     "write artifact (task 664 partial-write recovery)")
            if gate["profile"] != "mechanical" or not apply:
                # 2026-10-03: this resume is a PROD WRITE and used to run with
                # neither check -- a dry run (`--apply` absent, the mode that
                # proves a plan) upserted the owed ids into prod, took the
                # writer locks and marked the pack DONE. Same rule as the
                # content-pack resume above (task 673 review finding 1).
                source = ("a live twin check" if twin_ids_override is not None
                          else "this pack's own twin-written-ids artifact")
                record["gate"] = gate
                blocked = [] if gate["profile"] == "mechanical" else [
                    f"gate: {gate['contract_path']} profile is "
                    f"{gate['profile']!r} (flips to 'mechanical' once "
                    f"PR #{gate['pr']} merges)"]
                record["status"] = "PROVED-BLOCKED" if blocked else "PROVED-DRY-RUN"
                record["blocked_on"] = blocked
                record["would"] = (f"resume prod leg: {len(prod_missing)} ids "
                                   f"(twin ids from {source})")
                print(f"[{pack_id}] {record['status']}: would {record['would']}; "
                     f"{'; '.join(blocked) if blocked else '--apply not set'}")
                _Q.write_pack_receipt(pack_dir, record)
                results.append(record)
                continue
            write_attempt = _Q.WriteAttempt()
            try:
                state = _Q.resume_prod_only(
                    pack_id=pack_id, manifest=manifest, twin_url=twin_url,
                    prod_url=prod_url, task=task, batch=batch,
                    pause_s=pause_s, artifact_root=artifact_root,
                    write_attempt=write_attempt,
                    twin_ids_override=twin_ids_override,
                    row_check_ids=row_check_ids or None,
                    prod_collection=get_prod_collection(),
                    prod_alias=prod_alias)
                record["status"] = "LANDED"
                if twin_ids_override is not None:
                    record["note"] = ("duplicate of already-landed pack(s) "
                                      "(or a prod leg of this pack that died part-way) "
                                      "-- resumed via a live twin-vs-prod "
                                      "membership diff, no per-pack "
                                      "artifact")
                record["prod"] = state.get("prod_receipt", {})
                record["resumed_prod_only"] = True
                # Task 2364 F2: the resume only upserts. A url-group supersede left at `deleting` (the
                # prod leg died mid-upsert) is decided by prod lookup HERE, before DONE.
                _settled = _Q.settle_supersede_deleting(
                    manifest, manifest_sha, artifact_root, twin_url=twin_url, prod_url=prod_url,
                    prod_collection=get_prod_collection(), prod_alias=prod_alias, batch=batch,
                    task=task, apply=True, write_attempt=write_attempt)
                if _settled is not None:
                    record["supersede_recovery"] = _settled
                if CL.configured_dir() is not None:
                    # the resume's prod artifact names every id of the pack prod holds once the leg is done
                    _Q._log_resumed_landing(record, CL.read_written_ids(
                        CL.written_ids_path(artifact_root, pack_id, "prod")),
                        manifest=manifest, manifest_sha=manifest_sha,
                        tombstones=_Q._supersede_tombstones_on_resume(manifest, manifest_sha, artifact_root))
                _Q.atomic_write_text(pack_dir / "DONE", _Q.now() + "\n")
            except BaseException as exc2:  # noqa: BLE001
                record["reason"] = f"{type(exc2).__name__}: {exc2}"
                if write_attempt.started:
                    record["status"] = "ABORTED"
                    print(f"[{pack_id}] ABORTED (mid-write during prod-only "
                         f"resume, drain stops): {exc2}")
                    _Q.write_pack_receipt(pack_dir, record)
                    results.append(record)
                    raise
                record["status"] = "REFUSED"
                record["stage"] = "live-apply-preflight"
                print(f"[{pack_id}] REFUSED (prod-only resume, no write "
                     f"issued): {exc2}")
                _Q.write_pack_receipt(pack_dir, record)
                results.append(record)
                continue
            # Task 801 (review 802 S1): AFTER the write, outside the try that marks ABORTED. The
            # carry-over reads a receipt that can be 118 MB; a failure in it must never turn a LANDED
            # resume into ABORTED. It falls back to the derive-only shape.
            try:
                record.update(_Q.resume_receipt_contract(
                    _Q._read_prior_receipt(pack_dir), record["prod"], record))
            except Exception as exc3:  # noqa: BLE001
                print(f"[{pack_id}] resume receipt carry-over failed ({type(exc3).__name__}: {exc3}); "
                      "derive-only shape")
                record.update(_Q.resume_receipt_contract(None, record["prod"], record))
            _Q.write_pack_receipt(pack_dir, record)
            results.append(record)
            continue
        except _Q.REFUSAL_EXCEPTIONS as exc:
            print(f"[{pack_id}] REFUSED by plan(): "
                 f"{type(exc).__name__}: {exc}")
            record.update(status="REFUSED", stage="plan",
                         reason=f"{type(exc).__name__}: {exc}")
            _Q.write_pack_receipt(pack_dir, record)
            results.append(record)
            continue

        joined = plan.pop("_joined")
        plan.pop("_matrix", None)
        plan.pop("_id_index", None)
        record["german_parts"], gp_reason = GPG.guarded(GPG.check_joined, pack_id, joined)
        if gp_reason:
            # Not cached (stage not in `_CACHEABLE_REFUSAL_STAGES`): it depends on the
            # interpreter that runs THIS drain, so it retries on every pass.
            record.update(status="REFUSED", stage="german_parts", reason=gp_reason)
            _Q.write_pack_receipt(pack_dir, record)
            results.append(record)
            continue
        # Coordinator's ruling, task 664, 2026-09-26: rows whose PAGE this
        # collection already serves under another point id (below
        # PSP.ALREADY_SERVED_FAIL_SHARE_MAX of the pack's urls -- past that,
        # plan() itself already refused) are dropped from this apply, same
        # mechanism as a mechanical-content drop below, merged into the
        # SAME `drop_ids` set apply_side() already knows how to exclude.
        already_served_pids = plan.pop("_already_served_pids", frozenset())
        record["mechanical_plan"] = {"rows": plan["rows"],
                                     "identity_check": plan["identity_check"]}
        if plan.get("waiver") is not None:
            # Tim's ruling 2026-09-24: the waiver + full per-row drop
            # detail (id/url/text length/detector score) lands in the
            # RECEIPT, not just the in-memory plan -- this is the
            # accountability record for a Tim-issued, pack-scoped waiver
            # of the lang payload-carry floor.
            record["waiver"] = plan["waiver"]
        record["gate"] = gate

        blocked = []
        drop_ids: frozenset[str] = already_served_pids
        if already_served_pids:
            record["already_served_skipped"] = len(already_served_pids)
        if gate["profile"] != "mechanical":
            blocked.append(
                f"gate: {gate['contract_path']} profile is "
                f"{gate['profile']!r} (flips to 'mechanical' once PR "
                f"#{gate['pr']} merges)")
        else:
            # ADR-110's "content guard passes", read per the orchestrator's
            # 2026-09-23 ruling as `mechanical-content-v1`: a mechanical,
            # per-row, no-query check (the gold-query cytir/gates/content_guard.py arms
            # moved to the weekly battery). Only computed once gate is
            # mechanical -- no point scoring 1M+ rows for a pack that is
            # about to be blocked on the gate anyway.
            mc = _Q.mechanical_content_status(manifest, joined)
            record["mechanical_content"] = {
                k: v for k, v in mc.items() if k not in ("problems", "held_out")}
            record["mechanical_content"]["problem_sample"] = dict(
                list(mc["problems"].items())[:20])
            if mc.get("held_out"):        # absent on a clean pack: its receipt is unchanged
                record["mechanical_content"]["held_out_sample"] = dict(
                    list(mc["held_out"].items())[:20])
            if mc["verdict"] == "REFUSE":
                # Task 664, 2026-09-26 ruling: TWO independent ceilings, so
                # name whichever one(s) actually fired rather than a single
                # figure that may not be the one that refused.
                reasons = []
                if mc["other_failing_share"] > mc["fail_share_max"]:
                    reasons.append(
                        f"{mc['other_failing_share']:.2%} of rows failed a "
                        f"non-word-floor check (max {mc['fail_share_max']:.2%}) "
                        f"-- {mc['other_failing_rows']}/{mc['rows_scored']}")
                if mc["word_floor_share"] > mc["word_floor_fail_share_max"]:
                    reasons.append(
                        f"{mc['word_floor_share']:.2%} of rows fell below the "
                        f"word floor (max {mc['word_floor_fail_share_max']:.2%}) "
                        f"-- {mc['word_floor_dropped']}/{mc['rows_scored']}")
                blocked.append("mechanical_content: " + "; ".join(reasons))
            else:
                # task 777: rows held out as not prose are dropped too -- counted in the
                # receipt (`mechanical_content.held_out_rows`/`held_out_by_reason`), never a refusal
                drop_ids = (frozenset(mc["problems"]) | frozenset(mc.get("held_out") or {})
                            | already_served_pids)

        # ADR-067 am.1 point 4 (task 712): a pack that declares `supersede: url-group` replaces the
        # old rows of its URL groups. Planned here, read-only, before any lock; the export is
        # written only on a live apply. A pack without the field never enters this block.
        supersede_plan = None
        if manifest.get(UGS.FIELD) is not None:
            try:
                if single is not None:
                    raise QueueError("supersede: url-group needs the twin-first transaction (it "
                                     "reads the old rows from the twin); a Single drain has none")
                supersede_plan = UGS.plan_for_drain(
                    manifest, manifest_sha=manifest_sha, joined=joined, drop_ids=drop_ids,
                    twin_url=twin_url, artifact_root=artifact_root,
                    write=(apply and not blocked), contract_path=contract_path,
                    queue_dir=queue_dir)
            except (QueueError, *_Q.REFUSAL_EXCEPTIONS, TimeoutError, socket.timeout,
                    urllib.error.URLError) as exc:
                print(f"[{pack_id}] REFUSED (url-group supersede; nothing written to twin or "
                      f"prod): {type(exc).__name__}: {exc}")
                record.update(status="REFUSED", stage="supersede_check",
                              reason=f"{type(exc).__name__}: {exc}")
                _Q.write_pack_receipt(pack_dir, record)
                results.append(record)
                continue
            record["url_group_supersede"] = supersede_plan["summary"]

        if blocked or not apply:
            record["status"] = "PROVED-BLOCKED" if blocked else "PROVED-DRY-RUN"
            record["blocked_on"] = blocked
            reason = "; ".join(blocked) if blocked else "--apply not set"
            print(f"[{pack_id}] {record['status']}: {reason}")
            _Q.write_pack_receipt(pack_dir, record)
            results.append(record)
            continue

        # LIVE: clears the mechanical plan, ADR-110 is ratified, and the
        # mechanical-content-v1 check passed (with `drop_ids` rows excluded
        # if any). Twin, gate, prod, under both locks.
        write_attempt = _Q.WriteAttempt()
        try:
            # Task 675: refuse before the twin leg if any row cannot be written
            # in either target's declared payload form (see the content-pack
            # branch above for why this sits inside the try).
            record["consent_flags"] = _Q.consent_flag_census(
                v[2] for pid, v in joined.items() if pid not in drop_ids)
            if single is None:
                record["payload_profile"] = _Q.payload_profile_preflight_both(
                    manifest["collection"], get_prod_collection(),
                    [v[2] for pid, v in joined.items() if pid not in drop_ids])
                state = _Q.live_apply(pack_id=pack_id, manifest=manifest,
                                   twin_url=twin_url, prod_url=prod_url, task=task,
                                   batch=batch, pause_s=pause_s,
                                   artifact_root=artifact_root, drop_ids=drop_ids,
                                   twin_parallel=twin_parallel,
                                   write_attempt=write_attempt,
                                   prod_collection=get_prod_collection(),
                                   prod_alias=prod_alias,
                                   **({"supersede": supersede_plan}
                                      if supersede_plan is not None else {}))
                record["status"] = "LANDED"
                record["twin"] = state.get("twin_receipt", {})
                record["prod"] = state.get("prod_receipt", {})
            else:
                record["payload_profile"] = {"target": PSP.payload_profile_preflight(
                    single.collection, [v[2] for pid, v in joined.items() if pid not in drop_ids],
                    profile_collection=single.payload_profile_of)}
                state = _Q.single_apply(pack_id=pack_id, manifest=manifest, single=single,
                                     task=task, batch=batch, artifact_root=artifact_root,
                                     drop_ids=drop_ids, write_attempt=write_attempt)
                record["status"] = "LANDED"
                record["transaction"] = _Q._single_record(single)
                record["target"] = state.get("single_receipt", {})
                record["check5"] = state["check5"]
            if drop_ids:
                record["mechanical_content_dropped"] = len(drop_ids)
            if single is None:
                gone = ((supersede_plan or {}).get("prod") or {}).get("to_delete") or ()
                if CL.configured_dir() is not None and len(gone) != (record["prod"].get("superseded") or 0):
                    # the log must never under-record deletes: fail before DONE, loudly
                    raise CL.ChangeLogError(
                        f"{pack_id}: {len(gone)} superseded id(s) planned for prod but the prod receipt "
                        f"says {record['prod'].get('superseded')} deleted; not writing the segment")
                _Q._log_sealed_landing(record, manifest_sha=manifest_sha, artifact_root=artifact_root,
                                    superseded_ids=gone)
            _Q.atomic_write_text(pack_dir / "DONE", _Q.now() + "\n")
        except BaseException as exc:  # noqa: BLE001
            record["reason"] = f"{type(exc).__name__}: {exc}"
            if write_attempt.started:
                # A write was already issued (twin and/or prod) -- possible
                # partial-write divergence, not a clean refusal. Stop the
                # whole drain (task 660: exit non-zero only for this case).
                record["status"] = "ABORTED"
                print(f"[{pack_id}] ABORTED (mid-write, drain stops): {exc}")
                _Q.write_pack_receipt(pack_dir, record)
                results.append(record)
                raise
            # Nothing was written yet -- e.g. the write-time `plan()`
            # refused before `upsert()` ever ran (should now be rare: see
            # `apply_side`'s `plan_kwargs_with_scope` fix, task 660), OR a
            # lock acquire / other preflight step raised before any write
            # was attempted. Not in `_CACHEABLE_REFUSAL_STAGES` -- stage
            # "live-apply-preflight" always retries next drain (task 662,
            # 2026-09-25: this used to reach the skip-cache as `stage=None`
            # and get cached forever, indistinguishable from a genuine
            # content REFUSE).
            record["status"] = "REFUSED"
            record["stage"] = "live-apply-preflight"
            print(f"[{pack_id}] REFUSED (no write issued): {exc}")
            _Q.write_pack_receipt(pack_dir, record)
            results.append(record)
            continue
        _Q.write_pack_receipt(pack_dir, record)
        results.append(record)
    print("[drain] passed over without a result (retired markers; count only, nothing is re-run): "
          + (", ".join(f"{n} {m}" for m, n in sorted(retired_markers.items())) or "none"))
    return results


def _settle_guard(record: dict, pack_dir: pathlib.Path, results: list, *, apply: bool,
                  **kw) -> bool:
    """Run `settle_supersede_deleting` for an "already held" exit. True = go on. False = the pack was
    REFUSED (no delete issued; receipt written, the drain continues). A failure after a delete was
    issued writes an ABORTED receipt and re-raises (the drain stops). Review 1768 finding 3."""
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    attempt = _Q.WriteAttempt()
    try:
        got = _Q.settle_supersede_deleting(apply=apply, write_attempt=attempt, **kw)
    except BaseException as exc:  # noqa: BLE001
        # `ApplyAbort` is a SystemExit (the alias moved mid-drain): it stops the drain like a failure
        # after a delete, but it gets its receipt first (review 1768 round 3).
        stops = attempt.started or isinstance(exc, _Q.TFA.ApplyAbort)
        if not stops and not isinstance(exc, Exception):
            raise
        record["reason"] = f"{type(exc).__name__}: {exc}"
        record["stage"] = "supersede-settle"
        record["status"] = "ABORTED" if stops else "REFUSED"
        record["note"] = ("the supersede settle stopped this pack; no DONE was written"
                          + ("; a prod delete had been issued" if attempt.started else "; no delete was issued"))
        print(f"[{record.get('pack_id')}] {record['status']} (supersede settle, "
              f"{'a delete was issued' if attempt.started else 'no delete issued'}): {exc}")
        _Q.write_pack_receipt(pack_dir, record)
        results.append(record)
        if stops:
            raise
        return False
    if got is not None:
        record["supersede_recovery" if apply else "would_supersede_recovery"] = got
    return True
