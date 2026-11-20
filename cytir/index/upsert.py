#!/usr/bin/env python3
# Moved from the retired `upsert_wave_prod` script (task 677, task 643 S5); the old path is an alias shim.
"""Upsert a dumped staging wave (`cyt dump` jsonl.gz) into the LIVE
prod collection — the guarded merge step of the discovery/sitemap waves
(task 092 discipline, now a repo tool instead of an ephemeral /tmp script):

  * waits for / takes the purpose-stamped writer lock (/tmp/aw_prod_writer.lock,
    coordinated with the snippet write-orchestrator), released in `finally`;
  * batches of --batch points, `wait=true` — each upsert fully lands;
  * RAM guard: pauses while MemAvailable < --min-avail-gib (30G box, OOM
    incident 2026-06-28);
  * health watch: /healthz (or /health) probed every --health-every batches —
    aborts (lock released) if serving goes red;
  * exact `points/count` before/after, printed for the merge log;
  * SUPERSEDE-ON-UPSERT (task 348, default ON): after the upserts, still under
    the writer lock, every canonical page group the wave touched is reconciled —
    existing points of the same page whose id is not in the wave are deleted
    (old versions of changed Tier-B chunks; ALL stale chunks of shrunk pages).
    COMPLETENESS CONTRACT: the input must carry the complete current record set
    of every page it touches — full `cyt dump` dumps do by
    construction; a dump_netnew changed-set dump only in format 2 (manifest
    `supersede_complete: true`), and format-1 changed-set dumps are REFUSED
    before any write. The target needs a keyword payload index on `url`; on
    LIVE prod that is an ops change (create once, mirror on the twin) — this
    script never auto-creates it silently (--create-url-index makes the choice
    explicit).

  * TWIN-MIRROR (task 359 / ADR-081 amendment, 2026-07-30): every batch is
    upserted to the TWIN first, then to prod, IN THIS SAME PROCESS — never a
    second command a human/agent is trusted to also run (that gap is exactly
    what produced the 07-30 divergence: a background waiter died between the
    prod leg and the twin leg of a manual two-command dance). Supersede
    reconciliation runs twin-first too. The whole apply is NOT reportable as
    success (the final DONE line is not printed, exit code is non-zero) until
    the twin's count DELTA is verified against the same arithmetic used for
    prod (upsert_supersede.verify_count_delta). Twin connectivity (env-driven,
    the retired `upsert_supersede` script, twin_config — never a literal here) is
    checked BEFORE the writer lock is taken and before any write, so a dead
    twin never leaves prod partially written with no twin attempt at all.
    Escape hatch for genuine twin-down emergencies: AW_APPLY_NO_TWIN=1 (or
    --no-twin) — loud on stdout AND in the twin-mirror-pending ledger, and the
    success line is DONE-NO-TWIN-VERIFICATION, never the plain DONE that
    implies a verified mirror. (Replaces the old --peer-url post-hoc sample
    check, which the 2026-07-13 FINMA merge already found unusable cross-box —
    neither serve box reaches the other's bare localhost qdrant.)

  python upsert_wave_prod.py --in /tmp/wave1b_points.jsonl.gz \
      --collection aw_prod --health-url http://localhost:8080/healthz
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time

from cytir.core import internal_http as IH
from qdrant_client import models

from cytir.core.bm25_saturate import saturate_point_bm25  # noqa: E402
from cytir.core.qdrant import qdrant_connect  # noqa: E402
from cytir.index.preservation_contract import (ContractError,
                                   verify_publish_environment)  # noqa: E402
from cytir.index.supersede import (CardOverwriteRefused, Superseder,  # noqa: E402
                              client_url, connect_twin,
                              dump_is_changed_set, guard_card_overwrites,
                              load_card_waivers, read_dump_manifest,
                              twin_config, verify_count_delta)

LOCK = "/tmp/aw_prod_writer.lock"


def take_lock(purpose: str, wait_s: int = 3600, break_stale: bool = False):
    t0 = time.time()
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{purpose} pid={os.getpid()} at={time.ctime()}\n"
                     .encode())
            os.close(fd)
            print(f"[lock] took {LOCK}", flush=True)
            return
        except FileExistsError:
            held = ""
            try:
                held = open(LOCK).read().strip()
            except Exception:  # noqa: BLE001
                pass
            # STALE-LOCK detection (2026-07-13 incident: a SIGKILLed merge left an
            # empty lock; the next merge waited the full timeout). If the holder pid
            # is dead (or the lock is empty), this lock is stale.
            stale = True
            for tok in held.split():
                if tok.startswith("pid="):
                    try:
                        os.kill(int(tok[4:]), 0); stale = False  # holder alive
                    except (ProcessLookupError, ValueError):
                        stale = True
                    except PermissionError:
                        stale = False  # exists, not ours
            if stale:
                msg = (f"[lock] STALE lock detected ({held!r} — holder dead/empty). "
                       f"Rerun with --break-stale-lock to clear it.")
                if break_stale:
                    os.remove(LOCK)
                    print(f"[lock] --break-stale-lock: removed stale {LOCK}; retrying",
                          flush=True)
                    continue
                raise SystemExit(msg)
            if time.time() - t0 > wait_s:
                raise SystemExit(f"could not take {LOCK} within {wait_s}s "
                                 f"(holder ALIVE: {held!r})")
            print(f"[lock] busy ({held!r}) — waiting", flush=True)
            time.sleep(30)


def release_lock():
    try:
        os.remove(LOCK)
        print(f"[lock] released {LOCK}", flush=True)
    except FileNotFoundError:
        pass


def avail_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 99.0


def ram_guard(min_avail: float):
    while (a := avail_gib()) < min_avail:
        print(f"  [PAUSE] MemAvailable {a:.1f}G < {min_avail}G", flush=True)
        time.sleep(20)


def to_point(rec: dict) -> models.PointStruct:
    if "vector" not in rec or "id" not in rec:
        raise SystemExit(
            "ABORT: input is not a points dump (missing 'vector'/'id' keys; got "
            f"{sorted(rec)[:8]}…). This is a raw-passage jsonl — re-dump the "
            "STAGING COLLECTION with `cyt dump` (python -m cytir.distribute.dump) before merging.")
    vec = {}
    for name, v in rec["vector"].items():
        if isinstance(v, dict):
            vec[name] = models.SparseVector(indices=v["indices"],
                                            values=v["values"])
        else:
            vec[name] = v
    # task 212 Part 1: saturate raw-tf bm25 weights on the LIVE write path so
    # post-base-build content stops re-introducing the raw-tf flood (task 211).
    # Idempotent — an already-saturated dump passes through untouched.
    saturate_point_bm25(vec)
    return models.PointStruct(id=rec["id"], vector=vec, payload=rec["payload"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--collection", default="aw_prod")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--min-avail-gib", type=float, default=1.0)
    ap.add_argument("--health-url", default="")
    ap.add_argument("--health-every", type=int, default=20)
    ap.add_argument("--purpose", default="wave merge (upsert_wave_prod.py)")
    ap.add_argument("--skip-counts", action="store_true",
                    help="skip exact before/after PROD counts — REQUIRED on "
                         "the 32G serve box (unfiltered exact-count = OOM "
                         "trigger, the 2026-07 OOM incident in notes/). Does NOT skip twin "
                         "verification: the twin's exact count is always safe "
                         "(taken on the twin box) and remains the count-of-record.")
    ap.add_argument("--break-stale-lock", action="store_true",
                    help="clear the writer lock if its holder pid is dead "
                         "(2026-07-13 SIGKILLed-merge stale-lock incident)")
    ap.add_argument("--no-supersede", action="store_true",
                    help="legacy behaviour: leave the old points of re-crawled "
                         "pages in place (score-tied duplicates, task 348) — "
                         "emergencies only, recorded in the ledger")
    ap.add_argument("--create-url-index", action="store_true",
                    help="create the url keyword index supersede needs, on "
                         "PROD AND TWIN, if missing — an explicit ops choice "
                         "on a live box")
    ap.add_argument("--no-twin", action="store_true",
                    help="EMERGENCY ESCAPE HATCH (equivalent to env "
                         "AW_APPLY_NO_TWIN=1): apply to prod only; twin is "
                         "NOT written or verified and WILL diverge until "
                         "reconciled by hand. Loud on stdout + ledgered. "
                         "Genuine twin-down emergencies only (task 359 / "
                         "ADR-081 amendment).")
    ap.add_argument("--card-waivers", default=None,
                    help="sealed card-waiver file (task 728 format) naming a stored entity "
                         "card this wave may overwrite by id (task 820)")
    ap.add_argument("--card-waivers-sha256", default=None)
    args = ap.parse_args()

    # A twin-first write can still make a perfectly mirrored, unrebuildable
    # index.  Bind the complete after-state preservation manifest before any
    # Qdrant connection, twin check, or writer lock.  The manifest must include
    # the exact originals -> Markdown -> chunks -> vectors lineage and final
    # membership receipt for the collection named here.
    try:
        preservation = verify_publish_environment(
            expect_collection=args.collection)
    except ContractError as exc:
        raise SystemExit(
            f"FATAL: preservation contract did not PASS before live apply: {exc}")
    print(f"[preservation] PASS release={preservation['release_id']} "
          f"manifest_sha256={preservation['manifest_sha256']}", flush=True)

    cfg = twin_config()
    no_twin = args.no_twin or cfg["no_twin"]

    # twin-mirror ledger: every aw_prod write is recorded so an un-mirrored
    # write, or one that used the escape hatch, is discoverable (the daily
    # twin-sync-check + humans read this).
    LEDGER = os.environ.get(
        "AW_TWIN_MIRROR_LEDGER",
        os.path.expanduser("~/aw-data/twin-mirror-pending.log"))

    def _log_ledger(line, *, n=0, superseded=0, twin_n=0, twin_superseded=0):
        try:
            os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
            with open(LEDGER, "a") as lf:
                lf.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                         f"{args.collection} {os.path.basename(args.inp)} "
                         f"n={n} superseded={superseded} twin_n={twin_n} "
                         f"twin_superseded={twin_superseded}"
                         f"{' NO-SUPERSEDE' if args.no_supersede else ''} "
                         f"{line}\n")
        except Exception:  # noqa: BLE001
            pass

    if no_twin:
        # THE escape hatch (task 359): genuine twin-down emergencies only.
        # Fail-loud discipline says a silent fallback is a lie about system
        # state — so this is the opposite of silent: multi-line stdout
        # shouting + an unmissable ledger marker, and the final status line
        # is never the plain "DONE" that would imply a verified mirror.
        print("=" * 78, flush=True)
        print("[twin] *** AW_APPLY_NO_TWIN ESCAPE HATCH ACTIVE ***", flush=True)
        print("[twin] prod WILL be written WITHOUT the twin. The twin WILL "
              "diverge until this wave is manually replayed to it. This is "
              "for genuine twin-down emergencies ONLY (task 359 / ADR-081 "
              "amendment) — never a convenience.", flush=True)
        print("=" * 78, flush=True)
        _log_ledger("NO-TWIN-ESCAPE-HATCH-ACTIVE")
        _run(args, ledger=_log_ledger, twin_client=None, twin_coll=None)
        return

    # Twin connectivity is checked BEFORE the writer lock and before any
    # write (cheap check at the front, verification-doctrine #6) — a dead
    # twin must never leave prod partially written with no twin attempt.
    with connect_twin(log=print) as (twin_client, twin_coll):
        # task 864 (review 854 S6): from an aw_e10 T0 to the flip only the drain writes the twin
        from cytir.index import twin_write_hold as TWH
        try:
            TWH.refuse_if_held(twin_coll, tool="cytir.index.upsert")
        except TWH.TwinWriteHeld as exc:
            raise SystemExit(str(exc)) from exc
        _run(args, ledger=_log_ledger, twin_client=twin_client, twin_coll=twin_coll)


def _run(args, *, ledger, twin_client, twin_coll):
    """The actual apply. `twin_client`/`twin_coll` is None only when the
    AW_APPLY_NO_TWIN escape hatch is active (already logged loudly by the
    caller) — every other code path here treats the twin write as mandatory
    and gates the success report on it (task 359 / ADR-081 amendment)."""
    c = qdrant_connect(args.qdrant_url, timeout=300)

    # --- supersede preconditions, BEFORE the lock and any write ---
    opener = gzip.open if args.inp.endswith(".gz") else open
    sup = sup_twin = None
    if args.no_supersede:
        print("[supersede] *** --no-supersede: stale points of re-crawled "
              "pages will SURVIVE (task-348 duplicate class) ***", flush=True)
    else:
        with opener(args.inp, "rt", encoding="utf-8") as probe:
            first_line = next((ln for ln in probe if ln.strip()), "")
        first = json.loads(first_line) if first_line.strip() else {}
        if dump_is_changed_set(first) and \
                not read_dump_manifest(args.inp).get("supersede_complete"):
            raise SystemExit(
                "ABORT: changed-set dump without a supersede_complete manifest "
                "(format-1 dump_netnew output) — its keep-sets are not "
                "page-complete, superseding would delete unchanged siblings. "
                "Re-dump with the format-2 dump_netnew, or pass --no-supersede "
                "to merge WITH the known duplicate bug.")
        sup = Superseder(c, args.collection)
        sup.ensure_url_index(create=args.create_url_index)
        if twin_client is not None:
            sup_twin = Superseder(twin_client, twin_coll)
            sup_twin.ensure_url_index(create=args.create_url_index)

    # --- card-overwrite guard, BEFORE the lock and any write (task 820) ----------------------
    # A stored entity card must not be overwritten by id with a row that is not a card replacing
    # it (the L1-backfill failure: six cards overwritten with shop text). Read where the write
    # lands FIRST: the TWIN when there is one (it mirrors prod and is written first, so no extra
    # prod read), prod only under the no-twin escape hatch. The whole input is refused, nothing
    # written; one indexed id retrieve per 500 ids, never a scroll (`prod_guard` paces it).
    guard_card_overwrites_for_input(args, c, twin_client, twin_coll, ledger)

    before = twin_before = None
    if not args.skip_counts:
        before = c.count(args.collection, exact=True).count
        print(f"{args.collection} before: {before:,}", flush=True)
    if twin_client is not None:
        # always safe (the build box has no 32G OOM wall) — the count-of-record
        # regardless of whether --skip-counts spared prod the exact count.
        twin_before = twin_client.count(twin_coll, exact=True).count
        print(f"[twin] {twin_coll} before: {twin_before:,}", flush=True)

    take_lock(args.purpose, break_stale=args.break_stale_lock)
    n = twin_n = 0
    superseded = twin_superseded = 0
    try:
        batch, twin_batch = [], []
        with opener(args.inp, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if sup is not None:
                    pl = rec.get("payload") or {}
                    sup.observe_payload(rec["id"], pl)
                    if sup_twin is not None:
                        sup_twin.observe_payload(rec["id"], pl)
                if rec.get("_delta") == "unchanged":
                    continue     # format-2 marker: keep-set only, no vectors
                pt = to_point(rec)
                batch.append(pt)
                if twin_client is not None:
                    twin_batch.append(pt)
                if len(batch) >= args.batch:
                    ram_guard(args.min_avail_gib)
                    _flush_batch(twin_client, twin_coll, batch, twin_batch, c,
                                args.collection)
                    n += len(batch); twin_n += len(twin_batch)
                    batch, twin_batch = [], []
                    if n % (args.batch * args.health_every) == 0:
                        print(f"  upserted {n:,}"
                              + (f" (twin {twin_n:,}, retrieve-verified)"
                                 if twin_client is not None else ""), flush=True)
                        if args.health_url:
                            try:
                                status, _body = IH.send(args.health_url, timeout=15)
                                assert status == 200, status
                            except Exception as e:  # noqa: BLE001
                                raise SystemExit(
                                    f"ABORT: health check failed after "
                                    f"{n:,} points: {e!r}")
        if batch or twin_batch:
            ram_guard(args.min_avail_gib)
            _flush_batch(twin_client, twin_coll, batch, twin_batch, c,
                        args.collection)
            n += len(batch); twin_n += len(twin_batch)
        if sup is not None:
            # reconcile touched page groups — a prod mutation, so it happens
            # UNDER the writer lock. TWIN FIRST (same reasoning as the
            # upserts above): Superseder.run() already retrieve-verifies its
            # own deletes (task 348) on whichever client it is bound to.
            ram_guard(args.min_avail_gib)
            if sup_twin is not None:
                twin_superseded = sup_twin.run()["superseded"]
            superseded = sup.run()["superseded"]
    finally:
        release_lock()

    # --- twin verification: THE gate on reporting success (task 359) -------
    # every batch above was already retrieve-verified present on the twin
    # (_flush_batch); this is the aggregate close-out: superseded counts must
    # match (same input, same reconciliation, run against collections that
    # are supposed to be identical), and — when prod's exact count was taken
    # — the two sides' count DELTAS must match (the twin-mirror-rule
    # invariant: "every prod mutation applies to the twin in the same task,
    # matching count deltas verified").
    if twin_client is not None:
        twin_after = twin_client.count(twin_coll, exact=True).count
        if superseded != twin_superseded:
            ledger(f"TWIN-MISMATCH-FATAL superseded={superseded} "
                  f"twin_superseded={twin_superseded}", n=n, superseded=superseded,
                  twin_n=twin_n, twin_superseded=twin_superseded)
            raise SystemExit(
                f"FATAL: superseded count MISMATCH — prod {superseded:,} vs "
                f"twin {twin_superseded:,}. Apply is NOT reportable as "
                f"success; collections were not identical going in, or the "
                f"reconciliation diverged — investigate before any further "
                f"write.")
        if before is not None:
            prod_delta, twin_delta = None, twin_after - twin_before
            after = c.count(args.collection, exact=True).count
            prod_delta = after - before
            try:
                verify_count_delta(twin_client, twin_coll, twin_before, twin_after,
                                   expected_delta=prod_delta, label="twin-vs-prod",
                                   log=print)
            except SystemExit:
                ledger(f"TWIN-MISMATCH-FATAL prod_delta={prod_delta} "
                      f"twin_delta={twin_delta}", n=n, superseded=superseded,
                      twin_n=twin_n, twin_superseded=twin_superseded)
                raise
        else:
            after = None
        ledger("TWIN-VERIFIED-OK", n=n, superseded=superseded, twin_n=twin_n,
              twin_superseded=twin_superseded)
    else:
        after = None if args.skip_counts else c.count(args.collection, exact=True).count
        ledger("NO-TWIN-ESCAPE-HATCH-FINAL", n=n, superseded=superseded)

    status = "DONE" if twin_client is not None else "DONE-NO-TWIN-VERIFICATION"
    tail = "" if twin_client is not None else " *** TWIN NOT WRITTEN/VERIFIED ***"
    if before is None or after is None:
        print(f"{status} upserted {n:,}, superseded {superseded:,} (prod "
              f"counts skipped — verified twin exact-count is the "
              f"count-of-record)" + tail, flush=True)
        return
    print(f"{status} upserted {n:,}, superseded {superseded:,}; "
          f"{args.collection}: {before:,} -> {after:,} "
          f"(delta {after - before:+,}; pre-existing refreshed in place: "
          f"{n - (after - before) - superseded:,})" + tail, flush=True)


def guard_card_overwrites_for_input(args, c, twin_client, twin_coll, ledger) -> dict:
    """The card check over every row `args.inp` would upsert; raises `SystemExit` (and ledgers the
    refusal) before the caller has written anything."""
    opener = gzip.open if args.inp.endswith(".gz") else open

    def _rows():
        with opener(args.inp, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("_delta") == "unchanged":
                    continue     # keep-set marker only: never written
                yield rec

    target, coll = (twin_client, twin_coll) if twin_client is not None else (c, args.collection)
    try:
        report = guard_card_overwrites(
            _rows(), client_url(target), coll, writer="cytir.index.upsert",
            waivers=load_card_waivers(args.card_waivers, args.card_waivers_sha256), log=print)
    except CardOverwriteRefused as exc:
        ledger(f"CARD-GUARD-REFUSED {str(exc)[:1500]!r}")
        raise SystemExit(f"ABORT: {exc}") from exc
    ledger(f"CARD-GUARD-OK ids_read={report['ids_read']} "
           f"stored_cards={report['stored_cards']} "
           f"waived={[w['id'] for w in report['waived']]}")
    return report


def _flush_batch(twin_client, twin_coll, batch, twin_batch, c, collection):
    """TWIN FIRST, then prod — and the twin write is retrieve-verified
    present before prod is touched (task 359). If the twin write fails or a
    written id is not retrievable, this raises BEFORE prod advances past it,
    so a mid-run death leaves both sides at an identical, resumable prefix —
    never prod ahead of an unverified or unattempted twin."""
    if twin_client is not None and twin_batch:
        twin_client.upsert(twin_coll, points=twin_batch, wait=True)
        got = twin_client.retrieve(twin_coll, ids=[p.id for p in twin_batch],
                                   with_payload=False, with_vectors=False)
        if len(got) != len(twin_batch):
            raise SystemExit(
                f"FATAL: twin upsert of {len(twin_batch)} points landed only "
                f"{len(got)} retrievable — apply is NOT reportable as "
                f"success; prod was NOT advanced past this batch.")
    if batch:
        c.upsert(collection, points=batch, wait=True)


if __name__ == "__main__":
    main()
