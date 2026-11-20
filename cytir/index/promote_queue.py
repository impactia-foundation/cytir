#!/usr/bin/env python3
# Moved from `promote_queue.py` (task 677, task 643 S5); the old path is an alias shim.
"""Drain the sealed-pack promotion queue: twin first, then prod, one pack at a time.

Task 641 (growth lane). `promote_sealed_pack.py` already does the hard part --
reading a sealed 1024-d pack, minting ids, checking identity, refusing an
already-present id or an already-served url, enforcing the licence floor -- for
ONE pack against ONE target, dry-run by default. What did not exist is the
missing piece the growth lane needs nightly: a QUEUE of ready packs, a
gate-contract switch, and the twin-then-prod transaction protocol. This module
supplies exactly that, by importing every primitive rather than re-deriving one
(One Truth, One Home): `promote_sealed_pack.plan/build_points/upsert` do the
promotion; `twin_first_apply.twin_first` owns the cross-target safety
boundary (one real production writer lock, twin gated before prod is touched, a
durable ledger) exactly as it does for `task 629 run_batch.py`; the
the build box `TWIN-WRITE-LOCK`/`PROD-WRITE-LOCK` advisory pair is imported from that
same driver rather than re-implemented a second time; `rollback_restore_null_arm`
supplies the delete-and-verify used by the twin-only rehearsal.

QUEUE SHAPE
-----------
    <queue-dir>/<pack-id>/manifest.json   -- schema aw.promotion-queue-pack.v1
    <queue-dir>/<pack-id>/READY           -- sha256(manifest.json), one line
    <queue-dir>/<pack-id>/DONE            -- written after a pack lands on BOTH
                                              sides; the drain loop then skips it

A pack is drained only if READY's content equals a fresh sha256 of
manifest.json -- the same integrity discipline
`task 629 launch_b01.sh` applies to every pinned input before any
lock: a manifest edited after being marked ready is a manifest nobody
re-approved, and this refuses rather than trusting the READY file's mere
existence.

Manifest fields (required unless noted): `schema`, `pack_id`, `family`,
`pack_chunks`, `vectors`, `mrl`, `collection`. Optional, forwarded verbatim to
`promote_sealed_pack.plan()`: `consent`, `attribution`, `license_version`,
`allow_supersede`, `require_persisted`, `identity_check`, `identity_reason`,
`published_from`, `published_status_from`, `published_status_requires`,
`require_published`, `family_payload`, `title_from`, `license_from`. Optional, read by the drain
itself: `supersede` (`"url-group"`, needs `allow_supersede`; replaces the served rows of every URL
group the pack writes, export first -- `cytir/index/url_group_supersede.py`; only for a family on its
allowlist; the stage-time guard block is read from the SEALED pack's own PACK.json, never from
this manifest). Optional,
`mechanical_content_check.evaluate()` overrides: `word_floor`,
`coverage_floor`, `fail_share_max`, `l1_sources`, `detect_language`.

THE GATE-CONTRACT SWITCH -- ONE TRACKED FIELD, NEVER A LIVE PROBE
--------------------------------------------------------------------
ADR-110 (task 639, PR #302, `origin/task639-mechanical-gate-policy`) will
let a pack from a PORTFOLIO-admitted family land on five mechanical checks:
counts reconcile, consent+licence per row, dedup/supersede clean, the
content guard passes, twin == prod after apply. As of 2026-09-23 it is
OPEN, not merged. Until it merges, the contract every pack answers to today
is the full ADR-043 battery (arm-cutting against the gold-query set,
`task 351 apply_content_pack_shared.py`'s default
`post_twin_battery_gate`, `task 370 compare_corpus_arms.py
--profile content-pack`) -- real measurement work this module does not run
generically for an arbitrary family (that generalization is the explicit
growth-lane follow-up ADR-110's own text names, not yet built for anything
but W1's B01/B05).

So `gate_profile()` below reads ONE field, `profile`, from the tracked
contract file `data/promotion-gate-contract.v1.json` -- config as data, not
a live git check of what Tim has merged (a nightly cron has no business
depending on network access to decide what it is legally allowed to write).
`"battery-required"` (today's value): this driver proves every queued pack
clears the join/licence/dedup checks (the read-only `plan()`) and then
REFUSES to write anything, twin or prod -- never a silent skip of the gate
the current contract requires. `"mechanical"`: the five ADR-110 checks
govern and, on PASS, the pack proceeds to prod. The coordinator flips the
field in a receipt PR the moment #302 lands on main; nothing in this module
needs to change for the flip to take effect on the very next drain.

Content guard, specifically: `cytir/gates/content_guard.py`'s gold-query
BASELINE/CANDIDATE arms are RANKING measurement -- ADR-110 moves that to
the weekly whole-prod battery, never a per-pack gate. The per-pack "content
guard passes" check is `mechanical_content_check.py`'s `mechanical-
content-v1` profile: a mechanical, no-query, per-row content check (title,
text+word-floor, language match, stub/coverage vs L1), composed from
existing validators (orchestrator's ruling, 2026-09-23). See that module's
docstring for exactly which check comes from where. A pack whose failing-
row share exceeds `fail_share_max` (default 1%) is refused whole; below
that, the failing rows are dropped (never silently promoted) and named in
the receipt.

THROTTLE
--------
A burst of unthrottled upserts degrades the 30 GB prod box (memory
`aw-heavy-rewrite-degrades-prod`: an optimization storm, 60s API timeouts,
self-recovers but is worth avoiding). `promote_sealed_pack.upsert()` grew a
`pause_s` parameter for exactly this; `--batch`/`--pause-s` here are config,
not literals, and their defaults are a conservative starting point, not a
measured optimum -- revisit with real timing data before trusting them at
the very largest packs.

TWIN-ONLY REHEARSAL
--------------------
`--rehearse PACK_ID` applies one pack to the twin ONLY (never acquires the
prod lock, never touches prod), seals its membership, then rolls it back via
`rollback_restore_null_arm` and re-verifies the twin is back at its prior
count. This is a distinct action from a drain: its purpose is to prove the
plumbing (queue read, smoke test, throttled apply, membership seal, clean
rollback) round-trips safely, independent of whether the gate is ratified --
it deliberately does not go through the gate switch, because it never
writes anything durable.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import gzip
import hashlib
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]

from cytir.core import twin as TWIN  # noqa: E402
from cytir.core.canon import document_url_key, strip_tracking_url  # noqa: E402 -- THE same-document url key (task 698)
from cytir.core.prose_guard import binary_content  # noqa: E402 -- the prose test (task 767)
from cytir.sources import soft404 as SOFT404  # noqa: E402 -- the declared soft-404 detector (task 1363)
from cytir.core import paths as PATHS  # noqa: E402
from cytir.core.paths import (SettingRefused, SettingsError, box_file_setting,  # noqa: E402
                           operator_settings)
from cytir.core import projection as PROJ  # noqa: E402
from cytir.chunk import registry as CHREG  # noqa: E402 -- epoch -> chunk width (task 984)
from cytir.index import layout as LAYOUT  # noqa: E402
from cytir.index import change_log as CL  # noqa: E402 -- the delta change log (task 2314); off unless `change_log_dir` is set
from cytir.gates import mechanical_content_check as MCC  # noqa: E402
from cytir.index import news_window_filter as NWF  # noqa: E402
from cytir.index import payload_carry as PC  # noqa: E402
from cytir.corpus import persisted as PI  # noqa: E402
from cytir.index import point_delete as PD  # noqa: E402
from cytir.index import supersede as US  # noqa: E402 -- the ONE card test (`is_card`, task 698 R2)
from cytir.index import promote_sealed_pack as PSP  # noqa: E402
from cytir.index import sparse_rule as SR  # noqa: E402 -- the ONE BM25 prune rule (task 717)
from cytir.gates import rollback_restore_null_arm as RB  # noqa: E402
from cytir.index import twin_first as TFA  # noqa: E402
from cytir.index import payload_patch as PPK  # noqa: E402
from cytir.index import manifest_rows as MR  # noqa: E402 -- rows_staged / rows_to_write_max (2026-10-04)
from cytir.index import exclude_file as XF  # noqa: E402 -- sha-pinned exclusion files (task 680)
from cytir.index import payload_profile as PP  # noqa: E402 -- the ONE payload projection (task 675)
from cytir.index import id_rule as IR  # noqa: E402 -- the ONE I1 id rule (task 847/864)
from cytir.index import row_decode as RD  # noqa: E402 -- the ONE canonical stored row, any profile (task 855)
from cytir.index import live_target as LT  # noqa: E402 -- a build never writes a live endpoint
from cytir.index import dry_run as DRY  # noqa: E402 -- a dry drain never writes (2026-10-03)
from cytir.index import prod_guard as PG  # noqa: E402 -- no prod scroll, no prod replay (2026-10-07)
from cytir.index import chunk_checkpoint as CK  # noqa: E402 -- chunk-level resume of the prod leg (task 786)
from cytir.index import german_parts_guard as GPG  # noqa: E402 -- task 710 guard 1: German rows carry their parts
from cytir.index import url_group_supersede as UGS  # noqa: E402 -- `supersede: url-group` (ADR-067 am.1, task 712)
from cytir.index import card_waivers as CW  # noqa: E402 -- sealed per-id waivers for a wrong card (task 728)
from cytir.index import mirror_compare as MC  # noqa: E402 -- twin vs prod by content (task 833)

# task 2875 (RULING 80): the drain split. These names now live in cytir/index/drain/*; this facade
# re-exports every one so each importer and each monkeypatch on this module keeps resolving here.
from cytir.index.drain.common import (  # noqa: E402,F401
    _load_run_batch, _RESUME_RETRY_BACKOFF_S, _RESUME_RETRYABLE_EXCEPTIONS, _retry_resume,
    _would_settle_text, atomic_write_bytes, atomic_write_text, default_queue_dir, live_queue_dir,
    now, prod_lock_host, QueueError, READ_FAILURE_LOUD_AFTER, read_failure_streak, Single,
    Transaction, TransientReadError, TwinFirst, WriteAttempt
)
from cytir.index.drain.locks import (  # noqa: E402,F401
    hold_unit_lock, PROD_LOCK_PATH, single_target_lock, UnitLockRefused
)
from cytir.index.drain.receipts import (  # noqa: E402,F401
    _dry_run_receipt, _read_prior_receipt, _write_receipt_never_masking,
    content_pack_twin_state_path, drop_held_out_cards, HELD_CARD_OVERWRITE, HELD_OUT_CARDS_FILE,
    held_out_receipt_fields, inplace_ids_path, INPLACE_IDS_SCHEMA,
    load_content_pack_ids_from_state, lookup_record_scope, MAX_PRIOR_RECEIPT_BYTES,
    prod_call_receipt, prod_chunk_checkpoint, read_content_pack_twin_state,
    read_held_out_receipt_fields, read_inplace_ids, record_inplace_ids, resume_receipt_contract,
    TWIN_FIRST_KEYS, twin_written_ids_path, write_content_pack_twin_state, write_held_out_cards,
    write_pack_receipt
)

# task 2875 (RULING 80): the drain split. These names now live in cytir/index/drain/*; this facade
# re-exports every one so each importer and each monkeypatch on this module keeps resolving here.
from cytir.index.drain.changelog import (  # noqa: E402,F401
    _log_landing, _log_resumed_landing, _log_sealed_landing, _supersede_tombstones_on_resume
)

# task 2890 (RULING 80): the drain split, step 5. Re-exported so each importer and each monkeypatch
# on this module keeps resolving here.
from cytir.index.drain.queue_scope import (  # noqa: E402,F401
    GATE_CONTRACT_PATH, GATE_CONTRACT_SCHEMA, LIVE_QUEUE_MARKER, MANIFEST_SCHEMA,
    SINGLE_STATE_MARKER, _LOCK_SAFE, _dead_marker_tmps, _dry_run_never_writes,
    _foreign_state_entry, _in_pack_area, _inside_or_equal, _overlaps_queue, _pack_estimate_ops,
    _pack_family_best_effort, _single_record, _single_transaction_key, apply_startup_refusal,
    assert_real_path, assert_single_target_not_live, box_file_queue_dir, claim_state_entry,
    gate_profile, live_queue_refusal, marked_live_queue, order_pack_dirs_for_drain,
    plan_startup_refusal, read_manifest, refuse_live_queue_run, refuse_live_single,
    refuse_state_entries, single_startup_refusal, single_state_dir, sparse_rule_startup_refusal
)

#: `promote_sealed_pack.plan()` refuses through FOUR separate exception
#: hierarchies, not one -- `PromotionError` for its own checks,
#: `persisted_inputs.PersistedInputsError` (ADR-105, ~`assert_pages_
#: persisted`), `payload_carry_contract.PayloadCarryError` (~`assert_
#: scoreable`), and `aw_b1_projection.ProjectionError` (per-row licence/
#: consent/trust_tier refusals inside `project_record`) -- because each of
#: these modules is its own home for its own refusal and none of them
#: imports another's exception type. A per-pack catch on fewer than all
#: four lets the missing ones crash the WHOLE drain on the first pack that
#: hits them -- measured live against the real twin, 2026-09-23, TWICE:
#: LEGI raised `UnpersistedInputs` uncaught, and CC tranche 1 then raised
#: `ProjectionError` ("unresolved license") uncaught, each in turn taking
#: the process down mid-queue before the fix that covered it. This tuple is
#: every refusal `plan()` is known to raise today; anything outside it is a
#: genuine bug and is meant to crash loud, not be silently absorbed.
#: Task 680: a fifth -- `exclude_file.ExcludeFileError` (a manifest's pinned
#: exclusion file is missing, unhashable to its pin, or empty) is a
#: deterministic judgment about THIS pack, refused per pack like the others.
REFUSAL_EXCEPTIONS = (PSP.PromotionError, PI.PersistedInputsError,
                     PC.PayloadCarryError, PROJ.ProjectionError,
                     XF.ExcludeFileError)

#: Stages whose REFUSED verdict is a deterministic content judgment -- the
#: SAME manifest re-scored will REFUSE the SAME way every time, so caching
#: it (skip-until-reseal, see the SKIPPED-REFUSED-UNCHANGED cache in
#: `drain()`) costs nothing. A WHITELIST, not a blacklist (growth-lane,
#: 2026-09-25, task 662): task 660's fix excluded only `stage == "intake"`
#: by name, which meant every OTHER pre-write refusal -- including a lock
#: acquire that raised in `live_apply()`'s `except BaseException` before any
#: write, where the two call sites left `stage` unset entirely -- fell
#: through the `!= "intake"` blacklist and got cached as if it were a
#: content verdict. A lock contention, a 300s API timeout exhausted past
#: `_API_TIMEOUT_RETRIES`, or any other transient/environmental failure is a
#: fact about THIS run, not about the pack; caching it means the pack never
#: retries again until someone notices and reseals a byte-identical
#: manifest just to mint a new sha. Only `plan` (dry_plan()'s REFUSAL_EXCEPTIONS)
#: and `content_pack_check` (content_pack_dry_check()'s REFUSAL_EXCEPTIONS)
#: are content judgments computed with no side effects and no network
#: dependency beyond a read-only twin query already retried inside `_api()`;
#: everything else -- intake, and any pre-write live-apply preflight
#: failure -- retries every drain.
_CACHEABLE_REFUSAL_STAGES = frozenset({"plan", "content_pack_check"})

SCHEMA = "aw.promotion-queue-drain.v1"

#: task 673, review finding 5: a conservative floor for the drain's
#: pre-take duration estimate, ops/second on ONE leg. Real observed
#: rates: eric-official-pdf-tranche-020000's resumed-prod-only leg (task
#: 673's receipt) landed 161,373 rows in 805.385s (~200/s);
#: `promote_nightly.sh`'s own `--batch`/`--pause-s` comments record the
#: prod leg as low as ~150/s under a loaded box. Deliberately an
#: UNDER-estimate: the failure direction of "skip a pack that would
#: actually have fit" is cheap (it just stays READY for the next run);
#: the failure direction of "take a pack that doesn't fit" is a mid-pack
#: SIGKILL, the exact incident this whole task exists to close.
ESTIMATED_CONTENT_PACK_OPS_PER_SECOND = 150.0


#: Config, not a literal: neither number is measured for this project's
#: largest packs yet (memory `aw-heavy-rewrite-degrades-prod` names the
#: failure mode, not a validated batch size). Overridable on the CLI.
DEFAULT_BATCH = 2000
DEFAULT_PAUSE_S = 1.0











def __getattr__(name):
    # `PROD_LOCK_HOST` stays readable (other drivers read `Q.PROD_LOCK_HOST`), resolved lazily
    # so that importing this module never needs a configured prod host.
    if name == "PROD_LOCK_HOST":
        return prod_lock_host()
    if name == "DEFAULT_QUEUE_DIR":
        return default_queue_dir()
    raise AttributeError(name)


# task 2897 (RULING 80): the drain split, step 7. Re-exported so each importer and each monkeypatch
# on this module keeps resolving here.
from cytir.index.drain.plan import (  # noqa: E402,F401
    dry_plan, language_scope_and_waiver_exclusions, LOOKUP_WORKERS, mechanical_content_rows,
    mechanical_content_status, payload_profile_preflight_both, persisted_index_is_stale,
    PERSISTED_INDEX_ROOTS, plan_kwargs, plan_kwargs_with_scope, refresh_persisted_index
)





# task 2897 (RULING 80): the drain split, step 8.
from cytir.index.drain.verify import (  # noqa: E402,F401
    _already_landed_overlap_note, _read_ledger_twin_counts, _readback_sample,
    _revalidate_prod_collection, landed_with_twin_drops, outside_twin_set_reason,
    resolve_prod_alias, twin_set_matches_prod_pack_ids, verify_written_form, VERIFY_FORM_MAX,
    VERIFY_FORM_SAMPLE
)

# task 2893 (RULING 80): the drain split, step 6. Re-exported so each importer and each monkeypatch
# on this module keeps resolving here.
from cytir.index.drain.content_checks import (  # noqa: E402,F401
    _canonical_point_id, _chunk_epoch, _content_pack_dry_check, _export_row_problem, _is_card_row,
    _norm_source_type, _snippet_words, _verify_pack_file, BINARY_CONTENT, BINARY_FILE_HOSTS,
    BINARY_FILE_URL_RE, CANONICAL_DUPLICATE, CARD_GUARD_FIELDS, check_binary_content,
    check_canonical_duplicates, check_card_overwrites, check_generation_siblings,
    check_restore_rollback, check_rolled_back_pack, check_soft_404,
    check_superseded_generation_pack, content_pack_dry_check, COVERAGE_VIOLATION_SHARE_MAX,
    DEFAULT_RETAIN_MONTHS, DEFAULT_TOMBSTONE_WORD_COVERAGE_FLOOR, EXPORT_BACKED_REASONS,
    fetch_collection_sparse_names, fetch_collection_vector_config, GENERATION_SIBLING_FIELDS,
    GENERATION_SIBLING_URL_BATCH, HELD_SAME_URL, identity_is_weak, is_binary_file_url,
    load_content_pack_points, load_pack_waivers, load_tombstones, RESTORE_ROLLBACK,
    RETAIN_MONTHS_ENV, ROLLBACK_ENTITY_KEYS, ROLLBACK_IDENTITY_KEYS, row_identity,
    served_group_members, SOFT_404, soft_404_second_gate, SUPERSEDED_GENERATION,
    TOMBSTONE_MISSING_SHARE_MAX, TOMBSTONE_REASONS, validate_tombstone_word_coverage,
    validate_tombstones, validate_vector_shapes, verify_removed_rows_export
)


def consent_flag_census(payloads) -> dict:
    """Task 684 tripwire: COUNT the rows that reach the drain without consent flags. Never
    refuses (READY packs keep flowing); the receipt carries it and the daily digest shows
    it, so a writer that stops stamping is visible the day it happens."""
    rows = missing_both = missing_any = 0
    for p in payloads:
        rows += 1
        s, t = isinstance(p.get("search_ok"), bool), isinstance(p.get("train_ok"), bool)
        missing_both += (not s and not t)
        missing_any += (not s or not t)
    return {"rows": rows, "missing_both_flags": missing_both, "missing_any_flag": missing_any}


# ------------------------------------------------------------- apply a side
# task 2902 (RULING 80): the drain split, step 10.
from cytir.index.drain.write_sealed import (  # noqa: E402,F401
    _single_persisted_check, apply_side, live_apply, rehearse, single_apply
)

# task 2911 (RULING 80): the drain split, step 11.
from cytir.index.drain.write_content_pack import (  # noqa: E402,F401
    apply_content_pack_side, live_apply_content_pack
)

# task 2922 (RULING 80): the drain split, step 12.
from cytir.index.drain.write_patch import (  # noqa: E402,F401
    _drain_payload_patch, live_apply_payload_patch
)
from cytir.index.drain.resume import (  # noqa: E402,F401
    resume_content_pack_prod_only, resume_prod_only, settle_supersede_deleting,
)

# task 2897 (RULING 80): the drain split, step 9.
from cytir.index.drain.markers import (  # noqa: E402,F401
    finalize_carried_by_flip, mark_carried_by_flip, supersede_pack
)

# --------------------------------------------------------------- the drain
#: `--full-persisted-index` DELETES the sqlite index files before rebuilding; a dry run must never.
FULL_INDEX_NEEDS_APPLY = (
    "--full-persisted-index deletes the persisted-input index files, so it needs --apply; "
    "a dry run refuses it before anything is deleted")


# task 2957 (RULING 80): the drain split, step 14. `drain` and `_settle_guard` live in `cytir.index.drain.drain`;
# re-exported so each importer and each monkeypatch on this module keeps resolving here (the moved bodies
# look the patched names up on this module at call time).
from cytir.index.drain.drain import _settle_guard, drain  # noqa: E402,F401




# --------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--queue-dir", default=None,
                    help="promotion queue (default: the operator setting queue_dir)")
    ap.add_argument("--twin-url", default=TWIN.url())
    ap.add_argument("--prod-url", default="http://127.0.0.1:6334",
                    help="the aw-prod-tunnel.service forward "
                         "(local :6334 -> the serving box's qdrant, see the "
                         "operator's unit) to the REAL prod "
                         "qdrant. Was 16333 -- task 664, 2026-09-26: that "
                         "port has had no listener on the build box since at "
                         "least the aw_e8 flip (many other pipeline/*.py "
                         "scripts still default to 16333, a stale reference "
                         "to the pre-flip aw30m_v2 tunnel; NOT fixed here, "
                         "flagged as a follow-up -- this is the one caller "
                         "that silently diverges twin from PROD when it's "
                         "wrong). Found live: a real --apply run's PROD leg "
                         "ABORTED mid-write on ConnectionRefusedError; twin "
                         "was 200,616 points ahead of prod at the time.")
    ap.add_argument("--task", default="641")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--pause-s", type=float, default=DEFAULT_PAUSE_S)
    ap.add_argument("--artifact-root", default=None,
                    help="default: <queue-dir>/_artifacts")
    ap.add_argument("--gate-contract", default=str(GATE_CONTRACT_PATH),
                    help="the ONE tracked field this driver's switch reads "
                         "(default: data/promotion-gate-contract.v1.json)")
    ap.add_argument("--apply", action="store_true",
                    help="without this, every pack is proved (dry run) and "
                         "nothing is written, regardless of gate state")
    ap.add_argument("--dry-receipt-dir", default=None,
                    help="without --apply: write each pack's full dry-run receipt here "
                         "(must lie outside --queue-dir). A dry run never writes into the "
                         "queue; without this flag it prints a one-line summary per pack.")
    ap.add_argument("--rehearse", metavar="PACK_ID", default=None,
                    help="apply ONE pack to the twin only, then roll it back; "
                         "bypasses the gate switch; never touches prod")
    ap.add_argument("--supersede", metavar="PACK_ID", default=None,
                    help="retire ONE queued pack WITHOUT landing it -- a "
                         "proven duplicate or otherwise wrong pack (task "
                         "669). Requires --reason. Refuses a pack already "
                         "DONE (landed); never touches twin or prod.")
    ap.add_argument("--mark-carried-by-flip", metavar="PACK_ID", default=None,
                    help="retire ONE queued pack whose twin leg already "
                         "landed and verified for real, without writing "
                         "its prod leg -- an upcoming flip will make prod "
                         "equal the twin anyway (task 673). Requires "
                         "--reason. Refuses a pack already DONE; never "
                         "touches twin or prod.")
    ap.add_argument("--finalize-carried-by-flip", metavar="PACK_ID", default=None,
                    help="promote ONE CARRIED-BY-FLIP pack to DONE, after "
                         "verifying (read-only, against the collection "
                         "--prod-alias currently resolves to on "
                         "--prod-url) that prod now has every id actually "
                         "written with a matching content_digest, and "
                         "none of the tombstoned ids -- the intended way "
                         "out once the flip that was promised has "
                         "actually happened (task 673). Refuses if the "
                         "pack isn't CARRIED-BY-FLIP or the verification "
                         "fails; never writes a point or a delete.")
    ap.add_argument("--prod-alias", default="aw_prod",
                    help="the qdrant alias prod is served through -- "
                         "resolved live via GET /aliases, never taken "
                         "from a pack's own (possibly stale, pre-flip) "
                         "manifest collection field. Only used by "
                         "--finalize-carried-by-flip.")
    ap.add_argument("--reason", default=None,
                    help="required with --supersede or "
                         "--mark-carried-by-flip: why this pack is "
                         "retired, goes straight into its receipt")
    ap.add_argument("--budget-seconds", type=float, default=None,
                    help="the drain's own systemd TimeoutStartSec/"
                         "TimeoutStartUSec (promote_nightly.sh reads the "
                         "real value with `systemctl show ... -p "
                         "TimeoutStartUSec` rather than hard-coding it) -- "
                         "when set, guards the persisted-index refresh, "
                         "each new pack (estimated from its own op count "
                         "against --stop-margin-s), and the plain "
                         "remaining-budget check, rather than let a "
                         "mid-pack SIGKILL happen (task 673: the "
                         "weekend-stall shape that orphaned "
                         "l1-backfill-batch-00001 on twin). Default None: "
                         "no budget, unchanged behaviour (every existing "
                         "caller, and every test).")
    ap.add_argument("--started-at-epoch", type=float, default=None,
                    help="unix epoch seconds when the OUTER budget's clock "
                         "actually started (promote_nightly.sh captures "
                         "this before its own git fetch/reset, matching "
                         "systemd's TimeoutStartUSec clock as closely as "
                         "possible). Default None: this call's own start "
                         "time -- correct for a manual/interactive run, "
                         "wrong for a wrapper that does real work first.")
    ap.add_argument("--stop-margin-s", type=float, default=5400.0,
                    help="minimum remaining budget (seconds) required to "
                         "start a new pack, or to even attempt the "
                         "persisted-index refresh; default 5400 (90 min). "
                         "Only meaningful with --budget-seconds set.")
    ap.add_argument("--no-refresh-persisted-index", action="store_true",
                    help="skip the automatic persisted-input index "
                         "staleness check/rebuild at the start of a drain "
                         "(default: on -- a stale index false-refuses real "
                         "packs, task 641 2026-09-23). Use for a quick "
                         "interactive dry run against a known-fresh index; "
                         "the nightly window never passes this.")
    ap.add_argument("--full-persisted-index", action="store_true",
                    help="force a FULL persisted-input index rebuild (the "
                        "on-disk scan cache under _persisted-index/"
                        ".scan-cache is ignored, though it is still "
                        "refreshed for the next run). Default is "
                        "incremental -- only new/changed store files are "
                        "re-scanned. Use this 'on request', e.g. after "
                        "suspecting a stale/corrupt cache entry; a "
                        "registry edit already forces it automatically.")
    ap.add_argument("--only", metavar="PACK_ID", action="append", default=None,
                    help="drain exactly this pack (repeatable; order given "
                         "IS the run order, not resorted) instead of every "
                         "READY pack in --queue-dir. For running a subset "
                         "ahead of, or instead of, the rest of the queue.")
    ap.add_argument("--twin-parallel", type=int, default=1,
                    help="fire this many upsert batches concurrently on the "
                         "TWIN leg only (task 641, 2026-09-23 SE throughput "
                         "fix); prod always stays sequential at --pause-s. "
                         "Default 1 (unchanged, sequential) for every "
                         "existing caller.")
    ap.add_argument("--target-url", default=None,
                    help="SINGLE transaction (`cyt index apply`): apply the sealed packs to ONE "
                         "collection at this Qdrant, with no twin leg, no writer locks and no "
                         "prod alias. Needs --target-collection. --twin-url/--prod-url/"
                         "--prod-alias/--twin-parallel/--pause-s are not used.")
    ap.add_argument("--target-collection", default=None,
                    help="the collection a --target-url transaction writes (must exist)")
    ap.add_argument("--target-payload-profile-of", default=None, metavar="DECLARED_COLLECTION",
                    help="the DECLARED collection (data/collection-payload-profiles.v1.json) whose "
                         "payload form --target-collection stores; default: its own declaration")
    ap.add_argument("--target-state-dir", default=None,
                    help="where a --target-url transaction keeps its receipts, refusal cache and "
                         "DONE markers (default: <artifact-root>/single-state/<collection>); "
                         "never the queue's pack dirs, which the twin-first drain reads")
    ap.add_argument("--target-parallel", type=int, default=1,
                    help="concurrent upsert batches on the --target-url collection")
    ap.add_argument("--repeat-while-landing", action="store_true",
                    help="after a pass that LANDED at least one pack, scan the "
                         "queue again (task 689): packs sealed while the pass "
                         "ran land in this same run instead of waiting for the "
                         "next hourly timer firing. Stops on the first pass "
                         "that lands nothing; every pass obeys the same "
                         "--budget-seconds clock. Default off.")
    args = ap.parse_args(argv)

    if (args.target_url or args.target_collection) and not args.queue_dir:
        raise SystemExit("[REFUSED] a --target-url transaction needs an explicit --queue-dir (a "
                         "build queue of its own); it never defaults to the promotion queue")
    queue_dir = pathlib.Path(args.queue_dir or default_queue_dir())
    # task 696 slice 4: say where the queue and the write locks are, and why (never raises).
    print(PATHS.location_line("queue_dir", str(queue_dir), "queue_dir"), flush=True)
    print(PATHS.lock_location_line(), flush=True)
    artifact_root = pathlib.Path(args.artifact_root or (queue_dir / "_artifacts"))
    contract_path = pathlib.Path(args.gate_contract)

    if args.rehearse:
        pack_dir = queue_dir / args.rehearse
        if not pack_dir.is_dir():
            raise SystemExit(f"[REFUSED] no such pack dir: {pack_dir}")
        result = rehearse(pack_dir, twin_url=args.twin_url, task=args.task,
                          batch=args.batch, pause_s=args.pause_s,
                          artifact_root=artifact_root)
        print(json.dumps(result, indent=1))
        return 0 if result.get("status") == "REHEARSED-AND-ROLLED-BACK" else 1

    if args.supersede:
        if not args.reason:
            raise SystemExit("[REFUSED] --supersede requires --reason")
        pack_dir = queue_dir / args.supersede
        if not pack_dir.is_dir():
            raise SystemExit(f"[REFUSED] no such pack dir: {pack_dir}")
        receipt = supersede_pack(pack_dir, reason=args.reason, task=args.task)
        print(json.dumps(receipt, indent=1))
        return 0

    if args.mark_carried_by_flip:
        if not args.reason:
            raise SystemExit("[REFUSED] --mark-carried-by-flip requires --reason")
        pack_dir = queue_dir / args.mark_carried_by_flip
        if not pack_dir.is_dir():
            raise SystemExit(f"[REFUSED] no such pack dir: {pack_dir}")
        receipt = mark_carried_by_flip(pack_dir, reason=args.reason, task=args.task)
        print(json.dumps(receipt, indent=1))
        return 0

    if args.finalize_carried_by_flip:
        pack_dir = queue_dir / args.finalize_carried_by_flip
        if not pack_dir.is_dir():
            raise SystemExit(f"[REFUSED] no such pack dir: {pack_dir}")
        receipt = finalize_carried_by_flip(pack_dir, prod_url=args.prod_url,
                                           task=args.task,
                                           artifact_root=artifact_root,
                                           prod_alias=args.prod_alias,
                                           twin_url=args.twin_url)
        print(json.dumps(receipt, indent=1))
        return 0

    _live = live_queue_dir()
    is_single = bool(args.target_url or args.target_collection)
    if is_single and _live is not None and queue_dir.resolve() == _live:
        raise SystemExit("[REFUSED] a --target-url transaction refuses the live promotion "
                         "queue: pass an explicit --queue-dir (a build queue of its own)")
    # Fail at START without the operator settings the run needs, BEFORE the queue is opened and
    # before any dry-run or apply logic matters. With an explicit --queue-dir (every live unit
    # passes one) an unconfigured box used to mark EACH pack REFUSED at `live-apply-preflight`,
    # write a receipt, and exit 0 -- a green unit that landed nothing, with the refusals cached
    # (review of #567, round 2, item 2). A dry drain refuses by name too: it reads the queue and the
    # persisted store for its plan, and a plan made against no settings is not a plan.
    # The twin-first transaction needs all of that. A `Single` (a stranger's or a scratch build)
    # has no prod, no twin and no operator: it is asked only for what it reads (round 6, after #553).
    if is_single:
        codes = [single_startup_refusal(apply=args.apply)]
    else:
        codes = [plan_startup_refusal(), GPG.startup_refusal()]
        if args.apply:
            codes.append(apply_startup_refusal())
    if any(codes):
        return 2
    # Task 717: the prod term set is loaded before the first twin PUT (not in `PSP.upsert` on the
    # prod leg, after the twin was written). A dry drain writes nothing and needs no prod.
    if args.apply and not is_single and sparse_rule_startup_refusal(args.prod_url, args.prod_alias):
        return 2

    if not queue_dir.is_dir():
        raise SystemExit(f"[REFUSED] no such queue dir: {queue_dir}")
    if args.full_persisted_index and not args.apply:
        raise SystemExit(f"[REFUSED] {FULL_INDEX_NEEDS_APPLY}")
    # Which transaction: the default is twin-first against --twin-url/--prod-url, exactly the call
    # this driver always made; --target-url/--target-collection select the Single transaction.
    transaction_kwargs: dict = {"twin_url": args.twin_url, "prod_url": args.prod_url}
    if args.target_url or args.target_collection:
        if not (args.target_url and args.target_collection):
            raise SystemExit("[REFUSED] --target-url and --target-collection go together")
        transaction_kwargs = {"transaction": Single(
            url=args.target_url, collection=args.target_collection,
            parallel=args.target_parallel, payload_profile_of=args.target_payload_profile_of,
            state_dir=pathlib.Path(args.target_state_dir) if args.target_state_dir else None,
            prod_alias=args.prod_alias)}
    elif args.target_payload_profile_of:
        raise SystemExit("[REFUSED] --target-payload-profile-of needs --target-url")

    def one_pass() -> int:
        return _drain_pass(
            drain(queue_dir, **transaction_kwargs,
                  task=args.task, apply=args.apply, batch=args.batch,
                  pause_s=args.pause_s, artifact_root=artifact_root,
                  contract_path=contract_path,
                  refresh_index=not args.no_refresh_persisted_index,
                  full_persisted_index=args.full_persisted_index,
                  only=args.only, twin_parallel=args.twin_parallel,
                  budget_seconds=args.budget_seconds,
                  started_at_epoch=args.started_at_epoch,
                  stop_margin_s=args.stop_margin_s,
                  prod_alias=args.prod_alias,
                  dry_receipt_dir=args.dry_receipt_dir))

    return run_passes(one_pass, repeat_while_landing=args.repeat_while_landing)




def run_passes(one_pass, *, repeat_while_landing: bool) -> int:
    """Run `one_pass()` (returns how many packs it LANDED) once, or, with
    `repeat_while_landing`, again after every pass that landed something and
    stop on the first pass that lands nothing (task 689). The ending pass
    costs a persisted-index top-up and a queue scan, plus the dry check of any
    pack that is not skipped: content refusals at stage `plan` are cached
    (SKIPPED-REFUSED-UNCHANGED), but a refusal at `live-apply-preflight` (for
    example a held TWIN-WRITE-LOCK) is NOT cached and is re-checked every pass.
    The extra pass is what lets a pack sealed during pass N land in this run
    rather than wait for the next timer firing."""
    pass_no = 0
    while True:
        pass_no += 1
        if repeat_while_landing:
            print(f"[drain] pass {pass_no}")
        if one_pass() <= 0 or not repeat_while_landing:
            return 0


def _drain_pass(results: list[dict]) -> int:
    """Print the summary line the watchdog/daily digest read, return `landed`."""
    landed = sum(1 for r in results if r.get("status") == "LANDED")
    blocked = sum(1 for r in results if r.get("status") == "PROVED-BLOCKED")
    refused = sum(1 for r in results if r.get("status") == "REFUSED")
    skipped = sum(1 for r in results if r.get("status") == "SKIPPED-REFUSED-UNCHANGED")
    # Review round 2, finding 2: SKIPPED-BUDGET/BUDGET-EXCEEDS-TIMEOUT must
    # be visible in the summary line the watchdog/daily digest read, not
    # just buried in individual pack receipts -- a budget-starved pack is
    # an operational fact, not a quiet non-event.
    budget_skipped = sum(1 for r in results if r.get("status") == "SKIPPED-BUDGET")
    budget_exceeds = sum(1 for r in results if r.get("status") == "BUDGET-EXCEEDS-TIMEOUT")
    print(f"[drain] {len(results)} pack(s): {landed} landed, {blocked} blocked, "
         f"{refused} refused, {skipped} skipped (refused, unchanged), "
         f"{budget_skipped} skipped (budget), {budget_exceeds} exceed the "
         "timeout entirely")
    # Task 660, 2026-09-25: a REFUSED pack is a normal, expected drain
    # outcome (its own receipt names why) and must NOT mark the systemd
    # unit failed -- that is exactly the head-of-line-blocking symptom this
    # task fixes, just moved from "an uncaught exception mid-loop" to "a
    # clean loop that still exits 1 every run". A pack that failed
    # MID-WRITE (a possible twin/prod divergence) never reaches this line:
    # `drain()` re-raises it, which is an uncaught exception here and exits
    # non-zero on its own -- the one case this driver must still stop for.
    # (main() still returns 0 for every non-raising run: see run_passes.)
    return landed


if __name__ == "__main__":
    # Task 2880: `python3 -m cytir.index.promote_queue` runs this file as `__main__`, a SECOND copy of
    # the module beside the canonical `cytir.index.promote_queue` that `url_group_supersede` imports.
    # Two copies mean two `QueueError` classes and the drain's `except QueueError` misses what
    # url_group_supersede raises. Dispatch into the canonical copy: one copy of every class.
    from cytir.index.promote_queue import main as _canonical_main
    raise SystemExit(_canonical_main())
