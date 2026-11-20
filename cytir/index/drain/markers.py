"""Drain markers: SUPERSEDED / CARRIED-BY-FLIP receipt and marker files, and the finalize of a flip.

Task 2897 (RULING 80, drain split steps 7-9): moved verbatim out of `cytir/index/promote_queue.py`,
which re-exports every name here. A moved body that calls a name a test patches on the facade (or a
name that stays there) reads it at call time through `_Q` (the facade module).
"""
from __future__ import annotations

import hashlib
import json
import pathlib

from cytir.index import change_log as CL
from cytir.index import mirror_compare as MC
from cytir.index import point_delete as PD
from cytir.index import promote_sealed_pack as PSP
from cytir.index.drain.changelog import _log_landing
from cytir.index.drain.common import QueueError, atomic_write_bytes, atomic_write_text, now
from cytir.index.drain.content_checks import load_tombstones
from cytir.index.drain.queue_scope import read_manifest
from cytir.index.drain.receipts import _read_prior_receipt, read_content_pack_twin_state
from cytir.index.drain.verify import _read_ledger_twin_counts


# ---------------------------------------------------------- pack retirement
def supersede_pack(pack_dir: pathlib.Path, *, reason: str, task: str) -> dict:
    """Retire a queued pack WITHOUT landing it -- proven a duplicate or
    otherwise wrong (real case: task 669's `l1-backfill-batch-00001`,
    99.99% of its pack_points ids already present on the twin from a
    prior, already-landed batch). Writes `receipt.json` (`status`:
    `SUPERSEDED`, `reason`, `manifest_sha`) and a `SUPERSEDED` marker file
    the drain loop skips forever -- deliberately NOT `DONE`, which means
    "landed on both sides" (module docstring): a superseded pack never
    lands anywhere, and reusing `DONE` for that would make a future
    reader believe its rows are live when they are not.

    Refuses if the pack already landed (`DONE` present) -- retiring
    something already applied is a rollback, a different and much bigger
    operation this function does not attempt.
    """
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    if (pack_dir / "DONE").exists():
        raise QueueError(
            f"{pack_dir.name}: already DONE (landed on both sides) -- "
            "cannot supersede a pack that already applied")
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        raise QueueError(f"{pack_dir.name}: no manifest.json")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipt = {"schema": _Q.SCHEMA, "pack_id": pack_dir.name, "checked_at": now(),
              "manifest_sha": manifest_sha, "status": "SUPERSEDED",
              "reason": reason, "task": task}
    receipt_bytes = (json.dumps(receipt, indent=1) + "\n").encode()
    atomic_write_bytes(pack_dir / "receipt.json", receipt_bytes)
    atomic_write_text(pack_dir / "SUPERSEDED", receipt["checked_at"] + "\n")
    return receipt


def mark_carried_by_flip(pack_dir: pathlib.Path, *, reason: str, task: str) -> dict:
    """Retire a queued pack whose TWIN leg is real, verified, applied
    content -- not a duplicate -- but whose PROD leg is deliberately not
    being written to the current `aw_e8`, because an imminent flip
    (`aw_e9`) will make prod equal the twin anyway and a several-hundred-
    thousand-op direct write risks repeating the exact OOM history this
    box already has (task 673, `l1-backfill-batch-00001`: 202,052 upserts
    + 131,285 deletes). Deliberately distinct from both `DONE` ("landed on
    both sides") and `SUPERSEDED` ("proven wrong, never landed anywhere")
    -- this pack's rows ARE live, on twin only, on purpose, pending the
    flip. Writes `receipt.json` (`status`: `CARRIED-BY-FLIP`) and a
    `CARRIED-BY-FLIP` marker the drain loop skips forever, same as
    `SUPERSEDED` -- but NOT forever-forever: `finalize_carried_by_flip()`
    is the intended way out, once the flip that was promised has actually
    happened and been verified. There is no automatic transition; a human
    or a follow-up task calls it explicitly (review finding 6 -- this
    docstring previously said "a fresh manifest or an explicit `--only`
    invocation is the only way to touch it again", which was never true:
    `--only` still hits this same `CARRIED-BY-FLIP`-skip check first).

    If a `SUPERSEDED` marker already exists (task 673's exact real case --
    `l1-backfill-batch-00001` was superseded on a duplicate claim this
    disproved), it is removed: a pack cannot correctly carry both markers,
    and this call IS the correction that retracts the wrong one.
    """
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    if (pack_dir / "DONE").exists():
        raise QueueError(
            f"{pack_dir.name}: already DONE (landed on both sides) -- "
            "cannot mark carried-by-flip a pack that already applied to prod")
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        raise QueueError(f"{pack_dir.name}: no manifest.json")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipt = {"schema": _Q.SCHEMA, "pack_id": pack_dir.name, "checked_at": now(),
              "manifest_sha": manifest_sha, "status": "CARRIED-BY-FLIP",
              "reason": reason, "task": task}
    receipt_bytes = (json.dumps(receipt, indent=1) + "\n").encode()
    atomic_write_bytes(pack_dir / "receipt.json", receipt_bytes)
    # NEW marker first, THEN unlink the old one (review round 2, finding
    # 4): a kill between the two steps must never leave a pack with
    # NEITHER marker -- that reads as untouched/fresh to drain()'s
    # top-of-loop checks, and a fresh content_pack_dry_check + live apply
    # on THIS pack's manifest is exactly the direct-write-to-aw_e8 outcome
    # CARRIED-BY-FLIP exists to prevent.
    atomic_write_text(pack_dir / "CARRIED-BY-FLIP", receipt["checked_at"] + "\n")
    superseded_marker = pack_dir / "SUPERSEDED"
    if superseded_marker.exists():
        superseded_marker.unlink()
    return receipt




def finalize_carried_by_flip(pack_dir: pathlib.Path, *, prod_url: str,
                             task: str, artifact_root: pathlib.Path | None = None,
                             prod_alias: str = "aw_prod",
                             twin_url: str | None = None) -> dict:
    """The explicit way out of `CARRIED-BY-FLIP` (task 673 review finding
    6, hardened in round 2 findings 3+4, round 3 findings 3+4): once the
    flip that was promised has actually happened, verify prod for real --
    every id actually WRITTEN present with matching CONTENT (not just
    membership), every tombstoned id absent -- and only then promote it
    to `DONE`. Resolves prod's live collection through the `aw_prod`
    alias (`resolve_prod_alias`), never the pack's own possibly-stale
    manifest `collection`. Read-only against prod (and, for the twin-
    membership reconstruction below, against twin) until verification
    passes; never writes a point or a delete here (the flip itself is
    what's expected to have done that).

    Which ids to check, in priority order:
    1. A twin-write-state artifact for this exact manifest
       (`artifact_root` given and a record found) -- its `pack_ids`/
       `tomb_ids` are the ids ACTUALLY written, excluding any url a
       word-coverage skip dropped at apply time.
    2. No artifact, but `twin_url` given (round 3 finding 4 -- the real
       `l1-backfill-batch-00001` case: it predates the artifact
       mechanism, marked `SUPERSEDED` by a pre-fix drain run): reconstruct
       the actually-written set from TWIN's own LIVE membership instead
       of blindly trusting the full manifest-declared set. Twin is
       definitionally real, verified content for a CARRIED-BY-FLIP pack
       (that classification's own premise), so "present on twin" IS the
       ground truth for what this pack's twin leg actually wrote --
       matches the pack's own apply ledger exactly on the real case
       (`l1-backfill-batch-00001-ledger.jsonl`: `"written": 218208` of
       218,235 declared, `"tombstoned": 131285` -- both counts line up
       with twin's live membership, verified this task). A tombstone id
       still confirmed on twin was never actually deleted there and is
       excluded the same way. Round 4 finding 1: this reconstruction is
       circular on its own (see `_read_ledger_twin_counts`'s docstring) --
       its counts are cross-checked here against the pack's own apply
       ledger (`TWIN-VERIFIED`/role=twin), and finalize REFUSES on any
       mismatch or when no ledger record exists, rather than trusting a
       reconstruction that would trivially "match" a post-flip prod no
       matter how much of the twin leg actually landed.
    3. Neither available: falls back to the pack's full declared
       `pack_points`/`tombstones` files -- an approximation that cannot
       know what a word-coverage skip may have dropped.

    Refuses if the pack isn't `CARRIED-BY-FLIP` (nothing to finalize,
    including an already-`DONE` pack) or if verification fails (missing
    ids, a present row whose CONTENT differs, or a tombstoned id still
    present -- the flip hasn't landed, or landed short, or carried the old
    row: still `CARRIED-BY-FLIP`, retry later). A refusal is kept in the
    pack's `receipt.json` as `finalize_refused` (status unchanged), with the
    first 20 differing ids and their keys.

    Content (task 833, guard G1 of task 831): every payload key and the
    dense vector of each present row, prod AS STORED against the twin's row
    (the pack row when no twin is given) in prod's form -- see
    `cytir.index.mirror_compare`. It replaces a digest compare that, for a
    Tier-B id made from the digest, was a membership check: it finalized
    `l1-backfill-batch-00001` with 2,317 stale rows on prod (task 830).
    """
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    if not (pack_dir / "CARRIED-BY-FLIP").exists():
        raise QueueError(
            f"{pack_dir.name}: not CARRIED-BY-FLIP -- nothing to finalize")
    if (pack_dir / "DONE").exists():
        raise QueueError(f"{pack_dir.name}: already DONE")
    manifest = read_manifest(pack_dir)
    manifest_sha = hashlib.sha256(
        (pack_dir / "manifest.json").read_bytes()).hexdigest()

    state = (read_content_pack_twin_state(artifact_root, pack_dir.name,
                                          manifest_sha=manifest_sha)
             if artifact_root is not None else None)
    all_points = _Q.load_content_pack_points(manifest["pack_points"])
    all_tombstones = (load_tombstones(manifest["tombstones"])
                      if manifest.get("tombstones") else [])
    if state is not None and state.get("pack_ids") is not None:
        keep_ids = set(state["pack_ids"])
        points = [p for p in all_points if p["id"] in keep_ids]
        tomb_ids = list(state.get("tomb_ids") or [])
        ids_source = "twin-write-state"
    elif twin_url is not None:
        twin_collection = manifest["collection"]
        all_ids = [p["id"] for p in all_points]
        present_on_twin = set(PSP.already_present(twin_url, twin_collection, all_ids)) \
            if all_ids else set()
        points = [p for p in all_points if p["id"] in present_on_twin]
        all_tomb_ids = [t["id"] for t in all_tombstones]
        still_on_twin = set(PSP.already_present(twin_url, twin_collection, all_tomb_ids)) \
            if all_tomb_ids else set()
        tomb_ids = [t for t in all_tomb_ids if t not in still_on_twin]
        ids_source = "twin-membership-reconstructed"
        # Round 4 finding 1: this reconstruction alone is circular against
        # a post-flip prod (a copy of twin) -- cross-check it against what
        # the twin leg's OWN apply ledger recorded at write time, before
        # any resume/flip/reconstruction could have touched it.
        if artifact_root is None:
            raise QueueError(
                f"{pack_dir.name}: twin-membership reconstruction requires "
                "artifact_root to cross-check the pack's apply ledger -- "
                "refusing to finalize without it")
        ledger_counts = _read_ledger_twin_counts(artifact_root, pack_dir.name)
        if ledger_counts is None:
            raise QueueError(
                f"{pack_dir.name}: no TWIN-VERIFIED apply-ledger record "
                f"found ({artifact_root / f'{pack_dir.name}-ledger.jsonl'}) "
                "-- cannot verify the twin-membership reconstruction "
                "against what the twin leg actually wrote; refusing to "
                "finalize")
        ledger_written, ledger_tombstoned = ledger_counts
        if len(points) != ledger_written or len(tomb_ids) != ledger_tombstoned:
            raise QueueError(
                f"{pack_dir.name}: twin-membership reconstruction "
                f"({len(points)} written, {len(tomb_ids)} tombstoned) does "
                f"not match the pack's own apply ledger ({ledger_written} "
                f"written, {ledger_tombstoned} tombstoned) -- the twin leg "
                "may be only partly applied; refusing to finalize")
    else:
        points = all_points
        tomb_ids = [t["id"] for t in all_tombstones]
        ids_source = "manifest-declared"
    pack_ids = [p["id"] for p in points]

    collection = _Q.resolve_prod_alias(prod_url, prod_alias)

    present = set(PSP.already_present(prod_url, collection, pack_ids)) \
        if pack_ids else set()
    missing = set(pack_ids) - present
    still_present = set(PSP.already_present(prod_url, collection, tomb_ids)) \
        if tomb_ids else set()

    checkable = sorted(present)
    content: dict = {"checked": 0, "equal": 0, "mismatched": 0, "by_key": {}, "examples": []}
    reference = None
    if checkable:
        # Task 833 (G1): the full row and the dense vector, not a digest (see the note above
        # `resolve_prod_alias`). Prod is read by the SAME id retrieves as before (1,000 ids per
        # call, the count `fetch_payload_fields` made), now with the dense vector: no extra prod
        # call. The reference is the TWIN's stored row when a twin is given (the flip promised
        # prod == twin), else the pack row (what the twin leg wrote). Either way it is put in
        # prod's form by the writer's own steps; prod's row is compared AS STORED. The one
        # allowance: a flip-carried row may lack the BM25 epoch stamp (`mirror_compare`).
        prod_rows = PD.fetch_rows(prod_url, collection, checkable, batch=1000)
        if twin_url is not None:
            ref_rows = PD.fetch_rows(twin_url, manifest["collection"], checkable, batch=1000)
            reference = f"twin {manifest['collection']}"
        else:
            ref_rows = {p["id"]: p for p in points}
            reference = "pack rows (no twin given)"
        content = MC.Comparator(collection, allow_unstamped_rule_epochs=True).compare(
            checkable, ref_rows, prod_rows)
        content["reference"] = reference

    if missing or still_present or content["mismatched"]:
        refused = {"at": now(), "task": task, "collection_checked": collection,
                   "prod_alias": prod_alias, "ids_source": ids_source,
                   "missing": len(missing), "missing_examples": sorted(missing)[:MC.EXAMPLES],
                   "tomb_still_present": len(still_present),
                   "tomb_still_present_examples": sorted(still_present)[:MC.EXAMPLES],
                   "content": content}
        # The refusal is kept in the pack's receipt, status unchanged (still CARRIED-BY-FLIP):
        # the first differing ids and their keys, for whoever retries.
        prior = _read_prior_receipt(pack_dir) or {}
        _Q.write_pack_receipt(pack_dir, dict(prior, finalize_refused=refused))
        raise QueueError(
            f"{pack_dir.name}: flip verification failed against "
            f"{collection!r} (resolved from alias {prod_alias!r}) -- "
            f"{len(missing)} of {len(pack_ids)} pack id(s) missing, "
            f"{content['mismatched']} of {content['checked']} present row(s) differ by content "
            f"from the {reference} (keys {content['by_key']}; first: {content['examples'][:5]}), "
            f"{len(still_present)} of {len(tomb_ids)} tombstone id(s) still present. Still "
            "CARRIED-BY-FLIP; not finalizing.")
    receipt = {"schema": _Q.SCHEMA, "pack_id": pack_dir.name, "checked_at": now(),
              "status": "DONE", "finalized_from": "CARRIED-BY-FLIP",
              "task": task, "collection_checked": collection,
              "prod_alias": prod_alias, "pack_ids_verified": len(pack_ids),
              "tomb_ids_verified": len(tomb_ids),
              "content_checked": content["checked"],
              "content_compare": {"compared": "all payload keys + dense",
                                  "reference": reference, "equal": content["equal"]},
              "ids_source": ids_source}
    _Q.write_pack_receipt(pack_dir, receipt)
    # DONE first, THEN unlink CARRIED-BY-FLIP (round 2 finding 4): a kill
    # between the two steps must never leave neither marker -- that would
    # read as untouched to drain()'s top checks, and this pack's rows
    # were deliberately never force-written directly, so orphaning it
    # back to "fresh" risks the exact ~333k-op direct write CARRIED-BY-
    # FLIP exists to avoid.
    # Task 2314: a flip makes the old base an invalid ancestor -- one `rebase` line, never an id list.
    _log_landing(dict(receipt, status="LANDED"), {"rebase": "flip"},
                 manifest_sha=CL.receipt_sha(receipt))
    atomic_write_text(pack_dir / "DONE", receipt["checked_at"] + "\n")
    (pack_dir / "CARRIED-BY-FLIP").unlink()
    return receipt
