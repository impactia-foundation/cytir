"""Drain receipts: receipt, twin-state, in-place-ids and held-out file helpers.

Task 2875 (RULING 80): moved verbatim out of `cytir/index/promote_queue.py`, which re-exports
every name here.
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import sys

from cytir.index import chunk_checkpoint as CK
from cytir.index import dry_run as DRY
from cytir.index import prod_guard as PG
from cytir.index import promote_sealed_pack as PSP
from cytir.index.drain.common import atomic_write_text, now


HELD_OUT_CARDS_FILE = "held-out-cards.jsonl"


HELD_CARD_OVERWRITE = "card_overwrite"     # the guard's own reason (#613)


def held_out_receipt_fields(held: dict) -> dict:
    """The receipt fields for a pack's held rows (fresh check)."""
    rows = held["rows"]
    by = {}
    for h in rows:
        by[h["code"]] = by.get(h["code"], 0) + 1
    out = {"held_out_cards": by.get(HELD_CARD_OVERWRITE, 0), "held_out_rows": len(rows),
           "held_out_by_reason": by}
    if rows:
        out["held_out_cards_reasons"] = [h["reason"] for h in rows if h["code"] == HELD_CARD_OVERWRITE]
        out["held_out_cards_tombstones"] = held["tombstone_ids"]
    return out


def read_held_out_receipt_fields(pack_dir: pathlib.Path) -> dict:
    """The same fields read back from the sidecar, for a RESUME receipt (S2): a resumed pack
    never runs the check again, but its receipt must still carry the count."""
    path = pack_dir / HELD_OUT_CARDS_FILE
    if not path.exists():
        return {}
    rows, tombs = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        (tombs.append(d["id"]) if d["kind"] == "tombstone"
         else rows.append({"code": d["code"], "reason": d["reason"]}))
    return held_out_receipt_fields({"rows": rows, "tombstone_ids": tombs})


def write_held_out_cards(pack_dir: pathlib.Path, held: dict) -> None:
    """Task 728: the rows the drain did NOT write because of a card hold, kept beside the pack so
    a census finds them (`<pack>/held-out-cards.jsonl`; one JSON object per line: `kind` `point`
    with the whole point, its `code` and the reason, or `kind` `tombstone` with the dropped id).
    Never silently dropped. `atomic_write_bytes` refuses under a dry run."""
    lines = [json.dumps({"kind": "point", "code": h["code"], "reason": h["reason"],
                         "point": h["point"]}, sort_keys=True) for h in held["rows"]]
    lines += [json.dumps({"kind": "tombstone", "id": i,
                          "reason": "its url has a held-out card row: no replacement lands"},
                         sort_keys=True) for i in held["tombstone_ids"]]
    atomic_write_text(pack_dir / HELD_OUT_CARDS_FILE, "\n".join(lines) + "\n")


@DRY.refuses_when_dry("remove a stale held-out sidecar")
def drop_held_out_cards(pack_dir: pathlib.Path) -> None:
    """A later refusal of the pack leaves no sidecar from an earlier attempt (S1)."""
    (pack_dir / HELD_OUT_CARDS_FILE).unlink(missing_ok=True)


def prod_chunk_checkpoint(artifact_root: pathlib.Path, pack_id: str, manifest_sha: str | None,
                          step: str) -> "CK.ChunkCheckpoint":
    """The chunk-level resume record of one prod step of one pack (task 786). `step` is `lookup` (the
    pre-write presence check) or `write` (the upsert batches). The manifest digest is in the name, so a
    resealed pack never reads an older pack's record."""
    return CK.ChunkCheckpoint(
        artifact_root / f"{pack_id}-{(manifest_sha or 'nosha')[:12]}-prod-{step}.chunks.jsonl")


@contextlib.contextmanager
def lookup_record_scope(ck):
    """A lookup checkpoint outlives ONLY a lookup that failed (task 787, must-fix 2). Any other exit
    (a refusal raised after the lookup, e.g. AlreadyLandedError) spends it: otherwise every later attempt
    would read prod presence from the file and not from prod."""
    if ck is None:
        yield
        return
    try:
        yield
    except BaseException as exc:
        if not getattr(exc, "lookup_failed", False):
            ck.clear()
        raise
    else:
        ck.clear()


def prod_call_receipt(rows: int, before: dict) -> dict | None:
    """What this prod leg cost, for the receipt (task 786): ids per lookup request, chunk count, retries,
    the longest single prod call. Only for a big pack or a leg that had to retry, so the receipt of an
    ordinary pack is byte-for-byte what it was."""
    now_ = PSP.PROD_STATS.snapshot()
    delta = {k: (now_[k] - before[k]) if isinstance(now_[k], (int, float)) and k != "longest_call_s"
             else now_[k] for k in now_ if k != "retried_statuses"}
    # per leg, not per process: `before` came from `PROD_STATS.mark()`, which also reset the longest call
    delta["retried_statuses"] = {k: n - before["retried_statuses"].get(k, 0)
                                 for k, n in now_["retried_statuses"].items()
                                 if n - before["retried_statuses"].get(k, 0)}
    if not (PG.is_big_pack(rows) or delta["retries"]):
        return None
    return {"rows": rows, "lookup_ids_per_request": PSP.ALREADY_PRESENT_BATCH,
            "write_pause_s_floor": PG.PROD_BIG_PACK_PAUSE_S if PG.is_big_pack(rows) else None,
            **delta}


def twin_written_ids_path(artifact_root: pathlib.Path, pack_id: str) -> pathlib.Path:
    """The per-pack artifact a SEALED pack's twin leg leaves (`apply_side`), which the prod-only
    resume reads. One home for the name: the drain's two read sites and the compose-time
    resume guard (`cytir.index.served_exclusion`) all ask this function."""
    return artifact_root / f"{pack_id}-twin-written-ids.json"


def content_pack_twin_state_path(artifact_root: pathlib.Path, pack_id: str) -> pathlib.Path:
    """Per-pack resumability record for the CONTENT-PACK apply path (task
    673). A sealed pack gets a `{pack_id}-twin-written-ids.json` as a
    byproduct of `apply_side()` -- but `apply_content_pack_side()` (upsert
    + tombstone-delete in one call) writes no such artifact, so a SIGKILL
    between the twin leg and the prod leg left NOTHING to resume from. This
    is the exact shape that orphaned `l1-backfill-batch-00001`'s 202,052
    rows on twin with no trace anywhere (task 673, 2026-09-29): the
    2026-09-26/28 weekend drain stall SIGKILLed a cycle by systemd's 6h
    `TimeoutStartSec` after the twin leg wrote and verified but before the
    prod leg ever started, and no receipt, marker, or ids-file existed
    anywhere to say so -- task 669 later found the content "already on
    twin" and, with nothing recorded to explain WHY, retired the pack as a
    presumed duplicate instead of resuming it."""
    return artifact_root / f"{pack_id}-content-pack-twin-state.json"


def write_content_pack_twin_state(artifact_root: pathlib.Path, pack_id: str, *,
                                  stage: str, manifest_sha: str | None,
                                  pack_ids: list[str] | None = None,
                                  tomb_ids: list[str] | None = None) -> pathlib.Path:
    """Write/overwrite the resumability record. `stage` is
    `"twin-write-started"` (written BEFORE `apply_target("twin", ...)`
    runs) or `"twin-write-complete"` (written AFTER `verify_target("twin",
    ...)` passes) -- a SIGKILL any time in between leaves the `-started`
    record behind, which the drain treats as "no usable record" (the twin
    write is idempotent, so it just redoes it) rather than trying to guess
    a half-finished state."""
    path = content_pack_twin_state_path(artifact_root, pack_id)
    record: dict = {"schema": "aw.content-pack-twin-state.v1", "pack_id": pack_id,
                    "manifest_sha": manifest_sha, "stage": stage, "at": now()}
    if pack_ids is not None:
        record["pack_ids"] = pack_ids
    if tomb_ids is not None:
        record["tomb_ids"] = tomb_ids
    atomic_write_text(path, json.dumps(record, indent=1) + "\n")
    return path


def read_content_pack_twin_state(artifact_root: pathlib.Path, pack_id: str,
                                 *, manifest_sha: str | None) -> dict | None:
    """Return the record for THIS exact manifest (a resealed pack's old
    record must never be reused -- its content changed), whatever its
    `stage` -- `"twin-write-started"` (a kill happened somewhere inside
    the twin leg, upsert or delete, in progress or with the leg's own
    verify not yet run) or `"twin-write-complete"` (twin verified; only
    the prod leg remains). Review finding 2 (task 673): treating
    `-started` as "no usable record" sent the caller back to a fresh
    `content_pack_dry_check()`, which re-validates tombstones against
    twin and either silently drops the ones the twin leg already deleted
    (<=20% share) or refuses the whole pack (>20%) -- a refusal that then
    caches forever. The caller (`drain()`) must branch on `stage` and, in
    EITHER case, replay from these ids rather than re-derive anything.
    `None` only for a missing file, unreadable/corrupt JSON (`review
    finding 3`: a same-filesystem atomic rename makes this rare now, but a
    reader must never assume a writer upstream got it right), or a
    manifest_sha mismatch."""
    path = content_pack_twin_state_path(artifact_root, pack_id)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (record.get("stage") not in ("twin-write-started", "twin-write-complete")
            or record.get("manifest_sha") != manifest_sha):
        return None
    return record


INPLACE_IDS_SCHEMA = "aw.content-pack-inplace-ids.v1"


def inplace_ids_path(artifact_root: pathlib.Path, pack_id: str) -> pathlib.Path:
    """The per-pack list of IN-PLACE ids (task 833, guard G2): ids a content pack OVERWROTE, per
    leg, because the target already held them before the write. A DONE content pack otherwise keeps
    only counts; an audit of the in-place class (G3, `cyt ops mirror-audit`) needs the list."""
    return artifact_root / f"{pack_id}-inplace-ids.json"


def record_inplace_ids(artifact_root: pathlib.Path, pack_id: str, *, role: str, collection: str,
                       manifest_sha: str | None, ids) -> list[str]:
    """Add `role`'s in-place ids (the pre-write presence lookup of that leg) to the artifact and
    return the ids the artifact holds for `role`. The FIRST record of a role for this manifest is
    kept: a retry's lookup would count the rows the failed attempt already wrote as in-place. A
    record of another manifest is replaced whole."""
    path = inplace_ids_path(artifact_root, pack_id)
    record: dict = {}
    if path.exists():
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            record = {}
    if record.get("schema") != INPLACE_IDS_SCHEMA or record.get("manifest_sha") != manifest_sha:
        record = {"schema": INPLACE_IDS_SCHEMA, "pack_id": pack_id, "manifest_sha": manifest_sha}
    if isinstance(record.get(role), dict):
        return list(record[role].get("ids") or [])
    record[role] = {"collection": collection, "at": now(), "count": len(ids), "ids": sorted(ids)}
    atomic_write_text(path, json.dumps(record, indent=1) + "\n")
    return record[role]["ids"]


def read_inplace_ids(artifact_root: pathlib.Path, pack_id: str, *,
                     manifest_sha: str | None) -> tuple[list[str], str]:
    """(in-place ids, which leg recorded them) for THIS manifest: the prod leg's own lookup when it
    ran, else the twin leg's (a mirror holds the same ids), else ([], "none")."""
    path = inplace_ids_path(artifact_root, pack_id)
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return [], "none"
    if record.get("schema") != INPLACE_IDS_SCHEMA or record.get("manifest_sha") != manifest_sha:
        return [], "none"
    for role in ("prod", "twin"):
        if isinstance(record.get(role), dict):
            return list(record[role].get("ids") or []), role
    return [], "none"


def _write_receipt_never_masking(pack_dir, record, pack_id, status) -> bool:
    """Receipt write for a payload_patch ABORTED/REFUSED verdict (task 2817, #2165 model). The verdict
    and its reason are already decided and are about to be re-raised or returned; a receipt-write
    failure (disk full, unserialisable field, permission) is logged with that reason and never
    replaces it. The pack is left un-DONE either way, so the next tick re-runs it.
    Returns False when the write failed: a caller whose verdict is a plain return (nothing in flight
    for the error to mask) must then stop the drain itself."""
    from cytir.index import promote_queue as _Q
    try:
        _Q.write_pack_receipt(pack_dir, record)
    except Exception as werr:  # noqa: BLE001
        print(f"[{pack_id}] {status} receipt write failed ({werr!r}); original reason kept: "
              f"{record.get('reason')}", file=sys.stderr)
        record["receipt_write_error"] = f"{type(werr).__name__}: {werr}"
        return False
    return True


def load_content_pack_ids_from_state(manifest: dict, state_record: dict
                                     ) -> tuple[list[dict], list[dict]]:
    """Re-load the raw pack files and filter down to EXACTLY the ids a
    twin-write-state record names -- shared by both resume shapes (task
    673, review finding 2): `resume_content_pack_prod_only` (state
    `"twin-write-complete"`, only the prod leg replays) and the
    `"twin-write-started"` replay in `drain()` (both legs replay,
    idempotently). Never `content_pack_dry_check()` here: re-validating
    tombstones against twin would find the ones the twin leg already
    deleted and silently drop them, losing the exact id list a resume
    still needs."""
    from cytir.index import promote_queue as _Q
    pack_ids: list[str] = state_record.get("pack_ids") or []
    tomb_ids: list[str] = state_record.get("tomb_ids") or []
    pack_id_set = set(pack_ids)
    tomb_id_set = set(tomb_ids)

    all_points = _Q.load_content_pack_points(manifest["pack_points"])
    points = [p for p in all_points if p["id"] in pack_id_set]
    tombstones: list[dict] = []
    if tomb_ids and manifest.get("tombstones"):
        all_tombstones = _Q.load_tombstones(manifest["tombstones"])
        tombstones = [t for t in all_tombstones if t["id"] in tomb_id_set]
    return points, tombstones


#: Top-level receipt keys a twin-first receipt carries and a later reader (`cytir.ops.landing_check`,
#: `cytir.ops.cc_tranche_delta`) reads. A prod-only resume keeps them from the receipt it replaces.
TWIN_FIRST_KEYS = ("mechanical_plan", "twin", "german_parts", "gate", "mechanical_content",
                   "already_served_skipped", "waiver")


#: A prior receipt larger than this is not parsed (derive-only shape). The tranche 14 receipt is 118 MB
#: (mostly `prod.vector_key`); json.loads of it peaks at a few times its size, well inside this bound.
MAX_PRIOR_RECEIPT_BYTES = 512 * 1024 * 1024


def _read_prior_receipt(pack_dir: pathlib.Path) -> dict | None:
    """The receipt a resume is about to replace, or None (absent / unreadable / too large / not a dict)."""
    from cytir.index import promote_queue as _Q
    path = pack_dir / "receipt.json"
    try:
        if path.stat().st_size > _Q.MAX_PRIOR_RECEIPT_BYTES:
            return None
        prior = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return prior if isinstance(prior, dict) else None


def resume_receipt_contract(prior: dict | None, prod_receipt: dict, record: dict) -> dict:
    """Task 801: the fields a prod-only resume adds so its receipt keeps the twin-first shape.

    The resume used to replace the receipt with `{status, prod, resumed_prod_only}`: the twin-first
    top-level keys were gone, and `cytir.ops.cc_tranche_delta` (reads `mechanical_plan.rows.distinct_urls`
    of the first pack) exited 8 for tranche 14 on shape alone. Now:

    * every twin-first key the PRIOR receipt still holds and `record` does not is carried over (the
      twin leg's own record, when an earlier receipt kept it), but only from a prior receipt of the SAME
      `manifest_sha`;
    * `mechanical_plan` is always present. When no prior one exists it is derived from the prod leg's
      own plan: `plan()` counts the WHOLE pack projection before the resume narrows the write set, so
      `rows` (vectors, projected, distinct_urls ...) is the pack's, the same numbers the twin leg's
      plan recorded. It says so in `derived_from`.
    """
    out: dict = {}
    # Review 802 M2: a reseal keeps a non-REFUSED receipt of the OLD manifest; carry nothing from a
    # receipt of another manifest (the drain's own guard, `read_failure_streak`, is the model).
    if not (prior and record.get("manifest_sha") and prior.get("manifest_sha") == record.get("manifest_sha")):
        prior = None
    for key in TWIN_FIRST_KEYS:
        if key not in record and (prior or {}).get(key) is not None:
            out[key] = prior[key]
    if not isinstance(out.get("mechanical_plan"), dict):
        out.pop("mechanical_plan", None)
    if "mechanical_plan" not in out and "mechanical_plan" not in record:
        rows = (prod_receipt or {}).get("rows")
        if isinstance(rows, dict):
            out["mechanical_plan"] = {
                "rows": dict(rows),
                "identity_check": (prod_receipt or {}).get("identity_check"),
                "derived_from": "prod leg plan of the prod-only resume (task 801): the twin leg's "
                                "plan record was not kept; plan() counts the whole pack projection"}
    return out


def write_pack_receipt(pack_dir: pathlib.Path, record: dict) -> None:
    if DRY.active():
        _dry_run_receipt(pack_dir, record)
        return
    atomic_write_text(pack_dir / "receipt.json", json.dumps(record, indent=1) + "\n")


def _dry_run_receipt(pack_dir: pathlib.Path, record: dict) -> None:
    """A dry run's receipt never lands in the queue (2026-10-03): the next REAL drain reads
    `receipt.json` -- a REFUSED at a cacheable stage is skipped until a reseal, so a dry run
    against a stale persisted index or a scratch twin would park a good pack, and a dry receipt
    over an ABORTED one erases the record of a half-written pack. The summary is printed; with
    `--dry-receipt-dir` the full receipt goes there, outside the queue."""
    summary = {k: record[k] for k in ("status", "stage", "would", "note", "blocked_on")
               if record.get(k) not in (None, [])}
    if record.get("reason"):
        summary["reason"] = str(record["reason"])[:300]
    print(f"[{pack_dir.name}] dry-run receipt (not written to the queue): "
          f"{json.dumps(summary, sort_keys=True)}")
    out_dir = DRY.receipt_dir()
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{pack_dir.name}.receipt.json"
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(tmp, path)
