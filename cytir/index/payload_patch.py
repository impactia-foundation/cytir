"""The `payload_patch` pack kind of the promotion drain (task 2748): a payload-only value correction
that reaches the twin and then prod with a count delta of 0 and NO vector write.

WHY A KIND OF ITS OWN. `content_pack` upserts `{id, payload, vector}`: a snippet fix over 5.6M rows
would rewrite 5.6M vectors on the 30 GB prod box (memory `aw-heavy-rewrite-degrades-prod`). This kind
sends qdrant `points/batch` `set_payload` for ONE allow-listed field, by id, never a scroll.

MANIFEST (`schema` `aw.promotion-queue-pack.v1`, `kind` `payload_patch`, `payload_patch_version` 1)
  pack_id, family, collection (the TWIN; prod is the live `aw_prod` alias), field (one of
  `ALLOWED_FIELDS`), rows_file (JSONL, plain or `.gz`, relative to the pack dir or absolute),
  count (rows), sha256 (of the DECOMPRESSED rows file, the `_verify_pack_file` convention),
  optional expect_missing (default 0): how many rows may be absent from a leg.
  A row is `{"id", "expect", "value"}`: `expect` is the value the point must hold NOW, `value` the
  value it must hold after.

PER ROW, PER LEG (twin first, then prod), decided from the point's CURRENT value, in a READ-ONLY pass
BEFORE any write; a leg with any precondition_failed, or missing > expect_missing, is REFUSED (nothing
written, the drain stops), and the prod leg also needs the same id set (applied + already_applied) as the twin's:
  current == value   -> already_applied (never written: the leg is idempotent, a re-run is safe)
  current == expect  -> applied         (written, then read back and compared)
  point absent       -> missing
  anything else      -> precondition_failed (NEVER overwritten: somebody moved it since the producer)
CONSERVATION: selected == applied + already_applied + precondition_failed + missing, and the leg's
points_count before == after, or the leg raises `PayloadPatchError` and the pack is FAILED.

ROLLBACK is a generated inverse pack (`value` and `expect` swapped), written BEFORE the first write
to `<artifact_root>/<pack_id>-inverse/` (not into the queue: nothing arms itself). It covers every row:
on a leg where a row is still at its old value the inverse reads `already_applied`.

This module imports no drain code; the drain wires `QdrantPatchClient` (or a fake, in the tests).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import time
import uuid
from typing import Callable, Iterable, Protocol

from cytir.index import dry_run as DRY

KIND = "payload_patch"
VERSION = 1
ALLOWED_FIELDS = ("snippet",)          # a new field is a code change with a review, never a flag
MAX_BATCH = 1000
DEFAULT_PAUSE_S = 0.2
RECEIPT_SCHEMA = "aw.payload-patch-receipt.v1"
MANIFEST_REQUIRED = ("pack_id", "family", "collection", "field", "rows_file", "count", "sha256")
TERMS = ("applied", "already_applied", "precondition_failed", "missing")
SAMPLE = 20
SAMPLE_VERIFY = 200


class PayloadPatchError(RuntimeError):
    """The pack, or a leg of it, broke an invariant (conservation, count, read-back, shape)."""


class Client(Protocol):
    def count(self, collection: str) -> int: ...
    def read(self, collection: str, ids: list[str], field: str) -> dict[str, object]:
        """{id: current value} for the ids that exist (a point with no such field maps to None)."""
    def write(self, collection: str, field: str, pairs: list[tuple[str, object]]) -> None: ...


def check_manifest(manifest: dict) -> None:
    """Raises `PayloadPatchError` for a manifest this kind cannot run (shape and allow-list only)."""
    missing = [k for k in MANIFEST_REQUIRED if manifest.get(k) is None]
    if missing:
        raise PayloadPatchError(f"payload_patch manifest missing {missing}")
    if manifest.get("payload_patch_version") != VERSION:
        raise PayloadPatchError(f"payload_patch_version {manifest.get('payload_patch_version')!r}, "
                                f"this drain reads {VERSION}")
    if manifest["field"] not in ALLOWED_FIELDS:
        raise PayloadPatchError(f"field {manifest['field']!r} is not patchable; allow-list {ALLOWED_FIELDS}")
    em = manifest.get("expect_missing", 0)
    if not isinstance(em, int) or isinstance(em, bool) or em < 0:
        raise PayloadPatchError(f"expect_missing must be an integer >= 0, got {em!r}")
    if not isinstance(manifest["count"], int) or manifest["count"] < 1:
        raise PayloadPatchError(f"count must be a positive integer, got {manifest['count']!r}")


def rows_path(manifest: dict, pack_dir: pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(manifest["rows_file"])
    return p if p.is_absolute() else pathlib.Path(pack_dir) / p


def load_rows(manifest: dict, pack_dir: pathlib.Path) -> list[dict]:
    """The verified rows: sha256 of the decompressed file and the row count must match the manifest."""
    check_manifest(manifest)
    path = rows_path(manifest, pack_dir)
    opener = gzip.open if str(path).endswith(".gz") else open
    h = hashlib.sha256()
    rows: list[dict] = []
    seen: set[str] = set()
    with opener(path, "rb") as fh:
        for line in fh:
            h.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if not (isinstance(row, dict) and set(row) == {"id", "expect", "value"}
                    and isinstance(row["id"], str) and isinstance(row["expect"], str)
                    and isinstance(row["value"], str)):
                raise PayloadPatchError(f"{path}: row {len(rows) + 1} is not {{id, expect, value}} of strings")
            try:
                canonical = str(uuid.UUID(row["id"])) == row["id"]
            except ValueError:
                canonical = False
            if not canonical:
                raise PayloadPatchError(f"{path}: row {len(rows) + 1} id {row['id']!r} is not a canonical "
                                        "lowercase UUID (Qdrant would not return it: it would read missing)")
            if row["id"] in seen:
                raise PayloadPatchError(f"{path}: duplicate id {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    if h.hexdigest() != manifest["sha256"]:
        raise PayloadPatchError(f"{path}: sha256 {h.hexdigest()[:12]} != manifest's {manifest['sha256'][:12]}")
    if len(rows) != manifest["count"]:
        raise PayloadPatchError(f"{path}: {len(rows)} rows, manifest count {manifest['count']}")
    return rows


@DRY.refuses_when_dry("write the inverse pack")
def write_inverse_pack(out_dir: pathlib.Path, manifest: dict, rows: list[dict]) -> dict:
    """`<out_dir>/<pack_id>-inverse/{rows.jsonl,manifest.json}`: `value` and `expect` swapped, NO READY
    (the orchestrator arms it). Returns `{"dir", "rows_sha256", "manifest_sha256"}`."""
    inv_id = f"{manifest['pack_id']}-inverse"
    d = pathlib.Path(out_dir) / inv_id
    d.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps({"id": r["id"], "expect": r["value"], "value": r["expect"]},
                              sort_keys=True) + "\n" for r in rows).encode()
    (d / "rows.jsonl").write_bytes(body)
    inv = {"schema": "aw.promotion-queue-pack.v1", "kind": KIND, "payload_patch_version": VERSION,
           "pack_id": inv_id, "family": manifest["family"], "collection": manifest["collection"],
           "field": manifest["field"], "rows_file": "rows.jsonl", "count": len(rows),
           "expect_missing": manifest.get("expect_missing", 0),
           "sha256": hashlib.sha256(body).hexdigest(), "inverse_of": manifest["pack_id"]}
    mbytes = (json.dumps(inv, indent=1, sort_keys=True) + "\n").encode()
    (d / "manifest.json").write_bytes(mbytes)
    return {"dir": str(d), "rows_sha256": inv["sha256"], "manifest_sha256": hashlib.sha256(mbytes).hexdigest()}


def conserved(receipt: dict) -> bool:
    return receipt["selected"] == sum(receipt[t] for t in TERMS)


class PayloadPatchRefusal(PayloadPatchError):
    """A leg was refused by its pre-write classification: NOTHING was written. The drain stops on it
    (ABORTED receipt): the next tick finds the same state and stops again until someone acts."""


def classify(client: Client, collection: str, rows: list[dict], field: str, *, batch: int = MAX_BATCH,
             read_pause_s: float = 0.0, sleep: Callable[[float], None] = time.sleep) -> dict[str, list[str]]:
    """READ-ONLY pass: ids by outcome (`applied` = rows still at `expect`, i.e. to write). Writes nothing.
    `read_pause_s` paces the retrieves (the prod leg: a 5M-row re-run is 5,000 retrieves)."""
    if not 1 <= batch <= MAX_BATCH:
        raise PayloadPatchError(f"batch {batch} outside 1..{MAX_BATCH}")
    ids: dict[str, list[str]] = {t: [] for t in TERMS}
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        current = client.read(collection, [r["id"] for r in chunk], field)
        if read_pause_s:
            sleep(read_pause_s)
        for r in chunk:
            if r["id"] not in current:
                ids["missing"].append(r["id"])
            elif current[r["id"]] == r["value"]:
                ids["already_applied"].append(r["id"])
            elif current[r["id"]] == r["expect"]:
                ids["applied"].append(r["id"])
            else:
                ids["precondition_failed"].append(r["id"])
    return ids


def refuse_reason(ids: dict[str, list[str]], expect_missing: int, reference: dict | None = None) -> str | None:
    """Why a leg may not write, from its classification (None = go). `reference` is the TWIN's receipt:
    the prod leg must patch the same SET of ids (applied + already_applied, as ids) as the twin."""
    if ids["precondition_failed"]:
        return (f"{len(ids['precondition_failed'])} row(s) moved since the producer (precondition_failed, "
                f"first {ids['precondition_failed'][:SAMPLE]})")
    if len(ids["missing"]) > expect_missing:
        return f"{len(ids['missing'])} row(s) missing, manifest expect_missing is {expect_missing}"
    if reference is not None:
        mine = set(ids["applied"]) | set(ids["already_applied"])
        theirs = set(reference["ids"]["applied"]) | set(reference["ids"]["already_applied"])
        if mine != theirs:
            only_here, only_twin = sorted(mine - theirs), sorted(theirs - mine)
            return (f"applied + already_applied differs from the twin's: {len(only_here)} row(s) only on this leg "
                    f"(first {only_here[:SAMPLE]}), {len(only_twin)} row(s) only on the twin "
                    f"(first {only_twin[:SAMPLE]})")
    return None


def apply_leg(role: str, client: Client, collection: str, rows: list[dict], field: str, *,
              batch: int = MAX_BATCH, pause_s: float = DEFAULT_PAUSE_S, read_pause_s: float = 0.0,
              expect_missing: int = 0, reference: dict | None = None,
              sleep: Callable[[float], None] = time.sleep,
              on_write: Callable[[], None] | None = None,
              before_write: Callable[[], None] | None = None) -> dict:
    """One leg (`role` "twin" or "prod"): classify (read-only), REFUSE before any write when
    `refuse_reason` says so (`PayloadPatchRefusal`), then write the `applied` rows in batches.
    Returns the leg receipt (`points_after` not set yet: `verify_leg`) with the full id lists under `ids`.
    `on_write` is called right before each write call (the drain's `WriteAttempt.started`).
    `before_write` runs once, after the classification and the refusal check and before the first write,
    only when there is something to write (the drain re-resolves the prod alias there: task 2777); an
    exception from it leaves nothing written."""
    if field not in ALLOWED_FIELDS:
        raise PayloadPatchError(f"field {field!r} is not patchable")
    before = client.count(collection)
    ids = classify(client, collection, rows, field, batch=batch, read_pause_s=read_pause_s, sleep=sleep)
    why = refuse_reason(ids, expect_missing, reference)
    if why:
        raise PayloadPatchRefusal(f"{role}/{collection}: refused before any write: {why}")
    by_id = {r["id"]: r for r in rows}
    calls = 0
    todo = ids["applied"]
    if todo and before_write is not None:
        before_write()
    for i in range(0, len(todo), batch):
        if on_write is not None:
            on_write()
        client.write(collection, field, [(j, by_id[j]["value"]) for j in todo[i:i + batch]])
        calls += 1
        if pause_s:
            sleep(pause_s)
    receipt = {"role": role, "collection": collection, "field": field, "selected": len(rows),
               **{t: len(ids[t]) for t in TERMS}, "points_before": before, "write_calls": calls,
               "batch": batch, "pause_s": pause_s, "expect_missing": expect_missing,
               "precondition_failed_ids": ids["precondition_failed"][:SAMPLE], "missing_ids": ids["missing"][:SAMPLE],
               "ids": ids}
    if not conserved(receipt):
        raise PayloadPatchError(f"{role}: conservation broken in {receipt}")
    return receipt


def verify_leg(client: Client, collection: str, rows: list[dict], receipt: dict, *,
               sample: int = SAMPLE_VERIFY, read_pause_s: float = 0.0,
               sleep: Callable[[float], None] = time.sleep) -> dict:
    """INDEPENDENT check after the writes: a fresh retrieve of an evenly spaced sample (>= `sample` ids,
    all if fewer; written rows first) must read `value`, and a fresh points_count must equal the count
    before the leg. Sets `points_after` and `verify_sample`. Raises `PayloadPatchError` otherwise."""
    ids = receipt["ids"]
    pool = ids["applied"] + ids["already_applied"]
    step = max(1, len(pool) // sample)
    picked = (pool if len(pool) <= sample else pool[::step][:max(sample, 1)])
    if ids["applied"] and not set(ids["applied"][:1]) & set(picked):
        picked = [ids["applied"][0]] + picked
    value = {r["id"]: r["value"] for r in rows}
    bad: list[str] = []
    for i in range(0, len(picked), MAX_BATCH):
        part = picked[i:i + MAX_BATCH]
        got = client.read(collection, part, receipt["field"])
        if read_pause_s:
            sleep(read_pause_s)
        bad += [j for j in part if got.get(j) != value[j]]
    if bad:
        raise PayloadPatchError(f"{receipt['role']}/{collection}: {len(bad)} of {len(picked)} sampled row(s) do "
                                f"not read the new {receipt['field']} (first {bad[:SAMPLE]})")
    after = client.count(collection)
    receipt["points_after"] = after
    receipt["verify_sample"] = len(picked)
    if receipt["points_before"] != after:
        raise PayloadPatchError(f"{receipt['role']}/{collection}: points_count moved "
                                f"{receipt['points_before']} -> {after} on a payload-only pack: FAILED")
    return receipt


def run_leg(role, client, collection, rows, field, **kw) -> dict:
    """apply_leg then verify_leg (the pair the drain runs as `apply_target` / `verify_target`)."""
    read_pause = kw.get("read_pause_s", 0.0)
    receipt = apply_leg(role, client, collection, rows, field, **kw)
    return verify_leg(client, collection, rows, receipt, read_pause_s=read_pause,
                      sleep=kw.get("sleep", time.sleep))


@DRY.refuses_when_dry("write the leg id lists")
def write_leg_ids(art_dir: pathlib.Path, pack_id: str, receipt: dict) -> pathlib.Path:
    """`<artifact_root>/<pack_id>-<role>-ids.json`: the FULL id list per outcome (the receipt keeps 20)."""
    path = pathlib.Path(art_dir) / f"{pack_id}-{receipt['role']}-ids.json"
    path.write_text(json.dumps(receipt["ids"], sort_keys=True) + "\n")
    return path


class QdrantPatchClient:
    """The real client: id retrieves, `points/batch` `set_payload`, `GET /collections/<c>`. Never a scroll.
    Every call goes through `promote_sealed_pack._api` (the drain's one retrying door: a timeout or a
    transient 408/429/5xx is retried on a bounded ladder inside the leg's budget, any other 4xx is an
    immediate refusal), so the prod gate (`prod_guard.check`) and the dry-run guard apply too."""

    def __init__(self, url: str) -> None:
        self.url = url

    def count(self, collection: str) -> int:
        from cytir.index import promote_sealed_pack as PSP
        return int(PSP._api(self.url, f"/collections/{collection}")["points_count"])

    def read(self, collection: str, ids: list[str], field: str) -> dict[str, object]:
        from cytir.index import point_delete as PD
        from cytir.index import promote_sealed_pack as PSP
        got = PD.fetch_payload_fields(self.url, collection, ids, [field], batch=MAX_BATCH, api=PSP._api)
        return {i: p.get(field) for i, p in got.items()}

    def write(self, collection: str, field: str, pairs: Iterable[tuple[str, object]]) -> None:
        # One set_payload operation per point (each has its own value), one HTTP call per batch.
        from cytir.index import point_delete as PD
        from cytir.index import promote_sealed_pack as PSP
        PD.set_payload_batch(self.url, collection,
                             [{"payload": {field: v}, "points": [i]} for i, v in pairs], api=PSP._api)
