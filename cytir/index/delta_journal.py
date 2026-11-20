"""Chunk@2 post-cutoff delta journal and candidate replay (task 367).

Chunk@2 is a new ID epoch.  A pack applied after its base cutoff cannot be
replayed by forwarding the old point IDs: content points have to be re-chunked
and re-embedded, while payload changes and removals have to be addressed by
their stable source identity (canonical URL + source type).  This tool makes
that distinction an executable contract.

The journal contains immutable *copies* of every input artifact, SHA-256s for
the source and each copy, and a hash chain.  ``record`` refuses the familiar
legacy ``{id: ...}`` artifacts because an old point id is not an identity in a
new chunk epoch.  ``replay`` verifies the whole chain before writing a
chunk@2 candidate and emits a final caught-up manifest only after every entry
has been applied and verified.

This is deliberately not an applier for ``aw_b1``/``aw_prod``.  It is for the
non-serving chunk@2 candidate assembled on the build box.  Promotion remains the
separately gated ADR-083 mode-2 whole-collection alias flip.

Operation artifacts (JSONL, optionally gzip):

* content-rechunk-reembed: ``--source`` is the immutable L0/L1 replay input;
  ``replay_input=...`` records the exact re-chunk job input; ``chunk2_records``
  is its complete, embedded chunk@2 points dump.  Point payloads must carry
  ``stamps: {chunk: 2}``.
* payload-only: ``payload_patch=...`` rows are
  ``{"identity":{"url":...,"source_type":...},"set":{...},
  "delete_keys":[...]}``.  They are applied to every chunk@2 record in that
  source group.
* tombstone: ``tombstones=...`` rows are
  ``{"identity":{"url":...,"source_type":...},"reason":"..."}``.
  The source group is deleted from the candidate.  Point-id-only tombstones
  are rejected: they are not portable across a re-chunk.

The candidate manifest passed to replay must contain
``{"chunk_epoch": 2, "base_cutoff_sha256": "..."}``, binding the candidate
base to the journal cutoff before it receives any post-cutoff work.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from cytir.index.supersede import Superseder, group_key, guard_scroll, url_variants
from cytir.core.digest import canonical_utf8_bytes
from cytir.core.qdrant import qdrant_connect

try:
    from qdrant_client import models as qm
except ImportError:  # Pure journal checks remain testable without qdrant.
    qm = None

SCHEMA = "aw.chunk2-delta-journal.v1"
BASE_CUTOFF_SCHEMA = "aw.chunk2-base-candidate.v1"
MODES = ("content-rechunk-reembed", "payload-only", "tombstone")
MODE_ROLES = {
    "content-rechunk-reembed": {"replay_input", "chunk2_records"},
    "payload-only": {"payload_patch"},
    "tombstone": {"tombstones"},
}
JOURNAL = "journal.jsonl"
CUTOFF = "cutoff.json"
ARTIFACTS = "artifacts"
SETTLED_LANDING_OPERATIONS = (
    "fedlex-358b",
    "register-363-dates-status",
    "register-363-people",
)


def abort(msg: str) -> None:
    raise SystemExit(f"FATAL: {msg}")


canonical_bytes = canonical_utf8_bytes


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def validate_base_cutoff(raw: dict[str, Any]) -> None:
    """Reject a journal base that cannot truthfully bridge the landing gap.

    A point count alone is not a chunk@2 base.  The cutoff has to name the
    settled Fedlex + two register operations, while Task359 is deliberately
    the first post-cutoff operation: its old point IDs cannot cross the new
    chunk epoch and must be replayed by source identity instead.
    """
    if raw.get("schema") != BASE_CUTOFF_SCHEMA:
        abort(f"base cutoff schema must be {BASE_CUTOFF_SCHEMA!r}")
    if not isinstance(raw.get("app_git_sha"), str) or len(raw["app_git_sha"]) != 40:
        abort("base cutoff requires full app_git_sha")
    if not isinstance(raw.get("created_at"), str) or not raw["created_at"]:
        abort("base cutoff requires created_at")
    snapshot = raw.get("base_snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("collection") != "aw_b1":
        abort("base cutoff requires aw_b1 base_snapshot")
    if not isinstance(snapshot.get("exact_count"), int) or snapshot["exact_count"] < 1:
        abort("base cutoff requires positive base_snapshot.exact_count")
    if not isinstance(snapshot.get("qdrant_version"), str) or not snapshot["qdrant_version"]:
        abort("base cutoff requires base_snapshot.qdrant_version")
    # Task382's frozen input carries file bindings as nested evidence objects.
    # Keep accepting the original flat shape for historical pure tests, but
    # never make the journal re-derive a second cutoff schema.
    snapshot_sha = snapshot.get("snapshot_sha256")
    if snapshot_sha is None and isinstance(snapshot.get("snapshot"), dict):
        snapshot_sha = snapshot["snapshot"].get("sha256")
    if not is_sha256(snapshot_sha):
        abort("base cutoff requires base_snapshot.snapshot_sha256")
    landing = raw.get("settled_landing_operations")
    if not isinstance(landing, list) or {x.get("operation") for x in landing if isinstance(x, dict)} != set(SETTLED_LANDING_OPERATIONS):
        abort("base cutoff must bind exactly the settled Fedlex and two Task363 operations")
    for entry in landing:
        if not isinstance(entry, dict) or entry.get("status") != "TWIN-PROD-VERIFIED-OK":
            abort("base cutoff landing operation lacks terminal twin/prod verification")
        if entry.get("exact_post_apply_count") != snapshot["exact_count"]:
            abort("base cutoff landing count does not match base snapshot")
        receipt = entry.get("receipt")
        receipt_path = entry.get("receipt_path")
        receipt_sha = entry.get("receipt_sha256")
        if isinstance(receipt, dict):
            receipt_path = receipt.get("path")
            receipt_sha = receipt.get("sha256")
        if not isinstance(receipt_path, str) or not is_sha256(receipt_sha):
            abort("base cutoff landing operation requires immutable receipt path and SHA-256")
    if raw.get("first_journaled_operation") != "task359-intl":
        abort("base cutoff must declare Task359 as the first post-cutoff journaled operation")
    census = raw.get("source_census")
    if not isinstance(census, dict) or not isinstance(census.get("path"), str) or not is_sha256(census.get("sha256")):
        abort("base cutoff requires hashed source_census")
    if not isinstance(raw.get("inputs"), list) or not raw["inputs"]:
        abort("base cutoff requires non-empty source inputs")
    chunk = raw.get("chunk")
    config_sha = chunk.get("config_sha256") if isinstance(chunk, dict) else None
    if config_sha is None and isinstance(chunk, dict) and isinstance(chunk.get("config"), dict):
        config_sha = chunk["config"].get("sha256")
    if not isinstance(chunk, dict) or chunk.get("epoch") != 2 or not is_sha256(config_sha):
        abort("base cutoff requires chunk epoch 2 and config SHA-256")
    if not isinstance(chunk.get("rule_legend"), str) or not chunk["rule_legend"]:
        abort("base cutoff requires chunk rule legend")
    embed = raw.get("embed")
    if not isinstance(embed, dict) or not isinstance(embed.get("model"), str) or not embed["model"]:
        abort("base cutoff requires embedding contract")


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open_text(path) as inp:
        for number, line in enumerate(inp, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                abort(f"{path}:{number}: invalid JSON ({exc})")
            if not isinstance(row, dict):
                abort(f"{path}:{number}: JSONL row must be an object")
            yield row


def require_identity(row: dict[str, Any], context: str) -> tuple[str, str | None]:
    if "id" in row:
        abort(f"{context}: legacy point-id-only operation is unsupported across chunk epochs; use identity.url/source_type")
    identity = row.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("url"), str) or not identity["url"]:
        abort(f"{context}: require identity={{url: ..., source_type: ...}}")
    source_type = identity.get("source_type")
    if source_type is not None and not isinstance(source_type, str):
        abort(f"{context}: identity.source_type must be a string or null")
    return identity["url"], source_type


def validate_content_records(path: Path) -> int:
    count = 0
    for n, row in enumerate(jsonl(path), 1):
        if not isinstance(row.get("id"), (str, int)) or not isinstance(row.get("vector"), dict):
            abort(f"{path}:{n}: chunk2_records require id + vector")
        payload = row.get("payload")
        if not isinstance(payload, dict) or not isinstance(payload.get("url"), str) or not payload["url"]:
            abort(f"{path}:{n}: chunk2_records require payload.url")
        stamps = payload.get("stamps")
        if not isinstance(stamps, dict) or stamps.get("chunk") != 2:
            abort(f"{path}:{n}: content dump is not stamped chunk@2 (payload.stamps.chunk != 2)")
        count += 1
    if not count:
        abort(f"{path}: chunk2_records is empty")
    return count


def validate_payload_patch(path: Path) -> int:
    count = 0
    for n, row in enumerate(jsonl(path), 1):
        require_identity(row, f"{path}:{n}")
        set_values = row.get("set", {})
        delete_keys = row.get("delete_keys", [])
        if not isinstance(set_values, dict) or not isinstance(delete_keys, list) or not all(isinstance(x, str) for x in delete_keys):
            abort(f"{path}:{n}: payload patch requires object set and string-list delete_keys")
        if not set_values and not delete_keys:
            abort(f"{path}:{n}: empty payload patch")
        if set(set_values).intersection(delete_keys):
            abort(f"{path}:{n}: same key appears in set and delete_keys")
        count += 1
    if not count:
        abort(f"{path}: payload_patch is empty")
    return count


def validate_tombstones(path: Path) -> int:
    count = 0
    for n, row in enumerate(jsonl(path), 1):
        require_identity(row, f"{path}:{n}")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            abort(f"{path}:{n}: tombstone requires a non-empty reason")
        count += 1
    if not count:
        abort(f"{path}: tombstones is empty")
    return count


def validate_artifact(mode: str, role: str, path: Path) -> int | None:
    if role == "chunk2_records":
        return validate_content_records(path)
    if role == "payload_patch":
        return validate_payload_patch(path)
    if role == "tombstones":
        return validate_tombstones(path)
    # replay_input is intentionally opaque: it may be parquet, jsonl, or a
    # manifest.  Its immutable copy + content hash bind it to the content run.
    return None


def load_json(path: Path, what: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        abort(f"cannot read {what} {path}: {exc}")
    if not isinstance(value, dict):
        abort(f"{what} {path} must be a JSON object")
    return value


def journal_paths(root: Path) -> tuple[Path, Path, Path]:
    return root / CUTOFF, root / JOURNAL, root / ARTIFACTS


def verify_cutoff(root: Path) -> dict[str, Any]:
    cutoff_path, _, _ = journal_paths(root)
    cutoff = load_json(cutoff_path, "cutoff manifest")
    if cutoff.get("schema") != SCHEMA or not isinstance(cutoff.get("cutoff"), dict):
        abort(f"invalid cutoff manifest {cutoff_path}")
    expected = cutoff.get("cutoff_sha256")
    bare = dict(cutoff)
    bare.pop("cutoff_sha256", None)
    actual = sha256_bytes(canonical_bytes(bare))
    if expected != actual:
        abort(f"cutoff hash drift: recorded {expected}, recomputed {actual}")
    return cutoff


def load_entries(root: Path) -> list[dict[str, Any]]:
    cutoff = verify_cutoff(root)
    _, path, _ = journal_paths(root)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    previous = cutoff["cutoff_sha256"]
    for sequence, entry in enumerate(jsonl(path), 1):
        if entry.get("schema") != SCHEMA or entry.get("sequence") != sequence:
            abort(f"{path}:{sequence}: journal sequence/schema gap")
        if entry.get("previous_sha256") != previous:
            abort(f"{path}:{sequence}: previous hash mismatch (gap/reorder detected)")
        claimed = entry.get("entry_sha256")
        bare = dict(entry)
        bare.pop("entry_sha256", None)
        actual = sha256_bytes(canonical_bytes(bare))
        if claimed != actual:
            abort(f"{path}:{sequence}: entry hash drift")
        if entry.get("mode") not in MODES:
            abort(f"{path}:{sequence}: unsupported mode {entry.get('mode')!r}")
        roles = {a.get("role") for a in entry.get("artifacts", [])}
        if roles != MODE_ROLES[entry["mode"]]:
            abort(f"{path}:{sequence}: roles {sorted(roles)} do not match mode {entry['mode']}")
        source = entry.get("source") or {}
        if not isinstance(source.get("git_sha"), str) or not source["git_sha"]:
            abort(f"{path}:{sequence}: missing source git SHA")
        for artifact in [source, *entry["artifacts"]]:
            stored = root / artifact.get("stored_path", "")
            if not stored.is_file():
                abort(f"{path}:{sequence}: immutable artifact missing: {stored}")
            current = sha256_file(stored)
            if current != artifact.get("sha256"):
                abort(f"{path}:{sequence}: artifact hash drift for {stored}")
        previous = claimed
        entries.append(entry)
    return entries


def copy_immutable(root: Path, sequence: int, role: str, source: Path) -> dict[str, Any]:
    if not source.is_file():
        abort(f"artifact is not a file: {source}")
    _, _, store = journal_paths(root)
    store.mkdir(parents=True, exist_ok=False if not store.exists() else True)
    digest = sha256_file(source)
    safe_role = role.replace("/", "_")
    destination = store / f"{sequence:06d}-{safe_role}-{digest[:16]}{source.suffix}"
    if destination.exists():
        abort(f"immutable destination already exists: {destination}")
    # copy2, never a symlink/hardlink: later source changes cannot rewrite the
    # evidence we replay.
    shutil.copy2(source, destination)
    if sha256_file(destination) != digest:
        abort(f"copy hash mismatch for {source}")
    return {"role": role, "stored_path": str(destination.relative_to(root)),
            "sha256": digest, "bytes": destination.stat().st_size,
            "original_name": source.name}


def parse_role_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        abort(f"--artifact must be ROLE=PATH, got {value!r}")
    role, raw = value.split("=", 1)
    if not role or not raw:
        abort(f"--artifact must be ROLE=PATH, got {value!r}")
    return role, Path(raw)


def command_init(args: argparse.Namespace) -> int:
    root = args.journal_dir
    if root.exists():
        abort(f"journal directory already exists: {root}")
    source = Path(args.cutoff_manifest)
    raw = load_json(source, "cutoff input")
    if raw.get("schema") == SCHEMA:
        abort("cutoff input must be the base-build manifest, not another journal cutoff")
    validate_base_cutoff(raw)
    root.mkdir(parents=True)
    cutoff = {"schema": SCHEMA, "created_at": args.created_at or time.strftime("%FT%TZ", time.gmtime()),
              "cutoff": raw, "cutoff_source_name": source.name,
              "cutoff_source_sha256": sha256_file(source)}
    cutoff["cutoff_sha256"] = sha256_bytes(canonical_bytes(cutoff))
    (root / CUTOFF).write_text(json.dumps(cutoff, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "INITIALIZED", "journal": str(root),
                      "cutoff_sha256": cutoff["cutoff_sha256"]}, sort_keys=True))
    return 0


def command_record(args: argparse.Namespace) -> int:
    root = args.journal_dir
    cutoff = verify_cutoff(root)
    prior = load_entries(root)
    mode = args.mode
    wanted = MODE_ROLES[mode]
    values = [parse_role_path(value) for value in args.artifact]
    roles = {role for role, _ in values}
    if roles != wanted or len(values) != len(wanted):
        abort(f"{mode} needs exactly artifacts {sorted(wanted)}, got {sorted(roles)}")
    if any(entry["operation_id"] == args.operation_id for entry in prior):
        abort(f"operation_id already recorded: {args.operation_id!r}")
    if len(args.source_git_sha) != 40 or any(ch not in "0123456789abcdef" for ch in args.source_git_sha.lower()):
        abort("source git SHA must be a full 40-character hexadecimal commit")
    sequence = len(prior) + 1
    source_path = Path(args.source)
    if not source_path.is_file():
        abort(f"artifact is not a file: {source_path}")
    # Validate every input before copying anything.  A bad row must leave no
    # half-recorded artifact which could block a corrected rerun at sequence N.
    for role, path in values:
        if path.resolve() == source_path.resolve():
            abort("source and operation artifact must be separate files (bind the replay input explicitly)")
        validate_artifact(mode, role, path)
    source_rec = copy_immutable(root, sequence, "source", source_path)
    source_rec["git_sha"] = args.source_git_sha
    artifacts: list[dict[str, Any]] = []
    for role, path in values:
        rows = validate_artifact(mode, role, path)
        copied = copy_immutable(root, sequence, role, path)
        if rows is not None:
            copied["rows"] = rows
        artifacts.append(copied)
    previous = prior[-1]["entry_sha256"] if prior else cutoff["cutoff_sha256"]
    entry: dict[str, Any] = {
        "schema": SCHEMA, "sequence": sequence, "operation_id": args.operation_id,
        "mode": mode, "recorded_at": args.recorded_at or time.strftime("%FT%TZ", time.gmtime()),
        "previous_sha256": previous, "source": source_rec,
        "artifacts": sorted(artifacts, key=lambda item: item["role"]),
    }
    entry["entry_sha256"] = sha256_bytes(canonical_bytes(entry))
    with (root / JOURNAL).open("a", encoding="utf-8") as out:
        out.write(json.dumps(entry, sort_keys=True) + "\n")
    print(json.dumps({"status": "RECORDED", "sequence": sequence, "mode": mode,
                      "entry_sha256": entry["entry_sha256"]}, sort_keys=True))
    return 0


def client(url: str):
    if qm is None:
        abort("qdrant_client is required for replay")
    try:
        return qdrant_connect(url, timeout=900, check_compatibility=False)
    except TypeError:
        return qdrant_connect(url, timeout=900)


def point_from_dump(row: dict[str, Any]):
    vector: dict[str, Any] = {}
    for name, value in row["vector"].items():
        if isinstance(value, dict):
            vector[name] = qm.SparseVector(indices=value["indices"], values=value["values"])
        else:
            vector[name] = value
    return qm.PointStruct(id=row["id"], vector=vector, payload=row["payload"])


def candidate_ids(c: Any, collection: str, url: str, source_type: str | None) -> list[Any]:
    variants = sorted(url_variants(url))
    flt = qm.Filter(must=[qm.FieldCondition(key="url", match=qm.MatchAny(any=variants))])
    wanted = group_key(url, source_type)
    offset = None
    ids: list[Any] = []
    guard_scroll(c, collection)          # a prod target is refused (prod_guard); replay is candidate-only
    while True:
        rows, offset = c.scroll(collection, scroll_filter=flt, limit=1000, offset=offset,
                                with_payload=["url", "source_type"], with_vectors=False)
        for row in rows:
            payload = row.payload or {}
            if group_key(payload.get("url", ""), payload.get("source_type")) == wanted:
                ids.append(row.id)
        if offset is None:
            break
    if not ids:
        abort(f"candidate gap: no chunk@2 records for source identity {wanted!r}")
    return ids


def replay_content(c: Any, collection: str, artifact: Path, batch: int, dry_run: bool) -> dict[str, int]:
    sup = Superseder(c, collection, dry_run=dry_run)
    sup.ensure_url_index(create=False)
    pending: list[Any] = []
    rows = 0
    for row in jsonl(artifact):
        rows += 1
        payload = row["payload"]
        # the row's own entity markers: a card must group apart from its page rows
        sup.observe_payload(row["id"], payload)
        pending.append(point_from_dump(row))
        if len(pending) >= batch:
            if not dry_run:
                c.upsert(collection, points=pending, wait=True)
                present = c.retrieve(collection, ids=[point.id for point in pending],
                                     with_payload=False, with_vectors=False)
                if len(present) != len(pending):
                    abort("content upsert was not durably readable after wait=True")
            pending.clear()
    if pending and not dry_run:
        c.upsert(collection, points=pending, wait=True)
        present = c.retrieve(collection, ids=[point.id for point in pending],
                             with_payload=False, with_vectors=False)
        if len(present) != len(pending):
            abort("content upsert was not durably readable after wait=True")
    result = sup.run()
    return {"rows": rows, "groups": result["groups"], "superseded": result["superseded"]}


def replay_payload(c: Any, collection: str, artifact: Path, batch: int, dry_run: bool) -> dict[str, int]:
    patched = groups = 0
    for row in jsonl(artifact):
        url, source_type = require_identity(row, str(artifact))
        ids = candidate_ids(c, collection, url, source_type)
        groups += 1
        patched += len(ids)
        if dry_run:
            continue
        for i in range(0, len(ids), batch):
            target = qm.PointIdsList(points=ids[i:i + batch])
            if row.get("set"):
                c.set_payload(collection_name=collection, payload=row["set"], points=target, wait=True)
            if row.get("delete_keys"):
                c.delete_payload(collection_name=collection, keys=row["delete_keys"], points=target, wait=True)
            reread = c.retrieve(collection, ids=ids[i:i + batch], with_payload=True, with_vectors=False)
            if len(reread) != len(ids[i:i + batch]):
                abort("payload replay made candidate points unreadable")
            for point in reread:
                payload = point.payload or {}
                if any(payload.get(k) != v for k, v in row.get("set", {}).items()) or any(k in payload for k in row.get("delete_keys", [])):
                    abort("payload replay did not produce the requested exact values")
    return {"groups": groups, "points": patched}


def replay_tombstones(c: Any, collection: str, artifact: Path, batch: int, dry_run: bool) -> dict[str, int]:
    deleted = groups = 0
    for row in jsonl(artifact):
        url, source_type = require_identity(row, str(artifact))
        ids = candidate_ids(c, collection, url, source_type)
        groups += 1
        deleted += len(ids)
        if dry_run:
            continue
        for i in range(0, len(ids), batch):
            current = ids[i:i + batch]
            c.delete(collection, points_selector=qm.PointIdsList(points=current), wait=True)
            if c.retrieve(collection, ids=current, with_payload=False, with_vectors=False):
                abort("tombstone delete did not durably remove every candidate point")
    return {"groups": groups, "deleted": deleted}


def command_verify(args: argparse.Namespace) -> int:
    entries = load_entries(args.journal_dir)
    cutoff = verify_cutoff(args.journal_dir)
    tip = entries[-1]["entry_sha256"] if entries else cutoff["cutoff_sha256"]
    print(json.dumps({"status": "VERIFIED", "entries": len(entries), "tip_sha256": tip,
                      "cutoff_sha256": cutoff["cutoff_sha256"]}, sort_keys=True))
    return 0


def command_replay(args: argparse.Namespace) -> int:
    if args.collection in {"aw_b1", "aw_prod"}:
        abort(f"refusing live collection name {args.collection!r}; replay only into a chunk@2 candidate")
    cutoff = verify_cutoff(args.journal_dir)
    entries = load_entries(args.journal_dir)
    candidate = load_json(Path(args.candidate_manifest), "candidate manifest")
    if candidate.get("chunk_epoch") != 2:
        abort("candidate manifest must declare chunk_epoch=2")
    if candidate.get("base_cutoff_sha256") != cutoff["cutoff_sha256"]:
        abort("candidate base cutoff hash does not match journal; refusing a mixed base")
    c = client(args.qdrant_url)
    if args.collection not in {x.name for x in c.get_collections().collections}:
        abort(f"candidate collection is missing: {args.collection}")
    # A source-identity operation must never degrade into an accidental full
    # scan.  Creating this index is permitted only on the non-serving candidate
    # and remains explicit on the command line.
    Superseder(c, args.collection).ensure_url_index(create=args.create_url_index)
    before = c.count(args.collection, exact=True).count
    reports: list[dict[str, Any]] = []
    for entry in entries:
        item = {"sequence": entry["sequence"], "operation_id": entry["operation_id"],
                "mode": entry["mode"], "entry_sha256": entry["entry_sha256"],
                "source": {"git_sha": entry["source"]["git_sha"],
                           "sha256": entry["source"]["sha256"]},
                "artifacts": [{key: artifact[key] for key in ("role", "sha256", "rows")
                               if key in artifact}
                              for artifact in entry["artifacts"]]}
        artifact = args.journal_dir / entry["artifacts"][0]["stored_path"]
        if entry["mode"] == "content-rechunk-reembed":
            artifact = args.journal_dir / next(x["stored_path"] for x in entry["artifacts"] if x["role"] == "chunk2_records")
            item["result"] = replay_content(c, args.collection, artifact, args.batch, args.dry_run)
        elif entry["mode"] == "payload-only":
            item["result"] = replay_payload(c, args.collection, artifact, args.batch, args.dry_run)
        else:
            item["result"] = replay_tombstones(c, args.collection, artifact, args.batch, args.dry_run)
        reports.append(item)
    after = c.count(args.collection, exact=True).count
    final = {"schema": SCHEMA + ".caught-up.v1", "status": "DRY-RUN" if args.dry_run else "CAUGHT-UP",
             "journal_dir": str(args.journal_dir), "cutoff_sha256": cutoff["cutoff_sha256"],
             "journal_tip_sha256": entries[-1]["entry_sha256"] if entries else cutoff["cutoff_sha256"],
             "candidate": {"collection": args.collection, "manifest": str(args.candidate_manifest),
                           "manifest_sha256": sha256_file(Path(args.candidate_manifest)),
                           "chunk_epoch": 2, "qdrant_url": args.qdrant_url},
             "before_exact_count": before, "after_exact_count": after,
             "operations": reports}
    if args.dry_run:
        print(json.dumps(final, sort_keys=True))
        return 0
    out = Path(args.out)
    if out.exists():
        abort(f"caught-up manifest already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=out.parent, delete=False) as temp:
        json.dump(final, temp, sort_keys=True, indent=2)
        temp.write("\n")
        temp_path = Path(temp.name)
    os.replace(temp_path, out)
    print(json.dumps({"status": "CAUGHT-UP", "manifest": str(out), "entries": len(entries),
                      "before_exact_count": before, "after_exact_count": after}, sort_keys=True))
    return 0


def parser(prog: str | None = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="freeze the chunk@2 base cutoff")
    init.add_argument("--journal-dir", type=Path, required=True)
    init.add_argument("--cutoff-manifest", required=True)
    init.add_argument("--created-at", default="")
    init.set_defaults(func=command_init)
    record = sub.add_parser("record", help="copy and chain one post-cutoff operation")
    record.add_argument("--journal-dir", type=Path, required=True)
    record.add_argument("--operation-id", required=True)
    record.add_argument("--mode", choices=MODES, required=True)
    record.add_argument("--source", required=True, help="pack/L0/L1 source artifact (copied immutably)")
    record.add_argument("--source-git-sha", required=True)
    record.add_argument("--artifact", action="append", default=[], help="ROLE=PATH; repeat as required by mode")
    record.add_argument("--recorded-at", default="")
    record.set_defaults(func=command_record)
    verify = sub.add_parser("verify", help="verify cutoff, chain, and immutable artifacts")
    verify.add_argument("--journal-dir", type=Path, required=True)
    verify.set_defaults(func=command_verify)
    replay = sub.add_parser("replay", help="replay the verified journal into a chunk@2 candidate only")
    replay.add_argument("--journal-dir", type=Path, required=True)
    replay.add_argument("--candidate-manifest", required=True)
    replay.add_argument("--qdrant-url", required=True)
    replay.add_argument("--collection", required=True)
    replay.add_argument("--out", required=True, help="new final caught-up manifest")
    replay.add_argument("--batch", type=int, default=500)
    replay.add_argument("--create-url-index", action="store_true",
                        help="create the required url keyword index on the non-serving candidate")
    replay.add_argument("--dry-run", action="store_true")
    replay.set_defaults(func=command_replay)
    return ap


def main(argv: list[str] | None = None, prog: str = "cyt index delta-journal") -> int:
    args = parser(prog).parse_args(argv)
    # task 864 (review 869 S1): from an aw_e10 T0 to the flip only the drain writes the twin
    from cytir.index import twin_write_hold as _TWH
    _TWH.refuse_cli_write(args, tool="cyt index delta-journal")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
