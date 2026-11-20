"""Moved from the frozen script of the same name by task 1437 (S7); `PRODUCER` stays byte for byte
(sealed receipts carry it) and the old path is a forwarding shim.

Checkpointed exact JORF source-record recovery from sealed DILA archives.

The production identity is the JORFTEXT identifier from the sealed membership
frontier.  Archive filenames, titles, and XML bodies never create identity.
Each increment is an ordered state transition: its explicit suppression list
is applied before its struct/version XML upserts.  This stage recovers exact
source records and relationship IDs only; it does not claim full article text,
Markdown admission, chunks, vectors, or production-point subtraction.
"""
from __future__ import annotations
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import xml.etree.ElementTree as ET

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire.deterministic_gzip import DeterministicGzipWriter
from cytir.acquire import stage_receipt as SS


VERSION = "jorf-exact-archive-recovery.v1"
PRODUCER = "pipeline/recover_jorf_production_from_archives.py"
TARGET_ID = re.compile(r"^JORFTEXT[0-9]+$")
ARCHIVE_NAME = re.compile(r"^(?:Freemium_jorf_global_|JORF_)([0-9]{8})-([0-9]{6})\.tar\.gz$")


def _artifact(path: Path, root: Path) -> dict:
    relative = path.resolve().relative_to(root.resolve())
    return {"uri": f"artifact://{relative.as_posix()}",
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _checked_ref(root: Path, ref: dict, label: str) -> Path:
    uri = str(ref.get("uri") or "")
    if not uri.startswith("artifact://"):
        raise ValueError(f"{label} lacks artifact URI")
    relative = Path(uri.removeprefix("artifact://"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} has unsafe artifact URI")
    path = root / relative
    if (not path.is_file() or path.stat().st_size != ref.get("bytes")
            or sha256_file(path) != ref.get("sha256")):
        raise ValueError(f"{label} bytes/SHA drift")
    return path


def _stage_receipt_from_run(root: Path, run: dict, stage: str) -> tuple[Path, dict]:
    ref = ((run.get("stages") or {}).get(stage) or {}).get("receipt") or {}
    path = _checked_ref(root, ref, f"{stage} stage receipt")
    value = json.loads(path.read_text())
    if value.get("schema") != "aw.stage-receipt.v1":
        raise ValueError(f"{stage} stage receipt schema drift")
    return path, value


def _load_targets(root: Path, identity_run_path: Path) -> tuple[set[str], dict]:
    run = json.loads(identity_run_path.read_text())
    if (run.get("schema") != "aw.jorf-production-membership-frontier-run.v1"
            or run.get("status") != "complete-exact-production-identity-source-recovery-pending"):
        raise ValueError("identity RUN is not the terminal exact JORF frontier")
    _, receipt = _stage_receipt_from_run(root, run, "identity")
    refs = [row for row in receipt.get("artifacts") or []
            if str(row.get("uri") or "").endswith("/frontier/jorftext-source-ids.jsonl.gz")]
    if len(refs) != 1:
        raise ValueError("identity stage has no unique JORFTEXT frontier")
    frontier = _checked_ref(root, refs[0], "JORFTEXT frontier")
    targets: set[str] = set()
    with gzip.open(frontier, "rt", encoding="utf-8") as source:
        for line in source:
            value = json.loads(line)
            target = str(value.get("jorftext_id") or "")
            if not TARGET_ID.fullmatch(target) or target in targets:
                raise ValueError("identity frontier has invalid/duplicate JORFTEXT ID")
            targets.add(target)
    expected = int((run.get("counts") or {}).get("unique_jorftext_source_ids") or -1)
    if len(targets) != expected:
        raise ValueError("identity frontier count drift")
    return targets, _artifact(frontier, root)


def _archive_refs(root: Path, run_path: Path, schema: str) -> tuple[list[dict], dict]:
    run = json.loads(run_path.read_text())
    if (run.get("schema") != schema
            or run.get("status") != "source_bytes_complete_transport_incomplete"):
        raise ValueError(f"legacy L0 RUN is not terminal: {run_path}")
    _, receipt = _stage_receipt_from_run(root, run, "original")
    refs = [row for row in receipt.get("artifacts") or []
            if str(row.get("uri") or "").endswith(".tar.gz")]
    if not refs:
        raise ValueError("legacy L0 receipt has no archive artifacts")
    names = [PurePosixPath(str(row["uri"])).name for row in refs]
    if any(ARCHIVE_NAME.fullmatch(name) is None for name in names):
        raise ValueError("legacy L0 receipt contains an unexpected archive name")
    if names != sorted(names):
        raise ValueError("legacy increment archive receipt is not ordered")
    return refs, _artifact(run_path, root)


def prepare(*, artifact_root: Path, identity_run: Path, global_run: Path,
            increments_run: Path, output: Path, batch_size: int) -> dict:
    root = artifact_root.resolve(); output = output.resolve(); output.relative_to(root)
    if output.exists():
        raise ValueError(f"immutable recovery plan already exists: {output}")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    targets, frontier_ref = _load_targets(root, identity_run.resolve())
    global_refs, global_run_ref = _archive_refs(
        root, global_run.resolve(), "aw.legacy-official-archive-l0-run.v1")
    if len(global_refs) != 1:
        raise ValueError("global L0 must bind exactly one archive")
    increment_refs, increment_run_ref = _archive_refs(
        root, increments_run.resolve(), "aw.legacy-official-increment-set-l0-run.v1")
    sources = []
    for ordinal, ref in enumerate([*global_refs, *increment_refs]):
        sources.append({"archive_ordinal": ordinal,
                        "archive_kind": "global" if ordinal == 0 else "increment",
                        **ref})
    partitions = [{"partition": 0, "archive_start": 0, "archive_end": 1}]
    start = 1
    while start < len(sources):
        end = min(len(sources), start + batch_size)
        partitions.append({"partition": len(partitions),
                           "archive_start": start, "archive_end": end})
        start = end
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": "aw.jorf-exact-archive-recovery-plan.v1", "version": VERSION,
        "status": "complete-partition-scan-pending",
        "counts": {"target_jorftext_ids": len(targets),
                   "archives": len(sources), "partitions": len(partitions),
                   "increment_archives": len(increment_refs)},
        "identity_run": _artifact(identity_run.resolve(), root),
        "identity_frontier": frontier_ref,
        "global_l0_run": global_run_ref,
        "increments_l0_run": increment_run_ref,
        "ordered_archives": sources, "partitions": partitions,
        "semantics": "global base then increments in filename order; within each increment all explicit suppressions precede XML upserts",
        "admission": "recovery plan only; no source-content or vector admission",
    }
    atomic_publish(output / "PLAN.json",
                   (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode())
    return plan


class _HashingReader(io.RawIOBase):
    def __init__(self, source):
        self.source = source; self.digest = hashlib.sha256(); self.bytes = 0

    def readable(self):
        return True

    def read(self, size=-1):
        data = self.source.read(size)
        self.digest.update(data); self.bytes += len(data)
        return data


def _record_kind(name: str) -> str | None:
    normalized = "/" + name.casefold().strip("/") + "/"
    if "/texte/struct/" in normalized:
        return "struct"
    if "/texte/version/" in normalized:
        return "version"
    return None


def _target_from_path(name: str) -> str | None:
    base = PurePosixPath(name.strip()).name
    if base.endswith(".xml"):
        base = base[:-4]
    return base if TARGET_ID.fullmatch(base) else None


def _scan_archive(path: Path, ref: dict, targets: set[str], ordinal: int,
                  writer: DeterministicGzipWriter) -> dict:
    members = upserts = deletions = 0
    with path.open("rb") as raw:
        hashed = _HashingReader(raw)
        with tarfile.open(fileobj=hashed, mode="r|gz") as archive:
            for member_ordinal, member in enumerate(archive):
                members += 1
                # tarfile retains every TarInfo even in stream mode.  A global
                # DILA archive has millions of directory members, so discard
                # the internal history immediately; ``member`` remains live
                # for the current iteration and the stream stays sequential.
                archive.members.clear()
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise ValueError(f"unsafe archive member: {member.name!r}")
                if not member.isfile():
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError("regular archive member cannot be read")
                basename = PurePosixPath(member.name).name.casefold()
                if basename.startswith("liste_suppression_") and basename.endswith(".dat"):
                    payload = handle.read()
                    for line_ordinal, raw_line in enumerate(payload.decode("utf-8").splitlines()):
                        target = _target_from_path(raw_line)
                        if target not in targets:
                            continue
                        kind = _record_kind(raw_line)
                        writer.write({"schema": "aw.jorf-archive-state-event.v1",
                                      "event": "delete", "phase": 0,
                                      "archive_ordinal": ordinal,
                                      "archive_uri": ref["uri"],
                                      "member_ordinal": member_ordinal,
                                      "line_ordinal": line_ordinal,
                                      "member_name": member.name,
                                      "suppressed_path": raw_line,
                                      "jorftext_id": target, "record_kind": kind})
                        deletions += 1
                    continue
                target = _target_from_path(member.name)
                kind = _record_kind(member.name)
                if target not in targets or kind is None:
                    continue
                payload = handle.read()
                writer.write({"schema": "aw.jorf-archive-state-event.v1",
                              "event": "upsert", "phase": 1,
                              "archive_ordinal": ordinal,
                              "archive_uri": ref["uri"],
                              "member_ordinal": member_ordinal,
                              "member_name": member.name,
                              "jorftext_id": target, "record_kind": kind,
                              "xml_bytes": len(payload),
                              "xml_sha256": hashlib.sha256(payload).hexdigest(),
                              "xml_base64": base64.b64encode(payload).decode("ascii")})
                upserts += 1
        # The streaming gzip reader may not request the trailing raw bytes.
        while hashed.read(8 * 1024 * 1024):
            pass
    if hashed.bytes != ref.get("bytes") or hashed.digest.hexdigest() != ref.get("sha256"):
        raise ValueError(f"source archive bytes/SHA drift: {path}")
    return {"archive_members": members, "target_xml_upserts": upserts,
            "target_suppressions": deletions, "source_bytes": hashed.bytes}


def scan_partition(*, artifact_root: Path, plan_path: Path, partition: int,
                   output: Path) -> dict:
    root = artifact_root.resolve(); output = output.resolve(); output.relative_to(root)
    if output.exists():
        raise ValueError(f"immutable partition already exists: {output}")
    partial = output.with_name(f"{output.name}.partial.{os.getpid()}")
    if partial.exists():
        raise ValueError(f"partition partial already exists: {partial}")
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != "aw.jorf-exact-archive-recovery-plan.v1":
        raise ValueError("recovery plan schema drift")
    choices = [row for row in plan.get("partitions") or []
               if row.get("partition") == partition]
    if len(choices) != 1:
        raise ValueError("unknown/duplicate recovery partition")
    targets_path = _checked_ref(root, plan["identity_frontier"], "identity frontier")
    targets = set()
    with gzip.open(targets_path, "rt", encoding="utf-8") as source:
        for line in source:
            targets.add(json.loads(line)["jorftext_id"])
    if len(targets) != (plan.get("counts") or {}).get("target_jorftext_ids"):
        raise ValueError("recovery plan target count drift")
    choice = choices[0]
    selected = (plan.get("ordered_archives") or [])[choice["archive_start"]:choice["archive_end"]]
    partial.mkdir(parents=True, exist_ok=False)
    events = partial / "events.jsonl.gz"
    writer = DeterministicGzipWriter(events)
    totals = {"archive_members": 0, "target_xml_upserts": 0,
              "target_suppressions": 0, "source_bytes": 0}
    try:
        for ref in selected:
            path = root / Path(ref["uri"].removeprefix("artifact://"))
            result = _scan_archive(path, ref, targets, ref["archive_ordinal"], writer)
            for key, value in result.items(): totals[key] += value
    finally:
        writer.close()
    logical = sha256_file(events)
    stage = SS.seal(
        artifact_root=root, artifact_paths=[events],
        receipt_path=partial / "stages/events/receipt.json",
        records=totals["target_xml_upserts"] + totals["target_suppressions"],
        fmt="ordered-jorftext-struct-version-upserts+explicit-suppressions+canonical-jsonl+deterministic-gzip",
        logical_sha256=logical, producer_id=PRODUCER, producer_version=VERSION,
        publish_from=partial, publish_to=output)
    run = {
        "schema": "aw.jorf-exact-archive-recovery-partition-run.v1",
        "version": VERSION, "status": "complete",
        "partition": partition, "archive_start": choice["archive_start"],
        "archive_end": choice["archive_end"], "counts": {**totals, "archives": len(selected)},
        "plan": _artifact(plan_path.resolve(), root), "source_archives": selected,
        "stages": {"events": stage},
        "admission": "ordered source-state events only; no full text or embedding admission",
    }
    atomic_publish(partial / "RUN.json",
                   (json.dumps(run, indent=2, sort_keys=True) + "\n").encode())
    os.rename(partial, output)
    return run


def _xml_id_and_links(payload: bytes, expected: str, kind: str) -> list[dict]:
    root = ET.fromstring(payload)
    expected_root = "TEXTELR" if kind == "struct" else "TEXTE_VERSION"
    if root.tag.split("}")[-1] != expected_root:
        raise ValueError(f"{expected} {kind} XML has unexpected root")
    identifiers = [str(node.text or "").strip() for node in root.findall(".//ID")]
    if expected not in identifiers:
        raise ValueError(f"{expected} {kind} XML lacks exact META ID")
    links = []
    if kind == "struct":
        for node in root.iter():
            tag = node.tag.split("}")[-1]
            linked = str(node.attrib.get("id") or "")
            if tag.startswith("LIEN_") and linked:
                links.append({"relation": tag, "source_id": linked,
                              "attributes": dict(sorted(node.attrib.items()))})
    return links


def finalize(*, artifact_root: Path, plan_path: Path, partitions_root: Path,
             output: Path) -> dict:
    root = artifact_root.resolve(); output = output.resolve(); output.relative_to(root)
    if output.exists():
        raise ValueError(f"immutable recovery output already exists: {output}")
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != "aw.jorf-exact-archive-recovery-plan.v1":
        raise ValueError("recovery plan schema drift")
    target_path = _checked_ref(root, plan["identity_frontier"], "identity frontier")
    targets = []
    with gzip.open(target_path, "rt", encoding="utf-8") as source:
        for line in source: targets.append(json.loads(line)["jorftext_id"])
    state: dict[str, dict[str, dict]] = {target: {} for target in targets}
    history = {target: {"upserts": 0, "suppressions": 0} for target in targets}
    runs = []
    for part in plan.get("partitions") or []:
        run_path = partitions_root / f"partition-{part['partition']:04d}" / "RUN.json"
        value = json.loads(run_path.read_text())
        if (value.get("schema") != "aw.jorf-exact-archive-recovery-partition-run.v1"
                or value.get("status") != "complete"
                or value.get("partition") != part["partition"]):
            raise ValueError(f"partition RUN drift: {run_path}")
        receipt_path, receipt = _stage_receipt_from_run(root, value, "events")
        refs = receipt.get("artifacts") or []
        if len(refs) != 1:
            raise ValueError("partition events receipt is not singular")
        events_path = _checked_ref(root, refs[0], "partition events")
        events = []
        with gzip.open(events_path, "rt", encoding="utf-8") as source:
            for line in source: events.append(json.loads(line))
        events.sort(key=lambda row: (row["archive_ordinal"], row["phase"],
                                     row["member_ordinal"], row.get("line_ordinal", -1)))
        for event in events:
            target = event["jorftext_id"]
            if target not in state:
                raise ValueError("partition event escaped exact identity frontier")
            kind = event.get("record_kind")
            if event["event"] == "delete":
                history[target]["suppressions"] += 1
                if kind in {"struct", "version"}: state[target].pop(kind, None)
                else: state[target].clear()
            elif event["event"] == "upsert" and kind in {"struct", "version"}:
                history[target]["upserts"] += 1; state[target][kind] = event
            else:
                raise ValueError("invalid partition state event")
        runs.append(_artifact(run_path, root)); runs.append(_artifact(receipt_path, root))
    output.mkdir(parents=True, exist_ok=False)
    recovered_path = output / "recovered/jorftext-source-records.jsonl.gz"
    relations_path = output / "recovered/jorftext-relationships.jsonl.gz"
    unresolved_path = output / "unresolved/jorftext-source-ids.jsonl.gz"
    for path in (recovered_path, relations_path, unresolved_path): path.parent.mkdir(parents=True, exist_ok=True)
    recovered_writer = DeterministicGzipWriter(recovered_path)
    relations_writer = DeterministicGzipWriter(relations_path)
    unresolved_writer = DeterministicGzipWriter(unresolved_path)
    recovered = unresolved = relations = 0
    try:
        for target in sorted(targets):
            records = state[target]
            missing = sorted({"struct", "version"} - set(records))
            if missing:
                unresolved_writer.write({"schema": "aw.jorf-unresolved-source-record.v1",
                                         "jorftext_id": target, "missing_record_kinds": missing,
                                         "history": history[target]})
                unresolved += 1; continue
            decoded = {}
            links = []
            for kind in ("struct", "version"):
                event = records[kind]
                payload = base64.b64decode(event["xml_base64"], validate=True)
                if (len(payload) != event["xml_bytes"]
                        or hashlib.sha256(payload).hexdigest() != event["xml_sha256"]):
                    raise ValueError("recovered XML bytes/SHA drift")
                parsed_links = _xml_id_and_links(payload, target, kind)
                links.extend(parsed_links)
                decoded[kind] = {key: event[key] for key in
                                 ("archive_ordinal", "archive_uri", "member_name",
                                  "xml_bytes", "xml_sha256", "xml_base64")}
            recovered_writer.write({"schema": "aw.jorf-exact-source-record.v1",
                                    "jorftext_id": target, "records": decoded,
                                    "history": history[target],
                                    "relationship_count": len(links)})
            recovered += 1
            for ordinal, link in enumerate(links):
                relations_writer.write({"schema": "aw.jorf-source-relationship.v1",
                                        "jorftext_id": target, "ordinal": ordinal, **link})
                relations += 1
    finally:
        recovered_writer.close(); relations_writer.close(); unresolved_writer.close()
    if recovered + unresolved != len(targets):
        raise ValueError("JORF recovery conservation failure")
    artifacts = [recovered_path, relations_path, unresolved_path, *
                 [root / Path(ref["uri"].removeprefix("artifact://")) for ref in runs]]
    logical = hashlib.sha256((canonical_json({"targets": len(targets),
        "recovered": recovered, "unresolved": unresolved,
        "relationships": relations}) + "\n").encode()).hexdigest()
    stage = SS.seal(
        artifact_root=root, artifact_paths=artifacts,
        receipt_path=output / "stages/recovery/receipt.json", records=len(targets),
        fmt="exact-jorftext-struct+version-source-records+ordered-suppressions+relationships+deterministic-gzip",
        logical_sha256=logical, producer_id=PRODUCER, producer_version=VERSION)
    status = "source-records-complete-full-content-pending" if unresolved == 0 \
        else "recovery-frontier"
    run = {
        "schema": "aw.jorf-exact-archive-recovery-run.v1", "version": VERSION,
        "status": status,
        "counts": {"production_jorftext_ids": len(targets), "recovered": recovered,
                   "unresolved": unresolved, "relationships": relations,
                   "partitions": len(plan.get("partitions") or [])},
        "plan": _artifact(plan_path.resolve(), root), "partition_evidence": runs,
        "stages": {"recovery": stage},
        "next_gate": "recover every linked official article/section source record, then full Markdown/outlinks/profile/fidelity/chunks",
        "admission": "source record and relationship recovery only; no title-only reuse, chunks, vectors, point subtraction, or promotion authorization",
    }
    atomic_publish(output / "RUN.json",
                   (json.dumps(run, indent=2, sort_keys=True) + "\n").encode())
    return run
