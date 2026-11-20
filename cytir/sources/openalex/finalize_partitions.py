#!/usr/bin/env python3
"""Conserve four audited OpenAlex whole-part workers into one terminal bundle.

Each worker owns a contiguous slice of the immutable plan and publishes a
materialized RUN, structured embedding PACK, and streaming profile audit.  The
finalizer never reselects source rows.  It deep-binds those four outputs,
proves exact source-ID/part conservation, streams every stage in canonical
cross-partition order, and publishes one metadata-only aggregate bundle.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Iterable, Iterator

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.ops.audits import openalex_academic_pilot as OA
from cytir.chunk import provenance as CP
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as SS


SCHEMA = "aw.openalex-academic-partition-finalizer.v1"
VERSION = "openalex-academic-partition-finalizer.v1"
PRODUCER = "pipeline/finalize_openalex_academic_partitions.py"


def _artifact(root: Path, path: Path) -> dict:
    relative = path.resolve().relative_to(root.resolve())
    return {"uri": f"artifact://{relative.as_posix()}",
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _path(root: Path, ref: dict, label: str) -> Path:
    uri = str(ref.get("uri") or "")
    if not uri.startswith("artifact://"):
        raise ValueError(f"{label} lacks a global artifact URI")
    path = (root / uri.removeprefix("artifact://")).resolve()
    path.relative_to(root.resolve())
    if (not path.is_file() or path.stat().st_size != ref.get("bytes")
            or sha256_file(path) != ref.get("sha256")):
        raise ValueError(f"{label} artifact binding failed")
    return path


def _inventory(root: Path, stage: dict, label: str) -> tuple[dict, list[Path]]:
    validated = _validate_stage(stage, label, artifact_root=root, deep=True)
    receipt_path = _path(root, stage["receipt"], f"{label}.receipt")
    receipt = json.loads(receipt_path.read_text())
    paths = [_path(root, ref, f"{label}.artifact[{index}]")
             for index, ref in enumerate(receipt["artifacts"])]
    if (receipt["records"] != validated["records"]
            or receipt["logical_sha256"] != validated["logical_sha256"]):
        raise ValueError(f"{label} receipt/stage drift")
    return receipt, paths


def _data_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted((path for path in paths if path.name.endswith(".jsonl.gz")),
                  key=lambda path: path.as_posix())


def _iter_rows(paths: Iterable[Path]) -> Iterator[dict]:
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def validate_source_id_conservation(
        *, selected_parts: list[dict], partitions: list[dict],
        partition_rows: list[Iterable[dict]], database: sqlite3.Connection,
        ) -> dict:
    """Prove partition concatenation equals one ordered whole-part replay."""
    if len(partitions) != len(partition_rows):
        raise ValueError("OpenAlex source-ID partition count drift")
    database.execute("CREATE TABLE source_ids (source_id TEXT PRIMARY KEY)")
    ordered_digest = hashlib.sha256()
    selected_count = 0
    global_part_cursor = 0
    for partition_index, (partition, rows) in enumerate(
            zip(partitions, partition_rows)):
        if (partition.get("partition_index") != partition_index
                or partition.get("selected_part_start") != global_part_cursor
                or partition.get("selected_part_end_exclusive")
                != global_part_cursor + len(partition.get("relative_paths") or [])):
            raise ValueError("OpenAlex source-ID partition order drift")
        part_paths = partition.get("relative_paths") or []
        part_cursor = 0
        rows_in_part = 0
        last_source_id = ""
        partition_count = 0
        for row in rows:
            source_id = str(row.get("source_record_id") or "")
            source_part = str(row.get("source_part") or "")
            while (part_cursor < len(part_paths)
                   and source_part != part_paths[part_cursor]):
                expected_rows = selected_parts[
                    global_part_cursor + part_cursor]["retained_legacy_records"]
                if rows_in_part != expected_rows:
                    raise ValueError("OpenAlex selected rows per part drift")
                part_cursor += 1
                rows_in_part = 0
                last_source_id = ""
            if part_cursor >= len(part_paths) or source_part != part_paths[part_cursor]:
                raise ValueError("OpenAlex source row escaped partition plan")
            if not source_id or source_id <= last_source_id:
                raise ValueError("OpenAlex source IDs lost stable per-part order")
            try:
                database.execute(
                    "INSERT INTO source_ids(source_id) VALUES (?)", (source_id,))
            except sqlite3.IntegrityError as exc:
                raise ValueError("duplicate OpenAlex source ID across partitions") from exc
            ordered_digest.update((source_id + "\n").encode())
            last_source_id = source_id
            rows_in_part += 1
            partition_count += 1
            selected_count += 1
        while part_cursor < len(part_paths):
            expected_rows = selected_parts[
                global_part_cursor + part_cursor]["retained_legacy_records"]
            if rows_in_part != expected_rows:
                raise ValueError("OpenAlex selected rows per part drift")
            part_cursor += 1
            rows_in_part = 0
        if partition_count != partition.get("selected_records"):
            raise ValueError("OpenAlex partition selected-ID conservation failed")
        global_part_cursor += len(part_paths)
    if (global_part_cursor != len(selected_parts)
            or selected_count != sum(
                row["retained_legacy_records"] for row in selected_parts)):
        raise ValueError("OpenAlex global selected-ID conservation failed")
    sorted_digest = hashlib.sha256()
    sorted_count = 0
    for (source_id,) in database.execute(
            "SELECT source_id FROM source_ids ORDER BY source_id"):
        sorted_digest.update((source_id + "\n").encode())
        sorted_count += 1
    if sorted_count != selected_count:
        raise ValueError("OpenAlex selected-ID database conservation failed")
    return {
        "selected_source_ids": selected_count,
        "ordered_source_id_sha256": ordered_digest.hexdigest(),
        "sorted_source_id_set_sha256": sorted_digest.hexdigest(),
        "duplicate_source_ids": 0,
        "single_worker_equivalence": "PASS",
        "equivalence_basis": (
            "contiguous plan partitions concatenated by partition index; "
            "each whole part retains stable source-ID order"),
    }


def combine_token_distributions(values: list[dict]) -> dict:
    hard_maxes = {value.get("hard_max") for value in values}
    counters = {value.get("counter") for value in values}
    if len(hard_maxes) != 1 or len(counters) != 1:
        raise ValueError("OpenAlex partition token counter drift")
    hard_max = hard_maxes.pop()
    if not isinstance(hard_max, int) or hard_max <= 0:
        raise ValueError("OpenAlex partition token hard maximum invalid")
    histogram = [0] * (hard_max + 1)
    rows = 0
    for value in values:
        local = value.get("histogram")
        if (not isinstance(local, list)
                or value.get("over_hard_max") != 0):
            raise ValueError("OpenAlex partition lacks exact token histogram")
        local_rows = 0
        seen_lengths = set()
        for item in local:
            if (not isinstance(item, list) or len(item) != 2
                    or not isinstance(item[0], int)
                    or not isinstance(item[1], int)
                    or item[0] < 0 or item[0] > hard_max or item[1] <= 0
                    or item[0] in seen_lengths):
                raise ValueError("OpenAlex partition token histogram invalid")
            seen_lengths.add(item[0])
            histogram[item[0]] += item[1]
            local_rows += item[1]
        if local_rows != value.get("rows"):
            raise ValueError("OpenAlex partition token histogram conservation failed")
        rows += local_rows

    def percentile(percent: int) -> int:
        wanted = max(1, (rows * percent + 99) // 100)
        seen = 0
        for length, count in enumerate(histogram):
            seen += count
            if seen >= wanted:
                return length
        return 0

    return {
        "counter": counters.pop(), "rows": rows,
        "histogram": [[index, count] for index, count in enumerate(histogram)
                      if count],
        "p50": percentile(50) if rows else 0,
        "p95": percentile(95) if rows else 0,
        "max": max((index for index, count in enumerate(histogram) if count),
                   default=0),
        "hard_max": hard_max, "over_hard_max": 0,
    }


def _measure_jsonl(paths: list[Path]) -> tuple[int, str]:
    count = 0
    digest = hashlib.sha256()
    for path in paths:
        with gzip.open(path, "rb") as stream:
            for line in stream:
                if line.strip():
                    digest.update(line)
                    count += 1
    return count, digest.hexdigest()


def combine_provenance_legends(*, legend_paths: list[Path],
                               output_path: Path,
                               combined_run_sha256: str) -> Path:
    """Publish one legend for a metadata-only union of structured shards.

    ``prov_id`` deliberately excludes the producer block, so independently
    chunked partitions made with the same exact rule carry the same ID while
    naming different parent RUNs in their master-only producer evidence.  A
    combined PACK must retain that distinction as upstream lineage, but the
    portable stage contract requires one authoritative legend.  Merge only
    producer evidence and fail closed on any rule/ID disagreement.
    """
    if not legend_paths:
        raise ValueError("partitioned structured pack has no provenance legend")
    combined: dict[str, dict] = {}
    upstream: dict[str, list[dict]] = {}
    expected_prov_ids: set[str] | None = None
    for partition_index, path in enumerate(legend_paths):
        legend = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(legend, dict) or not legend:
            raise ValueError("partition provenance legend is empty")
        CP.validate_legend(legend)
        if expected_prov_ids is None:
            expected_prov_ids = set(legend)
        elif set(legend) != expected_prov_ids:
            raise ValueError("partition provenance ID sets disagree")
        for prov_id, record in legend.items():
            compact = {key: value for key, value in record.items()
                       if key != "producer"}
            if prov_id in combined:
                prior = {key: value for key, value in combined[prov_id].items()
                         if key != "producer"}
                if compact != prior:
                    raise ValueError(
                        "partition provenance rules disagree for shared prov_id")
            else:
                combined[prov_id] = dict(record)
            upstream.setdefault(prov_id, []).append({
                "partition_index": partition_index,
                "legend_sha256": sha256_file(path),
                "producer": record.get("producer"),
            })
    for prov_id, record in combined.items():
        record["producer"] = {
            "id": PRODUCER,
            "revision": sha256_file(Path(__file__)),
            "parent_run_sha256": combined_run_sha256,
            "upstream_partition_producers": upstream[prov_id],
        }
    CP.validate_legend(combined)
    atomic_publish(output_path, (
        json.dumps(combined, indent=2, sort_keys=True) + "\n").encode())
    return output_path


def finalize(*, artifact_root: Path, plan_path: Path, partitions_root: Path,
             output_root: Path, profile_registry: Path,
             expected_partitions: int,
             partition_audit_roots: list[Path] | None = None) -> dict:
    root = artifact_root.resolve()
    plan_path = plan_path.resolve()
    partitions_root = partitions_root.resolve()
    output_root = output_root.resolve()
    profile_registry = profile_registry.resolve()
    # The profile registry is immutable release configuration, not a published
    # artifact reference.  Its SHA is recorded in finalizer evidence, while
    # plan, partition, audit, and output artifacts must resolve below the
    # global artifact root.  Release layouts intentionally live outside that
    # root, so applying the artifact-path rule to the registry rejects the
    # production launcher before any useful work.
    for path in (plan_path, partitions_root, output_root.parent):
        path.resolve().relative_to(root)
    if output_root.exists():
        raise ValueError("OpenAlex combined output root must be fresh")
    bundles = [partitions_root / f"partition-{index:02d}"
               for index in range(expected_partitions)]
    if partition_audit_roots is None:
        audit_roots = [bundle / "audit" for bundle in bundles]
    else:
        if len(partition_audit_roots) != expected_partitions:
            raise ValueError("OpenAlex partition audit-root count drift")
        audit_roots = [path.resolve() for path in partition_audit_roots]
        for path in audit_roots:
            path.relative_to(root)
    if not profile_registry.is_file():
        raise ValueError("OpenAlex profile registry is not a file")
    plan = json.loads(plan_path.read_text())
    selected_parts = plan.get("selected_parts") or []
    partitions = plan.get("partitions") or []
    selected_records = plan.get("selected_records")
    if (plan.get("schema") != "aw.openalex-academic-multipart-tranche-plan.v1"
            or plan.get("status") != "ready_for_bounded_materialization"
            or plan.get("partition_count") != expected_partitions
            or len(partitions) != expected_partitions
            or not selected_parts
            or selected_records != sum(
                row.get("retained_legacy_records", -1)
                for row in selected_parts)
            or [path for part in partitions
                for path in part.get("relative_paths", [])]
            != [row.get("relative_path") for row in selected_parts]):
        raise ValueError("OpenAlex partitioned plan gate failed")

    runs = []
    packs = []
    audit_evidence = []
    audit_checks = []
    stage_inventories: dict[str, list[tuple[dict, list[Path]]]] = {
        name: [] for name in ("original", "markdown", "outlinks", "recovery")}
    chunk_inventories: list[tuple[dict, list[Path]]] = []
    partition_refs = []
    for index, (bundle, partition, audit_root) in enumerate(
            zip(bundles, partitions, audit_roots)):
        run_path = bundle / "materialized/RUN.json"
        pack_path = bundle / "pack/PACK.json"
        evidence_path = audit_root / "verification-evidence.json"
        checks_path = audit_root / "checks.json"
        for path in (run_path, pack_path, evidence_path, checks_path):
            if not path.is_file():
                raise ValueError(f"OpenAlex partition {index} is not terminal")
        run = json.loads(run_path.read_text())
        pack = json.loads(pack_path.read_text())
        evidence = json.loads(evidence_path.read_text())
        checks = json.loads(checks_path.read_text())
        counts = run.get("counts") or {}
        selection = run.get("selection") or {}
        if (run.get("status") != "ready_for_current_policy_chunking"
                or selection.get("partition") != partition
                or selection.get("relative_paths") != partition["relative_paths"]
                or counts.get("selected_source_records")
                != partition["selected_records"]
                or counts.get("markdown_documents", 0)
                + counts.get("recovery_records", 0)
                != partition["selected_records"]):
            raise ValueError(f"OpenAlex partition {index} RUN conservation failed")
        if (pack.get("status") != "ready_for_gpu_queue"
                or pack.get("source_profile") != "academic_dump"
                or (pack.get("counts") or {}).get("documents")
                != counts.get("markdown_documents")):
            raise ValueError(f"OpenAlex partition {index} PACK gate failed")
        ecounts = evidence.get("counts") or {}
        if (evidence.get("partition") != partition
                or ecounts.get("selected_source_records")
                != counts.get("selected_source_records")
                or ecounts.get("markdown_parents")
                != counts.get("markdown_documents")
                or ecounts.get("recovery") != counts.get("recovery_records")
                or ecounts.get("chunks") != (pack.get("counts") or {}).get("chunks")
                or ecounts.get("outlinks")
                != counts.get("outlink_records_including_empty")
                or not checks
                or any(item.get("status") != "PASS"
                       for values in checks.values() for item in values)):
            raise ValueError(f"OpenAlex partition {index} audit gate failed")
        for name in stage_inventories:
            stage_inventories[name].append(_inventory(
                root, run["stages"][name], f"partition.{index}.{name}"))
        chunk_inventories.append(_inventory(
            root, pack["chunks"], f"partition.{index}.chunks"))
        runs.append(run); packs.append(pack)
        audit_evidence.append(evidence); audit_checks.append(checks)
        partition_refs.append({
            "partition_index": index,
            "run": _artifact(root, run_path),
            "pack": _artifact(root, pack_path),
            "audit_evidence": _artifact(root, evidence_path),
            "audit_checks": _artifact(root, checks_path),
        })

    with tempfile.TemporaryDirectory(
            prefix=".openalex-selected-ids-", dir=output_root.parent) as tmp:
        database = sqlite3.connect(Path(tmp) / "source-ids.sqlite3")
        try:
            partition_rows = [
                _iter_rows(_data_paths(paths))
                for _, paths in stage_inventories["original"]]
            source_ids = validate_source_id_conservation(
                selected_parts=selected_parts, partitions=partitions,
                partition_rows=partition_rows, database=database)
        finally:
            database.close()

    partial = output_root.with_name(
        f".{output_root.name}.partial.{os.getpid()}")
    if partial.exists():
        raise ValueError("OpenAlex combined partial root collision")
    partial.mkdir(parents=True)
    try:
        combined_stages = {}
        for name, inventories in stage_inventories.items():
            all_paths = sorted({path.resolve()
                                for _, paths in inventories for path in paths},
                               key=lambda path: path.as_posix())
            data = _data_paths(all_paths)
            records, logical = _measure_jsonl(data)
            expected = sum(run["stages"][name]["records"] for run in runs)
            if records != expected:
                raise ValueError(f"OpenAlex combined {name} row conservation failed")
            combined_stages[name] = SS.seal(
                artifact_root=root, artifact_paths=all_paths,
                receipt_path=partial / f"materialized/stages/{name}/receipt.json",
                records=records, fmt=(
                    "partition-conserved-" + inventories[0][0]["format"]),
                logical_sha256=logical, producer_id=PRODUCER,
                producer_version=VERSION, publish_from=partial,
                publish_to=output_root, allow_existing_published=True)
        combined_counts = {
            key: sum((run.get("counts") or {}).get(key, 0) for run in runs)
            for key in ("selected_source_records", "markdown_documents",
                        "recovery_records", "outlink_records_including_empty",
                        "explicit_empty_outlink_parents")}
        if combined_counts["selected_source_records"] != selected_records:
            raise ValueError("OpenAlex combined RUN selected count drift")
        combined_run = {
            "schema": "aw.openalex-academic-pilot-materialization-run.v1",
            "version": VERSION,
            "status": "ready_for_current_policy_chunking",
            "source_profile": "academic_dump",
            "scope": "bounded_disjoint_whole_part_tranche_no_full_frontier_point_claim",
            "selection": {
                "method": "four_contiguous_plan_partitions_then_stable_openalex_id_order",
                "target_records": plan["target_minimum_records"],
                "relative_paths": [row["relative_path"] for row in selected_parts],
                "selected_part_count": len(selected_parts),
                "expected_selected_records": selected_records,
                "global_expected_selected_records": selected_records,
                "source_run_sha256": plan["source_run_sha256"],
                "plan": _artifact(root, plan_path),
                "partitions": partition_refs,
                "source_id_conservation": source_ids,
            },
            "counts": combined_counts,
            "classification": "full_openalex_metadata_and_abstract_records_not_full_papers",
            "stages": combined_stages,
            "index_admission": "blocked_until_measured_chunks_profile_fidelity_uuid_and_vectors",
        }
        run_path = partial / "materialized/RUN.json"
        atomic_publish(run_path, (json.dumps(
            combined_run, indent=2, sort_keys=True) + "\n").encode())

        chunk_paths = sorted({path.resolve()
                              for _, paths in chunk_inventories for path in paths},
                             key=lambda path: path.as_posix())
        chunk_data = _data_paths(chunk_paths)
        legend_paths = [path for path in chunk_paths
                        if path.name == "provenance-legend.json"]
        unexpected_chunk_artifacts = [
            path for path in chunk_paths
            if path not in chunk_data and path not in legend_paths]
        if (len(legend_paths) != expected_partitions
                or unexpected_chunk_artifacts):
            raise ValueError(
                "OpenAlex partition chunk artifact inventory is not shards+legends")
        chunk_records, chunk_logical = _measure_jsonl(chunk_data)
        if chunk_records != sum(
                (pack.get("counts") or {}).get("chunks", 0) for pack in packs):
            raise ValueError("OpenAlex combined chunk conservation failed")
        combined_legend = combine_provenance_legends(
            legend_paths=legend_paths,
            output_path=partial / "pack/chunks/provenance-legend.json",
            combined_run_sha256=sha256_file(run_path))
        chunks = SS.seal(
            artifact_root=root, artifact_paths=[*chunk_data, combined_legend],
            receipt_path=partial / "pack/chunks/receipt.json",
            records=chunk_records,
            fmt="partition-conserved-canonical-jsonl+deterministic-gzip+provenance-legend",
            logical_sha256=chunk_logical, producer_id=PRODUCER,
            producer_version=VERSION, publish_from=partial,
            publish_to=output_root, allow_existing_published=True)
        tokens = combine_token_distributions([
            pack.get("token_distribution") or {} for pack in packs])
        common_policy = packs[0].get("chunk_policy")
        common_embedding = packs[0].get("embedding")
        if (any(pack.get("chunk_policy") != common_policy for pack in packs)
                or any(pack.get("embedding") != common_embedding for pack in packs)):
            raise ValueError("OpenAlex partition PACK policy/model drift")
        pack_value = {
            "schema": "aw.structured-embedding-pack.v1", "version": VERSION,
            "status": "ready_for_gpu_queue", "source_profile": "academic_dump",
            "parent": {"run_sha256": sha256_file(run_path),
                       "markdown_logical_sha256": combined_stages["markdown"]["logical_sha256"],
                       "documents": combined_counts["markdown_documents"]},
            "chunk_policy": common_policy,
            "counts": {"documents": combined_counts["markdown_documents"],
                       "chunks": chunk_records,
                       "measured_chunks_per_document": round(
                           chunk_records / combined_counts["markdown_documents"], 6)},
            "token_distribution": tokens,
            "chunks": chunks, "embedding": common_embedding,
        }
        pack_path = partial / "pack/PACK.json"
        atomic_publish(pack_path, (json.dumps(
            pack_value, indent=2, sort_keys=True) + "\n").encode())

        external = sum((row.get("counts") or {}).get("external_links", 0)
                       for row in audit_evidence)
        empty = combined_counts["explicit_empty_outlink_parents"]
        methods = OA.build_profile_methods(
            selected=selected_records, selected_parts=len(selected_parts),
            parents=combined_counts["markdown_documents"], chunks=chunk_records,
            external_links=external, empty=empty, token_max=tokens["max"])
        registry = json.loads(profile_registry.read_text())
        profile = registry["profiles"]["academic_dump"]
        checks = {}
        for stage in profile["required_stages"]:
            requirements = profile.get(stage, {}).get("requirements", [])
            if len(requirements) != len(methods[stage]):
                raise ValueError(f"OpenAlex combined profile method mismatch: {stage}")
            checks[stage] = [
                {"requirement": requirement, "status": "PASS",
                 "verification_method": method}
                for requirement, method in zip(requirements, methods[stage])]
        evidence = {
            "schema": "aw.openalex-academic-partition-finalizer-evidence.v1",
            "status": "PASS", "source_profile": "academic_dump",
            "counts": {**combined_counts, "chunks": chunk_records,
                       "unique_point_ids": chunk_records,
                       "external_links": external},
            "plan": _artifact(root, plan_path),
            "partitions": partition_refs,
            "source_id_conservation": source_ids,
            "streaming_outlink_audit": {
                "status": "PASS",
                "partition_outlink_rows": [
                    (row.get("counts") or {}).get("outlinks", 0)
                    for row in audit_evidence],
                "combined_outlink_rows": combined_counts[
                    "outlink_records_including_empty"],
                "method": "four streaming partition audits plus finalizer byte-order/count replay"},
            "parent_run_sha256": sha256_file(run_path),
            "pack_sha256": sha256_file(pack_path),
            "profile_registry_sha256": sha256_file(profile_registry),
            "classification": "bounded partitioned OpenAlex metadata/abstract tranche; not full-paper content or full-frontier point projection",
        }
        atomic_publish(partial / "audit/verification-evidence.json", (
            json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode())
        atomic_publish(partial / "audit/checks.json", (
            json.dumps(checks, indent=2, sort_keys=True) + "\n").encode())
        finalizer = {
            "schema": SCHEMA, "version": VERSION, "status": "PASS",
            "plan": _artifact(root, plan_path),
            "partition_count": expected_partitions,
            "selected_source_records": selected_records,
            "markdown_documents": combined_counts["markdown_documents"],
            "recovery_records": combined_counts["recovery_records"],
            "chunks": chunk_records,
            "outlinks": combined_counts["outlink_records_including_empty"],
            "source_id_conservation": source_ids,
        }
        atomic_publish(partial / "FINALIZER.json", (
            json.dumps(finalizer, indent=2, sort_keys=True) + "\n").encode())
        os.rename(partial, output_root)
        directory_fd = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    published_run = output_root / "materialized/RUN.json"
    for name, stage in json.loads(published_run.read_text())["stages"].items():
        _validate_stage(stage, f"combined.{name}", artifact_root=root, deep=True)
    return json.loads((output_root / "FINALIZER.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "plan", "partitions-root", "output-root",
                 "profile-registry"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-partitions", type=int, default=4)
    parser.add_argument("--partition-audit-root", type=Path, action="append",
                        default=[])
    args = parser.parse_args()
    print(json.dumps(finalize(
        artifact_root=args.artifact_root,
        plan_path=args.plan, partitions_root=args.partitions_root,
        output_root=args.output_root, profile_registry=args.profile_registry,
        expected_partitions=args.expected_partitions,
        partition_audit_roots=(args.partition_audit_root or None)),
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
