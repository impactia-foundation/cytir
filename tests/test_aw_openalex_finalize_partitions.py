import hashlib
import gzip
import json
from pathlib import Path
import sqlite3

import pytest

from cytir.sources.openalex import finalize_partitions as subject
from cytir.extract import openalex_academic as planner
from cytir.chunk import provenance as CP
from cytir.acquire.artifact_io import canonical_json
from cytir.acquire import stage_receipt as seal_stage_receipt


def _plan_and_rows():
    counts = (2, 1, 3, 1, 2, 2, 4)
    selected = [
        {"relative_path": f"updated_date=2026-08-{index + 1:02d}/part.parquet",
         "retained_legacy_records": count}
        for index, count in enumerate(counts)]
    partitions = planner.partition_parts(selected, partition_count=4)
    partition_rows = []
    for partition in partitions:
        rows = []
        for global_index in range(partition["selected_part_start"],
                                  partition["selected_part_end_exclusive"]):
            path = selected[global_index]["relative_path"]
            for row_index in range(selected[global_index]["retained_legacy_records"]):
                rows.append({"source_part": path,
                             "source_record_id": (
                                 f"https://openalex.org/W{global_index:02d}{row_index:02d}")})
        partition_rows.append(rows)
    return selected, partitions, partition_rows


def test_four_partition_source_ids_equal_single_worker_order_and_set():
    selected, partitions, rows = _plan_and_rows()
    connection = sqlite3.connect(":memory:")
    result = subject.validate_source_id_conservation(
        selected_parts=selected, partitions=partitions,
        partition_rows=rows, database=connection)
    flattened = [row["source_record_id"] for partition in rows for row in partition]
    assert result["selected_source_ids"] == len(flattened)
    assert result["ordered_source_id_sha256"] == hashlib.sha256(
        "".join(value + "\n" for value in flattened).encode()).hexdigest()
    assert result["sorted_source_id_set_sha256"] == hashlib.sha256(
        "".join(value + "\n" for value in sorted(flattened)).encode()).hexdigest()
    assert result["single_worker_equivalence"] == "PASS"


@pytest.mark.parametrize("damage", ("missing", "duplicate"))
def test_source_id_conservation_fails_missing_or_duplicate(damage):
    selected, partitions, rows = _plan_and_rows()
    if damage == "missing":
        rows[1].pop()
    else:
        rows[3][0]["source_record_id"] = rows[0][0]["source_record_id"]
    connection = sqlite3.connect(":memory:")
    with pytest.raises(ValueError, match=(
            "per part drift" if damage == "missing" else "duplicate")):
        subject.validate_source_id_conservation(
            selected_parts=selected, partitions=partitions,
            partition_rows=rows, database=connection)


def test_combined_token_histogram_is_exact_and_rejects_incomplete_input():
    inputs = [
        {"counter": "exact", "rows": 2, "histogram": [[10, 1], [20, 1]],
         "hard_max": 2560, "over_hard_max": 0},
        {"counter": "exact", "rows": 3, "histogram": [[20, 2], [30, 1]],
         "hard_max": 2560, "over_hard_max": 0},
    ]
    combined = subject.combine_token_distributions(inputs)
    assert combined["rows"] == 5
    assert combined["histogram"] == [[10, 1], [20, 3], [30, 1]]
    assert combined["p50"] == 20
    assert combined["p95"] == 30
    assert combined["max"] == 30
    inputs[1].pop("histogram")
    with pytest.raises(ValueError, match="lacks exact token histogram"):
        subject.combine_token_distributions(inputs)


def _sealed_stage(root: Path, bundle: Path, name: str,
                  rows: list[dict]) -> dict:
    body = b"".join((canonical_json(row) + "\n").encode() for row in rows)
    artifact = bundle / "materialized/stages" / name / "data/shard-0000.jsonl.gz"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(gzip.compress(body, mtime=0))
    return seal_stage_receipt.seal(
        artifact_root=root, artifact_paths=[artifact],
        receipt_path=bundle / "materialized/stages" / name / "receipt.json",
        records=len(rows), fmt="canonical-jsonl+deterministic-gzip",
        logical_sha256=hashlib.sha256(body).hexdigest(),
        producer_id="fixture", producer_version="v1")


def _sealed_chunks(root: Path, bundle: Path, index: int) -> dict:
    row = {"point_id": index}
    body = (canonical_json(row) + "\n").encode()
    chunk_root = bundle / "pack/chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    shard = chunk_root / "shard-0000.texts.jsonl.gz"
    shard.write_bytes(gzip.compress(body, mtime=0))
    provenance = CP.build(
        family="generic-structured",
        chunker={"mode": "structured", "policy_path": "data/policy.json",
                 "policy_sha256": "b" * 64,
                 "chunk_tokenizer_revision": "r"},
        channels={"canonical": "hierarchy-plus-body",
                  "dense": "leaf-heading-plus-body",
                  "sparse": "hierarchy-plus-body", "channels_split": True},
        producer={"id": "fixture", "revision": "r",
                  "parent_run_sha256": f"{index:064x}"},
        confidence="stamped")
    legend = chunk_root / "provenance-legend.json"
    legend.write_text(json.dumps(
        {provenance["prov_id"]: provenance}, indent=2, sort_keys=True) + "\n")
    return seal_stage_receipt.seal(
        artifact_root=root, artifact_paths=[shard, legend],
        receipt_path=chunk_root / "receipt.json", records=1,
        fmt="canonical-jsonl+deterministic-gzip+provenance-legend",
        logical_sha256=hashlib.sha256(body).hexdigest(),
        producer_id="fixture", producer_version="v1")


def test_finalizer_publishes_one_atomic_four_partition_bundle(tmp_path):
    selected = [
        {"relative_path": f"part-{index}.parquet",
         "retained_legacy_records": 1}
        for index in range(4)]
    partitions = planner.partition_parts(selected, partition_count=4)
    plan = tmp_path / "plans/PLAN.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps({
        "schema": "aw.openalex-academic-multipart-tranche-plan.v1",
        "version": "openalex-academic-partitioned-tranche-plan.v1",
        "status": "ready_for_bounded_materialization",
        "target_minimum_records": 4, "selected_records": 4,
        "selected_parts": selected, "partition_count": 4,
        "partitions": partitions, "source_run_sha256": "a" * 64,
    }))
    workers = tmp_path / "workers"
    policy = {"id": "p", "sha256": "b" * 64}
    embedding = {"model": "m", "dimension": 1024}
    audit_roots = []
    for index, partition in enumerate(partitions):
        bundle = workers / f"partition-{index:02d}"
        source_id = f"https://openalex.org/W{index}"
        original = _sealed_stage(tmp_path, bundle, "original", [{
            "source_record_id": source_id,
            "source_part": selected[index]["relative_path"],
        }])
        markdown = _sealed_stage(tmp_path, bundle, "markdown", [{
            "document_id": f"d{index}", "source_record_id": source_id,
        }])
        outlinks = _sealed_stage(tmp_path, bundle, "outlinks", [{
            "document_id": f"d{index}", "is_empty": True,
        }])
        recovery = _sealed_stage(tmp_path, bundle, "recovery", [])
        counts = {
            "selected_source_records": 1, "markdown_documents": 1,
            "recovery_records": 0, "outlink_records_including_empty": 1,
            "explicit_empty_outlink_parents": 1,
        }
        run = {
            "schema": "aw.openalex-academic-pilot-materialization-run.v1",
            "status": "ready_for_current_policy_chunking",
            "source_profile": "academic_dump",
            "selection": {"partition": partition,
                          "relative_paths": partition["relative_paths"]},
            "counts": counts,
            "stages": {"original": original, "markdown": markdown,
                       "outlinks": outlinks, "recovery": recovery},
        }
        run_path = bundle / "materialized/RUN.json"
        run_path.write_text(json.dumps(run))
        chunks = _sealed_chunks(tmp_path, bundle, index)
        pack = {
            "schema": "aw.structured-embedding-pack.v1",
            "status": "ready_for_gpu_queue", "source_profile": "academic_dump",
            "counts": {"documents": 1, "chunks": 1},
            "token_distribution": {"counter": "exact", "rows": 1,
                                   "histogram": [[10 + index, 1]],
                                   "hard_max": 2560, "over_hard_max": 0,
                                   "max": 10 + index},
            "chunks": chunks, "chunk_policy": policy, "embedding": embedding,
        }
        (bundle / "pack").mkdir(exist_ok=True)
        (bundle / "pack/PACK.json").write_text(json.dumps(pack))
        audit = ((tmp_path / "recovery-audits/partition-02")
                 if index == 2 else bundle / "audit")
        audit.mkdir(parents=True)
        audit_roots.append(audit)
        (audit / "verification-evidence.json").write_text(json.dumps({
            "partition": partition,
            "counts": {"selected_source_records": 1, "markdown_parents": 1,
                       "recovery": 0, "chunks": 1, "outlinks": 1,
                       "external_links": 0},
        }))
        (audit / "checks.json").write_text(json.dumps({
            "original": [{"status": "PASS"}]}))
    # Immutable release configuration may live outside the global artifact
    # root.  The finalizer records its SHA but never emits it as artifact://.
    registry = tmp_path.parent / f"{tmp_path.name}-profiles.json"
    registry.write_text(json.dumps({"profiles": {"academic_dump": {
        "required_stages": ["original"],
        "original": {"requirements": ["a", "b", "c"]},
    }}}))
    output = tmp_path / "combined"
    result = subject.finalize(
        artifact_root=tmp_path, plan_path=plan, partitions_root=workers,
        output_root=output, profile_registry=registry, expected_partitions=4,
        partition_audit_roots=audit_roots)
    assert result["status"] == "PASS"
    assert result["selected_source_records"] == 4
    assert result["chunks"] == 4
    assert result["source_id_conservation"]["single_worker_equivalence"] == "PASS"
    assert (output / "materialized/RUN.json").is_file()
    assert (output / "pack/PACK.json").is_file()
    combined_receipt = json.loads(
        (output / "pack/chunks/receipt.json").read_text())
    legend_refs = [row for row in combined_receipt["artifacts"]
                   if row["uri"].endswith("/provenance-legend.json")]
    assert len(legend_refs) == 1
    combined_legend = json.loads(
        (output / "pack/chunks/provenance-legend.json").read_text())
    assert len(combined_legend) == 1
    only = next(iter(combined_legend.values()))
    assert len(only["producer"]["upstream_partition_producers"]) == 4
    assert not list(tmp_path.glob(".combined.partial.*"))
    registry.unlink()


def test_combined_provenance_legend_rejects_partition_rule_corruption(tmp_path):
    paths = []
    for index, policy_sha in enumerate(("a" * 64, "b" * 64)):
        provenance = CP.build(
            family="generic-structured",
            chunker={"mode": "structured", "policy_path": "data/policy.json",
                     "policy_sha256": policy_sha,
                     "chunk_tokenizer_revision": "r"},
            channels={"canonical": "hierarchy-plus-body",
                      "dense": "leaf-heading-plus-body",
                      "sparse": "hierarchy-plus-body",
                      "channels_split": True},
            producer={"id": "fixture", "revision": "r",
                      "parent_run_sha256": f"{index:064x}"},
            confidence="stamped")
        path = tmp_path / f"legend-{index}.json"
        path.write_text(json.dumps({provenance["prov_id"]: provenance}))
        paths.append(path)
    with pytest.raises(ValueError, match="ID sets disagree"):
        subject.combine_provenance_legends(
            legend_paths=paths, output_path=tmp_path / "combined.json",
            combined_run_sha256="c" * 64)


def test_finalizer_rejects_incomplete_recovery_audit_mapping(tmp_path):
    with pytest.raises(ValueError, match="audit-root count drift"):
        subject.finalize(
            artifact_root=tmp_path, plan_path=tmp_path / "plan",
            partitions_root=tmp_path / "workers",
            output_root=tmp_path / "combined",
            profile_registry=tmp_path / "profiles",
            expected_partitions=4,
            partition_audit_roots=[tmp_path / "audit"])
