import json
import gzip
import hashlib

import pytest

from cytir.sources.openalex import materialize_pilot as subject
from cytir.extract import openalex_academic as planner


def test_whole_part_selection_is_stable_and_never_record_capped():
    rows = [
        {"relative_path": "b.parquet", "retained_legacy_records": 240},
        {"relative_path": "a.parquet", "retained_legacy_records": 260},
        {"relative_path": "c.parquet", "retained_legacy_records": 999},
    ]
    assert subject.choose_whole_part(rows, 250) == rows[0]
    assert subject.choose_whole_part(reversed(rows), 250) == rows[0]


def test_planned_part_selection_uses_exact_relative_path():
    rows = [
        {"relative_path": "a.parquet", "retained_legacy_records": 250},
        {"relative_path": "b.parquet", "retained_legacy_records": 10010},
    ]
    selected = [row for row in rows if row["relative_path"] == "b.parquet"]
    assert len(selected) == 1
    assert selected[0]["retained_legacy_records"] >= 10000


def test_planned_parts_conserve_ordered_measured_rows_and_legacy_semantics():
    common = {
        "status": "ready_for_bounded_materialization",
        "source_run_sha256": "a" * 64,
        "target_minimum_records": 100,
    }
    multipart = dict(common,
        schema="aw.openalex-academic-multipart-tranche-plan.v1",
        selected_records=105,
        selected_parts=[
            {"relative_path": "a.parquet", "retained_legacy_records": 40},
            {"relative_path": "b.parquet", "retained_legacy_records": 35},
            {"relative_path": "c.parquet", "retained_legacy_records": 30},
        ])
    assert subject.planned_relative_paths(
        multipart, expected_source_run_sha256="a" * 64,
        target_records=100) == (
            ["a.parquet", "b.parquet", "c.parquet"], 105)
    legacy = dict(common,
        schema="aw.openalex-academic-tranche-plan.v1",
        selected_part={"relative_path": "one.parquet",
                       "retained_legacy_records": 101})
    assert subject.planned_relative_paths(
        legacy, expected_source_run_sha256="a" * 64,
        target_records=100) == (["one.parquet"], 101)


def test_planned_parts_fail_duplicate_or_missing_conservation():
    value = {
        "schema": "aw.openalex-academic-multipart-tranche-plan.v1",
        "status": "ready_for_bounded_materialization",
        "source_run_sha256": "a" * 64,
        "target_minimum_records": 100,
        "selected_records": 100,
        "selected_parts": [
            {"relative_path": "a.parquet", "retained_legacy_records": 50},
            {"relative_path": "a.parquet", "retained_legacy_records": 50},
        ],
    }
    with pytest.raises(ValueError, match="identity"):
        subject.planned_relative_paths(
            value, expected_source_run_sha256="a" * 64,
            target_records=100)
    value["selected_parts"][1]["relative_path"] = "b.parquet"
    value["selected_records"] = 101
    with pytest.raises(ValueError, match="contract"):
        subject.planned_relative_paths(
            value, expected_source_run_sha256="a" * 64,
            target_records=100)


def test_planned_partition_concatenation_equals_legacy_single_worker():
    selected = [
        {"relative_path": f"part-{index}.parquet",
         "retained_legacy_records": count}
        for index, count in enumerate((20, 30, 40, 50, 60, 70))]
    plan = {
        "schema": "aw.openalex-academic-multipart-tranche-plan.v1",
        "status": "ready_for_bounded_materialization",
        "source_run_sha256": "a" * 64,
        "target_minimum_records": 250,
        "selected_records": 270,
        "selected_parts": selected,
        "partition_count": 4,
        "partitions": planner.partition_parts(selected, partition_count=4),
    }
    single_paths, single_count = subject.planned_relative_paths(
        plan, expected_source_run_sha256="a" * 64, target_records=250)
    partition_results = [subject.planned_partition_relative_paths(
        plan, expected_source_run_sha256="a" * 64, target_records=250,
        partition_index=index, partition_count=4) for index in range(4)]
    assert [path for paths, _, _ in partition_results for path in paths] == single_paths
    assert sum(count for _, count, _ in partition_results) == single_count


def test_planned_partition_rejects_descriptor_drift():
    selected = [{"relative_path": f"p{index}", "retained_legacy_records": 1}
                for index in range(4)]
    plan = {
        "schema": "aw.openalex-academic-multipart-tranche-plan.v1",
        "status": "ready_for_bounded_materialization",
        "source_run_sha256": "a" * 64,
        "target_minimum_records": 4, "selected_records": 4,
        "selected_parts": selected, "partition_count": 4,
        "partitions": planner.partition_parts(selected, partition_count=4),
    }
    plan["partitions"][2]["relative_paths"] = ["wrong"]
    with pytest.raises(ValueError, match="identity"):
        subject.planned_partition_relative_paths(
            plan, expected_source_run_sha256="a" * 64, target_records=4,
            partition_index=2, partition_count=4)


def test_abstract_reconstruction_is_full_ordered_and_rejects_collisions():
    inverted = {"second": [1], "first": [0], "fourth": [3], "third": [2]}
    assert subject.reconstruct_abstract(json.dumps(inverted)) == (
        "first second third fourth")
    assert subject.reconstruct_abstract({"a": [0], "b": [0]}) == ""


def _raw_record():
    abstract = (
        "A complete source declared technical abstract with enough content "
        "to pass the explicit pilot fidelity boundary."
    )
    return {
        "id": "https://openalex.org/W123",
        "doi": "https://doi.org/10.1/example",
        "title": "An exact source title",
        "publication_date": "2026-01-01", "publication_year": 2026,
        "language": "en", "type": "article",
        "abstract_inverted_index": {
            word: [index] for index, word in enumerate(abstract.split())},
        "authorships": [{"author": {"display_name": "Ada Example",
                                      "orcid": "https://orcid.org/0000"}}],
        "primary_topic": {
            "display_name": "Robotics",
            "field": {"display_name": "Computer Science"},
            "subfield": {"display_name": "Artificial Intelligence"}},
        "topics": [{"display_name": "Robotics"}],
        "keywords": [{"display_name": "Robot"}],
        "primary_location": {
            "landing_page_url": "https://publisher.example/work",
            "license": "cc-by", "license_id": "cc-by-4.0",
            "version": "publishedVersion",
            "source": {"display_name": "Example Journal"}},
        "best_oa_location": None,
        "referenced_works": ["https://openalex.org/W999"],
    }


def test_render_preserves_full_abstract_identity_license_and_relations():
    raw = _raw_record()
    row, outlinks = subject.render_record(
        raw, relative_path="updated_date=x/part.parquet",
        parent_sha256="a" * 64)
    abstract = subject.reconstruct_abstract(raw["abstract_inverted_index"])
    assert abstract in row["markdown"]
    assert row["source_record_id"] == raw["id"]
    assert row["source_metadata"]["metadata_license"] == "CC0-1.0"
    assert row["source_metadata"]["underlying_work_license"] == "cc-by"
    assert row["source_metadata"]["record_classification"].endswith(
        "not_full_paper_content")
    targets = {item["target_url"] for item in outlinks}
    assert raw["doi"] in targets
    assert "https://publisher.example/work" in targets
    assert "https://openalex.org/W999" in targets


def test_render_rejects_missing_stable_identity_or_full_abstract():
    raw = _raw_record(); raw["id"] = "W123"
    with pytest.raises(ValueError, match="stable source ID"):
        subject.render_record(raw, relative_path="x.parquet",
                              parent_sha256="a" * 64)
    raw = _raw_record(); raw["abstract_inverted_index"] = {"thin": [0]}
    with pytest.raises(ValueError, match="full abstract"):
        subject.render_record(raw, relative_path="x.parquet",
                              parent_sha256="a" * 64)


def test_streaming_stage_writer_preserves_order_conservation_and_empty_stage(
        tmp_path):
    output = tmp_path / "materialized"
    output.mkdir()
    rows = [{"ordinal": index, "value": "x" * 32} for index in range(3)]
    writer = subject._StageWriter(
        artifact_root=tmp_path, output_root=output, name="markdown",
        shard_body_bytes=1)
    for row in rows:
        writer.append(row)
    stage = writer.seal()
    receipt_path = tmp_path / stage["receipt"]["uri"].removeprefix(
        "artifact://")
    receipt = json.loads(receipt_path.read_text())
    assert receipt["records"] == 3
    assert len(receipt["artifacts"]) == 3
    assert [item["uri"].rsplit("/", 1)[-1]
            for item in receipt["artifacts"]] == [
                "shard-000000.jsonl.gz", "shard-000001.jsonl.gz",
                "shard-000002.jsonl.gz"]
    replayed = []
    logical = hashlib.sha256()
    for artifact in receipt["artifacts"]:
        path = tmp_path / artifact["uri"].removeprefix("artifact://")
        body = gzip.decompress(path.read_bytes())
        logical.update(body)
        replayed.extend(json.loads(line) for line in body.splitlines())
    assert replayed == rows
    assert logical.hexdigest() == receipt["logical_sha256"]

    empty = subject._StageWriter(
        artifact_root=tmp_path, output_root=output, name="recovery",
        shard_body_bytes=1).seal()
    empty_receipt = json.loads((
        tmp_path / empty["receipt"]["uri"].removeprefix("artifact://")
    ).read_text())
    assert empty_receipt["records"] == 0
    assert len(empty_receipt["artifacts"]) == 1
    empty_path = tmp_path / empty_receipt["artifacts"][0][
        "uri"].removeprefix("artifact://")
    assert gzip.decompress(empty_path.read_bytes()) == b""


def test_legacy_stage_helper_has_identical_logical_row_semantics(tmp_path):
    output = tmp_path / "materialized"
    output.mkdir()
    rows = [{"ordinal": index} for index in range(4)]
    stage = subject._stage(
        artifact_root=tmp_path, output_root=output,
        name="legacy-compatible", rows=rows)
    receipt = json.loads((
        tmp_path / stage["receipt"]["uri"].removeprefix("artifact://")
    ).read_text())
    replayed = []
    logical = hashlib.sha256()
    for artifact in receipt["artifacts"]:
        body = gzip.decompress((
            tmp_path / artifact["uri"].removeprefix("artifact://")
        ).read_bytes())
        logical.update(body)
        replayed.extend(json.loads(line) for line in body.splitlines())
    assert replayed == rows
    assert receipt["records"] == len(rows)
    assert logical.hexdigest() == receipt["logical_sha256"]


def test_prefetched_batch_is_exact_resume_prefix_and_hash_bound(tmp_path):
    source = tmp_path / "source"
    relative = "updated_date=x/part.parquet"
    key = hashlib.sha256(relative.encode()).hexdigest()
    receipt_path = source / "receipts" / key[:2] / f"{key}.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps({"relative_path": relative}))
    cache = tmp_path / "scratch" / "part.parquet"
    cache.parent.mkdir(); cache.write_bytes(b"exact")
    row = {
        "relative_path": relative,
        "source_receipt_sha256": subject.sha256_file(receipt_path),
        "body_sha256": hashlib.sha256(b"exact").hexdigest(),
        "bytes": 5, "cache_path": str(cache),
    }
    manifest = {
        "schema": "aw.openalex-nvme-prefetch.v1", "status": "complete",
        "resume_manifest_sha256": "a" * 64, "parts": [row],
    }
    paths, cache_paths = subject._prefetched_batch(
        manifest=manifest, manifest_sha256="b" * 64,
        resume_sha256="a" * 64,
        remaining_paths=[relative, "next.parquet"], source_root=source)
    assert paths == [relative]
    assert cache_paths == {relative: cache}
    with pytest.raises(ValueError, match="prefetch/resume"):
        subject._prefetched_batch(
            manifest=manifest, manifest_sha256="b" * 64,
            resume_sha256="a" * 64,
            remaining_paths=["next.parquet", relative], source_root=source)


def test_zero_prefix_resume_never_replays_excluded_tail_shards(tmp_path):
    output = tmp_path / "fresh"; output.mkdir()
    writers = {name: subject._StageWriter(
        artifact_root=tmp_path, output_root=output, name=name)
        for name in ("source_records", "markdown", "outlinks", "recovery")}
    empty_sha = hashlib.sha256().hexdigest()
    resume = {
        "schema": "aw.openalex-partial-resume.v2", "status": "ready",
        "completed_parts": [], "remaining_parts": ["p"],
        "completed_records": 0, "duplicate_ids": 0,
        "migration": {name: {"reused_records": 0,
                              "logical_sha256": empty_sha}
                      for name in writers},
        "stage_shards": {name: [{"uri": "artifact://excluded/dangling",
                                  "bytes": 1, "sha256": "0" * 64}]
                         for name in writers},
    }
    ids, counts = subject._migrate_resume_prefix(
        resume=resume, artifact_root=tmp_path,
        source_writer=writers["source_records"],
        markdown_writer=writers["markdown"],
        outlink_writer=writers["outlinks"],
        recovery_writer=writers["recovery"], selected_paths=["p"])
    assert ids == set()
    assert counts["source_records"] == 0
