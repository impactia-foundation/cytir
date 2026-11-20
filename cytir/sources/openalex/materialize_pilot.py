#!/usr/bin/env python3
"""Project a deterministic whole-part OpenAlex academic-record tranche.

The preserved parquet snapshot remains L0 authority.  This stage selects the
planned whole preserved parts (or one legacy pilot part), then replays the
pinned selection policy against every part.  Every selected source row is
conserved as canonical JSON before a Markdown document or explicit recovery
record is emitted.  Stage shards are streamed with bounded memory.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from cytir.acquire.gzip_pinned import deterministic_gzip
from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as SS


VERSION = "openalex-academic-pilot-materialization.v2-streaming-sharded"
PARTITION_VERSION = (
    "openalex-academic-pilot-materialization.v3-whole-part-partition")
RESUME_VERSION = (
    "openalex-academic-pilot-materialization.v4-whole-part-resume-checkpoint")
PRODUCER = "pipeline/materialize_openalex_academic_pilot.py"
SOURCE_SCHEMA = "aw.openalex-snapshot-acquisition-run.v1"
OPENALEX_ID = re.compile(r"^https://openalex\.org/W[0-9]+$")
DEFAULT_SHARD_BODY_BYTES = 32 * 1024 * 1024

TARGET_FIELDS = (
    "Computer Science", "Mathematics", "Decision Sciences",
    "Economics, Econometrics and Finance",
    "Business, Management and Accounting",
)
TARGET_COUNTRIES = (
    "CH", "FR", "DE", "CA", "AT", "BE", "NL", "IT", "ES", "SE",
    "DK", "FI", "NO", "IE", "PT", "PL", "LU", "CZ", "GB",
)


def _artifact(path: Path, artifact_root: Path) -> dict:
    path = path.resolve()
    relative = path.relative_to(artifact_root.resolve())
    return {"uri": f"artifact://{relative.as_posix()}",
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


class _StageWriter:
    """Bounded-memory deterministic JSONL+gzip stage writer.

    Numbered shard names preserve append order through the canonical stage
    sealer (which sorts artifact URIs).  The writer holds at most one bounded
    raw shard per stage, so a million-record replay cannot recreate the prior
    source/Markdown/outlink list amplification.
    """

    def __init__(self, *, artifact_root: Path, output_root: Path, name: str,
                 shard_body_bytes: int = DEFAULT_SHARD_BODY_BYTES):
        if shard_body_bytes <= 0:
            raise ValueError("OpenAlex stage shard size must be positive")
        self.artifact_root = artifact_root
        self.output_root = output_root
        self.name = name
        self.shard_body_bytes = shard_body_bytes
        self.buffer = bytearray()
        self.logical = hashlib.sha256()
        self.records = 0
        self.artifact_paths: list[Path] = []

    def append(self, row: dict) -> None:
        line = (canonical_json(row) + "\n").encode()
        self.buffer.extend(line)
        self.logical.update(line)
        self.records += 1
        if len(self.buffer) >= self.shard_body_bytes:
            self._flush()

    def _flush(self, *, allow_empty: bool = False) -> None:
        if not self.buffer and not allow_empty:
            return
        path = (self.output_root / "stages" / self.name / "data" /
                f"shard-{len(self.artifact_paths):06d}.jsonl.gz")
        atomic_publish(path, deterministic_gzip(bytes(self.buffer)))
        self.artifact_paths.append(path)
        self.buffer.clear()

    def seal(self, *, fmt: str = "canonical-jsonl+deterministic-gzip",
             extra_artifact_paths: Iterable[Path] = ()) -> dict:
        self._flush(allow_empty=not self.artifact_paths)
        return SS.seal(
            artifact_root=self.artifact_root,
            artifact_paths=[*extra_artifact_paths, *self.artifact_paths],
            receipt_path=(self.output_root / "stages" / self.name /
                          "receipt.json"),
            records=self.records, fmt=fmt,
            logical_sha256=self.logical.hexdigest(),
            producer_id=PRODUCER, producer_version=VERSION,
        )


def _stage(*, artifact_root: Path, output_root: Path,
           name: str, rows: Iterable[dict]) -> dict:
    """Compatibility helper with the same row semantics as the stream path."""
    writer = _StageWriter(artifact_root=artifact_root,
                          output_root=output_root, name=name)
    for row in rows:
        writer.append(row)
    return writer.seal()


def choose_whole_part(rows: Iterable[dict], target_records: int) -> dict:
    """Choose one whole part deterministically; never cap records inside it."""
    if target_records <= 0:
        raise ValueError("OpenAlex pilot target must be positive")
    candidates = list(rows)
    if not candidates:
        raise ValueError("OpenAlex frontier has no source parts")
    for row in candidates:
        if (not isinstance(row.get("retained_legacy_records"), int)
                or row["retained_legacy_records"] <= 0
                or not str(row.get("relative_path") or "").endswith(".parquet")):
            raise ValueError("invalid OpenAlex frontier row")
    return min(candidates, key=lambda row: (
        abs(row["retained_legacy_records"] - target_records),
        row["retained_legacy_records"], row["relative_path"]))


def planned_relative_paths(plan: dict, *, expected_source_run_sha256: str,
                           target_records: int) -> tuple[list[str], int]:
    schema = plan.get("schema")
    if schema == "aw.openalex-academic-tranche-plan.v1":
        selected = [plan.get("selected_part") or {}]
        selected_records = selected[0].get("retained_legacy_records", 0)
    elif schema == "aw.openalex-academic-multipart-tranche-plan.v1":
        selected = plan.get("selected_parts") or []
        selected_records = plan.get("selected_records", 0)
    else:
        selected = []
        selected_records = 0
    if (schema not in {"aw.openalex-academic-tranche-plan.v1",
                       "aw.openalex-academic-multipart-tranche-plan.v1"}
            or plan.get("status") != "ready_for_bounded_materialization"
            or plan.get("source_run_sha256") != expected_source_run_sha256
            or plan.get("target_minimum_records") != target_records
            or not selected or selected_records < target_records
            or selected_records != sum(
                row.get("retained_legacy_records", -1) for row in selected)):
        raise ValueError("OpenAlex selection plan contract failed")
    paths = [row.get("relative_path") for row in selected]
    if (any(not isinstance(path, str) for path in paths)
            or len(paths) != len(set(paths))):
        raise ValueError("OpenAlex selection plan part identity failed")
    return paths, selected_records


def planned_partition_relative_paths(
        plan: dict, *, expected_source_run_sha256: str,
        target_records: int, partition_index: int | None,
        partition_count: int | None) -> tuple[list[str], int, dict | None]:
    """Return one exact contiguous plan slice or the legacy full selection."""
    paths, selected_records = planned_relative_paths(
        plan, expected_source_run_sha256=expected_source_run_sha256,
        target_records=target_records)
    if partition_index is None and partition_count is None:
        return paths, selected_records, None
    if (partition_index is None or partition_count is None
            or partition_count <= 1 or not 0 <= partition_index < partition_count):
        raise ValueError("OpenAlex partition index/count must be declared together")
    partitions = plan.get("partitions") or []
    if (plan.get("partition_count") != partition_count
            or len(partitions) != partition_count):
        raise ValueError("OpenAlex plan partition count drift")
    flattened: list[str] = []
    partition_records = 0
    cursor = 0
    selected = plan.get("selected_parts") or []
    for expected_index, part in enumerate(partitions):
        part_paths = part.get("relative_paths") or []
        start = part.get("selected_part_start")
        end = part.get("selected_part_end_exclusive")
        if (part.get("partition_index") != expected_index
                or part.get("partition_count") != partition_count
                or start != cursor or end != cursor + len(part_paths)
                or part.get("selected_part_count") != len(part_paths)
                or part.get("global_selected_records") != selected_records
                or part_paths != paths[start:end]
                or part.get("selected_records") != sum(
                    row.get("retained_legacy_records", -1)
                    for row in selected[start:end])
                or part.get("selection_logical_sha256") != hashlib.sha256(
                    "".join(
                        f"{path}\t{row['retained_legacy_records']}\n"
                        for path, row in zip(part_paths, selected[start:end])
                    ).encode()).hexdigest()):
            raise ValueError("OpenAlex plan partition identity/conservation failed")
        flattened.extend(part_paths)
        partition_records += int(part.get("selected_records") or 0)
        cursor = end
    if (flattened != paths or cursor != len(paths)
            or partition_records != selected_records):
        raise ValueError("OpenAlex plan partition union drift")
    descriptor = dict(partitions[partition_index])
    descriptor["global_selected_records"] = selected_records
    return (list(descriptor["relative_paths"]),
            int(descriptor["selected_records"]), descriptor)


def reconstruct_abstract(value: object) -> str:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        return ""
    positions: dict[int, str] = {}
    for word, indexes in value.items():
        if not isinstance(word, str) or not isinstance(indexes, list):
            return ""
        for index in indexes:
            if not isinstance(index, int) or index < 0 or index in positions:
                return ""
            positions[index] = word
    return " ".join(positions[index] for index in sorted(positions))


def _selection_sql() -> str:
    """The one selection rule lives in the package (ADR-102 ladder, task 1250)."""
    from cytir.extract.openalex_academic import selection_sql
    return selection_sql()


def query_selected_records(parquet_path: Path) -> list[dict]:
    import duckdb  # Lazy: pure contract tests do not require the runtime wheel.
    connection = duckdb.connect()
    try:
        values = [json.loads(row[0]) for row in connection.execute(
            _selection_sql(), [str(parquet_path)]).fetchall()]
    finally:
        connection.close()
    selected = []
    for value in values:
        abstract = reconstruct_abstract(value.get("abstract_inverted_index"))
        url = (value.get("doi")
               or (value.get("primary_location") or {}).get("landing_page_url")
               or (value.get("best_oa_location") or {}).get("landing_page_url"))
        if len(abstract) >= 40 and isinstance(url, str) and url.startswith("http"):
            selected.append(value)
    return selected


def _walk_urls(value: object, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            yield value, path
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_urls(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _walk_urls(value[key], f"{path}.{key}")


def _names(values: object, *path: str) -> list[str]:
    rows = values if isinstance(values, list) else []
    result = []
    for row in rows:
        current: object = row
        for key in path:
            current = current.get(key) if isinstance(current, dict) else None
        if isinstance(current, str) and current.strip():
            result.append(current.strip())
    return result


def render_record(raw: dict, *, relative_path: str,
                  parent_sha256: str) -> tuple[dict, list[dict]]:
    source_id = str(raw.get("id") or "")
    if not OPENALEX_ID.fullmatch(source_id):
        raise ValueError("OpenAlex work lacks stable source ID")
    title = str(raw.get("title") or raw.get("display_name") or "").strip()
    abstract = reconstruct_abstract(raw.get("abstract_inverted_index"))
    if not title or len(abstract) < 40:
        raise ValueError("OpenAlex work lacks title/full abstract")
    document_id = hashlib.sha256(
        f"openalex-academic-work\0{source_id}".encode()).hexdigest()
    primary = raw.get("primary_location") or {}
    source = primary.get("source") or {}
    topic = raw.get("primary_topic") or {}
    authors = _names(raw.get("authorships"), "author", "display_name")
    topics = _names(raw.get("topics"), "display_name")
    keywords = _names(raw.get("keywords"), "display_name")
    lines = [f"# {title}", "", f"OpenAlex work: {source_id}"]
    for label, value in (
        ("DOI", raw.get("doi")),
        ("Publication date", raw.get("publication_date")),
        ("Language", raw.get("language")),
        ("Work type", raw.get("type")),
        ("Venue", source.get("display_name")),
        ("Primary topic", topic.get("display_name")),
        ("Field", (topic.get("field") or {}).get("display_name")),
        ("Subfield", (topic.get("subfield") or {}).get("display_name")),
    ):
        if value is not None and str(value).strip():
            lines.append(f"- {label}: {value}")
    if authors:
        lines.extend(("", "## Authors", "", "; ".join(authors)))
    if topics:
        lines.extend(("", "## Topics", "", "; ".join(topics)))
    if keywords:
        lines.extend(("", "## Keywords", "", "; ".join(keywords)))
    lines.extend(("", "## Abstract", "", abstract))
    markdown = "\n".join(lines).strip() + "\n"
    links: dict[str, list[str]] = {}
    for url, context in _walk_urls(raw):
        links.setdefault(url, []).append(context)
    outlinks = [{
        "document_id": document_id, "source_url": source_id,
        "ordinal": ordinal, "target_url": url,
        "relation_contexts": sorted(set(contexts)), "is_empty": False,
    } for ordinal, (url, contexts) in enumerate(sorted(links.items()))]
    if not outlinks:
        outlinks = [{"document_id": document_id, "source_url": source_id,
                     "ordinal": 0, "target_url": None,
                     "relation_contexts": [], "is_empty": True}]
    row = {
        "document_id": document_id, "url": source_id,
        "lang": raw.get("language") or "en",
        "title": title, "markdown": markdown,
        "source_profile": "academic_dump",
        "source_record_id": source_id,
        "parent_archive": {"relative_path": relative_path,
                           "sha256": parent_sha256},
        "source_metadata": {
            "openalex_id": source_id, "doi": raw.get("doi"),
            "publication_date": raw.get("publication_date"),
            "publication_year": raw.get("publication_year"),
            "work_type": raw.get("type"),
            "metadata_license": "CC0-1.0",
            "underlying_work_license": primary.get("license"),
            "underlying_work_license_id": primary.get("license_id"),
            "underlying_work_version": primary.get("version"),
            "record_classification":
                "full_openalex_metadata_and_abstract_not_full_paper_content",
        },
        "full_prechunk_content": True,
    }
    return row, outlinks


def _source_evidence(*, source_root: Path, source_run: Path,
                     artifact_root: Path, expected_source_run_sha256: str,
                     target_records: int,
                     selected_relative_paths: list[str] | None = None,
                     body_overrides: dict[str, Path] | None = None,
                     ) -> tuple[dict, list[tuple[dict, Path, Path]], list[Path]]:
    if sha256_file(source_run) != expected_source_run_sha256:
        raise ValueError("OpenAlex portable source RUN identity drift")
    run = json.loads(source_run.read_text())
    counts = run.get("counts") or {}
    if (run.get("schema") != SOURCE_SCHEMA or run.get("status") != "complete"
            or counts.get("selected_parts") != 2216
            or counts.get("selected_source_bytes") != 724933693642):
        raise ValueError("OpenAlex source RUN is not exact terminal L0")
    _validate_stage(run["stages"]["original"], "source.original",
                    artifact_root=artifact_root, deep=False)
    frontier = source_root / "frontier.jsonl"
    rows = [json.loads(line) for line in frontier.read_text().splitlines()]
    if len(rows) != 2216:
        raise ValueError("OpenAlex frontier part conservation failed")
    if selected_relative_paths is None:
        selected_parts = [choose_whole_part(rows, target_records)]
    else:
        if (not selected_relative_paths
                or len(selected_relative_paths) != len(set(selected_relative_paths))):
            raise ValueError("OpenAlex planned parts must be nonempty and unique")
        by_path = {row.get("relative_path"): row for row in rows}
        try:
            selected_parts = [by_path[path] for path in selected_relative_paths]
        except KeyError as exc:
            raise ValueError("OpenAlex planned part is missing from frontier") from exc
    evidence_paths = [source_run, frontier]
    part_evidence = []
    for selected_part in selected_parts:
        relative = selected_part["relative_path"]
        key = hashlib.sha256(relative.encode()).hexdigest()
        receipt_path = source_root / "receipts" / key[:2] / f"{key}.json"
        receipt = json.loads(receipt_path.read_text())
        original_body = source_root / "original" / "parquet" / relative
        body = (body_overrides or {}).get(relative, original_body)
        body_ref = receipt.get("body") or {}
        if (receipt.get("schema") != "aw.openalex-parquet-part-receipt.v1"
                or receipt.get("status") != "complete"
                or receipt.get("relative_path") != relative
                or receipt.get("retained_legacy_records")
                != selected_part["retained_legacy_records"]
                or body_ref.get("bytes") != selected_part["expected_bytes"]
                or not body.is_file()
                or body.stat().st_size != body_ref.get("bytes")
                or sha256_file(body) != body_ref.get("sha256")):
            raise ValueError("OpenAlex pilot part receipt/body mismatch")
        part_evidence.append((selected_part, receipt_path, body))
        evidence_paths.append(receipt_path)
        if body == original_body:
            evidence_paths.append(body)
    return run, part_evidence, evidence_paths


def _resume_shard_path(ref: dict, artifact_root: Path) -> Path:
    uri = ref.get("uri")
    if not isinstance(uri, str) or not uri.startswith("artifact://"):
        raise ValueError("OpenAlex resume shard lacks global artifact URI")
    path = artifact_root / uri.removeprefix("artifact://")
    if (not path.is_file() or path.stat().st_size != ref.get("bytes")
            or sha256_file(path) != ref.get("sha256")):
        raise ValueError("OpenAlex resume shard identity drift")
    return path


def _iter_gzip_rows(paths: Iterable[Path]):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def _migrate_resume_prefix(*, resume: dict, artifact_root: Path,
                           source_writer: _StageWriter,
                           markdown_writer: _StageWriter,
                           outlink_writer: _StageWriter,
                           recovery_writer: _StageWriter,
                           selected_paths: list[str]) -> tuple[set[str], dict]:
    if (resume.get("schema") != "aw.openalex-partial-resume.v2"
            or resume.get("status") not in {"ready", "no_reusable_prefix"}
            or resume.get("completed_parts", []) + resume.get("remaining_parts", [])
            != selected_paths or resume.get("duplicate_ids") != 0):
        raise ValueError("OpenAlex resume prefix/plan contract failed")
    completed_parts = set(resume["completed_parts"])
    migration = resume.get("migration") or {}
    shard_refs = resume.get("stage_shards") or {}
    if set(shard_refs) != {"source_records", "markdown", "outlinks", "recovery"}:
        raise ValueError("OpenAlex resume stage inventory is incomplete")
    if not completed_parts:
        empty_sha = hashlib.sha256().hexdigest()
        if (resume.get("completed_records") != 0
                or any((resume.get("migration", {}).get(name) or {}).get(
                    "reused_records") != 0 for name in shard_refs)
                or any((resume.get("migration", {}).get(name) or {}).get(
                    "logical_sha256") != empty_sha for name in shard_refs)):
            raise ValueError("OpenAlex zero-prefix resume evidence drift")
        return set(), {
            "source_records": 0, "markdown_documents": 0,
            "outlink_records_including_empty": 0, "recovery_records": 0,
            "explicit_empty_outlink_parents": 0,
        }
    paths = {name: [_resume_shard_path(ref, artifact_root)
                    for ref in shard_refs[name]] for name in shard_refs}

    source_ids: set[str] = set()
    for row in _iter_gzip_rows(paths["source_records"]):
        if row.get("source_part") in completed_parts:
            source_id = row.get("source_record_id")
            if not isinstance(source_id, str) or source_id in source_ids:
                raise ValueError("OpenAlex migrated prefix has duplicate source ID")
            source_ids.add(source_id)
            source_writer.append(row)
    markdown_ids: set[str] = set()
    document_ids: set[str] = set()
    for row in _iter_gzip_rows(paths["markdown"]):
        if row.get("source_record_id") in source_ids:
            markdown_ids.add(row["source_record_id"])
            document_ids.add(row["document_id"])
            markdown_writer.append(row)
    recovery_ids: set[str] = set()
    for row in _iter_gzip_rows(paths["recovery"]):
        if row.get("source_record_id") in source_ids:
            recovery_ids.add(row["source_record_id"])
            recovery_writer.append(row)
    outlink_parents: set[str] = set()
    empty_parents = 0
    for row in _iter_gzip_rows(paths["outlinks"]):
        if row.get("document_id") in document_ids:
            outlink_parents.add(row["document_id"])
            empty_parents += bool(row.get("is_empty"))
            outlink_writer.append(row)
    if (markdown_ids & recovery_ids or markdown_ids | recovery_ids != source_ids
            or outlink_parents != document_ids):
        raise ValueError("OpenAlex migrated prefix stage conservation failed")
    writer_by_name = {
        "source_records": source_writer, "markdown": markdown_writer,
        "outlinks": outlink_writer, "recovery": recovery_writer,
    }
    for name, writer in writer_by_name.items():
        expected = migration.get(name) or {}
        if (writer.records != expected.get("reused_records")
                or writer.logical.hexdigest() != expected.get("logical_sha256")):
            raise ValueError(f"OpenAlex migrated {name} prefix equivalence failed")
    return source_ids, {
        "source_records": source_writer.records,
        "markdown_documents": markdown_writer.records,
        "outlink_records_including_empty": outlink_writer.records,
        "recovery_records": recovery_writer.records,
        "explicit_empty_outlink_parents": empty_parents,
    }


def _prefetched_batch(*, manifest: dict, manifest_sha256: str,
                      resume_sha256: str, remaining_paths: list[str],
                      source_root: Path) -> tuple[list[str], dict[str, Path]]:
    parts = manifest.get("parts") or []
    paths = [row.get("relative_path") for row in parts]
    if (manifest.get("schema") != "aw.openalex-nvme-prefetch.v1"
            or manifest.get("status") != "complete" or not paths
            or paths != remaining_paths[:len(paths)]
            or manifest.get("resume_manifest_sha256") != resume_sha256):
        raise ValueError("OpenAlex NVMe prefetch/resume contract failed")
    result: dict[str, Path] = {}
    for row in parts:
        relative = row["relative_path"]
        key = hashlib.sha256(relative.encode()).hexdigest()
        receipt_path = source_root / "receipts" / key[:2] / f"{key}.json"
        cache_path = Path(row.get("cache_path") or "")
        if (not receipt_path.is_file()
                or sha256_file(receipt_path) != row.get("source_receipt_sha256")
                or not cache_path.is_file()
                or cache_path.stat().st_size != row.get("bytes")
                or sha256_file(cache_path) != row.get("body_sha256")):
            raise ValueError("OpenAlex NVMe prefetched part identity drift")
        result[relative] = cache_path
    return paths, result


def materialize(*, source_root: Path, source_run: Path, output_root: Path,
                artifact_root: Path, expected_source_run_sha256: str,
                target_records: int, selection_plan: Path | None = None,
                expected_selection_plan_sha256: str | None = None,
                partition_index: int | None = None,
                partition_count: int | None = None,
                resume_manifest: Path | None = None,
                expected_resume_manifest_sha256: str | None = None,
                prefetch_manifest: Path | None = None,
                expected_prefetch_manifest_sha256: str | None = None) -> dict:
    if output_root.exists():
        raise ValueError("OpenAlex pilot output root must be fresh")
    selected_relative_paths = None
    planned_expected = None
    partition = None
    plan_ref = None
    if selection_plan is not None:
        if (not expected_selection_plan_sha256
                or sha256_file(selection_plan) != expected_selection_plan_sha256):
            raise ValueError("OpenAlex selection plan identity drift")
        plan = json.loads(selection_plan.read_text())
        selected_relative_paths, planned_expected, partition = (
            planned_partition_relative_paths(
                plan, expected_source_run_sha256=expected_source_run_sha256,
                target_records=target_records,
                partition_index=partition_index,
                partition_count=partition_count))
        plan_ref = _artifact(selection_plan, artifact_root)
    elif partition_index is not None or partition_count is not None:
        raise ValueError("OpenAlex partitioning requires a selection plan")
    resume = prefetch = None
    resume_ref = prefetch_ref = None
    batch_paths = selected_relative_paths
    cache_paths: dict[str, Path] = {}
    if resume_manifest is not None:
        if (not expected_resume_manifest_sha256
                or sha256_file(resume_manifest) != expected_resume_manifest_sha256
                or prefetch_manifest is None
                or not expected_prefetch_manifest_sha256
                or sha256_file(prefetch_manifest)
                != expected_prefetch_manifest_sha256):
            raise ValueError("OpenAlex resume/prefetch manifest identity drift")
        resume = json.loads(resume_manifest.read_text())
        if (resume.get("plan_sha256") != expected_selection_plan_sha256
                or resume.get("partition_index") != partition_index):
            raise ValueError("OpenAlex resume manifest selection identity drift")
        prefetch = json.loads(prefetch_manifest.read_text())
        batch_paths, cache_paths = _prefetched_batch(
            manifest=prefetch,
            manifest_sha256=expected_prefetch_manifest_sha256,
            resume_sha256=expected_resume_manifest_sha256,
            remaining_paths=resume.get("remaining_parts") or [],
            source_root=source_root)
        resume_ref = _artifact(resume_manifest, artifact_root)
        prefetch_ref = _artifact(prefetch_manifest, artifact_root)
    elif prefetch_manifest is not None:
        raise ValueError("OpenAlex prefetch requires a resume manifest")

    _, part_evidence, evidence_paths = _source_evidence(
        source_root=source_root, source_run=source_run,
        artifact_root=artifact_root,
        expected_source_run_sha256=expected_source_run_sha256,
        target_records=target_records,
        selected_relative_paths=batch_paths, body_overrides=cache_paths)
    if selection_plan is not None:
        evidence_paths.append(selection_plan)
    if resume_manifest is not None:
        evidence_paths.extend((resume_manifest, prefetch_manifest))
    batch_expected = sum(
        part["retained_legacy_records"] for part, _, _ in part_evidence)
    expected = batch_expected + int(
        (resume or {}).get("completed_records") or 0)
    if resume is None and planned_expected is not None and expected != planned_expected:
        raise ValueError("OpenAlex partition selected-record count drift")

    output_root.mkdir(parents=True)
    source_writer = _StageWriter(
        artifact_root=artifact_root, output_root=output_root,
        name="source_records")
    markdown_writer = _StageWriter(
        artifact_root=artifact_root, output_root=output_root, name="markdown")
    outlink_writer = _StageWriter(
        artifact_root=artifact_root, output_root=output_root, name="outlinks")
    recovery_writer = _StageWriter(
        artifact_root=artifact_root, output_root=output_root, name="recovery")
    all_ids: set[str] = set()
    markdown_documents = recovery_records = outlink_records = empty_parents = 0
    if resume is not None:
        all_ids, migrated = _migrate_resume_prefix(
            resume=resume, artifact_root=artifact_root,
            source_writer=source_writer, markdown_writer=markdown_writer,
            outlink_writer=outlink_writer, recovery_writer=recovery_writer,
            selected_paths=selected_relative_paths)
        markdown_documents = migrated["markdown_documents"]
        recovery_records = migrated["recovery_records"]
        outlink_records = migrated["outlink_records_including_empty"]
        empty_parents = migrated["explicit_empty_outlink_parents"]
    for selected_part, receipt_path, body in part_evidence:
        query_body = cache_paths.get(selected_part["relative_path"], body)
        raw_rows = query_selected_records(query_body)
        part_expected = selected_part["retained_legacy_records"]
        if len(raw_rows) != part_expected:
            raise ValueError(
                f"OpenAlex replay selection drift for {selected_part['relative_path']}: "
                f"{len(raw_rows)} != {part_expected}")
        ids = [str(row.get("id") or "") for row in raw_rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("OpenAlex part IDs are not stable ordered unique IDs")
        parent_sha = sha256_file(body)
        receipt_ref = _artifact(receipt_path, artifact_root)
        for raw in raw_rows:
            source_id = str(raw.get("id") or "")
            if source_id in all_ids:
                raise ValueError("OpenAlex planned parts contain duplicate work IDs")
            all_ids.add(source_id)
            source_writer.append({
                "source_record_id": source_id,
                "source_part": selected_part["relative_path"],
                "source_part_sha256": parent_sha,
                "source_receipt": receipt_ref,
                "metadata_license": "CC0-1.0", "record": raw,
            })
            try:
                markdown, outlinks = render_record(
                    raw, relative_path=selected_part["relative_path"],
                    parent_sha256=parent_sha)
            except Exception as exc:
                recovery_writer.append({
                    "source_record_id": source_id,
                    "source_part": selected_part["relative_path"],
                    "status": "recovery_required",
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                recovery_records += 1
            else:
                markdown_writer.append(markdown)
                markdown_documents += 1
                for outlink in outlinks:
                    outlink_writer.append(outlink)
                    outlink_records += 1
                    empty_parents += bool(outlink["is_empty"])
    if len(all_ids) != expected:
        raise ValueError("OpenAlex selected work ID conservation failed")
    if markdown_documents + recovery_records != expected:
        raise ValueError("OpenAlex Markdown/recovery conservation failed")
    stages = {
        "original": source_writer.seal(
            fmt="exact-openalex-parquet+full-selected-record-json+portable-lineage",
            extra_artifact_paths=evidence_paths),
        "markdown": markdown_writer.seal(),
        "outlinks": outlink_writer.seal(),
        "recovery": recovery_writer.seal(),
    }
    remaining_after = ((resume or {}).get("remaining_parts") or [
        part["relative_path"] for part, _, _ in part_evidence])[len(part_evidence):]
    terminal = not remaining_after
    result = {
        "schema": "aw.openalex-academic-pilot-materialization-run.v1",
        "version": RESUME_VERSION if resume else PARTITION_VERSION if partition else VERSION,
        "status": ("ready_for_current_policy_chunking" if terminal
                   else "resume_checkpoint_more_parts_required"),
        "source_profile": "academic_dump",
        "scope": ("bounded_disjoint_whole_part_tranche_partition_no_full_frontier_point_claim"
                  if partition else
                  "bounded_disjoint_whole_part_tranche_no_full_frontier_point_claim"
                  if plan_ref else
                  "bounded_whole_part_pilot_no_full_frontier_point_claim"),
        "selection": {
            "method": ("globally_planned_ordered_whole_parts_then_stable_openalex_id_order"
                       if plan_ref else
                       "whole_part_closest_to_target_then_stable_openalex_id_order"),
            "target_records": target_records,
            "relative_paths": ((resume or {}).get("completed_parts", [])
                               + [part["relative_path"]
                                  for part, _, _ in part_evidence]),
            "selected_part_count": len((resume or {}).get("completed_parts", []))
                                   + len(part_evidence),
            "expected_selected_records": expected,
            "global_expected_selected_records": (
                partition["global_selected_records"] if partition else expected),
            "source_run_sha256": expected_source_run_sha256,
            "plan": plan_ref,
            "resume_manifest": resume_ref,
            "prefetch_manifest": prefetch_ref,
            "remaining_parts": remaining_after,
        },
        "counts": {
            "selected_source_records": expected,
            "markdown_documents": markdown_documents,
            "recovery_records": recovery_records,
            "outlink_records_including_empty": outlink_records,
            "explicit_empty_outlink_parents": empty_parents,
        },
        "classification":
            "full_openalex_metadata_and_abstract_records_not_full_papers",
        "stages": stages,
        "index_admission": (
            "blocked_until_measured_chunks_profile_fidelity_uuid_and_vectors"
            if terminal else "blocked_incomplete_resume_checkpoint"),
    }
    if partition:
        result["selection"]["partition"] = partition
    atomic_publish(output_root / "RUN.json",
                   (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    for name, stage in stages.items():
        _validate_stage(stage, name, artifact_root=artifact_root, deep=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "source-run", "output-root", "artifact-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-source-run-sha256", required=True)
    parser.add_argument("--target-records", type=int, default=250)
    parser.add_argument("--selection-plan", type=Path)
    parser.add_argument("--expected-selection-plan-sha256")
    parser.add_argument("--partition-index", type=int)
    parser.add_argument("--partition-count", type=int)
    parser.add_argument("--resume-manifest", type=Path)
    parser.add_argument("--expected-resume-manifest-sha256")
    parser.add_argument("--prefetch-manifest", type=Path)
    parser.add_argument("--expected-prefetch-manifest-sha256")
    args = parser.parse_args()
    print(json.dumps(materialize(
        source_root=args.source_root.resolve(),
        source_run=args.source_run.resolve(),
        output_root=args.output_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        expected_source_run_sha256=args.expected_source_run_sha256,
        target_records=args.target_records,
        selection_plan=(args.selection_plan.resolve()
                        if args.selection_plan else None),
        expected_selection_plan_sha256=args.expected_selection_plan_sha256,
        partition_index=args.partition_index,
        partition_count=args.partition_count,
        resume_manifest=(args.resume_manifest.resolve()
                         if args.resume_manifest else None),
        expected_resume_manifest_sha256=args.expected_resume_manifest_sha256,
        prefetch_manifest=(args.prefetch_manifest.resolve()
                           if args.prefetch_manifest else None),
        expected_prefetch_manifest_sha256=args.expected_prefetch_manifest_sha256),
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
