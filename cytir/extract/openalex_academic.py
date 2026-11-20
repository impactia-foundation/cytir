"""OpenAlex works, from one parquet part of the public snapshot (task 696, slice 7 step 5).

The logic of the old OpenAlex materializer script, `cytir.sources.openalex.materialize_pilot` (the whole-part selection, the SQL
that picks the works, the abstract rebuild and the Markdown a work renders to), moved into the
package, with one change: the selection is the ADR-102 age-scaled ladder (`CAP_SQL`, ported from the N2 driver,
task 1250), not the script's flat `cited >= 1` rule. Rendering is the old script's, byte for byte, per work
(`tests/test_aw_family_openalex_academic_dump.py` pins the sha256 of the works the ladder keeps). What is NOT moved: the sealing
of stage receipts, the resume checkpoint of a partition that crashed (the family door keeps its own
ledger), and the outlink rows (they feed the link graph, which this family does not).

The snapshot is a CC0 dataset (https://docs.openalex.org). A part is a parquet file of about
160k to 400k works. `iter_selected` streams: DuckDB scans the file, filters and sorts in the
engine (spilling to disk as it needs), and Python receives the chosen works in batches, so a part
of any size never sits in memory as a list. DuckDB is imported on first use.

Persons: a work's authors are names of natural persons. `render_record` renders them in the
"Authors" section, as the old script does (the embedded surface of the served rows has them);
`author_names` returns the names for the master-only part of a row (ADR-039). `with_authors=False`
renders the work without that section; no caller in the family uses it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

from cytir.acquire.artifact_io import atomic_publish, sha256_file

OPENALEX_ID = re.compile(r"^https://openalex\.org/W[0-9]+$")
CC0 = "CC0-1.0"
SOURCE_PROFILE = "academic_dump"
BATCH = 1000

TARGET_FIELDS = (
    "Computer Science", "Mathematics", "Decision Sciences",
    "Economics, Econometrics and Finance",
    "Business, Management and Accounting",
)
TARGET_COUNTRIES = (
    "CH", "FR", "DE", "CA", "AT", "BE", "NL", "IT", "ES", "SE",
    "DK", "FI", "NO", "IE", "PT", "PL", "LU", "CZ", "GB",
)
PLAN_SCHEMAS = ("aw.openalex-academic-tranche-plan.v1",
                "aw.openalex-academic-multipart-tranche-plan.v1")

# The tranche planner (task 1273, was the standalone plan script). Plan files it writes
# keep this schema/version vocabulary byte for byte; `planned_relative_paths` below reads them.
SCHEMA = "aw.openalex-academic-tranche-plan.v1"
VERSION = "openalex-academic-whole-part-plan.v1"
MULTIPART_SCHEMA = "aw.openalex-academic-multipart-tranche-plan.v1"
MULTIPART_VERSION = "openalex-academic-ordered-whole-parts-plan.v1"
PARTITIONED_VERSION = "openalex-academic-ordered-whole-parts-plan.v2-partitioned"


# AGE-SCALED CITATION CAP: the L2 ladder (decisions.md ADR-102, ratified PR #82, 2026-09-02, Tim
# 2026-09-01 "L2, reprice to keep"). It replaces task 231 section 3's flat `cited >= 1 OR is_core`
# rule. Age is publication_year, not the snapshot year. `is_core` admits at every age band.
#   before 2010 : is_core OR cited >= 10      2010-2014 : is_core OR cited >= 5
#   2015-2019   : is_core OR cited >= 3       2020 on   : is_core OR cited >= 1 (or the carve-out)
CAP_SQL = ("(coalesce(primary_location.source.is_core, false)"
           " OR (publication_year < 2010 AND cited_by_count >= 10)"
           " OR (publication_year >= 2010 AND publication_year <= 2014 AND cited_by_count >= 5)"
           " OR (publication_year >= 2015 AND publication_year <= 2019 AND cited_by_count >= 3)"
           " OR (publication_year >= 2020 AND cited_by_count >= 1))")
# Mission-geography carve-out (ADR-032): recent (>= 2020) works with an author in a target country
# stay even when uncited and non-core. ADR-102 leaves it unchanged.
CAP_TC_SQL = ("(coalesce(len(list_filter(\n"
              "            list_distinct(flatten(list_transform(authorships, a -> a.countries))),\n"
              "            y -> y IN ('{countries}'))), 0) > 0 AND publication_year >= 2020)")


# ----------------------------------------------------------------------------- the part plan
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


def _artifact_ref(path: Path, artifact_root: Path) -> dict:
    path = path.resolve()
    relative = path.relative_to(artifact_root.resolve())
    return {"uri": f"artifact://{relative.as_posix()}",
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def choose_part(rows: list[dict], *, minimum_records: int,
                excluded_paths: set[str]) -> dict:
    if minimum_records <= 0:
        raise ValueError("OpenAlex tranche minimum must be positive")
    seen = set(); candidates = []
    for row in rows:
        relative = str(row.get("relative_path") or "")
        retained = row.get("retained_legacy_records")
        if (not relative.endswith(".parquet") or relative in seen
                or not isinstance(retained, int) or retained <= 0):
            raise ValueError("invalid or duplicate OpenAlex frontier row")
        seen.add(relative)
        if relative not in excluded_paths and retained >= minimum_records:
            candidates.append(row)
    if not candidates:
        raise ValueError("no disjoint whole OpenAlex part meets minimum")
    return min(candidates, key=lambda row: (
        row["retained_legacy_records"], row["relative_path"]))


def choose_parts(rows: list[dict], *, minimum_records: int,
                 excluded_paths: set[str]) -> list[dict]:
    """Choose the fewest largest whole parts in a stable global order.

    No part is split and no selected-record count is projected.  Ordering by
    decreasing measured retained rows makes the minimum number of source
    parts explicit; the relative path is the deterministic tie breaker.
    """
    if minimum_records <= 0:
        raise ValueError("OpenAlex tranche minimum must be positive")
    seen: set[str] = set()
    candidates = []
    for row in rows:
        relative = str(row.get("relative_path") or "")
        retained = row.get("retained_legacy_records")
        if (not relative.endswith(".parquet") or relative in seen
                or not isinstance(retained, int) or retained <= 0):
            raise ValueError("invalid or duplicate OpenAlex frontier row")
        seen.add(relative)
        if relative not in excluded_paths:
            candidates.append(row)
    ordered = sorted(candidates, key=lambda row: (
        -row["retained_legacy_records"], row["relative_path"]))
    selected = []
    measured = 0
    for row in ordered:
        selected.append(row)
        measured += row["retained_legacy_records"]
        if measured >= minimum_records:
            return selected
    raise ValueError("no disjoint whole OpenAlex parts meet minimum")


def partition_parts(rows: list[dict], *, partition_count: int) -> list[dict]:
    """Split the selected ordered whole parts into stable contiguous slices.

    Contiguous slices mean concatenating partition outputs by partition index
    is byte-for-row equivalent to one worker replaying the plan order.  Parts
    are never split, omitted, repeated, or reassigned by runtime timing.
    """
    if (partition_count <= 0 or not rows or partition_count > len(rows)):
        raise ValueError("invalid OpenAlex whole-part partition count")
    base, remainder = divmod(len(rows), partition_count)
    global_records = sum(
        int(row["retained_legacy_records"]) for row in rows)
    result = []
    cursor = 0
    for index in range(partition_count):
        size = base + (1 if index < remainder else 0)
        selected = rows[cursor:cursor + size]
        paths = [str(row["relative_path"]) for row in selected]
        records = sum(int(row["retained_legacy_records"]) for row in selected)
        logical = hashlib.sha256("".join(
            f"{path}\t{row['retained_legacy_records']}\n"
            for path, row in zip(paths, selected)).encode()).hexdigest()
        result.append({
            "partition_index": index,
            "partition_count": partition_count,
            "selected_part_start": cursor,
            "selected_part_end_exclusive": cursor + size,
            "selected_part_count": size,
            "selected_records": records,
            "global_selected_records": global_records,
            "relative_paths": paths,
            "selection_logical_sha256": logical,
        })
        cursor += size
    if (cursor != len(rows)
            or [path for part in result for path in part["relative_paths"]]
            != [str(row["relative_path"]) for row in rows]
            or sum(part["selected_records"] for part in result)
            != sum(int(row["retained_legacy_records"]) for row in rows)):
        raise ValueError("OpenAlex whole-part partition conservation failed")
    return result


def plan_tranche(*, source_root: Path, source_run: Path, output: Path,
         artifact_root: Path, expected_source_run_sha256: str,
         minimum_records: int, excluded_paths: set[str],
         multi_part: bool = False, partition_count: int = 1) -> dict:
    if output.exists():
        raise ValueError("OpenAlex tranche plan output must be fresh")
    if sha256_file(source_run) != expected_source_run_sha256:
        raise ValueError("OpenAlex portable source RUN identity drift")
    run = json.loads(source_run.read_text())
    counts = run.get("counts") or {}
    if (run.get("schema") != "aw.openalex-snapshot-acquisition-run.v1"
            or run.get("status") != "complete"
            or counts.get("selected_parts") != 2216):
        raise ValueError("OpenAlex portable source RUN is not terminal")
    frontier = source_root / "frontier.jsonl"
    rows = [json.loads(line) for line in frontier.read_text().splitlines()
            if line.strip()]
    if len(rows) != counts["selected_parts"]:
        raise ValueError("OpenAlex frontier part conservation failed")
    selected_rows = (choose_parts(rows, minimum_records=minimum_records,
                                  excluded_paths=excluded_paths)
                     if multi_part else
                     [choose_part(rows, minimum_records=minimum_records,
                                  excluded_paths=excluded_paths)])
    selected_parts = []
    for selected in selected_rows:
        relative = selected["relative_path"]
        key = hashlib.sha256(relative.encode()).hexdigest()
        receipt_path = source_root / "receipts" / key[:2] / f"{key}.json"
        receipt = json.loads(receipt_path.read_text())
        body = source_root / "original" / "parquet" / relative
        body_ref = receipt.get("body") or {}
        if (receipt.get("schema") != "aw.openalex-parquet-part-receipt.v1"
                or receipt.get("status") != "complete"
                or receipt.get("relative_path") != relative
                or receipt.get("retained_legacy_records")
                != selected["retained_legacy_records"]
                or body_ref.get("bytes") != selected["expected_bytes"]
                or not body.is_file()
                or body.stat().st_size != body_ref.get("bytes")
                or sha256_file(body) != body_ref.get("sha256")):
            raise ValueError("OpenAlex planned part receipt/body mismatch")
        selected_parts.append({
            "relative_path": relative,
            "retained_legacy_records": selected["retained_legacy_records"],
            "upstream_records": selected["upstream_records"],
            "expected_bytes": selected["expected_bytes"],
            "receipt": _artifact_ref(receipt_path, artifact_root),
            "body": _artifact_ref(body, artifact_root),
        })
    selected_records = sum(
        row["retained_legacy_records"] for row in selected_parts)
    result = {
        "schema": MULTIPART_SCHEMA if multi_part else SCHEMA,
        "version": MULTIPART_VERSION if multi_part else VERSION,
        "status": "ready_for_bounded_materialization",
        "source_run": _artifact_ref(source_run, artifact_root),
        "source_run_sha256": expected_source_run_sha256,
        "frontier": _artifact_ref(frontier, artifact_root),
        "target_minimum_records": minimum_records,
        "excluded_parts": sorted(excluded_paths),
        "selected_part_count": len(selected_parts),
        "selected_records": selected_records,
        "scope": ("ordered disjoint whole preserved parts; source-only selection evidence with no document, chunk, vector, cost, or yield claim"
                  if multi_part else
                  "one disjoint whole preserved part; source-only selection evidence with no document, chunk, vector, cost, or yield claim"),
    }
    if multi_part:
        result["selected_parts"] = selected_parts
        result["selection_order"] = (
            "retained_legacy_records_desc_then_relative_path_asc_until_minimum")
        if partition_count > 1:
            result["version"] = PARTITIONED_VERSION
            result["partition_count"] = partition_count
            result["partitions"] = partition_parts(
                selected_parts, partition_count=partition_count)
    else:
        if partition_count != 1:
            raise ValueError("partitioned OpenAlex planning requires --multi-part")
        result["selected_part"] = selected_parts[0]
    atomic_publish(output, (json.dumps(result, indent=2, sort_keys=True)
                            + "\n").encode())
    return result


def planned_relative_paths(plan: dict, *, expected_source_run_sha256: str,
                           target_records: int) -> tuple[list[str], int]:
    schema = plan.get("schema")
    if schema == PLAN_SCHEMAS[0]:
        selected = [plan.get("selected_part") or {}]
        selected_records = selected[0].get("retained_legacy_records", 0)
    elif schema == PLAN_SCHEMAS[1]:
        selected = plan.get("selected_parts") or []
        selected_records = plan.get("selected_records", 0)
    else:
        selected = []
        selected_records = 0
    if (schema not in PLAN_SCHEMAS
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
    """Return one exact contiguous plan slice or the whole selection."""
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


# ----------------------------------------------------------------------------- the works
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


def selection_sql() -> str:
    fields = "','".join(TARGET_FIELDS)
    countries = "','".join(TARGET_COUNTRIES)
    cap_tc = CAP_TC_SQL.format(countries=countries)
    return f"""
    SELECT to_json(t) AS source_json
    FROM read_parquet(?) AS t
    WHERE (primary_topic.field.display_name IN ('{fields}')
           OR primary_topic.subfield.display_name = 'Law')
      AND abstract_inverted_index IS NOT NULL
      AND NOT is_retracted AND NOT is_paratext
      AND publication_year >= 2000
      AND (language IN ('en', 'de', 'fr', 'it') OR language IS NULL)
      AND (doi IS NOT NULL OR primary_location.landing_page_url IS NOT NULL
           OR best_oa_location.landing_page_url IS NOT NULL)
      AND (
        {CAP_SQL}
        OR {cap_tc}
      )
    ORDER BY id
    """


def keeps(raw: dict) -> bool:
    """The Python half of the selection: a work with a rebuilt abstract of 40 characters or more
    and an http(s) address (DOI, landing page of the primary or the open-access location)."""
    abstract = reconstruct_abstract(raw.get("abstract_inverted_index"))
    url = (raw.get("doi")
           or (raw.get("primary_location") or {}).get("landing_page_url")
           or (raw.get("best_oa_location") or {}).get("landing_page_url"))
    return len(abstract) >= 40 and isinstance(url, str) and url.startswith("http")


def iter_selected(parquet_path) -> Iterator[dict]:
    """The works of one part that the pinned policy selects, in ascending OpenAlex id order,
    streamed: DuckDB (lazy import) filters and sorts, Python gets `BATCH` rows at a time."""
    import duckdb
    connection = duckdb.connect()
    try:
        # `to_json` writes a TIMESTAMPTZ (`created_date`) in the session's zone, which DuckDB takes
        # from the machine: the same part gave other bytes on a host with another TZ. UTC, always.
        connection.execute("SET TimeZone='UTC'")
        cursor = connection.execute(selection_sql(), [str(parquet_path)])
        while True:
            batch = cursor.fetchmany(BATCH)
            if not batch:
                return
            for (source_json,) in batch:
                raw = json.loads(source_json)
                if keeps(raw):
                    yield raw
    finally:
        connection.close()


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


def author_names(raw: dict) -> list[str]:
    """The display names of a work's authors, in the work's own order (names of natural persons:
    master-only in a row, never in the dump)."""
    return _names(raw.get("authorships"), "author", "display_name")


def render_record(raw: dict, *, relative_path: str, parent_sha256: str,
                  with_authors: bool = True) -> dict:
    """The Markdown row of one work: the old script's `render_record` row, byte for byte, when
    `with_authors` is True. False leaves out the "Authors" section and nothing else."""
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
    authors = author_names(raw) if with_authors else []
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
    return {
        "document_id": document_id, "url": source_id,
        "lang": raw.get("language") or "en",
        "title": title, "markdown": markdown,
        "source_profile": SOURCE_PROFILE,
        "source_record_id": source_id,
        "parent_archive": {"relative_path": relative_path,
                           "sha256": parent_sha256},
        "source_metadata": {
            "openalex_id": source_id, "doi": raw.get("doi"),
            "publication_date": raw.get("publication_date"),
            "publication_year": raw.get("publication_year"),
            "work_type": raw.get("type"),
            "metadata_license": CC0,
            "underlying_work_license": primary.get("license"),
            "underlying_work_license_id": primary.get("license_id"),
            "underlying_work_version": primary.get("version"),
            "record_classification":
                "full_openalex_metadata_and_abstract_not_full_paper_content",
        },
        "full_prechunk_content": True,
    }


def venue_of(raw: dict) -> str | None:
    """The work's venue (the source of its primary location): the master-only attribution."""
    source = (raw.get("primary_location") or {}).get("source") or {}
    name = source.get("display_name")
    return name.strip() if isinstance(name, str) and name.strip() else None


# ----------------------------------------------------------------------------- the planner door
def plan_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Plan a disjoint whole-part OpenAlex tranche without a yield claim.")
    for name in ("source-root", "source-run", "output", "artifact-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-source-run-sha256", required=True)
    parser.add_argument("--minimum-records", type=int, required=True)
    parser.add_argument("--exclude-part", action="append", default=[])
    parser.add_argument("--multi-part", action="store_true")
    parser.add_argument("--partition-count", type=int, default=1)
    args = parser.parse_args(argv)
    print(json.dumps(plan_tranche(
        source_root=args.source_root.resolve(),
        source_run=args.source_run.resolve(), output=args.output.resolve(),
        artifact_root=args.artifact_root.resolve(),
        expected_source_run_sha256=args.expected_source_run_sha256,
        minimum_records=args.minimum_records,
        excluded_paths=set(args.exclude_part), multi_part=args.multi_part,
        partition_count=args.partition_count),
        indent=2, sort_keys=True))



if __name__ == "__main__":
    plan_main()
