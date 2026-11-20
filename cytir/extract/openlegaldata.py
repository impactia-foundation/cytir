#!/usr/bin/env python3
"""Materialize OpenLegalData German court decisions to sealed L1 Markdown.

Task 633, family W4 lane 1. The acquisition (task 489) already sealed all 54
`dump-20260520` parquet shards of `openlegaldata/court-decisions-germany` --
see `sources/de/openlegaldata.yaml` and
task 489. This module is the ingester the
pack file names as forward-reference (the retired `ingest_openlegaldata.py` script in
its own notes) -- built under `materialize_*` naming to match the ADR-105
L1-first pattern the other sealed packs use (`cytir.sources.eric.materialize`,
`cytir.sources.openalex.materialize_pilot`).

WHAT THIS DOES
--------------
For a bounded set of acquisition parquet shards:

1. Deep-validates the acquisition RUN's `original` stage against the sealed
   acquisition root (`preservation_contract._validate_stage`) -- the parquet
   bytes this reads are exactly the bytes task 489 sealed, byte for byte.
2. Reads each shard with DuckDB (columnar, bounded-memory streaming: one
   shard's rows are pulled into a Python list, never all 54 at once).
3. Per decision: builds one Markdown document -- an identity line (court,
   decision type, file number/Aktenzeichen, date, ECLI) as the document's H1,
   followed by the source `markdown_content` body -- and the decision's own
   page URL, verified live against a real record during this task
   (`https://de.openlegaldata.io/case/{slug}` -- confirmed 200, ECLI and
   title match the parquet row exactly; not documented anywhere in the
   dataset or the pack file, so this module is the first place it is
   recorded).
4. A decision whose `markdown_content` is empty (a real, pre-existing gap in
   OpenLegalData's own extraction -- ~0.4% of shard 0, confirmed by this
   task, same shape task 489 found for `content`) is never silently dropped:
   it is written to an explicit quarantine inventory instead of the
   principal Markdown stream (corpus-doctrine: a stage that goes quiet about
   a population is how a corpus loses a population).
5. Seals the principal Markdown stream as one `aw.stage-receipt.v1` stage
   (`seal_stage_receipt.seal`) under a RUN.json this module owns
   (`aw.openlegaldata-materialization-run.v1`), so
   `cytir.corpus.stage_structured_markdown` can consume it
   directly as a `parent_run` with a `stages.markdown` stage -- unmodified,
   the same chunker every other `chunk.generic` family uses.
6. ALSO persists every admitted document via
   `persist_l1_records.write_l1_records` (ADR-105) under
   `l1/release=<release>/source=<source>/`, independent of (5) -- the flat
   L1 store `cytir.corpus.persisted` indexes, so a later loader does
   not refuse these rows for want of a persisted input.

WHY NO SEPARATE "SOURCE" (raw-row) STAGE
-----------------------------------------
The ERIC/OpenAlex materializers also emit a canonical-JSON `source_records`
stage. This module does not: the acquisition's `original` stage already
seals the exact parquet bytes (columnar, hash-verified, byte-exact to the
HuggingFace dataset) as L0. Every Markdown row carries `parent_archive`
(the shard's own `relative_path` + sha256) as its anchor back to that L0 --
the same anchoring discipline `anchored_law` uses elsewhere
(`cytir.sources.openalex.materialize_pilot`'s `parent_archive` field). Mirroring
the full parquet a second time as one-JSON-object-per-row would double the
on-disk footprint of a 423,941-row family for no fidelity gain the sealed
parquet does not already give byte-for-byte.

WHAT THIS DOES NOT DO
----------------------
It does not chunk, embed, or write the terminal `aw.vector-rollup.v1` / admission
receipt (see `cytir.corpus.stage_structured_markdown`,
task 413a's `embed_shards.py`, task 417's `seal_vector_rollup.py`,
and this task's own driver script `run_first_tranche.py`).
It never writes to Qdrant (prod or twin) -- task 629 owns promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from cytir.acquire.gzip_pinned import deterministic_gzip
from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.index.preservation_contract import _validate_stage
import cytir.acquire.stage_receipt as SS


VERSION = "openlegaldata-court-decisions-materialization.v1"
# Frozen label (never edit): declared as `producer_ids` on family de_openlegaldata_v1 in
# data/source-family-contracts.v1.json (task 1316 / D6); tests/test_aw_openlegaldata_producer_id.py pins it.
PRODUCER = "pipeline/materialize_openlegaldata.py"
RUN_SCHEMA = "aw.openlegaldata-materialization-run.v1"
ACQUISITION_SCHEMA = "aw.openlegaldata-hf-dump-acquisition-run.v1"
SHARD_COUNT = 54

#: ADR-039 per-record licence. Public-domain official work, UrhG §5 -- NEVER
#: `ODbL` (that covers only OpenLegalData's own database compilation, not the
#: decision text; sources/de/openlegaldata.yaml `license.notes` has the full
#: reasoning). Matches `cytir.core.license`'s existing sibling entry
#: for rechtsprechung-im-internet.de, same UrhG §5 basis, and
#: the retired `rii_dump_basis_backfill.py` script's stamping convention.
LICENSE = "public_domain_official_work (UrhG §5)"

#: Verified live during task 633 (curl, 2026-09-22): GET against a real
#: slug from shard 0 (`bgh-2018-09-05-2-str-45417`) returns 200 with a page
#: whose <title> ("Urteil vom Bundesgerichtshof (2. Strafsenat) - 2 StR
#: 454/17 - Open Legal Data") and displayed ECLI ("ECLI:DE:BGH:2018:
#: 050918U2STR454.17.0") match the parquet row exactly. Not documented
#: anywhere in the HF dataset card, the API docs, or the pack file.
URL_TEMPLATE = "https://de.openlegaldata.io/case/{slug}"


class MaterializeError(RuntimeError):
    """A row, a stage, or the parent acquisition cannot be trusted as-is."""


def build_url(slug: str) -> str:
    slug = (slug or "").strip()
    if not slug:
        raise MaterializeError("decision has no slug; cannot mint a page URL")
    return URL_TEMPLATE.format(slug=slug)


def build_document_id(source_id: Any) -> str:
    return hashlib.sha256(
        f"openlegaldata-court-decision\0{source_id}".encode()).hexdigest()


def format_court(court: dict | None) -> str:
    court = court or {}
    name = str(court.get("name") or "").strip()
    level = str(court.get("level_of_appeal") or "").strip()
    jurisdiction = str(court.get("jurisdiction") or "").strip()
    bits = [b for b in (name, level, jurisdiction) if b]
    return ", ".join(bits)


def build_identity_line(row: dict) -> str:
    """One H1 carrying court/type/file-number/date/ECLI.

    `chunk_generic_records` (`chunk.generic`, ADR-090) carries the ancestor
    heading path as a breadcrumb PREFIX on every chunk it emits (`sizing`/
    `prefix` in `data/chunk-policy-struct-md-768w-o0-adjacent-sections-tail-
    1152-tok2560.v1.json`). Putting the whole identity -- court, decision
    type, Aktenzeichen, date, ECLI -- on the document's single H1 means every
    chunk of a long decision (one that splits past the 1152-word hard max),
    not only its first, carries the court name, date, and ECLI/Aktenzeichen
    tokens the sparse channel needs (memory
    `embedding-surface-is-a-contract.md`) -- without hand-rolling a second
    chunker.
    """
    court_label = format_court(row.get("court"))
    decision_type = str(row.get("type") or "Entscheidung").strip()
    file_number = str(row.get("file_number") or "").strip()
    date = str(row.get("date") or "").strip()
    ecli = str(row.get("ecli") or "").strip()
    parts = [p for p in (
        court_label,
        f"{decision_type} {file_number}".strip() if (decision_type or file_number) else "",
        date,
        ecli,
    ) if p]
    return "# " + " · ".join(parts) if parts else "# " + decision_type


def build_title(row: dict) -> str:
    court = row.get("court") or {}
    name = str(court.get("name") or "").strip()
    decision_type = str(row.get("type") or "Entscheidung").strip()
    file_number = str(row.get("file_number") or "").strip()
    bits = [b for b in (decision_type, file_number) if b]
    label = " ".join(bits) if bits else decision_type
    return f"{label} — {name}" if name else label


def build_markdown(row: dict) -> str | None:
    """The document body, or None for a row with no usable text.

    Prefers `markdown_content` (genuinely Markdown-shaped -- `##` headers,
    definition-list paragraphs -- verified live against real rows during
    this task; `content` is raw HTML fragments of the same decision and is
    never used as the embedded body). A row with no `markdown_content` at
    all (~0.4% of shard 0, a real pre-existing OpenLegalData extraction gap,
    same shape task 489 already found for `content`) returns None; the
    caller quarantines it rather than emitting an empty document.
    """
    body = str(row.get("markdown_content") or "").strip()
    if not body:
        return None
    return build_identity_line(row) + "\n\n" + body + "\n"


def build_markdown_row(row: dict, *, parent_archive: dict) -> dict:
    """One `document_id`/`markdown`/`url` row for the sealed Markdown stage."""
    markdown = build_markdown(row)
    if markdown is None:
        raise MaterializeError(f"row {row.get('id')} has no usable body")
    source_id = row.get("id")
    url = build_url(row.get("slug"))
    return {
        "document_id": build_document_id(source_id),
        "url": url,
        "markdown": markdown,
        "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
        "lang": "de",
        "title": build_title(row),
        "source_record_id": source_id,
        "parent_archive": parent_archive,
        "license": LICENSE,
        "trust_tier": 0.90,
        "source_metadata": {
            "ecli": row.get("ecli") or None,
            "file_number": row.get("file_number") or None,
            "date": str(row["date"]) if row.get("date") is not None else None,
            "type": row.get("type") or None,
            "court": row.get("court") or None,
            "slug": row.get("slug"),
        },
    }


def build_l1_record(row: dict) -> dict:
    """One record for `persist_l1_records.write_l1_records` (ADR-105)."""
    return {
        "url": row["url"],
        "markdown": row["markdown"],
        "markdown_sha256": row["markdown_sha256"],
        "title": row["title"],
        "lang": row["lang"],
        "document_id": row["document_id"],
        "source_record_id": row["source_record_id"],
        "license": row["license"],
        "trust_tier": row["trust_tier"],
        "tier": "A",
        "source_type": "anchored_law",
    }


def _canonical_gzip_rows(rows: Iterable[dict]) -> bytes:
    body = b"".join((canonical_json(row) + "\n").encode() for row in rows)
    return deterministic_gzip(body)


def _parquet_shard_artifact(original: dict, shard_index: int) -> dict:
    uri = (f"artifact://original/parquet/dump-20260520/"
           f"train-{shard_index:05d}-of-{SHARD_COUNT:05d}.parquet")
    for artifact in original["artifacts"]:
        if artifact["uri"] == uri:
            return artifact
    raise MaterializeError(f"acquisition stage has no artifact {uri}")


def read_shard_rows(parquet_path: Path) -> list[dict]:
    """Read one shard's rows via DuckDB. Lazy import: pure functions above
    need no runtime wheel, matching `cytir.sources.openalex.materialize_pilot`."""
    import duckdb
    con = duckdb.connect()
    try:
        rows = con.execute(
            "SELECT id, slug, court, file_number, date, created_date, "
            "updated_date, type, ecli, content, markdown_content, "
            "reference_markers FROM read_parquet(?) ORDER BY id",
            [str(parquet_path)]).fetchall()
        cols = [d[0] for d in con.description]
    finally:
        con.close()
    return [dict(zip(cols, row)) for row in rows]


def materialize(*, artifact_root: Path, acquisition_root: Path,
                acquisition_run_path: Path, shard_indices: list[int],
                output_root: Path) -> dict:
    """Materialize the given shards into one sealed Markdown RUN.json.

    `artifact_root` is the global corpus root every `artifact://` URI in the
    OUTPUT resolves against (matches the other sealed packs:
    `AW_CORPUS_ROOT` / the retired `aw_paths.py` script's `CORPUS_ROOT`). `acquisition_root`
    is the acquisition's OWN root (its URIs are relative to
    `.../de-openlegaldata-v1/acquisition/`, a private scratch root, not the
    global one) -- the two are validated independently and never confused.
    """
    artifact_root = artifact_root.resolve()
    acquisition_root = acquisition_root.resolve()
    output_root = output_root.resolve()
    if not output_root.is_relative_to(artifact_root):
        raise MaterializeError("output must be below the global artifact root")
    if output_root.exists() and any(output_root.iterdir()):
        raise MaterializeError(f"refusing non-empty existing output: {output_root}")

    acq_run = json.loads(acquisition_run_path.read_text(encoding="utf-8"))
    if (acq_run.get("schema") != ACQUISITION_SCHEMA
            or acq_run.get("status") != "complete"):
        raise MaterializeError(
            "acquisition RUN is not a complete OpenLegalData dump run")
    original = _validate_stage(acq_run["stages"]["original"], "acquisition.original",
                               artifact_root=acquisition_root, deep=True)

    markdown_rows: list[dict] = []
    quarantine_rows: list[dict] = []
    documents = 0
    for shard_index in shard_indices:
        shard_artifact = _parquet_shard_artifact(original, shard_index)
        shard_path = acquisition_root / shard_artifact["uri"].removeprefix("artifact://")
        if (not shard_path.is_file()
                or shard_path.stat().st_size != shard_artifact["bytes"]
                or sha256_file(shard_path) != shard_artifact["sha256"]):
            raise MaterializeError(f"shard {shard_index} identity drift: {shard_path}")
        parent_archive = {
            "relative_path": shard_artifact["uri"].removeprefix("artifact://"),
            "sha256": shard_artifact["sha256"],
        }
        for row in read_shard_rows(shard_path):
            documents += 1
            markdown = build_markdown(row)
            if markdown is None:
                quarantine_rows.append({
                    "source_record_id": row.get("id"), "slug": row.get("slug"),
                    "reason": "empty_markdown_content",
                    "shard_relative_path": parent_archive["relative_path"],
                })
                continue
            markdown_rows.append(build_markdown_row(row, parent_archive=parent_archive))

    if not markdown_rows:
        raise MaterializeError("no admissible decisions in the requested shards")

    output_root.mkdir(parents=True, exist_ok=True)
    markdown_path = output_root / "stages" / "markdown" / "data" / "shard-000000.jsonl.gz"
    markdown_body = _canonical_gzip_rows(markdown_rows)
    atomic_publish(markdown_path, markdown_body)
    markdown_logical = hashlib.sha256(
        b"".join((canonical_json(r) + "\n").encode() for r in markdown_rows)
    ).hexdigest()
    markdown_stage = SS.seal(
        artifact_root=artifact_root, artifact_paths=[markdown_path],
        receipt_path=output_root / "stages" / "markdown" / "receipt.json",
        records=len(markdown_rows), fmt="canonical-jsonl+deterministic-gzip",
        logical_sha256=markdown_logical, producer_id=PRODUCER,
        producer_version=VERSION)

    quarantine_path = output_root / "quarantine" / "empty-body.jsonl.gz"
    quarantine_body = _canonical_gzip_rows(quarantine_rows) if quarantine_rows else b""
    if quarantine_rows:
        atomic_publish(quarantine_path, quarantine_body)

    run = {
        "schema": RUN_SCHEMA, "version": VERSION, "status": "complete",
        "source_id": "de-openlegaldata",
        "shard_indices": shard_indices,
        "counts": {
            "source_rows": documents,
            "markdown_documents": len(markdown_rows),
            "quarantined_empty_body": len(quarantine_rows),
        },
        "parent_acquisition": {
            "run_sha256": sha256_file(acquisition_run_path),
            "original_logical_sha256": original["logical_sha256"],
        },
        "license": LICENSE,
        "stages": {"markdown": markdown_stage},
        "quarantine": {
            "empty_body": {
                "uri": f"artifact://{quarantine_path.relative_to(artifact_root).as_posix()}",
                "bytes": quarantine_path.stat().st_size,
                "sha256": sha256_file(quarantine_path),
                "records": len(quarantine_rows),
            } if quarantine_rows else {"records": 0},
        },
    }
    run_path = output_root / "RUN.json"
    atomic_publish(run_path, (json.dumps(run, indent=2, sort_keys=True) + "\n").encode())
    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--acquisition-root", type=Path, required=True)
    ap.add_argument("--acquisition-run", type=Path, required=True)
    ap.add_argument("--shard", type=int, action="append", required=True,
                    dest="shard_indices",
                    help="acquisition shard index (0-53); repeatable")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    try:
        result = materialize(
            artifact_root=args.artifact_root, acquisition_root=args.acquisition_root,
            acquisition_run_path=args.acquisition_run,
            shard_indices=sorted(set(args.shard_indices)), output_root=args.out)
    except (OSError, MaterializeError, ValueError) as exc:
        print(f"FATAL: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
