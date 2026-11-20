#!/usr/bin/env python3
"""Audit disjoint FINMA partitions against the source-specific web_pdf contract."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.sources.finma import materialize_pilot as M
from cytir.index.preservation_contract import _validate_stage


VERSION = "finma-web-pdf-partition-audit.v1"


def _rows(artifact_root: Path, stage: dict, name: str) -> list[dict]:
    receipt = _validate_stage(stage, name, artifact_root=artifact_root, deep=True)
    rows: list[dict] = []
    for artifact in receipt.get("artifacts") or []:
        if not str(artifact.get("uri") or "").endswith(".jsonl.gz"):
            continue
        path = artifact_root / artifact["uri"].removeprefix("artifact://")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def _artifact(path: Path, artifact_root: Path) -> dict:
    path = path.resolve()
    return {"uri": "artifact://" + path.relative_to(
        artifact_root.resolve()).as_posix(), "bytes": path.stat().st_size,
        "sha256": sha256_file(path)}


def audit(*, artifact_root: Path, parent_runs: list[Path], output: Path,
          expected_start: int, expected_count: int) -> dict:
    artifact_root = artifact_root.resolve()
    output = output.resolve()
    output.relative_to(artifact_root)
    if output.exists():
        raise ValueError(f"immutable FINMA audit exists: {output}")
    if not parent_runs:
        raise ValueError("at least one FINMA partition is required")

    partitions = []
    l0_binding = None
    retained_population = None
    for path in parent_runs:
        path = path.resolve()
        run = json.loads(path.read_text())
        repaired = (
            run.get("schema")
            == "aw.finma-pdf-contextual-outlinks-supersession.v1"
            and run.get("status") == "ready_for_web_pdf_profile_audit"
            and run.get("content_reused_without_reextraction") is True)
        direct = (
            run.get("schema") == "aw.finma-pdf-markdown-pilot-run.v1"
            and run.get("status")
            == "bounded-pilot-complete-not-production-admission"
            and run.get("version")
            == "finma-pdf-markdown-pilot.v6-contextual-outlinks")
        if not (repaired or direct):
            raise ValueError(
                "FINMA partition is neither a contextual supersession nor "
                "a corrected direct producer")
        for name in ("original", "markdown", "outlinks"):
            _validate_stage(run["stages"][name], f"finma.{name}",
                            artifact_root=artifact_root, deep=True)
        if l0_binding is None:
            l0_binding = run["l0_receipt"]
            retained_population = run["retained_pdf_population"]
        elif (run["l0_receipt"] != l0_binding
              or run["retained_pdf_population"] != retained_population):
            raise ValueError("FINMA partitions do not share one sealed L0 population")
        partitions.append((int(run["start"]), int(run["limit"]), path, run))
    partitions.sort()
    expected_ranges = []
    cursor = expected_start
    for start, limit, _, _ in partitions:
        if start != cursor or limit <= 0:
            raise ValueError("FINMA partitions overlap or leave a source-order gap")
        expected_ranges.append({"start": start, "limit": limit})
        cursor += limit
    if cursor != expected_start + expected_count:
        raise ValueError("FINMA partition union does not conserve expected range")

    l0_path = artifact_root / l0_binding["uri"].removeprefix("artifact://")
    if (not l0_path.is_file()
            or sha256_file(l0_path) != l0_binding.get("sha256")):
        raise ValueError("FINMA L0 receipt binding mismatch")
    # Partition RUNs predate the exhaustive artifact-object contract and bind
    # the same immutable L0 receipt by URI+SHA only.  The aggregate audit is a
    # new publication boundary, so normalize that verified predecessor into a
    # complete global URI+bytes+SHA binding for recursive downstream sealers.
    l0_binding = _artifact(l0_path, artifact_root)
    l0_rows = list(M._iter_l0(l0_path, artifact_root))
    api_rows = [row for row in l0_rows if row[2].get("kind") == "api"]
    metadata = M._metadata(api_rows)
    pdf_rows = {}
    for uri, line_number, row in l0_rows:
        if row.get("kind") != "pdf":
            continue
        identity = M._source_record_id(row)
        if identity in pdf_rows:
            raise ValueError(f"duplicate canonical FINMA identity in L0: {identity}")
        pdf_rows[identity] = (uri, line_number, row)

    all_ids: set[str] = set()
    all_ordinals: set[int] = set()
    body_digests: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    metadata_counts: Counter[str] = Counter()
    total_pages = total_links = linked = empty = 0
    partition_evidence = []
    for start, limit, path, run in partitions:
        parents = _rows(artifact_root, run["stages"]["markdown"],
                        f"finma.{start}.markdown")
        outlinks = _rows(artifact_root, run["stages"]["outlinks"],
                         f"finma.{start}.outlinks")
        if len(parents) != limit or len(outlinks) != limit:
            raise ValueError("FINMA partition parent/outlink count mismatch")
        out_by_id = {str(row.get("source_record_id") or ""): row
                     for row in outlinks}
        if len(out_by_id) != len(outlinks):
            raise ValueError("FINMA outlink parent identities are not unique")
        local_ordinals = set()
        for parent in parents:
            identity = str(parent.get("source_record_id") or "")
            ordinal = parent.get("ordinal")
            markdown = str(parent.get("markdown") or "")
            extraction = parent.get("extraction") or {}
            pages = extraction.get("pages") or []
            expected_meta = metadata.get(identity, {})
            if (not identity.startswith("https://www.finma.ch/")
                    or identity in all_ids or identity not in pdf_rows
                    or parent.get("canonical_url") != identity
                    or M._source_record_id({"url": parent.get("url")}) != identity
                    or not isinstance(ordinal, int)
                    or ordinal < start or ordinal >= start + limit
                    or ordinal in local_ordinals or ordinal in all_ordinals
                    or not markdown.strip()
                    or parent.get("markdown_sha256")
                    != hashlib.sha256(markdown.encode()).hexdigest()):
                raise ValueError(f"FINMA canonical parent identity drift: {identity}")
            if identity == str((parent.get("parent_pdf") or {}).get("sha256") or ""):
                raise ValueError("FINMA body digest was mixed into source identity")
            l0_uri, l0_line, source = pdf_rows[identity]
            pdf_parent = parent.get("parent_pdf") or {}
            if (pdf_parent.get("sha256") != source.get("sha256")
                    or pdf_parent.get("bytes") != source.get("bytes")
                    or pdf_parent.get("l0_uri") != l0_uri
                    or pdf_parent.get("l0_line") != l0_line):
                raise ValueError(f"FINMA PDF lineage mismatch: {identity}")
            expected_fields = {
                "title": expected_meta.get("title", ""),
                "description": expected_meta.get("description", ""),
                "release_date": expected_meta.get("release_date", ""),
                "language": expected_meta.get("language", "und"),
                "api_parent": expected_meta.get("api_parent"),
            }
            if any(parent.get(key) != value for key, value in expected_fields.items()):
                raise ValueError(f"FINMA exact API metadata drift: {identity}")
            for key in ("title", "description", "release_date"):
                metadata_counts[f"nonempty_{key}"] += bool(parent.get(key))
            metadata_counts["language_known"] += parent.get("language") != "und"
            metadata_counts["api_bound"] += parent.get("api_parent") is not None
            if (extraction.get("adapter") != "docling"
                    or extraction.get("ocr") is not False
                    or extraction.get("table_structure") is not True
                    or extraction.get("accelerator_device") != "cpu"
                    or extraction.get("errors") != []
                    or not pages):
                raise ValueError(f"FINMA Docling/page diagnostics failed: {identity}")
            numbers = [page.get("page_number") for page in pages]
            if (numbers != list(range(1, len(pages) + 1))
                    or any(not isinstance(page.get("width"), (int, float))
                           or not isinstance(page.get("height"), (int, float))
                           or page["width"] <= 0 or page["height"] <= 0
                           for page in pages)):
                raise ValueError(f"FINMA page geometry diagnostics failed: {identity}")
            link_row = out_by_id.get(identity)
            if (not link_row or link_row.get("parent_url") != identity
                    or link_row.get("parent_markdown_sha256")
                    != parent["markdown_sha256"]):
                raise ValueError(f"FINMA outlinks lost parent binding: {identity}")
            links = link_row.get("links") or []
            if bool(link_row.get("explicit_empty")) == bool(links):
                raise ValueError(f"FINMA explicit-empty XOR failed: {identity}")
            for link in links:
                if (not str(link.get("target_url") or "").strip()
                        or not str(link.get("context") or "").strip()
                        or link.get("method") not in {
                            "pdf-annotation-uri", "markdown-uri"}
                        or link.get("relation")
                        != "citation-or-external-uri"):
                    raise ValueError(f"FINMA link target/context failed: {identity}")
                if (link["method"] == "pdf-annotation-uri"
                        and (not isinstance(link.get("page"), int)
                             or link["page"] < 1
                             or link["page"] > len(pages))):
                    raise ValueError(f"FINMA annotation page context failed: {identity}")
                method_counts[link["method"]] += 1
            all_ids.add(identity)
            all_ordinals.add(ordinal)
            local_ordinals.add(ordinal)
            body_digests[source["sha256"]] += 1
            total_pages += len(pages)
            total_links += len(links)
            linked += bool(links)
            empty += not bool(links)
        if local_ordinals != set(range(start, start + limit)):
            raise ValueError("FINMA partition ordinal conservation failed")
        partition_evidence.append({
            "start": start, "limit": limit, "run": _artifact(path, artifact_root),
            "lineage_mode": "metadata-only-context-repair" if repaired
            else "corrected-direct-producer",
            "markdown_stage_records": len(parents),
            "outlink_stage_records": len(outlinks),
        })

    if (len(all_ids) != expected_count
            or all_ordinals != set(range(expected_start, expected_start + expected_count))
            or linked + empty != expected_count):
        raise ValueError("FINMA combined range conservation failed")
    value = {
        "schema": "aw.finma-web-pdf-partition-audit.v1",
        "version": VERSION,
        "status": "PASS",
        "source_profile": "web_pdf",
        "selection": {
            "expected_start": expected_start,
            "expected_count": expected_count,
            "retained_pdf_population": retained_population,
            "partitions": expected_ranges,
        },
        "counts": {
            "canonical_source_records": len(all_ids),
            "unique_ordinals": len(all_ordinals),
            "unique_pdf_bodies": len(body_digests),
            "source_records_sharing_a_pdf_body": sum(
                count for count in body_digests.values() if count > 1),
            "pages_with_geometry": total_pages,
            "outlink_parents": linked + empty,
            "linked_parents": linked,
            "explicit_empty_parents": empty,
            "contextual_links": total_links,
            **dict(sorted(metadata_counts.items())),
            **{f"links_{key}": value for key, value in sorted(method_counts.items())},
        },
        "checks": {
            "disjoint_contiguous_partition_conservation": "PASS",
            "canonical_url_is_source_record_identity": "PASS",
            "body_sha_is_lineage_not_identity": "PASS",
            "exact_l0_pdf_and_api_metadata_binding": "PASS",
            "full_docling_markdown_no_cap": "PASS",
            "page_geometry_and_extraction_diagnostics": "PASS",
            "target_context_and_explicit_empty_outlinks": "PASS",
        },
        "l0_receipt": l0_binding,
        "partition_evidence": partition_evidence,
        "logical_identity_sha256": hashlib.sha256("".join(
            canonical_json({"ordinal": ordinal, "source_record_id": identity}) + "\n"
            for ordinal, identity in sorted(
                (parent.get("ordinal"), parent.get("source_record_id"))
                for _, _, _, run in partitions
                for parent in _rows(artifact_root, run["stages"]["markdown"],
                                    "finma.logical"))).encode()).hexdigest(),
    }
    atomic_publish(output,
                   (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
    return value


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog or "finma_web_pdf_partitions.py", description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-start", type=int, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit(artifact_root=args.artifact_root,
                           parent_runs=args.parent_run,
                           output=args.output,
                           expected_start=args.expected_start,
                           expected_count=args.expected_count),
                     indent=2, sort_keys=True))
    return 0
