"""Split full-article JORF parents from explicit source-content residuals.

Moved from the frozen script of the same name by task 1437 (S7). `PRODUCER` is the label sealed
receipts carry, so it stays byte for byte; the old path is a forwarding shim.
"""
from __future__ import annotations
import gzip
import json
import os
from pathlib import Path

from cytir.acquire.artifact_io import atomic_publish, sha256_file
from cytir.acquire.deterministic_gzip import DeterministicGzipWriter
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as SS
from cytir.sources.jorf.population import (
    JORF_FULL_PARENTS, JORF_FULL_PARENTS_EMPTY, JORF_FULL_PARENTS_WITH_CONTENT)


VERSION = "jorf-full-content-filter.v1"
PRODUCER = "pipeline/filter_jorf_full_content_parents.py"
EXPECTED_SOURCE = JORF_FULL_PARENTS
EXPECTED_FULL = JORF_FULL_PARENTS_WITH_CONTENT
EXPECTED_RESIDUAL = JORF_FULL_PARENTS_EMPTY


def _streams(root: Path, run: dict) -> tuple[Path, Path]:
    stage = _validate_stage(
        run.get("stages", {}).get("materialization"), "jorf.materialization",
        artifact_root=root, deep=True)
    paths = [root / ref["uri"].removeprefix("artifact://")
             for ref in stage.get("artifacts") or []]
    markdown = [path for path in paths if path.as_posix().endswith("/markdown/parents.jsonl.gz")]
    outlinks = [path for path in paths if path.as_posix().endswith("/outlinks/parents.jsonl.gz")]
    if len(markdown) != 1 or len(outlinks) != 1:
        raise ValueError("JORF materialization lacks singular parent/outlink streams")
    return markdown[0], outlinks[0]


def filter_full(*, artifact_root: Path, parent_run: Path, output: Path) -> dict:
    root = artifact_root.resolve(); output = output.resolve(); output.relative_to(root)
    if output.exists(): raise ValueError(f"immutable JORF full-content projection exists: {output}")
    run = json.loads(parent_run.read_text())
    if (run.get("schema") != "aw.jorf-exact-full-markdown-run.v1"
            or run.get("version") != "jorf-exact-full-markdown.v2"
            or run.get("status") != "materialization-complete-profile-fidelity-chunking-pending"
            or (run.get("counts") or {}).get("documents") != EXPECTED_SOURCE):
        raise ValueError("JORF v2 materialization is not terminal/exact")
    markdown_path, outlinks_path = _streams(root, run)
    partial = output.with_name(f"{output.name}.partial.{os.getpid()}")
    partial.mkdir(parents=True, exist_ok=False)
    paths = {"markdown": partial / "markdown/parents.jsonl.gz",
             "outlinks": partial / "outlinks/parents.jsonl.gz",
             "residual": partial / "residual/source-content.jsonl.gz"}
    writers = {key: DeterministicGzipWriter(path) for key, path in paths.items()}
    counts = {"source_documents": 0, "full_article_documents": 0,
              "source_content_residuals": 0, "full_article_bodies": 0,
              "empty_article_records_retained": 0, "links": 0,
              "unresolved_links": 0, "explicit_empty_outlink_parents": 0}
    accepted = set(); all_ids = set()
    try:
        with gzip.open(markdown_path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line); source_id = str(row.get("source_record_id") or "")
                if not source_id or source_id in all_ids: raise ValueError("JORF parent identity drift")
                all_ids.add(source_id); counts["source_documents"] += 1
                if int(row.get("article_bodies") or 0) > 0:
                    writers["markdown"].write(row); accepted.add(source_id)
                    counts["full_article_documents"] += 1
                    counts["full_article_bodies"] += int(row["article_bodies"])
                    counts["empty_article_records_retained"] += int(row.get("empty_article_records") or 0)
                else:
                    writers["residual"].write({
                        "schema": "aw.jorf-source-content-residual.v1",
                        "source_record_id": source_id,
                        "baseline_production_point_ids": row.get("baseline_production_point_ids") or [],
                        "reason": "no_full_article_body_in_terminal_official_linked_record_state",
                        "body_sections": int(row.get("body_sections") or 0),
                        "source_record_count": int(row.get("source_record_count") or 0),
                        "markdown_sha256": row.get("markdown_sha256"),
                        "parent_identity": row.get("parent_identity"),
                    })
                    counts["source_content_residuals"] += 1
        seen_outlinks = set()
        with gzip.open(outlinks_path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line); source_id = str(row.get("source_record_id") or "")
                if source_id in seen_outlinks or source_id not in all_ids:
                    raise ValueError("JORF outlink identity/conservation drift")
                seen_outlinks.add(source_id)
                if source_id in accepted:
                    writers["outlinks"].write(row)
                    counts["links"] += len(row.get("links") or [])
                    counts["unresolved_links"] += len(row.get("unresolved") or [])
                    counts["explicit_empty_outlink_parents"] += bool(row.get("explicit_empty"))
        if seen_outlinks != all_ids: raise ValueError("JORF parent/outlink population mismatch")
    finally:
        for writer in writers.values(): writer.close()
    if ((counts["source_documents"], counts["full_article_documents"],
         counts["source_content_residuals"]) !=
            (EXPECTED_SOURCE, EXPECTED_FULL, EXPECTED_RESIDUAL)):
        raise ValueError(f"JORF full-content conservation drift: {counts}")
    stages = {}
    for name, records, fmt in (
        ("markdown", counts["full_article_documents"],
         "full-official-jorftext-markdown-with-article-bodies+exact-source-lineage"),
        ("outlinks", counts["full_article_documents"],
         "parent-bound-jorf-legal-links+explicit-unresolved-and-empty"),
        ("residual", counts["source_content_residuals"],
         "explicit-no-full-article-body-source-residuals+baseline-point-binding"),
    ):
        stages[name] = SS.seal(
            artifact_root=root, artifact_paths=[paths[name]],
            receipt_path=partial / f"stages/{name}/receipt.json", records=records,
            fmt=fmt, logical_sha256=sha256_file(paths[name]), producer_id=PRODUCER,
            producer_version=VERSION, publish_from=partial, publish_to=output)
    result = {"schema": "aw.jorf-full-content-filter-run.v1", "version": VERSION,
        "status": "full-content-projection-complete-profile-fidelity-chunking-pending",
        "source_id": "dila-jorf-exact-production-v1", "source_profile": "anchored_law",
        "counts": counts, "parent_run": {
            "uri": f"artifact://{parent_run.resolve().relative_to(root).as_posix()}",
            "bytes": parent_run.stat().st_size, "sha256": sha256_file(parent_run)},
        "stages": stages,
        "admission": "74,459 full-article parents only; 759 explicit source-content residuals remain unadmitted; profile/fidelity/token/chunk/vector gates pending; no point subtraction"}
    atomic_publish(partial / "RUN.json",
                   (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    os.rename(partial, output); return result
