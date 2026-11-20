#!/usr/bin/env python3
"""Materialize preserved Statistics Canada WDS originals into Markdown/outlinks.

Moved from the pre-move driver by task 1405 (S7). The producer label below is written into sealed
receipts, so it is frozen byte-for-byte (declared in `cytir.ops.public_cut.FROZEN_RECEIPT_LABELS`)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS


#: frozen: the producer id carried by every receipt this driver has sealed
PRODUCER_ID = "pipeline/acquire_statcan_wds_preserved.py"
VERSION = "statcan-wds-preserved.v2"
RUN_SCHEMA = "aw.statcan-wds-acquisition-run.v1"
WDS = "https://www150.statcan.gc.ca/t1/wds/rest"


class StatCanError(RuntimeError):
    pass


def _logical(rows: list[dict[str, Any]]) -> tuple[bytes, str]:
    data = b"".join((canonical_json(row) + "\n").encode() for row in rows)
    return data, hashlib.sha256(data).hexdigest()


def _title(cube: dict[str, Any], lang: str) -> str:
    return str(cube.get("cubeTitleEn" if lang == "en" else "cubeTitleFr") or "").strip()


def _url(pid: int | str, lang: str) -> str:
    return f"https://www150.statcan.gc.ca/t1/tbl1/{lang}/tv.action?pid={pid}01"


def validate_transport(path: Path, body_path: Path, headers_path: Path,
               expected_url: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("schema") != "aw.http-fetch-metadata.v1" or
            value.get("requested_url") != expected_url or
            value.get("http_status") != 200 or not value.get("final_url") or
            not value.get("fetched_at")):
        raise StatCanError(f"invalid HTTP transport receipt: {path}")
    for label, artifact_path in (("body", body_path),
                                 ("response_headers", headers_path)):
        artifact = value.get(label) or {}
        if (artifact.get("bytes") != artifact_path.stat().st_size or
                artifact.get("sha256") != sha256_file(artifact_path)):
            raise StatCanError(f"{label} disagrees with transport receipt: {path}")
    return value


def materialize(*, cubes_path: Path, codesets_path: Path,
                cubes_headers_path: Path, codesets_headers_path: Path,
                cubes_metadata_path: Path, codesets_metadata_path: Path, root: Path,
                fetch_vantage: str, fetched_at: str,
                supersedes_run: Path | None = None,
                producer_version: str = VERSION) -> dict[str, Any]:
    try:
        cubes = json.loads(cubes_path.read_text(encoding="utf-8"))
        codesets = json.loads(codesets_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StatCanError(f"invalid WDS input: {exc}") from exc
    if not isinstance(cubes, list) or not isinstance(codesets, (list, dict)):
        raise StatCanError("WDS inputs have unexpected top-level shapes")
    root.mkdir(parents=True, exist_ok=True)

    originals = []
    original_artifacts = []
    inputs = (
        ("getAllCubesListLite", cubes_path, cubes_headers_path, cubes_metadata_path),
        ("getCodeSets", codesets_path, codesets_headers_path, codesets_metadata_path),
    )
    for endpoint, path, headers_path, metadata_path in sorted(inputs):
        requested_url = f"{WDS}/{endpoint}"
        transport = validate_transport(metadata_path, path, headers_path, requested_url)
        body = path.read_bytes()
        for family, suffix, source_path in (
                ("body", "json", path), ("headers", "headers", headers_path),
                ("metadata", "json", metadata_path)):
            artifact = PS.store_artifact(
                root, f"stages/original/data/{family}", suffix,
                source_path.read_bytes())
            artifact_path = PS.artifact_path(root, artifact)
            if artifact_path not in original_artifacts:
                original_artifacts.append(artifact_path)
        originals.append({
            "endpoint": requested_url,
            "requested_url": transport["requested_url"],
            "final_url": transport["final_url"],
            "http_status": transport["http_status"],
            "fetched_at": transport["fetched_at"],
            "body_bytes": len(body),
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "response_headers_bytes": headers_path.stat().st_size,
            "response_headers_sha256": sha256_file(headers_path),
            "transport_metadata_sha256": sha256_file(metadata_path),
        })
    original_logical, original_sha = _logical(originals)
    original_stage = SS.seal(
        artifact_root=root, artifact_paths=original_artifacts,
        receipt_path=root / "stages/original/receipt.json", records=2,
        fmt="exact-http-response-body+headers+canonical-transport-metadata+deterministic-gzip",
        logical_sha256=original_sha, producer_id=PRODUCER_ID,
        producer_version=producer_version)

    markdown_rows, outlink_rows = [], []
    skipped_archived = skipped_untitled = 0
    parent_sha = originals[0]["body_sha256"]
    for cube in sorted(cubes, key=lambda row: int(row.get("productId") or 0)):
        if str(cube.get("archived")) != "2":
            skipped_archived += 1
            continue
        pid = cube.get("productId")
        for lang in ("en", "fr"):
            title = _title(cube, lang)
            if not pid or not title:
                skipped_untitled += 1
                continue
            url = _url(pid, lang)
            labels = ({"source": "Source", "period": "Reference period",
                       "release": "Latest release"} if lang == "en" else
                      {"source": "Source", "period": "Période de référence",
                       "release": "Dernière diffusion"})
            markdown = (
                f"# {title}\n\n"
                f"- {labels['source']}: Statistics Canada / Statistique Canada\n"
                f"- Product ID: {pid}\n"
                f"- {labels['period']}: {str(cube.get('cubeStartDate') or '')[:10]}"
                f" – {str(cube.get('cubeEndDate') or '')[:10]}\n"
                f"- {labels['release']}: {str(cube.get('releaseTime') or '')[:10]}\n"
                f"- Canonical table: {url}\n"
            )
            doc_id = hashlib.sha256(f"statcan:{pid}:{lang}".encode()).hexdigest()
            markdown_rows.append({"document_id": doc_id, "product_id": pid,
                                  "lang": lang, "url": url,
                                  "parent_sha256": parent_sha, "markdown": markdown})
            outlink_rows.append({"document_id": doc_id, "source_url":
                                 f"{WDS}/getAllCubesListLite", "target_url": url,
                                 "anchor_or_context": "Canonical table"})

    markdown_data, markdown_sha = _logical(markdown_rows)
    markdown_artifact = PS.store_artifact(
        root, "stages/markdown/data", "jsonl", markdown_data)
    markdown_stage = SS.seal(
        artifact_root=root,
        artifact_paths=[PS.artifact_path(root, markdown_artifact)],
        receipt_path=root / "stages/markdown/receipt.json",
        records=len(markdown_rows), fmt="canonical-jsonl+deterministic-gzip",
        logical_sha256=markdown_sha,
        producer_id=PRODUCER_ID,
        producer_version=producer_version)

    outlink_data, outlink_sha = _logical(outlink_rows)
    outlink_artifact = PS.store_artifact(
        root, "stages/outlinks/data", "jsonl", outlink_data)
    outlink_stage = SS.seal(
        artifact_root=root,
        artifact_paths=[PS.artifact_path(root, outlink_artifact)],
        receipt_path=root / "stages/outlinks/receipt.json",
        records=len(outlink_rows), fmt="canonical-jsonl+deterministic-gzip",
        logical_sha256=outlink_sha,
        producer_id=PRODUCER_ID,
        producer_version=producer_version)

    run = {
        "schema": RUN_SCHEMA, "version": VERSION, "status": "complete",
        "fetched_at": fetched_at, "fetch_vantage": fetch_vantage,
        "route": "documented_source_api",
        "source": "Statistics Canada WDS",
        "content_scope": {
            "classification": "bilingual_catalog_pointer_records",
            "includes": ["cube title", "reference period", "latest release date",
                         "canonical table URL"],
            "excludes": ["table observations", "dimension members", "full table content"],
        },
        "counts": {"cube_records": len(cubes), "active_markdown_documents": len(markdown_rows),
                   "outlink_records": len(outlink_rows),
                   "archived_cubes_skipped": skipped_archived,
                   "untitled_language_records_skipped": skipped_untitled},
        "stages": {"original": original_stage, "markdown": markdown_stage,
                   "outlinks": outlink_stage},
        "outlink_cadence": ["after_completed_wave", "weekly"],
        "downstream": {"chunks": "pending_source_profile",
                       "vectors": "pending_1024d_fp16"},
    }
    if supersedes_run is not None:
        run["supersedes"] = {
            "run_path": str(supersedes_run),
            "run_sha256": sha256_file(supersedes_run),
            "reason": "v1 preserved exact bodies but omitted per-response HTTP headers/status/final URL",
        }
    data = (json.dumps(run, indent=2, sort_keys=True) + "\n").encode()
    run_path = root / "RUN.json"
    if run_path.exists():
        if sha256_file(run_path) != hashlib.sha256(data).hexdigest():
            raise StatCanError("immutable RUN receipt collision")
    else:
        atomic_publish(run_path, data)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cubes-json", type=Path, required=True)
    parser.add_argument("--codesets-json", type=Path, required=True)
    parser.add_argument("--cubes-headers", type=Path, required=True)
    parser.add_argument("--codesets-headers", type=Path, required=True)
    parser.add_argument("--cubes-metadata", type=Path, required=True)
    parser.add_argument("--codesets-metadata", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fetch-vantage", required=True)
    parser.add_argument("--fetched-at", required=True)
    parser.add_argument("--supersedes-run", type=Path)
    args = parser.parse_args()
    try:
        result = materialize(
            cubes_path=args.cubes_json.resolve(), codesets_path=args.codesets_json.resolve(),
            cubes_headers_path=args.cubes_headers.resolve(),
            codesets_headers_path=args.codesets_headers.resolve(),
            cubes_metadata_path=args.cubes_metadata.resolve(),
            codesets_metadata_path=args.codesets_metadata.resolve(),
            root=args.root.resolve(), fetch_vantage=args.fetch_vantage,
            fetched_at=args.fetched_at,
            supersedes_run=(args.supersedes_run.resolve()
                            if args.supersedes_run else None))
    except (OSError, StatCanError, SS.StageSealError) as exc:
        print(f"FATAL: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
