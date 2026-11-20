#!/usr/bin/env python3
"""Deep-verify ERIC PDF L0 once for all disjoint materialization workers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.sources.eric import materialize as M


SCHEMA = "aw.eric-pdf-partition-preflight.v1"
VERSION = "eric-pdf-partition-preflight.v1"
PRODUCER = "pipeline/audit_eric_pdf_partition_preflight.py"


def audit(*, acquisition_root: Path, artifact_root: Path, output: Path,
          limit: int) -> dict:
    run_path = acquisition_root / "RUN.json"
    run = json.loads(run_path.read_text())
    if (run.get("schema") != "aw.eric-official-pdf-acquisition-run.v1"
            or run.get("status") != "complete"):
        raise ValueError("terminal ERIC PDF acquisition RUN required")
    original = run["stages"]["original"]
    source_receipts = []
    for path in sorted((acquisition_root / "receipts").glob("*.json")):
        value = json.loads(path.read_text())
        if value.get("status") == "complete":
            source_receipts.append(path)
    if len(source_receipts) != run["counts"]["pdf_complete"]:
        raise ValueError("complete ERIC PDF receipt conservation failed")
    if limit <= 0 or len(source_receipts) < limit:
        raise ValueError("invalid ERIC partition preflight limit")
    source_receipts = source_receipts[:limit]
    inventory = M._selection_inventory(source_receipts)
    selected_bytes = 0
    for source_path in source_receipts:
        receipt = json.loads(source_path.read_text())
        M._validate_parent_receipt(receipt, artifact_root)
        selected_bytes += receipt["artifacts"]["response_body"]["bytes"]
    value = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "PASS",
        "acquisition_run": {
            "uri": "artifact://" + run_path.resolve().relative_to(
                artifact_root.resolve()).as_posix(),
            "bytes": run_path.stat().st_size,
            "sha256": sha256_file(run_path),
        },
        "acquisition_run_sha256": sha256_file(run_path),
        "original_stage_metadata_sha256": hashlib.sha256(
            canonical_json(original).encode()).hexdigest(),
        "selected_complete_pdfs": len(source_receipts),
        "selection_logical_sha256": M._selection_logical_sha256(inventory),
        "selection_inventory": inventory,
        "deep_verification": {
            "selected_response_bodies_rehashed": len(source_receipts),
            "selected_response_bytes_rehashed": selected_bytes,
        },
        "producer": {"id": PRODUCER, "version": VERSION},
    }
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if output.exists() and output.read_bytes() != encoded:
        raise ValueError("immutable ERIC partition preflight collision")
    atomic_publish(output, encoded)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(
        acquisition_root=args.acquisition_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        output=args.output.resolve(), limit=args.limit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
