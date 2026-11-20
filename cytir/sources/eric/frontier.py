#!/usr/bin/env python3
"""Build the exact official ERIC full-text PDF URL frontier from terminal API metadata.

Ported from the deleted `build_eric_pdf_frontier` script by task 2182; logic and output bytes
unchanged. The origin host is the family constant `PDF_HOST` (the contract row's consent host).

    python -m cytir.sources.eric.frontier --inventory-root D --root D --artifact-root D
"""
from __future__ import annotations

import argparse
from collections import Counter
from cytir.acquire.gzip_pinned import deterministic_gzip_open
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as SS


VERSION = "eric-official-pdf-frontier.v2"
PRODUCER = "aw/sources/eric/frontier.py"
ID = re.compile(r"[A-Z]{2}[0-9]{6,9}")


def build(*, inventory_root: Path, root: Path, artifact_root: Path) -> dict:
    from cytir.sources.families.eric_official_pdf import PDF_HOST as host   # lazy: the family imports this package
    if root.exists():
        raise ValueError("immutable ERIC PDF frontier root already exists; use a fresh versioned root")
    source_run_path = inventory_root / "RUN.portable.v2.json"
    source = json.loads(source_run_path.read_text())
    if (source.get("schema") != "aw.eric-api-inventory-run.v1" or
            source.get("status") != "complete"):
        raise ValueError("ERIC portable inventory is not terminal")
    # The immutable portable reseal already deep-rehashed all 13,161 source
    # artifacts. Validate that contract here, then hash each page body exactly
    # once while deriving the frontier; repeating the full deep walk would read
    # the 1.3 GB source stage twice without adding lineage evidence.
    _validate_stage(source["stages"]["original"], "eric.original",
                    artifact_root=artifact_root, deep=False)
    expected = int(source["counts"]["unique_document_ids"])
    logical = hashlib.sha256(); prefix_counts = Counter(); count = 0
    with tempfile.TemporaryDirectory(prefix="aw-eric-frontier-", dir=inventory_root.parent) as directory:
        temporary = Path(directory)
        database = sqlite3.connect(temporary/"ids.sqlite")
        database.execute("CREATE TABLE frontier (eric_id TEXT PRIMARY KEY, page_sha256 TEXT NOT NULL)")
        for path in sorted((inventory_root / "pages").glob("*.body.json")):
            body = json.loads(path.read_text())
            page_sha256 = sha256_file(path)
            docs = ((body.get("response") or {}).get("docs") or [])
            for record in docs:
                eric_id = str(record.get("id") or "")
                if record.get("e_fulltextauth") != 1 or not ID.fullmatch(eric_id):
                    raise ValueError(f"invalid authorized ERIC identity: {eric_id!r}")
                try:
                    database.execute("INSERT INTO frontier VALUES (?, ?)", (eric_id, page_sha256))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"duplicate authorized ERIC identity: {eric_id}") from exc
        database.commit()
        count = int(database.execute("SELECT COUNT(*) FROM frontier").fetchone()[0])
        if count != expected:
            raise ValueError(f"ERIC PDF frontier conservation failure: {count} != {expected}")
        compressed_path = temporary/"frontier.jsonl.gz"
        with compressed_path.open("wb") as raw, deterministic_gzip_open(
                raw, compresslevel=6) as zipped:
            for eric_id,page_sha256 in database.execute(
                    "SELECT eric_id,page_sha256 FROM frontier ORDER BY eric_id"):
                row={"eric_id":eric_id,
                    "pdf_url":f"https://{host}/fulltext/{eric_id}.pdf",
                    "source_page_body_sha256":page_sha256,
                    "derivation":"official_files_eric_fulltext_path_from_source_native_id"}
                line = (canonical_json(row)+"\n").encode()
                logical.update(line)
                zipped.write(line)
                prefix_counts[eric_id[:2]] += 1
        database.close()
        root.mkdir(parents=True, exist_ok=False)
        frontier_path = root/"frontier.jsonl.gz"
        with compressed_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(compressed_path, frontier_path)
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    source_artifacts = [source_run_path,
        inventory_root / "stages/original/receipt.portable.v2.json"]
    stage = SS.seal(artifact_root=artifact_root,
        artifact_paths=[frontier_path, *source_artifacts],
        receipt_path=root/"stages/frontier/receipt.json", records=count,
        fmt="deterministic-gzip-canonical-jsonl+portable-source-inventory-bindings",
        logical_sha256=logical.hexdigest(),
        producer_id=PRODUCER, producer_version=VERSION)
    prefix_counts = dict(sorted(prefix_counts.items()))
    value = {"schema": "aw.eric-official-pdf-frontier-run.v1", "version": VERSION,
        "status": "complete", "source_inventory": {
            "uri": f"artifact://{source_run_path.relative_to(artifact_root).as_posix()}",
            "bytes": source_run_path.stat().st_size, "sha256": sha256_file(source_run_path)},
        "counts": {"catalog_records": expected, "official_pdf_urls": count,
                   "source_native_id_prefixes": prefix_counts},
        "url_contract": f"https://{host}/fulltext/{{source-native ERIC id}}.pdf",
        "content_scope": "official PDF acquisition frontier only; no PDF/document/point claim",
        "stages": {"frontier": stage},
        "downstream": "bounded resumable exact PDF response acquisition with explicit terminal quarantine"}
    atomic_publish(root/"RUN.json", (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
    return value


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog or "frontier", description=__doc__)
    parser.add_argument("--inventory-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(inventory_root=args.inventory_root.resolve(), root=args.root.resolve(),
        artifact_root=args.artifact_root.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
