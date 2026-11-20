"""Seal a large exact HTTP bulk body in place without a duplicate gzip copy (moved from the flat
acquire_official_bulk_source_direct script by task 1394; the adapter calls it).

FROZEN LABEL: PRODUCER_ID is the string stage receipts sealed before the move carry; it never changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire import stage_receipt as SS
from cytir.sources.official_bulk.http_evidence import redact_url


VERSION = "official-bulk-source-direct.v1"


PRODUCER_ID = "pipeline/acquire_official_bulk_source_direct.py"


def check_transport(*, body: Path, headers: Path, metadata: Path, requested_url: str) -> dict:
    """Validate the sealed HTTP transport metadata against the files on disk; return it."""
    transport = json.loads(metadata.read_text(encoding="utf-8"))
    if (transport.get("schema") not in ("aw.http-fetch-metadata.v1",
                                        "aw.resumable-http-fetch-metadata.v1") or
            transport.get("requested_url") != redact_url(requested_url) or
            transport.get("http_status") != 200 or
            not transport.get("final_url")):
        raise ValueError("invalid large-bulk HTTP transport metadata")
    for key, path in (("body", body), ("response_headers", headers)):
        if (transport.get("schema") == "aw.resumable-http-fetch-metadata.v1" and
                key == "response_headers" and not transport.get(key)):
            continue
        expected = transport.get(key) or {}
        if (expected.get("bytes") != path.stat().st_size or
                expected.get("sha256") != sha256_file(path)):
            raise ValueError(f"{key} disagrees with transport metadata")
    return transport


def acquire(*, body: Path, headers: Path, metadata: Path, root: Path,
            artifact_root: Path,
            source_id: str, source_name: str, requested_url: str,
            license_name: str, content_scope: str, next_projection: str,
            fetch_vantage: str) -> dict:
    transport = check_transport(body=body, headers=headers, metadata=metadata,
                                requested_url=requested_url)
    fetched_at = transport.get("fetched_at") or next(
        (row.get("started_at") for row in transport.get("attempts", [])
         if row.get("started_at")), None)
    row = {"requested_url": redact_url(requested_url), "final_url": transport["final_url"],
           "http_status": 200, "fetched_at": fetched_at,
           "body_bytes": body.stat().st_size, "body_sha256": sha256_file(body),
           "response_headers_bytes": headers.stat().st_size,
           "response_headers_sha256": sha256_file(headers),
           "transport_metadata_sha256": sha256_file(metadata)}
    logical = hashlib.sha256((canonical_json(row) + "\n").encode()).hexdigest()
    extra_artifacts = []
    for artifact in transport.get("evidence_artifacts", []):
        uri = str(artifact.get("uri") or "")
        if not uri.startswith("artifact://"):
            raise ValueError("invalid resumable evidence artifact URI")
        path = root / uri.removeprefix("artifact://")
        if (path.stat().st_size != artifact.get("bytes") or
                sha256_file(path) != artifact.get("sha256")):
            raise ValueError("resumable evidence artifact mismatch")
        extra_artifacts.append(path)
    original = SS.seal(
        artifact_root=artifact_root,
        artifact_paths=[body, headers, metadata] + extra_artifacts,
        receipt_path=root / "stages/original/receipt.json", records=1,
        fmt="exact-http-response+headers+canonical-transport-metadata",
        logical_sha256=logical,
        producer_id=PRODUCER_ID,
        producer_version=VERSION)
    run = {
        "schema": "aw.official-bulk-source-run.v1", "version": VERSION,
        "status": "source_acquired_pending_projection", "source_id": source_id,
        "source": source_name, "route": "source_owned_bulk_dump",
        "fetch_vantage": fetch_vantage, "license": license_name,
        "content_scope": content_scope, "stages": {"original": original},
        "next_projection": next_projection,
        "index_admission": "blocked_until_markdown_chunks_outlinks_and_quality_gates_seal",
    }
    data = (json.dumps(run, indent=2, sort_keys=True) + "\n").encode()
    path = root / "RUN.json"
    if path.exists():
        if sha256_file(path) != hashlib.sha256(data).hexdigest():
            raise ValueError("immutable RUN collision")
    else:
        atomic_publish(path, data)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("body", "headers", "metadata", "root", "artifact-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("source-id", "source-name", "requested-url", "license-name",
                 "content-scope", "next-projection", "fetch-vantage"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    result = acquire(
        body=args.body.resolve(), headers=args.headers.resolve(),
        metadata=args.metadata.resolve(), root=args.root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        source_id=args.source_id, source_name=args.source_name,
        requested_url=args.requested_url, license_name=args.license_name,
        content_scope=args.content_scope, next_projection=args.next_projection,
        fetch_vantage=args.fetch_vantage)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
