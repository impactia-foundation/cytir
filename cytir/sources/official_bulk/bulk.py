"""Seal one official bulk response as a source-only, projection-pending lane (moved from the flat
acquire_official_bulk_source script by task 1394; the adapter `cytir.sources.families.official_bulk` calls it).

FROZEN LABEL: PRODUCER_ID is the string stage receipts sealed before the move carry; it never changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS
from cytir.sources.official_bulk.http_evidence import redact_url



PRODUCER_ID = "pipeline/acquire_official_bulk_source.py"


def check_transport(*, body: Path, headers: Path, metadata: Path, requested_url: str) -> dict:
    """Validate the sealed HTTP transport metadata against the files on disk; return it."""
    transport = json.loads(metadata.read_text(encoding="utf-8"))
    if (transport.get("schema") != "aw.http-fetch-metadata.v1" or
            transport.get("requested_url") != redact_url(requested_url) or
            transport.get("http_status") != 200 or
            not transport.get("final_url") or not transport.get("fetched_at")):
        raise ValueError("invalid bulk HTTP transport metadata")
    for key, path in (("body", body), ("response_headers", headers)):
        expected = transport.get(key) or {}
        if expected.get("bytes") != path.stat().st_size or expected.get(
                "sha256") != sha256_file(path):
            raise ValueError(f"{key} disagrees with transport metadata")
    return transport


def acquire(*, body: Path, headers: Path, metadata: Path, root: Path,
            source_id: str, source_name: str, country: str, requested_url: str,
            content_scope: str, next_projection: str, fetch_vantage: str) -> dict:
    transport = check_transport(body=body, headers=headers, metadata=metadata,
                                requested_url=requested_url)
    root.mkdir(parents=True, exist_ok=True)
    stored = []
    for family, suffix, path in (("body", body.suffix.lstrip(".") or "bin", body),
                                 ("headers", "headers", headers),
                                 ("metadata", "json", metadata)):
        artifact = PS.store_artifact(root, f"stages/original/data/{family}",
                                     suffix, path.read_bytes())
        stored.append(PS.artifact_path(root, artifact))
    row = {"requested_url": redact_url(requested_url), "final_url": transport["final_url"],
           "http_status": 200, "fetched_at": transport["fetched_at"],
           "body_bytes": body.stat().st_size, "body_sha256": sha256_file(body),
           "response_headers_bytes": headers.stat().st_size,
           "response_headers_sha256": sha256_file(headers),
           "transport_metadata_sha256": sha256_file(metadata)}
    logical = hashlib.sha256((canonical_json(row) + "\n").encode()).hexdigest()
    original = SS.seal(
        artifact_root=root, artifact_paths=stored,
        receipt_path=root / "stages/original/receipt.json", records=1,
        fmt="exact-http-response+headers+canonical-transport-metadata+deterministic-gzip",
        logical_sha256=logical,
        producer_id=PRODUCER_ID,
        producer_version="official-bulk-source.v1")
    run = {
        "schema": "aw.official-bulk-source-run.v1", "version": "v1",
        "status": "source_acquired_pending_projection", "source_id": source_id,
        "source": source_name, "country": country,
        "route": "source_owned_bulk_dump", "fetch_vantage": fetch_vantage,
        "content_scope": content_scope, "stages": {"original": original},
        "next_projection": next_projection,
        "index_admission": "blocked_until_markdown_chunks_outlinks_and_privacy_gates_seal",
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
    for name in ("body", "headers", "metadata", "root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("source-id", "source-name", "country", "requested-url",
                 "content-scope", "next-projection", "fetch-vantage"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    result = acquire(body=args.body.resolve(), headers=args.headers.resolve(),
                     metadata=args.metadata.resolve(), root=args.root.resolve(),
                     source_id=args.source_id, source_name=args.source_name,
                     country=args.country, requested_url=args.requested_url,
                     content_scope=args.content_scope,
                     next_projection=args.next_projection,
                     fetch_vantage=args.fetch_vantage)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
