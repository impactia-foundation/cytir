"""Validate and seal a resumable ERIC API inventory acquisition (moved from the flat
acquire_eric_api_inventory script by task 1386; the family adapter `cytir.sources.families.eric_official_pdf`
calls it). The page FETCH stays in the operator shell loop; this module validates each sealed page
receipt and seals the whole inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
import urllib.parse


from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file  # noqa: E402
from cytir.acquire import stage_receipt as SS  # noqa: E402


VERSION = "eric-api-inventory.v1"
SCHEMA = "aw.eric-api-inventory-run.v1"
BASE = "https://api.ies.ed.gov/eric/"
#: frozen label: stage receipts sealed before the move carry this producer string; a rewritten
#: label would make the same inventory seal to different bytes (task 1386 characterization)
PRODUCER_ID = "pipeline/acquire_eric_api_inventory.py"


class EricInventoryError(RuntimeError):
    pass


def _canonical_lines(rows: list[dict[str, Any]]) -> tuple[bytes, str]:
    data = b"".join((canonical_json(row) + "\n").encode() for row in rows)
    return data, hashlib.sha256(data).hexdigest()


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {
        "uri": f"artifact://{relative}",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def requested_url(query: str, fields: str, rows: int, start: int) -> str:
    return BASE + "?" + urllib.parse.urlencode({
        "search": query, "fields": fields, "rows": rows, "start": start,
        "sort": "id asc", "format": "json"})


def validate_page(root: Path, receipt_path: Path, expected_start: int,
                   expected_rows: int, expected_query: str,
                   expected_fields: str, expected_total: int | None) -> tuple[dict, list[Path]]:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EricInventoryError(f"invalid page receipt {receipt_path}: {exc}") from exc
    if receipt.get("schema") != "aw.eric-api-page-receipt.v1":
        raise EricInventoryError(f"wrong page receipt schema: {receipt_path}")
    if receipt.get("start") != expected_start or receipt.get("rows") != expected_rows:
        raise EricInventoryError(f"page offset mismatch: {receipt_path}")
    if receipt.get("search") != expected_query or receipt.get("fields") != expected_fields:
        raise EricInventoryError(f"page query mismatch: {receipt_path}")

    paths: list[Path] = []
    for label in ("body", "response_headers", "curl_metrics", "transport_metadata"):
        item = receipt.get("artifacts", {}).get(label) or {}
        uri = str(item.get("uri") or "")
        if not uri.startswith("artifact://"):
            raise EricInventoryError(f"invalid {label} URI: {receipt_path}")
        path = root / uri.removeprefix("artifact://")
        if not path.is_file() or path.stat().st_size != item.get("bytes"):
            raise EricInventoryError(f"missing/size-mismatched {label}: {path}")
        if sha256_file(path) != item.get("sha256"):
            raise EricInventoryError(f"digest-mismatched {label}: {path}")
        paths.append(path)

    transport = json.loads(paths[3].read_text(encoding="utf-8"))
    expected_url = requested_url(expected_query, expected_fields,
                                 expected_rows, expected_start)
    if (transport.get("http_status") != 200 or
            transport.get("requested_url") != expected_url or
            transport.get("final_url") != expected_url or
            not transport.get("fetched_at")):
        raise EricInventoryError(f"non-200 page transport: {receipt_path}")
    for label, path in (("body", paths[0]), ("response_headers", paths[1])):
        item = transport.get(label) or {}
        if item.get("bytes") != path.stat().st_size or item.get("sha256") != sha256_file(path):
            raise EricInventoryError(f"transport does not bind {label}: {receipt_path}")
    body = json.loads(paths[0].read_text(encoding="utf-8"))
    response = body.get("response") if isinstance(body, dict) else None
    if not isinstance(response, dict) or response.get("start") != expected_start:
        raise EricInventoryError(f"invalid ERIC response offset: {receipt_path}")
    total = response.get("numFound")
    if not isinstance(total, int) or response.get("numFoundExact") is not True:
        raise EricInventoryError(f"non-exact ERIC count: {receipt_path}")
    if expected_total is not None and total != expected_total:
        raise EricInventoryError(f"ERIC total changed {expected_total} -> {total}")
    docs = response.get("docs")
    if not isinstance(docs, list) or len(docs) > expected_rows:
        raise EricInventoryError(f"invalid docs page: {receipt_path}")
    ids = [str(row.get("id") or "") for row in docs if isinstance(row, dict)]
    if len(ids) != len(docs) or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise EricInventoryError(f"invalid/duplicate IDs within page: {receipt_path}")
    if any(row.get("e_fulltextauth") != 1 for row in docs):
        raise EricInventoryError(f"non-authorized row in full-text inventory: {receipt_path}")
    if receipt.get("num_found") != total or receipt.get("document_ids") != ids:
        raise EricInventoryError(f"page receipt/body mismatch: {receipt_path}")
    return {
        "start": expected_start,
        "documents": len(docs),
        "body_bytes": paths[0].stat().st_size,
        "body_sha256": sha256_file(paths[0]),
        "document_ids_sha256": hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest(),
    }, paths + [receipt_path]


def finalize(*, root: Path, query: str, fields: str, rows: int,
             producer_version: str = VERSION) -> dict[str, Any]:
    receipts = sorted((root / "pages").glob("*.receipt.json"))
    if not receipts:
        raise EricInventoryError("no page receipts")
    page_rows: list[dict[str, Any]] = []
    artifacts: list[Path] = []
    all_ids: set[str] = set()
    total: int | None = None
    start = 0
    while total is None or start < total:
        receipt_path = root / "pages" / f"{start:09d}.receipt.json"
        if not receipt_path.is_file():
            raise EricInventoryError(f"missing page receipt at start={start}")
        row, paths = validate_page(root, receipt_path, start, rows, query, fields, total)
        if total is None:
            total = json.loads(receipt_path.read_text())["num_found"]
        ids = json.loads(receipt_path.read_text())["document_ids"]
        overlap = all_ids.intersection(ids)
        if overlap:
            raise EricInventoryError(f"cross-page duplicate IDs: {sorted(overlap)[:3]}")
        all_ids.update(ids)
        page_rows.append(row)
        artifacts.extend(paths)
        start += rows
    assert total is not None
    if len(all_ids) != total or sum(row["documents"] for row in page_rows) != total:
        raise EricInventoryError(
            f"inventory conservation failed: unique={len(all_ids)} rows={sum(r['documents'] for r in page_rows)} total={total}")
    expected_receipts = (total + rows - 1) // rows
    if len(receipts) != expected_receipts:
        raise EricInventoryError(f"unexpected page receipt count: {len(receipts)} != {expected_receipts}")

    index_data, logical_sha = _canonical_lines(page_rows)
    index_path = root / "inventory-pages.jsonl"
    if index_path.exists() and index_path.read_bytes() != index_data:
        raise EricInventoryError("immutable inventory index collision")
    if not index_path.exists():
        atomic_publish(index_path, index_data)
    artifacts.append(index_path)
    stage = SS.seal(
        artifact_root=root, artifact_paths=artifacts,
        receipt_path=root / "stages/original/receipt.json",
        records=len(page_rows),
        fmt="exact-http-response-body+headers+curl-metrics+canonical-transport-metadata+page-receipts",
        logical_sha256=logical_sha,
        producer_id=PRODUCER_ID,
        producer_version=producer_version)
    run = {
        "schema": SCHEMA,
        "version": producer_version,
        "status": "complete",
        "source": "ERIC API / Institute of Education Sciences",
        "route": "documented_source_api",
        "query": {"search": query, "fields": fields, "rows": rows,
                  "requested_sort": "id asc",
                  "ordering_note": "request recorded verbatim; conservation does not assume lexical server ordering"},
        "content_scope": {
            "classification": "full_text_authorized_catalog_records_and_pdf_pointers",
            "records": total,
            "includes": ["source-native ERIC metadata", "full-text authorization flag", "record URL fields"],
            "excludes": ["PDF bytes", "PDF principal text", "production point projection"],
        },
        "counts": {"api_pages": len(page_rows), "catalog_records": total,
                   "unique_document_ids": len(all_ids)},
        "stages": {"original": stage},
        "downstream": {"full_text_pdf_acquisition": "pending_exact_inventory_terminal",
                       "markdown": "blocked_until_pdf_preservation",
                       "chunks": "blocked_until_full_markdown"},
    }
    run_data = (json.dumps(run, indent=2, sort_keys=True) + "\n").encode()
    run_path = root / "RUN.json"
    if run_path.exists() and run_path.read_bytes() != run_data:
        raise EricInventoryError("immutable RUN collision")
    if not run_path.exists():
        atomic_publish(run_path, run_data)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--fields", required=True)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--validate-page-start", type=int)
    parser.add_argument("--expected-total", type=int)
    args = parser.parse_args()
    try:
        if args.validate_page_start is not None:
            receipt_path = (args.root.resolve() / "pages" /
                            f"{args.validate_page_start:09d}.receipt.json")
            result, _ = validate_page(
                args.root.resolve(), receipt_path, args.validate_page_start,
                args.rows, args.query, args.fields, args.expected_total)
        else:
            result = finalize(root=args.root.resolve(), query=args.query,
                              fields=args.fields, rows=args.rows)
    except (OSError, ValueError, EricInventoryError, SS.StageSealError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
