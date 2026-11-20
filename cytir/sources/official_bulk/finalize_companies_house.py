"""Fail-closed finalizer for a pinned Companies House company-data ZIP."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import zipfile

from cytir.acquire.artifact_io import atomic_publish, sha256_file
from cytir.sources.official_bulk.http_evidence import redact_url


VERSION = "companies-house-company-data-fetch.v1"
DOC_URL = "https://download.companieshouse.gov.uk/en_output.html"
SCHEMA_URL = "https://resources.companieshouse.gov.uk/toolsToHelp/pdf/freeDataProductDataset.pdf"


def _headers(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="iso-8859-1").splitlines():
        if line.startswith("HTTP/"):
            values = {":status": line}
        elif ":" in line:
            key, value = line.split(":", 1); values[key.strip().casefold()] = value.strip()
    return values


def _transport(path: Path, body: Path, headers: Path, url: str) -> None:
    value = json.loads(path.read_text())
    if (value.get("schema") != "aw.http-fetch-metadata.v1" or
            value.get("requested_url") != redact_url(url) or value.get("http_status") != 200):
        raise ValueError("Companies House documentation/HEAD transport failed")
    for key, artifact in (("body", body), ("response_headers", headers)):
        declared = value.get(key) or {}
        if (declared.get("bytes") != artifact.stat().st_size or
                declared.get("sha256") != sha256_file(artifact)):
            raise ValueError("Companies House transport artifact mismatch")


def finalize(*, url: str, expected_member: str, partial: Path, body: Path,
             headers: Path, metadata: Path, attempts_dir: Path, head_body: Path,
             head_headers: Path, head_metrics: Path, head_transport: Path,
             docs_body: Path, docs_headers: Path, docs_metrics: Path,
             docs_transport: Path, schema_body: Path, schema_headers: Path,
             schema_metrics: Path, schema_transport: Path) -> dict:
    _transport(head_transport, head_body, head_headers, url)
    _transport(docs_transport, docs_body, docs_headers, DOC_URL)
    _transport(schema_transport, schema_body, schema_headers, SCHEMA_URL)
    identity = _headers(head_headers)
    expected_bytes = int(identity.get("content-length") or 0)
    etag = identity.get("etag", ""); last_modified = identity.get("last-modified", "")
    if (expected_bytes <= 0 or not etag or not last_modified or
            identity.get("content-type", "").casefold() != "application/zip"):
        raise ValueError("Companies House HEAD lacks pinned source identity")
    source = body if body.exists() else partial
    if not source.is_file() or source.stat().st_size != expected_bytes:
        raise ValueError("Companies House body has wrong final byte count")
    evidence = [head_body, head_headers, head_metrics, head_transport,
                docs_body, docs_headers, docs_metrics, docs_transport,
                schema_body, schema_headers, schema_metrics, schema_transport]
    attempts = []; previous = None
    for started in sorted(attempts_dir.glob("attempt-*.started")):
        prefix = started.with_suffix("")
        paths = {suffix: Path(str(prefix) + suffix) for suffix in
                 (".started", ".completed", ".rc", ".start_offset", ".end_offset",
                  ".headers", ".metrics")}
        if not all(path.is_file() for path in paths.values()):
            raise ValueError("incomplete Companies House attempt evidence")
        evidence.extend(paths.values())
        if paths[".metrics"].stat().st_size == 0:
            recovery_path = Path(str(prefix) + ".recovery.json")
            interrupted_path = Path(str(prefix) + ".interrupted")
            if recovery_path.is_file():
                recovery = json.loads(recovery_path.read_text()); evidence.append(recovery_path)
                if (recovery.get("schema") != "aw.interrupted-http-attempt-recovery.v1" or
                        recovery.get("start_offset") != int(paths[".start_offset"].read_text()) or
                        recovery.get("end_offset") != int(paths[".end_offset"].read_text())):
                    raise ValueError("interrupted attempt recovery boundary mismatch")
                metrics = {"http_code": recovery.get("http_status"),
                           "url_effective": recovery.get("final_url")}
            elif interrupted_path.is_file():
                evidence.append(interrupted_path)
                status_line = _headers(paths[".headers"]).get(":status", "")
                try: status = int(status_line.split()[1])
                except (IndexError, ValueError) as exc:
                    raise ValueError("interrupted attempt lacks HTTP status evidence") from exc
                metrics = {"http_code": status, "url_effective": url}
            else:
                raise ValueError("empty attempt metrics lack immutable interruption/recovery evidence")
        else:
            metrics = json.loads(paths[".metrics"].read_text())
        row = {"attempt": prefix.name, "started_at": started.read_text().strip(),
               "completed_at": paths[".completed"].read_text().strip(),
               "return_code": int(paths[".rc"].read_text()),
               "start_offset": int(paths[".start_offset"].read_text()),
               "end_offset": int(paths[".end_offset"].read_text()),
               "http_status": int(metrics.get("http_code") or 0),
               "final_url": metrics.get("url_effective")}
        row["status"] = "complete" if row["return_code"] == 0 else "failed_resumable"
        if (row["start_offset"] < 0 or row["end_offset"] < row["start_offset"] or
                row["end_offset"] > expected_bytes or
                (previous is not None and row["start_offset"] != previous)):
            raise ValueError("Companies House resumable offsets are not conservative")
        previous = row["end_offset"]; attempts.append(row)
    if not attempts or attempts[-1]["status"] != "complete":
        raise ValueError("Companies House transfer lacks terminal successful attempt")
    terminal = attempts[-1]
    if (terminal["end_offset"] != expected_bytes or
            terminal["http_status"] not in (200, 206) or
            (terminal["start_offset"] > 0 and terminal["http_status"] != 206) or
            terminal["final_url"] != url):
        raise ValueError("Companies House terminal response violates pinned identity")
    terminal_headers = Path(str(attempts_dir / terminal["attempt"]) + ".headers")
    if _headers(terminal_headers).get("etag") != etag:
        raise ValueError("Companies House terminal ETag differs from HEAD")
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        if (len(members) != 1 or members[0].is_dir() or
                members[0].filename != expected_member):
            raise ValueError("Companies House ZIP member identity is invalid")
        bad = archive.testzip()
    if bad is not None:
        raise ValueError(f"Companies House ZIP CRC failed: {bad}")
    artifact_root = metadata.parents[1].resolve(); evidence_rows = []
    for artifact in evidence:
        relative = artifact.resolve().relative_to(artifact_root)
        evidence_rows.append({"uri": f"artifact://{relative.as_posix()}",
            "bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)})
    value = {"schema": "aw.resumable-http-fetch-metadata.v1", "version": VERSION,
        "requested_url": url, "final_url": url, "http_status": 200,
        "fetch_mode": "range-resumed-if-match-etag-pinned",
        "body": {"bytes": expected_bytes, "sha256": sha256_file(source)},
        "response_headers": {"bytes": terminal_headers.stat().st_size,
                             "sha256": sha256_file(terminal_headers)},
        "source_identity": {"etag": etag, "last_modified": last_modified,
            "zip_members": 1, "csv_member": expected_member,
            "head_response_sha256": sha256_file(head_headers),
            "official_product_page_sha256": sha256_file(docs_body),
            "official_field_schema_sha256": sha256_file(schema_body)},
        "attempts": attempts, "evidence_artifacts": evidence_rows}
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if not headers.exists(): atomic_publish(headers, terminal_headers.read_bytes())
    if not body.exists():
        with partial.open("rb") as stream: os.fsync(stream.fileno())
        os.replace(partial, body); descriptor = os.open(body.parent, os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    if not metadata.exists(): atomic_publish(metadata, encoded)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True); parser.add_argument("--expected-member", required=True)
    for name in ("partial", "body", "headers", "metadata", "attempts-dir", "head-body",
                 "head-headers", "head-metrics", "head-transport", "docs-body", "docs-headers",
                 "docs-metrics", "docs-transport", "schema-body", "schema-headers",
                 "schema-metrics", "schema-transport"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv); values = vars(args)
    print(json.dumps(finalize(**{key.replace("-", "_"): (value.resolve() if isinstance(value, Path) else value)
        for key, value in values.items()}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
