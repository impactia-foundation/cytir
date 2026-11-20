"""Fail-closed finalizer for a resumable GLEIF Golden Copy CSV ZIP."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

from cytir.sources.official_bulk.http_evidence import redact_url
from cytir.acquire.artifact_io import atomic_publish, sha256_file


VERSION = "gleif-golden-copy-fetch.v1"


def _publish_large(source: Path, destination: Path) -> None:
    with source.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(source, destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _manifest_entry(path: Path) -> tuple[dict, dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    data = manifest.get("data") or {}
    entry = ((data.get("full_file") or {}).get("csv") or {})
    if (data.get("type") != "lei2" or entry.get("type") != "lei2"
            or entry.get("format") != "csv" or entry.get("delta_type") != "GoldenCopy"
            or not re.fullmatch(r"LEI_[0-9.]+", str(entry.get("cdf_version") or ""))
            or not re.fullmatch(r"https://goldencopy\.gleif\.org/storage/.+\.csv\.zip",
                                str(entry.get("url") or ""))
            or int(entry.get("record_count") or 0) <= 0
            or int(entry.get("size") or 0) <= 0
            or not data.get("publish_date")):
        raise ValueError("GLEIF latest manifest has no valid unique full CSV identity")
    return data, entry


def finalize(*, partial: Path, body: Path, headers: Path, metadata: Path,
             attempts_dir: Path, manifest_body: Path, manifest_headers: Path,
             manifest_metrics: Path, manifest_transport: Path,
             license_body: Path, license_headers: Path, license_metrics: Path,
             license_transport: Path) -> dict:
    data, entry = _manifest_entry(manifest_body)
    expected_bytes = int(entry["size"])
    source = body if body.exists() else partial
    if not source.is_file() or source.stat().st_size != expected_bytes:
        raise ValueError("GLEIF body does not match manifest byte count")
    manifest_fetch = json.loads(manifest_transport.read_text())
    license_fetch = json.loads(license_transport.read_text())
    for value, url, source_body, source_headers in (
            (manifest_fetch,
             "https://goldencopy.gleif.org/api/v2/golden-copies/publishes/lei2/latest",
             manifest_body, manifest_headers),
            (license_fetch, "https://www.gleif.org/en/meta/lei-data-terms-of-use",
             license_body, license_headers)):
        if (value.get("schema") != "aw.http-fetch-metadata.v1"
                or value.get("requested_url") != url or value.get("http_status") != 200):
            raise ValueError("GLEIF manifest/license transport is invalid")
        for key, path in (("body", source_body), ("response_headers", source_headers)):
            expected = value.get(key) or {}
            if (expected.get("bytes") != path.stat().st_size
                    or expected.get("sha256") != sha256_file(path)):
                raise ValueError("GLEIF manifest/license transport artifact mismatch")
    evidence = [manifest_body, manifest_headers, manifest_metrics, manifest_transport,
                license_body, license_headers, license_metrics, license_transport]
    attempts = []
    for started in sorted(attempts_dir.glob("attempt-*.started")):
        prefix = started.with_suffix("")
        paths = {suffix: Path(str(prefix) + suffix) for suffix in
                 (".started", ".completed", ".rc", ".start_offset",
                  ".end_offset", ".headers", ".metrics")}
        if not all(path.is_file() for path in paths.values()):
            raise ValueError("incomplete GLEIF range-attempt evidence")
        evidence.extend(paths.values())
        metrics = json.loads(paths[".metrics"].read_text() or "{}")
        row = {"attempt": prefix.name,
               "started_at": started.read_text().strip(),
               "completed_at": paths[".completed"].read_text().strip(),
               "return_code": int(paths[".rc"].read_text()),
               "start_offset": int(paths[".start_offset"].read_text()),
               "end_offset": int(paths[".end_offset"].read_text()),
               "http_status": int(metrics.get("http_code") or 0),
               "final_url": metrics.get("url_effective")}
        row["status"] = "complete" if row["return_code"] == 0 else "failed_resumable"
        attempts.append(row)
    if not attempts or attempts[-1]["status"] != "complete":
        raise ValueError("GLEIF transfer has no terminal successful attempt")
    previous = None
    for row in attempts:
        if (row["start_offset"] < 0 or row["end_offset"] < row["start_offset"]
                or row["end_offset"] > expected_bytes
                or (previous is not None and row["start_offset"] != previous)):
            raise ValueError("GLEIF resumable offset chain is not conservative")
        previous = row["end_offset"]
    terminal = attempts[-1]
    if (terminal["end_offset"] != expected_bytes
            or terminal["http_status"] not in (200, 206)
            or (terminal["start_offset"] > 0 and terminal["http_status"] != 206)
            or terminal["final_url"] != entry["url"]):
        raise ValueError("GLEIF terminal response disagrees with selected manifest entry")
    terminal_headers = Path(str(attempts_dir / terminal["attempt"]) + ".headers")
    with zipfile.ZipFile(source) as archive:
        members = archive.namelist()
        bad = archive.testzip()
    if bad is not None or len(members) != 1 or not members[0].endswith(".csv"):
        raise ValueError("GLEIF ZIP CRC/member validation failed")
    evidence_rows = []
    for path in evidence:
        relative = path.resolve().relative_to(metadata.parents[1].resolve())
        evidence_rows.append({"uri": f"artifact://{relative.as_posix()}",
                              "bytes": path.stat().st_size,
                              "sha256": sha256_file(path)})
    value = {
        "schema": "aw.resumable-http-fetch-metadata.v1", "version": VERSION,
        "requested_url": redact_url(entry["url"]), "final_url": terminal["final_url"],
        "http_status": 200, "fetch_mode": "range-resumed-composite",
        "body": {"bytes": expected_bytes, "sha256": sha256_file(source)},
        "source_identity": {
            "publish_date": data["publish_date"], "cdf_version": entry["cdf_version"],
            "record_count": int(entry["record_count"]), "format": "csv.zip",
            "manifest_response_sha256": sha256_file(manifest_body),
            "license_response_sha256": sha256_file(license_body),
            "license": "CC0 1.0 Universal", "zip_member": members[0]},
        "attempts": attempts, "evidence_artifacts": evidence_rows,
        "response_headers": {"bytes": terminal_headers.stat().st_size,
                             "sha256": sha256_file(terminal_headers)},
    }
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if not headers.exists():
        atomic_publish(headers, terminal_headers.read_bytes())
    if not body.exists():
        _publish_large(partial, body)
    if not metadata.exists():
        atomic_publish(metadata, encoded)
    elif sha256_file(metadata) != hashlib.sha256(encoded).hexdigest():
        raise ValueError("immutable GLEIF transport collision")
    return value


def main(argv: list[str] | None = None) -> int:
    args = argparse.ArgumentParser(description=__doc__)
    for name in ("partial", "body", "headers", "metadata", "attempts-dir",
                 "manifest-body", "manifest-headers", "manifest-metrics",
                 "manifest-transport", "license-body", "license-headers",
                 "license-metrics", "license-transport"):
        args.add_argument(f"--{name}", type=Path, required=True)
    a = args.parse_args(argv)
    result = finalize(
        partial=a.partial.resolve(), body=a.body.resolve(), headers=a.headers.resolve(),
        metadata=a.metadata.resolve(), attempts_dir=a.attempts_dir.resolve(),
        manifest_body=a.manifest_body.resolve(), manifest_headers=a.manifest_headers.resolve(),
        manifest_metrics=a.manifest_metrics.resolve(), manifest_transport=a.manifest_transport.resolve(),
        license_body=a.license_body.resolve(), license_headers=a.license_headers.resolve(),
        license_metrics=a.license_metrics.resolve(), license_transport=a.license_transport.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
