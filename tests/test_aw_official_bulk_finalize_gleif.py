import io
import json
from pathlib import Path
import zipfile

import cytir.sources.official_bulk.finalize_gleif as subject
from cytir.acquire.artifact_io import sha256_file


def _fetch_evidence(root: Path, name: str, url: str, body: bytes):
    path = root / "input" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    headers = Path(str(path) + ".headers"); headers.write_bytes(b"HTTP/2 200\r\n\r\n")
    metrics = Path(str(path) + ".curl.json"); metrics.write_text("{}")
    transport = Path(str(path) + ".transport.json")
    transport.write_text(json.dumps({
        "schema": "aw.http-fetch-metadata.v1", "requested_url": url,
        "http_status": 200, "body": {"bytes": len(body), "sha256": sha256_file(path)},
        "response_headers": {"bytes": headers.stat().st_size,
                             "sha256": sha256_file(headers)}}))
    return path, headers, metrics, transport


def test_finalizes_exact_manifest_selected_zip_and_range_evidence(tmp_path: Path):
    root = tmp_path / "root"; attempts = root / "input/attempts"; attempts.mkdir(parents=True)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("lei.csv", "LEI,EntityName\nX,Example\n")
    data = payload.getvalue()
    url = "https://goldencopy.gleif.org/storage/x/file.csv.zip"
    manifest_value = {"data": {"type": "lei2", "publish_date": "2026-08-15 08:00:00",
        "full_file": {"csv": {"type": "lei2", "format": "csv", "record_count": 1,
            "size": len(data), "delta_type": "GoldenCopy", "url": url,
            "cdf_version": "LEI_3.1"}}}}
    manifest = _fetch_evidence(root, "latest.json",
        "https://goldencopy.gleif.org/api/v2/golden-copies/publishes/lei2/latest",
        json.dumps(manifest_value).encode())
    license_set = _fetch_evidence(root, "license.html",
        "https://www.gleif.org/en/meta/lei-data-terms-of-use", b"CC0")
    partial = root / "input/file.csv.zip.partial"; partial.write_bytes(data)
    prefix = attempts / "attempt-0001"
    for suffix, value in ((".started", "t0"), (".completed", "t1"), (".rc", "0"),
                          (".start_offset", "0"), (".end_offset", str(len(data))),
                          (".headers", "HTTP/2 200\r\n"),
                          (".metrics", json.dumps({"http_code": 200,
                                                  "url_effective": url}))):
        Path(str(prefix) + suffix).write_text(value)
    body=root/"input/file.csv.zip"; headers=Path(str(body)+".headers")
    metadata=Path(str(body)+".transport.json")
    result = subject.finalize(
        partial=partial, body=body, headers=headers, metadata=metadata,
        attempts_dir=attempts, manifest_body=manifest[0], manifest_headers=manifest[1],
        manifest_metrics=manifest[2], manifest_transport=manifest[3],
        license_body=license_set[0], license_headers=license_set[1],
        license_metrics=license_set[2], license_transport=license_set[3])
    assert result["source_identity"]["record_count"] == 1
    assert result["source_identity"]["license"] == "CC0 1.0 Universal"
    assert result["body"]["bytes"] == len(data)
    assert body.is_file() and not partial.exists()
