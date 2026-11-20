import json
from pathlib import Path
import zipfile

import cytir.sources.official_bulk.finalize_companies_house as subject
from cytir.acquire import artifact_io as close_interrupted_http_attempt
import cytir.sources.official_bulk.http_evidence as record_http_fetch


def _transport(body: Path, headers: Path, url: str):
    metrics = body.with_suffix(body.suffix + ".curl.json")
    metrics.write_text(json.dumps({"http_code": 200, "url_effective": url}))
    output = body.with_suffix(body.suffix + ".transport.json")
    record_http_fetch.record(body=body, headers=headers, curl_metrics=metrics,
        requested_url=url, fetched_at="2026-08-15T00:00:00Z", output=output)
    return metrics, output


def test_exact_zip_and_interrupted_206_chain_are_preserved(tmp_path: Path):
    root = tmp_path / "root"; attempts = root / "input/attempts"; attempts.mkdir(parents=True)
    partial = root / "input/company.zip.partial"; member = "BasicCompanyDataAsOneFile-2026-07-01.csv"
    with zipfile.ZipFile(partial, "w") as archive:
        archive.writestr(member, "CompanyName,CompanyNumber\nA,1\n")
    size = partial.stat().st_size; etag = '"fixture-etag"'
    url = "https://download.companieshouse.gov.uk/BasicCompanyDataAsOneFile-2026-07-01.zip"
    head_body = root / "input/head.body"; head_body.write_bytes(b"")
    head_headers = root / "input/head.headers"
    head_headers.write_text(f"HTTP/2 200\r\ncontent-length: {size}\r\ncontent-type: application/zip\r\netag: {etag}\r\nlast-modified: now\r\n\r\n")
    head_metrics, head_transport = _transport(head_body, head_headers, url)
    docs_body = root / "input/docs.html"; docs_body.write_text("official product")
    docs_headers = root / "input/docs.headers"; docs_headers.write_text("HTTP/2 200\r\n\r\n")
    docs_metrics, docs_transport = _transport(docs_body, docs_headers, subject.DOC_URL)
    schema_body = root / "input/schema.pdf"; schema_body.write_bytes(b"%PDF-fields")
    schema_headers = root / "input/schema.headers"; schema_headers.write_text("HTTP/2 200\r\n\r\n")
    schema_metrics, schema_transport = _transport(schema_body, schema_headers, subject.SCHEMA_URL)
    split = size // 2
    for number, start, end, rc, status in ((1, 0, split, 18, 200), (2, split, size, 0, 206)):
        prefix = attempts / f"attempt-{number:04d}"
        prefix.with_suffix(".started").write_text(f"2026-08-15T00:00:0{number}Z\n")
        prefix.with_suffix(".completed").write_text(f"2026-08-15T00:00:1{number}Z\n")
        prefix.with_suffix(".rc").write_text(f"{rc}\n")
        prefix.with_suffix(".start_offset").write_text(f"{start}\n")
        prefix.with_suffix(".end_offset").write_text(f"{end}\n")
        prefix.with_suffix(".headers").write_text(f"HTTP/2 {status}\r\netag: {etag}\r\n\r\n")
        prefix.with_suffix(".metrics").write_text(json.dumps(
            {"http_code": status, "url_effective": url}))
    interrupted = attempts / "attempt-0001"
    for suffix in (".completed", ".rc", ".end_offset"):
        interrupted.with_suffix(suffix).unlink()
    interrupted.with_suffix(".metrics").write_bytes(b"")
    close_interrupted_http_attempt.close_attempt(prefix=interrupted,
        successor_prefix=attempts / "attempt-0002", url=url)
    body = root / "input/company.zip"; headers = root / "input/company.zip.headers"
    metadata = root / "input/company.zip.transport.json"
    result = subject.finalize(url=url, expected_member=member, partial=partial,
        body=body, headers=headers, metadata=metadata, attempts_dir=attempts,
        head_body=head_body, head_headers=head_headers, head_metrics=head_metrics,
        head_transport=head_transport, docs_body=docs_body, docs_headers=docs_headers,
        docs_metrics=docs_metrics, docs_transport=docs_transport, schema_body=schema_body,
        schema_headers=schema_headers, schema_metrics=schema_metrics,
        schema_transport=schema_transport)
    assert result["body"]["bytes"] == size
    assert [row["start_offset"] for row in result["attempts"]] == [0, split]
    assert result["attempts"][-1]["http_status"] == 206
    assert result["source_identity"]["csv_member"] == member
    assert result["source_identity"]["official_field_schema_sha256"]
    assert body.is_file() and not partial.exists()
