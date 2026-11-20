import json
from pathlib import Path

import pytest

import cytir.sources.official_bulk.http_evidence as subject


def test_records_exact_transport_evidence(tmp_path: Path):
    body = tmp_path / "body"
    headers = tmp_path / "headers"
    metrics = tmp_path / "metrics"
    output = tmp_path / "metadata"
    body.write_bytes(b"{}")
    headers.write_bytes(b"HTTP/2 200\r\ncontent-type: application/json\r\n\r\n")
    metrics.write_text(json.dumps({"http_code": 200,
                                   "url_effective": "https://example.test/final"}))
    value = subject.record(
        body=body, headers=headers, curl_metrics=metrics,
        requested_url="https://example.test/start",
        fetched_at="2026-08-15T00:00:00Z", output=output)
    assert value["http_status"] == 200
    assert value["requested_url"].endswith("/start")
    assert value["final_url"].endswith("/final")
    assert value == json.loads(output.read_text())


def test_rejects_non_success(tmp_path: Path):
    body = tmp_path / "body"
    headers = tmp_path / "headers"
    metrics = tmp_path / "metrics"
    body.write_bytes(b"no")
    headers.write_bytes(b"HTTP/2 404\r\n\r\n")
    metrics.write_text(json.dumps({"http_code": 404,
                                   "url_effective": "https://example.test"}))
    with pytest.raises(ValueError, match="not a complete success"):
        subject.record(body=body, headers=headers, curl_metrics=metrics,
                       requested_url="https://example.test",
                       fetched_at="2026-08-15T00:00:00Z",
                       output=tmp_path / "out")
