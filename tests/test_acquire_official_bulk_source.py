import json
from pathlib import Path

from cytir.sources.official_bulk import bulk as subject
from cytir.index.preservation_contract import _validate_stage
import cytir.sources.official_bulk.http_evidence as record_http_fetch


def test_seals_source_and_blocks_index_admission(tmp_path: Path):
    body = tmp_path / "source.parquet"
    headers = tmp_path / "headers"
    metrics = tmp_path / "metrics"
    metadata = tmp_path / "metadata"
    body.write_bytes(b"PAR1fixture")
    headers.write_bytes(b"HTTP/2 200\r\n\r\n")
    url = "https://official.example/stable"
    metrics.write_text(json.dumps({"http_code": 200, "url_effective": url}))
    record_http_fetch.record(body=body, headers=headers, curl_metrics=metrics,
                             requested_url=url,
                             fetched_at="2026-08-15T00:00:00Z", output=metadata)
    root = tmp_path / "root"
    run = subject.acquire(body=body, headers=headers, metadata=metadata,
                          root=root, source_id="x", source_name="X", country="FR",
                          requested_url=url, content_scope="identity spine",
                          next_projection="join website evidence",
                          fetch_vantage="fixture")
    assert run["status"] == "source_acquired_pending_projection"
    assert run["index_admission"].startswith("blocked")
    verified = _validate_stage(run["stages"]["original"], "bulk.original",
                               artifact_root=root, deep=True)
    assert verified["records"] == 1
    assert len(verified["artifacts"]) == 3
