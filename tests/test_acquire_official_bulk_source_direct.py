import json
from pathlib import Path

from cytir.sources.official_bulk import direct as subject
from cytir.index.preservation_contract import _validate_stage
import cytir.sources.official_bulk.http_evidence as record_http_fetch


def test_large_bulk_is_sealed_in_place_and_projection_blocked(tmp_path: Path):
    root = tmp_path / "root"
    body = root / "input/source.7z"
    body.parent.mkdir(parents=True)
    body.write_bytes(b"large-fixture")
    headers = root / "input/source.headers"
    headers.write_bytes(b"HTTP/2 200\r\n\r\n")
    metrics = tmp_path / "metrics.json"
    url = "https://official.example/source.7z"
    metrics.write_text(json.dumps({"http_code": 200, "url_effective": url}))
    metadata = root / "input/source.transport.json"
    record_http_fetch.record(
        body=body, headers=headers, curl_metrics=metrics,
        requested_url=url, fetched_at="2026-08-15T00:00:00Z",
        output=metadata)
    run = subject.acquire(
        body=body, headers=headers, metadata=metadata, root=root,
        artifact_root=tmp_path,
        source_id="x", source_name="X", requested_url=url,
        license_name="CC-BY-SA-4.0", content_scope="source rows",
        next_projection="full markdown", fetch_vantage="fetch-box")
    assert run["status"] == "source_acquired_pending_projection"
    assert run["index_admission"].startswith("blocked")
    verified = _validate_stage(run["stages"]["original"], "original",
                               artifact_root=tmp_path, deep=True)
    assert verified["records"] == 1
    assert len(verified["artifacts"]) == 3
    assert body in [tmp_path / a["uri"].removeprefix("artifact://")
                    for a in verified["artifacts"]]
    assert all(str(tmp_path) not in a["uri"] for a in verified["artifacts"])
