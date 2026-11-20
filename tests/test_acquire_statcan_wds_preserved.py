import json
from pathlib import Path

import cytir.sources.statcan.preserved as subject
import cytir.sources.official_bulk.http_evidence as record_http_fetch


def transport(tmp_path: Path, name: str, body: Path, url: str):
    headers = tmp_path / f"{name}.headers"
    headers.write_bytes(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n")
    metrics = tmp_path / f"{name}.curl.json"
    metrics.write_text(json.dumps({"http_code": 200, "url_effective": url}))
    metadata = tmp_path / f"{name}.transport.json"
    record_http_fetch.record(
        body=body, headers=headers, curl_metrics=metrics, requested_url=url,
        fetched_at="2026-08-15T00:00:00Z", output=metadata)
    return headers, metadata


def test_materialize_preserves_api_markdown_and_outlinks(tmp_path: Path):
    cubes = tmp_path / "cubes.json"
    codes = tmp_path / "codes.json"
    cubes.write_text(json.dumps([
        {"productId": 2, "archived": "2", "cubeTitleEn": "Active English title",
         "cubeTitleFr": "Titre français actif", "cubeStartDate": "2020-01-01",
         "cubeEndDate": "2026-01-01", "releaseTime": "2026-08-01"},
        {"productId": 1, "archived": "1", "cubeTitleEn": "Archived",
         "cubeTitleFr": "Archivé"},
    ]))
    codes.write_text("[]")
    cubes_url = f"{subject.WDS}/getAllCubesListLite"
    codes_url = f"{subject.WDS}/getCodeSets"
    cubes_headers, cubes_metadata = transport(tmp_path, "cubes", cubes, cubes_url)
    codes_headers, codes_metadata = transport(tmp_path, "codes", codes, codes_url)
    root = tmp_path / "root"
    result = subject.materialize(
        cubes_path=cubes, codesets_path=codes, root=root,
        cubes_headers_path=cubes_headers, codesets_headers_path=codes_headers,
        cubes_metadata_path=cubes_metadata, codesets_metadata_path=codes_metadata,
        fetch_vantage="fixture", fetched_at="2026-08-15T00:00:00Z")
    assert result["counts"] == {
        "cube_records": 2, "active_markdown_documents": 2,
        "outlink_records": 2, "archived_cubes_skipped": 1,
        "untitled_language_records_skipped": 0,
    }
    for stage in ("original", "markdown", "outlinks"):
        receipt = result["stages"][stage]
        assert receipt["deterministic_encoding"] is True
        assert receipt["receipt"]["uri"].startswith("artifact://")
        body = json.loads((root / receipt["receipt"]["uri"].removeprefix(
            "artifact://")).read_text())
        assert body["schema"] == "aw.stage-receipt.v1"
        assert body["artifacts"]
    inventory = json.loads((root / "stages/original/receipt.json").read_text())[
        "artifacts"]
    assert len(inventory) >= 5  # identical raw header blocks content-deduplicate
    assert len([row for row in inventory if "/metadata/" in row["uri"]]) == 2
    assert result["content_scope"]["classification"] == "bilingual_catalog_pointer_records"
