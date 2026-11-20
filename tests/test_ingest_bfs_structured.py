from cytir.extract.bfs_pxweb import structured_record


def test_bfs_structured_record_preserves_all_values_and_parent_response():
    meta = {"title": "Population", "variables": [{"code": "YEAR", "text": "Year",
             "time": True, "values": [str(year) for year in range(2000, 2025)],
             "valueTexts": [str(year) for year in range(2000, 2025)]}]}
    row = structured_record("px-x-1", "en", meta,
                            response_uri="artifact://responses/a.json",
                            response_sha256="a" * 64, retrieved_at="2026-08-15T00:00:00Z")
    assert row["text"].startswith("# Population\n\n## Record identity")
    assert "### 1. Year" in row["text"]
    assert "- `2024` — 2024" in row["text"]
    assert row["source_record_id"] == "en:px-x-1"
    assert row["parent_response"]["sha256"] == "a" * 64
    assert len(row["source_links"]) == 2


def test_the_old_driver_path_forwards_to_the_package_module():
    from cytir.extract import bfs_pxweb
    assert structured_record is bfs_pxweb.structured_record
