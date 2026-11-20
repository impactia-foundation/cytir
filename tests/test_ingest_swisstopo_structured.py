from cytir.extract.swisstopo import structured_record


def test_structured_record_preserves_semantic_fields_and_source_links():
    row = {"UUID": "{U}", "NAME": "Bern", "STATUS": "offiziell",
           "SPRACHCODE": "Hochdeutsch", "NAMEN_TYP": "einfacher Name",
           "NAME_UUID": "{N}", "NAMENGRUPPE_UUID": "{G}",
           "OBJEKTART": "Ort", "OBJEKTKLASSE_TLM": "TLM_SIEDLUNGSNAME",
           "KUNSTBAUTE": "k_W", "E": "2600000", "N": "1200000", "Z": "540"}
    record = structured_record(row, item_id="swissnames3d_2026",
                               item_datetime="2026-04-29T00:00:00Z",
                               parent_member="swissNAMES3D_PKT.csv",
                               parent_archive_sha256="a" * 64,
                               archive_url="https://data.geo.admin.ch/archive.zip")
    assert record["text"].startswith("# Bern\n\n## Record identity")
    assert "- SwissNAMES3D UUID: `{U}`" in record["text"]
    assert "## Classification" in record["text"]
    assert "## Coordinates" in record["text"]
    assert record["source_record_id"] == "{U}"
    assert record["parent_archive"]["member"] == "swissNAMES3D_PKT.csv"
    assert len(record["source_links"]) == 3
    assert all("relation" in link and "context" in link
               for link in record["source_links"])
