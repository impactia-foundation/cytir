import json

from cytir.news.enrich_tier1 import enrich_file, hint_to_country


def test_hint_to_country_maps_eu_and_rejects_unknown_values():
    assert hint_to_country("EU") == "INTL"
    assert hint_to_country(" ch ") == "CH"
    assert hint_to_country("world") is None


def test_enrich_file_uses_principal_country_logic_and_metadata(tmp_path):
    source = tmp_path / "input.jsonl"
    metadata = tmp_path / "meta.json"
    output = tmp_path / "output.jsonl"
    source.write_text(
        json.dumps(
            {
                "domain": "srf.ch",
                "source_type": "news",
                "title": "A title",
                "text": "Full text",
            }
        )
        + "\n"
    )
    metadata.write_text(
        json.dumps(
            {
                "srf.ch": {
                    "country_hint": "CH",
                    "license": "publisher terms",
                    "license_citation": "sources/ch/ch-media.yaml license.default "
                    "(task 448) -- an Act of the pack's own ratified admission.",
                }
            }
        )
    )

    counts = enrich_file(source, metadata, output)
    row = json.loads(output.read_text())

    assert counts == {"rows": 1, "country": 1, "license": 1}
    assert row["country"] == "CH"
    assert row["license"] == "publisher terms"
    assert row["anchor"] == "A title"
    assert row["snippet"] == ""


def test_enrich_file_drops_a_license_with_no_citation(tmp_path):
    """Task 638: a host_meta `license` with no `license_citation` is a
    topic-class guess (or any other uncited value), not a declared grant --
    it must never reach the record. This is the exact shape of the
    canada.ca / Tier-1-seed defect (task 638 licence-stamp audit)."""
    source = tmp_path / "input.jsonl"
    metadata = tmp_path / "meta.json"
    output = tmp_path / "output.jsonl"
    source.write_text(
        json.dumps({"domain": "example.gov", "source_type": "reference",
                    "title": "t"}) + "\n"
    )
    metadata.write_text(
        json.dumps({"example.gov": {"license": "gov-open",
                                    "license_note": "topic class: gov"}})
    )

    counts = enrich_file(source, metadata, output)
    row = json.loads(output.read_text())

    assert counts["license"] == 0
    assert "license" not in row or not row["license"]
