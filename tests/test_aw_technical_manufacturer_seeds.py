import json
from pathlib import Path

import cytir.sources.technical_manufacturer.seeds as subject


def test_query_gap_seed_derivation_is_domain_agnostic_and_canonical(tmp_path: Path):
    source = tmp_path / "queries.jsonl"
    source.write_text("\n".join((
        json.dumps({"id": "q1", "query": "robotics kit for raspberry pi",
                    "preferred_urls": [
                        "http://WWW.Example.com/products/x?utm_source=a&srsltid=gone&v=1",
                        "https://maker.test/docs/robot"]}),
        json.dumps({"id": "q2", "query": "restaurant", "preferred_urls": [
            "https://ignored.test/"]}),
    )) + "\n")
    result = subject.derive(source=source, frontier=tmp_path / "frontier.jsonl",
                            receipt=tmp_path / "receipt.json")
    rows = [json.loads(line) for line in (tmp_path / "frontier.jsonl").open()]
    assert result["frontier"]["domains"] == 2
    assert {row["domain"] for row in rows} == {"example.com", "maker.test"}
    assert rows[0]["seed_url"] == "https://example.com/products/x?v=1"
