import json
from pathlib import Path

import cytir.sources.routec.direct_preserved_html as subject


def test_frontier_requires_exact_receipt_and_unique_urls(tmp_path: Path):
    frontier = tmp_path / "frontier.jsonl"
    frontier.write_text(json.dumps({"url": "https://example.org/a",
                                    "retained_points": 2}) + "\n")
    receipt = tmp_path / "RECEIPT.json"
    receipt.write_text(json.dumps({
        "schema": "aw.exact-url-frontier-receipt.v1",
        "result": {"frontier": {"urls": 1,
                                  "sha256": subject.sha256_file(frontier)}}}))
    rows, _ = subject._frontier(frontier, receipt)
    assert rows["https://example.org/a"]["retained_points"] == 2
    frontier.write_text(frontier.read_text() + frontier.read_text())
    receipt.write_text(json.dumps({
        "schema": "aw.exact-url-frontier-receipt.v1",
        "result": {"frontier": {"urls": 2,
                                  "sha256": subject.sha256_file(frontier)}}}))
    try:
        subject._frontier(frontier, receipt)
    except ValueError as exc:
        assert "duplicate URL" in str(exc)
    else:
        raise AssertionError("duplicate direct frontier URL accepted")


def test_content_type_is_read_from_preserved_header_artifact(tmp_path: Path):
    headers = json.dumps([["Content-Type", "text/html; charset=utf-8"]]).encode()
    stored = subject.PS.store_artifact(tmp_path, "headers", "json", headers)
    receipt = {"artifacts": {"response_headers": stored}}
    assert subject._content_type(tmp_path, receipt) == "text/html"
