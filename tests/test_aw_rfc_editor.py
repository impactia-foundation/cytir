import gzip
import hashlib
import json
from cytir.extract import rfc_editor as subject

def test_production_selection_uses_source_native_numbers_and_global_uris(
        tmp_path, monkeypatch):
    artifact_root = tmp_path / "aw-corpus"
    source_root = artifact_root / "_work/source"
    (source_root / "input/rfcs").mkdir(parents=True)
    rows = []
    for number in (1, 2):
        source = source_root / f"input/rfcs/rfc{number}.txt"
        source.write_text(f"RFC {number}\n", encoding="utf-8")
        rows.append({
            "document_id": f"rfc-editor:rfc:{number}",
            "source_relative_path": f"input/rfcs/rfc{number}.txt",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        })
    links = [
        {"document_id": "rfc-editor:rfc:1", "target_url": "https://example/1"},
        {"document_id": "rfc-editor:rfc:2", "target_url": "https://example/2"},
    ]
    heading_run = source_root / "RUN.json"
    heading_run.write_text(json.dumps({
        "stages": {"markdown": {"fixture": "markdown"},
                   "outlinks": {"fixture": "outlinks"}},
    }), encoding="utf-8")
    membership = source_root / "membership.json"
    membership.write_text(json.dumps({
        "rfc_numbers": [1], "counts": {"distinct_rfc_numbers": 1},
    }), encoding="utf-8")

    monkeypatch.setattr(
        subject, "_read_stage_rows",
        lambda _root, stage: rows if stage["fixture"] == "markdown" else links)
    output_run = artifact_root / "_work/task417/rfc-production/RUN.json"
    result = subject.net_new(
        root=source_root, heading_run=heading_run, membership_path=membership,
        output_run=output_run, selection="production", artifact_root=artifact_root)

    assert result["schema"] == "aw.rfc-production-parent-run.v1"
    assert result["counts"]["production_documents"] == 1
    assert result["counts"]["production_outlinks"] == 1
    membership_uri = result["membership"]["uri"]
    membership_copy = artifact_root / membership_uri.removeprefix("artifact://")
    assert membership_copy.is_file()
    assert hashlib.sha256(membership_copy.read_bytes()).hexdigest() == result[
        "membership"]["sha256"]
    assert membership_uri.startswith(
        "artifact://_work/task417/rfc-production/membership/")
    receipt_uri = result["stages"]["markdown"]["receipt"]["uri"]
    assert receipt_uri.startswith("artifact://_work/task417/rfc-production/")
    receipt = json.loads(
        (artifact_root / receipt_uri.removeprefix("artifact://")).read_text())
    data_path = artifact_root / receipt["artifacts"][0]["uri"].removeprefix(
        "artifact://")
    with gzip.open(data_path, "rt", encoding="utf-8") as stream:
        selected = [json.loads(line) for line in stream]
    assert [row["document_id"] for row in selected] == ["rfc-editor:rfc:1"]


def test_rfc_heading_markdown_is_complete_and_reversible():
    text = "Lead\n\n1. Introduction\n\nEvery source byte remains.\n\nAPPENDIX A\n\nTail\n"
    markdown, headings = subject.render(
        9999, text, relative_path="input/rfcs/rfc9999.txt",
        source_sha256="a" * 64, encoding="utf-8")
    assert "## 1. Introduction" in markdown
    assert "## APPENDIX A" in markdown
    assert [row["line"] for row in headings] == [3, 7]
    assert "Every source byte remains." in markdown
