import gzip
import json
from pathlib import Path

import cytir.sources.context7.materialize as subject
from cytir.acquire import preserved_stage_io as PS


def test_materializes_commit_path_markdown_links_and_explicit_recovery(
        tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "aw-corpus"
    source = artifact_root / "source"
    source.mkdir(parents=True)
    raw = b"# Guide\n\nSee [API](https://example.test/api).\n"
    body = PS.store_artifact(source, "original/blobs", "raw", raw)
    transport = PS.store_artifact(source, "transport", "json", b"{}\n")
    receipt = {
        "schema": "aw.context7-exact-path-repository-receipt.v1",
        "status": "terminal", "owner_repo": "o/r", "requested_ref": "main",
        "resolved_revision": "a" * 40,
        "documents": [
            {"source_path": "docs/a.md", "status": "complete",
             "git_blob_sha1": "b" * 40,
             "artifacts": {"response_body": {
                 "uri": "artifact://source/" + body["uri"].removeprefix("artifact://"),
                 "bytes": body["bytes"], "sha256": body["sha256"],
                 "body_bytes": body["body_bytes"],
                 "body_sha256": body["body_sha256"],
                 "compression": body["compression"]},
                 "transport": {
                 "uri": "artifact://source/" + transport["uri"].removeprefix("artifact://"),
                 "bytes": transport["bytes"], "sha256": transport["sha256"]}}},
            {"source_path": "docs/missing.md", "status": "quarantine",
             "reason": "path_absent_from_preserved_tree", "artifacts": {}},
        ],
    }
    receipt_path = source / "receipts/o--r.json"
    receipt_path.parent.mkdir()
    receipt_path.write_text(json.dumps(receipt))
    run_path = source / "RUN.json"
    run_path.write_text(json.dumps({
        "schema": "aw.context7-exact-path-acquisition-run.v1",
        "status": "complete",
        "counts": {"tranche_repository_ref_pairs": 1,
                   "document_statuses": {"complete": 1, "quarantine": 1}},
        "stages": {"original": {"receipt": {"uri": "artifact://fake"}}},
    }))

    monkeypatch.setattr(subject, "_validate_stage", lambda *a, **k: None)
    monkeypatch.setattr(subject, "_upstream_evidence", lambda **kwargs: (
        json.loads(run_path.read_text()), [run_path, receipt_path]))
    monkeypatch.setattr(subject.SS, "seal", lambda **kwargs: {
        "schema": "aw.stage-receipt.v1", "records": kwargs["records"],
        "logical_sha256": kwargs["logical_sha256"],
        "receipt": {"uri": "artifact://fake"},
    })
    result = subject.materialize(
        source_root=source, source_run=run_path,
        output_root=artifact_root / "derived", artifact_root=artifact_root,
        expected_complete=1, expected_recovery=1)
    assert result["counts"]["markdown_documents"] == 1
    assert result["counts"]["recovery_documents"] == 1
    markdown_artifact = next((artifact_root / "derived/stages/markdown/data").rglob("*.gz"))
    markdown_row = json.loads(gzip.decompress(markdown_artifact.read_bytes()))
    # task 517: the provenance is record metadata and the body starts at the
    # document. v1 asserted `aw_source_path: "docs/a.md"` INSIDE the markdown.
    assert not markdown_row["markdown"].startswith("---\naw_")
    assert markdown_row["official_source_identity"] == {
        "repository": markdown_row["repository"],
        "revision": markdown_row["revision"],
        "source_path": "docs/a.md",
        "parent_sha256": markdown_row["parent_original_sha256"]}
    assert markdown_row["title"] == "Guide"
    assert markdown_row["markdown"].startswith("# Guide")
    outlinks_artifact = next((artifact_root / "derived/stages/outlinks/data").rglob("*.gz"))
    outlink = json.loads(gzip.decompress(outlinks_artifact.read_bytes()))
    assert outlink["target_url"] == "https://example.test/api"


def test_portable_stage_uses_global_artifact_root(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "aw-corpus"
    output = artifact_root / "derived"
    output.mkdir(parents=True)
    observed = {}
    monkeypatch.setattr(subject.SS, "seal", lambda **kwargs: observed.update(kwargs) or {})
    subject._stage(artifact_root=artifact_root, output_root=output,
                   name="markdown", rows=[{"x": 1}])
    assert observed["artifact_root"] == artifact_root
    assert observed["artifact_paths"][0].is_relative_to(artifact_root)


def test_missing_git_source_schema_is_an_admitted_source_class():
    assert "aw.context7-missing-git-source-run.v1" in subject.RUN_SCHEMAS
    assert "aw.context7-missing-git-source-receipt.v1" in subject.RECEIPT_SCHEMAS


def test_repository_pair_count_supports_both_source_run_classes():
    exact = {"tranche_repository_ref_pairs": 36}
    missing = {"repository_ref_pairs": 25}
    assert subject._repository_pair_count(exact) == 36
    assert subject._repository_pair_count(missing) == 25
