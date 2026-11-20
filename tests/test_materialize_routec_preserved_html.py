import gzip
import hashlib
import json
from pathlib import Path

from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS
import cytir.sources.routec.preserved_html as M


def _write(path: Path, value: dict | list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, list):
        path.write_text("".join(json.dumps(row) + "\n" for row in value))
    else:
        path.write_text(json.dumps(value) + "\n")


def _terminal(root: Path, artifact_root: Path, row: dict, plan_sha: str,
              *, html: str | None) -> Path:
    path = root / "receipts" / (hashlib.sha256(
        row["url"].encode()).hexdigest() + ".json")
    if html is None:
        value = {"schema": "aw.exact-url-cc-capture-receipt.v1",
                 "status": "cc_miss", "source_url": row["url"],
                 "plan_sha256": plan_sha, "artifacts": {}}
    else:
        family = root.relative_to(artifact_root).as_posix()
        artifacts = {
            "decoded_html": PS.store_artifact(
                artifact_root, family + "/original/decoded-html", "html",
                html.encode()),
            "raw_warc": PS.store_artifact(
                artifact_root, family + "/original/warc", "warc.gz", b"warc"),
            "transport": PS.store_artifact(
                artifact_root, family + "/original/transport", "json", b"{}\n"),
        }
        value = {"schema": "aw.exact-url-cc-capture-receipt.v1",
                 "status": "complete", "source_url": row["url"],
                 "plan_sha256": plan_sha, "artifacts": artifacts}
    _write(path, value)
    return path


def test_routes_known_source_and_materializes_only_generic_residual(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        M, "extract_content_markdown",
        lambda html, _url: ("# Guide\n\nComplete useful content.\n", "fixture"))
    artifact_root = tmp_path / "artifacts"
    root = artifact_root / "lane"
    cc_root = root / "cc"
    frontier = root / "frontier.jsonl"
    rows = [
        {"url": "https://laws.example/act/1", "retained_points": 5,
         "wave": "phase2_all"},
        {"url": "https://generic.example/page", "retained_points": 2,
         "wave": "phase2_all"},
        {"url": "https://generic.example/missing", "retained_points": 1,
         "wave": "phase2_all"},
    ]
    _write(frontier, rows)
    frontier_receipt = root / "frontier-receipt.json"
    _write(frontier_receipt, {
        "schema": "aw.exact-url-frontier-receipt.v1",
        "result": {"frontier": {"urls": 3,
            "sha256": M.sha256_file(frontier)}}})
    plan = cc_root / "plan/captures.jsonl"
    _write(plan, [{**row, "state": "CAPTURE" if index < 2 else "MISS"}
                  for index, row in enumerate(rows)])
    plan_sha = M.sha256_file(plan)
    terminals = [
        _terminal(cc_root, artifact_root, rows[0], plan_sha,
                  html="<html><title>Law</title><body>Official act text</body></html>"),
        _terminal(cc_root, artifact_root, rows[1], plan_sha,
                  html=("<html lang='en'><title>Generic title</title><body>"
                        "<main><h1>Guide</h1><p>Complete useful content.</p>"
                        "<a href='https://outside.example/evidence'>Evidence</a>"
                        "</main></body></html>")),
        _terminal(cc_root, artifact_root, rows[2], plan_sha, html=None),
    ]
    original_paths = [plan, *terminals]
    for terminal in terminals:
        receipt = json.loads(terminal.read_text())
        original_paths.extend(PS.artifact_path(artifact_root, artifact)
                              for artifact in receipt["artifacts"].values())
    original = SS.seal(
        artifact_root=artifact_root, artifact_paths=sorted(set(original_paths)),
        receipt_path=cc_root / "stages/original/receipt.json", records=2,
        fmt="fixture-original", logical_sha256="a" * 64,
        producer_id="fixture", producer_version="v1")
    cc_run_path = cc_root / "RUN.json"
    _write(cc_run_path, {
        "schema": "aw.exact-url-cc-preserved-run.v1", "status": "complete",
        "counts": {"frontier": 3, "cc_captures": 2, "cc_misses": 1},
        "stages": {"original": original}})
    registry = root / "routes.json"
    _write(registry, {
        "schema": "aw.routec-source-route-registry.v1",
        "identity_basis": "fixture exact URL routes",
        "routes": [{"route_id": "law", "hosts": ["laws.example"],
                    "path_prefixes": ["/"], "source_profile": "anchored_law",
                    "projection": "source-specific-pending",
                    "source_route": "official-law"}],
        "fallback": {"route_id": "generic", "source_profile": "web_html",
                     "projection": "generic-web-materialize",
                     "source_route": "cc-html"}})
    output = artifact_root / "materialized"
    result = M.materialize(
        artifact_root=artifact_root, frontier=frontier,
        frontier_receipt=frontier_receipt, cc_root=cc_root,
        cc_run_path=cc_run_path, route_registry_path=registry, output=output)
    assert result["counts"] == {
        "frontier_urls": 3, "production_points": 8,
        "cc_captures": 2, "cc_misses": 1,
        "generic_markdown_parents": 1, "generic_quarantine": 0,
        "capture_outlink_rows": 2,
        "outlink_rows": 1}
    with gzip.open(output / "markdown/generic-web.jsonl.gz", "rt") as source:
        parents = [json.loads(line) for line in source]
    assert [row["url"] for row in parents] == ["https://generic.example/page"]
    assert "Complete useful content" in parents[0]["markdown"]
    assert parents[0]["extraction"]["source_text_truncation_applied"] is False
    with gzip.open(output / "routes/all.jsonl.gz", "rt") as source:
        routes = [json.loads(line) for line in source]
    assert routes[0]["route_id"] == "law"
    assert routes[0]["projection"] == "source-specific-pending"
    assert result["classification"]["payload_grouping_used"] is False


def test_route_registry_rejects_ambiguous_url(tmp_path):
    registry = {"schema": "aw.routec-source-route-registry.v1",
        "routes": [
            {"route_id": "a", "hosts": ["same.example"],
             "path_prefixes": ["/"], "source_profile": "software_docs",
             "projection": "source-specific-pending", "source_route": "a"},
            {"route_id": "b", "hosts": ["same.example"],
             "path_prefixes": ["/x"], "source_profile": "software_docs",
             "projection": "source-specific-pending", "source_route": "b"}],
        "fallback": {"route_id": "g", "source_profile": "web_html",
             "projection": "generic-web-materialize", "source_route": "g"}}
    with __import__("pytest").raises(ValueError, match="ambiguous"):
        M.classify_url("https://same.example/x", registry)
