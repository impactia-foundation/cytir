import gzip
import hashlib
import json
from pathlib import Path

import cytir.sources.kali.combine_production_markdown as subject


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _gzip(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as sink:
        for row in rows:
            sink.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    body = path.read_bytes()
    return {"uri": "artifact://" + path.as_posix().split("/artifacts/", 1)[1],
            "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def _binding(path: Path) -> dict:
    body = path.read_bytes()
    return {"uri": "artifact://" + path.as_posix().split("/artifacts/", 1)[1],
            "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def _parent(identifier: str) -> dict:
    text = "# Law\n\n## Article identity\n\n## Article {#" + identifier + "}"
    return {"source_record_id": identifier, "profile": "anchored_law",
            "markdown": text, "markdown_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source_version": {"status": "in force"},
            "source_archive": {"member": f"{identifier}.xml", "sha256": "a" * 64},
            "url": f"https://example.test/{identifier}"}


def _link(row: dict) -> dict:
    return {"source_record_id": row["source_record_id"], "url": row["url"],
            "parent_markdown_sha256": row["markdown_sha256"],
            "source_version": row["source_version"], "parent_identity": row["source_archive"],
            "links": [], "unresolved_relations": []}


def test_combines_only_complete_increment_recovery(tmp_path):
    root = tmp_path / "artifacts"; base = root / "base"; recovery = root / "recovery"
    one, two = _parent("KALIARTI1"), _parent("KALIARTI2")
    bindings = {
        "base_markdown": _gzip(base / "markdown.jsonl.gz", [one]),
        "base_outlinks": _gzip(base / "outlinks.jsonl.gz", [_link(one)]),
        "recovery_markdown": _gzip(recovery / "markdown-supplement.jsonl.gz", [two]),
        "recovery_outlinks": _gzip(recovery / "outlinks-supplement.jsonl.gz", [_link(two)]),
    }
    for path, binding in (
            (base / "MARKDOWN-RECEIPT.json", bindings["base_markdown"]),
            (base / "OUTLINKS-RECEIPT.json", bindings["base_outlinks"]),
            (recovery / "MARKDOWN-RECEIPT.json", bindings["recovery_markdown"]),
            (recovery / "OUTLINKS-RECEIPT.json", bindings["recovery_outlinks"])):
        _json(path, {"artifacts": [binding]})
    archive = root / "originals/global.tar.gz"; archive.parent.mkdir(parents=True)
    archive.write_bytes(b"global")
    l0_receipt = root / "originals/global-receipt.json"; _json(l0_receipt, {"ok": True})
    increment = root / "originals/increment.tar.gz"; increment.write_bytes(b"increment")
    increment_receipt = root / "originals/increment-receipt.json"
    _json(increment_receipt, {"records": 1, "artifacts": [_binding(increment)]})
    _json(base / "RUN.json", {"status": "recovery-frontier",
                              "counts": {"markdown_documents": 1},
                              "l0": {**_binding(l0_receipt), "archive": _binding(archive)}})
    _json(recovery / "RUN.json", {"status": "complete",
                                  "counts": {"recovered_source_ids": 1,
                                             "unresolved_source_ids": 0},
                                  "increments": _binding(increment_receipt)})
    result = subject.combine(artifact_root=root, base_root=base,
                             recovery_root=recovery, output=root / "combined", expected=2)
    assert result["counts"] == {"production_parents": 2, "source_links": 0,
                                "unresolved_relations": 0,
                                "explicit_empty_outlink_parents": 2}
    assert result["status"] == "complete"


def test_refuses_unresolved_increment_frontier(tmp_path):
    base = tmp_path / "base"; recovery = tmp_path / "recovery"
    _json(base / "RUN.json", {"status": "recovery-frontier",
                              "counts": {"markdown_documents": 1}})
    _json(recovery / "RUN.json", {"status": "recovery-frontier",
                                  "counts": {"recovered_source_ids": 0,
                                             "unresolved_source_ids": 1}})
    import pytest
    with pytest.raises(ValueError, match="not complete"):
        subject.combine(artifact_root=tmp_path, base_root=base,
                        recovery_root=recovery, output=tmp_path / "combined", expected=2)
