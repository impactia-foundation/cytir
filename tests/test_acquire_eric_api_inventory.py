import hashlib
import json
from pathlib import Path
import urllib.parse

import cytir.sources.eric.inventory as eric


def _write(path: Path, value: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {"uri": "artifact://" + path.relative_to(path.parents[1]).as_posix(),
            "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def test_finalize_conserves_and_seals_exact_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(eric.SS, "seal", lambda **kw: {"records": kw["records"], "logical_sha256": kw["logical_sha256"]})
    root = tmp_path
    query, fields, rows = "e_fulltextauth:1", "id,e_fulltextauth", 2
    for start, ids in ((0, ["ED1", "ED2"]), (2, ["ED3"])):
        prefix = root / "pages" / f"{start:09d}"
        body = json.dumps({"response": {"numFound": 3, "start": start, "numFoundExact": True,
                                        "docs": [{"id": value, "e_fulltextauth": 1} for value in ids]}}).encode()
        artifacts = {}
        for label, suffix, value in (
                ("body", "body.json", body), ("response_headers", "headers", b"HTTP/2 200\n"),
                ("curl_metrics", "curl.json", b"{}\n"),
                ("transport_metadata", "metadata.json", b"")):
            path = Path(str(prefix) + "." + suffix)
            if label == "transport_metadata":
                expected_url = eric.requested_url(query, fields, rows, start)
                value = json.dumps({"http_status": 200, "requested_url": expected_url,
                                    "final_url": expected_url, "fetched_at": "2026-08-15T00:00:00Z",
                                    "body": {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()},
                                    "response_headers": {"bytes": len(b"HTTP/2 200\n"),
                                                         "sha256": hashlib.sha256(b"HTTP/2 200\n").hexdigest()}}).encode()
            artifacts[label] = _write(path, value)
        receipt = {"schema": "aw.eric-api-page-receipt.v1", "start": start, "rows": rows,
                   "search": query, "fields": fields, "num_found": 3,
                   "document_ids": ids, "artifacts": artifacts}
        Path(str(prefix) + ".receipt.json").write_text(json.dumps(receipt))
    run = eric.finalize(root=root, query=query, fields=fields, rows=rows)
    assert run["counts"] == {"api_pages": 2, "catalog_records": 3, "unique_document_ids": 3}
    assert run["content_scope"]["excludes"] == ["PDF bytes", "PDF principal text", "production point projection"]


def test_finalize_rejects_duplicate_ids_across_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(eric.SS, "seal", lambda **kw: {})
    query, fields, rows = "e_fulltextauth:1", "id,e_fulltextauth", 1
    for start in (0, 1):
        prefix = tmp_path / "pages" / f"{start:09d}"
        body = json.dumps({"response": {"numFound": 2, "start": start, "numFoundExact": True,
                                        "docs": [{"id": "ED1", "e_fulltextauth": 1}]}}).encode()
        artifacts = {}
        for label, suffix, value in (
                ("body", "body.json", body), ("response_headers", "headers", b"h"),
                ("curl_metrics", "curl.json", b"{}"),
                ("transport_metadata", "metadata.json", b"")):
            if label == "transport_metadata":
                expected_url = eric.requested_url(query, fields, rows, start)
                value = json.dumps({"http_status": 200, "requested_url": expected_url,
                                    "final_url": expected_url, "fetched_at": "2026-08-15T00:00:00Z",
                                    "body": {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()},
                                    "response_headers": {"bytes": 1, "sha256": hashlib.sha256(b"h").hexdigest()}}).encode()
            artifacts[label] = _write(Path(str(prefix) + "." + suffix), value)
        Path(str(prefix) + ".receipt.json").write_text(json.dumps(
            {"schema": "aw.eric-api-page-receipt.v1", "start": start, "rows": rows,
             "search": query, "fields": fields, "num_found": 2,
             "document_ids": ["ED1"], "artifacts": artifacts}))
    try:
        eric.finalize(root=tmp_path, query=query, fields=fields, rows=rows)
    except eric.EricInventoryError as exc:
        assert "cross-page duplicate" in str(exc)
    else:
        raise AssertionError("duplicate IDs accepted")


def test_shim_is_the_package_module():
    from cytir.sources.eric import inventory
    assert eric is inventory


def _fixture_and_adapter_run(tmp_path):
    """Copy the real fixture and seal it through the family adapter. Returns (fixture, new_root, adapter_ctx)."""
    import shutil
    from cytir.sources import contract as contract_mod
    from cytir.sources.families.eric_official_pdf import EricOfficialPdfFamily
    from cytir.sources.family import RunContext

    fixture = Path(__file__).resolve().parents[1] / "data" / "samples" / "eric"
    new = tmp_path / "new"
    shutil.copytree(fixture, new)
    ctx = RunContext(out_dir=tmp_path / "out", today="2026-10-28", options={"root": str(new)},
                     contract=contract_mod.by_id(contract_mod.load())["eric_official_pdf"])
    EricOfficialPdfFamily().seal_inventory(ctx)
    return fixture, new, ctx


def test_adapter_seals_real_fixture_to_frozen_golden_bytes(tmp_path):
    """Independent oracle (task 1386 review): the family adapter's RUN.json and inventory-pages.jsonl are
    byte-identical to golden/ files frozen from the PRE-MOVE script at origin/main 3c29bf94f."""
    fixture, new, _ = _fixture_and_adapter_run(tmp_path)
    golden = fixture / "golden"
    assert (new / "RUN.json").read_bytes() == (golden / "RUN.json").read_bytes()
    assert (new / "inventory-pages.jsonl").read_bytes() == (golden / "inventory-pages.jsonl").read_bytes()


def test_shim_and_adapter_forward_to_the_same_outputs(tmp_path):
    """Forwarding check only (task 1386 review): the old script path (the shim) and the family adapter both
    execute the same package module, so this proves the shim forwards, not that the output is correct.
    The correctness oracle is test_adapter_seals_real_fixture_to_frozen_golden_bytes."""
    import shutil
    import subprocess
    import sys

    fixture = Path(__file__).resolve().parents[1] / "data" / "samples" / "eric"
    first = json.loads((fixture / "pages" / "000000000.receipt.json").read_text())
    old = tmp_path / "old"
    shutil.copytree(fixture, old)
    subprocess.run([sys.executable, "-m", "cytir.sources.eric.inventory", "--root", str(old), "--query", first["search"],
                    "--fields", first["fields"], "--rows", str(first["rows"])],
                   check=True, capture_output=True)
    _, new, _ = _fixture_and_adapter_run(tmp_path / "adapter")
    assert (old / "RUN.json").read_bytes() == (new / "RUN.json").read_bytes()
    assert (old / "inventory-pages.jsonl").read_bytes() == (new / "inventory-pages.jsonl").read_bytes()
    old_r = json.loads((old / "stages/original/receipt.json").read_text())
    new_r = json.loads((new / "stages/original/receipt.json").read_text())
    assert old_r["logical_sha256"] == new_r["logical_sha256"] and old_r["records"] == new_r["records"]
