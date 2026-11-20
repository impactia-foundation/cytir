import json
from pathlib import Path
import zipfile

import cytir.sources.service_public.dump as subject
from cytir.index.preservation_contract import _validate_stage
import cytir.sources.official_bulk.http_evidence as record_http_fetch


def test_preserves_dump_markdown_outlinks_and_quarantine(tmp_path: Path):
    archive = tmp_path / "dump.zip"
    good = b'''<Publication xmlns:dc="http://purl.org/dc/elements/1.1/"
      spUrl="https://www.service-public.gouv.fr/particuliers/vosdroits/F1">
      <dc:title>Une fiche utile</dc:title><Introduction><Paragraphe>Texte utile
      <Lien href="https://example.fr/a"/></Paragraphe></Introduction></Publication>'''
    empty = b'''<Publication xmlns:dc="http://purl.org/dc/elements/1.1/"
      spUrl="https://www.service-public.gouv.fr/particuliers/vosdroits/F2">
      <dc:title>Vide</dc:title></Publication>'''
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("part/F1.xml", good)
        target.writestr("part/F2.xml", empty)
        target.writestr("part/N1.xml", b"<Navigation/>")
    headers = tmp_path / "headers"
    headers.write_bytes(b"HTTP/2 302\r\n\r\nHTTP/2 200\r\ncontent-type: application/zip\r\n\r\n")
    metrics = tmp_path / "metrics"
    requested = "https://data.example/stable"
    metrics.write_text(json.dumps({"http_code": 200,
                                   "url_effective": "https://source.example/dump.zip"}))
    metadata = tmp_path / "metadata"
    record_http_fetch.record(
        body=archive, headers=headers, curl_metrics=metrics,
        requested_url=requested, fetched_at="2026-08-15T00:00:00Z",
        output=metadata)
    root = tmp_path / "root"
    run = subject.materialize(
        archive=archive, headers=headers, metadata=metadata, root=root,
        requested_url=requested, fetch_vantage="fixture")
    assert run["counts"] == {"markdown_documents": 1, "outlink_records": 1,
                             "quarantined_members": 1}
    assert run["attribution"]["download_filename"] == "dump.zip"
    for name in ("original", "markdown", "outlinks", "quarantine"):
        value = _validate_stage(run["stages"][name], f"service_public.{name}",
                                artifact_root=root, deep=True)
        assert value["artifacts"]


def test_rejects_zip_path_traversal(tmp_path: Path):
    archive = tmp_path / "dump.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("../F1.xml", b"<Publication/>")
    headers = tmp_path / "headers"
    headers.write_bytes(b"HTTP/2 200\r\n\r\n")
    metrics = tmp_path / "metrics"
    requested = "https://data.example/stable"
    metrics.write_text(json.dumps({"http_code": 200,
                                   "url_effective": "https://data.example/final"}))
    metadata = tmp_path / "metadata"
    record_http_fetch.record(body=archive, headers=headers, curl_metrics=metrics,
                             requested_url=requested,
                             fetched_at="2026-08-15T00:00:00Z", output=metadata)
    try:
        subject.materialize(archive=archive, headers=headers, metadata=metadata,
                            root=tmp_path / "root", requested_url=requested,
                            fetch_vantage="fixture")
    except subject.ServicePublicError as exc:
        assert "unsafe archive member" in str(exc)
    else:
        raise AssertionError("unsafe member accepted")
