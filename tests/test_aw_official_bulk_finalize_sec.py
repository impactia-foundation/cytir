import json
from pathlib import Path
import zipfile

import cytir.sources.official_bulk.finalize_sec as F
import cytir.sources.official_bulk.http_evidence as record_http_fetch


def _transport(body: Path, headers: Path, url: str):
    metrics=body.with_suffix(body.suffix+".curl.json")
    metrics.write_text(json.dumps({"http_code":200,"url_effective":url}))
    out=body.with_suffix(body.suffix+".transport.json")
    record_http_fetch.record(body=body,headers=headers,curl_metrics=metrics,
                             requested_url=url,fetched_at="2026-08-15T00:00:00Z",output=out)
    return metrics,out


def test_version_pinned_resumed_sec_zip_is_preserved(tmp_path: Path):
    root=tmp_path/"root"; input_dir=root/"input"; attempts=input_dir/"attempts"
    attempts.mkdir(parents=True)
    partial=input_dir/"submissions.zip.partial"
    with zipfile.ZipFile(partial,"w") as z:
        z.writestr("CIK0000000001.json",'{}'); z.writestr("CIK0000000002.json",'{}')
        z.writestr("CIK0000000001-submissions-001.json",'{}')
        z.writestr("placeholder.txt",b"Placeholder file")
    size=partial.stat().st_size; etag='"fixture-etag"'; version="fixture-version"
    head_body=input_dir/"head.body"; head_body.write_bytes(b"")
    head_headers=input_dir/"head.headers"
    head_headers.write_text("HTTP/2 200\r\ncontent-length: %s\r\ncontent-type: application/zip\r\naccept-ranges: bytes\r\netag: %s\r\nx-amz-version-id: %s\r\nlast-modified: now\r\n\r\n"%(size,etag,version),encoding="iso-8859-1")
    head_metrics,head_transport=_transport(head_body,head_headers,F.URL)
    docs_body=input_dir/"docs.html"; docs_body.write_text("official bulk ZIP updated nightly")
    docs_headers=input_dir/"docs.headers"; docs_headers.write_text("HTTP/2 200\r\ncontent-type: text/html\r\n\r\n")
    docs_metrics,docs_transport=_transport(docs_body,docs_headers,F.DOC_URL)
    prefix=attempts/"attempt-0001"
    prefix.with_suffix(".started").write_text("2026-08-15T00:00:01Z\n")
    prefix.with_suffix(".completed").write_text("2026-08-15T00:00:02Z\n")
    prefix.with_suffix(".rc").write_text("0\n"); prefix.with_suffix(".start_offset").write_text("0\n")
    prefix.with_suffix(".end_offset").write_text(f"{size}\n")
    prefix.with_suffix(".headers").write_text(
        f"HTTP/2 200\r\netag: {etag}\r\nx-amz-version-id: {version}\r\n\r\n",encoding="iso-8859-1")
    prefix.with_suffix(".metrics").write_text(json.dumps({"http_code":200,"url_effective":F.URL}))
    body=input_dir/"submissions.zip"; headers=input_dir/"submissions.zip.headers"
    metadata=input_dir/"submissions.zip.transport.json"
    result=F.finalize(partial=partial,body=body,headers=headers,metadata=metadata,
        attempts_dir=attempts,head_body=head_body,head_headers=head_headers,
        head_metrics=head_metrics,head_transport=head_transport,docs_body=docs_body,
        docs_headers=docs_headers,docs_metrics=docs_metrics,docs_transport=docs_transport)
    assert result["body"]["bytes"]==size and result["source_identity"]["zip_members"]==4
    assert result["source_identity"]["primary_entity_members"]==2
    assert result["source_identity"]["historical_submission_members"]==1
    assert result["source_identity"]["x_amz_version_id"]==version
    assert body.is_file() and not partial.exists() and len(result["evidence_artifacts"])==15
