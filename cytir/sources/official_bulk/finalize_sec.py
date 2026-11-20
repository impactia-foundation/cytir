"""Fail-closed finalizer for the version-pinned SEC submissions bulk ZIP."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

from cytir.acquire.artifact_io import atomic_publish, sha256_file
from cytir.sources.official_bulk.http_evidence import redact_url
from cytir.sources.official_bulk.sec_archive import validate_submission_archive


VERSION = "sec-edgar-submissions-fetch.v2"
URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
DOC_URL = "https://www.sec.gov/search-filings/edgar-application-programming-interfaces"


def _headers(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="iso-8859-1").splitlines():
        if line.startswith("HTTP/"):
            values = {":status": line}
        elif ":" in line:
            key, value = line.split(":", 1); values[key.strip().casefold()] = value.strip()
    return values


def _publish(source: Path, destination: Path) -> None:
    with source.open("rb") as stream: os.fsync(stream.fileno())
    os.replace(source, destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def _transport(value_path: Path, body: Path, headers: Path, url: str) -> dict:
    value = json.loads(value_path.read_text())
    if (value.get("schema") != "aw.http-fetch-metadata.v1" or
            value.get("requested_url") != url or value.get("http_status") != 200):
        raise ValueError("SEC HEAD/document transport identity failed")
    for key, path in (("body", body), ("response_headers", headers)):
        declared = value.get(key) or {}
        if declared.get("bytes") != path.stat().st_size or declared.get("sha256") != sha256_file(path):
            raise ValueError("SEC HEAD/document transport artifact mismatch")
    return value


def finalize(*, partial: Path, body: Path, headers: Path, metadata: Path,
             attempts_dir: Path, head_body: Path, head_headers: Path,
             head_metrics: Path, head_transport: Path, docs_body: Path,
             docs_headers: Path, docs_metrics: Path, docs_transport: Path) -> dict:
    _transport(head_transport, head_body, head_headers, URL)
    _transport(docs_transport, docs_body, docs_headers, DOC_URL)
    identity = _headers(head_headers)
    expected_bytes = int(identity.get("content-length") or 0)
    etag = identity.get("etag", "")
    version_id = identity.get("x-amz-version-id", "")
    last_modified = identity.get("last-modified", "")
    if (expected_bytes <= 0 or not etag or not version_id or not last_modified or
            identity.get("accept-ranges", "").casefold() != "bytes" or
            identity.get("content-type", "").casefold() != "application/zip"):
        raise ValueError("SEC HEAD response lacks pinned resumable source identity")
    source = body if body.exists() else partial
    if not source.is_file() or source.stat().st_size != expected_bytes:
        raise ValueError("SEC submissions body has wrong final byte count")
    evidence = [head_body, head_headers, head_metrics, head_transport,
                docs_body, docs_headers, docs_metrics, docs_transport]
    attempts=[]; previous=None
    for started in sorted(attempts_dir.glob("attempt-*.started")):
        prefix=started.with_suffix("")
        paths={suffix:Path(str(prefix)+suffix) for suffix in
               (".started",".completed",".rc",".start_offset",".end_offset",".headers",".metrics")}
        if not all(path.is_file() for path in paths.values()):
            raise ValueError("incomplete SEC range-attempt evidence")
        evidence.extend(paths.values()); metrics=json.loads(paths[".metrics"].read_text() or "{}")
        row={"attempt":prefix.name,"started_at":started.read_text().strip(),
             "completed_at":paths[".completed"].read_text().strip(),
             "return_code":int(paths[".rc"].read_text()),
             "start_offset":int(paths[".start_offset"].read_text()),
             "end_offset":int(paths[".end_offset"].read_text()),
             "http_status":int(metrics.get("http_code") or 0),
             "final_url":metrics.get("url_effective"),
             "headers":_headers(paths[".headers"])}
        row["status"]="complete" if row["return_code"]==0 else "failed_resumable"
        if (row["start_offset"]<0 or row["end_offset"]<row["start_offset"] or
                row["end_offset"]>expected_bytes or
                (previous is not None and row["start_offset"] != previous)):
            raise ValueError("SEC resumable offsets are not conservative")
        previous=row["end_offset"]; attempts.append(row)
    if not attempts or attempts[-1]["status"]!="complete":
        raise ValueError("SEC transfer has no terminal successful attempt")
    terminal=attempts[-1]
    if (terminal["end_offset"]!=expected_bytes or terminal["http_status"] not in (200,206)
            or (terminal["start_offset"]>0 and terminal["http_status"]!=206)
            or terminal["final_url"]!=URL or terminal["headers"].get("etag")!=etag
            or terminal["headers"].get("x-amz-version-id")!=version_id):
        raise ValueError("SEC terminal response differs from pinned HEAD identity")
    terminal_headers=Path(str(attempts_dir/terminal["attempt"])+".headers")
    with zipfile.ZipFile(source) as archive:
        members=validate_submission_archive(archive)
        bad=archive.testzip()
    if bad is not None:
        raise ValueError(f"SEC submissions ZIP CRC failed: {bad}")
    artifact_root=metadata.parents[1].resolve()
    evidence_rows=[]
    for path in evidence:
        relative=path.resolve().relative_to(artifact_root)
        evidence_rows.append({"uri":f"artifact://{relative.as_posix()}",
                              "bytes":path.stat().st_size,"sha256":sha256_file(path)})
    value={"schema":"aw.resumable-http-fetch-metadata.v1","version":VERSION,
           "requested_url":redact_url(URL),"final_url":URL,"http_status":200,
           "fetch_mode":"range-resumed-if-match-version-pinned",
           "body":{"bytes":expected_bytes,"sha256":sha256_file(source)},
           "response_headers":{"bytes":terminal_headers.stat().st_size,
                               "sha256":sha256_file(terminal_headers)},
           "source_identity":{"etag":etag,"x_amz_version_id":version_id,
                              "last_modified":last_modified,"zip_members":members.total,
                              "primary_entity_members":len(members.primary),
                              "historical_submission_members":len(members.historical),
                              "placeholder_members":len(members.placeholders),
                              "head_response_sha256":sha256_file(head_headers),
                              "official_api_documentation_sha256":sha256_file(docs_body)},
           "attempts":[{k:v for k,v in row.items() if k!="headers"} for row in attempts],
           "evidence_artifacts":evidence_rows}
    encoded=(json.dumps(value,indent=2,sort_keys=True)+"\n").encode()
    if headers.exists() and sha256_file(headers)!=sha256_file(terminal_headers):
        raise ValueError("immutable SEC final headers collision")
    if not headers.exists(): atomic_publish(headers,terminal_headers.read_bytes())
    if not body.exists(): _publish(partial,body)
    if metadata.exists() and sha256_file(metadata)!=hashlib.sha256(encoded).hexdigest():
        raise ValueError("immutable SEC transport collision")
    if not metadata.exists(): atomic_publish(metadata,encoded)
    return value


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("partial","body","headers","metadata","attempts-dir","head-body",
                 "head-headers","head-metrics","head-transport","docs-body","docs-headers",
                 "docs-metrics","docs-transport"):
        p.add_argument(f"--{name}",type=Path,required=True)
    a=p.parse_args(argv); kwargs={name.replace("-","_"):getattr(a,name.replace("-","_")).resolve()
                              for name in ("partial","body","headers","metadata","attempts-dir",
                              "head-body","head-headers","head-metrics","head-transport","docs-body",
                              "docs-headers","docs-metrics","docs-transport")}
    print(json.dumps(finalize(**kwargs),indent=2,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
