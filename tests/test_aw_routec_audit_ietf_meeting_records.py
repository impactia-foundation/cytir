import gzip
import hashlib
from pathlib import Path

import pytest

from cytir.extract.routec import ietf_meeting_records as subject
from cytir.sources.routec.audit_profile import validate_meeting_parent


def test_minutes_parent_gate_replays_full_source_text(tmp_path: Path) -> None:
    kind = "minutes"; url = "https://datatracker.ietf.org/doc/minutes-114-example"
    html = ("<html><head><title>Minutes IETF114: example</title></head><body><main>"
            "<h1>Minutes IETF114: example</h1><pre>Full minutes text.</pre></main></body></html>").encode()
    body = tmp_path/"body.gz"
    with body.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as target: target.write(html)
    compressed = body.read_bytes(); digest = hashlib.sha256(html).hexdigest()
    artifact = {"uri": "artifact://body.gz", "bytes": len(compressed),
        "sha256": hashlib.sha256(compressed).hexdigest(), "compression": "gzip",
        "body_bytes": len(html), "body_sha256": digest}
    title, markdown, version = subject.render_meeting_record(
        html.decode(), kind=kind, url=url, final_url=url+"/", body_sha256=digest)
    identity = {"terminal_receipt": {"sha256": "a"*64},
        "artifacts": {key: artifact for key in
            ("response_body", "response_headers", "transport", "consent")},
        "body_sha256": digest, "transport_kind": "consent-gated-direct-after-cc-miss"}
    row = {"schema": "aw.ietf-datatracker-minutes-record-parent.v1",
        "document_id": hashlib.sha256(f"ietf-minutes-record\0{url}".encode()).hexdigest(),
        "source_record_id": url, "source_id": subject.source_id(kind),
        "source_profile": "structured_data_feed", "record_kind": kind,
        "url": url, "canonical_url": url+"/", "title": title, "markdown": markdown,
        "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
        "parent_source_sha256": digest, "parent_identity": identity,
        "source_version": version, "reported_production_points": 1,
        "content_scope": subject.CONFIG[kind]["scope"],
        "extraction": {"full_prechunk_principal_content": True,
                       "source_text_truncation_applied": False}}
    validate_meeting_parent(row, kind=kind, artifact_root=tmp_path)
    row["content_scope"] = "metadata only"
    with pytest.raises(ValueError, match="parent replay failed"):
        validate_meeting_parent(row, kind=kind, artifact_root=tmp_path)
