import gzip
import hashlib
from pathlib import Path

import pytest

from cytir.extract.routec import ietf_internet_drafts as subject
from cytir.sources.routec.audit_profile import validate_draft_parent


def test_parent_gate_replays_exact_datatracker_record(tmp_path: Path) -> None:
    html = b"<html><head><title>draft-example-02 - Example</title></head><body><main><h1>Example draft-example-02</h1><p>Abstract</p><p>Complete.</p></main></body></html>"
    body = tmp_path / "body.gz"
    with body.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as target: target.write(html)
    compressed = body.read_bytes(); digest = hashlib.sha256(html).hexdigest()
    artifact = {"uri": "artifact://body.gz", "bytes": len(compressed),
        "sha256": hashlib.sha256(compressed).hexdigest(), "compression": "gzip",
        "body_bytes": len(html), "body_sha256": digest}
    url = "https://datatracker.ietf.org/doc/draft-example"; final = url + "/"
    title, markdown, version = subject.render_draft_record(
        html.decode(), url=url, final_url=final, body_sha256=digest)
    identity = {"terminal_receipt": {"sha256": "a" * 64},
        "artifacts": {key: artifact for key in
            ("response_body", "response_headers", "transport", "consent")},
        "body_sha256": digest, "transport_kind": "consent-gated-direct-after-cc-miss"}
    row = {"schema": "aw.ietf-datatracker-draft-record-parent.v1",
        "document_id": hashlib.sha256(f"ietf-draft-record\0{url}".encode()).hexdigest(),
        "source_record_id": url, "source_id": subject.SOURCE_ID,
        "source_profile": "structured_data_feed", "url": url, "canonical_url": final,
        "title": title, "markdown": markdown,
        "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
        "parent_source_sha256": digest, "parent_identity": identity,
        "source_version": version, "reported_production_points": 2,
        "content_scope": "complete Datatracker document record; linked draft text is not claimed as embedded",
        "extraction": {"full_prechunk_principal_content": True,
                       "source_text_truncation_applied": False}}
    validate_draft_parent(row, artifact_root=tmp_path)
    row["content_scope"] = "full draft text"
    with pytest.raises(ValueError, match="source replay failed"):
        validate_draft_parent(row, artifact_root=tmp_path)
