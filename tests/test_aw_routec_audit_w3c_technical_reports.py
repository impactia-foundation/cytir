import hashlib

import pytest

from cytir.sources.routec.audit_profile import validate_w3c_parent as validate_parent
from cytir.extract.routec.w3c_technical_reports import SOURCE_ID


def _parent() -> dict:
    markdown = "# Test\n\nComplete report.\n"
    return {
        "schema": "aw.w3c-technical-report-parent.v1",
        "document_id": "doc", "source_record_id": "https://www.w3.org/TR/test/",
        "source_id": SOURCE_ID, "source_profile": "software_docs",
        "url": "https://www.w3.org/TR/test/", "markdown": markdown,
        "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
        "parent_source_sha256": "a" * 64,
        "parent_identity": {
            "terminal_receipt": {"sha256": "b" * 64},
            "response_body": {"body_sha256": "a" * 64},
            "response_headers": {"body_sha256": "c" * 64},
            "transport": {"body_sha256": "d" * 64},
            "consent": {"body_sha256": "e" * 64}},
        "source_version": {"requested_path": "/TR/test/",
                           "identity_basis": "exact-requested-TR-path+preserved-response"},
        "extraction": {"full_prechunk_principal_content": True,
                       "source_text_truncation_applied": False},
    }


def test_parent_gate_requires_exact_tr_path_and_full_identity() -> None:
    validate_parent(_parent())
    row = _parent(); row["url"] = "https://www.w3.org/blog/test"
    with pytest.raises(ValueError, match="identity/fidelity failed"):
        validate_parent(row)


def test_parent_gate_recomputes_markdown_digest() -> None:
    row = _parent(); row["markdown"] += "changed"
    with pytest.raises(ValueError, match="identity/fidelity failed"):
        validate_parent(row)
