import inspect

from cytir.ops.audits import finma_web_pdf_partitions as subject


def test_audit_requires_contextual_supersession_and_disjoint_ranges():
    source = inspect.getsource(subject.audit)
    assert "aw.finma-pdf-contextual-outlinks-supersession.v1" in source
    assert "finma-pdf-markdown-pilot.v6-contextual-outlinks" in source
    assert '"corrected-direct-producer"' in source
    assert "FINMA partitions overlap or leave a source-order gap" in source
    assert "set(range(expected_start, expected_start + expected_count))" in source


def test_audit_keeps_url_identity_separate_from_body_digest():
    source = inspect.getsource(subject.audit)
    assert "FINMA body digest was mixed into source identity" in source
    assert 'parent.get("canonical_url") != identity' in source
    assert "unique_pdf_bodies" in source


def test_audit_requires_page_metadata_and_link_context_without_relaxation():
    source = inspect.getsource(subject.audit)
    assert "FINMA exact API metadata drift" in source
    assert "FINMA page geometry diagnostics failed" in source
    assert 'not str(link.get("context") or "").strip()' in source
    assert "FINMA explicit-empty XOR failed" in source


def test_audit_normalizes_legacy_l0_binding_to_exhaustive_artifact_object():
    source = inspect.getsource(subject.audit)
    assert "l0_binding = _artifact(l0_path, artifact_root)" in source
