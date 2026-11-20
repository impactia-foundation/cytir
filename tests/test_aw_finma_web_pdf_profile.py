from pathlib import Path

from cytir.ops.audits import finma_web_pdf_profile as subject


def test_finma_profile_pins_full_web_pdf_semantics():
    source = Path(subject.__file__).read_text()
    assert subject.PROFILE == "web_pdf"
    assert "expected_documents" in source
    assert "metadata_normalization" in source
    assert "markdown_changed" in source
    assert "parent_identity" in source
    assert "pdf_sha256" in source
    assert "pages" in source
    assert "explicit_empty" in source
    assert "target_url" in source and "context" in source
    assert "content_cap" in source
    assert "over_hard_max" in source
    assert "point-ID mapping" in source
    assert "[:8000]" not in source
