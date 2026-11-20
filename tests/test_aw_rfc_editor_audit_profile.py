from cytir.sources.rfc_editor.audit_profile import _restore_decorated_source, _selection_counts


def test_restore_decorated_source_is_exact_and_counts_headings():
    prefix = "---\naw...\n---\n\n# RFC 1\n\n"
    source = "INTRODUCTION\n\n1. Scope\nbody\n"
    markdown = prefix + "## INTRODUCTION\n\n## 1. Scope\nbody\n"
    restored, headings = _restore_decorated_source(markdown, source, len(prefix))
    assert restored == source
    assert headings == 2


def test_restore_decorated_source_rejects_content_change():
    import pytest

    with pytest.raises(ValueError, match="non-reversible"):
        _restore_decorated_source("prefixchanged\n", "exact\n", len("prefix"))


def test_selection_counts_supports_exact_current_production_pack():
    run = {"selection": "production", "counts": {
        "production_documents": 4562, "production_outlinks": 13494,
        "empty_outlink_parents": 2908}}
    pack = {"counts": {"documents": 4562, "chunks": 44543}}
    assert _selection_counts(run, pack) == ("production", 4562, 13494, 2908, 44543)
