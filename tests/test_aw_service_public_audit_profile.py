from cytir.sources.service_public.audit_profile import _metadata_matches


def test_service_public_chunk_metadata_requires_all_fiche_lineage():
    parent = {
        "title": "T", "source_relative_path": "F1.xml", "source_sha256": "s",
        "source_heading_count": 2, "modified_date": "2026-01-01",
        "source_version": "v", "fiche_id": "F1", "fiche_type": "Fiche",
        "subject": "Sujet", "archive_member": "F1.xml", "archive_sha256": "a",
        "archive_member_sha256": "s",
    }
    assert _metadata_matches(dict(parent), parent)
    incomplete = dict(parent)
    incomplete.pop("modified_date")
    assert not _metadata_matches(incomplete, parent)
