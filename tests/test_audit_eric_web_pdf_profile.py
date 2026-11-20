from pathlib import Path

import pytest

import cytir.ops.eric_web_pdf_profile as subject
import cytir.chunk.registry as CREG


def test_eric_audit_pins_web_pdf_semantics_and_full_content():
    source = Path(subject.__file__).read_text()
    assert subject.PROFILE == "web_pdf"
    assert "Docling" in source
    assert "page_count" in source
    assert "pdf_sha256" in source
    assert "explicit_empty" in source
    assert "content_cap" in source
    assert "[:8000]" not in source


# -- task 663, 2026-09-25: the policy gate asks the REGISTRY, not a literal --
# Real incident: this module hardcoded POLICY_PATH to the old epoch-4 768w
# chunk.generic policy. `chain_eric_pdf_partition_admission.sh` re-pointed
# to ADR-113's ratified 1024w arm; this audit's own copy of the "right"
# policy never got the memo, so a pack correctly built under the ratified
# policy failed this gate with "ERIC web_pdf pack policy/token gate failed"
# -- tranche 010000's real first chain run hit this live.

def test_the_old_hardcoded_policy_path_literal_is_gone():
    assert not hasattr(subject, "POLICY_PATH"), (
        "a hardcoded policy path constant here is exactly the bug: it "
        "falls out of sync with chain_eric_pdf_partition_admission.sh's "
        "own policy pin the moment either one changes without the other")
    source = Path(subject.__file__).read_text()
    assert "cytir.chunk.registry" in source
    assert subject.CHUNK_POLICY_FAMILY_ID == "eric_official_pdf"


def test_the_adr113_ratified_arm_is_admitted_by_the_registry_for_eric():
    """The exact sha256 the real, currently-failing pack declares -- proves
    the fix actually accepts what production needs, not just that SOME
    policy passes."""
    ratified_arm_sha256 = (
        "3f3e24823cf2e0851b149690b6c43d579e3b7d1066a66617b97fd86c5cc7e8d4")
    fam = CREG.assert_selectable(ratified_arm_sha256,
                                 family_id=subject.CHUNK_POLICY_FAMILY_ID)
    assert fam == "chunk.generic"


def test_a_policy_not_admitted_for_eric_is_refused():
    """A policy that IS registered but not admitted for eric_official_pdf
    (or not registered at all) must still be refused -- this is a real
    gate, not a rubber stamp that accepts any sha256 string."""
    with pytest.raises(CREG.ChunkPolicyRegistryError):
        CREG.assert_selectable("0" * 64, family_id=subject.CHUNK_POLICY_FAMILY_ID)


def test_diagnostics_conserve_retained_parent_and_explicit_quarantine():
    """Legacy (pre-task-656) shape: no `extractor` field at all -- resumed
    tranches can carry a genuine mix of these and the new routed shape in
    the same run (already-terminal documents are never re-extracted)."""
    parents = [{"source_record_id": "ED1", "parent_identity": {"pdf_sha256": "a"}}]
    quarantine = [{"eric_id": "ED2", "source_pdf": {"sha256": "b", "bytes": 12}}]
    diagnostics = [
        {"eric_id": "ED1", "pdf_sha256": "a", "docling_version": "2.111.0",
         "do_ocr": False, "do_table_structure": True, "empty_markdown": False,
         "page_count": 3},
        {"eric_id": "ED2", "pdf_sha256": "b", "pdf_bytes": 12,
         "do_ocr": False, "do_table_structure": True,
         "status": "extraction_failure", "page_count": None},
    ]
    subject._validate_diagnostics(parents=parents, diagnostics=diagnostics,
                                  quarantine=quarantine)

    corrupt = [dict(diagnostics[0]), dict(diagnostics[1], pdf_sha256="wrong")]
    with pytest.raises(ValueError, match="quarantine diagnostics"):
        subject._validate_diagnostics(parents=parents, diagnostics=corrupt,
                                      quarantine=quarantine)


def test_diagnostics_accept_routed_liteparse_and_docling_rows():
    """task 656: the new routed shape -- `extractor`/`extractor_version`
    replace the Docling-only `docling_version`, and `do_table_structure` is
    only required when `extractor == "docling"`."""
    parents = [
        {"source_record_id": "ED1", "parent_identity": {"pdf_sha256": "a"}},
        {"source_record_id": "ED2", "parent_identity": {"pdf_sha256": "b"}},
    ]
    diagnostics = [
        {"eric_id": "ED1", "pdf_sha256": "a", "extractor": "liteparse",
         "extractor_version": "2.14.7", "do_ocr": False,
         "empty_markdown": False, "page_count": 5},
        {"eric_id": "ED2", "pdf_sha256": "b", "extractor": "docling",
         "extractor_version": "2.111.0", "docling_version": "2.111.0",
         "do_ocr": False, "do_table_structure": True,
         "empty_markdown": False, "page_count": 3},
    ]
    subject._validate_diagnostics(parents=parents, diagnostics=diagnostics,
                                  quarantine=[])

    # a liteparse row is NOT required to carry do_table_structure
    missing_table_flag = [dict(diagnostics[0]), dict(diagnostics[1])]
    subject._validate_diagnostics(parents=parents, diagnostics=missing_table_flag,
                                  quarantine=[])

    # but a docling row still is
    broken_docling = [dict(diagnostics[0]),
                      {**diagnostics[1], "do_table_structure": False}]
    with pytest.raises(ValueError, match="incomplete"):
        subject._validate_diagnostics(parents=parents, diagnostics=broken_docling,
                                      quarantine=[])
