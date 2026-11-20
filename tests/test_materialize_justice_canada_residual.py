from pathlib import Path


import cytir.sources.justice_canada.residual as _res
MODULE = Path(_res.__file__)


def test_residual_materializer_is_exact_class_and_lineage_bound() -> None:
    text = MODULE.read_text()
    assert '"eng/regulations": {"points": 38_321, "urls": 3_036}' in text
    assert '"fra/lois": {"points": 35_155, "urls": 853}' in text
    assert '"fra/reglements": {"points": 40_817, "urls": 3_067}' in text
    assert "legal_unit_anchors" in text
    assert "source_links_with_anchors" in text
    assert "prior_terminal_witness_overlap" in text
    assert "ledger_subtraction_authorized\": False" in text
