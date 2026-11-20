"""The contract must measure the surface the BINDER tokenises, not a stale copy.

Task 533 ran the ratified checker against the sealed entity-card stage and got
`conformance.sparse` on 20,000 of 20,000 rows -- a 100% FAIL on a family whose
sparse channel is correct.

The mechanism: `check_sample`'s `get_sparse` default hardcoded the pre-528a
binder line, `rec["text"] or rec["dense_text"]`. Task 528a then gave a family
that DECLARES `sparse_surface_field` its own channel
(`assembly_binder.py`: `if _sparse_channel_declared and sparse_surface and
sparse_surface != surface`), so from that day the checker measured a surface the
binder no longer used -- for exactly the two families, `entity_card` and
`law_article_pack`, that the contract exists to protect.

These tests pin the mirror to the binder's own conditions, so the next change to
either side fails here instead of turning the checker into noise.
"""
from __future__ import annotations

import pathlib

import pytest

from cytir.index.representation_contract import binder_sparse_surface, load_registry  # noqa: E402

REGISTRY = load_registry(None)
CARD = REGISTRY["families"]["entity_card"]
F2 = REGISTRY["families"]["f2_html"]


def test_a_declaring_family_gets_its_own_sparse_channel():
    rec = {"text": "Waldsieversdorf — Gemeinde im Land Brandenburg.",
           "dense_text": "Waldsieversdorf — Gemeinde im Land Brandenburg.",
           "sparse_text": "Waldsieversdorf municipality Germany Gemeinde"}
    assert binder_sparse_surface(rec, CARD) == rec["sparse_text"]


def test_a_non_declaring_family_still_gets_text_or_dense_text():
    rec = {"text": "the body", "dense_text": "Title. the body",
           "sparse_text": "a channel nobody reads for this family"}
    assert binder_sparse_surface(rec, F2) == "the body"
    assert binder_sparse_surface({"dense_text": "d"}, F2) == "d"


def test_an_identical_sparse_text_is_not_treated_as_a_channel():
    """The binder's own `!= surface` clause, mirrored.

    A row whose `sparse_text` equals its `text` is not carrying a distinct
    channel; treating it as one would make the checker and the binder disagree
    about rows where nothing is wrong.
    """
    rec = {"text": "same", "sparse_text": "same"}
    assert binder_sparse_surface(rec, CARD) == "same"


def test_an_empty_or_whitespace_sparse_text_falls_back():
    for v in ("", "   ", None):
        rec = {"text": "the body", "sparse_text": v}
        assert binder_sparse_surface(rec, CARD) == "the body"


def test_a_row_with_no_surface_at_all_yields_empty():
    assert binder_sparse_surface({}, CARD) == ""
    assert binder_sparse_surface({"sparse_text": "hooks"}, CARD) == ""


def test_the_declaring_families_are_the_ones_528a_scoped():
    """If a family gains or loses its own channel, this test says so."""
    declaring = {k for k, v in REGISTRY["families"].items()
                 if v.get("sparse_surface_field")}
    assert declaring == {"entity_card", "law_article_pack"}, sorted(declaring)


def test_the_mirror_matches_the_binder_source():
    """The binder's line, read from the file, must still be the one mirrored.

    A source check rather than a behavioural one, because the failure being
    guarded is a SILENT divergence: the checker kept passing its own tests
    while measuring the wrong field for two months.
    """
    binder = pathlib.Path(__file__).resolve().parents[1] / "cytir" / "index" / "assemble_stage.py"
    if not binder.is_file():
        pytest.skip("assemble_stage.py not present (public cut)")
    src = binder.read_text()
    assert 'sparse_surface = (rec.get("sparse_text") or "").strip()' in src
    assert "sparse_surface != surface" in src
    assert 'surface = rec.get("text") or rec.get("dense_text")' in src
    field = {v.get("sparse_surface_field")
             for v in REGISTRY["families"].values()
             if v.get("sparse_surface_field")}
    assert field == {"sparse_text"}, (
        "the binder reads the literal `sparse_text`; a family declaring a "
        f"different field name would be silently ignored: {field}")
