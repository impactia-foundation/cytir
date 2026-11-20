"""Every `run_family` stage row must compose (task 2204).

`compose` reads `document_id` through `default_anchor`; a family whose rows are legacy `Passage`
records (no `document_id`) must be listed in `FAMILY_ANCHOR_FN`, or compose dies with
`KeyError: 'document_id'`. This guard runs the family and asks compose's own anchor function.
"""
from __future__ import annotations

import pathlib

from cytir.core import common as C
from cytir.index import compose
from cytir.sources import run_family

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "statcan_wds"
OK = {"bucket": "ok", "search_ok": True, "train_ok": True, "status": "ok"}


def test_statcan_wds_rows_compose_and_reproduce_their_point_ids(tmp_path):
    run_family.run_family("statcan_wds", out_dir=tmp_path / "o", today="2026-10-01", log=lambda m: None,
                          consent=lambda h, d: {**OK, "host": h}, options={"raw": str(FIX)})
    _, recs = run_family.load_pack(tmp_path / "o" / "pack")
    assert recs and all("document_id" not in r for r in recs)
    fn = compose.anchor_fn_for_family("statcan_wds")
    for r in recs:
        assert C.mint_passage_point_id(r["url"], fn(r), r["text"]) == r["point_id"]


def test_uk_legislation_rows_reproduce_their_point_ids_through_compose(tmp_path):
    fix = pathlib.Path(__file__).resolve().parent / "fixtures" / "uk_legislation"
    run_family.run_family("uk_legislation", out_dir=tmp_path / "o", today="2026-11-01", log=lambda m: None,
                          consent=lambda h, d: {**OK, "host": h}, options={"akn_dir": str(fix)})
    _, recs = run_family.load_pack(tmp_path / "o" / "pack")
    assert recs
    fn = compose.anchor_fn_for_family("uk_legislation")
    bad = [r["url"] for r in recs if C.mint_passage_point_id(r["url"], fn(r), r["text"]) != r["point_id"]]
    assert not bad


def test_a_family_with_document_id_rows_keeps_the_default_anchor():
    assert compose.anchor_fn_for_family("stackoverflow_qa") is compose.default_anchor


def test_uk_legislation_uses_the_rows_own_anchor():
    assert compose.anchor_fn_for_family("uk_legislation") is compose.raw_anchor_field
