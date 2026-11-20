#!/usr/bin/env python3
"""The carried-BM25 contract (task 515, second defect).

`assembly_binder._SrcShim` set only `dense`. `project_record`'s no-text tail
then "keeps the existing posting" -- of which there was none -- and wrote an
empty sparse vector. Measured on task 514's staged tree, per partition:
19,619,656 rows (63.5% of `aw_e4`) reached the collection with NO BM25 posting
at all, including the entire 19,227,106-row Wikipedia slice. The BM25 leg of
two thirds of the index was silently absent and no count check could see it,
because a row with an empty sparse vector is still a row.
"""
from __future__ import annotations

import os
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault("AW_FOLD_ACCENTS", "1")

import cytir.core.common as AC  # noqa: E402
from cytir.index import b1_slice as B  # noqa: E402
from cytir.index import b1_projection as BP  # noqa: E402
from cytir.consent import build_gate as BG  # noqa: E402

TWIN_ROW = {
    "url": "https://de.wikipedia.org/wiki/Echtra_Fergusa_maic_Lete",
    "point_id": "000000e5-a128-5b29-b5c4-eca77b6c3307",
    "title": "Echtra Fergusa maic Lete", "lang": "de",
    "domain": "de.wikipedia.org", "source_type": "reference",
    "trust_tier": 0.75, "country": "INTL",
    "snippet": "Echtra Fergusa maic Lete ist der Titel einer Erzaehlung.",
    "rule_epochs": {"card": 1, "chunk": 3, "country": 2, "embed": 1,
                    "extract": 3, "tok": 2, "trust": 1},
}


class Sparse:
    def __init__(self, indices, values):
        self.indices = indices
        self.values = values


class Shim:
    """`assembly_binder._SrcShim`'s surface, including the new `bm25`."""

    def __init__(self, rec, bm25=None, dim=8):
        self.id = rec["point_id"]
        self.payload = dict(rec)
        self.vector = {"dense": [0.1] * dim}
        if bm25 is not None:
            self.vector["bm25"] = bm25


def _project(shim, corpus_row=None):
    stats = defaultdict(int)
    point, _meta = BP.project_record(
        shim, origin="wikipedia_413a", corpus_row=corpus_row,
        published_map={}, conf_map={},
        gate=BG.ConsentGate(mode="trust", grace_days=30, cache_path="",
                           log=lambda *a: None),
        stats=stats)
    return point, stats


def test_a_no_text_row_with_no_carried_posting_ships_with_no_bm25():
    """The defect, pinned as a fact."""
    point, stats = _project(Shim(TWIN_ROW))
    assert "bm25" not in point["vector"]
    assert stats["tail_no_sparse"] == 1


def test_a_no_text_row_keeps_the_carried_posting():
    point, stats = _project(
        Shim(TWIN_ROW, bm25=Sparse([11, 22, 33], [1.5, 0.5, 0.25])))
    assert list(point["vector"]["bm25"].indices) == [11, 22, 33]
    assert list(point["vector"]["bm25"].values) == [1.5, 0.5, 0.25]
    assert stats["tail_no_sparse"] == 0


def test_a_carried_posting_keeps_the_tok_epoch_it_was_built_at():
    """Stamping tok@1 onto a posting the source declares as tok@2 is a false
    statement in the fossil record the next rebuild reads."""
    point, stats = _project(Shim(TWIN_ROW, bm25=Sparse([1], [1.0])))
    assert point["payload"]["rule_epochs"]["tok"] == 2
    assert stats["tail_tok2"] == 1


def test_an_absent_declaration_still_means_tok_1():
    row = dict(TWIN_ROW)
    row.pop("rule_epochs")
    point, stats = _project(Shim(row, bm25=Sparse([1], [1.0])))
    assert point["payload"]["rule_epochs"]["tok"] == 1
    assert stats["tail_tok1"] == 1


def test_a_row_with_text_still_regenerates_rather_than_carrying():
    """The carry is for rows that CANNOT regenerate. A row with a text surface
    re-tokenizes at the current epoch, and the stale posting is ignored."""
    row = dict(TWIN_ROW)
    point, stats = _project(
        Shim(row, bm25=Sparse([999999], [9.0])),
        corpus_row={"text": "Echtra Fergusa maic Lete ist eine Erzaehlung "
                            "aus dem Ulster-Zyklus der irischen Mythologie."})
    assert stats["retokenized"] == 1
    assert list(point["vector"]["bm25"].indices) != [999999]
    assert point["payload"]["rule_epochs"]["tok"] == B.ID.EPOCHS["tok"]


def test_the_binder_reads_dense_text_as_a_text_surface():
    """The task467 entity-card stage seals `dense_text` and no `text` (its own
    stage.json: "dense_text_contract: record['dense_text']"), so a binder that
    reads only `text` sends 339,307 cards down the no-text tail."""
    from cytir.index import assemble_stage as AS
    src = pathlib.Path(AS.__file__).read_text()
    assert 'surface = rec.get("text") or rec.get("dense_text")' in src
    assert AC.canonical_url("https://www.rfc-editor.org/rfc/rfc1.txt") == \
        "https://rfc-editor.org/rfc/rfc1.txt"
