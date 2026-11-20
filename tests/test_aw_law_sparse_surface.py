"""Recording pin for `cytir.corpus.law_sparse_surface.resurfaced_sparse_text` (task 1476).

The module moved from the pipeline script blob 9281507d7426e3f3db9beac90699f2c8c2acd5f5
without a rewrite. The digest below was computed from that ORIGINAL file on the seven
verbatim fedlex pack rows (task 564): the composed sparse TEXT must stay byte-identical,
because the sparse surface is an embedding-surface contract.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

from cytir.corpus import law_sparse_surface as LS

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "law_sparse_pack_rows.json"
PIN = "8aa563c3e8923a80142a2d77312283bea5e505daf9e54f42dbbbf222d605f1f7"


def test_composed_sparse_text_is_byte_identical_to_the_original():
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["rows"]
    h = hashlib.sha256()
    for r in rows:
        h.update(LS.resurfaced_sparse_text(r["pack_row"], r["akn"]["heading_ctx"]).encode() + b"\0")
    assert h.hexdigest() == PIN


def test_version_and_main_exist():
    assert LS.VERSION == "aw.law-pack-sparse-resurface.v1"
    assert callable(LS.main)
