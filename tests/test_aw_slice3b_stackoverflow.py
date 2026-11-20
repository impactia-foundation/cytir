#!/usr/bin/env python3
"""Task 696 slice 3b (3/3) -- the Stack Overflow chain moved into `cytir/` behind alias shims.

The alias shims were deleted in task 1139 (S9 slice 9); MOVED stays as the table of sealed PRODUCER labels.

The materializer moved in task 1509 to `cytir/sources/stackoverflow/materialize_production_tranche.py`; the
driver hashes the package file's bytes into `AW_PRODUCER_REVISION` (future tranches only).
Its in-file PRODUCER label keeps the old path (sealed rows).
"""
from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent

MOVED = [
    ("seal_content_fidelity_audit", "cytir.index.seal_content_fidelity_audit"),
    ("seal_source_profile_compliance", "cytir.index.seal_source_profile_compliance"),
    ("run_observed_one_shot", "cytir.ops.run_observed_one_shot"),
    ("portableize_embedding_pack", "cytir.embed.portableize_pack"),
    ("adapt_pack_to_413a_stage", "cytir.embed.adapt_pack_to_stage"),
]


def test_the_materializer_lives_in_the_package_and_keeps_its_sealed_label():
    src = (REPO / "cytir/sources/stackoverflow/materialize_production_tranche.py").read_text()
    assert "PRODUCER = \"pipeline/materialize_stackoverflow_qa_production_tranche.py\"" in src
    assert "Shim: moved to" not in src[:400]


def test_producer_labels_keep_the_old_path_strings():
    """The PRODUCER labels are written into sealed receipts: they must not follow the move."""
    for old, new in MOVED:
        mod = importlib.import_module(new)
        label = getattr(mod, "PRODUCER", None)
        if label is not None:
            assert label == f"pipeline/{old}.py", (new, label)
