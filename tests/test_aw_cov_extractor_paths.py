"""The two COV packs name package extractors; the runner resolves both to importable modules."""
from __future__ import annotations

import importlib

import cytir.sources.pack_loader as PL
from cytir.sources.runner import _package_module
from tests.helpers.repo_files import repo_file_or_skip

repo_file_or_skip("sources/ch/regulator-docs.yaml")
repo_file_or_skip("sources/igo/igo-core.yaml")

EXPECTED = {"ch-regulator-docs": ("cytir/sources/cov/adapter.py", "cytir.sources.cov.adapter"),
            "igo-core": ("cytir/sources/cov/engine.py", "cytir.sources.cov.engine")}


def test_cov_pack_extractors_resolve_to_importable_modules():
    for pack_id, (path, module) in EXPECTED.items():
        assert PL.load_pack(PL.find_pack(pack_id))["extractor"] == path, pack_id
        assert _package_module(path) == module
        assert importlib.import_module(module).__name__ == module
