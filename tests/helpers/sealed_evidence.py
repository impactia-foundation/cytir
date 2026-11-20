"""Skip helpers for tests that need the sealed consent probes the public cut does not carry.

`cytir.consent.derive.load_sealed_probes` reads the evidence files the family contracts point
at. A stranger's tree holds none of them, so a real ERIC row has no derivable consent basis
there. `require_sealed_probe` skips on that fact (the loader's own answer), not on a path.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cytir.consent.derive import load_sealed_probes  # noqa: E402
from tests.helpers.repo_files import REPO_ROOT


def require_sealed_probe(family: str = "eric_official_pdf") -> None:
    contracts = json.loads((REPO_ROOT / "data" / "source-family-contracts.v1.json").read_text())
    if not load_sealed_probes(contracts["families"], str(REPO_ROOT)).get(family):
        pytest.skip(f"no sealed consent probe for {family} in this tree", allow_module_level=True)


@pytest.fixture
def sealed_eric_probe():
    require_sealed_probe()
