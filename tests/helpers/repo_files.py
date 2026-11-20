"""Skip-or-return helpers for tests that read repo files the public cut withholds.

Use this from any test that opens files under `deploy/`, `scripts/`, or the
operator-only directories (the trees the public cut leaves out). Call
`repo_file_or_skip(rel)` for one fixed file and `repo_glob_or_skip(pattern)` for a
set. Both skip with a reason naming the path when the file or glob is absent, and
both work at module level and inside a test function. Do not hand-write an
`is_file()` check plus `pytest.skip`: that is how each new copy drifts and turns the
cut rehearsal red.
"""
from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def repo_file_or_skip(rel: str | pathlib.PurePath) -> pathlib.Path:
    """Return REPO_ROOT / rel, or skip the test when that file is absent."""
    path = REPO_ROOT / rel
    if not path.is_file():
        pytest.skip(f"{rel} is not shipped in the public cut", allow_module_level=True)
    return path


def repo_glob_or_skip(pattern: str) -> list[pathlib.Path]:
    """Return the sorted matches of pattern under REPO_ROOT, or skip when there are none."""
    found = sorted(REPO_ROOT.glob(pattern))
    if not found:
        pytest.skip(f"{pattern} is not shipped in the public cut", allow_module_level=True)
    return found
