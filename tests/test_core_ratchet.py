#!/usr/bin/env python3
"""Task 646 S3 -- `uuid5(`, `def canon*`, `QdrantClient(` may appear only in
`cytir/core` (task 643 §6.1: "Rules the layout enforces (as CI checks, not
prose)"). This is a CEILING, not a target: today's offenders (recorded once,
2026-09-23, by AST scan -- see `data/core-ratchet-baseline.v1.json`) are
allow-listed by file; the list may only SHRINK as forks get replaced by an
`cytir.core` import (task 643 §6.2 S3: "Replace the forks only on the prod-
reachable set (class A)" -- a separate, larger effort this test does not do).
A new file that adds one of these three patterns outside `cytir/core` fails here
immediately, instead of becoming the 129th thing a future migration has to
sort out by hand.

This is the repo-wide sibling of `test_identity_minting_homes.py`'s
`KNOWN_LOCAL_UUID5_INGESTERS` ratchet (which predates `cytir/core` and stays
scoped to the private ingest scripts for the corpus-identity reasons documented
there); this one covers the whole tree and all three patterns cytir/core owns.

AST-based, not a substring grep: a docstring or comment that MENTIONS
`uuid5(...)` to document which scheme a file's records are under must not
count as a violation (test_identity_minting_homes.py's own lesson).
"""
from __future__ import annotations

import ast
import functools
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
BASELINE_PATH = REPO / "data" / "core-ratchet-baseline.v1.json"
LEGAL_HOME = REPO / "cytir" / "core"

#: Directories that are never source of truth for this ratchet: git internals
#: and build metadata. NOT the evidence tree -- the rule is repo-wide (task 643 §6.1
#: says "cytir/core", not "cytir/core, excusing the evidence tree").
EXCLUDE_DIR_NAMES = {".git", "aw.egg-info", "cytir.egg-info"}


def _iter_repo_py_files():
    for p in REPO.rglob("*.py"):
        if any(part in EXCLUDE_DIR_NAMES for part in p.parts):
            continue
        try:
            p.relative_to(LEGAL_HOME)
            continue                      # inside the legal home
        except ValueError:
            pass
        yield p


def _calls_named(tree: ast.AST, names: set[str]) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in names:
            return True
        if isinstance(fn, ast.Name) and fn.id in names:
            return True
    return False


def _defines_canon(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name.startswith("canon"):
            return True
    return False


@functools.lru_cache(maxsize=1)
def _scan_repo() -> dict[str, list[str]]:
    """Return {"uuid5": [...], "def_canon": [...], "qdrant_client": [...]},
    each a sorted list of repo-relative paths (outside cytir/core) whose AST
    contains the pattern.

    Cached: the three parametrized `test_no_new_offenders_outside_aw_core`
    cases would otherwise re-walk + re-parse the whole ~3,400-file repo
    three times in one pytest run for no reason."""
    uuid5, canon, qdrant = [], [], []
    for p in _iter_repo_py_files():
        rel = str(p.relative_to(REPO))
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=rel)
        except (SyntaxError, UnicodeDecodeError):
            # Not this ratchet's job to police parseability; a file this
            # test cannot parse cannot contain an AST-detectable violation.
            continue
        if _calls_named(tree, {"uuid5"}):
            uuid5.append(rel)
        if _defines_canon(tree):
            canon.append(rel)
        if _calls_named(tree, {"QdrantClient"}):
            qdrant.append(rel)
    return {"uuid5": sorted(uuid5), "def_canon": sorted(canon),
            "qdrant_client": sorted(qdrant)}


def test_the_ratchet_can_actually_see_a_violation(tmp_path, monkeypatch):
    """A guard that has never fired is an untested code path pointed at your
    data (test_identity_minting_homes.py's own lesson, applied here)."""
    (tmp_path / "prose.py").write_text(
        '"""Mentions uuid5(...), QdrantClient(...) and def canonical_x() in '
        'a docstring only -- must not count."""\nx = 1\n')
    (tmp_path / "mints.py").write_text(
        "import uuid\npid = uuid.uuid5(uuid.NAMESPACE_URL, 'u')\n")
    (tmp_path / "canons.py").write_text(
        "def canonical_thing(x):\n    return x\n")
    (tmp_path / "clients.py").write_text(
        "from qdrant_client import QdrantClient\nc = QdrantClient(':memory:')\n")

    import sys
    self_mod = sys.modules[__name__]
    monkeypatch.setattr(self_mod, "REPO", tmp_path)
    monkeypatch.setattr(self_mod, "LEGAL_HOME", tmp_path / "cytir" / "core")
    _scan_repo.cache_clear()          # else this reads the real repo's cached scan
    try:
        found = _scan_repo()
        assert found["uuid5"] == ["mints.py"]
        assert found["def_canon"] == ["canons.py"]
        assert found["qdrant_client"] == ["clients.py"]
    finally:
        _scan_repo.cache_clear()      # don't leak the tmp_path result to later tests


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
