"""Guard: the monorepo root holds only its known committed entries.

Two lanes once wrote their docs at the monorepo root instead of the app
subtree; the docs were invisible from the app tree. This test freezes the
set of top-level committed entries. Adding a root entry is a deliberate
change: update ROOT_ALLOW_LIST in the same PR and say why in its body.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT_ALLOW_LIST = frozenset(
    {
        ".gitattributes",
        ".github",
        ".gitignore",
        "CLAUDE.md",
        "README.md",
        "authoritative-web-app",
        "authoritative-web-org",
    }
)

HINT = (
    "a lane wrote at the monorepo root — move it under authoritative-web-app/ "
    "(see notes/lane-traps.md 2026-11-05)"
)


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return out.stdout


def test_monorepo_root_entries_are_the_known_set():
    if shutil.which("git") is None:
        pytest.skip("git is not available; cannot list committed root entries")
    root = Path(__file__).resolve().parents[2]
    try:
        toplevel = Path(_git(root, "rev-parse", "--show-toplevel").strip()).resolve()
    except (subprocess.CalledProcessError, OSError):
        pytest.skip("not a git checkout (e.g. the public cut); root guard does not apply")
    if toplevel != root:
        pytest.skip("app dir is not the root of its own git checkout; guard does not apply")
    try:
        names = _git(root, "ls-tree", "--name-only", "HEAD").split()
    except subprocess.CalledProcessError:
        pytest.skip("no HEAD commit to list; root guard does not apply")
    found = set(names)
    extra = sorted(found - ROOT_ALLOW_LIST)
    missing = sorted(ROOT_ALLOW_LIST - found)
    assert found == ROOT_ALLOW_LIST, (
        f"{HINT}. unexpected root entries: {extra}; allow-listed but absent: {missing}"
    )
