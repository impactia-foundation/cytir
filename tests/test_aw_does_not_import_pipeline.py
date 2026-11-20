#!/usr/bin/env python3
"""The dependency arrow (architecture-doctrine #4, task 643 section 6.1): `cytir/` never
imports `pipeline/` (nor `research/`, nor `tasks/`). Ops and pipeline import the
package; the package never reaches back.

Why an AST test and not `import-linter`: pipeline modules are addressed by FLAT
name (`import promote_queue`, `from serving_config import x`) after a
`sys.path.insert(0, .../pipeline)`, or as the namespace package
(`from pipeline import gold_presence_gate`). import-linter sees the first form as an
unrelated top-level module and cannot tell it from a third-party package. This
test resolves every import in `cytir/**/*.py` against the file names that actually
exist in `pipeline/`, so both forms are caught.

What it does NOT see (and the CI-guard `test_pipeline_aw_no_tasks_exec.py` does not
either): a `subprocess` call or a release-dir path that NAMES a pipeline script
(a release-dir path to the news stage contract script in
`cytir/sources/families/news_tier2_apply.py`). Those are script invocations inside a
pinned release, not imports; they go when those scripts move (task 643 S7).

`_DEFERRED_EDGES` is a ratchet, like the guard's `_DEFERRED_FILES`: an edge that is
no longer present must be deleted from it (the test fails on a stale row), and a
new edge fails the test. It is EMPTY since task 677's live-modules slice moved
`news_daily_gate` and `promote_queue` into `cytir/`. Do not add a row; move the module.
"""
from __future__ import annotations

import ast
import pathlib

from tests.helpers.repo_files import repo_glob_or_skip

REPO = pathlib.Path(__file__).resolve().parents[1]
AW = REPO / "cytir"
PIPELINE = REPO / "pipeline"

#: (file under cytir/, the pipeline module it imports). Empty when the arrow holds.
_DEFERRED_EDGES: dict[tuple[str, str], str] = {}


def _pipeline_names() -> set[str]:
    names = {p.stem for p in PIPELINE.glob("*.py")}
    names |= {p.name for p in PIPELINE.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))}
    names.discard("cytir")
    return names


def _edges() -> list[tuple[str, str, int]]:
    flat = _pipeline_names()
    found: list[tuple[str, str, int]] = []
    for path in sorted(AW.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top == "pipeline":
                        parts = alias.name.split(".")
                        found.append((rel, parts[1] if len(parts) > 1 else "pipeline", node.lineno))
                    elif top in flat:
                        found.append((rel, top, node.lineno))
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                parts = node.module.split(".")
                if parts[0] == "pipeline":
                    names = [parts[1]] if len(parts) > 1 else [a.name for a in node.names]
                    found += [(rel, n, node.lineno) for n in names]
                elif parts[0] in flat:
                    found.append((rel, parts[0], node.lineno))
            elif isinstance(node, ast.Call):
                # importlib.import_module("x") / __import__("x") with a literal name
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if name in ("import_module", "__import__") and node.args \
                        and isinstance(node.args[0], ast.Constant) \
                        and isinstance(node.args[0].value, str):
                    top = node.args[0].value.split(".")[0]
                    if top in flat or top == "pipeline":
                        found.append((rel, node.args[0].value, node.lineno))
    return found


def test_aw_imports_nothing_from_pipeline():
    repo_glob_or_skip("pipeline/*.py")   # the arrow is policed where that tree exists
    edges = [e for e in _edges() if (e[0], e[1]) not in _DEFERRED_EDGES]
    assert not edges, (
        "cytir/ imports pipeline/ (architecture-doctrine #4: the arrow points inward). "
        "Move the imported module into cytir/ (leave a shim in pipeline/), do not "
        "import it from there:\n  " + "\n  ".join(f"{f}:{ln} imports {m}" for f, m, ln in edges))


def test_deferred_edges_are_a_ratchet_not_a_parking_lot():
    repo_glob_or_skip("pipeline/*.py")   # the arrow is policed where that tree exists
    present = {(f, m) for f, m, _ln in _edges()}
    stale = sorted(set(_DEFERRED_EDGES) - present)
    assert not stale, f"_DEFERRED_EDGES row(s) with no edge left, delete: {stale}"


def test_the_scanner_sees_both_import_forms(tmp_path, monkeypatch):
    """The scanner must catch the flat form and the namespace form, or a green
    run proves nothing. Synthetic package, real scanner."""
    import sys
    me = sys.modules[__name__]
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "foo_mod.py").write_text("X = 1\n")
    (tmp_path / "cytir").mkdir()
    (tmp_path / "cytir" / "a.py").write_text("import foo_mod\n")
    (tmp_path / "cytir" / "b.py").write_text(f"from {PIPELINE.name} import foo_mod\n")
    (tmp_path / "cytir" / "c.py").write_text("from foo_mod import X\n")
    (tmp_path / "cytir" / "d.py").write_text("import importlib\nimportlib.import_module('foo_mod')\n")
    (tmp_path / "cytir" / "ok.py").write_text("import json\nfrom cytir import a\n")
    monkeypatch.setattr(me, "REPO", tmp_path)
    monkeypatch.setattr(me, "AW", tmp_path / "cytir")
    monkeypatch.setattr(me, "PIPELINE", tmp_path / "pipeline")
    got = {(f, m) for f, m, _ in me._edges()}
    assert got == {("cytir/a.py", "foo_mod"), ("cytir/b.py", "foo_mod"),
                   ("cytir/c.py", "foo_mod"), ("cytir/d.py", "foo_mod")}
