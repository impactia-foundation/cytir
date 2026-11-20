"""No module reads a lazily-resolved setting at import (task 696 slice 4, round 3; review of #567).

A setting nobody configured refuses by name where it is USED. Several `aw` modules therefore resolve
a module constant on first access (a module-level `__getattr__`: `LOCK_DIR`, `UA`, `DEFAULT_QUEUE_DIR`,
...). A `from cytir.ops.write_lock import LOCK_DIR`, or a module-level `UA = pack_fetch.UA`, runs that
resolution when the importer loads, so on a box with no operator settings the importer dies at
COLLECTION (`SettingRefused: lock_dir`) instead of where the value is used. The review found four.

This is a static scan (AST, no import, no execution) over every Python file in the checkout:

* the lazy names are DERIVED from the tree: every `aw` module that defines a module-level
  `__getattr__` contributes the upper-case names it answers (no hand list to forget);
* a pipeline shim ("Shim: moved to `cytir.x.y`") counts as that module;
* it flags a `from <lazy module> import <lazy name>` and an attribute read `<alias>.<lazy name>`
  that runs at import (module body, class bodies, decorators and default arguments; not function
  bodies, lambdas or the `if __name__ == "__main__"` block).

Bind lazily instead: read `module.NAME` inside the function that uses it.
"""
from __future__ import annotations

import ast
import pathlib
import re
import pytest

pytestmark = pytest.mark.guard

APP = pathlib.Path(__file__).resolve().parents[1]
ROOTS = ("cytir", "pipeline", "tasks", "server", "deploy", "scripts", "webapp", "tests")
_SHIM_RE = re.compile(r"Shim: moved to `(cytir(?:\.\w+)+)`")


def _files(app: pathlib.Path):
    for root in ROOTS:
        base = app / root
        if base.is_dir():
            for p in sorted(base.rglob("*.py")):
                if "__pycache__" not in p.parts:
                    yield p


def _parse(p: pathlib.Path):
    try:
        return ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, ValueError):
        return None


def _modname(app: pathlib.Path, p: pathlib.Path) -> str:
    parts = list(p.relative_to(app).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def lazy_names(app: pathlib.Path) -> dict[str, set[str]]:
    """{module: names it resolves lazily}, derived from the module-level `__getattr__` functions,
    plus the pipeline shims that alias such a module."""
    lazy: dict[str, set[str]] = {}
    for p in (app / "cytir").rglob("*.py"):
        tree = _parse(p)
        for n in (tree.body if tree else []):
            if isinstance(n, ast.FunctionDef) and n.name == "__getattr__":
                names = {c.value for c in ast.walk(n)
                         if isinstance(c, ast.Constant) and isinstance(c.value, str)
                         and re.fullmatch(r"_?[A-Z][A-Z0-9_]*", c.value)}
                lazy.setdefault(_modname(app, p), set()).update(names)
    for p in (app / "pipeline").glob("*.py") if (app / "pipeline").is_dir() else ():
        m = _SHIM_RE.search(p.read_text(encoding="utf-8", errors="ignore")[:600])
        if m and m.group(1) in lazy:
            lazy.setdefault(p.stem, set()).update(lazy[m.group(1)])
    return lazy


class _ImportTime(ast.NodeVisitor):
    """Walk what runs when the module loads; never a function or lambda body."""

    def __init__(self, lazy, modname, is_pkg):
        self.lazy, self.modname, self.is_pkg = lazy, modname, is_pkg
        self.alias: dict[str, str] = {}
        self.hits: list[tuple[int, str]] = []

    def _absolute(self, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        parts = self.modname.split(".")
        if not self.is_pkg:
            parts = parts[:-1]
        parts = parts[:len(parts) - (node.level - 1)] if node.level > 1 else parts
        return ".".join(parts + ([node.module] if node.module else []))

    def visit_ImportFrom(self, node):
        base = self._absolute(node)
        bad = [a.name for a in node.names if a.name in self.lazy.get(base, ())]
        if bad:
            self.hits.append((node.lineno, f"from {base} import {', '.join(bad)}"))
        for a in node.names:
            if f"{base}.{a.name}" in self.lazy:
                self.alias[a.asname or a.name] = f"{base}.{a.name}"

    def visit_Import(self, node):
        for a in node.names:
            if a.name in self.lazy:
                self.alias[a.asname or a.name.split(".")[0]] = a.name

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self.alias \
                and node.attr in self.lazy[self.alias[node.value.id]]:
            self.hits.append((node.lineno, f"{node.value.id}.{node.attr} read at import"))
        self.generic_visit(node)

    def _def(self, node):
        for d in [*node.decorator_list, *node.args.defaults, *[k for k in node.args.kw_defaults if k]]:
            self.visit(d)

    visit_FunctionDef = visit_AsyncFunctionDef = _def

    def visit_Lambda(self, node):
        for d in [*node.args.defaults, *[k for k in node.args.kw_defaults if k]]:
            self.visit(d)

    def visit_If(self, node):
        t = node.test
        if isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__":
            for b in node.orelse:
                self.visit(b)
            return                                     # the `__main__` block runs as a script
        self.generic_visit(node)


def scan(app: pathlib.Path) -> list[str]:
    lazy = lazy_names(app)
    # a shim's own name is a module a script imports: `import promote_queue`
    out = []
    for p in _files(app):
        tree = _parse(p)
        if tree is None:
            continue
        v = _ImportTime(lazy, _modname(app, p), p.name == "__init__.py")
        v.visit(tree)
        out += [f"{p.relative_to(app)}:{ln}: {what}" for ln, what in v.hits]
    return sorted(set(out))


def test_the_lazy_names_are_derived_from_the_tree():
    lazy = lazy_names(APP)
    assert {"LOCK_DIR", "LOCK_HOST"} <= lazy["cytir.ops.write_lock"]
    assert "UA" in lazy["cytir.acquire.crawl"]
    assert len(lazy) >= 8


def test_no_module_reads_a_lazy_setting_at_import():
    hits = scan(APP)
    assert not hits, ("an import-time read of a lazily-resolved setting refuses by name on a box "
                      "with no operator settings; bind lazily (read `module.NAME` inside the "
                      "function that uses it):\n" + "\n".join(hits))


def test_the_scan_detects_the_patterns_the_review_found(tmp_path):
    app = tmp_path / "app"
    (app / "cytir" / "ops").mkdir(parents=True)
    (app / "scripts").mkdir()
    (app / "cytir" / "ops" / "lockmod.py").write_text(
        "def __getattr__(name):\n    if name in ('LOCK_DIR', 'LOCK_HOST'):\n        return 1\n"
        "    raise AttributeError(name)\n")
    (app / "scripts" / "bad_from.py").write_text("from cytir.ops.lockmod import LOCK_DIR\n")
    (app / "scripts" / "bad_attr.py").write_text(
        "from cytir.ops import lockmod as lm\nX = lm.LOCK_HOST\n")
    (app / "scripts" / "bad_default.py").write_text(
        "from cytir.ops import lockmod\ndef f(d=lockmod.LOCK_DIR):\n    return d\n")
    (app / "scripts" / "good.py").write_text(
        "from cytir.ops import lockmod\nfrom cytir.ops.lockmod import other\n"
        "def f():\n    return lockmod.LOCK_DIR\n"
        "G = lambda: lockmod.LOCK_HOST\n"
        "if __name__ == '__main__':\n    print(lockmod.LOCK_DIR)\n")
    hits = scan(app)
    assert [h.split(":")[0] for h in hits] == [
        "scripts/bad_attr.py", "scripts/bad_default.py", "scripts/bad_from.py"], hits
