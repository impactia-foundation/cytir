"""Task 1283 (census 1281, R1 remainder): no code path defaults to the pre-e9 arm shape.

`served` is selectable by name; it is never what a caller gets by saying nothing. This scans
`cytir/` for a `DEFAULT_*` constant, a keyword/parameter default, or an `argparse` `default=`
whose value is the bare string "served" or `SP.SERVED`/`SERVED`. Lists and tuples of profile
names (`PUBLIC_PROFILES`) are name lists, not defaults, and are allowed."""
import ast
import pathlib

import cytir
import pytest

pytestmark = pytest.mark.guard

_ROOT = pathlib.Path(cytir.__file__).parent


def _is_served(node) -> bool:
    if isinstance(node, ast.Constant) and node.value == "served":
        return True
    if isinstance(node, ast.Name) and node.id == "SERVED":
        return True
    return isinstance(node, ast.Attribute) and node.attr == "SERVED"


def _offences(tree) -> list[int]:
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if (n.value is not None and _is_served(n.value) and any(
                    isinstance(t, ast.Name) and t.id.upper().startswith("DEFAULT") for t in targets)):
                hits.append(n.lineno)
        elif isinstance(n, ast.keyword) and n.arg == "default" and _is_served(n.value):
            hits.append(n.value.lineno)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = n.args
            hits += [d.lineno for d in a.defaults + [d for d in a.kw_defaults if d] if _is_served(d)]
    return hits


def test_no_package_code_defaults_to_the_served_layout():
    bad = {}
    for f in sorted(_ROOT.rglob("*.py")):
        h = _offences(ast.parse(f.read_text(), str(f)))
        if h:
            bad[str(f.relative_to(_ROOT))] = h
    assert not bad, f"a default resolves to the pre-e9 `served` shape: {bad}"


def test_the_scan_sees_the_shapes_it_forbids():
    for src in ('DEFAULT_LAYOUT = "served"', "f(default='served')", "def g(x='served'): pass",
                "DEFAULT_PROFILE = SP.SERVED"):
        assert _offences(ast.parse(src)), src
    assert not _offences(ast.parse('PUBLIC_PROFILES = ("prod-e9", "served")'))
