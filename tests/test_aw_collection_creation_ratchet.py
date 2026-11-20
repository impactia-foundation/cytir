# Slice 8 step 1 (specs/aw-index-serve-gate.md): the ratchet that keeps "one collection
# creator" true. Every rule below has a second test that PLANTS the evasion in a temp
# file and asserts the rule fires, because round 1 of PR 530 had a ratchet that passed
# for four of the forms it claimed to catch.
"""No code in `cytir/` creates a Qdrant collection except `cytir.core.serving_profile`.
(The `pipeline/` half and its empty creator baseline were removed in task 2409.)"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

_APP = pathlib.Path(__file__).resolve().parents[1]
_AW = _APP / "cytir"
_CREATOR = _AW / "core" / "serving_profile.py"
#: A second, independent number. The baseline file may only shrink; raising BOTH this and
#: the file is a visible two-place edit in review.
# ---------------------------------------------------------------- the shared detectors
_CREATE_NAMES = ("create_collection", "recreate_collection")
# A collection NAME at the end of a path: ".../collections/%s", ".../collections/{x}",
# ".../collections/$NAME". Paths that go on (/points, /index, /snapshots) are not creates.
_NAME_FINAL = re.compile(r"/collections/(%s|\{[^{}/]*\}|\$\w+|\$\{\w+\})(?![\w/%{$])")
_PUT_WORD = re.compile(r"(?<![A-Za-z])PUT(?![A-Za-z])")


def _text_of(node) -> str:
    """The text of a string expression, dynamic parts as `{}`."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
    return ""


def _stmt_has_raw_put_create(stmt: ast.stmt) -> bool:
    """One statement holds a collection-NAME-final path and a PUT marker: a
    `method="PUT"` keyword, a `.put(` call, or a "PUT" string argument."""
    name_final = put = False
    for n in ast.walk(stmt):
        t = _text_of(n)
        if t and _NAME_FINAL.search(t):
            name_final = True
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute) and n.func.attr in ("put", "put_async"):
                put = True
            for k in n.keywords:
                if (k.arg == "method" and isinstance(k.value, ast.Constant)
                        and str(k.value.value).upper() == "PUT"):
                    put = True
        if isinstance(n, ast.Constant) and n.value == "PUT":
            put = True
    return name_final and put


def _raw_put_lines(tree: ast.AST):
    seen = set()
    for stmt in ast.walk(tree):
        if isinstance(stmt, (ast.Expr, ast.Assign, ast.AnnAssign, ast.Return, ast.AugAssign)):
            if _stmt_has_raw_put_create(stmt) and stmt.lineno not in seen:
                seen.add(stmt.lineno)
                yield stmt.lineno


def _sh_raw_put_lines(text: str):
    """Shell: join `\\`-continued lines, then a line with PUT and a name-final path."""
    joined, buf, start = [], "", 1
    for i, line in enumerate(text.splitlines(), 1):
        if not buf:
            start = i
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
            continue
        joined.append((start, buf + line))
        buf = ""
    if buf:
        joined.append((start, buf))
    for lineno, line in joined:
        if _PUT_WORD.search(line) and _NAME_FINAL.search(line):
            yield lineno


# --------------------------------------------------------------------- cytir/ ratchet
def _aw_import_names(tree: ast.AST) -> set[str]:
    """Names bound by an import of an `aw` / `pipeline` module: the module aliases through
    which the (shim) creators are called, e.g. `SC`, `SP`, `CS`, `apply`."""
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] in ("cytir", "pipeline", ""):
            names |= {a.asname or a.name for a in n.names}
        elif isinstance(n, ast.Import):
            names |= {(a.asname or a.name).split(".")[0] for a in n.names
                      if a.name.split(".")[0] in ("cytir", "pipeline")}
    return names


def aw_violations(path: pathlib.Path) -> list[str]:
    """Why `path` creates a collection other than through the one creator. Empty = clean."""
    text = path.read_text()
    tree = ast.parse(text)
    modules = _aw_import_names(tree)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr in _CREATE_NAMES:
                # `SP.create_collection(client, ...)` is a call through an aw module (a shim
                # or the creator). `client.create_collection(...)`, `self.c.create_collection`,
                # `client.http.collections_api.create_collection` are Qdrant's own API.
                if not (isinstance(f.value, ast.Name) and f.value.id in modules):
                    out.append(f"{n.lineno}: {ast.unparse(f)}(...) is Qdrant's own create call")
            if isinstance(f, ast.Name) and f.id == "getattr" and len(n.args) >= 2:
                a = n.args[1]
                if isinstance(a, ast.Constant) and a.value in _CREATE_NAMES:
                    out.append(f"{n.lineno}: getattr(..., {a.value!r})")
        if isinstance(n, ast.Attribute) and n.attr == "collections_api":
            out.append(f"{n.lineno}: the low-level HTTP API (`collections_api`)")
    out += [f"{ln}: a raw HTTP PUT to /collections/<name>" for ln in _raw_put_lines(tree)]
    return sorted(set(out), key=lambda s: int(s.split(":")[0]))


def test_no_collection_is_created_in_aw_outside_the_one_creator():
    bad = {}
    for p in _AW.rglob("*.py"):
        if p == _CREATOR:
            continue
        v = aw_violations(p)
        if v:
            bad[str(p.relative_to(_APP))] = v
    assert bad == {}


def test_the_creator_is_exactly_one_raw_call():
    tree = ast.parse(_CREATOR.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "create_collection"
             and not (isinstance(n.func.value, ast.Name) and n.func.value.id == "SP")]
    assert len(calls) == 1


@pytest.mark.parametrize("label,code", [
    ("the creator's own form: **kw", "def f(c, kw):\n    c.create_collection(**kw)\n"),
    ("positional vectors_config",
     "def f(c):\n    c.create_collection('x', {'dense': 1})\n"),
    ("keyword vectors_config", "def f(c):\n    c.create_collection(collection_name='x', vectors_config={})\n"),
    ("recreate_collection", "def f(c):\n    c.recreate_collection('x', vectors_config={})\n"),
    ("the low-level http api",
     "def f(c):\n    c.http.collections_api.create_collection('x', {})\n"),
    ("getattr", "def f(c):\n    getattr(c, 'create_collection')('x')\n"),
    ("a raw PUT, % format",
     "def f(u, b):\n    _req('%s/collections/%s' % (u, 'x'), b, method='PUT')\n"),
    ("a raw PUT, f-string, requests.put",
     "import requests\ndef f(u, n, b):\n    requests.put(f'{u}/collections/{n}', json=b)\n"),
    ("a raw PUT, f-string into a variable, method kw",
     "def f(u, n, b):\n    r = _http(f'{u}/collections/{n}', method='PUT', body=b)\n"),
    ("a raw PUT, positional verb",
     "def f(u, b):\n    call('PUT', '%s/collections/%s' % (u, 'x'), b)\n"),
])
def test_aw_ratchet_fires_on_every_evasion_form(tmp_path, label, code):
    p = tmp_path / "evader.py"
    p.write_text(code)
    assert aw_violations(p), f"the cytir/ ratchet did not fire on: {label}"


@pytest.mark.parametrize("label,code", [
    ("a shim called through an aw module",
     "from cytir.core import serving_profile as SP\ndef f(c, p):\n    SP.create_collection(c, 'x', p)\n"),
    ("a PUT of points", "def f(u, b):\n    _req('%s/collections/%s/points' % (u, 'x'), b, method='PUT')\n"),
    ("a GET of the collection", "def f(u):\n    return _req('%s/collections/%s' % (u, 'x'))\n"),
])
def test_aw_ratchet_stays_quiet_on_legitimate_forms(tmp_path, label, code):
    p = tmp_path / "ok.py"
    p.write_text(code)
    assert aw_violations(p) == [], label



