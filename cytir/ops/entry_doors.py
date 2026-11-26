"""Static scan of every non-`aw` way to run a module under `cytir/` (task 2894, refactor-to-8 T3).

CONCERN: the package has a front door (`cytir/cli.py`), yet ~195 modules carry their own `def main` or
`if __name__ == "__main__"` block. Each is a second way to run something that a stranger cannot find.
CHOICE: derive only what is mechanical, and declare the rest in `data/entry-doors.v1.json`:
  - door         the module has a `def main` at module level and/or an `if __name__` block;
  - is_aw_verb   `cytir/cli.py` names the module (dotted string, `import`, or `from ... import`), i.e.
                 the CLI dispatch table reaches it. A verb is not a registry row;
  - self_start   the module builds a command list `[sys.executable, <its own file or own module>, ...]`
                 and spawns it (a child-handler: it restarts itself as a subprocess);
  - path_spawned_by  another file under cytir/ holds a spawn call and joins this file name onto a directory (`dir / "x.py"`): a path-spawned
                 handler, also a child-handler (started as `python <path>/x.py`, so no import names it);
  - block_lines  lines of the `def main` + `if __name__` blocks.
SCOPE: `cytir/**/*.py` only (files outside `cytir/` with a `__main__` block are not scanned; see the data file's `scope`).
A door is: `def main` / `async def main`, or a `__name__ == "__main__"` compare (either side, inside `and`, `if`,
`try`, `with`) at module level, or any `__main__.py`. `is_aw_verb` is true for ANY `aw.*` dotted name that
`cytir/cli.py` holds, a helper import included. `self_start` follows `python = sys.executable`-style variables and
`Path(__file__)` forms; a path counts as the module itself only when it rebuilds the SAME file name (a sibling
or the directory does not).
Kind, started_by and reason are DECLARED in the data file; a heuristic got them 5-30 % wrong (task 2891).
`tests/operator/test_aw_entry_door_registry.py` fails on an undeclared door, a stale row and a ceiling
that grew. Static only: no network, no imports of the scanned modules.
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import subprocess
import sys

APP = pathlib.Path(__file__).resolve().parents[2]
REGISTRY = APP / "data" / "entry-doors.v1.json"
KINDS = ("operator-tool", "live-unit-door", "child-handler", "shim-target", "guard")
CEILINGS = ("non_verb_doors", "docs_only", "child_handlers_without_spawn_test", "live_unit_doors", "unverified_rows")
_SPAWN = frozenset({"run", "Popen", "call", "check_call", "check_output", "execv", "execve", "execvp", "execvpe",
                    "spawnv", "spawnl", "create_subprocess_exec"})


def dotted(rel: str) -> str:
    return rel[:-3].replace("/", ".").removesuffix(".__init__")


def _is_main_compare(n: ast.AST) -> bool:
    """`__name__ == "__main__"` with the name or the constant on either side, or `__name__ in ("__main__",)`."""
    if (isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], ast.In)
            and isinstance(n.left, ast.Name) and n.left.id == "__name__"
            and isinstance(n.comparators[0], (ast.Tuple, ast.List, ast.Set))):
        return any(isinstance(x, ast.Constant) and x.value == "__main__" for x in n.comparators[0].elts)
    if not (isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], ast.Eq)):
        return False
    pair = [n.left, n.comparators[0]]
    has_name = any(isinstance(x, ast.Name) and x.id == "__name__" for x in pair)
    has_const = any(isinstance(x, ast.Constant) and x.value == "__main__" for x in pair)
    return has_name and has_const


def _has_main_compare(n: ast.AST) -> bool:
    return any(_is_main_compare(x) for x in ast.walk(n))


def _is_main_guard(n: ast.AST) -> bool:
    """A module-level statement whose test (or whole expression) holds the `__main__` compare:
    `if __name__ == ...`, `if x and __name__ == ...`, `__name__ == ... and f()`."""
    if isinstance(n, ast.If):
        return _has_main_compare(n.test)
    return isinstance(n, ast.Expr) and _has_main_compare(n.value)


def block_nodes(tree: ast.Module) -> list[ast.stmt]:
    """`def main` / `async def main` and `__main__` guards at module level, also inside `if`/`try`/`with`."""
    out: list[ast.stmt] = []

    def walk(body: list[ast.stmt]) -> None:
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "main":
                out.append(n)
            elif _is_main_guard(n):
                out.append(n)
            elif isinstance(n, (ast.If, ast.With, ast.AsyncWith, ast.Try, ast.For, ast.While)):
                for field in ("body", "orelse", "finalbody"):
                    walk(getattr(n, field, []) or [])
                for h in getattr(n, "handlers", []) or []:
                    walk(h.body)
    walk(tree.body)
    return out


_INTERP_NAME = re.compile(r"^python[\d.]*$")
_PATH_FUNCS = frozenset({"str", "Path", "PurePath", "abspath", "realpath", "fspath"})
_PATH_METHODS = frozenset({"resolve", "absolute", "as_posix", "expanduser"})
_EXEC = frozenset({"execv", "execve", "execvp", "execvpe", "execl", "execlp", "spawnv", "spawnl"})


class _Names:
    """Simple name -> value map over the module (assignments and parameter defaults): enough to follow
    `python = sys.executable` and `me = Path(__file__)`. Last binding wins; no flow analysis."""

    def __init__(self, tree: ast.Module) -> None:
        self.m: dict[str, ast.AST] = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                self.m[n.targets[0].id] = n.value
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
                self.m[n.target.id] = n.value
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = n.args
                pos = a.posonlyargs + a.args
                for arg, d in zip(pos[len(pos) - len(a.defaults):], a.defaults):
                    self.m.setdefault(arg.arg, d)
                for arg, d in zip(a.kwonlyargs, a.kw_defaults):
                    if d is not None:
                        self.m.setdefault(arg.arg, d)

    def get(self, n: ast.AST, seen: frozenset = frozenset()) -> ast.AST:
        while isinstance(n, ast.Name) and n.id in self.m and n.id not in seen and n.id != "__file__":
            seen = seen | {n.id}
            n = self.m[n.id]
        return n


def _call_name(c: ast.Call) -> str:
    return c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "")


def _is_sys_executable(n: ast.AST) -> bool:
    return (isinstance(n, ast.Attribute) and n.attr == "executable"
            and isinstance(n.value, ast.Name) and n.value.id == "sys")


def _is_sys_argv(n: ast.AST) -> bool:
    return isinstance(n, ast.Attribute) and n.attr == "argv" and isinstance(n.value, ast.Name) and n.value.id == "sys"


def _is_interp(n: ast.AST, names: _Names) -> bool:
    n = names.get(n)
    return _is_sys_executable(n) or (isinstance(n, ast.Constant) and isinstance(n.value, str)
                                     and bool(_INTERP_NAME.match(n.value)))


def _is_self_file(n: ast.AST, names: _Names, leaf: str, rel: str, _d: int = 0) -> bool:
    """The module's OWN file: `__file__`, `str/Path/resolve/abspath(...)` of it, `sys.argv[0]`, or a path that
    rebuilds the SAME file name (`Path(__file__).parent / "<own leaf>"`, `.with_name("<own leaf>")`,
    `os.path.join(dirname(__file__), "<own leaf>")`). A sibling (`.parent / "other.py"`) or the bare directory
    (`.parent`, `.parents[1]`) is NOT the module itself."""
    if _d > 8:
        return False
    n = names.get(n)
    if isinstance(n, ast.Name):
        return n.id == "__file__"
    if isinstance(n, ast.Subscript):
        return _is_sys_argv(n.value) and isinstance(n.slice, ast.Constant) and n.slice.value == 0
    if isinstance(n, ast.Call):
        nm = _call_name(n)
        if nm in _PATH_FUNCS and n.args:
            return _is_self_file(n.args[0], names, leaf, rel, _d + 1)
        if nm in _PATH_METHODS and isinstance(n.func, ast.Attribute):
            return _is_self_file(n.func.value, names, leaf, rel, _d + 1)
        if nm == "with_name" and isinstance(n.func, ast.Attribute) and n.args:
            return _own_leaf(n.args[0], leaf, rel) and (_is_self_file(n.func.value, names, leaf, rel, _d + 1)
                                                   or _is_self_dir(n.func.value, names, leaf, rel, _d + 1))
        if nm == "join" and len(n.args) >= 2:
            return _own_leaf(n.args[-1], leaf, rel) and _is_self_dir(n.args[0], names, leaf, rel, _d + 1)
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
        return _own_leaf(n.right, leaf, rel) and _is_self_dir(n.left, names, leaf, rel, _d + 1)
    return False


def _own_leaf(n: ast.AST, leaf: str, rel: str) -> bool:
    return isinstance(n, ast.Constant) and isinstance(n.value, str) and (
        n.value == leaf or n.value == rel or n.value.endswith("/" + leaf))


def _is_self_dir(n: ast.AST, names: _Names, leaf: str, rel: str, _d: int = 0) -> bool:
    """A directory derived from the module's own file: `<self>.parent`, `<self>.parents[k]`, `dirname(<self>)`."""
    if _d > 8:
        return False
    n = names.get(n)
    if isinstance(n, ast.Attribute) and n.attr == "parent":
        return _is_self_file(n.value, names, leaf, rel, _d + 1) or _is_self_dir(n.value, names, leaf, rel, _d + 1)
    if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == "parents":
        return _is_self_file(n.value.value, names, leaf, rel, _d + 1)
    if isinstance(n, ast.Call) and _call_name(n) == "dirname" and n.args:
        return _is_self_file(n.args[0], names, leaf, rel, _d + 1) or _is_self_dir(n.args[0], names, leaf, rel, _d + 1)
    if isinstance(n, ast.Call) and _call_name(n) in ("Path", "resolve", "absolute") and (n.args or isinstance(n.func, ast.Attribute)):
        inner = n.args[0] if n.args else n.func.value
        return _is_self_dir(inner, names, leaf, rel, _d + 1)
    return False


def _argv_starts_self(lst: list[ast.AST], names: _Names, mod: str, leaf: str, rel: str) -> bool:
    """An argv list that runs the interpreter on this module: `[interp, ..., <own file>, ...]` or
    `[interp, "-m", <own dotted name | __name__>, ...]`. `sys.executable` is not required literally: a
    variable that resolves to it, or `python`/`python3`, is the interpreter."""
    if not lst or not _is_interp(lst[0], names):
        return False
    for i, e in enumerate(lst[1:], 1):
        if _is_self_file(e, names, leaf, rel):
            return True
        prev = names.get(lst[i - 1])
        if isinstance(prev, ast.Constant) and prev.value == "-m":
            if (isinstance(e, ast.Name) and e.id == "__name__") or (
                    isinstance(e, ast.Constant) and e.value == mod):
                return True
    return False


def self_starts(tree: ast.Module, mod: str, rel: str) -> bool:
    """True when the module spawns itself as a child process. The module must contain a spawn call and one of:
    an argv list `[interp, ..., <own file>|-m <own name>, ...]` (see `_argv_starts_self`); an `exec*` call that
    re-runs `sys.argv` under the interpreter; an f-string shell command with `{__file__}` (or another own-file
    form) and `python`/`sys.executable` in it."""
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)]
    if not any(_call_name(c) in _SPAWN or _call_name(c) in _EXEC for c in calls):
        return False
    names, leaf = _Names(tree), rel.rsplit("/", 1)[-1]
    for n in ast.walk(tree):
        if isinstance(n, (ast.List, ast.Tuple)) and _argv_starts_self(list(n.elts), names, mod, leaf, rel):
            return True
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add) and isinstance(n.left, ast.List) \
                and n.left.elts and _is_interp(n.left.elts[0], names) and _is_sys_argv(n.right):
            return True            # os.execv(sys.executable, [sys.executable] + sys.argv)
        if isinstance(n, ast.JoinedStr):
            fv = [v.value for v in n.values if isinstance(v, ast.FormattedValue)]
            text = "".join(v.value for v in n.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
            if any(_is_self_file(v, names, leaf, rel) for v in fv) and (
                    "python" in text or any(_is_interp(v, names) for v in fv)):
                return True
    return False


def cli_reached(cli_path: pathlib.Path) -> set[str]:
    """Dotted `aw...` module names that `cytir/cli.py` names: string constants, imports, from-imports."""
    tree = ast.parse(cli_path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            v = n.value
            if v.startswith("cytir.") and all(p.isidentifier() for p in v.split(".")):
                out.add(v)
        elif isinstance(n, ast.ImportFrom) and n.module and (n.module == "cytir" or n.module.startswith("cytir.")) and not n.level:
            out.add(n.module)
            out.update(f"{n.module}.{a.name}" for a in n.names)
        elif isinstance(n, ast.Import):
            out.update(a.name for a in n.names if a.name.startswith("cytir."))
    return out


def path_spawners(tree: ast.Module) -> set[str]:
    """File names (`x.py`) that this module joins onto a path (`dir / "x.py"`) AND could spawn: it holds a spawn call.
    A caller that builds `release_dir / pkg / "gates" / "x.py"` and runs it names the file this way."""
    if not any(isinstance(c, ast.Call) and (_call_name(c) in _SPAWN or _call_name(c) in _EXEC) for c in ast.walk(tree)):
        return set()
    return {n.right.value.rsplit("/", 1)[-1] for n in ast.walk(tree)
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) and isinstance(n.right, ast.Constant)
            and isinstance(n.right.value, str) and n.right.value.endswith(".py")}


def scan_doors(app: pathlib.Path = APP) -> dict[str, dict]:
    """{dotted module: {"path", "has_def_main", "has_name_guard", "block_lines", "is_aw_verb", "self_start",
    "path_spawned_by"}}. `path_spawned_by`: other files under cytir/ that spawn this one by its file name (a
    path-spawned handler: started as `python <path>/x.py`, not as `-m`, so no import line names it)."""
    base = app / "cytir"
    reached = cli_reached(base / "cli.py") if (base / "cli.py").is_file() else set()
    out: dict[str, dict] = {}
    parsed: list[tuple[pathlib.Path, str, str, ast.Module]] = []
    spawners: dict[str, set[str]] = {}
    for p in sorted(base.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = str(p.relative_to(app))
        text = p.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            raise RuntimeError(f"entry-door scan: cannot parse {rel}")
        parsed.append((p, rel, text, tree))
        for name in path_spawners(tree):
            spawners.setdefault(name, set()).add(rel)
    for p, rel, text, tree in parsed:
        blocks = block_nodes(tree)
        if not blocks and p.name != "__main__.py":     # `python -m cytir.x` runs it: a door with no guard or `def main`
            continue
        mod = dotted(rel)
        out[mod] = {
            "path": rel,
            "has_def_main": any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in blocks),
            "has_name_guard": any(_is_main_guard(n) for n in blocks),
            "dunder_main_file": p.name == "__main__.py",
            "block_lines": sum(n.end_lineno - n.lineno + 1 for n in blocks) if blocks else len(text.splitlines()),
            "is_aw_verb": mod in reached,
            "self_start": self_starts(tree, mod, rel),
            "path_spawned_by": sorted(spawners.get(p.name, set()) - {rel}),
        }
    return out


def non_verb_doors(app: pathlib.Path = APP) -> dict[str, dict]:
    return {m: d for m, d in scan_doors(app).items() if not d["is_aw_verb"]}


def load_registry(path: pathlib.Path = REGISTRY) -> dict:
    return json.loads(path.read_text())


def registry_rows(reg: dict) -> dict[str, dict]:
    return {r["module"]: r for r in reg["rows"]}


def _docs_only(r: dict) -> bool:
    sb = [s for s in r.get("started_by", []) if s not in ("none", "unknown")]
    return not sb or all(s.startswith("doc") for s in sb)


def measures(reg: dict) -> dict[str, int]:
    rows = reg["rows"]
    return {
        "non_verb_doors": len(rows),
        "docs_only": sum(1 for r in rows if _docs_only(r)),
        "child_handlers_without_spawn_test": sum(
            1 for r in rows if r["kind"] == "child-handler" and not r.get("test_that_starts_it")),
        "live_unit_doors": sum(1 for r in rows if r["kind"] == "live-unit-door"),
        "unverified_rows": sum(1 for r in rows if not r.get("verified")),
    }


SCAN_ROOTS = ("cytir",)
CUT_MARKER = "PUBLIC-CUT-MANIFEST.json"
_PATH_CATS = frozenset({"unit", "script", "tasks-script", "test", "doc", "aw-subprocess"})
_SPAWN_WORDS = re.compile(r"subprocess|Popen|check_call|check_output|runpy|os\.exec|\.run\(")


def _path_ok(app: pathlib.Path, rel: str) -> bool:
    """The path exists, or git tracks it (a lean worktree prunes tracked data trees: tracked still counts)."""
    if (app / rel).exists():
        return True
    try:
        out = subprocess.run(["git", "ls-files", "--", rel], cwd=app, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(out.stdout.strip())


def _test_spawns(text: str, mod: str, rel: str) -> bool:
    """The test file runs a process AND names the module (dotted, path, file name, or `from pkg import leaf`)."""
    if not _SPAWN_WORDS.search(text):
        return False
    pkg, _, leaf = mod.rpartition(".")
    return any(x in text for x in (mod, rel, leaf + ".py", f"from {pkg} import {leaf}", f"{pkg}.{leaf}"))


def _row_field_problems(app: pathlib.Path, rows: dict[str, dict], found: dict[str, dict]) -> list[str]:
    msgs: list[str] = []
    for m, r in sorted(rows.items()):
        for s in r.get("started_by", []):
            cat, sep, val = s.partition(":")
            if sep and cat in _PATH_CATS and val != "unknown" and not _path_ok(app, val):
                msgs.append(f"{m}: started_by `{s}` names a path that does not exist in this tree: fix it, or set "
                            f"the value to `unknown` (never invent one).")
        t = r.get("test_that_starts_it")
        if t is None:
            continue
        if not _path_ok(app, t):
            msgs.append(f"{m}: test_that_starts_it `{t}` is not a file in this tree: fix it or set it to null.")
        elif r.get("kind") == "child-handler" and m in found and not _test_spawns(
                (app / t).read_text(encoding="utf-8", errors="replace"), m, found[m]["path"]):
            msgs.append(f"{m}: test_that_starts_it `{t}` does not spawn the module (needs a subprocess call that "
                        f"names it): fix it or set it to null.")
    return msgs


def problems(app: pathlib.Path = APP, reg: dict | None = None) -> list[str]:
    """Every way the registry and the tree disagree, as sentences that say what to do."""
    reg = reg if reg is not None else load_registry(app / "data" / "entry-doors.v1.json")
    found = non_verb_doors(app)
    rows = registry_rows(reg)
    msgs: list[str] = []
    for m in sorted(set(found) - set(rows)):
        msgs.append(f"{m}: a new entry door (`def main` / `if __name__` block, not reached from cytir/cli.py). Prefer an "
                    f"`aw` verb; else add a row to data/entry-doors.v1.json (see `cyt ops entry-doors --write-candidates`).")
    for m in sorted(set(rows) - set(found)):
        msgs.append(f"{m}: stale row. The module has no door any more (or is now an `aw` verb): remove the row "
                    f"and lower the ceilings in the same PR.")
    for m, r in sorted(rows.items()):
        if r.get("kind") not in KINDS:
            msgs.append(f"{m}: kind {r.get('kind')!r} is not one of {KINDS}")
        if not str(r.get("reason", "")).strip():
            msgs.append(f"{m}: row has no reason")
        if m in found:
            spawn = found[m]["self_start"]
            if spawn and r.get("kind") != "child-handler":
                msgs.append(f"{m}: the module starts itself as a child but the row says kind={r.get('kind')!r}: "
                            f"declare it a child-handler (a kind may not hide a self-starter).")
            if not spawn and r.get("kind") == "child-handler" and not found[m]["path_spawned_by"]:
                msgs.append(f"{m}: row says child-handler but the scanner finds no self-start and no file that spawns it "
                            f"by path: fix the kind.")
    msgs += _row_field_problems(app, rows, found)
    got, ceil = measures(reg), reg.get("ceilings", {})
    for k in CEILINGS:
        if got[k] > ceil.get(k, -1):
            msgs.append(f"ceiling {k}: {got[k]} > {ceil.get(k)}. Ceilings only shrink.")
    return msgs


def candidates(app: pathlib.Path = APP, reg: dict | None = None) -> list[dict]:
    if reg is None:
        path = app / "data" / "entry-doors.v1.json"
        reg = load_registry(path) if path.is_file() else {"rows": []}
    rows = registry_rows(reg)
    return [{"module": m, "kind": "child-handler" if d["self_start"] or d["path_spawned_by"] else "operator-tool", "started_by": [],
             "reason": "FILL IN: one line, why this door exists", "test_that_starts_it": None, "verified": False}
            for m, d in sorted(non_verb_doors(app).items()) if m not in rows]


def main(argv: list[str] | None = None, prog: str = "cyt ops entry-doors") -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="exit 1 if the tree and data/entry-doors.v1.json disagree")
    g.add_argument("--list", action="store_true", help="print every door the scanner finds")
    g.add_argument("--write-candidates", action="store_true",
                   help="print rows for undeclared doors (never rewrites the data file)")
    ap.add_argument("--root", default=str(APP), help="app root to scan (default: this checkout)")
    a = ap.parse_args(argv)
    app = pathlib.Path(a.root)
    reg_path = app / "data" / "entry-doors.v1.json"
    have_reg = reg_path.is_file()
    if a.write_candidates:
        print(json.dumps(candidates(app), indent=1))
        return 0
    if not have_reg and a.check:
        if (app / CUT_MARKER).is_file():       # an exported public cut: the registry is master-side by design
            print("entry-doors: registry file not shipped in this tree; nothing to check")
            return 0
        print(f"entry-doors: ERROR: {reg_path} is missing in a full checkout. Restore it from git.")
        return 2
    if a.list:
        reg = load_registry(reg_path) if have_reg else None
        total = len(reg["rows"]) if reg else None
        unv = measures(reg)["unverified_rows"] if reg else None
        print(f"# scanned {', '.join(r + '/**/*.py' for r in SCAN_ROOTS)}; unverified {unv if reg else 'n/a'}/{total if reg else 'n/a'}")
        for m, d in sorted(scan_doors(app).items()):
            print(f"{m}\tverb={int(d['is_aw_verb'])}\tself_start={int(d['self_start'])}\tblock_lines={d['block_lines']}")
        return 0
    msgs = problems(app)
    for m in msgs:
        print("entry-doors: " + m)
    if not msgs:
        reg = load_registry(reg_path)
        print(f"entry-doors: ok ({len(non_verb_doors(app))} non-verb doors declared; scanned "
              f"{', '.join(r + '/**/*.py' for r in SCAN_ROOTS)}; unverified {measures(reg)['unverified_rows']}/{len(reg['rows'])})")
    return 1 if msgs else 0


if __name__ == "__main__":
    sys.exit(main())
