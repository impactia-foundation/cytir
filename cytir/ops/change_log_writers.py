"""Static scan of every code location that can mutate a Qdrant collection (task 2849, ruling 75).

CONCERN: ruling 75 says no process mutates the twin or prod except through the drain, and a
legitimate other writer calls the change-log writer (`cytir.index.change_log`). Nothing pinned the
list of writers, so a new one could bypass the log unseen. CHOICE: scan the tree for write calls,
key each hit by (path, function), and compare with the declared registry
`data/change-log-writers.v1.json`. `tests/operator/test_aw_change_log_writer_registry.py` fails on an
unregistered site, on a stale row, and when the `direct-unlogged` count grows.

Static only: no Qdrant, no network. A hit is one of
  - an SDK call `.upsert / .set_payload / .overwrite_payload / .delete_payload / .clear_payload /
    .update_vectors / .delete_vectors / .upload_* / .batch_update_points`, or `.delete(points_selector=...)`;
  - a string literal naming a write path: `/points/payload`, `/points/vectors`, `/points/delete`,
    `/points/batch`, `/points?...` (qdrant's PUT upsert), or a `/points` PUT;
  - a call to a wrapper that writes for its caller (`WRAPPERS`);
  - in shell: a line with a write path of the same forms.

SCOPE: POINT writes only (rows added, replaced, patched or deleted). Collection-level mutations are
out of scope and not scanned here, by name: `serving_profile.create_collection(on_exists="recreate")`
which calls `delete_collection` (serving_profile.py:481, 7 callers incl. `source_cards.py:691`),
`staging_collection_retention.py:132`, `qdrant_hop_chain.py:291` (snapshot recover), and
`alias_swap.sh:111` (alias swap). The change log records ids, not collections; a collection rebuild is
a `rebase` (spec 3.5).
"""
from __future__ import annotations

import ast
import json
import pathlib
import re

APP = pathlib.Path(__file__).resolve().parents[2]
REGISTRY = APP / "data" / "change-log-writers.v1.json"
ROOTS = ("cytir", "server", "deploy")
CLASSES = ("drain-logged", "direct-unlogged", "never-live")

SDK_ATTRS = frozenset({
    "upsert", "set_payload", "overwrite_payload", "delete_payload", "clear_payload", "update_vectors",
    "delete_vectors", "upload_points", "upload_collection", "upload_records", "batch_update_points"})
#: Helpers that write for their caller. Each caller is its own hit, so a new writer that goes
#: through a helper is still caught.
WRAPPERS = frozenset({
    "delete_points_verified", "set_payload_batch",
    # the drain logs FOR these, so every other caller is a writer of its own (review 2853 finding 1):
    "Superseder", "delete_superseded", "QdrantPatchClient"})
#: A bare `upsert(...)` call is `promote_sealed_pack.upsert` (the sealed-pack PUT), a wrapper too.
_WRITE_PATH = re.compile(r"/points/(payload|vectors|delete|batch)|/points\?")
_POINTS_END = re.compile(r"/points(?![\w/])")


def _strings(node: ast.AST):
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            yield n.value


def _has_put(call: ast.Call) -> bool:
    for n in ast.walk(call):
        if isinstance(n, ast.Constant) and n.value == "PUT":
            return True
        if isinstance(n, ast.Attribute) and n.attr in ("put", "put_async"):
            return True
    return False


def _call_hits(call: ast.Call) -> list[str]:
    out: list[str] = []
    f = call.func
    name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
    if isinstance(f, ast.Attribute) and name in SDK_ATTRS:
        out.append(name)
    elif isinstance(f, ast.Attribute) and name == "delete" and (
            any(k.arg == "points_selector" for k in call.keywords) or len(call.args) >= 2):
        out.append("delete")
    read_only_ctor = name == "Superseder" and any(
        k.arg == "dry_run" and isinstance(k.value, ast.Constant) and k.value.value is True
        for k in call.keywords)       # a dry-run Superseder only observes: it deletes nothing
    if (name in WRAPPERS or (isinstance(f, ast.Name) and name == "upsert")) and not read_only_ctor:
        out.append(f"wrapper:{name}")
    texts = list(_strings(call))
    if any(_WRITE_PATH.search(t) for t in texts):
        out.append("http-write")
    elif any(_POINTS_END.search(t) for t in texts) and _has_put(call):
        out.append("http-put")
    return out


def scan_python(source: str) -> list[tuple[str, int, str]]:
    """(function, line, kind) for each write site in one module. `function` is dotted for nested
    defs and `<module>` outside any def. A call is reported once (its innermost call wins)."""
    tree = ast.parse(source)
    hits: list[tuple[str, int, str]] = []

    def visit(node: ast.AST, fn: str):
        for child in ast.iter_child_nodes(node):
            f = fn
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                f = child.name if fn == "<module>" else f"{fn}.{child.name}"
            elif isinstance(child, ast.ClassDef):
                f = child.name if fn == "<module>" else f"{fn}.{child.name}"
            if isinstance(child, ast.Call):
                own = _call_hits(child)
                # a nested call that holds the same string must not double-report: report the
                # outermost call, then skip hits that only repeat in its children
                for k in own:
                    hits.append((f, child.lineno, k))
            visit(child, f)

    visit(tree, "<module>")
    # an inner call whose strings are a subset of an outer hit repeats the same site: collapse by
    # (function, line, kind)
    return sorted(set(hits), key=lambda h: (h[1], h[0], h[2]))


_SH_WRITE = re.compile(r"/points/(payload|vectors|delete|batch)|/points\?")
_SH_PUT = re.compile(r"-X\s+PUT|\bPUT\b")


def scan_shell(text: str) -> list[tuple[str, int, str]]:
    """Shell hits: a (continuation-joined) line with a write path. Function = nearest `name()` above."""
    out: list[tuple[str, int, str]] = []
    func = "<script>"
    buf, start = "", 1
    for i, raw in enumerate(text.splitlines(), 1):
        m = re.match(r"\s*(?:function\s+)?([A-Za-z_][\w-]*)\s*\(\)\s*\{?", raw)
        if m and not buf:
            func = m.group(1)
        if not buf:
            start = i
        if raw.rstrip().endswith("\\"):
            buf += raw.rstrip()[:-1] + " "
            continue
        line, buf = buf + raw, ""
        if line.lstrip().startswith("#"):
            continue
        if _SH_WRITE.search(line) or (_SH_PUT.search(line) and re.search(r"/points(?![\w/])", line)):
            out.append((func, start, "shell-http-write"))
    return out


def scan_tree(app: pathlib.Path = APP) -> dict[tuple[str, str], list[dict]]:
    """{(relpath, function): [{"line", "kind"}, ...]} over `cytir/`, `server/`, `deploy/`."""
    found: dict[tuple[str, str], list[dict]] = {}
    for root in ROOTS:
        base = app / root
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix not in (".py", ".sh") or "__pycache__" in p.parts:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            try:
                hits = scan_python(text) if p.suffix == ".py" else scan_shell(text)
            except SyntaxError:
                raise RuntimeError(f"change-log writer scan: cannot parse {p}")
            for fn, line, kind in hits:
                found.setdefault((str(p.relative_to(app)), fn), []).append({"line": line, "kind": kind})
    return found


def load_registry(path: pathlib.Path = REGISTRY) -> dict:
    return json.loads(path.read_text())


def registry_keys(reg: dict) -> dict[tuple[str, str], dict]:
    return {(r["path"], r["function"]): r for r in reg["rows"]}


def counts(reg: dict) -> dict[str, int]:
    out = {c: 0 for c in CLASSES}
    for r in reg["rows"]:
        out[r["class"]] += 1
    return out


def _call_names(node: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


#: the change-log calls a `logged_by` function may rely on (all exist: `cytir/index/drain/changelog.py`,
#: `cytir/index/change_log.py`); `CL.record_landing` is the attribute call on the change_log module alias.
LANDING_LOGS = ("_log_landing", "_log_sealed_landing", "_log_resumed_landing", "record_rebase")
#: side function -> the transaction function `drain` calls (its nested `apply_target` runs the side).
#: `logged_by_problem` checks the chain with `ast`: the wrapper is defined in the row's own file and
#: calls the head (a nested def counts: that is where the call sits).
LIVE_APPLY_OF = {"apply_side": "live_apply", "live_apply": "live_apply",
                 "apply_content_pack_side": "live_apply_content_pack"}


def _own_calls(fn: ast.FunctionDef) -> list[ast.Call]:
    """Calls in `fn`'s own body: not inside a nested def / lambda / class, not in a constant-false branch."""
    out: list[ast.Call] = []

    def visit(n: ast.AST) -> None:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            return
        if isinstance(n, ast.If) and isinstance(n.test, ast.Constant):
            for c in (n.body if n.test.value else n.orelse):
                visit(c)
            return
        if isinstance(n, ast.While) and isinstance(n.test, ast.Constant) and not n.test.value:
            return
        if isinstance(n, ast.Call):
            out.append(n)
        for c in ast.iter_child_nodes(n):
            visit(c)

    for stmt in fn.body:
        visit(stmt)
    return out


def _is_landing_log(f: ast.expr) -> bool:
    if isinstance(f, ast.Attribute):
        if f.attr == "record_landing":
            return isinstance(f.value, ast.Name) and f.value.id in ("CL", "change_log")
        return f.attr in LANDING_LOGS
    return getattr(f, "id", "") in LANDING_LOGS


def _wrapper_chain_problem(row: dict, root: pathlib.Path, wrapper: str, head: str) -> str | None:
    """Why `wrapper` (in the row's own file) is not a transaction that runs `head`, or None."""
    f = root / row["path"]
    if not f.is_file():
        return f"{row['path']}: no such file"
    tops = [n for n in ast.parse(f.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == wrapper]
    if not tops:
        return f"{row['path']}: no wrapper {wrapper} (the transaction of {head})"
    fn = tops[0]
    own = _own_calls(fn)
    # a nested def of the wrapper counts only when the wrapper's own calls name it (called, or passed as an argument)
    refs = {x.id for c in own for x in ast.walk(c) if isinstance(x, ast.Name)}
    nested = [n for n in fn.body if isinstance(n, ast.FunctionDef) and n.name in refs]
    called = {c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "")
              for c in own + [c for d in nested for c in _own_calls(d)]}
    return None if head in called else f"{wrapper} in {row['path']} does not call {head}"


def logged_by_problem(row: dict, root: pathlib.Path = APP) -> str | None:
    """Why a drain-logged row's optional `logged_by` claim does not hold, or None.

    The claim: the writer in this row is logged by `logged_by.function` of `logged_by.path`. It holds
    when that function exists and, in its OWN body (not a nested def, lambda or class, not under a
    constant-false `if`), calls the row's function (its head name; `live_apply` also counts, only for
    a row whose head is a key of `LIVE_APPLY_OF`, when that wrapper is defined in the row file and calls the head) and makes a call to one of `LANDING_LOGS` or
    `CL.record_landing`."""
    lb = row["logged_by"]
    p = root / lb["path"]
    if not p.is_file():
        return f"{lb['path']}: no such file"
    tree = ast.parse(p.read_text())
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == lb["function"]]
    if not fns:
        return f"{lb['path']}: no function {lb['function']}"
    head = row["function"].split(".")[0]
    calls = _own_calls(fns[0])
    names = {c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "") for c in calls}
    accepted = {head}
    wrapper = LIVE_APPLY_OF.get(head)
    if wrapper and wrapper != head:
        chain = _wrapper_chain_problem(row, root, wrapper, head)
        if chain is None:
            accepted.add(wrapper)
        elif wrapper in names:
            return chain
    elif wrapper:
        accepted.add(wrapper)
    if not names & accepted:
        return f"{lb['function']} does not call {head}"
    if not any(_is_landing_log(c.func) for c in calls):
        return f"{lb['function']} makes no change-log call"
    return None
