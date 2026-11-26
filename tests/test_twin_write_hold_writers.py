#!/usr/bin/env python3
"""Task 864 (review 854 S6, review 869 S1): every Qdrant WRITE site outside the drain refuses while an
aw_e10 build holds the twin.

From an aw_e10 T0 to the flip only the drain writes the twin: the catch-up accounts every twin change
by the drain's receipts. A census, not a hand list: every non-test module under `cytir/` and `pipeline/`
whose text holds a Qdrant write call (`WRITE`) must call the hold (`refuse_if_held(` or the CLI line
`refuse_cli_write(`) or be in `EXEMPT` with the reason it cannot write the held twin. A new writer
fails this test until it does one or the other.
"""
from __future__ import annotations

import pathlib
import re
import sys
import types

import pytest

pytestmark = pytest.mark.guard

HERE = pathlib.Path(__file__).resolve().parent
APP = HERE.parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from cytir.index import twin_write_hold as TWH  # noqa: E402

WRITE = re.compile(r"points\?wait|points/delete|points/payload|points/batch|\.upsert\(|\.set_payload\("
                   r"|\.overwrite_payload\(|\.delete_payload\(|\.batch_update_points\(|\.delete\(")
CALLS_HOLD = re.compile(r"refuse_if_held\(|refuse_cli_write\(")
HOLD_CALLS = ("refuse_if_held", "refuse_cli_write")
#: where the hold call must sit: the entry point (review 869 S6: a real call, not text a comment matches)
ENTRY_FUNCS = ("main", "run", "<module>")

#: module -> why it does not (and must not) check the hold. Shrink only, except a 1:1 re-key when exempt code
#: moves (the old row leaves when its file holds no write call; promote_queue.py's row leaves at step 14).
EXEMPT = {
    "cytir/ops/change_log_writers.py": "static scanner (task 2849): it names write calls as patterns and reads source text; it opens no Qdrant client and writes no collection",
    "cytir/index/b1_rebuild.py": "library (no CLI): reached only through cytir/index/b1_cli.py, which calls refuse_cli_write before dispatch",
    "cytir/index/b1_slice.py": "library (no CLI): reached only through cytir/index/b1_cli.py, which calls refuse_cli_write before dispatch",
    "cytir/distribute/rehearsal_log_delta.py": "its only writer, _drain_sequence, is module-private and refuses (RehearsalFailure, before any write) any client that is not a local :memory: QdrantLocal; no URL, no twin",
    "cytir/index/promote_queue.py": "the drain: the one twin writer the catch-up accounts by its receipts",
    "cytir/index/drain/write_sealed.py": "the drain's sealed write leg (twin and prod), moved 1:1 out of cytir/index/promote_queue.py (drain split step 10); library, no CLI: reached only through the drain, the one twin writer the catch-up accounts by its receipts",
    "cytir/index/drain/write_content_pack.py": "the drain's content-pack write leg (twin and prod), moved 1:1 out of cytir/index/promote_queue.py (drain split step 11); library, no CLI: reached only through the drain, the one twin writer the catch-up accounts by its receipts",
    "cytir/index/promote_sealed_pack.py": "the drain's write door (`upsert`), called by the drain",
    "cytir/index/point_delete.py": "the drain's delete primitive (library, no CLI)",
    "cytir/index/payload_patch.py": "the drain's payload_patch leg (library, no CLI): the drain is the twin writer, counted by its receipts",
    "cytir/embed/collection_stage.py": "retired door (task 1378): `main()` refuses and writes nothing; `create_collection` and "
                                    "`embed_rows` are library primitives (no CLI), like point_delete; their one caller is "
                                    "`cytir/index/stage_pack_b1.py` writing pack-staging collections on the build box",
    "cytir/index/assemble_load.py": "library (no CLI): cmd_load is reached only through cytir/index/assemble_cli.py main(), which calls refuse_cli_write first (task 2118: the per-file census cannot see a hold call that sits in the sibling module)",
}


def hold_call_sites(text: str) -> list[str]:
    """The entry points (`main`, `run`, module level) that hold a real `Call` to the hold."""
    import ast
    found = []

    def walk(node, owner):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(ch, ch.name if owner == "<module>" else owner + "." + ch.name)
                continue
            if isinstance(ch, ast.Call):
                f = ch.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                if name in HOLD_CALLS:
                    found.append(owner)
            walk(ch, owner)
    walk(ast.parse(text), "<module>")
    return found


_OLD_TREE = "pipe" + "line"   # spelled in pieces: the public cut ships no such tree


def census() -> dict[str, bool]:
    """{module: it calls the hold from an entry point} for every Qdrant write site."""
    out = {}
    for root in ("cytir", _OLD_TREE):
        for p in sorted((APP / root).rglob("*.py")):
            rel = str(p.relative_to(APP))
            if p.name.startswith("test_") or p.name == "conftest.py" or "__pycache__" in p.parts:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            if WRITE.search(text):
                out[rel] = any(site in ENTRY_FUNCS for site in hold_call_sites(text))
    return out


def test_every_exemption_has_a_reason():
    assert all(isinstance(r, str) and r.strip() for r in EXEMPT.values()), EXEMPT


def test_a_comment_does_not_satisfy_the_census():
    assert hold_call_sites("# refuse_cli_write(args)\ndef main():\n    x = 'refuse_if_held('\n") == []
    assert hold_call_sites("def main():\n    _TWH.refuse_cli_write(a, tool='t')\n") == ["main"]
    assert hold_call_sites("def helper():\n    refuse_if_held('c', tool='t')\n") == ["helper"]


#: Runs ONLY the file's module-level `sys.path` prelude (never the script: some connect or write at
#: load), exactly as `python path.py` would from a clean cwd with no PYTHONPATH, then imports the hold.
_PRELUDE_LOADER = """
import ast, os, sys
p = os.path.abspath(sys.argv[1])
sys.path[0] = os.path.dirname(p)
body = ast.parse(open(p, encoding="utf-8").read()).body
STD = ("os", "sys", "pathlib")
def is_path_stmt(n):
    return isinstance(n, (ast.Expr, ast.If)) and "sys.path" in ast.unparse(n)
last = max((i for i, n in enumerate(body) if is_path_stmt(n)), default=-1)
g = {"__file__": p, "__name__": "prelude"}
for n in body[:last + 1]:
    if isinstance(n, ast.Import) and not all(a.name.split(".")[0] in STD for a in n.names):
        continue
    if isinstance(n, ast.ImportFrom) and n.module not in STD:
        continue
    if not isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.Expr, ast.If)):
        continue
    if isinstance(n, ast.Expr) and not is_path_stmt(n):
        continue
    try:
        exec(compile(ast.Module(body=[n], type_ignores=[]), p, "exec"), g)
    except Exception:
        pass
import cytir.index.twin_write_hold
print("ok", cytir.index.twin_write_hold.__file__)
"""


@pytest.mark.parametrize("rel", sorted(f for f, ok in census().items() if ok and f.startswith(_OLD_TREE + "/")))
def test_each_held_operator_script_imports_the_hold_when_started_by_path(rel, tmp_path):
    """Review 869 M3: the hold line imports `aw`; a script started as `python path.py` (sys.path[0] =
    its own dir, no PYTHONPATH, a clean cwd) must put the repo root on `sys.path` at module level."""
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-I", "-c", _PRELUDE_LOADER, str(APP / rel)], cwd=tmp_path,
                       env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.startswith("ok"), (
        f"{rel}: started by path it cannot import cytir.index.twin_write_hold; put the repo root on sys.path "
        f"at module level (the old-tree aw_common pattern)\n{r.stderr[-800:]}")
    assert str(APP) in r.stdout, f"{rel} imported another checkout's aw: {r.stdout}"


def test_every_qdrant_write_site_checks_the_hold_or_is_exempt_with_a_reason():
    found = census()
    missing = sorted(f for f, ok in found.items() if not ok and f not in EXEMPT)
    assert not missing, f"Qdrant write sites without the aw_e10 twin hold (add refuse_cli_write): {missing}"
    stale = sorted(f for f in EXEMPT if f in found and found[f])
    assert not stale, f"exempt but calls the hold anyway, drop from EXEMPT: {stale}"


def test_the_writers_review_869_named_are_covered():
    found = census()
    for rel in ("cytir/index/payload_profile_reconcile.py",
                "cytir/consent/apply.py",
                "cytir/index/surface_delta.py", "cytir/index/pack_delta.py"):
        assert CALLS_HOLD.search((APP / rel).read_text(encoding="utf-8")), rel


@pytest.mark.parametrize("mod", sorted(TWH.HELD_WRITERS))
def test_each_package_twin_writer_calls_refuse_if_held(mod):
    import importlib
    src = pathlib.Path(importlib.import_module(mod).__file__).read_text(encoding="utf-8")
    assert "refuse_if_held(" in src, TWH.HELD_WRITERS[mod]


def test_refuse_cli_write_reads_every_target_argument(tmp_path, monkeypatch):
    monkeypatch.setenv(TWH.ENV, str(tmp_path / "hold.json"))
    TWH.write_hold(collection="aw_e8", source_qdrant="http://x:1", plan_dir="/p", t0="T0")
    for name in TWH.CLI_TARGET_ARGS:
        with pytest.raises(SystemExit, match="aw_e10 build holds"):
            TWH.refuse_cli_write(types.SimpleNamespace(**{name: "aw_e8"}), tool="t")
    with pytest.raises(SystemExit):
        TWH.refuse_cli_write(tool="t", extra=("aw_e8",))
    TWH.refuse_cli_write(types.SimpleNamespace(collection="aw_e9", dst=None), tool="t")   # free


def test_every_exempt_drain_module_is_a_library_the_facade_imports():
    """Task 2902: "library, no CLI" is checked, not claimed: no `__main__` block, no `def main`, and the
    facade imports the module."""
    import ast
    facade = (APP / "cytir/index/promote_queue.py").read_text(encoding="utf-8")
    for rel in sorted(k for k in EXEMPT if k.startswith("cytir/index/drain/")):
        tree = ast.parse((APP / rel).read_text(encoding="utf-8"))
        assert not any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body), rel
        assert "__main__" not in ast.unparse(tree), rel
        module = rel[:-3].replace("/", ".")
        assert f"from {module} import" in facade or f"import {module}" in facade, rel


_IMPORT_ROOTS = ("cytir", "server", "deploy", "scripts")
_FACADE = "cytir/index/promote_queue.py"
_DRAIN_DIR = "cytir/index/drain/"


def _tracked_py(app: pathlib.Path, roots: tuple[str, ...] = _IMPORT_ROOTS) -> list[pathlib.Path]:
    import subprocess
    try:
        out = subprocess.run(["git", "ls-files", "--", *[f"{r}/*.py" for r in roots]], cwd=app,
                             capture_output=True, text=True, check=True).stdout.split()
        found = [app / o for o in out if (app / o).is_file()]
        if found:
            return found
    except (OSError, subprocess.CalledProcessError):
        pass
    return [p for r in roots for p in sorted((app / r).rglob("*.py"))]


def _is_import_module_call(n) -> bool:
    """`importlib.import_module(...)`, `import_module(...)` or `__import__(...)` with the name as first argument."""
    import ast
    f = n.func
    return (isinstance(f, ast.Attribute) and f.attr == "import_module") or \
        (isinstance(f, ast.Name) and f.id in ("import_module", "__import__"))


def importers_of(app: pathlib.Path, module: str) -> list[str]:
    """Files (relative to `app`) that import `module` ("a.b.c"): `import a.b.c`, `from a.b.c import x`,
    `from a.b import c`, and the relative forms."""
    import ast
    parent, _, leaf = module.rpartition(".")
    hits = []
    for p in _tracked_py(app):
        rel = p.relative_to(app).as_posix()
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        pkg = rel.split("/")[:-1]
        for n in ast.walk(tree):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                base = n.module or ""
                if n.level:
                    base = ".".join(pkg[:len(pkg) - n.level + 1] + ([base] if base else []))
                names = [base] + [f"{base}.{a.name}" for a in n.names]
            if isinstance(n, ast.Call) and _is_import_module_call(n):
                a0 = n.args[0] if n.args else None
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    names = [a0.value]
            if any(x == module or x.startswith(module + ".") for x in names):
                hits.append(rel)
                break
    return sorted(set(hits))


def _allowed_importer(rel: str) -> bool:
    return rel == _FACADE or rel.startswith(_DRAIN_DIR)


def test_exempt_drain_modules_are_imported_only_by_the_facade_and_drain_siblings():
    """Task 2908: the EXEMPT reason "reached only through the drain" is checked, not claimed."""
    for rel in sorted(k for k in EXEMPT if k.startswith(_DRAIN_DIR)):
        module = rel[:-3].replace("/", ".")
        rogue = [f for f in importers_of(APP, module) if not _allowed_importer(f) and f != rel]
        assert not rogue, f"{rel} has non-drain importers {rogue}; hold the hold or drop its EXEMPT row"


def test_a_rogue_importer_of_an_exempt_drain_module_is_caught(tmp_path):
    for sub, text in (("cytir/ops/a.py", "import cytir.index.drain.write_sealed\n"),
                      ("server/b.py", "from cytir.index.drain import write_sealed as W\n"),
                      ("scripts/c.py", "from cytir.index.drain.write_sealed import f\n"),
                      ("deploy/d.py", "x = 1\n"),
                      ("cytir/index/drain/ok.py", "from cytir.index.drain import write_sealed\n")):
        f = tmp_path / sub
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    got = importers_of(tmp_path, "cytir.index.drain.write_sealed")
    assert got == ["cytir/index/drain/ok.py", "cytir/ops/a.py", "scripts/c.py", "server/b.py"], got
    assert [g for g in got if not _allowed_importer(g)] == ["cytir/ops/a.py", "scripts/c.py", "server/b.py"]


#: EXEMPT rows whose file holds no write call yet and are kept on purpose: shrink only (empty today).
NO_WRITE_YET: tuple[str, ...] = ()


def test_every_exempt_row_still_names_a_file_with_a_write_call():
    """Task 2908: a row whose file no longer writes must leave (the facade row leaves at step 14)."""
    dead = sorted(k for k in EXEMPT if k not in NO_WRITE_YET
                  and not WRITE.search((APP / k).read_text(encoding="utf-8", errors="replace")))
    assert not dead, f"EXEMPT rows whose file has no write call, remove them: {dead}"
    assert all(k in EXEMPT for k in NO_WRITE_YET), "NO_WRITE_YET names a row that already left"


def test_the_drain_is_not_held():
    assert not any("promote" in m for m in TWH.HELD_WRITERS)
    files = ["cytir/index/promote_queue.py", "cytir/index/promote_sealed_pack.py"]
    files += [str(p.relative_to(APP)) for p in sorted(APP.glob("cytir/index/drain/*.py"))]
    for f in files:
        assert not CALLS_HOLD.search((APP / f).read_text(encoding="utf-8")), f


def test_stage_pack_b1_holds_at_its_entry():
    """Task 1397: `collection_stage` is an exempt library; its one caller holds the twin hold from `main`."""
    text = (APP / "cytir/index/stage_pack_b1.py").read_text(encoding="utf-8")
    assert "main" in hold_call_sites(text)
    assert "cytir/index/stage_pack_b1.py" not in EXEMPT


def test_stage_pack_b1_refuses_the_held_twin(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setenv(TWH.ENV, str(tmp_path / "hold.json"))
    TWH.write_hold(collection="aw_e8", source_qdrant="http://x:1", plan_dir="/p", t0="T0")
    r = subprocess.run([sys.executable, "-m", "cytir.index.stage_pack_b1", "--in", "x", "--collection", "aw_e8",
                        "--mrl-dim", "128", "--consent-source", "s",
                        "--consent-checked-at", "2026-10-30"], capture_output=True, text=True, timeout=120, cwd=str(APP),
                       env={**__import__("os").environ, TWH.ENV: str(tmp_path / "hold.json")})
    assert r.returncode != 0 and "aw_e10 build holds" in (r.stderr + r.stdout), r.stderr[-500:]


# ---- task 2919: callers that reach the drain's writers through the facade ----------------------------

_DRAIN_FILES_GLOB = "cytir/index/drain/*.py"
#: roots scanned for facade callers (the old tree's name is spelled in pieces: the public cut ships none)
_CALLER_ROOTS = _IMPORT_ROOTS + (_OLD_TREE,)


def drain_write_names(app: pathlib.Path) -> set[str]:
    """Public functions of the facade and `cytir/index/drain/*.py` that (transitively, by name, inside those
    files) reach a `WRITE` match or a row of the change-log writer registry. Computed, not listed."""
    import ast
    import json
    files = [_FACADE] + sorted(p.relative_to(app).as_posix() for p in app.glob(_DRAIN_FILES_GLOB))
    reg_path = app / "data" / "change-log-writers.v1.json"
    reg = set()
    if reg_path.is_file():
        reg = {(r["path"], r["function"]) for r in json.loads(reg_path.read_text(encoding="utf-8"))["rows"]}
    defs: dict[str, list] = {}
    for f in files:
        if not (app / f).is_file():
            continue
        text = (app / f).read_text(encoding="utf-8")
        for n in ast.walk(ast.parse(text)):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.setdefault(n.name, []).append((f, n, ast.get_source_segment(text, n) or ""))
    seeds, calls = set(), {}
    for name, lst in defs.items():
        cs = set()
        for f, n, seg in lst:
            if WRITE.search(seg) or (f, name) in reg:
                seeds.add(name)
            for c in ast.walk(n):
                if isinstance(c, ast.Call):
                    x = c.func
                    cs.add(x.attr if isinstance(x, ast.Attribute) else getattr(x, "id", None))
        calls[name] = cs & set(defs)
    reach, grew = set(seeds), True
    while grew:
        grew = False
        for n, cs in calls.items():
            if n not in reach and cs & reach:
                reach.add(n)
                grew = True
    return {n for n in reach if not n.startswith("_")}


def _is_drain_module(mod: str) -> bool:
    return mod == "cytir.index.promote_queue" or mod == "cytir.index.drain" or mod.startswith("cytir.index.drain.")


def facade_callers_in(text: str, names: set[str]) -> list[tuple[str, str]]:
    """[(function name, owner)] for calls to a drain write name where the callee resolves to the facade or
    a drain module: `Q.f()` with `Q` bound by an import of one, `f()` after a from-import of `f` from one,
    or the dotted path itself. A same-named function the file defines itself does not count."""
    import ast
    tree = ast.parse(text)
    mod_alias: set[str] = set()      # local names bound to the facade / a drain module
    name_alias: dict[str, str] = {}  # local name -> imported drain function name
    dotted: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if _is_drain_module(a.name):
                    if a.asname:
                        mod_alias.add(a.asname)
                    else:
                        dotted.add(a.name)
        elif isinstance(n, ast.ImportFrom) and not n.level:
            base = n.module or ""
            for a in n.names:
                if _is_drain_module(base) and a.name in names:
                    name_alias[a.asname or a.name] = a.name
                elif _is_drain_module(f"{base}.{a.name}"):
                    mod_alias.add(a.asname or a.name)
    out = []

    def walk(node, owner):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(ch, ch.name if owner == "<module>" else owner + "." + ch.name)
                continue
            if isinstance(ch, ast.Call):
                f = ch.func
                if isinstance(f, ast.Attribute) and f.attr in names:
                    base = f.value
                    if (isinstance(base, ast.Name) and base.id in mod_alias) or ast.unparse(base) in dotted:
                        out.append((f.attr, owner))
                elif isinstance(f, ast.Name) and f.id in name_alias:
                    out.append((name_alias[f.id], owner))
            walk(ch, owner)
    walk(tree, "<module>")
    return out


def facade_caller_census(app: pathlib.Path) -> dict[str, tuple[bool, set[str]]]:
    """{file: (it calls the hold from an entry point, the drain write names it calls)} for every tracked
    non-test file outside the drain that calls a drain write function through the facade or a drain module."""
    names = drain_write_names(app)
    out = {}
    for p in _tracked_py(app, _CALLER_ROOTS):
        rel = p.relative_to(app).as_posix()
        if rel == _FACADE or rel.startswith(_DRAIN_DIR) or p.name.startswith("test_") or p.name == "conftest.py":
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        try:
            hits = facade_callers_in(text, names)
            if hits:
                out[rel] = (any(s in ENTRY_FUNCS for s in hold_call_sites(text)), {n for n, _ in hits})
        except SyntaxError:
            continue
    return out


def unaccounted_callers(found: dict, exempt: dict) -> list[str]:
    """Files with no entry-point hold call that call a drain write name their exemption does not name."""
    return sorted(f for f, (held, called) in found.items()
                  if not held and called - set(exempt.get(f, {}).get("names", ())))


def calls_in_function(text: str, func: str) -> set[str]:
    """Names of the real `Call`s inside the function `func` (a comment or string is not a call)."""
    import ast
    out: set[str] = set()
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, ast.FunctionDef) and n.name == func:
            for c in ast.walk(n):
                if isinstance(c, ast.Call):
                    out.add(c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", ""))
    return out


#: Files that call a drain write function through the facade and are the drain's own door, pinned to the
#: names they may call: the drain is the one twin writer the catch-up accounts by its receipts. Shrink only;
#: a new row needs the orchestrator's ruling. (`cytir/index/build.py` left in review 2921: it holds the hold.)
FACADE_CALLER_EXEMPT = {
    "cytir/cli.py": {"names": {"main"},
                  "reason": "the `cyt index promote` verb: dispatches PQ.main, the drain itself (twin-first, accounted by "
                            "its receipts); `_index_main` first runs PQ.refuse_live_queue_run and takes PQ.hold_unit_lock"},
}
#: Callers found by task 2919 that neither hold nor are the drain's door: shrink only (empty today).
KNOWN_UNHELD_CALLERS: tuple[str, ...] = ()


def test_the_derived_drain_write_names_hold_the_known_writers():
    names = drain_write_names(APP)
    pinned = {"apply_side", "live_apply", "single_apply", "rehearse", "apply_content_pack_side",
              "live_apply_content_pack", "live_apply_payload_patch", "resume_prod_only",
              "resume_content_pack_prod_only", "drain", "main"}
    if not (APP / "data" / "change-log-writers.v1.json").is_file():
        # the public cut ships no writer registry: `live_apply_payload_patch` reaches its write only through
        # the registry row (payload_patch.QdrantPatchClient -> point_delete), which no text pattern matches
        pinned -= {"live_apply_payload_patch"}
    assert pinned <= names, sorted(pinned - names)
    assert not any(n.startswith("_") for n in names)


def test_every_facade_caller_of_a_drain_writer_holds_the_hold_or_is_the_drain_door():
    found = facade_caller_census(APP)
    missing = [f for f in unaccounted_callers(found, FACADE_CALLER_EXEMPT) if f not in KNOWN_UNHELD_CALLERS]
    assert not missing, (f"files that call a drain write function through the facade without the aw_e10 twin hold "
                         f"(add refuse_cli_write at the entry point): {missing}")
    stale = sorted(f for f in (*FACADE_CALLER_EXEMPT, *KNOWN_UNHELD_CALLERS) if not (f in found and not found[f][0]))
    assert not stale, f"exempt/known rows that are not unheld facade callers any more, drop them: {stale}"
    assert not (set(FACADE_CALLER_EXEMPT) & set(KNOWN_UNHELD_CALLERS))
    assert all(v["reason"].strip() and v["names"] for v in FACADE_CALLER_EXEMPT.values())


def test_a_file_exempt_for_one_name_that_calls_another_is_flagged():
    text = "from cytir.index import promote_queue as Q\ndef f():\n    Q.main()\n    Q.apply_side(1)\n"
    found = {"cytir/x.py": (False, {n for n, _ in facade_callers_in(text, {"main", "apply_side"})})}
    assert unaccounted_callers(found, {"cytir/x.py": {"names": {"main"}}}) == ["cytir/x.py"]
    assert unaccounted_callers(found, {"cytir/x.py": {"names": {"main", "apply_side"}}}) == []
    assert unaccounted_callers({"cytir/x.py": (True, {"apply_side"})}, {}) == []


def test_the_exempt_drain_door_keeps_its_refusals_as_real_calls():
    calls = calls_in_function((APP / "cytir/cli.py").read_text(encoding="utf-8"), "_index_main")
    assert {"refuse_live_queue_run", "hold_unit_lock"} <= calls, calls


def test_a_comment_or_string_does_not_satisfy_the_door_pin():
    text = ("def _index_main(a):\n    # PQ.refuse_live_queue_run(a); PQ.hold_unit_lock()\n"
            "    x = 'refuse_live_queue_run(hold_unit_lock('\n")
    assert calls_in_function(text, "_index_main") == set()
    real = "def _index_main(a):\n    PQ.refuse_live_queue_run(a)\n    with PQ.hold_unit_lock():\n        pass\n"
    assert calls_in_function(real, "_index_main") == {"refuse_live_queue_run", "hold_unit_lock"}


def test_build_holds_the_hold_at_its_entry():
    text = (APP / "cytir/index/build.py").read_text(encoding="utf-8")
    assert "main" in hold_call_sites(text)
    assert "cytir/index/build.py" not in FACADE_CALLER_EXEMPT


@pytest.mark.parametrize("mode,extra", [("build", ["--pack", "x"]), ("apply", ["--pack", "x"]),
                                        ("build", ["--from-collection", "http://x:1:src"])])
def test_aw_index_build_refuses_the_held_twin_before_any_client(tmp_path, monkeypatch, mode, extra):
    import urllib.request
    from cytir.index import build as BUILD
    monkeypatch.setenv(TWH.ENV, str(tmp_path / "hold.json"))
    TWH.write_hold(collection="aw_e8", source_qdrant="http://x:1", plan_dir="/p", t0="T0")

    def boom(*a, **k):
        raise AssertionError("a client or request was made before the hold refused")
    monkeypatch.setattr(BUILD, "qdrant_client", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(SystemExit, match="aw_e10 build holds"):
        BUILD.main(["--qdrant", "http://127.0.0.1:1", "--collection", "aw_e8", "--out", str(tmp_path / "o"), *extra],
                   mode=mode)
    assert not (tmp_path / "o").exists()


def test_aw_index_build_is_unchanged_for_a_collection_the_hold_does_not_name(tmp_path, monkeypatch):
    from cytir.index import build as BUILD
    monkeypatch.setenv(TWH.ENV, str(tmp_path / "hold.json"))
    TWH.write_hold(collection="aw_e8", source_qdrant="http://x:1", plan_dir="/p", t0="T0")
    monkeypatch.setattr(BUILD, "run", lambda **k: (0, {}))
    assert BUILD.main(["--qdrant", "http://127.0.0.1:1", "--collection", "aw_other", "--out", str(tmp_path / "o")]) == 0


def _fixture_census(tmp_path, files: dict[str, str]) -> list[str]:
    names = {"apply_side", "drain", "main", "live_apply"}
    out = []
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        if facade_callers_in(text, names):
            out.append(rel)
    return out


def test_rogue_callers_through_the_facade_are_caught_and_lookalikes_are_not(tmp_path):
    got = _fixture_census(tmp_path, {
        "cytir/ops/rogue_alias.py": "from cytir.index import promote_queue as Q\ndef main():\n    Q.apply_side(1)\n",
        "cytir/ops/rogue_bare.py": "from cytir.index.promote_queue import live_apply\nlive_apply(1)\n",
        "cytir/ops/rogue_drain_mod.py": "from cytir.index.drain.write_sealed import apply_side as a\ndef run():\n    a()\n",
        "cytir/ops/rogue_dotted.py": "import cytir.index.promote_queue\ncytir.index.promote_queue.drain(1)\n",
        "cytir/ops/rogue_pkg.py": "from cytir.index.drain import write_content_pack as W\nW.live_apply(1)\n",
        "cytir/ops/own_drain.py": "def drain():\n    pass\ndrain()\n",
        "cytir/ops/other_obj.py": "import queue as Q\nQ.drain(1)\nobj.apply_side(1)\n",
    })
    assert got == ["cytir/ops/rogue_alias.py", "cytir/ops/rogue_bare.py", "cytir/ops/rogue_drain_mod.py",
                   "cytir/ops/rogue_dotted.py", "cytir/ops/rogue_pkg.py"], got


def test_the_facade_census_skips_tests_and_the_drain_itself(tmp_path):
    (tmp_path / "cytir/index/drain").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    rogue = "from cytir.index import promote_queue as Q\nQ.apply_side(1)\n"
    (tmp_path / "cytir/index/promote_queue.py").write_text("def apply_side():\n    x.upsert(1)\n" + rogue)
    (tmp_path / "cytir/index/drain/w.py").write_text("def f():\n    x.upsert(1)\n" + rogue)
    (tmp_path / "cytir/ops").mkdir()
    (tmp_path / "cytir/ops/test_x.py").write_text(rogue)
    (tmp_path / "cytir/ops/conftest.py").write_text(rogue)
    (tmp_path / "cytir/ops/bad.py").write_text(rogue)
    assert facade_caller_census(tmp_path) == {"cytir/ops/bad.py": (False, {"apply_side"})}


def test_a_module_loaded_by_import_module_is_an_importer(tmp_path):
    (tmp_path / "cytir").mkdir()
    (tmp_path / "cytir" / "rogue.py").write_text(
        "import importlib\nm = importlib.import_module('cytir.index.drain.apply')\n")
    (tmp_path / "cytir" / "dyn.py").write_text("from importlib import import_module\nimport_module('cytir.index.drain')\n")
    (tmp_path / "cytir" / "other.py").write_text("import importlib\nimportlib.import_module('cytir.index.upsert')\n")
    assert importers_of(tmp_path, "cytir.index.drain.apply") == ["cytir/rogue.py"]
    assert importers_of(tmp_path, "cytir.index.drain") == ["cytir/dyn.py", "cytir/rogue.py"]
