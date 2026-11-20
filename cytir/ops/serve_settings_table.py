"""cytir.ops.serve_settings_table -- the S10 step-2 proof: every `AW_*` env read the serving code
made at a base revision, where it went, and whether the new parse answers exactly as the old one.

    cyt ops serve-settings-table --base <rev>                 # the markdown table (stdout)
    cyt ops serve-settings-table --base <rev> --out t.md      # write it
    cyt ops serve-settings-table --base <rev> --check         # exit 1 on any mismatch or unplaced read

For each `os.environ.get("AW_...", default)` in `cytir/serve/*.py` at `--base` (found with
`ast`, together with the coercion around it: `float(...)`, `int(...)`, `... == "1"`,
`... or WEIGHTS.get(...)`), the table names the read site and its destination
in the working tree:
  field     a declared `cytir.serve.settings` field (group.attr, kind, default, weights key)
  constant  a module constant marked `# fixed: was env <NAME>` (S10 D5) or `# fixed <date> (S10 step 3): was <NAME>, ...`
  weights   the env override is gone; the default still comes from the weights file
  schema    `schema.py` keeps its own read (the shared tokenizer; the field mirrors it)
  deleted   a generic read inside a deleted helper (its callers became constants)
The equivalence column evaluates the OLD expression and the NEW parse on the same inputs: unset,
"", "0", "1", "true", "7", "0.5", "-1.5", "abc", " 1 ", each under the shipped and the prod weights
file. Equal means the same value (NaN equals NaN) or the same exception type. A constant and a
weights-only field are compared on the unset input only (they ignore the env by design).
Spec: specs/s10-serve-split.md section 2; task 1071.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import types

APP = pathlib.Path(__file__).resolve().parent.parent.parent
from cytir.core.serve_layout import SERVER_REL  # noqa: E402

SCENARIOS = (None, "", "0", "1", "true", "7", "0.5", "-1.5", "abc", " 1 ")
FIXED = re.compile(r"#\s*fixed(?::| \d{4}-\d\d-\d\d \(S10 step 3\):) was (?:env )?(AW_[A-Z0-9_]+)")
WEIGHT_FILES = ("fusion_weights.json", "fusion_weights.candidate533.json")


def _modules_at(rev: str | None, app: pathlib.Path = APP) -> dict[str, str]:
    """{file name: source} of the serving modules at `rev` (None: the working tree)."""
    if rev is None:
        d = app / SERVER_REL
        return {p.name: p.read_text(encoding="utf-8") for p in sorted(d.glob("*.py"))
                if not p.name.startswith("test_") and p.name != "__init__.py"}
    rel = SERVER_REL
    ls = [pathlib.PurePosixPath(x).name for x in subprocess.run(
        ["git", "-C", str(app), "ls-tree", "--name-only", rev, rel + "/"],
        capture_output=True, text=True, check=True).stdout.split()]
    if not ls:
        raise SystemExit(f"serve_settings_table: no {SERVER_REL} at {rev}")
    out = {}
    for name in ls:
        if name.endswith(".py") and not name.startswith("test_") and name != "__init__.py":
            out[name] = subprocess.run(["git", "-C", str(app), "show", f"{rev}:./{rel}/{name}"],
                                       capture_output=True, text=True, check=True).stdout
    return out


def read_sites(sources: dict[str, str]) -> list[dict]:
    """Every `os.environ.get(...)` with the coercion around it (pure: a function of the sources)."""
    out = []
    for fname, src in sorted(sources.items()):
        tree = ast.parse(src)
        par = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}

        def func_of(n):
            while n in par:
                n = par[n]
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return n.name
            return None
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Call) and ast.unparse(n.func) == "os.environ.get"):
                continue
            name = n.args[0].value if isinstance(n.args[0], ast.Constant) else None
            kind, expr, wkey = "str", n, None
            d = n.args[1] if len(n.args) > 1 else None
            if isinstance(d, ast.Call) and ast.unparse(d.func) == "WEIGHTS.get":
                wkey = d.args[0].value
            p = par.get(n)
            if isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.args == [n] \
                    and p.func.id in ("float", "int"):
                kind, expr = p.func.id, p
            elif (isinstance(p, ast.Compare) and p.left is n and len(p.ops) == 1
                  and isinstance(p.ops[0], ast.Eq) and isinstance(p.comparators[0], ast.Constant)
                  and p.comparators[0].value == "1"):
                kind, expr = "flag", p
            elif isinstance(p, ast.BoolOp) and isinstance(p.op, ast.Or) and p.values[0] is n:
                kind, expr, wkey = "str_or_weights", p, p.values[1].args[0].value
            out.append({"file": fname, "line": n.lineno, "env": name, "kind": kind, "weights": wkey,
                        "func": func_of(n), "expr": ast.get_source_segment(src, expr),
                        "default": ast.get_source_segment(src, d) if d is not None else None})
    return out


def fixed_constants(sources: dict[str, str]) -> dict[str, dict]:
    """{env name: {file, target, value}} for every `# fixed: was env <NAME>` module constant."""
    out = {}
    for fname, src in sources.items():
        lines = src.splitlines()
        for n in ast.parse(src).body:
            if not isinstance(n, (ast.Assign, ast.AnnAssign)):
                continue
            m = FIXED.search(lines[n.end_lineno - 1])
            if m:
                tgt = n.targets[0] if isinstance(n, ast.Assign) else n.target
                try:
                    value = ast.literal_eval(n.value)
                except ValueError:
                    value = eval(compile(ast.Expression(n.value), fname, "eval"), {})  # noqa: S307
                out[m.group(1)] = {"file": fname, "target": ast.unparse(tgt), "value": value}
    return out


def load_settings_module(server: pathlib.Path):
    """`cytir.serve.settings` from `server` by file path (a leaf module: no server imports)."""
    spec = importlib.util.spec_from_file_location("_aw_settings_under_test", server / "settings.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _eval_old(site: dict, env: dict, weights: dict, server: pathlib.Path):
    fake_os = types.SimpleNamespace(environ=env, path=os.path)
    ns = {"os": fake_os, "WEIGHTS": weights, "__file__": str(server / site["file"])}
    return eval("(" + site["expr"] + ")", ns)  # noqa: S307 -- our own source at a pinned revision


def _outcome(fn):
    try:
        v = fn()
    except Exception as e:  # noqa: BLE001 -- the exception TYPE is the outcome compared
        return ("raise", type(e).__name__)
    return ("nan",) if isinstance(v, float) and math.isnan(v) else ("value", v)


def build(base: str, app: pathlib.Path = APP) -> tuple[list[dict], dict]:
    old = _modules_at(base, app)
    new = _modules_at(None, app)
    server = app / SERVER_REL
    st = load_settings_module(server)
    by_env = {m["env"]: (g, a, m) for g, a, m in st.specs() if m["env"]}
    by_wkey = {m["weights"]: (g, a, m) for g, a, m in st.specs() if m["env"] is None and m["weights"]}
    consts = fixed_constants(new)
    wfiles = []
    for w in WEIGHT_FILES:
        raw = json.loads((server / w).read_text(encoding="utf-8"))
        wfiles.append(({**st._DEFAULT_WEIGHTS, **raw}, frozenset(raw)))
    rows = []
    for s in read_sites(old):
        row = dict(s, site=f"{s['file']}:{s['line']}")
        n = s["env"]
        checks = []
        if n is None:
            row.update(dest="deleted", new=f"{s['func']} (generic reader; its callers are constants)")
        elif s["file"] == "schema.py":
            g, a, m = by_env[n]
            if m["kind"] != s["kind"]:
                # schema's own second reading of a name another module owns (a log level):
                # the line is unchanged, so there is nothing to compare
                row.update(dest="schema", new=f"unchanged (the field {g}.{a} is the {m['kind']})")
                rows.append(dict(row, equal=0, cases=0))
                continue
            row.update(dest="schema", new=f"{g}.{a} (mirror)")
            for raw in SCENARIOS:
                env = {} if raw is None else {n: raw}
                checks.append(_outcome(lambda: _eval_old(s, env, {}, server))
                              == _outcome(lambda: st._parse(m, env, {}, ())[0]))
        elif n in consts:
            c = consts[n]
            row.update(dest="constant", new=f"{c['file']}:{c['target']} = {c['value']!r}")
            # a constant that replaced a comma list's raw string holds the derived class set
            derive = st.class_set if isinstance(c["value"], frozenset) else (lambda v: v)
            for w, _ in wfiles:
                checks.append(_outcome(lambda: derive(_eval_old(s, {}, w, server)))
                              == _outcome(lambda: c["value"]))
        elif n in by_env:
            g, a, m = by_env[n]
            row.update(dest="field", new=f"{g}.{a}", new_kind=m["kind"], new_default=m["default"],
                       new_weights=m["weights"])
            for w, keys in wfiles:
                for raw in SCENARIOS:
                    env = {} if raw is None else {n: raw}
                    checks.append(_outcome(lambda: _eval_old(s, env, w, server))
                                  == _outcome(lambda: st._parse(m, env, w, keys)[0]))
        elif s["weights"] in by_wkey:
            g, a, m = by_wkey[s["weights"]]
            row.update(dest="weights", new=f"{g}.{a} (env override removed)", new_kind=m["kind"],
                       new_default=m["default"], new_weights=m["weights"])
            for w, keys in wfiles:
                checks.append(_outcome(lambda: _eval_old(s, {}, w, server))
                              == _outcome(lambda: st._parse(m, {}, w, keys)[0]))
        else:
            row.update(dest="UNPLACED", new="")
        row["equal"], row["cases"] = sum(checks), len(checks)
        rows.append(row)
    leftover = [s for s in read_sites(new) if s["file"] not in ("schema.py",)]
    summary = {"base": base, "sites": len(rows),
               "names": len({r["env"] for r in rows if r["env"]}),
               "by_dest": {d: sum(1 for r in rows if r["dest"] == d)
                           for d in ("field", "constant", "weights", "schema", "deleted", "UNPLACED")},
               "fields": len(st.specs()), "constants": len(consts),
               "mismatched": [r["site"] for r in rows if r["equal"] != r["cases"]],
               "env_reads_left_outside_schema": [f"{s['file']}:{s['line']}" for s in leftover]}
    return rows, summary


def markdown(rows: list[dict], summary: dict) -> str:
    out = [f"# S10 step 2: every serving env read at `{summary['base']}` and where it went",
           "", "Generated by `python -m cytir.ops.serve_settings_table --base "
           f"{summary['base']}` (do not edit).", "",
           f"Read sites {summary['sites']} ({summary['names']} names): "
           + ", ".join(f"{k} {v}" for k, v in summary["by_dest"].items())
           + f". Declared fields {summary['fields']}; fixed constants {summary['constants']}. "
           f"Mismatched sites: {len(summary['mismatched'])}. Env reads left outside `schema.py`: "
           f"{len(summary['env_reads_left_outside_schema'])}.", "",
           "| env name | old read site | old coercion | old default | destination | new | new kind | new default | weights key | equal / cases |",
           "|---|---|---|---|---|---|---|---|---|--:|"]
    for r in sorted(rows, key=lambda r: (r["env"] or "", r["site"])):
        def cell(v):
            return "" if v is None else str(v).replace("|", "\\|").replace("\n", " ")
        out.append(f"| `{r['env'] or '(name)'}` | {r['site']} | {r['kind']} | {cell(r['default'])[:60]} | "
                   f"{r['dest']} | {cell(r['new'])} | {cell(r.get('new_kind'))} | "
                   f"{cell(r.get('new_default'))[:60]} | {cell(r.get('new_weights') or r['weights'])} | "
                   f"{r['equal']} / {r['cases']} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cyt ops serve-settings-table", description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="the revision whose env reads are the reference")
    ap.add_argument("--out", help="write the markdown table here")
    ap.add_argument("--json", help="write rows + summary as JSON here")
    ap.add_argument("--check", action="store_true", help="exit 1 on a mismatch or an unplaced read")
    a = ap.parse_args(argv)
    rows, summary = build(a.base)
    md = markdown(rows, summary)
    if a.out:
        pathlib.Path(a.out).write_text(md, encoding="utf-8")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1,
                                                   default=repr), encoding="utf-8")
    if not a.out:
        sys.stdout.write(md)
    bad = summary["mismatched"] or summary["by_dest"]["UNPLACED"] or summary["env_reads_left_outside_schema"]
    print(json.dumps({k: summary[k] for k in ("sites", "by_dest", "fields", "constants", "mismatched",
                                              "env_reads_left_outside_schema")}), file=sys.stderr)
    return 1 if (a.check and bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
