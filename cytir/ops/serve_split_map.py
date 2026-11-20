"""cytir.ops.serve_split_map -- the S10 serve split, measured: every def of the serving monolith, the
module it moves to, and every `AW_*` knob the server reads, classified live / armed / dead.

    cyt ops serve-split-map  ==  python -m cytir.ops.serve_split_map
    python -m cytir.ops.serve_split_map                       # the markdown tables (stdout)
    python -m cytir.ops.serve_split_map --json                # every def and knob, machine-readable
    python -m cytir.ops.serve_split_map --write specs/s10-serve-split.md   # refresh the generated block
    python -m cytir.ops.serve_split_map --check specs/s10-serve-split.md   # exit 1: def table stale/unmapped

The design is `TARGET` below: one line per def (qualified name, target module, step). The code is
the measurement: the script parses `cytir/serve/*.py` with `ast`, finds where each def lives
TODAY, and reports it as `done` (it already lives in its target module) or `pending`. A def that
exists in the server and is not in `TARGET` is an error, so the map cannot fall behind the code;
a `TARGET` line whose def exists nowhere is an error too (a deleted def leaves the map in the same
PR). Spec and plan: specs/s10-serve-split.md.

Knob classes (every name in a string literal of a server module, not in a comment):
  live      the serving compose env sets it (aw-api base + every overlay, `cytir.ops.prod_env`; or
            aw-mcp)
  armed     not live, but named by a file the public cut ships (`cytir.ops.public_cut`), docs
            excluded: a gate, a test, a deploy script, a data file
  legacy    not live, named only by tracked files the cut does not ship, at least one of them
            outside the task folders (the legacy stage scripts, the dev forks): a typed field until
            S9 retires that setter
  explore   named only by task-folder exploration scripts: no product path sets it
  dead      named by no file but the server modules that read it (docs excluded)
For `explore` and `dead` the value is the code default (or, for `weights:` knobs, the weights file)
wherever the product runs.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import pathlib
import re
import subprocess
import sys
import tokenize

APP = pathlib.Path(__file__).resolve().parent.parent.parent
from cytir.core.serve_layout import SERVER_REL  # noqa: E402
SERVER = APP / SERVER_REL
BEGIN = "<!-- BEGIN generated: python -m cytir.ops.serve_split_map --write (do not edit) -->"
END = "<!-- END generated: serve_split_map -->"
KNOB = re.compile(r"\bAW_[A-Z0-9_]+\b")

#: Layer of each target module (the five concerns of the spec, plus config and the shell).
LAYERS = {
    "settings": "config", "trace": "config",
    "query": "query understanding", "textmatch": "query understanding",
    "geo": "query understanding", "heads": "query understanding",
    "plan": "leg execution", "qdrant_io": "leg execution", "legs": "leg execution", "operators": "leg execution",
    "tables": "fusion + signals", "signals": "fusion + signals", "fusion": "fusion + signals",
    "levers": "fusion + signals",
    "respond": "post-processing",
    "retrieval": "orchestrator",
}

#: The design: `qualname  target  step`. Nested defs keep their parent's prefix (`search.qk`).
TARGET_TABLE = """
_trust_tier signals 0
_domain signals 0
_academic_penalty signals 0
_machine_url signals 0
_exact signals 0
_lf_bulk signals 0
_minmax signals 0
_recency signals 0
_query_academic_intent query 0
_detect_identifier query 0
_query_commercial_intent query 1
_content_terms textmatch 1
_stem_match textmatch 1
_title_centrality textmatch 1
_card_aliases textmatch 1
_card_name_match textmatch 1
_query_terms textmatch 1
_title_versions_match textmatch 1
_title_centrality_v textmatch 1
_exact_v textmatch 1
_registrable_domain signals 1
effective_config settings 2
_person_name_tokens query 3
_person_name_query query 3
_name_token_primaries query 3
_required_word_groups query 3
_card_alias_in_query query 3
_name_token_coverage levers 3
_sparse_name_coverage levers 3
_apply_namegate levers 3
_refcanon_canonical levers 3
_apply_refcanon levers 3
_load_parasites tables 4
_parasite tables 4
_load_gazetteer geo 4
_weights_path settings 2
_load_weights settings 2
loaded_artifacts tables 4
weights_fingerprint tables 4
_load_hub_hosts tables 4
_hubness tables 4
_load_authority_hosts tables 4
_authority tables 4
_demotable_scope geo 4
_is_eu_supra geo 4
_terms_re geo 4
_query_jurisdiction geo 4
_gazetteer_scope geo 4
_scope_penalties geo 4
_country_is_binding geo 4
_off_country_hit geo 4
_effective_country geo 4
_nonbound_class geo 4
_learned_scope_conflict_penalty geo 4
_load_rankhead heads 5
_load_rankhead_v2 heads 5
_head_beta heads 5
_parse_country_scope_mode heads 5
_load_qhead heads 5
_qhead_vector heads 5
_load_scope_head heads 5
_load_query_policy_heads heads 5
_query_policy_vector heads 5
_scope_head_predict heads 5
_head_artifacts heads 5
_resolve_scope heads 5
_effective_free_text_country_scope heads 5
_country_scope_for_ranking heads 5
indexed_fields qdrant_io 6
_identifier_lookup qdrant_io 6
client qdrant_io 6
health_client qdrant_io 6
_leg qdrant_io 6
candidate_pool plan 6
fuse_margin plan 6
QueryPlan plan 6
QueryPlan.country_legs_enabled plan 6
_country_query_plan plan 6
_country_query_plan._quota plan 6
_country_partition_filter plan 6
_is_tracking_param respond 6
_canonical_key respond 6
_dbgc trace 6
_trustfloor_exempt fusion 7
_trustfloor_scale fusion 7
_is_primary_class fusion 7
_primacy_refund fusion 7
_url_tie_len fusion 7
_order_ids fusion 7
_auth_admit fusion 7
_extend_pool fusion 7
_order_scored fusion 7
_card_key levers 7
_card_winners levers 7
search retrieval 8
qk plan 8
stage1_meta legs 8
run_legs legs 8
assemble_pool legs 8
cut_pool legs 8
stage1 legs 8
trace_points trace 8
s1_rel fusion 8
s1_boost fusion 8
s1_key fusion 8
page_of fusion 8
_reciprocal_competition_rank levers 8
coverage_probe levers 8
build_rows levers 8
apply_levers levers 8
score_rows fusion 8
finish respond 8
"""


def target() -> dict[str, tuple[str, int]]:
    out = {}
    for line in TARGET_TABLE.strip().splitlines():
        name, mod, step = line.split()
        if name in out:
            raise SystemExit(f"TARGET: {name} listed twice")
        out[name] = (mod, int(step))
    return out


def server_modules(server: pathlib.Path = SERVER) -> list[pathlib.Path]:
    """The serving modules: `cytir/serve/*.py`, tests and `__init__` excluded."""
    return sorted(p for p in server.glob("*.py")
                  if p.name != "__init__.py" and not p.name.startswith("test_"))


def defs_of(path: pathlib.Path) -> list[dict]:
    """Every function/class def in `path`, nested ones as `parent.child`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[dict] = []

    def walk(node, prefix):
        for n in ast.iter_child_nodes(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = prefix + n.name
                out.append({"name": q, "module": path.stem, "line": n.lineno,
                            "lines": n.end_lineno - n.lineno + 1,
                            "kind": "class" if isinstance(n, ast.ClassDef) else "def"})
                walk(n, q + ".")
            else:
                walk(n, prefix)
    walk(tree, "")
    return out


def def_map(server: pathlib.Path = SERVER) -> tuple[list[dict], list[str]]:
    """Each mapped def with where it is and where it goes; the list of problems."""
    tgt = target()
    found: dict[str, dict] = {}
    for p in server_modules(server):
        for d in defs_of(p):
            if d["name"] in tgt:
                found.setdefault(d["name"], d)
                if found[d["name"]]["module"] != d["module"] and d["module"] == tgt[d["name"]][0]:
                    found[d["name"]] = d          # the copy in its target module wins
            elif d["module"] == "retrieval":
                found[d["name"]] = dict(d, unmapped=True)
    problems = [f"unmapped def in retrieval.py: {n}" for n, d in found.items() if d.get("unmapped")]
    problems += [f"TARGET names a def that exists nowhere: {n}" for n in tgt if n not in found]
    rows = []
    for name, (mod, step) in tgt.items():
        d = found.get(name)
        if d is None:
            continue
        rows.append(dict(d, target=mod, step=step, layer=LAYERS.get(mod, "?"),
                         status="done" if d["module"] == mod else "pending"))
    return rows, problems


def _frozen_lines(path: pathlib.Path, src: str) -> set[int]:
    """Line numbers of the `FROZEN = {...}` block of `settings.py` (task 1086): the only place a
    `# frozen-echo` tag means "echo, not a read". Any other module or line keeps its knobs counted."""
    if path.name != "settings.py":
        return set()
    out: set[int] = set()
    for n in ast.parse(src).body:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            tgt = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Name) and t.id == "FROZEN" for t in tgt):
                out |= set(range(n.lineno, (n.end_lineno or n.lineno) + 1))
    return out


def _string_knobs(path: pathlib.Path) -> set[str]:
    """`AW_*` names in string tokens (not comments) of one module. A `# frozen-echo` tag skips the
    line only inside the `FROZEN` block of `settings.py` (task 1086)."""
    names: set[str] = set()
    src = path.read_text(encoding="utf-8")
    frozen = _frozen_lines(path, src)
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.STRING and not (t.start[0] in frozen and "# frozen-echo" in t.line):
            names |= set(KNOB.findall(t.string))
    return names


def _weights_default_knobs(path: pathlib.Path) -> dict[str, str]:
    """knob -> weights key, for assignments whose default comes from `WEIGHTS` (a module-level
    read) or from a `settings` declaration's `weights=` key (a dataclass field, task 1071)."""
    src = path.read_text(encoding="utf-8")
    out = {}
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            seg = ast.get_source_segment(src, n) or ""
            m = re.search(r"(?:WEIGHTS\.get\(|\bweights=)\s*['\"]([a-z0-9_]+)['\"]", seg)
            if m:
                for k in KNOB.findall(seg):
                    out[k] = m.group(1)
    return out


def _tracked(app: pathlib.Path) -> list[str]:
    r = subprocess.run(["git", "-C", str(app), "ls-files"], capture_output=True, text=True, check=True)
    return r.stdout.split()


def _mcp_env_keys(app: pathlib.Path) -> set[str]:
    import yaml
    doc = yaml.safe_load((app / "deploy" / "docker-compose.yaml").read_text(encoding="utf-8"))
    env = ((doc.get("services") or {}).get("aw-mcp") or {}).get("environment") or {}
    return set(env if isinstance(env, dict) else (e.split("=", 1)[0] for e in env))


DOC_SUFFIXES = (".md", ".txt", ".rst")
TASK_DIR = "tasks"   # the per-delegation folders: exploration, never product (repo conventions)


def knob_census(app: pathlib.Path = APP) -> list[dict]:
    from cytir.ops import prod_env
    server = app / SERVER_REL
    live = set(prod_env.resolve_env().env) | _mcp_env_keys(app)
    readers: dict[str, list[str]] = {}
    weights_default: dict[str, str] = {}
    for p in server_modules(server):
        for k in _string_knobs(p):
            readers.setdefault(k, []).append(p.stem)
        weights_default.update(_weights_default_knobs(p))
    from cytir.ops import public_cut
    tracked = _tracked(app)
    shipped, _ = public_cut.select(public_cut.load_allowlist(app), tracked)
    own = {f"cytir/serve/{p.name}" for p in server_modules(server)}
    elsewhere: dict[str, list[str]] = {}
    for rel in tracked:
        if rel in own or rel.endswith(DOC_SUFFIXES):
            continue
        p = app / rel
        try:
            if p.stat().st_size > 2_000_000:
                continue
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for k in set(KNOB.findall(text)) & set(readers):
            elsewhere.setdefault(k, []).append(rel)
    rows = []
    for k in sorted(readers):
        by = elsewhere.get(k, [])
        tops = sorted({r.split("/", 1)[0] for r in by})
        cls = ("live" if k in live else "armed" if any(r in shipped for r in by)
               else "explore" if tops == [TASK_DIR] else "legacy" if by else "dead")
        rows.append({"knob": k, "class": cls, "readers": sorted(readers[k]),
                     "weights_default": weights_default.get(k),
                     "named_by_dirs": tops, "named_by": sorted(by)[:5]})
    return rows


def lines_of(path: pathlib.Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def defs_markdown(rows: list[dict], server: pathlib.Path = SERVER) -> str:
    """The def table: a pure function of the server code and `TARGET_TABLE` (CI checks it)."""
    out = []
    retr = server / "retrieval.py"
    n_defs = sum(1 for d in defs_of(retr))
    done = sum(1 for r in rows if r["status"] == "done")
    out.append(f"`retrieval.py`: {lines_of(retr):,} lines, {n_defs} defs (classes and nested defs "
               f"included). Mapped defs: {len(rows)}; in their target module: {done}; pending: "
               f"{len(rows) - done}.")
    out.append("")
    out.append("| layer | module | defs | step(s) | defs (**done**, others pending) |")
    out.append("|---|---|--:|---|---|")
    by_mod: dict[str, list[dict]] = {}
    for r in rows:
        by_mod.setdefault(r["target"], []).append(r)
    order = list(dict.fromkeys(LAYERS))
    for mod in sorted(by_mod, key=lambda m: (list(dict.fromkeys(LAYERS.values())).index(LAYERS[m]),
                                             order.index(m))):
        rs = by_mod[mod]
        steps = ",".join(str(s) for s in sorted({r["step"] for r in rs}))
        names = ", ".join((f"**`{r['name']}`**" if r["status"] == "done" else f"`{r['name']}`")
                          for r in sorted(rs, key=lambda r: (r["step"], r["name"])))
        out.append(f"| {LAYERS[mod]} | `{mod}` | {len(rs)} | {steps} | {names} |")
    out.append("")
    per_step = {}
    for r in rows:
        d = per_step.setdefault(r["step"], [0, 0])
        d[0 if r["status"] == "done" else 1] += 1
    out.append("Defs per step (done / pending): " + "; ".join(
        f"{k}: {v[0]} / {v[1]}" for k, v in sorted(per_step.items())) + ".")
    return "\n".join(out)


def knobs_markdown(knobs: list[dict]) -> str:
    """The knob census: depends on the compose files and the whole tree (refreshed by --write)."""
    out = []
    classes = ("live", "armed", "legacy", "explore", "dead")
    cnt = {c: sum(1 for k in knobs if k["class"] == c) for c in classes}
    out.append(f"`AW_*` knobs read by the server (string literals, comments excluded): {len(knobs)}: "
               + ", ".join(f"{c} {cnt[c]}" for c in classes) + ". Classes: module docstring of "
               "`cytir.ops.serve_split_map`; `weights:<key>` = the default comes from the weights file, "
               "so only the env override is unused.")
    out.append("")
    for cls in classes[2:]:
        names = [f"`{k['knob']}`" + (f" (`weights:{k['weights_default']}`)" if k["weights_default"] else "")
                 for k in knobs if k["class"] == cls]
        out.append(f"{cls} ({len(names)}): " + ", ".join(names) + ".")
        out.append("")
    out.pop()
    return "\n".join(out)


def _block(spec: pathlib.Path) -> tuple[str, str, str]:
    text = spec.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{spec}: no generated block markers")
    head, rest = text.split(BEGIN, 1)
    body, tail = rest.split(END, 1)
    return head, body, tail


def write_block(spec: pathlib.Path, block: str) -> bool:
    head, body, tail = _block(spec)
    new = head + BEGIN + "\n\n" + block + "\n\n" + END + tail
    if new == spec.read_text(encoding="utf-8"):
        return False
    spec.write_text(new, encoding="utf-8")
    return True


def defs_current(spec: pathlib.Path, rows: list[dict]) -> bool:
    """True when the spec's generated block holds today's def table. The knob census is not
    compared: it moves with every compose or tree change, so `--write` refreshes it with a step."""
    return defs_markdown(rows) in _block(spec)[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cyt ops serve-split-map",
                                 description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--json", action="store_true")
    g.add_argument("--write", metavar="SPEC")
    g.add_argument("--check", metavar="SPEC")
    a = ap.parse_args(argv)
    rows, problems = def_map()
    for p in problems:
        print(f"serve_split_map: {p}", file=sys.stderr)
    if a.check:
        if not defs_current(pathlib.Path(a.check), rows):
            print(f"serve_split_map: the def table in {a.check} is stale; run --write", file=sys.stderr)
            return 1
        return 1 if problems else 0
    knobs = knob_census()
    if a.json:
        print(json.dumps({"defs": rows, "knobs": knobs, "problems": problems}, indent=1))
        return 1 if problems else 0
    block = defs_markdown(rows) + "\n\n" + knobs_markdown(knobs)
    if a.write:
        write_block(pathlib.Path(a.write), block)
    else:
        print(block)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
