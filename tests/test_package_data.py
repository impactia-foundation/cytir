"""Task 733: a non-editable `pip install .` must find every `data/` file the package reads.

Four holds:
  1. the wheel's data list (pyproject `[tool.setuptools.package-data] "cytir._data"`) equals the
     public cut's runtime-data rules (`data-contracts`, `release-rehearsal`) -- one list, and a file
     the cut withholds (held-pending-anonymization.*, eval inputs, tasks) cannot reach a wheel;
  2. every reader that was moved onto `cytir.core.paths.data_path` resolves an EXISTING file, in a
     checkout and in an installed layout (cytir/_data beside the code, run from `/`);
  3. the readers that still read a checkout path are a named, shrink-only list (OWED_CHECKOUT_ONLY);
     a NEW module that reads a shipped data file without `data_path` fails here;
  4. `data_dir()` says which source is in force.
The real `pip install .` of the exported tree, from `/`, is the CI step `cut-rehearsal`.
"""
from __future__ import annotations

import ast
import fnmatch
import importlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.guard

APP = pathlib.Path(__file__).resolve().parents[1]
RUNTIME_RULES = ("data-contracts", "release-rehearsal")

#: (module, attribute) pairs that hold a path under data/ and now come from data_path().
CONVERTED = [
    ("cytir.chunk.policy", "DEFAULT_POLICY_PATH"),
    ("cytir.sources.soft404", "DEFAULT_PATH"),
    ("cytir.sources.pack_loader", "PRESETS_PATH"),
    ("cytir.index.payload_profile", "PROFILES_PATH"),
    ("cytir.index.payload_profile", "TABLES_PATH"),
    ("cytir.gates.convergence", "CONTRACT_PATH"),
    ("cytir.corpus.persisted", "DEFAULT_REGISTRY"),
    ("cytir.corpus.persisted", "DEFAULT_REGISTER"),
    ("cytir.distribute.rehearsal", "DEFAULT_FIXTURE_DIR"),
]

#: Modules that cannot be imported from a copied installed layout. Empty since task 735: the pack
#: loader imports the validator from the package and looks for `sources/` only when a pack is read.
NEEDS_SOURCES_TREE: set = set()

#: Modules that read a shipped data file through a CHECKOUT path (parents[N], a repo root, a
#: release dir). They work in a checkout and in a release checkout; they do NOT yet work from a
#: non-editable install. Each is owned by a lane that was running when task 733 landed. The list
#: only shrinks: when one moves to data_path() the test below tells you to delete its line.
OWED_CHECKOUT_ONLY = {
    "cytir/consent/block.py": "consent_declined_hosts.json",
    "cytir/consent/resolver.py": "consent-unresolved-register.yaml",
    "cytir/consent/stamp.py": "source-family-contracts.v1.json",
    "cytir/index/drain/queue_scope.py": "promotion-gate-contract.v1.json",
    "cytir/sources/families/fedlex_law_meta.py": "fedlex-law-names.v1.json",
    # reads through the RELEASE checkout (`env.release_dir`), by design: a release is a checkout
    "cytir/ops/assembly_guards.py": "adr101-class-precedence.v1.json",   # task 2277: reads through the GUARDED release root (ctx.root), by design
    "cytir/sources/families/news_tier2_apply.py": "source-preservation-profiles.v1.json",
    "cytir/index/apply.py": "news-daily-gate-contract.v1.json",   # task 2379: reads through the RELEASE checkout (args.release_dir), by design
    "cytir/news/seal_source_evidence.py": "source-preservation-profiles.v1.json",  # task 2054: the seal hashes the RELEASE's registry copy
    "cytir/ops/finma_exact_admission.py": "chunk-policy-struct-md",     # task 1396: the RELEASE's policy
    "cytir/ops/audits/finma_web_pdf_profile.py": "chunk-policy-struct-md",  # task 1523: literal compared with the pack's recorded path, no read
    # maintainer checks that take an explicit repo=/registry_path= argument
    "cytir/sources/stackoverflow/audit_profile.py": "chunk-policy-struct-md",
    # maintainer tool: scores the CHECKOUT (plan.md, specs/, git) -- meaningless outside one
    "cytir/ops/refactor_scoreboard.py": "source-family-contracts.v1.json",
    "cytir/ops/pipeline_reach_census.py": "script-entrypoints.v1.json",  # task 1854: the census scans the checkout tree (its subject is the repo), not installed package data
}


def _package_data_globs() -> list[str]:
    # No TOML library on Python 3.10 (and no setuptools in a bare venv): read the one list.
    text = (APP / "pyproject.toml").read_text("utf-8")
    block = text[text.index("[tool.setuptools.package-data]"):]
    m = re.search(r'^"cytir\._data"\s*=\s*\[(.*?)^\]', block, re.S | re.M)
    assert m, "pyproject.toml has no [tool.setuptools.package-data] \"cytir._data\" list"
    return re.findall(r'"([^"]+)"', m.group(1))


def _cut_runtime_globs() -> list[str]:
    cut = json.loads((APP / "data" / "public-cut.v1.json").read_text("utf-8"))
    out = []
    for rule in cut["rules"]:
        if rule["id"] in RUNTIME_RULES:
            out += [p[len("data/"):] for p in rule["paths"]]
    return out


def test_wheel_data_equals_the_cuts_runtime_data():
    assert sorted(_package_data_globs()) == sorted(_cut_runtime_globs())


def test_withheld_files_never_match_a_wheel_glob():
    globs = _package_data_globs()
    withheld = ["held-pending-anonymization.v1.yaml", "held-pending-anonymization.md",
                "eval_tasks.yaml", "oss-refactor-live-entrypoints.v1.json",
                "spam-host-lists.v1.json"]
    for name in withheld:
        assert not any(fnmatch.fnmatch(name, g) for g in globs), name


def test_every_matched_glob_matches_something():
    for g in _package_data_globs():
        assert list((APP / "data").glob(g)), f"package-data glob matches no file: {g}"


def test_setup_py_maps_data_onto_aw_data():
    assert (APP / "setup.py").is_file()
    assert "cytir._data" in (APP / "setup.py").read_text("utf-8")
    assert '"cytir._data" = "data"' in (APP / "pyproject.toml").read_text("utf-8")


@pytest.mark.parametrize("module,attr", CONVERTED)
def test_converted_reader_resolves_an_existing_file(module, attr):
    from cytir.core.paths import data_dir
    p = pathlib.Path(str(getattr(importlib.import_module(module), attr)))
    assert p.exists(), p
    assert data_dir() in p.parents


def test_data_dir_is_the_checkout_here(monkeypatch):
    from cytir.core import paths
    monkeypatch.delenv("AW_REPO_ROOT", raising=False)
    assert paths.data_dir() == paths.REPO_ROOT / "data"
    assert "[checkout]" in paths.data_dir_line()


def test_data_dir_falls_back_to_package_data_without_a_checkout(monkeypatch, tmp_path):
    from cytir.core import paths
    monkeypatch.delenv("AW_REPO_ROOT", raising=False)
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)       # a site-packages dir: no pyproject
    d = paths.data_dir()
    assert d.name == "_data" and d.parent.name == "cytir"
    assert "installed package data" in paths.data_dir_line()


def test_aw_repo_root_overrides(monkeypatch, tmp_path):
    from cytir.core import paths
    monkeypatch.setenv("AW_REPO_ROOT", str(tmp_path))
    assert paths.data_dir() == tmp_path / "data"


def test_installed_layout_reads_every_converted_file_from_slash(tmp_path):
    """cytir/ copied next to a data copy at cytir/_data, no pyproject beside it, cwd = / : the layout of
    a non-editable install. Every converted reader finds its file and the registry loads."""
    site = tmp_path / "site-packages"
    shutil.copytree(APP / "cytir", site / "cytir", ignore=shutil.ignore_patterns("__pycache__"))
    data = site / "cytir" / "_data"
    data.mkdir()
    for g in _package_data_globs():
        # setuptools' `dir/**` means every file below dir; Path.glob("dir/**") yields only the
        # directories before Python 3.13, so expand it by hand (this broke the 3.10 CI venv).
        pattern = g[:-3] + "/*" if g.endswith("/**") else g
        hits = (APP / "data" / g[:-3]).rglob("*") if g.endswith("/**") else (APP / "data").glob(pattern)
        for src in hits:
            if src.is_file():
                dst = data / src.relative_to(APP / "data")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import importlib, pathlib, cytir.core.paths as P\n"
        "assert P.data_dir() == pathlib.Path(%r) / 'cytir' / '_data', P.data_dir()\n"
        "for m, a in %r:\n"
        "    p = pathlib.Path(str(getattr(importlib.import_module(m), a)))\n"
        "    assert p.exists() and P.data_dir() in p.parents, (m, a, p)\n"
        "from cytir.chunk import registry\n"
        "assert registry.load()['schema'] == registry.SCHEMA\n"
        "print('ok')\n") % (str(site), str(site), [c for c in CONVERTED if c[0] not in NEEDS_SOURCES_TREE])
    r = subprocess.run([sys.executable, "-c", code], cwd="/", capture_output=True, text=True,
                       env={k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "AW_REPO_ROOT")})
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-1500:]


def _data_path_literals(py: pathlib.Path, names: set[str]) -> list[str]:
    """String constants in `py` that are exactly a shipped data file name (optionally `data/`
    prefixed or a `*.glob` pattern start). Docstrings, comments and prose messages do not match."""
    tree = ast.parse(py.read_text("utf-8"))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value.removeprefix("data/")
            if v in names or v.startswith(("chunk-policy-", "chunk-reference-selector")) and v.endswith("*.json"):
                hits.append(node.value)
    return hits


#: Names a module WRITES or RECORDS, never reads (task 739). Not a checkout read, so not in
#: OWED_CHECKOUT_ONLY (that list is shrink-only and is for real reads). Each entry is checked
#: below by an AST rule: the literal, and every name bound to it, is never a call argument (review 807 S1);
#: the set's size is pinned.
FROZEN_ARTIFACT_NAMES = {
    ("cytir/index/seal_content_pack_intent.py", "source-preservation-profiles.v1.json"):
        "output file name inside the sealed tree (not read)",
    ("cytir/news/seal_lane_stages.py", "data/chunk-policy-legacy-180w-o30.v1.json"):
        "frozen policy name recorded in receipts (not read)",
    # frozen-label exemption for the contracts file (task 1390): a name in FROZEN_RECEIPT_LABELS, not a read
    ("cytir/ops/public_cut.py", "data/source-family-contracts.v1.json"):
        "frozen label in FROZEN_RECEIPT_LABELS (not read)",
    ("cytir/ops/audits/news_article_profile.py", "chunk-policy-news-article-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json"):
        "declared-label comparator (the read uses the sibling basename NEWS_POLICY_FILE; task 1507/batch-g)",
    ("cytir/ops/audits/news_article_profile.py", "data/chunk-policy-news-article-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json"):
        "declared-label comparator (the read uses the sibling basename NEWS_POLICY_FILE; task 1507/batch-g)",
    ("cytir/ops/audits/news_article_profile.py", "chunk-policy-legacy-180w-o30.v1.json"):
        "declared-label comparator (the read uses the sibling basename LEGACY_POLICY_FILE; task 1507/batch-g)",
    ("cytir/ops/audits/news_article_profile.py", "data/chunk-policy-legacy-180w-o30.v1.json"):
        "declared-label comparator (the read uses the sibling basename LEGACY_POLICY_FILE; task 1507/batch-g)",
    ("cytir/sources/stackoverflow/successor_chain.py", "data/chunk-policy-struct-md-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json"):
        "argv value for a child process under RELEASE_DIR, never read here (task 1993)",
    ("cytir/sources/stackoverflow/successor_chain.py", "data/source-preservation-profiles.v1.json"):
        "argv value for a child process under RELEASE_DIR, never read here (task 1993)",
}
#: A call to one of these with the name as an argument is not a read.
_NOT_READS = {"dict", "str", "repr"}


def _frozen_name_is_never_read(rel: str, literal: str) -> bool:
    """AST rule (review 807 S1): the literal, and every name bound to it (aliases, annotated
    constants, chains), is never an argument of a call -- `open(X)`, `Path(X).read_text()`,
    `CP.load(Path(X))`, a call split over lines -- except to dict/str/repr. A Dict or f-string is a
    record of the name, not a use of it, and is not searched. Limit: a path built from the name by
    an operator (`root / X`) and bound to another name is not followed."""
    tree = ast.parse((APP / rel).read_text("utf-8"))

    def mentions(node: ast.AST, names: set[str]) -> bool:
        if isinstance(node, (ast.Dict, ast.JoinedStr)):
            return False
        if isinstance(node, ast.Constant):
            return node.value == literal
        if isinstance(node, ast.Name):
            return node.id in names
        return any(mentions(c, names) for c in ast.iter_child_nodes(node))

    names: set[str] = set()
    grew = True
    while grew:                                     # aliases, to a fixed point
        grew = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Constant, ast.Name)):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if mentions(value, names):
                for t in targets:
                    if isinstance(t, ast.Name) and t.id not in names:
                        names.add(t.id)
                        grew = True
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = node.func.id if isinstance(node.func, ast.Name) else None
            if callee in _NOT_READS:
                continue
            if any(mentions(a, names) for a in [node.func, *node.args, *(k.value for k in node.keywords)]):
                return False
    return True


def test_frozen_artifact_names_are_names_not_reads():
    assert len(FROZEN_ARTIFACT_NAMES) == 9, "a new exemption needs the orchestrator's decision"
    for (rel, literal), why in FROZEN_ARTIFACT_NAMES.items():
        assert why and (APP / rel).exists(), rel
        assert literal in _data_path_literals(APP / rel, {literal.removeprefix("data/")}), \
            "stale exemption (literal gone): %s" % rel
        assert _frozen_name_is_never_read(rel, literal), "%s reads %s: not a frozen name" % (rel, literal)


def test_no_unlisted_module_reads_shipped_data_through_a_checkout_path():
    names = set()
    for g in _package_data_globs():
        if "**" not in g and "*" not in g:
            names.add(g)
    names |= {p.name for p in (APP / "data").glob("chunk-policy-*.json")}
    offenders, still_owed = [], set()
    for py in sorted((APP / "cytir").rglob("*.py")):
        rel = py.relative_to(APP).as_posix()
        if rel == "cytir/core/paths.py":
            continue
        hit = _data_path_literals(py, names)
        if not hit:
            continue
        text = py.read_text("utf-8")
        if "data_path" in text or "data_dir" in text:
            if rel in OWED_CHECKOUT_ONLY:
                still_owed.add(rel)
            continue
        if rel not in OWED_CHECKOUT_ONLY:
            hit = [h for h in hit if (rel, h) not in FROZEN_ARTIFACT_NAMES]
            if hit:
                offenders.append((rel, hit[0]))
    assert not offenders, (
        "these modules name a shipped data/ file but do not resolve it with "
        "cytir.core.paths.data_path(); a non-editable install cannot find it: %s" % offenders)
    assert not still_owed, (
        "these now use data_path(): delete them from OWED_CHECKOUT_ONLY: %s" % sorted(still_owed))
    # (an OWED file the public cut withholds, e.g. the maintainer scoreboard, is simply absent there)
