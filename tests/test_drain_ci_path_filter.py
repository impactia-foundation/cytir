"""Round 4, follow-up 8 of #553: the Qdrant CI job's path filter covers the drain's whole import
closure. The job is the only end-to-end run of the drain (the golden, the Single transaction, the
build identity), and on a pull request it does real work only when a file matched by `PROFILE_PATHS`
changed. A closure file the filter does not match lets a change skip the golden until it is on
`main`, and the hourly drain takes `origin/main`.

The closure is DERIVED, never hand-listed: the static `aw.*` import closure of the drain and of the
harness that drives it (function-level imports included, RELATIVE imports followed, and every parent
package's `__init__.py` added: importing `cytir.embed.stage` runs `cytir/embed/__init__.py`), plus every
file under `data/` that those sources name by file name or by path relative to `data/` (the gate
contract, the payload-profile declaration, the serving-profile fixtures), plus the harness, the golden
and the fixtures. The test fails when the closure gains a file the filter does not match.

What it CANNOT see (a limit, not a pass; round 5, follow-up B of #553): a file loaded by a path that
is COMPUTED (`cytir/sources/pack_loader.py` loads `sources/_schema/validate.py` this way, and the drain
roots do not load it at run time, so `sources/` is deliberately not in the filter); a data file named
only in a shell wrapper or a YAML file; a data file read through a settings value that is not a
literal; and a module imported with `importlib.import_module(<variable>)`.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

import pytest

from tests.helpers.repo_files import repo_file_or_skip

HERE = pathlib.Path(__file__).resolve().parent
APP = HERE.parent
CI = repo_file_or_skip("../.github/workflows/ci.yml")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

#: The modules and tests that ARE the Qdrant job's subject, besides the closure files.
OWN_FILES = [
    "tests/drain_transaction_harness.py", "tests/test_drain_transaction.py",
    "tests/test_drain_single.py", "tests/test_aw_serving_profile.py",
    "tests/test_drain_build_identity.py", "tests/test_drain_build_guards.py", "tests/loda_stage_support.py",
    "tests/fixtures/legi_loda_full/sample_loi.xml",
    "tests/test_drain_build_refusals.py", "cytir/index/build.py",
    "tests/test_aw_serve.py", "tests/throwaway_qdrant.py",
    "tests/test_aw_index_mirror_guards.py", "tests/fixtures/mirror_guards/rows.v1.json",
    "tests/test_throwaway_qdrant_guard.py", "conftest.py", "pytest.ini",
    "tests/fixtures/drain_transaction/twin_first.golden.json",
    "tests/test_drain_content_pack_golden.py", "tests/fixtures/drain_content_pack/points.jsonl",
    "tests/fixtures/dila_juri/x.xml", "data/serving-profiles/aw_e9.config.v1.json",
    ".github/workflows/ci.yml",
]
DRAIN_ROOTS = ["cytir.index.promote_queue", "cytir.ops.write_lock", "cytir.index.live_target",
               "cytir.sources.families.dila_juri"]


def _filter() -> re.Pattern:
    m = re.search(r"^\s*PROFILE_PATHS: '(.*)'\s*$", CI.read_text(), re.M)
    assert m, "PROFILE_PATHS not found in ci.yml"
    return re.compile(m.group(1))


def _module_file(module: str) -> pathlib.Path | None:
    base = APP / module.replace(".", "/")
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.exists():
            return cand
    return None


def _imports_of(module: str, path: pathlib.Path) -> set[str]:
    """Every `aw.*` module the source imports, anywhere in it: absolute and RELATIVE imports, each
    `from X import name` also as `X.name` (a submodule), function-level imports included. `module`
    is the dotted name of the file (its package is itself for an `__init__.py`)."""
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    out: set[str] = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names if a.name.split(".")[0] == "cytir"}
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                base = package.split(".")
                base = base[: len(base) - (n.level - 1)]
                mod = ".".join([*base, *([n.module] if n.module else [])])
            else:
                mod = n.module or ""
            if mod.split(".")[0] == "cytir":
                out.add(mod)
                out |= {f"{mod}.{a.name}" for a in n.names}
    return out


def _closure_modules(roots: set[str]) -> set[str]:
    """The static closure; every parent package of a member is a member (its `__init__.py` runs)."""
    seen: set[str] = set()
    todo = sorted(roots)
    while todo:
        m = todo.pop()
        path = _module_file(m)
        if m in seen or path is None:
            continue
        seen.add(m)
        parts = m.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]
        todo += sorted(_imports_of(m, path))
    return seen


def closure_files() -> set[str]:
    """Repo-relative (to the app dir) files of the drain's and the harness's static import closure,
    and the `data/` files their sources name."""
    harness = APP / "tests" / "drain_transaction_harness.py"
    roots = set(DRAIN_ROOTS) | {m for m in _imports_of("tests.drain_transaction_harness", harness)}
    files: set[str] = {"cytir/__init__.py"}
    for m in _closure_modules(roots):
        p = _module_file(m)
        assert p is not None, m
        files.add(str(p.relative_to(APP)))
    # a data file is named by its path relative to data/ (`serving-profiles/x.json`); a file at the
    # top level of data/ is named by its bare name (a nested file is never matched by a bare name:
    # `manifest.json` names a pack's manifest too)
    named = {p.relative_to(APP / "data").as_posix(): f"data/{p.relative_to(APP / 'data').as_posix()}"
             for p in sorted((APP / "data").rglob("*")) if p.is_file()}
    sources = [APP / f for f in files] + [harness]
    for src in sources:
        for n in ast.walk(ast.parse(src.read_text())):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in named:
                files.add(named[n.value])
    return files


def test_every_file_of_the_drains_import_closure_matches_the_ci_path_filter():
    flt = _filter()
    files = closure_files()
    assert len(files) > 40, "the closure came out implausibly small"
    assert "cytir/embed/__init__.py" in files, "a parent package's __init__.py is part of the closure"
    # the gate contract and the payload declarations are the data files the drain reads
    assert {"data/promotion-gate-contract.v1.json", "data/collection-payload-profiles.v1.json"} <= files
    missing = sorted(f for f in files | set(OWN_FILES) if not flt.match(f if f.startswith(".github") else f"authoritative-web-app/{f}"))
    assert not missing, (
        f"the Qdrant job's PROFILE_PATHS (.github/workflows/ci.yml) does not match {missing}: a pull "
        "request that changes one of them would skip the drain golden until it is on main")


def test_the_filter_is_not_a_catch_all():
    """A filter that matches everything proves nothing (and would burn a runner slot per PR)."""
    flt = _filter()
    for f in ("authoritative-web-app/plan.md", "authoritative-web-app/cytir/cli.py",
              "authoritative-web-app/cytir/ops/dump.py", "authoritative-web-app/README.md"):
        assert not flt.match(f), f
