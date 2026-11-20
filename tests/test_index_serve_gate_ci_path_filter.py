"""The `index-serve-gate` CI job's path filter covers the import closure of the three verbs it runs
(`cyt index build`, `cyt serve`, `cyt gate`), the server it starts, and the data it reads (slice 8
step 5). On a pull request the job does real work only when a file matched by `INDEX_GATE_PATHS`
changed (a skipped run is a green check), so a closure file the filter misses lets a change skip the
end-to-end gate until it is on `main`.

The closure is DERIVED with the same static walk as `test_drain_ci_path_filter` (function-level and
relative imports, every parent package's `__init__.py`, the `data/` files the sources name by name),
from the three verbs' modules. `cytir/cli.py` is an explicit file instead of a root: its lazy imports
reach every verb, and the job runs it as the dispatcher of exactly these three. The server is a
subprocess of `cyt serve`, so every file under `cytir/serve/` is in. The same limits apply as
there (a computed path, a data file named only in a wrapper, `importlib.import_module(<variable>)`).
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

import test_drain_ci_path_filter as D  # noqa: E402  (the closure walk, one implementation)

ROOTS = {"cytir.index.build", "cytir.ops.serve", "cytir.gates.slice_gate"}
#: what the job runs and reads that is not an `aw.*` module
OWN_FILES = [
    "cytir/cli.py", "cytir/ops/serving_env.default.json", "pyproject.toml", "conftest.py", "pytest.ini",
    "tests/conftest.py", "tests/throwaway_qdrant.py", "tests/standin_embedder.py",
    "tests/test_aw_index_serve_gate.py", "tests/test_aw_serve_parity_fixture.py", "tests/standin_tei_server.py", "tests/test_aw_gate.py", "tests/test_aw_serving_env_default.py",
    "tests/test_aw_source_stage_out.py", "cytir/sources/stage_out.py",
    "data/samples/dila-kali/global/KALI/KALIARTI000005833790.xml", "data/index-fixture/dila_kali/PACK.json",
    "data/samples/stackexchange-nonso/SOURCE.md", "data/index-fixture/stackexchange_nonso_qa_p0/PACK.json",
    "data/samples/openalex-academic/SOURCE.md", "data/index-fixture/openalex_academic_dump/PACK.json",
    "data/samples/stackoverflow-qa/SOURCE.md", "data/index-fixture/stackoverflow_qa/PACK.json",
    "data/samples/service-public-fiche/SOURCE.md", "data/index-fixture/service_public_fiche/PACK.json",
    "tests/test_aw_serve.py", "data/gate-slices/ci-fixture.v1.json",
    "data/gate-slices/ci-fixture.baseline.v1.json", "data/index-fixture/loda_exact/PACK.json",
    "data/index-fixture/fedlex_exact/chunks/shard-0000.texts.jsonl.gz",
    "data/serving-profiles/aw_e9.config.v1.json",
    "deploy/docker-compose.yaml", "deploy/docker-compose.flip.yaml", "deploy/compose-overlays.manifest",
    "tests/test_aw_release_serving_gate.py", "tests/test_aw_serve_parity.py",
    "deploy/scripts/release-serving.sh", "deploy/scripts/rebuild-serving-image.sh",
    "deploy/scripts/serving-gate.sh", "tests/testdata/task758_serve_parity_main_vs_head.json",
    ".github/workflows/ci.yml", ".github/scripts/throwaway-qdrant.sh", ".github/scripts/path-gate.sh",
]


def _filter() -> re.Pattern:
    m = re.search(r"^\s*INDEX_GATE_PATHS: '(.*)'\s*$", CI.read_text(), re.M)
    assert m, "INDEX_GATE_PATHS not found in ci.yml"
    return re.compile(m.group(1))


def _match(flt: re.Pattern, f: str) -> bool:
    return bool(flt.match(f if f.startswith(".github") else f"authoritative-web-app/{f}"))


def closure_files() -> set[str]:
    files: set[str] = {"cytir/__init__.py"}
    for m in D._closure_modules(set(ROOTS)):
        p = D._module_file(m)
        assert p is not None, m
        files.add(str(p.relative_to(APP)))
    named = {p.relative_to(APP / "data").as_posix(): f"data/{p.relative_to(APP / 'data').as_posix()}"
             for p in sorted((APP / "data").rglob("*")) if p.is_file()}
    for src in [APP / f for f in files]:
        for n in ast.walk(ast.parse(src.read_text())):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in named:
                files.add(named[n.value])
    files |= {f"cytir/serve/{p.name}" for p in (APP / "cytir" / "serve").iterdir()
              if p.is_file() and p.suffix in (".py", ".json")}
    return files


def test_every_file_of_the_three_verbs_closure_matches_the_path_filter():
    flt = _filter()
    files = closure_files()
    assert len(files) > 60, "the closure came out implausibly small"
    for must in ("cytir/index/build.py", "cytir/index/idf_table.py", "cytir/ops/serve.py", "cytir/ops/prod_env.py",
                 "cytir/gates/slice_gate.py", "cytir/gates/authority.py", "cytir/core/serving_profile.py",
                 "cytir/distribute/idf.py", "cytir/serve/retrieval.py", "cytir/serve/bm25_idf.json"):
        assert must in files, must
    missing = sorted(f for f in files | set(OWN_FILES) if not _match(flt, f))
    assert not missing, (
        f"the index-serve-gate job's INDEX_GATE_PATHS (.github/workflows/ci.yml) does not match "
        f"{missing}: a pull request that changes one of them would skip the end-to-end gate")


_PRIVATE = ("tasks", "pipeline")   # trees the filter must not match


def test_the_filter_is_not_a_catch_all():
    flt = _filter()
    for f in ("plan.md", "log.md", "README.md", "cytir/ops/dump.py", "cytir/news/publishers.py",
              "cytir/embed/shards.py", "data/README.md", f"{_PRIVATE[0]}/696-x.md", f"{_PRIVATE[1]}/fedlex_dump.py"):
        assert not _match(flt, f), f


def test_the_two_qdrant_jobs_pin_the_same_release_and_use_the_one_script():
    text = CI.read_text()
    shas = re.findall(r"^\s*QDRANT_SHA256: ([0-9a-f]{64})", text, re.M)
    vers = re.findall(r"^\s*QDRANT_VERSION: (v[\d.]+)", text, re.M)
    assert len(shas) == len(vers) == 2 and len(set(shas)) == 1 and len(set(vers)) == 1, (shas, vers)
    assert text.count("bash ../.github/scripts/throwaway-qdrant.sh start") == 2
    assert text.count("bash ../.github/scripts/throwaway-qdrant.sh stop") == 2
    assert "curl -fsSL --retry 3 -o \"$q/qdrant.tar.gz\"" not in text      # no second copy of the start
    assert (APP.parent / ".github" / "scripts" / "throwaway-qdrant.sh").is_file()
