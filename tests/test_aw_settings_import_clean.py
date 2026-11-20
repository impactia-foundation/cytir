"""No import needs a configured box (task 696 slice 4, round 2; review of #563/#567).

With no operator overlay, no box file, and no setting variable, EVERY `aw` module must import: a
setting nobody configured refuses by name where it is USED, never when a module loads. And importing
creates nothing: not in the cwd, not in `$HOME`.

The imports run in ONE fresh subprocess with a scrubbed environment, so a module that cached a value
at import in this test process cannot hide a refusal. (The live entry scripts of the operator's
`deploy/` are covered the same way by the operator-side test next to the pipeline.)
"""
from __future__ import annotations

import json
import os
import pathlib
import site
import subprocess
import sys

import pytest

APP = pathlib.Path(__file__).resolve().parents[1]

_DRIVER = r'''
import importlib, importlib.util, json, pathlib, sys
app = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(app))
sys.path.insert(1, str(app / "pipeline"))
out = {"ok": [], "missing_dependency": {}, "missing_data": {}, "failed": {}}
names = json.loads(sys.argv[2])
scripts = json.loads(sys.argv[3])

def record(label, fn):
    try:
        fn()
        out["ok"].append(label)
    except ModuleNotFoundError as exc:
        top = (exc.name or "").split(".")[0]
        if top and top not in ("cytir", "pipeline"):
            out["missing_dependency"][label] = exc.name      # an optional third-party package
        else:
            out["failed"][label] = f"{type(exc).__name__}: {exc}"
    except FileNotFoundError as exc:                         # a data file this tree does not ship
        out["missing_data"][label] = str(exc)[:200]          # (never a setting: those refuse first)
    except BaseException as exc:                             # noqa: BLE001  (SystemExit too)
        out["failed"][label] = f"{type(exc).__name__}: {str(exc)[:400]}"

for n in names:
    record(n, lambda n=n: importlib.import_module(n))
for rel in scripts:
    def load(rel=rel):
        p = app / rel
        spec = importlib.util.spec_from_file_location("plscript_" + p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod                 # dataclasses look the module up by name
        spec.loader.exec_module(mod)
    record(rel, load)
print("RESULT " + json.dumps(out))
'''

#: Modules whose import must not be skipped for a missing dependency: they carry the settings.
MUST_IMPORT = ("cytir.index.promote_queue", "cytir.index.compose", "cytir.news.daily", "cytir.news.hn_daily",
               "cytir.acquire.crawl", "cytir.ops.write_lock", "cytir.embed.stage", "cytir.corpus.store")


def aw_modules() -> list[str]:
    mods = []
    for p in sorted((APP / "cytir").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        parts = list(p.relative_to(APP).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts[-1] == "__main__":
            continue                                   # `python -m cytir` runs the CLI at import
        mods.append(".".join(parts))
    return mods


def run_clean_import(tmp_path_factory, names: list[str], scripts: list[str]):
    """Import `names` and load `scripts` in a fresh interpreter with no configuration; returns
    (result, cwd, home)."""
    for dep in ("qdrant_client", "httpx", "yaml", "lxml"):
        pytest.importorskip(dep)        # else the roster would shrink to "missing dependency"
    cwd = tmp_path_factory.mktemp("clean-cwd")
    home = tmp_path_factory.mktemp("clean-home")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home),
           "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONUSERBASE": site.getuserbase(),  # deps may live in the user site; HOME is scrubbed
           # tests only: replaces the whole default overlay lookup (no box file, no checkout file),
           # see cytir.core.paths.TEST_LOOKUP_ENV. No setting, path or release variable is passed.
           # The ONLY other names are a throwaway crawler IDENTITY: `cytir.consent.*` and
           # `cytir.acquire.sitemap_ingest` build their User-Agent at import and fail closed without
           # one, by design (task 432, `cytir.core.user_agent`); an identity contract, not a setting.
           "AW_TEST_OVERLAY_LOOKUP": "",
           "AW_OPERATOR_URL": "https://example.invalid/test",
           "AW_OPERATOR_CONTACT": "nobody@example.invalid"}
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(APP), json.dumps(names), json.dumps(scripts)],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=900)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line, f"the import driver died: rc={proc.returncode}\n{proc.stderr[-2000:]}"
    return json.loads(line[len("RESULT "):]), cwd, home


@pytest.fixture(scope="module")
def clean_import(tmp_path_factory):
    return run_clean_import(tmp_path_factory, aw_modules(), [])


def test_the_roster_is_derived_from_the_tree_and_not_small():
    assert len(aw_modules()) > 100


def test_every_aw_module_imports_with_no_configuration(clean_import):
    result, _, _ = clean_import
    assert not result["failed"], "\n".join(f"{k}: {v}" for k, v in sorted(result["failed"].items()))
    assert len(result["ok"]) > 100
    for must in MUST_IMPORT:
        assert must in result["ok"], (f"{must} was not imported: {result['missing_dependency']} "
                                      f"{result['missing_data']}")


def test_importing_creates_no_directory_anywhere(clean_import):
    _, cwd, home = clean_import
    assert list(cwd.iterdir()) == [], f"imports created {sorted(p.name for p in cwd.iterdir())} in the cwd"
    assert list(home.iterdir()) == [], f"imports created {sorted(p.name for p in home.iterdir())} in $HOME"
