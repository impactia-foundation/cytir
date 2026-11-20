"""Task 696 slice 9 -- what the shell starters rely on in the package.

1. `python -m cytir.core.paths --get NAME` prints one setting, for a shell script that takes a box path
   from the settings (environment -> overlay -> neutral default), and refuses by name.
2. `cytir.news.source_queue` is the one reader of the checked-in news source queue.
3. A module started with `python -m` from a work directory that holds stray `.py` files (a box
   work directory does) must not import one of them: no module in the closure of a package entry
   the scripts start resolves a bare name that a work directory could shadow.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from cytir.core import paths as P
from cytir.news import source_queue as Q

APP = pathlib.Path(__file__).resolve().parents[1]


def _run(args, *, env=None, cwd="/"):
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(APP),
            "PYTHONDONTWRITEBYTECODE": "1", "AW_TEST_OVERLAY_LOOKUP": ""}
    base.update(env or {})
    return subprocess.run([sys.executable, *args], env=base, cwd=cwd, capture_output=True,
                          text=True, timeout=120)


def _overlay(tmp_path, settings):
    f = tmp_path / "overlay.json"
    f.write_text(json.dumps({"schema": P.OVERLAY_SCHEMA, "settings": settings}))
    return f


# -------------------------------------------------------------------------------- --get

def test_get_prints_only_the_value_from_the_environment(tmp_path):
    r = _run(["-m", "cytir.core.paths", "--get", "news_work"], env={"AW_NEWS_WORK": "/w/news"})
    assert (r.returncode, r.stdout, r.stderr) == (0, "/w/news\n", "")


def test_get_reads_the_overlay_when_the_environment_is_silent(tmp_path):
    ov = _overlay(tmp_path, {"work_root": "/w", "qpy": "/venv/bin/python", "hdd_root": "/hdd"})
    env = {"AW_OPERATOR_OVERLAY": str(ov)}
    assert _run(["-m", "cytir.core.paths", "--get", "qpy"], env=env).stdout == "/venv/bin/python\n"
    assert _run(["-m", "cytir.core.paths", "--get", "hdd_root"], env=env).stdout == "/hdd\n"
    # derived from the work root, like every other reader of the setting
    assert _run(["-m", "cytir.core.paths", "--get", "news_work"], env=env).stdout == "/w/newsfresh\n"


def test_environment_wins_over_the_overlay(tmp_path):
    ov = _overlay(tmp_path, {"qpy": "/from/overlay"})
    r = _run(["-m", "cytir.core.paths", "--get", "qpy"],
             env={"AW_OPERATOR_OVERLAY": str(ov), "AW_QPY": "/from/env"})
    assert r.stdout == "/from/env\n"


def test_get_refuses_by_name_when_nobody_configured_it():
    r = _run(["-m", "cytir.core.paths", "--get", "langgate_py"])
    assert r.returncode == 2 and r.stdout == ""
    assert "[REFUSED]" in r.stderr and "langgate_py" in r.stderr and "CYTIR_NEWS_LANGGATE_PY" in r.stderr


def _one_refusal_line(r, setting):
    assert r.returncode == 2 and r.stdout == "", (r.returncode, r.stdout, r.stderr)
    lines = r.stderr.splitlines()
    assert len(lines) == 1 and lines[0].startswith(f"[REFUSED] {setting}: "), r.stderr
    assert "Traceback" not in r.stderr


def test_get_malformed_overlay_is_one_refusal_line_not_a_traceback(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")
    _one_refusal_line(_run(["-m", "cytir.core.paths", "--get", "qpy"], env={"AW_OPERATOR_OVERLAY": str(bad)}),
                      "qpy")


def test_get_wrong_schema_overlay_is_one_refusal_line(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other", "settings": {}}))
    _one_refusal_line(_run(["-m", "cytir.core.paths", "--get", "qpy"], env={"AW_OPERATOR_OVERLAY": str(bad)}),
                      "qpy")


def test_get_overlay_with_an_unknown_key_is_one_refusal_line(tmp_path):
    ov = _overlay(tmp_path, {"work_root": "/w", "no_such_setting": "x"})
    r = _run(["-m", "cytir.core.paths", "--get", "work_root"], env={"AW_OPERATOR_OVERLAY": str(ov)})
    _one_refusal_line(r, "work_root")
    assert "no_such_setting" in r.stderr


def test_get_a_named_overlay_that_is_not_a_file_is_one_refusal_line(tmp_path):
    r = _run(["-m", "cytir.core.paths", "--get", "work_root"],
             env={"AW_OPERATOR_OVERLAY": str(tmp_path / "missing.json")})
    _one_refusal_line(r, "work_root")


def test_get_an_unset_path_setting_is_one_refusal_line():
    _one_refusal_line(_run(["-m", "cytir.core.paths", "--get", "work_root"]), "work_root")


def test_get_an_unknown_name_is_a_usage_error():
    r = _run(["-m", "cytir.core.paths", "--get", "nope"])
    assert r.returncode == 64 and "unknown setting 'nope'" in r.stderr
    assert _run(["-m", "cytir.core.paths", "--get"]).returncode == 64


def test_get_reads_the_two_roots_that_are_not_settings(tmp_path):
    r = _run(["-m", "cytir.core.paths", "--get", "data_root"], env={"AW_DATA_ROOT": "/d"})
    assert r.stdout == "/d\n"
    assert _run(["-m", "cytir.core.paths", "--get", "repo_root"], env={"AW_REPO_ROOT": "/r"}).stdout == "/r\n"
    assert set(P._ROOT_NAMES).isdisjoint(P._BY_NAME)        # no new overlay key, by construction


def test_the_no_argument_echo_is_unchanged():
    r = _run(["-m", "cytir.core.paths"])
    assert r.returncode == 0 and r.stdout.startswith("# operator overlay:")
    assert "work_root = REFUSED" in r.stdout


# -------------------------------------------------------------------------------- the queue reader

def test_queue_reader_points_at_the_checked_in_file_and_parses_it():
    assert Q.queue_path() == APP / "data" / "news-queue.example.json"
    assert Q.queue_relpath() == f"{Q.QUEUE_DIR}/{Q.QUEUE_NAME}"
    assert Q.queue_path().is_file(), "the queue ships in data/, so the public cut carries it (task 2107)"
    rows = Q.load_queue()
    assert rows and all("domain" in row for row in rows)
    assert Q.load_queue(APP) == rows


def test_the_roster_census_runs_from_the_tree_that_holds_the_queue():
    """Task 2107: the census reads the queue; in the public cut it used to find no file."""
    r = _run(["-m", "cytir.ops.news_census", "--help"])
    assert r.returncode == 0, r.stderr
    assert len(Q.load_queue(APP)) > 0


def test_queue_reader_takes_any_release_root(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "news-queue.example.json").write_text('[{"domain": "x.ch"}]')
    assert Q.load_queue(tmp_path) == [{"domain": "x.ch"}]
    r = _run(["-m", "cytir.news.source_queue", "path", str(tmp_path)])
    assert r.stdout == f"{tmp_path}/{Q.QUEUE_DIR}/{Q.QUEUE_NAME}\n"
    assert _run(["-m", "cytir.news.source_queue", "path"]).stdout.strip() == str(Q.queue_path())
    assert _run(["-m", "cytir.news.source_queue"]).returncode == 64


def test_the_census_and_the_release_file_list_use_the_one_reader(tmp_path):
    from cytir.news import release_preflight as RP
    from cytir.ops import news_census as NC
    (tmp_path / Q.QUEUE_DIR).mkdir()
    (tmp_path / Q.QUEUE_DIR / Q.QUEUE_NAME).write_text('[{"domain": "www.a.ch"}, {"domain": "b.ch"}]')
    (tmp_path / "sources").mkdir()
    roster = NC.declared_roster(tmp_path)
    assert roster["a.ch"] == [Q.QUEUE_NAME] and roster["b.ch"] == [Q.QUEUE_NAME]
    assert Q.queue_relpath() in RP.REQUIRED_RELEASE_FILES


# -------------------------------------------------------------------------------- a stray cwd

#: the package entries the shell scripts start with `-m`
ENTRIES = ["cytir.acquire.discovery_ingest", "cytir.acquire.sitemap_ingest", "cytir.consent.gate",
           "cytir.news.stage_contract", "cytir.news.release_preflight", "cytir.news.source_queue",
           "cytir.core.paths"]


@pytest.mark.parametrize("module", ENTRIES)
def test_a_package_entry_ignores_stray_modules_in_its_work_directory(module, tmp_path):
    """A box work directory holds old copies of pipeline modules (`aw_common.py`, `sitemap_ingest.py`,
    ...). `-m` puts the current directory first on the import path, unlike a start by file path. Every
    module name that exists as a pipeline file is planted here as a poison file; the entry must still
    import and print its usage."""
    pipeline = APP / "pipeline"
    names = {p.stem for p in pipeline.glob("*.py")} - {"__init__"}
    for name in names:
        (tmp_path / f"{name}.py").write_text("raise RuntimeError('a stray work-directory module was imported')\n")
    r = subprocess.run([sys.executable, "-m", module, "--help"], cwd=tmp_path, capture_output=True,
                       text=True, timeout=120,
                       env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(APP),
                            "PYTHONDONTWRITEBYTECODE": "1", "AW_TEST_OVERLAY_LOOKUP": "",
                            "AW_ALLOW_UNIDENTIFIED_CRAWL": "1"})     # import only; nothing is fetched
    assert "a stray work-directory module" not in r.stderr, r.stderr
    assert "Traceback" not in r.stderr, r.stderr
    assert r.returncode in (0, 2, 64), r.stderr


def test_no_package_module_imports_a_bare_pipeline_module_name():
    """The static twin of the test above, for the whole package (cheap, no imports)."""
    import ast
    names = {p.stem for p in (APP / "pipeline").glob("*.py")}
    bad = []
    for f in sorted((APP / "cytir").rglob("*.py")):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module.split(".")[0]]
            bad += [f"{f.relative_to(APP)}:{node.lineno} {m}" for m in mods if m in names]
    assert bad == []
