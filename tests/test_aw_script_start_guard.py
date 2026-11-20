"""Every `cytir/**/*.py` that a caller starts BY PATH must start from a clean cwd (task 683 S7b).

Why: `python path/to/aw/sources/families/news.py` puts that file's own directory on `sys.path[0]`,
not the repo root, so `import cytir...` raises ModuleNotFoundError unless the file inserts the repo
root itself. The moves into the package (#437) broke news.py this way while every test, which
imports the module, stayed green. Timers, chains and systemd units start these files by path.

The set of files is DERIVED, never typed (corpus-doctrine #8): every `cytir/<...>.py` named by a
caller outside `tasks/` and outside docs/data (deploy/, pipeline/ scripts and shell, server/,
scripts/, CI workflows, and other `cytir/` modules that spawn one by path). Each is (a) loaded as a script from a clean cwd with no
PYTHONPATH and with `sys.path[0]` set to the file's directory, exactly as `python path.py` does,
but under a non-main run name so nothing executes; and (b) when it parses arguments, run with
`--help`. A missing *repo-local* module fails; a missing third-party package is not this test's
business.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

import pytest

from tests.helpers.repo_files import repo_file_or_skip

REPO = pathlib.Path(__file__).resolve().parents[1]
AW = REPO / "cytir"
#: a file that is known broken and has its own fix in flight: xfail (non-strict). Empty now: news.py's
#: fix is on main (both cases xpassed, task 1752).
KNOWN_BROKEN: dict[str, str] = {}
_CALLER_DIRS = ("deploy", "pipeline", "server", "scripts")
_CALLER_SUFFIXES = {".sh", ".service", ".timer", ".py", ".yaml", ".yml", ""}
_PATH_RX = re.compile(r"(?<![\w.])cytir/((?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py)\b")
_MAIN_RX = re.compile(r"""if\s+__name__\s*==\s*['"]__main__['"]""")


def _caller_files():
    roots = [REPO / d for d in _CALLER_DIRS] + [REPO.parent / ".github", REPO / "cytir"]
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix not in _CALLER_SUFFIXES or "__pycache__" in p.parts:
                continue
            if p.name.startswith("test_") or "tasks" in p.parts or "archive" in p.parts:
                continue
            yield p


_PATH = r"""["']?(?:[\w.$/{}"'-]*/)?cytir/((?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py)\b"""
#: a command line that STARTS the file: an interpreter token (python, $PY, $QPY, "$VENV/bin/python")
#: then optional flags, then the path
_CMD_RX = re.compile(
    r"""(?:python3?(?:\.\d+)?|\$\{?[A-Z_]*PY\}?|"\$\{?VENV\}?[^"]*"|\$\{?VENV\}?\S*)["']?,?\s+(?:-[A-Za-z]\S*\s+)*"""
    + _PATH)
_EXEC_RX = re.compile(r"^\s*ExecStart=.*?" + _PATH)
#: a quoted `cytir/x.py` literal in a python file: a start only when the line (or the two before it)
#: builds a command: subprocess/Popen/run(/args/argv/python
_LIT_RX = re.compile(r"""["'](?:[^"']*/)?cytir/((?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py)["']""")
_SPAWN_RX = re.compile(r"subprocess|Popen|check_call|check_output|\.run\(|argv|args\b|python|sys\.executable|\bcmd\b")


_PARTS_RX = re.compile(r"""^\s*(?:runner|script)\s*=.*?["']aw["']\s*/\s*["']([A-Za-z0-9_]+)["']\s*/\s*["']([A-Za-z0-9_]+\.py)["']""", re.M)


def _starts_in(p: pathlib.Path, text: str):
    """Yield the `cytir/...py` names that a REAL line of `p` starts (prose mentions are dropped)."""
    lines = re.sub(r"\\\n\s*", " ", text).splitlines()   # join `\` continuations (units, shell)
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        for rx in (_CMD_RX, _EXEC_RX):
            for m in rx.finditer(line):
                yield m.group(1)
        if p.suffix == ".py":
            ctx = "\n".join(lines[max(0, i - 2):i + 1])
            if _SPAWN_RX.search(ctx):
                for m in _LIT_RX.finditer(line):
                    yield m.group(1)
    if p.suffix == ".py" and "subprocess" in text:   # `release / "cytir" / "gates" / "x.py"` handed to a child
        for m in _PARTS_RX.finditer(text):
            yield "/".join(m.groups()[:-1]) + "/" + m.group(m.lastindex)
    if p.name == "probe-entrypoints.json":
        import json
        for e in json.loads(text).get("entry_points", []):
            for r in e.get("runs", []):
                m = re.fullmatch(r"(?:.*/)?cytir/(.+\.py)", r)
                if m:
                    yield m.group(1)


def discover() -> dict[str, list[str]]:
    """{`cytir/...py` -> why it is in the set}: only files a caller REALLY starts by path."""
    found: dict[str, list[str]] = {}
    files = list(_caller_files())
    files.append(REPO / "deploy/operator/probe-entrypoints.json")
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for name in _starts_in(p, text):
            target = AW / name
            if target.exists() and target != p:
                found.setdefault(f"cytir/{name}", []).append(f"started by {p.relative_to(REPO.parent)}")
    return dict(sorted(found.items()))


FILES = discover()

_LOADER = (
    "import os, runpy, sys\n"
    "p = os.path.abspath(sys.argv[1])\n"
    "sys.path[0] = os.path.dirname(p)   # what `python p` does\n"
    "sys.argv = [p]\n"
    "runpy.run_path(p, run_name='aw_script_start_guard')\n"
)


def _clean_env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # task 1728: CI runs `pip install -e .`, which makes `aw` importable in the child; a bare checkout
    # has no install, so the child gets the repo root on PYTHONPATH (same effect, no install needed).
    env["PYTHONPATH"] = str(REPO)
    return env


def _repo_local_missing(stderr: str) -> str | None:
    m = re.search(r"No module named '([^']+)'", stderr)
    if not m:
        return None
    top = m.group(1).split(".")[0]
    if top == "cytir" or (REPO / "pipeline" / f"{top}.py").exists() or (REPO / top).is_dir() \
            or (REPO / "pipeline" / top).is_dir():
        return m.group(1)
    return None


def _run(args, tmp_path, timeout=90):
    return subprocess.run([sys.executable, *args], cwd=tmp_path, env=_clean_env(),
                          capture_output=True, text=True, timeout=timeout)


def _params():
    for rel in FILES:
        marks = []
        if rel in KNOWN_BROKEN:
            marks.append(pytest.mark.xfail(reason=KNOWN_BROKEN[rel], strict=False))
        yield pytest.param(rel, id=rel, marks=marks)


def test_discovery_finds_the_files_production_starts_by_path():
    """The derivation is not empty and includes the known by-path starters (a witness that the
    regex still sees systemd units, shell drivers and python subprocess calls)."""
    repo_file_or_skip("deploy/operator/probe-entrypoints.json")   # the callers live in withheld trees
    assert len(FILES) >= 6, FILES
    # cytir/index/apply.py is module-form since #1402 (task 1751), so it is no longer a by-path start.
    # cytir/ops/prod_env_run.py is no longer a witness: retired with aw-test-e8/aw-test-twin (task 2328, verified no live starter 2026-11-04).
    for witness in ("cytir/ops/german_parts_tripwire.py",
                    "cytir/index/compose.py"):
        assert witness in FILES, (witness, sorted(FILES))


@pytest.mark.parametrize("rel", list(_params()))
def test_started_as_a_script_from_a_clean_cwd_resolves_its_imports(rel, tmp_path):
    r = _run(["-c", _LOADER, str(REPO / rel)], tmp_path)
    missing = _repo_local_missing(r.stderr)
    assert missing is None, (
        f"{rel} ({'; '.join(FILES[rel][:2])}) fails as a script: ModuleNotFoundError "
        f"{missing!r}. Insert the repo root into sys.path before the first `aw` import, "
        f"as cytir/index/compose.py does.\n{r.stderr[-600:]}")


def _parses_args(rel: str) -> bool:
    text = (REPO / rel).read_text(encoding="utf-8")
    return bool(_MAIN_RX.search(text)) and "argparse" in text


@pytest.mark.parametrize("rel", [pytest.param(p.values[0], id=p.values[0], marks=p.marks)
                                 for p in _params() if _parses_args(p.values[0])])
def test_help_runs_when_started_by_path(rel, tmp_path):
    r = _run([str(REPO / rel), "--help"], tmp_path)
    missing = _repo_local_missing(r.stderr)
    assert missing is None, f"{rel} --help: ModuleNotFoundError {missing!r}\n{r.stderr[-600:]}"


@pytest.mark.parametrize("rel", FILES)
def test_bootstrap_under_S_without_pythonpath(rel, tmp_path):
    env = {k: v for k, v in _clean_env().items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-S", "-c", _LOADER, str(REPO / rel)], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=90)
    missing = _repo_local_missing(r.stderr)
    assert missing is None, f"{rel}: ModuleNotFoundError {missing!r} under -S, no PYTHONPATH\n{r.stderr[-500:]}"
