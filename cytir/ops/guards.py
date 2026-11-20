"""`cyt ops guards`: every public-cut guard in one pre-push run (task 2390).

Runs the guard-marked test modules (`pytest -m guard`, marker registered in pytest.ini), then the
two `--check` commands the guard modules do not cover (the test roster and the pipeline reach
census). Prints one summary line per step and stops at the first failing step. A lane that adds,
moves or deletes a shipped module runs this before it pushes: lanes run only their targeted tests,
so a guard that lives in another module stays silent until CI.

`--cut` (task 2403) adds the CI "cut rehearsal" for the branch's changed `tests/` modules: it exports
the public tree with `cyt ops cut --out` (the call the CI job makes) and runs those modules inside the
export. Difference from CI: no fresh venv and `pip install .[dev]`; the tests run with the current
interpreter, `PYTHONPATH` pointed at the export and the export as cwd, so what is checked is the
tree's bytes (withheld files are really absent), not the installed dependency set.

Tokenizer (task 2435): the CI job fetches the pinned embedder tokenizer before it tests; the chunker
refuses without it (no word-only fallback). A box without it shows ~19 reds in the family modules
(`service_public_fiche`, `stackexchange_nonso_qa`, `ietf_drafts`, ...) that CI does not show. `--cut` now says so
before the run (`TOKENIZER_HINT`), and repeats it when a module fails.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parents[2]
_TEST_LOCK = "/tmp/aw-pytest.lock"
_LOCK_TIMEOUT_RC = 75
# Task 2436: small runs (guard-marked, or <= 10 named test files) share a counting semaphore of slot
# lock files; suite-wide runs and `-n 4`+ keep the single exclusive _TEST_LOCK. Sizing is measured
# (task 2436 doc): a `-m guard -n 2` run peaks at ~0.7 GB.
_SLOT_PATTERN = "/tmp/aw-pytest.slot-{i}.lock"
_MAX_SLOTS = 4
_SMALL_RUN_PEAK_GB = 0.7
_SLOT_BUDGET = 0.5  # share of MemAvailable the small runs may use together
_SLOT_WAIT_S = 600


def slot_count(available_gb: float | None = None) -> int:
    """N = floor(available_GB * 0.5 / peak_GB), capped at 4, at least 1."""
    if available_gb is None:
        available_gb = 0.0
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    available_gb = int(line.split()[1]) / 1048576
                    break
        except OSError:
            pass
    return max(1, min(_MAX_SLOTS, int(available_gb * _SLOT_BUDGET / _SMALL_RUN_PEAK_GB)))


class _Held:
    """The open lock files of one small run; close() releases the slot and the shared suite-lock hold."""

    def __init__(self, *files):
        self.files = files

    def close(self):
        for fh in self.files:
            fh.close()


def acquire_slot(n: int | None = None, wait_s: float = _SLOT_WAIT_S, pattern: str = _SLOT_PATTERN,
                 poll_s: float = 1.0, suite_lock: str = _TEST_LOCK):
    """Take one of the first n slot files: `flock -n` on each in turn, polling until the deadline.

    A small run also holds the suite lock SHARED, so it never overlaps a suite-wide run (which holds
    it exclusive: `flock /tmp/aw-pytest.lock ...`). Returns a handle (close() releases) or None on timeout.
    """
    n = slot_count() if n is None else n
    deadline = time.monotonic() + wait_s
    suite = open(suite_lock, "a")
    files = [open(pattern.format(i=i), "a") for i in range(n)]
    have_suite = False
    while True:
        if not have_suite:
            try:
                fcntl.flock(suite, fcntl.LOCK_SH | fcntl.LOCK_NB)
                have_suite = True
            except OSError:
                pass
        if have_suite:
            for fh in files:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    continue
                for other in files:
                    if other is not fh:
                        other.close()
                return _Held(suite, fh)
        if time.monotonic() >= deadline:
            for fh in (suite, *files):
                fh.close()
            return None
        time.sleep(poll_s)


def run_in_slot(cmd: list[str], **kwargs):
    """Run a small test command while holding a slot; returns CompletedProcess, or None if no slot came free."""
    fh = acquire_slot()
    if fh is None:
        return None
    try:
        return subprocess.run(cmd, **kwargs)
    finally:
        fh.close()

_STEPS = (
    ("guard tests", [sys.executable, "-m", "pytest", "-m", "guard", "-q", "-n", "2", "-p", "no:cacheprovider"]),
    ("test roster", [sys.executable, "-m", "cytir.ops.test_suite_roster", "--check"]),
    ("entry doors", [sys.executable, "-m", "cytir.ops.entry_doors", "--check"]),
    ("reach census",[sys.executable, "-m", "cytir.ops.pipeline_reach_census", "--check"]),
    ("sealed probe expiry", [sys.executable, "-m", "cytir.consent.expiry_check", "--within-days", "7"]),
)


_EXPORT_PREFIX = "aw-guards-cut-"


def default_export_dir() -> Path:
    """A fresh private directory per run (task 2876): concurrent `--cut` runs must not share one."""
    return Path(tempfile.mkdtemp(prefix=_EXPORT_PREFIX))
_RESULT_RE = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b.*?(tests/[\w./-]+?\.py)")


def changed_test_modules(diff_names: list[str], export_dir: Path) -> tuple[list[str], list[str]]:
    """Split the diff's changed `tests/` modules into (shipped in the export, not shipped)."""
    shipped: list[str] = []
    missing: list[str] = []
    for name in sorted(set(diff_names)):
        if not (name.startswith("tests/") and name.endswith(".py") and Path(name).name.startswith("test_")):
            continue
        (shipped if (export_dir / name).is_file() else missing).append(name)
    return shipped, missing


def module_statuses(report: str, modules: list[str], returncode: int) -> dict[str, str]:
    """Per-module verdict from the runner's `-rA` lines: fail > pass > skipped-all."""
    seen: dict[str, set[str]] = {m: set() for m in modules}
    for line in report.splitlines():
        m = _RESULT_RE.match(line)
        if m and m.group(2) in seen:
            seen[m.group(2)].add("FAILED" if m.group(1) in ("FAILED", "ERROR") else m.group(1))
    out = {}
    for mod, kinds in seen.items():
        if "FAILED" in kinds:
            out[mod] = "fail"
        elif kinds - {"SKIPPED"}:
            out[mod] = "pass"
        elif kinds:
            out[mod] = "skipped-all"
        else:
            out[mod] = "fail" if returncode else "pass"
    return out


def summary_line(module: str, status: str) -> str:
    return f"guards: cut: {module}: {status}"


def _git_diff_names(base: str) -> list[str]:
    proc = subprocess.run(["git", "diff", "--name-only", f"{base}...HEAD", "--", "tests/"],
                          cwd=_APP_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git diff against {base} failed: {proc.stderr.strip()}")
    # git prints repo-root-relative paths; the app is a subdirectory of the monorepo.
    top = subprocess.run(["git", "rev-parse", "--show-prefix"], cwd=_APP_ROOT, capture_output=True, text=True)
    prefix = top.stdout.strip()
    return [n[len(prefix):] if prefix and n.startswith(prefix) else n for n in proc.stdout.split()]


TOKENIZER_HINT = ("guards: cut: WARN: the pinned embedder tokenizer is not cached here, so family tests that chunk "
                  "will be RED in this rehearsal and green in CI (CI fetches it). Fetch it the way CI does: "
                  "`pip install transformers==4.57.1 && python -m cytir.embed.fetch_models --tokenizer` "
                  "(22 MB; use a venv with --system-site-packages if the system Python is locked).")


def _tokenizer_cached() -> bool:
    from cytir.chunk import chunker
    return chunker.tokenizer_status() == "ok"


def _export(export_dir: Path) -> int:
    return subprocess.run([sys.executable, "-m", "cytir.cli", "ops", "cut", "--out", str(export_dir)],
                          cwd=_APP_ROOT).returncode


def _run_tests(export_dir: Path, modules: list[str]) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", "-n", "2", *modules]
    env = dict(os.environ, PYTHONPATH=str(export_dir))
    proc = run_in_slot(cmd, cwd=export_dir, env=env, capture_output=True, text=True)
    if proc is None:
        return _LOCK_TIMEOUT_RC, ""
    return proc.returncode, proc.stdout + proc.stderr


def run_cut(base: str, export_dir: Path, keep: bool, *, diff_names=None, export=None, run_tests=None,
            tokenizer_cached=None) -> int:
    """Export the cut, run the branch's changed test modules inside it. Collaborators are injectable."""
    diff_names = diff_names or _git_diff_names
    export = export or _export
    run_tests = run_tests or _run_tests
    tokenizer_cached = tokenizer_cached or _tokenizer_cached
    names = diff_names(base)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    try:
        t0 = time.monotonic()
        if export(export_dir) != 0:
            print("guards: cut: FAIL: export refused (see output above)")
            return 1
        print(f"guards: cut: exported in {time.monotonic() - t0:.0f}s to {export_dir}")
        shipped, missing = changed_test_modules(names, export_dir)
        for mod in missing:
            print(summary_line(mod, "not shipped"))
        if not shipped:
            print("guards: cut: no changed shipped test modules to run")
            return 0
        has_tokenizer = tokenizer_cached()
        if not has_tokenizer:
            print(TOKENIZER_HINT)
        rc, report = run_tests(export_dir, shipped)
        if rc == _LOCK_TIMEOUT_RC:
            print("guards: cut: FAIL: no test slot free within 600s")
            return rc
        statuses = module_statuses(report, shipped, rc)
        for mod in shipped:
            print(summary_line(mod, statuses[mod]))
        failed = [m for m, st in statuses.items() if st == "fail"]
        if failed or rc:
            print("\n".join(report.strip().splitlines()[-60:]))
            if not has_tokenizer:
                print(TOKENIZER_HINT)
            return rc or 1
        return 0
    finally:
        if not keep:
            shutil.rmtree(export_dir, ignore_errors=True)


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog or "cyt ops guards",
        description="Run every public-cut guard (guard-marked tests, the test roster, the reach census). "
                    "Run before pushing a PR that adds, moves or deletes a shipped module or test.")
    parser.add_argument("--cut", action="store_true",
                        help="after the guard steps, export the public tree once and run the branch's changed "
                             "tests/ modules inside it (the CI cut rehearsal, before the push)")
    parser.add_argument("--base", default="origin/main", help="with --cut: diff base (default origin/main)")
    parser.add_argument("--export-dir", default=None,
                        help="with --cut: export directory (default: a fresh per-run temp dir, removed at the end)")
    parser.add_argument("--keep", action="store_true", help="with --cut: keep the export directory")
    args = parser.parse_args(argv)
    for label, cmd in _STEPS:
        t0 = time.monotonic()
        if label == "guard tests":
            # Memory rule (task 2436): at most N small runs at once (slot files), never next to a suite run.
            # The wait is bounded; 75 is a timeout code, not a test result.
            proc = run_in_slot(cmd, cwd=_APP_ROOT, capture_output=True, text=True)
            if proc is None:
                print("guards: guard tests: FAIL: no test slot free within 600s "
                      f"(all of {_SLOT_PATTERN} are held)")
                return _LOCK_TIMEOUT_RC
        else:
            proc = subprocess.run(cmd, cwd=_APP_ROOT, capture_output=True, text=True)
        secs = time.monotonic() - t0
        out = (proc.stdout + proc.stderr).strip().splitlines()
        last = out[-1] if out else ""
        if proc.returncode == 0:
            print(f"guards: {label}: PASS in {secs:.0f}s  ({last})")
            continue
        print(f"guards: {label}: FAIL rc={proc.returncode} in {secs:.0f}s")
        print("\n".join(out[-60:]))
        return int(proc.returncode or 1)
    print("guards: all steps passed")
    if args.cut:
        export_dir = Path(args.export_dir) if args.export_dir else default_export_dir()
        return run_cut(args.base, export_dir, args.keep)
    return 0
