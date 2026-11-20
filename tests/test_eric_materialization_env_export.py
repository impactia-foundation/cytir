"""[test] task 663, 2026-09-25: every eric-materialization script that
sources `current.env` must EXPORT what it sources.

The real incident: `chain_and_smoke_test.sh` sourced `current.env` (plain
`VAR=value` lines, no `export`) and then ran
`deploy/<box>/eric-materialization/chain_eric_pdf_partition_admission.sh` as a CHILD `bash` process.
A plain `source` only makes variables visible in the CURRENT shell; the
child process inherited none of them. Tranche 010000's real finalize hit
this path live on 2026-09-25 and failed immediately with
"AW_ARTIFACT_ROOT: set global /aw-corpus root" -- current.env was read,
never exported. `set -a` / `set +a` around the `source` line is the fix:
everything assigned while sourcing is auto-exported.
"""
from __future__ import annotations

import pathlib
import re

from tests.helpers.repo_files import repo_glob_or_skip

HERE = pathlib.Path(__file__).resolve().parent
# the operator tree under `deploy/` is withheld from the public cut: the test skips there
ERIC_DIR = repo_glob_or_skip("deploy/*/eric-materialization/chain_all_tranches.sh")[0].parent

SOURCE_LINE = re.compile(
    r'source\s+"\$\{AW_ERIC_ENV_FILE:-\$\{?HERE\}?/current\.env\}"')


def _scripts_that_source_current_env() -> list[pathlib.Path]:
    out = []
    for path in sorted(ERIC_DIR.glob("*.sh")):
        if SOURCE_LINE.search(path.read_text()):
            out.append(path)
    return out


def test_at_least_the_four_known_scripts_source_current_env():
    """A canary for the regex itself: if this drops to 0, the pattern
    above stopped matching real scripts and every test below is
    vacuously passing."""
    names = {p.name for p in _scripts_that_source_current_env()}
    assert names == {"chain_and_smoke_test.sh", "embed_compose_enqueue.sh",
                     "chain_all_tranches.sh", "run_all_tranches.sh"}


def test_every_script_that_sources_current_env_exports_it():
    for path in _scripts_that_source_current_env():
        text = path.read_text()
        lines = text.splitlines()
        source_idx = next(i for i, line in enumerate(lines)
                          if SOURCE_LINE.search(line))
        before = lines[max(0, source_idx - 3):source_idx]
        after = lines[source_idx + 1:source_idx + 3]
        assert any(line.strip() == "set -a" for line in before), (
            f"{path.name}: `source .../current.env` at line "
            f"{source_idx + 1} has no `set -a` immediately before it -- "
            "every var it sources stays invisible to any child process "
            "this script spawns (the real 2026-09-25 tranche-010000 "
            "incident)")
        assert any(line.strip() == "set +a" for line in after), (
            f"{path.name}: `set -a` before the source line was found, but "
            "no matching `set +a` right after it -- every subsequent "
            "variable assignment in this script would ALSO leak into the "
            "environment of every child process it spawns")


def test_scripts_are_syntactically_valid_bash():
    """A cheap guard that the set -a/set +a edit did not break the script
    -- `bash -n` parses without executing."""
    import subprocess
    for path in _scripts_that_source_current_env():
        result = subprocess.run(["bash", "-n", str(path)],
                                capture_output=True, text=True)
        assert result.returncode == 0, f"{path.name}: {result.stderr}"
