"""Task 3022: `cytir/gates/run_corpus_arm.py` is started BY FILE PATH (`python <release>/<pkg>/gates/run_corpus_arm.py`,
as `content_pack_arm` and `news_daily` do), never as `-m`. This spawns it that way, as a stranger's checkout would.

Needs no Qdrant, GPU or network: the handler is run only up to argument parsing (`--help`, missing arguments),
which happens before it reads any manifest, cache or index."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HANDLER = Path(__file__).resolve().parents[1] / "cytir" / "gates" / "run_corpus_arm.py"


def _spawn(tmp_path, *args):
    return subprocess.run([sys.executable, str(HANDLER), *args], capture_output=True, text=True,
                          cwd=tmp_path, timeout=120)


def test_handler_file_exists_where_the_callers_look():
    assert HANDLER.is_file()


def test_spawned_by_path_it_imports_and_prints_its_arguments(tmp_path):
    proc = _spawn(tmp_path, "--help")
    assert proc.returncode == 0, proc.stderr[-500:]
    for flag in ("--cache", "--qvec-query-source-manifest", "--target-manifest", "--corpus-manifest",
                 "--out-dir", "--arm-id", "--capture-sparse"):
        assert flag in proc.stdout


def test_spawned_by_path_without_arguments_refuses_with_the_usage_exit_code(tmp_path):
    proc = _spawn(tmp_path)
    assert proc.returncode == 2
    assert "the following arguments are required" in proc.stderr
    assert not list(tmp_path.iterdir())          # it wrote nothing
