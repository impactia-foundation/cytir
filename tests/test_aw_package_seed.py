# Task 646 S1: the `aw` package + CLI seed (task 643 §6.1/§6.2 S1). Lives
# in tests/ with the rest of the package's tests (task 696 slice 2).
import subprocess
import sys

import pytest

import cytir
import cytir.cli


def test_package_has_a_version_string():
    assert isinstance(cytir.__version__, str)
    assert cytir.__version__


def test_cli_parser_accepts_version_flag():
    parser = cytir.cli.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--version"])
    assert excinfo.value.code == 0


def test_cli_main_with_no_args_returns_zero():
    assert cytir.cli.main([]) == 0


def test_console_script_prints_version():
    # Exercises the installed `cyt` console script end to end (not just the
    # importable module) -- this is what task 643 §6.2 S1's acceptance line
    # ("`pip install` in a clean venv") actually proves.
    result = subprocess.run([sys.executable, "-m", "cytir.cli", "--version"],
                             capture_output=True, text=True)
    assert result.returncode == 0
    lines = result.stdout.strip().splitlines()
    assert lines[0] == f"cyt {cytir.__version__}"
    assert lines[1].startswith("# data dir: ")      # task 733: says where data/ is read from
