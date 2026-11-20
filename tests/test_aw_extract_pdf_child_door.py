"""The PDF extractor starts ITSELF as a bounded child process.

`_run_liteparse` and `_run_docling` run `[sys.executable, <this file>, "--liteparse-one"|"--docling-one", ...]`.
The `main()` block of the module is the only handler of those flags. Without it the child exits 0,
writes nothing, and the parent reports "failed" for every PDF. This test pins that door.
"""
from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys

import pytest

from cytir.extract import pdf as pdf_mod


@pytest.mark.parametrize("flag", ["--liteparse-one", "--docling-one"])
def test_child_flag_is_handled_and_reports_failure_for_a_missing_input(tmp_path, flag):
    out = tmp_path / "result.json"
    proc = subprocess.run(
        [sys.executable, str(pdf_mod._THIS_FILE), flag, str(tmp_path / "missing.pdf"), str(out)],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 1, (proc.returncode, proc.stderr[-400:])
    record = json.loads(out.read_text())
    assert record["status"] == "failed"
    assert record["error"]


def test_every_flag_the_module_passes_to_its_own_child_is_handled_in_main():
    source = inspect.getsource(pdf_mod)
    passed = set(re.findall(r'str\(_THIS_FILE\), "(--[a-z]+-one)"', source))
    assert passed == {"--liteparse-one", "--docling-one"}
    main_source = inspect.getsource(pdf_mod.main)
    for flag in passed:
        assert flag in main_source, flag
