"""Chunker global state must not leak between test files (task 2317).

The delta-adapter stage tests run the real stage, which selects a chunk policy and a tokenizer revision on the
process-global chunker. tests/test_chunk.py then failed 13 tests when it ran after them in one process. The root
conftest restores that state after every test; this guard runs both files in both orders, one process per order.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.repo_files import repo_file_or_skip

ROOT = Path(__file__).resolve().parent.parent
ADAPTERS = "tests/test_docs_delta_adapters.py"
CHUNK = "tests/test_chunk.py"


@pytest.mark.parametrize("order", [(ADAPTERS, CHUNK), (CHUNK, ADAPTERS)])
def test_chunk_and_delta_adapter_tests_pass_in_either_order(order):
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", "-p", "no:cacheprovider", *order],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout[-2000:]
