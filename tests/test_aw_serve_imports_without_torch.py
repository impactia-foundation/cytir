"""The serve modules import with no torch stack installed (task 2281).

A stranger's serve-only install (server/requirements.txt, a TEI sidecar for query vectors) has no torch.
Every torch / FlagEmbedding / sentence-transformers import in cytir/serve is lazy, inside the function that
embeds in-process or reranks. Blocking those names in a fresh interpreter proves it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_PROBE = """
import importlib, sys
for name in ("torch", "FlagEmbedding", "sentence_transformers", "transformers"):
    sys.modules[name] = None          # `import name` now raises ImportError
for mod in ("cytir.serve.models", "cytir.serve.app", "cytir.serve.mcp_server"):
    importlib.import_module(mod)
print("imported")
"""


def test_serve_modules_import_without_the_torch_stack() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("mcp")
    proc = subprocess.run([sys.executable, "-c", _PROBE], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and "imported" in proc.stdout, proc.stderr[-1500:]
