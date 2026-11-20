"""A stranger's cache holds `snapshots/<pin>/` and no `refs/main` (task 2520).

`cyt embed models --tokenizer` runs `snapshot_download(revision=<pin>)`, which lays
out `snapshots/<sha>/` only. The chunker must resolve the tokenizer at that pin,
not at the hub's mutable "main", or every offline run reports it unavailable.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cytir.core import model_pin

_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "sentencepiece.bpe.model", "config.json")


def _repo_dir(root: Path) -> Path:
    return root / "hub" / ("models--" + model_pin.EMBED_MODEL_ID.replace("/", "--"))


def _source_snapshot() -> Path | None:
    home = Path(os.environ.get("AW_TEST_TOKENIZER_SOURCE_HF_HOME")
                or Path.home() / ".cache" / "huggingface")
    snap = _repo_dir(home) / "snapshots" / model_pin.EMBED_MODEL_REVISION
    return snap if (snap / "tokenizer.json").exists() else None


def test_pinned_tokenizer_resolves_from_fresh_offline_cache(tmp_path):
    src = _source_snapshot()
    if src is None:
        pytest.skip("no local copy of the pinned tokenizer to lay out a cache from")
    repo = _repo_dir(tmp_path)
    dst = repo / "snapshots" / model_pin.EMBED_MODEL_REVISION
    dst.mkdir(parents=True)
    for name in _FILES:
        if (src / name).exists():
            shutil.copyfile(src / name, dst / name)   # dereference the blob symlink
    assert not (repo / "refs").exists()               # the stranger's layout

    env = {k: v for k, v in os.environ.items()
           if k not in ("AW_CHUNK_TOKENIZER_REVISION", "AW_MODEL_REVISION")}
    env.update(HF_HOME=str(tmp_path), HF_HUB_CACHE=str(tmp_path / "hub"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    code = ("from cytir.chunk import chunker as C;"
            "print(C.tokenizer_status(), C.effective_tokenizer_revision())")
    out = subprocess.run([sys.executable, "-c", code], env=env, text=True,
                         capture_output=True, timeout=300)
    assert out.stdout.split() == ["ok", model_pin.EMBED_MODEL_REVISION], out.stderr[-600:]
