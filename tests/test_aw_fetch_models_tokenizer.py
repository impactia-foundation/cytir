"""`cyt embed models --tokenizer` fetches only the pinned embedder's tokenizer files (task 2289)."""
import sys
import types

from cytir.core import model_pin
from cytir.embed import fetch_models


def test_tokenizer_fetch_uses_the_pin_and_only_small_files(monkeypatch):
    seen = {}

    def fake_snapshot_download(repo, **kw):
        seen.update(repo=repo, **kw)
        return "/cache/snap"

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(snapshot_download=fake_snapshot_download))
    assert fetch_models.main(["--tokenizer"]) == 0
    assert seen["repo"] == model_pin.EMBED_MODEL_ID
    assert seen["revision"] == model_pin.embed_model_revision()
    assert "*.safetensors" not in seen["allow_patterns"]


def test_tokenizer_fetch_failure_is_a_nonzero_exit(monkeypatch):
    def boom(repo, **kw):
        raise OSError("no network")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=boom))
    assert fetch_models.main(["--tokenizer"]) == 1
