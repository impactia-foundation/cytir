"""The CI runners' model cache (task 1718): AW_FALLBACK_MODELS_DIR is a lookup-only
location, so a seeded dir satisfies fetch_models with no network; a wrong file fails loudly."""
import hashlib

import pytest

from cytir.core import paths
from cytir.embed import fetch_models, model_assets


def _asset(payload: bytes):
    return model_assets.ModelAsset(
        "t", "t.bin", "http://invalid.invalid/t.bin", hashlib.sha256(payload).hexdigest(),
        len(payload), "AW_T_BIN", "test")


@pytest.fixture
def cache(monkeypatch, tmp_path):
    for k in ("AW_T_BIN", "AW_MODELS_DIR"):
        monkeypatch.delenv(k, raising=False)
    d = tmp_path / "cache"
    d.mkdir()
    monkeypatch.setenv("AW_FALLBACK_MODELS_DIR", str(d))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(model_assets, "REPO_ROOT", tmp_path / "no-repo")
    paths.reset_settings_cache()
    monkeypatch.setattr(fetch_models, "_download",
                        lambda a, dest: pytest.fail("network download attempted"))
    yield d
    paths.reset_settings_cache()


def test_cached_correct_file_is_used_without_network(cache):
    (cache / "t.bin").write_bytes(b"payload")
    assert fetch_models.handle(_asset(b"payload"), check_only=False, force=False) is True


def test_cached_wrong_file_is_refused_not_used(cache):
    (cache / "t.bin").write_bytes(b"truncated")
    assert fetch_models.handle(_asset(b"payload"), check_only=False, force=False) is False
