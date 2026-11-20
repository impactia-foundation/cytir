"""Task 432 — the embedder pin is a mirror contract, and it is enforced here.

Task 429 blocker E: `load_embedder()` took no revision argument, so document
vectors came from whatever HuggingFace served on the day. Task 432 found the
read side worse still — TEI was launched without --revision and reported
`"model_sha": null`, so serving could not name the revision it answered with.

These tests pin: the shape of the pin, that it equals what production actually
runs, that the read side (server/) imports the SAME pin object rather than a
copy, and that a call site which forgets to pin fails loudly.

Task 646 S2: `model_pin.py` moved to `cytir/core/model_pin.py`; the pipeline shim is gone (task 1177), and `cytir/serve/models.py`
imports `cytir.core.model_pin` directly (the aw-api Docker image installs the
`aw` package) instead of carrying a hand-maintained byte-identical copy --
`test_server_copy_is_byte_identical` became
`test_server_imports_the_same_pin_object`.
"""
from __future__ import annotations

import os
import pathlib
import re
import sys
import types

import pytest

from tests.helpers.repo_files import repo_file_or_skip  # deploy/ compose only

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent

from cytir.core import model_pin  # noqa: E402


def test_pin_is_a_commit_sha_not_a_branch():
    """`main` moves; pinning to it pins nothing."""
    assert re.fullmatch(r"[0-9a-f]{40}", model_pin.EMBED_MODEL_REVISION)
    assert model_pin.EMBED_MODEL_ID == "Snowflake/snowflake-arctic-embed-l-v2.0"


def test_pin_matches_what_production_actually_runs():
    """The value is RECORDED from production, not chosen here. If someone edits
    the pin without re-pinning the embed path, this catches it.

    Task 657 moved the task-413a embed_shards script to
    `cytir/embed/shards.py` (a shim now sits at the old path -- task 643
    §6.2). The moved copy no longer restates the revision as a second
    literal default: it imports `embed_model_revision()` from
    `cytir.core.model_pin` directly, closing the exact fork this test exists
    to catch. Proving that import beats proving a literal string, because a
    literal would just be a second place this test has to also check."""
    shards = (REPO / "cytir" / "embed" / "shards.py").read_text()
    assert "from cytir.core.model_pin import" in shards
    assert 'default=EMBED_MODEL_ID' in shards
    assert 'default=embed_model_revision()' in shards
    assert f'default="{model_pin.EMBED_MODEL_REVISION}"' not in shards, (
        "the revision must be IMPORTED from cytir.core.model_pin, not "
        "restated as a second literal default -- that is the fork task 432 "
        "closed on the write side and task 657 closed here")


def test_serving_compose_pins_the_same_revision():
    """The read side of the contract, as configuration."""
    compose = repo_file_or_skip("deploy/docker-compose.yaml").read_text()
    assert '"--revision", "%s"' % model_pin.EMBED_MODEL_REVISION in compose


def test_pipeline_shim_is_gone():
    """Task 1177: the old flat `model_pin.py` shim is gone. The pin has ONE
    home, `cytir/core/model_pin.py`, and exports every name callers use.
    Lessons-ledger L170 points here (review 1179 renamed both ends)."""
    assert not (REPO / "pipeline" / "model_pin.py").exists()
    for name in ("EMBED_MODEL_ID", "EMBED_MODEL_REVISION", "REVISION_ENV",
                 "assert_pin", "embed_model_revision", "is_pinned_default",
                 "pin_note"):
        assert hasattr(model_pin, name), name


def test_server_imports_the_same_pin_object():
    """The read side (cytir/serve/models.py) imports `cytir.core.model_pin`
    directly -- the aw-api Docker image installs the `aw` package (task 646
    S2) -- so there is no second copy left to keep byte-identical. This
    replaces the old `test_server_copy_is_byte_identical`."""
    server_dir = REPO / "server"
    sys.path.insert(0, str(server_dir))
    import cytir.serve.models as models  # noqa: PLC0415
    assert models.EMBED_MODEL_ID is model_pin.EMBED_MODEL_ID
    assert models.EMBED_REVISION == model_pin.embed_model_revision()
    assert not (REPO / "cytir" / "serve" / "model_pin.py").exists(), (
        "cytir/serve/model_pin.py should be gone -- models.py imports "
        "cytir.core.model_pin directly now, not a hand-copy")


def test_override_is_honoured_but_never_silent(monkeypatch):
    other = "b" * 40
    monkeypatch.setenv(model_pin.REVISION_ENV, other)
    assert model_pin.embed_model_revision() == other
    assert model_pin.is_pinned_default() is False
    note = model_pin.pin_note()
    assert "OVERRIDE" in note and model_pin.EMBED_MODEL_REVISION in note


def test_override_rejects_a_branch_name(monkeypatch):
    monkeypatch.setenv(model_pin.REVISION_ENV, "main")
    with pytest.raises(ValueError, match="not a 40-hex commit sha"):
        model_pin.embed_model_revision()


def test_assert_pin_rejects_an_unpinned_call_site():
    with pytest.raises(ValueError, match="no revision pinned"):
        model_pin.assert_pin(model_pin.EMBED_MODEL_ID, None, "somewhere")


def test_assert_pin_rejects_a_different_revision():
    with pytest.raises(ValueError, match="!= pinned"):
        model_pin.assert_pin(model_pin.EMBED_MODEL_ID, "c" * 40, "somewhere")


# --- the plumbing: load_embedder must actually pass the revision -------------

def _stub_sentence_transformers(monkeypatch, captured):
    class _ST:
        def __init__(self, model, **kw):
            captured["model"] = model
            captured["kwargs"] = kw
            self.prompts = {}
            self.max_seq_length = 0

        def encode(self, texts, **kw):
            return [[0.0] * 4 for _ in texts]

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _ST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    torch = types.ModuleType("torch")
    torch.float16, torch.float32 = "fp16", "fp32"
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_load_embedder_passes_the_pinned_revision(monkeypatch):
    captured: dict = {}
    _stub_sentence_transformers(monkeypatch, captured)
    from cytir.embed import bakeoff
    _d, _q, spec = bakeoff.load_embedder("arctic-l-v2", use_fp16=False)
    assert captured["kwargs"]["revision"] == model_pin.EMBED_MODEL_REVISION
    assert spec["revision"] == model_pin.EMBED_MODEL_REVISION


def test_load_embedder_records_an_explicit_revision(monkeypatch):
    captured: dict = {}
    _stub_sentence_transformers(monkeypatch, captured)
    from cytir.embed import bakeoff
    _d, _q, spec = bakeoff.load_embedder("arctic-l-v2", use_fp16=False,
                                               revision="d" * 40)
    assert captured["kwargs"]["revision"] == "d" * 40
    assert spec["revision"] == "d" * 40


def test_local_model_path_does_not_get_a_hub_revision(monkeypatch, tmp_path):
    """Sealed offline workers name an immutable local snapshot; a local directory
    has no hub revision to resolve, and passing one would break the load."""
    captured: dict = {}
    _stub_sentence_transformers(monkeypatch, captured)
    from cytir.embed import bakeoff
    _d, _q, spec = bakeoff.load_embedder("arctic-l-v2", use_fp16=False,
                                               model_path=str(tmp_path))
    assert "revision" not in captured["kwargs"]
    assert captured["model"] == str(tmp_path)
    assert spec["revision"] == model_pin.EMBED_MODEL_REVISION   # still recorded


def test_bakeoff_only_embedders_are_not_falsely_pinned(monkeypatch):
    """qwen3/bge were never used to build a shipped index; claiming a pin for
    them would assert a reproducibility nobody established."""
    captured: dict = {}
    _stub_sentence_transformers(monkeypatch, captured)
    from cytir.embed import bakeoff
    _d, _q, spec = bakeoff.load_embedder("qwen3-0.6b", use_fp16=False)
    assert "revision" not in captured["kwargs"]
    assert "revision" not in spec
