"""`cyt serve` (slice 8 step 4): the env is derived from the compose aw-api block plus its
overlays, a collection that is not the declared serving profile is refused with the differing
fields named, and a collection made from `served` starts and answers a query over HTTP.

The first half needs no Qdrant. The second half runs where `AW_TEST_QDRANT_URL` names a
throwaway local Qdrant (the `serving-profile` CI job; there `AW_REQUIRE_QDRANT=1` turns a
missing service into a failure). The HTTP test uses a stub TEI endpoint and synthetic vectors:
it proves the serve path (shape check, env, receipt, uvicorn, `/health`, `/search`), not
retrieval quality, and it needs no model download.
"""
from __future__ import annotations
import json
import pytest
from cytir.ops import prod_env, serve  # noqa: E402


def test_a_listed_overlay_that_is_missing_refuses(tmp_path):
    m = tmp_path / "compose-overlays.manifest"
    m.write_text("# x\nnope.yaml\n")
    with pytest.raises(SystemExit, match="nope.yaml"):
        prod_env.overlay_files(m)


def test_credential_shaped_names_are_shown_as_a_digest_not_a_value():
    assert serve._shown("AW_S3_SECRET_KEY", "hunter2").startswith("<redacted sha256:")
    assert "hunter2" not in serve._shown("AW_S3_SECRET_KEY", "hunter2")
    assert serve._shown("AW_BM25_PER_TOKEN", "1") == "1"      # a ranking flag, not a credential


def test_a_tree_without_the_compose_file_serves_from_the_shipped_default_and_says_so(
        monkeypatch, tmp_path, capsys):
    """Was: refuse (the cut had no declared env). Now the cut ships `serving_env.default.json`
    (tests/test_aw_serving_env_default.py): the env is declared, and every value names its file."""
    monkeypatch.setattr(prod_env, "COMPOSE", tmp_path / "docker-compose.yaml")
    assert serve.run(["--collection", "c", "--print-env"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["sources"]["AW_RANKHEAD"] == "serving_env.default.json"
    assert doc["env"]["AW_COLLECTION"] == "c" and doc["env"]["AW_MRL_DIM"] == "256"


def test_in_process_query_embedding_reports_the_pinned_identity(monkeypatch):
    from cytir.core import model_pin
    monkeypatch.setattr(serve.importlib.util, "find_spec", lambda name: object())
    qe = serve.query_embedding({})
    assert qe["mode"] == "in-process"
    info = qe["info"]
    assert info["model_id"] == model_pin.EMBED_MODEL_ID
    assert info["model_sha"] == model_pin.embed_model_revision()
    assert info["device"] in ("cpu", "cuda") and info["dtype"] == "float32"
    assert {"sentence_transformers", "torch"} <= set(info)
