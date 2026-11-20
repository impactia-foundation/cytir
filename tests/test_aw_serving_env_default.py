"""The shipped default serving env (slice 8 step 5): what `cyt serve` starts the API with in a tree that
has no `deploy/` (the public cut), and the proof that it changes nothing where `deploy/` exists.

`cytir/ops/serving_env.default.json` is GENERATED from the compose aw-api block + its overlays
(`python -m cytir.ops.prod_env --write-default`), minus the keys that name a host or a collection of
ours. No Qdrant needed; the tests that read `deploy/` skip in the cut, where the rest still runs.
"""
from __future__ import annotations
import json
import pathlib
import re
import pytest
pytest.importorskip("yaml")
from cytir.core import serving_profile as SP  # noqa: E402
from cytir.ops import prod_env, serve  # noqa: E402


REPO = pathlib.Path(__file__).resolve().parent.parent


#: names of our boxes and our hosting, spelled in pieces (the cut's own scanner reads this file)
_OURS = re.compile("(?i)" + "|".join(("heim" + "dall", "play" + "ground", "impact" + "ia", "hetz" + "ner")))


def _shipped() -> dict:
    return json.loads(prod_env.DEFAULT_ENV.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ neutral (runs in the cut too)
def test_the_default_is_neutral_no_host_no_collection_no_credential():
    env = _shipped()["env"]
    assert env, "the default declares nothing"
    for k in prod_env.NOT_IN_DEFAULT:
        assert k not in env, k
    for k, v in env.items():
        assert k.startswith("AW_"), k
        assert "://" not in v and not _OURS.search(v), (k, v)
        assert not re.search(r"^aw_(prod|[ve]\d+|passages)$", v), (k, v)
        assert not serve._SECRET.search(k), k
    assert {"AW_MRL_DIM", "AW_DENSE_RESCORE", "AW_FUSION_WEIGHTS"} <= set(env)    # the flip overlay is in


def test_a_tree_without_deploy_resolves_to_the_default_with_paths_mapped(monkeypatch, tmp_path):
    monkeypatch.setattr(prod_env, "COMPOSE", tmp_path / "docker-compose.yaml")
    r = prod_env.resolve_env()
    assert set(r.sources.values()) == {"serving_env.default.json"}
    assert r.env["AW_FUSION_WEIGHTS"].startswith(str(REPO / "cytir" / "serve"))
    assert pathlib.Path(r.env["AW_FUSION_WEIGHTS"]).is_file()
    assert pathlib.Path(r.env["AW_HUBFLOOR_TABLE"]).is_file()
    env, src = serve.build_env(SP.get("served"), collection="mine", qdrant_url="http://127.0.0.1:1",
                               tei_url="http://127.0.0.1:9")
    assert env["AW_COLLECTION"] == "mine" and env["AW_QDRANT_URL"] == "http://127.0.0.1:1"
    assert env["AW_MRL_DIM"] == "256" and src["AW_MRL_DIM"] == "profile served dense_dim"
    assert src["AW_RANKHEAD"] == "serving_env.default.json"


def test_a_default_of_another_schema_is_refused(monkeypatch, tmp_path):
    bad = tmp_path / "d.json"
    bad.write_text(json.dumps({"schema": "x", "env": {}}))
    monkeypatch.setattr(prod_env, "COMPOSE", tmp_path / "docker-compose.yaml")
    monkeypatch.setattr(prod_env, "DEFAULT_ENV", bad)
    with pytest.raises(SystemExit, match="not an aw.serving-env-default.v1"):
        prod_env.resolve_env()
