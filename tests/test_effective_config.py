"""Task 522 — `/health` must echo the knobs a flip actually set, not just say
"ok" (deployment-doctrine "No Invisible State"). `retrieval.effective_config()`
and `retrieval.weights_fingerprint()` are the source `app.py`'s `/health`
route reads; these pin their contract without needing `fastapi` installed
(`retrieval.py` itself has no fastapi dependency), and without a live qdrant.

Each probe runs `aw_server.retrieval` in a FRESH interpreter under a chosen
env, mirroring `test_task519_dense_quant_levers.py`'s pattern: the module
reads its env at import time, so a same-process monkeypatch would test a
value the server never actually uses.

Run: python3 -m pytest tests/test_effective_config.py -q
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

_APP = pathlib.Path(__file__).resolve().parent.parent
_PROBE = (
    "from cytir.serve import retrieval as r; import json; "
    "print(json.dumps(r.effective_config()))"
)


def _interpreter_can_import_the_server():
    return subprocess.run(
        [sys.executable, "-c", "import qdrant_client"],
        capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(
    not _interpreter_can_import_the_server(),
    reason=f"{sys.executable} cannot import qdrant_client — run these under "
           "an interpreter that has the serving deps")


def _effective_config(**env_overrides):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("AW_DENSE_", "AW_MRL_DIM", "AW_COLLECTION",
                                "AW_FUSION_WEIGHTS", "AW_RANKHEAD"))}
    env["PYTHONPATH"] = str(_APP / "server")
    env.update({k: str(v) for k, v in env_overrides.items()})
    out = subprocess.run([sys.executable, "-c", _PROBE], env=env,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_default_config_is_byte_identical_to_todays_shipped_behaviour():
    """Unset env => exactly what production runs today (128-int8, rescore
    off, the incumbent weights file)."""
    cfg = _effective_config()
    assert cfg["mrl_dim"] == 128
    assert cfg["dense_rescore"] is False
    assert cfg["dense_oversampling"] == 1.0
    assert cfg["dense_ignore_quant"] is False
    assert cfg["fusion_weights"]["path"].endswith("fusion_weights.json")
    assert cfg["fusion_weights"]["sha256"] is not None


def test_flip_env_is_visible_end_to_end_through_effective_config():
    """The exact env `deploy/docker-compose.flip.yaml` carries, echoed back."""
    cfg = _effective_config(
        AW_MRL_DIM="256", AW_DENSE_RESCORE="1", AW_DENSE_OVERSAMPLING="3.0",
        AW_DENSE_IGNORE_QUANT="0",
        AW_FUSION_WEIGHTS=str(_APP / "cytir" / "serve" /
                             "fusion_weights.candidate519.json"))
    assert cfg["mrl_dim"] == 256
    assert cfg["dense_rescore"] is True
    assert cfg["dense_oversampling"] == 3.0
    assert cfg["dense_ignore_quant"] is False
    assert cfg["fusion_weights"]["path"].endswith("fusion_weights.candidate519.json")


def test_weights_fingerprint_changes_with_the_file_not_the_process():
    """A weights promotion is visible without a code change or a restart of
    THIS test — reading the path is enough; the sha256 must differ between
    the incumbent and the candidate file."""
    incumbent = _effective_config()["fusion_weights"]["sha256"]
    candidate = _effective_config(
        AW_FUSION_WEIGHTS=str(_APP / "cytir" / "serve" /
                             "fusion_weights.candidate519.json")
    )["fusion_weights"]["sha256"]
    assert incumbent != candidate


def test_missing_weights_file_reports_sha256_none_rather_than_crashing():
    cfg = _effective_config(AW_FUSION_WEIGHTS="/no/such/file.json")
    assert cfg["fusion_weights"]["sha256"] is None
    assert cfg["fusion_weights"]["path"] == "/no/such/file.json"
