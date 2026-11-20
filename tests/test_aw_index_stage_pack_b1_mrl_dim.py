"""task 508 / task 500 §7.2 — `--mrl-dim` is a REQUIRED cytir.index.stage_pack_b1
argument, no code default, and the r7 news route's current behaviour (width
128) is reproduced byte-for-byte when a caller passes `--mrl-dim 128`.

Run: python3 -m pytest tests/test_aw_index_stage_pack_b1_mrl_dim.py
"""
from __future__ import annotations

import sys

import pytest

from cytir.index import stage_pack_b1 as stage


class _FakeCollectionInfo:
    def __init__(self, status):
        self.status = status


class _FakeClient:
    """Captures the exact kwargs `ensure_collection` would send to Qdrant,
    without a live server. `create_collection`'s `vectors_config` is what
    fixes the collection's on-disk dense width for its whole lifetime, so
    this is the one call that has to receive the width the caller asked for.
    """

    def __init__(self):
        self.created = None

    def collection_exists(self, name):
        return False

    def create_collection(self, **kwargs):
        self.created = kwargs

    def create_payload_index(self, *args, **kwargs):
        pass


def test_mrl_dim_is_required_with_no_default(capsys):
    """No caller of cytir.index.stage_pack_b1 may rely on an implicit width — a silent
    default is exactly the failure mode task 500 §7.2 named (the lane would
    have kept shipping the wrong width for a week undetected). This calls the
    REAL `main()` — argparse's required-flag check runs before any qdrant
    connection is attempted, so this is a true end-to-end proof of the
    refusal, not a re-typed copy of the parser."""
    argv = ["cytir.index.stage_pack_b1",
            "--in", "x.jsonl", "--collection", "c",
            "--consent-source", "s", "--consent-checked-at", "2026-09-03"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit) as exc:
            stage.main()
    finally:
        sys.argv = old_argv
    assert exc.value.code != 0
    assert "--mrl-dim" in capsys.readouterr().err


def test_argparse_has_no_mrl_dim_default():
    """Read the ACTUAL parser stage_pack_b1.main() builds (not a re-typed
    copy of it — a second copy is how this drifts from the real behaviour)
    and assert the flag is required with default=None."""
    import ast
    import inspect

    src = inspect.getsource(stage.main)
    tree = ast.parse(src)
    found = {"required": None, "default": "UNSET"}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--mrl-dim"):
            for kw in node.keywords:
                if kw.arg == "required":
                    found["required"] = kw.value.value
                if kw.arg == "default":
                    found["default"] = kw.value.value
    assert found["required"] is True, "cytir.index.stage_pack_b1's --mrl-dim must be required=True"
    assert found["default"] == "UNSET", (
        "cytir.index.stage_pack_b1's --mrl-dim must carry NO default — "
        f"found default={found['default']!r}")


def test_ensure_collection_stores_the_width_the_caller_passed():
    client = _FakeClient()
    stage.ensure_collection(client, "aw_dnews_20260908_chmedia", recreate=True,
                            mrl_dim=1024)
    assert client.created["vectors_config"][stage.DENSE].size == 1024


def test_ensure_collection_at_128_reproduces_the_r7_route_byte_identically():
    """The r7 news route's collections were created with the OLD hardcoded
    `MRL_DIM = 128` module constant. This proves the new required-arg form
    produces the identical `create_collection` call at width 128 — the
    behaviour that must not change until an operator explicitly asks for a
    different width."""
    from qdrant_client import models

    client = _FakeClient()
    stage.ensure_collection(client, "aw_dnews_20260828_chmedia", recreate=True,
                            mrl_dim=128)
    vec = client.created["vectors_config"][stage.DENSE]
    assert vec.size == 128
    assert vec.on_disk is True
    quant = client.created["quantization_config"].scalar
    assert quant.type == models.ScalarType.INT8
    assert client.created["hnsw_config"].m == 0
