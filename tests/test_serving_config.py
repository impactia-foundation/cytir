"""Tests for the serving-config registry and its mirror contracts.

The point of these is not coverage. It is that `serving_config.py` claims to be
the ONE home for the serving shape, and a claim like that decays the moment
someone re-declares a width or a quantizer somewhere else. These assert the
claim is still true.

    python3 -m pytest tests/test_serving_config.py -q
"""
from __future__ import annotations

import os
import pathlib
import re
import sys

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent / "server")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cytir.core import serving_config as sc  # noqa: E402


# ---------------------------------------------------------------------------
# the measured byte table (task 412 SS10.2)
# ---------------------------------------------------------------------------
# These are MEASURED file sizes from 20,000-point collections on Qdrant v1.18.3,
# not nominal arithmetic. They are asserted here because the whole flip-config
# argument is "quality per resident byte", and a wrong byte count silently
# rewrites the conclusion. Binary codes pad to a 16-byte boundary with a 16 B
# floor; scalar int8 carries +4 B/vec of metadata; product x8 packs 2 dims per
# code byte (`get_bucket_size` maps x4->1, x8->2, x16->4, x32->8, x64->16).
@pytest.mark.parametrize("name,expected", [
    ("prod-128-int8", 132),
    ("256-int8", 260),
    ("256-pq8", 128),     # == the arctic card's pq256x4fs target, exactly
    ("256-bin", 32),
    ("256-none", 0),
    ("1024-bin", 128),
])
def test_code_bytes_per_vector_matches_task412_measurements(name, expected):
    assert sc.get(name).code_bytes_per_vector() == expected


def test_the_target_hits_the_model_cards_128_bytes_exactly():
    """"256/4-bit" must be 128.0 B/vec or it is not the card's configuration."""
    cfg = sc.get("256-pq8")
    assert cfg.quantizer == "pq_x8" and cfg.dense_dim == 256
    assert cfg.code_bytes_per_vector() == 128


# ---------------------------------------------------------------------------
# quantizers that NEED rescore must declare it
# ---------------------------------------------------------------------------
def test_sub_byte_quantizers_declare_rescore_and_oversampling():
    """Binary and PQ are unusable without rescore, and rescore alone is not enough.

    Task 412: binary WITHOUT rescore collapses -29.0 pp at d128 and -20.9 pp at
    d256; and because rescore only re-ranks what the beam already found,
    oversampling >= 3 is what actually recovers recall (48.97 at x1 -> 51.10 at
    x3). A registry entry that forgets this measures a configuration nobody would
    ship.
    """
    for name, cfg in sc.CONFIGS.items():
        if cfg.quantizer in ("binary_1bit", "pq_x8"):
            assert cfg.rescore is True, f"{name} must rescore"
            # the single named exception: task 2830/2831, Tim 2026-11-16
            floor = 1.0 if name == "256-pq8" else 3.0
            assert cfg.oversampling >= floor, f"{name} needs oversampling >= {floor}"


def test_int8_keeps_rescore_off():
    """ADR-023 is VINDICATED for int8 at n=3,000 (task 412 SS11.4).

    int8/d128 rescore-off scores 52.30 against an unquantized ceiling of 51.93 --
    indistinguishable from no quantization. Keep rescore off while the quantizer
    is int8; turning it on would only buy latency.
    """
    for name, cfg in sc.CONFIGS.items():
        if cfg.quantizer == "int8":
            assert cfg.rescore is False, f"{name} should not rescore"


# ---------------------------------------------------------------------------
# MRL: the transform, and the mirror contract with the query side
# ---------------------------------------------------------------------------
def test_mrl_trunc_renormalizes():
    """A prefix of a unit vector is not a unit vector, and the distance is Cosine."""
    v = np.random.RandomState(0).randn(8, 1024).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for d in (128, 256):
        out = np.asarray(sc.mrl_trunc(v, d), dtype="float32")
        assert out.shape == (8, d)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
        # it is a PREFIX, not a projection: direction must be preserved
        for i in range(8):
            pre = v[i, :d] / np.linalg.norm(v[i, :d])
            assert np.allclose(out[i], pre, atol=1e-5)


def test_mrl_trunc_is_a_noop_at_or_above_full_width():
    v = np.random.RandomState(1).randn(4, 1024).astype("float32")
    assert sc.mrl_trunc(v, 1024) is v
    assert sc.mrl_trunc(v, 0) is v


def test_bakeoff_reexports_the_same_function_object():
    """One implementation, not two that agree today.

    ~10 scripts do `from cytir.embed.bakeoff import mrl_trunc`. If someone ever pastes
    the body back into cytir/embed/bakeoff.py "to remove a dependency", this fails.
    """
    from cytir.embed import bakeoff
    assert bakeoff.mrl_trunc is sc.mrl_trunc


def test_query_side_reads_the_width_from_the_environment():
    """`cytir/serve/models.py:MRL_DIM` must be env-driven, defaulting to 128.

    This is the flip's atomicity contract. A 256-dim collection queried by a
    128-dim embedder is an OUTAGE, not a degradation, so `AW_MRL_DIM` has to move
    in the same deploy as `AW_COLLECTION`. Asserted by reading the source rather
    than by importing, because importing `models` pulls in the embedder stack.
    """
    src = (_HERE.parent / "cytir" / "serve" / "models.py").read_text()
    m = re.search(r"^MRL_DIM\s*=\s*(.+)$", src, re.M)
    assert m, "MRL_DIM disappeared from models.py"
    expr = m.group(1)
    # S10 step 2 (task 1071): the env read is a typed settings field
    assert expr.strip() == "_S.runtime.mrl_dim", (
        "MRL_DIM must read AW_MRL_DIM; a literal here means a 256-dim collection "
        "cannot be queried without editing and redeploying the server")
    settings_src = (_HERE.parent / "cytir" / "serve" / "settings.py").read_text()
    assert 'mrl_dim: int = knob("AW_MRL_DIM", INT, "128"' in settings_src, "the default must stay 128"


def test_there_is_no_default_build_config():
    # the retired "prod-128-int8" default is gone (slice 8 step 1): naming nothing refuses
    assert not hasattr(sc, "DEFAULT_BUILD_CONFIG")


def test_unknown_config_refuses_loudly():
    with pytest.raises(KeyError):
        sc.get("256-4bit")   # a plausible-looking name that is not a real cell


# ---------------------------------------------------------------------------
# payload indexes
# ---------------------------------------------------------------------------
def test_published_is_declared_so_rebuilds_stop_dropping_it():
    """`published` is indexed on the live collection but created by no repo code.

    It was written out-of-band in task 354, so every fresh `create_collection`
    silently shipped one fewer index than production had -- and the news lane
    filters on `published`, where an unindexed field means a full scan (the
    `prod-serving-box-fragile-scrolls` failure).
    """
    assert "published" in sc.PAYLOAD_INDEXES


def test_payload_indexes_cover_the_live_schema():
    live = {"domain", "url", "source_type", "country", "tier", "lang",
            "trust_tier", "register_id", "lei", "published"}
    assert live <= set(sc.PAYLOAD_INDEXES), (
        f"missing from the rebuild: {live - set(sc.PAYLOAD_INDEXES)}")


# ---------------------------------------------------------------------------
# the indexed-tolerance guard (task 478 incident)
# ---------------------------------------------------------------------------
def test_indexed_tolerance_is_chosen_by_the_quantizer_not_a_magic_number():
    """How much un-indexed tail is tolerable depends on whether it searches WRONG.

    MEASURED at 1,824,204 points -- the pattern is the mechanism, not noise:

        binary one_bit   residual  1,204  (0.066%)   appendable IS quantized
        none             residual 10,204  (0.56%)    nothing to quantize
        int8 / product   residual      0  (exact)    optimizer FORCED to convert

    `supports_appendable()` is true only for Binary and Turbo, so only for
    scalar/product does an un-indexed point mean an UNQUANTIZED scan -- task 412
    SS10.7's trap, and the only case that needs a tight bar. Written for the trap
    alone, `indexed == points` cost four hours on a wait that could not finish and
    then failed a perfectly good unquantized control arm.
    """
    assert sc.get("256-int8").appendable_is_quantized is False
    assert sc.get("256-pq8").appendable_is_quantized is False
    assert sc.get("256-bin").appendable_is_quantized is True
    assert sc.get("256-none").appendable_is_quantized is True

    strict = sc.get("256-pq8").indexed_tolerance()
    lenient = sc.get("256-none").indexed_tolerance()
    assert strict < lenient, "the trap-prone quantizers must have the tighter bar"

    pts = 1_824_204
    # the two measured residuals must pass under their own quantizer's bar
    assert 1_204 <= pts * sc.get("256-bin").indexed_tolerance()
    assert 10_204 <= pts * sc.get("256-none").indexed_tolerance()
    # and a materially unindexed scalar/product collection must still fail
    assert pts * 0.10 > pts * strict


# --------------------------------------------------------------------------
# the gate's view of the same config (task 604)
# --------------------------------------------------------------------------

def test_the_gate_env_is_the_deploy_env_with_the_weights_resolved_locally():
    """One home for the policy; only the weights PATH differs by host."""
    deploy = sc.flip_deploy_env()
    gate = sc.gate_dense_env("/rel/cytir/serve")
    for key in ("AW_DENSE_RESCORE", "AW_DENSE_OVERSAMPLING",
                "AW_DENSE_IGNORE_QUANT"):
        assert gate[key] == deploy[key]
    assert gate["AW_FUSION_WEIGHTS"] == \
        "/rel/cytir/serve/" + sc.FLIP_FUSION_WEIGHTS_FILE
    assert deploy["AW_FUSION_WEIGHTS"].startswith("/app/"), (
        "the deploy path is a CONTAINER path -- a gate arm that inherited it "
        "would read nothing and silently fall back")
    # the width is the target collection's, never the profile's (task 600)
    assert "AW_MRL_DIM" not in gate


def _binding(**over):
    doc = {"schema": sc.GATE_BINDING_SCHEMA, "mrl_dim": 256,
           "dense_rescore": True, "dense_oversampling": 1.0,
           "dense_ignore_quant": False, "dense_hnsw_ef": 384,
           "fusion_weights_sha256": sc.FLIP_FUSION_WEIGHTS_SHA256}
    doc.update(over)
    return doc


def test_an_arm_that_ran_productions_ranker_has_no_problems():
    assert sc.gate_serving_binding_problems(_binding()) == []
    # string-valued oversampling (as an env round-trip would leave it) is
    # compared as a number, not as text
    assert sc.gate_serving_binding_problems(_binding(dense_oversampling="1.0")) == []


@pytest.mark.parametrize("over, why", [
    ({"dense_rescore": False}, "dense_rescore"),
    ({"dense_oversampling": 3.0}, "dense_oversampling"),
    ({"dense_ignore_quant": True}, "dense_ignore_quant"),
    ({"fusion_weights_sha256": "0" * 64}, "fusion weights sha256"),
    ({"schema": "other"}, "schema"),
])
def test_an_arm_that_did_not_is_named_and_refused(over, why):
    problems = sc.gate_serving_binding_problems(_binding(**over))
    assert problems and any(why in p for p in problems)


@pytest.mark.parametrize("value", [None, {}, "rescore on, trust me", []])
def test_a_missing_or_malformed_binding_is_a_problem_never_a_pass(value):
    assert sc.gate_serving_binding_problems(value)


def test_the_flip_weights_pointer_still_names_a_real_promoted_artifact():
    """`gate_serving_binding_problems` compares against this sha256, so a
    pointer that names a file the repo does not hold would refuse every arm."""
    import hashlib
    path = (pathlib.Path(__file__).resolve().parents[1] / "cytir" / "serve" /
            sc.FLIP_FUSION_WEIGHTS_FILE)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == \
        sc.FLIP_FUSION_WEIGHTS_SHA256


def test_package_module_is_the_one_home():
    """Task 1184 (S9): the flat serving_config alias is retired; the package module is the one home."""
    assert (_HERE.parent / "cytir" / "core" / "serving_config.py").is_file()
