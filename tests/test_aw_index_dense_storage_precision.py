"""The guard for task 600's fix: an exact readback proof must compare against
the precision the target actually STORES, and must stay an equality.

The fixture is real: the ``testdata`` file ``task600_fp16_roundtrip.json`` holds
four (artifact, as-stored) point pairs captured on 2026-09-17 from the
task-598 news pack and the live ``aw_e8`` twin (the build host, dense
``datatype: float16``, 256-dim, product-x8 quantized) AFTER that pack's twin
write -- the same write whose exactness check reported 0 of 14,819 rows
identical and aborted the apply.

The two load-bearing tests are:

  * ``test_the_real_fp16_pack_fails_the_float32_comparison_and_passes_ours`` --
    the old comparison FAILS on every real pair and the new one PASSES;
  * ``test_one_float16_ulp_of_corruption_still_fails`` -- a vector wrong by a
    single float16 ULP still FAILS, i.e. this is a correction, not a tolerance.

Everything else here keeps the fix from quietly growing into a tolerance:
the sparse leg and the payload stay byte-exact, unknown datatypes refuse, and
the float32 path is unchanged.
"""
from __future__ import annotations

import json
import pathlib
import struct

import pytest

HERE = pathlib.Path(__file__).resolve().parent
from cytir.index import dense_precision as DSP  # noqa: E402
from cytir.index.noop import (  # noqa: E402
    PointComparisonError, compare_batches, normalize_point)

FIXTURE = HERE / "testdata" / "task600_fp16_roundtrip.json"


@pytest.fixture(scope="module")
def pairs() -> list[dict]:
    value = json.loads(FIXTURE.read_text())
    assert value["schema"] == "aw.fp16-roundtrip-fixture.v1"
    assert value["provenance"]["dense_datatype"].lower().endswith("float16")
    assert value["provenance"]["collection"] == "aw_e8"
    assert value["pairs"], "the fixture must carry real pairs"
    return value["pairs"]


def _point(point_id: str, dense: list[float], bm25: dict,
           payload: dict | None = None) -> dict:
    return {"id": point_id, "payload": payload or {"url": "https://x/1"},
            "vector": {"dense": list(dense), "bm25": dict(bm25)}}


# --------------------------------------------------------------- the defect

def test_the_real_fp16_pack_fails_the_float32_comparison_and_passes_ours(pairs):
    """Task 598's abort, reproduced, then fixed -- on real captured points."""
    for pair in pairs:
        artifact = _point(pair["id"], pair["artifact_dense"],
                          pair["artifact_bm25"])
        stored = _point(pair["id"], pair["stored_dense"], pair["stored_bm25"])

        # What the applier did before task 600: compare at float32.
        old = compare_batches([[artifact]], lambda _ids: [stored])
        assert (old.identical, old.changed) == (0, 1), (
            f"{pair['id']}: this pair is supposed to reproduce the defect")

        # What it does now: compare at the target's declared storage datatype.
        new = compare_batches([[artifact]], lambda _ids: [stored],
                              dense_datatype=DSP.FLOAT16)
        assert (new.identical, new.changed, new.missing) == (1, 0, 0), (
            f"{pair['id']}: a correct write must compare identical")
        assert new.dense_datatype == DSP.FLOAT16
        assert new.artifact_points_sha256 == new.existing_points_sha256


def test_every_real_stored_dense_value_is_exactly_as_stored_of_the_artifact(pairs):
    """The stored side is a deterministic function of the artifact side.

    This is the whole licence for keeping the check an equality: there is one
    right answer for what float16 storage returns, and it is reproducible
    off-line from the artifact alone.
    """
    for pair in pairs:
        predicted = DSP.quantize_dense(pair["artifact_dense"], DSP.FLOAT16)
        assert predicted == pair["stored_dense"], pair["id"]
        # ... and the pairs really do differ at float32, so the equality above
        # is not trivially true.
        assert pair["artifact_dense"] != pair["stored_dense"], pair["id"]


# ------------------------------------------------- it is not a tolerance

def test_one_float16_ulp_of_corruption_still_fails(pairs):
    """A vector wrong by the SMALLEST amount float16 can express still fails."""
    pair = pairs[0]
    corrupted = list(pair["stored_dense"])
    bits = struct.unpack("<H", struct.pack("<e", corrupted[0]))[0]
    corrupted[0] = struct.unpack("<e", struct.pack("<H", bits + 1))[0]
    assert corrupted[0] != pair["stored_dense"][0]

    artifact = _point(pair["id"], pair["artifact_dense"], pair["artifact_bm25"])
    stored = _point(pair["id"], corrupted, pair["stored_bm25"])
    result = compare_batches([[artifact]], lambda _ids: [stored],
                             dense_datatype=DSP.FLOAT16)
    assert (result.identical, result.changed) == (0, 1)
    assert pair["id"] in result.write_ids


def test_a_sub_ulp_sparse_difference_still_fails(pairs):
    """The sparse leg is NOT narrowed: bm25 postings come back bit-identical.

    Task 600 measured 60/60 real points whose bm25 values round-tripped
    exactly from float16-dense storage, so a difference far below a float16
    ULP is a real difference and must be written.
    """
    pair = pairs[0]
    bm25 = {"indices": list(pair["stored_bm25"]["indices"]),
            "values": list(pair["stored_bm25"]["values"])}
    bm25["values"][0] = struct.unpack(
        "!f", struct.pack("!I", struct.unpack(
            "!I", struct.pack("!f", bm25["values"][0]))[0] + 1))[0]

    artifact = _point(pair["id"], pair["artifact_dense"], bm25)
    stored = _point(pair["id"], pair["stored_dense"], pair["stored_bm25"])
    result = compare_batches([[artifact]], lambda _ids: [stored],
                             dense_datatype=DSP.FLOAT16)
    assert (result.identical, result.changed) == (0, 1)


def test_a_payload_difference_still_fails(pairs):
    pair = pairs[0]
    artifact = _point(pair["id"], pair["artifact_dense"], pair["artifact_bm25"],
                      payload={"url": "https://x/1", "title": "a"})
    stored = _point(pair["id"], pair["stored_dense"], pair["stored_bm25"],
                    payload={"url": "https://x/1", "title": "b"})
    result = compare_batches([[artifact]], lambda _ids: [stored],
                             dense_datatype=DSP.FLOAT16)
    assert (result.identical, result.changed) == (0, 1)


def test_narrowing_is_applied_to_the_artifact_side_only(pairs):
    """A target that returns something float16 could not hold is NOT excused."""
    pair = pairs[0]
    impossible = list(pair["stored_dense"])
    # halfway between two float16 neighbours -- representable in float32, never
    # the result of a float16 store.
    step = abs(struct.unpack("<e", struct.pack("<e", impossible[0]))[0])
    impossible[0] = impossible[0] + step * 2 ** -12
    assert DSP.as_stored(impossible[0], DSP.FLOAT16) == pair["stored_dense"][0]

    artifact = _point(pair["id"], pair["artifact_dense"], pair["artifact_bm25"])
    stored = _point(pair["id"], impossible, pair["stored_bm25"])
    result = compare_batches([[artifact]], lambda _ids: [stored],
                             dense_datatype=DSP.FLOAT16)
    assert (result.identical, result.changed) == (0, 1), (
        "narrowing the RETRIEVED side too would hide a target that is not "
        "storing what it says it stores")


# ------------------------------------------------------------ fail-closed

def test_the_pre_flip_float32_path_is_unchanged(pairs):
    """`prod-128-int8` stores float32 originals; that comparison must not move."""
    pair = pairs[0]
    point = _point(pair["id"], pair["artifact_dense"], pair["artifact_bm25"])
    assert normalize_point(point) == normalize_point(point, DSP.FLOAT32)
    result = compare_batches([[point]], lambda _ids: [point])
    assert (result.identical, result.changed, result.missing) == (1, 0, 0)
    assert result.dense_datatype == DSP.FLOAT32


@pytest.mark.parametrize("declared,expected", [
    (None, DSP.FLOAT32),
    ("float32", DSP.FLOAT32),
    ("float16", DSP.FLOAT16),
    ("FLOAT16", DSP.FLOAT16),
    ("Datatype.FLOAT16", DSP.FLOAT16),
    ("uint8", DSP.UINT8),
])
def test_normalize_datatype_accepts_every_surface_qdrant_reports(declared,
                                                                expected):
    assert DSP.normalize_datatype(declared) == expected


def test_an_unknown_datatype_refuses_rather_than_assuming_float32():
    with pytest.raises(DSP.DensePrecisionError):
        DSP.normalize_datatype("bfloat16")
    with pytest.raises(PointComparisonError):
        normalize_point(_point("i", [0.5], {"indices": [1], "values": [1.0]}),
                        "bfloat16")


def test_a_value_float16_cannot_hold_refuses_rather_than_becoming_inf():
    with pytest.raises(DSP.DensePrecisionError):
        DSP.as_stored(1e5, DSP.FLOAT16)
    assert DSP.as_stored(1e5, DSP.FLOAT32) == pytest.approx(1e5)


def test_uint8_storage_refuses_anything_it_cannot_round_trip():
    assert DSP.as_stored(7, DSP.UINT8) == 7.0
    for bad in (0.5, -1, 256):
        with pytest.raises(DSP.DensePrecisionError):
            DSP.as_stored(bad, DSP.UINT8)


def test_signed_zero_is_normalized_in_every_datatype():
    for datatype in (DSP.FLOAT32, DSP.FLOAT16):
        stored = DSP.as_stored(-0.0, datatype)
        assert stored == 0.0 and str(stored) == "0.0"


def test_dense_datatype_is_read_from_the_collection_not_the_caller():
    body = {"config": {"params": {"vectors": {
        "dense": {"size": 256, "datatype": "float16"}}}}}
    assert DSP.dense_datatype_of(body) == DSP.FLOAT16

    class _Params:
        vectors = {"dense": type("V", (), {"size": 128, "datatype": None})()}

    class _Info:
        config = type("C", (), {"params": _Params()})()

    assert DSP.dense_datatype_of(_Info()) == DSP.FLOAT32


@pytest.mark.parametrize("body", [
    {},
    {"config": {}},
    {"config": {"params": {}}},
    {"config": {"params": {"vectors": {"other": {"size": 1}}}}},
])
def test_an_unreadable_collection_config_refuses(body):
    with pytest.raises(DSP.DensePrecisionError):
        DSP.dense_datatype_of(body)


def test_the_dense_vector_name_has_not_drifted_from_serving_config():
    from cytir.core import serving_config

    assert DSP.DENSE_VECTOR_NAME == serving_config.DENSE


def test_every_serving_config_datatype_is_one_this_module_can_reproduce():
    """A new serving config must not be able to introduce a storage precision
    the exactness proof has no measured rounding for."""
    from cytir.core import serving_config

    for name, cfg in serving_config.CONFIGS.items():
        assert DSP.normalize_datatype(cfg.dense_datatype) in DSP.SUPPORTED, name


# ------------------------------- the width is a property of the target too

def test_dense_size_is_read_from_the_collection():
    body = {"config": {"params": {"vectors": {
        "dense": {"size": 256, "datatype": "float16"}}}}}
    assert DSP.dense_size_of(body) == 256


@pytest.mark.parametrize("size", [0, -1, None, "256", 12.0, True])
def test_an_unusable_dense_size_refuses_rather_than_defaulting(size):
    body = {"config": {"params": {"vectors": {"dense": {"size": size}}}}}
    with pytest.raises(DSP.DensePrecisionError):
        DSP.dense_size_of(body)


def test_the_real_fixture_collection_is_the_width_prod_serves(pairs):
    value = json.loads(FIXTURE.read_text())
    assert value["provenance"]["dense_size"] == len(pairs[0]["artifact_dense"])
    from cytir.core import serving_config

    assert (value["provenance"]["dense_size"] ==
            serving_config.get(serving_config.FLIP_SERVING_CONFIG).dense_dim)
