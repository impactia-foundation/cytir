# Moved from the retired `dense_storage_precision` script (task 677, task 643 S5); the old path was removed (task 683 S9).
"""The precision a Qdrant collection actually stores its dense vectors at.

One home for a rule that used to be an unstated assumption: *an exact readback
proof must compare the retrieved vector against the artifact vector AS STORED,
not against the float32 the artifact happens to be written in.*

Why this module exists
----------------------
``apply_content_pack_shared.py`` re-reads every point it wrote and requires a
byte-exact match against the artifact.  That check had only ever run against
``prod-128-int8`` -- a config whose ORIGINAL vectors are ``float32`` (its int8
is scalar *quantization*, a separate search-side copy), so the round trip was
exact and nobody noticed the assumption.

Prod now serves ``aw_e8`` at ``256-pq8``, whose original vectors are declared
``datatype: float16``.  A float32 artifact vector written into float16 storage
comes back rounded -- structurally, always, for every correct write.  Task 598
measured the consequence: a twin write that was correct in every other respect
(count delta exact, membership exact, payloads identical) reported **0 of
14,819** rows identical and aborted before the ADR-043 gate.

What this module does NOT do
----------------------------
It does not add a tolerance, and it does not skip the check.  ``float16``
storage has a single, exactly-defined rounding (IEEE 754 binary16,
round-to-nearest-even); the stored value is therefore a *deterministic
function* of the artifact value, and the comparison stays an equality.  A
vector that is wrong by one float16 ULP still fails, because one ULP is a
different float16.

Scope: the DENSE leg only.  Task 600 measured 60 real points of the task-598
pack against the live ``aw_e8`` twin: the dense values differ in 60/60 and
match ``as_stored(artifact, float16)`` in 60/60, while the bm25 sparse values
come back **bit-identical** in 60/60 -- ``SparseIndexParams(datatype=...)``
governs the sparse *index*, not the stored vector.  Payloads were identical in
60/60.  Casting the sparse leg would therefore be a real weakening of the
check, not a correction; it is deliberately left byte-exact.  The frozen
evidence is the task 600 fp16 round-trip test fixture.

Fail-closed: an unknown datatype raises rather than falling back to float32,
a value float16 cannot hold raises rather than becoming ``inf``, and a
non-integral value under uint8 storage raises rather than being silently
truncated to something that would then compare "equal".
"""
from __future__ import annotations

import math
import struct

FLOAT32 = "float32"
FLOAT16 = "float16"
UINT8 = "uint8"
SUPPORTED = (FLOAT32, FLOAT16, UINT8)

# The vector name every AW collection uses for its dense leg.  Duplicated from
# `serving_config.DENSE` rather than imported so this module stays stdlib-only
# (it is imported by the pure comparison layer); `test_dense_storage_precision`
# asserts the two never drift apart.
DENSE_VECTOR_NAME = "dense"


class DensePrecisionError(ValueError):
    """The value or the collection cannot support an exact storage comparison."""


def normalize_datatype(value) -> str:
    """Canonical datatype name for whatever a Qdrant surface reports.

    Accepts the client's enum (``Datatype.FLOAT16``), its raw JSON string
    (``"float16"``), and ``None``.  ``None`` means the collection declared no
    datatype, which is Qdrant's ``float32`` default -- and is also exactly the
    behaviour every caller had before this module existed.
    """
    if value is None:
        return FLOAT32
    text = str(getattr(value, "value", value)).strip().lower()
    text = text.rsplit(".", 1)[-1]          # "datatype.float16" -> "float16"
    if text in ("", "none", "null"):
        return FLOAT32
    if text not in SUPPORTED:
        raise DensePrecisionError(
            f"unsupported dense storage datatype {value!r}: the exact readback "
            f"proof has no measured rounding for it")
    return text


def _round_trip(fmt: str, number: float, datatype: str) -> float:
    try:
        result = struct.unpack(fmt, struct.pack(fmt, number))[0]
    except OverflowError as exc:
        raise DensePrecisionError(
            f"vector value {number!r} outside {datatype} range") from exc
    if not math.isfinite(result):
        raise DensePrecisionError(
            f"vector value {number!r} is not finite in {datatype}")
    # Qdrant/JSON transports need not preserve the sign bit of zero; the values
    # are numerically identical and must not trigger an upsert.
    return 0.0 if result == 0 else result


def as_stored(value, datatype: str) -> float:
    """The value this collection would hand back after storing ``value``.

    float16 goes through float32 first because that is the real path: the
    client sends float32, Qdrant narrows float32 -> float16.  Rounding once
    through each step is what the storage does, so it is what this reproduces.
    """
    number = float(value)
    if not math.isfinite(number):
        raise DensePrecisionError("non-finite vector value")
    if datatype == FLOAT32:
        return _round_trip("!f", number, FLOAT32)
    if datatype == FLOAT16:
        return _round_trip("<e", _round_trip("!f", number, FLOAT32), FLOAT16)
    if datatype == UINT8:
        if number != int(number) or not 0.0 <= number <= 255.0:
            raise DensePrecisionError(
                f"uint8 dense storage cannot hold {number!r}: only exact "
                f"0..255 integers round-trip, and this project has never "
                f"measured how Qdrant rounds anything else")
        return float(int(number))
    raise DensePrecisionError(f"unsupported dense storage datatype {datatype!r}")


def quantize_dense(values, datatype: str) -> list[float]:
    """``values`` as the collection storing them at ``datatype`` would return."""
    return [as_stored(component, datatype) for component in values]


def _attr_or_key(value, name):
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def dense_params_of(info):
    """The dense vector params of one collection, from a model or raw JSON."""
    config = _attr_or_key(info, "config")
    if config is None:
        raise DensePrecisionError("collection info carries no config")
    params = _attr_or_key(config, "params")
    if params is None:
        raise DensePrecisionError("collection config carries no params")
    vectors = _attr_or_key(params, "vectors")
    if vectors is None:
        raise DensePrecisionError("collection declares no dense vectors config")
    if isinstance(vectors, dict):
        if DENSE_VECTOR_NAME not in vectors:
            raise DensePrecisionError(
                f"collection has no {DENSE_VECTOR_NAME!r} vector: "
                f"{sorted(vectors)}")
        return vectors[DENSE_VECTOR_NAME]
    return vectors


def dense_datatype_of(info) -> str:
    """Canonical dense storage datatype declared by one collection.

    Read from the collection being compared, at comparison time, so no caller
    can hand in a softer answer than the target's own config.
    """
    return normalize_datatype(_attr_or_key(dense_params_of(info), "datatype"))


def dense_size_of(info) -> int:
    """Dense width one collection declares, as a positive int.

    The query side has no freedom here: `cytir/serve/models.py` truncates
    the embedder's 1024-dim output to ``AW_MRL_DIM`` and Qdrant rejects any
    other width outright.  So anything that queries a collection must take this
    number FROM that collection -- a pinned or defaulted width is not a tuning
    choice, it is an invalid request.

    Task 600: the ADR-043 content-pack gate inherited the shipped default
    (128) while pointed at 256-dim `aw_e8`, and every one of its 1,418 queries
    came back `__ERROR__:UnexpectedResponse`.  The arm was written anyway.
    """
    size = _attr_or_key(dense_params_of(info), "size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise DensePrecisionError(
            f"collection declares no usable dense size: {size!r}")
    return size
