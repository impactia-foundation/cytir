"""Exact, bounded-memory point comparison for content-pack transactions.

The applier passes the artifact through its real write projection before using
these helpers.  Sparse postings are ordered by index and the complete payload
is retained; dense values are normalized to the precision the TARGET actually
stores them at (``dense_datatype``), because that -- not float32 -- is what a
correct write round-trips to.  Equality of the resulting canonical bytes
therefore means an upsert would not change the stored point.

``dense_datatype`` defaults to float32, which is what every caller assumed
before the retired ``dense_storage_precision`` script existed; see that module for
why a float16-storage target could never satisfy the assumption.  It is
applied to the ARTIFACT side only: what the target returned is compared
exactly as returned.

Moved from task 351's `content_pack_noop.py` (task 646 S4, part of the
"351 applier + lineage + noop" group -- `cytir/index/apply.py` imports this
module directly as `cytir.index.noop`). A shim at the old path re-exports it
unchanged for any caller not yet updated (including
the dense-storage-precision test, which still loads it via a
task 351's scripts sys.path insert).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Iterable

from cytir.index.dense_precision import (  # noqa: E402
    FLOAT32, DensePrecisionError, as_stored)


class PointComparisonError(ValueError):
    """The input cannot prove exact config-B point equality."""


def _mapping(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    if hasattr(value, "dict"):
        return value.dict()
    raise PointComparisonError(f"unsupported point/vector value: {type(value).__name__}")


def _stored(value, datatype: str = FLOAT32) -> float:
    try:
        return as_stored(value, datatype)
    except DensePrecisionError as exc:
        raise PointComparisonError(str(exc)) from exc


def _f32(value) -> float:
    return _stored(value, FLOAT32)


def _normalize_sparse(value) -> dict:
    sparse = _mapping(value)
    indices = list(sparse.get("indices") or [])
    values = list(sparse.get("values") or [])
    if len(indices) != len(values):
        raise PointComparisonError("sparse index/value length mismatch")
    pairs = sorted((int(index), _f32(weight))
                   for index, weight in zip(indices, values))
    if len({index for index, _ in pairs}) != len(pairs):
        raise PointComparisonError("duplicate sparse index")
    return {"indices": [index for index, _ in pairs],
            "values": [weight for _, weight in pairs]}


def normalize_point(point, dense_datatype: str = FLOAT32) -> dict:
    """Return the exact normalized representation of one point.

    ``dense_datatype`` is the storage precision of the collection this point
    is being compared against.  Sparse values and the payload are NEVER
    re-quantized: task 600 measured bm25 postings coming back bit-identical
    from a float16 collection, so narrowing them would hide real differences.
    """
    value = _mapping(point)
    vector = _mapping(value.get("vector"))
    if set(vector) != {"dense", "bm25"}:
        raise PointComparisonError(
            f"point must contain exactly dense+bm25 vectors, got {sorted(vector)}")
    dense = vector["dense"]
    if isinstance(dense, dict) or not isinstance(dense, (list, tuple)):
        raise PointComparisonError("dense vector is not a flat sequence")
    payload = value.get("payload") or {}
    if not isinstance(payload, dict):
        raise PointComparisonError("payload is not an object")
    # JSON round-trip rejects objects that Qdrant's payload codec cannot
    # faithfully represent and normalizes tuples inside otherwise-valid data.
    try:
        payload = json.loads(json.dumps(
            payload, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise PointComparisonError(f"payload is not canonical JSON: {exc}") from exc
    return {
        "id": str(value["id"]),
        "payload": payload,
        "vector": {
            "bm25": _normalize_sparse(vector["bm25"]),
            "dense": [_stored(component, dense_datatype)
                      for component in dense],
        },
    }


def point_bytes(point, dense_datatype: str = FLOAT32) -> bytes:
    return _canonical_bytes(normalize_point(point, dense_datatype))


def _canonical_bytes(point: dict) -> bytes:
    return json.dumps(point, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def _add(hasher, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


@dataclass
class Comparison:
    rows: int
    existing: int
    identical: int
    changed: int
    missing: int
    artifact_points_sha256: str
    existing_points_sha256: str
    identical_points_sha256: str
    write_points_sha256: str
    decisions_sha256: str
    write_ids: set[str]
    # The storage precision the artifact side was narrowed to before
    # comparing.  Carried into every sealed proof so a ledger says what
    # precision its exactness claim was taken at, rather than leaving it to be
    # inferred from the collection's config at read time.
    dense_datatype: str = FLOAT32

    def evidence(self) -> dict:
        return {key: value for key, value in self.__dict__.items()
                if key != "write_ids"}


def compare_batches(
        batches: Iterable[list[object]],
        retrieve: Callable[[list[str]], Iterable[object]],
        dense_datatype: str = FLOAT32) -> Comparison:
    """Compare every artifact point to one target in bounded batches.

    ``batches`` must already contain points projected through the live write
    path (including BM25 saturation).  The only retained per-row state is the
    set of IDs that genuinely require a write.

    ``dense_datatype`` is the target's declared dense storage precision; the
    artifact side is narrowed to it, the retrieved side is compared exactly as
    the target returned it.
    """
    artifact_hash = hashlib.sha256()
    existing_hash = hashlib.sha256()
    identical_hash = hashlib.sha256()
    write_hash = hashlib.sha256()
    decision_hash = hashlib.sha256()
    rows = existing = identical = changed = missing = 0
    write_ids: set[str] = set()
    seen_ids: set[str] = set()

    for batch in batches:
        expected = []
        batch_ids = []
        for point in batch:
            normalized = normalize_point(point, dense_datatype)
            raw = _canonical_bytes(normalized)
            point_id = normalized["id"]
            expected.append((point_id, raw))
            batch_ids.append(point_id)
            _add(artifact_hash, raw)
            rows += 1
        requested = set(batch_ids)
        if len(requested) != len(batch_ids) or requested & seen_ids:
            raise PointComparisonError("duplicate artifact ID in comparison")
        seen_ids.update(requested)
        found = {}
        for point in retrieve(batch_ids):
            normalized = normalize_point(point)
            raw = _canonical_bytes(normalized)
            point_id = normalized["id"]
            if point_id in found:
                raise PointComparisonError(f"target returned duplicate ID {point_id}")
            if point_id not in requested:
                raise PointComparisonError(f"target returned unrequested ID {point_id}")
            found[point_id] = raw
        for point_id, artifact_raw in expected:
            target_raw = found.get(point_id)
            if target_raw is None:
                disposition = "missing"
                missing += 1
                write_ids.add(point_id)
                _add(write_hash, artifact_raw)
            else:
                existing += 1
                _add(existing_hash, target_raw)
                if target_raw == artifact_raw:
                    disposition = "identical"
                    identical += 1
                    _add(identical_hash, artifact_raw)
                else:
                    disposition = "changed"
                    changed += 1
                    write_ids.add(point_id)
                    _add(write_hash, artifact_raw)
            decision = json.dumps(
                {"artifact_sha256": hashlib.sha256(artifact_raw).hexdigest(),
                 "disposition": disposition,
                 "id": point_id,
                 "target_sha256": (hashlib.sha256(target_raw).hexdigest()
                                       if target_raw is not None else None)},
                sort_keys=True, separators=(",", ":")).encode()
            _add(decision_hash, decision)

    return Comparison(
        rows=rows, existing=existing, identical=identical, changed=changed,
        missing=missing,
        artifact_points_sha256=artifact_hash.hexdigest(),
        existing_points_sha256=existing_hash.hexdigest(),
        identical_points_sha256=identical_hash.hexdigest(),
        write_points_sha256=write_hash.hexdigest(),
        decisions_sha256=decision_hash.hexdigest(), write_ids=write_ids,
        dense_datatype=dense_datatype)
