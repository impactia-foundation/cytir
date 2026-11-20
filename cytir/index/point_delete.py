#!/usr/bin/env python3
# Moved from the pipeline-era `point_delete` module (task 677, task 643 S5); the old path was retired.
"""The one home for "delete these point ids, verified gone" (task 641).

Extracted from `upsert_supersede.Superseder.run()` (task 348), whose inline
delete loop was the only place this project deleted points from a live
collection before task 641's pack-declared tombstones needed the same
primitive for a second caller. Rather than a second, independently-written
delete-and-verify (the exact defect engineering-doctrine warns against --
"a second implementation converts one bug into two"), `Superseder.run()` now
calls this function too when constructed with a `url` (see its docstring);
callers with no URL handy keep the untouched, byte-identical fallback path,
so every existing caller's behaviour is unchanged.

Deletion is not "done" until VERIFIED gone: `wait=true` on the delete call
is not itself proof (a segment merge race, a retry-without-idempotency), so
every batch is re-read afterward and any surviving id is a hard failure --
"collection state untrustworthy" is the right reaction, never a retry that
might paper over a real problem.
"""
from __future__ import annotations

from cytir.core import internal_http as IH

from cytir.index import dry_run as DRY
from cytir.index import prod_guard as PG


class PointDeleteError(RuntimeError):
    """Points meant to be gone are still present after delete(wait=true)."""


def _api(url: str, path: str, body: dict | None = None, method: str | None = None):
    req_method = method or ("POST" if body is not None else "GET")
    DRY.refuse_http(req_method, path)
    PG.check(url, req_method, path)      # incident 2026-10-07: the one prod gate
    payload = IH.fetch_json(url.rstrip("/") + path, body=body, method=req_method, timeout=300)
    if payload.get("status") not in ("ok", None):
        raise PointDeleteError(f"{path} returned {payload.get('status')!r}")
    return payload.get("result")


def delete_points_verified(url: str, collection: str, ids: list[str], *,
                           batch: int = 1000, log=print) -> int:
    """Delete `ids` from `collection` at `url`, batched, each batch verified
    gone via a retrieve before moving to the next. Returns the count DELETED:
    ids that were present and are now gone.

    An id the collection did not hold before the delete (task 860, review 854
    S4: an old-id tombstone on a collection keyed by the I1 ids deletes
    nothing) is counted as NOT-FOUND, never as deleted; it is logged. The
    batch's presence is read first (one retrieve), only the present ids are
    sent to the delete, and the same retrieve afterwards proves them gone.

    Raises `PointDeleteError` (never a bare assert/SystemExit here — the
    caller decides what "untrustworthy collection state" means for its own
    transaction) if any id survives its batch's delete.
    """
    DRY.refuse_http("POST", f"/collections/{collection}/points/delete")   # before the pre-read: no I/O when dry
    deleted = not_found = 0
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        held = {str(r.get("id")) for r in (_api(
            url, f"/collections/{collection}/points",
            {"ids": chunk, "with_payload": False, "with_vector": False}) or [])}
        present = [x for x in chunk if str(x) in held]
        not_found += len(chunk) - len(present)
        if present:
            _api(url, f"/collections/{collection}/points/delete?wait=true",
                 {"points": present}, method="POST")
            back = _api(url, f"/collections/{collection}/points",
                        {"ids": present, "with_payload": False, "with_vector": False})
            if back:
                raise PointDeleteError(
                    f"{len(back)} of {len(present)} id(s) still present in "
                    f"{collection} after delete(wait=true) -- collection state "
                    "untrustworthy, refusing to proceed")
        deleted += len(present)
        log(f"[point_delete] {collection}: deleted+verified {deleted:,}, "
            f"not-found {not_found:,} of {len(ids):,}")
    return deleted


def set_payload_batch(url: str, collection: str, operations: list[dict], *, api=None) -> None:
    """The ONE `points/batch` `set_payload` writer (task 2777): `operations` is a list of qdrant
    `set_payload` bodies (`{"payload": {...}, "points": [ids]}`), sent as ONE `wait=true` call.
    The payload_patch drain (`QdrantPatchClient.write`), `twin-patch-fields` and the `published`
    reconcile apply (task 2786) all call this.
    `api` defaults to this module's `_api` (looked up at call time); the drain passes the retrying
    `promote_sealed_pack._api`."""
    (api or _api)(url, f"/collections/{collection}/points/batch?wait=true",
                  {"operations": [{"set_payload": o} for o in operations]})


def fetch_payload_fields(url: str, collection: str, ids: list[str],
                         fields: list[str] | bool, *, batch: int = 1000, api=None) -> dict[str, dict]:
    """`{id: {field: value, ...}}` for `ids` present in `collection`. An id
    this returns nothing for was not found -- callers that need every id to
    exist check that separately (this is a read, not a presence assertion).

    `fields` is qdrant's own `with_payload` value: a field-name allowlist,
    or `True` for the full payload (task 673 review round 4, finding 5 --
    a comparison that needs the whole payload, not a fixed subset)."""
    out: dict[str, dict] = {}
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        got = (api or _api)(url, f"/collections/{collection}/points",
                            {"ids": chunk, "with_payload": fields, "with_vector": False})
        for row in (got or []):
            out[str(row["id"])] = row.get("payload") or {}
    return out


def fetch_rows(url: str, collection: str, ids: list[str], *, vectors: list[str] | None = None,
               batch: int = 1000) -> dict[str, dict]:
    """`{id: {"payload": {...}, "vector": {"dense": [...]}}}` for `ids` present in `collection`:
    the FULL payload and the named vectors (default `dense`), by id. The same request count as
    `fetch_payload_fields` for the same `batch` (task 833: a content compare adds no prod call,
    only a bigger answer). An id this returns nothing for was not found."""
    out: dict[str, dict] = {}
    names = ["dense"] if vectors is None else list(vectors)
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        got = _api(url, f"/collections/{collection}/points",
                   {"ids": chunk, "with_payload": True, "with_vector": names})
        for row in (got or []):
            vec = row.get("vector")
            out[str(row["id"])] = {"payload": row.get("payload") or {},
                                   "vector": vec if isinstance(vec, dict) else {}}
    return out
