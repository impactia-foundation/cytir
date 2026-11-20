#!/usr/bin/env python3
"""Seal one collection's exact baseline membership as a portable receipt.

`content_pack_preservation` requires each apply target to present an
`aw.content-pack-baseline-membership-receipt.v1` whose `points` equals the
count the live preflight observes and whose `logical_membership_sha256` is
**identical on both mirrors** — that identity is what proves twin and
production start the transaction holding the same set of points, rather than
merely the same *number* of them (two collections can agree on a count and
disagree on membership).

The digest is deliberately trivial to re-derive by hand:

    logical_membership_sha256 = sha256( "".join(f"{id}\\n" for id in sorted(ids)) )

i.e. exactly ``sort | sha256sum`` over the point ids, one per line, in C
collation. Qdrant's unfiltered scroll already returns points in ascending id
order, so this streams in constant memory — and it **asserts** that order
rather than assuming it: an out-of-order id aborts the seal instead of
silently producing a digest that a re-derivation with `sort` would not match.

The scroll is id-only (no payload, no vectors) and paced, because production
here is a small box and an unpaced full scan of it is a known way to saturate
its API pool.

`python -m cytir.index.seal_collection_membership verify ...` is the other membership instrument:
a rebuilt collection's `{id, payload}` multiset against a sealed payload export (the second half
of this module; moved from the pipeline tree in task 1234, behaviour unchanged).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from cytir.core import internal_http as IH
from pathlib import Path
from typing import Iterable

from cytir.index import prod_guard as PG  # no prod scroll without a reason (2026-10-07)
from cytir.acquire.artifact_io import atomic_publish
from cytir.index import preservation_contract as full
from cytir.core.digest import canonical_utf8_bytes


RECEIPT_SCHEMA = "aw.content-pack-baseline-membership-receipt.v1"
EVIDENCE_SCHEMA = "aw.collection-membership-evidence.v1"
DIGEST_RULE = "sha256 over ascending point ids, one per line, C collation"


class MembershipError(RuntimeError):
    """The collection's membership cannot be sealed as an exact set."""


def _relative(root: Path, path: Path):
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise MembershipError(
            f"{path} is outside the portable artifact root {root}") from exc


def _artifact(root: Path, path: Path) -> dict:
    relative = _relative(root, path)
    body = path.read_bytes()
    return {"uri": f"artifact://{relative.as_posix()}", "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest()}


def stream_membership(scroll, *, batch: int, pace_seconds: float,
                      progress=None) -> tuple[str, int, list[dict]]:
    """(digest, count, checkpoints) over an ascending id stream.

    `scroll(offset, limit)` returns `(ids, next_offset)`; it is injected so
    this module is testable without a live Qdrant and so the caller owns the
    client's timeouts.
    """
    digest = hashlib.sha256()
    checkpoints: list[dict] = []
    previous: str | None = None
    count = 0
    offset = None
    index = 0
    while True:
        ids, offset = scroll(offset, batch)
        if not ids:
            if offset is not None:
                raise MembershipError("scroll returned an empty page with more to read")
            break
        for point_id in ids:
            point_id = str(point_id)
            if previous is not None and point_id <= previous:
                raise MembershipError(
                    "scroll is not strictly ascending by point id "
                    f"({previous!r} then {point_id!r}); the streamed digest "
                    "would not match a sorted re-derivation")
            digest.update(f"{point_id}\n".encode())
            previous = point_id
            count += 1
        checkpoints.append({"page": index, "rows": len(ids),
                            "first_id": str(ids[0]), "last_id": str(ids[-1]),
                            "cumulative_rows": count,
                            "running_sha256": digest.hexdigest()})
        index += 1
        if progress is not None:
            progress(count)
        if offset is None:
            break
        if pace_seconds:
            time.sleep(pace_seconds)
    if not count:
        raise MembershipError("collection is empty; nothing to seal")
    return digest.hexdigest(), count, checkpoints


def seal(*, artifact_root: Path, role: str, url: str, collection: str,
         expect_points: int, scroll, receipt_path: Path, evidence_path: Path,
         batch: int, pace_seconds: float, observed_points: int,
         progress=None) -> dict:
    if role not in ("twin", "prod"):
        raise MembershipError("role must be twin or prod")
    if observed_points != expect_points:
        raise MembershipError(
            f"{role} reports {observed_points} points, expected {expect_points}")
    # Refuse before writing anything a clone could not resolve.
    for path in (receipt_path, evidence_path):
        _relative(artifact_root, path)
    logical, count, checkpoints = stream_membership(
        scroll, batch=batch, pace_seconds=pace_seconds, progress=progress)
    if count != expect_points:
        raise MembershipError(
            f"{role} enumerated {count} point ids but the collection counts "
            f"{expect_points}; membership is not exact")
    evidence = {
        "schema": EVIDENCE_SCHEMA, "role": role, "url": url,
        "collection": collection, "points": count,
        "digest_rule": DIGEST_RULE, "logical_membership_sha256": logical,
        "scroll_batch": batch, "pages": len(checkpoints),
        "checkpoints": checkpoints,
    }
    _publish(evidence_path, evidence)
    receipt = {
        "schema": RECEIPT_SCHEMA, "status": "PASS", "role": role,
        "collection": collection, "points": count,
        "logical_membership_sha256": logical,
        "verification_evidence": _artifact(artifact_root, evidence_path),
    }
    _publish(receipt_path, receipt)
    return {**_artifact(artifact_root, receipt_path), "rows": count,
            "logical_membership_sha256": logical, "points": count}


def _publish(path: Path, value: dict) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise MembershipError(f"immutable membership collision: {path}")
        return
    atomic_publish(path, encoded)


def _post(url: str, path: str, body: dict, timeout: int, *,
          allow_prod_scroll: bool = False, reason: str | None = None) -> dict:
    """One Qdrant REST call over the standard library.

    Deliberately not `qdrant_client`: this tool must run anywhere the mirror
    is reachable — including a production box that has no AW virtualenv — and
    a stranger re-deriving the digest should need nothing but Python.
    """
    # incident 2026-10-07: a full id scroll of prod is refused unless an operator names a reason
    try:
        PG.check(url, "POST", path, allow_prod_scroll=allow_prod_scroll, reason=reason)
    except PG.ProdReadRefused as exc:
        raise MembershipError(str(exc)) from exc
    payload = IH.fetch_json(url.rstrip("/") + path, body=body, method="POST", timeout=timeout)
    if payload.get("status") != "ok":
        raise MembershipError(f"{path} returned {payload.get('status')!r}")
    return payload["result"]


def exact_count(url: str, collection: str, timeout: int) -> int:
    return int(_post(url, f"/collections/{collection}/points/count",
                     {"exact": True}, timeout)["count"])


def rest_scroll(url: str, collection: str, timeout: int, *, prod_scroll_reason: str | None = None):
    def scroll(offset, limit):
        body = {"limit": limit, "with_payload": False, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        result = _post(url, f"/collections/{collection}/points/scroll",
                       body, timeout, allow_prod_scroll=bool(prod_scroll_reason),
                       reason=prod_scroll_reason)
        return ([str(point["id"]) for point in result["points"]],
                result.get("next_page_offset"))
    return scroll


def main(argv: list[str] | None = None) -> int:
    """Seal (the default) or, with a leading `verify`, verify payload membership (`verify_main`)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["verify"]:
        return verify_main(argv[1:], prog="python -m cytir.index.seal_collection_membership verify")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--role", required=True, choices=("twin", "prod"))
    parser.add_argument("--expect-points", type=int, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=20000)
    parser.add_argument("--pace-seconds", type=float, default=0.05)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--allow-prod-scroll", metavar="REASON", default=None,
                        help="a full id scroll of PROD scans the serving box (incident 2026-10-07): "
                             "refused unless this names the operator's reason (it is logged)")
    args = parser.parse_args(argv)
    observed = exact_count(args.url, args.collection, args.timeout)
    last = [0.0]

    def progress(count: int) -> None:
        now = time.time()
        if now - last[0] >= 30:
            last[0] = now
            print(f"[membership] {args.role} {count}/{args.expect_points}",
                  file=sys.stderr, flush=True)

    try:
        result = seal(
            artifact_root=args.artifact_root.resolve(), role=args.role,
            url=args.url.rstrip("/"), collection=args.collection,
            expect_points=args.expect_points,
            scroll=rest_scroll(args.url, args.collection, args.timeout,
                               prod_scroll_reason=args.allow_prod_scroll),
            receipt_path=args.receipt.resolve(),
            evidence_path=args.evidence.resolve(), batch=args.batch,
            pace_seconds=args.pace_seconds, observed_points=observed,
            progress=progress)
    except (MembershipError, OSError, ValueError) as exc:
        print(f"MEMBERSHIP SEAL FAIL: {exc}", file=sys.stderr)
        return 1
    full._validate_artifact(
        {k: result[k] for k in ("uri", "bytes", "sha256")},
        "membership_receipt", artifact_root=args.artifact_root.resolve(),
        deep=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0



# --- verify: rebuilt payload membership against a sealed export receipt --- #

_VERIFY_DOC = """Verify rebuilt Qdrant payload membership against a sealed export receipt.

The expected side is a sharded payload export whose receipt records compressed
file size, SHA-256, and row count (the Task-416 receipt is supported directly).
The live side is a complete Qdrant scroll with payloads and no vectors.  Both
sides are reduced to the same order-independent cryptographic multiset proof of
canonical ``{"id": str, "payload": object}`` rows.  Count and three independent
256-bit accumulators must match; the collection count must also remain unchanged
across the scan.
"""

MASK = (1 << 256) - 1
PAYLOAD_RECEIPT_SCHEMAS = {
    "aw.task416.qdrant-membership-receipt.v1",
    "aw.qdrant-membership-receipt.v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def membership_row_bytes(row: dict) -> bytes:
    """The canonical `{"id": str, "payload": object}` bytes of one row (`cytir.core.digest` encodes)."""
    if not isinstance(row, dict) or "id" not in row:
        raise MembershipError("membership row must contain id")
    payload = row.get("payload") or {}
    if not isinstance(payload, dict):
        raise MembershipError(f"payload for id {row.get('id')!r} is not an object")
    return canonical_utf8_bytes({"id": str(row["id"]), "payload": payload})


def fingerprint(rows: Iterable[dict]) -> dict:
    count = xor = total = squares = 0
    for row in rows:
        value = int.from_bytes(hashlib.sha256(membership_row_bytes(row)).digest(), "big")
        count += 1
        xor ^= value
        total = (total + value) & MASK
        squares = (squares + value * value) & MASK
    return {
        "algorithm": "sha256-multiset-xor-sum-squares-v1",
        "rows": count,
        "xor": f"{xor:064x}",
        "sum": f"{total:064x}",
        "sum_squares": f"{squares:064x}",
    }


def _load_receipt(path: Path) -> dict:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MembershipError(f"cannot read membership receipt {path}: {exc}") from exc
    if not isinstance(receipt, dict) or receipt.get("schema") not in PAYLOAD_RECEIPT_SCHEMAS:
        raise MembershipError(
            f"unsupported membership receipt schema: {receipt.get('schema')!r}")
    if not isinstance(receipt.get("shards"), list) or not receipt["shards"]:
        raise MembershipError("membership receipt has no shards")
    return receipt


def expected_rows(receipt: dict, root: Path):
    total = 0
    for index, shard in enumerate(receipt["shards"]):
        where = f"shards[{index}]"
        if not isinstance(shard, dict):
            raise MembershipError(f"{where} is not an object")
        name = shard.get("file")
        if not isinstance(name, str) or not name or Path(name).is_absolute() \
                or ".." in Path(name).parts:
            raise MembershipError(f"{where}.file is not a safe relative path")
        path = root / name
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise MembershipError(f"missing membership shard {path}: {exc}") from exc
        if size != shard.get("bytes"):
            raise MembershipError(
                f"membership shard size mismatch {name}: {size} != {shard.get('bytes')}")
        actual_sha = sha256_file(path)
        if actual_sha != shard.get("sha256"):
            raise MembershipError(
                f"membership shard SHA mismatch {name}: {actual_sha} != {shard.get('sha256')}")
        rows = 0
        try:
            with gzip.open(path, "rt", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    rows += 1
                    total += 1
                    yield json.loads(line)
        except (OSError, json.JSONDecodeError) as exc:
            raise MembershipError(f"cannot decode membership shard {name}: {exc}") from exc
        if rows != shard.get("rows"):
            raise MembershipError(
                f"membership shard row mismatch {name}: {rows} != {shard.get('rows')}")
    if total != receipt.get("rows"):
        raise MembershipError(
            f"membership receipt conservation failed: {total} != {receipt.get('rows')}")


def api_json(url: str, body: dict | None = None, timeout: int = 300) -> dict:
    result = IH.fetch_json(url, body=body, timeout=timeout)
    if result.get("status") != "ok":
        raise MembershipError(f"Qdrant request failed: {result}")
    return result["result"]


def collection_count(base: str, collection: str) -> int:
    return int(api_json(f"{base}/collections/{collection}")["points_count"])


def live_rows(base: str, collection: str, batch: int, *, prod_scroll_reason: str | None = None):
    # incident 2026-10-07: a full payload scroll of prod is refused unless an operator names a reason
    try:
        PG.check(base, "POST", f"/collections/{collection}/points/scroll",
                 allow_prod_scroll=bool(prod_scroll_reason), reason=prod_scroll_reason)
    except PG.ProdReadRefused as exc:
        raise MembershipError(str(exc)) from exc
    offset = None
    while True:
        body = {"limit": batch, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        result = api_json(f"{base}/collections/{collection}/points/scroll", body)
        points = result.get("points") or []
        for point in points:
            yield {"id": point["id"], "payload": point.get("payload") or {}}
        offset = result.get("next_page_offset")
        if offset is None:
            break


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def verify_main(argv: list[str] | None = None, prog: str | None = None) -> int:
    """`python -m cytir.index.seal_collection_membership verify ...`: the payload-membership proof."""
    parser = argparse.ArgumentParser(prog=prog, description=_VERIFY_DOC)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--membership-root", type=Path,
                        help="shard directory (default: receipt directory)")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--expect-count", type=int)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--allow-prod-scroll", metavar="REASON", default=None,
                        help="a full payload scroll of PROD scans the serving box (incident "
                             "2026-10-07): refused unless this names the operator's reason")
    args = parser.parse_args(argv)
    try:
        receipt_path = args.receipt.expanduser().resolve()
        receipt = _load_receipt(receipt_path)
        root = (args.membership_root or receipt_path.parent).expanduser().resolve()
        expected = fingerprint(expected_rows(receipt, root))
        declared = int(receipt["rows"])
        if expected["rows"] != declared:
            raise MembershipError(
                f"expected fingerprint rows {expected['rows']} != receipt {declared}")
        if args.expect_count is not None and declared != args.expect_count:
            raise MembershipError(
                f"receipt rows {declared} != --expect-count {args.expect_count}")
        base = args.qdrant_url.rstrip("/")
        before = collection_count(base, args.collection)
        if before != declared:
            raise MembershipError(
                f"collection count before scan {before} != receipt rows {declared}")
        actual = fingerprint(live_rows(
            base, args.collection, args.batch, prod_scroll_reason=args.allow_prod_scroll))
        after = collection_count(base, args.collection)
        if after != before:
            raise MembershipError(
                f"collection count changed during verification: {before} -> {after}")
        if actual != expected:
            raise MembershipError(
                "rebuilt collection membership fingerprint differs from receipt: "
                f"expected={expected}, actual={actual}")
        proof = {
            "schema": "aw.collection-membership-verification.v1",
            "status": "PASS", "collection": args.collection,
            "qdrant_url": base, "receipt": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "fingerprint": actual,
        }
        if args.proof:
            atomic_json(args.proof.expanduser().resolve(), proof)
        print(json.dumps(proof, indent=2, sort_keys=True))
        return 0
    except (MembershipError, OSError, ValueError) as exc:
        print(f"MEMBERSHIP VERIFY FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
