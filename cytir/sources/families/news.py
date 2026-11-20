#!/usr/bin/env python3
"""Materialize one immutable news pack and explicit predecessor pointers.

Moved from the task 385 scratch script `materialize_news_pack.py` (task 646 S4: "nothing
in production runs from scratch dirs"). A shim at the old path re-exports this
module unchanged -- see that file's own docstring. The two live unattended
callers (the retired `hn_discussion_daily` script, `cytir/sources/refresh_direct/direct_compose.py`)
and the task 598 `news_tier2_apply.sh` `pack` step (the daily timer's
own driver) now point at this path directly.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections import defaultdict
from datetime import date
from pathlib import Path

from qdrant_client import QdrantClient, models

import sys
# This file now lives at cytir/sources/families/news.py -- three levels below
# the repo root (families/ -> sources/ -> cytir/ -> root), one deeper than its
# old home: the task 385 scratch dir (one level shallower).
ROOT = Path(__file__).resolve().parents[3]
# Run as a script (the daily pack step: `$QPY cytir/sources/families/news.py`) sys.path[0] is
# THIS directory, so `aw` is importable only if the repo root is put on the path. The
# reverse-import move (task 677 S5) used `cytir.index.supersede` here without doing so; the
# r20 release failed `ModuleNotFoundError: No module named 'cytir'` in the pack step.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cytir.acquire.gzip_pinned import deterministic_gzip_open  # noqa: E402
from cytir.core.digest import canonical_line_bytes as canonical  # noqa: E402
from cytir.core.qdrant import qdrant_connect  # noqa: E402
from cytir.index.supersede import payload_group_key, url_variants  # noqa: E402


class MaterializeError(RuntimeError):
    pass


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def vector_json(vector) -> dict:
    result = {}
    for name, value in (vector or {}).items():
        if hasattr(value, "indices"):
            result[name] = {"indices": list(value.indices),
                            "values": list(value.values)}
        elif isinstance(value, dict) and {"indices", "values"} <= set(value):
            result[name] = {"indices": list(value["indices"]),
                            "values": list(value["values"])}
        else:
            result[name] = list(value)
    return result


def point_json(point) -> dict:
    return {"id": str(point.id), "vector": vector_json(point.vector),
            "payload": point.payload or {}}


def atomic(path: Path, data: bytes) -> None:
    """Write once, immutably -- EXCEPT every path this module writes is
    content-addressed by a hash OF `data` itself (news-points.<sha>.jsonl.gz,
    tombstone-pointers.<sha>.json, staged-verification.<sha>.json), so a
    second call with a path that already exists can ONLY mean "the exact
    same bytes were computed again" -- never a real collision to guard
    against. Before this fix, a second call crashed with FileExistsError
    on the FIRST write (the artifact, always attempted first) whenever
    staging was unchanged since a prior run, which silently skipped the
    REST of this script -- including the twin-dependent tombstone/supersede
    scan below, whose answer depends on the CURRENT twin, not on whether
    staging changed. A catch-up day whose OWN staging is stable but whose
    target twin has moved (because an earlier day landed since this pack
    was last materialized) got a STALE tombstone-pointers file with no
    error, only a downstream "tombstones drifted" abort in
    apply_content_pack_shared.py's preflight, hours later -- found live
    against 2026-09-21 (task 626), the day after 2026-09-15 landed and
    moved twin forward. Idempotent re-write: if content differs from what
    is already on disk at this exact content-addressed path, that is a
    genuine hash collision (astronomically unlikely) or a logic bug
    upstream, and still raises loudly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        existing = path.read_bytes()
        if existing != data:
            raise MaterializeError(
                f"content-addressed path {path} exists with DIFFERENT bytes "
                "than what was just computed -- this should be impossible "
                "for a content-addressed name; do not silently proceed")
        return
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def content_json(out_dir: Path, stem: str, value: dict) -> Path:
    data = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    path = out_dir / f"{stem}.{digest_bytes(data)}.json"
    atomic(path, data)
    return path


def collect_points(client: QdrantClient, collection: str) -> list[dict]:
    rows, offset = [], None
    while True:
        points, offset = client.scroll(
            collection, limit=1000, offset=offset,
            with_payload=True, with_vectors=True)
        rows.extend(point_json(point) for point in points)
        if offset is None:
            break
    expected = client.count(collection, exact=True).count
    if len(rows) != expected:
        raise MaterializeError(f"{collection}: scroll {len(rows)} != count {expected}")
    return rows


def existing_group_points(client: QdrantClient, collection: str,
                          keep: dict[str, set[str]], variants: dict[str, set[str]]) -> dict[str, dict]:
    found = {}
    groups = sorted(keep)
    for start in range(0, len(groups), 64):
        batch = groups[start:start + 64]
        values = sorted(set().union(*(variants[group] for group in batch)))
        condition = models.Filter(must=[models.FieldCondition(
            key="url", match=models.MatchAny(any=values))])
        offset = None
        while True:
            points, offset = client.scroll(
                collection, scroll_filter=condition, limit=1000, offset=offset,
                with_payload=True, with_vectors=False)
            for point in points:
                payload = point.payload or {}
                group = payload_group_key(payload)
                point_id = str(point.id)
                if group in keep and point_id not in keep[group]:
                    found[point_id] = {"group": group, "payload": payload}
            if offset is None:
                break
    return found


def parse_day(value) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def resolve_aged_out_overlap(
        aged_ids_raw: list[str], stale_ids: list[str],
        fresh_ids: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Task 625: a point can be BOTH a stale same-URL-group predecessor
    (`stale_ids` -- today's pack carries a newer version at that URL) AND
    independently >36mo old by the twin's own `published` date
    (`aged_ids_raw`). That is not a double-delete: the point is retired by
    the REPLACEMENT already, so tombstoning it a second time as "aged out, no
    replacement" would be the wrong reason for a point that in fact has one.

    Separately (found running the real chain against 2026-09-17, after the
    first overlap fix landed): the aged-out candidate list is computed by an
    upstream, independent date scan (`--aged-ids`, from news_refresh_delta.sh)
    against the twin BEFORE this pack's own writes -- it has no way to know
    which ids THIS pack is about to mint. A point whose id is derived from
    (url, content) can come back with the SAME id as an already-aged twin
    point when a long catch-up re-stages a URL whose content has not changed
    since it first aged past the retention cutoff. Tombstoning it as
    "aged out, no replacement" while the SAME pack simultaneously upserts it
    as live content is not a double-delete like the stale case -- it is
    write-then-immediately-delete-what-you-just-wrote, and the apply's own
    exact-membership verify (apply_content_pack_shared.py's verify_target)
    is the only thing that ever caught it: it aborted with "twin artifact
    membership {short} != {rows}" after the twin write+delete had already
    executed, not before. `fresh_ids` is the artifact's own minted id set
    (`keep`'s union in the caller) -- any aged candidate also present there
    is dropped for the same reason as the stale case: it has a "replacement"
    (itself, unchanged), so it is not truly retiring.

    Returns (aged_ids, dropped_as_superseded, dropped_as_still_current) --
    the aged-out set with both overlaps removed, and each overlap itself, so
    the caller can attribute every drop in its own evidence rather than lose
    it silently.
    """
    stale = set(stale_ids)
    fresh = set(fresh_ids or ())
    raw = set(aged_ids_raw)
    dropped_as_superseded = sorted(raw & stale)
    dropped_as_still_current = sorted((raw - stale) & fresh)
    kept = sorted(raw - stale - fresh)
    return kept, dropped_as_superseded, dropped_as_still_current


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-url", required=True)
    parser.add_argument("--staging-collection", action="append", required=True)
    parser.add_argument("--twin-url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--aged-ids", type=Path, required=True)
    parser.add_argument("--cutoff", type=date.fromisoformat, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    staging = qdrant_connect(args.staging_url, timeout=900,
                             check_compatibility=False)
    twin = qdrant_connect(args.twin_url, timeout=900,
                          check_compatibility=False)
    rows = []
    source_counts = {}
    for collection in sorted(args.staging_collection):
        source = collect_points(staging, collection)
        source_counts[collection] = len(source)
        rows.extend(source)
    rows.sort(key=lambda row: row["id"])
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise MaterializeError("staging arms contain duplicate point ids")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    compressed = io.BytesIO()
    with deterministic_gzip_open(compressed, compresslevel=9) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical(row).decode())
    artifact_data = compressed.getvalue()
    artifact_sha = digest_bytes(artifact_data)
    artifact_path = args.out_dir / f"news-points.{artifact_sha}.jsonl.gz"
    atomic(artifact_path, artifact_data)

    keep, variants = defaultdict(set), defaultdict(set)
    for row in rows:
        payload = row["payload"]
        group = payload_group_key(payload)
        if not group:
            raise MaterializeError(f"point lacks groupable URL: {row['id']}")
        keep[group].add(row["id"])
        variants[group].update(url_variants(payload["url"]))
    stale = existing_group_points(twin, args.collection, keep, variants)
    stale_ids = sorted(stale)
    stale_rows = (twin.retrieve(args.collection, ids=stale_ids,
                                with_payload=True, with_vectors=True)
                  if stale_ids else [])
    stale_full = {str(point.id): point_json(point) for point in stale_rows}
    if set(stale_full) != set(stale_ids):
        raise MaterializeError("source-group predecessor readback incomplete")
    source_pointers = [{
        "predecessor_id": point_id,
        "predecessor_fingerprint": digest_bytes(canonical(stale_full[point_id])),
        "group": stale[point_id]["group"],
        "replacement_ids": sorted(keep[stale[point_id]["group"]]),
        "reason": "source_group_reemit",
    } for point_id in stale_ids]

    aged_ids_raw = sorted({line.strip() for line in args.aged_ids.read_text().splitlines()
                          if line.strip()})
    if len(aged_ids_raw) != sum(1 for line in args.aged_ids.read_text().splitlines()
                                if line.strip()):
        raise MaterializeError("aged tombstone input contains duplicate ids")
    # A long catch-up can surface a point that is both stale (superseded by
    # today's pack at the same URL) and independently aged-out, OR aged-out
    # by date yet re-minted with the SAME id by today's own artifact (content
    # unchanged since it first aged past the cutoff) -- neither is a true
    # retirement; see resolve_aged_out_overlap's docstring.
    fresh_ids = sorted(set().union(*keep.values())) if keep else []
    aged_ids, dropped_as_superseded, dropped_as_still_current = (
        resolve_aged_out_overlap(aged_ids_raw, stale_ids, fresh_ids))
    aged_full = {str(point.id): point_json(point) for point in twin.retrieve(
        args.collection, ids=aged_ids, with_payload=True, with_vectors=True)}
    # task626 (2026-09-22): the aged-ids CANDIDATE list is frozen at
    # whenever the upstream `delta` step last ran -- on a long catch-up,
    # twin moves in between (another day's own aged-out sweep, or a
    # source-group supersede, can retire the SAME point first). A
    # candidate already ABSENT from twin has already been retired by
    # someone; tombstoning it again is not possible and not needed, only
    # dropping it from THIS pack's pointer list is -- never silently, and
    # never conflated with `dropped_as_superseded`/`_still_current`, which
    # mean something different (a live point this pack's OWN logic
    # decided not to age). Found live running 09-12's re-materialize after
    # 09-13/09-15/09-21 had already applied and rolled back around it.
    dropped_as_already_retired = sorted(set(aged_ids) - set(aged_full))
    aged_ids = [pid for pid in aged_ids if pid in aged_full]
    aged_pointers = []
    for point_id in aged_ids:
        payload = aged_full[point_id]["payload"]
        published = parse_day(payload.get("published"))
        if payload.get("disc_class") != "news" or published is None or published >= args.cutoff:
            raise MaterializeError(f"aged tombstone is not proven old news: {point_id}")
        aged_pointers.append({
            "predecessor_id": point_id,
            "predecessor_fingerprint": digest_bytes(canonical(aged_full[point_id])),
            "group": payload_group_key(payload),
            "replacement_ids": [], "reason": "news_window_aged_out",
            "published": published.isoformat(), "cutoff": args.cutoff.isoformat(),
        })
    overlap = set(stale_ids) & set(aged_ids)
    if overlap:
        raise MaterializeError(f"point has two tombstone reasons: {sorted(overlap)[:5]}")

    pointers = {
        "schema": "aw.explicit-tombstone-pointers.v1",
        "artifact_sha256": artifact_sha,
        "source_group": source_pointers,
        "aged_out": aged_pointers,
        # `counts` is asserted by apply_content_pack_shared.py's
        # verify_tombstone_pointers as EXACT dict equality against
        # {"source_group": N, "aged_out": M} -- an extra key here reads as
        # "tombstone pointer counts mismatch" and aborts the whole preflight.
        # Found running this against 2026-09-17's real pack. The overlap
        # count lives OUTSIDE counts, at the top level, where nothing asserts
        # the dict's exact shape.
        "counts": {"source_group": len(source_pointers),
                   "aged_out": len(aged_pointers)},
        "aged_out_dropped_as_superseded": dropped_as_superseded,
        "aged_out_dropped_as_still_current": dropped_as_still_current,
        "aged_out_dropped_as_already_retired": dropped_as_already_retired,
    }
    pointers_path = content_json(args.out_dir, "tombstone-pointers", pointers)
    verification = {
        "schema": "aw.news-staged-verification.v1",
        "collection": "+".join(sorted(args.staging_collection)),
        "status": "green", "points_count": len(rows),
        "distinct_minted_ids": len(set(ids)), "count_match": True,
        "source_counts": source_counts, "artifact_sha256": artifact_sha,
        "tombstone_pointers": str(pointers_path),
        "tombstone_pointers_sha256": hashlib.sha256(pointers_path.read_bytes()).hexdigest(),
    }
    verification_path = content_json(args.out_dir, "staged-verification", verification)
    result = {"artifact": str(artifact_path), "artifact_sha256": artifact_sha,
              "rows": len(rows), "pointers": str(pointers_path),
              "pointers_sha256": hashlib.sha256(pointers_path.read_bytes()).hexdigest(),
              "verification": str(verification_path),
              "verification_sha256": hashlib.sha256(verification_path.read_bytes()).hexdigest(),
              "source_group_tombstones": len(source_pointers),
              "aged_tombstones": len(aged_pointers)}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
