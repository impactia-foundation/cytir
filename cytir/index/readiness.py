#!/usr/bin/env python3
# Moved from the retired `qdrant_readiness` script (task 677, task 643 S5); the old path is an alias shim.
"""Is this collection in its SETTLED state, segment by segment?

Why this exists
---------------
``apply_content_pack_shared.wait_green`` waits for ``status: green``,
``optimizer_status: ok`` and an empty update queue.  None of those three is a
statement about whether the segments a write just created have been indexed.
Qdrant keeps a segment below ``optimizer_config.indexing_threshold`` (which is
in KILOBYTES) as a **plain** segment -- brute-forced, no HNSW graph, no sparse
index -- and still reports the collection green and the optimizer ok.  A
routine content pack is small enough to land entirely in one such segment:
14,819 points x 256 dims x 2 bytes of float16 dense is about 7.6 MB against a
10,000 KB threshold.

Task 336 is the incident: *"a 90-point upsert on aw_b1 broke 4 of 663 gate
golds"*, because a filtered sparse query against a plain segment returns
filter-matching points scored 0.0.  Task 606 priced the same failure mode on
recorded pools at up to 0.74 of the BM25 leg's normalised range.  So "green"
is not "ready to be measured", and a gate that scores a collection in that
state is measuring the write as well as the content.

This module is the readiness predicate, read from Qdrant's own two surfaces
and nothing else:

* ``GET /collections/<name>`` -- ``status``, ``optimizer_status``,
  ``update_queue_length``, ``segments_count``;
* ``GET /telemetry?details_level=5`` -- per-segment ``segment_type``,
  ``num_points``, ``num_vectors``, ``num_indexed_vectors``,
  ``num_deleted_vectors``, ``is_appendable``.

It **fails closed**.  An unreachable endpoint, a collection that telemetry
does not mention, a shard with no segments, or a segment whose type field is
missing are all problems -- never a silent pass.  ``main()`` carries the
verdict in its exit code (0 ready, 1 not ready, 2 cannot tell), because a
check that always exits 0 is decoration.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
from cytir.core import internal_http as IH
from typing import Any, Iterable, Mapping, Sequence

#: The segment type Qdrant reports for a segment carrying built indexes.
#: Everything else -- notably ``plain`` -- is a segment a filtered sparse
#: query can answer with 0.0-scored rows.
INDEXED = "indexed"

#: Fields every segment row must carry for the predicate to be computable.
SEGMENT_FIELDS = ("segment_type", "num_points", "num_vectors",
                  "num_indexed_vectors", "num_deleted_vectors")


class ReadinessError(RuntimeError):
    """Readiness cannot be determined -- never treated as ready."""


# ------------------------------------------------------------------ transport
def get_json(url: str, timeout: float = 60.0) -> dict:
    try:
        return IH.fetch_json(url, timeout=timeout)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ReadinessError(f"cannot read {url}: {exc}") from exc


def fetch_collection(base_url: str, collection: str, timeout: float = 60.0) -> dict:
    doc = get_json(f"{base_url.rstrip('/')}/collections/{collection}", timeout)
    result = doc.get("result")
    if not isinstance(result, dict):
        raise ReadinessError(f"{collection}: collection info has no result object")
    return result


def fetch_telemetry(base_url: str, timeout: float = 120.0) -> dict:
    doc = get_json(f"{base_url.rstrip('/')}/telemetry?details_level=5", timeout)
    result = doc.get("result")
    if not isinstance(result, dict):
        raise ReadinessError("telemetry has no result object")
    return result


# -------------------------------------------------------------- segment rows
def segment_rows(telemetry: Mapping[str, Any], collection: str) -> list[dict]:
    """Every segment of every local shard of `collection`, flattened.

    Raises rather than returning an empty list: "no segments" is the reading
    that would make an unready collection look ready.
    """
    collections = (telemetry.get("collections") or {}).get("collections")
    if not isinstance(collections, list):
        raise ReadinessError("telemetry carries no collections list -- "
                             "details_level too low?")
    entries = [item for item in collections
               if isinstance(item, dict) and item.get("id") == collection]
    if not entries:
        raise ReadinessError(f"{collection}: not present in telemetry")
    rows: list[dict] = []
    for entry in entries:
        shards = entry.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ReadinessError(f"{collection}: telemetry lists no shards")
        for index, shard in enumerate(shards):
            local = (shard or {}).get("local")
            if not isinstance(local, dict):
                continue          # a remote shard reports no segments here
            segments = local.get("segments")
            if not isinstance(segments, list) or not segments:
                raise ReadinessError(
                    f"{collection}: local shard {index} lists no segments")
            for position, segment in enumerate(segments):
                info = (segment or {}).get("info")
                if not isinstance(info, dict):
                    raise ReadinessError(
                        f"{collection}: shard {index} segment {position} has no info")
                missing = [field for field in SEGMENT_FIELDS if field not in info]
                if missing:
                    raise ReadinessError(
                        f"{collection}: shard {index} segment {position} "
                        f"missing {missing}")
                rows.append({"shard": index, "segment": position,
                             "segment_type": info["segment_type"],
                             "num_points": info["num_points"],
                             "num_vectors": info["num_vectors"],
                             "num_indexed_vectors": info["num_indexed_vectors"],
                             "num_deleted_vectors": info["num_deleted_vectors"],
                             "is_appendable": info.get("is_appendable")})
    if not rows:
        raise ReadinessError(f"{collection}: no local segments found")
    return rows


def plain_segments(rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    return [dict(row) for row in rows if row.get("segment_type") != INDEXED]


def unindexed_segments(rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Segments carrying live points whose dense vectors are not all indexed.

    ``num_indexed_vectors`` counts the dense vectors with a built HNSW graph;
    ``num_points`` counts live points.  A segment where the first is short of
    the second has points that can only be answered by brute force.
    """
    out = []
    for row in rows:
        if int(row.get("num_indexed_vectors") or 0) < int(row.get("num_points") or 0):
            out.append(dict(row))
    return out


# --------------------------------------------------------------- the predicate
def update_queue_length(info: Mapping[str, Any]) -> int | None:
    """The queue depth, from either shape the surfaces report it in.

    `GET /collections/<name>` nests it as `update_queue.length`; the Python
    client's `apply_content_pack_shared.state()` flattens it to
    `update_queue_length`. Both are accepted; absent is None, never 0 --
    reading a missing field as an empty queue is exactly the mistake this
    module exists to refuse.
    """
    if "update_queue_length" in info and info["update_queue_length"] is not None:
        return int(info["update_queue_length"])
    queue = info.get("update_queue")
    if isinstance(queue, Mapping) and queue.get("length") is not None:
        return int(queue["length"])
    return None


def readiness_problems(info: Mapping[str, Any],
                       rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Empty list means SETTLED. Anything else names what is not settled.

    `info` is the `result` object of `GET /collections/<name>`; `rows` is
    `segment_rows(...)`.
    """
    problems: list[str] = []
    status = info.get("status")
    if status != "green":
        problems.append(f"status={status!r} (want 'green')")
    optimizer = info.get("optimizer_status")
    if optimizer != "ok":
        problems.append(f"optimizer_status={optimizer!r} (want 'ok')")
    queue = update_queue_length(info)
    if queue is None:
        problems.append("update_queue_length absent")
    elif queue:
        problems.append(f"update_queue_length={queue} (want 0)")
    plain = plain_segments(rows)
    if plain:
        problems.append(
            f"{len(plain)} non-indexed segment(s): "
            + ", ".join(f"{row['segment_type']}/{row['num_points']}pts"
                        for row in plain))
    short = unindexed_segments(rows)
    if short:
        problems.append(
            f"{len(short)} segment(s) with unindexed points: "
            + ", ".join(f"{row['num_indexed_vectors']}/{row['num_points']}"
                        for row in short))
    return problems


def snapshot(base_url: str, collection: str) -> dict:
    """One readiness reading, with everything the verdict was computed from."""
    info = fetch_collection(base_url, collection)
    rows = segment_rows(fetch_telemetry(base_url), collection)
    problems = readiness_problems(info, rows)
    return {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "collection": collection,
        "url": base_url,
        "status": info.get("status"),
        "optimizer_status": info.get("optimizer_status"),
        "update_queue_length": update_queue_length(info),
        "points_count": info.get("points_count"),
        "indexed_vectors_count": info.get("indexed_vectors_count"),
        "segments_count": info.get("segments_count"),
        "indexing_threshold": ((info.get("config") or {})
                               .get("optimizer_config") or {}).get("indexing_threshold"),
        "segments": [dict(row) for row in rows],
        "plain_segments": plain_segments(rows),
        "problems": problems,
        "ready": not problems,
    }


def wait_ready(base_url: str, collection: str, timeout: float,
               interval: float = 15.0, log=None) -> list[dict]:
    """Poll until settled or `timeout`. Returns the whole timeline.

    The caller decides what an unsettled deadline means; the timeline's last
    entry always carries `ready` and `problems`, so a caller cannot mistake
    "timed out" for "settled".
    """
    deadline = time.time() + timeout
    timeline: list[dict] = []
    while True:
        reading = snapshot(base_url, collection)
        timeline.append(reading)
        if log is not None:
            log(reading)
        if reading["ready"] or time.time() >= deadline:
            return timeline
        time.sleep(interval)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True, help="Qdrant base URL")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--wait", type=float, default=0.0,
                        help="poll up to this many seconds for readiness")
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--out", help="write the timeline as JSON here")
    args = parser.parse_args(argv)
    try:
        if args.wait > 0:
            timeline = wait_ready(args.url, args.collection, args.wait,
                                  args.interval,
                                  log=lambda r: print(
                                      f"[{r['at']}] ready={r['ready']} "
                                      f"segments={r['segments_count']} "
                                      f"problems={r['problems']}", flush=True))
        else:
            timeline = [snapshot(args.url, args.collection)]
    except ReadinessError as exc:
        print(f"READINESS UNKNOWN: {exc}", file=sys.stderr)
        return 2
    final = timeline[-1]
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"schema": "aw.qdrant-readiness-timeline.v1",
                       "timeline": timeline}, handle, indent=1, sort_keys=True)
    print(json.dumps({key: value for key, value in final.items()
                      if key != "segments"}, indent=1, sort_keys=True))
    return 0 if final["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
