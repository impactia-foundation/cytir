#!/usr/bin/env python3
# Moved from the retired `collection_layout` script (task 677, task 643 S5); the old path is an alias shim.
"""The one home for a Qdrant collection's LAYOUT FINGERPRINT — the state that
decides what the BM25 leg scores, and therefore whether two gate arms are
comparable at all.

WHY THIS EXISTS (task 627)
---------------------------------------------------------------
Qdrant v1.18.3 scores a sparse vector under `modifier: idf` as

    idf(t) = ln((n + 1) / (df + 0.5))

where `n` is summed from every segment's `sparse_index.vector_count()` and
`df(t)` from its `posting_list_len(t)` — both read from the segment's sparse-index
FILE HEADER, written when that index was last built
(`lib/segment/src/data_types/query_context.rs:223-240`,
`lib/segment/src/segment/read_view/search.rs:254-279`,
`lib/sparse/src/index/inverted_index/inverted_index_compressed_mmap.rs:134-173`).
The on-disk inverted index cannot represent a deletion — its `remove()` is a
`panic!` — so both keep counting points deleted or overwritten since.

Measured on 2026-09-21 with both mirrors at a byte-equal membership digest of
33,734,809 points: `aw_prod` scored against **n = 34,614,142** and the twin
against **34,620,572**. Same rows, same stored vectors, different corpus
statistics, therefore different BM25 scores for rows nobody touched, therefore a
different fused top-10 — 32 golds and 5 h@1 apart on the 639-gold daily slice.

Two routes move `n` and `df`, and **neither is visible to `points_count`**:

* a REBUILD (merge / vacuum / indexing) re-derives them from live points and
  discards the accumulated staleness — no write required;
* a WRITE adds postings, and an OVERWRITE of an existing id retains the
  predecessor's — so an overwrite-only pack moves every term's IDF while leaving
  `points_count` exactly unchanged. The 2026-09-17 news pack is 13,026 overwrites
  behind a net `+974`: 14.7x the posting churn its count delta suggests.

WHAT IS AND IS NOT A SETTLE PREDICATE
-------------------------------------
`status: green` and `optimizer_status: ok` are **not** settle predicates. Both
are true of a collection carrying 2.6% stale postings, and task 606 already
recorded both true of one carrying a plain (unindexed) segment. `wait_green` does
not wait either away. Settled means the optimizer log is quiet AND
`n == points_count`.

ACCORDINGLY, THIS MODULE REPORTS; IT DOES NOT SILENTLY REFUSE. `settled(...)`
is the honest predicate and today it is FALSE on both mirrors — so making it a
hard gate would close the door permanently rather than make the reading correct.
What decides comparability is the GAP between two fingerprints, which is a much
smaller and much better-behaved quantity, and which the caller gates on.
"""
from __future__ import annotations

from cytir.core import internal_http as IH
from typing import Iterable, Mapping

SCHEMA = "aw.collection-layout-fingerprint.v1"

#: Provenance for the threshold below. These are the integers read off the two
#: live mirrors on 2026-09-21 at 09:54:39Z, both at a byte-equal membership of
#: 33,734,809 points, sealed in the task 627 layout-fingerprints test fixture
#: and in task 627's `snapshot.pre-apply-0917.json`.
MEASURED_SAFE_BASIS = {
    "task": "627",
    "measured_at": "2026-09-21T09:54:39Z",
    "artifact": "tasks/627-artifacts/snapshot.pre-apply-0917.json",
    "points_count": 33734809,
    "named_vectors": 2,
    # the RAW counts, so the threshold below is derived by exactly the same
    # arithmetic as `gap()` and cannot be off by the half-posting that
    # `indexed_vectors_count / 2` produces on an odd count.
    "prod": {"indexed_vectors_count": 69228285},
    "twin": {"indexed_vectors_count": 69241144},
    "observed_cost": {"h@1": 0, "h@10": -1, "golds_moved": 2},
    "note": ("the largest stale-ratio gap measured to cost zero h@1; the "
             "2026-09-20 pair that cost 5 h@1 and 32 golds had its layout "
             "unrecorded, so nothing is interpolated"),
}


def _basis_stale_ratio(side: str) -> float:
    """The basis side's stale ratio, computed by EXACTLY the expression
    `build()` uses — same operations in the same order, so the threshold and a
    live `gap()` cannot differ by a float-associativity ulp."""
    b = MEASURED_SAFE_BASIS
    n = b[side]["indexed_vectors_count"] / b["named_vectors"]
    return (n - b["points_count"]) / b["points_count"]

#: The largest stale-postings gap MEASURED to cost zero h@1 on the 639-gold
#: daily slice (task 627 §7.4). COMPUTED from the integers above rather than
#: typed, so it cannot drift from its basis and cannot be read as a round
#: tolerance somebody chose: it is exactly the gap that was watched, on the day
#: it was watched, and the pair that produced it read h@1 206 on both sides.
#:
#: The other end of the scale is measured too and is much worse: on 2026-09-20
#: the same two mirrors, also at identical membership, read 201 vs 206 — 32 golds
#: and 5 h@1 apart — but their `n` was never captured, so the gap that cost 5 h@1
#: is UNKNOWN. Nothing here is interpolated between two points; this is simply
#: the largest gap anyone has actually watched do no harm.
#:
#: Raising it requires a measurement, not an argument: read both mirrors at
#: identical membership, score the slice on each, and show the h@1 delta is 0 at
#: the wider gap. Lowering it needs nothing.
MEASURED_SAFE_STALE_RATIO_GAP = abs(_basis_stale_ratio("twin")
                                    - _basis_stale_ratio("prod"))


def _get(url: str, path: str, timeout: int = 30):
    endpoint = f"{url.rstrip('/')}{path}"
    return IH.fetch_json(endpoint, timeout=timeout)["result"]


def fingerprint(url: str, collection: str, *, timeout: int = 60) -> dict:
    """Read one collection's layout fingerprint. Read-only: two GETs."""
    info = _get(url, f"/collections/{collection}", timeout=timeout)
    params = info["config"]["params"]
    named = len(params.get("vectors") or {}) + len(params.get("sparse_vectors") or {})
    points = int(info["points_count"])
    indexed = int(info.get("indexed_vectors_count") or 0)
    telemetry = _get(url, "/telemetry?details_level=6", timeout=timeout)
    local, physical = None, None
    for entry in telemetry.get("collections", {}).get("collections", []):
        shard = (entry.get("shards") or [{}])[0]
        candidate = shard.get("local") or {}
        if candidate.get("num_points") == points:
            local, physical = candidate, entry.get("id")
            break
    log = ((local or {}).get("optimizations") or {}).get("log") or []
    sizes = sorted(int(s.get("info", {}).get("num_points") or 0)
                   for s in ((local or {}).get("segments") or []))
    return build(
        collection=collection, url=url, physical=physical,
        points_count=points, indexed_vectors_count=indexed, named_vectors=named,
        segments_count=int(info.get("segments_count") or 0), segment_sizes=sizes,
        status=str(info.get("status")),
        optimizer_status=str(info.get("optimizer_status")),
        qdrant_startup=telemetry.get("app", {}).get("startup"),
        optimizer_entries=[
            {"name": e.get("name"), "status": e.get("status"),
             "start_at": e.get("start_at"), "end_at": e.get("end_at"),
             "segment_ids": e.get("segment_ids")}
            for e in sorted(log, key=lambda x: str(x.get("start_at") or ""))],
    )


def build(*, collection: str, url: str | None, physical: str | None,
          points_count: int, indexed_vectors_count: int, named_vectors: int,
          segments_count: int, segment_sizes: Iterable[int],
          status: str, optimizer_status: str, qdrant_startup: str | None,
          optimizer_entries: list) -> dict:
    """Assemble a fingerprint from already-read values.

    Separate from `fingerprint()` so a sealed fixture and a live read produce
    the SAME object through the SAME derivation — the derived fields are never
    computed twice in two places (engineering-doctrine: one home).
    """
    n = (indexed_vectors_count / named_vectors) if named_vectors else None
    stale = (n - points_count) if n is not None else None
    return {
        "schema": SCHEMA,
        "collection": collection, "physical": physical, "url": url,
        "points_count": points_count,
        "indexed_vectors_count": indexed_vectors_count,
        "named_vectors": named_vectors,
        "n_per_vector": n,
        "stale_postings": stale,
        "stale_ratio": (stale / points_count) if (stale is not None
                                                  and points_count) else None,
        "segments_count": segments_count,
        "segment_sizes": list(segment_sizes),
        "status": status, "optimizer_status": optimizer_status,
        "qdrant_startup": qdrant_startup,
        "optimizer_high_water": max(
            (str(e.get("end_at") or "") for e in optimizer_entries), default="")
        or None,
        "optimizer_entries": optimizer_entries,
    }


def settled(fp: Mapping) -> list[str]:
    """Why this collection is NOT settled. Empty list == settled.

    Deliberately strict, and deliberately NOT what the gate refuses on: today
    both mirrors fail it (2.6% stale postings), so refusing on it would close
    the door rather than correct the reading. The caller reports it.
    """
    problems: list[str] = []
    n = fp.get("n_per_vector")
    points = fp.get("points_count")
    if n is None:
        problems.append("indexed-vector count unavailable — `n` unknown, so the "
                        "IDF divisor cannot be checked")
    elif round(n) != points:
        problems.append(
            f"n={n:,.0f} != points_count={points:,} — "
            f"{fp['stale_postings']:+,.0f} stale postings "
            f"({fp['stale_ratio']:+.3%}) still count toward every term's IDF")
    for size in fp.get("segment_sizes") or ():
        if 0 < size < 10_000:
            problems.append(
                f"a {size:,}-point segment is below the indexing threshold — a "
                "plain segment scores a filtered sparse query differently "
                "(task 606) and `status: green` does not report it")
            break
    return problems


def optimizer_ran_between(before: Mapping, after: Mapping) -> list[dict]:
    """Optimizer passes present in `after` and not in `before`.

    Each one rebuilt a segment's sparse index from live points and therefore
    moved `n`, `df` and the BM25 leg's ranking of rows nobody touched. A pair of
    arms straddling one is not a converged reading, whatever the membership
    digest says — which is exactly what happened to task 626c's twin arms on
    2026-09-21 (baseline 08:07Z, merge 08:32:00Z→08:40:46Z, candidate 08:42Z).
    """
    if before.get("qdrant_startup") != after.get("qdrant_startup"):
        # A restart truncates the log, so absence proves nothing.
        return [{"name": "qdrant restart", "status": "unknown",
                 "start_at": before.get("qdrant_startup"),
                 "end_at": after.get("qdrant_startup"), "segment_ids": None}]
    seen = {(str(e.get("start_at")), str(e.get("end_at")))
            for e in (before.get("optimizer_entries") or ())}
    return [e for e in (after.get("optimizer_entries") or ())
            if (str(e.get("start_at")), str(e.get("end_at"))) not in seen]


def quiescent(before: Mapping, after: Mapping) -> tuple[bool, str]:
    """(quiet, reason) -- one home for "has this collection's layout moved",
    used by both am.9's fast settle path (`gate_convergence`) and the daily
    door's prod-baseline REUSE decision (task626, Tim 2026-09-21 catch-up
    acceleration item 1): a baseline is only safe to reuse across days when
    the target received no write AND its layout fingerprint has not drifted
    since it was cut -- exactly the two things a straddled optimizer pass or
    an unreadable fingerprint must never be read as "unchanged". Never
    raises: an absent/unreadable `n` just fails the check (never quiet)
    rather than aborting the caller.
    """
    if before is None or after is None:
        return False, "fingerprint unavailable"
    if before.get("n_per_vector") is None or after.get("n_per_vector") is None:
        return False, "fingerprint unreadable (n unknown)"
    if optimizer_ran_between(before, after):
        return False, "the optimizer ran between the two readings"
    unchanged = (
        before.get("points_count") == after.get("points_count")
        and before.get("n_per_vector") == after.get("n_per_vector")
        and before.get("stale_postings") == after.get("stale_postings"))
    if not unchanged:
        return False, "the layout fingerprint moved with no optimizer entry to explain it"
    return True, "optimizer log quiet, layout fingerprint unchanged"


def gap(a: Mapping, b: Mapping) -> dict:
    """The comparability gap between two fingerprints.

    `stale_ratio_gap` is the quantity to judge on, not the raw `n` difference:
    `n` legitimately rises when a pack is written, so a raw comparison of a
    pre-apply baseline against a post-apply candidate always looks broken. The
    ratio of stale postings to live points is the part of `n` that nobody chose
    and that the next optimizer pass will silently delete.
    """
    out = {
        "a": {"collection": a.get("collection"), "url": a.get("url"),
              "points_count": a.get("points_count"),
              "n_per_vector": a.get("n_per_vector"),
              "stale_postings": a.get("stale_postings"),
              "stale_ratio": a.get("stale_ratio")},
        "b": {"collection": b.get("collection"), "url": b.get("url"),
              "points_count": b.get("points_count"),
              "n_per_vector": b.get("n_per_vector"),
              "stale_postings": b.get("stale_postings"),
              "stale_ratio": b.get("stale_ratio")},
        "same_membership": a.get("points_count") == b.get("points_count"),
        "same_target": (a.get("url"), a.get("collection")) == (b.get("url"),
                                                               b.get("collection")),
    }
    ra, rb = a.get("stale_ratio"), b.get("stale_ratio")
    out["stale_ratio_gap"] = (abs(ra - rb) if (ra is not None and rb is not None)
                              else None)
    na, nb = a.get("n_per_vector"), b.get("n_per_vector")
    out["n_gap"] = (nb - na) if (na is not None and nb is not None) else None
    out["within_measured_safe"] = (
        out["stale_ratio_gap"] is not None
        and out["stale_ratio_gap"] <= MEASURED_SAFE_STALE_RATIO_GAP)
    out["measured_safe_threshold"] = MEASURED_SAFE_STALE_RATIO_GAP
    out["basis"] = MEASURED_SAFE_BASIS
    return out


def render(fp: Mapping, label: str) -> str:
    """One line per collection, for the operator who reads the log.

    Tolerant of a PARTIAL fingerprint on purpose. This is the output path: a
    renderer that raises turns an unreadable telemetry response into a traceback in
    the middle of a gate run, and so loses the very reading the operator needed in
    order to see what was wrong. Deciding whether a collection is settled belongs
    to `settled()` and to the caller; this function's only job is to say what it can
    see, and to say UNKNOWN where it cannot.
    """
    def count(value) -> str:
        return f"{value:,}" if isinstance(value, (int, float)) else "UNKNOWN"

    def signed(value) -> str:
        return f"{value:+,.0f}" if isinstance(value, (int, float)) else "UNKNOWN"

    def pct(value) -> str:
        return f"({value:+.4%})" if isinstance(value, (int, float)) else ""

    n = fp.get("n_per_vector")
    head = (f"  {label:22s} {fp.get('collection')}@{fp.get('url')} "
            f"points={count(fp.get('points_count'))}")
    if n is None:
        return f"{head} n=UNKNOWN settled=NO"
    parts = [head, f"n={n:,.0f}", f"stale={signed(fp.get('stale_postings'))}"]
    ratio = pct(fp.get("stale_ratio"))
    if ratio:
        parts.append(ratio)
    parts += [f"segs={fp.get('segments_count')}",
              f"opt_high_water={fp.get('optimizer_high_water')}",
              f"settled={'yes' if not settled(fp) else 'NO'}"]
    return " ".join(parts)


def render_gap(g: Mapping) -> str:
    srg = g.get("stale_ratio_gap")
    return (
        f"  layout gap: stale-ratio {srg:.3e} "
        f"(threshold {g['measured_safe_threshold']:.3e}, "
        f"{'within' if g['within_measured_safe'] else 'ABOVE'} the measured-safe "
        f"point) | n {g['a']['n_per_vector']:,.0f} -> {g['b']['n_per_vector']:,.0f} "
        f"({g['n_gap']:+,.0f}) | same_membership={g['same_membership']} "
        f"same_target={g['same_target']}"
        if srg is not None else "  layout gap: UNAVAILABLE (a fingerprint is missing `n`)")
