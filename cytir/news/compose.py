#!/usr/bin/env python3
"""Compose the news catch-up backlog into ONE pack (task 640, 2026-09-23).

Tim's ruling in-session, 2026-09-23 05:22Z (GO-626-standing-catchup Amendment
3): the remaining catch-up days land as ONE transaction with cross-day
supersede, not one day per transaction. ADR-110 then replaced the per-pack
battery with the five mechanical checks for PORTFOLIO-admitted families, and
`cytir/index/promote_queue.py` (task 641) became the one door -- this module
composes the backlog into the shape that driver's queue expects for a
CONTENT pack (payload+vector rows already materialized, unlike a sealed
1024-d chunks/vectors pack): merged points + a `tombstones` file (superseded
+ aged-out ids), never applied here.

WHAT THIS DOES
--------------
1. `read_pack_points(pack_dir)` -- reads one prepared day pack's
   `news-points.*.jsonl.gz` (the SAME row shape `run_catchup_day.py`
   applies: {"id", "payload", "vector"}).
2. `merge_rows(rows)` -- PURE, no I/O. Groups by url; a row with no
   `payload.published` is dropped and counted (never silently kept -- a
   pack that never learned when its own content was published is not
   dateable news). Within a url group the row with the latest
   `payload.last_seen` (ties: `published`) wins; the rest are recorded as
   internally-superseded (same day OR a different day -- one merge, one
   rule, cross-day and within-day are the same operation).
3. `compute_tombstones(...)` -- the EXISTING, already-tested reconciliation
   logic, called as functions, never re-implemented:
     * superseded: `upsert_supersede.Superseder.observe()` for every
       surviving row, then `._scan_stale()` -- the ONE selector `run()` and
       `export_stale()` both call (see that module's docstring) -- against
       the twin's CURRENT state.
     * aged_out: `news_window_filter.window_cutoff()` / `.is_news()` /
       `.in_window()` / `._partition()` against a fresh scroll of the
       twin's news points (AW_NEWS_RETAIN_MONTHS, default 36 -- Tim's
       2026-08-11 purge-horizon ruling).
4. `write_pack(...)` -- merged rows -> a `news-points.*.jsonl.gz` (same
   shape a prepared day pack already has) + `tombstones.jsonl` (schema
   below) + `manifest.json` (schema `aw.promotion-queue-pack.v1`, `kind:
   "content_pack"`) + `READY` (sha256 of manifest.json), written straight
   into the promotion-queue directory contract -- this module never writes
   to twin or prod itself.

TOMBSTONE SCHEMA (schema `aw.promotion-queue-tombstones.v1`)
--------------------------------------------------------------
    {"id": <point id>, "url": <url>, "reason": "superseded" | "aged_out"}
one per line (jsonl). `reason` is never invented here: "superseded" rows
come only from `Superseder._scan_stale()`, "aged_out" rows only from
`news_window_filter`'s own verdict -- the SAME two functions a human would
call to answer "what would this delete?" by hand.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import glob
import gzip
import hashlib
import json
import os
import pathlib
import sys
import time

from cytir.core import twin as TWIN
from cytir.core import common as AC
from cytir.core.canon import document_url_key
from cytir.core.qdrant import qdrant_client
from cytir.index import german_parts_guard as GPG
from cytir.index import news_window_filter as NWF
from cytir.index import supersede as US
from cytir.news import stage_contract as NSC
from cytir.index import row_decode as RD  # noqa: E402 -- canonical stored rows, any profile (task 855)

MANIFEST_SCHEMA = "aw.promotion-queue-pack.v1"
TOMBSTONE_SCHEMA = "aw.promotion-queue-tombstones.v1"


class DenseDimMismatch(RuntimeError):
    """A content pack's dense vectors disagree with the TARGET collection's
    own live config. Refuse to compose rather than let a bad pack pass every
    check in this module -- merge, capture-integrity, tombstones -- and fail
    only at apply time with a raw Qdrant 400 (task 648: the
    task640-news-backlog pack carried 128-dim vectors from the MRL-128 era
    against aw_e8's live 256-dim `dense`; nothing before the promotion
    driver's own upsert ever noticed)."""


def assert_dense_matches_collection(kept: list[dict], *, client, collection: str,
                                    log=print) -> dict:
    """Refuse to compose a pack whose `dense` vector name/width disagrees
    with `collection`'s OWN config, read LIVE from Qdrant -- never a
    hardcoded width, so a future collection resize (or a repoint to a
    different collection) is honoured automatically instead of silently
    re-guessed here. Checked at compose time (cheap: no GPU, no upsert)
    rather than left for the apply-time 400 the promotion driver would
    otherwise be the first to see. Every `kept` row must carry a `dense`
    vector of exactly the collection's width; a missing vector is as much a
    refusal as a wrong-width one.
    """
    info = client.get_collection(collection)
    vectors = info.config.params.vectors
    name = AC.DENSE_VEC
    if not isinstance(vectors, dict) or name not in vectors:
        raise DenseDimMismatch(
            f"{collection}: no named {name!r} vector in its live config "
            f"({vectors!r}) -- cannot verify dense width")
    expected_dim = vectors[name].size
    bad = []
    for row_ in kept:
        dense = (row_.get("vector") or {}).get(name)
        dim = None if dense is None else len(dense)
        if dim != expected_dim:
            bad.append((row_.get("id"), dim))
    if bad:
        sample = ", ".join(f"{pid}: got {dim!r}" for pid, dim in bad[:10])
        raise DenseDimMismatch(
            f"{collection} expects {name!r} dim {expected_dim}; "
            f"{len(bad)}/{len(kept)} point(s) disagree ({sample}"
            f"{', ...' if len(bad) > 10 else ''}). Refusing to compose.")
    log(f"[dense-dim-guard] {len(kept)} point(s) OK: {name!r} dim="
        f"{expected_dim} against {collection}", flush=True)
    return {"checked": len(kept), "vector_name": name, "expected_dim": expected_dim}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------- read a pack
def read_pack_points(pack_dir: str) -> list[dict]:
    """One prepared day pack's rows: [{"id", "payload", "vector"}, ...].

    Picks the LATEST-by-mtime `news-points.*.jsonl.gz` if more than one is
    present (a pack dir is never re-cut in place; the newest artifact is the
    one that survived every earlier attempt in that dir)."""
    shards = sorted(glob.glob(os.path.join(pack_dir, "news-points.*.jsonl.gz")),
                    key=os.path.getmtime)
    if not shards:
        raise FileNotFoundError(f"{pack_dir}: no news-points.*.jsonl.gz")
    rows = []
    with gzip.open(shards[-1], "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def capture_key(row: dict) -> tuple[str, str]:
    """A capture is every chunk that came out of the SAME crawl of the SAME
    url -- `(url, last_seen)`. `last_seen` is stamped once per crawl run and
    shared by every chunk that run produced (verified on real data, task
    640, 2026-09-23: rts.ch/fiction/.../27325474.html has 58 chunks, ALL
    sharing one `last_seen`; a swissinfo article had 13 chunks sharing one
    `last_seen` and one `published`). A DIFFERENT capture of the same url
    (a later re-crawl) stamps a DIFFERENT `last_seen` -- that is the one
    and only signal this schema carries for "which crawl produced this
    chunk", so it is also the one and only capture boundary."""
    payload = row.get("payload") or {}
    return (payload.get("url", ""), str(payload.get("last_seen") or ""))


def chunk_words(row: dict) -> int:
    """A word-count PROXY for one chunk: this compact row shape carries no
    full chunk text (only url/title/snippet + the embedding), so the
    closest available signal to "how much text this chunk holds" is its
    BM25 sparse vector's term count (memory present-is-not-complete:
    compare words per page, never row existence -- row existence is
    exactly what task 640's bug got right while silently discarding the
    text)."""
    bm25 = ((row.get("vector") or {}).get("bm25") or {})
    return len(bm25.get("indices") or [])


# ------------------------------------------------------------- pure merge
def merge_rows(rows: list[dict], *, require_published: bool = True) -> dict:
    """Cross-day (and within-day) supersede over a flat list of rows from
    every unlanded pack. Pure function -- no I/O, fully unit-testable.

    Supersedes by CAPTURE, never by individual row (task 640, 2026-09-23,
    coordinator finding): a real article can be many chunks at ONE url, and
    the original per-row implementation picked one arbitrary chunk per url
    and silently discarded every sibling chunk of the SAME capture as
    "superseded" -- 16,019 kept rows for 16,019 urls, when the source packs
    carried up to 58 chunks per url. Grouping by `capture_key()` first,
    THEN choosing the newest capture per url and keeping ALL of its rows,
    fixes that: only a genuinely OLDER capture (a real re-crawl, a
    different `last_seen`) is dropped, whole, never a sibling chunk.

    `require_published=False` (task 973, docs delta packs): a row with no
    `published` is a normal row, grouped by capture like any other -- the
    news-lane rule "undated = dropped" does not hold for a docs page whose
    sitemap has no `lastmod` (`published` is never a capture date).

    Returns {"kept": [...], "superseded_internal": [...], "undated": [...]}.
    `kept` and `superseded_internal` together are exactly the rows that HAD
    a published date; `undated` is reported, never silently dropped.
    """
    dated: list[dict] = []
    undated: list[dict] = []
    for row in rows:
        payload = row.get("payload") or {}
        if require_published and not payload.get("published"):
            undated.append(row)
        else:
            dated.append(row)

    captures: dict[tuple[str, str], list[dict]] = {}
    for row in dated:
        captures.setdefault(capture_key(row), []).append(row)

    captures_by_url: dict[str, list[tuple[str, str]]] = {}
    for key in captures:
        captures_by_url.setdefault(key[0], []).append(key)

    def capture_sort_key(key: tuple[str, str]) -> tuple:
        published = str((captures[key][0].get("payload") or {}).get("published") or "")
        return (key[1], published)  # (last_seen, published) -- same tiebreak as before

    kept: list[dict] = []
    superseded_internal: list[dict] = []
    for url, keys in captures_by_url.items():
        ordered = sorted(keys, key=capture_sort_key, reverse=True)
        kept.extend(captures[ordered[0]])
        for stale_key in ordered[1:]:
            superseded_internal.extend(captures[stale_key])

    return {"kept": kept, "superseded_internal": superseded_internal,
            "undated": undated}


def verify_capture_integrity(*, kept: list[dict], dated_input: list[dict],
                             log=print) -> dict[str, dict]:
    """The guard that would have caught task 640's chunk-collapse bug: for
    every url in `kept`, its kept chunk count AND word total (see
    `chunk_words()`) must equal that url's WINNING capture's chunk count
    and word total in `dated_input` -- rebuilt independently here, not read
    back from `merge_rows()`'s own bookkeeping, so a future regression in
    THAT bookkeeping still gets caught. Refuses (raises SystemExit) on any
    mismatch; a compose that silently drops chunks must never reach the
    promotion queue.

    Returns `{url: {"chunks": n, "words": n}}` for every kept url -- written
    into the manifest so the promotion driver can re-check it against the
    pack it is about to apply."""
    by_capture: dict[tuple[str, str], list[dict]] = {}
    for row in dated_input:
        by_capture.setdefault(capture_key(row), []).append(row)

    kept_by_url: dict[str, list[dict]] = collections.defaultdict(list)
    for row in kept:
        kept_by_url[(row.get("payload") or {}).get("url", "")].append(row)

    per_url: dict[str, dict] = {}
    mismatches: list[dict] = []
    for url, krows in kept_by_url.items():
        last_seens = {(r.get("payload") or {}).get("last_seen") for r in krows}
        if len(last_seens) != 1:
            mismatches.append({"url": url, "reason": "kept rows span >1 capture",
                              "last_seens": sorted(str(x) for x in last_seens)})
            continue
        source_rows = by_capture.get((url, str(next(iter(last_seens)) or "")), [])
        kept_chunks, source_chunks = len(krows), len(source_rows)
        kept_word_total = sum(chunk_words(r) for r in krows)
        source_word_total = sum(chunk_words(r) for r in source_rows)
        per_url[url] = {"chunks": kept_chunks, "words": kept_word_total}
        if kept_chunks != source_chunks or kept_word_total != source_word_total:
            mismatches.append({
                "url": url, "kept_chunks": kept_chunks, "source_chunks": source_chunks,
                "kept_words": kept_word_total, "source_words": source_word_total})

    if mismatches:
        log(f"[capture-integrity] REFUSING: {len(mismatches)} url(s) lost "
           f"chunks or words vs their source capture -- first 5: "
           f"{mismatches[:5]}")
        raise SystemExit(
            f"REFUSING compose: {len(mismatches)} url(s) failed the "
            "chunk/word capture-integrity guard (task 640) -- a compose "
            "that silently drops chunks must never reach the promotion "
            f"queue. First 5: {mismatches[:5]}")
    return per_url


# ------------------------------------------------------------- tombstones
def kept_source_types_by_group(kept: list[dict]) -> dict[str, set]:
    """Pure: `US.group_key(url, source_type)` -> the set of `source_type`s
    this pack actually keeps at that folded url."""
    out: dict[str, set] = {}
    for row in kept:
        payload = row.get("payload") or {}
        g = US.payload_group_key(payload)
        out.setdefault(g, set()).add(payload.get("source_type"))
    return out


def filter_cross_family_stale(stale_ids: list, stale_urls: list[str],
                              stale_types: dict, kept: list[dict],
                              *, stale_markers: dict | None = None,
                              log=print) -> list[dict]:
    """Pure (given `stale_types` already resolved): drop a stale match whose
    OWN `source_type` has no matching kept row at its folded url.

    `US.group_key()` folds purely on url (+ the entity_card special case),
    so it treats one kept row's url as superseding EVERY existing point at
    that same folded url, regardless of what kind of row each one is --
    right for a same-family re-crawl, wrong across families. Task 640,
    2026-09-23: one hn-freshness `source_type=reference` row stored
    `https://danluu.com/hn-comments/` (the DISCUSSED article's own url --
    AW's "HN-discussion profile" convention, not the HN thread's url), and
    that url folded onto the SAME group as 96 existing `tech_blogs`
    `source_type=blog` chunks of `https://danluu.com/hn-comments` -- a real
    page about HN comments, coincidentally also discussed on HN. Those 96
    chunks are not superseded by a lightweight HN reference pointer.

    A tombstone survives only if this pack keeps AT LEAST ONE row of the
    SAME `source_type` at that stale point's own folded url -- "a
    `superseded` tombstone is emitted only for a url that has a [matching]
    kept row in this pack" (coordinator, task 640).
    """
    # `stale_markers`: id -> {"entity_kind", "qid"} of the STORED row, so the
    # folded group carries the same card suffix the drain's stored-row check
    # derives (`supersede.is_card`); absent = no markers (legacy callers).
    stale_markers = stale_markers or {}
    kept_types = kept_source_types_by_group(kept)
    out: list[dict] = []
    dropped = 0
    for sid, surl in zip(stale_ids, stale_urls):
        stype = stale_types.get(str(sid))
        mk = stale_markers.get(str(sid)) or {}
        g = US.group_key(surl, stype, entity_kind=mk.get("entity_kind"),
                         qid=mk.get("qid"))
        if stype not in kept_types.get(g, set()):
            dropped += 1
            continue
        out.append({"id": str(sid), "url": surl, "reason": "superseded"})
    if dropped:
        log(f"[supersede-guard] dropped {dropped} stale match(es) whose own "
           "source_type has no matching kept row at that folded url "
           "(cross-family url-identity collision, not a real supersede)")
    return out


def served_digests_by_url(client, collection: str, urls: list[str], *,
                          group_batch: int = 64) -> dict[str, set]:
    """For every url in `urls`, the set of `content_digest`s CURRENTLY
    served in `collection` at that url (batched url-variant scroll, the
    SAME shape `Superseder._scan_stale()` uses -- never a per-url round
    trip). A url with nothing served yet is simply absent from the
    returned dict (an empty/missing set, never confused with "served but
    identical"). Each digest is its CANONICAL key (`row_decode`: the
    32-hex prefix the coded twin stores), so a raw twin and a coded one
    give the same set (task 855)."""
    decode = RD.reader(collection)
    out: dict[str, set] = {}
    for url, payloads in US.served_payloads_by_url(
            client, collection, urls, ("content_digest",),
            group_batch=group_batch).items():
        out[url] = {decode(pl).get(RD.DIGEST_KEY) for pl in payloads}
    return out


def filter_unchanged_recaptures(
        kept: list[dict], client, collection: str, *, now_ref: _dt.date,
        overhang_days: int = NSC.DEFAULT_NEWS_RECAPTURE_OVERHANG_DAYS,
        log=print) -> dict:
    """Tim's ruling, 2026-09-23: "we probably don't want to be continuously
    re-collecting previously ingested news articles... maybe a 1 week
    overhang is useful". Applied here as: a url NOT yet served in prod is
    always kept (a backfill catching up on a genuinely missing article is
    not a "re-collection"). A url ALREADY served is kept only if its
    `published` date is within `overhang_days` of `now_ref` -- AND, inside
    that window, only if its content actually changed (the set of
    `content_digest`s differs from what is served; the coordinator's
    earlier finding, 2026-09-23: most re-captures -- 11,073 of 16,019 in
    the real backlog -- are byte-identical re-crawls of unchanged articles).
    A served url published OLDER than the overhang is skipped
    UNCONDITIONALLY, changed or not -- that is the "stop re-collecting"
    rule, distinct from "unchanged, skip".

    Excluding a url from the returned `kept` means `compute_superseded_
    tombstones()` never `observe()`s it, so its served chunks are never
    scanned, let alone tombstoned (`Superseder._scan_stale()`: an
    unobserved group is never touched) -- "no tombstone, no re-write" for
    every skip reason here, not only the unchanged-content one.

    Returns {"kept": [...], "unchanged_skipped_urls": [...],
            "overhang_skipped_urls": [...], "changed_recaptured_urls": [...]}."""
    by_url: dict[str, list[dict]] = collections.defaultdict(list)
    for row in kept:
        by_url[(row.get("payload") or {}).get("url", "")].append(row)

    served = served_digests_by_url(client, collection, list(by_url))

    out_kept: list[dict] = []
    unchanged: list[str] = []
    overhang_skipped: list[str] = []
    changed_recaptured: list[str] = []
    for url, rows in by_url.items():
        served_set = served.get(url)
        if not served_set:  # never served -- always admit, no age limit
            out_kept.extend(rows)
            continue

        published = (rows[0].get("payload") or {}).get("published")
        age_days = None
        if published:
            try:
                age_days = (now_ref - _dt.date.fromisoformat(published)).days
            except ValueError:
                age_days = None
        if age_days is not None and age_days > overhang_days:
            overhang_skipped.append(url)
            continue

        # the pack's 64-hex digests in the served set's canonical (32-hex) key (task 855)
        new_set = {RD.digest_key((r.get("payload") or {}).get("content_digest")) for r in rows}  # canonical row
        if new_set == served_set:
            unchanged.append(url)
            continue
        out_kept.extend(rows)
        changed_recaptured.append(url)

    if unchanged:
        log(f"[unchanged-recapture] {len(unchanged)} url(s) re-captured with "
           "IDENTICAL content already served -- skipped, no tombstone, no "
           "re-write")
    if overhang_skipped:
        log(f"[recapture-overhang] {len(overhang_skipped)} url(s) already "
           f"served and published >{overhang_days}d ago -- skipped "
           "unconditionally, changed or not")
    return {"kept": out_kept, "unchanged_skipped_urls": unchanged,
           "overhang_skipped_urls": overhang_skipped,
           "changed_recaptured_urls": changed_recaptured}


# NOTE (task 659x, 2026-09-25): a compose-time `apply_word_coverage_floor()`
# lived here briefly, after news-daily-20260924 was REFUSED whole (2 of 40
# superseded urls below promote_queue's 50% word-coverage floor). Removed
# once task 660 shipped the canonical fix IN THE DRIVER
# (`promote_queue.py`'s `content_pack_dry_check` / `COVERAGE_VIOLATION_
# SHARE_MAX`, PR #356): per-url skip with a 20%-of-superseded-urls circuit
# breaker that still refuses the WHOLE pack when violations look systemic,
# not a handful of thin pages. A compose-time pre-filter here would have
# silently defeated that circuit breaker (a genuinely broken extraction run
# would never reach the >20% threshold if this module quietly stripped the
# bad urls first) -- one door, one check, and the driver is it.
def compute_superseded_tombstones(client, collection: str, kept: list[dict],
                                  *, log=print) -> list[dict]:
    """`Superseder._scan_stale()` -- the ONE existing selector, called
    exactly as `run()`/`export_stale()` call it, never re-implemented. Read
    only: this never deletes, it only proves what WOULD be deleted. Then
    `filter_cross_family_stale()` (see its docstring) drops a stale match
    that only collided on url identity, never on what it actually is."""
    sup = US.Superseder(client, collection, log=log, dry_run=True)
    for row in kept:
        payload = row.get("payload") or {}
        sup.observe_payload(row["id"], payload)
    stale_ids, stale_urls, per_domain = sup._scan_stale()  # noqa: SLF001
    if not stale_ids:
        return []
    stale_points = client.retrieve(collection, ids=stale_ids,
                                   with_payload=["source_type", "entity_kind", "qid"],
                                   with_vectors=False)
    stale_types = {str(p.id): (p.payload or {}).get("source_type")
                  for p in stale_points}
    stale_markers = {str(p.id): {"entity_kind": (p.payload or {}).get("entity_kind"),
                                 "qid": (p.payload or {}).get("qid")}
                     for p in stale_points}
    return filter_cross_family_stale(stale_ids, stale_urls, stale_types,
                                     kept, stale_markers=stale_markers, log=log)


def scroll_news_points(client, collection: str, *, page: int = 1000) -> list[dict]:
    """A plain, read-only scroll of every point this collection holds that
    `news_window_filter.is_news()` recognises -- the data `_partition()`
    needs, sourced directly rather than through the fuller
    `dump_news_points.py` production pipeline (which also tracks successor
    candidates this one-off census does not need)."""
    out: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection, limit=page, offset=offset, with_payload=True,
            with_vectors=False)
        for p in points:
            payload = p.payload or {}
            if NWF.is_news(payload):
                out.append({"id": str(p.id), "payload": payload})
        if offset is None:
            break
    return out


def reverify_aged_out(client, collection: str, aged: list[dict],
                      cutoff) -> dict:
    """Final guard immediately before a candidate is allowed to claim
    `aged_out`: `aged` was classified from a scroll snapshot that can be
    30+ minutes old (module docstring) -- long enough for an unrelated
    write under the SAME deterministic point id (e.g. an L1 backfill
    re-fetch, task 669) to have replaced the row's payload since the
    scroll read it, including dropping its `published` date entirely
    (real incident: news-aged-out-weekly-2026W39,
    `af49e590-6fa4-5c49-b59f-fe330331ac9c` -- scrolled as aged news, later
    re-verified as a `reference`-family row with no `published`).

    Re-reads the CURRENT payload for every candidate id and only
    tombstones a row whose FRESH `published` is still confirmably older
    than `cutoff`. "A record whose age cannot be proven must NOT be
    deleted" (growth-lane ruling, task 669/A): a row that still EXISTS but
    whose fresh age cannot be proven is held back and counted as
    `age_unknown_kept` -- never silently dropped, never tombstoned on a
    stale read. A row that has gone missing entirely between the scroll
    and this re-check is let through as a tombstone anyway: deleting an
    already-gone id is idempotent and `promote_queue.validate_tombstones`
    already treats "already gone" as the goal state, not a refusal (task
    664, 2026-09-28) -- re-litigating that here would only reintroduce the
    same problem this function exists to close, one level up.
    """
    if not aged:
        return {"tombstones": [], "age_unknown_kept": []}
    ids = [r["id"] for r in aged]
    fresh = {str(p.id): (p.payload or {})
            for p in client.retrieve(collection, ids=ids,
                                     with_payload=["published"],
                                     with_vectors=False)}
    tombstones: list[dict] = []
    age_unknown_kept: list[dict] = []
    for r in aged:
        pid = r["id"]
        url = r["payload"].get("url", "")
        current = fresh.get(pid)
        if current is None:
            tombstones.append({"id": pid, "url": url, "reason": "aged_out"})
            continue
        published = current.get("published")
        if NWF.in_window(published, cutoff) is False:
            tombstones.append({"id": pid, "url": url, "reason": "aged_out"})
        else:
            age_unknown_kept.append({"id": pid, "url": url,
                                     "published": published})
    return {"tombstones": tombstones, "age_unknown_kept": age_unknown_kept}


def compute_aged_out_tombstones(client, collection: str, *, now_ref,
                                months: int = 36) -> dict:
    """`news_window_filter.window_cutoff()`/`._partition()` -- the EXISTING
    purge-horizon logic (Tim's 2026-08-11 ruling, default 36 months), never
    re-derived. Scoped to news rows only (`is_news()` inside the scroll and
    again inside `_partition()`). Returns `{"tombstones", "age_unknown_kept"}`
    -- see `reverify_aged_out()` for why a second, fresh check runs right
    before a row is allowed to claim `aged_out`."""
    records = scroll_news_points(client, collection)
    cutoff = NWF.window_cutoff(now_ref, months=months)
    _keep, aged, _unknown, _nonnews = NWF._partition(records, cutoff, None)  # noqa: SLF001
    return reverify_aged_out(client, collection, aged, cutoff)


# ------------------------------------------------------- cross-pack tombstones
def other_queued_tombstone_ids(queue_dir: str, own_pack_id: str) -> set[str]:
    """Coordinator finding, 2026-09-23: `compute_aged_out_tombstones()` reads
    the twin's CURRENT state, so three packs composed back-to-back (none
    yet applied) each independently compute the SAME aged-out set --
    task640-news-backlog, news-daily-20260919 and news-daily-20260922 all
    carried the identical 4,164 ids. Applied in sequence, pack 2 would try
    to delete ids pack 1 already deleted -- the driver either refuses
    (id no longer exists) or, worse, silently no-ops a delete meant for a
    DIFFERENT id that happens to collide.

    Every OTHER pack directory in `queue_dir` (sibling of `own_pack_id`,
    skipping any without a READY marker -- a held/broken pack's tombstones
    are not "claimed") contributes its tombstone ids to the returned set.
    "Only the backlog pack carries the current aged_out set; later packs
    carry only rows that newly crossed the line" (coordinator ruling) is
    implemented by callers SUBTRACTING this set from what they compute,
    not by this function -- it only answers "what has some other queued
    pack already claimed"."""
    claimed: set[str] = set()
    base = pathlib.Path(queue_dir)
    if not base.is_dir():
        return claimed
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or entry.name == own_pack_id:
            continue
        if not (entry / "READY").is_file():
            continue
        if (entry / "SUPERSEDED").is_file():   # retired, never landed: claims nothing (task 974)
            continue
        tomb_path = entry / "tombstones.jsonl"
        if not tomb_path.is_file():
            continue
        for line in tomb_path.read_text().splitlines():
            if not line.strip():
                continue
            claimed.add(json.loads(line)["id"])
    return claimed


def dedupe_tombstones_against_queue(
        tombstones: list[dict], already_claimed: set[str], *,
        log=print) -> list[dict]:
    """Drop any tombstone (any reason) whose id another currently-queued
    pack already carries -- "each id appears in exactly one queued pack"
    (coordinator ruling). Pure given `already_claimed`; the network read
    lives in `other_queued_tombstone_ids()`."""
    out = [t for t in tombstones if t["id"] not in already_claimed]
    dropped = len(tombstones) - len(out)
    if dropped:
        log(f"[cross-pack-dedup] dropped {dropped} tombstone(s) already "
           "claimed by another currently-queued pack")
    return out


def assert_no_cross_pack_tombstone_collision(
        tombstones: list[dict], already_claimed: set[str]) -> None:
    """The compose-time guard the coordinator asked for: after dedup, NO id
    in this pack's tombstones may also sit in another queued pack's. This
    re-checks independently of `dedupe_tombstones_against_queue()`'s own
    bookkeeping (same discipline as `verify_capture_integrity()`) -- a
    future caller that forgets to dedup first still gets refused here,
    never a silent double-claim."""
    collide = {t["id"] for t in tombstones} & already_claimed
    if collide:
        raise SystemExit(
            f"REFUSING compose: {len(collide)} tombstone id(s) also sit in "
            f"another currently-queued pack's tombstones -- each id must "
            f"appear in exactly one queued pack. First 5: "
            f"{sorted(collide)[:5]}")


# ------------------------------------------------------------------- write
def drop_orphan_superseded_tombstones(tombstones: list[dict], kept: list[dict],
                                      *, log=print) -> tuple[list[dict], list[dict]]:
    """Pure: split off every `superseded` tombstone whose url is not, under
    `document_url_key`, the url of a row this pack keeps (a TRUE orphan).

    The selector (`Superseder._scan_stale`) matches on the canonical url
    (scheme/www folded), so a stored `https://swissinfo.ch/..` row is
    "superseded" by a captured `https://www.swissinfo.ch/..` row. The drain
    (`promote_queue.validate_tombstones`) now applies the same key, so that
    tombstone is KEPT and the old row is deleted (task 698, news-daily-
    20261002: 50 of 89 tombstones refused the whole pack when the drain
    compared exact strings). A tombstone whose url matches no kept row
    under either spelling is an orphan the drain still refuses; compose
    drops it itself and COUNTS it, instead of sealing a pack the drain will
    refuse. Returns (kept_tombstones, dropped_orphans)."""
    kept_keys = {k for k in (document_url_key((r.get("payload") or {}).get("url"))
                             for r in kept) if k}
    ok: list[dict] = []
    orphans: list[dict] = []
    for t in tombstones:
        if t.get("reason") == "superseded" and (
                document_url_key(t.get("url")) not in kept_keys):
            orphans.append(t)
        else:
            ok.append(t)
    if orphans:
        log(f"[orphan-tombstone-guard] dropped {len(orphans)} superseded "
            "tombstone(s) whose url matches no kept row's url, even with "
            "scheme/www folded (the drain refuses the whole pack on one). e.g. "
            f"{sorted({t.get('url') for t in orphans})[:3]}", file=sys.stderr)
    return ok, orphans


NEWS_DAILY_FAMILY = "news_daily"   # the contracts-file row the daily arm's consent is declared in
CONSENT_QUARANTINE_SIDECAR = "consent-quarantine.jsonl"
QUARANTINE_KINDS = ("no_record", "would_loosen", "declines_search")
DEFAULT_MAX_QUARANTINE_SHARE = 0.5   # `AW_NEWS_CONSENT_MAX_QUARANTINE_SHARE` / `--consent-max-quarantine-share`


def _row_host(row: dict) -> str:
    from cytir.consent.derive import host_of  # noqa: PLC0415
    return host_of((row.get("payload") or {}).get("url"))


DIRECT_VARIANT = "direct"   # the cache-file suffix of a record written by our own direct fetch


def direct_fetch_context(ctx):
    """A copy of the engine context whose cache index holds ONLY the `direct` variant of
    each host. News is our own direct fetch (Tim 2026-10-05, ruling 26 Q2): the stamp reads
    the host's direct-fetch record and never a Common Crawl (`cc`) record, which describes
    a different route and can be older and stricter (SRG hosts keep cc-mode records of
    2026-08-22 with `search_ok=false`). When the direct record is not usable (status
    `unknown`, held), the host has NO record: the rows are left out as `no_record`, not
    stamped from the cc record. The resolver says the same: a direct record is never served
    to a CC-provenance gate, and the other way round."""
    import copy  # noqa: PLC0415
    out = copy.copy(ctx)
    out.cache_index = {h: ({DIRECT_VARIANT: v[DIRECT_VARIANT]}
                           if isinstance(v, dict) and DIRECT_VARIANT in v else {})
                       if isinstance(v, dict) else v
                       for h, v in ctx.cache_index.items()}
    return out


STAGE_CACHE_DIRNAME = "consent-cache"   # the news stage script: CONSENT_CACHE="${AW_CONSENT_CACHE:-$WORK/consent-cache}"


def stage_consent_cache_dir(work_dir, env=None) -> str:
    """THE one place that names the cache the news stage writes its fresh `direct` records
    to: `AW_CONSENT_CACHE` if set, else `<news work dir>/consent-cache`. Same rule as
    the news stage script (a test reads the script and fails on drift). The
    engine's default cache (`AW_CONSENT_CACHE_DIR`, a different and older store) is NOT
    the news stage's cache and must not stamp news rows."""
    env = os.environ if env is None else env
    return str(env.get("AW_CONSENT_CACHE") or (pathlib.Path(str(work_dir)) / STAGE_CACHE_DIRNAME))


def stage_stamp_context(work_dir, env=None):
    """`(ctx, cache_dir)` from the stage's cache. Refuses (`ConsentStampRefused`) when the
    directory does not exist or holds no host record at all: an empty cache would read as
    `no_record` for every row, which is a refusal of the day and not a quarantine."""
    from cytir.consent import stamp as CS  # noqa: PLC0415
    cache_dir = stage_consent_cache_dir(work_dir, env)
    ctx = CS.load_context(cache_dir=cache_dir)       # refuses when the directory is missing
    if not [h for h in ctx.cache_index if h != "__unparseable__"]:
        raise CS.ConsentStampRefused(
            f"the news stage consent cache {cache_dir!r} holds no host record: refusing the "
            "day (every row would read as no_record; an error is not an empty)")
    return ctx, cache_dir


def stamp_consent(kept: list[dict], *, family: str = NEWS_DAILY_FAMILY, ctx=None,
                  max_quarantine_share: float = DEFAULT_MAX_QUARANTINE_SHARE,
                  log=print) -> tuple[list[dict], list[dict], dict]:
    """ADR-115 am.1, stamp at compose: the daily pack's rows carry the host's CURRENT
    consent-cache verdict and its own `checked_at`, derived by the ONE engine
    (`cytir.consent.derive`, through `cytir.consent.stamp`, the same call the growth composer
    makes) with the family the contracts file declares (the manifest says `family: "news"`;
    the contracts row is `news_daily`, named here). The stage's own older stamp
    (`robots-live-resolve`, dated when the cache record was written) is replaced; only
    `search_ok`, `train_ok`, `consent_basis`, `consent_checked_at` change.

    QUARANTINE, not refusal: a row the engine cannot stamp, or whose restamp would turn a
    stage `false` flag into `true` (Tim 2026-10-05, ruling 26 Q1; see
    `cytir.consent.stamp.stamp_payload_rows`), is left OUT of the pack. Consent is per host,
    so the unit is the URL: every row of a quarantined URL is left out. Call this BEFORE
    the superseded tombstones are computed: a quarantined URL then gets neither new rows
    nor a tombstone on its old rows (the served rows stay as they are), so the pack carries
    no orphan tombstone. It changes no id, vector, url or digest of the rows that stay.

    Returns (stamped rows, quarantined entries, report). Raises `ConsentStampRefused` when
    the cache is missing, or when the quarantined share of the day's rows is above
    `max_quarantine_share` (a day that is mostly unstampable is not a pack worth landing)."""
    from cytir.consent import stamp as CS  # noqa: PLC0415 -- the engine reads the contracts + cache dir
    if ctx is None:
        ctx = CS.load_context()
    ctx = direct_fetch_context(ctx)
    stamped, quarantined, report = CS.stamp_payload_rows(
        kept, ctx, family=family, replace_existing=True)
    bad = {q["url"]: q for q in reversed(quarantined)}
    out = []
    for r in stamped:
        url = str((r.get("payload") or {}).get("url"))
        if url in bad:
            sib = bad[url]
            quarantined.append({"id": r.get("id"), "url": url, "host": sib["host"],
                                "reason": f"same-url-as-quarantined-row ({sib['reason']})",
                                "kind": sib["kind"]})
        else:
            out.append(r)
    per_host: dict = {}
    for r in out:
        per_host.setdefault(_row_host(r), collections.Counter())["stamped"] += 1
    for q in quarantined:
        per_host.setdefault(q["host"], collections.Counter())["quarantined_" + q["kind"]] += 1
    counts = {"stamped": len(out),
              **{f"quarantined_{k}": sum(1 for q in quarantined if q["kind"] == k)
                 for k in QUARANTINE_KINDS},
              "per_host": {h: {"stamped": c["stamped"],
                               **{f"quarantined_{k}": c[f"quarantined_{k}"] for k in QUARANTINE_KINDS}}
                           for h, c in sorted(per_host.items())},
              "rules": report["rules"], "flags_by_host": report["flags_by_host"],
              "fields_changed": report["fields_changed"]}
    total = len(kept)
    share = (len(quarantined) / total) if total else 0.0
    report.update({"consent_stamp": counts, "quarantine_share": round(share, 4),
                   "max_quarantine_share": max_quarantine_share})
    log(f"[consent-stamp] {family}: rows={total} stamped={counts['stamped']} "
        f"quarantined_no_record={counts['quarantined_no_record']} "
        f"quarantined_would_loosen={counts['quarantined_would_loosen']} "
        f"quarantined_declines_search={counts['quarantined_declines_search']} "
        f"share={share:.1%} (max {max_quarantine_share:.0%}) per_host={counts['per_host']}")
    if share > max_quarantine_share:
        from cytir.consent.stamp import ConsentStampRefused  # noqa: PLC0415
        raise ConsentStampRefused(
            f"{family}: {len(quarantined):,} of {total:,} row(s) ({share:.1%}) cannot be stamped "
            f"from the host's current consent verdict, above the {max_quarantine_share:.0%} "
            f"bound: the day is refused, nothing written. {report['quarantine_samples']}")
    return out, quarantined, report


def write_consent_quarantine_sidecar(out_dir: str, quarantined: list[dict]) -> str | None:
    """`consent-quarantine.jsonl` (id, url, host, reason, kind) next to the pack. Write it
    BEFORE `write_pack`: the pack's READY marker is written last. No file when nothing was
    quarantined."""
    out = pathlib.Path(out_dir)
    path = out / CONSENT_QUARANTINE_SIDECAR
    if not quarantined:
        path.unlink(missing_ok=True)       # a re-run of the day must not keep a stale sidecar
        return None
    out.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in quarantined))
    return str(path)


def declared_content_words(kept: list[dict], tombstones: list[dict]) -> dict[str, int]:
    """Task 984 (RULING 43): `{document_url_key: words}` -- the words of the CHUNK text this
    pack writes for each url that has a `superseded` tombstone, summed from the rows'
    `payload.chunk_n_words` (the exact word count of the chunk the materializer cut). The drain's
    word-coverage check reads it when the chunk policy changed, because the 25-word snippet
    cannot measure a re-cut. A url is declared ONLY when every kept row of it carries a
    non-negative int `chunk_n_words`; otherwise it is left out (the check then keeps the snippet
    rule for it) -- a partial sum would understate the text and refuse a lossless pack."""
    wanted = {document_url_key(t.get("url")) or t.get("url")
              for t in tombstones if t.get("reason") == "superseded"}
    words: dict[str, int] = {}
    broken: set[str] = set()
    for row in kept:
        payload = row.get("payload") or {}
        raw = payload.get("url")
        key = (document_url_key(raw) or raw) if raw else None
        if key not in wanted:
            continue
        n = payload.get("chunk_n_words")
        if isinstance(n, int) and not isinstance(n, bool) and n >= 0:
            words[key] = words.get(key, 0) + n
        else:
            broken.add(key)
    return {k: v for k, v in sorted(words.items()) if k not in broken}


def _without_chunk_n_words(row: dict) -> dict:
    payload = row.get("payload") or {}
    if "chunk_n_words" not in payload:
        return row
    return {**row, "payload": {k: v for k, v in payload.items() if k != "chunk_n_words"}}


def write_pack(out_dir: str, *, kept: list[dict], tombstones: list[dict],
              pack_id: str, family: str, collection: str, days: list[str],
              extra_manifest: dict | None = None) -> dict:
    tombstones, orphans = drop_orphan_superseded_tombstones(tombstones, kept)
    # Task 710 guard 1, stage side: German rows must carry their parts BEFORE any file is written.
    # `warn` (default) records the count; `refuse` raises with no manifest and no READY on disk.
    german_parts, gp_reason = GPG.guarded(GPG.check_points, pack_id, kept)
    if gp_reason:
        raise GPG.GermanPartsRefused(f"{pack_id}: {gp_reason}")
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Task 984: the staging step carries each chunk's exact word count in `payload.chunk_n_words`.
    # It becomes the manifest's `content_words_by_url` and is NOT part of the pack row (ADR-039:
    # the pack row keeps its minimal payload keys).
    content_words = declared_content_words(kept, tombstones)
    kept = [_without_chunk_n_words(r) for r in kept]

    points_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        for row in kept)
    points_sha = hashlib.sha256(points_bytes).hexdigest()
    points_path = out / f"news-points.{points_sha}.jsonl.gz"
    with gzip.open(points_path, "wb") as f:
        f.write(points_bytes)

    tomb_path = out / "tombstones.jsonl"
    tomb_bytes = b"".join(
        (json.dumps(t, ensure_ascii=False) + "\n").encode("utf-8")
        for t in tombstones)
    tomb_path.write_bytes(tomb_bytes)
    tomb_sha = hashlib.sha256(tomb_bytes).hexdigest()

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "kind": "content_pack",
        "pack_id": pack_id,
        "family": family,
        "collection": collection,
        "pack_points": str(points_path),
        "pack_points_sha256": points_sha,
        "rows": len(kept),
        "days_covered": sorted(days),
        "tombstones": str(tomb_path),
        "tombstones_sha256": tomb_sha,
        "tombstones_count": len(tombstones),
        "tombstones_by_reason": {
            "superseded": sum(1 for t in tombstones if t["reason"] == "superseded"),
            "aged_out": sum(1 for t in tombstones if t["reason"] == "aged_out")},
        "orphan_tombstones_dropped": len(orphans),
        "orphan_tombstone_urls_dropped": sorted({t.get("url") for t in orphans}),
        "composed_at": now(),
        "german_parts": german_parts,
    }
    if content_words:
        manifest["content_words_by_url"] = content_words
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest_path = out / "manifest.json"
    manifest_bytes = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    (out / "READY").write_text(manifest_sha + "\n")
    return manifest


# ----------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack-dir", action="append", required=True,
                    help="a prepared day pack directory; repeatable")
    ap.add_argument("--out-dir", required=True,
                    help="promotion-queue/<pack-id> directory to write into")
    ap.add_argument("--pack-id", required=True)
    ap.add_argument("--family", default="news")
    ap.add_argument("--collection", default=TWIN.collection())
    ap.add_argument("--twin-url", default=TWIN.url())
    ap.add_argument("--retain-months", type=int, default=36)
    ap.add_argument("--now", default=None, help="reference today YYYY-MM-DD")
    ap.add_argument("--no-tombstones", action="store_true",
                    help="skip the live twin reconciliation (for a dry "
                         "merge-only check)")
    ap.add_argument("--recapture-overhang-days", type=int,
                    default=NSC.DEFAULT_NEWS_RECAPTURE_OVERHANG_DAYS,
                    help="an already-served url is re-packed only if its "
                         "published date is within this many days of --now "
                         "(Tim, 2026-09-23)")
    args = ap.parse_args(argv)

    all_rows: list[dict] = []
    days: list[str] = []
    for pd in args.pack_dir:
        rows = read_pack_points(pd)
        all_rows.extend(rows)
        days.append(pathlib.Path(pd).name)
        print(f"[read] {pd}: {len(rows)} rows", file=sys.stderr)

    merged = merge_rows(all_rows)
    dated_input = [r for r in all_rows if (r.get("payload") or {}).get("published")]
    per_url_stats = verify_capture_integrity(
        kept=merged["kept"], dated_input=dated_input,
        log=lambda msg: print(msg, file=sys.stderr))
    kept_urls = {(r.get("payload") or {}).get("url", "") for r in merged["kept"]}
    print(f"[merge] in={len(all_rows)} kept={len(merged['kept'])} "
         f"urls={len(kept_urls)} "
         f"internally_superseded={len(merged['superseded_internal'])} "
         f"undated_dropped={len(merged['undated'])}", file=sys.stderr)

    tombstones: list[dict] = []
    age_unknown_kept: list[dict] = []
    unchanged_result = {"unchanged_skipped_urls": [], "overhang_skipped_urls": [],
                        "changed_recaptured_urls": []}
    now_ref = (_dt.date.fromisoformat(args.now) if args.now
              else _dt.datetime.utcnow().date())
    if not args.no_tombstones:
        client = qdrant_client(args.twin_url, timeout=120)

        assert_dense_matches_collection(merged["kept"], client=client,
                                        collection=args.collection,
                                        log=lambda msg: print(msg, file=sys.stderr))

        unchanged_result = filter_unchanged_recaptures(
            merged["kept"], client, args.collection, now_ref=now_ref,
            overhang_days=args.recapture_overhang_days,
            log=lambda msg: print(msg, file=sys.stderr))
        merged["kept"] = unchanged_result["kept"]
        skipped_urls = (set(unchanged_result["unchanged_skipped_urls"]) |
                        set(unchanged_result["overhang_skipped_urls"]))
        per_url_stats = {u: s for u, s in per_url_stats.items()
                        if u not in skipped_urls}
        kept_urls = {(r.get("payload") or {}).get("url", "") for r in merged["kept"]}
        print(f"[recapture] unchanged_skipped={len(unchanged_result['unchanged_skipped_urls'])} "
             f"overhang_skipped={len(unchanged_result['overhang_skipped_urls'])} "
             f"changed_recaptured={len(unchanged_result['changed_recaptured_urls'])} "
             f"kept_after={len(merged['kept'])} urls_after={len(kept_urls)}",
             file=sys.stderr)

        tombstones = compute_superseded_tombstones(
            client, args.collection, merged["kept"])
        aged_out_result = compute_aged_out_tombstones(
            client, args.collection, now_ref=now_ref, months=args.retain_months)
        tombstones += aged_out_result["tombstones"]
        age_unknown_kept = aged_out_result["age_unknown_kept"]
        print(f"[tombstones] (pre-cross-pack-dedup) superseded="
             f"{sum(1 for t in tombstones if t['reason']=='superseded')} "
             f"aged_out={sum(1 for t in tombstones if t['reason']=='aged_out')} "
             f"age_unknown_kept={len(age_unknown_kept)}",
             file=sys.stderr)

        own_pack_id = pathlib.Path(args.out_dir).name
        queue_dir = pathlib.Path(args.out_dir).parent
        already_claimed = other_queued_tombstone_ids(str(queue_dir), own_pack_id)
        tombstones = dedupe_tombstones_against_queue(
            tombstones, already_claimed, log=lambda msg: print(msg, file=sys.stderr))
        assert_no_cross_pack_tombstone_collision(tombstones, already_claimed)
        print(f"[tombstones] (final) superseded="
             f"{sum(1 for t in tombstones if t['reason']=='superseded')} "
             f"aged_out={sum(1 for t in tombstones if t['reason']=='aged_out')}",
             file=sys.stderr)

    manifest = write_pack(
        args.out_dir, kept=merged["kept"], tombstones=tombstones,
        pack_id=args.pack_id, family=args.family, collection=args.collection,
        days=days,
        extra_manifest={
            "undated_dropped": len(merged["undated"]),
            "internally_superseded": len(merged["superseded_internal"]),
            "urls": len(kept_urls),
            "per_url_chunk_and_word_totals": per_url_stats,
            "unchanged_recaptures_skipped":
                len(unchanged_result["unchanged_skipped_urls"]),
            "overhang_recaptures_skipped":
                len(unchanged_result["overhang_skipped_urls"]),
            "changed_recaptures": len(unchanged_result["changed_recaptured_urls"]),
            "recapture_overhang_days": args.recapture_overhang_days,
            "age_unknown_kept": len(age_unknown_kept),
            "age_unknown_kept_rows": age_unknown_kept,
        })
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
