"""Task 640: the news backlog composer's pure merge + write logic.

Real records, not synthetic shapes: these mirror an actual
`news-points.*.jsonl.gz` row exactly (id/payload/vector, payload fields
pulled from a real pack sampled on the build box 2026-09-23 --
pack-20260831-task598), so a schema drift in the composer or in the pack
shape it reads would fail these, not just a hand-typed fixture.
"""
from __future__ import annotations

import datetime as _dt
import gzip
import json
import pathlib
from types import SimpleNamespace

from cytir.core.canon import document_url_key
import cytir.news.compose as C
import cytir.index.supersede as US


def row(id_, url, published, last_seen, domain="rts.ch", extra=None, bm25_terms=1):
    payload = {"url": url, "published": published, "last_seen": last_seen,
              "domain": domain, "disc_class": "news", "source_type": "reference",
              "point_id": id_, "title": f"title for {url}"}
    if extra:
        payload.update(extra)
    indices = list(range(bm25_terms))
    return {"id": id_, "payload": payload,
           "vector": {"bm25": {"indices": indices, "values": [1.0] * bm25_terms}}}


def test_a_later_capture_of_the_same_url_supersedes_an_earlier_one():
    older = row("id-a-2026-08-31", "https://www.rts.ch/x", "2026-08-30",
               "2026-08-31")
    newer = row("id-a-2026-09-05", "https://www.rts.ch/x", "2026-08-30",
               "2026-09-05")
    merged = C.merge_rows([older, newer])
    assert [r["id"] for r in merged["kept"]] == ["id-a-2026-09-05"]
    assert [r["id"] for r in merged["superseded_internal"]] == ["id-a-2026-08-31"]
    assert merged["undated"] == []


def test_supersede_is_the_same_rule_across_day_boundaries_and_within_one_day():
    same_day_a = row("id-x1", "https://www.srf.ch/y", "2026-09-01", "2026-09-01T08:00:00Z")
    same_day_b = row("id-x2", "https://www.srf.ch/y", "2026-09-01", "2026-09-01T18:00:00Z")
    cross_day = row("id-x3", "https://www.srf.ch/y", "2026-09-01", "2026-09-03T09:00:00Z")
    merged = C.merge_rows([same_day_a, same_day_b, cross_day])
    assert [r["id"] for r in merged["kept"]] == ["id-x3"]
    assert {r["id"] for r in merged["superseded_internal"]} == {"id-x1", "id-x2"}


def test_a_row_with_no_published_date_is_dropped_and_counted_not_silently_kept():
    dateless = row("id-nodate", "https://www.rsi.ch/z", None, "2026-09-02")
    dated = row("id-dated", "https://www.rsi.ch/other", "2026-09-01", "2026-09-02")
    merged = C.merge_rows([dateless, dated])
    assert [r["id"] for r in merged["kept"]] == ["id-dated"]
    assert [r["id"] for r in merged["undated"]] == ["id-nodate"]


def test_distinct_urls_all_survive():
    a = row("id-a", "https://www.rts.ch/a", "2026-09-01", "2026-09-01")
    b = row("id-b", "https://www.rts.ch/b", "2026-09-01", "2026-09-01")
    merged = C.merge_rows([a, b])
    assert {r["id"] for r in merged["kept"]} == {"id-a", "id-b"}
    assert merged["superseded_internal"] == []


def test_read_pack_points_round_trips_a_real_shaped_gzip_pack(tmp_path):
    rows = [row("id-1", "https://www.rts.ch/one", "2026-09-01", "2026-09-01"),
           row("id-2", "https://www.rts.ch/two", "2026-09-01", "2026-09-01")]
    pack_dir = tmp_path / "pack-20260901-test"
    pack_dir.mkdir()
    art = pack_dir / "news-points.deadbeef.jsonl.gz"
    with gzip.open(art, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    got = C.read_pack_points(str(pack_dir))
    assert [g["id"] for g in got] == ["id-1", "id-2"]


def test_read_pack_points_picks_the_newest_artifact_by_mtime(tmp_path):
    import os
    import time

    pack_dir = tmp_path / "pack-20260901-test"
    pack_dir.mkdir()
    old = pack_dir / "news-points.old.jsonl.gz"
    new = pack_dir / "news-points.new.jsonl.gz"
    with gzip.open(old, "wt", encoding="utf-8") as f:
        f.write(json.dumps(row("id-old", "https://x/old", "2026-09-01", "2026-09-01")) + "\n")
    time.sleep(0.01)
    with gzip.open(new, "wt", encoding="utf-8") as f:
        f.write(json.dumps(row("id-new", "https://x/new", "2026-09-01", "2026-09-01")) + "\n")
    got = C.read_pack_points(str(pack_dir))
    assert [g["id"] for g in got] == ["id-new"]


def test_write_pack_produces_a_valid_ready_sha_and_manifest_schema(tmp_path):
    # The pack goes in a SUBDIR: a READY directly in tmp_path made pytest's shared basetemp
    # "queue-shaped" and every later build test under it was refused (task 1516).
    tmp_path = tmp_path / "pack"
    tmp_path.mkdir()
    kept = [row("id-1", "https://www.rts.ch/one", "2026-09-01", "2026-09-01")]
    tombstones = [{"id": "id-old", "url": "https://www.rts.ch/one",
                  "reason": "superseded"},
                 {"id": "id-stale", "url": "https://www.rts.ch/stale",
                  "reason": "aged_out"}]
    manifest = C.write_pack(
        str(tmp_path), kept=kept, tombstones=tombstones, pack_id="task640-test",
        family="news", collection="aw_e8", days=["pack-20260901-test"])
    assert manifest["orphan_tombstones_dropped"] == 0

    assert manifest["schema"] == C.MANIFEST_SCHEMA
    assert manifest["kind"] == "content_pack"
    assert manifest["rows"] == 1
    assert manifest["tombstones_count"] == 2
    assert manifest["tombstones_by_reason"] == {"superseded": 1, "aged_out": 1}

    manifest_path = pathlib.Path(tmp_path, "manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    import hashlib
    expected_sha = hashlib.sha256(manifest_bytes).hexdigest()
    ready = pathlib.Path(tmp_path, "READY").read_text().strip()
    assert ready == expected_sha

    points_path = pathlib.Path(manifest["pack_points"])
    with gzip.open(points_path, "rt", encoding="utf-8") as f:
        written = [json.loads(line) for line in f]
    assert [w["id"] for w in written] == ["id-1"]

    tomb_path = pathlib.Path(manifest["tombstones"])
    tomb_rows = [json.loads(line) for line in tomb_path.read_text().splitlines()]
    assert {t["reason"] for t in tomb_rows} == {"superseded", "aged_out"}


# ---------------------------------------------------------------------------
# the chunk-collapse bug (task 640, coordinator finding, 2026-09-23): the
# ORIGINAL merge_rows() grouped by url alone and kept exactly one row per
# url, discarding every SIBLING CHUNK of the winning capture as
# "superseded" -- 16,019 kept rows for 16,019 urls, when a source pack
# carries up to 58 chunks at one url. Real ids and per-chunk BM25 term
# counts below are sampled verbatim from the build box 2026-09-23,
# pack-20260803, url=swissinfo.ch/.../90395898 (13 real chunks, one capture).
# ---------------------------------------------------------------------------
REAL_CAPTURE_URL = ("https://www.swissinfo.ch/ger/ungewohnliche-schweiz/"
                    "pesche-k%c3%a4thl%c3%a4-und-s%c3%a4mu-kleiner-ratgeber-"
                    "f%c3%bcr-berndeutsche-rufnamen/90395898")
REAL_CAPTURE_PUBLISHED = "2026-07-28"
REAL_CAPTURE_LAST_SEEN = "2026-08-03"
REAL_CAPTURE_CHUNKS = [  # (id, bm25 term count) -- both real, sampled verbatim
    ("0001fcb8-438f-5d1f-8a46-74fa3633060a", 137),
    ("25d18adf-fbda-5f2e-ab88-f74279f9b162", 127),
    ("265a6b62-bd41-52ce-94e0-503f0ee017d1", 137),
    ("43b2e9e5-89b3-537f-981e-c901ac551d8a", 129),
    ("43fc4b92-e66e-5ee6-aded-0c7684727054", 124),
    ("4bb6f4b3-35d2-597c-857f-e8cf7e2691a0", 151),
    ("4bdb1028-a8d9-55c1-85f7-10fe9d9c92c6", 140),
    ("625c6416-ab62-52a5-a6c5-7a9ac2cecdbe", 114),
    ("6f3d5c90-ec78-53a6-add9-862521a8b68e", 142),
    ("750eac7b-5a0d-5d8d-a4ea-2ca6b987dc9f", 100),
    ("82c28357-2e7b-544c-82af-cc30c09f7466", 123),
    ("b3d1b18f-f65d-5dfe-bfa3-2f4a6169b939", 127),
    ("e3d41c67-ca61-5c3b-82e1-b696965a47f9", 133),
]


def _real_capture_rows():
    return [row(cid, REAL_CAPTURE_URL, REAL_CAPTURE_PUBLISHED,
               REAL_CAPTURE_LAST_SEEN, domain="swissinfo.ch", bm25_terms=n)
           for cid, n in REAL_CAPTURE_CHUNKS]


def test_all_13_real_chunks_of_one_capture_survive_not_just_one():
    merged = C.merge_rows(_real_capture_rows())
    assert len(merged["kept"]) == 13
    assert {r["id"] for r in merged["kept"]} == {cid for cid, _ in REAL_CAPTURE_CHUNKS}
    assert merged["superseded_internal"] == []


def test_an_older_capture_of_the_same_url_is_dropped_whole_not_chunk_mixed():
    """A genuinely later re-crawl (different last_seen) supersedes the
    ENTIRE older capture -- every one of its 13 chunks -- never a partial,
    chunk-by-chunk mix of old and new."""
    older = _real_capture_rows()
    newer = [row("new-chunk-1", REAL_CAPTURE_URL, REAL_CAPTURE_PUBLISHED,
                "2026-09-01", domain="swissinfo.ch", bm25_terms=140),
             row("new-chunk-2", REAL_CAPTURE_URL, REAL_CAPTURE_PUBLISHED,
                "2026-09-01", domain="swissinfo.ch", bm25_terms=120)]
    merged = C.merge_rows(older + newer)
    assert {r["id"] for r in merged["kept"]} == {"new-chunk-1", "new-chunk-2"}
    assert {r["id"] for r in merged["superseded_internal"]} == \
        {cid for cid, _ in REAL_CAPTURE_CHUNKS}


def test_capture_integrity_guard_passes_on_the_real_13_chunk_capture():
    rows = _real_capture_rows()
    merged = C.merge_rows(rows)
    per_url = C.verify_capture_integrity(kept=merged["kept"], dated_input=rows)
    assert per_url[REAL_CAPTURE_URL]["chunks"] == 13
    assert per_url[REAL_CAPTURE_URL]["words"] == sum(n for _, n in REAL_CAPTURE_CHUNKS)


def test_capture_integrity_guard_refuses_a_short_kept_set():
    """The guard that would have caught the original bug: if `kept` is
    missing sibling chunks of its own winning capture (simulating the
    pre-fix behaviour directly, bypassing merge_rows), refuse rather than
    silently write a truncated pack."""
    rows = _real_capture_rows()
    truncated_kept = rows[:1]  # only 1 of 13 real chunks -- the old bug's output shape
    try:
        C.verify_capture_integrity(kept=truncated_kept, dated_input=rows)
        assert False, "expected SystemExit on a truncated kept set"
    except SystemExit as exc:
        assert "REFUSING" in str(exc)
        assert "13" in str(exc) or "chunks" in str(exc)


# ---------------------------------------------------------------------------
# unchanged re-captures (coordinator, 2026-09-23): a re-captured url that is
# BYTE-IDENTICAL to what is already served must produce no tombstone and no
# re-write -- most of the 11,073 re-captured urls found in the real backlog
# are exactly this (a later day's crawl re-seeing an unchanged article).
# ---------------------------------------------------------------------------
class FakeTwin:
    """A minimal qdrant_client-shaped fake: one page of scroll results,
    ignores the filter (fixtures are small enough that this is exact)."""

    def __init__(self, served_points):
        self._points = served_points  # list of (id, payload dict)

    def scroll(self, name, scroll_filter=None, limit=1000, offset=None,
              with_payload=True, with_vectors=True):
        if offset is not None:
            return [], None
        out = [SimpleNamespace(id=pid, payload=dict(pl))
              for pid, pl in self._points]
        return out, "done"


def test_unchanged_recapture_is_skipped_no_tombstone_no_rewrite():
    url = "https://www.rts.ch/already-served"
    served = [("served-1", {"url": url, "content_digest": "aaa"}),
             ("served-2", {"url": url, "content_digest": "bbb"})]
    new_capture = [
        row("new-1", url, "2026-09-01", "2026-09-10",
           extra={"content_digest": "aaa"}),
        row("new-2", url, "2026-09-01", "2026-09-10",
           extra={"content_digest": "bbb"}),
    ]
    result = C.filter_unchanged_recaptures(
        new_capture, FakeTwin(served), "aw_e8", now_ref=DATE_WITHIN_OVERHANG)
    assert result["kept"] == []
    assert result["unchanged_skipped_urls"] == [url]
    assert result["changed_recaptured_urls"] == []


def test_changed_recapture_is_kept_and_reported():
    url = "https://www.rts.ch/actually-changed"
    served = [("served-1", {"url": url, "content_digest": "old-digest"})]
    new_capture = [row("new-1", url, "2026-09-01", "2026-09-10",
                       extra={"content_digest": "new-digest"})]
    result = C.filter_unchanged_recaptures(
        new_capture, FakeTwin(served), "aw_e8", now_ref=DATE_WITHIN_OVERHANG)
    assert [r["id"] for r in result["kept"]] == ["new-1"]
    assert result["changed_recaptured_urls"] == [url]
    assert result["unchanged_skipped_urls"] == []


def test_a_new_url_never_served_before_is_kept_not_treated_as_unchanged():
    """A url NOT yet in prod is always admitted, regardless of age -- a
    backfill catching up on a genuinely missing old article is not a
    "re-collection" (Tim's ruling only bounds RE-fetching, never first
    admission). `now_ref` here is far outside the overhang window on
    purpose, to prove age never gates a first-time url."""
    url = "https://www.rts.ch/brand-new"
    new_capture = [row("new-1", url, "2026-01-01", "2026-01-10",
                       extra={"content_digest": "x"})]
    result = C.filter_unchanged_recaptures(
        new_capture, FakeTwin([]), "aw_e8", now_ref=DATE_WITHIN_OVERHANG)
    assert [r["id"] for r in result["kept"]] == ["new-1"]
    assert result["unchanged_skipped_urls"] == []
    assert result["changed_recaptured_urls"] == []
    assert result["overhang_skipped_urls"] == []


# ---------------------------------------------------------------------------
# the recapture overhang (Tim's ruling, 2026-09-23 ~11:15Z): "we probably
# don't want to be continuously re-collecting previously ingested news
# articles... maybe a 1 week overhang". `DATE_WITHIN_OVERHANG` /
# `DATE_OUTSIDE_OVERHANG` are both exactly `published + N days` around the
# default 7-day boundary, matching `NSC.DEFAULT_NEWS_RECAPTURE_OVERHANG_DAYS`.
# ---------------------------------------------------------------------------
DATE_WITHIN_OVERHANG = _dt.date(2026, 9, 8)   # published 09-01 + 7d = boundary, inclusive
DATE_OUTSIDE_OVERHANG = _dt.date(2026, 9, 9)  # published 09-01 + 8d = past the boundary


def test_an_already_served_url_older_than_the_overhang_is_skipped_even_if_changed():
    """The rule Tim actually asked for, distinct from unchanged-skip: a
    served url published MORE than the overhang ago is skipped
    UNCONDITIONALLY -- changed or not."""
    url = "https://www.rts.ch/old-but-changed"
    served = [("served-1", {"url": url, "content_digest": "old-digest"})]
    new_capture = [row("new-1", url, "2026-09-01", "2026-09-10",
                       extra={"content_digest": "genuinely-new-digest"})]
    result = C.filter_unchanged_recaptures(
        new_capture, FakeTwin(served), "aw_e8", now_ref=DATE_OUTSIDE_OVERHANG)
    assert result["kept"] == []
    assert result["overhang_skipped_urls"] == [url]
    assert result["changed_recaptured_urls"] == []
    assert result["unchanged_skipped_urls"] == []


def test_an_already_served_url_at_the_overhang_boundary_is_still_admitted():
    url = "https://www.rts.ch/at-boundary"
    served = [("served-1", {"url": url, "content_digest": "old-digest"})]
    new_capture = [row("new-1", url, "2026-09-01", "2026-09-10",
                       extra={"content_digest": "changed-digest"})]
    result = C.filter_unchanged_recaptures(
        new_capture, FakeTwin(served), "aw_e8", now_ref=DATE_WITHIN_OVERHANG)
    assert [r["id"] for r in result["kept"]] == ["new-1"]
    assert result["changed_recaptured_urls"] == [url]
    assert result["overhang_skipped_urls"] == []


def test_recapture_overhang_days_is_a_config_value_not_hardcoded():
    """A caller can override the default (the ONE config value, per Tim's
    ruling) -- a 30-day overhang admits the same "old-but-changed" url the
    default 7-day overhang would have skipped."""
    url = "https://www.rts.ch/wide-overhang"
    served = [("served-1", {"url": url, "content_digest": "old-digest"})]
    new_capture = [row("new-1", url, "2026-09-01", "2026-09-10",
                       extra={"content_digest": "changed-digest"})]
    result = C.filter_unchanged_recaptures(
        new_capture, FakeTwin(served), "aw_e8", now_ref=DATE_OUTSIDE_OVERHANG,
        overhang_days=30)
    assert [r["id"] for r in result["kept"]] == ["new-1"]
    assert result["overhang_skipped_urls"] == []


def test_default_overhang_matches_the_shared_news_stage_contract_constant():
    import cytir.news.stage_contract as NSC
    assert NSC.DEFAULT_NEWS_RECAPTURE_OVERHANG_DAYS == 7
    import inspect
    assert (inspect.signature(C.filter_unchanged_recaptures)
           .parameters["overhang_days"].default
           == NSC.DEFAULT_NEWS_RECAPTURE_OVERHANG_DAYS)


# ---------------------------------------------------------------------------
# cross-pack tombstone collision (coordinator, 2026-09-23): three packs
# composed back-to-back before any of them applied each independently
# computed aged_out against the SAME twin state -- task640-news-backlog,
# news-daily-20260919 and news-daily-20260922 all carried the identical
# 4,164 ids. "Each tombstone id goes in exactly one queued pack."
# ---------------------------------------------------------------------------
def _write_queued_pack(base: pathlib.Path, pack_id: str, tombstone_ids: list[str],
                       ready: bool = True) -> None:
    d = base / pack_id
    d.mkdir(parents=True)
    tomb = [{"id": tid, "url": f"https://example/{tid}", "reason": "aged_out"}
           for tid in tombstone_ids]
    (d / "tombstones.jsonl").write_text(
        "\n".join(json.dumps(t) for t in tomb) + ("\n" if tomb else ""))
    (d / "manifest.json").write_text(json.dumps({"pack_id": pack_id}))
    if ready:
        (d / "READY").write_text("deadbeef\n")


def test_other_queued_tombstone_ids_collects_siblings_not_self(tmp_path):
    _write_queued_pack(tmp_path, "task640-news-backlog", ["id-a", "id-b"])
    _write_queued_pack(tmp_path, "news-daily-20260922", ["id-c"])
    claimed = C.other_queued_tombstone_ids(str(tmp_path), "news-daily-20260919")
    assert claimed == {"id-a", "id-b", "id-c"}


def test_other_queued_tombstone_ids_ignores_held_packs_without_ready(tmp_path):
    """A HELD pack (READY renamed away, per the orchestrator's own held
    convention) has not actually claimed anything -- its ids must remain
    available, not silently excluded forever."""
    _write_queued_pack(tmp_path, "held-pack", ["id-x"], ready=False)
    claimed = C.other_queued_tombstone_ids(str(tmp_path), "news-daily-20260919")
    assert claimed == set()


def test_dedupe_tombstones_against_queue_drops_only_claimed_ids():
    tombstones = [{"id": "id-a", "url": "u1", "reason": "aged_out"},
                 {"id": "id-b", "url": "u2", "reason": "aged_out"},
                 {"id": "id-c", "url": "u3", "reason": "superseded"}]
    out = C.dedupe_tombstones_against_queue(tombstones, {"id-a"})
    assert [t["id"] for t in out] == ["id-b", "id-c"]


def test_cross_pack_collision_guard_passes_after_dedup():
    tombstones = [{"id": "id-b", "url": "u2", "reason": "aged_out"}]
    C.assert_no_cross_pack_tombstone_collision(tombstones, {"id-a"})  # no raise


def test_cross_pack_collision_guard_refuses_an_undeduped_collision():
    """The real incident, reproduced directly: the SAME 4,164 aged_out ids
    in two packs must refuse, not silently apply twice."""
    tombstones = [{"id": "id-a", "url": "u1", "reason": "aged_out"}]
    try:
        C.assert_no_cross_pack_tombstone_collision(tombstones, {"id-a"})
        assert False, "expected SystemExit on a cross-pack collision"
    except SystemExit as exc:
        assert "REFUSING" in str(exc)
        assert "id-a" in str(exc)


# ---------------------------------------------------------------------------
# the danluu.com/hn-comments cross-family collision (task 640, orchestrator
# check, 2026-09-23): a `superseded` tombstone must never land for a url
# whose only "kept row" is a DIFFERENT source_type than the stale point.
# Real ids and urls from the actual incident, not shortened.
# ---------------------------------------------------------------------------
HN_REFERENCE_ROW = row(
    "5934679e-62cd-562f-93b1-07d240cfea4b", "https://danluu.com/hn-comments/",
    "2026-08-21", "2026-08-27", domain="danluu.com",
    extra={"source_type": "reference", "provenance": "hn-freshness",
          "disc_class": "news"})
STALE_BLOG_CHUNK_ID = "00c3747a-70ed-5544-ba35-1928fe8599eb"
STALE_BLOG_URL = "https://danluu.com/hn-comments"  # no trailing slash, as stored


def test_kept_source_types_by_group_folds_on_url_not_source_type():
    grouped = C.kept_source_types_by_group([HN_REFERENCE_ROW])
    g = US.group_key("https://danluu.com/hn-comments/", "reference")
    assert grouped[g] == {"reference"}
    # the same fold Superseder itself uses -- the no-slash stored form
    # resolves to the SAME group key
    assert g == US.group_key(STALE_BLOG_URL, "blog")


def test_a_cross_family_url_collision_never_becomes_a_superseded_tombstone():
    """The real incident: one hn-freshness `reference` row's url folds onto
    an unrelated `tech_blogs` `blog` article's url. None of the 96 real
    chunks may be tombstoned by it."""
    stale_ids = [STALE_BLOG_CHUNK_ID, "018c3256-87e9-514c-b4ac-6d704ec7d70c"]
    stale_urls = [STALE_BLOG_URL, STALE_BLOG_URL]
    stale_types = {sid: "blog" for sid in stale_ids}
    out = C.filter_cross_family_stale(stale_ids, stale_urls, stale_types,
                                      [HN_REFERENCE_ROW])
    assert out == []


def test_a_genuine_same_family_supersede_still_lands():
    """The fix must not become a blanket refusal: a kept row of the SAME
    source_type as the stale point is a real supersede and must survive."""
    older = row("id-older", "https://www.rts.ch/x", "2026-08-30", "2026-08-31")
    newer = row("id-newer", "https://www.rts.ch/x", "2026-08-30", "2026-09-05")
    # merge_rows would normally drop `older` before this step ever runs, but
    # exercise the guard directly against Superseder's own selector shape:
    # an EXISTING (already-in-twin) stale point of the same source_type as
    # the surviving kept row.
    stale_ids = ["id-existing-older"]
    stale_urls = ["https://www.rts.ch/x"]
    stale_types = {"id-existing-older": "reference"}
    out = C.filter_cross_family_stale(stale_ids, stale_urls, stale_types,
                                      [newer])
    assert out == [{"id": "id-existing-older", "url": "https://www.rts.ch/x",
                   "reason": "superseded"}]


def test_a_stale_match_with_no_group_at_all_is_also_dropped_not_crashed():
    """Defensive: a stale id whose url never grouped with anything kept
    (should not happen given Superseder's own filter, but the guard must
    not KeyError if it somehow does)."""
    out = C.filter_cross_family_stale(
        ["orphan-id"], ["https://nowhere.example/x"], {"orphan-id": "blog"},
        [HN_REFERENCE_ROW])
    assert out == []


# ---------------------------------------------------------------------------
# dense-vector dimension guard (task 648): task640-news-backlog carried
# 128-dim `dense` vectors from the MRL-128 era; aw_e8's live config is
# 256-dim. Every check above -- merge, capture-integrity, tombstones -- is
# blind to vector width, so the first thing that ever noticed was Qdrant's
# own upsert 400 at APPLY time. This guard catches it at COMPOSE time,
# against the collection's own live config, never a hardcoded width.
# ---------------------------------------------------------------------------
class FakeVectorParams:
    def __init__(self, size):
        self.size = size


class FakeCollectionClient:
    """A minimal qdrant_client-shaped fake exposing only get_collection(),
    named-vectors shape (dict of {name: VectorParams}) -- the same shape
    aw_e8 (and every other named-vector aw_b1 collection) reports."""

    def __init__(self, dim, name="dense"):
        self._dim = dim
        self._name = name

    def get_collection(self, name):
        vectors = {self._name: FakeVectorParams(self._dim)}
        params = SimpleNamespace(vectors=vectors)
        config = SimpleNamespace(params=params)
        return SimpleNamespace(config=config)


def _dense_row(id_, url, dim):
    r = row(id_, url, "2026-09-01", "2026-09-01")
    r["vector"]["dense"] = [0.1] * dim
    return r


def test_dense_dim_guard_passes_when_every_kept_row_matches_the_live_collection():
    kept = [_dense_row("id-1", "https://www.rts.ch/a", 256),
           _dense_row("id-2", "https://www.rts.ch/b", 256)]
    result = C.assert_dense_matches_collection(
        kept, client=FakeCollectionClient(256), collection="aw_e8")
    assert result == {"checked": 2, "vector_name": "dense", "expected_dim": 256}


def test_dense_dim_guard_refuses_a_128d_pack_against_a_256d_collection():
    """The real incident, reproduced: an MRL-128-era pack against aw_e8."""
    kept = [_dense_row("id-1", "https://www.rts.ch/a", 128),
           _dense_row("id-2", "https://www.rts.ch/b", 128)]
    try:
        C.assert_dense_matches_collection(
            kept, client=FakeCollectionClient(256), collection="aw_e8")
        assert False, "expected DenseDimMismatch"
    except C.DenseDimMismatch as exc:
        assert "256" in str(exc)
        assert "128" in str(exc)
        assert "2/2" in str(exc)


def test_dense_dim_guard_refuses_a_row_with_no_dense_vector_at_all():
    kept = [row("id-1", "https://www.rts.ch/a", "2026-09-01", "2026-09-01")]
    try:
        C.assert_dense_matches_collection(
            kept, client=FakeCollectionClient(256), collection="aw_e8")
        assert False, "expected DenseDimMismatch"
    except C.DenseDimMismatch as exc:
        assert "None" in str(exc) or "1/1" in str(exc)


def test_dense_dim_guard_refuses_a_mixed_pack_naming_only_the_bad_rows():
    kept = [_dense_row("id-good", "https://www.rts.ch/a", 256),
           _dense_row("id-bad", "https://www.rts.ch/b", 128)]
    try:
        C.assert_dense_matches_collection(
            kept, client=FakeCollectionClient(256), collection="aw_e8")
        assert False, "expected DenseDimMismatch"
    except C.DenseDimMismatch as exc:
        assert "id-bad" in str(exc)
        assert "id-good" not in str(exc)
        assert "1/2" in str(exc)


def test_dense_dim_guard_reads_the_width_live_never_hardcoded():
    """The collection's OWN config decides, not a constant in this module --
    a 1024-dim collection is satisfied by 1024-dim rows just as readily as a
    256-dim one is by 256-dim rows."""
    kept = [_dense_row("id-1", "https://www.rts.ch/a", 1024)]
    result = C.assert_dense_matches_collection(
        kept, client=FakeCollectionClient(1024), collection="aw_dfrlaw_cov")
    assert result["expected_dim"] == 1024


# ---------------------------------------------------------------------------
# task 669/A -- news-aged-out-weekly-2026W39 real incident: a scroll classed
# af49e590-6fa4-5c49-b59f-fe330331ac9c ("computerenhance.com/p/clean-code-
# horrible-performance") as aged news, but an L1 backfill re-fetch landed
# under the SAME deterministic point id before the drain-time gate re-read
# it -- replacing the payload with a `reference`-family row carrying no
# `published` at all. "A record whose age cannot be proven must NOT be
# deleted": the composer must re-verify age on a FRESH read right before
# claiming `aged_out`, never trust a scroll snapshot that can be 30+
# minutes stale. Real in-memory Qdrant (`qdrant_client(":memory:")`), not a
# hand-rolled fake, so these exercise the actual retrieve() call
# `reverify_aged_out` makes.
# ---------------------------------------------------------------------------
REAL_INCIDENT_ID = "af49e590-6fa4-5c49-b59f-fe330331ac9c"
REAL_INCIDENT_URL = "https://www.computerenhance.com/p/clean-code-horrible-performance"


def _memory_client_with(collection: str, points: list):
    from cytir.core.qdrant import qdrant_client
    from qdrant_client.http import models as qm
    client = qdrant_client(":memory:")
    client.create_collection(
        collection,
        vectors_config={"dense": qm.VectorParams(size=3, distance=qm.Distance.COSINE)})
    client.upsert(collection, points=[
        qm.PointStruct(id=pid, vector={"dense": [0.1, 0.2, 0.3]}, payload=payload)
        for pid, payload in points])
    return client


def test_reverify_aged_out_holds_back_a_row_the_fresh_read_shows_as_unpublished():
    """The real incident, reproduced directly against a real Qdrant store:
    the scroll-time candidate looked like aged news (published long ago),
    but the CURRENT row (what the fresh retrieve() sees) is a
    `reference`-family row with no `published` at all -- the id was
    overwritten by an unrelated L1 backfill re-fetch (task 669) between the
    scroll and this re-check. Must be held back, not tombstoned."""
    client = _memory_client_with("aw_e8", [
        (REAL_INCIDENT_ID, {
            "title": "\"Clean\" Code, Horrible Performance - by Casey Muratori",
            "lang": "en", "domain": "www.computerenhance.com",
            "source_type": "reference", "trust_tier": 0.6, "last_seen": "2026-09-26",
            "provenance": "l1_backfill_refetch", "url": REAL_INCIDENT_URL,
            "point_id": REAL_INCIDENT_ID,
            # published: NOT present -- exactly the real drifted row
        }),
    ])
    stale_scroll_snapshot = [{
        "id": REAL_INCIDENT_ID,
        "payload": {"url": REAL_INCIDENT_URL, "published": "2020-01-01",
                   "disc_class": "news"},
    }]
    cutoff = _dt.date(2023, 9, 27)  # well after 2020-01-01 -- would have aged out

    result = C.reverify_aged_out(client, "aw_e8", stale_scroll_snapshot, cutoff)

    assert result["tombstones"] == []
    assert result["age_unknown_kept"] == [
        {"id": REAL_INCIDENT_ID, "url": REAL_INCIDENT_URL, "published": None}]


def test_reverify_aged_out_confirms_a_row_still_genuinely_aged_on_fresh_read():
    """The normal, unaffected case: nothing wrote over the row between the
    scroll and the re-check, so the fresh read agrees, and the row is
    tombstoned exactly as before this guard existed."""
    pid = "11111111-1111-1111-1111-111111111111"
    url = "https://www.rts.ch/genuinely-old"
    client = _memory_client_with("aw_e8", [
        (pid, {"url": url, "published": "2020-01-01", "disc_class": "news"}),
    ])
    aged = [{"id": pid, "payload": {"url": url, "published": "2020-01-01"}}]
    cutoff = _dt.date(2023, 9, 27)

    result = C.reverify_aged_out(client, "aw_e8", aged, cutoff)

    assert result["tombstones"] == [{"id": pid, "url": url, "reason": "aged_out"}]
    assert result["age_unknown_kept"] == []


def test_reverify_aged_out_lets_an_already_gone_id_through_as_a_tombstone():
    """A row deleted entirely between the scroll and the re-check is the
    GOAL STATE for a purge, not a refusal (task 664, 2026-09-28) -- letting
    it through as a tombstone is idempotent; `promote_queue.validate_
    tombstones` already treats an already-missing id as fine."""
    pid = "22222222-2222-2222-2222-222222222222"
    url = "https://www.rts.ch/already-gone"
    client = _memory_client_with("aw_e8", [])  # nothing stored under pid
    aged = [{"id": pid, "payload": {"url": url, "published": "2020-01-01"}}]
    cutoff = _dt.date(2023, 9, 27)

    result = C.reverify_aged_out(client, "aw_e8", aged, cutoff)

    assert result["tombstones"] == [{"id": pid, "url": url, "reason": "aged_out"}]
    assert result["age_unknown_kept"] == []


def test_compute_aged_out_tombstones_returns_tombstones_and_age_unknown_kept_shape():
    """End-to-end through the real scroll + reverify path: the public
    function's return shape is a dict, not a bare list -- both callers
    (news_aged_out_weekly.py, this module's own main()) rely on this."""
    pid = "33333333-3333-3333-3333-333333333333"
    url = "https://www.rts.ch/aged-e2e"
    payload = {"url": url, "published": "2020-01-01", "disc_class": "news",
              "last_seen": "2020-01-01"}
    client = _memory_client_with("aw_e8", [(pid, payload)])
    now_ref = _dt.date(2026, 9, 27)

    result = C.compute_aged_out_tombstones(client, "aw_e8", now_ref=now_ref,
                                           months=36)

    assert result["tombstones"] == [{"id": pid, "url": url, "reason": "aged_out"}]
    assert result["age_unknown_kept"] == []


# ---------------------------------------------------------------------------
# task 698: news-daily-20261002 was refused by the drain -- 50 superseded
# tombstones carried the STORED url `https://swissinfo.ch/...` (no www) while
# the pack kept the same pages as `https://www.swissinfo.ch/...`. Real urls
# and ids from that pack's tombstones.jsonl / points.
_REAL_STALE_NO_WWW = (
    "https://swissinfo.ch/ger/aussenpolitik/"
    "die-neutralit%c3%a4t-der-schweiz-wohin-des-weges/45810048")
_REAL_KEPT_WWW = _REAL_STALE_NO_WWW.replace("://swissinfo.ch", "://www.swissinfo.ch")


_REAL_STALE_FRE = (
    "https://swissinfo.ch/fre/affaires-etrangeres/"
    "dans-quelle-direction-la-neutralit%c3%a9-suisse-%c3%a9volue-t-elle/45810512")


def test_www_replacement_tombstones_its_apex_predecessor_true_orphan_dropped(tmp_path):
    tmp_path = tmp_path / "pack"   # not tmp_path itself: see the write_pack test above (task 1516)
    tmp_path.mkdir()
    """The apex-spelled row IS superseded by the kept www. row (canonical_url
    folds scheme and www.), so its tombstone is sealed; a url that matches no
    kept row under any spelling is still a counted, dropped orphan."""
    srf = ("https://www.srf.ch/news/dialog/"
           "algorithmen-im-hr-ki-sortiert-vor-so-veraendert-sich-die-jobsuche")
    kept = [row("kept-1", _REAL_KEPT_WWW, "2026-10-02", "2026-10-02"),
            row("kept-2", srf, "2026-10-02", "2026-10-02")]
    tombstones = [
        {"id": "5f86df29-5bac-518c-a648-427943b8f1b2", "url": _REAL_STALE_NO_WWW,
         "reason": "superseded"},
        {"id": "95d0dd34-7107-5982-bd22-4b5414e2b660", "url": srf,
         "reason": "superseded"},
        # a different real swissinfo article, not in this pack in any spelling
        {"id": "orphan-1", "url": _REAL_STALE_FRE, "reason": "superseded"},
        {"id": "aged-1", "url": "https://www.rts.ch/gone", "reason": "aged_out"}]
    msgs = []
    ok, orphans = C.drop_orphan_superseded_tombstones(
        tombstones, kept, log=lambda m, **kw: msgs.append(m))
    assert [t["id"] for t in orphans] == ["orphan-1"]
    assert [t["id"] for t in ok] == [
        "5f86df29-5bac-518c-a648-427943b8f1b2",
        "95d0dd34-7107-5982-bd22-4b5414e2b660", "aged-1"]
    assert msgs and "dropped 1" in msgs[0]

    manifest = C.write_pack(
        str(tmp_path), kept=kept, tombstones=tombstones, pack_id="t698",
        family="news", collection="aw_e8", days=["2026-10-02"])
    assert manifest["orphan_tombstones_dropped"] == 1
    assert manifest["orphan_tombstone_urls_dropped"] == [_REAL_STALE_FRE]
    assert manifest["tombstones_count"] == 3
    # what is sealed passes the drain's own check
    pack_urls = {r["payload"]["url"] for r in kept}
    sealed = [json.loads(line) for line in
              pathlib.Path(tmp_path, "tombstones.jsonl").read_text().splitlines()]
    pack_canon = {document_url_key(u) for u in pack_urls}
    assert all(t["url"] in pack_urls or document_url_key(t["url"]) in pack_canon
               for t in sealed if t["reason"] == "superseded")

