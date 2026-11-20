#!/usr/bin/env python3
"""Daily HN-discussion arm: fetch+stage+pack+tombstone ONE day (or a
backfill range) and write it into the promotion queue. Never applies
anything, never takes a lock -- same contract as `news_daily_pack_and_queue.py`
(task 640), which this module mirrors deliberately rather than duplicating:
`current_last_landed_day`/`backfill_days`/`write_status`/`iface_bytes` are
imported from there, parameterized by this module's own
`hn-discussion-daily-` pack prefix and high-water-mark file.

WHY THIS EXISTS (task 661 F2, 2026-09-25)
------------------------------------------
ADR-108 (ratified, task 601) proved a `hn_discussion` preservation profile
that passes the ADR-098 tier-2 contract -- the HN record IS the discussion
thread (Firebase item + selected comments), never the linked page
(`.outlinks` is copied verbatim, never fetched as part of this record).
Its one production run (09-17) was manual; `hn_discussion_stage.sh` had no
timer. Meanwhile the OLD hn-freshness arm (linked article, indexes
`z.ai`/`huggingface.co`/... under the DOMAIN of the linked page, never
`news.ycombinator.com`) has been excluded from every pack since 2026-08-28
(ADR-098 Ruling 2 -- it cannot prove `news_article.original`). Net effect,
measured in task 661 (glm53-freshness-root-cause): no HN thread from
2026-07-11 to 2026-09-09 is in the index at all. This module is the fix:
put `hn_discussion_stage.sh` on the SAME daily cadence as the news arms,
through the SAME promotion-queue mechanical path (ADR-110) the news lane
already uses -- not the old ADR-098 battery
(task 601's `hn_discussion_tier2_apply.sh`'s `gate-baseline` /
`preflight-dry` / `intent` / `runtime` / `verify` / `apply-armed` steps),
which predates ADR-110 and is superseded by it (news's own tier-2 driver
predates it identically -- see `news_daily_pack_and_queue.py`'s own
`run_pack()`). Only `hn_discussion_tier2_apply.sh`'s `pack` step is reused
here, DIRECTLY (`materialize_news_pack.py`, the same script both drivers
call) -- the rest of that file is not invoked and is dead weight this
module makes visible, not something it silently duplicates.

CORRECTION (coordinator, 2026-09-25, after Tim): an earlier draft of this
fix (`hn_linked_page_ingest.py`, since removed) tried to route
HN-linked pages into the news pack through the publisher-page path. That
contradicted ADR-108 as written: the linked page is an outlink, fetched
(if at all) by the separate consent-gated external-ref/discovery lane,
never by this arm. This module stages and packs the THREAD only.

OUTLINKS: each staged discussion's `.outlinks` (the story's external URL,
already retained verbatim by `cytir.sources.hn.discussion_ingest`, never re-fetched
here) is the input list for the external-ref discovery lane
(`specs/refresh-cadence.md`) -- wiring that lane is a separate piece
(task 661 §3), not this module's job.

RETENTION: hn_discussion rows are ordinary news-shaped records for aging
purposes -- ADR-085's amendment (2026-08-10, FRESH 12 / RETAIN 36) applies,
the SAME `--retain-months` this module shares with the news arms (36,
matching `news_daily_pack_and_queue.py`'s own default). No "unchanged
recapture" filter is needed (unlike the news arms): a thread is captured
once per HN item id; ADR-108's own deferred "matured-recapture (>=48h
re-selection)" branch is out of scope here, same as it was for ADR-108
itself.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import subprocess
import sys

from cytir.core import twin as TWIN  # noqa: E402
from cytir.core.paths import lazy_setting_path  # noqa: E402
from cytir.news import compose as C  # noqa: E402
from cytir.news import daily as D  # noqa: E402

R = D.R  # same pinned release the news arms run from
W = lazy_setting_path("hn_work")  # resolves on first use, never at import
QUEUE_DIR = D.QUEUE_DIR
TWIN_URL, TWIN_COLL = D.TWIN_URL, D.TWIN_COLL

PACK_PREFIX = "hn-discussion-daily-"
HIGH_WATER_MARK_PATH = W / "state" / "hn-discussion-daily-high-water-mark.json"
#: Ground truth at hand-off (task 661, 2026-09-25): the ONE prior production
#: run (09-17) covered stories 09-10..09-16; nothing before that is staged.
#: Bump only by evidence (a LANDED receipt), never by hand.
BOOTSTRAP_LAST_LANDED_DAY = "2026-09-09"
DEFAULT_MAX_BACKFILL_DAYS = 5

MATERIALIZE_SCRIPT = R / "cytir/sources/families/news.py"


def day_env(day: str) -> dict:
    return {
        **os.environ, **D.stable_env(),
        "AW_NOW": day,
        "AW_HN_DISCUSSION_WORK": str(W),
        "AW_NEWS_MRL_DIM": "256",
    }


def landed_holes(last_landed_day: str, *, queue_dir: pathlib.Path | None = None,
                 window_days: int = 14) -> list[str]:
    """Days in the `window_days` up to and including `last_landed_day` with
    no LANDED `hn-discussion-daily-<day>` pack (the high-water mark is
    monotonic, so a failed middle day is otherwise invisible)."""
    landed = set(D.scan_landed_days(
        QUEUE_DIR if queue_dir is None else queue_dir, pack_prefix=PACK_PREFIX))
    end = _dt.date.fromisoformat(last_landed_day)
    days = [(end - _dt.timedelta(days=i)).isoformat()
            for i in range(window_days - 1, -1, -1)]
    return [d for d in days if d not in landed]


def run_stage(day: str, *, out_log: pathlib.Path, since_days: int = 1,
             min_points: int = 40,
             egress_iface: str = D.DEFAULT_EGRESS_IFACE) -> dict:
    """Run `hn_discussion_stage.sh` for exactly ONE calendar day: `--now
    <day+1> --since-days 1` (via AW_NOW/AW_HN_DISC_SINCE_DAYS) gives the
    Algolia window [day 00:00, day+1 00:00) -- the SAME anchoring
    `news_daily_pack_and_queue.run_stage` uses for the news arms' own
    `AW_NOW=day` (unshifted) convention, just expressed through this
    script's `--since-days` window instead of a `--window-months` one."""
    next_day = (_dt.date.fromisoformat(day) + _dt.timedelta(days=1)).isoformat()
    # hn_discussion_stage.sh's own $SPOOL is $WORK/stage-$NOW, and $NOW is
    # `next_day` here (the --now/AW_NOW value passed above), NOT `day` --
    # caught live, task 661 F4 (2026-09-25): looking for stage-{day} found
    # nothing while the real manifest sat in stage-{next_day}, the exact
    # "shifting to day+1 alone leaves the pack step looking at the wrong
    # directory" trap news_daily_pack_and_queue.run_stage()'s own
    # docstring already names for the news arms.
    manifest_path = W / f"stage-{next_day}" / "stage_manifest.json"
    # REUSE a day already staged successfully, rather than re-invoking the
    # shell script (task 661 F2 incident, 2026-09-28): the backfill driver
    # re-attempts every day below `current_last_landed_day` on EVERY run
    # until it lands -- and while the promotion drain is stuck (a separate
    # bug), that day never advances, so the SAME days get re-attempted
    # daily. aw.news.hn_discussion_frontier/aw.sources.hn.discussion_ingest write their
    # SPOOL artifacts (frontier.jsonl, preserved/receipts/*) as immutable
    # -- correctly, since re-fetching an already-retained Firebase item
    # byte-for-byte is pointless and a silent overwrite would be a real
    # provenance defect -- so a blind re-run crashes on its own PRIOR
    # SUCCESS: days 09-23..09-26 all staged cleanly on 2026-09-27 (real
    # points_count each), then every retry on 2026-09-28 hit "immutable
    # frontier/receipt artifact collision" against that same completed
    # work, and the whole daily run failed with exit=1. A day whose
    # manifest already exists and parses is done; skip the subprocess
    # entirely and reuse it.
    spool = manifest_path.parent
    if spool.is_dir() and not manifest_path.is_file():
        # A spool dir WITHOUT a manifest = a run that died mid-stage (task
        # 1347). The stage script's immutable-artifact rule would fail every
        # retry against it; keep it aside as provenance and stage afresh.
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        aside = spool.with_name(f"{spool.name}.failed-{stamp}")
        spool.rename(aside)
        print(f"[hn_discussion stage] day={day} moved half-staged spool "
             f"{spool} aside to {aside}", file=sys.stderr)
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            staged = int(manifest["points_count"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise SystemExit(
                f"hn_discussion stage: {manifest_path} exists but does not "
                f"parse as a valid manifest ({exc}) -- a corrupt leftover "
                "must not be silently reused or silently overwritten; "
                "investigate by hand") from exc
        print(f"[hn_discussion stage] day={day} REUSING already-staged "
             f"{manifest_path} (staged={staged} "
             f"collection={manifest.get('collection')}) -- no subprocess run",
             file=sys.stderr)
        return {"staged": staged, "collection": manifest.get("collection"),
               "elapsed_s": 0.0, "egress_mbps": None}

    env = {
        **day_env(day), "AW_NOW": next_day,
        "AW_HN_DISC_SINCE_DAYS": str(since_days),
        "AW_HN_DISC_MIN_POINTS": str(min_points),
    }
    before = D.iface_bytes(egress_iface)
    import time
    t0 = time.monotonic()
    with out_log.open("w") as fh:
        result = subprocess.run(
            ["bash", str(R / "deploy/scripts/hn_discussion_stage.sh")],
            cwd=str(W), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
    elapsed = max(time.monotonic() - t0, 1e-6)
    after = D.iface_bytes(egress_iface)
    mbps = (after - before) / 1e6 / elapsed if after >= before else None
    if result.returncode:
        raise SystemExit(f"hn_discussion stage FAILED for day={day} "
                         f"(exit={result.returncode}) -- see {out_log}")
    if not manifest_path.is_file():
        raise SystemExit(f"hn_discussion stage claimed success for day={day} "
                         f"but wrote no {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    staged = int(manifest.get("points_count", 0))
    print(f"[hn_discussion stage] day={day} staged={staged} "
         f"collection={manifest.get('collection')} elapsed={elapsed:.0f}s "
         f"egress={'n/a' if mbps is None else f'{mbps:.3f} MB/s'}",
         file=sys.stderr)
    return {"staged": staged, "collection": manifest.get("collection"),
           "elapsed_s": elapsed, "egress_mbps": mbps}


def run_pack(day: str, staged_collection: str, *, out_log: pathlib.Path,
            collection: str = TWIN.collection()) -> dict:
    """The read-only pack step: stage a materialized artifact from the
    hn_discussion staging collection. No lock, no twin/prod write. Calls
    `materialize_news_pack.py` DIRECTLY -- the same script
    task 601's `hn_discussion_tier2_apply.sh`'s own `pack` step
    calls; the rest of that battery is not needed on the ADR-110
    promotion-queue path (see module docstring)."""
    aged_ids_path = W / f"empty-aged-ids-{day}.txt"
    if not aged_ids_path.exists():
        aged_ids_path.write_text("")
    out_dir = W / f"pack-{day.replace('-', '')}-hndisc"
    with out_log.open("w") as fh:
        result = subprocess.run(
            [D._STABLE_ENV["AW_QPY"], str(MATERIALIZE_SCRIPT),
             "--staging-url", D._STABLE_ENV.get(
                 "AW_STAGING_URL", "http://localhost:6398"),
             "--staging-collection", staged_collection,
             "--twin-url", TWIN_URL, "--collection", collection,
             "--aged-ids", str(aged_ids_path), "--cutoff", day,
             "--out-dir", str(out_dir)],
            cwd=str(W), stdout=fh, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise SystemExit(f"hn_discussion pack build FAILED (exit="
                         f"{result.returncode}) -- see {out_log}")
    summary = D.parse_pack_summary(out_log.read_text())
    art = pathlib.Path(summary["artifact"])
    if not art.is_file():
        art = W / summary["artifact"]
    if not art.is_file():
        raise SystemExit(f"pack summary names {art}, which does not exist")
    return {"artifact": art, "rows": summary["rows"]}


def run_one_day(day: str, args) -> dict:
    dayc = day.replace("-", "")
    stage_out_log = W / f"stage-{day}.out"
    pack_out_log = W / f"packrerun-{dayc}.out"

    staged = run_stage(day, out_log=stage_out_log, since_days=args.since_days,
                       min_points=args.min_points,
                       egress_iface=args.egress_iface)
    if staged["staged"] == 0:
        raise SystemExit(f"REFUSING: day={day} staged 0 hn_discussion points "
                         "-- a publisher/Algolia outage must not look like "
                         "a quiet day")

    packed = run_pack(day, staged["collection"], out_log=pack_out_log,
                      collection=args.collection)
    print(f"[hn_discussion pack] day={day} rows={packed['rows']} "
         f"artifact={packed['artifact'].name}", file=sys.stderr)

    rows = C.read_pack_points(str(packed["artifact"].parent))
    merged = C.merge_rows(rows)
    dated_input = [r for r in rows if (r.get("payload") or {}).get("published")]
    per_url_stats = C.verify_capture_integrity(
        kept=merged["kept"], dated_input=dated_input,
        log=lambda msg, **kw: print(msg, file=sys.stderr, **kw))
    kept_urls = {(r.get("payload") or {}).get("url", "") for r in merged["kept"]}
    print(f"[hn_discussion merge] day={day} kept={len(merged['kept'])} "
         f"urls={len(kept_urls)} undated_dropped={len(merged['undated'])}",
         file=sys.stderr)

    tombstones: list = []
    if not args.no_tombstones:
        from cytir.core.qdrant import qdrant_client  # noqa: PLC0415
        client = qdrant_client(args.twin_url, timeout=120)
        C.assert_dense_matches_collection(
            merged["kept"], client=client, collection=args.collection,
            log=lambda msg, **kw: print(msg, file=sys.stderr, **kw))
        # Aged-out tombstones: same reasoning as
        # news_daily_pack_and_queue.py's identical cut (coordinator,
        # 2026-09-25) -- moved to the shared weekly pack
        # (`news_aged_out_weekly.py`), not computed on this daily
        # path.
        tombstones = C.compute_superseded_tombstones(
            client, args.collection, merged["kept"])
        already_claimed = C.other_queued_tombstone_ids(
            str(QUEUE_DIR), f"{PACK_PREFIX}{dayc}")
        tombstones = C.dedupe_tombstones_against_queue(
            tombstones, already_claimed,
            log=lambda msg, **kw: print(msg, file=sys.stderr, **kw))
        C.assert_no_cross_pack_tombstone_collision(tombstones, already_claimed)
        # No compose-time word-coverage floor here: task 660 (PR #356) put
        # the canonical check in promote_queue.py itself (per-url skip +
        # 20%-of-superseded-urls circuit breaker) -- see
        # news_backlog_composer.py's NOTE above compute_superseded_
        # tombstones().

    out_dir = QUEUE_DIR / f"{PACK_PREFIX}{dayc}"
    manifest = C.write_pack(
        str(out_dir), kept=merged["kept"], tombstones=tombstones,
        pack_id=f"{PACK_PREFIX}{dayc}", family="news",
        collection=args.collection, days=[day],
        extra_manifest={
            "arm": "hn_discussion",
            "undated_dropped": len(merged["undated"]),
            "urls": len(kept_urls),
            "per_url_chunk_and_word_totals": per_url_stats,
            "staged_points": staged["staged"],
        })

    # Per-arm receipt (task 661 F3), the same shared ledger the news arms
    # write to -- one thread record per url here, so "packed" is simply
    # the final kept-row count (no per-host bucketing needed, unlike the
    # multi-arm news pack).
    D.append_arm_ledger(day, {"hn_discussion": {"staged": staged["staged"],
                                                "packed": len(merged["kept"])}},
                        ledger_path=D.ARM_LEDGER_PATH)

    print(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--day", default=None,
                    help="YYYY-MM-DD; a SINGLE day, no backfill (default: "
                         "backfill every day since the last LANDED day "
                         "through yesterday UTC)")
    ap.add_argument("--collection", default=TWIN.collection())
    ap.add_argument("--twin-url", default=TWIN_URL)
    ap.add_argument("--retain-months", type=int, default=36)
    ap.add_argument("--no-tombstones", action="store_true")
    ap.add_argument("--since-days", type=int, default=1)
    ap.add_argument("--min-points", type=int, default=40)
    ap.add_argument("--egress-iface", default=D.DEFAULT_EGRESS_IFACE)
    ap.add_argument("--max-backfill-days", type=int,
                    default=DEFAULT_MAX_BACKFILL_DAYS)
    args = ap.parse_args(argv)

    W.mkdir(parents=True, exist_ok=True)
    yesterday = (_dt.datetime.utcnow().date() - _dt.timedelta(days=1)).isoformat()

    if args.day:
        days_to_run, gap_before_cap = [args.day], 0
    else:
        last_landed = D.current_last_landed_day(
            QUEUE_DIR, HIGH_WATER_MARK_PATH, pack_prefix=PACK_PREFIX,
            bootstrap=BOOTSTRAP_LAST_LANDED_DAY)
        days_to_run, gap_before_cap = D.backfill_days(
            last_landed, yesterday, max_days=args.max_backfill_days)
        if not days_to_run:
            print(f"[hn_discussion daily] already caught up through "
                 f"{last_landed} (yesterday={yesterday})", file=sys.stderr)

    attempted = []
    for day in days_to_run:
        print(f"[hn_discussion daily] === day={day} "
             f"({days_to_run.index(day) + 1}/{len(days_to_run)}) ===",
             file=sys.stderr)
        try:
            manifest = run_one_day(day, args)
            attempted.append({"day": day, "ok": True, "manifest": manifest})
        except SystemExit as exc:
            print(f"[hn_discussion daily] day={day} FAILED: {exc}",
                 file=sys.stderr)
            attempted.append({"day": day, "ok": False, "error": str(exc)})

    last_landed_now = D.current_last_landed_day(
        QUEUE_DIR, HIGH_WATER_MARK_PATH, pack_prefix=PACK_PREFIX,
        bootstrap=BOOTSTRAP_LAST_LANDED_DAY)
    pending_enqueued_days = sorted(
        a["day"] for a in attempted if a["ok"] and a["day"] > last_landed_now)
    status = D.write_status(
        last_landed_day=last_landed_now, yesterday=yesterday,
        pending_enqueued_days=pending_enqueued_days,
        attempted=attempted, gap_before_cap=gap_before_cap,
        status_path=W / "STATUS.json")
    print(f"[hn_discussion daily] STATUS last_landed_day="
         f"{status['last_landed_day']} lag_days={status['lag_days']} "
         f"pending={status['pending_enqueued_days']} "
         f"holes={landed_holes(last_landed_now)}", file=sys.stderr)

    # Refresh the SHARED newsfresh_294/STATUS.json's arms/alarms with this
    # run's ledger contribution -- that file (not this module's own, above)
    # is the one the orchestrator's digest reads (task 661 F3). Never
    # touches the news arm's own last_landed_day/lag_days/pending fields.
    D.merge_arms_into_status(D.W / "STATUS.json", ledger_path=D.ARM_LEDGER_PATH)

    if days_to_run and not all(a["ok"] for a in attempted):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
