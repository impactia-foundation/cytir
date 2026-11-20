#!/usr/bin/env python3
"""Daily news: fetch one day, compose it, write it into the promotion queue.
Never applies anything itself -- no twin write, no prod write, no lock taken
(task 640, 2026-09-23, per ADR-110 / the promotion-queue directory contract).

WHY THIS EXISTS
---------------
`run_catchup_day.py` is a one-shot, hand-run CATCH-UP tool: its "step -1"
(`rerun_pack`) is the pure fetch+stage step, but the rest of that script
immediately takes TWIN-WRITE-LOCK and writes (Phase A) -- fine for a
supervised catch-up run, wrong for an unattended nightly timer that must
NEVER apply on its own. This module runs ONLY the fetch+stage+compose part,
for a single day, and hands the result to `cytir/index/promote_queue.py` (task
641) as a queue entry, exactly like `news_backlog_composer.py` does for the
historical backlog -- reusing the SAME merge/tombstone functions (one pack
here instead of many; `merge_rows` and `compute_*_tombstones` do not care).

SELF-CONTAINED (task 640, 2026-09-23, coordinator finding)
------------------------------------------------------------
The first live-fire attempt found this module still depended on a STAGED
staging-Qdrant collection that only the BENDER-orchestrated cron
(`news_refresh_cron.sh`) ever populates, via a two-machine (orchestrator+the build box)
preflight that has been failing since 2026-09-17 -- so 2026-09-18/19/20/22
were never staged at all, and this timer would fail the same way on its own
first fire. `run_stage()` below closes that gap: it runs
`news_refresh_stage.sh` (the discover+fetch+embed step) DIRECTLY,
already build-box-native by design (all its paths are build-box paths), under
ONE single-machine preflight (`news_release_preflight.py` against this
pinned repo-main checkout only -- never the orchestrator-side "local" half of the
two-machine check that has been the actual source of drift). No SSH, no
dependency on the orchestrator box's crontab or its copy of the release (memory
no-fetch rule: fetch and build on the build box, never the orchestrator box). `main()`
calls it before `run_pack()` unless `--no-stage` is passed (day already
staged). `staged_points(day)` sums `stage_manifest.json`'s arms and refuses
(SystemExit) at 0 -- before any pack/compose work runs, not after.

Egress is metered by comparing `/sys/class/net/<iface>/statistics/
{rx,tx}_bytes` before and after the stage subprocess and reporting measured
MB/s (memory zywall-protections-stay: measure on enp37s0f0, never assume).

DUPLICATION NOTE (engineering-doctrine: name debt, don't hide it)
-------------------------------------------------------------------
`run_catchup_day.py`'s `day_env` dict is NOT an importable function -- it is
built inline, after code that already has side effects (twin restore) by the
time it runs. Importing that module is unsafe (module-level `sys.argv[1]`
and unconditional twin-state calls). The STABLE subset of that dict is
mirrored here (`_STABLE_ENV`); if `run_catchup_day.py`'s constants change
(release dir, candidate registry sha, retention months), this must be
updated too. `parse_pack_summary` is copied verbatim (10 lines, pure) rather
than imported for the same reason.

QUEUE OUTPUT
------------
    promotion-queue/news-daily-<YYYYMMDD>/{manifest.json,tombstones.jsonl,
    news-points.<sha>.jsonl.gz,READY}
Same shape `news_backlog_composer.py` writes, so `promote_queue.py`'s
content-pack drain support (task 641, in flight) handles both identically.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import subprocess
import sys
import time

# cytir/news/daily.py -> the checkout (repo root). The systemd unit still starts the
# `news_daily_pack_and_queue.py` shim by path (no PYTHONPATH); the shim puts the
# repo root on sys.path before it imports this module, so `aw` always resolves.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

from cytir.core import twin as TWIN  # noqa: E402
from cytir.news import compose as C  # noqa: E402
from cytir.news import publishers as PUBLISHER_REGISTRY  # noqa: E402
from cytir.news import stage_contract as NSC  # noqa: E402

from cytir.core.paths import SettingRefused, lazy_setting_path, operator_settings  # noqa: E402

# Lazy: each resolves on first USE (`str(W)`, `W / x`, `.exists()`), never at import, and an
# unconfigured box refuses there by name (`SettingRefused`). See `cytir.core.paths.LazyPath`.
W = lazy_setting_path("news_work")
RELEASES_ROOT = lazy_setting_path("releases_root")
# task 683 S7 / 671 follow-up: with AW_NEWS_RELEASE_FROM_CHECKOUT=1 (set by the tracked unit) the release is
# DERIVED from this checkout's HEAD (`cytir.news.release`, nothing to re-pin after a merge). HEAD without a cut
# release is cut here (task 769); the run refuses (FATAL) only when the cut fails or an existing
# release fails its recorded digest, and never falls back to an older release. Unset, the release is the pinned
# operator setting `news_release_dir` (`cytir.core.paths`; refused by name when nobody configured it).
_PIN_ENV: dict = {}


R = lazy_setting_path("news_release_dir")
if os.environ.get("AW_NEWS_RELEASE_FROM_CHECKOUT") == "1":
    from cytir.news import release as _release  # noqa: E402
    try:
        # task 769: HEAD without a release is CUT here (the 11:30Z refresh moves HEAD daily);
        # it refuses only when the cut itself fails. The log line goes to stderr.
        _pin = _release.ensure_pin(REPO_ROOT, pathlib.Path(RELEASES_ROOT), pathlib.Path(W),
                                   log=lambda m: print(m, file=sys.stderr))
    except _release.ReleaseError as _exc:
        raise SystemExit(f"FATAL news release: {_exc}")
    R, _PIN_ENV = _pin.release_dir, _pin.env()


def __getattr__(name):
    # `D._STABLE_ENV` stays readable for the drivers that import it, but resolves lazily so
    # that importing this module never needs the lang-gate interpreter to be configured.
    if name == "_STABLE_ENV":
        return stable_env()
    raise AttributeError(name)


# The sealed-pack apply driver, relative to the release: the package module, always (task 2304).
# `cyt news routine-apply` runs a `.py` driver with python3. AW_NEWS_RELEASE_FROM_CHECKOUT now only
# selects how the release is cut (above); it no longer picks the driver.
APPLY_DRIVER_RELPATH = "cytir/sources/families/news_tier2_apply.py"
QUEUE_DIR = lazy_setting_path("queue_dir")
TWIN_URL, TWIN_COLL = TWIN.url(), TWIN.collection()   # the twin_url / twin_collection settings (cytir.core.twin)

# Mirrors run_catchup_day.py's day_env -- see DUPLICATION NOTE above.
def _static_env() -> dict:
    """The pins that do not depend on the release or the lang-gate interpreter."""
    qvec_dir = os.path.join(operator_settings().work_root, "task617", "gate-inputs")
    return {
        "AW_NEWS_WORK": str(W),
        "AW_NEWS_RELEASE_MANIFEST": str(W / "release-manifest-r22.json"),
        # task 676, 2026-09-30 (r22): cut from merged main 37a91639 by the same recipe (manifest
        # sha256 equal on the orchestrator box and the build box, 11,020 files, preflight check PASS on both, every
        # chain entrypoint starts from the release dir). It adds #468: the page-date clamp is the
        # real current day, so a backfill run on a free past as-of day keeps the page dates.
        # task 676, 2026-09-30 (r21): r20 (30206a3a) failed the post-cut check, `news.py` could
        # not import `aw` when run as a script (#458). r21 is cut from 1e473853 with that fix;
        # manifest sha256 equal on the orchestrator box and the build box, preflight PASS on both, and every
        # entrypoint of the chain starts clean from the release dir.
        # task 676 (lane 1), 2026-09-30: r20 was cut from origin/main 30206a3a by the same
        # recipe (git archive of the merged commit, extracted on the orchestrator box and the build box,
        # write-manifest on both, manifest sha256 equal, preflight check PASS on both).
        # It carries #432 (window-pruned child sitemaps, www.-folded host rules),
        # #444 (expected hosts), #445 (fetch-time recapture skip), #449 (page date wins).
        # task 671b, 2026-09-28: r19 was re-cut from origin/main e6c3f07b (post-#404/
        # #407 -- the apply-code move into cytir/), the same cut_release.sh-style
        # process r2..r18 used (git archive of the merged commit, extracted
        # byte-identical on the orchestrator box and the build box, write-manifest + check run on
        # BOTH copies, local/remote manifest_sha256 compared equal before
        # trusting either). No hotfix layered on top this time -- the manifest
        # is a clean archive of the commit's own tree, so AW_NEWS_RELEASE_SHA
        # is exactly this manifest's git_sha, nothing more.
        "AW_NEWS_RELEASE_MANIFEST_SHA256":
            "24246b7bd91ae8ce7c6e013538bc21b179f989f98701c37c7af7d17c7065c77b",
        "AW_NEWS_RELEASE_SHA": "37a91639263cbe7b387fb3f9daf54a81c32ca626",
        "AW_NEWS_TWIN_URL": TWIN_URL, "AW_NEWS_TWIN_COLLECTION": TWIN_COLL,
        "AW_QPY": operator_settings().qpy,
        "AW_QVEC_CACHE": str(W / "task617" / "gate-inputs" / "qvecs-1394.json")
            if (W / "task617").exists()
            else os.path.join(qvec_dir, "qvecs-1394.json"),
        "AW_QVEC_QUERY_SOURCE_MANIFEST":
            os.path.join(qvec_dir, "qvec-query-source.v7-task617.json"),
        "AW_NEWS_POINT_CANDIDATES":
            str(W / "state/news-point-candidates-bootstrap-d3906540.jsonl"),
        "AW_NEWS_POINT_CANDIDATES_SHA256":
            "bea7e1f6841b6ad896a1726ebdee02a2ee02c1bd2efd34e893ba44306f68831c",
        "AW_NEWS_RETAIN_MONTHS": "36", "AW_NEWS_FRESH_MONTHS": "12",
        "PYTHONDONTWRITEBYTECODE": "1",
        "AW_NEWS_CADENCE_LEDGER": str(W / "state/news-gate-cadence.v1.jsonl"),
        "AW_NEWS_MRL_DIM": "256",
        "AW_NEWS_CONFIG": str(W / "state" / operator_settings().news_config_name),
        "AW_NEWS_LANGGATE_REQUIREMENTS_SHA256":
            "e8b82a00b896165bb52b405549246f3fde5793bcb27c2715e5f79e069d343407",
    }

def stable_env() -> dict:
    """The stable env (`D._STABLE_ENV`, resolved lazily): the static pins plus what needs the
    release and the lang-gate interpreter, operator settings that are refused by name when
    nobody configured them (never at import)."""
    return {
        **_static_env(),
        "AW_RELEASE_DIR": str(R), "AW_REPO_DIR": str(R),
        "AW_NEWS_LANGGATE_PY": operator_settings().langgate_py,
        "AW_NEWS_APPLY_DRIVER": str(R / APPLY_DRIVER_RELPATH),
        **_PIN_ENV,
    }


def day_env(day: str) -> dict:
    dayc = day.replace("-", "")
    return {
        **os.environ, **stable_env(),
        "AW_NOW": day,
        "AW_PACK_NAME": f"news-daily-{dayc}",
        "AW_TASK598_EV": str(W / f"task640-daily-packcheck-{dayc}"),
        "DELTA_DIR": str(W / f"task640-daily-delta-{dayc}"),
    }


# ---------------------------------------------------------------------------
# Backfill (task 659x / news lane, 2026-09-25): the daily run used to always
# compute day = yesterday-UTC, so a single missed/killed run (09-23 was
# killed twice: once by the old 90-min systemd timeout, once by a
# __pycache__ preflight refusal) left a permanent hole -- the NEXT day's run
# only ever asks for "yesterday" again and silently skips the gap forever.
# `current_last_landed_day()` is the ground truth the backfill range is
# computed from: the newest day this job's OWN promotion-queue packs
# (`news-daily-<day>/receipt.json`) show `status: LANDED`, floored by a
# bootstrap high-water mark (the orchestrator's confirmed prod state at
# hand-off) so a fresh/emptied work dir never re-walks the whole history.
# The mark only ever advances (max()), and is persisted so a later run does
# not need every historical receipt to still be on disk.
# ---------------------------------------------------------------------------
#: Ground truth at hand-off (coordinator, 06:05Z 2026-09-25, task 659x):
#: prod held news through this day BEFORE this job's own receipts became
#: the record of truth. Bump only by evidence (a LANDED receipt), never by
#: hand.
BOOTSTRAP_LAST_LANDED_DAY = "2026-09-22"
HIGH_WATER_MARK_PATH = W / "state" / "news-daily-high-water-mark.json"  # lazy (LazyPath)
#: A real gap must stay VISIBLE (STATUS.json's lag_days), never silently
#: absorbed by one run trying to fetch a month of days in a row -- that
#: would also blow the day's egress budget. Cap it; the remainder is still
#: a real, reported lag that the next day's run (or a human) picks up.
DEFAULT_MAX_BACKFILL_DAYS = 5


def scan_landed_days(queue_dir: pathlib.Path = QUEUE_DIR,
                     pack_prefix: str = "news-daily-") -> list[str]:
    """Days whose `<pack_prefix><day>` promotion-queue pack has a receipt
    showing `status == "LANDED"` (written by `cytir/index/promote_queue.py`
    once it has applied the pack to twin AND prod). Directory-driven, never
    a hand-list (memory: census-roster-from-manifest). `pack_prefix` is
    parameterized so `hn_discussion_daily.py` (task 661 F2) reuses this
    same backfill/high-water-mark machinery under its own
    `hn-discussion-daily-<day>` naming instead of a second copy."""
    days: list[str] = []
    if not queue_dir.is_dir():
        return days
    for entry in sorted(queue_dir.glob(f"{pack_prefix}????????")):
        receipt_path = entry / "receipt.json"
        if not receipt_path.is_file():
            continue
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, ValueError):
            continue
        if receipt.get("status") != "LANDED":
            continue
        dayc = entry.name[len(pack_prefix):]
        if len(dayc) == 8 and dayc.isdigit():
            days.append(f"{dayc[0:4]}-{dayc[4:6]}-{dayc[6:8]}")
    return sorted(days)


def read_high_water_mark(path: pathlib.Path = HIGH_WATER_MARK_PATH,
                         bootstrap: str = BOOTSTRAP_LAST_LANDED_DAY) -> str:
    if path.is_file():
        try:
            return json.loads(path.read_text())["last_landed_day"]
        except (OSError, ValueError, KeyError):
            pass
    return bootstrap


def write_high_water_mark(day: str, path: pathlib.Path = HIGH_WATER_MARK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(
        {"last_landed_day": day,
         "updated_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
        indent=1) + "\n")
    tmp.replace(path)  # atomic same-filesystem rename


def current_last_landed_day(queue_dir: pathlib.Path = QUEUE_DIR,
                            hw_path: pathlib.Path = HIGH_WATER_MARK_PATH,
                            pack_prefix: str = "news-daily-",
                            bootstrap: str = BOOTSTRAP_LAST_LANDED_DAY) -> str:
    """The backfill floor: max(persisted high-water mark, any newer LANDED
    receipt found now). Monotonic -- a pruned/rotated queue dir with fewer
    visible receipts than a prior run saw must never REGRESS the mark."""
    hw = read_high_water_mark(hw_path, bootstrap=bootstrap)
    landed = scan_landed_days(queue_dir, pack_prefix=pack_prefix)
    best = max([hw, *landed]) if landed else hw
    if best != hw:
        write_high_water_mark(best, hw_path)
    return best


def backfill_days(last_landed_day: str, through_day: str,
                  max_days: int = DEFAULT_MAX_BACKFILL_DAYS) -> tuple[list, int]:
    """Every day strictly after `last_landed_day` through `through_day`
    (inclusive), oldest first, capped at `max_days` (the NEWEST days are
    kept -- catching up to today matters more than an old single day).
    Returns (days, gap_before_cap) so callers can report a truncated
    backfill honestly instead of just running the cap silently."""
    start = _dt.date.fromisoformat(last_landed_day) + _dt.timedelta(days=1)
    end = _dt.date.fromisoformat(through_day)
    if start > end:
        return [], 0
    all_days = []
    d = start
    while d <= end:
        all_days.append(d.isoformat())
        d += _dt.timedelta(days=1)
    gap = len(all_days)
    if gap > max_days:
        all_days = all_days[-max_days:]
    return all_days, gap


# ---------------------------------------------------------------------------
# Per-arm receipts + alarm (task 661 F3, 2026-09-25): "The daily timer still
# fetches and embeds HN, then DROPS it. We pay the egress and GPU cost and
# lose the output every day" -- staged but never packed, with no ledger
# saying so, is exactly the class of gap task 661 found. This makes it
# visible mechanically instead of relying on someone re-deriving it by hand
# again.
# ---------------------------------------------------------------------------
ARM_LEDGER_PATH = W / "state" / "arm-daily-ledger.jsonl"
#: The OLD hn-freshness (linked-article) arm is staged then packed=0 EVERY
#: day BY DESIGN (ADR-098 Ruling 2 -- it cannot prove `news_article.
#: original`; task 661's fix moved HN coverage to `hn_discussion_daily.py`'s
#: thread-based arm instead of re-admitting this one). Alarming on it daily
#: would just be permanent, correctly-ignored noise -- exempt it BY NAME,
#: not by "packed==0 is fine sometimes", so a real new zero-pack arm still
#: alarms.
KNOWN_ZERO_PACK_ARMS = frozenset({"hn"})

#: bare host -> arm label, generated from the publisher registry
#: (`cytir/news/publishers.v1.json`): the same rows `news_refresh_stage.sh` reads through
#: `news_stage_contract.py routes`.
_ARM_HOST_MAP = PUBLISHER_REGISTRY.routes()


def arm_of_domain(domain: str) -> str | None:
    domain = (domain or "").lower()
    for host, arm in _ARM_HOST_MAP.items():
        if domain == host or domain.endswith("." + host):
            return arm
    return None


def packed_counts_by_arm(kept_rows: list[dict]) -> dict:
    """How many of the FINAL pack's rows came from each arm, by the row's
    own `domain` payload field -- the pack itself carries no arm label
    (chmedia+swissinfo are merged into one artifact by `ARMS`), so this is
    reconstructed post hoc rather than threaded through as new state."""
    counts: dict[str, int] = {}
    for row in kept_rows:
        arm = arm_of_domain((row.get("payload") or {}).get("domain", ""))
        if arm:
            counts[arm] = counts.get(arm, 0) + 1
    return counts


def append_arm_ledger(day: str, arm_stats: dict,
                      ledger_path: pathlib.Path = ARM_LEDGER_PATH) -> None:
    """One line per (day, arm): `{day, arm, staged, packed}`. Append-only,
    never rewritten -- `compute_arm_alarms` reads the tail, it never
    trusts a single day's silence as "nothing to report" (memory:
    smoke-test-yield-end-to-end -- a lane that stopped writing its own
    ledger is indistinguishable from a lane with nothing to report unless
    every expected arm writes a row every day, staged=0 included)."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a") as fh:
        for arm, stats in sorted(arm_stats.items()):
            fh.write(json.dumps({
                "day": day, "arm": arm,
                "staged": stats.get("staged", 0),
                "packed": stats.get("packed", 0),
                "checked_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }) + "\n")


def compute_arm_alarms(ledger_path: pathlib.Path = ARM_LEDGER_PATH,
                       known_zero_pack_arms: frozenset = KNOWN_ZERO_PACK_ARMS,
                       min_consecutive_days: int = 2) -> list:
    """An arm alarms when its last `min_consecutive_days` ledger rows (by
    day, most recent first) ALL show `staged > 0` and `packed == 0` --
    "we paid to fetch it and threw it away" for more than one bad day,
    not a single day's fluke (a starved-day refusal upstream already
    stops a 0-staged day from reaching this ledger at all). Exempts
    `known_zero_pack_arms` by name."""
    if not ledger_path.is_file():
        return []
    by_arm: dict[str, list[dict]] = {}
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_arm.setdefault(row["arm"], []).append(row)
    alarms = []
    for arm, rows in sorted(by_arm.items()):
        if arm in known_zero_pack_arms:
            continue
        by_day = {r["day"]: r for r in rows}  # last write per day wins
        recent_days = sorted(by_day, reverse=True)[:min_consecutive_days]
        if len(recent_days) < min_consecutive_days:
            continue
        recent = [by_day[d] for d in recent_days]
        if all(r["staged"] > 0 and r["packed"] == 0 for r in recent):
            alarms.append({
                "arm": arm,
                "reason": f"staged>0, packed=0 for {min_consecutive_days} "
                         "consecutive day(s)",
                "days": sorted(recent_days),
            })
    return alarms


def write_status(*, last_landed_day: str, yesterday: str,
                 pending_enqueued_days: list, attempted: list,
                 gap_before_cap: int,
                 status_path: pathlib.Path = None,
                 ledger_path: pathlib.Path = ARM_LEDGER_PATH) -> dict:
    """`newsfresh_294/STATUS.json` -- the orchestrator's digest reads this
    instead of grepping logs. `lag_days` counts from `yesterday` (the most
    recent day that CAN be landed as of this run), so 0 means "caught up
    through yesterday" even while today's own day is still in progress.
    Also folds in `arms`/`alarms` (task 661 F3) computed fresh from
    `ledger_path` every call -- cheap, and it means whichever script runs
    LAST in a day's sequence (the wrapper runs news then hn_discussion)
    always leaves the freshest combined view, without the two scripts
    needing to coordinate write order."""
    status_path = status_path or (W / "STATUS.json")
    last_landed = _dt.date.fromisoformat(last_landed_day)
    lag_days = max((_dt.date.fromisoformat(yesterday) - last_landed).days, 0)
    doc = {
        "schema": "aw.news-daily-status.v1",
        "checked_at": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_landed_day": last_landed_day,
        "yesterday_utc": yesterday,
        "lag_days": lag_days,
        "pending_enqueued_days": pending_enqueued_days,
        "gap_before_cap_days": gap_before_cap,
        "last_run_attempted": [
            {"day": a["day"], "ok": a["ok"],
             "error": (None if a["ok"] else a["error"])}
            for a in attempted
        ],
        "arms": latest_arm_snapshot(ledger_path),
        "alarms": compute_arm_alarms(ledger_path),
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = status_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(status_path)
    return doc


def merge_arms_into_status(status_path: pathlib.Path,
                           ledger_path: pathlib.Path = ARM_LEDGER_PATH) -> dict:
    """For a script that does NOT own STATUS.json's day-tracking fields
    (`hn_discussion_daily.py`, which runs after the news arms in the same
    daily wrapper invocation) but DOES contribute rows to the shared arm
    ledger: refresh `arms`/`alarms` from the ledger's current tail without
    touching any other key. A missing STATUS.json (news arm's write
    hasn't happened yet, or never will in this invocation, e.g. `--day`
    testing) is not an error -- `arms`/`alarms` alone are still useful."""
    doc = {}
    if status_path.is_file():
        try:
            doc = json.loads(status_path.read_text())
        except (OSError, ValueError):
            doc = {}
    doc["arms"] = latest_arm_snapshot(ledger_path)
    doc["alarms"] = compute_arm_alarms(ledger_path)
    doc.setdefault("schema", "aw.news-daily-status.v1")
    doc["checked_at"] = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = status_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(status_path)
    return doc


def latest_arm_snapshot(ledger_path: pathlib.Path = ARM_LEDGER_PATH) -> dict:
    """The most recent ledger row per arm: `{arm: {day, staged, packed}}`."""
    if not ledger_path.is_file():
        return {}
    latest: dict[str, dict] = {}
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prior = latest.get(row["arm"])
        if prior is None or row["day"] >= prior["day"]:
            latest[row["arm"]] = {"day": row["day"], "staged": row["staged"],
                                  "packed": row["packed"]}
    return latest


DEFAULT_EGRESS_IFACE = "enp37s0f0"  # the build box's real NIC (verified 2026-09-23)


def iface_bytes(iface: str) -> int:
    """rx+tx bytes for `iface` right now, or 0 if unreadable (a missing/
    renamed NIC must not crash a run -- it just makes the egress figure
    read 0, which is visibly not a real measurement)."""
    total = 0
    for direction in ("rx_bytes", "tx_bytes"):
        try:
            total += int(pathlib.Path(
                f"/sys/class/net/{iface}/statistics/{direction}"
            ).read_text().strip())
        except (OSError, ValueError):
            return 0
    return total


def staged_points(day: str) -> dict:
    """Sum `points_count` across every arm in `stage-$day/stage_manifest.json`.
    Pure (given the manifest's own content) -- separated from `run_stage()`
    so a refusal-on-zero test never has to shell out or touch the network."""
    manifest_path = W / f"stage-{day}" / "stage_manifest.json"
    if not manifest_path.is_file():
        return {"total": 0, "arms": {}, "manifest_path": str(manifest_path)}
    manifest = json.loads(manifest_path.read_text())
    arms = manifest.get("arms", {})
    per_arm = {label: int(arm.get("points_count", 0)) for label, arm in arms.items()}
    return {"total": sum(per_arm.values()), "arms": per_arm,
           "manifest_path": str(manifest_path)}


def run_stage(day: str, *, out_log: pathlib.Path, min_rows: int = 1,
             min_publisher_rows: int = 100,
             host_rate: float = 0.4, global_rate: float = 0.8,
             workers: int = 2, window_months: int = 1,
             hn_since_days: int = 3,
             max_child_sitemaps: int = 200,
             egress_iface: str = DEFAULT_EGRESS_IFACE) -> dict:
    """Self-contained discover+fetch+embed for ONE day, build-box-only (no
    SSH, no orchestrator box). Runs `news_refresh_stage.sh` directly -- the SAME
    script the orchestrator cron drives over SSH, already build-box-native by
    design -- under its own single-machine preflight only.

    `window_months`/`hn_since_days` default NARROW (1 month / 3 days), not
    the live cron's 12-month/7-day defaults: those defaults exist for a
    "what's new since I last ran an hour ago" tick, and re-scanning a full
    year's sitemap entries on every backfill day multiplies egress and
    runtime for no gain -- a date-route backfill for day D only needs
    content published near D. `AW_NOW=day` (unshifted) is the SAME value
    `day_env()`/`run_pack()` use for the pack step's `collection_of()`
    lookup -- staging and packing must key off the identical day or pack
    looks for a collection stage never wrote under (caught in a real
    the build box run, 2026-09-23: shifting to day+1 here alone left the pack
    step and `staged_points()` both silently looking at the wrong
    `stage-<day>` directory).

    `max_child_sitemaps` overrides `news_refresh_stage.sh`'s own
    AW_MAX_CHILD_SM default (1000, sized for the live cron's full-history
    tick) DOWN to a number sized for a ONE-MONTH freshness window (task
    659x, 2026-09-25): a live 09-24 run was observed stalled on a single
    host for 40+ minutes with NO other host's fetch progress visible --
    `sitemap_ingest.py discover_sitemap_urls` walks up to
    `max_child_sitemaps` child sitemap files per host BEFORE any article is
    selected, each child fetch drawing from the SAME shared rate-limit
    bucket the article fetches use, so a host with hundreds of dated child
    sitemaps (srf.ch discovered 357k historical URLs) can spend most of a
    run's egress/time budget on discovery of content outside the freshness
    window it will discard anyway. 200 is a conservative cut (most
    publisher sitemap indexes are date/month-partitioned, recent-first);
    it is NOT yet validated against a real measured before/after run --
    the next daily run's `[stage] ... elapsed=`/`egress=` line is that
    validation. `--max-child-sitemaps 1000` restores the old behaviour.

    Refuses (SystemExit) if the resulting stage_manifest sums to fewer than
    `min_rows` points, BEFORE any pack/compose work is attempted -- a
    publisher outage or a bad release pin must not silently look like "no
    news happened that day". A SECOND, stricter refusal (coordinator,
    2026-09-23, after a 09-19 day-fetch's raw yield was questioned): the
    four Swiss publisher sources (chmedia = srf.ch+rts.ch+rsi.ch,
    +swissinfo -- see news_refresh_stage.sh's `routes` dict; hn is a
    community source, not counted here) together must stage at least
    `min_publisher_rows` raw points, or refuse -- this is a DISCOVERY-level
    check (did the sitemap crawl actually find publisher content that day),
    independent of how much survives dedup/recapture-overhang downstream. A
    low KEPT count after dedup is not evidence of an outage; a low STAGED
    count is."""
    now_arg = day
    env = {
        **day_env(day), "AW_NOW": now_arg, "AW_MODE": "daily",
        "AW_NEWS_FRESH_MONTHS": str(window_months),
        "AW_HOST_RATE": str(host_rate), "AW_GLOBAL_RATE": str(global_rate),
        "AW_WORKERS": str(workers), "AW_HN_SINCE_DAYS": str(hn_since_days),
        "AW_MAX_CHILD_SM": str(max_child_sitemaps),
        "AW_STAGING_URL": os.environ.get("AW_STAGING_URL", "http://localhost:6398"),
    }
    ensure_log = out_log.with_suffix(".ensure-qdrant.log")
    with ensure_log.open("w") as fh:
        ensure = subprocess.run(
            ["bash", str(R / "deploy/scripts/ensure-staging-qdrant.sh")],
            cwd=str(W), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
    if ensure.returncode:
        raise SystemExit(f"staging qdrant lifecycle check FAILED "
                         f"(exit={ensure.returncode}) -- see {ensure_log}")

    before = iface_bytes(egress_iface)
    t0 = time.monotonic()
    with out_log.open("w") as fh:
        result = subprocess.run(
            [sys.executable, "-m", "cytir.cli", "news", "stage"],
            cwd=str(W), env={**env, "PYTHONPATH": os.pathsep.join(
                p for p in (str(R), env.get("PYTHONPATH")) if p)}, stdout=fh, stderr=subprocess.STDOUT, text=True)
    elapsed = max(time.monotonic() - t0, 1e-6)
    after = iface_bytes(egress_iface)
    mbps = (after - before) / 1e6 / elapsed if after >= before else None

    if result.returncode:
        raise SystemExit(f"stage FAILED for day={day} (AW_NOW={now_arg}, "
                         f"exit={result.returncode}) -- see {out_log}")

    staged = staged_points(day)
    print(f"[stage] day={day} now={now_arg} staged_total={staged['total']} "
         f"arms={staged['arms']} elapsed={elapsed:.0f}s "
         f"egress={'n/a' if mbps is None else f'{mbps:.3f} MB/s'} "
         f"(iface={egress_iface})", file=sys.stderr)
    if staged["total"] < min_rows:
        raise SystemExit(
            f"REFUSING: day={day} staged {staged['total']} points across "
            f"{staged['manifest_path']}, below the required minimum "
            f"({min_rows}) -- a publisher outage or bad pin must not look "
            "like a quiet news day")
    publisher_total = staged["arms"].get("chmedia", 0) + staged["arms"].get("swissinfo", 0)
    if publisher_total < min_publisher_rows:
        raise SystemExit(
            f"REFUSING: day={day} the four Swiss publishers (chmedia+"
            f"swissinfo) staged only {publisher_total} raw points "
            f"(chmedia={staged['arms'].get('chmedia', 0)}, "
            f"swissinfo={staged['arms'].get('swissinfo', 0)}), below the "
            f"required minimum ({min_publisher_rows}) -- a publisher "
            "outage, a sitemap cap, or a broken date route must refuse, "
            "not queue a starved day")
    return {**staged, "elapsed_s": elapsed, "egress_mbps": mbps,
           "egress_iface": egress_iface}


def parse_pack_summary(pack_out_text: str) -> dict:
    """Verbatim copy of run_catchup_day.py's parser -- see module docstring's
    DUPLICATION NOTE. materialize_news_pack.py's own JSON is content-
    addressed, so this must read the LAST summary line, never pick a file by
    mtime (an unchanged artifact is not rewritten, so mtime does not move)."""
    text = pack_out_text.strip()
    start = text.rfind("\n{")
    doc = json.loads(text[start + 1:] if start >= 0 else text)
    for key in ("artifact", "pointers", "verification", "rows"):
        if key not in doc:
            raise SystemExit(f"pack summary lacks {key!r}: {doc}")
    return doc


def run_pack(day: str, *, out_log: pathlib.Path) -> dict:
    """The read-only fetch+stage step. No lock, no twin/prod write."""
    env = day_env(day)
    # `news_tier2_apply.sh pack` passes `$DELTA_DIR/aged_out.delete-ids.txt`
    # to materialize_news_pack.py unconditionally -- normally written by
    # news_refresh_delta.sh, which this self-contained path never runs (the
    # composer computes aged-out independently via
    # `compute_aged_out_tombstones`). A fresh per-day DELTA_DIR (day_env()'s
    # own convention) never has this file; an empty one is the correct
    # "nothing pre-computed to delete here" input, not a workaround for a
    # missing step (caught live on the build box, 2026-09-23, day=2026-09-22:
    # FileNotFoundError, no chunks lost -- the pack step aborted before
    # writing anything).
    delta_dir = pathlib.Path(env["DELTA_DIR"])
    delta_dir.mkdir(parents=True, exist_ok=True)
    aged_ids_path = delta_dir / "aged_out.delete-ids.txt"
    if not aged_ids_path.exists():
        aged_ids_path.write_text("")
    with out_log.open("w") as fh:
        # task 671 (task 646 S4 follow-up): the pack step's own orchestrator
        # moved to cytir/sources/families/news_tier2_apply.py -- a shim still
        # lives at the old task 598/ path for any caller not yet
        # repointed (an operator running task598's runbook by hand), but
        # this unattended timer now calls the package entry point directly.
        result = subprocess.run(
            ["python3", str(R / "cytir/sources/families/news_tier2_apply.py"), "pack"],
            cwd=str(W), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise SystemExit(f"pack build FAILED (exit={result.returncode}) -- "
                         f"see {out_log}")
    summary = parse_pack_summary(out_log.read_text())
    art = pathlib.Path(summary["artifact"])
    if not art.is_file():
        # materialize_news_pack.py sometimes prints a bare filename relative
        # to its own staging dir -- resolve against W as a fallback rather
        # than silently accepting a path that does not exist.
        art = W / summary["artifact"]
    if not art.is_file():
        raise SystemExit(f"pack summary names {art}, which does not exist")
    return {"artifact": art, "rows": summary["rows"],
           "collection": summary.get("collection")}


def _consent_refused():
    from cytir.consent.stamp import ConsentStampRefused  # noqa: PLC0415
    return ConsentStampRefused


def run_one_day(day: str, args) -> dict:
    """Fetch+stage+pack+tombstone ONE day and write it into the promotion
    queue. Never applies anything, never takes a lock (see module
    docstring). Raises SystemExit on a refusal (starved day, preflight
    failure, ...) -- callers in a backfill loop catch it per-day so one bad
    day does not swallow the rest of the range."""
    dayc = day.replace("-", "")
    pack_out_log = W / f"task640-daily-packrerun-{dayc}.out"
    stage_out_log = W / f"task640-daily-stage-{dayc}.out"

    if not args.no_stage:
        print(f"[daily] day={day} -- self-contained stage (runs on the operator "
             "box only)", file=sys.stderr)
        run_stage(day, out_log=stage_out_log, min_rows=args.min_rows,
                 min_publisher_rows=args.min_publisher_rows,
                 host_rate=args.host_rate, global_rate=args.global_rate,
                 workers=args.workers, window_months=args.window_months,
                 hn_since_days=args.hn_since_days,
                 max_child_sitemaps=args.max_child_sitemaps,
                 egress_iface=args.egress_iface)

    print(f"[daily] day={day} -- fetch+stage (never applies)", file=sys.stderr)
    packed = run_pack(day, out_log=pack_out_log)
    print(f"[daily] pack rows={packed['rows']} artifact={packed['artifact'].name}",
         file=sys.stderr)

    rows = C.read_pack_points(str(packed["artifact"].parent))
    merged = C.merge_rows(rows)
    dated_input = [r for r in rows if (r.get("payload") or {}).get("published")]
    per_url_stats = C.verify_capture_integrity(
        kept=merged["kept"], dated_input=dated_input,
        log=lambda msg, **kw: print(msg, file=sys.stderr, **kw))
    kept_urls = {(r.get("payload") or {}).get("url", "") for r in merged["kept"]}
    print(f"[daily] merge kept={len(merged['kept'])} urls={len(kept_urls)} "
         f"internally_superseded={len(merged['superseded_internal'])} "
         f"undated_dropped={len(merged['undated'])}", file=sys.stderr)

    # ADR-115 am.1: stamp consent from the host's CURRENT verdict in the STAGE's own cache
    # (`C.stage_consent_cache_dir`: the cache the stage just refreshed, not the engine's
    # older default store). Rows that cannot be stamped are QUARANTINED, left out of the
    # pack, BEFORE the tombstones are computed: a quarantined url gets no new rows and no
    # tombstone on its old rows (the served rows stay), so the pack has no orphan tombstone.
    # Any failure of this block (a refusal: cache missing or empty, or too large a
    # quarantined share; or an unexpected error: sidecar write, contracts or cache read)
    # becomes a SystemExit. main()'s per-day loop then records the day as failed, writes
    # STATUS.json (unless --no-status) and runs the later days.
    out_dir = QUEUE_DIR / f"news-daily-{dayc}"
    try:
        ctx, cache_dir = C.stage_stamp_context(W)
        merged["kept"], quarantined, consent_report = C.stamp_consent(
            merged["kept"], ctx=ctx, max_quarantine_share=args.consent_max_quarantine_share,
            log=lambda msg, **kw: print(f"[LOUD] day={day} {msg}", file=sys.stderr, **kw))
        consent_report["consent_stamp"] = {**consent_report["consent_stamp"],
                                           "cache_dir": cache_dir}
        sidecar = C.write_consent_quarantine_sidecar(str(out_dir), quarantined)
    except _consent_refused() as exc:
        raise SystemExit(f"consent stamp refused day {day}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 -- recorded as a failed day, never an escape
        print(f"[LOUD] day={day} consent stamp ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(f"consent stamp error day {day}: {type(exc).__name__}: {exc}") from exc
    quarantined_urls = {q["url"] for q in quarantined}
    per_url_stats = {u: st for u, st in per_url_stats.items() if u not in quarantined_urls}
    kept_urls = {(r.get("payload") or {}).get("url", "") for r in merged["kept"]}

    tombstones: list = []
    unchanged_result = {"unchanged_skipped_urls": [], "overhang_skipped_urls": [],
                        "changed_recaptured_urls": []}
    now_ref = _dt.date.fromisoformat(day)
    if not args.no_tombstones:
        from cytir.core.qdrant import qdrant_client  # noqa: PLC0415
        client = qdrant_client(args.twin_url, timeout=120)

        C.assert_dense_matches_collection(
            merged["kept"], client=client, collection=args.collection,
            log=lambda msg, **kw: print(msg, file=sys.stderr, **kw))

        unchanged_result = C.filter_unchanged_recaptures(
            merged["kept"], client, args.collection, now_ref=now_ref,
            overhang_days=args.recapture_overhang_days,
            log=lambda msg, **kw: print(msg, file=sys.stderr, **kw))
        merged["kept"] = unchanged_result["kept"]
        skipped_urls = (set(unchanged_result["unchanged_skipped_urls"]) |
                        set(unchanged_result["overhang_skipped_urls"]))
        per_url_stats = {u: s for u, s in per_url_stats.items()
                        if u not in skipped_urls}
        kept_urls = {(r.get("payload") or {}).get("url", "") for r in merged["kept"]}
        print(f"[daily] recapture unchanged_skipped="
             f"{len(unchanged_result['unchanged_skipped_urls'])} "
             f"overhang_skipped={len(unchanged_result['overhang_skipped_urls'])} "
             f"changed_recaptured={len(unchanged_result['changed_recaptured_urls'])} "
             f"kept_after={len(merged['kept'])} urls_after={len(kept_urls)}",
             file=sys.stderr)

        # Aged-out tombstones are NOT computed here (coordinator, 2026-09-25):
        # an unfiltered scroll of compute_aged_out_tombstones() against the
        # 35M+-point twin added 30+ minutes to the 09-24 daily run for a
        # trickle of rows (a 36-month horizon crosses very few rows/day).
        # It moved to news_aged_out_weekly.py, its own weekly pack,
        # same tombstone rules + cross-pack dedup, so deletions lose nothing
        # by waiting a week and the daily run stays on the fast path.
        tombstones = C.compute_superseded_tombstones(
            client, args.collection, merged["kept"])
        print(f"[daily] tombstones (pre-cross-pack-dedup) superseded="
             f"{sum(1 for t in tombstones if t['reason']=='superseded')} "
             "(aged_out: see the weekly pack, not computed here)",
             file=sys.stderr)

        already_claimed = C.other_queued_tombstone_ids(
            str(QUEUE_DIR), f"news-daily-{dayc}")
        tombstones = C.dedupe_tombstones_against_queue(
            tombstones, already_claimed,
            log=lambda msg, **kw: print(msg, file=sys.stderr, **kw))
        C.assert_no_cross_pack_tombstone_collision(tombstones, already_claimed)
        print(f"[daily] tombstones (final) superseded="
             f"{sum(1 for t in tombstones if t['reason']=='superseded')} "
             f"aged_out={sum(1 for t in tombstones if t['reason']=='aged_out')}",
             file=sys.stderr)
        # No word-coverage floor here: task 660 (PR #356) put the canonical
        # check in the driver itself (promote_queue.py's per-url skip +
        # 20%-of-superseded-urls circuit breaker), see
        # news_backlog_composer.py's NOTE above compute_superseded_
        # tombstones(). A compose-time pre-filter would defeat that
        # breaker.

    manifest = C.write_pack(
        str(out_dir), kept=merged["kept"], tombstones=tombstones,
        pack_id=f"news-daily-{dayc}", family="news", collection=args.collection,
        days=[day],
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
            "consent_stamp": consent_report["consent_stamp"],
            "consent_quarantine_sidecar": sidecar,
            "consent_quarantine_share": consent_report["quarantine_share"],
        })

    # Per-arm receipt (task 661 F3): staged (from this day's own
    # stage_manifest.json, independent of --no-stage) vs packed
    # (reconstructed from the FINAL kept rows' own domain, post word-
    # coverage/recapture filtering -- the true "made it to the pack" count).
    staged_arms = staged_points(day)["arms"]
    packed_arms = packed_counts_by_arm(merged["kept"])
    arm_stats = {arm: {"staged": staged_arms.get(arm, 0),
                       "packed": packed_arms.get(arm, 0)}
                for arm in set(staged_arms) | set(packed_arms)}
    append_arm_ledger(day, arm_stats)

    print(json.dumps(manifest, indent=1))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """The CLI's parser, importable so `news_coverage_backfill.py` (task 676)
    drives `run_one_day` with exactly these defaults and flags."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--day", default=None,
                    help="YYYY-MM-DD; a SINGLE day, no backfill (default: "
                         "backfill every day since the last LANDED day "
                         "through yesterday UTC)")
    ap.add_argument("--collection", default=TWIN.collection())
    ap.add_argument("--twin-url", default=TWIN_URL)
    ap.add_argument("--retain-months", type=int, default=36)
    ap.add_argument("--no-tombstones", action="store_true")
    ap.add_argument("--recapture-overhang-days", type=int,
                    default=NSC.DEFAULT_NEWS_RECAPTURE_OVERHANG_DAYS,
                    help="an already-served url is re-packed only if its "
                         "published date is within this many days of --day "
                         "(Tim, 2026-09-23)")
    ap.add_argument("--consent-max-quarantine-share", type=float,
                    default=float(os.environ.get(
                        "AW_NEWS_CONSENT_MAX_QUARANTINE_SHARE",
                        C.DEFAULT_MAX_QUARANTINE_SHARE)),
                    help="refuse the day when more than this share of its rows cannot be "
                         "stamped from the host's current consent verdict and are "
                         "quarantined (default 0.5; env AW_NEWS_CONSENT_MAX_QUARANTINE_SHARE)")
    ap.add_argument("--no-stage", action="store_true",
                    help="skip the self-contained discover+fetch+embed step "
                         "(day is already staged -- e.g. re-running pack)")
    ap.add_argument("--min-rows", type=int, default=1,
                    help="refuse the run if the day stages fewer points")
    ap.add_argument("--min-publisher-rows", type=int, default=100,
                    help="refuse if chmedia+swissinfo (the four Swiss "
                         "publishers) stage fewer raw points combined "
                         "(coordinator, 2026-09-23)")
    ap.add_argument("--window-months", type=int, default=1,
                    help="sitemap freshness lookback for the stage step "
                         "(narrow: a date-route backfill, not a live tick)")
    ap.add_argument("--hn-since-days", type=int, default=3)
    ap.add_argument("--host-rate", type=float, default=0.4,
                    help="req/s per host -- tuned down from an initial "
                         "1.5/3.0 default after a real run on the operator box measured "
                         "3.88 MB/s against the ≤1 MB/s target (2026-09-23)")
    ap.add_argument("--global-rate", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent hosts (task 659x: raised from 2 -- "
                         "there are 4 publisher hosts per day and 2 workers "
                         "serialized the smallest two behind the biggest "
                         "two; the shared global-rate token bucket is still "
                         "the real throughput ceiling, so this does not by "
                         "itself raise aggregate egress)")
    ap.add_argument("--max-child-sitemaps", type=int, default=200,
                    help="cap on child sitemaps walked per host during "
                         "discovery (task 659x: down from the live cron's "
                         "1000 -- see run_stage()'s docstring; a real "
                         "before/after measurement is still pending)")
    ap.add_argument("--egress-iface", default=DEFAULT_EGRESS_IFACE)
    ap.add_argument("--max-backfill-days", type=int,
                    default=DEFAULT_MAX_BACKFILL_DAYS,
                    help="cap on how many missed days ONE run will fetch; "
                         "a bigger gap stays visible (STATUS.json "
                         "gap_before_cap_days) rather than one run trying "
                         "to fetch a month in a row")
    ap.add_argument("--no-status", action="store_true",
                    help="do not write STATUS.json (a coverage backfill run "
                         "must never overwrite the daily timer's status)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Fail at START, before any day runs and before a pack is written: the work tree, the queue,
    # the pinned release and the lang-gate interpreter are operator settings (an unconfigured box
    # refuses by name; review of #567).
    try:
        stable_env()
        str(QUEUE_DIR)
    except SettingRefused as exc:
        print(f"[REFUSED] news daily: {exc}", file=sys.stderr, flush=True)
        return 2

    yesterday =(_dt.datetime.utcnow().date() - _dt.timedelta(days=1)).isoformat()

    if args.day:
        days_to_run, gap_before_cap = [args.day], 0
    else:
        last_landed = current_last_landed_day()
        days_to_run, gap_before_cap = backfill_days(
            last_landed, yesterday, max_days=args.max_backfill_days)
        if not days_to_run:
            print(f"[daily] already caught up through {last_landed} "
                 f"(yesterday={yesterday}) -- nothing to do", file=sys.stderr)

    attempted = []
    for day in days_to_run:
        print(f"[daily] === day={day} ({days_to_run.index(day) + 1}/"
             f"{len(days_to_run)}) ===", file=sys.stderr)
        try:
            manifest = run_one_day(day, args)
            attempted.append({"day": day, "ok": True, "manifest": manifest})
        except SystemExit as exc:
            print(f"[daily] day={day} FAILED: {exc}", file=sys.stderr)
            attempted.append({"day": day, "ok": False, "error": str(exc)})
            # Keep going: a starved/failed day (publisher outage, a bad
            # pin) must not block a LATER good day from queuing. The gap
            # stays real and visible in STATUS.json, not silently skipped.

    last_landed_now = current_last_landed_day()  # re-check: this run's own
    # packs are ENQUEUED, not yet applied, so this is unchanged by them --
    # but a concurrent promote_queue.py drain may have landed something
    # while this run was fetching.
    pending_enqueued_days = sorted(
        a["day"] for a in attempted if a["ok"]
        and a["day"] > last_landed_now)
    if not args.no_status:
        status = write_status(
            last_landed_day=last_landed_now, yesterday=yesterday,
            pending_enqueued_days=pending_enqueued_days,
            attempted=attempted, gap_before_cap=gap_before_cap)
        print(f"[daily] STATUS last_landed_day={status['last_landed_day']} "
             f"lag_days={status['lag_days']} "
             f"pending={status['pending_enqueued_days']}", file=sys.stderr)

    if days_to_run and not all(a["ok"] for a in attempted):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
