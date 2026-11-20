"""lane_lib — shared due-date and ledger logic for the task-632
unattended refresh scheduler (new-external-refs discovery + periodic re-fetch).

Kept as a small, dependency-free library (stdlib only) so it can run on
the operator box inside the pinned langgate interpreter the news lane already uses,
without adding a second requirements surface for one script family
(engineering-doctrine).

State model
-----------
One JSON state file (the `--state` path the scheduler driver passes) tracks, per source id, the last completed run's timestamp for
each of the two lanes ("new_refs" and "periodic_refetch"). A source with no
recorded run is treated as infinitely overdue (due immediately, ratio =
infinity) — the correct cold-start reading, not a bug: every source is
"due" the first time the lane runs.

Locking
-------
`PROD-WRITE-LOCK` is a single, host-local flock file
(the path the operator's deploy names) taken before ANY step
of this lane that could lead to a production write — today that is only the
news-lane call-out (see the scheduler driver); periodic re-fetch and
new-refs discovery in THIS task never write prod themselves (they queue).
The lock is deliberately a plain flock over one well-known path so any
concurrent lane (the news catch-up lane, task 626; a future task 629) can
take the same lock with `flock` directly, in bash, with no dependency on
this module. `acquire_prod_write_lock` below is a thin Python convenience
wrapper for callers that are already in Python; it is NOT a different lock.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# A source is "alarm-overdue" once it has gone this many multiples of its own
# cadence without a completed run. Mission: "alert if any due source is > 2x
# its cadence overdue".
OVERDUE_ALARM_RATIO = 2.0


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


@dataclass
class DueResult:
    source_id: str
    lane: str
    cadence_days: int | None
    last_run: str | None
    days_since_last_run: float | None
    due: bool
    overdue_ratio: float | None  # days_since_last_run / cadence_days; None if never-due lane
    alarm: bool  # overdue_ratio > OVERDUE_ALARM_RATIO

    def to_json_dict(self) -> dict:
        """JSON-safe view: math.inf (cold-start "never run") serializes as
        `null`, not the non-standard `Infinity` token `json.dumps` would
        otherwise emit — a strict JSON consumer (the monitor check, a queued
        report read back by another tool) must not choke on this lane's own
        output."""

        def _finite(x):
            return None if (x is None or math.isinf(x)) else x

        return {
            "source_id": self.source_id,
            "lane": self.lane,
            "cadence_days": self.cadence_days,
            "last_run": self.last_run,
            "days_since_last_run": _finite(self.days_since_last_run),
            "due": self.due,
            "overdue_ratio": _finite(self.overdue_ratio),
            "alarm": self.alarm,
        }


def compute_due(
    source_id: str,
    lane: str,
    cadence_days: int | None,
    last_run_iso: str | None,
    now: datetime | None = None,
) -> DueResult:
    """Pure function: given a cadence and the last completed run, is this
    source due today, and how overdue is it (as a multiple of its own cadence)?

    cadence_days=None means "not a periodic target for this lane" (e.g. a news
    pack under periodic_refetch, or a cards-adapter pack under new_refs) —
    always NOT due, never alarms.
    """
    now = now or datetime.now(timezone.utc)
    if cadence_days is None:
        return DueResult(source_id, lane, None, last_run_iso, None, False, None, False)

    if last_run_iso is None:
        # cold start: infinitely overdue, due immediately.
        return DueResult(source_id, lane, cadence_days, None, math.inf, True, math.inf, True)

    last_run = _parse_iso(last_run_iso)
    days_since = (now - last_run).total_seconds() / 86400.0
    ratio = days_since / cadence_days
    due = days_since >= cadence_days
    alarm = ratio > OVERDUE_ALARM_RATIO
    return DueResult(source_id, lane, cadence_days, last_run_iso, days_since, due, ratio, alarm)


def load_state(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"schema": "aw.refresh-lane-state.v1", "sources": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def save_state(path: str | Path, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)  # atomic on the same filesystem


def mark_run_complete(state: dict, source_id: str, lane: str, when_iso: str | None = None) -> dict:
    when_iso = when_iso or utcnow_iso()
    state.setdefault("sources", {}).setdefault(source_id, {})[lane] = when_iso
    return state


def due_report(
    cadence_rows: Iterable[dict],
    state: dict,
    lane: str,
    now: datetime | None = None,
) -> list[DueResult]:
    """cadence_rows: the `sources` list from data/refresh-cadence.v1.json.
    lane: "new_refs" (uses row['new_refs_discovery'], always daily-cadence=1
    when true) or "periodic_refetch" (uses row['periodic_refetch_days']).
    Only rows with eligible_now=True are considered — a source with nothing
    in the index yet is never "due".
    """
    out = []
    for row in cadence_rows:
        if not row.get("eligible_now"):
            continue
        if lane == "new_refs":
            cadence_days = 1 if row.get("new_refs_discovery") else None
        elif lane == "periodic_refetch":
            cadence_days = row.get("periodic_refetch_days")
        else:
            raise ValueError(f"unknown lane: {lane}")
        last_run = state.get("sources", {}).get(row["id"], {}).get(lane)
        out.append(compute_due(row["id"], lane, cadence_days, last_run, now=now))
    return out


def append_ledger(path: str | Path, entry: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


@contextlib.contextmanager
def acquire_prod_write_lock(lock_path: str | Path, blocking: bool = False):
    """flock convenience wrapper around the single, host-local PROD-WRITE-LOCK
    (see module docstring). Raises TimeoutError if non-blocking and another
    holder has it -- callers must treat that as "skip this run", not retry
    forever (single-flight, same discipline as the news cron).
    """
    p = Path(lock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(p, "a+")
    try:
        flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(fh.fileno(), flags)
        except BlockingIOError as exc:
            raise TimeoutError(f"PROD-WRITE-LOCK held by another process: {p}") from exc
        yield
    finally:
        with contextlib.suppress(Exception):
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()
