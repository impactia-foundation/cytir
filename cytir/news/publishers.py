"""The news publisher registry (data: `publishers.v1.json`).

One declaration of which publisher hosts the news stage knows, which staging arm each routes to, and whether the
DAILY run requires it (`daily_required`) or only a coverage backfill may name it (`backfill_only`).
`stage_contract`, `daily` and the stage script (through `news_stage_contract.py routes|status-hosts`) all read it,
so adding a publisher is one row here, and a backfill-only host cannot become a required host of the daily run by
being added.

Stdlib only: the stage contract is dependency-free by design.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

ROLE_DAILY = "daily_required"
ROLE_BACKFILL = "backfill_only"
ROLES = (ROLE_DAILY, ROLE_BACKFILL)
PATH = Path(__file__).with_name("publishers.v1.json")


class Publisher(NamedTuple):
    host: str   # the queue / state host, `www.` form
    arm: str
    role: str

    @property
    def bare(self) -> str:
        return self.host[4:] if self.host.startswith("www.") else self.host


@lru_cache(maxsize=1)
def registry() -> tuple[Publisher, ...]:
    value = json.loads(PATH.read_text(encoding="utf-8"))
    rows = tuple(Publisher(r["host"], r["arm"], r["role"]) for r in value["publishers"])
    hosts = [p.host for p in rows]
    if len(set(hosts)) != len(hosts):
        raise ValueError(f"{PATH.name}: a host is declared twice")
    for p in rows:
        if p.role not in ROLES or not p.host.startswith("www."):
            raise ValueError(f"{PATH.name}: bad row {p}")
    if not any(p.role == ROLE_DAILY for p in rows):
        raise ValueError(f"{PATH.name}: no daily_required publisher")
    return rows


def hosts(*, daily: bool = False) -> tuple[str, ...]:
    """All registered hosts, or only the daily run's required ones, in registry order."""
    return tuple(p.host for p in registry() if not daily or p.role == ROLE_DAILY)


def host_arm() -> dict[str, str]:
    return {p.host: p.arm for p in registry()}


def routes() -> dict[str, str]:
    """bare host (`srf.ch`) -> arm, registry order (the stage script matches by substring of the row's domain)."""
    return {p.bare: p.arm for p in registry()}


def status_hosts(expected: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The hosts the stage manifest's `source_status` reports: the daily four, then any other expected host."""
    daily = hosts(daily=True)
    return daily + tuple(h for h in hosts() if h in expected and h not in daily)
