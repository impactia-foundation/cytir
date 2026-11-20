"""`news_daily` on the family Protocol: a facade over `cytir.news` (task 1388).

No network, no fixture from live. The plan is a dry run through the framework (discover +
consent, nothing read); the delegation tests replace the `cytir.news` entry points and check each
stage calls the lane's own function with the lane's own arguments.
"""
from __future__ import annotations

import json

import pytest

from cytir.news import daily, publishers, seal_lane_stages
from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.family import RunContext, SourceFamily
from cytir.sources.families.news_daily import NewsDailyFamily
from cytir.sources.run_family import run_family


def _row() -> dict:
    return contract_mod.by_id(contract_mod.load())["news_daily"]


def _ctx(tmp_path, **options) -> RunContext:
    return RunContext(out_dir=tmp_path / "out", today="2026-10-29", skip_consent=True,
                      options=options, contract=_row(), log=lambda m: None)


def test_registered_and_conformant():
    assert "news_daily" in registry.registered()
    adapter, row, spec = registry.admit("news_daily")
    assert isinstance(adapter, SourceFamily)
    assert spec.declared == row["chunk_policy"]["policy_id"]
    assert not getattr(adapter, "PARTIAL", ())      # extract/chunk serve page captures (task 2783)
    assert not hasattr(adapter, "l1")           # the lane writes no L1


def test_plan_is_a_dry_run_over_the_publisher_hosts(tmp_path):
    seen = []

    def consent(host, direct):
        seen.append((host, direct))
        return {"host": host, "bucket": "ok"}

    plan = run_family("news_daily", out_dir=tmp_path / "o", today="2026-10-29", dry_run=True,
                      consent=consent, options={"day": "2026-10-28"}, log=lambda m: None)
    assert plan["dry_run"]
    assert plan["units"] == [f"day-2026-10-28@{h}" for h in publishers.hosts(daily=True)]
    assert plan["runnable"] == plan["units"]
    assert sorted(h for h, _ in seen) == sorted(publishers.hosts(daily=True))
    assert all(direct for _, direct in seen)
    assert not (tmp_path / "o").exists()        # a plan writes nothing


def test_day_defaults_to_yesterday_and_a_bad_day_refuses(tmp_path):
    units = NewsDailyFamily().discover(_ctx(tmp_path))
    assert {u.meta["day"] for u in units} == {"2026-10-28"}
    assert [u.consent_hosts for u in units] == [(h,) for h in publishers.hosts(daily=True)]
    with pytest.raises(ValueError):
        NewsDailyFamily().discover(_ctx(tmp_path, day="not-a-day"))


def test_terms_come_from_the_contract_row(tmp_path):
    ctx = _ctx(tmp_path)
    NewsDailyFamily().discover(ctx)
    t, row = ctx.notes["contract_terms"], _row()
    assert t["search_ok"] is row["consent"]["search_ok"]
    assert t["train_ok"] is row["consent"]["train_ok"]
    assert t["consent_basis"] == row["consent"]["basis"]
    assert t["licence_basis"] == row["licence"]["basis"]
    assert t["licence_spdx"] == row["licence"]["spdx"]
    bare = RunContext(out_dir=tmp_path, today="2026-10-29", contract={})
    with pytest.raises(ValueError, match="contract row"):
        NewsDailyFamily().discover(bare)


def test_acquire_refuses_without_execute(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "run_stage", lambda *a, **k: pytest.fail("fetched"))
    fam, ctx = NewsDailyFamily(), _ctx(tmp_path, day="2026-10-28")
    unit = fam.discover(ctx)[0]
    with pytest.raises(ValueError, match="execute=1"):
        fam.acquire(unit, ctx)


def test_acquire_delegates_to_run_stage_with_the_lane_defaults(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(daily, "W", tmp_path)
    monkeypatch.setattr(daily, "run_stage", lambda day, **kw: calls.update(day=day, **kw))
    monkeypatch.setattr(daily, "staged_points", lambda day: {"total": 3, "arms": {"swissinfo": 3}, "manifest_path": "m"})
    fam, ctx = NewsDailyFamily(), _ctx(tmp_path, day="2026-10-28", execute="1")
    unit = next(u for u in fam.discover(ctx) if u.meta["arm"] == "swissinfo")
    (cap,) = fam.acquire(unit, ctx)
    d = daily.build_parser().parse_args([])
    assert calls["day"] == "2026-10-28"
    assert calls["host_rate"] == d.host_rate and calls["workers"] == d.workers
    assert calls["min_publisher_rows"] == d.min_publisher_rows
    assert json.loads(cap.body)["arms"] == {"swissinfo": 3}


def test_compose_day_delegates_to_run_one_day_without_staging(tmp_path, monkeypatch):
    got = {}
    monkeypatch.setattr(daily, "run_one_day",
                        lambda day, args: got.update(day=day, args=args) or {"ok": True})
    out = NewsDailyFamily().compose_day(_ctx(tmp_path, day="2026-10-28"))
    assert out == {"ok": True} and got["day"] == "2026-10-28"
    assert got["args"].no_stage is True


def test_seal_arm_delegates_to_the_lane(monkeypatch):
    monkeypatch.setattr(seal_lane_stages, "seal_arm", lambda **kw: {"seen": kw})
    assert NewsDailyFamily().seal_arm(arm="srf", day="2026-10-28") == {
        "seen": {"arm": "srf", "day": "2026-10-28"}}


# --- one unit per (day, host): task 1446 ---------------------------------------------------
# PINS, produced once from the single-day-unit adapter on this fixture (before the split). Each host
# pin is the sha256 of that day's body cut to the host's arm entry, in the same key order and encoding.
# The per-host pins are the parity surface; no all-arm day body is pinned (the adapter does not compose one).
_STAGED = {"total": 10, "arms": {"swissinfo": 3, "chmedia": 5, "hn": 2},
           "manifest_path": "stage-2026-10-28/stage_manifest.json"}
HOST_PINS = {"www.swissinfo.ch": "d8889beb0c74ad853e50169043ff1feba0799f914d646162455ca0f85b1c42a6",
             "www.srf.ch": "bbdaaea458f9bc7178e90418e4546713d4d5c7cd00fe8845c514d4d6f32571f4",
             "www.rts.ch": "bbdaaea458f9bc7178e90418e4546713d4d5c7cd00fe8845c514d4d6f32571f4"}


def _sha(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()


def test_per_host_units_reproduce_the_pinned_arm_bytes(tmp_path, monkeypatch):
    stages = []
    monkeypatch.setattr(daily, "W", tmp_path)
    monkeypatch.setattr(daily, "run_stage", lambda day, **kw: stages.append(day))
    monkeypatch.setattr(daily, "staged_points", lambda day: _STAGED)
    fam, ctx = NewsDailyFamily(), _ctx(tmp_path, day="2026-10-28", execute="1")
    units = {u.meta["host"]: u for u in fam.discover(ctx)}
    assert set(HOST_PINS) <= set(units)
    for host, pin in HOST_PINS.items():
        (cap,) = fam.acquire(units[host], ctx)
        assert _sha(cap.body) == pin and cap.unit.key == f"day-2026-10-28@{host}"
    assert stages == ["2026-10-28"]             # the day is staged once, not once per host


def test_a_declined_host_excludes_only_its_unit(tmp_path):
    def consent(host, direct):
        return {"host": host, "bucket": "declined" if host == "www.srf.ch" else "ok"}

    plan = run_family("news_daily", out_dir=tmp_path / "o", today="2026-10-29", dry_run=True,
                      consent=consent, options={"day": "2026-10-28"}, log=lambda m: None)
    assert list(plan["blocked"]) == ["day-2026-10-28@www.srf.ch"]
    assert "day-2026-10-28@www.srf.ch" not in plan["runnable"]
    assert len(plan["runnable"]) == len(plan["units"]) - 1
