"""A second RTS discovery pass (generation 2) runs on as-of days of its own (BAU shift 5).

One as-of day keys one spool and one pack (`occupied()`): generation 1 used day 2 of every next
month, 2026-07-18 for July, 2026-07-16 for the tail and 2025-10-01 for the smoke. The urls the
fixed discovery finds (the four hubs that ended `stale`) need runs too, and they cannot reuse
those days: a same-day pack id would be skipped by the drain (DONE) and its rows lost silently
(the 2026-10-02 refresh-lane incident class). Generation g takes the days one step away.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from cytir.news import coverage_backfill as B  # noqa: E402

TODAY = dt.date(2026, 10, 3)


def _disc(tmp_path, rows):
    d = tmp_path / "state" / "discovery"
    d.mkdir(parents=True)
    (d / "urls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "state.json").write_text(json.dumps({"window_start": "2025-09-28"}))
    return tmp_path / "state"


def _rows():
    def u(i):
        return f"https://www.rts.ch/info/suisse/2026/article/x-{29000000 + i}.html"
    days = ["2025-10-20", "2026-03-05", "2026-06-30", "2026-07-03", "2026-07-17",
            "2026-07-25", "2026-09-10"]
    return [{"url": u(i), "section": "/info/suisse/", "card_date": d, "cursor_date": d}
            for i, d in enumerate(days)]


def test_generation_1_is_the_plan_that_already_ran(tmp_path):
    runs = {r.key: r for r in B.rts_runs(_disc(tmp_path, _rows()), smoke=False)}
    assert runs["rts-2025-10"].as_of == "2025-11-02"
    assert runs["rts-2026-07"].as_of == "2026-07-18" and runs[B.RTS_TAIL_KEY].as_of == "2026-07-16"
    assert B.rts_as_of("2026-12") == "2027-01-02"


def test_generation_2_uses_other_days_and_other_keys_for_every_run(tmp_path):
    sd = _disc(tmp_path, _rows())
    g1 = B.rts_runs(sd, smoke=False)
    g2 = B.rts_runs(sd, smoke=False, generation=2)
    assert {r.key.replace("-g2", "") for r in g2} == {r.key for r in g1}
    assert all(r.key.endswith("-g2") for r in g2)
    assert not ({r.as_of for r in g1} & {r.as_of for r in g2}), "no as-of day is used twice"
    by = {r.key: r for r in g2}
    assert by["rts-2025-10-g2"].as_of == "2025-11-03"
    assert by["rts-2026-07-g2"].as_of == "2026-07-13" and by["rts-tail-g2"].as_of == "2026-07-14"
    assert all(dt.date.fromisoformat(r.as_of) < dt.date(2026, 7, 15) for r in g2), \
        "07-15 to 07-20 belong to other runs, 07-21 on to the daily timer"


def test_the_july_split_moves_with_the_generation(tmp_path):
    sd = _disc(tmp_path, _rows())
    g2 = {r.key: r for r in B.rts_runs(sd, smoke=False, generation=2)}
    early = {u for u in g2["rts-2026-07-g2"].urls}
    late = set(g2["rts-tail-g2"].urls)
    # 07-03 is before the edge (07-17, inclusive); 07-25 is after it; Sep goes to the tail
    assert any("29000003" in u for u in early) and any("29000004" in u for u in early)
    assert not any("29000005" in u for u in early)
    assert any("29000005" in u for u in late) and any("29000006" in u for u in late)


def test_the_smoke_day_moves_too_and_every_url_is_in_exactly_one_run(tmp_path):
    sd = _disc(tmp_path, _rows())
    (smoke,) = B.rts_runs(sd, smoke=True, generation=2)
    assert smoke.as_of == "2025-09-30" and smoke.key.endswith("-smoke-g2")
    runs = B.rts_runs(sd, smoke=False, generation=2)
    urls = [u for r in runs for u in r.urls]
    assert len(urls) == len(set(urls)) == len(_rows())


def test_the_resume_command_carries_the_generation():
    cmd = B.resume_command(["rts"], pathlib.Path("/x/state"), 2)
    assert cmd.endswith("--state-dir /x/state --generation 2")
    assert "--generation" not in B.resume_command(["rts"], pathlib.Path("/x/state"))


def test_a_generation_out_of_range_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        B.rts_runs(_disc(tmp_path, _rows()), smoke=False, generation=0)


def test_no_generation_shares_a_day_with_another_or_with_a_taken_july_day(tmp_path):
    sd = _disc(tmp_path, _rows())
    days = {}
    taken = {"2026-07-02", "2026-07-15", "2026-07-16", "2026-07-17", "2026-07-18", "2026-07-19",
             "2026-07-20"}                    # on the GPU box 2026-10-03: stages / packs / other runs
    for g in range(2, 6):
        for r in B.rts_runs(sd, smoke=False, generation=g):
            assert r.as_of not in days, (g, r.key, days[r.as_of])
            assert r.as_of not in taken, (g, r.key)
            days[r.as_of] = r.key
