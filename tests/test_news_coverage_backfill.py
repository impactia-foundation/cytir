"""News coverage backfill driver (task 676): seeds, plan, smoke gate, resume.

No network: `run_one_day` and the NIC counters are injected. The seed shapes are
the ones measured on the publishers' own sitemap indexes on 2026-09-30 (see
task 676, news 12-month coverage, section 3).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("AW_OPERATOR_URL", "https://websear.ch")
os.environ.setdefault("AW_OPERATOR_CONTACT", "bender@impactia.org")

from cytir.news import coverage_backfill as B  # noqa: E402
from cytir.core.paths import SettingRefused  # noqa: E402
from tests.helpers.repo_files import repo_file_or_skip  # noqa: E402
from cytir.news import daily as D  # noqa: E402

TODAY = dt.date(2026, 9, 30)
SITEMAP_ONLY = {n: p for n, p in B.PUBLISHERS.items() if not p.urls_source}
SITEMAP_PUBS = sorted(SITEMAP_ONLY)


def test_swissinfo_seeds_are_one_urlset_per_day_and_language():
    seeds = B.swissinfo_seeds("2025-11")
    assert len(seeds) == 30 * 4
    assert "https://www.swissinfo.ch/ita/sitemap-2025.xml?mm=11&dd=05" in seeds
    assert all("&amp;" not in s for s in seeds)
    langs = {s.split("/")[3] for s in seeds}
    assert langs == {"eng", "ger", "fre", "ita"}, "the daily run's defect was English only"


def test_swissinfo_seeds_stop_at_the_as_of_day():
    assert len(B.swissinfo_seeds("2026-07", dt.date(2026, 7, 20))) == 20 * 4


def test_srf_and_rsi_seeds():
    assert B.srf_seeds("2026-03") == ["https://www.srf.ch/sitemaps/aron/articles/2026_03.xml"]
    rsi = B.rsi_seeds("2026-07")
    assert rsi[0].endswith("/archive/story/243.xml") and rsi[-1].endswith("/archive/story/259.xml")
    assert len(rsi) == 17


def test_publishers_roster_excludes_paywalled_and_unarchived_hosts():
    assert sorted(B.PUBLISHERS) == ["rsi", "rts", "srf", "swissinfo", "swissinfo-tail", "watson"]
    assert SITEMAP_PUBS == ["rsi", "srf", "swissinfo"]
    for name in ("nzz", "letemps"):          # paywalled; watson.ch is admitted since 2026-10-03 (POLICY §4.7)
        assert name not in B.PUBLISHERS
    assert B.PUBLISHERS["watson"].urls_source and B.PUBLISHERS["watson"].host_rate == 0.5
    # rts has no archive sitemap: it is an explicit-url publisher, fed by section pagination
    assert B.PUBLISHERS["rts"].urls_source and B.PUBLISHERS["rts"].host_rate == 0.1


def test_plan_covers_the_months_and_never_shares_an_as_of_day():
    runs = B.plan(SITEMAP_PUBS, today=TODAY, work=pathlib.Path("/nonexistent"))
    days = [r.as_of for r in runs]
    assert len(days) == len(set(days))
    sw = [r for r in runs if r.publisher == "swissinfo"]
    assert [r.month for r in sw][0] == "2025-10" and [r.month for r in sw][-1] == "2026-07"
    # every as-of day is before the daily timer's first spool (2026-07-21)
    assert max(days) <= "2026-08-29"
    assert {r.publisher: r.window_months for r in runs}["rsi"] == 11


def test_plan_refuses_a_day_inside_the_daily_timers_territory():
    with pytest.raises(SystemExit):
        B.plan(["swissinfo"], today=dt.date(2025, 10, 31))


def test_smoke_runs_are_tiny_and_use_distinct_free_days():
    runs = B.plan(SITEMAP_PUBS, today=TODAY, smoke=True)
    assert {r.publisher: len(r.seeds) for r in runs} == {"swissinfo": 4, "srf": 1, "rsi": 1}
    assert all(r.cap == 25 and r.smoke for r in runs)
    assert len({r.as_of for r in runs}) == 3


def test_queue_row_has_explicit_seeds_and_the_publishers_own_domain():
    run = B.plan(["srf"], today=TODAY)[0]
    row = run.queue()[0]
    assert row["domain"] == "www.srf.ch" and row["sitemaps"] == run.seeds
    assert row["cap"] == 100000 and row["include"] == []


def test_the_queue_is_the_one_publisher_with_no_filler_rows():
    """No other host is queued, not even as a one-page filler: the stage's source
    boundary and the pack arms are narrowed by AW_NEWS_EXPECTED_HOSTS instead."""
    import cytir.news.stage_contract as NSC
    assert not hasattr(B, "filler_seeds")
    for name, pub in SITEMAP_ONLY.items():
        run = B.plan([name], today=TODAY)[0]
        rows = run.queue()
        assert [r["domain"] for r in rows] == [pub.domain]
        assert rows[0]["cap"] == run.cap and rows[0]["sitemaps"] == run.seeds
        assert pub.domain in NSC.PUBLISHERS
        state = {"hosts": {pub.domain: {"status": "ok"}}}
        NSC.validate_sources(state, {"completeness": "time-window-complete"},
                             NSC.parse_expected_hosts(pub.domain))


def test_each_run_names_its_publisher_as_the_expected_host_and_restores_the_env(
        tmp_path, monkeypatch):
    monkeypatch.delenv(B.EXPECTED_HOSTS_ENV, raising=False)
    seen = []
    calls, runner, nic = _fake_world(tmp_path, monkeypatch, {"2025-09-28": 2})

    def spying(day, args):
        seen.append(os.environ.get(B.EXPECTED_HOSTS_ENV))
        runner(day, args)

    run = B.plan(["srf"], today=TODAY, smoke=True)[0]
    B.execute(run, state_dir=tmp_path / "state", runner=spying, iface_bytes=nic)
    assert seen == ["www.srf.ch"]
    assert B.EXPECTED_HOSTS_ENV not in os.environ


def test_the_real_release_tree_honours_expected_hosts():
    for rel in B.EXPECTED_HOSTS_FILES:
        repo_file_or_skip(rel)  # deploy/ is not shipped in the cut
    assert B.release_supports_expected_hosts(HERE.parent) is True


def test_a_release_without_expected_hosts_support_is_refused_before_any_run(
        tmp_path, monkeypatch):
    """r19 and older: the stage script does not know the variable, so a one-publisher
    queue would die inside the stage. The driver refuses up front instead."""
    old = tmp_path / "r19"
    (old / "pipeline").mkdir(parents=True)
    (old / "deploy/scripts").mkdir(parents=True)
    (old / "cytir/sources/families").mkdir(parents=True)
    for rel in B.EXPECTED_HOSTS_FILES:
        (old / rel).write_text("# an older release\n")
    (old / "cytir/news").mkdir(parents=True)
    (old / "cytir/news/stage_contract.py").write_text(
        (HERE.parent / "cytir/news/stage_contract.py").read_text())
    assert B.release_supports_expected_hosts(old) is False
    monkeypatch.setattr(D, "R", old)
    with pytest.raises(SystemExit, match="does not honour AW_NEWS_EXPECTED_HOSTS"):
        B.cmd_smoke(["srf"], tmp_path / "state", TODAY)
    with pytest.raises(SystemExit, match="does not honour AW_NEWS_EXPECTED_HOSTS"):
        B.cmd_run(["srf"], tmp_path / "state", TODAY)
    assert B.release_supports_expected_hosts(tmp_path / "missing") is False


def _fake_world(tmp_path, monkeypatch, rows_by_day):
    qd = tmp_path / "queue"
    qd.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(D, "QUEUE_DIR", qd)
    monkeypatch.setattr(D, "W", work)
    monkeypatch.setattr(B, "free_gb", lambda path=None: 500.0)
    calls = []

    def runner(day, args):
        calls.append((day, args.window_months, os.environ["AW_NEWS_QUEUE"],
                      os.environ["AW_RECENT_CAP"], args.no_status))
        n = rows_by_day.get(day, 0)
        if n is None:
            raise SystemExit("REFUSING: staged 0 points")
        pack = qd / f"news-daily-{day.replace('-', '')}"
        pack.mkdir()
        (pack / "manifest.json").write_text(json.dumps(
            {"rows": n, "urls": n, "tombstones_count": 0}))
        stage = work / f"stage-{day}"
        stage.mkdir()
        (stage / "stage_manifest.json").write_text(json.dumps(
            {"arms": {"swissinfo": {"points_count": n}}}))

    monkeypatch.setattr(D, "staged_points", lambda day: {
        "total": rows_by_day.get(day) or 0, "arms": {}, "manifest_path": ""})
    counter = iter(range(0, 10**12, 500_000))
    return calls, runner, (lambda iface: next(counter))


def test_run_refuses_without_a_passed_smoke(tmp_path, monkeypatch):
    calls, runner, nic = _fake_world(tmp_path, monkeypatch, {})
    with pytest.raises(SystemExit) as exc:
        B.cmd_run(["swissinfo"], tmp_path / "state", TODAY, runner=runner, iface_bytes=nic)
    assert "no passed smoke" in str(exc.value) and not calls


def test_smoke_fails_on_zero_rows_and_blocks_the_full_run(tmp_path, monkeypatch):
    calls, runner, nic = _fake_world(tmp_path, monkeypatch, {"2025-09-29": 0})
    rc = B.cmd_smoke(["swissinfo"], tmp_path / "state", TODAY, runner=runner, iface_bytes=nic)
    assert rc == 1
    st = json.loads((tmp_path / "state" / "state.json").read_text())
    assert st["smoke"]["swissinfo"]["status"] == "FAIL"
    with pytest.raises(SystemExit):
        B.cmd_run(["swissinfo"], tmp_path / "state", TODAY, runner=runner, iface_bytes=nic)


def test_smoke_pass_then_full_run_then_resume_skips_done_runs(tmp_path, monkeypatch):
    months = {r.as_of: 10 for r in B.plan(["srf"], today=TODAY)}
    calls, runner, nic = _fake_world(tmp_path, monkeypatch, {**months, "2025-09-28": 3})
    sd = tmp_path / "state"
    assert B.cmd_smoke(["srf"], sd, TODAY, runner=runner, iface_bytes=nic) == 0
    assert B.cmd_run(["srf"], sd, TODAY, runner=runner, iface_bytes=nic) == 0
    st = json.loads((sd / "state.json").read_text())
    done = [k for k, r in st["runs"].items() if r["status"] == "done"]
    assert len(done) == 1 + 11 and st["runs"]["srf-2026-03"]["rows"] == 10
    assert all(c[4] for c in calls), "--no-status: never overwrite the daily STATUS.json"
    assert all(c[3] == "25" for c in calls[:1]) and calls[1][3] == "100000"
    n = len(calls)
    assert B.cmd_run(["srf"], sd, TODAY, runner=runner, iface_bytes=nic) == 0
    assert len(calls) == n, "a restart must not re-fetch a done run"
    assert "srf-2026-03" in (sd / "NEXT.md").read_text()


def test_a_refusal_is_recorded_as_failed_not_as_an_empty_month(tmp_path, monkeypatch):
    months = {r.as_of: 10 for r in B.plan(["srf"], today=TODAY)}
    months["2025-12-15"] = None
    calls, runner, nic = _fake_world(tmp_path, monkeypatch, {**months, "2025-09-28": 1})
    sd = tmp_path / "state"
    B.cmd_smoke(["srf"], sd, TODAY, runner=runner, iface_bytes=nic)
    assert B.cmd_run(["srf"], sd, TODAY, runner=runner, iface_bytes=nic) == 1
    st = json.loads((sd / "state.json").read_text())
    assert st["runs"]["srf-2025-12"]["status"] == "failed"
    assert "REFUSING" in st["runs"]["srf-2025-12"]["note"]
    assert st["runs"]["srf-2026-01"]["status"] == "done", "one bad month must not stop the rest"


def test_a_taken_as_of_day_is_a_failure_not_an_overwrite(tmp_path, monkeypatch):
    months = {r.as_of: 10 for r in B.plan(["srf"], today=TODAY)}
    calls, runner, nic = _fake_world(tmp_path, monkeypatch, {**months, "2025-09-28": 1})
    (D.QUEUE_DIR / "news-daily-20251115").mkdir()
    sd = tmp_path / "state"
    B.cmd_smoke(["srf"], sd, TODAY, runner=runner, iface_bytes=nic)
    B.cmd_run(["srf"], sd, TODAY, runner=runner, iface_bytes=nic)
    st = json.loads((sd / "state.json").read_text())
    assert st["runs"]["srf-2025-11"]["status"] == "failed"
    assert "taken" in st["runs"]["srf-2025-11"]["note"]


def test_adapt_rate_slows_the_next_run_after_an_over_budget_run():
    pub = B.PUBLISHERS["srf"]
    assert B.adapt_rate(pub, None, 1.0) == 1.0
    assert B.adapt_rate(pub, 0.6, 1.0) == 1.0
    assert B.adapt_rate(pub, 2.0, 1.0) == 0.45
    assert B.adapt_rate(pub, 50.0, 1.0) == 0.2


def test_unknown_publisher_is_refused_at_the_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path))  # stranger mode: the state dir derives from it
    with pytest.raises(SystemExit):
        B.main(["plan", "nzz"])


def test_only_the_days_own_arm_collections_are_ever_dropped():
    gone = []
    assert B.drop_staging_collections("2025-09-28", delete=gone.append) == gone
    assert gone == ["aw_dnews256_20250928_chmedia", "aw_dnews256_20250928_swissinfo",
                    "aw_dnews256_20250928_hn"]


def test_a_failed_runs_spool_is_set_aside_so_the_retry_starts_clean(tmp_path, monkeypatch):
    calls, runner, nic = _fake_world(tmp_path, monkeypatch, {"2025-09-28": 2})
    run = B.plan(["srf"], today=TODAY, smoke=True)[0]
    spool = D.W / f"stage-{run.as_of}"
    spool.mkdir()
    (spool / "state.json").write_text("{}")
    state = {"runs": {run.key: {"status": "failed"}}}
    dropped = []
    moved = B.set_aside_failed_spool(run, state, delete=dropped.append)
    assert moved and not spool.exists() and pathlib.Path(moved).is_dir()
    assert len(dropped) == 3 and all("20250928" in d for d in dropped)
    assert B.occupied(run.as_of) == []


def test_a_spool_with_a_pack_or_not_failed_by_this_state_is_never_touched(tmp_path, monkeypatch):
    _fake_world(tmp_path, monkeypatch, {})
    run = B.plan(["srf"], today=TODAY, smoke=True)[0]
    spool = D.W / f"stage-{run.as_of}"
    spool.mkdir()
    assert B.set_aside_failed_spool(run, {"runs": {}}) is None          # not ours
    assert B.set_aside_failed_spool(run, {"runs": {run.key: {"status": "done"}}}) is None
    (D.QUEUE_DIR / f"news-daily-{run.as_of.replace('-', '')}").mkdir()
    assert B.set_aside_failed_spool(run, {"runs": {run.key: {"status": "failed"}}}) is None
    assert spool.exists()


def test_the_controller_steers_on_the_runs_own_fetched_bytes_not_the_whole_nic(tmp_path, monkeypatch):
    calls, runner, _ = _fake_world(tmp_path, monkeypatch, {"2025-09-28": 2})
    nic = iter([0, 900_000_000])            # 900 MB on the card: other jobs
    clock = iter([0.0, 10.0])               # the run took 10 s
    monkeypatch.setattr(B.time, "monotonic", lambda: next(clock))
    run = B.plan(["srf"], today=TODAY, smoke=True)[0]
    spool = D.W / f"stage-{run.as_of}"
    (spool / "pages").mkdir(parents=True)
    (spool / "pages" / "a.html").write_bytes(b"x" * 2_000_000)
    (spool / "hn_linked").mkdir()
    (spool / "hn_linked" / "b.html").write_bytes(b"y" * 1_000_000)
    entry = B.execute(run, state_dir=tmp_path / "state", runner=lambda d, a: None,
                      iface_bytes=lambda iface: next(nic))
    assert entry["fetched_mb"] == 3.0
    assert entry["egress_mbps"] == 0.3 and entry["nic_mbps_whole_card"] == 90.0
    assert B.adapt_rate(B.PUBLISHERS["srf"], entry["egress_mbps"], 1.0) == 1.0


def test_a_full_spool_disk_stops_the_run_before_any_fetch(tmp_path, monkeypatch):
    months = {r.as_of: 10 for r in B.plan(["srf"], today=TODAY)}
    calls, runner, nic = _fake_world(tmp_path, monkeypatch, {**months, "2025-09-28": 1})
    sd = tmp_path / "state"
    B.cmd_smoke(["srf"], sd, TODAY, runner=runner, iface_bytes=nic)
    n = len(calls)
    monkeypatch.setattr(B, "free_gb", lambda path=None: 10.0)
    assert B.cmd_run(["srf"], sd, TODAY, runner=runner, iface_bytes=nic) == 1
    assert len(calls) == n
    st = json.loads((sd / "state.json").read_text())
    assert any("REFUSING" in r.get("note", "") for r in st["runs"].values())


def test_a_run_left_running_by_a_dead_driver_is_failed_then_retried_clean(tmp_path, monkeypatch):
    months = {r.as_of: 10 for r in B.plan(["srf"], today=TODAY)}
    calls, runner, nic = _fake_world(tmp_path, monkeypatch, {**months, "2025-09-28": 1})
    sd = tmp_path / "state"
    B.cmd_smoke(["srf"], sd, TODAY, runner=runner, iface_bytes=nic)
    state = json.loads((sd / "state.json").read_text())
    state["runs"]["srf-2025-10"] = {"as_of": "2025-10-15", "status": "running", "publisher": "srf"}
    (sd / "state.json").write_text(json.dumps(state))
    spool = D.W / "stage-2025-10-15"
    spool.mkdir()
    dropped = []
    monkeypatch.setattr(B, "drop_staging_collections",
                        lambda day, delete=None: dropped.append(day) or [])
    assert B.cmd_run(["srf"], sd, TODAY, runner=runner, iface_bytes=nic) == 0
    st = json.loads((sd / "state.json").read_text())
    assert st["runs"]["srf-2025-10"]["status"] == "done"
    assert dropped == ["2025-10-15"]
    assert list(D.W.glob("stage-2025-10-15.failed-*"))


# ---------------------------------------------------------------------------------------
# swissinfo tail (ger, fre, ita of 2026-09-19..09-29) and the page-date clamp guard
# ---------------------------------------------------------------------------------------
def _urlset(*urls):
    return ("<?xml version='1.0'?><urlset>"
            + "".join(f"<url><loc>{u}</loc><lastmod>2026-09-20T10:00:00Z</lastmod></url>"
                      for u in urls) + "</urlset>")


def _fake_fetch(table, calls=None):
    def fetch(url):
        if calls is not None:
            calls.append(url)
        return table(url)
    return fetch


def _art(lang, n):
    return f"https://www.swissinfo.ch/{lang}/politik/artikel-{n}/{n}"


def test_the_tail_has_33_day_sitemaps_three_languages_eleven_days():
    seeds = B.swissinfo_tail_seeds()
    assert len(seeds) == 33
    assert {lang for lang, _, _ in seeds} == {"ger", "fre", "ita"}, "eng is already held"
    assert seeds[0][1] == "2026-09-19" and seeds[-1][1] == "2026-09-29"
    assert "https://www.swissinfo.ch/ger/sitemap-2026.xml?mm=09&dd=19" == seeds[0][2]
    assert all("&amp;" not in u for _, _, u in seeds)


def test_discovery_gives_every_seed_one_terminal_and_keeps_the_urls(tmp_path):
    def table(url):
        lang = url.split("/")[3]
        if "dd=21" in url and lang == "fre":
            return 503, ""
        if "dd=22" in url and lang == "ita":
            return 200, _urlset()                       # a readable, empty day
        if "dd=23" in url and lang == "ger":
            return 200, _urlset("https://www.swissinfo.ch/ger/sitemap-2026.xml?mm=09&dd=23&p=2")
        if "dd=24" in url and lang == "ger":
            return 0, ""
        if "dd=25" in url and lang == "ita":
            return -1, ""
        n = int(url.split("dd=")[1])
        return 200, _urlset(_art(lang, n), _art(lang, n + 100))
    acct = B.swissinfo_tail_discover(tmp_path, _fake_fetch(table))
    t = acct["terminals"]
    assert sum(t.values()) == acct["seeds"] == 33
    assert t["http_503"] == 1 and t["empty"] == 1 and t["nested_sitemap"] == 1
    assert t["error"] == 1 and t["robots_disallowed"] == 1 and t["ok"] == 28
    assert not acct["clean"], "an unread seed is never a clean discovery"
    urls = [json.loads(line) for line in (tmp_path / "urls.jsonl").read_text().splitlines()]
    assert len(urls) == 28 * 2 and len({u["url"] for u in urls}) == len(urls)
    assert all(f"/{u['lang']}/" in u["url"] for u in urls)


def test_discovery_resumes_reading_only_what_is_not_ok_or_empty(tmp_path):
    def flaky(url):
        return (200, _urlset(_art("ger", 1))) if "/ger/" in url else (0, "")
    first = B.swissinfo_tail_discover(tmp_path, _fake_fetch(flaky))
    assert first["terminals"]["ok"] == 11 and first["terminals"]["error"] == 22
    calls = []
    second = B.swissinfo_tail_discover(tmp_path, _fake_fetch(
        lambda url: (200, _urlset(_art(url.split("/")[3], len(calls)))), calls))
    assert len(calls) == 22, "the 11 ok seeds were not fetched again"
    assert second["clean"] and second["terminals"] == {"ok": 33}
    urls = [json.loads(line)["url"] for line in (tmp_path / "urls.jsonl").read_text().splitlines()]
    assert len(urls) == len(set(urls))


def _tail_state(tmp_path, n_per_seed=2):
    disc = tmp_path / "state" / "discovery"
    B.swissinfo_tail_discover(disc, _fake_fetch(lambda url: (200, _urlset(*[
        _art(url.split("/")[3], int(url.split("dd=")[1]) * 100 + i)
        for i in range(n_per_seed)]))))
    return tmp_path / "state"


def test_the_tail_runs_one_explicit_url_run_on_a_free_past_day(tmp_path):
    sd = _tail_state(tmp_path)
    (run,) = B.plan(["swissinfo-tail"], today=TODAY, state_dir=sd)
    assert run.as_of == B.SWISSINFO_TAIL_ASOF == "2026-07-17" and not run.blocked
    assert len(run.urls) == 33 * 2 and run.seeds == []
    (row,) = run.queue()
    assert row["domain"] == "www.swissinfo.ch" and row["urls"] == run.urls and "sitemaps" not in row
    (smoke,) = B.plan(["swissinfo-tail"], today=TODAY, smoke=True, state_dir=sd)
    assert smoke.smoke and smoke.as_of == "2025-09-26" and len(smoke.urls) == 25
    assert smoke.as_of != run.as_of


def test_the_tail_refuses_an_unclean_or_missing_discovery(tmp_path):
    with pytest.raises(SystemExit, match="discover swissinfo-tail"):
        B.plan(["swissinfo-tail"], today=TODAY, state_dir=tmp_path)
    disc = tmp_path / "discovery"
    B.swissinfo_tail_discover(disc, _fake_fetch(lambda url: (503, "")))
    with pytest.raises(SystemExit, match="not clean"):
        B.plan(["swissinfo-tail"], today=TODAY, state_dir=tmp_path)


def test_no_two_runs_share_an_as_of_day_across_publishers(tmp_path):
    sd = _tail_state(tmp_path)
    days = [B.SWISSINFO_TAIL_ASOF, B.RTS_TAIL_ASOF, B.SWISSINFO_TAIL_SMOKE_ASOF,
            B.RTS_JULY_LAST_FREE_ASOF, B.RTS_SMOKE_ASOF, "2026-07-19", "2026-07-20",
            "2026-08-29"]
    assert len(set(days)) == len(days)
    assert all(d < B.DAILY_TERRITORY_START or d == "2026-08-29" or d in ("2026-07-19", "2026-07-20")
               for d in days)
    assert B.SWISSINFO_TAIL_ASOF < B.DAILY_TERRITORY_START and B.RTS_TAIL_ASOF < B.DAILY_TERRITORY_START


def _release(tmp_path, *, clamp):
    rel = tmp_path / "rel" / "pipeline"
    rel.mkdir(parents=True)
    text = "def select_recent(): pass\n"
    if clamp:
        text += ("def page_date_clamp_day(now_ref, today=None): return today\n"
                 "x = published_for_page(html, now_ref=page_date_clamp_day(ctx))\n")
    else:
        text += "x = published_for_page(html, now_ref=ctx.get('now_ref'))\n"
    (rel / "sitemap_ingest.py").write_text(text)
    return rel.parent


def test_a_release_that_clamps_page_dates_to_the_as_of_day_is_refused(tmp_path):
    assert B.release_supports_page_date_clamp(_release(tmp_path / "a", clamp=True))
    assert not B.release_supports_page_date_clamp(_release(tmp_path / "b", clamp=False))
    assert not B.release_supports_page_date_clamp(tmp_path / "nowhere")
    with pytest.raises(SystemExit, match="clamps page dates to the as-of day"):
        B.require_page_date_clamp_release(_release(tmp_path / "c", clamp=False))
    B.require_page_date_clamp_release(_release(tmp_path / "d", clamp=True))


def test_r22_pin_carries_the_clamp():
    """The pinned release of the checkout under test must honour the clamp once the pin is
    r22: read from the tree the pin names, only when that tree exists on this box."""
    try:
        pinned = pathlib.Path(D.R)
    except SettingRefused:
        pytest.skip("no pinned release is configured (stranger run)")
    if not pinned.is_dir():
        pytest.skip("pinned release not on this box")
    if D.R.name.endswith(("-r19", "-r20", "-r21")):
        pytest.skip("pin predates r22")
    assert B.release_supports_page_date_clamp(D.R)


def test_next_md_carries_the_exact_resume_command(tmp_path, monkeypatch):
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "ops@example.org")
    months = {r.as_of: 10 for r in B.plan(["srf"], today=TODAY)}
    calls, runner, nic = _fake_world(tmp_path, monkeypatch, {**months, "2025-09-28": 3})
    sd = tmp_path / "state"
    B.cmd_smoke(["srf"], sd, TODAY, runner=runner, iface_bytes=nic)
    B.cmd_run(["srf"], sd, TODAY, runner=runner, iface_bytes=nic)
    cmd = json.loads((sd / "state.json").read_text())["resume_command"]
    assert f"{sys.executable} -m cytir.news.coverage_backfill run srf --state-dir {sd}" in cmd
    assert "AW_OPERATOR_CONTACT=ops@example.org" in cmd
    assert cmd in (sd / "NEXT.md").read_text()


def test_infra_failure_reads_the_note_and_the_stage_log():
    assert B.infra_failure("staging qdrant lifecycle check FAILED (exit=2) -- see /x")
    note = "stage FAILED for day=2026-05-31 (exit=1) -- see /log"
    assert B.infra_failure(note, read=lambda p: "FATAL news stage contract: staging Qdrant unavailable at x")
    assert not B.infra_failure(note, read=lambda p: "ok\nREFUSING: staged 0 points")
    assert not B.infra_failure("REFUSING: staged 0 points")


def test_a_dead_staging_qdrant_stops_the_driver_instead_of_failing_every_month(tmp_path, monkeypatch):
    months = {r.as_of: 10 for r in B.plan(["srf"], today=TODAY)}
    calls, runner, nic = _fake_world(tmp_path, monkeypatch, {**months, "2025-09-28": 1})
    sd = tmp_path / "state"
    B.cmd_smoke(["srf"], sd, TODAY, runner=runner, iface_bytes=nic)

    def dead(day, args):
        raise SystemExit("staging qdrant lifecycle check FAILED (exit=2) -- see /x")
    assert B.cmd_run(["srf"], sd, TODAY, runner=dead, iface_bytes=nic) == 1
    st = json.loads((sd / "state.json").read_text())
    failed = [k for k, r in st["runs"].items() if r["status"] == "failed"]
    assert failed == ["srf-2025-10"], "one failed month, the rest not attempted"
    assert B.cmd_run(["srf"], sd, TODAY, runner=runner, iface_bytes=nic) == 0
    assert json.loads((sd / "state.json").read_text())["runs"]["srf-2025-10"]["status"] == "done"
