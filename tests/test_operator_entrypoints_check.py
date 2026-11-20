"""`deploy/operator/check_entrypoints.py` and the refreshed `probe-entrypoints.json` (review of #567).

The fixture is the live runner list read from the box on 2026-10-04 (`ps` + `/proc/<pid>/cwd`): the
govinfo bulk run, the watson chain and run, the RTS second pass, the FR and GOV.UK API families, the
SO driver. Every one of them must have a row; the table as it stood before the refresh must not.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

from tests.helpers.repo_files import repo_file_or_skip

APP = pathlib.Path(__file__).resolve().parents[1]
# `deploy/operator` is withheld from the public cut: the module skips there
OP = repo_file_or_skip("deploy/operator/check_entrypoints.py").parent
_spec = importlib.util.spec_from_file_location("aw_check_entrypoints", OP / "check_entrypoints.py")
CE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CE)

# The runner roots and the interpreter come from the table's own `vars`, so the fixture names no box path.
_VARS = json.loads((OP / "probe-entrypoints.json").read_text())["vars"]
G3 = _VARS["G3"]
BAU7 = _VARS["BAU7"]
VENV = _VARS["VENV"]
WORK = BAU7.rsplit("/", 1)[0]       # the sibling work dirs the runners' state lives in
LIVE = f"""\
857343 {WORK}/news_backfill_watson bash -c cd {WORK}/news_backfill_watson && setsid nohup bash chain.sh > chain.out 2>&1 < /dev/null & sleep 45
857345 {BAU7}/repo-watson/authoritative-web-app bash chain.sh
1080840 {BAU7}/repo-watson/authoritative-web-app {VENV} -m cytir.news.coverage_backfill run watson --state-dir {WORK}/news_backfill_watson
2303167 {BAU7}/repo-rts3/authoritative-web-app {VENV} -m cytir.news.coverage_backfill run rts --generation 2
3936472 {BAU7}/repo-rts3/authoritative-web-app bash {WORK}/news_backfill_700_rts2/chain.sh
2409205 {G3}/fr4/authoritative-web-app bash state/run_govinfo.sh
2409259 {G3}/fr4/authoritative-web-app {VENV} -m cytir.sources.run_windows us_federal_register_ecfr --opt leg=govinfo
3889045 {G3}/fr3/authoritative-web-app bash state/run_api_family.sh fr
3889048 {G3}/fr3/authoritative-web-app {VENV} -m cytir.sources.run_windows us_federal_register_ecfr --opt leg=fr
4098955 {G3}/fr3/authoritative-web-app bash state/run_api_family.sh govuk
2489015 {G3}/main/authoritative-web-app {VENV} -m cytir.ops.stackoverflow_tranches --state-dir {WORK}/so-rest-691
"""


def table() -> dict:
    return json.loads((OP / "probe-entrypoints.json").read_text())


def test_every_live_runner_of_the_box_has_a_row():
    live = CE.parse_live(LIVE)
    assert len(live) == 11 and all(cwd for _, cwd, _ in live)
    assert CE.missing_rows(live, table()) == []


def test_the_table_before_the_refresh_misses_the_runs_that_had_no_row(tmp_path):
    old = table()
    for row in old["entry_points"]:
        if row["name"] == "api-family-fr":
            row["cwd"] = "$G3/fr/authoritative-web-app"                       # the stale cwd
        if row["name"] == "news-backfill":
            row["cwd"] = f"{WORK}/bau-backfill-repo/authoritative-web-app"
    old["entry_points"] = [r for r in old["entry_points"]
                           if r["name"] not in ("news-backfill-watson-chain", "news-backfill-rts2",
                                                "api-family-govinfo-bulk")]
    missing = {cwd for _, cwd, _ in CE.missing_rows(CE.parse_live(LIVE), old)}
    assert missing == {f"{BAU7}/repo-watson/authoritative-web-app", f"{BAU7}/repo-rts3/authoritative-web-app",
                       f"{G3}/fr3/authoritative-web-app", f"{G3}/fr4/authoritative-web-app"}


def test_the_launcher_shell_is_not_a_runner_and_an_unreadable_cwd_is_reported():
    live = CE.parse_live("11 /x bash -c cd /y && bash chain.sh\n12  python -m cytir.sources.run_windows x\n")
    assert [p for p, _, _ in CE.missing_rows(live, table())] == ["12"]


def test_the_rows_the_review_named_carry_the_scripts_they_run():
    rows = {r["name"]: r for r in table()["entry_points"]}
    assert "cytir.news.hn_daily" in rows["news-daily-queue"]["runs"]
    assert "cytir.ops.cc_tranche_delta" in rows["cron-cc-seal-watchdog"]["runs"]
    assert rows["api-family-fr"]["cwd"] == "$G3/fr3/authoritative-web-app"
    assert {"news-backfill-watson-chain", "news-backfill-rts2", "api-family-govinfo-bulk"} <= set(rows)


def test_live_cwd_glob_covers_a_pinned_release():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("ce", pathlib.Path(__file__).parents[1] / "deploy/operator/check_entrypoints.py")
    ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)
    doc = {"vars": {"G": "/r/*/app"}, "entry_points": [{"name": "a", "cwd": "/r/current/app", "live_cwd_glob": "$G"}]}
    live = [("1", "/r/abc123/app", "bash x"), ("2", "/elsewhere", "bash y")]
    assert [m[0] for m in ce.missing_rows(live, doc)] == ["2"]
