"""`cyt news routine-apply`: armed on purpose, refusing by default (task 2379 port of the shell driver).

The news lane has been handoff-only since task 500, and this script is what
lets it apply on its own.  Everything here is about the two ways that can go
wrong without anyone noticing: applying when it should not (an unarmed run, a
due battery, a missing input read as a default), and the cron silently
carrying on as if it had applied when it had not.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cytir.news import routine_apply as RA
from tests.helpers.repo_files import repo_file_or_skip

ROOT = Path(__file__).resolve().parents[1]
CRON_REL = "deploy/scripts/news_refresh_cron.sh"


def env_for(tmp_path: Path, **overrides) -> dict:
    driver = tmp_path / "driver.sh"
    driver.write_text("#!/usr/bin/env bash\necho \"driver $1\" >> "
                      f"{tmp_path}/steps.txt\n")
    driver.chmod(0o755)
    ledger = tmp_path / "cadence.jsonl"
    ledger.write_text(json.dumps({
        "schema": "aw.news-gate-cadence.v1", "event": "battery",
        # One day ago, not a fixed date: a fixed date turned this test red once the
        # 7-day battery cadence elapsed on the wall clock (main red 2026-09-24).
        "at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        "release_sha": "a" * 40}) + "\n")
    for name in ("pack.jsonl.gz", "sv.json", "tp.json"):
        (tmp_path / name).write_text("{}")
    env = {
        **os.environ,
        "AW_RELEASE_DIR": str(ROOT),
        "PYTHONPATH": str(ROOT),
        "AW_PROD_LOCK_HOST": "lockhost",
        "AW_QPY": sys.executable,
        "AW_NEWS_APPLY_DRIVER": str(driver),
        "AW_NEWS_CADENCE_LEDGER": str(ledger),
        "AW_TASK598_EV": str(tmp_path / "evidence"),
        "AW_NEWS_WORK": str(tmp_path),
        "AW_PACK_NAME": "news598-20260916",
        "AW_NEWS_TWIN_URL": "http://127.0.0.1:6457",
        "AW_NEWS_TWIN_COLLECTION": "aw_e8",
        "AW_PROD_URL": "http://127.0.0.1:6334",
        "AW_PROD_COLLECTION": "aw_prod",
        "AW_QVEC_CACHE": str(tmp_path / "sv.json"),
        "AW_QVEC_QUERY_SOURCE_MANIFEST": str(tmp_path / "sv.json"),
        "ART": str(tmp_path / "pack.jsonl.gz"),
        "SV": str(tmp_path / "sv.json"),
        "TP": str(tmp_path / "tp.json"),
        "ROWS": "14819",
        "STAGED_COLLECTION": "aw_dnews256_20260916_chmedia",
    }
    env.update(overrides)
    return env


def run(args, env, **kwargs):
    return subprocess.run([sys.executable, "-m", "cytir.cli", "news", "routine-apply", *args], env=env,
                          text=True, capture_output=True, timeout=300, cwd=ROOT, **kwargs)


def test_the_dry_run_prints_the_ordered_plan(tmp_path):
    result = run(["--dry-run"], env_for(tmp_path))
    assert result.returncode == 0, result.stderr
    lines = [line.strip() for line in result.stdout.splitlines()]
    plan = [line for line in lines if line.startswith("driver:")]
    assert plan[0].endswith("delta")
    assert plan[-1].endswith("verify")
    # The daily baseline arm is cut BETWEEN the sealing steps and the steps
    # that bind it: `preflight-dry` refuses without BASELINE_ARM.
    arm_at = next(i for i, line in enumerate(lines)
                  if line.startswith("daily-gate baseline arm:"))
    seal_at = next(i for i, line in enumerate(lines)
                   if line.endswith("baseline-refs"))
    plan_at = next(i for i, line in enumerate(lines)
                   if line.endswith("preflight-dry"))
    assert seal_at < arm_at < plan_at
    assert "apply-armed" not in result.stdout.replace("apply-armed-daily", "")
    assert "--gate-profile news-daily" in result.stdout
    assert '"battery_due": false' in result.stdout


def test_an_unarmed_run_applies_nothing(tmp_path):
    """task626 (Tim, 2026-09-21): exit 0 here was indistinguishable, to a
    caller checking only `$?`, from a real applied pack -- a cron/monitor
    saw "success" whether this ran for real or did nothing at all. A
    distinct, non-zero exit code (4) now says "nothing happened" on its
    own, without needing to also parse stdout for DISARMED."""
    env = env_for(tmp_path)
    env.pop("AW_NEWS_AUTO_APPLY", None)
    result = run([], env)
    assert result.returncode == 4
    assert "DISARMED" in result.stdout
    assert "NOTHING APPLIED" in result.stdout
    assert not (tmp_path / "steps.txt").exists()


def test_a_due_battery_refuses_before_any_work(tmp_path):
    """The refusal costs no fetch, no embed and no twin write."""
    env = env_for(tmp_path, AW_NEWS_AUTO_APPLY="1")
    ledger = Path(env["AW_NEWS_CADENCE_LEDGER"])
    rows = [json.loads(ledger.read_text())]
    for index in range(7):
        rows.append({"schema": "aw.news-gate-cadence.v1", "event": "pack",
                     "at": (datetime.now(timezone.utc) - timedelta(hours=20 - index)).isoformat(),
                     "release_sha": "a" * 40, "rows": 1000})
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = run([], env)
    assert result.returncode == 2
    assert "battery is due" in result.stdout
    assert not (tmp_path / "steps.txt").exists()


def test_a_missing_cadence_ledger_refuses(tmp_path):
    env = env_for(tmp_path, AW_NEWS_AUTO_APPLY="1",
                  AW_NEWS_CADENCE_LEDGER=str(tmp_path / "absent.jsonl"))
    result = run([], env)
    assert result.returncode != 0
    assert not (tmp_path / "steps.txt").exists()


@pytest.mark.parametrize("missing", [
    "AW_RELEASE_DIR", "AW_NEWS_APPLY_DRIVER", "AW_NEWS_CADENCE_LEDGER",
    "ART", "SV", "TP", "ROWS", "AW_PACK_NAME"])
def test_every_input_is_required_not_defaulted(tmp_path, missing):
    env = env_for(tmp_path, AW_NEWS_AUTO_APPLY="1")
    env.pop(missing)
    result = run(["--dry-run"], env)
    assert result.returncode != 0
    assert missing in result.stderr


def test_the_cron_is_disarmed_by_default():
    """Merging this PR changes nothing that runs on the box."""
    text = repo_file_or_skip(CRON_REL).read_text()
    assert '"${AW_NEWS_AUTO_APPLY:-0}" = "1"' in text
    assert "APPLY_STATUS=\"handoff-only\"" in text


def test_the_cron_reports_what_it_did_not_what_it_used_to_do():
    text = repo_file_or_skip(CRON_REL).read_text()
    assert "SUBJECT_STATE" in text
    assert "GATE REFUSED" in text
    assert 'if [ "$APPLY_STATUS" = "gate-refused" ]; then exit 2; fi' in text


def test_the_cron_applies_only_through_the_tracked_driver():
    """No second apply path: step 3b calls this script, nothing else."""
    text = repo_file_or_skip(CRON_REL).read_text()
    assert "python3 -m cytir.cli news routine-apply" in text
    assert "AW_NEWS_ROUTINE_ENV" in text


def _stub_qdrant(points_count: int):
    """A GET /collections/<name> that answers like qdrant, so the routine's
    live-count read succeeds and the driver loop is actually reached."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"result": {"points_count": points_count}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_a_failing_driver_step_names_itself_and_writes_a_failed_row(tmp_path):
    """task626 (2026-09-22, 09-12 second stop): `preflight-dry` refused with
    its cause one unlabeled line up and a bare exit 1, while the caller held
    both write locks over a dry run. A failed step must now say WHICH step
    failed, point at its cause, leave a FAILED row in the transaction's own
    ledger, stop everything after it, and propagate the step's exit code."""
    server, url = _stub_qdrant(33736546)
    try:
        env = env_for(tmp_path, AW_NEWS_AUTO_APPLY="1",
                      AW_NEWS_TWIN_URL=url, AW_PROD_URL=url)
        driver = Path(env["AW_NEWS_APPLY_DRIVER"])
        driver.write_text(
            "#!/usr/bin/env bash\n"
            f"echo \"driver $1\" >> {tmp_path}/steps.txt\n"
            "if [ \"$1\" = pack ]; then\n"
            "  echo 'twin explicit source-group tombstones drifted'\n"
            "  exit 7\n"
            "fi\n")
        result = run([], env)
    finally:
        server.shutdown()
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAILED: driver step pack exit=7" in result.stdout
    assert "tombstones drifted" in result.stdout  # the cause stays visible
    steps = (tmp_path / "steps.txt").read_text().split()
    assert "pack" in steps and "durability" not in steps  # nothing after it ran
    ledger = tmp_path / "evidence" / "apply-armed-daily-ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    assert rows[-1]["status"] == "FAILED"
    assert rows[-1]["step"] == "pack"
    assert rows[-1]["exit"] == 7
    assert rows[-1]["pack"] == env["AW_PACK_NAME"]


def test_the_preflight_requires_the_module_and_the_cron_runs_it_from_the_release():
    from cytir.news.release_preflight import REQUIRED_RELEASE_FILES
    assert "cytir/news/routine_apply.py" in REQUIRED_RELEASE_FILES
    assert (ROOT / "cytir/news/routine_apply.py").is_file()
    text = repo_file_or_skip(CRON_REL).read_text()
    assert "cd '$REMOTE_RELEASE' &&" in text
    assert "AW_RELEASE_DIR='$REMOTE_RELEASE' AW_NEWS_AUTO_APPLY=1 AW_NOW='$NOW' python3 -m cytir.cli news routine-apply" in text


def _aw(*args):
    return subprocess.run([sys.executable, "-m", "cytir.cli", *args], capture_output=True, text=True,
                          cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})


def test_the_door_is_reached_from_aw_news_help():
    for args in (["news", "--help"],):
        out = _aw(*args)
        assert out.returncode == 0, out.stderr
        assert "routine-apply" in out.stdout
    out = _aw("news", "routine-apply", "--help")
    assert out.returncode == 0 and "--dry-run" in out.stdout


def test_the_transaction_argv_threads_the_prod_baseline_and_opt_in_flags(tmp_path):
    """ADR-043 am.7: the prod baseline arm always reaches the transaction; stop-after-twin,
    waiver and reuse-proof flags are opt-in; the gate profile is news-daily."""
    import inspect
    src = inspect.getsource(RA.apply_routine)
    assert src.index("cut_prod_baseline") < src.index("transaction_argv")
    base, prod = tmp_path / "b.json", tmp_path / "p.json"
    base.write_text("{}")
    prod.write_text("{}")
    argv = RA.transaction_argv(RA.Cfg(env_for(tmp_path)), base, prod, "")
    assert "--prod-baseline-arm" in argv and "--prod-baseline-arm-sha256" in argv
    assert "--stop-after-twin-gate" not in argv and "--waiver-ref" not in argv
    assert "--prod-baseline-reuse-proof" not in argv
    assert argv[argv.index("--gate-profile") + 1] == "news-daily"
    assert argv[-2:] == ["--arm", "APPLY-CONTENT-PACK"]
    env2 = env_for(tmp_path, AW_STOP_AFTER_TWIN_GATE="1", AW_WAIVER_REF="r", AW_WAIVER_ISSUER="tim")
    argv2 = RA.transaction_argv(RA.Cfg(env2), base, prod, "/x/reuse-check.json")
    assert "--stop-after-twin-gate" in argv2
    assert argv2[argv2.index("--waiver-issuer") + 1] == "tim"
    assert argv2[argv2.index("--prod-baseline-reuse-proof") + 1] == "/x/reuse-check.json"


def test_a_waiver_without_an_issuer_refuses_before_any_work(tmp_path):
    result = run([], env_for(tmp_path, AW_NEWS_AUTO_APPLY="1", AW_WAIVER_REF="r"))
    assert result.returncode == 1 and "AW_WAIVER_ISSUER" in result.stderr
    assert not (tmp_path / "steps.txt").exists()


def test_an_armed_run_without_the_work_dir_refuses_up_front(tmp_path):
    env = env_for(tmp_path, AW_NEWS_AUTO_APPLY="1")
    env.pop("AW_NEWS_WORK")
    result = run([], env)
    assert result.returncode == 1 and "AW_NEWS_WORK" in result.stderr
    assert not (tmp_path / "steps.txt").exists()


def test_latest_arm_picks_the_latest_on_a_reused_round_dir(tmp_path):
    """task 626: a re-attempt against the SAME evidence dir leaves an earlier arm file behind;
    the cutter takes the LATEST by mtime instead of asserting exactly one."""
    assert RA.latest_arm(tmp_path) is None
    older = tmp_path / "news-daily-gate-arm.attempt1.aaaa.json"
    newer = tmp_path / "news-daily-gate-arm.attempt2.bbbb.json"
    older.write_text("{}")
    newer.write_text("{}")
    os.utime(older, (1, 1))
    assert RA.latest_arm(tmp_path) == newer


def test_fail_row_appends_a_failed_row(tmp_path):
    RA.fail_row(tmp_path / "ev", "pack", 7, "p1")
    row = json.loads((tmp_path / "ev" / "apply-armed-daily-ledger.jsonl").read_text())
    assert (row["status"], row["step"], row["exit"], row["pack"]) == ("FAILED", "pack", 7, "p1")


def test_count_of_reads_the_live_count_and_empty_when_unreadable():
    server, url = _stub_qdrant(42)
    try:
        assert RA.count_of(url, "c") == "42"
    finally:
        server.shutdown()
    assert RA.count_of("http://127.0.0.1:9", "c") == ""


def test_print_three_readings_reads_the_verdict_log(tmp_path, capsys):
    v = tmp_path / "v.json"
    v.write_text(json.dumps({"metrics": {"h@1": {"before": 5, "after": 7, "attributable": 2}},
                             "candidate": "cand", "baseline_disagreement": 3}))
    arm = tmp_path / "a.json"
    arm.write_text(json.dumps({"identity": "idA"}))
    RA.print_three_readings(v, arm, arm)
    out = capsys.readouterr().out
    assert "THREE READINGS" in out and "net=+2" in out and "disagreement: 3" in out


def test_full_digest_counts_up_then_runs_every_nth_and_stops_on_a_mismatch(tmp_path, monkeypatch):
    counter = tmp_path / "state" / "c.txt"
    cfg = RA.Cfg(env_for(tmp_path, AW_FULL_DIGEST_COUNTER_FILE=str(counter), AW_FULL_DIGEST_EVERY="2"))
    RA.full_digest_step(cfg)
    assert counter.read_text().strip() == "1"
    server, url = _stub_qdrant(10)
    try:
        cfg.twin_url = cfg.prod_url = url
        for shas, anomaly in ((("a", "a"), False), (("a", "b"), True)):
            counter.write_text("1\n")

            def fake(cfg_, role, url_, coll, points, digest_dir, _s=dict(zip(("twin", "prod"), shas))):
                digest_dir.mkdir(parents=True, exist_ok=True)
                (digest_dir / f"{role}-receipt.json").write_text(json.dumps({"logical_membership_sha256": _s[role]}))
            monkeypatch.setattr(RA, "_membership", fake)
            if not anomaly:
                RA.full_digest_step(cfg)
                assert counter.read_text().strip() == "0"
            else:
                with pytest.raises(RA.Refusal) as stop:
                    RA.full_digest_step(cfg)
                assert stop.value.code == 3 and "FULL-DIGEST ANOMALY" in stop.value.message
    finally:
        server.shutdown()


def _stub_layout(monkeypatch):
    import cytir.index.layout as CL
    monkeypatch.setattr(CL, "fingerprint", lambda url, coll: {"points_count": 100, "layout": "x"})
    monkeypatch.setattr(CL, "quiescent", lambda before, now: (True, "stub-quiet"))


def _reuse(tmp_path, monkeypatch, release_name, state):
    _stub_layout(monkeypatch)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state_file = tmp_path / "state.json"
    if state is not None:
        state_file.write_text(json.dumps({**state, "layout_fingerprint": {"points_count": 100, "layout": "x"},
                                          "points_count": 100}))
    allowed = RA.check_prod_baseline_reuse(tmp_path / release_name, "http://x", "aw_prod", state_file, out_dir)
    check = out_dir / "reuse-check.json"
    return allowed, (json.loads(check.read_text()) if check.is_file() else None)


def test_reuse_refuses_across_a_release_boundary_even_with_identical_layout(tmp_path, monkeypatch):
    arm = tmp_path / "prev-arm.json"
    arm.write_text("{}")
    state = {"pack": "p", "arm": str(arm), "release": "task625-news-catchup-051aadba-r16"}
    allowed, proof = _reuse(tmp_path, monkeypatch, "task625-news-catchup-cdc90788-r17", state)
    assert allowed is False and proof["reused"] is False and "different release" in proof["reason"]


def test_reuse_proceeds_within_the_same_release(tmp_path, monkeypatch):
    arm = tmp_path / "prev-arm.json"
    arm.write_text("{}")
    allowed, proof = _reuse(tmp_path, monkeypatch, "r17", {"pack": "p", "arm": str(arm), "release": "r17"})
    assert allowed is True and proof["reused"] is True
    assert (tmp_path / "out" / "prod-baseline-arm.json").is_file()


def test_reuse_falls_through_on_no_state_and_on_a_broken_state(tmp_path, monkeypatch):
    allowed, proof = _reuse(tmp_path, monkeypatch, "r17", None)
    assert allowed is False and "no persisted" in proof["reason"]
    (tmp_path / "state.json").write_text("{not json")
    assert RA.check_prod_baseline_reuse(tmp_path / "r17", "u", "c", tmp_path / "state.json", tmp_path / "out") is False


def test_postapply_persists_the_release_name_for_tomorrows_reuse_check(tmp_path, monkeypatch):
    from cytir.gates import convergence as GC
    _stub_layout(monkeypatch)
    arm = tmp_path / "arm.json"
    arm.write_text("{}")
    monkeypatch.setattr(GC, "converged_arms", lambda cut, **kw: {"arm": str(arm)})
    monkeypatch.setattr(RA, "make_cutter", lambda *a, **k: None)
    cfg = RA.Cfg(env_for(tmp_path, AW_RELEASE_DIR=str(tmp_path / "rel-name")))
    state_file = tmp_path / "state" / "s.json"
    RA.persist_postapply_reading(cfg, tmp_path / "post", state_file)
    state = json.loads(state_file.read_text())
    assert state["release"] == "rel-name" and state["schema"] == RA.REUSE_STATE_SCHEMA
    assert state["points_count"] == 100 and (tmp_path / "post" / "prod-postapply-arm.json").is_file()


def test_the_routine_runs_a_python_driver_with_python3_and_a_shell_driver_with_bash(tmp_path):
    server, url = _stub_qdrant(33736546)
    try:
        for kind in ("py", "sh"):
            work = tmp_path / kind
            work.mkdir()
            env = env_for(work, AW_NEWS_AUTO_APPLY="1", AW_NEWS_TWIN_URL=url, AW_PROD_URL=url)
            steps = work / "steps.txt"
            driver = work / f"driver.{kind}"
            if kind == "py":
                driver.write_text("import sys\nopen(%r, 'a').write('py ' + sys.argv[1] + '\\n')\n"
                                  "if sys.argv[1] == 'pack': sys.exit(7)\n" % str(steps))
            else:
                driver.write_text("#!/usr/bin/env bash\necho \"sh $1\" >> %r\n"
                                  "if [ \"$1\" = pack ]; then exit 7; fi\n" % str(steps))
            env["AW_NEWS_APPLY_DRIVER"] = str(driver)
            result = run([], env)
            assert result.returncode == 1, (kind, result.stdout[-400:] + result.stderr[-400:])
            assert "FAILED: driver step pack exit=7" in result.stdout
            assert f"{kind} pack" in steps.read_text().split("\n")
    finally:
        server.shutdown()


# ---- the armed path past the seal steps, in-process (review round 1) -------------------------------

def _armed(tmp_path, monkeypatch, *, count="100", converge=True, txn_rc=0, post_status_rc=0, **extra):
    """Run `main([])` armed with every child process, count read and convergence faked.
    Returns (rc, events, envs, digest_calls, postapply_calls)."""
    from cytir.gates import convergence as GC
    import cytir.index.layout as CL
    env = env_for(tmp_path, AW_NEWS_AUTO_APPLY="1", AW_NEWS_APPLY_DRIVER=str(tmp_path / "driver.py"), **extra)
    monkeypatch.setattr(os, "environ", dict(env))
    events, envs, digest, post = [], {}, [], []

    def fake_run(argv, **kw):
        child_env = dict(kw.get("env") or os.environ)
        if "status" in argv:
            events.append(("status",))
            late = sum(1 for e in events if e == ("txn",))
            return subprocess.CompletedProcess(argv, post_status_rc if late else 0, '{"battery_due": false}\n', "")
        if "arm" in argv:
            ident = argv[argv.index("--identity") + 1]
            events.append(("arm", ident))
            out = Path(argv[argv.index("--out-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "news-daily-gate-arm.x.json").write_text(json.dumps({"identity": ident}))
            return subprocess.CompletedProcess(argv, 0)
        if "cytir.index.apply" in argv:
            events.append(("txn",))
            envs["txn"] = child_env
            return subprocess.CompletedProcess(argv, txn_rc)
        step = argv[-1]
        events.append(("driver", step))
        envs[step] = child_env
        return subprocess.CompletedProcess(argv, 0)

    def converged(cut, **kw):
        if not converge:
            raise GC.ConvergenceUnknown("arms cannot be compared")
        return {"arm": str(cut(0))}

    monkeypatch.setattr(RA.subprocess, "run", fake_run)
    monkeypatch.setattr(RA, "count_of", lambda url, coll: count)
    monkeypatch.setattr(RA, "gate_env", lambda cfg, url, coll: dict(os.environ))
    monkeypatch.setattr(GC, "converged_arms", converged)
    monkeypatch.setattr(CL, "fingerprint", lambda url, coll: {"points_count": 1})
    monkeypatch.setattr(RA, "full_digest_step", lambda cfg: (digest.append(1), events.append(("digest",))))
    monkeypatch.setattr(RA, "persist_postapply_reading", lambda *a: (post.append(1), events.append(("postapply",))))
    return RA.main([]), events, envs, digest, post


def _failed_rows(tmp_path):
    ledger = tmp_path / "evidence" / "apply-armed-daily-ledger.jsonl"
    return [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.is_file() else []


def test_the_armed_path_runs_in_order_and_hands_each_child_its_environment(tmp_path, monkeypatch):
    rc, events, envs, digest, post = _armed(tmp_path, monkeypatch)
    assert rc == 0
    names = [e[0] if e[0] != "driver" else e[1] for e in events]
    assert names[0] == "status"
    assert names[1:10] == list(RA.SEAL_STEPS)
    assert names[10] == "arm" and names[11] == "arm"
    assert names[12:17] == list(RA.PLAN_STEPS)
    assert names[17] == "txn" and names[18] == "status"
    # the digest runs before the post-apply reading (item 3b)
    assert names[19:] == ["digest", "postapply"]
    # the preflight dry run gets the live point counts (item 3a)
    assert envs["preflight-dry"]["AW_EXPECT_POINTS"] == envs["preflight-dry"]["AW_EXPECT_POINTS_PROD"] == "100"
    assert digest == [1] and post == [1]
    assert "daily-preapply" in events[10][1] and "daily-prod-baseline-r0" in events[11][1]
    # live counts are in the seal steps' environment; the arms are in the plan steps' and the transaction's
    assert envs["baseline-twin"]["AW_EXPECT_POINTS"] == "100"
    assert envs["baseline-prod"]["AW_EXPECT_POINTS_PROD"] == "100"
    assert "BASELINE_ARM" not in envs["baseline-refs"]
    ev = tmp_path / "evidence"
    for key in ("preflight-dry", "txn"):
        e = envs[key]
        assert e["BASELINE_ARM"].startswith(str(ev / "gate" / "daily-preapply"))
        assert e["PROD_BASELINE_ARM"] == str(ev / "gate" / "daily-prod-baseline" / "prod-baseline-arm.json")
    t = envs["txn"]
    assert t["AW_ARTIFACT_ROOT"] == str(tmp_path)
    assert t["AW_CONTENT_PACK_PRESERVATION_INTENT"] == str(ev / "intent.json")
    assert t["AW_CONTENT_PACK_PRESERVATION_PROFILE_REGISTRY"].endswith("source-preservation-profiles.v1.json")
    assert t["AW_CONTENT_PACK_PRESERVATION_DEEP"] == "1"
    assert t["AW_EXPECT_POINTS"] == t["AW_EXPECT_POINTS_PROD"] == "100"
    assert t["PYTHONPATH"] == str(ROOT)


def test_an_unreadable_live_count_refuses_before_any_driver_step(tmp_path, monkeypatch):
    rc, events, *_ = _armed(tmp_path, monkeypatch, count="")
    assert rc == 2
    assert [e for e in events if e[0] != "status"] == []


def test_a_prod_baseline_that_does_not_converge_refuses_before_the_plan_steps(tmp_path, monkeypatch):
    rc, events, envs, digest, post = _armed(tmp_path, monkeypatch, converge=False)
    assert rc == 2
    ran = [e[1] for e in events if e[0] == "driver"]
    assert ran == list(RA.SEAL_STEPS) and "txn" not in envs
    assert digest == [] and post == []


def test_a_failed_transaction_writes_its_row_and_skips_digest_and_postapply(tmp_path, monkeypatch):
    rc, events, envs, digest, post = _armed(tmp_path, monkeypatch, txn_rc=5)
    assert rc == 1
    row = _failed_rows(tmp_path)[-1]
    assert (row["status"], row["step"], row["exit"]) == ("FAILED", "transaction", 5)
    assert digest == [] and post == []
    assert events[-1] == ("txn",)


def test_stop_after_twin_gate_skips_digest_and_postapply(tmp_path, monkeypatch):
    rc, events, envs, digest, post = _armed(tmp_path, monkeypatch, AW_STOP_AFTER_TWIN_GATE="1")
    assert rc == 0 and ("txn",) in events
    assert digest == [] and post == []


def test_a_release_dir_that_is_not_the_running_release_refuses_before_any_work(tmp_path):
    other = tmp_path / "other-release"
    other.mkdir()
    result = run([], env_for(tmp_path, AW_NEWS_AUTO_APPLY="1", AW_RELEASE_DIR=str(other)))
    assert result.returncode == 1 and "AW_RELEASE_DIR" in result.stderr
    assert not (tmp_path / "steps.txt").exists() and not (tmp_path / "evidence").exists()


def test_the_contract_path_comes_from_the_data_home_of_the_running_release(tmp_path, monkeypatch):
    import cytir.core.paths as paths
    release = tmp_path / "release"
    monkeypatch.setattr(paths, "data_dir", lambda: release / "data")
    contract = RA.Cfg(env_for(tmp_path)).contract
    assert str(contract).startswith(str(release))
    assert contract.name == "news-daily-gate-contract.v1.json"


def test_a_failed_post_land_status_exits_1_and_skips_digest_and_postapply(tmp_path, monkeypatch, capsys):
    rc, events, envs, digest, post = _armed(tmp_path, monkeypatch, post_status_rc=3)
    assert rc == 1
    assert "FAILED: post-apply cadence status exit=3" in capsys.readouterr().err
    assert digest == [] and post == []


def test_an_empty_live_count_refuses_the_full_digest(tmp_path, monkeypatch):
    counter = tmp_path / "state" / "c.txt"
    cfg = RA.Cfg(env_for(tmp_path, AW_FULL_DIGEST_COUNTER_FILE=str(counter), AW_FULL_DIGEST_EVERY="1"))
    monkeypatch.setattr(RA, "count_of", lambda url, coll: "")
    monkeypatch.setattr(RA, "_membership", lambda *a: pytest.fail("no membership read on an empty count"))
    with pytest.raises(RA.Refusal) as stop:
        RA.full_digest_step(cfg)
    assert stop.value.code == 1 and "point count" in stop.value.message


def test_missing_operator_settings_refuse_before_any_evidence_is_written(tmp_path, monkeypatch):
    import cytir.core.paths as paths
    env = env_for(tmp_path, AW_NEWS_AUTO_APPLY="1")
    del env["AW_PROD_LOCK_HOST"]
    monkeypatch.setattr(os, "environ", dict(env))

    def refused():
        raise paths.SettingRefused("prod_lock_host is not configured")
    monkeypatch.setattr(paths, "operator_settings", refused)
    assert RA.main([]) == 1
    assert not (tmp_path / "steps.txt").exists() and not (tmp_path / "evidence").exists()
