"""The ONE unattended refresh-scheduler entrypoint (task 632; ported from bash by task 2176).

    python3 -m cytir.sources.refresh_direct.lane_run

Runs from a SINGLE cron entry on the fetch box (see the refresh-lane deploy README) through
`deploy/scripts/refresh-lane-run-scheduled.sh`. Unlike the news lane, driven by a cron
over SSH, this lane does its real work locally on the fetch box ("all fetch/compute there,
never on the orchestrator", task 632 brief).

What ONE run does, in order:
  1. new-refs discovery  (`cytir.sources.refresh_direct.new_refs`)  -- daily
  2. periodic re-fetch   (`cytir.sources.refresh_direct.periodic`)  -- per-source cadence
  3. news-lane call-out  -- OPTIONAL, OFF unless `AW_NEWS_LANE_CALLOUT` names an executable and
     the lane is armed; under the PROD-WRITE-LOCK, taken only around this step.
  4. queue every due, non-news pack this run could not apply itself (this lane NEVER writes
     prod for a non-news family, armed or not).
  5. append one line to the ledger (read by deploy/scripts/aw-monitor-check.sh), mail the
     operator when something needs attention.

ARMED defaults OFF (`AW_REFRESH_LANE_ARMED=0`): every run is a dry-run report until an operator
flips it. This module never arms itself and must never be crontab'd with the flag pre-set to 1.

Same environment variables, log lines, file names, ledger line and exit codes as the
bash driver it replaces: a failed step exits with that step's code, a busy
single-flight lock exits 0, a finished run exits 0.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[3]
QUEUE_SCHEMA = "aw.refresh-lane-queue.v1"


def _now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _queued(report: dict, lane: str) -> list[dict]:
    out = []
    for e in report["sources"]:
        if e.get("status") in ("not_implemented", "consent_declined_stop", "checked"):
            row = dict(e)
            row["lane"] = lane
            out.append(row)
    return out


def build_queue(new_refs: dict, periodic: dict) -> dict:
    return {"schema": QUEUE_SCHEMA, "generated_at": periodic["generated_at"],
            "items": _queued(new_refs, "new_refs") + _queued(periodic, "periodic_refetch")}


def ledger_entry(new_refs: dict, periodic: dict, queue: dict, *, armed: bool,
                 news_status: str, stamp: str) -> dict:
    return {
        "run": stamp,
        "armed": armed,
        "new_refs_due": new_refs["due_count"],
        "new_refs_alarm": new_refs["alarm_count"],
        "periodic_due": periodic["due_count"],
        "periodic_alarm": periodic["alarm_count"],
        "periodic_consent_declined": periodic["consent_declined_count"],
        "queued_items": len(queue["items"]),
        "news_callout_status": news_status,
    }


def _alert_recipient(env: dict, log) -> str:
    """The summary-mail recipient: `AW_MAILTO` (its `CYTIR_MAILTO` twin is mirrored by
    `cytir.core.envnames`; empty or blank falls through), else the operator contact
    (`cytir.core.user_agent.operator()`, which reads the PROCESS environment and counts as
    configured only with URL and contact), else "" -- no code default. Never raises: a broken
    operator config costs the optional mail, not the run."""
    explicit = (env.get("AW_MAILTO") or "").strip()
    if explicit:
        return explicit
    try:
        from cytir.core import user_agent  # noqa: PLC0415
        _url, contact, configured = user_agent.operator()
    except Exception as exc:  # noqa: BLE001
        log(f"WARN: operator contact lookup failed ({type(exc).__name__}) -- no summary mail")
        return ""
    return contact if configured else ""


def main(argv: list[str] | None = None, env: dict | None = None) -> int:
    env = dict(os.environ if env is None else env)
    repo = Path(env.get("AW_REFRESH_LANE_REPO") or REPO_DIR)
    if not env.get("AW_REFRESH_LANE_STATE_DIR"):
        print("AW_REFRESH_LANE_STATE_DIR: set it (the cron wrapper exports the box default)",
              file=sys.stderr)
        return 2
    state_dir = Path(env["AW_REFRESH_LANE_STATE_DIR"])
    cadence = env.get("AW_REFRESH_LANE_CADENCE_TABLE") or str(repo / "data/refresh-cadence.v1.json")
    due_state = env.get("AW_REFRESH_LANE_DUE_STATE") or str(state_dir / "state.json")
    ledger = Path(env.get("AW_REFRESH_LANE_LEDGER") or state_dir / "ledger.ndjson")
    logdir = Path(env.get("AW_REFRESH_LANE_LOGDIR") or state_dir / "logs")
    cache_dir = env.get("AW_REFRESH_LANE_CONSENT_CACHE") or str(state_dir / "consent-cache")
    queue_dir = Path(env.get("AW_REFRESH_LANE_QUEUE_DIR") or state_dir / "queue")
    run_lock = env.get("AW_REFRESH_LANE_RUN_LOCK") or str(state_dir / ".refresh-lane-run.lock")
    prod_lock = env.get("AW_PROD_WRITE_LOCK") or ""
    armed = (env.get("AW_REFRESH_LANE_ARMED") or "0") == "1"
    armed_s = "1" if armed else "0"
    now = env.get("AW_NOW") or _now_z()
    py = env.get("AW_REFRESH_LANE_PY") or "python3"

    for d in (state_dir, logdir, queue_dir):
        d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runlog = logdir / f"run-{stamp}.log"

    def log(msg: str) -> None:
        line = f"{_now_z()} {msg}"
        print(line, flush=True)
        with open(runlog, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # single-flight: never overlap a slow cron tick with the next one.
    lock_fh = open(run_lock, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("another refresh-lane run holds the lock -- skip")
        return 0

    log(f"refresh-lane START armed={armed_s} now={now} state_dir={state_dir}")
    child_env = dict(env)
    child_env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    armed_flag = ["--armed"] if armed else []

    def step(module: str, *args: str) -> int:
        with open(runlog, "ab") as out:
            return subprocess.run([py, "-m", module, *args], env=child_env,
                                  stdout=out, stderr=subprocess.STDOUT).returncode

    new_refs_report = state_dir / f"new-refs-{stamp}.json"
    log(f"step1: new-refs discovery -> {new_refs_report}")
    rc = step("cytir.sources.refresh_direct.new_refs", "--cadence-table", cadence, "--state", due_state,
              "--out", str(new_refs_report), *armed_flag)
    if rc:
        return rc

    periodic_report = state_dir / f"periodic-{stamp}.json"
    log(f"step2: periodic re-fetch -> {periodic_report}")
    rc = step("cytir.sources.refresh_direct.periodic", "--cadence-table", cadence, "--state", due_state,
              "--resolver-cache-dir", cache_dir, "--out", str(periodic_report), *armed_flag)
    if rc:
        return rc

    # 3. News-lane call-out: integration point only; the prod-write lock covers only this step.
    news_status = "not-wired"
    callout = env.get("AW_NEWS_LANE_CALLOUT", "")
    if armed and callout:
        if os.path.isfile(callout) and os.access(callout, os.X_OK) and not prod_lock:
            news_status = "callout-no-prod-write-lock"
            log("WARN: AW_PROD_WRITE_LOCK is unset -- the call-out is not run without its lock")
        elif os.path.isfile(callout) and os.access(callout, os.X_OK):
            log(f"step3: news-lane call-out ({callout}), under PROD-WRITE-LOCK")
            lock8 = open(prod_lock, "w")
            try:
                fcntl.flock(lock8, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                news_status = "prod-write-lock-busy"
                log("WARN: PROD-WRITE-LOCK held by another writer -- skipped this run's news call-out")
            else:
                with open(runlog, "ab") as out:
                    ok = subprocess.run([callout], env=env, stdout=out,
                                        stderr=subprocess.STDOUT).returncode == 0
                if ok:
                    news_status = "called-ok"
                else:
                    news_status = "called-failed"
                    log(f"WARN: news-lane call-out exited non-zero (see {runlog})")
                fcntl.flock(lock8, fcntl.LOCK_UN)
            finally:
                lock8.close()
        else:
            news_status = "callout-not-executable"
            log(f"WARN: AW_NEWS_LANE_CALLOUT={callout} is not an executable file")
    else:
        log("step3: news-lane call-out not wired (AW_NEWS_LANE_CALLOUT unset, or lane not armed)"
            " -- expected until task 614 lands")

    # 4. Queue.
    new_refs = json.loads(new_refs_report.read_text(encoding="utf-8"))
    periodic = json.loads(periodic_report.read_text(encoding="utf-8"))
    queue_file = queue_dir / f"queue-{stamp}.json"
    queue = build_queue(new_refs, periodic)
    queue_file.write_text(json.dumps(queue, indent=2), encoding="utf-8")
    queued_n = len(queue["items"])
    log(f"step4: queued {queued_n} item(s) -> {queue_file}")

    # 5. Ledger + summary mail.
    ledger_line = json.dumps(ledger_entry(new_refs, periodic, queue, armed=armed,
                                          news_status=news_status, stamp=stamp), sort_keys=True)
    with open(ledger, "a", encoding="utf-8") as fh:
        fh.write(ledger_line + "\n")
    log(f"step5: ledger += {ledger_line}")

    alarm_n = new_refs["alarm_count"] + periodic["alarm_count"]
    mailer = shutil.which("msmtp", path=env.get("PATH", ""))
    mailto = ""
    if alarm_n > 0 or queued_n > 0:
        if not env.get("PATH"):
            log("WARN: no summary mail sent: the environment has no PATH, so msmtp is not looked up")
        elif mailer:
            mailto = _alert_recipient(env, log)
            if not mailto:
                log("WARN: no summary mail sent: neither AW_MAILTO nor the operator contact is configured")
    if mailto:
        body = "\n".join([
            f"To: {mailto}",
            f"Subject: [AW] refresh-lane {stamp} -- armed={armed_s} queued={queued_n} "
            f"alarm={alarm_n} news={news_status}",
            "",
            "Refresh-lane run (task 632). This lane NEVER writes prod for non-news",
            "families -- everything below is queued for review, not applied.",
            "",
            ledger_line,
            "",
            f"New-refs report:  {new_refs_report}",
            f"Periodic report:  {periodic_report}",
            f"Queue file:       {queue_file}",
            f"Log:              {runlog}",
            "",
        ])
        with open(runlog, "ab") as err:
            sent = subprocess.run([mailer, "-t"], input=body.encode(), stderr=err, env=env).returncode == 0
        if not sent:
            log("WARN: mail send failed")

    log(f"refresh-lane DONE armed={armed_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
