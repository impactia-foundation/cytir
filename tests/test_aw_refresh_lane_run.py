"""The refresh-lane scheduler entrypoint (task 632, ported to Python by task 2176).

The two step modules are replaced by a stub interpreter that writes a canned report to `--out`,
so no network, cadence table or Heimdall path is touched.
"""
from __future__ import annotations

import fcntl
import json
import stat
import sys
from pathlib import Path

from cytir.sources.refresh_direct import lane_run

STUB = """#!%s
import json, sys
a = sys.argv[1:]
out = a[a.index("--out") + 1]
mod = a[1]
src = [{"id": "x", "status": "checked"}, {"id": "y", "status": "up_to_date"}]
rep = {"generated_at": "2026-11-01T00:00:00Z", "sources": src, "due_count": 2,
       "alarm_count": 1, "consent_declined_count": 0}
json.dump(rep, open(out, "w"))
"""


def _env(tmp_path: Path, **extra) -> dict:
    stub = tmp_path / "stubpy"
    stub.write_text(STUB % sys.executable)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    env = {
        "AW_REFRESH_LANE_STATE_DIR": str(tmp_path / "state"),
        "AW_PROD_WRITE_LOCK": str(tmp_path / "PROD-WRITE-LOCK"),
        "AW_REFRESH_LANE_PY": str(stub),
        "PATH": "/nonexistent",  # no msmtp: the mail branch stays dark (lane_run looks in THIS PATH)
        "AW_MAILTO": "nobody@example.invalid",
    }
    env.update(extra)
    return env


def test_unarmed_run_writes_ledger_queue_and_exits_zero(tmp_path):
    assert lane_run.main([], env=_env(tmp_path)) == 0
    state = tmp_path / "state"
    line = json.loads((state / "ledger.ndjson").read_text().splitlines()[0])
    assert line["armed"] is False
    assert line["queued_items"] == 2  # one "checked" row from each report
    assert line["news_callout_status"] == "not-wired"
    assert line["new_refs_alarm"] == 1 and line["periodic_due"] == 2
    queue = json.loads(next((state / "queue").glob("queue-*.json")).read_text())
    assert queue["schema"] == "aw.refresh-lane-queue.v1"
    assert {i["lane"] for i in queue["items"]} == {"new_refs", "periodic_refetch"}
    log = next((state / "logs").glob("run-*.log")).read_text()
    assert "refresh-lane START armed=0" in log and "refresh-lane DONE armed=0" in log


def test_callout_runs_only_when_armed_and_executable(tmp_path):
    callout = tmp_path / "callout.sh"
    callout.write_text("#!/bin/sh\nexit 0\n")
    callout.chmod(0o755)
    env = _env(tmp_path, AW_REFRESH_LANE_ARMED="1", AW_NEWS_LANE_CALLOUT=str(callout))
    assert lane_run.main([], env=env) == 0
    line = json.loads((tmp_path / "state" / "ledger.ndjson").read_text().splitlines()[-1])
    assert line["armed"] is True and line["news_callout_status"] == "called-ok"


def test_callout_not_executable_and_lock_busy(tmp_path):
    plain = tmp_path / "plain"
    plain.write_text("x")
    env = _env(tmp_path, AW_REFRESH_LANE_ARMED="1", AW_NEWS_LANE_CALLOUT=str(plain))
    assert lane_run.main([], env=env) == 0
    ledger = tmp_path / "state" / "ledger.ndjson"
    assert json.loads(ledger.read_text().splitlines()[-1])["news_callout_status"] == "callout-not-executable"

    ok = tmp_path / "ok.sh"
    ok.write_text("#!/bin/sh\nexit 0\n")
    ok.chmod(0o755)
    held = open(tmp_path / "PROD-WRITE-LOCK", "w")
    fcntl.flock(held, fcntl.LOCK_EX)
    env = _env(tmp_path, AW_REFRESH_LANE_ARMED="1", AW_NEWS_LANE_CALLOUT=str(ok))
    assert lane_run.main([], env=env) == 0
    assert json.loads(ledger.read_text().splitlines()[-1])["news_callout_status"] == "prod-write-lock-busy"
    held.close()


def test_single_flight_lock_skips_with_exit_zero(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    held = open(state / ".refresh-lane-run.lock", "w")
    fcntl.flock(held, fcntl.LOCK_EX)
    assert lane_run.main([], env=_env(tmp_path)) == 0
    assert not (state / "ledger.ndjson").exists()
    held.close()


def test_failed_step_propagates_its_exit_code(tmp_path):
    env = _env(tmp_path)
    Path(env["AW_REFRESH_LANE_PY"]).write_text("#!/bin/sh\nexit 7\n")
    assert lane_run.main([], env=env) == 7


def test_state_dir_is_required(tmp_path, capsys):
    env = _env(tmp_path)
    del env["AW_REFRESH_LANE_STATE_DIR"]
    assert lane_run.main([], env=env) == 2
    assert "AW_REFRESH_LANE_STATE_DIR" in capsys.readouterr().err


def test_broken_operator_config_costs_the_mail_not_the_run(tmp_path, monkeypatch):
    from cytir.core import user_agent

    def boom():
        raise RuntimeError("operator.json unreadable")

    calls = tmp_path / "calls"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sentinel = bindir / "msmtp"
    sentinel.write_text(f"#!/bin/sh\necho called >> {calls}\ncat > /dev/null\n")
    sentinel.chmod(0o755)
    monkeypatch.setattr(user_agent, "operator", boom)
    env = _env(tmp_path, PATH=str(bindir))
    env.pop("AW_MAILTO")
    assert lane_run.main([], env=env) == 0
    log = next((tmp_path / "state" / "logs").glob("run-*.log")).read_text()
    assert "operator contact lookup failed (RuntimeError)" in log
    assert not calls.exists()


def test_no_path_in_env_means_no_mail_and_says_so(tmp_path):
    env = _env(tmp_path)
    env.pop("PATH")
    assert lane_run.main([], env=env) == 0
    log = next((tmp_path / "state" / "logs").glob("run-*.log")).read_text()
    assert "environment has no PATH" in log


def test_mail_is_sent_only_through_the_given_path_never_a_real_binary(tmp_path, monkeypatch):
    """lane_run looks for msmtp in the PATH of the env it is given, not the process PATH. A sentinel
    msmtp records calls: the process PATH finds it but the env PATH does not -> no call; the env PATH
    finds it -> exactly one call (so no test can reach a real mail binary by accident)."""
    calls = tmp_path / "calls"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sentinel = bindir / "msmtp"
    sentinel.write_text(f"#!/bin/sh\necho called >> {calls}\ncat > /dev/null\n")
    sentinel.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir) + ":/usr/bin:/bin")
    assert lane_run.main([], env=_env(tmp_path)) == 0
    assert not calls.exists()
    second = tmp_path / "second"
    second.mkdir()
    assert lane_run.main([], env=_env(second, PATH=str(bindir))) == 0
    assert calls.read_text().strip() == "called"
