import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path




def _run(tmp_path: Path, name: str, code: str) -> tuple[subprocess.CompletedProcess, Path]:
    attempt = tmp_path / name
    result = subprocess.run(
        [
            sys.executable,
            "-m", "cytir.ops.run_observed_one_shot",
            "--attempt-root",
            str(attempt),
            "--artifact-root",
            str(tmp_path),
            "--release",
            "release-test-sha",
            "--",
            sys.executable,
            "-c",
            code,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result, attempt


def test_success_persists_exact_immutable_launch_and_outcome(tmp_path: Path) -> None:
    result, attempt = _run(
        tmp_path, "success", "import sys; print('complete'); print('note', file=sys.stderr)"
    )
    assert result.returncode == 0
    launch = json.loads((attempt / "LAUNCH.json").read_text())
    outcome = json.loads((attempt / "OUTCOME.json").read_text())
    assert launch["schema"] == "aw.detached-one-shot-launch.v1"
    assert outcome["schema"] == "aw.detached-one-shot-launch-outcome.v1"
    assert outcome["status"] == "succeeded"
    assert outcome["rc"] == outcome["child_returncode"] == 0
    assert outcome["pid"] == launch["pid"]
    assert outcome["argv"] == launch["argv"]
    assert outcome["release"] == launch["release"] == "release-test-sha"
    assert (attempt / "stdout.log").read_text() == "complete\n"
    assert (attempt / "stderr.log").read_text() == "note\n"
    assert outcome["stdout"]["uri"] == "artifact://success/stdout.log"
    assert outcome["stderr"]["uri"] == "artifact://success/stderr.log"
    assert outcome["launch"]["uri"] == "artifact://success/LAUNCH.json"
    second, _ = _run(tmp_path, "success", "raise SystemExit(0)")
    assert second.returncode != 0
    assert json.loads((attempt / "OUTCOME.json").read_text()) == outcome


def test_prepublication_failure_keeps_logs_and_terminal_rc(tmp_path: Path) -> None:
    result, attempt = _run(
        tmp_path,
        "failure",
        "import sys; print('gate failed', file=sys.stderr); raise SystemExit(23)",
    )
    assert result.returncode == 23
    outcome = json.loads((attempt / "OUTCOME.json").read_text())
    assert outcome["status"] == "failed"
    assert outcome["rc"] == outcome["child_returncode"] == 23
    assert outcome["started_at"].endswith("Z")
    assert outcome["finished_at"].endswith("Z")
    assert (attempt / "stderr.log").read_text() == "gate failed\n"
    assert (attempt / "stdout.log").read_bytes() == b""


def test_spawn_failure_is_terminal_evidence(tmp_path: Path) -> None:
    attempt = tmp_path / "spawn-failure"
    result = subprocess.run(
        [
            sys.executable,
            "-m", "cytir.ops.run_observed_one_shot",
            "--attempt-root",
            str(attempt),
            "--artifact-root",
            str(tmp_path),
            "--release",
            "release-test-sha",
            "--",
            "/definitely/missing/command",
        ],
        check=False,
    )
    assert result.returncode == 127
    assert not (attempt / "LAUNCH.json").exists()
    outcome = json.loads((attempt / "OUTCOME.json").read_text())
    assert outcome["status"] == "spawn_failed"
    assert outcome["pid"] is None
    assert outcome["rc"] == 127
    assert outcome["error"]["type"] == "FileNotFoundError"


def test_signal_is_forwarded_and_terminal_outcome_is_published(tmp_path: Path) -> None:
    attempt = tmp_path / "signaled"
    supervisor = subprocess.Popen(
        [
            sys.executable,
            "-m", "cytir.ops.run_observed_one_shot",
            "--attempt-root",
            str(attempt),
            "--artifact-root",
            str(tmp_path),
            "--release",
            "release-test-sha",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
        ]
    )
    deadline = time.monotonic() + 5
    while not (attempt / "LAUNCH.json").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert (attempt / "LAUNCH.json").exists()
    os.kill(supervisor.pid, signal.SIGTERM)
    assert supervisor.wait(timeout=5) == 143
    outcome = json.loads((attempt / "OUTCOME.json").read_text())
    assert outcome["status"] == "failed"
    assert outcome["received_signal"] == "SIGTERM"
    assert outcome["child_returncode"] == -signal.SIGTERM
    assert outcome["rc"] == 143
