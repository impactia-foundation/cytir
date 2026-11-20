#!/usr/bin/env python3
"""Run one command with immutable launch and terminal outcome evidence.

The caller may detach this supervisor with ``setsid``/``nohup``.  Each attempt
gets a fresh directory so failed pre-publication runs remain observable and no
later retry can overwrite their evidence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file


SCHEMA_LAUNCH = "aw.detached-one-shot-launch.v1"
SCHEMA_OUTCOME = "aw.detached-one-shot-launch-outcome.v1"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _artifact_uri(path: Path, artifact_root: Path) -> str:
    resolved = path.resolve()
    root = artifact_root.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact path is outside global artifact root: {path}") from exc
    return f"artifact://{relative.as_posix()}"


def _file_artifact(path: Path, artifact_root: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "uri": _artifact_uri(path, artifact_root),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _normalized_rc(returncode: int) -> int:
    return returncode if returncode >= 0 else 128 + (-returncode)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    attempt_root = args.attempt_root.resolve()
    artifact_root = args.artifact_root.resolve()
    if not artifact_root.is_dir():
        raise ValueError(f"global artifact root is not a directory: {artifact_root}")
    _artifact_uri(attempt_root, artifact_root)
    attempt_root.mkdir(parents=True, exist_ok=False)
    _fsync_directory(attempt_root.parent)

    stdout_path = attempt_root / "stdout.log"
    stderr_path = attempt_root / "stderr.log"
    launch_path = attempt_root / "LAUNCH.json"
    outcome_path = attempt_root / "OUTCOME.json"
    started_at = _utc_now()
    supervisor_pid = os.getpid()
    child: subprocess.Popen[bytes] | None = None
    received_signal: int | None = None

    # Exclusive log creation makes the whole attempt immutable by construction.
    with stdout_path.open("xb", buffering=0) as stdout, \
            stderr_path.open("xb", buffering=0) as stderr:
        try:
            child = subprocess.Popen(
                args.command,
                cwd=args.cwd,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        except BaseException as exc:
            finished_at = _utc_now()
            failure = {
                "argv": args.command,
                "cwd": str(args.cwd.resolve()) if args.cwd else None,
                "error": {"message": str(exc), "type": type(exc).__name__},
                "finished_at": finished_at,
                "pid": None,
                "rc": 127,
                "release": args.release,
                "schema": SCHEMA_OUTCOME,
                "started_at": started_at,
                "status": "spawn_failed",
                "stderr": _file_artifact(stderr_path, artifact_root),
                "stdout": _file_artifact(stdout_path, artifact_root),
                "supervisor_pid": supervisor_pid,
            }
            atomic_publish(outcome_path, _json_bytes(failure))
            _fsync_directory(attempt_root)
            return 127

        def forward(signum: int, _frame: Any) -> None:
            nonlocal received_signal
            received_signal = signum
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    # The child exited between poll and signal delivery.  wait()
                    # below remains the sole authority for the terminal rc.
                    pass

        # Install the forwarding handlers BEFORE publishing LAUNCH.json, not
        # after. LAUNCH.json is the documented external signal that this
        # supervisor is ready to be signaled; a caller (or this module's own
        # test) may poll for it and send SIGTERM/SIGINT the instant it exists.
        # Publishing it first left a real race window where a signal arriving
        # between the write and the `signal.signal(...)` call hit Python's
        # default disposition (the process dies outright, no forward, no
        # OUTCOME.json status, `wait()` returns the raw negative signal
        # instead of the documented 128+signal `rc`) -- reproduced on a
        # loaded CI runner (task 532, CI baseline task530) though never
        # locally.
        previous = {
            signum: signal.signal(signum, forward)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        launch = {
            "argv": args.command,
            "cwd": str(args.cwd.resolve()) if args.cwd else None,
            "pid": child.pid,
            "release": args.release,
            "schema": SCHEMA_LAUNCH,
            "started_at": started_at,
            "stderr_uri": _artifact_uri(stderr_path, artifact_root),
            "stdout_uri": _artifact_uri(stdout_path, artifact_root),
            "supervisor_pid": supervisor_pid,
        }
        atomic_publish(launch_path, _json_bytes(launch))
        _fsync_directory(attempt_root)
        try:
            child_returncode = child.wait()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())

    finished_at = _utc_now()
    rc = _normalized_rc(child_returncode)
    outcome = {
        "argv": args.command,
        "child_returncode": child_returncode,
        "cwd": str(args.cwd.resolve()) if args.cwd else None,
        "finished_at": finished_at,
        "launch": _file_artifact(launch_path, artifact_root),
        "pid": child.pid,
        "rc": rc,
        "received_signal": signal.Signals(received_signal).name
        if received_signal is not None else None,
        "release": args.release,
        "schema": SCHEMA_OUTCOME,
        "started_at": started_at,
        "status": "succeeded" if rc == 0 else "failed",
        "stderr": _file_artifact(stderr_path, artifact_root),
        "stdout": _file_artifact(stdout_path, artifact_root),
        "supervisor_pid": supervisor_pid,
    }
    atomic_publish(outcome_path, _json_bytes(outcome))
    _fsync_directory(attempt_root)
    return rc


if __name__ == "__main__":
    sys.exit(main())
