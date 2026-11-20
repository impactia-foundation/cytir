"""Drive the serving-only consent sweep to completion, verify it, and keep it alive.

Task 721 (S4): the package home of task 479's two shell scripts, the resume loop +
verification (`run_4crawl_consent_sweep.sh`) and the relaunch-if-down check
(`watchdog_4crawl_consent.sh`). The old paths are shims that exec this module.

    cyt consent sweep drive     == python -m cytir.consent.sweep_driver drive
    cyt consent sweep watchdog  == python -m cytir.consent.sweep_driver watchdog

`drive` runs `cytir.consent.sweep.run` pass after pass until RUN.json says `complete`,
then runs `cytir.consent.sweep_verify.verify` against the baseline. Exit codes are the
shell driver's: 0 verified, 3 pass budget exhausted, 4 unexpected runner outcome,
1 verification refused. `watchdog` re-probes nothing: the runner skips every key
already in results.jsonl, so a relaunch costs no origin traffic.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from cytir.consent import resolver as cr
from cytir.consent import sweep, sweep_verify

REPO = Path(__file__).resolve().parents[2]
FRONTIER_FILE = "uncovered-live-frontier.jsonl.gz"


def run_status(run_json: Path) -> str:
    """RUN.json's `status`, or `missing` when absent or unreadable."""
    try:
        return json.loads(Path(run_json).read_text()).get("status", "missing")
    except Exception:  # noqa: BLE001 - any unreadable receipt is "missing"
        return "missing"


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def drive(*, frontier_run: Path, out_dir: Path, baseline: Path, revision: str,
          shared_cache: Path | None = None, batch_size: int = 100,
          concurrency: int = 2, poll_seconds: float = 30, max_passes: int = 4000,
          run_fn=None, verify_fn=None, sleep=time.sleep) -> int:
    run_fn = run_fn or sweep.run
    verify_fn = verify_fn or sweep_verify.verify
    shared_cache = Path(shared_cache or cr.CACHE_DIR)
    frontier = Path(frontier_run).parent / FRONTIER_FILE
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "process.log"

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{_stamp()} {message}\n")

    log(f"driver start frontier={frontier} out={out_dir}")
    os.environ["AW_PRODUCER_REVISION"] = revision
    pass_no = 0
    while True:
        pass_no += 1
        if pass_no > max_passes:
            log(f"ABORT pass budget {max_passes} exhausted")
            return 3
        rc = 0
        try:
            receipt = run_fn(frontier, Path(frontier_run), out_dir, batch_size, concurrency)
            rc = 0 if receipt["status"] == "complete" else 2
        except Exception as exc:  # noqa: BLE001 - logged, then the shell driver's rc
            log(f"runner error {type(exc).__name__}: {exc}")
            rc = 1
        status = run_status(out_dir / "RUN.json")
        log(f"pass={pass_no} rc={rc} status={status}")
        if rc == 0 and status == "complete":
            break
        if rc != 2 or status != "running":
            log(f"ABORT unexpected runner outcome rc={rc} status={status}")
            return 4
        sleep(poll_seconds)

    log(f"sweep complete after {pass_no} passes; verifying")
    base = json.loads(Path(baseline).read_text())
    out = out_dir / "CONSENT-VERIFICATION.json"
    try:
        if out.exists():
            raise ValueError(f"refusing existing verification output: {out}")
        result = verify_fn(
            frontier=frontier, run_dir=out_dir, shared_cache=shared_cache,
            expected_shared_files=base["files"],
            expected_shared_tree_sha256=base["tree_sha256"])
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        vrc = 0
    except Exception as exc:  # noqa: BLE001
        log(f"verification refused: {type(exc).__name__}: {exc}")
        vrc = 1
    log(f"verification rc={vrc}")
    return vrc


def watchdog_action(sweep_status: str, unit_active: bool) -> str:
    """`done` (nothing to do) | `alive` (unit running) | `relaunch`."""
    if sweep_status == "complete":
        return "done"
    if unit_active:
        return "alive"
    return "relaunch"


def watchdog_paths(corpus_root: Path, tag: str) -> dict:
    work = Path(corpus_root) / "_work" / "task479"
    return {
        "frontier_dir": work / f"r0-task479-4crawl-servingonly-frontier-{tag}",
        "sweep_dir": work / f"r0-task479-4crawl-servingonly-consent-{tag}",
        "unit": f"aw-task479-4crawl-consent-{tag}",
    }


def relaunch_command(*, release: Path, tag: str, revision: str, corpus_root: Path,
                     shared_cache: Path, operator_url: str, operator_contact: str,
                     batch_size: int = 100, concurrency: int = 2,
                     python: str = "python3") -> list[str]:
    """The `systemd-run` argv that restarts the sweep under the pinned release."""
    p = watchdog_paths(corpus_root, tag)
    env = {
        "AW_OPERATOR_URL": operator_url,
        "AW_OPERATOR_CONTACT": operator_contact,
    }
    cmd = ["systemd-run", f"--unit={p['unit']}", "--collect",
           f"--description=Task 479 4-crawl serving-only consent sweep ({Path(release).name})",
           "-p", f"WorkingDirectory={release}"]
    for key, value in env.items():
        cmd += ["-p", f"Environment={key}={value}"]
    cmd += [python, "-m", "cytir.consent.sweep_driver", "drive",
            "--frontier-run", str(p["frontier_dir"] / "RUN.json"),
            "--out-dir", str(p["sweep_dir"]),
            "--baseline", str(p["frontier_dir"] / "SHARED-CACHE-BASELINE.json"),
            "--shared-cache", str(shared_cache),
            "--revision", revision,
            "--batch-size", str(batch_size), "--concurrency", str(concurrency)]
    return cmd


def watchdog(*, release: Path, tag: str, revision: str, corpus_root: Path,
             shared_cache: Path, operator_url: str, operator_contact: str,
             is_active=None, launch=None, out=print) -> str:
    p = watchdog_paths(corpus_root, tag)
    status = run_status(p["sweep_dir"] / "RUN.json")
    is_active = is_active or (lambda unit: subprocess.run(
        ["systemctl", "is-active", "-q", f"{unit}.service"], check=False).returncode == 0)
    launch = launch or (lambda cmd: subprocess.run(cmd, check=False).returncode)
    action = watchdog_action(status, is_active(p["unit"]) if status != "complete" else False)
    if action == "relaunch":
        out(f"{_stamp()} sweep not complete (status={status}) and unit down; relaunching")
        launch(relaunch_command(
            release=release, tag=tag, revision=revision, corpus_root=corpus_root,
            shared_cache=shared_cache, operator_url=operator_url,
            operator_contact=operator_contact))
    return action


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog or "python -m cytir.consent.sweep_driver",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("drive", help="resume loop + verification")
    d.add_argument("--frontier-run", required=True, type=Path)
    d.add_argument("--out-dir", required=True, type=Path)
    d.add_argument("--baseline", required=True, type=Path)
    d.add_argument("--revision", required=True)
    d.add_argument("--shared-cache", type=Path, default=None)
    d.add_argument("--batch-size", type=int, default=100)
    d.add_argument("--concurrency", type=int, default=2)
    d.add_argument("--poll-seconds", type=float, default=30)
    d.add_argument("--max-passes", type=int, default=4000)
    w = sub.add_parser("watchdog", help="relaunch the sweep unit if it is down and not done")
    w.add_argument("--release", required=True, type=Path)
    w.add_argument("--tag", required=True)
    w.add_argument("--revision", required=True)
    w.add_argument("--corpus-root", type=Path, default=None,
                   help="default: $CYTIR_CORPUS_ROOT (required one way or the other)")
    w.add_argument("--shared-cache", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "drive":
        return drive(frontier_run=a.frontier_run, out_dir=a.out_dir, baseline=a.baseline,
                     revision=a.revision, shared_cache=a.shared_cache,
                     batch_size=a.batch_size, concurrency=a.concurrency,
                     poll_seconds=a.poll_seconds, max_passes=a.max_passes)
    root = a.corpus_root or os.environ.get("AW_CORPUS_ROOT")
    url, contact = os.environ.get("AW_OPERATOR_URL"), os.environ.get("AW_OPERATOR_CONTACT")
    if not (root and url and contact):
        ap.error("watchdog needs --corpus-root (or CYTIR_CORPUS_ROOT), CYTIR_OPERATOR_URL and "
                 "AW_OPERATOR_CONTACT in the environment")
    watchdog(release=a.release, tag=a.tag, revision=a.revision, corpus_root=Path(root),
             shared_cache=Path(a.shared_cache or cr.CACHE_DIR),
             operator_url=url, operator_contact=contact)
    return 0


if __name__ == "__main__":
    sys.exit(main())
