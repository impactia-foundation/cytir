"""`cyt news routine-apply`: drive ONE routine news pack from a sealed spool to an applied
ADR-098 transaction under the ADR-043 am.4 daily gate.

Ported from the 588-line shell driver of the same name (task 2379); same env contract,
same exit codes, same refusals, same artifacts at the same paths.

WHAT IT DOES NOT DO
  * It does not seal a pack. The lane's own driver (`AW_NEWS_APPLY_DRIVER`, a `.py` package
    module or a shell shim) seals, one named step at a time; this door CALLS it.
  * It does not decide anything. The daily gate's standard lives in
    `data/news-daily-gate-contract.v1.json` (`cytir.gates.news_daily`); nothing here softens it.
  * It does not run unless ARMED (`AW_NEWS_AUTO_APPLY=1`) and every input is present. A missing
    input is a refusal, never a default.

EXIT CODES  0 applied (or `--dry-run` plan printed) | 1 missing input / internal failure |
  2 refused (battery due, no live count, no baseline arm, prod baseline did not converge) |
  3 full-digest anomaly twin != prod | 4 DISARMED, nothing applied.
  A failed driver step, transaction or child process exits 1 (its own code, which could collide
  with 2/3/4, is kept in the FAILED row written to `$AW_TASK598_EV/apply-armed-daily-ledger.jsonl`).

Shell-only constructs: `flock`, `nohup`, `ssh` and the `set -a; . lane.env` sourcing stay in the
caller (`deploy/scripts/news_refresh_cron.sh`); this door reads its inputs from the environment.

IN-PROCESS IMPORTS come from the release this door runs from (the caller `cd`s into the
release). `AW_RELEASE_DIR` still names the release whose contract file is read and whose tree
every child process runs in (cwd / PYTHONPATH); a mismatch with the running package is REFUSED
(exit 1, nothing written) so the in-process checks and the children never run from two releases.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

# The steps the lane's own driver owns, in the ratified order, split where the gate profile
# changes what happens. `gate-baseline` is the driver's BATTERY arm and is not used here; the
# daily-gate arm is cut between the two groups (`preflight-dry` and later need BASELINE_ARM).
# `apply-armed` is run by this door itself with `--gate-profile news-daily`.
SEAL_STEPS = ("delta", "staged-ids", "seal-stages", "pack", "durability", "source-evidence",
              "baseline-twin", "baseline-prod", "baseline-refs")
PLAN_STEPS = ("preflight-dry", "mutation-plan", "intent", "runtime", "verify")
CONTRACT_NAME = "news-daily-gate-contract.v1.json"
REUSE_STATE_SCHEMA = "aw.news-daily-prod-baseline-reuse-state.v1"
DIGEST_EVERY_DEFAULT = 5

# name -> refusal text. NO env default: an absent name refuses (the old `${X:?msg}` contract).
REQUIRED = (
    ("AW_RELEASE_DIR", "set AW_RELEASE_DIR to the immutable release"),
    ("AW_NEWS_APPLY_DRIVER", "set AW_NEWS_APPLY_DRIVER to the sealed-pack driver of the lane"),
    ("AW_NEWS_CADENCE_LEDGER", "set AW_NEWS_CADENCE_LEDGER to the ADR-043 am.4 cadence ledger"),
    ("AW_TASK598_EV", "set AW_TASK598_EV to the lane evidence dir"),
    ("AW_PACK_NAME", "set AW_PACK_NAME to the sealed pack name"),
    ("AW_NEWS_TWIN_URL", "set AW_NEWS_TWIN_URL"),
    ("AW_NEWS_TWIN_COLLECTION", "set AW_NEWS_TWIN_COLLECTION"),
    ("AW_PROD_URL", "set AW_PROD_URL"),
    ("AW_PROD_COLLECTION", "set AW_PROD_COLLECTION"),
    ("AW_QVEC_CACHE", "set AW_QVEC_CACHE to the gate query-vector cache"),
    ("AW_QVEC_QUERY_SOURCE_MANIFEST", "set AW_QVEC_QUERY_SOURCE_MANIFEST"),
    ("ART", "set ART to the sealed pack artifact"),
    ("SV", "set SV to the staged-verification receipt"),
    ("TP", "set TP to the tombstone pointers"),
    ("ROWS", "set ROWS to the row count of the pack"),
    ("STAGED_COLLECTION", "set STAGED_COLLECTION"),
)


class Refusal(Exception):
    """Stop with this exit code and message (the shell `exit N` paths)."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class Cfg:
    """The environment contract, read once. Missing required names refuse (exit 1) in the
    order the shell read them, so the first missing one is the one named."""

    def __init__(self, env=None):
        env = os.environ if env is None else env
        for name, hint in REQUIRED:
            if not env.get(name):
                raise Refusal(1, f"{name}: {hint}")
        self.env = env
        self.rd = Path(env["AW_RELEASE_DIR"])
        self.driver = env["AW_NEWS_APPLY_DRIVER"]
        self.ledger = env["AW_NEWS_CADENCE_LEDGER"]
        self.ev = Path(env["AW_TASK598_EV"])
        self.pack = env["AW_PACK_NAME"]
        self.twin_url, self.twin_coll = env["AW_NEWS_TWIN_URL"], env["AW_NEWS_TWIN_COLLECTION"]
        self.prod_url, self.prod_coll = env["AW_PROD_URL"], env["AW_PROD_COLLECTION"]
        self.cache, self.qvec = env["AW_QVEC_CACHE"], env["AW_QVEC_QUERY_SOURCE_MANIFEST"]
        self.art, self.sv, self.tp = env["ART"], env["SV"], env["TP"]
        self.rows, self.staged = env["ROWS"], env["STAGED_COLLECTION"]

    @property
    def prod_lock_host(self) -> str:
        """`AW_PROD_LOCK_HOST`, else the operator settings (the one home the lane driver also reads)."""
        from cytir.core.paths import operator_settings
        return self.env.get("AW_PROD_LOCK_HOST") or operator_settings().prod_lock_host

    @property
    def qpy(self) -> str:
        from cytir.core.paths import operator_settings
        return self.env.get("AW_QPY") or operator_settings().qpy

    @property
    def contract(self) -> Path:
        """The gate contract, from the one data home: the running release's `data/` when run from
        a release dir, the package data in a non-editable install."""
        from cytir.core.paths import data_path
        return data_path(CONTRACT_NAME)

    @property
    def news_work(self) -> str:
        return self.env.get("AW_NEWS_WORK") or ""

    @property
    def stop_after_twin(self) -> bool:
        return bool(self.env.get("AW_STOP_AFTER_TWIN_GATE"))


def say(message: str) -> None:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{stamp} news-routine-apply: {message}", flush=True)


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fail_row(ev: Path, step: str, rc: int, pack: str) -> None:
    """A FAILED row in the transaction's own ledger (task 626): names the step, the exit."""
    ev.mkdir(parents=True, exist_ok=True)
    row = {"status": "FAILED", "step": step, "exit": int(rc), "pack": pack,
           "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "note": "the step's own output names the cause; nothing after it ran"}
    with open(ev / "apply-armed-daily-ledger.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _child_env(cfg: Cfg) -> dict:
    return {**os.environ, "PYTHONPATH": str(cfg.rd)}


def gate_status(cfg: Cfg) -> subprocess.CompletedProcess:
    """`python3 -m cytir.gates.news_daily --contract C status --ledger L`, output captured."""
    return subprocess.run(
        [sys.executable, "-m", "cytir.gates.news_daily", "--contract", str(cfg.contract),
         "status", "--ledger", cfg.ledger],
        env=_child_env(cfg), text=True, capture_output=True)


def battery_not_due(cfg: Cfg) -> bool:
    """Cheap refusal first: True only if the status call succeeded AND says `battery_due: false`.
    The status JSON is echoed to stderr (the old `tee /dev/stderr`)."""
    done = gate_status(cfg)
    sys.stderr.write(done.stdout)
    sys.stderr.write(done.stderr)
    sys.stderr.flush()
    return done.returncode == 0 and '"battery_due": false' in done.stdout


def count_of(url: str, collection: str) -> str:
    """The live points_count of a collection; "" when unreadable (the caller refuses)."""
    try:
        from cytir.core.internal_http import fetch_json  # the package's one door to the operator's own services
        return str(fetch_json(f"{url}/collections/{collection}", timeout=30)["result"]["points_count"])
    except Exception as exc:  # noqa: BLE001 - any read failure is a refusal upstream
        print(f"count_of {url}/{collection}: {exc!r}", file=sys.stderr)
        return ""


def run_step(cfg: Cfg, step: str) -> None:
    """One step of the lane's driver; a non-zero exit writes a FAILED row and stops."""
    say(f"driver step: {step}")
    argv = [sys.executable, cfg.driver, step] if cfg.driver.endswith(".py") else ["bash", cfg.driver, step]
    rc = subprocess.run(argv).returncode
    if rc != 0:
        say(f"FAILED: driver step {step} exit={rc} -- the cause is the step's own last lines above; "
            f"FAILED row written to {cfg.ev}/apply-armed-daily-ledger.jsonl; nothing after this step ran")
        fail_row(cfg.ev, step, rc, cfg.pack)
        raise Refusal(1, "")


def arm_cmd(cfg: Cfg, url: str, collection: str, identity: str, arm_id: str, out_dir: Path) -> list:
    return [sys.executable, "-m", "cytir.gates.news_daily", "--contract", str(cfg.contract),
            "arm", "--release-dir", str(cfg.rd), "--qdrant-url", url, "--collection", collection,
            "--identity", identity, "--cache", f"{sha256_of(cfg.cache)}={cfg.cache}",
            "--out-dir", str(out_dir), "--arm-id", arm_id.lower()]


def gate_env(cfg: Cfg, url: str, collection: str) -> dict:
    from cytir.index import apply as apply_mod  # in-process: the running release's applier
    env = apply_mod.gate_environment(cfg.rd, url, collection)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def latest_arm(directory: Path) -> Path | None:
    """LATEST arm by mtime. A re-attempt against a reused evidence dir leaves an earlier file
    behind (arms are named by identity+hash, never overwritten), so 'exactly one' would break
    the very re-attempt a round-numbered directory exists for."""
    arms = sorted(Path(directory).glob("news-daily-gate-arm.*.json"), key=lambda p: p.stat().st_mtime)
    return arms[-1] if arms else None


def cut_twin_baseline(cfg: Cfg) -> Path:
    """The daily gate's BASELINE arm: the pre-apply twin, read-only; never reused across days."""
    out = cfg.ev / "gate" / "daily-preapply"
    subprocess.run(arm_cmd(cfg, cfg.twin_url, cfg.twin_coll,
                           f"current-b1:{cfg.pack}-daily-preapply", f"{cfg.pack}-daily-preapply", out),
                   cwd=cfg.rd, env=gate_env(cfg, cfg.twin_url, cfg.twin_coll), check=True)
    arm = latest_arm(out)
    if arm is None:
        raise Refusal(2, "REFUSED: no baseline arm materialized")
    return arm


def make_cutter(cfg: Cfg, label: str, out_dir: Path) -> Callable[[int], Path]:
    """`cut_arm(round)` for a converged prod reading ('prod-baseline' / 'prod-postapply')."""
    env = gate_env(cfg, cfg.prod_url, cfg.prod_coll)

    def cut_arm(index: int) -> Path:
        round_dir = out_dir / f"r{index}"
        subprocess.run(arm_cmd(cfg, cfg.prod_url, cfg.prod_coll,
                               f"current-b1:{cfg.pack}-daily-{label}-r{index}",
                               f"{cfg.pack}-daily-{label}-r{index}", round_dir),
                       cwd=cfg.rd, env=env, check=True)
        arm = latest_arm(round_dir)
        if arm is None:
            raise SystemExit(f"no arm materialized in {round_dir}")
        return arm

    return cut_arm


def check_prod_baseline_reuse(release: Path, url: str, collection: str, state_file, out_dir: Path) -> bool:
    """Tim 2026-09-21: reuse the last transaction's post-apply prod reading when prod took no
    write and the #218 layout fingerprint did not move. Opt-in per state file; any doubt (no
    state, unreadable, different release, moved count or layout, ANY error) returns False and
    the caller cuts fresh. Every decision is recorded in `reuse-check.json`."""
    from cytir.index import layout as CL
    check_path = out_dir / "reuse-check.json"

    def refuse(reason, **extra):
        check_path.write_text(json.dumps({"reused": False, "reason": reason, **extra},
                                         indent=1, sort_keys=True) + "\n")
        print(f"[baseline-reuse] not reusing: {reason}", file=sys.stderr)
        return False

    try:
        state_p = Path(state_file)
        if not state_p.is_file():
            return refuse("no persisted prod-baseline state yet (first transaction, or "
                          "the state file was never written)")
        try:
            state = json.loads(state_p.read_text())
        except (OSError, ValueError) as exc:
            return refuse(f"persisted state unreadable: {exc}")
        arm_path = Path(state["arm"])
        if not arm_path.is_file():
            return refuse(f"persisted arm no longer on disk: {arm_path}")
        # task 626: a release cut alone moves every release-prefixed `path` in the arm's
        # serving_binding, so a cross-release baseline fails the gate AFTER a twin write.
        # Reuse never crosses a release boundary.
        if state.get("release") != Path(release).name:
            return refuse(f"persisted state was cut under a different release "
                          f"({state.get('release')!r} != {Path(release).name!r}) -- serving_binding "
                          "paths would differ even with identical content")
        now_fp = CL.fingerprint(url, collection)
        before_fp = state["layout_fingerprint"]
        same_count = now_fp["points_count"] == state["points_count"]
        quiet, reason = CL.quiescent(before_fp, now_fp)
        proof = {"reused": bool(same_count and quiet),
                 "previous_transaction": state.get("pack"), "previous_cut_at": state.get("cut_at"),
                 "same_points_count": same_count,
                 "points_count": {"persisted": state["points_count"], "now": now_fp["points_count"]},
                 "layout_quiescent": quiet, "layout_quiescent_reason": reason,
                 "layout_fingerprint_before": before_fp, "layout_fingerprint_now": now_fp,
                 "arm": str(arm_path), "arm_sha256": state.get("arm_sha256")}
        check_path.write_text(json.dumps(proof, indent=1, sort_keys=True) + "\n")
        if not proof["reused"]:
            print(f"[baseline-reuse] not reusing: same_count={same_count} quiet={quiet} ({reason})",
                  file=sys.stderr)
            return False
        dest = out_dir / "prod-baseline-arm.json"
        dest.write_text(arm_path.read_text())
        print(f"[baseline-reuse] REUSING {state.get('pack')}'s post-apply reading as today's "
              f"prod baseline -- {reason}", file=sys.stderr)
        print(str(dest))
        return True
    except Exception as exc:  # noqa: BLE001 - the shell fell through to a fresh cut on any failure
        print(f"[baseline-reuse] not reusing: {exc!r}", file=sys.stderr)
        return False


def cut_prod_baseline(cfg: Cfg, out_dir: Path) -> Path:
    """ADR-043 am.7: a CONVERGED concurrent reading of PRODUCTION (am.5's rule applied to the
    other input). A reading that will not converge aborts the whole apply (exit 2)."""
    from cytir.gates import convergence as gate_convergence
    from cytir.index import layout as CL
    cutter = make_cutter(cfg, "prod-baseline", out_dir)
    # No readiness snapshot: nothing is written to production here, and the snapshot reads
    # PHYSICAL names out of /telemetry, never the `aw_prod` alias (task 626).
    try:
        convergence = gate_convergence.converged_arms(
            cutter, log=print, layout_fingerprint=lambda: CL.fingerprint(cfg.prod_url, cfg.prod_coll))
    except (gate_convergence.ConvergenceUnadjudicated, gate_convergence.ConvergenceUnknown) as exc:
        raise Refusal(2, f"REFUSED: prod baseline did not converge: {exc}") from exc
    (out_dir / "prod-baseline-convergence.json").write_text(
        json.dumps(convergence, indent=1, sort_keys=True) + "\n")
    dest = out_dir / "prod-baseline-arm.json"
    dest.write_text(Path(convergence["arm"]).read_text())
    print(f"prod baseline arm: {dest}")
    return dest


def persist_postapply_reading(cfg: Cfg, out_dir: Path, state_file: Path) -> None:
    """One converged post-write reading of production, persisted for tomorrow's reuse check.
    Read-only, off the critical path: a reading that will not converge leaves no state and
    never aborts a transaction that already landed."""
    from cytir.gates import convergence as gate_convergence
    from cytir.index import layout as CL
    cutter = make_cutter(cfg, "prod-postapply", out_dir)
    try:
        convergence = gate_convergence.converged_arms(
            cutter, log=print, layout_fingerprint=lambda: CL.fingerprint(cfg.prod_url, cfg.prod_coll))
    except (gate_convergence.ConvergenceUnadjudicated, gate_convergence.ConvergenceUnknown) as exc:
        print(f"[postapply] did not converge, leaving no persisted state for tomorrow to reuse: {exc}",
              file=sys.stderr)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "prod-postapply-arm.json"
    dest.write_text(Path(convergence["arm"]).read_text())
    fp = CL.fingerprint(cfg.prod_url, cfg.prod_coll)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "schema": REUSE_STATE_SCHEMA, "pack": cfg.pack, "arm": str(dest), "release": cfg.rd.name,
        "arm_sha256": sha256_of(dest), "points_count": fp["points_count"], "layout_fingerprint": fp,
    }, indent=1, sort_keys=True) + "\n")
    tmp.replace(state_file)
    print(f"[postapply] persisted {state_file} for tomorrow's reuse check")


def transaction_argv(cfg: Cfg, baseline_arm: Path, prod_baseline_arm: Path, reuse_proof: str) -> list:
    """The ADR-098 transaction (`cytir.index.apply`) under `--gate-profile news-daily`."""
    env = cfg.env
    argv = [cfg.qpy, "-m", "cytir.index.apply", "--pack", cfg.pack,
            "--artifact", cfg.art, "--artifact-sha256", sha256_of(cfg.art), "--rows", cfg.rows,
            "--staged-verification", cfg.sv, "--staged-verification-sha256", sha256_of(cfg.sv),
            "--staged-collection", cfg.staged,
            "--tombstone-pointers", cfg.tp, "--tombstone-pointers-sha256", sha256_of(cfg.tp),
            "--twin-url", cfg.twin_url, "--prod-url", cfg.prod_url,
            "--twin-collection", cfg.twin_coll, "--prod-collection", cfg.prod_coll,
            "--prod-lock-host", cfg.prod_lock_host, "--batch", env.get("BATCH") or "256",
            "--settle-timeout", env.get("SETTLE_TIMEOUT") or "21600",
            "--ledger", str(cfg.ev / "apply-armed-daily-ledger.jsonl"), "--release-dir", str(cfg.rd),
            "--gate-dir", str(cfg.ev / "gate" / "apply-armed-daily"),
            "--baseline-arm", str(baseline_arm), "--baseline-arm-sha256", sha256_of(baseline_arm),
            "--prod-baseline-arm", str(prod_baseline_arm),
            "--prod-baseline-arm-sha256", sha256_of(prod_baseline_arm),
            "--cache", f"{sha256_of(cfg.cache)}={cfg.cache}",
            "--qvec-query-source-manifest", f"{sha256_of(cfg.qvec)}={cfg.qvec}",
            "--gate-profile", "news-daily", "--cadence-ledger", cfg.ledger]
    if env.get("AW_PERSISTED_OVERLAY"):
        argv += ["--persisted-overlay", env["AW_PERSISTED_OVERLAY"]]
    if cfg.stop_after_twin:
        argv += ["--stop-after-twin-gate"]
    if env.get("AW_WAIVER_REF"):
        argv += ["--waiver-ref", env["AW_WAIVER_REF"], "--waiver-issuer", env["AW_WAIVER_ISSUER"]]
    if reuse_proof:
        argv += ["--prod-baseline-reuse-proof", reuse_proof]
    return argv + ["--arm", "APPLY-CONTENT-PACK"]


def print_three_readings(verdict_path, baseline_path, prod_baseline_path) -> None:
    """The twin baseline, prod baseline, candidate, and the disagreement the gate recorded,
    straight from the verdict log this run wrote - never re-derived."""
    v = json.loads(Path(verdict_path).read_text(encoding="utf-8"))
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    prod_baseline = json.loads(Path(prod_baseline_path).read_text(encoding="utf-8"))
    m = v["metrics"]
    print(f"THREE READINGS -- twin pre-apply baseline: {baseline.get('identity')} "
          f"| prod baseline: {prod_baseline.get('identity')} | candidate: {v.get('candidate')}")
    print(f"h@1: baseline(effective)={m['h@1']['before']} "
          f"candidate={m['h@1']['after']} net={m['h@1']['attributable']:+d}")
    print(f"twin-vs-prod disagreement: {v.get('baseline_disagreement')}")


def _membership(cfg: Cfg, role: str, url: str, collection: str, points: str, digest_dir: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "cytir.index.seal_collection_membership", "--url", url,
         "--collection", collection, "--role", role, "--expect-points", points,
         "--artifact-root", cfg.news_work, "--receipt", str(digest_dir / f"{role}-receipt.json"),
         "--evidence", str(digest_dir / f"{role}-evidence.json")],
        env=_child_env(cfg), check=True)


def full_digest_step(cfg: Cfg) -> None:
    """Every Nth landed transaction, the FULL membership digest twin vs prod (Tim 2026-09-21).
    Raises Refusal(3) on an anomaly: STOP, escalate."""
    env = cfg.env
    counter = Path(env.get("AW_FULL_DIGEST_COUNTER_FILE")
                   or os.path.join(os.path.dirname(cfg.ledger), "..", "state", "full-digest-counter.txt"))
    counter.parent.mkdir(parents=True, exist_ok=True)
    try:
        prev = counter.read_text().strip()
    except OSError:
        prev = "0"
    n = (int(prev) if prev.isdigit() else 0) + 1
    every = int(env.get("AW_FULL_DIGEST_EVERY") or DIGEST_EVERY_DEFAULT)
    if n < every:
        counter.write_text(f"{n}\n")
        say(f"pack-scoped twin==prod verified this transaction (full digest in {every - n} more)")
        return
    say(f"FULL membership digest due (every {every}th transaction, {n} since last) -- twin vs prod")
    digest_dir = cfg.ev / "full-digest"
    digest_dir.mkdir(parents=True, exist_ok=True)
    twin_points = count_of(cfg.twin_url, cfg.twin_coll)
    prod_points = count_of(cfg.prod_url, cfg.prod_coll)
    if not (twin_points and prod_points):
        raise Refusal(1, "could not read a live point count from twin and/or prod for the full digest")
    _membership(cfg, "twin", cfg.twin_url, cfg.twin_coll, twin_points, digest_dir)
    _membership(cfg, "prod", cfg.prod_url, cfg.prod_coll, prod_points, digest_dir)
    twin_sha = json.loads((digest_dir / "twin-receipt.json").read_text())["logical_membership_sha256"]
    prod_sha = json.loads((digest_dir / "prod-receipt.json").read_text())["logical_membership_sha256"]
    if twin_sha != prod_sha or twin_points != prod_points:
        raise Refusal(3, f"FULL-DIGEST ANOMALY: twin({twin_points}, {twin_sha}) != "
                         f"prod({prod_points}, {prod_sha}) -- STOP, escalate, do not land another day until resolved")
    say(f"FULL-DIGEST-OK: twin == prod == {twin_points} points, digest {twin_sha}")
    counter.write_text("0\n")


def print_plan(cfg: Cfg) -> int:
    say(f"PLAN for {cfg.pack} ({cfg.rows} rows) -> {cfg.twin_coll} then {cfg.prod_coll}")
    for step in SEAL_STEPS:
        print(f"  driver: {cfg.driver} {step}")
    print("  daily-gate baseline arm: python3 -m cytir.gates.news_daily arm (pre-apply twin)")
    print("  daily-gate PROD baseline (am.7): converged pair, python3 -m cytir.gates.news_daily arm x2 + "
          "gate_convergence, against $AW_PROD_URL/$AW_PROD_COLLECTION")
    for step in PLAN_STEPS:
        print(f"  driver: {cfg.driver} {step}")
    print("  transaction: cytir.index.apply --gate-profile news-daily")
    print(f"  cadence ledger: {cfg.ledger}")
    sys.stdout.flush()
    done = gate_status(cfg)
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    return done.returncode


def apply_routine(cfg: Cfg) -> int:
    env = cfg.env
    if env.get("AW_NEWS_AUTO_APPLY") != "1":
        # task626 (Tim, 2026-09-21): never exit 0 when nothing was applied.
        bar = "=" * 58
        say(bar)
        say("DISARMED -- NOTHING APPLIED (AW_NEWS_AUTO_APPLY != 1)")
        say("set AW_NEWS_AUTO_APPLY=1 to actually run this transaction")
        say(bar)
        return 4
    # Inputs the shell only discovered mid-run (after seal writes) are refused up front now.
    if not cfg.news_work:
        raise Refusal(1, "AW_NEWS_WORK: set AW_NEWS_WORK")
    if env.get("AW_WAIVER_REF") and not env.get("AW_WAIVER_ISSUER"):
        raise Refusal(1, "AW_WAIVER_ISSUER: set AW_WAIVER_ISSUER alongside AW_WAIVER_REF -- "
                         "a waiver needs a named human")
    here = Path(__file__).resolve().parents[2]
    if cfg.rd.resolve() != here:
        raise Refusal(1, f"AW_RELEASE_DIR: {cfg.rd} is not the release this door runs from ({here}); "
                         "run the door from AW_RELEASE_DIR (the cron passes AW_RELEASE_DIR=$REMOTE_RELEASE)")
    # Operator settings are read with the early checks: an absent or incomplete settings file
    # refuses here, before the seal/arm/plan steps write any evidence.
    from cytir.core.paths import SettingRefused, SettingsError
    try:
        cfg.prod_lock_host, cfg.qpy
    except (SettingRefused, SettingsError) as exc:
        raise Refusal(1, f"operator settings: {exc} -- set AW_PROD_LOCK_HOST and AW_QPY or fix the settings file")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    # Cheap refusals first: a due battery must not cost a fetch, an embed or a twin write.
    if not battery_not_due(cfg):
        raise Refusal(2, "REFUSED: a full battery is due (or the cadence ledger is unreadable) -- handoff only")

    # Live counts, read fresh: the baseline steps need AW_EXPECT_POINTS(_PROD).
    twin_n, prod_n = count_of(cfg.twin_url, cfg.twin_coll), count_of(cfg.prod_url, cfg.prod_coll)
    if not (twin_n and prod_n):
        raise Refusal(2, "REFUSED: could not read a live point count from twin and/or prod")
    os.environ["AW_EXPECT_POINTS"], os.environ["AW_EXPECT_POINTS_PROD"] = twin_n, prod_n
    say(f"twin={twin_n} prod={prod_n} (baseline expectation, read now)")

    if env.get("AW_NEWS_SKIP_SEAL_STEPS") == "1":
        say("SKIP seal steps (AW_NEWS_SKIP_SEAL_STEPS=1) -- already sealed by the caller")
    else:
        for step in SEAL_STEPS:
            run_step(cfg, step)

    say("daily-gate baseline arm")
    baseline_arm = cut_twin_baseline(cfg)
    os.environ["BASELINE_ARM"] = str(baseline_arm)

    prod_dir = cfg.ev / "gate" / "daily-prod-baseline"
    prod_dir.mkdir(parents=True, exist_ok=True)
    reuse_state = Path(env.get("AW_PROD_BASELINE_REUSE_STATE")
                       or os.path.join(cfg.news_work, "state", "last-prod-baseline-state.json"))
    reuse_proof = ""
    if check_prod_baseline_reuse(cfg.rd, cfg.prod_url, cfg.prod_coll, reuse_state, prod_dir):
        prod_baseline_arm = prod_dir / "prod-baseline-arm.json"
        reuse_proof = str(prod_dir / "reuse-check.json")
        say(f"daily-gate PROD baseline (am.7): REUSED (see {reuse_proof})")
    else:
        say("daily-gate PROD baseline (am.7): a converged concurrent production reading")
        prod_baseline_arm = cut_prod_baseline(cfg, prod_dir)
    if not prod_baseline_arm.is_file():
        raise Refusal(2, "REFUSED: no prod baseline arm materialized")
    os.environ["PROD_BASELINE_ARM"] = str(prod_baseline_arm)
    os.environ["PROD_BASELINE_REUSE_PROOF"] = reuse_proof

    for step in PLAN_STEPS:
        run_step(cfg, step)

    say("ADR-098 transaction under the ADR-043 am.4 daily gate")
    os.environ["AW_CONTENT_PACK_PRESERVATION_INTENT"] = str(cfg.ev / "intent.json")
    # the evidence copy the `intent` step wrote, named after the shipped registry (one name, one home)
    from cytir.sources.schema import PRESERVATION_PROFILES_PATH
    os.environ["AW_CONTENT_PACK_PRESERVATION_PROFILE_REGISTRY"] = str(
        cfg.ev / "intent-evidence" / PRESERVATION_PROFILES_PATH.name)
    os.environ["AW_ARTIFACT_ROOT"] = cfg.news_work
    os.environ["AW_CONTENT_PACK_PRESERVATION_DEEP"] = "1"
    (cfg.ev / "gate" / "apply-armed-daily").mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(transaction_argv(cfg, baseline_arm, prod_baseline_arm, reuse_proof),
                        env=_child_env(cfg)).returncode
    if rc != 0:
        say(f"FAILED: transaction (cytir.index.apply) exit={rc} -- the cause is its own last lines above; "
            f"FAILED row written to {cfg.ev}/apply-armed-daily-ledger.jsonl")
        fail_row(cfg.ev, "transaction", rc, cfg.pack)
        raise Refusal(1, "")

    verdicts = sorted((cfg.ev / "gate" / "apply-armed-daily").glob("daily-gate.*.json"),
                      key=lambda p: p.stat().st_mtime)
    if verdicts:
        print_three_readings(verdicts[-1], baseline_arm, prod_baseline_arm)

    if cfg.stop_after_twin:
        say(f"TWIN-GATE-PASS-STOPPED for {cfg.pack} ({cfg.rows} rows) -- production NOT touched "
            "(AW_STOP_AFTER_TWIN_GATE); cadence unchanged:")
    else:
        say(f"APPLIED {cfg.pack} ({cfg.rows} rows); cadence:")
    status = gate_status(cfg)
    sys.stdout.write(status.stdout)
    sys.stderr.write(status.stderr)
    if status.returncode != 0:
        # The pack landed; the cadence read failed. One exit code (1) like every other failed step.
        raise Refusal(1, f"FAILED: post-apply cadence status exit={status.returncode} -- "
                         "the pack landed, the digest and post-apply reading did not run")

    if not cfg.stop_after_twin:
        full_digest_step(cfg)
        say("post-apply PROD reading (for tomorrow's baseline reuse check)")
        persist_postapply_reading(cfg, cfg.ev / "gate" / "daily-prod-postapply", reuse_state)
    return 0


def main(argv: Sequence[str] | None = None, prog: str = "cyt news routine-apply") -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print the ordered plan and the cadence status, write nothing, exit 0")
    args = parser.parse_args(argv)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    try:
        cfg = Cfg()
        return print_plan(cfg) if args.dry_run else apply_routine(cfg)
    except Refusal as stop:
        if stop.message and stop.code == 1:
            print(stop.message, file=sys.stderr)
        elif stop.message:
            say(stop.message)
        return stop.code
    except subprocess.CalledProcessError as exc:
        print(f"{prog}: a child step failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
