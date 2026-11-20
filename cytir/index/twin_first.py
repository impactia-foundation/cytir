# Moved from the retired `twin_first_apply` script (task 677, task 643 S5); the old path was removed (task 683 S9).
"""Shared fail-closed twin-first apply primitives.

This is deliberately a protocol, not a Qdrant-format-specific applier.  A
task supplies its immutable artifact validation and target-local write/readback
callbacks; this module owns the cross-target safety boundary: one real
production writer lock, twin completion before production, and a durable
append-only ledger.  It defaults to dry-run at the caller layer.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Callable, TypeVar

from cytir.index import dry_run as DRY
from cytir.index.preservation_contract import ContractError, verify_publish_environment

T = TypeVar("T")


class ApplyAbort(SystemExit):
    pass


class ProductionWriterLock:
    """Atomically own the actual prod writer lock over SSH for one operation.

    The lock host is explicit configuration: a local wrapper lock is not a
    substitute when production is reached through a tunnel.
    """
    def __init__(self, host: str, lock_path: str, purpose: str):
        self.host, self.lock_path, self.purpose = host, lock_path, purpose
        self.token = f"{os.getpid()}-{time.time_ns()}"
        self.held = False

    def _run(self, program: str) -> subprocess.CompletedProcess[str]:
        # OpenSSH concatenates command argv into one remote-shell string.  A
        # script passed as the argument after ``sh -c`` therefore loses its
        # argument boundary (notably when it begins with ``(``).  Feed the
        # exact program on stdin instead; the remote command remains constant
        # and no second quoting/parsing layer can reinterpret the script.
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
             self.host, "sh", "-s"],
            input=program, text=True, capture_output=True,
        )

    @DRY.refuses_when_dry("production writer lock acquire (ssh)")
    def acquire(self) -> None:
        # The lock file must contain its owner's token from the instant it
        # exists.  The original two-step form -- `(set -C; : > lock)` then
        # `printf token > lock` -- leaves a TOKEN-LESS lock if the process dies
        # between them, and `release()` deliberately removes a lock only when
        # its first field is the caller's own token, so such a file can never
        # be released and blocks every future production write permanently.
        # That is not hypothetical: task 385 found one born 2026-08-05, and
        # task 452 found another born 2026-08-26 (its recommended fix, made
        # here).  So the token is written to a private staging file first and
        # `ln` publishes it: hard-linking is atomic and fails when the target
        # exists, giving the same mutual exclusion with no token-less window.
        # A crash before the link leaves only the staging file, which the next
        # attempt overwrites; a crash after it leaves a complete, releasable
        # lock.
        value = shlex.quote(f"token={self.token} purpose={self.purpose} started={time.strftime('%FT%TZ', time.gmtime())}\n")
        lock = shlex.quote(self.lock_path)
        staging = shlex.quote(f"{self.lock_path}.acquire.{self.token}")
        command = (f"printf %s {value} > {staging} && "
                   f"ln {staging} {lock} 2>/dev/null; rc=$?; "
                   f"rm -f {staging}; exit $rc")
        run = self._run(command)
        if run.returncode:
            holder = self._run(f"test -e {lock} && cat {lock} || true").stdout.strip()
            raise ApplyAbort(f"production writer lock unavailable on {self.host}:{self.lock_path}: {holder or run.stderr.strip()}")
        self.held = True

    @DRY.refuses_when_dry("production writer lock release (ssh)")
    def release(self) -> None:
        if not self.held:
            return
        lock, token = shlex.quote(self.lock_path), shlex.quote(f"token={self.token}")
        # POSIX ``test =`` is equality, not glob matching. Compare the first
        # whitespace-delimited field exactly so a purpose/timestamp suffix
        # neither prevents our release nor lets us remove another owner's lock.
        run = self._run(f"set -- $(head -n1 {lock} 2>/dev/null); test \"${{1-}}\" = {token} && rm {lock}")
        self.held = False
        if run.returncode:
            raise ApplyAbort("refusing to remove production lock whose ownership changed")


@DRY.refuses_when_dry("transaction ledger append")
def append_ledger(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps({"ts": time.strftime("%FT%TZ", time.gmtime()), **event}, sort_keys=True) + "\n")
        out.flush(); os.fsync(out.fileno())


def verify_preservation(
        verifier: Callable[[], dict] | None = None) -> dict:
    """Run the default or explicit authority and require an actual PASS proof."""
    proof = (verifier or verify_publish_environment)()
    if not isinstance(proof, dict) or proof.get("status") != "PASS":
        raise ContractError(
            f"preservation verifier returned a non-PASS proof: {proof!r}")
    return proof


def twin_first(ledger: Path, lock: ProductionWriterLock, *, preflight: Callable[[], T],
               apply_target: Callable[[str, T], None], verify_target: Callable[[str, T], dict],
               armed: bool, post_twin_gate: Callable[[T, dict], dict] | None = None,
               locked_preflight: Callable[[T], T] | None = None,
               preservation_verifier: Callable[[], dict] | None = None,
               stop_after_twin_gate: bool = False) -> T:
    """Execute the common mirror protocol; callers retain data-format logic.

    ``stop_after_twin_gate`` (task 626): writes and gates the TWIN for real,
    then returns without ever calling ``apply_target("prod", ...)`` -- a
    deliberate stop on a PASS, not the failure path a gate REFUSE already
    takes. Exists for exactly one situation: proving a real, armed twin
    write clears a fixed gate before a human-gated go-ahead (a GO file, a
    ticket, an approval) authorizes the production leg, without having to
    either fake the write or risk a PASS silently completing onto
    production. Default False, so every existing caller's behavior is
    unchanged; the production writer lock is still acquired and released
    exactly as before (task 384's cross-transaction guarantee does not
    become optional just because this run stops early).
    """
    state = preflight()
    append_ledger(ledger, {"status": "PREFLIGHT-OK", "armed": armed})
    if not armed:
        return state
    try:
        # Full-release preservation remains the default for every caller.
        # Narrow transaction types may inject a distinct fail-closed verifier;
        # the caller must do so explicitly at the call site.
        preservation = verify_preservation(preservation_verifier)
    except ContractError as exc:
        append_ledger(ledger, {"status": "FAILED",
                               "error": f"preservation contract: {exc}"})
        raise ApplyAbort(f"preservation contract did not PASS: {exc}") from exc
    append_ledger(ledger, {"status": "PRESERVATION-VERIFIED",
                           "proof": preservation})
    lock.acquire()
    try:
        # A caller whose safety premise depends on live remote state can bind
        # it again while the one real production lock is held.  This happens
        # before even the twin write so stale inventory can never cross the
        # transaction boundary (Task384's prod-only refresh incident).
        if locked_preflight is not None:
            state = locked_preflight(state)
            append_ledger(ledger, {"status": "LOCKED-PREFLIGHT-OK"})
        apply_target("twin", state)
        twin = verify_target("twin", state)
        append_ledger(ledger, {"status": "TWIN-VERIFIED", "proof": twin})
        # Corpus changes require their gate on the actually written twin, not
        # a stale scratch substitute.  A gate exception is deliberately
        # terminal before prod: the shared failure ledger records its mechanism.
        if post_twin_gate is not None:
            gate = post_twin_gate(state, twin)
            if not isinstance(gate, dict) or gate.get("status") != "PASS":
                raise ApplyAbort(f"post-twin gate did not PASS: {gate!r}")
            append_ledger(ledger, {"status": "TWIN-GATE-PASS", "proof": gate})
        if stop_after_twin_gate:
            append_ledger(ledger, {
                "status": "STOPPED-AFTER-TWIN-GATE",
                "note": "requested stop: twin written and gated for real, "
                        "production was never touched (task 626)"})
            return state
        apply_target("prod", state)
        prod = verify_target("prod", state)
        append_ledger(ledger, {"status": "PROD-VERIFIED", "proof": prod})
    except BaseException as exc:
        append_ledger(ledger, {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        lock.release()
    append_ledger(ledger, {"status": "TWIN-PROD-VERIFIED-OK", "rollback": "artifact and target readback proofs retained; no destructive automatic rollback"})
    return state


def resume_after_verified_twin_gate(
        ledger: Path, lock: ProductionWriterLock, *,
        preflight: Callable[[], T], post_twin_gate: Callable[[T], dict],
        apply_prod: Callable[[T], None], verify_mirrors: Callable[[T], dict],
        armed: bool,
        preservation_verifier: Callable[[], dict] | None = None) -> T:
    """Resume only the gated production half of a failed twin-first apply.

    ``preflight`` must bind the terminal failed predecessor ledger and the
    current twin/prod states.  It is deliberately repeated after acquiring the
    real production lock; unequal results mean the recovery premise changed
    and production remains untouched.
    """
    state = preflight()
    append_ledger(ledger, {"status": "RESUME-PREFLIGHT-OK", "armed": armed,
                           "proof": state})
    if not armed:
        return state
    try:
        preservation = verify_preservation(preservation_verifier)
    except ContractError as exc:
        append_ledger(ledger, {"status": "FAILED",
                               "error": f"preservation contract: {exc}"})
        raise ApplyAbort(f"preservation contract did not PASS: {exc}") from exc
    append_ledger(ledger, {"status": "PRESERVATION-VERIFIED",
                           "proof": preservation})
    lock.acquire()
    try:
        locked_state = preflight()
        if locked_state != state:
            raise ApplyAbort("resume premise changed after production lock")
        append_ledger(ledger, {"status": "RESUME-PREDECESSOR-BOUND",
                               "proof": locked_state})
        gate = post_twin_gate(state)
        if not isinstance(gate, dict) or gate.get("status") != "PASS":
            raise ApplyAbort(f"post-twin gate did not PASS: {gate!r}")
        append_ledger(ledger, {"status": "TWIN-GATE-PASS", "proof": gate})
        apply_prod(state)
        mirrors = verify_mirrors(state)
        if not isinstance(mirrors, dict) or mirrors.get("status") != "PASS":
            raise ApplyAbort(f"terminal mirror verification did not PASS: {mirrors!r}")
        append_ledger(ledger, {"status": "PROD-VERIFIED", "proof": mirrors})
    except BaseException as exc:
        append_ledger(ledger, {"status": "FAILED",
                               "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        lock.release()
    append_ledger(ledger, {
        "status": "TWIN-PROD-VERIFIED-OK",
        "recovery": "resumed from immutable failed predecessor after verified twin",
        "rollback": "artifact and target readback proofs retained; no destructive automatic rollback",
    })
    return state
