"""The cross-task write-lock pair (`TWIN-WRITE-LOCK` / `PROD-WRITE-LOCK`) on the build box.

MOVED HERE, verbatim, from task 629's `run_batch.py` (task 677, task 643 S5):
`cytir/index/promote_queue.py`, the live drain, loaded that script by file path to reuse
these helpers ("not copied: a second lock implementation is exactly how two lanes end up
disagreeing about who holds the mutex"). A production module may not execute a task
script (architecture-doctrine #1/#4), so the ONE home is now the package. `run_batch.py`
imports the names back, so `RB.acquire` / `RB.LOCKS` / ... still exist for the W1 driver.

Nothing changed in the bodies: the ssh transport, the atomic `set -C` acquire, the
holder's pid, the sticky lock and the `task=` marker are as task 626/629 built them.
`LOCK_HOST` and `LOCK_DIR` are operator settings (`AW_LOCK_HOST`, `AW_LOCK_DIR`; `cytir.core.paths`),
resolved on first ACCESS: an unconfigured box refuses by name instead of guessing a directory,
because a lock taken in a different place is no lock. `LOCK_HOST=local`
runs the same commands in a local shell instead of over ssh: the test suite uses it (root
`conftest.py`) so no test ever reads or writes the real registry. CI runs ON the build box, and
a test that read the real registry turned CI red whenever a lane held a lock.
"""
from __future__ import annotations

import datetime as dt
import os
import shlex
import socket
import subprocess

from cytir.core.paths import operator_settings

#: The two cross-task locks, in acquisition order. Twin first, because the twin is
#: written first; releasing happens in reverse. Paths are the convention task 626
#: uses (PR #227 for prod, and the twin lock added alongside it), so the two lanes
#: exclude each other rather than each holding a private mutex.
LOCAL_HOST = "local"


def lock_location(env=None) -> tuple[str, str]:
    """(host, directory) of the registry: `AW_LOCK_HOST` / `AW_LOCK_DIR`, else the operator
    settings `lock_host` / `lock_dir` (refused by name when nobody configured them)."""
    env = os.environ if env is None else env
    return _resolve_host(env), _resolve_dir(env)


def _resolve_host(env=None) -> str:
    env = os.environ if env is None else env
    return env.get("AW_LOCK_HOST") or operator_settings().lock_host


def _resolve_dir(env=None) -> str:
    env = os.environ if env is None else env
    return env.get("AW_LOCK_DIR") or operator_settings().lock_dir


def effective_location() -> tuple[str, str]:
    """(host, directory) the lock calls in THIS process use right now: a value pinned on this
    module (a test's isolated registry) first, else `lock_location()`. Refuses by name (a
    `SettingRefused`) when nothing configured it. A startup check calls it before any lock is read."""
    return _host(), _dir()


def __getattr__(name):
    # `LOCK_HOST` / `LOCK_DIR` / `DEFAULT_LOCK_*` stay readable (the W1 driver imports them) but
    # resolve on first ACCESS, never at import. A test that pins `LOCK_HOST` / `LOCK_DIR` writes
    # the module dict (conftest), which this is never asked about.
    if name in ("LOCK_HOST", "DEFAULT_LOCK_HOST"):
        return _resolve_host()
    if name in ("LOCK_DIR", "DEFAULT_LOCK_DIR"):
        return _resolve_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _host() -> str:
    return globals().get("LOCK_HOST") or _resolve_host()


def _dir() -> str:
    return globals().get("LOCK_DIR") or _resolve_dir()


LOCKS = ("TWIN-WRITE-LOCK", "PROD-WRITE-LOCK")


class BatchError(RuntimeError):
    """A refusal. The driver stops; it never proceeds with a downgraded plan."""


class LockRegistryUnreadable(BatchError):
    """The lock directory is missing or the host did not answer. A lock that is taken in a
    different place is no lock, so "I could not look" must never read as "free"."""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ssh(host: str, command: str, timeout: int = 120) -> subprocess.CompletedProcess:
    if host == LOCAL_HOST:
        return subprocess.run(["bash", "-c", command], capture_output=True,
                              text=True, timeout=timeout)
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
         "-o", "ServerAliveInterval=15", host, command],
        capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------- the lock pair
def read_lock(name: str) -> str | None:
    """The lock's contents, or None if absent. Read-only. Refuses (`LockRegistryUnreadable`)
    when the lock DIRECTORY does not exist or the host cannot be read: an absent lock file in
    an existing directory is "free", a missing directory is a box that looks in the wrong place."""
    directory, host = _dir(), _host()
    path = f"{directory}/{name}"
    run = ssh(host, f"test -d {shlex.quote(directory)} || exit 3; "
                    f"cat {shlex.quote(path)} 2>/dev/null || true")
    if run.returncode == 3:
        raise LockRegistryUnreadable(
            f"the lock directory {directory!r} does not exist on {host!r}: refusing to read "
            f"{name} as free. Set AW_LOCK_HOST / AW_LOCK_DIR (or the operator overlay) to the "
            f"place the writers really take their locks.")
    if run.returncode:
        raise LockRegistryUnreadable(
            f"could not read {name} on {host!r} (rc={run.returncode}: {(run.stderr or '').strip()!r}): "
            f"refusing to treat an unreadable registry as free.")
    value = (run.stdout or "").strip()
    return value or None


def lock_state() -> dict:
    return {name: read_lock(name) for name in LOCKS}


def assert_locks_free(task: str | None = None) -> dict:
    """Refuse when either lock is held BY ANOTHER TASK, and report the holder.

    A held lock is never cleared here, in any state. If its pid is dead the right
    outcome is a human looking at why a writer died mid-transaction -- task 626's
    own release requires confirmed twin == prod, so a lock left behind is
    deliberately sticky (PR #227).

    `task` exists because the first version refused OUR OWN lock. The operator
    takes the twin lock to seal the gate's IDF table (P4) and holds it across the
    batch that consumes the table -- releasing in between would hand the twin to
    another lane mid-transaction, which is the opposite of what the lock is for.
    With no task-awareness this driver then refused to run inside the very
    exclusion it had been given, and the only ways out were to drop the lock or to
    skip the check: a guard whose correct use requires disabling it. Measured
    2026-09-22 on B01's dry run.

    A lock held by ANOTHER task still refuses, which is the whole property. What
    changes is only that `task=629` no longer counts as a stranger to task 629.
    """
    state = lock_state()
    held = {name: value for name, value in state.items() if value}
    marker = f"task={task}" if task else None
    foreign = {name: value for name, value in held.items()
               if not (marker and marker in value)}
    if foreign:
        raise BatchError(
            "refusing to start: cross-task lock(s) held -- "
            + "; ".join(f"{name}: {value}" for name, value in foreign.items())
            + ". Wait for the holder, or escalate if its pid is dead. This driver "
              "never clears a lock it did not take.")
    return state


# A dry drain never takes, or drops, a writer lock (cytir/index/dry_run.py, 2026-10-03).
from cytir.index import dry_run as DRY  # noqa: E402


@DRY.refuses_when_dry("writer lock acquire (ssh)")
def acquire(name: str, task: str, purpose: str, *, pid: object = None) -> str:
    """O_CREAT|O_EXCL acquire, with the owner in the file from the instant it
    exists. `set -C` + a single redirect is the atomic form; a two-step create-then-
    write leaves a token-less lock if the process dies between them, which task 385
    and task 452 both found in the wild and which nothing can then release."""
    path = f"{_dir()}/{name}"
    # The pid is THIS DRIVER's, on the host the driver runs on, and both are
    # named. Two wrong versions preceded this one, and each was wrong in a way
    # that destroyed the only thing the lock is for -- `assert_locks_free` reports
    # the holder so a human can ask whether it is still alive:
    #   1. `pid=$$` inside the quoted body wrote the four literal characters
    #      `pid=$$`, which answers nothing.
    #   2. `$$` as a printf argument expanded correctly -- to the pid of the
    #      REMOTE ssh shell, which exits milliseconds later. A lock naming a
    #      guaranteed-dead pid is worse than one naming none, because it invites
    #      "the holder is dead, clear it" about a writer that is very much alive.
    # Found 2026-09-22 by reading back a lock this driver had just taken.
    #
    # `pid` is overridable for exactly one honest reason: this driver's `main()`
    # holds the lock for the life of ONE process, so its own pid is the right
    # answer there -- but a lock taken interactively and held across a SEQUENCE of
    # short-lived processes has no single pid, and writing the first one's would
    # recreate failure (2) with extra steps. Such a caller passes
    # pid="n/a-multiprocess" and lets `owner` identify it.
    holder = os.getpid() if pid is None else pid
    fmt = (f"task={task} owner={purpose} host={socket.gethostname()} "
           f"pid={holder} at={now()} state=WRITING\n")
    program = f"set -C; printf %s {shlex.quote(fmt)} > {shlex.quote(path)}"
    run = ssh(_host(), program)
    if run.returncode:
        raise BatchError(f"could not acquire {name}: {run.stderr.strip()!r} "
                         f"(holder: {read_lock(name)!r})")
    return path


@DRY.refuses_when_dry("writer lock release (ssh)")
def release(name: str, task: str) -> None:
    """Remove the lock only when it is still ours."""
    path = f"{_dir()}/{name}"
    program = (f"grep -q {shlex.quote('task=' + task)} {shlex.quote(path)} "
               f"2>/dev/null && rm -f {shlex.quote(path)}")
    ssh(_host(), program)
