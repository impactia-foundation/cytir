"""Drain locks: the unit lock and the Single target lock.

Task 2875 (RULING 80): moved verbatim out of `cytir/index/promote_queue.py`, which re-exports
every name here.
"""
from __future__ import annotations

import contextlib
import os
import pathlib

from cytir.core.paths import operator_settings
from cytir.index import dry_run as DRY
from cytir.index.drain.common import QueueError, Single


PROD_LOCK_PATH = "/tmp/aw_prod_writer.lock"


@contextlib.contextmanager
def single_target_lock(single: Single, task: str, purpose: str):
    """The Single transaction's ONE lock, `SINGLE-WRITE-LOCK-<collection>`, as a context manager: a
    build holds it around its own delete/create of the target (round 4, B1), the same lock
    `single_apply` takes around each pack's write, so the liveness check made under it cannot go
    stale before the delete."""
    DRY.refuse("single target lock")
    from cytir.index import promote_queue as _Q
    runb = _Q._load_run_batch()
    lock = f"SINGLE-WRITE-LOCK-{single.collection}"
    runb.acquire(lock, task, purpose, pid=os.getpid())
    try:
        yield
    finally:
        runb.release(lock, task)


class UnitLockRefused(QueueError):
    """The promotion unit's lock file is held or cannot be opened: nothing was started."""


@contextlib.contextmanager
def hold_unit_lock():
    """A hand run of `cyt index promote` on an operator box takes the unit's lock (`flock -n`, the
    same `flock(2)` the unit's `flock -n <file> ...` takes) for the whole run, so it can never
    interleave with the hourly drain. The file is the `promotion_lock` setting. A box with the
    setting empty (a stranger's box: no such unit) proceeds without a lock. With the setting set, the
    file is created if absent; a lock that is held, a file that cannot be opened and a missing parent
    directory REFUSE by name (`UnitLockRefused`)."""
    path = operator_settings().promotion_lock
    if not path:
        yield None          # the setting is empty: a stranger's box has no such unit
        return
    import fcntl
    lock = pathlib.Path(path)
    if lock.is_dir() or not lock.parent.is_dir():
        raise UnitLockRefused(f"the promotion unit's lock {path} cannot be used (a directory, or its "
                              "parent directory is missing): refusing to run beside the unit")
    try:
        fh = open(lock, "ab")      # creates it if absent (/run is cleared at reboot), as flock(1) does
    except OSError as exc:
        raise UnitLockRefused(f"the promotion unit's lock file {path} cannot be opened ({exc}): "
                              "refusing to run beside the unit") from exc
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise UnitLockRefused(
                f"the promotion unit's lock file {path} is held by another process (the hourly "
                "drain, or another hand run): refusing to run beside it. Wait for it to finish "
                "or run `systemctl status aw-promotion-nightly.service`") from exc
        yield path
    finally:
        fh.close()
