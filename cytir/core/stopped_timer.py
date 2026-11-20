"""Stop a systemd unit for the duration of a block, and start it again at the end (task 1241).

Moved from the payload-profile reconcile; the consent backfill and the `published` backfill both
take the promotion timer off while they write. Generic: it knows no unit name.
"""
from __future__ import annotations

import sys
import time


class StoppedTimerError(RuntimeError):
    """The unit could not be stopped, or the service it started would not finish."""


class StoppedTimer:
    """Stop a systemd unit for the duration of a block and start it again at the end,
    on success AND on failure (the drain must never stay off because this crashed).

    Stopping a TIMER does not stop the service it already started. `wait_unit` (the
    service) is then waited on, up to `wait_s`, before the block runs: the writer lock is
    held by that run, and a payload write must not be the thing that interrupts it. A wait
    that times out raises, and the timer is started again."""

    def __init__(self, unit: str, *, run=None, wait_unit: str = "", wait_s: float = 0.0,
                 sleep=time.sleep, clock=time.monotonic):
        import subprocess
        self.unit = unit
        self._run = run or (lambda *a: subprocess.run(a, capture_output=True, text=True))
        self.was_active = False
        self.wait_unit, self.wait_s, self._sleep, self._clock = wait_unit, wait_s, sleep, clock
        self.waited_s = 0.0
        self._restart_due = False

    def __enter__(self):
        try:
            return self._enter()
        except BaseException:
            self._restart()      # a `with` block never calls __exit__ when __enter__ raises
            raise

    def _enter(self):
        self.was_active = self._run("systemctl", "is-active", self.unit).stdout.strip() == "active"
        if self.was_active:
            self._restart_due = True
            r = self._run("systemctl", "stop", self.unit)
            if r.returncode:
                raise StoppedTimerError(f"could not stop {self.unit}: {r.stderr.strip()}")
        if self.wait_unit:
            t0 = self._clock()
            while self._run("systemctl", "is-active", self.wait_unit).stdout.strip() in (
                    "active", "activating", "reloading"):
                if self._clock() - t0 > self.wait_s:
                    raise StoppedTimerError(
                        f"{self.wait_unit} still running after {self.wait_s:.0f}s; not writing")
                self._sleep(30)
            self.waited_s = self._clock() - t0
        return self

    def __exit__(self, *exc):
        self._restart()
        return False

    def _restart(self):
        """Start the unit again once; later calls (a caller's own `finally`) do nothing."""
        if self._restart_due:
            self._restart_due = False
            r = self._run("systemctl", "start", self.unit)
            if r.returncode:
                print(f"!! COULD NOT RESTART {self.unit}: {r.stderr.strip()}", file=sys.stderr)
