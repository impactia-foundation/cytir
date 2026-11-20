"""`StoppedTimer` (task 676, moved by task 1241): the unit comes back on success AND failure; a unit
that was off stays off; a service the timer already started is waited for first."""
from __future__ import annotations

import types

import pytest

from cytir.core.stopped_timer import StoppedTimer, StoppedTimerError


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s




def _timer_run(active, calls):
    def run(*cmd):
        calls.append(cmd)
        out = "active\n" if (cmd[1] == "is-active" and active) else (
            "inactive\n" if cmd[1] == "is-active" else "")
        return types.SimpleNamespace(stdout=out, stderr="", returncode=0)
    return run




def test_the_timer_is_restarted_even_when_the_block_fails():
    calls = []
    with pytest.raises(RuntimeError):
        with StoppedTimer("aw-promotion-nightly.timer", run=_timer_run(True, calls)):
            raise RuntimeError("leg failed")
    assert [c[1] for c in calls] == ["is-active", "stop", "start"]


def test_a_timer_that_was_off_is_left_off():
    calls = []
    with StoppedTimer("aw-promotion-nightly.timer", run=_timer_run(False, calls)):
        pass
    assert [c[1] for c in calls] == ["is-active"]



def test_a_running_drain_is_waited_for_before_the_block_and_the_timer_comes_back():
    states = iter(["activating", "activating", "inactive"])
    calls, clock = [], Clock()

    def run(*cmd):
        calls.append(cmd[1:])
        if cmd[1] == "is-active" and cmd[2] == "aw-promotion-nightly.timer":
            out = "active\n"
        elif cmd[1] == "is-active":
            out = next(states) + "\n"
        else:
            out = ""
        return types.SimpleNamespace(stdout=out, stderr="", returncode=0)
    with StoppedTimer("aw-promotion-nightly.timer", run=run,
                          wait_unit="aw-promotion-nightly.service", wait_s=600,
                          sleep=clock.sleep, clock=clock) as t:
        assert t.waited_s == 60, "two polls at 30 s while the drain was still running"
    assert calls[-1] == ("start", "aw-promotion-nightly.timer")


def test_a_drain_that_never_ends_aborts_and_the_timer_is_restarted():
    calls, clock = [], Clock()

    def run(*cmd):
        calls.append(cmd[1:])
        out = "active\n" if cmd[1] == "is-active" else ""
        return types.SimpleNamespace(stdout=out, stderr="", returncode=0)
    t = StoppedTimer("aw-promotion-nightly.timer", run=run,
                         wait_unit="aw-promotion-nightly.service", wait_s=100,
                         sleep=clock.sleep, clock=clock)
    with pytest.raises(StoppedTimerError, match="still running"):
        try:
            t.__enter__()
        finally:
            t.__exit__(None, None, None)
    assert calls[-1] == ("start", "aw-promotion-nightly.timer")


def _scripted(calls, *, wait_states=None, stop_rc=0):
    """A fake systemctl: the timer is active; the service answers from `wait_states`."""
    states = iter(wait_states or [])

    def run(*cmd):
        calls.append(cmd[1:])
        out, rc = "", 0
        if cmd[1] == "is-active" and cmd[2].endswith(".timer"):
            out = "active\n"
        elif cmd[1] == "is-active":
            out = next(states, "active") + "\n"
        elif cmd[1] == "stop":
            rc = stop_rc
        return types.SimpleNamespace(stdout=out, stderr="boom" if rc else "", returncode=rc)
    return run


def test_with_block_wait_timeout_restarts_the_timer_and_raises():
    calls, clock = [], Clock()
    with pytest.raises(StoppedTimerError, match="still running"):
        with StoppedTimer("x.timer", run=_scripted(calls), wait_unit="x.service", wait_s=60,
                          sleep=clock.sleep, clock=clock):
            raise AssertionError("the block must not run")
    assert calls[-1] == ("start", "x.timer")
    assert [c for c in calls if c[0] == "start"] == [("start", "x.timer")]


def test_with_block_failed_stop_restarts_the_timer_and_raises():
    calls = []
    with pytest.raises(StoppedTimerError, match="could not stop"):
        with StoppedTimer("x.timer", run=_scripted(calls, stop_rc=1)):
            pass
    assert calls[-1] == ("start", "x.timer")


def test_with_block_exception_restarts_once():
    calls = []
    with pytest.raises(ValueError):
        with StoppedTimer("x.timer", run=_scripted(calls)):
            raise ValueError("leg failed")
    assert [c[0] for c in calls].count("start") == 1


def test_with_block_normal_exit_restarts_once():
    calls = []
    with StoppedTimer("x.timer", run=_scripted(calls)):
        assert [c[0] for c in calls] == ["is-active", "stop"]
    assert [c[0] for c in calls].count("start") == 1


def test_a_callers_own_finally_after_the_block_does_not_restart_twice():
    calls = []
    t = StoppedTimer("x.timer", run=_scripted(calls))
    with t:
        pass
    t.__exit__(None, None, None)
    assert [c[0] for c in calls].count("start") == 1


def test_enter_timeout_then_callers_finally_does_not_restart_twice():
    calls, clock = [], Clock()
    t = StoppedTimer("x.timer", run=_scripted(calls), wait_unit="x.service", wait_s=0,
                     sleep=clock.sleep, clock=clock)
    with pytest.raises(StoppedTimerError):
        try:
            t.__enter__()
        finally:
            t.__exit__(None, None, None)
    assert [c[0] for c in calls].count("start") == 1
