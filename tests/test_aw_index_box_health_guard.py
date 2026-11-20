"""The box health guard (task 676, moved with the guard by task 1241): pauses on low memory or
two bad searches in a row, stops at once on a qdrant restart, refuses to start on a sick box.
No ssh, no network: `run` and `search` are injected."""
from __future__ import annotations

import pytest

from cytir.index.box_health_guard import BoxHealthGuard, GuardTripped


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _guard(samples, clock=None, **kw):
    clock = clock or Clock()
    it = iter(samples)
    state = {}

    def run(command):
        if "MemAvailable" in command:
            state["s"] = next(it)
            return "MemAvailable: %d kB" % int(state["s"][0] * 1048576)
        return "%d\n" % state["s"][1]
    return BoxHealthGuard(ssh_host="prodbox", container="qdrant-test", search_url="http://prod.invalid/s", run=run, search=lambda url: state["s"][2], sleep=clock.sleep,
                         clock=clock, **kw), clock


def test_the_guard_refuses_to_start_on_a_sick_box():
    g, _ = _guard([(3.0, 0, 200)])
    with pytest.raises(GuardTripped, match="refusing to start"):
        g.start()


def test_the_guard_pauses_on_low_memory_and_resumes_when_it_recovers():
    g, clock = _guard([(20, 0, 200), (5.0, 0, 200), (5.5, 0, 200), (9.0, 0, 200)])
    g.start()
    clock.t += 200
    g.tick()
    events = [e["event"] for e in g.log]
    assert events.count("pause") == 2 and events[-1] == "check"
    assert clock.t >= 320, "it waited, it did not spin"


def test_one_bad_search_is_noise_two_in_a_row_pause():
    g, clock = _guard([(20, 0, 200), (20, 0, 503), (20, 0, 200)])
    g.start()
    clock.t += 200
    g.tick()
    assert "pause" not in [e["event"] for e in g.log]
    # two consecutive checks (one per interval) both non-200: pause, then re-check
    g2, clock2 = _guard([(20, 0, 200), (20, 0, 503), (20, 0, 503), (20, 0, 200)])
    g2.start()
    clock2.t += 200
    g2.tick()
    assert "pause" not in [e["event"] for e in g2.log]
    clock2.t += 200
    g2.tick()
    assert [e["event"] for e in g2.log].count("pause") == 1 and g2.bad_searches == 0


def test_a_qdrant_restart_stops_the_apply_at_once():
    g, clock = _guard([(20, 0, 200), (20, 1, 200)])
    g.start()
    clock.t += 200
    with pytest.raises(GuardTripped, match="qdrant restarted"):
        g.tick()


def test_a_pause_that_never_ends_stops_the_apply():
    g, clock = _guard([(20, 0, 200)] + [(2.0, 0, 200)] * 100, max_pause_s=300)
    g.start()
    clock.t += 200
    with pytest.raises(GuardTripped, match="paused longer"):
        g.tick()


def test_the_guard_checks_no_more_often_than_its_interval():
    g, clock = _guard([(20, 0, 200)])
    g.start()
    clock.t += 30
    g.tick()  # would raise StopIteration if it sampled again
    assert len(g.log) == 1


def test_the_guard_has_no_built_in_target():
    with pytest.raises(TypeError):
        BoxHealthGuard()


def test_a_search_host_the_internal_client_refuses_trips_the_guard(monkeypatch):
    # D1 (task 1668): the internal client refuses a non-configured host (ValueError); the guard
    # must trip with a message naming the knob, not raise a bare ValueError.
    for knob in ("AW_QDRANT_URL", "AW_TEI_URL", "AW_API_URL"):
        monkeypatch.delenv(knob, raising=False)
    monkeypatch.setenv("AW_PROD_HEALTH_SEARCH_URL", "http://prod.example.org:6333/collections/x/points/search")
    guard = BoxHealthGuard(ssh_host="box", container="qdrant",
                           search_url="http://prod.example.org:6333/collections/x/points/search",
                           run=lambda command: ("MemAvailable: %d kB" % (64 * 1048576)
                                                if "MemAvailable" in command else "0"),
                           sleep=lambda s: None)
    with pytest.raises(GuardTripped, match="AW_API_URL"):
        guard.start()
