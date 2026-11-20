"""The serving box's health between write batches (task 1241; moved here from the payload-profile reconcile).

Not a read gate: `cytir.index.prod_guard.check` is the one gate for Qdrant READS. This module is kept out of
`prod_guard` because that one sits in the drain's import closure, whose network and subprocess sites are
censused by the drain dry-run test. Targets are the caller's; no host, container or URL is built in.
"""
from __future__ import annotations

import time


class GuardTripped(RuntimeError):
    """The prod box or the search path is unhealthy; the apply stops here."""


class BoxHealthGuard:
    """The serving box's health between write batches (a different job from `check`, which gates
    READS). Between batches, every `interval_s`: prod MemAvailable, qdrant RestartCount
    and ONE search. Pause (wait, re-check) on MemAvailable < floor or on
    non-200 searches; stop at once on a qdrant restart (the baseline moved) or when a
    pause outlasts `max_pause_s`. `run`/`search` are injectable for tests."""

    def __init__(self, *, ssh_host: str, container: str, search_url: str,
                 min_mem_gb=6.0, interval_s=120.0, max_pause_s=1800.0,
                 consecutive_bad=2, run=None, search=None, sleep=time.sleep,
                 clock=time.monotonic):
        self.ssh_host, self.container, self.search_url = ssh_host, container, search_url
        self.min_mem_gb, self.interval_s = min_mem_gb, interval_s
        self.max_pause_s, self.consecutive_bad = max_pause_s, consecutive_bad
        self._run, self._search = run or self._ssh, search or self._http
        self._sleep, self._clock = sleep, clock
        self.restart_baseline: int | None = None
        self.bad_searches = 0
        self._last = None
        self.log: list[dict] = []

    def _ssh(self, command: str) -> str:
        import subprocess
        out = subprocess.run(
            ["timeout", "60", "ssh", "-o", "ServerAliveInterval=10", "-o",
             "ConnectTimeout=10", "-o", "BatchMode=yes", self.ssh_host, command],
            capture_output=True, text=True, timeout=90)
        if out.returncode:
            raise GuardTripped(f"guard ssh failed ({out.returncode}): {out.stderr.strip()[:200]}")
        return out.stdout

    def _http(self, url: str) -> int:
        from cytir.core import internal_http as IH
        from cytir.sources.http import InternalHostRefused
        try:
            return IH.send(url, timeout=30)[0]
        except OSError:
            return 0
        except InternalHostRefused as e:
            raise GuardTripped(
                f"search host refused by the internal client ({e}); AW_PROD_HEALTH_SEARCH_URL "
                "is not loopback or a configured service: set its host in AW_API_URL") from e

    def sample(self) -> dict:
        mem = self._run("grep MemAvailable /proc/meminfo")
        mem_gb = int(mem.split()[1]) / 1048576.0
        restarts = int(self._run(
            "docker inspect -f '{{.RestartCount}}' %s" % self.container).strip())
        status = self._search(self.search_url)
        return {"mem_available_gb": round(mem_gb, 2), "restart_count": restarts,
                "search_status": status}

    def start(self) -> dict:
        s = self.sample()
        self.restart_baseline = s["restart_count"]
        self._last = self._clock()
        self.log.append({"event": "start", **s})
        if s["mem_available_gb"] < self.min_mem_gb or s["search_status"] != 200:
            raise GuardTripped(f"refusing to start: {s}")
        return s

    def tick(self) -> None:
        if self._last is not None and self._clock() - self._last < self.interval_s:
            return
        paused_at = None
        while True:
            s = self.sample()
            self._last = self._clock()
            self.bad_searches = self.bad_searches + 1 if s["search_status"] != 200 else 0
            self.log.append({"event": "check", **s})
            if s["restart_count"] != self.restart_baseline:
                raise GuardTripped(
                    "qdrant restarted (RestartCount %s -> %s): stop, do not resume "
                    "blindly" % (self.restart_baseline, s["restart_count"]))
            low = s["mem_available_gb"] < self.min_mem_gb
            down = self.bad_searches >= self.consecutive_bad
            if not (low or down):
                return
            paused_at = paused_at if paused_at is not None else self._clock()
            self.log.append({"event": "pause", "low_memory": low, "search_down": down})
            if self._clock() - paused_at > self.max_pause_s:
                raise GuardTripped(f"paused longer than {self.max_pause_s}s: {s}")
            self._sleep(60)
