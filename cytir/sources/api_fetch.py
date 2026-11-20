"""Paced, retrying GETs and date windows for the API-route families (task 699).

Two families read a publisher API one document at a time (`us_federal_register_ecfr` leg `fr`:
<= 5 req/s; `govuk_content`: <= 2 req/s, both from their contract's `acquisition.rate_limit`).
This module is the ONE place that spaces the requests and retries a transient failure, so the
rate the contract declares is a number in one function, not a habit of each adapter.

* `Pacer(max_req_s)` keeps the average request rate under the cap on ONE connection (serial).
* `get(client, url, pacer, ...)` retries a 5xx, a 429 (it waits for `Retry-After`) and a transport
  error a few times, and then RAISES: our own failure is never a fact about the source (the
  framework quarantines the unit with the exception class; a 404/410 is the caller's decision).
* `month_bins` / `pack_windows` cut a date range into calendar months and greedy-pack the months
  into windows of at most `cap` documents: one window is one run, one pack, one stage, so the
  first rows reach the promotion queue after one window, not after the whole fetch.
"""
from __future__ import annotations

import calendar
import datetime as dt
import time
from typing import Callable

RETRIES = 4
BACKOFF_S = (2.0, 8.0, 30.0)
RETRY_STATUS = (429, 500, 502, 503, 504)


class Pacer:
    """Keep the mean request rate <= `max_req_s` (monotonic clock; `sleep` is injectable)."""

    def __init__(self, max_req_s: float, *, clock=time.monotonic, sleep=time.sleep):
        if max_req_s <= 0:
            raise ValueError("max_req_s must be > 0")
        self.gap = 1.0 / float(max_req_s)
        self._clock, self._sleep = clock, sleep
        self._next = 0.0

    def wait(self) -> None:
        now = self._clock()
        if now < self._next:
            self._sleep(self._next - now)
            now = self._next
        self._next = now + self.gap


def get(client, url: str, pacer: Pacer, *, params=None, sleep=time.sleep, timeout: float = 60.0,
        ok: tuple[int, ...] = (200,), absent: tuple[int, ...] = (404, 410)):
    """GET with pacing and retries. Returns the response when its status is in `ok` or `absent`
    (the caller decides what a 404/410 means); raises `RuntimeError` after the retries."""
    last: BaseException | str = ""
    for attempt in range(RETRIES):
        pacer.wait()
        try:
            resp = client.get(url, params=params, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 -- a transport error is retried, then raised
            last = exc
        else:
            if resp.status_code in ok or resp.status_code in absent:
                return resp
            last = f"HTTP {resp.status_code}"
            if resp.status_code not in RETRY_STATUS:
                break
            if resp.status_code == 429:
                try:
                    sleep(min(float(resp.headers.get("retry-after", "30")), 300.0))
                except ValueError:
                    sleep(30.0)
        if attempt < RETRIES - 1:
            sleep(BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)])
    raise RuntimeError(f"GET {url} failed after {RETRIES} attempts: {last!r}")


def month_bins(first: dt.date, last: dt.date) -> list[tuple[str, str]]:
    """Calendar months from `first` to `last` inclusive, as ISO `(from, to)` day pairs."""
    out, y, m = [], first.year, first.month
    while (y, m) <= (last.year, last.month):
        d1 = dt.date(y, m, 1)
        d2 = dt.date(y, m, calendar.monthrange(y, m)[1])
        out.append((max(d1, first).isoformat(), min(d2, last).isoformat()))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def pack_windows(bins: list[tuple[str, str, int]], cap: int) -> list[dict]:
    """Greedy-pack consecutive `(from, to, n_docs)` bins into windows of <= `cap` documents (a bin
    larger than `cap` is its own window). Empty bins are carried but never open a window."""
    wins: list[dict] = []
    cur: dict | None = None
    for frm, to, n in bins:
        if cur is not None and cur["docs"] + n <= cap:
            cur["to"], cur["docs"] = to, cur["docs"] + n
            continue
        if n == 0:
            if cur is not None:
                cur["to"] = to
            continue
        cur = {"from": frm, "to": to, "docs": n}
        wins.append(cur)
    if wins and bins:
        wins[0]["from"] = bins[0][0]            # the empty lead-in months belong to window 1
        wins[-1]["to"] = bins[-1][1]            # and the empty tail to the last: no gap, no overlap
    for i, w in enumerate(wins):
        w["key"] = f"w{i:03d}-{w['from']}_{w['to']}"
    return wins


def iso(d: dt.date | str) -> str:
    return d if isinstance(d, str) else d.isoformat()


Log = Callable[[str], None]
