"""Read one Common Crawl WARC record by byte range (task 683, S7).

A Common Crawl capture is addressed by `(warc_filename, offset, length)`; the record is one gzip
member inside a multi-gigabyte WARC file, read with an HTTP `Range` request. Three pipeline
modules each wrote this read for themselves (`cc_ingest.warc_body`,
`extract_outlinks.fetch_html`, `fetch_hn_cc`); this is the package home, written once, with the
two rules the copies learned the hard way:

* **politeness is part of the read**: the CC mirror IP-bans bulk fetching (ADR-014), so every
  read passes through a `RateLimiter`, default `MAX_REQUESTS_PER_S = 5`, and the limit cannot be
  raised above it by an argument (raise the constant by a ruling, not a flag);
* **an error is not an empty**: a transport failure, a 429/5xx after the retries, a bad gzip
  member is raised (`WarcReadError`), never returned as "no body". Only the origin's own
  authoritative miss (403/404/416 on the mirror) is `None`.

The pipeline copies are not touched by this task (live lanes use them); they move to this
function in S9. Nothing here needs the network to be tested: `cytir.acquire.tape` replays a sealed
range read.
"""
from __future__ import annotations

import gzip
import threading
import time
from dataclasses import dataclass, field

WARC_BASE = "https://data.commoncrawl.org/"
#: ADR-014: bulk fetch from the CC mirror gets an IP ban above this. Not a parameter.
MAX_REQUESTS_PER_S = 5.0
_RETRYABLE = frozenset({408, 425, 429, 500, 502, 503, 504})


class WarcReadError(RuntimeError):
    """The record could not be read for a reason that is ours or transient, not the mirror's
    authoritative miss."""


@dataclass
class WarcRecord:
    warc_headers: dict
    status: int | None
    http_headers: dict = field(default_factory=dict)
    body: bytes = b""


class RateLimiter:
    """At most `rate` calls per second across threads (a fixed minimum interval)."""

    def __init__(self, rate: float = MAX_REQUESTS_PER_S, *, clock=time.monotonic,
                 sleep=time.sleep):
        if rate <= 0 or rate > MAX_REQUESTS_PER_S:
            raise ValueError(f"CC request rate must be in (0, {MAX_REQUESTS_PER_S}] per second "
                             f"(ADR-014), got {rate}")
        self._interval = 1.0 / rate
        self._next = 0.0
        self._lock = threading.Lock()
        self._clock, self._sleep = clock, sleep

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            slot = max(now, self._next)
            self._next = slot + self._interval
        delay = slot - self._clock()
        if delay > 0:
            self._sleep(delay)


def _split_headers(block: bytes) -> tuple[str, dict]:
    lines = block.decode("utf-8", "replace").split("\r\n")
    first, rest = lines[0], lines[1:]
    hdrs: dict = {}
    for line in rest:
        if ":" in line:
            k, v = line.split(":", 1)
            hdrs[k.strip().lower()] = v.strip()
    return first, hdrs


def parse_warc_record(member_gz: bytes) -> WarcRecord:
    """One gzip member of a WARC response record -> headers, status, body bytes.

    Layout: WARC headers CRLFCRLF HTTP headers CRLFCRLF body. A record that does not have the
    three parts (a `warcinfo` or `request` record, a truncated member) raises `WarcReadError`
    rather than returning a half parse."""
    try:
        raw = gzip.decompress(member_gz)
    except (OSError, EOFError) as exc:
        raise WarcReadError(f"not a gzip member: {exc}") from exc
    head, sep, rest = raw.partition(b"\r\n\r\n")
    if not sep:
        raise WarcReadError("record has no WARC-headers / HTTP-headers / body structure")
    _wline, warc_headers = _split_headers(head)
    # The WARC block is exactly `Content-Length` bytes; a real record ends with a 4-byte
    # CRLFCRLF terminator OUTSIDE the block, which the old pipeline copies left on every body
    # (found on the first real record, task 683 S7 part 2: the body was 4 bytes longer than the
    # HTTP `Content-Length` and its sha1 did not equal `WARC-Payload-Digest`).
    declared = warc_headers.get("content-length", "")
    if declared.isdigit():
        if int(declared) > len(rest):
            raise WarcReadError(f"truncated record: block is {len(rest)} of {declared} bytes")
        rest = rest[: int(declared)]
    elif rest.endswith(b"\r\n\r\n"):
        rest = rest[:-4]
    parts = [head, *rest.split(b"\r\n\r\n", 1)]
    if len(parts) != 3:
        raise WarcReadError("record has no WARC-headers / HTTP-headers / body structure")
    status_line, http_headers = _split_headers(parts[1])
    status = None
    bits = status_line.split()
    if len(bits) >= 2 and bits[0].upper().startswith("HTTP/") and bits[1].isdigit():
        status = int(bits[1])
    return WarcRecord(warc_headers, status, http_headers, parts[2])


def range_get(client, rec: dict, *, limiter: RateLimiter, retries: int = 3,
              sleep=time.sleep) -> bytes | None:
    """The raw gzip member of `rec` (`warc_filename`/`filename`, `offset`, `length`), or None on
    the mirror's authoritative miss. Raises `WarcReadError` after `retries` transient failures."""
    filename = rec.get("warc_filename") or rec["filename"]
    off, ln = int(rec["offset"]), int(rec["length"])
    url = WARC_BASE + filename
    hdr = {"Range": f"bytes={off}-{off + ln - 1}"}
    last = "no attempt"
    for attempt in range(retries):
        limiter.wait()
        try:
            r = client.get(url, headers=hdr)
        except Exception as exc:  # noqa: BLE001 -- transport error: retry, then raise
            last = f"{type(exc).__name__}: {exc}"
        else:
            if r.status_code in (200, 206):
                return r.content
            if r.status_code not in _RETRYABLE:
                return None              # 403/404/416: the mirror's authoritative miss
            last = f"http {r.status_code}"
        sleep(3 * 2 ** attempt)
    raise WarcReadError(f"{url} {hdr['Range']}: {last} after {retries} attempts")
