"""A dry run never writes: the one switch every write primitive of the promotion drain checks.

THE DEFECT THIS CLOSES (2026-10-03). `promote_queue.drain()` without `--apply` -- the mode agents
and operators use to "prove a plan" -- reached `resume_prod_only()` on the sealed-pack
"already on twin, prod owed" branch without looking at `apply` or the gate profile. Reproduced on
a throwaway Qdrant: given a pack whose twin leg had landed and whose prod leg had not, a dry drain
upserted every owed id into the prod collection, took both writer locks and the prod writer lock,
wrote the resume ledger and the prod written-ids artifact, and marked the pack DONE with a LANDED
receipt. The content-pack resume branch had the guard (task 673 review finding 1); the sealed
branch never did. The explicit `apply` checks in `drain()` fix the branch; this module is the
backstop that makes the NEXT forgotten check fail loudly instead of writing.

HOW. `guard(active=True)` marks the process dry for its duration. A plain module global, not a
contextvar: the twin leg fans upserts out over a thread pool, and a contextvar does not reach those
threads. Every write primitive the drain can reach calls `refuse_http()` first or carries
`@refuses_when_dry`:

  * `promote_sealed_pack._api` and `point_delete._api` -- every Qdrant HTTP call of the drain
    (upsert, delete, payload, collection, alias); reads pass, see `is_mutating_http`;
  * `twin_first.ProductionWriterLock.acquire/release` (ssh to the prod host) and
    `twin_first.append_ledger` (the transaction ledger);
  * `aw.ops.write_lock.acquire/release` (ssh to the lock host);
  * `promote_queue.atomic_write_bytes` -- receipts, DONE markers, written-id and state artifacts.

ONE DELIBERATE WRITE REMAINS: the incremental persisted-input index top-up
(`promote_queue.refresh_persisted_index`, run at the start of a drain unless
`--no-refresh-persisted-index`). It is a cache in the corpus root, not the queue, and it is
idempotent: it merges only new or changed store files. A dry plan needs it: with a stale index
the ADR-105 "pages are persisted" check false-refuses real packs (task 641 refused 82% of one
session's rows), so a dry verdict on a stale index is wrong, not merely old. It is therefore not
guarded. What a dry run does NOT do is the destructive form: `--full-persisted-index` deletes
the index files before rebuilding, so `drain(apply=False, full_persisted_index=True)` is refused
up front (`QueueError`, before anything is deleted). An `--apply` run is unchanged.

The switch is process-wide: one dry drain per process; a second thread writing for real in the
same process would be refused too. No such caller exists.

A refusal raises `DryRunWriteRefused` and is also recorded, so a caller that catches it (the
drain turns most per-pack exceptions into a REFUSED receipt) cannot hide it: `guard()` raises again
on exit if anything was refused. The drain's dry-run test module pins the roster of raw
network call sites in the drain's import closure, so a new writer that bypasses these primitives
fails a test until it is either guarded or proven read-only.
"""
from __future__ import annotations

import contextlib
import functools
import pathlib
import re
import threading


class DryRunWriteRefused(BaseException):
    """A write was attempted while a dry run was active.

    BaseException on purpose, like KeyboardInterrupt: a write primitive's caller may wrap it in a
    generic `except Exception` (a retry, a best-effort cleanup), and a dry run's refused write must
    never be swallowed and read as "nothing to do"."""


_LOCK = threading.Lock()
_STATE: dict = {"depth": 0, "refused": [], "receipt_dir": None}


def active() -> bool:
    return _STATE["depth"] > 0


def receipt_dir() -> pathlib.Path | None:
    """Where a dry run may put its receipts (outside the queue), or None: print only."""
    return _STATE["receipt_dir"]


def refuse(what: str) -> None:
    """Call FIRST in every write primitive. No-op unless a dry run is active."""
    if not active():
        return
    with _LOCK:
        _STATE["refused"].append(what)
    raise DryRunWriteRefused(f"dry run: refused {what}")


def refuses_when_dry(what: str):
    """Decorator form of `refuse()` for a primitive that is a write whatever its arguments (a
    lock acquire/release, a ledger append, a file write). The first str/path positional argument
    is named in the refusal (a lock name, a file)."""
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if active():
                named = next((a for a in args if isinstance(a, (str, pathlib.PurePath))), None)
                refuse(what if named is None else f"{what} {str(named)!r}")
            return fn(*args, **kwargs)
        wrapper.__dry_run_guarded__ = what
        return wrapper
    return decorate


#: Qdrant REST calls that only READ. Everything else is a write. An allowlist, not a blocklist: an
#: endpoint nobody thought about is refused in a dry run, which fails loudly, never silently.
_READ_POST = re.compile(
    r"^/collections/[^/]+/points"
    r"(?:/(?:scroll|count|search|search/batch|search/groups|query|query/batch|query/groups"
    r"|recommend|recommend/batch|recommend/groups|discover|discover/batch"
    r"|search/matrix/pairs|search/matrix/offsets))?$"
    # facet is a collection-level route (`POST /collections/{c}/facet`, Qdrant 1.18.3); the
    # `points/facet` spelling does not exist there (404).
    r"|^/collections/[^/]+/facet$")


def is_mutating_http(method: str, path: str) -> bool:
    """True unless (method, path) is a Qdrant read. `POST /collections/c/points` is a retrieve by
    id (a read); `PUT` on the same path is an upsert (a write)."""
    m = (method or "GET").upper()
    if m in ("GET", "HEAD"):
        return False
    if m == "POST":
        return not _READ_POST.match(path.split("?", 1)[0].rstrip("/"))
    return True


def refuse_http(method: str, path: str) -> None:
    if active() and is_mutating_http(method, path):
        refuse(f"{method.upper()} {path}")


@contextlib.contextmanager
def guard(active: bool, *, receipts_to: pathlib.Path | None = None):
    """Mark the process dry while the block runs (`active=False`: no-op, an apply run).

    `receipts_to`: a directory OUTSIDE the queue where a dry run's receipts go (the caller
    checks it is outside); None prints them only."""
    if not active:
        yield
        return
    with _LOCK:
        if _STATE["depth"] == 0:
            _STATE["refused"] = []
            _STATE["receipt_dir"] = receipts_to
        _STATE["depth"] += 1
    try:
        yield
    finally:
        with _LOCK:
            _STATE["depth"] -= 1
            refused = list(_STATE["refused"]) if _STATE["depth"] == 0 else []
            if _STATE["depth"] == 0:
                _STATE["refused"] = []
                _STATE["receipt_dir"] = None
    if refused:
        raise DryRunWriteRefused(
            f"dry run attempted {len(refused)} write(s), all refused: {refused[:5]} -- a code path "
            "reached a write without checking --apply; that is a bug, nothing was written")
