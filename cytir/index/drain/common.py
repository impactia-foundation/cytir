"""Drain common: queue errors, the Single/TwinFirst transactions, atomic writes, config getters, retry.

Task 2875 (RULING 80): moved verbatim out of `cytir/index/promote_queue.py`, which re-exports
every name here.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import socket
import time
import urllib.parse

from cytir.core.paths import SettingRefused
from cytir.core.paths import operator_settings
from cytir.index import dry_run as DRY
from cytir.index import live_target as LT


#: task 673, review round 2 finding 5, NARROWED in round 3 finding 2: a
#: RESUME attempt (either shape -- prod-only or a full twin-leg replay)
#: retries ONLY a genuine network/transport failure -- a timeout or a
#: connection error -- before the drain gives up and ABORTs. Matches
#: `promote_sealed_pack.py`'s own `_API_TIMEOUT_RETRIES`/`_API_TIMEOUT_
#: BACKOFF_S` convention. Deliberately NOT `REFUSAL_EXCEPTIONS` (those are
#: deterministic content judgments -- retrying one would just get the
#: same refusal three times slower).
#:
#: Deliberately NEVER `TFA.ApplyAbort` (round 3 finding 2 -- round 2's
#: version wrongly included it): `ApplyAbort` is the SAME exception type
#: for a momentarily-held production lock (genuinely transient) AND for
#: `verify_preservation()`'s preservation-contract failure, `resume_
#: after_verified_twin_gate`'s "resume premise changed after production
#: lock" check, and its terminal mirror-verification failure -- all three
#: of THOSE are exactly the "something is genuinely wrong, stop now"
#: shape retrying would paper over, not retry past. Since the type alone
#: cannot tell them apart, the safe answer is to retry NEITHER: a real
#: lock-contention retry can be added later behind its own, narrower
#: signal (e.g. matching the specific "lock unavailable" message) if it
#: turns out to matter in practice.
#:
#: Retried at the WHOLE-ATTEMPT level, not just inside one HTTP call: a
#: resume's preflight/apply/verify sequence is idempotent end to end
#: (every write it can issue is), so replaying the whole thing on a
#: transport failure is always safe.
_RESUME_RETRYABLE_EXCEPTIONS = (TimeoutError, socket.timeout,
                                urllib.error.URLError, ConnectionError)


_RESUME_RETRY_BACKOFF_S = (5, 15, 30)


def _retry_resume(fn, *, pack_id: str, sleep=time.sleep):
    """Call `fn()`, retrying up to `len(_RESUME_RETRY_BACKOFF_S)` more
    times (task 673, review round 2 finding 5) if it raises a
    `_RESUME_RETRYABLE_EXCEPTIONS` member. Any other exception propagates
    on the first attempt, unchanged. `sleep` is injectable so tests never
    actually wait."""
    last_exc: BaseException | None = None
    for attempt, backoff in enumerate((0,) + _RESUME_RETRY_BACKOFF_S):
        if backoff:
            print(f"[{pack_id}] resume retry {attempt}/{len(_RESUME_RETRY_BACKOFF_S)} "
                 f"in {backoff}s after a transient error: "
                 f"{type(last_exc).__name__}: {last_exc}")
            sleep(backoff)
        try:
            return fn()
        except _RESUME_RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            continue
    raise last_exc


def default_queue_dir() -> str:
    """The promotion queue: operator setting `queue_dir` (refused by name when unconfigured)."""
    return operator_settings().queue_dir


def live_queue_dir() -> pathlib.Path | None:
    """The configured promotion queue, resolved, or None when the `queue_dir` setting REFUSES.
    A box with no configured queue (a stranger's) has no live queue: nothing to protect."""
    try:
        return pathlib.Path(default_queue_dir()).resolve()
    except SettingRefused:
        return None


#: The prod writer mutex twin_first_apply guards -- same host/path/protocol
#: run_batch.py uses, so the two drivers exclude each other rather than each
#: holding a private lock that the other does not know about.
def prod_lock_host() -> str:
    """The prod writer mutex host: an operator setting (refused when unconfigured)."""
    return operator_settings().prod_lock_host


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@DRY.refuses_when_dry("queue file write")
def atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    """Write `data` to `path` so a kill at ANY point leaves either the OLD
    complete file or the NEW complete file, never a truncated one (task
    673, review finding 3): a plain `write_text`/`write_bytes` truncates
    the target in place first, so a SIGKILL mid-write leaves broken JSON
    that reads as a parse failure -- and for the twin-write-state artifact
    specifically, `read_content_pack_twin_state` treating a broken file as
    "no record" is exactly the silent-orphan failure mode this whole
    mechanism exists to close. Write to a same-directory temp file, fsync
    the file, `os.replace` (atomic on the same filesystem), then fsync the
    directory entry too -- durable across a kill, not just a crash-free
    process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp), str(path))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    atomic_write_bytes(path, text.encode())


class QueueError(RuntimeError):
    """A refusal. Never downgraded to a skip."""


class TransientReadError(RuntimeError):
    """A twin READ in the content-pack dry check failed at transport level (HTTP 5xx, a
    timeout after `_api`'s retries, a connection error). A fact about THIS run, not about the
    pack. Deliberately a SIBLING of `QueueError`, not a subclass, and not a `PromotionError`:
    no `except (QueueError, *REFUSAL_EXCEPTIONS)` may catch it and cache it as a content
    refusal. `drain` records it at stage `content_pack_read` (not cacheable) and retries."""


#: Consecutive hourly `content_pack_read` failures of one pack (same manifest sha) after which
#: the drain prints a loud `[drain]` line. Still never a cached content refusal.
READ_FAILURE_LOUD_AFTER = 6


def read_failure_streak(pack_dir: pathlib.Path, manifest_sha: str | None) -> int:
    """This failure's place in a run of consecutive `content_pack_read` failures of the same
    manifest: 1 + the count in the pack's previous receipt, if that receipt is also a
    `content_pack_read` refusal of the same manifest sha. Anything else restarts at 1."""
    try:
        prior = json.loads((pack_dir / "receipt.json").read_text())
    except (OSError, json.JSONDecodeError):
        return 1
    if (isinstance(prior, dict) and prior.get("stage") == "content_pack_read"
            and prior.get("manifest_sha") == manifest_sha):
        try:
            return int(prior.get("consecutive_read_failures", 1)) + 1
        except (TypeError, ValueError):
            return 1
    return 1


@dataclasses.dataclass(frozen=True)
class TwinFirst:
    """The transaction our own drain runs: write the TWIN, verify it, gate, then write PROD
    (resolved through `prod_alias`), under the writer locks and the twin-first ledger.
    `drain(transaction=None)` IS `TwinFirst(twin_url, prod_url, prod_alias)` -- today's behaviour,
    unchanged."""
    twin_url: str
    prod_url: str
    prod_alias: str = "aw_prod"


@dataclasses.dataclass(frozen=True)
class Single:
    """A stranger's transaction (slice 8 step 2, specs/aw-index-serve-gate.md §2): ONE target
    collection, no twin leg. It differs from `TwinFirst` in four named places and no others:
    (1) the caller creates the collection (the drain refuses a missing one either way); (2) no
    TWIN/PROD writer locks and no twin-first ledger; (3) ADR-110 check 5 reads "target == the pack's
    counts" where `TwinFirst` reads "twin == prod"; (4) the prod-only knobs are off: no alias, no
    `--twin-parallel`, no throttle (`pause_s`; 0 unless set here), and the persisted-input check
    runs only when `AW_PERSISTED_INDEX_SQLITE_DIR` names an index (otherwise RECORDED skipped).
    It takes ONE lock of its own, `SINGLE-WRITE-LOCK-<collection>`, around each pack's write.
    Everything else -- compose's manifest, the plan, the projection, the mechanical content
    check, the write, the receipts, DONE -- is the same code.

    What it can NEVER do (round 2 of the review of #553): touch the live queue's state, or write a
    live collection. (a) Its queue dir must be explicit and not the live queue; (b) its
    receipts, refusal cache and DONE markers live in `state_dir`, never in the queue's pack dirs
    (the twin-first drain reads those); (c) its refusal-cache key carries the transaction and the
    target; (d) `assert_single_target_not_live` asks the ENDPOINT what it holds (`cytir.index.live_target`) and
    refuses one that carries the prod alias or a declared live collection, whatever the URL is
    spelled like, and any alias name; it fails closed when the listing cannot be read. There is
    no override: `cyt index promote` is the door to the real drain, not this transaction.

    `payload_profile_of` names the DECLARED collection whose payload form `collection` stores
    (`data/collection-payload-profiles.v1.json`); None = `collection`'s own declaration."""
    url: str
    collection: str
    pause_s: float = 0.0
    parallel: int = 1
    payload_profile_of: str | None = None
    state_dir: pathlib.Path | None = None     # default: <artifact_root>/single-state/<collection>
    prod_alias: str = LT.DEFAULT_PROD_ALIAS   # the prod alias this box is configured with (CLI --prod-alias)


Transaction = TwinFirst | Single


class WriteAttempt:
    """Whether a live-apply leg has actually issued a write yet.

    Task 660, 2026-09-25: `drain()`'s per-pack LIVE block already wrapped
    `live_apply()`/`live_apply_content_pack()` in a `try/except BaseException`
    -- but it unconditionally re-raised, which is correct for a genuine
    mid-write twin/prod divergence and WRONG for a refusal that happens to
    surface late. CC tranche 1 proved this live: `apply_side()`'s OWN
    `PSP.plan()` call (re-derived at write time, not reusing the read-only
    dry-run's already-proved `joined`/`matrix`/`id_index`) raised
    `PayloadCarryError` inside `apply_target("twin", ...)`, BEFORE
    `PSP.upsert` ever ran -- a pure refusal, not a divergence -- and the
    unconditional re-raise killed every later pack in the queue for four
    straight hourly runs (02:25-07:25Z, 2026-09-25).

    The fix distinguishes the two by WHETHER A WRITE WAS ISSUED, not by
    exception type (four separate refusal hierarchies already exist and a
    fifth will exist someday -- matching on "did we write" generalizes to
    all of them, per engineering-doctrine's "fix the class"). `apply_side()`
    and `apply_content_pack_side()` each flip `.started` to True
    immediately before their own `PSP.upsert` call -- the one moment a
    remote mutation becomes possible. Anything that raises before that is
    provably read-only and safe to treat as a per-pack refusal; anything
    that raises after it (including inside `verify_target`, which reads
    back what was just written) is a possible partial write and must stop
    the drain rather than silently move on.
    """
    __slots__ = ("started",)

    def __init__(self) -> None:
        self.started = False


# --------------------------------------------------------- reused locking
def _load_run_batch():
    """The lock helpers -- the ONE home for the build-box `TWIN-WRITE-LOCK` /
    `PROD-WRITE-LOCK` advisory pair, `cytir.ops.write_lock`. They used to be loaded
    from `task 629 run_batch.py` by file path; task 677 moved them into
    the package verbatim (a production module may not execute a task script) and
    `run_batch.py` imports them back. The name stays: tests stub this function.
    Not copied: a second lock implementation is exactly how two lanes end up
    disagreeing about who holds the mutex (engineering-doctrine, One Truth One
    Home)."""
    from cytir.ops import write_lock
    return write_lock


def _would_settle_text(record: dict) -> str:
    """The dry run's words for the supersede-settle decision ('' when there is none to settle)."""
    w = record.get("would_supersede_recovery")
    if not w:
        return ""
    if w["decision"] == "would-delete":
        return (f"; first delete the {w['present_on_prod']} superseded row(s) still on prod "
                "(supersede state `deleting`)")
    return "; first mark the supersede state applied (no superseded row is left on prod)"
