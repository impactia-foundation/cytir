"""The drain's change log, slice 1 (task 2314, ruling 68d; spec `specs/delta-change-log.md`).

CONCERN: a daily delta that scrolls the whole collection takes hours at 61.7M points and cannot see
deletes. CHOICE: the drain, which is the only writer, records what it landed, once, at landing time:
one JSON-lines segment per landed pack, listing the ids that reached PROD (the second leg, so the log
never runs ahead of prod). Master-only; no pack, manifest or dump contains it (release-doctrine #1).

Off by default: the `change_log_dir` setting (`AW_CHANGE_LOG_DIR`) is empty, `record_landing` returns
None and the drain writes the same bytes as before. Written BEFORE the `DONE` marker, so a crash
between the two leaves a segment without `DONE`; the next pass rewrites the same file (idempotent).
The resume and "already held" `DONE` sites write their segment too (`_log_resumed_landing`, task 2337).

Line shape, `aw.change-log.v1`. Header, then one line per id, in landing order (upserts, then deletes):

    {"schema": "aw.change-log.v1", "pack_id": ..., "pack_sha": <manifest sha256>, "landed_at": ..., "n": N}
    {"id": ..., "op": "add" | "replace" | "upsert"}     -- `upsert`: landed, prior presence not recorded
    {"id": ..., "op": "tombstone", "reason": <public, `tombstone_vocab.TOMBSTONE_REASONS`>, "queue_reason": <the queue's own>}
    {"op": "rebase", "why": ...}                        -- at most once, last; `n` counts it

Deviations from the mini-spec, each for a reason (task doc `2314-change-log-writer.md`): no `fp`, ever
(task 2497 ruling: `cyt dump --delta` fetches each net add/replace row by id and hashes it itself);
`upsert` for content packs (their upsert does not look the id up first); the tombstone `reason` is
mapped at write time since task 2493 (`QUEUE_TO_PUBLIC_REASON`) and `queue_reason` keeps the queue's
own, so a segment can be re-labelled later. The log is master-only; a reader builds `{id, reason, at}`
and must never copy `queue_reason`.

    python3 -m cytir.index.change_log verify <segment> <receipt.json>
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import socket
import sys
import time
from collections.abc import Iterable

from cytir.distribute.tombstone_vocab import TOMBSTONE_REASONS as PUBLIC_TOMBSTONE_REASONS
from cytir.index import dry_run as DRY

SCHEMA = "aw.change-log.v1"
UPSERT_OPS = ("add", "replace", "upsert")
OPS = UPSERT_OPS + ("tombstone", "rebase")


#: Task 2493: the queue's tombstone reasons (`promote_queue.TOMBSTONE_REASONS`) to the PUBLIC
#: vocabulary (`tombstone_vocab.TOMBSTONE_REASONS`). The table lives here, not in `pack.py`: the queue owns the
#: left side and `pack` must not import the queue. A test asserts it is total over both vocabularies.
#: Never `takedown` unless the queue said `takedown`. The four public names map to themselves; the rest
#: are forced deletes that are not takedowns, so they take the public non-takedown names (a Tim-visible
#: choice, task doc): `canonical_duplicate` is a pure pack (the canonical row is already served; the
#: queue says `superseded` needs replacement rows in the same pack), so it is a retired generation; a
#: defective or rolled-back row (`binary_content`, `soft_404`, `restore_rollback`) leaves as a retired
#: generation (`superseded_generation`).
QUEUE_TO_PUBLIC_REASON = {
    "superseded": "superseded", "aged_out": "aged_out", "takedown": "takedown",
    "superseded_generation": "superseded_generation",
    "canonical_duplicate": "superseded_generation",
    "binary_content": "superseded_generation",
    "soft_404": "superseded_generation",
    "restore_rollback": "superseded_generation",
}

class ChangeLogError(RuntimeError):
    """A segment that cannot be written or does not agree with its receipt. Loud on purpose."""


def public_reason(queue_reason: str | None) -> str:
    """The public tombstone reason for a queue reason; ChangeLogError when it maps to nothing."""
    public = QUEUE_TO_PUBLIC_REASON.get(queue_reason or "")
    if public is None or public not in PUBLIC_TOMBSTONE_REASONS:
        raise ChangeLogError(f"tombstone reason {queue_reason!r} has no public mapping "
                             f"(known: {sorted(QUEUE_TO_PUBLIC_REASON)})")
    return public


def configured_dir() -> pathlib.Path | None:
    """The `change_log_dir` setting, or None when empty (the log is off)."""
    from cytir.core.paths import operator_settings
    value = operator_settings().change_log_dir
    return pathlib.Path(value) if value else None


def _setting_dir(name: str) -> str:
    """An operator setting that names a directory, or '' when the setting refuses (unset)."""
    from cytir.core.paths import SettingRefused, operator_settings
    try:
        return str(getattr(operator_settings(), name) or "")
    except SettingRefused:
        return ""


def written_ids_path(artifact_root: pathlib.Path, pack_id: str, role: str = "prod") -> pathlib.Path:
    """The drain's per-leg written-ids artifact (`apply_side`): every id the leg wrote."""
    return pathlib.Path(artifact_root) / f"{pack_id}-{role}-written-ids.json"


def read_written_ids(path: pathlib.Path) -> list[str]:
    return list(json.loads(pathlib.Path(path).read_text()))


def _stamp(landed_at: str) -> str:
    return landed_at.replace("-", "").replace(":", "")


def _find_existing(out_dir: pathlib.Path, pack_id: str) -> pathlib.Path | None:
    """The pack's own segment: the EXACT name `<stamp>-<pack_id>.jsonl`. A suffix glob would let
    `batch-00001` find (and overwrite) the segment of `l1-backfill-batch-00001`."""
    if not out_dir.is_dir():      # nothing landed yet: the directory is created on the first real write
        return None
    exact = re.compile(r"\d{8}T\d{6}Z-" + re.escape(pack_id) + r"\.jsonl")
    found = sorted(p for p in out_dir.iterdir() if exact.fullmatch(p.name))
    return found[-1] if found else None


def segment_for(out_dir: pathlib.Path | str, pack_id: str) -> pathlib.Path | None:
    """The pack's own segment in `out_dir`, or None (also when the directory does not exist yet)."""
    out = pathlib.Path(out_dir)
    return _find_existing(out, pack_id) if out.is_dir() else None


@DRY.refuses_when_dry("change-log segment write")
def _write_segment(path: pathlib.Path, text: str) -> None:
    """Temp file, fsync, rename, fsync the directory: a kill leaves the old or the new complete file
    (the same recipe as the drain's `atomic_write_bytes`). The directory is created here, after the
    dry-run guard, so a dry run never leaves an empty change-log directory behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(text.encode("utf-8"))
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


def build_lines(*, pack_id: str, pack_sha: str, landed_at: str, plan: dict,
                seq: int | None = None) -> list[dict]:
    """`plan`: {"upserts": ids or (id, op) pairs, "op": default op, "tombstones": [{"id", "reason"}],
    "rebase": why | None}. A plain id takes `plan["op"]` (default `upsert`); a pair overrides it."""
    default_op = plan.get("op", "upsert")
    lines: list[dict] = []
    for item in plan.get("upserts") or ():
        pid, op = item if isinstance(item, tuple) else (item, default_op)
        if op not in UPSERT_OPS or not pid:
            raise ChangeLogError(f"{pack_id}: bad upsert entry {item!r}")
        lines.append({"id": pid, "op": op})
    for t in plan.get("tombstones") or ():
        if not t.get("id") or not t.get("reason"):
            raise ChangeLogError(f"{pack_id}: a tombstone needs an id and a reason, got {t!r}")
        lines.append({"id": t["id"], "op": "tombstone", "reason": public_reason(t["reason"]),
                      "queue_reason": t["reason"]})
    if plan.get("rebase"):
        lines.append({"op": "rebase", "why": plan["rebase"]})
    header = {"schema": SCHEMA, "pack_id": pack_id, "pack_sha": pack_sha, "landed_at": landed_at,
              "n": len(lines)}
    if seq is not None:
        header["seq"] = seq
    return [header] + lines


SEQ_LOCK_FILE = ".seq.lock"


@contextlib.contextmanager
def _seq_lock(out: pathlib.Path):
    """An exclusive `flock` on `<dir>/.seq.lock` around next_seq + segment write + `_write_seq`
    (review 2853: eight processes all took seq 1; a segment that shares a seq with a shipped one is
    never replayed). Not taken in a dry run: the write refuses there and must create no directory."""
    if DRY.active():
        yield
        return
    out.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(out / SEQ_LOCK_FILE), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)        # closing releases the lock


def record_landing(receipt: dict, plan: dict, out_dir: pathlib.Path | str | None, *,
                   landed_at: str | None = None) -> pathlib.Path | None:
    """Write the segment of one landed pack and return its path; None when nothing is written.

    Nothing is written when `out_dir` is empty (log off), when the receipt is not `LANDED` (a refused
    or dry-run pack landed nothing), or when the plan holds no line (a pack that wrote nothing).
    Atomic: temp name, then rename. A second call for the same pack replaces the same file."""
    if not out_dir:
        return None
    if receipt.get("status") != "LANDED":
        return None
    pack_id = receipt["pack_id"]
    if not (plan.get("upserts") or plan.get("tombstones") or plan.get("rebase")):
        return None
    build_lines(pack_id=pack_id, pack_sha="", landed_at="", plan=plan)   # refuse a bad plan BEFORE the lock makes the directory
    out = pathlib.Path(out_dir)
    with _seq_lock(out):    # one lock for the drain and direct writers: no two segments share a seq
        return _land_locked(out, receipt, plan, landed_at)


def _land_locked(out: pathlib.Path, receipt: dict, plan: dict, landed_at: str | None) -> pathlib.Path:
    """The body of `record_landing`; the caller holds `_seq_lock(out)` (flock is not re-entrant)."""
    pack_id = receipt["pack_id"]
    existing = _find_existing(out, pack_id)
    if existing is not None:      # a crash between segment and DONE: keep name and original landed_at
        old = json.loads(existing.read_text().splitlines()[0])
        landed_at, seq = old["landed_at"], old.get("seq")
        path = existing
        if seq is None:
            seq = next_seq(out)
    else:
        landed_at = landed_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        path = out / f"{_stamp(landed_at)}-{pack_id}.jsonl"
        seq = next_seq(out)
    lines = build_lines(pack_id=pack_id, pack_sha=receipt.get("pack_sha") or "", landed_at=landed_at,
                        plan=plan, seq=seq)
    _write_segment(path, "".join(json.dumps(x, sort_keys=True) + "\n" for x in lines))
    _write_seq(out, seq)
    return path


def receipt_sha(receipt: dict | bytes) -> str:
    """The sha256 a `rebase` segment carries as `pack_sha` when no pack exists: the flip receipt's
    own bytes, or a dict receipt as canonical JSON (sorted keys). Never empty."""
    raw = receipt if isinstance(receipt, bytes) else json.dumps(receipt, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def record_rebase(pack_id: str, receipt: dict | bytes, out_dir: pathlib.Path | str | None, *,
                  why: str = "flip", landed_at: str | None = None) -> pathlib.Path | None:
    """One `rebase` segment for a step that carries no pack (a ship+flip, spec 3.5). `pack_sha` is the
    sha256 of the flip receipt. Off (None) when `out_dir` is empty."""
    return record_landing({"pack_id": pack_id, "status": "LANDED", "pack_sha": receipt_sha(receipt)},
                          {"rebase": why}, out_dir, landed_at=landed_at)


_DIRECT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")


def direct_pack_id(tool: str, write_id: str) -> str:
    """`direct-<tool>-<write_id>`: the segment name of one direct write. Both parts are file-name safe
    (checked), so the pack id is a plain name and `_find_existing` matches it exactly."""
    for part in (tool, write_id):
        if not isinstance(part, str) or not _DIRECT_ID.fullmatch(part):
            raise ChangeLogError(f"direct write: {part!r} is not a file-name safe tool or write id")
    return f"direct-{tool}-{write_id}"


def record_direct(*, tool: str, write_id: str, collection: str, verified_written: bool,
                  upserts: Iterable[str] = (), replaces: Iterable[str] = (),
                  payload_patches: Iterable[str] = (), tombstones: Iterable[dict] = (),
                  out_dir: pathlib.Path | str | None = None,
                  landed_at: str | None = None) -> pathlib.Path | None:
    """Ruling 75 (task 2849): the log call of a writer that is not the drain. Same segment, same line
    shape (`aw.change-log.v1`), same `seq` counter and directory as `record_landing`; the segment is
    named `direct-<tool>-<write_id>` and its `pack_sha` is the sha256 of the canonical write
    description (tool, write id, collection, the ids per op), so a retry of the same write is the
    same bytes.

    ORDER (spec 3.3, the drain's rule, applied unchanged): call this AFTER the write is verified on
    the collection, never before. `verified_written` must be True; False raises, so a caller cannot log
    a write that did not land (a segment never exists for a write that did not land; the log is a lower
    bound). A crash between the write and this call leaves a write with no line: that is a missed
    entry, which spec 3.7 detects (shadow compare, rehearsal). It is not repaired here. Retrying with
    the same `write_id` re-lands the same file (idempotent), so a writer that is resumed logs once.

    Lines: `upserts` -> `upsert` (landed, prior presence not recorded), `replaces` and
    `payload_patches` -> `replace` (an in-place rewrite lists every id, spec 3.5), `tombstones`
    (`{"id", "reason"}`, reason mapped to the public vocabulary, an unmapped one refused). `out_dir`
    defaults to the `change_log_dir` setting; unset = log off, returns None. Nothing is written for an
    empty write. `write_id` must be unique per logical write and stable across a retry (derive it from
    the pack or the batch, never the clock)."""
    if verified_written is not True:
        raise ChangeLogError(f"{tool}/{write_id}: refusing to log a write that is not verified landed "
                             "(spec 3.3: log after the verified write)")
    pack_id = direct_pack_id(tool, write_id)
    out = out_dir if out_dir is not None else configured_dir()
    if not out:
        return None
    upserts, replaces, payload_patches = list(upserts), list(replaces), list(payload_patches)
    tombstones = list(tombstones)
    ups = ([(i, "upsert") for i in upserts] + [(i, "replace") for i in replaces]
           + [(i, "replace") for i in payload_patches])
    desc = {"tool": tool, "write_id": write_id, "collection": collection,
            "upserts": sorted(map(str, upserts)), "replaces": sorted(map(str, replaces)),
            "payload_patches": sorted(map(str, payload_patches)),
            "tombstones": sorted(f"{t.get('id')}|{t.get('reason')}" for t in tombstones)}
    receipt = {"pack_id": pack_id, "status": "LANDED", "pack_sha": receipt_sha(desc)}
    plan = {"upserts": ups, "tombstones": tombstones}
    if not (ups or tombstones):
        return None                    # nothing to log for an empty write
    build_lines(pack_id=pack_id, pack_sha="", landed_at="", plan=plan)   # refuse a bad plan BEFORE the lock makes the directory
    out = pathlib.Path(out)
    with _seq_lock(out):               # the different-content check and the write sit under ONE lock (review 2853 (a))
        held = segment_for(out, pack_id)
        if held is not None:
            old_sha = json.loads(held.read_text().splitlines()[0]).get("pack_sha")
            if old_sha != receipt["pack_sha"]:
                raise ChangeLogError(
                    f"{held.name}: this write id already has a segment with different content (pack_sha "
                    f"{old_sha} != {receipt['pack_sha']}); a write id names ONE write. Refusing to rewrite it")
        return _land_locked(out, receipt, plan, landed_at)


PENDING_DIR = ".pending"


def preflight_dir() -> pathlib.Path | None:
    """Task 3002 review 3009 (3a): run BEFORE a direct writer's first write. None when the log is off.
    Otherwise the configured directory must already exist (never `mkdir` a second log from a box whose
    setting names a path that is not there), be writable, hold a log (a segment or the counter file),
    have a readable counter, and hold no unrepaired pending line (`replay_pending` first). Refuses with
    `ChangeLogError`; a refusal means NOTHING was written."""
    out = configured_dir()
    if out is None:
        return None
    if not out.is_dir():
        raise ChangeLogError(f"{out}: the change-log directory does not exist on this box; refusing to "
                             "write before the log can take the record (a writer must never start a "
                             "second log at seq 1). Fix the `change_log_dir` setting or unset it")
    if not os.access(out, os.W_OK | os.X_OK):
        raise ChangeLogError(f"{out}: the change-log directory is not writable; refusing to write")
    if not (out / SEQ_FILE).is_file() and next(out.glob("*.jsonl"), None) is None:
        raise ChangeLogError(f"{out}: holds no change log (no counter file, no segment); refusing to "
                             "start a second log from a direct writer")
    next_seq(out)                       # raises on an unreadable counter
    pend = out / PENDING_DIR
    left, running = [], []
    for f in (sorted(pend.glob("*.json")) if pend.is_dir() else []):
        w = _spool_writer(f)
        (running if _writer_alive(w) else left).append((f, w))
    if left:    # a spool whose writer is still running is a write IN FLIGHT, not a lost line: ignored
        more = (f"; {len(running)} more are in flight (writer pid "
                f"{', '.join(str(w['pid']) for _, w in running)} still running)" if running else "")
        raise ChangeLogError(f"{out}: {len(left)} change-log line(s) of an earlier write were never "
                             f"recorded (e.g. {left[0][0].name}); run `python -m cytir.index.change_log "
                             f"replay-pending` first{more}")
    return out


def preflight_or_abort(ledger=None) -> None:
    """Task 3020: `preflight_dir` for a CLI writer. A refusal is ledgered (when a ledger is given) and
    stops the run with `SystemExit` BEFORE the first write; the log being off is a no-op."""
    try:
        preflight_dir()
    except ChangeLogError as exc:
        if ledger is not None:
            ledger(f"CHANGE-LOG-PREFLIGHT-REFUSED {str(exc)[:500]!r}")
        raise SystemExit(f"ABORT: {exc}") from exc


def log_failure_exit(tool: str, write_id: str, exc: BaseException, *, ledger=None,
                     spooled: bool = False) -> SystemExit:
    """Task 3020: the loud stop of a direct writer whose write landed but whose record did not (the
    upsert wave's rule): the failure is ledgered and the caller RAISES the returned `SystemExit`."""
    if ledger is not None:
        ledger(f"CHANGE-LOG-FAILED {tool}/{write_id} {type(exc).__name__}: {str(exc)[:500]!r}")
    return SystemExit(f"FATAL: the write {tool}/{write_id} landed but its change-log record failed "
                      f"({type(exc).__name__}: {exc}); the segment is MISSING from the log"
                      + (" (spooled: run `python -m cytir.index.change_log replay-pending`)" if spooled else "")
                      + ". Stop and repair before any further write.")


def log_this_run(collection: str, url: str, *, not_the_twin: bool = False) -> bool:
    """Task 3020 (review 3024, third pass): the ONE decision of a CLI writer, called only by a run that WRITES
    (a dry run is never refused: it writes nothing). True = log this run (preflight, spool, record). False = do not.
    Fails closed like `cyt dump`: with the log on, a collection NAMED like the twin on a URL that is not the
    configured twin URL is refused before the first write (a `localhost` spelling of the real twin would
    otherwise write with no line, and the delta would miss it silently); `not_the_twin=True` is the operator's
    word that it is another Qdrant, and the run proceeds unlogged. No spelling is normalised."""
    if configured_dir() is None:
        return False
    from cytir.core import twin as TWIN
    if collection != TWIN.configured_collection():
        return False
    if str(url).rstrip("/") == TWIN.configured_url():
        if not_the_twin:
            raise SystemExit(f"REFUSED: --not-the-twin was passed but {url!r} with collection {collection!r} IS "
                             "the configured twin; the flag may only skip logging when the URL differs.")
        return True
    if not_the_twin:
        return False
    raise SystemExit(f"REFUSED: collection {collection!r} is the twin's name but --qdrant {url!r} is not the "
                     f"configured twin URL ({TWIN.configured_url()!r}); a write here would reach the delta source "
                     "with no change-log line. Use the configured twin URL, or pass --not-the-twin if this is a "
                     "different Qdrant that holds a collection of the same name.")


def uncertain_write_exit(tool: str, write_id: str, detail: str, *, ledger=None) -> SystemExit:
    """Task 3020 (review 3024 item 4): the stop of a direct writer whose write did not return `completed`
    (or raised): the same `CHANGE-LOG-FAILED` ledger line as `log_failure_exit`; the spool is KEPT (the write may
    have landed) and the message names `replay-pending`. Every direct writer stays blocked at preflight
    until that verb has run."""
    if ledger is not None:
        ledger(f"CHANGE-LOG-FAILED {tool}/{write_id} uncertain write: {detail[:500]!r}")
    return SystemExit(f"FATAL: the write {tool}/{write_id} is uncertain ({detail}); it may have landed. Its "
                      "change-log spool is kept. Every direct writer stays blocked at preflight until "
                      "`python -m cytir.index.change_log replay-pending` has run. Stop and repair first.")


def write_completed(status) -> bool:
    """True when a Qdrant update result says `completed` (a `wait=True` write that was applied)."""
    st = status.get("status") if isinstance(status, dict) else getattr(status, "status", None)
    return str(getattr(st, "value", st)) == "completed"


def _proc_start(pid: int) -> str | None:
    """The process start time (clock ticks since boot, /proc/<pid>/stat field 22), or None."""
    try:
        raw = pathlib.Path(f"/proc/{pid}/stat").read_text()
        return raw[raw.rindex(")") + 2:].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def _writer_identity() -> dict:
    return {"pid": os.getpid(), "host": socket.gethostname(), "start": _proc_start(os.getpid())}


def _spool_writer(path: pathlib.Path) -> dict | None:
    try:
        w = json.loads(path.read_text()).get("writer")
    except (OSError, ValueError):
        return None
    return w if isinstance(w, dict) and isinstance(w.get("pid"), int) else None


def _writer_alive(w: dict | None) -> bool:
    """True only when the spool's writer runs on THIS host with the SAME pid and start time (a recycled
    pid has another start time, so it is not alive). No record, another host, or no /proc: not alive."""
    if not w or w.get("host") != socket.gethostname():
        return False
    start = _proc_start(w["pid"])
    return start is not None and start == w.get("start")


def spool_pending(tool: str, write_id: str, collection: str, ops: dict) -> pathlib.Path | None:
    """Write the line a direct write is ABOUT to log, before the write (a `replace` / payload patch set
    only: re-logging an attempted id is safe, the delta builder fetches the row again). Removed by
    `clear_pending` once `record_direct` landed. A leftover file is a write whose line was lost:
    `replay_pending` re-records it. None when the log is off."""
    out = configured_dir()
    if out is None:
        return None
    path = out / PENDING_DIR / f"{direct_pack_id(tool, write_id)}.json"
    _write_segment(path, json.dumps({"tool": tool, "write_id": write_id, "collection": collection,
                                     "writer": _writer_identity(),
                                     "ops": {k: list(v) for k, v in ops.items()}}, sort_keys=True) + "\n")
    return path


@DRY.refuses_when_dry("pending-spool delete")
def clear_pending(path: pathlib.Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)


@DRY.refuses_when_dry("pending-spool replay")
def replay_pending(out_dir: pathlib.Path | str | None = None) -> list[str]:
    """Re-record every pending line (same write id: idempotent), then remove its file. Returns the
    write ids replayed. Holds no state of its own: the leftover files ARE the lost lines.

    Task 3011 (review 2309 r2, item 1): a replay during a running writer records the line before the
    write landed and a delta cut in between ships the old row. So it REFUSES (`ChangeLogError`, nothing
    recorded) while TWIN-WRITE-LOCK or PROD-WRITE-LOCK is held (`write_lock.assert_locks_free`, the one
    lock protocol; a sticky lock of a dead writer needs a human first) and while any spool's writer is
    still running on this host."""
    out = pathlib.Path(out_dir) if out_dir else configured_dir()
    if out is None or not (out / PENDING_DIR).is_dir():
        return []
    files = sorted((out / PENDING_DIR).glob("*.json"))
    if not files:
        return []
    from cytir.ops import write_lock as WL     # local: the log module stays importable without the lock stack
    try:
        WL.assert_locks_free()
    except WL.BatchError as exc:
        raise ChangeLogError(f"replay-pending refused: {exc}") from exc
    for f in files:
        w = _spool_writer(f)
        if _writer_alive(w):
            raise ChangeLogError(f"replay-pending refused: writer {w['pid']} is still running "
                                 f"({f.name}); its line is not lost, it has not landed yet")
    done = []
    for f in files:
        try:
            d = json.loads(f.read_text())
        except FileNotFoundError:
            continue
        record_direct(tool=d["tool"], write_id=d["write_id"], collection=d["collection"],
                      verified_written=True, out_dir=out, **d["ops"])
        f.unlink(missing_ok=True)
        done.append(d["write_id"])
    return done


def read_segment(path: pathlib.Path | str) -> tuple[dict, list[dict]]:
    rows = [json.loads(x) for x in pathlib.Path(path).read_text().splitlines() if x.strip()]
    if not rows or rows[0].get("schema") != SCHEMA:
        raise ChangeLogError(f"{path}: first line is not an {SCHEMA} header")
    header, body = rows[0], rows[1:]
    if header.get("n") != len(body):
        raise ChangeLogError(f"{path}: header n={header.get('n')} but {len(body)} line(s)")
    for line in body:
        if line.get("op") not in OPS:
            raise ChangeLogError(f"{path}: unknown op in {line!r}")
        if line["op"] == "tombstone" and line.get("reason") not in PUBLIC_TOMBSTONE_REASONS:
            raise ChangeLogError(f"{path}: tombstone reason {line.get('reason')!r} is not public "
                                 "(a segment written before task 2493?)")
    return header, body


def verify(segment: pathlib.Path | str, receipt: dict) -> dict:
    """Check a segment against its receipt; raise ChangeLogError on any disagreement.

    The upsert lines must equal the PROD leg's `written`; the tombstone lines its
    `tombstones_requested` plus `superseded` (url-group supersede deletes; absent = 0 each).
    The pack id must match. Returns the counts."""
    header, body = read_segment(segment)
    if header["pack_id"] != receipt.get("pack_id"):
        raise ChangeLogError(f"segment is for {header['pack_id']!r}, receipt for {receipt.get('pack_id')!r}")
    counts = {op: sum(1 for x in body if x["op"] == op) for op in OPS}
    ups = sum(counts[o] for o in UPSERT_OPS)
    prod = receipt.get("prod") or receipt.get("target") or {}
    problems = []
    if "written" in prod and prod["written"] != ups:
        problems.append(f"upsert lines {ups} != receipt written {prod['written']}")
    want_tomb = (prod.get("tombstones_requested") or 0) + (prod.get("superseded") or 0)
    if counts["tombstone"] != want_tomb:
        problems.append(f"tombstone lines {counts['tombstone']} != receipt tombstones_requested "
                        f"{prod.get('tombstones_requested', 0)} + superseded {prod.get('superseded', 0)}")
    if problems:
        raise ChangeLogError(f"{segment}: " + "; ".join(problems))
    return counts


_SEGMENT_NAME = re.compile(r"(\d{8}T\d{6}Z)-(.+)\.jsonl")


def _done_at(done: pathlib.Path) -> str:
    """A DONE marker's time: the ISO stamp it holds, else its mtime (UTC, same shape)."""
    text = done.read_text().strip()
    if re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", text):
        return text
    return dt.datetime.fromtimestamp(done.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def audit(queue_dir: pathlib.Path | str, log_dir: pathlib.Path | str, *,
          since: str | None = None) -> dict:
    """Spec 3.7 / task 2337: every pack with a `DONE` marker has a segment, and every segment has a
    `DONE`. Returns `{since, done_without_segment, segment_without_done, done_checked}`.

    Only packs that reached DONE at or after `since` are checked (default: the landed_at of the
    log's oldest segment, i.e. the day the log was switched on): earlier packs landed with the log
    off and have no segment by design. A `flip-*` segment is a flip's rebase line, not a queue pack.
    An empty log has nothing to compare against, so a default `since` checks nothing."""
    queue, logs = pathlib.Path(queue_dir), pathlib.Path(log_dir)
    segs: dict[str, str] = {}
    if logs.is_dir():
        for f in sorted(logs.iterdir()):
            m = _SEGMENT_NAME.fullmatch(f.name)
            if m:
                segs[m.group(2)] = m.group(1)
    start = since
    if start is None and segs:
        first = min(segs.values())
        start = f"{first[:4]}-{first[4:6]}-{first[6:8]}T{first[9:11]}:{first[11:13]}:{first[13:15]}Z"
    done = {}
    for d in sorted(p for p in queue.iterdir() if p.is_dir()) if queue.is_dir() else ():
        if (d / "DONE").is_file():
            done[d.name] = _done_at(d / "DONE")
    checked = {k: v for k, v in done.items() if start is not None and v >= start}
    left = sorted((logs / PENDING_DIR).glob("*.json")) if logs.is_dir() else []
    ages = [max(0, int(time.time() - f.stat().st_mtime)) for f in left if f.exists()]
    return {"since": start, "done_checked": len(checked),
            "pending_spools": len(ages), "oldest_pending_spool_age_s": max(ages) if ages else None,
            "done_without_segment": sorted(k for k in checked if k not in segs),
            "segment_without_done": sorted(k for k in segs if k not in done
                                           and not k.startswith("flip-"))}


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cytir.index.change_log")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="check a segment against its receipt")
    v.add_argument("segment")
    v.add_argument("receipt")
    a = sub.add_parser("audit", help="DONE packs without a segment, segments without a DONE (exit 1)")
    a.add_argument("queue_dir", nargs="?", default=None, help="default: the queue_dir setting")
    a.add_argument("log_dir", nargs="?", default=None, help="default: the change_log_dir setting")
    a.add_argument("--since", default=None, help="ISO time; default: the oldest segment's landed_at")
    sub.add_parser("replay-pending", help="re-record the lines of direct writes whose line was lost")
    r = sub.add_parser("rebase", help="write the rebase segment of a flip that carries no pack")
    r.add_argument("--pack-id", required=True, help="a name for the segment, e.g. flip-aw_e11")
    r.add_argument("--flip-receipt", required=True, help="the flip receipt file, or - for stdin")
    r.add_argument("--out", default=None, help="the log dir (default: the change_log_dir setting)")
    args = ap.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "audit":
        queue_dir = args.queue_dir or _setting_dir("queue_dir")
        log_dir = args.log_dir or configured_dir()
        for what, d in (("queue dir", queue_dir), ("change-log dir", log_dir)):
            if not d or not pathlib.Path(d).is_dir():
                print(f"AUDIT NOT RUN: the {what} is unset or missing ({d!r}); nothing was checked",
                      file=sys.stderr)
                return 2
        rep = audit(queue_dir, log_dir, since=args.since)
        print(json.dumps(rep, sort_keys=True))
        bad = rep["done_without_segment"] or rep["segment_without_done"]
        if bad:
            print("AUDIT FAILED: " + json.dumps({k: rep[k] for k in ("done_without_segment",
                                                                      "segment_without_done")}),
                  file=sys.stderr)
        return 1 if bad else 0
    if args.cmd == "replay-pending":
        ids = replay_pending()
        print(json.dumps({"replayed": ids}))
        return 0
    if args.cmd == "rebase":
        raw = (sys.stdin.buffer.read() if args.flip_receipt == "-"
               else pathlib.Path(args.flip_receipt).read_bytes())
        try:
            ok = isinstance(json.loads(raw), dict) and bool(json.loads(raw))
        except ValueError:
            ok = False
        if not ok:      # an ssh that failed feeds nothing: never a segment whose pack_sha is sha256(b"")
            print("REFUSED: the flip receipt is empty or not a JSON object; no segment written",
                  file=sys.stderr)
            return 2
        out = args.out or configured_dir()
        if not out:
            print("change log is off (no --out, change_log_dir empty): nothing written")
            return 0
        print(record_rebase(args.pack_id, raw, out))
        return 0
    try:
        counts = verify(args.segment, json.loads(pathlib.Path(args.receipt).read_text()))
    except ChangeLogError as exc:
        print(f"MISMATCH: {exc}", file=sys.stderr)
        return 1
    print("OK " + json.dumps(counts, sort_keys=True))
    return 0


# ------------------------------------------------------------- monotonic seq (task 2508)
#: A segment header carries `seq`, a counter that grows by one per new segment. A dump records
#: `{stamp, seq}` as its cut and the next delta replays `seq > cut.seq`: a file name cannot order
#: same-second segments or show a gap, the counter does both.
SEQ_FILE = ".seq"


def _header_seq(path: pathlib.Path) -> int | None:
    try:
        with open(path, encoding="utf-8") as fh:
            v = json.loads(fh.readline()).get("seq")
    except (OSError, ValueError) as e:
        raise ChangeLogError(f"{path}: header unreadable ({e})") from e
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def latest_cut(log_dir: pathlib.Path | str | None) -> dict | None:
    """`{stamp, seq}` of the newest segment (highest seq among those at the newest stamp), or None
    when the log is off, empty, or the newest segment has no `seq` (a legacy segment)."""
    if not log_dir or not pathlib.Path(log_dir).is_dir():
        return None
    stamped = [(m.group(1), f) for f in pathlib.Path(log_dir).iterdir()
               if (m := re.fullmatch(r"(\d{8}T\d{6}Z)-.+\.jsonl", f.name))]
    if not stamped:
        return None
    top = max(st for st, _ in stamped)
    seqs = [(_header_seq(f), st) for st, f in stamped]
    if any(x is None for x, st in seqs if st == top):
        return None
    known = [x for x, _ in seqs if x is not None]
    return {"stamp": top, "seq": max(known)}      # highest seq over ALL headers (task 2512 C)


def next_seq(out_dir: pathlib.Path) -> int:
    """The seq for a new segment: one above the counter file and the newest segments' headers.
    Refuses when the counter file exists and is not a non-negative integer."""
    n = 0
    f = out_dir / SEQ_FILE
    if f.is_file():
        try:
            n = int(f.read_text().strip())
        except ValueError as e:
            raise ChangeLogError(f"{f}: counter unreadable ({e}); refusing to write a segment") from e
        if n < 0:
            raise ChangeLogError(f"{f}: counter is negative; refusing to write a segment")
    cut = latest_cut(out_dir) if out_dir.is_dir() else None
    return max(n, cut["seq"] if cut else 0) + 1


@DRY.refuses_when_dry("change-log seq counter write")
def _write_seq(out_dir: pathlib.Path, seq: int) -> None:
    f = out_dir / SEQ_FILE
    cur = int(f.read_text().strip()) if f.is_file() else 0
    if seq <= cur:
        return
    tmp = f.with_name(f".seq.tmp{os.getpid()}")
    tmp.write_text(f"{seq}\n")
    os.replace(str(tmp), str(f))


class MissingSegmentError(ChangeLogError):
    """The writer's counter is ahead of the newest segment a reader can see, or a seq is skipped."""


def segment_stamp(name: str) -> str | None:
    """The landing stamp in a segment file name, or None when it is not a segment file."""
    m = _SEGMENT_NAME.fullmatch(name)
    return m.group(1) if m else None


def _list_segment_seqs(d: pathlib.Path, after_stamp: str | None = None) -> list[int]:
    """Seqs of the segment files in `d`. With `after_stamp`, files stamped BEFORE it are skipped
    unopened: the same selection the delta builder makes (task 2592), so a pre-cut file is never read."""
    return sorted(s for f in d.iterdir()
                  if (st := segment_stamp(f.name))
                  if after_stamp is None or st >= after_stamp
                  if (s := _header_seq(f)) is not None)


def assert_segments_complete(log_dir: pathlib.Path | str, after_seq: int = 0,
                             after_stamp: str | None = None) -> int:
    """Task 2588: the highest `seq` read from the segment headers, or MissingSegmentError when the
    writer's `.seq` counter is ahead of it (the newest segment is not synced or was deleted), when
    segments carry a seq but `.seq` is absent, or when a seq above `after_seq` is skipped. An empty
    log passes. Raises ChangeLogError on an unreadable counter or header.

    Read order matters: `.seq` is read BEFORE the segments are listed. The writer lands the segment
    (fsync, rename) and only then bumps `.seq`, so a counter read first can never be ahead of a
    segment that was already on disk; a segment landing between the two reads only makes the listing
    newer than the counter, which passes.

    Task 2592: pass the cut's `after_stamp` and the check opens exactly the files the delta builder
    opens (`LogDeltaBuilder` skips names stamped before the cut), so an unreadable archived pre-cut
    segment cannot refuse a delta that never reads it."""
    d = pathlib.Path(log_dir)
    f = d / SEQ_FILE
    counter = None
    if f.is_file():
        try:
            counter = int(f.read_text().strip())
        except ValueError as e:
            raise ChangeLogError(f"{f}: counter unreadable ({e})") from e
    seqs = _list_segment_seqs(d, after_stamp)
    last = seqs[-1] if seqs else 0
    want = after_seq + 1
    for s in (x for x in seqs if x > after_seq):
        if s > want:
            raise MissingSegmentError(
                f"change-log seq is not contiguous at seq {s}: expected {want}. A segment in the middle "
                f"is missing from {d}; the delta cannot be proven complete. Sync the log or cut a full dump.")
        want = max(want, s + 1)
    if counter is None:
        if seqs:
            raise MissingSegmentError(
                f"{f} is missing but {d} holds segments with a seq: the writer's counter cannot be "
                f"compared, so completeness cannot be proven. Sync the log directory (with `.seq`) or cut a full dump.")
    elif counter > max(last, after_seq):
        raise MissingSegmentError(
            f"the writer's counter {f} is at seq {counter} but the newest segment read holds seq "
            f"{last}: the newest segment is missing (not yet synced or deleted). "
            f"Sync the log directory and retry, or cut a full dump; no delta was written.")
    return last


if __name__ == "__main__":
    sys.exit(main())
