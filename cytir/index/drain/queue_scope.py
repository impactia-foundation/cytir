"""Drain queue scope: startup refusals, the live-queue and `Single` state-dir gates, pack ordering.

Task 2890 (RULING 80, drain split step 5): moved verbatim out of `cytir/index/promote_queue.py`, which
re-exports every name here. A moved body that calls a name which stays in the facade, or that a test
patches on the facade, reads it at call time through `_Q` (the facade module).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import re
import sys

from cytir.core import paths as PATHS
from cytir.core.paths import SettingRefused, SettingsError, box_file_setting
from cytir.index import dry_run as DRY
from cytir.index import live_target as LT
from cytir.index import payload_patch as PPK
from cytir.index import payload_profile as PP
from cytir.index import prod_guard as PG
from cytir.index import promote_sealed_pack as PSP
from cytir.index import sparse_rule as SR
from cytir.index.drain.common import QueueError, Single, atomic_write_text, default_queue_dir, now
from cytir.index.drain.receipts import read_content_pack_twin_state

ROOT = pathlib.Path(__file__).resolve().parents[3]


MANIFEST_SCHEMA = "aw.promotion-queue-pack.v1"


#: The ONE tracked field this driver's gate switch reads. See the module
#: docstring: a receipt PR flips `profile` once PR #302 (ADR-110) merges.
GATE_CONTRACT_PATH = ROOT / "data" / "promotion-gate-contract.v1.json"


GATE_CONTRACT_SCHEMA = "aw.promotion-gate-contract.v1"


def apply_startup_refusal() -> int | None:
    """What an `--apply` drain needs, resolved before the queue is read: the prod writer-mutex
    host, the write-lock location (host and directory; `AW_LOCK_*` or the operator settings) and
    the twin ssh host (unless the twin leg is deliberately off, `AW_APPLY_NO_TWIN=1`). Returns the
    exit code (2) after naming each unconfigured setting on stderr, else None. Writes nothing."""
    from cytir.ops import write_lock  # noqa: PLC0415
    bad = PATHS.refused_settings("prod_lock_host")
    try:
        write_lock.effective_location()
    except PATHS.SettingRefused as exc:
        bad.append(f"lock_host/lock_dir: {exc}")
    if os.environ.get("AW_APPLY_NO_TWIN") != "1" and not os.environ.get("AW_TWIN_SSH_HOST"):
        bad.extend(PATHS.refused_settings("twin_ssh_host"))
    for line in bad:
        print(f"[REFUSED] promote_queue --apply: setting {line}", file=sys.stderr, flush=True)
    return 2 if bad else None


def sparse_rule_startup_refusal(prod_url: str, prod_alias: str) -> int | None:
    """Task 717: what an `--apply` twin-first drain needs BEFORE its first twin PUT. The prod leg
    writes under the BM25 prune rule its physical collection declares, and the term set is read
    there; a missing or wrong file found at the prod leg would leave the twin one pack ahead of
    prod (the twin-mirror rule). So the alias is resolved and the rule loaded (and hashed, once:
    `SparseRule.for_collection` keeps it) here. A declared collection with no `sparse_rule` key
    refuses the same way. Prints ONE `[REFUSED]` line and returns 2, else None. Writes nothing.
    An unreachable prod refuses too: an apply drain cannot write without it, and the per-pack
    refusals a late failure would leave exit 0."""
    from cytir.index import promote_queue as _Q
    try:
        physical = _Q.resolve_prod_alias(prod_url, prod_alias)
    except Exception as exc:  # noqa: BLE001 -- any failure here means the rule cannot be proved
        print(f"[REFUSED] promote_queue --apply: cannot resolve alias {prod_alias!r} on {prod_url} to "
              f"check the BM25 prune rule before the twin leg: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 2
    # task 847: a `staged` prod collection (aw_e10 after the alias move, before the flip commit sets
    # it live) refuses here, before the twin leg, not pack by pack. (Undeclared stays the per-pack
    # projector's refusal, as before.)
    declared = PP.load_profiles()
    if physical in declared and PP.is_staged(declared[physical]):
        print(f"[REFUSED] promote_queue --apply: prod alias {prod_alias!r} resolves to {physical!r}, "
              "which is declared `staged`: the writer contract is off until the flip commit sets it "
              "`live` (task 847 runbook F7)", file=sys.stderr, flush=True)
        return 2
    try:
        rule = SR.SparseRule.for_collection(physical)
    except SR.SparseRuleError as exc:
        print(f"[REFUSED] promote_queue --apply: {exc}", file=sys.stderr, flush=True)
        return 2
    if rule is None:
        print(f"[sparse-rule] prod collection {physical!r} declares no BM25 prune rule (explicit null)",
              flush=True)
    else:
        print(f"[sparse-rule] prod collection {physical!r} writes under {rule.id} epoch {rule.epoch}; "
              f"term set {rule.drop.path} sha256 {rule.drop.sha256} ({len(rule.drop)} terms), "
              "loaded once for this run", flush=True)
    return None


def plan_startup_refusal() -> int | None:
    """What EVERY drain, dry or `--apply`, reads for its plan: the corpus store root (the
    persisted-input index and the stores behind the "pages are persisted" check). Returns 2 after
    naming it on stderr when unconfigured, else None. Writes nothing."""
    return PATHS.startup_refusal("promote_queue", "corpus_root")


def single_startup_refusal(*, apply: bool) -> int | None:
    """What a `Single` drain (a stranger's, `--target-url`) reads, resolved before the queue is
    opened. NOT `prod_lock_host`, `twin_ssh_host` or the live lock location: it has no prod and no
    twin. It reads (1) under `--apply`, the lock registry its one `SINGLE-WRITE-LOCK-<collection>`
    lives in (`AW_WORK_ROOT`, or `AW_LOCK_HOST` + `AW_LOCK_DIR`: the box that has neither has no
    place to lock, and a drain that cannot lock must not mark every pack REFUSED and exit 0);
    (2) `corpus_root`, only when `AW_PERSISTED_INDEX_SQLITE_DIR` names an index (the persisted-input
    check then reads the persisted stores; otherwise the check is RECORDED as skipped and no store
    is read). Queue, state dir and artifacts are explicit arguments, never settings. Returns 2
    after naming each unconfigured setting on stderr, else None. Writes nothing."""
    from cytir.index import promote_queue as _Q
    from cytir.ops import write_lock  # noqa: PLC0415
    bad: list[str] = []
    if apply:
        try:
            write_lock.effective_location()
        except PATHS.SettingRefused as exc:
            bad.append(f"lock_host/lock_dir: {exc}")
    if _Q._single_persisted_check():
        bad.extend(PATHS.refused_settings("corpus_root"))
    for line in bad:
        print(f"[REFUSED] promote_queue --target-url: setting {line}", file=sys.stderr, flush=True)
    return 2 if bad else None


# ------------------------------------------------------------ the switch
def gate_profile(contract_path: pathlib.Path = GATE_CONTRACT_PATH) -> dict:
    """The ONE field that decides mechanical vs battery, read from the
    tracked contract file -- never a live probe of git/GitHub state. See the
    module docstring for why a nightly cron must not depend on network
    access to decide what it may legally write.
    """
    if not contract_path.exists():
        raise QueueError(
            f"no gate contract at {contract_path} -- refusing to guess a "
            "profile. This file is the one home for the switch; it must "
            "exist and name a profile explicitly.")
    doc = json.loads(contract_path.read_text())
    if doc.get("schema") != GATE_CONTRACT_SCHEMA:
        raise QueueError(
            f"{contract_path}: schema {doc.get('schema')!r}, expected "
            f"{GATE_CONTRACT_SCHEMA!r}")
    profile = doc.get("profile")
    if profile not in ("mechanical", "battery-required"):
        raise QueueError(
            f"{contract_path}: profile {profile!r} is not one of "
            "'mechanical'/'battery-required' -- refusing rather than "
            "defaulting to either.")
    return {"adr": doc.get("adr"), "pr": doc.get("pr"), "profile": profile,
           "contract_path": str(contract_path), "checked_at": now()}


# ---------------------------------------------------------------- queue IO
def read_manifest(pack_dir: pathlib.Path) -> dict:
    manifest_path = pack_dir / "manifest.json"
    ready_path = pack_dir / "READY"
    if not manifest_path.exists():
        raise QueueError(f"{pack_dir.name}: no manifest.json")
    if not ready_path.exists():
        raise QueueError(f"{pack_dir.name}: no READY -- not queued yet")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    ready_sha = ready_path.read_text().strip().split()[0].lower()
    if manifest_sha != ready_sha:
        raise QueueError(
            f"{pack_dir.name}: READY names sha {ready_sha[:12]} but "
            f"manifest.json is now {manifest_sha[:12]} -- edited after being "
            "marked ready. Refusing rather than draining a manifest nobody "
            "re-approved (same integrity discipline as launch_b01.sh's "
            "preflight sha checks).")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise QueueError(
            f"{pack_dir.name}: manifest schema {manifest.get('schema')!r}, "
            f"expected {MANIFEST_SCHEMA!r}")
    if manifest.get("kind") == PPK.KIND:
        # Task 2748: a payload-only value patch (no vectors); the kind module owns the shape.
        try:
            PPK.check_manifest(manifest)
        except PPK.PayloadPatchError as exc:
            raise QueueError(f"{pack_dir.name}: {exc}") from exc
        required = ("pack_id", "family", "collection")
    elif manifest.get("kind") == "content_pack":
        # Already-projected points (the news lane's own pipeline resolved
        # identity/licence/embedding) + optional tombstones -- a different
        # shape from a sealed vector pack, so a different required-field set.
        required = ("pack_id", "family", "pack_points", "rows", "collection")
    else:
        required = ("pack_id", "family", "pack_chunks", "vectors", "mrl", "collection")
    # `is None`, not bare truthiness -- bug found live, task 660,
    # 2026-09-25: `not manifest.get("rows")` treats a legitimate `rows: 0`
    # (task 653's pure-tombstone pack: zero new points, only deletions) the
    # same as the field being absent, refusing it at intake forever with
    # "manifest missing ['rows']". A present-but-zero value is not a
    # missing field for any of these keys.
    missing = [f for f in required if manifest.get(f) is None]
    if missing:
        raise QueueError(f"{pack_dir.name}: manifest missing {missing}")
    return manifest


def assert_real_path(manifest: dict) -> int:
    """The real-path smoke test: refuse a pack with 0 passages, before any
    lock or any plan runs against a live collection (memory
    `smoke-test-yield-end-to-end`: a pipeline ran a month yielding 0
    passages with green tests and nothing refused at the start).

    A `content_pack` with 0 points but a genuine `tombstones_count` > 0 is
    NOT the yielded-nothing defect this guards against -- it is a
    deliberate pure-delete pack (task 653/ADR-103 amendment 1: consent-
    declined pages carry no replacement row, only a takedown). Refusing
    THAT would refuse the only mechanism that removes a declined
    publisher's pages from the index. Still refuses a pack with BOTH 0
    points and 0 tombstones -- that combination has no possible effect on
    the index at all and is exactly the silent-yield shape.
    """
    from cytir.index import promote_queue as _Q
    if manifest.get("kind") == PPK.KIND:
        return manifest["count"]        # check_manifest (intake) already refused count < 1
    if manifest.get("kind") == "content_pack":
        for _ in _Q.load_content_pack_points(manifest["pack_points"]):
            return 1
        if manifest.get("tombstones_count"):
            return 1
        raise QueueError(
            f"{manifest['pack_id']}: 0 points in {manifest['pack_points']} "
            "and 0 tombstones -- refusing at the start, before any lock or "
            "plan (a pack with neither adds nor deletes anything).")
    for _ in PSP.read_records(manifest["pack_chunks"]):
        return 1  # at least one row exists; that is all this check asserts
    raise QueueError(
        f"{manifest['pack_id']}: 0 passages under {manifest['pack_chunks']} "
        "-- refusing at the start, before any lock or plan.")


def _pack_estimate_ops(pack_dir: pathlib.Path, artifact_root: pathlib.Path
                       ) -> tuple[bool, float]:
    """`(has_pending_resume, estimated_ops)` for ONE pack -- best-effort,
    NEVER raises: an unreadable/malformed manifest or a non-content_pack
    reads as `(False, 0.0)` (the main loop's own `read_manifest()` is
    still what decides whether it is actually a REFUSED-at-intake pack;
    this is scheduling, not judgment). A resume with stage
    `"twin-write-complete"` only has the PROD leg left (1x ops); a fresh
    pack or a `"twin-write-started"` replay still needs both legs (2x)."""
    try:
        manifest_bytes = (pack_dir / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError):
        return False, 0.0
    if manifest.get("kind") != "content_pack":
        return False, 0.0
    rows = manifest.get("rows") or 0
    tomb = manifest.get("tombstones_count") or 0
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    state = read_content_pack_twin_state(artifact_root, pack_dir.name,
                                         manifest_sha=manifest_sha)
    if state is not None:
        ops = (rows + tomb) if state.get("stage") == "twin-write-complete" \
            else 2 * (rows + tomb)
        return True, float(ops)
    return False, float(2 * (rows + tomb))


def _pack_family_best_effort(pack_dir: pathlib.Path) -> str:
    """The pack's declared `family` -- required for both content_pack and
    sealed manifests, so this is reliable when the manifest parses at
    all. An unreadable manifest or a missing family gets a SINGLETON
    group keyed by the pack's own name (its own name is already unique
    among pack_dirs), never silently merged into some other pack's
    family -- that would let this ordering reshuffle two UNRELATED
    packs' relative order, exactly what finding 1 says never to do."""
    try:
        manifest = json.loads((pack_dir / "manifest.json").read_bytes())
    except (OSError, json.JSONDecodeError):
        return f"__unknown__:{pack_dir.name}"
    fam = manifest.get("family")
    return fam if fam else f"__unknown__:{pack_dir.name}"


def order_pack_dirs_for_drain(pack_dirs: list[pathlib.Path],
                              artifact_root: pathlib.Path) -> list[pathlib.Path]:
    """Order `drain()`'s pack_dirs for review round 3 findings 1+2.

    Finding 1 (HIGH): a size/resumable-first sort that reordered
    INDIVIDUAL packs (round 2's version) broke ordering WITHIN a family --
    e.g. a larger `news-daily-DAY_N+1` pack could sort ahead of the
    smaller `news-daily-DAY_N` it depends on landing first, and DAY_N's
    later apply would overwrite DAY_N+1's rows and resurrect tombstones
    DAY_N+1 had already aged out. Fix: group by `family` FIRST; within a
    group, packs stay in plain NAME order (chronological for date-named
    packs like `news-daily-YYYYMMDD`) -- ordering only ever happens
    BETWEEN families, never within one.

    Finding 2 (MEDIUM), preserved: resumable families sort before fresh
    ones (closes a known twin!=prod gap while the budget is fullest --
    finding 1's "never skip a resume" fix depends on resumes being
    reachable early), then largest-estimated-total-ops families first
    among the rest, so a family is never systematically starved by
    always landing last in an unbudgeted alphabetical order.

    A family's own priority is `(has ANY pending resume, total estimated
    ops across the WHOLE family)` -- summed, not per-pack, since the
    family is the scheduling unit now."""
    groups: dict[str, list[pathlib.Path]] = {}
    for p in sorted(pack_dirs, key=lambda p: p.name):
        groups.setdefault(_pack_family_best_effort(p), []).append(p)

    def group_key(fam: str) -> tuple:
        members = groups[fam]
        estimates = [_pack_estimate_ops(p, artifact_root) for p in members]
        has_resume = any(r for r, _ in estimates)
        total_ops = sum(ops for _, ops in estimates)
        return (0 if has_resume else 1, -total_ops, fam)

    ordered_families = sorted(groups.keys(), key=group_key)
    return [p for fam in ordered_families for p in groups[fam]]


def _dry_run_never_writes(fn):
    """`drain(apply=False)` runs under `DRY.guard`: no Qdrant write, no lock, no ledger, no queue
    file, whatever branch it takes (2026-10-03, see cytir/index/dry_run.py). `dry_receipt_dir`, a
    dry run's only output on disk, must lie outside the queue."""
    import functools

    @functools.wraps(fn)
    def wrapper(queue_dir, *args, apply: bool, dry_receipt_dir=None, **kwargs):
        if kwargs.get("full_persisted_index") and not apply:
            # Task 2957: looked up at CALL time. The decorator now runs while `cytir.index.drain.drain` is
            # being imported by the facade; under `python -m cytir.index.promote_queue` an import of the
            # canonical facade here, at decoration time, would re-enter that half-loaded module.
            from cytir.index import promote_queue as _Q
            raise QueueError(_Q.FULL_INDEX_NEEDS_APPLY)
        if dry_receipt_dir is not None:
            if apply:
                raise QueueError("dry_receipt_dir is for a dry run; an --apply run writes "
                                 "receipts into the queue")
            dry_receipt_dir = pathlib.Path(dry_receipt_dir)
            if dry_receipt_dir.resolve().is_relative_to(pathlib.Path(queue_dir).resolve()):
                raise QueueError(f"dry_receipt_dir {dry_receipt_dir} lies inside the queue "
                                 f"{queue_dir}: a dry run writes nothing into the queue")
        # EVERY drain run without --apply is a replay (incident 2026-10-07: two review replays
        # scrolled prod 187k times; widened from "with --dry-receipt-dir" on review). A replay
        # never queries prod: no id retrieve, no count, no scroll -- `cytir.index.prod_guard.replay`.
        with DRY.guard(not apply, receipts_to=dry_receipt_dir), \
                (PG.replay() if not apply else contextlib.nullcontext()):
            return fn(queue_dir, *args, apply=apply, **kwargs)
    return wrapper


def _single_record(single: Single) -> dict:
    from cytir.index import promote_queue as _Q
    return {"kind": "single", "url": single.url, "collection": single.collection,
            "payload_profile_of": single.payload_profile_of,
            "persisted_inputs": "checked" if _Q._single_persisted_check() else "skipped"}


def _single_transaction_key(single: Single | None) -> str | None:
    return None if single is None else f"single|{single.url}|{single.collection}"


_LOCK_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _inside_or_equal(path: pathlib.Path, live: pathlib.Path | None) -> bool:
    return live is not None and (path == live or live in path.parents)


#: A file the LIVE unit's wrapper (`promote_nightly.sh`) creates in the queue
#: it drains, if absent, at every start. It lives IN the queue, so no environment variable, scratch
#: work root or overlay can move it away from the queue: a path the caller names that is that
#: queue, inside it, above it (one level) or an ancestor of it carries the marker. The drain
#: ignores it (it lists directories only: `queue_dir.iterdir()` filtered by `is_dir()`).
LIVE_QUEUE_MARKER = "LIVE-QUEUE"


def box_file_queue_dir() -> pathlib.Path | None:
    """The live queue according to the BOX file (`/etc/aw/operator.json`) alone: the `queue_dir`
    setting resolved from that file with an empty environment, by the settings module itself, so a
    box file that names only `work_root` (impactIA's does) yields `<work_root>/promotion-queue`. No
    environment variable (`AW_WORK_ROOT`, `AW_QUEUE_DIR`, `AW_OPERATOR_OVERLAY`) can move it. None
    when the box has no file, or the file leaves the queue unconfigured. Raises `SettingsError` for
    a box file that exists and cannot be read (the callers refuse: it cannot be proven that no
    queue is live)."""
    named = box_file_setting("queue_dir")
    return pathlib.Path(named).resolve() if named else None


def marked_live_queue(path: pathlib.Path) -> pathlib.Path | None:
    """The directory that carries `LIVE_QUEUE_MARKER` and that `path` equals, lies inside (any
    ancestor of `path`), or sits directly above (a child of `path`); else None. A deeper
    descendant is not searched (a queue's parent is covered by the settings' own paths)."""
    p = pathlib.Path(path).resolve()
    for d in (p, *p.parents):
        if (d / LIVE_QUEUE_MARKER).is_file():
            return d
    if p.is_dir():
        try:
            for child in p.iterdir():
                if child.is_dir() and (child / LIVE_QUEUE_MARKER).is_file():
                    return child
        except OSError:
            return None
    return None


def live_queue_refusal(queue: pathlib.Path) -> str | None:
    """Why `queue` (the queue a verb is asked to read or run) is live, or None. Three sources, and
    none of them is the caller's environment alone: (1) the `queue_dir` setting; (2) the BOX file's
    `queue_dir`, read without any override; (3) a `LIVE_QUEUE_MARKER` in the queue directory. The
    queue is live when it equals, lies inside it or contains it (any ancestor: a path above a live
    queue is refused too, because a drain there would read the live queue as a pack directory)."""
    from cytir.index import promote_queue as _Q
    queue = pathlib.Path(queue).resolve()
    lives: list[tuple[str, pathlib.Path]] = []
    try:
        configured = _Q.live_queue_dir()
        boxed = box_file_queue_dir()
    except SettingsError as exc:
        return (f"the box file cannot be read ({exc}), so it cannot be proven that {queue} is not "
                "the live promotion queue")
    if configured is not None:
        lives.append(("the `queue_dir` setting", configured))
    if boxed is not None:
        lives.append(("the box file's `queue_dir`", boxed))
    for why, live in lives:
        if queue == live or live in queue.parents or queue in live.parents:
            return f"{queue} is the live promotion queue ({live}, from {why})"
    marked = marked_live_queue(queue)
    if marked is not None:
        return (f"{queue} is, or is beside, the live promotion queue {marked}: it carries a "
                f"{LIVE_QUEUE_MARKER} marker (the hourly unit's wrapper puts it there)")
    return None


def single_state_dir(single: Single, artifact_root: pathlib.Path) -> pathlib.Path:
    """The directory the transaction owns: its receipts, refusal cache and DONE markers."""
    return pathlib.Path(single.state_dir) if single.state_dir else (
        pathlib.Path(artifact_root) / "single-state" / single.collection)


def _in_pack_area(path: pathlib.Path, queue: pathlib.Path) -> bool:
    """`path` is the queue dir itself, or lies under a NON-underscore child of it: a pack dir, where
    a receipt or a DONE marker would hide the pack from the twin-first drain. `queue/_artifacts` and
    other `_`-prefixed children are not pack dirs (the drain skips them)."""
    if path == queue:
        return True
    if queue in path.parents:
        return not path.relative_to(queue).parts[0].startswith("_")
    return False


def _overlaps_queue(path: pathlib.Path, queue: pathlib.Path) -> bool:
    """A state dir or artifact root that is equal to, above, or in the pack area of `queue`."""
    path = pathlib.Path(path).resolve()
    return path in queue.parents or _in_pack_area(path, queue)


def refuse_live_single(single: Single, queue_dir: pathlib.Path,
                       artifact_root: pathlib.Path) -> pathlib.Path:
    """Everything a `Single` drain must refuse BEFORE it reads the queue or writes anywhere
    (reviews of #553: F1, F2, R1, R2; round 4 F2). Returns the state dir the transaction owns. Pure
    path and read-only network checks.

    Two sets of queues are protected, and the second does not depend on any setting. (1) The queue
    the drain READS (`queue_dir`, whatever it is): a state dir or artifact root equal to, above or
    in its pack area (a non-`_` child) is refused, because a DONE or a receipt written there lands
    in a pack dir and the twin-first drain then skips, or loses the evidence of, that pack. (2) The
    configured live queue (the `queue_dir` setting; a box with none has nothing more to protect):
    refused as the queue read, and refused for a state dir or artifact root anywhere in it."""
    from cytir.index import promote_queue as _Q
    live = _Q.live_queue_dir()
    queue = pathlib.Path(queue_dir).resolve()
    if live is not None and (queue == live or live in queue.parents or queue in live.parents):
        raise QueueError(
            f"a Single transaction refuses the live promotion queue ({live}): pass an explicit "
            "--queue-dir that is a copy WITHOUT the LIVE-QUEUE marker, or a build queue of "
            "its own")
    root = single_state_dir(single, artifact_root)
    if _inside_or_equal(root.resolve(), live):
        raise QueueError(f"a Single transaction's state dir may not sit in the live queue ({live})")
    if _inside_or_equal(pathlib.Path(artifact_root).resolve(), live):
        raise QueueError(f"a Single transaction's artifact root may not sit in the live queue "
                         f"({live})")
    # The marker (round 6, follow-up of #559): the queue is live when the queue itself says so,
    # whatever the settings and the environment hold.
    marked = marked_live_queue(queue)
    if marked is not None:
        raise QueueError(f"a Single transaction refuses the live promotion queue ({marked} carries "
                         f"a {LIVE_QUEUE_MARKER} marker): pass an explicit --queue-dir that is a "
                         "copy WITHOUT the LIVE-QUEUE marker, or a build queue of its own")
    for what, path in (("state dir", root), ("artifact root", pathlib.Path(artifact_root))):
        marked = next((d for d in (path.resolve(), *path.resolve().parents)
                       if (d / LIVE_QUEUE_MARKER).is_file()), None)
        if marked is not None:
            raise QueueError(f"a Single transaction's {what} may not sit in the live queue "
                             f"({marked} carries a {LIVE_QUEUE_MARKER} marker)")
    for what, path in (("state dir", root), ("artifact root", pathlib.Path(artifact_root))):
        if _overlaps_queue(path, queue):
            raise QueueError(
                f"a Single transaction's {what} {path} is equal to, above, or inside a pack dir of "
                f"the queue it reads ({queue}): a DONE marker or receipt written there would hide "
                "the pack from the twin-first drain. Refused, nothing written")
    if not _LOCK_SAFE.match(single.collection):
        raise QueueError(f"{single.collection!r} is not a plain collection name")
    _Q.assert_single_target_not_live(single)
    return root.resolve()


#: Written into a state entry on its first use by a `Single` transaction. An entry WITH it is ours; an
#: entry WITHOUT it that holds anything a pack dir or a drain holds is somebody else's (round 5,
#: follow-up A of #553: a `--target-state-dir` pointing at a different queue turned a pack's receipt from
#: ABORTED into LANDED when the settings did not know that queue).
SINGLE_STATE_MARKER = "SINGLE-STATE"


def _dead_marker_tmps(entry: pathlib.Path) -> list[pathlib.Path]:
    """`.SINGLE-STATE.tmp<pid>` files a crashed claim left (`atomic_write_bytes` writes the marker
    through `.<name>.tmp<pid>`): only those whose pid is dead. A live pid's file is a claim in
    progress and still counts as content."""
    out = []
    prefix = f".{SINGLE_STATE_MARKER}.tmp"
    for c in entry.iterdir():
        if c.name.startswith(prefix) and c.name[len(prefix):].isdigit() and c.is_file():
            try:
                os.kill(int(c.name[len(prefix):]), 0)
            except ProcessLookupError:
                out.append(c)
            except OSError:
                pass            # alive (or not ours to signal): treat as in progress
    return out


def _foreign_state_entry(entry: pathlib.Path) -> list[str]:
    """What makes the existing `entry` somebody else's. A state entry is ours when it carries the
    marker, and claimable when it is ABSENT or EMPTY. Anything else (round 6, follow-up A of #559: a
    pack dir still being sealed holds `vectors/` and `passages.jsonl` and no manifest yet, and was
    claimed) is foreign: the names it holds are returned. A non-directory at the path is foreign."""
    if not entry.exists() and not entry.is_symlink():
        return []
    if not entry.is_dir():
        return [entry.name]
    if (entry / SINGLE_STATE_MARKER).exists():
        return []
    dead = _dead_marker_tmps(entry)
    return sorted(c.name for c in entry.iterdir() if c not in dead)


def refuse_state_entries(root: pathlib.Path, pack_ids: list[str],
                         queue: pathlib.Path | None = None) -> None:
    """R1 (review of #553): a state dir's `<pack_id>` entry that is a symlink, or that resolves
    into the live queue or into the pack area of the queue being read, would send a receipt or a
    DONE into a live pack dir. Refused before any pack runs, for every pack the run will touch.
    Follow-up A (round 5, widened in round 6): also refused, whatever queue it is in and whether or
    not the settings know that queue, an existing entry that is not EMPTY and that this transaction
    did not mark (no `SINGLE_STATE_MARKER`): a pack dir, still being sealed or sealed, or a drain's
    receipts. Only an absent or an empty entry is claimed."""
    from cytir.index import promote_queue as _Q
    live = _Q.live_queue_dir()
    read = pathlib.Path(queue).resolve() if queue is not None else None
    for pack_id in pack_ids:
        entry = pathlib.Path(root) / pack_id
        target = entry.resolve()
        if (entry.is_symlink() or _inside_or_equal(target, live)
                or (read is not None and _in_pack_area(target, read))):
            raise QueueError(f"a Single transaction's state entry {entry} is a symlink or leads "
                             f"into the live queue or the queue it reads ({read or live}): "
                             "refused, nothing written")
        foreign = _foreign_state_entry(entry)
        if foreign:
            raise QueueError(f"a Single transaction's state entry {entry} already holds {foreign} "
                             f"and no {SINGLE_STATE_MARKER} marker: it is a pack dir or another "
                             "run's state, not this transaction's. Refused, nothing written")


def claim_state_entry(entry: pathlib.Path) -> None:
    """First use of a state entry: re-check it (a second look, right before the first write) and mark
    it ours. Idempotent."""
    foreign = _foreign_state_entry(entry)
    if foreign:
        raise QueueError(f"state entry {entry} holds {foreign} and no {SINGLE_STATE_MARKER} marker: "
                         "refused, nothing written")
    if not (entry / SINGLE_STATE_MARKER).exists():
        if entry.is_dir():
            for stale in _dead_marker_tmps(entry):
                stale.unlink(missing_ok=True)
        atomic_write_text(entry / SINGLE_STATE_MARKER, now() + "\n")


def assert_single_target_not_live(single: Single) -> None:
    """Refuse a live target: a `Single` transaction writes a BUILD collection under its own lock
    and no gate, so it must never be pointed at what serves. Read-only. Called at plan time (drain
    start) and again, per pack, under the write lock -- the alias can move between the two.

    The endpoint is asked what it HOLDS (`cytir.index.live_target.refuse_live_target`): one that carries
    the prod alias (`single.prod_alias`) or any declared live collection is live under every
    spelling of its URL, and refused for every collection name; an alias name as the target is
    refused; the listing is read fail-closed."""
    try:
        LT.refuse_live_target(single.url, single.collection, prod_aliases=(single.prod_alias,))
    except LT.LiveTargetRefused as exc:
        raise QueueError(f"a Single transaction refuses its target: {exc}") from exc


def refuse_live_queue_run(argv: list[str]) -> str | None:
    """The reason `cyt index promote` must not run these args, or None. Round 5, B4 of #559; round 6.

    The hourly unit runs its wrapper `promote_nightly.sh`, which takes the unit's flock, resets the
    checkout to origin/main, sets the persisted-input index (`AW_PERSISTED_INDEX_SQLITE_DIR`) and the
    budget and throttle flags before it execs the drain. This verb is none of that. On the LIVE queue
    (the queue itself, inside it, or above it) it therefore REFUSES outright, dry or `--apply`, and
    names the unit.

    "Live" is not decided by the caller's environment (round 5 compared with the `queue_dir`
    setting, which `AW_WORK_ROOT` or `AW_QUEUE_DIR` moves, and let the live queue through): it is the
    union of (1) the `queue_dir` setting, (2) the box file's `queue_dir` read directly, and (3) a
    `LIVE-QUEUE` marker in the queue directory (`live_queue_refusal`). Only a variable that
    changes WHICH files are read can defeat it, and none of the three reads one: the marker and the
    box file are on disk. It runs only on a queue none of them calls live: a stranger's box (no
    box file, no marker), a scratch queue. `--help` is never refused. The unit itself does not go
    through this verb.

    LIMIT (by design): only `cyt index promote` is guarded. Running this module any other way
    (the script shim, `python -m cytir.index.promote_queue`, `main()` or `drain()` called directly)
    skips all three checks and the lock, because the hourly unit runs through that shim and no
    new check may sit on its path."""
    if any(a in ("-h", "--help") for a in argv):
        return None
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--queue-dir", default=None)
    known, _ = ap.parse_known_args(argv)
    try:
        queue = pathlib.Path(known.queue_dir or default_queue_dir()).resolve()
    except SettingRefused:
        return None        # no queue is named and none is configured: `main` refuses by name itself
    why = live_queue_refusal(queue)
    if why is None:
        return None
    return (f"`cyt index promote` does not run on the live promotion queue: {why}. The hourly unit "
            "owns it (its flock, its checkout reset, its persisted index, its budget). "
            "Start the unit instead: `systemctl start aw-promotion-nightly.service`. This verb "
            "runs on a queue that is not live (`--queue-dir <other>`)")
