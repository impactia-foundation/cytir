"""`cyt ops twin-patch-fields`: set or unset a CLOSED list of payload keys on TWIN rows, by id (task 845).

WHY. Ruling 29 T17 (Tim 2026-10-10): the L1 re-fetch (batches 00000/00001) rewrote rows in place on
the twin and (Q1) dropped the `license` the old row held on 4,234 of them, and (stamp patch) moved
`rule_epochs.chunk` 5 -> 1 and lost `stamps` on 5,587 rows whose chunk text is byte-identical
(tasks 830, 831, 834). The twin is the source of the `aw_e10` build, so the twin must hold the old
value again BEFORE that build. A content pack would re-write the vectors and needs no GPU-free path;
no queue kind patches a payload key. This verb is that path, and nothing wider:

  * TWIN ONLY. The target is `cytir.core.twin` (or `--twin-url/--twin-collection`); a target that
    `prod_guard.is_prod` calls prod is refused before any call. Prod gets the value at the flip.
  * A CLOSED key list. `PATCHABLE` names the keys this verb may touch (`license`, `rule_epochs`,
    `stamps`, and from task 2734 `snippet`); `--fields` must be a subset, and every input row may set or unset only those. A new
    key is a code change with a review, never a flag.
  * BY ID, WITH A PRECONDITION PER ROW. Each input row names the values it expects the twin row to
    hold NOW (`expect`, which must include `content_digest`). A row whose twin value moved since the
    map was built (a later pack, the Q2 re-run) is `precondition_failed` and blocks `--apply`; a row
    that is gone is `absent` and is only counted. A row already at its target is `already`
    (idempotent: a second run writes nothing).
  * DRY BY DEFAULT. Without `--apply` it reads, classifies and writes a receipt; the process runs
    under `cytir.index.dry_run.guard`, so any write primitive it reached would raise.
  * `--apply` NEEDS `--export PATH`: the prior value of every key it will change, written BEFORE the
    first write, in the verb's own input format. Rollback is the same verb on that file:
    `cyt ops twin-patch-fields --from PATH --fields <same> --apply --export <new path>`.
  * RAW KEYS ONLY. On a collection declared with a coded payload profile (`normalized`,
    `rebuild_e10`: the twin from the aw_e10 flip) the verb refuses any key that profile codes or drops
    in `--fields` or `expect` (`payload_profile.PROFILE_RULES`); a verbatim key (`snippet`, task 2734)
    patches as on a raw twin. `content_digest` is compared as stored (32 hex).
  * THE aw_e10 HOLD. `--apply` is refused while an aw_e10 build holds this collection
    (`cytir.index.twin_write_hold`, from T0 to the flip): only the drain writes the twin then (task 864).
  * LOCKS AND THE DRAIN. `--apply` holds the promotion unit's lock (`promote_queue.hold_unit_lock`,
    the hourly drain's flock) and the cross-task `TWIN-WRITE-LOCK` for the whole write, and refuses
    while a content pack waits for its prod leg (a `*-content-pack-twin-state.json` without `DONE`)
    or a pack is `CARRIED-BY-FLIP`: review 836 S1, a twin-only write beside a pending prod leg makes
    the G2 readback refuse and stops the drain.
  * VERIFIED. After the write every patched id is read back; a row not at its target, or whose
    `content_digest` moved, ends the run REFUSED (exit 4).
  * ACCOUNTED. The ids written are listed in `<receipt-dir>/<run>-twin-written-ids.json`, the
    receipt shape the deleted `aw.index.rebuild_e10.load_receipts` read (task 1312; the `rebuild_e10` profile key stays) (`catchup --receipts <receipt-dir>`), so
    a patch that lands after an `aw_e10` T0 is a named writer, not `CatchupUnaccounted`.

INPUT (`--from`, JSON lines, one row per id):
    {"id": "<uuid>", "set": {"license": "reference-open"}, "unset": [],
     "expect": {"content_digest": "<hex>", "license": null, "provenance": "l1_backfill_refetch"}}
`expect` values compare by canonical JSON; `null` matches a missing key.

EXIT 0 = nothing blocks (dry: the plan; apply: written and verified). 3 = at least one
`precondition_failed` row (nothing written). 4 = REFUSED (bad input, prod target, a lock, a pending
prod leg, an export that exists, a read or write error, a failed readback).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import pathlib
import sys
import time
from typing import Any, Callable, Iterable

from cytir.index import dry_run as DRY
from cytir.index import point_delete as PD
from cytir.index import prod_guard as PG

PATCHABLE = ("license", "rule_epochs", "stamps", "snippet")
RECEIPT_SCHEMA = "aw.twin-patch-fields-receipt.v1"
EXIT_OK, EXIT_PRECONDITION, EXIT_REFUSED = 0, 3, 4
BATCH = 500
EXAMPLES = 20
LOCK_NAME = "TWIN-WRITE-LOCK"
TERMINALS = ("patch", "already", "absent", "precondition_failed")


class Refused(RuntimeError):
    """Nothing may be claimed or written (exit 4)."""


# --------------------------------------------------------------------------------- the input
def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _same(a: Any, b: Any) -> bool:
    """Equality by canonical JSON; a missing key reads as None."""
    return _canon(a) == _canon(b)


def parse_fields(text: str) -> tuple[str, ...]:
    fields = tuple(f.strip() for f in (text or "").split(",") if f.strip())
    if not fields:
        raise Refused("--fields is empty")
    bad = sorted(set(fields) - set(PATCHABLE))
    if bad:
        raise Refused(f"--fields {bad} not patchable: this verb touches only {list(PATCHABLE)} "
                      "(a new key is a reviewed code change)")
    return fields


def load_rows(path: pathlib.Path, fields: Iterable[str]) -> tuple[list[dict], str]:
    """The validated input rows and the file's sha256. Any malformed row refuses the whole file."""
    allowed = set(fields)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise Refused(f"--from {path} cannot be read: {exc}") from exc
    rows, seen, problems = [], set(), []
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError as exc:
            problems.append(f"line {n}: not JSON ({exc})")
            continue
        pid, sets, unset, expect = r.get("id"), r.get("set") or {}, r.get("unset") or [], r.get("expect")
        if not isinstance(pid, str) or not pid:
            problems.append(f"line {n}: no id")
            continue
        if pid in seen:
            problems.append(f"line {n}: duplicate id {pid}")
        seen.add(pid)
        if not isinstance(sets, dict) or not isinstance(unset, list):
            problems.append(f"line {n}: `set` must be an object and `unset` a list")
            continue
        if not sets and not unset:
            problems.append(f"line {n}: nothing to set or unset")
        touched = set(sets) | set(unset)
        if touched - allowed:
            problems.append(f"line {n}: keys {sorted(touched - allowed)} are outside --fields {sorted(allowed)}")
        if set(sets) & set(unset):
            problems.append(f"line {n}: keys {sorted(set(sets) & set(unset))} are both set and unset")
        if not isinstance(expect, dict) or not expect.get("content_digest"):
            problems.append(f"line {n}: `expect` must name the row's content_digest (the text this patch is for)")
            continue
        rows.append({"id": pid, "set": sets, "unset": list(unset), "expect": expect})
    if problems:
        raise Refused(f"--from {path}: {len(problems)} bad row(s): " + "; ".join(problems[:5]))
    if not rows:
        raise Refused(f"--from {path} holds no row")
    return rows, hashlib.sha256(raw).hexdigest()


def read_keys(rows: list[dict]) -> list[str]:
    keys = {"content_digest"}
    for r in rows:
        keys |= set(r["set"]) | set(r["unset"]) | set(r["expect"])
    return sorted(keys)


# ---------------------------------------------------------------------------------- the plan
def classify(row: dict, current: dict | None) -> tuple[str, list[str]]:
    """(terminal, the expect keys that did not hold) for one row against the twin's row now."""
    if current is None:
        return "absent", []
    moved = [k for k, v in sorted(row["expect"].items()) if not _same(current.get(k), v)]
    at_target = (all(_same(current.get(k), v) for k, v in row["set"].items())
                 and all(k not in current for k in row["unset"]))
    if at_target:
        return "already", []
    if moved:
        return "precondition_failed", moved
    return "patch", []


def build_plan(rows: list[dict], current: dict[str, dict]) -> dict:
    by: dict[str, list[dict]] = {t: [] for t in TERMINALS}
    moved_keys: dict[str, int] = {}
    examples: list[dict] = []
    for r in rows:
        t, moved = classify(r, current.get(r["id"]))
        by[t].append(r)
        for k in moved:
            moved_keys[k] = moved_keys.get(k, 0) + 1
        if t == "precondition_failed" and len(examples) < EXAMPLES:
            cur = current.get(r["id"]) or {}
            examples.append({"id": r["id"], "moved": moved,
                             "expected": {k: r["expect"][k] for k in moved},
                             "found": {k: cur.get(k) for k in moved}})
    return {"by": by, "moved_keys": moved_keys, "examples": examples}


def export_rows(rows: list[dict], current: dict[str, dict]) -> list[dict]:
    """The rollback file: for each row about to change, the twin's prior value of every key it
    touches (`set` the ones present, `unset` the ones absent), expecting the patched state."""
    out = []
    for r in rows:
        cur = current[r["id"]]
        keys = list(r["set"]) + list(r["unset"])
        prior_set = {k: cur[k] for k in keys if k in cur}
        prior_unset = [k for k in keys if k not in cur]
        expect = {"content_digest": cur.get("content_digest")}
        expect.update({k: r["set"].get(k) for k in keys})
        out.append({"id": r["id"], "set": prior_set, "unset": prior_unset, "expect": expect})
    return out


def write_groups(rows: list[dict], batch: int, *, coalesce: bool = False) -> list[tuple[str, Any, list[str]]]:
    """(op, payload or keys, ids): one call per distinct set-dict / unset-list, `batch` ids each.

    `coalesce` (task 2734, `snippet`: one distinct value per row) packs the set-groups into
    `multi` calls: `what` is a list of `{"payload", "points"}` operations, at most `batch` points
    per call (qdrant `points/batch`), instead of one HTTP call per row."""
    sets: dict[str, tuple[dict, list[str]]] = {}
    unsets: dict[str, tuple[list, list[str]]] = {}
    for r in rows:
        if r["set"]:
            sets.setdefault(_canon(r["set"]), (r["set"], []))[1].append(r["id"])
        if r["unset"]:
            unsets.setdefault(_canon(sorted(r["unset"])), (sorted(r["unset"]), []))[1].append(r["id"])
    out = []
    if coalesce:
        cur: list[dict] = []
        n = 0
        for _, (what, ids) in sorted(sets.items()):
            for i in range(0, len(ids), batch):
                part = ids[i:i + batch]
                if cur and n + len(part) > batch:
                    out.append(("multi", cur, [p for o in cur for p in o["points"]]))
                    cur, n = [], 0
                cur.append({"payload": what, "points": part})
                n += len(part)
        if cur:
            out.append(("multi", cur, [p for o in cur for p in o["points"]]))
        sets = {}
    for op, groups in (("set", sets), ("unset", unsets)):
        for _, (what, ids) in sorted(groups.items()):
            for i in range(0, len(ids), batch):
                out.append((op, what, ids[i:i + batch]))
    return out


# ------------------------------------------------------------------------- the drain contract
def pending_prod_legs(queue_dir: pathlib.Path) -> list[str]:
    """Packs whose twin leg has a state record and no DONE marker, and packs CARRIED-BY-FLIP.

    Review 836 S1: a twin-only write while one of these waits makes the content-pack resume's G2
    readback refuse a declared divergence, and the ABORTED result stops the whole drain."""
    found = []
    art = queue_dir / "_artifacts"
    for f in sorted(art.glob("*-content-pack-twin-state.json")) if art.is_dir() else []:
        pack = f.name[:-len("-content-pack-twin-state.json")]
        try:
            stage = json.loads(f.read_text()).get("stage")
        except (OSError, ValueError):
            stage = "unreadable"
        if not (queue_dir / pack / "DONE").exists():
            found.append(f"{pack} (twin state {stage!r}, no DONE)")
    found += [f"{m.parent.name} (CARRIED-BY-FLIP)" for m in sorted(queue_dir.glob("*/CARRIED-BY-FLIP"))]
    return found


@contextlib.contextmanager
def apply_locks(task: str):
    """The promotion unit's flock, then TWIN-WRITE-LOCK (refused when held by another task)."""
    from cytir.index import promote_queue as PQ   # the drain's own lock helper, not a second copy
    from cytir.ops import write_lock as WL
    with PQ.hold_unit_lock() as unit:
        if unit is None:
            # `hold_unit_lock` is inert when the `promotion_lock` setting is empty (a stranger's box
            # has no drain). For a twin write that is not "no lock needed", it is "no exclusion":
            # the hourly drain could land a pack between this verb's read and its write.
            raise Refused("the `promotion_lock` setting is empty, so the drain's flock cannot be "
                          "held: set CYTIR_PROMOTION_LOCK to the unit's lock file "
                          "(`flock -n <file>` in aw-promotion-nightly.service)")
        WL.assert_locks_free(task)
        path = WL.acquire(LOCK_NAME, task, "cyt ops twin-patch-fields")
        try:
            yield path
        finally:
            WL.release(LOCK_NAME, task)


# ------------------------------------------------------------------------------- I/O wrappers
def fetch(url: str, coll: str, ids: list[str], keys: list[str], batch: int) -> dict[str, dict]:
    return PD.fetch_payload_fields(url, coll, ids, keys, batch=batch)


def write(url: str, coll: str, op: str, what, ids: list[str]) -> None:
    if op == "multi":
        PD.set_payload_batch(url, coll, what)
    elif op == "set":
        PD._api(url, f"/collections/{coll}/points/payload?wait=true", {"payload": what, "points": ids})
    else:
        PD._api(url, f"/collections/{coll}/points/payload/delete?wait=true", {"keys": what, "points": ids})


def _write_json(path: pathlib.Path, doc: Any) -> str:
    data = (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return hashlib.sha256(data).hexdigest()


def _write_lines(path: pathlib.Path, rows: list[dict]) -> str:
    data = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as fh:          # never overwrite an export
        fh.write(data)
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------------- main
def _parser(prog: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    ap.add_argument("--from", dest="src", required=True, help="the input rows (JSON lines)")
    ap.add_argument("--fields", required=True, help=f"comma list, a subset of {','.join(PATCHABLE)}")
    ap.add_argument("--apply", action="store_true", help="write (default: dry plan only)")
    ap.add_argument("--export", help="with --apply: the rollback file (must not exist)")
    ap.add_argument("--twin-url")
    ap.add_argument("--twin-collection")
    ap.add_argument("--queue-dir", help="default: the queue_dir setting")
    ap.add_argument("--receipt-dir", help="default: <work_root>/twin-patch-fields")
    ap.add_argument("--task", default="845", help="the TWIN-WRITE-LOCK owner label")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to sleep between write calls")
    return ap


def run(argv: list[str], *, prog: str = "cyt ops twin-patch-fields", settings=None,
        fetch_fn: Callable = fetch, write_fn: Callable = write, locks: Callable = apply_locks,
        now: Callable[[], dt.datetime] | None = None, out=print, sleep_fn: Callable = time.sleep) -> int:
    args = _parser(prog).parse_args(argv)
    clock = now or (lambda: dt.datetime.now(dt.timezone.utc))
    started = clock()
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, "mode": "apply" if args.apply else "dry",
                               "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"), "input": args.src}
    code, receipt_dir = EXIT_OK, None
    try:
        if settings is None:
            from cytir.core import paths
            settings = paths.load_settings()
        receipt_dir = pathlib.Path(args.receipt_dir or pathlib.Path(settings.work_root) / "twin-patch-fields")
        from cytir.core import twin as TW
        url = (args.twin_url or TW.url()).rstrip("/")
        coll = args.twin_collection or TW.collection()
        receipt["target"] = {"url": url, "collection": coll}
        if PG.is_prod(url, coll):
            raise Refused(f"{url} ({coll}) is a prod target: this verb writes the twin only")
        # task 864 (review of task 855's reader ratchet): this verb reads, compares and writes RAW
        # payload values. From the aw_e10 flip the twin is CODED (enum codes, a 32-hex digest): a
        # raw `expect` never equals a coded row, and a raw write would store an uncoded value. The
        # declaration says which profile the twin stores; any coded profile is refused, dry or not.
        from cytir.index import row_decode as RD
        prof = RD.profile_of(coll)
        fields = parse_fields(args.fields)
        rows, sha = load_rows(pathlib.Path(args.src), fields)
        receipt.update(fields=list(fields), input_sha256=sha, rows=len(rows))
        if prof not in (None, "prod"):
            # Task 2734: a coded profile refuses only what it CODES. A key the profile stores
            # verbatim (`snippet`: free text, no rule in `PROFILE_RULES`) reads and writes the same
            # on a coded twin; `content_digest` is compared as STORED (its 32-hex form, which the
            # producer copies from the row), so a raw 64-hex expectation is a precondition failure,
            # never a wrong write. Any other coded or dropped key in `--fields` or `expect` refuses.
            from cytir.index import payload_profile as PP
            coded = set(PP.PROFILE_RULES.get(prof, ())) - {RD.DIGEST_KEY}
            touched = set(fields) | {k for r in rows for k in r["expect"]}
            if touched & coded or prof not in PP.PROFILE_RULES:
                raise Refused(f"{coll} is declared with the CODED payload profile {prof!r}: this verb "
                              f"reads and writes raw values; {sorted(touched & coded)} are coded or dropped "
                              "there (the raw twin before the aw_e10 T0 only)")
        if args.apply and not args.export:
            raise Refused("--apply needs --export PATH (the prior values, for rollback)")
        if args.apply and pathlib.Path(args.export).exists():
            raise Refused(f"--export {args.export} exists: an export is never overwritten")
        ids = [r["id"] for r in rows]

        def classify_now() -> tuple[dict, dict]:
            current = fetch_fn(url, coll, ids, read_keys(rows), args.batch)
            plan = build_plan(rows, current)
            receipt.update(counts={t: len(v) for t, v in plan["by"].items()},
                           moved_keys=plan["moved_keys"], precondition_examples=plan["examples"])
            return current, plan

        if not args.apply:
            with DRY.guard(active=True):
                current, plan = classify_now()
        else:
            # task 864 (review 854 S6): from an aw_e10 T0 to the flip only the drain writes the twin
            # (the catch-up accounts every twin change by the drain's receipts). Refused while the
            # builder's hold names this collection; the T17 patches land before T0 (runbook P5).
            from cytir.index import twin_write_hold as TWH
            try:
                TWH.refuse_if_held(coll, tool="cyt ops twin-patch-fields")
            except TWH.TwinWriteHeld as exc:
                raise Refused(str(exc)) from exc
            queue_dir = pathlib.Path(args.queue_dir or settings.queue_dir)
            pending = pending_prod_legs(queue_dir)
            if pending:
                raise Refused(f"{len(pending)} pack(s) wait for their prod leg (review 836 S1): "
                              + "; ".join(pending[:5]))
            with locks(args.task) as lock:
                receipt["lock"] = lock
                pending = pending_prod_legs(queue_dir)   # re-read under the drain's lock
                if pending:
                    raise Refused(f"{len(pending)} pack(s) wait for their prod leg (review 836 S1): "
                                  + "; ".join(pending[:5]))
                current, plan = classify_now()
                if plan["by"]["precondition_failed"]:
                    code = EXIT_PRECONDITION
                else:
                    todo = plan["by"]["patch"]
                    receipt["export"] = {"path": args.export, "rows": len(todo),
                                         "sha256": _write_lines(pathlib.Path(args.export),
                                                                export_rows(todo, current))}
                    # The ids this run writes, BEFORE the first write: a run that dies half way
                    # still names every row it may have touched (the aw_e10 catch-up's receipt).
                    written_file = receipt_dir / f"twin-patch-fields-{stamp}-{sha[:12]}-twin-written-ids.json"
                    _write_json(written_file, sorted(r["id"] for r in todo))
                    receipt["written_ids_file"] = str(written_file)
                    calls = 0
                    for op, what, chunk in write_groups(todo, args.batch, coalesce="snippet" in fields):
                        write_fn(url, coll, op, what, chunk)
                        calls += 1
                        if args.pause:
                            sleep_fn(args.pause)
                    receipt["write_calls"] = calls
                    back = fetch_fn(url, coll, [r["id"] for r in todo], read_keys(rows), args.batch)
                    bad = []
                    for r in todo:
                        b = back.get(r["id"])
                        ok = (b is not None
                              and _same(b.get("content_digest"), current[r["id"]].get("content_digest"))
                              and classify(r, b)[0] == "already")
                        if not ok:
                            bad.append(r["id"])
                    receipt["verified"] = {"read_back": len(todo), "not_at_target": len(bad),
                                           "examples": bad[:EXAMPLES]}
                    if bad:
                        raise Refused(f"{len(bad)} row(s) are not at their target after the write "
                                      f"(e.g. {bad[:3]}): roll back with --from {args.export}")
        if plan["by"]["precondition_failed"]:
            code = EXIT_PRECONDITION
    except Refused as exc:
        code, receipt["refused"] = EXIT_REFUSED, str(exc)
    except DRY.DryRunWriteRefused as exc:            # a dry run reached a write primitive
        code, receipt["refused"] = EXIT_REFUSED, f"dry run reached a write: {exc}"
    except Exception as exc:                          # nothing may end the run without a receipt
        code, receipt["refused"] = EXIT_REFUSED, f"{type(exc).__name__}: {exc}"
    receipt["exit"] = code
    receipt["finished"] = clock().strftime("%Y-%m-%dT%H:%M:%SZ")
    c = receipt.get("counts", {})
    out(f"[twin-patch-fields] {receipt['mode']} exit={code} rows={receipt.get('rows')} "
        + " ".join(f"{t}={c.get(t, 0)}" for t in TERMINALS)
        + (f" REFUSED: {receipt['refused']}" if "refused" in receipt else ""))
    if receipt_dir is not None:
        try:
            path = receipt_dir / f"twin-patch-fields-{stamp}-{(receipt.get('input_sha256') or 'noinput')[:12]}-{receipt['mode']}.json"
            _write_json(path, receipt)
            out(f"[twin-patch-fields] receipt {path}")
        except OSError as exc:
            out(f"[twin-patch-fields] REFUSED: the receipt cannot be written: {exc}")
            code = EXIT_REFUSED
    return code


def main(argv: list[str] | None = None, prog: str = "cyt ops twin-patch-fields") -> int:
    return run(list(sys.argv[1:] if argv is None else argv), prog=prog)


if __name__ == "__main__":
    raise SystemExit(main())
