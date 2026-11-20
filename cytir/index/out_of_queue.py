"""cytir.index.out_of_queue -- the receipts of a writer that applied packs OUTSIDE the promotion queue.

Task 834 (finding A) named one such writer: the task-626 news catch-up runner applied
`news626f-20260921` (14,464 rows) to twin AND prod on 2026-09-23, from its own work dir
(`newsfresh_294/`, operator setting `news_work`). It wrote receipts, but not in the queue's
`_artifacts/` tree: apply ledgers (`*ledger*.jsonl`) whose rows name the pack artifact by
`artifact_sha256` + `artifact_ids`, and the aged-out / re-emit tombstones by
`tombstone_pointers_sha256`. The ids are the artifact's own rows and the pointer file's
`predecessor_id`s.

Two readers, one definition:
* `cyt ops mirror-audit` (task 837) declares that pack's ids (`apply-ledger-pack` source);
* the `aw_e10` catch-up (task 847, ruling 29 T8 (h)) accounts a twin change by these ledgers too,
  so an out-of-queue apply after T0 is NAMED (and reported apart), never mistaken for a writer with
  no receipt -- and never silently missed by a scan that reads the queue only.

Task 863 (S1 of review 854): only an APPLIED artifact counts (a plan row followed by a twin-write
outcome), dated by its own rows, never by the file's mtime.

Stdlib only.
"""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Iterable

POINTS_GLOB = "news-points.{sha}.jsonl.gz"
TOMBS_GLOB = "tombstone-pointers.{sha}.json"
LEDGER_GLOBS = ("*ledger*.jsonl", "*/*ledger*.jsonl")


class OutOfQueueError(RuntimeError):
    """A ledger names an artifact that cannot be found or does not hash to the ledger's sha256."""


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as raw:
        for chunk in iter(lambda: raw.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_point_ids(points: str, sha: str, expected: int | None = None) -> list[str]:
    """The ids of a pack artifact, after its sha256 (and, when given, its row count) is checked
    against the ledger's. A difference is a refusal, never a silent re-read."""
    try:
        got = _sha256_file(points)
        with gzip.open(points, "rt", encoding="utf-8") as fh:
            ids = [json.loads(line)["id"] for line in fh if line.strip()]
    except (OSError, ValueError, KeyError) as exc:
        raise OutOfQueueError(f"pack artifact {points} cannot be read: {exc}") from exc
    if got != sha:
        raise OutOfQueueError(f"pack artifact {points}: sha256 {got[:12]} is not the ledger's {sha[:12]}")
    if expected is not None and len(ids) != expected:
        raise OutOfQueueError(f"pack artifact {points} holds {len(ids)} ids, the ledger says {expected}")
    return ids


def tombstone_ids(pointers: str, sha: str) -> list[str]:
    """The `predecessor_id` of every pointer of an explicit tombstone-pointer file (schema
    `aw.explicit-tombstone-pointers.v1`). Lists of bare ids (`aged_out_dropped_as_*`: rows that were
    NOT tombstoned) are not pointers and are not read."""
    try:
        if _sha256_file(pointers) != sha:
            raise OutOfQueueError(f"tombstone pointers {pointers}: sha256 is not the ledger's {sha[:12]}")
        doc = json.loads(open(pointers, encoding="utf-8").read())
    except (OSError, ValueError) as exc:
        raise OutOfQueueError(f"tombstone pointers {pointers} cannot be read: {exc}") from exc
    out = []
    for v in (doc.values() if isinstance(doc, dict) else []):
        if isinstance(v, list):
            out += [str(x["predecessor_id"]) for x in v if isinstance(x, dict) and x.get("predecessor_id")]
    return out


def _find(root: str, pattern: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    return hits[0] if hits else None


#: Task 863 S1 (review 854): the outcome rows that PROVE an artifact reached the twin. A row that
#: names an artifact (`CONTENT-PACK-PLAN`, `RESTORE-PLAN`, ...) is only a PLAN; the real ledgers name
#: artifacts on plan rows of dry runs (`preflight-dry-*`) and of runs that ended `FAILED` before
#: writing. An artifact's ids count as an apply only when one of these follows its plan row in the
#: same ledger, before the next plan row. A `FAILED` after a twin verification does not undo the
#: twin write (task 600 ledgers: `TWIN-VERIFIED` then `FAILED` on the prod leg), so it still counts.
APPLIED_OUTCOMES = frozenset({"TWIN-VERIFIED", "RESTORE-VERIFIED", "TWIN-PROD-VERIFIED-OK"})


def _entry_epoch(entry: dict) -> float | None:
    """The row's own time (`ts`/`at`), or None. Never the file's mtime: one unrelated append re-dates
    every undated row of the file (review 854 S1 b)."""
    ts = entry.get("ts") or entry.get("at")
    if isinstance(ts, str):
        try:
            return datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            return None
    return None


def _status(entry: dict) -> str:
    return str(entry.get("status") or entry.get("event") or entry.get("stage") or "")


def ledger_applies(path: str) -> tuple[list[dict], list[dict]]:
    """(applied, plan_only) of ONE ledger file, in file order. An entry is a plan row (it names
    `artifact_sha256` and/or `tombstone_pointers_sha256`) with the outcome that followed it:
    `applied` when an `APPLIED_OUTCOMES` row follows before the next plan row (its time = that
    outcome row's time, else the plan row's, else None = undated), else `plan_only`."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if isinstance(e, dict):
                rows.append(e)
    applied, plan_only = [], []
    cur = None

    def close():
        if cur is not None:
            (applied if cur["outcome"] else plan_only).append(cur)

    for e in rows:
        if e.get("artifact_sha256") or e.get("tombstone_pointers_sha256"):
            close()
            cur = {"sha": e.get("artifact_sha256"), "tsha": e.get("tombstone_pointers_sha256"),
                   "artifact_ids": e.get("artifact_ids"), "plan_status": _status(e),
                   "plan_epoch": _entry_epoch(e), "outcome": None, "epoch": None,
                   "statuses": [_status(e)]}
            continue
        if cur is None:
            continue
        st = _status(e)
        cur["statuses"].append(st)
        if cur["outcome"] is None and st in APPLIED_OUTCOMES:
            cur["outcome"] = st
            cur["epoch"] = _entry_epoch(e) or cur["plan_epoch"]
    close()
    return applied, plan_only


def ledger_receipts(dirs: Iterable[str], since_epoch: float) -> tuple[dict[str, str], dict]:
    """{id: "oob_written:<ledger>" | "oob_tombstoned:<ledger>"} of every APPLIED artifact (see
    `APPLIED_OUTCOMES`) in every `*ledger*.jsonl` of `dirs` (and one level down), whose apply time (the
    outcome row's `ts`/`at`, else its plan row's) is at or after `since_epoch`. Not counted, and
    reported apart in the summary: plan rows with no applied outcome (`plan_only`, dry runs and failed
    runs) and applies with no time of their own (`undated_applied`). Both directions are safe: a twin
    change no counted receipt names stops the catch-up (exit 4) and is reported, never explained by a
    plan that did not run. A named artifact of a COUNTED apply that is missing or does not hash to
    the ledger's sha256 raises `OutOfQueueError`."""
    ids: dict[str, str] = {}
    files, rows_since = [], 0
    plan_only_since, undated = [], []
    seen: set[tuple[str, str]] = set()
    for d in dirs:
        for pat in LEDGER_GLOBS:
            for f in sorted(glob.glob(os.path.join(d, pat))):
                if os.path.getmtime(f) < since_epoch:
                    continue          # nothing in a file last written before T0 can be after it
                stem = os.path.basename(f)[: -len(".jsonl")]
                applied, plan_only = ledger_applies(f)
                for a in plan_only:
                    if a["plan_epoch"] is None or a["plan_epoch"] >= since_epoch:
                        plan_only_since.append({"ledger": f, "artifact_sha256": a["sha"],
                                                "tombstone_pointers_sha256": a["tsha"],
                                                "statuses": a["statuses"]})
                n_f = 0
                for a in applied:
                    if a["epoch"] is None:
                        undated.append({"ledger": f, "artifact_sha256": a["sha"], "statuses": a["statuses"]})
                        continue
                    if a["epoch"] < since_epoch:
                        continue
                    rows_since += 1
                    n_f += 1
                    sha, tsha = a["sha"], a["tsha"]
                    if sha and ("p", sha) not in seen:
                        seen.add(("p", sha))
                        pts = _find(d, POINTS_GLOB.format(sha=sha))
                        if pts is None:
                            raise OutOfQueueError(f"{f}: names pack artifact {sha[:12]}, which no "
                                                  f"{POINTS_GLOB.format(sha='<sha>')} under {d} holds")
                        for x in pack_point_ids(pts, sha, a["artifact_ids"]):
                            ids.setdefault(str(x), f"oob_written:{stem}")
                    if tsha and ("t", tsha) not in seen:
                        seen.add(("t", tsha))
                        tp = _find(d, TOMBS_GLOB.format(sha=tsha))
                        if tp is None:
                            raise OutOfQueueError(f"{f}: names tombstone pointers {tsha[:12]}, which "
                                                  f"no file under {d} holds")
                        for x in tombstone_ids(tp, tsha):
                            ids.setdefault(str(x), f"oob_tombstoned:{stem}")
                if n_f:
                    files.append({"file": f, "rows_since": n_f})
    return ids, {"dirs": list(dirs), "ledgers_with_rows_since": files, "rows_since": rows_since,
                 "ids_unique": len(ids), "applied_outcomes": sorted(APPLIED_OUTCOMES),
                 "plan_only_not_counted": plan_only_since, "undated_applied_not_counted": undated}
