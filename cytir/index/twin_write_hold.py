"""cytir.index.twin_write_hold -- the mechanical hold on twin writers OUTSIDE the drain during an aw_e10
build (task 756 after review 776 item 2).

From T0 to the flip, only the hourly drain may write the twin (spec `archive/aw-e10-rebuild.md` section 2
step 1): its receipts are what the catch-up accounts every change by. Two tools write the twin with
no drain receipt -- the consent backfill (`cyt consent backfill twin`) and the ADR-083
vector delta (`cyt index surface-delta --apply`). A prose hold in a runbook is not a
guard: the catch-up's receipt check is set membership, so a second, receipt-less write to an id the
drain also wrote would pass it unseen. So a HOLD FILE (`write_hold`) makes both tools refuse to write the held collection
while it exists. `release_hold(plan_dir)` removes it (after the flip, or when the build is abandoned);
only the plan that wrote it may remove it. (The aw_e10 builder that wrote the hold is deleted, task 1312;
see git history before that PR.)

Task 864 (review 854 S6) widened the hold to every direct twin writer in the repo: `HELD_WRITERS`
names them, each calls `refuse_if_held` before its first write, and a test pins the list against the
code. The drain is NOT on the list: it is the one writer the catch-up accounts by its receipts.

The file's path: `AW_TWIN_WRITE_HOLD`, else `/var/lib/aw/twin-write-hold.json` (one per box).
"""
from __future__ import annotations

import json
import os
import pathlib
import time

from cytir.index import dry_run as DRY

#: Every twin writer outside the drain IN THIS PACKAGE (module -> what it is). Each refuses while a
#: hold names its target collection (`refuse_if_held`); a test pins this list against the code. The
#: operator tool tree has two more (the consent backfill, the aw_b1-native
#: delta merge); its own test pins those.
HELD_WRITERS = {
    "cytir.ops.twin_patch_fields": "`cyt ops twin-patch-fields --apply`, the T17 verbs (task 845)",
    "cytir.index.upsert": "the wave upsert, twin leg first (task 359)",
    "cytir.index.apply": "the canonical-dump content-pack transaction, armed",
    "cytir.index.supersede": "the page-group reconcile CLI, not `--dry-run`",
    "cytir.index.surface_delta": "`cyt index surface-delta --apply`, the ADR-083 vector delta (task 1822)",
}

ENV = "AW_TWIN_WRITE_HOLD"
DEFAULT_PATH = "/var/lib/aw/twin-write-hold.json"
SCHEMA = "aw.twin-write-hold.v1"


class TwinWriteHeld(RuntimeError):
    """A twin writer outside the drain was asked to write a collection an aw_e10 build holds."""


def hold_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get(ENV) or DEFAULT_PATH)


def read_hold() -> dict | None:
    p = hold_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # an unreadable hold is still a hold: fail closed
        return {"schema": SCHEMA, "collection": "*", "plan_dir": "?", "unreadable": str(p)}


@DRY.refuses_when_dry("the aw_e10 twin-writer hold (write)")
def write_hold(*, collection: str, source_qdrant: str, plan_dir: str, t0: str | None = None) -> pathlib.Path:
    """Write (or keep) the hold for `plan_dir`. A hold of ANOTHER plan refuses: two builds of one
    twin at once is not a state the catch-up can account for."""
    p = hold_path()
    cur = read_hold()
    if cur and cur.get("plan_dir") != str(plan_dir):
        raise TwinWriteHeld(f"REFUSED: {p} already holds the twin for another build ({cur.get('plan_dir')})")
    if cur:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps({"schema": SCHEMA, "collection": collection, "source_qdrant": source_qdrant,
                               "plan_dir": str(plan_dir), "t0": t0,
                               "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "why": "aw_e10 build: only the drain writes the twin until the flip "
                                      "(cytir.index.twin_write_hold)"}, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def refuse_if_held(collection: str, *, tool: str) -> None:
    """Raise `TwinWriteHeld` when a hold names `collection` (or is unreadable). Called by every twin
    writer outside the drain before its first write."""
    cur = read_hold()
    if cur and cur.get("collection") in (collection, "*"):
        raise TwinWriteHeld(
            f"REFUSED: {tool} would write {collection!r}, which an aw_e10 build holds since "
            f"{cur.get('t0') or cur.get('written_at')} ({hold_path()}, plan {cur.get('plan_dir')}): only "
            "the drain writes the twin until the flip. Release with `cytir.index.twin_write_hold."
            "release_hold(<plan dir>)` once the build is flipped or abandoned")


#: The argparse names a write CLI takes its target collection(s) under (review 869 S1).
CLI_TARGET_ARGS = ("collection", "twin_collection", "coll", "dst", "target", "new", "coll_b")


def refuse_cli_write(args=None, *, tool: str, extra=()) -> None:
    """A write CLI's one line (review 869 S1): refuse (SystemExit) when ANY collection it may write
    -- the `CLI_TARGET_ARGS` its parsed `args` carry, plus `extra` fixed names -- is the collection an
    aw_e10 build holds. Run right after the arguments are parsed, dry run or not: during the freeze a
    twin tool has no business against the held twin at all."""
    names = [getattr(args, a, None) for a in CLI_TARGET_ARGS] if args is not None else []
    for c in [*names, *extra]:
        if isinstance(c, str) and c:
            try:
                refuse_if_held(c, tool=tool)
            except TwinWriteHeld as exc:
                raise SystemExit(str(exc)) from exc


@DRY.refuses_when_dry("the aw_e10 twin-writer hold (release)")
def release_hold(plan_dir: str) -> bool:
    cur = read_hold()
    if not cur:
        return False
    if cur.get("plan_dir") != str(plan_dir):
        raise TwinWriteHeld(f"REFUSED: the hold at {hold_path()} belongs to {cur.get('plan_dir')}, not {plan_dir}")
    hold_path().unlink()
    return True
