"""Sha-pinned point-id exclusion files that a promotion-queue manifest may name
(task 680, 2026-09-30).

WHY THIS EXISTS
---------------
Some sealed packs carry rows the ratified admission floor does not admit. The
OpenAlex L2 citation ladder (ADR-102) trims 966,649 of 6,404,873 points; the
trim is already computed and sealed as a list of point ids
(task 629's `openalex-l2-trim-receipt.v1.json`). The drain can already
drop rows by id -- `promote_queue.language_scope_and_waiver_exclusions` builds an
`exclude_ids` map for the language rules and hands it to
`promote_sealed_pack.plan()`, which reports every dropped row with its reason
(ADR-086: every row reaches exactly one terminal). This module is the third
source of that map: a manifest names a file, its sha256 and a reason, and the
drain excludes exactly those ids, on the read-only dry plan and on the write
alike (promote_queue derives the map in ONE place for both).

Why a file with a hash and not a filtered copy of the pack: the vectors are
12.8 GB of sealed fp16 shards whose receipts bind their bytes; rewriting them to
drop a sixth of the rows would invalidate every receipt to express a rule that
is one list of ids. The pack stays whole and sealed, and the rule is data.

Fail loud (engineering-doctrine #3): a wrong hash, an unreadable file or a file
that excludes nothing refuses. A manifest that names an exclusion file which
silently excluded nothing would admit the rows it was written to keep out.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


class ExcludeFileError(RuntimeError):
    """The named exclusion file cannot be trusted; the pack must not drain."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_exclude_ids(path: str, *, sha256: str | None,
                     reason: str | None) -> dict[str, str]:
    """`{point_id: reason}` from a `.jsonl` / `.jsonl.gz` of `{"point_id": ...}`
    rows. A row may carry its OWN `"reason"` (ADR-116: one file can hold ids excluded for
    different causes, and each id's terminal must say its own); a row without one gets the
    manifest's `reason`. Refuses without a pinned hash and a reason, on a hash mismatch, and
    on a file that holds no ids."""
    if not sha256:
        raise ExcludeFileError(
            f"{path}: manifest names an exclusion file without "
            "`exclude_ids_sha256` -- an unpinned file can change under a "
            "drain, and the rows it keeps out would change with it")
    if not reason:
        raise ExcludeFileError(
            f"{path}: manifest names an exclusion file without "
            "`exclude_ids_reason` -- every dropped row is reported with one")
    p = Path(path)
    if not p.is_file():
        raise ExcludeFileError(f"{path}: exclusion file does not exist")
    actual = sha256_file(p)
    if actual != sha256:
        raise ExcludeFileError(
            f"{path}: sha256 {actual} is not the manifest's pinned {sha256}")
    opener = gzip.open if p.name.endswith(".gz") else open
    ids: dict[str, str] = {}
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids[str(row["point_id"])] = row.get("reason") or reason
    if not ids:
        raise ExcludeFileError(
            f"{path}: holds 0 point ids -- a file that excludes nothing is a "
            "mis-declared pack, not an empty rule")
    return ids
