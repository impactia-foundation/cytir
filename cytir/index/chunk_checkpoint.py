"""A chunk-level resume record for the drain's long prod calls (task 786).

THE INCIDENT. Tranche 14's pack (1,409,761 rows) passed its twin leg and then aborted on its first prod
call: one of the 2,820 id retrieves of `already_present` answered HTTP 408. Without a record, an attempt that
dies after thousands of chunks re-asks (or re-sends) all of them on the 30 GB box that has to keep serving search.

WHERE IT HELPS, AND WHERE IT DOES NOT (task 787 review, must-fix 3).
  * LOOKUP record: an aborted lookup (retries spent) is re-entered at the first unfinished chunk. Kept ONLY
    when the lookup itself raised; any other exit spends it (a stale "present" answer must never replace prod).
  * WRITE record, CONTENT packs: the prod resume (`resume_content_pack_prod_only`) skips the batches already
    sent, then asks prod for every id.
  * WRITE record, SEALED packs: it is NOT reached after a partial prod write. Once any batch has landed,
    `resume_prod_only` -> `apply_side` -> `plan(prod)` raises AlreadyLandedError and the drain records
    "REFUSED live-apply-preflight" every run (an old #414 defect, same on main). Closing that is its own task.

WHAT. One JSONL file per (pack, role, step). Each line is one finished chunk: `{"k": <key>, "found": [...]}`.
The key is a digest of the collection name and the chunk's ids, so a chunk is skipped only when the same ids
were finished against the same collection (an alias that moved, or a resealed pack, is a different key and
is asked again). The file is read once at construction; a torn last line (a kill mid-write) is dropped.

WHAT IT IS NOT. It is not a proof. The terminal membership check of the transaction still asks prod, with no
checkpoint, after the last write: a skipped chunk that prod lost would be caught there.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import threading

from cytir.index import dry_run as DRY


def chunk_key(collection: str, ids: list) -> str:
    h = hashlib.sha1(collection.encode())
    for i in ids:
        h.update(b"\n")
        h.update(str(i).encode())
    return h.hexdigest()


class ChunkCheckpoint:
    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)
        self._lock = threading.Lock()
        self._done: dict[str, list[str]] = {}
        self.resumed = 0          # chunks served from the file in this process
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        for line in text.splitlines():
            try:
                rec = json.loads(line)
                self._done[rec["k"]] = list(rec.get("found") or [])
            except (ValueError, KeyError, TypeError):
                continue      # a torn line: that chunk is simply asked again

    def get(self, key: str) -> list[str] | None:
        with self._lock:
            found = self._done.get(key)
            if found is not None:
                self.resumed += 1
            return found

    def __len__(self) -> int:
        return len(self._done)

    @DRY.refuses_when_dry("chunk checkpoint write")
    def put(self, key: str, found: list[str]) -> None:
        with self._lock:
            self._done[key] = list(found)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as out:
                out.write(json.dumps({"k": key, "found": list(found)}) + "\n")
                out.flush()
                os.fsync(out.fileno())

    @DRY.refuses_when_dry("chunk checkpoint clear")
    def clear(self) -> None:
        with self._lock:
            self._done.clear()
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
