"""Per-pack count declaration for a Route-C materializer or audit.

A Route-C pack is sealed against the record counts its route resolved on the
day it was built. Those counts are data about one pack, not code: they live in
a small declaration file, and the materializer and the audit both read it.
A missing, unreadable or mismatched declaration is a refusal; nothing here
ever guesses a count.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "aw.routec-pack-declaration.v1"


def load_declaration(path: Path | str | None, *, source_id: str,
                     keys: tuple[str, ...]) -> dict[str, int]:
    """Return the declared counts for `source_id`; raise ValueError on any doubt."""
    if not path:
        raise ValueError(f"Route-C pack declaration required for {source_id}")
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Route-C pack declaration unreadable: {path}") from error
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"Route-C pack declaration has the wrong schema: {path}")
    if document.get("source_id") != source_id:
        raise ValueError(
            f"Route-C pack declaration is for {document.get('source_id')!r}, "
            f"not {source_id!r}")
    counts = document.get("counts")
    if not isinstance(counts, dict):
        raise ValueError(f"Route-C pack declaration lacks counts: {path}")
    missing = [key for key in keys if key not in counts]
    bad = [key for key in keys if key in counts
           and (not isinstance(counts[key], int) or isinstance(counts[key], bool)
                or counts[key] < 0)]
    if missing or bad:
        raise ValueError(
            f"Route-C pack declaration for {source_id}: missing {missing}, invalid {bad}")
    return {key: counts[key] for key in keys}
