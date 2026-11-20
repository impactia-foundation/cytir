"""The two row counts of a sealed pack's manifest, and the one way to read them.

A pack has TWO different numbers, and `manifest["rows"]` meant one or the other by how the pack
was composed (hygiene 2026-10-04):

    rows_staged     rows in the sealed stage the pack points at (what was verified and embedded)
    rows_to_write_max   an UPPER BOUND on the rows the drain writes: rows_staged less the ids dropped
                    at compose, that is the already-served ids (ADR-116) and the ids of the caller's
                    own exclusion file that are in this pack. The drain also drops language-scope,
                    waiver, undated and projection rows (ERIC: 3,291,605 staged, 2,214,072 written),
                    so NEVER compare it for equality with a written count; the receipt's `written`
                    is the count.

`rows` stays as a documented ALIAS and is never rewritten: a pack composed before these names
exist has only `rows`, and every reader below falls back to it, so such a pack is read exactly as
it always was. For a new pack `rows` keeps the value it has always had (the compose-time to-write bound when the
served check dropped ids, else `rows_staged`), so a tool that still reads `rows` sees no change.
New code reads the name it means:

    cap or backlog arithmetic (queue depth, family ceiling)  -> `rows_to_write_max`
    "what did the stage hold"                                -> `rows_staged`

Content packs (`kind == "content_pack"`, the news lane) have one number: `rows` is the count of
their `pack_points`, there is no exclusion step, and they do not use these helpers.
"""
from __future__ import annotations

from typing import Mapping


def _int(v) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def rows_staged(manifest: Mapping) -> int | None:
    """Rows in the sealed stage. A legacy pack (no `rows_staged`) answers with `rows`, which for
    it is the stage count unless the served check dropped ids (then `rows_before_exclusion`)."""
    v = _int(manifest.get("rows_staged"))
    if v is not None:
        return v
    v = _int(manifest.get("rows_before_exclusion"))
    return v if v is not None else _int(manifest.get("rows"))


def rows_to_write_max(manifest: Mapping) -> int | None:
    """Upper bound on the rows the drain writes (never an equality target). A legacy pack answers with `rows`, exactly as before."""
    v = _int(manifest.get("rows_to_write_max"))
    return v if v is not None else _int(manifest.get("rows"))
