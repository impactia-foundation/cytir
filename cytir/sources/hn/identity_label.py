"""Q5 (identity minimization) label scan for HN discussion text.

The HN-discussion profile forbids the PIPELINE from labeling a username into
the embedded surface (``Author: <by>``).  The renderer adds no label, so a
line that looks like one in a rendered discussion is either a commenter's own
prose (a signature or citation such as ``Author: Prof. X``) or a renderer
regression.  ``classify_discussion`` tells the two apart so the ingest can
hold the first kind OUT of the pack and refuse the second.
"""
from __future__ import annotations

import re
from typing import Iterable

LABEL_PATTERN = re.compile(r"(?im)^[^\S\r\n]*(author|by|submitted by|user)\s*:\s*\S")

CLEAN = "clean"
PROSE = "prose"          # the label sits inside a source text block: hold the row out
RENDERER = "renderer"    # only the composed document shows it: the renderer added it


def find_label(text: str) -> str | None:
    """The matched label text, or None."""
    match = LABEL_PATTERN.search(text or "")
    return match.group(0) if match else None


def classify_discussion(rendered: str, source_blocks: Iterable[str]) -> str:
    """``clean`` / ``prose`` / ``renderer`` for one rendered discussion."""
    if find_label(rendered) is None:
        return CLEAN
    if any(find_label(block) is not None for block in source_blocks):
        return PROSE
    return RENDERER


def _scan_matches(text: str) -> list[str]:
    """Every label match, re-scanning from each match's end minus one char.

    The pattern's trailing ``\\s*\\S`` can swallow the first character of a
    genuine label that follows a stitched match; resuming one char early
    lets ``^`` see that label (``^`` checks the real preceding newline).
    """
    out: list[str] = []
    pos = 0
    while (m := LABEL_PATTERN.search(text, pos)) is not None:
        out.append(m.group(0))
        pos = max(m.end() - 1, m.start() + 1)
    return out


REASON_PROSE = "q5_identity_label_in_prose"
REASON_SPANS_BLOCKS = "q5_identity_label_spans_blocks"


def classify_with_reason(rendered: str, source_blocks: Iterable[str]) -> tuple[str, str | None]:
    """``(verdict, hold_reason)``; ``hold_reason`` is set only for ``prose``.

    A match that contains a blank line (``\\n\\n``) was stitched together by
    ``\\s*`` across the join between two source blocks (``By`` / ``: x``,
    or ``Author:`` / ``Bob``).  No renderer adds that; it is commenter prose,
    so the row is held out (reason ``q5_identity_label_spans_blocks``), not
    the pack failed.  The regex itself is unchanged.
    """
    blocks = list(source_blocks)
    verdict = classify_discussion(rendered, blocks)
    if verdict == PROSE:
        return PROSE, REASON_PROSE
    if verdict == RENDERER:
        matches = _scan_matches(rendered)
        if matches and all("\n\n" in m.strip() for m in matches):
            return PROSE, REASON_SPANS_BLOCKS
    return verdict, None
