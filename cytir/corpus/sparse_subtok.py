#!/usr/bin/env python3
"""Task 282 — alphanumeric name sub-token expansion for the register-card SPARSE
channel (ADR-060 recipe).

Root cause (confirmed on the live twin aw_v4_sat 2026-07-19): the BM25 tokenizer
(`aw_server.schema.bm25_tokens` / `cytir.core.common`) is `_TOKEN_RE = \\w+`, so a
name like `Vector8` is ONE token. The card `Vector8 AG` (vector8.com) therefore
carries the sparse token `vector8` but NEITHER `vector` NOR `8` as standalone hooks.
A query `vector8` matches (#1); a SPACED query `vector 8` tokenizes to
`{vector, 8}` and matches nothing on the name leg, so the card never enters the
pool and the Wikipedia "Vector" pages flood it.

FIX (SPARSE ONLY — the ADR-060 Zweck-prose DENSE channel is untouched): for an
alphanumeric-adjacent name, index BOTH the joined form AND the letter/digit
sub-tokens, e.g. `Vector8` -> {vector8, vector, 8}. Generalises the `3M` /
`Section 8` / `web3` class: split at every letter<->digit boundary, keep the
joined form.

IMPORTANT tokenizer interaction: `bm25_tokens` drops tokens of len < 2, so a bare
single digit/letter sub-token (`8`, the `M` of `3M`) is dropped at BOTH index and
query time — the load-bearing sub-token is the multi-char run (`vector`). We still
EMIT the short pieces for completeness/generality; the tokenizer filters them.

Scope discipline: expansion is applied ONLY to the entity NAME material (title +
domain label + non-UID aliases). The register id hooks (CHE-341.859.062 /
CHE341859062) are deliberately NOT split — they are id-fastpath identity tokens,
splitting them adds only a useless digit-run and would falsely mark all 72k
register cards as "affected".
"""
from __future__ import annotations
import re

# a maximal alphanumeric run that mixes letters and digits (the only runs a split
# can change). Pure-alpha or pure-digit runs are returned unchanged by split_runs.
_ALNUM_RUN = re.compile(r"[^\W_]+", re.UNICODE)          # \w minus underscore
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_HAS_DIGIT = re.compile(r"\d", re.UNICODE)
# letter-run | digit-run — used to slice a mixed run into its pieces
_PIECE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)

# CHE register-id shapes to exclude from expansion (identity hooks, not names):
#   CHE341859062  (norm)   /  the 9-digit body  /  MWST suffix marker
_UID_RUN = re.compile(r"^(?:che\d{6,}|\d{6,})$", re.I)


def split_alnum_run(run: str) -> list[str]:
    """Given ONE \\w run, if it mixes letters and digits return its letter/digit
    pieces (joined form NOT included — caller keeps the original). Otherwise []."""
    if not (_HAS_LETTER.search(run) and _HAS_DIGIT.search(run)):
        return []
    if _UID_RUN.match(run):
        return []
    return _PIECE.findall(run)


def expand_name_subtokens(name_text: str) -> list[str]:
    """Return the ADDITIVE sub-tokens for all alphanumeric-adjacent runs in
    `name_text` (a name / title / domain-label string). Deterministic order,
    de-duplicated (case-insensitively), joined forms excluded (already present in
    the source sparse). Empty when the name has no letter<->digit adjacency."""
    out: list[str] = []
    seen: set[str] = set()
    for run in _ALNUM_RUN.findall(name_text or ""):
        for piece in split_alnum_run(run):
            k = piece.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(piece)
    return out
