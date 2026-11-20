"""Digit-first query-side fused/split variants (task 2856, RULING 79, flag `AW_QUERY_FUSED_SPLIT`, OFF).

Principle: the tokenizer drops a lone letter (len < 2) on BOTH sides. A DIGIT-first word loses a part there:
the index spells `Art. 336 c` (token `336` only) while the query says `336c` (one token, no posting). A
LETTER-first word (`rfc8446`, `cas9`, `ipv4`) is ONE token on both sides, so splitting or fusing it repairs
nothing (measured: letter-first fuse `rfc 8446` -> `rfc8446` lost 4 golds, the split of `Cas9` moved gx11 6 -> 8).
For EVERY query, with no keyword trigger, this appends (originals always stay):

* SPLIT  a word of one digit run then one letter run (`336c`, `4bis`) also yields its parts; a part shorter
         than 2 chars is not added (`336c` -> `336`).
* FUSE   two ADJACENT words (whitespace only between them): digits (<= 6) THEN one letter or `bis|ter|quater`
         (`336 c` -> `336c`, `4 bis` -> `4bis`). Digits then a longer word is NOT fused (`15 de`, `815 code`).

Non-explosion: a letter run > 8 is skipped; a word glued by `.,-:/` to a digit on the other side (`3.14`,
`2.5g`, `v2.1`, `2024-11-16`) is skipped; a variant already present as a token is not added; at most
`MAX_VARIANTS` per query. Same weight as the originals (deduped). Pure, no I/O, language-neutral.
"""
from __future__ import annotations

import re

MAX_VARIANTS = 8
_MAX_LETTERS_SPLIT = 8
_MAX_DIGITS_FUSE = 6
_SUFFIX_WORDS = frozenset({"bis", "ter", "quater"})  # Latin article suffixes: `4 bis`

_WORD = re.compile(r"\w+", re.UNICODE)
_DIGIT_LETTER = re.compile(r"(\d+)([^\W\d_]+)\Z")
_DIGITS = re.compile(r"\d+\Z")
_LETTERS = re.compile(r"[^\W\d_]+\Z")
_JOINERS = ".,-:/"


def _embedded(q: str, start: int, end: int) -> bool:
    """True when the word is glued by `.,-:/` to a digit (decimal, date, version, range)."""
    before = q[start - 2:start] if start >= 2 else ""
    after = q[end:end + 2]
    left = len(before) == 2 and before[1] in _JOINERS and before[0].isdigit()
    right = len(after) == 2 and after[0] in _JOINERS and after[1].isdigit()
    return left or right


def fused_split_variants(q: str) -> str:
    """`q` plus the fused/split variants of its letter-digit words; `q` unchanged if none."""
    low = (q or "").lower()
    words = [(m.group(), m.start(), m.end()) for m in _WORD.finditer(low)]
    have = {w for w, _s, _e in words}
    extra: list[str] = []

    def add(tok: str) -> None:
        if len(tok) >= 2 and tok not in have and len(extra) < MAX_VARIANTS:
            have.add(tok)
            extra.append(tok)

    for w, s, e in words:
        if _embedded(low, s, e):
            continue
        m = _DIGIT_LETTER.match(w)  # digit-first only: a letter-first word is ONE token on both sides
        if m and len(m.group(2)) <= _MAX_LETTERS_SPLIT:
            add(m.group(2))
            add(m.group(1))
    for (a, s1, e1), (b, s2, e2) in zip(words, words[1:]):
        gap = low[e1:s2]
        if not gap or gap.strip() or _embedded(low, s1, e1) or _embedded(low, s2, e2):
            continue
        if _DIGITS.match(a) and _LETTERS.match(b) and len(a) <= _MAX_DIGITS_FUSE and (len(b) == 1 or b in _SUFFIX_WORDS):
            add(a + b)
    return f"{q} {' '.join(extra)}" if extra else q
