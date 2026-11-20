"""Stage-time guard: a staged row never holds less text than its source by accident (task 706).

THE DEFECT THIS CLOSES. Task 633 (commit `f2199fac`) staged every DILA LEGI/LODA article through
`f"{title}\\n\\n{markdown}"[:8000]`. 13,948 of 222,499 rows lost their tail (34 % of the family's
text), for the dense AND the sparse leg, and every stage count, every id check and every gate
passed: a cap is not a change that any count sees. Row counts, ids and digests were all computed
FROM the cut text, so nothing downstream had the source to compare with.

THE GUARD. An adapter that hands `Doc.source_text` (the whole extracted text before `chunk` ran)
is measured: the share of the source's word 5-grams that the staged rows of that doc still hold.
Below `MIN_COVERAGE` the unit is QUARANTINED (`staged_text_shortfall`, bytes sealed, counted in
the pack manifest), never admitted. A split costs the 5-grams on its boundary
(`BOUNDARY_ALLOWANCE` per extra row, forgiven); the floor 0.98 covers the rest.

An adapter that does NOT hand a source text is not checked, and that is said, not hidden: the
manifest counts its docs as `unchecked`. Families whose extractor cuts text BEFORE the framework
sees it (`dila_juri`) cannot be caught by a comparison, because the
source is gone by then. (`dila_kali` and the entity cards lost theirs on 2026-10-07, ADR-090 am. and
ADR-060 am.: their staged text is never cut.) The caps that remain are DECLARED below, with the file they live in and the founder
question they raise, and the manifest reports how many staged rows sit at each cap. A declared cap
is a counted fact for the founder; it is never silent and never "fixed" here.

Pure functions, no I/O: `observe` per doc, `summary` once per run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

#: DEFAULT threshold: a staged doc must keep at least this share of its source's word 5-grams.
#: A family sets its own by the adapter attribute `staged_text_min_coverage` (the ONE place a
#: family's threshold lives; `run_family` reads it, the manifest records it). It is not yet a
#: contract field (legislative follow-up). A family whose split loses nothing declares 1.0.
MIN_COVERAGE = 0.98
SHINGLE = 5
SHORTFALL_REASON = "staged_text_shortfall"
#: how many offending units the manifest names (the rest are counted)
EXAMPLES = 10


def _shingles(text: str) -> set[int]:
    words = text.split()
    if len(words) < SHINGLE:
        return {hash(" ".join(words))} if words else set()
    return {hash(" ".join(words[i:i + SHINGLE])) for i in range(len(words) - SHINGLE + 1)}


#: 5-grams a split can legitimately cost: the 4 that straddle the cut, and the 4 around a heading a
#: split adds ("### Paragraph 7 (part 2/3) {#...}"). Read on a real 5-row article (22 lost 5-grams):
#: every one sat on a boundary. Without it the lowest of 13,957 split articles read 0.9836.
BOUNDARY_ALLOWANCE = 2 * (SHINGLE - 1)


def coverage(source_text: str, staged_texts: Iterable[str]) -> float:
    """Share (0..1) of the source's word 5-grams that the staged texts hold, counting up to
    `BOUNDARY_ALLOWANCE` lost 5-grams per extra staged row as held (a split costs those and
    nothing else). A source with no words is fully covered. Whitespace and repeated headers do not
    matter; a removed tail does."""
    staged = list(staged_texts)
    src = _shingles(source_text)
    if not src:
        return 1.0
    held: set[int] = set()
    for text in staged:
        held |= _shingles(text)
    lost = len(src) - len(src & held)
    forgiven = min(lost, BOUNDARY_ALLOWANCE * max(len(staged) - 1, 0))
    return (len(src) - lost + forgiven) / len(src)


# --- per-part comparison (task 1045): one doc, many articles --------------------------------

@dataclass(frozen=True)
class SourcePart:
    """One article of a doc: its `key`, its source `text` and the word offsets (`cuts`) where the
    adapter's own split cuts it. A doc made of many parts (a law: one part per article) is
    compared PART BY PART, so a lost article or a lost tail window cannot hide in the slack of
    its neighbours (task 1044 C1)."""

    key: str
    text: str
    cuts: tuple[int, ...] = ()


def _straddlers(part: SourcePart) -> set[int]:
    """The word 5-grams that straddle one of the part's cut points: a split loses exactly these
    (they sit in no row) and nothing else. Fixed by the SOURCE and the cut offsets, so a damaged
    row cannot forge them."""
    words = part.text.split()
    out: set[int] = set()
    for c in part.cuts:
        for i in range(max(0, c - SHINGLE + 1), min(c, len(words) - SHINGLE + 1)):
            out.add(hash(" ".join(words[i:i + SHINGLE])))
    return out


def _contains_words(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def part_coverage(part: SourcePart, staged_texts: Iterable[str]) -> float:
    """Share (0..1) of the part's word 5-grams held by the rows of THAT part, the 5-grams on its
    declared cut points counted as held. A part with no staged row reads 0.0; a part with no words
    reads 1.0."""
    words = part.text.split()
    if 0 < len(words) < SHINGLE:
        # Task 2198: a part shorter than one shingle can never equal a 5-gram of a longer row, so
        # it is held when its word sequence occurs contiguously (on word boundaries) in a row.
        return 1.0 if any(_contains_words(t.split(), words) for t in staged_texts) else 0.0
    src = _shingles(part.text)
    if not src:
        return 1.0
    held: set[int] = set()
    for text in staged_texts:
        held |= _shingles(text)
    lost = src - held - _straddlers(part)
    return 1.0 - len(lost) / len(src)


# --- declared caps: text cut BEFORE the framework sees it ----------------------------------

@dataclass(frozen=True)
class DeclaredCap:
    """A cap that exists in code today and that this guard does not change. `at_cap` says whether
    one staged row's text sits at it (a cut row ends exactly there)."""

    name: str
    family_ids: tuple[str, ...]
    chars: int
    where: str
    question: str
    at_cap: Callable[[str], bool] = field(compare=False, repr=False)


_JURI_TAGS = ("Sommaire : ", "ECLI : ", "Solution : ")


def _juri_at_cap(text: str) -> bool:
    """A cut decision ends in a body excerpt of exactly 2,000 chars (`text_body[:2000]`, which is
    the last part), or in the final `[:3600]` cut. The body starts after the last header line
    (`Sommaire : `, `ECLI : `, `Solution : `); a headnote that itself spans lines hides that start,
    so this is a LOWER bound on the rows cut."""
    if len(text) == 3600:
        return True
    lines = text.split("\n")
    last_tag = max((i for i, ln in enumerate(lines) if ln.startswith(_JURI_TAGS)), default=0)
    body = "\n".join(lines[last_tag + 1:])
    return len(body) == 2000


DECLARED_CAPS: tuple[DeclaredCap, ...] = (
    DeclaredCap(
        "dila_juri.body_excerpt", ("dila_juri",), 2000,
        "cytir/extract/dila_juri.py:30,115,118 (MAX_BODY_CHARS 2000, sommaire[:1200], final [:3600])",
        "Q-b: may dila_juri chunk the full decision instead of a 2,000-char locator excerpt?",
        _juri_at_cap),
)


def declared_caps_for(family_id: str) -> tuple[DeclaredCap, ...]:
    return tuple(c for c in DECLARED_CAPS if family_id in c.family_ids)


def count_declared_caps(family_id: str, texts: Sequence[str]) -> list[dict]:
    """For each cap declared for `family_id`: how many of `texts` (the staged rows' text) sit at
    it. A row at the cap may or may not have lost text (a source of exactly the cap length is not
    cut); a row NOT at it was not cut by it."""
    out = []
    for cap in declared_caps_for(family_id):
        out.append({"cap": cap.name, "chars": cap.chars, "rows_at_cap": sum(
            1 for t in texts if cap.at_cap(t)), "rows": len(texts), "where": cap.where,
            "founder_question": cap.question, "status": "declared, not changed (task 706)"})
    return out


# --- the per-run guard ----------------------------------------------------------------------

class StagedTextGuard:
    """Accumulates the per-doc verdicts of one `run_family` run."""

    def __init__(self, family_id: str, *, min_coverage: float | None = None):
        """`min_coverage=None` = the family declares no threshold: the default applies and the
        manifest says so (`threshold_declared: false`)."""
        self.family_id = family_id
        self.threshold_declared = min_coverage is not None
        self.min_coverage = MIN_COVERAGE if min_coverage is None else min_coverage
        self.docs_checked = 0
        self.docs_unchecked = 0
        self.docs_shortfall = 0
        self.source_chars = 0
        self.staged_chars = 0
        self.lowest: float | None = None     # None until a doc was checked
        self.examples: list[dict] = []

    def observe(self, unit_key: str, source_text: str | None, staged_texts: Sequence[str]
                ) -> float | None:
        """Record one doc; returns its coverage, or None when it was not checked."""
        if source_text is None:
            self.docs_unchecked += 1
            return None
        cov = coverage(source_text, staged_texts)
        self.docs_checked += 1
        self.source_chars += len(source_text)
        self.staged_chars += sum(len(t) for t in staged_texts)
        self.lowest = cov if self.lowest is None else min(self.lowest, cov)
        if cov < self.min_coverage:
            self.docs_shortfall += 1
            if len(self.examples) < EXAMPLES:
                self.examples.append({"unit": unit_key, "coverage": round(cov, 4),
                                      "source_chars": len(source_text),
                                      "staged_chars": sum(len(t) for t in staged_texts)})
        return cov

    def observe_parts(self, unit_key: str, parts: Sequence[SourcePart],
                      records: Sequence[dict], part_of: Callable[[dict], str]) -> float:
        """Record one doc that is many parts (one article each). The doc counts once in
        `docs_checked` (the url-group drain compares that count with the pack's url groups) and
        reads its LOWEST part. A doc with no part reads 0.0: an empty source is a wrong input or a
        broken parser, never a fully covered one. A staged row whose key matches no part is not
        counted for any part."""
        by_key: dict[str, list[str]] = {}
        for r in records:
            by_key.setdefault(part_of(r), []).append(str(r.get("text") or ""))
        worst_key, cov = None, 1.0
        if not parts:
            worst_key, cov = "<no part>", 0.0
        for part in parts:
            c = part_coverage(part, by_key.get(part.key, ()))
            if c < cov:
                worst_key, cov = part.key, c
        src_chars = sum(len(p.text) for p in parts)
        staged_chars = sum(len(t) for ts in by_key.values() for t in ts)
        self.docs_checked += 1
        self.source_chars += src_chars
        self.staged_chars += staged_chars
        self.lowest = cov if self.lowest is None else min(self.lowest, cov)
        if cov < self.min_coverage:
            self.docs_shortfall += 1
            if len(self.examples) < EXAMPLES:
                self.examples.append({"unit": unit_key, "part": worst_key, "coverage": round(cov, 4),
                                      "source_chars": src_chars, "staged_chars": staged_chars})
        return cov

    def shortfall(self, cov: float | None) -> bool:
        return cov is not None and cov < self.min_coverage

    def summary(self, staged_texts: Sequence[str] = ()) -> dict:
        """The manifest block. `staged_texts` = every admitted row's text (for declared caps)."""
        return {
            "rule": (f"word 5-gram coverage of the doc's source text (per part, where the adapter gives "
                     f"parts) by its staged rows >= "
                     f"{self.min_coverage}; below it the unit is quarantined "
                     f"({SHORTFALL_REASON})"),
            "min_coverage": self.min_coverage,
            "threshold_declared": self.threshold_declared,
            "docs_checked": self.docs_checked,
            "docs_unchecked": self.docs_unchecked,
            "docs_shortfall": self.docs_shortfall,
            "lowest_coverage": None if self.lowest is None else round(self.lowest, 4),
            "source_chars_checked": self.source_chars,
            "staged_chars_checked": self.staged_chars,
            "shortfall_examples": self.examples,
            "declared_caps": count_declared_caps(self.family_id, staged_texts),
        }

    def loud_line(self) -> str | None:
        if self.docs_shortfall or self.docs_unchecked:
            short = (f"{self.docs_shortfall} SHORT of their source (quarantined)"
                     if self.docs_shortfall else "0 short of their source")
            return (f"[{self.family_id}] staged-text guard: {self.docs_checked} checked, "
                    f"{short}, "
                    f"{self.docs_unchecked} UNCHECKED (the adapter gives no source text)")
        return None


def door_refusal(guard_block: dict | None) -> str | None:
    """The stage door's reading of a pack's `staged_text_guard` block (`family_stage.check_pack`).
    A pack from a family that DECLARES a threshold and quarantined a doc for falling under it is
    refused: the quarantine is a verdict, and a pack that carries one is not a finished input. A
    pack with no block (staged before the guard existed) or from a family on the default
    threshold passes as before. Returns the refusal text, or None."""
    if not isinstance(guard_block, dict):
        return None
    short = guard_block.get("docs_shortfall") or 0
    if short > 0 and guard_block.get("threshold_declared"):
        return (f"staged-text guard: {short} doc(s) fell below the family's declared coverage "
                f"{guard_block.get('min_coverage')} (lowest {guard_block.get('lowest_coverage')}, "
                f"examples {guard_block.get('shortfall_examples')!r}); a pack that lost text to a "
                "cap is not staged. Fix the chunker and re-run")
    return None
