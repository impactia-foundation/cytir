#!/usr/bin/env python3
"""chunk.py — THE AW chunker (rule-epoch `chunk@2`, ADR-2.6 / task 340).

What this is
------------
The structure-aware markdown chunker mandated by `specs/chunking-experiments.md`
(tasks 237 + 237b) and listed as gap **G1** in `specs/ingestion-pipeline.md`
("the mandated 768w structure-aware chunker does not exist"). It is the
canonical chunker for the 100M rebuild and for every new ingest path; the legacy
180-word paragraph windower (`fetch_process.chunk`, rule-epoch `chunk@1`) stays
only until the epoch migration replays the corpus (`specs/epochs.md` §3: a
`chunk` bump is a full rebuild, never a delta).

The rule, and why each constant is what it is
---------------------------------------------
* **~768 words per chunk** (`TARGET_WORDS`). 237b measured a size sweep against
  the embedder's REAL ceiling (`arctic-embed-l-v2` is XLM-R with
  `max_position_embeddings: 8194`, not the ~512 tokens task 237 wrongly assumed):
  768w matched/beat 380w on BOTH page-level (MRR 0.782 vs 0.777) and deep-fact
  (0.96 vs 0.94) intents while emitting ~35% fewer passages (2219 vs 3403).
  Larger is NOT better — whole-page-as-one-chunk tanks deep-fact to 0.817 by
  topical averaging, and 8192-token inputs are ~26x slower per doc.
* **overlap 0** (`OVERLAP_WORDS`). 237 §4.1: overlap 30 -> 60 cost +20..40%
  passages and bought no retrieval (ov60 had the WORST page-level MRR).
* **cut on headers, not blind word windows** — 237 §4.4: header cuts kept a
  `class transformers.Trainer(...)` signature whole where the word windower
  split it mid-signature; every fragment keeps its section topic via the
  breadcrumb prefix.
* **merge sections < 40 words** (`MERGE_MIN_WORDS`) — a bare "## Parameters"
  heading with two lines under it is not a retrievable unit.
* **hard-split any block over target** (`HARD_MAX_WORDS`). 237 §4.1 found 153
  baseline chunks ALREADY over the assumed ceiling because a long code/table
  block arrives as one "paragraph": *"the extractor must hard-split any block
  over target, not trust paragraph boundaries."* The 237b s768 variant did NOT
  do this (its own `_counts.json` records `max_words: 2860`) — this module does.
* **token-aware second hard-split** (`MAX_CHUNK_TOKENS`, task 340 F1, fixed
  task 360). 340 §2.3 measured word->token ratio 2.2-7.3 on the real arctic
  tokenizer (code/tables at the high end) — a WORD ceiling cannot bound
  tokens, and 4.4% of chunk@2 passages exceeded TEI's configured
  `max_seq=2048` and would have been silently truncated at embed. Every
  candidate is re-measured with the real tokenizer and split further at a
  token-exact boundary (binary search, not a ratio estimate) if it would
  still overflow.
* **`min_words` is overridable per call** (task 340 F6, fixed task 360). The
  15-word floor (`MIN_CHUNK_WORDS`) exists to drop unretrievable stub
  fragments from windowed PROSE — applied blind to a route whose records are
  already atomic short cards (LCSH subject headings, ch_courts headnotes,
  ai_autodoc_reference entries; median 10-14 words), it silently drops the
  WHOLE record. Callers processing a known card-shaped route pass a lower
  floor explicitly; the module default is unchanged for the generic path.

Contracts this module respects
------------------------------
* **Deterministic / byte-stable**: `chunk_markdown` is a pure function of the
  markdown string and the module config. Same input + same config => byte-
  identical chunk texts, in the same order. Verified by `test_chunk.py`.
* **Identity (`specs/identity.md`, task 325/326 GAP A33)**: chunk ids are minted
  ONLY through `aw_b1_identity` — never a local `uuid5`. `assign_chunk_ids`
  guarantees that distinct chunks of one anchor-less URL mint DISTINCT point
  ids (the anthropic.com/constitution collapse, 414 -> 15 points, was exactly
  this failure) and reports any residual collision loudly instead of letting a
  later record silently overwrite an earlier one.
* **One truth, one home** (engineering doctrine 1/6): word counting, section
  parsing and the breadcrumb rule exist here once. `aw_common._content_passages`
  dispatches to this module; nothing copies it.
* **Config is data** (doctrine 2): every boundary rule comes from one validated,
  versioned JSON policy whose byte hash is recorded in manifests. Historical
  `AW_CHUNK_*` experiment overrides remain visible in `config()`; candidate
  materialization selects a policy explicitly and disables inherited overrides.

Usage
-----
    from cytir.chunk import chunker as awchunk
    for c in awchunk.chunk_markdown(md):      # -> list[Chunk]
        ...
    texts = awchunk.chunk_texts(md)           # -> list[str], drop-in for fp.chunk
    recs  = awchunk.assign_chunk_ids(url, awchunk.chunk_markdown(md))

In the extraction funnel candidate runs select `AW_CHUNKER=structured` plus
an explicit `AW_CHUNK_POLICY` path and `AW_CHUNK_POLICY_SHA256`
(`aw_common.chunker_selection`); ordinary recurring runs read the tracked
`data/build_config.json` profile, which stays `legacy` until the atomic
chunk-epoch alias flip.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, NamedTuple, Optional

from cytir.core import ids as ident
from cytir.chunk.policy import (
    ADJACENT_SECTION_PACK,
    LAW_ARTICLE_SUB_PACK,
    STRUCT_WORD_PACK,
    ChunkPolicy,
    load_chunk_policy,
)

# --- I2 rule-epoch stamp (specs/epochs.md §4 rule 3: declared in code, in the
# same commit as the behaviour) --------------------------------------------
# chunk@1 = fp.chunk paragraph windows (180w/ov30), LIVE.
# chunk@2 = THIS module. A chunk bump is a FULL REBUILD, never a delta
# (epochs.md §3): new boundaries => new text => new digest => new Tier-B ids.
def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        # Fail loud (doctrine 3): a typo'd override must not silently chunk the
        # corpus at the default size and produce an un-diagnosable index.
        raise ValueError(f"{name}={v!r} is not an integer") from None


_tokenizer_cache: dict[tuple[str, str | None], object] = {}
_TOKENIZER_STATE = {"status": None}   # None=untried, "ok", "unavailable"
# A materializer must name the immutable tokenizer revision used by its worker.
# The policy owns model + boundary values; the materialization config owns the
# worker snapshot because an image/model refresh need not change chunk rules.
# Leaving this optional keeps the historical pure chunker usable for local
# word-boundary experiments, but a real materializer configures it explicitly.
TOKENIZER_REVISION = os.environ.get("AW_CHUNK_TOKENIZER_REVISION") or None


def select_tokenizer_revision(revision: str | None) -> None:
    """Bind token measurement to one immutable HF snapshot.

    ``AutoTokenizer.from_pretrained(model)`` follows the mutable default branch.
    That is not a valid preflight for an embedding worker whose tokenizer is
    revision-locked.  Materializers call this from their tracked config before
    emitting identities, so their token ceiling is a mirror contract with the
    worker rather than an estimate from whatever ``main`` names today.
    """
    global TOKENIZER_REVISION
    if revision is not None and (
            not isinstance(revision, str) or
            re.fullmatch(r"[0-9a-f]{40}", revision) is None):
        raise ValueError("tokenizer revision must be a lowercase 40-hex commit or None")
    if revision != TOKENIZER_REVISION:
        TOKENIZER_REVISION = revision
        _tokenizer_cache.clear()
        _TOKENIZER_STATE["status"] = None


def _apply_policy(selected: ChunkPolicy, *, allow_env_overrides: bool) -> None:
    """Install one validated policy as the module's effective configuration.

    Candidate/materialization runs select a policy explicitly with overrides
    disabled, so the artifact hash alone identifies every boundary rule.
    Existing experiment callers retain the historical ``AW_CHUNK_*`` knobs
    on the default import path; ``config()`` records both the selected policy
    and the resulting effective values so an override is never invisible.
    """
    global _ACTIVE_POLICY, CHUNK_EPOCH, CHUNK_RULE_LEGEND
    global TARGET_WORDS, MERGE_MIN_WORDS, MIN_CHUNK_WORDS, OVERLAP_WORDS
    global MAX_PREFIX_WORDS, HARD_MAX_WORDS, TOKENIZER_MODEL
    global MAX_CHUNK_TOKENS, BREADCRUMB_SEP

    def configured(name: str, value: int) -> int:
        return _env_int(name, value) if allow_env_overrides else value

    old_tokenizer = globals().get("TOKENIZER_MODEL")
    _ACTIVE_POLICY = selected
    CHUNK_EPOCH = selected.epoch
    CHUNK_RULE_LEGEND = selected.rule_legend
    TARGET_WORDS = configured("AW_CHUNK_TARGET_WORDS", selected.soft_target)
    MERGE_MIN_WORDS = configured(
        "AW_CHUNK_MERGE_MIN_WORDS", selected.merge_below_words)
    MIN_CHUNK_WORDS = configured("AW_CHUNK_MIN_WORDS", selected.floor_words)
    OVERLAP_WORDS = configured("AW_CHUNK_OVERLAP_WORDS", selected.overlap_words)
    # Preserve the prior override contract: changing the target without an
    # explicit prefix override derives target/8, while the canonical artifact
    # itself declares the exact 96-word default.
    prefix_default = (TARGET_WORDS // 8 if allow_env_overrides and
                      TARGET_WORDS != selected.soft_target
                      else selected.prefix_max_words)
    MAX_PREFIX_WORDS = configured("AW_CHUNK_MAX_PREFIX_WORDS", prefix_default)
    # A tail maximum is policy data.  The original policy derives it as
    # target+floor (783); adjacent-section candidate policies may pin a wider
    # bounded tail while retaining the measured 768-word target.
    HARD_MAX_WORDS = selected.word_hard_max
    TOKENIZER_MODEL = (
        os.environ.get("AW_CHUNK_TOKENIZER_MODEL", selected.tokenizer_model)
        if allow_env_overrides else selected.tokenizer_model)
    MAX_CHUNK_TOKENS = configured(
        "AW_CHUNK_MAX_TOKENS", selected.token_hard_max)
    BREADCRUMB_SEP = selected.prefix_separator
    if old_tokenizer is not None and old_tokenizer != TOKENIZER_MODEL:
        _tokenizer_cache.clear()
        _TOKENIZER_STATE["status"] = None


def select_policy(path: str | Path, *, allow_env_overrides: bool = False) -> ChunkPolicy:
    """Explicitly select, validate, and activate a versioned policy artifact."""
    selected = load_chunk_policy(path)
    _apply_policy(selected, allow_env_overrides=allow_env_overrides)
    return selected


def active_policy() -> ChunkPolicy:
    return _ACTIVE_POLICY


def effective_policy_value() -> dict:
    """Reconstruct the selected policy with any experiment overrides applied.

    Materializers compare this generic object with the loaded artifact instead
    of knowing word-policy field names. A future token-native algorithm adds
    its implementation and effective-value branch here; callers remain policy
    consumers and do not need a new materializer/dispatcher contract.
    """
    value = {
        key: (dict(item) if isinstance(item, dict) else item)
        for key, item in _ACTIVE_POLICY.value.items()
    }
    if _ACTIVE_POLICY.algorithm in {STRUCT_WORD_PACK, ADJACENT_SECTION_PACK,
                                    LAW_ARTICLE_SUB_PACK}:
        value["sizing"]["soft_target"] = TARGET_WORDS
        value["word_guard"]["hard_max"] = HARD_MAX_WORDS
        value["token_hard_max"]["exact"] = MAX_CHUNK_TOKENS
        value["token_hard_max"]["tokenizer_model"] = TOKENIZER_MODEL
        value["overlap"]["size"] = OVERLAP_WORDS
        value["merge"]["below"] = MERGE_MIN_WORDS
        value["floor"]["minimum"] = MIN_CHUNK_WORDS
        value["prefix"]["maximum"] = MAX_PREFIX_WORDS
        value["prefix"]["separator"] = BREADCRUMB_SEP
        return value
    raise ValueError(
        f"active chunk algorithm {_ACTIVE_POLICY.algorithm!r} has no implementation")


def policy_matches_effective(policy: Optional[ChunkPolicy] = None) -> bool:
    selected = policy or _ACTIVE_POLICY
    return selected.sha256 == _ACTIVE_POLICY.sha256 and selected.value == effective_policy_value()


# Default behaviour remains the historical chunk@2 policy. A future campaign
# can name another tracked artifact via AW_CHUNK_POLICY or ``select_policy``;
# materialization uses the explicit latter path so inherited env cannot choose
# a build identity accidentally.
_apply_policy(
    load_chunk_policy(os.environ.get("AW_CHUNK_POLICY") or None),
    allow_env_overrides=True,
)

# --- F1 (task 340 §5, fixed task 360): token-aware hard-split -
# Measured on the real arctic tokenizer (340 §2.3): word->token ratio ranges
# 2.2-7.3 (code/tables tokenize ~3x worse than prose), so a WORD ceiling
# (HARD_MAX_WORDS above) cannot bound tokens. The selected policy records the
# exact tokenizer and token ceiling used by this shared implementation.


def effective_tokenizer_revision() -> str | None:
    """The revision the tokenizer is loaded at.

    An explicit ``TOKENIZER_REVISION`` (env or ``select_tokenizer_revision``) wins.
    Otherwise, for the pinned embedder, the committed pin from ``cytir.core.model_pin``
    -- never the hub's mutable "main". ``fetch_models --tokenizer`` writes
    ``snapshots/<pin>/`` only (no ``refs/main``), so a loader asking for "main"
    cannot resolve from a fresh offline cache (task 2520).
    """
    if TOKENIZER_REVISION is not None:
        return TOKENIZER_REVISION
    from cytir.core import model_pin
    if TOKENIZER_MODEL == model_pin.EMBED_MODEL_ID:
        return model_pin.embed_model_revision()
    return None


def _get_tokenizer():
    """Lazy singleton -- only imported/loaded the first time a chunk's token
    length actually needs checking. `transformers` is already a pipeline
    dependency (build requirements file) everywhere chunk@2 is meant to
    run for real (the build box replay). Returns None, ONCE loudly announcing it on
    stderr, if unavailable -- this lets the pure word-boundary algorithm stay
    testable with zero heavy deps (every non-token test in test_chunk.py) while
    making the F1 degradation impossible to miss in a real run: `config()`
    exposes `tokenizer_available` so a wave manifest can assert this fired
    'ok', not silently shipped 'unavailable' (doctrine 3: fail loud OR degrade
    VISIBLY -- never silently)."""
    revision = effective_tokenizer_revision()
    cache_key = (TOKENIZER_MODEL, revision)
    tok = _tokenizer_cache.get(cache_key)
    if tok is not None:
        return tok
    if _TOKENIZER_STATE["status"] == "unavailable":
        return None
    try:
        # ONE HOME (doctrine 1/6). This used to call
        # `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` directly,
        # WITHOUT pinning `use_fast` — so the backend was decided by whatever
        # happened to be installed in the process. In the task389 campaign's
        # materializer process sentencepiece was present and it resolved to the
        # SLOW `XLMRobertaTokenizer`, which strips trailing whitespace and
        # undercounts by one; the embed worker counted with the FAST backend.
        # 702 rows shipped one token over a 2000-token ceiling. The counter now
        # lives in `dense_token_counter`, imported by the materializer and the
        # embed worker too, with `use_fast=True` pinned and `is_fast` asserted.
        # See that module's docstring for the full mechanism.
        from cytir.chunk import token_counter as dense_token_counter
        tok = dense_token_counter.resolve_tokenizer(
            TOKENIZER_MODEL, revision)
    except Exception as e:  # noqa: BLE001 — import OR download/cache failure
        _TOKENIZER_STATE["status"] = "unavailable"
        print(f"[chunk.py] token-aware hard-split (F1) DISABLED: could not "
              f"load tokenizer {TOKENIZER_MODEL!r} at {revision!r} "
              f"({type(e).__name__}: {e}). "
              f"Chunks are word-bounded ONLY on this run -- the measured tail "
              f"(340 §2.3: 4.4% of chunk@2 passages > TEI max_seq) is NOT "
              f"mitigated. Install `transformers` (the build requirements) "
              f"before a real chunk@2 replay.", file=sys.stderr, flush=True)
        return None
    _TOKENIZER_STATE["status"] = "ok"
    _tokenizer_cache[cache_key] = tok
    return tok


def _token_len(text: str) -> Optional[int]:
    """THE token-length check for this module (doctrine 1/6: one home). Counts
    special tokens (CLS/SEP) so the result is directly comparable to a
    max_seq configured on the embed server. None if the tokenizer is
    unavailable (see `_get_tokenizer`) -- caller treats None as 'cannot check,
    pass through unsplit' rather than a false failure.

    The counting itself lives in `dense_token_counter.token_len`, which the
    materializer and the embed worker call too: this function is a
    policy-bound wrapper around the shared counter, never a second
    implementation of it."""
    tok = _get_tokenizer()
    if tok is None:
        return None
    from cytir.chunk import token_counter as dense_token_counter
    return dense_token_counter.count(tok, text)


def token_len(text: str) -> int:
    """Public embed-time token length under the ACTIVE chunk policy's tokenizer.

    The materializer's composed-text check calls this rather than reaching for a
    tokenizer object and writing its own `len(tok(...))` -- that second call site
    is precisely how the write side and the embed side came to disagree
    (task389). Raises rather than returning None: a caller that has already
    gated on `tokenizer_status() == "ok"` must not silently receive "cannot
    check" as an answer."""
    value = _token_len(text)
    if value is None:
        raise RuntimeError(
            "embed-time token counter unavailable; refusing to report a token "
            "length the chunker could not measure (tokenizer_status="
            f"{tokenizer_status()!r})")
    return value


#: Test-only opt-in (task 2289 ruling): `conftest.py::word_only_chunking_allowed` sets it for one
#: test. No production code writes it; a test asserts that. Pinning `tokenizer_status` to "ok" instead
#: would test a path production does not have.
_ALLOW_WORD_ONLY_FOR_TESTS = False


class TokenizerUnavailable(RuntimeError):
    """The exact embed tokenizer cannot load; generic chunking refuses (task 2395)."""


def _require_tokenizer_for_chunking() -> None:
    status = tokenizer_status()
    if status != "ok":
        raise TokenizerUnavailable(
            f"chunk_markdown: embed tokenizer {TOKENIZER_MODEL!r} unavailable "
            f"(tokenizer_status={status!r}); refusing word-only chunks. Install "
            f"`transformers` (build requirements), or pass allow_word_only=True "
            f"if word-only output is intended.")


def require_exact_tokenizer(caller: str) -> None:
    """Refuse a production write path on a box that cannot measure tokens.

    `_get_tokenizer` degrades visibly (word-only chunks) so pure-algorithm tests
    run without heavy deps; a stage that WRITES rows must not ship those chunks.
    The structured staging adapter does this check inline; callers outside it
    call this (task 2289). Raises rather than returning, with `caller` named."""
    status = tokenizer_status()
    if status != "ok" and not _ALLOW_WORD_ONLY_FOR_TESTS:
        raise RuntimeError(
            f"{caller}: exact embed tokenizer unavailable (tokenizer_status="
            f"{status!r}); refusing to write word-only chunks")


def _bisect_split(units: list[str], sep: str, prefix: str) -> list[str]:
    """Generic binary-search hard-split: cut `units` (lines, or words as a
    last resort) into the fewest `sep`-joined groups such that every
    `prefix + group` fits MAX_CHUNK_TOKENS, measured with the REAL tokenizer
    (never an estimated ratio). O(log n) tokenizer calls per cut; always makes
    progress (a single pathological unit is emitted alone) even if it alone
    exceeds the budget."""
    n = len(units)
    if not units:
        return []
    # The library's documented test/development degradation is word-only.  A
    # production staging adapter must preflight ``tokenizer_status == ok``;
    # here, keep the degradation internally coherent and never compare None to
    # an integer or lose source text halfway through a split.
    if _token_len(prefix + sep.join(units)) is None:
        return [sep.join(units)]
    parts: list[str] = []
    i = 0
    while i < n:
        unit_tokens = _token_len(prefix + units[i])
        if unit_tokens is not None and unit_tokens > MAX_CHUNK_TOKENS:
            if sep == "" or len(units[i]) <= 1:
                raise ValueError(
                    "token hard split cannot fit prefix plus one Unicode code point")
        lo, hi, best = i + 1, n, i + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = prefix + sep.join(units[i:mid])
            candidate_tokens = _token_len(candidate)
            if candidate_tokens is None or candidate_tokens <= MAX_CHUNK_TOKENS:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        parts.append(sep.join(units[i:best]))
        i = best
    return parts


def _token_hard_split(prefix: str, body: str) -> list[str]:
    """Token-bound split of one chunk whose (prefix + body) exceeds
    MAX_CHUNK_TOKENS. Last resort, reached only by the pathological tail (340
    measured 4.4% of chunk@2 passages). LINE-first, mirroring `_split_block`'s
    own word-based hierarchy: a table row / code line stays whole unless it
    ALONE still overflows the budget, in which case only that one line falls
    to a word-level split. (A naive whitespace-flattening split here would
    silently merge table rows across the `|`...`|` boundary — caught by
    `test_single_oversized_block_is_split_internally`.)"""
    lines = body.split("\n")
    out: list[str] = []
    for chunk_of_lines in _bisect_split(lines, "\n", prefix):
        solo_lines = chunk_of_lines.split("\n")
        if len(solo_lines) == 1:
            tlen = _token_len(prefix + chunk_of_lines)
            if tlen is not None and tlen > MAX_CHUNK_TOKENS:
                words = chunk_of_lines.split()
                if words:
                    word_parts = _bisect_split(words, " ", prefix)
                    for word_part in word_parts:
                        # A minified/table/blob token can itself exceed the
                        # model ceiling.  Retain every character and make
                        # progress at Unicode-code-point boundaries rather
                        # than emitting an illegal chunk for a later worker.
                        word_tokens = _token_len(prefix + word_part)
                        if word_tokens is not None and word_tokens > MAX_CHUNK_TOKENS:
                            out.extend(_bisect_split(list(word_part), "", prefix))
                        else:
                            out.append(word_part)
                else:
                    out.extend(_bisect_split(list(chunk_of_lines), "", prefix))
                continue
        out.append(chunk_of_lines)
    return out

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
_TABLE_RE = re.compile(r"^\s*\|")
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s)")


def word_count(s: str) -> int:
    """The ONE word-count used for every size decision in this module.

    Whitespace split, deliberately identical to `fp.chunk`'s `len(x.split())`
    so chunk@1 and chunk@2 passage-size numbers are directly comparable. This
    is a sizing heuristic, not a mirror contract — BM25 tokenization
    (`aw_common.bm25_tokens`) and digesting (`aw_b1_identity.content_digest`)
    remain the only tokenizers of record and are never re-implemented here."""
    return len(s.split())


# --------------------------------------------------------------------------
# 1. markdown -> blocks
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Block:
    kind: str          # "heading" | "code" | "table" | "list" | "para"
    text: str          # verbatim block text (inner newlines preserved)
    level: int = 0     # heading level, 0 for non-headings
    title: str = ""    # heading title, "" for non-headings

    @property
    def words(self) -> int:
        return word_count(self.text)


def parse_blocks(markdown: str) -> list[Block]:
    """Split markdown into atomic blocks, fence-aware.

    A fenced code block is ONE block even when it contains blank lines or lines
    that look like headings — `# comment` inside a python fence is not a section
    header. (The 237b prototype header-matched line-by-line and consequently cut
    inside code fences; see task 340 findings.)
    """
    lines = (markdown or "").split("\n")
    blocks: list[Block] = []
    buf: list[str] = []
    buf_kind = "para"

    def flush():
        nonlocal buf, buf_kind
        if buf:
            txt = "\n".join(buf).strip("\n")
            if txt.strip():
                blocks.append(Block(kind=buf_kind, text=txt))
        buf = []
        buf_kind = "para"

    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        fm = _FENCE_RE.match(ln)
        if fm:
            flush()
            fence = fm.group(1)[:3]
            fenced = [ln]
            i += 1
            while i < n:
                fenced.append(lines[i])
                if lines[i].strip().startswith(fence):
                    i += 1
                    break
                i += 1
            blocks.append(Block(kind="code", text="\n".join(fenced)))
            continue
        if not ln.strip():
            flush()
            i += 1
            continue
        hm = _HEADER_RE.match(ln)
        if hm:
            flush()
            blocks.append(Block(kind="heading", text=ln.strip(),
                                level=len(hm.group(1)), title=hm.group(2).strip()))
            i += 1
            continue
        kind = "table" if _TABLE_RE.match(ln) else ("list" if _LIST_RE.match(ln) else "para")
        if buf and kind != buf_kind:
            flush()
        buf_kind = kind
        buf.append(ln.rstrip())
        i += 1
    flush()
    return blocks


# --------------------------------------------------------------------------
# 2. blocks -> sections
# --------------------------------------------------------------------------

@dataclass
class Section:
    path: tuple[str, ...]        # heading breadcrumb, () for the page lead
    blocks: list[Block] = field(default_factory=list)

    @property
    def body(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    @property
    def words(self) -> int:
        return word_count(self.body)


def parse_sections(markdown: str) -> list[Section]:
    """Group blocks under their heading breadcrumb. Content before the first
    heading is the page lead (path `()`). Empty sections (a heading with no
    body) are dropped, but their title stays in the breadcrumb of the
    subsections beneath them."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    cur = Section(path=())
    for b in parse_blocks(markdown):
        if b.kind == "heading":
            if cur.blocks:
                sections.append(cur)
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, b.title))
            cur = Section(path=tuple(t for _, t in stack))
            continue
        cur.blocks.append(b)
    if cur.blocks:
        sections.append(cur)
    return sections


# --------------------------------------------------------------------------
# 3. sections -> chunks
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    ordinal: int          # 0-based position in the document; STABLE per rerun
    text: str             # breadcrumb-prefixed text — this is what gets embedded
    body: str             # text without the breadcrumb prefix
    header_path: str      # "Trainer > Parameters" or ""
    n_words: int          # word_count(text)
    unit: str             # "section" | "section_part" | "lead" | "lead_part"
    part: int             # 0-based part index inside a hard-split section
    n_parts: int          # number of parts the section was split into
    chunk_epoch: int = field(default_factory=lambda: CHUNK_EPOCH)
    chunk_rule: str = field(default_factory=lambda: CHUNK_RULE_LEGEND)
    # Adjacent-section packs have more than one local heading.  Their dense
    # projection must retain each atom's local heading in source order, rather
    # than pretending the whole pack has one (or leaking full breadcrumbs into
    # dense).  Normal chunks leave this unset and keep the long-standing
    # leaf-heading-plus-body projection below.
    dense_override: str | None = None
    channel_atoms: tuple[ChannelAtom, ...] | None = None


def local_heading(header_path: str) -> str:
    """Return the deepest component of a retained chunk breadcrumb."""
    if not header_path:
        return ""
    return header_path.rsplit(BREADCRUMB_SEP, 1)[-1].strip()


def dense_text(chunk: Chunk) -> str:
    """Project one chunk onto the dense retrieval surface.

    Canonical ``Chunk.text`` deliberately retains the full breadcrumb: it is
    the identity/content surface and the BM25 surface.  Dense retrieval gets
    only the most local retained heading plus the body, avoiding repeated
    ancestor labels without changing boundaries or point identity.
    """
    if chunk.dense_override is not None:
        return chunk.dense_override
    heading = local_heading(chunk.header_path)
    return f"{heading}\n{chunk.body}" if heading else chunk.body


def _breadcrumb(path: tuple[str, ...]) -> str:
    """Breadcrumb prefix, truncated from the LEFT (the deepest headings carry
    the most section-specific signal) to MAX_PREFIX_WORDS."""
    if not path:
        return ""
    keep: list[str] = []
    total = 0
    for title in reversed(path):
        w = word_count(title)
        if keep and total + w > MAX_PREFIX_WORDS:
            break
        keep.insert(0, title)
        total += w
        if total >= MAX_PREFIX_WORDS:
            break
    crumb = BREADCRUMB_SEP.join(keep)
    if word_count(crumb) > MAX_PREFIX_WORDS:      # a single very long heading
        crumb = " ".join(crumb.split()[:MAX_PREFIX_WORDS])
    return crumb


def _split_long_text(text: str, budget: int) -> list[str]:
    """Word-level split of one atomic line that alone exceeds the budget.
    Last resort — reached only by pathological single lines (minified JS, a
    base64 blob, a 900-word table row)."""
    words = text.split()
    step = max(budget - OVERLAP_WORDS, 1)
    parts, i = [], 0
    while i < len(words):
        parts.append(" ".join(words[i:i + budget]))
        i += step
    return parts


def _split_block(b: Block, budget: int) -> list[str]:
    """Split one oversized block, preferring line boundaries (a table row, a
    list item, a code line) over mid-line cuts."""
    lines = b.text.split("\n")
    out: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for ln in lines:
        w = word_count(ln)
        if w > budget:
            if cur:
                out.append("\n".join(cur))
                cur, cur_n = [], 0
            out.extend(_split_long_text(ln, budget))
            continue
        if cur and cur_n + w > budget:
            out.append("\n".join(cur))
            # Overlap carry. NOTE the fp.chunk (chunk@1) defect this avoids:
            # there the loop breaks on `tn >= overlap_words`, which is true
            # after the FIRST insert when overlap is 0 — so "overlap 0" carried
            # a whole paragraph and inflated the corpus 1.31x (task 340 §2.2).
            # Here overlap 0 carries nothing, and a carry that would not leave
            # room for progress is dropped.
            tail, tn = [], 0
            if OVERLAP_WORDS:
                for q in reversed(cur):
                    tail.insert(0, q)
                    tn += word_count(q)
                    if tn >= OVERLAP_WORDS:
                        break
                if tn >= budget:
                    tail, tn = [], 0
            cur, cur_n = tail, tn
        cur.append(ln)
        cur_n += w
    if cur:
        out.append("\n".join(cur))
    return out


def _pack_blocks(blocks: list[Block], budget: int, min_words: int) -> list[str]:
    """Greedily fill parts up to `budget`, cutting at BLOCK boundaries; only an
    individually-oversized block is cut internally (237 §4.1's mandate)."""
    parts: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for b in blocks:
        w = b.words
        if w > budget:
            if cur:
                parts.append("\n\n".join(cur))
                cur, cur_n = [], 0
            pieces = _split_block(b, budget)
            # the trailing piece may be small — let it start the next part
            for p in pieces[:-1]:
                parts.append(p)
            if pieces:
                cur, cur_n = [pieces[-1]], word_count(pieces[-1])
            continue
        if cur and cur_n + w > budget:
            parts.append("\n\n".join(cur))
            cur, cur_n = [], 0
        cur.append(b.text)
        cur_n += w
    if cur:
        parts.append("\n\n".join(cur))
    # A tail below the retrievability floor is merged back into its predecessor
    # rather than dropped: 14 words of overflow beats 14 words of lost content.
    if len(parts) > 1 and word_count(parts[-1]) < min_words:
        tail = parts.pop()
        parts[-1] = parts[-1] + "\n\n" + tail
    return parts


def _can_use_balanced_word_guard(text: str) -> bool:
    """Whether one near-target chunk can safely use the policy's word guard.

    The word target is a retrieval-shape heuristic, while the tokenizer ceiling
    is the actual embedder safety boundary.  The adjacent-section experiment
    may therefore keep a *terminal* 769--783-word unit whole, but only after
    the exact tokenizer says it fits.  If the tokenizer is unavailable, this
    returns false: a visible word-bounded degradation must never pretend to
    have checked an exact token constraint.
    """
    words = word_count(text)
    if not TARGET_WORDS < words <= HARD_MAX_WORDS:
        return False
    tokens = _token_len(text)
    return tokens is not None and tokens <= MAX_CHUNK_TOKENS


class ChannelAtom(NamedTuple):
    """One indivisible atom, carrying its channel prefixes EXPLICITLY.

    The prefixes are the whole point.  A consumer that has to split an
    oversized atom needs to know where the heading ends and the body begins,
    and that boundary is NOT recoverable from the composed strings: a
    heading-less lead section produces canonical == dense with no prefix at
    all, which is byte-indistinguishable from a genuine "heading\\nbody" atom.
    Re-deriving it downstream by partitioning misreads body content as a
    heading (silently, when the atom has no blank line) or reports a spurious
    channel divergence (loudly, when it does -- 93 quarantined documents in the
    Task389 campaign).  The producer knows the answer, so it says it.

    A NamedTuple so existing `atom[0]` / `atom[1]` indexing keeps working.
    """
    canonical: str
    dense: str
    canonical_prefix: str = ""
    dense_prefix: str = ""

    @property
    def surface_body(self) -> str:
        return self.canonical[len(self.canonical_prefix):]

    @property
    def dense_body(self) -> str:
        return self.dense[len(self.dense_prefix):]


@dataclass(frozen=True)
class _SectionAtom:
    """One source-adjacent structured unit with explicit retrieval channels."""
    canonical: str
    dense: str
    members: tuple[ChannelAtom, ...] = ()

    @property
    def words(self) -> int:
        return word_count(self.canonical)


def _inline_section_atoms(sections: list[Section], min_words: int) -> list[_SectionAtom]:
    """Turn ordered sections into independently movable, heading-full atoms.

    This is deliberately separate from the legacy section emitter.  Its old
    contract was one ordinary section per embedding; this experimental policy
    instead permits several adjacent sections in the same embedding.  A
    section heading is copied into its atom body before packing, so crossing a
    chunk boundary never makes a retained section anonymous.
    """
    atoms: list[_SectionAtom] = []
    for section in sections:
        heading = _breadcrumb(section.path)
        local = local_heading(heading)
        canonical_prefix = (heading + "\n\n") if heading else ""
        dense_prefix = (local + "\n") if local else ""
        def append_body(body: str) -> None:
            dense_parts = _token_hard_split(dense_prefix, body)
            for dense_part in dense_parts:
                dense_tokens = _token_len(dense_prefix + dense_part)
                if dense_tokens is not None and dense_tokens > MAX_CHUNK_TOKENS:
                    raise ValueError("source-level dense token splitter emitted an overflow")
                canonical = canonical_prefix + dense_part
                dense = dense_prefix + dense_part
                atoms.append(_SectionAtom(canonical, dense, (ChannelAtom(
                    canonical, dense, canonical_prefix, dense_prefix),)))

        whole = canonical_prefix + section.body
        if word_count(whole) <= TARGET_WORDS or _can_use_balanced_word_guard(whole):
            append_body(section.body)
            continue

        # A long section is the one legitimate exception to section atomicity.
        # Repeating its heading on every hard-split part preserves the local
        # topic signal that the old breadcrumb prefix supplied for each part.
        budget = max(TARGET_WORDS - word_count(heading), 1)
        for body in _pack_blocks(section.blocks, budget, min_words):
            # Token safety is a source-level split: both channel projections
            # retain the same body and heading semantics.  The materializer
            # later performs its stricter title/site-composed check.
            append_body(body)
    return atoms


def _pack_adjacent_section_atoms(atoms: list[_SectionAtom]) -> list[_SectionAtom]:
    """Greedily pack source-adjacent section atoms, then balance the tail.

    Ordinary packing stops at the soft word target.  On the terminal boundary
    only, the policy may merge the otherwise-orphaned tail into its predecessor
    up to the immutable word guard, conditional on the exact tokenizer limit.
    This keeps e.g. a 760-word document whole when it is actually safe instead
    of creating a tiny second embedding merely because it crossed the heuristic
    target by a few words.
    """
    parts: list[_SectionAtom] = []
    current: _SectionAtom | None = None
    for atom in atoms:
        if current is None:
            current = atom
        elif current.words + atom.words <= TARGET_WORDS:
            candidate = _SectionAtom(current.canonical + "\n\n" + atom.canonical,
                                     current.dense + "\n\n" + atom.dense,
                                     current.members + atom.members)
            # Dense is the model surface; do not create an atom-boundary pack
            # that already exceeds its exact source-level token ceiling.
            candidate_tokens = _token_len(candidate.dense)
            if candidate_tokens is None or candidate_tokens <= MAX_CHUNK_TOKENS:
                current = candidate
            else:
                parts.append(current)
                current = atom
        else:
            parts.append(current)
            current = atom
    if current is not None:
        parts.append(current)

    if len(parts) >= 2:
        balanced = _SectionAtom(parts[-2].canonical + "\n\n" + parts[-1].canonical,
                                parts[-2].dense + "\n\n" + parts[-1].dense,
                                parts[-2].members + parts[-1].members)
        balanced_tokens = _token_len(balanced.dense)
        if (TARGET_WORDS < balanced.words <= HARD_MAX_WORDS and
                (balanced_tokens is None or balanced_tokens <= MAX_CHUNK_TOKENS)):
            parts[-2:] = [balanced]
    return parts


def _pack_whole_balanced_atoms(atoms: list[_SectionAtom]) -> list[_SectionAtom]:
    """One-home implementation of Task389's measured structural partition."""
    if not atoms:
        return []
    weights = [word_count(atom.dense) for atom in atoms]
    total = sum(weights)
    if total <= HARD_MAX_WORDS:
        whole = _SectionAtom("\n\n".join(a.canonical for a in atoms),
                             "\n\n".join(a.dense for a in atoms),
                             tuple(m for a in atoms for m in a.members))
        whole_tokens = _token_len(whole.dense)
        if whole_tokens is None or whole_tokens <= MAX_CHUNK_TOKENS:
            return [whole]
        return _pack_adjacent_section_atoms(atoms)
    wanted = (total + HARD_MAX_WORDS - 1) // HARD_MAX_WORDS
    out, start = [], 0
    for part in range(wanted - 1):
        remaining = wanted - part - 1
        remain_words = sum(weights[start:])
        running, choices = 0, []
        for end in range(start, len(atoms) - remaining):
            running += weights[end]
            if running > HARD_MAX_WORDS: break
            if remain_words - running <= remaining * HARD_MAX_WORDS:
                choices.append((abs(running - remain_words / (remaining + 1)), -running, end + 1))
        if not choices:
            return _pack_adjacent_section_atoms(atoms)
        end = min(choices)[2]; group = atoms[start:end]
        out.append(_SectionAtom("\n\n".join(a.canonical for a in group), "\n\n".join(a.dense for a in group), tuple(m for a in group for m in a.members)))
        start = end
    group = atoms[start:]
    out.append(_SectionAtom("\n\n".join(a.canonical for a in group), "\n\n".join(a.dense for a in group), tuple(m for a in group for m in a.members)))
    if any((_token_len(part.dense) or 0) > MAX_CHUNK_TOKENS for part in out):
        return _pack_adjacent_section_atoms(atoms)
    return out


def _chunk_adjacent_sections(sections: list[Section], min_words: int) -> list[Chunk]:
    """Experimental epoch-3 adjacent-section policy implementation."""
    atoms = _inline_section_atoms(sections, min_words)
    parts = (_pack_whole_balanced_atoms(atoms)
             if _ACTIVE_POLICY.section_packing_policy == "keep-whole-balanced"
             else _pack_adjacent_section_atoms(atoms))
    n_parts = len(parts)
    out: list[Chunk] = []
    for ordinal, part in enumerate(parts):
        # Unlike the legacy prose windower, this policy is conserving: an
        # unmergeable short tail is emitted, never silently discarded.
        out.append(Chunk(
            ordinal=ordinal, text=part.canonical, body=part.canonical, header_path="",
            n_words=part.words,
            unit=("adjacent_sections" if n_parts == 1
                  else "adjacent_sections_part"),
            part=ordinal, n_parts=n_parts, dense_override=part.dense,
            channel_atoms=part.members))
    return out


def chunk_markdown(markdown: str, *, min_words: Optional[int] = None,
                   allow_word_only: bool = False) -> list[Chunk]:
    """Structure-aware markdown chunking at `TARGET_WORDS`, overlap 0.

    Refuses with `TokenizerUnavailable` when the embed tokenizer cannot load
    (task 2395): without it F1's token ceiling is not enforced and the chunks are
    word-bounded only. `allow_word_only=True` is the explicit opt-in for a caller
    that knowingly wants that (tests of the pure algorithm); production passes
    nothing. `config()` / `tokenizer_status()` still report "unavailable" then.

    Pure function: same markdown + same config => byte-identical output.
    Documents with no markdown headings degrade gracefully to block-packing at
    the same target (NOT to a different chunker — doctrine 3: a fallback with
    different semantics is a lie about system state).

    `min_words` (F6, task 340 / task 360):
    per-call override for MIN_CHUNK_WORDS. The module default (15) exists to
    drop unretrievable STUB fragments left over from packing PROSE into 768w
    windows. It is the wrong rule applied to a source that is already ONE
    atomic short record -- LCSH subject headings (median 10w), ch_courts
    headnotes (median 12w), ai_autodoc_reference entries (median 14w) were
    built by dedicated source-specific builders, never by a windower, and
    never needed to clear a fragment floor. Replaying the generic chunker over
    one of those routes at the module default SILENTLY DROPS THE WHOLE RECORD
    (a 10-word doc chunks to zero output). Callers processing a known
    card/reference-shaped route pass a lower floor (e.g. 1) explicitly; the
    module default is unchanged for the generic prose path, so every existing
    caller and test stays byte-identical.
    """
    if not allow_word_only and not _ALLOW_WORD_ONLY_FOR_TESTS:
        _require_tokenizer_for_chunking()
    eff_min = MIN_CHUNK_WORDS if min_words is None else min_words
    sections = parse_sections(markdown)
    if not sections:
        return []

    if _ACTIVE_POLICY.algorithm in (ADJACENT_SECTION_PACK, LAW_ARTICLE_SUB_PACK):
        # `law-article-sub-pack-v1` (task 573 / E573-1) is the adjacent-section
        # packer applied to ONE law article's text. The outer boundary -- the
        # article -- is imposed by the caller (`cytir/chunk/law_subchunk.py`) from
        # the source-native anchor and is not this function's to move; ADR-090
        # §4 binds it in both directions. What differs from `chunk.reference` is
        # therefore not the packing but WHAT IS HANDED TO IT, which is exactly
        # why the two carry different policy ids, epochs and families: a row's
        # rule legend has to say which boundaries produced it.
        return _chunk_adjacent_sections(sections, eff_min)
    if _ACTIVE_POLICY.algorithm != STRUCT_WORD_PACK:
        raise ValueError(
            f"active chunk algorithm {_ACTIVE_POLICY.algorithm!r} has no implementation")

    # (a) merge sections below MERGE_MIN_WORDS into a neighbour: backwards into
    # the previous unit when that keeps it under target, else forwards into the
    # next. A merged-in section keeps its own heading INLINE so its title is
    # still searchable.
    def _title_marker(path: tuple[str, ...], host_path: tuple[str, ...]):
        """A merged-away section keeps its own heading INLINE so its title stays
        searchable — dropping it would lose real text (the heading is often the
        most distinctive token on a stub section)."""
        if not path or path == host_path:
            return None
        return Block(kind="para", text=path[-1] + ":")

    units: list[Section] = []
    carry: list[Block] = []
    for sec in sections:
        blocks = carry + sec.blocks
        carry = []
        s = Section(path=sec.path, blocks=blocks)
        if s.words < MERGE_MIN_WORDS:
            if units and units[-1].words + s.words <= TARGET_WORDS:
                prev = units[-1]
                m = _title_marker(s.path, prev.path)
                if m:
                    prev.blocks.append(m)
                prev.blocks.extend(s.blocks)
                continue
            # push forward onto the next section, carrying the heading with it
            m = _title_marker(s.path, ())
            carry = ([m] if m else []) + s.blocks
            continue
        units.append(s)
    if carry:
        if units:
            units[-1].blocks.extend(carry)
        else:
            units.append(Section(path=(), blocks=carry))

    # (b) emit; hard-split any unit over target, budgeting for the breadcrumb.
    out: list[Chunk] = []
    ordinal = 0
    for u in units:
        crumb = _breadcrumb(u.path)
        prefix = (crumb + "\n") if crumb else ""
        budget = max(TARGET_WORDS - word_count(crumb), MERGE_MIN_WORDS)
        parts = _pack_blocks(u.blocks, budget, eff_min)
        # F1: word ceiling doesn't bound tokens (340 §2.3) -- re-check every
        # candidate against the real tokenizer and split further BEFORE
        # computing n_parts, so part/n_parts number the TRUE final pieces.
        # `forced` tracks pieces born from a TOKEN split: the eff_min floor
        # below exists to drop naturally-tiny fragments, not to discard real
        # content whose only "fault" is being the small remainder of a
        # safety split -- exempt those from the floor (a rare short chunk
        # beats silently losing the words in it; the token-split tail cannot
        # be merged back into its predecessor the way `_pack_blocks` merges a
        # word-tail, because the predecessor is already packed to the token
        # ceiling and a merge would just reopen the overflow F1 exists to
        # close).
        expanded: list[str] = []
        forced: list[bool] = []
        for body in parts:
            tlen = _token_len(prefix + body)
            if tlen is not None and tlen > MAX_CHUNK_TOKENS:
                pieces = _token_hard_split(prefix, body)
                expanded.extend(pieces)
                forced.extend([True] * len(pieces))
            else:
                expanded.append(body)
                forced.append(False)
        parts = expanded
        n_parts = len(parts)
        base_unit = "lead" if not u.path else "section"
        for k, body in enumerate(parts):
            text = prefix + body
            n = word_count(text)
            if n < eff_min and not forced[k]:
                continue
            out.append(Chunk(
                ordinal=ordinal, text=text, body=body, header_path=crumb,
                n_words=n,
                unit=base_unit if n_parts == 1 else base_unit + "_part",
                part=k, n_parts=n_parts))
            ordinal += 1
    return out


def chunk_texts(markdown: str, *, min_words: Optional[int] = None,
                allow_word_only: bool = False) -> list[str]:
    """Drop-in replacement for `fetch_process.chunk(md)`. `min_words`: see
    `chunk_markdown` (F6)."""
    return [c.text for c in chunk_markdown(
        markdown, min_words=min_words, allow_word_only=allow_word_only)]


# --------------------------------------------------------------------------
# 4. identity — the A33 guarantee
# --------------------------------------------------------------------------

@dataclass
class ChunkRecord:
    ordinal: int
    text: str
    header_path: str
    n_words: int
    unit: str
    digest: str
    tier: str
    anchor_component: str
    id_salt: int
    point_id: str
    chunk_epoch: int = field(default_factory=lambda: CHUNK_EPOCH)
    chunk_rule: str = field(default_factory=lambda: CHUNK_RULE_LEGEND)


def assign_chunk_ids(canonical_url: str, chunks: Iterable[Chunk], *,
                     source_type: Optional[str] = None,
                     title: Optional[str] = None,
                     native_anchor: Optional[str] = None,
                     force_tier_b: bool = False,
                     stats: Optional[dict] = None) -> list[ChunkRecord]:
    """Mint a point_id per chunk through `aw_b1_identity` — never locally.

    The failure this exists to make impossible (tasks 318/319/325, GAP A33):
    an anchor-less multi-chunk page whose chunks all resolve to the SAME
    anchor component collapses to ONE surviving point
    (anthropic.com/constitution: 218 chunks -> 1; ~1.8M collisions corpus-wide).
    Here every chunk digests its OWN text, so distinct chunks are distinct
    records by construction. Two guards remain:

      * identical chunk TEXT within one url (repeated boilerplate section) is a
        genuine duplicate — the later copy is dropped, counted as
        `duplicate_text` rather than silently overwriting;
      * any residual id collision gets an explicit persisted salt
        (`aw_b1_identity.salted_anchor_component`) and increments `collisions`,
        which callers must surface. It should always be 0 for this chunker;
        a non-zero value is a bug, not routine.

    Note a native per-chunk anchor (fedlex `art_336`) is Tier A and is supplied
    per chunk by the source-specific splitter, not by this generic chunker;
    `native_anchor` here is for callers that have ONE page-level anchor and
    know it is per-chunk-valid. Passing a page-level label (a title) is exactly
    the A33 trap — `classify_tier` demotes anchor==title to Tier B for you.
    """
    st = dict(collisions=0, duplicate_text=0) if stats is None else stats
    st.setdefault("collisions", 0)
    st.setdefault("duplicate_text", 0)
    seen_digest: set[str] = set()
    used_ids: set[str] = set()
    out: list[ChunkRecord] = []
    for c in chunks:
        digest = ident.content_digest(c.text)
        if digest in seen_digest:
            st["duplicate_text"] += 1
            continue
        seen_digest.add(digest)
        tier, comp = ident.classify_tier(source_type, native_anchor, True,
                                         digest, title=title,
                                         force_tier_b=force_tier_b)
        salt = 0
        pid = ident.mint_point_id(canonical_url, comp)
        while pid in used_ids:
            salt += 1
            st["collisions"] += 1
            pid = ident.mint_point_id(
                canonical_url, ident.salted_anchor_component(comp, salt))
        used_ids.add(pid)
        out.append(ChunkRecord(
            ordinal=c.ordinal, text=c.text, header_path=c.header_path,
            n_words=c.n_words, unit=c.unit, digest=digest, tier=tier,
            anchor_component=ident.salted_anchor_component(comp, salt),
            id_salt=salt, point_id=pid,
            # The record inherits the stamps of the CHUNK it is minting, not
            # of whatever policy happens to be installed in this module when
            # the minting runs (task 565). `ChunkRecord`'s default_factory
            # reads the module globals, and the only caller that chunks under
            # one policy and mints under another -- `chunk_reference_selector`,
            # whose context manager deliberately puts the previous policy back
            # before returning -- therefore stamped every chunk.reference row
            # `struct-md-768w-o0@2` / `rule_epochs.chunk = 2`. Epoch 2 is a
            # SUPERSEDED generic policy, so the rows would have entered the
            # index claiming a generation that never produced them, and no
            # count check, hash check or width check can see that
            # (corpus-doctrine 4). Caught on task 565's first materialized
            # record; it would have shipped task 541's sample the same way.
            chunk_epoch=c.chunk_epoch, chunk_rule=c.chunk_rule))
    return out


# --------------------------------------------------------------------------
# 5. stamps + config echo
# --------------------------------------------------------------------------

def stamps() -> dict:
    """The rule-epoch stamp carried by every record this chunker produces, so a
    mixed chunk@1/chunk@2 corpus stays legal and VISIBLE (specs/epochs.md §1)."""
    return {"chunk": CHUNK_EPOCH}


def tokenizer_status() -> str:
    """'ok' | 'unavailable' | 'untried' -- probes (and caches) the tokenizer if
    untried. A wave manifest MUST record this as 'ok'; 'unavailable' means F1's
    token ceiling was NOT enforced on that run (340 §2.3's 4.4%-tail risk is
    live) — never discover that after the fact (F6's own lesson, applied to F1)."""
    _get_tokenizer()
    return _TOKENIZER_STATE["status"] or "untried"


def config() -> dict:
    """Effective config — for the MANIFEST / task doc, so a rerun on another box
    can prove it used the same rule (doctrine 5)."""
    return {
        "policy": _ACTIVE_POLICY.manifest(),
        "policy_effective": effective_policy_value(),
        "algorithm": _ACTIVE_POLICY.algorithm,
        "chunk_epoch": CHUNK_EPOCH,
        "rule_legend": CHUNK_RULE_LEGEND,
        "sizing_unit": _ACTIVE_POLICY.sizing_unit,
        "target_words": TARGET_WORDS,
        "merge_min_words": MERGE_MIN_WORDS,
        "min_chunk_words": MIN_CHUNK_WORDS,
        "overlap_words": OVERLAP_WORDS,
        "max_prefix_words": MAX_PREFIX_WORDS,
        "hard_max_words": HARD_MAX_WORDS,
        "tokenizer_model": TOKENIZER_MODEL,
        "tokenizer_revision": TOKENIZER_REVISION,
        "max_chunk_tokens": MAX_CHUNK_TOKENS,
        "tokenizer_status": tokenizer_status(),
        # The CLASS, not just the model+revision. Task389: the shipped corpus's
        # 256 manifests all recorded the right model and the right revision and
        # `tokenizer_status: "ok"` -- and were still produced by the wrong
        # backend, because the class was the variable and nothing wrote it down.
        # A run could not notice, and an auditor could not tell from the
        # artifact. Recorded here so the manifest carries the evidence.
        **_tokenizer_identity(),
    }


def _tokenizer_identity() -> dict:
    """Resolved tokenizer class/backend, or an explicit unavailable marker."""
    tok = _get_tokenizer()
    if tok is None:
        return {"tokenizer_class": None, "tokenizer_is_fast": None}
    from cytir.chunk import token_counter as dense_token_counter
    identity = dense_token_counter.identity_of(
        tok, TOKENIZER_MODEL, effective_tokenizer_revision())
    return {"tokenizer_class": identity["tokenizer_class"],
            "tokenizer_is_fast": identity["tokenizer_is_fast"]}
