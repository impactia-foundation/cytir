"""Qdrant schema constants + query-side helpers (mirrors aw_common)."""
from __future__ import annotations

import functools
import hashlib
import json
import re
import unicodedata

from cytir.serve import settings as _settings  # the one place AW_* names are declared (task 1180)

# env-configurable so the arctic index deploys as a SEPARATE collection with the
# BGE-M3 `aw_passages` kept as an instant rollback (cut over by flipping the env).
COLLECTION = _settings.current("AW_COLLECTION")
DENSE_VEC = "dense"
SPARSE_LEXICAL = "lexical"
SPARSE_BM25 = "bm25"
# NOTE (task 307 / preflight A31/F13): a stale DENSE_DIM=256 constant lived here
# and was never imported anywhere — the live truncation dim is
# `models.py:MRL_DIM=128` (config B). Removed rather than "fixed" since nothing
# consumed it; re-add only if a real consumer needs a schema-level dim constant.

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# apostrophe-joined names (O'Hear, O'Brien, d'Angelo) — \w+ splits them and the
# 1-char prefix is dropped, destroying the distinctive surname. Index the joined
# form ADDITIVELY (keeps French elisions' clean token too). Must match pipeline.
_APOS_RE = re.compile(r"\w+(?:['’]\w+)+", re.UNICODE)

# Mirror of cytir/core/common.py: index- and query-side BM25 tokenization MUST match
# (German decompounding included). Both call the ONE rule in `cytir.core.decompound`
# (CharSplit, MIT; ADR-121). A missing splitter or table RAISES; no silent no-split.
from cytir.core.decompound import decompound_parts as _decompound_parts  # noqa: E402
from cytir.core.common import host_query_parts, url_host_label  # noqa: E402,F401  (task 2050; serve modules import them from here)


@functools.lru_cache(maxsize=200_000)
def _decompound(token: str) -> tuple[str, ...]:
    """German compound split (memoized). The splitter is deterministic per token but
    ~0.5 ms/call, and title-centrality runs it on every candidate's title tokens (task
    199 k-stability floor), so repeat calls are cached (byte-identical output). Returns
    a tuple so the cache value is immutable. A load failure raises and is not cached."""
    return _decompound_parts(token)


def _hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


# --- AW_FOLD_ACCENTS — query-side mirror of cytir/core/canon.py (task 290) ----
# MUST stay byte-identical to the write-side `_fold`/`bm25_tokens`; folding a
# query token against an un-folded index is identity, so this flag flips in
# lockstep with the rebuilt index at cutover. Default OFF ⇒ byte-identical to the
# pre-fold tokenizer (ASCII exact-lookups cannot regress).
_FOLD_ACCENTS = _settings.current("AW_FOLD_ACCENTS")


def _fold(s: str) -> str:
    """NFKD-normalize and strip combining marks (accent-fold). Must be
    byte-identical to cytir/core/canon.py::_fold."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


# --------------------------------------------------------------------------- #
#  ADR-099 RULING 2 — query-side IDF-threshold suppression (AW_BM25_QSTOP).
#
#  Task 446 measured, per token, that 100% of the trust-flooder's BM25 score on
#  the reported query is English function words (60% from `the` alone), and that
#  `idf('the') = 0.4442` on the live index — i.e. the "high-DF terms are
#  IDF-crushed" assumption in specs/rebuild-100m-plan-v2.md §2.4 is FALSE: a
#  small idf multiplied by an unbounded document term frequency decides the leg.
#  The existing per-token mitigation (AW_BM25_PER_TOKEN) is capped at 4 query
#  tokens and therefore never runs on the long questions that flood.
#
#  MECHANISM — measured IDF, not curated stopword lists. A query term is dropped
#  when its LIVE-INDEX IDF is below AW_BM25_QSTOP_IDF and the query holds more
#  than AW_BM25_QSTOP_MINTOKENS tokens. The IDF numbers come from the index
#  itself (cytir/serve/bm25_idf.json, produced by
#  measure_bm25_idf.py), so EN/DE/FR/IT are covered by ONE rule and the
#  behaviour follows the corpus rather than a hand-maintained per-language list.
#  Note the corpus, not the rule, decides how a language fares: German and
#  French function words score materially higher IDF than English ones in this
#  EN-heavy index (`der` 1.86, `della` 2.84 vs `the` 0.44), which is a true
#  statement about their discriminative power HERE, not a coverage gap.
#
#  STORAGE IS UNTOUCHED — this is query-construction only. cytir/core/canon.py
#  (the write side) is deliberately NOT mirrored: suppressing a term at index
#  time would delete evidence, while suppressing it at query time only declines
#  to ask for it.
#
#  MINTOKENS seam: 2..4-token queries are the AW_BM25_PER_TOKEN regime, where a
#  function word can BE the query ("the who", "let it be") and its df is already
#  handled by that machinery's rare/common window test. Suppression therefore
#  starts above that band.
#
#  NEVER-EMPTY guard: if suppression would leave no tokens, the unsuppressed bag
#  is returned — a query must never lose its whole lexical leg.
#
#  Flag-gated, default OFF => byte-identical tokenization (ADR-099 gates it on).
# --------------------------------------------------------------------------- #
_QSTOP = _settings.current("AW_BM25_QSTOP")
# Derived in task 461 from the measured distribution
# over 6,830 live-index tokens, then SELECTED by the gates:
#   * lower bound — must suppress `the` (0.4442), 60% of the flooder's score;
#   * upper bound — must keep the numerals (`10` at 1.845: article and version
#     numbers are load-bearing here) and the first content words (`one` 1.758);
#   * inside that window, 1.40 vs 1.50 was MEASURED, not argued. 1.50 also drops
#     `des` 1.420, `or` 1.429, `de` 1.430, `be` 1.432, `la` 1.494 — and `OR` is
#     the Swiss abbreviation for Obligationenrecht while `de la` names French
#     statutes, so those five cost two legal-slice ranks in eval_tasks
#     (le04 1->3, fr25 2->3) and one Slice B gold rank, and buy nothing on any
#     in-scope metric. 1.40 stops just below that band.
# At 1.40 the suppressed set is 19 tokens, every one an English function word
# with df >= 8.5M (25% of the index).
_QSTOP_IDF = _settings.current("AW_BM25_QSTOP_IDF")
_QSTOP_MINTOKENS = _settings.current("AW_BM25_QSTOP_MINTOKENS")
_IDF_TABLE = _settings.current("AW_BM25_IDF_TABLE")
# An unmeasured token is treated as high-IDF (kept): the table stores only the
# suppression candidates, so "absent" must mean "not known to be common".
UNKNOWN_IDF = float("inf")


def _load_idf_table() -> dict[int, float]:
    """{token hash -> measured live-index IDF}. Empty (=> lever inert) when the
    table is missing/bad or was measured under a different fold setting — a bad
    table must never break search (ADR-054 discipline, as _load_hub_hosts)."""
    try:
        with open(_IDF_TABLE) as f:
            cfg = json.load(f)
        if bool(cfg.get("fold_accents")) != bool(_FOLD_ACCENTS):
            raise ValueError(
                f"table fold_accents={cfg.get('fold_accents')} != "
                f"AW_FOLD_ACCENTS={_FOLD_ACCENTS}: token hashes would not match")
        cap = float(cfg.get("capture_max_idf", 0.0))
        if _QSTOP and _QSTOP_IDF > cap:
            # Silently under-suppressing is worse than not suppressing at all:
            # above the capture bound the table cannot answer "is this common?"
            raise ValueError(
                f"AW_BM25_QSTOP_IDF={_QSTOP_IDF} exceeds the table's "
                f"capture_max_idf={cap}; re-measure with a higher bound")
        return {int(h): float(v) for h, v in (cfg.get("idf") or {}).items()}
    except Exception as e:  # noqa: BLE001 — a bad idf table must not break search
        import logging
        # WARNING only when something is actually armed to use the table (the
        # ADR-099 Ruling 3 trust-floor term reads it too, hence the second env
        # check — a voided table would silently make that lever permissive).
        # Otherwise DEBUG: an unused artifact must not train operators to ignore
        # warnings.
        armed = _QSTOP or _settings.current("AW_TRUSTFLOOR_ABS")
        logging.getLogger("aw_server.schema").log(
            logging.WARNING if armed else logging.DEBUG,
            "bm25 idf table load failed (%s: %s) — ADR-099 IDF levers inert",
            type(e).__name__, e)
        return {}


IDF = _load_idf_table()


def token_idf(h: int) -> float:
    """Measured IDF of a token hash; UNKNOWN_IDF when unmeasured (=> high)."""
    return IDF.get(h, UNKNOWN_IDF)


def qstop_dropped(tf: dict[int, float]) -> frozenset[int]:
    """The hashes AW_BM25_QSTOP suppresses from this query bag (empty = inert).

    Split out of `bm25_tokens` so the token-GROUP view below and the ADR-099
    Ruling 3 trust-floor term reach the identical decision from the identical
    numbers — one truth, no second implementation of the threshold."""
    if not (_QSTOP and IDF) or len(tf) <= _QSTOP_MINTOKENS:
        return frozenset()
    drop = frozenset(h for h in tf if IDF.get(h, UNKNOWN_IDF) < _QSTOP_IDF)
    if len(drop) >= len(tf):
        return frozenset()          # never leave a query with no lexical leg
    return drop


def _bm25_tf(text: str, lang: str | None = None) -> dict[int, float]:
    """The raw {token hash -> query tf} bag, before any ADR-099 suppression."""
    tf: dict[int, float] = {}
    de = lang in (None, "de")
    fold = _FOLD_ACCENTS
    for tok in _TOKEN_RE.findall(text.lower()):
        if len(tok) < 2:
            continue
        key = _fold(tok) if fold else tok
        tf[_hash(key)] = tf.get(_hash(key), 0.0) + 1.0
        if de:
            # Decompound the raw (un-folded) token so char_split still sees the
            # umlauts it splits on, then fold each part before hashing.
            for part in _decompound(tok):
                pk = _fold(part) if fold else part
                tf[_hash(pk)] = tf.get(_hash(pk), 0.0) + 0.5
    for m in _APOS_RE.findall(text.lower()):
        j = m.replace("'", "").replace("’", "")
        if len(j) >= 2:
            jk = _fold(j) if fold else j
            tf[_hash(jk)] = tf.get(_hash(jk), 0.0) + 1.0
    return tf


def bm25_tokens(text: str, lang: str | None = None) -> dict[int, float]:
    tf = _bm25_tf(text, lang)
    drop = qstop_dropped(tf)
    return {h: v for h, v in tf.items() if h not in drop} if drop else tf


# --------------------------------------------------------------------------- #
#  TASK 411 — the SOURCE-WORD grouping of the same tokens `bm25_tokens` emits.
#
#  `bm25_tokens` returns a flat {hash: tf} bag, which is the right unit for
#  scoring and the WRONG unit for asking "which query terms must the candidate
#  match?" (task 411 directive c). Two live tokenizer behaviours make the flat
#  bag unusable as a requirement set, and both are real in this corpus:
#
#    * APOSTROPHE SPLIT — "O'Hear" emits BOTH `hear` (the \w+ part; `o` is
#      dropped at len<2) and the joined `ohear`. `hear` is a junk token no page
#      about this person need contain, so "every emitted token must match" is
#      UNSATISFIABLE on apostrophe names.
#    * GERMAN DE-COMPOUNDING — "Arbeitsvertrag" emits `arbeitsvertrag` +
#      `arbeits` + `vertrag` at half weight, so one source word inflates to
#      three requirements and a 3-word query counts as 5 terms (measured:
#      `Kuendigungsfrist Arbeitsvertrag BGB` -> 5 `q_content` terms, which is
#      why it already exceeds AW_NAMEGATE_MAXTERMS=4 — task 406 §3.2).
#
#  The requirement unit is therefore the SOURCE WORD, satisfied by ANY of the
#  tokens it emitted. `primary` is the word's canonical token — the joined form
#  for an apostrophe word, else the whole word — which is what a NAME token must
#  match (a name is not a morphological variant, mirroring _card_name_match).
#
#  ONE TRUTH (engineering-doctrine 1/6): this shares `_TOKEN_RE`, `_APOS_RE`,
#  `_decompound`, `_fold` and `_hash` with `bm25_tokens` above rather than
#  re-implementing them, and `test_task411_tokengroups.py` asserts
#  byte-identity of the two token SETS over a corpus. Query-side only: the
#  index side has no query to group, so this is deliberately NOT mirrored into
#  cytir/core/canon.py — there is nothing there to keep in sync with.
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)


# --------------------------------------------------------------------------- #
#  TASK 665 — DOTTED VERSION TOKENS ("5.3", "3.12.2", "v0.28.0" -> "0.28.0").
#
#  Every lexical feature AW has is blind to a dotted version: `_TOKEN_RE = \w+`
#  splits "GLM-5.3" into `glm` + `5` + `3`, and all three consumers then drop
#  the digits — `bm25_tokens` at len < 2, `retrieval._query_terms` at len < 3,
#  `retrieval._content_terms` for having no letter. Measured on aw_e8
#  (task 665 §1): the query "GLM 5.3" reaches every lexical
#  feature as the single token `glm`, so the HF page titled "GLM-5, GLM-5.1,
#  GLM-5.2" scores a PERFECT title-centrality (1.0) and the three HN pages
#  titled "GLM-5.3 ..." score 0.25-0.33.
#
#  This helper is READ-SIDE ONLY: it is applied to the query and to payload
#  display text (title/snippet) at stage 2, never to a stored sparse vector, so
#  there is no write-side mirror to keep in sync (cf. `bm25_token_groups`). The
#  BM25 leg is deliberately NOT changed: its index holds no version token, and
#  adding one is a re-materialization of every stored sparse vector (ADR-037 /
#  ADR-109 territory, a rebuild decision) — see task 665 §4.
#
#  Shape: a run of digits with at least one ".digits" group, not glued to a
#  preceding digit/dot or a currency sign, and not followed by a digit or "%".
#  So "Qwen3.8-27B" -> 3.8, "Python 3.12.2" -> 3.12.2, "94.9%" -> nothing,
#  "$1.4B" -> nothing. A letter prefix is allowed on purpose ("glm5.3", "v9.3").
# --------------------------------------------------------------------------- #
_VERSION_RE = re.compile(r"(?<![\d.$€£])(\d+(?:\.\d+)+)(?![\d%])")


def version_tokens(text: str | None) -> frozenset[str]:
    """The dotted version tokens in `text` (see the block comment)."""
    return frozenset(_VERSION_RE.findall(text or ""))


def version_match(q: str, t: str) -> bool:
    """Does text version `t` name query version `q`? Equal, or `t` is a
    dot-boundary refinement of `q` ("3.12.2" names "3.12"). Never a plain
    prefix: "10.15" does not name "10.1", "5.30" does not name "5.3"."""
    return t == q or t.startswith(q + ".")


def bm25_token_groups(text: str,
                      lang: str | None = None) -> list[tuple[str, int, tuple[int, ...]]]:
    """[(source_word, primary_hash, all_hashes)] for `text`.

    The union of `all_hashes` over the returned groups is EXACTLY
    `set(bm25_tokens(text, lang))`; source words that emit no token (len < 2)
    are omitted. Order follows the raw text.

    ADR-099 Ruling 2: the SAME suppression decision is applied here, from the
    same `qstop_dropped` call on the same raw bag, so the partition contract
    above holds with AW_BM25_QSTOP on as well as off. A term the lexical leg
    does not ask for is not a term a candidate can be REQUIRED to match; a word
    whose every token is suppressed drops out of the group list entirely.
    """
    de = lang in (None, "de")
    fold = _FOLD_ACCENTS
    out: list[tuple[str, int, tuple[int, ...]]] = []
    for word in _WORD_RE.findall(text.lower()):
        hs: list[int] = []
        primary: int | None = None
        for tok in _TOKEN_RE.findall(word):
            if len(tok) < 2:
                continue
            key = _fold(tok) if fold else tok
            h = _hash(key)
            if primary is None:
                primary = h
            hs.append(h)
            if de:
                for part in _decompound(tok):
                    pk = _fold(part) if fold else part
                    hs.append(_hash(pk))
        for m in _APOS_RE.findall(word):
            j = m.replace("'", "").replace("’", "")
            if len(j) >= 2:
                jk = _fold(j) if fold else j
                # the JOINED form is the canonical token of an apostrophe word
                # ("ohear", not the "hear" fragment) — see the block comment.
                primary = _hash(jk)
                hs.append(primary)
        if hs and primary is not None:
            out.append((word, primary, tuple(dict.fromkeys(hs))))
    drop = qstop_dropped(_bm25_tf(text, lang))
    if drop:
        kept = []
        for word, primary, hs in out:
            hs2 = tuple(h for h in hs if h not in drop)
            if not hs2:
                continue
            # a suppressed primary is re-pointed at the word's first surviving
            # token, so `primary in all_hashes` still holds for every group.
            kept.append((word, primary if primary not in drop else hs2[0], hs2))
        out = kept
    return out
