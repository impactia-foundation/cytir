"""Lexical matching shared by the query side and the candidate side (S10 step 1, task 1061;
moved verbatim from `retrieval.py`, which imports every name back).

Content terms, light stem matching, title centrality (task 104), entity-card name matching
(task 163), exact-match query terms, and their dotted-version-aware variants (task 665, the
`AW_VERSION_MATCH` lever; its flags stay with the caller in `retrieval.py`).

Pure: no environment read, no module state, no I/O. Its only server imports are the tokenizer
contract (`schema`: German decompounding and version tokens) and `signals._exact`.
"""
from __future__ import annotations

import re

from .schema import url_host_label
from . import schema
from .schema import _decompound
from .signals import _exact

# The word tokenizer of the lexical features (apostrophe-joined words stay one token).
_WORD = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)


# --------------------------------------------------------------------------- #
#  Title-centrality (task 104) — the relevance-side fix for the junk@1 class
#  that task 103 isolated: tangential pages outrank the canonical page because
#  dense+bm25 can't tell "the page is ABOUT X" from "the page MENTIONS X". A
#  page whose TITLE *is* the query topic ("Neural network" for "machine
#  learning neural networks") is far more likely the canonical source than a
#  faculty e-paper / CfP / journal article that merely mentions the terms in
#  its body. We reward that with a POSITIVE additive boost (never a penalty):
#  a low-centrality title (a French-titled page on an EN query, a synonym
#  title) simply doesn't get the boost and still ranks on its other legs —
#  cross-lingual / synonym recall is preserved.
#
#    coverage = fraction of query content-terms present in the title (recall)
#    focus    = fraction of the title's content-terms that are on-query
#               (precision -> a short on-topic title "Neural network" beats a
#                long title that merely contains the terms)
#    title_centrality = coverage * focus                         (in [0,1])
#
#  Matching is stem/prefix-aware ("networks"<->"network", "learning"<->"learn")
#  and German compounds are split via the SAME `_decompound` the bm25 leg uses,
#  so the signal aligns with the tokenizer. Stopwords (EN/DE/FR/IT function
#  words) are stripped so they don't dilute coverage/focus. v2 (future work,
#  NOT built here): a dense title-embedding similarity to catch cross-lingual /
#  synonym titles that surface-token matching misses.
# --------------------------------------------------------------------------- #
_ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)   # >=1 unicode letter
_STOPWORDS = frozenset({
    # EN
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "is", "are",
    "what", "how", "why", "when", "where", "which", "who", "with", "from", "by",
    "at", "as", "be", "this", "that", "does", "do", "vs", "into", "about", "it",
    # DE
    "der", "die", "das", "und", "oder", "von", "für", "ist", "sind", "wie",
    "was", "warum", "wann", "wo", "welche", "welcher", "wer", "mit", "im",
    "am", "an", "auf", "den", "dem", "des", "ein", "eine", "einer", "zur",
    "zum", "über", "nach", "bei",
    # FR
    "le", "la", "les", "de", "des", "du", "et", "ou", "pour", "est", "sont",
    "comment", "quoi", "pourquoi", "quand", "où", "quel", "quelle", "qui",
    "avec", "dans", "sur", "un", "une", "au", "aux", "en", "ce", "cette",
    # IT
    "il", "lo", "gli", "di", "del", "della", "dei", "come", "cosa", "perché",
    "quando", "dove", "quale", "chi", "con", "su", "che", "gli", "nel",
})


def _content_terms(text: str, lang: str | None = None) -> set[str]:
    """Distinctive content tokens of a query/title: lowercased, apostrophe-
    stripped, stopwords + pure-digit/symbol tokens dropped, German compounds
    split (aligned with the bm25 tokenizer)."""
    de = lang in (None, "de")
    out: set[str] = set()
    for m in _WORD.findall((text or "").lower()):
        t = m.replace("'", "").replace("’", "")
        if len(t) < 2 or t in _STOPWORDS or not _ALPHA_RE.search(t):
            continue
        out.add(t)
        if de:
            for part in _decompound(t):
                if part not in _STOPWORDS:
                    out.add(part)
    return out


def _stem_match(a: str, b: str) -> bool:
    """Light prefix/stem match so 'networks'<->'network', 'learning'<->'learn'
    unify without a full stemmer. Requires the shorter token be >=4 chars to
    avoid spurious prefixes ('cat'/'category')."""
    if a == b:
        return True
    return min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a))


def _title_centrality(q_content: set[str], title: str, lang: str | None = None) -> float:
    """coverage(query in title) * focus(title on-query). [0,1]. POSITIVE only —
    a title that doesn't match just scores 0 (no demotion)."""
    if not q_content:
        return 0.0
    t_terms = _content_terms(title, lang)
    if not t_terms:
        return 0.0
    mq = sum(1 for q in q_content if any(_stem_match(q, t) for t in t_terms))
    mt = sum(1 for t in t_terms if any(_stem_match(t, q) for q in q_content))
    coverage = mq / len(q_content)
    focus = mt / len(t_terms)
    return coverage * focus


def _card_aliases(title: str, names: str) -> list:
    """Distinct alias strings of an entity card (title + pipe-joined names)."""
    seen, out = set(), []
    for a in ((title or "") + " | " + (names or "")).split("|"):
        a = a.strip()
        if a and a.lower() not in seen:
            seen.add(a.lower()); out.append(a)
    return out


def _card_name_match(q_content: set[str], title: str, names: str,
                     lang: str | None = None, *, host_label: str | None = None,
                     card_url: str = "") -> float:
    """Does the query NAME this entity card? (task 163 v2). [0,1].
      coverage = fraction of query content-terms present in the card's name
                 tokens (title+all aliases → cross-lingual recall)
      focus    = the BEST single alias's fraction-of-terms present in the query
                 (so a card whose SHORT canonical alias == the query, e.g. "Roche"
                 for Hoffmann-La Roche, scores 1.0 and qualifies — v1's title-only
                 focus wrongly excluded long canonical titles).
    Matching is EXACT token equality (NOT prefix-stem): entity names are not
    morphological variants — "Swisscom" must NOT match Swiss Air's "Swiss" card.
    _content_terms still decompounds German, so compound recall is preserved.
    Disambiguation among same-name entities is by PROMINENCE at the call site.
    `host_label` (AW_HOST_QUERY_CARD, task 2050): a host-shaped query's label; equal to the
    card URL's host label means the query names this card (1.0). None = unchanged."""
    if host_label and host_label == url_host_label(card_url):
        return 1.0
    if not q_content:
        return 0.0
    all_terms = _content_terms((title or "") + " | " + (names or ""), lang)
    if not all_terms:
        return 0.0
    cov = sum(1 for q in q_content if q in all_terms) / len(q_content)
    best_focus = 0.0
    for a in _card_aliases(title, names):
        at = _content_terms(a, lang)
        if not at:
            continue
        f = sum(1 for t in at if t in q_content) / len(at)
        if f > best_focus:
            best_focus = f
    return cov * best_focus


def _query_terms(q: str) -> set[str]:
    """Distinctive query tokens (>=3 chars, apostrophes kept then stripped) for
    the exact-match feature. 'Tim O'Hear' -> {tim, ohear}."""
    out = set()
    for t in _WORD.findall(q.lower()):
        t2 = t.replace("'", "").replace("’", "")
        if len(t2) >= 3:
            out.add(t2)
    return out


# Version-aware variants (task 665). Used only when `retrieval._VERSION_MATCH` is on and the
# query carries a dotted version token; see the AW_VERSION_MATCH block in `retrieval.py`.
def _title_versions_match(q_versions, title: str) -> bool:
    tv = schema.version_tokens(title)
    return any(schema.version_match(q, t) for q in q_versions for t in tv)


def _title_centrality_v(q_content: set[str], q_versions, title: str,
                        lang: str | None = None) -> float:
    """`_title_centrality` with the query's and the title's dotted version
    tokens counted as content terms (matched by schema.version_match)."""
    if not q_versions:
        return _title_centrality(q_content, title, lang)
    t_terms = _content_terms(title, lang)
    t_vers = schema.version_tokens(title)
    n_q = len(q_content) + len(q_versions)
    n_t = len(t_terms) + len(t_vers)
    if not n_q or not n_t:
        return 0.0
    mq = (sum(1 for q in q_content if any(_stem_match(q, t) for t in t_terms))
          + sum(1 for q in q_versions
                if any(schema.version_match(q, t) for t in t_vers)))
    mt = (sum(1 for t in t_terms if any(_stem_match(t, q) for q in q_content))
          + sum(1 for t in t_vers
                if any(schema.version_match(q, t) for q in q_versions)))
    return (mq / n_q) * (mt / n_t)


def _exact_v(q_terms: set[str], q_versions, pl: dict) -> float:
    """`_exact` with the query's version tokens as extra terms, matched
    against the title+snippet's version tokens (not by substring)."""
    if not q_versions:
        return _exact(q_terms, pl)
    n = len(q_terms) + len(q_versions)
    base = _exact(q_terms, pl) * len(q_terms) if q_terms else 0.0
    hv = schema.version_tokens(
        str(pl.get("title", "")) + " " + str(pl.get("snippet", "")))
    hits = sum(1 for q in q_versions
               if any(schema.version_match(q, t) for t in hv))
    return (base + hits) / n
