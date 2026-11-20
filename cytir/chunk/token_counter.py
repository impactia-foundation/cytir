#!/usr/bin/env python3
"""THE embed-time token counter — one home for "how long will the embedder see this?".

Why this module exists
----------------------
This is a **mirror contract** (engineering doctrine 6): the chunker/materializer
counts tokens at *write* time to decide where to cut, and the embed worker counts
tokens at *embed* time to decide whether to refuse truncation. Those two numbers
must be the same number. They were not.

**The incident (task 389, 2026-08-08).** Two `it` embed contracts died with
``composed dense text has 2001 tokens; refusing truncation`` against a 2000-token
ceiling. A census of the whole 19,227,122-row corpus found **702 overflows, every
single one at exactly 2001, zero at 2002 or above** — the fingerprint of an
off-by-one, not of a size distribution. The mechanism:

* the worker counted with ``SentenceTransformer.tokenizer`` → ``XLMRobertaTokenizerFast``
  (the Rust ``tokenizers`` backend);
* the materializer counted with ``AutoTokenizer.from_pretrained(...)`` **without
  pinning ``use_fast``**, which in the campaign's process resolved to the *slow*
  ``XLMRobertaTokenizer`` (the sentencepiece backend).

Same model id. Same pinned revision. Different answer, because sentencepiece's
normalizer **strips trailing whitespace** while the ``tokenizers`` pipeline emits
a lone ``▁`` for it. For any text ending in whitespace, ``slow = fast − 1``. The
splitter's bisect therefore packed one extra empty unit, stopped exactly on the
slow counter's limit, and handed the worker a string one token too long.

Two things follow, and both live here rather than in a comment:

1. ``use_fast=True`` is **pinned**, so the backend can never again be decided by
   whatever happens to be installed in a process (doctrine 2: config is data).
2. ``WHITESPACE_DIVERGENCE_PROBES`` is the tracked probe set that *exposes* the
   divergence, so both sides and the tests assert against one list instead of
   three copies of a remembered example (doctrine 1/4).

Every caller imports from here. A second ``len(tok(text)["input_ids"])`` anywhere
in the tree is a defect by definition, whatever its comment says.
"""
from __future__ import annotations

from typing import Any


# The counter is identified by (model, revision) exactly as every artifact that
# records it is: the chunk policy names the model (`token_hard_max.tokenizer_model`)
# and the materialization config names the immutable snapshot
# (`dense_tokenizer.revision`). Neither is defaulted here — a counter that guesses
# its own identity is the thing this module exists to stop.
_CACHE: dict[tuple[str, str | None], Any] = {}

# (text, fast_count, slow_count) — strings whose count differs between the fast
# (Rust `tokenizers`) and slow (sentencepiece) backends. The FAST count is the
# authoritative one, because it is what the embedder actually consumes.
#
# MEASURED 2026-08-08 at Snowflake/snowflake-arctic-embed-l-v2.0 revision
# ac6544c8a46e00af67e330e85a9028c66b8cfd9a, both backends loaded side by side in
# the campaign's own worker image with sentencepiece added. Not remembered, not
# inherited from a summary.
#
# Every entry ends in whitespace, because trailing whitespace IS the divergence:
# `Einzelnachweise` without its newline counts 5 on both backends and would prove
# nothing. Every entry diverges by exactly 1, which is why the corpus overflow
# distribution was a spike at 2001 with nothing above it.
WHITESPACE_DIVERGENCE_PROBES: tuple[tuple[str, int, int], ...] = (
    ("Einzelnachweise\n", 6, 5),
    ("Einzelnachweise\n\n", 6, 5),
    ("Voci correlate\n", 7, 6),
    ("Notes et références\n", 8, 7),
)


class TokenCounterError(RuntimeError):
    """The embed-time token counter is unavailable or is not the fast backend."""


def resolve_tokenizer(model: str, revision: str | None):
    """The one tokenizer object every AW token count is taken with.

    ``use_fast=True`` is not a preference, it is the fix for the 702-row
    off-by-one described in this module's docstring. ``is_fast`` is then asserted
    rather than trusted: ``use_fast=True`` silently falls back to the slow class
    when a fast implementation is unavailable, and a silent fallback to the exact
    backend that caused the incident is the one degradation that must never pass
    unnoticed (doctrine 3: fail loud, never silently substitute a different
    thing).
    """
    key = (model, revision)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, trust_remote_code=True, use_fast=True)
    except Exception as exc:  # noqa: BLE001 — import OR download/cache failure
        raise TokenCounterError(
            f"cannot load embed-time tokenizer {model!r} at revision {revision!r}: "
            f"{type(exc).__name__}: {exc}") from exc
    if not getattr(tokenizer, "is_fast", False):
        raise TokenCounterError(
            f"embed-time tokenizer {model!r} at revision {revision!r} resolved to "
            f"the SLOW backend ({type(tokenizer).__name__}) despite use_fast=True. "
            f"The slow sentencepiece backend strips trailing whitespace and "
            f"undercounts by one token, which is exactly the defect that shipped "
            f"702 over-length rows in the task389 corpus. Refusing to count.")
    _CACHE[key] = tokenizer
    return tokenizer


def count(tokenizer: Any, text: str) -> int:
    """THE counting call. Every token length in AW comes through this line.

    Split out from ``token_len`` so that a caller which already holds a resolved
    tokenizer (``cytir.chunk.chunker``, which owns the policy-bound one and must stay
    stubbable in tests) still counts with the *same implementation* rather than
    writing its own ``len(tok(...)["input_ids"])``. Resolution and counting are
    two separable decisions; only the counting one is the mirror contract.

    Special tokens are counted because the ceiling being compared against is the
    model's ``max_seq``, which includes them.
    """
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def count_batch(tokenizer: Any, texts: list[str]) -> list[int]:
    """Batched ``count``. Same implementation, one tokenizer call."""
    if not texts:
        return []
    return [len(ids) for ids in
            tokenizer(texts, add_special_tokens=True)["input_ids"]]


def token_len(text: str, model: str, revision: str | None) -> int:
    """Tokens the embedder will see for ``text``, at the named model/revision."""
    return count(resolve_tokenizer(model, revision), text)


def token_lens(texts: list[str], model: str, revision: str | None) -> list[int]:
    """Batched ``token_len``. Same counter, same answer, one call."""
    if not texts:
        return []
    return count_batch(resolve_tokenizer(model, revision), texts)


def identity_of(tokenizer: Any, model: str, revision: str | None) -> dict[str, Any]:
    """What an artifact must record so an auditor can tell which backend ran.

    The materialization manifest used to record only ``tokenizer_model`` and
    ``tokenizer_revision``. Both were *correct* during the incident — the class
    was the variable, and nothing wrote it down. A run therefore could not notice
    it had measured with the wrong backend, and an auditor could not tell from
    the artifact. That gap is closed here.

    Takes the tokenizer the caller actually counted with, rather than resolving a
    fresh one: an identity record that describes a *different* object than the
    one that produced the numbers would be worse than none at all.
    """
    return {
        "tokenizer_model": model,
        "tokenizer_revision": revision,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
    }


def counter_identity(model: str, revision: str | None) -> dict[str, Any]:
    """``identity_of`` for the counter this module resolves itself."""
    return identity_of(resolve_tokenizer(model, revision), model, revision)


def assert_mirrors(other: Any, model: str, revision: str | None,
                   where: str) -> dict[str, Any]:
    """Assert ``other`` counts identically to this module's counter.

    Called by the embed worker against the tokenizer bound to the loaded
    ``SentenceTransformer``, at model-load time, before a single batch is spent.
    This is the assertion that would have caught the incident on the first run
    instead of 702 rows later, and it runs against the *deployed* object rather
    than a test double (verification doctrine 1).

    Returns the evidence to put in the receipt; raises ``TokenCounterError`` on
    any disagreement.
    """
    probes = [text for text, _fast, _slow in WHITESPACE_DIVERGENCE_PROBES]
    mine = token_lens(probes, model, revision)
    theirs = [count(other, text) for text in probes]
    disagreements = [
        {"probe": probe, "shared_counter": a, "other": b}
        for probe, a, b in zip(probes, mine, theirs) if a != b
    ]
    if disagreements:
        raise TokenCounterError(
            f"token counter mirror broken at {where}: {type(other).__name__} "
            f"disagrees with the shared counter on {len(disagreements)} of "
            f"{len(probes)} whitespace probes: {disagreements}. Write-time and "
            f"embed-time counts must be the same number; a divergence here means "
            f"chunks were cut against a ceiling the embedder does not honour.")
    expected = [fast for _text, fast, _slow in WHITESPACE_DIVERGENCE_PROBES]
    if mine != expected:
        raise TokenCounterError(
            f"the shared counter itself drifted at {where}: probe counts {mine} "
            f"!= the recorded fast-backend counts {expected} for model {model!r} "
            f"at revision {revision!r}. Either the pinned revision moved or the "
            f"tokenizers library changed its whitespace handling; re-derive the "
            f"probe table before trusting any count.")
    return {
        "status": "TOKEN_COUNTER_MIRROR_VERIFIED",
        "where": where,
        "probes": len(probes),
        "other_class": type(other).__name__,
        **counter_identity(model, revision),
    }
