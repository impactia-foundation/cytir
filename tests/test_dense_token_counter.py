"""The embed-time token counter is one home, and it is the FAST backend.

WHY THIS EXISTS: on 2026-08-08 an `it` Vast rental died with "composed dense
text has 2001 tokens; refusing truncation".  A census of all 19,227,122 rows
found 702 overflows, **every one at exactly 2001 and none at 2002+** — an
off-by-one, not a size distribution.  The materializer counted with the SLOW
`XLMRobertaTokenizer` (sentencepiece, strips trailing whitespace) because
`use_fast` was not pinned; the worker counted with the FAST
`XLMRobertaTokenizerFast`.  For any text ending in whitespace, slow = fast - 1.

Two classes of test live here:

* **structural** — run anywhere, no model needed: the counter must be one
  module, `use_fast=True` must be pinned in the source, and the probe table must
  be internally consistent and discriminating.
* **real-tokenizer** — need `transformers` plus the cached model, so they use
  the established `importorskip` pattern from `test_chunk.py`.  These are the
  ones that would have caught the incident; the worker also asserts the same
  mirror at model-load time on the deployed object, which is stronger than any
  test double.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from cytir.chunk import token_counter as counter

HERE = Path(__file__).resolve().parent
# Moved to the package (task 677, task 643 S5); the old pipeline module is a shim.
SOURCE = HERE.parent / "cytir" / "chunk" / "token_counter.py"
MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
REVISION = "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"


# --------------------------------------------------------------------------
# structural — no model required
# --------------------------------------------------------------------------

def test_use_fast_is_pinned_in_the_only_load_site():
    """`use_fast=True` must be pinned, and there must be exactly one load site.

    An unpinned `from_pretrained` lets the installed package set decide the
    backend, which is the whole incident.  Asserted against the source because
    the failure is a *missing* argument: no runtime call can observe an argument
    that was never written.
    """
    import ast
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    loads = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "from_pretrained"]
    assert len(loads) == 1, f"expected exactly one tokenizer load site, found {len(loads)}"
    pinned = [kw for kw in loads[0].keywords
              if kw.arg == "use_fast" and getattr(kw.value, "value", None) is True]
    assert pinned, "the single tokenizer load site does not pin use_fast=True"


def test_no_other_module_counts_tokens_itself():
    """No second `len(<tok>(...)["input_ids"])` anywhere in the private pipeline tree.

    Mirror contracts are sacred: a second counting call site is a forked
    invariant, identical today and divergent on the first edit.  This is the
    grep an auditor would otherwise have to remember to run.
    """
    offenders = []
    pattern = re.compile(r'len\(\s*[A-Za-z_][\w.]*\(\s*[^)]*add_special_tokens')
    private = HERE.parent / "pipeline"
    scanned = sorted(private.glob("*.py")) if private.is_dir() else []
    scanned += sorted(HERE.glob("*.py")) + sorted((HERE.parent / "cytir").rglob("*.py"))
    for path in scanned:
        if path == SOURCE or path.name == Path(__file__).name:
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert not offenders, (
        f"{offenders} count tokens locally instead of calling "
        f"dense_token_counter.token_len")


def test_probe_table_is_discriminating_and_off_by_exactly_one():
    """Every probe must actually separate the two backends, by exactly one token.

    A probe on which the backends agree proves nothing, and would silently
    weaken `assert_mirrors` into a no-op.  "Exactly one" is the measured
    signature that explains the 2001-and-never-2002 distribution.
    """
    assert counter.WHITESPACE_DIVERGENCE_PROBES
    for text, fast, slow in counter.WHITESPACE_DIVERGENCE_PROBES:
        assert text != text.rstrip(), f"{text!r} does not end in whitespace"
        assert fast - slow == 1, f"{text!r}: fast {fast} - slow {slow} != 1"


def test_identity_names_the_class_not_only_the_model_and_revision():
    """The field whose absence let the divergence ship unnoticed.

    All 256 shipped manifests recorded the right model, the right revision and
    `tokenizer_status: "ok"` — and were produced by the wrong backend anyway.
    The class was the variable and nothing wrote it down.
    """

    class SlowLookalike:
        is_fast = False

    identity = counter.identity_of(SlowLookalike(), MODEL, REVISION)
    assert identity == {
        "tokenizer_model": MODEL,
        "tokenizer_revision": REVISION,
        "tokenizer_class": "SlowLookalike",
        "tokenizer_is_fast": False,
    }


def test_identity_describes_the_tokenizer_that_did_the_counting():
    """`identity_of` must not resolve a fresh tokenizer behind the caller's back.

    An identity record describing a different object than the one that produced
    the numbers is worse than no record: it would have made the incident's
    manifests look *more* trustworthy, not less.
    """

    class Marker:
        is_fast = True

    assert counter.identity_of(Marker(), MODEL, REVISION)["tokenizer_class"] == "Marker"


# --------------------------------------------------------------------------
# real tokenizer — the regression pins
# --------------------------------------------------------------------------

def test_resolved_counter_is_the_fast_backend():
    pytest.importorskip("transformers")
    tokenizer = counter.resolve_tokenizer(MODEL, REVISION)
    assert tokenizer.is_fast is True
    assert type(tokenizer).__name__ == "XLMRobertaTokenizerFast"


def test_counts_match_the_recorded_fast_table():
    """If the pinned revision or the tokenizers library moves, fail loudly.

    The probe counts are measured constants, not conveniences: `assert_mirrors`
    compares against them, so a silent drift here would make every later
    "counters agree" claim vacuous.
    """
    pytest.importorskip("transformers")
    for text, fast, _slow in counter.WHITESPACE_DIVERGENCE_PROBES:
        assert counter.token_len(text, MODEL, REVISION) == fast, repr(text)


def test_slow_and_fast_cannot_silently_converge():
    """The regression pin: the two backends still disagree, and by one.

    If a future `transformers`/`tokenizers` release makes the backends agree,
    this fails — which is the correct outcome.  The probe table would then be
    non-discriminating and must be re-derived before anyone trusts
    `assert_mirrors` again.  Requires sentencepiece, which the worker image
    deliberately does not ship (that is itself part of the defence: with no slow
    backend installed, `AutoTokenizer` cannot resolve one).
    """
    pytest.importorskip("transformers")
    pytest.importorskip("sentencepiece")
    from transformers import AutoTokenizer
    slow = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=True, use_fast=False)
    assert slow.is_fast is False
    for text, fast, expected_slow in counter.WHITESPACE_DIVERGENCE_PROBES:
        measured = len(slow(text, add_special_tokens=True)["input_ids"])
        assert measured == expected_slow, repr(text)
        assert counter.token_len(text, MODEL, REVISION) == fast


def test_assert_mirrors_accepts_the_matching_backend():
    pytest.importorskip("transformers")
    same = counter.resolve_tokenizer(MODEL, REVISION)
    evidence = counter.assert_mirrors(same, MODEL, REVISION, where="test")
    assert evidence["status"] == "TOKEN_COUNTER_MIRROR_VERIFIED"
    assert evidence["tokenizer_is_fast"] is True


def test_assert_mirrors_rejects_a_divergent_counter():
    """The assertion must actually fire — an untested guard is not a guard."""
    pytest.importorskip("transformers")

    class OffByOne:
        """Stands in for the slow backend: undercounts whitespace-terminated text."""

        def __call__(self, text, add_special_tokens=True):
            real = counter.token_len(text, MODEL, REVISION)
            return {"input_ids": [0] * (real - 1)}

    with pytest.raises(counter.TokenCounterError, match="mirror broken"):
        counter.assert_mirrors(OffByOne(), MODEL, REVISION, where="test")
