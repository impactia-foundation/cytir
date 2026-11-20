"""Tests for `cytir/chunk/chunker.py` — the chunk@2 structure-aware chunker (task 340).

The properties asserted here are the ones whose regression would be SILENT and
expensive (verification doctrine 1 — unit tests are for exactly that class):

  1. header-cut          — sections become chunks; a code fence is not a header
  2. <40w merge          — tiny sections fold into a neighbour, no orphan stubs
  3. hard-split          — no emitted chunk exceeds the guaranteed ceiling, even
                           when ONE block (a table / code fence) is oversized.
                           237 §4.1's explicit mandate; the 237b prototype did
                           NOT do this (its own _counts.json shows max 2860w)
  4. determinism         — byte-stable across reruns and across a fresh import
  5. overlap 0           — no word is emitted twice across a section boundary
  6. id distinctness     — GAP A33 (tasks 318/319/325): a multi-chunk ANCHOR-LESS
                           document must mint one distinct point_id per chunk.
                           anthropic.com/constitution collapsed 218 chunks -> 1.
  7. dispatch fail-loud  — AW_CHUNKER typo raises instead of silently running
                           the legacy epoch

Run: python3 -m pytest tests/test_chunk.py -q
"""
import importlib
import importlib.util
import hashlib
import json
import os
import pathlib
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cytir.chunk import chunker as awchunk            # noqa: E402
import cytir.core.ids as ident     # noqa: E402


_ADJACENT_POLICY = (
    _HERE.parent / "data" /
    "chunk-policy-struct-md-768w-o0-adjacent-sections.v1.json"
)
_ADJACENT_TAIL_896_POLICY = (
    _HERE.parent / "data" /
    "chunk-policy-struct-md-768w-o0-adjacent-sections-tail-896.v1.json"
)
_WHOLE_BALANCED_POLICY = (
    _HERE.parent / "data" /
    "chunk-policy-struct-md-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json"
)


@pytest.fixture
def adjacent_section_policy():
    """Select the experimental artifact only for the test that names it.

    The tracked default remains epoch 2.  This fixture deliberately restores
    the prior policy so one experiment cannot leak its epoch into another
    test's identity or default-profile assertion.
    """
    previous = awchunk.active_policy()
    awchunk.select_policy(_ADJACENT_POLICY)
    try:
        yield awchunk.active_policy()
    finally:
        awchunk.select_policy(previous.path, allow_env_overrides=True)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _words(n, seed="lorem"):
    return " ".join(f"{seed}{i}" for i in range(n))


DOC = f"""# Trainer

{_words(60, 'lead')}

## Parameters

{_words(120, 'param')}

### model

tiny

## Usage

{_words(90, 'usage')}
"""


# --------------------------------------------------------------------------
# 1. header cut
# --------------------------------------------------------------------------

def test_header_cut_produces_one_chunk_per_section():
    cs = awchunk.chunk_markdown(DOC)
    paths = [c.header_path for c in cs]
    assert "Trainer" in paths                       # page lead under its h1
    assert "Trainer > Parameters" in paths
    assert "Trainer > Usage" in paths
    # every chunk carries its breadcrumb as the first line (section topic
    # travels with the fragment — 237 §4.4)
    for c in cs:
        if c.header_path:
            assert c.text.split("\n", 1)[0] == c.header_path


def test_headings_inside_a_code_fence_are_not_headers():
    md = "# Real\n\n" + _words(50) + "\n\n```python\n# not a header\nx = 1\n```\n"
    cs = awchunk.chunk_markdown(md)
    assert {c.header_path for c in cs} == {"Real"}
    assert "```python" in "\n".join(c.text for c in cs)


def test_no_headings_at_all_still_chunks_at_target():
    md = "\n\n".join(_words(200, f"p{i}_") for i in range(10))   # 2000 words
    cs = awchunk.chunk_markdown(md)
    assert len(cs) >= 2
    assert all(c.header_path == "" for c in cs)
    assert all(c.n_words <= awchunk.HARD_MAX_WORDS for c in cs)


# --------------------------------------------------------------------------
# 1b. Experimental adjacent-section packing (epoch 3, explicitly selected)
# --------------------------------------------------------------------------

def test_adjacent_section_policy_packs_several_sections_with_headings_inline(
        adjacent_section_policy, monkeypatch):
    """The policy fixes W2's observed one-small-section-per-vector shape."""
    monkeypatch.setattr(awchunk, "_token_len", lambda text: len(text.split()))
    assert adjacent_section_policy.epoch == 3
    assert adjacent_section_policy.section_packing_policy == "adjacent-greedy-balanced-tail"
    assert awchunk.config()["policy"]["sha256"] == adjacent_section_policy.sha256
    md = (
        "# Overview\n\n" + _words(240, "overview_") +
        "\n\n## Background\n\n" + _words(240, "background_") +
        "\n\n## Findings\n\n" + _words(240, "findings_")
    )
    chunks = awchunk.chunk_markdown(md)
    assert len(chunks) == 1
    text = chunks[0].text
    assert "Overview" in text
    assert "Overview > Background" in text
    assert "Overview > Findings" in text
    assert text.index("overview_0") < text.index("background_0") < text.index("findings_0")
    # A packed point retains the whole hierarchy for canonical/BM25 but only
    # each atom's local heading for dense retrieval.
    dense = awchunk.dense_text(chunks[0])
    assert "Overview > Background" not in dense
    assert "Overview\n" in dense
    assert "Background\n" in dense
    assert "Findings\n" in dense
    assert dense.index("overview_0") < dense.index("background_0") < dense.index("findings_0")
    assert chunks[0].chunk_epoch == 3
    assert chunks[0].chunk_rule == "struct-md-768w-o0-adjacent-sections@3"


def test_adjacent_section_policy_balances_safe_slightly_over_target_tail(
        adjacent_section_policy, monkeypatch):
    monkeypatch.setattr(awchunk, "_token_len", lambda text: len(text.split()))
    md = "# First\n\n" + _words(750, "first_") + "\n\n## Tail\n\n" + _words(20, "tail_")
    chunks = awchunk.chunk_markdown(md)
    assert len(chunks) == 1
    assert awchunk.TARGET_WORDS < chunks[0].n_words <= awchunk.HARD_MAX_WORDS
    assert "First > Tail" in chunks[0].text
    assert "tail_19" in chunks[0].text


def test_adjacent_tail_max_is_a_versioned_candidate_not_a_new_target(monkeypatch):
    previous = awchunk.active_policy()
    awchunk.select_policy(_ADJACENT_TAIL_896_POLICY)
    monkeypatch.setattr(awchunk, "_token_len", lambda text: len(text.split()))
    try:
        assert awchunk.TARGET_WORDS == 768
        assert awchunk.HARD_MAX_WORDS == 896
        md = "# First\n\n" + _words(750, "first_") + "\n\n## Tail\n\n" + _words(100, "tail_")
        assert len(awchunk.chunk_markdown(md)) == 1
    finally:
        awchunk.select_policy(previous.path, allow_env_overrides=True)


def test_adjacent_section_policy_does_not_balance_when_exact_tokens_overflow(
        adjacent_section_policy, monkeypatch):
    def token_len(text):
        # Both individual sections fit; only their proposed balanced merge is
        # over the immutable 2,000-token ceiling.
        return awchunk.MAX_CHUNK_TOKENS + 1 if len(text.split()) > awchunk.TARGET_WORDS else 1

    monkeypatch.setattr(awchunk, "_token_len", token_len)
    md = "# First\n\n" + _words(750, "first_") + "\n\n## Tail\n\n" + _words(20, "tail_")
    chunks = awchunk.chunk_markdown(md)
    assert len(chunks) == 2
    assert ["first_0" in c.text for c in chunks] == [True, False]
    assert ["tail_0" in c.text for c in chunks] == [False, True]


def test_whole_balanced_policy_falls_back_when_dense_channel_overflows(monkeypatch):
    previous = awchunk.active_policy()
    awchunk.select_policy(_WHOLE_BALANCED_POLICY)
    try:
        monkeypatch.setattr(
            awchunk, "_token_len",
            lambda text: awchunk.MAX_CHUNK_TOKENS + 1
            if "second_0" in text and "first_0" in text else len(text.split()))
        md = ("# First\n\n" + _words(300, "first_") +
              "\n\n## Second\n\n" + _words(300, "second_"))
        chunks = awchunk.chunk_markdown(md)
        assert len(chunks) == 2
        assert all(awchunk._token_len(awchunk.dense_text(chunk)) <=
                   awchunk.MAX_CHUNK_TOKENS for chunk in chunks)
    finally:
        awchunk.select_policy(previous.path, allow_env_overrides=True)


def test_adjacent_section_policy_conserves_source_words_in_order(
        adjacent_section_policy, monkeypatch):
    monkeypatch.setattr(awchunk, "_token_len", lambda text: len(text.split()))
    bodies = [_words(320, f"section{i}_") for i in range(3)]
    md = "\n\n".join(
        f"## Heading {i}\n\n{body}" for i, body in enumerate(bodies))
    emitted = " ".join(c.body for c in awchunk.chunk_markdown(md))
    source_words = " ".join(bodies).split()
    positions = [emitted.index(word) for word in source_words]
    assert positions == sorted(positions)
    assert all(emitted.split().count(word) == 1 for word in source_words)


def test_adjacent_section_policy_keeps_exact_token_split_contract(
        adjacent_section_policy, monkeypatch):
    """A packed body still takes the existing line-then-word exact splitter."""
    monkeypatch.setattr(awchunk, "_token_len", lambda text: 3 * len(text.split()))
    source = _words(780, "highratio_")
    chunks = awchunk.chunk_markdown("# Dense\n\n" + source)
    assert len(chunks) >= 2
    assert all(awchunk._token_len(c.text) <= awchunk.MAX_CHUNK_TOKENS for c in chunks)
    assert [w for c in chunks for w in c.body.split() if w.startswith("highratio_")] == source.split()


def test_bisect_split_tokenizer_unavailable_conserves_input(monkeypatch):
    monkeypatch.setattr(awchunk, "_token_len", lambda text: None)
    assert awchunk._bisect_split(["one", "two", "three"], " ", "prefix\n") == [
        "one two three"]


def test_adjacent_section_policy_splits_indivisible_dense_source_at_codepoints(
        adjacent_section_policy, monkeypatch):
    """A minified source must be retained or fail, never emitted over limit."""
    monkeypatch.setattr(awchunk, "_token_len", lambda text: len(text))
    source = "x" * (awchunk.MAX_CHUNK_TOKENS + 40)
    chunks = awchunk.chunk_markdown("# Dense\n\n" + source)
    assert len(chunks) >= 2
    assert "".join(c.body.split("\n\n")[-1] for c in chunks) == source
    assert all(awchunk._token_len(awchunk.dense_text(c)) <= awchunk.MAX_CHUNK_TOKENS
               for c in chunks)


# --------------------------------------------------------------------------
# 2. <40w merge
# --------------------------------------------------------------------------

def test_tiny_section_merges_into_neighbour():
    cs = awchunk.chunk_markdown(DOC)
    # "### model" has a 1-word body -> must not be its own chunk
    assert all(c.header_path != "Trainer > Parameters > model" for c in cs)
    assert all(c.n_words >= awchunk.MIN_CHUNK_WORDS for c in cs)
    # its title survives inline so it stays searchable
    joined = "\n".join(c.text for c in cs)
    assert "model" in joined and "tiny" in joined


def test_leading_tiny_section_carries_forward_with_its_heading(word_only_chunking_allowed):
    md = "# Alpha\n\nshort body here\n\n## B\n\n" + _words(100, 'b')
    cs = awchunk.chunk_markdown(md)
    assert len(cs) == 1
    assert "short body here" in cs[0].text
    # the merged-away heading must survive as BODY text, not only as a breadcrumb
    assert "Alpha:" in cs[0].body


# --------------------------------------------------------------------------
# 2b. F6 -- overridable floor for card/reference-shaped routes (task 340/360)
# --------------------------------------------------------------------------

_SHORT_ATOMIC_DOC = "just a short ten word line of plain undecorated body text"


def test_short_whole_doc_is_dropped_by_default_floor(word_only_chunking_allowed):
    """Regression guard for the DEFAULT behaviour (generic prose path,
    unchanged): a whole 'document' shorter than MIN_CHUNK_WORDS chunks to
    NOTHING today. Correct for a genuine prose stub -- the trap (F6) is
    applying this same default to a route that is ALREADY atomic (LCSH/
    ch_courts/ai_autodoc_reference: median 10-14w, one row = one whole
    record)."""
    assert awchunk.word_count(_SHORT_ATOMIC_DOC) < awchunk.MIN_CHUNK_WORDS
    assert awchunk.chunk_markdown(_SHORT_ATOMIC_DOC) == []


def test_min_words_override_keeps_the_short_atomic_record():
    """F6 fix: a caller that knows the source is a card-shaped route passes
    min_words=1 and gets the record back as ONE chunk instead of it silently
    vanishing."""
    cs = awchunk.chunk_markdown(_SHORT_ATOMIC_DOC, min_words=1)
    assert len(cs) == 1
    assert cs[0].text.split() == _SHORT_ATOMIC_DOC.split()


def test_min_words_override_is_call_scoped_not_global(word_only_chunking_allowed):
    """The override must not leak into the module default -- a subsequent
    call with no override reproduces the default-floor behaviour exactly."""
    _ = awchunk.chunk_markdown(_SHORT_ATOMIC_DOC, min_words=1)
    assert awchunk.chunk_markdown(_SHORT_ATOMIC_DOC) == []
    assert awchunk.MIN_CHUNK_WORDS == 15


def test_min_words_override_via_chunk_texts(word_only_chunking_allowed):
    assert awchunk.chunk_texts(_SHORT_ATOMIC_DOC) == []
    assert awchunk.chunk_texts(_SHORT_ATOMIC_DOC, min_words=1) == [_SHORT_ATOMIC_DOC]


# --------------------------------------------------------------------------
# 3. hard split
# --------------------------------------------------------------------------

def test_oversized_section_is_hard_split_under_the_ceiling(word_only_chunking_allowed):
    md = "# Big\n\n" + "\n\n".join(_words(300, f"q{i}_") for i in range(10))
    cs = awchunk.chunk_markdown(md)
    assert len(cs) >= 4
    assert max(c.n_words for c in cs) <= awchunk.HARD_MAX_WORDS
    assert all(c.unit == "section_part" for c in cs)
    assert {c.n_parts for c in cs} == {len(cs)}
    assert [c.part for c in cs] == list(range(len(cs)))


def test_single_oversized_block_is_split_internally():
    """A 3000-word table arrives as ONE block. 237 §4.1: paragraph boundaries
    cannot be trusted to respect the ceiling."""
    rows = "\n".join("| " + _words(30, f"r{i}_") + " |" for i in range(100))
    md = "# T\n\n" + rows
    cs = awchunk.chunk_markdown(md)
    assert max(c.n_words for c in cs) <= awchunk.HARD_MAX_WORDS
    assert len(cs) >= 4
    # split at ROW boundaries, not mid-row
    for c in cs:
        for line in c.body.split("\n"):
            if line.startswith("|"):
                assert line.endswith("|")


def test_single_oversized_line_is_word_split():
    md = "# X\n\n" + _words(4000, "w")
    cs = awchunk.chunk_markdown(md)
    assert max(c.n_words for c in cs) <= awchunk.HARD_MAX_WORDS
    assert sum(c.n_words for c in cs) >= 4000       # nothing dropped


def test_no_chunk_exceeds_the_ceiling_on_any_fixture():
    for md in (DOC, "# T\n\n" + _words(5000), "\n\n".join(_words(77) for _ in range(30))):
        for c in awchunk.chunk_markdown(md):
            assert c.n_words <= awchunk.HARD_MAX_WORDS, c.n_words


# --------------------------------------------------------------------------
# 3b. F1 -- token-aware hard-split (task 340/360). 340 §2.3 measured a
# word/token ratio of 2.2-7.3 on the real arctic tokenizer (code/tables at the
# high end), so a word-only ceiling let 4.4% of chunk@2 passages exceed TEI's
# configured max_seq=2048. These tests need the real tokenizer -- skip (not
# fail) where `transformers` isn't installed (this module's algorithmic tests
# above must never require it; see test_tokenizer_unavailable_* below for the
# degrade-without-it contract).
# --------------------------------------------------------------------------

def _high_ratio_words(n):
    """Hyphen/underscore-chained pseudo-identifiers: BPE/sentencepiece
    subword-splits these far more than dictionary words, pushing tok/word
    well above prose's ~2.2 without needing a huge word count."""
    return " ".join(f"very_long_identifier_name_{i}-with-many-subword-pieces_{i}"
                     for i in range(n))


@pytest.fixture
def pinned_tokenizer():
    """Measure with the pinned embedder snapshot (the one CI and `cyt embed models --tokenizer` fetch),
    not the mutable `main` ref, which a cache holding only the pin does not carry."""
    from cytir.core import model_pin  # noqa: PLC0415
    before = awchunk.TOKENIZER_REVISION
    awchunk.select_tokenizer_revision(model_pin.embed_model_revision())
    yield
    awchunk.select_tokenizer_revision(before)


def test_token_ceiling_never_exceeded_on_pathological_content(pinned_tokenizer):
    pytest.importorskip("transformers")
    assert awchunk.tokenizer_status() == "ok", (
        "tokenizer failed to load -- check the stderr warning")
    md = "# Big\n\n" + _high_ratio_words(700)
    cs = awchunk.chunk_markdown(md)
    assert len(cs) >= 2, "a single oversized-in-tokens block must be split"
    for c in cs:
        tlen = awchunk._token_len(c.text)
        assert tlen is not None and tlen <= awchunk.MAX_CHUNK_TOKENS, tlen


def test_token_split_drops_nothing(pinned_tokenizer):
    pytest.importorskip("transformers")
    assert awchunk.tokenizer_status() == "ok"
    words = _high_ratio_words(600)
    md = "# T\n\n" + words
    cs = awchunk.chunk_markdown(md)
    joined = " ".join(c.body for c in cs).split()
    assert len(joined) == len(words.split())


def test_token_split_keeps_breadcrumb_on_every_piece(pinned_tokenizer):
    pytest.importorskip("transformers")
    assert awchunk.tokenizer_status() == "ok"
    md = "# Section\n\n## Deep\n\n" + _high_ratio_words(600)
    cs = awchunk.chunk_markdown(md)
    assert len(cs) >= 2
    for c in cs:
        assert c.text.split("\n", 1)[0] == c.header_path
    # n_parts/part number the TRUE post-token-split pieces, densely
    assert {c.n_parts for c in cs} == {len(cs)}
    assert [c.part for c in cs] == list(range(len(cs)))


def test_ordinary_prose_chunk_is_unaffected_by_token_check():
    """Regression guard: normal prose well under both ceilings chunks
    IDENTICALLY whether or not the tokenizer is available -- F1 must not
    perturb the common case."""
    before = [c.text for c in awchunk.chunk_markdown(DOC)]
    pytest.importorskip("transformers")
    if awchunk.tokenizer_status() != "ok":
        pytest.skip("tokenizer unavailable in this env")
    after = [c.text for c in awchunk.chunk_markdown(DOC)]
    assert before == after


def test_tokenizer_unavailable_degrades_loudly_not_silently(monkeypatch, capsys):
    """When the tokenizer truly can't load, chunk_markdown must still run
    (word-bounded only, pre-F1 behaviour) rather than crash -- but MUST
    announce the degradation (doctrine 3) and `tokenizer_status()` must
    report it, so a wave manifest can catch a run that silently shipped
    without the F1 guarantee."""
    monkeypatch.setitem(awchunk._TOKENIZER_STATE, "status", None)
    # Both caches are keyed (model, revision): empty them (restored after) so a tokenizer cached by an
    # earlier test cannot answer past the blocked import (task 2289: CI now has a real one).
    from cytir.chunk import token_counter  # noqa: PLC0415
    monkeypatch.setattr(awchunk, "_tokenizer_cache", {})
    monkeypatch.setattr(token_counter, "_CACHE", {})
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    cs = awchunk.chunk_markdown(DOC)
    assert len(cs) > 0                      # still produces output
    assert awchunk.tokenizer_status() == "unavailable"
    assert "DISABLED" in capsys.readouterr().err


# --------------------------------------------------------------------------
# 4. determinism
# --------------------------------------------------------------------------

def test_deterministic_across_reruns_and_reimport():
    a = [c.text for c in awchunk.chunk_markdown(DOC)]
    b = [c.text for c in awchunk.chunk_markdown(DOC)]
    assert a == b
    spec = importlib.util.spec_from_file_location("chunk_fresh", _HERE.parent / "cytir" / "chunk" / "chunker.py")
    fresh = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fresh          # dataclasses resolves annotations via sys.modules
    spec.loader.exec_module(fresh)
    assert [c.text for c in fresh.chunk_markdown(DOC, allow_word_only=True)] == a


def test_ordinals_are_dense_and_ordered():
    cs = awchunk.chunk_markdown(DOC)
    assert [c.ordinal for c in cs] == list(range(len(cs)))


def test_empty_and_whitespace_input():
    assert awchunk.chunk_markdown("") == []
    assert awchunk.chunk_markdown("   \n\n  \n") == []


# --------------------------------------------------------------------------
# 5. overlap 0
# --------------------------------------------------------------------------

def test_overlap_zero_emits_every_word_exactly_once():
    md = "# A\n\n" + _words(200, 'a') + "\n\n# B\n\n" + _words(200, 'b')
    cs = awchunk.chunk_markdown(md)
    bodies = " ".join(c.body for c in cs).split()
    assert len(bodies) == len(set(bodies)) == 400


# --------------------------------------------------------------------------
# 6. identity — GAP A33
# --------------------------------------------------------------------------

def _anthropic_class_doc(n_sections=218):
    """The task-325 shape: a long ANCHOR-LESS prose page that chunks into many
    passages. Under the old join-by-(url, anchor=None) path all of these
    collapsed onto ONE point id."""
    parts = ["# Constitution\n"]
    for i in range(n_sections):
        parts.append(f"## Principle {i}\n\n" + _words(300, f"s{i}w") + "\n")
    return "\n".join(parts)


def test_a33_multi_chunk_anchorless_doc_mints_distinct_ids():
    url = "https://anthropic.com/constitution"
    cs = awchunk.chunk_markdown(_anthropic_class_doc())
    assert len(cs) >= 100
    stats = {}
    recs = awchunk.assign_chunk_ids(url, cs, source_type="content", stats=stats)
    assert len(recs) == len(cs), "no chunk may be dropped as a duplicate here"
    assert len({r.point_id for r in recs}) == len(recs)
    assert len({r.digest for r in recs}) == len(recs)
    assert stats["collisions"] == 0 and stats["duplicate_text"] == 0
    assert {r.tier for r in recs} == {"B"}
    # ids are a pure function of (url, chunk text) — rerunnable, position-free
    again = awchunk.assign_chunk_ids(url, awchunk.chunk_markdown(_anthropic_class_doc()),
                                     source_type="content")
    assert [r.point_id for r in again] == [r.point_id for r in recs]


def test_a33_page_level_anchor_does_not_collapse_chunks():
    """Passing the page TITLE as an anchor (the swissinfo/ch-media news-pack
    shape) must be demoted to Tier B, not collapse every chunk onto one id."""
    url = "https://example.org/article"
    cs = awchunk.chunk_markdown("# Art\n\n" + "\n\n".join(
        _words(300, f"n{i}") for i in range(5)))
    recs = awchunk.assign_chunk_ids(url, cs, source_type="news",
                                    native_anchor="The Article Title",
                                    title="The Article Title")
    assert {r.tier for r in recs} == {"B"}
    assert len({r.point_id for r in recs}) == len(recs)


def test_identical_repeated_section_text_dedups_not_overwrites():
    body = _words(100, 'dup')
    md = f"# A\n\n{body}\n\n# A\n\n{body}\n"
    cs = awchunk.chunk_markdown(md)
    assert len(cs) == 2 and cs[0].text == cs[1].text
    stats = {}
    recs = awchunk.assign_chunk_ids("https://x.test/p", cs, stats=stats)
    assert len(recs) == 1 and stats["duplicate_text"] == 1


def test_ids_come_from_aw_b1_identity():
    """Mirror contract: no local uuid5. The id must equal what identity.py
    mints for the same (url, digest)."""
    cs = awchunk.chunk_markdown(DOC)
    recs = awchunk.assign_chunk_ids("https://x.test/p", cs)
    for r in recs:
        assert r.point_id == ident.mint_point_id("https://x.test/p",
                                                 ident.content_digest(r.text))


# --------------------------------------------------------------------------
# 7. epoch stamp + dispatch
# --------------------------------------------------------------------------

def test_epoch_stamp():
    policy = awchunk.active_policy()
    assert awchunk.CHUNK_EPOCH == policy.epoch
    assert awchunk.CHUNK_RULE_LEGEND == policy.rule_legend
    assert awchunk.stamps() == {"chunk": policy.epoch}
    assert all(c.chunk_epoch == policy.epoch for c in awchunk.chunk_markdown(DOC))
    cfg = awchunk.config()
    assert cfg["target_words"] == policy.soft_target
    assert cfg["max_chunk_tokens"] == policy.token_hard_max
    assert cfg["policy"] == policy.manifest()
    assert cfg["algorithm"] == policy.algorithm
    assert cfg["policy_effective"] == policy.value
    assert awchunk.policy_matches_effective(policy)


def test_default_policy_is_loaded_validated_and_byte_hashed():
    policy = awchunk.active_policy()
    assert policy.path.is_file()
    assert policy.sha256 == hashlib.sha256(policy.path.read_bytes()).hexdigest()
    assert policy.word_hard_max == policy.soft_target + policy.floor_words
    assert policy.value["tail"]["under_floor"] == policy.tail_under_floor


def test_policy_loader_rejects_inconsistent_word_guard(tmp_path):
    policy = awchunk.active_policy()
    raw = json.loads(policy.path.read_text(encoding="utf-8"))
    raw["word_guard"]["hard_max"] += 1
    invalid = tmp_path / "invalid-policy.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must equal"):
        awchunk.load_chunk_policy(invalid)


def test_policy_loader_rejects_unimplemented_token_native_algorithm(tmp_path):
    policy = awchunk.active_policy()
    raw = json.loads(policy.path.read_text(encoding="utf-8"))
    raw["algorithm"] = "struct-markdown-token-pack-v1"
    raw["sizing"]["unit"] = "tokens"
    invalid = tmp_path / "token-policy.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="not implemented"):
        awchunk.load_chunk_policy(invalid)


@pytest.fixture(autouse=True)
def _restore_aw_common_after_reload():
    """`importlib.reload` re-executes `cytir.core.common` in place, so every function in it becomes a new
    object while modules imported earlier keep the old ones (`from cytir.core.common import reg_domain`
    then fails an identity check in another module). Put the original namespace back after each test."""
    mod = sys.modules.get("cytir.core.common")
    saved = dict(vars(mod)) if mod is not None else None
    yield
    mod = sys.modules.get("cytir.core.common")
    if mod is not None and saved is not None:
        vars(mod).clear()
        vars(mod).update(saved)


def _reload_aw_common():
    from cytir.core import common as aw_common
    return importlib.reload(aw_common)


def test_chunker_mode_default_is_legacy_and_unknown_raises(monkeypatch):
    ac = _reload_aw_common()
    monkeypatch.delenv("AW_CHUNKER", raising=False)
    assert ac.chunker_mode() == "legacy"
    monkeypatch.setenv("AW_CHUNKER", "struct-768")
    with pytest.raises(ValueError):
        ac.chunker_mode()


def test_chunk_dispatch_selects_the_struct_chunker(monkeypatch, word_only_chunking_allowed):
    ac = _reload_aw_common()
    md = DOC
    policy = awchunk.active_policy()
    monkeypatch.setenv("AW_CHUNKER", "structured")
    monkeypatch.setenv("AW_CHUNK_POLICY", str(policy.path))
    monkeypatch.setenv("AW_CHUNK_POLICY_SHA256", policy.sha256)
    monkeypatch.setenv(
        "AW_CHUNK_TOKENIZER_REVISION",
        "ac6544c8a46e00af67e330e85a9028c66b8cfd9a")
    units, stamps = ac.chunk_markdown_units(md)
    # task 407: the stamp names the policy, not only the epoch family. Five
    # tracked policy artifacts declare epoch 3 and the go-forward epoch-4
    # policy differs from the shipped epoch-3 one in a single field, so the
    # epoch int alone cannot identify the producing rule.
    assert stamps["chunk"] == policy.epoch
    assert stamps["prov_id"] == ac.chunker_selection().provenance()["prov_id"]
    assert units == awchunk.chunk_texts(md)
    monkeypatch.setenv("AW_CHUNKER", "legacy")
    pytest.importorskip("trafilatura")     # fetch_process's import chain
    units2, stamps2 = ac.chunk_markdown_units(md)
    # legacy rows are stamped too — an absent stamp reads as "no information",
    # never as "chunk@1"
    assert stamps2["chunk"] == 1
    assert stamps2["prov_id"] != stamps["prov_id"]
    from cytir.acquire import fetch_process as fp
    assert units2 == fp.chunk(md)


def test_passage_stamps_are_compact_and_absent_by_default():
    ac = _reload_aw_common()
    p = ac.Passage(url="https://x.test/", title="t", snippet="s", text="body",
                   lang="en", domain="x.test", tier="B", source_type="content",
                   trust_tier=0.5)
    assert "stamps" not in p.payload()
    p2 = ac.Passage(url="https://x.test/", title="t", snippet="s", text="body",
                    lang="en", domain="x.test", tier="B", source_type="content",
                    trust_tier=0.5, stamps={"chunk": 2})
    assert p2.payload()["stamps"] == {"chunk": 2}
    assert ac.compact_payload(p2.payload())["stamps"] == {"chunk": 2}


@pytest.fixture(autouse=True)
def _pure_algorithm_word_only(word_only_chunking_allowed):
    """These tests exercise the pure word-boundary algorithm on a box that may lack
    `transformers`. Generic `chunk_markdown` refuses without the tokenizer (task 2395),
    so the module opts in once, visibly, via the conftest fixture (task 2289)."""
