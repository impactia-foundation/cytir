#!/usr/bin/env python3
"""Which pages belong to `chunk.reference`, and why. One verdict per page.

Task 541, for Tim's chunk-width ruling. Task 535 measured that the epoch-4
generic packer cut long, densely-sectioned reference documents at a 3.73x lower
chunk count than the collection production serves, and that 150 of the 262 golds
lost on the held-out 1,418 set are that effect. `chunk.reference`
(`data/chunk-policy-struct-md-192w-o0-adjacent-sections-tail-288-tok2560.v1.json`)
is the narrower policy; THIS module decides which pages get it.

WHAT MAKES THIS SELECTOR LEGITIMATE
-----------------------------------
It reads the page's own markdown and the `chunk.generic` policy artifact, and
nothing else. It never reads the gold set, the incumbent collection, a retrieval
rank, a host name, or a row count -- so it cannot select on the outcome it
reports, and it is not blind to a family nobody has measured yet. Task 535's 82
hosts are this rule's VALIDATION population, never its definition; a host
allow-list would fix the 82 and leave the class (engineering-doctrine 9).

TWO RATIFIED MODES, BOTH REPRODUCIBLE
-------------------------------------
The mode is read from the config's `selection_mode`; a config that names none
is the narrow rule, so `data/chunk-reference-selector.v1.json` keeps behaving
byte-identically. An unrecognised mode is refused, never defaulted.

`narrow-partitioned-and-merged` (v1, task 541, now `superseded` in the
registry) -- arithmetic about what the generic packer DID to this page:

  P1  `generic_chunks >= 2`             the packer partitioned the page at all
  P2  `n_sections / generic_chunks >= 2` it merged independent source sections

`whole-density-class` (v2, ADR-090 amendment 1, PR #130 option 2) -- the
amendment retires P1 ("short pages are in scope, deliberately") and P2 ("the
whole density class"), and decides on the mechanism itself:

  P4  `reference_chunks - generic_chunks >= 1`  the reference policy cuts this
                                                page finer than the generic one

P1 and P2 are still measured and reported under v2, so the share of the class
the narrow selector would have taken is a number in the artifacts rather than
an argument in a task document.

Every mode carries P3, the `chunk.law` exclusion (ADR-090 §4), which is already
enforced by the canonical materializer's `validate_page` and is restated here,
not re-implemented.

P3 AND WHAT COUNTS AS A NATIVE ANCHOR (`native_anchor_rule`, task 575)
----------------------------------------------------------------------
P3 was a CALLER-ASSERTED boolean, and the callers spelled it three different
ways. Task 575 measured what that cost: `build_pages_565.py` spells it
`bool(rec.anchor or rec.identity_anchor_component)`, and
`identity_anchor_component` is the TIER-RESOLVED component -- the content DIGEST
for Tier B. On the staged rows that carry it, the flag is `bool(digest)`: true
of every row with text. Measured over task 572's own 231,226-page run, P3
refused 206,707 of 206,707 ADR-093 L1-b pages (100.000%, 31 waves, no
exception) and 0 of 19,762 sha256-keyed pages -- a split that tracks which
staging era wrote which field name, not any anchor. **Not one refused page was a
law family**; four independent stages (L1-b markdown, corpus store `anchor`
column over 53,990,262 rows, staged pack rows, `aw_e8` payloads) agree that the
lane carries no source-native anchor at all.

So `native_anchor_rule` joins `selection_mode` in the artifact:

  `caller-asserted`     (v1, v2) believe the caller's boolean -- unchanged
  `source-native-only`  (v3) read the RECORD through
                        `aw_b1_identity.has_source_native_anchor`; a
                        tier-resolved identity component is not an anchor

A config naming no rule is `caller-asserted`, so v1 and v2 classify
byte-identically. Under `source-native-only` a `classify()` call with no
`record=` is REFUSED rather than defaulted to False -- defaulting would silently
admit every law article whose caller forgot the record.

No threshold is a chosen number. Each is the smallest integer at which the
property it names is true at all -- for P4, at a gain of 0 the two policies
produce the same boundaries, the same text and the same content-bound point
ids, so there is nothing to supersede. `data/chunk-reference-selector.v{1,2}
.json` state the derivations and `tests/operator/test_aw_chunk_reference_selector.py`
asserts them.

THE TOKENIZER IS NOT OPTIONAL
-----------------------------
`generic_chunks` is tokenizer-sensitive: the generic policy's 1,152-word guard
is overridden by its exact 2,560-token ceiling on code- and table-heavy pages
(word->token 2.2-7.3, task 340 §2.3), and `chunk._token_len` returns None when
`transformers` is unavailable, which the packer reads as "fits". A selector run
without the real tokenizer therefore classifies a different corpus. `classify()`
records `tokenizer` in every verdict and `require_tokenizer()` refuses -- an
error is not an empty (verification-doctrine 6).

Usage
-----
    from cytir.chunk.reference_selector import Selector
    sel = Selector()                       # loads both policies + the config
    sel.require_tokenizer()                # refuse a degraded run, loudly
    v = sel.classify(markdown, has_native_anchor=False, source_type="reference")
    v.policy_family   # "chunk.reference" | "chunk.generic"
    v.reasons         # every predicate, with its measured value
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import cytir.core.ids as IDENT  # the one home for "native anchor"
from cytir.chunk import chunker as CH  # the one chunker; never re-implemented here
import cytir.chunk.registry as REG
from cytir.chunk.policy import ChunkPolicy, load_chunk_policy
from cytir.core.paths import data_path

SCHEMA = "aw.chunk-reference-selector.v1"
DEFAULT_CONFIG = data_path("chunk-reference-selector.v4.json")
REPO = pathlib.Path(__file__).resolve().parents[2]

GENERIC = "chunk.generic"
REFERENCE = "chunk.reference"

#: The two selection modes, both ratified, both reproducible.
#:
#: `narrow-partitioned-and-merged` is the rule task 541 measured and shipped:
#: P1 `generic_chunks >= 2` and P2 `n_sections / generic_chunks >= 2`. It is
#: the DEFAULT when a config names no mode, so `chunk-reference-selector.v1.json`
#: keeps behaving byte-identically after this file grew a second mode.
#:
#: `whole-density-class` is ADR-090 amendment 1's widened rule (PR #130, option
#: 2): P1 and P2 retire and P4 `reference_chunks - generic_chunks >= 1` decides.
#: See `data/chunk-reference-selector.v2.json` for the derivation of each.
MODE_NARROW = "narrow-partitioned-and-merged"
MODE_WHOLE_CLASS = "whole-density-class"
MODES = frozenset({MODE_NARROW, MODE_WHOLE_CLASS})

#: How P3 decides that a record carries a native anchor. The rule is read from
#: the selector artifact, so v1 and v2 keep behaving byte-identically and the
#: change is a visible diff in a ratified file rather than an edit in a function.
#:
#: `caller-asserted` (v1, v2) -- P3 believes the caller's `has_native_anchor`
#: boolean. Every caller spells that boolean differently, and task 575 measured
#: what the spellings do:
#:   the retired `chunk_reference_plan.py`   bool(rec.anchor or rec.anchor_component)
#:   `materialize_565/572.py`    bool(page.has_anchor or payload.anchor),
#:                               where `page.has_anchor` is build_pages_565's
#:                               bool(rec.anchor or rec.identity_anchor_component)
#: `identity_anchor_component` is the TIER-RESOLVED component -- the content
#: DIGEST for Tier B -- so on the staged rows that carry it the flag is
#: `bool(digest)`, true of every row with text. Result, measured: P3 refused
#: 206,707 of 206,707 ADR-093 L1-b pages (100.000%, 31 waves, no exception) and
#: 0 of 19,762 sha256-keyed pages, a split that tracks WHICH STAGING ERA WROTE
#: THE FIELD, not any anchor. Not one refused page was a law family.
#:
#: `source-native-only` (v3, task 575) -- P3 reads the RECORD through
#: `aw_b1_identity.has_source_native_anchor`, the one home for the predicate: a
#: tier-resolved identity component is not an anchor. The `chunk.law` source_type
#: deny list is unchanged and still decides on its own.
NATIVE_CALLER_ASSERTED = "caller-asserted"
NATIVE_SOURCE_ONLY = "source-native-only"
NATIVE_ANCHOR_RULES = frozenset({NATIVE_CALLER_ASSERTED, NATIVE_SOURCE_ONLY})

#: ADR-090 §4 / ADR-079: source families whose boundaries are their own article
#: anchors. The authoritative deny list lives with the canonical materializer's
#: `validate_page`; a caller that has a record in hand passes
#: `has_native_anchor` and this list is the belt to that caller's braces.
CHUNK_LAW_SOURCE_TYPES = frozenset({
    "ch_cantonal_law", "ch_zh_fulltext", "fedlex", "law", "statute",
})


class SelectorError(Exception):
    """A selector that cannot honestly classify the page in front of it."""


@dataclass(frozen=True)
class Verdict:
    """One page, one family, every predicate's measured value.

    ADR-086's shape applied to a classifier: a page never merely stops being
    mentioned. `policy_family` is always set and `reasons` always names the
    predicate that decided it.
    """
    policy_family: str
    decided_by: str
    total_words: int
    n_sections: int
    generic_chunks: int
    sections_per_generic_chunk: Optional[float]
    reference_chunks: Optional[int]
    tokenizer: str
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Selector:
    """The ratified selector, bound to two pinned policy artifacts."""

    def __init__(self, config_path: str | pathlib.Path | None = None,
                 *, repo: pathlib.Path = REPO,
                 registry: dict | None = None) -> None:
        self.repo = repo
        self.config_path = pathlib.Path(config_path or DEFAULT_CONFIG)
        try:
            self.config = json.loads(self.config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SelectorError(
                f"cannot load selector config {self.config_path}: {exc}") from exc
        if self.config.get("schema") != SCHEMA:
            raise SelectorError(
                f"{self.config_path}: schema {self.config.get('schema')!r} is "
                f"not {SCHEMA!r}")

        self.registry = registry if registry is not None else REG.load()
        self.generic = self._pinned("generic_policy", GENERIC)
        self.reference = self._pinned("reference_policy", REFERENCE)

        # A config that names no mode is the narrow rule, so v1 keeps behaving
        # exactly as it did before this module learned a second mode. An
        # UNKNOWN mode is a refusal, never a silent fall back to the narrow one
        # -- a selector that quietly classified a different corpus than its
        # config asked for is the failure this whole family exists to detect.
        self.mode = self.config.get("selection_mode", MODE_NARROW)
        if self.mode not in MODES:
            raise SelectorError(
                f"{self.config_path.name}: selection_mode {self.mode!r} is "
                f"not one of {sorted(MODES)}. An unrecognised mode is refused "
                "rather than defaulted (engineering-doctrine 3).")

        # A config that names no native-anchor rule is the caller-asserted one,
        # so v1 and v2 keep classifying byte-identically after this module
        # learned a second rule. An UNKNOWN rule is refused, never defaulted.
        self.native_anchor_rule = self.config.get(
            "native_anchor_rule", NATIVE_CALLER_ASSERTED)
        if self.native_anchor_rule not in NATIVE_ANCHOR_RULES:
            raise SelectorError(
                f"{self.config_path.name}: native_anchor_rule "
                f"{self.native_anchor_rule!r} is not one of "
                f"{sorted(NATIVE_ANCHOR_RULES)}. An unrecognised rule is "
                "refused rather than defaulted (engineering-doctrine 3).")

        # Thresholds are read from the config, so a change to any of them is a
        # visible diff in a ratified artifact and not an edit in a function.
        # The narrow mode's P1/P2 are read in BOTH modes: the widened selector
        # still reports what the narrow one would have decided, so the share of
        # its population v1 would have selected is a measurement rather than an
        # argument.
        src = (self.config.get("predicates") if self.mode == MODE_NARROW
               else self.config.get("retired_predicates"))
        self.min_generic_chunks = self._int_rule(
            "P1_partitioned", block=src, key=("rule" if self.mode == MODE_NARROW
                                              else "rule_in_v1"))
        self.min_sections_per_chunk = self._int_rule(
            "P2_sections_merged", block=src,
            key=("rule" if self.mode == MODE_NARROW else "rule_in_v1"))
        self.min_reference_gain = (
            self._int_rule("P4_reference_cuts_finer")
            if self.mode == MODE_WHOLE_CLASS else None)

    # -- construction helpers ------------------------------------------------

    def _pinned(self, key: str, expect_family: str) -> ChunkPolicy:
        block = self.config.get(key) or {}
        path = self.repo / block.get("file", "")
        policy = load_chunk_policy(path)
        want = block.get("sha256")
        if want and policy.sha256 != want:
            raise SelectorError(
                f"{block.get('file')}: sha256 {policy.sha256} != the "
                f"{key} pin {want} in {self.config_path.name}. A policy "
                "artifact is identified by its bytes; a moved pin is a "
                "different rule, not a typo.")
        got_family = REG.family_of(policy.sha256, self.registry)
        if got_family != expect_family:
            raise SelectorError(
                f"{block.get('file')} is registered under policy_family "
                f"{got_family!r}, but {key} must be {expect_family!r}")
        return policy

    def _int_rule(self, predicate: str, *, block: dict | None = None,
                  key: str = "rule") -> int:
        """Read the integer out of a predicate's stated rule.

        The rule string in the artifact IS the specification ('generic_chunks
        >= 2'); parsing it here keeps one copy of the threshold rather than a
        literal in code that agrees with the artifact only until someone edits
        one of them (engineering-doctrine 1/2).
        """
        src = self.config.get("predicates") if block is None else block
        rule = ((src or {}).get(predicate) or {}).get(key)
        if not isinstance(rule, str) or ">=" not in rule:
            raise SelectorError(
                f"{self.config_path.name}: predicate {predicate} has no "
                f"'>=' rule to read a threshold from (got {rule!r})")
        tail = rule.split(">=", 1)[1].strip()
        try:
            return int(tail)
        except ValueError as exc:
            raise SelectorError(
                f"{self.config_path.name}: predicate {predicate} threshold "
                f"{tail!r} is not an integer") from exc

    # -- the tokenizer refusal ----------------------------------------------

    def tokenizer_status(self) -> str:
        """`ok` once the real tokenizer has been loaded, else its state."""
        return CH.tokenizer_status()

    def require_tokenizer(self) -> str:
        """Refuse a run whose chunk counts would be measured with a ruler that
        is not the one the corpus was cut with.

        `chunk.tokenizer_status()` probes and caches the tokenizer itself, so
        this is the load attempt as well as the check. It is deliberately NOT
        `chunk.token_len()`, which raises when the counter is unavailable: this
        method's job is to convert that state into ONE named refusal, not to
        surface a RuntimeError from three frames down.
        """
        with self._active(self.generic):
            status = CH.tokenizer_status()
        if status != "ok":
            raise SelectorError(
                f"the arctic tokenizer is {status!r}. `generic_chunks` is "
                "tokenizer-sensitive (the 2,560-token ceiling overrides the "
                "1,152-word guard on code/table pages), so a selector run "
                "without it classifies a different corpus. Install "
                "`transformers` and the pinned snapshot, or do not run.")
        return status

    # -- the classification --------------------------------------------------

    @contextlib.contextmanager
    def _active(self, policy: ChunkPolicy):
        """Select a policy and PUT THE MODULE BACK.

        `cytir/chunk/chunker.py` keeps its active policy in module globals, so
        selecting one is a process-wide side effect. A classifier that selects
        two policies per page and leaves the last one installed silently
        re-configures every later caller in the same process -- which is
        exactly what it did until the chunk-pipeline test module started failing when
        run AFTER this module's tests and passing alone. That is a defect in
        this module, not a test-ordering quirk.

        `_apply_policy` is reached directly, deliberately: `select_policy()`
        restores with `allow_env_overrides=False`, and the module's own
        import-time call uses True, so restoring through the public helper
        would not restore the state we found. The restoration is then ASSERTED
        against the effective config captured before, so a drift fails loudly
        instead of leaking (engineering-doctrine 3).
        """
        before = CH.active_policy()
        before_cfg = CH.effective_policy_value()
        CH._apply_policy(policy, allow_env_overrides=False)
        try:
            yield
        finally:
            CH._apply_policy(before, allow_env_overrides=True)
            if CH.effective_policy_value() != before_cfg:
                raise SelectorError(
                    "failed to restore chunk.py's active policy after "
                    f"selecting {policy.policy_id!r}; the module is now "
                    "configured differently than it was found and every later "
                    "caller in this process would chunk under the wrong rule")

    def chunk_under_reference(self, markdown: str) -> list:
        """Public: chunk one page under the pinned reference policy.

        The materializer uses this rather than selecting the policy itself, so
        there is one call site that knows how to put the module back.
        """
        return self._chunks_under(self.reference, markdown)

    def chunk_under(self, policy: ChunkPolicy, markdown: str) -> list:
        """Public: chunk one page under ANY loaded policy, module restored.

        `chunk.py` keeps its active policy in module globals, so selecting one
        is a process-wide side effect and putting it back is the caller's
        problem. `_active` is the one place that knows how (and asserts the
        restoration), so a width sweep that cuts the same page under four
        policies must come through here rather than calling
        `chunk._apply_policy` itself -- a second implementation of the restore
        is how the chunk-pipeline test module started failing only when run after
        this module's tests (engineering-doctrine 1/6).

        The caller is responsible for the STATUS check
        (`chunk_policy_registry.assert_selectable`): this method takes an
        already-loaded policy and does not adjudicate whether that policy may
        be selected. Task 569 passes `experiment=True` there.
        """
        return self._chunks_under(policy, markdown)

    def _chunks_under(self, policy: ChunkPolicy, markdown: str) -> list:
        with self._active(policy):
            return CH.chunk_markdown(markdown)

    # -- P3, the one place the native-anchor question is answered -----------

    def native_anchor_verdict(self, *, has_native_anchor: bool,
                              source_type: str | None,
                              record: dict | None) -> tuple[bool, str | None, str]:
        """(refuse, source_type, evidence) for P3, under the config's rule.

        Delegates to `aw_b1_identity.has_source_native_anchor` under
        `source-native-only`; there is no second implementation of the predicate
        here (engineering-doctrine 1/6).
        """
        if self.native_anchor_rule == NATIVE_CALLER_ASSERTED:
            st = source_type
            return (bool(has_native_anchor)
                    or (st or "") in CHUNK_LAW_SOURCE_TYPES), st, (
                f"has_native_anchor={bool(has_native_anchor)} (caller-asserted)")

        # source-native-only. The rule reads the RECORD, so a caller that
        # supplies none cannot be given a verdict: refusing is the only honest
        # answer, because defaulting to False would silently ADMIT every law
        # article whose caller forgot the record (verification-doctrine 6).
        if record is None:
            raise SelectorError(
                f"{self.config_path.name} sets native_anchor_rule="
                f"{NATIVE_SOURCE_ONLY!r}, which reads the record's own anchor "
                "field, but classify() was called without `record=`. A "
                "caller-asserted boolean is not admissible under this rule -- "
                "pass the record, or use a selector artifact whose rule is "
                f"{NATIVE_CALLER_ASSERTED!r}.")
        native = IDENT.source_native_anchor(record)
        st = source_type if source_type is not None else record.get("source_type")
        return (native is not None) or ((st or "") in CHUNK_LAW_SOURCE_TYPES), st, (
            f"source_native_anchor={native!r} (read from the record; a "
            "tier-resolved identity component is not an anchor)")

    def classify_and_cut(self, markdown: str, *,
                         has_native_anchor: bool = False,
                         source_type: str | None = None,
                         record: dict | None = None,
                         ) -> tuple[Verdict, list]:
        """The verdict AND the chunks, with each policy applied exactly once.

        A materializer that calls `classify()` and then
        `chunk_under_reference()` cuts the same page under the reference
        policy TWICE -- once to count, once to keep -- and a corpus-scale
        re-chunk pays that on every page. This is the one call that does not,
        and it decides through `classify()` rather than beside it, so there is
        still exactly one implementation of the rule (engineering-doctrine 1).

        Returns `(verdict, [])` for a page the selector assigns to
        `chunk.generic`; the caller must not chunk it.
        """
        v = self.classify(markdown, has_native_anchor=has_native_anchor,
                          source_type=source_type, record=record,
                          count_reference=False)
        if v.decided_by in ("P3_no_native_anchor", "no_sections"):
            return v, []          # decided before any reference cut was made
        chunks = self._chunks_under(self.reference, markdown)
        v = self.classify(markdown, has_native_anchor=has_native_anchor,
                          source_type=source_type, record=record,
                          precut_reference=chunks)
        return v, (chunks if v.policy_family == REFERENCE else [])

    def classify(self, markdown: str, *, has_native_anchor: bool = False,
                 source_type: str | None = None,
                 record: dict | None = None,
                 count_reference: bool = True,
                 precut_reference: list | None = None) -> Verdict:
        """One page's family, with every predicate's measured value.

        P3, the chunk.law exclusion (ADR-090 §4), is answered by
        `native_anchor_verdict` under the artifact's `native_anchor_rule`:
        `caller-asserted` believes `has_native_anchor`; `source-native-only`
        reads `record`'s own anchor field through `aw_b1_identity`. The caller
        has the record; this module has the markdown.
        """
        tokenizer = CH.tokenizer_status()
        sections = CH.parse_sections(markdown)
        total_words = sum(s.words for s in sections)
        n_sections = len(sections)

        def verdict(family: str, decided_by: str, reasons: list[str],
                    generic_chunks: int = 0,
                    ratio: Optional[float] = None,
                    reference_chunks: Optional[int] = None) -> Verdict:
            return Verdict(policy_family=family, decided_by=decided_by,
                           total_words=total_words, n_sections=n_sections,
                           generic_chunks=generic_chunks,
                           sections_per_generic_chunk=ratio,
                           reference_chunks=reference_chunks,
                           tokenizer=tokenizer, reasons=reasons)

        # -- P3, first, because it is a hard scope bound and costs nothing ----
        refuse, p3_source_type, p3_evidence = self.native_anchor_verdict(
            has_native_anchor=has_native_anchor, source_type=source_type,
            record=record)
        if refuse:
            return verdict(
                GENERIC, "P3_no_native_anchor",
                [f"P3 FAIL [{self.native_anchor_rule}]: {p3_evidence}, "
                 f"source_type={p3_source_type!r} -- chunk.law is out of scope "
                 "(ADR-090 §4)"])

        if not sections:
            return verdict(GENERIC, "no_sections",
                           ["the markdown parses to zero sections; there is "
                            "nothing to re-cut"])

        generic_chunks = len(self._chunks_under(self.generic, markdown))

        if self.mode == MODE_WHOLE_CLASS:
            # ADR-090 amendment 1: the whole density class, short pages
            # included. P1 and P2 are still MEASURED and reported -- so the
            # share of this population the narrow selector would have taken
            # stays a number in the artifacts -- but they no longer decide.
            ratio = n_sections / generic_chunks if generic_chunks else None
            n_ref = (len(precut_reference) if precut_reference is not None
                     else len(self._chunks_under(self.reference, markdown)))
            gain = n_ref - generic_chunks
            narrow = (generic_chunks >= self.min_generic_chunks
                      and ratio is not None
                      and ratio >= self.min_sections_per_chunk)
            common = [
                f"P3 PASS [{self.native_anchor_rule}]: {p3_evidence}",
                f"P1 (retired, reported): generic_chunks={generic_chunks} "
                f"{'>=' if generic_chunks >= self.min_generic_chunks else '<'} "
                f"{self.min_generic_chunks}",
                f"P2 (retired, reported): sections/chunk="
                f"{'n/a' if ratio is None else format(ratio, '.3f')}",
                f"narrow_selector_v1_would_select={narrow}",
            ]
            if gain < self.min_reference_gain:
                return verdict(
                    GENERIC, "P4_reference_cuts_finer",
                    common + [
                        f"P4 FAIL: reference_chunks={n_ref} - "
                        f"generic_chunks={generic_chunks} = {gain} < "
                        f"{self.min_reference_gain} -- the reference policy "
                        "does not cut this page any finer, so re-chunking it "
                        "mints the same content-bound ids and there is "
                        "nothing to supersede"],
                    generic_chunks=generic_chunks, ratio=ratio,
                    reference_chunks=n_ref)
            return verdict(
                REFERENCE, "P4",
                common + [
                    f"P4 PASS: reference_chunks={n_ref} - "
                    f"generic_chunks={generic_chunks} = {gain} >= "
                    f"{self.min_reference_gain}"],
                generic_chunks=generic_chunks, ratio=ratio,
                reference_chunks=n_ref)

        if generic_chunks < self.min_generic_chunks:
            return verdict(
                GENERIC, "P1_partitioned",
                [f"P1 FAIL: generic_chunks={generic_chunks} < "
                 f"{self.min_generic_chunks} -- the generic packer returned "
                 "the page whole, so it made no partitioning decision to "
                 "revisit (short-document policy is unruled, ADR-090 §5)"],
                generic_chunks=generic_chunks)

        ratio = n_sections / generic_chunks
        if ratio < self.min_sections_per_chunk:
            return verdict(
                GENERIC, "P2_sections_merged",
                [f"P1 PASS: generic_chunks={generic_chunks}",
                 f"P2 FAIL: sections/chunk={ratio:.3f} < "
                 f"{self.min_sections_per_chunk} -- the generic packer did not "
                 "merge independent sections on this page"],
                generic_chunks=generic_chunks, ratio=ratio)

        reference_chunks = (len(self._chunks_under(self.reference, markdown))
                            if count_reference else None)
        return verdict(
            REFERENCE, "P1+P2",
            [f"P1 PASS: generic_chunks={generic_chunks} >= "
             f"{self.min_generic_chunks}",
             f"P2 PASS: sections/chunk={ratio:.3f} >= "
             f"{self.min_sections_per_chunk}",
             f"P3 PASS [{self.native_anchor_rule}]: {p3_evidence}"],
            generic_chunks=generic_chunks, ratio=ratio,
            reference_chunks=reference_chunks)


def main(argv: Optional[list[str]] = None, prog: Optional[str] = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description="Classify one markdown file.")
    ap.add_argument("markdown", help="path to a markdown file, or '-' for stdin")
    ap.add_argument("--config", default=None,
                    help="selector artifact (default "
                         f"{DEFAULT_CONFIG}); v3 sets native_anchor_rule="
                         f"{NATIVE_SOURCE_ONLY!r} and needs --record")
    ap.add_argument("--source-type", default=None)
    ap.add_argument("--has-native-anchor", action="store_true")
    ap.add_argument("--record", default=None,
                    help="path to a JSON record, required under "
                         f"native_anchor_rule={NATIVE_SOURCE_ONLY!r}")
    ap.add_argument("--allow-no-tokenizer", action="store_true",
                    help="report anyway with the degraded ruler; the verdict "
                         "carries `tokenizer` either way, and a run with this "
                         "flag may not be quoted as a corpus measurement")
    a = ap.parse_args(argv)
    sel = Selector(a.config)
    if not a.allow_no_tokenizer:
        sel.require_tokenizer()
    text = (sys.stdin.read() if a.markdown == "-"
            else pathlib.Path(a.markdown).read_text())
    record = (json.loads(pathlib.Path(a.record).read_text())
              if a.record else None)
    v = sel.classify(text, has_native_anchor=a.has_native_anchor,
                     source_type=a.source_type, record=record)
    print(json.dumps(v.as_dict(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
