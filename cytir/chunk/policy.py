#!/usr/bin/env python3
"""Versioned chunk-policy artifact loader.

Chunk boundaries and identities depend on every value represented here.  The
policy is therefore tracked data with an exact byte hash, not a collection of
constants copied between the chunker, materializer, and build documentation.
"""
from __future__ import annotations

from cytir.core.paths import data_dir, data_path

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "aw.chunk-policy.v1"
REPO = Path(__file__).resolve().parents[2]


def _display_path(path: Path) -> str:
    """The path a manifest records: repo-relative in a checkout, `data/<name>` for a policy read
    from an installed package's data (so the same policy binds to the same string on every box)."""
    for base, prefix in ((REPO, ""), (data_dir(), "data/")):
        try:
            return prefix + path.relative_to(base).as_posix()
        except ValueError:
            continue
    return str(path)


DEFAULT_POLICY_PATH = data_path("chunk-policy-struct-md-768w-o0.v1.json")

COMMON_TOP_KEYS = frozenset({
    "schema", "policy_id", "policy_version", "algorithm", "epoch",
    "rule_legend",
})
BASE_TOP_KEYS = COMMON_TOP_KEYS | frozenset({
    "sizing", "word_guard", "token_hard_max", "overlap",
    "merge", "floor", "prefix", "tail",
})
SECTION_PACKING_KEY = "section_packing"
STRUCT_WORD_PACK = "struct-markdown-word-pack-v1"
ADJACENT_SECTION_PACK = "struct-markdown-adjacent-section-pack-v1"

# Task 573 / E573-1. `chunk.law` cuts one chunk per source-native article
# (ADR-090 3/4) and that OUTER boundary is not up for negotiation here: the
# law-article sub-pack keeps it and splits only the BODY of an article that is
# wider than the tail, at the same widths `chunk.reference` uses. It is a
# distinct algorithm name because it is a distinct producer -- declaring these
# rows under the adjacent-section pack would say the section packer chose their
# outer boundaries, which it did not (corpus-doctrine 1: the representation is
# a contract, and a producer described as "the same, really" is the tell).
LAW_ARTICLE_SUB_PACK = "law-article-sub-pack-v1"
LAW_ARTICLE_KEY = "law_article"

# ADR-100 Ruling A: the daily news lane chunks with the legacy 180w/o30
# windower (`cytir/acquire/fetch_process.chunk`), which has no tunable
# parameters in production — it is ONE deployed algorithm, not a family.
# This schema therefore binds the deployed function's exact literal
# parameters and its exact producer identity, rather than describing a
# parameterizable policy the way the two structured schemas do.
LEGACY_FIXED_WINDOW_PACK = "legacy-fixed-window-pack-v1"
SUPPORTED_ALGORITHMS = frozenset({
    STRUCT_WORD_PACK, ADJACENT_SECTION_PACK, LEGACY_FIXED_WINDOW_PACK,
    LAW_ARTICLE_SUB_PACK})

LEGACY_WINDOW_WORDS = 180
LEGACY_OVERLAP_WORDS = 30
LEGACY_RULE_LEGEND = "legacy-180w-o30@1"
LEGACY_PRODUCER_MODULE = "pipeline/fetch_process.py"
LEGACY_PRODUCER_FUNCTION = "chunk"
LEGACY_TOP_KEYS = COMMON_TOP_KEYS | frozenset({
    "window_words", "overlap_words", "producer"})
LEGACY_PRODUCER_KEYS = frozenset({"module", "function", "git_revision"})
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
NESTED_KEYS = {
    "sizing": frozenset({"unit", "soft_target"}),
    "word_guard": frozenset({"unit", "hard_max", "derivation"}),
    "token_hard_max": frozenset({
        "unit", "exact", "tokenizer_model", "count_special_tokens",
        "split_policy",
    }),
    "overlap": frozenset({"unit", "size", "scope"}),
    "merge": frozenset({"unit", "below", "policy"}),
    "floor": frozenset({"unit", "minimum", "per_call_override"}),
    "prefix": frozenset({"unit", "maximum", "separator", "truncation"}),
    "tail": frozenset({"under_floor", "token_split_remainder"}),
    "section_packing": frozenset({"unit", "policy", "headings", "order"}),
    # The outer boundary and the identity rule of the law sub-pack, declared as
    # data so a later reader can state what a `chunk.law.sub` row means without
    # reading the producer (corpus-doctrine 7).
    LAW_ARTICLE_KEY: frozenset({
        "unit", "boundary", "split_body_above_words", "short_article",
        "anchor_carried", "identity",
    }),
}


class ChunkPolicyError(ValueError):
    """A missing, malformed, or internally inconsistent chunk policy."""


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ChunkPolicyError(f"{field} must be an integer >= 1")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ChunkPolicyError(f"{field} must be an integer >= 0")
    return value


def _literal(value: Any, expected: Any, field: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ChunkPolicyError(f"{field} must be {expected!r}, got {value!r}")


def _object(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict) or set(value) != NESTED_KEYS[key]:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ChunkPolicyError(
            f"{key} must contain exactly {sorted(NESTED_KEYS[key])}, got {got}")
    return value


@dataclass(frozen=True)
class ChunkPolicy:
    path: Path
    sha256: str
    value: dict[str, Any]
    schema: str
    policy_id: str
    policy_version: int
    algorithm: str
    epoch: int
    rule_legend: str
    sizing_unit: str
    soft_target: int
    word_hard_max: int
    token_hard_max: int
    tokenizer_model: str
    overlap_words: int
    merge_below_words: int
    floor_words: int
    prefix_max_words: int
    prefix_separator: str
    tail_under_floor: str
    token_split_remainder: str
    section_packing_policy: str | None
    #: Present only for `law-article-sub-pack-v1` (task 573 / E573-1); None for
    #: every other algorithm, so an existing caller sees no change.
    law_article: dict[str, Any] | None = None

    def manifest(self) -> dict[str, Any]:
        """Exact policy binding carried by every materialization manifest."""
        display_path = _display_path(self.path)
        return {
            "path": display_path,
            "sha256": self.sha256,
            "schema": self.schema,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "algorithm": self.algorithm,
            "value": copy.deepcopy(self.value),
        }


@dataclass(frozen=True)
class LegacyChunkPolicy:
    """ADR-100 Ruling A: the descriptor for the deployed legacy chunker.

    Unlike `ChunkPolicy`, this describes exactly one already-deployed
    algorithm (`cytir/acquire/fetch_process.chunk`) rather than a family of
    tunable structured policies — `window_words`/`overlap_words` are the
    literal values that function already runs with in production, bound
    here as explicit provenance, not as configuration a caller may vary.
    """

    path: Path
    sha256: str
    value: dict[str, Any]
    schema: str
    policy_id: str
    policy_version: int
    algorithm: str
    epoch: int
    rule_legend: str
    window_words: int
    overlap_words: int
    producer_module: str
    producer_function: str
    producer_git_revision: str

    def manifest(self) -> dict[str, Any]:
        """Exact policy binding carried by every materialization manifest."""
        display_path = _display_path(self.path)
        return {
            "path": display_path,
            "sha256": self.sha256,
            "schema": self.schema,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "algorithm": self.algorithm,
            "value": copy.deepcopy(self.value),
        }


def _load_legacy_fixed_window_policy(
        raw: dict[str, Any], resolved: Path, content: bytes, *,
        version: int, epoch: int) -> LegacyChunkPolicy:
    """Strict loader for `legacy-fixed-window-pack-v1` (ADR-100 Ruling A).

    Called after `load_chunk_policy` has already checked the fields common
    to every algorithm (schema, policy_id, policy_version, epoch positivity,
    rule_legend/@epoch suffix). Every field here is checked against the ONE
    deployed algorithm's literal values — this schema has no tunable knobs,
    so "strict" here means exact-match, not merely well-typed.
    """
    _literal(epoch, 1, "epoch")
    _literal(raw["rule_legend"], LEGACY_RULE_LEGEND, "rule_legend")

    window_words = raw.get("window_words")
    _literal(window_words, LEGACY_WINDOW_WORDS, "window_words")
    overlap_words = raw.get("overlap_words")
    _literal(overlap_words, LEGACY_OVERLAP_WORDS, "overlap_words")

    producer = raw.get("producer")
    if not isinstance(producer, dict) or set(producer) != LEGACY_PRODUCER_KEYS:
        got = sorted(producer) if isinstance(producer, dict) else type(producer).__name__
        raise ChunkPolicyError(
            f"producer must contain exactly {sorted(LEGACY_PRODUCER_KEYS)}, got {got}")
    _literal(producer["module"], LEGACY_PRODUCER_MODULE, "producer.module")
    _literal(producer["function"], LEGACY_PRODUCER_FUNCTION, "producer.function")
    revision = producer["git_revision"]
    if not isinstance(revision, str) or not _GIT_SHA_RE.fullmatch(revision):
        raise ChunkPolicyError(
            "producer.git_revision must be a lowercase 40-hex git commit SHA")

    return LegacyChunkPolicy(
        path=resolved,
        sha256=hashlib.sha256(content).hexdigest(),
        value=raw,
        schema=raw["schema"],
        policy_id=raw["policy_id"],
        policy_version=version,
        algorithm=LEGACY_FIXED_WINDOW_PACK,
        epoch=epoch,
        rule_legend=raw["rule_legend"],
        window_words=window_words,
        overlap_words=overlap_words,
        producer_module=producer["module"],
        producer_function=producer["function"],
        producer_git_revision=revision,
    )


def load_chunk_policy(
        path: str | Path | None = None) -> "ChunkPolicy | LegacyChunkPolicy":
    resolved = Path(path or DEFAULT_POLICY_PATH).expanduser().resolve()
    try:
        content = resolved.read_bytes()
        raw = json.loads(content)
    except (OSError, json.JSONDecodeError) as exc:
        raise ChunkPolicyError(f"cannot load chunk policy {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ChunkPolicyError("policy must be an object")
    algorithm = raw.get("algorithm")
    if algorithm == ADJACENT_SECTION_PACK:
        expected_keys = BASE_TOP_KEYS | {SECTION_PACKING_KEY}
    elif algorithm == LAW_ARTICLE_SUB_PACK:
        expected_keys = BASE_TOP_KEYS | {SECTION_PACKING_KEY, LAW_ARTICLE_KEY}
    elif algorithm == LEGACY_FIXED_WINDOW_PACK:
        expected_keys = LEGACY_TOP_KEYS
    else:
        expected_keys = BASE_TOP_KEYS
    if set(raw) != expected_keys:
        got = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise ChunkPolicyError(
            f"policy must contain exactly {sorted(expected_keys)}, got {got}")

    _literal(raw["schema"], SCHEMA, "schema")
    if not isinstance(raw["policy_id"], str) or not raw["policy_id"].strip():
        raise ChunkPolicyError("policy_id must be a non-empty string")
    version = _positive_int(raw["policy_version"], "policy_version")
    algorithm = raw["algorithm"]
    if not isinstance(algorithm, str) or algorithm not in SUPPORTED_ALGORITHMS:
        raise ChunkPolicyError(
            f"algorithm {algorithm!r} is not implemented; supported: "
            f"{sorted(SUPPORTED_ALGORITHMS)}")
    epoch = _positive_int(raw["epoch"], "epoch")
    if not isinstance(raw["rule_legend"], str) or not raw["rule_legend"].strip():
        raise ChunkPolicyError("rule_legend must be a non-empty string")
    if not raw["rule_legend"].endswith(f"@{epoch}"):
        raise ChunkPolicyError("rule_legend must end with the declared @epoch")

    if algorithm == LEGACY_FIXED_WINDOW_PACK:
        return _load_legacy_fixed_window_policy(
            raw, resolved, content, version=version, epoch=epoch)

    sizing = _object(raw, "sizing")
    _literal(sizing["unit"], "words", "sizing.unit")
    soft_target = _positive_int(sizing["soft_target"], "sizing.soft_target")

    floor = _object(raw, "floor")
    _literal(floor["unit"], "words", "floor.unit")
    floor_words = _positive_int(floor["minimum"], "floor.minimum")
    _literal(floor["per_call_override"], True, "floor.per_call_override")

    word_guard = _object(raw, "word_guard")
    _literal(word_guard["unit"], "words", "word_guard.unit")
    derivation = word_guard["derivation"]
    if derivation not in {"soft-target-plus-floor", "explicit-tail-maximum"}:
        raise ChunkPolicyError(
            "word_guard.derivation must be 'soft-target-plus-floor' or "
            "'explicit-tail-maximum'")
    word_hard_max = _positive_int(word_guard["hard_max"], "word_guard.hard_max")
    if derivation == "soft-target-plus-floor" and word_hard_max != soft_target + floor_words:
        raise ChunkPolicyError(
            "word_guard.hard_max must equal sizing.soft_target + floor.minimum")
    if derivation == "explicit-tail-maximum" and word_hard_max < soft_target:
        raise ChunkPolicyError(
            "explicit word_guard.hard_max must be at least sizing.soft_target")

    token = _object(raw, "token_hard_max")
    _literal(token["unit"], "tokens", "token_hard_max.unit")
    token_hard_max = _positive_int(token["exact"], "token_hard_max.exact")
    if not isinstance(token["tokenizer_model"], str) or not token["tokenizer_model"].strip():
        raise ChunkPolicyError("token_hard_max.tokenizer_model must be non-empty")
    _literal(token["count_special_tokens"], True,
             "token_hard_max.count_special_tokens")
    _literal(token["split_policy"], "line-then-word-exact",
             "token_hard_max.split_policy")

    overlap = _object(raw, "overlap")
    _literal(overlap["unit"], "words", "overlap.unit")
    overlap_words = _nonnegative_int(overlap["size"], "overlap.size")
    if overlap_words >= soft_target:
        raise ChunkPolicyError("overlap.size must be smaller than sizing.soft_target")
    _literal(overlap["scope"], "oversized-unit", "overlap.scope")

    merge = _object(raw, "merge")
    _literal(merge["unit"], "words", "merge.unit")
    merge_words = _positive_int(merge["below"], "merge.below")
    if merge_words > soft_target:
        raise ChunkPolicyError("merge.below cannot exceed sizing.soft_target")
    _literal(merge["policy"], "previous-if-fits-else-next", "merge.policy")

    prefix = _object(raw, "prefix")
    _literal(prefix["unit"], "words", "prefix.unit")
    prefix_words = _nonnegative_int(prefix["maximum"], "prefix.maximum")
    if prefix_words >= soft_target:
        raise ChunkPolicyError("prefix.maximum must be smaller than sizing.soft_target")
    if not isinstance(prefix["separator"], str) or not prefix["separator"]:
        raise ChunkPolicyError("prefix.separator must be a non-empty string")
    _literal(prefix["truncation"], "keep-deepest", "prefix.truncation")

    tail = _object(raw, "tail")
    tail_under_floor = tail["under_floor"]
    expected_tail = ("merge-into-previous" if algorithm == STRUCT_WORD_PACK
                     else "merge-into-previous-or-emit")
    _literal(tail_under_floor, expected_tail, "tail.under_floor")
    _literal(tail["token_split_remainder"], "emit-below-floor",
             "tail.token_split_remainder")

    section_packing_policy: str | None = None
    law_article: dict[str, Any] | None = None
    if algorithm in (ADJACENT_SECTION_PACK, LAW_ARTICLE_SUB_PACK):
        section_packing = _object(raw, SECTION_PACKING_KEY)
        _literal(section_packing["unit"], "sections", "section_packing.unit")
        if section_packing["policy"] not in {"adjacent-greedy-balanced-tail",
                                              "keep-whole-balanced"}:
            raise ChunkPolicyError("unsupported section_packing.policy")
        _literal(section_packing["headings"], "inline-breadcrumb",
                 "section_packing.headings")
        _literal(section_packing["order"], "source-order", "section_packing.order")
        section_packing_policy = section_packing["policy"]

    if algorithm == LAW_ARTICLE_SUB_PACK:
        law_article = _object(raw, LAW_ARTICLE_KEY)
        _literal(law_article["unit"], "articles", "law_article.unit")
        _literal(law_article["boundary"], "source-native-anchor",
                 "law_article.boundary")
        # The threshold is not a free number: an article at or below the tail
        # is emitted whole by construction, so the only honest value is the
        # policy's own hard max. A policy that claims to split above some other
        # width would describe boundaries its own packer cannot produce.
        split_above = _positive_int(law_article["split_body_above_words"],
                                    "law_article.split_body_above_words")
        if split_above != word_hard_max:
            raise ChunkPolicyError(
                "law_article.split_body_above_words must equal "
                f"word_guard.hard_max ({word_hard_max}), got {split_above}")
        _literal(law_article["short_article"], "emit-whole-unchanged",
                 "law_article.short_article")
        _literal(law_article["anchor_carried"], "every-sub-chunk",
                 "law_article.anchor_carried")
        _literal(law_article["identity"], "anchor-plus-sub-ordinal",
                 "law_article.identity")

    return ChunkPolicy(
        path=resolved,
        sha256=hashlib.sha256(content).hexdigest(),
        value=raw,
        schema=raw["schema"],
        policy_id=raw["policy_id"],
        policy_version=version,
        algorithm=algorithm,
        epoch=epoch,
        rule_legend=raw["rule_legend"],
        sizing_unit=sizing["unit"],
        soft_target=soft_target,
        word_hard_max=word_hard_max,
        token_hard_max=token_hard_max,
        tokenizer_model=token["tokenizer_model"],
        overlap_words=overlap_words,
        merge_below_words=merge_words,
        floor_words=floor_words,
        prefix_max_words=prefix_words,
        prefix_separator=prefix["separator"],
        tail_under_floor=tail_under_floor,
        token_split_remainder=tail["token_split_remainder"],
        section_packing_policy=section_packing_policy,
        law_article=law_article,
    )
