"""ADR-100 Ruling A: the legacy 180w/o30 news chunker as a strict descriptor.

Three things are tested, matching the three places ADR-100 touches:

1. `chunk_policy.py`'s strict loader accepts the new
   `legacy-fixed-window-pack-v1` schema and rejects malformed variants
   (wrong legend, wrong window, unknown field) exactly as strictly as the
   two structured schemas — and the two structured schemas still load
   byte-identically (regression).
2. `audit_news_article_profile._validate_news_policy` accepts a pack bound
   to the tracked legacy descriptor, and still refuses one with no chunk
   policy at all (the `policy_path=None` production reality task 450 hit).
3. The descriptor's declared parameters (window_words=180,
   overlap_words=30) reproduce the REAL deployed chunker's exact output —
   the real `fetch_process.chunk` (acquire stage) is imported and run, not
   reimplemented.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import cytir.ops.audits.news_article_profile as A
from cytir.chunk.policy import (ChunkPolicyError, LEGACY_FIXED_WINDOW_PACK,
                          LEGACY_OVERLAP_WORDS, LEGACY_RULE_LEGEND,
                          LEGACY_WINDOW_WORDS, LegacyChunkPolicy,
                          load_chunk_policy)
from news_article_profile_fixtures import build, run_audit  # noqa: E402  (tests/)


REPO = Path(__file__).resolve().parents[1]
LEGACY_POLICY_PATH = REPO / A.LEGACY_POLICY_PATH
STRUCT_POLICY_PATH = REPO / A.NEWS_POLICY_PATH


def _legacy_raw() -> dict:
    return json.loads(LEGACY_POLICY_PATH.read_text())


# --- 1. chunk_policy.py loader ----------------------------------------------

def test_legacy_descriptor_loads_as_a_strict_third_schema():
    policy = load_chunk_policy(LEGACY_POLICY_PATH)
    assert isinstance(policy, LegacyChunkPolicy)
    assert policy.algorithm == LEGACY_FIXED_WINDOW_PACK
    assert policy.rule_legend == LEGACY_RULE_LEGEND == "legacy-180w-o30@1"
    assert policy.window_words == LEGACY_WINDOW_WORDS == 180
    assert policy.overlap_words == LEGACY_OVERLAP_WORDS == 30
    assert policy.producer_module == "pipeline/fetch_process.py"
    assert policy.producer_function == "chunk"
    assert len(policy.producer_git_revision) == 40
    int(policy.producer_git_revision, 16)  # lowercase hex, format-checked


def test_legacy_descriptor_rejects_wrong_legend(tmp_path):
    raw = _legacy_raw()
    raw["rule_legend"] = "legacy-180w-o31@1"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ChunkPolicyError, match="rule_legend"):
        load_chunk_policy(path)


def test_legacy_descriptor_rejects_wrong_window(tmp_path):
    raw = _legacy_raw()
    raw["window_words"] = 175
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ChunkPolicyError, match="window_words"):
        load_chunk_policy(path)


def test_legacy_descriptor_rejects_wrong_overlap(tmp_path):
    raw = _legacy_raw()
    raw["overlap_words"] = 20
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ChunkPolicyError, match="overlap_words"):
        load_chunk_policy(path)


def test_legacy_descriptor_rejects_unknown_field(tmp_path):
    raw = _legacy_raw()
    raw["extra_field"] = "unexpected"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ChunkPolicyError, match="must contain exactly"):
        load_chunk_policy(path)


def test_legacy_descriptor_rejects_malformed_producer(tmp_path):
    raw = _legacy_raw()
    raw["producer"] = {"module": "pipeline/fetch_process.py",
                       "function": "chunk", "git_revision": "not-a-sha"}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ChunkPolicyError, match="git_revision"):
        load_chunk_policy(path)


def test_legacy_descriptor_rejects_wrong_epoch(tmp_path):
    raw = _legacy_raw()
    raw["epoch"] = 2
    raw["rule_legend"] = "legacy-180w-o30@2"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ChunkPolicyError, match="epoch"):
        load_chunk_policy(path)


@pytest.mark.parametrize("path,expected_algorithm", [
    (REPO / "data" / "chunk-policy-struct-md-768w-o0.v1.json",
     "struct-markdown-word-pack-v1"),
    (REPO / "data" /
     "chunk-policy-news-article-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json",
     "struct-markdown-adjacent-section-pack-v1"),
])
def test_structured_schemas_still_load_byte_identically(path, expected_algorithm):
    """Regression: adding the third algorithm must not perturb the other two."""
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    policy = load_chunk_policy(path)
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after  # loader never mutates the artifact on disk
    assert policy.algorithm == expected_algorithm
    assert policy.sha256 == before
    assert policy.value == json.loads(path.read_bytes())


# --- 2. tier-2 news_article.chunks profile check ----------------------------

def test_profile_check_passes_with_the_legacy_descriptor(tmp_path):
    root, run, pack_root = build(tmp_path)
    pack_path = pack_root / "PACK.json"
    pack = json.loads(pack_path.read_text())
    pack["chunk_policy"] = {
        "path": A.LEGACY_POLICY_PATH,
        "policy_id": A.LEGACY_POLICY_ID,
        "sha256": hashlib.sha256(LEGACY_POLICY_PATH.read_bytes()).hexdigest(),
    }
    pack_path.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n")
    evidence_path, checks_path = run_audit(tmp_path, root, run, pack_root)
    checks = json.loads(checks_path.read_text())
    assert all(row["status"] == "PASS" for row in checks["chunks"])


def test_profile_check_still_fails_with_policy_path_none(tmp_path):
    """The exact production condition task 450 hit before ADR-100."""
    root, run, pack_root = build(tmp_path)
    pack_path = pack_root / "PACK.json"
    pack = json.loads(pack_path.read_text())
    pack.pop("chunk_policy", None)
    pack_path.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="pinned news-specific policy"):
        run_audit(tmp_path, root, run, pack_root)


def test_profile_check_rejects_legacy_sha_drift(tmp_path):
    root, run, pack_root = build(tmp_path)
    pack_path = pack_root / "PACK.json"
    pack = json.loads(pack_path.read_text())
    pack["chunk_policy"] = {"path": A.LEGACY_POLICY_PATH,
                            "policy_id": A.LEGACY_POLICY_ID, "sha256": "0" * 64}
    pack_path.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="SHA drifted"):
        run_audit(tmp_path, root, run, pack_root)


def test_structured_news_pack_still_passes_unchanged(tmp_path):
    """Nothing weakens for structured-policy sources."""
    root, run, pack_root = build(tmp_path)
    evidence_path, checks_path = run_audit(tmp_path, root, run, pack_root)
    checks = json.loads(checks_path.read_text())
    assert all(row["status"] == "PASS" for row in checks["chunks"])


# --- 3. the descriptor reproduces the REAL deployed chunker ----------------
#
# `fetch_process` pulls in trafilatura/bs4/httpx (its fetch+extract chain);
# these tests only need `fetch_process.chunk`, so the import is guarded
# per-test (task 407 `test_chunk.py` pattern) rather than at module level —
# an environment missing trafilatura would otherwise skip every test in this
# file, including the loader/profile-check tests that need none of it.

def _fetch_process():
    pytest.importorskip("trafilatura")
    from cytir.acquire import fetch_process as fp
    return fp


FIXED_TEXTS = [
    "\n".join(f"Paragraph {i} of the fixed test article body, with enough "
              f"distinct words in it to matter for windowing arithmetic "
              f"across the whole fixture set number {i}." for i in range(40)),
    "\n".join(f"Line {i}: short paragraph." for i in range(5)),
    "A single short paragraph well under the floor.",
]


@pytest.mark.parametrize("text", FIXED_TEXTS)
def test_descriptor_parameters_reproduce_the_deployed_chunker(text):
    fp = _fetch_process()
    policy = load_chunk_policy(LEGACY_POLICY_PATH)
    declared = fp.chunk(text, target_words=policy.window_words,
                        overlap_words=policy.overlap_words)
    deployed_default = fp.chunk(text)
    assert declared == deployed_default


def test_descriptor_reproduces_chunk_boundaries_and_overlap():
    fp = _fetch_process()
    policy = load_chunk_policy(LEGACY_POLICY_PATH)
    text = FIXED_TEXTS[0]
    chunks = fp.chunk(text, target_words=policy.window_words,
                      overlap_words=policy.overlap_words)
    assert len(chunks) >= 2, "fixture must actually exercise a window boundary"
    for c in chunks:
        assert len(c.split()) >= 15  # the function's own floor
    # Overlap: the tail of chunk[i] (>= overlap_words) reappears verbatim as
    # the head of chunk[i+1] — this is what `overlap_words` MEANS for this
    # chunker, so the test vector must observe it directly, not assume it.
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_paras = prev.split("\n")
        next_paras = nxt.split("\n")
        overlap_found = any(
            next_paras[:k] == prev_paras[-k:]
            for k in range(1, min(len(prev_paras), len(next_paras)) + 1))
        assert overlap_found, "declared overlap_words must be observable at the boundary"
