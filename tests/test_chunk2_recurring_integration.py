"""Permanence guards for generic recurring chunk@2 ingestion."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from cytir.core import common as aw_common
from cytir.chunk import chunker as chunk
from cytir.chunk.policy import DEFAULT_POLICY_PATH, load_chunk_policy
from cytir.sources.cov import engine as ingest_cov_docs
from cytir.chunk import generic as generic_chunking


POLICY = load_chunk_policy(DEFAULT_POLICY_PATH)
TOKENIZER_REVISION = "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"


def _select_structured(monkeypatch):
    monkeypatch.setenv("AW_CHUNKER", "structured")
    monkeypatch.setenv("AW_CHUNK_POLICY", str(POLICY.path))
    monkeypatch.setenv("AW_CHUNK_POLICY_SHA256", POLICY.sha256)
    monkeypatch.setenv("AW_CHUNK_TOKENIZER_REVISION", TOKENIZER_REVISION)


def _expected_stamps() -> dict:
    """task 407: the per-row stamp is the epoch PLUS the interned rule id.

    Derived from the live selection rather than hardcoded, because the whole
    point of `prov_id` is that it changes when any rule byte changes — a
    hardcoded literal here would have to be edited on every policy bump and
    would silently stop testing the binding.
    """
    return aw_common.chunker_selection().stamps()


def test_coverage_pdf_path_uses_chunk2_dispatch(monkeypatch, word_only_chunking_allowed):
    _select_structured(monkeypatch)
    md = "# Section\n\n" + " ".join(f"word{i}" for i in range(900))
    units = ingest_cov_docs.chunk_generic_records(md)
    assert len(units) >= 2
    assert all(unit.stamps == _expected_stamps() for unit in units)
    assert chunk.active_policy().sha256 == POLICY.sha256
    assert chunk.policy_matches_effective(POLICY)


def test_size_specific_mode_name_is_rejected(monkeypatch):
    monkeypatch.setenv("AW_CHUNKER", "struct768")
    with pytest.raises(ValueError, match="unknown"):
        aw_common.chunker_selection()


@pytest.mark.parametrize("missing", ["AW_CHUNK_POLICY", "AW_CHUNK_POLICY_SHA256"])
def test_structured_candidate_requires_policy_path_and_sha(monkeypatch, missing):
    _select_structured(monkeypatch)
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match="requires"):
        aw_common.chunker_selection()


def test_structured_candidate_rejects_stale_policy_sha(monkeypatch):
    _select_structured(monkeypatch)
    monkeypatch.setenv("AW_CHUNK_POLICY_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA mismatch"):
        aw_common.chunker_selection()


def test_structured_candidate_requires_immutable_tokenizer_revision(monkeypatch):
    _select_structured(monkeypatch)
    monkeypatch.delenv("AW_CHUNK_TOKENIZER_REVISION")

    with pytest.raises(ValueError, match="chunk_tokenizer_revision"):
        aw_common.chunker_selection()


def _write_profile(path: Path, **overrides) -> None:
    value = {
        "chunker_mode": "structured",
        "chunk_epoch": POLICY.epoch,
        "chunk_rule_legend": POLICY.rule_legend,
        "chunk_policy_path": str(POLICY.path),
        "chunk_policy_sha256": POLICY.sha256,
        "chunk_tokenizer_revision": TOKENIZER_REVISION,
    }
    value.update(overrides)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_structured_profile_binds_policy_sha_epoch_and_legend(monkeypatch, tmp_path, word_only_chunking_allowed):
    monkeypatch.delenv("AW_CHUNKER", raising=False)
    profile = tmp_path / "build.json"
    _write_profile(profile)
    monkeypatch.setenv("AW_BUILD_CONFIG", str(profile))

    selected = aw_common.chunker_selection()
    assert selected.mode == "structured"
    assert selected.policy_path == str(POLICY.path)
    assert selected.policy_sha256 == POLICY.sha256
    assert selected.tokenizer_revision == TOKENIZER_REVISION
    assert (selected.epoch, selected.legend) == (
        POLICY.epoch, POLICY.rule_legend)

    units = aw_common.chunk_markdown_channels("# Title\n\n" + "word " * 100)
    assert units
    assert all(unit.stamps == _expected_stamps() for unit in units)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("chunk_epoch", 999, "policy requires"),
        ("chunk_rule_legend", "wrong@2", "policy requires"),
        ("chunk_policy_sha256", "f" * 64, "SHA mismatch"),
    ],
)
def test_structured_profile_rejects_policy_inconsistency(
        monkeypatch, tmp_path, field, value, match):
    monkeypatch.delenv("AW_CHUNKER", raising=False)
    profile = tmp_path / "build.json"
    _write_profile(profile, **{field: value})
    monkeypatch.setenv("AW_BUILD_CONFIG", str(profile))
    with pytest.raises(ValueError, match=match):
        aw_common.chunker_selection()


def test_live_profile_remains_legacy_without_policy_activation(monkeypatch):
    monkeypatch.delenv("AW_CHUNKER", raising=False)
    monkeypatch.delenv("AW_BUILD_CONFIG", raising=False)
    selected = aw_common.chunker_selection()
    assert selected.mode == "legacy"
    assert selected.epoch == 1
    assert selected.legend == "legacy-180w-o30@1"
    assert selected.policy_path is None
    assert selected.policy_sha256 is None


# Moved here from the generic chunking module (task 764): it names private
# `pipeline/` scripts, so it lives with the test that reads them.
# This small, reviewed registry makes the permanence test a routing contract,
# rather than an aspirational grep in a task note.  The list names only generic
# prose/doc producers; law/article-specific splitters are intentionally absent.
GENERIC_DIRECT_PRODUCERS = (
    # task 1901: fetch_corpus.py (dead P0 fetch stage, never run) deleted
    # task 1396: ingest_finma_circulars.py moved to the package (MOVED_DIRECT_PRODUCERS)
    # task 1235: the two producers below moved into the package and are read from there
)
MOVED_DIRECT_PRODUCERS = ("cytir/sources/cov/engine.py", "cytir/extract/service_public_fiche.py", "cytir/extract/wikipedia_dump.py",
                          "cytir/sources/finma/circulars.py")


def test_every_direct_generic_producer_routes_through_the_profile_dispatcher():
    """Keep future direct Passage writers from silently reviving chunk@1.

    HTML/CC/pack producers already converge through ``aw_common.process_html``;
    this covers the small set that construct generic prose/doc Passages directly.
    Native-law article splitters are intentionally not registered (ADR-079).
    """
    root = Path(__file__).resolve().parent
    for name in GENERIC_DIRECT_PRODUCERS:
        source = (root / name).read_text(encoding="utf-8")
        assert "from cytir.chunk.generic import chunk_generic_records" in source
        assert "passage_from_chunk" in source
        assert ("passage_source_record" in source or
                "passage_corpus_record" in source)
        assert "fp.chunk(" not in source
    for rel in MOVED_DIRECT_PRODUCERS:
        source = (root.parent / rel).read_text(encoding="utf-8")
        assert "from cytir.chunk.generic import chunk_generic_records" in source
        assert "passage_from_chunk" in source
        assert "passage_source_record" in source or "passage_corpus_record" in source
        assert "fp.chunk(" not in source
    moved = (root.parent / "cytir" / "extract" / "hf_model_cards.py").read_text(encoding="utf-8")
    assert "from cytir.chunk.generic import chunk_generic_records" in moved   # task 1101: moved out
    assert "passage_from_chunk" in moved and "passage_source_record" in moved
    # task 1237: the docs-repo and tech-docs renderers moved into the package
    tree = (root.parent / "cytir" / "acquire" / "github_docs_tree.py").read_text(encoding="utf-8")
    assert "from cytir.chunk.generic import chunk_generic_records" in tree
    assert "passage_from_chunk" in tree and "passage_corpus_record" in tree
    assert "fp.chunk(" not in tree


def test_structured_channels_keep_identity_surface_and_trim_dense_hierarchy(monkeypatch, word_only_chunking_allowed):
    _select_structured(monkeypatch)
    md = "# Global title\n\n## Local section\n\n" + "body fact " * 30
    units = generic_chunking.chunk_generic_records(md)
    assert len(units) == 1
    unit = units[0]
    assert unit.text.startswith("Global title > Local section\n")
    assert unit.sparse_text == unit.text
    assert unit.dense_text.startswith("Local section\n")
    assert "Global title" not in unit.dense_text

    passage = aw_common.passage_from_chunk(
        unit, url="https://example.test/doc", title="Doc", snippet="body",
        lang="en", domain="example.test", tier="B", source_type="docs",
        trust_tier=0.8)
    expected_id = aw_common.Passage(
        url=passage.url, title=passage.title, snippet=passage.snippet,
        text=unit.text, lang=passage.lang, domain=passage.domain,
        tier=passage.tier, source_type=passage.source_type,
        trust_tier=passage.trust_tier, stamps=unit.stamps).point_id
    assert passage.point_id == expected_id
    assert aw_common.passage_source_record(passage)["dense_text"] == unit.dense_text
    assert aw_common.passage_corpus_record(passage)["sparse_text"] == unit.text


def test_legacy_serializers_do_not_emit_null_channel_keys():
    unit = aw_common.ChunkTextChannels("legacy body", None, None, None)
    passage = aw_common.passage_from_chunk(
        unit, url="https://example.test/legacy", title="Legacy", snippet="body",
        lang="en", domain="example.test", tier="B", source_type="docs",
        trust_tier=0.8)
    assert "dense_text" not in aw_common.passage_source_record(passage)
    assert "sparse_text" not in aw_common.passage_source_record(passage)
    assert "dense_text" not in aw_common.passage_corpus_record(passage)
    assert "sparse_text" not in aw_common.passage_corpus_record(passage)


def test_html_funnel_uses_channel_dispatch_and_channel_aware_factory():
    source = (Path(aw_common.__file__).resolve().parents[1] / "extract" / "html_extract.py").read_text(encoding="utf-8")
    body = source[source.index("def _content_passages"):]
    assert "chunk_markdown_channels(md, selection)" in body
    assert "passage_from_chunk(" in body
