"""Wikipedia dump loader: chunk-anchor identity contract (moved from the shared
chunk@2 recurring-integration tests, task 1235)."""
from __future__ import annotations

from cytir.core import common as aw_common
from cytir.chunk import generic as generic_chunking
from cytir.chunk.policy import DEFAULT_POLICY_PATH, load_chunk_policy
from cytir.extract import wikipedia_dump as wiki_dump

POLICY = load_chunk_policy(DEFAULT_POLICY_PATH)
TOKENIZER_REVISION = "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"


def _select_structured(monkeypatch):
    monkeypatch.setenv("AW_CHUNKER", "structured")
    monkeypatch.setenv("AW_CHUNK_POLICY", str(POLICY.path))
    monkeypatch.setenv("AW_CHUNK_POLICY_SHA256", POLICY.sha256)
    monkeypatch.setenv("AW_CHUNK_TOKENIZER_REVISION", TOKENIZER_REVISION)


def _expected_stamps() -> dict:
    return aw_common.chunker_selection().stamps()


def test_wiki_dump_preserves_legacy_anchor_and_point_identity(monkeypatch):
    """The live legacy profile must remain byte/identity-compatible."""
    monkeypatch.delenv("AW_CHUNKER", raising=False)
    monkeypatch.delenv("AW_BUILD_CONFIG", raising=False)
    assert aw_common.chunker_selection().mode == "legacy"
    # The legacy splitter has optional extraction dependencies; isolate the
    # profile dispatch contract from those dependencies here.
    monkeypatch.setattr(
        generic_chunking, "chunk_markdown_channels", lambda _md: [
            aw_common.ChunkTextChannels("legacy", None, None, None)])
    units = generic_chunking.chunk_generic_records("legacy wiki chunk")
    stamps = units[0].stamps
    assert stamps is None
    text = "legacy wiki chunk"
    # task 407: a REAL legacy row is now stamped `{"chunk": 1, "prov_id": …}`.
    # Identity must key on the epoch, not on the stamp's absence, or stamping
    # legacy rows would re-mint every legacy wiki point id.
    legacy_stamps = aw_common.chunker_selection().stamps()
    assert legacy_stamps["chunk"] == 1
    assert wiki_dump.chunk_anchor(0, legacy_stamps) is None
    assert wiki_dump.chunk_anchor(3, legacy_stamps) == "chunk-3"
    assert wiki_dump.chunk_anchor(0, stamps) is None
    assert wiki_dump.chunk_anchor(3, stamps) == "chunk-3"
    before = aw_common.Passage(
        url="https://en.wikipedia.org/wiki/Example", title="Example",
        snippet="legacy", text=text, lang="en", domain="wikipedia.org",
        tier="B", source_type="reference", trust_tier=0.82,
        anchor="chunk-3")
    after = aw_common.Passage(
        url="https://en.wikipedia.org/wiki/Example", title="Example",
        snippet="legacy", text=text, lang="en", domain="wikipedia.org",
        tier="B", source_type="reference", trust_tier=0.82,
        anchor=wiki_dump.chunk_anchor(3, stamps))
    assert after.point_id == before.point_id
    assert after.payload() == before.payload()


def test_wiki_dump_structured_chunks_have_epoch_stamp_not_ordinal_anchor(monkeypatch, word_only_chunking_allowed):
    """Epoch-2 wiki chunks are Tier B, so insertion cannot re-point IDs."""
    _select_structured(monkeypatch)
    units = generic_chunking.chunk_generic_records(
        "# Global\n\n## Local\n\n" + "structured wiki chunk " * 20)
    stamps = units[0].stamps
    assert POLICY.epoch == 2
    assert stamps == _expected_stamps()
    assert stamps["chunk"] == 2
    # the stamp names the exact policy, not just the epoch family
    assert stamps["prov_id"] == aw_common.chunker_selection().provenance()["prov_id"]
    assert wiki_dump.chunk_anchor(0, stamps) is None
    assert wiki_dump.chunk_anchor(3, stamps) is None
    p = aw_common.Passage(
        url="https://en.wikipedia.org/wiki/Example", title="Example",
        snippet="structured", text="structured wiki chunk", lang="en",
        domain="wikipedia.org", tier="B", source_type="reference",
        trust_tier=0.82, anchor=wiki_dump.chunk_anchor(3, stamps),
        stamps=stamps)
    assert "anchor" not in p.payload()
    assert p.payload()["stamps"] == stamps
