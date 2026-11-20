"""Task 601: chunk.generic (epoch-4) chunking for HN-discussion records."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cytir.chunk import chunker as CHUNK
from cytir.chunk import hn_discussion as subject


POLICY = (Path(__file__).resolve().parent.parent / "data" /
          "chunk-policy-struct-md-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json")


@pytest.fixture(autouse=True)
def _select_policy():
    # `chunk.py`'s active policy is process-global state, shared with every
    # other test module in the same pytest session -- restore it so this
    # module's tests never leak the pinned hn_discussion policy into
    # test_chunk.py's own default-policy assumptions.
    previous = CHUNK.active_policy()
    CHUNK.select_policy(POLICY, allow_env_overrides=False)
    try:
        yield
    finally:
        CHUNK._apply_policy(previous, allow_env_overrides=True)


def _record(text: str, url: str = "https://news.ycombinator.com/item?id=1") -> dict:
    return {"url": url, "title": "t", "text": text, "snippet": "s",
            "lang": "en", "hn_object_id": "1"}


def test_epoch4_is_the_pinned_chunk_generic_policy():
    assert CHUNK.active_policy().epoch == 4
    assert CHUNK.active_policy().policy_id == (
        "struct-md-768w-o0-adjacent-sections-tail-1152-tok2560")


def test_short_discussion_stays_one_chunk(word_only_chunking_allowed):
    text = "Title line\n\n" + "\n\n".join(
        f"short comment {i}" for i in range(3))
    rows = subject.chunk_record(_record(text))
    assert len(rows) == 1
    assert rows[0]["chunk_ordinal"] == 0
    assert rows[0]["chunk_count"] == 1
    assert rows[0]["text"] == text


def test_long_discussion_genuinely_splits_between_comment_blocks(word_only_chunking_allowed):
    """Mechanism check for specs/hn-discussion-embedding.md Q3: a discussion
    comfortably over the hard-max word guard splits BETWEEN whole comment
    blocks, never mid-comment (each comment here is its own unique marker, so
    a mid-comment cut would show up as a marker split across two chunks)."""
    comment = ("padding word " * 90).strip()  # ~180 words/comment
    comments = [f"COMMENT_{i}_START {comment} COMMENT_{i}_END" for i in range(12)]
    text = "Title line\n\n" + "\n\n".join(comments)
    rows = subject.chunk_record(_record(text))

    assert len(rows) > 1
    assert all(row["chunk_count"] == len(rows) for row in rows)
    assert [row["chunk_ordinal"] for row in rows] == list(range(len(rows)))
    # every comment appears whole, in exactly one chunk -- never split, never
    # duplicated, never lost.
    for i in range(12):
        start = f"COMMENT_{i}_START"
        end = f"COMMENT_{i}_END"
        holders = [row for row in rows if start in row["text"]]
        assert len(holders) == 1, f"comment {i} split across chunks"
        assert end in holders[0]["text"]
    parents = {row["parent_markdown_sha256"] for row in rows}
    assert len(parents) == 1  # every chunk traces to the SAME parent document


def test_chunk_provenance_never_merges_two_different_discussions(tmp_path, word_only_chunking_allowed):
    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "chunks.jsonl"
    records = [
        _record("Title A\n\n" + "\n\n".join(f"a comment {i}" for i in range(3)),
               url="https://news.ycombinator.com/item?id=1"),
        _record("Title B\n\n" + "\n\n".join(f"b comment {i}" for i in range(3)),
               url="https://news.ycombinator.com/item?id=2"),
    ]
    input_path.write_text("".join(json.dumps(r) + "\n" for r in records))

    result = subject.chunk_file(input_path, output_path, policy_path=POLICY)

    assert result["documents"] == 2
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert result["chunks"] == len(rows)
    by_parent = {}
    for row in rows:
        by_parent.setdefault(row["parent_markdown_sha256"], set()).add(row["url"])
    for urls in by_parent.values():
        assert len(urls) == 1  # a parent digest never spans two discussions


def test_empty_text_refuses_rather_than_silently_dropping():
    with pytest.raises(subject.ChunkingError):
        subject.chunk_record(_record(""))


def test_chunk_rows_carry_rule_epochs_for_stage_pack_b1_preserve_rule_epochs(word_only_chunking_allowed):
    """A non-baseline chunk policy (epoch-4, never the b1-native single-chunk
    news exception) must be provable at staging time
    (`stage_pack_b1.py --preserve-rule-epochs` /
    `aw_b1_projection.project_record`), not silently fall back to whatever
    the CURRENT baseline default is -- the exact fossil-record failure ADR-090
    amendment 1 / task 565 named."""
    import cytir.core.projection as PROJ

    long_text = "Title\n\n" + "\n\n".join(
        f"COMMENT_{i} {'padding word ' * 90}" for i in range(12))
    record = _record(long_text)
    record.update(lang="en", tier="C", trust_tier=0.6, source_type="reference",
                  license="api-only", search_ok=True, train_ok=False)
    rows = subject.chunk_record(record)
    assert len(rows) > 1
    for row in rows:
        assert row["rule_epochs"] == {"chunk": 4}
        result = PROJ.project_record(
            row, consent={"consent_state": "resolved",
                          "consent_basis": "official-hn-firebase-api-only",
                          "consent_source": "official Firebase API",
                          "consent_checked_at": "2026-09-17"},
            preserve_rule_epochs=True)
        assert result is not None
        _, _, _, payload = result
        assert payload["rule_epochs"]["chunk"] == 4
        assert payload["stamps"] == {"chunk": 4, "prov_id": None}


def test_seal_chunks_stage_is_deep_valid_and_matches_chunk_file_count(tmp_path, word_only_chunking_allowed):
    from cytir.index import preservation_contract as PC

    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "chunks.jsonl"
    seal_root = tmp_path / "pack"
    long_text = "Title\n\n" + "\n\n".join(
        f"COMMENT_{i} {'padding word ' * 90}" for i in range(12))
    records = [_record(long_text), _record("Title B\n\ncomment", url="https://news.ycombinator.com/item?id=2")]
    input_path.write_text("".join(json.dumps(r) + "\n" for r in records))

    flat = subject.chunk_file(input_path, output_path, policy_path=POLICY)
    manifest = subject.seal_chunks_stage(
        seal_root, input_path=input_path, policy_path=POLICY)

    assert manifest["counts"]["documents"] == 2
    assert manifest["counts"]["chunks"] == flat["chunks"]
    PC._validate_stage(manifest["stages"]["chunks"], "chunks",
                       artifact_root=seal_root, deep=True)


def test_seal_chunks_stage_relative_to_a_shared_artifact_root(tmp_path, word_only_chunking_allowed):
    from cytir.index import preservation_contract as PC

    input_path = tmp_path / "records.jsonl"
    artifact_root = tmp_path / "shared-root"
    pack_root = artifact_root / "stage-2026-09-17" / "preservation-pack"
    input_path.write_text(json.dumps(_record("Title\n\ncomment")) + "\n")

    manifest = subject.seal_chunks_stage(
        pack_root, input_path=input_path, policy_path=POLICY,
        seal_root=artifact_root)

    assert manifest["stages"]["chunks"]["receipt"]["uri"].startswith(
        "artifact://stage-2026-09-17/preservation-pack/")
    PC._validate_stage(manifest["stages"]["chunks"], "chunks",
                       artifact_root=artifact_root, deep=True)
    assert (pack_root / "PACK.json").is_file()


@pytest.mark.parametrize("entry", ["chunk_file", "seal_chunks_stage"])
def test_write_paths_refuse_word_only_chunking(monkeypatch, tmp_path, entry):
    """task 2289: no tokenizer -> refuse with a named reason before any row is written."""
    monkeypatch.setattr(CHUNK, "tokenizer_status", lambda: "unavailable")
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(_record("hello world " * 50)) + "\n")
    with pytest.raises(RuntimeError, match="hn_discussion: exact embed tokenizer unavailable"):
        if entry == "chunk_file":
            subject.chunk_file(src, tmp_path / "out.jsonl", policy_path=POLICY)
        else:
            subject.seal_chunks_stage(tmp_path / "pack", input_path=src, policy_path=POLICY)
    assert not (tmp_path / "out.jsonl").exists()


def test_word_only_flag_is_false_and_only_the_fixture_writes_it():
    """task 2289 ruling: production never sets the test-only opt-in."""
    import subprocess
    assert CHUNK._ALLOW_WORD_ONLY_FOR_TESTS is False
    root = Path(__file__).resolve().parent.parent
    hits = subprocess.run(["git", "grep", "-n", "_ALLOW_WORD_ONLY_FOR_TESTS", "--", "cytir"],
                          cwd=root, capture_output=True, text=True).stdout.splitlines()
    assert all(h.startswith("cytir/chunk/chunker.py:") for h in hits), hits
    assert not [h for h in hits if "= True" in h or "global" in h], hits


def test_funnel_refuses_word_only_chunking(monkeypatch):
    """task 2289: the structured branch of chunk_markdown_channels preflights the tokenizer."""
    from cytir.core import common as C
    from cytir.core.model_pin import EMBED_MODEL_REVISION
    monkeypatch.setattr(CHUNK, "tokenizer_status", lambda: "unavailable")
    sel = C.explicit_structured_chunker_selection(
        str(POLICY), CHUNK.active_policy().sha256, EMBED_MODEL_REVISION)
    with pytest.raises(RuntimeError, match="chunk_markdown_channels: exact embed tokenizer unavailable"):
        C.chunk_markdown_channels("# t\n\nsome text\n", sel)
