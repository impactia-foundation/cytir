import gzip
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from cytir.acquire.artifact_io import canonical_json
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as seal_stage_receipt
from cytir.corpus import stage_structured_markdown as subject
from cytir.core.common import (ChunkerSelection, Passage,
                       explicit_structured_chunker_selection,
                       mint_passage_point_id)
from cytir.chunk import chunker as aw_chunk
from cytir.chunk.generic import chunk_generic_records


def test_structured_pack_conserves_docs_and_measures_chunks(tmp_path: Path,
                                                            monkeypatch):
    parent = tmp_path / "parent"
    row = {"document_id": "d1", "url": "https://example.test/d1",
           "lang": "fr", "markdown": "# Titre\n\nUn texte.",
           "title": "Titre", "source_relative_path": "input/d1.md",
           "source_sha256": "c" * 64,
           "site": "stackoverflow.com", "answer_url": "https://stackoverflow.com/a/2",
           "question": {"post_id": "1"},
           "accepted_answer": {"post_id": "2"},
           "product": {"sku": "R-1", "identity_kind": "source_sku_or_mpn"}}
    logical = (canonical_json(row) + "\n").encode()
    artifact = parent / "stages/markdown/data/docs.jsonl.gz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(gzip.compress(logical, mtime=0))
    stage = seal_stage_receipt.seal(
        artifact_root=tmp_path, artifact_paths=[artifact],
        receipt_path=parent / "stages/markdown/receipt.json", records=1,
        fmt="canonical-jsonl+deterministic-gzip",
        logical_sha256=hashlib.sha256(logical).hexdigest(),
        producer_id="fixture", producer_version="v1")
    run = parent / "RUN.json"
    run.write_text(json.dumps({"stages": {"markdown": stage}}))
    monkeypatch.setattr(subject, "chunk_generic_records", lambda markdown, selection=None: [
        SimpleNamespace(text="# Titre\n\nUn", dense_text="# Titre\nUn",
                        sparse_text="# Titre\n\nUn", stamps={}),
        SimpleNamespace(text="texte.", dense_text="texte.", sparse_text="texte.",
                        stamps={}),
    ])
    fixture_selection = ChunkerSelection("structured", 4, "p@4",
                                         str(tmp_path / "policy.json"), "f" * 64,
                                         "a" * 40)
    monkeypatch.setattr(subject, "_preflight_chunker", lambda *args: (
        fixture_selection, {"tokenizer_class": "FixtureFastTokenizer",
                            "tokenizer_is_fast": True}))
    monkeypatch.setattr(subject.aw_chunk, "token_len", lambda text: len(text.split()))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"policy_id": "p", "epoch": 4,
                                  "rule_legend": "p@4",
                                  "token_hard_max": {"exact": 2560}}))
    out = tmp_path / "pack"
    result = subject.stage(
        artifact_root=tmp_path,
        parent_root=parent, parent_run=run, output_root=out,
        policy_path=policy, tokenizer_revision="a" * 40,
        producer_revision="b" * 64, source_profile="software_docs",
        rows_per_shard=1)
    assert result["counts"] == {"documents": 1, "chunks": 2,
                                "measured_chunks_per_document": 2.0}
    assert result["source_profile"] == "software_docs"
    assert result["token_distribution"]["histogram"] == [[1, 1], [3, 1]]
    assert sum(count for _, count in result["token_distribution"]["histogram"]) == 2
    with gzip.open(out / "chunks/shard-0000.texts.jsonl.gz", "rt") as source:
        emitted = [json.loads(line) for line in source]
    assert emitted[0]["point_id"] == mint_passage_point_id(
        row["url"], "document:d1:chunk-0", emitted[0]["canonical_text"])
    assert emitted[0]["parent_markdown_sha256"] == hashlib.sha256(
        row["markdown"].encode()).hexdigest()
    assert emitted[0]["section_heading"] == "# Titre"
    assert emitted[0]["source_metadata"] == {
        "title": "Titre", "source_relative_path": "input/d1.md",
        "source_sha256": "c" * 64, "site": "stackoverflow.com",
        "answer_url": "https://stackoverflow.com/a/2",
        "question": {"post_id": "1"},
        "accepted_answer": {"post_id": "2"},
        "product": {"sku": "R-1", "identity_kind": "source_sku_or_mpn"}}
    verified = _validate_stage(result["chunks"], "pack.chunks",
                               artifact_root=tmp_path, deep=True)
    assert verified["records"] == 2
    assert len([artifact for artifact in verified["artifacts"]
                if artifact["uri"].endswith(".jsonl.gz")]) == 2


def test_structured_pack_identical_chunks_get_distinct_deterministic_ids(
        tmp_path: Path, monkeypatch):
    parent = tmp_path / "parent"
    row = {"document_id": "d1", "url": "https://example.test/d1",
           "lang": "en", "markdown": "# Repeated\n\nbody"}
    logical = (canonical_json(row) + "\n").encode()
    artifact = parent / "stages/markdown/data/docs.jsonl.gz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(gzip.compress(logical, mtime=0))
    stage = seal_stage_receipt.seal(
        artifact_root=tmp_path, artifact_paths=[artifact],
        receipt_path=parent / "stages/markdown/receipt.json", records=1,
        fmt="canonical-jsonl+deterministic-gzip",
        logical_sha256=hashlib.sha256(logical).hexdigest(),
        producer_id="fixture", producer_version="v1")
    run = parent / "RUN.json"
    run.write_text(json.dumps({"stages": {"markdown": stage}}))
    unit = SimpleNamespace(text="same", dense_text="same", sparse_text="same",
                           stamps={})
    monkeypatch.setattr(subject, "chunk_generic_records",
                        lambda markdown, selection=None: [unit, unit])
    fixture_selection = ChunkerSelection("structured", 4, "p@4",
                                         str(tmp_path / "policy.json"), "f" * 64,
                                         "a" * 40)
    monkeypatch.setattr(subject, "_preflight_chunker", lambda *args: (
        fixture_selection, {"tokenizer_class": "FixtureFastTokenizer",
                            "tokenizer_is_fast": True}))
    monkeypatch.setattr(subject.aw_chunk, "token_len", lambda text: len(text.split()))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"policy_id": "p", "epoch": 4,
                                  "rule_legend": "p@4",
                                  "token_hard_max": {"exact": 2560}}))
    out = tmp_path / "pack"
    first = subject.stage(
        artifact_root=tmp_path,
        parent_root=parent, parent_run=run, output_root=out,
        policy_path=policy, tokenizer_revision="a" * 40,
        producer_revision="b" * 64)
    second = subject.stage(
        artifact_root=tmp_path,
        parent_root=parent, parent_run=run, output_root=out,
        policy_path=policy, tokenizer_revision="a" * 40,
        producer_revision="b" * 64)
    assert first == second
    with gzip.open(out / "chunks/shard-0000.texts.jsonl.gz", "rt") as source:
        ids = [json.loads(line)["point_id"] for line in source]
    assert len(ids) == len(set(ids)) == 2
    assert ids == [
        mint_passage_point_id(row["url"], "document:d1:chunk-0", "same"),
        mint_passage_point_id(row["url"], "document:d1:chunk-1", "same"),
    ]


def test_explicit_dispatcher_does_not_read_env_or_build_config(monkeypatch, tmp_path: Path, word_only_chunking_allowed):
    policy = Path(__file__).resolve().parents[1] / "data" / (
        "chunk-policy-struct-md-768w-o0-adjacent-sections-tail-1152-tok2560.v1.json")
    policy_sha = hashlib.sha256(policy.read_bytes()).hexdigest()
    for name in ("AW_CHUNKER", "AW_CHUNK_POLICY", "AW_CHUNK_POLICY_SHA256",
                 "AW_CHUNK_TOKENIZER_REVISION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AW_BUILD_CONFIG", str(tmp_path / "does-not-exist.json"))
    monkeypatch.setattr(aw_chunk, "_token_len", lambda text: len(text.split()))
    previous_policy = aw_chunk.active_policy()
    previous_revision = aw_chunk.TOKENIZER_REVISION
    try:
        selection = explicit_structured_chunker_selection(
            str(policy), policy_sha, "a" * 40)
        rows = chunk_generic_records("# Title\n\nA short complete document.",
                                     selection=selection)
        assert len(rows) == 1
        assert rows[0].stamps["chunk"] == 4
    finally:
        aw_chunk.select_tokenizer_revision(previous_revision)
        aw_chunk.select_policy(previous_policy.path, allow_env_overrides=True)


def test_anchored_law_requires_and_preserves_source_archive():
    archive = {"member": "legi/article.xml", "sha256": "a" * 64}
    assert subject._source_parent_identity(
        {"source_archive": archive}, "anchored_law", 7) == archive
    try:
        subject._source_parent_identity({}, "anchored_law", 8)
    except subject.StructuredStageError as exc:
        assert "source parent identity" in str(exc)
    else:
        raise AssertionError("anchored-law row without parent identity accepted")


def test_reboot_resume_admits_only_closed_shards_and_rejoins_parent_boundary(
        tmp_path: Path, monkeypatch):
    parent = tmp_path / "parent"; artifact = parent / "stages/markdown/data/docs.jsonl.gz"
    artifact.parent.mkdir(parents=True)
    rows = [{"document_id": f"d{index}", "url": f"https://example.test/d{index}",
             "lang": "en", "markdown": f"# Doc {index}\n\nBody {index}"}
            for index in range(1, 4)]
    logical = b"".join((canonical_json(row) + "\n").encode() for row in rows)
    artifact.write_bytes(gzip.compress(logical, mtime=0))
    stage = seal_stage_receipt.seal(
        artifact_root=tmp_path, artifact_paths=[artifact],
        receipt_path=parent / "stages/markdown/receipt.json", records=3,
        fmt="canonical-jsonl+deterministic-gzip",
        logical_sha256=hashlib.sha256(logical).hexdigest(),
        producer_id="fixture", producer_version="v1")
    run = parent / "RUN.json"; run.write_text(json.dumps({"stages": {"markdown": stage}}))
    monkeypatch.setattr(subject, "chunk_generic_records", lambda markdown, selection=None: [
        SimpleNamespace(text=markdown + " A", dense_text=markdown + " A",
                        sparse_text=markdown + " A", stamps={}),
        SimpleNamespace(text=markdown + " B", dense_text=markdown + " B",
                        sparse_text=markdown + " B", stamps={})])
    fixture_selection = ChunkerSelection("structured", 4, "p@4",
                                         str(tmp_path / "policy.json"), "f" * 64,
                                         "a" * 40)
    monkeypatch.setattr(subject, "_preflight_chunker", lambda *args: (
        fixture_selection, {"tokenizer_class": "FixtureFastTokenizer",
                            "tokenizer_is_fast": True}))
    monkeypatch.setattr(subject.aw_chunk, "token_len", lambda text: len(text.split()))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"policy_id": "p", "epoch": 4,
                                  "rule_legend": "p@4",
                                  "token_hard_max": {"exact": 2560}}))
    baseline_root = tmp_path / "baseline"
    baseline = subject.stage(
        artifact_root=tmp_path, parent_root=parent, parent_run=run,
        output_root=baseline_root, policy_path=policy,
        tokenizer_revision="a" * 40, producer_revision="b" * 64,
        source_profile="software_docs", rows_per_shard=2)
    interrupted = tmp_path / "interrupted.partial.1"; (interrupted / "chunks").mkdir(parents=True)
    for shard in sorted((baseline_root / "chunks").glob("shard-*.texts.jsonl.gz")):
        shutil.copy2(shard, interrupted / "chunks" / shard.name)
    resumed_root = tmp_path / "resumed"
    resumed = subject.stage(
        artifact_root=tmp_path, parent_root=parent, parent_run=run,
        output_root=resumed_root, policy_path=policy,
        tokenizer_revision="a" * 40, producer_revision="b" * 64,
        source_profile="software_docs", rows_per_shard=2,
        resume_partial=interrupted)
    assert resumed["counts"] == baseline["counts"]
    assert resumed["chunks"]["logical_sha256"] == baseline["chunks"]["logical_sha256"]
    assert resumed["resume"]["sha256"]
    resume = json.loads((resumed_root / "resume-evidence/RESUME.json").read_text())
    assert resume["closed_shards"] == 2
    assert resume["closed_rows"] == 4
    assert resume["excluded_open_shard"]["reason"].startswith("highest shard")
    for name in ("shard-0000.texts.jsonl.gz", "shard-0001.texts.jsonl.gz",
                 "shard-0002.texts.jsonl.gz"):
        assert (resumed_root / "chunks" / name).read_bytes() == (
            baseline_root / "chunks" / name).read_bytes()


def _multishard_parent(tmp_path: Path, name: str, rows: list[dict],
                       *, declared_records: int | None = None) -> tuple[Path, Path]:
    parent = tmp_path / name
    artifacts = []
    logical = b""
    for index, row in enumerate(rows):
        encoded = (canonical_json(row) + "\n").encode()
        logical += encoded
        artifact = parent / "stages/markdown/data" / (
            f"shard-{index:04d}.jsonl.gz")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(gzip.compress(encoded, mtime=0))
        artifacts.append(artifact)
    markdown = seal_stage_receipt.seal(
        artifact_root=tmp_path, artifact_paths=artifacts,
        receipt_path=parent / "stages/markdown/receipt.json",
        records=declared_records if declared_records is not None else len(rows),
        fmt="canonical-jsonl+deterministic-gzip",
        logical_sha256=hashlib.sha256(logical).hexdigest(),
        producer_id="fixture", producer_version="v1")
    quarantine_row = {"document_id": "q1", "reason": "fixture quarantine"}
    quarantine_bytes = (canonical_json(quarantine_row) + "\n").encode()
    quarantine_artifact = parent / "stages/quarantine/data/records.jsonl.gz"
    quarantine_artifact.parent.mkdir(parents=True)
    quarantine_artifact.write_bytes(gzip.compress(quarantine_bytes, mtime=0))
    quarantine = seal_stage_receipt.seal(
        artifact_root=tmp_path, artifact_paths=[quarantine_artifact],
        receipt_path=parent / "stages/quarantine/receipt.json", records=1,
        fmt="explicit-quarantine", logical_sha256=hashlib.sha256(
            quarantine_bytes).hexdigest(), producer_id="fixture",
        producer_version="v1")
    run = parent / "RUN.json"
    run.write_text(json.dumps({"stages": {
        "markdown": markdown, "quarantine": quarantine}}))
    return parent, run


def _patch_structured_fixture(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(subject, "chunk_generic_records",
                        lambda markdown, selection=None: [SimpleNamespace(
                            text=markdown, dense_text=markdown,
                            sparse_text=markdown, stamps={})])
    selection = ChunkerSelection("structured", 4, "p@4",
                                 str(tmp_path / "policy.json"), "f" * 64,
                                 "a" * 40)
    monkeypatch.setattr(subject, "_preflight_chunker", lambda *args: (
        selection, {"tokenizer_class": "FixtureFastTokenizer",
                    "tokenizer_is_fast": True}))
    monkeypatch.setattr(subject.aw_chunk, "token_len",
                        lambda text: len(text.split()))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"policy_id": "p", "epoch": 4,
                                  "rule_legend": "p@4",
                                  "token_hard_max": {"exact": 2560}}))
    return policy


def test_structured_pack_consumes_three_ordered_shards_and_excludes_quarantine(
        tmp_path: Path, monkeypatch):
    rows = [{"document_id": f"d{index}",
             "url": f"https://example.test/d{index}",
             "markdown": f"# Document {index}\n\nFull body {index}."}
            for index in range(3)]
    parent, run = _multishard_parent(tmp_path, "parent-multi", rows)
    policy = _patch_structured_fixture(monkeypatch, tmp_path)
    result = subject.stage(
        artifact_root=tmp_path, parent_root=parent, parent_run=run,
        output_root=tmp_path / "pack-multi", policy_path=policy,
        tokenizer_revision="a" * 40, producer_revision="b" * 64,
        source_profile="web_pdf")
    assert result["counts"] == {
        "documents": 3, "chunks": 3, "measured_chunks_per_document": 1.0}
    assert result["parent"]["documents"] == 3


def test_structured_pack_rejects_duplicate_parent_across_shards(
        tmp_path: Path, monkeypatch):
    rows = [
        {"document_id": "d1", "url": "https://example.test/d1",
         "markdown": "# First\n\nBody."},
        {"document_id": "d1", "url": "https://example.test/d1-copy",
         "markdown": "# Duplicate\n\nBody."},
    ]
    parent, run = _multishard_parent(tmp_path, "parent-duplicate", rows)
    policy = _patch_structured_fixture(monkeypatch, tmp_path)
    with pytest.raises(subject.StructuredStageError,
                       match="duplicate Markdown parent d1"):
        subject.stage(
            artifact_root=tmp_path, parent_root=parent, parent_run=run,
            output_root=tmp_path / "pack-duplicate", policy_path=policy,
            tokenizer_revision="a" * 40, producer_revision="b" * 64)


def test_structured_pack_rejects_missing_parent_from_shard_inventory(
        tmp_path: Path, monkeypatch):
    rows = [
        {"document_id": "d1", "url": "https://example.test/d1",
         "markdown": "# First\n\nBody."},
        {"document_id": "d2", "url": "https://example.test/d2",
         "markdown": "# Second\n\nBody."},
    ]
    parent, run = _multishard_parent(
        tmp_path, "parent-missing", rows, declared_records=3)
    policy = _patch_structured_fixture(monkeypatch, tmp_path)
    with pytest.raises(subject.StructuredStageError,
                       match="parent count mismatch: receipt=3 rows=2"):
        subject.stage(
            artifact_root=tmp_path, parent_root=parent, parent_run=run,
            output_root=tmp_path / "pack-missing", policy_path=policy,
            tokenizer_revision="a" * 40, producer_revision="b" * 64)
