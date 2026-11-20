#!/usr/bin/env python3
"""Split HN-discussion records into `chunk.generic` chunks before staging.

`hn_discussion_ingest.py` emits one row per discussion (up to ~1,500 words of
title + story self-text + selected comments). `specs/hn-discussion-embedding.md`
Q3 (task 484, re-confirmed by reading `chunk.py` directly) recommends riding
the PINNED `chunk.generic` / epoch-4 policy rather than the legacy single-
chunk "b1-native" path the rest of the news lane uses: a capped discussion is
a flat sequence of short comment paragraphs, comfortably over the 768-word
target for any thread with more than a handful of comments, and epoch-4's
adjacent-section packer is documented (chunk.py:668-704) to cut BETWEEN
blocks (here: between whole comments), never mid-comment, as long as each
comment is its own blank-line-delimited Markdown block -- which
`render_discussion` already guarantees.

This module re-emits one row per resulting `Chunk`, carrying every original
payload field plus chunk provenance (`parent_markdown_sha256`,
`chunk_ordinal`, `chunk_count`). `stage_pack_b1.py` needs NO change to embed
these: its `aw_b1_identity` path already mints `point_id` from
content-digest-as-anchor when a row carries no explicit per-chunk anchor
identity, so two chunks of the same discussion URL with different `text`
already produce two distinct points.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from cytir.acquire.artifact_io import atomic_publish, canonical_json
from cytir.chunk import chunker as CHUNK
from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS

# FROZEN LABEL (task 1884 ruling): sealed into stage receipts as producer_id; not a path.
PRODUCER = "pipeline/chunk_hn_discussion_records.py"
VERSION = "chunk-hn-discussion-records.v1"


class ChunkingError(RuntimeError):
    """A discussion record could not be chunked under the pinned policy."""


def chunk_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Chunk one hn_discussion_ingest.py record under the ACTIVE chunk policy.

    Caller selects the policy first (`chunk.select_policy`) -- this module
    never selects one itself, so a run's policy identity is always explicit
    and pinned by whoever drives it, exactly like every other chunk.generic
    caller in this repo.
    """
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ChunkingError(f"{record.get('url')}: record has no text to chunk")
    parent_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunks = CHUNK.chunk_markdown(text)
    if not chunks:
        raise ChunkingError(f"{record.get('url')}: chunk.generic produced zero chunks")
    policy = CHUNK.active_policy()
    rows = []
    for chunk_obj in chunks:
        row = dict(record)
        row["text"] = chunk_obj.text
        row["snippet"] = " ".join(chunk_obj.text.split()[:25])
        row["chunk_ordinal"] = chunk_obj.ordinal
        row["chunk_count"] = len(chunks)
        row["parent_markdown_sha256"] = parent_sha256
        row["chunk_policy_id"] = policy.policy_id
        row["chunk_rule"] = policy.rule_legend
        # Consumed by aw_b1_projection.project_record(..., preserve_rule_epochs
        # =True): without this, a non-baseline chunk policy (epoch-4 here,
        # never the b1-native single-chunk news exception) would silently
        # stamp payload["rule_epochs"]["chunk"] as whatever the CURRENT
        # baseline default is at staging time -- exactly the fossil-record
        # failure ADR-090 amendment 1 / task 565 named.
        row["rule_epochs"] = {"chunk": policy.epoch}
        rows.append(row)
    return rows


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                yield json.loads(line)


def chunk_file(input_path: Path, output_path: Path,
               *, policy_path: Path, min_words: int | None = None) -> dict:
    CHUNK.select_policy(policy_path, allow_env_overrides=False)
    CHUNK.require_exact_tokenizer("hn_discussion")
    documents = chunks = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as sink:
        for record in iter_records(input_path):
            documents += 1
            for row in chunk_record(record):
                sink.write(json.dumps(row, sort_keys=True) + "\n")
                chunks += 1
    return {"documents": documents, "chunks": chunks,
            "policy_sha256": CHUNK.active_policy().sha256}


def seal_chunks_stage(root: Path, *, input_path: Path, policy_path: Path,
                      seal_root: Path | None = None) -> dict[str, Any]:
    """Chunk every hn_discussion record and seal the result as a preservation
    stage (`hn_discussion.chunks`, data/source-preservation-profiles-hn-discussion.v1.json).

    One row per chunk, never merging two discussions (each row's
    `parent_markdown_sha256` binds it to exactly one document).

    ``seal_root``, when given, is a shared ancestor of ``root`` that the
    sealed URI is expressed relative to (task 601: one tier-2 apply's
    artifact root, shared with the ingester's original/markdown/outlinks
    stages). Files are still written under ``root``; defaults to ``root``.
    """
    seal_root = seal_root if seal_root is not None else root
    CHUNK.select_policy(policy_path, allow_env_overrides=False)
    CHUNK.require_exact_tokenizer("hn_discussion")
    policy = CHUNK.active_policy()
    rows: list[dict[str, Any]] = []
    documents = 0
    for record in iter_records(input_path):
        documents += 1
        rows.extend(chunk_record(record))
    body = b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)
    logical_sha256 = hashlib.sha256(body).hexdigest()
    artifact = PS.store_artifact(root, "stages/chunks/data", "jsonl", body)
    stage = SS.seal(
        artifact_root=seal_root, artifact_paths=[PS.artifact_path(root, artifact)],
        receipt_path=root / "stages" / "chunks" / "receipt.json",
        records=len(rows), fmt="canonical-jsonl+deterministic-gzip",
        logical_sha256=logical_sha256, producer_id=PRODUCER,
        producer_version=VERSION)
    manifest = {
        "schema": "aw.hn-discussion-chunk-pack.v1", "version": VERSION,
        "chunk_policy": {"policy_id": policy.policy_id, "epoch": policy.epoch,
                         "sha256": policy.sha256,
                         "rule_legend": policy.rule_legend},
        "counts": {"documents": documents, "chunks": len(rows)},
        "stages": {"chunks": stage},
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    pack_path = root / "PACK.json"
    if pack_path.exists():
        if pack_path.read_bytes() != encoded:
            raise ChunkingError(f"immutable HN discussion chunk PACK collision: {pack_path}")
    else:
        atomic_publish(pack_path, encoded)
    return manifest


def main(argv: list[str] | None = None, prog: str = "cyt chunk hn-discussion") -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--in", dest="input_path", type=Path, required=True)
    parser.add_argument("--out", dest="output_path", type=Path, required=True)
    parser.add_argument("--chunk-policy", type=Path, required=True)
    parser.add_argument("--seal-root", type=Path, default=None,
                        help="also seal a hn_discussion.chunks preservation "
                             "stage (PACK.json) under this root")
    parser.add_argument(
        "--seal-artifact-root", type=Path, default=None,
        help="seal the chunks stage URI relative to this shared ancestor of "
             "--seal-root instead of --seal-root itself, so one content-pack "
             "intent can validate this stage against the SAME root as the "
             "ingester's original/markdown/outlinks stages")
    args = parser.parse_args(argv)
    result = chunk_file(args.input_path, args.output_path,
                        policy_path=args.chunk_policy)
    if args.seal_root is not None:
        args.seal_root.mkdir(parents=True, exist_ok=True)
        result["chunks_pack"] = seal_chunks_stage(
            args.seal_root, input_path=args.input_path,
            policy_path=args.chunk_policy, seal_root=args.seal_artifact_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
