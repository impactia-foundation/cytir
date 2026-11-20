#!/usr/bin/env python3
"""Chunk a sealed Markdown stage under an explicit structured policy pin."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire.deterministic_gzip import deterministic_gzip_text
from cytir.acquire import stage_receipt as SS
from cytir.chunk import chunker as aw_chunk
from cytir.chunk import provenance as CP
from cytir.chunk.generic import chunk_generic_records
from cytir.core.common import mint_passage_point_id, require_qdrant_point_id
from cytir.chunk.dispatch import ChunkerSelection, explicit_structured_chunker_selection
from cytir.index.preservation_contract import _validate_stage


VERSION = "structured-markdown-embed-stage.v8-ordered-multishard"


class StructuredStageError(RuntimeError):
    pass


def _ordered_markdown_paths(validated_stage: dict,
                            artifact_root: Path) -> list[Path]:
    """Resolve the exhaustive ordered principal Markdown shard inventory.

    Preservation receipts may bind supporting per-document evidence alongside
    one or more canonical JSONL gzip shards.  Only the JSONL gzip artifacts are
    the principal row stream.  Their receipt order must already be the stable
    URI order so a rerun cannot silently change cross-shard document order.
    Every artifact was recursively size/SHA checked by ``_validate_stage``
    before this function is called.
    """
    references = [
        artifact for artifact in (validated_stage.get("artifacts") or [])
        if str(artifact.get("uri") or "").endswith(".jsonl.gz")
    ]
    if not references:
        raise StructuredStageError(
            "structured projection has no canonical Markdown JSONL gzip shard")
    ordered = sorted(references, key=lambda artifact: str(artifact["uri"]))
    if references != ordered:
        raise StructuredStageError(
            "Markdown shard inventory is not in stable URI order")
    paths = [
        artifact_root / str(artifact["uri"]).removeprefix("artifact://")
        for artifact in references
    ]
    if len(paths) != len(set(paths)):
        raise StructuredStageError("Markdown shard inventory contains duplicates")
    return paths


def _resume_closed_shards(*, artifact_root: Path, resume_partial: Path,
                          partial: Path, rows_per_shard: int,
                          source_profile: str) -> dict:
    """Admit only fully closed pre-reboot shards and seal their row boundary.

    The highest-numbered shard is always treated as open/excluded because an
    unclean process death cannot prove that its gzip footer was published.
    Every preceding shard must contain exactly ``rows_per_shard`` canonical
    rows with valid deterministic point IDs and contiguous document ordinals.
    """
    resume_partial = resume_partial.resolve()
    resume_partial.relative_to(artifact_root)
    candidates = sorted((resume_partial / "chunks").glob(
        "shard-*.texts.jsonl.gz"))
    if len(candidates) < 2:
        raise StructuredStageError(
            "resume partial needs a closed shard plus one excluded open shard")
    for index, path in enumerate(candidates):
        if path.name != f"shard-{index:04d}.texts.jsonl.gz":
            raise StructuredStageError("resume shard sequence is not contiguous")
    closed, excluded = candidates[:-1], candidates[-1]
    evidence_dir = partial / "resume-evidence"
    evidence_dir.mkdir(parents=True)
    boundaries_path = evidence_dir / "document-boundaries.jsonl.gz"
    closed_refs = []
    current_document = None
    current_last_ordinal = -1
    current_rows: list[str] = []
    final_document_rows: list[str] = []
    documents = rows = 0
    with deterministic_gzip_text(boundaries_path, compresslevel=6) as boundaries:
        for index, source_path in enumerate(closed):
            target = partial / "chunks" / source_path.name
            os.link(source_path, target)
            shard_rows = 0
            with gzip.open(source_path, "rt", encoding="utf-8") as source:
                for line_no, line in enumerate(source, 1):
                    row = json.loads(line)
                    if canonical_json(row) + "\n" != line:
                        raise StructuredStageError(
                            f"resume shard {index} row {line_no} is not canonical")
                    point_id = require_qdrant_point_id(str(row.get("point_id") or ""))
                    document_id = str(row.get("document_id") or "")
                    ordinal = row.get("chunk_ordinal")
                    canonical = str(row.get("canonical_text") or "")
                    url = str(row.get("url") or "")
                    if (not document_id or not isinstance(ordinal, int) or ordinal < 0
                            or not canonical or not url
                            or row.get("source_profile") != source_profile):
                        raise StructuredStageError(
                            f"resume shard {index} row {line_no} schema drift")
                    expected = require_qdrant_point_id(mint_passage_point_id(
                        url, f"document:{document_id}:chunk-{ordinal}", canonical))
                    if point_id != expected:
                        raise StructuredStageError(
                            f"resume shard {index} row {line_no} point identity drift")
                    if document_id != current_document:
                        if current_document is not None:
                            boundaries.write(canonical_json({
                                "document_id": current_document,
                                "last_chunk_ordinal": current_last_ordinal}) + "\n")
                            documents += 1
                        if ordinal != 0:
                            raise StructuredStageError(
                                "resume document does not begin at chunk ordinal zero")
                        current_document = document_id
                        current_last_ordinal = -1
                        current_rows = []
                    if ordinal != current_last_ordinal + 1:
                        raise StructuredStageError(
                            f"resume chunk ordinal gap for {document_id}")
                    current_last_ordinal = ordinal
                    current_rows.append(line.rstrip("\n"))
                    shard_rows += 1; rows += 1
            if shard_rows != rows_per_shard:
                raise StructuredStageError(
                    f"resume shard {source_path.name} has {shard_rows} rows, "
                    f"expected {rows_per_shard}")
            closed_refs.append({
                "uri": f"artifact://{source_path.relative_to(artifact_root).as_posix()}",
                "bytes": source_path.stat().st_size,
                "sha256": sha256_file(source_path), "rows": shard_rows})
        if current_document is None:
            raise StructuredStageError("resume closed shards contain no rows")
        boundaries.write(canonical_json({"document_id": current_document,
                                         "last_chunk_ordinal": current_last_ordinal}) + "\n")
        documents += 1; final_document_rows = current_rows
    return {
        "closed_paths": [partial / "chunks" / path.name for path in closed],
        "closed_shards": len(closed), "closed_rows": rows,
        "closed_documents": documents, "boundaries_path": boundaries_path,
        "last_document_id": current_document,
        "last_document_rows": final_document_rows,
        "last_document_ordinal": current_last_ordinal,
        "closed_artifacts": closed_refs,
        "excluded_open_shard": {
            "uri": f"artifact://{excluded.relative_to(artifact_root).as_posix()}",
            "bytes": excluded.stat().st_size, "sha256": sha256_file(excluded),
            "reason": "highest shard at unclean reboot boundary; never admitted"},
        "source_partial": f"artifact://{resume_partial.relative_to(artifact_root).as_posix()}",
    }


def _source_parent_identity(value: dict, source_profile: str,
                            line_no: int) -> object | None:
    """Return the preserved source-native parent witness for a chunk."""
    identity = (value.get("parent_identity") or value.get("source_archive")
                or value.get("parent_archive") or value.get("parent_response"))
    if source_profile == "anchored_law" and not identity:
        raise StructuredStageError(
            f"Markdown line {line_no} anchored_law lacks source parent identity")
    return identity


def _preflight_chunker(policy_path: Path, policy_sha: str,
                       tokenizer_revision: str) -> tuple[ChunkerSelection, dict]:
    """Activate argv pins and prove the exact tokenizer is usable up front.

    This stage is clone-runnable from its declared CLI.  It does not require a
    second, hidden copy of the policy identity in AW_CHUNK_* environment
    variables; the explicit path, byte SHA and tokenizer revision are the
    effective configuration checked below.
    """
    selection = explicit_structured_chunker_selection(
        str(policy_path), policy_sha, tokenizer_revision)
    aw_chunk.select_tokenizer_revision(selection.tokenizer_revision)
    activated = aw_chunk.select_policy(selection.policy_path,
                                       allow_env_overrides=False)
    if activated.sha256 != policy_sha:
        raise StructuredStageError("activated chunk policy SHA disagrees with stage pin")
    config = aw_chunk.config()
    if (config.get("policy", {}).get("sha256") != policy_sha or
            config.get("tokenizer_revision") != tokenizer_revision):
        raise StructuredStageError(
            "effective chunker config disagrees with explicit stage pins")
    if config.get("tokenizer_status") != "ok" or not config.get("tokenizer_is_fast"):
        raise StructuredStageError(
            "exact fast tokenizer counter unavailable; refusing word-only production stage")
    return selection, config


def stage(*, artifact_root: Path, parent_root: Path, parent_run: Path, output_root: Path,
          policy_path: Path, tokenizer_revision: str,
          producer_revision: str,
          source_profile: str = "structured_data_feed",
          rows_per_shard: int = 50_000,
          resume_partial: Path | None = None) -> dict:
    artifact_root = artifact_root.resolve()
    output_root = output_root.resolve()
    if not output_root.is_relative_to(artifact_root):
        raise StructuredStageError("output must be below the global artifact root")
    run = json.loads(parent_run.read_text(encoding="utf-8"))
    markdown_stage = run.get("stages", {}).get("markdown")
    validated = _validate_stage(markdown_stage, "parent.markdown",
                                artifact_root=artifact_root, deep=True)
    markdown_paths = _ordered_markdown_paths(validated, artifact_root)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_sha = sha256_file(policy_path)
    declared_profiles = policy.get("source_profiles")
    if (declared_profiles is not None and source_profile not in declared_profiles):
        raise StructuredStageError(
            f"chunk policy does not admit source profile {source_profile}")
    if rows_per_shard <= 0:
        raise StructuredStageError("rows_per_shard must be positive")
    if (output_root / "PACK.json").is_file():
        prior = json.loads((output_root / "PACK.json").read_text(encoding="utf-8"))
        if (prior.get("parent", {}).get("run_sha256") != sha256_file(parent_run)
                or prior.get("chunk_policy", {}).get("sha256") != policy_sha
                or prior.get("source_profile") != source_profile):
            raise StructuredStageError("existing pack identity differs from requested run")
        _validate_stage(prior.get("chunks"), "existing.chunks",
                        artifact_root=artifact_root, deep=True)
        return prior
    if output_root.exists():
        raise StructuredStageError(f"refusing incomplete existing output: {output_root}")
    selection, tokenizer_identity = _preflight_chunker(
        policy_path, policy_sha, tokenizer_revision)
    provenance = CP.build(
        family="generic-structured",
        chunker={"mode": "structured", "chunk_epoch": policy["epoch"],
                 "rule_legend": policy["rule_legend"],
                 "policy_path": f"data/{policy_path.name}",
                 "policy_sha256": policy_sha,
                 "chunk_tokenizer_revision": tokenizer_revision},
        channels={"canonical": "hierarchy-plus-body",
                  "dense": "leaf-heading-plus-body",
                  "sparse": "hierarchy-plus-body", "channels_split": True},
        producer={"id": "pipeline/stage_structured_markdown_for_embedding.py",
                  "revision": producer_revision,
                  "parent_run_sha256": sha256_file(parent_run)},
        confidence="stamped")
    partial = output_root.with_name(output_root.name + f".partial.{os.getpid()}")
    if partial.exists():
        raise StructuredStageError(f"partial output already exists: {partial}")
    (partial / "chunks").mkdir(parents=True)
    resume = None
    if resume_partial is not None:
        resume = _resume_closed_shards(
            artifact_root=artifact_root, resume_partial=resume_partial,
            partial=partial, rows_per_shard=rows_per_shard,
            source_profile=source_profile)
    logical_digest = hashlib.sha256()
    token_histogram = [0] * (policy["token_hard_max"]["exact"] + 1)
    chunk_count = resume["closed_rows"] if resume else 0
    shard_paths = list(resume["closed_paths"]) if resume else []
    shard_context = None
    shard_file = None
    documents = 0
    seen_document_ids: set[str] = set()
    try:
        boundary_context = (gzip.open(resume["boundaries_path"], "rt", encoding="utf-8")
                            if resume else None)
        boundary_iterator = iter(boundary_context) if boundary_context else iter(())
        boundary = json.loads(next(boundary_iterator)) if resume else None
        try:
            boundary_next = (json.loads(next(boundary_iterator))
                             if resume and resume["closed_documents"] > 1 else None)
        except StopIteration:
            boundary_next = None
        line_no = 0
        for markdown_path in markdown_paths:
            shard_rows = 0
            with gzip.open(markdown_path, "rt", encoding="utf-8") as source:
                for shard_line_no, line in enumerate(source, 1):
                    line_no += 1
                    shard_rows += 1
                    value = json.loads(line)
                # Exact-production reconstruction materializers already carry the
                # canonical parent identity as point_id.  Treat it as the parent
                # document ID; chunk identities are still freshly minted below.
                    document_id = str(value.get("document_id") or value.get("point_id") or "")
                    markdown = str(value.get("markdown") or "")
                    if not document_id or not markdown:
                        raise StructuredStageError(
                            f"Markdown shard {markdown_path.name} line {shard_line_no} "
                            "lacks document_id or markdown")
                    if document_id in seen_document_ids:
                        raise StructuredStageError(
                            f"duplicate Markdown parent {document_id} across ordered shards")
                    seen_document_ids.add(document_id)
                    documents += 1
                    resume_last_document = False
                    if boundary is not None:
                        if document_id != boundary["document_id"]:
                            raise StructuredStageError(
                                f"resume/source document order drift at Markdown line {line_no}: "
                                f"{document_id} != {boundary['document_id']}")
                        if boundary_next is not None:
                            boundary = boundary_next
                            try:
                                boundary_next = json.loads(next(boundary_iterator))
                            except StopIteration:
                                boundary_next = None
                            continue
                        resume_last_document = True
                        boundary = None
                    parent_markdown_sha256 = hashlib.sha256(markdown.encode()).hexdigest()
                    declared_parent_sha = value.get("markdown_sha256")
                    if declared_parent_sha and declared_parent_sha != parent_markdown_sha256:
                        raise StructuredStageError(
                            f"Markdown line {line_no} digest differs from declared markdown_sha256")
                    parent_identity = _source_parent_identity(
                        value, source_profile, line_no)
                    last_generated_ordinal = -1
                    for ordinal, unit in enumerate(chunk_generic_records(
                            markdown, selection=selection)):
                        last_generated_ordinal = ordinal
                        canonical = unit.text
                        dense = unit.dense_text or canonical
                        sparse = unit.sparse_text or canonical
                        url = str(value.get("url") or "")
                        if not url:
                            raise StructuredStageError(
                                f"Markdown line {line_no} lacks URL required for point identity")
                        point_id = require_qdrant_point_id(
                            mint_passage_point_id(
                                url, f"document:{document_id}:chunk-{ordinal}", canonical))
                        dense_tokens = aw_chunk.token_len(dense)
                        token_hard_max = policy["token_hard_max"]["exact"]
                        if dense_tokens > token_hard_max:
                            raise StructuredStageError(
                                f"Markdown line {line_no} chunk {ordinal} has "
                                f"{dense_tokens} tokens > hard max {token_hard_max}")
                        token_histogram[dense_tokens] += 1
                        source_metadata = {
                            key: value[key] for key in (
                                "title", "source_relative_path", "source_sha256",
                                "source_text_encoding", "repository", "revision",
                                "source_path", "parent_source_sha256",
                                "source_heading_count", "source_reconstruction_sha256",
                                "updated_at", "published_at", "modified_date",
                                "headline", "byline", "byline_status",
                                "publication_time", "publication_time_status",
                                "publisher", "fetched_at",
                                "source_version", "fiche_id", "fiche_type", "subject",
                                "archive_member", "archive_sha256",
                                "archive_member_sha256",
                                "site", "answer_url", "tags", "cell", "question",
                                "accepted_answer",
                                "comparison_prior_amin3_qmin8_kans2", "product")
                            if key in value and value[key] is not None
                        }
                        row = {
                            "point_id": point_id,
                            "document_id": document_id, "chunk_ordinal": ordinal,
                            "source_record_id": value.get("source_record_id") or document_id,
                            "source_profile": source_profile,
                            "text": dense, "canonical_text": canonical,
                            "sparse_text": sparse, "url": value.get("url"),
                            "lang": value.get("lang"), "prov_id": provenance["prov_id"],
                            "parent_identity": parent_identity,
                            "source_links": value.get("source_links", []),
                            "section_heading": next((text_line.strip() for text_line in
                                                     dense.splitlines() if text_line.strip()), ""),
                            "source_metadata": source_metadata,
                            "parent_markdown_sha256": parent_markdown_sha256,
                        }
                        encoded_row = canonical_json(row) + "\n"
                        if resume_last_document and ordinal <= resume["last_document_ordinal"]:
                            expected_rows = resume["last_document_rows"]
                            if ordinal >= len(expected_rows) or encoded_row.rstrip("\n") != expected_rows[ordinal]:
                                raise StructuredStageError(
                                    f"resume boundary row drift for {document_id} chunk {ordinal}")
                            continue
                        if chunk_count % rows_per_shard == 0:
                            if shard_context is not None:
                                shard_context.__exit__(None, None, None)
                            shard = partial / "chunks" / (
                                f"shard-{chunk_count // rows_per_shard:04d}.texts.jsonl.gz")
                            shard_paths.append(shard)
                            shard_context = deterministic_gzip_text(shard, compresslevel=6)
                            shard_file = shard_context.__enter__()
                        shard_file.write(encoded_row)
                        logical_digest.update(encoded_row.encode())
                        chunk_count += 1
                        if chunk_count % 100_000 == 0:
                            print(f"[structured-stage] documents={documents:,} "
                                  f"chunks={chunk_count:,}", flush=True)
                    if (resume_last_document and
                            last_generated_ordinal < resume["last_document_ordinal"]):
                        raise StructuredStageError(
                            f"resume boundary extends beyond regenerated document {document_id}")
            if shard_rows == 0:
                raise StructuredStageError(
                    f"Markdown shard is empty: {markdown_path.name}")
        if boundary_context is not None:
            boundary_context.close()
        if boundary is not None or boundary_next is not None:
            raise StructuredStageError("parent Markdown ended before resume boundary")
        if shard_context is not None:
            shard_context.__exit__(None, None, None)
            shard_context = None
        if documents != validated["records"]:
            raise StructuredStageError(
                f"parent count mismatch: receipt={validated['records']} rows={documents}")
        # Re-derive the logical digest and exact token distribution over both
        # admitted pre-reboot closed shards and newly written shards.  The
        # expensive parent chunking is resumed at the proven boundary; this
        # terminal pass only validates emitted rows and counts tokens.
        logical_digest = hashlib.sha256()
        token_histogram = [0] * (policy["token_hard_max"]["exact"] + 1)
        verified_chunks = 0
        for shard in shard_paths:
            with gzip.open(shard, "rt", encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    if canonical_json(row) + "\n" != line:
                        raise StructuredStageError(
                            f"noncanonical terminal chunk row: {shard}")
                    dense_tokens = aw_chunk.token_len(str(row.get("text") or ""))
                    token_hard_max = policy["token_hard_max"]["exact"]
                    if dense_tokens > token_hard_max:
                        raise StructuredStageError(
                            f"terminal chunk has {dense_tokens} tokens > hard max")
                    token_histogram[dense_tokens] += 1
                    logical_digest.update(line.encode())
                    verified_chunks += 1
        if verified_chunks != chunk_count:
            raise StructuredStageError(
                f"terminal chunk recount drift: {verified_chunks} != {chunk_count}")
        logical_sha = logical_digest.hexdigest()
        legend = partial / "chunks" / "provenance-legend.json"
        legend_data = (json.dumps({provenance["prov_id"]: provenance},
                                  indent=2, sort_keys=True) + "\n").encode()
        atomic_publish(legend, legend_data)
        # PACK.json is the terminal publication gate.  Move completed data to
        # its final location first so every receipt URI can be global-root
        # relative; seal the receipt there, and publish PACK last.
        os.rename(partial, output_root)
        final_shards = [output_root / "chunks" / path.name for path in shard_paths]
        legend = output_root / "chunks" / legend.name
        resume_binding = None
        if resume:
            resume_evidence = dict(resume)
            for transient in ("closed_paths", "boundaries_path", "last_document_rows"):
                resume_evidence.pop(transient, None)
            boundaries_final = output_root / "resume-evidence/document-boundaries.jsonl.gz"
            resume_evidence["document_boundaries"] = {
                "uri": f"artifact://{boundaries_final.relative_to(artifact_root).as_posix()}",
                "bytes": boundaries_final.stat().st_size,
                "sha256": sha256_file(boundaries_final),
                "records": resume["closed_documents"]}
            resume_evidence.update({
                "schema": "aw.structured-stage-resume-evidence.v1",
                "status": "closed-shards-admitted-open-shard-excluded",
                "parent_run_sha256": sha256_file(parent_run),
                "chunk_policy_sha256": policy_sha,
                "source_profile": source_profile,
                "rows_per_shard": rows_per_shard})
            resume_path = output_root / "resume-evidence/RESUME.json"
            atomic_publish(resume_path, (json.dumps(
                resume_evidence, indent=2, sort_keys=True) + "\n").encode())
            resume_binding = {
                "uri": f"artifact://{resume_path.relative_to(artifact_root).as_posix()}",
                "bytes": resume_path.stat().st_size,
                "sha256": sha256_file(resume_path)}
        chunks = SS.seal(
            artifact_root=artifact_root, artifact_paths=[*final_shards, legend],
            receipt_path=output_root / "chunks" / "receipt.json",
            records=chunk_count,
            fmt="canonical-jsonl+deterministic-gzip+provenance-legend",
            logical_sha256=logical_sha,
            producer_id="pipeline/stage_structured_markdown_for_embedding.py",
            producer_version=VERSION)

        def percentile(percent: int) -> int:
            wanted = max(1, (chunk_count * percent + 99) // 100)
            seen = 0
            for length, count in enumerate(token_histogram):
                seen += count
                if seen >= wanted:
                    return length
            return 0

        result = {
            "schema": "aw.structured-embedding-pack.v1", "version": VERSION,
            "status": "ready_for_gpu_queue", "source_profile": source_profile,
            "parent": {"run_sha256": sha256_file(parent_run),
                       "markdown_logical_sha256": validated["logical_sha256"],
                       "documents": documents},
            "chunk_policy": {"path": f"data/{policy_path.name}",
                             "sha256": policy_sha, "policy_id": policy["policy_id"],
                             "tokenizer_revision": tokenizer_revision,
                             "tokenizer_class": tokenizer_identity["tokenizer_class"],
                             "tokenizer_is_fast": tokenizer_identity["tokenizer_is_fast"]},
            "counts": {"documents": documents, "chunks": chunk_count,
                       "measured_chunks_per_document": round(
                           chunk_count / documents, 6) if documents else 0},
            "token_distribution": {
                "counter": "exact-fast-tokenizer-with-special-tokens",
                "rows": chunk_count,
                "histogram": [[index, count]
                              for index, count in enumerate(token_histogram)
                              if count],
                "p50": percentile(50) if chunk_count else 0,
                "p95": percentile(95) if chunk_count else 0,
                "max": max((index for index, count in enumerate(token_histogram)
                            if count), default=0),
                "hard_max": policy["token_hard_max"]["exact"],
                "over_hard_max": 0,
            },
            "chunks": chunks,
            "embedding": {
                "model": "Snowflake/snowflake-arctic-embed-l-v2.0",
                "revision": "ac6544c8a46e00af67e330e85a9028c66b8cfd9a",
                "dimensions": 1024, "dtype": "float16",
                "encoding": "row-major little-endian binary",
                "launch_state": "wait_for_shared_gpu_predecessors"},
        }
        if resume_binding:
            result["resume"] = resume_binding
        body = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        atomic_publish(output_root / "PACK.json", body)
        return result
    except BaseException:
        if shard_context is not None:
            shard_context.__exit__(None, None, None)
        shutil.rmtree(partial, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--producer-revision", required=True)
    parser.add_argument("--source-profile", default="structured_data_feed",
                        help="Declared preservation profile for the pack")
    parser.add_argument("--rows-per-shard", type=int, default=50_000)
    parser.add_argument("--resume-partial", type=Path,
                        help="Reboot-interrupted prior partial; only fully closed shards are admitted")
    args = parser.parse_args()
    result = stage(artifact_root=args.artifact_root.resolve(),
                   parent_root=args.parent_root.resolve(),
                   parent_run=args.parent_run.resolve(),
                   output_root=args.out.resolve(),
                   policy_path=args.policy.resolve(),
                   tokenizer_revision=args.tokenizer_revision,
                   producer_revision=args.producer_revision,
                   source_profile=args.source_profile,
                   rows_per_shard=args.rows_per_shard,
                   resume_partial=(args.resume_partial.resolve()
                                   if args.resume_partial else None))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
