#!/usr/bin/env python3
"""Audit a Context7 exact-path pack against the software-docs profile."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import uuid
from urllib.parse import unquote

from cytir.extract.github_agent_resources import compose_document_title
from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire import preserved_stage_io as PS
from cytir.index.preservation_contract import _validate_stage


URL = re.compile(r"^https://github\.com/[^/]+/[^/]+/blob/[0-9a-f]{40}/.+$")
# The URL is composed independently of the identity block, so parsing the
# lineage back OUT of it is a real cross-check and not a tautology -- which is
# what the front-matter parse this replaces was actually testing (task 517).
URL_LINEAGE = re.compile(
    r"^https://github\.com/(?P<repo>[^/]+/[^/]+)/blob/(?P<revision>[0-9a-f]{40})/"
    r"(?P<path>.+)$")
IDENTITY_FIELDS = ("repository", "revision", "source_path", "parent_sha256")


def _front_matter_absent(markdown: str) -> bool:
    """Task 517: the L0 binding is record metadata, never the document's head.

    v1 opened every document with a four-line YAML block, the chunker's
    section-adjacent policy made it the head of chunk 0, and 72,330 of the
    126,427 staged rows shipped with provenance as their snippet AND as the
    head of their embedding. A materialization that puts it back must fail here.
    """
    return not markdown.lstrip().startswith("---\naw_")


def _rows(artifact_root: Path, stage: dict) -> list[dict]:
    validated = _validate_stage(stage, "rows", artifact_root=artifact_root,
                                deep=True)
    shards = [PS.artifact_path(artifact_root, row)
              for row in validated["artifacts"]
              if row["uri"].endswith(".jsonl.gz")]
    if len(shards) != 1:
        raise ValueError("expected exactly one canonical JSONL shard")
    with gzip.open(shards[0], "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit(*, artifact_root: Path, parent_run: Path, pack_path: Path,
          output_root: Path, profile_registry: Path,
          expected_documents: int, expected_recovery: int,
          source_id: str) -> tuple[Path, Path]:
    run = json.loads(parent_run.read_text())
    pack = json.loads(pack_path.read_text())
    if (run.get("schema") != "aw.context7-exact-path-materialization-run.v1"
            or run.get("source_profile") != "software_docs"
            or pack.get("source_profile") != "software_docs"
            or run.get("counts", {}).get("markdown_documents") != expected_documents
            or run.get("counts", {}).get("recovery_documents") != expected_recovery
            or pack.get("counts", {}).get("documents") != expected_documents):
        raise ValueError("Context7 parent/pack identity or count drift")
    for name, stage in run["stages"].items():
        _validate_stage(stage, f"parent.{name}", artifact_root=artifact_root,
                        deep=True)
    _validate_stage(pack["chunks"], "pack.chunks",
                    artifact_root=artifact_root, deep=True)
    markdown = _rows(artifact_root, run["stages"]["markdown"])
    outlinks = _rows(artifact_root, run["stages"]["outlinks"])
    recovery = _rows(artifact_root, run["stages"]["recovery"])
    chunks = _rows(artifact_root, pack["chunks"])
    if (len(markdown) != expected_documents
            or len(recovery) != expected_recovery
            or len(chunks) != pack["counts"]["chunks"]
            or len(outlinks) != run["counts"]["outlink_records_including_empty"]):
        raise ValueError("Context7 stage row conservation failure")

    parents: dict[str, dict] = {}
    markdown_by_id: dict[str, str] = {}
    heading_documents = 0
    for row in markdown:
        body = row.get("markdown") or ""
        identity = row.get("official_source_identity")
        url_match = URL_LINEAGE.fullmatch(row.get("url") or "")
        if (not url_match or not URL.fullmatch(row.get("url") or "")
                or not isinstance(identity, dict)
                or any(not identity.get(k) for k in IDENTITY_FIELDS)):
            raise ValueError("Context7 Markdown lost repo/path/commit/blob lineage")
        if not _front_matter_absent(body):
            raise ValueError(
                "Context7 Markdown opens with the provenance front matter: the "
                "L0 binding belongs in official_source_identity, not at the "
                "head of the document's first chunk (task 517)")
        lineage = {"repo": identity["repository"],
                   "revision": identity["revision"],
                   "path": identity["source_path"],
                   "sha": identity["parent_sha256"]}
        if (lineage["repo"] != row.get("repository")
                or lineage["revision"] != row.get("revision")
                or lineage["path"] != row.get("source_path")
                or lineage["sha"] != row.get("parent_original_sha256")
                or lineage["repo"] != url_match.group("repo")
                or lineage["revision"] != url_match.group("revision")
                or lineage["path"] != unquote(url_match.group("path"))
                or not row.get("full_prechunk_content")):
            raise ValueError("Context7 Markdown lineage fields disagree")
        # The title is what the ranker reads and what `source_metadata.title`
        # carries into the payload. Re-derive it from the Markdown that actually
        # shipped, so a title from a different document, or from a body that was
        # edited after the title was taken, fails here.
        if row.get("title") != compose_document_title(body, row["source_path"]):
            raise ValueError(
                "Context7 document title is not the title its own Markdown "
                "composes")
        if re.search(r"^#{1,6}\s+\S", body, re.MULTILINE):
            heading_documents += 1
        parents[row["document_id"]] = {**lineage, "url": row["url"],
                                        "markdown_sha256": hashlib.sha256(
                                            body.encode()).hexdigest()}
        markdown_by_id[row["document_id"]] = body
    if len(parents) != expected_documents:
        raise ValueError("Context7 Markdown document IDs are not unique")

    outlink_parents = set()
    empty_parents = set()
    for row in outlinks:
        document_id = row.get("document_id")
        if document_id not in parents or row.get("source_url") != parents[document_id]["url"]:
            raise ValueError("Context7 outlink is not parent-bound")
        outlink_parents.add(document_id)
        if row.get("is_empty"):
            if row.get("target_url") is not None:
                raise ValueError("explicit empty Context7 outlink has target")
            empty_parents.add(document_id)
        elif not row.get("target_url"):
            raise ValueError("nonempty Context7 outlink lacks target")
    if outlink_parents != set(parents):
        raise ValueError("Context7 outlinks omit one or more explicit parents")

    point_ids = set()
    ordinals: dict[str, list[int]] = {}
    for row in chunks:
        uuid.UUID(str(row["point_id"]))
        document_id = row.get("document_id")
        parent = parents.get(document_id)
        metadata = row.get("source_metadata") or {}
        if (row["point_id"] in point_ids or parent is None
                or row.get("url") != parent["url"]
                or row.get("parent_markdown_sha256") != parent["markdown_sha256"]
                or metadata.get("repository") != parent["repo"]
                or metadata.get("revision") != parent["revision"]
                or metadata.get("source_path") != parent["path"]
                or not row.get("section_heading")
                or not row.get("canonical_text")
                or not row.get("text") or not row.get("sparse_text")):
            raise ValueError("Context7 chunk lost identity, hierarchy, or channel evidence")
        point_ids.add(row["point_id"])
        ordinals.setdefault(document_id, []).append(row["chunk_ordinal"])
    if any(sorted(values) != list(range(len(values)))
           for values in ordinals.values()):
        raise ValueError("Context7 chunk parent/ordinal conservation failed")
    # A parent with NO chunk is legal now and was impossible before. v1 wrote a
    # six-line provenance block at the head of every document, so a document
    # whose own content is nothing but headings still produced one chunk -- a
    # row whose entire text was the materializer's bookkeeping. With the block
    # gone (task 517) that document produces nothing, which is right.
    #
    # It is legal only for that reason, so the reason is checked rather than
    # allowed by a threshold: a zero-chunk parent must have no line that is
    # neither blank nor a heading. A parent with body text and no chunk is a
    # chunker defect and still fails here.
    silent = sorted(set(parents) - set(ordinals))
    for document_id in silent:
        body = markdown_by_id[document_id]
        if any(line.strip() and not line.lstrip().startswith("#")
               for line in body.splitlines()):
            raise ValueError(
                f"Context7 parent {document_id} has body text and produced no "
                "chunk")
    documents_without_a_chunk = len(silent)
    distribution = pack.get("token_distribution") or {}
    if (distribution.get("rows") != len(chunks)
            or distribution.get("over_hard_max") != 0
            or distribution.get("max", 1) > distribution.get("hard_max", 0)):
        raise ValueError("Context7 exact token gate failed")
    if any(row.get("status") != "recovery_required" for row in recovery):
        raise ValueError("Context7 recovery stage contains admitted documents")

    registry = json.loads(profile_registry.read_text())
    requirements = registry["profiles"]["software_docs"]
    methods = {
        "original": [
            (f"Deep-rehashed the terminal pinned Git tree/raw-blob/transport stage; "
             f"{expected_documents} exact blobs are admitted and {expected_recovery} "
             "failed paths remain a separate recovery stage."),
            (f"Scanned all {expected_documents} admitted parent rows and bound each to "
             "its exact terminal source receipt, response-body digest, commit, and path."),
        ],
        "markdown": [
            (f"Scanned {expected_documents} full pre-chunk Markdown parents: every row "
             f"retains path identity and source heading structure ({heading_documents} "
             "documents contain Markdown headings)."),
            f"Scanned {expected_documents} parents: every URL and official_source_identity pins the same 40-hex commit.",
            f"Scanned {expected_documents} parents: every official_source_identity binds the exact parent blob digest, no parent opens with a provenance block, and every title is the one its own Markdown composes.",
        ],
        "chunks": [
            f"Scanned {len(chunks)} chunks: commit+path URL, section heading, and contiguous parent-local ordinal are retained; {documents_without_a_chunk} heading-only parent(s) produced no chunk and carry no body text.",
            f"Scanned {len(chunks)} chunks: every parent Markdown digest matches the sealed L1 row.",
            (f"Deep-validated the chunk receipt and pinned policy/tokenizer; exact token "
             f"max={distribution['max']}/{distribution['hard_max']} with zero over-limit rows."),
        ],
        "outlinks": [
            f"Scanned {len(outlinks)} rows: every row carries repository/revision/path parent identity.",
            "Scanned every nonempty target emitted by the repository-Markdown link extractor.",
            f"Verified one outlink row per parent minimum, including {len(empty_parents)} explicit empty parents.",
        ],
        "vectors": [],
    }
    checks = {}
    for stage in requirements["required_stages"]:
        required = requirements.get(stage, {}).get("requirements", [])
        if len(required) != len(methods[stage]):
            raise ValueError(f"profile method count mismatch for {stage}")
        checks[stage] = [{"requirement": requirement, "status": "PASS",
                          "verification_method": method}
                         for requirement, method in zip(required, methods[stage])]
    output_root.mkdir(parents=True, exist_ok=False)
    evidence = {
        "schema": "aw.context7-software-docs-profile-evidence.v1",
        "source_id": source_id,
        "counts": {"markdown_parents": len(markdown), "recovery": len(recovery),
                   "heading_documents": heading_documents, "chunks": len(chunks),
                   "documents_without_a_chunk": documents_without_a_chunk,
                   "outlinks": len(outlinks), "empty_outlink_parents": len(empty_parents)},
        "logical_sha256": {
            "original": run["stages"]["original"]["logical_sha256"],
            "markdown": run["stages"]["markdown"]["logical_sha256"],
            "outlinks": run["stages"]["outlinks"]["logical_sha256"],
            "recovery": run["stages"]["recovery"]["logical_sha256"],
            "chunks": pack["chunks"]["logical_sha256"],
        },
        "parent_run": {"uri": "artifact://" + parent_run.relative_to(artifact_root).as_posix(),
                       "sha256": sha256_file(parent_run)},
        "pack": {"uri": "artifact://" + pack_path.relative_to(artifact_root).as_posix(),
                 "sha256": sha256_file(pack_path)},
    }
    evidence_path = output_root / "verification-evidence.json"
    checks_path = output_root / "checks.json"
    atomic_publish(evidence_path,
                   (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode())
    atomic_publish(checks_path,
                   (json.dumps(checks, indent=2, sort_keys=True) + "\n").encode())
    return evidence_path, checks_path


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog or "context7_software_docs_pack.py", description=__doc__)
    for name in ("artifact-root", "parent-run", "pack", "output-root",
                 "profile-registry"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-documents", type=int, required=True)
    parser.add_argument("--expected-recovery", type=int, required=True)
    parser.add_argument("--source-id", required=True)
    args = parser.parse_args(argv)
    evidence, checks = audit(
        artifact_root=args.artifact_root.resolve(),
        parent_run=args.parent_run.resolve(), pack_path=args.pack.resolve(),
        output_root=args.output_root.resolve(),
        profile_registry=args.profile_registry.resolve(),
        expected_documents=args.expected_documents,
        expected_recovery=args.expected_recovery, source_id=args.source_id)
    print(json.dumps({"verification_evidence": str(evidence),
                      "checks": str(checks)}, indent=2, sort_keys=True))
    return 0
