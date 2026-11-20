"""Deep-audit a net-new RFC Editor pack against the software-docs profile."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import uuid

from cytir.acquire.artifact_io import sha256_file
from cytir.index.preservation_contract import _validate_stage

from cytir.sources import audit as shared


FRONT = re.compile(
    r'^---\naw_rfc_number: (?P<number>\d+)\n'
    r'aw_source_path: "(?P<path>input/rfcs/rfc\d+\.txt)"\n'
    r'aw_source_sha256: (?P<sha>[0-9a-f]{64})\n'
    r'aw_source_encoding: (?P<encoding>utf-8|iso-8859-1)\n'
    r'---\n\n# RFC (?P=number)\n\n')
URL = re.compile(r"^https://www\.rfc-editor\.org/rfc/rfc(?P<number>\d+)\.txt$")

def _restore_decorated_source(markdown: str, source: str, prefix_end: int) -> tuple[str, int]:
    decorated = markdown[prefix_end:].splitlines(keepends=True)
    original = source.splitlines(keepends=True)
    if len(decorated) != len(original):
        raise ValueError("RFC Markdown/source line conservation failure")
    restored = []
    headings = 0
    for projected, exact in zip(decorated, original):
        if projected == exact:
            restored.append(projected)
            continue
        match = re.match(r"^#{2,6} ", projected)
        if match is None or projected[match.end():] != exact:
            raise ValueError("RFC Markdown contains a non-reversible source change")
        restored.append(projected[match.end():])
        headings += 1
    return "".join(restored), headings


def _selection_counts(run: dict, pack: dict) -> tuple[str, int, int, int, int]:
    selection = str(run.get("selection") or "net-new")
    if selection == "production":
        documents_key, outlinks_key = "production_documents", "production_outlinks"
    elif selection == "net-new":
        documents_key, outlinks_key = "net_new_documents", "net_new_outlinks"
    else:
        raise ValueError(f"unsupported RFC selection: {selection}")
    counts = run.get("counts") or {}
    documents = int(counts.get(documents_key) or 0)
    outlinks = int(counts.get(outlinks_key) or 0)
    empty = int(counts.get("empty_outlink_parents") or 0)
    chunks = int((pack.get("counts") or {}).get("chunks") or 0)
    if not documents or chunks < documents or outlinks + empty < documents:
        raise ValueError("RFC selection count conservation is invalid")
    return selection, documents, outlinks, empty, chunks


def audit(*, parent_root: Path, parent_run: Path, pack_root: Path,
          output_root: Path, profile_registry: Path,
          artifact_root: Path | None = None) -> tuple[Path, Path]:
    artifact_root = (artifact_root or parent_root).resolve()
    run = json.loads(parent_run.read_text(encoding="utf-8"))
    shared.validate_run_stages(run, artifact_root)
    if (run.get("status") != "complete"
            or (run.get("counts") or {}).get("production_overlap") != 4562
            or (run.get("counts") or {}).get("production_rfc_numbers") != 4562
            or (run.get("counts") or {}).get("source_documents") != 9821):
        raise ValueError("RFC source-native subtraction conservation drifted")
    pack_path = pack_root / "PACK.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    if pack.get("source_profile") != "software_docs":
        raise ValueError("RFC PACK must declare software_docs")
    _validate_stage(pack["chunks"], "pack.chunks", artifact_root=pack_root, deep=True)
    selection, expected_documents, expected_outlinks, expected_empty, expected_chunks = (
        _selection_counts(run, pack))
    markdown = shared.stage_rows(artifact_root, run["stages"]["markdown"], single_shard=True)
    outlinks = shared.stage_rows(artifact_root, run["stages"]["outlinks"], single_shard=True)
    empty = shared.stage_rows(artifact_root, run["stages"]["outlink_empty_parents"], single_shard=True)
    chunks = shared.stage_rows(pack_root, pack["chunks"], single_shard=True)
    if (len(markdown), len(outlinks), len(empty), len(chunks)) != (
            expected_documents, expected_outlinks, expected_empty, expected_chunks):
        raise ValueError("RFC terminal stage counts drifted")

    original_checked = _validate_stage(
        run["stages"]["original"], "parent.original",
        artifact_root=artifact_root, deep=True)
    original_receipt = json.loads((artifact_root / original_checked["receipt"]["uri"].removeprefix(
        "artifact://")).read_text(encoding="utf-8"))
    source_files: dict[str, Path] = {}
    for artifact in original_receipt["artifacts"]:
        uri = str(artifact.get("uri") or "")
        marker = "/input/rfcs/"
        if marker not in uri:
            continue
        relative = "input/rfcs/" + uri.rsplit(marker, 1)[1]
        if relative in source_files:
            raise ValueError(f"duplicate RFC source artifact: {relative}")
        source_files[relative] = artifact_root / uri.removeprefix("artifact://")

    parents: dict[str, dict] = {}
    heading_documents = 0
    heading_total = 0
    for row in markdown:
        document_id = str(row.get("document_id") or "")
        match = FRONT.match(str(row.get("markdown") or ""))
        url_match = URL.fullmatch(str(row.get("url") or ""))
        if not match or not url_match or match.group("number") != url_match.group("number"):
            raise ValueError("RFC Markdown lacks exact number/path/version front matter")
        number = int(match.group("number"))
        if document_id != f"rfc-editor:rfc:{number}" or document_id in parents:
            raise ValueError("RFC document identity is invalid or duplicated")
        if (row.get("source_relative_path") != match.group("path")
                or row.get("source_sha256") != match.group("sha")
                or row.get("source_text_encoding") != match.group("encoding")):
            raise ValueError("RFC Markdown lineage fields disagree with front matter")
        source_path = source_files.get(row["source_relative_path"])
        if source_path is None:
            raise ValueError("RFC parent source is absent from original-stage inventory")
        raw = source_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["source_sha256"]:
            raise ValueError("RFC source bytes disagree with Markdown lineage")
        encoding = row["source_text_encoding"]
        source = raw.decode("utf-8-sig" if encoding == "utf-8" else encoding)
        restored, headings = _restore_decorated_source(
            row["markdown"], source, match.end())
        if (restored != source or headings != row.get("source_heading_count")
                or hashlib.sha256(source.encode()).hexdigest()
                != row.get("source_reconstruction_sha256")):
            raise ValueError("RFC full-source reconstruction evidence failed")
        heading_documents += headings > 0
        heading_total += headings
        parents[document_id] = row

    linked: set[str] = set()
    for row in outlinks:
        if row.get("document_id") not in parents or not row.get("target_url"):
            raise ValueError("RFC outlink is not parent-bound")
        linked.add(row["document_id"])
    empty_ids = {str(row.get("document_id") or "") for row in empty}
    if (len(empty_ids) != len(empty) or empty_ids != set(parents) - linked
            or any(row.get("status") != "no_external_outlinks" for row in empty)):
        raise ValueError("RFC explicit empty-outlink inventory is incomplete")

    point_ids: set[str] = set()
    ordinals: dict[str, list[int]] = {}
    expected_parent_sha = {
        key: hashlib.sha256(str(value["markdown"]).encode()).hexdigest()
        for key, value in parents.items()}
    metadata_keys = ("title", "source_relative_path", "source_sha256",
                     "source_text_encoding", "source_heading_count",
                     "source_reconstruction_sha256")
    for row in chunks:
        point_id = str(row.get("point_id") or "")
        uuid.UUID(point_id)
        document_id = str(row.get("document_id") or "")
        if point_id in point_ids or document_id not in parents:
            raise ValueError("RFC chunk point identity/parent conservation failed")
        point_ids.add(point_id)
        if (row.get("parent_markdown_sha256") != expected_parent_sha[document_id]
                or not row.get("section_heading") or not row.get("canonical_text")
                or not row.get("text") or not row.get("sparse_text")):
            raise ValueError("RFC chunk lost anchor/content/parent digest")
        metadata = row.get("source_metadata") or {}
        parent = parents[document_id]
        if any(metadata.get(key) != parent.get(key) for key in metadata_keys):
            raise ValueError("RFC chunk lost path/heading/version lineage")
        if row.get("url") != parent.get("url"):
            raise ValueError("RFC chunk URL differs from parent")
        ordinals.setdefault(document_id, []).append(row.get("chunk_ordinal"))
    if set(ordinals) != set(parents) or any(
            sorted(values) != list(range(len(values))) for values in ordinals.values()):
        raise ValueError("RFC chunk ordinal/document conservation failed")

    registry = json.loads(profile_registry.read_text(encoding="utf-8"))
    requirements = registry["profiles"]["software_docs"]
    methods = {
        "original": [
            f"Deep-rehashed the source-native RFC membership and all {expected_documents:,} admitted exact RFC text files; source acquisition is bound to the RFC Editor rsync snapshot and {selection} RUN SHA.",
            "Read and SHA-256 checked every admitted source file against its L1 lineage; the original stage exhaustively inventories membership plus all exact files.",
        ],
        "markdown": [
            f"Scanned all {expected_documents:,} RFC parents and reversibly removed heading decoration to reproduce every complete decoded source; {heading_documents} documents carry {heading_total} detected source headings.",
            "Checked every parent RFC number, immutable RFC Editor path/URL, source digest and encoding as the source-version identity.",
            "Checked every parent source SHA and source reconstruction SHA against the exact preserved source bytes.",
        ],
        "chunks": [
            f"Scanned all {expected_chunks:,} chunks for source path, non-empty section anchor, contiguous parent-local ordinal, RFC URL and parent identity.",
            "Recomputed every parent canonical Markdown digest and matched it on every chunk.",
            "Deep-validated the portable pack receipt plus pinned policy SHA, tokenizer revision and fast-tokenizer identity.",
        ],
        "outlinks": [
            f"Scanned all {expected_outlinks:,} outlinks for RFC parent/release identity; every row is bound to an admitted RFC document.",
            "Checked every extracted RFC reference target is non-empty and parent-bound.",
            f"Deep-validated the hashed inventory of exactly {expected_empty:,} admitted parents with no external outlinks.",
        ],
        "vectors": [],
    }
    checks = shared.profile_checks(requirements, methods, label="RFC")
    evidence = {
        "schema": "aw.rfc-software-docs-profile-evidence.v1",
        "source_id": str(run.get("source_id") or f"rfc-editor-{selection}"),
        "selection": selection,
        "counts": {"source_documents": 9821, "production_overlap": 4562,
                   "markdown_parents": len(markdown), "heading_documents": heading_documents,
                   "source_headings": heading_total, "chunks": len(chunks),
                   "unique_point_ids": len(point_ids), "outlinks": len(outlinks),
                   "empty_outlink_parents": len(empty)},
        "logical_sha256": {
            "original": run["stages"]["original"]["logical_sha256"],
            "markdown": run["stages"]["markdown"]["logical_sha256"],
            "outlinks": run["stages"]["outlinks"]["logical_sha256"],
            "chunks": pack["chunks"]["logical_sha256"],
        },
        "source_parent_run_sha256": sha256_file(parent_run),
        "pack_sha256": sha256_file(pack_path),
        "profile_registry_sha256": sha256_file(profile_registry),
    }
    return shared.publish_evidence(output_root, evidence, checks, label="RFC")


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog or "audit_profile.py", description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path,
                        help="global artifact:// root; defaults to --parent-root for legacy packs")
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile-registry", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence, checks = audit(
        parent_root=args.parent_root.resolve(), parent_run=args.parent_run.resolve(),
        pack_root=args.pack_root.resolve(), output_root=args.output_root.resolve(),
        profile_registry=args.profile_registry.resolve(),
        artifact_root=args.artifact_root.resolve() if args.artifact_root else None)
    print(json.dumps({"evidence": str(evidence), "checks": str(checks)},
                     indent=2, sort_keys=True))
    return 0
