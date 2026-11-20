"""Deep-audit source-specific Service-Public fiches as structured feed records."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import uuid
import zipfile

from cytir.acquire.artifact_io import canonical_json, sha256_file
from cytir.chunk import token_counter as dense_token_counter
from cytir.index.preservation_contract import _validate_stage
from cytir.extract.service_public_fiche import render

from cytir.sources import audit as shared


METADATA_KEYS = (
    "title", "source_relative_path", "source_sha256", "source_heading_count",
    "modified_date", "source_version", "fiche_id", "fiche_type", "subject",
    "archive_member", "archive_sha256", "archive_member_sha256")

def _metadata_matches(metadata: dict, parent: dict) -> bool:
    return all(metadata.get(key) == parent.get(key) for key in METADATA_KEYS)


def audit(*, parent_root: Path, parent_run: Path, pack_root: Path,
          output_root: Path, registry_path: Path) -> tuple[Path, Path]:
    run = json.loads(parent_run.read_text(encoding="utf-8"))
    shared.validate_run_stages(run, parent_root)
    if (run.get("status") != "complete"
            or run.get("counts", {}).get("heading_date_version_documents") != 2997
            or run.get("counts", {}).get("empty_outlink_parents") != 117):
        raise ValueError("Service-Public source-specific RUN is not terminal")
    pack_path = pack_root / "PACK.json"
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    if pack.get("source_profile") != "structured_data_feed":
        raise ValueError("Service-Public pack declares the wrong source profile")
    _validate_stage(pack["chunks"], "chunks", artifact_root=pack_root, deep=True)
    markdown = shared.stage_rows(parent_root, run["stages"]["markdown"], single_shard=True)
    outlinks = shared.stage_rows(parent_root, run["stages"]["outlinks"], single_shard=True)
    empty = shared.stage_rows(parent_root, run["stages"]["outlink_empty_parents"], single_shard=True)
    chunks = shared.stage_rows(pack_root, pack["chunks"], single_shard=True)
    if len(markdown) != 2997 or len(outlinks) != 45421 or len(empty) != 117:
        raise ValueError("Service-Public parent/outlink counts drifted")

    original_receipt = json.loads((parent_root / run["stages"]["original"][
        "receipt"]["uri"].removeprefix("artifact://")).read_text(encoding="utf-8"))
    bodies = [parent_root / row["uri"].removeprefix("artifact://")
              for row in original_receipt["artifacts"] if "/body/" in row["uri"]]
    if len(bodies) != 1:
        raise ValueError("expected one complete DILA bulk response body")
    archive = gzip.decompress(bodies[0].read_bytes())
    archive_sha = hashlib.sha256(archive).hexdigest()
    parents: dict[str, dict] = {}
    headings = 0
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        names = set(source.namelist())
        for row in markdown:
            member = str(row.get("archive_member") or "")
            if member not in names or row.get("source_relative_path") != member:
                raise ValueError("Service-Public archive member lineage is missing")
            raw = source.read(member)
            expected = render(raw, archive_sha, member)
            if expected is None or row != expected:
                raise ValueError("Service-Public Markdown differs from exact source-specific rerender")
            document_id = row["document_id"]
            if document_id in parents:
                raise ValueError("duplicate Service-Public fiche identity")
            if (not row.get("modified_date") or not row.get("source_version")
                    or not row.get("fiche_id") or not row.get("fiche_type")
                    or not row.get("source_heading_count")):
                raise ValueError("Service-Public fiche semantic metadata is incomplete")
            headings += int(row["source_heading_count"])
            parents[document_id] = row

    linked: set[str] = set()
    for row in outlinks:
        if (row.get("document_id") not in parents or not row.get("source_url")
                or not row.get("target_url") or not row.get("anchor_or_context")):
            raise ValueError("Service-Public outlink lacks parent/relation/target")
        linked.add(row["document_id"])
    empty_ids = {str(row.get("document_id") or "") for row in empty}
    if (len(empty_ids) != len(empty) or empty_ids != set(parents) - linked
            or any(row.get("status") != "no_external_outlinks" for row in empty)):
        raise ValueError("Service-Public explicit empty-outlink inventory is incomplete")

    ids: set[str] = set()
    ordinals: dict[str, list[int]] = {}
    policy_path = registry_path.parent.parent / pack["chunk_policy"]["path"]
    if sha256_file(policy_path) != pack["chunk_policy"]["sha256"]:
        raise ValueError("Service-Public pack policy bytes disagree with manifest")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    token_max = int(policy["token_hard_max"]["exact"])
    tokenizer = dense_token_counter.resolve_tokenizer(
        policy["token_hard_max"]["tokenizer_model"],
        pack["chunk_policy"]["tokenizer_revision"])
    measured_token_max = 0
    parent_sha = {key: hashlib.sha256(canonical_json(value).encode()).hexdigest()
                  for key, value in parents.items()}
    for row in chunks:
        point_id = str(row.get("point_id") or "")
        uuid.UUID(point_id)
        document_id = str(row.get("document_id") or "")
        if point_id in ids or document_id not in parents:
            raise ValueError("Service-Public chunk point/parent identity failure")
        ids.add(point_id)
        if (row.get("parent_markdown_sha256") != parent_sha[document_id]
                or row.get("url") != parents[document_id]["url"]
                or not row.get("section_heading") or not row.get("canonical_text")
                or not row.get("text") or not row.get("sparse_text")
                or not _metadata_matches(row.get("source_metadata") or {},
                                         parents[document_id])):
            raise ValueError("Service-Public chunk lost fiche hierarchy/version/member lineage")
        dense_tokens = dense_token_counter.count(tokenizer, row["text"])
        measured_token_max = max(measured_token_max, dense_tokens)
        if dense_tokens > token_max:
            raise ValueError("Service-Public dense chunk exceeds pinned model token maximum")
        ordinals.setdefault(document_id, []).append(row.get("chunk_ordinal"))
    if set(ordinals) != set(parents) or any(
            sorted(values) != list(range(len(values))) for values in ordinals.values()):
        raise ValueError("Service-Public chunk document/ordinal conservation failure")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    profile = registry["profiles"]["structured_data_feed"]
    methods = {
        "original": [
            "Deep-rehashed the complete official DILA bulk response body plus transport headers/metadata and exact ZIP bytes.",
            "Rerendered all 2,997 fiche members by stable fiche ID, archive member and modified-date/member-digest source version.",
            "SHA-256 checked the archive and every admitted XML member against all L1 lineage fields.",
        ],
        "markdown": [
            f"Exact-rerendered all 2,997 XML members; fiche title, subject, type and {headings} hierarchy headings remain explicit semantic fields.",
            "Scanned every record for unique fiche/document ID, modified date, source version and official source URL.",
            "Matched every Markdown parent to archive SHA, member path and member SHA in the complete official feed.",
        ],
        "chunks": [
            f"All {len(chunks)} chunks remain within one of exactly 2,997 fiche parents with contiguous parent-local ordinals; no source records are merged.",
            f"Recomputed all 2,997 canonical parent Markdown digests and matched them across {len(chunks)} chunks.",
            "Deep-validated the source-specific fiche projection lineage plus portable structured chunk policy SHA, tokenizer revision and channel provenance.",
            f"Counted every dense channel with the exact pinned tokenizer; measured maximum {measured_token_max} tokens is within the {token_max}-token model limit.",
        ],
        "outlinks": [
            "Scanned all 45,421 links for fiche document ID, official source URL and relation/context.",
            "Checked every emitted link has a resolved non-empty external target.",
            "Deep-validated the hashed inventory of exactly 117 fiche parents with no external outlinks.",
        ],
        "vectors": [],
    }
    checks = shared.profile_checks(profile, methods, label="Service-Public")
    evidence = {
        "schema": "aw.service-public-structured-profile-evidence.v1",
        "source_id": "service-public-dila-fiche-v4",
        "counts": {"original_bulk_responses": 1, "markdown_parents": len(markdown),
                   "source_headings": headings, "chunks": len(chunks),
                   "unique_point_ids": len(ids), "outlinks": len(outlinks),
                   "empty_outlink_parents": len(empty),
                   "measured_dense_token_max": measured_token_max,
                   "pinned_dense_token_max": token_max},
        "logical_sha256": {name: stage["logical_sha256"]
                           for name, stage in run["stages"].items()},
        "chunk_logical_sha256": pack["chunks"]["logical_sha256"],
        "source_parent_run_sha256": sha256_file(parent_run),
        "pack_sha256": sha256_file(pack_path),
        "profile_registry_sha256": sha256_file(registry_path),
    }
    return shared.publish_evidence(output_root, evidence, checks, label="Service-Public")


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog or "audit_profile.py", description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile-registry", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence, checks = audit(
        parent_root=args.parent_root.resolve(), parent_run=args.parent_run.resolve(),
        pack_root=args.pack_root.resolve(), output_root=args.output_root.resolve(),
        registry_path=args.profile_registry.resolve())
    print(json.dumps({"evidence": str(evidence), "checks": str(checks)},
                     indent=2, sort_keys=True))
    return 0
