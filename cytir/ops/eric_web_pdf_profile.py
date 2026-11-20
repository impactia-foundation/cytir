#!/usr/bin/env python3
"""Deep web_pdf profile audit for a measured ERIC PDF tranche."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import uuid

from cytir.acquire.artifact_io import atomic_publish, sha256_file
from cytir.index.preservation_contract import _validate_stage
import cytir.chunk.registry as CHUNK_REG


PROFILE = "web_pdf"
#: The family this audit is FOR -- eric_official_pdf, always (this module's
#: whole purpose is the ERIC web_pdf profile). Kept as a name, not a second
#: hardcoded policy path (task 663, 2026-09-25): a literal POLICY_PATH here
#: duplicated `chain_eric_pdf_partition_admission.sh`'s own policy pin, and
#: the two fell out of sync the moment ADR-113 re-pointed the chunker to the
#: 1024w arm -- the chunk step passed, this audit failed on a pack that was
#: correctly built under the ratified policy. `cytir.chunk.registry` is the
#: ONE place a policy's admissibility for a family is decided; this audit
#: asks it the same question the chunker already asked, rather than keeping
#: its own copy of the answer.
CHUNK_POLICY_FAMILY_ID = "eric_official_pdf"


def _rows(root: Path, stage: dict) -> list[dict]:
    receipt = _validate_stage(stage, "rows", artifact_root=root, deep=True)
    rows = []
    for artifact in receipt["artifacts"]:
        if artifact["uri"].endswith(".jsonl.gz"):
            with gzip.open(root / artifact["uri"].removeprefix("artifact://"),
                           "rt", encoding="utf-8") as stream:
                rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def _validate_diagnostics(*, parents: list[dict], diagnostics: list[dict],
                          quarantine: list[dict]) -> None:
    """Conserve diagnostics for retained and explicitly quarantined PDFs.

    task 656: a retained row now names its own `extractor` (liteparse or
    docling, cytir/extract/pdf.py's router) -- rows sealed before task 656 carry
    no `extractor` field at all and are exactly the pre-router Docling-only
    shape, so a missing field means "docling" (legacy default), not "unknown".
    `do_table_structure` is a Docling-specific pipeline option and is only
    required on docling rows; liteparse has no equivalent toggle.
    """
    retained = {str(row.get("source_record_id") or ""): row for row in parents}
    rejected = {str(row.get("eric_id") or ""): row for row in quarantine}
    if ("" in retained or "" in rejected or set(retained) & set(rejected)
            or len(retained) != len(parents) or len(rejected) != len(quarantine)):
        raise ValueError("ERIC retained/quarantine diagnostic identity failed")
    diagnostic_ids = set()
    for row in diagnostics:
        eric_id = str(row.get("eric_id") or "")
        if not eric_id or eric_id in diagnostic_ids:
            raise ValueError("ERIC diagnostic identity is missing or duplicate")
        diagnostic_ids.add(eric_id)
        if eric_id in retained:
            parent = retained[eric_id]
            extractor = row.get("extractor") or "docling"
            version = row.get("extractor_version") or row.get("docling_version")
            ok = (row.get("pdf_sha256") == parent["parent_identity"]["pdf_sha256"]
                  and bool(version)
                  and row.get("do_ocr") is False
                  and row.get("empty_markdown") is False
                  and isinstance(row.get("page_count"), int)
                  and row.get("page_count") > 0)
            if extractor == "docling":
                ok = ok and row.get("do_table_structure") is True
            if not ok:
                raise ValueError("ERIC retained page-level extraction diagnostics are incomplete")
            continue
        rejected_row = rejected.get(eric_id)
        source_pdf = (rejected_row or {}).get("source_pdf") or {}
        if (rejected_row is None
                or row.get("pdf_sha256") != source_pdf.get("sha256")
                or row.get("pdf_bytes") != source_pdf.get("bytes")
                or row.get("do_ocr") is not False
                or row.get("do_table_structure") is not True
                or row.get("status") not in {"extraction_failure", "wall_time_timeout"}
                or row.get("page_count") is not None):
            raise ValueError("ERIC quarantine diagnostics are incomplete")
    if diagnostic_ids != set(retained) | set(rejected):
        raise ValueError("ERIC diagnostics lost a selected PDF")


def audit(*, artifact_root: Path, parent_run: Path, pack_root: Path,
          output_root: Path, registry_path: Path, source_id: str,
          expected_selected: int) -> tuple[Path, Path]:
    run = json.loads(parent_run.read_text())
    counts = run.get("counts") or {}
    if (run.get("schema") != "aw.eric-pdf-materialization-run.v1"
            or run.get("status") != "ready_for_web_pdf_profile_fidelity_and_current_chunking"
            or run.get("source_profile") != PROFILE
            or (run.get("selection") or {}).get("selected_complete_pdfs") != expected_selected
            or counts.get("markdown_documents", 0) + counts.get("docling_quarantine", 0)
            != expected_selected
            or (run.get("extraction") or {}).get("engine") != "aw.extract.pdf"
            or (run.get("extraction") or {}).get("docling_do_ocr") is not False
            or (run.get("extraction") or {}).get("docling_do_table_structure") is not True
            or (run.get("extraction") or {}).get("content_cap") is not None):
        raise ValueError("ERIC web_pdf parent identity/conservation failed")
    for name, stage in run["stages"].items():
        _validate_stage(stage, f"parent.{name}", artifact_root=artifact_root,
                        deep=True)
    parents = _rows(artifact_root, run["stages"]["markdown"])
    outlinks = _rows(artifact_root, run["stages"]["outlinks"])
    diagnostics = _rows(artifact_root, run["stages"]["diagnostics"])
    quarantine = _rows(artifact_root, run["stages"]["quarantine"])
    if len(parents) != counts["markdown_documents"] or len(quarantine) != counts["docling_quarantine"]:
        raise ValueError("ERIC Markdown/quarantine conservation failed")
    by_id = {}
    by_eric = {}
    for row in parents:
        document_id = str(row.get("document_id") or "")
        eric_id = str(row.get("source_record_id") or "")
        markdown = str(row.get("markdown") or "")
        identity = row.get("parent_identity") or {}
        if (not document_id or document_id in by_id or not eric_id
                or eric_id in by_eric or not str(row.get("url") or "").startswith("https://")
                or not markdown.strip()
                or row.get("markdown_sha256") != hashlib.sha256(markdown.encode()).hexdigest()
                or identity.get("eric_id") != eric_id
                or not identity.get("pdf_sha256")
                or not (row.get("parent_response") or {}).get("response_body")):
            raise ValueError("ERIC parent Markdown lost PDF/source identity")
        by_id[document_id] = row
        by_eric[eric_id] = row
    _validate_diagnostics(parents=parents, diagnostics=diagnostics,
                          quarantine=quarantine)
    linked = set(); empty = set(); external_links = 0
    for row in outlinks:
        document_id = str(row.get("document_id") or "")
        parent = by_id.get(document_id)
        links = row.get("links") or []
        if parent is None or row.get("parent_url") != parent.get("url"):
            raise ValueError("ERIC outlinks lost parent identity")
        if row.get("explicit_empty"):
            if links:
                raise ValueError("ERIC explicit-empty parent has links")
            empty.add(document_id)
        else:
            if not links or any(not item.get("target_url") or not item.get("context")
                                for item in links):
                raise ValueError("ERIC PDF URI annotation lacks target/context")
            linked.add(document_id); external_links += len(links)
    if linked | empty != set(by_id) or linked & empty or len(outlinks) != len(parents):
        raise ValueError("ERIC outlink parent conservation failed")

    pack_path = pack_root / "PACK.json"
    pack = json.loads(pack_path.read_text())
    policy = pack.get("chunk_policy") or {}
    try:
        # The registry, not a literal path here, decides whether THIS
        # policy may back a NEW eric_official_pdf materialization -- the
        # same question `chain_eric_pdf_partition_admission.sh` already
        # asked before chunking. A pack sealed under a policy the registry
        # would refuse (superseded, revoked admission, unregistered) fails
        # here; a pack under any policy the registry currently admits for
        # this family (the ratified default, OR a family-scoped arm like
        # ADR-113's) passes, with no second literal to keep in sync.
        CHUNK_REG.assert_selectable(policy.get("sha256"),
                                    family_id=CHUNK_POLICY_FAMILY_ID)
        policy_admitted = True
    except CHUNK_REG.ChunkPolicyRegistryError:
        policy_admitted = False
    if (pack.get("status") != "ready_for_gpu_queue"
            or pack.get("source_profile") != PROFILE
            or (pack.get("counts") or {}).get("documents") != len(parents)
            or not policy_admitted
            or (pack.get("token_distribution") or {}).get("over_hard_max") != 0
            or (pack.get("token_distribution") or {}).get("max", 2561) > 2560):
        raise ValueError("ERIC web_pdf pack policy/token gate failed")
    chunks = _rows(artifact_root, pack["chunks"])
    point_ids = set(); ordinals: dict[str, list[int]] = {}; sections = 0
    for row in chunks:
        point_id = str(row.get("point_id") or ""); uuid.UUID(point_id)
        document_id = str(row.get("document_id") or "")
        parent = by_id.get(document_id)
        if (point_id in point_ids or parent is None
                or row.get("source_record_id") != parent.get("source_record_id")
                or row.get("parent_identity") != parent.get("parent_identity")
                or row.get("parent_markdown_sha256") != parent.get("markdown_sha256")):
            raise ValueError("ERIC chunk lost point/PDF/parent identity")
        point_ids.add(point_id)
        ordinals.setdefault(document_id, []).append(row.get("chunk_ordinal"))
        sections += bool(str(row.get("section_heading") or "").strip())
    if (set(ordinals) != set(by_id)
            or any(sorted(values) != list(range(len(values))) for values in ordinals.values())
            or sections != len(chunks)):
        raise ValueError("ERIC chunk section/ordinal conservation failed")

    registry = json.loads(registry_path.read_text())
    profile = registry["profiles"][PROFILE]
    methods = {
        "original": [
            f"Deep-rehashed the exact selected official PDF response stage containing {expected_selected} source records; this measured selection retains {len(parents)} and quarantines {len(quarantine)}.",
            "Matched every retained ERIC source record to its globally portable response body, headers/transport receipt, URL, byte count and SHA-256."],
        "markdown": [
            f"Scanned all {len(parents)} full uncapped Markdown parents including headings and tables.",
            (f"Verified the cytir.extract.pdf router {run['extraction'].get('router_version', run['extraction'].get('version'))} "
             "(liteparse by default, Docling routed per task 655/656 for short+complex/table-dense documents) with OCR "
             "disabled on every retained parent, and table structure enabled on every parent Docling extracted "
             f"({sum(1 for row in diagnostics if str(row.get('eric_id') or '') in by_eric and (row.get('extractor') or 'docling') == 'docling')} of {len(parents)})."),
            f"Matched {len(diagnostics)} page-level diagnostics to exact parent PDF digests and positive page counts."],
        "chunks": [
            f"Verified nonempty section provenance for all {len(chunks)} document-local chunks.",
            "Recomputed parent Markdown digests and matched every chunk to its parent PDF identity.",
            f"Deep-validated the pinned structured policy/tokenizer; max {(pack.get('token_distribution') or {}).get('max')} <=2560 and zero over-limit."],
        "outlinks": [
            f"Verified {external_links} URI annotations retain target plus extraction context.",
            f"Verified all {len(parents)} parents have one receipted outlink representation, including {len(empty)} explicit empty parents."],
        "vectors": [],
    }
    checks = {}
    for stage in profile["required_stages"]:
        requirements = profile.get(stage, {}).get("requirements", [])
        if len(requirements) != len(methods[stage]):
            raise ValueError(f"ERIC profile method count mismatch: {stage}")
        checks[stage] = [{"requirement": requirement, "status": "PASS",
                          "verification_method": method}
                         for requirement, method in zip(requirements, methods[stage])]
    evidence = {
        "schema": "aw.eric-web-pdf-profile-evidence.v1",
        "source_id": source_id,
        "counts": {"selected": expected_selected, "parents": len(parents),
                   "quarantine": len(quarantine), "chunks": len(chunks),
                   "unique_point_ids": len(point_ids), "outlink_parents": len(outlinks),
                   "external_uri_annotations": external_links,
                   "explicit_empty_parents": len(empty)},
        "parent_run_sha256": sha256_file(parent_run),
        "pack_sha256": sha256_file(pack_path),
        "profile_registry_sha256": sha256_file(registry_path),
        "classification": "bounded measured official-PDF tranche; catalog pointers and unmaterialized PDFs are not documents or points",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_path = output_root / "verification-evidence.json"
    checks_path = output_root / "checks.json"
    for path, value in ((evidence_path, evidence), (checks_path, checks)):
        body = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        if path.exists() and path.read_bytes() != body:
            raise ValueError(f"immutable ERIC audit collision: {path}")
        if not path.exists(): atomic_publish(path, body)
    return evidence_path, checks_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "parent-run", "pack-root", "output-root",
                 "profile-registry"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--expected-selected", type=int, required=True)
    args = parser.parse_args()
    evidence, checks = audit(
        artifact_root=args.artifact_root.resolve(),
        parent_run=args.parent_run.resolve(), pack_root=args.pack_root.resolve(),
        output_root=args.output_root.resolve(), registry_path=args.profile_registry.resolve(),
        source_id=args.source_id, expected_selected=args.expected_selected)
    print(json.dumps({"evidence": str(evidence), "checks": str(checks)}, indent=2))


if __name__ == "__main__": main()
