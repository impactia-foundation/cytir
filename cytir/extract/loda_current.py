#!/usr/bin/env python3
"""Project exact current-production LODA IDs from sealed DILA LEGI XML.

LODA and named codes share the official LEGI XML containers, but they have
different canonical web identities.  The sealed production membership is the
only selector: this producer accepts rows explicitly labelled ``provenance =
loda`` and rejects same-domain rows or malformed LODA URLs.
"""
from __future__ import annotations
import argparse
import gzip
from cytir.acquire.gzip_pinned import deterministic_gzip_open
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile

from cytir.acquire.artifact_io import canonical_json, sha256_file
from cytir.extract.legi import parse_article_xml_legal_markdown
from cytir.acquire import stage_receipt as SS


VERSION = "loda-exact-production-anchored-markdown.v1"
PREFIX = "https://www.legifrance.gouv.fr/loda/article_lc/"
ARTICLE_ID = re.compile(r"^LEGIARTI[0-9]+$")


def _point_id(row: dict, payload: dict, identifier: str) -> str:
    values = {str(value) for value in
              (row.get("id"), row.get("point_id"), payload.get("point_id"))
              if value is not None}
    if len(values) != 1:
        raise ValueError(f"production point identity conflict: {identifier}")
    return next(iter(values))


def membership_witness(*, artifact_root: Path, membership: Path,
                       expected_points: int,
                       expected_source_ids: int) -> tuple[dict[str, dict], dict]:
    receipt_path = membership / "RECEIPT.json"
    receipt_body = receipt_path.read_bytes()
    receipt = json.loads(receipt_body)
    wanted: dict[str, dict] = {}
    seen_point_ids: set[str] = set()
    scanned = selected = 0
    for reference in receipt.get("shards") or []:
        path = membership / reference["file"]
        if path.stat().st_size != reference["bytes"]:
            raise ValueError(f"membership shard byte drift: {path}")
        if reference.get("sha256") and sha256_file(path) != reference["sha256"]:
            raise ValueError(f"membership shard digest drift: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                scanned += 1
                row = json.loads(line); payload = row.get("payload") or {}
                if payload.get("provenance") != "loda":
                    continue
                url = str(payload.get("url") or "")
                identifier = url.removeprefix(PREFIX).strip("/") \
                    if url.startswith(PREFIX) else ""
                if not ARTICLE_ID.fullmatch(identifier):
                    raise ValueError(f"invalid LODA production URL: {url}")
                item = wanted.setdefault(identifier, {
                    "url": url, "point_ids": [], "titles": []})
                if item["url"] != url:
                    raise ValueError(f"LODA source identity conflict: {identifier}")
                point_id = _point_id(row, payload, identifier)
                if point_id in seen_point_ids:
                    raise ValueError(f"duplicate LODA production point ID: {point_id}")
                seen_point_ids.add(point_id)
                item["point_ids"].append(point_id)
                item["titles"].append(str(payload.get("title") or ""))
                selected += 1
    if scanned != receipt.get("rows"):
        raise ValueError("membership row conservation failed")
    if selected != expected_points:
        raise ValueError(
            f"LODA production point count drift: {selected} != {expected_points}")
    if len(wanted) != expected_source_ids:
        raise ValueError(
            f"LODA source ID count drift: {len(wanted)} != {expected_source_ids}")
    multiplicity_groups = sum(
        len(item["point_ids"]) > 1 for item in wanted.values())
    return wanted, {
        "uri": f"artifact://{receipt_path.resolve().relative_to(artifact_root).as_posix()}",
        "bytes": len(receipt_body),
        "sha256": hashlib.sha256(receipt_body).hexdigest(),
        "rows_scanned": scanned, "selected_points": selected,
        "selected_source_ids": len(wanted), "selector": {"provenance": "loda",
            "url_prefix": PREFIX},
        "multiplicity_groups": multiplicity_groups,
        "multiplicity_excess_points": selected - len(wanted)}


def _write(handle, logical: hashlib._Hash, row: dict) -> None:
    encoded = (canonical_json(row) + "\n").encode()
    handle.write(encoded); logical.update(encoded)


def _archive_cohort(member: str) -> str:
    if "/code_et_TNC_en_vigueur/" in f"/{member}":
        return "official-current"
    if "/code_et_TNC_non_vigueur/" in f"/{member}":
        return "official-historical"
    return "unclassified"


def _resolve_candidates(identifier: str, candidates: list[dict]) -> tuple[dict, str]:
    """Select one source-native XML version without relying on tar order.

    DILA's global LEGI archive can carry the same immutable article ID in both
    its current and historical cohorts while status metadata differs.  The
    current cohort is authoritative for the reconstructed current source
    record.  Multiple divergent current candidates (or divergent candidates
    without a current-cohort witness) remain ambiguous and fail closed.
    """
    by_digest: dict[str, list[dict]] = {}
    for candidate in candidates:
        by_digest.setdefault(candidate["source_record_sha256"], []).append(candidate)
    current = [row for row in candidates if row["archive_cohort"] == "official-current"]
    current_digests = {row["source_record_sha256"] for row in current}
    if current and len(current_digests) == 1:
        return min(current, key=lambda row: row["archive_member"]), (
            "official-current-cohort-precedence" if len(by_digest) > 1
            else "identical-source-record-current-cohort")
    if len(by_digest) == 1:
        return min(candidates, key=lambda row: (
            row["archive_cohort"] != "official-current", row["archive_member"])), \
            "identical-source-record-deterministic-member"
    raise ValueError(
        f"ambiguous divergent immutable LODA ID: {identifier}; "
        f"candidates={len(candidates)} current_candidates={len(current)} "
        f"current_digests={len(current_digests)}")


def derive(*, artifact_root: Path, membership: Path, l0_receipt: Path,
           output: Path, expected_points: int,
           expected_source_ids: int) -> dict:
    artifact_root = artifact_root.resolve(); output = output.resolve()
    output.relative_to(artifact_root)
    if output.exists():
        raise ValueError(f"immutable output exists: {output}")
    wanted, membership_ref = membership_witness(
        artifact_root=artifact_root, membership=membership.resolve(),
        expected_points=expected_points,
        expected_source_ids=expected_source_ids)
    l0_body = l0_receipt.read_bytes(); l0 = json.loads(l0_body)
    archive_ref = next(row for row in l0["artifacts"]
                       if str(row["uri"]).endswith(".tar.gz"))
    archive = artifact_root / archive_ref["uri"].removeprefix("artifact://")
    if (archive.stat().st_size != archive_ref["bytes"] or
            sha256_file(archive) != archive_ref["sha256"]):
        raise ValueError("sealed LEGI global archive binding mismatch")

    partial = output.with_name(output.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    try:
        candidates_path = partial / "archive-candidates.jsonl.gz"
        markdown_path = partial / "markdown.jsonl.gz"
        outlinks_path = partial / "outlinks.jsonl.gz"
        markdown_logical = hashlib.sha256(); outlinks_logical = hashlib.sha256()
        candidates: dict[str, list[dict]] = {}
        candidate_rows = 0
        with candidates_path.open("wb") as craw:
            with deterministic_gzip_open(craw, compresslevel=9) as candidate_file, \
                    tarfile.open(archive, "r|gz") as source:
                for member in source:
                    identifier = Path(member.name).stem
                    if not member.isfile() or identifier not in wanted:
                        continue
                    extracted = source.extractfile(member)
                    if extracted is None:
                        raise ValueError(f"cannot extract DILA member: {member.name}")
                    xml = extracted.read()
                    row = parse_article_xml_legal_markdown(
                        xml, archive_member=member.name,
                        archive_sha256=archive_ref["sha256"],
                        source_release="2025-07-13-global",
                        article_url_family="loda",
                        source_id="dila-loda-exact-production-v1")
                    evidence = wanted[identifier]
                    if row is None or row["url"] != evidence["url"]:
                        raise ValueError(f"LODA source semantic drift: {identifier}")
                    candidate_file.write((canonical_json(row) + "\n").encode())
                    candidates.setdefault(identifier, []).append({
                        "archive_member": member.name,
                        "archive_cohort": _archive_cohort(member.name),
                        "source_record_sha256": row["source_record_sha256"],
                        "markdown_sha256": row["markdown_sha256"],
                        "source_version": row["source_version"],
                    })
                    candidate_rows += 1

        selected: dict[str, dict] = {}
        resolution_groups = []
        for identifier, rows in sorted(candidates.items()):
            chosen, method = _resolve_candidates(identifier, rows)
            selected[identifier] = chosen
            if len(rows) > 1:
                resolution_groups.append({
                    "source_record_id": identifier, "method": method,
                    "selected_archive_member": chosen["archive_member"],
                    "candidates": sorted(rows, key=lambda row: row["archive_member"]),
                })
        resolution_path = partial / "source-record-version-resolution.json"
        resolution_path.write_text(json.dumps({
            "schema": "aw.loda-source-record-version-resolution.v1",
            "archive_candidates": candidate_rows,
            "source_ids_with_candidates": len(candidates),
            "duplicate_source_id_groups": len(resolution_groups),
            "policy": "official-current-cohort-precedence; otherwise exact-digest-only",
            "groups": resolution_groups,
        }, indent=2, sort_keys=True) + "\n")

        seen: set[str] = set(); records = links = unresolved = empty = 0
        with markdown_path.open("wb") as mraw, outlinks_path.open("wb") as oraw:
            with deterministic_gzip_open(mraw, compresslevel=9) as markdown, \
                    deterministic_gzip_open(oraw, compresslevel=9) as outlinks, \
                    gzip.open(candidates_path, "rt", encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    identifier = row["source_record_id"]
                    if (identifier not in selected or
                            row["source_archive"]["member"] !=
                            selected[identifier]["archive_member"]):
                        continue
                    if identifier in seen:
                        raise ValueError(f"selected LODA source ID duplicated: {identifier}")
                    evidence = wanted[identifier]
                    row["baseline_production_point_ids"] = sorted(evidence["point_ids"])
                    row["baseline_production_titles"] = sorted(set(evidence["titles"]))
                    _write(markdown, markdown_logical, row)
                    link_row = {
                        "schema": "aw.parent-source-links.v1",
                        "source_id": row["source_id"],
                        "source_record_id": identifier, "url": row["url"],
                        "article_anchor": row["anchor"],
                        "source_version": row["source_version"],
                        "parent_markdown_sha256": row["markdown_sha256"],
                        "parent_identity": row["source_archive"],
                        "links": row["source_links"],
                        "unresolved_relations": row["source_relations_unresolved"],
                    }
                    _write(outlinks, outlinks_logical, link_row)
                    links += len(row["source_links"])
                    unresolved += len(row["source_relations_unresolved"])
                    empty += not row["source_links"]
                    seen.add(identifier); records += 1
                    if records and records % 20_000 == 0:
                        print(f"[loda-global] records={records:,}/{len(wanted):,}",
                              flush=True)
        missing = sorted(set(wanted) - seen)
        missing_path = partial / "missing-production-ids.json"
        missing_path.write_text(json.dumps({
            "schema": "aw.loda-production-recovery-frontier.v1",
            "production_source_ids": len(wanted), "missing_source_ids": missing,
        }, indent=2, sort_keys=True) + "\n")
        multiplicity_path = partial / "production-point-multiplicities.json"
        multiplicities = [{
            "source_record_id": identifier,
            "url": item["url"],
            "baseline_production_point_ids": sorted(item["point_ids"]),
            "baseline_production_titles": sorted(set(item["titles"])),
        } for identifier, item in sorted(wanted.items())
            if len(item["point_ids"]) > 1]
        multiplicity_path.write_text(json.dumps({
            "schema": "aw.loda-production-point-multiplicities.v1",
            "production_points": membership_ref["selected_points"],
            "source_ids": len(wanted),
            "multiplicity_groups": len(multiplicities),
            "multiplicity_excess_points": sum(
                len(row["baseline_production_point_ids"]) - 1
                for row in multiplicities),
            "groups": multiplicities,
        }, indent=2, sort_keys=True) + "\n")
        markdown_stage = SS.seal(
            artifact_root=artifact_root,
            artifact_paths=[markdown_path, missing_path, multiplicity_path,
                            candidates_path, resolution_path],
            receipt_path=partial / "MARKDOWN-RECEIPT.json", records=records,
            fmt="anchored-law-loda-global+deterministic-gzip+recovery-frontier",
            logical_sha256=markdown_logical.hexdigest(),
            # frozen label: sealed receipts embed this string (never changes; task 1883)
            producer_id="pipeline/derive_loda_production_markdown.py",
            producer_version=VERSION, publish_from=partial, publish_to=output)
        outlink_stage = SS.seal(
            artifact_root=artifact_root, artifact_paths=[outlinks_path],
            receipt_path=partial / "OUTLINKS-RECEIPT.json", records=records,
            fmt="parent-bound-loda-legal-relations+deterministic-gzip",
            logical_sha256=outlinks_logical.hexdigest(),
            producer_id="pipeline/derive_loda_production_markdown.py",
            producer_version=VERSION, publish_from=partial, publish_to=output)
        run = {
            "schema": "aw.loda-exact-production-run.v1", "version": VERSION,
            "status": "complete" if not missing else "recovery-frontier",
            "source_profile": "anchored_law",
            "counts": {"production_points": membership_ref["selected_points"],
                       "production_source_ids": len(wanted),
                       "global_recovered_source_ids": records,
                       "archive_candidate_rows": candidate_rows,
                       "source_record_version_resolution_groups":
                       len(resolution_groups),
                       "missing_source_ids": len(missing), "links": links,
                       "unresolved_relations": unresolved,
                       "explicit_empty_outlink_parents": empty},
            "production_point_multiplicities": {
                "groups": membership_ref["multiplicity_groups"],
                "excess_points": membership_ref["multiplicity_excess_points"]},
            "l0": {"uri": f"artifact://{l0_receipt.resolve().relative_to(artifact_root).as_posix()}",
                   "bytes": len(l0_body), "sha256": hashlib.sha256(l0_body).hexdigest()},
            "membership": membership_ref,
            "stages": {"markdown": markdown_stage, "outlinks": outlink_stage},
            "next_gate": ("profile/fidelity/token-safe chunking" if not missing else
                          "recover exact missing IDs from sealed ordered LEGI increments"),
        }
        (partial / "RUN.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
        os.rename(partial, output)
        return run
    except BaseException:
        failed = output.with_name(output.name + f".failed.{os.getpid()}")
        if partial.exists():
            os.rename(partial, failed)
        raise


def main(argv=None, prog=None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--membership-dir", type=Path, required=True)
    parser.add_argument("--l0-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-points", type=int, required=True)
    parser.add_argument("--expected-source-ids", type=int, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(derive(artifact_root=args.artifact_root,
                            membership=args.membership_dir,
                            l0_receipt=args.l0_receipt, output=args.output,
                            expected_points=args.expected_points,
                            expected_source_ids=args.expected_source_ids),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
