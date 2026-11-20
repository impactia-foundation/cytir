"""Materialize the exact current Justice residual from pinned official XML.

Moved from pipeline/materialize_justice_canada_residual.py (task 1724); the functions are the
old file's, byte for byte. The `PRODUCER` string is a frozen label written into sealed receipts.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file  # noqa: E402
from cytir.acquire.deterministic_gzip import deterministic_gzip_text  # noqa: E402
from cytir.sources.justice_canada.render import (  # noqa: E402
    COMMIT,
    _attrs,
    _tag,
    law_descriptor,
    legal_unit_anchors,
    render_official_markdown,
    source_links_with_anchors,
)
from cytir.index.preservation_contract import _validate_stage  # noqa: E402
from cytir.acquire import stage_receipt as SS  # noqa: E402


VERSION = "justice-canada-current-official-xml-residual-materialization.v1"
PRODUCER = "pipeline/materialize_justice_canada_residual.py"
SOURCE_ID = "justice-canada-current-official-xml-residual-v1"
EXPECTED_CLASSES = {
    "eng/regulations": {"points": 38_321, "urls": 3_036},
    "fra/lois": {"points": 35_155, "urls": 853},
    "fra/reglements": {"points": 40_817, "urls": 3_067},
}


def binding(root: Path, path: Path) -> dict:
    path = path.resolve()
    return {"uri": "artifact://" + path.relative_to(root.resolve()).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _rows(root: Path, stage: dict):
    receipt = _validate_stage(stage, "justice_identity", artifact_root=root,
                              deep=True)
    for artifact in receipt["artifacts"]:
        path = root / str(artifact["uri"]).removeprefix("artifact://")
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                yield json.loads(line)


def materialize(*, artifact_root: Path, identity_run_path: Path,
                output: Path) -> dict:
    artifact_root = artifact_root.resolve()
    identity_run_path = identity_run_path.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError(f"refusing existing Justice materialization: {output}")
    identity = json.loads(identity_run_path.read_text())
    counts = identity.get("counts") or {}
    if (identity.get("schema") !=
            "aw.justice-canada-residual-current-identity-run.v1"
            or identity.get("status") != "complete-exact-current-source-identity"
            or identity.get("source_id") != SOURCE_ID
            or counts.get("current_points") != 114_293
            or counts.get("current_urls") != 6_956):
        raise ValueError("Justice residual identity is not terminal/exact")
    for source_class, expected in EXPECTED_CLASSES.items():
        observed = (identity.get("classes") or {}).get(source_class) or {}
        if any(observed.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Justice residual class drift: {source_class}")
    if set(identity.get("classes") or {}) != set(EXPECTED_CLASSES):
        raise ValueError("Justice residual identity has an unexpected source class")
    parents: dict[str, dict] = {}
    members: dict[str, dict] = {}
    for row in _rows(artifact_root, identity["stages"]["source_registry"]):
        url = str(row.get("current_url") or "")
        member = str(row.get("source_member") or "")
        descriptor = law_descriptor(url)
        if (row.get("schema") != "aw.justice-canada-residual-source-parent.v1"
                or row.get("source_id") != SOURCE_ID
                or descriptor["source_class"] != row.get("source_class")
                or descriptor["lang"] != row.get("lang")
                or not isinstance(row.get("current_points"), int)
                or row["current_points"] < 1
                or url in parents or member in members):
            raise ValueError("invalid/duplicate Justice residual source parent")
        parents[url] = row
        members[member] = row
    if (len(parents) != 6_956
            or sum(row["current_points"] for row in parents.values()) != 114_293
            or sum(row["current_document_evidence_keys"]
                   for row in parents.values()) != 109_990):
        raise ValueError("Justice residual parent registry conservation failed")

    source_archive = (identity.get("inputs") or {}).get("source_archive") or {}
    archive_path = artifact_root / str(source_archive.get("uri") or "").removeprefix(
        "artifact://")
    if (not archive_path.is_file()
            or archive_path.stat().st_size != source_archive.get("bytes")
            or sha256_file(archive_path) != source_archive.get("sha256")):
        raise ValueError("Justice residual archive binding failed")

    partial = output.with_name(output.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    markdown_path = partial / "markdown/parents.jsonl.gz"
    outlinks_path = partial / "outlinks/parent-links.jsonl.gz"
    markdown_path.parent.mkdir(parents=True)
    outlinks_path.parent.mkdir(parents=True)
    logical = {"markdown": hashlib.sha256(), "outlinks": hashlib.sha256()}
    records = {"markdown": 0, "outlinks": 0}
    processed: set[str] = set()
    selected_xml_bytes = 0
    legal_anchor_count = 0
    linked_parents = 0

    def write(name: str, row: dict, target) -> None:
        encoded = canonical_json(row) + "\n"
        target.write(encoded)
        logical[name].update(encoded.encode())
        records[name] += 1

    try:
        with deterministic_gzip_text(markdown_path) as markdown_out, \
                deterministic_gzip_text(outlinks_path) as outlinks_out, \
                tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                selected = members.get(member.name)
                if selected is None:
                    continue
                if not member.isfile() or member.name in processed:
                    raise ValueError("Justice selected XML member invalid/duplicate")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"unreadable Justice XML member: {member.name}")
                xml_bytes = stream.read()
                selected_xml_bytes += len(xml_bytes)
                xml_sha = hashlib.sha256(xml_bytes).hexdigest()
                root = ET.fromstring(xml_bytes.lstrip(b"\xef\xbb\xbf"))
                url = selected["current_url"]
                descriptor = law_descriptor(url)
                expected_root = "Statute" if descriptor["kind"] == "act" else "Regulation"
                root_lang = _attrs(root).get("lang")
                if _tag(root) != expected_root or root_lang != descriptor["lang"]:
                    raise ValueError("Justice XML root kind/language mismatch")
                title, markdown, source_version = render_official_markdown(
                    root, url=url, member=member.name, member_sha256=xml_sha)
                markdown_sha = hashlib.sha256(markdown.encode()).hexdigest()
                source_record_id = (f"{descriptor['source_class']}:"
                                    f"{descriptor['slug']}")
                anchors = legal_unit_anchors(root)
                links = source_links_with_anchors(
                    root, source_slug=source_record_id, lang=descriptor["lang"])
                identity_value = {"member": member.name, "sha256": xml_sha,
                                  "release": COMMIT}
                document_id = hashlib.sha256(
                    f"justice-canada-residual-law\0{url}".encode()).hexdigest()
                parent = {
                    "schema": "aw.justice-canada-residual-law-parent.v1",
                    "document_id": document_id,
                    "source_record_id": source_record_id,
                    "source_id": SOURCE_ID,
                    "source_profile": "anchored_law", "profile": "anchored_law",
                    "url": url, "canonical_url": url, "title": title,
                    "lang": descriptor["lang"], "markdown": markdown,
                    "markdown_sha256": markdown_sha,
                    "parent_source_sha256": xml_sha,
                    "source_record_sha256": xml_sha,
                    "parent_identity": identity_value,
                    "source_archive": identity_value,
                    "source_version": source_version,
                    "source_class": descriptor["source_class"],
                    "source_links": links, "source_links_explicit": True,
                    "source_relations_unresolved": [],
                    "source_relations_unresolved_explicit": True,
                    "legal_unit_anchors": anchors,
                    "legal_unit_anchors_explicit": True,
                    "anchor": f"{descriptor['kind']}:{descriptor['slug']}",
                    "current_production_points": selected["current_points"],
                    "current_document_evidence_keys":
                        selected["current_document_evidence_keys"],
                    "current_identity_run": binding(
                        artifact_root, identity_run_path),
                }
                write("markdown", parent, markdown_out)
                legal_anchor_count += len(anchors)
                if links:
                    linked_parents += 1
                output_links = links or [{
                    "relation": None, "target": None, "target_id": None,
                    "context": None, "direction": "outbound",
                    "article_anchor": f"document:{source_record_id}",
                }]
                for ordinal, link in enumerate(output_links):
                    write("outlinks", {
                        "schema": "aw.parent-source-link.v1",
                        "document_id": document_id,
                        "source_record_id": source_record_id,
                        "source_url": url, "ordinal": ordinal,
                        "target_url": link["target"],
                        "relation": link["relation"],
                        "target_id": link["target_id"],
                        "anchor_or_context": link["context"],
                        "article_anchor": link["article_anchor"],
                        "is_empty": not bool(link["target"]),
                        "parent_markdown_sha256": markdown_sha,
                        "parent_identity": identity_value,
                        "source_version": source_version,
                    }, outlinks_out)
                processed.add(member.name)
        if processed != set(members) or records["markdown"] != 6_956:
            raise ValueError("Justice residual XML member conservation failed")
        stages = {}
        for name, path, fmt in (
            ("markdown", markdown_path,
             "justice-canada-full-consolidated-law-markdown+canonical-jsonl+deterministic-gzip"),
            ("outlinks", outlinks_path,
             "justice-canada-parent-bound-anchored-legal-relations+canonical-jsonl+deterministic-gzip"),
        ):
            stages[name] = SS.seal(
                artifact_root=artifact_root, artifact_paths=[path],
                receipt_path=partial / name / "receipt.json",
                records=records[name], fmt=fmt,
                logical_sha256=logical[name].hexdigest(), producer_id=PRODUCER,
                producer_version=VERSION, publish_from=partial,
                publish_to=output)
        result = {
            "schema": "aw.justice-canada-residual-law-run.v1",
            "version": VERSION, "status": "complete",
            "source_id": SOURCE_ID, "source_profile": "anchored_law",
            "commit": COMMIT,
            "counts": {
                "current_points": 114_293,
                "current_document_evidence_keys": 109_990,
                "source_parents": records["markdown"],
                "outlink_rows_including_empty": records["outlinks"],
                "linked_parents": linked_parents,
                "legal_unit_anchors": legal_anchor_count,
                "selected_xml_bytes": selected_xml_bytes,
            },
            "classes": EXPECTED_CLASSES,
            "source_identity": binding(artifact_root, identity_run_path),
            "source_l0": {
                "run": identity["inputs"]["source_l0_run"],
                "original_receipt":
                    identity["inputs"]["source_l0_original_receipt"],
                "archive": source_archive,
            },
            "point_identity": {
                "status": "exact-current-pre-vector-identity",
                "prior_terminal_witness_overlap": 0,
                "ledger_subtraction_authorized": False,
            },
            "stages": stages,
        }
        atomic_publish(partial / "RUN.json", (
            json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        os.rename(partial, output)
        return result
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--identity-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(
        artifact_root=args.artifact_root,
        identity_run_path=args.identity_run,
        output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
