#!/usr/bin/env python3
"""Preserve and materialize the official DILA Service-Public XML dump.

Moved from the flat pipeline script (task 1403, S7); the old path is a
forwarding shim. `PRODUCER_ID` keeps the OLD path byte for byte: it is written into sealed stage
receipts, and a renamed id would seal the same dump to different bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.parse
import zipfile

from lxml import etree

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS


VERSION = "service-public-dila-preserved.v1"
PRODUCER_ID = "pipeline/acquire_service_public_dump_preserved.py"
DC = "{http://purl.org/dc/elements/1.1/}"
CONTENT = {"Introduction", "Texte", "ListeSituations", "CommentFaireSi",
           "Questionnaire", "Problematiques", "Avertissement"}
PROSE = ("Titre", "Paragraphe")
URL_RE = re.compile(r"https?://[^\s<>\"']+")


class ServicePublicError(RuntimeError):
    pass


def _logical(rows: list[dict]) -> tuple[bytes, str]:
    body = b"".join((canonical_json(row) + "\n").encode() for row in rows)
    return body, hashlib.sha256(body).hexdigest()


def _transport(path: Path, body: Path, headers: Path, expected_url: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("schema") != "aw.http-fetch-metadata.v1" or
            value.get("requested_url") != expected_url or
            value.get("http_status") != 200 or not value.get("final_url") or
            not value.get("fetched_at")):
        raise ServicePublicError("invalid dump HTTP transport receipt")
    for key, artifact in (("body", body), ("response_headers", headers)):
        expected = value.get(key) or {}
        if (expected.get("bytes") != artifact.stat().st_size or
                expected.get("sha256") != sha256_file(artifact)):
            raise ServicePublicError(f"{key} disagrees with transport receipt")
    return value


def _text(element) -> str:
    return " ".join((element.xpath("string()") or "").split())


def extract_member(member: str, xml_bytes: bytes, archive_sha: str) -> tuple[dict | None, list[dict]]:
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError:
        return None, []
    if etree.QName(root).localname != "Publication":
        return None, []
    source_url = str(root.get("spUrl") or "").strip()
    title = str(root.findtext(DC + "title") or "").strip()
    if not source_url.startswith("http") or not title:
        return None, []
    blocks: list[str] = []
    for child in root:
        if etree.QName(child).localname not in CONTENT:
            continue
        leaves = list(child.iter(*PROSE))
        candidates = leaves or [child]
        for value in candidates:
            text = _text(value)
            if text and (not blocks or blocks[-1] != text):
                blocks.append(text)
    if not blocks:
        return None, []
    markdown = f"# {title}\n\n" + "\n\n".join(blocks) + "\n"
    doc_id = hashlib.sha256(f"service-public:{source_url}".encode()).hexdigest()
    row = {
        "document_id": doc_id, "url": source_url, "lang": "fr",
        "title": title, "markdown": markdown, "archive_sha256": archive_sha,
        "archive_member": member,
        "archive_member_sha256": hashlib.sha256(xml_bytes).hexdigest(),
    }
    source_host = (urllib.parse.urlsplit(source_url).hostname or "").lower()
    outlinks = []
    seen = set()
    for element in root.iter():
        context = etree.QName(element).localname
        candidates = list(element.attrib.values())
        if element.text:
            candidates.append(element.text)
        for candidate in candidates:
            for match in URL_RE.findall(str(candidate)):
                target = match.rstrip(".,;:)]}")
                host = (urllib.parse.urlsplit(target).hostname or "").lower()
                if not host or host == source_host or host.endswith(".service-public.gouv.fr"):
                    continue
                key = (target, context)
                if key in seen:
                    continue
                seen.add(key)
                outlinks.append({"document_id": doc_id, "source_url": source_url,
                                 "target_url": target,
                                 "anchor_or_context": context})
    return row, outlinks


def materialize(*, archive: Path, headers: Path, metadata: Path, root: Path,
                requested_url: str, fetch_vantage: str) -> dict:
    transport = _transport(metadata, archive, headers, requested_url)
    root.mkdir(parents=True, exist_ok=True)
    archive_sha = sha256_file(archive)
    original_artifacts = []
    for family, suffix, path in (("body", "zip", archive),
                                 ("headers", "headers", headers),
                                 ("metadata", "json", metadata)):
        artifact = PS.store_artifact(root, f"stages/original/data/{family}",
                                     suffix, path.read_bytes())
        original_artifacts.append(PS.artifact_path(root, artifact))
    original_rows = [{
        "requested_url": requested_url, "final_url": transport["final_url"],
        "http_status": 200, "fetched_at": transport["fetched_at"],
        "body_bytes": archive.stat().st_size, "body_sha256": archive_sha,
        "response_headers_bytes": headers.stat().st_size,
        "response_headers_sha256": sha256_file(headers),
        "transport_metadata_sha256": sha256_file(metadata),
    }]
    _, original_sha = _logical(original_rows)
    original_stage = SS.seal(
        artifact_root=root, artifact_paths=original_artifacts,
        receipt_path=root / "stages/original/receipt.json", records=1,
        fmt="exact-zip-response+headers+canonical-transport-metadata+deterministic-gzip",
        logical_sha256=original_sha,
        producer_id=PRODUCER_ID,
        producer_version=VERSION)

    markdown_rows: list[dict] = []
    outlink_rows: list[dict] = []
    quarantine_rows: list[dict] = []
    seen_urls: set[str] = set()
    with zipfile.ZipFile(archive) as source:
        members = sorted(source.infolist(), key=lambda item: item.filename)
        if len(members) > 100_000 or sum(item.file_size for item in members) > 4_000_000_000:
            raise ServicePublicError("dump exceeds bounded archive limits")
        for item in members:
            safe = PurePosixPath(item.filename)
            if safe.is_absolute() or ".." in safe.parts:
                raise ServicePublicError(f"unsafe archive member: {item.filename}")
            if item.is_dir() or not safe.name.startswith("F") or not safe.name.endswith(".xml"):
                continue
            xml_bytes = source.read(item)
            row, links = extract_member(item.filename, xml_bytes, archive_sha)
            if row is None:
                quarantine_rows.append({"archive_member": item.filename,
                                        "reason": "no_admissible_practical_fiche_markdown",
                                        "member_sha256": hashlib.sha256(xml_bytes).hexdigest()})
                continue
            if row["url"] in seen_urls:
                quarantine_rows.append({"archive_member": item.filename,
                                        "reason": "duplicate_canonical_url",
                                        "url": row["url"],
                                        "member_sha256": row["archive_member_sha256"]})
                continue
            seen_urls.add(row["url"])
            markdown_rows.append(row)
            outlink_rows.extend(links)
    markdown_rows.sort(key=lambda row: row["url"])
    outlink_rows.sort(key=lambda row: (row["source_url"], row["target_url"],
                                      row["anchor_or_context"]))
    quarantine_rows.sort(key=lambda row: row["archive_member"])

    stages = {"original": original_stage}
    for name, rows in (("markdown", markdown_rows), ("outlinks", outlink_rows),
                       ("quarantine", quarantine_rows)):
        data, logical_sha = _logical(rows)
        artifact = PS.store_artifact(root, f"stages/{name}/data", "jsonl", data)
        stages[name] = SS.seal(
            artifact_root=root,
            artifact_paths=[PS.artifact_path(root, artifact)],
            receipt_path=root / f"stages/{name}/receipt.json",
            records=len(rows), fmt="canonical-jsonl+deterministic-gzip",
            logical_sha256=logical_sha,
            producer_id=PRODUCER_ID,
            producer_version=VERSION)
    run = {
        "schema": "aw.service-public-dila-acquisition-run.v1",
        "version": VERSION, "status": "complete",
        "source": "Service-Public.gouv.fr / DILA practical fiches",
        "country": "FR", "route": "source_owned_bulk_dump",
        "fetch_vantage": fetch_vantage,
        "license": "Licence Ouverte 2.0",
        "attribution": {"producer": "Service-Public.gouv.fr / DILA",
                        "download_url": requested_url,
                        "download_filename": Path(urllib.parse.urlsplit(
                            transport["final_url"]).path).name,
                        "download_date": transport["fetched_at"][:10]},
        "counts": {"markdown_documents": len(markdown_rows),
                   "outlink_records": len(outlink_rows),
                   "quarantined_members": len(quarantine_rows)},
        "stages": stages,
        "outlink_cadence": ["after_completed_wave", "weekly"],
        "downstream": {"chunks": "pending_generic_structured_profile",
                       "vectors": "pending_1024d_fp16"},
    }
    body = (json.dumps(run, indent=2, sort_keys=True) + "\n").encode()
    run_path = root / "RUN.json"
    if run_path.exists():
        if sha256_file(run_path) != hashlib.sha256(body).hexdigest():
            raise ServicePublicError("immutable RUN collision")
    else:
        atomic_publish(run_path, body)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--requested-url", required=True)
    parser.add_argument("--fetch-vantage", required=True)
    args = parser.parse_args()
    result = materialize(
        archive=args.archive.resolve(), headers=args.headers.resolve(),
        metadata=args.metadata.resolve(), root=args.root.resolve(),
        requested_url=args.requested_url, fetch_vantage=args.fetch_vantage)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
