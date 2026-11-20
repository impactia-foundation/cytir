#!/usr/bin/env python3
"""Materialize exact W3C /TR/ reports from preserved Route-C responses."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire.deterministic_gzip import deterministic_gzip_text
from cytir.extract.html_extract import extract_content_markdown, extract_outlink_context_records, _parse_soup
from cytir.extract.routec import common as ROUTEC
from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS


VERSION = "routec-w3c-technical-report-materialization.v1"
PRODUCER = "aw.extract.routec.w3c_technical_reports"
ROUTE_ID = "w3c-standards-html"
SOURCE_ID = "routec-w3c-technical-reports-current"
VERSION_PATH = re.compile(
    r"^/TR/(?:[0-9]{4}/)?(?:(REC|PR|CR|WD|NOTE)-)?[^/]+?(?:-([0-9]{6,8}))?(?:\.html)?/?$",
    re.I)


def is_technical_report_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.hostname == "www.w3.org" and parsed.path.lower().startswith("/tr/")


def report_path_identity(url: str, final_url: str | None, soup) -> dict:
    requested = urlsplit(url).path
    status_hint = date_hint = None
    match = VERSION_PATH.fullmatch(requested)
    if match:
        status_hint = match.group(1).upper() if match.group(1) else None
        date_hint = match.group(2)
    canonical = None
    if soup is not None:
        node = soup.find("link", rel=lambda value: value and
                         "canonical" in ([value.casefold()] if isinstance(value, str)
                                         else [str(v).casefold() for v in value]))
        href = str(node.get("href") or "").strip() if node is not None else ""
        if href:
            canonical = urljoin(final_url or url, href)
    return {
        "requested_path": requested,
        "final_url": final_url,
        "declared_canonical_url": canonical,
        "status_hint_from_path": status_hint,
        "version_date_hint_from_path": date_hint,
        "identity_basis": "exact-requested-TR-path+preserved-response",
    }


def _binding(path: Path, artifact_root: Path) -> dict:
    return ROUTEC.binding(path.resolve(), artifact_root.resolve())


def _validate_binding(artifact_root: Path, artifact: dict) -> Path:
    """Validate both plain receipt bindings and compressed body artifacts."""
    path = PS.artifact_path(artifact_root, artifact)
    if artifact.get("compression") == "gzip":
        PS.validate_artifact(artifact_root, artifact)
    elif (not path.is_file() or path.stat().st_size != artifact.get("bytes")
          or sha256_file(path) != artifact.get("sha256")):
        raise ValueError(f"plain artifact binding failed: {path}")
    return path


def materialize(*, artifact_root: Path, routes_path: Path,
                output_root: Path) -> dict:
    artifact_root = artifact_root.resolve()
    routes_path = routes_path.resolve()
    output_root = output_root.resolve()
    output_root.relative_to(artifact_root)
    if output_root.exists():
        raise ValueError(f"refusing existing W3C materialization: {output_root}")
    partial = output_root.with_name(output_root.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    paths = {
        "markdown": partial / "markdown/w3c-technical-reports.jsonl.gz",
        "outlinks": partial / "outlinks/w3c-technical-report-links.jsonl.gz",
        "residual": partial / "residual/non-technical-report-route-rows.jsonl.gz",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    logical = {name: hashlib.sha256() for name in paths}
    counts = {name: 0 for name in paths}
    route_urls = route_points = admitted_points = residual_points = 0
    admitted_originals: set[Path] = set()

    def write(name: str, row: dict, handle) -> None:
        encoded = canonical_json(row) + "\n"
        handle.write(encoded)
        logical[name].update(encoded.encode())
        counts[name] += 1

    try:
        with deterministic_gzip_text(paths["markdown"]) as markdown_out, \
                deterministic_gzip_text(paths["outlinks"]) as outlinks_out, \
                deterministic_gzip_text(paths["residual"]) as residual_out, \
                gzip.open(routes_path, "rt", encoding="utf-8") as source:
            seen_urls: set[str] = set()
            for line_number, line in enumerate(source, 1):
                route = json.loads(line)
                if route.get("route_id") != ROUTE_ID:
                    continue
                url = str(route.get("url") or "")
                points = route.get("production_points")
                if not url or url in seen_urls or not isinstance(points, int) or points < 0:
                    raise ValueError(f"invalid W3C route row at line {line_number}")
                seen_urls.add(url); route_urls += 1; route_points += points
                if route.get("direct_status") != "complete" or not is_technical_report_url(url):
                    reason = ("terminal_source_response_quarantined"
                              if route.get("direct_status") != "complete"
                              else "non_TR_W3C_object_requires_separate_projection")
                    write("residual", {
                        "schema": "aw.routec-w3c-residual.v1", "url": url,
                        "reported_production_points": points,
                        "reason": reason,
                        "terminal_receipt": route.get("terminal_receipt"),
                    }, residual_out)
                    residual_points += points
                    continue
                artifacts = route.get("original_artifacts") or {}
                required = {"response_body", "response_headers", "transport", "consent"}
                if set(artifacts) != required:
                    raise ValueError(f"W3C source artifact set drift: {url}")
                terminal = route.get("terminal_receipt")
                terminal_path = _validate_binding(artifact_root, terminal)
                artifact_paths = []
                for artifact in artifacts.values():
                    artifact_paths.append(_validate_binding(artifact_root, artifact))
                media_type = ROUTEC.content_type(
                    artifact_root, {"artifacts": artifacts})
                if media_type not in ROUTEC.HTML_TYPES:
                    write("residual", {
                        "schema": "aw.routec-w3c-residual.v1", "url": url,
                        "reported_production_points": points,
                        "reason": "non_HTML_TR_response_requires_media_specific_projection",
                        "media_type": media_type,
                        "terminal_receipt": terminal,
                    }, residual_out)
                    residual_points += points
                    continue
                admitted_originals.add(terminal_path)
                admitted_originals.update(artifact_paths)
                html_bytes = ROUTEC.logical(
                    artifact_root, artifacts["response_body"])
                html = html_bytes.decode("utf-8", "replace")
                soup = _parse_soup(html)
                principal, extraction_mode = extract_content_markdown(html, url)
                if not principal.strip():
                    raise ValueError(f"empty W3C /TR/ principal content: {url}")
                markdown = principal.strip() + "\n"
                markdown_sha = hashlib.sha256(markdown.encode()).hexdigest()
                document_id = hashlib.sha256(
                    f"w3c-technical-report\0{url}".encode()).hexdigest()
                title = " ".join((soup.title.get_text(" ", strip=True)
                                  if soup is not None and soup.title else "").split())
                parent_identity = {
                    "terminal_receipt": terminal,
                    "response_body": artifacts["response_body"],
                    "response_headers": artifacts["response_headers"],
                    "transport": artifacts["transport"],
                    "consent": artifacts["consent"],
                }
                parent = {
                    "schema": "aw.w3c-technical-report-parent.v1",
                    "document_id": document_id,
                    "source_record_id": url,
                    "source_id": SOURCE_ID,
                    "source_profile": "software_docs",
                    "url": url,
                    "title": title or None,
                    "lang": ((soup.html.get("lang") if soup is not None and soup.html
                              else None) or None),
                    "markdown": markdown,
                    "markdown_sha256": markdown_sha,
                    "parent_source_sha256": artifacts["response_body"]["body_sha256"],
                    "parent_identity": parent_identity,
                    "source_version": report_path_identity(
                        url, route.get("final_url"), soup),
                    "extraction": {
                        "producer": "aw_common.extract_content_markdown",
                        "mode": extraction_mode,
                        "full_prechunk_principal_content": True,
                        "source_text_truncation_applied": False,
                    },
                    "reported_production_points": points,
                }
                write("markdown", parent, markdown_out)
                admitted_points += points
                links = extract_outlink_context_records(soup, url)
                if not links:
                    links = [{"url": None, "nofollow": False,
                              "anchor_or_context": None}]
                for ordinal, link in enumerate(links):
                    write("outlinks", {
                        "schema": "aw.parent-source-link.v1",
                        "document_id": document_id,
                        "source_record_id": url,
                        "source_url": url,
                        "ordinal": ordinal,
                        "target_url": link["url"],
                        "nofollow": link["nofollow"],
                        "anchor_or_context": link["anchor_or_context"],
                        "is_empty": not bool(link["url"]),
                        "parent_markdown_sha256": markdown_sha,
                        "parent_identity": parent_identity,
                    }, outlinks_out)
        if route_urls != counts["markdown"] + counts["residual"] or \
                route_points != admitted_points + residual_points or \
                not counts["markdown"] or not admitted_originals:
            raise ValueError("W3C route/materialization conservation failed")
        original_logical = hashlib.sha256(b"".join(
            (canonical_json(_binding(path, artifact_root)) + "\n").encode()
            for path in sorted(admitted_originals)))
        stages = {
            "original": SS.seal(
                artifact_root=artifact_root,
                artifact_paths=sorted(admitted_originals),
                receipt_path=partial / "original/receipt.json",
                records=counts["markdown"],
                fmt="exact-direct-body+headers+transport+consent+terminal-receipts",
                logical_sha256=original_logical.hexdigest(), producer_id=PRODUCER,
                producer_version=VERSION, publish_from=partial,
                publish_to=output_root, allow_existing_published=True),
        }
        for name, fmt in (
            ("markdown", "w3c-TR-full-principal-markdown+canonical-jsonl+deterministic-gzip"),
            ("outlinks", "w3c-TR-parent-bound-context-outlinks+canonical-jsonl+deterministic-gzip"),
            ("residual", "explicit-non-TR-or-quarantined-W3C-routes+canonical-jsonl+deterministic-gzip"),
        ):
            stages[name] = SS.seal(
                artifact_root=artifact_root, artifact_paths=[paths[name]],
                receipt_path=partial / name / "receipt.json", records=counts[name],
                fmt=fmt, logical_sha256=logical[name].hexdigest(),
                producer_id=PRODUCER, producer_version=VERSION,
                publish_from=partial, publish_to=output_root)
        result = {
            "schema": "aw.routec-w3c-technical-report-run.v1",
            "version": VERSION, "status": "complete",
            "source_id": SOURCE_ID, "source_profile": "software_docs",
            "route_rows": _binding(routes_path, artifact_root),
            "point_identity": {
                "status": "not_present_in_route_registry",
                "reported_counts_are_not_a_point_id_partition": True,
                "ledger_subtraction_authorized": False,
            },
            "counts": {
                "route_urls": route_urls,
                "route_reported_production_points": route_points,
                "technical_report_parents": counts["markdown"],
                "technical_report_reported_production_points": admitted_points,
                "outlink_rows_including_empty": counts["outlinks"],
                "residual_urls": counts["residual"],
                "residual_reported_production_points": residual_points,
            },
            "stages": stages,
        }
        atomic_publish(partial / "RUN.json", (
            json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        os.rename(partial, output_root)
        return result
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
