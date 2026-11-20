"""Materialize registry-approved generic HTML from terminal direct CC misses.

The direct lane runs only after exact signed-S3 misses.  This stage never
turns that transport fact into a source-family guess: reviewed URL routes
stay source-specific, and only fallback HTML is projected through the generic
web profile.  Complete non-HTML responses remain explicit residuals.

Moved from the pre-move driver by task 1432 (S7). The producer label `PRODUCER` is written into the
sealed stage receipts and stays byte for byte as it was. Reads sealed direct receipts and contacts no
host. The consent-gated receipt validators belong to the direct exact-URL acquirer; see `direct_validators`.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.parse import urlsplit

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire.deterministic_gzip import deterministic_gzip_text
from cytir.extract.html_extract import extract_outlink_context_records, reg_domain, _parse_soup
from cytir.acquire import preserved_stage_io as PS
from cytir.sources.routec import preserved_html as BASE
from cytir.extract.routec.common import (HTML_TYPES, content_type as _content_type,
                                      logical as _logical)
from cytir.acquire import stage_receipt as SS


VERSION = "routec-direct-preserved-html-materialization.v1"

#: The module that holds `validate_consent_gated_run` / `validate_consent_gated_receipt` (the direct
#: exact-URL acquirer). Set by the caller (the forwarding shim, a test) or resolved lazily by
#: `direct_validators`; the package never names a flat pipeline module. (`globals().get` keeps a value
#: the shim placed before it ran this file as a script.)
DIRECT = globals().get("DIRECT")


def direct_validators():
    """The consent-gated receipt validators of the direct exact-URL acquirer."""
    if DIRECT is not None:
        return DIRECT
    import importlib
    return importlib.import_module("cytir.sources.exact_url.direct_preserved")

PRODUCER = "pipeline/materialize_routec_direct_preserved_html.py"
def _frontier(path: Path, receipt_path: Path) -> tuple[dict[str, dict], dict]:
    receipt = json.loads(receipt_path.read_text())
    declared = (receipt.get("result") or {}).get("frontier") or {}
    if (receipt.get("schema") != "aw.exact-url-frontier-receipt.v1"
            or declared.get("sha256") != sha256_file(path)):
        raise ValueError("Route-C exact production frontier binding failed")
    rows = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line); url = str(row.get("url") or "")
            if not url or url in rows:
                raise ValueError("Route-C frontier contains an empty or duplicate URL")
            rows[url] = row
    if len(rows) != declared.get("urls"):
        raise ValueError("Route-C frontier URL count drift")
    return rows, receipt


def materialize(*, artifact_root: Path, frontier: Path,
                frontier_receipt: Path, direct_root: Path,
                route_registry_path: Path, output: Path) -> dict:
    artifact_root = artifact_root.resolve(); output = output.resolve()
    output.relative_to(artifact_root)
    if output.exists():
        raise ValueError(f"refusing existing Route-C direct materialization: {output}")
    frontier_rows, frontier_seal = _frontier(frontier, frontier_receipt)
    direct_run = direct_validators().validate_consent_gated_run(
        direct_root.resolve(), artifact_root, deep=True)
    registry = BASE._load_registry(route_registry_path.resolve())
    receipt_paths = sorted((direct_root / "receipts").glob("*.json"))
    if len(receipt_paths) != int(direct_run["counts"]["frontier"]):
        raise ValueError("Route-C direct receipt population drift")

    partial = output.with_name(output.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    registry_copy = partial / "config/route-registry.v1.json"
    registry_copy.parent.mkdir(parents=True)
    atomic_publish(registry_copy, route_registry_path.read_bytes())
    paths = {
        "routes": partial / "routes/all.jsonl.gz",
        "markdown": partial / "markdown/generic-web.jsonl.gz",
        "outlinks": partial / "outlinks/generic-web-parent-bound.jsonl.gz",
        "residual": partial / "residual/non-generic-or-unresolved.jsonl.gz",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    logical = {name: hashlib.sha256() for name in paths}
    counts = {name: 0 for name in paths}
    status_counts = {"complete": 0, "quarantine": 0}
    residual_reasons: dict[str, int] = {}
    route_counts: dict[str, dict[str, int]] = {}
    generic_originals: set[Path] = set()
    production_points = 0

    def write(name: str, value: dict, handle) -> None:
        encoded = canonical_json(value) + "\n"
        handle.write(encoded); logical[name].update(encoded.encode())
        counts[name] += 1

    try:
        with deterministic_gzip_text(paths["routes"]) as routes_out, \
                deterministic_gzip_text(paths["markdown"]) as markdown_out, \
                deterministic_gzip_text(paths["outlinks"]) as outlinks_out, \
                deterministic_gzip_text(paths["residual"]) as residual_out:
            for receipt_path in receipt_paths:
                receipt = json.loads(receipt_path.read_text())
                direct_validators().validate_consent_gated_receipt(receipt, artifact_root)
                url = str(receipt.get("source_url") or "")
                frontier_row = frontier_rows.get(url)
                if frontier_row is None:
                    raise ValueError(f"direct receipt is outside exact frontier: {url}")
                points = int(frontier_row["retained_points"])
                production_points += points
                route = BASE.classify_url(url, registry)
                state = receipt["status"]; status_counts[state] += 1
                by_route = route_counts.setdefault(route["route_id"], {
                    "urls": 0, "production_points": 0, "complete": 0,
                    "quarantine": 0, "generic_markdown": 0, "residual": 0})
                by_route["urls"] += 1; by_route["production_points"] += points
                by_route[state] += 1
                route_row = {
                    "schema": "aw.routec-direct-source-route.v1",
                    "url": url, "final_url": receipt.get("final_url"),
                    "wave": frontier_row.get("wave"), "production_points": points,
                    "route_id": route["route_id"],
                    "source_profile": route["source_profile"],
                    "projection": route["projection"],
                    "source_route": route["source_route"],
                    "classification_basis": "reviewed-exact-url-host-and-path-registry",
                    "direct_status": state,
                    "terminal_receipt": BASE._binding(receipt_path, artifact_root),
                    "original_artifacts": receipt.get("artifacts") or {},
                }
                write("routes", route_row, routes_out)
                reason = None
                if state == "quarantine":
                    reason = str(receipt.get("quarantine_reason") or "direct-quarantine")
                elif route["projection"] == "source-specific-pending":
                    reason = "source-specific-projection-pending"
                elif _content_type(artifact_root, receipt) not in HTML_TYPES:
                    reason = "generic-fallback-media-type-is-not-web-html"
                if reason:
                    by_route["residual"] += 1
                    residual_reasons[reason] = residual_reasons.get(reason, 0) + 1
                    write("residual", {**route_row, "residual_reason": reason}, residual_out)
                    continue

                artifacts = receipt.get("artifacts") or {}
                html_bytes = _logical(artifact_root, artifacts["response_body"])
                html = html_bytes.decode("utf-8", "replace")
                soup = _parse_soup(html)
                principal, extraction_mode, title, lang = BASE.project_html(html, url, soup)
                if not principal.strip():
                    reason = "principal-content-extraction-empty"
                    by_route["residual"] += 1
                    residual_reasons[reason] = residual_reasons.get(reason, 0) + 1
                    write("residual", {**route_row, "residual_reason": reason}, residual_out)
                    continue
                markdown = principal.strip() + "\n"
                document_id = hashlib.sha256(f"routec-direct-web-html\0{url}".encode()).hexdigest()
                parent = {
                    "schema": "aw.web-html-parent.v1",
                    "document_id": document_id, "source_record_id": url,
                    "source_id": "routec-phase2-all-direct-after-cc-miss",
                    "source_profile": "web_html", "url": url,
                    "final_url": receipt.get("final_url"), "title": title or None,
                    "lang": lang,
                    "markdown": markdown,
                    "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
                    "parent_source_sha256": artifacts["response_body"]["body_sha256"],
                    "parent_identity": {
                        "terminal_receipt": BASE._binding(receipt_path, artifact_root),
                        "response_body": artifacts["response_body"],
                        "response_headers": artifacts["response_headers"],
                        "transport": artifacts["transport"],
                        "consent": artifacts["consent"],
                    },
                    "extraction": {"producer": "aw_common.extract_content_markdown",
                                   "mode": extraction_mode,
                                   "full_prechunk_principal_content": True,
                                   "source_text_truncation_applied": False},
                    "baseline_production_points": points,
                }
                write("markdown", parent, markdown_out)
                by_route["generic_markdown"] += 1
                generic_originals.add(receipt_path)
                generic_originals.update(PS.artifact_path(artifact_root, artifact)
                                         for artifact in artifacts.values())
                links = extract_outlink_context_records(soup, url)
                source_host = (urlsplit(url).hostname or "").casefold()
                if not links:
                    links = [{"url": None, "nofollow": False,
                              "anchor_or_context": None}]
                for ordinal, link in enumerate(links):
                    target = link["url"]
                    target_host = (urlsplit(target).hostname or "").casefold() if target else ""
                    outlink = {
                        "schema": "aw.parent-source-link.v1",
                        "document_id": document_id, "source_record_id": url,
                        "source_url": url, "ordinal": ordinal,
                        "target_url": target, "nofollow": link["nofollow"],
                        "anchor_or_context": link["anchor_or_context"],
                        "relation_scope": (None if not target else
                            ("same-site" if reg_domain(source_host) ==
                             reg_domain(target_host) else "external")),
                        "is_empty": not bool(target),
                        "parent_markdown_sha256": parent["markdown_sha256"],
                    }
                    write("outlinks", outlink, outlinks_out)

        if (status_counts["complete"] != int(direct_run["counts"]["complete"])
                or status_counts["quarantine"] != int(direct_run["counts"]["quarantine"])
                or counts["routes"] != int(direct_run["counts"]["frontier"])
                or counts["markdown"] + counts["residual"] != counts["routes"]):
            raise ValueError("Route-C direct materialization conservation failed")
        if not generic_originals or counts["markdown"] <= 0:
            raise ValueError("Route-C direct materialization has no generic HTML parents")
        original_logical = hashlib.sha256(b"".join(
            (canonical_json(BASE._binding(path, artifact_root)) + "\n").encode()
            for path in sorted(generic_originals)))
        stages = {
            "original": SS.seal(
                artifact_root=artifact_root, artifact_paths=sorted(generic_originals),
                receipt_path=partial / "original/receipt.json",
                records=counts["markdown"],
                fmt="exact-direct-body+headers+transport+consent+terminal-receipts",
                logical_sha256=original_logical.hexdigest(), producer_id=PRODUCER,
                producer_version=VERSION, publish_from=partial, publish_to=output,
                allow_existing_published=True),
            "config": SS.seal(
                artifact_root=artifact_root, artifact_paths=[registry_copy],
                receipt_path=partial / "config/receipt.json", records=1,
                fmt="reviewed-exact-url-source-route-registry+canonical-json",
                logical_sha256=sha256_file(registry_copy), producer_id=PRODUCER,
                producer_version=VERSION, publish_from=partial, publish_to=output),
        }
        for name, fmt in (
            ("routes", "direct-source-routes+canonical-jsonl+deterministic-gzip"),
            ("markdown", "full-principal-content-markdown+canonical-jsonl+deterministic-gzip"),
            ("outlinks", "parent-bound-context-outlinks+canonical-jsonl+deterministic-gzip"),
            ("residual", "explicit-non-generic-or-unresolved-routes+canonical-jsonl+deterministic-gzip"),
        ):
            stages[name] = SS.seal(
                artifact_root=artifact_root, artifact_paths=[paths[name]],
                receipt_path=partial / name / "receipt.json", records=counts[name],
                fmt=fmt, logical_sha256=logical[name].hexdigest(),
                producer_id=PRODUCER, producer_version=VERSION,
                publish_from=partial, publish_to=output)
        result = {
            "schema": "aw.routec-direct-materialization-run.v1",
            "version": VERSION, "status": "complete",
            "source_profile": "web_html",
            "admitted_source_profile": "web_html",
            "frontier_receipt": BASE._binding(frontier_receipt, artifact_root),
            "direct_run": BASE._binding(direct_root / "RUN.json", artifact_root),
            "route_registry": BASE._published_binding(
                registry_copy, partial, output, artifact_root),
            "counts": {"frontier_urls": counts["routes"],
                       "production_points": production_points,
                       "direct_complete": status_counts["complete"],
                       "direct_quarantine": status_counts["quarantine"],
                       "generic_markdown_parents": counts["markdown"],
                       "outlink_records_including_empty": counts["outlinks"],
                       "residual_urls": counts["residual"]},
            "residual_reasons": dict(sorted(residual_reasons.items())),
            "route_counts": dict(sorted(route_counts.items())),
            "stages": stages,
        }
        (partial / "RUN.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.rename(partial, output)
        return result
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
