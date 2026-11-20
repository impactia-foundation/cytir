"""Classify a sealed Route-C frontier and materialize only generic HTML.

Source-family routing is based exclusively on the reviewed URL route registry.
Known structured, legal, and software-document families remain queued for
their source-specific projections; the fallback is the only path allowed to
emit generic principal-content Markdown.

Moved from the pre-move driver by task 1432 (S7). The producer label `PRODUCER` is written into the
sealed stage receipts and stays byte for byte as it was: changing it would change receipt bytes.
Reads sealed Common Crawl terminal receipts and contacts no host.
"""
from __future__ import annotations
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.parse import urlsplit

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire.deterministic_gzip import deterministic_gzip_text
from cytir.extract.html_extract import extract_content_markdown, extract_outlink_context_records, reg_domain, _parse_soup
from cytir.acquire import preserved_stage_io as PS
from cytir.extract.routec.common import binding as _binding
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as SS


VERSION = "routec-preserved-html-materialization.v1"
PRODUCER = "pipeline/materialize_routec_preserved_html.py"


def _published_binding(staged_path: Path, partial: Path, output: Path,
                       artifact_root: Path) -> dict:
    relative = staged_path.resolve().relative_to(partial.resolve())
    published = output.resolve() / relative
    return {"uri": "artifact://" + published.relative_to(
                artifact_root.resolve()).as_posix(),
            "bytes": staged_path.stat().st_size,
            "sha256": sha256_file(staged_path)}


def _load_registry(path: Path) -> dict:
    registry = json.loads(path.read_text())
    if (registry.get("schema") != "aw.routec-source-route-registry.v1"
            or not isinstance(registry.get("routes"), list)
            or not isinstance(registry.get("fallback"), dict)):
        raise ValueError("invalid Route-C source route registry")
    seen_ids = set()
    for route in [*registry["routes"], registry["fallback"]]:
        if (not route.get("route_id") or route["route_id"] in seen_ids
                or route.get("source_profile") not in {
                    "web_html", "software_docs", "structured_data_feed",
                    "anchored_law"}
                or route.get("projection") not in {
                    "source-specific-pending", "generic-web-materialize"}):
            raise ValueError("invalid or duplicate Route-C route")
        seen_ids.add(route["route_id"])
    if registry["fallback"].get("projection") != "generic-web-materialize":
        raise ValueError("Route-C fallback must be the generic materializer")
    return registry


def classify_url(url: str, registry: dict) -> dict:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    path = parsed.path or "/"
    matches = []
    for route in registry["routes"]:
        hosts = {str(value).casefold() for value in route.get("hosts") or []}
        prefixes = tuple(route.get("path_prefixes") or ["/"])
        if host in hosts and any(path.startswith(prefix) for prefix in prefixes):
            matches.append(route)
    if len(matches) > 1:
        raise ValueError(f"ambiguous Route-C source route: {url}")
    return matches[0] if matches else registry["fallback"]


def project_html(html: str, url: str, soup) -> tuple[str, str, str, str | None]:
    """The generic-web projection of one HTML body: (principal markdown, extraction mode, title,
    lang). The one place the materializers and the family adapter share."""
    principal, mode = extract_content_markdown(html, url)
    title = " ".join((soup.title.get_text(" ", strip=True)
                      if soup is not None and soup.title else "").split())
    lang = (soup.html.get("lang") if soup is not None and soup.html else None) or None
    return principal, mode, title, lang


def _receipt_path(root: Path, url: str) -> Path:
    return root / "receipts" / (hashlib.sha256(url.encode()).hexdigest() + ".json")


def _logical_artifact(artifact_root: Path, artifact: dict) -> bytes:
    PS.validate_artifact(artifact_root, artifact)
    return gzip.decompress(PS.artifact_path(artifact_root, artifact).read_bytes())


def _stage(*, artifact_root: Path, partial: Path, output: Path, name: str,
           data_path: Path, records: int, logical: hashlib._Hash,
           fmt: str) -> dict:
    return SS.seal(
        artifact_root=artifact_root, artifact_paths=[data_path],
        receipt_path=partial / name / "receipt.json", records=records,
        fmt=fmt, logical_sha256=logical.hexdigest(), producer_id=PRODUCER,
        producer_version=VERSION, publish_from=partial, publish_to=output)


def materialize(*, artifact_root: Path, frontier: Path,
                frontier_receipt: Path, cc_root: Path, cc_run_path: Path,
                route_registry_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(f"refusing existing Route-C materialization: {output}")
    frontier_seal = json.loads(frontier_receipt.read_text())
    declared = (frontier_seal.get("result") or {}).get("frontier") or {}
    if (frontier_seal.get("schema") != "aw.exact-url-frontier-receipt.v1"
            or declared.get("sha256") != sha256_file(frontier)):
        raise ValueError("Route-C frontier is not sealed")
    cc_run = json.loads(cc_run_path.read_text())
    if (cc_run.get("schema") != "aw.exact-url-cc-preserved-run.v1"
            or cc_run.get("status") != "complete"):
        raise ValueError("terminal signed-S3 Route-C RUN is required")
    original_stage = cc_run.get("stages", {}).get("original")
    _validate_stage(original_stage, "routec.original",
                    artifact_root=artifact_root, deep=True)
    plan_path = cc_root / "plan/captures.jsonl"
    plan_sha = sha256_file(plan_path)
    registry = _load_registry(route_registry_path)
    partial = output.with_name(output.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    registry_copy = partial / "config/route-registry.v1.json"
    registry_copy.parent.mkdir(parents=True)
    atomic_publish(registry_copy, route_registry_path.read_bytes())
    paths = {
        "routes": partial / "routes/all.jsonl.gz",
        "capture_outlinks": partial / "outlinks/capture-context.jsonl.gz",
        "markdown": partial / "markdown/generic-web.jsonl.gz",
        "outlinks": partial / "outlinks/generic-web-parent-bound.jsonl.gz",
        "residual": partial / "residual/cc-misses.jsonl.gz",
        "quarantine": partial / "quarantine/generic-web.jsonl.gz",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    logical = {key: hashlib.sha256() for key in paths}
    counts = {key: 0 for key in paths}
    route_counts: dict[str, dict[str, int]] = {}
    generic_original_paths: set[Path] = set()
    frontier_urls = frontier_points = captures = misses = 0
    try:
        with frontier.open(encoding="utf-8") as source, \
                deterministic_gzip_text(paths["routes"]) as routes_out, \
                deterministic_gzip_text(paths["capture_outlinks"]) as capture_links_out, \
                deterministic_gzip_text(paths["markdown"]) as markdown_out, \
                deterministic_gzip_text(paths["outlinks"]) as outlinks_out, \
                deterministic_gzip_text(paths["residual"]) as residual_out, \
                deterministic_gzip_text(paths["quarantine"]) as quarantine_out:
            for line in source:
                if not line.strip():
                    continue
                frontier_row = json.loads(line)
                url = frontier_row["url"]
                points = int(frontier_row["retained_points"])
                route = classify_url(url, registry)
                receipt_path = _receipt_path(cc_root, url)
                receipt = json.loads(receipt_path.read_text())
                if (receipt.get("schema") != "aw.exact-url-cc-capture-receipt.v1"
                        or receipt.get("source_url") != url
                        or receipt.get("plan_sha256") != plan_sha
                        or receipt.get("status") not in {"complete", "cc_miss"}):
                    raise ValueError(f"invalid Route-C terminal receipt: {url}")
                for artifact in (receipt.get("artifacts") or {}).values():
                    PS.validate_artifact(artifact_root, artifact)
                state = receipt["status"]
                route_row = {
                    "schema": "aw.routec-source-route.v1",
                    "url": url, "wave": frontier_row.get("wave"),
                    "production_points": points,
                    "route_id": route["route_id"],
                    "source_profile": route["source_profile"],
                    "projection": route["projection"],
                    "source_route": route["source_route"],
                    "classification_basis": "reviewed-exact-url-host-and-path-registry",
                    "cc_status": state,
                    "terminal_receipt": _binding(receipt_path, artifact_root),
                    "original_artifacts": receipt.get("artifacts") or {},
                }
                encoded = canonical_json(route_row) + "\n"
                routes_out.write(encoded); logical["routes"].update(encoded.encode())
                counts["routes"] += 1
                by_route = route_counts.setdefault(route["route_id"], {
                    "urls": 0, "production_points": 0,
                    "cc_captures": 0, "cc_misses": 0,
                    "generic_markdown": 0, "generic_quarantine": 0})
                by_route["urls"] += 1; by_route["production_points"] += points
                frontier_urls += 1; frontier_points += points
                if state == "cc_miss":
                    misses += 1; by_route["cc_misses"] += 1
                    residual = {**route_row,
                        "next_step": ("source-specific-feed-or-official-source-recovery"
                                      if route["projection"] == "source-specific-pending"
                                      else "direct-retrieval-after-common-crawl-miss")}
                    encoded = canonical_json(residual) + "\n"
                    residual_out.write(encoded); logical["residual"].update(encoded.encode())
                    counts["residual"] += 1
                    continue
                captures += 1; by_route["cc_captures"] += 1
                html_bytes = _logical_artifact(
                    artifact_root, receipt["artifacts"]["decoded_html"])
                html = html_bytes.decode("utf-8", "replace")
                soup = _parse_soup(html)
                context_links = extract_outlink_context_records(soup, url)
                source_host = (urlsplit(url).hostname or "").casefold()
                raw_rows = context_links or [{"url": None, "nofollow": False,
                                              "anchor_or_context": None}]
                for ordinal, link in enumerate(raw_rows):
                    target_host = (urlsplit(link["url"]).hostname or "").casefold()
                    capture_link = {
                        "schema": "aw.routec-capture-outlink.v1",
                        "source_url": url, "ordinal": ordinal,
                        "route_id": route["route_id"],
                        "source_profile": route["source_profile"],
                        "target_url": link["url"],
                        "nofollow": link["nofollow"],
                        "anchor_or_context": link["anchor_or_context"],
                        "relation_scope": (None if not link["url"] else
                            ("same-site" if reg_domain(source_host) ==
                             reg_domain(target_host) else "external")),
                        "is_empty": not bool(link["url"]),
                        "parent_original_sha256": receipt["artifacts"][
                            "decoded_html"]["body_sha256"],
                        "terminal_receipt": _binding(receipt_path, artifact_root),
                    }
                    encoded = canonical_json(capture_link) + "\n"
                    capture_links_out.write(encoded)
                    logical["capture_outlinks"].update(encoded.encode())
                    counts["capture_outlinks"] += 1
                if route["projection"] == "source-specific-pending":
                    continue
                principal, extraction_mode, title, lang = project_html(html, url, soup)
                if not principal.strip():
                    quarantine = {**route_row,
                        "reason": "principal-content-extraction-empty"}
                    encoded = canonical_json(quarantine) + "\n"
                    quarantine_out.write(encoded)
                    logical["quarantine"].update(encoded.encode())
                    counts["quarantine"] += 1
                    by_route["generic_quarantine"] += 1
                    continue
                markdown = principal.strip() + "\n"
                document_id = hashlib.sha256(
                    f"routec-web-html\0{url}".encode()).hexdigest()
                parent = {
                    "schema": "aw.web-html-parent.v1",
                    "document_id": document_id,
                    "source_record_id": url,
                    "source_id": "routec-phase2-all-common-crawl",
                    "source_profile": "web_html", "url": url,
                    "title": title or None,
                    "lang": lang,
                    "markdown": markdown,
                    "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
                    "parent_source_sha256": receipt["artifacts"][
                        "decoded_html"]["body_sha256"],
                    "parent_identity": {
                        "terminal_receipt": _binding(receipt_path, artifact_root),
                        "decoded_html": receipt["artifacts"]["decoded_html"],
                        "raw_warc": receipt["artifacts"]["raw_warc"],
                        "transport": receipt["artifacts"]["transport"],
                    },
                    "extraction": {
                        "producer": "aw_common.extract_content_markdown",
                        "mode": extraction_mode,
                        "full_prechunk_principal_content": True,
                        "source_text_truncation_applied": False,
                    },
                    "baseline_production_points": points,
                }
                encoded = canonical_json(parent) + "\n"
                markdown_out.write(encoded); logical["markdown"].update(encoded.encode())
                counts["markdown"] += 1; by_route["generic_markdown"] += 1
                generic_original_paths.add(receipt_path)
                generic_original_paths.update(
                    PS.artifact_path(artifact_root, artifact)
                    for artifact in receipt["artifacts"].values())
                if context_links:
                    for ordinal, link in enumerate(context_links):
                        target_host = (urlsplit(link["url"]).hostname or "").casefold()
                        outlink = {
                            "schema": "aw.parent-source-link.v1",
                            "document_id": document_id,
                            "source_record_id": url,
                            "source_url": url, "ordinal": ordinal,
                            "target_url": link["url"],
                            "nofollow": link["nofollow"],
                            "anchor_or_context": link["anchor_or_context"],
                            "relation_scope": ("same-site" if reg_domain(source_host) ==
                                               reg_domain(target_host) else "external"),
                            "is_empty": False,
                            "parent_markdown_sha256": parent["markdown_sha256"],
                        }
                        encoded = canonical_json(outlink) + "\n"
                        outlinks_out.write(encoded); logical["outlinks"].update(encoded.encode())
                        counts["outlinks"] += 1
                else:
                    outlink = {
                        "schema": "aw.parent-source-link.v1",
                        "document_id": document_id,
                        "source_record_id": url,
                        "source_url": url, "ordinal": 0,
                        "target_url": None, "nofollow": False,
                        "anchor_or_context": None, "relation_scope": None,
                        "is_empty": True,
                        "parent_markdown_sha256": parent["markdown_sha256"],
                    }
                    encoded = canonical_json(outlink) + "\n"
                    outlinks_out.write(encoded); logical["outlinks"].update(encoded.encode())
                    counts["outlinks"] += 1
        if (frontier_urls != declared.get("urls")
                or frontier_urls != (cc_run.get("counts") or {}).get("frontier")
                or captures != (cc_run.get("counts") or {}).get("cc_captures")
                or misses != (cc_run.get("counts") or {}).get("cc_misses")
                or captures + misses != frontier_urls):
            raise ValueError("Route-C frontier/capture conservation failed")
        if not generic_original_paths:
            raise ValueError("Route-C generic-web admission has no source-complete parents")
        generic_original_logical = hashlib.sha256(b"".join(
            (canonical_json(_binding(path, artifact_root)) + "\n").encode()
            for path in sorted(generic_original_paths)))
        stages = {
            "original": SS.seal(
                artifact_root=artifact_root,
                artifact_paths=sorted(generic_original_paths),
                receipt_path=partial / "original/receipt.json",
                records=counts["markdown"],
                fmt="exact-cc-warc+http+decoded-html+transport+terminal-receipts",
                logical_sha256=generic_original_logical.hexdigest(),
                producer_id=PRODUCER, producer_version=VERSION,
                publish_from=partial, publish_to=output,
                allow_existing_published=True),
            "config": SS.seal(
                artifact_root=artifact_root, artifact_paths=[registry_copy],
                receipt_path=partial / "config/receipt.json", records=1,
                fmt="reviewed-exact-url-source-route-registry+canonical-json",
                logical_sha256=sha256_file(registry_copy),
                producer_id=PRODUCER, producer_version=VERSION,
                publish_from=partial, publish_to=output),
            "routes": _stage(artifact_root=artifact_root, partial=partial,
                output=output, name="routes", data_path=paths["routes"],
                records=counts["routes"], logical=logical["routes"],
                fmt="exact-url-source-routes+canonical-jsonl+deterministic-gzip"),
            "capture_outlinks": _stage(
                artifact_root=artifact_root, partial=partial, output=output,
                name="capture_outlinks", data_path=paths["capture_outlinks"],
                records=counts["capture_outlinks"],
                logical=logical["capture_outlinks"],
                fmt="capture-bound-context-outlinks+canonical-jsonl+deterministic-gzip"),
            "markdown": _stage(artifact_root=artifact_root, partial=partial,
                output=output, name="markdown", data_path=paths["markdown"],
                records=counts["markdown"], logical=logical["markdown"],
                fmt="full-principal-content-markdown+canonical-jsonl+deterministic-gzip"),
            "outlinks": _stage(artifact_root=artifact_root, partial=partial,
                output=output, name="outlinks", data_path=paths["outlinks"],
                records=counts["outlinks"], logical=logical["outlinks"],
                fmt="parent-bound-context-outlinks+canonical-jsonl+deterministic-gzip"),
            "residual": _stage(artifact_root=artifact_root, partial=partial,
                output=output, name="residual", data_path=paths["residual"],
                records=counts["residual"], logical=logical["residual"],
                fmt="explicit-cc-miss-routes+canonical-jsonl+deterministic-gzip"),
            "quarantine": _stage(artifact_root=artifact_root, partial=partial,
                output=output, name="quarantine", data_path=paths["quarantine"],
                records=counts["quarantine"], logical=logical["quarantine"],
                fmt="explicit-generic-web-quarantine+canonical-jsonl+deterministic-gzip"),
        }
        result = {
            "schema": "aw.routec-preserved-html-materialization-run.v1",
            "status": "complete",
            "admitted_source_profile": "web_html",
            "counts": {"frontier_urls": frontier_urls,
                       "production_points": frontier_points,
                       "cc_captures": captures, "cc_misses": misses,
                       "generic_markdown_parents": counts["markdown"],
                       "generic_quarantine": counts["quarantine"],
                       "capture_outlink_rows": counts["capture_outlinks"],
                       "outlink_rows": counts["outlinks"]},
            "route_counts": dict(sorted(route_counts.items())),
            "classification": {
                "basis": registry["identity_basis"],
                "payload_grouping_used": False,
                "registry": _published_binding(
                    registry_copy, partial, output, artifact_root),
                "source_registry_input_sha256": sha256_file(route_registry_path),
            },
            "inputs": {"frontier": _binding(frontier, artifact_root),
                       "frontier_receipt": _binding(frontier_receipt, artifact_root),
                       "cc_run": _binding(cc_run_path, artifact_root)},
            "stages": {"all_capture_original": original_stage, **stages},
            "next": {
                "source_specific": "consume route queues with the declared source-native adapters",
                "generic_web": "seal fidelity/profile then source-record-local chunk",
                "cc_misses": "official/feed route first; direct retrieval only for generic residual",
            },
        }
        atomic_publish(partial / "RUN.json", (json.dumps(
            result, indent=2, sort_keys=True) + "\n").encode())
        os.rename(partial, output)
        return result
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
