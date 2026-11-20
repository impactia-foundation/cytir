#!/usr/bin/env python3
"""Materialize exact Route-C Internet-Draft records from preserved Datatracker responses."""
from __future__ import annotations

import argparse
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
from cytir.extract.html_extract import extract_outlink_context_records, _parse_soup
from cytir.extract.routec import common as ROUTEC
from cytir.extract.routec.declaration import load_declaration
from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS


VERSION = "routec-ietf-datatracker-draft-record.v1"
PRODUCER = "aw.extract.routec.ietf_internet_drafts"
SOURCE_ID = "routec-ietf-datatracker-internet-draft-records-current"
DISPOSITION = "residual_internet_draft_requires_source_specific_projection"


def _binding(path: Path, root: Path) -> dict:
    return ROUTEC.binding(path.resolve(), root.resolve())


def _validate_binding(root: Path, artifact: dict) -> Path:
    path = PS.artifact_path(root, artifact)
    if artifact.get("compression") == "gzip":
        PS.validate_artifact(root, artifact)
    elif (not path.is_file() or path.stat().st_size != artifact.get("bytes")
          or sha256_file(path) != artifact.get("sha256")):
        raise ValueError(f"IETF draft artifact binding failed: {path}")
    return path


def _logical(root: Path, artifact: dict) -> bytes:
    _validate_binding(root, artifact)
    data = PS.artifact_path(root, artifact).read_bytes()
    return gzip.decompress(data) if artifact.get("compression") == "gzip" else data


def render_draft_record(html: str, *, url: str, final_url: str,
                        body_sha256: str) -> tuple[str, str, dict]:
    soup = _parse_soup(html)
    if soup is None:
        raise ValueError(f"Datatracker HTML parse failed: {url}")
    root = soup.find("main")
    if root is None:
        raise ValueError(f"Datatracker record lacks main: {url}")
    for element in root.find_all(("script", "style", "nav")):
        element.decompose()
    title = " ".join((soup.title.get_text(" ", strip=True)
                      if soup.title else "").split())
    heading = root.find("h1")
    heading_text = " ".join(heading.get_text(" ", strip=True).split()) if heading else ""
    if not title or not heading_text or "draft-" not in heading_text.casefold():
        raise ValueError(f"Datatracker draft identity absent: {url}")
    lines = [f"# {title}", "", "## Source identity",
             f"- Official Datatracker record URL: {url}",
             f"- Final response URL: {final_url}",
             f"- Preserved response SHA-256: `{body_sha256}`", "",
             "## Official Internet-Draft record"]
    emitted: list[str] = []

    def text(node) -> str:
        return " ".join(node.get_text(" ", strip=True).split())

    supported = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "dt", "dd",
                 "pre", "li", "tr"}
    for element in root.find_all(list(supported)):
        if any(parent is not root and getattr(parent, "name", None) in supported
               and parent.name != "tr" for parent in element.parents):
            continue
        value = text(element)
        if not value:
            continue
        if element.name.startswith("h"):
            lines.extend(("", "#" * min(6, int(element.name[1]) + 1) + " " + value))
            emitted.append(value)
        elif element.name == "tr":
            cells = [text(cell) for cell in element.find_all(("th", "td"), recursive=False)]
            if cells: lines.extend(("", " | ".join(cells))); emitted.extend(cells)
        elif element.name == "li": lines.append(f"- {value}"); emitted.append(value)
        elif element.name == "dt": lines.extend(("", f"**{value}**")); emitted.append(value)
        elif element.name == "dd": lines.append(value); emitted.append(value)
        elif element.name == "pre": lines.extend(("", "```text", value, "```")); emitted.append(value)
        else: lines.extend(("", value)); emitted.append(value)
    emitted_text = "\n".join(emitted)
    missing = []
    for value in (" ".join(item.split()) for item in root.stripped_strings):
        if value and value not in emitted_text and value not in missing: missing.append(value)
    if missing:
        lines.extend(("", "## Additional Datatracker fields", ""))
        lines.extend(f"- {value}" for value in missing)
    markdown = "\n".join(lines).strip() + "\n"
    normalized = " ".join(markdown.split())
    if any(value and value not in normalized
           for value in (" ".join(item.split()) for item in root.stripped_strings)):
        raise ValueError(f"Datatracker visible record text lost: {url}")
    version_match = re.search(r"\b(draft-[a-z0-9-]+)-(\d{2})\b", heading_text, re.I)
    draft_name = re.search(r"/doc/(draft-[^/?#]+)", urlsplit(url).path, re.I)
    version = {"draft_name": draft_name.group(1).lower() if draft_name else None,
               "displayed_version": version_match.group(0).lower() if version_match else None,
               "requested_url": url, "final_url": final_url,
               "identity_basis": "exact-Datatracker-draft-record-url+preserved-current-response"}
    return title, markdown, version


def materialize(*, artifact_root: Path, routes_path: Path, resolution_path: Path,
                output_root: Path, declaration: Path) -> dict:
    expected = load_declaration(declaration, source_id=SOURCE_ID, keys=('route_urls', 'route_reported_production_points', 'source_parents', 'source_parent_reported_production_points', 'redirected_current_rfc_records', 'redirected_current_rfc_reported_production_points'))
    artifact_root = artifact_root.resolve(); output_root = output_root.resolve()
    output_root.relative_to(artifact_root)
    if output_root.exists(): raise ValueError(f"refusing existing IETF draft output: {output_root}")
    partial = output_root.with_name(output_root.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    paths = {"markdown": partial / "markdown/ietf-draft-records.jsonl.gz",
             "outlinks": partial / "outlinks/ietf-draft-record-links.jsonl.gz"}
    for path in paths.values(): path.parent.mkdir(parents=True, exist_ok=True)
    logical = {name: hashlib.sha256() for name in paths}; counts = {name: 0 for name in paths}
    originals: set[Path] = set(); route_points = admitted_points = 0
    rfc_redirect_points = 0; rfc_redirects: list[dict] = []

    def write(name: str, row: dict, handle) -> None:
        encoded = canonical_json(row) + "\n"; handle.write(encoded)
        logical[name].update(encoded.encode()); counts[name] += 1

    try:
        selected = {}
        with gzip.open(resolution_path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row.get("disposition") == DISPOSITION:
                    url = str(row.get("url") or "")
                    if url in selected: raise ValueError(f"duplicate IETF draft resolution: {url}")
                    selected[url] = row
        if (len(selected) != expected["route_urls"] or
                sum(row["reported_production_points"] for row in selected.values()) != expected["route_reported_production_points"]):
            raise ValueError("IETF draft exact closure selector drift")
        seen = set()
        with gzip.open(routes_path, "rt", encoding="utf-8") as routes, \
                deterministic_gzip_text(paths["markdown"]) as parents_out, \
                deterministic_gzip_text(paths["outlinks"]) as links_out:
            for line in routes:
                route = json.loads(line); url = str(route.get("url") or "")
                resolution = selected.get(url)
                if resolution is None: continue
                points = route.get("production_points")
                if (url in seen or urlsplit(url).hostname != "datatracker.ietf.org"
                        or route.get("direct_status") != "complete"
                        or points != resolution.get("reported_production_points")):
                    raise ValueError(f"IETF draft exact route join failed: {url}")
                seen.add(url); route_points += points
                terminal = route.get("terminal_receipt") or {}
                artifacts = route.get("original_artifacts") or {}
                if set(artifacts) != {"response_body", "response_headers", "transport", "consent"}:
                    raise ValueError(f"IETF draft artifact set drift: {url}")
                originals.add(_validate_binding(artifact_root, terminal))
                for artifact in artifacts.values(): originals.add(_validate_binding(artifact_root, artifact))
                body = _logical(artifact_root, artifacts["response_body"])
                body_sha = hashlib.sha256(body).hexdigest()
                if body_sha != artifacts["response_body"].get("body_sha256"):
                    raise ValueError(f"IETF draft body digest failed: {url}")
                final_url = str(route.get("final_url") or url)
                if re.fullmatch(r"/doc/rfc\d+/?", urlsplit(final_url).path, re.I):
                    rfc_redirect_points += points
                    rfc_redirects.append({"requested_url": url, "final_url": final_url,
                        "reported_production_points": points,
                        "terminal_receipt": terminal,
                        "disposition": "current-response-is-RFC-record-not-draft-record"})
                    continue
                admitted_points += points
                title, markdown, version = render_draft_record(
                    body.decode("utf-8", "replace"), url=url, final_url=final_url,
                    body_sha256=body_sha)
                markdown_sha = hashlib.sha256(markdown.encode()).hexdigest()
                document_id = hashlib.sha256(f"ietf-draft-record\0{url}".encode()).hexdigest()
                identity = {"terminal_receipt": terminal, "artifacts": artifacts,
                            "body_sha256": body_sha,
                            "transport_kind": "consent-gated-direct-after-cc-miss"}
                parent = {"schema": "aw.ietf-datatracker-draft-record-parent.v1",
                          "document_id": document_id, "source_record_id": url,
                          "source_id": SOURCE_ID, "source_profile": "structured_data_feed",
                          "url": url, "canonical_url": final_url, "title": title, "lang": "en",
                          "markdown": markdown, "markdown_sha256": markdown_sha,
                          "parent_source_sha256": body_sha, "parent_identity": identity,
                          "source_version": version, "reported_production_points": points,
                          "content_scope": "complete Datatracker document record; linked draft text is not claimed as embedded",
                          "extraction": {"producer": PRODUCER,
                                         "full_prechunk_principal_content": True,
                                         "source_text_truncation_applied": False}}
                write("markdown", parent, parents_out)
                soup = _parse_soup(body.decode("utf-8", "replace")); main = soup.find("main")
                links = extract_outlink_context_records(main, final_url) if main else []
                if not links: links = [{"url": None, "nofollow": False, "anchor_or_context": None}]
                for ordinal, link in enumerate(links):
                    write("outlinks", {"schema": "aw.parent-source-link.v1",
                        "document_id": document_id, "source_record_id": url,
                        "source_url": url, "ordinal": ordinal,
                        "target_url": urljoin(final_url, link["url"]) if link["url"] else None,
                        "nofollow": link["nofollow"], "anchor_or_context": link["anchor_or_context"],
                        "relation": "ietf-draft-record-source-link" if link["url"] else None,
                        "is_empty": not bool(link["url"]), "parent_markdown_sha256": markdown_sha,
                        "parent_identity": identity, "source_version": version}, links_out)
        if (len(seen) != expected["route_urls"] or route_points != expected["route_reported_production_points"]
                or counts["markdown"] != expected["source_parents"]
                or admitted_points != expected["source_parent_reported_production_points"]
                or len(rfc_redirects) != expected["redirected_current_rfc_records"]
                or rfc_redirect_points != expected["redirected_current_rfc_reported_production_points"]):
            raise ValueError("IETF draft route conservation failed")
        original_logical = hashlib.sha256(b"".join(
            (canonical_json(_binding(path, artifact_root)) + "\n").encode()
            for path in sorted(originals)))
        stages = {"original": SS.seal(artifact_root=artifact_root,
            artifact_paths=sorted(originals), receipt_path=partial / "original/receipt.json",
            records=expected["route_urls"], fmt="exact-official-Datatracker-current-response+transport+terminal-receipts",
            logical_sha256=original_logical.hexdigest(), producer_id=PRODUCER,
            producer_version=VERSION, publish_from=partial, publish_to=output_root,
            allow_existing_published=True)}
        for name, fmt in (("markdown", "Datatracker-draft-record-markdown+canonical-jsonl+deterministic-gzip"),
                          ("outlinks", "Datatracker-parent-bound-draft-record-links+canonical-jsonl+deterministic-gzip")):
            stages[name] = SS.seal(artifact_root=artifact_root, artifact_paths=[paths[name]],
                receipt_path=partial / name / "receipt.json", records=counts[name], fmt=fmt,
                logical_sha256=logical[name].hexdigest(), producer_id=PRODUCER,
                producer_version=VERSION, publish_from=partial, publish_to=output_root)
        result = {"schema": "aw.routec-ietf-datatracker-draft-record-run.v1",
                  "version": VERSION, "status": "complete", "source_id": SOURCE_ID,
                  "source_profile": "structured_data_feed",
                  "route_rows": _binding(routes_path, artifact_root),
                  "resolution_selector": _binding(resolution_path, artifact_root),
                  "point_identity": {"status": "not_present_in_route_registry",
                                     "ledger_subtraction_authorized": False},
                  "counts": {"route_urls": expected["route_urls"],
                             "route_reported_production_points": expected["route_reported_production_points"],
                             "source_parents": expected["source_parents"],
                             "source_parent_reported_production_points": expected["source_parent_reported_production_points"],
                             "direct_responses": expected["route_urls"],
                             "redirected_current_rfc_records": expected["redirected_current_rfc_records"],
                             "redirected_current_rfc_reported_production_points": expected["redirected_current_rfc_reported_production_points"],
                             "outlink_rows_including_empty": counts["outlinks"]},
                  "redirected_current_rfc_records": rfc_redirects,
                  "content_scope": "complete official Datatracker records; archive-linked draft full text remains a separate source object",
                  "stages": stages}
        atomic_publish(partial / "RUN.json", (json.dumps(result, indent=2, sort_keys=True)+"\n").encode())
        os.rename(partial, output_root); return result
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True); raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--resolution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(artifact_root=args.artifact_root,
        routes_path=args.routes, resolution_path=args.resolution,
        output_root=args.output, declaration=args.declaration), indent=2, sort_keys=True))


if __name__ == "__main__": main()
