#!/usr/bin/env python3
"""Materialize exact Route-C IANA registry records from preserved official HTML."""
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


VERSION = "routec-iana-registry-materialization.v1"
PRODUCER = "aw.extract.routec.iana_registries"
ROUTE_ID = "iana-registries"
SOURCE_ID = "routec-iana-official-registry-records-current"


def _binding(path: Path, root: Path) -> dict:
    return ROUTEC.binding(path.resolve(), root.resolve())


def _validate_binding(root: Path, artifact: dict) -> Path:
    path = PS.artifact_path(root, artifact)
    if artifact.get("compression") == "gzip":
        PS.validate_artifact(root, artifact)
    elif (not path.is_file() or path.stat().st_size != artifact.get("bytes")
          or sha256_file(path) != artifact.get("sha256")):
        raise ValueError(f"IANA artifact binding failed: {path}")
    return path


def _logical(root: Path, artifact: dict) -> bytes:
    _validate_binding(root, artifact)
    data = PS.artifact_path(root, artifact).read_bytes()
    return gzip.decompress(data) if artifact.get("compression") == "gzip" else data


def render_registry_markdown(html: str, *, url: str, body_sha256: str,
                             transport_kind: str) -> tuple[str, str, dict]:
    soup = _parse_soup(html)
    if soup is None:
        raise ValueError(f"IANA HTML parse failed: {url}")
    root = soup.find("main") or soup.find("article")
    if root is None:
        raise ValueError(f"IANA record lacks main/article: {url}")
    for element in root.find_all(("script", "style", "nav")):
        element.decompose()
    title = " ".join((soup.title.get_text(" ", strip=True)
                      if soup.title else "").split())
    if not title:
        heading = root.find(("h1", "h2"))
        title = " ".join(heading.get_text(" ", strip=True).split()) if heading else url
    lines = [f"# {title}", "", "## Source identity",
             f"- Official IANA URL: {url}",
             f"- Preserved response SHA-256: `{body_sha256}`",
             f"- Transport route: {transport_kind}", "", "## Official registry record"]
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
            level = min(6, int(element.name[1]) + 1)
            lines.extend(("", "#" * level + " " + value)); emitted.append(value)
        elif element.name == "tr":
            cells = [text(cell) for cell in element.find_all(("th", "td"), recursive=False)]
            if cells:
                lines.extend(("", " | ".join(cells))); emitted.extend(cells)
        elif element.name == "li":
            lines.append(f"- {value}"); emitted.append(value)
        elif element.name == "dt":
            lines.extend(("", f"**{value}**")); emitted.append(value)
        elif element.name == "dd":
            lines.append(value); emitted.append(value)
        elif element.name == "pre":
            lines.extend(("", "```text", value, "```")); emitted.append(value)
        else:
            lines.extend(("", value)); emitted.append(value)
    emitted_text = "\n".join(emitted)
    missing = []
    for value in (" ".join(item.split()) for item in root.stripped_strings):
        if value and value not in emitted_text and value not in missing:
            missing.append(value)
    if missing:
        lines.extend(("", "## Additional registry fields", ""))
        lines.extend(f"- {value}" for value in missing)
    visible = " ".join(root.get_text(" ", strip=True).split())
    markdown = "\n".join(lines).strip() + "\n"
    normalized_markdown = " ".join(markdown.split())
    missing_visible = [value for value in (" ".join(s.split()) for s in root.stripped_strings)
                       if value and value not in normalized_markdown]
    if missing_visible:
        raise ValueError(f"IANA visible record text lost: {url}")
    updated = None
    match = re.search(r"(?:Last Updated|Record last updated)\s*[: ]\s*"
                      r"([0-9]{4}-[0-9]{2}(?:-[0-9]{2})?)", visible, re.I)
    if match:
        updated = match.group(1)
    version = {"record_last_updated": updated, "transport_kind": transport_kind,
               "identity_basis": "exact-route-url+preserved-official-response"}
    return title, markdown, version


def materialize(*, artifact_root: Path, routes_path: Path, direct_root: Path,
                output_root: Path, declaration: Path) -> dict:
    expected = load_declaration(declaration, source_id=SOURCE_ID, keys=('route_urls', 'direct_responses', 'cc_responses'))
    artifact_root = artifact_root.resolve(); routes_path = routes_path.resolve()
    direct_root = direct_root.resolve(); output_root = output_root.resolve()
    output_root.relative_to(artifact_root)
    if output_root.exists():
        raise ValueError(f"refusing existing IANA materialization: {output_root}")
    partial = output_root.with_name(output_root.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    paths = {"markdown": partial / "markdown/iana-records.jsonl.gz",
             "outlinks": partial / "outlinks/iana-record-links.jsonl.gz"}
    for path in paths.values(): path.parent.mkdir(parents=True, exist_ok=True)
    logical = {name: hashlib.sha256() for name in paths}
    counts = {name: 0 for name in paths}; originals = set()
    route_points = direct_count = cc_count = 0

    def write(name: str, row: dict, handle) -> None:
        encoded = canonical_json(row) + "\n"; handle.write(encoded)
        logical[name].update(encoded.encode()); counts[name] += 1

    try:
        with gzip.open(routes_path, "rt", encoding="utf-8") as routes, \
                deterministic_gzip_text(paths["markdown"]) as parents_out, \
                deterministic_gzip_text(paths["outlinks"]) as links_out:
            seen = set()
            for line_number, line in enumerate(routes, 1):
                route = json.loads(line)
                if route.get("route_id") != ROUTE_ID: continue
                url = str(route.get("url") or ""); points = route.get("production_points")
                if (url in seen or urlsplit(url).hostname != "www.iana.org"
                        or not isinstance(points, int) or points != 1):
                    raise ValueError(f"invalid IANA route row: {line_number}")
                seen.add(url); route_points += points
                if route.get("cc_status") == "complete":
                    terminal = route.get("terminal_receipt") or {}
                    artifacts = route.get("original_artifacts") or {}
                    body_key = "decoded_html"; transport_kind = "signed-s3-common-crawl"
                    required = {"decoded_html", "http_headers", "raw_warc", "response_body",
                                "transport"}; cc_count += 1
                elif route.get("cc_status") == "cc_miss":
                    digest = hashlib.sha256(url.encode()).hexdigest()
                    receipt_path = direct_root / "receipts" / f"{digest}.json"
                    receipt = json.loads(receipt_path.read_text())
                    if (receipt.get("schema") != "aw.exact-url-direct-receipt.v1"
                            or receipt.get("status") != "complete"
                            or receipt.get("source_url") != url):
                        raise ValueError(f"IANA direct terminal receipt failed: {url}")
                    terminal = _binding(receipt_path, artifact_root)
                    artifacts = receipt.get("artifacts") or {}
                    body_key = "response_body"; transport_kind = "consent-gated-direct-after-cc-miss"
                    required = {"response_body", "response_headers", "transport", "consent"}
                    direct_count += 1
                else:
                    raise ValueError(f"IANA route has no terminal source response: {url}")
                if set(artifacts) != required:
                    raise ValueError(f"IANA artifact set drift: {url}")
                terminal_path = _validate_binding(artifact_root, terminal)
                originals.add(terminal_path)
                for artifact in artifacts.values(): originals.add(_validate_binding(artifact_root, artifact))
                body = _logical(artifact_root, artifacts[body_key])
                body_sha = hashlib.sha256(body).hexdigest()
                if body_sha != artifacts[body_key].get("body_sha256"):
                    raise ValueError(f"IANA decompressed body digest failed: {url}")
                html = body.decode("utf-8", "replace")
                title, markdown, version = render_registry_markdown(
                    html, url=url, body_sha256=body_sha, transport_kind=transport_kind)
                markdown_sha = hashlib.sha256(markdown.encode()).hexdigest()
                document_id = hashlib.sha256(f"iana-registry-record\0{url}".encode()).hexdigest()
                identity = {"terminal_receipt": terminal, "artifacts": artifacts,
                            "body_sha256": body_sha, "transport_kind": transport_kind}
                parent = {"schema": "aw.iana-registry-parent.v1",
                          "document_id": document_id, "source_record_id": url,
                          "source_id": SOURCE_ID, "source_profile": "structured_data_feed",
                          "url": url, "title": title, "lang": "en",
                          "markdown": markdown, "markdown_sha256": markdown_sha,
                          "parent_source_sha256": body_sha, "parent_identity": identity,
                          "source_version": version, "reported_production_points": points,
                          "extraction": {"producer": PRODUCER,
                                         "full_prechunk_principal_content": True,
                                         "source_text_truncation_applied": False}}
                write("markdown", parent, parents_out)
                soup = _parse_soup(html); root = soup.find("main") or soup.find("article")
                links = extract_outlink_context_records(root, url) if root is not None else []
                if not links: links = [{"url": None, "nofollow": False,
                                        "anchor_or_context": None}]
                for ordinal, link in enumerate(links):
                    write("outlinks", {"schema": "aw.parent-source-link.v1",
                        "document_id": document_id, "source_record_id": url,
                        "source_url": url, "ordinal": ordinal,
                        "target_url": (urljoin(url, link["url"]) if link["url"] else None),
                        "nofollow": link["nofollow"],
                        "anchor_or_context": link["anchor_or_context"],
                        "relation": "iana-registry-source-link" if link["url"] else None,
                        "is_empty": not bool(link["url"]),
                        "parent_markdown_sha256": markdown_sha,
                        "parent_identity": identity, "source_version": version}, links_out)
        if (len(seen) != expected["route_urls"] or route_points != expected["route_urls"]
                or direct_count != expected["direct_responses"] or cc_count != expected["cc_responses"] or counts["markdown"] != expected["route_urls"]):
            raise ValueError("IANA exact Route-C conservation failed")
        original_logical = hashlib.sha256(b"".join(
            (canonical_json(_binding(path, artifact_root)) + "\n").encode()
            for path in sorted(originals)))
        stages = {"original": SS.seal(
            artifact_root=artifact_root, artifact_paths=sorted(originals),
            receipt_path=partial / "original/receipt.json", records=expected["route_urls"],
            fmt="exact-official-IANA-current-direct-or-CC-response+transport+terminal-receipts",
            logical_sha256=original_logical.hexdigest(), producer_id=PRODUCER,
            producer_version=VERSION, publish_from=partial, publish_to=output_root,
            allow_existing_published=True)}
        for name, fmt in (("markdown", "IANA-registry-record-markdown+canonical-jsonl+deterministic-gzip"),
                          ("outlinks", "IANA-parent-bound-registry-links+canonical-jsonl+deterministic-gzip")):
            stages[name] = SS.seal(artifact_root=artifact_root, artifact_paths=[paths[name]],
                receipt_path=partial / name / "receipt.json", records=counts[name], fmt=fmt,
                logical_sha256=logical[name].hexdigest(), producer_id=PRODUCER,
                producer_version=VERSION, publish_from=partial, publish_to=output_root)
        result = {"schema": "aw.routec-iana-registry-run.v1", "version": VERSION,
                  "status": "complete", "source_id": SOURCE_ID,
                  "source_profile": "structured_data_feed",
                  "route_rows": _binding(routes_path, artifact_root),
                  "point_identity": {"status": "not_present_in_route_registry",
                                     "ledger_subtraction_authorized": False},
                  "counts": {"route_urls": expected["route_urls"],
                             "route_reported_production_points": expected["route_urls"],
                             "source_parents": expected["route_urls"], "direct_responses": direct_count,
                             "signed_s3_cc_responses": cc_count,
                             "outlink_rows_including_empty": counts["outlinks"]},
                  "stages": stages}
        atomic_publish(partial / "RUN.json", (
            json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        os.rename(partial, output_root); return result
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True); raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--routes", type=Path, required=True)
    ap.add_argument("--direct-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--declaration", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(materialize(artifact_root=args.artifact_root,
        routes_path=args.routes, direct_root=args.direct_root,
        output_root=args.output, declaration=args.declaration), indent=2, sort_keys=True))


if __name__ == "__main__": main()
