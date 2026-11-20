#!/usr/bin/env python3
"""Materialize separate exact Datatracker agenda or minutes record lanes."""
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


VERSION = "routec-ietf-datatracker-meeting-record.v1"
PRODUCER = "aw.extract.routec.ietf_meeting_records"
DISPOSITION = "residual_other_ietf_object_requires_source_specific_projection"
CONFIG = {
    "agenda": {"pattern": re.compile(r"https://datatracker\.ietf\.org/doc/agenda-[^?#]+$"),
        "slug": "agenda", "identity_terms": ("agenda",),
        "scope": "complete official Datatracker agenda record and full displayed agenda text"},
    "minutes": {"pattern": re.compile(r"https://datatracker\.ietf\.org/doc/minutes-[^/?#]+$"),
        "slug": "minutes", "identity_terms": ("minutes",),
        "scope": "complete official Datatracker minutes record and full displayed minutes text"},
    "charter": {"pattern": re.compile(r"https://datatracker\.ietf\.org/doc/charter-[^/?#]+$"),
        "slug": "charter", "identity_terms": ("charter",),
        "scope": "complete official Datatracker charter record and full displayed charter text"},
    "bof_request": {"pattern": re.compile(r"https://datatracker\.ietf\.org/doc/bofreq-[^/?#]+$"),
        "slug": "bof-request", "identity_terms": ("bof request", "bofreq-"),
        "scope": "complete official Datatracker BOF-request record and full displayed request text"},
    "narrative_minutes": {"pattern": re.compile(r"https://datatracker\.ietf\.org/doc/narrative-minutes-[^/?#]+$"),
        "slug": "narrative-minutes",
        "identity_terms": ("narrative minutes", "narrative-minutes-"),
        "scope": "complete official Datatracker narrative-minutes record and full displayed narrative text"},
    "conflict_review": {"pattern": re.compile(r"https://datatracker\.ietf\.org/doc/conflict-review-[^/?#]+$"),
        "slug": "conflict-review",
        "identity_terms": ("conflict review", "conflict-review-"),
        "scope": "complete official Datatracker conflict-review record and full displayed conflict-review text"},
}


def source_id(kind: str) -> str:
    return f"routec-ietf-datatracker-{CONFIG[kind]['slug']}-records-current"


def _binding(path: Path, root: Path) -> dict:
    return ROUTEC.binding(path.resolve(), root.resolve())


def _validate(root: Path, artifact: dict) -> Path:
    path = PS.artifact_path(root, artifact)
    if artifact.get("compression") == "gzip":
        PS.validate_artifact(root, artifact)
    elif (not path.is_file() or path.stat().st_size != artifact.get("bytes")
          or sha256_file(path) != artifact.get("sha256")):
        raise ValueError(f"IETF meeting artifact binding failed: {path}")
    return path


def _logical(root: Path, artifact: dict) -> bytes:
    data = _validate(root, artifact).read_bytes()
    return gzip.decompress(data) if artifact.get("compression") == "gzip" else data


def render_meeting_record(html: str, *, kind: str, url: str, final_url: str,
                          body_sha256: str) -> tuple[str, str, dict]:
    config = CONFIG[kind]
    if not config["pattern"].fullmatch(url):
        raise ValueError(f"Datatracker {kind} requested identity drift: {url}")
    soup = _parse_soup(html)
    root = soup.find("main") if soup is not None else None
    if root is None:
        raise ValueError(f"Datatracker {kind} lacks main: {url}")
    for element in root.find_all(("script", "style", "nav")):
        element.decompose()
    title = " ".join((soup.title.get_text(" ", strip=True) if soup.title else "").split())
    heading = root.find("h1")
    heading_text = " ".join(heading.get_text(" ", strip=True).split()) if heading else ""
    displayed_identity = f"{title} {heading_text}".lower()
    if (not title or not heading_text
            or not any(term in displayed_identity for term in config["identity_terms"])):
        raise ValueError(f"Datatracker {kind} displayed identity absent: {url}")
    path = urlsplit(url).path.rstrip("/")
    requested_name = path.split("/doc/", 1)[-1]
    displayed = re.search(rf"\b({kind}-[a-z0-9-]+?)(?:-(\d{{2}}))?\b", heading_text, re.I)
    lines = [f"# {title}", "", "## Source identity",
        f"- Official Datatracker {kind} URL: {url}",
        f"- Final response URL: {final_url}",
        f"- Preserved response SHA-256: `{body_sha256}`", "",
        f"## Official {kind} record"]
    emitted = []

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
        lines.extend(("", "## Additional Datatracker fields", ""))
        lines.extend(f"- {value}" for value in missing)
    markdown = "\n".join(lines).strip() + "\n"
    normalized = " ".join(markdown.split())
    if any(value and value not in normalized
           for value in (" ".join(item.split()) for item in root.stripped_strings)):
        raise ValueError(f"Datatracker {kind} visible text lost: {url}")
    version = {"record_kind": kind, "requested_name": requested_name,
        "displayed_version": displayed.group(0).lower() if displayed else None,
        "requested_url": url, "final_url": final_url,
        "identity_basis": f"exact-Datatracker-{kind}-record-url+preserved-current-response"}
    return title, markdown, version


def materialize(*, artifact_root: Path, routes_path: Path, resolution_path: Path,
                output_root: Path, kind: str, declaration: Path) -> dict:
    expected = load_declaration(declaration, source_id=source_id(kind), keys=('route_urls', 'route_reported_production_points'))
    config = CONFIG[kind]
    artifact_root = artifact_root.resolve(); output_root = output_root.resolve()
    output_root.relative_to(artifact_root)
    if output_root.exists():
        raise ValueError(f"refusing existing {kind} output: {output_root}")
    selected = {}
    with gzip.open(resolution_path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line); url = str(row.get("url") or "")
            if row.get("disposition") == DISPOSITION and config["pattern"].fullmatch(url):
                if url in selected:
                    raise ValueError(f"duplicate {kind} selector: {url}")
                selected[url] = row
    if (len(selected) != expected["route_urls"]
            or sum(row["reported_production_points"] for row in selected.values())
            != expected["route_reported_production_points"]):
        raise ValueError(f"exact Datatracker {kind} selector drift")
    partial = output_root.with_name(output_root.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    paths = {"markdown": partial/f"markdown/{kind}.jsonl.gz",
        "outlinks": partial/f"outlinks/{kind}-links.jsonl.gz"}
    for path in paths.values(): path.parent.mkdir(parents=True, exist_ok=True)
    logical = {name: hashlib.sha256() for name in paths}
    counts = {name: 0 for name in paths}; originals = set(); seen = set(); points = 0

    def write(name: str, row: dict, handle) -> None:
        encoded = canonical_json(row) + "\n"; handle.write(encoded)
        logical[name].update(encoded.encode()); counts[name] += 1
    try:
        with gzip.open(routes_path, "rt", encoding="utf-8") as routes, \
                deterministic_gzip_text(paths["markdown"]) as parents_out, \
                deterministic_gzip_text(paths["outlinks"]) as links_out:
            for line in routes:
                route = json.loads(line); url = str(route.get("url") or "")
                selector = selected.get(url)
                if selector is None: continue
                if (url in seen or route.get("direct_status") != "complete"
                        or route.get("production_points") != selector["reported_production_points"]
                        or route.get("final_url") not in (url, url + "/")):
                    raise ValueError(f"Datatracker {kind} route join failed: {url}")
                seen.add(url); points += route["production_points"]
                terminal = route.get("terminal_receipt") or {}
                artifacts = route.get("original_artifacts") or {}
                if set(artifacts) != {"response_body", "response_headers", "transport", "consent"}:
                    raise ValueError(f"Datatracker {kind} response inventory drift: {url}")
                originals.add(_validate(artifact_root, terminal))
                for artifact in artifacts.values(): originals.add(_validate(artifact_root, artifact))
                body = _logical(artifact_root, artifacts["response_body"])
                body_sha = hashlib.sha256(body).hexdigest()
                if body_sha != artifacts["response_body"].get("body_sha256"):
                    raise ValueError(f"Datatracker {kind} body digest failed: {url}")
                final_url = str(route.get("final_url"))
                title, markdown, version = render_meeting_record(
                    body.decode("utf-8", "replace"), kind=kind, url=url,
                    final_url=final_url, body_sha256=body_sha)
                markdown_sha = hashlib.sha256(markdown.encode()).hexdigest()
                document_id = hashlib.sha256(f"ietf-{kind}-record\0{url}".encode()).hexdigest()
                identity = {"terminal_receipt": terminal, "artifacts": artifacts,
                    "body_sha256": body_sha,
                    "transport_kind": "consent-gated-direct-after-cc-miss"}
                write("markdown", {"schema": f"aw.ietf-datatracker-{config['slug']}-record-parent.v1",
                    "document_id": document_id, "source_record_id": url,
                    "source_id": source_id(kind), "source_profile": "structured_data_feed",
                    "record_kind": kind, "url": url, "canonical_url": final_url,
                    "title": title, "lang": "en", "markdown": markdown,
                    "markdown_sha256": markdown_sha, "parent_source_sha256": body_sha,
                    "parent_identity": identity, "source_version": version,
                    "reported_production_points": route["production_points"],
                    "content_scope": config["scope"], "extraction": {"producer": PRODUCER,
                    "full_prechunk_principal_content": True,
                    "source_text_truncation_applied": False}}, parents_out)
                soup = _parse_soup(body.decode("utf-8", "replace")); main = soup.find("main")
                links = extract_outlink_context_records(main, final_url) if main else []
                if not links: links = [{"url": None, "nofollow": False, "anchor_or_context": None}]
                for ordinal, link in enumerate(links):
                    write("outlinks", {"schema": "aw.parent-source-link.v1",
                        "document_id": document_id, "source_record_id": url,
                        "source_url": url, "ordinal": ordinal,
                        "target_url": urljoin(final_url, link["url"]) if link["url"] else None,
                        "nofollow": link["nofollow"], "anchor_or_context": link["anchor_or_context"],
                        "relation": f"ietf-{kind}-record-source-link" if link["url"] else None,
                        "is_empty": not bool(link["url"]),
                        "parent_markdown_sha256": markdown_sha,
                        "parent_identity": identity, "source_version": version}, links_out)
        if len(seen) != expected["route_urls"] or points != expected["route_reported_production_points"]:
            raise ValueError(f"Datatracker {kind} conservation failed")
        original_logical = hashlib.sha256(b"".join(
            (canonical_json(_binding(path, artifact_root)) + "\n").encode()
            for path in sorted(originals)))
        stages = {"original": SS.seal(artifact_root=artifact_root,
            artifact_paths=sorted(originals), receipt_path=partial/"original/receipt.json",
            records=expected["route_urls"], fmt=f"exact-official-Datatracker-{kind}-response+transport+terminal-receipts",
            logical_sha256=original_logical.hexdigest(), producer_id=PRODUCER,
            producer_version=VERSION, publish_from=partial, publish_to=output_root,
            allow_existing_published=True)}
        for name, fmt in (("markdown", f"Datatracker-{kind}-record-markdown+canonical-jsonl+deterministic-gzip"),
                          ("outlinks", f"Datatracker-parent-bound-{kind}-links+canonical-jsonl+deterministic-gzip")):
            stages[name] = SS.seal(artifact_root=artifact_root, artifact_paths=[paths[name]],
                receipt_path=partial/name/"receipt.json", records=counts[name], fmt=fmt,
                logical_sha256=logical[name].hexdigest(), producer_id=PRODUCER,
                producer_version=VERSION, publish_from=partial, publish_to=output_root)
        result = {"schema": f"aw.routec-ietf-datatracker-{kind}-record-run.v1",
            "version": VERSION, "status": "complete", "record_kind": kind,
            "source_id": source_id(kind), "source_profile": "structured_data_feed",
            "route_rows": _binding(routes_path, artifact_root),
            "resolution_selector": _binding(resolution_path, artifact_root),
            "point_identity": {"status": "not_present_in_route_registry",
                               "ledger_subtraction_authorized": False},
            "counts": {"route_urls": expected["route_urls"],
                "route_reported_production_points": expected["route_reported_production_points"],
                "source_parents": expected["route_urls"], "direct_responses": expected["route_urls"],
                "outlink_rows_including_empty": counts["outlinks"]}, "stages": stages}
        atomic_publish(partial/"RUN.json", (json.dumps(result, indent=2, sort_keys=True)+"\n").encode())
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
    parser.add_argument("--record-kind", choices=sorted(CONFIG), required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(artifact_root=args.artifact_root,
        routes_path=args.routes, resolution_path=args.resolution,
        output_root=args.output, declaration=args.declaration, kind=args.record_kind), indent=2, sort_keys=True))


if __name__ == "__main__": main()
