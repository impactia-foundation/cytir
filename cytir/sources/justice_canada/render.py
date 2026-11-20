"""Materialize exact Route-C Justice Canada Act URLs from pinned official XML.

Moved from pipeline/materialize_routec_justice_canada_laws.py (task 1724); the functions are the
old file's, byte for byte. The `PRODUCER` string is a frozen label written into sealed receipts.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlsplit

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.acquire.deterministic_gzip import deterministic_gzip_text
import cytir.sources.routec.preserved_html as ROUTEC
from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS


VERSION = "routec-justice-canada-laws-materialization.v1"
PRODUCER = "pipeline/materialize_routec_justice_canada_laws.py"
ROUTE_ID = "justice-canada-official-xml"
SOURCE_ID = "routec-justice-canada-current-official-xml"
COMMIT = "076f5b3c3dc9b0c62d72900496b3643d75e412c2"
ACT_PATH = re.compile(r"^/eng/acts/([^/]+)/?$", re.I)
LAW_PATH = re.compile(
    r"^/(?P<source_class>eng/acts|fra/lois|eng/regulations|fra/reglements)/"
    r"(?P<slug>[^/]+)/?$",
    re.I,
)

STRUCTURAL = {
    "Statute", "Regulation", "Identification", "Body", "Heading", "Section",
    "Article", "Subsection",
    "Paragraph", "Subparagraph", "Clause", "Subclause", "Schedule",
    "ScheduleFormHeading", "TableGroup", "table", "tgroup", "thead", "tbody",
    "row", "RecentAmendments", "Amendment", "Definition", "Definitions",
    "EnablingAuthority", "BillHistory", "Stages", "Chapter",
}
MAJOR = {"Identification", "Body", "Schedule", "RecentAmendments"}
LEGAL_UNIT = {
    "Heading", "Section", "Article", "Subsection", "Paragraph", "Subparagraph", "Clause",
    "Subclause", "ScheduleFormHeading", "TableGroup", "Definition",
}


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def act_slug(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.hostname != "laws-lois.justice.gc.ca":
        raise ValueError(f"not an official Justice Canada URL: {url}")
    match = ACT_PATH.fullmatch(parsed.path)
    if not match:
        raise ValueError(f"not an English consolidated Act URL: {url}")
    slug = unquote(match.group(1))
    if not slug or "/" in slug or slug in {".", ".."}:
        raise ValueError(f"unsafe consolidated Act slug: {url}")
    return slug


def law_descriptor(url: str) -> dict[str, str]:
    """Map one exact official consolidated-law root URL to its XML class."""
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname !=
            "laws-lois.justice.gc.ca" or parsed.query or parsed.fragment):
        raise ValueError(f"not an exact official Justice Canada URL: {url}")
    match = LAW_PATH.fullmatch(parsed.path)
    if not match:
        raise ValueError(f"not a consolidated Justice Canada law URL: {url}")
    slug = unquote(match.group("slug"))
    if not slug or "/" in slug or slug in {".", ".."}:
        raise ValueError(f"unsafe consolidated-law slug: {url}")
    source_class = match.group("source_class").lower()
    lang = "fr" if source_class.startswith("fra/") else "en"
    kind = "act" if source_class.endswith(("acts", "lois")) else "regulation"
    return {"source_class": source_class, "slug": slug,
            "lang": lang, "kind": kind}


def _attrs(element: ET.Element) -> dict[str, str]:
    return {key.rsplit("}", 1)[-1]: value for key, value in element.attrib.items()}


def render_official_markdown(root: ET.Element, *, url: str, member: str,
                             member_sha256: str) -> tuple[str, str, dict]:
    """Render all official XML text with explicit legal hierarchy markers."""
    short = next((_text(node) for node in root.iter() if _tag(node) == "ShortTitle"
                  and _text(node)), "")
    long = next((_text(node) for node in root.iter() if _tag(node) == "LongTitle"
                 and _text(node)), "")
    title = short or long or law_descriptor(url)["slug"]
    attrs = _attrs(root)
    source_version = {
        "commit": COMMIT,
        "current_date": attrs.get("current-date"),
        "point_in_time_date": attrs.get("pit-date"),
        "last_amended_date": attrs.get("lastAmendedDate"),
        "in_force_start_date": attrs.get("inforce-start-date"),
        "in_force": attrs.get("in-force"),
    }
    lines = [
        f"# {title}", "", "## Source identity",
        f"- Official URL: {url}", f"- Pinned source commit: `{COMMIT}`",
        f"- Archive member: `{member}`", f"- XML SHA-256: `{member_sha256}`",
    ]
    for label, key in (
        ("Current date", "current_date"), ("Point-in-time date", "point_in_time_date"),
        ("Last amended", "last_amended_date"), ("In force since", "in_force_start_date"),
        ("In force", "in_force"),
    ):
        lines.append(f"- {label}: {source_version[key] or 'unspecified'}")
    lines.extend(("", "## Official consolidated text"))

    def walk(element: ET.Element) -> None:
        tag = _tag(element)
        if tag in MAJOR:
            lines.extend(("", f"## {tag}"))
        elif tag in LEGAL_UNIT:
            identity = _attrs(element).get("id") or _attrs(element).get("fid")
            suffix = f" `{identity}`" if identity else ""
            lines.extend(("", f"### {tag}{suffix}"))
        if tag not in STRUCTURAL:
            value = _text(element)
            if value:
                lines.extend(("", value))
            return
        for child in element:
            walk(child)

    walk(root)
    markdown = "\n".join(lines).strip() + "\n"
    return title, markdown, source_version


def _link_target(*, kind: str, link: str, lang: str) -> str:
    if kind == "act":
        segment = "fra/lois" if lang == "fr" else "eng/acts"
        return f"https://laws-lois.justice.gc.ca/{segment}/{link}/"
    if kind in {"regulation", "regulations"}:
        segment = "fra/reglements" if lang == "fr" else "eng/regulations"
        return f"https://laws-lois.justice.gc.ca/{segment}/{link}/"
    return f"justice-canada:{kind}:{link}"


def source_links(root: ET.Element, *, source_slug: str,
                 lang: str = "en") -> list[dict]:
    links: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for element in root.iter():
        tag = _tag(element)
        attrs = _attrs(element)
        link = str(attrs.get("link") or "").strip()
        if not link:
            continue
        kind = str(attrs.get("reference-type") or tag).lower()
        target = _link_target(kind=kind, link=link, lang=lang)
        context = _text(element) or link
        key = (kind, target, context)
        if key in seen:
            continue
        seen.add(key)
        links.append({
            "relation": kind, "target": target, "target_id": link,
            "context": context, "direction": "outbound",
            "article_anchor": source_slug,
        })
    return links


def legal_unit_anchors(root: ET.Element) -> list[dict]:
    """Return source-native legal unit IDs/labels without inventing anchors."""
    anchors: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for element in root.iter():
        tag = _tag(element)
        if tag not in {"Section", "Article", "Schedule", "Definition"}:
            continue
        attrs = _attrs(element)
        source_id = str(attrs.get("id") or attrs.get("fid") or "")
        label = next((_text(child) for child in element
                      if _tag(child) == "Label" and _text(child)), "")
        if not source_id and not label:
            continue
        key = (tag, source_id, label)
        if key in seen:
            continue
        seen.add(key)
        anchors.append({"kind": tag.lower(), "source_id": source_id or None,
                        "label": label or None,
                        "anchor": f"{tag.lower()}:{source_id or label}"})
    return anchors


def source_links_with_anchors(root: ET.Element, *, source_slug: str,
                              lang: str) -> list[dict]:
    """Reconstruct legal relations with their nearest official unit anchor."""
    links: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()

    def walk(element: ET.Element, current_anchor: str) -> None:
        tag = _tag(element)
        attrs = _attrs(element)
        if tag in {"Section", "Article", "Schedule", "Definition"}:
            source_id = str(attrs.get("id") or attrs.get("fid") or "")
            label = next((_text(child) for child in element
                          if _tag(child) == "Label" and _text(child)), "")
            if source_id or label:
                current_anchor = f"{tag.lower()}:{source_id or label}"
        link = str(attrs.get("link") or "").strip()
        if link:
            kind = str(attrs.get("reference-type") or tag).lower()
            target = _link_target(kind=kind, link=link, lang=lang)
            context = _text(element) or link
            key = (kind, target, context, current_anchor)
            if key not in seen:
                seen.add(key)
                links.append({
                    "relation": kind, "target": target, "target_id": link,
                    "context": context, "direction": "outbound",
                    "article_anchor": current_anchor,
                })
        for child in element:
            walk(child, current_anchor)

    walk(root, f"document:{source_slug}")
    return links


def _binding(path: Path, root: Path) -> dict:
    return ROUTEC._binding(path.resolve(), root.resolve())


def _validate_plain_binding(artifact_root: Path, artifact: dict) -> Path:
    """Validate a stage-receipt artifact without imposing body compression."""
    path = PS.artifact_path(artifact_root, artifact)
    if (not path.is_file() or path.stat().st_size != artifact.get("bytes")
            or sha256_file(path) != artifact.get("sha256")):
        raise ValueError(f"plain artifact binding failed: {path}")
    return path


def materialize(*, artifact_root: Path, routes_path: Path, l0_run_path: Path,
                l0_receipt_path: Path, archive_path: Path,
                output_root: Path) -> dict:
    artifact_root = artifact_root.resolve(); routes_path = routes_path.resolve()
    l0_run_path = l0_run_path.resolve(); l0_receipt_path = l0_receipt_path.resolve()
    archive_path = archive_path.resolve(); output_root = output_root.resolve()
    output_root.relative_to(artifact_root)
    if output_root.exists():
        raise ValueError(f"refusing existing Justice materialization: {output_root}")
    l0_run = json.loads(l0_run_path.read_text())
    l0_receipt = json.loads(l0_receipt_path.read_text())
    if (l0_run.get("commit") != COMMIT or l0_run.get("status") !=
            "source_acquired_pending_source_native_projection"):
        raise ValueError("Justice L0 RUN is not the exact pinned current source")
    archive_refs = [ref for ref in l0_receipt.get("artifacts", [])
                    if str(ref.get("uri", "")).endswith("/snapshot.tar.gz")]
    if len(archive_refs) != 1:
        raise ValueError("Justice L0 receipt lacks one exact snapshot archive")
    if _validate_plain_binding(artifact_root, archive_refs[0]).resolve() != archive_path:
        raise ValueError("Justice archive path differs from the L0 receipt")

    partial = output_root.with_name(output_root.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    paths = {
        "markdown": partial / "markdown/parents.jsonl.gz",
        "outlinks": partial / "outlinks/parent-links.jsonl.gz",
    }
    for path in paths.values(): path.parent.mkdir(parents=True, exist_ok=True)
    logical = {name: hashlib.sha256() for name in paths}; counts = {name: 0 for name in paths}
    route_points = 0; source_bytes = 0

    def write(name: str, row: dict, handle) -> None:
        value = canonical_json(row) + "\n"; handle.write(value)
        logical[name].update(value.encode()); counts[name] += 1

    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = {member.name: member for member in archive.getmembers()
                       if member.isfile()}
            prefixes = {name.split("/", 1)[0] for name in members}
            if len(prefixes) != 1:
                raise ValueError("Justice source archive root-entry drift")
            prefix = next(iter(prefixes))
            selected: list[tuple[str, int, dict, tarfile.TarInfo]] = []
            seen_urls: set[str] = set()
            with gzip.open(routes_path, "rt", encoding="utf-8") as routes:
                for line_number, line in enumerate(routes, 1):
                    route = json.loads(line)
                    if route.get("route_id") != ROUTE_ID: continue
                    url = str(route.get("url") or ""); points = route.get("production_points")
                    if url in seen_urls or not isinstance(points, int) or points < 0:
                        raise ValueError(f"invalid Justice route row at line {line_number}")
                    slug = act_slug(url); member_name = f"{prefix}/eng/acts/{slug}.xml"
                    member = members.get(member_name)
                    if member is None: raise ValueError(f"Justice source member absent: {member_name}")
                    seen_urls.add(url); route_points += points
                    selected.append((url, points, route, member))
            if len(selected) != 850 or route_points != 34197:
                raise ValueError("Justice exact Route-C membership drift")
            selected.sort(key=lambda value: value[0])
            with deterministic_gzip_text(paths["markdown"]) as parents_out, \
                    deterministic_gzip_text(paths["outlinks"]) as links_out:
                for url, points, route, member in selected:
                    stream = archive.extractfile(member)
                    if stream is None: raise ValueError(f"unreadable Justice member: {member.name}")
                    xml_bytes = stream.read(); source_bytes += len(xml_bytes)
                    xml_sha = hashlib.sha256(xml_bytes).hexdigest()
                    root = ET.fromstring(xml_bytes.lstrip(b"\xef\xbb\xbf"))
                    slug = act_slug(url)
                    title, markdown, version = render_official_markdown(
                        root, url=url, member=member.name, member_sha256=xml_sha)
                    markdown_sha = hashlib.sha256(markdown.encode()).hexdigest()
                    identity = {"member": member.name, "sha256": xml_sha,
                                "release": COMMIT}
                    links = source_links(root, source_slug=slug)
                    parent = {
                        "schema": "aw.justice-canada-law-parent.v1",
                        "document_id": hashlib.sha256(
                            f"justice-canada-law\0{url}".encode()).hexdigest(),
                        "source_record_id": slug, "source_id": SOURCE_ID,
                        "source_profile": "anchored_law", "profile": "anchored_law",
                        "url": url, "canonical_url": url, "title": title, "lang": "en",
                        "markdown": markdown, "markdown_sha256": markdown_sha,
                        "parent_source_sha256": xml_sha, "source_record_sha256": xml_sha,
                        "parent_identity": identity, "source_archive": identity,
                        "source_version": version, "source_links": links,
                        "source_links_explicit": True,
                        "source_relations_unresolved": [],
                        "source_relations_unresolved_explicit": True,
                        "anchor": f"act:{slug}",
                        "reported_production_points": points,
                        "route_terminal_receipt": route.get("terminal_receipt"),
                    }
                    write("markdown", parent, parents_out)
                    if not links:
                        links = [{"relation": None, "target": None, "target_id": None,
                                  "context": None, "direction": "outbound",
                                  "article_anchor": slug}]
                    for ordinal, link in enumerate(links):
                        write("outlinks", {
                            "schema": "aw.parent-source-link.v1", "document_id": parent["document_id"],
                            "source_record_id": slug, "source_url": url, "ordinal": ordinal,
                            "target_url": link["target"], "relation": link["relation"],
                            "target_id": link["target_id"], "anchor_or_context": link["context"],
                            "article_anchor": slug, "is_empty": not bool(link["target"]),
                            "parent_markdown_sha256": markdown_sha,
                            "parent_identity": identity, "source_version": version,
                        }, links_out)
        stages = {}
        for name, fmt in (
            ("markdown", "justice-canada-full-consolidated-act-markdown+canonical-jsonl+deterministic-gzip"),
            ("outlinks", "justice-canada-parent-bound-legal-relations+canonical-jsonl+deterministic-gzip"),
        ):
            stages[name] = SS.seal(
                artifact_root=artifact_root, artifact_paths=[paths[name]],
                receipt_path=partial / name / "receipt.json", records=counts[name],
                fmt=fmt, logical_sha256=logical[name].hexdigest(), producer_id=PRODUCER,
                producer_version=VERSION, publish_from=partial, publish_to=output_root)
        result = {
            "schema": "aw.routec-justice-canada-law-run.v1", "version": VERSION,
            "status": "complete", "source_id": SOURCE_ID, "source_profile": "anchored_law",
            "commit": COMMIT, "route_rows": _binding(routes_path, artifact_root),
            "source_l0": {"run": _binding(l0_run_path, artifact_root),
                          "original_receipt": _binding(l0_receipt_path, artifact_root),
                          "archive": archive_refs[0]},
            "point_identity": {"status": "not_present_in_route_registry",
                               "ledger_subtraction_authorized": False},
            "counts": {"route_urls": 850, "route_reported_production_points": route_points,
                       "source_parents": counts["markdown"],
                       "outlink_rows_including_empty": counts["outlinks"],
                       "selected_xml_bytes": source_bytes},
            "stages": stages,
        }
        atomic_publish(partial / "RUN.json", (json.dumps(
            result, indent=2, sort_keys=True) + "\n").encode())
        os.rename(partial, output_root); return result
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True); raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--l0-run", type=Path, required=True)
    parser.add_argument("--l0-receipt", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(
        artifact_root=args.artifact_root, routes_path=args.routes,
        l0_run_path=args.l0_run, l0_receipt_path=args.l0_receipt,
        archive_path=args.archive, output_root=args.output), indent=2, sort_keys=True))


if __name__ == "__main__": main()
