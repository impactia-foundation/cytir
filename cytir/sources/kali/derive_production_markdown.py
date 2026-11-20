"""Project exact current-production KALI IDs from sealed DILA XML to Markdown.

The production membership is the identity witness.  Only rows explicitly
labelled ``kali_article`` or ``kali_conv`` are selected; the shared
Legifrance domain is never used as a source-family proxy.  Every selected
source XML member is preserved by its sealed archive identity and rendered in
full, without the legacy article-length or per-convention caps.
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

from lxml import etree

from cytir.acquire.artifact_io import canonical_json, sha256_file
from cytir.core.common import mint_passage_point_id
from cytir.extract.dila_content import content_markdown
from cytir.acquire import stage_receipt as SS


VERSION = "kali-exact-production-anchored-markdown.v1"
ARTICLE_PREFIX = "https://www.legifrance.gouv.fr/conv_coll/article/"
CONTAINER_PREFIX = "https://www.legifrance.gouv.fr/conv_coll/id/"
ARTICLE_ID = re.compile(r"^KALIARTI[0-9]+$")
CONTAINER_ID = re.compile(r"^KALICONT[0-9]+$")
WS = re.compile(r"\s+")
def _target(identifier: str) -> str:
    if ARTICLE_ID.fullmatch(identifier):
        return ARTICLE_PREFIX + identifier
    if identifier.startswith(("KALICONT", "KALITEXT")):
        return CONTAINER_PREFIX + identifier
    return ""


def _links(root, *, source_version: dict, anchor: str) -> tuple[list[dict], list[dict]]:
    links, unresolved = [], []
    for node in root.findall(".//LIENS/LIEN"):
        identifier = str(node.get("id") or node.get("cidtexte") or "").strip()
        context = WS.sub(" ", "".join(node.itertext())).strip()
        relation = {
            "relation": str(node.get("typelien") or "legal-reference").casefold(),
            "direction": str(node.get("sens") or ""),
            "target": _target(identifier), "target_id": identifier,
            "context": context, "source_version": source_version,
            "article_anchor": anchor,
        }
        if relation["target"] and context:
            links.append(relation)
        else:
            relation["unresolved_reasons"] = [
                reason for missing, reason in (
                    (not relation["target"], "target_absent"),
                    (not context, "context_absent")) if missing]
            unresolved.append(relation)
    return links, unresolved


def parse_article(body: bytes, *, member: str, archive_sha256: str,
                  baseline_point_ids: list[str], baseline_titles: list[str],
                  source_release: str = "2025-07-13-global") -> dict | None:
    root = etree.fromstring(body)
    identifier = (root.findtext(".//META_COMMUN/ID") or "").strip()
    if not ARTICLE_ID.fullmatch(identifier):
        return None
    number = (root.findtext(".//META_ARTICLE/NUM") or "").strip()
    state = (root.findtext(".//META_ARTICLE/ETAT") or "").strip()
    start = (root.findtext(".//META_ARTICLE/DATE_DEBUT") or "").strip()
    end = (root.findtext(".//META_ARTICLE/DATE_FIN") or "").strip()
    kind = (root.findtext(".//META_ARTICLE/TYPE") or "").strip()
    body_markdown = content_markdown(root.find(".//BLOC_TEXTUEL/CONTENU"))
    if not body_markdown:
        return None
    titles = []
    for node in root.findall(".//CONTEXTE/TEXTE/TITRE_TXT"):
        value = WS.sub(" ", "".join(node.itertext())).strip()
        if value and value not in titles:
            titles.append(value)
    title = titles[-1] if titles else "Convention collective"
    container = root.find(".//CONTEXTE/CONTENEUR")
    container_id = str(container.get("cid") or "").strip() if container is not None else ""
    hierarchy = []
    for node in root.findall(".//CONTEXTE/TEXTE//TITRE_TM"):
        value = WS.sub(" ", "".join(node.itertext())).strip()
        if value:
            hierarchy.append({"id": str(node.get("id") or ""), "title": value,
                              "start": str(node.get("debut") or ""),
                              "end": str(node.get("fin") or "")})
    version = {"status": state, "effective_from": start,
               "effective_until": end, "source_release": source_release}
    heading = f"Article {number}" if number else "Article"
    markdown = [f"# {title}", "", "## Article identity",
                f"- Article ID: `{identifier}`", f"- Article number: {number or 'unspecified'}",
                f"- Status: {state or 'unspecified'}", f"- Effective from: {start or 'unspecified'}",
                f"- Effective until: {end or 'unspecified'}", f"- Type: {kind or 'unspecified'}",
                f"- Convention container: `{container_id or 'unspecified'}`",
                f"- Archive member: `{member}`", f"- Archive SHA-256: `{archive_sha256}`",
                "", "## Legal hierarchy"]
    if hierarchy:
        markdown.extend(f"- {row['title']} (`{row['id'] or 'no-section-id'}`)"
                        for row in hierarchy)
    else:
        markdown.append("- No intermediate section in the official article context")
    paragraphs = [value.strip() for value in re.split(r"\n\s*\n|\n", body_markdown)
                  if value.strip()]
    markdown.extend(("", f"## {heading} {{#{identifier}}}"))
    for ordinal, paragraph in enumerate(paragraphs, 1):
        markdown.extend(("", f"### Paragraph {ordinal} {{#{identifier}-p{ordinal}}}",
                         "", paragraph))
    text = "\n".join(markdown).strip()
    links, unresolved = _links(root, source_version=version, anchor=identifier)
    url = ARTICLE_PREFIX + identifier
    return {
        "schema": "aw.l1-markdown.v1", "profile": "anchored_law",
        "source_id": "dila-kali-exact-production-v1",
        "source_record_id": identifier, "url": url, "canonical_url": url,
        "title": f"{title} — {heading}", "lang": "fr", "jurisdiction": "FR",
        "point_id": mint_passage_point_id(url, f"article:{identifier}", text),
        "baseline_production_point_ids": sorted(baseline_point_ids),
        "baseline_production_titles": sorted(set(baseline_titles)),
        "anchor": f"article:{identifier}", "markdown": text,
        "markdown_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_record_sha256": hashlib.sha256(body).hexdigest(),
        "source_archive": {"member": member, "sha256": archive_sha256,
                           "release": source_release},
        "source_version": version, "container_id": container_id,
        "legal_hierarchy": hierarchy, "article_number": number,
        "paragraph_anchors": [f"{identifier}-p{i}" for i in range(1, len(paragraphs) + 1)],
        "source_links": links, "source_links_explicit": True,
        "source_relations_unresolved": unresolved,
        "source_relations_unresolved_explicit": True,
    }


def parse_container(body: bytes, *, member: str, archive_sha256: str,
                    baseline_point_ids: list[str], baseline_titles: list[str],
                    source_release: str = "2025-07-13-global") -> dict | None:
    root = etree.fromstring(body)
    identifier = (root.findtext(".//META_COMMUN/ID") or "").strip()
    title = (root.findtext(".//META_CONTENEUR/TITRE") or "").strip()
    if not CONTAINER_ID.fullmatch(identifier) or not title:
        return None
    state = (root.findtext(".//META_CONTENEUR/ETAT") or "").strip()
    number = (root.findtext(".//META_CONTENEUR/NUM") or "").strip()
    published = (root.findtext(".//META_CONTENEUR/DATE_PUBLI") or "").strip()
    activities = [WS.sub(" ", "".join(node.itertext())).strip()
                  for node in root.findall(".//ACTS_PRO/ACT_PRO")]
    brochures = [WS.sub(" ", "".join(node.itertext())).strip()
                 for node in root.findall(".//NUMS_BROCH/NUM_BROCH")]
    sections, links = [], []
    for section in root.findall(".//STRUCTURE_TXT/TM"):
        heading = (section.findtext("./TITRE_TM") or "Section").strip()
        section_links = []
        for node in section.findall(".//LIEN_TXT"):
            target_id = str(node.get("idtxt") or "").strip()
            context = str(node.get("titretxt") or "").strip()
            relation = {"relation": "contains-text", "direction": "outbound",
                        "target": _target(target_id), "target_id": target_id,
                        "context": context, "article_anchor": identifier,
                        "source_version": {"status": state,
                                           "published": published,
                                           "source_release": source_release}}
            if relation["target"] and context:
                links.append(relation)
            section_links.append({"id": target_id, "title": context})
        sections.append({"heading": heading, "links": section_links})
    markdown = [f"# {title}", "", "## Convention identity",
                f"- Container ID: `{identifier}`", f"- IDCC: {number or 'unspecified'}",
                f"- Status: {state or 'unspecified'}", f"- Published: {published or 'unspecified'}",
                f"- Archive member: `{member}`", f"- Archive SHA-256: `{archive_sha256}`"]
    if activities:
        markdown.extend(("", "## Covered activities", *[f"- {x}" for x in activities]))
    if brochures:
        markdown.extend(("", "## Brochure numbers", *[f"- {x}" for x in brochures]))
    markdown.extend(("", "## Official structure"))
    for section in sections:
        markdown.extend(("", f"### {section['heading']}"))
        markdown.extend(f"- {row['title'] or row['id']} (`{row['id'] or 'unspecified'}`)"
                        for row in section["links"])
    text = "\n".join(markdown).strip()
    url = CONTAINER_PREFIX + identifier
    version = {"status": state, "published": published,
               "source_release": source_release}
    return {
        "schema": "aw.l1-markdown.v1", "profile": "anchored_law",
        "source_id": "dila-kali-exact-production-v1",
        "source_record_id": identifier, "url": url, "canonical_url": url,
        "title": title, "lang": "fr", "jurisdiction": "FR",
        "point_id": mint_passage_point_id(url, f"container:{identifier}", text),
        "baseline_production_point_ids": sorted(baseline_point_ids),
        "baseline_production_titles": sorted(set(baseline_titles)),
        "anchor": f"container:{identifier}", "markdown": text,
        "markdown_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_record_sha256": hashlib.sha256(body).hexdigest(),
        "source_archive": {"member": member, "sha256": archive_sha256,
                           "release": source_release},
        "source_version": version, "idcc": number, "covered_activities": activities,
        "brochure_numbers": brochures, "legal_hierarchy": sections,
        "source_links": links, "source_links_explicit": True,
        "source_relations_unresolved": [],
        "source_relations_unresolved_explicit": True,
    }


def _membership(membership: Path, expected_articles: int,
                expected_containers: int) -> tuple[dict[str, dict], dict]:
    receipt_path = membership / "RECEIPT.json"
    body = receipt_path.read_bytes(); receipt = json.loads(body)
    wanted: dict[str, dict] = {}; counts = {"kali_article": 0, "kali_conv": 0}
    scanned = 0
    for shard in receipt.get("shards") or []:
        path = membership / shard["file"]
        if path.stat().st_size != shard["bytes"]:
            raise ValueError(f"membership shard size drift: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                scanned += 1; row = json.loads(line); payload = row.get("payload") or {}
                provenance = payload.get("provenance")
                if provenance not in counts:
                    continue
                url = str(payload.get("url") or "")
                prefix = ARTICLE_PREFIX if provenance == "kali_article" else CONTAINER_PREFIX
                identifier = url.removeprefix(prefix).strip("/") if url.startswith(prefix) else ""
                pattern = ARTICLE_ID if provenance == "kali_article" else CONTAINER_ID
                if not pattern.fullmatch(identifier):
                    raise ValueError(f"invalid {provenance} production URL: {url}")
                item = wanted.setdefault(identifier, {"url": url, "provenance": provenance,
                                                       "point_ids": [], "titles": []})
                if item["url"] != url or item["provenance"] != provenance:
                    raise ValueError(f"production identity conflict: {identifier}")
                item["point_ids"].append(
                    _membership_point_id(row, payload, identifier))
                item["titles"].append(str(payload.get("title") or ""))
                counts[provenance] += 1
    if scanned != receipt["rows"]:
        raise ValueError("membership row conservation failed")
    if counts != {"kali_article": expected_articles, "kali_conv": expected_containers}:
        raise ValueError(f"KALI production count drift: {counts}")
    return wanted, {"uri": f"artifact://{receipt_path.relative_to(membership.parents[1]).as_posix()}",
                    "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                    "rows_scanned": scanned, "selected_points": sum(counts.values()),
                    "selected_source_ids": len(wanted), "by_provenance": counts}


def _write_row(handle, logical, row: dict) -> None:
    encoded = (canonical_json(row) + "\n").encode()
    handle.write(encoded); logical.update(encoded)


def _membership_point_id(row: dict, payload: dict, identifier: str) -> str:
    point_id = str(row.get("id") or row.get("point_id") or
                   payload.get("point_id") or "")
    mirrored = {str(value) for value in (
        row.get("id"), row.get("point_id"), payload.get("point_id"))
                if value is not None}
    if not point_id or mirrored != {point_id}:
        raise ValueError(f"production point identity conflict: {identifier}")
    return point_id


def derive(*, artifact_root: Path, membership: Path, l0_receipt: Path,
           output: Path, expected_articles: int, expected_containers: int) -> dict:
    artifact_root = artifact_root.resolve(); output = output.resolve()
    output.relative_to(artifact_root)
    if output.exists():
        raise ValueError(f"immutable output exists: {output}")
    wanted, membership_ref = _membership(
        membership.resolve(), expected_articles, expected_containers)
    l0_body = l0_receipt.read_bytes(); l0 = json.loads(l0_body)
    archive_ref = next(row for row in l0["artifacts"] if row["uri"].endswith(".tar.gz"))
    archive = artifact_root / archive_ref["uri"].removeprefix("artifact://")
    if archive.stat().st_size != archive_ref["bytes"] or sha256_file(archive) != archive_ref["sha256"]:
        raise ValueError("sealed KALI archive binding mismatch")
    partial = output.with_name(output.name + f".partial.{os.getpid()}")
    partial.mkdir(parents=True)
    try:
        markdown_path = partial / "markdown.jsonl.gz"
        outlinks_path = partial / "outlinks.jsonl.gz"
        markdown_logical = hashlib.sha256(); outlinks_logical = hashlib.sha256()
        seen = {}; records = article_records = container_records = 0
        links = unresolved = empty = 0
        with markdown_path.open("wb") as mraw, outlinks_path.open("wb") as oraw:
            with deterministic_gzip_open(mraw, compresslevel=9) as markdown, \
                    deterministic_gzip_open(oraw, compresslevel=9) as outlinks, \
                    tarfile.open(archive, "r|gz") as tar:
                for member in tar:
                    identifier = Path(member.name).stem
                    if not member.isfile() or identifier not in wanted:
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise ValueError(f"cannot extract {member.name}")
                    xml = extracted.read(); digest = hashlib.sha256(xml).hexdigest()
                    if identifier in seen:
                        if seen[identifier] != digest:
                            raise ValueError(f"divergent immutable KALI ID: {identifier}")
                        continue
                    evidence = wanted[identifier]
                    parser = parse_article if evidence["provenance"] == "kali_article" else parse_container
                    row = parser(xml, member=member.name, archive_sha256=archive_ref["sha256"],
                                 baseline_point_ids=evidence["point_ids"],
                                 baseline_titles=evidence["titles"])
                    if row is None or row["url"] != evidence["url"]:
                        raise ValueError(f"KALI source/member identity mismatch: {identifier}")
                    seen[identifier] = digest; _write_row(markdown, markdown_logical, row)
                    link_row = {"schema": "aw.parent-source-links.v1",
                                "source_id": row["source_id"],
                                "source_record_id": identifier, "url": row["url"],
                                "article_anchor": row["anchor"],
                                "source_version": row["source_version"],
                                "parent_markdown_sha256": row["markdown_sha256"],
                                "parent_identity": row["source_archive"],
                                "links": row["source_links"],
                                "unresolved_relations": row["source_relations_unresolved"]}
                    _write_row(outlinks, outlinks_logical, link_row)
                    links += len(row["source_links"])
                    unresolved += len(row["source_relations_unresolved"])
                    empty += not row["source_links"]
                    records += 1
                    article_records += evidence["provenance"] == "kali_article"
                    container_records += evidence["provenance"] == "kali_conv"
                    if records % 20_000 == 0:
                        print(f"[kali] records={records:,}/{len(wanted):,}", flush=True)
        missing = sorted(set(wanted) - set(seen))
        missing_path = partial / "missing-production-ids.json"
        missing_data = (json.dumps({"schema": "aw.kali-production-recovery-frontier.v1",
                                    "wanted_source_ids": len(wanted),
                                    "missing_source_ids": missing},
                                   indent=2, sort_keys=True) + "\n").encode()
        missing_path.write_bytes(missing_data)
        markdown_stage = SS.seal(
            artifact_root=artifact_root, artifact_paths=[markdown_path, missing_path],
            receipt_path=partial / "MARKDOWN-RECEIPT.json", records=records,
            fmt="anchored-law-full-kali-markdown+deterministic-gzip+explicit-recovery-frontier",
            logical_sha256=markdown_logical.hexdigest(),
            producer_id="pipeline/derive_kali_production_markdown.py",
            producer_version=VERSION, publish_from=partial, publish_to=output)
        outlink_stage = SS.seal(
            artifact_root=artifact_root, artifact_paths=[outlinks_path],
            receipt_path=partial / "OUTLINKS-RECEIPT.json", records=records,
            fmt="parent-bound-kali-legal-relations+deterministic-gzip",
            logical_sha256=outlinks_logical.hexdigest(),
            producer_id="pipeline/derive_kali_production_markdown.py",
            producer_version=VERSION, publish_from=partial, publish_to=output)
        run = {"schema": "aw.kali-exact-production-run.v1", "version": VERSION,
               "status": "complete" if not missing else "recovery-frontier",
               "source_profile": "anchored_law",
               "counts": {"production_points": membership_ref["selected_points"],
                          "production_source_ids": len(wanted), "markdown_documents": records,
                          "articles": article_records, "containers": container_records,
                          "links": links, "unresolved_relations": unresolved,
                          "explicit_empty_outlink_parents": empty,
                          "missing_source_ids": len(missing)},
               "l0": {"uri": f"artifact://{l0_receipt.resolve().relative_to(artifact_root).as_posix()}",
                      "bytes": len(l0_body), "sha256": hashlib.sha256(l0_body).hexdigest(),
                      "archive": archive_ref},
               "membership": membership_ref,
               "stages": {"markdown": markdown_stage, "outlinks": outlink_stage},
               "next_gate": "zero missing IDs, anchored-law profile/fidelity audit, token-safe source-record-local chunks"}
        (partial / "RUN.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
        os.rename(partial, output)
        return run
    except BaseException:
        failed = output.with_name(output.name + f".failed.{os.getpid()}")
        if partial.exists():
            os.rename(partial, failed)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--membership-dir", type=Path, required=True)
    parser.add_argument("--l0-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-articles", type=int, required=True)
    parser.add_argument("--expected-containers", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(artifact_root=args.artifact_root,
                            membership=args.membership_dir,
                            l0_receipt=args.l0_receipt, output=args.output,
                            expected_articles=args.expected_articles,
                            expected_containers=args.expected_containers),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
