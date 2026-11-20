"""Render exact recovered JORF texts and linked article XML to full Markdown.

Moved from the frozen script of the same name by task 1437 (S7). `PRODUCER` is the label sealed
receipts carry, so it stays byte for byte; the old path is a forwarding shim.
"""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import re

from lxml import etree

from cytir.acquire.artifact_io import atomic_publish, canonical_json
from cytir.core.common import mint_passage_point_id
from cytir.extract.dila_content import content_markdown as _content_markdown
from cytir.sources.jorf.population import JORF_FULL_PARENTS
from cytir.acquire.deterministic_gzip import DeterministicGzipWriter
from cytir.sources.jorf.recover_archives import (
    _artifact, _checked_ref, _stage_receipt_from_run,
)
from cytir.acquire import stage_receipt as SS


VERSION = "jorf-exact-full-markdown.v2"
PRODUCER = "pipeline/materialize_jorf_full_markdown.py"
WS = re.compile(r"\s+")
JORFTEXT = re.compile(r"^JORFTEXT[0-9]+$")
JORFARTI = re.compile(r"^JORFARTI[0-9]+$")
JORFSCTA = re.compile(r"^JORFSCTA[0-9]+$")
LEGIARTI = re.compile(r"^LEGIARTI[0-9]+$")
LEGITEXT = re.compile(r"^LEGITEXT[0-9]+$")


def _artifact_from_stage(root: Path, run: dict, stage: str, suffix: str) -> tuple[Path, Path]:
    receipt_path, receipt = _stage_receipt_from_run(root, run, stage)
    refs = [ref for ref in receipt.get("artifacts") or []
            if str(ref.get("uri") or "").endswith(suffix)]
    if len(refs) != 1: raise ValueError(f"stage has no unique {suffix} artifact")
    return _checked_ref(root, refs[0], suffix), receipt_path


def _load_identity(root: Path, run_path: Path) -> tuple[dict[str, dict], list[Path]]:
    run = json.loads(run_path.read_text())
    if run.get("schema") != "aw.jorf-production-membership-frontier-run.v1":
        raise ValueError("JORF identity RUN schema drift")
    path, receipt = _artifact_from_stage(
        root, run, "identity", "/frontier/jorftext-source-ids.jsonl.gz")
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line); target = row["jorftext_id"]
            if target in rows: raise ValueError("duplicate JORF identity row")
            rows[target] = row
    if len(rows) != JORF_FULL_PARENTS: raise ValueError("JORF identity population drift")
    return rows, [run_path, receipt, path]


def _load_text_records(root: Path, run_path: Path) -> tuple[dict[str, dict], list[Path]]:
    run = json.loads(run_path.read_text())
    if (run.get("schema") != "aw.jorf-exact-archive-recovery-run.v1"
            or run.get("status") != "source-records-complete-full-content-pending"
            or (run.get("counts") or {}).get("unresolved") != 0):
        raise ValueError("JORFTEXT recovery is not zero-miss terminal")
    path, receipt = _artifact_from_stage(
        root, run, "recovery", "/recovered/jorftext-source-records.jsonl.gz")
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line); rows[row["jorftext_id"]] = row
    if len(rows) != JORF_FULL_PARENTS: raise ValueError("recovered JORFTEXT population drift")
    return rows, [run_path, receipt, path]


def _load_linked_records(root: Path, run_path: Path) -> tuple[dict[str, dict], list[Path]]:
    run = json.loads(run_path.read_text())
    if (run.get("schema") != "aw.jorf-linked-source-record-recovery-run.v1"
            or run.get("status") != "linked-source-records-complete-markdown-pending"
            or (run.get("counts") or {}).get("unresolved_relations") != 0):
        raise ValueError("linked JORF source recovery is not zero-miss terminal")
    path, receipt = _artifact_from_stage(
        root, run, "recovery", "/recovered/linked-source-records.jsonl.gz")
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line); source_id = row["source_id"]
            if source_id in rows: raise ValueError("duplicate linked source record")
            rows[source_id] = row
    if len(rows) != (run.get("counts") or {}).get("reachable_source_records"):
        raise ValueError("linked source-record count drift")
    return rows, [run_path, receipt, path]


def _decode_record(row: dict, expected_sha: str | None = None) -> bytes:
    record = row.get("record") or row
    payload = base64.b64decode(record["xml_base64"], validate=True)
    if (len(payload) != record["xml_bytes"]
            or hashlib.sha256(payload).hexdigest() != record["xml_sha256"]):
        raise ValueError("source XML bytes/SHA drift")
    if expected_sha is not None and record["xml_sha256"] != expected_sha:
        raise ValueError("source XML expected SHA drift")
    return payload


def _norm(value: str) -> str:
    return WS.sub(" ", value or "").strip()


def _iter_text(node) -> str:
    return _norm("".join(node.itertext())) if node is not None else ""


def _target_url(identifier: str) -> str:
    if JORFTEXT.fullmatch(identifier):
        return f"https://www.legifrance.gouv.fr/jorf/id/{identifier}"
    if JORFARTI.fullmatch(identifier):
        return f"https://www.legifrance.gouv.fr/jorf/article_jo/{identifier}"
    if LEGIARTI.fullmatch(identifier):
        return f"https://www.legifrance.gouv.fr/codes/article_lc/{identifier}"
    if LEGITEXT.fullmatch(identifier):
        return f"https://www.legifrance.gouv.fr/codes/id/{identifier}"
    return ""


def _links(root, *, anchor: str) -> tuple[list[dict], list[dict]]:
    links, unresolved = [], []
    for node in root.findall(".//LIENS/LIEN"):
        target_id = str(node.get("id") or node.get("cidtexte") or "").strip()
        target = _target_url(target_id); context = _iter_text(node)
        row = {"relation": str(node.get("typelien") or "legal-reference").casefold(),
               "direction": str(node.get("sens") or ""), "target": target,
               "target_id": target_id, "context": context, "article_anchor": anchor}
        if target and context: links.append(row)
        else:
            row["unresolved_reasons"] = [reason for missing, reason in (
                (not target, "target_absent"), (not context, "context_absent")) if missing]
            unresolved.append(row)
    return links, unresolved


def _structural_roots(struct_root) -> list[str]:
    values = []
    structure = struct_root.find("./STRUCT")
    if structure is None: return values
    for node in structure:
        source_id = str(node.get("id") or "")
        if JORFARTI.fullmatch(source_id) or JORFSCTA.fullmatch(source_id):
            values.append(source_id)
    return values


def _render_document(target: str, identity: dict, text_row: dict,
                     linked: dict[str, dict]) -> tuple[dict, dict, dict]:
    version_bytes = _decode_record(text_row["records"]["version"])
    struct_bytes = _decode_record(text_row["records"]["struct"])
    version = etree.fromstring(version_bytes); struct = etree.fromstring(struct_bytes)
    if ((version.findtext(".//META_COMMUN/ID") or "").strip() != target
            or (struct.findtext(".//META_COMMUN/ID") or "").strip() != target):
        raise ValueError("JORFTEXT XML identity drift during materialization")
    title = _norm(version.findtext(".//META_TEXTE_VERSION/TITREFULL") or
                  version.findtext(".//META_TEXTE_VERSION/TITRE") or identity.get("title") or "")
    if not title: raise ValueError(f"JORF source title absent: {target}")
    nature = _norm(version.findtext(".//META_COMMUN/NATURE") or "")
    published = _norm(version.findtext(".//META_TEXTE_CHRONICLE/DATE_PUBLI") or "")
    signed = _norm(version.findtext(".//META_TEXTE_CHRONICLE/DATE_TEXTE") or "")
    nor = _norm(version.findtext(".//META_TEXTE_CHRONICLE/NOR") or "")
    publication = _norm(version.findtext(".//META_TEXTE_CHRONICLE/ORIGINE_PUBLI") or "")
    markdown = [f"# {title}", "", "## Official text identity",
                f"- JORFTEXT ID: `{target}`", f"- Nature: {nature or 'unspecified'}",
                f"- Publication date: {published or 'unspecified'}",
                f"- Signature date: {signed or 'unspecified'}",
                f"- NOR: {nor or 'unspecified'}",
                f"- Official publication: {publication or 'unspecified'}"]
    body_sections = 0
    for xml_name, heading in (("NOTICE", "Notice"), ("VISAS", "Visas"),
                              ("SIGNATAIRES", "Signatories"),
                              ("TP", "Preliminary provisions"),
                              ("ABRO", "Abrogation notes"),
                              ("RECT", "Corrections"), ("SM", "Summary")):
        value = _content_markdown(version.find(f"./{xml_name}/CONTENU"))
        if value:
            markdown.extend(("", f"## {heading}", "", value)); body_sections += 1
    source_records = [{"source_id": target, "record_kind": "version",
                       "xml_sha256": text_row["records"]["version"]["xml_sha256"],
                       "archive_uri": text_row["records"]["version"]["archive_uri"],
                       "member_name": text_row["records"]["version"]["member_name"]},
                      {"source_id": target, "record_kind": "struct",
                       "xml_sha256": text_row["records"]["struct"]["xml_sha256"],
                       "archive_uri": text_row["records"]["struct"]["archive_uri"],
                       "member_name": text_row["records"]["struct"]["member_name"]}]
    all_links, all_unresolved = _links(version, anchor=target)
    visited = set(); article_bodies = empty_articles = 0

    def render(source_id: str, level: int) -> None:
        nonlocal article_bodies, empty_articles, body_sections
        if source_id in visited: return
        visited.add(source_id); row = linked.get(source_id)
        if row is None or row["jorftext_id"] != target:
            raise ValueError(f"linked record reachability drift: {target} -> {source_id}")
        payload = _decode_record(row); root = etree.fromstring(payload)
        record = row["record"]
        source_records.append({"source_id": source_id, "record_kind": row["record_kind"],
            "xml_sha256": record["xml_sha256"], "archive_uri": record["archive_uri"],
            "member_name": record["member_name"]})
        if row["record_kind"] == "section":
            heading = _norm(root.findtext("./TITRE_TA") or "Section")
            markdown.extend(("", f"{'#' * min(level, 6)} {heading} {{#{source_id}}}"))
            body_sections += 1
            for child in row.get("children") or []: render(child, level + 1)
            return
        number = _norm(root.findtext(".//META_ARTICLE/NUM") or "")
        body = _content_markdown(root.find(".//BLOC_TEXTUEL/CONTENU"))
        heading = f"Article {number}" if number else "Article"
        markdown.extend(("", f"{'#' * min(level, 6)} {heading} {{#{source_id}}}"))
        if body:
            markdown.extend(("", body)); article_bodies += 1; body_sections += 1
        else:
            markdown.extend(("", "_The official source record contains no article body text._"))
            empty_articles += 1
        links, unresolved = _links(root, anchor=source_id)
        all_links.extend(links); all_unresolved.extend(unresolved)

    for source_id in _structural_roots(struct): render(source_id, 2)
    text = "\n".join(markdown).strip()
    url = f"https://www.legifrance.gouv.fr/jorf/id/{target}"
    parent_id = mint_passage_point_id(url, f"document:{target}", text)
    parent_identity = {
        "source_id": target,
        "version": {key: text_row["records"]["version"][key] for key in
                    ("archive_uri", "member_name", "xml_sha256")},
        "struct": {key: text_row["records"]["struct"][key] for key in
                   ("archive_uri", "member_name", "xml_sha256")},
    }
    source_version = {"nature": nature, "published": published, "signed": signed,
                      "nor": nor, "official_publication": publication}
    row = {"schema": "aw.l1-markdown.v1", "profile": "anchored_law",
        "source_id": "dila-jorf-exact-production-v1", "source_record_id": target,
        "url": url, "canonical_url": url, "title": title, "lang": "fr",
        "jurisdiction": "FR", "point_id": parent_id,
        "baseline_production_point_ids": [identity["point_id"]],
        "anchor": f"document:{target}", "markdown": text,
        "markdown_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "parent_identity": parent_identity, "source_version": source_version,
        "source_records": source_records,
        "source_record_count": len(source_records), "article_bodies": article_bodies,
        "empty_article_records": empty_articles, "body_sections": body_sections,
        "source_links": all_links, "source_links_explicit": True,
        "source_relations_unresolved": all_unresolved,
        "source_relations_unresolved_explicit": True}
    outlinks = {"schema": "aw.parent-outlinks.v1", "parent_id": parent_id,
        "source_record_id": target, "url": url, "links": all_links,
        "unresolved": all_unresolved, "explicit_empty": not all_links and not all_unresolved}
    counts = {"article_bodies": article_bodies, "empty_article_records": empty_articles,
              "body_sections": body_sections, "links": len(all_links),
              "unresolved_links": len(all_unresolved), "source_records": len(source_records)}
    return row, outlinks, counts


def materialize(*, artifact_root: Path, identity_run: Path, text_recovery_run: Path,
                linked_recovery_run: Path, output: Path) -> dict:
    root = artifact_root.resolve(); output = output.resolve(); output.relative_to(root)
    if output.exists(): raise ValueError(f"immutable JORF materialization exists: {output}")
    partial = output.with_name(f"{output.name}.partial.{os.getpid()}")
    identity, identity_evidence = _load_identity(root, identity_run.resolve())
    texts, text_evidence = _load_text_records(root, text_recovery_run.resolve())
    linked, linked_evidence = _load_linked_records(root, linked_recovery_run.resolve())
    if set(identity) != set(texts): raise ValueError("identity/text recovery population mismatch")
    partial.mkdir(parents=True, exist_ok=False)
    markdown_path = partial / "markdown/parents.jsonl.gz"
    outlinks_path = partial / "outlinks/parents.jsonl.gz"
    markdown_writer = DeterministicGzipWriter(markdown_path)
    outlinks_writer = DeterministicGzipWriter(outlinks_path)
    totals = {"documents": 0, "article_bodies": 0, "empty_article_records": 0,
              "body_sections": 0, "links": 0, "unresolved_links": 0,
              "source_records": 0, "metadata_only_documents": 0,
              "explicit_empty_outlink_parents": 0}
    try:
        for target in sorted(identity):
            row, outlinks, counts = _render_document(target, identity[target], texts[target], linked)
            markdown_writer.write(row); outlinks_writer.write(outlinks)
            totals["documents"] += 1
            for key, value in counts.items(): totals[key] += value
            if counts["body_sections"] == 0: totals["metadata_only_documents"] += 1
            if outlinks["explicit_empty"]: totals["explicit_empty_outlink_parents"] += 1
    finally:
        markdown_writer.close(); outlinks_writer.close()
    evidence = [*identity_evidence, *text_evidence, *linked_evidence]
    artifacts = [markdown_path, outlinks_path, *evidence]
    logical = hashlib.sha256((canonical_json(totals) + "\n").encode()).hexdigest()
    stage = SS.seal(artifact_root=root, artifact_paths=artifacts,
        receipt_path=partial / "stages/materialization/receipt.json",
        records=totals["documents"],
        fmt="full-source-native-jorf-markdown+parent-bound-outlinks+exact-source-record-lineage",
        logical_sha256=logical, producer_id=PRODUCER, producer_version=VERSION,
        publish_from=partial, publish_to=output, allow_existing_published=True)
    run = {"schema": "aw.jorf-exact-full-markdown-run.v1", "version": VERSION,
        "status": "materialization-complete-profile-fidelity-chunking-pending",
        "counts": totals, "identity_run": _artifact(identity_run.resolve(), root),
        "text_recovery_run": _artifact(text_recovery_run.resolve(), root),
        "linked_recovery_run": _artifact(linked_recovery_run.resolve(), root),
        "stages": {"materialization": stage},
        "transport_limitation": "historical global/increment HTTP status, headers, requested/final URLs, and fetch timestamps were not retained; exact archive bytes are bound",
        "admission": "full source-native materialization only; profile/fidelity/token/chunk/vector gates remain pending and no production points are subtracted"}
    atomic_publish(partial / "RUN.json",
                   (json.dumps(run, indent=2, sort_keys=True) + "\n").encode())
    os.rename(partial, output); return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--identity-run", type=Path, required=True)
    parser.add_argument("--text-recovery-run", type=Path, required=True)
    parser.add_argument("--linked-recovery-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(artifact_root=args.artifact_root,
        identity_run=args.identity_run, text_recovery_run=args.text_recovery_run,
        linked_recovery_run=args.linked_recovery_run, output=args.output),
        indent=2, sort_keys=True))


if __name__ == "__main__": main()
