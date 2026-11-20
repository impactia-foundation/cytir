"""DILA Service-Public fiche XML -> one Markdown document (`service_public_fiche`).

Moved out of the restage tool (task 696) so the source family
(`cytir.sources.families.service_public_fiche`) and the restage tool share one renderer. The rendering
of the body is UNCHANGED: headings from the nested `Titre` / `Chapitre` / `Section` levels, list items
as `- `, every principal XML text block conserved (a missing block raises).

`provenance_head=True` (the restage tool, the sealed stages) writes the YAML frontmatter with the
`aw_*` lineage fields before the title. `provenance_head=False` (the family) writes none: the
declared surface of the family is the fiche's own title and body only, so no bookkeeping of ours is
embedded or searched (see the family's declaration, `representation`).

`restage` re-renders a sealed DILA run's fiches as v4 Markdown stages (moved from the former restage tool, 1261).
The legacy vosdroits Passage-JSONL CLI (ported from the former `ingest_vosdroits` script, 1235) lives
at the end of the module as `main` (`python -m` no longer starts it, task 2943; the family runs as `cyt source run service_public_fiche`).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import urllib.parse
import zipfile

from lxml import etree

from cytir.acquire import preserved_stage_io as PS
from cytir.acquire import stage_receipt as SS
from cytir.acquire.artifact_io import atomic_publish, canonical_json
from cytir.chunk.generic import chunk_generic_records
from cytir.core.common import Passage, low_quality, passage_corpus_record, passage_from_chunk
from cytir.index.preservation_contract import _validate_stage

RESTAGE_VERSION = "service-public-fiche-markdown.v4"
RESTAGE_PRODUCER = "aw/extract/service_public_fiche.py#restage"

DC = "{http://purl.org/dc/elements/1.1/}"
CONTENT = {"Introduction", "Texte", "ListeSituations", "CommentFaireSi",
           "Questionnaire", "Problematiques", "Avertissement"}


def _local(element) -> str:
    return etree.QName(element).localname


def _text(element) -> str:
    return " ".join((element.xpath("string()") or "").split())


def _inside(element, name: str) -> bool:
    return any(_local(value) == name for value in element.iterancestors())


def _heading_level(element) -> int:
    depth = sum(_local(value) in {"Chapitre", "SousChapitre", "Section"}
                for value in element.iterancestors())
    return min(2 + depth, 6)


def render(xml_bytes: bytes, archive_sha256: str, member: str, *,
           provenance_head: bool = True) -> dict | None:
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError:
        return None
    if _local(root) != "Publication":
        return None
    url = str(root.get("spUrl") or "").strip()
    title = str(root.findtext(DC + "title") or "").strip()
    identifier = str(root.findtext(DC + "identifier") or root.get("ID") or "").strip()
    date_raw = str(root.findtext(DC + "date") or "").strip()
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", date_raw)
    if not url.startswith("http") or not title or not identifier or not date_match:
        return None
    modified_date = date_match.group(0)
    blocks: list[str] = []
    baseline: list[str] = []
    headings = 1
    for child in root:
        if _local(child) not in CONTENT:
            continue
        candidates = list(child.iter("Titre", "Paragraphe"))
        if not candidates:
            candidates = [child]
        for value in candidates:
            text = _text(value)
            if not text:
                continue
            if not baseline or baseline[-1] != text:
                baseline.append(text)
            tag = _local(value)
            if tag == "Paragraphe" and _inside(value, "Titre"):
                continue
            if tag == "Titre":
                rendered = "#" * _heading_level(value) + " " + text
                headings += 1
            elif tag == "Paragraphe" and _inside(value, "Item"):
                rendered = "- " + text
            else:
                rendered = text
            if not blocks or blocks[-1] != rendered:
                blocks.append(rendered)
    if not blocks:
        return None
    subject = str(root.findtext(DC + "subject") or "").strip()
    fiche_type = str(root.findtext(DC + "type") or root.get("type") or "").strip()
    member_sha = hashlib.sha256(xml_bytes).hexdigest()
    metadata = [
        "---", f'aw_fiche_id: "{identifier}"',
        f'aw_archive_member: "{member}"', f"aw_archive_sha256: {archive_sha256}",
        f"aw_member_sha256: {member_sha}", f"aw_modified_date: {modified_date}",
        f'aw_fiche_type: {json.dumps(fiche_type, ensure_ascii=False)}',
        f'aw_subject: {json.dumps(subject, ensure_ascii=False)}', "---", "",
    ]
    head = "\n".join(metadata) if provenance_head else ""
    markdown = head + f"# {title}\n\n" + "\n\n".join(blocks) + "\n"
    # Every principal block from the source-specific baseline must survive in L1.
    normalized_markdown = " ".join(markdown.split())
    if any(" ".join(value.split()) not in normalized_markdown for value in baseline):
        raise ValueError("DILA principal XML text missing from Markdown")
    document_id = hashlib.sha256(f"service-public:{url}".encode()).hexdigest()
    return {
        "document_id": document_id, "url": url, "lang": "fr", "title": title,
        "markdown": markdown, "fiche_id": identifier, "fiche_type": fiche_type,
        "subject": subject, "modified_date": modified_date,
        "source_relative_path": member, "source_sha256": member_sha,
        "archive_sha256": archive_sha256, "archive_member": member,
        "archive_member_sha256": member_sha, "source_heading_count": headings,
        "source_version": f"{modified_date}:{member_sha}",
    }


# --- legacy vosdroits Passage CLI (ported from ingest_vosdroits, 1235) -----------
# The XML parser below emits Passage JSONL for the depth108 funnel; `render` above
# is the Markdown renderer the family uses. Both read the same CONTENT containers.
# Leaf prose blocks within a content container (in document order). Paragraphe
# never nests Paragraphe; Item wraps Paragraphe (captured here); Titre = the
# section/situation headings that give each chunk context.
PROSE_BLOCKS = ("Titre", "Paragraphe")

# Guard against one pathological fiche (the 18 MB "Recherche guidée" decision
# trees) flooding the index with repetitive branch text.
MAX_PASSAGES_PER_FICHE = 80


def fiche_text(root) -> str:
    """Ordered prose of a fiche: the title line, then Titre/Paragraphe blocks
    from every content container. Newline-joined so chunk() can window on
    paragraph boundaries. Falls back to a container's full string() when it has
    no Titre/Paragraphe descendants."""
    blocks: list[str] = []
    for child in root:
        if etree.QName(child).localname not in CONTENT:
            continue
        leaves = list(child.iter(*PROSE_BLOCKS))
        if leaves:
            for leaf in leaves:
                t = _text(leaf)
                if t:
                    blocks.append(t)
        else:
            t = _text(child)
            if t:
                blocks.append(t)
    # dedup consecutive identical blocks (a Titre sometimes echoes into a child)
    out: list[str] = []
    for b in blocks:
        if not out or out[-1] != b:
            out.append(b)
    return "\n".join(out)


def modified_date(root) -> str | None:
    """<dc:date> is 'modified YYYY-MM-DD' (or 'created …'); return the ISO date
    for last_seen (freshness). None if absent/unparseable."""
    raw = (root.findtext(DC + "date") or "").strip()
    m = re.search(r"(\d{4}-\d{2}-\d{2})", raw)
    return m.group(1) if m else None


def parse_fiche(path: str) -> list[Passage]:
    try:
        root = etree.parse(path).getroot()
    except etree.XMLSyntaxError:
        return []
    if etree.QName(root).localname != "Publication":
        return []
    spurl = (root.get("spUrl") or "").strip()
    if not spurl.startswith("http"):
        return []
    title = (root.findtext(DC + "title") or "").strip() or spurl
    host = (urllib.parse.urlsplit(spurl).hostname or "").lower()
    domain = host[4:] if host.startswith("www.") else host   # strip www (legifrance convention)
    last_seen = modified_date(root)

    body = fiche_text(root)
    if not body:
        return []
    # Give the embedder/BM25 the fiche title as context on the first line, same
    # as ingest_gesetze / ingest_canlaw ("<title>\n<body>").
    full = f"{title}\n{body}"
    out: list[Passage] = []
    chunks = chunk_generic_records(full)
    for unit in chunks:
        ch = unit.text
        if low_quality(ch):
            continue
        snippet = " ".join(ch.split()[:25])
        out.append(passage_from_chunk(
            unit, url=spurl, title=title[:300], snippet=snippet,
            lang="fr", domain=domain, tier="A",
            source_type="official", trust_tier=1.0,
            last_seen=last_seen,
        ))
        if len(out) >= MAX_PASSAGES_PER_FICHE:
            break
    return out


def _stage_rows(root: Path, stage: dict) -> list[dict]:
    checked = _validate_stage(stage, "input", artifact_root=root, deep=True)
    receipt = json.loads((root / checked["receipt"]["uri"].removeprefix(
        "artifact://")).read_text(encoding="utf-8"))
    shard = next(root / row["uri"].removeprefix("artifact://")
                 for row in receipt["artifacts"] if row["uri"].endswith(".jsonl.gz"))
    with gzip.open(shard, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def restage(*, root: Path, parent_run: Path, output_run: Path) -> dict:
    parent = json.loads(parent_run.read_text(encoding="utf-8"))
    original = _validate_stage(parent["stages"]["original"], "original",
                               artifact_root=root, deep=True)
    old_outlinks = _stage_rows(root, parent["stages"]["outlinks"])
    receipt = json.loads((root / parent["stages"]["original"]["receipt"]["uri"].removeprefix(
        "artifact://")).read_text(encoding="utf-8"))
    body_path = next(root / row["uri"].removeprefix("artifact://")
                     for row in receipt["artifacts"] if "/body/" in row["uri"])
    archive_bytes = gzip.decompress(body_path.read_bytes())
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    rows, quarantine = [], []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source:
        members = sorted(source.infolist(), key=lambda item: item.filename)
        for item in members:
            safe = PurePosixPath(item.filename)
            if (safe.is_absolute() or ".." in safe.parts or item.is_dir()
                    or not safe.name.startswith("F") or not safe.name.endswith(".xml")):
                continue
            raw = source.read(item)
            row = render(raw, archive_sha, item.filename)
            if row is None:
                quarantine.append({"archive_member": item.filename,
                                   "archive_member_sha256": hashlib.sha256(raw).hexdigest(),
                                   "reason": "missing_title_url_identifier_date_or_content"})
            else:
                rows.append(row)
    rows.sort(key=lambda row: row["url"])
    if len(rows) != parent["stages"]["markdown"]["records"] or quarantine:
        raise ValueError("DILA v4 must conserve all previously admitted fiche parents")
    parent_ids = {row["document_id"] for row in rows}
    linked = {row["document_id"] for row in old_outlinks}
    if not linked <= parent_ids:
        raise ValueError("DILA outlink parent drift")
    empty = [{"document_id": value, "status": "no_external_outlinks"}
             for value in sorted(parent_ids - linked)]
    stages = {"original": original, "outlinks": parent["stages"]["outlinks"]}
    for name, values in (("markdown", rows), ("outlink_empty_parents", empty),
                         ("quarantine", quarantine)):
        data = b"".join((canonical_json(row) + "\n").encode() for row in values)
        logical = hashlib.sha256(data).hexdigest()
        artifact = PS.store_artifact(root, f"stages/{name}-v4/data", "jsonl", data)
        stages[name] = SS.seal(
            artifact_root=root, artifact_paths=[PS.artifact_path(root, artifact)],
            receipt_path=root / f"stages/{name}-v4/receipt.json",
            records=len(values), fmt="canonical-jsonl+deterministic-gzip",
            logical_sha256=logical, producer_id=RESTAGE_PRODUCER, producer_version=RESTAGE_VERSION)
    result = {**parent, "schema": "aw.service-public-fiche-run.v4",
              "version": RESTAGE_VERSION, "status": "complete", "stages": stages,
              "counts": {**parent["counts"],
                         "heading_date_version_documents": len(rows),
                         "empty_outlink_parents": len(empty)},
              "supersedes_pack": "service-public-dila-v3",
              "source_profile_state": "ready_for_v4_chunk_and_compliance_audit"}
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if output_run.exists() and output_run.read_bytes() != encoded:
        raise ValueError("immutable Service-Public v4 RUN collision")
    if not output_run.exists():
        atomic_publish(output_run, encoded)
    for name, stage in stages.items():
        _validate_stage(stage, name, artifact_root=root, deep=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", action="append", required=True,
                    help="extracted dump dir with F*.xml (repeatable)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # De-dup fiche IDs across dumps: a fiche can appear in both particuliers and
    # entreprendre (same spUrl) — keep the first, so uuid5(url) points don't
    # collide and we don't double-count.
    seen_ids: set[str] = set()
    files: list[str] = []
    for d in args.dump_dir:
        for f in sorted(glob.glob(os.path.join(os.path.expanduser(d), "F*.xml"))):
            fid = os.path.splitext(os.path.basename(f))[0]
            if fid in seen_ids:
                continue
            seen_ids.add(fid)
            files.append(f)
    print(f"{len(files)} unique F* fiches across {len(args.dump_dir)} dump(s)",
          flush=True)

    n_fiches = n_pass = n_empty = 0
    with open(os.path.expanduser(args.out), "w", encoding="utf-8") as w:
        for i, path in enumerate(files):
            rows = parse_fiche(path)
            if not rows:
                n_empty += 1
                continue
            n_fiches += 1
            for p in rows:
                # `country` is not a Passage field; it rides in the row dict so
                # it lands in the qdrant payload (embed_p2 copies every non-text
                # key) and the corpus-store country column — matching the
                # legifrance FR points (country=FR).
                d = passage_corpus_record(p)
                d["country"] = "FR"
                w.write(json.dumps(d, ensure_ascii=False) + "\n")
            n_pass += len(rows)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(files)} files, {n_fiches} fiches, "
                      f"{n_pass} passages", flush=True)
    print(f"SUMMARY fiches_in={len(files)} fiches_with_content={n_fiches} "
          f"empty={n_empty} passages_out={n_pass} out={args.out}", flush=True)
