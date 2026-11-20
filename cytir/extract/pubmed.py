#!/usr/bin/env python3
"""PubMed baseline parser: NCBI MEDLINE XML.gz → passages. Authoritative
biomedical abstracts (PMID, title, abstract). URL = pubmed.ncbi.nlm.nih.gov/<pmid>/.
Bulk, clean, URL-able. Tier B / reference. Moved from the flat driver script (task 1404,
S7); the adapter is `cytir.sources.families.pubmed`. The old path forwards here.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import asdict

from lxml import etree

from cytir.core.common import Passage, low_quality

#: The producer id written into sealed L1 receipts (task 417); frozen, declared under
#: `producer_ids` of family `named_slice:pubmed`. It names the retired script path, not a code location.
PRODUCER = "pipeline/ingest_pubmed.py:iter_pubmed_articles"

BASE = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"

_URL_RE = re.compile(r"https?://[^\s<>\]\[(){}\"']+")
_DOI_RE = re.compile(r"(?<![\w/])(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)


def _node_text(node) -> str:
    """Return all inline XML text as one normalized, lossless word stream."""
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def _section_name(node, fallback: str) -> str:
    label = " ".join((node.get("Label") or "").split())
    category = " ".join((node.get("NlmCategory") or "").split())
    if label and category and label.casefold() != category.casefold():
        return f"{label} ({category})"
    return label or category or fallback


def _source_links(elem) -> list[dict]:
    """Extract source-native identifiers/links with relation and XML context."""
    links: dict[tuple[str, str, str], dict] = {}

    def add(url: str, relation: str, context: str) -> None:
        url = url.rstrip(".,;:")
        if not url:
            return
        key = (url, relation, context)
        links[key] = {"url": url, "relation": relation, "context": context}

    for node in elem.findall(".//PubmedData/ArticleIdList/ArticleId"):
        value = _node_text(node)
        kind = (node.get("IdType") or "").casefold()
        if kind == "doi" and value:
            add(f"https://doi.org/{value}", "doi", "PubmedData/ArticleIdList")
        elif kind == "pmc" and value:
            add(f"https://pmc.ncbi.nlm.nih.gov/articles/{value}/", "full-text",
                "PubmedData/ArticleIdList")
    for node in elem.findall(".//Article/ELocationID"):
        if (node.get("EIdType") or "").casefold() == "doi":
            value = _node_text(node)
            if value:
                add(f"https://doi.org/{value}", "doi", "Article/ELocationID")
    for reference in elem.findall(".//ReferenceList//Reference"):
        citation = _node_text(reference.find("Citation"))
        context = f"ReferenceList: {citation[:240]}" if citation else "ReferenceList"
        for node in reference.findall(".//ArticleId"):
            value = _node_text(node)
            kind = (node.get("IdType") or "").casefold()
            if kind == "doi" and value:
                add(f"https://doi.org/{value}", "cited-source", context)
            elif kind == "pubmed" and value:
                add(f"https://pubmed.ncbi.nlm.nih.gov/{value}/", "cited-source", context)
            elif kind == "pmc" and value:
                add(f"https://pmc.ncbi.nlm.nih.gov/articles/{value}/",
                    "cited-source", context)
    searchable = _node_text(elem.find(".//Article"))
    for match in _URL_RE.finditer(searchable):
        add(match.group(0), "mentioned-link", "Article text")
    for match in _DOI_RE.finditer(searchable):
        add(f"https://doi.org/{match.group(1)}", "mentioned-doi", "Article text")
    return [links[key] for key in sorted(links)]


def _record_markdown(*, pmid: str, title: str, primary_nodes: list,
                     other_abstracts: list, source_archive: dict | None) -> str:
    lines = [f"# {title}", "", "## Record identity", "", f"- PMID: `{pmid}`"]
    if source_archive:
        lines.append(f"- Parent archive: `{source_archive['member']}`")
        if source_archive.get("sha256"):
            lines.append(f"- Parent archive SHA-256: `{source_archive['sha256']}`")
    lines.extend(("", "## Abstract", ""))
    structured = len(primary_nodes) > 1 or any(
        node.get("Label") or node.get("NlmCategory") for node in primary_nodes)
    for index, node in enumerate(primary_nodes, 1):
        if structured:
            lines.extend((f"### {_section_name(node, f'Section {index}')}", ""))
        lines.extend((_node_text(node), ""))
    for other_index, other in enumerate(other_abstracts, 1):
        suffix = "" if len(other_abstracts) == 1 else f" {other_index}"
        lines.extend((f"## Other abstract{suffix}", ""))
        abstract_type = " ".join((other.get("Type") or "").split())
        language = " ".join((other.get("Language") or "").split())
        if abstract_type:
            lines.append(f"- Type: {abstract_type}")
        if language:
            lines.append(f"- Language: {language}")
        if abstract_type or language:
            lines.append("")
        nodes = other.findall("./AbstractText")
        structured_other = len(nodes) > 1 or any(
            node.get("Label") or node.get("NlmCategory") for node in nodes)
        for section_index, node in enumerate(nodes, 1):
            if structured_other:
                lines.extend((f"### {_section_name(node, f'Section {section_index}')}", ""))
            lines.extend((_node_text(node), ""))
    return "\n".join(lines).rstrip()


#: ISO 639-2 codes of the four target languages. An article in any other language, or with no
#: Language element, is dropped, as the other extractors do (`bfs_dam.norm_lang` -> None).
_ISO639_2 = {"eng": "en", "fre": "fr", "fra": "fr", "ger": "de", "deu": "de", "ita": "it"}


def _target_lang(value: str | None) -> str | None:
    """First listed language wins; None when it is not en/de/fr/it."""
    if not value:
        return None
    return _ISO639_2.get(value.strip().lower())


def iter_pubmed_articles(content: bytes, *, source_archive: dict | None = None):
    """Yield source-structured PubMed record Markdown from decompressed XML."""
    for _, elem in etree.iterparse(io.BytesIO(content), tag="PubmedArticle"):
        pmid = elem.findtext(".//MedlineCitation/PMID")
        title = _node_text(elem.find(".//Article/ArticleTitle"))
        absnodes = elem.findall(".//Article/Abstract/AbstractText")
        abstract = " ".join(_node_text(node) for node in absnodes).strip()
        lang = _target_lang(elem.findtext(".//Article/Language"))
        if not pmid or not title or len(abstract) < 80 or lang is None:
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
            continue
        # Keep the legacy admission decision on title + primary abstract so
        # source-link and OtherAbstract enrichment cannot change population.
        if low_quality(f"{title}\n{abstract}"):
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
            continue
        other_abstracts = elem.findall(".//MedlineCitation/OtherAbstract")
        text = _record_markdown(pmid=pmid, title=title, primary_nodes=absnodes,
                                other_abstracts=other_abstracts,
                                source_archive=source_archive)
        source_links = _source_links(elem)
        p = Passage(url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    title=title, snippet=" ".join(abstract.split()[:30]),
                    text=text, lang=lang,
                    domain="pubmed.ncbi.nlm.nih.gov", tier="B",
                    source_type="reference", trust_tier=0.85,
                    outlinks=[{"url": link["url"], "nofollow": False}
                              for link in source_links])
        row = asdict(p)
        row["source_record_id"] = pmid
        row["parent_archive"] = dict(source_archive) if source_archive else None
        row["source_links"] = source_links  # explicit [] is required evidence
        yield row
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]


def parse_gz(content: bytes, out_f) -> int:
    n = 0
    for row in iter_pubmed_articles(content):
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    return n
