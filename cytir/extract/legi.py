#!/usr/bin/env python3
"""Parse the DILA LEGI dump (French consolidated law) -> Passage JSONL.
Dumps-first (no scraping). Default: the named CODES, in-force (VIGUEUR) articles
only — the high-value, on-mission slice (TNC = millions of individual decrees,
opt-in via --include-tnc).

  python parse_legi.py --root <untarred legi dir> --out legi.jsonl

# TODO(linkgraph): LEGI articles carry statutory cross-references (renvois:
# "art. L. 123-4", "code … art. N", liens to other codes). Emit these as edges
# into the same link-graph store cc_ingest/extract_outlinks feed (ADR-020) — a
# *legal citation graph* (statute → cited statute) alongside the web endorsement
# graph. Same {src, links} shape; targets are fedlex/legifrance article URLs.
# Not done here: leave the unified web extractor stable first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict

from lxml import etree

from cytir.core.common import Passage, extractive_snippet, low_quality, mint_passage_point_id

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
# Body floor (task 322): the old 120-char floor silently dropped the SHORT
# famous code articles — C.civ. art. 212 (74 chars), C.pén. 311-1 (60), C.civ.
# 144, C.trav. L3121-27 — 4 of the 15 legifrance gold URLs. An in-force code
# article with any real body is on-mission however short; the floor only needs
# to kill empty/placeholder bodies ("Abrogé", lone article numbers).
MIN_BODY = 30
_BLOCK_TAGS = frozenset({"p", "div", "section", "article", "blockquote",
                         "table", "tr", "ul", "ol"})


def text_of(node) -> str:
    if node is None:
        return ""
    raw = etree.tostring(node, encoding="unicode", method="xml")
    return _WS.sub(" ", _TAG.sub(" ", raw)).strip()


def _xml_content_markdown(node) -> str:
    """Render every XML text byte into deterministic block-aware Markdown."""
    if node is None:
        return ""
    parts: list[str] = []

    def walk(item) -> None:
        if not isinstance(item.tag, str):
            return
        tag = etree.QName(item).localname.lower()
        if tag == "br":
            parts.append("\n")
        elif tag == "li":
            parts.append("\n- ")
        elif tag in _BLOCK_TAGS:
            parts.append("\n\n")
        if item.text:
            parts.append(item.text)
        for child in item:
            walk(child)
            if child.tail:
                parts.append(child.tail)
        if tag in _BLOCK_TAGS:
            parts.append("\n\n")

    walk(node)
    lines = [_WS.sub(" ", line).strip() for line in "".join(parts).splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


_ARTICLE_FAMILIES = frozenset({"codes", "loda"})


def _article_url(article_id: str, family: str = "codes") -> str:
    if family not in _ARTICLE_FAMILIES:
        raise ValueError(f"unsupported Legifrance article URL family: {family}")
    return f"https://www.legifrance.gouv.fr/{family}/article_lc/{article_id}"


def _link_target(link, article_url_family: str = "codes") -> str:
    target_id = str(link.get("id") or link.get("cidtexte") or "").strip()
    if target_id.startswith("LEGIARTI"):
        return _article_url(target_id, article_url_family)
    if target_id:
        return f"https://www.legifrance.gouv.fr/loda/id/{target_id}"
    return ""


def parse_article_xml_legal_markdown(body: bytes, *, archive_member: str,
                                     archive_sha256: str,
                                     source_release: str | None = None,
                                     article_url_family: str = "codes",
                                     source_id: str =
                                     "legifrance-dila-global-production-v2") -> dict | None:
    """Project one official DILA article into complete anchored legal Markdown."""
    # Keep URL-family selection explicit.  LEGI XML backs both the named-code
    # (`codes`) and non-codified-law (`loda`) production families, but the
    # canonical web identities are not interchangeable.
    _article_url("LEGIARTI000000000000", article_url_family)
    root = etree.fromstring(body)
    article_id = (root.findtext(".//META_COMMUN/ID") or "").strip()
    number = (root.findtext(".//META_ARTICLE/NUM") or "").strip()
    state = (root.findtext(".//META_ARTICLE/ETAT") or "").strip()
    start = (root.findtext(".//META_ARTICLE/DATE_DEBUT") or "").strip()
    end = (root.findtext(".//META_ARTICLE/DATE_FIN") or "").strip()
    kind = (root.findtext(".//META_ARTICLE/TYPE") or "").strip()
    body_markdown = _xml_content_markdown(root.find(".//BLOC_TEXTUEL/CONTENU"))
    if not article_id or not body_markdown:
        return None
    titles = []
    for node in root.findall(".//CONTEXTE/TEXTE/TITRE_TXT"):
        title = _WS.sub(" ", "".join(node.itertext())).strip()
        if title and title not in titles:
            titles.append(title)
    title = titles[-1] if titles else "Légifrance"
    hierarchy = []
    for node in root.findall(".//CONTEXTE/TEXTE//TITRE_TM"):
        value = _WS.sub(" ", "".join(node.itertext())).strip()
        if value:
            hierarchy.append({"id": str(node.get("id") or ""), "title": value,
                              "start": str(node.get("debut") or ""),
                              "end": str(node.get("fin") or "")})
    paragraphs = [value.strip() for value in re.split(r"\n\s*\n|\n", body_markdown)
                  if value.strip()]
    article_heading = f"Article {number}" if number else "Article"
    markdown = [f"# {title}", "", "## Article identity",
                f"- Article ID: `{article_id}`", f"- Article number: {number or 'unspecified'}",
                f"- Status: {state or 'unspecified'}", f"- Effective from: {start or 'unspecified'}",
                f"- Effective until: {end or 'unspecified'}", f"- Type: {kind or 'unspecified'}",
                f"- Archive member: `{archive_member}`", f"- Archive SHA-256: `{archive_sha256}`",
                f"- Source release: {source_release or 'global-archive'}",
                "", "## Legal hierarchy"]
    if hierarchy:
        for level, item in enumerate(hierarchy, 1):
            markdown.append(
                f"{'  ' * (level - 1)}- {item['title']} "
                f"(`{item['id'] or 'no-section-id'}`, {item['start'] or '?'}–{item['end'] or '?'})")
    else:
        markdown.append("- No intermediate section in the official article context")
    markdown.extend(("", f"## {article_heading} {{#{article_id}}}"))
    for ordinal, paragraph in enumerate(paragraphs, 1):
        markdown.extend(("", f"### Paragraph {ordinal} {{#{article_id}-p{ordinal}}}", "", paragraph))
    markdown_text = "\n".join(markdown).strip()
    links = []
    unresolved_relations = []
    for link in root.findall(".//LIENS/LIEN"):
        target = _link_target(link, article_url_family)
        context = _WS.sub(" ", "".join(link.itertext())).strip()
        relation = {
            "relation": str(link.get("typelien") or "legal-reference").lower(),
            "direction": str(link.get("sens") or ""), "target": target,
            "target_id": str(link.get("id") or link.get("cidtexte") or ""),
            "context": context,
            "source_version": {"status": state, "effective_from": start,
                               "effective_until": end,
                               "source_release": source_release},
            "article_anchor": article_id,
        }
        if target and context:
            links.append(relation)
        else:
            unresolved_relations.append({
                **relation,
                "unresolved_reasons": [
                    reason for missing, reason in (
                        (not target, "target_absent"),
                        (not context, "context_absent")) if missing],
            })
    url = _article_url(article_id, article_url_family)
    anchor = f"article:{article_id}"
    return {
        "schema": "aw.l1-markdown.v1", "profile": "anchored_law",
        "source_id": source_id,
        "source_record_id": article_id, "url": url, "canonical_url": url,
        "title": f"{title} — {article_heading}", "lang": "fr", "jurisdiction": "FR",
        "point_id": mint_passage_point_id(url, anchor, markdown_text),
        "anchor": anchor, "markdown": markdown_text,
        "markdown_sha256": hashlib.sha256(markdown_text.encode()).hexdigest(),
        "source_record_sha256": hashlib.sha256(body).hexdigest(),
        "source_archive": {"member": archive_member, "sha256": archive_sha256,
                           "release": source_release},
        "source_version": {"status": state, "effective_from": start,
                           "effective_until": end,
                           "source_release": source_release},
        "legal_hierarchy": hierarchy, "article_number": number,
        "paragraph_anchors": [f"{article_id}-p{i}" for i in range(1, len(paragraphs) + 1)],
        "source_links": links, "source_links_explicit": True,
        "source_relations_unresolved": unresolved_relations,
        "source_relations_unresolved_explicit": True,
    }


# --- chunking of an over-long article (task 706) -----------------------------------------------
# One article is one chunk while its embed text fits `ROW_CHARS`. A longer article is split on its
# `### Paragraph N` boundaries into the FEWEST, balanced chunks that each fit, and every chunk
# carries the full article header (title line, identity block, legal hierarchy, article heading):
# the representation contract (law title + heading path on every row) is the same for every chunk.
# NOTHING is cut: the chunks together hold every paragraph (a paragraph larger than one chunk is
# split on spaces into numbered parts). `ROW_CHARS` is the size the family always used as its
# bound (task 633); it is now a split point, not a cut. Measured on all 13,948 long loda articles
# with the real arctic tokenizer (task 706).
ROW_CHARS = 8000
#: the embedder's window (`cyt embed shards --max-seq`); a chunk over it is cut again, silently
MAX_ROW_TOKENS = 2560
# A header that nearly fills a row still leaves room for body text.
MIN_BODY_ROOM = 2000
_PARA_SEP = "\n\n### Paragraph "
_PART_ROOM = len(" (part 999/999)")


def article_chunk_anchor(article_id: str, index: int) -> str:
    """Anchor of chunk `index` (1-based) of a split article: `article:<LEGIARTI>:chunk-<n>`."""
    return f"article:{article_id}:chunk-{index}"


def _split_paragraph(section: str, room: int) -> list[str]:
    """One `### Paragraph N {#id-pN}\\n\\ntext` section larger than `room` -> numbered parts that
    each fit, split on spaces (the text is whitespace-normalised, so a space is the only
    boundary). A single word longer than the room is sliced; no character is dropped."""
    heading, _, text = section.partition("\n\n")
    head_core, _, anchor = heading.partition(" {#")
    room_text = max(room - len(heading) - _PART_ROOM - 2, 200)
    pieces: list[str] = []
    cur = ""
    for word in text.split(" "):
        while len(word) > room_text:                 # an unbroken run longer than a part
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(word[:room_text])
            word = word[room_text:]
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= room_text:
            cur += " " + word
        else:
            pieces.append(cur)
            cur = word
    if cur or not pieces:
        pieces.append(cur)
    total = len(pieces)
    return [f"{head_core} (part {k}/{total}) {{#{anchor}\n\n{piece}"
            for k, piece in enumerate(pieces, 1)]


def _pack(units: list[str], limit: int) -> list[list[str]]:
    groups: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for unit in units:
        add = len(unit) + (2 if cur else 0)
        if cur and size + add > limit:
            groups.append(cur)
            cur, size = [unit], len(unit)
        else:
            cur.append(unit)
            size += add
    if cur:
        groups.append(cur)
    return groups


def _split_at(markdown: str, title: str, row_chars: int) -> list[str]:
    """The fewest balanced chunks of `markdown`, each `header + paragraph sections`, none over
    `row_chars` characters of embed text unless the header alone leaves under `MIN_BODY_ROOM`."""
    header, *rest = markdown.split(_PARA_SEP)
    sections = ["### Paragraph " + s for s in rest]
    if "\n\n".join([header, *sections]) != markdown:       # pragma: no cover - a parser change
        raise ValueError("article markdown does not re-assemble from its sections")
    room = max(row_chars - (len(title) + 2 + len(header) + 2), MIN_BODY_ROOM)
    units: list[str] = []
    for sec in sections:
        units.extend(_split_paragraph(sec, room) if len(sec) > room else [sec])
    count = len(_pack(units, room))
    lo, hi = max(len(u) for u in units), room
    while lo < hi:                       # balance: the smallest limit that keeps the chunk count
        mid = (lo + hi) // 2
        if len(_pack(units, mid)) == count:
            hi = mid
        else:
            lo = mid + 1
    return ["\n\n".join([header, *g]) for g in _pack(units, hi)]


def split_article_markdown(markdown: str, *, title: str, row_chars: int = ROW_CHARS,
                           count_tokens=None, max_tokens: int = MAX_ROW_TOKENS,
                           split_over_window: bool = False) -> list[str]:
    """The Markdown of each chunk of one anchored-law article (`parse_article_xml_legal_markdown`'s
    `markdown`); the embed text of a row is `f"{title}\\n\\n{chunk}"`.

    `[markdown]` itself, byte for byte, when `title + markdown` fits `row_chars`: no token counter
    is asked and nothing changes for a short article. Otherwise the fewest balanced chunks, each
    `header + paragraph sections`, each at most `row_chars` characters AND at most `max_tokens`
    embedder tokens. Characters alone cannot promise the second: on the 13,948 long loda articles a
    8,000-char chunk is 1,500 to 3,700 tokens, so 36 % of the rows would be cut again by the
    embedder. `count_tokens(text) -> int` is therefore REQUIRED when a split is needed (the real
    arctic tokenizer, `cytir.chunk.token_counter`): the char budget shrinks, in proportion to the
    worst chunk, until every chunk fits. That is the largest chunk that fits, per article.

    `split_over_window` (ruling 27 T6, the loda restage): a SHORT article is also tested in tokens.
    One that fits `row_chars` but not `max_tokens` (574 task-633 rows, up to 3,197 tokens, which
    the embedder cut) is split like a long one; one that fits both stays `[markdown]`, byte for
    byte. Needs `count_tokens`. Off by default: a full run keeps every short row byte-identical."""
    if len(title) + 2 + len(markdown) <= row_chars:
        if not split_over_window:
            return [markdown]
        if count_tokens is None:
            raise ValueError("split_over_window needs `count_tokens` (the embed tokenizer)")
        if count_tokens(f"{title}\n\n{markdown}") <= max_tokens:
            return [markdown]
    if count_tokens is None:
        raise ValueError("splitting an over-long article needs `count_tokens` (the embed "
                         "tokenizer); refusing to guess a character budget")
    budget = row_chars
    prefix = len(title) + 2 + len(markdown.split(_PARA_SEP)[0]) + 2
    while True:
        chunks = _split_at(markdown, title, budget)
        worst = max(count_tokens(f"{title}\n\n{c}") for c in chunks)
        if worst <= max_tokens:
            return chunks
        nxt = min(int(budget * max_tokens / worst * 0.97), budget - 50)
        if nxt < prefix + 500:
            raise ValueError(f"no split of this article fits {max_tokens} tokens "
                             f"(worst chunk {worst} tokens at {budget} chars)")
        budget = nxt


def parse_article(path: str):
    try:
        with open(path, "rb") as source:
            return parse_article_xml(source.read())
    except Exception:  # noqa: BLE001
        return None


def _parse_article_xml(body: bytes, *, reconstruction: bool):
    """Parse one DILA article from preserved XML bytes.

    Filesystem walks and archive replay share this exact implementation so
    status filtering, article boundaries, text composition, URLs, and point
    IDs cannot drift between live ingest and reconstruction.
    """
    root = etree.fromstring(body)
    # In-force TODAY = VIGUEUR or ABROGE_DIFF (abrogated with DEFERRED effect —
    # still applicable law until the future date; e.g. the whole CPP under
    # Ord. 2025-1091 is ABROGE_DIFF until 2029-01-01, incl. gold Art. 63,
    # task 322). VIGUEUR_DIFF (enacted, not yet applicable) stays excluded.
    state = (root.findtext(".//META_ARTICLE/ETAT") or "").strip()
    if not reconstruction and state not in ("VIGUEUR", "ABROGE_DIFF"):
        return None
    aid = root.findtext(".//META_COMMUN/ID")
    num = (root.findtext(".//META_ARTICLE/NUM") or "").strip()
    article_body = text_of(root.find(".//BLOC_TEXTUEL/CONTENU"))
    if (not aid or not article_body or
            (not reconstruction and
             (len(article_body) < MIN_BODY or low_quality(article_body)))):
        return None
    titres = root.findall(".//CONTEXTE/TEXTE/TITRE_TXT")
    parent = (titres[0].text or "").strip() if titres else "Légifrance"
    title = f"{parent} — Article {num}" if num else parent
    p = Passage(url=f"https://www.legifrance.gouv.fr/codes/article_lc/{aid}",
                title=title[:300], snippet=extractive_snippet(article_body, 30),
                text=f"{title}\n{article_body}", lang="fr",
                domain="legifrance.gouv.fr", tier="A",
                source_type="official", trust_tier=1.0)
    d = asdict(p)
    # provenance/country parity with the ingest_dila_* pack scripts (task 322)
    d.update(country="FR", provenance="legi_code", article_state=state,
             source_xml_id=aid)
    return {k: v for k, v in d.items() if v is not None}


def parse_article_xml(body: bytes):
    """Live/in-force view, with the complete normalized article body."""
    return _parse_article_xml(body, reconstruction=False)


def parse_article_xml_reconstruction(body: bytes):
    """Historical reconstruction view retaining every state and full body."""
    return _parse_article_xml(body, reconstruction=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-tnc", action="store_true",
                    help="also parse non-codified texts (millions; default codes-only)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    n = ok = 0
    seen = set()
    with open(args.out, "w", encoding="utf-8") as f:
        for dirpath, _, files in os.walk(args.root):
            if not args.include_tnc and "/code_en_vigueur/" not in dirpath + "/":
                continue
            for fn in files:
                if not (fn.startswith("LEGIARTI") and fn.endswith(".xml")):
                    continue
                n += 1
                rec = parse_article(os.path.join(dirpath, fn))
                if rec and rec["point_id"] not in seen:
                    seen.add(rec["point_id"])
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ok += 1
                if ok and ok % 20000 == 0:
                    print(f"  {ok} articles kept ({n} scanned)", flush=True)
                if args.limit and ok >= args.limit:
                    print(f"DONE (limit): {ok} kept / {n} scanned -> {args.out}")
                    return
    print(f"DONE: {ok} in-force code articles kept / {n} scanned -> {args.out}")


if __name__ == "__main__":
    main()
