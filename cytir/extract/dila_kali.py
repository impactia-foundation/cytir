"""DILA KALI collective-agreement article -> one record, or several when it is over the embedder
window (`kali_article`).

An article is NEVER cut (ADR-090 am., Tim 2026-10-07, ruling 27 T6 Q-b: the 4,000-character cap is
gone). One that fits the window is one record, as before. A longer one is split at its paragraph
marks (`<p>`) into the fewest balanced chunks that each fit (`parse_article_rows`), the article
header on every chunk. The chunk policy is `native-chunked`.

Moved out of the former `ingest_dila_kali` pipeline script (task 696) so the source family
(`cytir.sources.families.dila_kali`) and the legacy CLI share one parser. Unlike the legacy function
this one says WHY an article was not taken (corpus-doctrine #2). The convention containers
(`KALICONT`, provenance `kali_conv`) are already served and are not part of this family (task 695).

The per-convention flood guard (<= `MAX_ARTICLES_PER_CONV`) is NOT here: it is a cross-unit
count and belongs to the caller (the family adapter, deterministic unit order).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict

from lxml import etree

from cytir.core.common import (Passage, extraction_junk, extractive_snippet, low_quality,
                            mint_passage_point_id)
# the one row budget of the law families (8,000 chars, 2,560 tokens) and its packer: the unification
# of the loda and kali splitters is the follow-up at the next rebuild (ADR-043), not this module
from cytir.extract.legi import MAX_ROW_TOKENS, MIN_BODY_ROOM, ROW_CHARS, _pack, article_chunk_anchor

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

MIN_ARTICLE_CHARS = 260          # drop thin/placeholder articles (number-soup gate)
MAX_ARTICLES_PER_CONV = 30       # flood guard: each CCN's core clauses, not its avenants


def txt(node) -> str:
    if node is None:
        return ""
    raw = etree.tostring(node, encoding="unicode", method="xml")
    return _WS.sub(" ", _TAG.sub(" ", raw)).strip()


def in_force(etat: str) -> bool:
    return (etat or "").upper().startswith("VIGUEUR")


def parse_article_bytes(body: bytes) -> tuple[dict | None, str | None, str]:
    """XML bytes of one KALIARTI -> `(record, convention_key, "kept")` or `(None, None, reason)`.

    The record holds the WHOLE article, never cut (it may be over the embedder window: the family
    takes `parse_article_rows`, which splits it)."""
    rec, conv_key, reason = _parse_article(body)
    return (_public(rec) if rec is not None else None), conv_key, reason


def _parse_article(body: bytes) -> tuple[dict | None, str | None, str]:
    """`parse_article_bytes` with the private `_paragraphs`/`_head`/`_body`/`_aid` keys kept."""
    try:
        root = etree.fromstring(body)
    except Exception as exc:  # noqa: BLE001
        return None, None, f"xml_parse_error:{exc.__class__.__name__}"
    aid = (root.findtext(".//META_COMMUN/ID") or "").strip()
    if not aid:
        return None, None, "no_article_id"
    etat = (root.findtext(".//META_ARTICLE/ETAT") or "").strip()
    if not in_force(etat):
        return None, None, "not_in_force"
    num = (root.findtext(".//META_ARTICLE/NUM") or "").strip()
    art_titre = (root.findtext(".//META_ARTICLE/TITRE") or "").strip()
    conv = ""
    tt = root.find(".//CONTEXTE//TITRE_TXT")
    if tt is not None:
        conv = txt(tt)
    body_txt = txt(root.find(".//BLOC_TEXTUEL/CONTENU"))
    if len(body_txt) < MIN_ARTICLE_CHARS:
        return None, None, "article_too_short"
    tnode = root.find(".//CONTEXTE//TEXTE")
    conv_key = (tnode.get("cid") if tnode is not None else None) or conv[:40]
    head = " — ".join(x for x in
                      [conv[:160], (f"Article {num}" if num else ""), art_titre] if x)
    text = f"{head}\n{body_txt}"
    if low_quality(text):
        return None, None, "low_quality"
    junk = extraction_junk(text)
    if junk:
        return None, None, f"extraction_junk:{junk}"
    url = f"https://www.legifrance.gouv.fr/conv_coll/article/{aid}"
    p = Passage(url=url, title=(head or conv or "Article")[:300],
                snippet=extractive_snippet(body_txt, 30), text=text, lang="fr",
                domain="legifrance.gouv.fr", tier="A", source_type="official", trust_tier=1.0)
    d = asdict(p)
    d.update(country="FR", provenance="kali_article")
    rec = {k: v for k, v in d.items() if v is not None}
    # the paragraph marks, for `parse_article_rows` (a private key, stripped before it is returned)
    paras = [txt(x) for x in root.findall(".//BLOC_TEXTUEL/CONTENU//p")]
    rec["_paragraphs"] = [x for x in paras if x] if " ".join(x for x in paras if x) == body_txt else [body_txt]
    rec["_aid"], rec["_head"], rec["_body"] = aid, head, body_txt
    return rec, conv_key, "kept"


def _public(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def _pieces(units: list[str], room: int) -> list[str]:
    """Every unit at most `room` chars: an over-long paragraph is split on spaces (a word longer
    than the room is sliced); no character is dropped."""
    out: list[str] = []
    for unit in units:
        if len(unit) <= room:
            out.append(unit)
            continue
        cur = ""
        for word in unit.split(" "):
            while len(word) > room:
                if cur:
                    out.append(cur)
                    cur = ""
                out.append(word[:room])
                word = word[room:]
            if not cur:
                cur = word
            elif len(cur) + 1 + len(word) <= room:
                cur += " " + word
            else:
                out.append(cur)
                cur = word
        if cur:
            out.append(cur)
    return out


def split_article_body(head: str, units: list[str], *, count_tokens, row_chars: int = ROW_CHARS,
                       max_tokens: int = MAX_ROW_TOKENS) -> list[str]:
    """The body of each chunk of one article: the fewest balanced groups of paragraphs, each
    `f"{head}\\n{body}"` at most `row_chars` characters AND `max_tokens` tokens. The budget shrinks
    until the worst chunk fits the window (the loda rule, `cytir.extract.legi.split_article_markdown`)."""
    room = max(row_chars - len(head) - 1, MIN_BODY_ROOM)
    while True:
        pieces = _pieces(units, room)
        count = len(_pack(pieces, room))
        lo, hi = max(len(u) for u in pieces), room
        while lo < hi:               # balance: the smallest limit that keeps the chunk count
            mid = (lo + hi) // 2
            if len(_pack(pieces, mid)) == count:
                hi = mid
            else:
                lo = mid + 1
        bodies = [" ".join(g) for g in _pack(pieces, hi)]
        worst = max(count_tokens(f"{head}\n{b}") for b in bodies)
        if worst <= max_tokens:
            return bodies
        nxt = min(int(room * max_tokens / worst * 0.97), room - 50)
        if nxt < MIN_BODY_ROOM // 2:
            raise ValueError(f"no split of this article fits {max_tokens} tokens "
                             f"(worst chunk {worst} tokens at {room} chars)")
        room = nxt


#: below this many characters an article cannot be over the window (2,560 tokens): no count is asked
TOKEN_CHECK_FLOOR_CHARS = 4000


def parse_article_rows(body: bytes, *, count_tokens=None
                       ) -> tuple[list[dict], str | None, str, str]:
    """XML bytes of one KALIARTI -> `(rows, convention_key, reason, source_text)`.

    `source_text` is the whole embed text before any split (what the stage-time guard compares the
    rows with). One row (byte-identical to the unsplit record) when the text fits `ROW_CHARS`
    characters and the window; else several, anchors `article:<KALIARTI>:chunk-<n>`, same url,
    title and header. `count_tokens(text) -> int` is required for an article over
    `TOKEN_CHECK_FLOOR_CHARS`: a stage that cannot measure the window must not guess it."""
    rec, conv_key, reason = _parse_article(body)
    if rec is None:
        return [], None, reason, ""
    head, aid = rec["_head"], rec["_aid"]
    source_text = rec["text"]
    if len(source_text) > TOKEN_CHECK_FLOOR_CHARS and count_tokens is None:
        raise ValueError("an article over %d chars needs `count_tokens` (the embed tokenizer); "
                         "refusing to guess the window" % TOKEN_CHECK_FLOOR_CHARS)
    fits = len(source_text) <= ROW_CHARS and (
        len(source_text) <= TOKEN_CHECK_FLOOR_CHARS or count_tokens(source_text) <= MAX_ROW_TOKENS)
    if fits:
        return [_public(rec)], conv_key, reason, source_text
    bodies = split_article_body(head, rec["_paragraphs"], count_tokens=count_tokens)
    rows = []
    for n, part in enumerate(bodies, 1):
        row = _public(rec)
        row["text"] = f"{head}\n{part}"
        row["snippet"] = extractive_snippet(part, 30)
        row["anchor"] = article_chunk_anchor(aid, n)
        # generation-0 id rule `text-anchor-v0` (ADR-067 am.2), the rule `Passage` itself mints by
        row["point_id"] = mint_passage_point_id(row["url"], row["anchor"], row["text"])
        rows.append(row)
    return rows, conv_key, reason, source_text


#: the convention-container locator text stays a short excerpt (it is not an article, and it is not
#: in the family: ADR-090 am. lifted the cap of the ARTICLES only)
MAX_CONTAINER_CHARS = 4000


# --------------------------------------------------------------------------- #
# Container locators (KALICONT) and the legacy CLI (ported from ingest_dila_kali, 1235)
# --------------------------------------------------------------------------- #
def parse_container(path: str):
    try:
        root = etree.parse(path).getroot()
    except Exception:  # noqa: BLE001
        return None
    cid = (root.findtext(".//META_COMMUN/ID") or "").strip()
    titre = (root.findtext(".//META_CONTENEUR/TITRE") or "").strip()
    etat = (root.findtext(".//META_CONTENEUR/ETAT") or "").strip()
    num = (root.findtext(".//META_CONTENEUR/NUM") or "").strip()
    date_publi = (root.findtext(".//META_CONTENEUR/DATE_PUBLI") or "").strip()
    if not cid or not titre or not in_force(etat):
        return None
    liens = [(l.get("titretxt") or "").strip()
             for l in root.findall(".//STRUCTURE_TXT//LIEN_TXT")]
    liens = [x for x in liens if x][:25]
    parts = [titre]
    if num:
        parts.append(f"Convention collective IDCC {num}.")
    if liens:
        parts.append("Textes : " + " ; ".join(liens) + ".")
    parts.append("Source : Légifrance (base KALI, DILA).")
    text = "\n".join(parts)
    if low_quality(text) or extraction_junk(text):
        return None
    url = f"https://www.legifrance.gouv.fr/conv_coll/id/{cid}"
    p = Passage(url=url, title=titre[:300],
                snippet=extractive_snippet(titre, 30), text=text[:MAX_CONTAINER_CHARS],
                lang="fr", domain="legifrance.gouv.fr", tier="A",
                source_type="official", trust_tier=1.0,
                last_seen=(date_publi if len(date_publi) == 10 and
                           not date_publi.startswith("2999") else None))
    d = asdict(p)
    d.update(country="FR", provenance="kali_conv", idcc=(num or None))
    return {k: v for k, v in d.items() if v is not None}


# --------------------------------------------------------------------------- #
# In-force articles (KALIARTI)
# --------------------------------------------------------------------------- #
def parse_article(path: str):
    """Legacy signature: `(record, convention_key)` or None. One parser: `cytir.extract.dila_kali`."""
    with open(path, "rb") as fh:
        rec, conv_key, _reason = parse_article_bytes(fh.read())
    return (rec, conv_key) if rec is not None else None


def over_window_refusal(rec: dict) -> str | None:
    """One clear line when an article is over the embedder row budget; None when it fits."""
    if len(rec["text"]) <= ROW_CHARS:
        return None
    return (f"{rec['url']}: article is {len(rec['text']):,} chars, over the embedder row budget "
            f"({ROW_CHARS:,}); this legacy CLI cannot split it. Run the family door instead: "
            "`dila_kali` (cytir.sources.families.dila_kali, native-chunked)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-articles", action="store_true",
                    help="containers only (skip KALIARTI)")
    args = ap.parse_args()

    n_cont = n_art = scanned = 0
    seen = set()
    per_conv = defaultdict(int)
    capped = Counter()
    with open(args.out, "w", encoding="utf-8") as w:
        for dirpath, _, files in os.walk(args.root):
            for fn in files:
                if not fn.endswith(".xml"):
                    continue
                path = os.path.join(dirpath, fn)
                if fn.startswith("KALICONT"):
                    scanned += 1
                    rec = parse_container(path)
                    if rec and rec["point_id"] not in seen:
                        seen.add(rec["point_id"])
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_cont += 1
                elif fn.startswith("KALIARTI") and not args.no_articles:
                    scanned += 1
                    res = parse_article(path)
                    if not res:
                        continue
                    rec, conv_key = res
                    refusal = over_window_refusal(rec)
                    if refusal:
                        raise SystemExit(f"REFUSED: {refusal}")
                    if per_conv[conv_key] >= MAX_ARTICLES_PER_CONV:
                        capped[conv_key] += 1
                        continue
                    if rec["point_id"] in seen:
                        continue
                    seen.add(rec["point_id"])
                    per_conv[conv_key] += 1
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_art += 1
                    if (n_cont + n_art) % 20000 == 0:
                        print(f"  {n_cont} conv + {n_art} art "
                              f"({scanned} scanned)", flush=True)
    print(f"DONE containers={n_cont} articles={n_art} scanned={scanned} "
          f"conv_capped={len(capped)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
