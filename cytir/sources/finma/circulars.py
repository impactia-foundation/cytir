#!/usr/bin/env python3
"""Task 209 — FINMA authoritative-core pack (the gap flagged by task 190).

The existing 2,267 finma.ch points are mostly press releases; FINMA's actual
regulatory INSTRUMENTS were absent (190 audit: Rundschreiben 0, supervisory
notices 1, insurance-law rulings 0). This pack fills that core:

  * Rundschreiben (circulars)                     — the FINMA reference corpus
  * FINMA-Aufsichtsmitteilungen (supervisory notices)
  * Versicherungsrechtliche Entscheide (insurance-supervision rulings)
  * FINMA-Stellungnahmen (statements)

PDF-BASED pack. The finma.ch document center lists these via a JS/AJAX widget
backed by the site's own search API (POST /<lang>/api/search/getresult with
`ds=<datasource-GUID>`, ?Skip=<n>-paginated) — the ONLY machine-readable route
to the current in-force set (the /suche/ human UI is robots-disallowed; this
per-widget datasource endpoint is not). Each API item carries Title +
Description + the PDF Link (+ OtherLanguageLinks for the trilingual
circulars/notices).

TWO STAGES (venv split — Docling needs torch/numpy2 in the main venv, the
fasttext language gate 166 needs numpy<2 in the hygiene venv):

  extract  (.venv)          enumerate -> polite-fetch each PDF -> L0 raw
                            retention (ADR-026, kind=pdf; API JSON kind=api)
                            -> Docling markdown (OCR off — every sampled FINMA
                            PDF incl. 1998 vintage is digital-native, task-209
                            verdict) -> docs_md.jsonl (one row/doc: metadata +
                            raw markdown; ok=False rows carry no md and become
                            a metadata fallback in the gate). Resumable.

  gate     (hygiene venv)   docs_md.jsonl -> prepend the curated Title +
                            Description as an H1 (so the circular number
                            "2015/02" and subject "Liquiditätsrisiken" always
                            embed) -> fp.chunk -> gates 097 (extraction junk) /
                            166 (content-LID scope — catches finma.ch's
                            untranslated-page quirk) + the 175 apparatus filter
                            + 50-chunk head-cap -> passages.jsonl. A doc whose
                            Docling output is empty/low-quality gets ONE
                            metadata record (Title + Description) so the
                            authoritative URL still surfaces (AW's mission: a
                            finder of authoritative URLs, never ship junk — the
                            176 rule; here there is no HTML landing to fall back
                            to, so the curated metadata is the fallback).

provenance=finma_circulars, tier A (regulator), trust 0.95, source_type
official. Languages de/fr/it (FINMA publishes trilingually; EN passive — the
166 gate keeps an English PDF, drops out-of-scope).

Robots verified live (finma.ch: Allow /, disallows /suche//sitemap//error/ +
/sitecore/media library/ — the /<lang>/~/media/... PDF paths are allowed).

  .venv/bin/python        -m cytir.sources.finma.circulars extract --out-dir <work dir>
  hygiene/bin/python      -m cytir.sources.finma.circulars gate    --out-dir <work dir>

MOVED (task 1396, S7): this module was the pipeline driver `ingest_finma_circulars.py` (the old
path is a forwarding shim). The family adapter (`cytir.sources.families.finma_circulars`) DELEGATES
to `enumerate_source`, `pdf_to_markdown` and `doc_passages` here, so the standalone `extract` and
`gate` stages and the Protocol adapter share one body of logic.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import re
import sys
import time
import urllib.parse

from cytir.sources.family import HttpSpec  # noqa: E402
from cytir.sources.http import make_client  # noqa: E402

from cytir.acquire import crawl as pack_fetch
from cytir.chunk.generic import chunk_generic_records  # noqa: E402
from cytir.acquire.crawl import L0Store, fetch_robots  # noqa: E402

PROV = "finma_circulars"
HOST = "www.finma.ch"
BASE = "https://www.finma.ch"
API = "/de/api/search/getresult"          # de widget enumerates the full set
TODAY = dt.date.today().isoformat()

# datasource GUIDs of the four documentation widgets (extracted live from the
# section pages' data-source attribute, task-209 recon)
SOURCES = [
    # key,                          datasource GUID,                          multilang
    ("rundschreiben",
     "{3009DAA1-E9A3-4CF1-B0F0-8059B9A37AFA}", True),
    ("aufsichtsmitteilungen",
     "{ED0E8A7F-25EA-4DB1-B53B-C2327A435359}", True),
    ("versicherungsrechtliche_entscheide",
     "{F475205A-A058-469A-88B2-FBAFA2C00FD1}", False),
    ("stellungnahmen",
     "{E4030428-9A63-4B9F-97CA-2E74CE7771A9}", True),
]

_LANGWORD = {
    "deutsch": "de", "französisch": "fr", "franzosisch": "fr", "franz": "fr",
    "italienisch": "it", "englisch": "en", "rätoromanisch": "de"}
_LANGWORD_RE = re.compile(
    r"\s+(Deutsch|Französisch|Franz\.?|Italienisch|Englisch|Rätoromanisch)\s*$",
    re.I)

canon = pack_fetch.make_canon()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _abs(u: str) -> str:
    return u if u.startswith("http") else urllib.parse.urljoin(BASE, u)


def _lang_from_url(u: str) -> str | None:
    m = re.search(r"[?&]sc_lang=(de|fr|it|en)\b", u)
    if m:
        return m.group(1)
    m = re.search(r"^/(de|fr|it|en)/", urllib.parse.urlsplit(u).path)
    return m.group(1) if m else None


def _clean_vre_title(t: str) -> str:
    return _LANGWORD_RE.sub("", t).strip()


# ============================================================================
# STAGE: extract  (.venv — Docling)
# ============================================================================
def enumerate_source(client, key, ds, multilang, l0, delay):
    """De-duplicated fetch targets; stores each raw API response as L0(api)."""
    docs: dict[str, dict] = {}
    skip = 0
    while True:
        url = f"{BASE}{API}" + (f"?Skip={skip}" if skip else "")
        r = client.post(url, data={"ds": ds},
                        headers={"Content-Type":
                                 "application/x-www-form-urlencoded"})
        r.raise_for_status()
        if l0 is not None:
            l0.put(f"{API}?ds={ds}&Skip={skip}", r.content, status=r.status_code,
                   content_type="application/json", kind="api")
        j = r.json()
        items = j.get("Items") or []
        for it in items:
            if (it.get("Extension") or "").lower() != "pdf":
                continue
            title = (it.get("Title") or "").strip()
            desc = (it.get("Description") or "").strip()
            date = it.get("Date") or ""
            variants: list[tuple[str, str | None]] = []
            if multilang and it.get("OtherLanguageLinks"):
                for ol in it["OtherLanguageLinks"]:
                    variants.append((_abs(ol["Url"]),
                                     (ol.get("Name") or "").lower()[:2] or None))
            elif it.get("Link"):
                variants.append((_abs(it["Link"]), None))
            m = _LANGWORD_RE.search(title)
            title_lang = _LANGWORD.get(m.group(1).lower(), None) if m else None
            for u, lname in variants:
                cu = canon(u, keep_query=True)
                if not cu or urllib.parse.urlsplit(cu).netloc != HOST:
                    continue
                lang = (_lang_from_url(u) or lname or title_lang or "de")
                dtitle = title if multilang else _clean_vre_title(title)
                docs.setdefault(cu, {
                    "url": cu, "lang": lang, "title": dtitle,
                    "description": desc, "date": date, "source": key})
        count = j.get("Count") or 0
        skip += len(items)
        if not items or skip >= count:
            break
        time.sleep(delay)
    return list(docs.values())


def make_converter():
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    po = PdfPipelineOptions()
    po.do_ocr = False
    po.do_table_structure = False
    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=po)})


def pdf_to_markdown(conv, body: bytes, name: str) -> str:
    from docling.datamodel.base_models import DocumentStream
    src = DocumentStream(name=name, stream=io.BytesIO(body))
    return conv.convert(src).document.export_to_markdown()


def stage_extract(args):
    D = args.out_dir
    os.makedirs(D, exist_ok=True)
    md_path = os.path.join(D, "docs_md.jsonl")
    inv_path = os.path.join(D, "inventory.jsonl")
    state_path = os.path.join(D, "extract_state.json")
    open(os.path.join(D, "outlinks.jsonl"), "a").close()  # pack contract (194)

    state = json.load(open(state_path)) if os.path.exists(state_path) else {
        "done": [], "fetched": 0, "docs_ok": 0, "docs_empty": 0,
        "http_err": 0, "docling_err": 0, "per_source": {}, "started_at": _now()}
    done = set(state["done"])

    client = make_client(HttpSpec(timeout=90, user_agent_kind="pack_fetch",
                                  follow_redirects=True))
    rb = fetch_robots(client, HOST)
    if rb is None:
        print("[robots] UNAVAILABLE/JS-shell/403 -> ABORT", flush=True)
        sys.exit(1)
    print(f"[robots] ok crawl-delay={rb.delay}", flush=True)
    eff = max(args.delay, rb.delay or 0.0)

    if not args.no_l0 and not args.l0_root:
        sys.exit("extract: give --l0-root (or AW_L0_ROOT), or --no-l0")
    l0 = None if args.no_l0 else L0Store(args.l0_root, PROV)
    if l0 is not None:
        print(f"[l0] -> {l0.dir} ({l0.records} carried over)", flush=True)

    only = set(args.only.split(",")) if args.only else None
    if os.path.exists(inv_path):
        docs = [json.loads(l) for l in open(inv_path) if l.strip()]
        print(f"[enum] inventory cached: {len(docs)} docs", flush=True)
    else:
        docs = []
        for key, ds, multilang in SOURCES:
            if only and key not in only:
                continue
            got = enumerate_source(client, key, ds, multilang, l0, eff)
            print(f"[enum] {key}: {len(got)} pdf targets", flush=True)
            state["per_source"].setdefault(key, {})["targets"] = len(got)
            docs += got
            time.sleep(eff)
        with open(inv_path, "w") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    if only:
        docs = [d for d in docs if d["source"] in only]

    print(f"[fetch] {len(docs)} targets, {len(done)} already done", flush=True)
    conv = make_converter()
    mf = open(md_path, "a", encoding="utf-8")
    t0 = time.time()
    last = [time.time()]
    new = 0

    def row(doc, ok, md):
        r = dict(doc)
        r["ok"] = ok
        r["md"] = md
        mf.write(json.dumps(r, ensure_ascii=False) + "\n")

    def persist():
        state["done"] = list(done)
        if l0 is not None:
            l0.persist()
        tmp = state_path + ".tmp"
        json.dump(state, open(tmp, "w"), ensure_ascii=False)
        os.replace(tmp, state_path)
        mf.flush()

    for doc in docs:
        u = doc["url"]
        if u in done:
            continue
        if args.limit and new >= args.limit:
            print(f"--limit {args.limit} reached (resumable)", flush=True)
            break
        pq = urllib.parse.urlsplit(u)
        if not rb.allowed(pq.path + (("?" + pq.query) if pq.query else "")):
            done.add(u)
            continue
        new += 1
        state["fetched"] += 1
        ps = state["per_source"].setdefault(doc["source"], {})
        try:
            r = client.get(u)
            ctype = r.headers.get("content-type", "").lower()
            if r.status_code != 200 or "pdf" not in ctype:
                state["http_err"] += 1
                ps["http_err"] = ps.get("http_err", 0) + 1
                row(doc, False, "")
                done.add(u)
            else:
                if l0 is not None:
                    l0.put(u, r.content, final_url=str(r.url),
                           status=r.status_code, content_type=ctype, kind="pdf")
                name = os.path.basename(pq.path) or "doc.pdf"
                try:
                    md = pdf_to_markdown(conv, r.content, name)
                except Exception as e:  # noqa: BLE001
                    md = ""
                    state["docling_err"] += 1
                    print(f"  ! docling {u}: {str(e)[:100]}", flush=True)
                if md and len(md) >= 120:
                    row(doc, True, md)
                    state["docs_ok"] += 1
                    ps["docs_ok"] = ps.get("docs_ok", 0) + 1
                else:
                    row(doc, False, "")
                    state["docs_empty"] += 1
                    ps["docs_empty"] = ps.get("docs_empty", 0) + 1
                done.add(u)
        except Exception as e:  # noqa: BLE001
            print(f"  ! fetch {u}: {str(e)[:100]}", flush=True)
            # transient -> not marked done, a resume retries it
        time.sleep(eff)
        if args.hb_secs and time.time() - last[0] > args.hb_secs:
            print(f"  [hb] fetched={state['fetched']} ok={state['docs_ok']}",
                  flush=True)
            last[0] = time.time()
        if new % 50 == 0:
            persist()
            rate = new / max(time.time() - t0, 1)
            print(f"  {new} fetched  ok={state['docs_ok']} "
                  f"empty={state['docs_empty']}  {rate*3600:.0f}/h", flush=True)
            last[0] = time.time()

    persist()
    mf.close()
    if l0 is not None:
        drained = all(d["url"] in done for d in docs)
        l0.close(complete=drained)
        print(f"[l0] {'COMPLETE' if drained else 'partial'} — {l0.records} "
              f"records, {l0.body_bytes/1e9:.2f} GB", flush=True)
    client.close()
    print("EXTRACT DONE", flush=True)
    print(f"  fetched={state['fetched']} docs_ok={state['docs_ok']} "
          f"docs_empty={state['docs_empty']} http_err={state['http_err']} "
          f"docling_err={state['docling_err']}", flush=True)
    print(f"  per_source={json.dumps(state['per_source'])}", flush=True)


# ============================================================================
# STAGE: gate  (hygiene venv — fp.chunk + fasttext LID 166 + gates)
# ============================================================================
_COMMON = dict(domain="finma.ch", tier="A", source_type="official", trust_tier=0.95)


def new_stats() -> dict:
    return {"docs": 0, "docs_with_passages": 0, "meta_fallback": 0,
            "passages": 0, "capped": 0, "rejects": {}, "apparatus": {},
            "per_source": {}, "per_lang": {}}


def _bump(d, k, n=1):
    d[k] = d.get(k, 0) + n


def md_passages(doc, stats, max_chunks, last_seen=None):
    """Docling-markdown passages of one document (Title + Description as the H1), or [] when the
    body is empty, junk, low quality or out of language scope (the caller then falls back to
    `meta_passage`)."""
    from cytir.acquire.crawl import is_apparatus
    from cytir.core.common import (extraction_junk, extractive_snippet, low_quality,
                                passage_from_chunk)
    from cytir.acquire import fetch_process as fp
    common = dict(_COMMON, last_seen=last_seen or TODAY)
    md = doc.get("md") or ""
    head = f"# {doc['title']}\n\n"
    if doc.get("description"):
        head += f"{doc['description']}\n\n"
    full = head + md
    if len(full) < 120 or low_quality(full) or extraction_junk(full):
        return []
    lang, conf = fp.detect_lang(md or full, doc["url"], doc["lang"])
    if conf and lang not in fp.LANG_SCOPE:
        _bump(stats["rejects"], "lang_out_of_scope")
        return []
    ptitle = (doc["title"] or "").strip() or doc["url"]
    out = []
    for unit in chunk_generic_records(full):
        ch = unit.text
        if low_quality(ch):
            continue
        if is_apparatus(ch):
            _bump(stats["apparatus"], "bibliography")
            continue
        rr = extraction_junk(ch)
        if rr:
            _bump(stats["rejects"], rr)
            continue
        out.append(passage_from_chunk(
            unit, url=doc["url"], title=ptitle,
            snippet=extractive_snippet(ch), lang=lang, **common))
    if len(out) > max_chunks:
        stats["capped"] += len(out) - max_chunks
        out = out[:max_chunks]
    return out


def meta_passage(doc, last_seen=None):
    """The ONE metadata record (Title + Description) of a document with no usable body."""
    from cytir.core.common import Passage, extractive_snippet
    common = dict(_COMMON, last_seen=last_seen or TODAY)
    text = doc["title"] + (". " + doc["description"]
                           if doc.get("description") else "")
    return Passage(url=doc["url"], title=doc["title"],
                   snippet=extractive_snippet(text), text=text,
                   lang=doc["lang"], **common)


def doc_passages(doc, stats, max_chunks=50, last_seen=None):
    """All passages of one document: the body chunks, else the metadata fallback."""
    psrc = stats["per_source"].setdefault(doc["source"], {})
    stats["docs"] += 1
    pl = md_passages(doc, stats, max_chunks, last_seen) if doc.get("ok") else []
    if pl:
        stats["docs_with_passages"] += 1
        _bump(psrc, "docs_ok")
    else:
        pl = [meta_passage(doc, last_seen)]
        stats["meta_fallback"] += 1
        _bump(psrc, "meta_fallback")
    for p in pl:
        _bump(stats["per_lang"], p.lang)
    stats["passages"] += len(pl)
    _bump(psrc, "passages", len(pl))
    return pl


def passage_record(p) -> dict:
    """One passage as the corpus record this family writes (provenance `finma_circulars`)."""
    from cytir.core.common import passage_corpus_record
    rec = passage_corpus_record(p)
    rec.pop("content_digest", None)
    rec["provenance"] = PROV
    return rec


def stage_gate(args):
    D = args.out_dir
    md_path = os.path.join(D, "docs_md.jsonl")
    passages_path = os.path.join(D, "passages.jsonl")
    stats = new_stats()
    pf = open(passages_path, "w", encoding="utf-8")
    for line in open(md_path, encoding="utf-8"):
        if not line.strip():
            continue
        for p in doc_passages(json.loads(line), stats, args.max_chunks):
            pf.write(json.dumps(passage_record(p), ensure_ascii=False) + "\n")
    pf.close()
    json.dump(stats, open(os.path.join(D, "gate_stats.json"), "w"), indent=2)
    print("GATE DONE", flush=True)
    print(f"  docs={stats['docs']} passages={stats['passages']} "
          f"docs_with_passages={stats['docs_with_passages']} "
          f"meta_fallback={stats['meta_fallback']} capped={stats['capped']}",
          flush=True)
    print(f"  per_source={json.dumps(stats['per_source'])}", flush=True)
    print(f"  per_lang={json.dumps(stats['per_lang'])}", flush=True)
    print(f"  rejects={json.dumps(stats['rejects'])} "
          f"apparatus={json.dumps(stats['apparatus'])}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--out-dir", required=True)
    e.add_argument("--l0-root", default=os.environ.get("AW_L0_ROOT"),
                   help="L0 store root (default: $CYTIR_L0_ROOT); required unless --no-l0")
    e.add_argument("--delay", type=float, default=1.5)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--only", default="")
    e.add_argument("--no-l0", action="store_true")
    e.add_argument("--hb-secs", type=float, default=240.0)
    e.set_defaults(func=stage_extract)

    g = sub.add_parser("gate")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--max-chunks", type=int, default=50)
    g.set_defaults(func=stage_gate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
