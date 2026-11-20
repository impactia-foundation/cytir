"""The Federal Register leg of `us_federal_register_ecfr` (task 699, contract row of task 695).

Route: the federalregister.gov documents API and the per-document text URL the API itself returns
(`raw_text_url`), one document per GET, one connection, <= 5 req/s (the contract's
`acquisition.rate_limit`; the default here is 2: the server queues above that and a higher rate gives LESS: the growth lane's pace measurement of 2026-10-03). No HTML crawl of the site, no govinfo bulk route
(not declared). Public domain (17 U.S.C. 105).

Units: one calendar month (`fr-YYYY-MM`): the month's document list (`per_page=1000`, cursor
pages) and then the text of every document of the month. A month whose list holds 10,000 documents
refuses (the API caps a query there; a truncated month must never look complete). Document level
terminals are counted in the manifest notes (`documents`: `admitted` + `excluded_*` = `seen`).
Options: `from`, `to` (ISO days; default 1994-01-01 .. today), `max_req_s`.

A record states its Federal Register citation, type, agencies and publication date; the canonical
URL is the document's own federalregister.gov page (a pointer, never fetched).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import re

from cytir.sources import api_fetch as AF
from cytir.sources.family import Captured, Doc, RunContext, Unit, UnitExcluded

LICENSE = "public-domain"
HOST = "www.federalregister.gov"
LIST_URL = "https://www.federalregister.gov/api/v1/documents.json"
FIRST_DAY = "1994-01-01"
DEFAULT_MAX_REQ_S = 2.0           # measured 2026-10-03: the server queues above ~2 req/s (the growth lane's pace measurement)
MIN_BODY_WORDS = 30
API_QUERY_CAP = 10_000
FIELDS = ("document_number", "title", "abstract", "type", "publication_date", "raw_text_url",
          "html_url", "citation", "agencies")

# THE ROUTE SPLIT (orchestrator D1, 2026-10-03). The family has two routes with two text surfaces: the
# per-document API (HTML-derived text, `ROUTE_API`) and the govinfo bulk issue XML (`ROUTE_BULK`, in
# `federal_register_govinfo`). Point ids hash the chunk text and the two surfaces differ, so the SAME
# document gets different ids on the two routes; one document must therefore never be written by both. A
# document's PUBLICATION DATE decides its route: the bulk route owns `BULK_DATES` (2000-01-01 ..
# 2025-11-30), the API route owns every other date (1994-1999 and 2025-12-01 on). Each leg refuses a
# document dated for the other route and counts it (`route_date_bulk` / `route_date_api`); the effective
# split is written into the manifest notes (`route_split`). `--opt bulk_dates=FROM..TO` declares another
# split, `--opt bulk_dates=none` declares that the API route owns every date (a replay: the stage-level
# URL guard still drops anything served).
ROUTE_API = "federalregister-api"
ROUTE_BULK = "govinfo-bulk-xml"
BULK_DATES = ("2000-01-01", "2025-11-30")


def bulk_dates(options: dict) -> tuple[str, str] | None:
    """The date range the bulk route owns, from `--opt bulk_dates=` (default `BULK_DATES`); None = none."""
    raw = options.get("bulk_dates")
    if raw is None:
        return BULK_DATES
    if str(raw).lower() in ("none", "off"):
        return None
    a, _, b = str(raw).partition("..")
    dt.date.fromisoformat(a), dt.date.fromisoformat(b)
    return a, b


def route_for(date: str, options: dict) -> str:
    """The route that owns a publication date (ISO)."""
    rng = bulk_dates(options)
    return ROUTE_BULK if rng and rng[0] <= date[:10] <= rng[1] else ROUTE_API


# NOTICES THAT NAME PRIVATE PERSONS (Tim's ruling 24, 2026-10-03; orchestrator D3 + the 19:15Z addition).
# Held at extract IN DOUBT until Tim rules, on BOTH routes, counted `held_personal_data_<class>`. A held
# document is never chunked: no row, snippet or embedding. The rule reads the API's structured fields
# (type, agency names, title), never the body:
#   debarment_notice        type Notice, "debarment" in the title: FCC E-Rate debarments (a party is named), FCC
#                           suspension proceedings, FDA "<Name>: Final Debarment Order" (the person IS the title)
#   dea_decision_and_order  agency Drug Enforcement Administration, type Notice, an order about a registrant in the title
#                           ("Jane Roe, M.D.; Decision and Order", Rescission, Order to Show Cause, Revocation)
#   bis_export_denial_order agency Industry and Security Bureau, type Notice, an order about a named party: "In the
#                           Matter of" or "Order ... Denying/Renewing ... Export Privileges" in the title (persons
#                           after a conviction, and companies; the title cannot tell them apart: class held whole)
#   ofac_sanctions_notice   agency Foreign Assets Control (Treasury) or State Department, type Notice, "Sanctions
#                           Action" in the title (designations of individuals AND entities/vessels; the title cannot
#                           tell them apart: class held whole)
# A Rule or Proposed Rule is never held (only Notices). Measured on the 6,696 documents of w116 + w117 and the
# near-misses of the same agencies (the held-classes fixture of the family tests).
_DEBARMENT = re.compile(r"\bdebarment\b", re.I)
_DEA_ORDER = re.compile(r"decision and order|rescission of|order to show cause|revocation of|surrender of|"
                        r"denial of (registration|application)", re.I)
_BIS_ORDER = re.compile(r"in the matter of|decision and order|(denying|denial of|renewing)[^;]{0,60}export privileges|"
                        r"order (renewing|denying)", re.I)


def held_notice_reason(doc_type: str | None, title: str | None, agencies=None) -> str | None:
    """`held_personal_data_<class>` for a held document, or None. Shared by every route."""
    if (doc_type or "") != "Notice":
        return None
    t = title or ""
    names = " | ".join((a.get("name") or a.get("raw_name") or "") if isinstance(a, dict) else str(a)
                       for a in (agencies or [])).lower()
    if _DEBARMENT.search(t):
        return "held_personal_data_debarment_notice"
    if "drug enforcement administration" in names and _DEA_ORDER.search(t):
        return "held_personal_data_dea_decision_and_order"
    if "industry and security" in names and _BIS_ORDER.search(t):
        return "held_personal_data_bis_export_denial_order"
    if ("foreign assets control" in names or "state department" in names) and "sanctions action" in t.lower():
        return "held_personal_data_ofac_sanctions_notice"
    return None


_PRE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_HEAD_LINE = re.compile(r"^\s*(\[.*\]|From the Federal Register Online.*)\s*$")   # first lines only
_FOOT_LINE = re.compile(r"^\s*(\[\[Page \d+\]\]|BILLING CODE.*|\[FR Doc\. .*\])\s*$")
_RULE = re.compile(r"^\s*[=\-_]{8,}\s*$")
_BANNER_RULE = re.compile(r"^\s*={20,}\s*$")        # the GPO section banner of a printed page
_RUNNING_HEAD = re.compile(r"Federal Register\s*/\s*Vol\.\s*\d+,\s*No\.\s*\d+")
_HEAD_TAIL = re.compile(r"^\s*(Rules|Regulations|Documents|and Regulations|Rules and Regulations)\s*$")
_WS = re.compile(r"\s+")


def raw_text_to_body(raw: bytes) -> str:
    """The text of one Federal Register document from its `raw_text_url` body (an HTML page that
    wraps the GPO text in `<pre>`): tags and the GPO header/footer lines dropped, hard-wrapped
    lines joined into paragraphs."""
    page = raw.decode("utf-8", errors="replace")
    m = _PRE.search(page)
    text = html.unescape(_TAG.sub("", m.group(1) if m else page)).replace("\xa0", " ").replace("\x00", "")
    lines = text.splitlines()
    rules = [i for i, ln in enumerate(lines[:60]) if _BANNER_RULE.match(ln)]
    if len(rules) >= 2:                      # "Proposed Rules / Federal Register / This section of ..."
        del lines[rules[0]:rules[1] + 1]
    kept, skip_tail = [], False
    for ln in lines:
        if _RUNNING_HEAD.search(ln) and len(ln) < 160:      # the page header, wrapped onto a 2nd line
            skip_tail = True
            continue
        if skip_tail and _HEAD_TAIL.match(ln):
            skip_tail = False
            continue
        skip_tail = False
        kept.append(ln)
    text = "\n".join(kept)
    paras, cur = [], []
    for i, line in enumerate(text.splitlines()):
        if _RULE.match(line) or _FOOT_LINE.match(line) or (i < 12 and _HEAD_LINE.match(line)):
            continue
        if not line.strip():
            if cur:
                paras.append(" ".join(cur))
                cur = []
            continue
        cur.append(_WS.sub(" ", line.strip()))
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(p for p in paras if p)


def _list_params(gte: str, lte: str, per_page: int) -> list[tuple[str, object]]:
    return ([("per_page", per_page), ("order", "oldest"),
             ("conditions[publication_date][gte]", gte),
             ("conditions[publication_date][lte]", lte)]
            + [("fields[]", f) for f in FIELDS])


def count_documents(client, pacer: AF.Pacer, gte: str, lte: str) -> int:
    resp = AF.get(client, LIST_URL, pacer, params=_list_params(gte, lte, 1))
    resp.raise_for_status()
    return int(resp.json().get("count", 0))


def list_documents(client, pacer: AF.Pacer, gte: str, lte: str) -> list[dict]:
    """Every document published in [gte, lte]; refuses a query the API would truncate."""
    resp = AF.get(client, LIST_URL, pacer, params=_list_params(gte, lte, 1000))
    resp.raise_for_status()
    doc = resp.json()
    total = int(doc.get("count", 0))
    if total >= API_QUERY_CAP:
        raise RuntimeError(f"federal register {gte}..{lte}: {total} documents is at the API's "
                           f"{API_QUERY_CAP} cap; split the range, never take a truncated list")
    out = list(doc.get("results") or [])
    while doc.get("next_page_url"):
        resp = AF.get(client, doc["next_page_url"], pacer)
        resp.raise_for_status()
        doc = resp.json()
        out.extend(doc.get("results") or [])
    if len(out) != total:
        raise RuntimeError(f"federal register {gte}..{lte}: the list holds {len(out)} documents, "
                           f"the API said {total}")
    return out


def plan_windows(client, pacer: AF.Pacer, options: dict, today: str, *, cap: int = 25_000
                 ) -> list[dict]:
    """Month bins counted by the API (one request each), packed into windows of <= `cap`."""
    first = dt.date.fromisoformat(options.get("from") or FIRST_DAY)
    last = dt.date.fromisoformat(options.get("to") or today)
    bins = [(a, b, count_documents(client, pacer, a, b)) for a, b in AF.month_bins(first, last)]
    return AF.pack_windows(bins, cap)


class FederalRegisterLeg:
    """The `leg=fr` behaviour of the family; `UsFederalRegisterEcfrFamily` delegates here."""

    def __init__(self, family) -> None:
        self.family = family

    def _pacer(self, ctx: RunContext) -> AF.Pacer:
        if not hasattr(ctx, "_fr_pacer"):
            ctx._fr_pacer = AF.Pacer(float(ctx.options.get("max_req_s", DEFAULT_MAX_REQ_S)))
        return ctx._fr_pacer

    def discover(self, ctx: RunContext):
        first = dt.date.fromisoformat(ctx.options.get("from") or FIRST_DAY)
        last = dt.date.fromisoformat(ctx.options.get("to") or ctx.today)
        units = [Unit(key=f"fr-{a[:7]}", url=LIST_URL, consent_hosts=(HOST,),
                      meta={"gte": a, "lte": b}) for a, b in AF.month_bins(first, last)]
        rng = bulk_dates(ctx.options)
        ctx.notes.update({"leg": "fr", "months": len(units), "from": first.isoformat(),
                          "to": last.isoformat(), "route": ROUTE_API,
                          "route_split": {"bulk_dates": list(rng) if rng else None, "this_route": ROUTE_API},
                          "documents": {"seen": 0, "admitted": 0}})
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        client, pacer = ctx.http, self._pacer(ctx)
        docs = list_documents(client, pacer, unit.meta["gte"], unit.meta["lte"])
        caps = []
        for d in docs:
            ctx.notes["documents"]["seen"] += 1
            if route_for(d["publication_date"], ctx.options) != ROUTE_API:
                self._excluded(ctx, "route_date_bulk")        # the bulk route's date: no text GET
                continue
            resp = AF.get(client, d["raw_text_url"], pacer)
            if resp.status_code != 200:
                self._excluded(ctx, f"text_{resp.status_code}")
                continue
            meta = {"status": 200, "source_url": d["raw_text_url"],
                    "sha256": hashlib.sha256(resp.content).hexdigest(),
                    "doc": {k: d.get(k) for k in FIELDS if k != "raw_text_url"}}
            caps.append(Captured(Unit(key=f"fr-{d['document_number']}", url=d["raw_text_url"],
                                      meta={}), resp.content, meta))
        if not caps:
            raise UnitExcluded("month_without_documents")
        return caps

    @staticmethod
    def _excluded(ctx: RunContext, reason: str) -> None:
        notes = ctx.notes["documents"]
        notes[reason] = notes.get(reason, 0) + 1

    def extract(self, cap: Captured, ctx: RunContext):
        d = cap.meta["doc"]
        if route_for(d["publication_date"], ctx.options) != ROUTE_API:
            self._excluded(ctx, "route_date_bulk")
            return []
        if (why := held_notice_reason(d.get("type"), d.get("title"), d.get("agencies"))):
            self._excluded(ctx, why)
            return []
        body = raw_text_to_body(cap.body)
        if len(body.split()) < MIN_BODY_WORDS:
            self._excluded(ctx, "body_under_word_floor")
            return []
        agencies = ", ".join(a.get("name") or a.get("raw_name") or "" for a in d.get("agencies") or []
                             if isinstance(a, dict)).strip(", ")
        title = (d.get("title") or d["document_number"]).strip()
        where = (f"Federal Register {d.get('citation') or d['document_number']} "
                 f"({d.get('type') or 'document'}), published {d['publication_date']}"
                 + (f". Agencies: {agencies}" if agencies else "") + ".")
        ctx.notes["documents"]["admitted"] += 1
        return [Doc(url=d["html_url"], text=f"# {title}\n\n{where}\n\n{body}", title=title[:300],
                    lang="en", extra={"body": body})]

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch

        rows, ids = [], set()
        for unit in _dispatch.chunk_markdown_channels(doc.text):
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(doc.extra["body"], 25), lang="en",
                domain="federalregister.gov", tier="A", source_type="official", trust_tier=1.0,
                last_seen=ctx.today, license=LICENSE,
                master={"route": getattr(self, "ROUTE", ROUTE_API)})
            rec = C.passage_source_record(p)
            if rec["point_id"] in ids:
                # an identical chunk of the same document (the govinfo XML repeats a block in 2 of the
                # 31k documents of 2000): same url, same text, same id. A second row would fail the
                # whole year as `duplicate_point_id`; the first stays, the drop is counted.
                ctx.notes["chunks_dropped_identical_in_document"] = (
                    ctx.notes.get("chunks_dropped_identical_in_document", 0) + 1)
                continue
            ids.add(rec["point_id"])
            rows.append(rec)
        return rows
