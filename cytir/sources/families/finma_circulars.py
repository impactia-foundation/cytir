"""`finma_circulars` -- FINMA document-center PDFs on the family Protocol (task 1396, task 643 S7).

The family of task 209: the four FINMA document-center widgets (Rundschreiben, Aufsichtsmitteilungen,
Versicherungsrechtliche Entscheide, Stellungnahmen), 2,956 PDFs on `www.finma.ch`, listed by the
site's own search API and read with Docling. The old driver (`ingest_finma_circulars.py`, now a
shim) moved to `cytir.sources.finma.circulars`; this adapter DELEGATES to it, so the standalone
`extract`/`gate` stages and the Protocol run share one body of logic.

STAGE MAP
  discover  delegated  `circulars.enumerate_source` per widget (POST `/de/api/search/getresult`),
                       one unit per canonical PDF URL, de-duplicated across widgets.
  acquire   adapter    GET the PDF through `ctx.http`; the framework seals the bytes (L0). A 4xx or
                       non-PDF answer is not an error: the document keeps its metadata record, as
                       the old `extract` stage did. A 5xx or a transport failure raises (QUARANTINED).
  extract   delegated  `circulars.pdf_to_markdown` (Docling, OCR off); a body under 120 characters
                       is the same "no usable body" case.
  chunk     delegated  `circulars.doc_passages` (Title + Description H1, `chunk_generic_records`,
                       gates 097/166/175, 50-chunk head cap, else ONE metadata record).

CONSENT AND LICENCE are read from the contract row. `discover` refuses a row without `search_ok`
and copies the row's consent resolution and licence basis into the run notes. Consent is live per
origin: the framework resolves `www.finma.ch` (robots.txt, TDMRep, AIPREF) before `acquire`.

KNOWN GAPS (all four stages run; no `l1()` yet, ADR-105 am.1 makes it optional):
  * `source_text` is not given, so the stage-time text guard counts every document `unchecked`
    (the 50-chunk head cap is a cap, not a declared transformation; giving the text would
    quarantine every capped document). Follow-up: declare the cap in the contract.
  * the widget API responses are not sealed as L0 by this adapter (the old driver's `L0Store`
    wrote them with `kind=api`); only the PDF bodies are. Follow-up: a Protocol-level hook.
  * robots.txt is resolved per host by the framework, not per path as the old `extract` did;
    the contract records that the `/<lang>/~/media` PDF paths are allowed.
Options (`--opt KEY=VALUE`): `only` (comma-separated widget keys), `delay` (seconds between API
pages, default 1.5), `max_chunks` (default 50).
"""
from __future__ import annotations

import time

from cytir.sources.family import CHUNK_DECLARED, HttpSpec, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.finma import circulars as C
from cytir.sources.registry import register_family

MIN_BODY_CHARS = 120      # the old `extract` stage: a shorter Docling body is "empty"


def contract_terms(ctx: RunContext, family_id: str = "finma_circulars") -> dict:
    """The consent and licence terms of the family's contract row, as the run notes carry them."""
    row = ctx.contract
    if not row or "consent" not in row or "licence" not in row:
        raise ValueError(f"{family_id}: the run has no contract row with consent and licence")
    consent, licence = row["consent"], row["licence"]
    if consent.get("search_ok") is not True:
        raise ValueError(f"{family_id}: the contract does not declare search_ok; the family "
                         "does not run")
    return {"search_ok": consent["search_ok"], "train_ok": consent.get("train_ok"),
            "consent_resolution": consent.get("resolution"),
            "licence_basis": licence.get("basis"), "licence_spdx": licence.get("spdx")}


def _converter(ctx: RunContext):
    conv = getattr(ctx, "_finma_converter", None)
    if conv is None:
        conv = ctx._finma_converter = C.make_converter()
    return conv


class _FinmaBase:
    """The one body of the FINMA document-center family; `FinmaCircularsFamily` and
    `FinmaExactFamily` differ only in identity and in what each declares as not ported."""
    family_id = ""
    route = "direct"
    direct_crawl = True
    chunk_policy_id = CHUNK_DECLARED
    http_spec = HttpSpec(timeout=90, user_agent_kind="pack_fetch")   # task 1556: the runner builds and closes the client
    require_records = True

    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = contract_terms(ctx, self.family_id)
        only = {k for k in (ctx.options.get("only") or "").split(",") if k}
        delay = float(ctx.options.get("delay", 1.5))
        client, seen, units, per_source = ctx.http, set(), [], {}
        for key, ds, multilang in C.SOURCES:
            if only and key not in only:
                continue
            got = C.enumerate_source(client, key, ds, multilang, None, delay)
            per_source[key] = len(got)
            for d in got:
                if d["url"] in seen:
                    continue
                seen.add(d["url"])
                units.append(Unit(key=d["url"], url=d["url"], meta=dict(d)))
            time.sleep(delay)
        if not units:
            raise RuntimeError(f"{self.family_id}: the document center listed no PDF; an empty "
                               "walk is a broken enumeration, not an empty source")
        ctx.notes.update({"pdf_targets": len(units), "per_widget": per_source})
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        r = ctx.http.get(unit.url)
        if r.status_code >= 500:
            r.raise_for_status()
        ctype = r.headers.get("content-type", "").lower()
        return [Captured(unit, r.content, {
            "status": r.status_code, "content_type": ctype, "source_url": unit.url,
            "final_url": str(r.url)})]

    def extract(self, cap: Captured, ctx: RunContext):
        doc = dict(cap.unit.meta)
        md = ""
        if cap.meta["status"] == 200 and "pdf" in cap.meta["content_type"]:
            name = cap.unit.url.rsplit("/", 1)[-1].split("?")[0] or "doc.pdf"
            md = C.pdf_to_markdown(_converter(ctx), cap.body, name)
        ok = len(md) >= MIN_BODY_CHARS
        doc.update(ok=ok, md=md if ok else "")
        return [Doc(url=doc["url"], text=doc["md"], title=doc["title"], lang=doc["lang"],
                    extra={"doc": doc})]

    def chunk(self, doc: Doc, ctx: RunContext):
        stats = C.new_stats()
        max_chunks = int(ctx.options.get("max_chunks", 50))
        passages = C.doc_passages(doc.extra["doc"], stats, max_chunks, last_seen=ctx.today)
        if not passages:
            raise UnitExcluded("no_passage")
        return [C.passage_record(p) for p in passages]


@register_family
class FinmaCircularsFamily(_FinmaBase):
    family_id = "finma_circulars"


@register_family
class FinmaExactFamily(_FinmaBase):
    """`finma_exact` (task 2171): the retained-PDF release of the same 2,956 document-center PDFs.

    Its producer is `cytir.ops.finma_exact_admission` (pack -> profile audit -> seals -> stage), an ops
    chain under the release's `chunk-policy-struct-md` policy; the contract row is `backfilled` with
    route, consent, licence and chunk policy all `unknown`. So:
      discover / acquire / extract  the same delegated code as `finma_circulars`;
      chunk       PARTIAL: the exact release chunks with `cytir.corpus.stage_structured_markdown`
                  under its own policy, not `circulars.doc_passages`; this adapter does not chunk.
      consent     `contract_terms` refuses the run while the row's `search_ok` is not true.
    Nothing here changes a consent, licence, route or tier value."""
    family_id = "finma_exact"
    PARTIAL = ("chunk",)

    def chunk(self, doc: Doc, ctx: RunContext):
        raise NotImplementedError("finma_exact.chunk is PARTIAL: the exact release is chunked by "
                                  "cytir.corpus.stage_structured_markdown (cytir.ops.finma_exact_admission)")
