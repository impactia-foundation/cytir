"""`technical_manufacturer_catalog` -- manufacturer product catalogs as a declared door (task 1401).

Route `direct`, consent live per origin (robots.txt + TDMRep), licence from the instrument of each
origin: all three are read from the family's contract row, never written here (`discover` copies the
row's terms into the run notes). The code is `cytir.sources.technical_manufacturer.catalog`, moved from
the flat acquire script; this adapter DELEGATES to its steps and copies none of them.

STAGE MAP (the unit is one seed domain, `domain:<host>`):

    discover      `catalog.load_seeds`: the sealed seed frontier (receipt-checked); one unit per domain
    consent       FRAMEWORK: the domain's host, resolved live per origin (`plan` stops here)
    acquire       `catalog.acquire_domain`: agents.md, well-known UCP, root sitemap, qualification,
                  product JSON, canonical product pages; the domain receipt is the captured body
    seal          `finalize` (Python only, like news_daily.compose_day): `catalog.seal_run` --
                  SKU dedup, inventories, outlinks, the `original` and `outlinks` stage receipts, RUN.json

EXTRACT AND CHUNK (task 2138): `extract` is the product Markdown projection
(`cytir.sources.technical_manufacturer.product_markdown.render_product`, ported verbatim from the deleted
`materialize_technical_product_catalog.py` that sealed `technical-product-markdown.v1`). Its input is one
admitted product page: a `Captured` whose `body` is the preserved page bytes and whose `meta["decision"]`
is the admission decision row (`admitted-products.jsonl`); `acquire` above yields domain receipts, not
pages, so the page-level Captured comes from the preserved run. `chunk` runs the structured chunker under
the contract's DECLARED policy (`7dcb54c4...@4`), anchor `document:<id>:chunk-<n>`, as the sealed pack's
chunk stage does; byte-equal on sealed rows.

WHAT STAYS OUT: the fidelity / token / UUID gates and the pack are the downstream audit and admission scripts (`audit_technical_product_*`),
which read the sealed run; the former shell chains `chain_technical_*.sh` (deleted task 1966; pins git-blob:20b44cd50b2b80b6ddecddf40150e07148b6d671, 5d6103ae02b47161271f30f14e17f3730b1a2023, 2b2f118769e3418d9d226307e8f6d62d7264d1cd, 9930eb7b38a713812d8735214bad75b38417a9a1: the four chain_technical_*.sh files) were the operator's wait-and-run
glue between those scripts. No row is composed here.

Options (`--opt KEY=VALUE`): `frontier`, `frontier_receipt` (required: the sealed seed frontier and
its receipt); `root`, `artifact_root` (required by `acquire`: a fresh run root inside the artifact
root); `rate` (requests per second, default 1), `max_products_per_domain` (1000),
`max_technical_pages_per_domain` (200).
"""
from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from urllib.parse import urlsplit

from cytir.sources.chunk_selection import declared_policy_selection
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit
from cytir.sources.registry import register_family


def contract_terms(ctx: RunContext) -> dict:
    """The consent and licence terms of the family's contract row, as the run notes carry them."""
    row = ctx.contract
    if not row or "consent" not in row or "licence" not in row:
        raise ValueError("technical_manufacturer_catalog: the run has no contract row with "
                         "consent and licence")
    consent, licence = row["consent"], row["licence"]
    return {"search_ok": consent.get("search_ok"), "train_ok": consent.get("train_ok"),
            "consent_resolution": consent.get("resolution"),
            "licence_basis": licence.get("basis"), "licence_instrument": licence.get("instrument")}


def _need(ctx: RunContext, key: str) -> pathlib.Path:
    value = ctx.options.get(key)
    if not value:
        raise ValueError(f"technical_manufacturer_catalog needs --opt {key}=<path>")
    return pathlib.Path(value).resolve()


@register_family
class TechnicalManufacturerCatalogFamily:
    family_id = "technical_manufacturer_catalog"
    route = "direct"
    direct_crawl = True
    chunk_policy_id = CHUNK_DECLARED

    def __init__(self):
        self._fetcher = None
        self._state = {"qualified": [], "product_rows": [], "admitted": [], "receipts": []}

    def discover(self, ctx: RunContext):
        from cytir.sources.technical_manufacturer import catalog
        ctx.notes["contract_terms"] = contract_terms(ctx)
        seeds = catalog.load_seeds(_need(ctx, "frontier"), _need(ctx, "frontier_receipt"))
        ctx.notes["seed_domains"] = len(seeds)
        self._seeds = seeds
        return [Unit(key=f"domain:{seed['domain']}", url=f"https://{seed['domain']}/",
                     consent_hosts=(seed["domain"],), meta={"seed": seed}) for seed in seeds]

    def _fetcher_for(self, ctx: RunContext):
        from cytir.sources.technical_manufacturer import catalog
        if self._fetcher is None:
            root, artifact_root = _need(ctx, "root"), _need(ctx, "artifact_root")
            root.mkdir(parents=True, exist_ok=True)
            self._root, self._artifact_root = root, artifact_root
            self._fetcher = catalog.Fetcher(root=root, artifact_root=artifact_root,
                                            rate=float(ctx.options.get("rate", 1.0)),
                                            client=ctx.http)
        return self._fetcher

    def acquire(self, unit: Unit, ctx: RunContext):
        from cytir.sources.technical_manufacturer import catalog
        fetcher = self._fetcher_for(ctx)
        one = catalog.acquire_domain(
            fetcher, unit.meta["seed"], root=self._root,
            max_products_per_domain=int(ctx.options.get("max_products_per_domain", 1000)),
            max_technical_pages_per_domain=int(
                ctx.options.get("max_technical_pages_per_domain", 200)))
        st = self._state
        if one["qualified"]:
            st["qualified"].append(unit.meta["seed"]["domain"])
        st["product_rows"].extend(one["product_rows"])
        st["admitted"].extend(one["admitted"])
        st["receipts"].append(one["receipt_path"])
        body = one["receipt_path"].read_bytes()
        return [Captured(unit, body, {"status": 200, "content_type": "application/json",
                                      "source_url": unit.url, "qualified": one["qualified"],
                                      "admitted": len(one["admitted"])})]

    def extract(self, cap: Captured, ctx: RunContext):
        from cytir.sources.technical_manufacturer import product_markdown as PM
        decision = cap.meta.get("decision")
        if not decision:
            raise ValueError("technical_manufacturer_catalog.extract needs the admission decision "
                             "row in Captured.meta['decision'] and the page bytes as body")
        row = PM.render_product(decision=decision, body=cap.body)
        return [Doc(url=row["url"], text=row["markdown"], title=row["title"], lang=row["lang"],
                    extra={"row": row})]

    def preflight(self, ctx: RunContext) -> None:
        self._selection = declared_policy_selection(ctx)

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch
        selection = getattr(self, "_selection", None)
        if selection is None:
            raise RuntimeError("technical_manufacturer_catalog.chunk ran before preflight: the "
                               "declared chunk policy is resolved in preflight")
        row, out = doc.extra["row"], []
        stamps = selection.stamps()
        digest = C.sha1(doc.text)
        parent_sha = hashlib.sha256(doc.text.encode()).hexdigest()
        domain = urlsplit(doc.url).hostname or ""
        for ordinal, unit in enumerate(_dispatch.chunk_markdown_channels(doc.text, selection=selection)):
            unit = dataclasses.replace(unit, stamps=stamps)
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(unit.dense_text or unit.text, 25),
                lang=doc.lang, domain=domain, tier=None, source_type=None,
                trust_tier=None, last_seen=ctx.today, content_digest=digest,
                anchor=f"document:{row['document_id']}:chunk-{ordinal}")
            rec = C.passage_source_record(p)
            rec["document_id"], rec["chunk_ordinal"] = row["document_id"], ordinal
            rec["source_record_id"] = row["source_record_id"]
            rec["source_profile"] = row["source_profile"]
            rec["parent_identity"] = row["parent_identity"]
            rec["source_links"] = row["source_links"]
            rec["parent_markdown_sha256"] = parent_sha
            rec["source_metadata"] = {"product": row["product"], "publisher": row["publisher"],
                                      "title": row["title"]}
            out.append(rec)
        return out

    def finalize(self, ctx: RunContext) -> dict:
        """Seal the acquired domains (`catalog.seal_run`). Reachable from Python only;
        `run_family` does not call this stage."""
        from cytir.sources.technical_manufacturer import catalog
        st = self._state
        try:
            return catalog.seal_run(
                frontier=_need(ctx, "frontier"), frontier_receipt=_need(ctx, "frontier_receipt"),
                root=self._root, artifact_root=self._artifact_root, seeds=self._seeds,
                qualified=st["qualified"], product_rows=st["product_rows"],
                admitted=st["admitted"], domain_receipts=st["receipts"])
        finally:
            if self._fetcher is not None:
                self._fetcher.close()
