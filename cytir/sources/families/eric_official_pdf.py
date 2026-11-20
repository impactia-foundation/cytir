"""`eric_official_pdf` -- ERIC full-text PDFs: the inventory (slice 1, task 1386) and the PDF stages (slice 2, task 1392).

The family of task 629 P2 / ADR-113: the Education Resources Information Center (IES, US
Department of Education), full-text-authorized records and their PDF pointers. The family is
DORMANT since 2026-09-30 (no process runs it); this module is the first slice of its move onto the
`SourceFamily` Protocol (task 643 S7, template: task 1373).

TWO STAGES, one option: `--opt stage=inventory` (default, slice 1) or `--opt stage=pdf` (slice 2).

INVENTORY (task 1386). The operator loop fetches the API pages with `curl` and writes one sealed
page receipt per page. `discover`: one unit per page receipt, walked from offset 0 by the receipt's
own row count up to its own `num_found`; a missing page is an error, not a shorter inventory.
`acquire`: validates the page (`cytir.sources.eric.inventory.validate_page`) and returns the API body,
which the framework seals. `seal_inventory`: the whole-inventory seal (conservation of ids, stage
receipt, `RUN.json`).

PDF (task 1392). Options: `artifact_root` (the root the PDF acquisition's `artifact://` URIs resolve
against) and `receipts` (the directory of its `aw.eric-official-pdf-receipt.v1` files).
  * `discover`: one unit per receipt, consent host `files.eric.ed.gov` (resolved live by the
    framework, never inherited).
  * `acquire`: the receipt's parent body is validated (`materialize._validate_parent_receipt`) and
    its PDF bytes are returned for the framework to seal.
  * `extract`: the routed PDF extractor (`cytir.extract.pdf.extract_pdf`) and the one record shape
    (`materialize.build_record`) the batch materializer also uses; language by the language gate. An
    empty Markdown (an OCR-required PDF) is QUARANTINED, as the batch materializer does.
  * `chunk`: the structured chunker under the contract's DECLARED policy (ADR-113), one source row
    per chunk; `TIER`, `SOURCE_TYPE` and `TRUST_TIER` are the family's constants below.
  * `l1`: with `parent_root` + `release_name`, files the terminal materialization's
    `markdown/documents.jsonl.gz` under `<out_dir>/l1/release=<name>/` (`l1_seal.register`); without
    them it returns an explicit `skipped` manifest, never a silent no-op.

WHAT STAYS OPS (not the adapter, ADR-105): the L1 COVERAGE PROOF against the shared persisted index
(`cytir.sources.eric.l1_seal.seal`), the tranche partitioning workers, the two audits and the chain
scripts. They read the same package modules the adapter calls.

CONSENT AND LICENCE are read from the family's contract row, never written here: `discover`
refuses a row that does not declare `search_ok`, and copies the row's consent resolution and
licence basis into the run notes (so the manifest carries the terms the run was admitted under).
The contract's consent covers the PDF host (`files.eric.ed.gov`); the inventory pages come from the
documented API, fetched outside this adapter, so a page unit is a `file:` URI and contacts no host.

Options (`--opt KEY=VALUE`, or `--from DIR`): `root` (required; the inventory directory that holds
`pages/`), optional `query`, `fields`, `rows` (default: read from the first page receipt).
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import tempfile

from cytir.sources.chunk_selection import declared_policy_selection
from cytir.sources.eric import inventory as inv
from cytir.sources.eric import l1_seal
from cytir.sources.eric import materialize as mat
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit
from cytir.sources.registry import register_family

PAGE_RECEIPT = "pages/{start:09d}.receipt.json"
PDF_HOST = "files.eric.ed.gov"
DOMAIN = PDF_HOST
#: values = the family's served rows (twin check 2026-10-30, task 1392): `source_type` reference,
#: `trust_tier` 0.7, `license` "none declared" (the host publishes no licence text). The payload `tier`
#: is the IDENTITY tier (cytir.core.ids.classify_tier), not an allowlist tier: no letter of
#: `cytir.core.common.TIER_TRUST` maps to 0.7 (A 1.0, B 0.82, C 0.72, D 0.5). "B" is the identity rule for
#: prose with no per-chunk source anchor; the live row's `tier` letter is NOT verified (see the task doc).
#: The adapter stamps no `country` (the Passage has no such field); the served rows carry INTL.
#: Served ERIC rows carry NO `tier` field at all (twin check 2026-10-30, task 1392: three live rows for
#: files.eric.ed.gov, payload keys without `tier`), so `cytir.core.ids.classify_tier` is the only source:
#: B is its answer for anchorless prose.
TIER = "B"
SOURCE_TYPE = "reference"
TRUST_TIER = 0.7
LICENSE = "none declared"
STAGES = ("inventory", "pdf")


def contract_terms(ctx: RunContext) -> dict:
    """The consent and licence terms of the family's contract row, as the run notes carry them."""
    row = ctx.contract
    if not row or "consent" not in row or "licence" not in row:
        raise ValueError("eric_official_pdf: the run has no contract row with consent and licence")
    consent, licence = row["consent"], row["licence"]
    if consent.get("search_ok") is not True:
        raise ValueError("eric_official_pdf: the contract does not declare search_ok; the family "
                         "does not run")
    return {"search_ok": consent["search_ok"], "train_ok": consent.get("train_ok"),
            "consent_resolution": consent.get("resolution"),
            "licence_basis": licence.get("basis"), "licence_spdx": licence.get("spdx")}


def _root(ctx: RunContext) -> pathlib.Path:
    root = ctx.options.get("root")
    if not root:
        raise ValueError("eric_official_pdf needs --opt root=<inventory directory holding pages/> "
                         "(or --from DIR)")
    if not pathlib.Path(root).is_dir():
        raise ValueError(f"eric_official_pdf: root {root!r} is not a directory")
    return pathlib.Path(root).resolve()


def _query(ctx: RunContext, first: dict) -> tuple[str, str, int]:
    return (ctx.options.get("query") or first["search"],
            ctx.options.get("fields") or first["fields"],
            int(ctx.options.get("rows") or first["rows"]))


def _first_receipt(root: pathlib.Path) -> dict:
    path = root / PAGE_RECEIPT.format(start=0)
    if not path.is_file():
        raise RuntimeError(f"eric_official_pdf: no page receipt at {path}; "
                           "an empty walk is a wrong root, not an empty source")
    return json.loads(path.read_text(encoding="utf-8"))


@register_family
class EricOfficialPdfFamily:
    family_id = "eric_official_pdf"
    route = "documented_source_api"
    direct_crawl = True
    chunk_policy_id = CHUNK_DECLARED
    require_records = True

    def _stage(self, ctx: RunContext) -> str:
        stage = ctx.options.get("stage", "inventory")
        if stage not in STAGES:
            raise ValueError(f"eric_official_pdf: --opt stage must be one of {STAGES}, got {stage!r}")
        return stage

    def preflight(self, ctx: RunContext) -> None:
        if self._stage(ctx) == "pdf":
            self._selection = declared_policy_selection(ctx)

    def l1(self, ctx: RunContext) -> dict:
        """File the terminal materialization's Markdown shard as an L1 release under the out dir."""
        parent, name = ctx.options.get("parent_root"), ctx.options.get("release_name")
        if not (parent and name):
            return {"skipped": "no parent_root and release_name option: no L1 release filed"}
        reg = l1_seal.register(parent, release_name=name, corpus_root=str(ctx.out_dir))
        return {"release": name, "files": reg["files"], "dest": reg["dest"]}

    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = contract_terms(ctx)
        if self._stage(ctx) == "pdf":
            return self._discover_pdf(ctx)
        root = _root(ctx)
        first = _first_receipt(root)
        query, fields, rows = _query(ctx, first)
        total = first["num_found"]
        units = []
        for start in range(0, total, rows):
            receipt = root / PAGE_RECEIPT.format(start=start)
            if not receipt.is_file():
                raise RuntimeError(f"eric_official_pdf: missing page receipt at start={start}")
            units.append(Unit(key=f"page-{start:09d}", url=receipt.as_uri(),
                              meta={"root": str(root), "start": start, "rows": rows,
                                    "query": query, "fields": fields, "total": total}))
        ctx.notes.update({"pages": len(units), "catalog_records": total})
        return units

    def _discover_pdf(self, ctx: RunContext):
        artifact_root, receipts = ctx.options.get("artifact_root"), ctx.options.get("receipts")
        if not (artifact_root and receipts):
            raise ValueError("eric_official_pdf stage=pdf needs --opt artifact_root=<dir> and "
                             "--opt receipts=<dir of PDF acquisition receipts>")
        found = sorted(pathlib.Path(receipts).glob("*.json"))
        if not found:
            raise RuntimeError(f"eric_official_pdf: no receipt under {receipts!r}; "
                               "an empty walk is a wrong root, not an empty source")
        ctx.notes.update({"pdf_receipts": len(found)})
        return [Unit(key=path.stem, url=path.resolve().as_uri(), consent_hosts=(PDF_HOST,),
                     meta={"receipt": str(path), "artifact_root": str(artifact_root)})
                for path in found]

    def acquire(self, unit: Unit, ctx: RunContext):
        if self._stage(ctx) == "pdf":
            return self._acquire_pdf(unit)
        m = unit.meta
        root = pathlib.Path(m["root"])
        summary, paths = inv.validate_page(root, root / PAGE_RECEIPT.format(start=m["start"]),
                                           m["start"], m["rows"], m["query"], m["fields"], m["total"])
        body = paths[0].read_bytes()
        transport = json.loads(paths[3].read_text(encoding="utf-8"))
        return [Captured(unit, body, {
            "status": transport["http_status"], "content_type": "application/json",
            "fetched_at": transport["fetched_at"], "source_url": transport["requested_url"],
            "start": m["start"], "documents": summary["documents"],
            "document_ids_sha256": summary["document_ids_sha256"]})]

    def _acquire_pdf(self, unit: Unit):
        receipt = json.loads(pathlib.Path(unit.meta["receipt"]).read_text(encoding="utf-8"))
        pdf = mat._validate_parent_receipt(receipt, pathlib.Path(unit.meta["artifact_root"]))
        return [Captured(unit, pdf.read_bytes(), {
            "status": 200, "content_type": "application/pdf",
            "source_url": receipt["source_url"], "eric_id": receipt["eric_id"],
            "pdf_sha256": receipt["artifacts"]["response_body"]["sha256"],
            "receipt": receipt})]

    def extract(self, cap: Captured, ctx: RunContext) -> list[Doc]:
        if self._stage(ctx) != "pdf":
            raise ValueError("eric_official_pdf: the inventory stage has no documents to extract; "
                             "run with --opt stage=pdf")
        from cytir.acquire.fetch_process import detect_lang
        from cytir.extract import pdf as extract_pdf_mod

        with tempfile.TemporaryDirectory(prefix="eric-pdf-") as tmp:
            pdf = pathlib.Path(tmp) / f"{cap.meta['eric_id']}.pdf"
            pdf.write_bytes(cap.body)
            markdown, diagnostics = mat._extract_inprocess(extract_pdf_mod.extract_pdf, pdf)
        if not markdown.strip():
            raise ValueError("extraction emitted empty Markdown; OCR-required PDF "
                             "(quarantined, an accounted drop)")
        record, _outlinks = mat.build_record(cap.meta["receipt"], markdown)
        lang, _confident = detect_lang(markdown, record["url"], "en")
        return [Doc(url=record["url"], text=markdown, title=record["title"], lang=lang,
                    source_text=markdown,
                    extra={"record": record, "diagnostics": diagnostics})]

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch

        selection = getattr(self, "_selection", None)
        if selection is None:
            raise RuntimeError("eric_official_pdf.chunk ran before preflight: the declared chunk "
                               "policy is resolved in preflight (stage=pdf); call preflight(ctx) first")
        record = doc.extra["record"]
        out = []
        stamps = selection.stamps()
        digest = C.sha1(doc.text)
        for ordinal, unit in enumerate(_dispatch.chunk_markdown_channels(doc.text,
                                                                 selection=selection)):
            unit = dataclasses.replace(unit, stamps=stamps)
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(unit.dense_text or unit.text, 25),
                lang=doc.lang, domain=DOMAIN, tier=TIER, source_type=SOURCE_TYPE,
                trust_tier=TRUST_TIER, last_seen=ctx.today, content_digest=digest,
                license=LICENSE,
                anchor=f"document:{record['document_id']}:chunk-{ordinal}",
                master={"document_id": record["document_id"],
                        "source_record_id": record["source_record_id"],
                        "source_part_sha256": record["parent_identity"]["pdf_sha256"],
                        "parent_markdown_sha256": record["markdown_sha256"]})
            rec = C.passage_source_record(p)
            rec["document_id"], rec["chunk_ordinal"] = record["document_id"], ordinal
            out.append(rec)
        return out

    def seal_inventory(self, ctx: RunContext) -> dict:
        """Seal the whole inventory: id conservation, the stage receipt and `RUN.json`."""
        contract_terms(ctx)
        root = _root(ctx)
        query, fields, rows = _query(ctx, _first_receipt(root))
        return inv.finalize(root=root, query=query, fields=fields, rows=rows)
