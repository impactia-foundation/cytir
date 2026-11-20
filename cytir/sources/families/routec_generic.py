"""`routec_generic_complete_exact`: Route-C generic web HTML on the family door (task 1432, S7).

The contract row is a FROZEN BUILD (19,849 points; `source_unit` names the direct-after-CC-miss
generic-web materialization). Its producers were two materializers, now moved into the package:

    route                    materializer (code home)                   reads (sealed, on disk)
    Common Crawl capture     cytir.sources.routec.preserved_html           CC terminal receipts + decoded HTML
    direct after a CC miss   cytir.sources.routec.direct_preserved_html    consent-gated direct receipts + body

Both read already-preserved bytes and CONTACT NO HOST. They classify every frontier URL against a
reviewed host/path route registry; a URL the registry sends to a source-specific projection is not
generic web and is left to that projection (EXCLUDED here); only the registry FALLBACK is projected to
generic principal-content markdown.

STAGE MAP
  discover   one unit per frontier URL (sealed frontier, route registry)                          PORTED
  acquire    reads one sealed terminal, validates it, returns the preserved HTML bytes             PORTED
  extract    the generic-web projection (`project_html`, the code the materializers call)          PORTED
  chunk      declared 7dcb54c4@4 policy, anchor document:<id>:chunk-<n>   PORTED

`acquire` decisions: a CC miss, a registry route with a source-specific projection, a non-HTML media type,
a recorded consent or robots decline and an empty principal-content extraction are terminal EXCLUDED facts
about the unit. A direct receipt without affirmative search consent, an unreadable policy and a malformed
receipt are our own failure and QUARANTINE the unit (never a fact about the source).

Options (`--opt KEY=VALUE`): `artifact_root` (the global `artifact://` root), `frontier`,
`frontier_receipt`, `route_registry`; for the CC route `cc_root` and `cc_run`; for the direct route
`direct_root` (when `direct_root` is given the run is the direct route).

CONSENT AND LICENCE are read from the contract row and copied into the run notes as they stand (all
`unknown` for this backfilled row); nothing here admits on them. Chunk policy (task 2135, from task 2133): the row declares the sealed
stage's policy 7dcb54c4...@4; `chunk` runs it and sets no tier, source_type or trust_tier (the sealed rows carry none).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import urllib.parse

from cytir.acquire import preserved_stage_io as PS
from cytir.acquire.artifact_io import sha256_file
from cytir.extract.html_extract import _parse_soup
from cytir.extract.routec.common import HTML_TYPES, content_type
from cytir.index.preservation_contract import _validate_stage
from cytir.sources.chunk_selection import declared_policy_selection
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family
from cytir.sources.routec import direct_preserved_html as DM
from cytir.sources.routec import preserved_html as CM

#: direct receipts that record a decision of the origin or of its policy (a terminal EXCLUDED)
_DECLINES = ("consent_declined", "robots_path_declined", "non_textual_empty_or_non_2xx")
_SOURCE_ID = {"cc": "routec-phase2-all-common-crawl", "direct": "routec-phase2-all-direct-after-cc-miss"}


def contract_terms(ctx: RunContext) -> dict:
    """The consent and licence terms of the contract row, exactly as declared (unknown stays unknown)."""
    row = ctx.contract
    if not row or "consent" not in row or "licence" not in row:
        raise ValueError("routec generic: the run has no contract row with consent and licence")
    consent, licence = row["consent"], row["licence"]
    return {"search_ok": consent.get("search_ok"), "train_ok": consent.get("train_ok"),
            "consent_resolution": consent.get("resolution"),
            "licence_basis": licence.get("basis"), "licence_spdx": licence.get("spdx")}


def _opt(ctx: RunContext, name: str, *, directory: bool = False) -> pathlib.Path:
    value = ctx.options.get(name)
    if not value:
        raise ValueError(f"routec generic needs --opt {name}=<path>")
    path = pathlib.Path(value).resolve()
    if directory and not path.is_dir() or not directory and not path.is_file():
        raise ValueError(f"routec generic: {name} {value!r} does not exist")
    return path


@register_family
class RoutecGenericCompleteExactFamily:
    family_id = "routec_generic_complete_exact"
    route = "common_crawl"
    #: the stages read sealed bytes and contact no host; the direct receipts carry their own preserved
    #: consent evidence, which `acquire` re-validates
    direct_crawl = False
    chunk_policy_id = CHUNK_DECLARED
    require_records = True

    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = contract_terms(ctx)
        artifact_root = _opt(ctx, "artifact_root", directory=True)
        frontier, frontier_receipt = _opt(ctx, "frontier"), _opt(ctx, "frontier_receipt")
        registry = CM._load_registry(_opt(ctx, "route_registry"))
        rows, _ = DM._frontier(frontier, frontier_receipt)
        meta = {"artifact_root": str(artifact_root)}
        if ctx.options.get("direct_root"):
            mode = "direct"
            root = _opt(ctx, "direct_root", directory=True)
            run = DM.direct_validators().validate_consent_gated_run(root, artifact_root, deep=True)
            if int(run["counts"]["frontier"]) != len(rows):
                raise RuntimeError("routec generic: direct receipt population drift")
        else:
            mode = "cc"
            root = _opt(ctx, "cc_root", directory=True)
            run = json.loads(_opt(ctx, "cc_run").read_text(encoding="utf-8"))
            if run.get("schema") != "aw.exact-url-cc-preserved-run.v1" or run.get("status") != "complete":
                raise RuntimeError("routec generic: a terminal signed-S3 Route-C RUN is required")
            _validate_stage(run.get("stages", {}).get("original"), "routec.original",
                            artifact_root=artifact_root, deep=True)
            meta["plan_sha256"] = sha256_file(root / "plan/captures.jsonl")
        meta.update({"mode": mode, "root": str(root)})
        ctx.notes.update({"route": mode, "frontier": len(rows),
                          "registry_routes": len(registry["routes"])})
        units = []
        for url, row in rows.items():
            route = CM.classify_url(url, registry)
            units.append(Unit(key=hashlib.sha256(url.encode()).hexdigest(), url=url,
                              meta={**meta, "route_id": route["route_id"],
                                    "projection": route["projection"],
                                    "points": int(row["retained_points"])}))
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        m = unit.meta
        artifact_root, root = pathlib.Path(m["artifact_root"]), pathlib.Path(m["root"])
        path = CM._receipt_path(root, unit.url)
        if not path.is_file():
            raise RuntimeError(f"{self.family_id}: no sealed terminal receipt for {unit.url}; "
                               "acquire reads the sealed run and fetches nothing")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if m["mode"] == "cc":
            return self._acquire_cc(unit, receipt, artifact_root)
        return self._acquire_direct(unit, receipt, artifact_root)

    def _acquire_cc(self, unit, receipt, artifact_root):
        m = unit.meta
        if (receipt.get("schema") != "aw.exact-url-cc-capture-receipt.v1"
                or receipt.get("source_url") != unit.url
                or receipt.get("plan_sha256") != m["plan_sha256"]
                or receipt.get("status") not in {"complete", "cc_miss"}):
            raise ValueError(f"invalid Route-C terminal receipt: {unit.url}")
        for artifact in (receipt.get("artifacts") or {}).values():
            PS.validate_artifact(artifact_root, artifact)
        if receipt["status"] == "cc_miss":
            raise UnitExcluded("cc_miss")
        if m["projection"] == "source-specific-pending":
            raise UnitExcluded("source-specific-projection-pending")
        html = CM._logical_artifact(artifact_root, receipt["artifacts"]["decoded_html"])
        return [Captured(unit, html, {"status": 200, "content_type": "text/html", "source_url": unit.url,
                                      "mode": "cc", "route_id": m["route_id"],
                                      "parent_source_sha256": receipt["artifacts"]["decoded_html"]["body_sha256"]})]

    def _acquire_direct(self, unit, receipt, artifact_root):
        m = unit.meta
        DM.direct_validators().validate_consent_gated_receipt(receipt, artifact_root)
        for artifact in (receipt.get("artifacts") or {}).values():
            PS.validate_artifact(artifact_root, artifact)
        if receipt["status"] != "complete":
            reason = str(receipt.get("quarantine_reason") or "direct-quarantine")
            if reason in _DECLINES:
                raise UnitExcluded(reason)
            raise RuntimeError(f"{self.family_id}: {unit.url} is quarantined in the sealed run "
                               f"({reason}); an instrument failure or unreadable policy is never "
                               "a fact about the source")
        if m["projection"] == "source-specific-pending":
            raise UnitExcluded("source-specific-projection-pending")
        ctype = content_type(artifact_root, receipt)
        if ctype not in HTML_TYPES:
            raise UnitExcluded("generic-fallback-media-type-is-not-web-html")
        body = CM._logical_artifact(artifact_root, receipt["artifacts"]["response_body"])
        return [Captured(unit, body, {"status": 200, "content_type": ctype, "source_url": unit.url,
                                      "final_url": receipt.get("final_url"), "mode": "direct",
                                      "route_id": m["route_id"],
                                      "parent_source_sha256": receipt["artifacts"]["response_body"]["body_sha256"]})]

    def extract(self, cap: Captured, ctx: RunContext):
        html = cap.body.decode("utf-8", "replace")
        url = cap.unit.url
        soup = _parse_soup(html)
        principal, mode, title, lang = CM.project_html(html, url, soup)
        if not principal.strip():
            raise UnitExcluded("principal-content-extraction-empty")
        markdown = principal.strip() + "\n"
        prefix = "routec-web-html" if cap.meta["mode"] == "cc" else "routec-direct-web-html"
        return [Doc(url=url, text=markdown, title=title, lang=lang or "", source_text=markdown, extra={
            "document_id": hashlib.sha256(f"{prefix}\0{url}".encode()).hexdigest(),
            "source_id": _SOURCE_ID[cap.meta["mode"]], "source_profile": "web_html",
            "route_id": cap.meta["route_id"], "extraction_mode": mode,
            "parent_source_sha256": cap.meta["parent_source_sha256"],
            "final_url": cap.meta.get("final_url"),
            "baseline_production_points": cap.unit.meta["points"]})]

    def preflight(self, ctx: RunContext) -> None:
        self._selection = declared_policy_selection(ctx)

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch

        selection = getattr(self, "_selection", None)
        if selection is None:
            raise RuntimeError("routec_generic_complete_exact.chunk ran before preflight: the declared chunk "
                               "policy is resolved in preflight; call preflight(ctx) first")
        out, stamps = [], selection.stamps()
        digest = C.sha1(doc.text)
        for ordinal, unit in enumerate(_dispatch.chunk_markdown_channels(doc.text, selection=selection)):
            unit = dataclasses.replace(unit, stamps=stamps)
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(unit.dense_text or unit.text, 25),
                lang=doc.lang, domain=urllib.parse.urlsplit(doc.url).hostname or "", tier=None, source_type=None,
                trust_tier=None, last_seen=ctx.today, content_digest=digest,
                anchor=f"document:{doc.extra['document_id']}:chunk-{ordinal}")
            rec = C.passage_source_record(p)
            rec["document_id"], rec["chunk_ordinal"] = doc.extra["document_id"], ordinal
            out.append(rec)
        return out
