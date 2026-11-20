"""`w3c_exact`: W3C /TR/ technical reports on the family door (task 1456, S7 L2).

The code home is the package extractor `cytir.extract.routec.w3c_technical_reports` (PRODUCER label
`cytir.extract.routec.w3c_technical_reports`, frozen in sealed receipts). Its `materialize` consumes a
sealed Route-C routes file and writes three sealed streams (markdown parents, outlinks, residual).
This adapter reads the SAME sealed inputs one unit at a time and contacts no host. The extractor
is not edited; `extract` calls the functions `materialize` calls, in the same order, so a parent
here is byte-equal to the sealed markdown row (pinned in `tests/test_aw_family_w3c.py`).

STAGE MAP
  discover   one unit per route row of `w3c-standards-html` (sealed routes, gz jsonl)         PORTED
  acquire    validates the row's bindings, returns the preserved HTML; the extractor's
             residual reasons are terminal EXCLUDED facts (non-/TR/ URL, non-HTML media,
             a direct status other than `complete`)                                          PORTED
  extract    principal-content markdown + parent identity + context outlinks                 PORTED
  chunk      the structured chunker under the contract's DECLARED policy (7dcb54c4@4, the
             policy of the sealed stage, task 2139 from task 2133 method); the record carries no
             tier / source_type / trust_tier (the sealed stage rows hold none)                 PORTED
  seal       `seal_streams` delegates to the extractor's `materialize` (the three sealed
             streams + RUN.json), from the same sealed inputs and `--opt output_root`;
             it re-implements nothing (the `seal_inventory` pattern of `rfc_editor`)         PORTED

Options (`--opt KEY=VALUE`): `artifact_root` (the global `artifact://` root), `routes` (the routes
file, `*.jsonl.gz`); `output_root` (only `seal_streams`: a NEW directory under `artifact_root`). CONSENT AND LICENCE are read from the contract row as they stand (all
`unknown` for this backfilled row) and copied into the run notes; nothing here admits on them.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import dataclasses
import pathlib
from urllib.parse import urlparse

from cytir.extract.html_extract import _parse_soup, extract_content_markdown, extract_outlink_context_records
from cytir.extract.routec import common as ROUTEC
from cytir.extract.routec import w3c_technical_reports as W3C
from cytir.sources.chunk_selection import declared_policy_selection
from cytir.sources.stage_out import OFFLINE_ROUTE
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

_REQUIRED = {"response_body", "response_headers", "transport", "consent"}


def contract_terms(ctx: RunContext) -> dict:
    """The consent and licence terms of the contract row, exactly as declared (unknown stays unknown)."""
    row = ctx.contract
    if not row or "consent" not in row or "licence" not in row:
        raise ValueError("w3c: the run has no contract row with consent and licence")
    consent, licence = row["consent"], row["licence"]
    return {"search_ok": consent.get("search_ok"), "train_ok": consent.get("train_ok"),
            "consent_resolution": consent.get("resolution"),
            "licence_basis": licence.get("basis"), "licence_spdx": licence.get("spdx")}


def _opt(ctx: RunContext, name: str, *, directory: bool = False) -> pathlib.Path:
    value = ctx.options.get(name)
    if not value:
        raise ValueError(f"w3c needs --opt {name}=<path>")
    path = pathlib.Path(value).resolve()
    if directory and not path.is_dir() or not directory and not path.is_file():
        raise ValueError(f"w3c: {name} {value!r} does not exist")
    return path


@register_family
class W3cExactFamily:
    family_id = "w3c_exact"
    route = OFFLINE_ROUTE       # the Protocol label for a sealed-replay adapter that fetches nothing; the row says unknown
    #: the sealed bodies were fetched earlier with preserved consent evidence; this stage reads
    #: them and contacts no host
    direct_crawl = False
    chunk_policy_id = CHUNK_DECLARED
    require_records = True

    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = contract_terms(ctx)
        artifact_root = _opt(ctx, "artifact_root", directory=True)
        routes = _opt(ctx, "routes")
        units, seen, points = [], set(), 0
        with gzip.open(routes, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                route = json.loads(line)
                if route.get("route_id") != W3C.ROUTE_ID:
                    continue
                url, pts = str(route.get("url") or ""), route.get("production_points")
                if not url or url in seen or not isinstance(pts, int) or pts < 0:
                    raise ValueError(f"invalid W3C route row at line {line_number}")
                seen.add(url)
                points += pts
                units.append(Unit(key=hashlib.sha256(url.encode()).hexdigest(), url=url, meta={
                    "artifact_root": str(artifact_root), "points": pts,
                    "direct_status": route.get("direct_status"),
                    "final_url": route.get("final_url"),
                    "terminal_receipt": route.get("terminal_receipt"),
                    "original_artifacts": route.get("original_artifacts") or {}}))
        ctx.notes.update({"route_urls": len(units), "route_reported_production_points": points})
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        m = unit.meta
        if m["direct_status"] != "complete":
            raise UnitExcluded("terminal_source_response_quarantined")
        if not W3C.is_technical_report_url(unit.url):
            raise UnitExcluded("non_TR_W3C_object_requires_separate_projection")
        root, artifacts = pathlib.Path(m["artifact_root"]), m["original_artifacts"]
        if set(artifacts) != _REQUIRED:
            raise ValueError(f"W3C source artifact set drift: {unit.url}")
        W3C._validate_binding(root, m["terminal_receipt"])
        for artifact in artifacts.values():
            W3C._validate_binding(root, artifact)
        media_type = ROUTEC.content_type(root, {"artifacts": artifacts})
        if media_type not in ROUTEC.HTML_TYPES:
            raise UnitExcluded("non_HTML_TR_response_requires_media_specific_projection")
        body = ROUTEC.logical(root, artifacts["response_body"])
        return [Captured(unit, body, {
            "status": 200, "content_type": media_type, "source_url": unit.url,
            "final_url": m["final_url"], "parent_source_sha256": artifacts["response_body"]["body_sha256"],
            "parent_identity": {"terminal_receipt": m["terminal_receipt"],
                                "response_body": artifacts["response_body"],
                                "response_headers": artifacts["response_headers"],
                                "transport": artifacts["transport"],
                                "consent": artifacts["consent"]}})]

    def extract(self, cap: Captured, ctx: RunContext):
        url = cap.unit.url
        html = cap.body.decode("utf-8", "replace")
        soup = _parse_soup(html)
        principal, mode = extract_content_markdown(html, url)
        if not principal.strip():
            raise UnitExcluded("principal-content-extraction-empty")
        markdown = principal.strip() + "\n"
        title = " ".join((soup.title.get_text(" ", strip=True) if soup is not None and soup.title else "").split())
        links = extract_outlink_context_records(soup, url) or [
            {"url": None, "nofollow": False, "anchor_or_context": None}]
        return [Doc(url=url, text=markdown, title=title,
                    lang=((soup.html.get("lang") if soup is not None and soup.html else None) or ""),
                    source_text=markdown, extra={
            "document_id": hashlib.sha256(f"w3c-technical-report\0{url}".encode()).hexdigest(),
            "source_id": W3C.SOURCE_ID, "source_profile": "software_docs",
            "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
            "parent_source_sha256": cap.meta["parent_source_sha256"],
            "parent_identity": cap.meta["parent_identity"],
            "source_version": W3C.report_path_identity(url, cap.meta["final_url"], soup),
            "extraction_mode": mode, "outlinks": links,
            "reported_production_points": cap.unit.meta["points"]})]

    def seal_streams(self, ctx: RunContext) -> dict:
        """Seal the three streams: the extractor's `materialize`, from the same options."""
        out = ctx.options.get("output_root")
        if not out:
            raise ValueError("w3c_exact.seal_streams needs --opt output_root=<new dir under artifact_root>")
        return W3C.materialize(artifact_root=_opt(ctx, "artifact_root", directory=True),
                               routes_path=_opt(ctx, "routes"), output_root=pathlib.Path(out))

    def preflight(self, ctx: RunContext) -> None:
        self._selection = declared_policy_selection(ctx)

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch

        selection = getattr(self, "_selection", None)
        if selection is None:
            raise RuntimeError("w3c_exact.chunk ran before preflight: the declared chunk "
                               "policy is resolved in preflight; call preflight(ctx) first")
        out, stamps, digest = [], selection.stamps(), C.sha1(doc.text)
        for ordinal, unit in enumerate(_dispatch.chunk_markdown_channels(doc.text, selection=selection)):
            unit = dataclasses.replace(unit, stamps=stamps)
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(unit.dense_text or unit.text, 25),
                lang=doc.lang, domain=urlparse(doc.url).hostname or "", tier=None,
                source_type=None, trust_tier=None, last_seen=ctx.today, content_digest=digest,
                anchor=f"document:{doc.extra['document_id']}:chunk-{ordinal}")
            rec = C.passage_source_record(p)
            rec["document_id"], rec["chunk_ordinal"] = doc.extra["document_id"], ordinal
            out.append(rec)
        return out
