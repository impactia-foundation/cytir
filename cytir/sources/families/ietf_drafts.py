"""`ietf_drafts_exact`: Route-C IETF Internet-Draft records on the family door (task 1455, S7).

The contract row is a sealed replacement family (`routec-ietf-datatracker-internet-draft-records-current-v1`).
Its producer is `cytir.extract.routec.ietf_internet_drafts` (already a package module; unchanged here). That
module's `materialize(...)` reads preserved Datatracker responses (consent-gated direct, after a CC miss) and
seals a run: `RUN.json`, `markdown/ietf-draft-records.jsonl.gz` (one parent per draft record) and the
outlink stream. This adapter runs over that SEALED run. It contacts no host and re-fetches nothing.

STAGE MAP
  discover   one unit per sealed parent record, plus one per current-RFC redirect in RUN.json   PORTED
  acquire    validates the sealed artifact bindings and returns the preserved response body    PORTED
  extract    `render_draft_record` over that body; must reproduce the sealed parent markdown   PORTED
  chunk      declared 7dcb54c4@4 policy, anchor document:<id>:chunk-<n>   PORTED

`acquire` decisions: a requested draft whose current response is an RFC record is a fact about the origin
(EXCLUDED `current-response-is-RFC-record-not-draft-record`, the materializer's own disposition). A failed
binding, a body digest mismatch or a re-render that differs from the sealed parent is OUR failure and
QUARANTINES the unit.

Options (`--opt KEY=VALUE`): `artifact_root` (the global `artifact://` root) and `run_root` (the
`materialize(output_root=...)` directory, which must sit under `artifact_root`).

CONSENT AND LICENCE are read from the contract row and copied into the run notes as they stand (all
`unknown` for this backfilled row); nothing here admits on them. The consent evidence of each response is
inside the sealed run (`consent` artifact) and is not re-resolved. The row declares the sealed stage's chunk policy (7dcb54c4...@4, task 2135
from task 2133); `chunk` runs it and sets no tier, source_type or trust_tier (the sealed stage rows carry none).
"""
from __future__ import annotations

import gzip
import dataclasses
import hashlib
import json
import pathlib

from cytir.extract.routec import ietf_internet_drafts as IETF
from cytir.index.preservation_contract import _validate_stage
from cytir.sources.chunk_selection import declared_policy_selection
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family
from cytir.sources.stage_out import OFFLINE_ROUTE

RFC_REDIRECT = "current-response-is-RFC-record-not-draft-record"
_RUN_SCHEMA = "aw.routec-ietf-datatracker-draft-record-run.v1"
_PARENT_SCHEMA = "aw.ietf-datatracker-draft-record-parent.v1"
_PARENTS = "markdown/ietf-draft-records.jsonl.gz"


def contract_terms(ctx: RunContext) -> dict:
    """The consent and licence terms of the contract row, exactly as declared (unknown stays unknown)."""
    row = ctx.contract
    if not row or "consent" not in row or "licence" not in row:
        raise ValueError("ietf drafts: the run has no contract row with consent and licence")
    consent, licence = row["consent"], row["licence"]
    return {"search_ok": consent.get("search_ok"), "train_ok": consent.get("train_ok"),
            "consent_resolution": consent.get("resolution"),
            "licence_basis": licence.get("basis"), "licence_spdx": licence.get("spdx")}


def _dir(ctx: RunContext, name: str) -> pathlib.Path:
    value = ctx.options.get(name)
    if not value:
        raise ValueError(f"ietf drafts needs --opt {name}=<directory>")
    path = pathlib.Path(value).resolve()
    if not path.is_dir():
        raise ValueError(f"ietf drafts: {name} {value!r} is not a directory")
    return path


def _key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


@register_family
class IetfDraftsExactFamily:
    family_id = "ietf_drafts_exact"
    route = OFFLINE_ROUTE       # label of what the code does: a local dump, no host contacted; the row's route value is Tim's (memo 1424 item B)
    #: the stages read a sealed run and contact no host; each response carries its own preserved
    #: consent artifact inside the run
    direct_crawl = False
    chunk_policy_id = CHUNK_DECLARED
    require_records = True

    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = contract_terms(ctx)
        artifact_root, run_root = _dir(ctx, "artifact_root"), _dir(ctx, "run_root")
        run = json.loads((run_root / "RUN.json").read_text(encoding="utf-8"))
        if run.get("schema") != _RUN_SCHEMA or run.get("status") != "complete":
            raise RuntimeError("ietf drafts: a complete sealed Internet-Draft run is required")
        for name in ("original", "markdown", "outlinks"):
            _validate_stage(run.get("stages", {}).get(name), f"ietf.{name}",
                            artifact_root=artifact_root, deep=True)
        meta = {"artifact_root": str(artifact_root)}
        units = []
        with gzip.open(run_root / _PARENTS, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("schema") != _PARENT_SCHEMA:
                    raise ValueError("ietf drafts: unexpected parent schema")
                identity = row["parent_identity"]
                units.append(Unit(key=_key(row["url"]), url=row["url"], meta={
                    **meta, "kind": "draft", "document_id": row["document_id"],
                    "final_url": row["canonical_url"], "markdown_sha256": row["markdown_sha256"],
                    "body_sha256": row["parent_source_sha256"],
                    "body_artifact": identity["artifacts"]["response_body"],
                    "points": int(row["reported_production_points"])}))
        for redirect in run.get("redirected_current_rfc_records", []):
            units.append(Unit(key=_key(redirect["requested_url"]), url=redirect["requested_url"], meta={
                **meta, "kind": "rfc-redirect", "final_url": redirect["final_url"],
                "terminal_receipt": redirect["terminal_receipt"],
                "points": int(redirect["reported_production_points"])}))
        counts = run["counts"]
        if (len(units) != counts["route_urls"] or sum(u.meta["points"] for u in units)
                != counts["route_reported_production_points"]):
            raise RuntimeError("ietf drafts: sealed parents plus redirects do not close to the route count")
        ctx.notes.update({"parents": counts["source_parents"],
                          "rfc_redirects": counts["redirected_current_rfc_records"]})
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        m = unit.meta
        root = pathlib.Path(m["artifact_root"])
        if m["kind"] == "rfc-redirect":
            IETF._validate_binding(root, m["terminal_receipt"])
            raise UnitExcluded(RFC_REDIRECT)
        body = IETF._logical(root, m["body_artifact"])
        if hashlib.sha256(body).hexdigest() != m["body_sha256"]:
            raise ValueError(f"ietf drafts: body digest drift for {unit.url}")
        return [Captured(unit, body, {"status": 200, "content_type": "text/html",
                                      "source_url": unit.url, "final_url": m["final_url"],
                                      "parent_source_sha256": m["body_sha256"]})]

    def extract(self, cap: Captured, ctx: RunContext):
        m = cap.unit.meta
        title, markdown, version = IETF.render_draft_record(
            cap.body.decode("utf-8", "replace"), url=cap.unit.url, final_url=m["final_url"],
            body_sha256=m["body_sha256"])
        if hashlib.sha256(markdown.encode()).hexdigest() != m["markdown_sha256"]:
            raise RuntimeError(f"ietf drafts: re-render differs from the sealed parent for {cap.unit.url}")
        return [Doc(url=cap.unit.url, text=markdown, title=title, lang="en", source_text=markdown, extra={
            "document_id": m["document_id"], "source_id": IETF.SOURCE_ID,
            "source_profile": "structured_data_feed", "canonical_url": m["final_url"],
            "parent_source_sha256": m["body_sha256"], "source_version": version,
            "baseline_production_points": m["points"]})]

    def preflight(self, ctx: RunContext) -> None:
        self._selection = declared_policy_selection(ctx)

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch

        selection = getattr(self, "_selection", None)
        if selection is None:
            raise RuntimeError("ietf_drafts_exact.chunk ran before preflight: the declared chunk "
                               "policy is resolved in preflight; call preflight(ctx) first")
        out, stamps = [], selection.stamps()
        digest = C.sha1(doc.text)
        for ordinal, unit in enumerate(_dispatch.chunk_markdown_channels(doc.text, selection=selection)):
            unit = dataclasses.replace(unit, stamps=stamps)
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(unit.dense_text or unit.text, 25),
                lang=doc.lang, domain="datatracker.ietf.org", tier=None, source_type=None,
                trust_tier=None, last_seen=ctx.today, content_digest=digest,
                anchor=f"document:{doc.extra['document_id']}:chunk-{ordinal}")
            rec = C.passage_source_record(p)
            rec["document_id"], rec["chunk_ordinal"] = doc.extra["document_id"], ordinal
            out.append(rec)
        return out
