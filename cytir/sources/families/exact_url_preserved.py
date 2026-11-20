"""The exact-URL preserved acquirers as four families (task 1410, S7).

Four contract rows, two drivers, two sites. Each row is a FROZEN BUILD of the 2026-08 rebuild
(`data/source-family-contracts.v1.json`, source units in the row), not a live crawl:

    row                          site (host)         route         driver (code home)
    ca_canadaca_captured_exact   canada.ca           CC range-GET  cytir.sources.exact_url.cc_preserved
    ca_canadaca_direct_exact     canada.ca           direct GET    cytir.sources.exact_url.direct_preserved
    cochrane_preserved_exact     cochrane.org        CC range-GET  cytir.sources.exact_url.cc_preserved
    cochrane_direct_exact        cochrane.org        direct GET    cytir.sources.exact_url.direct_preserved

The driver<->row mapping is read from the code and the sealed run shapes, not from the row ids: the
canada.ca ledger (`data/r0-slice-ledger.v1.json`) names a `common_crawl_signed_s3` run and a
`consent_direct_after_cc_miss` run; the Cochrane direct materializer asserts the direct driver's run
schema `aw.exact-url-direct-preserved-run.v1` and version `exact-url-direct-preserved.v4-consent-gated`;
the Cochrane CC route is `common-crawl-signed-s3-exact-url` over a frontier prepared "for exact-URL CC".
The two routes chain: the direct run takes ONLY the terminal CC misses of a sealed CC run.

STAGE MAP (CC rows / direct rows)
  discover   one unit per plan row (CC) / per terminal CC miss (direct), from the sealed run   PORTED
  acquire    reads one sealed terminal receipt and returns the preserved body bytes          PORTED (delegates)
  plan_run   CC rows: `plan` (CDX/columnar lookup of the frontier URLs)                      PORTED (delegates)
  acquire_run  the live fetch + `RUN.json` seal: CC `acquire` / direct `acquire`            PORTED (delegates)
  extract    decoded HTML -> markdown (`cytir.sources.exact_url.extract`, task 2137)              PORTED
  chunk      cochrane rows: the declared 7dcb54c4@4 policy (task 2137)                       PORTED
             canada.ca rows: MIXED (reference 192w / generic 768w per page): the ratified     PORTED
             selector `chunk-reference-selector.v2` decides per page (task 2431)

`discover`/`acquire` fetch NOTHING: they read terminals already on disk and validate them with the
driver's own checks (`validate_terminal` / `validate_receipt`: schema, plan hash, artifact digests,
Common Crawl content digest, preserved consent evidence). Only `acquire_run` contacts a host: Common Crawl
(signed S3 range GET, <= 4 req/s) or the origin (direct GET, <= 1 req/s aggregate, one request per host at a
time, consent and robots resolved per host and per path BEFORE the request, a decline or an unreadable
policy is a quarantined receipt and never a fetch).

CONSENT AND LICENCE are read from the contract row and copied into the run notes as they stand. Every
value of the four rows is `unknown` (backfilled frozen builds), so nothing here admits on them; the direct
driver does its own live consent resolution, and `acquire_run` refuses a row that declares `search_ok`
false.

Options (`--opt KEY=VALUE`): `root` (the run root: CC `plan/` + `receipts/`, or direct `receipts/`),
`artifact_root` (the global `artifact://` root containing `root`); direct rows also `cc_root` (the sealed CC
run whose misses are the frontier). `plan_run`: `frontier`, `frontier_receipt`, `crawl` (required), optional
`content_class`, `exact_url_only`, `index_workers`. `acquire_run`: optional `rate`, `transport`,
`worker_index`, `worker_count`, `partition_only`, `fetch_concurrency` (CC) / `rate`, `workers` (direct).
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import pathlib
import urllib.parse
from functools import lru_cache

from cytir.sources.chunk_selection import declared_policy_selection
from cytir.sources.exact_url import cc_preserved as CCP
from cytir.sources.exact_url import direct_preserved as DIRECT
from cytir.sources.exact_url import extract as X
from cytir.sources.family import (CHUNK_DECLARED, Captured, Doc, RunContext, Unit,
                               UnitExcluded)
from cytir.sources.registry import register_family

#: direct receipts that record a decision of the origin or of its policy (a terminal EXCLUDED);
#: every other non-complete receipt is our own failure or an unreadable policy (QUARANTINED)
_DECLINES = ("consent_declined", "robots_path_declined", "non_textual_empty_or_non_2xx")


def contract_terms(ctx: RunContext) -> dict:
    """The consent and licence terms of the contract row, exactly as declared (unknown stays unknown)."""
    row = ctx.contract
    if not row or "consent" not in row or "licence" not in row:
        raise ValueError("exact-URL preserved: the run has no contract row with consent and licence")
    consent, licence = row["consent"], row["licence"]
    return {"search_ok": consent.get("search_ok"), "train_ok": consent.get("train_ok"),
            "consent_resolution": consent.get("resolution"),
            "licence_basis": licence.get("basis"), "licence_spdx": licence.get("spdx")}


def _path(ctx: RunContext, name: str) -> pathlib.Path:
    value = ctx.options.get(name)
    if not value:
        raise ValueError(f"exact-URL preserved needs --opt {name}=<directory> (or --from DIR for root)")
    path = pathlib.Path(value)
    if not path.is_dir():
        raise ValueError(f"exact-URL preserved: {name} {value!r} is not a directory")
    return path.resolve()


def _receipt_path(root: pathlib.Path, url: str) -> pathlib.Path:
    return root / "receipts" / (hashlib.sha256(url.encode()).hexdigest() + ".json")


def _artifact_bytes(artifact_root: pathlib.Path, receipt: dict, name: str) -> bytes:
    return DIRECT.PS.load_validated_artifact(artifact_root, receipt["artifacts"][name])


def _refuse_declined(ctx: RunContext) -> None:
    if contract_terms(ctx)["search_ok"] is False:
        raise ValueError("exact-URL preserved: the contract row declares search_ok false; "
                         "the live route does not run")


@lru_cache(maxsize=4)
def _current_titles(path: str) -> dict:
    """url -> {title, lang} from a sealed Cochrane source-manifest (`source-manifest.jsonl.gz`)."""
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["url"]] = {"title": r["title"], "lang": r["lang"]}
    return out


class _Site:
    """extract (and, for cochrane, chunk) by `site`; the byte rules live in `cytir.sources.exact_url.extract`."""
    site = ""

    def extract(self, cap: Captured, ctx: RunContext) -> list[Doc]:
        url = cap.unit.url
        html = cap.body.decode("utf-8", "replace")
        if self.site == "canada":
            try:
                p = X.canada_parent(html, url)
            except ValueError as e:
                raise UnitExcluded(str(e))
            return [Doc(url=url, text=p["markdown"], title=p["title"] or "", lang=p["lang"] or "",
                        source_text=p["markdown"], extra={
                            "document_id": hashlib.sha256(url.encode()).hexdigest(),
                            "source_profile": "web_html", "extraction": p["extraction"],
                            "markdown_sha256": p["markdown_sha256"]})]
        manifest = ctx.options.get("current_manifest")
        if not manifest:
            raise ValueError(f"{self.family_id}: cochrane title and lang are the CURRENT served values, "
                             "read from the sealed source manifest: pass --opt current_manifest=<file>")
        cur = _current_titles(str(manifest)).get(url)
        if cur is None:
            raise UnitExcluded("no-current-title-for-url")
        try:
            md, mode = X.cochrane_markdown(html, url, cur["title"])
        except ValueError as e:
            raise UnitExcluded(str(e))
        return [Doc(url=url, text=md, title=cur["title"], lang=cur["lang"], source_text=md, extra={
            "document_id": X.cochrane_document_id(url), "source_profile": "web_html",
            "extraction": {"producer": "aw_common.extract_content_markdown", "mode": mode,
                           "full_prechunk_content": True}})]

    def preflight(self, ctx: RunContext) -> None:
        self._selection = declared_policy_selection(ctx)
        if self.site == "canada":
            self._selector, self._unselected = _mixed_selector(ctx)

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch

        selection = getattr(self, "_selection", None)
        if selection is None:
            raise RuntimeError(f"{self.family_id}.chunk ran before preflight: the declared chunk policy "
                               "is resolved in preflight; call preflight(ctx) first")
        if self.site == "canada":
            # the row is MIXED: the ratified selector (`cytir.chunk.reference_selector`, the contract's
            # `selected_by` artifact) decides per page; a page it does not assign to `chunk.reference`
            # is cut under the row's `unselected` (generic 768w) policy.
            from cytir.chunk.reference_selector import REFERENCE
            if self._selector.classify(doc.text, has_native_anchor=False).policy_family != REFERENCE:
                selection = self._unselected
        out, stamps = [], selection.stamps()
        digest = C.sha1(doc.text)
        for ordinal, unit in enumerate(_dispatch.chunk_markdown_channels(doc.text, selection=selection)):
            unit = dataclasses.replace(unit, stamps=stamps)
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(unit.dense_text or unit.text, 25),
                lang=doc.lang, domain=urllib.parse.urlsplit(doc.url).hostname or "", tier=None,
                source_type=None, trust_tier=None, last_seen=ctx.today, content_digest=digest,
                anchor=f"document:{doc.extra['document_id']}:chunk-{ordinal}")
            rec = C.passage_source_record(p)
            rec["document_id"], rec["chunk_ordinal"] = doc.extra["document_id"], ordinal
            out.append(rec)
        return out


def _mixed_selector(ctx: RunContext):
    """(selector, unselected-policy selection) of a MIXED row: the `selected_by` artifact must be the
    file the contract pins (sha256), and the `unselected` policy is built the way the declared one is."""
    import hashlib as _h
    from cytir.chunk import dispatch as _dispatch
    from cytir.chunk import registry as chunk_registry
    from cytir.chunk.reference_selector import Selector, REFERENCE
    cp = (ctx.contract or {}).get("chunk_policy") or {}
    sel_meta, uns = cp.get("selected_by"), cp.get("unselected")
    if not (sel_meta and uns) or cp.get("policy_family") != REFERENCE:
        raise ValueError("the run has no contract row with a mixed chunk_policy (selected_by + unselected)")
    repo = pathlib.Path(__file__).resolve().parents[3]
    path = repo / sel_meta["selector"]
    if _h.sha256(path.read_bytes()).hexdigest() != sel_meta["sha256"]:
        raise ValueError(f"{sel_meta['selector']} is not the selector artifact the contract pins")
    entry = chunk_registry.entry(uns["sha256"])
    selection = _dispatch.explicit_structured_chunker_selection(
        str(repo / entry["file"]), uns["sha256"], cp["tokenizer_revision"])
    return Selector(path), selection


class _CcBase(_Site):
    """The Common Crawl range-GET route: shared by the two CC rows."""
    route = "common_crawl"
    direct_crawl = False
    require_records = True

    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = contract_terms(ctx)
        root, artifact_root = _path(ctx, "root"), _path(ctx, "artifact_root")
        plan_receipt_path, plan_path = root / "plan/RECEIPT.json", root / "plan/captures.jsonl"
        plan_receipt = json.loads(plan_receipt_path.read_text(encoding="utf-8"))
        plan = plan_receipt.get("plan") or {}
        if (plan_receipt.get("schema") != "aw.exact-url-cc-plan-receipt.v1"
                or plan_receipt.get("status") != "complete"):
            raise RuntimeError(f"{self.family_id}: a terminal exact-URL CC plan is required")
        plan_sha = CCP.sha256_file(plan_path)
        if plan_sha != plan.get("sha256"):
            raise RuntimeError(f"{self.family_id}: CC plan artifact drift")
        rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if len(rows) != (plan_receipt.get("frontier") or {}).get("urls"):
            raise RuntimeError(f"{self.family_id}: CC plan row conservation failed")
        units = [Unit(key=hashlib.sha256(r["url"].encode()).hexdigest(), url=r["url"],
                      meta={"root": str(root), "artifact_root": str(artifact_root),
                            "plan_sha256": plan_sha, "state": r["state"]})
                 for r in rows]
        ctx.notes.update({"frontier": len(units), "crawl": plan_receipt.get("crawl"),
                          "plan_captures": plan.get("captures"), "plan_misses": plan.get("misses")})
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        m = unit.meta
        root, artifact_root = pathlib.Path(m["root"]), pathlib.Path(m["artifact_root"])
        if m["state"] == "MISS":
            raise UnitExcluded("cc_miss")        # the sibling direct row takes the CC misses
        path = _receipt_path(root, unit.url)
        if not path.is_file():
            raise RuntimeError(f"{self.family_id}: no sealed terminal receipt for {unit.url}; "
                               "acquire reads the sealed run and fetches nothing (use acquire_run)")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        CCP.validate_terminal(artifact_root, receipt, unit.url, m["plan_sha256"])
        if receipt["status"] == "cc_miss":
            raise UnitExcluded("cc_miss")
        key = "decoded_html" if "decoded_html" in receipt["artifacts"] else "decoded_source"
        transport = json.loads(_artifact_bytes(artifact_root, receipt, "transport"))
        capture = transport.get("capture") or {}
        return [Captured(unit, _artifact_bytes(artifact_root, receipt, key), {
            "status": 200, "source_url": unit.url, "capture_url": receipt.get("capture_url"),
            "fetched_at": capture.get("fetch_time"), "warc_filename": capture.get("warc_filename"),
            "content_digest": capture.get("digest"), "artifact": key})]

    def plan_run(self, ctx: RunContext) -> dict:
        """Plan the frontier against the Common Crawl index: the old `--mode plan`."""
        contract_terms(ctx)
        o = ctx.options
        for need in ("frontier", "frontier_receipt", "crawl"):
            if not o.get(need):
                raise ValueError(f"{self.family_id}.plan_run needs --opt {need}=...")
        root = pathlib.Path(o["root"]).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return CCP.plan(
            frontier_path=pathlib.Path(o["frontier"]).resolve(),
            frontier_receipt_path=pathlib.Path(o["frontier_receipt"]).resolve(), root=root,
            crawl=o["crawl"], content_class=o.get("content_class", "html"),
            index_workers=int(o.get("index_workers", 4)),
            exact_url_only=str(o.get("exact_url_only", "")).lower() in ("1", "true", "yes"))

    def acquire_run(self, ctx: RunContext) -> dict:
        """Fetch the planned captures, seal `stages/original` and `RUN.json`: the old `--mode acquire`."""
        _refuse_declined(ctx)
        o = ctx.options
        return CCP.acquire(
            root=_path(ctx, "root"), artifact_root=_path(ctx, "artifact_root"),
            rate=float(o.get("rate", 4.0)), transport=o.get("transport", "signed-s3"),
            worker_index=int(o.get("worker_index", 0)), worker_count=int(o.get("worker_count", 1)),
            partition_only=str(o.get("partition_only", "")).lower() in ("1", "true", "yes"),
            fetch_concurrency=int(o.get("fetch_concurrency", 1)))


class _DirectBase(_Site):
    """The consent-gated direct GET of the terminal CC misses: shared by the two direct rows."""
    route = "direct_origin_crawl"
    direct_crawl = True
    require_records = True

    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = contract_terms(ctx)
        root, cc_root = _path(ctx, "root"), _path(ctx, "cc_root")
        artifact_root = _path(ctx, "artifact_root")
        frontier = DIRECT.miss_frontier(cc_root, artifact_root)
        units = [Unit(key=hashlib.sha256(r["source_url"].encode()).hexdigest(), url=r["source_url"],
                      meta={"root": str(root), "artifact_root": str(artifact_root),
                            "cc_receipt_sha256": r["cc_receipt_sha256"]})
                 for r in frontier]
        ctx.notes["frontier"] = len(units)
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        m = unit.meta
        root, artifact_root = pathlib.Path(m["root"]), pathlib.Path(m["artifact_root"])
        path = _receipt_path(root, unit.url)
        if not path.is_file():
            raise RuntimeError(f"{self.family_id}: no sealed direct receipt for {unit.url}; "
                               "acquire reads the sealed run and fetches nothing (use acquire_run)")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        DIRECT.validate_receipt(artifact_root, receipt,
                                {"source_url": unit.url, "cc_receipt_sha256": m["cc_receipt_sha256"]})
        if receipt["status"] != "complete":
            reason = receipt.get("content_classification") or "quarantine"
            if reason in _DECLINES:
                raise UnitExcluded(reason)
            raise RuntimeError(f"{self.family_id}: {unit.url} is quarantined in the sealed run "
                               f"({reason}); an instrument failure or unreadable policy is never "
                               "a fact about the source")
        transport = json.loads(_artifact_bytes(artifact_root, receipt, "transport"))
        headers = json.loads(_artifact_bytes(artifact_root, receipt, "response_headers"))
        ctype = next((v for k, v in headers if k.casefold() == "content-type"), "")
        return [Captured(unit, _artifact_bytes(artifact_root, receipt, "response_body"), {
            "status": transport.get("http_status"), "content_type": ctype,
            "fetched_at": transport.get("retrieved_at"), "source_url": unit.url,
            "final_url": receipt.get("final_url")})]

    def acquire_run(self, ctx: RunContext) -> dict:
        """Direct-fetch the CC misses (consent and robots first), seal and write `RUN.json`."""
        _refuse_declined(ctx)
        o = ctx.options
        return DIRECT.acquire(
            cc_root=_path(ctx, "cc_root"), root=_path(ctx, "root"),
            artifact_root=_path(ctx, "artifact_root"),
            rate=float(o.get("rate", 1.0)), workers=int(o.get("workers", 4)))


@register_family
class CanadacaCapturedExactFamily(_CcBase):
    family_id = "ca_canadaca_captured_exact"
    site = "canada"
    chunk_policy_id = CHUNK_DECLARED          # the row declares a (mixed) policy


@register_family
class CochranePreservedExactFamily(_CcBase):
    family_id = "cochrane_preserved_exact"
    site = "cochrane"
    chunk_policy_id = CHUNK_DECLARED


@register_family
class CanadacaDirectExactFamily(_DirectBase):
    family_id = "ca_canadaca_direct_exact"
    site = "canada"
    chunk_policy_id = CHUNK_DECLARED          # the row declares a (mixed) policy


@register_family
class CochraneDirectExactFamily(_DirectBase):
    family_id = "cochrane_direct_exact"
    site = "cochrane"
    chunk_policy_id = CHUNK_DECLARED
