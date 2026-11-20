"""Shared behaviour of families whose unit is one HTML page (task 683, S7).

Two of the three exemplar families (a sitemap pack and the Common Crawl route) end in the same
place: a page's bytes go through ONE extractor and ONE chunk dispatcher
(`cytir.core.common.process_html`, the door every `ingest_*` script has always called) and come out
as `Passage` records. That tail is written here once. A family differs only in how it DISCOVERS
and ACQUIRES pages (a sitemap GET; a CDX record and a WARC range read), and in what it stamps on
its rows (`passage_common`).

`extract` decodes the sealed bytes and records the page title. `chunk` calls `process_html`: the
markdown extraction, the extraction-junk and listing-shell gates, the language gate and the
chunker the pinned selection names all live behind it, exactly as before this task.

The chunk policy: `preflight` reads the dispatcher's ACTUAL selection (`chunker_selection()`:
env `AW_CHUNKER`/`AW_CHUNK_POLICY*`, else the pinned build profile) and
  * records it in the pack manifest (`discover.chunker`), and
  * REFUSES the run when the family's contract declares a policy (by sha256) that the selection
    does not match. A declaration and the process that claims to honour it must agree before a
    byte is fetched.
"""
from __future__ import annotations

import re

from cytir.sources.family import Captured, Doc, RunContext, Unit

_CHARSET = re.compile(r"charset\s*=\s*([\w.-]+)", re.I)


def decode_html(body: bytes, content_type: str | None) -> str:
    """Text of a fetched page: the declared charset if Python knows it, else UTF-8, never
    raising (an undecodable byte becomes U+FFFD, as `cc_ingest`/`pack_fetch` have always done)."""
    m = _CHARSET.search(content_type or "")
    for enc in ([m.group(1)] if m else []) + ["utf-8"]:
        try:
            return body.decode(enc, "replace")
        except LookupError:
            continue
    return body.decode("utf-8", "replace")


class WebPageMixin:
    """`extract`, `chunk` and `preflight` for HTML-page families. A subclass supplies
    `discover`, `acquire`, and `passage_common`."""

    chunker_via_dispatcher = True

    def passage_common(self, ctx: RunContext, unit: Unit) -> dict:      # pragma: no cover
        """Fields forwarded to every `Passage` of this unit (domain, tier, source_type,
        trust_tier, last_seen, license...)."""
        raise NotImplementedError

    def preflight(self, ctx: RunContext) -> None:
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch
        from cytir.extract import html_extract as _html_extract
        from cytir.sources.registry import FamilyRefused

        sel = _dispatch.chunker_selection()
        ctx.notes["chunker"] = {
            "mode": sel.mode, "epoch": sel.epoch, "legend": sel.legend,
            "policy_sha256": sel.policy_sha256,
            "tokenizer_revision": sel.tokenizer_revision}
        declared = ctx.chunk_spec.sha256 if ctx.chunk_spec else None
        if declared and sel.policy_sha256 != declared:
            raise FamilyRefused(
                f"{self.family_id}: the contract declares chunk policy "
                f"{ctx.chunk_spec.declared!r} (sha256 {declared[:12]}...) but the chunk "
                f"dispatcher resolves to mode={sel.mode!r} policy_sha256="
                f"{(sel.policy_sha256 or 'none')[:12]}. Select it the way a candidate run does "
                "(AW_CHUNKER=structured + AW_CHUNK_POLICY + AW_CHUNK_POLICY_SHA256 + "
                "AW_CHUNK_TOKENIZER_REVISION), or re-declare the family (POLICY PR).")

    def page_html(self, cap: Captured) -> str:
        """The page's HTML from the sealed bytes. A family whose bytes are wrapped (a WARC
        member) overrides this; the seal keeps what arrived, this unwraps it."""
        return decode_html(cap.body, cap.meta.get("content_type"))

    def extract(self, cap: Captured, ctx: RunContext):
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch
        from cytir.extract import html_extract as _html_extract

        html = self.page_html(cap)
        return [Doc(url=cap.unit.url, text=html, title=_html_extract.html_title(html) or "",
                    lang=cap.unit.meta.get("lang") or "",
                    extra={"passage_common": self.passage_common(ctx, cap.unit),
                           "digest": cap.meta.get("digest")})]

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch
        from cytir.extract import html_extract as _html_extract

        meta = {"content": True, "links": False,
                "lang_default": doc.lang or "en",
                "passage_common": doc.extra["passage_common"]}
        if doc.extra.get("digest"):
            meta["digest"] = doc.extra["digest"]
        passages, _links = _html_extract.process_html(doc.url, doc.text, meta)
        rows = []
        for p in passages:
            row = p.payload()
            row["text"] = p.text
            row["point_id"] = p.point_id
            rows.append(row)
        return rows
