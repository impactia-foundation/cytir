"""`rfc_editor_net_new` and `rfc_exact` -- the RFC Editor's text corpus on the Protocol (task 1395, S7).

Two declared families, one acquisition: the RFC Editor's `rfcN.txt` files, fetched by the operator
with `rsync` (outside this adapter) into a raw directory. The two rows differ only in what they
declare: `rfc_editor_net_new` (net-new addition, route `direct`) is fully declared; `rfc_exact` (the
sealed replacement set `rfc-editor-production-current-v2`) declares consent, licence and route as
`unknown`. Both classes share `_RfcEditorBase`; each reads ITS OWN contract row.

STAGE MAP
  * `discover`: one unit per `rfcN.txt` in `raw_dir` (a `file:` URI; no host is contacted). An empty
    directory or a duplicate RFC number is an error (`preserved.inventory`).
  * `acquire`: the file's bytes, digest in the capture meta.
  * `extract`: `preserved.rfc_row` (the one place a file becomes a markdown row); an empty text is
    EXCLUDED `empty_source_text`.
  * `chunk`: both families, one shared method. It RE-RENDERS the doc's raw RFC text into the heading
    Markdown (front matter + `RFC n > heading` lines) the sealed packs were chunked from
    (`cytir.extract.rfc_editor.render`, the same function `restage` calls), then chunks it under the
    contract's declared policy (`declared_policy_selection`, resolved in `preflight`). `extract` keeps
    returning the raw `rfcN.txt` text. The families differ in anchor (`rfc_editor_net_new`:
    `chunk-<n>`; `rfc_exact`: `document:<document_id>:chunk-<n>`, the sealed stage's shape) and in
    what they declare: net_new adds tier / source_type / trust_tier / licence from its row; the
    `rfc_exact` sealed rows carry none of those, so it emits none (task 2136). The prod-subtraction
    selection (`cytir.extract.rfc_editor.net_new`) works on a sealed parent run and stays where it is.
  * `seal_inventory`: the old `materialize` (rsync transport metadata, original / markdown / outlinks
    / quarantine stage receipts, `RUN.json`), delegated unchanged.

CONSENT AND LICENCE are read from the row, never written here. `discover` refuses a row that does
not declare `search_ok: true` (so `rfc_exact`, all `unknown`, refuses until a POLICY PR declares it)
and copies the row's consent resolution and licence basis into the run notes.

Options (`--opt KEY=VALUE`): `raw_dir` (required; the rsync'ed tree holding `rfcN.txt`). For
`seal_inventory` also `before_list`, `after_list`, `root`, `module`, `started_at`, `completed_at`,
`rsync_version`, `fetch_vantage`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import pathlib

from cytir.sources.chunk_selection import declared_policy_selection
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family
from cytir.sources.rfc_editor import preserved

SEAL_OPTIONS = ("before_list", "after_list", "root", "module", "started_at", "completed_at",
                "rsync_version", "fetch_vantage")


class _RfcEditorBase:
    family_id = ""
    route = "direct"
    direct_crawl = False        # files are already on disk; the rsync module is the operator's fetch
    chunk_policy_id = CHUNK_DECLARED
    require_records = True
    PARTIAL = ()
    # what a chunk record carries beyond the sealed row's own fields; `None` is dropped from the payload
    DECLARED = {"tier": None, "source_type": None, "trust_tier": None, "license": None}
    ANCHOR = "chunk-{ordinal}"

    def _terms(self, ctx: RunContext) -> dict:
        row = ctx.contract
        if not row or "consent" not in row or "licence" not in row:
            raise ValueError(f"{self.family_id}: the run has no contract row with consent and licence")
        consent, licence = row["consent"], row["licence"]
        if consent.get("search_ok") is not True:
            raise ValueError(f"{self.family_id}: the contract does not declare search_ok "
                             f"(consent resolution {consent.get('resolution')!r}); the family does not run")
        return {"search_ok": True, "train_ok": consent.get("train_ok"),
                "consent_resolution": consent.get("resolution"),
                "licence_basis": licence.get("basis"), "licence_spdx": licence.get("spdx")}

    def _raw_dir(self, ctx: RunContext) -> pathlib.Path:
        raw = ctx.options.get("raw_dir")
        if not raw:
            raise ValueError(f"{self.family_id} needs --opt raw_dir=<directory holding rfcN.txt>")
        if not pathlib.Path(raw).is_dir():
            raise ValueError(f"{self.family_id}: raw_dir {raw!r} is not a directory")
        return pathlib.Path(raw).resolve()

    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = self._terms(ctx)
        files = preserved.inventory(self._raw_dir(ctx))
        ctx.notes["files"] = len(files)
        return [Unit(key=p.name, url=p.as_uri(), meta={"path": str(p)}) for p in files]

    def acquire(self, unit: Unit, ctx: RunContext):
        body = pathlib.Path(unit.meta["path"]).read_bytes()
        return [Captured(unit, body, {"sha256": hashlib.sha256(body).hexdigest(),
                                      "content_type": "text/plain"})]

    def extract(self, cap: Captured, ctx: RunContext) -> list[Doc]:
        name = cap.unit.key
        number = int(preserved.RFC_NAME.fullmatch(name).group(1))
        orig = {"sha256": cap.meta["sha256"], "relative_path": f"input/rfcs/{name}",
                "bytes": len(cap.body)}
        row, quarantine = preserved.rfc_row(number, cap.body, orig)
        if quarantine:
            raise UnitExcluded(quarantine["reason"])
        return [Doc(url=row["url"], text=row["markdown"], title=row["title"], lang=row["lang"],
                    source_text=row["markdown"], extra={"row": row})]

    def preflight(self, ctx: RunContext) -> None:
        self._selection = declared_policy_selection(ctx)

    def chunk(self, doc: Doc, ctx: RunContext):
        """Re-render the raw RFC text into the heading Markdown the sealed pack was chunked from, then
        chunk it under the contract's declared policy. The two families differ only in `ANCHOR` and
        in what `DECLARED` adds to the record."""
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch
        from cytir.extract.rfc_editor import render

        selection = getattr(self, "_selection", None)
        if selection is None:
            raise RuntimeError(f"{self.family_id}.chunk ran before preflight: the declared chunk "
                               "policy is resolved in preflight; call preflight(ctx) first")
        row = doc.extra["row"]
        number = int(row["document_id"].rsplit(":", 1)[1])
        markdown, _ = render(number, doc.source_text, relative_path=row["source_relative_path"],
                             source_sha256=row["source_sha256"], encoding=row["source_text_encoding"])
        stamps = selection.stamps()
        digest = C.sha1(markdown)
        out = []
        for ordinal, unit in enumerate(_dispatch.chunk_markdown_channels(markdown, selection=selection)):
            unit = dataclasses.replace(unit, stamps=stamps)
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(unit.dense_text or unit.text, 25),
                lang=doc.lang, domain=DOMAIN, last_seen=ctx.today, content_digest=digest,
                anchor=self.ANCHOR.format(ordinal=ordinal, document_id=row["document_id"]),
                master={"document_id": row["document_id"], "source_sha256": row["source_sha256"]},
                **self.DECLARED)
            rec = C.passage_source_record(p)
            rec["document_id"], rec["chunk_ordinal"] = row["document_id"], ordinal
            out.append(rec)
        return out

    def seal_inventory(self, ctx: RunContext) -> dict:
        """Seal the whole tree: the old `materialize`, from the same options."""
        self._terms(ctx)
        o = ctx.options
        missing = [k for k in SEAL_OPTIONS if not o.get(k)]
        if missing:
            raise ValueError(f"{self.family_id}.seal_inventory needs --opt {', --opt '.join(missing)}")
        P = pathlib.Path
        return preserved.materialize(
            raw_dir=self._raw_dir(ctx), before_list=P(o["before_list"]).resolve(),
            after_list=P(o["after_list"]).resolve(), root=P(o["root"]).resolve(),
            module=o["module"], started_at=o["started_at"], completed_at=o["completed_at"],
            rsync_version=o["rsync_version"], fetch_vantage=o["fetch_vantage"])


DOMAIN = "rfc-editor.org"
TIER = "A"                      # trust 1.0 / `official`: the row's declared_payload
SOURCE_TYPE = "official"
TRUST_TIER = 1.0
LICENSE = "IETF-TLP-5.0-portion-BCP78"      # cytir/core/license.py, rfc-editor.org


@register_family
class RfcEditorNetNewFamily(_RfcEditorBase):
    family_id = "rfc_editor_net_new"
    DECLARED = {"tier": TIER, "source_type": SOURCE_TYPE, "trust_tier": TRUST_TIER, "license": LICENSE}


@register_family
class RfcExactFamily(_RfcEditorBase):
    family_id = "rfc_exact"
    ANCHOR = "document:{document_id}:chunk-{ordinal}"      # the sealed stage's anchor shape (task 2133)
