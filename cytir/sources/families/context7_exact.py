"""`context7_exact` -- preserved Context7 GitHub documents, on the Protocol (task 1411 + task 2134, S7), FULL.

What the producer is. `cytir.sources.context7.materialize` reads
a preserved acquisition ROOT (one terminal receipt per repository/ref pair, each holding the exact
pinned-commit blobs) and writes sealed Markdown / outlink / recovery stages. It contacts no host.
Nothing in this code contacts context7.com: "Context7" names the repository/path list the blobs were
acquired from (by an acquirer that is not in the package); the document text is the repository's own
file at a 40-hex commit, at `https://github.com/<repo>/blob/<sha>/<path>`.

Stages. `discover` (a unit is one terminal receipt) -> `acquire` (the receipt's own bytes) -> `extract`
(the documents of that receipt: `receipt_rows`, the SAME function `materialize` runs) -> `chunk` are on
the Protocol. `chunk` (task 2134) runs the structured chunker under the contract's DECLARED policy
(`7dcb54c4...@4`, the policy of the 27 sealed stages, task 2133) over the extracted Markdown, with the
point-identity anchor `document:<id>:chunk-<n>` the sealed stage mints; byte-equal on sealed rows.
The record carries NO tier / source_type / trust_tier: the sealed stage rows hold none of them (their keys are
document_id, chunk_ordinal, url, lang, text channels, point_id, prov_id, source_metadata{repository, revision,
source_path}, ...; task 2134 read them), so they are bound later at assembly and are Tim's choice. They are
passed as None, which `Passage.payload()` drops.
The sealed-stage run (`materialize`: original/markdown/outlinks/recovery receipts, the label
frozen in them) stays a function of the moved module; the shell chains that call it are deploy-side.

Options (`--opt KEY=VALUE`): `root` (required): the acquisition root holding `receipts/`;
`artifact_root` (required): the global artifact root the receipts' artifact URIs resolve against.

Dispositions, said. A receipt with no `complete` document is EXCLUDED (`recovery_required:N`): the
materializer's own recovery stage, 1:1. A schema/status drift or an unpinned revision raises: the unit
is QUARANTINED, never an empty. Consent and licence are read from the contract row and are `unknown`
today; the code asserts nothing about the origin's terms and sets no consent field.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from cytir.sources.chunk_selection import declared_policy_selection
from cytir.sources.context7 import materialize as M
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

HOSTS = ("github.com",)


@register_family
class Context7ExactFamily:
    family_id = "context7_exact"
    route = "unknown"
    direct_crawl = False
    chunk_policy_id = CHUNK_DECLARED

    def __init__(self) -> None:
        self._root = Path()
        self._artifacts = Path()

    def discover(self, ctx: RunContext):
        root, art = ctx.options.get("root"), ctx.options.get("artifact_root")
        if not root or not art:
            raise ValueError(f"{self.family_id} needs --opt root=<acquisition root with receipts/> "
                             "and --opt artifact_root=<global artifact root>")
        self._root, self._artifacts = Path(root), Path(art)
        paths = sorted((self._root / "receipts").glob("*.json"))
        if not paths:
            raise RuntimeError(f"{self.family_id}: {self._root / 'receipts'} holds no terminal receipt")
        ctx.notes.update({"root": str(root), "receipts": len(paths)})
        return [Unit(key=p.stem, url=f"https://github.com/{p.stem.replace('--', '/', 1)}",
                     meta={"path": str(p)}, consent_hosts=HOSTS) for p in paths]

    def acquire(self, unit: Unit, ctx: RunContext):
        path = Path(unit.meta["path"])
        return [Captured(unit, path.read_bytes(), {"receipt": path.name})]

    def extract(self, cap: Captured, ctx: RunContext):
        markdown, outlinks, recovery = M.receipt_rows(Path(cap.unit.meta["path"]), self._artifacts)
        if not markdown:
            raise UnitExcluded(f"recovery_required:{len(recovery)}")
        by_doc: dict[str, list[dict]] = {}
        for link in outlinks:
            by_doc.setdefault(link["document_id"], []).append(link)
        return [Doc(url=r["url"], text=r["markdown"], title=r["title"] or "", lang="",
                    extra={"row": r, "outlinks": by_doc.get(r["document_id"], []),
                           "recovery": recovery})
                for r in markdown]

    def preflight(self, ctx: RunContext) -> None:
        self._selection = declared_policy_selection(ctx)

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch

        selection = getattr(self, "_selection", None)
        if selection is None:
            raise RuntimeError("context7_exact.chunk ran before preflight: the declared chunk "
                               "policy is resolved in preflight; call preflight(ctx) first")
        row, out = doc.extra["row"], []
        stamps = selection.stamps()
        digest = C.sha1(doc.text)
        for ordinal, unit in enumerate(_dispatch.chunk_markdown_channels(doc.text, selection=selection)):
            unit = dataclasses.replace(unit, stamps=stamps)
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(unit.dense_text or unit.text, 25),
                lang=doc.lang, domain=HOSTS[0], tier=None, source_type=None,
                trust_tier=None, last_seen=ctx.today, content_digest=digest,
                anchor=f"document:{row['document_id']}:chunk-{ordinal}")
            rec = C.passage_source_record(p)
            rec["document_id"], rec["chunk_ordinal"] = row["document_id"], ordinal
            out.append(rec)
        return out
