"""The Markdown row builder: one fetched Markdown document -> the b1-native rows the refresh lane stages.

Moved here from the refresh lane (task 1157, S7), AST-identical: `docs_chunk_selection`, `require_docs_chunker` and
`build_rows_for_markdown` are the lane's own defs, byte for byte. A family adapter that has Markdown (a repo doc,
a PDF's extraction) and wants rows calls `build_rows_for_markdown`; the lane re-exports the names unchanged.
Task 1169 adds `stamp_docs_row`, the second half of "a Markdown doc -> the rows the lane serves": the fields the builder
cannot infer (`published`, `last_seen`, `rule_epochs`, `license`). It sits here, not in a materialize seam, because
`rule_epochs` is read off `docs_chunk_selection(meta)`, the chunk policy this module owns, and no `cytir/` materialize
module stamps rows (family adapters stamp their own `last_seen` from `ctx.today`). The lane's own def is now a thin
wrapper that supplies the clock; the package never reads one.

This module is the SEAM to the chunker, not a second chunker. It owns nothing but the choice of
policy: for `source_type == "docs"` rows the pinned epoch-4 `chunk.generic` selection (RULING 39, task 976),
for every other family `None`, so `cytir.core.common.passages_from_markdown` applies the process-wide selection.
The language gate (`cytir.acquire.fetch_process.detect_lang`, lid.176) and the embedder tokenizer
(`cytir.chunk.chunker._token_len` -> `cytir.chunk.token_counter`) are reached THROUGH `passages_from_markdown` and the
chunker, the same way every other caller reaches them; nothing is injected or duplicated here. A box without the
models refuses (`LangGateInactive`; `require_docs_chunker`), as the lane always did.
"""
from __future__ import annotations

import json
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import cytir.core.common as AC
from cytir.extract import html_extract as _html_extract
from cytir.chunk import registry as CHREG
from cytir.core.model_pin import EMBED_MODEL_REVISION
from cytir.core.paths import data_path

# -- the chunk policy docs rows are cut under (task 976, RULING 39) ------------------------------
# RULING 39 (Tim 2026-10-17): the recurring lanes force the epoch-4 chunk.generic policy for
# `source_type: docs` rows, overriding the ratified selector v4 (which would pick chunk.reference@5
# for short pages). Reason: the rebuilt docs collection holds these sites at generic width; the
# legacy 180w window of the tracked build profile cut them ~4x finer. News and every other family
# on this path keep their current behaviour (ADR-090 section 5 keeps news on legacy).
# The policy is resolved from the tracked selector artifact (path + exact sha256) and verified
# against the registry; no word count is written here.
DOCS_SOURCE_TYPE = "docs"
_SELECTOR_ARTIFACT = data_path("chunk-reference-selector.v4.json")
_DOCS_SELECTION: dict = {}


def _pinned_policy_path(pinned: str):
    """The pinned policy file, resolved through the same `data_path()` seam as the selector (task 1169): the
    selector names it as `data/<file>`; in an installed wheel `data/` is the package data dir, not a sibling."""
    parts = PurePosixPath(pinned).parts
    if parts[0] != "data":
        raise ValueError(f"the selector pins {pinned!r}; expected a path under data/")
    return data_path(*parts[1:])


def docs_chunk_selection(meta: dict):
    """The pinned epoch-4 generic `ChunkerSelection` for a docs row, else None (current behaviour)."""
    if meta.get("source_type") != DOCS_SOURCE_TYPE:
        return None
    if not _DOCS_SELECTION:
        from cytir.core import common as ACOM  # noqa: PLC0415
        from cytir.chunk import dispatch as _dispatch
        pin = json.loads(_SELECTOR_ARTIFACT.read_text(encoding="utf-8"))["generic_policy"]
        path = _pinned_policy_path(pin["file"])
        sel = _dispatch.explicit_structured_chunker_selection(
            str(path), pin["sha256"], EMBED_MODEL_REVISION)
        if CHREG.family_of(sel.policy_sha256) != "chunk.generic" or \
                CHREG.status_of(sel.policy_sha256) != "current":
            raise SystemExit("ABORT: the docs chunk policy is not registry-current chunk.generic")
        _DOCS_SELECTION["sel"] = sel
    return _DOCS_SELECTION["sel"]


def require_docs_chunker() -> None:
    """Refuse a run on a box that cannot measure tokens (the structured chunker needs the tokenizer)."""
    from cytir.chunk import chunker as CH  # noqa: PLC0415
    sel = docs_chunk_selection({"source_type": DOCS_SOURCE_TYPE})
    CH.select_tokenizer_revision(sel.tokenizer_revision)
    from cytir.core import common as ACOM  # noqa: PLC0415
    from cytir.chunk import dispatch as _dispatch
    _dispatch.chunk_markdown_channels("# t\n\nprobe text for the tokenizer.\n", sel)


def build_rows_for_markdown(url: str, markdown: str, *, title: str | None, meta: dict) -> list[dict]:
    """The Markdown twin of `build_rows_for_candidate` (task 972): the candidate IS Markdown (a repo
    doc), so it goes through `cytir.core.common.passages_from_markdown`, the tail `process_html`
    itself uses -- same language gate, chunker, quality gates and `Passage` stamping. No HTML wrapper."""
    domain = urlsplit(url).netloc.lower()
    common = {"domain": domain, "tier": meta["tier"], "source_type": meta["source_type"],
              "trust_tier": meta["trust_tier"]}
    passages = _html_extract.passages_from_markdown(
        url, markdown, title=title or url, site_name=None, common=common,
        lang_default="en", selection=docs_chunk_selection(meta))
    rows = []
    for p in passages:
        row = p.payload()
        row["text"] = p.text
        row["chunk_n_words"] = len(p.text.split())   # task 984: the composer's content_words_by_url
        rows.append(row)
    return rows


def stamp_docs_row(row: dict, *, meta: dict, published: str | None, license_: str | None, today: str) -> dict:
    """Fields a docs delta row owes that the extractor cannot infer: `published` (see PACK_META),
    this crawl's `last_seen` (the supersede key; `today` comes from the caller, never the package's clock), and the pack's licence string."""
    # `published` is NEVER a capture date (ruling, task 973): no lastmod -> no key at all
    if row.get("published") or published:
        row["published"] = row.get("published") or published
    else:
        row.pop("published", None)
    row["last_seen"] = today
    sel = docs_chunk_selection(meta)
    if sel:   # RULING 39: the staging step must carry this epoch, not stamp its build default
        row["rule_epochs"] = {"chunk": sel.epoch}
    if license_:
        row["license"] = license_
    return row
