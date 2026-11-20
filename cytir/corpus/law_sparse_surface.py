#!/usr/bin/env python3
"""Re-surface an anchored-law pack's SPARSE channel to the family's declared
representation, without moving a single point id.

WHAT IS WRONG WITH THE SHIPPED SURFACE (task 564, measured; task 562 attributed)

`fedlex_dump.parse_akn` -- the live loader, and the path production was built
through -- composes the law's sparse channel as

    "; ".join(fedlex_akn.ancestor_heading_path(article)) + ". " + <article body>

and the comment beside it records why: task 358b proved that truncating the
ancestor path cuts OR art. 257's query-relevant "Die Miete" ancestor, and that
the full path is a SPARSE-ONLY retrieval hook (task 358c measured that putting
it into dense dilutes the article semantics).

The epoch-4 re-materialization
(task 417 fedlex materializer script) emits that breadcrumb as
FLAT sibling `## <label> {#anchor}` headings ahead of the article's own
`## <cite> {#anchor}` heading. `generic_chunking`'s hierarchy channel -- the
channel that fills the pack's `sparse_text` -- therefore sees the article
heading as a SIBLING of the breadcrumb rather than a child of it, and the
breadcrumb never reaches the article's own chunk. The pack's `sparse_text` ends
up carrying the law title repeated per section and nothing else: over the seven
golds task 564 measured, `sparse_text` has 40-100% more characters than `text`
and EXACTLY the same number of distinct BM25 terms.

The same `render()` writes the AKN eId identifiers inline, so 22,927 of 22,927
pack rows carry `{#art_N}` / `{#art_N/para_M}` in both channels against 0 of 299
matched rows in the twin (task 562 §4.3).

WHAT THIS DOES

Per pack row, keeping `point_id` and `canonical_text` untouched (so
`aw_common.mint_passage_point_id` cannot move an id and the delta is
appliable):

    sparse_text := fedlex_akn.compose_law_sparse_surface(
                       heading_ctx, fedlex_akn.strip_anchor_markup(row["text"]))

`heading_ctx` is re-derived from the PINNED AKN XML the row itself names in
`parent_identity.xml` (uri + sha256, verified before the bytes are trusted), so
this replays from L0 with no network and no re-fetch (ADR-026). `text` and
`canonical_text` are copied through unchanged: this is a SPARSE-only change and
the dense surface is task 537's, already A/B'd and applied.

CONSERVATION. Every input row leaves as `written` or as `quarantined` with a
reason, `written + quarantined == read`, and the counts are in the manifest
(ADR-086). A row whose article is absent from its own pinned XML is quarantined,
never dropped and never guessed.

ORDER. The output shards are written in the order of the sealed vectors'
`row_ids.jsonl`, because `cyt index surface-delta` binds row i of the
vectors to line i of the staged texts and refuses a mismatch.

    python3 -m cytir.corpus.law_sparse_surface \
        --pack <pack>/chunks --vectors <sealed vector dir> \
        --corpus /mnt/.../aw-corpus --out <stage dir> --manifest <json>
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import subprocess
import sys

from lxml import etree

from cytir.sources.families import fedlex_akn as FA  # noqa: E402
# tok@2 (AW_FOLD_ACCENTS=1) is an aw_b1 INVARIANT and it is not this
# module's to choose: importing `aw_b1_projection` is how every write-side
# caller in the repo pins it, so the tokeniser here hashes exactly what the
# index hashes. Task 564 measured what happens without it -- the accented
# half of three golds' terms read as ABSENT from postings that hold them.
import cytir.core.projection as _PIN_TOK2  # noqa: E402,F401
from cytir.core.common import compose_index_text
from cytir.core.bm25_tokenizer import bm25_tokens, require_index_fold

VERSION = "aw.law-pack-sparse-resurface.v1"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def resurfaced_sparse_text(rec: dict, heading_ctx: str) -> str:
    """The row's corrected sparse surface -- the ONE definition, so the stage
    writer and the test exercise the same code (a test that re-implements the
    fix tests nothing).

    The article body is the pack's own `text` with the AKN scaffolding removed;
    the hierarchy is the pinned XML's ancestor heading path. `canonical_text`
    and `point_id` are untouched by construction: this function never sees them.
    """
    return FA.compose_law_sparse_surface(
        heading_ctx, FA.strip_anchor_markup(rec.get("text") or ""))


def binder_surface(rec: dict, sparse_text: str, site_name: str | None) -> str:
    """What `cytir.index.surface_delta.sparse_for` will actually tokenise: the
    family's sparse channel with the payload title (and site name) composed over
    it by `aw_common.compose_index_text`.

    The containment check below MUST use this and not the bare sparse text. A
    repealed article's body is only its authorial note -- production's own
    `parse_akn` pointer branch does the same -- and its citation ("Art. 25")
    reaches the posting through the TITLE. Comparing against the bare surface
    quarantined 925 of 22,901 rows on exactly that mismatch, all of them
    `repealed` (855) or `source_note` (70): the check was wrong, not the rows.
    """
    return compose_index_text({
        "text": sparse_text,
        "title": (rec.get("source_metadata") or {}).get("title"),
        "site_name": site_name,
        "source_type": rec.get("source_type")})


def sealed_order(vectors: str) -> list[str]:
    """The point ids in the order the sealed vectors carry them."""
    ids: list[str] = []
    mans = sorted(glob.glob(os.path.join(vectors, "shard-*.vectors.manifest.json")))
    if not mans:
        raise SystemExit(f"no vector manifests under {vectors}")
    for man_path in mans:
        tag = os.path.basename(man_path).removesuffix(".vectors.manifest.json")
        rid = os.path.join(vectors, tag + ".row_ids.jsonl")
        ids += [json.loads(line)["point_id"] for line in open(rid)]
    return ids


class XmlCache:
    """Pinned AKN XML, read once per document and digest-checked before use."""

    def __init__(self, corpus: str):
        self.corpus = corpus
        self._trees: dict[str, etree._Element] = {}
        self.documents = 0

    def article(self, xml_ref: dict, anchor: str):
        uri = xml_ref["uri"]
        if uri not in self._trees:
            assert uri.startswith("artifact://"), uri
            path = os.path.join(self.corpus, uri[len("artifact://"):])
            blob = open(path, "rb").read()
            sha = hashlib.sha256(blob).hexdigest()
            if sha != xml_ref["sha256"]:
                raise SystemExit(f"pinned XML digest drift: {uri} {sha} != "
                                 f"{xml_ref['sha256']}")
            self._trees[uri] = etree.fromstring(blob)
            self.documents += 1
        root = self._trees[uri]
        for node in root.iter(FA.tag("article")):
            if (node.get("eId") or node.get("id")) == anchor:
                return node
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="the pack's chunks/ directory")
    ap.add_argument("--vectors", required=True,
                    help="the sealed vector dir whose row_ids fix the order")
    ap.add_argument("--corpus", required=True, help="artifact:// root")
    ap.add_argument("--out", required=True, help="stage directory to write")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--site-name", default=None,
                    help="the family's site_name, as the collection's payload "
                         "carries it (fedlex: 'Fedlex' -- task 526 proved that "
                         "IS production's surface, cos 0.999999). Only the "
                         "containment check reads it; the stage never writes "
                         "it, because the applier composes it from the LIVE "
                         "payload, not from the stage.")
    ap.add_argument("--rows-per-shard", type=int, default=0,
                    help="0 = mirror the vector shards exactly (the default, "
                         "and the only value the applier's row_ids binding "
                         "accepts without a re-shard)")
    a = ap.parse_args()

    # fail closed before a byte is read: this stage's report counts terms,
    # and a report counted at the wrong tokenizer epoch is worse than none.
    fold = require_index_fold(who="resurface_law_pack_sparse")

    order = sealed_order(a.vectors)
    order_pos = {pid: i for i, pid in enumerate(order)}
    if len(order_pos) != len(order):
        raise SystemExit("the sealed row_ids repeat a point id")
    print(f"[resurface] sealed order: {len(order):,} ids", flush=True)

    shards = sorted(glob.glob(os.path.join(a.pack, "shard-*.texts.jsonl.gz")))
    print(f"[resurface] pack shards rostered from the tree: {len(shards)}",
          flush=True)
    if not shards:
        raise SystemExit(f"no pack shards under {a.pack}")

    # How many chunks did each parent produce? A parent that produced exactly
    # one chunk has its WHOLE article in that chunk, so the article's own text
    # -- `fedlex_akn.clean_text(article)`, which is what production embedded --
    # must be wholly inside the re-surfaced surface. That containment check is
    # the only thing standing between `strip_anchor_markup` and a regex that
    # eats a line of law, and it cannot be asked of a multi-chunk parent, whose
    # chunk legitimately holds only part of the article.
    chunks_per_document: collections.Counter = collections.Counter()
    for sh in shards:
        with gzip.open(sh, "rt", encoding="utf-8") as fh:
            for line in fh:
                chunks_per_document[json.loads(line).get("document_id")] += 1
    single = sum(1 for v in chunks_per_document.values() if v == 1)
    print(f"[resurface] parents: {len(chunks_per_document):,}, "
          f"single-chunk: {single:,}", flush=True)

    cache = XmlCache(a.corpus)
    read = 0
    slots: list[dict | None] = [None] * len(order)
    quarantined: list[dict] = []
    unplaced: list[dict] = []
    stats = collections.Counter()
    ctx_lens: list[int] = []

    for sh in shards:
        with gzip.open(sh, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                read += 1
                pid = str(rec.get("point_id") or "")
                pi = rec.get("parent_identity") or {}
                anchor = pi.get("article_anchor") or rec.get("article_anchor")
                art = None
                try:
                    if pi.get("xml") and anchor:
                        art = cache.article(pi["xml"], anchor)
                except SystemExit:
                    raise
                except Exception as exc:                      # noqa: BLE001
                    quarantined.append({"point_id": pid, "url": rec.get("url"),
                                        "reason": f"xml_unreadable:{exc}"})
                    continue
                if art is None:
                    quarantined.append({"point_id": pid, "url": rec.get("url"),
                                        "reason": "article_absent_from_pinned_xml",
                                        "anchor": anchor})
                    continue
                path, _marginal = FA.ancestor_heading_path(art)
                heading_ctx = "; ".join(path)
                ctx_lens.append(len(path))
                body = FA.strip_anchor_markup(rec.get("text") or "")
                sparse = resurfaced_sparse_text(rec, heading_ctx)
                if FA.ANCHOR_MARKUP.search(sparse):
                    quarantined.append({"point_id": pid, "url": rec.get("url"),
                                        "reason": "anchor_markup_survived_strip"})
                    continue
                if chunks_per_document.get(rec.get("document_id")) == 1:
                    want = set(bm25_tokens(FA.norm(FA.clean_text(art)),
                                           rec.get("lang")))
                    got = set(bm25_tokens(binder_surface(rec, sparse, a.site_name),
                                          rec.get("lang")))
                    if not want <= got:
                        quarantined.append({
                            "point_id": pid, "url": rec.get("url"),
                            "reason": "article_tokens_lost_by_resurfacing",
                            "missing": sorted(want - got)[:8]})
                        continue
                    stats["single_chunk_containment_checked"] += 1
                if not path:
                    stats["rows_without_ancestor_path"] += 1
                if sparse != (rec.get("sparse_text") or ""):
                    stats["sparse_text_changed"] += 1
                out = dict(rec)
                out["sparse_text"] = sparse
                out["heading_ctx"] = heading_ctx
                out["sparse_body"] = body
                out["resurfaced_by"] = VERSION
                pos = order_pos.get(pid)
                if pos is None:
                    unplaced.append({"point_id": pid, "url": rec.get("url"),
                                     "reason": "not_in_sealed_vector_row_ids"})
                    continue
                slots[pos] = out

    missing = [order[i] for i, v in enumerate(slots) if v is None]
    if missing:
        raise SystemExit(
            f"{len(missing)} sealed ids have no pack record "
            f"(first: {missing[:3]}) -- the stage would not align with the "
            f"vectors and the applier's row_ids check would refuse it")

    os.makedirs(a.out, exist_ok=True)
    written = 0
    logical = hashlib.sha256()
    mans = sorted(glob.glob(os.path.join(a.vectors, "shard-*.vectors.manifest.json")))
    cursor = 0
    for man_path in mans:
        tag = os.path.basename(man_path).removesuffix(".vectors.manifest.json")
        n = json.load(open(man_path))["n"]
        path = os.path.join(a.out, tag + ".texts.jsonl.gz")
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for row in slots[cursor:cursor + n]:
                blob = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                fh.write(blob)
                logical.update(blob.encode())
                written += 1
        cursor += n
        print(f"[resurface] {tag}: {n:,} rows", flush=True)

    # The posting the binder built BEFORE (pre-528a it fed `text` to both legs)
    # against the one it will build AFTER, over the first 2,000 rows. Composed
    # through `binder_surface`, the same function the containment check uses, so
    # the report and the guard cannot disagree about what gets tokenised.
    surface_terms_before = surface_terms_after = 0
    sample = slots[:2000]
    for row in sample:
        pl_lang = row.get("lang")
        before = binder_surface(row, row.get("text") or "", a.site_name)
        after = binder_surface(row, row["sparse_text"], a.site_name)
        surface_terms_before += len(bm25_tokens(before, pl_lang))
        surface_terms_after += len(bm25_tokens(after, pl_lang))

    manifest = {
        "schema": "aw.law-pack-sparse-resurface-manifest.v1",
        "version": VERSION,
        "fold_accents": fold,
        "site_name": a.site_name,
        "git_sha": git_sha(),
        "pack": a.pack, "vectors": a.vectors, "corpus": a.corpus, "out": a.out,
        "pinned_xml_documents_read": cache.documents,
        "conservation": {
            "read": read, "written": written,
            "quarantined": len(quarantined),
            "unplaced": len(unplaced),
            "balances": read == written + len(quarantined) + len(unplaced),
        },
        "sealed_row_ids": len(order),
        "counts": dict(stats),
        "ancestor_path_depth": {
            "rows": len(ctx_lens),
            "min": min(ctx_lens) if ctx_lens else None,
            "max": max(ctx_lens) if ctx_lens else None,
            "mean": round(sum(ctx_lens) / len(ctx_lens), 3) if ctx_lens else None,
        },
        "distinct_bm25_terms_over_first_2000_rows": {
            "shipped_surface": surface_terms_before,
            "resurfaced": surface_terms_after,
        },
        "logical_sha256": logical.hexdigest(),
        "quarantined_examples": quarantined[:20],
        "unplaced_examples": unplaced[:20],
    }
    json.dump(manifest, open(a.manifest, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: manifest[k] for k in (
        "conservation", "counts", "ancestor_path_depth",
        "distinct_bm25_terms_over_first_2000_rows",
        "pinned_xml_documents_read")}, indent=1))
    if not manifest["conservation"]["balances"]:
        raise SystemExit("conservation failed: read != written + quarantined + unplaced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
