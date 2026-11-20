"""Wikipedia dump loader: a Wikimedia XML dump → passages.jsonl.

The canonical, structured Wikipedia source — no crawling, no JS, complete.
Raw XML dumps at
  https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2
are refreshed ~biweekly (fresher than the few-times-a-year HF parquet, which is
a stale Nov-2023 snapshot). Articles only (ns 0), redirects skipped, wikitext
stripped to plain text (markdown via the HTML/Enterprise dump is a later upgrade).

  python -m cytir.extract.wikipedia_dump --dump dewiki.xml.bz2 --lang de --out build/wiki_de.jsonl [--limit N]
"""
from __future__ import annotations

import argparse
import bz2
import datetime as dt
import json
import pathlib

from cytir.core.common import extractive_snippet, passage_from_chunk, passage_source_record
from cytir.chunk.generic import chunk_generic_records

TODAY = dt.date.today().isoformat()


def open_dump(path: str):
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


LEGACY_CHUNK_EPOCH = 1


def chunk_anchor(chunk_index: int, stamps: dict | None) -> str | None:
    """Preserve v1 IDs; drop the old ordinal only for a structured epoch.

    ``chunk-N`` is part of the legacy dump's byte/identity contract, even
    though it is not a source-native anchor.  The tracked profile stays
    legacy until the atomic chunk@2 cutover, so changing that representation
    early would rewrite IDs into the epoch-1 collection.  Structured output
    must not use positional pseudo-anchors (the Tier-B ruling in
    specs/identity.md §4).

    task 407: this used to branch on ``stamps is None``, which was a PROXY for
    "legacy" that held only while the legacy chunker left rows unstamped.
    Stamping every row (so a row can name its chunker) would have silently
    re-minted every legacy wiki point id through that proxy.  The condition is
    now the thing it always meant — the chunk epoch — so provenance and
    identity stop being coupled through an absence.
    """
    epoch = (stamps or {}).get("chunk", LEGACY_CHUNK_EPOCH)
    if epoch == LEGACY_CHUNK_EPOCH:
        return None if chunk_index == 0 else f"chunk-{chunk_index}"
    return None


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    # Keep the pure identity helper importable by the focused contract tests;
    # parsing a dump still fails normally if this optional producer dependency
    # is not installed.
    ap = argparse.ArgumentParser(prog=prog or "wikipedia_dump")
    ap.add_argument("--dump", required=True, help="path to *-pages-articles.xml(.bz2)")
    ap.add_argument("--lang", required=True, help="wiki language code (de/fr/it/en)")
    ap.add_argument("--out", default="build/wiki.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="max articles (0 = all)")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="cap chunks per article (1 = page-level lead only, for scale)")
    args = ap.parse_args(argv)
    import mwparserfromhell
    import mwxml

    common = dict(domain="wikipedia.org", tier="B", trust_tier=0.82,
                  source_type="reference", last_seen=TODAY)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dump = mwxml.Dump.from_file(open_dump(args.dump))
    n = arts = skipped = 0
    with out.open("w") as f:
        for page in dump:
            if page.namespace != 0 or page.redirect:   # articles only
                continue
            text = None
            for rev in page:
                text = rev.text                          # last revision = current
            if not text:
                continue
            plain = mwparserfromhell.parse(text).strip_code()
            if len(plain) < args.min_chars:
                skipped += 1
                continue
            url = "https://%s.wikipedia.org/wiki/%s" % (args.lang, page.title.replace(" ", "_"))
            chunks = chunk_generic_records(plain)
            for ci, unit in enumerate(chunks):
                anchor = chunk_anchor(ci, unit.stamps)
                p = passage_from_chunk(
                    unit, url=url, title=page.title,
                    snippet=extractive_snippet(unit.text), lang=args.lang,
                    anchor=anchor, **common)
                row = passage_source_record(p)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                if args.max_chunks and ci + 1 >= args.max_chunks:
                    break   # page-level: lead section only
            arts += 1
            if arts % 5000 == 0:
                print("  …%d articles, %d passages" % (arts, n))
            if args.limit and arts >= args.limit:
                break
    print("wrote %d passages from %d articles (%d too-short) -> %s" % (n, arts, skipped, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
