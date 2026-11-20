"""`cyt ops snippet-repatch`: re-derive the served `snippet` of live rows from their TEXT (task 2734).

WHY. Task 2733 (fedlex Akoma `{#...}` anchors) and task 2737 (the `## Record identity` provenance
block of bfs / swisstopo / pubmed) fixed the snippet the drain projects from now on. The rows that
are already live (5.6M twin points, the same in prod) still carry the old snippet. This verb
produces the PATCH FILE that `cyt ops twin-patch-fields --fields snippet` applies on the twin and the
prod delta pack carries to prod.

  * SELECT the twin rows of `--domain` (a filtered scroll, payload keys `content_digest`+`snippet`).
  * TEXT comes from `--text-source`: a sealed pack's `shard-*.texts.jsonl.gz` (field `text`) or a
    corpus wave's `part-*.parquet` (column `text`). A row is joined BY `content_digest`: the
    payload digest is the first 32 hex of `ids.content_digest(text)` of the chunk text (`project_record`), so the
    join proves itself, and a pack whose point ids drifted (task 564 finding B) is no problem.
    A row no text hashes to is `no_text` and is LEFT ALONE: a snippet is never derived from the
    payload (that would only shorten the old one).
  * The new snippet is `b1_projection.projected_snippet(domain, text)`, the function the drain
    calls. Equal to the current one: `skipped_equal`.
  * OUTPUT `<out>/snippet-patch.jsonl` in the `twin-patch-fields` input format
    (`{"id", "set": {"snippet": new}, "unset": [], "expect": {"content_digest", "snippet": old}}`:
    the old snippet is the precondition and the rollback), `<out>/snippet-patch.manifest.json`
    (counts per domain + sha256 of the file). Read only: no write primitive is reached.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import pathlib
import sys
from collections import Counter
from typing import Callable, Iterable, Iterator

from cytir.core import ids as ID
from cytir.index import point_delete as PD
from cytir.index import prod_guard as PG
from cytir.index import row_decode as RD

SCHEMA = "aw.snippet-repatch-manifest.v1"
DOMAINS = ("fedlex.admin.ch", "bfs.admin.ch", "swisstopo.admin.ch", "pubmed.ncbi.nlm.nih.gov")
PAGE = 1000


class Refused(RuntimeError):
    pass


def _key(digest: str) -> str:
    """The join key: the twin stores the first 32 hex of `ids.content_digest` (verified 2026-11-12)."""
    return digest[:32]


def scroll_domain(url: str, coll: str, domain: str, page: int = PAGE) -> Iterator[dict]:
    """Twin rows of one domain: `{"id", "digest", "snippet"}` (filtered scroll, no vectors)."""
    dec = RD.reader(coll)
    offset = None
    while True:
        body = {"limit": page, "with_vector": False,
                "with_payload": {"include": ["content_digest", "snippet"]},
                "filter": {"must": [{"key": "domain", "match": {"value": domain}}]}}
        if offset is not None:
            body["offset"] = offset
        res = PD._api(url, f"/collections/{coll}/points/scroll", body)
        for p in res.get("points", []):
            pl = dec.decode(p.get("payload") or {})          # canonical row
            yield {"id": str(p["id"]), "digest": RD.digest_key(pl.get("content_digest")),  # canonical row
                   "snippet": pl.get("snippet")}
        offset = res.get("next_page_offset")
        if offset is None:
            return


def pack_texts(path: str) -> Iterator[str]:
    for f in sorted(glob.glob(os.path.join(path, "shard-*.texts.jsonl.gz"))):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                t = json.loads(line).get("text")
                if t:
                    yield t


def parquet_texts(path: str) -> Iterator[str]:
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(os.path.join(path, "part-*.parquet"))):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(columns=["text"], batch_size=20000):
            for t in batch.column("text").to_pylist():
                if t:
                    yield t


def texts_of(spec: str) -> Iterator[str]:
    kind, _, path = spec.partition("=")
    if kind == "pack":
        return pack_texts(path)
    if kind == "parquet":
        return parquet_texts(path)
    raise Refused(f"--text-source {spec!r}: expected pack=DIR or parquet=DIR")


def build_patch(rows_by_domain: dict[str, Iterable[dict]], text_sources: Iterable[Iterable[str]],
                project: Callable[[str, str], str]) -> tuple[list[dict], dict]:
    """(patch rows, per-domain counts). `rows_by_domain`: domain -> twin rows. Pure of I/O."""
    want: dict[str, list[tuple[str, dict]]] = {}     # digest -> [(domain, row)]
    counts: dict[str, Counter] = {}
    for dom, rows in rows_by_domain.items():
        c = counts.setdefault(dom, Counter())
        for r in rows:
            c["selected"] += 1
            if r.get("digest"):
                want.setdefault(_key(r["digest"]), []).append((dom, r))
            else:
                c["no_text"] += 1
    done: dict[str, dict[str, str]] = {}               # digest -> {domain: new snippet}
    for src in text_sources:
        for text in src:
            d = _key(ID.content_digest(text))
            if d not in want or d in done:
                continue
            done[d] = {dom: project(dom, text) for dom in {dom for dom, _ in want[d]}}
    patch = []
    for d, items in want.items():
        for dom, r in items:
            c = counts[dom]
            new = (done.get(d) or {}).get(dom)
            if new is None:
                c["no_text"] += 1
                continue
            c["recomputed"] += 1
            if new == r["snippet"]:
                c["skipped_equal"] += 1
                continue
            c["changed"] += 1
            patch.append({"id": r["id"], "set": {"snippet": new}, "unset": [],
                          "expect": {"content_digest": r["digest"], "snippet": r["snippet"]}})
    patch.sort(key=lambda p: p["id"])
    return patch, {d: dict(c) for d, c in counts.items()}


def write_outputs(out: pathlib.Path, patch: list[dict], counts: dict, meta: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in patch).encode()
    f = out / "snippet-patch.jsonl"
    with open(f, "xb") as fh:                            # never overwrite a pinned patch
        fh.write(data)
    manifest = {"schema": SCHEMA, "patch": f.name, "sha256": hashlib.sha256(data).hexdigest(),
                "rows": len(patch), "counts": counts, **meta}
    (out / "snippet-patch.manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None, prog: str = "cyt ops snippet-repatch") -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", action="append", required=True, choices=DOMAINS)
    ap.add_argument("--text-source", action="append", required=True,
                    help="pack=DIR (shard-*.texts.jsonl.gz) or parquet=DIR (part-*.parquet); repeatable")
    ap.add_argument("--out", required=True, help="new directory for the patch + manifest")
    ap.add_argument("--twin-url")
    ap.add_argument("--twin-collection")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    from cytir.core import twin as TW
    from cytir.index.b1_projection import projected_snippet
    url = (a.twin_url or TW.url()).rstrip("/")
    coll = a.twin_collection or TW.collection()
    if PG.is_prod(url, coll):
        print(f"REFUSED: {url} ({coll}) is prod; the producer reads the twin only")
        return 4
    try:
        rows = {d: scroll_domain(url, coll, d) for d in a.domain}
        patch, counts = build_patch(rows, [texts_of(s) for s in a.text_source], projected_snippet)
        manifest = write_outputs(pathlib.Path(a.out), patch, counts,
                                 {"twin": {"url": url, "collection": coll},
                                  "text_sources": a.text_source})
    except (Refused, FileExistsError) as exc:
        print(f"REFUSED: {exc}")
        return 4
    print(json.dumps({"rows": manifest["rows"], "sha256": manifest["sha256"], "counts": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
