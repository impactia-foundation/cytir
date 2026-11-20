#!/usr/bin/env python3
"""Write L1 documents into the ADR-026 store, so a lane's own rows are loadable.

WHY (task 581, ADR-105)
-----------------------
The daily news stage already extracts every article to markdown and hashes it
(`news_documents.jsonl`, url + markdown + markdown_sha256 — the
`aw.l1-markdown.v1` shape). It writes that into
`<work-root>/newsfresh_294/stage-<date>/`, a scratch tree outside
every ADR-026 layer. Measured on the 2026-09-11/12 stages: **every news source
fails the persisted-inputs guard** (news 5,188 pages refused of 5,189,
swissinfo 1,603 of 1,603, chmedia 3,585 of 3,586, hn 488 of 494) — and the
stage's OWN L1 holds 5,188 of the 5,676 refused pages. The text was never
missing. It was never filed (lesson L581-2).

So the fix is not to widen the guard to cover a scratch directory. It is to
file the records: this module writes them into
`l1/release=<release>/source=<source>/l1-markdown-*.jsonl.gz`, which the
tracked store registry already covers with `l1/**/*.jsonl.gz`. Nothing in the
registry changes; the lane starts persisting, and then it passes.

CONTRACT
--------
* one JSON object per line, `aw.l1-markdown.v1`: `url`, `canonical_url`,
  `markdown`, `markdown_sha256`, plus whatever else the caller carries;
* a record with no url or no text is REFUSED, not skipped — a writer that
  silently drops rows is how a population stops being mentioned;
* `markdown_sha256` is recomputed here and must match if the caller supplied
  one (a sha nobody verifies is decoration);
* the file is written to a temp name and renamed, so a crashed run leaves no
  half-file that a later index would read as a store;
* a `RECEIPT.json` records what was written, from what, when, and by which
  git revision.

Returned: `{canonical_url: written_file_path}` — the caller hands that to
`persisted_inputs.overlay_from_paths` so rows persisted in THIS run pass the
guard before the next index rebuild. The overlay is not a bypass: every entry
is checked to be a real file inside a registered store.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Iterable, Optional

import cytir.core.common as AC  # noqa: E402
from cytir.core import paths as aw_paths  # noqa: E402

SCHEMA = "aw.l1-markdown.v1"
RECEIPT_SCHEMA = "aw.l1-persist-receipt.v1"
#: records per output shard -- small enough that one corrupt file costs little,
#: large enough that a daily run is one or two files.
SHARD_ROWS = 50_000


class L1PersistError(RuntimeError):
    """A record this module refuses to write, or a store it cannot write to."""


def _git_sha(repo: Optional[str] = None) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo or str(aw_paths.REPO_ROOT),
            capture_output=True, text=True, timeout=20).stdout.strip() or "?"
    except Exception:
        return "?"


#: field names a lane may carry its page address and its text under. A lane
#: whose names are not here passes them explicitly (`--url-field`/`--text-field`
#: on the CLI); the module does not grow a per-lane special case.
URL_FIELDS = ("url", "canonical_url", "final_url", "requested_url")
TEXT_FIELDS = ("markdown", "text", "body_markdown")


def normalize(record: dict, *, origin: str,
              url_fields: Iterable[str] = URL_FIELDS,
              text_fields: Iterable[str] = TEXT_FIELDS) -> dict:
    """One input record -> one `aw.l1-markdown.v1` record, or raise."""
    url = next((record[f] for f in url_fields if record.get(f)), None)
    text = next((record[f] for f in text_fields if record.get(f)), None)
    if not url:
        raise L1PersistError(f"record has no url: {sorted(record)[:8]}")
    if not text or not str(text).strip():
        raise L1PersistError(f"record for {url} carries no text")
    text = str(text)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    declared = record.get("markdown_sha256")
    if declared and declared != sha:
        raise L1PersistError(
            f"{url}: declared markdown_sha256 {declared[:16]}... does not "
            f"match the text ({sha[:16]}...). A sha nobody verifies is "
            "decoration; refusing to file a record whose own hash is wrong.")
    out = {"schema": SCHEMA,
           "url": url,
           "canonical_url": AC.canonical_url(url),
           "markdown": text,
           "markdown_sha256": sha,
           # WHAT THIS TEXT IS. A curated manifest body and a full article
           # extraction are both "the input this lane embedded", and both
           # belong in the store -- but a reader must be able to tell them
           # apart without archaeology (corpus-doctrine 1). Every record says
           # which lane filed it and what kind of text it holds.
           "l1_origin": origin}
    for k in ("title", "headline", "lang", "publisher", "domain",
              "fetched_at", "publication_time", "document_id",
              "source_record_id", "consent_checked_at",
              # task 588: a record filed from a scratch tree carries its own
              # extraction tool/version and trust/consent stamp -- dropping
              # them here would file the text but lose the provenance that
              # makes it a persisted L1 document rather than a bare string.
              "extraction_rule", "extraction_version", "trust_tier", "tier",
              "source_type", "consent_search_ok", "consent_train_ok",
              "license", "license_version", "wave"):
        if record.get(k) is not None:
            out.setdefault(k, record[k])
    return out


def write_l1_records(records: Iterable[dict], *, release: str, source: str,
                     origin: str, root: Optional[str] = None, note: str = "",
                     url_fields: Iterable[str] = URL_FIELDS,
                     text_fields: Iterable[str] = TEXT_FIELDS,
                     log=print) -> dict:
    """File records into `l1/release=<release>/source=<source>/`. Returns
    `{canonical_url: path}` for everything written."""
    if not origin:
        raise L1PersistError(
            "write_l1_records needs an `origin`: a reader must be able to tell "
            "a curated manifest body from a full article extraction without "
            "archaeology.")
    # ADR-106 / task 586. Recording the origin was necessary and not
    # sufficient: nothing downstream READ it. `persisted_inputs` asks whether a
    # document exists for the page, never whether the document IS the page, so
    # filing an authored body here satisfied the ADR-105 guard with evidence
    # the lane manufactured -- and a rebuild replays L1 as content, which is
    # how 178 rows of manifest prose acquired `build_id aw_b1`, a
    # `content_digest` and a `rule_epochs` stamp. An authored body may be kept
    # as a note; it may not be filed as a page's L1 document.
    from cytir.corpus import authored_text_guard as ATG  # local: keeps this module import-light
    if ATG.is_authored_origin(origin, ATG.load_policy()):
        raise L1PersistError(
            f"refusing to file L1 records under the authored origin {origin!r} "
            f"(data/authored-text-origins.v1.json). L1 is replayed as page "
            f"content by every rebuild; authored prose is not the page. Fetch "
            f"the page and file the fetch, or file nothing.")
    if not release or "/" in release or not source or "/" in source:
        raise L1PersistError(
            f"release/source must be single path segments, got "
            f"{release!r}/{source!r}")
    base = pathlib.Path(root or aw_paths.CORPUS_ROOT)
    if not base.is_dir():
        raise L1PersistError(
            f"corpus root {base} is not a directory -- set CYTIR_CORPUS_ROOT. "
            "Refusing to invent a store.")
    out_dir = base / "l1" / f"release={release}" / f"source={source}"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict = {}
    files: list = []
    shard = 0
    n = 0
    t0 = time.time()
    buf: list = []

    def flush():
        nonlocal shard, buf
        if not buf:
            return
        name = f"l1-markdown-{shard:05d}.jsonl.gz"
        final = out_dir / name
        tmp = out_dir / (name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            for rec in buf:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, final)          # a half-written file is never a store
        files.append({"file": name, "rows": len(buf),
                      "bytes": final.stat().st_size,
                      "sha256": _sha256_file(str(final))})
        for rec in buf:
            written[rec["canonical_url"]] = str(final)
        shard += 1
        buf = []

    for raw in records:
        rec = normalize(raw, origin=origin, url_fields=url_fields,
                        text_fields=text_fields)
        buf.append(rec)
        n += 1
        if len(buf) >= SHARD_ROWS:
            flush()
    flush()

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "release": release, "source": source,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": n, "distinct_pages": len(written),
        "files": files, "note": note, "origin": origin,
        "git_sha": _git_sha(),
        "record_schema": SCHEMA,
    }
    (out_dir / "RECEIPT.json").write_text(
        json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    log(f"[persist-l1] {n:,} record(s), {len(written):,} page(s) -> {out_dir} "
        f"({time.time()-t0:.1f}s)")
    return written


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: str) -> Iterable[dict]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-jsonl", required=True,
                    help="news_documents.jsonl or any url+markdown jsonl(.gz)")
    ap.add_argument("--release", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--corpus-root", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--origin", required=True,
                    help="what this text IS, e.g. "
                         "'news-stage:news_documents.jsonl markdown extraction'")
    ap.add_argument("--url-field", action="append", default=None,
                    help="field carrying the page address, repeatable "
                         f"(default: {' '.join(URL_FIELDS)})")
    ap.add_argument("--text-field", action="append", default=None,
                    help="field carrying the page text, repeatable "
                         f"(default: {' '.join(TEXT_FIELDS)})")
    args = ap.parse_args(argv)
    try:
        written = write_l1_records(read_jsonl(args.from_jsonl),
                                   release=args.release, source=args.source,
                                   origin=args.origin,
                                   root=args.corpus_root, note=args.note,
                                   url_fields=tuple(args.url_field or URL_FIELDS),
                                   text_fields=tuple(args.text_field or TEXT_FIELDS))
    except L1PersistError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"pages": len(written)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
