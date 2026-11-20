#!/usr/bin/env python3
"""Canonical corpus store — partitioned zstd Parquet, one row per passage.

WHY (Tim, 2026-07-04): the 30M base corpus was lost because extracted text
only ever lived in a build box + a deleted qdrant collection. From now on the
CORPUS IS PRIMARY and every qdrant collection is a DERIVED artifact that must
be rebuildable from this store.

Layout
------
    <dest>/wave=<name>/part-0000.parquet     (zstd, files <= ~512 MB)
    <dest>/wave=<name>/_meta.json           (source, rows, sha256, notes, ...)

Canonical dest (the build box):  <corpus-root>/
Bender staging dest:        ~/aw-data/corpus-staging/   (rsync to the build box:
    rsync -av ~/aw-data/corpus-staging/wave=<name> \
        <host>:<corpus-root>/ )

A "wave" is one ingest tranche (e.g. wave=legifrance, wave=gesetze_v2,
wave=discovery_w0). Waves are append-only history: superseded tranches keep
their data and get `"superseded": true` in their _meta.json.

Schema (all columns nullable — never fabricate missing data):
    point_id, url, anchor, domain, host, title, text, snippet, lang, country,
    tier, trust_tier, source_type, provenance, disc_class, content_digest,
    last_seen, fetched_at, crawl_id, extraction_version,
    consent_search_ok, consent_train_ok, consent_checked_at,
    wave, ingested_at, extra
`extra` is a JSON object string of any source fields beyond the schema
(e.g. hn_score, importance, story_url, se_score, jurisdiction) so nothing a
tranche carried is dropped. Old tranches predate the consent gate → their
consent_* columns are null (ungated), NOT false.

Library:  write_wave(rows_iter, wave_name, dest, ...) / read_wave / list_waves
CLI:
  python corpus_store.py backfill --src FILE.jsonl --wave NAME [--dest DIR]
      [--note TEXT] [--superseded] [--extraction-version V] [--force]
  python corpus_store.py backfill-qdrant --collection aw_passages --wave NAME
      [--qdrant-url http://localhost:6333] [--dest DIR] [--note TEXT]
  python corpus_store.py list [--dest DIR]
  python corpus_store.py verify --wave NAME --src FILE.jsonl [--dest DIR]

Requires pyarrow (the build box's main venv and the orchestrator box ~/aw-data/venv have it).
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import datetime as _dt
import glob
import hashlib
import json
import os
import pathlib
import sys
import urllib.parse
import urllib.request

from cytir.core.ids import record_point_id
from cytir.core.paths import operator_settings
from cytir.index import prod_guard as PG  # noqa: E402 -- no prod scroll, no prod replay (2026-10-07)



def default_dest() -> str:
    """The ADR-026 corpus store: operator setting `hdd_root` + `aw-corpus` (refused by name
    when unconfigured)."""
    return os.path.join(operator_settings().hdd_root, "aw-corpus")


def __getattr__(name):
    # `DEFAULT_DEST` stays readable but resolves on first ACCESS, never at import.
    if name == "DEFAULT_DEST":
        return default_dest()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

DEFAULT_EXTRACTION_VERSION = "unknown-2026-06"
MAX_FILE_BYTES = 512 * 1024 * 1024
BATCH_ROWS = 20_000

_STR_COLS = [
    "point_id", "url", "anchor", "domain", "host", "title", "text", "snippet",
    "lang", "country", "tier", "source_type", "provenance", "disc_class",
    "content_digest", "last_seen", "fetched_at", "crawl_id",
    "extraction_version", "consent_checked_at", "wave", "ingested_at",
]
_KNOWN = set(_STR_COLS) | {
    "trust_tier", "consent_search_ok", "consent_train_ok", "extra",
    # source-side names that map onto schema columns:
    "search_ok", "train_ok",
}


def _schema():
    import pyarrow as pa
    fields = []
    for c in _STR_COLS:
        fields.append((c, pa.string()))
    fields.insert(11, ("trust_tier", pa.float64()))          # after tier
    fields.append(("consent_search_ok", pa.bool_()))
    fields.append(("consent_train_ok", pa.bool_()))
    fields.append(("extra", pa.string()))
    return pa.schema(fields)


def _now_iso():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_point_id(r: dict) -> str | None:
    """Same semantics as index_bakeoff.point_id, but never invents a random id:
    use r['point_id'] (or url) — UUID as-is, else uuid5(NAMESPACE_URL, value)."""
    if not (r.get("point_id") or r.get("url")):
        return None
    return record_point_id(r)        # cytir.core.ids owns the uuid5 mint (test_core_ratchet)


def _s(v):
    """Nullable string: keep None as None, stringify everything else."""
    if v is None:
        return None
    v = str(v)
    return v if v != "" else None


def normalize_row(r: dict, wave: str, ingested_at: str,
                  extraction_version: str = DEFAULT_EXTRACTION_VERSION) -> dict:
    url = r.get("url")
    host = None
    if url:
        try:
            host = (urllib.parse.urlsplit(url).hostname or "").lower() or None
        except ValueError:
            host = None
    tt = r.get("trust_tier")
    try:
        tt = float(tt) if tt is not None else None
    except (TypeError, ValueError):
        tt = None
    extra = {k: v for k, v in r.items() if k not in _KNOWN and v is not None}
    return {
        "point_id": stable_point_id(r),
        "url": _s(url),
        "anchor": _s(r.get("anchor")),
        "domain": _s(r.get("domain")),
        "host": host,
        "title": _s(r.get("title")),
        "text": _s(r.get("text")),
        "snippet": _s(r.get("snippet")),
        "lang": _s(r.get("lang")),
        "country": _s(r.get("country")),
        "tier": _s(r.get("tier")),
        "trust_tier": tt,
        "source_type": _s(r.get("source_type")),
        "provenance": _s(r.get("provenance")),
        "disc_class": _s(r.get("disc_class")),
        "content_digest": _s(r.get("content_digest")),
        "last_seen": _s(r.get("last_seen")),
        "fetched_at": _s(r.get("fetched_at")),
        "crawl_id": _s(r.get("crawl_id")),
        "extraction_version": extraction_version,
        "consent_search_ok": (bool(r["search_ok"])
                              if r.get("search_ok") is not None else None),
        "consent_train_ok": (bool(r["train_ok"])
                             if r.get("train_ok") is not None else None),
        "consent_checked_at": _s(r.get("consent_checked_at")),
        "wave": wave,
        "ingested_at": ingested_at,
        "extra": json.dumps(extra, ensure_ascii=False) if extra else None,
    }


def wave_dir(dest: str | None, wave: str) -> pathlib.Path:
    return pathlib.Path(os.path.expanduser(dest or default_dest())) / f"wave={wave}"


def write_wave(rows_iter, wave_name: str, dest: str | None = None, *,
               meta: dict | None = None,
               extraction_version: str = DEFAULT_EXTRACTION_VERSION,
               force: bool = False,
               batch_rows: int = BATCH_ROWS,
               max_file_bytes: int = MAX_FILE_BYTES) -> dict:
    """Stream raw source rows (dicts) into <dest>/wave=<name>/part-*.parquet.

    Normalizes each row to the canonical schema, batches, writes zstd parquet,
    rolls files at ~max_file_bytes, and writes _meta.json last (its presence
    marks the wave complete). Returns the meta dict.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    wdir = wave_dir(dest, wave_name)
    if wdir.exists() and any(wdir.iterdir()):
        if not force:
            raise FileExistsError(
                f"{wdir} exists and is non-empty — pass force=True/--force to "
                f"rewrite this wave")
        for p in list(wdir.glob("part-*.parquet")) + [wdir / "_meta.json"]:
            if p.exists():
                p.unlink()
    wdir.mkdir(parents=True, exist_ok=True)

    schema = _schema()
    ingested_at = _now_iso()
    writer = None
    part = 0
    cur_path = None
    n_rows = 0
    files = []

    def _open_next():
        nonlocal writer, part, cur_path
        cur_path = wdir / f"part-{part:04d}.parquet"
        writer = pq.ParquetWriter(str(cur_path), schema, compression="zstd")
        files.append(cur_path.name)
        part += 1

    def _flush(batch):
        nonlocal writer
        if not batch:
            return
        if writer is None:
            _open_next()
        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        # roll to a new part when the current file grows past the cap
        if cur_path.stat().st_size >= max_file_bytes:
            writer.close()
            writer = None

    batch = []
    for r in rows_iter:
        batch.append(normalize_row(r, wave_name, ingested_at,
                                   extraction_version))
        n_rows += 1
        if len(batch) >= batch_rows:
            _flush(batch)
            batch = []
    _flush(batch)
    if writer is not None:
        writer.close()
    if n_rows == 0:
        # keep an explicit empty wave marker rather than a half-made dir
        _open_next()
        writer.close()

    out_meta = {
        "wave": wave_name,
        "rows": n_rows,
        "files": files,
        "bytes": sum((wdir / f).stat().st_size for f in files),
        "extraction_version": extraction_version,
        "ingested_at": ingested_at,
        "written_by": "aw/corpus/store.py",
    }
    if meta:
        out_meta.update(meta)
    (wdir / "_meta.json").write_text(
        json.dumps(out_meta, ensure_ascii=False, indent=2) + "\n")
    return out_meta


def read_wave(wave_name: str, dest: str | None = None, columns=None):
    """Yield rows (dicts) of a wave, streaming (never loads a wave whole)."""
    import pyarrow.parquet as pq
    wdir = wave_dir(dest, wave_name)
    for path in sorted(wdir.glob("part-*.parquet")):
        pf = pq.ParquetFile(str(path))
        for rb in pf.iter_batches(batch_size=4096, columns=columns):
            yield from rb.to_pylist()


def wave_row_count(wave_name: str, dest: str | None = None) -> int:
    import pyarrow.parquet as pq
    return sum(pq.ParquetFile(str(p)).metadata.num_rows
               for p in sorted(wave_dir(dest, wave_name).glob("part-*.parquet")))


def list_waves(dest: str | None = None) -> list[dict]:
    out = []
    root = pathlib.Path(os.path.expanduser(dest or default_dest()))
    for mp in sorted(glob.glob(str(root / "wave=*" / "_meta.json"))):
        try:
            out.append(json.loads(pathlib.Path(mp).read_text()))
        except (OSError, json.JSONDecodeError) as e:  # surface, don't hide
            out.append({"wave": pathlib.Path(mp).parent.name,
                        "error": str(e)})
    return out


# ---------------------------------------------------------------- CLI helpers

def _iter_jsonl(path: str):
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(os.path.expanduser(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(path: str) -> int:
    n = 0
    with open(os.path.expanduser(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            n += chunk.count(b"\n")
        # count a final unterminated line
        f.seek(0, 2)
        if f.tell():
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                n += 1
    return n


def _qdrant_scroll(qdrant_url: str, collection: str, batch: int = 1000):
    """Payload-only scroll over a collection (stdlib only, no qdrant_client)."""
    url = f"{qdrant_url.rstrip('/')}/collections/{collection}/points/scroll"
    PG.check_url(url, "POST")               # no prod scroll, no prod replay (2026-10-07)
    offset = None
    while True:
        body = {"limit": batch, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with IH.urlopen(req, timeout=120) as resp:
            res = json.load(resp)["result"]
        for p in res["points"]:
            row = dict(p.get("payload") or {})
            row.setdefault("point_id", str(p["id"]))
            yield row
        offset = res.get("next_page_offset")
        if offset is None:
            return


def cmd_backfill(args):
    src = os.path.expanduser(args.src)
    meta = {
        "source": src,
        "source_bytes": os.path.getsize(src),
        "source_sha256": _sha256(src),
        "source_lines": _count_lines(src),
    }
    if args.note:
        meta["notes"] = args.note
    if args.superseded:
        meta["superseded"] = True
    out = write_wave(_iter_jsonl(src), args.wave, args.dest, meta=meta,
                     extraction_version=args.extraction_version,
                     force=args.force)
    ok = out["rows"] == meta["source_lines"]
    print(json.dumps(out, indent=2))
    print(f"rows={out['rows']} source_lines={meta['source_lines']} "
          f"match={'OK' if ok else 'MISMATCH'}")
    if not ok:
        sys.exit(1)


def cmd_backfill_qdrant(args):
    meta = {
        "source": f"qdrant:{args.qdrant_url}/collections/{args.collection}",
        "source_kind": "qdrant-payload-scroll (no vectors; text is null — "
                       "the collection stores snippet only)",
    }
    if args.note:
        meta["notes"] = args.note
    out = write_wave(_qdrant_scroll(args.qdrant_url, args.collection),
                     args.wave, args.dest, meta=meta,
                     extraction_version=args.extraction_version,
                     force=args.force)
    print(json.dumps(out, indent=2))


def cmd_list(args):
    waves = list_waves(args.dest)
    total_rows = total_bytes = 0
    for m in waves:
        rows, byt = m.get("rows", 0), m.get("bytes", 0)
        total_rows += rows
        total_bytes += byt
        flags = " SUPERSEDED" if m.get("superseded") else ""
        print(f"{m['wave']:32s} {rows:>12,d} rows  "
              f"{byt / 1e9:8.2f} GB{flags}")
    print(f"{'TOTAL':32s} {total_rows:>12,d} rows  {total_bytes / 1e9:8.2f} GB "
          f"({len(waves)} waves)")


def cmd_verify(args):
    """Count check vs the source + spot-check 3 rows exist in the source."""
    n_parquet = wave_row_count(args.wave, args.dest)
    n_src = _count_lines(args.src)
    print(f"wave={args.wave}: parquet={n_parquet:,} source_lines={n_src:,} "
          f"{'OK' if n_parquet == n_src else 'MISMATCH'}")
    if n_parquet != n_src:
        sys.exit(1)
    # spot rows: first / middle / last by position
    want = sorted({0, n_parquet // 2, n_parquet - 1})
    picked = {}
    for i, row in enumerate(read_wave(args.wave, args.dest,
                                      columns=["point_id", "url", "title"])):
        if i in want:
            picked[i] = row
        if i >= want[-1]:
            break
    needles = {i: (r.get("url") or r.get("point_id")) for i, r in picked.items()}
    found = {i: False for i in needles}
    with open(os.path.expanduser(args.src), encoding="utf-8") as f:
        for line in f:
            for i, needle in needles.items():
                if not found[i] and needle and needle in line:
                    found[i] = True
            if all(found.values()):
                break
    for i in sorted(picked):
        r = picked[i]
        print(f"  row {i}: {'OK ' if found[i] else 'MISSING '} "
              f"{(r.get('url') or r.get('point_id') or '')[:90]}")
    if not all(found.values()):
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backfill", help="jsonl file -> one parquet wave")
    b.add_argument("--src", required=True)
    b.add_argument("--wave", required=True)
    b.add_argument("--dest", default=None)
    b.add_argument("--note", default="")
    b.add_argument("--superseded", action="store_true")
    b.add_argument("--extraction-version", default=DEFAULT_EXTRACTION_VERSION)
    b.add_argument("--force", action="store_true")
    b.set_defaults(func=cmd_backfill)

    q = sub.add_parser("backfill-qdrant",
                       help="qdrant payload scroll -> one parquet wave")
    q.add_argument("--collection", required=True)
    q.add_argument("--qdrant-url", default="http://localhost:6333")
    q.add_argument("--wave", required=True)
    q.add_argument("--dest", default=None)
    q.add_argument("--note", default="")
    q.add_argument("--extraction-version", default=DEFAULT_EXTRACTION_VERSION)
    q.add_argument("--force", action="store_true")
    q.set_defaults(func=cmd_backfill_qdrant)

    ls = sub.add_parser("list", help="wave inventory (rows, GB)")
    ls.add_argument("--dest", default=None)
    ls.set_defaults(func=cmd_list)

    v = sub.add_parser("verify", help="row-count + spot-row check vs source")
    v.add_argument("--wave", required=True)
    v.add_argument("--src", required=True)
    v.add_argument("--dest", default=None)
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
