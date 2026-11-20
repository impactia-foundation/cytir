"""`cyt index loda-resurface` -- the producer of the E6 loda dense store (task 886).

the deleted `aw.index.rebuild_e10` (task 1312; inputs `loda: {"mode": "twin-restaged"}`) took each loda row's dense vector
from a pre-built sqlite store and refused on a miss (`DenseSource`, `write_row`). Nothing produced
that store. This verb does:

  1. ids   -- scroll the TWIN (never prod) for the tier-A loda rows: payload `provenance == loda` (name or enum code)
              and `tier != B` (deleted `rebuild_e10.route`, task 1312: tier B is the removed old generation).
  2. text  -- the row text is NOT on the twin (the payload keeps a ~250-char snippet only), so it
              comes from the staged loda texts shards (`--texts`, `shard-*.texts.jsonl.gz`,
              `{point_id, text, title}`, the contract `cytir.sources.families.loda` writes). The join
              is `twin id == shard point_id`; a twin id with no text is a REFUSAL, not a skip.
              The dense surface is `cytir.core.common.compose_index_text` (the one composer).
  3. embed -- `cytir.embed.stage.embed_stage` under the committed model pin (GPU lease, subprocess).
  4. store -- MRL-truncate + renormalise to the twin's served dense width (`mrl_trunc`), write
              `dense_db` (`dense(id TEXT PRIMARY KEY, vec BLOB)`, little-endian float32),
              `restage_ids.txt` (== the store's ids, sorted) and `manifest.json` (sha256 of both,
              counts, model pin, twin collection + `points_count`, the ids digest).

Idempotent and resumable: ids already in the store are skipped; the store commits per batch.
Run on the GPU box after the restage pack lands, before T0 (the task doc has the recipe).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import pathlib
import sqlite3
import sys
import time
import uuid
from typing import Callable, Iterable

SCHEMA = "aw.loda-resurface.v1"
LODA = "loda"
OLD_GENERATION_TIER = "B"
DB_NAME = "dense.sqlite"
IDS_NAME = "restage_ids.txt"
MANIFEST_NAME = "manifest.json"


class ResurfaceRefused(RuntimeError):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def ids_digest(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def norm_id(v) -> str:
    """The one id spelling both sides of the join use (twin `str(p.id)` and shard `point_id`):
    the canonical lower-case hyphenated UUID, so case, braces, hyphens and whitespace do not matter."""
    try:
        return str(uuid.UUID(str(v).strip()))
    except (ValueError, AttributeError, TypeError):
        raise ResurfaceRefused(f"not a point id: {v!r}")


def is_tier_a_loda(row) -> bool:
    """`row` is a CANONICAL row (`cytir.index.row_decode`): `provenance` is the name, never a code."""
    return row.get("provenance") == LODA and row.get("tier") != OLD_GENERATION_TIER  # canonical row


def loda_provenance_values() -> list:
    """The stored spellings of provenance `loda`: the name (raw collections) and its enum code
    (coded collections, `data/payload-enum-tables.v1.json`); the server filter matches either."""
    from cytir.index import payload_profile as PP
    vals: list = [LODA]
    code = PP.load_tables().get("provenance", {}).get(json.dumps(LODA))  # canonical row
    return vals + ([code] if code is not None else [])


# ----------------------------------------------------------------------------------- 1. the ids
def twin_loda_ids(client, collection: str, *, batch: int = 2000) -> list[str]:
    """Tier-A loda ids on the twin. Server-side filter, then re-checked on the payload."""
    from qdrant_client import models as qm
    from cytir.index import row_decode as RD
    decode = RD.reader(collection)       # the twin's rows as canonical rows, any profile
    flt = qm.Filter(should=[qm.FieldCondition(key="provenance", match=qm.MatchValue(value=v))
                            for v in loda_provenance_values()],
                    must_not=[qm.FieldCondition(key="tier",
                                                match=qm.MatchValue(value=OLD_GENERATION_TIER))])
    out, offset = [], None
    while True:
        points, offset = client.scroll(collection, scroll_filter=flt, limit=batch, offset=offset,
                                       with_payload=["provenance", "tier"], with_vectors=False)
        out += [norm_id(p.id) for p in points if is_tier_a_loda(decode(p.payload or {}))]
        if offset is None or not points:
            break
    if len(set(out)) != len(out):
        raise ResurfaceRefused("the twin scroll returned a duplicate id")
    return sorted(out)


# ----------------------------------------------------------------------------------- 2. the text
def iter_text_rows(texts_dirs):
    for texts_dir in texts_dirs:
        paths = sorted(glob.glob(os.path.join(texts_dir, "**", "*.texts.jsonl.gz"), recursive=True))
        if not paths:
            raise ResurfaceRefused(f"no *.texts.jsonl.gz under {texts_dir}")
        for p in paths:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)


def dense_surface(row: dict) -> str:
    """The stage `text` as is: for loda `compose_index_text` (no site_name) returns it unchanged."""
    from cytir.core import common as AC
    return AC.compose_index_text({"text": row.get("text") or "", "title": row.get("title")})


def write_chunks(texts_dirs, want: set, out_dir: str, *, rows_per_shard: int = 20000, log=print) -> dict:
    """The texts of `want` (ids still to embed) as embed shards `{point_id, text}`.
    A duplicate id across the shards is counted and logged; it is refused if its text differs."""
    if isinstance(texts_dirs, str):
        texts_dirs = [texts_dirs]
    os.makedirs(out_dir, exist_ok=True)
    found: dict = {}                        # norm id -> dense surface text
    dups, n_shard, buf = 0, 0, []

    def flush():
        nonlocal n_shard, buf
        if not buf:
            return
        path = os.path.join(out_dir, f"shard-{n_shard:05d}.texts.jsonl.gz")
        with gzip.open(path + ".part", "wt", encoding="utf-8") as fh:
            for r in buf:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(path + ".part", path)
        n_shard, buf = n_shard + 1, []

    for row in iter_text_rows(texts_dirs):
        pid = norm_id(row.get("point_id"))
        if pid not in want:
            continue
        text = dense_surface(row)
        if pid in found:
            if found[pid] != text:
                raise ResurfaceRefused(f"REFUSED: id {pid} has two different texts across --texts")
            dups += 1
            continue
        found[pid] = text
        buf.append({"point_id": pid, "text": text, "url": row.get("url"), "title": row.get("title")})
        if len(buf) >= rows_per_shard:
            flush()
    flush()
    if dups:
        log(f"duplicate text rows across --texts shards: {dups} (identical text; first kept)")
    missing = want - set(found)
    if missing:
        raise ResurfaceRefused(
            f"REFUSED: {len(missing)} twin tier-A loda ids have no text in {list(texts_dirs)} "
            f"(sample {sorted(missing)[:3]}): the stage does not cover the twin")
    return {"shards": n_shard, "rows": len(found), "duplicates": dups}


# ----------------------------------------------------------------------------------- 4. the store
def open_store(path: str) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS dense (id TEXT PRIMARY KEY, vec BLOB NOT NULL)")
    db.commit()
    return db


def store_ids(db) -> set:
    return {r[0] for r in db.execute("SELECT id FROM dense")}


def put_vectors(db, ids: list[str], matrix, *, width: int) -> None:
    import numpy as np
    m = np.ascontiguousarray(np.asarray(matrix, dtype="<f4"))
    if m.shape != (len(ids), width):
        raise ResurfaceRefused(f"vector matrix {m.shape} != ({len(ids)}, {width})")
    if not np.isfinite(m).all():
        raise ResurfaceRefused("non-finite values in the embedded vectors")
    db.executemany("INSERT OR REPLACE INTO dense (id, vec) VALUES (?, ?)",
                   ((i, m[k].tobytes()) for k, i in enumerate(ids)))
    db.commit()


def twin_points_count(client, collection: str) -> int:
    return int(client.get_collection(collection).points_count)


def served_width(url: str, collection: str) -> int:
    from cytir.index import promote_queue as PQ
    size = ((PQ.fetch_collection_vector_config(url, collection) or {}).get("dense") or {}).get("size")
    if not size:
        raise ResurfaceRefused(f"{collection} at {url}: no dense vector width")
    return int(size)


def default_embed(chunks_dir: str, vectors_dir: str) -> None:
    from cytir.embed.stage import embed_stage
    rc = embed_stage(chunks_dir, vectors_dir)
    if rc != 0:
        raise ResurfaceRefused(f"embed_stage rc={rc}")


def read_embedded(vectors_dir: str, width: int):
    from cytir.index import promote_sealed_pack as PSP
    return PSP.read_vectors(vectors_dir, width)


def check_resume(out: pathlib.Path, db, width: int) -> None:
    """An existing store is reused only at the same model pin and dense width (manifest + stored blobs)."""
    from cytir.core.model_pin import EMBED_MODEL_ID, embed_model_revision
    man_path = out / MANIFEST_NAME
    if man_path.exists():
        man = json.loads(man_path.read_text())
        pin = man.get("model") or {}
        if (pin.get("id"), pin.get("revision"), man.get("dense_width")) != (
                EMBED_MODEL_ID, embed_model_revision(), width):
            raise ResurfaceRefused(
                f"REFUSED: {out} was written with model {pin.get('id')}@{pin.get('revision')} width "
                f"{man.get('dense_width')}; this run is {EMBED_MODEL_ID}@{embed_model_revision()} width "
                f"{width}: use a fresh --out")
    widths = {r[0] for r in db.execute("SELECT DISTINCT length(vec) FROM dense")}
    if widths - {4 * width}:
        raise ResurfaceRefused(f"REFUSED: {out} holds vectors of {sorted(w // 4 for w in widths)} dims; "
                               f"this run is {width}: use a fresh --out")


def run(*, ids: list[str], texts_dirs: list, out_dir: str, width: int, twin_url: str,
        twin_collection: str, points_count: int | None,
        embed: Callable[[str, str], None] = default_embed,
        read_vectors: Callable = read_embedded, log=print) -> dict:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    db = open_store(str(out / DB_NAME))
    check_resume(out, db, width)
    have = store_ids(db)
    stale = have - set(ids)
    if stale:
        raise ResurfaceRefused(f"REFUSED: the store holds {len(stale)} ids that are not twin tier-A "
                               f"loda ids (sample {sorted(stale)[:3]}): use a fresh --out")
    todo = sorted(set(ids) - have)
    log(f"twin tier-A loda ids {len(ids)}; in store {len(have)}; to embed {len(todo)}")
    if todo:
        chunks, vecs = str(work / "chunks"), str(work / "vectors")
        for d in (chunks, vecs):       # a half-written earlier attempt is rebuilt, never reused
            if os.path.isdir(d):
                import shutil
                shutil.rmtree(d)
        dup = write_chunks(texts_dirs, set(todo), chunks, log=log)
        log(f"text shards {dup['shards']}, rows {dup['rows']}, duplicates {dup['duplicates']}")
        embed(chunks, vecs)
        matrix, got = read_vectors(vecs, width)
        got = [norm_id(g) for g in got]
        if set(got) != set(todo) or len(got) != len(todo):
            raise ResurfaceRefused(f"embedded ids differ from the requested ones "
                                   f"({len(set(got) ^ set(todo))} differ)")
        put_vectors(db, list(got), matrix, width=width)
    final = store_ids(db)
    db.close()
    if final != set(ids):
        raise ResurfaceRefused("the store does not equal the twin ids after the run")
    ids_path = out / IDS_NAME
    ids_path.write_text("".join(i + "\n" for i in sorted(final)), encoding="utf-8")
    from cytir.core.model_pin import EMBED_MODEL_ID, embed_model_revision
    manifest = {
        "schema": SCHEMA, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": len(final), "embedded_this_run": len(todo), "skipped_resumed": len(have),
        "dense_width": width, "dtype": "float32-le", "ids_digest": ids_digest(final),
        "model": {"id": EMBED_MODEL_ID, "revision": embed_model_revision()},
        "surface": "cytir.core.common.compose_index_text over the staged loda text",
        "twin": {"url": twin_url, "collection": twin_collection, "points_count": points_count,
                 "filter": f"provenance == {LODA} and tier != {OLD_GENERATION_TIER}"},
        "files": {DB_NAME: sha256_file(out / DB_NAME), IDS_NAME: sha256_file(ids_path)},
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def verify(out_dir: str) -> dict:
    """Re-check a written store: file sha256s and store ids == restage_ids."""
    out = pathlib.Path(out_dir)
    man = json.loads((out / MANIFEST_NAME).read_text())
    for name, want in man["files"].items():
        if sha256_file(out / name) != want:
            raise ResurfaceRefused(f"{name}: sha256 does not match the manifest")
    db = sqlite3.connect(f"file:{out / DB_NAME}?mode=ro", uri=True)
    ids = store_ids(db)
    listed = {x.strip() for x in open(out / IDS_NAME) if x.strip()}
    if ids != listed or len(ids) != man["rows"] or ids_digest(ids) != man["ids_digest"]:
        raise ResurfaceRefused("store ids, restage_ids and the manifest disagree")
    return man


def build_parser(prog: str = "cyt index loda-resurface"):
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--twin", default=None, help="twin qdrant URL (default: the operator twin_url)")
    ap.add_argument("--collection", default=None, help="twin collection (default: twin_collection)")
    ap.add_argument("--texts", action="append", default=[],
                    help="dir of the staged loda *.texts.jsonl.gz shards (repeatable; each is globbed)")
    ap.add_argument("--out", required=True, help="output dir (dense.sqlite, restage_ids.txt, manifest.json)")
    ap.add_argument("--verify", action="store_true", help="only re-check --out and exit")
    return ap


def main(argv=None, prog: str = "cyt index loda-resurface") -> int:
    args = build_parser(prog).parse_args(argv)
    try:
        if args.verify:
            print(json.dumps(verify(args.out), indent=1))
            return 0
        if not args.texts:
            raise ResurfaceRefused("--texts is required (repeat it for each texts directory)")
        from cytir.core import twin as TWIN
        from cytir.core.qdrant import qdrant_client
        url = (args.twin or TWIN.url()).rstrip("/")
        coll = args.collection or TWIN.collection()
        from cytir.index import prod_guard
        if prod_guard.is_prod(url, coll):
            raise ResurfaceRefused(f"{url} {coll} is PROD: this verb reads the twin only")
        client = qdrant_client(url, timeout=300)
        ids = twin_loda_ids(client, coll)
        man = run(ids=ids, texts_dirs=list(args.texts), out_dir=args.out, width=served_width(url, coll),
                  twin_url=url, twin_collection=coll, points_count=twin_points_count(client, coll))
        verify(args.out)
        print(json.dumps({k: man[k] for k in ("rows", "embedded_this_run", "ids_digest")}))
        return 0
    except ResurfaceRefused as exc:
        print(f"[REFUSED] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
