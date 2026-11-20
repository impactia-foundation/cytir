"""Compose a RESTORE pack: rows that were lost or overwritten, put back from a pre-incident copy (task 826).

A restore pack is an ordinary promotion-queue `content_pack` (schema `aw.promotion-queue-pack.v1`)
with points and, optionally, `superseded` tombstones (task 844). The hourly drain applies it twin
first, then prod, through the same card-overwrite guard as every other pack. Nothing here writes to
a collection.

    python -m cytir.ops.restore_pack export  --qdrant-url U --collection C --ids-file F --out rows.jsonl.gz
    python -m cytir.ops.restore_pack tombstones --qdrant-url U --collection C --ids-file F --rows rows.jsonl.gz \\
                                          --out tombstones.jsonl --export-out stored.jsonl.gz
    python -m cytir.ops.restore_pack compose --rows rows.jsonl.gz --queue-dir Q --pack-id P \\
                                          --collection C --family F --note "..." --task N [--tombstones T]

`export` is a by-id retrieve (payload + both vectors) from a source that holds the ORIGINAL rows
(a backup restored read-only, a pre-incident copy). It refuses an id the source does not hold: a
restore pack never guesses. The rows keep their ORIGINAL vectors and payloads byte for byte; the
pack re-embeds nothing.

`compose` checks the rows (unique ids, one dense list, one sparse leg, an `url`) and writes
`<queue-dir>/<pack-id>/{manifest.json, <name>-points.<sha>.jsonl.gz, READY}`. It refuses an existing
pack directory. When `--collection` declares `"id_rule": "I1"` (aw_e10 and its twin, after the flip
commit), each row is re-keyed to the I1 id of its own payload (`cytir.index.id_rule.key_points_for`;
a card with an identity becomes `canon#card:<identity>`) and the manifest carries the old -> new
map (`id_map`): the export holds the ORIGINAL ids, the collection holds I1 ids (task 864). The drain, not this module, judges the rows against the target (vector shape, card
guard, consent): run the drain without `--apply` and with `AW_REPLAY=1` to see its verdict first.
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import gzip
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

from cytir.index import supersede as US

MANIFEST_SCHEMA = "aw.promotion-queue-pack.v1"
DENSE = "dense"
SPARSE = "bm25"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _post(url: str, body: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with IH.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def export_rows(qdrant_url: str, collection: str, ids: list[str], *, batch: int = 64) -> list[dict]:
    """One by-id retrieve per `batch` ids (payload and vectors). Refuses an id the source lacks."""
    if not ids:
        raise SystemExit("REFUSED: no ids")
    if len(set(ids)) != len(ids):
        raise SystemExit("REFUSED: duplicate ids in the id list")
    got: dict[str, dict] = {}
    for i in range(0, len(ids), batch):
        res = _post(f"{qdrant_url.rstrip('/')}/collections/{collection}/points",
                    {"ids": ids[i:i + batch], "with_payload": True, "with_vector": True})["result"]
        for p in res:
            got[str(p["id"])] = {"id": str(p["id"]), "payload": p["payload"], "vector": p["vector"]}
    missing = [i for i in ids if i not in got]
    if missing:
        raise SystemExit(f"REFUSED: {len(missing)} id(s) are not in {collection!r} at {qdrant_url} "
                         f"(first {missing[:3]}); a restore pack never guesses")
    return [got[i] for i in ids]


def check_rows(rows: list[dict]) -> None:
    if not rows:
        raise SystemExit("REFUSED: 0 rows")
    seen: set[str] = set()
    for r in rows:
        pid = str(r.get("id") or "")
        if not pid or pid in seen:
            raise SystemExit(f"REFUSED: missing or duplicate id {pid!r}")
        seen.add(pid)
        vec = r.get("vector")
        if not isinstance(vec, dict) or not isinstance(vec.get(DENSE), list) or not vec[DENSE]:
            raise SystemExit(f"REFUSED: {pid} has no dense vector")
        sp = vec.get(SPARSE)
        if not isinstance(sp, dict) or not sp.get("indices") or len(sp["indices"]) != len(sp.get("values", [])):
            raise SystemExit(f"REFUSED: {pid} has no usable {SPARSE} sparse vector")
        if not (r.get("payload") or {}).get("url"):
            raise SystemExit(f"REFUSED: {pid} has no payload url")


SUPERSEDED = "superseded"


def build_tombstones(stored: dict[str, dict], rows: list[dict], *, detail: str) -> list[dict]:
    """`superseded` tombstones for the stored rows `stored` ({id: payload}), each REPLACED by a card
    of the pack `rows` by the one supersede rule (`supersede.card_replaces`: both cards, same
    canonical url, same raw type, identity overlap by precedence). Refuses a stored row no pack card
    replaces: the drain would refuse the whole pack on it."""
    out = []
    for pid, old in stored.items():
        repl = [r for r in rows if US.card_replaces(r["payload"], old)]
        if not repl:
            raise SystemExit(f"REFUSED: no card of the pack replaces the stored row {pid} ({old.get('url')!r}); "
                             "the identity rule of supersede.card_replaces does not admit it")
        out.append({"id": pid, "url": old["url"], "reason": SUPERSEDED, "detail": detail,
                    "replaced_by": [r["id"] for r in repl]})
    return out


def check_tombstones(tombstones: list[dict], rows: list[dict]) -> None:
    ids = [t.get("id") for t in tombstones]
    if len(set(ids)) != len(ids) or not all(ids):
        raise SystemExit("REFUSED: missing or duplicate tombstone id")
    if set(ids) & {r["id"] for r in rows}:
        raise SystemExit("REFUSED: a tombstone id is also a pack row id (a pack cannot write and delete one id)")
    for t in tombstones:
        if t.get("reason") != SUPERSEDED or not t.get("url"):
            raise SystemExit(f"REFUSED: tombstone {t.get('id')} must be reason {SUPERSEDED!r} with a url")


def write_pack(queue_dir: str, pack_id: str, rows: list[dict], *, collection: str, family: str,
               note: str, task: int, source: dict | None = None, name: str = "restore",
               tombstones: list[dict] | None = None) -> dict:
    check_rows(rows)
    # task 864: a collection that declares I1 takes only I1 ids; a restored row keeps its payload and
    # vectors byte for byte, and only its id moves (recorded in the manifest)
    from cytir.index import id_rule as IR
    try:
        rows, id_map = IR.key_points_for(collection, rows)
    except IR.IdRuleRefused as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    if tombstones:
        tombstones = [dict(t, replaced_by=[id_map.get(x, x) for x in t["replaced_by"]])
                      if t.get("replaced_by") else t for t in tombstones]
        check_tombstones(tombstones, rows)
    out = Path(queue_dir) / pack_id
    if out.exists():
        raise SystemExit(f"REFUSED: {out} already exists")
    out.mkdir(parents=True)
    body = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in rows)
    sha = hashlib.sha256(body).hexdigest()
    points_path = out / f"{name}-points.{sha}.jsonl.gz"
    with gzip.open(points_path, "wb") as fh:
        fh.write(body)
    tomb = b"".join((json.dumps(t, ensure_ascii=False) + "\n").encode("utf-8") for t in tombstones or [])
    manifest = {
        "schema": MANIFEST_SCHEMA, "kind": "content_pack", "pack_id": pack_id, "family": family,
        "collection": collection, "pack_points": str(points_path), "pack_points_sha256": sha,
        "rows": len(rows), "composed_at": _now(), "reason": note, "task": task,
        "restore_source": source or {},
    }
    if IR.declared_id_rule(collection):
        manifest.update(id_rule=IR.declared_id_rule(collection), id_map=id_map)
    if tombstones:
        (out / "tombstones.jsonl").write_bytes(tomb)
        manifest.update({
            "tombstones": str(out / "tombstones.jsonl"), "tombstones_sha256": hashlib.sha256(tomb).hexdigest(),
            "tombstones_count": len(tombstones),
            "tombstones_by_reason": {SUPERSEDED: len(tombstones)},
        })
    mbytes = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    (out / "manifest.json").write_bytes(mbytes)
    (out / "READY").write_text(hashlib.sha256(mbytes).hexdigest() + "\n")
    return manifest


def read_rows(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--qdrant-url", required=True)
    ex.add_argument("--collection", required=True)
    ex.add_argument("--ids-file", required=True, help="one point id per line")
    ex.add_argument("--out", required=True)
    tb = sub.add_parser("tombstones", help="build `superseded` tombstones for stored rows the pack's cards replace")
    tb.add_argument("--qdrant-url", required=True)
    tb.add_argument("--collection", required=True)
    tb.add_argument("--ids-file", required=True, help="ids of the STORED rows to tombstone, one per line")
    tb.add_argument("--rows", required=True, help="the pack's rows (replacing cards)")
    tb.add_argument("--out", required=True)
    tb.add_argument("--export-out", required=True, help="the stored rows (payload + vectors) kept for a rollback")
    tb.add_argument("--detail", required=True)
    co = sub.add_parser("compose")
    co.add_argument("--rows", required=True)
    co.add_argument("--queue-dir", required=True)
    co.add_argument("--pack-id", required=True)
    co.add_argument("--collection", required=True)
    co.add_argument("--family", default="general_web")
    co.add_argument("--note", required=True)
    co.add_argument("--task", type=int, required=True)
    co.add_argument("--source-note", default="")
    co.add_argument("--tombstones", default=None, help="tombstones.jsonl from the `tombstones` verb")
    args = ap.parse_args(argv)
    if args.cmd == "export":
        ids = [ln.strip() for ln in Path(args.ids_file).read_text().splitlines() if ln.strip()]
        rows = export_rows(args.qdrant_url, args.collection, ids)
        body = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in rows)
        with gzip.open(args.out, "wb") as fh:
            fh.write(body)
        print(f"[restore-pack] exported {len(rows)} row(s) from {args.qdrant_url}/{args.collection} "
              f"to {args.out} (sha256 {hashlib.sha256(body).hexdigest()})")
        return 0
    if args.cmd == "tombstones":
        rows = read_rows(args.rows)
        ids = [ln.strip() for ln in Path(args.ids_file).read_text().splitlines() if ln.strip()]
        stored_rows = export_rows(args.qdrant_url, args.collection, ids)
        tombs = build_tombstones({r["id"]: r["payload"] for r in stored_rows}, rows, detail=args.detail)
        Path(args.out).write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tombs), encoding="utf-8")
        body = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in stored_rows)
        with gzip.open(args.export_out, "wb") as fh:
            fh.write(body)
        print(f"[restore-pack] {len(tombs)} superseded tombstone(s) to {args.out}; stored rows kept in "
              f"{args.export_out} (sha256 {hashlib.sha256(body).hexdigest()})")
        return 0
    rows = read_rows(args.rows)
    tombs = ([json.loads(x) for x in Path(args.tombstones).read_text(encoding="utf-8").splitlines() if x.strip()]
             if args.tombstones else None)
    man = write_pack(args.queue_dir, args.pack_id, rows, collection=args.collection, family=args.family,
                     note=args.note, task=args.task, source={"rows_file": args.rows, "note": args.source_note},
                     tombstones=tombs)
    print(f"[restore-pack] composed {args.pack_id}: {man['rows']} row(s), {man.get('tombstones_count', 0)} tombstone(s), "
          f"points sha256 {man['pack_points_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
