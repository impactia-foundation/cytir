"""Tombstone pack for spam / hijacked-host rows that already SERVE (task 701).

`cytir.ops.spam_rows` keeps spam out of a pack that has not landed (reseal + exclusion file). When the
rule or the host list improves AFTER a pack landed, the rows are on twin and prod and only a delete
removes them. This module composes that delete the way the queue takes it: a pure-tombstone
`content_pack` (zero points, `tombstones.jsonl`, reason `takedown`), written to
`<queue-dir>/<pack-id>`; the hourly drain applies it twin first, then prod, and verifies.

    python -m cytir.ops.spam_tombstones plan    --exclude-file F --twin-url U --collection C
    python -m cytir.ops.spam_tombstones compose --exclude-file F --twin-url U --collection C \\
                                             --queue-dir Q --pack-id P --note "..."
    python -m cytir.ops.spam_tombstones domains --twin-url U --collection C --host-lists H [--stage-domains ...]

`plan` reads ids by id retrieval only (no scan) and prints how many of the exclusion file's ids are
SERVING; `compose` writes only those (an absent id is already at its goal state). `domains` counts the
serving rows of every held-out host by `domain` (indexed field, exact count): the screen of the
previously landed tranches; it writes nothing. Refuses a plan of 0 serving rows.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

from cytir.index import point_delete as PD
from cytir.ops import spam_rows as SR
from cytir.sources import url_admission as UA

MANIFEST_SCHEMA = "aw.promotion-queue-pack.v1"
REASON = "takedown"


def load_exclude(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"REFUSED: {path} is empty")
    return rows


def serving(url: str, collection: str, rows: list[dict]) -> list[dict]:
    """The rows of the exclusion file whose id is present in `collection` (id retrieval)."""
    ids = [r["point_id"] for r in rows]
    got = PD.fetch_payload_fields(url, collection, ids, ["url", "domain"])
    out = []
    for r in rows:
        p = got.get(str(r["point_id"]))
        if p is not None:
            out.append({"id": r["point_id"], "url": p.get("url") or r.get("url", ""),
                        "domain": p.get("domain") or r.get("domain", ""), "reason": REASON,
                        "detail": r.get("reason", "")})
    return out


def _scroll(url: str, collection: str, must: list[dict]) -> list[dict]:
    """Every row matching `must` (exact payload matches), id + url + domain, read-only, paged."""
    out, offset = [], None
    while True:
        body = {"filter": {"must": must}, "limit": 256, "with_payload": ["url", "domain"], "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        res = PD._api(url, f"/collections/{collection}/points/scroll", body)
        out += [{"id": str(pt["id"]), **(pt.get("payload") or {})} for pt in res["points"]]
        offset = res.get("next_page_offset")
        if offset is None:
            return out


def label(url: str) -> str | None:
    """The admission rule's label for a stored url, or None when the rule has no quarrel with it."""
    why = UA.refusal(url)
    if why:
        return why
    return UA.REASON_TRACKING if "?" in url and UA.fold_url(url) != url else None


def resolve(url: str, collection: str, *, urls: list[str], domains: list[str],
            allow_orphan_variants: bool = False, warn=print) -> list[dict]:
    """The exclusion-file rows (`point_id`, `url`, `reason` = the admission rule's label) of the rows the
    URL admission rule (`cytir.sources.url_admission`) refuses or folds. `urls` are exact urls (filtered
    scroll on the indexed `url` field); `domains` are scrolled whole (small hosts only) and classified by
    the rule. Task 2713. Reads one cluster (the twin); writes nothing."""
    found: dict[str, dict] = {}
    for u in urls:
        for r in _scroll(url, collection, [{"key": "url", "match": {"value": u}}]):
            found[r["id"]] = r
    for d in domains:
        for r in _scroll(url, collection, [{"key": "domain", "match": {"value": d}}]):
            found[r["id"]] = r
    out = []
    served = {r.get("url") for r in found.values()}
    for i, r in sorted(found.items()):
        why = label(r.get("url") or "")
        if why == UA.REASON_TRACKING and not allow_orphan_variants:
            canon = UA.fold_url(r["url"])
            if canon not in served and not _scroll(url, collection, [{"key": "url", "match": {"value": canon}}]):
                warn(f"WARNING: {r['url']} is the only copy (its folded url {canon} is not served); "
                     "left out (--allow-orphan-variants to include)")
                continue
        if why:
            out.append({"point_id": i, "url": r.get("url", ""), "domain": r.get("domain", ""), "reason": why})
    return out


def plan(url: str, collection: str, exclude_file: Path) -> dict:
    rows = load_exclude(exclude_file)
    live = serving(url, collection, rows)
    return {"excluded_ids": len(rows), "serving": len(live),
            "by_detail": dict(collections.Counter(t["detail"] for t in live)),
            "by_domain": dict(collections.Counter(t["domain"] for t in live).most_common(40))}


def write_pack(queue_dir: str, pack_id: str, tombstones: list[dict], *, family: str, collection: str,
               note: str, now: str | None = None, reason: str = REASON, task: int = 701,
               extra: dict | None = None) -> dict:
    """`reason`, `task` and `extra` (manifest keys, e.g. the `export` of an export-backed reason)
    are for `cytir.ops.index_text_tombstones` (task 767); the defaults are the spam pack of task 701."""
    if not tombstones:
        raise SystemExit("REFUSED: 0 serving rows, nothing to tombstone")
    out = Path(queue_dir) / pack_id
    if out.exists():
        raise SystemExit(f"REFUSED: {out} already exists")
    out.mkdir(parents=True)
    points = b""
    points_sha = hashlib.sha256(points).hexdigest()
    points_path = out / f"spam-points.{points_sha}.jsonl.gz"
    with gzip.open(points_path, "wb") as fh:
        fh.write(points)
    tomb = b"".join((json.dumps(t, ensure_ascii=False) + "\n").encode("utf-8") for t in tombstones)
    (out / "tombstones.jsonl").write_bytes(tomb)
    manifest = {
        "schema": MANIFEST_SCHEMA, "kind": "content_pack", "pack_id": pack_id, "family": family,
        "collection": collection, "pack_points": str(points_path), "pack_points_sha256": points_sha,
        "rows": 0, "tombstones": str(out / "tombstones.jsonl"),
        "tombstones_sha256": hashlib.sha256(tomb).hexdigest(), "tombstones_count": len(tombstones),
        "tombstones_by_reason": {reason: len(tombstones)},
        "tombstones_by_detail": dict(collections.Counter(t["detail"] for t in tombstones)),
        "note": note, "task": task,
        "composed_at": now or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(extra or {}),
    }
    body = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    (out / "manifest.json").write_bytes(body)
    (out / "READY").write_text(hashlib.sha256(body).hexdigest() + "\n")
    return manifest


def domain_counts(url: str, collection: str, domains: list[str]) -> dict[str, int]:
    """Exact count of serving rows per `domain` (indexed payload field). The `www.` spelling is counted too."""
    out = {}
    for d in domains:
        n = 0
        for spelling in (d, "www." + d):
            n += PD._api(url, f"/collections/{collection}/points/count",
                         {"filter": {"must": [{"key": "domain", "match": {"value": spelling}}]},
                          "exact": True})["count"]
        out[d] = n
    return out


def main(argv=None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rs = sub.add_parser("resolve")
    rs.add_argument("--twin-url", default=None, help="default: cytir.core.twin.configured_url()")
    rs.add_argument("--collection", default=None, help="default: cytir.core.twin.collection()")
    rs.add_argument("--allow-orphan-variants", action="store_true")
    rs.add_argument("--urls-file", type=Path, help="one exact url per line")
    rs.add_argument("--domain", action="append", default=[], help="scroll this whole domain (small hosts)")
    rs.add_argument("--out", required=True, type=Path, help="exclusion JSONL for plan / compose")
    for name in ("plan", "compose", "domains"):
        p = sub.add_parser(name)
        p.add_argument("--twin-url", required=True)
        p.add_argument("--collection", required=True)
        if name != "domains":
            p.add_argument("--exclude-file", required=True, type=Path)
    c = sub.choices["compose"]
    c.add_argument("--queue-dir", required=True)
    c.add_argument("--pack-id", required=True)
    c.add_argument("--family", required=True)
    c.add_argument("--note", required=True)
    sub.choices["domains"].add_argument("--host-lists", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "resolve":
        us = ([x.strip() for x in a.urls_file.read_text(encoding="utf-8").splitlines() if x.strip()]
              if a.urls_file else [])
        from cytir.core import twin as TW  # noqa: PLC0415
        rows = resolve(a.twin_url or TW.configured_url(), a.collection or TW.collection(), urls=us,
                       domains=a.domain, allow_orphan_variants=a.allow_orphan_variants)
        a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(json.dumps({"rows": len(rows), "by_reason": dict(collections.Counter(r["reason"] for r in rows))}))
    elif a.cmd == "plan":
        print(json.dumps(plan(a.twin_url, a.collection, a.exclude_file), indent=1))
    elif a.cmd == "compose":
        live = serving(a.twin_url, a.collection, load_exclude(a.exclude_file))
        m = write_pack(a.queue_dir, a.pack_id, live, family=a.family, collection=a.collection, note=a.note)
        print(json.dumps({k: m[k] for k in ("pack_id", "tombstones_count", "tombstones_by_detail")}))
    else:
        lists = SR.load_host_lists(a.host_lists)
        res = domain_counts(a.twin_url, a.collection, sorted(lists["hold_out"]))
        print(json.dumps({"serving_by_domain": {d: n for d, n in res.items() if n},
                          "hosts_with_rows": sum(1 for n in res.values() if n),
                          "rows": sum(res.values())}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
