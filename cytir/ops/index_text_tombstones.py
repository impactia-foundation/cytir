"""Tombstone packs for the index-text defects: the two of 2026-10-08 (task 767) and `soft_404` (task 1363).

    canonical_duplicate  a `?authuser=N` (or other tracking-key) copy of a page whose canonical
                         row is served; the variant row goes, the canonical row stays.
    binary_content       a row whose stored snippet is a binary file read as text
                         (`cytir.core.prose_guard`).
    soft_404             a row whose stored title and snippet match the declared soft-404 detector
                         (`cytir.sources.soft404`, `data/soft-404-signals.v1.json`): an error page.

All three are PURE, export-backed tombstone packs (`promote_queue.EXPORT_BACKED_REASONS`): zero points, one
reason per pack, a restore copy of every removed row (all vectors, full payload) written and
verified BEFORE the pack is composed, twin first through the hourly drain, prod the same pass.
Every candidate is re-checked here with the drain's own per-id rules (`check_canonical_duplicates`,
`check_binary_content`, `check_soft_404`), so a row the drain would refuse never enters a pack; those ids are
counted in the plan, not hidden.

    python -m cytir.ops.index_text_tombstones census  --reason soft_404 --twin-url U --collection C --out F
    python -m cytir.ops.index_text_tombstones plan    --reason R --candidates F --twin-url U --collection C
    python -m cytir.ops.index_text_tombstones compose --reason R --candidates F --twin-url U --collection C \\
        --export-dir E --queue-dir Q --pack-prefix P [--max-ids 5000] --note "..."

`census` is one niced, read-only scroll of the TWIN (title, snippet, url, family, card flag only); it
writes the candidates JSONL `{"id","url","title","snippet","match"}` and never touches prod.
`--candidates` is a JSONL of `{"id", "url", ...}` read off the TWIN (a census run). Reads are by id
and by filtered url scroll on the twin; prod is never read. `compose` refuses a queue-dir that is the
live queue unless `--queue-dir` is given explicitly and refuses 0 passing rows.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import sys

from cytir.core.canon import strip_tracking_url
from cytir.index import point_delete as PD
from cytir.index import promote_queue as Q
from cytir.index import row_decode as RD   # canonical stored rows, any profile (task 855)
from cytir.index import url_group_supersede as UGS
from cytir.ops import spam_tombstones as ST
from cytir.sources import soft404 as SOFT404

TASK = 767
SOFT_404_TASK = 1363
CENSUS_FIELDS = ["url", "title", "snippet", "provenance", "source_type", "entity_kind", "qid"]
CENSUS_PAGE = 1000
MAX_IDS = 5000          # per pack: a delete of this size is a small write for the 30 GB serving box
FIELDS = ["url", "source_type", "entity_kind", "qid", "provenance", "snippet", "domain", "content_digest",
          "title"]


def load_candidates(path: pathlib.Path) -> list[dict]:
    rows = [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise SystemExit(f"REFUSED: {path} is empty")
    return rows


def classify(reason: str, candidates: list[dict], twin_url: str, collection: str) -> dict:
    """`{"pass": [tombstone...], "absent": n, "failed": {id: why}, "stored": {id: payload}}`.
    A tombstone is `{id, url, domain, reason, detail}`."""
    if reason not in (Q.CANONICAL_DUPLICATE, Q.BINARY_CONTENT, Q.SOFT_404):
        raise SystemExit(f"REFUSED: reason {reason!r} is not one of canonical_duplicate, "
                         "binary_content, soft_404")
    ids = [c["id"] for c in candidates]
    stored = RD.reader(collection).decode_map(PD.fetch_payload_fields(twin_url, collection, ids,
                                                                       FIELDS))
    absent = [i for i in ids if i not in stored]
    tombs = [{"id": i, "url": stored[i].get("url", ""), "reason": reason,
              "domain": stored[i].get("domain", ""),
              "detail": ("variant of " + strip_tracking_url(stored[i].get("url", ""))
                         if reason == Q.CANONICAL_DUPLICATE
                         else "soft-404 page" if reason == Q.SOFT_404 else "binary snippet")}
             for i in ids if i in stored]
    pre: list[str] = []
    if reason == Q.CANONICAL_DUPLICATE:
        tombs, pre = _plan_pages(tombs, stored, twin_url, collection)
        problems = pre + Q.check_canonical_duplicates(
            tombs, stored, target_url=twin_url, collection=collection,
            export_ids={t["id"] for t in tombs})
    elif reason == Q.SOFT_404:
        problems = Q.check_soft_404(tombs, stored)
    else:
        problems = Q.check_binary_content(tombs, stored)
    failed = {p.split(":", 1)[0]: p.split(":", 1)[1].strip() for p in problems}
    return {"pass": [t for t in tombs if t["id"] not in failed], "absent": len(absent),
            "failed": failed, "stored": stored}


def _plan_pages(tombs: list[dict], stored: dict, twin_url: str, collection: str
                ) -> "tuple[list[dict], list[str]]":
    """Which variant rows go, and which spelling of each page stays. Per (canonical url, family)
    group of candidates: if a row at the CANONICAL spelling is served (same family, not a card), the
    canonical rows are the survivor. Otherwise ONE spelling stays as `survivor_url`: the one whose
    digests cover the most rows of the other spellings (ties: more rows, then the smallest url).
    A variant row goes ONLY when its `content_digest` is on the survivor (task 773): a row whose
    text the survivor does not hold stays, and is named. A candidate that is already canonical is
    the page itself: out, named."""
    pre: list[str] = []
    variants = []
    for t in tombs:
        if strip_tracking_url(t["url"]) == t["url"]:
            pre.append(f"{t['id']}: already canonical")
        else:
            variants.append(t)
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for t in variants:
        groups[(strip_tracking_url(t["url"]), stored[t["id"]].get("provenance"))].append(t)  # canonical row
    members = Q.served_group_members(twin_url, collection, sorted({k[0] for k in groups}))
    out = []
    for (canon, prov), rows in groups.items():
        gkey = Q.US.payload_group_key({"url": canon})
        canon_digests = {pl.get("content_digest") for _mid, pl in members.get(gkey, [])  # canonical row
                         if pl.get("provenance") == prov and str(pl.get("url") or "") == canon  # canonical row
                         and not Q._is_card_row(pl)}
        digest = lambda t: stored[t["id"]].get("content_digest")  # noqa: E731 -- canonical row
        if canon_digests:
            for t in rows:
                if digest(t) in canon_digests:
                    out.append(dict(t))
                else:
                    pre.append(f"{t['id']}: text not on the canonical row (content_digest)")
            continue
        by_url: dict[str, list[dict]] = collections.defaultdict(list)
        for t in rows:
            by_url[t["url"]].append(t)
        digs = {u: {digest(t) for t in ts if digest(t)} for u, ts in by_url.items()}

        def covered(u):
            return sum(1 for t in rows if t["url"] != u and digest(t) in digs[u])
        keep = min(by_url, key=lambda u: (-covered(u), -len(by_url[u]), u))
        for t in rows:
            if t["url"] == keep:
                continue
            if digest(t) in digs[keep]:
                out.append(dict(t, survivor_url=keep))
            else:
                pre.append(f"{t['id']}: text not on the kept spelling (content_digest)")
    return out, pre


def detector_sha256() -> str:
    """sha256 of the committed detector file the per-id rule runs against (kept in the pack note)."""
    return hashlib.sha256(pathlib.Path(SOFT404.DEFAULT_PATH).read_bytes()).hexdigest()


def census(reason: str, twin_url: str, collection: str, out: pathlib.Path) -> dict:
    """One read-only scroll of the TWIN; the candidates JSONL for `reason` (soft_404 only).
    Reads title, snippet, url, family and the card flag; a card is never a candidate."""
    if reason != Q.SOFT_404:
        raise SystemExit(f"REFUSED: census is defined for {Q.SOFT_404!r} only, not {reason!r}")
    scanned = 0
    hits: list[dict] = []
    offset = None
    while True:
        body = {"limit": CENSUS_PAGE, "with_payload": CENSUS_FIELDS, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        res = PD._api(twin_url, f"/collections/{collection}/points/scroll", body)
        if not isinstance(res, dict) or "points" not in res:
            raise SystemExit(f"REFUSED: the scroll returned {type(res).__name__} without `points` after "
                             f"{scanned} rows; a census that stops early under-reports")
        for pt in res.get("points") or []:
            scanned += 1
            pl = RD.as_canonical_row(pt.get("payload") or {})
            if Q._is_card_row(pl):
                continue
            match = SOFT404.detect(str(pl.get("title") or ""), str(pl.get("snippet") or ""))
            if match is not None:
                hits.append({"id": str(pt["id"]), "url": pl.get("url", ""), "title": pl.get("title"),
                             "snippet": pl.get("snippet"), "match": match,
                             "error_path": SOFT404.error_path(pl.get("url"))})
        offset = res.get("next_page_offset")
        if offset is None:
            break
    pathlib.Path(out).write_text("".join(json.dumps(h, ensure_ascii=False) + "\n" for h in hits),
                                 encoding="utf-8")
    info = PD._api(twin_url, f"/collections/{collection}") or {}
    return {"reason": reason, "scanned": scanned, "points_count": info.get("points_count"), "candidates": len(hits), "out": str(out),
            "detector_sha256": detector_sha256()}


def summary(reason: str, res: dict, n_candidates: int) -> dict:
    ok = res["pass"]
    return {"reason": reason, "candidates": n_candidates, "absent_on_twin": res["absent"],
            "refused_by_the_drains_rule": len(res["failed"]), "tombstones": len(ok),
            "by_domain": dict(collections.Counter(t["domain"] for t in ok).most_common(15)),
            "by_provenance": dict(collections.Counter(
                res["stored"][t["id"]].get("provenance") for t in ok).most_common(10)),  # canonical row
            "refusal_examples": list(res["failed"].items())[:5],
            "packs_at_max_ids": -(-len(ok) // MAX_IDS), "twin_count_delta": -len(ok)}


def compose(reason: str, candidates: list[dict], twin_url: str, collection: str, *, export_dir: str,
            queue_dir: str, pack_prefix: str, max_ids: int, note: str,
            prod_url: "str | None" = None, prod_collection: str = "aw_prod") -> list[dict]:
    res = classify(reason, candidates, twin_url, collection)
    if not res["pass"]:
        raise SystemExit("REFUSED: 0 rows pass the drain's rule, nothing to tombstone")
    out = []
    ok = res["pass"]
    for n, i in enumerate(range(0, len(ok), max_ids), 1):
        part = ok[i:i + max_ids]
        pack_id = f"{pack_prefix}-{n:03d}"
        path = pathlib.Path(export_dir) / f"{pack_id}-removed-rows.jsonl.gz"
        urls = {t["id"]: t["url"] for t in part}
        export = UGS.write_export(twin_url, collection, [t["id"] for t in part], urls, path)
        family = "index-text-defects"
        extra = {"export": {"path": export["path"], "sha256": export["sha256"], "rows": export["rows"]}}
        if prod_url:
            # PROD's own restore copy, by id (retrieve endpoint, never a scroll), beside the twin's
            ppath = pathlib.Path(export_dir) / f"{pack_id}-prod-removed-rows.jsonl.gz"
            pexp, present = UGS.write_prod_export(prod_url, prod_collection, [t["id"] for t in part],
                                                  urls, ppath)
            extra["prod_export"] = {"path": pexp["path"], "sha256": pexp["sha256"], "rows": pexp["rows"],
                                    "requested": len(part), "absent_on_prod": len(part) - len(present)} \
                if pexp else {"path": None, "rows": 0, "requested": len(part), "absent_on_prod": len(part)}
        if reason == Q.SOFT_404:
            note = f"{note} [detector data/soft-404-signals.v1.json sha256={detector_sha256()}]"
        m = ST.write_pack(queue_dir, pack_id, part, family=family, collection=collection, note=note,
                          reason=reason, task=SOFT_404_TASK if reason == Q.SOFT_404 else TASK,
                          extra=extra)
        out.append({k: m[k] for k in ("pack_id", "tombstones_count", "tombstones_by_reason", "export")}
                   | ({"prod_export": m["prod_export"]} if prod_url else {}))
    return out


def main(argv=None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    cp = sub.add_parser("census")
    cp.add_argument("--reason", required=True)
    cp.add_argument("--twin-url", required=True)
    cp.add_argument("--collection", required=True)
    cp.add_argument("--out", required=True, type=pathlib.Path)
    for name in ("plan", "compose"):
        p = sub.add_parser(name)
        p.add_argument("--reason", required=True)
        p.add_argument("--candidates", required=True, type=pathlib.Path)
        p.add_argument("--twin-url", required=True)
        p.add_argument("--collection", required=True)
    c = sub.choices["compose"]
    c.add_argument("--export-dir", required=True)
    c.add_argument("--queue-dir", required=True)
    c.add_argument("--pack-prefix", required=True)
    c.add_argument("--max-ids", type=int, default=MAX_IDS)
    c.add_argument("--note", required=True)
    c.add_argument("--prod-url", default=None,
                   help="the prod tunnel: take PROD's restore copy by id, beside the twin export")
    c.add_argument("--prod-collection", default="aw_prod")
    a = ap.parse_args(argv)
    if a.cmd == "census":
        try:
            os.nice(19)            # a read on the twin yields to everything else
        except OSError:
            pass
        print(json.dumps(census(a.reason, a.twin_url, a.collection, a.out), indent=1))
        return 0
    cands = load_candidates(a.candidates)
    if a.cmd == "plan":
        res = classify(a.reason, cands, a.twin_url, a.collection)
        print(json.dumps(summary(a.reason, res, len(cands)), indent=1, ensure_ascii=False))
    else:
        print(json.dumps(compose(a.reason, cands, a.twin_url, a.collection,
                                 export_dir=a.export_dir, queue_dir=a.queue_dir,
                                 pack_prefix=a.pack_prefix, max_ids=a.max_ids, note=a.note,
                                 prod_url=a.prod_url, prod_collection=a.prod_collection), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
