#!/usr/bin/env python3
"""Re-derive `published` from the sealed originals (task 676, plan.md ruling 18).

Tim 2026-09-30: `published` is the page's own date where the page has one; the sitemap
`lastmod` is only the fallback. `sitemap_ingest` now stamps new rows that way
(`extract_published.published_for_page`). Rows already served still carry the
lastmod-derived day (swissinfo and RSI: their sitemaps carry `lastmod`). This module
plans the rewrite of those rows.

Where the page date lives
-------------------------
Not in the L1 documents: `news_documents.jsonl` stores `publication_time` = the
sitemap-derived value for exactly these publishers. The page's own date is in the
sealed ORIGINAL (`stage-<day>/pages/original/publisher_pages/<sha[:2]>/<sha>.html.gz`),
which each document cites by `lineage.original_sha256`. So the plan reads the
originals.

The join key is the URL, not the point id
-----------------------------------------
A document's `chunk_point_ids` are the STAGING ids. None of the 123,400 of them is an id
of the serving twin (measured 2026-09-30: 123,400 of 123,400 absent): the promotion
pack mints its own. So the join is the canonical URL (`upsert_supersede.group_key`),
read from the twin through the indexed `url` field (every spelling of the url,
`url_variants`), and every stored point of a url gets that page's date (`published`
is per page, all chunks share it).

Selection and proof
-------------------
* Roster: every `stage-*/news_documents.jsonl` globbed under the news work dir and
  printed with the result (never a typed list). The LATEST capture of a url wins (by
  `fetched_at`); a url captured with two different bodies is counted (`recaptured`).
* Terminals, urls: `absent_on_twin` (no stored point), `original_missing`, `page_none`
  (the page names no date: its stored rows keep their value and are COUNTED), `dated`.
  Terminals, rows of `dated` urls: `same`, `changed`, `filled` (stored row had none).
  Both tables must add up or the plan refuses.
* The date is `extract_published(html, sitemap_lastmod=None, use_htmldate=False,
  now_ref=<capture day>)`: the extractor's own page-first chain, never the sitemap.
"""
from __future__ import annotations

import collections
import datetime as _dt
import gzip
import hashlib
import json
import pathlib

from cytir.core.paths import operator_settings
from cytir.index import promote_sealed_pack as PSP
from cytir.index import supersede as US
from cytir.news import extract_published as EP



def default_work() -> str:
    """The news refresh work tree: operator setting `news_work` (refused by name when unconfigured)."""
    return operator_settings().news_work


def __getattr__(name):
    # `DEFAULT_WORK` stays readable but resolves on first ACCESS, never at import.
    if name == "DEFAULT_WORK":
        return default_work()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


PLAN_SCHEMA = "aw.published-backfill.v1"
URL_BATCH = 48
SCROLL_PAGE = 1000


def spool_roster(work: pathlib.Path) -> list[pathlib.Path]:
    """Every spool directory holding a `news_documents.jsonl`, sorted by name."""
    return sorted(p.parent for p in pathlib.Path(work).glob("stage-*/news_documents.jsonl"))


def original_path(spool: pathlib.Path, sha256: str) -> pathlib.Path:
    return spool / "pages" / "original" / "publisher_pages" / sha256[:2] / f"{sha256}.html.gz"


def latest_captures(roster: list[pathlib.Path]) -> dict[str, dict]:
    """{canonical url -> latest capture {url, sha256, spool, fetched_at,
    captures_distinct}}. The latest `fetched_at` wins."""
    out: dict[str, dict] = {}
    shas: dict[str, set] = collections.defaultdict(set)
    for spool in roster:
        with (spool / "news_documents.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                doc = json.loads(line)
                key = US.group_key(doc["url"])
                sha = doc["lineage"]["original_sha256"]
                shas[key].add(sha)
                cap = {"url": doc["url"], "sha256": sha, "spool": str(spool),
                       "fetched_at": doc.get("fetched_at") or doc["lineage"].get("fetched_at", "")}
                prev = out.get(key)
                if prev is None or cap["fetched_at"] >= prev["fetched_at"]:
                    out[key] = cap
    for key, cap in out.items():
        cap["captures_distinct"] = len(shas[key])
    return out


def page_date_of(path: pathlib.Path, capture_day: _dt.date) -> str | None:
    """The page's own date from one sealed original, or None (no defensible date)."""
    html = gzip.open(path, "rt", encoding="utf-8", errors="replace").read()
    return EP.extract_published(html, sitemap_lastmod=None, now_ref=capture_day,
                                use_htmldate=False)


def _domain(url: str) -> str:
    host = url.split("/")[2] if "//" in url else url
    return host[4:] if host.startswith("www.") else host


def rows_by_url(twin_url: str, collection: str, urls: list[str], *, api=None) -> dict:
    """{canonical url group -> [{id, published, url}]} for every stored point under any
    spelling of any of `urls`: an indexed `match any` scroll on `url`, URL_BATCH urls
    (with their variants) per call; ids and `published` only."""
    api = api or PSP._api
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for i in range(0, len(urls), URL_BATCH):
        chunk = urls[i:i + URL_BATCH]
        values = sorted(set().union(*(US.url_variants(u) for u in chunk)))
        offset = None
        while True:
            body = {"filter": {"must": [{"key": US.URL_INDEX_FIELD,
                                         "match": {"any": values}}]},
                    "limit": SCROLL_PAGE, "with_payload": ["url", "published"],
                    "with_vector": False}
            if offset is not None:
                body["offset"] = offset
            got = api(twin_url, f"/collections/{collection}/points/scroll", body) or {}
            for pt in got.get("points", []):
                pl = pt.get("payload") or {}
                out[US.group_key(pl.get("url", ""))].append(
                    {"id": str(pt["id"]), "published": pl.get("published"),
                     "url": pl.get("url")})
            offset = got.get("next_page_offset")
            if offset is None:
                break
    return dict(out)


def plan_from_pages(work, twin_url: str, twin_collection: str, *, api=None,
                    limit_urls: int | None = None) -> dict:
    """The dry run. `rows` = the stored points to rewrite; `mode: "explicit"` tells the
    apply leg to set `after` only where the store still holds `before`."""
    work = pathlib.Path(work)
    roster = spool_roster(work)
    caps = latest_captures(roster)
    keys = sorted(caps)
    if limit_urls:
        keys = keys[:limit_urls]
    stored = rows_by_url(twin_url, twin_collection, [caps[k]["url"] for k in keys], api=api)
    dates: dict[str, str | None] = {}
    urls_t: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    rows_t: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    rows: list[dict] = []
    for key in keys:
        cap = caps[key]
        dom = _domain(cap["url"])
        u = urls_t[dom]
        u["urls"] += 1
        if cap["captures_distinct"] > 1:
            u["recaptured"] += 1
        found = stored.get(key, [])
        if not found:
            u["absent_on_twin"] += 1
            continue
        r = rows_t[dom]
        r["rows"] += len(found)
        path = original_path(pathlib.Path(cap["spool"]), cap["sha256"])
        if not path.is_file():
            u["original_missing"] += 1
            r["original_missing"] += len(found)
            continue
        if cap["sha256"] not in dates:
            day = _dt.date.fromisoformat((cap["fetched_at"] or cap["spool"][-10:])[:10])
            dates[cap["sha256"]] = page_date_of(path, day)
        after = dates[cap["sha256"]]
        if after is None:
            u["page_none"] += 1
            r["page_none"] += len(found)
            continue
        u["dated"] += 1
        for row in found:
            if row["published"] == after:
                r["same"] += 1
            else:
                r["filled" if row["published"] in (None, "") else "changed"] += 1
                rows.append({"id": row["id"], "before": row["published"], "after": after,
                             "url": row["url"], "original_sha256": cap["sha256"]})
    ut, rt = collections.Counter(), collections.Counter()
    for c in urls_t.values():
        ut.update(c)
    for c in rows_t.values():
        rt.update(c)
    if ut["urls"] != ut["absent_on_twin"] + ut["original_missing"] + ut["page_none"] + ut["dated"]:
        raise RuntimeError("conservation (urls): %s" % dict(ut))
    if rt["rows"] != (rt["original_missing"] + rt["page_none"] + rt["same"]
                      + rt["changed"] + rt["filled"]):
        raise RuntimeError("conservation (rows): %s" % dict(rt))
    shifts = collections.Counter()
    for r in rows:
        if r["before"]:
            try:
                d = (_dt.date.fromisoformat(r["before"][:10])
                     - _dt.date.fromisoformat(r["after"])).days
            except ValueError:
                continue
            shifts["lastmod_later" if d > 0 else "lastmod_earlier" if d < 0 else "same_day"] += 1
    digest = hashlib.sha256("\n".join(sorted(r["id"] for r in rows)).encode()).hexdigest()
    return {"schema": PLAN_SCHEMA, "verb": "published-from-pages", "mode": "explicit",
            "source": "page-dates-from-sealed-originals", "collection": twin_collection,
            "roster": [p.name for p in roster], "roster_size": len(roster),
            "url_terminals": dict(ut), "row_terminals": dict(rt),
            "url_terminals_by_domain": {d: dict(c) for d, c in sorted(urls_t.items())},
            "row_terminals_by_domain": {d: dict(c) for d, c in sorted(rows_t.items())},
            "selected_rows": len(rows), "distinct_originals_parsed": len(dates),
            "direction_of_change": dict(shifts), "ids_sha256": digest, "rows": rows}
