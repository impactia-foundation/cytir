"""Shared world for the page-date backfill tests: the real-pages fixture, the spool writer and a fake
twin that answers the calls the tools make. Used by `test_aw_index_published_page_date.py` and the
page-date refetch test (task 1241: moved with the reconcile tests that owned them)."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import pathlib
import sys
import uuid

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import cytir.news.published_page_date as PPD  # noqa: E402
FIXTURE = HERE / "fixtures" / "payload_profile_reconcile" / "page-date-rows-fixture.jsonl.gz"
TWIN_URL, TWIN = "http://twin", "aw_e8"
CHUNKS = 2


def _pages():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _pid(url: str, n: int = 0) -> str:
    return str(uuid.UUID(bytes=hashlib.sha256(f"twin#{url}#{n}".encode()).digest()[:16]))


def _staging_pid(url: str) -> str:
    return str(uuid.UUID(bytes=hashlib.sha256(f"staging#{url}".encode()).digest()[:16]))


def _write_spool(work: pathlib.Path, day: str, pages, *, fetched_at=None, drop_original=()):
    spool = work / f"stage-{day}"
    spool.mkdir(parents=True)
    docs = []
    for r in pages:
        sha = r["original_sha256"]
        if r["url"] not in drop_original:
            path = PPD.original_path(spool, sha)
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                fh.write(r["html_signals"])
        docs.append({"url": r["url"], "fetched_at": fetched_at or f"{day}T07:00:00Z",
                     "lineage": {"original_sha256": sha},
                     "chunk_point_ids": [_staging_pid(r["url"])]})
    (spool / "news_documents.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in docs))
    return spool


class FakeTwin:
    """Qdrant's `scroll` (filtered `match any` on `url`), `set_payload` batch, point
    retrieve and collection info. The twin's ids are NOT the staging ids."""

    def __init__(self, collection, pages, *, absent=(), ignore_writes=False,
                 spelled=None, page=3):
        self.collection, self.ignore_writes, self.page = collection, ignore_writes, page
        self.points = {}
        for r in pages:
            if r["url"] in absent:
                continue
            stored = (spelled or {}).get(r["url"], r["url"])
            for n in range(CHUNKS):
                self.points[_pid(r["url"], n)] = {
                    "url": stored, "domain": r["domain"],
                    **({"published": r["row_published"]} if r["row_published"] else {})}
        self.ops = []

    def api(self, url, path, body=None, method=None):
        base = f"/collections/{self.collection}"
        if path == base:
            return {"points_count": len(self.points)}
        if path == base + "/points/scroll":
            (cond,) = body["filter"]["must"]
            assert cond["key"] == "url" and "any" in cond["match"]
            ids = sorted(i for i, pl in self.points.items() if pl["url"] in cond["match"]["any"])
            start = ids.index(body["offset"]) if body.get("offset") in ids else 0
            chunk = ids[start:start + self.page]
            nxt = ids[start + self.page] if start + self.page < len(ids) else None
            return {"points": [{"id": i, "payload": {k: v for k, v in self.points[i].items()
                                                     if k in body["with_payload"]}}
                               for i in chunk], "next_page_offset": nxt}
        if path == base + "/points/batch?wait=true":
            for op in body["operations"]:
                assert list(op) == ["set_payload"]
                spec = op["set_payload"]
                assert list(spec["payload"]) == ["published"]
                self.ops.append(op)
                if not self.ignore_writes:
                    for pid in spec["points"]:
                        self.points[pid].update(spec["payload"])
            return {}
        raise AssertionError(path)

    def fetch(self, url, collection, ids, fields, batch=1000):
        return {i: {k: copy.deepcopy(v) for k, v in self.points[i].items()
                    if fields is True or k in fields}
                for i in ids if i in self.points}
