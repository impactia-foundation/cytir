"""Re-fetch of pages the twin holds and no sealed original covers (task 676).

Real pages: the 13 pages of task 676's page-date-rows fixture (gzipped JSONL)
(read from sealed originals of the 2026-09-28 stage). The spool roster, the canonical
url join, the preserved-fetch seal and `published-from-pages` are the real code; only
the network (an httpx MockTransport) and Qdrant's scroll are faked.

What each group would have caught:
* a url the twin holds that a spool already captures must not be fetched again, and
  `twin_urls == captured + missing` must hold;
* every url reaches exactly one terminal, and a transport failure is `error`, never
  a dead url;
* a sealed page must be readable by the roster (`stage-*/news_documents.jsonl`) and
  carry the date the existing `published-from-pages` verb derives from it;
* resuming costs no request.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import httpx
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "tests"))

import cytir.news.page_date_refetch as NR  # noqa: E402
import cytir.news.published_page_date as PPD  # noqa: E402
import published_page_date_world as TP  # noqa: E402  (tests/)

UA = "authoritative-web-test/1.0 (+https://websear.ch)"


class DomainTwin(TP.FakeTwin):
    """FakeTwin plus the `domain` scroll the tool uses (indexed keyword, `match any`)."""

    def api(self, url, path, body=None, method=None):
        base = f"/collections/{self.collection}"
        if path == base + "/points/scroll":
            (cond,) = body["filter"]["must"]
            if cond["key"] == "domain":
                ids = sorted(i for i, pl in self.points.items()
                             if pl.get("domain") in cond["match"]["any"])
                start = ids.index(body["offset"]) if body.get("offset") in ids else 0
                chunk = ids[start:start + self.page]
                nxt = ids[start + self.page] if start + self.page < len(ids) else None
                return {"points": [{"id": i, "payload": {"url": self.points[i]["url"]}}
                                   for i in chunk], "next_page_offset": nxt}
        return super().api(url, path, body, method)


@pytest.fixture
def world(tmp_path):
    pages = [p for p in TP._pages() if p["domain"].endswith("swissinfo.ch")]
    assert len(pages) >= 4
    covered, open_ = pages[:2], pages[2:]
    work = tmp_path / "work"
    TP._write_spool(work, "2026-09-28", covered)
    twin = DomainTwin(TP.TWIN, pages)
    return work, twin, covered, open_


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": UA},
                        follow_redirects=True)


class Clock:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def _pol(clock=None, **kw):
    c = clock or Clock()
    return NR.Politeness(kw.get("delay", 1.0), kw.get("mbps", 1.0), sleep=c.sleep, clock=c.now)


def test_plan_lists_only_the_urls_no_spool_captures(world):
    work, twin, covered, open_ = world
    p = NR.plan(work, TP.TWIN_URL, TP.TWIN, ["swissinfo.ch"], api=twin.api)
    assert p["roster"] == ["stage-2026-09-28"] and p["roster_size"] == 1
    assert p["twin_urls"] == len(covered) + len(open_)
    assert p["captured"] == len(covered)
    assert p["missing"] == len(open_)
    assert sorted(p["missing_urls"]) == sorted(r["url"] for r in open_)
    assert p["twin_urls"] == p["captured"] + p["missing"]


def test_plan_joins_on_the_canonical_url_not_the_spelling(tmp_path):
    pages = [p for p in TP._pages() if p["domain"].endswith("swissinfo.ch")]
    work = tmp_path / "work"
    TP._write_spool(work, "2026-09-28", pages[:1])
    other = pages[0]["url"]
    spelled = {other: other.replace("https://www.", "http://")}
    twin = DomainTwin(TP.TWIN, pages[:1], spelled=spelled)
    p = NR.plan(work, TP.TWIN_URL, TP.TWIN, ["swissinfo.ch"], api=twin.api)
    assert p["captured"] == 1 and p["missing"] == 0


def test_fetch_seals_each_page_and_reaches_one_terminal_per_url(world, tmp_path):
    work, twin, _, open_ = world
    urls = [r["url"] for r in open_]
    html = {r["url"]: r["html_signals"] for r in open_}
    calls = collections.Counter()

    def handler(req):
        u = str(req.url)
        calls[u] += 1
        if u == urls[0] and calls[u] == 1:
            return httpx.Response(503, headers={"retry-after": "1"})   # retried, then 200
        if len(urls) > 2 and u == urls[1]:
            return httpx.Response(404, text="gone")
        if len(urls) > 3 and u == urls[2]:
            return httpx.Response(200, text="{}", headers={"content-type": "application/json"})
        return httpx.Response(200, text=html[u], headers={"content-type": "text/html; charset=utf-8"})

    blocked = urls[-1] if len(urls) > 4 else None
    spool = NR.spool_dir(work, "swissinfo")
    out = NR.fetch(urls, spool, _client(handler), lambda u: u != blocked, UA, _pol(),
                   now=lambda: "2026-09-30T18:00:00Z")
    t = out["terminals"]
    assert sum(t.values()) == len(urls) == out["urls"]
    assert t.get("http_404", 0) == (1 if len(urls) > 2 else 0)
    assert t.get("non_article", 0) == (1 if len(urls) > 3 else 0)
    assert t.get("robots_disallowed", 0) == (1 if blocked else 0)
    assert calls[urls[0]] == 2                                # the 503 was retried
    docs = [json.loads(line) for line in (spool / "news_documents.jsonl").read_text().splitlines()]
    assert len(docs) == t["sealed"] >= 1
    for d in docs:
        assert PPD.original_path(spool, d["lineage"]["original_sha256"]).is_file()
        assert d["fetched_at"] == "2026-09-30T18:00:00Z"


def test_a_transport_failure_is_an_error_not_a_verdict(world):
    work, _, _, open_ = world
    urls = [open_[0]["url"]]

    def handler(req):
        raise httpx.ConnectError("boom")

    out = NR.fetch(urls, NR.spool_dir(work, "x"), _client(handler), lambda u: True, UA, _pol())
    assert out["terminals"] == {"error": 1}
    assert not (NR.spool_dir(work, "x") / "news_documents.jsonl").read_text()


def test_resume_makes_no_request_and_writes_no_second_document(world):
    work, _, _, open_ = world
    urls = [r["url"] for r in open_]
    html = {r["url"]: r["html_signals"] for r in open_}
    spool = NR.spool_dir(work, "swissinfo")
    n = []

    def handler(req):
        n.append(1)
        return httpx.Response(200, text=html[str(req.url)], headers={"content-type": "text/html"})

    first = NR.fetch(urls, spool, _client(handler), lambda u: True, UA, _pol())
    sent = len(n)
    second = NR.fetch(urls, spool, _client(handler), lambda u: True, UA, _pol())
    assert len(n) == sent and second["terminals"] == first["terminals"]
    lines = (spool / "news_documents.jsonl").read_text().splitlines()
    assert len(lines) == first["terminals"]["sealed"]


def test_refetched_originals_feed_published_from_pages(world):
    """The point of the tool: after the fetch the roster covers every twin url, and the
    existing verb derives the page date from the new originals (no `original_missing`)."""
    work, twin, covered, open_ = world
    urls = [r["url"] for r in open_]
    html = {r["url"]: r["html_signals"] for r in open_}

    def handler(req):
        return httpx.Response(200, text=html[str(req.url)], headers={"content-type": "text/html"})

    NR.fetch(urls, NR.spool_dir(work, "swissinfo"), _client(handler), lambda u: True, UA, _pol(),
             now=lambda: "2026-09-29T12:00:00Z")
    after = NR.plan(work, TP.TWIN_URL, TP.TWIN, ["swissinfo.ch"], api=twin.api)
    assert after["missing"] == 0 and after["captured"] == after["twin_urls"]
    plan = PPD.plan_from_pages(work, TP.TWIN_URL, TP.TWIN, api=twin.api)
    assert plan["url_terminals"].get("original_missing", 0) == 0
    assert plan["url_terminals"]["dated"] + plan["url_terminals"].get("page_none", 0) == \
        plan["url_terminals"]["urls"]


def test_politeness_spaces_requests_and_steers_on_sealed_bytes():
    c = Clock()
    pol = NR.Politeness(0.2, 1.0, sleep=c.sleep, clock=c.now)
    assert pol.delay_s == NR.MIN_DELAY_S                      # never faster than 1 a second
    pol.wait()
    assert c.slept == []
    pol.wait()
    assert sum(c.slept) == pytest.approx(1.0)
    c.slept.clear()
    pol.bytes = 10_000_000                                    # 10 MB sealed in ~1 s
    pol.wait()
    assert c.t >= 10.0                                        # owed time at 1 MB/s
