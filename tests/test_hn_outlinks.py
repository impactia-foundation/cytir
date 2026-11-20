"""HN linked pages as one discovery source of the refresh lane (task 700, 697 F1, ruling 23.1).

Real bytes: `testdata/hn-linked/` holds seven pages as the build box fetched and sealed them on
2026-10-03 (honest AWbot UA, live consent first) for stories >= 100 points: a JavaScript shell
(extrabigassintelligence.com), a GitHub repo page, a vendor blog (cloudflare), the Anthropic
model page, an LWN article, an EFF post and frogandtoad.ai (an image page with no text). The
network is not called: a scripted client serves those bytes.
"""
from __future__ import annotations

import gzip
import json
import os
import pathlib
import sys

import httpx
import pytest

@pytest.fixture(autouse=True)
def _operator_identity(monkeypatch):
    """Operator identity and offline language gate for each test, restored after it. A module-level setdefault leaked them into every later test."""
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")  # offline: degraded language gate, as the lane tests do
    monkeypatch.setenv("AW_OPERATOR_URL", "https://websear.ch")
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "bender@impactia.org")

HERE = pathlib.Path(__file__).resolve().parent

from cytir.acquire import fetch_process as FP  # noqa: E402
from cytir.sources.refresh_direct import direct_compose as RC  # noqa: E402
from cytir.corpus.source_cards import PAYWALLED  # noqa: E402
from cytir.news import hn_outlinks as HO  # noqa: E402

DATA = HERE / "testdata" / "hn-linked"
MAN = {m["url"]: m for m in json.loads((DATA / "manifest.json").read_text())}
UA = "AWbot/test (+https://websear.ch/bot)"

SHELL = "https://www.extrabigassintelligence.com/"
GITHUB = "https://github.com/ninjahawk/livenerf"
VENDOR = "https://blog.cloudflare.com/clef-decision-models/"
MODEL = "https://www.anthropic.com/claude-sonnet-5-5"
LWN = "https://lwn.net/Articles/1097401/"
FROG = "https://www.frogandtoad.ai/"


def body(url: str) -> bytes:
    return gzip.decompress((DATA / MAN[url]["file"]).read_bytes())


def hit(url: str) -> dict:
    h = MAN[url]["hn"]
    return {"objectID": h["objectID"], "title": h["title"], "points": h["points"],
            "created_at_i": h["created_at_i"], "url": url}


class Client:
    """Serves `script(url) -> httpx.Response`; records every request."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def get(self, url, follow_redirects=False):
        self.calls.append(url)
        return self.script(url)


def ok_page(url):
    req = httpx.Request("GET", url, headers={"User-Agent": UA})
    return httpx.Response(200, content=body(url), headers={"content-type": "text/html"},
                          request=req)


def status(url, code, **headers):
    return httpx.Response(code, content=b"", headers=headers,
                          request=httpx.Request("GET", url, headers={"User-Agent": UA}))


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def fetcher(tmp_path, script, *, robots_ok=lambda rp, u: True, delay=0.0, clock=None):
    clock = clock or Clock()
    c = Client(script)
    f = HO.LinkedFetcher(c, preserve_root=tmp_path / "sealed", ua=UA,
                         robots=lambda host: (None, delay), robots_ok=robots_ok,
                         sleep=clock.sleep, clock=clock)
    return f, c, clock


def build_rows(url, html, meta):
    return RC.build_rows_for_candidate(url, html, meta=meta)


def lang(text, url):
    return FP.detect_lang(text, url, "en")[0]


# ------------------------------------------------------------------ candidates
def test_story_candidates_keep_one_per_outlink_the_strongest_story():
    weak = {**hit(GITHUB), "points": 120, "objectID": "1"}
    strong = {**hit(GITHUB), "points": 920, "objectID": "2", "url": GITHUB + "/"}
    below = {**hit(VENDOR), "points": 99}
    selfpost = {"objectID": "3", "title": "Ask HN: x", "points": 400, "url": None,
                "created_at_i": 1}
    got = HO.story_candidates([weak, strong, below, selfpost, hit(SHELL)])
    urls = {HO.url_key(c["url"]) for c in got}
    assert urls == {HO.url_key(GITHUB), HO.url_key(SHELL)}
    assert next(c for c in got if HO.url_key(c["url"]) == HO.url_key(GITHUB))["points"] == 920
    assert all(c["published"] and len(c["published"]) == 10 for c in got)


def test_screen_drops_hn_itself_media_paywalled_and_declined_hosts():
    cands = [{"url": u, "title": "t", "points": 200, "hn_id": "1", "published": "2026-10-01"}
             for u in (GITHUB, "https://news.ycombinator.com/item?id=1",
                       "https://www.nytimes.com/2026/10/01/x.html", "https://nzz.ch/a",
                       "https://example.org/photo.PNG", "https://blocked.example/a",
                       "ftp://example.org/x", VENDOR)]
    kept, dropped = HO.screen(cands, paywalled=PAYWALLED,
                              declined=lambda h: h == "blocked.example")
    assert [c["url"] for c in kept] == [GITHUB, VENDOR]
    assert len(dropped["excluded_self_or_media"]) == 3
    assert len(dropped["excluded_paywalled"]) == 2
    assert dropped["excluded_declined_host"] == ["https://blocked.example/a"]


def test_url_variants_cover_scheme_www_and_trailing_slash():
    v = HO.url_variants("https://example.org/a/b")
    assert {"https://example.org/a/b", "http://example.org/a/b/", "https://www.example.org/a/b/",
            "http://www.example.org/a/b"} <= set(v)
    assert HO.url_key("http://www.example.org/a/") == HO.url_key("https://example.org/a")


class Twin:
    """Honest fake of `client.scroll(collection, scroll_filter=..., ...)` over stored urls."""

    def __init__(self, stored):
        self.stored = stored
        self.filters = []

    def scroll(self, collection, scroll_filter, limit, with_payload, with_vectors, offset=None):
        cond = scroll_filter.must[0]
        assert cond.key == "url"                       # the indexed field, never a scan
        self.filters.append(set(cond.match.any))
        pts = [type("P", (), {"payload": {"url": u}})() for u in self.stored
               if u in set(cond.match.any)]
        return pts, None


def test_indexed_urls_reads_by_the_url_field_in_every_spelling():
    tw = Twin(["http://www.github.com/ninjahawk/livenerf/", "https://x.example/other"])
    have = HO.indexed_urls(tw, "aw_e8", [GITHUB, VENDOR])
    assert have == {GITHUB} and tw.filters


# --------------------------------------------------------------------- consent
def test_host_verdicts_unanswered_host_is_unknown_never_ok():
    receipts = {"github.com": {"status": "ok", "search_ok": True, "train_ok": False}}
    from cytir.consent import policy
    got = HO.host_verdicts(
        ["github.com", "lwn.net"], lambda hosts: receipts,
        lambda rec: policy.gate_bucket(rec, search_indexing=True))
    assert got == {"github.com": "ok_search_only", "lwn.net": "unknown"}
    assert HO.FETCHABLE_BUCKETS == ("ok", "ok_search_only")


def test_a_ccbot_or_ai_token_decline_is_train_false_only_and_still_fetches():
    from cytir.consent import policy
    rec = {"status": "ok", "search_ok": True, "train_ok": False}
    assert policy.gate_bucket(rec, search_indexing=True) in HO.FETCHABLE_BUCKETS
    assert policy.gate_bucket({"status": "ok", "search_ok": False, "train_ok": False},
                              search_indexing=True) == "declined"
    assert policy.gate_bucket({"status": "unknown", "search_ok": True, "train_ok": True},
                              search_indexing=True) == "unknown"


# ----------------------------------------------------------------------- fetch
def test_a_page_is_sealed_as_bytes_before_it_is_read_and_the_bytes_round_trip(tmp_path):
    f, c, _ = fetcher(tmp_path, ok_page)
    got = f.fetch(VENDOR)
    assert got["terminal"] is None and got["receipt"]["http_status"] == 200
    assert f.PF.load_body(f.root, got["receipt"]) == body(VENDOR)
    assert got["receipt"]["body_sha256"] == MAN[VENDOR]["sha256"]
    assert (f.root / "receipts").is_dir() and c.calls == [VENDOR]


def test_402_is_paywalled_403_is_a_challenge_never_retried(tmp_path):
    f, c, _ = fetcher(tmp_path, lambda u: status(u, 402 if "a1" in u else 403))
    assert f.fetch("https://example.org/a1")["terminal"] == "excluded_paywalled"
    assert f.fetch("https://example.net/a2")["terminal"] == "challenged"
    assert len(c.calls) == 2, "one GET each, no retry past a challenge"


def test_schema_org_not_accessible_for_free_is_paywalled(tmp_path):
    html = ('<html><head><script type="application/ld+json">{"@type":"NewsArticle",'
            '"isAccessibleForFree": false}</script></head><body>' + "word " * 400 + "</body></html>")

    def script(u):
        return httpx.Response(200, content=html.encode(), headers={"content-type": "text/html"},
                              request=httpx.Request("GET", u, headers={"User-Agent": UA}))
    f, _, _ = fetcher(tmp_path, script)
    assert f.fetch("https://example.org/wall")["terminal"] == "excluded_paywalled"


def test_robots_disallow_means_no_request(tmp_path):
    f, c, _ = fetcher(tmp_path, ok_page, robots_ok=lambda rp, u: False)
    assert f.fetch(VENDOR)["terminal"] == "robots_disallowed" and c.calls == []


def test_a_redirect_that_leaves_the_host_is_not_requested(tmp_path):
    def script(u):
        return status(u, 301, location="https://tracker.example/x")
    f, c, _ = fetcher(tmp_path, script)
    got = f.fetch("https://example.org/a")
    assert got["terminal"] == "off_host_redirect" and c.calls == ["https://example.org/a"]


def test_a_same_host_redirect_is_followed_and_www_is_the_same_host(tmp_path):
    def script(u):
        if u == "https://example.org/a":
            return status(u, 301, location="https://www.example.org/b")
        return httpx.Response(
            200, content=body(VENDOR), headers={"content-type": "text/html"},
            request=httpx.Request("GET", u, headers={"User-Agent": UA}))
    f, c, _ = fetcher(tmp_path, script)
    got = f.fetch("https://example.org/a")
    assert got["terminal"] is None and c.calls == ["https://example.org/a",
                                                   "https://www.example.org/b"]


def test_requests_to_one_host_are_spaced_by_the_larger_of_floor_and_crawl_delay(tmp_path):
    times = []
    clock = Clock()

    def script(u):
        times.append(clock())
        return ok_page(VENDOR) if u == VENDOR else status(u, 404)
    f, _, _ = fetcher(tmp_path, script, delay=12.0, clock=clock)
    f.fetch("https://blog.cloudflare.com/x")
    f.fetch("https://blog.cloudflare.com/y")
    f.fetch("https://other.example/z")
    assert times[1] - times[0] >= 12.0 - 1e-9, "the host's Crawl-delay wins over the 5 s floor"
    assert times[2] - times[1] < 1.0, "another host is not delayed by the first"
    assert HO.HOST_INTERVAL_S == 5.0


# ------------------------------------------------------ candidate -> terminal
def run_cand(tmp_path, url, script=None):
    f, _, _ = fetcher(tmp_path, script or ok_page)
    cand = {"url": url, "title": MAN[url]["hn"]["title"], "points": MAN[url]["hn"]["points"],
            "hn_id": MAN[url]["hn"]["objectID"], "published": "2026-10-01"}
    return HO.process_candidate(cand, f, build_rows=build_rows, detect_lang=lang)


@pytest.mark.parametrize("url", [GITHUB, VENDOR, MODEL, LWN])
def test_real_pages_become_rows_with_the_hn_day_as_published(tmp_path, url):
    terminal, rows, _ = run_cand(tmp_path, url)
    assert terminal == "rows" and rows
    assert all(HO.url_key(r["url"]) == HO.url_key(url) for r in rows)
    assert all(HO.host_of(url).endswith(r["domain"].removeprefix("www.")) for r in rows)
    assert all(r["tier"] == "C" and r["trust_tier"] == 0.6 for r in rows)
    assert all(r.get("published") for r in rows)


def test_a_javascript_shell_becomes_one_title_only_record_that_carries_the_url(tmp_path):
    html = body(SHELL).decode()
    assert len(html) < 3000 and HO.is_js_shell(html, [])
    terminal, rows, _ = run_cand(tmp_path, SHELL)
    assert terminal == "title_only" and len(rows) == 1
    r = rows[0]
    assert r["url"] == SHELL and r["title"] == "Extra Big Ass Intelligence"
    assert r["published"] == "2026-10-01" and r["source_extract"] == "hn_story_title"
    assert r["text"].startswith("Extra Big Ass Intelligence")


def test_a_page_with_no_text_and_no_app_root_is_no_passage_not_title_only(tmp_path):
    html = body(FROG).decode()
    terminal, rows, _ = run_cand(tmp_path, FROG)
    assert terminal in ("no_passage", "rows")      # what the extractor finds is the fact recorded
    if terminal == "no_passage":
        assert rows == [] and not HO.is_js_shell(html, [])


def test_a_casino_doorway_page_is_held_out_whole_before_seal(tmp_path):
    spam = ("<html><head><title>Best casino bonus</title></head><body>"
            + "<p>Play online slots at the best casino with free spins, a welcome bonus, "
              "no deposit jackpots, roulette and blackjack for real money. " * 40
            + "</p></body></html>")

    def script(u):
        return httpx.Response(200, content=spam.encode(), headers={"content-type": "text/html"},
                              request=httpx.Request("GET", u, headers={"User-Agent": UA}))
    cand = {"url": "https://doorway.example/p", "title": "Best casino", "points": 150,
            "hn_id": "9", "published": "2026-10-01"}
    f, _, _ = fetcher(tmp_path, script)
    terminal, rows, _ = HO.process_candidate(cand, f, build_rows=build_rows, detect_lang=lang)
    assert terminal in ("spam", "no_passage") and rows == []


def test_accounting_every_candidate_has_exactly_one_terminal():
    outcomes = {"a": "rows", "b": "title_only", "c": "consent_declined", "d": "indexed_already"}
    acct = HO.accounting(outcomes)
    assert acct["candidates"] == 4 and acct["rows"] == 1 and acct["title_only"] == 1
    assert sum(acct[t] for t in HO.TERMINALS) == 4


# ----------------------------------------------------------- the lane's glue
def test_the_switch_is_off_by_default_and_the_unattended_default_excludes_the_route():
    assert HO.enabled({}) is False and HO.enabled({"AW_HN_OUTLINKS": "0"}) is False
    assert HO.enabled({"AW_HN_OUTLINKS": "1"}) is True
    assert HO.PACK_ID not in RC.default_pack_ids({})
    assert HO.PACK_ID in RC.default_pack_ids({"AW_HN_OUTLINKS": "1"})
    assert RC.PACK_META[HO.PACK_ID]["gated_by"] == "AW_HN_OUTLINKS"


def test_naming_the_route_with_the_switch_off_is_refused(monkeypatch, capsys, tmp_path, word_only_chunking_allowed):
    monkeypatch.setattr(RC, "W", tmp_path)
    monkeypatch.delenv("AW_HN_OUTLINKS", raising=False)
    monkeypatch.delenv("AW_LANGGATE_OPTIONAL")      # `main` forbids it; the test sets it for the extractor
    with pytest.raises(SystemExit):
        RC.main(["--pack-id", HO.PACK_ID])
    assert "switched off" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="REFUSED"):
        RC.run_stage(HO.PACK_ID, cadence_path=pathlib.Path("unused"),
                     out_log=pathlib.Path("unused"))


def test_hn_discover_gates_every_host_before_any_page_request(monkeypatch):
    hits = [hit(u) for u in (GITHUB, VENDOR, SHELL, LWN)]
    hits.append({"objectID": "7", "title": "NYT story", "points": 500, "created_at_i": 1790000000,
                 "url": "https://www.nytimes.com/2026/10/01/a.html"})
    hits.append({"objectID": "8", "title": "Self", "points": 500, "created_at_i": 1790000000})
    from cytir.consent import policy
    asked = []

    def resolve(hosts):
        asked.extend(hosts)
        return {"github.com": {"status": "ok", "search_ok": True, "train_ok": True},
                "blog.cloudflare.com": {"status": "ok", "search_ok": True, "train_ok": False},
                "www.extrabigassintelligence.com": {"status": "ok", "search_ok": False,
                                                    "train_ok": False},
                "lwn.net": {"status": "unknown", "search_ok": True, "train_ok": True}}
    got = RC.hn_discover(
        since_days=7, twin_url="unused", collection="aw_e8", now_i=1790000100,
        client=object(), algolia=lambda client, lo, hi, tag: (hits, {"window": [lo, hi]}),
        resolve=resolve, bucket=lambda r: policy.gate_bucket(r, search_indexing=True),
        indexed=lambda urls: set(),
        paywalled=PAYWALLED, declined=lambda h: False)
    assert [c["url"] for c in got["candidates"]] == [GITHUB, VENDOR]
    out = got["hn"]["outcomes"]
    assert out["https://www.nytimes.com/2026/10/01/a.html"] == "excluded_paywalled"
    assert out[SHELL] == "consent_declined" and out[LWN] == "consent_unknown"
    assert "www.nytimes.com" not in asked, "a paywalled host is never even probed"
    assert got["hn"]["stories_in_window"] == 6
