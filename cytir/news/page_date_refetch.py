#!/usr/bin/env python3
# Moved from the retired `news_page_date_refetch` script (task 683 S7 tranche 4); the old path is an alias shim.
"""Re-fetch the news pages the twin holds but no sealed original covers (task 676).

Why. `published_page_date` re-derives `published` from the sealed originals of the
stage spools (ruling 18: the page's own date wins over the sitemap `lastmod`). Rows
that came in through an earlier pack (swissinfo Jul to Sep 2026, before the pages were
sealed) have no original, so their `published` still carries the `lastmod`. This tool
closes that gap: it finds those urls, fetches each once, politely, and SEALS the
response (bytes, transport, request identity: `preserved_page_fetch`) in a spool that
`published_page_date.spool_roster` picks up like any other. The date itself is then
derived by the existing `published-from-pages` verb; this tool never writes the index.

Verbs
-----
  plan    read-only. Roster of spools (globbed, printed), every url the twin holds for
          `--domain`, the ones a spool already captures, the rest written to
          `missing-urls.jsonl`. `twin_urls == captured + missing` or it refuses.
  fetch   one polite GET per missing url into `<work>/stage-refetch-<name>/`:
          consent verdict and robots checked first, >= 1 s between requests, news
          egress under `--max-mbps` (steered on the bytes this run sealed), resumable
          (a valid receipt is never re-fetched). Every url ends in exactly one terminal.

Terminals (fetch): sealed | http_<status> | non_article | robots_disallowed | error.
Only `sealed` writes a `news_documents.jsonl` line. A non-200 response is sealed as
honest evidence of what the publisher returned and produces no document. A transport
failure is `error` and is retried by the next run; it is never read as a dead url.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]   # started by path: the repo root first
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cytir.acquire import preserved_page_fetch as PF  # noqa: E402
from cytir.index import promote_sealed_pack as PSP  # noqa: E402
from cytir.news import published_page_date as PPD  # noqa: E402
from cytir.index import supersede as US  # noqa: E402

SCHEMA = "aw.news-page-date-refetch.v1"
SCROLL_PAGE = 1000
MIN_DELAY_S = 1.0
DEFAULT_MAX_MBPS = 1.0
TERMINALS = ("sealed", "non_article", "robots_disallowed", "error")


def domain_variants(domain: str) -> list[str]:
    bare = domain[4:] if domain.startswith("www.") else domain
    return [bare, "www." + bare]


def twin_urls(twin_url: str, collection: str, domains: list[str], *, api=None) -> dict[str, str]:
    """{canonical url group -> one stored spelling} for every point whose `domain` is
    one of `domains` (and its `www.` spelling). `domain` is an indexed keyword; the
    scroll reads the `url` field only."""
    api = api or PSP._api
    values = sorted({v for d in domains for v in domain_variants(d)})
    out: dict[str, str] = {}
    offset = None
    while True:
        body = {"filter": {"must": [{"key": "domain", "match": {"any": values}}]},
                "limit": SCROLL_PAGE, "with_payload": ["url"], "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        got = api(twin_url, f"/collections/{collection}/points/scroll", body) or {}
        for pt in got.get("points", []):
            url = (pt.get("payload") or {}).get("url") or ""
            if url:
                out.setdefault(US.group_key(url), url)
        offset = got.get("next_page_offset")
        if offset is None:
            return out


def plan(work, twin_url: str, collection: str, domains: list[str], *, api=None) -> dict:
    """The dry run: which twin urls no sealed original covers."""
    work = pathlib.Path(work)
    roster = PPD.spool_roster(work)
    captured_keys = set(PPD.latest_captures(roster))
    held = twin_urls(twin_url, collection, domains, api=api)
    missing = sorted(u for k, u in held.items() if k not in captured_keys)
    captured = len(held) - len(missing)
    if len(held) != captured + len(missing):  # pragma: no cover -- arithmetic guard
        raise RuntimeError("conservation: twin urls != captured + missing")
    return {"schema": SCHEMA, "verb": "plan", "collection": collection,
            "domains": domains, "roster": [p.name for p in roster],
            "roster_size": len(roster), "twin_urls": len(held), "captured": captured,
            "missing": len(missing), "missing_urls": missing}


def spool_dir(work, name: str) -> pathlib.Path:
    return pathlib.Path(work) / f"stage-refetch-{name}"


class Politeness:
    """>= `delay_s` between requests and `max_mbps` on the bytes this run sealed."""

    def __init__(self, delay_s: float, max_mbps: float, *, sleep=time.sleep,
                 clock=time.monotonic):
        self.delay_s = max(delay_s, MIN_DELAY_S)
        self.max_bps = max_mbps * 1_000_000
        self._sleep, self._clock = sleep, clock
        self._next = 0.0
        self._t0 = None
        self.bytes = 0

    def wait(self) -> None:
        now = self._clock()
        if self._t0 is None:
            self._t0 = now
        wait = self._next - now
        if wait > 0:
            self._sleep(wait)
        floor = self.bytes / self.max_bps - (self._clock() - self._t0)  # time still owed
        if floor > 0:
            self._sleep(floor)
        self._next = self._clock() + self.delay_s


def _get(client, url: str, pol: Politeness, tries: int = 3):
    """One GET with back-off on 429/503/5xx. Returns the response or None (transport
    failure / retries exhausted: the caller records `error`, never a verdict)."""
    for attempt in range(tries):
        pol.wait()
        try:
            r = client.get(url)
        except Exception:  # noqa: BLE001 -- recorded as an error terminal
            pol._sleep(min(5 * (attempt + 1), 30))
            continue
        if r.status_code in (429, 503):
            ra = r.headers.get("retry-after", "")
            pol._sleep(min(float(ra) if ra.isdigit() else 30 * (attempt + 1), 120))
            continue
        if 500 <= r.status_code <= 599:
            pol._sleep(2 * (attempt + 1))
            continue
        return r
    return None


def fetch(urls: list[str], spool: pathlib.Path, client, robots_ok, user_agent: str,
          pol: Politeness, *, limit: int = 0, now=None) -> dict:
    """Seal one response per url into `spool/pages`; return the accounting. Resumable."""
    spool = pathlib.Path(spool)
    root = spool / "pages"
    root.mkdir(parents=True, exist_ok=True)
    docs_path = spool / "news_documents.jsonl"
    have_docs = set()
    if docs_path.is_file():
        for line in docs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                have_docs.add(json.loads(line)["url"])
    terminals: collections.Counter = collections.Counter()
    todo = urls[:limit] if limit else list(urls)
    with docs_path.open("a", encoding="utf-8") as docs:
        for url in todo:
            receipt = PF.valid_receipt(root, PF.receipt_path(root, url), url)
            if receipt is None:
                if not robots_ok(url):
                    terminals["robots_disallowed"] += 1
                    continue
                r = _get(client, url, pol)
                if r is None:
                    terminals["error"] += 1
                    continue
                fetched_at = (now or (lambda: time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime())))()
                try:
                    receipt = PF.seal_page_fetch(root, url, r, fetched_at, user_agent)
                    PF.publish_receipt(root, url, receipt)
                except Exception:  # noqa: BLE001 -- unsealable = no evidence = error
                    terminals["error"] += 1
                    continue
                pol.bytes += int(receipt["body_bytes"])
            status = int(receipt["http_status"])
            if not PF.is_admissible_article(receipt):
                terminals["non_article" if status == 200 else f"http_{status}"] += 1
                continue
            terminals["sealed"] += 1
            if url not in have_docs:
                docs.write(json.dumps({
                    "url": url, "fetched_at": receipt["fetched_at"],
                    "lineage": {"original_sha256": receipt["body_sha256"],
                                "source": "news_page_date_refetch",
                                "receipt": PF.object_id(url)}}) + "\n")
                have_docs.add(url)
    done = sum(terminals.values())
    if done != len(todo):  # pragma: no cover -- arithmetic guard
        raise RuntimeError(f"conservation: {done} terminals for {len(todo)} urls")
    return {"schema": SCHEMA, "verb": "fetch", "urls": len(todo),
            "terminals": dict(terminals), "sealed_bytes": pol.bytes,
            "spool": str(spool)}


def _live(args):  # pragma: no cover -- network
    import httpx
    from cytir.acquire import sitemap_ingest as SI
    from cytir.consent.resolver import resolve_hosts
    host = "www." + args.domain.removeprefix("www.")
    cres = resolve_hosts([host], cache_dir=os.path.expanduser(args.cache_dir),
                         progress=False, direct_crawl=True)
    verdict = SI._search_index_bucket(cres[host])
    if verdict not in {"ok", "ok_search_only"}:
        raise SystemExit(f"REFUSING: consent verdict for {host} is {verdict!r}")
    rp, delay = SI.load_robots(host, os.path.expanduser(args.cache_dir))
    from cytir.sources.family import HttpSpec
    from cytir.sources.http import make_client
    client = make_client(HttpSpec(      # consent-guarded (task 1621); the discovery UA, as before
        timeout=httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=60.0),
        headers={"user-agent": SI.UA["User-Agent"]}, follow_redirects=True))
    return client, (lambda u: SI.robots_ok(rp, u)), SI.UA["User-Agent"], delay


def main(argv=None, prog: str | None = None) -> int:  # pragma: no cover -- network
    ap = argparse.ArgumentParser(prog=prog or "page_date_refetch", description=__doc__.split("\n\n")[0])
    ap.add_argument("verb", choices=("plan", "fetch"))
    ap.add_argument("--work", default=None,
                    help="news work tree (default: the operator setting news_work)")
    ap.add_argument("--twin-url", required=True)
    ap.add_argument("--twin-collection", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--name", required=True, help="spool name: stage-refetch-<name>")
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--delay", type=float, default=MIN_DELAY_S)
    ap.add_argument("--max-mbps", type=float, default=DEFAULT_MAX_MBPS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache-dir", default=None)
    a = ap.parse_args(argv)
    a.work = a.work or PPD.default_work()
    if a.cache_dir is None:
        from cytir.consent import resolver as cr
        a.cache_dir = cr.CACHE_DIR
    state = pathlib.Path(a.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    urls_path = state / "missing-urls.jsonl"
    if a.verb == "plan" or not urls_path.is_file():
        p = plan(a.work, a.twin_url, a.twin_collection, [a.domain])
        urls_path.write_text("".join(json.dumps({"url": u}) + "\n" for u in p["missing_urls"]))
        summary = {k: v for k, v in p.items() if k != "missing_urls"}
        summary["planned_at"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        (state / "plan.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary), flush=True)
        if a.verb == "plan":
            return 0
    urls = [json.loads(line)["url"] for line in urls_path.read_text().splitlines() if line.strip()]
    client, robots_ok, ua, robots_delay = _live(a)
    pol = Politeness(max(a.delay, robots_delay), a.max_mbps)
    out = fetch(urls, spool_dir(a.work, a.name), client, robots_ok, ua, pol, limit=a.limit)
    (state / "fetch-result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)
    return 0 if out["terminals"].get("error", 0) == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
