#!/usr/bin/env python3
# Moved from the pipeline-era `rts_section_discovery` module (task 683 S7 tranche 4); the old path was retired.
"""RTS article discovery by section pagination (task 676, plan.md ruling 18).

WHY
---
rts.ch publishes no article archive: its only sitemaps list section and hub pages
(609 urls, no `lastmod`; task 676 section 3), and Common Crawl holds none of it.
Tim 2026-09-30 (ruling 18): the 12-month backfill is done by section pagination at the
host's `Crawl-delay: 10`, not forward-only.

HOW THE SITE PAGINATES (read from the live site, 2026-09-30)
-----------------------------------------------------------
Every section hub (`/info/suisse/`) carries a "load more" control,
`data-loadmore-src="/info/suisse/page/1"`. Fetching that path returns the next cards;
the page names its own successor: `data-loadmore-src="/info/suisse/page/61?lastDate=
2026-01-05"`, where `lastDate` is the ISO day of the last card on the page. Page 1 of
/info/suisse is today, page 60 is 9 to 5 January 2026, page 400 is August 2022. A card
is `<a href=ARTICLE> ... <p class="card-time">le 9 janvier 2026</p> </a>`; about 24 dated
cards a page (plus a static strip of podcast and app links WITHOUT a `card-time`, which
is never an article of the section).

WHAT THIS MODULE DOES
---------------------
* derives the section list from the site's own sitemaps (never typed): every hub of
  `sitemaps/pages/info.xml` and `sport.xml`, minus dossiers. `/info/page/N` is a curated
  front-page feed (about 24 cards for 1.5 days today, 6 days in early September), not the
  union of the sections, and a child hub (`/info/culture/cinema/`) is not known to be a
  subset of its parent, so every hub is paginated and urls are de-duplicated;
* reads each hub once for its `data-loadmore-src` (a hub without one is reported, not
  guessed) and follows the successor links until the MEDIAN day of a page's own cards passes the window
  start, the site stops giving a successor, or the section's pagination itself stalls
  (`STALE_PAGES` consecutive pages with no article card, or a successor that points at the
  page just read). A page of already-known urls is NOT a stall: the first pages of a hub
  overlap the `/info/` front feed processed first (2026-10-03: `/info/suisse/page/30`,
  late May, had 4 of 24 articles known), and counting only NEW urls ended four hubs
  after 3 pages in September; The SPORT hubs load more through `/hbv7/ajax/...`, which robots.txt
  disallows (`Disallow: /hbv7/`): that is the publisher's statement, the section ends
  `robots_disallowed` and only the cards on the hub page itself are taken;
* one request at a time at the host's crawl delay (robots `Crawl-delay`, never faster
  than `MIN_DELAY_S`); robots `Disallow` and the consent verdict are checked before the
  first request; an HTTP failure is an ERROR terminal for that section (resumable), never
  an empty section;
* writes `urls.jsonl` (one line per article url, first seen wins) and `state.json`
  (per section: status, pages, cursor, counts) so a restart resumes where it stopped.
  Every section ends in exactly one of: complete, window_reached, stale, error,
  no_loadmore, robots_disallowed.

Only article-shaped urls (`...<6+ digits>.html`) of DATED cards are taken; the article's
own date is read from the page at fetch time (ruling 18), the card date only buckets the
url into a run.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html as _html
import json
import os
import pathlib
import re
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]   # started by path: the repo root first
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

HOST = "www.rts.ch"
ORIGIN = f"https://{HOST}"
SITEMAP_FILES = ("info", "sport")
EXCLUDED_SECTION_PARTS = ("dossiers",)
MIN_DELAY_S = 10.0
STALE_PAGES = 3
MAX_PAGES_PER_SECTION = 600
MAX_CONSECUTIVE_ERRORS = 3
WINDOW_MONTHS = 12
STATE_SCHEMA = "aw.rts-section-discovery.v3"
#: v2 judged the window by the successor link's `lastDate`, which the site fills from ONE outlier card per page
#: (a card dated 2024-02-06 under a page of 2-5 September 2026): 24 hubs ended `window_reached` after 9 pages.
STATE_SCHEMA_V2 = "aw.rts-section-discovery.v2"
#: v1 ended a section `stale` after 3 pages without NEW urls (overlap with the front feed).
STATE_SCHEMA_V1 = "aw.rts-section-discovery.v1"
#: Fetcher status for a url robots.txt disallows (no request was made).
ROBOTS_DISALLOWED = -1
TERMINALS = ("complete", "window_reached", "stale", "error", "no_loadmore",
             "robots_disallowed")

_CARD = re.compile(r'<a\b[^>]*href="([^"]+)"[^>]*>(?:(?!</a>).)*?card-time">([^<]+)<', re.S)
_NEXT = re.compile(r'data-loadmore-src="([^"]+)"')
_LASTDATE = re.compile(r"[?&]lastDate=(\d{4}-\d{2}-\d{2})")
#: article-shaped: the last path segment carries a 6+ digit id, at the end
#: (`...-29113505.html`), at the start (`13341000-slug.html`, 2022) or alone
#: (`27415950.html`)
_ARTICLE = re.compile(r"(?:^|[/-])\d{6,}(?:-[^/]*)?\.html$")
_LOC = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")
_FR_MONTHS = {"janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5,
              "juin": 6, "juillet": 7, "août": 8, "aout": 8, "septembre": 9,
              "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12}
_FR_WEEKDAYS = {"lundi": 0, "mardi": 1, "mercredi": 2, "jeudi": 3, "vendredi": 4,
                "samedi": 5, "dimanche": 6}


# ------------------------------------------------------------------- parsing
def parse_card_date(text: str, today: _dt.date) -> _dt.date | None:
    """The day a card names: `le 9 janvier 2026`, `aujourd'hui à 11:50`, `hier à 18:25`
    or a weekday within the last week (`dimanche à 15:30`). None when it names none."""
    t = _html.unescape(text or "").strip().lower()
    m = re.match(r"le\s+(\d{1,2})(?:er)?\s+([a-zéèêûô]+)\s+(\d{4})", t)
    if m and m[2] in _FR_MONTHS:
        try:
            return _dt.date(int(m[3]), _FR_MONTHS[m[2]], int(m[1]))
        except ValueError:
            return None
    if t.startswith("aujourd"):
        return today
    if t.startswith("hier"):
        return today - _dt.timedelta(days=1)
    first = t.split()[0] if t.split() else ""
    if first in _FR_WEEKDAYS:
        back = (today.weekday() - _FR_WEEKDAYS[first]) % 7 or 7
        return today - _dt.timedelta(days=back)
    return None


def parse_page(html: str, today: _dt.date) -> dict:
    """{cards: [{url, date}], next: path|None, last_date: date|None}. Only dated cards
    whose href is an article-shaped rts.ch url are returned, each url once per page."""
    cards, seen = [], set()
    for href, when in _CARD.findall(html):
        url = _html.unescape(href).split("#")[0]
        if url.startswith("/"):
            url = ORIGIN + url
        if not (url.startswith(ORIGIN + "/") and _ARTICLE.search(url.split("?")[0])):
            continue
        if url in seen:
            continue
        seen.add(url)
        cards.append({"url": url, "date": parse_card_date(when, today)})
    nxt = _NEXT.findall(html)
    nxt = _html.unescape(nxt[-1]) if nxt else None
    last = _LASTDATE.search(nxt or "")
    return {"cards": cards, "next": nxt,
            "last_date": _dt.date.fromisoformat(last[1]) if last else None}


def hub_loadmore(html: str) -> str | None:
    found = _NEXT.findall(html)
    return _html.unescape(found[0]) if found else None


def sections_from_sitemaps(files: dict[str, str]) -> list[str]:
    """Every hub path (`/info/suisse/`) of the given sitemap bodies, dossiers excluded,
    shallowest first."""
    hubs = set()
    for body in files.values():
        for loc in _LOC.findall(body):
            if not loc.startswith(ORIGIN + "/"):
                continue
            path = loc[len(ORIGIN):]
            parts = [p for p in path.split("/") if p]
            if not parts or any(p in EXCLUDED_SECTION_PARTS for p in parts):
                continue
            if not path.endswith("/") or _ARTICLE.search(path.rstrip("/")):
                continue
            hubs.add("/" + "/".join(parts) + "/")
    return sorted(hubs, key=lambda h: (h.count("/"), h))


# --------------------------------------------------------------------- state
def load_state(path: pathlib.Path) -> dict:
    if path.is_file():
        state = json.loads(path.read_text())
        if state.get("schema") != STATE_SCHEMA:
            # v1 `stale` meant "3 pages without NEW urls", which overlap alone could cause:
            # reopen those sections at their saved cursor under the v2 rule.
            for entry in state.get("sections", {}).values():
                # v1 `stale` and v1/v2 `window_reached` (the outlier-card stop) are re-judged
                if entry.get("status") == "stale" or (
                        entry.get("status") == "window_reached"
                        and state.get("schema") in (STATE_SCHEMA_V1, STATE_SCHEMA_V2, None)):
                    entry.pop("status")
                    entry.pop("note", None)
                    entry["stale_run"] = 0
                    entry["reopened_from_v1"] = True
            state["schema"] = STATE_SCHEMA
        return state
    return {"schema": STATE_SCHEMA, "sections": {}}


def save_state(path: pathlib.Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def window_start(today: _dt.date, months: int = WINDOW_MONTHS) -> _dt.date:
    y, m = divmod(today.year * 12 + today.month - 1 - months, 12)
    return _dt.date(y, m + 1, min(today.day, 28))


# ------------------------------------------------------------------- crawling
def _body_text(response) -> str:
    """The response text; a body still gzip-wrapped after the transport layer was removed
    (rts.ch serves some sitemaps nested up to four layers, at random: the same url was
    readable XML on one request and gzip bytes on the next, 2026-09-30) is unwrapped by
    the bounded decoder `sitemap_ingest` already uses."""
    raw = getattr(response, "content", b"") or b""
    if raw[:2] == b"\x1f\x8b":
        from cytir.acquire.sitemap_ingest import _decode_sitemap_bytes
        return _decode_sitemap_bytes(raw) or ""
    return response.text


class Fetcher:
    """One polite GET at a time: robots `Disallow` checked, `delay_s` between requests,
    429/503 backed off. Returns (status, text); a transport failure is status 0. Never
    raises on a network error: the caller records it as an error terminal."""

    def __init__(self, client, robots_ok, delay_s: float, sleep=time.sleep,
                 clock=time.monotonic):
        self.client, self.robots_ok = client, robots_ok
        self.delay_s = max(delay_s, MIN_DELAY_S)
        self._sleep, self._clock = sleep, clock
        self._next = 0.0
        self.requests = 0

    def get(self, path_or_url: str, tries: int = 3) -> tuple[int, str]:
        url = path_or_url if path_or_url.startswith("http") else ORIGIN + path_or_url
        if not self.robots_ok(url):
            return ROBOTS_DISALLOWED, ""
        status, text = 0, ""
        for attempt in range(tries):
            wait = self._next - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._next = self._clock() + self.delay_s
            self.requests += 1
            try:
                r = self.client.get(url)
            except Exception:  # noqa: BLE001 -- recorded as status 0
                status, text = 0, ""
                self._sleep(min(5 * (attempt + 1), 30))
                continue
            status, text = r.status_code, _body_text(r)
            if status in (429, 503):
                ra = r.headers.get("retry-after", "")
                self._sleep(min(float(ra) if ra.isdigit() else 30 * (attempt + 1), 120))
                continue
            break
        return status, text


def discover_section(section: str, fetcher: Fetcher, *, today: _dt.date,
                     start: _dt.date, urls_out: pathlib.Path, seen: set,
                     entry: dict, save=lambda: None) -> dict:
    """Follow one section's pagination. Mutates and returns `entry` (the state row);
    appends new urls to `urls_out`. Resumes from `entry["cursor"]` when present."""
    entry.setdefault("pages", 0)
    entry.setdefault("urls_new", 0)
    entry.setdefault("stale_run", 0)
    entry.setdefault("errors", 0)
    if not entry.get("cursor"):
        status, html = fetcher.get(section)
        if status == ROBOTS_DISALLOWED:
            entry.update(status="robots_disallowed", note="the hub itself is disallowed")
            return entry
        if status != 200:
            entry.update(status="error", note=f"hub http {status}")
            return entry
        nxt = hub_loadmore(html)
        page = parse_page(html, today)
        _take(page, section, urls_out, seen, entry)
        if not nxt:
            entry.update(status="no_loadmore", note="hub has no data-loadmore-src")
            return entry
        if not fetcher.robots_ok(ORIGIN + nxt.split("#")[0]):
            entry.update(status="robots_disallowed",
                         note=f"robots.txt disallows the load-more path {nxt.split('?')[0]}")
            return entry
        entry["cursor"] = nxt
    errors = 0
    while entry.get("status") not in TERMINALS:
        if entry["pages"] >= MAX_PAGES_PER_SECTION:
            entry.update(status="error", note="page cap reached before the window start")
            break
        status, html = fetcher.get(entry["cursor"])
        if status != 200:
            errors += 1
            entry["errors"] += 1
            if errors >= MAX_CONSECUTIVE_ERRORS:
                entry.update(status="error", note=f"http {status} x{errors} at {entry['cursor']}")
                break
            continue
        errors = 0
        page = parse_page(html, today)
        entry["pages"] += 1
        _take(page, section, urls_out, seen, entry)
        # a stall is the PAGINATION giving nothing (no article card, or the same successor
        # again), never "no new urls": known urls from the front feed are normal here
        stalled = not page["cards"] or page["next"] == entry["cursor"]
        entry["stale_run"] = entry["stale_run"] + 1 if stalled else 0
        days = sorted(c["date"] for c in page["cards"] if c["date"])
        edge = days[len(days) // 2] if days else page["last_date"]
        entry["last_date"] = edge.isoformat() if edge else None
        if page["next"] and not fetcher.robots_ok(ORIGIN + page["next"].split("#")[0]):
            entry.update(status="robots_disallowed",
                         note=f"robots.txt disallows {page['next'].split('?')[0]}")
        elif not page["next"]:
            entry["status"] = "complete"
        elif _before_window(page, start):
            entry["status"] = "window_reached"
        elif entry["stale_run"] >= STALE_PAGES:
            entry["status"] = "stale"
        else:
            entry["cursor"] = page["next"]
        save()
    return entry


def _before_window(page: dict, start: _dt.date) -> bool:
    """The page's own evidence that the window is behind us: the MEDIAN day of its dated
    cards is older than the window start. The successor link's `lastDate` is not evidence
    (the site fills it from one outlier card per page, and ignores it when serving the next
    page); a single odd card must neither stop nor extend a section. No dated card: the
    link's day is the only evidence left."""
    days = sorted(c["date"] for c in page["cards"] if c["date"])
    if days:
        return days[len(days) // 2] < start
    return bool(page["last_date"]) and page["last_date"] < start


def _take(page: dict, section: str, urls_out: pathlib.Path, seen: set, entry: dict) -> int:
    new = 0
    with urls_out.open("a", encoding="utf-8") as fh:
        for card in page["cards"]:
            if card["url"] in seen:
                continue
            seen.add(card["url"])
            new += 1
            fh.write(json.dumps({
                "url": card["url"], "section": section,
                "card_date": card["date"].isoformat() if card["date"] else None,
                "cursor_date": page["last_date"].isoformat() if page["last_date"] else None,
            }, sort_keys=True) + "\n")
    entry["urls_new"] = entry.get("urls_new", 0) + new
    return new


def load_seen(urls_out: pathlib.Path) -> set:
    if not urls_out.is_file():
        return set()
    return {json.loads(line)["url"] for line in urls_out.open(encoding="utf-8") if line.strip()}


def discover(state_dir: pathlib.Path, fetcher: Fetcher, sections: list[str], *,
             today: _dt.date, start: _dt.date, known: set | None = None) -> dict:
    """Run every section to a terminal (or stop at the first fetcher-fatal), saving
    state after every page. Returns the state. `known` = urls an EARLIER pass already
    found (and fetched): they are never written again, so `urls.jsonl` of a second
    pass holds only what the first pass missed."""
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path, urls_out = state_dir / "state.json", state_dir / "urls.jsonl"
    state = load_state(state_path)
    state.update(today=today.isoformat(), window_start=start.isoformat(),
                 sections_planned=sections)
    seen = load_seen(urls_out) | set(known or ())
    for section in sections:
        entry = state["sections"].setdefault(section, {})
        if entry.get("status") in TERMINALS and entry.get("status") != "error":
            continue
        entry.pop("status", None)
        entry.pop("note", None)
        discover_section(section, fetcher, today=today, start=start, urls_out=urls_out,
                         seen=seen, entry=entry, save=lambda: save_state(state_path, state))
        save_state(state_path, state)
    state["urls_total"] = len(seen) - len(known or ())
    state["requests"] = state.get("requests", 0) + fetcher.requests
    save_state(state_path, state)
    return state


def accounting(state: dict) -> dict:
    """Every planned section has exactly one terminal, or is reported as open."""
    terminals = set(TERMINALS)
    out = {k: 0 for k in sorted(terminals)}
    out["open"] = 0
    for s in state.get("sections_planned", []):
        st = (state["sections"].get(s) or {}).get("status")
        out[st if st in terminals else "open"] += 1
    out["sections"] = len(state.get("sections_planned", []))
    return out


# ------------------------------------------------------------------------ CLI
def _live(args) -> tuple[Fetcher, list[str]]:  # pragma: no cover -- network
    import httpx
    from cytir.acquire import sitemap_ingest as SI
    from cytir.consent.resolver import resolve_hosts
    cres = resolve_hosts([HOST], cache_dir=os.path.expanduser(args.cache_dir),
                         progress=False, direct_crawl=True)
    verdict = SI._search_index_bucket(cres[HOST])
    if verdict not in {"ok", "ok_search_only"}:
        raise SystemExit(f"REFUSING: consent verdict for {HOST} is {verdict!r}")
    rp, delay = SI.load_robots(HOST, os.path.expanduser(args.cache_dir))
    from cytir.sources.family import HttpSpec
    from cytir.sources.http import make_client
    client = make_client(HttpSpec(      # consent-guarded (task 1621); the discovery UA, as before
        timeout=httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=60.0),
        headers={"user-agent": SI.UA["User-Agent"]}, follow_redirects=True))
    fetcher = Fetcher(client, lambda u: SI.robots_ok(rp, u), delay)
    files = {}
    for name in SITEMAP_FILES:
        status, text = fetcher.get(f"/sitemaps/pages/{name}.xml")
        if status != 200:
            raise SystemExit(f"REFUSING: sitemap {name}.xml http {status}")
        files[name] = text
    return fetcher, sections_from_sitemaps(files)


def run_live(state_dir: pathlib.Path, *, today: _dt.date, only=(), max_pages: int = 0,
             months: int = WINDOW_MONTHS, cache_dir: str | None = None,
             known_urls: str | None = None) -> int:  # pragma: no cover
    """Discovery against the live site (consent verdict and robots checked first)."""
    global MAX_PAGES_PER_SECTION
    if cache_dir is None:
        from cytir.consent import resolver as cr
        cache_dir = cr.CACHE_DIR
    fetcher, sections = _live(argparse.Namespace(cache_dir=cache_dir))
    if only:
        sections = [s for s in sections if s in set(only)]
    if max_pages:
        MAX_PAGES_PER_SECTION = max_pages
    print(f"sections ({len(sections)}): {sections}", flush=True)
    known = load_seen(pathlib.Path(known_urls)) if known_urls else None
    state = discover(pathlib.Path(state_dir), fetcher, sections, today=today,
                     start=window_start(today, months), known=known)
    acct = accounting(state)
    print(json.dumps({"accounting": acct, "urls_total": state["urls_total"]}), flush=True)
    return 0 if acct["error"] == 0 and acct["open"] == 0 else 1


def main(argv=None) -> int:  # pragma: no cover -- network
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--today", default=_dt.datetime.now(_dt.timezone.utc).date().isoformat())
    ap.add_argument("--window-months", type=int, default=WINDOW_MONTHS)
    ap.add_argument("--only", default="", help="comma list of section paths (smoke)")
    ap.add_argument("--max-pages", type=int, default=0, help="smoke: stop a section after N pages")
    ap.add_argument("--known-urls", default=None,
                    help="urls.jsonl of an earlier pass: its urls are not written again")
    a = ap.parse_args(argv)
    return run_live(pathlib.Path(a.state_dir), today=_dt.date.fromisoformat(a.today),
                    only=[x for x in a.only.split(",") if x], max_pages=a.max_pages,
                    months=a.window_months, cache_dir=a.cache_dir, known_urls=a.known_urls)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
