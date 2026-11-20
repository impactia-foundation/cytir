"""Hacker News linked pages as ONE discovery source of the refresh lane (task 697 F1, ruling 23.1).

WHY
---
The HN discussion arm packs the THREAD (ADR-108) and copies the story's outlink verbatim; no lane
fetched the page the story links to. Measured in task 697: for stories with >= 100 points
08-28 .. 09-30 the linked page was in the index for 2 % of them, the thread for 94 %. Tim
(ruling 23.1) approved that the daily refresh lane takes those pages, each host gated by live
consent. The ADR text (ADR-098 Ruling 2 amendment) becomes law when Tim merges it, so the route
ships behind a switch that is OFF unless the environment says `AW_HN_OUTLINKS=1`.

WHAT THIS MODULE IS
-------------------
The pure and checkable parts of the route; the lane (`direct_compose`) calls them:

* `story_candidates`   Algolia story hits -> one candidate per outlink url (strongest story wins);
* `screen`             drops the hn host itself, non-http(s) urls, media files, paywalled hosts
                       (`cytir.corpus.source_cards.PAYWALLED`, curated) and hosts on the recorded
                       consent-declined list (`cytir.consent.block`), each counted by reason;
* `indexed_urls`       urls the twin already holds (exact `url` filter, an indexed field; http/https,
                       www and trailing-slash spellings), so a page is never re-staged;
* `host_verdicts`      live consent per host, from the crawl vantage (the resolver, direct-crawl):
                       only `ok` / `ok_search_only` fetch (search_ok is the gate; a CCBot or
                       AI-token decline is train_ok=false only); `unknown` (incl. a 403 on robots
                       from both vantages: unreadable, not declined) and `declined` do not;
* `LinkedFetcher`      one polite GET per url: per-host spacing (the host's Crawl-delay, never
                       below `HOST_INTERVAL_S`), robots `Disallow` per url, same-host redirects only
                       (a hop that leaves the host is NOT requested), the response SEALED as bytes +
                       transport + request identity (`cytir.acquire.preserved_page_fetch`) before it
                       is read; 402 and schema.org `isAccessibleForFree: false` = paywalled (no row);
                       403 = challenge, never routed around (no row);
* `is_js_shell` / `title_only_row`  a page that is a JavaScript shell (the extractor finds no
                       passage, the page names an app root) becomes ONE title-only record that
                       carries the url, the HN title and the HN submission date;
* `spam_screen`        `cytir.ops.spam_rows` over the rows of a page, before anything is sealed.

Every candidate ends in exactly one terminal (`TERMINALS`); `accounting()` proves it.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import re
import time
from urllib.parse import urljoin, urlsplit

SWITCH_ENV = "AW_HN_OUTLINKS"
PACK_ID = "hn-linked-pages"
LABEL = "hn-linked"
HN_HOST = "news.ycombinator.com"
DEFAULT_MIN_POINTS = 100
DEFAULT_SINCE_DAYS = 7
#: spacing between two requests to one host; a longer robots Crawl-delay wins
HOST_INTERVAL_S = 5.0
MAX_REDIRECTS = 3
FETCHABLE_BUCKETS = ("ok", "ok_search_only")
#: HN tier C community endorsement, as the old linked-page arm carried it (trust 0.6)
META = {"label": LABEL, "tier": "C", "source_type": "reference", "trust_tier": 0.6}
SKIP_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".mp4", ".webm", ".mov", ".mp3",
            ".zip", ".gz", ".tar", ".tgz", ".7z", ".dmg", ".exe", ".iso", ".pdf")
TERMINALS = (
    "indexed_already",        # the twin holds this url
    "excluded_self_or_media",  # hn host, non-http(s), media/archive file
    "excluded_paywalled",     # curated paywall host, 402, or isAccessibleForFree false
    "excluded_declined_host",  # on the recorded consent-declined list
    "consent_declined",       # live verdict: search_ok false
    "consent_unknown",        # live verdict unresolved (incl. unreadable robots)
    "robots_disallowed",      # the url's own path is disallowed
    "challenged",             # 403 access-control challenge: never routed around
    "off_host_redirect",      # a redirect left the host: not requested
    "fetch_error",            # transport / http error (named, resumable next day)
    "non_html",               # 200 but not html
    "no_passage",             # page fetched, extractor found nothing and it is no JS shell
    "spam",                   # cytir.ops.spam_rows excluded every row
    "rows",                   # at least one row staged
    "title_only",             # JS shell: one title-only row
)

_ROOT_MARKERS = re.compile(
    r'<div[^>]+id=["\'](?:root|__next|app|__nuxt|svelte)["\']|<noscript[^>]*>[^<]*(?:enable|activez|aktivieren|javascript)',
    re.IGNORECASE)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_PAYWALL_LD = re.compile(r'"isAccessibleForFree"\s*:\s*(?:false|"false"|"False")', re.IGNORECASE)


def enabled(env=None) -> bool:
    """The route is OFF unless AW_HN_OUTLINKS=1."""
    return (os.environ if env is None else env).get(SWITCH_ENV, "") == "1"


# ------------------------------------------------------------------ candidates
def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _bare(host: str) -> str:
    return host[4:] if host.startswith("www.") else host


def url_key(url: str) -> str:
    p = urlsplit(url)
    return f"{_bare((p.hostname or '').lower())}{p.path.rstrip('/')}" + (f"?{p.query}" if p.query else "")


def url_variants(url: str) -> list[str]:
    """http/https x www/no-www x trailing slash spellings of one url, original first."""
    p = urlsplit(url)
    host = _bare((p.hostname or "").lower())
    path = p.path.rstrip("/")
    out = [url]
    for scheme in ("https", "http"):
        for h in (host, "www." + host):
            for tail in (path, path + "/"):
                v = f"{scheme}://{h}{tail}" + (f"?{p.query}" if p.query else "")
                if v not in out:
                    out.append(v)
    return out


def story_candidates(hits: list[dict], *, min_points: int = DEFAULT_MIN_POINTS) -> list[dict]:
    """One candidate per outlink: {url, title, points, hn_id, published (HN submission day)}.
    Self-posts (no url) are not candidates; a url submitted twice keeps the stronger story."""
    best: dict[str, dict] = {}
    for h in hits:
        url = (h.get("url") or "").split("#")[0].strip()
        pts = h.get("points")
        if not url or not isinstance(pts, int) or pts < min_points:
            continue
        created = h.get("created_at_i")
        cand = {"url": url, "title": (h.get("title") or "").strip(), "points": pts,
                "hn_id": str(h.get("objectID")),
                "published": (_dt.datetime.fromtimestamp(created, _dt.timezone.utc).date().isoformat()
                              if isinstance(created, int) else None)}
        key = url_key(url)
        if key not in best or pts > best[key]["points"]:
            best[key] = cand
    return sorted(best.values(), key=lambda c: (-c["points"], c["url"]))


def paywalled_host(host: str, paywalled) -> bool:
    host = _bare(host)
    return any(host == d or host.endswith("." + d) for d in paywalled)


def screen(cands: list[dict], *, paywalled, declined=lambda host: False) -> tuple[list[dict], dict]:
    """(kept, dropped {terminal: [url, ...]}) by the host/url rules that need no request."""
    kept, dropped = [], {"excluded_self_or_media": [], "excluded_paywalled": [],
                         "excluded_declined_host": []}
    for c in cands:
        p = urlsplit(c["url"])
        host = (p.hostname or "").lower()
        if (p.scheme not in ("http", "https") or not host or _bare(host) == HN_HOST
                or p.path.lower().endswith(SKIP_EXT)):
            dropped["excluded_self_or_media"].append(c["url"])
        elif paywalled_host(host, paywalled):
            dropped["excluded_paywalled"].append(c["url"])
        elif declined(host):
            dropped["excluded_declined_host"].append(c["url"])
        else:
            kept.append(c)
    return kept, dropped


def indexed_urls(client, collection: str, urls: list[str], *, batch: int = 40) -> set[str]:
    """The given urls the collection already holds (any spelling). Reads by the indexed `url`
    field only, never a scan."""
    from qdrant_client import models as qm  # noqa: PLC0415
    spelled: dict[str, str] = {}
    for u in urls:
        for v in url_variants(u):
            spelled.setdefault(v, u)
    allv = sorted(spelled)
    have: set[str] = set()
    for i in range(0, len(allv), batch * 8):
        chunk = allv[i:i + batch * 8]
        flt = qm.Filter(must=[qm.FieldCondition(key="url", match=qm.MatchAny(any=chunk))])
        offset = None
        while True:
            pts, offset = client.scroll(collection, scroll_filter=flt, limit=256,
                                        with_payload=["url"], with_vectors=False, offset=offset)
            for pt in pts:
                u = (pt.payload or {}).get("url")
                if u in spelled:
                    have.add(spelled[u])
            if offset is None:
                break
    return have


# --------------------------------------------------------------------- consent
def host_verdicts(hosts, resolve, bucket) -> dict[str, str]:
    """{host: bucket} from the live resolver (`resolve(hosts) -> {host: receipt}`, run from the
    crawl vantage). A host the resolver does not answer for is `unknown`, never `ok`."""
    receipts = resolve(sorted(set(hosts))) or {}
    return {h: (bucket(receipts[h]) if h in receipts else "unknown") for h in sorted(set(hosts))}


# ----------------------------------------------------------------------- fetch
class LinkedFetcher:
    """One polite sealed GET per url. `client.get(url, follow_redirects=False)` must return an
    httpx-like response; `robots(host) -> (rp|None, crawl_delay)`; `robots_ok(rp, url)`."""

    def __init__(self, client, *, preserve_root, ua: str, robots, robots_ok, seal=None,
                 sleep=time.sleep, clock=time.monotonic, interval_s: float = HOST_INTERVAL_S):
        from cytir.acquire import preserved_page_fetch as PF  # noqa: PLC0415
        self.client, self.root, self.ua = client, pathlib.Path(preserve_root), ua
        self.robots, self.robots_ok = robots, robots_ok
        self.PF = PF
        self.seal = seal or PF.seal_page_fetch
        self._sleep, self._clock, self.interval_s = sleep, clock, interval_s
        self._next: dict[str, float] = {}
        self._robots_cache: dict[str, tuple] = {}
        self.requests = 0

    def _wait(self, host: str, delay: float) -> None:
        wait = self._next.get(host, 0.0) - self._clock()
        if wait > 0:
            self._sleep(wait)
        self._next[host] = self._clock() + max(self.interval_s, delay)

    def fetch(self, url: str) -> dict:
        """{terminal|None, html, receipt, note}: `terminal` is None for a readable 200 html page."""
        host = host_of(url)
        if host not in self._robots_cache:
            self._robots_cache[host] = self.robots(host)
        rp, delay = self._robots_cache[host]
        current = url
        for hop in range(MAX_REDIRECTS + 1):
            if not self.robots_ok(rp, current):
                return {"terminal": "robots_disallowed", "html": None, "receipt": None,
                        "note": current}
            self._wait(host, delay)
            self.requests += 1
            try:
                r = self.client.get(current, follow_redirects=False)
            except Exception as exc:  # noqa: BLE001 -- named, never swallowed
                return {"terminal": "fetch_error", "html": None, "receipt": None,
                        "note": f"{type(exc).__name__}: {exc}"}
            if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                nxt = urljoin(current, r.headers["location"])
                if _bare(host_of(nxt)) != _bare(host):
                    return {"terminal": "off_host_redirect", "html": None, "receipt": None,
                            "note": nxt}
                current = nxt
                continue
            break
        else:
            return {"terminal": "fetch_error", "html": None, "receipt": None,
                    "note": f"more than {MAX_REDIRECTS} redirects"}
        if r.status_code == 402:
            return {"terminal": "excluded_paywalled", "html": None, "receipt": None,
                    "note": "http 402"}
        if r.status_code == 403:
            return {"terminal": "challenged", "html": None, "receipt": None, "note": "http 403"}
        if r.status_code != 200:
            return {"terminal": "fetch_error", "html": None, "receipt": None,
                    "note": f"http {r.status_code}"}
        fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            receipt = self.seal(self.root, url, r, fetched_at, self.ua)
            self.PF.publish_receipt(self.root, url, receipt)
        except Exception as exc:  # noqa: BLE001 -- no sealed bytes, no row
            return {"terminal": "fetch_error", "html": None, "receipt": None,
                    "note": f"unsealed: {type(exc).__name__}: {exc}"}
        if not self.PF.is_admissible_article(receipt):
            return {"terminal": "non_html", "html": None, "receipt": receipt, "note": ""}
        html = self.PF.decode_body(self.PF.load_body(self.root, receipt),
                                   receipt.get("content_type"))
        if _PAYWALL_LD.search(html):
            return {"terminal": "excluded_paywalled", "html": None, "receipt": receipt,
                    "note": "schema.org isAccessibleForFree false"}
        return {"terminal": None, "html": html, "receipt": receipt, "note": ""}


# ------------------------------------------------------------------------ rows
def page_title(html: str) -> str:
    m = _TITLE.search(html or "")
    return " ".join(re.sub(r"<[^>]+>", " ", m.group(1)).split()) if m else ""


def is_js_shell(html: str, passages: list) -> bool:
    """No passage came out of the extractor AND the page names a JavaScript app root (or asks
    for JavaScript): a shell, not an empty page."""
    return not passages and bool(_ROOT_MARKERS.search(html or ""))


def title_only_row(cand: dict, html: str, *, lang: str, meta: dict = META) -> dict:
    """The record of a JS-shell page: the HN title (and the page's own title when it adds
    something), the url, the HN submission day. `source_extract` names the provenance."""
    own = page_title(html)
    title = cand["title"] or own
    text = title if not own or own.lower() in title.lower() else f"{title} - {own}"
    return {"url": cand["url"], "text": text, "title": title, "domain": host_of(cand["url"]),
            "tier": meta["tier"], "source_type": meta["source_type"],
            "trust_tier": meta["trust_tier"], "lang": lang, "published": cand["published"],
            "source_extract": "hn_story_title"}


def spam_screen(rows: list[dict]) -> list[dict]:
    """Rows of one page that `cytir.ops.spam_rows` does not flag: the row rules (R1, R2, R4, R6),
    then the page rule (R3). A flagged page keeps no row: a hijacked or doorway page is spam
    whole, and a linked page carries no other evidence of being the page the story meant."""
    from cytir.ops import spam_rows as SR  # noqa: PLC0415
    if not rows:
        return []
    url = rows[0]["url"]
    probe = [(str(i), url, SR.row_reason(r.get("title") or "", r.get("text") or "", url),
              len(SR.lexicon_hits(f"{r.get('title') or ''}\n{r.get('text') or ''}")))
             for i, r in enumerate(rows)]
    return [] if SR.page_flags(probe) else rows


def process_candidate(cand: dict, fetcher: LinkedFetcher, *, build_rows, detect_lang) -> tuple[str, list[dict], str]:
    """One candidate, request to terminal: (terminal, rows, note).
    `build_rows(url, html, meta) -> [row]` is the lane's extractor, `detect_lang(text, url) -> lang`
    its language detector; rows carry `published` = the HN submission day when the page names none."""
    got = fetcher.fetch(cand["url"])
    if got["terminal"]:
        return got["terminal"], [], got["note"]
    html = got["html"]
    meta = {**META, "published": cand["published"]}
    rows = build_rows(cand["url"], html, meta)
    if rows:
        kept = spam_screen(rows)
        return ("rows" if kept else "spam"), kept, ""
    if is_js_shell(html, rows):
        text = cand["title"] or page_title(html)
        if not text:
            return "no_passage", [], "js shell without any title"
        return "title_only", [title_only_row(cand, html, lang=detect_lang(text, cand["url"]))], ""
    return "no_passage", [], ""


def accounting(outcomes: dict[str, str]) -> dict:
    """Every candidate url has exactly one terminal."""
    out = {t: 0 for t in TERMINALS}
    for terminal in outcomes.values():
        out[terminal] += 1
    out["candidates"] = len(outcomes)
    assert sum(out[t] for t in TERMINALS) == out["candidates"]
    return out


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)
