"""watson.ch 12-month backfill: discovery from the weekly sitemaps and the monthly runs (plan.md ruling 23.13).

watson.ch (CH Media online news, DE and FR) is admitted as a BACKFILL-ONLY publisher (`cytir/news/publishers.v1.json`):
the daily run does not require it and does not fetch it. Its two sitemap indexes (`/sitemap.xml`,
`/fr/sitemap.xml`) list one urlset per ISO week (`?date=YYYYWW`, about 590 DE and 395 FR urls a week, every url with
a `lastmod`). This module

  * discovers: reads every weekly urlset of the window once, at the publisher's pace (>= `MIN_GAP_S`), and writes
    `<state-dir>/discovery/urls.jsonl` (url, lang, week, lastmod day) and `state.json` (one terminal per seed);
  * plans: one run per calendar month of the window (12), by the `lastmod` day, as an explicit url list on its own
    free as-of day (`ASOF_FIRST` + n: no spool, no pack, no other backfill's day), through the same compose ->
    promotion-queue path as every other backfill (`coverage_backfill.execute`). The current month's urls join the
    last full month's run; a run over the queue row cap REFUSES, it never truncates.

Left out of the url list, and counted, never silently: `videos/`, and the marketing sections (`vermarktung`,
`commercialization`, `commercialisation`, `publicite`): advertising, not journalism. A page's date comes from the page
itself at fetch time (`published`); `lastmod` only buckets the url into a run.
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
import re

HOST = "www.watson.ch"
ORIGIN = f"https://{HOST}"
INDEXES = (("de", "/sitemap.xml"), ("fr", "/fr/sitemap.xml"))
WINDOW_MONTHS = 12
#: robots.txt says `Crawl-delay: 1`; the backfill is slower (brief 2026-10-03: >= 2 s).
MIN_GAP_S = 2.0
HOST_RATE = 1.0 / MIN_GAP_S
#: the queue row cap per run (`sources/ch/watson.yaml` `cap`): a month holds about 4,200 urls.
CAP = 8000
EXCLUDED_SEGMENTS = ("videos", "vermarktung", "commercialization", "commercialisation", "publicite")
SCHEMA = "aw.watson-discovery.v1"
#: Free as-of days: no daily spool (from 2026-07-21), no other backfill (swissinfo month ends, SRF the 15th, RTS days
#: 2 to 6 of each month and 2026-07-07 to 07-20, RSI 07-19). One day keys one spool and one pack; the run start
#: refuses a day that is taken (`coverage_backfill.occupied`).
ASOF_FIRST = _dt.date(2026, 6, 16)
SMOKE_ASOF = "2026-06-29"
SMOKE_URLS = 25
_LOC = re.compile(r"<url[\s>].*?</url>", re.S)
_LOC_ONE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")
_LASTMOD = re.compile(r"<lastmod>\s*([^<\s]+)\s*</lastmod>")
_ARTICLE = re.compile(r"/\d{6,}-[^/]+$")


def window_start(today: _dt.date, months: int = WINDOW_MONTHS) -> _dt.date:
    y, m = divmod(today.year * 12 + today.month - 1 - months, 12)
    return _dt.date(y, m + 1, min(today.day, 28))


def iso_week_code(d: _dt.date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}{w:02d}"


def weeks(start: _dt.date, end: _dt.date) -> list[str]:
    """ISO week codes (`202640`) of every week that holds a day of [start, end], oldest first."""
    out, d = [], start - _dt.timedelta(days=start.weekday())
    while d <= end:
        out.append(iso_week_code(d))
        d += _dt.timedelta(days=7)
    return out


def seeds(today: _dt.date, months: int = WINDOW_MONTHS) -> list[tuple[str, str, str]]:
    """(lang, week, url) of every weekly urlset of the window."""
    return [(lang, w, f"{ORIGIN}{path}?date={w}")
            for w in weeks(window_start(today, months), today) for lang, path in INDEXES]


def parse_urlset(text: str) -> list[tuple[str, str]]:
    """[(url, lastmod ISO string or "")] of a weekly urlset."""
    out = []
    for block in _LOC.findall(text):
        loc, mod = _LOC_ONE.search(block), _LASTMOD.search(block)
        if loc:
            out.append((loc.group(1), mod.group(1) if mod else ""))
    return out


def classify(url: str) -> str:
    """`article`, or the reason the url is left out."""
    path = url[len(ORIGIN):] if url.startswith(ORIGIN) else ""
    if not path:
        return "foreign_host"
    parts = [p for p in path.split("/") if p]
    if parts and parts[0] == "fr":
        parts = parts[1:]
    if parts and parts[0] in EXCLUDED_SEGMENTS:
        return f"section_{parts[0]}"
    if not _ARTICLE.search(path):
        return "not_article_shaped"
    return "article"


def discover(disc_dir: pathlib.Path, fetch, today: _dt.date, months: int = WINDOW_MONTHS) -> dict:
    """Read each weekly urlset once. `fetch(url) -> (status, text)`; status 0 = transport failure, -1 = robots
    disallows. Every seed ends in one terminal: `ok`, `empty` (a readable urlset with no url), `nested_sitemap`
    (it lists sitemaps), `http_<n>`, `robots_disallowed`, `error`. Resumable: `ok` and `empty` seeds are not
    fetched again. Returns the accounting (`clean` = every seed ok or empty)."""
    disc_dir.mkdir(parents=True, exist_ok=True)
    st_path, urls_path = disc_dir / "state.json", disc_dir / "urls.jsonl"
    state = (json.loads(st_path.read_text()) if st_path.is_file()
             else {"schema": SCHEMA, "seeds": {}})
    start = window_start(today, months).isoformat()
    state.update(today=today.isoformat(), window_start=start)
    seen = set()
    if urls_path.is_file():
        seen = {json.loads(line)["url"] for line in urls_path.read_text().splitlines() if line.strip()}
    with urls_path.open("a", encoding="utf-8") as out:
        for lang, week, seed in seeds(today, months):
            if state["seeds"].get(seed, {}).get("status") in ("ok", "empty"):
                continue
            status, text = fetch(seed)
            entry = {"lang": lang, "week": week, "http": status}
            if status == -1:
                entry["status"] = "robots_disallowed"
            elif status == 0:
                entry["status"] = "error"
            elif status != 200:
                entry["status"] = f"http_{status}"
            else:
                rows = parse_urlset(text)
                entry["locs"] = len(rows)
                if any(u.split("?")[0].endswith(".xml") for u, _ in rows):
                    entry["status"] = "nested_sitemap"
                else:
                    entry["status"] = "ok" if rows else "empty"
                left = {}
                taken = 0
                for url, mod in rows:
                    why = classify(url)
                    if why != "article":
                        left[why] = left.get(why, 0) + 1
                    elif url not in seen:
                        seen.add(url)
                        taken += 1
                        out.write(json.dumps({"url": url, "lang": lang, "week": week, "day": mod[:10]},
                                             sort_keys=True) + "\n")
                entry["urls"] = taken
                entry["left_out"] = left
            state["seeds"][seed] = entry
            st_path.write_text(json.dumps(state, indent=1, sort_keys=True))
    st_path.write_text(json.dumps(state, indent=1, sort_keys=True))
    return accounting(state, today, months)


def accounting(state: dict, today: _dt.date, months: int = WINDOW_MONTHS) -> dict:
    planned = seeds(today, months)
    terminals: dict[str, int] = {}
    left: dict[str, int] = {}
    for _, _, url in planned:
        e = state["seeds"].get(url, {})
        t = e.get("status", "open")
        terminals[t] = terminals.get(t, 0) + 1
        for k, v in (e.get("left_out") or {}).items():
            left[k] = left.get(k, 0) + v
    if sum(terminals.values()) != len(planned):  # pragma: no cover -- arithmetic guard
        raise RuntimeError("conservation: seeds != terminals")
    return {"seeds": len(planned), "terminals": terminals, "left_out": left,
            "clean": set(terminals) <= {"ok", "empty"}}


def month_buckets(rows: list[dict], start: str, today: _dt.date) -> tuple[dict[str, list[str]], dict]:
    """{month -> urls oldest first, as many runs as full months} plus what was left out. A url with no `day`, or a
    day before the window start, is counted and left out. The current month joins the previous month's run."""
    by: dict[str, list[tuple[str, str]]] = {}
    left = {"urls": len(rows), "no_day": 0, "before_window": 0, "bucketed": 0}
    this_month = today.isoformat()[:7]
    for r in rows:
        day = r.get("day") or ""
        if not day:
            left["no_day"] += 1
        elif day < start:
            left["before_window"] += 1
        else:
            by.setdefault(day[:7], []).append((day, r["url"]))
            left["bucketed"] += 1
    if this_month in by:
        prev = (_dt.date(today.year, today.month, 1) - _dt.timedelta(days=1)).isoformat()[:7]
        by.setdefault(prev, []).extend(by.pop(this_month))
    return {m: [u for _, u in sorted(v)] for m, v in sorted(by.items())}, left


FIRST_MONTH = "2025-10"


def as_of_for(month: str) -> str:
    """The run's own free day, from its MONTH (stable across a resume that discovers again later)."""
    index = (int(month[:4]) - int(FIRST_MONTH[:4])) * 12 + int(month[5:7]) - int(FIRST_MONTH[5:7])
    if not 0 <= index <= 12:
        raise SystemExit(f"watson {month}: outside the free as-of days reserved for it (2025-10 .. 2026-10)")
    return (ASOF_FIRST + _dt.timedelta(days=index)).isoformat()


def runs(state_dir: pathlib.Path | None, *, smoke: bool, make_run):
    """The `coverage_backfill.Run`s of watson (`make_run` builds one), from a clean discovery."""
    if state_dir is None:
        raise SystemExit("watson needs --state-dir with a discovery/ directory")
    disc = state_dir / "discovery"
    urls_path, st_path = disc / "urls.jsonl", disc / "state.json"
    if not urls_path.is_file() or not st_path.is_file():
        raise SystemExit(f"no watson discovery in {disc}: run `discover watson` first")
    state = json.loads(st_path.read_text())
    day = _dt.date.fromisoformat(state["today"])
    acct = accounting(state, day)
    if not acct["clean"]:
        raise SystemExit(f"watson discovery is not clean: {acct['terminals']} "
                         "(every weekly urlset must be ok or empty; re-run `discover watson`)")
    rows = [json.loads(line) for line in urls_path.read_text().splitlines() if line.strip()]
    buckets, _ = month_buckets(rows, state["window_start"], day)
    if not buckets:
        raise SystemExit("watson discovery holds no url inside the window")
    if smoke:
        month = next(iter(buckets))
        return [make_run(key=f"watson-{month}-smoke", month=month, as_of=SMOKE_ASOF, cap=SMOKE_URLS,
                         urls=buckets[month][:SMOKE_URLS], smoke=True)]
    out = []
    for month, urls in buckets.items():
        if len(urls) > CAP:
            raise SystemExit(f"watson {month}: {len(urls)} urls exceed the run cap {CAP}; "
                             "the stage would truncate. Split the month before running.")
        out.append(make_run(key=f"watson-{month}", month=month, as_of=as_of_for(month), cap=CAP, urls=urls,
                            smoke=False))
    return out
