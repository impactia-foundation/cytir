#!/usr/bin/env python3
# Moved from the legacy news_coverage_backfill.py (task 683 S7 tranche 4); the old path is an alias shim.
"""News 12-month coverage backfill driver (task 676, ruling 11).

Closes the gap between what prod holds (one month of swissinfo, two of SRF, two
of RSI) and the rolling 12-month window, through the SAME compose -> promotion
queue path the daily timer uses: one `news_daily_pack_and_queue.run_one_day`
per (publisher, as-of day). It never applies, never writes the twin or prod and
takes no lock. The queue drain applies what it queues.

WHY A DRIVER AND NOT THE DAILY TIMER'S ARGUMENTS
-------------------------------------------------
The daily run's discovery cannot reach old content. Measured 2026-09-30 from the
stage state of 09-25 and 09-28: swissinfo's 200-child sitemap budget is consumed
by the English seed (798 of 798 fetched URLs are /eng/); SRF's monthly article
sitemaps carry no `lastmod`, so the run takes the first 1,500 of 357,399
undated URLs; RSI's sitemap lags the site. This driver feeds the existing stage
script an EXPLICIT seed list per (publisher, month): the publisher's own
date-partitioned sitemaps for that month, nothing else. No stage, compose or
consent code is forked.

  swissinfo  /{eng,ger,fre,ita}/sitemap-YYYY.xml?mm=MM&dd=DD   one urlset per day
             and language; every URL has a `lastmod`
  srf        /sitemaps/aron/articles/YYYY_MM.xml               one urlset per month,
             no `lastmod`: the date comes from the page (JSON-LD), never a guess
  rsi        /generic-static/sitemap/archive/story/N.xml       1,000 stories per file,
             files 243..259 = Aug 2025 .. Jul 2026; every URL has a `lastmod`
  rts        NO archive exists (17 section files, 609 URLs, no dates, robots
             Crawl-delay 10). Backfilled by section pagination (`rts_section_discovery`,
             `discover rts`) and explicit url runs; see "RTS" below.

RTS (plan.md ruling 18, Tim 2026-09-30)
---------------------------------------
Two steps. `discover rts` paginates the section hubs at the host's 10 s crawl delay and
writes `<state-dir>/discovery/urls.jsonl` (resumable, every section ends in one terminal).
`smoke rts` / `run rts` then feed those urls, bucketed by month, to the stage as an explicit
`urls` list (no sitemap). A month runs on a FREE as-of day on or after its last article:
day 2 of the next month (2025-10-02 .. 2026-07-02), and 2026-07-18 for July 1 to 17.
The daily timer owns every day from 2026-07-21 (spools), SRF owns 2026-08-29, so articles
from 2026-07-18 on have no as-of day on or after them. The explicit-url runs do not need
one: their pages carry their own date, and the page-date clamp is the REAL current day
(`sitemap_ingest.page_date_clamp_day`, #468, release r22), so the tail runs on a free past
day (`RTS_TAIL_ASOF`) and keeps its page dates. A release without that clamp makes
`smoke`/`run` refuse (`release_supports_page_date_clamp`).

SWISSINFO TAIL (orchestrator 2026-09-30)
----------------------------------------
The daily run fetched English only until 2026-09-29, so ger, fre and ita of 2026-09-19
to 09-29 are missing and no free as-of day is on or after them (sitemap selection wants
`lastmod <= as-of`). `discover swissinfo-tail` reads the 33 day sitemaps once (one request
a second, consent and robots first) and writes the urls; `smoke`/`run swissinfo-tail`
feed them as an explicit list on `SWISSINFO_TAIL_ASOF`, the same mechanism as the RTS tail.

AS-OF DAYS
----------
`run_one_day(day)` keys its spool (`stage-<day>`), its staging collections and its
pack id (`news-daily-<yyyymmdd>`) off the day, so a backfill must use a day the
daily timer never uses. `plan()` refuses a day whose spool, pack or staging name
already exists, and the daily timer's own days (>= 2026-07-21 have spools) are
out of reach by construction. Paywalled and consent-closed publishers are not in
`PUBLISHERS` and cannot be added by flag.

SMOKE BEFORE ANY FULL RUN
-------------------------
`smoke PUBLISHER` runs one tiny slice through the whole chain and asserts
at least one row reached the queue pack. `run` refuses a publisher without a
passed smoke on record (memory smoke-test-yield-end-to-end).

STATE
-----
`STATE_DIR/state.json` (one entry per run key) and `STATE_DIR/NEXT.md`, rewritten
after every run. A run already `done` is skipped on restart.
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import datetime as _dt
import json
import os
from cytir.core import envnames as _envnames
import pathlib
import re
import shlex
import shutil
import sys
import time
from dataclasses import dataclass, field

#: The app root (the checkout the driver runs from). The resume command prints it, and prints the
#: legacy shim path (units and NEXT.md files name that path).
APP_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cytir.core.paths import operator_settings  # noqa: E402
from cytir.news import daily as D  # noqa: E402
from cytir.news import stage_contract as NSC  # noqa: E402
from cytir.news import watson_backfill as WB  # noqa: E402



def default_state_dir() -> pathlib.Path:
    """The backfill state tree: operator setting `news_backfill_dir` (refused by name when unconfigured)."""
    return pathlib.Path(operator_settings().news_backfill_dir)


def __getattr__(name):
    # `STATE_DIR` stays readable but resolves on first ACCESS, never at import.
    if name == "STATE_DIR":
        return default_state_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: Egress budget for news (plan.md ruling 5): <= 1 MB/s.
EGRESS_BUDGET_MBPS = 1.0
#: The NIC counter (`iface_bytes`) counts EVERY job on enp37s0f0, so it reads 2.5 to
#: 5 MB/s during a 25-page smoke that fetched 14 MB (measured 2026-09-30: the other
#: lanes share the card). The controller therefore steers on this run's OWN fetched
#: bytes (the sealed originals it wrote: `pages/`, `raw/`, `hn_linked/` under its
#: spool, which the preservation contract writes for every body it fetches). The NIC
#: figure is still recorded, labelled as whole-NIC.
SPOOL_PAYLOAD_DIRS = ("pages", "raw", "hn_linked")
#: Refuse to start a run with less free space than this on the spool's disk. The
#: sealed originals of a full swissinfo month are about 2 GB (measured: 14 MB for
#: 26 pages); nvme4t had 126 GB free at 97% on 2026-09-30.
MIN_FREE_GB = 60

SWISSINFO_LANGS = ("eng", "ger", "fre", "ita")
RSI_ARCHIVE_FILES = tuple(range(243, 260))

#: Free as-of days for the explicit-url tails (no spool, no pack: `occupied()` still
#: checks). Both are before the daily timer's territory (2026-07-21) and away from the
#: days other runs own (swissinfo 07-20, RSI 07-19, RTS July 07-18).
RTS_TAIL_ASOF = "2026-07-16"
SWISSINFO_TAIL_ASOF = "2026-07-17"
SWISSINFO_TAIL_SMOKE_ASOF = "2025-09-26"
SWISSINFO_TAIL_FIRST, SWISSINFO_TAIL_LAST = "2026-09-19", "2026-09-29"
SWISSINFO_TAIL_LANGS = ("ger", "fre", "ita")
TAIL_SMOKE_URLS = 25
#: The page-date clamp lives in the release's own `cytir/acquire/sitemap_ingest.py`. The pre-move `pipeline/`
#: path is no longer listed: the news lane pins a release that has the package file (task 3017).
CLAMP_FILE = "cytir/acquire/sitemap_ingest.py"
CLAMP_FILES = (CLAMP_FILE,)
CLAMP_MARKS = ("def page_date_clamp_day", "now_ref=page_date_clamp_day(")


def release_supports_page_date_clamp(release: pathlib.Path | None = None) -> bool:
    """True when the pinned stage release clamps PAGE dates to the real day (r22 and
    later). Read from the release itself: a release that clamps to the as-of day throws
    away every page date after a past as-of day, and the explicit-url tails run on one."""
    for rel in CLAMP_FILES:
        try:
            text = (pathlib.Path(release or D.R) / rel).read_text()
        except OSError:
            continue
        if all(mark in text for mark in CLAMP_MARKS):
            return True
    return False


def require_page_date_clamp_release(release: pathlib.Path | None = None) -> None:
    if not release_supports_page_date_clamp(release):
        raise SystemExit(
            f"REFUSING: the pinned stage release {release or D.R} clamps page dates to "
            "the as-of day (needs r22 or later, #468). An explicit-url tail on a past "
            "as-of day would lose every page date after it. Re-cut the release "
            "(task 671 recipe) before this run.")


# ------------------------------------------------------------------ seeds
def month_days(month: str, last_day: _dt.date | None = None) -> list[_dt.date]:
    y, m = int(month[:4]), int(month[5:7])
    d = _dt.date(y, m, 1)
    out = []
    while d.month == m:
        if last_day is not None and d > last_day:
            break
        out.append(d)
        d += _dt.timedelta(days=1)
    return out


def swissinfo_seeds(month: str, last_day: _dt.date | None = None) -> list[str]:
    seeds = []
    for d in month_days(month, last_day):
        for lang in SWISSINFO_LANGS:
            seeds.append(f"https://www.swissinfo.ch/{lang}/sitemap-{d.year}.xml"
                         f"?mm={d.month:02d}&dd={d.day:02d}")
    return seeds


def swissinfo_smoke_seeds(month: str, last_day: _dt.date | None = None) -> list[str]:
    """One day, four languages: the smoke must be tiny."""
    d = last_day or month_days(month)[-1]
    return [f"https://www.swissinfo.ch/{lang}/sitemap-{d.year}.xml"
            f"?mm={d.month:02d}&dd={d.day:02d}" for lang in SWISSINFO_LANGS]


def rsi_smoke_seeds(month: str, last_day: _dt.date | None = None) -> list[str]:
    return [f"https://www.rsi.ch/generic-static/sitemap/archive/story/{RSI_ARCHIVE_FILES[0]}.xml"]


def srf_seeds(month: str, last_day: _dt.date | None = None) -> list[str]:
    return [f"https://www.srf.ch/sitemaps/aron/articles/{month[:4]}_{month[5:7]}.xml"]


def rsi_seeds(month: str, last_day: _dt.date | None = None) -> list[str]:
    return [f"https://www.rsi.ch/generic-static/sitemap/archive/story/{n}.xml"
            for n in RSI_ARCHIVE_FILES]


#: A coverage backfill is partial by design: it covers ONE publisher. The stage's
#: source-boundary contract and the pack step's arm list take that publisher as
#: AW_NEWS_EXPECTED_HOSTS (task 677 follow-up of 676), so no other host is fetched,
#: not even as a one-page filler. The pinned release must carry that support; a
#: release that does not (r19 and older) makes `smoke`/`run` refuse, loudly, rather
#: than fail inside the stage.
EXPECTED_HOSTS_ENV = "AW_NEWS_EXPECTED_HOSTS"
#: Files of the pinned release that must honour EXPECTED_HOSTS_ENV.
EXPECTED_HOSTS_FILES = ("deploy/scripts/news_refresh_stage.sh",
                        "cytir/sources/families/news_tier2_apply.py")


def release_supports_expected_hosts(release: pathlib.Path | None = None) -> bool:
    """True when the pinned stage release honours AW_NEWS_EXPECTED_HOSTS: the stage
    script and the apply driver name it, and the contract's `arms` subcommand
    answers for one publisher. Read from the release itself, never assumed."""
    import subprocess
    release = pathlib.Path(release or D.R)
    try:
        if not all(EXPECTED_HOSTS_ENV in (release / f).read_text()
                   for f in EXPECTED_HOSTS_FILES):
            return False
        out = subprocess.run(
            [sys.executable, "-m", "cytir.news.stage_contract",
             "arms", "--expected-hosts", "www.srf.ch"],
            cwd=release, env={**os.environ, "PYTHONPATH": str(release)},
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and out.stdout.split() == ["chmedia"]


def require_expected_hosts_release(release: pathlib.Path | None = None) -> None:
    if not release_supports_expected_hosts(release):
        raise SystemExit(
            f"REFUSING: the pinned stage release {release or D.R} does not honour "
            f"{EXPECTED_HOSTS_ENV} (needs r20 or later). Re-cut the release "
            "(task 671 recipe) before a backfill run.")


@dataclass(frozen=True)
class Publisher:
    name: str
    domain: str              # the queue row's `domain` (and the state.json host key)
    seeds: object            # (month, last_day) -> [url]
    cap: int                 # queue row cap; also AW_RECENT_CAP for the run
    #: month -> as-of day. The day must be free (see module docstring).
    as_of: object
    months: tuple[str, ...]
    window_months: int = 1
    smoke_month: str = ""
    smoke_as_of: str = ""
    smoke_cap: int = 25
    smoke_seeds: object = None
    host_rate: float = 1.0
    #: explicit-url publisher: the runs come from `rts_runs`, not from `seeds`
    urls_source: bool = False


def _month_end(month: str) -> str:
    return month_days(month)[-1].isoformat()


def _as_of_month_end(month: str) -> str:
    return _month_end(month)


def _months(a: str, b: str) -> tuple[str, ...]:
    out, y, m = [], int(a[:4]), int(a[5:7])
    while (y, m) <= (int(b[:4]), int(b[5:7])):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return tuple(out)


PUBLISHERS: dict[str, Publisher] = {
    # Oct 2025 .. Jun 2026 by month end; July 1..20 by 2026-07-20 (stage-2026-07-21.. exist).
    "swissinfo": Publisher(
        "swissinfo", "www.swissinfo.ch", swissinfo_seeds, 60000,
        lambda m: "2026-07-20" if m == "2026-07" else _as_of_month_end(m),
        _months("2025-10", "2026-07"), smoke_month="2025-09",
        smoke_as_of="2025-09-29", smoke_seeds=swissinfo_smoke_seeds),
    # SRF month files are undated: any free day works, so the 15th of the month
    # (never the swissinfo month-end). Aug 2026 by 2026-08-29, its only free day.
    "srf": Publisher(
        "srf", "www.srf.ch", srf_seeds, 100000,
        lambda m: "2026-08-29" if m == "2026-08" else f"{m}-15",
        _months("2025-10", "2026-08"), smoke_month="2025-10",
        smoke_as_of="2025-09-28", smoke_seeds=srf_seeds),
    # One run: files 243..259 hold Aug 2025 .. Jul 2026; window 11 months back from 2026-07-20.
    "rsi": Publisher(
        "rsi", "www.rsi.ch", rsi_seeds, 60000,
        lambda m: "2026-07-19", ("2026-07",), window_months=11,
        smoke_month="2025-10", smoke_as_of="2025-09-27",
        smoke_seeds=rsi_smoke_seeds),
    # No archive sitemap: urls come from `rts_section_discovery`. host_rate 0.1 = the
    # robots Crawl-delay of 10 s (the stage takes the slower of the two).
    "rts": Publisher(
        "rts", "www.rts.ch", lambda m, d=None: [], 100000, lambda m: rts_as_of(m),
        (), host_rate=0.1, urls_source=True),
    # ger, fre, ita of 2026-09-19..09-29: urls from `discover swissinfo-tail`, one run.
    "swissinfo-tail": Publisher(
        "swissinfo-tail", "www.swissinfo.ch", lambda m, d=None: [], 60000,
        lambda m: SWISSINFO_TAIL_ASOF, (), host_rate=1.0, urls_source=True),
    # watson.ch, backfill-only (cytir/news/publishers.v1.json): weekly sitemaps -> `discover watson` -> one explicit-url
    # run per month on its own free as-of day (cytir/news/watson_backfill.py). host_rate 0.5 = one request per 2 s.
    "watson": Publisher(
        "watson", WB.HOST, lambda m, d=None: [], WB.CAP, lambda m: WB.as_of_for(m), (),
        host_rate=WB.HOST_RATE, urls_source=True),
}


@dataclass
class Run:
    key: str
    publisher: str
    month: str
    as_of: str
    window_months: int
    seeds: list[str]
    cap: int
    host_rate: float
    smoke: bool = False
    extra: dict = field(default_factory=dict)
    #: explicit urls (RTS); when set the queue row carries `urls`, not `sitemaps`
    urls: list = field(default_factory=list)
    #: why this run cannot start (no free as-of day); recorded, never skipped silently
    blocked: str = ""

    def queue(self) -> list[dict]:
        target = PUBLISHERS[self.publisher].domain
        row = {"domain": target, "wave": "news", "class": "news",
               "cap": self.cap, "include": []}
        if self.urls:
            row["urls"] = self.urls
        else:
            row["sitemaps"] = self.seeds
        return [row]


def _dayc(day: str) -> str:
    return day.replace("-", "")


def occupied(day: str, *, work: pathlib.Path | None = None,
             queue_dir: pathlib.Path | None = None) -> list[str]:
    """Names that would collide with a run keyed on `day`."""
    work = work or D.W
    queue_dir = queue_dir or D.QUEUE_DIR
    hits = []
    for p in (work / f"stage-{day}", queue_dir / f"news-daily-{_dayc(day)}"):
        if p.exists():
            hits.append(str(p))
    return hits


STAGING_URL = os.environ.get("AW_STAGING_URL", "http://localhost:6398")
ARMS = ("chmedia", "swissinfo", "hn")


def drop_staging_collections(day: str, *, delete=None) -> list[str]:
    """Delete this day's OWN staged arm collections (`aw_dnews256_<day>_<arm>`) on
    the staging Qdrant before a retry: the pack step scrolls each whole collection,
    so a half-filled leftover would be packed with the retry's rows. Only names
    built for this exact day are ever deleted."""
    import urllib.request
    dropped = []
    for arm in ARMS:
        name = NSC.news_stage_collection_name(
            day, arm, int(os.environ.get("AW_NEWS_MRL_DIM", "256")))
        if not name.startswith("aw_dnews") or f"_{_dayc(day)}_" not in name:
            raise SystemExit(f"refusing to delete {name!r}")
        if delete is not None:
            delete(name)
        else:
            req = urllib.request.Request(f"{STAGING_URL}/collections/{name}", method="DELETE")
            try:
                IH.urlopen(req, timeout=60).read()
            except Exception:  # noqa: BLE001 -- an absent collection is the normal case
                continue
        dropped.append(name)
    return dropped


def set_aside_failed_spool(run: "Run", state: dict, *, work: pathlib.Path | None = None,
                           queue_dir: pathlib.Path | None = None,
                           delete=None) -> str | None:
    """A run recorded `failed` that never produced a pack left its spool behind;
    that spool would make a retry look like a day collision. Rename it to
    `stage-<day>.failed-<epoch>` (kept as evidence) so the retry starts clean.
    A spool with a pack, or one this state file did not fail, is never touched."""
    work = work or D.W
    queue_dir = queue_dir or D.QUEUE_DIR
    spool = work / f"stage-{run.as_of}"
    pack = queue_dir / f"news-daily-{_dayc(run.as_of)}"
    if (state.get("runs", {}).get(run.key, {}).get("status") == "failed"
            and spool.exists() and not pack.exists()):
        target = work / f"stage-{run.as_of}.failed-{int(time.time())}"
        spool.rename(target)
        drop_staging_collections(run.as_of, delete=delete)
        return str(target)
    return None


#: The daily timer owns every day from here on (spools exist), SRF owns 2026-08-29.
DAILY_TERRITORY_START = "2026-07-21"
RTS_JULY_LAST_FREE_ASOF = "2026-07-18"     # covers July 1 to 17 (SWISSINFO 07-20, RSI 07-19)
RTS_TAIL_KEY = "rts-tail"
RTS_SMOKE_ASOF = "2025-10-01"
RTS_SMOKE_URLS = 25


def rts_as_of(month: str, generation: int = 1) -> str:
    """The free as-of day for RTS articles of `month`: day 2 of the next month
    (swissinfo owns month ends, SRF the 15th, RSI 2026-07-19), 2026-07-18 for July.
    A later GENERATION (a second discovery pass whose new urls need runs of their own: one
    as-of day keys one spool and one pack, and generation 1 used the days above) takes
    day `1 + generation` of the next month, and for July the free days 07-13 / 07-14 and
    two earlier per generation (`_rts_july_days`; 07-15 to 07-20 are owned by other runs).
    Whether such a day is really free is `occupied()`'s answer at run time."""
    if month == "2026-07":
        return _rts_july_days(generation)[0]
    y, m = int(month[:4]), int(month[5:7])
    y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{y:04d}-{m:02d}-{1 + generation:02d}"


def _rts_july_days(generation: int) -> tuple[str, str]:
    """(July-early as-of, tail as-of) of a generation. Generation 1 = the days that already ran
    (07-18, 07-16). Later ones step down from 07-13 / 07-14 by two (07-02 is rts-2026-06, 07-03 to
    07-07 are the June runs of generations 2 to 5): free on the work box 2026-10-03."""
    if generation == 1:
        return RTS_JULY_LAST_FREE_ASOF, RTS_TAIL_ASOF
    step = _dt.timedelta(days=2 * (generation - 2))
    return ((_dt.date(2026, 7, 13) - step).isoformat(), (_dt.date(2026, 7, 14) - step).isoformat())


def _gen_key(key: str, generation: int) -> str:
    return key if generation == 1 else f"{key}-g{generation}"


def rts_url_buckets(state_dir: pathlib.Path) -> tuple[dict[str, list[str]], dict]:
    """{month -> urls, oldest first} from the discovery output, plus the accounting of
    what was left out. A url's day is its card date, else its page cursor date; a url
    older than the discovery window start (the last page overshoots) or with no day at
    all is counted as left out, never put in a month."""
    disc = state_dir / "discovery"
    urls_path, st_path = disc / "urls.jsonl", disc / "state.json"
    if not urls_path.is_file() or not st_path.is_file():
        raise SystemExit(f"no RTS discovery in {disc}: run `discover rts` first")
    start = json.loads(st_path.read_text()).get("window_start", "")
    rows = [json.loads(line) for line in urls_path.open(encoding="utf-8") if line.strip()]
    by_month: dict[str, list[tuple[str, str]]] = {}
    left = {"urls": len(rows), "before_window": 0, "no_day": 0, "bucketed": 0}
    for r in rows:
        day = r.get("card_date") or r.get("cursor_date")
        if not day:
            left["no_day"] += 1
        elif day < start:
            left["before_window"] += 1
        else:
            by_month.setdefault(day[:7], []).append((day, r["url"]))
            left["bucketed"] += 1
    return ({m: [u for _, u in sorted(v)] for m, v in sorted(by_month.items())}, left)


def rts_runs(state_dir: pathlib.Path | None, *, smoke: bool, generation: int = 1) -> list[Run]:
    if generation < 1 or generation > 5:
        raise SystemExit(f"generation {generation} is out of range 1..5")
    if state_dir is None:
        raise SystemExit("rts needs --state-dir with a discovery/ directory")
    pub = PUBLISHERS["rts"]
    buckets, _ = rts_url_buckets(state_dir)
    days = {}
    for line in (state_dir / "discovery" / "urls.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        days[r["url"]] = r.get("card_date") or r.get("cursor_date") or ""
    if smoke:
        month = next(iter(buckets))
        smoke_asof = (_dt.date.fromisoformat(RTS_SMOKE_ASOF)
                      - _dt.timedelta(days=generation - 1)).isoformat()
        return [Run(key=_gen_key(f"rts-{month}-smoke", generation), publisher="rts", month=month,
                    as_of=smoke_asof, window_months=1, seeds=[],
                    cap=RTS_SMOKE_URLS, host_rate=pub.host_rate, smoke=True,
                    urls=buckets[month][:RTS_SMOKE_URLS])]
    runs = []
    g = generation
    tail_asof = _rts_july_days(g)[1]
    july_edge = "2026-07-17"
    tail_key = _gen_key(RTS_TAIL_KEY, g)
    for month, urls in buckets.items():
        if month == "2026-07":
            early = [u for u in urls if days[u] <= july_edge]
            late = [u for u in urls if days[u] > july_edge]
            parts = [(_gen_key(f"rts-{month}", g), rts_as_of(month, g), early, ""),
                     (tail_key, tail_asof, late, "")]
        elif month > "2026-07":
            parts = [(tail_key, tail_asof, urls, "")]
        else:
            parts = [(_gen_key(f"rts-{month}", g), rts_as_of(month, g), urls, "")]
        for key, as_of, part, _ in parts:
            existing = next((r for r in runs if r.key == key), None)
            if existing:
                existing.urls += part
                continue
            blocked = "" if as_of else (
                "no free as-of day: every day from %s has a daily spool" % DAILY_TERRITORY_START)
            runs.append(Run(key=key, publisher="rts", month=month, as_of=as_of,
                            window_months=1, seeds=[], cap=pub.cap,
                            host_rate=pub.host_rate, urls=list(part), blocked=blocked))
    return [r for r in runs if r.urls]


# ------------------------------------------------------------ swissinfo tail
_LOC = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")
SWISSINFO_DISC_SCHEMA = "aw.swissinfo-tail-discovery.v1"


def swissinfo_tail_seeds() -> list[tuple[str, str, str]]:
    """(lang, day, url) of every day sitemap of the tail: 3 languages x 11 days."""
    first, last = (_dt.date.fromisoformat(SWISSINFO_TAIL_FIRST),
                   _dt.date.fromisoformat(SWISSINFO_TAIL_LAST))
    out, d = [], first
    while d <= last:
        for lang in SWISSINFO_TAIL_LANGS:
            out.append((lang, d.isoformat(),
                        f"https://www.swissinfo.ch/{lang}/sitemap-{d.year}.xml"
                        f"?mm={d.month:02d}&dd={d.day:02d}"))
        d += _dt.timedelta(days=1)
    return out


def swissinfo_tail_discover(disc_dir: pathlib.Path, fetch) -> dict:
    """Read each day sitemap once; `fetch(url) -> (status, text)` (status 0 = transport
    failure, -1 = robots disallows). Every seed ends in one terminal: `ok` (urls found),
    `empty` (a readable urlset with no article), `nested_sitemap` (it lists sitemaps, not
    articles: this tool does not recurse, so it says so), `http_<n>`, `robots_disallowed`
    or `error`. Resumable: an `ok` or `empty` seed is not fetched again; the others are
    retried."""
    disc_dir.mkdir(parents=True, exist_ok=True)
    st_path, urls_path = disc_dir / "state.json", disc_dir / "urls.jsonl"
    state = (json.loads(st_path.read_text()) if st_path.is_file()
             else {"schema": SWISSINFO_DISC_SCHEMA, "seeds": {}})
    seen = set()
    if urls_path.is_file():
        seen = {json.loads(line)["url"] for line in urls_path.read_text().splitlines()
                if line.strip()}
    with urls_path.open("a", encoding="utf-8") as out:
        for lang, day, seed in swissinfo_tail_seeds():
            if state["seeds"].get(seed, {}).get("status") in ("ok", "empty"):
                continue
            status, text = fetch(seed)
            entry = {"lang": lang, "day": day, "http": status}
            if status == -1:
                entry["status"] = "robots_disallowed"
            elif status == 0:
                entry["status"] = "error"
            elif status != 200:
                entry["status"] = f"http_{status}"
            else:
                locs = _LOC.findall(text)
                articles = [u for u in locs if not u.split("?")[0].endswith(".xml")
                            and f"/{lang}/" in u and "swissinfo.ch" in u]
                entry["locs"] = len(locs)
                if len(articles) < len(locs):
                    entry["status"] = "nested_sitemap"
                    entry["non_article_locs"] = len(locs) - len(articles)
                else:
                    entry["status"] = "ok" if articles else "empty"
                for u in articles:
                    if u not in seen:
                        seen.add(u)
                        out.write(json.dumps({"url": u, "lang": lang, "day": day}) + "\n")
                entry["urls"] = len(articles)
            state["seeds"][seed] = entry
            st_path.write_text(json.dumps(state, indent=1, sort_keys=True))
    return swissinfo_tail_accounting(state)


def swissinfo_tail_accounting(state: dict) -> dict:
    seeds = swissinfo_tail_seeds()
    terminals: dict[str, int] = {}
    for _, _, url in seeds:
        t = state["seeds"].get(url, {}).get("status", "open")
        terminals[t] = terminals.get(t, 0) + 1
    if sum(terminals.values()) != len(seeds):  # pragma: no cover -- arithmetic guard
        raise RuntimeError("conservation: seeds != terminals")
    return {"seeds": len(seeds), "terminals": terminals,
            "clean": set(terminals) <= {"ok", "empty"}}


def swissinfo_tail_runs(state_dir: pathlib.Path | None, *, smoke: bool) -> list[Run]:
    if state_dir is None:
        raise SystemExit("swissinfo-tail needs --state-dir with a discovery/ directory")
    disc = state_dir / "discovery"
    urls_path, st_path = disc / "urls.jsonl", disc / "state.json"
    if not urls_path.is_file() or not st_path.is_file():
        raise SystemExit(f"no swissinfo-tail discovery in {disc}: run "
                         "`discover swissinfo-tail` first")
    acct = swissinfo_tail_accounting(json.loads(st_path.read_text()))
    if not acct["clean"]:
        raise SystemExit(f"swissinfo-tail discovery is not clean: {acct['terminals']} "
                         "(every seed must be ok or empty; re-run `discover`)")
    rows = [json.loads(line) for line in urls_path.read_text().splitlines() if line.strip()]
    urls = [r["url"] for r in sorted(rows, key=lambda r: (r["day"], r["lang"], r["url"]))]
    pub = PUBLISHERS["swissinfo-tail"]
    if smoke:
        return [Run(key="swissinfo-tail-smoke", publisher=pub.name, month="2026-09",
                    as_of=SWISSINFO_TAIL_SMOKE_ASOF, window_months=1, seeds=[],
                    cap=TAIL_SMOKE_URLS, host_rate=pub.host_rate, smoke=True,
                    urls=urls[:TAIL_SMOKE_URLS])]
    return [Run(key="swissinfo-tail", publisher=pub.name, month="2026-09",
                as_of=SWISSINFO_TAIL_ASOF, window_months=1, seeds=[], cap=pub.cap,
                host_rate=pub.host_rate, urls=urls)]


def _live_swissinfo_fetch(cache_dir: str | None = None, host: str = "www.swissinfo.ch",
                          min_gap: float = 1.0):  # pragma: no cover -- network
    """One polite GET a second (`min_gap`; never faster than the host's Crawl-delay): consent verdict
    and robots first. Used for swissinfo-tail and, with its own host and gap, for watson."""
    import httpx
    from cytir.acquire import sitemap_ingest as SI
    from cytir.consent import resolver as cr
    cache = os.path.expanduser(cache_dir or cr.CACHE_DIR)
    cres = cr.resolve_hosts([host], cache_dir=cache, progress=False, direct_crawl=True)
    verdict = SI._search_index_bucket(cres[host])
    if verdict not in {"ok", "ok_search_only"}:
        raise SystemExit(f"REFUSING: consent verdict for {host} is {verdict!r}")
    rp, delay = SI.load_robots(host, cache)
    from cytir.sources.family import HttpSpec
    from cytir.sources.http import make_client
    client = make_client(HttpSpec(      # consent-guarded (task 1621); the discovery UA, as before
        timeout=httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=60.0),
        headers={"user-agent": SI.UA["User-Agent"]}, follow_redirects=True))
    gap, last = max(min_gap, delay), [0.0]

    def fetch(url: str) -> tuple[int, str]:
        if not SI.robots_ok(rp, url):
            return -1, ""
        for attempt in range(3):
            wait = last[0] + gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last[0] = time.monotonic()
            try:
                r = client.get(url)
            except Exception:  # noqa: BLE001 -- status 0, retried by the next run
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code in (429, 503):
                time.sleep(30 * (attempt + 1))
                continue
            return r.status_code, r.text
        return 0, ""
    return fetch


def watson_runs(state_dir: pathlib.Path | None, *, smoke: bool) -> list[Run]:
    pub = PUBLISHERS["watson"]

    def make(*, key, month, as_of, cap, urls, smoke):
        return Run(key=key, publisher="watson", month=month, as_of=as_of, window_months=1, seeds=[],
                   cap=cap, host_rate=pub.host_rate, smoke=smoke, urls=urls)
    return WB.runs(state_dir, smoke=smoke, make_run=make)


def plan(names: list[str], *, today: _dt.date, smoke: bool = False,
         work: pathlib.Path | None = None,
         queue_dir: pathlib.Path | None = None,
         state_dir: pathlib.Path | None = None, generation: int = 1) -> list[Run]:
    runs: list[Run] = []
    for name in names:
        pub = PUBLISHERS[name]
        if pub.urls_source:
            maker = (swissinfo_tail_runs if name == "swissinfo-tail"
                     else watson_runs if name == "watson" else rts_runs)
            kw = {"generation": generation} if name == "rts" else {}
            for r in maker(state_dir, smoke=smoke, **kw):
                if r.as_of and _dt.date.fromisoformat(r.as_of) >= today - _dt.timedelta(days=2):
                    raise SystemExit(f"{r.key}: as-of {r.as_of} is within 2 days of today "
                                     "-- that is the daily timer's territory")
                runs.append(r)
            continue
        if smoke:
            spec = [(pub.smoke_month, pub.smoke_as_of)]
        else:
            spec = [(m, pub.as_of(m)) for m in pub.months]
        for month, as_of in spec:
            if _dt.date.fromisoformat(as_of) >= today - _dt.timedelta(days=2):
                raise SystemExit(f"{name} {month}: as-of {as_of} is within 2 days of "
                                 "today -- that is the daily timer's territory")
            last = _dt.date.fromisoformat(as_of)
            key = f"{name}-{month}" + ("-smoke" if smoke else "")
            runs.append(Run(
                key=key, publisher=name, month=month, as_of=as_of,
                window_months=pub.window_months,
                seeds=(pub.smoke_seeds if smoke else pub.seeds)(month, last),
                cap=pub.smoke_cap if smoke else pub.cap,
                host_rate=pub.host_rate, smoke=smoke))
    seen: dict[str, str] = {}
    for r in runs:
        if not r.as_of:
            continue
        if r.as_of in seen and seen[r.as_of] != r.key:
            raise SystemExit(f"{r.key} and {seen[r.as_of]} share as-of day {r.as_of}: "
                             "one day keys one spool and one pack")
        seen[r.as_of] = r.key
    return runs


# ------------------------------------------------------------------ state
def load_state(state_dir: pathlib.Path) -> dict:
    path = state_dir / "state.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": "aw.news-coverage-backfill.v1", "runs": {}, "smoke": {}}


def save_state(state_dir: pathlib.Path, state: dict) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    tmp = state_dir / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    tmp.replace(state_dir / "state.json")
    (state_dir / "NEXT.md").write_text(render_next(state))


def resume_command(names: list[str], state_dir: pathlib.Path, generation: int = 1) -> str:
    """The EXACT command that resumes this driver: the one `cmd_run` was started
    with (python, checkout, publishers, state dir, operator identity). NEXT.md
    prints it, so a stopped backfill is resumable from the file alone (task 676:
    four state dirs stalled for days because NEXT.md said only "re-run the same
    command" and the command lived in a hand-written OPERATOR.md, or nowhere)."""
    env = " ".join(f"{k}={shlex.quote(os.environ[k])}"
                   for k in ("AW_OPERATOR_URL", "AW_OPERATOR_CONTACT") if os.environ.get(k))
    if os.environ.get("AW_NEWS_RELEASE_FROM_CHECKOUT") == "1":
        # A run started on the checkout's own release must resume on it: without this the
        # resume would run on the pinned release (`cytir.news.daily`), silently older code.
        env = (env + " " if env else "") + "AW_NEWS_RELEASE_FROM_CHECKOUT=1"
    app = APP_ROOT
    return (f"cd {shlex.quote(str(app))} && PYTHONDONTWRITEBYTECODE=1 "
            + (env + " " if env else "")
            + f"{shlex.quote(sys.executable)} -m cytir.news.coverage_backfill run "
            f"{','.join(names)} --state-dir {shlex.quote(str(state_dir))}"
            + (f" --generation {generation}" if generation != 1 else ""))


#: A failure of the shared staging Qdrant is the infrastructure's, not the month's:
#: every later month would fail the same way and be recorded as a failed month.
INFRA_MARKERS = ("staging qdrant lifecycle check FAILED", "staging Qdrant unavailable")


def infra_failure(note: str, *, read=None) -> bool:
    """True when a failed run's note (or the stage log it points at) says the staging
    Qdrant was down. The stage script reports that only as `exit=1` plus a FATAL
    line in its log (task 676: swissinfo 2026-05 and RTS 2026-05 died that way)."""
    if any(m in note for m in INFRA_MARKERS):
        return True
    m = re.search(r"see (\S+)", note)
    if not m:
        return False
    try:
        text = (read or (lambda q: pathlib.Path(q).read_text(errors="replace")[-20000:]))(m.group(1))
    except OSError:
        return False
    return any(mk in text for mk in INFRA_MARKERS)


def render_next(state: dict) -> str:
    lines = ["# NEXT -- news coverage backfill (task 676)", "",
             "State file: `state.json` beside this file. Driver: "
             "`python3 -m cytir.news.coverage_backfill`. Egress budget 1 MB/s "
             "(steered on each run's own fetched bytes; the whole-NIC counter is recorded "
             "as `nic_mbps_whole_card` and includes other jobs).", "",
             f"updated: {_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}", ""]
    if state.get("pid"):
        lines += [f"driver pid: {state['pid']} (alive check: `kill -0 {state['pid']}`)", ""]
    lines += ["## Smoke", ""]
    for pub, s in sorted(state.get("smoke", {}).items()):
        lines.append(f"- {pub}: {s.get('status')} rows={s.get('rows')} "
                     f"as_of={s.get('as_of')}")
    if state.get("resume_command"):
        lines += ["", "## Resume (exact command)", "", "```", state["resume_command"], "```"]
    lines += ["", "## Runs", "",
              "| run | as-of | status | staged | rows | urls | MB/s | note |",
              "|---|---|---|--:|--:|--:|--:|---|"]
    for key, r in sorted(state.get("runs", {}).items()):
        lines.append(f"| {key} | {r.get('as_of')} | {r.get('status')} | "
                     f"{r.get('staged', '')} | {r.get('rows', '')} | "
                     f"{r.get('urls', '')} | {r.get('fetched_mb', '')} | "
                     f"{r.get('egress_mbps', '')} | "
                     f"{(r.get('note') or '')[:90]} |")
    lines += ["", "## If it stopped", "",
              "1. `kill -0 <pid>`; if dead, read the last `failed` row and its log.",
              "2. Run the command under \"Resume\" above: `done` runs are skipped, nothing is "
              "re-fetched for them.",
              "3. A `failed` run with `staged=0` is a refusal (outage or empty "
              "source), never an empty month.", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ execute
def spool_payload_bytes(day: str, work: pathlib.Path | None = None) -> int:
    """Bytes this run fetched and sealed: the sum of its spool's originals."""
    spool = (work or D.W) / f"stage-{day}"
    total = 0
    for sub in SPOOL_PAYLOAD_DIRS:
        root = spool / sub
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_file():
                    total += path.stat().st_size
    return total


def free_gb(path: pathlib.Path | None = None) -> float:
    target = path or D.W
    target = target if target.exists() else target.parent
    return shutil.disk_usage(target).free / 1e9


def pack_manifest(day: str, queue_dir: pathlib.Path | None = None) -> dict | None:
    path = (queue_dir or D.QUEUE_DIR) / f"news-daily-{_dayc(day)}" / "manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def execute(run: Run, *, state_dir: pathlib.Path, runner=None,
            iface_bytes=None) -> dict:
    """One run -> state entry. `runner(run_one_day)` and `iface_bytes` are
    injectable so the tests need no network."""
    runner = runner or D.run_one_day
    iface_bytes = iface_bytes or D.iface_bytes
    qdir = state_dir / "queues"
    qdir.mkdir(parents=True, exist_ok=True)
    queue_path = qdir / f"{run.key}.json"
    queue_path.write_text(json.dumps(run.queue(), indent=1) + "\n")
    saved = {k: os.environ.get(k)
             for k in ("AW_NEWS_QUEUE", "AW_RECENT_CAP", EXPECTED_HOSTS_ENV)}
    os.environ["AW_NEWS_QUEUE"] = str(queue_path)
    os.environ[EXPECTED_HOSTS_ENV] = PUBLISHERS[run.publisher].domain
    os.environ["AW_RECENT_CAP"] = str(run.cap)
    args = D.build_parser().parse_args([
        "--day", run.as_of, "--window-months", str(run.window_months),
        "--max-child-sitemaps", str(max(200, len(run.seeds) + 50)),
        "--host-rate", str(run.host_rate), "--global-rate", str(run.host_rate),
        "--workers", "1", "--hn-since-days", "1", "--min-rows", "1",
        "--min-publisher-rows", "1", "--no-status"])
    entry = {"as_of": run.as_of, "month": run.month, "publisher": run.publisher,
             "seeds": len(run.seeds), "urls_in": len(run.urls), "cap": run.cap, "host_rate": run.host_rate,
             "started": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    before, t0 = iface_bytes(D.DEFAULT_EGRESS_IFACE), time.monotonic()
    try:
        runner(run.as_of, args)
        entry["status"] = "done"
    except SystemExit as exc:
        entry["status"] = "failed"
        entry["note"] = str(exc)[:300]
    finally:
        for k, v in saved.items():
            if v is None:
                _envnames.pop_knob(os.environ, k)
            else:
                os.environ[k] = v
    elapsed = max(time.monotonic() - t0, 1e-6)
    after = iface_bytes(D.DEFAULT_EGRESS_IFACE)
    if after >= before:
        entry["nic_mbps_whole_card"] = round((after - before) / 1e6 / elapsed, 3)
    fetched = spool_payload_bytes(run.as_of)
    entry["fetched_mb"] = round(fetched / 1e6, 1)
    entry["egress_mbps"] = round(fetched / 1e6 / elapsed, 3)   # this run's own bytes
    entry["elapsed_s"] = round(elapsed)
    pack = pack_manifest(run.as_of)
    if pack:
        entry["rows"] = pack.get("rows")
        entry["urls"] = pack.get("urls")
        entry["tombstones"] = pack.get("tombstones_count")
        entry["overhang_skipped"] = pack.get("overhang_recaptures_skipped")
        entry["undated_dropped"] = pack.get("undated_dropped")
    try:
        staged = D.staged_points(run.as_of)
        entry["staged"] = staged["total"]
    except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
        entry["staged_error"] = repr(exc)[:200]
    entry["finished"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return entry


def adapt_rate(pub: Publisher, last_mbps: float | None, current: float) -> float:
    """Hold the 1 MB/s budget: a run that measured over it slows the next one."""
    if last_mbps is not None and last_mbps > EGRESS_BUDGET_MBPS:
        return round(max(0.2, current * EGRESS_BUDGET_MBPS / last_mbps * 0.9), 2)
    return current


def cmd_smoke(names, state_dir, today, runner=None, iface_bytes=None, generation: int = 1) -> int:
    if runner is None:
        require_expected_hosts_release()
        if any(PUBLISHERS[n].urls_source for n in names):
            require_page_date_clamp_release()
    state = load_state(state_dir)
    ok = True
    for run in plan(names, today=today, smoke=True, state_dir=state_dir, generation=generation):
        set_aside_failed_spool(run, state)
        clash = occupied(run.as_of)
        if clash:
            raise SystemExit(f"{run.key}: as-of day is taken: {clash}")
        entry = execute(run, state_dir=state_dir, runner=runner, iface_bytes=iface_bytes)
        rows = entry.get("rows") or 0
        passed = entry["status"] == "done" and rows >= 1
        state["runs"][run.key] = entry
        state["smoke"][run.publisher] = {
            "status": "PASS" if passed else "FAIL", "rows": rows,
            "as_of": run.as_of, "staged": entry.get("staged"),
            "checked": entry["finished"]}
        save_state(state_dir, state)
        ok &= passed
    return 0 if ok else 1


def mark_interrupted(state: dict) -> list[str]:
    """A run still `running` when a new driver starts belongs to a driver that
    died (one driver owns a state dir). It becomes `failed`, so its spool is set
    aside and its retry starts clean; it is never silently treated as done."""
    marked = []
    for key, r in state.get("runs", {}).items():
        if r.get("status") == "running":
            r["status"] = "failed"
            r["note"] = "interrupted: the driver that owned this run is gone"
            marked.append(key)
    return marked


def cmd_run(names, state_dir, today, runner=None, iface_bytes=None, generation: int = 1) -> int:
    if runner is None:
        require_expected_hosts_release()
        if any(PUBLISHERS[n].urls_source for n in names):
            require_page_date_clamp_release()
    state = load_state(state_dir)
    mark_interrupted(state)
    for name in names:
        if state.get("smoke", {}).get(name, {}).get("status") != "PASS":
            raise SystemExit(f"REFUSING {name}: no passed smoke on record "
                             f"in {state_dir / 'state.json'} -- run `smoke {name}` first")
    state["pid"] = os.getpid()
    state["resume_command"] = resume_command(names, state_dir, generation)
    save_state(state_dir, state)
    failed = 0
    blocked = 0
    for run in plan(names, today=today, state_dir=state_dir, generation=generation):
        prior = state["runs"].get(run.key, {})
        if prior.get("status") == "done":
            continue
        if run.blocked:
            # Recorded with its reason and its url count: a run that cannot start is
            # never skipped silently, and never handed a day the daily timer needs.
            state["runs"][run.key] = {"as_of": run.as_of, "status": "blocked",
                                      "publisher": run.publisher, "urls_in": len(run.urls),
                                      "note": run.blocked}
            save_state(state_dir, state)
            blocked += 1
            continue
        set_aside_failed_spool(run, state)
        clash = occupied(run.as_of)
        if clash:
            state["runs"][run.key] = {"as_of": run.as_of, "status": "failed",
                                      "note": f"as-of day is taken: {clash}"}
            save_state(state_dir, state)
            failed += 1
            continue
        if free_gb() < MIN_FREE_GB:
            state["runs"][run.key] = {
                "as_of": run.as_of, "status": "failed", "publisher": run.publisher,
                "note": f"REFUSING: {free_gb():.0f} GB free on the spool disk, "
                        f"below {MIN_FREE_GB} GB"}
            save_state(state_dir, state)
            failed += 1
            break
        rate = PUBLISHERS[run.publisher].host_rate
        last = [r.get("egress_mbps") for k, r in sorted(state["runs"].items())
                if r.get("publisher") == run.publisher and r.get("egress_mbps")]
        run.host_rate = adapt_rate(PUBLISHERS[run.publisher],
                                   last[-1] if last else None,
                                   state.get("rate", {}).get(run.publisher, rate))
        state.setdefault("rate", {})[run.publisher] = run.host_rate
        state["runs"][run.key] = {"as_of": run.as_of, "status": "running",
                                  "publisher": run.publisher}
        save_state(state_dir, state)
        entry = execute(run, state_dir=state_dir, runner=runner, iface_bytes=iface_bytes)
        state["runs"][run.key] = entry
        save_state(state_dir, state)
        if entry["status"] != "done":
            failed += 1
            if infra_failure(entry.get("note", "")):
                # Stop: the next months would fail on the same dead dependency and
                # be recorded `failed` too (2026-10-01: five months in a row).
                print(f"STOPPING: staging Qdrant down at {run.key}; later runs not "
                      "attempted. Fix the lifecycle, then run the Resume command.",
                      file=sys.stderr)
                break
    state["pid"] = None
    save_state(state_dir, state)
    if blocked:
        print(f"{blocked} run(s) blocked: see state.json (status `blocked`)", file=sys.stderr)
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["plan", "discover", "smoke", "run"])
    ap.add_argument("publishers", help="comma list of: " + ",".join(PUBLISHERS))
    ap.add_argument("--state-dir", default=None,
                    help="state tree (default: the operator setting news_backfill_dir)")
    ap.add_argument("--today", default=_dt.datetime.now(_dt.timezone.utc).date().isoformat())
    ap.add_argument("--only", default="", help="discover: comma list of section paths (smoke)")
    ap.add_argument("--max-pages", type=int, default=0,
                    help="discover: stop a section after N pages (smoke)")
    ap.add_argument("--known-urls", default=None,
                    help="discover rts: urls.jsonl of an earlier pass (a second pass writes "
                         "only the urls the first one missed)")
    ap.add_argument("--generation", type=int, default=1,
                    help="rts: a second discovery pass (new urls only) runs on other as-of "
                         "days than generation 1 (see rts_as_of)")
    a = ap.parse_args(argv)
    a.state_dir = a.state_dir or str(default_state_dir())
    names = [n for n in a.publishers.split(",") if n]
    unknown = [n for n in names if n not in PUBLISHERS]
    if unknown:
        raise SystemExit(f"unknown publisher(s) {unknown}; allowed {sorted(PUBLISHERS)}")
    today = _dt.date.fromisoformat(a.today)
    state_dir = pathlib.Path(a.state_dir)
    if a.cmd == "discover":
        if names == ["swissinfo-tail"]:  # pragma: no cover -- network
            acct = swissinfo_tail_discover(state_dir / "discovery", _live_swissinfo_fetch())
            print(json.dumps({"accounting": acct}), flush=True)
            return 0 if acct["clean"] else 1
        if names == ["watson"]:  # pragma: no cover -- network
            acct = WB.discover(state_dir / "discovery",
                               _live_swissinfo_fetch(host=WB.HOST, min_gap=WB.MIN_GAP_S), today)
            print(json.dumps({"accounting": acct}), flush=True)
            return 0 if acct["clean"] else 1
        if names != ["rts"]:
            raise SystemExit("discover is for rts, watson and swissinfo-tail only (the others have "
                             "sitemaps)")
        from cytir.news import rts_section_discovery as RSD
        return RSD.run_live(state_dir / "discovery", today=today,
                            only=[x for x in a.only.split(",") if x],
                            max_pages=a.max_pages, known_urls=a.known_urls)
    if a.cmd == "plan":
        for r in plan(names, today=today, state_dir=state_dir, generation=a.generation):
            print(f"{r.key:24s} as_of={r.as_of or '-':10s} window={r.window_months} "
                  f"seeds={len(r.seeds)} urls={len(r.urls)} cap={r.cap} "
                  f"taken={occupied(r.as_of) if r.as_of else '-'}"
                  + (f" BLOCKED: {r.blocked}" if r.blocked else ""))
        return 0
    if a.cmd == "smoke":
        return cmd_smoke(names, state_dir, today, generation=a.generation)
    return cmd_run(names, state_dir, today, generation=a.generation)


if __name__ == "__main__":
    raise SystemExit(main())
