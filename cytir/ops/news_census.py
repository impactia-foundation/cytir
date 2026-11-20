"""cytir.ops.news_census -- the exhaustive news coverage census (task 676).

READ-ONLY. Uses only Qdrant's facet and count endpoints (indexed keys `domain`
and `published`), so it never scrolls and is safe against the twin.

For every domain on the roster it reports, per calendar month of the last
`--months`+1 months: the rows published in that month; and per domain: total
rows, rows with a `published` value, and the `published` FORMAT CLASS
(iso_date / iso_year_month / iso_year / iso_datetime / rfc822 /
dotted_dmy / other / missing).

The roster is never a hand list (corpus-doctrine 8). It is the union of

  * `declared`   -- the news pipeline's own publisher list
                    (`data/news-queue.example.json`), every source pack whose
                    `layer` is `news`, every `canonical_domain` of the freshness
                    refresh packs (`sources/**/*-tracked.yaml`), and the two HN
                    arms (`news.ycombinator.com`);
  * `observed`   -- the store itself: every domain that holds at least one row
                    whose `published` is an ISO date inside the census window
                    (facet over `domain`, filtered on the window's date strings);
  * `--extra`    -- domains a person asks about; printed as such.

A domain is matched under both `d` and `www.d` (the join is probed in both
spellings and each spelling's count is reported: corpus-doctrine 5).

    python -m cytir.ops.news_census --url <twin-url> \
        --collection <twin-collection> --today 2026-09-30 --json out.json --markdown out.md
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import collections
import datetime as dt
import json
import re
import pathlib
import sys
import urllib.request

from cytir.core import twin as TWIN
from cytir.core.published import classify_published, normalize_published
from cytir.news.source_queue import QUEUE_NAME, load_queue

REPO = pathlib.Path(__file__).resolve().parents[2]

FORMATS = ("iso_date", "iso_year_month", "iso_year", "iso_datetime", "rfc822",
           "dotted_dmy", "long_date", "other", "missing")
classify = classify_published  # ONE classifier: cytir.core.published


def month_of(value: str) -> str | None:
    """`YYYY-MM` of a published value, or None when it has no month (a bare
    year) or cannot be read. Uses the contract normaliser, so a datetime, an
    RFC-822 and a dd.mm.yyyy value are bucketed by their UTC day."""
    got = normalize_published(value)
    if got is None or len(got) < 7:
        return None
    return got[:7]


# ------------------------------------------------------------------- roster
def _bare(host: str) -> str:
    h = host.strip().lower().split("/")[0].split("?")[0]
    return h[4:] if h.startswith("www.") else h


def declared_roster(repo: pathlib.Path = REPO) -> dict[str, list[str]]:
    """domain -> the declarations that name it."""
    out: dict[str, list[str]] = collections.defaultdict(list)
    for row in load_queue(repo):
        out[_bare(row["domain"])].append(QUEUE_NAME)
    for path in sorted((repo / "sources").rglob("*.yaml")):
        text = path.read_text()
        if re.search(r"^layer:\s*news\s*$", text, re.M):
            for m in re.finditer(r"^\s*-\s*host:\s*(\S+)", text, re.M):
                out[_bare(m.group(1))].append(str(path.relative_to(repo)))
        if path.name.endswith("-tracked.yaml"):
            for m in re.finditer(r"^\s*canonical_domain:\s*(\S+)", text, re.M):
                out[_bare(m.group(1))].append(str(path.relative_to(repo)))
    out["news.ycombinator.com"].append(
        "HN arms (hn_freshness_ingest + hn_discussion_daily)")
    return dict(out)


# -------------------------------------------------------------------- qdrant
class Twin:
    def __init__(self, url: str, collection: str, timeout: float = 120.0):
        self.url, self.collection, self.timeout = url.rstrip("/"), collection, timeout

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.url}/collections/{self.collection}{path}",
            data=json.dumps(body).encode(), method="POST",
            headers={"content-type": "application/json"})
        with IH.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["result"]

    def count(self, must: list[dict]) -> int:
        return self._post("/points/count", {"filter": {"must": must},
                                            "exact": True})["count"]

    def facet(self, key: str, must: list[dict], limit: int) -> list[dict]:
        return self._post("/facet", {"key": key, "filter": {"must": must},
                                     "limit": limit, "exact": True})["hits"]


def _domain_filter(domains: list[str]) -> list[dict]:
    return [{"key": "domain", "match": {"any": domains}}]


def window_months(today: dt.date, months: int) -> list[str]:
    """The `months`+1 calendar months ending with `today`'s month, oldest first."""
    y, m = today.year, today.month
    out = []
    for _ in range(months + 1):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return out[::-1]


def observed_roster(tw: Twin, today: dt.date, months: int, top: int) -> list[dict]:
    first = dt.date.fromisoformat(window_months(today, months)[0] + "-01")
    days = [(first + dt.timedelta(days=i)).isoformat()
            for i in range((today - first).days + 2)]
    must = [{"key": "published", "match": {"any": days}}]
    return tw.facet("domain", must, top)


def census_domain(tw: Twin, domain: str, months: list[str], today: dt.date) -> dict:
    variants = [domain, f"www.{domain}"]
    per_variant = {v: tw.count(_domain_filter([v])) for v in variants}
    total = sum(per_variant.values())
    fmt = collections.Counter()
    by_month = collections.Counter()
    older = future = no_month = 0
    with_published = 0
    if total:
        hits = tw.facet("published", _domain_filter(variants), 1_000_000)
        for h in hits:
            n, val = h["count"], h["value"]
            with_published += n
            c = classify(val)
            fmt[c] += n
            mo = month_of(val)
            if mo is None:
                no_month += n
            elif mo in months:
                by_month[mo] += n
            elif mo < months[0]:
                older += n
            else:
                future += n
    fmt["missing"] = total - with_published
    return {"domain": domain, "rows": total, "per_variant": per_variant,
            "with_published": with_published, "formats": dict(fmt),
            "by_month": dict(by_month), "older_than_window": older,
            "after_window": future, "no_month": no_month}


def run(tw: Twin, today: dt.date, months: int, extra: list[str], top: int,
        repo: pathlib.Path = REPO) -> dict:
    declared = declared_roster(repo)
    obs = observed_roster(tw, today, months, top)
    observed = {}
    for h in obs:
        observed[_bare(h["value"])] = observed.get(_bare(h["value"]), 0) + h["count"]
    roster: dict[str, dict] = {}
    for d, why in declared.items():
        roster[d] = {"declared_by": sorted(set(why))}
    for d in extra:
        roster.setdefault(_bare(d), {})["extra"] = True
    for d, n in observed.items():
        roster.setdefault(d, {})["observed_dated_rows_in_window"] = n
    win = window_months(today, months)
    rows = []
    for d in sorted(roster):
        r = census_domain(tw, d, win, today)
        r.update(roster[d])
        rows.append(r)
    return {"schema": "aw.news-census/v1", "collection": tw.collection,
            "today": today.isoformat(), "window": win,
            "roster": {"declared": sorted(declared),
                       "extra": sorted(_bare(d) for d in extra),
                       "observed_top": top, "observed_domains": len(observed)},
            "domains": rows}


def to_markdown(doc: dict, min_rows: int = 0) -> str:
    win = doc["window"]
    lines = [f"Census of `{doc['collection']}` on {doc['today']}. Window: "
             f"{win[0]} .. {win[-1]} ({len(win)} calendar months).", ""]
    cols = ["iso_date", "iso_datetime", "iso_year", "rfc822", "dotted_dmy", "long_date", "other", "missing"]
    lines.append("| domain | rows | published | " + " | ".join(cols) + " | older | "
                 + " | ".join(m[2:] for m in win) + " |")
    lines.append("|" + "---|" * (4 + len(cols) + len(win)))
    for r in sorted(doc["domains"], key=lambda r: -r["rows"]):
        if r["rows"] < min_rows and not r.get("declared_by") and not r.get("extra"):
            continue
        f = r["formats"]
        mc = " | ".join(str(r["by_month"].get(m, 0)) for m in win)
        fc = " | ".join(str(f.get(c, 0)) for c in cols)
        lines.append(f"| {r['domain']} | {r['rows']} | {r['with_published']} | "
                     f"{fc} | {r['older_than_window']} | {mc} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--collection", default=TWIN.collection())
    ap.add_argument("--today", default=dt.datetime.now(dt.timezone.utc).date().isoformat())
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--top", type=int, default=200,
                    help="observed roster size (domains with dated rows in the window)")
    ap.add_argument("--extra", default="")
    ap.add_argument("--json")
    ap.add_argument("--markdown")
    ap.add_argument("--min-rows", type=int, default=500,
                    help="markdown: hide observed-only domains below this many rows")
    a = ap.parse_args(argv)
    doc = run(Twin(a.url, a.collection), dt.date.fromisoformat(a.today), a.months,
              [d for d in a.extra.split(",") if d.strip()], a.top)
    print("ROSTER declared:", ", ".join(doc["roster"]["declared"]), file=sys.stderr)
    print("ROSTER extra:", ", ".join(doc["roster"]["extra"]) or "-", file=sys.stderr)
    print("ROSTER observed:", doc["roster"]["observed_domains"], "domains", file=sys.stderr)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(doc, indent=1))
    md = to_markdown(doc, a.min_rows)
    if a.markdown:
        pathlib.Path(a.markdown).write_text(md)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
