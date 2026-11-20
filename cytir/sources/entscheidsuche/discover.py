"""Discovery (`toc`) for the `scratch_entscheidsuche_exact` family (task 1486).

Reads the entscheidsuche.ch sitemap index, applies the per-spider P0 year windows (`SLICE`),
collapses BGE date-variants, and writes the work queue (`queue.jsonl`, one
`{spider, viewid, year, json_url, view_url}` per line) that the family adapter
(`cytir.sources.families.entscheidsuche`) takes as its `root` option. The `toc` body is moved
unchanged out of the former ingest_entscheidsuche driver (now a shim), which forwards here.

    python3 -m cytir.sources.entscheidsuche.discover toc --out queue.jsonl [--all-years] [--spiders A,B]
    python3 -m cytir.sources.entscheidsuche.discover fetch --queue queue.jsonl --out es.jsonl

The old `fetch` loop is REPLACED by the adapter: `fetch` runs the `scratch_entscheidsuche_exact`
family (one decision JSON per queue line; 404 and no-headnote units end `excluded`, HTTP/parse
failures `quarantined`) and appends the pack rows to `--out`. A pack run never overwrites a pack,
so the old `--state` resume file is accepted and unused: re-run into a fresh `--out`.

P0 slice (task 136): CH_BGE all + CH_BGer last-3y + CH_BSTG last-10y + CH_BPatG all +
CH_BVGer last-2y + cantonal-supreme current year. `--all-years` / `--spiders` override.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

from cytir.core.user_agent import user_agent
from cytir.extract.entscheidsuche import dedup_base, spider_of_sitemap, year_of

BASE = "https://entscheidsuche.ch"
SITEMAP_INDEX = f"{BASE}/sitemaps/sitemapindex.xml"
UA = {"User-Agent": user_agent()}

# --- P0 slice: spider -> min year kept (0 = every year) ----------------------
# Federal courts deep; cantonal-supreme a recent starter slice (fuller cantonal =
# round 2). Windows chosen to land ~20-60k passages after the headnote gate.
SLICE = {
    # Federal
    "CH_BGE": 0,        # leading decisions (Regeste) — ALL, the crown jewel
    "CH_BGer": 2024,    # Federal Supreme Court, last ~2y (~7.5k/yr)
    "CH_BSTG": 2016,    # Federal Criminal Court (~all recent, small)
    "CH_BPatG": 0,      # Federal Patent Court — ALL (229)
    "CH_BVGer": 2025,   # Federal Administrative Court, recent (~4.5k/yr)
    # Cantonal supreme / high courts — current-year starter slice (de/fr/it).
    # Fuller cantonal depth = round 2; kept small here to respect the cap and the
    # non-profit server (courts corpora are >1M decisions total).
    "ZH_Obergericht": 2026,
    "ZH_Verwaltungsgericht": 2026,
    "BE_Verwaltungsgericht": 2026,
    "BE_ZivilStraf": 2026,
    "GE_Gerichte": 2026,        # Cour de justice (fr)
    "VD_FindInfo": 2026,        # Tribunal cantonal VD (fr)
    "TI_Gerichte": 2026,        # Tribunale d'appello (it)
    "LU_Gerichte": 2026,
    "SG_Gerichte": 2026,
    "BS_Omni": 2026,
}


def cmd_toc(args):
    client = make_client(HttpSpec(timeout=120, headers={"user-agent": UA["User-Agent"]}, follow_redirects=True))
    spiders = set(args.spiders.split(",")) if args.spiders else set(SLICE)
    print(f"sitemap index {SITEMAP_INDEX}", flush=True)
    r = client.get(SITEMAP_INDEX)
    r.raise_for_status()
    parts = defaultdict(list)
    for loc in re.findall(r"<loc>([^<]+)</loc>", r.text):
        sp = spider_of_sitemap(loc)
        if sp and sp in spiders:
            parts[sp].append(loc)

    # gather + slice
    per_spider = Counter()
    best: dict[str, dict] = {}   # dedup_base -> best queue rec
    for sp in sorted(parts):
        minyear = 0 if args.all_years else SLICE.get(sp, 0)
        got = 0
        for part in sorted(parts[sp]):
            try:
                pr = client.get(part)
                pr.raise_for_status()
            except Exception as e:  # noqa: BLE001
                print(f"  WARN {part}: {e}", flush=True)
                continue
            for loc in re.findall(r"<loc>([^<]+)/view/([^<]+)</loc>", pr.text):
                viewid = loc[1]
                yr = year_of(viewid)
                if minyear and yr and yr < minyear:
                    continue
                rec = {"spider": sp, "viewid": viewid, "year": yr,
                       "json_url": f"{BASE}/docs/{sp}/{viewid}.json",
                       "view_url": f"{BASE}/view/{viewid}"}
                base = f"{sp}/{dedup_base(viewid)}"
                cur = best.get(base)
                # prefer a full-date variant (year>0) over year-only/nodate
                if cur is None or (yr and not cur["year"]):
                    best[base] = rec
                got += 1
        per_spider[sp] = got
        print(f"  {sp}: {got} in-slice urls (minyear={minyear})", flush=True)

    kept = list(best.values())
    kc = Counter(r["spider"] for r in kept)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"QUEUE {len(kept)} decisions after date-variant dedup -> {args.out}", flush=True)
    print(f"  by spider: {dict(kc)}", flush=True)


def cmd_fetch(args):
    """Replaces the legacy fetch loop: run the family adapter, append its rows to `--out`."""
    from cytir.sources import run_family
    if args.state:
        print("--state is accepted for CLI compatibility and ignored: the family run appends to --out; "
              "re-run into a fresh --out", file=sys.stderr, flush=True)
    out = Path(args.out).resolve()
    run_dir = Path(tempfile.mkdtemp(prefix="entscheidsuche-fetch-", dir=out.parent))
    run_family.run_family("scratch_entscheidsuche_exact", out_dir=run_dir / "o",
                          today=datetime.date.today().isoformat(),
                          options={"root": args.queue, "rate": str(args.rate)})
    _, rows = run_family.load_pack(run_dir / "o" / "pack")
    with open(out, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"FETCH done rows={len(rows)} out={out} pack={run_dir / 'o' / 'pack'}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("toc")
    t.add_argument("--out", required=True)
    t.add_argument("--all-years", action="store_true", help="ignore year windows")
    t.add_argument("--spiders", default="", help="comma list override (default SLICE keys)")
    t.set_defaults(func=cmd_toc)
    f = sub.add_parser("fetch")
    f.add_argument("--queue", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--state", default="", help="accepted for the old CLI; unused (see module doc)")
    f.add_argument("--rate", type=float, default=4.0)
    f.set_defaults(func=cmd_fetch)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
