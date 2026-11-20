#!/usr/bin/env python3
"""cytir/probe/sample_sliceb.py — the SAMPLER for SLICE B of the two-slice probe loop
(task 269).

Slice B is AW's TARGET-MATCHED, CH-FIRST authoritative panel: statute / spec /
gov-stat / entity / authoritative-fact lookups in EN/DE/FR/IT, drawn from the
VERSIONED pool probes/slice_b_pool.jsonl (see probes/build_slice_b_pool.py).
Unlike Slice A (cytir/probe/sample.py — verbatim search-arena, novel every run,
deduped against seen.txt), Slice B:

  * draws from a CURATED, checked-in pool — auditable, not random web queries;
  * INTENTIONALLY REPEATS — it does NOT touch seen.txt. Re-probing the same
    authoritative lookups day over day is how a regression on a core path
    surfaces (a sentinel that was green yesterday and red today). Slice A owns
    novelty/surprise; Slice B owns longitudinal regression watch;
  * is CH-FIRST BY CONSTRUCTION every day: all `core` sentinels (mostly CH) run
    daily, plus a date-rotated remainder drawn stratified CH-heavy so no single
    day drifts off the Swiss-first skew.

Selection each run (deterministic by --date):
  * ALL core=true rows (the daily sentinels), plus
  * a date-rotated window of CH non-core rows (N_CH_ROTATE), plus
  * a date-rotated window of non-CH non-core rows (N_OTHER_ROTATE).
Windows slide by date so the whole pool is exercised over a few days while each
day stays CH-first and comparable in size to Slice A (~20).

No network, no LLM, no prod calls. Output:
    probes/runs/<date>/sample_sliceb.jsonl

Usage:
    python3 -m cytir.probe.sample_sliceb                    # date=today
    python3 -m cytir.probe.sample_sliceb --date 2026-07-18
    python3 -m cytir.probe.sample_sliceb --n-ch-rotate 7 --n-other-rotate 5
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PROBES = os.path.join(REPO, "probes")
POOL = os.path.join(PROBES, "slice_b_pool.jsonl")

# CH-first ordering (Tim): heaviest CH, then FR/DE/CA, then AI/software/international.
N_CH_ROTATE = 6      # CH non-core queries added per day (on top of core)
N_GEO_ROTATE = 4     # FR/DE/CA non-core queries added per day
N_REST_ROTATE = 2    # ai_ml/software/international non-core queries added per day
GEO_DOMAINS = {"france", "germany", "canada"}


def load_pool():
    if not os.path.exists(POOL):
        sys.exit(f"no Slice-B pool: {POOL} (run probes/build_slice_b_pool.py)")
    rows = []
    with open(POOL, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def window(items, offset, n):
    """Deterministic sliding window of n items starting at offset (wraps)."""
    if not items or n <= 0:
        return []
    k = len(items)
    n = min(n, k)
    start = (offset * n) % k
    return [items[(start + i) % k] for i in range(n)]


def rotation_offset(date, seed=None):
    """The rotation offset: --seed when given (reproducibility), else the ordinal of the UTC day, so
    the draw changes every day and is the same for the same day (task 2041; probe 2033 re-measured
    one fixed set)."""
    return seed if seed is not None else datetime.date.fromisoformat(date).toordinal()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
                    help="UTC day (default today, UTC); the rotation offset is derived from it")
    ap.add_argument("--seed", type=int, default=None,
                    help="rotation offset override (reproduce any past draw); default = date ordinal")
    ap.add_argument("--n-ch-rotate", type=int, default=N_CH_ROTATE)
    ap.add_argument("--n-geo-rotate", type=int, default=N_GEO_ROTATE)
    ap.add_argument("--n-rest-rotate", type=int, default=N_REST_ROTATE)
    args = ap.parse_args()

    pool = load_pool()
    offset = rotation_offset(args.date, args.seed)

    core = [r for r in pool if r.get("core")]
    noncore = [r for r in pool if not r.get("core")]
    # stable ordering so the rotation is reproducible independent of file order
    def strat(pred):
        return sorted([r for r in noncore if pred(r)], key=lambda r: r["qid"])
    ch = strat(lambda r: r.get("target_domain") == "switzerland")
    geo = strat(lambda r: r.get("target_domain") in GEO_DOMAINS)       # FR/DE/CA
    rest = strat(lambda r: r.get("target_domain") not in ({"switzerland"} | GEO_DOMAINS))

    picked = list(core)
    picked += window(ch, offset, args.n_ch_rotate)
    picked += window(geo, offset, args.n_geo_rotate)
    picked += window(rest, offset, args.n_rest_rotate)

    rows = []
    for i, r in enumerate(picked):
        row = dict(r)
        row["probe_id"] = f"{args.date}-b{i:03d}"
        row["slice"] = "B"
        row["origin"] = r.get("source")            # arena | synth
        row["dataset"] = "slice-b-pool"
        row["focus_topic"] = r.get("target_domain")
        row["template_id"] = r.get("qid")
        row["stratum"] = r.get("targets")          # fragile classes -> reviewer 'targets'
        # baseline params; cytir/probe/run.py draws the real per-probe profile (k/lang/country)
        row["k"] = 10
        row["params"] = {"k": 10, "authorities": 1}
        if r.get("country"):
            row["params"]["country"] = r["country"]
        rows.append(row)

    outdir = os.path.join(PROBES, "runs", args.date)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "sample_sliceb.jsonl")
    with open(outpath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    import collections
    by_dom = collections.Counter(r["target_domain"] for r in rows)
    by_lang = collections.Counter(r["lang"] for r in rows)
    by_diff = collections.Counter(r["difficulty"] for r in rows)
    n_ch = sum(1 for r in rows if r["target_domain"] == "switzerland")
    print(f"cytir.probe.sample_sliceb: date={args.date} offset={offset}", file=sys.stderr)
    print(f"WROTE {len(rows)} Slice-B queries -> {outpath}", file=sys.stderr)
    print(f"  core={len(core)} ch-rotate={args.n_ch_rotate} geo-rotate={args.n_geo_rotate} "
          f"rest-rotate={args.n_rest_rotate}", file=sys.stderr)
    print(f"  CH-first: {n_ch}/{len(rows)} ({100*n_ch//max(1,len(rows))}%) switzerland",
          file=sys.stderr)
    print(f"  by target_domain: {dict(by_dom)}", file=sys.stderr)
    print(f"  by lang: {dict(by_lang)}  by difficulty: {dict(by_diff)}", file=sys.stderr)


if __name__ == "__main__":
    main()
