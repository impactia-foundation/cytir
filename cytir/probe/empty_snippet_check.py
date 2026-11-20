#!/usr/bin/env python3
"""cytir/probe/empty_snippet_check.py — task 349: per-domain empty-snippet-rate check
for the standing probe loop.

WHY (log 2026-07-28, dardagny class): extraction failures at ingest leave
title-only points whose empty snippets survive honestly into serving. A user
(or agent) then gets a result with a title and NO snippet — invisible in
aggregate metrics, obvious per domain. Tim's spot-check found dardagny.ch at
48/60 snippet-less; the 349 master scan found 1.17M such points across 1,654
domains. This check makes the class self-surfacing: every probe run reports
which domains RETURNED empty-snippet results, so a regression (a broken
extractor wave, a bad pack) shows up the next morning instead of at the next
manual spot-check.

Mechanical, no LLM, offline over the run's results files:

  python3 -m cytir.probe.empty_snippet_check --date 2026-07-29
    reads  probes/runs/<date>/results.jsonl + results_sliceb.jsonl
    writes probes/runs/<date>/empty_snippet_domains.json

Flag rules (tuned to be quiet unless actionable):
  * flagged        — domain returned >= --min-results results and
                     empty_rate >= --rate (defaults 3 / 0.5)
  * rank_visible   — ANY empty-snippet result at rank <= 3 (a user saw it
                     prominently, even once)
The reviewer reads the file (REVIEW_PROMPT.md step for it); a flagged domain
routes per the loop's triage: bug task (extractor/ingest) or repair-lane wave
(task 349 pattern: refetch + supersede).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "probes", "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--min-results", type=int, default=3)
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--runs-dir", default=RUNS)
    args = ap.parse_args()

    rundir = os.path.join(args.runs_dir, args.date)
    stats = defaultdict(lambda: {"results": 0, "empty": 0, "best_empty_rank": None,
                                 "example": None})
    n_files = 0
    for fname in ("results.jsonl", "results_sliceb.jsonl"):
        path = os.path.join(rundir, fname)
        if not os.path.exists(path):
            continue
        n_files += 1
        for line in open(path):
            if not line.strip():
                continue
            rec = json.loads(line)
            for r in rec.get("results") or []:
                url = r.get("url") or ""
                host = url.split("/")[2].lower() if "://" in url else ""
                if host.startswith("www."):
                    host = host[4:]
                if not host:
                    continue
                s = stats[host]
                s["results"] += 1
                if not (r.get("snippet") or "").strip():
                    s["empty"] += 1
                    rank = r.get("rank")
                    if rank and (s["best_empty_rank"] is None
                                 or rank < s["best_empty_rank"]):
                        s["best_empty_rank"] = rank
                        s["example"] = url
    if n_files == 0:
        raise SystemExit(f"no results files under {rundir} — refusing to write "
                         "an empty report (an error is not an empty)")

    flagged, rank_visible = [], []
    for host, s in sorted(stats.items()):
        if s["empty"] == 0:
            continue
        rate = s["empty"] / s["results"]
        row = {"domain": host, "results": s["results"], "empty": s["empty"],
               "rate": round(rate, 3), "best_empty_rank": s["best_empty_rank"],
               "example": s["example"]}
        if s["results"] >= args.min_results and rate >= args.rate:
            flagged.append(row)
        elif s["best_empty_rank"] is not None and s["best_empty_rank"] <= 3:
            rank_visible.append(row)

    out = {"date": args.date, "files_read": n_files,
           "domains_seen": len(stats),
           "domains_with_empty": sum(1 for s in stats.values() if s["empty"]),
           "flagged": sorted(flagged, key=lambda r: -r["rate"]),
           "rank_visible": sorted(rank_visible,
                                  key=lambda r: r["best_empty_rank"])}
    out_path = os.path.join(rundir, "empty_snippet_domains.json")
    json.dump(out, open(out_path, "w"), indent=1, ensure_ascii=False)
    print(f"empty-snippet check: {out['domains_with_empty']} domains with an "
          f"empty snippet, {len(flagged)} flagged, {len(rank_visible)} "
          f"rank-visible -> {out_path}")


if __name__ == "__main__":
    main()
