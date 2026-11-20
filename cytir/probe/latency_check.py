#!/usr/bin/env python3
"""cytir/probe/latency_check.py — task 362 (ratified Tim 2026-07-30, task 361 T6):
novel-query latency sentinel, folded into the standing probe loop.

WHY: three separate times this project accepted a "the index grew, so it's
slower" narrative for novel-query latency (07-06, 07-21, 07-27) — each time
Tim noticed the slowdown manually before the loop did, and each time a
mechanism existed (postings residency, an unset AW_QDRANT_TIMEOUT, concurrent
bulk writes). This check makes the threshold self-surfacing instead of
depending on a founder spot-check: it computes the novel-query (Slice A) p50
latency for the day's run and compares it to Tim's stated band (repeated 4x:
novel-query p50 should be <= ~1s).

Mechanical, no LLM, offline over the run's results file:

  python3 -m cytir.probe.latency_check --date 2026-07-30
    reads  probes/runs/<date>/results.jsonl        (Slice A only — "novel")
    writes probes/runs/<date>/latency_check.json

Config (the probe runner's latency-threshold config, per the fold-in spec):
  NOVEL_P50_THRESHOLD_S = 1.0   (Tim's stated band; override via --threshold)

If p50 exceeds the threshold, this is flagged loudly (exit code 1 + an email
alert, same channel as the P0 path) so investigation starts before the rest
of the day's work, per the proposal's own wording ("investigate-to-mechanism
before the day's other work"). The probe report/reviewer summary always
states the novel p50, exceeded or not.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Alert recipient and sender. Env-backed, default empty: the deploy shell
# (deploy/scripts/probe-run.sh) sets both. Empty = no mail (task 1521).
ALERT_TO = os.environ.get("AW_PROBE_ALERT_TO", "")
ALERT_FROM = os.environ.get("AW_PROBE_ALERT_FROM", "")
RUNS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "probes", "runs")

NOVEL_P50_THRESHOLD_S = 1.0  # Tim's stated band, repeated 4x (07-06, 07-21, 07-27, 07-30)


def percentile50(values):
    if not values:
        return None
    return statistics.median(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--threshold", type=float, default=NOVEL_P50_THRESHOLD_S)
    ap.add_argument("--runs-dir", default=RUNS)
    ap.add_argument("--no-mail", action="store_true",
                     help="compute + write only, skip the alert email (for dry-runs)")
    args = ap.parse_args()

    rundir = os.path.join(args.runs_dir, args.date)
    path = os.path.join(rundir, "results.jsonl")  # Slice A = novel every run
    if not os.path.exists(path):
        raise SystemExit(f"no {path} — refusing to write an empty report "
                          "(an error is not an empty)")

    lat = []
    for line in open(path):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("outcome") in ("timeout", "error", "not-run"):
            continue                    # task 766: a timeout is not a latency sample
        v = rec.get("latency_s")
        if isinstance(v, (int, float)):
            lat.append(v)

    if not lat:
        raise SystemExit(f"{path} had no latency_s values — refusing to write "
                          "an empty report (an error is not an empty)")

    p50 = percentile50(lat)
    exceeded = p50 > args.threshold
    out = {
        "date": args.date,
        "n": len(lat),
        "novel_p50_s": round(p50, 3),
        "threshold_s": args.threshold,
        "exceeded": exceeded,
    }
    out_path = os.path.join(rundir, "latency_check.json")
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"latency check: novel-query p50={p50:.3f}s over n={len(lat)} "
          f"(threshold {args.threshold}s) -> {out_path}"
          + (" *** EXCEEDED ***" if exceeded else ""))

    if exceeded and not args.no_mail:
        body = (
            f"Novel-query (Slice A) p50 latency for {args.date} = {p50:.3f}s, "
            f"above the {args.threshold}s band (Tim's stated threshold, repeated "
            f"4x: 07-06, 07-21, 07-27, 07-30).\n\n"
            f"n={len(lat)} queries. Per doctrine, this is a regression with a "
            f"mechanism, not a narrative — investigate to a mechanism (config "
            f"knob, residency change, contaminated measurement window, unset "
            f"timeout) BEFORE the day's other work.\n\n"
            f"See probes/runs/{args.date}/latency_check.json and "
            f"probes/runs/{args.date}/results.jsonl.\n"
        )
        if not ALERT_TO or not ALERT_FROM:
            print("latency_check: AW_PROBE_ALERT_TO/AW_PROBE_ALERT_FROM unset; no alert mail sent", file=sys.stderr)
            if exceeded:
                sys.exit(1)
            return
        msg = (
            f"To: {ALERT_TO}\n"
            f"From: {ALERT_FROM}\n"
            f"Subject: [AW] novel-query p50 {p50:.3f}s exceeds {args.threshold}s band ({args.date})\n\n"
            f"{body}"
        )
        try:
            subprocess.run(["msmtp", "-a", "default", ALERT_TO],
                            input=msg.encode(), check=True)
            print("latency check: ALERT MAILED")
        except Exception as e:  # fail loud, but don't crash the probe chain
            print(f"latency check: ALERT MAIL FAILED: {e}", file=sys.stderr)

    if exceeded:
        sys.exit(1)


if __name__ == "__main__":
    main()
