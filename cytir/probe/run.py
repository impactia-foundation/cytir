#!/usr/bin/env python3
"""cytir/probe/run.py — the RUNNER leg of the standing random-probe quality loop
(task 173; parameter-profile randomization added in task 200).

Runs each sampled query against LIVE PROD (the websear.ch backend on the serving box,
localhost:8090), POLITELY sequential (~1 query/second), and captures the full
results + authority-card JSON to
    probes/runs/<date>/results.jsonl

PARAM PROFILES (task 200): real users do not hit one fixed profile — k=10 is
the unified default (web + API/MCP after task 199), the page dropdown offers
3/5/10/20/50, lang is usually UNSET, the page sends authorities=1 while the
MCP tool doesn't. Probing only k=10/lang=en/authorities=1 graded a different
ranking reality than users saw (and masked the task-199 k-instability hole).
Each probe now DRAWS its profile, seeded by the probe's STABLE IDENTITY (task
766): the pool `qid` for Slice B, the normalized query text for Slice A -- never
the date, never the per-day index in probe_id. So a repeating Slice-B sentinel
runs under the SAME profile every day and two days compare like with like. The
seed key is recorded as `params_seed` next to `params` in every results record;
the reviewer grades against it (probes/REVIEW_PROMPT.md).

EVERY query gets an `outcome` (ok / empty / timeout / error / not-run) and the
run writes `run_summary[_sliceb].json` (expected vs recorded, per-outcome counts,
gaps). A run that did not record every sampled query says so; `--check` re-derives
that from the files alone (it also catches a SIGKILLed run, which no handler can).

READ-ONLY: only issues GET /search. Never writes to prod.

Environment (defaults keep today's behaviour):
    AW_PROBE_SSH_HOST      ssh alias of the box that serves prod        (default: none; probe-run.sh sets it)
    AW_PROBE_PROD_API_URL  prod API base URL, as seen FROM that box     (default: http://localhost:8090)
    AW_PROBE_ALERT_TO      latency alert recipient (cytir.probe.latency_check)  (default empty; the deploy shell sets it)
    AW_PROBE_ALERT_FROM    latency alert sender (default empty; the deploy shell sets it)

Usage:
    python3 -m cytir.probe.run                      # date=today, ssh $AW_PROBE_SSH_HOST
    python3 -m cytir.probe.run --date 2026-07-11
    python3 -m cytir.probe.run --api http://localhost:8090   # direct (on the box)
    python3 -m cytir.probe.run --sleep 1.0
    python3 -m cytir.probe.run --fixed-params --k 10  # legacy fixed profile
"""
from __future__ import annotations
from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import datetime
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
# Where the probe reaches prod. Env-backed, defaults = the live values (task 1521).
SSH_HOST = os.environ.get("AW_PROBE_SSH_HOST", "")
PROD_API_URL = os.environ.get("AW_PROBE_PROD_API_URL", "http://localhost:8090").rstrip("/")
REPO = os.path.dirname(os.path.dirname(HERE))
PROBES = os.path.join(REPO, "probes")

RESULT_FIELDS = ["url", "anchor", "title", "snippet", "source_type", "lang",
                 "trust_tier", "last_seen", "score"]


def build_qs(query, params, k):
    qs = {"q": query, "k": params.get("k", k)}
    for p in ("lang", "country"):
        if params.get(p):
            qs[p] = params[p]
    if params.get("authorities"):
        qs["authorities"] = params["authorities"]
    return urllib.parse.urlencode(qs)


# ------------------------------------------- parameter profiles (task 200) --
# k is drawn from the real user-reachable set. 10 is the UNIFIED default
# (web page default; task 199 flips the API/MCP default 8->10 too) and gets
# the most weight. 3/5 are page-dropdown options and, with 8 (legacy MCP
# default), carry deliberate mass: small k is where the task-199 recall hole
# lived — probes MUST keep exercising it so a regression can't hide.
# 20/50 are the dropdown's deep options and the k-instability sentinels.
K_CHOICES = [(3, 12), (5, 12), (8, 16), (10, 30), (20, 15), (50, 15)]
P_LANG_SET = 0.40        # else unset — the dominant real-user state
P_AUTHORITIES_ON = 0.60  # web page always sends 1; MCP sends nothing
P_COUNTRY_SET = 0.35     # only when the query is jurisdiction-flavored


def _weighted(rng, choices):
    total = sum(w for _, w in choices)
    x = rng.random() * total
    for v, w in choices:
        x -= w
        if x <= 0:
            return v
    return choices[-1][0]


# Bump to deliberately re-draw every profile (recorded in params_seed).
PROFILE_EPOCH = 2


def _norm_query(q):
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def params_seed(row):
    """The stable identity a probe's profile is seeded by (task 766).

    Slice B: the pool qid (the same query repeats daily -> same profile daily).
    Slice A: the normalized query text (novel every run, but a re-sample or a
    re-run that shifts the per-day index must not change the query's profile).
    Never the date and never the probe_id index: that made every day's draw
    differ and day-over-day diffs incomparable.
    """
    qid = row.get("qid") or row.get("template_id") if row.get("slice") == "B" else None
    if qid:
        return f"aw-probe-params:v{PROFILE_EPOCH}:qid:{qid}"
    return f"aw-probe-params:v{PROFILE_EPOCH}:q:{_norm_query(row.get('query'))}"


def draw_profile(row):
    """Deterministic per-probe parameter profile, a pure function of
    params_seed(row): fixed per probe identity, independent of the date."""
    rng = random.Random(zlib.crc32(params_seed(row).encode()))
    profile = {"k": _weighted(rng, K_CHOICES)}
    # lang: usually UNSET (site default). When set, match the query's own
    # language — never probe a de query with lang=fr; that's a user error,
    # not a ranking reality worth grading.
    qlang = (row.get("lang") or "").lower()
    # Slice A keeps its exact historical gate (en/de/fr); Slice B adds `it`
    # so the CH/IT authoritative panel can exercise Italian language-leak.
    allowed = ("en", "de", "fr", "it") if row.get("slice") == "B" else ("en", "de", "fr")
    if qlang in allowed and rng.random() < P_LANG_SET:
        profile["lang"] = qlang
    profile["authorities"] = 1 if rng.random() < P_AUTHORITIES_ON else 0
    # country: occasionally, and ONLY for jurisdiction-flavored queries (the
    # sampler attaches country on those) — mirrors an explicit user selection.
    country = row.get("country") or (row.get("params") or {}).get("country")
    if country and rng.random() < P_COUNTRY_SET:
        profile["country"] = country
    return profile


def classify_error(msg):
    m = (msg or "").lower()
    if "timed out" in m or "timeout" in m or "curl_rc=28" in m:
        return "timeout"
    return "error"


def fetch_ssh(host, qs, retries=5, max_time=25, backoff=2):
    inner = PROD_API_URL + "/search?" + qs
    cmd = ["ssh", host, f"curl -s --max-time {max_time} '{inner}'"]
    last = "no attempt"
    for attempt in range(retries):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=max_time + 20)
            if r.returncode == 28:
                last = f"curl timed out (curl_rc=28) after {max_time}s"
            else:
                d = json.loads(r.stdout)
                if "results" in d:
                    return d, None
                last = "no 'results' in response"
        except subprocess.TimeoutExpired as e:
            last = f"ssh timed out after {e.timeout}s"
        except Exception as e:
            last = str(e)
        time.sleep(backoff + attempt * backoff)
    return None, last


# The ONLY public host the probe may read (task 2041, from probe 2033: `--api https://websear.ch` was
# refused 24/24 by the internal door, by design). `--public` routes the probe's GETs through
# `_public_urlopen`, which accepts exactly these hosts over https and nothing else. The guard in
# `cytir.core.internal_http` / `cytir.sources.http.make_internal_client` is unchanged for every other caller.
PUBLIC_PROBE_HOSTS = frozenset({"websear.ch"})


def _public_urlopen(url, timeout=30):
    """A GET to an allow-listed public prod host. Refuses any other host or a non-https scheme."""
    u = urllib.parse.urlsplit(url)
    if u.scheme != "https" or (u.hostname or "").lower() not in PUBLIC_PROBE_HOSTS:
        raise ValueError(f"--public allows only https://{sorted(PUBLIC_PROBE_HOSTS)}, not {u.scheme}://{u.hostname}")
    req = urllib.request.Request(url, headers={"user-agent": "aw-probe/1 (+https://websear.ch)"})
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 (scheme+host checked above)


def fetch_http(api, qs, retries=5, timeout=30, backoff=2, public=False):
    url = api.rstrip("/") + "/search?" + qs
    opener = _public_urlopen if public else IH.urlopen
    last = "no attempt"
    for attempt in range(retries):
        try:
            with opener(url, timeout=timeout) as resp:
                d = json.loads(resp.read())
            if "results" in d:
                return d, None
            last = "no 'results' in response"
        except Exception as e:
            last = str(e) if str(e) else type(e).__name__
            if isinstance(e, (TimeoutError,)) or "timed out" in last:
                last = f"timed out after {timeout}s ({last})"
        time.sleep(backoff + attempt * backoff)
    return None, last


def slim_result(r, rank):
    o = {"rank": rank}
    for f in RESULT_FIELDS:
        o[f] = r.get(f)
    return o


def slim_card(c):
    # keep the fields a reviewer needs to spot duplicate / wrong cards
    keep = ["title", "name", "url", "canonical_url", "register_id", "lei",
            "country", "kind", "type"]
    return {k: c.get(k) for k in keep if k in c}


class Interrupted(BaseException):
    """SIGTERM/SIGINT turned into an exception so the run can still account for
    the queries it never reached (a silent gap is the defect, task 766)."""


def _paths(args):
    rundir = os.path.join(PROBES, "runs", args.date)
    if args.slice == "a":
        return rundir, "sample.jsonl", "results.jsonl", "run_summary.json"
    return rundir, "sample_sliceb.jsonl", "results_sliceb.jsonl", "run_summary_sliceb.json"


def _read_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        rows.append(json.loads(ln))
                    except ValueError:
                        rows.append({"_corrupt_line": ln[:80]})
    return rows


def outcome_of(rec):
    """Outcome of a recorded result. Pre-766 records carry no `outcome`."""
    if rec.get("outcome"):
        return rec["outcome"]
    if rec.get("error"):
        return classify_error(rec["error"])
    return "empty" if not rec.get("n_results") else "ok"


def coverage(sample_rows, result_rows):
    """Re-derive from the files alone (so it also catches a SIGKILLed run):
    which sampled probe_ids have no record, and the per-outcome counts."""
    got = {r.get("probe_id") for r in result_rows}
    missing = [r.get("probe_id") for r in sample_rows if r.get("probe_id") not in got]
    counts = {}
    for r in result_rows:
        o = outcome_of(r)
        counts[o] = counts.get(o, 0) + 1
    bad = {o: n for o, n in counts.items() if o in ("timeout", "error", "not-run")}
    return {
        "expected": len(sample_rows),
        "recorded": len(result_rows),
        "outcomes": counts,
        "missing_probe_ids": missing,
        "complete": bool(sample_rows) and not missing and not bad,
        "gaps": len(missing) + sum(bad.values()) + (0 if sample_rows else 1),
        **({} if sample_rows else {"reason": "sample is empty: 0 queries expected (a run that "
                                   "samples nothing is not a complete run; task 1058)"}),
    }


def write_summary(path, state, cov, extra=None):
    out = {"state": state, **cov, "written_utc": datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")}
    if extra:
        out.update(extra)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def say_gaps(cov, label):
    if cov["gaps"] or cov["recorded"] != cov["expected"]:
        if cov.get("reason"):
            print(f"*** {label}: {cov['reason']}", file=sys.stderr)
        print(f"*** {label}: RUN HAS GAPS -- {cov['recorded']}/{cov['expected']} recorded, "
              f"outcomes {cov['outcomes']}, missing {cov['missing_probe_ids'][:10]}"
              f"{'...' if len(cov['missing_probe_ids']) > 10 else ''}", file=sys.stderr)
        return False
    print(f"{label}: complete -- {cov['recorded']}/{cov['expected']} recorded, "
          f"outcomes {cov['outcomes']}", file=sys.stderr)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--slice", choices=["a", "b"], default="a",
                    help="a (default) = Slice A verbatim search-arena "
                         "(sample.jsonl -> results.jsonl); b = Slice B "
                         "target-matched CH-first panel "
                         "(sample_sliceb.jsonl -> results_sliceb.jsonl)")
    ap.add_argument("--ssh-host", default=SSH_HOST)
    ap.add_argument("--api", default=None, help="direct http endpoint (skips ssh)")
    ap.add_argument("--public", action="store_true",
                    help="let --api be the public prod host (allow-list PUBLIC_PROBE_HOSTS, https only)")
    ap.add_argument("--k", type=int, default=10,
                    help="k for --fixed-params mode (ignored otherwise)")
    ap.add_argument("--fixed-params", action="store_true",
                    help="legacy behavior: use the sampler's fixed params "
                         "instead of the per-probe randomized profile")
    ap.add_argument("--sleep", type=float, default=1.0, help="polite gap between queries (s)")
    ap.add_argument("--timeout", type=float, default=25.0,
                    help="per-attempt request timeout (s)")
    ap.add_argument("--retries", type=int, default=5, help="attempts per query")
    ap.add_argument("--backoff", type=float, default=2.0, help="retry back-off unit (s)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a same-day results file: the old one is MOVED to "
                         "<name>.superseded-<HHMMSSZ>, never truncated (task 766)")
    ap.add_argument("--check", action="store_true",
                    help="do not run: compare sample vs results for --date/--slice, "
                         "print gaps, exit 4 if any")
    ap.add_argument("--runs-dir", default=None, help=argparse.SUPPRESS)  # tests
    args = ap.parse_args()

    rundir, sample_name, results_name, summary_name = _paths(args)
    if args.runs_dir:
        rundir = os.path.join(args.runs_dir, args.date)
    sample_path = os.path.join(rundir, sample_name)
    outpath = os.path.join(rundir, results_name)
    summary_path = os.path.join(rundir, summary_name)

    if args.check:
        if not os.path.exists(sample_path):
            sys.exit(f"no sample file: {sample_path}")
        cov = coverage(_read_jsonl(sample_path), _read_jsonl(outpath))
        ok = say_gaps(cov, f"slice {args.slice.upper()} {args.date}")
        sys.exit(0 if ok else 4)

    if not os.path.exists(sample_path):
        sampler = "python3 -m cytir.probe.sample" if args.slice == "a" else "python3 -m cytir.probe.sample_sliceb"
        sys.exit(f"no sample file: {sample_path} (run {sampler} first)")
    rows = _read_jsonl(sample_path)
    if not rows:
        cov = coverage([], [])
        write_summary(summary_path, "refused", cov)
        sys.exit(f"REFUSED: {sample_path} holds 0 queries; nothing to run (task 1058)")
    if not args.api and not args.ssh_host:
        sys.exit("REFUSED: no route to prod; set CYTIR_PROBE_SSH_HOST (or pass --ssh-host / --api)")

    # Never silently overwrite a same-day results file (a worse re-run replaced the
    # cron's good 05:00Z results on 2026-10-08). Refuse, or move it aside.
    if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
        old = _read_jsonl(outpath)
        started = next((r.get("ts") for r in old if r.get("ts")), "unknown")
        if not args.force:
            print(f"REFUSED: {outpath} already exists ({len(old)} records, first query "
                  f"started {started}); pass --force to move it aside and re-run.",
                  file=sys.stderr)
            sys.exit(3)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%H%M%SZ")
        for path in (outpath, summary_path):
            if os.path.exists(path):
                os.replace(path, f"{path}.superseded-{stamp}")
        print(f"--force: moved old results to {outpath}.superseded-{stamp}", file=sys.stderr)

    n_ok = n_empty = n_err = 0
    print(f"probe_run: {len(rows)} queries -> {outpath} "
          f"({'api ' + args.api if args.api else 'ssh ' + args.ssh_host}), "
          f"~{args.sleep}s gap", file=sys.stderr)

    def _raise(signum, _frame):
        raise Interrupted(signal.Signals(signum).name)
    old = {sg: signal.signal(sg, _raise) for sg in (signal.SIGTERM, signal.SIGINT)}

    def base_rec(row, params, seed):
        rec = {
            "probe_id": row.get("probe_id"),
            "query": row["query"],
            "slice": row.get("slice", "A"),
            "origin": row.get("origin"),
            "dataset": row.get("dataset"),
            "lang": row.get("lang"),
            "focus_topic": row.get("focus_topic"),
            "template_id": row.get("template_id"),
            "targets": row.get("stratum") if row.get("origin") in ("adapted", "synth", "arena") else None,
            "params": params,
            "params_seed": seed,
        }
        # Slice B carries scope metadata for scope-AWARE grading (task 269):
        if row.get("slice") == "B":
            for f in ("qid", "target_domain", "subdomain", "difficulty",
                      "in_scope", "gold_domains", "gold_match", "authoritative_target",
                      "source", "arena_id"):
                rec[f] = row.get(f)
        return rec

    interrupted = None
    written = 0
    # Mark the run as started BEFORE the first query: a SIGKILL leaves "running"
    # + fewer records than the sample, which --check and the scorer report.
    write_summary(summary_path, "running",
                  {"expected": len(rows), "recorded": 0, "outcomes": {},
                   "missing_probe_ids": [r.get("probe_id") for r in rows],
                   "complete": False, "gaps": len(rows)})
    try:
        with open(outpath, "w", encoding="utf-8") as out:
            for i, row in enumerate(rows):
                if args.fixed_params:
                    params = row.get("params", {"k": args.k, "authorities": 1})
                    seed = "fixed-params"
                else:
                    params = draw_profile(row)
                    seed = params_seed(row)
                qs = build_qs(row["query"], params, args.k)
                # task 352: wall-clock start timestamp per query
                ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
                t0 = time.time()
                if args.api:
                    d, err = fetch_http(args.api, qs, retries=args.retries,
                                        timeout=args.timeout, backoff=args.backoff,
                                        public=args.public)
                else:
                    d, err = fetch_ssh(args.ssh_host, qs, retries=args.retries,
                                       max_time=int(args.timeout), backoff=args.backoff)
                dt = round(time.time() - t0, 2)

                rec = base_rec(row, params, seed)
                rec.update({"ts": ts, "latency_s": dt})
                if d is None:
                    kind = classify_error(err)
                    rec.update({"outcome": kind, "error": err, "n_results": 0,
                                "results": [], "authorities": []})
                    n_err += 1
                    flag = f"{kind.upper()}({err})"
                else:
                    results = d.get("results", []) or []
                    cards = d.get("authorities", []) or []
                    rec.update({
                        "error": None,
                        "n_results": len(results),
                        "results": [slim_result(r, j + 1) for j, r in enumerate(results)],
                        "authorities": [slim_card(c) for c in cards] if isinstance(cards, list) else cards,
                    })
                    if len(results) == 0:
                        rec["outcome"] = "empty"
                        n_empty += 1
                        flag = "EMPTY"
                    else:
                        rec["outcome"] = "ok"
                        n_ok += 1
                        flag = f"{len(results)}r"
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                written += 1
                prof = "k={k}{lang}{ctry}{auth}".format(
                    k=params.get("k"),
                    lang="," + params["lang"] if params.get("lang") else "",
                    ctry="," + params["country"] if params.get("country") else "",
                    auth=",auth" if params.get("authorities") else "")
                print(f"  [{i+1:>2}/{len(rows)}] {dt:>4}s {flag:>10} "
                      f"[{prof:<16}] {row['query'][:60]}",
                      file=sys.stderr)
                write_summary(summary_path, "running", coverage(rows, _read_jsonl(outpath)))
                if i < len(rows) - 1:
                    time.sleep(args.sleep)
    except Interrupted as e:
        interrupted = str(e)
    finally:
        # Account for every query the loop never reached: a record, not a gap.
        if written < len(rows):
            reason = f"interrupted: {interrupted or 'abnormal exit'}"
            with open(outpath, "a", encoding="utf-8") as out:
                for row in rows[written:]:
                    rec = base_rec(row, {}, None)
                    rec.update({"outcome": "not-run", "error": reason,
                                "n_results": 0, "results": [], "authorities": [],
                                "ts": None, "latency_s": None})
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for sg, h in old.items():
            signal.signal(sg, h)
        cov = coverage(rows, _read_jsonl(outpath))
        write_summary(summary_path,
                      "interrupted" if written < len(rows) else "complete", cov,
                      {"interrupted_by": interrupted} if interrupted else None)

    print(f"\nDONE: {n_ok} ok / {n_empty} empty / {n_err} error -> {outpath}",
          file=sys.stderr)
    ok = say_gaps(cov, f"slice {args.slice.upper()} {args.date}")
    if interrupted:
        sys.exit(130)
    sys.exit(0)  # gaps are reported + recorded in run_summary, never silent; the chain continues


if __name__ == "__main__":
    main()
