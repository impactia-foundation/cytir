#!/usr/bin/env python3
"""cytir/probe/score_sliceb.py — SCOPE-AWARE scorer for Slice B (task 269).

Slice B is graded against AW's ACTUAL SCOPE: the metric is "did we surface the
authoritative URL for something we're BUILT for" — an out-of-scope query is NOT
a miss. This scorer turns each Slice-B pool record's `gold_domains` + `in_scope`
into a deterministic, mechanical signal the investigative reviewer starts from
(it does NOT replace the reviewer — it hands them leads to dig into).

For each results_sliceb.jsonl record it computes:
  * whether an AUTHORITATIVE host (any of gold_domains) is present in the
    returned top-k, and at what rank;
  * a scope-aware verdict:
      hit         — the gold's IDENTITY is present within the probe's k window: the
                    URL matches the per-gold `gold_match` rule declared in the pool
                    (exact URL / URL prefix / regex), or -- for a gold with no rule
                    (an entity's whole site) -- its host (task 766)
      host-only   — a gold HOST is present but NO result matches the declared
                    gold_match rule (e.g. AufenthG §91k on gesetze-im-internet.de
                    for a §9 query). NOT a hit; in-scope => counted as a non-hit
      deep-hit    — present but below the k window / below rank 10 (k-instability lead)
      MISS        — in_scope=yes, owner exists, absent  => a REAL candidate defect
      miss-exempt — in_scope=borderline, absent          => scope-borderline, NOT penalized
      scope-exempt— no single authoritative owner (gold_domains=[]) => out of scope by design
      empty       — 0 results / error (P0 lead if it persisted through retries)

Output:
    probes/runs/<date>/sliceb_scores.jsonl   (one annotated line per query)
  + a printed summary (scope-aware hit-rate over the IN-SCOPE subset only).

READ-ONLY over local files. No network.

Usage:
    python3 -m cytir.probe.score_sliceb                 # date=today
    python3 -m cytir.probe.score_sliceb --date 2026-07-18
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import re
import sys
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PROBES = os.path.join(REPO, "probes")
DEFAULT_K = 10  # the unified web+MCP default; used for the "deep" (k-instability) cut


def host_of(url):
    try:
        h = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""
    if h.startswith("www."):
        h = h[4:]
    for p in ("m.", "mobile."):
        if h.startswith(p):
            h = h[len(p):]
    return h


def host_matches(host, gold):
    gold = gold.lower().lstrip(".")
    return host == gold or host.endswith("." + gold)


def norm_url(url):
    """host/path?query#fragment -- lower-cased host, no scheme, no www/m, no
    trailing '/' on a bare path. Rules in the pool are written in THIS form."""
    try:
        u = urllib.parse.urlparse(url or "")
    except Exception:
        return ""
    out = host_of(url) + (u.path or "")
    if out.endswith("/") and not u.query and not u.fragment:
        out = out.rstrip("/")
    if u.query:
        out += "?" + u.query
    if u.fragment:
        out += "#" + u.fragment
    return out


def rule_matches(rule, url):
    """One declared gold_match rule: {"kind": "url"|"prefix"|"regex", "value": str}.
    url    -- normalized URL equals the normalized value
    prefix -- normalized URL starts with the normalized value
    regex  -- re.search(value, normalized URL)   (written against norm_url form)
    """
    kind, value = rule.get("kind"), rule.get("value") or ""
    n = norm_url(url)
    if not n:
        return False
    if kind == "url":
        return n == norm_url(value)
    if kind == "prefix":
        return n.startswith(norm_url(value))
    if kind == "regex":
        return re.search(value, n) is not None
    raise ValueError(f"unknown gold_match kind {kind!r}")


def score_record(rec, pool_rules=None):
    gold = rec.get("gold_domains") or []
    # rule: the record's own (copied from the pool at sample time) else the pool's
    # CURRENT declaration for this qid (re-scoring runs recorded before task 766)
    rules = rec.get("gold_match") or (pool_rules or {}).get(rec.get("qid")) or []
    in_scope = rec.get("in_scope") or "yes"
    results = rec.get("results") or []
    k = (rec.get("params") or {}).get("k", DEFAULT_K)

    # Best (lowest) rank of a result that carries the gold's IDENTITY, and of one
    # that merely sits on a gold HOST (task 766: a host is not a document).
    hit_rank = hit_host = hit_url = None
    host_rank = host_url = None
    for r in results:
        url = r.get("url") or ""
        h = host_of(url)
        if not h:
            continue
        on_gold_host = any(host_matches(h, g) for g in gold)
        if on_gold_host and host_rank is None:
            host_rank, host_url = r.get("rank"), url
        if rules:
            ident = any(rule_matches(x, url) for x in rules)
        else:
            ident = on_gold_host        # undeclared rule: host-level (entity sites)
        if ident and hit_rank is None:
            hit_rank, hit_host, hit_url = r.get("rank"), h, url
    match_rule = "identity" if rules else "host(default)"

    empty = (rec.get("error") is not None) or (rec.get("n_results", len(results)) == 0)

    outcome = rec.get("outcome")
    if outcome in ("timeout", "error", "not-run"):
        verdict = "no-data"             # the runner did not get an answer: NOT a miss, NOT an empty
    elif empty:
        verdict = "empty"
    elif not gold:
        verdict = "scope-exempt"        # no single authoritative owner -> out of scope by design
    elif hit_rank is not None and hit_rank <= min(k, DEFAULT_K):
        verdict = "hit"
    elif hit_rank is not None:
        verdict = "deep-hit"            # present but below the window -> k-instability lead
    elif rules and host_rank is not None and in_scope == "yes":
        verdict = "host-only"           # right site, wrong/absent document -> NOT a hit
    elif in_scope == "yes":
        verdict = "MISS"               # owner exists, in scope, absent -> REAL candidate defect
    else:
        verdict = "miss-exempt"        # borderline scope, absent -> NOT penalized

    return {
        "probe_id": rec.get("probe_id"),
        "qid": rec.get("qid"),
        "query": rec.get("query"),
        "slice": "B",
        "target_domain": rec.get("target_domain"),
        "subdomain": rec.get("subdomain"),
        "difficulty": rec.get("difficulty"),
        "in_scope": in_scope,
        "params": rec.get("params"),
        "gold_domains": gold,
        "authoritative_target": rec.get("authoritative_target"),
        "gold_match": rules or None,
        "match_rule": match_rule,
        "hit_rank": hit_rank,
        "hit_host": hit_host,
        "hit_url": hit_url,
        "host_only_rank": host_rank if (rules and hit_rank is None) else None,
        "host_only_url": host_url if (rules and hit_rank is None) else None,
        "n_results": rec.get("n_results", len(results)),
        "top3_hosts": [host_of(r.get("url") or "") for r in results[:3]],
        "outcome": outcome,
        "verdict": verdict,
    }


def load_pool_rules(path):
    rules = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    r = json.loads(ln)
                    if r.get("gold_match"):
                        rules[r["qid"]] = r["gold_match"]
    return rules


def no_data_line(row):
    """A scored line for a sampled query that has NO result record at all."""
    return {"probe_id": row.get("probe_id"), "qid": row.get("qid"), "query": row.get("query"),
            "slice": "B", "target_domain": row.get("target_domain"),
            "subdomain": row.get("subdomain"), "difficulty": row.get("difficulty"),
            "in_scope": row.get("in_scope"), "params": None,
            "gold_domains": row.get("gold_domains") or [], "gold_match": None,
            "match_rule": None, "hit_rank": None, "hit_host": None, "hit_url": None,
            "n_results": 0, "top3_hosts": [], "outcome": "missing",
            "verdict": "no-data"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--runs-dir", default=os.path.join(PROBES, "runs"), help=argparse.SUPPRESS)
    ap.add_argument("--pool", default=os.path.join(PROBES, "slice_b_pool.jsonl"))
    args = ap.parse_args()
    pool_rules = load_pool_rules(args.pool)

    rundir = os.path.join(args.runs_dir, args.date)
    inpath = os.path.join(rundir, "results_sliceb.jsonl")
    if not os.path.exists(inpath):
        sys.exit(f"no Slice-B results: {inpath} (run python3 -m cytir.probe.run --slice b first)")

    scored = []
    with open(inpath, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                scored.append(score_record(json.loads(ln), pool_rules))
    # A sampled query with no result record is a GAP, said out loud (task 766).
    sample_path = os.path.join(rundir, "sample_sliceb.jsonl")
    if os.path.exists(sample_path):
        have = {s["probe_id"] for s in scored}
        with open(sample_path, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    row = json.loads(ln)
                    if row.get("probe_id") not in have:
                        scored.append(no_data_line(row))

    outpath = os.path.join(rundir, "sliceb_scores.jsonl")
    with open(outpath, "w", encoding="utf-8") as out:
        for s in scored:
            out.write(json.dumps(s, ensure_ascii=False) + "\n")

    import collections
    by_v = collections.Counter(s["verdict"] for s in scored)
    # scope-aware hit-rate: over the IN-SCOPE-and-scorable subset only
    scorable = [s for s in scored if s["verdict"] in ("hit", "deep-hit", "MISS", "host-only")]
    hits = [s for s in scorable if s["verdict"] in ("hit", "deep-hit")]
    exempt = [s for s in scored if s["verdict"] in ("scope-exempt", "miss-exempt")]
    misses = [s for s in scored if s["verdict"] in ("MISS", "host-only")]
    deep = [s for s in scored if s["verdict"] == "deep-hit"]

    print(f"cytir.probe.score_sliceb: {len(scored)} Slice-B queries ({args.date})", file=sys.stderr)
    print(f"  verdicts: {dict(by_v)}", file=sys.stderr)
    if by_v.get("no-data"):
        print(f"  *** {by_v['no-data']} queries have NO answer (timeout/error/not-run/missing) "
              f"-- the rate below excludes them", file=sys.stderr)
    rate = (100 * len(hits) // len(scorable)) if scorable else 0
    print(f"  SCOPE-AWARE authoritative-hit rate (in-scope only): "
          f"{len(hits)}/{len(scorable)} ({rate}%)", file=sys.stderr)
    print(f"  scope-exempt (NOT counted as misses): {len(exempt)}", file=sys.stderr)
    if misses:
        print("  REAL candidate defects (in-scope, owner absent):", file=sys.stderr)
        for s in misses:
            print(f"    - {s['probe_id']} [{s['difficulty']}] {s['query'][:55]} "
                  f"[{s['verdict']}] want {s['gold_domains']}, got top3 {s['top3_hosts']}", file=sys.stderr)
    if deep:
        print("  k-instability leads (authority present but DEEP):", file=sys.stderr)
        for s in deep:
            print(f"    - {s['probe_id']} k={s['params'].get('k')} hit_rank={s['hit_rank']} "
                  f"{s['query'][:50]}", file=sys.stderr)
    print(f"WROTE {outpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
