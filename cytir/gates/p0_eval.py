#!/usr/bin/env python3
"""AW P0 acceptance gate: run data/eval.yaml against a running search API and
report target-URL hit-rate@k + cross-lingual breakdown (spec §8).

  cyt gate p0-eval --eval data/eval.yaml --api http://localhost:8090 --k 10
"""
from __future__ import annotations

import argparse
import collections
import urllib.parse

import yaml

from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client


def norm(u: str) -> str:
    if not u:
        return ""
    p = urllib.parse.urlsplit(u.strip())
    return f"{p.netloc.lower()}{p.path.rstrip('/')}"


def hit_rank(target: str, results: list[dict]) -> int | None:
    t = norm(target)
    for i, r in enumerate(results):
        ru = norm(r.get("url", ""))
        if ru == t or ru.startswith(t) or t.startswith(ru):
            return i + 1
    return None


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--eval", default="data/eval.yaml")
    ap.add_argument("--api", default="http://localhost:8090")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args(argv)

    items = yaml.safe_load(open(args.eval))["eval"]
    by_src = collections.defaultdict(lambda: [0, 0])   # source -> [hits, total]
    by_lang = collections.defaultdict(lambda: [0, 0])
    h1 = h3 = hk = 0
    mrr = 0.0
    misses = []
    errors = 0

    for q in items:
        try:
            with make_client(HttpSpec(timeout=60.0, follow_redirects=False)) as client:
                r = client.get(f"{args.api}/search",
                               params={"q": q.get("question") or q.get("query"), "k": args.k})
            # task 711: a 503 carries a JSON body with no `results`, so it used to be scored as a
            # silent miss. It is an error; it is still scored as a miss below, but it is COUNTED.
            r.raise_for_status()
            results = r.json().get("results", [])
        except Exception as e:  # noqa: BLE001
            results = []
            errors += 1
            print(f"  ! query error {q['id']}: {e}")
        rank = hit_rank(q["target_url"], results)
        src, lang = q.get("source", "?"), q.get("lang", "?")
        by_src[src][1] += 1; by_lang[lang][1] += 1
        if rank:
            hk += 1; mrr += 1.0 / rank
            by_src[src][0] += 1; by_lang[lang][0] += 1
            if rank == 1: h1 += 1
            if rank <= 3: h3 += 1
        else:
            misses.append(q["id"])

    n = len(items)
    print(f"\n=== AW P0 eval — {n} questions, k={args.k} ===")
    print(f"hit@1 : {h1}/{n}  ({h1/n:.0%})")
    print(f"hit@3 : {h3}/{n}  ({h3/n:.0%})")
    print(f"hit@{args.k}: {hk}/{n}  ({hk/n:.0%})")
    print(f"MRR   : {mrr/n:.3f}")
    print(f"errors: {errors}/{n}  (errored queries are scored as misses above; a caller that gates "
          f"on these numbers must refuse a non-zero count)")
    print("\nby source:")
    for s, (h, t) in sorted(by_src.items()):
        print(f"  {s:12s} {h}/{t}")
    print("by language:")
    for l, (h, t) in sorted(by_lang.items()):
        print(f"  {l:4s} {h}/{t}")
    if misses:
        print(f"\nmisses ({len(misses)}): {', '.join(misses)}")


if __name__ == "__main__":
    main()
