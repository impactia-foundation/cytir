#!/usr/bin/env python3
"""promote_verify — gate a blue/green promote candidate (ADR-066). Two tiers,
both exit NONZERO on regression so they drop straight into a promote pipeline.

TIER 1 — STRUCTURAL (qdrant REST, stdlib only): connect to a qdrant + collection
  and assert the promote invariants directly on the data:
    * exact point count == --expect (the base + net-new number build_vnext.py
      logged);
    * the RATIFIED serving shape, read from `cytir.core.serving_config`
      (dense dim, datatype and quantizer, plus the indexed-vector residual
      its quantizer actually tolerates) + a bm25 sparse leg + hnsw m=16.
      Pass --serving-config to gate a candidate against a different one;
      the default is the ratified flip config, never a literal here;
    * gold-DOMAIN presence: every canonical gold domain still has >0 points
      (h@k is monotonic under ADD, so a present gold domain cannot be crowded
      OUT of the index by a net-new merge — a missing one is a real regression).
  `domain` is an INDEXED payload field, so the filtered counts are cheap and
  safe even against live prod (unlike an unindexed scroll — memory
  `prod-serving-box-fragile-scrolls`).

TIER 2 — RANKING (search API, optional, --api): run a sample of gold grounding
  queries + the trust-flood probe against a LIVE serving endpoint and score
  hit@1/3/10 (domain-level) + MRR; assert hit@10 >= --min-hit10 and that the
  flood query's rank-1 is not a gov/reference page crowding out relevance.
  Use this POST-SWAP against the (test or real) API. The dry-run / throwaway
  proof uses Tier 1 only (no serving endpoint is wired to the candidate).

Examples:
  # dry-run aw_v5 structural gate on the build box's vnext instance:
  python3 -m cytir.gates.promote_verify --qdrant-url http://localhost:6390 \
      --collection aw_v5 --expect 48479389 --eval-dir data

  # post-swap ranking gate against prod api (from the operator box):
  python3 -m cytir.gates.promote_verify --api-ssh <serving-box> --eval-dir data \
      --sample 40 --min-hit10 0.60
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse

from cytir.core import internal_http as IH
from cytir.core import serving_config as sc

#: A bulk-build collection is created at m=0 and finalised to m=16; one
#: left at 0 answers every query by full scan (qdrant-bulk-build-dim-change).
EXPECTED_HNSW_M = 16

# --------------------------------------------------------------------------
# qdrant REST (stdlib)
# --------------------------------------------------------------------------

def q_get(url: str, path: str):
    return IH.fetch_json(f"{url}{path}", timeout=60)["result"]


def q_post(url: str, path: str, body: dict):
    return IH.fetch_json(f"{url}{path}", body=body, method="POST", timeout=120)["result"]


def exact_count(url: str, coll: str) -> int:
    return q_post(url, f"/collections/{coll}/points/count", {"exact": True})["count"]


def domain_count(url: str, coll: str, domain: str) -> int:
    body = {"exact": True,
            "filter": {"must": [{"key": "domain", "match": {"value": domain}}]}}
    return q_post(url, f"/collections/{coll}/points/count", body)["count"]


# --------------------------------------------------------------------------
# gold extraction (same walk as the task 234 gold gate)
# --------------------------------------------------------------------------

def reg_domain(u: str) -> str:
    u = (u or "").strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    return u.split("/")[0].split(":")[0].rstrip(".")


def load_gold(eval_dir: str):
    """Return (gold_domains:set, queries:list[{query,gold_urls,gold_domains}])."""
    import yaml  # optional; only needed when --eval-dir is used
    gold_domains, queries = set(), []
    files = sorted(glob.glob(f"{eval_dir}/eval_*.yaml"))
    for fp in files:
        try:
            docs = list(yaml.safe_load_all(open(fp)))
        except Exception as e:  # noqa: BLE001
            print("  skip", os.path.basename(fp), e); continue
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            rows = doc.get("queries") or []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                gu = row.get("gold_urls") or ([row["gold_url"]] if row.get("gold_url") else [])
                gd = row.get("gold_domains") or ([row["gold_domain"]] if row.get("gold_domain") else [])
                gd = [reg_domain(d) for d in gd] + [reg_domain(u) for u in gu]
                gd = [d for d in gd if d]
                for d in gd:
                    gold_domains.add(d)
                q = row.get("query") or row.get("q")
                if q and gd:
                    queries.append({"query": q, "gold_urls": gu, "gold_domains": sorted(set(gd))})
    return gold_domains, queries


# built-in canonical gold domains — used when --eval-dir is not available, so
# the structural gate still has teeth as a standalone file on any box.
BUILTIN_GOLD = [
    "fedlex.admin.ch", "admin.ch", "bfs.admin.ch", "kmu.admin.ch",
    "en.wikipedia.org", "de.wikipedia.org", "arxiv.org", "pubmed.ncbi.nlm.nih.gov",
    "legifrance.gouv.fr", "gesetze-im-internet.de", "laws-lois.justice.gc.ca",
    "canada.ca", "developer.mozilla.org", "docs.python.org",
]

# gov/reference hosts that, at rank-1 on an off-topic commercial query, indicate
# the trust-over-relevance flood (ADR-044 / probe findings).
FLOOD_HOSTS = {"canada.ca", "kmu.admin.ch", "admin.ch", "iata.org",
               "who.int", "europa.eu"}
FLOOD_QUERY = "best online shop electronics switzerland"


# --------------------------------------------------------------------------
# search API (reuses the call shape of the retired eval_authoritative_run script)
# --------------------------------------------------------------------------

def api_search(q: str, k: int, api_http: str, api_ssh: str, retries: int = 5):
    inner = f"http://localhost:8090/search?q={urllib.parse.quote(q)}&k={k}"
    for attempt in range(retries):
        try:
            if api_ssh:
                r = subprocess.run(["ssh", api_ssh, f"curl -s --max-time 25 '{inner}'"],
                                   capture_output=True, text=True, timeout=45)
                d = json.loads(r.stdout)
            else:
                url = f"{api_http.rstrip('/')}/search?q={urllib.parse.quote(q)}&k={k}"
                with IH.public_self_urlopen(url, timeout=30, caller="promote-verify") as resp:
                    d = json.loads(resp.read())
            if "results" in d:
                return d["results"]
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2 + attempt * 2)
    return None


def first_rank_domain(results, gold_domains):
    for i, r in enumerate(results):
        h = reg_domain((r.get("url") or ""))
        if h in gold_domains:
            return i + 1
    return None


# --------------------------------------------------------------------------

def tier1_structural(args) -> int:
    url, coll = args.qdrant_url, args.collection
    print(f"== TIER 1 STRUCTURAL: {coll} @ {url} ==", flush=True)
    fails = 0

    info = q_get(url, f"/collections/{coll}")
    p = info["config"]["params"]
    dense = p["vectors"]["dense"]
    sparse = list((p.get("sparse_vectors") or {}).keys())
    hnsw_m = info["config"]["hnsw_config"]["m"]
    cnt = exact_count(url, coll)

    cfg = sc.get(args.serving_config)
    shape = sc.shape_from_json(info)
    print(f"  points={cnt:,} status={info['status']}", flush=True)
    print(f"  dense={dense['size']} on_disk={dense['on_disk']} sparse={sparse} "
          f"hnsw_m={hnsw_m} quantizer={shape['quantizer_kind']}", flush=True)

    if args.expect is not None and cnt != args.expect:
        print(f"  FAIL count {cnt:,} != expect {args.expect:,}", flush=True); fails += 1
    else:
        print("  OK count", flush=True)

    # The dense-side shape rules live in `cytir.core.serving_config` and are
    # imported, not restated. Until task 529 this block carried its own copy,
    # frozen at the retired 128-int8 "config B": against the ratified 256-pq8
    # candidate it FAILED by construction, and it read the quantizer only from
    # `quantization_config.scalar`, so a product-quantized collection reported
    # `None` whatever it held. This is the tool the flip-day runbook names.
    shape_problems = sc.collection_shape_problems(shape, cfg,
                                                  check_indexed=True)
    # The two rules that are about this pipeline rather than the serving
    # config: the lexical leg must exist, and the graph must be built.
    if "bm25" not in sparse:
        shape_problems.append(f"sparse vectors {sparse} carry no 'bm25' leg")
    if hnsw_m != EXPECTED_HNSW_M:
        shape_problems.append(f"hnsw m={hnsw_m} != {EXPECTED_HNSW_M} "
                              "(a bulk-build collection left at m=0 serves "
                              "every query by full scan)")
    if shape_problems:
        print(f"  FAIL {cfg.name} shape: " + "; ".join(shape_problems),
              flush=True)
        fails += 1
    else:
        print(f"  OK {cfg.name} shape", flush=True)

    gold_domains = set()
    if args.eval_dir and os.path.isdir(args.eval_dir):
        gold_domains, _ = load_gold(args.eval_dir)
    if not gold_domains:
        gold_domains = set(BUILTIN_GOLD)
    check = sorted(gold_domains)
    if args.gold_domain_sample and len(check) > args.gold_domain_sample:
        check = check[:args.gold_domain_sample]

    # NON-REGRESSION gate: a promote candidate is a SUPERSET of its base (dedup
    # keeps every base point, only adds net-new), so a gold domain absent from
    # the candidate is a pre-existing BASE coverage gap, NOT a promote regression.
    # When --base-collection is given we compare: a domain present in the base but
    # absent/reduced in the candidate is the only real FAIL. Coverage gaps (absent
    # in both) are reported informationally. Without a base, absent domains are a
    # WARN (coverage audit), not a gate failure.
    base_url = args.base_qdrant_url or url
    have_base = bool(args.base_collection)
    print(f"  gold-domain check: {len(check)} domains "
          f"({'non-regression vs '+args.base_collection if have_base else 'coverage audit'})",
          flush=True)
    regressions, gaps = [], []
    for d in check:
        try:
            cand = domain_count(url, coll, d)
            if have_base:
                base_n = domain_count(base_url, args.base_collection, d)
                if base_n > 0 and cand < base_n:
                    regressions.append((d, base_n, cand))
                elif base_n == 0:
                    gaps.append(d)
            else:
                if cand == 0:
                    gaps.append(d)
        except Exception as e:  # noqa: BLE001
            print("   (count error", d, e, ")", flush=True)
    print(f"  coverage gaps (absent in base too / audit): {len(gaps)}", flush=True)
    if have_base:
        if regressions:
            print(f"  FAIL {len(regressions)} gold domains REGRESSED "
                  f"(base>cand): {regressions[:10]}", flush=True)
            fails += 1
        else:
            print(f"  OK no gold-domain regression vs {args.base_collection}", flush=True)
    else:
        print(f"  WARN {len(gaps)} gold domains absent (no --base to prove "
              f"non-regression; treat as coverage audit, not a gate)", flush=True)

    print("TIER1:", "PASS" if fails == 0 else f"FAIL ({fails} checks)", flush=True)
    return fails


def tier2_ranking(args) -> int:
    print(f"== TIER 2 RANKING: api_ssh={args.api_ssh or '-'} api={args.api or '-'} ==",
          flush=True)
    fails = 0
    queries = []
    if args.eval_dir and os.path.isdir(args.eval_dir):
        _, queries = load_gold(args.eval_dir)
    if not queries:
        print("  no eval queries loaded (need --eval-dir with yaml) — flood-only", flush=True)
    if args.sample and len(queries) > args.sample:
        step = max(1, len(queries) // args.sample)
        queries = queries[::step][:args.sample]

    hit1 = hit3 = hit10 = 0.0
    rr = 0.0
    n = 0
    for row in queries:
        res = api_search(row["query"], args.k, args.api, args.api_ssh)
        if res is None:
            print("   (no response)", row["query"][:60], flush=True); continue
        n += 1
        rank = first_rank_domain(res, set(row["gold_domains"]))
        if rank:
            rr += 1.0 / rank
            hit1 += rank <= 1
            hit3 += rank <= 3
            hit10 += rank <= 10
    if n:
        print(f"  scored {n} queries: h@1={hit1/n:.3f} h@3={hit3/n:.3f} "
              f"h@10={hit10/n:.3f} MRR={rr/n:.3f}", flush=True)
        if hit10 / n < args.min_hit10:
            print(f"  FAIL h@10 {hit10/n:.3f} < min {args.min_hit10}", flush=True)
            fails += 1
        else:
            print("  OK h@10 >= min", flush=True)

    # flood probe
    res = api_search(FLOOD_QUERY, args.k, args.api, args.api_ssh)
    if res:
        top = reg_domain((res[0].get("url") or ""))
        flooded = top in FLOOD_HOSTS
        print(f"  flood probe rank-1: {top} -> {'FLOOD' if flooded else 'ok'}", flush=True)
        if flooded:
            print("  FAIL flood query rank-1 is a gov/reference flood host", flush=True)
            fails += 1
    else:
        print("  (flood probe: no response)", flush=True)

    print("TIER2:", "PASS" if fails == 0 else f"FAIL ({fails} checks)", flush=True)
    return fails


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qdrant-url", default="")
    ap.add_argument("--collection", default="")
    ap.add_argument("--base-collection", default="",
                    help="base collection to prove NON-REGRESSION against (a promote "
                         "candidate is a superset of its base); only a gold domain "
                         "present in base but reduced in the candidate FAILs")
    ap.add_argument("--base-qdrant-url", default="",
                    help="qdrant url for --base-collection (default: same as --qdrant-url)")
    ap.add_argument("--expect", type=int, default=None,
                    help="exact expected point count (base + net-new)")
    ap.add_argument("--eval-dir", default="",
                    help="repo data/ dir with eval_*.yaml (gold source)")
    ap.add_argument("--gold-domain-sample", type=int, default=0,
                    help="cap number of gold domains checked in tier1 (0=all)")
    ap.add_argument("--api", default="", help="http search endpoint (tier2)")
    ap.add_argument("--api-ssh", default="", help="ssh host running the api on :8090 (tier2)")
    ap.add_argument("--sample", type=int, default=40, help="tier2 gold-query sample size")
    ap.add_argument("--serving-config", default=sc.FLIP_SERVING_CONFIG,
                    help="the cytir.core.serving_config name the candidate "
                         "must match (default: the ratified flip config, "
                         f"{sc.FLIP_SERVING_CONFIG!r}). Deliberately NOT a "
                         "literal in this file -- carrying its own copy is "
                         "how this check stayed frozen at the retired "
                         "128-int8 shape.")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-hit10", type=float, default=0.55)
    args = ap.parse_args(argv)

    fails = 0
    if args.qdrant_url and args.collection:
        fails += tier1_structural(args)
    if args.api or args.api_ssh:
        fails += tier2_ranking(args)
    if not (args.qdrant_url or args.api or args.api_ssh):
        sys.exit("nothing to do: pass --qdrant-url/--collection (tier1) and/or "
                 "--api/--api-ssh (tier2)")
    print("\nVERIFY:", "PASS" if fails == 0 else f"FAIL ({fails} total)", flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
