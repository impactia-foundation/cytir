"""Scope-separated scoring for external benchmark runs (task 1026, RULING 49).

A public benchmark score is reported in two halves (verification doctrine):
the IN-SCOPE hit rate (the KPI: questions whose gold source POLICY lets us
index) and the OUT-OF-SCOPE share (the honesty line). This module classifies a
gold URL / DOI, scores per-query result rows in the two halves with Wilson 95 %
CIs, and lists the in-scope or undecided hosts we missed (an input to a POLICY
decision, never an admission).

Classes
- ``in``        host or family is admitted by POLICY/PORTFOLIO, or the index already
                serves rows for the host (an admitted row exists by definition).
- ``out``       POLICY forbids it (clause named): D2 paywall, D3 mirror, social.
- ``undecided`` no rule applies. Tim rules through POLICY.

The classifier is deterministic. The only live input is ``served``: a mapping
host -> row count on the twin, produced by ``probe`` (keyed ``domain`` count,
read-only, one call per host, no scroll).
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import json
import sys
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

from cytir.corpus.l0_census import wilson

IN, OUT, UNDECIDED = "in", "out", "undecided"

# POLICY section 3 D3 + section 4.4 "docs are the canonical instance (not a docs-mirror)".
MIRROR_HOSTS = frozenset({
    "gnu.huihoo.com", "gnu.ist.utl.pt", "gnu.cs.utah.edu", "mirrors.aliyun.com",
    "ecma262.docschina.org",   # translated copy of the ECMAScript spec
    "spring.pleiades.io",      # translated copy of the Spring docs
    "gensoft.pasteur.fr",      # institute install-tree copy of tool docs
})
# POLICY section 3 D2 (paywalled/subscription core content), generic cases.
PAYWALL_HOSTS = frozenset({
    "nytimes.com", "wsj.com", "ft.com", "bloomberg.com", "economist.com",
    "sciencedirect.com", "link.springer.com", "onlinelibrary.wiley.com",
    "ieeexplore.ieee.org", "dl.acm.org", "jstor.org", "nzz.ch",
})
# POLICY section 4.9/4.8: social platforms carry no authority and no admission path.
SOCIAL_HOSTS = frozenset({
    "x.com", "twitter.com", "facebook.com", "instagram.com", "tiktok.com",
    "linkedin.com", "pinterest.com", "youtube.com", "snapchat.com",
})
# Subscription-gated vendor docs (D2 doubt): no rule settles it.
GATED_DOUBT = frozenset({"docs.enterprise.spring.io"})


@dataclass(frozen=True)
class Scope:
    klass: str
    rule: str
    clause: str


def host_of(url: str) -> str:
    h = urllib.parse.urlparse(url if "//" in url else "//" + url).netloc.lower()
    return h.split("@")[-1].split(":")[0]


def _bare(h: str) -> str:
    return h[4:] if h.startswith("www.") else h


def _under(h: str, family: frozenset) -> bool:
    b = _bare(h)
    return any(b == f or b.endswith("." + f) for f in family)


def served_rows(host: str, served: Mapping[str, int]) -> int:
    b = _bare(host)
    return max(served.get(host, 0), served.get(b, 0), served.get("www." + b, 0))


def classify_url(url: str, served: Mapping[str, int] | None = None) -> Scope:
    h = host_of(url)
    if h in MIRROR_HOSTS or h.startswith("mirrors."):
        return Scope(OUT, "docs-mirror", "POLICY 3 D3 + 4.4 (not a docs-mirror)")
    if _under(h, PAYWALL_HOSTS):
        return Scope(OUT, "paywalled", "POLICY 3 D2 / U3")
    if _under(h, SOCIAL_HOSTS):
        return Scope(OUT, "social", "POLICY 4.8 (no authority; no admission path)")
    if h in GATED_DOUBT:
        return Scope(UNDECIDED, "gated-vendor-docs", "POLICY 3 D2 doubt; 4.4 vendor boundary")
    n = served_rows(h, served or {})
    if n > 0:
        return Scope(IN, f"served-on-twin({n})", "index serves the host (admitted)")
    if h.startswith("docs.") or h.endswith(".readthedocs.io"):
        return Scope(IN, "docs-subdomain", "POLICY 4.4 vendor boundary (docs subdomains keep the docs class)")
    return Scope(UNDECIDED, "no-rule", "none")


def doi_prefix(doi: str) -> str:
    return doi.split("/", 1)[0].lower()


def classify_doi(doi: str) -> Scope:
    """A scholarly work: admitted by family (OpenAlex/Crossref CC0 backbone,
    DOI-landing pointers for paywalled papers). Whether this single work is in
    the capped OpenAlex subset is not knowable by keyed filter (the twin keys
    rows by OpenAlex id, not DOI), so the scope is by family, not by row."""
    return Scope(IN, "scholarship-family", "POLICY 4.2 + PORTFOLIO 4.2a (OpenAlex backbone)")


# ---- scoring -------------------------------------------------------------

def half_table(rows: Iterable[tuple[Scope, bool]]) -> dict:
    """rows: (scope, hit). Returns counts and Wilson CIs per half."""
    c = Counter()
    for sc, hit in rows:
        c[(sc.klass, bool(hit))] += 1
    out = {}
    tot = sum(c.values())
    for k in (IN, UNDECIDED, OUT):
        n = c[(k, True)] + c[(k, False)]
        hits = c[(k, True)]
        lo, hi = wilson(hits, n) if n else (0.0, 0.0)
        out[k] = {"n": n, "hits": hits, "rate": hits / n if n else None, "ci": [lo, hi]}
    lo, hi = wilson(sum(c[(k, True)] for k in (IN, UNDECIDED, OUT)), tot) if tot else (0.0, 0.0)
    out["all"] = {"n": tot, "hits": sum(v for (k, h), v in c.items() if h),
                  "rate": sum(v for (k, h), v in c.items() if h) / tot if tot else None, "ci": [lo, hi]}
    olo, ohi = wilson(out[OUT]["n"], tot) if tot else (0.0, 0.0)
    out["out_share"] = {"rate": out[OUT]["n"] / tot if tot else None, "ci": [olo, ohi]}
    return out


def missed_hosts(rows: Iterable[tuple[str, Scope, bool]], served: Mapping[str, int]) -> list[dict]:
    """rows: (gold_url, scope, hit). One row per in/undecided host with >= 1 miss."""
    miss: Counter = Counter()
    seen: Counter = Counter()
    scope_of: dict[str, Scope] = {}
    for url, sc, hit in rows:
        if sc.klass == OUT:
            continue
        h = _bare(host_of(url))
        seen[h] += 1
        scope_of.setdefault(h, sc)
        if not hit:
            miss[h] += 1
    res = []
    for h, m in miss.items():
        sc = scope_of[h]
        n = served_rows(h, served)
        if n == 0:
            route = "admission: no rows served; canonical-docs route via sitemap_ingest (POLICY 4.4)"
        else:
            route = f"coverage depth: host served ({n} rows) but gold page missed; deepen crawl or fix ranking, no admission"
        res.append({"kind": "host", "host": h, "n_queries_missed": m, "n_queries": seen[h],
                    "scope": sc.klass, "rule": sc.rule, "served_rows_on_twin": n,
                    "suggested_route": route, "policy_clause": sc.clause})
    res.sort(key=lambda r: (-r["n_queries_missed"], r["host"]))
    return res


# ---- read-only twin probe --------------------------------------------------

def probe(hosts: Iterable[str], qdrant: str, collection: str) -> dict[str, int]:
    """Keyed exact count on the indexed ``domain`` field, one call per host
    variant. No scroll. Run it against a twin only."""
    out: dict[str, int] = {}
    for h in sorted(set(hosts)):
        for v in sorted({h, _bare(h), "www." + _bare(h)}):
            body = json.dumps({"filter": {"must": [{"key": "domain", "match": {"value": v}}]}, "exact": True}).encode()
            rq = urllib.request.Request(f"{qdrant}/collections/{collection}/points/count", body,
                                        {"Content-Type": "application/json"})
            out[v] = json.load(IH.urlopen(rq, timeout=60))["result"]["count"]
    return out


def _jl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def report(dir1024: str, dir1028: str, served: Mapping[str, int]) -> tuple[dict, list[dict]]:
    """Two-half tables for the 1024 tracks and the 1028 ladder, plus the
    admission-candidate rows. Inputs are the per-query artifacts of those tasks."""
    tabs: dict = {}
    url = _jl(f"{dir1024}/webcode-url-per-query.jsonl")
    sc = {r["id"]: classify_url(r["gold"], served) for r in url}
    tabs["webcode_url_at10"] = half_table((sc[r["id"]], r["hit_at_10"]) for r in url)
    rag = _jl(f"{dir1024}/webcode-rag-per-query.jsonl")
    tabs["webcode_rag_correct"] = half_table((sc[r["id"]], r["judge"].get("correctness") == "CORRECT") for r in rag)
    tabs["webcode_rag_grounded"] = half_table((sc[r["id"]], r["judge"].get("groundedness") == "GROUNDED") for r in rag)
    for name in ("publication", "publication-tot"):
        rows = _jl(f"{dir1024}/{name}-per-query.jsonl")
        tabs[name.replace("-", "_") + "_at10"] = half_table(
            (classify_doi(r["gold"]["doi"]), r["hit_at_10"]) for r in rows)
    hops = {r["id"]: r for r in _jl(f"{dir1028}/webcode-url-hops.jsonl")}
    rungs = {"url": lambda h: h["url"], "host": lambda h: h["host"],
             "hop1_findable": lambda h: h["hop1_findable"], "hop2_findable": lambda h: h["hop2_findable"],
             "cumulative": lambda h: h["url"] or h["hop1_findable"] or h["hop2_findable"]}
    for k, f in rungs.items():
        tabs["ladder_" + k] = half_table((sc[i], bool(f(h))) for i, h in hops.items())
    cands = missed_hosts(((r["gold"], sc[r["id"]], r["hit_at_10"]) for r in url), served)
    pub = _jl(f"{dir1024}/publication-per-query.jsonl") + _jl(f"{dir1024}/publication-tot-per-query.jsonl")
    pm: Counter = Counter()
    pn: Counter = Counter()
    for r in pub:
        p = doi_prefix(r["gold"]["doi"])
        pn[p] += 1
        pm[p] += 0 if r["hit_at_10"] else 1
    s0 = classify_doi("")
    for p, m in pm.items():
        if m:
            cands.append({"kind": "doi-prefix", "host": "doi:" + p, "n_queries_missed": m, "n_queries": pn[p],
                          "scope": s0.klass, "rule": s0.rule, "served_rows_on_twin": None,
                          "suggested_route": "scholarship family: OpenAlex depth for this registrant prefix (PORTFOLIO 4.2a draw)",
                          "policy_clause": s0.clause})
    cands.sort(key=lambda r: (-r["n_queries_missed"], r["host"]))
    return tabs, cands


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="count served rows per host on a twin (keyed, read-only)")
    p.add_argument("--qdrant", required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("hosts", nargs="+")
    r = sub.add_parser("report", help="two-half tables + admission candidates from the per-query artifacts")
    r.add_argument("--dir1024", required=True)
    r.add_argument("--dir1028", required=True)
    r.add_argument("--served", required=True, help="JSON from `probe`")
    r.add_argument("--out-tables", required=True)
    r.add_argument("--out-candidates", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        json.dump(probe(a.hosts, a.qdrant, a.collection), sys.stdout, indent=1)
        return 0
    served = json.load(open(a.served))
    tabs, cands = report(a.dir1024, a.dir1028, served)
    json.dump(tabs, open(a.out_tables, "w"), indent=1)
    with open(a.out_candidates, "w") as f:
        for c in cands:
            f.write(json.dumps(c, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
