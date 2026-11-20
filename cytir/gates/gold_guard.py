# Moved from the retired `gold_guard` module (task 677, task 643 S5); the old path is an alias shim.
#!/usr/bin/env python3
"""HARD PRECONDITION for every index cull: never delete a page that is a gold.

Why this exists (task 334 -> 335, 2026-07-27): the ratified hygiene cull
deleted 51 URLs, and 10 of them were the answer some eval case was scored on
— `go.dev/doc/devel/release` is a *literal* `gold_urls` entry (sw-docs-015),
and 9 more were the exact result that satisfied a domain-primary case at its
best rank (`angular.dev/reference/releases` for sw-docs-022,
`kotlinlang.org/docs/releases.html` for sw-docs-030, `kubernetes.io/releases/`
for ho-sw-004/rvt-14, ...). Four of those cases regressed on the spot. Culling
a page that is legitimately the #1 answer for its own query is a self-inflicted
defect: the page is *correct index content*: any flood it causes on OTHER
queries is a RANKING problem, to be fixed in the ranker or in the ingest-side
canonicalizer — never by deleting the authoritative page.

The rule, enforced here:

  * REFUSE (hard, always) — the candidate URL matches a `gold_urls` /
    `target_url` entry in ANY eval set. No override.
  * REFUSE (overridable only by an explicit, listed acknowledgement) — the
    candidate URL is, in a supplied results leg, the SCORING hit of a
    domain-primary case (`intent: freshness` or empty `gold_urls`), i.e. the
    result the case's rank was read off. Pass `--ack-displacer URL` per URL to
    proceed; every ack is echoed loudly and recorded.
  * WARN (informational) — the candidate sits on a gold DOMAIN but was not the
    scoring hit (a lower sibling / displacer). Reported, never blocking.

Gold semantics are imported from `cytir/gates/authority.py` (`gold_of`,
`rank_of`, `norm_url`, `reg_domain`, `SETS`) so the guard can never drift from
what the gate actually scores.

Library use (preferred — call before any delete):

    from cytir.gates.gold_guard import assert_no_golds
    assert_no_golds(urls, leg="/path/leg_pre.json", acks=[...])   # raises

CLI use:

    gold_guard.py --urls cull_urls.json [--leg leg_pre.json]
                  [--ack-displacer URL ...] [--json report.json]
    # exit 0 = clean, 2 = refused (golds found), 3 = bad input
"""
from __future__ import annotations

import argparse
import json
import os
import sys

APP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cytir.gates import authority as G  # noqa: E402  (same gold semantics as the gate)


class GoldCullRefused(RuntimeError):
    """Raised when a cull candidate is a gold (or an un-acked gold answer)."""


def _load_urls(spec):
    """Accepts a list, a plain-text file (one url per line), or a json file
    holding either a list of urls or the 334-style {"cull": [{"url": ...}]}."""
    if isinstance(spec, (list, tuple, set)):
        return [str(u) for u in spec]
    with open(spec) as fh:
        txt = fh.read()
    try:
        doc = json.loads(txt)
    except ValueError:
        return [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if isinstance(doc, dict):
        doc = doc.get("cull") or doc.get("urls") or doc.get("restore_urls") or []
    out = []
    for row in doc:
        out.append(row["url"] if isinstance(row, dict) else str(row))
    return out


# Gold-carrying eval files that the 663-gold GATE does not score, and therefore
# are absent from `gate_authority.SETS`, but whose `target_url`s are golds all
# the same. Found by task 338: `eval_tasks.yaml` carries three PubMed
# `target_url` golds (gx61 SAMtools, gx62 ImageJ, gx63 Toll-like receptors) that
# the PubMed reshape would otherwise have been free to delete — the guard read
# "any eval set" but the code read `SETS`. The gate's scored set list is NOT
# touched (verification-doctrine 2: the yardstick never moves); the guard simply
# refuses on a wider set than the gate scores, which is the safe direction.
EXTRA_GOLD_SETS = [
    ("tasks", f"{G.DATA}/eval_tasks.yaml"),
]


def _eval_rows():
    for name, path in list(G.SETS) + EXTRA_GOLD_SETS:
        if os.path.exists(path):
            for r in G.load_rows(path):
                yield name, (r.get("id") or r.get("case_id") or r.get("query")), r


def audit(urls, leg=None):
    """Classify every candidate url. Returns (gold_urls, gold_answers, on_gold_domain)."""
    cand = {}
    for u in urls:
        cand.setdefault(G.norm_url(u), u)
    # Page-keyed index of the candidates, so the literal-gold test costs
    # O(golds) instead of O(golds x candidates). Task 338 runs this guard over a
    # 23M-URL delete list (the PubMed reshape); the old per-row scan over every
    # candidate was ~1e10 iterations and could not be run at campaign scale — a
    # guard nobody can afford to run is a guard that gets skipped.
    # Semantics are UNCHANGED: the old test was
    #   `n in gu_n or n.split("#")[0] in gu_page`, and since gu_page is exactly
    # {u.split("#")[0] for u in gu_n}, the first disjunct implies the second, so
    # the page test alone is equivalent. (Parity: replaying task 334's cull list
    # reproduces `335-scripts/gold_guard_334.json` exactly.)
    cand_by_page = {}
    for n, raw in cand.items():
        cand_by_page.setdefault(n.split("#")[0], []).append((n, raw))
    legsets = {}
    if leg:
        with open(leg) as fh:
            legsets = json.load(fh).get("sets") or {}

    gold_hits, answer_hits, domain_hits = [], [], []
    for sname, cid, r in _eval_rows():
        gu, gd = G.gold_of(r)
        gu_n = {G.norm_url(u) for u in gu}
        gu_page = {u.split("#")[0] for u in gu_n}
        for page in gu_page:
            for _n, raw in cand_by_page.get(page, ()):
                gold_hits.append({"url": raw, "set": sname, "case": cid,
                                  "query": r.get("query")})
        urls_leg = ((legsets.get(sname) or {}).get("per_q") or {}).get(cid, {}).get("urls") or []
        if not urls_leg:
            continue
        rk = G.rank_of(urls_leg, gu, gd)
        if rk:
            hit = G.norm_url(urls_leg[rk - 1])
            if hit in cand:
                answer_hits.append({"url": cand[hit], "set": sname, "case": cid,
                                    "rank": rk, "query": r.get("query"),
                                    "primary": "url" if gu else "domain"})
        for u in urls_leg:
            n = G.norm_url(u)
            if n in cand and gd:
                host = G.reg_domain(u)
                if any(host == g or host.endswith("." + g) for g in gd):
                    domain_hits.append({"url": cand[n], "set": sname, "case": cid})
    return gold_hits, answer_hits, domain_hits


def assert_no_golds(urls, leg=None, acks=(), report_path=None, stream=sys.stderr):
    """HARD precondition. Raises GoldCullRefused unless the candidate set is clean.

    `acks` = urls explicitly acknowledged as removable gold-domain answers.
    A literal gold_url can NEVER be acked.
    """
    urls = _load_urls(urls)
    gold_hits, answer_hits, domain_hits = audit(urls, leg=leg)
    acked = {G.norm_url(u) for u in acks}
    unacked = [h for h in answer_hits if G.norm_url(h["url"]) not in acked]

    def say(msg):
        print(msg, file=stream, flush=True)

    say(f"[gold_guard] {len(urls)} cull candidates checked against "
        f"{len(G.SETS)} eval sets" + (f" + leg {os.path.basename(leg)}" if leg else ""))
    for h in gold_hits:
        say(f"[gold_guard] REFUSE literal gold_url: {h['url']}  "
            f"({h['set']}/{h['case']})")
    for h in unacked:
        say(f"[gold_guard] REFUSE gold answer (rank {h['rank']}, "
            f"{h['primary']}-primary): {h['url']}  ({h['set']}/{h['case']})")
    for h in answer_hits:
        if G.norm_url(h["url"]) in acked:
            say(f"[gold_guard] ACKED gold-domain answer, proceeding: {h['url']}"
                f"  ({h['set']}/{h['case']} rank {h['rank']})")
    seen = set()
    for h in domain_hits:
        k = (h["url"], h["case"])
        if k not in seen and G.norm_url(h["url"]) not in {G.norm_url(a["url"])
                                                         for a in answer_hits}:
            seen.add(k)
            say(f"[gold_guard] warn: on gold domain (not the answer): "
                f"{h['url']} ({h['set']}/{h['case']})")

    report = {"candidates": len(urls), "gold_urls": gold_hits,
              "gold_answers": answer_hits, "on_gold_domain": domain_hits,
              "acked": sorted(acked),
              "refused": sorted({h["url"] for h in gold_hits} |
                                {h["url"] for h in unacked})}
    if report_path:
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=1, ensure_ascii=False)
    if report["refused"]:
        raise GoldCullRefused(
            f"{len(report['refused'])} cull candidate(s) are golds — refusing: "
            + ", ".join(report["refused"][:8])
            + (" ..." if len(report["refused"]) > 8 else ""))
    say("[gold_guard] OK — no candidate is a gold or a gold answer")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urls", required=True,
                    help="cull candidates: txt (one url/line) or json")
    ap.add_argument("--leg", help="results leg json (gate_authority --out) used "
                                  "to find each case's scoring answer")
    ap.add_argument("--ack-displacer", action="append", default=[],
                    help="explicitly acknowledge removing this gold-domain answer")
    ap.add_argument("--json", help="write the audit report here")
    a = ap.parse_args()
    try:
        assert_no_golds(a.urls, leg=a.leg, acks=a.ack_displacer,
                        report_path=a.json, stream=sys.stdout)
    except GoldCullRefused as e:
        print(f"GOLD-GUARD REFUSED: {e}")
        return 2
    except (OSError, ValueError) as e:
        print(f"GOLD-GUARD INPUT ERROR: {e}")
        return 3
    print("GOLD-GUARD OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
