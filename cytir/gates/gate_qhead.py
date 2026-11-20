#!/usr/bin/env python3
"""ADR-062 Phase-1 BYTE-IDENTICAL gate (ADR-054 persisted-script discipline).

Proves the cold-start unified head is do-no-harm: it runs the IN-PROCESS scorer
(cytir/serve/retrieval.search) against the TWIN aw_v4_sat
(READ-ONLY — never prod, never a full-scan) with AW_QHEAD=1 (all knobs) vs
AW_QHEAD=0 over every gold class + the 2026-07-19 Slice-B probe set, and asserts:

  (1) PER-QUERY ordered top-k IDENTICAL between the two runs  (byte-identical rank),
  (2) => per-class h@1/h@10/MRR delta = 0, no class broken   (ADR-043 pass cond).

(1) is strictly stronger than (2); if any query's ordering differs the gate FAILS
loudly and names it. Both runs share ONE baseline env (only AW_QHEAD / AW_QHEAD_KNOBS
differ), so the experiment is the flag, not a data change (ADR-054).

Run BOTH legs then compare (env is read at import, so one process = one flag state):
  AW_QHEAD=0                         python3 -m cytir.gates.gate_qhead --out /tmp/qh_off.json
  AW_QHEAD=1 AW_QHEAD_KNOBS=all      python3 -m cytir.gates.gate_qhead --out /tmp/qh_on.json
  python3 -m cytir.gates.gate_qhead --compare /tmp/qh_off.json /tmp/qh_on.json

Env for the twin (set the SAME for both legs):
  AW_QDRANT_URL=http://localhost:6333  AW_COLLECTION=aw_v4_sat  AW_RERANK=0
  (+ whatever prod baseline flags you want exercised, e.g. AW_RANKHEAD=1)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse

import yaml

from cytir.core.paths import REPO_ROOT

APP = str(REPO_ROOT)
DATA = os.path.join(APP, "data")

# Gold / probe classes. Each is (name, path). Queries are extracted generically.
SETS = [
    ("authoritative", f"{DATA}/eval_authoritative.yaml"),
    ("names", f"{DATA}/eval_names.yaml"),
    ("legal", f"{DATA}/eval_legal_authority.yaml"),
    ("jurisdiction", f"{DATA}/eval_jurisdiction.yaml"),
    ("commercial", f"{DATA}/eval_commercial.yaml"),
    ("general_flood", f"{DATA}/eval_general.yaml"),
    ("entity_cards", f"{DATA}/eval_entity_cards.yaml"),
    ("zero_signal", f"{DATA}/eval_zero_signal.yaml"),
    ("freshness", f"{DATA}/eval_freshness_heldout.yaml"),
    ("vocab_robustness", f"{DATA}/eval_vocab_robustness.yaml"),
    ("dom_civic", f"{DATA}/eval/eval-civic.yaml"),
    ("dom_coding", f"{DATA}/eval/eval-coding.yaml"),
    ("dom_entity", f"{DATA}/eval/eval-entity.yaml"),
    ("dom_legal", f"{DATA}/eval/eval-legal.yaml"),
    ("dom_reference", f"{DATA}/eval/eval-reference.yaml"),
    ("dom_research", f"{DATA}/eval/eval-research.yaml"),
]
SLICEB = f"{APP}/probes/runs/2026-07-19/sample_sliceb.jsonl"

# ADR-062 Phase-2 JURISDICTION-FLOOD set. Scoped queries on which the off-country
# knob (knob 12) can fire — the class the 2026-07-19 probe flagged. The flood metric
# is `foreign_in_topk`: count of top-10 results stamped a DEMOTABLE scope country that
# conflicts with the query scope (INTL/EU/None/in-country are exempt by omission).
# These carry an explicit country= (the scope the head is given; Q2 deterministic).
# b012 (CH FADP) reproduces the probe's cross-jurisdiction-leak case with scope forced.
FLOOD_QUERIES = [
    ("flood-b012-fadp",  "que dit le droit d'accès de la personne concernée dans la nLPD suisse révisée", "CH", "fr"),
    ("flood-b000-notice","how much notice to terminate an employment contract in Switzerland after probation", "CH", "en"),
    ("flood-shop-elec",  "best online shop electronics switzerland", "CH", "en"),
    ("flood-shop-shoes", "online shoe store switzerland", "CH", "en"),
    ("flood-car-lease",  "car leasing switzerland", "CH", "en"),
    ("flood-realtor",    "real estate agency zurich", "CH", "en"),
    ("flood-bank-de",    "best savings account germany", "DE", "en"),
    ("flood-stat-pop",   "population statistics switzerland", "CH", "en"),
    ("flood-vat-ch",     "current VAT rates", "CH", "en"),
    ("flood-vat-de",     "current VAT rates", "DE", "en"),
]


# ADR-062 Phase-2b (task 281) SCOPE-DETECTION set. Queries with a KNOWN intended
# country but NO explicit country= arg — they test what the scope layer INFERS.
# `exp` = expected inferred scope (None = must abstain). `via` documents the tier
# that should catch it (kw=keyword lexicon already; gaz=gazetteer rollup NEW;
# head=learned/markerless NEW; abstain=homograph must stay None). The gate reports
# detection off (keyword-only) vs on (keyword+gazetteer+head) and correctness.
SCOPE_DETECTION = [
    # --- already caught by the keyword lexicon (must stay identical) ---
    ("kw-ch-1", "Art. 336 CO termination employment switzerland", "CH", "kw"),
    ("kw-fr-1", "impôt sur le revenu en france", "FR", "kw"),
    ("kw-de-1", "Grundgesetz Deutschland Artikel 5", "DE", "kw"),
    # --- NEW: gazetteer sub-national rollup (the Ontario case + peers) ---
    ("gaz-ca-ontario", "Ontario building code requirements", "CA", "gaz"),
    ("gaz-ca-quebec", "Quebec civil code article 1457", "CA", "gaz"),
    ("gaz-ca-bc", "British Columbia employment standards act", "CA", "gaz"),
    ("gaz-ca-toronto", "business licence Toronto", "CA", "gaz"),
    ("gaz-de-bavaria", "Bavaria trade register Handelsregister", "DE", "gaz"),
    ("gaz-de-nrw", "Nordrhein-Westfalen building permit", "DE", "gaz"),
    ("gaz-fr-idf", "impôts locaux en Île-de-France", "FR", "gaz"),
    ("gaz-it-lombardy", "VAT registration in Lombardy", "IT", "gaz"),
    ("gaz-gb-scotland", "residential tenancy law Scotland", "GB", "gaz"),
    ("gaz-gb-london", "London council tax bands", "GB", "gaz"),
    ("gaz-us-california", "California consumer privacy act text", "US", "gaz"),
    # --- markerless / aliased: only the learned head can catch (may abstain) ---
    ("head-ca-gta", "property transfer tax in the GTA", "CA", "head"),
    ("head-us-socal", "wildfire building codes in SoCal", "US", "head"),
    ("head-ca-belle", "droit du travail dans la belle province", "CA", "head"),
    # --- homographs / ambiguous: MUST abstain (a wrong scope is worse) ---
    ("abst-georgia", "Georgia business registration requirements", None, "abstain"),
    ("abst-washington", "Washington state tax filing", None, "abstain"),
    ("abst-paris", "Paris property tax office", None, "abstain"),
    # --- genuinely unscoped: must abstain ---
    ("none-general", "how does TLS 1.3 handshake work", None, "abstain"),
    ("none-generic", "best practices for REST API pagination", None, "abstain"),
]


def scope_detect(retrieval, q, lang):
    """Mirror serve-time scope resolution: keyword lexicon, then (when the
    country_scope knob is active) the gazetteer + learned head. Returns
    (country|None, source)."""
    _cantons, kw = retrieval._query_jurisdiction(q)
    if kw is not None:
        return kw, "kw"
    if not getattr(retrieval, "_SCOPE_ACTIVE", False):
        return None, "off"
    gaz, gaz_abstain = retrieval._gazetteer_scope(q)
    if gaz is not None:
        return gaz, "gaz"
    if gaz_abstain:
        return None, "abstain-gaz"
    try:
        retrieval.models.embed_query(q)
        vec = retrieval.models.last_full_query_vec()
    except Exception:
        return None, "embed-fail"
    hc, hconf = retrieval._scope_head_predict(vec)
    if hc is not None:
        return hc, f"head:{hconf:.2f}"
    return None, "abstain-head"


def foreign_in_topk(hits, q_country, k=10):
    """Count top-k results stamped a demotable scope country != the query scope.
    Requires the `country` payload field, which search() returns only when a
    country= arg is passed (so this is 0 for unscoped queries by construction)."""
    n = 0
    for h in hits[:k]:
        cy = h.get("country")
        if cy and cy not in ("INTL", "EU", "") and cy != q_country:
            n += 1
    return n


def norm_url(u: str) -> str:
    u = (u or "").strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    u = urllib.parse.unquote(u).replace("_", " ")
    return u.rstrip("/")


def reg_domain(u: str) -> str:
    u = (u or "").strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    return u.split("/")[0].split(":")[0].rstrip(".")


def _walk_rows(obj):
    """Yield every dict that looks like a query row (has query/q) anywhere in the
    parsed YAML — robust to each set's own top-level shape."""
    if isinstance(obj, dict):
        if obj.get("query") or obj.get("q"):
            yield obj
        for v in obj.values():
            yield from _walk_rows(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_rows(v)


def load_rows(path):
    rows = []
    for i, doc in enumerate(d for d in yaml.safe_load_all(open(path)) if d):
        for r in _walk_rows(doc):
            rows.append(r)
    # de-dup by (query,country) preserving order
    seen, out = set(), []
    for r in rows:
        q = r.get("query") or r.get("q")
        key = (q, r.get("country"))
        if q and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def load_sliceb(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def gold_of(r):
    gold_urls = ([r["target_url"]] if r.get("target_url") else []) \
        + list(r.get("gold_urls") or [])
    gold_domains = [reg_domain(g) for g in (r.get("gold_domains") or [])]
    if gold_urls and r.get("intent") != "freshness":
        gold_domains = []
    return gold_urls, gold_domains


def rank_of(urls, gold_urls, gold_domains):
    gu = {norm_url(u) for u in gold_urls}
    gu_page = {u.split("#")[0] for u in gu}
    for i, u in enumerate(urls, 1):
        n = norm_url(u)
        if gu and (n in gu or n.split("#")[0] in gu_page):
            return i
        if gold_domains:
            host = reg_domain(u)
            if any(host == g or host.endswith("." + g) for g in gold_domains):
                return i
    return None


def run(out_path):
    from cytir.serve import retrieval  # noqa: E402
    print(f"AW_QHEAD={os.environ.get('AW_QHEAD','0')} "
          f"AW_QHEAD_KNOBS={os.environ.get('AW_QHEAD_KNOBS','-')} "
          f"COLLECTION={os.environ.get('AW_COLLECTION','?')} "
          f"qhead_model={'loaded' if retrieval._QHEAD_MODEL else 'None'}", flush=True)

    result = {"label": os.environ.get("AW_QHEAD", "0"),
              "env": {k: os.environ.get(k) for k in
                      ("AW_QHEAD", "AW_QHEAD_KNOBS", "AW_RANKHEAD", "AW_COMMERCIAL",
                       "AW_COLLECTION")},
              "sets": {}}

    all_sets = [(n, load_rows(p)) for n, p in SETS if os.path.exists(p)]
    all_sets.append(("slice_b", load_sliceb(SLICEB)))

    for name, rows in all_sets:
        per_q = {}
        h1 = h10 = ngold = 0
        mrr_sum = 0.0
        for j, r in enumerate(rows):
            q = r.get("query") or r.get("q")
            country = r.get("country")
            lang = r.get("lang")
            try:
                hits = retrieval.search(q, k=10, lang=lang, country=country)
            except Exception as e:  # record error identically in both legs
                per_q[str(r.get("id") or r.get("qid") or f"{name}-{j}")] = {
                    "urls": [f"__ERROR__:{type(e).__name__}"], "rank": None}
                continue
            urls = [h.get("url") or "" for h in hits]
            gold_urls, gold_domains = gold_of(r)
            rk = rank_of(urls, gold_urls, gold_domains) if (gold_urls or gold_domains) else None
            qid = str(r.get("id") or r.get("qid") or f"{name}-{j}")
            per_q[qid] = {"urls": urls, "rank": rk}
            if gold_urls or gold_domains:
                ngold += 1
                if rk == 1:
                    h1 += 1
                if rk and rk <= 10:
                    h10 += 1
                if rk:
                    mrr_sum += 1.0 / rk
        result["sets"][name] = {
            "n": len(rows), "n_gold": ngold, "h1": h1, "h10": h10,
            "mrr": round(mrr_sum / ngold, 4) if ngold else None,
            "per_q": per_q,
        }
        print(f"  {name:18s} n={len(rows):>4} gold={ngold:>4} "
              f"h@1={h1:>4} h@10={h10:>4}", flush=True)

    # ADR-062 Phase-2: jurisdiction-flood metric (foreign-in-top-10 on scoped queries)
    flood = {}
    ftot = 0
    for qid, q, ctry, lang in FLOOD_QUERIES:
        try:
            hits = retrieval.search(q, k=10, lang=lang, country=ctry)
        except Exception as e:
            flood[qid] = {"foreign": None, "urls": [f"__ERROR__:{type(e).__name__}"]}
            continue
        fs = foreign_in_topk(hits, ctry, 10)
        ftot += fs
        flood[qid] = {"foreign": fs, "country": ctry,
                      "urls": [h.get("url") or "" for h in hits]}
    result["flood"] = {"total_foreign_top10": ftot, "per_q": flood}
    print(f"  {'FLOOD':18s} total_foreign_in_top10={ftot}  (over {len(FLOOD_QUERIES)} scoped queries)", flush=True)

    # ADR-062 Phase-2b: SCOPE-DETECTION (query-side; measurable regardless of doc
    # stamps). Records the inferred scope + its source per query for this leg.
    scope = {}
    n_correct = 0
    for qid, q, exp, via in SCOPE_DETECTION:
        det, src = scope_detect(retrieval, q, None)
        ok = (det == exp)
        n_correct += int(ok)
        scope[qid] = {"detected": det, "expected": exp, "source": src,
                      "via": via, "ok": ok, "query": q}
    result["scope"] = {"n": len(SCOPE_DETECTION), "n_correct": n_correct,
                       "per_q": scope}
    print(f"  {'SCOPE-DETECT':18s} correct={n_correct}/{len(SCOPE_DETECTION)} "
          f"(scope_active={getattr(retrieval,'_SCOPE_ACTIVE',False)})", flush=True)

    json.dump(result, open(out_path, "w"))
    print(f"wrote {out_path}", flush=True)


def compare(off_path, on_path):
    A = json.load(open(off_path))   # AW_QHEAD=0
    B = json.load(open(on_path))    # AW_QHEAD=1
    print("\n=== BYTE-IDENTICAL GATE  off(QHEAD=0) -> on(QHEAD=1) ===")
    print(f"off env: {A.get('env')}")
    print(f"on  env: {B.get('env')}")
    # ADR-062 Phase-2b: country_scope GRADUATES a knob, so intentional reorderings
    # are EXPECTED — the ADR-043 bar is PER-CLASS NON-REGRESSION on h@1 / h@10 / MRR
    # (net-non-negative + no-class-broken), NOT byte-identical ordering. order-diff
    # is shown as informational (how many queries reordered); a per-gold rank drop
    # is caught by the h@1/h@10/MRR deltas + the explicit rank-regression scan.
    print(f"\n{'class':18s} {'n':>4} {'gold':>4} {'h@1 Δ':>8} {'h@10 Δ':>8} "
          f"{'MRR Δ':>9} {'reordered':>10}  verdict")
    all_ok = True
    total_qs = total_diff = 0
    rank_regressions = []
    for name in A["sets"]:
        a, b = A["sets"][name], B["sets"].get(name, {})
        d1 = b.get("h1", 0) - a.get("h1", 0)
        d10 = b.get("h10", 0) - a.get("h10", 0)
        dmrr = round((b.get("mrr") or 0.0) - (a.get("mrr") or 0.0), 4)
        diffs = []
        for qid, av in a["per_q"].items():
            bv = b["per_q"].get(qid, {})
            if av.get("urls") != bv.get("urls"):
                diffs.append(qid)
            # explicit per-gold rank regression: rank got strictly worse (or lost).
            ra, rb = av.get("rank"), bv.get("rank")
            if ra is not None and (rb is None or rb > ra):
                rank_regressions.append((name, qid, ra, rb))
        total_qs += len(a["per_q"])
        total_diff += len(diffs)
        # NON-REGRESSION verdict (ADR-043): no class may drop on h@1/h@10/MRR.
        ok = (d1 >= 0 and d10 >= 0 and dmrr >= -0.0005)
        all_ok = all_ok and ok
        verdict = "OK" if ok else "*** REGRESSED ***"
        print(f"{name:18s} {a['n']:>4} {a.get('n_gold',0):>4} "
              f"{d1:>+8} {d10:>+8} {dmrr:>+9} {len(diffs):>10}  {verdict}")
    print(f"\nTOTAL queries={total_qs}  reordered={total_diff}  "
          f"per-gold rank-regressions={len(rank_regressions)}")
    if rank_regressions:
        print("  RANK REGRESSIONS (gold dropped):")
        for name, qid, ra, rb in rank_regressions[:20]:
            print(f"    {name}/{qid}: {ra} -> {rb}")

    # ADR-062 Phase-2: jurisdiction-flood before/after (foreign-in-top-10, scoped).
    fa, fb = A.get("flood"), B.get("flood")
    if fa and fb:
        print("\n=== JURISDICTION-FLOOD (foreign-in-top-10, scoped queries) ===")
        print(f"{'query':22s} {'off':>5} {'on':>5} {'Δ':>5}")
        fa_q, fb_q = fa["per_q"], fb["per_q"]
        for qid in fa_q:
            a_f = fa_q[qid].get("foreign")
            b_f = fb_q.get(qid, {}).get("foreign")
            d = (b_f - a_f) if (a_f is not None and b_f is not None) else None
            mark = "" if not d else ("  <-- improved" if d < 0 else "  <-- WORSE")
            print(f"{qid:22s} {str(a_f):>5} {str(b_f):>5} {str(d):>5}{mark}")
        dt = fb["total_foreign_top10"] - fa["total_foreign_top10"]
        print(f"{'TOTAL':22s} {fa['total_foreign_top10']:>5} {fb['total_foreign_top10']:>5} "
              f"{dt:>+5}   ({'flood DROPS' if dt < 0 else 'no flood change' if dt == 0 else 'flood RISES'})")

    # ADR-062 Phase-2b: SCOPE-DETECTION before/after (query-side accuracy).
    sa, sb = A.get("scope"), B.get("scope")
    if sa and sb:
        print("\n=== SCOPE-DETECTION (query-side; off=keyword-only  on=+gaz+head) ===")
        print(f"{'id':18s} {'expect':>7} {'off':>6} {'on':>7} {'src(on)':>12}  verdict")
        gained = regressed = 0
        for qid in sa["per_q"]:
            av, bv = sa["per_q"][qid], sb["per_q"].get(qid, {})
            a_det, b_det, exp = av.get("detected"), bv.get("detected"), av.get("expected")
            # regression = off was correct, on is not (a wrong/lost scope)
            a_ok, b_ok = (a_det == exp), (b_det == exp)
            if b_ok and not a_ok:
                gained += 1
            if a_ok and not b_ok:
                regressed += 1
            mark = ("  <-- GAINED" if (b_ok and not a_ok) else
                    "  *** REGRESSED ***" if (a_ok and not b_ok) else
                    "  ok" if b_ok else "  miss")
            print(f"{qid:18s} {str(exp):>7} {str(a_det):>6} {str(b_det):>7} "
                  f"{str(bv.get('source')):>12}{mark}")
        print(f"scope-detection: off={sa['n_correct']}/{sa['n']}  "
              f"on={sb['n_correct']}/{sb['n']}  gained={gained}  regressed={regressed}")

    ship = all_ok and fa and fb and (fb["total_foreign_top10"] < fa["total_foreign_top10"])
    print("\nGATE (byte-identical / no-class-broken):",
          "PASS" if all_ok else "FAIL — see BROKEN classes above")
    if fa and fb:
        print("SHIP-CRITERION (zero-regression AND flood measurably drops):",
              "MET — graduate the knob" if ship else
              "NOT MET — zero-regression holds but flood does not drop; do NOT graduate")
    return 0 if all_ok else 1


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("OFF", "ON"))
    args = ap.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if not args.out:
        ap.error("--out required unless --compare")
    run(args.out)


if __name__ == "__main__":
    main()
