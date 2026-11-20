#!/usr/bin/env python3
"""cytir/probe/sample.py — the SAMPLER leg of the standing random-probe quality loop
(task 173).

Picks N queries per run (default 25) for probing against live prod:
  * ~60% VERBATIM random, stratified across the held agent-query datasets
    (probes/datasets.json — DRGym + search-arena today).
  * ~40% ADAPTED to AW's focus topics (CH / AI / tech / foundation-domains /
    FR / DE / CA) via a curated, DETERMINISTIC template bank (no LLM). Each
    adapted query logs its template_id + slot fills + any lang/country param.

Dedup: every emitted query is normalized and checked against probes/seen.txt
(the running ledger of everything ever probed); new picks are appended to it.

No network, no LLM, no prod calls — pure sampling. Output:
    probes/runs/<date>/sample.jsonl   (one JSON record per query)

Usage:
    python3 -m cytir.probe.sample                 # 25, seed=today, date=today
    python3 -m cytir.probe.sample --n 25 --seed 20260711
    python3 -m cytir.probe.sample --date 2026-07-11 --verbatim-frac 0.6
"""
from __future__ import annotations
import argparse
import datetime
import glob
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PROBES = os.path.join(REPO, "probes")
SEEN = os.path.join(PROBES, "seen.txt")
MANIFEST = os.path.join(PROBES, "datasets.json")

FOCUS_TOPICS = ["switzerland", "ai_ml", "software", "foundation-domains",
                "france", "germany", "canada"]


# ---------------------------------------------------------------- normalize --
def norm(q: str) -> str:
    """Normalization key for dedup: lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def load_seen() -> set:
    if not os.path.exists(SEEN):
        return set()
    with open(SEEN, encoding="utf-8") as f:
        return {ln.rstrip("\n") for ln in f if ln.strip()}


def append_seen(keys, seen=None):
    """Append-only. Keys already in the file (a re-eligible query drawn again) are
    not written twice."""
    known = load_seen() if seen is None else seen
    with open(SEEN, "a", encoding="utf-8") as f:
        for k in keys:
            if k not in known:
                f.write(k + "\n")


def last_probed(runs_dir=None, before=None):
    """normalized query -> ISO date of its LAST appearance in runs/<date>/sample.jsonl
    (seen.txt carries no dates). Days >= `before` are ignored. A query never found
    in a run has no date and stays seen forever."""
    runs_dir = runs_dir or os.path.join(PROBES, "runs")
    last = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "sample.jsonl"))):
        day = os.path.basename(os.path.dirname(path))
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) or (before and day >= before):
            continue
        with open(path, encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                if r.get("query"):
                    last[norm(r["query"])] = day  # sorted dirs -> latest wins
    return last


def blocking_seen(seen, today, window_days, runs_dir=None):
    """Task 2004: the part of the seen ledger that still blocks a draw. A query last
    probed more than `window_days` before `today` is eligible again. seen.txt is not
    rewritten; window_days <= 0 keeps the old never-repeat policy."""
    if window_days <= 0:
        return set(seen)
    cutoff = (datetime.date.fromisoformat(today) - datetime.timedelta(days=window_days)).isoformat()
    last = last_probed(runs_dir, before=today)
    return {k for k in seen if not (k in last and last[k] < cutoff)}


# ------------------------------------------------------------------ datasets --
def load_datasets():
    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)
    out = []
    for spec in man["datasets"]:
        paths = sorted(glob.glob(spec["glob"]))
        if not paths:
            print(f"  WARN: no file for dataset '{spec['name']}' "
                  f"(glob {spec['glob']}) — skipping", file=sys.stderr)
            continue
        path = paths[-1]  # newest
        rows = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                q = d.get(spec["query_field"])
                if not q or not q.strip():
                    continue
                stratum = "|".join(str(d.get(s, "")) for s in spec.get("strata", []))
                rows.append({
                    "query": q.strip(),
                    "lang": d.get(spec.get("lang_field", "lang")),
                    "source_id": d.get("id"),
                    "stratum": stratum,
                })
        if rows:
            out.append({"name": spec["name"], "path": path, "rows": rows})
            print(f"  loaded {len(rows):>5} rows from {spec['name']} ({os.path.basename(path)})",
                  file=sys.stderr)
    return out


def sample_verbatim(datasets, n, rng, seen):
    """Round-robin across datasets (=> stratified across datasets); within each
    dataset shuffle by stratum so verticals/langs spread. Skips dedup hits."""
    # shuffle each dataset's rows, but order by stratum-interleave for spread
    pools = []
    for ds in datasets:
        rows = ds["rows"][:]
        rng.shuffle(rows)
        # stable interleave across strata within the dataset
        by_str = {}
        for r in rows:
            by_str.setdefault(r["stratum"], []).append(r)
        buckets = list(by_str.values())
        rng.shuffle(buckets)
        interleaved = []
        i = 0
        while any(buckets):
            b = buckets[i % len(buckets)] if buckets else None
            if b:
                interleaved.append(b.pop())
                if not b:
                    buckets.remove(b)
                    i -= 1
            i += 1
            if not buckets:
                break
        pools.append({"name": ds["name"], "rows": interleaved})

    picked, local_seen, di = [], set(seen), 0
    exhausted = set()
    while len(picked) < n and len(exhausted) < len(pools):
        pool = pools[di % len(pools)]
        di += 1
        if pool["name"] in exhausted:
            continue
        if not pool["rows"]:
            exhausted.add(pool["name"])
            continue
        r = pool["rows"].pop()
        k = norm(r["query"])
        if k in local_seen:
            continue
        local_seen.add(k)
        picked.append({
            "query": r["query"],
            "origin": "verbatim",
            "dataset": pool["name"],
            "source_id": r["source_id"],
            "lang": r["lang"],
            "stratum": r["stratum"],
            "focus_topic": None,
            "template_id": None,
            "adaptation": None,
        })
    return picked


# ----------------------------------------------------- adapted template bank --
# Deterministic focus-topic templates. Each exercises a target domain and, where
# noted, a KNOWN-FRAGILE class (entity/dup, language-leak, listing-shell, PDF-
# asset flood, freshness/staleness, cantonal cross-jurisdiction, trust-over-
# relevance). Slots are filled by seeded random choice; template_id + fills are
# logged so an adapted query is reproducible and auditable. NO LLM.
TEMPLATES = [
    # --- Switzerland: law (language + cantonal-flood fragile) ---
    {"id": "ch-law-de", "focus": "switzerland", "lang": "de", "country": "CH",
     "targets": "language-leak,cantonal-flood",
     "t": "was regelt {art} des Schweizer Obligationenrechts",
     "slots": {"art": ["Artikel 335c (Kündigungsfristen)", "Artikel 336 (missbräuchliche Kündigung)",
                        "Artikel 257 (Miete)", "Artikel 62 (ungerechtfertigte Bereicherung)"]}},
    {"id": "ch-law-fr", "focus": "switzerland", "lang": "fr", "country": "CH",
     "targets": "language-leak,cantonal-flood",
     "t": "que dit {art} du code des obligations suisse",
     "slots": {"art": ["l'article 335c sur le délai de congé", "l'article 328 sur la protection du travailleur",
                        "l'article 41 sur la responsabilité"]}},
    {"id": "ch-tax", "focus": "switzerland", "lang": "de", "country": "CH",
     "targets": "pdf-asset-flood,cantonal-flood",
     "t": "{topic} in der Schweiz offizielle Angaben",
     "slots": {"topic": ["AHV Beiträge für Selbständige", "Mehrwertsteuer Steuerpflicht Grenze",
                          "Quellensteuer Tarif"]}},
    {"id": "ch-stats", "focus": "switzerland", "lang": "en", "country": "CH",
     "targets": "pdf-asset-flood,coverage",
     "t": "latest official {metric} statistic Switzerland from the Federal Statistical Office",
     "slots": {"metric": ["resident population", "consumer price index inflation rate",
                          "unemployment rate", "median gross wage"]}},
    # --- Foundation domains: entities (duplicate-card fragile) ---
    {"id": "entity-card", "focus": "foundation-domains", "lang": "en", "country": None,
     "targets": "duplicate-cards,junk-class",
     "t": "{entity}",
     "slots": {"entity": ["Migros", "Roche", "Swisscom", "ETH Zurich", "Nestlé",
                          "Novartis", "SBB CFF FFS"]}},
    {"id": "entity-uid", "focus": "foundation-domains", "lang": "en", "country": "CH",
     "targets": "duplicate-cards,junk-class",
     "t": "{entity} company official website and registration",
     "slots": {"entity": ["Migros Genossenschafts-Bund", "Postfinance", "Zurich Insurance Group"]}},
    # --- AI/ML: freshness + listing-shell fragile ---
    {"id": "ai-fresh", "focus": "ai_ml", "lang": "en", "country": None,
     "targets": "stale,freshness",
     "t": "latest {model} release documentation and model card",
     "slots": {"model": ["Llama", "Mistral", "Qwen", "DeepSeek", "Claude"]}},
    {"id": "ai-governance", "focus": "ai_ml", "lang": "en", "country": None,
     "targets": "listing-shell,trust-over-relevance",
     "t": "{topic} playbook",
     "slots": {"topic": ["Finance AI governance", "enterprise AI risk management",
                        "MLOps model governance", "AI compliance"]}},
    {"id": "ai-consulting", "focus": "ai_ml", "lang": "en", "country": None,
     "targets": "trust-over-relevance",
     "t": "{firms} AI regulatory playbook",
     "slots": {"firms": ["BCG, McKinsey, Bain", "Deloitte and PwC", "Accenture"]}},
    # --- Software/tech: docs canonicalization + version fragile ---
    {"id": "sw-api", "focus": "software", "lang": "en", "country": None,
     "targets": "canonicalization,version",
     "t": "{sym} official documentation reference",
     "slots": {"sym": ["Python asyncio.gather", "Rust tokio::spawn", "Go strings.Split",
                        "React useEffect", "Node fs.promises.readFile"]}},
    {"id": "sw-spec", "focus": "software", "lang": "en", "country": None,
     "targets": "version,obsolete",
     "t": "current authoritative specification for {spec}",
     "slots": {"spec": ["OAuth 2.1", "HTTP semantics RFC", "JSON Web Token", "TLS 1.3"]}},
    # --- France: institutional (coverage + listing-shell fragile) ---
    {"id": "fr-gov", "focus": "france", "lang": "fr", "country": "FR",
     "targets": "listing-shell,coverage",
     "t": "{topic} démarches officielles en France",
     "slots": {"topic": ["demande de carte vitale", "déclaration impôt sur le revenu",
                          "création d'une SAS", "RGPD droit d'accès"]}},
    {"id": "fr-law", "focus": "france", "lang": "fr", "country": "FR",
     "targets": "coverage,ranking",
     "t": "que dit le {pair}",
     "slots": {"pair": ["code du travail sur le licenciement pour motif personnel",
                        "code du travail sur le préavis de démission",
                        "code civil sur le contrat de bail d'habitation",
                        "code civil sur la responsabilité du fait des choses"]}},
    # --- Germany: gesetze (URL-artifact + cantonal/cross fragile) ---
    {"id": "de-law", "focus": "germany", "lang": "de", "country": "DE",
     "targets": "gesetze-url,ranking",
     "t": "was regelt {para} im deutschen Recht",
     "slots": {"para": ["§ 573 BGB (Kündigung Mietvertrag)", "§ 9 AufenthG",
                        "§ 823 BGB (Schadensersatz)", "§ 1 KSchG (Kündigungsschutz)"]}},
    {"id": "de-benefit", "focus": "germany", "lang": "de", "country": "DE",
     "targets": "coverage",
     "t": "{topic} offizielle Informationen Deutschland",
     "slots": {"topic": ["Elterngeld Antrag", "Regelaltersrente Deutsche Rentenversicherung",
                          "Kindergeld Höhe"]}},
    # --- Canada: federal (coverage + bilingual fragile) ---
    {"id": "ca-gov", "focus": "canada", "lang": "en", "country": "CA",
     "targets": "coverage,language-leak",
     "t": "{topic} official Government of Canada guidance",
     "slots": {"topic": ["employment insurance eligibility", "GST HST registration threshold",
                          "Canada Pension Plan contribution rates"]}},
    {"id": "ca-law", "focus": "canada", "lang": "en", "country": "CA",
     "targets": "coverage,ranking",
     "t": "authoritative text of {law} Canada",
     "slots": {"law": ["the Canada Labour Code termination notice", "the Privacy Act",
                        "the Income Tax Act basic personal amount"]}},
]


def sample_adapted(n, rng, seen):
    picked, local_seen = [], set(seen)
    tries, max_tries = 0, n * 40
    tmpl_order = TEMPLATES[:]
    while len(picked) < n and tries < max_tries:
        tries += 1
        tmpl = rng.choice(tmpl_order)
        fills = {slot: rng.choice(vals) for slot, vals in tmpl["slots"].items()}
        q = tmpl["t"].format(**fills)
        k = norm(q)
        if k in local_seen:
            continue
        local_seen.add(k)
        picked.append({
            "query": q,
            "origin": "adapted",
            "dataset": "focus-template",
            "source_id": None,
            "lang": tmpl["lang"],
            "stratum": tmpl["targets"],
            "focus_topic": tmpl["focus"],
            "template_id": tmpl["id"],
            "adaptation": {"template_id": tmpl["id"], "slots": fills,
                           "country": tmpl.get("country"), "targets": tmpl["targets"]},
            "country": tmpl.get("country"),
        })
    return picked


# ------------------------------------------------------------ re-probe tier --
def sample_reprobe(n, seen, today, min_age_days, runs_dir=None):
    """Task 1482: when fresh pools run dry (verbatim datasets + template bank all in
    seen.txt), re-probe the queries whose LAST probe is oldest, at least
    `min_age_days` old. Last-probe dates come from probes/runs/<date>/sample.jsonl
    (seen.txt carries none). Oldest first; ties by query. Deterministic, no network.
    Rows are copies of the original record, origin='reprobe', so the reviewer can
    compare against the earlier run of the same query."""
    runs_dir = runs_dir or os.path.join(PROBES, "runs")
    cutoff = (datetime.date.fromisoformat(today) - datetime.timedelta(days=min_age_days)).isoformat()
    last = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "sample.jsonl"))):
        day = os.path.basename(os.path.dirname(path))
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) or day >= today:
            continue
        with open(path, encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                if not r.get("query"):
                    continue
                last[norm(r["query"])] = (day, r)  # sorted dirs -> latest wins
    cands = sorted((d, k, r) for k, (d, r) in last.items() if d <= cutoff)
    picked = []
    for d, k, r in cands[:n]:
        row = {x: r.get(x) for x in ("query", "dataset", "source_id", "lang", "stratum",
                                     "focus_topic", "template_id", "adaptation", "country")}
        row["origin"] = "reprobe"
        row["reprobe_of"] = f"{d}"
        picked.append(row)
    return picked


# ------------------------------------------------------------------- driver ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--verbatim-frac", type=float, default=0.6)
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--seed", type=int, default=None,
                    help="default: YYYYMMDD of --date (deterministic per week)")
    ap.add_argument("--reprobe-min-age-days", type=int, default=0,
                    help="task 1482: when fresh pools are exhausted, top up with queries last "
                         "probed >= this many days ago (oldest first). Default 0 = OFF: probes/README says never re-probe Slice A, Tim rules")
    ap.add_argument("--reeligible-days", type=int, default=30,
                    help="task 2004: a seen query last probed more than this many days ago may "
                         "be drawn again (dates from probes/runs; seen.txt untouched). 0 = never repeat")
    ap.add_argument("--no-seen-update", action="store_true",
                    help="do not append picks to seen.txt (dry test)")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else int(args.date.replace("-", ""))
    rng = random.Random(seed)

    print(f"cytir.probe.sample: n={args.n} seed={seed} date={args.date}", file=sys.stderr)
    ledger = load_seen()
    seen = blocking_seen(ledger, args.date, args.reeligible_days)
    print(f"  seen ledger: {len(ledger)} queries ({len(ledger) - len(seen)} re-eligible "
          f"after {args.reeligible_days}d)", file=sys.stderr)

    datasets = load_datasets()
    n_verb = round(args.n * args.verbatim_frac)
    n_adapt = args.n - n_verb

    verb = sample_verbatim(datasets, n_verb, rng, seen) if datasets else []
    if len(verb) < n_verb:
        print(f"  WARN: only {len(verb)}/{n_verb} verbatim (dedup/exhaustion); "
              f"topping up adapted", file=sys.stderr)
        n_adapt += (n_verb - len(verb))
    adapt = sample_adapted(n_adapt, rng, seen | {norm(v["query"]) for v in verb})

    rows = verb + adapt
    if len(rows) < args.n and args.reprobe_min_age_days > 0:
        rep = sample_reprobe(args.n - len(rows), seen, args.date, args.reprobe_min_age_days)
        print(f"  fresh pools short ({len(rows)}/{args.n}); re-probing {len(rep)} queries "
              f"last probed >= {args.reprobe_min_age_days}d ago", file=sys.stderr)
        rows += rep
    for i, r in enumerate(rows):
        r["probe_id"] = f"{args.date}-{i:03d}"
        r["k"] = 10
        r["params"] = {"k": 10, "authorities": 1}
        if r.get("lang"):
            r["params"]["lang"] = r["lang"] if r["lang"] in ("en", "de", "fr", "it") else None
            if r["params"].get("lang") is None:
                r["params"].pop("lang", None)
        if r.get("country"):
            r["params"]["country"] = r["country"]

    outdir = os.path.join(PROBES, "runs", args.date)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "sample.jsonl")
    with open(outpath, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if not rows:
        # task 1058: 0 sampled = pool exhausted (verbatim datasets + template bank all in
        # seen.txt). Say so in run_summary.json and exit non-zero; never pass as complete.
        reason = (f"sampled 0 queries: verbatim and adapted pools exhausted against seen ledger "
                  f"({len(seen)} seen); widen the source or change the dedup policy")
        summ = {"state": "refused", "expected": 0, "recorded": 0, "outcomes": {},
                "missing_probe_ids": [], "complete": False, "gaps": 1, "reason": reason,
                "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        with open(os.path.join(outdir, "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=1)
        print(f"REFUSED: {reason}", file=sys.stderr)
        sys.exit(4)

    if not args.no_seen_update:
        append_seen((norm(r["query"]) for r in rows if r["origin"] != "reprobe"), ledger)

    print(f"\nWROTE {len(rows)} queries -> {outpath}", file=sys.stderr)
    print(f"  verbatim={len(verb)} adapted={len(adapt)} reprobe={len(rows) - len(verb) - len(adapt)}", file=sys.stderr)
    # tiny distribution report
    import collections
    by_ds = collections.Counter(r["dataset"] for r in rows)
    by_focus = collections.Counter(r["focus_topic"] for r in rows if r["focus_topic"])
    print(f"  by dataset: {dict(by_ds)}", file=sys.stderr)
    print(f"  by focus  : {dict(by_focus)}", file=sys.stderr)


if __name__ == "__main__":
    main()
