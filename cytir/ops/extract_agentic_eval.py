#!/usr/bin/env python3
"""Build data/eval_agentic.yaml — a standing UNLABELED agentic-realism eval set
(task 088, T23) sampled from real agent-issued search queries:

  ~70%  cx-cmu/deepresearchgym-agentic-search-logs  (14.4M real agent queries,
        CC BY 4.0, EN-only corpus) — reservoir-sampled from the full stream,
        then curated.
  ~30%  lmarena-ai/search-arena-24k — search-intent USER TURNS (first turn),
        fr/de/it preferred where available (CC BY 4.0 for user prompts).
        NOTE (verified 2026-07-04): the PUBLIC release does NOT contain the
        model-issued search-query strings — all 13 models ran with
        search_engine="built-in" and web_search_trace holds only
        ["[n]", url] citation pairs (181k pairs checked, zero query strings).
        The task originally asked for model-issued queries; user turns are
        the closest available substitute and are the cleanest-licensed part.

Curation goals (see task 088):
  * keep the REAL distribution's shape (NL fact-seeking questions, 7-13 tok
    majority) — length-shape-matched sampling, no keyword cherry-picking;
  * bias the topical FILTER toward AW verticals (legal/gov/health/tech-ai/
    finance/academic, Europe/CH where present) ~70%, general control ~30%;
  * dedup (incl. template floods), strip PII-ish queries;
  * tag: source_dataset, lang, style (question/keyword/mixed), vertical,
    len_tokens.

NO GOLD LABELS: the eval metric is judge-based (like eval_realistic).

Usage (venv with pyyaml + pyarrow; datasets pre-downloaded to ~/aw-data/hf-datasets):
  cyt ops extract-agentic-eval sample-drgym   # stream TSV -> reservoir jsonl
  cyt ops extract-agentic-eval extract-arena  # parquet -> arena queries jsonl
  cyt ops extract-agentic-eval curate         # both jsonl -> data/eval_agentic.yaml
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re

HF_DIR = os.path.expanduser("~/aw-data/hf-datasets")
DRGYM_TSV = os.path.join(HF_DIR, "agentic_search_sessions.tsv")
ARENA_PQ = os.path.join(HF_DIR, "search-arena-chat-24k.parquet")
RESERVOIR = os.path.join(HF_DIR, "drgym_reservoir.jsonl")
ARENA_Q = os.path.join(HF_DIR, "arena_model_queries.jsonl")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_YAML = os.path.join(_REPO, "data", "eval_agentic.yaml")

SEED = 20260704

# ---------------------------------------------------------------- filters ----
_PII_RE = re.compile(
    r"""(?xi)
    [\w.+-]+@[\w-]+\.\w{2,}                       # email
  | (?<!\w)@\w{3,}                                 # @handle
  | \+?\d[\d\s().-]{8,}\d                          # phone-like digit runs
  | \b\d{1,4}\s+\w+\s+(street|st\.|avenue|ave|road|rd\.|strasse|straße|rue|via)\b
  | \b(my|meine?r?|mon|ma|mes|mio|mia)\s+(name|address|phone|email|account|ssn|iban)\b
  | \b(redacted|private_info|pii)\b
  | \bIBAN\b | \b[A-Z]{2}\d{2}[A-Z0-9]{10,}\b      # IBAN-ish
    """)
# person-lookup intents we drop wholesale (privacy posture: a *named private
# person* + locator intent is unverifiable, so skip the whole intent class)
_PERSON_RE = re.compile(
    r"(?i)\b(linkedin|facebook|instagram) (profile|page) of\b"
    r"|\b(phone number|home address|email address) (of|for)\b"
    r"|\bwho lives at\b|\bdate of birth of\b")

_Q_START = re.compile(
    r"(?i)^(who|what|when|where|why|which|how|whom|whose|is|are|was|were|do|does|"
    r"did|can|could|will|would|should|has|have|qui|que|quel|quelle|quand|où|comment|"
    r"pourquoi|combien|wer|was|wann|wo|wie|warum|welche|wieviel|chi|che|cosa|quando|"
    r"dove|come|perché|quale|quanto)\b")
_STOPWORDS = set("the a an of in on for to and or with from by at is are was were "
                 "der die das und oder für von im le la les de du des et ou il lo "
                 "un une what how".split())

VERTICALS = {
    "legal": re.compile(
        r"(?i)\b(law|laws|statute|legislation|legal|regulation|regulations|directive|"
        r"treaty|court|ruling|judgment|lawsuit|liability|contract law|copyright|"
        r"patent|trademark|gdpr|constitution|criminal code|civil code|"
        r"gesetz\w*|verordnung|urteil|recht(s|lich)\w*|loi|décret|jurisprudence|"
        r"tribunal|code civil|code pénal|legge|decreto|penale|giuridic\w+|"
        r"compliance requirements?|article \d+)\b"),
    "gov": re.compile(
        r"(?i)\b(government|ministry|federal|parliament|senate|congress|municipality|"
        r"visa|passport|citizenship|immigration|census|election|referendum|policy|"
        r"public administration|tax(es|ation)?|customs|tariffs?|sanction\w*|"
        r"european commission|united nations|embassy|regierung|bundesamt|behörde|"
        r"gouvernement|ministère|governo|ministero)\b"),
    "health": re.compile(
        r"(?i)\b(health|medical|medicine|disease|drug|vaccine|therapy|clinical|"
        r"patient|cancer|diabetes|virus|epidemi\w+|pandemic|who guidelines|symptom\w*|"
        r"treatment|hospital|pharma\w*|mental health|nutrition|gesundheit|krankheit|"
        r"impf\w*|santé|maladie|salute|malattia)\b"),
    "tech-ai": re.compile(
        r"(?i)\b(ai|artificial intelligence|machine learning|deep learning|llm|"
        r"language model|neural|transformer|gpt|algorithm\w*|software|programming|"
        r"python|javascript|java|api|cloud|database|kubernetes|linux|open.?source|"
        r"cybersecurity|encryption|blockchain|semiconductor|gpu|quantum computing|"
        r"data science|künstliche intelligenz|intelligence artificielle)\b"),
    "finance": re.compile(
        r"(?i)\b(finance|financial|bank\w*|interest rate|inflation|stock|shares|"
        r"equity|bond|investment|investor|gdp|central bank|monetary|fiscal|"
        r"insurance|pension|mortgage|revenue|earnings|market cap|ipo|dividend|"
        r"exchange rate|currency|crypto\w*|finanz\w*|zins\w*|inflation|banque|"
        r"taux|bourse|banca|borsa)\b"),
    "academic": re.compile(
        r"(?i)\b(research|study|studies|journal|paper|thesis|dissertation|"
        r"university|professor|peer.?review\w*|citation|arxiv|doi|meta.?analysis|"
        r"empirical|hypothesis|experiment\w*|literature review|forschung|"
        r"universität|recherche|université|ricerca|università)\b"),
}
_EU_CH_RE = re.compile(
    r"(?i)\b(switzerland|swiss|schweiz\w*|suisse|svizzer\w+|zurich|zürich|geneva|"
    r"genève|basel|bern|lausanne|europe(an)?|eu\b|germany|german|deutschland|france|"
    r"french|italy|italian|italia|austria|netherlands|belgium|spain|sweden|norway|"
    r"denmark|poland|portugal|greece|ireland|finland|luxembourg|liechtenstein|"
    r"brussels|strasbourg|frankfurt|paris|berlin|münchen|munich|vienna|rome|milan)\b")


def tokens(q: str) -> list[str]:
    return q.split()


def style_of(q: str) -> str:
    t = tokens(q)
    if q.rstrip().endswith("?") or _Q_START.match(q.strip()):
        return "question"
    low = {w.lower().strip(",.;:") for w in t}
    if len(t) <= 6 and not (low & _STOPWORDS):
        return "keyword"
    return "mixed"


def vertical_of(q: str) -> str | None:
    for name, rx in VERTICALS.items():
        if rx.search(q):
            return name
    return None


def is_clean(q: str) -> bool:
    if not (2 <= len(tokens(q)) <= 40):
        return False
    if _PII_RE.search(q) or _PERSON_RE.search(q):
        return False
    if sum(c.isalpha() for c in q) < 0.5 * max(len(q.replace(" ", "")), 1):
        return False  # mostly digits/symbols
    return True


def template_sig(q: str) -> str:
    """Collapse template floods ('Give me an introduction over 200 words for X…')."""
    ws = [w.lower() for w in re.sub(r"[^\w\s]", " ", q).split()]
    return " ".join(ws[:5])


# ------------------------------------------------------------ sample-drgym ---
def sample_drgym(n_reservoir: int, path: str) -> None:
    rng = random.Random(SEED)
    res: list[dict] = []
    seen = 0
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            q = (row.get("query") or "").strip().strip("\r")
            if not q:
                continue
            seen += 1
            rec = {"query": q, "session_len": row.get("session_len"),
                   "query_id": row.get("query_id"),
                   "retrieval_depth": row.get("retrieval_depth")}
            if len(res) < n_reservoir:
                res.append(rec)
            else:
                j = rng.randrange(seen)
                if j < n_reservoir:
                    res[j] = rec
            if seen % 2_000_000 == 0:
                print(f"  streamed {seen/1e6:.0f}M rows", flush=True)
    with open(RESERVOIR, "w") as out:
        for r in res:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"reservoir: {len(res)} of {seen} rows -> {RESERVOIR}")


# ----------------------------------------------------------- extract-arena ---
_LANG_MAP = {"English": "en", "German": "de", "French": "fr", "Italian": "it"}
_REDACT_RE = re.compile(r"(?i)\[?(redacted|pii|email|phone|name)[\]_:]|\bREDACTED\b")


def extract_arena(path: str) -> None:
    """Pull the FIRST USER TURN of each conversation (search intent), tagged
    with the dataset's own language annotation. The public release has no
    model-issued query strings (see module docstring), so user turns are the
    closest available agent-search-intent text; they are also the CC BY 4.0
    licensed part."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    n_conv = n_q = 0
    with open(ARENA_Q, "w") as out:
        for batch in pf.iter_batches(
                batch_size=512, columns=["messages_a", "languages", "primary_intent"]):
            for row in batch.to_pylist():
                n_conv += 1
                langs = row.get("languages") or []
                lang = _LANG_MAP.get(langs[0]) if langs else None
                if lang is None:      # keep AW target languages only
                    continue
                q = next((m["content"] for m in (row.get("messages_a") or [])
                          if m.get("role") == "user" and m.get("content")), None)
                if not q or _REDACT_RE.search(q):
                    continue
                n_q += 1
                out.write(json.dumps(
                    {"query": q.strip(), "lang": lang,
                     "intent": row.get("primary_intent")}, ensure_ascii=False) + "\n")
    print(f"arena: {n_conv} conversations -> {n_q} en/de/fr/it first user turns -> {ARENA_Q}")


# ------------------------------------------------------------------ curate ---
_LANG_HINT = {
    "fr": re.compile(r"(?i)\b(le|la|les|des|une?|est|sont|pour|avec|dans|qui|que|"
                     r"quel(le)?s?|combien|où|français\w*)\b"),
    "de": re.compile(r"(?i)\b(der|die|das|und|ist|sind|für|mit|nicht|ein(e[mnrs]?)?|"
                     r"wie|was|wer|wo|welche[mnrs]?|schweiz)\b|[äöüß]"),
    "it": re.compile(r"(?i)\b(il|lo|gli|delle?|una?|è|sono|per|con|che|chi|come|"
                     r"quando|dove|quale|qual è|perché)\b"),
}


def guess_lang(q: str, hint: str | None = None) -> str:
    if hint and str(hint).lower()[:2] in ("fr", "de", "it", "en"):
        return str(hint).lower()[:2]
    scores = {lg: len(rx.findall(q)) for lg, rx in _LANG_HINT.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "en"


def _shape_buckets(lens: list[int]) -> dict[tuple[int, int], float]:
    """Empirical token-length bucket shares — the SHAPE we must preserve."""
    edges = [(1, 4), (5, 6), (7, 9), (10, 13), (14, 20), (21, 40)]
    cnt = collections.Counter()
    for L in lens:
        for lo, hi in edges:
            if lo <= L <= hi:
                cnt[(lo, hi)] += 1
                break
    tot = sum(cnt.values()) or 1
    return {b: cnt[b] / tot for b in edges}


def _bucket(L: int):
    for lo, hi in [(1, 4), (5, 6), (7, 9), (10, 13), (14, 20), (21, 40)]:
        if lo <= L <= hi:
            return (lo, hi)
    return (21, 40)


def _pick_shaped(pool: list[dict], n: int, shape: dict, rng) -> list[dict]:
    """Sample n items from pool matching the target length-shape as closely as
    the pool allows (falls back to neighbouring buckets when short)."""
    by_b = collections.defaultdict(list)
    for it in pool:
        by_b[_bucket(it["len_tokens"])].append(it)
    for lst in by_b.values():
        rng.shuffle(lst)
    want = {b: round(n * s) for b, s in shape.items()}
    picked, leftovers = [], []
    for b, w in want.items():
        got = by_b.get(b, [])
        picked.extend(got[:w])
        leftovers.extend(got[w:])
    rng.shuffle(leftovers)
    picked.extend(leftovers[: max(0, n - len(picked))])
    return picked[:n]


def curate(n_total: int, frac_arena: float, frac_vertical: float) -> None:
    import yaml  # noqa: F401  (only to fail fast if missing)
    rng = random.Random(SEED + 1)

    # ---- DRGym pool
    dr = [json.loads(l) for l in open(RESERVOIR, encoding="utf-8")]
    all_lens = [len(tokens(r["query"])) for r in dr]
    shape = _shape_buckets(all_lens)
    print("DRGym reservoir length shape:",
          {f"{lo}-{hi}": round(s, 3) for (lo, hi), s in shape.items()})

    seen_sig: collections.Counter = collections.Counter()
    seen_q: set[str] = set()
    clean = []
    for r in dr:
        q = re.sub(r"\s+", " ", r["query"]).strip()
        if not is_clean(q):
            continue
        nq = q.lower()
        if nq in seen_q:
            continue
        sig = template_sig(q)
        if seen_sig[sig] >= 2:      # cap template floods at 2 per signature
            continue
        seen_q.add(nq)
        seen_sig[sig] += 1
        clean.append({"query": q, "len_tokens": len(tokens(q)),
                      "style": style_of(q), "vertical": vertical_of(q),
                      "eu_ch": bool(_EU_CH_RE.search(q)),
                      "source_dataset": "deepresearchgym-agentic-search-logs",
                      "lang": "en"})
    print(f"DRGym clean pool: {len(clean)} / {len(dr)}")

    n_arena = round(n_total * frac_arena)
    n_dr = n_total - n_arena
    n_dr_vert = round(n_dr * frac_vertical)
    n_dr_gen = n_dr - n_dr_vert

    vert_pool = [c for c in clean if c["vertical"]]
    # prefer Europe/CH-flavoured items inside the vertical pool, but don't
    # distort the length shape: split the vertical quota 40/60 euch/rest
    euch_pool = [c for c in vert_pool if c["eu_ch"]]
    rest_pool = [c for c in vert_pool if not c["eu_ch"]]
    n_euch = min(round(n_dr_vert * 0.4), len(euch_pool))
    dr_vert = _pick_shaped(euch_pool, n_euch, shape, rng)
    chosen_q = {c["query"] for c in dr_vert}
    dr_vert += _pick_shaped([c for c in rest_pool if c["query"] not in chosen_q],
                            n_dr_vert - len(dr_vert), shape, rng)
    gen_pool = [c for c in clean if not c["vertical"]]
    dr_gen = _pick_shaped(gen_pool, n_dr_gen, shape, rng)
    for c in dr_gen:
        c["vertical"] = "general"
    print(f"DRGym picked: {len(dr_vert)} vertical (incl. {n_euch} EU/CH) "
          f"+ {len(dr_gen)} general")

    # ---- Arena pool (model-issued queries; prefer fr/de/it)
    ar_raw = [json.loads(l) for l in open(ARENA_Q, encoding="utf-8")]
    ar_seen: set[str] = set()
    ar_clean = []
    for r in ar_raw:
        q = re.sub(r"\s+", " ", r["query"]).strip()
        if not is_clean(q):
            continue
        nq = q.lower()
        if nq in ar_seen:
            continue
        ar_seen.add(nq)
        lg = guess_lang(q, r.get("lang"))
        ar_clean.append({"query": q, "len_tokens": len(tokens(q)),
                         "style": style_of(q),
                         "vertical": vertical_of(q) or "general",
                         "lang": lg,
                         "source_dataset": "search-arena-24k"})
    print(f"Arena clean pool: {len(ar_clean)} / {len(ar_raw)} "
          f"(langs: {collections.Counter(a['lang'] for a in ar_clean).most_common(6)})")
    # keep the OVERALL set at ~frac_vertical topical bias: arena contributes
    # its share of vertical items too (vertical-first within each lang group)
    n_ar_vert = round(n_arena * frac_vertical)
    ar_fdi = [a for a in ar_clean if a["lang"] in ("fr", "de", "it")]
    ar_en = [a for a in ar_clean if a["lang"] == "en"]
    n_fdi = min(round(n_arena * 0.6), len(ar_fdi))
    n_en = n_arena - n_fdi
    arena = []
    for pool, quota in ((ar_fdi, n_fdi), (ar_en, n_en)):
        v_pool = [a for a in pool if a["vertical"] != "general"]
        g_pool = [a for a in pool if a["vertical"] == "general"]
        v_quota = min(round(quota * (n_ar_vert / max(n_arena, 1))), len(v_pool))
        got = _pick_shaped(v_pool, v_quota, shape, rng)
        got += _pick_shaped(g_pool, quota - len(got), shape, rng)
        arena += got

    items = dr_vert + dr_gen + arena
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["id"] = f"ag{i:03d}"
        it.pop("eu_ch", None)

    hdr = """\
# eval_agentic.yaml — standing agentic-realism eval gate (task 088, T23)
#
# ~{n} REAL agent-issued search queries sampled from:
#   * cx-cmu/deepresearchgym-agentic-search-logs (CC BY 4.0) — 14.4M in-the-wild
#     queries from deep-research-style agents; reservoir-sampled from the full
#     stream, then curated. English-only corpus.
#   * lmarena-ai/search-arena-24k (user prompts CC BY 4.0) — FIRST USER TURNS
#     of search-LLM conversations, fr/de/it preferred. NOTE: the public
#     release contains NO model-issued search-query strings (verified: all
#     web_search_trace entries are ["[n]", url] citation pairs, engines all
#     "built-in") — user turns are the closest available substitute for the
#     consumer-assistant search-intent style. Attribution: see the task doc.
#
# Curation preserved the real distribution's SHAPE (length buckets of the full
# DRGym reservoir; NL fact-seeking majority) while biasing the topical FILTER
# ~70% toward AW verticals (legal/gov/health/tech-ai/finance/academic, EU/CH
# where present) with a ~30% general-knowledge control slice. Deduped
# (template floods capped), PII-suspect queries dropped.
#
# NO GOLD LABELS — these are UNLABELED realism gates. Metric when run is
# judge-based, exactly like data/eval_realistic.yaml (LLM/human judge:
# authoritative-source-in-top10 yes/no, domain-level). Report-only in
# run_agentic_report.py (proxy metrics; no caller since task 724);
# it must NOT gate deploys until it has golds or a standing judge.
#
# Fields: id, query, source_dataset, lang (en|fr|de|it heuristic),
#         style (question|keyword|mixed), vertical (legal|gov|health|tech-ai|
#         finance|academic|general), len_tokens (whitespace).
# Built by {tool} (seed {seed}); methodology and
# stats in task 088.
""".format(n=len(items), seed=SEED, tool="aw ops extract-agentic-eval")  # rename-codemod: keep (persisted: header text written into data/eval_agentic.yaml)

    def yq(s: str) -> str:
        return "'" + s.replace("'", "''") + "'"

    with open(OUT_YAML, "w", encoding="utf-8") as f:
        f.write(hdr + "\nqueries:\n")
        for it in items:
            f.write(f"  - id: {it['id']}\n"
                    f"    query: {yq(it['query'])}\n"
                    f"    source_dataset: {it['source_dataset']}\n"
                    f"    lang: {it['lang']}\n"
                    f"    style: {it['style']}\n"
                    f"    vertical: {it['vertical']}\n"
                    f"    len_tokens: {it['len_tokens']}\n")
    # stats to stdout for the task doc
    def dist(key):
        return dict(collections.Counter(i[key] for i in items).most_common())
    from cytir.ops import _provenance
    _provenance.write_provenance(OUT_YAML, "extract-agentic-eval", {"reservoir": RESERVOIR, "arena_queries": ARENA_Q})
    print(f"\nwrote {len(items)} -> {OUT_YAML}")
    print("by source :", dist("source_dataset"))
    print("by lang   :", dist("lang"))
    print("by style  :", dist("style"))
    print("by vertical:", dist("vertical"))
    lens = [i["len_tokens"] for i in items]
    print(f"len tokens: mean {sum(lens)/len(lens):.1f}  "
          f"shape {[f'{lo}-{hi}:{s:.2f}' for (lo,hi),s in _shape_buckets(lens).items()]}")


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("cmd", choices=["sample-drgym", "extract-arena", "curate"])
    ap.add_argument("--reservoir-size", type=int, default=80_000)
    ap.add_argument("--n-total", type=int, default=200)
    ap.add_argument("--frac-arena", type=float, default=0.30)
    ap.add_argument("--frac-vertical", type=float, default=0.70)
    args = ap.parse_args(argv)
    if args.cmd == "sample-drgym":
        sample_drgym(args.reservoir_size, DRGYM_TSV)
    elif args.cmd == "extract-arena":
        extract_arena(ARENA_PQ)
    else:
        curate(args.n_total, args.frac_arena, args.frac_vertical)


if __name__ == "__main__":
    main()
