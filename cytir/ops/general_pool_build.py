"""Task 206 Part A/B (restored into cyt ops, task 1088: `cyt ops general-pool-build [OUT] [--data DIR]`) — build the REAL general-query candidate pool.

Sources the messy real-query distribution AW's agent users actually issue:
  * data/labels_search_arena.jsonl   — 3,786 lmarena search-arena user turns
  * data/eval_agentic.yaml           — ~200 DRGym + arena agent queries
  * a hand-curated FLOOD seed list   — the named trust-flood cases (march
    madness teams, startup tips, "top 3 ai right now", cover letters, code
    review best practices, vague AI/news/product queries)

Dedupes by normalized text, drops empties / pure site: operators / absurdly
long template dumps, and emits one JSONL row per query:
  {id, query, lang, src, vertical}
This pool feeds BOTH the general-eval selection (Part A) and the intent-label
training augmentation (Part B); the quarantine split downstream keeps the two
disjoint (204 method: exact + arctic cos>=0.93)."""
from __future__ import annotations
import argparse, json, os, re, hashlib
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA = os.path.join(HERE, "data")

# The named live floods + the vague general classes the head must route "general".
FLOOD_SEED = [
    ("march madness tournament teams", "en", "sports"),
    ("march madness bracket predictions 2025", "en", "sports"),
    ("who is going to win march madness", "en", "sports"),
    ("nba playoff bracket this year", "en", "sports"),
    ("startup management tips", "en", "business"),
    ("how to manage a startup team", "en", "business"),
    ("best advice for first time founders", "en", "business"),
    ("top 3 ai right now", "en", "ai"),
    ("best ai models this year", "en", "ai"),
    ("what is the best llm for coding", "en", "ai"),
    ("coolest ai tools 2025", "en", "ai"),
    ("how to write a good cover letter", "en", "howto"),
    ("tips for a great resume", "en", "howto"),
    ("best practices for code review", "en", "howto"),
    ("how to give good feedback on a pull request", "en", "howto"),
    ("what are the latest tech trends", "en", "news"),
    ("biggest news in tech today", "en", "news"),
    ("best productivity apps", "en", "product"),
    ("what laptop should i buy for programming", "en", "product"),
    ("how to stay motivated while working from home", "en", "howto"),
    ("fun facts about space", "en", "trivia"),
    ("interesting psychology experiments", "en", "trivia"),
    ("what to watch this weekend", "en", "entertainment"),
    ("best restaurants for a date night", "en", "local"),
    ("how to learn a new language fast", "en", "howto"),
]

_ws = re.compile(r"\s+")


def norm(q: str) -> str:
    return _ws.sub(" ", (q or "").strip().lower())


def keep(q: str) -> bool:
    if not q:
        return False
    n = norm(q)
    if len(n) < 3 or len(n) > 300:
        return False
    if n.startswith("site:") or n.startswith('site '):
        return False
    # near-pure operator / url-dump turns aren't natural NL search
    if n.count("http") >= 2:
        return False
    return True


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("out", nargs="?", default="/tmp/gen_pool.jsonl")
    ap.add_argument("--data", default=DEFAULT_DATA, help="directory holding labels_search_arena.jsonl and eval_agentic.yaml")
    args = ap.parse_args(argv)
    OUT, DATA = args.out, args.data
    seen: set[str] = set()
    rows = []

    def add(q, lang, src, vertical):
        if not keep(q):
            return
        n = norm(q)
        if n in seen:
            return
        seen.add(n)
        qid = "g" + hashlib.sha1(n.encode()).hexdigest()[:12]
        rows.append({"id": qid, "query": q.strip(), "lang": lang or "en",
                     "src": src, "vertical": vertical})

    # 1) flood seed first (priority; these are the anchor cases)
    for q, lang, v in FLOOD_SEED:
        add(q, lang, "flood_seed", v)

    # 2) search-arena user turns
    p = os.path.join(DATA, "labels_search_arena.jsonl")
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            add(r.get("query", ""), r.get("lang"), "search_arena", None)

    # 3) agentic eval (DRGym + arena) — has vertical
    p = os.path.join(DATA, "eval_agentic.yaml")
    d = yaml.safe_load(open(p))
    items = d if isinstance(d, list) else (d.get("queries") or d.get("eval") or [])
    for e in items:
        if isinstance(e, dict):
            add(e.get("query", ""), e.get("lang"), "agentic", e.get("vertical"))

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    bs = Counter(r["src"] for r in rows)
    bl = Counter(r["lang"] for r in rows)
    print(f"pool: {len(rows)} unique queries -> {OUT}")
    print("  by src:", dict(bs))
    print("  by lang:", dict(bl.most_common(8)))
    from cytir.ops._provenance import write_provenance
    write_provenance(OUT, "general-pool-build", {"arena_labels": os.path.join(DATA, "labels_search_arena.jsonl"),
                                                 "eval_agentic": os.path.join(DATA, "eval_agentic.yaml")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
