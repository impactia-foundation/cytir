"""Derive WEAK pairwise relevance labels for fusion tuning from Search Arena
preference votes (task 088, T24) -> data/labels_search_arena.jsonl.

Source: lmarena-ai/search-arena-24k (24,069 convs, 12,652 votes). Each row has
two search-LLM responses to the same user prompt, each side's SHOWN/CITED
result URLs (web_search_trace: ["[n]", url] pairs), and a human vote.

Label derivation, per conversation:
    query          = first user turn (the shared search intent)
    preferred_urls = URLs shown ONLY by the winning side
    rejected_urls  = URLs shown ONLY by the losing side
    (URLs shown by BOTH sides cannot explain the preference -> excluded from
    both lists; their count is kept as shared_urls for diagnostics.)

Filters — keep only votes that plausibly reflect RESULT quality:
    * decisive vote (winner == model_a|model_b); ties and tie(bothbad) skipped
    * BOTH sides actually searched (>=1 shown URL each)
    * the result sets DIFFER (symmetric difference non-empty after URL
      normalization; identical-results votes can only reflect answer style)
    * single-turn conversations only (turn == 1) — in multi-turn convs the
      vote covers the whole dialogue, not one retrieval
    * query 3..80 whitespace tokens, no DLP-redaction placeholders

confidence in (0,1]: (1 - jaccard(url sets)) * decisiveness prior (0.6).
    Low overlap -> the sides retrieved genuinely different evidence -> the
    vote carries more retrieval signal. The 0.6 prior encodes that votes judge
    ANSWERS (style, synthesis, formatting), not results alone — this is weak
    supervision and must never outweigh gold labels (integration recipe in
    the task-088 agentic-datasets note: tie-breaking regularizer, not objective).

License: user prompts CC BY 4.0 (lmarena-ai/search-arena-24k); shown-URL lists
are factual citation metadata from model outputs (provider ToS govern the
full response text, which we do NOT redistribute here).

Usage:
  ~/tei-parity-venv/bin/python -m cytir.ops.extract_arena_labels \
      [--parquet ~/aw-data/hf-datasets/search-arena-chat-24k.parquet] \
      [--out data/labels_search_arena.jsonl] [--max-urls 12]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import urllib.parse

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REDACT_RE = re.compile(r"(?i)\[?(redacted|pii_|email_|phone_)|\bREDACTED\b")
_LANG_MAP = {"English": "en", "German": "de", "French": "fr", "Italian": "it"}


def nurl(u: str) -> str:
    """Same normalization family as tune_fusion._nurl / run_eval.norm."""
    p = urllib.parse.urlsplit((u or "").strip())
    return f"{p.netloc.lower()}{p.path.rstrip('/')}"


def shown_urls(meta: dict) -> list[str]:
    """Ordered, deduped URLs from web_search_trace [["[1]", url], ...]."""
    seen, out = set(), []
    for turn in meta.get("web_search_trace") or []:
        for pair in turn:
            if len(pair) == 2 and isinstance(pair[1], str) and pair[1].startswith("http"):
                n = nurl(pair[1])
                if n and n not in seen:
                    seen.add(n)
                    out.append(pair[1].strip())
    return out


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--parquet", default=os.path.expanduser(
        "~/aw-data/hf-datasets/search-arena-chat-24k.parquet"))
    ap.add_argument("--out", default=os.path.join(_REPO, "data/labels_search_arena.jsonl"))
    ap.add_argument("--max-urls", type=int, default=12,
                    help="cap preferred/rejected lists (shown order preserved)")
    args = ap.parse_args(argv)

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(args.parquet)
    stats = collections.Counter()
    langs = collections.Counter()
    n_out = 0
    cols = ["messages_a", "languages", "winner", "turn",
            "system_a_metadata", "system_b_metadata", "model_a", "model_b"]
    with open(args.out, "w", encoding="utf-8") as out:
        for batch in pf.iter_batches(batch_size=512, columns=cols):
            for r in batch.to_pylist():
                stats["conversations"] += 1
                if r["winner"] not in ("model_a", "model_b"):
                    stats["skip_no_decisive_vote"] += 1
                    continue
                if (r.get("turn") or 0) != 1:
                    stats["skip_multi_turn"] += 1
                    continue
                q = next((m["content"] for m in (r.get("messages_a") or [])
                          if m.get("role") == "user" and m.get("content")), "")
                q = re.sub(r"\s+", " ", q).strip()
                if not (3 <= len(q.split()) <= 80) or _REDACT_RE.search(q):
                    stats["skip_query_unusable"] += 1
                    continue
                ua = shown_urls(r["system_a_metadata"] or {})
                ub = shown_urls(r["system_b_metadata"] or {})
                if not ua or not ub:
                    stats["skip_side_without_results"] += 1
                    continue
                na, nb = {nurl(u) for u in ua}, {nurl(u) for u in ub}
                if na == nb:
                    stats["skip_identical_results"] += 1
                    continue
                win, lose = ("a", "b") if r["winner"] == "model_a" else ("b", "a")
                uw, ul = (ua, ub) if win == "a" else (ub, ua)
                nw, nl = (na, nb) if win == "a" else (nb, na)
                preferred = [u for u in uw if nurl(u) not in nl][: args.max_urls]
                rejected = [u for u in ul if nurl(u) not in nw][: args.max_urls]
                if not preferred:
                    stats["skip_no_winner_only_urls"] += 1
                    continue
                jac = len(nw & nl) / max(len(nw | nl), 1)
                lang = _LANG_MAP.get((r.get("languages") or [None])[0],
                                     ((r.get("languages") or [None])[0] or "other"))
                if isinstance(lang, str) and lang not in ("en", "de", "fr", "it"):
                    lang = lang.lower()[:16]
                langs[lang] += 1
                out.write(json.dumps({
                    "id": f"sa{n_out+1:05d}",
                    "query": q,
                    "lang": lang,
                    "preferred_urls": preferred,
                    "rejected_urls": rejected,
                    "shared_urls": len(nw & nl),
                    "confidence": round((1.0 - jac) * 0.6, 3),
                    "winner_model": r[f"model_{win}"],
                    "loser_model": r[f"model_{lose}"],
                    "source": "lmarena-ai/search-arena-24k",
                }, ensure_ascii=False) + "\n")
                n_out += 1
    from cytir.ops._provenance import write_provenance
    write_provenance(args.out, "extract-arena-labels", {"parquet": args.parquet})
    stats["labels_written"] = n_out
    print(json.dumps(dict(stats), indent=2))
    print("langs:", langs.most_common(12))
    print(f"-> {args.out}")


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main() or 0)
