"""Loader and validator for the GitHub-docs gold set (`data/eval_github_docs_gold.v1.jsonl`).

Eval set for benchmarking only. It is not wired into any gate contract or the 663-gold sets.
No network: `validate` checks every `gold_url` against a URL list the caller passes in.

    cyt ops github-docs-gold [GOLD.jsonl [URLS.txt]]   (default GOLD: the committed data file)

Row: id, query, gold_url (the form AW stores), repo, stars_bucket, stars, page_words,
answer_sentence, author, verified_by, tim_spot_check.
"""
from __future__ import annotations

import json
import pathlib
import sys

REQUIRED = ("id", "query", "gold_url", "repo", "stars_bucket", "page_words", "answer_sentence",
            "author", "verified_by", "tim_spot_check")
BUCKETS = ("ge20k", "10k-20k", "5k-10k", "lt5k")
MIN_WORDS = 300


def load(path: pathlib.Path | str) -> list[dict]:
    rows = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def validate(rows: list[dict], known_urls: set[str] | None = None) -> list[str]:
    """Return a list of problems; empty means the set is sound."""
    errs: list[str] = []
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    for i, r in enumerate(rows):
        tag = r.get("id", f"row{i}")
        for k in REQUIRED:
            if k not in r or r[k] in (None, ""):
                if k == "tim_spot_check" and r.get(k) is False:
                    continue
                errs.append(f"{tag}: missing {k}")
        if r.get("id") in seen_ids:
            errs.append(f"{tag}: duplicate id")
        seen_ids.add(r.get("id"))
        if r.get("gold_url") in seen_urls:
            errs.append(f"{tag}: duplicate gold_url")
        seen_urls.add(r.get("gold_url"))
        if r.get("stars_bucket") not in BUCKETS:
            errs.append(f"{tag}: bad stars_bucket {r.get('stars_bucket')!r}")
        if isinstance(r.get("page_words"), int) and r["page_words"] < MIN_WORDS:
            errs.append(f"{tag}: page_words < {MIN_WORDS}")
        if known_urls is not None and r.get("gold_url") not in known_urls:
            errs.append(f"{tag}: gold_url not in the URL list")
    return errs


GOLD_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "eval_github_docs_gold.v1.jsonl"


def main(argv: list[str] | None = None, prog: str = "cyt ops github-docs-gold") -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog=prog, description="Validate the GitHub-docs gold set (shape, buckets, duplicates, page size); "
        "with URLS.txt also check every gold_url against that list. No network.")
    ap.add_argument("gold", nargs="?", type=pathlib.Path, default=GOLD_PATH,
                    help="gold JSONL (default: the committed data file)")
    ap.add_argument("urls", nargs="?", type=pathlib.Path, help="optional file with one known URL per line")
    args = ap.parse_args(argv)
    if not args.gold.is_file():
        print(f"REFUSED: no gold set at {args.gold} (the public cut ships none)", file=sys.stderr)
        return 1
    rows = load(args.gold)
    urls = None
    if args.urls:
        urls = {u.strip() for u in args.urls.read_text(encoding="utf-8").splitlines() if u.strip()}
    errs = validate(rows, urls)
    for e in errs:
        print(e)
    print(f"{len(rows)} rows, {len(errs)} problems")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
