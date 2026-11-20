#!/usr/bin/env python3
"""Prove a fetch's gated passages are actually VISIBLE to the shared
persisted-input index before a seal watchdog marks a tranche `.sealed`
(task 664, 2026-09-26 incident).

WHY THIS EXISTS: CC tranche4's `register_l1_release.py` step ran cleanly
and wrote an identical-shaped manifest to tranche2/3's (which worked) --
but a drain hours later refused it with `UnpersistedInputs: 84,579 of
84,582 pages have no L1 (ADR-105)`. The registration succeeded; the SHARED
PERSISTED INDEX that `plan()` actually reads never picked the new files
up (`persisted_index_is_stale()` reported "fresh" regardless). Registering
the files was necessary but not sufficient -- nothing had ever checked
that the index a real drain will query actually SEES them. This script is
that check, run once per tranche right after registration, so a coverage
gap is a same-cycle seal-watchdog refusal (retried next tick) instead of a
silent gap a human discovers days later via an unrelated drain.

Refreshes the persisted index (incremental by default -- the same
`promote_queue.refresh_persisted_index()` every drain calls) and then runs
the REAL ADR-105 check (`persisted_inputs.assert_pages_persisted`) against
a sample of the just-registered urls. A sample, not the full tranche: the
check IS the real one `plan()` uses, and a coverage bug is either present
for a tranche's files (this sample WILL see it) or it is not -- there is no
scenario where "most sampled urls pass, but the tranche still refuses",
since coverage is a property of the registered files being scanned at
all, not of any individual url.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys  # noqa: F401  # re-exported: read as V.sys by the characterization test

from cytir.corpus import persisted as PI


def sample_pages(passages_path: str, *, sample: int, seed: int = 0) -> list[dict]:
    """Up to `sample` distinct-url `{"url", "sha"}` pages from a gated
    passages_ok.jsonl (or an equivalent consolidated L1 shard, e.g. ERIC's
    `markdown/documents.jsonl.gz` -- task 669/C) -- `sha` mirrors exactly
    what `persisted_inputs._normalize` accepts (`sha`/`digest`/
    `content_digest`/`markdown_sha256`, whichever the row carries).
    Transparently gzip-aware by extension, the same convention every other
    reader in this codebase uses."""
    urls_seen: dict[str, dict] = {}
    opener = gzip.open if passages_path.endswith(".gz") else open
    with opener(passages_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            url = row.get("url")
            if not url or url in urls_seen:
                continue
            urls_seen[url] = {"url": url,
                              "sha": row.get("digest") or row.get("content_digest")
                              or row.get("markdown_sha256")}
    pages = list(urls_seen.values())
    if len(pages) <= sample:
        return pages
    random.Random(seed).shuffle(pages)
    return pages[:sample]


def verify(passages_path: str, *, sample: int = 200, require: str = "l1",
          label: str = "", refresh_index: bool = True,
          full_refresh: bool = False,
          log=lambda msg: print(msg, flush=True)) -> dict:
    """Returns `{"status": "PASS"|"REFUSED", "reason": str|None,
    "sampled": int}` -- never raises; the caller (a shell watchdog) decides
    what a non-PASS status means, this module only measures."""
    if refresh_index:
        from cytir.index import promote_queue as PQ  # only needed on this path
        refreshed = PQ.refresh_persisted_index(full=full_refresh)
        log(f"[verify-l1-coverage] index "
           f"{'REBUILT' if refreshed['rebuilt'] else 'fresh'} "
           f"({refreshed['reason']})")
    pages = sample_pages(passages_path, sample=sample)
    if not pages:
        return {"status": "REFUSED",
               "reason": f"{passages_path} has no urls to sample",
               "sampled": 0}
    try:
        PI.assert_pages_persisted(pages, require=require,
                                  label=label or passages_path)
    except PI.UnpersistedInputs as exc:
        return {"status": "REFUSED", "reason": str(exc), "sampled": len(pages)}
    return {"status": "PASS", "reason": None, "sampled": len(pages)}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--passages", required=True,
                    help="gated passages_ok.jsonl to sample urls from")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--require", default="l1", choices=("l1", "l1b"))
    ap.add_argument("--label", default="")
    ap.add_argument("--no-refresh-index", dest="refresh_index",
                    action="store_false", default=True,
                    help="skip the incremental persisted-index refresh "
                        "(testing only -- a real seal-watchdog run always "
                        "refreshes)")
    ap.add_argument("--full-refresh", action="store_true",
                    help="force a FULL (non-incremental) index rebuild "
                        "before checking -- expensive, use sparingly")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = verify(args.passages, sample=args.sample, require=args.require,
                    label=args.label, refresh_index=args.refresh_index,
                    full_refresh=args.full_refresh)
    if result["status"] == "PASS":
        print(f"[verify-l1-coverage] PASS: {result['sampled']} sampled "
             f"page(s) persisted ({args.label or args.passages})")
        return 0
    print(result["reason"])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
