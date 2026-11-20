"""The b1 builder's command line (moved from the builder script's `main`, task 2123).

    cyt index b1-rebuild --mode slice  [options]    # ~150k vertical slice + acceptance checks, then STOP
    cyt index b1-rebuild --mode full   [options]    # resumable full rebuild (an explicit go only)
    python3 -m cytir.index.b1_cli --mode {slice,full} [options]

`refuse_cli_write` is still the first call after parsing, before dispatch. Every option and
default is unchanged, and so are the exit codes. Three differences from the builder's `main`:
`main` takes `argv` and `prog`; the directory defaults of `--out-dir` and `--consent-cache`,
the corpus directory and the resume-command prefix are the settings `b1_build_dir`,
`b1_corpus_dir` and `b1_resume_prefix` (read after the hold call, so `--help` and a refused
run touch no setting); and the parser description is the short text below, not the
builder's module docstring."""
from __future__ import annotations

import argparse
import os

from cytir.index import b1_manifest as MF
from cytir.index.b1_manifest import assert_ignition
from cytir.index.b1_rebuild import run_full
from cytir.index.b1_slice import run_slice


def main(argv=None, prog="cyt index b1-rebuild") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="The b1 corpus-replay builder: --mode slice builds the vertical slice and "
                    "runs the acceptance checks; --mode full is the resumable full rebuild.")
    ap.add_argument("--mode", choices=["slice", "full"], default="slice")
    ap.add_argument("--collection", default="aw_b1_slice")
    ap.add_argument("--out-dir", default=None,
                    help="default: the b1_build_dir setting")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--consent-mode", choices=["trust", "live", "cache", "pipelined"], default="trust",
                    help="slice default 'trust' (use corpus verdict; missing->allow-"
                         "flagged); full run should use 'live'")
    ap.add_argument("--consent-grace-days", type=int, default=30)
    ap.add_argument("--consent-census-complete", action="store_true",
                    help="slice+pipelined only: treat the consent cache as the "
                         "COMPLETE census (cache-miss = census-absent -> "
                         "grace-retain, task 319 fix 1)")
    ap.add_argument("--consent-cache", default=None,
                    help="default: consent_cache.json under the b1_build_dir setting")
    ap.add_argument("--news-published", action="append", default=[],
                    help="newsfresh work-dir jsonl(.gz) with url->published; repeatable")
    ap.add_argument("--cards-audit", default="",
                    help="depth296/cards_p2_audit.jsonl (domain->resolve_confidence)")
    ap.add_argument("--corpus-max-rows", type=int, default=6_000_000,
                    help="slice-only corpus scan budget; unmatched keys -> tail")
    ap.add_argument("--max-per-source", type=int, default=0,
                    help="smoke-test cap applied to every SLICE_PLAN selector (0=off)")
    ap.add_argument("--git-sha", default="",
                    help="repo HEAD sha for the MANIFEST (the build dir is not "
                         "a git repo; pass the pinned repo commit)")
    # --- full-run args ---
    ap.add_argument("--fedlex", default="",
                    help="fedlex.jsonl (re-extracted Swiss-law text) folded into the KV")
    ap.add_argument("--romansh", default="",
                    help="romansh_lang_rm.delete-ids.txt (twin point_ids to evict)")
    ap.add_argument("--extra-jsonl", action="append", default=[],
                    help="extra {url,anchor,text,lang} jsonl folded into the KV "
                         "(re-extracted accent-bearing law tail); repeatable")
    ap.add_argument("--coverage-sample", type=int, default=400_000,
                    help="sample size for the pre-assembly tok-split coverage gate")
    ap.add_argument("--scroll-batch", type=int, default=2000,
                    help="twin/staging scroll batch (with vectors)")
    ap.add_argument("--upsert-batch", type=int, default=2000,
                    help="aw_b1 upsert batch")
    ap.add_argument("--gate-only", action="store_true",
                    help="stop after the coverage gate (report, await go)")
    args = ap.parse_args(argv)
    # task 864 (review 869 S1): from an aw_e10 T0 to the flip only the drain writes the twin
    from cytir.index import twin_write_hold as _TWH
    _TWH.refuse_cli_write(args, tool='cytir/index/b1_cli.py')
    from cytir.core import paths
    if args.out_dir is None:
        args.out_dir = paths.get_value("b1_build_dir")
    if args.consent_cache is None:
        args.consent_cache = os.path.join(paths.get_value("b1_build_dir"), "consent_cache.json")
    corpus_dir = paths.get_value("b1_corpus_dir")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.git_sha:
        MF._GIT_SHA_OVERRIDE = args.git_sha.strip()

    def log(*a):
        print(*a, flush=True)

    if args.mode == "slice":
        run_slice(args, log=log, corpus_dir=corpus_dir)
        return 0
    if args.mode == "full":
        # provenance gate (task 319): a full run must come from a clean tree
        # whose HEAD is merged to main — the 318 build ran an unmerged
        # worktree commit and main's builder could not even reproduce it.
        assert_ignition(log=log)
        st, status = run_full(
            args, log=log, corpus_dir=corpus_dir,
            resume_prefix=paths.get_value("b1_resume_prefix"))
        log(f"[full] exit status: {status}")
        return 0 if status in ("DONE", "GATE_PASS") else (3 if status.startswith("STOP") else 0)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
