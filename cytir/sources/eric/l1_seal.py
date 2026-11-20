#!/usr/bin/env python3
"""Task 669/C: the ADR-105 L1-coverage proving step for ERIC -- closing the
same gap `cc_tranche_seal_watchdog.sh` already closes for CC (task 664,
2026-09-26 incident: a tranche's registration ran clean, but the shared
persisted-input index never picked it up, and a drain hours later refused
84,579 of 84,582 pages with `UnpersistedInputs`).

REAL INCIDENT THIS CLOSES
--------------------------
`eric-official-pdf-tranche-020000` refused 10,000 of 10,000 pages with
`UnpersistedInputs` (ADR-105). Its materialized markdown --
`finalize_eric_pdf_materialization_workers.py`'s consolidated
`markdown/documents.jsonl.gz` under the tranche's `AW_ERIC_PARENT_ROOT` --
already carries exactly the `l1-release-gz` store's expected shape
(`url`/`markdown_sha256`/`markdown`, `data/persisted-input-stores.v1.json`).
It was simply never FILED where the registry's `l1/**/*.jsonl.gz` glob
looks: no script anywhere in the ERIC chain
(`chain_eric_pdf_partition_admission.sh`, `chain_all_tranches.sh`,
`embed_compose_enqueue.sh`) ever calls `register_l1_release.py`. Tranche
010000 happened to pass this same check only because an earlier, one-off
manual registration covered it -- not because the chain does this
reliably.

BOTH PRIMITIVES BELOW ARE ALREADY BUILT, ALREADY TESTED, FAMILY-AGNOSTIC
(task 641/664) -- reused unchanged. This module supplies only the one
ERIC-specific fact: where its consolidated markdown shard actually lives.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

from cytir.corpus import register_l1_release as RL1  # noqa: E402
from cytir.corpus import verify_l1_coverage as VLC  # noqa: E402

#: The variable `persisted_inputs.sqlite_index_dir` reads; the drain exports it.
SQLITE_DIR_ENV = "AW_PERSISTED_INDEX_SQLITE_DIR"
MARKDOWN_SUBDIR = "markdown"
MARKDOWN_FILENAME = "documents.jsonl.gz"


class EricL1SealError(RuntimeError):
    """Registration ran, or the coverage check after it, refused."""


def register(parent_root: str, *, release_name: str, corpus_root: str | None = None) -> dict:
    """File `<parent_root>/markdown/` into `<corpus_root>/l1/release=<release_name>/` (step 1 of
    `seal`; the `eric_official_pdf` adapter's `l1()` calls it with `corpus_root` = its out dir).
    Raises `EricL1SealError` when the registration refuses."""
    try:
        return RL1.register_l1_release(str(pathlib.Path(parent_root) / MARKDOWN_SUBDIR),
                                       release_name, corpus_root=corpus_root)
    except RL1.RegisterL1Error as exc:
        raise EricL1SealError(f"{release_name}: L1 registration FAILED: {exc}") from exc


def seal(parent_root: str, *, release_name: str, sample: int = 200,
        corpus_root: str | None = None, refresh_index: bool = True,
        log=lambda msg: print(msg, flush=True)) -> dict:
    """Register `<parent_root>/markdown/documents.jsonl.gz` into the `l1/`
    tree, then PROVE the shared persisted-input index actually sees it --
    the same two-step discipline `cc_tranche_seal_watchdog.sh` uses
    (register, then verify; registering alone is not proof, task 664's own
    incident). Raises `EricL1SealError` on either step's refusal; never
    partially -- a caller (a chain script) retries the whole thing on its
    next cycle, same as the CC watchdog.
    """
    if refresh_index and not os.environ.get(SQLITE_DIR_ENV):
        # Task 676, 2026-09-30: tranche 080000's seal ran 21 h with this
        # unset. `persisted_inputs.sqlite_index_dir` then falls back to the
        # HDD copy under the corpus root, and the merge began populating a
        # SECOND index from scratch (41 of 157,674 files done, IO-bound at
        # ~5 MB/s) that the promotion drain never reads -- while the real
        # one, on NVMe, already held the registered file. Refuse up front:
        # a seal that proves coverage against the wrong index proves nothing.
        raise EricL1SealError(
            f"{release_name}: {SQLITE_DIR_ENV} is not set -- the coverage "
            "check would refresh (and could start building) a persisted "
            "index the drain does not read. Set it to the drain's index "
            "directory (the operator's nightly promote script exports "
            "the value; the chain's env example carries it).")
    markdown_dir = pathlib.Path(parent_root) / MARKDOWN_SUBDIR
    reg = register(parent_root, release_name=release_name, corpus_root=corpus_root)
    log(f"[eric-l1-seal] {release_name}: registered {reg['files']} file(s) "
       f"-> {reg['dest']}")

    passages_path = str(markdown_dir / MARKDOWN_FILENAME)
    result = VLC.verify(passages_path, sample=sample, require="l1",
                        label=release_name, refresh_index=refresh_index,
                        log=log)
    if result["status"] != "PASS":
        raise EricL1SealError(
            f"{release_name}: L1 coverage NOT PROVEN after registration "
            f"(registered, but not yet visible to the persisted index) -- "
            f"{result['reason']}")
    log(f"[eric-l1-seal] {release_name}: PASS, {result['sampled']} sampled "
       "page(s) persisted")
    return {"register": reg, "verify": result}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parent-root", required=True,
                    help="CYTIR_ERIC_PARENT_ROOT -- the terminal materialization "
                         "root holding markdown/documents.jsonl.gz")
    ap.add_argument("--release-name", required=True,
                    help="l1/release=<this> in the corpus root")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--corpus-root", default=None)
    ap.add_argument("--no-refresh-index", dest="refresh_index",
                    action="store_false", default=True,
                    help="skip the incremental persisted-index refresh "
                        "before the coverage check (testing only -- a real "
                        "chain run always refreshes, same convention "
                        "verify_l1_coverage.py itself uses)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        seal(args.parent_root, release_name=args.release_name,
            sample=args.sample, corpus_root=args.corpus_root,
            refresh_index=args.refresh_index)
    except EricL1SealError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
