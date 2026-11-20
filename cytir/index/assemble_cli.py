"""The ADR-101 assembly binder's command line (task 2118, binder slice A5).

Moved verbatim from the binder's `main`: `cyt index assemble {guards,plan,stage,load,verify}`.
`refuse_cli_write` is still the first call after parsing, before dispatch; every option,
default and exit code is unchanged. `main` returns the exit code instead of calling
`sys.exit` (the `aw` dispatcher and `__main__` below turn it into one).

    python3 -m cytir.index.assemble_cli guards --plan data/assembly/assembly-plan.current.v1.json
    python3 -m cytir.index.assemble_cli plan   --plan ... --work DIR
    python3 -m cytir.index.assemble_cli stage  --work DIR [--sets S,S] [--jobs N]
    python3 -m cytir.index.assemble_cli load   --work DIR --config 256-pq8 --collection aw_e4
    python3 -m cytir.index.assemble_cli verify --work DIR --collection aw_e4
"""
from __future__ import annotations

import argparse
import os

from cytir.core import twin as TWIN
from cytir.core.paths import REPO_ROOT
from cytir.index.assemble_load import cmd_load
from cytir.index.assemble_plan import cmd_guards, cmd_plan
from cytir.index.assemble_stage import cmd_stage
from cytir.index.assemble_verify import cmd_verify


def main(argv=None, prog="cyt index assemble"):
    ap = argparse.ArgumentParser(
        prog=prog,
        description="The ADR-101 assembly binder: sealed 1024-dim populations -> the serving collection.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("guards")
    g.add_argument("--plan", default=str(REPO_ROOT / "data" / "assembly" /
                                         "assembly-plan.current.v1.json"))
    g.add_argument("--work", default=None)
    g.add_argument("--report", default=None)
    g.add_argument("--require-green", action="store_true")
    g.set_defaults(fn=cmd_guards)

    p = sub.add_parser("plan")
    p.add_argument("--plan", default=str(REPO_ROOT / "data" / "assembly" /
                                         "assembly-plan.current.v1.json"))
    p.add_argument("--work", required=True)
    p.add_argument("--artifact-root", default=None)
    p.add_argument("--sets", default=None)
    p.set_defaults(fn=cmd_plan)

    s = sub.add_parser("stage")
    s.add_argument("--work", required=True)
    s.add_argument("--sets", default=None)
    s.add_argument("--force", action="store_true")
    s.add_argument("--consent-mode", default="trust", choices=["trust", "live"])
    s.add_argument("--grace-days", type=int, default=30)
    s.add_argument("--consent-cache", default=str(REPO_ROOT / "data" / "consent_cache.json"))
    s.add_argument("--skip-projection-pin", action="store_true",
                   help="DEPRECATED, no effect (task 2076): the projection pin check was removed.")
    # payload_route="twin": the read-only join for in-place re-embeds (413a).
    s.add_argument("--twin-url", default=None,
                   help="READ-ONLY qdrant holding the base payload (opt-in: unset = no twin join; "
                        "the twin is `python -m cytir.core.twin`). Never written.")
    s.add_argument("--twin-collection", default=TWIN.collection())
    s.add_argument("--twin-batch", type=int, default=1000)
    s.add_argument("--jobs", type=int, default=1,
                   help="stage this many partitions in parallel. Partitions "
                        "are independent (own inputs, own outputs, own "
                        "conservation check); 1 keeps the historical serial "
                        "order for reproducing a prior tree.")
    # task 536: the news `published`-date sidecar. `NEWS_PACKS` rows whose own
    # sealed record has no `published` (the origin was hydrated from a source
    # that never carried one) fall through to this map -- the same
    # `load_published_map` the full-rebuild path (`build_aw_b1.py` main())
    # already uses, reused rather than reimplemented (one truth, one home).
    # This is DISTINCT from the serving-metadata registry below: the registry
    # is keyed off the SEALED membership snapshot (what a REPLACED point used
    # to carry) and cannot know a date for a page never in that snapshot; this
    # sidecar is the news pipeline's own freshly-fetched dates. Empty by
    # default -- historically `assembly_binder.py:1504` passed `{}` here,
    # which is why the fix in task 536 is to wire this flag through rather
    # than construct a map inline.
    s.add_argument("--news-published", action="append", default=[],
                   help="newsfresh work-dir jsonl(.gz) file(s) of "
                        "{url, published}, one per --news-published flag. "
                        "Read by build_aw_b1.load_published_map.")
    # task 515: the serving-metadata registry. A PATH, never a compiled-in
    # location -- the artifact is rebuilt per membership snapshot and its
    # digest is recorded in the stage receipts.
    s.add_argument("--serving-metadata-registry", default=os.environ.get(
        "AW_SERVING_METADATA_REGISTRY"),
        help="sqlite built by the task-515 serving-metadata registry build script "
             "from the sealed membership snapshot. Without it the projection "
             "passes serving metadata through and a sealed embedding-stage "
             "record ships with none (task 514: 36.5%% of the collection).")
    s.add_argument("--serving-metadata-domain-registry", default=os.environ.get(
        "AW_SERVING_METADATA_DOMAIN_REGISTRY"),
        help="per-domain fallback registry (JSON) for URLs the snapshot does "
             "not carry -- e.g. a source re-materialized onto a different URL "
             "surface.")
    s.add_argument("--serving-metadata-manifest", default=None,
                   help="the registry's manifest; recorded, not read for data.")
    s.add_argument("--serving-metadata-min-domain-share", type=float,
                   default=0.90,
                   help="accept a per-domain stamp only when it covers at "
                        "least this share of the domain's snapshot points. "
                        "The threshold is config because it is a judgement "
                        "about how much of a guess a ranking weight may be.")
    s.add_argument("--payload-shape-contract", default=None,
                   help="data/payload-shape-contract.v1.json -- the DECLARED "
                        "per-set and per-family limits on rows that reach the "
                        "collection with no title/trust_tier/source_type/lang, "
                        "no BM25 posting, or a title that does not carry the "
                        "handle the family is searched by. Defaults to the "
                        "tracked file.")
    s.add_argument("--max-missing-serving-signal-frac", type=float, default=0.0,
                   help="refuse the stage if more than this fraction of "
                        "non-excluded staged rows still carry no "
                        "title/trust_tier/source_type/lang.")
    # ADR-103 (task 505 §2.3 / task 511 §0): the census-recorded
    # consent_state / non_refreshable stamp for the 7,935 carried
    # declined-or-unreadable-origin rows. On by default -- a shipping build
    # must carry it; --skip is for isolated dry runs / tests only.
    s.add_argument("--census", default=str(REPO_ROOT / "data" /
                                           "r0-1024-coverage-census.v1.json"))
    s.add_argument("--skip-adr103-flags", action="store_true")
    s.add_argument("--require-persisted", default="l1", choices=["l1", "l1b"],
                   help="ADR-105 bar for the staged pages; see `load`.")
    s.set_defaults(fn=cmd_stage)

    for _sub in (p, s):
        _sub.add_argument("--family-contracts", default=None,
                          help="data/source-family-contracts.v1.json -- what "
                               "each source family DECLARES about itself "
                               "(representation surfaces, chunk policy, "
                               "consent basis + terminal schema version, "
                               "licence instrument, title shape, point-id "
                               "rule, ADR-039 payload minimum, gate classes, "
                               "guards). An undeclared set is refused. "
                               "Defaults to the tracked file.")
        _sub.add_argument("--strict-families", action="store_true",
                          help="also refuse a family whose declaration still "
                               "carries an `unknown` value. Every family "
                               "backfilled from the pre-contract era does; "
                               "this is the bar a new family must meet.")

    l = sub.add_parser("load")
    l.add_argument("--work", required=True)
    l.add_argument("--require-persisted", default="l1",
                   choices=["l1", "l1b"],
                   help="ADR-105 bar for the staged pages: 'l1' (a persisted "
                        "L1 markdown record in a store registered in "
                        "data/persisted-input-stores.v1.json) or 'l1b' "
                        "(additionally admits pages whose text survives only "
                        "as L2 passages -- the ADR-093 overlap-chaining "
                        "route, recorded separately in the verdict). No level "
                        "admits a page with no stored text at all.")
    l.add_argument("--config", required=True)
    l.add_argument("--collection", default=None)
    l.add_argument("--qdrant", default=os.environ.get("AW_BUILD_QDRANT",
                                                      "http://localhost:6456"))
    l.add_argument("--batch", type=int, default=512)
    l.add_argument("--collapse-ids", default=None)
    l.add_argument("--pack-route-exclusions", default=None,
                   help="data/assembly/pack-route-exclusions.v1.json -- pack "
                        "sources whose own claim fragment declares a non-flip "
                        "lane. Defaults to the tracked file if it exists.")
    l.add_argument("--lag-every", type=int, default=250_000,
                   help="check how far the collection is behind the loader "
                        "every N upserted rows (0 disables)")
    l.add_argument("--max-lag", type=int, default=2_000_000,
                   help="refuse to keep writing while more than N rows are "
                        "written-but-not-applied")
    l.add_argument("--lag-timeout", type=float, default=900.0,
                   help="seconds to wait for the collection to catch up before "
                        "refusing")
    l.add_argument("--row-suppression", default=None,
                   help="set-scoped rows a ratified precedence forfeits "
                        "(jsonl.gz of {scope, set_id, point_id}); the row is "
                        "never upserted")
    l.add_argument("--collision-winners", default=None,
                   help="jsonl(.gz) of {point_id, winner_set_id} -- for every id "
                        "staged by more than one set, WHICH set's row loads. "
                        "Without it the survivor is decided by receipt iteration "
                        "order, which is an accident of filenames "
                        "(derived once from ratified precedence; the deriver is retired).")
    l.add_argument("--force", action="store_true")
    l.add_argument("--finalize", action="store_true")
    l.set_defaults(fn=cmd_load)

    v = sub.add_parser("verify")
    v.add_argument("--work", required=True)
    v.add_argument("--payload-shape-contract", default=None,
                   help="re-evaluate data/payload-shape-contract.v1.json from "
                        "the stage receipts, so a tree staged before the "
                        "contract existed cannot be loaded and shipped "
                        "unchecked.")
    v.add_argument("--config", default=None)
    v.add_argument("--collection", default=None)
    v.add_argument("--qdrant", default=os.environ.get("AW_BUILD_QDRANT",
                                                      "http://localhost:6456"))
    v.add_argument("--tombstone-ids", default=None)
    v.add_argument("--delete-ids", default=None)
    v.add_argument("--absence-sample", type=int, default=5000)
    v.set_defaults(fn=cmd_verify)

    args = ap.parse_args(argv)
    # task 864 (review 869 S1): from an aw_e10 T0 to the flip only the drain writes the twin
    from cytir.index import twin_write_hold as _TWH
    _TWH.refuse_cli_write(args, tool='cyt index assemble')
    rc = args.fn(args)
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
