#!/usr/bin/env python3
"""apply_delta.py — apply a dump_netnew.py CHANGED-SET dump (+ an optional
delete-id list) into an OFFLINE aw_vN clone. This is the prod side of the
delta-promote (ADR-066): the ~100GB base is cloned locally on prod (aw_v4_sat
snapshot -> aw_vN on the scratch-backed build instance), then only the small
delta dump (shipped over the WAN) is applied here — NO embedding on prod, just
upsert of the pre-computed dense+bm25 vectors.

FULL DELTA:
  * each dump line is upserted -> ADD (new id) or MODIFY (overwrites the base
    copy of an existing id); qdrant upsert is id-keyed, so a modify replaces the
    vector+payload in place.
  * `_delta:"unchanged"` marker lines (format-2 dumps, task 348) are NOT
    upserted — they complete the page keep-sets for supersede.
  * SUPERSEDE-ON-UPSERT (task 348, default ON): after the upserts, every
    canonical page group the delta touched is reconciled — existing points of
    the same page whose id is not in the delta's keep-set are deleted (the old
    versions of changed Tier-B chunks, ALL chunks a shrunk page no longer has).
    Requires a page-complete input: a format-2 dump (manifest
    `supersede_complete: true`); refuses format-1 changed-set dumps BEFORE any
    write. --no-supersede restores the legacy duplicate-leaving behaviour.
  * --delete-ids <file> removes those point_ids (removal wave).
bm25 is re-saturated idempotently (already saturated at dump time -> no-op).

Runs against the scratch-backed BUILD instance (never the live serving qdrant);
the live aw_prod alias and aw_v4_sat are untouched — which is also why the
`url` keyword index supersede needs is auto-created here when missing.

  python -m cytir.index.apply_delta \
    --qdrant-url http://localhost:6390 --collection aw_v5 \
    --in /opt/aw/promote/aw_dtier1_cov.netnew.jsonl.gz \
    --delete-ids /opt/aw/promote/deletes.txt      # optional
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys

from qdrant_client import models as qm

from cytir.core.qdrant import qdrant_connect

from cytir.index import live_target as LT  # noqa: E402
from cytir.index import prod_guard as PG  # noqa: E402
from cytir.index.supersede import (CardOverwriteRefused, Superseder,  # noqa: E402
                              client_url, dump_is_changed_set,
                              guard_card_overwrites, load_card_waivers,
                              read_dump_manifest)
import cytir.corpus.persisted as PI  # noqa: E402

K1, B, BASE_AVGDL, SAT_CEIL = 1.2, 0.75, 199.71502927412075, 2.25


def _saturate(indices, values):
    fidx, fvals = [], []
    for i, v in zip(indices, values):
        if v > 0:
            fidx.append(i); fvals.append(v)
    if not fvals or max(fvals) <= SAT_CEIL:
        return fidx, fvals
    dl = float(sum(fvals))
    denom = K1 * (1.0 - B + B * dl / BASE_AVGDL)
    return fidx, [v * (K1 + 1.0) / (v + denom) for v in fvals]


def to_point(rec: dict) -> qm.PointStruct:
    vec = {}
    for name, v in rec["vector"].items():
        if isinstance(v, dict):                       # sparse
            idx, val = v["indices"], v["values"]
            if name == "bm25":
                idx, val = _saturate(idx, val)        # idempotent
            vec[name] = qm.SparseVector(indices=idx, values=val)
        else:
            vec[name] = v
    return qm.PointStruct(id=rec["id"], vector=vec, payload=rec["payload"])


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog=prog or "apply_delta.py",
        description="WRITES: upserts and deletes in the named --qdrant-url/--collection (an OFFLINE aw_vN clone); "
                    "refuses the serving alias, prod and live targets per its write guards.")
    ap.add_argument("--qdrant-url", default="http://localhost:6390")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--delete-ids", default="")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--expect-base", type=int, default=None,
                    help="base count before apply; asserts final == "
                         "base + add - del - superseded")
    ap.add_argument("--no-supersede", action="store_true",
                    help="legacy behaviour: leave the old points of re-crawled "
                         "pages in place (score-tied duplicates, task 348) — "
                         "emergencies only")
    ap.add_argument("--require-persisted", default="l1",
                    choices=sorted(PI.REQUIRE_LEVELS),
                    help="ADR-105 bar for the dump's pages: 'l1' (a persisted "
                         "L1 markdown record) or 'l1b' (additionally admits "
                         "pages whose text survives only as L2 passages, the "
                         "ADR-093 route). No level admits a page with no "
                         "stored text at all.")
    ap.add_argument("--card-waivers", default=None,
                    help="task 820: sealed card-waiver file (task 728 format) naming a stored "
                         "entity card this delta may overwrite by id")
    ap.add_argument("--card-waivers-sha256", default=None)
    args = ap.parse_args(argv)
    # task 864 (review 869 S1): from an aw_e10 T0 to the flip only the drain writes the twin
    from cytir.index import twin_write_hold as _TWH
    _TWH.refuse_cli_write(args, tool='cytir.index.apply_delta')

    # WRITE GUARD, before the first client call: this applier's contract is the OFFLINE clone
    # (ADR-066/083). A live endpoint, the serving alias or a declared live collection is refused.
    try:
        url = LT.normalize_endpoint(args.qdrant_url)   # the ONE string every check and the client use
    except LT.LiveTargetRefused as exc:
        sys.exit(f"FATAL: {exc}")
    if PG.is_prod(url, args.collection):
        sys.exit(f"FATAL: {args.collection!r} on {url} is the serving prod target "
                 "(prod_guard); apply_delta writes an offline clone only")
    try:
        LT.refuse_live_target(url, args.collection)
    except LT.LiveTargetRefused as exc:
        sys.exit(f"FATAL: {exc}")
    try:
        c = qdrant_connect(url, timeout=600, check_compatibility=False)
    except TypeError:            # older qdrant_client without check_compatibility
        c = qdrant_connect(url, timeout=600)
    if args.collection not in {x.name for x in c.get_collections().collections}:
        sys.exit(f"FATAL: {args.collection} missing on {url}")

    # --- supersede preconditions, BEFORE any write (cheap front checks) ---
    sup = None
    if args.no_supersede:
        print("[apply] *** --no-supersede: stale points of re-crawled pages "
              "will SURVIVE (task-348 duplicate class) ***", flush=True)
    else:
        with gzip.open(args.inp, "rt", encoding="utf-8") as probe:
            first_line = next(iter(probe), "")
        first = json.loads(first_line) if first_line.strip() else {}
        if dump_is_changed_set(first) and \
                not read_dump_manifest(args.inp).get("supersede_complete"):
            sys.exit(
                "FATAL: changed-set dump without a supersede_complete manifest "
                "(format-1 dump_netnew output). Its keep-sets are not "
                "page-complete — superseding would delete unchanged siblings. "
                "Re-dump with the format-2 dump_netnew, or pass --no-supersede "
                "to apply WITH the known duplicate bug.")
        sup = Superseder(c, args.collection)
        # this applier's contract is the OFFLINE clone — auto-create is safe
        sup.ensure_url_index(create=True)

    # --- persisted-inputs guard, BEFORE any write (ADR-105, task 581) ---
    # A delta-promote ships whatever the changed-set dump holds. If a row's
    # page has no persisted L1 record, the clone it lands in is not
    # rebuildable from the store either -- the debt rides along with the
    # promotion, invisibly, which is how 492,211 pages of it accumulated.
    def _dump_pages():
        with gzip.open(args.inp, "rt", encoding="utf-8") as fh:
            for line in fh:
                pl = (json.loads(line).get("payload") or {})
                yield {"url": pl.get("url") or pl.get("canonical_url"),
                       "sha": pl.get("content_digest")}
    try:
        pv = PI.assert_pages_persisted(
            _dump_pages(), require=args.require_persisted,
            label=f"{args.inp} -> {args.collection}")
    except PI.PersistedInputsError as exc:
        sys.exit(f"FATAL: {exc}")
    print(f"[apply] persisted-inputs OK: {pv['pages']:,} page(s), "
          f"{pv['persisted']:,} L1/L0, {pv['l1b_reconstructible']:,} L2-only",
          flush=True)

    # --- card-overwrite guard, BEFORE any write (task 820) ---
    # The drain's own check: a stored entity card overwritten by id (an `add` or a `modify` row)
    # with a row that is not a card replacing it refuses the whole delta. One indexed id retrieve
    # per 500 ids on the offline clone, never a scroll; `unchanged` marker lines are never written.
    def _dump_rows():
        with gzip.open(args.inp, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("_delta", "add") != "unchanged":
                    yield rec

    try:
        guard_card_overwrites(
            _dump_rows(), client_url(c), args.collection, writer="apply_delta",
            waivers=load_card_waivers(args.card_waivers, args.card_waivers_sha256),
            log=lambda m: print(f"[apply] {m}", flush=True))
    except CardOverwriteRefused as exc:
        sys.exit(f"FATAL: {exc}")

    before = c.count(args.collection, exact=True).count
    print(f"apply_delta -> {args.collection} (before={before:,})", flush=True)

    add = mod = unchanged = 0
    batch = []

    def flush():
        if batch:
            c.upsert(args.collection, points=batch, wait=True)
            batch.clear()

    op = gzip.open(args.inp, "rt", encoding="utf-8")
    for line in op:
        rec = json.loads(line)
        d = rec.get("_delta", "add")
        if sup is not None:
            pl = rec.get("payload") or {}
            sup.observe_payload(rec["id"], pl)
        if d == "unchanged":     # format-2 marker: keep-set only, no upsert
            unchanged += 1
            continue
        add += d == "add"; mod += d == "modify"
        batch.append(to_point(rec))
        if len(batch) >= args.batch:
            flush()
            print(f"  upserted add {add:,} modify {mod:,}", flush=True)
    flush()
    op.close()
    print(f"[apply] upserted {add + mod:,} ({add:,} add + {mod:,} modify; "
          f"{unchanged:,} unchanged markers)", flush=True)

    deleted = 0
    if args.delete_ids:
        ids = [ln.strip() for ln in open(args.delete_ids) if ln.strip()]
        norm = [int(i) if i.isdigit() else i for i in ids]
        if norm:
            present = len(c.retrieve(args.collection, ids=norm,
                                     with_payload=False, with_vectors=False))
            c.delete(args.collection,
                     points_selector=qm.PointIdsList(points=norm), wait=True)
            deleted = len(norm)
            print(f"[apply] deleted {deleted:,} ids ({present:,} were present)",
                  flush=True)

    superseded = 0
    if sup is not None:
        superseded = sup.run()["superseded"]

    after = c.count(args.collection, exact=True).count
    print(f"[apply] {args.collection}: {before:,} -> {after:,} "
          f"(+{add:,} add, ~{mod:,} modify, -{deleted:,} delete, "
          f"-{superseded:,} superseded)", flush=True)
    if args.expect_base is not None:
        expect = args.expect_base + add - deleted - superseded
        ok = after == expect
        print(f"INVARIANT: {'OK' if ok else 'MISMATCH'} "
              f"(after {after:,} vs base+add-del-superseded {expect:,})",
              flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
