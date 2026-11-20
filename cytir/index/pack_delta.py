"""cyt index pack-delta (was apply_pack_delta.py) — merge an aw_b1-NATIVE staged pack (built by
stage_pack_b1.py) into a finished aw_b1 collection as a POST-BUILD DELTA.

Why this exists (task 322): `build_aw_b1.run_staged` consumes a FIXED list
(`STAGING_PACKS`) — a pack staged after ignition cannot be injected mid-build.
Packs whose records are already in final aw_b1 shape (identity minted by
aw_b1_identity, sparse folded+saturated, payload conventions + rule_epochs +
consent-basis stamps) merge with a plain scroll+upsert; no re-projection.

DO NOT run while the build is in the base/deferred/staged phases — apply only
after the builder reports phase=done (or at minimum finalize), then re-run the
gate (ADR-043) before any promote decision.

SUPERSEDE-ON-UPSERT (task 348, default ON): after the merge, every canonical
page group the pack touched is reconciled — existing target points of the same
page whose id is not in the pack are deleted (old versions of changed Tier-B
chunks; ALL stale chunks of shrunk pages). COMPLETENESS CONTRACT: a staged pack
carries the complete current record set of every page it stages (stage_pack_b1
stages whole pages) — never feed this script a hand-filtered subset of a page's
chunks with supersede on. The target is an offline build collection, so the
`url` keyword index is auto-created when missing. --no-supersede restores the
legacy duplicate-leaving behaviour.

ROLLBACK-COMPLETENESS GUARD (task 574): `--export-superseded PATH` writes the
FULL rollback copy (dense+sparse vectors + payload) of exactly the set
`Superseder` is about to delete — computed live, immediately before the
write, never from a separately-staged manifest — via
`upsert_supersede.Superseder.export_stale`. `--verify-superseded-against
PATH` additionally checks that live delete set against a pre-existing
export (e.g. one a materialize/roster step computed earlier) and REFUSES,
before any delete, if the two id sets disagree — the guard task 572 needed:
its roster-keyed export undercounted the live delete set by 315 points,
discovered only after the delete had already run. Both flags are optional
and OFF by default; omitting them reproduces this script's pre-574 output
byte-for-byte.

THE OTHER HALF OF THE ROLLBACK (task 576): `--export-overwritten PATH` writes
the PRE-IMAGE of every incoming id that ALREADY EXISTS in the target -- the
`~N overwrote-existing` rows. `--export-superseded` cannot carry them: the
superseder PROTECTS incoming ids, so an overwritten id is never in the stale
set. Reverting a delta without them either deletes a row that existed before it
ran or leaves it carrying the delta's payload stamps. Task 576 measured 6 such
rows on a 5,478-row tranche and 1,440 on a 533,724-row one. The retrieve that
classifies added-vs-modified already happens; the only cost of closing the gap
is asking that retrieve for the payload and vectors. Off by default.

  cyt index pack-delta --pack aw_dfrlaw_cov \
      --pack-url http://localhost:6398 \
      --target aw_b1 --target-url http://localhost:6390 [--dry-run] \
      [--export-superseded rollback.jsonl.gz] \
      [--verify-superseded-against roster_export.jsonl.gz] \
      [--export-overwritten preimages.jsonl.gz]

Prints added/modified and verifies the target count delta == added.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time

from qdrant_client import models

from cytir.core.qdrant import qdrant_connect  # noqa: E402
from cytir.index.supersede import (  # noqa: E402
    CardOverwriteRefused, Superseder, client_url, encode_point_vectors,
    guard_card_overwrites, guard_scroll, load_card_waivers, read_export_ids)
from cytir.core import serving_config as SC  # noqa: E402
import cytir.corpus.persisted as PI  # noqa: E402


def _pack_pages(src, pack: str, batch: int):
    """Every (url, content_digest) in the pack -- payload-only, before any write.

    ADR-105: a row whose page has no persisted L1 record is not loadable. The
    pack's own rows are the population, read from the pack collection rather
    than from a roster the caller hands in, because a roster is a list somebody
    typed and the pack is the thing about to be written (corpus-doctrine 8).
    """
    offset = None
    while True:
        pts, offset = src.scroll(
            pack, limit=batch, offset=offset, with_vectors=False,
            with_payload=["url", "canonical_url", "content_digest"])
        if not pts:
            return
        for p in pts:
            pl = p.payload or {}
            # a pack is staged raw (stage_pack_b1, 64-hex sha256) and its sha is checked against the L1
            # store capture sha (assert_pages_persisted), never a twin digest: no decode, no digest_key
            yield {"url": pl.get("url") or pl.get("canonical_url"),
                   "sha": pl.get("content_digest")}  # canonical row
        if offset is None:
            return


def _pack_rows(src, pack: str, batch: int):
    """Every `{"id", "payload"}` of the pack, payload only (no vectors) -- what the card guard
    (task 820) reads before the first write."""
    offset = None
    while True:
        pts, offset = src.scroll(pack, limit=batch, offset=offset,
                                 with_vectors=False, with_payload=True)
        for p in pts:
            yield {"id": p.id, "payload": p.payload or {}}
        if not pts or offset is None:
            return


def main(argv=None, prog: str = "cyt index pack-delta"):
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--pack-url", default="http://localhost:6398")
    ap.add_argument("--target", default="aw_b1")
    ap.add_argument("--target-url", default="http://localhost:6390")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--require-persisted", default="l1",
                    choices=sorted(PI.REQUIRE_LEVELS),
                    help="ADR-105 bar for this pack's pages: 'l1' (a persisted "
                         "L1 markdown record) or 'l1b' (additionally admits "
                         "pages whose text survives only as L2 passages, the "
                         "ADR-093 overlap-chaining route). There is no level "
                         "that admits a page with no stored text at all.")
    ap.add_argument("--no-supersede", action="store_true",
                    help="legacy behaviour: leave stale points of re-staged "
                         "pages in place (task-348 duplicate class)")
    ap.add_argument("--export-superseded", default=None,
                    help="task 574: write the full rollback copy (dense+"
                         "sparse vectors + payload) of the LIVE delete set "
                         "here, computed immediately before the write")
    ap.add_argument("--verify-superseded-against", default=None,
                    help="task 574: refuse before any delete if this "
                         "pre-existing export's id set does not match the "
                         "live delete set exactly (prints both counts + "
                         "the symmetric difference)")
    ap.add_argument("--export-overwritten", default=None,
                    help="task 576: write the PRE-IMAGE (dense+sparse vectors "
                         "+ payload) of every incoming id that already exists "
                         "in the target. --export-superseded cannot carry "
                         "these: the superseder protects incoming ids, so an "
                         "overwritten id is never in the stale set, and a "
                         "revert without them loses or re-stamps those rows")
    ap.add_argument("--card-waivers", default=None,
                    help="task 820: sealed card-waiver file (task 728 format) naming a stored "
                         "entity card this delta may overwrite by id")
    ap.add_argument("--card-waivers-sha256", default=None)
    args = ap.parse_args(argv)
    if not args.dry_run:
        # task 864 (review 854 S6): from an aw_e10 T0 to the flip only the drain writes the twin
        from cytir.index import twin_write_hold as TWH
        try:
            TWH.refuse_if_held(args.target, tool="apply_pack_delta")
        except TWH.TwinWriteHeld as exc:
            raise SystemExit(f"[delta] {exc}") from None

    src = qdrant_connect(args.pack_url, timeout=600)
    # the pack is scrolled whole: a prod pack URL/collection is refused (no override), as for the other readers
    guard_scroll(src, args.pack, url=args.pack_url)
    dst = qdrant_connect(args.target_url, timeout=600)

    # -- dim guard: before ANY write (task 563) -- the pack's own vector
    # config IS its declared embedding shape (stage_pack_b1.py stamps
    # VectorParams(size=mrl_dim) at staging time), so it is read the same
    # way the target's shape is, and the two are reconciled through
    # serving_config.assert_dim_matches -- never a second comparison.
    pack_dim = SC.shape_from_info(src.get_collection(args.pack))["dense_size"]
    try:
        SC.assert_dim_matches(dst, args.target, pack_dim,
                              source=f"pack {args.pack}")
    except RuntimeError as exc:
        raise SystemExit(f"[delta] REFUSED: {exc}") from None

    pack_n = src.count(args.pack, exact=True).count

    # -- persisted-inputs guard: before ANY write (ADR-105, task 581) --
    # The dim guard above asks whether the rows FIT. This asks whether the
    # pages behind them still exist on disk. Between 2026-07 and 2026-08 live
    # campaigns wrote rows into prod and the twin whose only surviving input
    # was their own emitted passage; task 579 measured 492,211 pages with no
    # L0, no L1 and no L2 text anywhere, for which ADR-026's "every collection
    # is rebuildable from the store" is simply false. Cheap check at the front
    # (one payload-only scroll) beats an acceptance check after the write.
    # An EMPTY pack is skipped explicitly and said so -- never folded into a
    # silent pass, and never reported as a check that ran.
    if pack_n:
        try:
            v = PI.assert_pages_persisted(
                _pack_pages(src, args.pack, args.batch),
                require=args.require_persisted,
                label=f"pack {args.pack} -> {args.target}")
        except PI.PersistedInputsError as exc:
            raise SystemExit(f"[delta] REFUSED: {exc}") from None
        print(f"[delta] persisted-inputs OK: {v['pages']:,} page(s), "
              f"{v['persisted']:,} with L1/L0, {v['l1b_reconstructible']:,} "
              f"L2-only (require={v['require']}, index {v['index_built_at']})",
              flush=True)
    else:
        print("[delta] persisted-inputs: pack is empty, nothing to check",
              flush=True)

    # -- card-overwrite guard: before ANY write (task 820) -- the drain's own check
    # (`check_card_overwrites`): a stored entity card overwritten by id with a row that is not a
    # card replacing it refuses the whole delta. One indexed id retrieve per 500 ids on the
    # TARGET (an offline build collection), never a scroll of it. Runs on --dry-run too.
    try:
        card_guard = guard_card_overwrites(
            _pack_rows(src, args.pack, args.batch), client_url(dst), args.target,
            writer="apply_pack_delta",
            waivers=load_card_waivers(args.card_waivers, args.card_waivers_sha256),
            log=lambda m: print(f"[delta] {m}", flush=True))
    except CardOverwriteRefused as exc:
        raise SystemExit(f"[delta] REFUSED: {exc}") from None
    del card_guard

    before = dst.count(args.target, exact=True).count
    print(f"[delta] pack {args.pack}={pack_n:,} -> target {args.target}={before:,}"
          f"{' (DRY RUN)' if args.dry_run else ''}", flush=True)

    sup = None
    if args.no_supersede:
        print("[delta] *** --no-supersede: stale points of re-staged pages "
              "will SURVIVE (task-348 duplicate class) ***", flush=True)
    else:
        sup = Superseder(dst, args.target, dry_run=args.dry_run)
        # offline build collection — auto-create is this applier's contract
        sup.ensure_url_index(create=not args.dry_run)

    added = modified = 0
    overwritten_pre = 0
    # Opened BEFORE the first upsert and closed after the loop: a pre-image
    # written after its row was overwritten is not a pre-image.
    pre_fh = None
    if args.export_overwritten is not None and not args.dry_run:
        opener = (gzip.open if args.export_overwritten.endswith(".gz")
                  else open)
        pre_fh = opener(args.export_overwritten, "wt", encoding="utf-8")
    offset = None
    t0 = time.time()
    while True:
        pts, offset = src.scroll(args.pack, limit=args.batch, offset=offset,
                                 with_payload=True, with_vectors=True)
        if not pts:
            break
        ids = [p.id for p in pts]
        # task 576: the pre-image of an id the delta OVERWRITES is the other
        # half of a complete rollback, and `--export-superseded` does not
        # carry it -- the superseder PROTECTS incoming ids, so an id that
        # already existed is never in the stale set and never in that export.
        # Reverting such a delta therefore either deletes a row that existed
        # before it ran, or leaves it carrying the delta's payload stamps.
        # This is the same class as task 572's 315-point gap, on the other
        # side of the ledger; the retrieve below already happens, so the only
        # cost of closing it is asking for the payload and vectors.
        want_pre = args.export_overwritten is not None and not args.dry_run
        rows = dst.retrieve(args.target, ids=ids, with_payload=want_pre,
                            with_vectors=want_pre)
        existing = {r.id for r in rows}
        if want_pre:
            for r in rows:
                pre_fh.write(json.dumps({
                    "id": str(r.id),
                    "vector": encode_point_vectors(r.vector or {}),
                    "payload": r.payload or {}}) + "\n")
                overwritten_pre += 1
        added += sum(1 for i in ids if i not in existing)
        modified += len(existing)
        if sup is not None:
            for p in pts:
                pl = p.payload or {}
                sup.observe_payload(p.id, pl)
        if not args.dry_run:
            batch = [models.PointStruct(
                id=p.id,
                vector={k: v for k, v in (p.vector or {}).items()},
                payload=p.payload) for p in pts]
            dst.upsert(args.target, points=batch, wait=False)
        if (added + modified) % (args.batch * 25) == 0:
            rate = (added + modified) / max(time.time() - t0, 1)
            print(f"  {added+modified:,}/{pack_n:,}  {rate:.0f} pts/s", flush=True)
        if offset is None:
            break

    if pre_fh is not None:
        pre_fh.close()
        # Counted back OFF THE FILE, not off the loop variable: a counter
        # incremented next to a write proves the intention, never the
        # artifact, and this check exists precisely for the case where the
        # write did not land (verification-doctrine 12).
        on_disk = len(read_export_ids(args.export_overwritten))
        if on_disk != modified or overwritten_pre != modified:
            raise SystemExit(
                f"FATAL: {args.export_overwritten} holds {on_disk:,} "
                f"pre-image(s) ({overwritten_pre:,} written) for "
                f"{modified:,} overwritten id(s) on {args.target!r}. The "
                "rollback is incomplete; do not proceed.")
        print(f"[delta] pre-images of {on_disk:,} overwritten id(s) "
              f"-> {args.export_overwritten}", flush=True)

    print(f"[delta] scrolled {added+modified:,}: +{added:,} new, "
          f"~{modified:,} overwrote-existing", flush=True)

    # task 574: the rollback-completeness guard, read once, passed to BOTH
    # the dry-run and real-run sup.run() calls below — a pre-existing export
    # (e.g. a materialize/roster step's own manifest) is checked against
    # the LIVE delete set Superseder is about to compute, before either
    # branch touches a single point.
    expect_ids = (read_export_ids(args.verify_superseded_against)
                 if args.verify_superseded_against else None)

    if args.dry_run:
        if sup is not None:
            sup.run(export_path=args.export_superseded,
                   expect_ids=expect_ids)  # dry-run: what WOULD be superseded
        return
    # settle, then supersede + verify the arithmetic (claim = count)
    for _ in range(300):
        if dst.get_collection(args.target).status == \
                models.CollectionStatus.GREEN:
            break
        time.sleep(5)
    superseded = sup.run(export_path=args.export_superseded,
                         expect_ids=expect_ids)["superseded"] \
        if sup is not None else 0
    after = dst.count(args.target, exact=True).count
    print(f"[verify] target {before:,} -> {after:,} (delta {after-before:+,}; "
          f"expected +{added:,} -{superseded:,} superseded)", flush=True)
    if after - before != added - superseded:
        raise SystemExit(f"COUNT MISMATCH: delta {after-before} != "
                         f"added-superseded {added-superseded}")
    print("OK", flush=True)

