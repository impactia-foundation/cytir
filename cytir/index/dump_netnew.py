#!/usr/bin/env python3
"""dump_netnew.py — dump the CHANGED SET of a staging coverage collection vs the
base (point_id = hash(url[+chunk]) is stable, so an id maps to the same document
across crawls) to a portable jsonl.gz carrying dense + bm25 vectors + payload.
This is the delta-promote input (ADR-066): the base (~100GB) is already on prod
as aw_v4_sat, so we ship only the ~1% delta over the flaky build-host->prod WAN
instead of a 101GB snapshot.

FULL DELTA semantics (add + modify; deletes handled separately, see below):
  * id NOT in base                         -> ADD    (emit for upsert)
  * id in base, digest DIFFERS             -> MODIFY (emit for upsert/overwrite)
  * id in base, digest SAME (or both none) -> UNCHANGED: no vectors shipped, but
    a slim `_delta:"unchanged"` MARKER line {id, payload:{url, source_type}} IS
    emitted (format 2, task 348). The markers make the dump PAGE-COMPLETE: for
    every canonical page the delta touches, the full current id-set is known, so
    apply_delta can supersede stale sibling points (old versions of changed
    Tier-B chunks, removed chunks of shrunk pages) without ever mistaking an
    unchanged-and-therefore-omitted sibling for a stale one. The `.count`
    manifest records `supersede_complete: true`; appliers refuse to supersede
    format-1 dumps. (Pre-348 appliers crash loudly on a format-2 dump — ship
    both sides together.)
Change is detected on the payload digest field (`--digest-field`, ADR-039). When
neither side carries a digest (e.g. a purely-additive coverage wave whose base
was payload-slimmed) NO in-base id can be judged changed, so the emit reduces to
additions-only — today's behavior, unchanged.

DELETES are a separate removal wave: `dump_netnew.py` does not invent them; pass a
newline-delimited point-id file to the APPLY step (`build_vnext.py --delete-ids`
or `apply_delta.py --delete-ids`) which removes them from the offline aw_vN clone
— never from the live/base collection.

Same bm25 saturation as build_vnext.py, so the delta upserted on prod is
byte-consistent with the base build. Emits {id, vector:{dense, bm25}, payload,
_delta} per line — the shape the wave upsert tool already consumes.

  python3 -m cytir.index.dump_netnew \
    --staging-url http://localhost:6398 --staging-collection aw_dtier1_cov \
    --base-url http://localhost:6333    --base-collection aw_v4_sat \
    --digest-field digest \
    --out /data/promote/delta/aw_dtier1_cov.netnew.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

from cytir.core.qdrant import qdrant_connect
from cytir.index.supersede import guard_scroll

# --- bm25 saturation (kept in sync with build_vnext.py / bm25_saturate.py) ---
K1, B, BASE_AVGDL, SAT_CEIL = 1.2, 0.75, 199.71502927412075, 2.25


def _saturate(indices, values):
    fidx, fvals = [], []
    for i, v in zip(indices, values):
        if v > 0:
            fidx.append(i); fvals.append(v)
    if not fvals or max(fvals) <= SAT_CEIL:      # empty or already saturated
        return fidx, fvals
    dl = float(sum(fvals))
    denom = K1 * (1.0 - B + B * dl / BASE_AVGDL)
    return fidx, [v * (K1 + 1.0) / (v + denom) for v in fvals]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging-url", default="http://localhost:6398")
    ap.add_argument("--staging-collection", required=True)
    ap.add_argument("--base-url", default="http://localhost:6333")
    ap.add_argument("--base-collection", default="aw_v4_sat")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--no-saturate", action="store_true")
    ap.add_argument("--digest-field", default="digest",
                    help="payload key holding the content digest (ADR-039) used to "
                         "detect MODIFIED points; if absent on both sides, in-base "
                         "ids are treated as unchanged (additions-only)")
    args = ap.parse_args()

    sc = qdrant_connect(args.staging_url, timeout=600, check_compatibility=False)
    bc = qdrant_connect(args.base_url, timeout=600, check_compatibility=False)
    dfield = args.digest_field
    # The news chain passes STAGING (localhost:6398) here, never prod: a prod endpoint or serving
    # collection is refused (prod_guard.check, no override) before the first scroll.
    guard_scroll(sc, args.staging_collection, url=args.staging_url)

    total = sc.count(args.staging_collection, exact=True).count
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"{args.staging_collection}: {total:,} pts; delta vs "
          f"{args.base_collection}@{args.base_url}; digest_field={dfield!r}; "
          f"saturate={not args.no_saturate} -> {args.out}", flush=True)

    seen = added = modified = skipped = 0
    offset = None
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        while True:
            pts, offset = sc.scroll(args.staging_collection, limit=args.batch,
                                    offset=offset, with_payload=True,
                                    with_vectors=True)
            if not pts:
                break
            ids = [p.id for p in pts]
            # retrieve base points WITH payload so we can read the base digest and
            # classify add vs modify vs unchanged
            base = {r.id: (r.payload or {}) for r in
                    bc.retrieve(args.base_collection, ids=ids,
                                with_payload=True, with_vectors=False)}
            for p in pts:
                seen += 1
                if p.id in base:
                    base_dig = base[p.id].get(dfield)
                    new_dig = (p.payload or {}).get(dfield)
                    # UNCHANGED when we can affirmatively see the digests match
                    # (or both are absent -> cannot detect change -> unchanged):
                    # emit the slim marker (no vectors) so the dump stays
                    # page-complete for supersede (format 2, task 348)
                    if base_dig == new_dig:
                        skipped += 1
                        pl = p.payload or {}
                        f.write(json.dumps({
                            "id": p.id, "_delta": "unchanged",
                            "payload": {"url": pl.get("url"),
                                        "source_type": pl.get("source_type")},
                        }) + "\n")
                        continue
                    delta = "modify"
                    modified += 1
                else:
                    delta = "add"
                    added += 1
                vec = {}
                for name, v in (p.vector or {}).items():
                    if hasattr(v, "indices"):            # SparseVector
                        idx, val = list(v.indices), list(v.values)
                        if name == "bm25" and not args.no_saturate:
                            idx, val = _saturate(idx, val)
                        vec[name] = {"indices": idx, "values": val}
                    elif isinstance(v, dict):
                        idx, val = v["indices"], v["values"]
                        if name == "bm25" and not args.no_saturate:
                            idx, val = _saturate(idx, val)
                        vec[name] = {"indices": idx, "values": val}
                    else:
                        vec[name] = v                    # dense list
                f.write(json.dumps({"id": p.id, "vector": vec,
                                    "payload": p.payload, "_delta": delta}) + "\n")
            print(f"  seen {seen:,}/{total:,} add {added:,} modify {modified:,} "
                  f"unchanged {skipped:,}", flush=True)
            if offset is None:
                break
    emitted = added + modified
    sz = os.path.getsize(args.out)
    print(f"DONE: {emitted:,} changed ({added:,} add + {modified:,} modify), "
          f"{skipped:,} unchanged-skipped -> {args.out} ({sz/1e6:.1f} MB gzip)",
          flush=True)
    # manifest for the prod side to assert the exact expected delta
    with open(args.out + ".count", "w") as m:
        m.write(json.dumps({"changed": emitted, "add": added,
                            "modify": modified, "unchanged": skipped,
                            # format 2 (task 348): unchanged ids are present as
                            # marker lines -> keep-sets are page-complete and
                            # the applier may supersede stale siblings
                            "supersede_complete": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
