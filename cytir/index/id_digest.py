"""cytir.index.id_digest -- the order-independent MEMBERSHIP digest over point ids.

The one definition of the id digest (task 637 / S6; the flip's command-line digest tool imports
`id_terms` from here). Three independent sha256 accumulators,
summed mod 2**256 over the ids alone, so a scroll that reorders (Qdrant's does, across segment
merges) cannot change the answer. `digest_of(ids)` gives the same `accumulators` and `digest` the
tool writes for a collection holding exactly those ids, which is how the `aw_e10` catch-up proves
"candidate ids == map(twin ids)" (spec `archive/aw-e10-rebuild.md` section 6 steps 9 and 13).

`digest_collection` and `main` are the command-line tool over a live collection (moved here from
the pipeline tree in task 1234, behaviour and output schema unchanged).

WHY THIS EXISTS (task 637 / S6)
-------------------------------
`cytir.index.seal_collection_membership verify` proves a collection's `{id, payload}` multiset
against a sealed export. That is the right instrument for a rebuild that must
preserve payloads, and the WRONG one for this flip: S6 deliberately changes the
payload (normalization) and the sparse vectors (the prune), so a payload-inclusive
digest is guaranteed to differ and proves nothing about membership.

The flip's actual question is narrower and has to be answerable in minutes:
**does `aw_e9` hold exactly the point ids `aw_e8` holds at flip time?** A
vocabulary prune changes no point's membership, so any difference is a pack that
landed on one mirror and not the other.

The digest is an order-independent multiset accumulator over the ids alone —
XOR-free, three independent moduli, the same shape
the payload-membership verifier uses, so a scroll that reorders (and Qdrant's
does, across segment merges) cannot change the answer. Ids-only means
`with_payload=false, with_vectors=false`, which is the cheapest scroll Qdrant
offers.

    python3 -m cytir.index.id_digest \\
        --qdrant http://127.0.0.1:6461 --collection aw_e9 --out e9.ids.json

    python3 -m cytir.index.id_digest \\
        --qdrant <twin-url> --collection <twin-collection> --out e8.ids.json

    python3 -m cytir.index.id_digest --compare e8.ids.json e9.ids.json

`--compare` exits 0 when the two digests agree, 2 when they do not, and it names
the counts on both sides so "not equal" arrives with its size.

`--dump-ids PATH` also writes every id, one per line, so `--diff A B` can name
exactly which points one mirror holds and the other does not. That turns a flip's
catch-up from ledger archaeology ("which packs landed after the snapshot?") into a
set difference over the ids themselves - the same distinction
the gold-presence gate makes between a measured absence and an inferred one.

It also re-reads `points_count` after the scroll and REFUSES if it moved: a digest
taken over a collection that was being written is a digest of nothing (the same
reason the payload-membership verifier re-checks its count).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from cytir.core import internal_http as IH
from typing import Iterable

from cytir.index import prod_guard as PG  # no prod scroll without a reason (2026-10-07)

MASK = (1 << 256) - 1
#: Three accumulators over the same ids: a collision is then not a bug in one hash's use.
MODULI = 3


def id_terms(point_id) -> list[int]:
    """The three per-id values the accumulators sum, from the id alone."""
    raw = json.dumps(str(point_id), sort_keys=True, separators=(",", ":")).encode()
    return [int.from_bytes(hashlib.sha256(bytes([salt]) + raw).digest(), "big")
            for salt in range(MODULI)]


def digest_of(ids: Iterable) -> dict:
    """{ids, accumulators, digest} over `ids` (a multiset: pass each id once for a set)."""
    acc = [0] * MODULI
    n = 0
    for i in ids:
        for k, t in enumerate(id_terms(i)):
            acc[k] = (acc[k] + t) & MASK
        n += 1
    hexes = ["%064x" % a for a in acc]
    return {"ids": n, "accumulators": hexes,
            "digest": hashlib.sha256(("|".join(hexes) + "|%d" % n).encode()).hexdigest()}


def _req(url, body=None, method=None, timeout=300):
    return IH.fetch_json(url, body=body, method=method, timeout=timeout)


def digest_collection(qdrant, collection, batch=4000, progress_every=40,
                      dump_ids=None, prod_scroll_reason=None):
    # incident 2026-10-07: an id scroll of prod is refused unless an operator names a reason
    try:
        PG.check("%s" % qdrant, "POST", "/collections/%s/points/scroll" % collection,
                 allow_prod_scroll=bool(prod_scroll_reason), reason=prod_scroll_reason)
    except PG.ProdReadRefused as exc:
        sys.exit(str(exc))
    info = _req("%s/collections/%s" % (qdrant, collection), method="GET")["result"]
    before = info.get("points_count")
    acc = [0] * MODULI
    n = 0
    offset = None
    pages = 0
    t0 = time.time()
    fh = open(dump_ids, "w") if dump_ids else None
    while True:
        body = {"limit": batch, "with_payload": False, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        res = _req("%s/collections/%s/points/scroll" % (qdrant, collection), body)["result"]
        pts = res.get("points") or []
        for p in pts:
            terms = id_terms(p["id"])
            for i in range(MODULI):
                acc[i] = (acc[i] + terms[i]) & MASK
            n += 1
            if fh:
                fh.write("%s\n" % p["id"])
        offset = res.get("next_page_offset")
        pages += 1
        if progress_every and pages % progress_every == 0:
            print("  %d ids, %.0f s" % (n, time.time() - t0), file=sys.stderr, flush=True)
        if offset is None:
            break
    if fh:
        fh.close()
    after = _req("%s/collections/%s" % (qdrant, collection),
                 method="GET")["result"].get("points_count")
    if before != after:
        sys.exit("REFUSED: points_count moved during the scan (%s -> %s). A "
                 "membership digest taken over a collection being written is a "
                 "digest of nothing." % (before, after))
    if n != before:
        print("WARNING: scrolled %d ids but points_count is %s — report both"
              % (n, before), file=sys.stderr)
    return {
        "schema": "aw.collection-id-digest.v1",
        "qdrant": qdrant, "collection": collection,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "points_count_before": before, "points_count_after": after,
        "ids_scrolled": n,
        "accumulators": ["%064x" % a for a in acc],
        "digest": hashlib.sha256(
            ("|".join("%064x" % a for a in acc) + "|%d" % n).encode()).hexdigest(),
        "seconds": round(time.time() - t0, 1),
        "note": ("ids only — payloads and vectors are deliberately excluded, "
                 "because a footprint rebuild changes both by design and a "
                 "vocabulary prune changes no point's membership"),
    }


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    ap.add_argument("--qdrant")
    ap.add_argument("--collection")
    ap.add_argument("--batch", type=int, default=4000)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--allow-prod-scroll", metavar="REASON", default=None,
                    help="a full id scroll of PROD scans the serving box (incident 2026-10-07): "
                         "refused unless this names the operator's reason (it is logged)")
    ap.add_argument("--dump-ids", help="also write every id, one per line")
    ap.add_argument("--diff", nargs=2, metavar=("IDS_A", "IDS_B"),
                    help="two --dump-ids files; prints and writes the set difference")
    ap.add_argument("--diff-out", default="",
                    help="with --diff: write the only-in-A ids here (the catch-up list)")
    a = ap.parse_args(argv)

    if a.diff:
        A = set(l.strip() for l in open(a.diff[0]) if l.strip())
        B = set(l.strip() for l in open(a.diff[1]) if l.strip())
        only_a, only_b = sorted(A - B), sorted(B - A)
        if a.diff_out:
            open(a.diff_out, "w").write("".join(i + "\n" for i in only_a))
        print(json.dumps({"a_ids": len(A), "b_ids": len(B),
                          "only_in_a": len(only_a), "only_in_b": len(only_b),
                          "sample_only_in_a": only_a[:5],
                          "sample_only_in_b": only_b[:5],
                          "catch_up_list": a.diff_out or None}, indent=1))
        return 0 if not (only_a or only_b) else 2

    if a.compare:
        x, y = (json.load(open(p)) for p in a.compare)
        same = x["digest"] == y["digest"]
        print(json.dumps({
            "equal": same,
            "a": {"collection": x["collection"], "ids": x["ids_scrolled"],
                  "digest": x["digest"], "measured_at": x["measured_at"]},
            "b": {"collection": y["collection"], "ids": y["ids_scrolled"],
                  "digest": y["digest"], "measured_at": y["measured_at"]},
            "count_delta": y["ids_scrolled"] - x["ids_scrolled"],
        }, indent=1))
        return 0 if same else 2

    if not (a.qdrant and a.collection and a.out):
        ap.error("--qdrant, --collection and --out are required without --compare")
    rep = digest_collection(a.qdrant, a.collection, a.batch,
                            dump_ids=a.dump_ids,
                            prod_scroll_reason=a.allow_prod_scroll)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in
                      ("collection", "ids_scrolled", "digest", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
