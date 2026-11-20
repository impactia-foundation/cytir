#!/usr/bin/env python3
"""PAYLOAD CONTRACT gate -- refuse a pack whose rows the fusion cannot score.

WHY THIS EXISTS (task 583 §3). Task 580 restored 12,586 pages of missing page
text into the candidate, gated it, measured -17 h@1, concluded that text
coverage is not the flip lever, and reverted. Task 583 read the rows the apply
actually wrote: all three of 580's tranches carry **0% `title`, 0%
`trust_tier`, 0% `lang`, 0% `domain`**, against 100% in the live collection and
100% in task 579's tranche -- the one that read +13. The legs were fine: on the
15 exposed golds the restored rows' dense cosine and BM25 score were unchanged
or BETTER. They lost their seats because `retrieval.search()` scores a row with
no `title` at title-centrality 0 (a 2.5-weight term) and a row with no
`trust_tier` at `_trust_tier`'s 0.6 unknown default instead of the register's
value -- on `service-public.gouv.fr`, trust 1.0 -> 0.6 is -1.2 fused before the
title term is counted.

So an entire lane's conclusion rested on rows that were never scoreable, and
nothing in the pipeline could say so. This gate can, and it fails closed:

  * it reads the rows, not the intention -- a staging collection or a JSONL
    pack, whichever exists at the point of use;
  * a field below its floor is a FAILURE, and so is an unreadable input, an
    empty collection, or a collection that does not exist (an error is not an
    empty -- verification-doctrine 6);
  * the verdict is carried in the EXIT CODE, so a driver that forgets to read
    stdout still stops (verification-doctrine 12);
  * there is no `--force`. A pack that cannot meet the contract is a build
    defect, and waiving it is a legislative act that belongs to whoever owns
    the standard, not a step inside the run that needs excusing.

    python3 gate_payload_contract.py --qdrant http://localhost:6400 \\
        --collection aw_task580_b_rest
    python3 gate_payload_contract.py --pack /path/to/pack.jsonl
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import gzip
import json
import sys
import urllib.error
import urllib.request

from cytir.index import prod_guard as PG  # no prod scroll without a named reason (incident 2026-10-07)


# The floors live in `payload_carry_contract.py`, with the composition they govern: the
# gate that refuses a pack at the apply and the contract that composes the row
# at materialisation must apply ONE standard, or the earlier of the two becomes
# decoration (task 584).
from cytir.index.payload_carry import CONTRACT  # noqa: E402,F401


def _api(base, path, body):
    req = urllib.request.Request(
        base.rstrip("/") + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with IH.urlopen(req, timeout=180) as r:
        return json.load(r)


def from_collection(base, coll, limit=0, prod_scroll_reason=None):
    # a full payload scroll scans the serving box: refused on prod unless an operator names a reason
    # (raises PG.ProdReadRefused BEFORE any HTTP call)
    PG.check(base, "POST", f"/collections/{coll}/points/scroll",
             allow_prod_scroll=bool(prod_scroll_reason), reason=prod_scroll_reason)
    n, have = 0, {f: 0 for f in CONTRACT}
    offset = None
    while True:
        body = {"limit": 1024, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        res = _api(base, f"/collections/{coll}/points/scroll", body)["result"]
        for p in res["points"]:
            pl = p.get("payload") or {}
            n += 1
            for f in CONTRACT:
                if pl.get(f) is not None:
                    have[f] += 1
        offset = res.get("next_page_offset")
        if offset is None or (limit and n >= limit):
            break
    return n, have


def from_pack(path):
    n, have = 0, {f: 0 for f in CONTRACT}
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pl = d.get("payload") if isinstance(d.get("payload"), dict) else d
            n += 1
            for fld in CONTRACT:
                if pl.get(fld) is not None:
                    have[fld] += 1
    return n, have


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant")
    ap.add_argument("--collection")
    ap.add_argument("--pack")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N rows (0 = the whole input). A sample "
                         "is reported AS a sample; it never becomes 'all'.")
    ap.add_argument("--out")
    ap.add_argument("--allow-prod-scroll", metavar="REASON", default=None,
                    help="a --qdrant scroll of the prod box is refused unless this names the "
                         "operator's reason (it is logged); the TWIN needs no flag")
    a = ap.parse_args(argv)

    try:
        if a.pack:
            src = a.pack
            n, have = from_pack(a.pack)
        elif a.qdrant and a.collection:
            src = f"{a.qdrant}/{a.collection}"
            n, have = from_collection(a.qdrant, a.collection, a.limit,
                                    prod_scroll_reason=a.allow_prod_scroll)
        else:
            print("[payload-contract] FAIL: give --pack or "
                  "--qdrant/--collection", file=sys.stderr)
            return 2
    except PG.ProdReadRefused as e:
        print(f"[payload-contract] FAIL: {e}", file=sys.stderr)
        return 2
    except (OSError, urllib.error.URLError, ValueError, KeyError) as e:
        # An unreadable input is a FAILURE, never a skip: reporting green over
        # ground you could not read is the defect this gate exists to stop.
        print(f"[payload-contract] FAIL: cannot read the input "
              f"({type(e).__name__}: {e})", file=sys.stderr)
        return 2

    if n == 0:
        print("[payload-contract] FAIL: the input holds 0 rows -- an empty "
              "read is an error, not a pass", file=sys.stderr)
        return 2

    rows, failures = [], []
    for f, floor in sorted(CONTRACT.items()):
        rate = have[f] / n
        ok = rate >= floor
        rows.append({"field": f, "present": round(rate, 4), "floor": floor,
                     "ok": ok})
        if not ok:
            failures.append(f"{f} {rate:.1%} < {floor:.0%}")

    verdict = "PASS" if not failures else "FAIL"
    out = {"schema": "aw.payload-contract-gate.v1", "source": src,
           "rows_checked": n, "sampled": bool(a.limit),
           "fields": rows, "failures": failures, "verdict": verdict}
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    if failures:
        print(f"[payload-contract] FAIL on {len(failures)} field(s): "
              f"{'; '.join(failures)}. These rows are not scoreable: a row "
              f"with no `title` scores title-centrality 0 (weight 2.5) and a "
              f"row with no `trust_tier` is scored at the 0.6 unknown "
              f"default, not at the register's value.", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
