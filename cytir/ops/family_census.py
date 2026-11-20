"""Points per `family_id` on ONE collection (task 1391, ADR-124).

    python3 -m cytir.ops.family_census --collection-url http://HOST:PORT/collections/NAME

Read-only. One `GET /collections/<c>` (does the payload schema index `family_id`?), then one
`POST /collections/<c>/points/count` per family of `data/source-family-contracts.v1.json`
(`exact: false`, a keyword filter, 0.2 s apart). Never a scroll. There is no default URL: the
caller names the collection, and a census never points at prod (prod rows carry no `family_id`
until the aw_e11 rebuild, so the answer there is "no family_id on this collection").
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import json
import sys
import time
import urllib.request
from typing import Callable

from cytir.sources import contract as contract_mod

FIELD = "family_id"
SLEEP_S = 0.2


def _http(method: str, url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        url, data=None if body is None else json.dumps(body).encode(), method=method,
        headers={"content-type": "application/json"})
    with IH.urlopen(req, timeout=30) as r:
        return json.load(r)["result"]


def has_field(collection_url: str, http: Callable = _http) -> bool:
    """True when the collection declares a payload index on `family_id`."""
    info = http("GET", collection_url.rstrip("/"))
    return FIELD in (info.get("payload_schema") or {})


def family_ids(contracts_path=None) -> list[str]:
    doc = contract_mod.load(contracts_path)
    return sorted(f["family_id"] for f in doc["families"] if f.get("status") != "retired")


def census(collection_url: str, families: list[str], *, http: Callable = _http,
           sleep: Callable = time.sleep) -> dict | None:
    """`{family_id: approx points}`, or None when the collection has no `family_id` index."""
    base = collection_url.rstrip("/")
    if not has_field(base, http):
        return None
    out: dict[str, int] = {}
    for fid in families:
        out[fid] = http("POST", f"{base}/points/count", {
            "exact": False,
            "filter": {"must": [{"key": FIELD, "match": {"value": fid}}]}})["count"]
        sleep(SLEEP_S)
    return out


def main(argv=None, prog: str | None = None, *, http: Callable = _http, sleep: Callable = time.sleep) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__)
    ap.add_argument("--collection-url", required=True,
                    help="http://host:port/collections/<name> (no default; never prod)")
    ap.add_argument("--contracts", default=None)
    a = ap.parse_args(argv)
    res = census(a.collection_url, family_ids(a.contracts), http=http, sleep=sleep)
    if res is None:
        print(f"no {FIELD} on this collection ({a.collection_url}): rows built before ADR-124 "
              "carry none; the aw_e11 rebuild stamps it")
        return 0
    for fid, n in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"{n:>12,}  {fid}")
    print(f"{sum(res.values()):>12,}  TOTAL (approximate, exact=false)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
