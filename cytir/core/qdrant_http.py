#!/usr/bin/env python3
"""One qdrant HTTP helper, because two copies of it grew the same bug twice.

Both bugs are worth naming, since the helper exists to make them impossible:

  1. A JSON body sent with the default GET reaches a DIFFERENT handler --
     `/points/search` becomes "Can not recognize \"search\" as point id", and
     `/points` becomes a 404 -- so the method is INFERRED from the presence of
     a body unless stated.
  2. `urllib` swallows the response body on an HTTPError, which turns qdrant's
     actual reason into "HTTP Error 400: Bad Request". The body is read and
     raised.

Import this; do not re-implement it (engineering doctrine 1).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class QdrantHTTPError(RuntimeError):
    pass


def api(base: str, path: str, payload=None, method: str = "", timeout: float = 600):
    data = json.dumps(payload).encode() if payload is not None else None
    method = method or ("POST" if data is not None else "GET")
    req = urllib.request.Request(base.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return json.load(fh)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:600]
        raise QdrantHTTPError(f"{method} {path} -> {exc.code}: {body}") from exc


def points_count(base: str, collection: str, exact: bool = True) -> int:
    """Point count. EXACT by default, and that default is the whole point.

    `/collections/<c>`'s `points_count` is an ESTIMATE taken from segment
    metadata. While a collection is `yellow` -- which is what it is for minutes
    after any large in-place rewrite -- that estimate reads low: task 528b's
    legifrance apply saw 30,895,409 / 30,895,414 / 30,895,438 from four workers
    finishing seconds apart, and `/points/count {"exact": true}` said
    30,895,438 for all of them. Reading the estimate is how a conservation
    check reports a violation that did not happen (memory:
    aw-heavy-rewrite-degrades-prod names the same false mismatch).
    """
    if not exact:
        return api(base, f"/collections/{collection}")["result"]["points_count"]
    return api(base, f"/collections/{collection}/points/count",
               {"exact": True}, timeout=900)["result"]["count"]
