#!/usr/bin/env python3
"""`cyt index layout-check --url URL --collection NAME [--profile prod-e9]` (task 2405, S6).

Reads one collection's vector config (a single read-only GET) and asserts it carries the SERVED
LAYOUT: the dense vector's width, storage datatype and residency, and the dense quantizer, all as the
named serving profile declares them (`cytir.core.serving_profile`; the default is the profile `cyt index
build` builds, 256-d float16 + PQ x8, ruling 63(c)). The numbers live in the profile, not here, so the
check cannot drift from the declaration.

Prints ONE line, `layout: MATCH <facts>` or `layout: MISMATCH <field: read X, profile Y; ...>`, and
exits 0 on MATCH, 1 on MISMATCH, 2 on usage or an unreadable collection. It scopes itself to the
layout: payload indexes, HNSW and optimizer settings are `cyt index build`'s own read-back
(`serving_profile.verify_created`).
"""
from __future__ import annotations

import argparse
import sys

from cytir.core import internal_http as IH
from cytir.core import serving_profile as SP

EXIT_MATCH, EXIT_MISMATCH, EXIT_USAGE = 0, 1, 2

#: The layout facts, as paths into `serving_profile.shape_of` / `expected_shape`.
LAYOUT_PATHS = (
    ("vectors", SP.DENSE, "size"),
    ("vectors", SP.DENSE, "datatype"),
    ("vectors", SP.DENSE, "on_disk"),
    ("quantization",),
)


def _pick(shape: dict, path: tuple):
    cur = shape
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def layout_of(shape: dict) -> dict:
    return {".".join(p): _pick(shape, p) for p in LAYOUT_PATHS}


def compare(read_shape: dict, profile: "SP.ServingProfile") -> tuple[bool, str]:
    read, want = layout_of(read_shape), layout_of(profile.expected_shape())
    bad = [f"{k}: read {read[k]!r}, profile {want[k]!r}" for k in want if read[k] != want[k]]
    if bad:
        return False, f"{profile.name}: " + "; ".join(bad)
    q = want["quantization"]
    quant = "none" if not q else f"{q['kind']}/{q.get('compression') or q.get('type') or q.get('encoding')}"
    return True, (f"{profile.name}: dense {want['vectors.dense.size']}-d "
                  f"{want['vectors.dense.datatype']} on_disk={want['vectors.dense.on_disk']}, quantizer {quant}")


def main(argv=None, prog: str = "cyt index layout-check") -> int:
    ap = argparse.ArgumentParser(prog=prog, description="assert a collection has the served vector layout "
                                 "(read-only: one GET); prints `layout: MATCH|MISMATCH ...`")
    ap.add_argument("--url", required=True, help="Qdrant base URL")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--profile", default=SP.PROD_E9.name,
                    help="serving profile that declares the layout (default: %(default)s)")
    ap.add_argument("--timeout", type=int, default=30)
    a = ap.parse_args(argv)
    try:
        profile = SP.get(a.profile)
    except (KeyError, ValueError) as exc:
        print(f"layout: unknown profile {a.profile!r} ({exc})", file=sys.stderr)
        return EXIT_USAGE
    try:
        result = IH.fetch_json(f"{a.url.rstrip('/')}/collections/{a.collection}", timeout=a.timeout)["result"]
        shape = SP.shape_of(result)
    except Exception as exc:  # unreadable is not a verdict: say so, exit 2
        print(f"layout: UNREADABLE {a.collection} at {a.url}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    ok, detail = compare(shape, profile)
    print(f"layout: {'MATCH' if ok else 'MISMATCH'} {a.collection} {detail}")
    return EXIT_MATCH if ok else EXIT_MISMATCH


if __name__ == "__main__":
    sys.exit(main())
