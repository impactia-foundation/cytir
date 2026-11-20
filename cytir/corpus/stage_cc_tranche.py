#!/usr/bin/env python3
"""Stage a consent-gated CC tranche (`cc_authority_frontier_w7`) for the
ratified GPU embed chain, and write the matching `pack_chunks` shards
`promote_sealed_pack.py`/`cytir.index.compose` read.

Task 664 promoted this from a per-tranche throwaway script into a real module; task 726 moved
it into the package (the old script path is an alias shim). It recurs for every new CC tranche
(the seal watchdog calls it once per tranche, unattended), so it lives in the package.

Per pack P (rows partitioned by `crc32(domain) % --packs`, so no url --
hence no page -- ever spans two packs; `promote_sealed_pack.py` refuses a
pack whose urls the target already serves, so a url split across packs
would refuse pack 2):

  <out>/<P>/stage/shard-NNNN.texts.jsonl.gz   {"point_id", "text"} where text
      is `aw_common.compose_index_text(row)` -- the family's declared
      dense_surface (tranche 1's stage rows have exactly this shape).
  <out>/<P>/chunks/shard-NNNN.texts.jsonl.gz  the full source row minus
      `outlinks` (not ADR-039 payload, never read downstream), same ids,
      same order.

Never drops a row silently: empty-text rows and duplicate point_ids (first
wins) are counted in <out>/stage_report.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import zlib
from typing import Iterable

from cytir.core import common as AC


def _pack_names(packs: int) -> list[str]:
    return [chr(ord("a") + i) for i in range(packs)] if packs > 1 else ["all"]


def stage_tranche(inp: str, out: str, *, packs: int = 1,
                  shard_size: int = 20_000) -> dict:
    """Read the gated `passages_ok.jsonl` at `inp`, write the staged
    stage/chunks shards under `out`, and return the same report dict this
    also writes to `<out>/stage_report.json`."""
    names = _pack_names(packs)
    state: dict[str, dict] = {}
    for n in names:
        for sub in ("stage", "chunks"):
            os.makedirs(os.path.join(out, n, sub), exist_ok=True)
        state[n] = {"shard": 0, "rows_in_shard": 0, "stage_fh": None,
                    "chunk_fh": None, "rows": 0, "urls": set()}

    def rotate(n: str) -> None:
        st = state[n]
        for k in ("stage_fh", "chunk_fh"):
            if st[k]:
                st[k].close()
        tag = f"shard-{st['shard']:04d}.texts.jsonl.gz"
        st["stage_fh"] = gzip.open(os.path.join(out, n, "stage", tag), "wt",
                                   encoding="utf-8")
        st["chunk_fh"] = gzip.open(os.path.join(out, n, "chunks", tag), "wt",
                                   encoding="utf-8")
        st["shard"] += 1
        st["rows_in_shard"] = 0

    seen: set[str] = set()
    rep = {"input_rows": 0, "empty_text": 0, "duplicate_point_id": 0,
          "missing_point_id": 0, "staged": 0}
    with open(inp, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rep["input_rows"] += 1
            row = json.loads(line)
            pid = row.get("point_id")
            if not pid:
                rep["missing_point_id"] += 1
                continue
            if pid in seen:
                rep["duplicate_point_id"] += 1
                continue
            seen.add(pid)
            if not (row.get("text") or "").strip():
                rep["empty_text"] += 1
                continue
            row.pop("outlinks", None)
            dense = AC.compose_index_text(row)
            n = (names[zlib.crc32((row.get("domain") or "").encode()) % len(names)]
                if len(names) > 1 else names[0])
            st = state[n]
            if st["stage_fh"] is None or st["rows_in_shard"] >= shard_size:
                rotate(n)
            st["stage_fh"].write(json.dumps({"point_id": pid, "text": dense},
                                            ensure_ascii=False) + "\n")
            st["chunk_fh"].write(json.dumps(row, ensure_ascii=False) + "\n")
            st["rows_in_shard"] += 1
            st["rows"] += 1
            st["urls"].add(row.get("url"))
            rep["staged"] += 1
    for n in names:
        for k in ("stage_fh", "chunk_fh"):
            if state[n][k]:
                state[n][k].close()
    rep["packs"] = {n: {"rows": state[n]["rows"], "shards": state[n]["shard"],
                        "distinct_urls": len(state[n]["urls"])} for n in names}
    with open(os.path.join(out, "stage_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    return rep


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--packs", type=int, default=1)
    ap.add_argument("--shard-size", type=int, default=20_000)
    args = ap.parse_args(argv)
    rep = stage_tranche(args.inp, args.out, packs=args.packs,
                        shard_size=args.shard_size)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
