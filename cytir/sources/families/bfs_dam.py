"""`bfs_dam_exact` -- the Swiss Federal Statistical Office asset catalogue (task 1101, S7).

ONE authoritative-locator record per DAM asset (title + the asset's own summary + theme + period +
agency, pointing at `https://www.bfs.admin.ch/asset/<lang>/<damId>`). The builder is
`cytir.extract.bfs_dam.build_passage`, moved unchanged out of the legacy `ingest_bfs_dam.py`, which
now imports it: the family and the CLI cannot drift.

Options (`--opt KEY=VALUE`): `root` (required): the enumerated catalogue, one asset JSON per line
(`assets_raw.jsonl`, the legacy `enumerate` phase's output), or a directory that holds it.
`now_iso` (optional): the instant an embargo is compared with (default: `<today>T00:00:00Z`).
A unit is one catalogue line, so every asset ends `admitted` or `excluded(<the builder's own
reason>)`; a line that is not JSON is `quarantined`.

Consent: the assets were fetched from the DAM API host, so that host is resolved live (declared
basis: robots.txt/TDMRep per origin) before any unit is read; a decline is a terminal.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from collections import Counter

from cytir.extract import bfs_dam as B
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

RAW = "assets_raw.jsonl"
API_HOST = "dam-api.bfs.admin.ch"


def _raw_path(root: str) -> str:
    return os.path.join(root, RAW) if os.path.isdir(root) else root


@register_family
class BfsDamFamily:
    family_id = "bfs_dam_exact"
    route = "documented_source_api"
    direct_crawl = True
    chunk_policy_id = CHUNK_DECLARED
    require_records = True     # a catalogue that yields 0 records is a wrong input

    def discover(self, ctx: RunContext):
        root = ctx.options.get("root")
        if not root:
            raise ValueError(f"{self.family_id} needs --opt root=<assets_raw.jsonl or its directory>")
        path = _raw_path(root)
        if not os.path.isfile(path):
            raise ValueError(f"{self.family_id}: {path!r} is not a file")
        uri = pathlib.Path(path).resolve().as_uri()
        units = []
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh):
                if line.strip():
                    units.append(Unit(key=f"{n:08d}", url=f"{uri}#{n}", meta={"line": line.strip()},
                                      consent_hosts=(API_HOST,)))
        if not units:
            raise RuntimeError(f"{self.family_id}: {path!r} holds no asset; "
                               "an empty catalogue is a wrong input, not an empty source")
        ctx.notes.update({"raw": str(path), "assets": len(units)})
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        body = unit.meta["line"].encode("utf-8")
        return [Captured(unit, body, {"member": unit.key, "sha256": hashlib.sha256(body).hexdigest()})]

    def extract(self, cap: Captured, ctx: RunContext):
        asset = json.loads(cap.body.decode("utf-8"))      # not JSON: quarantined, not an origin fact
        now_iso = ctx.options.get("now_iso") or f"{ctx.today}T00:00:00Z"
        reasons: Counter = Counter()
        rec = B.build_passage(asset, now_iso, reasons)
        if rec is None:
            raise UnitExcluded(next(iter(reasons), "not_taken"))
        return [Doc(url=rec["url"], text=rec["text"], title=rec["title"], lang=rec["lang"],
                    extra={"row": rec})]

    def chunk(self, doc: Doc, ctx: RunContext):
        # chunk.atomic: one asset is one record (the builder already caps the text)
        return [doc.extra["row"]]
