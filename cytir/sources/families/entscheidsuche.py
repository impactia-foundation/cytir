"""`scratch_entscheidsuche_exact` -- Swiss court decisions from entscheidsuche.ch (task 1101, S7).

ONE locator record per decision: the Kopfzeile (court, chamber, date) + the official Regeste or
Abstract + the Aktenzeichen, pointing at the original court's stable permalink where there is
one, else at the entscheidsuche `/view/` page. The builder is
`cytir.extract.entscheidsuche.build_passage`, moved unchanged out of the legacy
`ingest_entscheidsuche.py`, which now imports it.

Options (`--opt KEY=VALUE`): `root` (required): the work queue the legacy `toc` phase wrote
(`queue.jsonl`, one `{spider, viewid, view_url, json_url, date}` per line), or a directory that
holds it. `rate` (optional, requests per second, default 4: the driver's politeness budget).
A unit is one queue line: it ends `admitted`, `excluded(http_404)` (the origin says it is gone),
or `excluded(no_headnote_or_text_quality)` (the builder's own gate: no real Regeste); an HTTP
or parse failure is OUR failure and is `quarantined`.

Consent: `entscheidsuche.ch` is resolved live (declared basis) before any decision is fetched.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time

from cytir.extract import entscheidsuche as E
from cytir.sources.family import CHUNK_DECLARED, HttpSpec, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

QUEUE = "queue.jsonl"
DEFAULT_RATE = 4.0


@register_family
class EntscheidsucheFamily:
    family_id = "scratch_entscheidsuche_exact"
    route = "documented_source_api"
    direct_crawl = True
    chunk_policy_id = CHUNK_DECLARED
    http_spec = HttpSpec(timeout=45)   # task 1556: the runner builds and closes the client
    require_records = True

    def discover(self, ctx: RunContext):
        root = ctx.options.get("root")
        if not root:
            raise ValueError(f"{self.family_id} needs --opt root=<queue.jsonl or its directory>")
        path = os.path.join(root, QUEUE) if os.path.isdir(root) else root
        if not os.path.isfile(path):
            raise ValueError(f"{self.family_id}: {path!r} is not a file")
        units = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    q = json.loads(line)
                    units.append(Unit(key=q["json_url"], url=q["json_url"], meta={"q": q}))
        if not units:
            raise RuntimeError(f"{self.family_id}: {path!r} holds no decision; "
                               "an empty queue is a wrong input, not an empty source")
        ctx.notes.update({"queue": str(path), "decisions": len(units)})
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        r = ctx.http.get(unit.url)
        time.sleep(1.0 / max(float(ctx.options.get("rate", DEFAULT_RATE)), 0.1))
        if r.status_code in (404, 410):
            raise UnitExcluded(f"http_{r.status_code}")
        r.raise_for_status()
        body = r.content
        return [Captured(unit, body, {"status": r.status_code, "source_url": unit.url,
                                      "sha256": hashlib.sha256(body).hexdigest()})]

    def extract(self, cap: Captured, ctx: RunContext):
        data = json.loads(cap.body.decode("utf-8"))
        res = E.build_passage(data, cap.unit.meta["q"])
        if res is None:
            raise UnitExcluded("no_headnote_or_text_quality")
        p, extra = res
        row = dataclasses.asdict(p)
        row.update(extra)
        return [Doc(url=row["url"], text=row["text"], title=row["title"], lang=row["lang"],
                    extra={"row": row})]

    def chunk(self, doc: Doc, ctx: RunContext):
        # chunk.atomic: one decision is one record (the builder already caps the text)
        return [doc.extra["row"]]
