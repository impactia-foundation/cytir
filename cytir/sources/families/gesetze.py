"""`gesetze` -- German federal law from gesetze-im-internet.de, on the Protocol (task 1400, S7).

The BMJ's bulk XML: `gii-toc.xml` lists every law, each law is `/<abbr>/xml.zip`. The code is
`cytir.sources.gesetze.ingest` (moved there from the old flat `ingest_gesetze` script, which is now a
forwarding shim): this adapter DELEGATES to its functions and adds only the door.

  * `discover`: fetch `gii-toc.xml` (`toc_links`), one unit per law ZIP, keyed by the law's
    abbreviation. An empty table of contents is an error, not an empty source.
  * `acquire`: fetch the ZIP with the driver's own retry, back-off and politeness pause
    (`fetch_law_zip`), returning the bytes for the framework to seal. A 404/410 after the
    retries is the origin's word (`UnitExcluded`); every other failure is ours (quarantine).
  * `extract`: `parse_law_zip`, the one parser the live driver and the L0 replay both call;
    one `Doc` per norm it keeps.
  * `chunk`: the norm IS the row (`CHUNK_NATIVE`: the driver has never split a norm). The row is
    built from the driver's own `Passage` by `passage_source_record`.

CONSENT, LICENCE AND ROUTE: the contract row for `gesetze` declares consent (UrhG §5, plan.md RULING 77,
task 2854); the route stays as declared there. This module writes none of them. It stamps NO licence on a row (the driver set none);
the `route` label below is only the manifest label for what the code does (it fetches the
publisher's bulk ZIPs), not a declared basis. What the code shows, for Tim to rule on, is in
the task 1400 doc (`## Result`).

Options (`--opt KEY=VALUE`): `toc` (optional; a local `gii-toc.xml` file used instead of fetching it,
for an offline plan). Units still point at the host; acquire fetches them through `ctx.http`.

PARTIAL: `l1` (the optional ADR-105 stage) is not ported -- the L1 filing of this family stays the
L0-ZIP derivation of task 417, and the family has no declared chunk policy to port it under.
"""
from __future__ import annotations

import hashlib
import pathlib

from cytir.core import common as C
from cytir.sources.family import CHUNK_NATIVE, HttpSpec, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.gesetze import ingest as G
from cytir.sources.registry import register_family

HOST = "www.gesetze-im-internet.de"
_ORIGIN_GONE = (404, 410)


@register_family
class GesetzeFamily:
    family_id = "gesetze"
    route = "source_owned_bulk_dump"      # manifest label of what the code does; the row says unknown
    direct_crawl = True                   # we contact the origin ourselves
    chunk_policy_id = CHUNK_NATIVE
    http_spec = HttpSpec(timeout=G._TIMEOUT)   # task 1556: the runner builds and closes the client
    require_records = True
    PARTIAL = ("l1",)

    def discover(self, ctx: RunContext):
        toc = ctx.options.get("toc")
        if toc:
            links = G.toc_links(pathlib.Path(toc).read_bytes())
        else:
            r = ctx.http.get(G.TOC)
            if r.status_code != 200:
                raise RuntimeError(f"gesetze: {G.TOC} answered {r.status_code}")
            links = G.toc_links(r.content)
        if not links:
            raise RuntimeError("gesetze: the table of contents lists no law ZIP; "
                               "an empty walk is a wrong source, not an empty one")
        ctx.notes["laws"] = len(links)
        return [Unit(key=u.rsplit("/", 2)[-2], url=u, consent_hosts=(HOST,)) for u in links]

    def acquire(self, unit: Unit, ctx: RunContext):
        r = G.fetch_law_zip(ctx.http, unit.url)
        if r is None:
            raise RuntimeError(f"gesetze: {unit.url} unreachable after retries")
        if r.status_code in _ORIGIN_GONE:
            raise UnitExcluded(f"http_{r.status_code}")
        if r.status_code != 200:
            raise RuntimeError(f"gesetze: {unit.url} answered {r.status_code}")
        return [Captured(unit, r.content, {
            "status": r.status_code, "content_type": r.headers.get("content-type", ""),
            "source_url": unit.url, "sha256": hashlib.sha256(r.content).hexdigest()})]

    def extract(self, cap: Captured, ctx: RunContext):
        rows = G.parse_law_zip(cap.unit.url, cap.body)
        if not rows:
            raise UnitExcluded("no_norm_with_text")
        return [Doc(url=p.url, text=p.text, title=p.title, lang=p.lang, source_text=p.text,
                    extra={"passage": p}) for p in rows]

    def chunk(self, doc: Doc, ctx: RunContext):
        p = doc.extra["passage"]
        p.last_seen = ctx.today
        p.content_digest = C.sha1(p.text)
        return [C.passage_source_record(p)]
