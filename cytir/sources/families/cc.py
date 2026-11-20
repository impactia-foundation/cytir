"""The Common Crawl route as a family (task 683, S7): `cc_authority_frontier_w7`.

Common Crawl holds the page bytes; it does not hold consent. The route is: RESOLVED CAPTURE
RECORDS in (`url`, `warc_filename`, `offset`, `length`, the digest, the discovery stage's stamps)
-> a byte-range read of the WARC member -> the shared web tail (`cytir.sources.families.web`).

What this adapter owns:
  discover  read the resolved-records JSONL (`--opt records=PATH`), validate each record, one
            unit per capture. A record without the discovery stage's stamps (`tier`,
            `source_type`, `trust_tier`) is refused AT DISCOVER: the adapter never invents a
            trust price (ADR-098; task 440 priced discovery classes).
  acquire   `cytir.acquire.warc.range_get` (<= 5 requests/s, ADR-014; an error is raised, never an
            empty) and the member is SEALED whole (gzip, as it arrived) by the framework.
  extract   unwrap the WARC member, require the capture's own HTTP 200.

What it does NOT own, and where that lives:
  * RESOLVING records from the CC index (`cytir/acquire/cc_columnar.py`, DuckDB over the columnar
    index / CDX): still a pipeline module, it moves in S9. It writes the records file this
    adapter reads; the file format is `cc_columnar._record`'s.
  * consent: the framework resolves it in CC mode (`direct_crawl = False`) per record host
    through the one resolver BEFORE any byte is read; a CCBot disallow prices `train_ok` only,
    never `search_ok` (ADR-024 am.5, enforced in `cytir.consent`, not here).
  * the chunk policy: declared in the contract (`chunk.generic`), read by `CHUNK_DECLARED`; the
    web tail refuses to start when the dispatcher's selection disagrees (`WebPageMixin.preflight`).

`--opt tape=PATH` replays a sealed tape (`cytir.acquire.tape`) instead of reading the mirror: a dry
run / golden with no egress.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

from cytir.acquire import warc
from cytir.sources.families.web import WebPageMixin
from cytir.sources.family import CHUNK_DECLARED, HttpSpec, Captured, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

REQUIRED_STAMPS = ("tier", "source_type", "trust_tier")
REQUIRED_FIELDS = ("url", "warc_filename", "offset", "length")


class RecordsError(ValueError):
    """The resolved-records file is not what this adapter reads."""


def load_records(path: str | Path) -> list[dict]:
    recs = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        missing = [k for k in REQUIRED_FIELDS + REQUIRED_STAMPS if rec.get(k) in (None, "")]
        if missing:
            raise RecordsError(f"{path}:{n}: record lacks {missing}. The discovery stage "
                               "decides the stamps; this adapter does not guess them.")
        recs.append(rec)
    if not recs:
        raise RecordsError(f"{path}: no records; an empty tranche is not a tranche")
    return recs


@register_family
class CCAuthorityFrontierW7(WebPageMixin):
    """Common Crawl pages of curated authority hosts (discovery frontier, wave 7)."""

    family_id = "cc_authority_frontier_w7"
    route = "common_crawl"
    direct_crawl = False
    chunk_policy_id = CHUNK_DECLARED
    http_spec = HttpSpec(timeout=60)   # task 1556: the runner builds and closes the client

    def preflight(self, ctx: RunContext) -> None:
        """A `--opt tape=PATH` run replays a sealed tape: it injects the tape client through
        `ctx.http`, the same door a test uses, so the runner builds no network client (task 1556)."""
        super().preflight(ctx)
        tape = ctx.options.get("tape")
        if tape and ctx.http is None:
            from cytir.acquire.tape import Tape
            ctx.http = Tape.load(tape).client()

    def discover(self, ctx: RunContext):
        path = ctx.options.get("records")
        if not path or not isinstance(path, str):
            raise ValueError("cc_authority_frontier_w7 needs --opt records=PATH (resolved CC "
                             "capture records, JSONL; see cytir.sources.families.cc)")
        recs = load_records(path)
        ctx.notes.update({"records_file": Path(path).name, "records": len(recs),
                          "bytes_to_read": sum(int(r["length"]) for r in recs),
                          "max_requests_per_s": warc.MAX_REQUESTS_PER_S})
        units = []
        for r in recs:
            key = f"{r['warc_filename']}:{r['offset']}:{r['length']}"
            units.append(Unit(key=key, url=r["url"], meta={"record": r},
                              consent_hosts=(urlsplit(r["url"]).hostname or "",)))
        return units

    def _limiter(self, ctx: RunContext) -> warc.RateLimiter:
        lim = self.__dict__.get("_lim")
        if lim is None:
            lim = self._lim = warc.RateLimiter(
                float(ctx.options.get("rate") or warc.MAX_REQUESTS_PER_S))
        return lim

    def acquire(self, unit: Unit, ctx: RunContext):
        rec = unit.meta["record"]
        if rec.get("content_truncated"):
            raise UnitExcluded(f"cc_content_truncated:{rec.get('truncation_reason') or 'flagged'}")
        member = warc.range_get(ctx.http, rec, limiter=self._limiter(ctx))
        if member is None:
            raise UnitExcluded("cc_mirror_miss")
        return [Captured(unit, member, {
            "warc_filename": rec["warc_filename"], "offset": int(rec["offset"]),
            "length": int(rec["length"]), "digest": rec.get("digest"),
            "fetch_time": rec.get("fetch_time"), "stamps": {k: rec[k] for k in REQUIRED_STAMPS}})]

    def page_html(self, cap: Captured) -> str:
        from cytir.sources.families.web import decode_html

        rec = warc.parse_warc_record(cap.body)
        if rec.status != 200:
            raise UnitExcluded(f"captured_http_{rec.status}")     # the capture's own status
        return decode_html(rec.body, rec.http_headers.get("content-type"))

    def passage_common(self, ctx: RunContext, unit: Unit) -> dict:
        from cytir.core.common import fold_www

        r = unit.meta["record"]
        seen = (r.get("fetch_time") or "")[:10] or ctx.today
        return {"domain": fold_www(urlsplit(unit.url).netloc), "tier": r["tier"],
                "source_type": r["source_type"], "trust_tier": float(r["trust_tier"]),
                "last_seen": seen}
