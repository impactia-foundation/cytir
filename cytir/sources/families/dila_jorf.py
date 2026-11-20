"""`dila_jorf` -- Journal officiel texts from the DILA open-data dump (task 696).

The declared family of task 695 (POLICY PR #499): bulk-dump route, Etalab-2.0, `chunk.atomic` (ONE
title-only locator per `JORFTEXT`: full title, nature, JO reference, NOR, ELI; the permalink points
at the full text). The task-140 floors are UNCHANGED by the declaration: publication year >= 2023
(`--opt min_year=N` overrides for a deliberate re-run), LOI / ORDONNANCE / DECRET always, an ARRETE
only when substantive. The parser is `cytir.extract.dila_jorf`, shared with the legacy CLI.

Options (`--opt KEY=VALUE`): `root` (required; a directory that contains the extracted
`Freemium_jorf_global_*.tar.gz` and/or increments, walked recursively; the newest version of a
record wins), `skip_ids_under` (optional; a tree whose records this run must not take again),
`min_year` (default 2023).
A unit is one `JORFTEXT*.xml`; every one ends `admitted`, `excluded(<reason>)` or `quarantined`.

A dump-route family contacts no host (ADR-076; the licence is the consent basis, plan ruling 17).
"""
from __future__ import annotations

import hashlib
import json

from cytir.extract.dila_jorf import MIN_YEAR_DEFAULT, parse_text_bytes
from cytir.sources.families import dila_dump
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

LICENSE = "Etalab-2.0"


@register_family
class DilaJorfFamily:
    family_id = "dila_jorf"
    route = "source_owned_bulk_dump"
    direct_crawl = False
    chunk_policy_id = CHUNK_DECLARED
    require_records = True     # a run that admits 0 texts refuses (smoke-test-yield)

    def __init__(self) -> None:
        self._versions: dila_dump.Versions | None = None
        self._l1_digests: dict[str, tuple[int, str]] = {}   # member -> (bytes, sha256); digests only

    def l1(self, ctx: RunContext) -> dict:
        """Persist the list of dump members this run reads (key, bytes, sha256): the L1 input
        record a stranger needs to replay the run (ADR-105 am.1). Writes `l1/dump-members.json`.

        Lists exactly the members the run takes (same `discover_units` order, cut at `ctx.limit`
        as `run_family` cuts). Only `{key: (bytes, sha256)}` is parked for `acquire` (about 100 B
        per member, never a body: a full dump is millions of members); `acquire` re-reads the
        member and verifies it against the parked digest (the ADR-105 am.1 replay guarantee)."""
        units = dila_dump.discover_units(self.family_id, ctx.options, ("JORFTEXT",), {})
        if ctx.limit and len(units) > ctx.limit:
            units = units[:ctx.limit]
        members = []
        self._l1_digests.clear()
        for u in units:
            body = dila_dump.read_unit(u)
            digest = hashlib.sha256(body).hexdigest()
            self._l1_digests[u.key] = (len(body), digest)
            members.append({"member": u.key, "bytes": len(body), "sha256": digest})
        blob = (json.dumps(members, sort_keys=True, indent=1) + "\n").encode("utf-8")
        target = ctx.out_dir / "l1"
        target.mkdir(parents=True, exist_ok=True)
        (target / "dump-members.json").write_bytes(blob)
        return {"path": "l1/dump-members.json", "members": len(members),
                "sha256": hashlib.sha256(blob).hexdigest()}

    def discover(self, ctx: RunContext):
        ctx.notes["min_year"] = int(ctx.options.get("min_year", MIN_YEAR_DEFAULT))
        units = dila_dump.discover_units(self.family_id, ctx.options, ("JORFTEXT",), ctx.notes)
        self._versions = dila_dump.Versions(units, ctx.options, ('JORFTEXT',), ctx.notes)
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        parked = self._l1_digests.pop(unit.key, None)    # pop first: a refused unit leaves no entry
        self._versions.check(unit)
        body = dila_dump.read_unit(unit)
        digest = hashlib.sha256(body).hexdigest()
        if parked is not None and parked[1] != digest:
            raise RuntimeError(f"{unit.key}: member changed since l1() recorded it (l1 sha256 "
                               f"{parked[1]}, acquire sha256 {digest}); the L1 record no longer "
                               "describes this run")
        return [Captured(unit, body, {"member": unit.key, "sha256": digest})]

    def extract(self, cap: Captured, ctx: RunContext):
        rec, reason = parse_text_bytes(cap.body, min_year=int(ctx.notes["min_year"]))
        if rec is None:
            raise UnitExcluded(reason)
        rec["license"] = LICENSE
        return [Doc(url=rec["url"], text=rec["text"], title=rec["title"], lang="fr",
                    extra={"row": rec})]

    def chunk(self, doc: Doc, ctx: RunContext):
        return [doc.extra["row"]]      # chunk.atomic: one JORF text is one record
