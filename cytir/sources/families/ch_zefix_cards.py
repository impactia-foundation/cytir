"""`ch_zefix_cards` -- Swiss commercial-register entity cards, on the Protocol (task 2485, S7), PARTIAL (dialect only).

What exists in the package. `cytir.corpus.register_cards.compose_register_card` is the ratified card
composer (ADR-060a: identity, purpose and officers from the register only; ADR-060 two channels) and
`assert_payload_contract` is its fail-closed payload check (both from task 621). `cytir.sources.zefix.facets`
parses a ZefixREST firm response into status / dates / people (task 363). This adapter moves NO logic: it
calls those two card functions on a PRESERVED seed file.

Consent is declared on the contract row (task 2879, RULING 81: licence, linked-data service only, no refresh by
the website backend, search_ok true, train_ok false); this adapter writes no consent field.

What it lacks, said with `PARTIAL`:
  * `dialect:zefix-api`: the producer never lived in the package. The live pull (LINDAS purpose + EHRAID
    batches, ZefixREST per firm) and the verify/grade step (ADR-041, needs the candidate website's page text)
    were one-shot task scripts from the 2026-07 register-card and 2026-08 refresh work (retired, not shipped). No stage here contacts a host.

Input (`--opt seeds=<file.jsonl>`): one JSON object per line, `{"seed": {...}, "evidence": {...},
"confidence": 0.0-1.0}`, exactly the three arguments of `compose_register_card` (its docstring lists the seed
keys). `seed.last_seen` defaults to the run date. The unit key is `register_id`.

Chunk: the contract declares `chunk-policy-atomic-record.v1`: a card is ONE row and is never cut, so `chunk`
returns the composed card. An empty seed file is a wrong path, not an empty register: discover refuses.
"""
from __future__ import annotations

import json
from pathlib import Path

from cytir.corpus import register_cards as RC
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit
from cytir.sources.registry import register_family


@register_family
class ChZefixCardsFamily:
    family_id = "ch_zefix_cards"
    route = "api"
    direct_crawl = False
    chunk_policy_id = CHUNK_DECLARED
    require_records = True
    PARTIAL = ("dialect:zefix-api",)

    def discover(self, ctx: RunContext):
        path = ctx.options.get("seeds")
        if not path:
            raise ValueError(f"{self.family_id} needs --opt seeds=<jsonl of {{seed, evidence, confidence}}>")
        units = []
        for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            units.append(Unit(key=str(row["seed"]["register_id"]), url=row["seed"]["url"],
                              meta={"line": n, "row": row}))
        if not units:
            raise RuntimeError(f"{self.family_id}: {path} holds no seed")
        ctx.notes["seeds"] = len(units)
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        body = json.dumps(unit.meta["row"], sort_keys=True, ensure_ascii=False).encode("utf-8")
        return [Captured(unit, body, {"status": 200, "content_type": "application/json",
                                      "source_url": unit.url, "line": unit.meta["line"]})]

    def extract(self, cap: Captured, ctx: RunContext):
        row = json.loads(cap.body)
        seed = dict(row["seed"])
        seed.setdefault("last_seen", ctx.today)
        rec = RC.compose_register_card(seed, evidence=row.get("evidence") or {},
                                       confidence=row["confidence"])
        RC.assert_payload_contract(rec)
        return [Doc(url=rec["url"], text=rec["dense_text"], title=rec["title"], lang=rec["lang"],
                    source_text=rec["dense_text"], extra={"record": rec})]

    def chunk(self, doc: Doc, ctx: RunContext):
        return [doc.extra["record"]]
