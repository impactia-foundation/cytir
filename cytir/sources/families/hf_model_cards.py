"""`scratch_ai_hf_model_cards_exact` -- Hugging Face model cards (task 1101, S7).

Per model: ONE structured summary passage (params, licence, task, languages, dates, downloads) and
the README body chunked (skipped for a gated model, an empty card or junk). The card shape is
`cytir.extract.hf_model_cards.card_records`, moved unchanged out of the legacy
`cytir.sources.hf_model_cards.ingest`; `records_for_model` is the body of that driver's `parse` loop. The
legacy CLI and the refresh lane call the same functions.

Options (`--opt KEY=VALUE`): `root` (required): a directory with `model_list.json` (the legacy
`enumerate` phase's output) and `cards/` (the `fetch` phase's cache: `<org>__<model>.meta.json`,
`.detail.json`, `.README.md`); `cache_dir` (optional, default `<root>/cards`). A unit is one listed
model: `admitted`, or `excluded(no_detail | private_disabled | quant_spam_skipped |
no_passage)`.

Chunking is the central dispatcher's selection (`CHUNK_DISPATCHER`), recorded in the manifest; the
contract declares none yet.
Consent: `huggingface.co` is resolved live before any unit is read.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib

from cytir.extract import hf_model_cards as H
from cytir.sources.family import CHUNK_DISPATCHER, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

HOST = "huggingface.co"
LIST = "model_list.json"
#: the parse counters that name why a model produced no record, in the legacy order
_WHY = ("no_detail", "private_disabled", "quant_spam_skipped")


@register_family
class HfModelCardsFamily:
    family_id = "scratch_ai_hf_model_cards_exact"
    route = "documented_source_api"
    direct_crawl = True
    chunk_policy_id = CHUNK_DISPATCHER
    require_records = True

    def __init__(self) -> None:
        self._top500: set[str] = set()
        self._cache = ""

    def discover(self, ctx: RunContext):
        root = ctx.options.get("root")
        if not root:
            raise ValueError(f"{self.family_id} needs --opt root=<dir with {LIST} and cards/>")
        lst_path = os.path.join(root, LIST)
        if not os.path.isfile(lst_path):
            raise ValueError(f"{self.family_id}: {lst_path!r} is not a file")
        self._cache = ctx.options.get("cache_dir") or os.path.join(root, "cards")
        models = json.load(open(lst_path, encoding="utf-8"))["models"]
        self._top500 = {m["id"] for m in models if m.get("dl_rank") and m["dl_rank"] <= 500}
        if not models:
            raise RuntimeError(f"{self.family_id}: {lst_path!r} lists no model; "
                               "an empty list is a wrong input, not an empty source")
        uri = pathlib.Path(lst_path).resolve().as_uri()
        ctx.notes.update({"list": lst_path, "cache": self._cache, "models": len(models)})
        return [Unit(key=m["id"], url=f"{uri}#{m['id']}", meta={"m": m}, consent_hosts=(HOST,))
                for m in models]

    def acquire(self, unit: Unit, ctx: RunContext):
        body = json.dumps(unit.meta["m"], sort_keys=True, ensure_ascii=False).encode("utf-8")
        return [Captured(unit, body, {"member": unit.key,
                                      "sha256": hashlib.sha256(body).hexdigest()})]

    def extract(self, cap: Captured, ctx: RunContext):
        stats = H.new_parse_stats(1)
        recs = H.records_for_model(cap.unit.meta["m"], self._cache, self._top500, set(), stats)
        if not recs:
            raise UnitExcluded(next((k for k in _WHY if stats[k]), "no_passage"))
        return [Doc(url=r["url"], text=r["text"], title=r["title"], lang=r["lang"],
                    extra={"row": r}) for r in recs]

    def chunk(self, doc: Doc, ctx: RunContext):
        # the card shape already chunked the README body (one record per chunk)
        return [doc.extra["row"]]
