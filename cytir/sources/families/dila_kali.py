"""`dila_kali` -- in-force KALI collective-agreement articles from the DILA open-data dump (task 696).

The declared family of task 695 (POLICY PR #499): bulk-dump route, Etalab-2.0, `native-chunked`
(one record per in-force `KALIARTI` article with real clause text, >= 260 chars, at most 30 per
convention, the flood guard; an article over the embedder window is split at its paragraph marks,
never cut: ADR-090 am., Tim 2026-10-07, ruling 27 T6 Q-b; the 4,000-char cap is gone). Convention containers (`kali_conv`) are already served
and are not part of this family. The parser is `cytir.extract.dila_kali`, shared with the legacy CLI.

Options (`--opt KEY=VALUE`): `root` (required; a directory that contains the extracted
`Freemium_kali_global_*.tar.gz` and/or increment extractions, walked recursively; the newest
version of a record wins), `skip_ids_under` (optional; a tree whose records this run must not take
again). A unit is one `KALIARTI*.xml`, so every file
scanned ends `admitted`, `excluded(<reason>)` or `quarantined`.

A dump-route family contacts no host: no per-host consent is resolved (ADR-076; the licence is the
consent basis, plan ruling 17). `www.legifrance.gouv.fr` is never fetched.
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter

from cytir.extract.dila_kali import (MAX_ARTICLES_PER_CONV, TOKEN_CHECK_FLOOR_CHARS,
                                   parse_article_bytes, parse_article_rows)
from cytir.sources.families import dila_dump
from cytir.sources.families import loda     # `loda.embed_token_counter`: the one embed-window counter
from cytir.sources.family import CHUNK_NATIVE_CHUNKED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

LICENSE = "Etalab-2.0"


@register_family
class DilaKaliFamily:
    family_id = "dila_kali"
    route = "source_owned_bulk_dump"
    direct_crawl = False
    chunk_policy_id = CHUNK_NATIVE_CHUNKED
    require_records = True     # a run that admits 0 articles refuses (smoke-test-yield)
    #: the split loses nothing, so any loss is a defect (the loda reading; `staged_text_guard`)
    staged_text_min_coverage = 1.0

    def preflight(self, ctx: RunContext):
        """Refuse before a file is read when the run needs the embed tokenizer and cannot load it.
        An article over `TOKEN_CHECK_FLOOR_CHARS` characters is split by the embedder's own token
        count; without it a run would quarantine every such article (a stageable pack that lost
        them). An article that long is a file of more than that many bytes, so a dump without such
        a file never needs the tokenizer (the offline samples)."""
        root = ctx.options.get("root")
        if not root or not os.path.isdir(root):
            return                       # `discover` says what is wrong with the root
        big = None
        for d, _s, fs in os.walk(root):
            for f in fs:
                if not (f.startswith("KALIARTI") and f.endswith(".xml")):
                    continue
                path = os.path.join(d, f)
                if os.path.getsize(path) <= TOKEN_CHECK_FLOOR_CHARS:
                    continue             # text characters <= file bytes: cannot be long enough
                with open(path, "rb") as fh:
                    rec, _k, _r = parse_article_bytes(fh.read())
                if rec is not None and len(rec["text"]) > TOKEN_CHECK_FLOOR_CHARS:
                    big = path
                    break
            if big:
                break
        if big is None:
            return
        try:
            loda.embed_token_counter()
        except Exception as exc:                                        # noqa: BLE001
            from cytir.sources.run_family import RunError
            raise RunError(f"dila_kali: {big} may be over the embed window and the embed tokenizer "
                           f"cannot be loaded ({type(exc).__name__}); see loda_exact's preflight "
                           "for the fetch recipe. A run that cannot measure the window is "
                           "refused") from exc

    def __init__(self) -> None:
        self._per_conv: Counter = Counter()
        self._versions: dila_dump.Versions | None = None

    def discover(self, ctx: RunContext):
        self._per_conv = Counter()
        units = dila_dump.discover_units(self.family_id, ctx.options, ("KALIARTI",), ctx.notes)
        self._versions = dila_dump.Versions(units, ctx.options, ('KALIARTI',), ctx.notes)
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        self._versions.check(unit)
        body = dila_dump.read_unit(unit)
        return [Captured(unit, body, {"member": unit.key,
                                      "sha256": hashlib.sha256(body).hexdigest()})]

    def extract(self, cap: Captured, ctx: RunContext):
        rows, conv_key, reason, source_text = parse_article_rows(
            cap.body, count_tokens=lambda text: loda.embed_token_counter()(text))
        if not rows:
            raise UnitExcluded(reason)
        if self._per_conv[conv_key] >= MAX_ARTICLES_PER_CONV:
            raise UnitExcluded(f"convention_cap_{MAX_ARTICLES_PER_CONV}")
        self._per_conv[conv_key] += 1
        for rec in rows:
            rec["license"] = LICENSE
        return [Doc(url=rows[0]["url"], text=rows[0]["text"], title=rows[0]["title"], lang="fr",
                    source_text=source_text, extra={"rows": rows})]

    def chunk(self, doc: Doc, ctx: RunContext):
        return list(doc.extra["rows"])   # one article is one row, or several if over the window
