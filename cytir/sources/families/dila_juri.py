"""`dila_juri` -- French supreme-court decisions from the DILA open-data dumps (task 696).

JADE (Conseil d'Etat, `CETATEXT*.xml`) and CASS (Cour de cassation, `JURITEXT*.xml`), the
declared family of task 695 (POLICY PR #499): bulk-dump route, Etalab-2.0, `chunk.atomic`
(one locator record per decision), no year floor (task 695 section 4), the quality floor
stays. The parser is `cytir.extract.dila_juri` (the legacy CLI wrapper is deleted, task 1221).

Options (`--opt KEY=VALUE`): `root` (required; a directory that contains the extracted `cass/`
and/or `jade/` trees of the DILA `Freemium_{cass,jade}_global_*.tar.gz` dumps, walked
recursively; the newest increment version of a decision wins), `skip_ids_under` (optional; a
tree whose decisions this run must not take again: an increments-only run adds the NEW decisions).
A unit is one decision file, so every file scanned ends `admitted`,
`excluded(<reason>)` or `quarantined`, and the pack manifest prints `in = admitted +
excluded + quarantined`.

A dump-route family contacts no host: no per-host consent is resolved (ADR-076; the
licence is the consent basis, plan ruling 17). `www.legifrance.gouv.fr` is never fetched; it
is only the canonical URL the record points the agent to.
"""
from __future__ import annotations

import hashlib
import os
import pathlib

from cytir.extract.dila_juri import parse_decision_bytes
from cytir.sources.families import dila_dump
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

_PREFIXES = ("JURITEXT", "CETATEXT")
LICENSE = "Etalab-2.0"


def iter_decision_paths(root: str):
    """Every `JURITEXT*.xml` / `CETATEXT*.xml` under `root`, sorted (stable unit order)."""
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".xml") and fn.startswith(_PREFIXES):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


@register_family
class DilaJuriFamily:
    family_id = "dila_juri"
    route = "source_owned_bulk_dump"
    direct_crawl = False
    chunk_policy_id = CHUNK_DECLARED
    require_records = True     # a run that admits 0 decisions refuses (smoke-test-yield)

    def __init__(self) -> None:
        self._versions: dila_dump.Versions | None = None

    def discover(self, ctx: RunContext):
        root = ctx.options.get("root")
        if not root:
            raise ValueError("dila_juri needs --opt root=<dir containing the extracted "
                             "cass/ and/or jade/ dump trees>")
        if not os.path.isdir(root):
            raise ValueError(f"dila_juri: root {root!r} is not a directory")
        paths = iter_decision_paths(root)
        if not paths:
            raise RuntimeError(f"dila_juri: no JURITEXT*.xml / CETATEXT*.xml under {root!r}; "
                               "an empty walk is a wrong root, not an empty source")
        by_court = {"cass": sum(os.path.basename(p).startswith("JURITEXT") for p in paths),
                    "jade": sum(os.path.basename(p).startswith("CETATEXT") for p in paths)}
        ctx.notes.update({"root": str(root), "files": len(paths), "files_by_court": by_court})
        units = [Unit(key=os.path.relpath(p, root), url=pathlib.Path(p).resolve().as_uri(),
                      meta={"path": p}) for p in paths]
        self._versions = dila_dump.Versions(units, ctx.options, _PREFIXES, ctx.notes)
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        self._versions.check(unit)
        with open(unit.meta["path"], "rb") as fh:
            body = fh.read()
        return [Captured(unit, body, {"member": unit.key,
                                      "sha256": hashlib.sha256(body).hexdigest()})]

    def extract(self, cap: Captured, ctx: RunContext):
        rec, reason = parse_decision_bytes(cap.body, floors=None)
        if rec is None:
            raise UnitExcluded(reason)
        rec["license"] = LICENSE
        return [Doc(url=rec["url"], text=rec["text"], title=rec["title"], lang="fr",
                    extra={"row": rec})]

    def chunk(self, doc: Doc, ctx: RunContext):
        # chunk.atomic: one decision is one record (the declared policy; the text is
        # already capped by the parser)
        return [doc.extra["row"]]
