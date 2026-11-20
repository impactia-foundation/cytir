"""`service_public_fiche` -- Service-Public information sheets from the DILA open-data archive (task 696).

The declared family of task 629 (POLICY PR): bulk-dump route, Licence Ouverte / Etalab-2.0, chunked
under the declared `chunk.generic` policy (the structured Markdown chunker, one fiche to a few
passages). The renderer is `cytir.extract.service_public_fiche`, shared with the restage tool: the
fiche's own title, then its body as Markdown with the headings of its chapters and sections, and
every principal XML text block conserved. The surface is the declared, head-free one: no bookkeeping
of ours is written into the text that is embedded and searched.

Options (`--opt KEY=VALUE`): `root` (required; a directory that contains the extracted
`vosdroits-*.zip` archive, or a cut of it, walked recursively). A unit is one `F*.xml` (the fiche
files; the archive's `N*`, `R*` and other members are not fiches and are not walked), so every file
scanned ends `admitted`, `excluded(<reason>)` or `quarantined`.

A dump-route family contacts no host: no per-host consent is resolved (ADR-076; the licence is the
consent basis). The Service-Public site is never fetched.
"""
from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from urllib.parse import urlsplit

from cytir.sources.families import dila_dump
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

LICENSE = "Etalab-2.0"
LICENSE_VERSION = "2.0"
ATTRIBUTION = "Direction de l'information légale et administrative (DILA), Service-Public.fr"
NOT_A_FICHE = "not_a_fiche_or_missing_title_url_identifier_date_or_content"


def _policy_selection(ctx: RunContext):
    """The family's declared chunk policy as an explicit structured selection (the file the
    registry names for the contract's sha256, and the contract's tokenizer revision): the run
    needs no `AW_CHUNK*` variable, and a policy that is not the declared one never runs."""
    from cytir.chunk import registry as chunk_registry
    from cytir.core import common as C
    from cytir.chunk import dispatch as _dispatch

    cp = ctx.contract["chunk_policy"]
    entry = chunk_registry.entry(cp["sha256"])
    repo = pathlib.Path(__file__).resolve().parents[3]
    return _dispatch.explicit_structured_chunker_selection(
        str(repo / entry["file"]), cp["sha256"], cp["tokenizer_revision"])


def _after_title(text: str, title: str) -> str:
    """The passage without its leading title line (the snippet says what the page says, not its
    own name again); the whole text when nothing follows the title."""
    rest = text[len(title):].strip() if title and text.startswith(title) else text
    return rest or text


@register_family
class ServicePublicFicheFamily:
    family_id = "service_public_fiche"
    route = "source_owned_bulk_dump"
    direct_crawl = False
    chunk_policy_id = CHUNK_DECLARED
    require_records = True     # a run that admits 0 fiches refuses (smoke-test-yield)

    def preflight(self, ctx: RunContext) -> None:
        sel = _policy_selection(ctx)
        ctx.notes["chunker"] = {"mode": sel.mode, "epoch": sel.epoch, "legend": sel.legend,
                                "policy_sha256": sel.policy_sha256,
                                "tokenizer_revision": sel.tokenizer_revision}
        self._selection = sel

    def discover(self, ctx: RunContext):
        # the DILA walk takes the `<PREFIX>*.xml` files; a fiche's id is `F<digits>`
        units = [u for u in dila_dump.discover_units(self.family_id, ctx.options, ("F",), ctx.notes)
                 if pathlib.PurePosixPath(u.key).name[1:-len(".xml")].isdigit()]
        if not units:
            raise RuntimeError(f"{self.family_id}: no F<number>.xml under {ctx.options['root']!r}; "
                               "an empty walk is a wrong root, not an empty source")
        ctx.notes["files"] = len(units)
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        body = dila_dump.read_unit(unit)
        return [Captured(unit, body, {"member": unit.key,
                                      "sha256": hashlib.sha256(body).hexdigest()})]

    def extract(self, cap: Captured, ctx: RunContext):
        from cytir.extract.service_public_fiche import render

        row = render(cap.body, "", cap.unit.key, provenance_head=False)
        if row is None:
            raise UnitExcluded(NOT_A_FICHE)
        return [Doc(url=row["url"], text=row["markdown"], title=row["title"], lang="fr",
                    extra={"row": row})]

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch

        row = doc.extra["row"]
        host = urlsplit(doc.url).netloc
        out = []
        stamps = self._selection.stamps()
        for ordinal, unit in enumerate(_dispatch.chunk_markdown_channels(doc.text,
                                                                 selection=self._selection)):
            unit = dataclasses.replace(unit, stamps=stamps)
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(_after_title(unit.text, doc.title), 25), lang="fr",
                domain=host, tier="A", source_type="official", trust_tier=1.0,
                last_seen=ctx.today, license=LICENSE, license_version=LICENSE_VERSION,
                attribution=ATTRIBUTION, anchor=f"chunk-{ordinal}",
                master={"fiche_id": row["fiche_id"], "modified_date": row["modified_date"],
                        "source_sha256": row["source_sha256"]})
            out.append(C.passage_source_record(p))
        return out
