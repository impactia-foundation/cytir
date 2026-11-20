"""`service_public_dila_exact` -- the preserved DILA Service-Public dump, exact markdown (task 1403).

The sibling of `service_public_fiche` (same `vosdroits-*.zip` archive of the DILA open-data
dump, same `F*.xml` fiche members). The two do NOT share a renderer, so the port cannot delegate
to the fiche adapter: this row keeps the archive's exact preserved surface (`# title`, then every
`Introduction`/`Texte`/... prose leaf of the fiche, in document order, plus the outlink records),
sealed by the driver `cytir.sources.service_public.dump`; the fiche family renders a declared,
head-free surface through `cytir.extract.service_public_fiche`. Which of the two feeds the index is
a ruling of Tim's, not this adapter's.

STAGE MAP
  discover     adapter   one unit per `F*.xml` member of the zip given as `--opt archive=PATH`
  consent      framework a dump-route family contacts no host (units are `file:` URIs)
  acquire      adapter   the member's bytes (the zip is already on disk; the framework seals them)
  extract      moved     `dump.extract_member`, the driver's own function (one home)
  chunk        adapter   the row's declared generic policy via `declared_policy_selection`
                         (task 2260), as the sibling `service_public_fiche` does
  seal_dump    moved     the driver's `materialize` (original/markdown/outlinks/quarantine
                         stage receipts + RUN.json) from `--opt archive/headers/metadata/...`

CONSENT AND LICENCE are read from the family's contract row, never written here: the row still
holds `unknown` for both, and the run notes carry those `unknown` values as they are.

Options (`--opt KEY=VALUE`): `archive` (required, the zip). For `seal_dump` also `headers`,
`metadata` (the transport receipt), `root`, `requested_url`, `fetch_vantage`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import urllib.parse
import zipfile

from cytir.acquire.artifact_io import sha256_file
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.chunk_selection import declared_policy_selection
from cytir.sources.registry import register_family
from cytir.sources.families import service_public_fiche as sibling
from cytir.sources.families.service_public_fiche import _after_title
from cytir.sources.service_public import dump

NO_FICHE = "no_admissible_practical_fiche_markdown"
DUPLICATE = "duplicate_canonical_url"


def contract_terms(ctx: RunContext) -> dict:
    """Consent and licence exactly as the row declares them (today: `unknown`)."""
    row = ctx.contract or {}
    consent, licence = row.get("consent", {}), row.get("licence", {})
    return {"search_ok": consent.get("search_ok"), "train_ok": consent.get("train_ok"),
            "consent_resolution": consent.get("resolution"),
            "licence_basis": licence.get("basis"), "licence_spdx": licence.get("spdx")}


def _archive(ctx: RunContext) -> pathlib.Path:
    path = ctx.options.get("archive")
    if not path:
        raise ValueError("service_public_dila_exact needs --opt archive=<the DILA vosdroits zip>")
    if not pathlib.Path(path).is_file():
        raise ValueError(f"service_public_dila_exact: archive {path!r} is not a file")
    return pathlib.Path(path).resolve()


@register_family
class ServicePublicDilaExactFamily:
    family_id = "service_public_dila_exact"
    route = "source_owned_bulk_dump"
    direct_crawl = False
    chunk_policy_id = CHUNK_DECLARED
    require_records = True

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = contract_terms(ctx)
        archive = _archive(ctx)
        sha = sha256_file(archive)
        units = []
        with zipfile.ZipFile(archive) as z:
            members = sorted(z.infolist(), key=lambda i: i.filename)
            if len(members) > 100_000 or sum(i.file_size for i in members) > 4_000_000_000:
                raise dump.ServicePublicError("dump exceeds bounded archive limits")
            for item in members:
                safe = pathlib.PurePosixPath(item.filename)
                if safe.is_absolute() or ".." in safe.parts:
                    raise dump.ServicePublicError(f"unsafe archive member: {item.filename}")
                if item.is_dir() or not safe.name.startswith("F") or not safe.name.endswith(".xml"):
                    continue
                units.append(Unit(key=item.filename, url=f"{archive.as_uri()}#{item.filename}",
                                  meta={"archive": str(archive), "archive_sha256": sha}))
        if not units:
            raise RuntimeError("service_public_dila_exact: no F*.xml member in the archive; "
                               "an empty walk is a wrong archive, not an empty source")
        ctx.notes.update({"archive_sha256": sha, "files": len(units)})
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        with zipfile.ZipFile(unit.meta["archive"]) as z:
            body = z.read(unit.key)
        return [Captured(unit, body, {"member": unit.key,
                                      "sha256": hashlib.sha256(body).hexdigest()})]

    def extract(self, cap: Captured, ctx: RunContext):
        row, links = dump.extract_member(cap.unit.key, cap.body, cap.unit.meta["archive_sha256"])
        if row is None:
            raise UnitExcluded(NO_FICHE)
        if row["url"] in self._seen:
            raise UnitExcluded(DUPLICATE)
        self._seen.add(row["url"])
        return [Doc(url=row["url"], text=row["markdown"], title=row["title"], lang="fr",
                    extra={"row": row, "outlinks": links})]

    def preflight(self, ctx: RunContext) -> None:
        self._selection = declared_policy_selection(ctx)

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.core import common as C

        selection = getattr(self, "_selection", None)
        if selection is None:
            raise RuntimeError(f"{self.family_id}.chunk ran before preflight: the declared chunk "
                               "policy is resolved in preflight; call preflight(ctx) first")
        row = doc.extra["row"]
        host = urllib.parse.urlsplit(doc.url).netloc
        stamps = selection.stamps()
        out = []
        for ordinal, unit in enumerate(C.chunk_markdown_channels(doc.text, selection=selection)):
            unit = dataclasses.replace(unit, stamps=stamps)
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(_after_title(unit.text, doc.title), 25), lang="fr",
                domain=host, tier="A", source_type="official", trust_tier=1.0,
                last_seen=ctx.today, license=sibling.LICENSE, license_version=sibling.LICENSE_VERSION,
                attribution=sibling.ATTRIBUTION, anchor=f"chunk-{ordinal}",
                master={"document_id": row["document_id"],
                        "archive_member_sha256": row["archive_member_sha256"]})
            out.append(C.passage_source_record(p))
        return out

    def seal_dump(self, ctx: RunContext) -> dict:
        """Seal the preserved stages (the driver's `materialize`)."""
        o = ctx.options
        p = pathlib.Path
        return dump.materialize(archive=_archive(ctx), headers=p(o["headers"]).resolve(),
                                metadata=p(o["metadata"]).resolve(), root=p(o["root"]).resolve(),
                                requested_url=o["requested_url"],
                                fetch_vantage=o["fetch_vantage"])
