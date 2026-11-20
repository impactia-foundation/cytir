"""`openalex_academic_dump` -- scholarly works from the OpenAlex snapshot (task 696).

The declared family of tasks 629 and 684: bulk-dump route, a CC0-1.0 dataset (the instrument is
OpenAlex's own statement, https://docs.openalex.org), chunked under the declared `chunk.generic`
policy. The extractor is `cytir.extract.openalex_academic` (the logic of
`cytir.sources.openalex.materialize_pilot`, the old materializer script, but with the ADR-102 age-scaled ladder as its selection, task 1250): one parquet part of the snapshot is
one unit; the pinned selection (target fields and `Law`, an abstract, not retracted, 2000 or later,
EN/DE/FR/IT or no language, a resolvable address, cited or core or a recent work of a target
country) picks its works; each work renders to one Markdown document and then to one passage (more
than one only if the declared chunker cuts a long abstract), under the work's OpenAlex address.

Options (`--opt KEY=VALUE`): `root` (required; a directory laid out as the snapshot,
`updated_date=YYYY-MM-DD/part_NNNN.parquet`). Optional: `plan` (a selection-plan file of the
old materializer, `.json` or `.json.gz`), with `source_run_sha256` and `target_records`, to take only the
parts that plan names (the whole-part tranche policy; without it every part under `root` is a unit).
A unit ends `admitted` (one or more passages) or `excluded(no_work_selected_by_the_policy)`.

L0: the part is a public, content-addressed file that the operator already holds (the snapshot is the
durable archive, 150 to 460 MB a part), so the framework seals a POINTER to it: the part's relative
path, size and sha256 (read in 1 MiB blocks, never whole). The extractor then streams the works out of
the part itself. A part is never loaded into memory; the passages of one part are (the door keeps a
unit's records in a list).

Persons (ADR-039; founder ruling 7): a work's authors are natural persons. They ARE in the embedded
text (the dense and sparse surfaces), exactly as the served rows of this family have them: the surface
is a contract. They are NOT in the snippet (the first 25 words of the abstract) and not in any payload
field that ships: the display names also go in the master-only `_master.authors`; the venue is the
master-only `attribution`. No ORCID is read. Neither ships in the compact dump; the licence name does.

A dump-route family contacts no host: no per-host consent is resolved (ADR-076; the licence is the
consent basis, ADR-115 r1-openalex). Nothing is fetched from OpenAlex.
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import os
import pathlib

from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

NO_WORK_SELECTED = "no_work_selected_by_the_policy"
POINTER_SCHEMA = "aw.openalex-part-pointer.v1"
DOMAIN = "openalex.org"
BLOCK = 1 << 20


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


def _open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(
        path, encoding="utf-8")


def _planned_paths(ctx: RunContext) -> list[str] | None:
    from cytir.extract import openalex_academic as E

    plan_file = ctx.options.get("plan")
    if not plan_file:
        return None
    run_sha, target = ctx.options.get("source_run_sha256"), ctx.options.get("target_records")
    if not run_sha or not target:
        raise ValueError("--opt plan=FILE needs --opt source_run_sha256=SHA256 and "
                         "--opt target_records=N (the plan is checked against both)")
    with _open_text(plan_file) as fh:
        plan = json.load(fh)
    paths, records = E.planned_relative_paths(
        plan, expected_source_run_sha256=run_sha, target_records=int(target))
    ctx.notes.update({"plan_parts": len(paths), "plan_selected_records": records})
    return paths


def _sha256_file(path) -> tuple[int, str]:
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(BLOCK), b""):
            h.update(block)
            n += len(block)
    return n, h.hexdigest()


@register_family
class OpenAlexAcademicDumpFamily:
    family_id = "openalex_academic_dump"
    route = "source_owned_bulk_dump"
    direct_crawl = False
    chunk_policy_id = CHUNK_DECLARED
    require_records = True     # a run that admits 0 passages refuses (smoke-test-yield)

    def preflight(self, ctx: RunContext) -> None:
        sel = _policy_selection(ctx)
        ctx.notes["chunker"] = {"mode": sel.mode, "epoch": sel.epoch, "legend": sel.legend,
                                "policy_sha256": sel.policy_sha256,
                                "tokenizer_revision": sel.tokenizer_revision}
        self._selection = sel

    def discover(self, ctx: RunContext):
        root = ctx.options.get("root")
        if not root:
            raise ValueError(f"{self.family_id} needs --opt root=<directory laid out as the "
                             "snapshot: updated_date=YYYY-MM-DD/part_NNNN.parquet> (or --from DIR)")
        if not os.path.isdir(root):
            raise ValueError(f"{self.family_id}: root {root!r} is not a directory")
        planned = _planned_paths(ctx)
        found = sorted(str(p.relative_to(root).as_posix()) for p in pathlib.Path(root).rglob("*.parquet"))
        if planned is not None:
            have = set(found)
            missing = [p for p in planned if p not in have]
            if missing:
                raise ValueError(f"{self.family_id}: the plan names {len(missing)} part(s) not "
                                 f"under {root!r}, first {missing[0]!r}")
            found = planned
        if not found:
            raise RuntimeError(f"{self.family_id}: no .parquet part under {root!r}; "
                               "an empty walk is a wrong root, not an empty source")
        ctx.notes.update({"parts": len(found)})
        return [Unit(key=rel, url=(pathlib.Path(root) / rel).resolve().as_uri(),
                     meta={"path": str(pathlib.Path(root) / rel), "relative_path": rel})
                for rel in found]

    def acquire(self, unit: Unit, ctx: RunContext):
        size, digest = _sha256_file(unit.meta["path"])
        pointer = {"schema": POINTER_SCHEMA, "relative_path": unit.meta["relative_path"],
                   "bytes": size, "sha256": digest, "metadata_license": "CC0-1.0",
                   "snapshot": "https://docs.openalex.org"}
        body = (json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n").encode()
        return [Captured(unit, body, {"member": unit.key, "sha256": digest, "bytes": size})]

    def extract(self, cap: Captured, ctx: RunContext):
        from cytir.extract import openalex_academic as E

        rel, part_sha = cap.unit.meta["relative_path"], cap.meta["sha256"]
        docs, selected, refused = [], 0, 0
        for raw in E.iter_selected(cap.unit.meta["path"]):
            selected += 1
            try:
                row = E.render_record(raw, relative_path=rel, parent_sha256=part_sha)
            except ValueError:
                refused += 1        # the old script writes these to its recovery stage; counted here
                continue
            docs.append(Doc(
                url=row["url"], text=row["markdown"], title=row["title"], lang=row["lang"],
                extra={"document_id": row["document_id"], "part_sha256": part_sha,
                       "abstract": E.reconstruct_abstract(raw.get("abstract_inverted_index")),
                       "venue": E.venue_of(raw), "authors": E.author_names(raw),
                       "metadata": row["source_metadata"]}))
        ctx.notes.setdefault("per_part", {})[rel] = {
            "selected": selected, "rendered": len(docs), "recovery_required": refused}
        if not docs:
            raise UnitExcluded(NO_WORK_SELECTED)
        return docs

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch

        x = doc.extra
        out = []
        stamps = self._selection.stamps()
        digest = C.sha1(doc.text)
        for ordinal, unit in enumerate(_dispatch.chunk_markdown_channels(doc.text,
                                                                 selection=self._selection)):
            unit = dataclasses.replace(unit, stamps=stamps)
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(x["abstract"] if ordinal == 0 else unit.text, 25),
                lang=doc.lang, domain=DOMAIN, tier="C", source_type="reference",
                trust_tier=0.70, last_seen=ctx.today, content_digest=digest,
                license="CC0-1.0", license_version="1.0",
                attribution=x["venue"], anchor=f"chunk-{ordinal}",
                master={"authors": x["authors"], "document_id": x["document_id"],
                        "doi": x["metadata"]["doi"],
                        "underlying_work_license": x["metadata"]["underlying_work_license"],
                        "underlying_work_license_id": x["metadata"]["underlying_work_license_id"],
                        "publication_year": x["metadata"]["publication_year"],
                        "work_type": x["metadata"]["work_type"],
                        "source_part_sha256": x["part_sha256"]})
            out.append(C.passage_source_record(p))
        return out
