"""`named_slice:pubmed` -- NCBI PubMed baseline abstracts on the `SourceFamily` Protocol (task 1404).

The ingester of task 417 / S7 port order L2: the NLM MEDLINE/PubMed baseline (`pubmed<yy>n<NNNN>.xml.gz`),
one passage per abstract, URL `https://pubmed.ncbi.nlm.nih.gov/<pmid>/`. The parser MOVED to
`cytir.extract.pubmed` (the old flat driver path forwards to it); this module only wraps it:

  * `discover`: one unit per `*.xml.gz` baseline file under `--opt root=DIR`, as a `file:` URI. The
    download (the driver's `main()`, httpx against the NCBI FTP mirror) is not an adapter stage:
    the adapter reads files already on disk, so a unit contacts no host. An empty walk is a wrong
    root, not an empty source.
  * `acquire`: the gzip bytes of the file, which the framework seals; meta carries their SHA-256
    and size (the parent-archive identity the rows quote).
  * `extract`: `cytir.extract.pubmed.iter_pubmed_articles` over the decompressed XML, one `Doc` per
    admitted abstract; the row's own fields travel in `Doc.extra`. A file with no admitted
    abstract is EXCLUDED, not an error.

CHUNK: declared by the row (task 2260): `chunk-policy-atomic-record.v1`, one record = one passage;
`chunk` returns the record `extract` built, the way `statcan_wds` does. No `l1()`: the L1 seal of this family was written by
one-shot task scripts, not by this driver.

CONSENT AND LICENCE are read from the family's contract row, never written here. The row is all
`unknown` (`search_ok`/`train_ok` null): `discover` copies the row's terms into the run notes and
refuses nothing itself; the framework's own consent step governs.

Options (`--opt KEY=VALUE`, or `--from DIR`): `root` (required; directory holding `*.xml.gz`).
"""
from __future__ import annotations

import gzip
import hashlib
import pathlib

from cytir.extract import pubmed as P
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

NO_ADMITTED_ABSTRACT = "no_admitted_abstract"


def contract_terms(ctx: RunContext) -> dict:
    """The consent and licence terms of the family's contract row, as the run notes carry them."""
    row = ctx.contract
    if not row or "consent" not in row or "licence" not in row:
        raise ValueError("named_slice:pubmed: the run has no contract row with consent and licence")
    consent, licence = row["consent"], row["licence"]
    return {"search_ok": consent.get("search_ok"), "train_ok": consent.get("train_ok"),
            "consent_basis": consent.get("basis"), "consent_resolution": consent.get("resolution"),
            "licence_basis": licence.get("basis"), "licence_spdx": licence.get("spdx")}


@register_family
class PubmedFamily:
    family_id = "named_slice:pubmed"
    route = "source_owned_bulk_dump"
    direct_crawl = False
    #: chunk-policy-atomic-record.v1 (task 2260): one record = one passage
    chunk_policy_id = CHUNK_DECLARED
    require_records = True     # a run that admits 0 passages refuses (smoke-test-yield)

    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = contract_terms(ctx)
        root = ctx.options.get("root")
        if not root:
            raise ValueError("named_slice:pubmed needs --opt root=<directory holding the baseline "
                             "*.xml.gz files> (or --from DIR)")
        if not pathlib.Path(root).is_dir():
            raise ValueError(f"named_slice:pubmed: root {root!r} is not a directory")
        found = sorted(p.relative_to(root).as_posix() for p in pathlib.Path(root).rglob("*.xml.gz"))
        if not found:
            raise RuntimeError(f"named_slice:pubmed: no .xml.gz file under {root!r}; "
                               "an empty walk is a wrong root, not an empty source")
        ctx.notes["files"] = len(found)
        return [Unit(key=rel, url=(pathlib.Path(root) / rel).resolve().as_uri(),
                     meta={"path": str(pathlib.Path(root) / rel), "relative_path": rel})
                for rel in found]

    def acquire(self, unit: Unit, ctx: RunContext):
        body = pathlib.Path(unit.meta["path"]).read_bytes()
        return [Captured(unit, body, {"status": 200, "content_type": "application/gzip",
                                      "source_url": unit.url, "member": unit.key,
                                      "sha256": hashlib.sha256(body).hexdigest(),
                                      "bytes": len(body)})]

    def extract(self, cap: Captured, ctx: RunContext):
        archive = {"member": cap.meta["member"], "sha256": cap.meta["sha256"]}
        docs = []
        for row in P.iter_pubmed_articles(gzip.decompress(cap.body), source_archive=archive):
            docs.append(Doc(url=row["url"], text=row["text"], title=row["title"], lang=row["lang"],
                            source_text=row["text"], extra={"row": row}))
        if not docs:
            raise UnitExcluded(NO_ADMITTED_ABSTRACT)
        return docs

    def chunk(self, doc: Doc, ctx: RunContext):
        return [doc.extra["row"]]                          # atomic record: the record is the chunk
