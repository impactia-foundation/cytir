"""`named_slice:bfs-pxweb` -- BFS PX-Web table metadata on the `SourceFamily` Protocol (task 1412).

The ingester of task 417 / S7 port order L2: the Swiss Federal Statistical Office PX-Web (STAT-TAB)
API, one record per (table, language), the METADATA response `GET /api/v1/{lang}/{dbid}/{dbid}.px`
(title + dimensions + value labels; never the data matrix). The parse functions MOVED to
`cytir.extract.bfs_pxweb` (the old flat driver path forwards to it); this module only wraps them:

  * `discover`: one unit per entry of `catalog.jsonl` under `--opt root=DIR`, the layout of the
    sealed exact-response (L0) preservation: `{url, retrieved_at, response: {uri, bytes, sha256}}`
    and the response bodies as files next to it. The unit URL is the API URL the entry names, but
    the unit is read from disk: the live walk (the driver's `list_tables` / `get_json`, with its
    retry and politeness) stays in `cytir.extract.bfs_pxweb.main` and is not an adapter stage, so a
    run contacts no host. An empty catalog is a wrong root, not an empty source.
  * `acquire`: the response bytes of the entry, checked against the catalog's byte count and
    SHA-256, which the framework then seals.
  * `extract`: `structured_record` (the structured-record Markdown of the sealed L1), one `Doc` per
    response; a response with no title or no variables is EXCLUDED, not an error. The row's own
    fields travel in `Doc.extra`.

CHUNK: declared by the row (task 2260): `chunk-policy-atomic-record.v1`, one record = one passage;
`chunk` returns the record `extract` built, the way `statcan_wds` does. No `l1()`: the L1 seal of this family was
written by a one-shot task script, not by this driver.

CONSENT AND LICENCE are read from the family's contract row, never written here. The row is all
`unknown` (`search_ok`/`train_ok` null): `discover` copies the row's terms into the run notes and
refuses nothing itself; the framework's own consent step governs.

Options (`--opt KEY=VALUE`, or `--from DIR`): `root` (required; directory holding `catalog.jsonl`).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from urllib.parse import urlsplit

from cytir.extract import bfs_pxweb as B
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

NO_TITLE_OR_VARIABLES = "no_title_or_variables"
CATALOG = "catalog.jsonl"


def contract_terms(ctx: RunContext) -> dict:
    """The consent and licence terms of the family's contract row, as the run notes carry them."""
    row = ctx.contract
    if not row or "consent" not in row or "licence" not in row:
        raise ValueError("named_slice:bfs-pxweb: the run has no contract row with consent and licence")
    consent, licence = row["consent"], row["licence"]
    return {"search_ok": consent.get("search_ok"), "train_ok": consent.get("train_ok"),
            "consent_basis": consent.get("basis"), "consent_resolution": consent.get("resolution"),
            "licence_basis": licence.get("basis"), "licence_spdx": licence.get("spdx")}


def identity(url: str) -> tuple[str, str]:
    """(lang, dbid) of a PX-Web metadata URL `.../api/v1/{lang}/{dbid}/{dbid}.px`."""
    parts = [p for p in urlsplit(url).path.split("/") if p]
    if len(parts) < 5 or parts[-1] != f"{parts[-2]}.px":
        raise ValueError(f"unexpected PXWeb metadata URL: {url}")
    return parts[-3], parts[-2]


@register_family
class BfsPxwebFamily:
    family_id = "named_slice:bfs-pxweb"
    route = "api"
    direct_crawl = False
    #: chunk-policy-atomic-record.v1 (task 2260): one record = one passage
    chunk_policy_id = CHUNK_DECLARED
    require_records = True     # a run that admits 0 records refuses (smoke-test-yield)

    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = contract_terms(ctx)
        root = ctx.options.get("root")
        if not root:
            raise ValueError("named_slice:bfs-pxweb needs --opt root=<directory holding "
                             f"{CATALOG} and its response files> (or --from DIR)")
        base = pathlib.Path(root)
        catalog = base / CATALOG
        if not catalog.is_file():
            raise RuntimeError(f"named_slice:bfs-pxweb: no {CATALOG} under {root!r}")
        units = []
        for line in catalog.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            lang, dbid = identity(entry["url"])
            ref = entry["response"]
            rel = pathlib.PurePosixPath(ref["uri"].removeprefix("artifact://"))
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe response artifact URI: {ref['uri']}")
            units.append(Unit(key=f"{lang}:{dbid}", url=entry["url"],
                              meta={"path": str(base / rel), "api_url": entry["url"],
                                    "lang": lang, "dbid": dbid, "response": ref,
                                    "retrieved_at": entry["retrieved_at"]}))
        if not units:
            raise RuntimeError(f"named_slice:bfs-pxweb: {CATALOG} under {root!r} lists no response; "
                               "an empty walk is a wrong root, not an empty source")
        ctx.notes["responses"] = len(units)
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        body = pathlib.Path(unit.meta["path"]).read_bytes()
        ref = unit.meta["response"]
        digest = hashlib.sha256(body).hexdigest()
        if len(body) != ref["bytes"] or digest != ref["sha256"]:
            raise ValueError(f"BFS response differs from its catalog entry: {unit.key}")
        return [Captured(unit, body, {"status": 200, "content_type": "application/json",
                                      "source_url": unit.meta["api_url"], "member": ref["uri"],
                                      "sha256": digest, "bytes": len(body),
                                      "retrieved_at": unit.meta["retrieved_at"]})]

    def extract(self, cap: Captured, ctx: RunContext):
        m = cap.unit.meta
        row = B.structured_record(m["dbid"], m["lang"], json.loads(cap.body),
                                  response_uri=cap.meta["member"],
                                  response_sha256=cap.meta["sha256"],
                                  retrieved_at=cap.meta["retrieved_at"])
        if row is None:
            raise UnitExcluded(NO_TITLE_OR_VARIABLES)
        return [Doc(url=row["url"], text=row["text"], title=row["title"], lang=row["lang"],
                    source_text=row["text"], extra={"row": row})]

    def chunk(self, doc: Doc, ctx: RunContext):
        return [doc.extra["row"]]                          # atomic record: the record is the chunk
