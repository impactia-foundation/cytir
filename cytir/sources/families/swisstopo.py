"""`named_slice:swisstopo-v2` -- swisstopo SwissNAMES3D gazetteer on the `SourceFamily` Protocol (task 1413).

The ingester of task 417 / S7 port order L2: the official Swiss gazetteer, one record per named
feature (name, feature type, language, LV95 coordinates), URL a map.geo.admin.ch pin. The parser
MOVED to `cytir.extract.swisstopo` (the old flat driver path forwards to it); this module wraps it:

  * `discover`: one unit per `*.csv.zip` under `--opt root=DIR`, as a `file:` URI. The STAC
    resolution and the download (the driver's `main()`: data.geo.admin.ch, httpx) are not adapter
    stages; the adapter reads archives already on disk, so a unit contacts no host. An empty walk
    is a wrong root, not an empty source.
  * `acquire`: the zip bytes, which the framework seals; meta carries their SHA-256 and size (the
    parent-archive identity every row quotes).
  * `extract`: `cytir.extract.swisstopo.rows` + `structured_record` over the three CSVs (PKT/LIN/PLY),
    one `Doc` per admitted feature, as a generator (the real archive holds ~500k rows); the
    record's own fields travel in `Doc.extra`. An archive with no admitted feature is EXCLUDED.

RELEASE IDENTITY lives in the STAC item, which the adapter does not call, so the run supplies it:
`--opt item_datetime=...` and `--opt archive_url=...` (required: a record without its release
datetime and archive link would be lossy), and `--opt item_id=...` (default: the zip name up to
`_2056`, e.g. `swissnames3d_2026`).

CHUNK: declared by the row (task 2260): `chunk-policy-atomic-record.v1`, one record = one passage;
`chunk` returns the record `extract` built, the way `statcan_wds` does. No `l1()`: the L1 seal of this family was written by a one-shot task script, not by this driver.

CONSENT AND LICENCE are read from the family's contract row, never written here. Consent is all
`unknown` (`search_ok`/`train_ok` null); the licence basis is `instrument` (swisstopo OGD terms).
`discover` copies the row's terms into the run notes and refuses nothing itself.

Options (`--opt KEY=VALUE`, or `--from DIR`): `root`, `item_datetime`, `archive_url` (required),
`item_id` (optional).
"""
from __future__ import annotations

import hashlib
import io
import pathlib
import zipfile

from cytir.extract import swisstopo as S
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

NO_ADMITTED_FEATURE = "no_admitted_feature"
REQUIRED_OPTIONS = ("root", "item_datetime", "archive_url")


def contract_terms(ctx: RunContext) -> dict:
    """The consent and licence terms of the family's contract row, as the run notes carry them."""
    row = ctx.contract
    if not row or "consent" not in row or "licence" not in row:
        raise ValueError("named_slice:swisstopo-v2: the run has no contract row with consent and licence")
    consent, licence = row["consent"], row["licence"]
    return {"search_ok": consent.get("search_ok"), "train_ok": consent.get("train_ok"),
            "consent_basis": consent.get("basis"), "consent_resolution": consent.get("resolution"),
            "licence_basis": licence.get("basis"), "licence_spdx": licence.get("spdx")}


def item_id_of(rel: str, options) -> str:
    """The STAC item id: the `item_id` option, else the zip name up to `_2056`."""
    return options.get("item_id") or pathlib.PurePosixPath(rel).name.split("_2056")[0].removesuffix(".csv.zip")


@register_family
class SwisstopoFamily:
    family_id = "named_slice:swisstopo-v2"
    route = "source_owned_bulk_dump"
    direct_crawl = False
    #: chunk-policy-atomic-record.v1 (task 2260): one record = one passage
    chunk_policy_id = CHUNK_DECLARED
    require_records = True     # a run that admits 0 records refuses (smoke-test-yield)

    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = contract_terms(ctx)
        missing = [k for k in REQUIRED_OPTIONS if not ctx.options.get(k)]
        if missing:
            raise ValueError("named_slice:swisstopo-v2 needs --opt " + ", --opt ".join(
                f"{k}=<...>" for k in missing) + " (root = directory holding the *.csv.zip)")
        root = pathlib.Path(ctx.options["root"])
        if not root.is_dir():
            raise ValueError(f"named_slice:swisstopo-v2: root {str(root)!r} is not a directory")
        found = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.csv.zip"))
        if not found:
            raise RuntimeError(f"named_slice:swisstopo-v2: no .csv.zip under {str(root)!r}; "
                               "an empty walk is a wrong root, not an empty source")
        ctx.notes["files"] = len(found)
        return [Unit(key=rel, url=(root / rel).resolve().as_uri(),
                     meta={"path": str(root / rel), "relative_path": rel,
                           "item_id": item_id_of(rel, ctx.options),
                           "item_datetime": ctx.options["item_datetime"],
                           "archive_url": ctx.options["archive_url"]})
                for rel in found]

    def acquire(self, unit: Unit, ctx: RunContext):
        body = pathlib.Path(unit.meta["path"]).read_bytes()
        return [Captured(unit, body, {"status": 200, "content_type": "application/zip",
                                      "source_url": unit.url, "member": unit.key,
                                      "sha256": hashlib.sha256(body).hexdigest(),
                                      "bytes": len(body)})]

    def extract(self, cap: Captured, ctx: RunContext):
        m = cap.unit.meta
        zf = zipfile.ZipFile(io.BytesIO(cap.body))
        n = 0
        for member, row in S.rows(zf):
            rec = S.structured_record(row, item_id=m["item_id"], item_datetime=m["item_datetime"],
                                      parent_member=member, parent_archive_sha256=cap.meta["sha256"],
                                      archive_url=m["archive_url"])
            if rec is None:
                continue
            n += 1
            yield Doc(url=rec["url"], text=rec["text"], title=rec["title"], lang=rec["lang"],
                      source_text=rec["text"], extra={"record": rec})
        if not n:
            raise UnitExcluded(NO_ADMITTED_FEATURE)

    def chunk(self, doc: Doc, ctx: RunContext):
        return [doc.extra["record"]]                          # atomic record: the record is the chunk
