"""`scratch_statcan_wds_exact` -- the preserved Statistics Canada WDS originals (task 1405, S7).

The row is a FROZEN BUILD, not a second live family. `statcan_wds` (the sibling adapter) fetches the
WDS catalogue and composes its locator cards; this row names the earlier "preserved" run: the operator
fetched `getAllCubesListLite` and `getCodeSets` with `curl`, kept each response as exact bytes plus
headers plus a transport receipt, and the driver sealed three stages from them (`original`,
`markdown`, `outlinks`) and wrote `RUN.json`. The driver's code now lives in
`cytir.sources.statcan.preserved` (MOVED, not copied; the old driver path is a forwarding shim).

STAGE MAP
  discover   one unit per endpoint (`file:` URI; no host is contacted)         PORTED
  acquire    validates the transport receipt, body and header digests          PORTED (delegates)
  seal       `seal_preserved`: the old `materialize` (3 stages + RUN.json)      PORTED (delegates)
  extract    the markdown rows are written by `seal_preserved`; the Protocol    PARTIAL
  chunk      stage is not wired (chunks are `pending_source_profile`)           PARTIAL

NO live route: the adapter never fetches. The `curl` loop stays in the operator shell driver.

CONSENT AND LICENCE are read from the contract row and copied into the run notes as they stand.
Every value of this row is `unknown` (a backfilled frozen build), so nothing here admits or refuses on
them: the adapter reads originals already on disk and fetches nothing. Consent governs the fetch.

Options (`--opt KEY=VALUE`): `inbox` (the directory holding `<endpoint>.json`, `.json.headers` and
`.json.transport.json` for both endpoints); for `seal_preserved` also `root`, `fetch_vantage`,
`fetched_at`, optional `supersedes_run`.
"""
from __future__ import annotations

import pathlib

from cytir.sources.family import CHUNK_DISPATCHER, Captured, Doc, RunContext, Unit
from cytir.sources.registry import register_family
from cytir.sources.statcan import preserved as P

ENDPOINTS = ("getAllCubesListLite", "getCodeSets")


def contract_terms(ctx: RunContext) -> dict:
    """The consent and licence terms of the contract row, exactly as declared (unknown stays unknown)."""
    row = ctx.contract
    if not row or "consent" not in row or "licence" not in row:
        raise ValueError("scratch_statcan_wds_exact: the run has no contract row with consent and licence")
    consent, licence = row["consent"], row["licence"]
    return {"search_ok": consent.get("search_ok"), "train_ok": consent.get("train_ok"),
            "consent_resolution": consent.get("resolution"),
            "licence_basis": licence.get("basis"), "licence_spdx": licence.get("spdx")}


def _inbox(ctx: RunContext) -> pathlib.Path:
    inbox = ctx.options.get("inbox")
    if not inbox:
        raise ValueError("scratch_statcan_wds_exact needs --opt inbox=<directory of preserved responses> "
                         "(or --from DIR)")
    path = pathlib.Path(inbox)
    if not path.is_dir():
        raise ValueError(f"scratch_statcan_wds_exact: inbox {inbox!r} is not a directory")
    return path.resolve()


def _paths(inbox: pathlib.Path, endpoint: str) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    body = inbox / f"{endpoint}.json"
    return body, inbox / f"{endpoint}.json.headers", inbox / f"{endpoint}.json.transport.json"


@register_family
class StatcanWdsPreservedFamily:
    family_id = "scratch_statcan_wds_exact"
    route = "documented_source_api"
    direct_crawl = True
    #: the row declares no chunk policy (`unknown`) and the run says chunks are `pending_source_profile`:
    #: the dispatcher is the only honest value; the chunk stage is PARTIAL and never exercised here
    chunk_policy_id = CHUNK_DISPATCHER
    require_records = True
    #: the sealing is `seal_preserved`; the Protocol extract/chunk stages are not wired
    PARTIAL = ("extract", "chunk")

    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = contract_terms(ctx)
        inbox = _inbox(ctx)
        units = []
        for endpoint in ENDPOINTS:
            for part in _paths(inbox, endpoint):
                if not part.is_file():
                    raise RuntimeError(f"scratch_statcan_wds_exact: missing {part}; "
                                       "a partial evidence set is a wrong inbox, not a shorter source")
            units.append(Unit(key=endpoint, url=_paths(inbox, endpoint)[0].as_uri(),
                              meta={"inbox": str(inbox), "endpoint": endpoint}))
        ctx.notes["endpoints"] = len(units)
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        endpoint = unit.meta["endpoint"]
        body, headers, metadata = _paths(pathlib.Path(unit.meta["inbox"]), endpoint)
        receipt = P.validate_transport(metadata, body, headers, f"{P.WDS}/{endpoint}")
        return [Captured(unit, body.read_bytes(), {
            "status": receipt["http_status"], "content_type": "application/json",
            "fetched_at": receipt["fetched_at"], "source_url": receipt["requested_url"]})]

    def extract(self, cap: Captured, ctx: RunContext) -> list[Doc]:
        raise NotImplementedError("scratch_statcan_wds_exact: the markdown rows are written by "
                                  "seal_preserved; the Protocol extract stage is not wired (PARTIAL)")

    def chunk(self, doc: Doc, ctx: RunContext):
        raise NotImplementedError("scratch_statcan_wds_exact: chunks are pending_source_profile "
                                  "(PARTIAL)")

    def seal_preserved(self, ctx: RunContext) -> dict:
        """Seal original + markdown + outlinks and `RUN.json`: the old driver's `materialize`."""
        contract_terms(ctx)
        inbox = _inbox(ctx)
        cubes, cubes_h, cubes_m = (p.resolve() for p in _paths(inbox, ENDPOINTS[0]))
        codes, codes_h, codes_m = (p.resolve() for p in _paths(inbox, ENDPOINTS[1]))
        supersedes = ctx.options.get("supersedes_run")
        return P.materialize(
            cubes_path=cubes, codesets_path=codes, cubes_headers_path=cubes_h,
            codesets_headers_path=codes_h, cubes_metadata_path=cubes_m, codesets_metadata_path=codes_m,
            root=pathlib.Path(ctx.options["root"]).resolve(),
            fetch_vantage=ctx.options["fetch_vantage"], fetched_at=ctx.options["fetched_at"],
            supersedes_run=pathlib.Path(supersedes).resolve() if supersedes else None)
