"""`justice_exact` and `justice_residual_exact` -- Justice Canada consolidated laws (task 1402).

Two rows of the contracts file, one source: the bilingual consolidated Acts and Regulations XML
repository `justicecanada/laws-lois-xml`, pinned to one Git commit. One private base class holds the
acquire stage; the two registered classes only name their row (a registry key is one id per class).

WHAT THIS SLICE COVERS: the snapshot acquirer, for both rows.
  * `discover`: one unit, the pinned archive (`file:` URI; the operator's `curl` of the codeload
    URL and the `git ls-remote` HEAD observation happen outside this adapter, so it contacts no host).
  * `acquire`: `cytir.sources.justice_canada.snapshot.validate` (HEAD evidence, HTTP transport, digests,
    member safety, XML present) and returns the archive bytes, which the framework seals.
  * `seal_snapshot`: the old driver's `seal` (stage receipt and `RUN.json`), from the same options.

  * `extract`, `chunk` (`justice_exact` ONLY, task 2431): `extract` opens the acquired archive and renders
    each `eng/acts/*.xml` member with `cytir.sources.justice_canada.render.render_official_markdown` (the very
    function the Route-C materializer calls; the Doc carries its frozen `PRODUCER` label); `chunk` is the
    central dispatcher under the row's declared policy (`cytir.sources.chunk_selection`).

WHAT STAYS OUT (PARTIAL), and why:
  * `extract`, `chunk` (`justice_residual_exact`): the residual row's URLs and parents come from its own
    materializer; that row stays as it was.
  * `materialize` (`justice_residual_exact` only): the framework has no materialize hook, so the row
    keeps "materialize" in `PARTIAL`. The code is ported: `materialize_residual(ctx)` delegates to
    `cytir.sources.justice_canada.residual.materialize` (options `artifact_root`, `identity_run`, `output`).

CONSENT AND LICENCE are read from the contract row, never written here. The rows still declare
`unknown` for both; this adapter does NOT refuse on `unknown` (orchestrator ruling 60(c): consent
governs FETCH, not the sealing of files already held; the unit is a `file:` URI with no consent host,
and the framework, task 1406, will refuse host-contacting acquire on such rows) and copies the row's values into `notes.contract_terms`
so the manifest carries the terms the run was admitted under, `unknown` included.

Options (`--opt KEY=VALUE`, or `--from DIR`): `root` (the run directory that holds `RUN.json`),
`artifact_root`, `input` (the directory with `snapshot.tar.gz` and the evidence files, named as in
the operator driver), `commit`, `requested_url`.
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import io
import json
import pathlib
import tarfile
import urllib.parse
import xml.etree.ElementTree as ET

from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit
from cytir.sources.justice_canada import snapshot as snap
from cytir.sources.registry import register_family

#: evidence file names in `input`, as the operator's acquisition shell driver writes them
EVIDENCE = {
    "archive": "snapshot.tar.gz", "headers": "snapshot.headers", "curl_metrics": "snapshot.curl.json",
    "transport": "snapshot.transport.json", "ls_remote": "git-ls-remote.txt",
    "ls_remote_stderr": "git-ls-remote.stderr", "ls_remote_started": "git-ls-remote.started",
    "ls_remote_completed": "git-ls-remote.completed", "ls_remote_raw": "git-ls-remote.raw",
    "ls_remote_rc": "git-ls-remote.rc", "fetch_started": "fetch.started",
    "fetch_completed": "fetch.completed", "fetch_rc": "fetch.rc",
    "curl_stderr": "snapshot.curl.stderr",
}


def contract_terms(ctx: RunContext) -> dict:
    """The consent and licence terms of the family's row, as the run notes carry them (unknown stays unknown)."""
    row = ctx.contract
    if not row or "consent" not in row or "licence" not in row:
        raise ValueError("justice_canada: the run has no contract row with consent and licence")
    consent, licence = row["consent"], row["licence"]
    return {"family_id": row.get("family_id"), "search_ok": consent.get("search_ok"),
            "train_ok": consent.get("train_ok"), "consent_resolution": consent.get("resolution"),
            "licence_basis": licence.get("basis"), "licence_spdx": licence.get("spdx")}


def _kwargs(ctx: RunContext) -> dict:
    o = ctx.options
    for need in ("root", "artifact_root", "input", "commit", "requested_url"):
        if not o.get(need):
            raise ValueError(f"justice_canada needs --opt {need}=... (see the module docstring)")
    inp = pathlib.Path(o["input"])
    if not inp.is_dir():
        raise ValueError(f"justice_canada: input {str(inp)!r} is not a directory")
    out = {k: inp / name for k, name in EVIDENCE.items()}
    out.update(root=pathlib.Path(o["root"]), artifact_root=pathlib.Path(o["artifact_root"]),
               commit=o["commit"], requested_url=o["requested_url"])
    return out


class _JusticeCanadaBase:
    route = "source_owned_bulk_dump"
    direct_crawl = False
    chunk_policy_id = CHUNK_DECLARED
    require_records = True
    #: both rows stop after the snapshot acquirer (the residual row adds `materialize`)
    PARTIAL = ("extract", "chunk")

    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = contract_terms(ctx)
        kw = _kwargs(ctx)
        archive = kw["archive"]
        if not archive.is_file():
            raise RuntimeError(f"{self.family_id}: no snapshot archive at {archive}; "
                               "an empty walk is a wrong root, not an empty source")
        ctx.notes.update({"commit": kw["commit"], "requested_url": kw["requested_url"],
                          "repository": snap.REPO})
        return [Unit(key=f"snapshot-{kw['commit']}", url=archive.resolve().as_uri(),
                     meta={"commit": kw["commit"]})]

    def acquire(self, unit: Unit, ctx: RunContext):
        kw = _kwargs(ctx)
        found = snap.validate(**kw)
        ctx.notes.update({"archive_members": found["members"], "xml_members": found["xml_members"],
                          "observed_language_paths": sorted(found["languages"])})
        return [Captured(unit, kw["archive"].read_bytes(), {
            "status": 200, "content_type": "application/x-gzip", "source_url": kw["requested_url"],
            "commit": kw["commit"], "xml_members": found["xml_members"]})]

    def extract(self, cap: Captured, ctx: RunContext) -> list[Doc]:
        raise NotImplementedError(f"{self.family_id}: the XML -> markdown stage is a later slice (PARTIAL)")

    def chunk(self, doc: Doc, ctx: RunContext):
        raise NotImplementedError(f"{self.family_id}: the chunk stage is a later slice (PARTIAL)")

    def seal_snapshot(self, ctx: RunContext) -> dict:
        """Seal the pinned snapshot: the stage receipt and `RUN.json` (the old driver's `seal`)."""
        contract_terms(ctx)
        return snap.seal(**_kwargs(ctx))


ACT_BASE = "https://laws-lois.justice.gc.ca/eng/acts/"


@register_family
class JusticeExactFamily(_JusticeCanadaBase):
    family_id = "justice_exact"
    PARTIAL = ()

    def preflight(self, ctx: RunContext) -> None:
        from cytir.sources.chunk_selection import declared_policy_selection
        self._selection = declared_policy_selection(ctx)

    def extract(self, cap: Captured, ctx: RunContext) -> list[Doc]:
        """One Doc per English Act member of the archive, rendered exactly as the Route-C materializer
        renders it. Option `routes` (a gzip JSONL of route rows) restricts the Acts to the
        `justice-canada-official-xml` route's URLs, as the materializer does; absent = every Act member."""
        from cytir.sources.justice_canada import render as R
        want = None
        if ctx.options.get("routes"):
            with gzip.open(ctx.options["routes"], "rt", encoding="utf-8") as fh:
                rows = (json.loads(line) for line in fh)
                want = {r["url"] for r in rows if r.get("route_id") == R.ROUTE_ID}
        docs = []
        with tarfile.open(fileobj=io.BytesIO(cap.body), mode="r:gz") as tar:
            for m in tar.getmembers():
                parts = m.name.split("/")
                if not (m.isfile() and len(parts) == 4 and parts[1:3] == ["eng", "acts"]
                        and parts[3].endswith(".xml")):
                    continue
                url = ACT_BASE + urllib.parse.quote(parts[3][:-4]) + "/"
                if want is not None and url not in want:
                    continue
                xml_bytes = tar.extractfile(m).read()
                sha = hashlib.sha256(xml_bytes).hexdigest()
                root = ET.fromstring(xml_bytes.lstrip(b"\xef\xbb\xbf"))
                title, md, version = R.render_official_markdown(root, url=url, member=m.name, member_sha256=sha)
                docs.append(Doc(url=url, text=md, title=title, lang="en", source_text=md, extra={
                    "document_id": hashlib.sha256(f"justice-canada-law\0{url}".encode()).hexdigest(),
                    "source_profile": "anchored_law", "producer": R.PRODUCER, "version": R.VERSION,
                    "source_record_sha256": sha, "source_version": version}))
        docs.sort(key=lambda d: d.url)
        return docs

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.chunk import dispatch as _dispatch
        from cytir.core import common as C
        selection = getattr(self, "_selection", None)
        if selection is None:
            raise RuntimeError(f"{self.family_id}.chunk ran before preflight: the declared chunk policy "
                               "is resolved in preflight; call preflight(ctx) first")
        out, stamps, digest = [], selection.stamps(), C.sha1(doc.text)
        for ordinal, unit in enumerate(_dispatch.chunk_markdown_channels(doc.text, selection=selection)):
            unit = dataclasses.replace(unit, stamps=stamps)
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(unit.dense_text or unit.text, 25),
                lang=doc.lang, domain=urllib.parse.urlsplit(doc.url).hostname or "", tier=None,
                source_type=None, trust_tier=None, last_seen=ctx.today, content_digest=digest,
                anchor=f"document:{doc.extra['document_id']}:chunk-{ordinal}")
            rec = C.passage_source_record(p)
            rec["document_id"], rec["chunk_ordinal"] = doc.extra["document_id"], ordinal
            out.append(rec)
        return out


@register_family
class JusticeResidualExactFamily(_JusticeCanadaBase):
    family_id = "justice_residual_exact"
    PARTIAL = ("materialize", "extract", "chunk")

    def materialize_residual(self, ctx: RunContext) -> dict:
        """Run the residual materializer (the old pipeline driver's `materialize`), from options."""
        from cytir.sources.justice_canada import residual
        o = ctx.options
        for need in ("artifact_root", "identity_run", "output"):
            if not o.get(need):
                raise ValueError(f"justice_residual_exact needs --opt {need}=... (see the module docstring)")
        return residual.materialize(artifact_root=pathlib.Path(o["artifact_root"]),
                                    identity_run_path=pathlib.Path(o["identity_run"]),
                                    output=pathlib.Path(o["output"]))
