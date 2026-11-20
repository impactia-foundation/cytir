"""`uk_legislation` and `it_senato_gazzetta` -- official bulk-dump seals on the Protocol (task 1394).

Both families are class A, route `bulk-dump`, licence from the official-dump instrument. Their
shared acquirer is the old `acquire_official_bulk_source[_direct]` driver pair, now
`cytir.sources.official_bulk.{bulk,direct}` (MOVED, not copied; the flat paths are forwarding shims).

WHAT THIS ADAPTER COVERS: the `original` stage of one bulk response that the operator already
fetched with `curl` (+ `cyt source http-evidence`): body, response headers and transport metadata on disk.

  * `discover`: one unit, the response (a `file:` URI; the adapter contacts no host).
  * `acquire`: validates the transport metadata against the files
    (`official_bulk.{bulk,direct}.check_transport`) and returns the metadata record as the
    captured body. The bulk body itself (a 1.71 GB Senato zip, a UK XML dump) is NOT returned:
    returning it would make the framework hold and re-seal a second copy, which the `direct`
    driver exists to avoid.
  * `seal_original`: the stage seal and `RUN.json` (the old driver's `acquire`), run from the same
    options. `direct=1` (or an `artifact_root`) selects the seal-in-place driver.

WHAT STAYS OUT (`PARTIAL = extract, chunk`): the projection of the dump to markdown (UK section
Akoma Ntoso, Senato article Akoma Ntoso) and the chunk stage. The sealed RUN says
`source_acquired_pending_projection` and blocks index admission until those gates seal; this
adapter does not claim them. The Gazzetta ELI direct-fetch leg is gated in the contract on an
adapter smoke test and is not ported either.

CONSENT AND LICENCE are read from the family's contract row, never written here: `discover`
refuses a row that does not declare `search_ok`, copies the row's consent resolution and licence
basis into the run notes, and the direct seal's `license` field is the row's `licence.spdx` (the
official-dump instrument), not an option.

Options (`--opt KEY=VALUE`): `body`, `headers`, `metadata` (paths), `requested_url`, `root`
(RUN.json directory), `source_id`, `source_name`, `content_scope`, `next_projection`,
`fetch_vantage`; plain seal also `country`; `direct=1` also `artifact_root`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import re
from collections.abc import Sequence

from cytir.sources.chunk_selection import declared_policy_selection
from cytir.sources.family import CHUNK_DECLARED, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.official_bulk import bulk, direct
from cytir.sources import staged_text_guard
from cytir.sources.registry import register_family

_PATHS = ("body", "headers", "metadata")
_SEAL_COMMON = ("root", "source_id", "source_name", "requested_url", "content_scope",
                "next_projection", "fetch_vantage")


def contract_terms(ctx: RunContext, family_id: str) -> dict:
    """The consent and licence terms of the family's contract row, as the run notes carry them."""
    row = ctx.contract
    if not row or "consent" not in row or "licence" not in row:
        raise ValueError(f"{family_id}: the run has no contract row with consent and licence")
    consent, licence = row["consent"], row["licence"]
    state = consent.get("consent_state")
    if state == "unreadable" or (row.get("acquisition") or {}).get("parked"):
        # consent governs fetch, never retention: a parked family does not discover or acquire
        raise ValueError(f"{family_id}: the family is parked, consent_state={state!r}; consent "
                         "governs fetch, so no discover or acquire until the contract unparks it")
    if consent.get("search_ok") is not True:
        raise ValueError(f"{family_id}: the contract does not declare search_ok; the family "
                         "does not run")
    return {"search_ok": consent["search_ok"], "train_ok": consent.get("train_ok"),
            "consent_resolution": consent.get("resolution"),
            "licence_basis": licence.get("basis"), "licence_spdx": licence.get("spdx")}


def _need(ctx: RunContext, family_id: str, names) -> dict:
    missing = [n for n in names if not ctx.options.get(n)]
    if missing:
        raise ValueError(f"{family_id} needs --opt " + ", --opt ".join(missing))
    return {n: ctx.options[n] for n in names}


def _paths(ctx: RunContext, family_id: str) -> dict:
    got = _need(ctx, family_id, _PATHS + ("requested_url",))
    paths = {n: pathlib.Path(got[n]).resolve() for n in _PATHS}
    for n, p in paths.items():
        if not p.is_file():
            raise ValueError(f"{family_id}: {n} {str(p)!r} is not a file")
    return {**paths, "requested_url": got["requested_url"]}


def _is_direct(ctx: RunContext) -> bool:
    return str(ctx.options.get("direct", "")).lower() in ("1", "true", "yes") or bool(
        ctx.options.get("artifact_root"))


class _OfficialBulkFamily:
    """Shared stages; a subclass names only its `family_id`."""
    route = "bulk-dump"
    direct_crawl = True
    chunk_policy_id = CHUNK_DECLARED
    require_records = True
    #: the dump -> markdown projection and the chunk stage stay unported
    PARTIAL = ("extract", "chunk")

    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = contract_terms(ctx, self.family_id)
        p = _paths(ctx, self.family_id)
        _need(ctx, self.family_id, _SEAL_COMMON)
        return [Unit(key="bulk-original", url=p["metadata"].as_uri(),
                     meta={**{k: str(v) for k, v in p.items()}, "direct": _is_direct(ctx)})]

    def acquire(self, unit: Unit, ctx: RunContext):
        m = unit.meta
        check = direct.check_transport if m["direct"] else bulk.check_transport
        transport = check(body=pathlib.Path(m["body"]), headers=pathlib.Path(m["headers"]),
                          metadata=pathlib.Path(m["metadata"]), requested_url=m["requested_url"])
        return [Captured(unit, pathlib.Path(m["metadata"]).read_bytes(), {
            "status": transport["http_status"], "content_type": "application/json",
            "fetched_at": transport.get("fetched_at"), "source_url": transport["requested_url"],
            "final_url": transport["final_url"],
            "bulk_body_sha256": (transport.get("body") or {}).get("sha256"),
            "bulk_body_bytes": (transport.get("body") or {}).get("bytes")})]

    def extract(self, cap: Captured, ctx: RunContext):
        raise NotImplementedError(f"{self.family_id}: the dump -> markdown projection is a later "
                                  "slice (PARTIAL)")

    def chunk(self, doc, ctx: RunContext):
        raise NotImplementedError(f"{self.family_id}: the chunk stage is a later slice (PARTIAL)")

    def seal_original(self, ctx: RunContext) -> dict:
        """Seal the `original` stage and write `RUN.json` (the old driver's `acquire`)."""
        terms = contract_terms(ctx, self.family_id)
        p = _paths(ctx, self.family_id)
        o = _need(ctx, self.family_id, _SEAL_COMMON)
        common = dict(body=p["body"], headers=p["headers"], metadata=p["metadata"],
                      root=pathlib.Path(o["root"]).resolve(), source_id=o["source_id"],
                      source_name=o["source_name"], requested_url=o["requested_url"],
                      content_scope=o["content_scope"], next_projection=o["next_projection"],
                      fetch_vantage=o["fetch_vantage"])
        if _is_direct(ctx):
            ar = _need(ctx, self.family_id, ("artifact_root",))["artifact_root"]
            return direct.acquire(artifact_root=pathlib.Path(ar).resolve(),
                                  license_name=terms["licence_spdx"], **common)
        return bulk.acquire(country=_need(ctx, self.family_id, ("country",))["country"], **common)


def _guard_words(body: str) -> tuple[str, tuple[int, ...]]:
    """The provision words the staged-text guard compares, and the word offsets where this adapter
    REMOVED something from the source.

    Two things are removed: markdown sub-heading lines (the structured chunker restates each as a
    `Title > Label > Heading` breadcrumb: declared, not a loss) and bare dot-runs (repeal markers).
    Each removal splices two words into a new 5-gram that no staged row holds. The offset of every
    removal is returned as a cut, so the guard forgives exactly those 5-grams
    (`staged_text_guard.SourcePart`) and nothing else. The cuts come from the source alone."""
    words: list[str] = []
    cuts: list[int] = []

    def cut() -> None:
        if words and (not cuts or cuts[-1] != len(words)):
            cuts.append(len(words))

    for ln in body.splitlines():
        if ln.lstrip().startswith("#"):
            cut()
            continue
        for w in ln.split():
            if w.strip("."):
                words.append(w)
            else:
                cut()
    return " ".join(words), tuple(c for c in cuts if c < len(words))


_LETTERED_RUN = re.compile(r"[^\W\d_]{2,}")


def _row_cuts(words: list[str], row_texts: Sequence[str]) -> tuple[int, ...]:
    """The word offsets where the chunker's own windows start inside the guard words.

    The structured chunker splits a long section into rows and starts every row after the first
    with a `Title > Label > Heading` breadcrumb paragraph. A split costs exactly the 5-grams that
    straddle it (they sit in no row), so each split is declared as a cut. Offsets are found by
    reading the rows the chunker made from the SOURCE text, in order: a row's body is its text minus
    the breadcrumb paragraph, and its start is the first place at or after the previous start where
    the body's first words (up to five) occur in the guard words. A row that is not found (the lead
    title row holds no provision words) declares no cut, so a real loss stays a loss."""
    crumbs = {t.split("\n\n", 1)[0] for t in row_texts if " > " in t.split("\n\n", 1)[0]}
    pieces: list[list[str]] = []
    splitter = re.compile("|".join(re.escape(c) for c in sorted(crumbs, key=len, reverse=True))) if crumbs else None
    for text in row_texts:
        # the breadcrumb opens a row, or re-opens a window inside one (glued to the text before it)
        for chunk in (splitter.split(text) if splitter else [text]):
            piece = [w for w in chunk.split() if w.strip(".")]   # dot-runs are not guard words
            if piece:
                pieces.append(piece)
    cuts: list[int] = []
    prev, expected = -1, 0
    for body in pieces:
        head = body[:8]
        # rows are verbatim runs of the source in order, so a row starts about where the last one
        # ended: take the occurrence of its first words nearest to that (a table repeats phrases)
        hits = [i for i in range(prev + 1, len(words) - len(head) + 1) if words[i:i + len(head)] == head]
        if not hits:
            continue
        start = min(hits, key=lambda i: abs(i - expected))
        if start > 0 and start not in cuts:
            cuts.append(start)
        prev, expected = start, start + len(body)
    return tuple(cuts)


def _guard_parts(doc_id: str, body: str, row_texts: Sequence[str] = ()):
    """The guard's SourcePart list for one section, or None when the body is structurally empty.

    A body is checked only if it holds a lettered word (a run of two or more letters). Numbers,
    labels such as "(2)", "(a)" or "18J", and punctuation are structure, not provision text: a body
    made only of them has nothing to compare, so the manifest counts the doc UNCHECKED, never
    checked-and-passed. `row_texts` are the rows the chunker makes from the section: their split
    points are declared cuts too (`_row_cuts`)."""
    src, cuts = _guard_words(body)
    if not _LETTERED_RUN.search(src):
        return None
    split = _row_cuts(src.split(), row_texts)
    return [staged_text_guard.SourcePart(doc_id, src, tuple(sorted(set(cuts) | set(split))))]


@register_family
class UkLegislationFamily(_OfficialBulkFamily):
    """UK legislation: the AKN item -> section-bounded markdown -> passages (task 2181).

    `extract` takes one `/{uri}/data.akn` body (`Captured.body`) and returns one `Doc` per citable
    provision (section / article / regulation / schedule, plus the introductory text), titled and
    cited. `chunk` runs the declared contract policy over that one provision: a provision under the
    window is one passage; a longer one is split by the policy, the title and citation on the
    provision's first chunk. The sealed bulk-response record (a JSON metadata body, not an item) is
    excluded with a reason, never parsed.
    """
    family_id = "uk_legislation"
    PARTIAL = ()
    #: the guard floor: every provision word must survive (1.0 family norm, as fedlex/loda/dila_kali)
    staged_text_min_coverage = 1.0
    #: the row leaves `tokenizer_revision` null by design (its 40-hex string is stated once, on this
    #: sibling row, because the secret scanner flags a bare 40-hex string): the adapter reads it there
    _REVISION_ROW = "de_openlegaldata_v1"

    #: the only host a local AKN file may name as its work URI
    _AKN_HOST = "www.legislation.gov.uk"

    def discover(self, ctx: RunContext):
        """`--opt akn_dir=<dir>`: one unit per `*.akn` / `*.xml` file, in sorted file-name order.

        The item identity (unit key, `source_url`) is the document's own FRBR work URI, never the
        file name; the unit's url is the metadata URI (`https://www.legislation.gov.uk<path>`), never a local path,
        and no host is contacted (the file is read from disk). A file with no
        readable work URI is a unit keyed by its content hash (extract excludes it with the
        parser's reason); a work URI on any other host, or with userinfo, is excluded at acquire.
        Without `akn_dir` the family seals the one bulk response as before."""
        akn_dir = ctx.options.get("akn_dir")
        if not akn_dir:
            return super().discover(ctx)
        from urllib.parse import urlsplit
        from cytir.extract import uk_legislation_akn as UK
        ctx.notes["contract_terms"] = contract_terms(ctx, self.family_id)
        root = pathlib.Path(akn_dir).resolve()
        if not root.is_dir():
            raise ValueError(f"{self.family_id}: akn_dir {str(root)!r} is not a directory")
        files = sorted(f for f in root.iterdir()
                       if not f.is_symlink() and f.is_file() and f.suffix in (".akn", ".xml"))
        if not files:
            raise ValueError(f"{self.family_id}: akn_dir {str(root)!r} holds no *.akn / *.xml file")
        units = []
        for f in files:
            body = f.read_bytes()
            sha = hashlib.sha256(body).hexdigest()
            meta = {"path": str(f), "source_file_sha256": sha, "route": self.route, "refusal": None}
            try:
                uri = UK.work_uri(body)
            except UK.NotAkn:
                units.append(Unit(key=f"unreadable-{sha[:16]}", url=f"urn:sha256:{sha}", meta=meta))
                continue
            parts = urlsplit(uri)
            try:
                port = parts.port
            except ValueError:          # a malformed port (e.g. ":abc") refuses this unit, not the directory
                port = "malformed"
            if parts.hostname != self._AKN_HOST or parts.username or parts.password or port:
                meta["refusal"] = "work_uri_host_not_legislation_gov_uk"
                units.append(Unit(key=f"refused-{sha[:16]}", url=f"urn:sha256:{sha}", meta=meta))
                continue
            meta["source_url"] = f"https://{self._AKN_HOST}{parts.path}"
            units.append(Unit(key=parts.path.strip("/"), url=meta["source_url"], meta=meta))
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        if "source_file_sha256" not in unit.meta:
            return super().acquire(unit, ctx)
        if unit.meta.get("refusal"):
            raise UnitExcluded(unit.meta["refusal"])
        body = pathlib.Path(unit.meta["path"]).read_bytes()
        if hashlib.sha256(body).hexdigest() != unit.meta["source_file_sha256"]:
            raise RuntimeError(f"{unit.key}: the source file changed between discover and acquire")
        return [Captured(unit, body, {
            "route": self.route, "status": 200, "content_type": "application/akn+xml",
            "source_file_sha256": unit.meta["source_file_sha256"],
            "source_url": unit.meta.get("source_url")})]

    @staticmethod
    def _declared_license() -> str:
        """The licence the host table declares for legislation.gov.uk (ADR-039); never written here."""
        from cytir.core import license as LIC
        lic = LIC.resolve_license(None, "legislation.gov.uk", "www.legislation.gov.uk", "gov-legal", None)
        if not lic:
            raise RuntimeError("cytir.core.license has no licence for www.legislation.gov.uk; "
                               "uk_legislation stamps none and does not run")
        return lic

    def preflight(self, ctx: RunContext) -> None:
        cp = (ctx.contract or {}).get("chunk_policy") or {}
        if cp and not cp.get("tokenizer_revision"):
            from cytir.sources import contract as contract_mod
            rev = contract_mod.by_id(contract_mod.load())[self._REVISION_ROW]["chunk_policy"]["tokenizer_revision"]
            ctx = dataclasses.replace(ctx, contract={**ctx.contract, "chunk_policy": {**cp, "tokenizer_revision": rev}})
        self._selection = declared_policy_selection(ctx)

    def extract(self, cap: Captured, ctx: RunContext):
        from cytir.extract import uk_legislation_akn as UK
        if cap.unit.key == "bulk-original":
            raise UnitExcluded("bulk_original_record_is_not_an_akn_item")
        try:
            item = UK.parse(cap.body)
        except UK.NotAkn as exc:
            raise UnitExcluded(exc.reason)
        if not item.sections:
            raise UnitExcluded("no_citable_provision_text")
        ctx.notes["empty_provisions"] = ctx.notes.get("empty_provisions", 0) + item.empty_provisions
        docs = []
        for sec in item.sections:
            text = UK.compose_text(item, sec)
            doc_id = hashlib.sha256(f"uk-legislation\0{sec.url}".encode()).hexdigest()
            # a structurally empty body has nothing to compare: source_text stays None, the
            # manifest counts the doc UNCHECKED (never checked-and-passed)
            rows = [u.text for u in self._units(text, ctx)]
            parts = _guard_parts(doc_id, sec.body, rows)
            docs.append(Doc(
                url=sec.url, text=text, title=UK.compose_title(item, sec), lang="en", anchor=sec.e_id,
                source_parts=parts, part_of=(lambda r: r["document_id"]) if parts else None, extra={
                    "document_id": doc_id,
                    "work": item.work_path, "legislation_type": item.type, "kind": sec.kind,
                    "label": sec.label, "citation": item.citation}))
        return docs

    def _units(self, text: str, ctx: RunContext):
        """The chunker's units for one section text. `extract` reads the row splits from them (the
        guard's cuts) and `chunk` stages them: one computation, kept for the section until staged."""
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch
        selection = getattr(self, "_selection", None)
        if selection is None:
            raise RuntimeError("uk_legislation.chunk ran before preflight: the declared chunk "
                               "policy is resolved in preflight; call preflight(ctx) first")
        cache = self.__dict__.setdefault("_unit_cache", {})
        key = hashlib.sha1(text.encode()).hexdigest()
        if key not in cache:
            cache[key] = _dispatch.chunk_markdown_channels(text, selection=selection)
        return cache[key]

    def chunk(self, doc: Doc, ctx: RunContext):
        from urllib.parse import urlparse
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch
        selection = getattr(self, "_selection", None)
        if selection is None:
            raise RuntimeError("uk_legislation.chunk ran before preflight: call preflight(ctx) first")
        units = self._units(doc.text, ctx)
        stamps, digest = selection.stamps(), C.sha1(doc.text)
        self.__dict__.get("_unit_cache", {}).pop(hashlib.sha1(doc.text.encode()).hexdigest(), None)
        out = []
        for ordinal, unit in enumerate(units):
            unit = dataclasses.replace(unit, stamps=stamps)
            # one chunk: the provision's own eId; several: eId plus the ordinal (ids stay unique)
            anchor = doc.anchor if len(units) == 1 else f"{doc.anchor}:chunk-{ordinal}"
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(unit.dense_text or unit.text, 25),
                lang=doc.lang, domain=urlparse(doc.url).hostname or "", tier=None,
                source_type=None, trust_tier=None, last_seen=ctx.today, content_digest=digest,
                anchor=anchor, license=self._declared_license())
            rec = C.passage_source_record(p)
            rec["document_id"], rec["chunk_ordinal"] = doc.extra["document_id"], ordinal
            out.append(rec)
        return out


@register_family
class ItSenatoGazzettaFamily(_OfficialBulkFamily):
    family_id = "it_senato_gazzetta"
