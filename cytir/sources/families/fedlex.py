#!/usr/bin/env python3
"""fedlex loader via the structured filestore (Akoma Ntoso XML) — no rendering.

CC only captures fedlex's Angular shell (verified), and the Playwright render is
fragile. But fedlex publishes every consolidated law as Akoma Ntoso XML at its
filestore. We resolve the current consolidation date from the fedlex RDF, build
the filestore XML URL, and emit one passage per article (anchored #art_N).

  url pattern:
    https://fedlex.data.admin.ch/filestore/fedlex.data.admin.ch/eli/<eli>/<date>/<lang>/xml/
      fedlex-data-admin-ch-eli-<eli dashed>-<date>-<lang>-xml.xml

  python fedlex_dump.py --eli cc/1999/404 --langs de fr it en --out build/fedlex.jsonl

--- task 358b (lane R2, citation-enriched re-ingest) ---------------------

The 358a audit found statute records lexically STARVED of the vocabulary
agents actually query with: no law abbreviation (OR/ZGB/BV...), no SR
number, no site_name ("fedlex" only matchable via the portal card), no
section-heading context ("Miete" nowhere in art. 257's record even though
it lives under "Achter Titel: Die Miete"), amendment-history footnotes
flattened into the front of the normative text (the 335c-class defect), no
EN records despite fedlex publishing EN translations, and — separately — a
suspected tail-chunk-loss for long articles under the Tier-A identity
scheme (same anchor on 2+ chunks of one article collides, dedup silently
keeps one). This revision:

  * `resolve_law_meta` — live SPARQL (jolux:title / jolux:titleShort /
    jolux:historicalLegalId) for the per-lang abbreviation + SR number,
    with cross-lang SR backfill (the SR number is language-invariant but
    fedlex's SPARQL occasionally answers it for only some langs — verified
    empirically: SchKG/EN and DSG/de-fr-it both had gaps the other langs
    filled). `LAW_META` is the curated fallback for the one field SPARQL
    doesn't expose — a short *common* name distinct from the bare
    abbreviation and from the full formal docTitle (engineering-doctrine:
    curated knowledge lives in a small versioned table, not a fragile
    title-parsing heuristic) — for the ~18 ELIs ingested as of 2026-07-30.
    An ELI outside this table still gets a record (title falls back to the
    formal docTitle + abbrev/SR when SPARQL has them), just without the
    curated short name — loud WARNING, never a silent omission.
  * `_clean_text` — itertext() that SKIPS `authorialNote` subtrees (the
    "Eingefügt durch Ziff. I des BG vom 18. März 1988..." amendment/
    editorial notes AKN nests inside `<num>` and, on some paragraphs,
    inline in the body) instead of flattening them into the citable text.
  * `ancestor_heading_path` — walks AKN ancestors (part/title/chapter/
    level/... — any container carrying `<num>`/`<heading>`) bottom-up,
    cleaned of authorialNotes, giving both the prepended index-text context
    (last 3 levels — enough to add "Die Miete"/"Allgemeine Bestimmungen"
    without diluting every record with book-level boilerplate shared by
    hundreds of siblings) and the title's marginal note (closest level).
  * `window` — the SAME 240-word/300-word-threshold convention already
    used by `ingest_lexwork.py`/`extract_zh_articles.py`/the retired `ingest_vd` driver (deleted, task 1552)
    (task 117) for WHERE an article is cut. Every window keeps the BARE
    article url `…#art_N` (RULING 55, task 1040): the window's place is the
    payload field `window = {"index": K, "count": N}`, never a `-cK` url
    suffix. Identity is tier B (url + digest of the window text), so the
    windows of one article stay distinct ids at one url.
  * representation: the DENSE surface is `compose_index_text` over the
    article-local text, i.e. `"<title>. Fedlex. <body>"` — the surface the
    served vector was made from (task 526 §1.1; task 1040 R1); the heading
    path is retained in `sparse_text` for BM25. Dense retrieval had
    a measured natural-language loss when the complete hierarchy was put in
    both channels (task 358c).  `--representation legacy` reproduces the
    prior both-channels surface for a controlled ablation.
  * short-source pointers: an article with no operative text but an Akoma
    Ntoso `authorialNote` is retained as a deliberately minimal pointer.
    Repeal notices receive `status="repealed"`; other source notes receive
    `status="source_note"`.  The note is authoritative evidence about that
    canonical article URL, not normative body text.  Conversely, a genuinely
    short operative provision is retained as a normal provision: a generic
    character floor is not a valid reason to delete a statute article.
  * removed-source tombstones: on a delta rebuild, `--prior-records` compares
    the prior indexed Fedlex payload snapshot with the raw article eIds in the
    newly fetched consolidation.  A prior anchor absent from that raw XML is
    retained as `status="removed_from_current_source"` with the exact current
    source version and filestore URL.  If the raw XML still contains the
    anchor but parsing emitted no row, the build aborts: parser omissions must
    never be disguised as source removals.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import pathlib
import re
import sys

import httpx
from lxml import etree

from cytir.core.common import Passage, canonical_url, compose_index_text, extractive_snippet
from cytir.core import ids as ID
from cytir.index import id_rule as IR
from cytir.sources.families.fedlex_contract import fedlex_xml_url, source_native_key, split_anchor
# ONE HOME for the source-native AKN text + title contracts (task 516). These
# five helpers existed here AND in `fedlex_akn.py` as byte-identical copies --
# a forked contract between the live loader and the pinned re-materialization,
# which is exactly how the epoch-4 titles came to be composed by a different
# rule than the one production shipped. They are imported now, not duplicated.
from cytir.sources.families.fedlex_akn import (ancestor_heading_path, article_citation,
                        compose_article_title, compose_law_sparse_surface,
                        clean_text as _clean_text,
                        norm as _norm, source_note_pointer,
                        text_without as _text_without)
# ONE HOME for the law's short name / abbreviation / SR number and for the
# prefix composition (task 516). The curated table moved out of this file into
# `data/fedlex-law-names.v1.json` the moment a second producer needed it -- the
# epoch-4 re-materializer -- because a curated table referenced by two scripts
# is stock, not cattle. `resolve_law_meta` lives there too, so the live path and
# the sealed path resolve the same fields from the same query.
from cytir.sources.families import fedlex_law_meta as LM
from cytir.acquire import fetch_process as fp  # noqa: F401  (kept: resolves lazy trafilatura import elsewhere)

H: httpx.Client | None = None      # built on first use by `_h()`; tests may replace it


def _h() -> httpx.Client:
    """The consent-guarded fedlex client, built on first use (spec step 4)."""
    global H
    if H is None:
        H = make_client(HttpSpec(  # HttpSpec/make_client are imported below (E402 block)
            timeout=60, follow_redirects=True,
            headers={"user-agent": "authoritative-web/0.1 (+https://websear.ch)"}))
    return H


TODAY = dt.date.today().isoformat()
AKN = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
SPARQL = "https://fedlex.data.admin.ch/sparqlendpoint"
JOLUX = "http://data.legilux.public.lu/resource/ontology/jolux#"
#: The per-record `license` of every fedlex row (RULING 53, Tim 2026-10-22: "public domain by statute
#: is fine"). The vocabulary value is `public-domain`, the one `sources/ch/fedlex.yaml` declares
#: (`license.default`) and the US federal-law doors stamp. THE BASIS is the instrument, not the host:
#: Art. 5 URG (SR 231.1) puts laws, ordinances, treaties and other official enactments outside
#: copyright (contract row `licence`). The licence version and an attribution are master-only
#: (ADR-039) and are not stamped on the row: a public-domain work has neither.
LICENSE = "public-domain"
COMMON = dict(domain="fedlex.admin.ch", tier="A", trust_tier=1.0,
              source_type="official", last_seen=TODAY, site_name="Fedlex")

# The ELI/anchor pattern and the window-suffix rule live in
# `fedlex_contract.py` and are imported, not re-declared. Task 529 found the
# copy that used to sit here: a byte-for-byte reimplementation of
# `source_native_key`, identical to the original on the day it was written and
# free to diverge from it on any later day. Engineering doctrine 1/6 -- a
# second implementation of an identity function converts one bug into two, and
# a divergent Fedlex identity re-mints a whole family under different point
# ids.
SOURCE_VERSION_MANIFEST_SCHEMA = "aw.fedlex-source-version-manifest.v1"
_SOURCE_VERSION = re.compile(r"\d{8}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _t(tag: str) -> str:
    return "{%s}%s" % (AKN, tag)


# The curated short/common law names now live in
# `data/fedlex-law-names.v1.json` (see `fedlex_law_meta.load_curated_names`).
# `LAW_META` stays as a lazily-loaded module attribute so existing callers and
# tests keep working, but there is exactly one copy of the data and it is a
# versioned file with a schema, not a literal in a loader.
_LANG_SPARQL = LM.LANG_SPARQL


def _law_names() -> dict:
    global LAW_META
    if LAW_META is None:
        LAW_META = LM.load_curated_names()
    return LAW_META


LAW_META = None


def resolve_date(eli: str, today: str | None = None, client=None) -> str | None:
    """Current in-force consolidation date (YYYYMMDD) — latest dateApplicability
    ≤ today among the law's consolidations (future amendments are excluded).
    `today` (ISO date) and `client` default to the import-time `TODAY` and `_h()`; the family
    adapter passes the run's own date and client (task 683), so a replay of sealed input is
    date-stable and never touches the module-global client."""
    q = ("PREFIX jolux: <%s> SELECT ?date WHERE { "
         "?c jolux:isMemberOf <https://fedlex.data.admin.ch/eli/%s> ; "
         "jolux:dateApplicability ?date . } ORDER BY DESC(?date)") % (JOLUX, eli)
    r = (client or _h()).post(SPARQL, data={"query": q},
                           headers={"Accept": "application/sparql-results+json"})
    if r.status_code != 200:
        return None
    dates = [b["date"]["value"] for b in r.json()["results"]["bindings"]]
    in_force = [d for d in dates if d <= (today or TODAY)]   # already DESC-ordered
    return in_force[0].replace("-", "") if in_force else None


def resolve_law_meta(eli: str, langs: list[str]) -> dict[str, dict]:
    """Live SPARQL, through the shared resolver in `fedlex_law_meta`.

    The query, the de/fr/it probe set and the cross-language SR backfill live in
    exactly one place now, so the live loader and
    the retired `seal_fedlex_law_meta` script (which produces the sealed artifact the
    pinned re-materialization reads) cannot answer differently.
    """
    return LM.resolve_law_meta(eli, langs, _h(),
                               log=lambda m: print(m, file=sys.stderr))


def window(body: str, base_anchor: str, target_words: int = 240,
          threshold_words: int = 300) -> list[tuple[str, str]]:
    """Task 117 windowing convention (ingest_lexwork.py / extract_zh_articles.py
    / the retired `ingest_vd` driver, deleted in task 1552) for WHERE to cut: <=300 words stays one chunk; longer
    articles split into ~240-word windows. EVERY window keeps the BARE anchor
    (RULING 55, Tim 2026-10-22, task 1040): a link to the article lands on the
    article, not on a window. The window's place is `window_field()`, a payload
    field. Identity is tier B (url + digest of the window text, `stamp_i1_identity`),
    so two windows at one url are two ids; `parse_akn` refuses an id collision."""
    words = body.split()
    if len(words) <= threshold_words:
        return [(base_anchor, body)]
    return [(base_anchor, " ".join(words[ci:ci + target_words]))
            for ci in range(0, len(words), target_words)]


def window_field(index: int, count: int) -> dict | None:
    """The payload field of window `index` (1-based) of `count`: `{"index", "count"}`, or None for
    a one-row article (RULING 55: the field replaces the old `-cK` url suffix)."""
    return {"index": index, "count": count} if count > 1 else None


def xml_url(eli: str, date: str, lang: str) -> str:
    """Backward-compatible public name for the shared Fedlex URL contract."""
    return fedlex_xml_url(eli, date, lang)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_prior_records(value: str | None, *, pinned: bool) -> pathlib.Path | None:
    """Require immutable prior-payload identity for a pinned Task370 replay."""
    if value is None:
        if pinned:
            raise ValueError("pinned source mode requires --prior-records SHA256=PATH")
        return None
    if not pinned:
        return pathlib.Path(value)
    expected, separator, raw_path = value.partition("=")
    path = pathlib.Path(raw_path)
    if not separator or not _SHA256.fullmatch(expected) or not path.is_file():
        raise ValueError("pinned --prior-records must be SHA256=PATH to a file")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"prior-records SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return path


def load_source_version_manifest(binding: str, elis: list[str],
                                 langs: list[str]) -> dict[str, dict[str, dict]]:
    """Load the immutable Task-370 historical source selection.

    The normal live path intentionally remains available.  Once a historical
    source manifest is requested, however, a partial ELI/language selection is
    worse than no run: the exact 18-ELI source set, every requested language,
    its version date, and its filestore URL must all be present. Each record
    must be either successful or explicitly retained as historically unavailable.
    """
    expected, separator, raw_path = binding.partition("=")
    path = pathlib.Path(raw_path)
    if not separator or not _SHA256.fullmatch(expected) or not path.is_file():
        raise ValueError("--source-version-manifest must be SHA256=PATH to a file")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"source-version manifest SHA-256 mismatch for {path}: "
            f"expected {expected}, got {actual}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"source-version manifest is invalid JSON: {path}") from exc
    if not isinstance(document, dict) or document.get("schema") != SOURCE_VERSION_MANIFEST_SCHEMA:
        raise ValueError("source-version manifest has an unsupported schema")
    entries = document.get("elis")
    if not isinstance(entries, list) or len(entries) != 18:
        raise ValueError("source-version manifest must contain exactly 18 ELI entries")
    selected = set(elis)
    if len(elis) != 18 or len(selected) != 18:
        raise ValueError("pinned source mode requires exactly 18 unique --eli values")
    result: dict[str, dict[str, dict]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("source-version manifest ELI entry must be an object")
        eli = entry.get("eli")
        per_language = entry.get("languages")
        if not isinstance(eli, str) or not eli or not isinstance(per_language, dict):
            raise ValueError("source-version manifest ELI entry lacks eli/languages")
        if eli in result:
            raise ValueError(f"source-version manifest has duplicate ELI {eli!r}")
        for language, record in per_language.items():
            if language not in {"de", "fr", "it", "en"} or not isinstance(record, dict):
                raise ValueError(f"source-version manifest has malformed language record for {eli}")
            status = record.get("status")
            if status not in {"success", "parse_failed"}:
                raise ValueError(f"source-version manifest has invalid status for {eli}/{language}")
            if status == "parse_failed" and (
                    not isinstance(record.get("error"), str) or not record["error"]):
                raise ValueError(
                    f"source-version manifest parse_failed {eli}/{language} lacks error evidence")
        result[eli] = per_language
    if set(result) != selected:
        raise ValueError("source-version manifest ELI coverage does not exactly match --eli")
    for eli in elis:
        for lang in langs:
            record = result[eli].get(lang)
            if not isinstance(record, dict):
                raise ValueError(f"source-version manifest lacks {eli}/{lang} coverage")
            version, url = record.get("source_version"), record.get("source_version_url")
            if not isinstance(version, str) or not _SOURCE_VERSION.fullmatch(version):
                raise ValueError(f"source-version manifest has invalid version for {eli}/{lang}")
            expected_url = xml_url(eli, version, lang)
            if url != expected_url:
                raise ValueError(
                    f"source-version manifest URL/version mismatch for {eli}/{lang}: "
                    f"expected {expected_url!r}, got {url!r}")
            if record.get("status") == "success" and (
                    not isinstance(record.get("title_prefix"), str)
                    or not record["title_prefix"]):
                raise ValueError(f"source-version manifest lacks pinned title_prefix for {eli}/{lang}")
    return result


def stamp_i1_identity(row: dict) -> dict:
    """The I1 id of a door row, tier B (ADR-067 am.3, task 1035): `id_rule.mint_i1` over the row's
    url and the 64-hex `content_digest` of its dense text; no formula is written here.

    Fedlex rows are digest-addressed, not anchor-addressed: the twin's 22,927 fedlex rows are tier B
    (proven 7/7 on sealed pack rows, task 1035 section 0). The fragment
    (`#art_N`, `#art_N-cK`) stays in the url, so a page keeps its address. The row carries NO
    `anchor`: the projection (`cytir.core.projection.project_record`) classifies a row that has a
    native anchor as tier A and would mint the b1 id, not this one. `base_anchor` stays (a bookkeeping
    key the staging strips). The digest is over `text`, the surface the projection digests too."""
    row.pop("anchor", None)
    digest = ID.content_digest(row["text"])
    new, why = IR.mint_i1({**row, "tier": "B", "content_digest": digest})
    if new is None:  # unreachable for a row with a url; a refusal is better than a guessed id
        raise ValueError(f"{row.get('url')!r}: no I1 id ({why})")
    row["point_id"] = new
    return row


def parse_akn(xml_bytes: bytes, canon: str, lang: str, eli: str,
              meta: dict[str, dict], *, source_version: str | None = None,
              source_version_url: str | None = None,
              seen_anchors: set[str] | None = None,
              representation: str = "channel_split",
              title_prefix: str | None = None,
              last_seen: str | None = None,
              source_parts: list | None = None) -> list[dict]:
    """One law/language of Akoma Ntoso -> article rows. `last_seen` (ISO date) overrides the
    import-time `COMMON["last_seen"]`: the family adapter stamps the RUN's date (task 683).
    `source_parts`, when given, collects one `SourcePart` per article: the text its rows must hold
    (its body, or the repeal pointer notice that replaces it) and the word offsets where `window()`
    cut it. The stage-time guard compares article by article (tasks 1038, 1045)."""
    common = COMMON if last_seen is None else {**COMMON, "last_seen": last_seen}
    if representation not in {"channel_split", "legacy"}:
        raise ValueError("representation must be channel_split or legacy")
    root = etree.fromstring(xml_bytes)
    tnode = root.find(".//" + _t("docTitle"))
    doc_title = _norm(_clean_text(tnode)) if tnode is not None else canon

    m = meta.get(lang) or {}
    abbrev = m.get("short")
    sr = m.get("sr")
    name = LM.short_name(_law_names(), eli, lang, abbrev, doc_title,
                         warn=lambda msg: print("  ! " + msg, file=sys.stderr))

    rows = []
    for art in root.iter(_t("article")):
        eid = art.get("eId") or art.get("id")
        if not eid:
            continue
        if seen_anchors is not None:
            seen_anchors.add(eid)
        art_cite = article_citation(art, eid)
        body = _norm(_clean_text(art))
        pointer = None
        if len(body) < 60:
            pointer = source_note_pointer(art)
            if pointer is not None:
                # The source note, rather than its absent operative body, is
                # the entire pointer payload.  This preserves the canonical
                # (url, anchor) without presenting the note as law text.
                pointer_notice, pointer_status = pointer
            else:
                substantive = _text_without(art, {"num", "authorialNote"})
                # Fedlex is a structured, authoritative statute source.  A
                # short actual provision (for example, a one-sentence cross
                # reference) remains useful and must not disappear merely
                # because it misses a web-crawl-oriented character floor.
                if not re.search(r"\w", substantive, flags=re.UNICODE):
                    continue
                pointer_notice = None
                pointer_status = None
        else:
            pointer_notice = None
            pointer_status = None
        path, marginal = ancestor_heading_path(art)
        # Full path, not just the closest levels: task 358b proved truncation
        # can cut OR art. 257's query-relevant "Die Miete" ancestor. It is a
        # sparse-only retrieval hook in the corrected representation. Putting
        # it into dense as well diluted the article semantics (task 358c).
        heading_ctx = "; ".join(path)

        prefix = title_prefix or LM.compose_title_prefix(name, abbrev, sr)
        title = compose_article_title(prefix, art_cite, marginal)

        # A repeal pointer has the same canonical (url, anchor) identity as
        # the old stub.  Its explicit status lets presentation/ranking layers
        # distinguish it from operative law while preserving a useful source
        # link and the repeal citation.
        chunks = [(eid, pointer_notice)] if pointer_notice else window(body, eid)
        if source_parts is not None:
            cuts, at = [], 0
            for _a, piece in chunks[:-1]:
                at += len(piece.split())
                cuts.append(at)
            source_parts.append(SourcePart(eid, pointer_notice or body, tuple(cuts)))
        for w_index, (anchor, chunk_body) in enumerate(chunks, start=1):
            url = "%s#%s" % (canon, anchor)     # the BARE article url on every window (RULING 55)
            # ONE HOME (task 564): the epoch-4 re-materialization needs the
            # identical composition to reconstruct this family's sparse surface
            # from its pack, and a second copy of the format string is how the
            # two paths drift.
            sparse_text = compose_law_sparse_surface(heading_ctx, chunk_body)
            # `text` remains the canonical article-local source surface. The
            # staging path selects explicit channels below; legacy mode leaves
            # the historical hierarchy-in-both surface byte-for-byte shaped.
            text = chunk_body if representation == "channel_split" else sparse_text
            p = Passage(url=url, title=title, snippet=extractive_snippet(chunk_body),
                        text=text, lang=lang, anchor=anchor, **common)
            row = p.payload()
            row["text"] = p.text
            row["license"] = LICENSE          # RULING 53: public domain by statute (Art. 5 URG)
            if representation == "channel_split":
                # R1 (task 1040): the dense surface is the one the served vector was made from,
                # `compose_index_text` = "<title>. Fedlex. <body>" (task 526 §1.1: body only cost
                # 20 hit@1 on `legal`). ONE composer; `project_record` re-applies it and its
                # dedup guard makes that a no-op. The id stays the digest of `text` (the body).
                row["dense_text"] = compose_index_text(
                    {"text": chunk_body, "title": title, "site_name": COMMON["site_name"],
                     "source_type": COMMON["source_type"]})
                row["sparse_text"] = sparse_text
            if (wf := window_field(w_index, len(chunks))) is not None:
                row["window"] = wf
            # Identity (ADR-067 am.3, task 1035): I1 tier B, minted by `stamp_i1_identity` from the
            # row's own text. A changed article is a new id; the supersede pack tombstones the old
            # row by URL group.
            stamp_i1_identity(row)
            if pointer_notice:
                row["status"] = pointer_status
            if source_version:
                row["source_version"] = source_version
            if source_version_url:
                row["source_version_url"] = source_version_url
            row["base_anchor"] = eid          # 358b bookkeeping: old-id lookup key for the supersede pack builder (stripped before staging)
            rows.append(row)
    # Several windows share one url now (RULING 55): their ids differ only by the digest of their
    # text. Two equal windows at one url would mint ONE id and the second row would overwrite the
    # first in the index, so a collision is refused here, never left to the upsert.
    seen: set[str] = set()
    for row in rows:
        if row["point_id"] in seen:
            raise RuntimeError(f"{eli}/{lang}: two rows mint id {row['point_id']} "
                               f"({row['url']}); refusing a silent overwrite")
        seen.add(row["point_id"])
    return rows


def load_law(eli: str, langs: list[str],
             source_state: dict[tuple[str, str], dict] | None = None,
             *, representation: str = "channel_split",
             source_versions: dict[str, dict[str, dict]] | None = None) -> list[dict]:
    pinned = source_versions is not None
    date = None if pinned else resolve_date(eli)
    if not pinned and not date:
        print("  ! %s: could not resolve date" % eli)
        return []
    # Task370 pins title prefixes from historical output; live SPARQL metadata
    # must not become a representation-arm confound.
    meta = {} if pinned else resolve_law_meta(eli, langs)
    out = []
    for lang in langs:
        try:
            if pinned:
                spec = source_versions.get(eli, {}).get(lang)
                if not isinstance(spec, dict):
                    raise RuntimeError(f"pinned source manifest lacks {eli}/{lang}")
                date = spec["source_version"]
                version_url = spec["source_version_url"]
                if spec["status"] == "parse_failed":
                    print("  · %s/%s (%s) → SKIP historical parse_failed: %s" %
                          (eli, lang, date, spec["error"]))
                    continue
            else:
                version_url = xml_url(eli, date, lang)
            r = _h().get(version_url)
            if r.status_code != 200 or not r.content:
                raise RuntimeError("http %s" % r.status_code)
            fetched_url = str(getattr(r, "url", version_url))
            if pinned and fetched_url != version_url:
                raise RuntimeError(
                    f"pinned source URL drift for {eli}/{lang}: "
                    f"requested {version_url}, fetched {fetched_url}")
            canon = "https://www.fedlex.admin.ch/eli/%s/%s" % (eli, lang)
            seen_anchors: set[str] = set()
            rows = parse_akn(
                r.content, canon, lang, eli, meta, source_version=date,
                source_version_url=version_url, seen_anchors=seen_anchors,
                representation=representation,
                title_prefix=spec.get("title_prefix") if pinned else None)
            if source_state is not None:
                source_state[(eli, lang)] = {
                    "source_version": date,
                    "source_version_url": version_url,
                    "raw_article_anchors": seen_anchors,
                }
            out.extend(rows)
            print("  · %s/%s (%s) → %d article-passages" % (eli, lang, date, len(rows)))
        except Exception as e:  # noqa: BLE001
            if pinned:
                raise RuntimeError(f"pinned Fedlex source failed for {eli}/{lang}: {e}") from e
            print("  ! %s/%s: %s" % (eli, lang, e))
    return out


def _source_key(url: str) -> tuple[str, str, str] | None:
    """The shared Fedlex identity. Kept as a name because callers use it."""
    return source_native_key(url)


def load_prior_records(path: pathlib.Path) -> list[dict]:
    """Load a replayable prior payload snapshot (plain JSONL or JSONL.gz).

    Both raw payload rows and standard qdrant dumps
    (`{"id", "vector", "payload"}`) are accepted.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            record = item.get("payload", item)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno}: expected payload object")
            rows.append(record)
    return rows


def removed_source_tombstones(
        prior_records: list[dict], current_rows: list[dict],
        source_state: dict[tuple[str, str], dict]) -> list[dict]:
    """Build non-operative pointers for prior anchors absent from current AKN.

    Raw-source presence is the safety boundary.  If a missing emitted record's
    eId still exists in the fetched XML, this is a parser regression and the
    run fails rather than minting a misleading tombstone.
    """
    current_keys = {
        key for row in current_rows
        if (key := _source_key(row.get("url") or "")) is not None
    }
    prior_by_key: dict[tuple[str, str, str], dict] = {}
    for row in prior_records:
        if (row.get("domain") or "").lower() != "fedlex.admin.ch":
            continue
        if row.get("source_type") != "official":
            continue
        key = _source_key(row.get("url") or "")
        if key is not None:
            prior_by_key.setdefault(key, row)

    tombstones = []
    for key in sorted(set(prior_by_key) - current_keys):
        eli, lang, anchor = key
        state = source_state.get((eli, lang))
        if state is None:
            raise RuntimeError(
                f"cannot verify removed Fedlex anchor {key}: current source "
                "fetch/parse did not succeed")
        if anchor in state["raw_article_anchors"]:
            raise RuntimeError(
                f"Fedlex parser omission for {key}: raw current AKN still "
                "contains the article eId; refusing a source-removal tombstone")

        prior = prior_by_key[key]
        version = state["source_version"]
        version_url = state["source_version_url"]
        prior_base_url = (prior.get("url") or "").split("#", 1)[0]
        url = f"{prior_base_url}#{anchor}"
        # This is deliberately source-state metadata, not reconstructed law
        # text.  The prior snippet/body is never copied into the tombstone.
        pointer_text = (
            f"Fedlex source-state pointer. Anchor {anchor} is absent from "
            f"official consolidated source version {version}.")
        prior_title = (prior.get("title") or
                       (LAW_META.get(eli) or {}).get(lang) or eli)
        title = f"{prior_title} — {anchor} — source pointer"
        passage = Passage(
            url=url, title=title, snippet=pointer_text, text=pointer_text,
            lang=lang, anchor=anchor, **COMMON)
        row = passage.payload()
        row.update({
            "text": passage.text,
            "status": "removed_from_current_source",
            "source_version": version,
            "source_version_url": version_url,
            "source_anchor_evidence": "absent_from_current_akn_article_eids",
            "base_anchor": anchor,
            "license": LICENSE,               # RULING 53, as every door row (task 1040)
            # R1 (task 1040): the same composed dense surface as the article rows
            "dense_text": compose_index_text({"text": passage.text, "title": title,
                                              "site_name": COMMON["site_name"],
                                              "source_type": COMMON["source_type"]}),
        })
        stamp_i1_identity(row)
        tombstones.append(row)
    return tombstones


def removed_share_cap(contract: dict) -> float | None:
    """The contract row's `refresh.removed_share_cap` (a share in (0, 1]), or None when undeclared."""
    cap = ((contract or {}).get("refresh") or {}).get("removed_share_cap")
    if isinstance(cap, (int, float)) and not isinstance(cap, bool) and 0 < cap <= 1:
        return float(cap)
    return None


def removed_rows_for_act(prior_records: list[dict], rows: list[dict], *, eli: str, lang: str,
                         raw_article_anchors: set[str], source_version: str | None,
                         source_version_url: str | None, cap: float) -> list[dict]:
    """The removed-source pointer rows of ONE act-language in a refresh (task 1040, 1038 section 7
    item 5, review 1044 R7d). `prior_records` = the pinned served payloads (the drain's victims);
    only rows in this act-language scope are read. A removal is confirmed from the raw XML
    (`removed_source_tombstones` refuses an anchor whose article eId is still there: a parse loss is
    never turned into a removal). `cap` = the most this act-language may remove, as a share of the
    prior base anchors (contract row `refresh.removed_share_cap`); above it the door refuses.

    A prior row with a non-string `domain`/`source_type` (an encoded payload profile) is refused:
    `removed_source_tombstones` would skip it and the removal would be silently 0."""
    scoped = [r for r in prior_records
              if (k := source_native_key(r.get("url") or "")) is not None and k[:2] == (eli, lang)]
    for r in scoped:
        if not isinstance(r.get("domain"), str) or not isinstance(r.get("source_type"), str):
            raise RuntimeError(f"{eli}/{lang}: prior record {r.get('url')!r} carries an encoded "
                               "domain/source_type; decode the payload profile first")
    state = {(eli, lang): {"raw_article_anchors": set(raw_article_anchors),
                           "source_version": source_version,
                           "source_version_url": source_version_url}}
    removed = removed_source_tombstones(scoped, rows, state)
    prior_anchors = {source_native_key(r["url"])[2] for r in scoped}
    if prior_anchors and len(removed) / len(prior_anchors) > cap:
        raise RuntimeError(
            f"{eli}/{lang}: the refresh removes {len(removed)} of {len(prior_anchors)} prior "
            f"articles ({len(removed) / len(prior_anchors):.1%}), above the declared cap {cap:.0%}; "
            "refusing (a removal this large is a wrong input or a broken parse until a human reads it)")
    return removed


# ---------------------------------------------------------------------------------------
# The family adapter (task 683, S7): `cyt source run fedlex_exact --opt eli=cc/1999/404`.
#
# `main()` below is the old script door and stays (it is what `sources/ch/fedlex.yaml`'s
# `extractor:` names). The adapter is the SAME loader driven by the source-family framework:
# the framework owns consent, the L0 seal, the terminals and the pack; this class owns only
# the fedlex dialect. Every parse / window / title / identity function is the one above.
# ---------------------------------------------------------------------------------------
from cytir.sources.staged_text_guard import SourcePart  # noqa: E402
from cytir.sources.family import CHUNK_NATIVE_CHUNKED, Captured, Doc, HttpSpec, RunContext, Unit  # noqa: E402
from cytir.sources.http import make_client  # noqa: E402
from cytir.sources.registry import register_family  # noqa: E402

FEDLEX_DATA_HOST = "fedlex.data.admin.ch"


def _opt_list(value, default=None) -> list[str]:
    if value is None:
        return list(default or [])
    return [value] if isinstance(value, str) else list(value)


#: What this door MINTS, SPLITS and STAMPS with (the PR #651 preflight compares them with the contract).
#: ID RULE. The door mints the I1 rule, tier B (`stamp_i1_identity`: `id_rule.mint_i1` over url + the
#: 64-hex digest of the text); the twin's ids reproduce under it (task 1035 section 0). The contract row
#: declares it by the name the four PDF replay families use: `identity.minted_by` NAMES `mint_point_id`
#: (ruling 30, task 1038). The old label `text-anchor-v0` is the generation-0 mint and is refused.
#: LICENCE. The contract row declares the per-record value in `licence.spdx` (RULING 53): the door's
#: `LICENSE` must equal it, so a contract edit cannot leave the door stamping another string.
#: The preflight reads the contract row (`ctx.contract`); it cannot see what the door mints.
#: SPLITTER. The door runs `window-240-words` = `window()` (bare url on every window, RULING 55).
#: The refusal was ON (ADR-067 am.3, task 871, review 1037) until the per-gold zero-loss gate of task
#: 1038 section 6 had run. LIFTED by ADR-067 amendment 4 (task 1040): on a faithful clone of the twin,
#: the refresh of cc/2022/491 (de fr it en) lost no fedlex gold on any of the three legs of either
#: instrument, the held-out LEGAL/NAMES tripwires and every other class equal the control
#: (task 1040 doc). The declaration is now the splitter the door runs. The one law
#: splitter of loda and fedlex stays a follow-up under ADR-043.
DOOR_ID_RULE = "mint_point_id"
GEN0_ID_RULE = "text-anchor-v0"
DOOR_SPLITTER = "window-240-words"
DECLARED_SPLITTER = DOOR_SPLITTER
_NOT_CHECKED = object()


def door_declaration_refusal(declared_rule: str | None, declared_licence=_NOT_CHECKED) -> str | None:
    """One line naming the mismatch between the contract and this door, or None when they agree.
    `declared_licence` (the contract's `licence.spdx`) is compared only when given."""
    bad = []
    rule = str(declared_rule or "")
    if DOOR_ID_RULE not in rule or GEN0_ID_RULE in rule:
        bad.append(f"the contract declares id rule `{declared_rule}`, the door mints `{DOOR_ID_RULE}`")
    if declared_licence is not _NOT_CHECKED and declared_licence != LICENSE:
        bad.append(f"the contract declares licence `{declared_licence}`, the door stamps `{LICENSE}`")
    if DOOR_SPLITTER != DECLARED_SPLITTER:
        bad.append(f"the contract declares the one law splitter (`{DECLARED_SPLITTER}`), the door "
                   f"splits with `{DOOR_SPLITTER}`")
    if not bad:
        return None
    return ("fedlex_exact: " + "; ".join(bad) + ". A pack from this door would mint ids and chunks "
            "the declaration does not describe, so the run is refused until the splitter and id fix "
            "land (ADR-067 am.2; at aw_e10)")


def _row_part(record: dict) -> str:
    """The guard part a staged row belongs to: its BASE anchor (`art_14` for `art_14-c2`)."""
    return split_anchor(str(record.get("url") or ""))[1] or ""


@register_family
class FedlexFamily:
    """Swiss federal law: Akoma Ntoso XML from the fedlex filestore, one article per row.

    Options: `eli` (required, repeatable, e.g. `cc/1999/404`), `langs` (repeatable, default
    de fr it), `representation` (`channel_split` default, or `legacy`). A unit is one ELI:
    its consolidation date and its per-language title/abbreviation/SR metadata are resolved
    together (the cross-language SR backfill needs them together), then one XML per language.
    Chunking is the article window (`window()`): native to the source's structure, so this
    family's chunk policy is `native-chunked` (Tim 2026-10-07, ruling 27). The 240-word window is
    LEGACY (task 117's cantonal convention): it is a split, not a cut, and it stays until the
    law splitter of loda and fedlex is one function (follow-up at the rebuild, ADR-043)."""

    family_id = "fedlex_exact"
    #: The split is `window()`: it cuts words and adds none, so every word of the source is in a row
    #: and the floor is 1.0 (the url-group supersede only skips its snippet guard at 1.0).
    staged_text_min_coverage = 1.0
    #: a dump family: an act that yields no record is a wrong input or a broken parser (task 1045)
    require_records = True
    route = "source_owned_bulk_dump"
    direct_crawl = True
    chunk_policy_id = CHUNK_NATIVE_CHUNKED

    def preflight(self, ctx: RunContext):
        """Refuse before anything is read while the door and the contract disagree (see above)."""
        from cytir.sources.run_family import RunError
        refusal = door_declaration_refusal((ctx.contract.get("identity") or {}).get("minted_by"),
                                           (ctx.contract.get("licence") or {}).get("spdx"))
        if refusal:
            raise RunError(refusal)
        if ctx.options.get("prior_records") and removed_share_cap(ctx.contract) is None:
            raise RunError("fedlex_exact: --opt prior_records needs the contract row to declare "
                           "`refresh.removed_share_cap` (task 1040); refusing an uncapped removal")

    def discover(self, ctx: RunContext):
        elis = _opt_list(ctx.options.get("eli"))
        if not elis:
            raise ValueError("fedlex_exact needs at least one ELI: --opt eli=cc/1999/404")
        langs = _opt_list(ctx.options.get("langs"), ["de", "fr", "it"])
        rep = ctx.options.get("representation", "channel_split")
        # task 1040: the pinned prior records (`<sha256>=<path>`, the served rows the refresh
        # supersedes) turn an article the new text no longer has into a removed-source pointer
        self._prior = None
        if ctx.options.get("prior_records"):
            self._prior = load_prior_records(
                resolve_prior_records(ctx.options["prior_records"], pinned=True))
            self._cap = removed_share_cap(ctx.contract)
        return [Unit(key=eli, url=f"https://{FEDLEX_DATA_HOST}/eli/{eli}",
                     meta={"eli": eli, "langs": langs, "representation": rep},
                     consent_hosts=(FEDLEX_DATA_HOST,)) for eli in elis]

    def acquire(self, unit: Unit, ctx: RunContext):
        eli, langs = unit.meta["eli"], list(unit.meta["langs"])
        client = ctx.http if ctx.http is not None else _h()
        date = resolve_date(eli, ctx.today, client)
        if not date:
            raise RuntimeError(f"{eli}: could not resolve a consolidation date")
        meta = LM.resolve_law_meta(eli, langs, client,
                                   log=lambda m: print(m, file=sys.stderr))
        caps = []
        for lang in langs:
            version_url = xml_url(eli, date, lang)
            r = client.get(version_url)
            if r.status_code != 200 or not r.content:
                raise RuntimeError(f"{eli}/{lang}: http {r.status_code}")
            caps.append(Captured(unit, r.content, {
                "eli": eli, "lang": lang, "source_version": date,
                "source_version_url": version_url,
                "fetched_url": str(getattr(r, "url", version_url)),
                # the externally resolved metadata a cold rebuild needs offline
                # (corpus-doctrine #6): sealed beside the bytes that it describes
                "law_meta": meta}))
        return caps

    def extract(self, cap: Captured, ctx: RunContext):
        m = cap.meta
        eli, lang = m["eli"], m["lang"]
        canon = "https://www.fedlex.admin.ch/eli/%s/%s" % (eli, lang)
        parts: list[str] = []
        seen: set[str] = set()
        rows = parse_akn(cap.body, canon, lang, eli, m["law_meta"],
                         source_version=m["source_version"],
                         source_version_url=m["source_version_url"],
                         seen_anchors=seen,
                         representation=cap.unit.meta["representation"],
                         last_seen=ctx.today, source_parts=parts)
        # a law is many articles: the parsed rows ARE the source's native chunks, so the
        # Doc carries them for `chunk` to hand on (no text is duplicated here)
        if not parts:
            # an XML that parses to 0 articles is a wrong input or a broken parser, never an empty
            # act (tasks 440/634: a month of 0-passage runs with green tests; task 1044 C2)
            raise RuntimeError(f"{eli}/{lang}: the XML parsed to 0 articles; refusing, not scoring "
                               "an empty source as fully covered")
        if getattr(self, "_prior", None) is not None:
            rows = rows + removed_rows_for_act(
                self._prior, rows, eli=eli, lang=lang, raw_article_anchors=seen,
                source_version=m["source_version"], source_version_url=m["source_version_url"],
                cap=self._cap)
        # One part per article: the stage-time guard compares each article with the rows at its
        # base anchor, forgiving only the 5-grams on `window()` cut points (task 1045).
        return [Doc(url=canon, text="", lang=lang, source_parts=parts, part_of=_row_part,
                    source_text="\n".join(p.text for p in parts), extra={"rows": rows})]

    def chunk(self, doc: Doc, ctx: RunContext):
        return doc.extra["rows"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eli", nargs="+", required=True, help="ELI(s), e.g. cc/1999/404")
    ap.add_argument("--langs", nargs="+", default=["de", "fr", "it"])
    ap.add_argument("--out", default="build/fedlex.jsonl")
    ap.add_argument(
        "--representation", choices=("channel_split", "legacy"),
        default="channel_split",
        help="channel_split keeps hierarchy in BM25 only (default); legacy "
             "reproduces the task-358b hierarchy-in-both-channels surface")
    ap.add_argument(
        "--prior-records", metavar="[SHA256=]PATH",
        help="prior Fedlex payload snapshot; emit source-version tombstones "
             "only for anchors proven absent from current raw AKN; pinned "
             "source mode requires SHA256=PATH")
    ap.add_argument(
        "--source-version-manifest", metavar="SHA256=PATH",
        help="optional Task-370 immutable 18-ELI source selection; bypasses "
             "live consolidation-date and metadata resolution")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    source_state: dict[tuple[str, str], dict] = {}
    source_versions = (load_source_version_manifest(
        args.source_version_manifest, args.eli, args.langs)
        if args.source_version_manifest else None)
    try:
        prior_records = resolve_prior_records(
            args.prior_records, pinned=source_versions is not None)
    except ValueError as exc:
        ap.error(str(exc))
    for eli in args.eli:
        rows.extend(load_law(eli, args.langs, source_state,
                             representation=args.representation,
                             source_versions=source_versions))
    tombstones = []
    if prior_records:
        tombstones = removed_source_tombstones(
            load_prior_records(prior_records), rows, source_state)
        rows.extend(tombstones)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote %d fedlex article-passages (%d removed-source tombstones; "
          "representation=%s) -> %s" %
          (len(rows), len(tombstones), args.representation, out))


if __name__ == "__main__":
    main()
