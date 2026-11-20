"""Shared schema + helpers for the AW build pipeline and loader.

A *passage* is the unit AW retrieves: the minimum record needed to rank and
locate an authoritative source (spec §4.3). We never store full content.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import sys

import uuid
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import urlparse

# Started by path (`python3 aw/...py`), `sys.path[0]` is this file's own directory, not the repo
# root: put this checkout's root first so `import cytir` resolves HERE and never to a foreign editable
# install (test_shim_defensive_import.py; test_aw_script_start_guard.py).
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cytir.core.paths import data_path as _data_path  # noqa: E402 -- after the sys.path line above

# --- canon (URL canonicalization) — MOVED to cytir/core/canon.py (task 646 S2,
# task 643 §6.1/§6.2). Re-exported here so every existing
# `aw_common.canonical_url(...)` / `from aw_common import canonical_url` call
# site keeps working unchanged: `aw_common.canonical_url is
# cytir.core.canon.canonical_url` (see test_aw_common_canon_shim.py).
# `tokenize`/`fold` (`_fold`, `bm25_tokens`, `_FOLD_ACCENTS`) stay HERE for
# now — that pair has its own independent hand-copy in
# `cytir/serve/schema.py` (the query-side mirror) and a cross-module
# fold-epoch check (`require_index_fold`); moving it safely is its own PR.
#
# MOVED HERE from `cytir/core/canon.py` whole (task 677, task 643 S5): the
# old path is an alias shim (`import aw_common` returns THIS module object).
# The shim carries the sys.path guard that used to sit here, so a stale
# editable `aw` install can never shadow this checkout's package.
from cytir.core.canon import (  # noqa: E402,F401
    canonicalize_url,
    canonical_url,
    credential_url_hit_count,
    fold_www,
    normalize_register_id,
    reset_credential_url_hit_count,
    strip_tracking_url,
)
# Qdrant collection + named-vector conventions (spec §5.2: three RRF legs).
COLLECTION = "aw_passages"
DENSE_VEC = "dense"          # BGE-M3 dense, 1024-d, cosine
SPARSE_LEXICAL = "lexical"   # BGE-M3 learned-sparse (token-id -> weight)
SPARSE_BM25 = "bm25"         # classic BM25 sparse (term-hash -> tf), IDF modifier
DENSE_DIM = 1024


def mint_passage_point_id(url: str, anchor: Optional[str], text: str) -> str:
    """Mint the legacy ``Passage`` UUID from its persisted identity inputs.

    This is the single write/read contract for source-native passages.  Keep
    callers on this helper rather than duplicating the UUID key formula: a
    reconstruction row must mint exactly the ID that ``Passage`` itself would
    mint for the same URL, anchor, and text.
    """
    key = f"{url}#{anchor or ''}:{sha1(text)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def mint_card_point_id(url: str) -> str:
    """The S2 card point id: ``uuid5(NAMESPACE_URL, url)``, one record per URL.

    The registry-record builders (task 1407: `scratch_ctgov_psy_exact`) minted this inline and
    their records are in the index under it; this helper is that exact expression, so it moves
    no id (a different scheme would orphan every card). Not the passage id: see `mint_passage_point_id`.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))


def require_qdrant_point_id(value: object) -> str:
    """Return a canonical Qdrant point id or raise before pack publication.

    AW emits UUID point ids.  Qdrant also accepts unsigned 64-bit integers, but
    accepting arbitrary strings here would let lineage identifiers (for
    example a 64-hex document SHA) travel as point ids until the GPU/load step.
    Staging adapters call this at the mint boundary so an invalid id cannot be
    sealed into an otherwise valid pack.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value < 2 ** 64:
            return str(value)
        raise ValueError(f"Qdrant integer point id outside uint64: {value}")
    if not isinstance(value, str):
        raise ValueError(f"Qdrant point id must be UUID or uint64, got {type(value).__name__}")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Qdrant point id is not a canonical UUID: {value!r}") from exc
    canonical = str(parsed)
    if value != canonical:
        raise ValueError(f"Qdrant UUID must use canonical lowercase form: {value!r}")
    return canonical


@dataclass
class Passage:
    url: str                       # the result — where the agent goes to read
    title: str
    snippet: str                   # extractive, <= ~30 words (spec §4.3)
    text: str                      # passage text — build-time input for embedding; NOT shipped
    lang: str                      # en | de | fr | it
    domain: str
    tier: str                      # A | B | C | D
    source_type: str               # official | reference | docs | entity-self
    trust_tier: float              # 0..1 ranking signal
    # --- task 274: entity/site-name suffix trafilatura strips from the title ---
    # trafilatura's metadata title drops the site-name suffix ("Gestion des
    # déchets – Dardagny" -> "Gestion des déchets"). Sensible for "Article | CNN"
    # (the suffix is the outlet), DISASTROUS for a commune/gov page where the
    # suffix IS the entity name + the key disambiguating token — the commune name
    # then lives ONLY in the domain, findable by no query token. We keep the
    # cleaned title for DISPLAY / title-centrality (unchanged; protects the "|CNN"
    # centrality math) and capture the stripped entity name here as a SEPARATE
    # searchable field folded into BOTH the dense embed and the BM25 tokens on
    # EVERY chunk (compose_index_text). Compact (ships in the redistributable dump).
    site_name: Optional[str] = None
    anchor: Optional[str] = None   # #fragment
    last_seen: Optional[str] = None
    content_digest: Optional[str] = None
    point_id: Optional[str] = None
    # --- ADR-039: per-record license + master/compact split -----------------
    # `license` is an SPDX-style string (CC-BY-SA-4.0, CC0-1.0, Apache-2.0,
    # public-domain, gov codes like SI/97-5, …). COMPACT: it ships in the
    # redistributable dump so each record self-declares its terms.
    license: Optional[str] = None
    # MASTER-ONLY (captured at ingest, STRIPPED from the compact dump): owner/
    # author attribution + the license version. Irreversible if skipped — can't
    # be reconstructed without re-fetching. Bundled under `_master` in payload()
    # and dropped wholesale by compact_payload().
    attribution: Optional[str] = None       # owner / author
    license_version: Optional[str] = None    # e.g. SE post-date -> 2.5/3.0/4.0
    master: Optional[dict] = None            # extra master-only fields
    # --- task 312: outlink capture, folded into the extraction funnel -------
    # `[{"url": <canonical_url()-normalized target>, "nofollow": bool}, ...]`,
    # deduped, attached IDENTICALLY to every chunk of one page (same pattern as
    # title/site_name, task 274 bug #2). Populated only when process_html is
    # called with meta["outlinks_column"]=True (see process_html); None for
    # every existing caller, so this is additive-only. Corpus/master-only —
    # MASTER_ONLY_KEYS strips it from the compact redistributable dump (ADR-039):
    # not part of the CLAUDE.md "minimum content" index fields.
    outlinks: Optional[list] = None
    # --- task 340: I2 rule-epoch stamps (specs/epochs.md §1) ----------------
    # `{"chunk": 2, ...}` — which rule VERSION produced this record's derived
    # values. Populated only on paths that run a non-LIVE epoch (today: the
    # versioned structured chunker, `AW_CHUNKER=structured`); None everywhere
    # else, so the
    # payload of every existing caller is byte-identical. Compact (a
    # redistributor must be able to tell which chunking rule a record carries).
    stamps: Optional[dict] = None

    def __post_init__(self):
        text_hash = sha1(self.text)
        if self.content_digest is None:
            # page-level digest (CC supplies one) is for cross-crawl dedup;
            # absent that, fall back to the chunk text hash.
            self.content_digest = text_hash
        if self.point_id is None:
            # stable UUID per CHUNK (text-based) so chunks of one page don't
            # collide. Qdrant requires int or UUID point ids.
            self.point_id = mint_passage_point_id(self.url, self.anchor, self.text)

    def payload(self) -> dict:
        """Stored (master) payload — minimum to rank + locate; excludes raw
        `text`. Master-only fields (attribution / license_version / `master`
        extras) are bundled under a single `_master` key so the compact-dump
        export can strip them wholesale via compact_payload() (ADR-039). The
        compact field `license` stays top-level."""
        d = asdict(self)
        d.pop("text", None)
        # collect master-only fields into one `_master` bundle
        master = dict(d.pop("master", None) or {})
        for k in ("attribution", "license_version"):
            v = d.pop(k, None)
            if v is not None:
                master[k] = v
        out = {k: v for k, v in d.items() if v is not None}
        if master:
            out["_master"] = master
        return out


@dataclass(frozen=True)
class ChunkTextChannels:
    """One canonical chunk and its optional explicit retrieval surfaces.

    ``dense_text``/``sparse_text`` are absent for chunk@1 so recurring legacy
    output stays byte-compatible and downstream ``record_text_channels`` keeps
    providing its historical fallback.  For chunk@2, ``text`` and
    ``sparse_text`` are the canonical full-hierarchy surface while
    ``dense_text`` is local-heading plus body.
    """
    text: str
    dense_text: Optional[str]
    sparse_text: Optional[str]
    stamps: Optional[dict]


def passage_from_chunk(unit: ChunkTextChannels, **fields) -> Passage:
    """Construct a Passage and retain explicit channels without schema drift.

    The channel attributes are intentionally sidecars rather than dataclass
    fields: existing legacy callers using ``asdict(Passage(...))`` must not
    start serializing new null keys merely because chunk@2 exists.
    """
    passage = Passage(text=unit.text, stamps=unit.stamps, **fields)
    if unit.dense_text is not None or unit.sparse_text is not None:
        if not unit.dense_text or not unit.sparse_text:
            raise ValueError("explicit text channels must be a non-empty pair")
        passage.dense_text = unit.dense_text
        passage.sparse_text = unit.sparse_text
    return passage


def passage_channel_fields(passage: Passage) -> dict[str, str]:
    """Return a validated explicit channel pair, or nothing for legacy rows."""
    dense = getattr(passage, "dense_text", None)
    sparse = getattr(passage, "sparse_text", None)
    if dense is None and sparse is None:
        return {}
    if not isinstance(dense, str) or not dense.strip():
        raise ValueError("passage dense_text must be a non-empty string")
    if not isinstance(sparse, str) or not sparse.strip():
        raise ValueError("passage sparse_text must be a non-empty string")
    if sparse != passage.text:
        raise ValueError("passage sparse_text must equal canonical text")
    return {"dense_text": dense, "sparse_text": sparse}


def passage_corpus_record(passage: Passage) -> dict:
    """Full build-corpus row, preserving legacy bytes and explicit channels."""
    record = asdict(passage)
    record.update(passage_channel_fields(passage))
    return record


def passage_source_record(passage: Passage) -> dict:
    """Minimal payload plus build text surfaces for source JSONL writers."""
    record = passage.payload()
    record["text"] = passage.text
    record.update(passage_channel_fields(passage))
    # payload() already carries point_id; assign explicitly as an assertion of
    # this source-row contract and to keep the field adjacent to build text.
    record["point_id"] = passage.point_id
    return record


# --- ADR-039 compact-vs-master dump contract --------------------------------
# The single enforcement point for the redistributable ("compact") dump. Any
# key here is MASTER-ONLY and never ships in the compact export. `_master` is
# the canonical bundle (Passage.payload() + the N2/N3 raw-dict ingesters emit
# it); the loose top-level names are belt-and-suspenders for records that stamp
# owner/attribution directly. `license` is deliberately NOT here — it is the
# one compact license field and must survive the strip.
MASTER_ONLY_KEYS = frozenset({
    "_master", "attribution", "owner", "author", "license_version",
    # task 1391 / ADR-124: which source family built the row. A master-side audit key
    # (tranche order, family-level licence audit); never part of the redistributable dump.
    "family_id",
    # task 312: outlinks are corpus/rebuild fuel (re-crawl frontier, future
    # endorsement re-derivation from L0), not a ranking/locating field — keep
    # them out of the redistributable dump, same master-only treatment as L0
    # raw HTML itself (architecture.md §4.2: "L0 is master-only, never
    # redistributed").
    "outlinks",
})


def compact_payload(payload: dict) -> dict:
    """Project a stored (master) payload onto the compact redistributable form
    (ADR-039): drop every master-only field (owner/attribution + license_version,
    whether nested under `_master` or stamped loose), keep `license` and all
    ranking/locating fields. Idempotent; safe on records that never carried
    master fields (returns them unchanged minus nothing)."""
    return {k: v for k, v in payload.items() if k not in MASTER_ONLY_KEYS}


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def extractive_snippet(text: str, max_words: int = 25) -> str:
    """First sentence(s) up to max_words — cheap, build-time (spec §4.3:
    never an LLM summary). Default 25 (ADR-027 / task 307 preflight
    resolution: "keep the snippet, standardize at 25 words" — was 30,
    A5's ≤25-word-every-record requirement was unenforced until this fix;
    see specs/aw_b1-preflight.md §4)."""
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    if len(words) <= max_words:
        return text
    snip = " ".join(words[:max_words])
    # trim to last sentence boundary if one is reasonably close
    m = list(re.finditer(r"[.!?]", snip))
    if m and m[-1].end() > len(snip) * 0.5:
        return snip[: m[-1].end()]
    return snip + "…"


from cytir.core.prose_guard import binary_content as _binary_content  # noqa: E402 -- task 767


def low_quality(text: str) -> bool:
    """Content-quality gate (the impactia.org lesson: WP attachment pages / CC
    captures can leak image EXIF/JPEG binary as 'text'). True ⇒ drop. Shared by
    every ingester (cc_ingest, the retired small-site ingester (task 1433), …) so junk never reaches the index."""
    head = text[:1000]
    if any(m in head for m in ("ExifII", "8BIM", "ICC_PROFILE", "JFIF", "Photoshop", "Ducky")):
        return True
    if _binary_content(text):       # task 767: a PDF read as text is alphabetic mojibake
        return True
    sample = text[:600]
    printable = sum(c.isalpha() or c.isspace() or c in ".,;:!?'\"-()" for c in sample)
    return printable / max(len(sample), 1) < 0.6


# --- extraction-junk gate (task 097: the wave-0 fedpol.report lesson) -------
# JS-shell sites leak their embedded JSON search-index / breadcrumb dump into
# CC captures as the page "content"; trafilatura passes it through verbatim and
# it ranks absurdly high on generic terms (fedpol.report polluted 20% of eval
# queries, 2026-07-04). Same class: bare sitemap URL lists (southampton.ac.uk
# sitemapxml.page), separator-soup nav crumbs, and asset-shell i18n
# placeholders (BFS "Momentan können keine Daten angezeigt werden…").
# NOTE (calibration, task 097): naive brace/quote *density* does NOT catch the
# JSON dumps — huge prose values dilute the structural chars below legit-corpus
# levels. The reliable signals are the JSON envelope + "key": pairs and the
# escaped `\/` permalink sequences.

_JSON_KEY_RE = re.compile(r'"[\w\-. ]{1,60}"\s*:')
_URL_TOKEN_RE = re.compile(r"(?i)^(?:https?://|www\.)")
# task 2050 / 2044: a host-shaped query ("ethz.ch", "www.ethz.ch", "fedlex.admin.ch") is
# one name to the user but tokenizes to {label, tld}; the TLD token is in nearly every
# page, so it must not count toward the df=0 zero-signal guard. Short curated list of
# the TLDs the allowlist actually holds; an unlisted TLD means "not host-shaped" and
# the query takes the unchanged path. Used only behind AW_HOST_QUERY_CARD.
HOST_QUERY_TLDS = frozenset({
    "ch", "de", "fr", "it", "at", "li", "eu", "com", "org", "net", "edu", "gov", "int",
    "info", "io", "uk", "us", "swiss"})
_HOST_QUERY_RE = re.compile(r"^(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+$")


def host_query_parts(q: str):
    """(label, tld) when `q` is a bare host ("www.ethz.ch" -> ("ethz", "ch"); "fedlex.admin.ch"
    -> ("fedlex", "ch")), else None. The label is the leftmost label after any "www."."""
    h = (q or "").strip().lower()
    if not _HOST_QUERY_RE.match(h):
        return None
    labels = h[4:].split(".") if h.startswith("www.") else h.split(".")
    if len(labels) < 2 or labels[-1] not in HOST_QUERY_TLDS:
        return None
    return labels[0], labels[-1]


def url_host_label(url: str) -> str:
    """The same label for a URL: host without "www.", leftmost label ("https://www.ethz.ch/" -> "ethz")."""
    h = (url or "").split("://", 1)[-1].split("/", 1)[0].split("?", 1)[0].split(":", 1)[0].lower()
    parts = (h[4:] if h.startswith("www.") else h).split(".")
    return parts[0] if len(parts) >= 2 else ""


_ALPHA_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)

# i18n empty-state / license placeholders seen on JS/asset shells (BFS wave-1
# diag pages). Lowercase, straight apostrophes; matching requires the marker
# AND essentially nothing else, so an unseen variant here is harmless.
SHELL_PLACEHOLDERS = (
    "momentan können keine daten angezeigt werden",
    "aucune donnée ne peut être affichée",
    "al momento non è possibile visualizzare i dati",
    "no data can be displayed at the moment",
    "freie nutzung. quellenangabe ist pflicht",
    "kommerzielle nutzung nur mit bewilligung des datenlieferanten zulässig",
    "utilisation libre. obligation d'indiquer la source",
    "utilisation commerciale uniquement avec l'autorisation du fournisseur des données",
    "libero utilizzo. indicazione della fonte obbligatoria",
    "utilizzo commerciale soltanto con il consenso del fornitore dei dati",
    "open use. must provide the source",
    "use for commercial purposes requires permission of the data owner",
)


def extraction_junk(text: str) -> Optional[str]:
    """Machine-content gate: reason string if `text` is extraction leakage
    rather than prose, else None. Applied by _content_passages to the whole
    extracted page AND to every chunk (a JSON dump split mid-chunk no longer
    starts with '[{'). Reasons: binary_content | json_dump | url_list | breadcrumb | no_prose |
    shell_boilerplate — counted per ingest run as `extraction_rejects`."""
    t = text.strip()
    if not t:
        return "no_prose"
    # 0) a binary file read as text (task 767: raw PDF streams on weko/bazl.admin.ch)
    if _binary_content(t):
        return "binary_content"
    # 1) embedded JSON search-index / breadcrumb dump
    if t[0] in "[{" and len(_JSON_KEY_RE.findall(t[:2000])) >= 3:
        return "json_dump"
    if t.count("\\/") >= 20:          # \/escaped\/permalinks — JSON mid-chunk
        return "json_dump"
    # 2) bare URL list (sitemap page rendered as text)
    toks = t.split()
    if len(toks) >= 10:
        url_toks = sum(1 for x in toks if _URL_TOKEN_RE.match(x))
        if url_toks / len(toks) >= 0.5:
            return "url_list"
    # 3) separator soup / no real words (nav crumbs, whitespace-comma shells)
    words = _ALPHA_WORD_RE.findall(t)
    if len({w.lower() for w in words}) < 5:   # distinct: "DE , , DE ," shells
        return "no_prose"
    # markdown table rows ('| a | b |') are legit data, not crumbs — exclude
    # them from the separator count (fedpol RIPOL stats table lesson).
    sep_src = "\n".join(l for l in t.splitlines() if not l.lstrip().startswith("|"))
    seps = sum(sep_src.count(c) for c in "/|>»›")
    if seps / len(words) >= 3.0:      # legit corpus max ≈1.4 (swissauth)
        return "breadcrumb"
    # 4) shell i18n placeholder is (almost) all there is
    norm = re.sub(r"\s+", " ", t.lower()).replace("’", "'")
    found = False
    for marker in SHELL_PLACEHOLDERS:
        if marker in norm:
            found = True
            norm = norm.replace(marker, " ")
    if found and len(_ALPHA_WORD_RE.findall(norm)) < 15:
        return "shell_boilerplate"
    return None


# ---------------------------------------------------------------------------
# Listing-shell gate (task 167) — tag / category / archive / pagination pages.
#
# Blog-platform taxonomy/nav pages (`/tags/<x>`, `/category/<x>`, `/archive/`,
# `/page/N`, `?page=N`) are near-identical SHELLS: each points at ONE underlying
# article and carries that article's copied snippet, so one blog post can occupy
# many top-10 slots via its tag taxonomy (Tim hit mlflow.org: 9/10 results were
# `/articles/tags/<tag>` "One post tagged with …" shells). ~69% of mlflow.org's
# points were such shells.
#
# extraction_junk() does NOT catch these — they DO carry prose (the copied
# article snippet). The signal is the URL path pattern PLUS a listing-boilerplate
# TITLE. Path alone is insufficient (github.com/topics, gov "/topics/" prose
# hubs are real content), so the ingest gate requires BOTH. The offline audit
# additionally confirms via snippet near-duplication across siblings.
# ---------------------------------------------------------------------------

# Path-segment patterns for listing/taxonomy/pagination pages. `/tags?`, etc.
# anchor on a full path segment (`(?:/|$)`) so `/tagliatelle`, `/topical` etc.
# do NOT match.
_LISTING_PATH_RES = (
    ("tag",      re.compile(r"/tags?(?:/|$)", re.I)),
    ("category", re.compile(r"/categor(?:y|ies)(?:/|$)", re.I)),
    ("label",    re.compile(r"/labels?(?:/|$)", re.I)),   # Blogger taxonomy
    ("archive",  re.compile(r"/archives?(?:/|$)", re.I)),
    ("topic",    re.compile(r"/topics?(?:/|$)", re.I)),    # some are real hubs
    ("page",     re.compile(r"/page/\d+/?$", re.I)),       # /page/3 pagination
    # /newnsb — the Bund-CMS "news service bund" rolling news-LIST artifact
    # (task 349 / probe 2026-07-28-postcut): admin.ch-family sites expose
    # /de/newnsb etc. as an ever-churning teaser list ("News · 2951
    # Ergebnisse"), identical boilerplate across ≥12 subdomains. The segment
    # only exists on that CMS, but the 167 conjunction still applies — the
    # title must be a bare news-list label (see _NEWSLIST_TITLE_PARTS).
    ("newslist", re.compile(r"/newnsb(?:/|$)", re.I)),
)
_LISTING_QUERY_RE = re.compile(r"(?:^|[?&])(?:page|paged)=\d+\b", re.I)

# Bare news-list labels (whole separator-delimited title part) accepted as the
# title signal for the `newslist` path tag ONLY — a generic "News" title on any
# other path is a real page, never a shell verdict.
_NEWSLIST_TITLE_PARTS = frozenset({
    "news", "aktuell", "aktuelles", "news und medienmitteilungen",
    "medienmitteilungen", "actualité", "actualités", "attualità",
    "comunicati stampa", "communiqués", "communiqués de presse",
})

# Listing-boilerplate title signatures. High-precision: "One post tagged with X",
# "5 posts tagged", "Tagged with X", "Category: X", "Archive for X", "Page 2 of 9".
_LISTING_TITLE_RES = (
    re.compile(r"\b\w+\s+posts?\s+tagged\b", re.I),   # "One/5/N posts tagged"
    re.compile(r"^\s*posts?\s+tagged\b", re.I),
    re.compile(r"\btagged\s+with\b", re.I),
    # "Category: X" / "Archive : 05/10/2022". COLON only — the dash variant
    # ("Archives - Guide de la détermination…", canada.ca) titles REAL archived
    # content pages and must NOT match (task 167 audit false-positive).
    re.compile(r"(?:^|\|)\s*(?:category|tag|archive|archives|label|topics?)\s*:",
               re.I),
    re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.I),   # "Page 2 of 9"
    re.compile(r"\barchives?\s+for\b", re.I),          # "Archives for June 2025"
)


# ---------------------------------------------------------------------------
# Sitemap/link-dump extension (task 184) — the evernote.com/sitemap class.
#
# HTML sitemap pages ("Sitemap", "Plan du site", "Inhaltsverzeichnis", A-Z
# indexes) are giant navigation link dumps that weakly match everything: the
# anchor texts concatenate into prose-looking text, so extraction_junk passes
# them (evernote.com/sitemap: 18,964 anchors, anchor-text ratio 0.994, and it
# ranked top-5 on three unrelated live queries). Same conjunction design as
# 167: URL signal alone is NOT a drop (mural.co/templates/sitemap is a real
# product page; docs.djangoproject.com/…/sitemaps/ is the framework doc), and
# a "…sitemap…" title alone is NOT a drop (Wikipedia's "Sitemaps" article,
# "Sitemap Checker | BlazeHive"). Apache autoindex pages ("Index of /<path>")
# are the one title-alone drop: that exact title prefix is server boilerplate.
# ---------------------------------------------------------------------------

_SEG_EXT_RE = re.compile(r"\.(?:html?|php|aspx?|shtml|xml|jsp|cfm)$", re.I)
# any path segment containing sitemap / site-map / site_map (loose on purpose —
# the boilerplate-title conjunction provides the precision)
_SITEMAP_SEG_RE = re.compile(r"(?:^|[\w-]*[-_])?site[-_]?maps?(?:[-_][\w-]*)?$",
                             re.I)
# exact-segment names for the non-"sitemap" members of the class
_SITEMAP_SEG_EXACT = frozenset({
    "site-index", "site_index", "siteindex",
    "plan-du-site", "plan_du_site", "plandusite", "plan-site", "plan_site",
    "inhaltsverzeichnis", "seitenuebersicht",
    "mappa-del-sito", "mappa_del_sito",
    "a-z", "a-to-z", "atoz", "az-index", "azindex", "a-z-index",
    "index-of", "index_of",
})

# Boilerplate sitemap titles, matched as a whole separator-delimited title part
# ("U.S. Access Board - Site Map" → part "site map"; "Sitemap | Evernote" →
# part "sitemap"). NOT plain "sitemaps" (Wikipedia article) and NOT plain
# "a-z" (real glossaries title entries "Evolution - A-Z").
_SITEMAP_TITLE_PARTS = frozenset({
    "sitemap", "site map", "site-map", "the sitemap", "html sitemap",
    "sitemap page", "sitemap & index", "xml sitemap index",
    "site index", "a-z index", "a to z index",
    "plan du site", "plan de site",
    "inhaltsverzeichnis", "inhaltsübersicht", "inhaltsuebersicht",
    "seitenübersicht", "seitenuebersicht",
    "mappa del sito",
})
# generic-title parts accepted ONLY when the last path segment is literally a
# sitemap page (…/sitemap, …/sitemap.html, …/sitemap_node.html): real prose
# pages titled "Übersicht"/"Inhalt" exist, but not at that exact URL.
_SITEMAP_TITLE_GENERIC = frozenset({
    "übersicht", "uebersicht", "inhalt", "website navigation",
})
# title ENDS with a sitemap word ("Elastic Sitemap", "PR Newswire Site Map") —
# also only honored with an exact sitemap last segment.
_SITEMAP_TITLE_SUFFIX_RE = re.compile(r"\bsite[\s_-]?map$", re.I)
_TITLE_SEP_RE = re.compile(r"\s*(?:\||—|–|::|·|•|»|›|>>|//|\s-\s)\s*")
# Apache/nginx autoindex boilerplate — title-alone signal.
_AUTOINDEX_TITLE_RE = re.compile(r"^\s*index of /", re.I)


def _path_segs(path: str) -> list[str]:
    from urllib.parse import unquote
    return [unquote(s).lower() for s in (path or "").split("/") if s]


def _sitemap_url(path: str) -> Optional[str]:
    """'sitemap' if a path segment is sitemap-family, 'sitemap-exact' if the
    LAST effective segment is literally a sitemap page. MediaWiki article
    paths (/wiki/…) and glossary trees (MDN /Glossary/Site_map) are excluded —
    encyclopedia/glossary articles ABOUT sitemaps live there."""
    segs = _path_segs(path)
    if not segs or "wiki" in segs or "glossary" in segs or "glossar" in segs \
            or "glossaire" in segs:
        return None
    hit = None
    for s in segs:
        base = _SEG_EXT_RE.sub("", s)
        if _SITEMAP_SEG_RE.match(base) or base in _SITEMAP_SEG_EXACT:
            hit = "sitemap"
    last = _SEG_EXT_RE.sub("", segs[-1])
    if last in ("sitemap", "sitemap_node") or (
            last == "index" and len(segs) >= 2
            and _SEG_EXT_RE.sub("", segs[-2]) == "sitemap"):
        return "sitemap-exact"
    if hit and last in _SITEMAP_SEG_EXACT and last in (
            "a-z", "a-to-z", "atoz"):
        return "az-exact"
    return hit


def _sitemap_title(title: str, exact_url: bool, az_url: bool) -> bool:
    t = " ".join((title or "").split()).lower()
    if not t:
        return False
    parts = [p.strip(" :.") for p in _TITLE_SEP_RE.split(t)]
    if any(p in _SITEMAP_TITLE_PARTS for p in parts):
        return True
    if exact_url:
        if any(p in _SITEMAP_TITLE_GENERIC for p in parts):
            return True
        if _SITEMAP_TITLE_SUFFIX_RE.search(t):
            return True
    if az_url and re.search(r"\ba[ -]?(?:to[ -]?)?z\b", t):
        # ".../a-z" page titled "… A-Z" / "X A to Z" ("Public holidays A-Z",
        # "UNC A to Z"); safe because the last segment is itself /a-z.
        return True
    return False


def listing_shell_url(url: str) -> Optional[str]:
    """Return a listing-shell pattern tag (tag/category/label/archive/topic/
    page/query-page/sitemap) if the URL path or query is a blog-platform
    taxonomy, pagination, or sitemap/link-dump listing, else None. Path/pattern
    signal ONLY — never a drop decision on its own (see `listing_shell`)."""
    try:
        pr = urlparse(url or "")
    except Exception:  # noqa: BLE001
        return None
    path = pr.path or ""
    for tag, rx in _LISTING_PATH_RES:
        if rx.search(path):
            return tag
    if _LISTING_QUERY_RE.search(pr.query or ""):
        return "query-page"
    if _sitemap_url(path):
        return "sitemap"
    return None


def listing_shell_title(title: str) -> bool:
    """True if `title` is listing/taxonomy boilerplate ("One post tagged with
    …", "Category: …", "Page N of M", "Archives for …")."""
    if not title:
        return False
    return any(rx.search(title) for rx in _LISTING_TITLE_RES)


def listing_shell(url: str, title: str) -> Optional[str]:
    """Ingest gate (task 167 + 184): reason string when the page is a
    navigation shell.

    - 'listing_shell' — listing/taxonomy/pagination path AND listing-
      boilerplate title (167). BOTH required — path alone hits real content
      hubs (github.com/topics, gov "/topics/" prose pages).
    - 'sitemap_shell' — sitemap-family path AND sitemap-boilerplate title
      (184, the evernote.com/sitemap class). BOTH required — mural.co
      /templates/sitemap and Wikipedia's "Sitemaps" article each carry only
      one signal and are real content.
    - 'autoindex_shell' — title is Apache/nginx autoindex boilerplate
      ("Index of /<path>"), a server-generated directory listing (184);
      title-alone, it is never a human-written page title.
    """
    if _AUTOINDEX_TITLE_RE.match(title or ""):
        return "autoindex_shell"
    ltag = listing_shell_url(url)
    if ltag == "newslist":
        # conjunction for the Bund-CMS news list: /newnsb path + a bare
        # news-list label title ("News", "Aktuell", …). The generic
        # _LISTING_TITLE_RES don't know these labels, and adding a bare
        # /news/i there would poison the other path tags.
        parts = [p.strip().lower() for p in _TITLE_SEP_RE.split(title or "")]
        if any(p in _NEWSLIST_TITLE_PARTS for p in parts if p):
            return "listing_shell"
    elif ltag and listing_shell_title(title or ""):
        return "listing_shell"
    try:
        path = urlparse(url or "").path or ""
    except Exception:  # noqa: BLE001
        return None
    smu = _sitemap_url(path)
    if smu and _sitemap_title(title or "", exact_url=(smu == "sitemap-exact"),
                              az_url=(smu == "az-exact")):
        return "sitemap_shell"
    return None


# ---------------------------------------------------------------------------
# task 274 — put the title (+ entity/site name) into BOTH vectors, EVERY chunk.
#
# Bug: index_bakeoff embedded the dense vector from the chunk BODY only and
# computed BM25 from the body only; `title`/`site_name` were payload-only fields
# used solely by the title-centrality RE-RANK signal (ADR-031) — which fires only
# AFTER a page is already in the candidate pool. For a thin-body or vocab-mismatch
# chunk (dardagny's "objets encombrants" chunk never says "Dardagny"; the commune
# name lives only in the domain) the page never ENTERS the pool, so the title
# never helps retrieval at all. Prepending the title + site_name to the embed +
# BM25 input on EVERY chunk rescues every thin-body page — the title is the
# highest-signal page summary and the site_name is the disambiguating entity token.
#
# Dedup-guarded: a head token already present verbatim in the body is not repeated
# (byte-parity where the title is already redundant with the text). entity_card
# records are EXEMPT — ADR-048 deliberately carries the name in the SPARSE leg and
# the multi-facet profile in DENSE; prepending the name to a card's dense vector
# would regress that split. Index- and query-side agree because only the DOCUMENT
# (index) side composes; the query is the user's text unchanged.
# ---------------------------------------------------------------------------

def compose_index_text(rec: dict) -> str:
    """Build the DENSE-embed + BM25 input for a content passage (task 274):
    `"<title>. <site_name>. <body>"`, with a head token skipped when it is
    already present in the body (no double-count). Returns the body unchanged for
    entity_card records (ADR-048 exemption) or when there is no title/site_name."""
    text = rec.get("text") or ""
    if rec.get("source_type") == "entity_card":
        return text
    title = (rec.get("title") or "").strip()
    site = (rec.get("site_name") or "").strip()
    body_lc = text.lower()
    head_parts = []
    for part in (title, site):
        if part and part.lower() not in body_lc \
                and part.lower() not in " ".join(head_parts).lower():
            head_parts.append(part)
    if not head_parts:
        return text
    return ". ".join(head_parts) + ". " + text if text else ". ".join(head_parts)


def record_text_channels(rec: dict) -> tuple[str, str]:
    """Return explicit ADR-060 dense/sparse text with legacy fallback.

    Embedders must share this selection contract: a card carrying distinct
    channels must never silently collapse back to its legacy `text` field.
    """
    dense = rec.get("dense_text") or rec.get("text")
    sparse = rec.get("sparse_text") or rec.get("text")
    if not dense or not sparse:
        raise ValueError("record has no complete dense/sparse text channels")
    return dense, sparse


def empty_chunk_quarantine(rec: dict) -> dict | None:
    """Describe a provably empty materialized chunk, else return ``None``.

    Task 389 found five zero-word trailing chunks in the 19,227,122-row
    Wikipedia materialization.  They contain no retrievable content and cannot
    be embedded, but they must remain visible in row-conservation ledgers.

    This is deliberately narrower than ``not text``: the producer's text,
    word-count, and digest signals must all agree, and the locating URL must be
    intact.  Any other incomplete record remains a hard failure.
    """
    if rec.get("text") or "":
        return None
    if not (rec.get("url") or "").strip():
        return None
    if rec.get("chunk_n_words") != 0:
        return None
    if rec.get("content_digest") != hashlib.sha256(b"").hexdigest():
        return None
    return {
        "reason": "materializer emitted a zero-word chunk",
        "url": rec.get("url"),
        "point_id": rec.get("point_id"),
        "chunk_ordinal": rec.get("chunk_ordinal"),
        "chunk_unit": rec.get("chunk_unit"),
        "chunk_rule": rec.get("chunk_rule"),
        "content_digest": rec.get("content_digest"),
    }


# ---------------------------------------------------------------------------
# The allowlist tier -> (source_type, trust) map. ONE home (task 548).
#
# This table shipped as three byte-identical copies -- `fetch_corpus.py`,
# `cc_ingest.py`, `fetch_hn_cc.py` -- which is a forked contract by definition:
# identical today, divergent the first time one is edited (engineering doctrine
# 1). It decides what a host's rows CLAIM ABOUT THEMSELVES, so a fork here is a
# fork in what the index asserts.
#
# NOTE THE NAME COLLISION, because it has already misled a reading of the
# payload: this "tier" is the ALLOWLIST'S CURATION tier (A/B/C/D, an editorial
# judgement about a source), while the `tier` FIELD in a projected payload is
# the IDENTITY tier from `aw_b1_identity.classify_tier` (A = source-native
# anchor, B = digest-addressed, C = card). `impactia.org` is allowlist tier D
# and carries payload `tier: "B"`; neither value is wrong and they are not the
# same quantity.
TIER_TRUST = {"A": 1.0, "B": 0.82, "C": 0.72, "D": 0.5}
TIER_SOURCE = {"A": "official", "B": "reference", "C": "docs", "D": "entity-self"}
TIER_TRUST_DEFAULT = 0.6
TIER_SOURCE_DEFAULT = "official"


def tier_stamps(tier: str) -> dict:
    """`{"tier", "trust_tier", "source_type"}` for an allowlist curation tier."""
    return {"tier": tier,
            "trust_tier": TIER_TRUST.get(tier, TIER_TRUST_DEFAULT),
            "source_type": TIER_SOURCE.get(tier, TIER_SOURCE_DEFAULT)}


def allowlist_entry(domain: str, path=None) -> Optional[dict]:
    """This domain's entry in `data/allowlist.yaml`, or None.

    The allowlist is the REGISTER: it is where a curation decision about a
    source is recorded and, per the project's legislative rule, changed only by
    a PR the founder merges. An ingester that takes its stamps from a command
    line instead is not reading the register -- it is overruling it, silently
    and per invocation.

    MEASURED (task 548): `impactia.org` is declared `tier: D` -- "Tier-D
    entity-homepage example. Authoritative only about itself." -- and the twin
    serves its 145 rows as `source_type: official`, `trust_tier: 1.0`, because
    a historical run of the retired small-site ingester (task 1433), `--tier A --source-type official --trust 1.0`,
    run said so. The epoch-4 re-materialisation read the register and produced
    `entity-self` / 0.5; the whole 146 -> 20 trust drift is those two answers
    swapping places, with no ruling anywhere in between.
    """
    import yaml
    p = pathlib.Path(path) if path else (
        _data_path("allowlist.yaml"))
    if not p.exists():
        return None
    doc = yaml.safe_load(p.read_text()) or {}
    d = (domain or "").lower().removeprefix("www.")
    for entry in doc.get("allowlist") or []:
        if str(entry.get("domain", "")).lower().removeprefix("www.") == d:
            return entry
    return None


def resolve_register_stamps(domain, *, allowlist, tier=None, source_type=None,
                            trust=None, override_reason=""):
    """`(tier, source_type, trust, basis)` -- the REGISTER first.

    `data/allowlist.yaml` is where a curation decision about a source is
    recorded, and per the project's legislative rule it changes only by a PR
    the founder merges. The retired small-site ingester (task 1433) used to
    take all three stamps from defaults on its own command line (`--tier A
    --source-type official --trust 1.0`), which is how impactia.org came to
    serve 145 rows at `official` / 1.0 while its register entry said `tier: D`
    (task 548).

    So: the register decides unless the caller says otherwise IN WORDS. A
    contradicting `tier`/`source_type`/`trust` without `override_reason` is
    refused (SystemExit, as the old ingester did), and the reason lands in the
    returned `basis` for the receipt. `allowlist` is the register path (falsy
    = no register: explicit stamps only). No caller exists yet; this is the
    one function that enforces "the register decides unless overridden in
    words" for the next site-family adapter (task 1433, slice A of 1430).
    """
    entry = allowlist_entry(domain, allowlist) if allowlist else None
    explicit = {k: v for k, v in
                (("tier", tier), ("source_type", source_type),
                 ("trust", trust)) if v is not None}
    if entry is None:
        if not explicit:
            raise SystemExit(
                f"REFUSED: {domain} has no entry in {allowlist} and no "
                "--tier/--source-type/--trust was given. An unregistered "
                "source has no curation tier to stamp; add it to the register "
                "(a PR Tim merges) or state the stamps explicitly.")
        return (explicit.get("tier", "A"),
                explicit.get("source_type", "official"),
                explicit.get("trust", 1.0),
                "explicit (domain not in the register)")

    reg = tier_stamps(str(entry.get("tier")))
    if not explicit:
        return reg["tier"], reg["source_type"], reg["trust_tier"], \
            f"register {allowlist} tier={reg['tier']}"

    want = {"tier": explicit.get("tier", reg["tier"]),
            "source_type": explicit.get("source_type", reg["source_type"]),
            "trust": explicit.get("trust", reg["trust_tier"])}
    contradicts = {k: (reg[{"trust": "trust_tier"}.get(k, k)], v)
                   for k, v in want.items()
                   if v != reg[{"trust": "trust_tier"}.get(k, k)]}
    if contradicts and not override_reason:
        raise SystemExit(
            f"REFUSED: {domain} is registered as tier {reg['tier']} "
            f"({reg['source_type']} / {reg['trust_tier']}) and this invocation "
            f"contradicts it: {contradicts}. The register is changed by a PR "
            "Tim merges, not by a command line. Pass --override-register REASON "
            "to stamp differently anyway, and the reason goes in the receipt.")
    return (want["tier"], want["source_type"], want["trust"],
            (f"OVERRIDE of register tier={reg['tier']}: "
             f"{override_reason}") if contradicts
            else f"register {allowlist} tier={reg['tier']} (restated)")

# --- task 2333: tokenizer / HTML extract / chunk dispatch MOVED out of this module.
# Lazy PEP-562 shim so `from cytir.core.common import <moved name>` keeps working without
# dragging trafilatura/bs4/the chunker into every process that imports core. Callers
# inside `cytir/` import the new homes directly; pipeline/ and tests use this shim for one
# round. REMOVE once `grep -rn "cytir.core.common import" pipeline tests` names none of these.
# `_FOLD_ACCENTS` is deliberately NOT forwarded: the flag lives in `cytir.core.bm25_tokenizer`;
# `require_index_fold` raises if anything still writes `cytir.core.common._FOLD_ACCENTS`.
_MOVED = {
    "BOILERPLATE": "cytir.extract.html_extract",
    "CHUNKER_MODES": "cytir.chunk.dispatch",
    "ChunkerSelection": "cytir.chunk.dispatch",
    "DATE_PARSE_FAILED": "cytir.extract.html_extract",
    "TOK_EPOCH_FOLD": "cytir.core.bm25_tokenizer",
    "_APOS_RE": "cytir.core.bm25_tokenizer",
    "_CHUNKER_ANNOUNCED": "cytir.chunk.dispatch",
    "_ENTITY_SOURCE_TYPES": "cytir.extract.html_extract",
    "_GENERIC_SITE_LABELS": "cytir.extract.html_extract",
    "_LEGACY_CHUNK_EPOCH": "cytir.chunk.dispatch",
    "_LEGACY_CHUNK_LEGEND": "cytir.chunk.dispatch",
    "_METADATA_LOCK": "cytir.extract.html_extract",
    "_MISPARSE_FORBIDDEN_ANCESTORS": "cytir.extract.html_extract",
    "_OUTLINK_SINK": "cytir.extract.html_extract",
    "_REPAIR_MIN_DOM_CHARS": "cytir.extract.html_extract",
    "_REPAIR_MIN_GAIN": "cytir.extract.html_extract",
    "_REPAIR_MIN_GAIN_CHARS": "cytir.extract.html_extract",
    "_REPAIR_THIN_RATIO": "cytir.extract.html_extract",
    "_SHA256_RE": "cytir.chunk.dispatch",
    "_TITLE_SEP_SPLIT": "cytir.extract.html_extract",
    "_TOKEN_RE": "cytir.core.bm25_tokenizer",
    "_announce_chunker": "cytir.chunk.dispatch",
    "_capture_outlinks": "cytir.extract.html_extract",
    "_chunk_policy_path": "cytir.chunk.dispatch",
    "_chunk_tokenizer_revision": "cytir.chunk.dispatch",
    "_content_passages": "cytir.extract.html_extract",
    "_decisively_longer": "cytir.extract.html_extract",
    "_decompound": "cytir.core.bm25_tokenizer",
    "_dom_text_chars": "cytir.extract.html_extract",
    "_entity_display_title": "cytir.extract.html_extract",
    "_env_truthy": "cytir.core.bm25_tokenizer",
    "_extract_content_md": "cytir.extract.html_extract",
    "_fold": "cytir.core.bm25_tokenizer",
    "_hash": "cytir.core.bm25_tokenizer",
    "_load_pinned_chunk_policy": "cytir.chunk.dispatch",
    "_not_markup": "cytir.extract.html_extract",
    "_outlink_sink_active": "cytir.extract.html_extract",
    "_parse_soup": "cytir.extract.html_extract",
    "_portable_policy_path": "cytir.chunk.dispatch",
    "_raw_html_title": "cytir.extract.html_extract",
    "_site_entity_name": "cytir.extract.html_extract",
    "bm25_tokens": "cytir.core.bm25_tokenizer",
    "chunk_markdown_channels": "cytir.chunk.dispatch",
    "chunk_markdown_units": "cytir.chunk.dispatch",
    "chunker_mode": "cytir.chunk.dispatch",
    "chunker_selection": "cytir.chunk.dispatch",
    "coerce_title": "cytir.extract.html_extract",
    "disable_outlink_capture": "cytir.extract.html_extract",
    "enable_outlink_capture": "cytir.extract.html_extract",
    "explicit_structured_chunker_selection": "cytir.chunk.dispatch",
    "extract_content_markdown": "cytir.extract.html_extract",
    "extract_metadata_dateguard": "cytir.extract.html_extract",
    "extract_outlink_context_records": "cytir.extract.html_extract",
    "extract_outlink_records": "cytir.extract.html_extract",
    "html_title": "cytir.extract.html_extract",
    "landmark_misparse": "cytir.extract.html_extract",
    "outbound_link_anchors": "cytir.extract.html_extract",
    "outbound_links": "cytir.extract.html_extract",
    "outlinks_column_enabled": "cytir.extract.html_extract",
    "parse_html_soup": "cytir.extract.html_extract",
    "passages_from_markdown": "cytir.extract.html_extract",
    "process_html": "cytir.extract.html_extract",
    "reg_domain": "cytir.extract.html_extract",
    "repair_markup": "cytir.extract.html_extract",
    "require_index_fold": "cytir.core.bm25_tokenizer",
    "resolve_title_and_site_name": "cytir.extract.html_extract",
    "site_entity_label": "cytir.extract.html_extract",
}


def __getattr__(name: str):
    mod = _MOVED.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(mod), name)


class _GuardedModule(type(sys)):
    """task 2333 review: `_FOLD_ACCENTS` moved to `cytir.core.bm25_tokenizer`. A write to THIS module
    would be a silent no-op (tokens hash at the wrong epoch), so refuse it at the write."""

    def __setattr__(self, name, value):
        if name == "_FOLD_ACCENTS":
            raise RuntimeError(
                "cytir.core.common._FOLD_ACCENTS was assigned, but the flag lives in "
                "cytir.core.bm25_tokenizer._FOLD_ACCENTS (task 2333); the write would have no effect")
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _GuardedModule
