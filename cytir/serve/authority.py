"""Source-card lookup ("find_authority") — the source-authority layer.

Queries the tiny, always-warm `source_cards` collection (2.6k cards, config B:
dense MRL-128 + bm25) and returns *authority cards*: which SOURCE is
authoritative for a topic + how to navigate/query it directly. Cards are the
answer for sources we can't or shouldn't mirror (paywalled, consent-declined,
JS-only, live data) — see specs/adr-source-authority-layer.md.

ADR-021 guard: cards live in their OWN collection and are NEVER blended into
the grounding passage pool or scored against ranked results. The /search
sidecar attaches them as a separate top-level block only.

Failure isolation: this module raises on backend errors; callers catch —
/search continues without a sidecar, /find_authority returns a clear JSON
error (never a 500). A missing `source_cards` collection therefore cannot
break the main search path.
"""
from __future__ import annotations

import re
import unicodedata
import urllib.parse

from qdrant_client import models as qm

from . import models, qdrant_io, retrieval
from .settings import SETTINGS as _S
from .schema import DENSE_VEC, SPARSE_BM25, bm25_tokens
from .signals import _minmax

CARDS_COLLECTION = _S.runtime.cards_collection

# fused = W_DENSE*minmax(dense) + W_BM25*minmax(bm25); dense carries topic
# semantics (the embedding surface is "what the source is an authority FOR"),
# bm25 rescues exact domain-name / rare-term matches.
_W_DENSE = 0.7  # fixed: was env AW_CARDS_W_DENSE (S10 D5: no product setter)
_W_BM25 = 0.3  # fixed: was env AW_CARDS_W_BM25 (S10 D5: no product setter)
# Quality floors on the RAW dense cosine (comparable across queries, unlike
# the min-max fused score). A card below the floor is dropped UNLESS the query
# names it (domain-head token match — "search the NZZ …" keeps nzz.ch). This
# kills the bm25 stopword-tie junk ("… suisse" matching every *-suisse.ch
# card at dense sim 0) without losing by-name lookups.
MIN_SIM = 0.3  # fixed: was env AW_CARDS_MIN_SIM (S10 D5: no product setter)
# Sidecar is stricter: only attach when the match is confident.
SIDECAR_FLOOR = 0.5  # fixed: was env AW_CARDS_SIDECAR_FLOOR (S10 D5: no product setter)
# Score honesty (task 2656): `score` is a per-query min-max, so the top card
# always reads ~0.7 whatever its quality. `confidence` is the RAW dense cosine
# (comparable across queries); below WEAK_MATCH_FLOOR the card is flagged
# `weak_match` unless the query names its domain. 0.45 sits in the gap between
# the 12-query probe's wrong-and-low top cards (<=0.37) and its good ones
# (>=0.478), see the task 2656 doc. Flag only: ranking
# is unchanged.
WEAK_MATCH_FLOOR = 0.45
WEAK_NOTE = ("weak match: no card is a confident authority for this query "
             f"(dense similarity < {WEAK_MATCH_FLOOR}); treat the cards as "
             "guesses, not as the authoritative source")
# fused-score bonus when the query literally names the domain (nzz -> nzz.ch)
_NAME_BOOST = 0.6  # fixed: was env AW_CARDS_NAME_BOOST (S10 D5: no product setter)
# domain heads that are generic query words — a query containing "search" or
# "news" is NOT naming search.ch / news.ch; no boost/rescue for these.
# Task 091 extension: the boost also must not fire on heads that are ordinary
# de/fr/it/en vocabulary or place names that appear incidentally in queries
# ("zug fahrplan" is about trains, not Canton Zug; "train timetable geneva
# lausanne" is not naming lausanne.ch). The name boost is for NAMES (nzz,
# zefix, sbb, comparis); topic-word domains must earn their card through the
# dense leg — if the card is genuinely authoritative for the topic, its dense
# sim clears the floor without a bypass. Set derived from a scan of all 2,621
# card domain heads (2026-07-04).
_GENERIC_HEADS = {
    # original generic query words
    "search", "news", "jobs", "local", "info", "blog", "shop",
    "web", "online", "index", "data", "open", "swiss", "suisse",
    "schweiz", "svizzera", "map", "maps", "wiki", "portal",
    # de common nouns colliding with card domains
    "zug",          # de train — zug.ch (Canton Zug); the "zug fahrplan" bug
    "post",         # mail — post.ch
    "zeitung",      # newspaper — zeitung.ch
    "anzeiger",     # gazette — anzeiger.ch
    "mietrecht",    # tenancy law — mietrecht.ch (fired on a legal control at sim .21)
    "strom",        # electricity — strom.ch
    "arbeit",       # work — arbeit.swiss
    "abfall",       # waste — abfall.ch
    "bienen",       # bees — bienen.ch
    "schulden",     # debts — schulden.ch
    "alimente",     # alimony — alimente.ch
    "laerm", "lärm",  # noise — laerm.ch / lärm.ch
    "hagel",        # hail — hagel.ch
    "armee",        # army — armee.ch
    "kath",         # kath. (catholic) — kath.ch
    "fokus",        # focus — fokus.swiss
    "bilanz",       # balance sheet — Bilanz magazine
    "monster",      # monster.com (jobs) — de/en common noun
    "gaudi",        # de slang fun / the architect — gaudi.ch
    # fr / it
    "meteo",        # weather — meteo.ch
    "bail",         # fr lease — bail.ch
    "bulle",        # fr bubble / town of Bulle — bulle.ch
    "bilan",        # fr balance sheet — Bilan magazine
    "revue",        # fr/de journal — revue.ch
    "poesia",       # it poetry — poesia.ch
    "editore",      # it publisher — editore.ch
    "subito",       # it immediately — subito.ch
    # en (and cross-language)
    "cash",         # cash.ch
    "alert",        # alert.swiss
    "radio",        # radio.ch
    "nature",       # nature.com — "nature reserves …" is not naming the journal
    "jura",         # de law studies / fr canton+range — jura.ch
    "total",        # total.ch
    "digital",      # digital.swiss
    "men",          # men.ch
    "stat", "stats",  # stat.ch / stats.ch
    "books",        # books.ch
    "museums",      # museums.ch
    "students",     # students.ch
    "lawyers",      # lawyers.ch
    "marketing",    # marketing.ch
    "football",     # football.ch
    "curling",      # curling.ch
    "karate",       # karate.ch
    "bridge",       # the game / the structure — bridge.ch
    "soil",         # soil.ch
    "panorama",     # panorama.ch
    "wired",        # wired.com — "wired ethernet …"
    "uri",          # canton / Uniform Resource Identifier — uri.ch, uri.swiss
    # major city heads: constantly incidental in travel/event queries
    # ("train timetable geneva lausanne" fired lausanne.ch at dense sim 0.0);
    # dense still serves true city-administration intent. Small communes are
    # deliberately NOT listed — a query naming Giswil almost surely wants
    # giswil.ch.
    "bern", "geneve", "lausanne", "fribourg", "winterthur", "glarus",
}


def _fold(s: str) -> str:
    """NFKD accent fold (lowercased input): 'nestlé' -> 'nestle' (task 183)."""
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


# membership checks compare FOLDED heads, so accented entries ("lärm") guard
# their unaccented query form too.
_GENERIC_HEADS_FOLDED = {_fold(h) for h in _GENERIC_HEADS}

# payload disposition -> serving vocabulary
_DISPO = {"deep-index": "indexed", "both": "indexed", "card": "card-only",
          "consent-declined": "consent-declined"}

_CARDS_SP = qm.SearchParams(
    quantization=qm.QuantizationSearchParams(rescore=True, oversampling=3.0))


def _navigate(nav: dict, query: str) -> dict:
    """Fill the search-URL template + pass through the machine affordances."""
    tmpl = nav.get("search_url_template") or ""
    out = {
        "search_url": tmpl.replace("{q}", urllib.parse.quote_plus(query)),
        "search_url_template": tmpl or None,
        "query_format": nav.get("query_format"),
        "sitemap": nav.get("sitemap"),
        "api": nav.get("api"),
        # "guess" = conventional ?q= pattern, unverified — the agent should
        # be ready for it to 404 and fall back to the site's visible search.
        "confidence": nav.get("confidence")
                      or ("verified" if nav.get("derivation") == "curated" else "guess"),
    }
    alts = nav.get("alternatives")
    if alts:
        out["alternatives"] = [a.replace("{q}", urllib.parse.quote_plus(query))
                               for a in alts]
    return out


def _card(pl: dict, query: str, fused: float, dense_sim: float,
          named: bool = False) -> dict:
    dispo = _DISPO.get(pl.get("disposition"), pl.get("disposition"))
    confidence = round(float(dense_sim), 4)
    return {
        "authority": pl.get("domain"),
        "name": pl.get("name") or pl.get("domain"),
        "description": pl.get("description") or None,
        "authoritative_for": (pl.get("authoritative_for") or [])[:8],
        "jurisdiction": pl.get("jurisdiction"),
        "trust_tier": pl.get("trust_tier"),
        "disposition": dispo,
        # consent-declined: the publisher's signals decline automated access
        # (ADR-024). We name the source (we point) but neither fetch nor hold
        # its passages, and the agent should not auto-fetch it.
        "fetchable": dispo != "consent-declined",
        "freshness": pl.get("freshness_class"),
        "why": pl.get("why"),
        "navigate": _navigate(pl.get("navigation") or {}, query),
        "score": round(float(fused), 4),
        "similarity": confidence,
        "confidence": confidence,
        "weak_match": confidence < WEAK_MATCH_FLOOR and not named,
    }


def annotate(cards: list[dict]) -> dict:
    """Response-level fields for /find_authority: `weak_match` is true only
    when there are cards and ALL of them are weak; `note` then says so."""
    weak = bool(cards) and all(c.get("weak_match") for c in cards)
    out: dict = {"weak_match": weak}
    if weak:
        out["note"] = WEAK_NOTE
    if any(c.get("fetchable") is False for c in cards):
        out["declined_note"] = ("cards with fetchable=false are consent-declined: "
                                "name them, do not fetch them")
    return out


#: `country` aliases for the /find_authority route. The jurisdiction enum has no `EU`; EU law
#: is carded as `INTL`, so `EU` can never match and is mapped instead of returning zero.
COUNTRY_ALIASES = {"EU": "INTL"}


def find_authorities_relaxed(q: str, k: int = 3, country: str | None = None,
                             **kw) -> tuple[list[dict], str | None]:
    """/find_authority entry: `find_authorities` plus the zero-card fallback.
    Returns (cards, filter_relaxed). `EU` is mapped to `INTL`. When a `country`
    filter yields no card, retry ONCE without it; those cards carry
    filter_relaxed="country" (they may not be from the requested jurisdiction).
    A call that returns cards under the filter, or none either way, gets None."""
    c = (country or "").strip().upper() or None
    c = COUNTRY_ALIASES.get(c, c)
    cards = find_authorities(q, k=k, country=c, **kw)
    if cards or c in (None, "NONE", "ANY", "ALL"):
        return cards, None
    cards = find_authorities(q, k=k, country=None, **kw)
    return cards, ("country" if cards else None)


def find_authorities(q: str, k: int = 3, country: str | None = None,
                     floor: float | None = None, host_ok=None) -> list[dict]:
    """Top-k source cards for a query. Raises on backend failure (caller
    isolates). `country` is a HARD jurisdiction filter here (unlike the
    /search soft-boost) — the cards pool is tiny, filtering is safe.
    `floor`: min raw dense cosine (default MIN_SIM; sidecar passes the
    stricter SIDECAR_FLOOR). A by-name domain match bypasses the floor.
    """
    k = max(1, min(int(k), 10))
    country = (country or "").strip().upper() or None
    if country in ("NONE", "ANY", "ALL"):
        country = None

    dense, _ = models.embed_query(q)
    bm = bm25_tokens(q)

    qfilter = None
    if country:
        qfilter = qm.Filter(must=[qm.FieldCondition(
            key="jurisdiction", match=qm.MatchValue(value=country))])

    pool = max(4 * k, 12)
    legs = {"dense": qdrant_io.client().query_points(
        collection_name=CARDS_COLLECTION, query=dense, using=DENSE_VEC,
        limit=pool, query_filter=qfilter, with_payload=True,
        search_params=_CARDS_SP).points}
    if bm:
        # index-side parity: build_source_cards._bm25_vector / consent_gate
        # fold token hashes to 31 bits (`hash(t) & 0x7FFFFFFF`) — the query
        # side MUST apply the same fold or ~half the tokens never match.
        # (The main passage index uses the raw 32-bit hashes; cards differ.)
        folded: dict[int, float] = {}
        for i, v in bm.items():
            fi = i & 0x7FFFFFFF
            folded[fi] = folded.get(fi, 0.0) + v
        legs["bm25"] = qdrant_io.client().query_points(
            collection_name=CARDS_COLLECTION,
            query=qm.SparseVector(indices=list(folded.keys()),
                                  values=list(folded.values())),
            using=SPARSE_BM25, limit=pool, query_filter=qfilter,
            with_payload=True).points

    payloads, raw = {}, {}
    for leg, pts in legs.items():
        raw[leg] = {}
        for p in pts:
            payloads[p.id] = p.payload
            raw[leg][p.id] = p.score
    if not payloads:
        return []

    # by-name lookup: query token == registrable domain head ("nzz" -> nzz.ch).
    # len>=3 keeps 2-letter heads (ch.ch) from matching noise.
    # task 183: NFKD-accent-fold BOTH sides — the probe-024 bug was
    # "nestlé" (query token) ≠ "nestle" (domain head), so an accented
    # by-name query lost the bypass and the card dropped at the floor.
    q_toks = {_fold(t) for t in re.findall(r"\w+", q.lower()) if len(t) >= 3}

    def _named(pl: dict) -> bool:
        head = _fold((pl.get("domain") or "").split(".")[0])
        return (len(head) >= 3 and head not in _GENERIC_HEADS_FOLDED
                and head in q_toks)

    ndense = _minmax(raw.get("dense", {}), flat=0.0)
    nbm25 = _minmax(raw.get("bm25", {}), flat=0.0)
    scored = sorted(
        ((_W_DENSE * ndense.get(pid, 0.0) + _W_BM25 * nbm25.get(pid, 0.0)
          + (_NAME_BOOST if _named(payloads[pid]) else 0.0), pid)
         for pid in payloads),
        reverse=True)

    min_sim = MIN_SIM if floor is None else floor
    out, seen = [], set()
    for fused, pid in scored:
        pl = payloads[pid]
        dom = pl.get("domain")
        if dom in seen:
            continue
        if host_ok is not None and not host_ok(dom or ""):   # task 2640 site:/-site:
            continue
        sim = raw.get("dense", {}).get(pid, 0.0)
        if sim < min_sim and not _named(pl):
            continue
        seen.add(dom)
        out.append(_card(pl, q, fused, sim, named=_named(pl)))
        if len(out) >= k:
            break
    return out
