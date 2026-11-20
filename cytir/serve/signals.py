"""Per-candidate scoring signals that are pure functions of a payload or a URL (S10 slice 1,
task 752; moved verbatim from `retrieval.py`, which imports them back).

Pure means: no environment read, no module state, no I/O. Each value here depends only on its
arguments (and, for `_recency`, on today's date). The weights that multiply these signals and
the flags that arm them stay in `retrieval` until the typed-config slice.
"""
from __future__ import annotations

import datetime as dt
import math
import re
import urllib.parse


def _trust_tier(payload: dict) -> float:
    """Return the documented unknown-trust default without changing 0.0.

    Current staging rejects non-finite values, but old collections can contain
    a Qdrant-null payload from JSON NaN. A malformed historic point must not
    make the whole query fail; null has the same semantics as an omitted trust
    value everywhere else in retrieval.
    """
    value = payload.get("trust_tier")
    return 0.6 if value is None else float(value)


# --------------------------------------------------------------------------- #
#  Academic down-weight (task 089, feeds T2/T3).
#  pubmed/arxiv flood NON-academic queries on bare token overlap ("node" ->
#  oncology "node-positive" staging, "rust" -> fungal-rust genomes). We add a
#  penalty (weight <= 0) to candidates on academic domains WHEN the query shows
#  no academic intent. Intent is a cheap in-process lexicon — NO extra model.
#  Measured on labelled eval queries: recall 0.92, precision 0.96 (task 089).
# --------------------------------------------------------------------------- #
_ACADEMIC_DOMAINS = {"pubmed.ncbi.nlm.nih.gov", "arxiv.org",
                     "www.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov"}


def _domain(url: str) -> str:
    return urllib.parse.urlsplit(url or "").netloc.lower()


def _academic_penalty(url: str, q_academic: bool) -> float:
    """1.0 when this candidate should be down-weighted as off-topic academic
    noise (academic domain + non-academic query), else 0.0."""
    if q_academic:
        return 0.0
    return 1.0 if _domain(url) in _ACADEMIC_DOMAINS else 0.0


# --------------------------------------------------------------------------- #
#  Machine-URL penalty (task 089, feeds T2/T3).
#  Raw API/data endpoints (BFS pxweb `.px`, versioned `/api/vN`, sparql, raw
#  data files) are unreadable by an agent as a page and outrank the human
#  landing page. Down-weight them (weight <= 0). Tuned NOT to hit human doc
#  pages: MDN `/docs/Web/API/...` has no `/api/vN`; w3.org `/TR/sparql...`
#  standards docs have no `/sparql` path segment.
# --------------------------------------------------------------------------- #
_MACHINE_URL_RE = re.compile(
    r"\.px($|\?)"          # BFS pxweb .px endpoints
    r"|/api/v\d"           # versioned REST API paths (pxweb /api/v1/...)
    r"|/sparql\b"          # sparql query endpoints
    r"|[?&]swisssearch="   # map.geo.admin.ch search-marker deep-links (task 102):
                           # E/N coordinate viewer links, not a readable page —
                           # they flood Swiss topical queries at trust 0.95
                           # (e.g. "gotthard base tunnel") above the real article
    r"|map\.geo\.admin\.ch/\?[^ ]*\bE=\d"  # bare coordinate map-viewer deep-links
    r"|\.(?:json|csv|tsv|rdf|ttl|geojson)($|\?)",  # raw data files
    re.I)


def _machine_url(url: str) -> float:
    """1.0 when the URL is a machine/data endpoint (not a human page)."""
    return 1.0 if _MACHINE_URL_RE.search(url or "") else 0.0


# task 165: legifrance bulk-legal demotion (LODA /loda/article_lc/, KALI /conv_coll/,
# JORF /jorf/, Cass /juri/id/, Conseil d'Etat /ceta/id/). These bulk parallel-text
# articles flood OUT-OF-LANE queries on bare token overlap and, stamped trust 1.0
# source_type=official, ride the additive trust floor above the genuinely relevant
# authority (Tim 2026-07-08: LODA "zones franches" decret > EU AI Act on "AI
# regulatory playbook"). Task 152 established bulk sub-provenance is NEVER a gold.
# rel-GATED: demote a bulk article ONLY when it is not competitive with the pool's
# best relevance (rel_core < gate * best_rel_core) -> a genuine high-relevance bulk
# answer (a real /conv_coll/ query) keeps full ranking; the low-relevance flood is
# demoted. Additive penalty (weight <= 0) so it composes with every trust mode.
# Default weight 0.0 = fully inert (byte-identical); rollback = set 0.0.
_LF_HOST_RE = re.compile(r"(?:^|\.)legifrance\.gouv\.fr$", re.I)
_LF_BULK_RE = re.compile(r"/(?:loda/article_lc|conv_coll|jorf|juri/id|ceta/id)/", re.I)


def _lf_bulk(url: str) -> float:
    """1.0 for a legifrance.gouv.fr BULK-legal article (by URL path), else 0.0."""
    if not url:
        return 0.0
    h = _domain(url)
    if h.startswith("www."):
        h = h[4:]
    if not _LF_HOST_RE.search(h):
        return 0.0
    return 1.0 if _LF_BULK_RE.search(url) else 0.0


def _exact(q_terms: set[str], pl: dict) -> float:
    """Fraction of distinctive query terms appearing in the candidate's title +
    snippet (the only text we hold). Distinguishes the page ABOUT an entity from
    a page merely sharing a common first name. 0..1."""
    if not q_terms:
        return 0.0
    hay = (str(pl.get("title", "")) + " " + str(pl.get("snippet", ""))).lower()
    hay = hay.replace("'", "").replace("’", "")
    return sum(t in hay for t in q_terms) / len(q_terms)


def _minmax(scores: dict, drop_nonpositive: bool = False, flat: float = 1.0) -> dict:
    """Min-max normalise to [0, 1]. `flat` is the value every key takes when all scores are equal:
    1.0 for the fusion legs; the authority leg (D7, task 1104) passes 0.0, its own historic value."""
    if not scores:
        return {}
    if drop_nonpositive:
        scores = {k: v for k, v in scores.items() if v > 0.0}
        if not scores:
            return {}
    vs = list(scores.values())
    lo, hi = min(vs), max(vs)
    if hi <= lo:
        return {k: flat for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _recency(last_seen) -> float:
    """1.0 = fresh, decaying with a ~1-year half-life. Mechanism is in place
    even though most P0 content shares the build date."""
    if not last_seen:
        return 0.0
    try:
        d = dt.date.fromisoformat(str(last_seen)[:10])
        age = (dt.date.today() - d).days
        return math.exp(-max(age, 0) / 365.0)
    except Exception:  # noqa: BLE001
        return 0.0


def _registrable_domain(url: str) -> str:
    """Last two labels of the host (collapses roche.com/... and www.roche.com)."""
    h = _domain(url)
    if h.startswith("www."):
        h = h[4:]
    parts = h.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else h
