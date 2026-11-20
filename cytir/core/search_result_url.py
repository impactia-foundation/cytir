"""cytir.core.search_result_url -- the site-search result URL predicate (task 2009).

A site's own search-result page (`https://example.ch/?s=term`, `/search?q=term`, `/suche?query=term`)
is a view over the site's index, not an authoritative source: it is never a page worth citing.
Task 2003 b015 found `fintechnews.ch/?s=postfinance` indexed as `official`, trust 0.6.

`cytir.core.projection.project_record` -- the one record-to-point projection every pack path calls --
applies this predicate and returns None (the existing "skip" outcome its callers already count).
The predicate is pure: it looks at the URL only, never fetches, never reads a source registry.
"""
from __future__ import annotations

import threading
import urllib.parse

# Query keys that carry a search term.
SEARCH_QUERY_KEYS = frozenset({"s", "q", "query", "search", "suche", "recherche", "cerca", "k"})
# Paths that are a site-search page when they carry one of those keys.
SEARCH_PATHS = frozenset({"/", "/search", "/suche", "/recherche", "/ricerca", "/cerca", "/find"})

_REJECTS = 0
_REJECTS_LOCK = threading.Lock()


def is_search_result_url(url: str | None) -> bool:
    """True if `url` is a site-search result page.

    `?s=<term>` is a search on ANY path (the WordPress form). The other keys count only on the
    site root, a known search path, or a path ending in `/search`.
    """
    try:
        parts = urllib.parse.urlsplit((url or "").strip())
    except ValueError:
        return False
    if not parts.query:
        return False
    keys = {k.lower() for k, _v in urllib.parse.parse_qsl(parts.query, keep_blank_values=False)}
    if "s" in keys:
        return True
    if not keys & SEARCH_QUERY_KEYS:
        return False
    stripped = (parts.path or "").lower().strip("/")
    path = "/" + stripped if stripped else "/"
    if path in SEARCH_PATHS or path.endswith("/search"):
        return True
    # MediaWiki search (task 2048): /w/index.php?search=<term>, on any language host.
    if path.endswith("/index.php") and "search" in keys:
        return True
    # A `suche` path segment at any language prefix (task 2048): /suche/?q=, /de/suche/?query=.
    return "suche" in stripped.split("/") and bool(keys & {"q", "query"})


def note_search_result_reject() -> None:
    """Count one rejected URL (process-wide, like `canon.credential_url_hit_count`)."""
    global _REJECTS
    with _REJECTS_LOCK:
        _REJECTS += 1


def search_result_reject_count() -> int:
    """Census counter: URLs rejected as site-search results since process start or last reset."""
    return _REJECTS


def reset_search_result_reject_count() -> None:
    """Test/census hook: zero the counter."""
    global _REJECTS
    with _REJECTS_LOCK:
        _REJECTS = 0


# task 2242: the ONE reason vocabulary for a record `project_record` rejects. `cytir.index.compose`
# and `cytir.index.promote_sealed_pack` both name a rejected row with it.
REJECTED_REASON = "projection-rejected"
REASON_SEARCH_RESULT_URL = f"{REJECTED_REASON}: search-result-url"
REASON_OTHER = f"{REJECTED_REASON}: other"


def rejected_reason(url: str | None) -> str:
    """The reason for a record the projection refused: a site-search URL is named positively,
    every other cause (empty url/text, any future one) is `other`, never mislabelled."""
    return REASON_SEARCH_RESULT_URL if is_search_result_url(url) else REASON_OTHER
