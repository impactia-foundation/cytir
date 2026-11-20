"""One URL admission rule for the stage door of every family (task 2713).

Two junk classes entered the served index through a URL that is not a page of the source:

* a search-result page (`docs.github.com/en/search?query=...`, two with spam in the query);
* site chrome of the statute host (`fedlex.admin.ch/page-not-found`, the root, `/de/cc`, `/de/fga`).

and one duplicate class: a hub page crawled with a version or referral parameter
(`/en/rest?latest`, `/en/rest?apiVersion=2026-03-10`, `/de/rest?ref=...`).

`refusal(url)` names why a URL is not admitted (None = admitted). `fold_url(url)` is the spelling a page
is deduplicated under: `strip_tracking_url` plus the version parameters of `FOLD_PARAMS`. The fold is a
dedup KEY only; it never changes `canonical_url`, which is an id input (canon epochs, ADR-067).
"""
from __future__ import annotations

import urllib.parse

from cytir.core.canon import strip_tracking_url

REASON_SEARCH = "search-result-page"
REASON_CHROME = "site-chrome-not-content"
REASON_TRACKING = "tracking-query-duplicate"

#: Version / view parameters of docs sites that select a rendering of the same page, and the hosts
#: where that is known (a new host is a deliberate row here, not a global rule).
FOLD_PARAMS = frozenset({"latest", "apiversion"})
FOLD_HOSTS = frozenset({"docs.github.com"})
SEARCH_PARAMS = frozenset({"query", "q"})
#: Hosts whose only content is under `/eli/` (the Akoma Ntoso acts); every other path is site chrome.
ELI_ONLY_HOSTS = frozenset({"fedlex.admin.ch", "www.fedlex.admin.ch"})


def refusal(url: str) -> str | None:
    try:
        p = urllib.parse.urlsplit((url or "").strip())
    except ValueError:
        return None
    host = (p.hostname or "").lower()
    segments = [s for s in p.path.split("/") if s]
    if segments and segments[-1].lower() == "search":
        keys = {k.lower() for k, _ in urllib.parse.parse_qsl(p.query, keep_blank_values=True)}
        if keys & SEARCH_PARAMS:
            return REASON_SEARCH
    if host in ELI_ONLY_HOSTS and not (segments and segments[0] == "eli"):
        return REASON_CHROME
    return None


def fold_url(url: str) -> str:
    """The dedup spelling of `url`: tracking and version parameters removed, the rest kept."""
    clean = strip_tracking_url(url)
    try:
        p = urllib.parse.urlsplit(clean)
    except ValueError:
        return clean
    if not p.query or (p.hostname or "").lower() not in FOLD_HOSTS:
        return clean
    kept = [(k, v) for k, v in urllib.parse.parse_qsl(p.query, keep_blank_values=True)
            if k.lower() not in FOLD_PARAMS]
    return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, urllib.parse.urlencode(kept), p.fragment))
