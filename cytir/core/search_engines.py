"""ONE list of search-engine hosts, and `is_search_engine(url)` (task 2710; moved from the agent probe).

Fetching a search-engine result page is using a second search tool, not reading a source (Tim 2026-11-11).
The agent probe and the local reader (`cytir.reader`) both refuse these URLs; neither restates the list.
"""
from __future__ import annotations

from urllib.parse import urlsplit

# A host matches itself and its sub-hosts.
SEARCH_ENGINE_HOSTS = ("bing.com", "duckduckgo.com", "google.com", "baidu.com", "startpage.com", "ecosia.org",
                       "kagi.com", "perplexity.ai", "you.com")


def host_of(url: str) -> str:
    h = (urlsplit(url if "//" in url else "//" + url).hostname or "").lower()
    return h[4:] if h.startswith("www.") else h


def host_matches(url: str, hosts: list[str] | tuple[str, ...]) -> bool:
    h = host_of(url)
    return any(h == g or h.endswith("." + g) for g in hosts)


def is_search_engine(url: str | None) -> bool:
    """True when `url` points at a search engine (the host list, `yandex.*`, search.yahoo.com,
    yahoo.com/search, brave.com/search, search.brave.com)."""
    if not url:
        return False
    path = urlsplit(url if "//" in url else "//" + url).path or ""
    h = host_of(url)
    if host_matches(url, SEARCH_ENGINE_HOSTS):
        return True
    if "yandex" in h.split("."):
        return True
    if h == "yahoo.com" or h.endswith(".yahoo.com") or h == "brave.com" or h.endswith(".brave.com"):
        return "search" in h.split(".") or path.startswith("/search")   # search.yahoo.com, uk.search.yahoo.com, ...
    return False
