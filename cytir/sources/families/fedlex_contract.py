"""Dependency-light shared contracts for pinned Fedlex source artifacts."""
from __future__ import annotations

import re


ELI_ANCHOR_RE = re.compile(r"/eli/([a-z0-9/_]+)/(de|fr|it|en)#(.+)$")
WINDOW_SUFFIX_RE = re.compile(r"^(.*)-c\d+$")


def fedlex_xml_url(eli: str, version: str, lang: str) -> str:
    """Return the canonical Fedlex filestore URL for one AKN representation."""
    path = f"fedlex.data.admin.ch/eli/{eli}/{version}/{lang}/xml"
    filename = re.sub(r"[./]", "-", path) + ".xml"
    return f"https://fedlex.data.admin.ch/filestore/{path}/{filename}"


def fedlex_html_url(eli: str, version: str, lang: str) -> str:
    """The filestore HTML manifestation: a no-JavaScript copy of the consolidated act (task 2709).

    Same path rule as the XML twin. A WRONG `version` still answers HTTP 200 with a ~9 KB SPA
    shell, so a caller that vouches for this URL must check the body size (`cyt source
    fedlex-versions` does)."""
    path = f"fedlex.data.admin.ch/eli/{eli}/{version}/{lang}/html"
    filename = re.sub(r"[./]", "-", path) + ".html"
    return f"https://fedlex.data.admin.ch/filestore/{path}/{filename}"


def source_native_key(url: str) -> tuple[str, str, str] | None:
    """Return the stable ELI/language/base-anchor identity of a Fedlex URL."""
    match = ELI_ANCHOR_RE.search(url)
    if not match:
        return None
    eli, lang, anchor = match.groups()
    window = WINDOW_SUFFIX_RE.match(anchor)
    return eli, lang, window.group(1) if window else anchor


# --- the refreshed act: the scope a url-group supersede replaces (task 1038) -------------------
# A fedlex refresh replaces a WHOLE ACT in one language. The unit the drain supersedes is that
# act-language, not an article url: the door writes `...#art_14-c1`, `...#art_14-c2` for an article
# the twin holds at the bare `...#art_14`, so a per-url group never meets the old row (task 1035
# section 4: 294 of 308 twin rows covered). Served fedlex rows carry no `provenance`, so the family
# is told by its URL scope: host `fedlex.admin.ch` and path `/eli/<eli>/<lang>` (the ELI may hold
# underscores: `cc/27/317_321_377`). The path is matched WHOLE, never as a text prefix: `cc/2022/49`
# is not `cc/2022/491`, and `/fr` is not `/de`.
FEDLEX_HOST = "fedlex.admin.ch"
_ACT_PATH_RE = re.compile(r"^/eli/([a-z0-9/_]+)/(de|fr|it|en)/?$")


def act_scope(url: str | None) -> tuple[str, str] | None:
    """`(eli, lang)` of a fedlex act url (any fragment, `www.` or not, http or https), or None when
    the url is not a fedlex act page. Host and WHOLE path are compared; the fragment is ignored."""
    from urllib.parse import urlsplit
    try:
        p = urlsplit(url or "")
    except ValueError:
        return None
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host != FEDLEX_HOST:
        return None
    m = _ACT_PATH_RE.match(p.path or "")
    return (m.group(1), m.group(2)) if m else None


def split_anchor(url: str) -> tuple[str, str | None, int | None]:
    """`(url without fragment, base anchor, window index)` of a fedlex url. `...#art_14-c2` gives
    `(".../fr", "art_14", 2)`; `...#art_14` gives `(".../fr", "art_14", None)`; no fragment gives
    `(url, None, None)`."""
    base, _, frag = (url or "").partition("#")
    if not frag:
        return base, None, None
    m = WINDOW_SUFFIX_RE.match(frag)
    if m:
        return base, m.group(1), int(frag[len(m.group(1)) + 2:])
    return base, frag, None


def coverage_url(url: str) -> str:
    """The url the word-coverage guard compares at: the BASE anchor (`...#art_14` for `-c1`, `-c2`
    and the bare row), so an article that became windows is compared as one article."""
    base, anchor, _ = split_anchor(url)
    return f"{base}#{anchor}" if anchor else url
