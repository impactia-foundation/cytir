"""JavaScript-shell detection (task 2710): is this page an empty app shell rather than the document?"""
from __future__ import annotations

import re

import lxml.html

from . import config

# Markers (lower-case) of "this page needs JavaScript", in the four languages.
SHELL_MARKERS = (
    "javascript-fähigen browser", "javascript fähigen browser", "javascript aktivieren", "javascript ist deaktiviert",
    "enable javascript", "requires javascript", "javascript is required", "javascript is disabled",
    "activer javascript", "activez javascript", "javascript est désactivé", "navigateur compatible javascript",
    "abilitare javascript", "abilita javascript", "javascript è disabilitato", "browser con javascript",
)
_NOSCRIPT = re.compile(r"<noscript\b[^>]*>(.*?)</noscript>", re.I | re.S)
_STRIP = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.I | re.S)
NEAR_EMPTY_BODY_CHARS = 400


def _has_landmark_text(html: str) -> bool:
    """A <main> or <article> that holds real text."""
    try:
        root = lxml.html.fromstring(html)
    except (ValueError, lxml.etree.ParserError):
        return False
    return any(len(" ".join(e.text_content().split())) >= config.SHELL_MIN_TEXT_CHARS
               for e in root.iter("main", "article"))


def is_shell(html: str, text: str) -> bool:
    """True when `html` (raw or rendered) is a JavaScript shell, given the `text` extracted from it.

    Shell = (a) little text relative to the HTML (under 1 %), from a large page (over 20 KB), with no <main> or
    <article> holding text (so inline CSS / JSON-LD around a short static page does not count); or (b) a JavaScript
    marker INSIDE a <noscript>, or in a near-empty body of a large (app-bundle sized) page (a page that only mentions JavaScript is not a shell)."""
    n = len(text.strip())
    size = len(html.encode("utf-8", "ignore"))
    if (n < config.SHELL_MIN_TEXT_CHARS and size > config.SHELL_MIN_HTML_BYTES
            and n / size < config.SHELL_MAX_TEXT_RATIO and not _has_landmark_text(html)):
        return True
    if any(any(m in nm.group(1).lower() for m in SHELL_MARKERS) or "javascript" in nm.group(1).lower()
           for nm in _NOSCRIPT.finditer(html)) and n < config.SHELL_MIN_TEXT_CHARS * 5:
        return True
    body = _STRIP.sub(" ", html)
    visible = " ".join(re.sub(r"<[^>]+>", " ", body).split())
    return (size > config.SHELL_MIN_HTML_BYTES and len(visible) < NEAR_EMPTY_BODY_CHARS
            and any(m in visible.lower() for m in SHELL_MARKERS))
