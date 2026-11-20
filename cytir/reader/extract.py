"""Text extraction and anchor trim (task 2710)."""
from __future__ import annotations

from urllib.parse import unquote, urlsplit

import lxml.etree
import lxml.html
import trafilatura


def to_text(html: str) -> str:
    """Markdown-ish text of `html` (trafilatura; tables kept), '' when nothing is extracted.

    Why not the index pipeline's extraction (`cytir.extract.html_extract`, favor_precision then a recall fallback plus
    markup repair and a quality filter)? That path decides what ENTERS the index and drops thin or low-quality pages.
    The reader serves one page to one agent: it must never return less than the page holds, so it runs the recall
    pass only (favor_recall) and returns whatever comes out, including short text."""
    if not html.strip():
        return ""
    return trafilatura.extract(html, output_format="markdown", include_tables=True, include_comments=False,
                               favor_recall=True) or ""


def _heading_level(tag: str) -> int | None:
    return int(tag[1]) if len(tag) == 2 and tag[0] == "h" and tag[1] in "123456" else None


def anchor_html(html: str, fragment: str) -> str | None:
    """The element with id `fragment` plus its following siblings up to the next element whose id shares the
    anchor's prefix (`art_`) or the next heading of equal or higher level. None when the id is absent."""
    try:
        root = lxml.html.fromstring(html)
    except (ValueError, lxml.etree.ParserError):
        return None
    hits = root.xpath("//*[@id=$i]", i=fragment)
    if not hits:
        return None
    el = hits[0]
    prefix = fragment.split("_", 1)[0] + "_" if "_" in fragment else None
    level = _heading_level(el.tag)
    parts = [lxml.html.tostring(el, encoding="unicode")]
    for sib in el.itersiblings():
        if not isinstance(sib.tag, str):
            continue
        if prefix and (sib.get("id") or "").startswith(prefix):
            break
        lv = _heading_level(sib.tag)
        if level is not None and lv is not None and lv <= level:
            break
        parts.append(lxml.html.tostring(sib, encoding="unicode"))
    return "<html><body>" + "".join(parts) + "</body></html>"


def extract(html: str, url: str) -> tuple[str, bool]:
    """(text, anchored). With a `#fragment` whose id exists, the text is that element only; else the page."""
    frag = unquote(urlsplit(url).fragment)
    if frag:
        sub = anchor_html(html, frag)
        if sub:
            t = to_text(sub)
            if t.strip():
                return t, True
    return to_text(html), False
