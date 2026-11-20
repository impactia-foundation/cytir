"""The decoded-HTML -> markdown stage of the exact-URL preserved families (task 2137, S7).

Four contract rows (canada.ca CC + direct, cochrane.org CC + direct) were materialized by one-off
drivers (three of them deleted in task 1219, two kept as task 417 exploration scripts). All
call the same extractor, `cytir.core.common.extract_content_markdown(html, url)`; they differ only in how
they wrap its output. This module is the ONE home of that wrapping, taken verbatim from the sealed
drivers. No new behaviour; every difference is listed:

  canada.ca   parent row of the retired `materialize_ca_canadaca_preserved_html` script (CC) and its direct twin:
              markdown = principal.strip() + "\\n"; title = the HTML <title> (whitespace-collapsed);
              lang = <html lang>; an empty extraction OR a heading-only one (`has_indexable_body`)
              is quarantined (a terminal EXCLUDED fact about the unit).
  cochrane    `materialize_r0_cochrane_{direct_terminal,preserved_partial}.py`: markdown =
              "# {title}\\n\\nSource: [Cochrane evidence record]({url})\\n\\n" + principal.strip() + "\\n".
              The title and lang are NOT read from the HTML: the sealed rows carry the CURRENT served
              title/lang, so the caller supplies them (sealed source manifest rows). Only an empty
              extraction is quarantined. Exceptions: an extractor result of None is treated as empty (the drivers would raise AttributeError); the direct driver tests `not principal.strip()`,
              the preserved driver tests `not body_markdown`; the two differ only on a whitespace-only
              extraction, which this port treats as empty for both.
"""
from __future__ import annotations

import hashlib

from cytir.extract.html_extract import _parse_soup, extract_content_markdown


def has_indexable_body(markdown: str) -> bool:
    """Reject heading-only extractor output (the canada.ca materializer's own gate)."""
    return any(line.strip() and not line.lstrip().startswith("#")
               for line in markdown.splitlines())


def html_title_lang(html: str) -> tuple[str | None, str | None]:
    """The canada.ca parent's title and lang, read from the HTML exactly as the sealed driver did."""
    soup = _parse_soup(html)
    title = (" ".join(soup.title.get_text(" ", strip=True).split())
             if soup is not None and soup.title else None)
    lang = ((soup.html.get("lang") if soup is not None and soup.html else None) or None)
    return title, lang


def canada_parent(html: str, url: str) -> dict:
    """The canada.ca markdown parent fields. Raises ValueError(reason) for a terminal quarantine."""
    principal, mode = extract_content_markdown(html, url)
    if not principal or not principal.strip():
        raise ValueError("principal-content-extraction-empty")
    if not has_indexable_body(principal):
        raise ValueError("principal-content-has-no-indexable-body")
    markdown = principal.strip() + "\n"
    title, lang = html_title_lang(html)
    return {"title": title, "lang": lang, "markdown": markdown,
            "markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
            "extraction": {"producer": "aw_common.extract_content_markdown", "mode": mode,
                           "full_prechunk_principal_content": True,
                           "source_text_truncation_applied": False}}


def cochrane_markdown(html: str, url: str, title: str) -> tuple[str, str]:
    """The cochrane markdown: (rendered markdown, extraction mode). Raises ValueError when empty."""
    principal, mode = extract_content_markdown(html, url)
    if not principal or not principal.strip():
        raise ValueError("principal-content-empty")
    return f"# {title}\n\nSource: [Cochrane evidence record]({url})\n\n{principal.strip()}\n", mode


def cochrane_document_id(url: str) -> str:
    return hashlib.sha256(f"cochrane-evidence\0{url}".encode()).hexdigest()
