"""DILA `CONTENU` XML node -> plain Markdown-ish text (task 1437, S7).

Text surface shared by the JORF full-text producers (`cytir.sources.jorf.full_markdown`). The body is
the same function the frozen KALI materializer carries; this copy exists only because that
materializer is a separate lane's file, and a parity test pins that the two agree.
"""
from __future__ import annotations

import re

from lxml import etree

WS = re.compile(r"\s+")
BLOCK_TAGS = frozenset({"p", "div", "section", "article", "blockquote",
                        "table", "tr", "ul", "ol"})


def content_markdown(node) -> str:
    if node is None:
        return ""
    parts: list[str] = []

    def walk(item) -> None:
        if not isinstance(item.tag, str):
            return
        tag = etree.QName(item).localname.casefold()
        if tag == "br":
            parts.append("\n")
        elif tag == "li":
            parts.append("\n- ")
        elif tag in BLOCK_TAGS:
            parts.append("\n\n")
        if item.text:
            parts.append(item.text)
        for child in item:
            walk(child)
            if child.tail:
                parts.append(child.tail)
        if tag in BLOCK_TAGS:
            parts.append("\n\n")

    walk(node)
    lines = [WS.sub(" ", line).strip() for line in "".join(parts).splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
