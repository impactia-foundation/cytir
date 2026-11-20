"""Task 2737: the projected snippet skips the structured-markdown provenance header.

Four materializers open a row's text with the same machine-identity block (a `## Record identity`
section of `- Label: value` bullets) after the `# title` line:

  cytir/extract/bfs_dam_current.py  `- Canonical asset:` `- BFS DAM asset ID:` `- API UUID:` `- Language:`
  cytir/extract/bfs_pxweb.py        `- PXWeb database ID:` `- Language:` `- API endpoint:` `- Parent response:` ...
  cytir/extract/swisstopo.py        `- SwissNAMES3D UUID:` `- Name UUID:` `- Release:` `- Parent archive member:` ...
  cytir/extract/pubmed.py           `- PMID:` `- Parent archive:` `- Parent archive SHA-256:`

The snippet is cut from the top of the text, so on 5.6M twin rows it was that block, not the
record. The rule is anchored to the template: the text starts (optionally after one `# title`
line) with the `## Record identity` heading, and the block is that heading plus the bullet lines
that follow it up to the first blank line. Anything else is returned unchanged.
"""
from __future__ import annotations

import re

# `# title\n\n` (optional) + `## Record identity\n\n` + one or more `- ...` lines + blank line / end.
_HEADER = re.compile(
    r"\A(?P<title>#[ \t][^\n]*\r?\n\r?\n)?(?:##[ \t]+)?Record identity[ \t]*\r?\n(?:\r?\n)?"
    r"(?:- [^\n]*(?:\r?\n|\Z))+(?:\r?\n|\Z|(?=##\s))")


def strip_provenance_header(text: str) -> str:
    """Return `text` without a LEADING `Record identity` block; the `# title` line stays.

    Task 2734: the heading is `## Record identity` in the extractor template, but the SEALED PACK
    `text` (what the drain and the live rows hold) is the chunk text with the markdown hashes
    removed and no blank line after the heading: `Record identity\n- PMID: ...\n\nAbstract\n...`
    (bfs, swisstopo, pubmed packs, read 2026-11-12). Both forms are matched."""
    m = _HEADER.match(text or "")
    if not m:
        return text
    return (m.group("title") or "") + text[m.end():]
