"""cytir.core.prose_guard -- is this text prose, or a binary file read as text?  (task 767, defect B)

THE DEFECT. Raw PDF streams (`%PDF-`, `12 0 obj`, `endobj`, Flate bytes shown as Latin-1) were
indexed as `snippet` on the Swiss federal PDF hosts (weko.admin.ch 1.4 %, bazl.admin.ch 5 % of
rows). They win unrelated queries: a Russian metal-supplier query ranked a WEKO PDF first.
`low_quality` (cytir.core.common) did not catch them: it counts `str.isalpha()` letters, and Latin-1
mojibake (`Þ`, `û`, `ð`) IS alphabetic. `extraction_junk` has no byte-noise reason either.

THE TEST. Pure, stdlib only, cheap (one regex pass and one char pass over at most
`SCAN_CHARS`). A text is BINARY when any one of:

  * `pdf_structure`: >= 2 PDF object-syntax tokens (`%PDF-1.`, `endobj`, `endstream`, `N 0 obj`,
    `/FlateDecode`, `xref`, and `N G R` references beside one of those; task 777: a bare `N N R`
    table is not PDF, only generation-0 references `N 0 R` count without a marker) making >= 4 % of
    the tokens. One token is not enough, and neither is
    a long page ABOUT PDF with a few code samples (the density test).
  * `pdf_syntax`: >= 4 slash-names (` /Type /Annot /Subtype /Link`) making >= 20 % of the tokens
    AND a dictionary marker (`<<`, `>>`, `N 0 R`, `N 0 obj`, `/Type`). Markup (`</Table>`),
    compiler flags (`/OUT:x`) and path lists (`/ODataFeed/OData/7`) have no marker.
  * `byte_noise`: > 4 % of the characters are control characters, U+FFFD, C1/Latin-1 symbol
    bytes or other typographic-symbol noise, over a text of at least 40 characters. Real prose
    carries up to ~2 % (Word bullets in the private-use area, U+F02D, in WEKO decisions, `″`);
    soft hyphens (also written `¬`), ordinals (`1ª`, `2º`), zero-width/bidi marks, primes and the
    other prose symbols of Latin-1 are not noise;
    measured on the twin (task 767).

No prose language is assumed: German/French/Italian letters (a-z with diacritics), `«»§°`,
quotes, dashes and the usual symbols are NOT noise. Returns a reason string or None.
"""
from __future__ import annotations

import re
from typing import Optional

BINARY_REASON = "binary_content"
SCAN_CHARS = 4000
MIN_NOISE_CHARS = 40
NOISE_SHARE_MAX = 0.04
PDF_STRUCTURE_MIN = 2
PDF_STRUCTURE_SHARE_MIN = 0.04
PDF_SYNTAX_MIN = 4
PDF_SYNTAX_SHARE_MIN = 0.20

#: `N N R` alone is NOT PDF syntax (task 777: Wikipedia tables "R 93 102 R", `iosnoop` output
#: "100 15795 R"): a reference counts only beside a real object marker (`_PDF_MARKER`) in the text,
#: or when it is a generation-0 reference (`12 0 R`, every live PDF object; three sealed binary
#: rows are pure reference arrays with no marker in the 600-char snippet).
_PDF_MARKER = re.compile(r"%PDF-\d|\bendobj\b|\bendstream\b|\b\d+ \d+ obj\b|/FlateDecode|\bxref\b")
_PDF_REF = re.compile(r"\b\d+ \d+ R\b")
_PDF_REF0 = re.compile(r"\b\d+ 0 R\b")
#: A slash-name as a PDF dictionary writes it: after whitespace or a bracket, never after `<` (a
#: closing tag `</Table>`) and never before `:`/`=` (`/OUT:` compiler flags). Path lists
#: (`/ODataFeed/OData/7`) and markup carry no dictionary marker, which `pdf_syntax` also needs.
_PDF_NAME = re.compile(r"(?:^|[\s\[(])/[A-Z][A-Za-z]{2,}(?![A-Za-z:=])")
_PDF_DICT_MARK = re.compile(r"<<|>>|\b\d+ \d+ (?:obj|R)\b|/Type\b|/Subtype\b")

#: Format characters that carry no glyph and no byte damage: zero-width and bidi marks (U+200B-F,
#: U+202A-E, U+2060-4, U+2066-9), primes (U+2032-7: coordinates `51°39′15″N`), the word joiner.
_PROSE_FORMAT = frozenset(
    "".join(chr(c) for c in (*range(0x2000, 0x2010), *range(0x2028, 0x2030), 0x202F,
                            *range(0x2032, 0x2038), *range(0x2060, 0x206A))))
#: Latin-1 / punctuation code points that appear in real European prose; everything else in
#: the C1 / Latin-1-symbol / general-punctuation blocks counts as noise.
_PROSE_LATIN1 = frozenset("\xa0\xa7\xa9\xaa\xab\xac\xad\xae\xb0\xb2\xb3\xb4\xb5\xb6\xb7\xba\xbb"
                          "\xbc\xbd"
                          "\xa1\xa2\xa3\xa5\xb1\xb9\xbf")   # NBSP § © ª « ¬ SHY ® ° ² ³ ´ µ ¶ · º » ¼ ½
#: + ¡ ¢ £ ¥ ± ¹ ¿ (task 777: fees, tables, footnotes, Spanish)
_PROSE_PUNCT = frozenset("‘’‚‛“”„‐‑‒–—"
                         "†‡•…‰′‹›€‖―")    # ‖ norm sign, ‡ (sibling of †), ― bar dash (task 777)


#: Symbols exempted in task 777 (real prose: fees, tables, footnotes, Spanish, player tables).
#: Mojibake writes the same characters NEXT TO other non-ASCII letters: UTF-8 read as Latin-1 /
#: cp1252 gives `Ã¡`, `Ø¹`, `Ù‡`, `â€¢`; EUC-JP / Big5 read as Latin-1 gives `¥Æ¥¹¥È`, `¡¼`. Prose puts
#: them beside ASCII (`£50`, `5 ±2`, `¿Qué`, `Fussnote¹`). Beside a non-ASCII neighbour they stay
#: noise, which is how mojibake was caught before the exemption.
_AMBIG = frozenset("\xa1\xa2\xa3\xa5\xb1\xb9\xbf\u2021\u2015")


def _nonascii_neighbour(c: str) -> bool:
    return ord(c) >= 0x80 and not c.isspace() and c not in _PROSE_FORMAT


def noise_share(text: str) -> float:
    """Share of characters that are byte noise (see module docstring)."""
    if not text:
        return 0.0
    n = 0
    last = len(text) - 1
    for i, c in enumerate(text):
        o = ord(c)
        if c in _AMBIG:
            if (i and _nonascii_neighbour(text[i - 1])) or (i < last and _nonascii_neighbour(text[i + 1])):
                n += 1
        elif o < 32:
            if c not in "\t\n\r\x1b":      # ESC: ANSI colours in terminal output (task 777)
                n += 1
        elif o == 0xFFFD:
            n += 1
        elif 0x7F <= o <= 0xBF:
            if c not in _PROSE_LATIN1:
                n += 1
        elif 0x2000 <= o <= 0x206F and c not in _PROSE_PUNCT and c not in _PROSE_FORMAT:
            n += 1
    return n / len(text)


def binary_content(text: Optional[str]) -> Optional[str]:
    """`binary_content:<why>` when `text` is a binary file read as text, else None."""
    t = (text or "")[:SCAN_CHARS]
    if not t.strip():
        return None
    toks = t.split()
    markers = len(_PDF_MARKER.findall(t))
    hits = markers + len((_PDF_REF if markers else _PDF_REF0).findall(t))
    if hits >= PDF_STRUCTURE_MIN and hits / max(1, len(toks)) >= PDF_STRUCTURE_SHARE_MIN:
        return BINARY_REASON + ":pdf_structure"
    if toks:
        names = len(_PDF_NAME.findall(t))
        if (names >= PDF_SYNTAX_MIN and names / len(toks) >= PDF_SYNTAX_SHARE_MIN
                and _PDF_DICT_MARK.search(t)):
            return BINARY_REASON + ":pdf_syntax"
    if len(t) >= MIN_NOISE_CHARS and noise_share(t) > NOISE_SHARE_MAX:
        return BINARY_REASON + ":byte_noise"
    return None
