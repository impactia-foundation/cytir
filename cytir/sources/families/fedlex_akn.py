"""Dependency-light source-native text contracts for pinned Fedlex AKN XML."""
from __future__ import annotations

import re

from lxml import etree


AKN = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
REPEAL_NOTICE = re.compile(r"\b(?:aufgehoben|abrog\w*|repealed)\b", re.IGNORECASE)


def tag(name: str) -> str:
    return "{%s}%s" % (AKN, name)


def norm(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _clean_parts(node, result: list[str]) -> None:
    if node.text: result.append(node.text)
    for child in node:
        if etree.QName(child).localname != "authorialNote":
            _clean_parts(child, result)
        if child.tail: result.append(child.tail)


def clean_text(node) -> str:
    result: list[str] = []
    _clean_parts(node, result)
    return " ".join(result)


def text_without(node, skipped_tags: set[str]) -> str:
    result: list[str] = []

    def visit(value) -> None:
        if value.text: result.append(value.text)
        for child in value:
            if etree.QName(child).localname not in skipped_tags: visit(child)
            if child.tail: result.append(child.tail)

    visit(node)
    return norm(" ".join(result))


def source_note_pointer(article) -> tuple[str, str] | None:
    substantive = text_without(article, {"num", "authorialNote"})
    if re.search(r"\w", substantive, flags=re.UNICODE): return None
    notes = norm(" ".join(" ".join(node.itertext())
                          for node in article.iter(tag("authorialNote"))))
    if not notes: return None
    return notes, "repealed" if REPEAL_NOTICE.search(notes) else "source_note"


# --- the article TITLE contract (task 516) ---------------------------------
# Production's fedlex records are titled
#
#     Obligationenrecht (OR; SR 220) — Art. 411 — IV. Kreditnehmer und Auftraggeber
#     |__ title_prefix ____________|   |_ cite _|   |_______ marginal ________|
#
# and 98.4% of the twin's 23,872 fedlex rows carry the SR number, 97.7% the
# abbreviation. The epoch-4 re-materialization
# (task 417's `materialize_r0_fedlex_anchored.py`) composed
# `docTitle - cite` instead and shipped 0 of 22,928 with either -- measured in
# task 515 section 7.3, and the direct cause of 14 of the 34 golds the 1024
# build lost (every lost legal query is abbreviation-led: URG, DSG, StPO, ZGB,
# CO, LCR, LTVA).
#
# The three parts are NOT prose to be parsed out of a title. `title_prefix` is
# a pinned per-(ELI, language) field of the task-370 source-version manifest
# (built from fedlex SPARQL `jolux:titleShort` + `jolux:historicalLegalId`);
# `cite` is the article's own `<num>`; `marginal` is the closest AKN ancestor
# carrying a `<num>`/`<heading>`. Both the live loader (`fedlex_dump.parse_akn`)
# and the pinned re-materialization compose them HERE, so the two paths cannot
# drift again: this is a mirror contract, and the retired `test_fedlex_akn` script
# asserts byte-identity against the production formula.

def ancestor_heading_path(article) -> tuple[list[str], str]:
    """Bottom-up over AKN ancestors (any container carrying `<num>`/
    `<heading>` -- part/title/chapter/level/section/... -- AKN doesn't
    constrain the tag name, only the convention; `body`/`act`/`akomaNtoso`
    never carry either and are skipped naturally). Returns (path in
    root->article order, marginal note = the closest ancestor's text).

    The two components are joined by a SPACE ("IV." + "Kreditnehmer und
    Auftraggeber" -> "IV. Kreditnehmer und Auftraggeber"), which is what
    production wrote 23,327 times. Joining them with an em dash instead --
    which the epoch-4 materializer's own `hierarchy()` did for its section
    headers -- produces "IV. - Kreditnehmer und Auftraggeber" and is a
    different string.
    """
    path: list[str] = []
    for ancestor in article.iterancestors():
        num = ancestor.find(tag("num"))
        heading = ancestor.find(tag("heading"))
        parts = (norm(clean_text(num)) if num is not None else "",
                 norm(clean_text(heading)) if heading is not None else "")
        combined = " ".join(part for part in parts if part)
        if combined:
            path.append(combined)
    path.reverse()
    return path, (path[-1] if path else "")


def article_citation(article, fallback_anchor: str = "") -> str:
    """The article's own `<num>` ("Art. 411"), or `fallback_anchor` (its eId)
    when the element is absent.

    An element that is PRESENT but empty yields "", exactly as both historical
    implementations did (`fedlex_dump.parse_akn` and the epoch-4 materializer
    agree on this); `compose_article_title` then keeps the separator, because
    that is what production wrote.
    """
    num = article.find(tag("num"))
    return norm(clean_text(num)) if num is not None else fallback_anchor


def compose_article_title(title_prefix: str, citation: str,
                          marginal: str = "") -> str:
    """`prefix - citation[ - marginal]` (em dashes).

    Byte-identical to `fedlex_dump.parse_akn`'s historical composition, INCLUDING
    its treatment of an empty citation (the separator stays):

        title = title_prefix
        title = "%s \u2014 %s" % (title, art_cite)
        if marginal: title = "%s \u2014 %s" % (title, marginal)
    """
    title = "%s \u2014 %s" % (title_prefix, citation)
    if marginal:
        title = "%s \u2014 %s" % (title, marginal)
    return title


# --- the law SPARSE-surface contract (task 564) -----------------------------
# Production's fedlex sparse channel is
#
#     "; ".join(ancestor_heading_path(article)) + ". " + <article body>
#
# and `fedlex_dump.parse_akn` has always composed it, with the reason in the
# comment beside it: task 358b proved that truncating the ancestor path cuts OR
# art. 257's query-relevant "Die Miete" ancestor, and that the full path is a
# SPARSE-ONLY retrieval hook (putting it into dense diluted the article
# semantics, task 358c).
#
# The epoch-4 re-materialization did not keep it. `materialize_r0_fedlex_anchored
# .render()` emits the ancestor breadcrumb as FLAT sibling `## <label> {#anchor}`
# headings ahead of the article's own `## <cite> {#anchor}` heading, so
# `generic_chunking`'s hierarchy channel -- the channel that feeds the pack's
# `sparse_text` -- sees the article heading as a SIBLING of the breadcrumb
# rather than a child of it, and the breadcrumb never reaches the article's own
# chunk. Measured on task 564's roster: 5 of the 7 fedlex golds lose exactly a
# word that IS in `ancestor_heading_path` and is in NO channel of the sealed
# record -- "aktiengesellschaft" + "verantwortlichkeit" (OR art. 754),
# "anonyme" (CO art. 656b), "contratto" (CO art. 336c), "contractual" (CO
# art. 341), "droit" (CC art. 111). See task 564's `fedlex-hierarchy.json`.
#
# The same `render()` also puts the AKN eId identifiers into the indexed text:
# 22,927 of 22,927 pack rows carry `{#art_N}` / `{#art_N/para_M}` in `text` AND
# in `sparse_text`, against 0 of 299 matched rows in production (task 562 \u00a74.3).
# `aw_common.bm25_tokens` splits on `\w+`, so `{#art_341}` becomes the single
# token `art_341`, which no query can ever produce ("art 341" tokenises to
# `art` + `341`) -- a posting that cannot match, counted into the `dl` that
# discounts every real term on the row.
#
# These two functions are the one home for that surface. `fedlex_dump.parse_akn`
# and the epoch-4 re-surfacing both call them, so the live loader and the
# re-materializer cannot drift again (engineering doctrine 1/6), and
# `cytir/index/representation_contract.py` refuses a law row whose indexed surface
# still carries the markup.

#: The pandoc/Akoma-Ntoso identifier form the epoch-4 markdown emits.
ANCHOR_MARKUP = re.compile(r"\{#[^}\s]+\}")
#: A whole line that is nothing but a label and an anchor identifier -- the
#: structural section headers `render()` writes (`paragraph {#art_1/para_1}`,
#: `listIntroduction {#.../listintro}`, `Art. 1 {#art_1}`) and the
#: `Official Fedlex article {#anchor}` line `stage_r0_fedlex_anchored.bind_anchor`
#: injects. The label is not vocabulary this module has to know: any line whose
#: entire content is "<label> {#<anchor>}" is a section header by construction,
#: so no curated list of AKN element names can go stale here.
ANCHOR_HEADING_LINE = re.compile(r"(?m)^[ \t]*#*[ \t]*[^{}\n]*\{#[^}\s]+\}[ \t]*$")


def strip_anchor_markup(text: str) -> str:
    """Remove the AKN eId scaffolding from a body destined for an INDEX surface.

    Whole-line section headers go entirely -- their label is the AKN element
    name (`paragraph`, `listIntroduction`) or a repeat of the article citation
    that the body's own `<num>` already carries. An identifier appearing inside
    a line of prose loses just the identifier. Blank runs left behind are
    collapsed, so the result renders like the body production composed.

    The identifiers are NOT lost: `anchor`/`base_anchor` are payload fields and
    the url carries the eId fragment. This is about what gets TOKENISED.
    """
    out = ANCHOR_HEADING_LINE.sub("", text or "")
    out = ANCHOR_MARKUP.sub("", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def compose_law_sparse_surface(heading_ctx: str, body: str) -> str:
    """Production's law sparse surface: the full ancestor heading path, then the
    body. Byte-identical to what `fedlex_dump.parse_akn` has always written --
    that call site now imports this instead of repeating the format string."""
    body = (body or "").strip()
    heading_ctx = (heading_ctx or "").strip()
    return ("%s. %s" % (heading_ctx, body)) if heading_ctx else body
