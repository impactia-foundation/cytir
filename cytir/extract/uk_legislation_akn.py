"""legislation.gov.uk Akoma Ntoso (`/{uri}/data.akn`) -> section-bounded markdown (task 2181).

One AKN file is one Act or statutory instrument. It yields one `Section` per provision a reader can
cite: each `section` / `article` / regulation / rule, each schedule (paragraphs stay inside it as
markdown headings), and one introductory unit (long title + preamble). Explanatory notes
(`conclusions`) and signature blocks are NOT enacted text and are left out, by design.

The feed is the "latest" view with tracked changes: `<del>` is text no longer in force (repealed or
substituted) and is dropped, together with any `<ins>` nested in it; `<ins>` is current text and is
kept. Note references and footnotes (`noteRef`, `authorialNote`) are editorial and are dropped. A
provision whose body is empty after that (repealed in full) yields no unit and is counted.

The XML standard is the one fedlex uses, so the namespace, `tag` and `norm` come from
`cytir.sources.families.fedlex_akn` (not copied). The UK-specific parts are the `del`/`ins` view, the
`hcontainer name=...` provision kinds, and the work URI -> canonical URL rule.

Pure functions, no network, no clock: the same bytes give the same units.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree

from cytir.sources.families.fedlex_akn import AKN, norm, tag

SITE = "https://www.legislation.gov.uk"
#: not enacted / editorial: dropped from the text (and everything below them)
_DROP = {"del", "noteRef", "authorialNote", "commentary", "img", "math"}
#: containers whose descendants are quoted text of ANOTHER instrument: never provisions of this one
_QUOTED = {"quotedStructure", "embeddedStructure", "mod"}
#: a cited provision: element local name, or `hcontainer name=`
_PROVISION_TAGS = {"section": "Section", "article": "Article"}
_PROVISION_HC = {"regulation": "Regulation", "rule": "Rule"}
_INLINE = {"inline", "i", "b", "u", "sup", "sub", "abbr", "term", "ref", "span", "a", "quotedText", "ins",
           "docTitle", "docNumber", "docDate", "date", "organization", "concept"}
_ENG = {"eng", "en"}


class NotAkn(Exception):
    """The body is not an AKN act (a redirect page, an HTML error, a metadata-only record)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Section:
    e_id: str
    kind: str                 # intro | provision | schedule
    label: str                # "Section 3", "Schedule 1", "Introductory text"
    heading: str
    url: str
    body: str                 # markdown, without the document header
    words: int = 0


@dataclass
class Item:
    work_path: str            # "ukpga/2026/10"
    type: str                 # ukpga | uksi | ...
    title: str
    citation: str             # "2026 c. 10", "S.I. 2005/3593"
    work_url: str
    sections: list[Section] = field(default_factory=list)
    empty_provisions: int = 0


def _ln(e) -> str:
    return etree.QName(e).localname


def _inline(e) -> str:
    """Text of `e` in document order, current-view: no `del`, no notes."""
    parts: list[str] = []

    def visit(x):
        if x.text:
            parts.append(x.text)
        for c in x:
            if not isinstance(c.tag, str):
                if c.tail:
                    parts.append(c.tail)
                continue
            if _ln(c) not in _DROP:
                visit(c)
            if c.tail:
                parts.append(c.tail)

    visit(e)
    return norm("".join(parts))


def _long_title(lt) -> str:
    """The long title as one line: its paragraphs joined by a space (the XML has no space between them)."""
    ps = [_inline(p) for p in lt.iter(tag("p"))]
    return norm(" ".join(x for x in ps if x)) if ps else _inline(lt)


def _child(e, name):
    for c in e:
        if isinstance(c.tag, str) and _ln(c) == name:
            return c
    return None


def _table(t, out: list[str]) -> None:
    rows = []
    # the table is XHTML inside `<foreign>`; its cells hold AKN `<p>`
    for tr in (x for x in t.iter() if isinstance(x.tag, str) and _ln(x) == "tr"):
        cells = [(_ln(c), " ".join(filter(None, (_inline(p) for p in c.iter(tag("p")))))
                  or _inline(c)) for c in tr if isinstance(c.tag, str) and _ln(c) in ("th", "td")]
        if cells:
            rows.append(cells)
    for i, cells in enumerate(rows):
        out.append("| " + " | ".join(c.replace("|", "/") for _, c in cells) + " |")
        if i == 0 and all(k == "th" for k, _ in cells):
            out.append("|" + " --- |" * len(cells))


def _blocks(e, out: list[str], depth: int) -> None:
    n = _ln(e)
    if n in _DROP:
        return
    if n in ("p", "th", "td"):
        t = _inline(e)
        if t:
            out.append(t)
        return
    if n == "table":
        _table(e, out)
        return
    if n in ("num", "heading"):
        return
    kids = [c for c in e if isinstance(c, etree._Element) and isinstance(c.tag, str)]
    if all(_ln(c) in _INLINE or _ln(c) in _DROP for c in kids):
        t = _inline(e)                                     # a text leaf (subheading, block, ...)
        if t:
            out.append(t)
        return
    num_el, head_el = _child(e, "num"), _child(e, "heading")
    num = _inline(num_el) if num_el is not None else ""
    head = _inline(head_el) if head_el is not None else ""
    body: list[str] = []
    if n == "blockList":
        for it in kids:
            if _ln(it) == "item":
                lines: list[str] = []
                for c in it:
                    if isinstance(c.tag, str):
                        _blocks(c, lines, depth + 1)
                n_it = _child(it, "num")
                pre = _inline(n_it) if n_it is not None else ""
                if lines:
                    lines[0] = (pre + " " + lines[0]).strip()
                    body.extend(lines)
            else:
                _blocks(it, body, depth)
        out.extend(body)
        return
    for c in kids:
        if _ln(c) not in ("num", "heading"):
            _blocks(c, body, depth + 1)
    if head:
        out.append("#" * min(depth + 3, 6) + " " + (num + " " + head).strip())
        out.extend(body)
    else:
        if num:
            if body:
                body[0] = (num + " " + body[0]).strip()
            else:
                body = [num]
        out.extend(body)


def _render(e, depth: int = 0) -> str:
    lines: list[str] = []
    for c in e:
        if isinstance(c.tag, str) and _ln(c) not in ("num", "heading"):
            _blocks(c, lines, depth)
    return "\n\n".join(x for x in lines if x.strip())


def _in_quoted(e, stop) -> bool:
    for a in e.iterancestors():
        if a is stop:
            return False
        if _ln(a) in _QUOTED:
            return True
    return False


def _label(kind_word: str, num: str, e_id: str) -> str:
    num = num.strip().rstrip(".").strip()
    if not num:
        num = e_id.split("-", 1)[-1]
    if re.match(r"^[A-Za-z]{3,}\b", num):                     # "Article 1", "SCHEDULE 1"
        return num.title() if num.isupper() else num
    return f"{kind_word} {num}"


def _section_url(work_url: str, e_id: str) -> str:
    m = re.match(r"^(section|article|regulation|rule|schedule|annex)(?:-(.+))?$", e_id)
    if not m:
        return f"{work_url}#{e_id}"
    return f"{work_url}/{m.group(1)}" + (f"/{m.group(2)}" if m.group(2) else "")


def work_uri(body: bytes) -> str:
    """The document's own work URI (`FRBRWork/FRBRthis`), exactly as the metadata states it.

    Raises `NotAkn` for a body that is not an AKN act or carries no work URI. The caller decides
    what the URI's host means; this function never reads a file name."""
    try:
        root = etree.fromstring(body)
    except etree.XMLSyntaxError as exc:
        raise NotAkn(f"xml_syntax_error:{exc}"[:120])
    if root.tag != tag("akomaNtoso") or len(root) == 0:
        raise NotAkn("not_akn_act")
    this = root[0].find(".//" + tag("FRBRWork") + "/" + tag("FRBRthis"))
    uri = this.get("value") if this is not None else ""
    if not uri:
        raise NotAkn("no_work_uri")
    return uri


def parse(body: bytes) -> Item:
    head = body.lstrip()[:300].lower()
    if not head.startswith(b"<?xml") and not head.startswith(b"<akomantoso"):
        raise NotAkn("not_akn_redirect_or_html")
    try:
        root = etree.fromstring(body)
    except etree.XMLSyntaxError as exc:
        raise NotAkn(f"xml_syntax_error:{exc}"[:120])
    if root.tag != tag("akomaNtoso") or len(root) == 0:
        raise NotAkn("not_akn_act")
    act = root[0]
    work = act.find(".//" + tag("FRBRWork"))
    lang = act.find(".//" + tag("FRBRlanguage"))
    if lang is not None and lang.get("language", "eng") not in _ENG:
        raise NotAkn("non_english_expression")
    uri = work.find(tag("FRBRthis")).get("value") if work is not None else ""
    m = re.search(r"/id/(.+)$", uri or "")
    if not m:
        raise NotAkn("no_work_uri")
    work_path = m.group(1).strip("/")
    work_url = f"{SITE}/{work_path}"
    name_el = work.find(tag("FRBRname"))
    citation = norm(name_el.get("value")) if name_el is not None else ""
    pref = act.find(tag("preface"))
    title = ""
    if pref is not None:
        dt = pref.find(".//" + tag("docTitle"))
        if dt is not None:
            title = _inline(dt)
        elif (lt := pref.find(tag("longTitle"))) is not None:
            title = _long_title(lt)             # EU instruments carry only a long title
    item = Item(work_path=work_path, type=work_path.split("/")[0], title=title or citation,
                citation=citation, work_url=work_url)
    bodyel = act.find(tag("body"))
    if bodyel is None:
        raise NotAkn("no_body_metadata_only")

    # introductory unit: the long title (when it is not the document title) and the preamble
    intro: list[str] = []
    if pref is not None:
        lt = pref.find(tag("longTitle"))
        if lt is not None and _long_title(lt) != item.title:
            intro.append(_long_title(lt))
    pre = act.find(tag("preamble"))
    if pre is not None:
        intro.extend(x for x in (_render(pre),) if x)
    if "".join(intro).strip():
        item.sections.append(Section("preamble", "intro", "Introductory text", "",
                                     f"{work_url}/introduction", "\n\n".join(intro)))

    def emit(e, kind, label, heading):
        txt = _render(e, 0)
        if not txt.strip():
            item.empty_provisions += 1
            return
        e_id = e.get("eId") or label.lower().replace(" ", "-")
        item.sections.append(Section(e_id, kind, label, heading, _section_url(work_url, e_id), txt))

    orphans: list[str] = []

    def walk(parent):
        for e in parent:
            if not isinstance(e.tag, str) or _ln(e) in _DROP:
                continue
            n, name = _ln(e), e.get("name")
            num_el, head_el = _child(e, "num"), _child(e, "heading")
            num = _inline(num_el) if num_el is not None else ""
            head = _inline(head_el) if head_el is not None else ""
            if n in _PROVISION_TAGS or (n == "hcontainer" and name in _PROVISION_HC):
                word = _PROVISION_TAGS.get(n) or _PROVISION_HC[name]
                emit(e, "provision", _label(word, num, e.get("eId") or ""), head)
            elif n == "hcontainer" and name == "schedule":
                emit(e, "schedule", _label("Schedule", num, e.get("eId") or "schedule"), head)
            elif n == "hcontainer" and name in ("signatures", "signatureBlock"):
                continue
            elif n in ("part", "chapter", "hcontainer", "level", "division", "crossheading", "subheading",
                       "tblock", "container") and any(
                    isinstance(c.tag, str) and _ln(c) in (set(_PROVISION_TAGS) | {"hcontainer", "part", "chapter",
                                                                               "level", "paragraph"}) for c in e):
                walk(e)
            elif n in ("heading", "num"):
                continue
            else:
                t = _render(e) if _ln(e) not in ("p",) else _inline(e)
                if t.strip():
                    orphans.append(t)

    walk(bodyel)
    if orphans:
        txt = "\n\n".join(orphans)
        item.sections.append(Section("body", "provision", "Provisions", "", work_url, txt))
    for s in item.sections:
        s.words = len(s.body.split())
    return item


def compose_text(item: Item, s: Section) -> str:
    """The markdown the chunker sees: document title, citation, the provision label and heading, text."""
    head = f"{s.label}. {s.heading}" if s.heading else s.label
    return f"# {item.title}\n\n{item.citation}\n\n## {head}\n\n{s.body}\n"


def compose_title(item: Item, s: Section) -> str:
    t = f"{item.title} ({item.citation}) — {s.label}" if item.citation else f"{item.title} — {s.label}"
    return f"{t} — {s.heading}" if s.heading else t
