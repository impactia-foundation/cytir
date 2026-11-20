"""Query-side normaliser for glued article numbers (task 2778, flag `AW_QUERY_GLUED_ARTICLE`, OFF).

`\\w+` makes `336c` one token, while fedlex titles and snippets spell `Art. 336 c`, which the
tokenizer reads as `336` (the lone `c` is dropped at len < 2, both sides). A glued query token
therefore has no posting. This appends the split form `336 c` after the glued one, so the lexical
leg also asks for `336`. Query side only: the index is untouched. Pure, no I/O.
"""
from __future__ import annotations

import re

_GLUED = re.compile(r"\b(\d+)([a-z])\b", re.IGNORECASE)


def glued_article_variants(q: str) -> str:
    """`q` plus, for each `<digits><letter>` word, its split form; `q` unchanged if none."""
    extra = [f"{m.group(1)} {m.group(2)}" for m in _GLUED.finditer(q)]
    return f"{q} {' '.join(extra)}" if extra else q


# task 2792: exact-article-in-title signal (flag `AW_ARTICLE_EXACT_W`, weight 0 = OFF).
# The lexical leg reads `Art. 336 b` and `Art. 336 c` as the same bag (the lone letter is dropped),
# so rank 1 needs a stage-2 signal: does the candidate's own article id equal the asked one?
_QUERY_ART = re.compile(r"\b(?:art(?:ikel|icle|icolo|icoli|icles)?\.?)\s*(\d+)(?:\s?([a-z])\b)?", re.IGNORECASE)
_TITLE_ART = re.compile(r"\bArt\.?\s*(\d+)(?:\s([a-z])\b)?", re.IGNORECASE)
_FRAGMENT_ART = re.compile(r"#art_(\d+)(?:_([a-z]))?(?:_|$)")

ArticleId = tuple[str, str]  # (digits, lower-case letter or "")


def query_article_ids(q: str) -> frozenset[ArticleId]:
    """Article ids named by `Art.|Artikel|Article|...` + number + optional letter (glued or split)."""
    return frozenset((m.group(1), (m.group(2) or "").lower()) for m in _QUERY_ART.finditer(q or ""))


def row_article_id(title: str, url: str) -> ArticleId | None:
    """The candidate's own article id: the url fragment `#art_336_c`, else the first `Art. 336 c` in the title."""
    m = _FRAGMENT_ART.search(url or "") or _TITLE_ART.search(title or "")
    return (m.group(1), (m.group(2) or "").lower()) if m else None


def article_exact(ids: frozenset[ArticleId], title: str, url: str) -> bool:
    """True when the query names an article and the candidate carries exactly that number + letter."""
    return bool(ids) and row_article_id(title, url) in ids


# task 2797: statute narrowing (flag `AW_ARTICLE_EXACT_STATUTE`, 0 = OFF). `article_exact` matches
# number+letter whatever the statute, so `article 7 de la Charte canadienne` boosted Swiss BV art. 7.
# When the query NAMES a statute (abbreviation, SR/RS number, or DE/FR/IT/EN name), the boost is kept
# only for candidates that carry it. Fedlex titles read `Obligationenrecht (OR; SR 220) — Art. 336`
# and urls `/eli/cc/27/317_321_377/de#art_336`. A narrowing: it only ever REMOVES the boost.
# (key, SR, ELI path, url markers, case-sensitive abbreviations, case-folded names). The ELI/names are
# `data/fedlex-law-names.v1.json`; key "x:..." rows are non-fedlex statutes with no SR.
_STATUTES: tuple[tuple[str, str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    ("220", "220", "cc/27/317_321_377", (), ("OR", "CO"),
     ("obligationenrecht", "code des obligations", "codice delle obbligazioni", "code of obligations")),
    ("210", "210", "cc/24/233_245_233", (), ("ZGB", "CC"),
     ("zivilgesetzbuch", "code civil", "codice civile", "civil code")),
    ("311.0", "311.0", "cc/54/757_781_799", ("gesetze-im-internet.de/stgb/",), ("StGB", "CP"),
     ("strafgesetzbuch", "code pénal", "code penal", "codice penale", "criminal code")),
    ("312.0", "312.0", "cc/2010/267", ("gesetze-im-internet.de/stpo/",), ("StPO", "CPP"),
     ("strafprozessordnung", "code de procédure pénale", "codice di procedura penale",
      "criminal procedure code")),
    ("272", "272", "cc/2010/262", ("gesetze-im-internet.de/zpo/",), ("ZPO", "CPC"),
     ("zivilprozessordnung", "code de procédure civile", "codice di procedura civile",
      "civil procedure code")),
    ("281.1", "281.1", "cc/11/529_488_529", (), ("SchKG", "LP", "LEF"),
     ("schuldbetreibungs- und konkursgesetz", "loi sur la poursuite pour dettes",
      "legge sulla esecuzione e sul fallimento", "debt enforcement and bankruptcy act")),
    ("142.20", "142.20", "cc/2007/758", (), ("AIG", "AIGE", "LEI", "LStrI"),
     ("ausländer- und integrationsgesetz", "loi sur les étrangers", "legge sugli stranieri",
      "foreign nationals")),
    ("642.11", "642.11", "cc/1991/1184_1184_1184", (), ("DBG", "LIFD"),
     ("bundesgesetz über die direkte bundessteuer", "loi sur l'impôt fédéral direct",
      "legge sull'imposta federale diretta")),
    ("101", "101", "cc/1999/404", (), ("BV", "Cst", "Cst."),
     ("bundesverfassung", "constitution fédérale", "costituzione federale", "federal constitution")),
    ("142.31", "142.31", "cc/1999/358", (), ("AsylG", "LAsi"),
     ("asylgesetz", "loi sur l'asile", "legge sull'asilo", "asylum act")),
    ("831.10", "831.10", "cc/63/837_843_843", (), ("AHVG", "LAVS"),
     ("alters- und hinterlassenenversicherungsgesetz", "loi sur l'assurance-vieillesse",
      "legge sull'assicurazione per la vecchiaia")),
    ("741.01", "741.01", "cc/1959/679_705_685", (), ("SVG", "LCR"),
     ("strassenverkehrsgesetz", "loi sur la circulation routière", "legge sulla circolazione stradale")),
    ("831.20", "831.20", "cc/1959/827_857_845", (), ("IVG", "LAI"),
     ("invalidenversicherungsgesetz", "loi sur l'assurance-invalidité",
      "legge sull'assicurazione per l'invalidità")),
    ("641.20", "641.20", "cc/2009/615", (), ("MWSTG", "MWST", "LTVA", "LIVA"),
     ("mehrwertsteuergesetz", "loi sur la tva", "legge sull'iva", "value added tax act")),
    ("231.1", "231.1", "cc/1993/1798_1798_1798", (), ("URG", "LDA"),
     ("urheberrechtsgesetz", "loi sur le droit d'auteur", "legge sul diritto d'autore", "copyright act")),
    ("830.1", "830.1", "cc/2002/510", (), ("ATSG", "LPGA"),
     ("allgemeinen teil des sozialversicherungsrechts", "partie générale du droit des assurances sociales",
      "parte generale del diritto delle assicurazioni sociali")),
    ("235.1", "235.1", "cc/2022/491", (), ("DSG", "LPD", "nDSG"),
     ("datenschutzgesetz", "loi sur la protection des données", "legge sulla protezione dei dati",
      "data protection act")),
    ("822.11", "822.11", "", (), ("ArG", "LTr"), ("arbeitsgesetz", "loi sur le travail", "legge sul lavoro")),
    # non-Swiss statutes a Swiss-lookalike query can name: no fedlex candidate carries them
    ("x:gg", "", "", ("gesetze-im-internet.de/gg/",), ("GG",), ("grundgesetz", "basic law")),
    ("x:bgb", "", "", ("gesetze-im-internet.de/bgb/",), ("BGB",), ("bürgerliches gesetzbuch",)),
    ("x:ca-charter", "", "", (), (),
     ("charte canadienne", "canadian charter", "charter of rights and freedoms")),
)
_SR_IN_QUERY = re.compile(r"\b(?:SR|RS)\s*(\d{2,3}(?:\.\d+)*)\b")
_SR_IN_TITLE = re.compile(r"\bSR\s*(\d{2,3}(?:\.\d+)*)\s*\)")
_ABBREV = {k: re.compile(r"(?<![\w.])(?:" + "|".join(re.escape(a) for a in abbr) + r")(?![\w])")
           for k, _sr, _e, _m, abbr, _n in _STATUTES if abbr}


def query_statutes(q: str) -> frozenset[str]:
    """Statute keys the query names: an abbreviation (`OR`, `Cst.`), an `SR`/`RS` number, or a law name.

    Abbreviations match case-sensitively as whole words; names match case-folded. A key is the SR
    number (`220`) or `x:<name>` for a statute outside fedlex. Empty set: the query names none.
    """
    q = q or ""
    low = q.casefold()
    keys = {m.group(1) for m in _SR_IN_QUERY.finditer(q)}
    for key, _sr, _eli, _marks, _abbr, names in _STATUTES:
        if (key in _ABBREV and _ABBREV[key].search(q)) or any(n in low for n in names):
            keys.add(key)
    return frozenset(keys)


def row_carries_statute(keys: frozenset[str], title: str, url: str) -> bool:
    """True when the candidate's title hint `(OR; SR 220)`, its url ELI path or a url marker names a key."""
    m = _SR_IN_TITLE.search(title or "")
    if m and m.group(1) in keys:
        return True
    url = url or ""
    for key, _sr, eli, marks, _abbr, _names in _STATUTES:
        if key in keys and ((eli and f"/eli/{eli}/" in url) or any(mk in url for mk in marks)):
            return True
    return False


def article_exact_narrowed(ids: frozenset[ArticleId], keys: frozenset[str], title: str, url: str) -> bool:
    """`article_exact`, and when the query names a statute, only for a candidate carrying that statute."""
    return article_exact(ids, title, url) and (not keys or row_carries_statute(keys, title, url))
