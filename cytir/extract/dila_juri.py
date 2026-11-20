"""DILA jurisprudence decision -> ONE locator record (Cour de cassation CASS, Conseil d'Etat JADE).

Moved out of the retired `ingest_dila_juri` module (task 696) so the source family
(`cytir.sources.families.dila_juri`) and the legacy CLI share one parser. The record is the
shape the ingester always wrote: header (TITRE, or court + date), `Solution :`, `ECLI :`,
`Sommaire :` (headnote, first 1,200 chars), a body excerpt capped at `MAX_BODY_CHARS`,
the whole text capped at 3,600 chars. NOT a full-text mirror (spec minimum-content rule).

Unlike the legacy function this one says WHY a decision was not taken: every unit of a
family run reaches exactly one terminal with a reason (corpus-doctrine #2).

Year floors are an argument, not constants of the parser. `FLOORS_P0` is the task-140 cap
(CASS >= 2010, JADE >= 2024, recueil landmarks >= 2005) that the legacy CLI still passes;
`floors=None` is what the declared family uses (task 695 section 4 lifted the floors: the
quality floor stays).
"""
from __future__ import annotations

import re
from dataclasses import asdict

from lxml import etree

from cytir.core.common import Passage, extraction_junk, extractive_snippet, low_quality

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{3,}")

MAX_BODY_CHARS = 2000            # locator excerpt cap: never a full-text mirror

#: root tag -> (court label, provenance, legifrance path, id prefix)
ROOTS = {
    "TEXTE_JURI_JUDI": ("Cour de cassation", "cass_juri", "juri", "JURITEXT"),
    "TEXTE_JURI_ADMIN": ("Conseil d'État", "ce_juri", "ceta", "CETATEXT"),
}

#: the task-140 P0 recency caps; the legacy CLI passes these, the declared family does not
FLOORS_P0 = {"TEXTE_JURI_JUDI": 2010, "TEXTE_JURI_ADMIN": 2024, "landmark": 2005}


def clean(raw: str) -> str:
    if not raw:
        return ""
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)</(p|dd|dt|div|li|alinea)>", "\n", raw)
    txt = _TAG.sub(" ", raw)
    txt = txt.replace("&amp;", "&").replace("&nbsp;", " ").replace("&#13;", " ")
    txt = _WS.sub(" ", txt)
    txt = "\n".join(line.strip() for line in txt.splitlines())
    return _NL.sub("\n\n", txt).strip()


def _first_text(root, *tags) -> str:
    for t in tags:
        el = root.find(f".//{t}")
        if el is not None and (el.text or "").strip():
            return el.text.strip()
    return ""


def is_landmark(root, court_root: str) -> bool:
    if court_root == "TEXTE_JURI_JUDI":
        pb = root.find(".//PUBLI_BULL")
        return pb is not None and (pb.get("publie") or "").lower() == "oui"
    return _first_text(root, "PUBLI_RECUEIL").upper() in ("A", "B")


def parse_decision_bytes(body: bytes, floors: dict | None = None) -> tuple[dict | None, str]:
    """XML bytes of one decision -> `(record, "kept")` or `(None, reason)`."""
    try:
        root = etree.fromstring(body)
    except Exception as exc:  # noqa: BLE001
        return None, f"xml_parse_error:{exc.__class__.__name__}"
    court_root = etree.QName(root).localname
    if court_root not in ROOTS:
        return None, f"unknown_root:{court_root}"
    court, prov, legi_path, _ = ROOTS[court_root]

    did = (root.findtext(".//META_COMMUN/ID") or "").strip()
    if not did:
        return None, "no_decision_id"
    titre = (root.findtext(".//META_JURI/TITRE") or "").strip()
    date_dec = (root.findtext(".//META_JURI/DATE_DEC") or "").strip()
    solution = (root.findtext(".//META_JURI/SOLUTION") or "").strip()
    ecli = (root.findtext(".//ECLI") or "").strip()
    year = int(date_dec[:4]) if date_dec[:4].isdigit() else 0

    landmark = is_landmark(root, court_root)
    if floors is not None:
        keep = year >= floors[court_root] or (landmark and year >= floors["landmark"])
        if not keep:
            return None, f"below_year_floor:{prov}"

    sommaire = ""
    for tag in ("SOMMAIRE", "ANALYSE", "TEXTE_SOMMAIRE"):
        el = root.find(f".//{tag}")
        if el is not None:
            sommaire = clean(etree.tostring(el, encoding="unicode", method="xml"))
            if sommaire:
                break

    text_body = ""
    cont = root.find(".//BLOC_TEXTUEL/CONTENU")
    if cont is not None:
        text_body = clean(etree.tostring(cont, encoding="unicode", method="xml"))

    header = titre or f"{court} {date_dec}"
    parts = [header]
    if solution:
        parts.append(f"Solution : {solution}")
    if ecli:
        parts.append(f"ECLI : {ecli}")
    if sommaire:
        parts.append(f"Sommaire : {sommaire[:1200]}")
    if text_body:
        parts.append(text_body[:MAX_BODY_CHARS])
    text = "\n".join(parts)[: MAX_BODY_CHARS + 1600]

    if low_quality(text):
        return None, "low_quality"
    junk = extraction_junk(text)
    if junk:
        return None, f"extraction_junk:{junk}"

    snippet = extractive_snippet(sommaire or titre or text_body or header, 30)
    url = f"https://www.legifrance.gouv.fr/{legi_path}/id/{did}"
    p = Passage(url=url, title=header[:300], snippet=snippet, text=text, lang="fr",
                domain="legifrance.gouv.fr", tier="A", source_type="official",
                trust_tier=1.0, last_seen=(date_dec if len(date_dec) == 10 else None))
    d = asdict(p)
    d.update(country="FR", provenance=prov, court=court,
             landmark=bool(landmark), ecli=ecli or None)
    return {k: v for k, v in d.items() if v is not None}, "kept"
