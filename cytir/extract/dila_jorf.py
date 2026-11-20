"""DILA JORF text -> ONE title-only locator record (`jorf_recent`).

Moved out of the retired `ingest_dila_jorf` module (task 696) so the source family
(`cytir.sources.families.dila_jorf`) and the legacy CLI share one parser; it now says WHY a text was
not taken (corpus-doctrine #2). The floors are the task-140 caps and are UNCHANGED by the
declaration (task 695: "floor unchanged"): publication year >= `MIN_YEAR_DEFAULT`; LOI / ORDONNANCE
/ DECRET always; an ARRETE only when its title is substantive and not a nomination/decoration.
"""
from __future__ import annotations

import re
from dataclasses import asdict

from lxml import etree

from cytir.core.common import Passage, extraction_junk, extractive_snippet, low_quality

_WS = re.compile(r"\s+")
MIN_YEAR_DEFAULT = 2023
MIN_ARRETE_TITLE_WORDS = 9       # a bare "Arrêté du … portant nomination …" is short
KEEP_ALWAYS = {"LOI", "ORDONNANCE", "DECRET", "DÉCRET"}
_NOISE = re.compile(r"(?i)portant (nomination|promotion|titularisation|"
                    r"élévation|inscription au tableau|attribution de|"
                    r"radiation)|au grade de|à titre (posthume|étranger)|"
                    r"légion d'honneur|ordre national du mérite|"
                    r"médaille (militaire|d'honneur)")


def norm(s: str) -> str:
    return _WS.sub(" ", (s or "")).strip()


def parse_text_bytes(body: bytes, min_year: int = MIN_YEAR_DEFAULT,
                     keep_all: bool = False) -> tuple[dict | None, str]:
    """XML bytes of one JORF text -> `(record, "kept")` or `(None, reason)`."""
    try:
        root = etree.fromstring(body)
    except Exception as exc:  # noqa: BLE001
        return None, f"xml_parse_error:{exc.__class__.__name__}"
    if etree.QName(root).localname != "TEXTE_VERSION":
        return None, f"unknown_root:{etree.QName(root).localname}"
    tid = (root.findtext(".//META_COMMUN/ID") or "").strip()
    nature = (root.findtext(".//META_COMMUN/NATURE") or "").strip().upper()
    date_publi = (root.findtext(".//META_TEXTE_CHRONICLE/DATE_PUBLI") or "").strip()
    nor = (root.findtext(".//META_TEXTE_CHRONICLE/NOR") or "").strip()
    origine = (root.findtext(".//META_TEXTE_CHRONICLE/ORIGINE_PUBLI") or "").strip()
    eli = (root.findtext(".//META_COMMUN/ID_ELI") or "").strip()
    titre_full = norm(root.findtext(".//META_TEXTE_VERSION/TITREFULL") or "")
    titre = norm(root.findtext(".//META_TEXTE_VERSION/TITRE") or "")
    title = titre_full or titre
    if not tid or not title:
        return None, "no_id_or_title"
    year = int(date_publi[:4]) if date_publi[:4].isdigit() else 0
    if not keep_all and year < min_year:
        return None, "below_year_floor"

    n_words = len(title.split())
    if not keep_all and nature not in KEEP_ALWAYS:
        if n_words < MIN_ARRETE_TITLE_WORDS or _NOISE.search(title):
            return None, "not_substantive"
    elif _NOISE.search(title) and n_words < MIN_ARRETE_TITLE_WORDS:
        return None, "not_substantive"

    label = {"LOI": "Loi", "ORDONNANCE": "Ordonnance", "DECRET": "Décret",
             "DÉCRET": "Décret", "ARRETE": "Arrêté", "ARRÊTÉ": "Arrêté"}.get(
                 nature, nature.capitalize() or "Texte")
    parts = [title]
    parts.append(f"{label}. {origine}." if origine else f"{label}.")
    if nor:
        parts.append(f"NOR : {nor}.")
    if eli:
        parts.append(f"ELI : {eli}")
    text = "\n".join(parts)
    if low_quality(text):
        return None, "low_quality"
    junk = extraction_junk(text)
    if junk:
        return None, f"extraction_junk:{junk}"
    url = f"https://www.legifrance.gouv.fr/jorf/id/{tid}"
    p = Passage(url=url, title=title[:300], snippet=extractive_snippet(title, 30),
                text=text, lang="fr", domain="legifrance.gouv.fr", tier="A",
                source_type="official", trust_tier=1.0,
                last_seen=(date_publi if len(date_publi) == 10 else None))
    d = asdict(p)
    d.update(country="FR", provenance="jorf_recent", nature=label)
    return {k: v for k, v in d.items() if v is not None}, "kept"
