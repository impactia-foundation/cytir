"""BFS DAM asset -> ONE authoritative-locator passage (`bfs_dam_exact`).

Moved out of `ingest_bfs_dam.py` (task 1101, S7) so the source family
(`cytir.sources.families.bfs_dam`) and the legacy CLI share one builder. The functions are the
driver's own, UNCHANGED: `build_passage` still names every drop reason in the `reasons`
counter it is given (corpus-doctrine #2).
"""
from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import asdict

from cytir.core.common import Passage, extraction_junk, extractive_snippet, low_quality


DOMAIN = "bfs.admin.ch"
ASSET_URL = "https://www.bfs.admin.ch/asset/{lang}/{dam_id}"
VALID_LANGS = ("de", "fr", "it", "en")
MIN_TITLE_WORDS = 4     # descriptive-title threshold when no summary present
MAX_TEXT = 6000

AGENCY = {
    "de": "Bundesamt für Statistik (BFS)",
    "fr": "Office fédéral de la statistique (OFS)",
    "it": "Ufficio federale di statistica (UST)",
    "en": "Swiss Federal Statistical Office (FSO)",
}
THEME_LABEL = {"de": "Thema", "fr": "Thème", "it": "Tema", "en": "Topic"}
PERIOD_LABEL = {"de": "Zeitraum", "fr": "Période", "it": "Periodo", "en": "Period"}

_TAG_RE = re.compile(r"<[^>]+>")
_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def norm_lang(v: str | None) -> str | None:
    """Map the asset language field to a single en/de/fr/it code. Multi-language
    assets ('DE/EN/FR') collapse to the first listed language (one passage, not
    N near-duplicates). Unknown -> None (dropped)."""
    if not v:
        return None
    first = re.split(r"[/,]", v.strip())[0].strip().lower()
    return first if first in VALID_LANGS else None


def strip_html(s: str | None) -> str:
    if not s:
        return ""
    return html.unescape(_TAG_RE.sub(" ", s)).strip()


def summary_text(desc: dict) -> str:
    """The asset's own machine-readable summary (plain text)."""
    a = (desc.get("abstractShort") or "").strip()
    if a:
        return a
    ss = desc.get("shortSummary") or {}
    if isinstance(ss, dict):
        return strip_html(ss.get("raw") or ss.get("html"))
    return strip_html(ss)


def theme_path(desc: dict) -> list[str]:
    """Ordered, de-duplicated prodima theme labels (Gesundheit — ... — ...)."""
    out, seen = [], set()
    for c in (desc.get("categorization") or {}).get("prodima") or []:
        name = (c.get("name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def is_future_embargo(bfs: dict, now_iso: str) -> bool:
    emb = bfs.get("embargo")
    return bool(emb and isinstance(emb, str) and emb > now_iso)


def last_seen_of(bfs: dict, period: str) -> str | None:
    """Freshness signal: the diffusion/embargo date (YYYY-MM-DD) if present,
    else the bibliography period string."""
    emb = bfs.get("embargo")
    if emb and isinstance(emb, str) and len(emb) >= 10:
        return emb[:10]
    return period or None


def build_passage(asset: dict, now_iso: str, reasons: Counter) -> dict | None:
    desc = asset.get("description") or {}
    bfs = asset.get("bfs") or {}
    ids = asset.get("ids") or {}

    lang = norm_lang(desc.get("language"))
    if not lang:
        reasons["bad_lang"] += 1
        return None
    dam_id = ids.get("damId")
    if not dam_id:
        reasons["no_damid"] += 1
        return None
    if is_future_embargo(bfs, now_iso):
        reasons["future_embargo"] += 1
        return None

    titles = desc.get("titles") or {}
    main = (titles.get("main") or "").strip()
    sub = (titles.get("sub") or "").strip()
    sup = (titles.get("super") or "").strip()
    if not main:
        reasons["no_title"] += 1
        return None

    summary = summary_text(desc)
    title_words = len({w.lower() for w in _WORD_RE.findall(main)})
    if not summary and title_words < MIN_TITLE_WORDS:
        # pure-data / code-only asset with no prose -> the number-soup class
        reasons["thin_no_prose"] += 1
        return None

    themes = theme_path(desc)
    period = ((desc.get("bibliography") or {}).get("period") or "").strip()
    amg = ((bfs.get("articleModelGroup") or {}).get("name")
           or (bfs.get("articleModel") or {}).get("name") or "").strip()

    parts: list[str] = []
    if sup and sup.lower() not in main.lower():
        parts.append(sup + ".")
    parts.append(main + ("." if not main.endswith((".", "!", "?")) else ""))
    if sub and sub.lower() != main.lower():
        parts.append(sub + ("." if not sub.endswith((".", "!", "?")) else ""))
    if summary and summary.lower() not in main.lower():
        parts.append(summary if summary.endswith((".", "!", "?")) else summary + ".")
    if themes:
        parts.append(f"{THEME_LABEL[lang]}: " + " — ".join(themes) + ".")
    if period:
        parts.append(f"{PERIOD_LABEL[lang]}: {period}.")
    tail = AGENCY[lang]
    if amg:
        tail = f"{amg}. {tail}"
    parts.append(tail + ".")
    text = " ".join(p for p in parts if p).strip()[:MAX_TEXT]

    if low_quality(text):
        reasons["low_quality"] += 1
        return None
    jr = extraction_junk(text)
    if jr:
        reasons[f"junk_{jr}"] += 1
        return None

    snippet = extractive_snippet(summary or main, 25)
    p = Passage(
        url=ASSET_URL.format(lang=lang, dam_id=dam_id),
        title=main,
        snippet=snippet[:200],
        text=text,
        lang=lang,
        domain=DOMAIN,
        tier="A",
        source_type="gov-stat",
        trust_tier=0.95,
        last_seen=last_seen_of(bfs, period),
    )
    d = asdict(p)                # full dict incl. `text` (embed reads r["text"])
    d["country"] = "CH"
    d["provenance"] = "bfs_dam"
    return d
