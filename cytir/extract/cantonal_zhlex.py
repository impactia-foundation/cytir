"""ZH-Lex law-level passage (`scratch_ch_cantonal_law_exact`, portal `zhlex`).

Moved out of `ingest_zhlex.py` (task 1108, S7): the row construction that sat inline in the
driver's `main()` loop, cut out UNCHANGED into `law_row` so the source family
(`cytir.sources.families.ch_cantonal_law`) and the legacy CLI share one builder.
"""
from __future__ import annotations

from cytir.core.common import Passage, extractive_snippet

HOST = "https://www.zh.ch"


def law_row(ref: str, it: dict) -> dict:
    """ONE law-level row for an in-force index item that has a title and a link."""
    title = (it.get("enactmentTitle") or "").strip()
    link = it.get("link") or ""
    date = (it.get("enactmentDate") or "").strip()
    url = HOST + link if link.startswith("/") else link
    disp_title = f"{title} — ZH-Lex {ref}"[:300]
    body = (f"{title} · Zürcher Gesetzessammlung (Loseblattsammlung) {ref} · "
            f"Kanton Zürich" + (f" · Erlass vom {date}" if date else ""))
    p = Passage(
        url=url, title=disp_title,
        snippet=extractive_snippet(body, 25), text=body,
        lang="de", domain="www.zh.ch", tier="A",
        source_type="official", trust_tier=1.0, anchor="law",
        last_seen=date or None,
    )
    return {**p.payload(), "text": p.text,
            "country": "CH", "provenance": "ch_cantonal_law", "canton": "ZH"}


def skip_reason(it: dict, include_repealed: bool = False) -> str | None:
    """Why the driver does not emit an index item: `no_title_or_link` | `repealed` | None (kept)."""
    if not (it.get("enactmentTitle") or "").strip() or not (it.get("link") or ""):
        return "no_title_or_link"
    if (it.get("withdrawalDate") or "").strip() and not include_repealed:
        return "repealed"
    return None
