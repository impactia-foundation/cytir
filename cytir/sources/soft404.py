"""The framework's soft-404 detector (task 683 S7 part 2; corpus-doctrine #2, #7).

A soft 404 is an error page served with HTTP 200. The transport says "fine", so no adapter's
status check sees it, and its text ("Ein Fehler ist aufgetreten") becomes passages that an agent
can be handed as if they were the publisher's content. ElCom's sitemap lists `/de/404` and
`/de/500`; both were admitted by the pack route and by the first `SourceFamily` port.

The detector is DATA plus this one function: `data/soft-404-signals.v1.json` declares the
title and head patterns (DE/FR/IT/EN) and the length guard. `cytir.sources.run_family` applies it to
every unit's chunked records, for every family, so a new adapter cannot forget it. A match ends
the unit `excluded(soft_404:<signal>)`; the sealed bytes stay in L0. The index is never mutated
here.

Rule: the unit's total text is at most `max_unit_chars` AND (its title matches a `title_regex`
OR the head of its text matches a `head_regex`). Length is the guard: a long page that merely
talks about HTTP errors is admitted.
"""
from __future__ import annotations

from cytir.core.paths import data_path

import functools
import json
import pathlib
import re
import urllib.parse
from typing import Iterable, Mapping

DEFAULT_PATH = data_path("soft-404-signals.v1.json")
SCHEMA = "aw.soft-404-signals.v1"


@functools.lru_cache(maxsize=4)
def _load(path: str) -> dict:
    doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, expected {SCHEMA!r}")
    return {
        "max": int(doc["max_unit_chars"]), "head": int(doc["head_chars"]),
        "title": [re.compile(p, re.I) for p in doc["title_regex"]],
        "head_rx": [re.compile(p, re.I) for p in doc["head_regex"]],
        # task 1363: read by the tombstone rule only; the ingest path ignores it
        "error_path": re.compile(doc["error_path_regex"], re.I) if doc.get("error_path_regex") else None,
    }


def detect(title: str, text: str, *, path: str | pathlib.Path | None = None) -> str | None:
    """`title`, `text` of one unit (all its records joined) -> the signal that matched, or None."""
    sig = _load(str(path or DEFAULT_PATH))
    if len(text) > sig["max"]:
        return None
    for i, rx in enumerate(sig["title"]):
        if title and rx.search(title):
            return f"title_{i}"
    head = text[: sig["head"]]
    for i, rx in enumerate(sig["head_rx"]):
        if rx.search(head):
            return f"head_{i}"
    return None


def error_path(url: "str | None", *, path: "str | pathlib.Path | None" = None) -> bool:
    """True when the url's PATH is an error-page path (`/it/500`, `.../pagina-defecta.html`): the
    second, independent gate of the `soft_404` tombstone rule (task 1363)."""
    rx = _load(str(path or DEFAULT_PATH))["error_path"]
    return bool(rx and rx.search(urllib.parse.urlsplit(str(url or "")).path))


def detect_records(records: Iterable[Mapping], *, path: str | pathlib.Path | None = None
                   ) -> str | None:
    """Apply `detect` to one unit's chunked records (their titles and text)."""
    recs = list(records)
    if not recs:
        return None
    text = "\n".join(str(r.get("text") or "") for r in recs)
    title = next((str(r["title"]) for r in recs if r.get("title")), "")
    return detect(title, text, path=path)
