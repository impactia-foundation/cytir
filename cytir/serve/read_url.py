"""`read_url`: a no-JavaScript twin of a result's `url`, where one exists (task 2709).

The authoritative URL a search returns is not always one an agent can read: a fedlex article URL is
an Angular shell. `read_url_for(payload)` answers the readable twin, or None. The field is ADDITIVE:
a result without a twin has no `read_url` key at all (never null), so a response for any other
family stays byte-identical.

Fedlex: the twin is the filestore HTML manifestation of the in-force consolidation. Its path holds
the consolidation date, which only the side table `data/fedlex-versions.v1.json` knows (generated
and size-verified by `cyt source fedlex-versions`). A missing table, an ELI or language the table
does not vouch for, or a malformed url all answer None: a wrong date serves a 9 KB shell with
HTTP 200, so a guess is worse than no field.
"""
from __future__ import annotations

import json
import logging
import re
from urllib.parse import urlsplit

from cytir.core.paths import data_path
from cytir.sources.families.fedlex_contract import fedlex_html_url

from .settings import SETTINGS as _S

log = logging.getLogger("aw_server.read_url")

TABLE_NAME = "fedlex-versions.v1.json"
_FEDLEX_PATH_RE = re.compile(r"^/eli/([a-z0-9/_]+)/(de|fr|it|en)/?$")

_UNSET = object()
_table = _UNSET   # dict | None; None = disabled (file missing or unreadable)


def load_table(path=None) -> dict | None:
    """The side table, or None (one warning) when the file is missing or unreadable."""
    p = path or _S.runtime.fedlex_versions_path or data_path(TABLE_NAME)
    try:
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        if not isinstance(doc, dict):
            raise ValueError("not a JSON object")
        return doc
    except (OSError, ValueError) as e:
        log.warning("read_url disabled: cannot load %s (%s)", p, e)
        return None


def _get_table() -> dict | None:
    global _table
    if _table is _UNSET:
        _table = load_table()
    return _table


def fedlex_read_url(url: str, table: dict | None) -> str | None:
    if not table:
        return None
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host != "fedlex.admin.ch" and not host.endswith(".fedlex.admin.ch"):
        return None
    m = _FEDLEX_PATH_RE.match(parts.path)
    if not m:
        return None
    eli, lang = m.groups()
    date = table.get(eli)
    if not isinstance(date, str) or not re.fullmatch(r"\d{8}", date):
        return None
    if lang not in table.get("_langs", ()) or f"{eli}/{lang}" in table.get("_dropped", ()):
        return None
    twin = fedlex_html_url(eli, date, lang)
    return f"{twin}#{parts.fragment}" if parts.fragment else twin


def read_url_for(payload: dict, table: "dict | None | object" = _UNSET) -> str | None:
    """The readable twin of `payload["url"]`, or None. Family dispatch on the url's host."""
    url = payload.get("url")
    if not isinstance(url, str):
        return None
    return fedlex_read_url(url, _get_table() if table is _UNSET else table)


def attach_read_urls(results: list[dict]) -> None:
    """Add `read_url` in place to each result that has a twin. Never raises."""
    for item in results:
        try:
            twin = read_url_for(item)
        except Exception:  # noqa: BLE001 - an additive field must never break search
            log.exception("read_url failed for %r", item.get("url"))
            continue
        if twin:
            item["read_url"] = twin
