"""URL canonical form shared by the technical-manufacturer seeds, catalog and product-page stages
(moved from the flat seeds script by task 1401; one definition)."""
from __future__ import annotations

import re
import urllib.parse


def manufacturer_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value.strip())
    host = (parsed.hostname or "").lower()
    if not host or parsed.scheme not in {"http", "https"}:
        return ""
    if host.startswith("www."):
        host = host[4:]
    port = f":{parsed.port}" if parsed.port else ""
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query = [(key, val) for key, val in query
             if not key.lower().startswith("utm_") and key.lower() not in
             {"gclid", "fbclid", "srsltid", "ref", "source"}]
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    return urllib.parse.urlunsplit(("https", host + port, path,
                                   urllib.parse.urlencode(query), ""))
