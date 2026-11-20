"""The cantonal Common-Crawl dialect (`cantonal_cc`) of `scratch_ch_cantonal_law_exact` (tasks 130, 1108, 1174).

GE/NE/JU/SZ/TI have no structured law route, so their official hosts are read from Common Crawl.
A page is the generic web page (`cytir.core.common.process_html`: markdown extraction, junk screen,
the fasttext language gate) and this module owns only what is CANTONAL: the target table, the
common stamps every passage of a canton carries, and the canton stamp on each row. Moved from
the legacy ingest driver, which imports them back (the CLI keeps the CC index resolve and
the WARC range read). `TARGETS` and `stamp_rows` are the driver's own lines.
"""
from __future__ import annotations

# --- per-canton target spec -------------------------------------------------
# resolve_keys : surtkey-prefix entries to resolve (host or registrable domain).
#                NOTE surt folds a leading `www\d*.` into the apex, so `ti.ch`
#                covers www4/www3/www.ti.ch; other subdomains (m3.ti.ch) need
#                their own entry.
# hosts        : allowed netlocs AFTER resolve (guards against prefix folding).
# path_re      : law-path filter (case-insensitive) applied to url path; None = keep all.
# lang         : the forced lang stamp for this canton (task 130).
TARGETS = {
    "GE": {  # Genève — rsGE "SIL" ASP.NET law texts. ge.ch DROPPED: its CC captures
             # are Grand-Conseil *Mémorial* (session debates) + geodata, NOT the
             # systematic law collection. rsGE authoritative law = silgeneve.ch/legis.
        "lang": "fr",
        "resolve_keys": ["silgeneve.ch"],
        "hosts": {"silgeneve.ch", "www.silgeneve.ch"},
        "path_re": r"(?i)/legis",
        "domain": "silgeneve.ch",
    },
    "NE": {  # Neuchâtel — rsn.ne.ch (whole host is the recueil)
        "lang": "fr",
        "resolve_keys": ["rsn.ne.ch"],
        "hosts": {"rsn.ne.ch"},
        "path_re": None,
        "domain": "rsn.ne.ch",
    },
    "JU": {  # Jura — rsju.jura.ch
        "lang": "fr",
        "resolve_keys": ["rsju.jura.ch"],
        "hosts": {"rsju.jura.ch"},
        "path_re": None,
        "domain": "rsju.jura.ch",
    },
    "SZ": {  # Schwyz — sz.ch systematic law collection (PDF-heavy; HTML index/meta only)
        "lang": "de",
        "resolve_keys": ["sz.ch"],
        "hosts": {"sz.ch", "www.sz.ch"},
        "path_re": r"(?i)(gesetz|/recht|erlass|srsz|rechtssammlung|/kanton/gesetze)",
        "domain": "www.sz.ch",
    },
    "TI": {  # Ticino — RLeggi (legge-piatta rendered IT text)
        "lang": "it",
        "resolve_keys": ["ti.ch", "m3.ti.ch", "www3.ti.ch"],
        "hosts": None,
        "path_re": r"(?i)(legge-piatta|raccolta-leggi|rleggi|/can/rl|/CAN/RLeggi)",
        "domain": "www4.ti.ch",
    },
}


def passage_common(spec: dict, last_seen: str) -> dict:
    """The stamps every passage of one canton carries (the driver's `common`)."""
    return dict(domain=spec["domain"], tier="A", source_type="official",
                trust_tier=1.0, last_seen=last_seen)


def stamp_row(row: dict, spec: dict, canton: str) -> dict:
    """The canton stamp on one row (the driver's four lines)."""
    row["lang"] = spec["lang"]          # force canton lang (task 130)
    row["provenance"] = "ch_cantonal_law"
    row["canton"] = canton
    row["country"] = "CH"
    return row


def stamp_rows(passages, spec, canton):
    rows = []
    for p in passages:
        row = p.payload()
        row["text"] = p.text
        row["point_id"] = p.point_id
        rows.append(stamp_row(row, spec, canton))
    return rows
