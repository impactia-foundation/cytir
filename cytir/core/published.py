"""cytir.core.published -- the ONE definition of the `published` payload value.

Contract (task 676). A row's `published` is an ISO-8601 calendar date in UTC:

    YYYY-MM-DD        the normal form; every date-precision source lands here
    YYYY-MM, YYYY     reduced precision, kept as is: the source knows no more
                      (bibliographic records carry a bare year -- 4.97M rows on
                      the twin -- and inventing a day would be a guess)

Everything else that names a day is rewritten to `YYYY-MM-DD` by the WRITER, so
no producer needs its own converter and no consumer needs a second parser:

    2026-09-16T23:30:00-05:00   ISO datetime      -> 2026-09-17 (converted to UTC)
    2026-07-13T04:09:22.000Z    ISO datetime      -> 2026-07-13
    2026-07-13 04:09:22         naive datetime    -> 2026-07-13 (date part as written)
    Wed, 16 Sep 2026 12:00:00 GMT   RFC-822/2822  -> 2026-09-16 (UTC)
    17.03.2015                  dd.mm.yyyy (CH/DE)-> 2015-03-17
    9. März 2026                D. Monat YYYY (de)-> 2026-03-09
    1er mars 2026 / 5 mai 2026  D mois YYYY (fr)  -> 2026-03-01 / 2026-05-05
    21 aprile 2026              D mese YYYY (it)  -> 2026-04-21

A value that names no day we can read is returned as None. The writer then keeps
the original text untouched (a value is never dropped by normalisation) and the
census counts it as `other`.

Stdlib only, no clock: a date in the future is not judged here (the extractors
clamp against their own `now_ref`).
"""
from __future__ import annotations

import datetime as _dt
import email.utils
import re
from typing import Any, Mapping

_YEAR = re.compile(r"^(\d{4})$")
_YEAR_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DATETIME = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(?:[.,]\d+)?)?"
    r"\s*(Z|[+-]\d{2}:?\d{2})?$")
_RFC822 = re.compile(r"^(?:[A-Za-z]{3},\s*)?\d{1,2}\s+[A-Za-z]{3}\s+\d{4}")
_DMY = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")

# Long dates in the news publishers' own languages (task 676, orchestrator
# 2026-09-30: 91 German rows such as `9. März 2026` sat on the twin as class
# `other`). The day may carry a period (de) or `er` (fr `1er`); the month name is
# case-insensitive. Each name maps to ONE month number across the three languages,
# so a single table cannot misread one language as another.
_LONG_DATE = re.compile(r"^(\d{1,2})(?:\.|er)?\s+([^\W\d_]+)\.?\s+(\d{4})$")
_MONTHS = {}
for _n, _names in enumerate((
        ("januar", "janvier", "gennaio"),
        ("februar", "février", "fevrier", "febbraio"),
        ("märz", "maerz", "mars", "marzo"),
        ("april", "avril", "aprile"),
        ("mai", "maggio"),
        ("juni", "juin", "giugno"),
        ("juli", "juillet", "luglio"),
        ("august", "août", "aout", "agosto"),
        ("september", "septembre", "settembre"),
        ("oktober", "octobre", "ottobre"),
        ("november", "novembre"),
        ("dezember", "décembre", "decembre", "dicembre")), start=1):
    for _name in _names:
        _MONTHS[_name] = _n

MIN_YEAR, MAX_YEAR = 1000, 2100


def _date(y: int, m: int, d: int) -> _dt.date | None:
    if not (MIN_YEAR <= y <= MAX_YEAR):
        return None
    try:
        return _dt.date(y, m, d)
    except ValueError:
        return None


def normalize_published(raw: Any) -> str | None:
    """The contract form of `raw`, or None when `raw` names no readable day/month/year."""
    if raw is None or isinstance(raw, bool):
        return None
    s = str(raw).strip()
    if not s:
        return None
    m = _YEAR.match(s)
    if m:
        return s if MIN_YEAR <= int(s) <= MAX_YEAR else None
    m = _YEAR_MONTH.match(s)
    if m:
        y, mo = int(m[1]), int(m[2])
        return s if (MIN_YEAR <= y <= MAX_YEAR and 1 <= mo <= 12) else None
    m = _DATE.match(s)
    if m:
        d = _date(int(m[1]), int(m[2]), int(m[3]))
        return d.isoformat() if d else None
    m = _DATETIME.match(s)
    if m:
        d = _date(int(m[1]), int(m[2]), int(m[3]))
        if d is None:
            return None
        hour, minute = int(m[4]), int(m[5])
        if hour > 23 or minute > 59:
            return None
        tz = m[7]
        if tz and tz != "Z":
            sign = 1 if tz[0] == "+" else -1
            digits = tz[1:].replace(":", "")
            offset = _dt.timedelta(hours=int(digits[:2]), minutes=int(digits[2:]))
            local = _dt.datetime.combine(d, _dt.time(hour, minute))
            return (local - sign * offset).date().isoformat()
        return d.isoformat()
    m = _LONG_DATE.match(s)
    if m and m[2].casefold() in _MONTHS:
        d = _date(int(m[3]), _MONTHS[m[2].casefold()], int(m[1]))
        return d.isoformat() if d else None
    if _RFC822.match(s):
        try:
            parsed = email.utils.parsedate_to_datetime(s)
        except (TypeError, ValueError):
            return None
        if parsed is None:
            return None
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(_dt.timezone.utc)
        d = _date(parsed.year, parsed.month, parsed.day)
        return d.isoformat() if d else None
    m = _DMY.match(s)
    if m:
        d = _date(int(m[3]), int(m[2]), int(m[1]))
        return d.isoformat() if d else None
    return None


def classify_published(raw: Any) -> str:
    """The format CLASS of a stored value (census vocabulary)."""
    if raw is None or str(raw).strip() == "":
        return "missing"
    s = str(raw).strip()
    if _DATE.match(s):
        return "iso_date"
    if _YEAR_MONTH.match(s):
        return "iso_year_month"
    if _YEAR.match(s):
        return "iso_year"
    if _DATETIME.match(s):
        return "iso_datetime"
    m = _LONG_DATE.match(s)
    if m and m[2].casefold() in _MONTHS:
        return "long_date"
    if _RFC822.match(s):
        return "rfc822"
    if _DMY.match(s):
        return "dotted_dmy"
    return "other"


def normalize_payload_published(payload: Mapping[str, Any]) -> dict:
    """A copy of `payload` whose `published`, if present and readable, is in
    contract form. An unreadable value is left exactly as it was."""
    out = dict(payload)
    if "published" in out:
        got = normalize_published(out["published"])
        if got is not None:
            out["published"] = got
    return out
