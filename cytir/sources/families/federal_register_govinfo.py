"""The govinfo bulk leg (`leg=govinfo`) of `us_federal_register_ecfr` (task 702).

ROUTE. govinfo publishes every Federal Register issue 2000-today as XML in one zip per year
(`/bulkdata/FR/{year}/FR-{year}.zip`, entries `MM/FR-YYYY-MM-DD.xml`, ~100-160 MB a year). One GET
per year replaces the ~30,000 per-document text GETs of the API leg. Public domain (17 U.S.C. 105).

LEGALITY. The route is declared (ruling 23.2, POLICY PR #566, merged 2026-10-03). The leg still refuses
to run unless the contract row's `acquisition.route_allowed` carries the token
`www.govinfo.gov/bulkdata/FR` (`ROUTE_TOKEN`): a plugin must not do what its declaration does not say.

IDENTITY (what is and is not the same as the API leg).
  * URL: the API leg's record URL is the document's `html_url` from the federalregister.gov
    listing (it carries a title slug). This leg takes the SAME `html_url`, from the SAME listing
    (month by month, `per_page=1000`: ~36 requests a year, not 30,000), joined on the document
    number the issue XML prints in `[FR Doc. ...]`. It never builds a URL from a title: the
    slugging rule has exceptions the listing already resolved. A document the listing does not
    know is EXCLUDED (`no_api_url`), counted, never given a guessed URL.
  * POINT ID: the legacy mint hashes url + anchor + CHUNK TEXT. GPO's XML flattens differently
    from the API's `raw_text_url` text, so the same document gets DIFFERENT ids from the two
    routes. The routes therefore split the corpus by PUBLICATION DATE (`federal_register.BULK_DATES`,
    2000-01-01 .. 2025-11-30 for this leg; the API leg owns every other date): each leg refuses a
    document dated for the other (`route_date_api` here, `route_date_bulk` there), counted, and
    the split is in the manifest notes (`route_split`). Second line: the drain's overlap guard
    (`family_stage overlap --exclude-served-urls`, which `run_windows` runs on every stage) drops any
    row whose URL is already served; its count is in the stage's OVERLAP.json. Same URL, so it sees it.
  * PROVENANCE: every row's `_master.route` names its route (`govinfo-bulk-xml` here).
  * REPRESENTATION: the family has TWO TEXT SURFACES, split by date: documents of 2000-01 .. 2025-11 are
    the govinfo issue XML flattened to paragraphs (tables as `a | b` lines), documents of 1994-1999 and
    2025-12 on are the API's HTML-derived text. Same URL, same title/agency/date header, different body
    bytes (a measured 13 % of documents have an identical word sequence on both).

UNITS. One year (`fr-govinfo-YYYY`): two sealed captures, the listing (`fr-list-YYYY`, JSON) and
the zip (`fr-zip-YYYY`). Document terminals are counted in the manifest notes (`documents`:
`admitted` + `excluded_*` == `seen`; `reconcile`: XML documents vs listed documents).
Options: `from`, `to` (ISO days; default the bulk date range), `bulk_dates` (`FROM..TO`; the split),
`max_req_s` (listing, default 2), `max_mb_s` (zip read rate, default 4).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import re
import time
import zipfile

from cytir.sources import api_fetch as AF
from cytir.sources.families import federal_register as FR
from cytir.sources.family import Captured, Doc, RunContext, Unit, UnitExcluded

ROUTE_TOKEN = "www.govinfo.gov/bulkdata/FR"
BULK_URL = "https://www.govinfo.gov/bulkdata/FR/{year}/FR-{year}.zip"
HOSTS = ("www.govinfo.gov", FR.HOST)
FIRST_YEAR = 2000            # 1994-1999 are not in the bulk tree (API leg only)
DEFAULT_MAX_REQ_S = 2.0
DEFAULT_MAX_MB_S = 4.0
MIN_BODY_WORDS = FR.MIN_BODY_WORDS

_NUM = re.compile(r"FR\s*Doc\.?\s+([A-Za-z0-9][\w.-]*)")
_CONTAINERS = frozenset({"RULES", "PRORULES", "NOTICES", "PRESDOCS"})
_SKIP = frozenset({"FRDOC", "BILCOD", "PRTPAGE", "GID", "GPH", "MATH", "STARS"})
_INLINE = frozenset({"E", "I", "B", "SU", "SUB", "SUP", "FTREF", "FR", "AC", "PRTPAGE", "FRDOC"})
_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:)])")


class RouteNotDeclared(RuntimeError):
    """The contract does not name the govinfo bulk route; nothing was fetched."""


def require_route_declared(contract: dict | None) -> None:
    allowed = ((contract or {}).get("acquisition") or {}).get("route_allowed") or ""
    if ROUTE_TOKEN not in allowed:
        raise RouteNotDeclared(
            f"us_federal_register_ecfr leg=govinfo: the contract's acquisition.route_allowed does "
            f"not carry {ROUTE_TOKEN!r}. The route becomes legal when the POLICY PR that declares "
            "it is merged; until then the leg does not run.")


def _text(el) -> str:
    """The text of one block, inline markup folded in, tags and page markers dropped."""
    parts = []

    def rec(e):
        if e.tag in _SKIP:
            return
        if e.text:
            parts.append(e.text)
        for c in e:
            rec(c)
            if c.tail:
                parts.append(c.tail)
    rec(el)
    return _SPACE_BEFORE_PUNCT.sub(r"\1", _WS.sub(" ", "".join(parts))).strip()


def _table(el) -> list[str]:
    rows = []
    for row in el.iter("ROW"):
        cells = [_text(c) for c in row if c.tag in ("ENT", "ROW")]
        line = " | ".join(c for c in cells if c)
        if line:
            rows.append(line)
    return rows


def flatten(el) -> list[str]:
    """The paragraphs of one document element, in order. A block is an element whose children
    are all inline markup; anything else is a container and is entered."""
    out: list[str] = []

    def rec(e):
        if e.tag in _SKIP:
            return
        if e.tag == "GPOTABLE":
            out.extend(_table(e))
            return
        kids = [c for c in e if c.tag not in _SKIP]
        if all(c.tag in _INLINE for c in kids):
            t = _text(e)
            if t:
                out.append(t)
            return
        if e.text and e.text.strip():
            out.append(_WS.sub(" ", e.text).strip())
        for c in e:
            rec(c)
            if c.tail and c.tail.strip():
                out.append(_WS.sub(" ", c.tail).strip())
    rec(el)
    return out


def document_number(frdoc) -> str | None:
    m = _NUM.search("".join(frdoc.itertext()))
    return m.group(1).rstrip(".") if m else None


def parse_issue(xml: bytes):
    """Yield `(document_number | None, element_tag, paragraphs)` for every document of one issue
    XML: one per `FRDOC`, the document being the top element under the section container (a
    presidential document is a `PRESDOCU` holding the `PROCLA`/`EXECORD` that holds the `FRDOC`)."""
    from lxml import etree

    root = etree.fromstring(xml, parser=etree.XMLParser(huge_tree=True, recover=False))
    for fr in root.iter("FRDOC"):
        top = fr
        while top.getparent() is not None and top.getparent().tag not in _CONTAINERS:
            top = top.getparent()
        if top.getparent() is None:          # an FRDOC outside any section container
            yield document_number(fr), fr.getparent().tag if fr.getparent() is not None else "?", []
            continue
        yield document_number(fr), top.tag, flatten(top)


def issue_names(zf: zipfile.ZipFile) -> list[str]:
    return sorted(n for n in zf.namelist() if n.lower().endswith(".xml"))


class GovinfoLeg:
    """The `leg=govinfo` behaviour of the family; `UsFederalRegisterEcfrFamily` delegates here."""

    def __init__(self, family) -> None:
        self.family = family

    # ------------------------------------------------------------------ plan / discover
    @staticmethod
    def years(options: dict, today: str) -> list[int]:
        rng = FR.bulk_dates(options)
        if rng is None:
            raise ValueError("leg=govinfo: bulk_dates=none leaves the bulk route no dates")
        first = dt.date.fromisoformat(options.get("from") or rng[0])
        last = dt.date.fromisoformat(options.get("to") or rng[1])
        if first.isoformat() < rng[0] or last.isoformat() > f"{int(rng[1][:4])}-12-31":
            raise ValueError(f"leg=govinfo: {first}..{last} is outside the bulk route's years "
                             f"{rng[0]}..{rng[1]}; the other dates belong to the API route")
        if first.year < FIRST_YEAR:
            raise ValueError(f"leg=govinfo: the bulk tree starts at {FIRST_YEAR}; "
                             "1994-1999 are the API leg's")
        return list(range(first.year, last.year + 1))

    def plan_windows(self, client, options: dict, today: str) -> list[dict]:
        """One window per year, newest first is the driver's order; the count is the API's."""
        pacer = AF.Pacer(float(options.get("max_req_s", DEFAULT_MAX_REQ_S)))
        rng = FR.bulk_dates(options)
        wins = []
        for i, y in enumerate(self.years(options, today)):
            a, b = f"{y}-01-01", f"{y}-12-31"
            # the API counts a long range at its 10,000 query cap: count month by month, inside the
            # bulk route's dates (the year-end of the last year belongs to the API route)
            lo, hi = max(a, rng[0]), min(b, rng[1])
            n = sum(FR.count_documents(client, pacer, x, z)
                    for x, z in AF.month_bins(dt.date.fromisoformat(lo), dt.date.fromisoformat(hi)))
            wins.append({"from": a, "to": b, "docs": n, "key": f"w{i:03d}-{a}_{b}"})
        return wins

    def discover(self, ctx: RunContext):
        require_route_declared(ctx.contract)
        units = [Unit(key=f"fr-govinfo-{y}", url=BULK_URL.format(year=y), consent_hosts=HOSTS,
                      meta={"year": y}) for y in self.years(ctx.options, ctx.today)]
        rng = FR.bulk_dates(ctx.options)
        ctx.notes.update({"leg": "govinfo", "years": len(units), "route": FR.ROUTE_BULK,
                          "route_split": {"bulk_dates": list(rng), "this_route": FR.ROUTE_BULK},
                          "documents": {"seen": 0, "admitted": 0},
                          "reconcile": {"xml_documents": 0, "api_listed": 0, "api_only": 0}})
        return units

    # ------------------------------------------------------------------ acquire
    def acquire(self, unit: Unit, ctx: RunContext):
        year = unit.meta["year"]
        client = ctx.http
        pacer = AF.Pacer(float(ctx.options.get("max_req_s", DEFAULT_MAX_REQ_S)))
        listed: list[dict] = []
        for a, b in AF.month_bins(dt.date(year, 1, 1), dt.date(year, 12, 31)):
            listed.extend(FR.list_documents(client, pacer, a, b))
        listing = json.dumps(listed, sort_keys=True, separators=(",", ":")).encode()
        list_cap = Captured(Unit(key=f"fr-list-{year}", url=FR.LIST_URL, meta={"year": year}),
                            listing, {"status": 200, "source_url": FR.LIST_URL,
                                      "sha256": hashlib.sha256(listing).hexdigest(),
                                      "documents": len(listed)})
        cap_bps = float(ctx.options.get("max_mb_s", DEFAULT_MAX_MB_S)) * 1_000_000
        t0, got, chunks = time.monotonic(), 0, []
        with client.stream("GET", unit.url) as resp:
            if resp.status_code in (404, 410):
                raise UnitExcluded(f"origin_{resp.status_code}")
            resp.raise_for_status()
            for piece in resp.iter_bytes(65536):
                chunks.append(piece)
                got += len(piece)
                ahead = got / cap_bps - (time.monotonic() - t0)
                if ahead > 0:
                    time.sleep(ahead)
            body = b"".join(chunks)
            meta = {"status": resp.status_code, "source_url": unit.url,
                    "content_type": resp.headers.get("content-type"),
                    "sha256": hashlib.sha256(body).hexdigest()}
        return [list_cap, Captured(Unit(key=f"fr-zip-{year}", url=unit.url, meta={"year": year}),
                                   body, meta)]

    # ------------------------------------------------------------------ extract
    @staticmethod
    def _excluded(ctx: RunContext, reason: str) -> None:
        notes = ctx.notes["documents"]
        notes[reason] = notes.get(reason, 0) + 1

    def extract(self, cap: Captured, ctx: RunContext):
        year = cap.unit.meta["year"]
        if not hasattr(ctx, "_govinfo_listing"):
            ctx._govinfo_listing = {}
        if cap.unit.key.startswith("fr-list-"):
            ctx._govinfo_listing[year] = {d["document_number"]: d for d in json.loads(cap.body)}
            ctx.notes["reconcile"]["api_listed"] += len(ctx._govinfo_listing[year])
            return []
        listing = ctx._govinfo_listing.get(year)
        if listing is None:
            raise RuntimeError(f"fr-zip-{year}: its listing capture was not extracted first")
        docs, seen_numbers = [], set()
        with zipfile.ZipFile(io.BytesIO(cap.body)) as zf:
            for name in issue_names(zf):
                for number, tag, paras in parse_issue(zf.read(name)):
                    ctx.notes["documents"]["seen"] += 1
                    ctx.notes["reconcile"]["xml_documents"] += 1
                    d = self._admit(number, paras, listing, seen_numbers, ctx)
                    if d is not None:
                        docs.append(d)
        ctx.notes["reconcile"]["api_only"] += len(set(listing) - seen_numbers)
        if not docs:
            raise UnitExcluded("year_without_documents")
        return docs

    def _admit(self, number, paras, listing, seen_numbers, ctx):
        if not number:
            self._excluded(ctx, "no_document_number")
            return None
        if number in seen_numbers:
            self._excluded(ctx, "duplicate_document_number")
            return None
        seen_numbers.add(number)
        d = listing.get(number)
        if d is None:
            self._excluded(ctx, "no_api_url")
            return None
        if FR.route_for(d["publication_date"], ctx.options) != FR.ROUTE_BULK:
            self._excluded(ctx, "route_date_api")     # this date is the API route's
            return None
        if (why := FR.held_notice_reason(d.get("type"), d.get("title"), d.get("agencies"))):
            self._excluded(ctx, why)
            return None
        body = "\n\n".join(paras)
        if len(body.split()) < MIN_BODY_WORDS:
            self._excluded(ctx, "body_under_word_floor")
            return None
        agencies = ", ".join(a.get("name") or a.get("raw_name") or ""
                             for a in d.get("agencies") or [] if isinstance(a, dict)).strip(", ")
        title = (d.get("title") or number).strip()
        where = (f"Federal Register {d.get('citation') or number} "
                 f"({d.get('type') or 'document'}), published {d['publication_date']}"
                 + (f". Agencies: {agencies}" if agencies else "") + ".")
        ctx.notes["documents"]["admitted"] += 1
        return Doc(url=d["html_url"], text=f"# {title}\n\n{where}\n\n{body}", title=title[:300],
                   lang="en", extra={"body": body})

    ROUTE = FR.ROUTE_BULK

    # the row shape is the API leg's, in one place
    def chunk(self, doc: Doc, ctx: RunContext):
        return FR.FederalRegisterLeg.chunk(self, doc, ctx)
