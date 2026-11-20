"""`us_federal_register_ecfr` -- the eCFR leg: one record per live CFR section (task 698).

The declared family of task 695 (POLICY PR #499): official bulk route, 17 U.S.C. 105 (public
domain), `chunk.generic`. This adapter builds the **eCFR leg only**. The Federal Register leg
(1.0M documents from the federalregister.gov API) is NOT built: `--opt leg=` accepts only
`ecfr`, and any other value refuses (a family must not silently do half of what it declares).

Route (the contract's `acquisition.route_allowed`): the title list comes from the eCFR
versioner API (`/api/versioner/v1/titles.json`), the section text from govinfo's bulk ECFR XML
(`/bulkdata/ECFR/title-N/ECFR-titleN.xml`). The forbidden eCFR paths (`/api/versioner/v1/full/`,
`/api/renderer/v1/content/`, any HTML page) are never called. One request per title, serial.

Units: one **title** (50 live ones; title 35 is reserved and ends `excluded(title_reserved)`).
The sealed L0 is the title's XML, exactly as govinfo served it. Sections are the unit of the
CONTRACT (`source_unit`), so the per-section terminals are counted in the manifest notes
(`sections`: `admitted` + `excluded_*` = `seen`), and the run refuses when they do not balance.

A section record states that the eCFR is a current but UNOFFICIAL consolidation and names the
official text (the annual CFR edition on govinfo). The canonical URL is the eCFR's own section
page (`https://www.ecfr.gov/current/title-N/section-X`), the place an agent goes to read; it is
only a pointer, never fetched. Options: `leg` (default `ecfr`), `titles` (comma list, a
smoke/debug subset), `max_mb_s` (read rate cap, default 4).
"""
from __future__ import annotations

import hashlib
import re
import time

from cytir.sources.families.web import WebPageMixin
from cytir.sources.family import CHUNK_DECLARED, HttpSpec, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

LICENSE = "public-domain"      # 17 U.S.C. 105; the instrument is in the contract's licence block
TITLES_URL = "https://www.ecfr.gov/api/versioner/v1/titles.json"
BULK_URL = "https://www.govinfo.gov/bulkdata/ECFR/title-{n}/ECFR-title{n}.xml"
SECTION_URL = "https://www.ecfr.gov/current/title-{n}/section-{sec}"
CONSENT_HOSTS = ("www.ecfr.gov", "www.govinfo.gov")
MIN_BODY_CHARS = 20
DEFAULT_MAX_MB_S = 4.0         # option `max_mb_s`; CC 5 + consent 2 + news 1 share the 8 MB/s cap
_WS = re.compile(r"\s+")
_RESERVED = re.compile(r"\[\s*reserved\s*\]|\[\s*removed\s*\]", re.I)
# the section identifier is the whole token after an optional "§": "1.1(a)-1" (title 26), "1.000" (title 48,
# no "§"), "86.1860-17". A token that does not start with a digit ("Section 03", an appendix) is not a section
# number and ends excluded, never guessed.
_SEC_NUM = re.compile(r"^\s*(?:§+\s*)?(\d\S*)")

NOTE = ("Source: eCFR, a current but UNOFFICIAL consolidation of the Code of Federal "
        "Regulations (up to date as of {asof}). The official text is the annual CFR edition "
        "on govinfo.gov.")


def _text(el) -> str:
    return _WS.sub(" ", "".join(el.itertext())).strip()


def parse_title(body: bytes, *, number: int, name: str, asof: str):
    """Yield `(kind, payload)` for every section of one title XML.

    `("section", dict)` for a kept section, `("excluded", reason)` for a reserved/empty one.
    """
    from lxml import etree

    root = etree.fromstring(body, parser=etree.XMLParser(huge_tree=True, recover=False))
    seen: set[str] = set()
    for sec in root.iter("DIV8"):
        if sec.get("TYPE") != "SECTION":
            continue
        head_el = sec.find("HEAD")
        head = _text(head_el) if head_el is not None else ""
        if _RESERVED.search(head):
            yield "excluded", "section_reserved"
            continue
        m = _SEC_NUM.match(sec.get("N") or "") or _SEC_NUM.match(head)
        if not m:
            yield "excluded", "no_section_number"
            continue
        key = m.group(1).rstrip(".,;:")
        if key in seen:             # a section number is unique per title; a repeat is excluded, counted
            yield "excluded", "duplicate_section_number"
            continue
        paras = [_text(p) for p in sec if p.tag not in ("HEAD", "SECAUTH", "CITA", "EDNOTE",
                                                         "SOURCE", "AUTH") and _text(p)]
        body_txt = "\n\n".join(paras)
        if len(body_txt) < MIN_BODY_CHARS:
            yield "excluded", "section_empty"
            continue
        path = []
        for anc in sec.iterancestors():
            if anc.tag.startswith("DIV") and anc.get("TYPE") in ("PART", "SUBPART", "SUBCHAP",
                                                                  "CHAPTER", "SUBJGRP"):
                h = anc.find("HEAD")
                if h is not None and _text(h):
                    path.append(_text(h))
        seen.add(key)
        yield "section", {"number": number, "name": name, "asof": asof, "sec": key,
                          "head": head, "path": list(reversed(path)), "body": body_txt}


@register_family
class UsFederalRegisterEcfrFamily(WebPageMixin):
    family_id = "us_federal_register_ecfr"
    route = "official_api"
    direct_crawl = True            # we contact ecfr.gov / govinfo.gov ourselves
    chunk_policy_id = CHUNK_DECLARED
    http_spec = HttpSpec(timeout=300)   # task 1556: the runner builds and closes the client
    require_records = True         # a run that admits 0 sections refuses (smoke-test-yield)

    def _fr(self, ctx: RunContext):
        """The windowed-API / bulk leg named by `--opt leg=` (`fr`, `govinfo`), or None for eCFR."""
        leg = ctx.options.get("leg", "ecfr")
        if leg == "govinfo":
            from cytir.sources.families.federal_register_govinfo import GovinfoLeg
            return GovinfoLeg(self)
        from cytir.sources.families.federal_register import FederalRegisterLeg
        return FederalRegisterLeg(self) if leg == "fr" else None

    def plan_windows(self, ctx: RunContext, *, cap: int = 25_000) -> list[dict]:
        """The leg's date windows (`cytir.sources.run_windows`): the Federal Register leg only."""
        from cytir.sources.families import federal_register as FR
        from cytir.sources import api_fetch as AF
        if ctx.options.get("leg") == "govinfo":
            from cytir.sources.families.federal_register_govinfo import GovinfoLeg, require_route_declared
            require_route_declared(ctx.contract)
            return GovinfoLeg(self).plan_windows(ctx.http, ctx.options, ctx.today)
        if ctx.options.get("leg") != "fr":
            raise ValueError("us_federal_register_ecfr: only leg=fr and leg=govinfo are windowed")
        pacer = AF.Pacer(float(ctx.options.get("max_req_s", FR.DEFAULT_MAX_REQ_S)))
        return FR.plan_windows(ctx.http, pacer, ctx.options, ctx.today, cap=cap)

    def discover(self, ctx: RunContext):
        leg = ctx.options.get("leg", "ecfr")
        if leg in ("fr", "govinfo"):
            return self._fr(ctx).discover(ctx)
        if leg != "ecfr":
            raise ValueError(f"us_federal_register_ecfr: leg={leg!r} is not built; "
                             "legs are `ecfr` (default), `fr` and `govinfo`")
        resp = ctx.http.get(TITLES_URL)
        resp.raise_for_status()
        titles = resp.json().get("titles") or []
        if not titles:
            raise RuntimeError("us_federal_register_ecfr: the eCFR titles list is empty; a "
                               "read error is not an empty source")
        want = {int(x) for x in str(ctx.options.get("titles", "")).split(",") if x.strip()}
        units = []
        for t in sorted(titles, key=lambda t: int(t["number"])):
            n = int(t["number"])
            if want and n not in want:
                continue
            units.append(Unit(
                key=f"title-{n}", url=BULK_URL.format(n=n), consent_hosts=CONSENT_HOSTS,
                meta={"number": n, "name": t.get("name", ""), "reserved": bool(t.get("reserved")),
                      "asof": t.get("up_to_date_as_of") or ctx.today}))
        ctx.notes.update({"leg": leg, "titles": len(units),
                          "sections": {"seen": 0, "admitted": 0}})
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        if (leg := self._fr(ctx)) is not None:
            return leg.acquire(unit, ctx)
        if unit.meta["reserved"]:
            raise UnitExcluded("title_reserved")
        # one title = one serial GET, read at most `max_mb_s` (ZyWALL cap 8 MB/s shared; plan
        # ruling 5): the sleep keeps the average rate under the cap whatever the link does.
        cap_bps = float(ctx.options.get("max_mb_s", DEFAULT_MAX_MB_S)) * 1_000_000
        t0, got, chunks = time.monotonic(), 0, []
        with ctx.http.stream("GET", unit.url) as resp:
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
        return [Captured(unit, body, meta)]

    def extract(self, cap: Captured, ctx: RunContext):
        if (leg := self._fr(ctx)) is not None:
            return leg.extract(cap, ctx)
        m = cap.unit.meta
        docs = []
        sec = ctx.notes["sections"]
        for kind, p in parse_title(cap.body, number=m["number"], name=m["name"], asof=m["asof"]):
            sec["seen"] += 1
            if kind == "excluded":
                sec[p] = sec.get(p, 0) + 1
                continue
            sec["admitted"] += 1
            url = SECTION_URL.format(n=p["number"], sec=p["sec"])
            where = f"Title {p['number']}, {p['name']}" + "".join(f"; {h}" for h in p["path"])
            md = (f"# eCFR (unofficial) {p['number']} CFR {p['head']}\n\n{where}\n\n"
                  f"{p['body']}\n\n{NOTE.format(asof=p['asof'])}")
            docs.append(Doc(url=url, text=md, title=f"{p['number']} CFR {p['head']}"[:300],
                            lang="en", extra={"asof": p["asof"], "body": p["body"]}))
        if not docs:
            raise UnitExcluded("title_without_sections")
        return docs

    def chunk(self, doc: Doc, ctx: RunContext):
        if (leg := self._fr(ctx)) is not None:
            return leg.chunk(doc, ctx)
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch

        rows = []
        for unit in _dispatch.chunk_markdown_channels(doc.text):
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(doc.extra["body"], 25), lang="en",
                domain="ecfr.gov", tier="A", source_type="official", trust_tier=1.0,
                last_seen=doc.extra["asof"], license=LICENSE)
            rows.append(C.passage_source_record(p))
        return rows
