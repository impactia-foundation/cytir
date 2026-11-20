"""`govuk_content` -- GOV.UK content items from the Content API (task 699; contract row of task 695).

Route (the contract's `acquisition.route_allowed`): `https://www.gov.uk/api/search.json` to
ENUMERATE and `https://www.gov.uk/api/content/{path}` to read, on www.gov.uk only (the
content-store host is HELD and never contacted), one connection, <= 2 req/s (the default here;
`max_req_s`), honest UA. No HTML page is fetched. OGL v3.0.

Units: one calendar month of `public_timestamp` (`govuk-YYYY-MM`; `before-2001` is one unit):
the month's listing (`count=1000` pages) and then one content item per link. A month whose
listing is shorter than the API's `total` refuses. Items are composed from the JSON
(`title`, `description`, `details.body`, `details.parts[]` ...), the HTML of the body rendered to
markdown; English only (a `locale` other than `en` is excluded, Welsh is outside the ratified
scope); redirects/gone/vanish items hold no content and are excluded with their schema name;
a page of fewer than `MIN_BODY_WORDS` words is excluded at extract (measured 2026-10-03: 25 % of a
100-item sample). Document terminals are counted in the manifest notes (`documents`).
Options: `from`, `to` (ISO days), `max_req_s`.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re

from cytir.sources import api_fetch as AF
from cytir.sources.family import CHUNK_DECLARED, HttpSpec, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.families.web import WebPageMixin
from cytir.sources.registry import register_family

LICENSE = "OGL-UK-3.0"
HOST = "www.gov.uk"
SEARCH_URL = "https://www.gov.uk/api/search.json"
CONTENT_URL = "https://www.gov.uk/api/content"
DEFAULT_MAX_REQ_S = 2.0
MIN_BODY_WORDS = 30
PAGE = 1000
FIRST_MONTH = dt.date(2001, 1, 1)
# Document types whose `details.body` is only "Read the full decision in <title>" links to PDF
# attachments (measured 2026-10-03, 40 items across 2018-2026 of the 134,506 employment tribunal
# decisions: body 14-60 words, all link lines; the multi-attachment ones pass the 30-word floor by
# 1-30 words). The decision text sits in the PDF (and, for part of the items, in
# `details.metadata.hidden_indexable_content`, 14 to 25,000 words) and names private claimants. They are
# held at extract, counted as `link_only_<type>`. A useful version needs the attachment text under a
# declared rule for personal data in decisions (Tim's ruling), not this body.
LINK_ONLY_DOCUMENT_TYPES = frozenset({"employment_tribunal_decision"})
# Document types whose items name PRIVATE persons (Tim's ruling 24, 2026-10-03: sources with personal
# data stay held until we can anonymise; tracked for reconsideration). Held at extract, counted as
# `held_personal_data_<type>`; a held item is never chunked, so no row, snippet or embedding exists.
# Totals on the Content API 2026-10-03 and what names people:
#   traffic_commissioner_regulatory_decision 222 (transport managers, sole traders, drivers)
#   utaac_decision 2,085 (benefit claimants, health data)      tax_tribunal_decision 1,445 (individual taxpayers)
#   employment_appeal_tribunal_decision 2,627 (claimants in titles)   sfo_case 29 (charged individuals)
#   residential_property_tribunal_decision 18,284 (private addresses)  asylum_support_decision 101
#   decision 13,040 (the WHOLE type: teacher-misconduct outcomes with dates of birth, union-rule and
#   named-person hearing decisions; a subset may be released later by a ruling)
#   fatality_notice 512 (named deceased personnel; held in doubt)  foi_release 12,193 (responses can name
#   officials and requesters; held in doubt)
# The positive list below is the gate; this set documents why each held type is held. Add a type
# here the day it is found to name a private person.
HELD_PERSONAL_DATA_TYPES = frozenset({
    "traffic_commissioner_regulatory_decision", "utaac_decision", "tax_tribunal_decision",
    "employment_appeal_tribunal_decision", "sfo_case", "residential_property_tribunal_decision",
    "asylum_support_decision", "decision", "fatality_notice", "foi_release"})
# POSITIVE LIST (orchestrator D2, 2026-10-03): only a document type somebody read is admitted. The
# 53 types below are every type the Content API facet counted at >= 500 items on 2026-10-03 (712,559
# items, 137 types) that is not held above: they hold 517,914 of the 527,515 items of the 126 non-held
# types (the other 73 types, 1.8 %, are the tail; `guide`, `answer`, `regulation` are in).
# An unknown or new type is held by default and counted
# `held_unlisted_type_<type>`, so a new decision-like type can never slip in. `HELD_*` and
# `LINK_ONLY_*` stay as the record of WHY each held type is held. Adding a type = a hand read of
# 10 items for named private persons, then a PR.
ADMITTED_DOCUMENT_TYPES = frozenset({
    "hmrc_manual_section", "news_story", "press_release", "research_for_development_output",
    "guidance", "official_statistics", "transparency", "national_statistics", "research",
    "corporate_report", "world_news_story", "armed_forces_covenant_business", "policy_paper",
    "speech", "aaib_report", "detailed_guide", "correspondence", "document_collection",
    "consultation_outcome", "notice", "person", "protected_food_drink_name",
    "official_statistics_announcement", "form", "independent_report",
    "product_safety_alert_report_recall", "national_statistics_announcement", "manual_section",
    "cma_case", "case_study", "written_statement", "export_health_certificate", "impact_assessment",
    "international_treaty", "medical_safety_alert", "esi_fund", "statutory_guidance",
    "organisation", "statistical_data_set", "about", "closed_consultation", "maib_report",
    "ministerial_role", "promotional", "oral_statement", "working_group", "drug_safety_update",
    "guide", "regulation", "answer", "government_response", "raib_report", "authored_article"})
NO_CONTENT = {"redirect", "gone", "vanish", "special_route", "placeholder", "coming_soon"}
_HEADING = {f"h{i}": i for i in range(1, 7)}
_BLOCK = set(_HEADING) | {"p", "div", "ul", "ol", "li", "table", "tr", "blockquote", "pre",
                          "section", "article", "figure", "details", "dl", "dt", "dd", "tbody",
                          "thead", "figcaption", "header", "footer", "nav", "aside"}
_WS = re.compile(r"\s+")


def _text(el) -> str:
    return _WS.sub(" ", "".join(el.itertext())).strip()


def html_to_markdown(fragment: str) -> str:
    """The text of a GOV.UK body: headings as `#`, list items as `- `, table rows as `a | b`,
    one block per paragraph; scripts, styles and images dropped, link targets dropped."""
    import lxml.html

    if not (fragment or "").strip():
        return ""
    root = lxml.html.fragment_fromstring(fragment, create_parent="div")
    for bad in root.xpath("//script|//style|//img|//svg"):
        bad.drop_tree()
    out: list[str] = []

    def walk(el) -> None:
        tag = el.tag if isinstance(el.tag, str) else ""
        if tag in _HEADING:
            t = _text(el)
            if t:
                out.append("#" * min(_HEADING[tag] + 1, 6) + " " + t)
            return
        if tag == "tr":
            cells = [_text(c) for c in el if c.tag in ("td", "th")]
            if any(cells):
                out.append(" | ".join(cells))
            return
        if tag == "li":
            kids = [c for c in el if c.tag in ("ul", "ol")]
            own = _WS.sub(" ", (el.text or "") + "".join(
                _text(c) + (c.tail or "") for c in el if c not in kids)).strip()
            if own:
                out.append("- " + own)
            for k in kids:
                walk(k)
            return
        kids = [c for c in el if isinstance(c.tag, str)]
        if tag in _BLOCK and not any((c.tag in _BLOCK) for c in el.iterdescendants()
                                     if isinstance(c.tag, str)):
            t = _text(el)
            if t:
                out.append(t)
            return
        if (el.text or "").strip():
            out.append(_WS.sub(" ", el.text).strip())
        for c in kids:
            walk(c)
            if (c.tail or "").strip():
                out.append(_WS.sub(" ", c.tail).strip())

    walk(root)
    return "\n\n".join(out)


def _html_value(v) -> str:
    """A `details` value that holds HTML: a string, or a list of `{content_type, content}`."""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        for item in v:
            if isinstance(item, dict) and item.get("content_type", "text/html") == "text/html":
                return str(item.get("content") or "")
    return ""


def compose_body(details: dict) -> str:
    """Markdown of the readable fields of one content item (body, parts, introduction ...)."""
    blocks = []
    for key in ("introduction", "body", "more_information", "summary"):
        md = html_to_markdown(_html_value(details.get(key)))
        if md:
            blocks.append(md)
    for part in details.get("parts") or []:
        if not isinstance(part, dict):
            continue
        md = html_to_markdown(_html_value(part.get("body")))
        if md:
            blocks.append(f"## {part.get('title') or part.get('slug') or ''}".rstrip() + "\n\n" + md)
    return "\n\n".join(blocks)


def _search(client, pacer: AF.Pacer, gte: str | None, lte: str | None, *, start: int, count: int):
    rng = (f"from:{gte}," if gte else "") + (f"to:{lte}" if lte else "")
    params = {"count": count, "start": start, "order": "public_timestamp",
              "filter_public_timestamp": rng.rstrip(","), "fields": ["link"]}
    resp = AF.get(client, SEARCH_URL, pacer, params=params)
    resp.raise_for_status()
    return resp.json()


def list_links(client, pacer: AF.Pacer, gte: str | None, lte: str | None) -> list[str]:
    first = _search(client, pacer, gte, lte, start=0, count=PAGE)
    total = int(first.get("total", 0))
    out = [r["link"] for r in first.get("results", [])]
    while len(out) < total:
        page = _search(client, pacer, gte, lte, start=len(out), count=PAGE)
        got = [r["link"] for r in page.get("results", [])]
        if not got:
            break
        out.extend(got)
    if len(out) != total:
        raise RuntimeError(f"gov.uk {gte}..{lte}: the listing holds {len(out)} links, the API "
                           f"said {total}; never take a truncated month")
    return out


def count_range(client, pacer: AF.Pacer, gte: str | None, lte: str | None) -> int:
    return int(_search(client, pacer, gte, lte, start=0, count=1).get("total", 0))


@register_family
class GovukContentFamily(WebPageMixin):   # preflight: the declared chunk policy must be selected
    family_id = "govuk_content"
    route = "official_api"
    direct_crawl = True
    chunk_policy_id = CHUNK_DECLARED
    http_spec = HttpSpec(timeout=120, follow_redirects=False)   # task 1556: the runner builds and closes the client
    require_records = True

    def _pacer(self, ctx: RunContext) -> AF.Pacer:
        if not hasattr(ctx, "_govuk_pacer"):
            ctx._govuk_pacer = AF.Pacer(float(ctx.options.get("max_req_s", DEFAULT_MAX_REQ_S)))
        return ctx._govuk_pacer

    def _bins(self, ctx: RunContext):
        """`(key, gte, lte)`: the one `before-2001` bin (when `from` reaches it), then calendar
        months up to `to`. A bin is never split and never overlaps its neighbour."""
        last = dt.date.fromisoformat(ctx.options.get("to") or ctx.today)
        frm = dt.date.fromisoformat(ctx.options.get("from") or "1900-01-01")
        bins = []
        if frm < FIRST_MONTH:
            bins.append(("before-2001", None, min(last, FIRST_MONTH - dt.timedelta(days=1)).isoformat()))
        first = max(frm, FIRST_MONTH)
        if first <= last:
            bins += [(f"govuk-{a[:7]}", a, b) for a, b in AF.month_bins(first, last)]
        return bins

    def plan_windows(self, ctx: RunContext, *, cap: int = 10_000) -> list[dict]:
        """Date windows of <= `cap` listed items (`cytir.sources.run_windows`)."""
        client, pacer = ctx.http, self._pacer(ctx)
        counted = [(a or "1900-01-01", b, count_range(client, pacer, a, b))
                   for _k, a, b in self._bins(ctx)]
        return AF.pack_windows(counted, cap)

    def discover(self, ctx: RunContext):
        units = [Unit(key=key, url=SEARCH_URL, consent_hosts=(HOST,), meta={"gte": a, "lte": b})
                 for key, a, b in self._bins(ctx)]
        if not units:
            raise ValueError("govuk_content: no month inside from/to")
        ctx.notes.update({"months": len(units), "documents": {"seen": 0, "admitted": 0}})
        return units

    def acquire(self, unit: Unit, ctx: RunContext):
        client, pacer = ctx.http, self._pacer(ctx)
        links = list_links(client, pacer, unit.meta["gte"], unit.meta["lte"])
        caps = []
        notes = ctx.notes["documents"]
        for link in links:
            notes["seen"] += 1
            if not link.startswith("/"):
                notes["link_not_on_gov_uk"] = notes.get("link_not_on_gov_uk", 0) + 1
                continue
            resp = AF.get(client, CONTENT_URL + link, pacer)
            if resp.status_code != 200:
                key = f"content_{resp.status_code}"
                notes[key] = notes.get(key, 0) + 1
                continue
            caps.append(Captured(Unit(key=f"govuk:{link}", url=CONTENT_URL + link),
                                 resp.content, {"status": 200, "source_url": CONTENT_URL + link,
                                                "sha256": hashlib.sha256(resp.content).hexdigest()}))
        if not caps:
            raise UnitExcluded("month_without_documents")
        return caps

    @staticmethod
    def _excluded(ctx: RunContext, reason: str) -> None:
        notes = ctx.notes["documents"]
        notes[reason] = notes.get(reason, 0) + 1

    def extract(self, cap: Captured, ctx: RunContext):
        import json

        item = json.loads(cap.body)
        schema = item.get("schema_name") or ""
        if schema in NO_CONTENT or not item.get("details"):
            self._excluded(ctx, f"schema_{schema or 'empty'}")
            return []
        dtype = item.get("document_type") or ""
        if dtype in LINK_ONLY_DOCUMENT_TYPES:
            self._excluded(ctx, f"link_only_{dtype}")
            return []
        if dtype in HELD_PERSONAL_DATA_TYPES:
            self._excluded(ctx, f"held_personal_data_{dtype}")
            return []
        if dtype not in ADMITTED_DOCUMENT_TYPES:
            self._excluded(ctx, f"held_unlisted_type_{dtype or 'none'}")
            return []
        if (item.get("locale") or "en") != "en":
            self._excluded(ctx, "locale_not_en")
            return []
        body = compose_body(item["details"])
        if len(body.split()) < MIN_BODY_WORDS:
            self._excluded(ctx, "body_under_word_floor")
            return []
        title = (item.get("title") or item.get("base_path") or "").strip()
        desc = (item.get("description") or "").strip()
        upd = (item.get("public_updated_at") or "")[:10]
        head = f"# {title}\n\n" + (f"{desc}\n\n" if desc else "")
        tail = f"\n\nGOV.UK{', last updated ' + upd if upd else ''}. Open Government Licence v3.0."
        ctx.notes["documents"]["admitted"] += 1
        return [Doc(url="https://www.gov.uk" + item["base_path"], text=head + body + tail,
                    title=title[:300], lang="en", extra={"body": body})]

    def chunk(self, doc: Doc, ctx: RunContext):
        from cytir.core import common as C
        from cytir.chunk import dispatch as _dispatch

        rows = []
        for unit in _dispatch.chunk_markdown_channels(doc.text):
            p = C.passage_from_chunk(
                unit, url=doc.url, title=doc.title,
                snippet=C.extractive_snippet(doc.extra["body"], 25), lang="en",
                domain="www.gov.uk", tier="A", source_type="official", trust_tier=1.0,
                last_seen=ctx.today, license=LICENSE)
            rows.append(C.passage_source_record(p))
        return rows
