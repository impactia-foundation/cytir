"""`scratch_ctgov_psy_exact` -- the ClinicalTrials.gov / EU CTIS / regulator-page psychedelic slice (task 1407).

The frozen one-shot build of task 177 (round-5 ingest #3): three kinds of source, one wave, provenance
`ctgov_psy`. The code is `cytir.sources.ctgov_psy.harvest` (MOVED from the flat driver script; its record
builders are pure and the adapter calls the very same functions, so there is one copy):

  * unit `ctg-<slice>`  : one ClinicalTrials.gov API v2 query slice (psilocybin, MDMA, ... , two curated
    psychiatry slices). `acquire` pages the API (<= 1 req/s) and returns one Captured per page.
    `extract`: one Doc per study (one study = one record, tier B, `registry_record`), URL deduplicated
    across the whole run.
  * unit `ctis-<term>`  : one EU CTIS search term (ketamine terms kept only when psychiatric).
  * unit `html-NN`      : one curated regulator page (FDA, CH BAG, Swissmedic; tier A, `official`):
    robots.txt is read first, a disallow or a non-200 answer ends the unit `excluded` (we do not fight a
    WAF); a transport error or a 5xx is an instrument failure and quarantines the unit (the old driver
    counted it as a skip); the page text goes through the shared `process_html`, so the extraction-junk, language-scope
    and listing-shell gates apply.

CHUNK: `dispatcher` (`CHUNK_DISPATCHER`). The regulator passages are cut by the shared `process_html`, i.e.
the central dispatcher's pinned selection, and carry its `stamps`. The registry records (CTG studies, CTIS
trials) are whole-record rows that no chunker touches; the Protocol declares ONE policy per family (per
contract row, `chunk_policy_for`), not per unit, so the label names the only unit kind that has a chunker.
The contract row declares `unknown`, so no registry policy is named here.

OUTLINKS: `extract` of a regulator page arms the shared outlink sink (`_html_extract.enable_outlink_capture`, the call the old CLI
made once in its `main`) on `<out_dir>/outlinks.jsonl` around `html_records` and disarms it after, so the external links
of every fetched regulator page stream to the same file the old CLI wrote (append mode; the process-wide sink is never
left open).

STAYS PARTIAL: `l1` (no existing code files this family's dump / API pages as L1 inputs: the old CLI wrote only
`state.json` and the pack; a new L1 filer would be new code, not a port). In-unit resume (the old `state.json`
checkpointed per query slice, which the unit ledger now does) is the framework's.

CONSENT AND LICENCE are read from the contract row, never written here; the row is `unknown` for both, and
`discover` copies the row's values into the run notes (so the manifest shows what the run was admitted
under) without refusing: refusing a host-contacting `acquire` on an undeclared row is the framework's one
rule (task 1406), not this module's.

Options (`--opt KEY=VALUE`): `skip_ctis` (any value: leave out the EU CTIS units, the old `--skip-ctis`).
`ctx.http`, when set, is a client with `get(url, params=None)` and `post_json(url, body)` (the old
`CurlClient` shape); the default is that curl client, because the CTG host blocks python TLS clients.
"""
from __future__ import annotations

import json
import time
import urllib.parse
from urllib import robotparser

from cytir.core import common as C
from cytir.extract import html_extract as _html_extract
from cytir.sources.ctgov_psy import harvest as H
from cytir.sources.family import CHUNK_DISPATCHER, HttpSpec, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

CTG_HOST = "clinicaltrials.gov"
CTIS_HOST = "euclinicaltrials.eu"
#: politeness pauses; a test sets this to a no-op
PAUSE = time.sleep


def contract_terms(ctx: RunContext) -> dict:
    """The consent and licence values of the family's contract row, as the run notes carry them."""
    row = ctx.contract
    if not row or "consent" not in row or "licence" not in row:
        raise ValueError("scratch_ctgov_psy_exact: the run has no contract row with consent and licence")
    c, lic = row["consent"], row["licence"]
    return {"search_ok": c.get("search_ok"), "train_ok": c.get("train_ok"),
            "consent_basis": c.get("basis"), "consent_resolution": c.get("resolution"),
            "licence_basis": lic.get("basis"), "licence_spdx": lic.get("spdx")}


def _json_capture(unit: Unit, data: dict, url: str, page: int) -> Captured:
    body = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return Captured(unit, body, {"status": 200, "content_type": "application/json",
                                 "source_url": url, "page": page})


@register_family
class CtgovPsyFamily:
    family_id = "scratch_ctgov_psy_exact"
    route = "documented_source_api"
    direct_crawl = True
    chunk_policy_id = CHUNK_DISPATCHER
    http_spec = HttpSpec(kind="curl", timeout=60)   # task 1556: the runner builds and closes the client
    require_records = True
    PARTIAL = ("l1",)

    def __init__(self):
        self._seen: set[str] = set()
        self._robots: dict[str, robotparser.RobotFileParser] = {}

    # -- discover ------------------------------------------------------------------------------
    def discover(self, ctx: RunContext):
        ctx.notes["contract_terms"] = contract_terms(ctx)
        units = [Unit(key=f"ctg-{tag}", url=H.CTG_API, consent_hosts=(CTG_HOST,),
                      meta={"kind": "ctg", "tag": tag, "params": params})
                 for tag, params in H.CTG_QUERIES]
        if not ctx.options.get("skip_ctis"):
            units += [Unit(key=f"ctis-{term}", url=H.CTIS_API, consent_hosts=(CTIS_HOST,),
                           meta={"kind": "ctis", "term": term}) for term in H.CTIS_TERMS]
        for n, (url, lang) in enumerate(H.HTML_PAGES):
            host = urllib.parse.urlparse(url).netloc
            units.append(Unit(key=f"html-{n:02d}", url=url, consent_hosts=(host,),
                              meta={"kind": "html", "lang_default": lang, "host": host}))
        ctx.notes["units"] = {"ctg": len(H.CTG_QUERIES),
                              "ctis": 0 if ctx.options.get("skip_ctis") else len(H.CTIS_TERMS),
                              "html": len(H.HTML_PAGES)}
        return units

    # -- acquire -------------------------------------------------------------------------------
    def acquire(self, unit: Unit, ctx: RunContext):
        kind = unit.meta["kind"]
        client = ctx.http
        if kind == "ctg":
            return self._ctg_pages(unit, client)
        if kind == "ctis":
            return self._ctis_pages(unit, client)
        return self._html_page(unit, client)

    def _ctg_pages(self, unit: Unit, client) -> list[Captured]:
        out, token = [], None
        while True:
            q = {"pageSize": "1000", "fields": H.CTG_FIELDS, **unit.meta["params"]}
            if token:
                q["pageToken"] = token
            r = client.get(H.CTG_API, params=q)
            r.raise_for_status()
            data = r.json()
            out.append(_json_capture(unit, data, H.CTG_API, len(out)))
            token = data.get("nextPageToken")
            PAUSE(1.0)                                   # CTG politeness <= 1 req/s
            if not token:
                return out

    def _ctis_pages(self, unit: Unit, client) -> list[Captured]:
        out, page = [], 1
        while True:
            body = {"pagination": {"page": page, "size": 100},
                    "searchCriteria": {"containAll": unit.meta["term"]},
                    "sort": {"property": "decisionDate", "direction": "DESC"}}
            r = client.post_json(H.CTIS_API, body)
            r.raise_for_status()
            data = r.json()
            out.append(_json_capture(unit, data, H.CTIS_API, page))
            PAUSE(1.5)
            if not data.get("pagination", {}).get("nextPage"):
                return out
            page += 1

    def _robots_for(self, client, host: str) -> robotparser.RobotFileParser:
        return H.robots_for(client, host, self._robots)

    def _html_page(self, unit: Unit, client) -> list[Captured]:
        host = unit.meta["host"]
        rp = self._robots_for(client, host)
        if not rp.can_fetch(H.ROBOTS_TOKEN, unit.url):
            raise UnitExcluded("robots_disallowed")
        delay = max(float(rp.crawl_delay(H.ROBOTS_TOKEN) or 0), H.HTML_DELAY_DEFAULT)
        r = client.get(unit.url)       # a transport error raises: quarantined, never a fact about the origin
        PAUSE(delay)
        if r.status_code == 0 or r.status_code >= 500:
            raise RuntimeError(f"HTTP {r.status_code} for {unit.url}: instrument failure, not an exclusion")
        if r.status_code != 200 or "html" not in r.headers.get("content-type", "").lower():
            raise UnitExcluded(f"http_{r.status_code}_skipped")   # WAF / 404: skip, do not fight it
        return [Captured(unit, r.text.encode("utf-8"),
                         {"status": 200, "content_type": r.headers.get("content-type", ""),
                          "source_url": unit.url})]

    # -- extract / chunk -----------------------------------------------------------------------
    def extract(self, cap: Captured, ctx: RunContext) -> list[Doc]:
        kind = cap.unit.meta["kind"]
        dropped = ctx.notes.setdefault("dropped", {})
        if kind == "html":
            _html_extract.enable_outlink_capture(str(ctx.out_dir))
            try:
                rows = H.html_records(cap.unit.url, cap.body.decode("utf-8"), cap.unit.meta["lang_default"],
                                      cap.unit.meta["host"], today=ctx.today)
            finally:
                _html_extract.disable_outlink_capture()
            return [self._doc(r) for r in rows]
        data = json.loads(cap.body)
        docs = []
        for item in data.get("studies" if kind == "ctg" else "data", []):
            if kind == "ctg":
                url = H.ctg_url(item)
            else:
                ct = item.get("ctNumber")
                url = f"https://euclinicaltrials.eu/ctis-public/view/{ct}" if ct else None
            if url is None or url in self._seen:
                continue
            rec, why = (H.ctg_record(item, ctx.today) if kind == "ctg"
                        else H.ctis_record(item, cap.unit.meta["term"], ctx.today))
            if rec is None:
                dropped[why] = dropped.get(why, 0) + 1
                continue
            self._seen.add(url)
            docs.append(self._doc(rec))
        return docs

    @staticmethod
    def _doc(rec: dict) -> Doc:
        return Doc(url=rec["url"], text=rec["text"], title=rec.get("title") or "",
                   lang=rec.get("lang") or "", source_text=rec["text"], extra={"row": rec})

    def chunk(self, doc: Doc, ctx: RunContext):
        return [doc.extra["row"]]                        # a study / trial record is its own row; a regulator passage is already cut by process_html
