"""Task 177 — ClinicalTrials.gov psychedelic+psychiatry slice (round-5 ingest #3). API-first (dump/API route, not a crawl), anti-hype bar:
registries / regulators / gov ONLY (no MAPS/advocacy hosts — separate decision).

Four sources, one wave (provenance=ctgov_psy):

  A. ClinicalTrials.gov API v2 (US gov, public domain, structured) —
     * psychedelic slice by intervention: psilocybin, MDMA/midomafetamine,
       LSD, DMT/5-MeO-DMT, ayahuasca, mescaline, ibogaine (ALL phases/status —
       the seminal Swiss trials are old), + ketamine/esketamine restricted to
       psychiatric indications, + query.term=psychedelic sweep.
     * curated psychiatry broadening: depression/PTSD/anxiety/OCD,
       INTERVENTIONAL phase 2/3, RECRUITING or COMPLETED-with-results.
     One study -> ONE record: url=https://clinicaltrials.gov/study/NCTxxxxxxxx,
     title=official title (fallback brief), snippet <=25 words from brief
     summary, tier=B trust 0.9 source_type=registry_record.
  B. EU CTIS public API (euclinicaltrials.eu) — same psychedelic terms;
     ketamine hits kept only when psychiatric. Same record shape,
     url=https://euclinicaltrials.eu/ctis-public/view/<ctNumber>.
  C. FDA guidance landing pages (curated tiny list; fda.gov robots
     crawl-delay 30 honored; Akamai-WAF'd from datacenters — the operator's build host fetches
     clean; any non-200 is SKIPPED, we do not fight the WAF).
  D. Swissmedic + CH BAG Art. 8 BetmG compassionate-use pages (de/fr/it/en —
     CH's special-access role). C+D are official regulators: tier A trust 1.0
     source_type=official, extracted via aw_common.process_html so gates
     097 extraction-junk + 166 language-scope + 167 listing-shell all apply.

API-derived registry records still pass the 166 language gate (fetch_process.
detect_lang over the record text; confident out-of-scope -> dropped).

Resumable + detached: per-query / per-term / per-URL progress in state.json,
atomic append to passages.jsonl. Politeness: UA AuthoritativeWebBot, robots
respected, CTG API <=1 req/s.

Moved here from the flat driver script (task 1407, S7); the old path forwards to `main()`.
The record builders (`ctg_record`, `ctis_record`, `html_records`) are pure so the family
adapter (`cytir.sources.families.ctgov_psy`) calls the very same code; the loops below keep the
operator's resumable one-shot behaviour.

  python3 -m cytir.sources.ctgov_psy.harvest --out-dir <dir>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
import urllib.parse
from dataclasses import asdict
from urllib import robotparser

from cytir.acquire import fetch_process as fp
from cytir.consent.block import assert_crawlable
from cytir.core import user_agent
from cytir.core.common import mint_card_point_id
from cytir.extract.html_extract import enable_outlink_capture, process_html

#: the robots.txt product token; the full User-Agent comes from the operator identity (fails closed)
ROBOTS_TOKEN = user_agent.PACK_FETCH_PRODUCT
TODAY = dt.date.today().isoformat()
CTG_API = "https://clinicaltrials.gov/api/v2/studies"
CTIS_API = "https://euclinicaltrials.eu/ctis-public-api/search"

CTG_FIELDS = ("NCTId,OfficialTitle,BriefTitle,BriefSummary,OverallStatus,"
              "Phase,StudyType,Condition,InterventionName,HasResults,"
              "LeadSponsorName,CollaboratorName,LocationCountry")

# --- A. ClinicalTrials.gov query plan (tag, params) --------------------------
PSYCH_COND = ("depression OR depressive disorder OR PTSD OR post-traumatic "
              "stress disorder OR anxiety disorder OR obsessive-compulsive "
              "disorder OR suicidal ideation OR bipolar disorder")
CTG_QUERIES = [
    ("psilocybin",  {"query.intr": "psilocybin"}),
    ("mdma",        {"query.intr": "MDMA OR midomafetamine"}),
    ("lsd",         {"query.intr": "lysergic acid diethylamide OR LSD"}),
    ("dmt",         {"query.intr": "N,N-dimethyltryptamine OR DMT"}),
    ("5meodmt",     {"query.intr": "5-MeO-DMT"}),
    ("ayahuasca",   {"query.intr": "ayahuasca"}),
    ("mescaline",   {"query.intr": "mescaline"}),
    ("ibogaine",    {"query.intr": "ibogaine OR noribogaine"}),
    ("ketamine_psy", {"query.intr": "ketamine OR esketamine",
                      "query.cond": PSYCH_COND}),
    ("psychedelic_term", {"query.term": "psychedelic"}),
    # curated psychiatry broadening: interventional phase 2/3 only,
    # recruiting OR completed-with-results (172 scout: keep it curated).
    ("psychiatry_ph23_recruiting", {
        "query.cond": ("depression OR depressive disorder OR PTSD OR "
                       "post-traumatic stress disorder OR anxiety disorder OR "
                       "obsessive-compulsive disorder"),
        "filter.advanced": ("AREA[StudyType]INTERVENTIONAL AND "
                            "(AREA[Phase]PHASE2 OR AREA[Phase]PHASE3)"),
        "filter.overallStatus": "RECRUITING"}),
    ("psychiatry_ph23_completed_results", {
        "query.cond": ("depression OR depressive disorder OR PTSD OR "
                       "post-traumatic stress disorder OR anxiety disorder OR "
                       "obsessive-compulsive disorder"),
        "filter.advanced": ("AREA[StudyType]INTERVENTIONAL AND "
                            "(AREA[Phase]PHASE2 OR AREA[Phase]PHASE3) AND "
                            "AREA[HasResults]true"),
        "filter.overallStatus": "COMPLETED"}),
]

# --- B. CTIS terms (ALL kept unless ketamine -> psychiatric filter) ----------
CTIS_TERMS = ["psilocybin", "psilocin", "MDMA", "midomafetamine", "lysergide",
              "LSD", "dimethyltryptamine", "5-MeO-DMT", "ayahuasca",
              "mescaline", "ibogaine", "esketamine", "ketamine"]
CTIS_KETAMINE_TERMS = {"esketamine", "ketamine"}
_PSY_WORDS = ("depress", "ptsd", "stress disorder", "anxiet", "obsess",
              "suicid", "bipolar", "psychiatr")

# --- C+D. curated regulator pages (official: tier A) -------------------------
HTML_PAGES = [  # (url, lang_default)
    # FDA (public domain; robots crawl-delay 30; skip on any non-200)
    ("https://www.fda.gov/regulatory-information/search-fda-guidance-documents/psychedelic-drugs-considerations-clinical-investigations", "en"),
    ("https://www.fda.gov/news-events/press-announcements/fda-issues-first-draft-guidance-clinical-trials-psychedelic-drugs", "en"),
    ("https://www.fda.gov/regulatory-information/search-fda-guidance-documents/major-depressive-disorder-developing-drugs-treatment", "en"),
    # CH BAG Art. 8 BetmG limited-medical-use (compassionate access) de/fr/it
    ("https://www.bag.admin.ch/de/beschraenkte-medizinische-anwendung-von-verbotenen-betaeubungsmitteln", "de"),
    ("https://www.bag.admin.ch/fr/application-medicale-limitee-de-stupefiants-interdits", "fr"),
    ("https://www.bag.admin.ch/it/applicazione-medica-limitata-di-stupefacenti-vietati", "it"),
    # Swissmedic authorised-narcotics legal basis (en/de/fr/it) + section pages
    ("https://www.swissmedic.ch/swissmedic/en/home/humanarzneimittel/besondere-arzneimittelgruppen--ham-/authorised-narcotics/legal-basis.html", "en"),
    ("https://www.swissmedic.ch/swissmedic/de/home/humanarzneimittel/besondere-arzneimittelgruppen--ham-/narcotics/rechtsgrundlagen.html", "de"),
    ("https://www.swissmedic.ch/swissmedic/fr/home/medicaments-a-usage-humain/besondere-arzneimittelgruppen--ham-/stupefiants-autorises/bases-juridiques.html", "fr"),
    ("https://www.swissmedic.ch/swissmedic/it/home/medicamenti-per-uso-umano/besondere-arzneimittelgruppen--ham-/stupefacenti-omologati/basi-giuridiche.html", "it"),
    ("https://www.swissmedic.ch/swissmedic/en/home/humanarzneimittel/besondere-arzneimittelgruppen--ham-/authorised-narcotics.html", "en"),
    ("https://www.swissmedic.ch/swissmedic/de/home/humanarzneimittel/besondere-arzneimittelgruppen--ham-/narcotics.html", "de"),
    ("https://www.swissmedic.ch/swissmedic/fr/home/medicaments-a-usage-humain/besondere-arzneimittelgruppen--ham-/stupefiants-autorises.html", "fr"),
    ("https://www.swissmedic.ch/swissmedic/it/home/medicamenti-per-uso-umano/besondere-arzneimittelgruppen--ham-/stupefacenti-omologati.html", "it"),
]
HTML_DELAY_DEFAULT = 2.0  # s; per-host robots crawl-delay overrides upward


# --- HTTP via curl subprocess ---------------------------------------------------
# clinicaltrials.gov (and some regulator hosts) TLS-fingerprint-block python
# httpx/requests while accepting plain curl with the SAME honest UA (verified
# 2026-07-12: httpx 403, curl 200, headers identical). The CTG API v2 is the
# official public bulk route — using curl is a client choice, not WAF evasion.
class Resp:
    def __init__(self, status, text, ctype, url):
        self.status_code, self.text, self.ctype, self.url = status, text, ctype, url
        self.headers = {"content-type": ctype}

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code} for {self.url}")


class CurlClient:
    """Minimal curl-subprocess client: get(url, params) / post_json(url, body)."""

    def __init__(self, timeout=60):
        self.timeout = timeout

    def close(self):
        """No pooled connection to release (one curl subprocess per call); the runner calls it."""

    MAX_HOPS = 10   # same limit as the bench hop loop

    def _once(self, args, url):
        """One curl run, no redirect following: (status, location, ctype, body)."""
        out = subprocess.run(
            ["curl", "-sS", "-A", user_agent.pack_fetch_user_agent(), "--max-time", str(self.timeout),
             "-w", "\n__AW_META__ %{http_code}\t%{redirect_url}\t%{content_type}", *args, url],
            capture_output=True, text=True, timeout=self.timeout + 15)
        body, _, meta = out.stdout.rpartition("\n__AW_META__ ")
        parts = meta.split("\t", 2) if meta else ["0", "", ""]
        try:
            status = int(parts[0])
        except ValueError:
            status = 0
        return (status, parts[1].strip() if len(parts) > 1 else "",
                parts[2] if len(parts) > 2 else "", body)

    def _run(self, args, url):
        """Manual redirect loop: every hop's host passes the consent choke point
        (spec section 2, curl) BEFORE its curl run. The -w trailer carries the
        status, redirect target and content type after the body; it is parsed
        with one rpartition on a marker, so the body is never split or re-read
        (a header block from -D - would need splitting off 1xx/proxy blocks)."""
        cur = url
        for _ in range(self.MAX_HOPS + 1):
            assert_crawlable(urllib.parse.urlsplit(cur).hostname or "")
            status, loc, ctype, body = self._once(args, cur)
            if status not in (301, 302, 303, 307, 308):
                return Resp(status, body, ctype, url)
            if not loc:
                raise RuntimeError(f"HTTP {status} without Location for {cur}")
            # Every hop re-issues the SAME method with the SAME body and headers:
            # curl -X POST -L kept POST on every hop; we keep parity.
            cur = urllib.parse.urljoin(cur, loc)
        raise RuntimeError(f"HTTP {status}: more than {self.MAX_HOPS} redirects from {url}")

    def get(self, url, params=None):
        if params:
            url = url + ("&" if "?" in url else "?") + \
                urllib.parse.urlencode(params)
        return self._run([], url)

    def post_json(self, url, body):
        return self._run(["-X", "POST", "-H", "Content-Type: application/json",
                          "-d", json.dumps(body)], url)


# --- helpers ------------------------------------------------------------------
def load_state(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"ctg_done": [], "ctis_done": [], "html_done": [],
            "seen": [], "ctg_records": 0, "ctis_records": 0,
            "ctis_ketamine_skipped": 0, "html_pages_ok": 0,
            "html_pages_skipped": 0, "html_passages": 0,
            "lang_dropped": 0, "per_slice": {}}


def save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def snippet25(text: str) -> str:
    words = (text or "").split()
    return " ".join(words[:25]) + ("…" if len(words) > 25 else "")


def make_record(url: str, domain: str, title: str, text: str, snippet: str,
                tier: str, trust: float, source_type: str, lang: str,
                today: str | None = None) -> dict:
    return {
        "url": url, "title": title, "snippet": snippet, "text": text,
        "lang": lang, "domain": domain, "tier": tier,
        "source_type": source_type, "trust_tier": trust, "anchor": None,
        "last_seen": today or TODAY,
        "point_id": mint_card_point_id(url),
        "provenance": "ctgov_psy",
    }


def lang_gate(text: str, url: str, default: str):
    """166 gate for API-derived records: confident out-of-scope -> None."""
    lang, conf = fp.detect_lang(text, url, default)
    if conf and lang not in fp.LANG_SCOPE:
        return None
    return lang if lang in fp.LANG_SCOPE else default


def ctg_url(study: dict) -> str | None:
    """The record URL of one API v2 study, or None when it has no NCT id."""
    nct = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
    return f"https://clinicaltrials.gov/study/{nct}" if nct else None


def ctg_record(st: dict, today: str | None = None) -> tuple[dict | None, str]:
    """One API v2 study -> (record, ""), or (None, reason): `no_nct`, `lang_dropped`."""
    proto = st.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    nct = ident.get("nctId")
    if not nct:
        return None, "no_nct"
    url = f"https://clinicaltrials.gov/study/{nct}"
    title = (ident.get("officialTitle")
             or ident.get("briefTitle") or nct).strip()
    summary = (proto.get("descriptionModule", {})
               .get("briefSummary") or "").strip()
    conds = proto.get("conditionsModule", {}).get("conditions", [])
    intrs = [i.get("name", "") for i in
             proto.get("armsInterventionsModule", {})
             .get("interventions", [])]
    design = proto.get("designModule", {})
    status = proto.get("statusModule", {}).get("overallStatus", "")
    spons = proto.get("sponsorCollaboratorsModule", {})
    lead = spons.get("leadSponsor", {}).get("name", "")
    collab = [x.get("name", "") for x in
              spons.get("collaborators", [])][:4]
    countries = []
    for loc in (proto.get("contactsLocationsModule", {})
                .get("locations") or []):
        ctry = loc.get("country")
        if ctry and ctry not in countries:
            countries.append(ctry)
    meta_line = (
        f"Conditions: {'; '.join(conds[:8])}. "
        f"Interventions: {'; '.join(intrs[:8])}. "
        f"Phase: {', '.join(design.get('phases', []) or ['N/A'])}. "
        f"Study type: {design.get('studyType', '')}. "
        f"Status: {status}. "
        f"Sponsor: {lead}"
        f"{(' with ' + '; '.join(c for c in collab if c)) if collab else ''}. "
        f"Countries: {', '.join(countries[:10])}. "
        f"Registry: ClinicalTrials.gov {nct}.")
    text = "\n".join(x for x in (title, meta_line, summary) if x)
    lang = lang_gate(text, url, "en")
    if lang is None:
        return None, "lang_dropped"
    return make_record(url, "clinicaltrials.gov", title, text,
                       snippet25(summary or meta_line),
                       "B", 0.9, "registry_record", lang, today), ""


# --- A. ClinicalTrials.gov ----------------------------------------------------
def ctg_harvest(client, state, emit, checkpoint):
    for tag, params in CTG_QUERIES:
        if tag in state["ctg_done"]:
            continue
        n_new = 0
        page_token = None
        while True:
            q = {"pageSize": "1000", "fields": CTG_FIELDS, **params}
            if page_token:
                q["pageToken"] = page_token
            r = client.get(CTG_API, params=q)
            r.raise_for_status()
            data = r.json()
            for st in data.get("studies", []):
                url = ctg_url(st)
                if url is None or url in state["seen"]:
                    continue
                rec, why = ctg_record(st)
                if rec is None:
                    state["lang_dropped"] += 1
                    continue
                emit(rec)
                state["seen"].append(url)
                state["ctg_records"] += 1
                n_new += 1
            page_token = data.get("nextPageToken")
            time.sleep(1.0)  # CTG politeness <=1 req/s
            if not page_token:
                break
        state["per_slice"][tag] = n_new
        state["ctg_done"].append(tag)
        checkpoint()
        print(f"[ctg] {tag}: +{n_new} new (total {state['ctg_records']})",
              flush=True)


def ctis_record(t: dict, term: str, today: str | None = None) -> tuple[dict | None, str]:
    """One CTIS search hit -> (record, ""), or (None, reason): `no_ct`,
    `ketamine_not_psy`, `lang_dropped`."""
    ct = t.get("ctNumber")
    if not ct:
        return None, "no_ct"
    url = f"https://euclinicaltrials.eu/ctis-public/view/{ct}"
    blob = " ".join([t.get("ctTitle", ""),
                     t.get("conditions", "") or "",
                     " ".join(t.get("therapeuticAreas") or [])])
    if term in CTIS_KETAMINE_TERMS and \
            not any(w in blob.lower() for w in _PSY_WORDS):
        return None, "ketamine_not_psy"
    title = (t.get("ctTitle") or ct).strip()
    countries = ", ".join(
        c.split(":")[0] for c in (t.get("trialCountries") or []))
    meta_line = (
        f"Conditions: {t.get('conditions') or 'N/A'}. "
        f"Sponsor: {t.get('sponsor', '')}. "
        f"Countries: {countries}. "
        f"Therapeutic areas: "
        f"{'; '.join(t.get('therapeuticAreas') or [])}. "
        f"Decision date: {t.get('decisionDateOverall', '')}. "
        f"Registry: EU CTIS {ct}.")
    text = "\n".join(x for x in (title, meta_line) if x)
    lang = lang_gate(text, url, "en")
    if lang is None:
        return None, "lang_dropped"
    return make_record(url, "euclinicaltrials.eu", title, text,
                       snippet25(meta_line), "B", 0.9,
                       "registry_record", lang, today), ""


# --- B. EU CTIS ----------------------------------------------------------------
def ctis_harvest(client, state, emit, checkpoint):
    for term in CTIS_TERMS:
        if term in state["ctis_done"]:
            continue
        n_new, page = 0, 1
        while True:
            body = {"pagination": {"page": page, "size": 100},
                    "searchCriteria": {"containAll": term},
                    "sort": {"property": "decisionDate", "direction": "DESC"}}
            r = client.post_json(CTIS_API, body)
            r.raise_for_status()
            data = r.json()
            for t in data.get("data", []):
                ct = t.get("ctNumber")
                if not ct:
                    continue
                url = f"https://euclinicaltrials.eu/ctis-public/view/{ct}"
                if url in state["seen"]:
                    continue
                rec, why = ctis_record(t, term)
                if rec is None:
                    state["ctis_ketamine_skipped" if why == "ketamine_not_psy"
                          else "lang_dropped"] += 1
                    continue
                emit(rec)
                state["seen"].append(url)
                state["ctis_records"] += 1
                n_new += 1
            time.sleep(1.5)
            if not data.get("pagination", {}).get("nextPage"):
                break
            page += 1
        state["ctis_done"].append(term)
        checkpoint()
        print(f"[ctis] {term}: +{n_new} new (total {state['ctis_records']})",
              flush=True)


# --- C+D. curated regulator pages ----------------------------------------------
_ROBOTS: dict[str, robotparser.RobotFileParser] = {}


def robots_for(client, host, cache=None):
    cache_ = _ROBOTS if cache is None else cache
    if host not in cache_:
        rp = robotparser.RobotFileParser()
        try:
            r = client.get(f"https://{host}/robots.txt")
            rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        except Exception:  # noqa: BLE001
            rp.parse([])
        cache_[host] = rp
    return cache_[host]


def html_records(url: str, html: str, lang_default: str, host: str,
                 today: str | None = None, max_chunks: int = 50) -> list[dict]:
    """One regulator page -> its passages (tier A, official), through the shared
    `process_html` so the extraction-junk, language-scope and listing-shell gates apply."""
    meta = {"content": True, "links": False, "want_stats": True,
            "lang_default": lang_default,
            "passage_common": {"domain": host.removeprefix("www."),
                               "tier": "A", "trust_tier": 1.0,
                               "source_type": "official",
                               "last_seen": today or TODAY}}
    passages, _links, _stats = process_html(url, html, meta)
    out = []
    for p in passages[:max_chunks]:
        rec = asdict(p)
        rec.pop("content_digest", None)
        rec["provenance"] = "ctgov_psy"
        out.append(rec)
    return out


def html_harvest(client, state, emit, max_chunks=50):
    for url, lang_default in HTML_PAGES:
        if url in state["html_done"]:
            continue
        host = urllib.parse.urlparse(url).netloc
        rp = robots_for(client, host)
        if not rp.can_fetch(ROBOTS_TOKEN, url):
            print(f"[html] robots-disallowed, skip: {url}", flush=True)
            state["html_pages_skipped"] += 1
            state["html_done"].append(url)
            continue
        delay = max(float(rp.crawl_delay(ROBOTS_TOKEN) or 0), HTML_DELAY_DEFAULT)
        try:
            r = client.get(url)
        except Exception as e:  # noqa: BLE001
            print(f"[html] fetch error, skip: {url} ({e})", flush=True)
            state["html_pages_skipped"] += 1
            state["html_done"].append(url)
            time.sleep(delay)
            continue
        if r.status_code != 200 or "html" not in \
                r.headers.get("content-type", "").lower():
            # WAF/404 -> skip, do NOT fight it (task rule)
            print(f"[html] {r.status_code}, skip: {url}", flush=True)
            state["html_pages_skipped"] += 1
            state["html_done"].append(url)
            time.sleep(delay)
            continue
        passages = html_records(url, r.text, lang_default, host, max_chunks=max_chunks)
        for rec in passages:
            emit(rec)
        state["html_pages_ok"] += 1
        state["html_passages"] += len(passages)
        state["html_done"].append(url)
        print(f"[html] ok {len(passages):3d} passages  {url}", flush=True)
        time.sleep(delay)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-ctis", action="store_true",
                    help="skip EU CTIS if the API misbehaves")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    # harvest every fetched page's external links for the
    # endorsement graph (streamed to <out-dir>/outlinks.jsonl; passage
    # extraction unchanged).
    enable_outlink_capture(args.out_dir)
    state_path = os.path.join(args.out_dir, "state.json")
    passages_path = os.path.join(args.out_dir, "passages.jsonl")
    state = load_state(state_path)

    pf = open(passages_path, "a", encoding="utf-8")

    def emit(rec):
        pf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def checkpoint():
        pf.flush()
        save_state(state_path, state)

    client = CurlClient(timeout=60)
    print(f"=== ctg harvest ({len(CTG_QUERIES)} query slices) ===", flush=True)
    try:
        ctg_harvest(client, state, emit, checkpoint)
    finally:
        checkpoint()
    if not args.skip_ctis:
        print(f"=== ctis harvest ({len(CTIS_TERMS)} terms) ===", flush=True)
        try:
            ctis_harvest(client, state, emit, checkpoint)
        except Exception as e:  # noqa: BLE001
            print(f"[ctis] API flaky, skipping remainder per task rule: "
                  f"{e}", flush=True)
        finally:
            checkpoint()
    print(f"=== regulator pages ({len(HTML_PAGES)}) ===", flush=True)
    try:
        html_harvest(client, state, emit)
    finally:
        checkpoint()

    pf.close()
    save_state(state_path, state)
    total = state["ctg_records"] + state["ctis_records"] + \
        state["html_passages"]
    print(json.dumps({k: v for k, v in state.items() if k not in
                      ("seen", "ctg_done", "ctis_done", "html_done")},
                     indent=1), flush=True)
    print(f"DONE total rows={total} -> {passages_path}", flush=True)


if __name__ == "__main__":
    main()
