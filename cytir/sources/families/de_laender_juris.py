"""Task 146 — juris jportal (recherche3) German LÄNDER law ingester.

Cracks the ONE XHR recipe that task 139 deferred for the 10 juris jportal Länder:
the modern juris "recherche3" web service the jportal SPA itself calls. Policy (a):
this is the site's own PUBLIC, unauthenticated JSON API — observed once in a browser
to derive the request shape, then replayed with plain httpx (no headless rendering
for bulk collection). Law-LEVEL locator layer (mirrors task 139 / ZH-Lex 126): one
passage per consolidated law = title + canonical permalink + minimal ranking context.

THE RECIPE (per host, all POST /jportal/wsrest/recherche3/<ep> with JSON body that
carries clientID + clientVersion + a per-session r3ID timestamp):
  session bootstrap  cookie jar seeded with  r3autologin="<clientID>"  (literal quotes)
                     -> POST init  returns {csrfToken, user:{login:"Buergerservice…"}}
                        and sets JSESSIONID. The auto-login cookie authenticates the
                        public anonymous account; without csrfToken every later call
                        500s "security_notAuthenticated".
  header on every call   juris-portalid:<clientID>, x-csrf-token:<csrfToken>,
                         Referer/Origin to the host, browser UA.
  enumerate          POST search {searchTasks.RESULT_LIST{start,size<=500,sort:"juris",
                     addCategory:true}, filters.CATEGORY:["Gesetze"], searches:[]}
                     -> paginated resultList; each law-FRAME is a docId ending "rahmen"
                        with titleList[0]=abbreviation, subtitleList[1]=full title,
                        subtitleList[-1]="gültig ab: <dd.mm.yyyy>", date=<dd.mm.yyyy>.
                        (§-level docs jlr-…pP1/pELS/pIVZ are skipped; article-level is
                        a documented follow-up via POST document{docId,format:"xsl"}.)
  canonical url      https://<host>/perma?d=<docId>   (official citable permalink)

Confirmed working on THREE of the ten (clientIDs bsbw / bsst / bssh):
  BW  Baden-Württemberg  www.landesrecht-bw.de              *** NOW BLOCKED, see below
  ST  Sachsen-Anhalt     landesrecht.sachsen-anhalt.de      *** NOW BLOCKED, see below
  SH  Schleswig-Holstein gesetze-rechtsprechung.sh.juris.de     consent UNKNOWN, see below
The other seven are BLOCKED NOT by the platform (the recipe is portable) but by
policy: BE/MV/RP/SL/TH robots disallow AW's token (consent-declined); HH/HE
robots/tdmrep were unreachable (recorded consent-unknown, held).
See task 146 (jsshell platforms).

CONSENT — WHAT WAS OBSERVED, WHEN, FROM WHAT EVIDENCE, AND WHAT THE CODE DID
(task 433 forensics; the bare claim "consent-OK" above was not safe to trust):

  ORDERING WAS CORRECT. All three hosts were resolved 19:27:49-19:27:58 UTC on
  2026-07-09; the single ingest run was 19:50:38-19:52:22 UTC (supervise_146.log,
  SUMMARY total_kept=6007) — consent preceded ingest by ~23 minutes. The five
  declined Laender were never in LANDER, from the first commit (3b4ed395,
  19:48:25 UTC) onward, and NO recherche3 session or content call is attributable
  to any of them.

  BUT THE PRIOR RECON DID TOUCH THEM. Task 139's scouts (depth139_scout*.sh)
  ran 12:56-13:10 UTC against BE/MV/RP/SL/TH/HH/HE — sitemap/toc probes, ~5-10
  GETs per host, honest UA, metadata only. Their robots.txt was first fetched at
  19:27 UTC, ~6.5 HOURS LATER. Recon is crawling; resolve consent before it.

  AND THE THREE "OK" VERDICTS WERE NOT READINGS OF THESE SITES' robots.txt:
    BW  the cached body was 65 KB of the juris.de HTML homepage served as HTTP
        200. Zero User-agent: lines, so every flag defaulted True and the verdict
        read "no opt-out signal found". BW's REAL robots allowlists Googlebot/
        CCBot/OnCrawl and ends "User-agent: * / Disallow: /" — our token is
        disallowed at root. www.landesrecht-bw.de is cached search_ok=false
        (2026-07-14) and is now in data/consent_declined_hosts.json. BLOCKED.
    ST  resolved against juris.de's robots via an unrecorded cross-host redirect
        (the cached body's Sitemap: line points at www.juris.de), so the bare
        key this ingester crawls has NO valid own-host evidence at all. The www
        form, www.landesrecht.sachsen-anhalt.de, carries a valid robots that
        DECLINES: a named-bot allowlist then "User-agent: * / Disallow: /",
        recorded 2026-07-14 as search_ok=false with "our token
        'authoritative-web' disallowed on /". BLOCKED 2026-08-19 (task 433) --
        registered under the bare host, both keys covered. Future crawling only;
        the 999 indexed points are Tim's ruling, not this script's.
    SH  same cross-host redirect; no independent evidence for this host exists
        anywhere. Consent here is genuinely UNKNOWN, not OK.
  2,328 (BW) + 999 (ST) + 2,680 (SH) = 6,007 points are in aw_v4_sat under
  provenance=de_laender_juris. Whether they stay is a DATA decision for Tim.
  BW and ST are blocked for FUTURE crawling; their indexed points are untouched.
  SH is the only one of the three still crawlable, and it is the one with no
  valid evidence in either direction -- treat that as a gap, not a green light.

  HH is NOT "unknown" either: its cached body carries an HTML
  <meta name="tdm-reservation" content="1"> that the resolver's robots-syntax
  matcher cannot see. That is an explicit reservation — a DECLINE we could not
  read. Held-and-unreadable was masking it.

  Full analysis + the resolver fix spec: task 433 (consent violation audit).

Per law: emit an aw_common Passage stamped country=DE, provenance=de_laender_juris,
land=<CODE>. Amendment/repeal/announcement acts dropped by title (139 filter) so the
layer is consolidated in-force statutes & ordinances. Resumable: per-Land done markers
+ appended rows; a re-run skips finished Länder.

  python -m cytir.sources.families.de_laender_juris --land BW,ST,SH --out juris.jsonl [--cap N]


MOVED ONTO THE PROTOCOL (task 1414, S7): family `scratch_de_laender_juris_exact`. One unit per admitted
Land; `acquire` runs the recherche3 session and enumerates the law frames (sealed as one JSON bundle per
Land), `extract` is `rows_for_land` (the old `build_rows` row logic, now shared by the legacy CLI and the
adapter), `chunk` is atomic (one law = one row). The legacy CLI (`main`) is kept; the old
the old flat driver (now a forwarding shim) path is a forwarding shim. One behaviour change: requests now carry
the configured crawler identity (`cytir.core.user_agent`) instead of the browser UA the first run used (the
6,007 points of 2026-07-09 were crawled under an impersonating UA, lessons ledger); the header is built per
request, never at import. Row drops (amendment/repeal titles, no title) are counted in the manifest notes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import threading
import time
import urllib.parse

import httpx

from cytir.consent.block import ConsentBlocked, assert_crawlable
from cytir.core.common import Passage, extractive_snippet
from cytir.sources.family import CHUNK_NATIVE, Captured, Doc, HttpSpec, RunContext, Unit, UnitExcluded
from cytir.sources.http import make_client
from cytir.sources.registry import register_family

# Amendment / repeal / pure-announcement acts carry no consolidated statutory
# content — drop them (mission wants the consolidated in-force law). (139 filter.)
_DROP_TITLE = re.compile(
    r"zur\s+Änderung|zur\s+Aufhebung|Aufhebung\s+der|außer\s+Kraft|"
    r"^\s*(Änderung|Aufhebung|Berichtigung|Bekanntmachung|Neubekanntmachung|"
    r"Neufassung|Erste[sr]?\s+Gesetz\s+zur\s+Änderung)\b",
    re.IGNORECASE)

LANDER = {
    "BW": dict(host="www.landesrecht-bw.de", cid="bsbw", name="Baden-Württemberg",
               collection="juris Landesrecht Baden-Württemberg"),
    "ST": dict(host="landesrecht.sachsen-anhalt.de", cid="bsst", name="Sachsen-Anhalt",
               collection="juris Landesrecht Sachsen-Anhalt"),
    "SH": dict(host="gesetze-rechtsprechung.sh.juris.de", cid="bssh",
               name="Schleswig-Holstein",
               collection="juris Gesetze und Rechtsprechung Schleswig-Holstein"),
}
FAMILY_ID = "scratch_de_laender_juris_exact"

# ------------------------------------------------------------- global politeness
_MIN_INTERVAL = 0.25  # <=4 req/s aggregate per host
_lock = threading.Lock()
_last: dict = {}


def _throttle(host):
    with _lock:
        t = _last.get(host, 0.0)
        dts = time.time() - t
        if dts < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - dts)
        _last[host] = time.time()


def _user_agent() -> str:
    """The configured crawler identity, read when a request is built (never at import)."""
    from cytir.core import user_agent
    return user_agent.user_agent("de_laender_juris")


class Portal:
    """One authenticated recherche3 session for a Land."""

    def __init__(self, host, cid, client=None):
        # task 433: the five consent-DECLINED Laender (BE/MV/RP/SL/TH) and the
        # two consent-UNKNOWN ones (HH/HE) were previously kept out only by
        # their absence from LANDER. Absence is not enforcement -- re-adding a
        # row would have silently resumed crawling a host that said no. The
        # registry now refuses them explicitly.
        assert_crawlable(host)
        self.host, self.cid = host, cid
        self.base = f"https://{host}/jportal/wsrest/recherche3"
        self.rid = dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z")
        if client is None:
            jar = httpx.Cookies()
            jar.set("r3autologin", '"' + cid + '"', domain=host)
            client = make_client(HttpSpec(timeout=45.0, follow_redirects=True))
            client.cookies = jar
        self.c = client
        self.token = None

    def _headers(self) -> dict:
        hh = {"User-Agent": _user_agent(), "Referer": f"https://{self.host}/{self.cid}/search",
              "Origin": f"https://{self.host}", "Content-Type": "application/json",
              "Accept": "*/*", "juris-portalid": self.cid,
              "Cookie": f'r3autologin="{self.cid}"'}
        if self.token:
            hh["x-csrf-token"] = self.token
        return hh

    def post(self, ep, body, tries=4):
        body = {**body, "clientID": self.cid,
                "clientVersion": f"{self.cid} - V08_32_00", "r3ID": self.rid}
        for i in range(tries):
            _throttle(self.host)
            hdr = self._headers()       # outside the try: an unidentified crawler stops, it is not retried
            try:
                r = self.c.post(f"{self.base}/{ep}", json=body, headers=hdr)
            except Exception:  # noqa: BLE001
                if i == tries - 1:
                    return None
                time.sleep(1.5 * (i + 1))
                continue
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:  # noqa: BLE001
                    return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (i + 1))
                continue
            return None
        return None

    def bootstrap(self):
        j = self.post("init", {})
        if not j or "csrfToken" not in j:
            raise RuntimeError(f"init failed for {self.cid}: {j}")
        self.token = j["csrfToken"]
        return j.get("user", {}).get("login")

    def enumerate_gesetze(self, cap=0):
        """Yield law-frame (rahmen) resultList entries across the Gesetze category."""
        start, size, seen = 1, 500, 0
        empty_streak = 0
        while True:
            j = self.post("search", {
                "searchTasks": {"RESULT_LIST": {"start": start, "size": size,
                                                "sort": "juris", "addCategory": True}},
                "filters": {"CATEGORY": ["Gesetze"]}, "searches": []})
            rl = (j or {}).get("resultList", [])
            if not rl:
                empty_streak += 1
                if empty_streak >= 2:
                    break
                start += size
                continue
            empty_streak = 0
            for x in rl:
                if str(x.get("docId", "")).endswith("rahmen"):
                    yield x
                    seen += 1
                    if cap and seen >= cap:
                        return
            start += size


def _last_seen(entry):
    sub = entry.get("subtitleList") or []
    for s in sub:
        m = re.search(r"gültig ab:\s*(\d{1,2})\.(\d{1,2})\.(\d{4})", s or "")
        if m:
            return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    d = entry.get("date") or ""
    m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", d)
    if m:
        return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return None


def _title(entry):
    sub = entry.get("subtitleList") or []
    # subtitleList = [ "Landesnorm <Land>", "<full title>", "gültig ab: …" ]
    full = ""
    for s in sub:
        s = (s or "").strip()
        if not s or s.startswith("Landesnorm") or s.startswith("gültig ab"):
            continue
        full = s
        break
    abbr = ((entry.get("titleList") or [""])[0] or "").strip()
    return full, abbr


def rows_for_land(land, cfg, entries, counts=None):
    """The row of every kept law frame of `entries`, in order. `counts` (a dict) receives the
    drops: `dropped_amend_repeal` (139 title filter) and `no_title` (missing or under 5 chars)."""
    counts = counts if counts is not None else {}
    for entry in entries:
        docid = entry["docId"]
        full, abbr = _title(entry)
        title = full or abbr
        if not title or len(title) < 5:
            counts["no_title"] = counts.get("no_title", 0) + 1
            continue
        if _DROP_TITLE.search(title):
            counts["dropped_amend_repeal"] = counts.get("dropped_amend_repeal", 0) + 1
            continue
        url = (f"https://{cfg['host']}/perma?d="
               + urllib.parse.quote(docid, safe=""))
        ls = _last_seen(entry)
        name_ctx = f"{title}" + (f" ({abbr})" if abbr and abbr not in title else "")
        text = (f"{name_ctx}\n{cfg['collection']} · {cfg['name']} "
                f"(Landesrecht) · amtliche Vorschriftensammlung.")
        pg = Passage(
            url=url, title=title[:300], snippet=extractive_snippet(text, 25),
            text=text, lang="de", domain=cfg["host"], tier="A",
            source_type="official", trust_tier=1.0, anchor="law", last_seen=ls)
        yield {**pg.payload(), "text": pg.text, "country": "DE",
               "provenance": "de_laender_juris", "land": land}


def build_rows(land, cfg, cap, out_path):
    p = Portal(cfg["host"], cfg["cid"])
    login = p.bootstrap()
    print(f"[{land}] authenticated as {login} (cid={cfg['cid']})", flush=True)
    kept, counts = 0, {}
    with open(out_path, "a", encoding="utf-8") as f:
        for row in rows_for_land(land, cfg, p.enumerate_gesetze(cap=cap), counts):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 200 == 0:
                print(f"  [{land}] kept={kept} dropped={counts.get('dropped_amend_repeal', 0)}",
                      flush=True)
    print(f"[{land}] DONE kept={kept} dropped(amend/repeal)={counts.get('dropped_amend_repeal', 0)} "
          f"notitle={counts.get('no_title', 0)}", flush=True)
    return kept


def _admitted_lands(spec):
    requested = [x.strip().upper() for x in spec.split(",") if x.strip()]
    unknown = [x for x in requested if x not in LANDER]
    if unknown:
        # Fail loud: a silently-dropped Land code used to look like a clean run.
        raise SystemExit(
            f"unknown or non-admitted Land code(s): {','.join(unknown)}. "
            f"Admitted: {','.join(sorted(LANDER))}. Codes BE/MV/RP/SL/TH are "
            f"consent-DECLINED and HH/HE consent-UNKNOWN (data/"
            f"consent_declined_hosts.json) -- they are not crawlable.")
    return requested


# ----------------------------------------------------------------- the Protocol
@register_family
class DeLaenderJurisFamily:
    """One unit per admitted Land. Options (`--opt KEY=VALUE`): `land` (default `BW,ST,SH`), `cap`
    (max laws per Land, smoke), `raw` (a directory holding `<LAND>.json`, the enumerated frames:
    replay with no network)."""

    family_id = "scratch_de_laender_juris_exact"      # a literal: the scoreboard reads the class body statically (task 2171)
    assert family_id == FAMILY_ID
    route = "direct_origin_crawl"
    direct_crawl = True
    chunk_policy_id = CHUNK_NATIVE
    require_records = True

    def discover(self, ctx: RunContext):
        lands = _admitted_lands(ctx.options.get("land", ",".join(LANDER)))
        ctx.notes["lands"] = lands
        return [Unit(key=f"land-{land}", url=f"https://{LANDER[land]['host']}/perma",
                     meta={"land": land}, consent_hosts=(LANDER[land]["host"],))
                for land in lands]

    def acquire(self, unit: Unit, ctx: RunContext):
        land = unit.meta["land"]
        cfg = LANDER[land]
        raw = ctx.options.get("raw")
        if raw:
            path = os.path.join(raw, f"{land}.json")
            with open(path, encoding="utf-8") as f:
                entries = json.load(f)
        else:
            try:
                portal = Portal(cfg["host"], cfg["cid"], client=ctx.http)
            except ConsentBlocked as exc:
                raise UnitExcluded(f"consent_registry_blocked:{cfg['host']}") from exc
            portal.bootstrap()                      # raises: the unit is quarantined, not empty
            entries = list(portal.enumerate_gesetze(cap=int(ctx.options.get("cap", 0) or 0)))
        if not entries:
            raise RuntimeError(f"{land}: the enumeration returned no law frames (a failed "
                               "session reads the same as an empty source)")
        body = json.dumps({"land": land, "entries": entries}, ensure_ascii=False,
                          sort_keys=True).encode("utf-8")
        return [Captured(unit, body, {"status": 200, "source_url": unit.url, "frames": len(entries)})]

    def extract(self, cap: Captured, ctx: RunContext):
        bundle = json.loads(cap.body)
        land = bundle["land"]
        counts: dict = {}
        rows = list(rows_for_land(land, LANDER[land], bundle["entries"], counts))
        dropped = ctx.notes.setdefault("dropped", {})
        dropped[land] = dict(sorted(counts.items()))
        if not rows:
            raise UnitExcluded("no_kept_law")
        return [Doc(url=r["url"], text=r["text"], title=r["title"], lang=r["lang"],
                    source_text=r["text"], extra={"row": r}) for r in rows]

    def chunk(self, doc: Doc, ctx: RunContext):
        return [doc.extra["row"]]                    # atomic record: the law locator is the chunk


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--land", required=True, help="comma-sep: BW,ST,SH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=0, help="max laws per land (smoke)")
    args = ap.parse_args(argv)

    done_lands = set()
    marker = args.out + ".lands_done"
    if os.path.exists(marker):
        done_lands = {l.strip() for l in open(marker) if l.strip()}
        print(f"[resume] lands done: {sorted(done_lands)}", flush=True)

    lands = _admitted_lands(args.land)
    grand = 0
    for land in lands:
        if land in done_lands:
            print(f"[{land}] already done — skip", flush=True)
            continue
        kept = build_rows(land, LANDER[land], args.cap, args.out)
        grand += kept
        if not args.cap:  # only mark complete on a full (non-smoke) run
            with open(marker, "a") as f:
                f.write(land + "\n")
    print(f"\nSUMMARY total_kept={grand} out={args.out}", flush=True)


if __name__ == "__main__":
    main()
