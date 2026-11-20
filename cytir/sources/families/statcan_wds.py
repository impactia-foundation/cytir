"""`statcan_wds` -- Statistics Canada Web Data Service locator cards (task 133; moved onto the Protocol, task 1237).

Source (Open Government Licence, Canada; no key): the WDS REST API at
`https://www150.statcan.gc.ca/t1/wds/rest`. ONE authoritative-locator passage per ACTIVE data table
(cube), in EN and FR, pointing at the stable table page
`https://www150.statcan.gc.ca/t1/tbl1/{en|fr}/tv.action?pid={pid}01`; plus one "recent releases" locator
per recent publication day of *The Daily*. No data values are fetched (number-soup penalty, tasks 097 and
108c): the card says WHAT the table measures, its reference period, its latest release; the agent fetches
the table. `last_seen` = the release date. Archived cubes (flag "1") are dropped as superseded.

    getAllCubesListLite   the whole catalogue (~8,200 cubes) in one call
    getCodeSets           subject / frequency code -> EN + FR labels
    getChangedCubeList/D  the product ids released on day D (drives The Daily locators)

The legacy driver had two phases, `enumerate` (network, resumable) and `parse`. Here the three inputs are
ONE unit, `wds-catalogue`: `acquire` fetches them (paced, retried by `cytir.sources.api_fetch`) and returns a
single JSON bundle, which the framework seals; `extract` is the old `parse` over that bundle, one `Doc`
per locator. Row-level drops (archived, no title, thin title, gate, duplicate...) are counted in the
manifest notes (`dropped`), as the driver printed them.

Options (`--opt KEY=VALUE`): `raw` (optional): a directory holding `cubes.json`, `codesets.json` and
`daily.jsonl` (the legacy `enumerate` output) -- replay with no network; `daily_days` (default 120);
`max_req_s` (default 3: the driver's pace of <= 4 req/s, with margin).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
import re
from collections import Counter

from cytir.core import license as LIC
from cytir.core.common import Passage, extraction_junk, extractive_snippet, low_quality
from cytir.sources import api_fetch as AF
from cytir.sources.family import CHUNK_DECLARED, HttpSpec, Captured, Doc, RunContext, Unit, UnitExcluded
from cytir.sources.registry import register_family

log = logging.getLogger(__name__)

WDS = "https://www150.statcan.gc.ca/t1/wds/rest"
HOST = "www150.statcan.gc.ca"
DOMAIN = "statcan.gc.ca"
TABLE_URL = "https://www150.statcan.gc.ca/t1/tbl1/{lang}/tv.action?pid={tid}"
DAILY_URL = "https://www150.statcan.gc.ca/n1/daily-quotidien/{ymd}/dq{ymd}a-{suf}.htm"
LANGS = ("en", "fr")
LANG_SUF = {"en": "eng", "fr": "fra"}
MIN_TITLE_WORDS = 4
MAX_TEXT = 6000
DAILY_DAYS = 120           # look-back window for The Daily recent-release locators
DAILY_MAX_TITLES = 15      # cap titles listed in one Daily locator
DEFAULT_MAX_REQ_S = 3.0
NOT_RELEASED = "not released yet"   # the 409 body WDS gives a recent day whose changed list is not out (task 2184)

AGENCY = {"en": "Statistics Canada", "fr": "Statistique Canada"}
DAILY_NAME = {"en": "The Daily", "fr": "Le Quotidien"}
SUBJECT_L = {"en": "Subject", "fr": "Sujet"}
FREQ_L = {"en": "Frequency", "fr": "Fréquence"}
PERIOD_L = {"en": "Reference period", "fr": "Période de référence"}
RELEASE_L = {"en": "Latest release", "fr": "Dernière diffusion"}
RELEASES_L = {"en": "Statistical releases", "fr": "Diffusions statistiques"}

_WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def _fmt_period(s: str | None) -> str | None:
    return s[:7] if s and len(s) >= 7 else None            # YYYY-MM


def _declared_license() -> str:
    """The licence the host table declares for the WDS host (ADR-039); never a value written here."""
    lic = LIC.resolve_license(None, DOMAIN, HOST, "gov-stat", None)
    if not lic:
        raise RuntimeError(f"cytir.core.license has no licence for {HOST}; statcan_wds stamps none and does not run")
    return lic


def _passage(*, url, title, text, lang, last_seen, provenance) -> dict | None:
    text = text.strip()[:MAX_TEXT]
    if low_quality(text) or extraction_junk(text):
        return None
    p = Passage(url=url, title=title[:300], snippet=extractive_snippet(title, 25)[:200],
                text=text, lang=lang, domain=DOMAIN, tier="A", license=_declared_license(),
                source_type="gov-stat", trust_tier=0.95, last_seen=last_seen)
    d = dataclasses.asdict(p)
    d["country"] = "CA"
    d["provenance"] = provenance
    return d


def cube_cards(cubes: list, codesets: dict, dropped: Counter, seen: set):
    """One card per (active cube, language): the legacy `parse` cube loop, unchanged."""
    subj = {str(x["subjectCode"]): {"en": x["subjectEn"], "fr": x["subjectFr"]}
            for x in codesets.get("subject", [])}
    freq = {str(x["frequencyCode"]): {"en": x["frequencyDescEn"], "fr": x["frequencyDescFr"]}
            for x in codesets.get("frequency", [])}
    for c in cubes:
        if str(c.get("archived")) != "2":                   # keep only active tables
            dropped["archived"] += 1
            continue
        pid = c.get("productId")
        if not pid:
            dropped["no_pid"] += 1
            continue
        subj_codes = [str(x) for x in (c.get("subjectCode") or [])]
        fcode = str(c.get("frequencyCode"))
        p_start, p_end = _fmt_period(c.get("cubeStartDate")), _fmt_period(c.get("cubeEndDate"))
        rel = (c.get("releaseTime") or "")[:10] or None
        for lang in LANGS:
            title = ((c.get("cubeTitleEn") if lang == "en" else c.get("cubeTitleFr")) or "").strip()
            if not title:
                dropped["no_title"] += 1
                continue
            if len({w.lower() for w in _WORD_RE.findall(title)}) < MIN_TITLE_WORDS:
                dropped["thin_title"] += 1
                continue
            parts = [title + ("." if not title.endswith((".", "!", "?")) else "")]
            names = [subj[s][lang] for s in subj_codes if s in subj]
            if names:
                parts.append(f"{SUBJECT_L[lang]}: " + "; ".join(names) + ".")
            if fcode in freq:
                parts.append(f"{FREQ_L[lang]}: {freq[fcode][lang]}.")
            if p_start and p_end:
                parts.append(f"{PERIOD_L[lang]}: {p_start} – {p_end}.")
            if rel:
                parts.append(f"{RELEASE_L[lang]}: {rel}.")
            parts.append(f"{AGENCY[lang]}.")
            rec = _passage(url=TABLE_URL.format(lang=lang, tid=f"{pid}01"), title=title,
                           text=" ".join(parts), lang=lang, last_seen=rel or p_end,
                           provenance="statcan_wds")
            if rec is None:
                dropped["gate"] += 1
            elif rec["point_id"] in seen:
                dropped["dup"] += 1
            else:
                seen.add(rec["point_id"])
                yield rec


def daily_cards(cubes: list, daily: list, dropped: Counter, seen: set):
    """One card per (release day, language): the legacy `parse` Daily loop, unchanged."""
    title_by_pid = {str(c.get("productId")): {"en": c.get("cubeTitleEn"), "fr": c.get("cubeTitleFr")}
                    for c in cubes}
    for rec in daily:
        day = rec["date"]                                    # YYYY-MM-DD
        pids = [str(p) for p in rec.get("productIds") or []]
        if not pids:
            dropped["daily_empty"] += 1
            continue
        ymd = day[2:].replace("-", "")                       # YYMMDD
        for lang in LANGS:
            titles = []
            for p in pids:
                t = (title_by_pid.get(p) or {}).get(lang)
                if t:
                    titles.append(t.strip())
                if len(titles) >= DAILY_MAX_TITLES:
                    break
            if not titles:
                dropped["daily_no_titles"] += 1
                continue
            body = (f"{AGENCY[lang]}, {DAILY_NAME[lang]}, {day}. "
                    f"{RELEASES_L[lang]}: " + "; ".join(titles) + ".")
            drec = _passage(url=DAILY_URL.format(ymd=ymd, suf=LANG_SUF[lang]),
                            title=f"{DAILY_NAME[lang]} — {day}", text=body, lang=lang,
                            last_seen=day, provenance="statcan_daily")
            if drec is None:
                dropped["gate"] += 1
            elif drec["point_id"] in seen:
                dropped["dup"] += 1
            else:
                seen.add(drec["point_id"])
                yield drec


def _read_raw(raw: str) -> dict:
    raw = os.path.expanduser(raw)
    daily = []
    path = os.path.join(raw, "daily.jsonl")
    if os.path.exists(path):
        daily = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    return {"cubes": json.load(open(os.path.join(raw, "cubes.json"), encoding="utf-8")),
            "codesets": json.load(open(os.path.join(raw, "codesets.json"), encoding="utf-8")),
            "daily": daily}


@register_family
class StatcanWdsFamily:
    family_id = "statcan_wds"
    route = "documented_source_api"
    direct_crawl = True
    chunk_policy_id = CHUNK_DECLARED
    http_spec = HttpSpec(timeout=90)   # task 1556: the runner builds and closes the client
    require_records = True

    def discover(self, ctx: RunContext):
        raw = ctx.options.get("raw")
        ctx.notes.update({"raw": raw or None, "daily_days": int(ctx.options.get("daily_days", DAILY_DAYS))})
        return [Unit(key="wds-catalogue", url=f"{WDS}/getAllCubesListLite", consent_hosts=(HOST,))]

    def acquire(self, unit: Unit, ctx: RunContext):
        raw = ctx.options.get("raw")
        if raw:
            bundle = _read_raw(raw)
        else:
            client = ctx.http
            pacer = AF.Pacer(float(ctx.options.get("max_req_s", DEFAULT_MAX_REQ_S)))
            bundle = {"cubes": AF.get(client, f"{WDS}/getAllCubesListLite", pacer).json(),
                      "codesets": AF.get(client, f"{WDS}/getCodeSets", pacer).json()["object"],
                      "daily": []}
            today = dt.date.fromisoformat(ctx.today)
            for d in range(1, ctx.notes["daily_days"] + 1):
                day = (today - dt.timedelta(days=d)).isoformat()
                resp = AF.get(client, f"{WDS}/getChangedCubeList/{day}", pacer, absent=(409,))
                if resp.status_code == 409:
                    if NOT_RELEASED not in resp.text.lower():
                        raise RuntimeError(f"GET getChangedCubeList/{day}: HTTP 409 {resp.text[:120]!r}")
                    ctx.notes.setdefault("daily_skipped", []).append(day)
                    log.warning("statcan_wds: %s skipped, WDS answers 409 %r (day not released yet)",
                                day, resp.text[:80])
                    continue
                obj = resp.json().get("object") or []
                bundle["daily"].append(
                    {"date": day, "productIds": [x.get("productId") for x in obj if x.get("productId")]})
        if not bundle["cubes"]:
            raise UnitExcluded("empty_catalogue")
        if "object" in bundle["codesets"]:                   # the raw file keeps the API envelope
            bundle["codesets"] = bundle["codesets"]["object"]
        body = json.dumps(bundle, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return [Captured(unit, body, {"status": 200, "source_url": unit.url,
                                      "cubes": len(bundle["cubes"]), "days": len(bundle["daily"])})]

    def extract(self, cap: Captured, ctx: RunContext):
        bundle = json.loads(cap.body)
        dropped, seen = Counter(), set()
        recs = list(cube_cards(bundle["cubes"], bundle["codesets"], dropped, seen))
        recs += list(daily_cards(bundle["cubes"], bundle["daily"], dropped, seen))
        ctx.notes["dropped"] = dict(sorted(dropped.items()))
        if not recs:
            raise UnitExcluded("no_card")
        return [Doc(url=r["url"], text=r["text"], title=r["title"], lang=r["lang"], source_text=r["text"],
                    extra={"row": r})
                for r in recs]

    def chunk(self, doc: Doc, ctx: RunContext):
        return [doc.extra["row"]]                            # atomic record: the card is the chunk
