#!/usr/bin/env python3
"""Task 187 — fundamental-rights canon pack (droits_fondamentaux; scout 185
§2.2 recommended ingest #2). Polite multi-host crawl of court / UN / EU / NHRI
fundamental-rights authorities + two non-crawl routes:

  crawl (sitemap/BFS, robots verified live per host):
    ohchr.org        — UN OHCHR. robots grants `Content-Signal:
                       search=yes,ai-train=no` for `*` and Disallows CCBot →
                       direct polite crawl is the ONLY route; consent gate runs
                       --direct-crawl --accept-search-grant and every row
                       carries train_ok=false. EN+FR canon slice + 2025/26
                       press releases.
    fra.europa.eu    — EU Fundamental Rights Agency (en/fr/de; Charterpedia
                       law/case-law references, publications, news).
    cncdh.fr         — French NHRI (no sitemap → whole-site BFS, small).
    defenseurdesdroits.fr — Défenseur des droits (sitemap; locs carry a
                       Drupal `http://default/` base → host rewritten).
    cnil.fr          — AI/rights/sanctions SLICE only (host already indexed
                       whole-site by an earlier wave).
    bundesverfassungsgericht.de — basepage sitemap + Aktuelles/Presse BFS +
                       curated landmark /SharedDocs/Entscheidungen pages
                       (full decision corpus already held via
                       rechtsprechung-im-internet.de, task 132). Crawl-delay 30.
    antidiskriminierungsstelle.de — federal anti-discrimination body
                       (sitemap; crawl-delay 30).
  API:
    hudoc.echr.coe.int — leading ECtHR case law via the de-facto JSON query
                       API (echr.coe.int itself is WAF'd): Grand Chamber +
                       importance=1 ("key case") judgments, EN (HEJUD) + FR
                       (HFJUD). ONE landing/metadata record per judgment
                       (ctgov pattern, task 177) with the canonical
                       hudoc.echr.coe.int/eng?i=<itemid> URL.
  bulk (proven task-145 EUR-Lex legal-content HTML endpoint, NOT a crawl):
    eur-lex.europa.eu — EU Charter of Fundamental Rights (CELEX 12012P)
                       article-level EN/FR/DE. (AI Act 2024/1689 + GDPR are
                       ALREADY held art-addressed from task 145 — skipped.)

Pre-verified SKIPS (185 map; noted for cards / CC route later):
  institut-fuer-menschenrechte.de (DIMR, 403 datacenter), echr.coe.int +
  coe.int + rm.coe.int (WAF-403), equalityhumanrights.com (EHRC, WAF-403).

Extraction via aw_common.process_html => gates 097 + 166 (fasttext lid) +
167 + 184 (sitemap/autoindex shells) all live. Apparatus filter + over-chunk
head-cap 50/url (175 pattern). provenance=rights_canon; tier=A;
trust 1.0 courts / 0.95 UN+EU / 0.9 national bodies.

Resumable + detached (175/176 pattern): per-URL atomic append to
passages.jsonl, done-set in state.json; re-run skips completed work.

  python3 -m cytir.sources.rights_canon.crawl --out-dir <dir>

Task 1408 (S7): moved here from the retired driver `ingest_rights_pack`. The
per-record and per-page logic (`page_records`, `hudoc_record`, `charter_records`) is what the
`scratch_rights_canon_exact` family adapter (the `rights_canon` module of `cytir.sources.families`) calls; the
resumable crawl loop below calls the same functions. One home for the record shape.
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import gzip
import json
import os
import re
import sys
import time
import urllib.parse
from dataclasses import asdict

import httpx
from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

from cytir.core.canon import https_frontier_url
from cytir.core.common import Passage
from cytir.extract.html_extract import enable_outlink_capture, process_html

def _identity_ua() -> str:
    """The bot's UA token plus the operator identity this install configured (never hard-coded)."""
    from cytir.core import user_agent
    url, contact, _configured = user_agent.operator()
    return f"AuthoritativeWebBot (+{url}; {contact})"


UA_TOKEN = "authoritativewebbot"


def headers() -> dict:
    """Built at request time, not import time: importing the module must not read the crawler identity (cut test, task 1408)."""
    return {"User-Agent": _identity_ua()}


def __getattr__(name: str):
    """Lazy compat for the retired driver shim: `UA` and `HEADERS` resolve per access, never at import."""
    if name == "HEADERS":
        return headers()
    if name == "UA":
        return _identity_ua()
    raise AttributeError(name)


def require_identified() -> None:
    """Fail closed: a crawl with no configured operator identity does not start (`cytir.core.user_agent`)."""
    from cytir.core import user_agent
    user_agent.user_agent("rights_canon")
PROVENANCE = "rights_canon"

BINARY_RE = re.compile(
    r"(?i)\.(pdf|jpe?g|png|gif|svg|zip|docx?|xlsx?|pptx?|mp[34]|avi|mov|ico|"
    r"css|js|json|rss|epub|webp|odt|ods)(\?|$)")

_CITE_RE = re.compile(
    r'(?:\b(?:19|20)\d{2}[a-z]?[,)\].]|doi:|\b\d{1,4}[–-]\d{1,4}\b|ISBN|, pp?\.)')
_ITEM_RE = re.compile(r'^\s*(?:[-*]\s|–––)')


def is_apparatus(text: str) -> str | None:
    """Citation-list detector (175 pattern, precision-first)."""
    t = text.strip()
    if not _ITEM_RE.match(t):
        return None
    lines = [l for l in t.split("\n") if l.strip()]
    items = sum(1 for l in lines if _ITEM_RE.match(l))
    if items >= 3 and len(_CITE_RE.findall(t)) >= 3:
        return "bibliography"
    return None


_TITLE_NOISE = re.compile(
    r"(\s*[|–-]\s*(OHCHR|UN Human Rights|European Union Agency for Fundamental"
    r" Rights|CNIL|CNCDH|Défenseur des Droits|Bundesverfassungsgericht|"
    r"Antidiskriminierungsstelle)\s*)+$", re.I)


def clean_title(passages, fallback: str) -> str:
    if passages:
        first = passages[0].text.lstrip()
        if first.startswith("# "):
            h1 = first.split("\n", 1)[0][2:].strip()
            if 2 <= len(h1) <= 200:
                return h1
    t = _TITLE_NOISE.sub("", (fallback or "").strip())
    return t or fallback


# --- robots: wildcard-capable, Google-style longest-match; ALL groups that
# --- bind us are MERGED (ohchr publishes two `*` groups — a Cloudflare-managed
# --- one and its own Drupal one; both must apply). Improvement over the 176
# --- parser, which took the first matching group only.
class Robots:
    def __init__(self, text: str | None):
        self.rules: list[tuple[bool, re.Pattern, int]] = []
        self.delay: float | None = None
        self.broken = False           # robots endpoint served HTML (JS shell)
        if text is None:
            return
        if text.lstrip()[:1] == "<":
            self.broken = True
            return
        groups, agents, rules, delay, last_ua = [], [], [], None, False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            m = re.match(r"(?i)(user-agent|disallow|allow|crawl-delay)\s*:\s*(.*)",
                         line)
            if not m:
                continue
            key, val = m.group(1).lower(), m.group(2).strip()
            if key == "user-agent":
                if not last_ua and agents:
                    groups.append((agents, rules, delay))
                    agents, rules, delay = [], [], None
                agents.append(val.lower())
                last_ua = True
            else:
                last_ua = False
                if key == "disallow" and val:
                    rules.append((False, val))
                elif key == "allow" and val:
                    rules.append((True, val))
                elif key == "crawl-delay":
                    try:
                        delay = float(val)
                    except ValueError:
                        pass
        if agents:
            groups.append((agents, rules, delay))
        specific = [(rs, d) for ags, rs, d in groups
                    if any(a != "*" and a in UA_TOKEN for a in ags)]
        chosen = specific or [(rs, d) for ags, rs, d in groups if "*" in ags]
        for rs, d in chosen:                       # merge all binding groups
            if d is not None:
                self.delay = max(self.delay or 0.0, d)
            for allow, pat in rs:
                rx = re.escape(pat).replace(r"\*", ".*")
                if rx.endswith(r"\$"):
                    rx = rx[:-2] + "$"
                self.rules.append((allow, re.compile("^" + rx), len(pat)))

    def allowed(self, path_q: str) -> bool:
        best, best_len = None, -1
        for allow, rx, plen in self.rules:
            if rx.match(path_q):
                if plen > best_len or (plen == best_len and allow):
                    best, best_len = allow, plen
        return True if best is None else best


def fetch_robots(client: httpx.Client, host: str) -> Robots | None:
    """None => host must be skipped (5xx / JS shell). Empty Robots => allow."""
    try:
        r = client.get(f"https://{host}/robots.txt")
        if r.status_code >= 500:
            return None
        if r.status_code >= 400:
            return Robots("")          # no robots -> allow-all
        rb = Robots(r.text)
        return None if rb.broken else rb
    except Exception:  # noqa: BLE001
        return None


# --- sitemap walker ------------------------------------------------------------
LOC_RE = re.compile(r"<loc>\s*([^<\s]+?)\s*</loc>", re.I)


def sitemap_urls(client: httpx.Client, url: str, delay: float = 1.0,
                 depth: int = 0) -> list[str]:
    try:
        r = client.get(url)
        if r.status_code != 200:
            return []
        body = r.content
        if url.endswith(".gz") or body[:2] == b"\x1f\x8b":
            body = gzip.decompress(body)
        text = body.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return []
    locs = [l.replace("&amp;", "&") for l in LOC_RE.findall(text)]
    if "<sitemapindex" in text[:2000]:
        out: list[str] = []
        if depth < 3:
            for l in locs:
                time.sleep(delay)          # crawl-delay applies to sitemaps too
                out += sitemap_urls(client, l, delay, depth + 1)
        return out
    return locs


# --- URL helpers ----------------------------------------------------------------
HREF_RE = re.compile(r'href="([^"#]+)')

ALIAS = {"ohchr.org": "www.ohchr.org",
         "www.fra.europa.eu": "fra.europa.eu",
         "cncdh.fr": "www.cncdh.fr",
         "defenseurdesdroits.fr": "www.defenseurdesdroits.fr",
         "default": "www.defenseurdesdroits.fr",   # DDD sitemap base-url bug
         "cnil.fr": "www.cnil.fr",
         "bundesverfassungsgericht.de": "www.bundesverfassungsgericht.de",
         "antidiskriminierungsstelle.de": "www.antidiskriminierungsstelle.de"}


canon = functools.partial(https_frontier_url, alias=ALIAS)


def page_links(base_url: str, html: str, host: str) -> list[str]:
    out = []
    for h in HREF_RE.findall(html):
        u = urllib.parse.urljoin(base_url, h.strip())
        if not u.startswith("http"):
            continue
        cu = canon(u)
        if not cu or urllib.parse.urlsplit(cu).netloc != host:
            continue
        if BINARY_RE.search(cu):
            continue
        out.append(cu)
    return out


def _path(u: str) -> str:
    return urllib.parse.urlsplit(u).path or "/"


def _seg(u: str, i: int) -> str:
    parts = [p for p in _path(u).split("/") if p]
    return parts[i] if len(parts) > i else ""


# --- host configs ----------------------------------------------------------------

# ohchr: DEPTH-1 EN+FR pages ONLY. Measured live (2026-07-12, from two datacenter
# vantages, curl AND httpx, honest UA): Cloudflare serves depth-1 pages
# (/en/universal-declaration-of-human-rights, /en/what-are-human-rights,
# /en/treaty-bodies, hubs) with 200 but CHALLENGES (403) virtually every
# depth-2+ path (instruments/*, countries/*, press-releases/*, hr-bodies/*)
# for non-verified bots — despite the robots Content-Signal search=yes grant.
# The legal grant and the WAF config disagree; we take what is politely
# fetchable (~623 depth-1 pages, thin session shells dropped by gates
# 167/184) and leave the deep canon for a follow-up route (e.g. Cloudflare
# verified-bots registration).
_OHCHR_D1 = re.compile(r"^/(en|fr)/[^/]+/?$")
_OHCHR_SHELL = re.compile(r"(?i)/\d+(st|nd|rd|th)-(special-)?session")


def ohchr_include(u: str) -> bool:
    return bool(_OHCHR_D1.match(_path(u)))


def ohchr_prio(u: str) -> int:
    return 1 if _OHCHR_SHELL.search(_path(u)) else 0


FRA_SKIP = {"video", "event", "call-for-tender", "job", "vacancies"}
FRA_P0 = {"publication", "law-reference", "caselaw-reference", "content",
          "promising-practices", "country-data", "themes"}


def fra_include(u: str) -> bool:
    return _seg(u, 0) in ("en", "fr", "de") and _seg(u, 1) not in FRA_SKIP


def fra_prio(u: str) -> int:
    seg = _seg(u, 1)
    if seg in FRA_P0 or not seg:
        return 0
    if seg == "news":
        return 2
    return 1


CNIL_KW = re.compile(
    r"(?i)(intelligence-artificielle|/ia-|-ia/|/ai-|sanction|rgpd|"
    r"droits|libertes|donnees-personnelles|biometri|reconnaissance-faciale|"
    r"videosurveillance|algorithm)")

CNCDH_SKIP = re.compile(
    r"(?i)/(user|search|node|form|ajax|media|file|taxonomy|print)(/|$)")

# BVerfG: landmark decisions (curated, fetch-verified; full decision corpus is
# already in the index via rechtsprechung-im-internet.de — this pins the
# court's own canonical pages for the leading cases incl. the df-011 gold).
BVERFG_LANDMARKS = [
    "https://www.bundesverfassungsgericht.de/SharedDocs/Entscheidungen/DE/2021/03/rs20210324_1bvr265618.html",   # Klimabeschluss
    "https://www.bundesverfassungsgericht.de/SharedDocs/Entscheidungen/EN/2021/03/rs20210324_1bvr265618en.html",
    "https://www.bundesverfassungsgericht.de/SharedDocs/Entscheidungen/DE/2019/11/rs20191106_1bvr001613.html",   # Recht auf Vergessen I
    "https://www.bundesverfassungsgericht.de/SharedDocs/Entscheidungen/DE/2019/11/rs20191106_1bvr027617.html",   # Recht auf Vergessen II
    "https://www.bundesverfassungsgericht.de/SharedDocs/Entscheidungen/DE/2020/05/rs20200519_1bvr283517.html",   # BND
    "https://www.bundesverfassungsgericht.de/SharedDocs/Entscheidungen/DE/2023/02/rs20230216_1bvr154719.html",   # automatisierte Datenanalyse
    "https://www.bundesverfassungsgericht.de/SharedDocs/Entscheidungen/DE/2017/01/bs20170117_2bvb000113.html",   # NPD
    "https://www.bundesverfassungsgericht.de/SharedDocs/Entscheidungen/DE/2019/01/cs20190129_2bvc006214.html",   # Wahlrechtsausschluss
    "https://www.bundesverfassungsgericht.de/SharedDocs/Entscheidungen/DE/2020/02/rs20200226_2bvr234715.html",   # Suizidhilfe
]


def bverfg_allow(u: str) -> bool:
    p = _path(u)
    if re.match(r"^/SharedDocs/(Entscheidungen|Pressemitteilungen)/(DE|EN)/", p):
        return True
    return bool(re.match(r"^/(DE|EN)/", p)) and "/Service/" not in p


def bverfg_lang(u: str) -> str:
    p = _path(u)
    return "en" if ("/EN/" in p or p.endswith("en.html")) else "de"


HOSTS: dict[str, dict] = {
    "ohchr": dict(host="www.ohchr.org", mode="sitemap",
                  # NOTE: /sitemap.xml (bare) is Cloudflare-challenged; the
                  # /en/sitemap.xml index + ?page=N urlsets serve 200.
                  sitemaps=["https://www.ohchr.org/en/sitemap.xml?page=1",
                            "https://www.ohchr.org/en/sitemap.xml?page=2",
                            "https://www.ohchr.org/en/sitemap.xml?page=3"],
                  include=ohchr_include, prio=ohchr_prio,
                  lang=lambda u: "fr" if _seg(u, 0) == "fr" else "en",
                  delay=2.0, cap=700, source_type="official", trust=0.95,
                  license="UN © (re-read ok) — robots Content-Signal: "
                          "search=yes, ai-train=no → train_ok=false"),
    "fra": dict(host="fra.europa.eu", mode="sitemap",
                sitemaps=[f"https://fra.europa.eu/sitemap.xml?page={i}"
                          for i in range(1, 8)],
                include=fra_include, prio=fra_prio,
                lang=lambda u: _seg(u, 0) if _seg(u, 0) in
                ("en", "fr", "de") else "en",
                delay=2.0, cap=9000, source_type="official", trust=0.95),
    "cncdh": dict(host="www.cncdh.fr", mode="bfs",
                  seeds=["https://www.cncdh.fr/"],
                  allow=lambda u: _path(u) != "/" and
                  not CNCDH_SKIP.search(_path(u)),
                  hub=lambda u: not CNCDH_SKIP.search(_path(u)), hub_depth=6,
                  lang=lambda u: "fr", delay=2.0, cap=2000,
                  source_type="official", trust=0.9),
    "ddd": dict(host="www.defenseurdesdroits.fr", mode="sitemap",
                sitemaps=["https://www.defenseurdesdroits.fr/sitemap.xml"],
                include=lambda u: True, prio=lambda u: 0,
                lang=lambda u: "en" if _seg(u, 0) == "en" else "fr",
                delay=2.0, cap=1500, source_type="official", trust=0.9),
    "cnil": dict(host="www.cnil.fr", mode="sitemap",
                 sitemaps=["https://www.cnil.fr/fr/sitemap.xml",
                           "https://www.cnil.fr/en/sitemap.xml"],
                 include=lambda u: bool(CNIL_KW.search(_path(u))),
                 prio=lambda u: 0,
                 lang=lambda u: "en" if _seg(u, 0) == "en" else "fr",
                 delay=2.0, cap=900, source_type="official", trust=0.9),
    "bverfg": dict(host="www.bundesverfassungsgericht.de", mode="sitemap",
                   sitemaps=["https://www.bundesverfassungsgericht.de/Sitemap_Index.xml"],
                   include=lambda u: bverfg_allow(u), prio=lambda u: 1,
                   seeds=(BVERFG_LANDMARKS +
                          ["https://www.bundesverfassungsgericht.de/DE/Homepage/homepage_node.html",
                           "https://www.bundesverfassungsgericht.de/DE/Entscheidungen/entscheidungen_node.html",
                           "https://www.bundesverfassungsgericht.de/DE/Presse/presse_node.html"]),
                   expand=True, allow=bverfg_allow,
                   hub=lambda u: bool(re.match(r"^/(DE|EN)/", _path(u))),
                   hub_depth=2,
                   lang=bverfg_lang, delay=30.0, cap=450,
                   source_type="official", trust=1.0,
                   license="§ 5 UrhG gemeinfrei (official works)"),
    "ads": dict(host="www.antidiskriminierungsstelle.de", mode="sitemap",
                sitemaps=["https://www.antidiskriminierungsstelle.de/Sitemap_Index.xml"],
                include=lambda u: not BINARY_RE.search(u),
                # basepage explainer content first, press releases next,
                # download-landing pages last (1,501 inventory > cap)
                prio=lambda u: (2 if "/SharedDocs/downloads/" in _path(u).lower()
                                else 1 if "/SharedDocs/" in _path(u)
                                else 3 if "/service/" in _path(u).lower()
                                else 0),
                lang=lambda u: "en" if "/EN/" in u else "de",
                delay=30.0, cap=900, source_type="official", trust=0.9),
}

SKIPPED_HOSTS = {
    "institut-fuer-menschenrechte.de": "DIMR 403s datacenter IPs → CC route later (185)",
    "echr.coe.int": "WAF-403 datacenter (HUDOC API used instead)",
    "coe.int / rm.coe.int": "WAF-403 datacenter → CC route later",
    "equalityhumanrights.com": "EHRC WAF-403 → CC route later",
}

KIND_PRIO = {"content": 0, "cand": 1, "list": 2, "hub": 3}
LAST_SEEN = dt.date.today().isoformat()


# --- HUDOC leading-case-law slice (JSON API; ctgov one-study-one-record) -------

HUDOC_API = "https://hudoc.echr.coe.int/app/query/results"
HUDOC_SELECT = ("itemid,docname,appno,importance,kpdate,languageisocode,"
                "doctypebranch,article,conclusion,respondent")
# doctype:HEJUD = judgments (English text); HFJUD = arrêts (French text).
# The API's OR operator is unreliable (GC OR importance:1 returns the GC count
# only — verified live) → separate queries, client-side union per lang.
# importance:2 is included because the pre-Protocol-11 landmark canon is
# labelled CHAMBER/importance-2 in HUDOC (e.g. Handyside 5493/72, the df-004
# gold) — GC + importance 1 alone misses the old plenary-era leading cases.
HUDOC_QUERIES = [
    ("en", "contentsitename:ECHR AND (doctypebranch:GRANDCHAMBER) AND (doctype:HEJUD)"),
    ("en", "contentsitename:ECHR AND (importance:1) AND (doctype:HEJUD)"),
    ("en", "contentsitename:ECHR AND (importance:2) AND (doctype:HEJUD)"),
    ("fr", "contentsitename:ECHR AND (doctypebranch:GRANDCHAMBER) AND (doctype:HFJUD)"),
    ("fr", "contentsitename:ECHR AND (importance:1) AND (doctype:HFJUD)"),
    ("fr", "contentsitename:ECHR AND (importance:2) AND (doctype:HFJUD)"),
]
HUDOC_PAGE = 500
HUDOC_DELAY = 2.0


def _hudoc_get(client: httpx.Client, query: str, start: int, length: int):
    params = {"query": query, "select": HUDOC_SELECT, "sort": "",
              "start": str(start), "length": str(length)}
    for attempt in range(4):
        try:
            r = client.get(HUDOC_API, params=params)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:  # noqa: BLE001 — WAF/static-404 interstitial
                    pass
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5.0 * (attempt + 1))
    return None


def hudoc_record(c: dict, lang: str, itemid: str, detect, last_seen: str) -> dict | None:
    """One HUDOC result row -> the landing/metadata record (None = language gate dropped it)."""
    docname = (c.get("docname") or "").strip()
    appno = (c.get("appno") or "").replace(";", "; ")
    date = (c.get("kpdate") or "")[:10]
    arts = (c.get("article") or "").replace(";", ", ")
    concl = (c.get("conclusion") or "").strip()
    resp = (c.get("respondent") or "").replace(";", ", ")
    branch = (c.get("doctypebranch") or "").title()
    if lang == "en":
        text = (f"{docname}. European Court of Human Rights "
                f"{branch or 'Chamber'} judgment, {date}. "
                f"Application no. {appno}. "
                f"ECHR Articles: {arts}. Respondent state: {resp}. "
                f"Conclusion: {concl}")
        url = f"https://hudoc.echr.coe.int/eng?i={itemid}"
    else:
        text = (f"{docname}. Arrêt de la Cour européenne des "
                f"droits de l'homme ({branch or 'chambre'}), "
                f"{date}. Requête no {appno}. "
                f"Articles CEDH: {arts}. État défendeur: {resp}. "
                f"Conclusion: {concl}")
        url = f"https://hudoc.echr.coe.int/fre?i={itemid}"
    dl, conf = detect(text, url, lang)
    if conf and dl not in ("en", "de", "fr", "it"):
        return None
    words = (concl or text).split()
    snip = " ".join(words[:25]) + ("…" if len(words) > 25 else "")
    p = Passage(url=url, title=docname, snippet=snip, text=text,
                lang=lang, domain="hudoc.echr.coe.int", tier="A",
                source_type="case_law", trust_tier=1.0,
                last_seen=last_seen)
    rec = asdict(p)
    rec.pop("content_digest", None)
    rec["provenance"] = PROVENANCE
    rec["license"] = "© Council of Europe / ECtHR (pointer+snippet only)"
    rec["itemid"] = itemid
    rec["importance"] = c.get("importance")
    rec["doctypebranch"] = c.get("doctypebranch")
    rec["kpdate"] = date
    return rec


def harvest_hudoc(client: httpx.Client, pf, state: dict, detect) -> None:
    if state.get("hudoc_done"):
        print("[hudoc] already done — skipping", flush=True)
        return
    seen: set[tuple[str, str]] = set()
    n_rec, n_langdrop = 0, 0
    for lang, q in HUDOC_QUERIES:
        start, total = 0, None
        while total is None or start < total:
            data = _hudoc_get(client, q, start, HUDOC_PAGE)
            if data is None:
                raise RuntimeError(f"hudoc query failed: {q} @ {start}")
            total = int(data.get("resultcount", 0))
            rows = data.get("results", [])
            if not rows:
                break
            for row in rows:
                c = row.get("columns", {})
                itemid = c.get("itemid", "")
                if not itemid or (lang, itemid) in seen:
                    continue
                seen.add((lang, itemid))
                rec = hudoc_record(c, lang, itemid, detect, LAST_SEEN)
                if rec is None:
                    n_langdrop += 1
                    continue
                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_rec += 1
            pf.flush()
            start += HUDOC_PAGE
            print(f"[hudoc] {lang} '{q[:60]}…' {min(start, total)}/{total} "
                  f"(records so far {n_rec})", flush=True)
            time.sleep(HUDOC_DELAY)
    state["hudoc_records"] = n_rec
    state["hudoc_langdrop"] = n_langdrop
    state["hudoc_done"] = True
    print(f"[hudoc] DONE: {n_rec} records ({n_langdrop} lang-dropped)",
          flush=True)


# --- EUR-Lex Charter (proven 145 route: legal-content HTML, article-level) -----

CHARTER_CELEX = "12012P/TXT"
CHARTER_LANGS = {"EN": ("en", "Charter of Fundamental Rights of the European Union"),
                 "FR": ("fr", "Charte des droits fondamentaux de l'Union européenne"),
                 "DE": ("de", "Charta der Grundrechte der Europäischen Union")}


def _charter_sections(html: str) -> list[tuple[str, str, str]]:
    """[(anchor, heading, body)] — tries the modern eli-subdivision layout,
    then the older OJ ti-art layout (12012P renders in the old format)."""
    from lxml import html as LH
    if isinstance(html, str):        # lxml rejects str with an XML decl
        html = html.encode("utf-8", "replace")
    doc = LH.fromstring(html)
    out = []
    # layout 1: consolidated/ELI divs
    for div in doc.xpath("//div[contains(@class,'eli-subdivision') and @id]"):
        sid = div.get("id", "")
        if re.match(r"^art_\d+$", sid):
            txt = re.sub(r"\s+", " ", div.text_content()).strip()
            head = txt[:120]
            out.append((sid, head, txt))
    if out:
        return out
    # layout 2: old OJ format — p.ti-art "Article N" + p.sti-art subtitle
    cur = None
    for el in doc.iter():
        cls = (el.get("class") or "") if el.tag == "p" else ""
        if "ti-art" in cls and "sti-art" not in cls:
            if cur and cur[2].strip():
                out.append(tuple(cur))
            artno = re.sub(r"\s+", " ", el.text_content()).strip()
            m = re.search(r"(\d+)", artno)
            cur = [f"art_{m.group(1)}" if m else artno.replace(" ", "_"),
                   artno, ""]
        elif cur is not None and el.tag == "p":
            t = re.sub(r"\s+", " ", el.text_content()).strip()
            if "sti-art" in cls:
                cur[1] = f"{cur[1]} — {t}"
            elif t:
                cur[2] += t + " "
    if cur and cur[2].strip():
        out.append(tuple(cur))
    return out


def charter_url(L: str) -> str:
    return f"https://eur-lex.europa.eu/legal-content/{L}/TXT/HTML/?uri=CELEX:{CHARTER_CELEX}"


def charter_records(L: str, secs, last_seen: str) -> list[dict]:
    """The article-level records of one Charter language from its parsed sections."""
    code, label = CHARTER_LANGS[L]
    base = f"https://eur-lex.europa.eu/legal-content/{L}/TXT/?uri=CELEX:{CHARTER_CELEX}"
    out = []
    for sid, head, body in secs:
        body = body.strip()
        if len(body) < 40:
            continue
        words = body.split()
        snip = " ".join(words[:25]) + ("…" if len(words) > 25 else "")
        title = f"{label} — {head.strip()}"
        p = Passage(url=base, title=title[:220], snippet=snip,
                    text=f"{title}. {body}"[:4000], lang=code,
                    domain="eur-lex.europa.eu", tier="A",
                    source_type="official", trust_tier=0.95,
                    anchor=sid, last_seen=last_seen)
        rec = asdict(p)
        rec.pop("content_digest", None)
        rec["provenance"] = PROVENANCE
        rec["celex"] = "12012P"
        out.append(rec)
    return out


def harvest_eurlex_charter(client: httpx.Client, pf, state: dict) -> None:
    if state.get("eurlex_done"):
        print("[eurlex] already done — skipping", flush=True)
        return
    n = 0
    for L in CHARTER_LANGS:
        r = client.get(charter_url(L))
        if r.status_code != 200:
            raise RuntimeError(f"eurlex charter {L}: HTTP {r.status_code}")
        secs = _charter_sections(r.text)
        if len(secs) < 40:                     # Charter has 54 articles
            raise RuntimeError(f"eurlex charter {L}: only {len(secs)} sections "
                               "parsed — layout changed, refusing to ship junk")
        for rec in charter_records(L, secs, LAST_SEEN):
            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
        pf.flush()
        print(f"[eurlex] charter {L}: {len(secs)} sections", flush=True)
        time.sleep(3.0)
    state["eurlex_records"] = n
    state["eurlex_done"] = True
    print(f"[eurlex] DONE: {n} article records", flush=True)


def page_records(url: str, html: str, cfg: dict, last_seen: str,
                 max_chunks: int = 50) -> tuple[list[dict], dict]:
    """One crawled page -> its admitted record dicts and the counters the run state keeps.

    `cfg` is a `HOSTS` entry (lang, source_type, trust, host, optional license). The extraction
    (gates 097/166/167/184), the citation-list filter, the head-cap and the title clean are the
    driver's own, unchanged."""
    dom = re.sub(r"^www\d*\.", "", urllib.parse.urlsplit(url).netloc.lower()) or \
        re.sub(r"^www\d*\.", "", cfg["host"])
    meta = {"content": True, "links": False, "want_stats": True,
            "lang_default": cfg["lang"](url),
            "passage_common": {
                "domain": dom,
                "tier": "A", "source_type": cfg["source_type"],
                "trust_tier": cfg["trust"], "last_seen": last_seen}}
    passages, _links, stats = process_html(url, html, meta)
    counts = {"rejects": dict(stats.get("extraction_rejects", {})), "apparatus": {}, "capped": 0}
    kept = []
    for p in passages:
        reason = is_apparatus(p.text)
        if reason:
            counts["apparatus"][reason] = counts["apparatus"].get(reason, 0) + 1
        else:
            kept.append(p)
    if len(kept) > max_chunks:
        counts["capped"] = len(kept) - max_chunks
        kept = kept[:max_chunks]
    if not kept:
        return [], counts
    title = clean_title(kept, kept[0].title)
    rows = []
    for p in kept:
        p.title = title
        rec = asdict(p)
        rec.pop("content_digest", None)
        rec["provenance"] = PROVENANCE
        if cfg.get("license"):
            rec["license"] = cfg["license"]
        rows.append(rec)
    return rows, counts


# --- state ----------------------------------------------------------------------
def load_state(path: str) -> dict:
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:  # noqa: BLE001
            pass
    return {"done": [], "queued": {}, "started_at":
            dt.datetime.utcnow().isoformat() + "Z",
            "hosts": {}, "passages": 0, "apparatus": {}, "capped": 0,
            "rejects": {}, "fetches": 0}


def save_state(path: str, state: dict):
    tmp = path + ".tmp"
    json.dump(state, open(tmp, "w"), ensure_ascii=False)
    os.replace(tmp, path)


def hstat(state: dict, key: str) -> dict:
    return state["hosts"].setdefault(key, {
        "pages_ok": 0, "empty": 0, "error": 0, "robots_skipped": 0,
        "not_scope": 0, "passages": 0, "fetched": 0})


def host_inventory(client: httpx.Client, key: str, cfg: dict, inv_dir: str) -> list[str]:
    """The ordered URL inventory of a sitemap host: the committed `<inv_dir>/<key>.txt` when it
    exists, else built from the host's sitemaps (canonical, in scope, same host, no binaries,
    priority order) and written there."""
    inv_path = os.path.join(inv_dir, f"{key}.txt")
    if os.path.exists(inv_path):
        return [l.strip() for l in open(inv_path) if l.strip()]
    raw = []
    for sm in cfg["sitemaps"]:
        raw += sitemap_urls(client, sm, delay=cfg["eff_delay"])
    urls = sorted({canon(u) for u in raw
                   if u and cfg["include"](canon(u))} - {""})
    urls = [u for u in urls
            if urllib.parse.urlsplit(u).netloc == cfg["host"]
            and not BINARY_RE.search(u)]
    urls.sort(key=cfg.get("prio", lambda u: 0))
    os.makedirs(inv_dir, exist_ok=True)
    with open(inv_path, "w") as f:
        f.write("\n".join(urls))
    print(f"[{key}] inventory built: {len(urls)} urls "
          f"(from {len(raw)} sitemap locs)", flush=True)
    return urls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="depth187")
    ap.add_argument("--max-chunks", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N NEW fetches this run (smoke test)")
    ap.add_argument("--only", default="",
                    help="comma-separated host keys or 'hudoc'/'eurlex' (smoke)")
    ap.add_argument("--budget", type=int, default=45000,
                    help="hard total-fetch stop (safety; task cap ≤50k)")
    ap.add_argument("--skip-api", action="store_true",
                    help="skip the hudoc+eurlex stages (crawl only)")
    args = ap.parse_args()

    require_identified()
    D = args.out_dir
    os.makedirs(os.path.join(D, "inv"), exist_ok=True)
    # harvest every fetched page's external links for the
    # endorsement graph (streamed to <out-dir>/outlinks.jsonl; passage
    # extraction unchanged).
    enable_outlink_capture(D)
    passages_path = os.path.join(D, "passages.jsonl")
    state_path = os.path.join(D, "state.json")

    client = make_client(HttpSpec(timeout=45, headers={"user-agent": headers()["User-Agent"]},
                             follow_redirects=True))

    only = [s for s in args.only.split(",") if s] if args.only else []
    hosts = {k: v for k, v in HOSTS.items() if not only or k in only}

    state = load_state(state_path)
    done: set[str] = set(state["done"])
    pf = open(passages_path, "a", encoding="utf-8")

    # --- API/bulk stages first (fast, resumable-by-flag) ----------------------
    from cytir.acquire.fetch_process import detect_lang
    if not args.skip_api and (not only or "hudoc" in only):
        harvest_hudoc(client, pf, state, detect_lang)
        save_state(state_path, state)
    if not args.skip_api and (not only or "eurlex" in only):
        harvest_eurlex_charter(client, pf, state)
        save_state(state_path, state)
    hosts = {k: v for k, v in hosts.items() if k in HOSTS}

    # --- robots per host (live, before any content fetch) --------------------
    robots: dict[str, Robots] = {}
    active: dict[str, dict] = {}
    for key, cfg in hosts.items():
        rb = fetch_robots(client, cfg["host"])
        if rb is None:
            print(f"[{key}] robots UNAVAILABLE/JS-shell -> HOST SKIPPED", flush=True)
            hstat(state, key)["robots_skipped"] += 1
            continue
        eff = max(cfg["delay"], rb.delay or 0.0)
        robots[key] = rb
        cfg = dict(cfg, eff_delay=eff)
        active[key] = cfg
        print(f"[{key}] robots ok  crawl-delay={rb.delay}  eff_delay={eff}s",
              flush=True)
        time.sleep(0.5)

    # --- queues ---------------------------------------------------------------
    queues: dict[str, list[list]] = {}
    qseen: dict[str, set[str]] = {}

    def enq(key: str, url: str, kind: str, depth: int = 0):
        if not url or url in done or url in qseen[key]:
            return
        if BINARY_RE.search(url):
            return
        qseen[key].add(url)
        queues[key].append([url, kind, depth])

    for key, cfg in active.items():
        queues[key], qseen[key] = [], set()
        if cfg["mode"] == "sitemap":
            urls = host_inventory(client, key, cfg, os.path.join(D, "inv"))
            for u in urls[:cfg["cap"]]:
                enq(key, u, "content")
            for s in cfg.get("seeds", []):
                enq(key, canon(s), "hub")
        elif cfg["mode"] == "bfs":
            for s in cfg["seeds"]:
                enq(key, canon(s), "hub")
        for entry in state["queued"].get(key, []):
            enq(key, entry[0], entry[1], entry[2] if len(entry) > 2 else 0)
        if not queues[key] and cfg["mode"] == "bfs":
            for s in cfg.get("seeds", []):
                u = canon(s)
                done.discard(u)
                enq(key, u, "hub")
            print(f"[{key}] empty queue -> re-seeded ({len(queues[key])})",
                  flush=True)

    next_ok = {k: 0.0 for k in active}
    new_fetches = 0
    t0 = time.time()

    def persist():
        state["done"] = list(done)
        merged = dict(state.get("queued", {}))
        for k in active:
            merged[k] = queues[k]
        state["queued"] = {k: q for k, q in merged.items() if q}
        save_state(state_path, state)

    def index_page(key: str, cfg: dict, url: str, html: str) -> int:
        st = hstat(state, key)
        kept, counts = page_records(url, html, cfg, LAST_SEEN, args.max_chunks)
        for r, n in counts["rejects"].items():
            state["rejects"][r] = state["rejects"].get(r, 0) + n
        for reason, n in counts["apparatus"].items():
            state["apparatus"][reason] = state["apparatus"].get(reason, 0) + n
        state["capped"] += counts["capped"]
        if not kept:
            st["empty"] += 1
            return 0
        for rec in kept:
            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pf.flush()
        st["pages_ok"] += 1
        st["passages"] += len(kept)
        state["passages"] += len(kept)
        return len(kept)

    while any(queues.values()):
        if args.limit and new_fetches >= args.limit:
            print(f"--limit {args.limit} reached, stopping (resumable)",
                  flush=True)
            break
        if state["fetches"] >= args.budget:
            print("FETCH BUDGET reached — stopping", flush=True)
            break
        now = time.time()
        ready = [k for k, q in queues.items() if q and next_ok[k] <= now]
        if not ready:
            nxt = min(next_ok[k] for k, q in queues.items() if q)
            time.sleep(max(0.05, min(nxt - now, 5.0)))
            continue
        key = min(ready, key=lambda k: next_ok[k])
        cfg = active[key]
        st = hstat(state, key)
        if st["fetched"] >= cfg["cap"]:
            print(f"[{key}] cap {cfg['cap']} reached — draining queue "
                  f"({len(queues[key])} left)", flush=True)
            queues[key] = []
            continue
        qi = min(range(len(queues[key])),
                 key=lambda i: (KIND_PRIO.get(queues[key][i][1], 0), i))
        url, kind, depth = queues[key].pop(qi)
        if url in done:
            continue
        pq = urllib.parse.urlsplit(url)
        if not robots[key].allowed(
                pq.path + (("?" + pq.query) if pq.query else "")):
            st["robots_skipped"] += 1
            done.add(url)
            continue
        next_ok[key] = time.time() + cfg["eff_delay"]
        new_fetches += 1
        state["fetches"] += 1
        try:
            r = client.get(url)
            final_host = urllib.parse.urlsplit(str(r.url)).netloc.lower()
            ctype = r.headers.get("content-type", "").lower()
            ok_hosts = cfg.get("ok_hosts", {cfg["host"]})
            if r.status_code != 200 or "html" not in ctype or \
                    ALIAS.get(final_host, final_host) not in ok_hosts:
                st["error"] += 1
            else:
                html = r.text
                allow = cfg.get("allow")
                indexable = allow is None or allow(url)
                if indexable:
                    index_page(key, cfg, str(r.url).split("#")[0], html)
                else:
                    st["not_scope"] += 1
                if cfg["mode"] == "bfs" or cfg.get("expand"):
                    hub = cfg.get("hub")
                    for l in page_links(url, html, cfg["host"]):
                        if allow is None or allow(l):
                            enq(key, l, "content", depth + 1)
                        elif hub and hub(l) and \
                                depth + 1 <= cfg.get("hub_depth", 0):
                            enq(key, l, "hub", depth + 1)
            st["fetched"] += 1
        except Exception as e:  # noqa: BLE001
            st["error"] += 1
            print(f"  ! {url}: {e}", file=sys.stderr, flush=True)
        done.add(url)
        if new_fetches % 50 == 0:
            persist()
            rate = new_fetches / max(time.time() - t0, 1)
            qs = {k: len(q) for k, q in queues.items() if q}
            print(f"  {new_fetches} fetched this run  passages={state['passages']}"
                  f"  {rate*3600:.0f}/h  queues={qs}", flush=True)

    persist()
    pf.close()
    print("CRAWL DONE" if not any(queues.values()) else "CRAWL PAUSED",
          flush=True)
    for key in sorted(state["hosts"]):
        print(f"  {key}: {json.dumps(state['hosts'][key])}", flush=True)
    print(f"  hudoc={state.get('hudoc_records')} "
          f"eurlex={state.get('eurlex_records')} "
          f"total passages={state['passages']} rejects={state['rejects']} "
          f"apparatus={state['apparatus']} capped={state['capped']}", flush=True)


if __name__ == "__main__":
    main()
