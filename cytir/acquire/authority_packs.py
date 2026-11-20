#!/usr/bin/env python3
"""Task 399 — leading-authority-domain packs (biomedical, energy/climate,
education, urban governance).

Ordered by Tim 2026-08-07 ("add biomedical + the other leading domains you've
identified"); evidence base the task-395 theme-gap mining, which localised
four gaps by live spot-check, and the task-399 census-before artifact,
which confirmed them mechanically against the serving corpus (ipcc.ch = 3
points, oecd.org = 4, IEA/IRENA/UNEP/IAEA/EEF/WWC/city-networks = exact zero).

INVENTORY ADAPTER ONLY. Everything that is not the per-host inventory decision
comes from the shared layer `cytir.acquire.crawl` (ADR-026): robots
verification + per-host politeness, the crawl loop, resume/state, L0 raw-HTML
retention, outlink capture, fetch log, budgets and caps. Extraction and its
gates (097 junk / 166 language / 167) come from `aw_common.process_html`.
Chunking resolves from the tracked build profile `data/build_config.json`
(currently mode=legacy, epoch 1, `legacy-180w-o30@1`) — this wave deliberately
lands in the CURRENT production epoch and migrates with everything else at the
epoch-3 flip, so it must NOT be run under an AW_CHUNK_POLICY override.

Routes are what live recon could VERIFY (the task-399 recon script,
artifact recon.json of task 399). Hosts that fail closed are listed
in SKIPPED with the reason and are NOT worked around — no UA spoofing, per the
task-351 precedent.

  python3 -m cytir.acquire.authority_packs --pack energy --out-dir /path/run

Task 1393 (S7): this module is the old authority-packs driver script, moved whole (the
old path is an alias shim). The four packs are also families on the Protocol
(`cytir.sources.families.authority_packs`), which read the same hosts from the pack YAML; this
module keeps the `run_pack` driver (L0 retention, resume, the ipcc BFS) and is the reference the
selection-equivalence test compares the YAML against.
"""
from __future__ import annotations

import argparse
import re
import sys

from cytir.acquire import crawl as pack_fetch
from cytir.acquire.crawl import _path

# ---------------------------------------------------------------------------
# Trust tiers. §4.6 health is the strictest floor by policy (harm asymmetry);
# IGO/official statistical publishers sit at the §4.1/§4.5 official level.
# Nothing here is a discovery-sourced admission, so nothing inherits the
# vendor-content default (POLICY §4.4 standing rule).
# ---------------------------------------------------------------------------
TRUST_IGO_PRIMARY = 0.95      # the treaty/assessment body itself (IPCC, WHO)
TRUST_OFFICIAL = 0.90         # official agency / regulator publications
TRUST_NETWORK = 0.75          # membership networks (ICLEI, Cities Alliance)


def _segs(u: str) -> list[str]:
    return [s for s in _path(u).strip("/").split("/") if s]


def _seg1(u: str) -> str:
    s = _segs(u)
    return s[0] if s else ""


# --- generic noise ----------------------------------------------------------
_GENERIC_DENY = re.compile(
    r"(?i)/(search|suche|recherche|ricerca|login|user|node/add|print|"
    r"newsletter|rss|feed|calendar|termine|contact|kontakt|sitemap|"
    r"privacy|disclaimer|copyright|cookie)(/|$|\?)")


def _clean(u: str) -> bool:
    return not _GENERIC_DENY.search(_path(u))


def _deep(u: str, n: int = 2) -> bool:
    """True when the URL is an ITEM, not a section landing page.

    Smoke run evidence (2026-08-07): `https://www.iea.org/analysis` — the bare
    section hub — extracted 84 passages of pure navigation furniture
    ("- Energy Security (233)", "- Climate Change (158)", …). Those are listing
    counters, not authority content, and the brief is explicit that this wave
    prioritises reports/publications/fact-sheets over navigation. Section
    landings are still CRAWLED (they carry the frontier) but not INDEXED.
    """
    return len(_segs(u)) >= n


# ===========================================================================
# PACK: energy & climate transition (international authority layer)
# The CH leg (the federal energy office) is already good per task 395 §5a — this pack is
# deliberately the INTERNATIONAL layer that spot-check found missing.
# ===========================================================================

# IPCC has NO sitemap (verified: every common location 404s) but its robots.txt
# permits everything except /site/wp-admin/, and content pages serve 200. So the
# route is a BFS from the report hubs. `/report/<slug>/...` is the crown-jewel
# content — the AR6 WG1 SPM alone extracts to 130 passages / 164k chars.
IPCC_ALLOW = re.compile(r"(?i)^/(report/[^/]+/.+|(sr15|srccl|srocc)/.+)$")
IPCC_HUB = re.compile(r"(?i)^/(report|reports|sr15|srccl|srocc)(/|$)")

# IEA: /data-and-statistics/charts is 7,084 of the 27,692 sitemap rows and is
# chart furniture, not prose — excluded on authority density, not on principle.
IEA_KEEP = {"reports", "analysis", "commentaries", "policies", "articles"}

# Umweltbundesamt: publications + the climate/energy topic tree. /en/node is a
# canonical-duplicate space and /service|/presse are announcements.
UBA_KEEP = re.compile(
    r"(?i)^/(en/publikationen|en/topics|themen/klima-energie|"
    r"publikationen|themen/nachhaltigkeit-strategien-internationales)(/|$)")

ENERGY_HOSTS: dict[str, dict] = {
    "ipcc": dict(host="www.ipcc.ch", mode="bfs",
                 seeds=["https://www.ipcc.ch/reports/",
                        "https://www.ipcc.ch/report/sixth-assessment-report-cycle/",
                        "https://www.ipcc.ch/report/ar6/syr/",
                        "https://www.ipcc.ch/report/ar6/wg1/",
                        "https://www.ipcc.ch/report/ar6/wg2/",
                        "https://www.ipcc.ch/report/ar6/wg3/",
                        "https://www.ipcc.ch/sr15/",
                        "https://www.ipcc.ch/srccl/",
                        "https://www.ipcc.ch/srocc/"],
                 allow=lambda u: bool(IPCC_ALLOW.match(_path(u))) and _clean(u),
                 hub=lambda u: bool(IPCC_HUB.match(_path(u))),
                 hub_depth=4,
                 lang=lambda u: "en", delay=2.0, cap=2500,
                 source_type="official", tier="A", trust=TRUST_IGO_PRIMARY,
                 license="ipcc-reports-terms"),
    "iea": dict(host="www.iea.org", mode="sitemap",
                sitemaps=["https://www.iea.org/sitemap.xml"],
                include=lambda u: (_seg1(u) in IEA_KEEP and _deep(u)
                                   and _clean(u)),
                lang=lambda u: "en", delay=2.0, cap=6000,
                source_type="official", tier="A", trust=TRUST_OFFICIAL,
                license="iea-terms"),
    "encom": dict(host="energy.ec.europa.eu", mode="sitemap",
                  sitemaps=["https://energy.ec.europa.eu/sitemap.xml"],
                  include=lambda u: _deep(u) and _clean(u),
                  lang=lambda u: "en", delay=2.0, cap=2200,
                  source_type="official", tier="A", trust=TRUST_OFFICIAL,
                  license="eu-reuse-2011-833"),
    # umweltbundesamt declares crawl-delay: 10 — at 4,000 pages that is 11
    # hours for the pack's lowest passage yield (1.0/page in the smoke run).
    # Capped to the climate/energy topic prose, which is the DE-language
    # authority content this pack actually needs.
    "uba": dict(host="www.umweltbundesamt.de", mode="sitemap",
                sitemaps=["https://www.umweltbundesamt.de/sitemap.xml"],
                include=lambda u: (bool(UBA_KEEP.match(_path(u))) and _deep(u)
                                   and _clean(u)),
                lang=lambda u: "en" if _path(u).startswith("/en/") else "de",
                delay=2.0, cap=700,
                source_type="official", tier="A", trust=TRUST_OFFICIAL,
                license="uba-terms"),
    "eia": dict(host="www.eia.gov", mode="sitemap",
                sitemaps=["https://www.eia.gov/sitemap.xml"],
                include=_clean,
                lang=lambda u: "en", delay=2.0, cap=400,
                source_type="official", tier="A", trust=TRUST_OFFICIAL,
                license="us-gov-public-domain"),
}

# ===========================================================================
# PACK: biomedical / health authority core  (POLICY §4.6 — strictest floor)
# ===========================================================================

# who.int enumerates 268,098 URLs; the authority mass is fact sheets,
# publications, guidance and the news-room Q&A, not the country/event furniture.
# Measured against the host's own declared index
# (http://www.who.int/sitemaps/sitemapindex.xml, 268,098 locs, 2026-08-07) —
# NOT assumed. The declared inventory is the LEGACY who.int tree, whose mass is
# /bulletin/volumes (29,316 — the peer-reviewed WHO Bulletin), /csr/don (20,812
# outbreak-news items), /mediacentre/news, /publications/m and /data/gho.
# An earlier guess at the modern /news-room/fact-sheets/... shape matched
# almost nothing in this index.
WHO_ALLOW = re.compile(
    r"(?i)^/(publications|bulletin|news-room|health-topics|initiatives|"
    r"teams|csr|mediacentre)(/|$)")

# cochrane.org canonicalises /CD<id>/<slug> -> /evidence/CD<id>_<canonical-slug>
# (verified 2026-08-07). Enumerate the canonical /evidence/ space only, and let
# the fetch layer index the post-redirect URL.
COCHRANE_ALLOW = re.compile(r"(?i)^/(evidence|news)(/|$)")

EMA_ALLOW = re.compile(
    r"(?i)^/en/(medicines|documents|human-regulatory-overview|"
    r"veterinary-regulatory-overview|news)(/|$)")

ECDC_ALLOW = re.compile(
    r"(?i)^/en/(publications-data|all-topics|infectious-disease-topics|news-events)(/|$)")

NICE_ALLOW = re.compile(r"(?i)^/(guidance|advice|standards|process)(/|$)")

# Measured: the publications space is /docs/... (3,260 /docs/article alone).
# /regions-et-territoires is ~14k regional epidemiological bulletins - high
# volume, low global authority density - and is deliberately excluded.
SPF_ALLOW = re.compile(
    r"(?i)^/(docs|maladies-et-traumatismes|determinants-de-sante|"
    r"etudes-et-enquetes)(/|$)")

# Mission languages are EN/DE/FR/IT (Romansh dropped 2026-06-21). Several IGO
# sitemaps carry every UN language; excluded BEFORE fetch rather than fetched
# and then dropped by the language gate.
OFFLANG = re.compile(
    r"(?i)^/(ar|es|ru|zh|pt|nl|da|el|fi|sv|et|lv|lt|pl|cs|sk|sl|hu|ro|bg|hr|ga|mt)(/|$)")

# Measured: /DE/Themen 4,279 · /DE/Aktuelles 2,143 · /DE/Institut 706 ·
# /EN/News 580 · /EN/Institute 371 · /EN/Topics 284.
RKI_ALLOW = re.compile(
    r"(?i)^/(DE|EN)/(Themen|Topics|Institut|Institute|Service|News|"
    r"Aktuelles)(/|$)")

BIOMED_HOSTS: dict[str, dict] = {
    # Caps are set by WAVE BUDGET, not by what each host could yield. The wave
    # ceiling is ~300k points; energy projects to ~57k, so biomed gets ~160k,
    # education ~25k, urban ~15k. At the smoke-measured ~5-8 passages/page that
    # puts biomed at ~20k pages. Trimmed by authority density: Cochrane (the
    # §4.6-named evidence-synthesis body, zero points held) and WHO fact
    # sheets/publications take the largest shares.
    "cochrane": dict(host="www.cochrane.org", mode="sitemap",
                     sitemaps=["https://www.cochrane.org/sitemap.xml"],
                     include=lambda u: bool(COCHRANE_ALLOW.match(_path(u)))
                     and _deep(u) and _clean(u),
                     lang=lambda u: "en", delay=2.0, cap=6000,
                     source_type="official", tier="A", trust=TRUST_IGO_PRIMARY,
                     license="cochrane-terms"),
    # sitemap roots below are the hosts' OWN robots-declared indexes where they
    # declare one (who.int, rki.de, ema, medlineplus) — guessing /sitemap.xml
    # silently produced empty inventories for who.int and rki.de on the first
    # launch. Verified against the task-399 recon.json `route.roots`.
    "who": dict(host="www.who.int", mode="sitemap",
                sitemaps=["http://www.who.int/sitemaps/sitemapindex.xml"],
                include=lambda u: (bool(WHO_ALLOW.match(_path(u))) and _deep(u)
                                   and _clean(u)
                                   and not OFFLANG.match(_path(u))
                                   and "/ar/" not in _path(u)
                                   and "/zh/" not in _path(u)
                                   and "/ru/" not in _path(u)
                                   and "/es/" not in _path(u)),
                lang=lambda u: "en", delay=2.0, cap=5000,
                source_type="official", tier="A", trust=TRUST_IGO_PRIMARY,
                license="who-web-terms"),
    "ecdc": dict(host="www.ecdc.europa.eu", mode="sitemap",
                 sitemaps=["https://www.ecdc.europa.eu/sitemap.xml"],
                 include=lambda u: (bool(ECDC_ALLOW.match(_path(u)))
                                    and _deep(u) and _clean(u)),
                 lang=lambda u: "en", delay=2.0, cap=1500,
                 source_type="official", tier="A", trust=TRUST_OFFICIAL,
                 license="eu-reuse-2011-833"),
    # /spanish/ is excluded: Spanish is outside the mission languages
    # (EN/DE/FR/IT) and the sources schema's lang enum will not accept it.
    "medlineplus": dict(host="medlineplus.gov", mode="sitemap",
                        sitemaps=["https://medlineplus.gov/sitemap.xml"],
                        include=lambda u: (_deep(u) and _clean(u)
                                           and not _path(u).startswith("/spanish/")),
                        lang=lambda u: "en",
                        delay=2.0, cap=1500,
                        source_type="official", tier="A",
                        trust=TRUST_OFFICIAL,
                        license="us-gov-public-domain"),
    "spf": dict(host="www.santepubliquefrance.fr", mode="sitemap",
                sitemaps=["https://www.santepubliquefrance.fr/sitemap.xml"],
                include=lambda u: (bool(SPF_ALLOW.match(_path(u))) and _deep(u)
                                   and _clean(u)),
                lang=lambda u: "fr", delay=2.0, cap=1500,
                source_type="official", tier="A", trust=TRUST_OFFICIAL,
                license="etalab-2.0"),
    # rki.de declares crawl-delay: 30 — at 1,000 pages that is 8.3 hours and it
    # would gate the whole biomed pack. Capped to what 30s/page can deliver in
    # a reasonable window; RKI already holds 365 points, so this is a top-up.
    "rki": dict(host="www.rki.de", mode="sitemap",
                sitemaps=["https://www.rki.de/Sitemap_Index.xml"],
                include=lambda u: (bool(RKI_ALLOW.match(_path(u))) and _deep(u)
                                   and _clean(u)),
                lang=lambda u: "en" if _path(u).startswith("/EN/") else "de",
                delay=2.0, cap=250,
                source_type="official", tier="A", trust=TRUST_OFFICIAL,
                license="rki-terms"),
}

# ===========================================================================
# PACK: education & learning-science research
# Task 395 §5b: the official-statistics layer already serves PISA/OECD-flavoured
# queries well (edudoc.ch 38,748 pts, cedefop 18,220). The genuine gap is
# research/policy content, so this pack is scoped narrowly to that.
# ===========================================================================
IES_ALLOW = re.compile(r"(?i)^/(ncee|use-work|schoolsurvey|pubs|ncer|"
                       r"transfer|blogs|rel|wwc)(/|$)")
# Measured: /en|fr|es/publication (SINGULAR, ~1,600 each) + /articles (~900/800).
# `/es/` is out of the mission languages (EN/DE/FR/IT) and is excluded here
# rather than fetched and dropped later by the language gate.
IIEP_ALLOW = re.compile(r"(?i)^/(en|fr)/(publication|articles|research|"
                        r"our-expertise|learning-portal)(/|$)")

EDUCATION_HOSTS: dict[str, dict] = {
    "ies": dict(host="ies.ed.gov", mode="sitemap",
                sitemaps=["https://ies.ed.gov/sitemap.xml"],
                include=lambda u: (bool(IES_ALLOW.match(_path(u))) and _deep(u)
                                   and _clean(u)),
                lang=lambda u: "en", delay=2.0, cap=2000,
                source_type="official", tier="A", trust=TRUST_OFFICIAL,
                license="us-gov-public-domain"),
    # NOTE train_ok=False for this host (Content-Signal ai-train=no with an
    # EXPLICIT search=yes grant). It must be staged with its own consent stamp,
    # never under a pack-wide train_ok=True.
    "iiep": dict(host="www.iiep.unesco.org", mode="sitemap",
                 sitemaps=["https://www.iiep.unesco.org/sitemap.xml"],
                 include=lambda u: (bool(IIEP_ALLOW.match(_path(u)))
                                    and _deep(u) and _clean(u)),
                 lang=lambda u: "fr" if _path(u).startswith("/fr/") else "en",
                 delay=2.0, cap=2000,
                 source_type="official", tier="A", trust=TRUST_OFFICIAL,
                 license="unesco-cc-by-sa-igo"),
}

# ===========================================================================
# PACK: urban governance / public policy
# ===========================================================================
URBAN_HOSTS: dict[str, dict] = {
    # NOTE train_ok=False for citiesalliance (robots disallows the training-only
    # token family). Own consent stamp at staging, as with iiep.
    "citiesalliance": dict(host="www.citiesalliance.org", mode="sitemap",
                           sitemaps=["https://www.citiesalliance.org/sitemap.xml"],
                           include=lambda u: _deep(u) and _clean(u),
                           lang=lambda u: "en", delay=2.0, cap=1500,
                           source_type="official", tier="B",
                           trust=TRUST_NETWORK,
                           license="citiesalliance-terms"),
    # iclei.org is a FLAT site: its 7,162 sitemap URLs are mostly depth-1
    # slugs (/news, /<article-slug>). The _deep() rule that protects the other
    # hosts from listing hubs would delete almost the entire inventory here,
    # so this host keeps depth-1 items and only drops asset paths.
    # /author/ is excluded on MEASURED evidence, not taste: the completed
    # 2026-08-07 run fetched 1,200 iclei URLs and 474 of them (39.5%) were
    # /author/... archive pages that 404 — stale entries in iclei's own
    # sitemap. They are navigation tail with no authority content, and
    # re-fetching known-404s is a politeness cost we should not spend on a
    # host that is otherwise healthy (the other 726 fetches were all 200).
    "iclei": dict(host="iclei.org", mode="sitemap",
                  sitemaps=["https://iclei.org/sitemap.xml"],
                  include=lambda u: (_clean(u) and _segs(u)
                                     and not _path(u).startswith("/wp-content/")
                                     and not _path(u).startswith("/author/")),
                  lang=lambda u: "en", delay=2.0, cap=1200,
                  source_type="official", tier="B", trust=TRUST_NETWORK,
                  license="iclei-terms"),
}

PACKS = {"energy": ENERGY_HOSTS, "biomed": BIOMED_HOSTS,
         "education": EDUCATION_HOSTS, "urban": URBAN_HOSTS}

# Hosts that failed closed at recon 2026-08-07. Recorded, never worked around.
SKIPPED_HOSTS = {
    "www.irena.org": "live consent search_ok=False (direct mode) — DECLINED",
    "www.ademe.fr": "live consent search_ok=False (direct mode) — DECLINED",
    "www.cochranelibrary.com": "robots fail-closed (WAF/JS-shell on robots.txt)",
    "www.aifa.gov.it": "robots fail-closed",
    "www.salute.gov.it": "robots fail-closed",
    "www.epicentro.iss.it": "robots fail-closed",
    "unhabitat.org": "robots fail-closed (holds 232 pts from an earlier route)",
    "uis.unesco.org": "robots fail-closed",
    "www.nrel.gov": "robots fail-closed",
    "www.oecd.org": "403 to BOTH our UA and a browser UA on every path except "
                    "robots.txt — infrastructure/IP-level block, not a "
                    "UA-targeted one and not spoofable. robots.txt itself "
                    "ALLOWS * and declares a sitemap, so this is a "
                    "contradiction to resolve with OECD, not around them.",
    "educationendowmentfoundation.org.uk": "403 to both UAs (same as OECD)",
    "www.unesco.org": "no sitemap at any common location; needs a bespoke route",
    "www.iaea.org": "no sitemap; robots.txt returns 405",
    "www.c40.org": "no sitemap at any common location",
    "www.unep.org": "sitemap enumerates but content pages 403 (WAF)",
    "unfccc.int": "sitemap enumerates but sampled pages extract EMPTY — "
                  "JS-composed; needs a bespoke route",
    "www.bfarm.de": "no sitemap at common locations",
    "www.gesundheitsinformation.de": "no sitemap at common locations",
    "www.has-sante.fr": "no sitemap at common locations (holds 9,436 pts already)",
    "ansm.sante.fr": "no sitemap at common locations",
    "www.ema.europa.eu": "declared sitemap is a DOCUMENT dump: 4,000 rows, "
                         "99.1% .pdf/.doc binaries, fanned across 24 EU "
                         "languages. The medicine/EPAR pages that carry the "
                         "authority (verified extractable by hand: 25 passages "
                         "on the Comirnaty EPAR) are NOT in it. Deferred "
                         "pending a bespoke route; holds 1,121 pts already.",
    "www.nice.org.uk": "declared sitemap's /guidance/* entries are STUBS "
                       "(awaiting-development, deferred, inconsultation) — the "
                       "actual guideline corpus is not in the sitemap; the "
                       "remaining mass is /news/articles. Deferred pending a "
                       "bespoke route; holds 4,420 pts already.",
    "www.cdc.gov": "sitemap enumerates only 45 rows — inventory route is "
                   "elsewhere; deferred rather than shipped as a token slice",
    "www.iss.it": "sitemap enumerates only 9 rows — same as cdc.gov",
    "iris.who.int": "HELD this wave per the task-399 brief; the governing "
                    "ruling (PR #47, merged 2026-07-26) already permits the "
                    "minimal-record posture and 19,573 points are live",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, choices=sorted(PACKS))
    pack_fetch.add_standard_args(ap, out_dir="./authority_packs_run", budget=60000)
    args = ap.parse_args()
    hosts = PACKS[args.pack]
    print(f"[task399] pack={args.pack} hosts={list(hosts)} "
          f"skipped={len(SKIPPED_HOSTS)}", flush=True)
    pack_fetch.run_pack(pack=f"aw399_{args.pack}", hosts=hosts, args=args,
                        canon=pack_fetch.make_canon({}),
                        default_tier="A",
                        kind_prio={"content": 0, "hub": 2})
    return 0


if __name__ == "__main__":
    sys.exit(main())
