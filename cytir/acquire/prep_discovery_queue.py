"""Discovery-engine candidate PREP (ADR-020 step: candidates → prioritized queue).

Takes the endorsement-run output (`endorse_all.json` edge graph + the scored
`discovery_candidates.json` list) and turns it into a clean, prioritized
ingestion queue:

  1. Re-derive candidates at **HOST granularity** from the edge graph (the
     targets are full URLs, so per-host endorsement scores are recoverable),
     falling back to the registrable-level candidate list for sources that
     carry no edge provenance (e.g. HN / wiki candidate lists folded in).
  2. DEDUP vs what's already in the live index — **per host, not per
     registrable domain** (see "Granularity" below).
  3. DROP obvious noise (platform hosts, CDNs, link-shorteners, translate
     wrappers, social, dev/file dumps) — these are chrome, not authoritative
     endorsements.
  4. CLASSIFY each survivor into gov / edu / health / ref / tech / news / other,
     using (a) the endorsement graph provenance (which seed CLASSES endorsed it,
     from endorse_all.json edges) and (b) domain/TLD heuristics.
  5. RANK by  endorsement_score × authority_weight  (class trust + Swiss boost).

Granularity (2026-07-03 fix — the bug that hid the whole agency layer)
----------------------------------------------------------------------
The old prep deduped at REGISTRABLE-domain granularity: because
`fedlex.admin.ch` is live, the entire `admin.ch` family (estv/bag/seco/news…)
was treated as "already indexed" and never became a candidate. The same
mechanism would hide `canada.ca`, `europa.eu`, `gouv.fr`, `gc.ca` sub-sites.

The fix: candidates and dedup are now HOST-granular. Rules per pattern class:

* **Multi-tenant institutional registrables** (admin.ch, gc.ca, canada.ca,
  europa.eu, gouv.fr, …): every host is a distinct authority → host-level
  candidates, host-level dedup. An already-indexed host still dedups exactly
  (`fedlex.admin.ch` stays excluded; `estv.admin.ch` becomes eligible).
* **Single-owner registrables** (nzz.ch, doi.org, python.org, …): host-level
  too. An indexed host no longer shadows sibling hosts (`docs.python.org`
  being live no longer hides `python.org` human pages) — siblings only enter
  if *independently endorsed*, and they rank on their own endorsement score,
  so junk subdomains don't flood in.
* **Personal-publishing clouds** (*.blogspot.*, *.wordpress.com, medium.com,
  *.wixsite.com, …): every subdomain IS a different owner, but that is not
  the bar — the *content class* is personal publishing, off-mission for
  authoritative grounding at any granularity. Stay pattern-DENYLISTED
  (NOISE_SUB). Specific high-value tenants enter via curated seed queues,
  never via the noise filter.
* **Dev-docs hosting clouds** (*.github.io, *.gitlab.io, *.readthedocs.io,
  *.pages.dev, *.netlify.app, *.vercel.app, …): every subdomain is a
  different owner AND the typical tenant is an OSS project's documentation —
  on-mission tech docs. Host granularity applies: a *tenant host* is eligible
  and competes on its own endorsement score; the *bare platform apex*
  (github.io itself) is still dropped — it is infrastructure, not an
  authority. (Legacy registrable mode keeps the old blanket denylist.)
* **Host normalization**: lowercase, port stripped, leading `www\\d*.` folded
  (mirrors CC/surt canonicalisation AND the live index's `domain` payload,
  which is host-granular and www-less). NOTE (verified 2026-07-03): a
  registrable-level queue entry's surtkey range does NOT cover subdomain
  captures — `ch,admin,estv)/…` sorts outside `ch,admin)/…` — so per-host
  queue entries are also what actually makes the fetch reach agency hosts.

`--granularity registrable` keeps the legacy behaviour (for A/B comparison).

Output: discovery_queue.json — a list of
  {domain, reg, class, score, endorse_score, authority, swiss, by_classes, why}
sorted best-first, ready for `discovery_ingest.py` (policy-(a) CC fetch + embed).
(`domain` is a HOST in host mode; `reg` is its registrable domain.)

  python -m cytir.acquire.prep_discovery_queue \
      --candidates ~/aw-data/discovery_candidates.json \
      --endorse    ~/aw-data/endorse_all.json \
      --live-domains /tmp/live_domains.json \
      --out        ~/aw-data/discovery_queue.json

No network needed except the optional live-domain facet pull (done out-of-band;
pass the dumped facet JSON via --live-domains so this stays offline + reproducible).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from urllib.parse import urlparse

from cytir.extract.html_extract import reg_domain


# Seed-class trust weights — KEEP IN SYNC with cytir.acquire.extract_outlinks.TRUST (the
# weights that built `domain_authority`); re-used here so host-level scores
# re-aggregated from the edges sum back to the registrable-level scores.
# `disc-*` classes are HARVESTED endorsers (aggregate_harvested_links.py —
# tier-C discovered domains whose fetched pages' outlinks fold back into the
# graph, the compounding-loop step): deliberately weighted below the curated
# seed classes, with news/other endorsers discounted hardest.
SEED_TRUST = {"federal-gov": 1.0, "cantonal": 0.9, "universities-research": 0.9,
              "health": 0.9, "reference-cultural": 0.85, "portals": 0.8,
              "disc-gov": 0.85, "disc-edu": 0.8, "disc-health": 0.8,
              "disc-ref": 0.75, "disc-tech": 0.7, "disc-news": 0.5,
              "disc-other": 0.4}
SEED_TRUST_DEFAULT = 0.8
SEED_VOTE_CAP = 5          # extract_outlinks: strength = min(cnt, 5)
# Per-(endorser, target-HOST) cap (2026-07-04). SEED_VOTE_CAP bounds one
# endorser's vote per target *URL*, but a host's score sums over its URLs — so
# a single endorser linking thousands of distinct URLs on one host dominated
# the queue (measured: bauernzeitung.ch → 4,912 reader.epaper.guru URLs;
# jmlr.org → 4,347 openreview.net URLs). One endorser now contributes at most
# 5 fully-endorsed URLs' worth (25×w) to any single target host; independent
# endorsers still compound.
ENDORSER_HOST_CAP = 25.0

_WWW = re.compile(r"^www\d*\.")
_HOST_OK = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$")


def norm_host(h: str) -> str:
    """Canonical host key: lowercase, no port, no trailing dot, leading
    www\\d*. folded — mirrors CC/surt canonicalisation and the live index's
    `domain` payload convention (host-granular, www-less)."""
    h = (h or "").lower().strip().strip(".")
    h = re.sub(r":\d+$", "", h)
    return _WWW.sub("", h)


# --- NOISE denylist ---------------------------------------------------------
# Platform/CDN/aggregator hosts that are link CHROME, not content endorsements.
# Substring match on host AND registrable domain. Pattern classes are
# documented in the module docstring ("Granularity"): chrome/SaaS/social/CDN
# and personal-publishing clouds are denylisted at ANY granularity;
# dev-docs hosting clouds are in PLATFORM_DOC_CLOUDS below instead (tenant
# hosts eligible at host granularity, platform apex dropped).
NOISE_SUB = (
    # blogging / site builders / CMS clouds (multi-tenant; not one authority)
    "blogger.com", "blogspot.", "wordpress.com", "wix.com", "weebly.com",
    "squarespace.com", "jimdo.", "webnode.", "tumblr.com", "medium.com",
    "sitecore", "sharepoint.com", "netvibes.com", "typepad.com", "over-blog",
    "force.com", "myshopify.com", "wixsite.com", "godaddysites.com",
    # social / messaging / forums-as-platform
    "bsky.app", "bsky.social", "facebook.", "twitter.com", "x.com",
    "instagram.com", "linkedin.com", "youtube.com", "youtu.be", "tiktok.com",
    "pinterest.", "reddit.com", "t.me", "telegram.", "whatsapp.com",
    "mastodon.", "threads.net", "vk.com", "flickr.com",
    # translate / cache / proxy / shorteners
    "translate.goog", "translate.google", "webcache.googleusercontent",
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "lnkd.in",
    "yahoo.com", "yahoo.net", "search.yahoo",
    # CDNs / asset / analytics / fonts (rarely a top endorsement, but seen)
    "cloudfront.net", "akamai", "fastly.net", "jsdelivr.net", "unpkg.com",
    "gstatic.com", "googleapis.com", "gravatar.com", "cloudflare.com",
    "doubleclick", "googletagmanager", "google-analytics",
    # file / paste dumps + raw cloud storage
    "amazonaws.com", "blob.core.windows.net", "azurewebsites.net",
    "herokuapp.com",
    "pastebin.com", "scribd.com", "slideshare.net", "issuu.com", "epaper.guru",
    "dropbox.com", "drive.google", "docs.google", "sites.google",
    # generic portals / SaaS marketing surfaces
    "sitecorecontenthub.cloud", "typeform.com", "surveymonkey", "eventbrite",
    "mailchimp", "hubspot", "salesforce", "zendesk.com", "atlassian.net",
    # accessibility / TTS / widget SaaS embedded site-wide (chrome, not authority)
    "readspeaker.com", "browsealoud", "recite-me", "usercentrics", "onetrust",
    "cookiebot", "matomo.", "piwik.", "addtoany", "sharethis", "addthis",
    # conferencing / business-network / storage SaaS (not a content authority)
    "zoom.us", "webex.com", "teams.microsoft", "xing.com", "wasabisys.com",
    "backblazeb2.com", "digitaloceanspaces", "exlibrisgroup.com", "primo.exlibris",
    "calendly.com", "doodle.com", "padlet.com", "menti.com", "kahoot",
    # share-links / podcast-and-mail chrome / shops (surfaced by the harvested
    # graph 2026-07-04 — link chrome on many endorsers, not content authority)
    "wa.me", "spotify.com", "mailchi.mp", "outlook.live", "forms.gle",
    "cookiedatabase.org", "omniupdate.com", "weibo.com",
    "newsletter.infomaniak", "ondemand-mp3.",
    "amazon.com", "amazon.de", "amazon.fr", "amazon.it", "amazon.co.uk",
)


# Dev-docs hosting clouds — each tenant subdomain is a distinct OSS project's
# docs (on-mission). Host granularity: tenant hosts eligible, bare apex noise.
# Legacy registrable mode keeps the old blanket drop.
PLATFORM_DOC_CLOUDS = ("github.io", "gitlab.io", "netlify.app", "vercel.app",
                       "pages.dev", "web.app", "firebaseapp.com",
                       "readthedocs.io")


def is_noise(dom: str, granularity: str = "host") -> str | None:
    d = (dom or "").lower()
    for s in NOISE_SUB:
        if s in d:
            return s
    for s in PLATFORM_DOC_CLOUDS:
        if d == s or d.endswith("." + s):
            # tenant hosts are eligible in host mode; the bare apex never is
            if granularity != "host" or d == s:
                return s
    # bare IPs, localhost, single-label, malformed hosts ("admin.chhttps:" …)
    if re.fullmatch(r"[\d.]+", d) or "." not in d or not _HOST_OK.fullmatch(d):
        return "non-domain"
    return None


# --- CLASSIFICATION heuristics ----------------------------------------------
# Order matters: gov/edu/health are the high-trust on-mission classes.
# All keyword matching is substring-on-host, so it works unchanged at host
# granularity (e.g. "estv.admin.ch" hits the admin.ch/estv. gov keywords).
SWISS_TLD = re.compile(r"\.(ch|swiss|li)$")
GOV_TLD = re.compile(r"\.(gov|gouv\.[a-z]{2}|gov\.[a-z]{2}|admin\.ch|gv\.at|bund\.de|"
                     r"gc\.ca|europa\.eu|go\.[a-z]{2})$")
EDU_TLD = re.compile(r"\.(edu|ac\.[a-z]{2}|edu\.[a-z]{2}|uni-[a-z]+\.de)$")

GOV_KW = ("admin.ch", "bund.de", ".gov", "gouv.", ".gv.at", ".gc.ca", "europa.eu",
          "bfs.", "seco.", "estv.", "swisstopo", "parlament.ch", "bag.admin",
          ".canton.", "ge.ch", "vd.ch", "zh.ch", "be.ch", "ti.ch", "vs.ch",
          "opendata.swiss", "bk.admin", "edi.admin")
EDU_KW = ("epfl.ch", "ethz.ch", "uzh.ch", "unige.ch", "unibe.ch", "unil.ch",
          "unibas.ch", "usi.ch", "hes-so", ".edu", "university", "universite",
          "universita", "universitaet", "hochschule", "swissuniversities",
          "snf.ch", "snsf.ch", "scnat", "lib4ri.ch", "dasch.swiss", "switch.ch")
HEALTH_KW = ("hug.ch", "chuv.ch", "insel.ch", "usz.ch", "unispital", "spital",
             "hopital", "ospedale", "fmh.ch", "swissmedic", "bag-anw", "who.int",
             "ncbi.nlm.nih", "nih.gov", "cochrane", "planetesante", "addiction",
             "pharma", "medic", "health", "sante", "gesundheit", "salute",
             "rosenfluh", "sams.ch", "samw.ch")
REF_KW = ("hls-dhs-dss.ch", "wikidata", "viaf.org", "d-nb.info", "worldcat",
          "gallica.bnf", "data.bnf", "idref.fr", "doi.org", "crossref",
          "datacite", "orcid.org", "zenodo.org", "jstor.org", "id.loc.gov",
          "loc.gov", "gnd.network", "lobid.org", "e-periodica", "e-rara",
          "ssrn.com", "semanticscholar", "openalex", "gbif.org", "museum",
          "bibliothek", "archive", "katalog", "swisscovery", "rero.ch",
          "e-helvetica", "nb.admin")
TECH_KW = ("rfc-editor", "ietf.org", "w3.org", "iana.org", "ecma-international",
           "iso.org", "iec.ch", "unicode.org", "mozilla.org", "developer.",
           "docs.", "github.com", "gitlab.com", "stackoverflow", "npmjs",
           "pypi.org", "kernel.org", "apache.org", "gnu.org", "python.org",
           "postgresql.org", "khronos.org", "whatwg.org", "caniuse")
NEWS_KW = ("nytimes", "bbc.co", "bbc.com", "theguardian", "lemonde", "letemps",
           "nzz.ch", "tagesanzeiger", "srf.ch", "rts.ch", "rsi.ch", "swissinfo",
           "reuters", "bloomberg", "spiegel", "lefigaro", "liberation", "20min",
           "watson.ch", "blick.ch", "heise.de", "arstechnica", "techcrunch",
           # Swiss regional press (CH-Media / Tamedia titles) — publisher class
           "bazonline", "aargauerzeitung", "luzernerzeitung", "bzbasel",
           "tagblatt", "limmattalerzeitung", "grenchnertagblatt", "nau.ch",
           "watson", "20minuten", "handelszeitung", "cash.ch", "finews",
           "bote.ch", "zsz.ch", "landbote", "thurgauerzeitung", "solothurnerzeitung")


# --- seed class -> candidate class -------------------------------------------
# Hoisted to module level (task 440) so classify() and the corroboration counter
# below cannot drift apart. `disc-*` = harvested endorsers; curated classes
# always outrank them.
SEED_CLASS_TO_CANDIDATE_CLASS = {
    "federal-gov": "gov", "cantonal": "gov", "portals": "gov",
    "universities-research": "edu", "health": "health",
    "reference-cultural": "ref",
    "disc-gov": "gov", "disc-edu": "edu", "disc-health": "health",
    "disc-ref": "ref", "disc-tech": "tech", "disc-news": "news"}
# Priority order: highest-trust endorsing class present wins.
SEED_CLASS_PRIORITY = ("health", "federal-gov", "cantonal",
                       "universities-research", "reference-cultural", "portals",
                       "disc-health", "disc-gov", "disc-edu", "disc-ref",
                       "disc-tech", "disc-news")


# --- earned vs inherited class (task 440; supersedes the task-203 denylist) --
# Task 203 capped the fallback with a hardcoded 90-host COMMERCIAL_DENY list.
# That list was removed here. It could not generalize -- it enumerates the past:
# task 438's fresh sample found six commercial hosts carrying gov-tier trust and
# NONE were on it, and it caught 503 hosts out of 90,598 that hold a high-trust
# class purely by inheritance (0.6%). Accreted host lists are also forbidden
# doctrine (ADR-043 A1 §7 "never to accreted special-case scorer gates";
# Tim 2026-07-28 "fix error CLASSES ... no host special-cased").
#
# The general rule that replaces it does not demote a class at all. It records
# HOW the class was reached, and discovery_ingest.resolve_trust prices an
# endorsed class from the vendor-content floor instead of the mission tier
# (POLICY.md v2 §4.4). A commercial host linked once by a university therefore
# stops inheriting edu-tier trust because it was linked once -- not because
# anyone wrote its name down.
def class_basis(why: str) -> str:
    """'self-signal' when the hostname itself carried the credential (branch 1),
    'endorsed' when only the endorsement graph did (branch 2)."""
    return "endorsed" if why.startswith("endorsed-by") else "self-signal"


def endorser_regs_by_class(endorse: dict) -> tuple[dict, dict]:
    """(host_cls_regs, reg_cls_regs): for each target host / registrable, the set
    of DISTINCT endorsing REGISTRABLE domains per seed class.

    Separate from host_scores_from_edges (which two other callers unpack by
    arity) because it answers a different question: that one asks how strongly a
    host is endorsed for RANKING, this one asks by how many independent parties
    for TRUST. Registrables on both sides: one endorser's many links are one
    voice (why ENDORSER_HOST_CAP exists), and its many hosts are one voice too.
    """
    host_cls_regs: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    reg_cls_regs: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for seed, info in (endorse.get("edges") or {}).items():
        sc = info.get("class")
        if not sc:
            continue
        seed_reg = reg_domain(norm_host(seed))
        for tgt in (info.get("targets") or {}):
            try:
                h = norm_host(urlparse(tgt if "//" in tgt else "http://" + tgt).netloc)
            except Exception:  # noqa: BLE001
                continue
            if not h or "." not in h:
                continue
            host_cls_regs[h][sc].add(seed_reg)
            reg_cls_regs[reg_domain(h)][sc].add(seed_reg)
    return host_cls_regs, reg_cls_regs


def n_corroborating_regs(host: str, cls: str, host_cls_regs: dict) -> int:
    """How many DISTINCT registrable domains endorse `host` for candidate class
    `cls`.

    Counts DIRECT endorsements of this host only. An endorsement of a sibling
    host under the same registrable is evidence about the registrable, not about
    this host -- build_queue still lets that sibling evidence make the host
    DISCOVERABLE (it stays in `bc`), but it contributes nothing to trust. Task
    439 measured 11,968 hosts holding a high-trust class with no direct
    endorsement of that class at all."""
    acc: set = set()
    for sc, regs in (host_cls_regs.get(host) or {}).items():
        if SEED_CLASS_TO_CANDIDATE_CLASS.get(sc) == cls:
            acc |= regs
    return len(acc)


def classify(dom: str, by_classes: set[str]) -> tuple[str, str]:
    """Return (class, why). Heuristics first (precise), then graph provenance.

    `why` records WHICH branch fired, and class_basis() reads it back: branch 1
    is a credential in the hostname, branch 2 is hearsay from the link graph.
    Trust is priced off that distinction in discovery_ingest.resolve_trust --
    classify itself never demotes a class (task 440)."""
    d = dom.lower()
    # 1) strong domain heuristics
    if any(k in d for k in GOV_KW) or GOV_TLD.search(d):
        return "gov", "gov TLD/keyword"
    if any(k in d for k in HEALTH_KW):
        return "health", "health keyword"
    if any(k in d for k in EDU_KW) or EDU_TLD.search(d):
        return "edu", "edu/research keyword"
    if any(k in d for k in REF_KW):
        return "ref", "reference/library/registry keyword"
    if any(k in d for k in TECH_KW):
        return "tech", "tech-standard/docs keyword"
    if any(k in d for k in NEWS_KW):
        return "news", "news/publisher keyword"
    # 2) fall back to the endorsement graph provenance (which seed classes link it)
    if by_classes:
        for sc in SEED_CLASS_PRIORITY:
            if sc in by_classes:
                cls = SEED_CLASS_TO_CANDIDATE_CLASS.get(sc, "other")
                return cls, f"endorsed-by {sc} seed(s)"
    return "other", "unclassified"


# Per-candidate-class authority weight (TrustRank intuition; gov/edu/health top).
# `commercial` (task-203) sits below `other`: findable, no authority boost.
CLASS_W = {"gov": 1.0, "health": 0.95, "edu": 0.9, "ref": 0.85,
           "tech": 0.8, "news": 0.55, "other": 0.5, "commercial": 0.4}
SWISS_BOOST = 1.5   # on-mission (P0 = Swiss-first)


def load_live_hosts(path: str) -> tuple[set[str], set[str]]:
    """(hosts, regs) already in the live index. `hosts` are the facet values
    normalized via norm_host (the index `domain` payload is host-granular);
    `regs` their registrable domains — used only by the legacy
    `--granularity registrable` mode."""
    if not path:
        return set(), set()
    d = json.load(open(path))
    hits = d.get("result", {}).get("hits", d if isinstance(d, list) else [])
    hosts, regs = set(), set()
    for h in hits:
        v = h["value"] if isinstance(h, dict) else h
        nh = norm_host(v)
        hosts.add(nh)
        regs.add(reg_domain(nh))
        regs.add(nh)
    return hosts, regs


def host_scores_from_edges(endorse: dict) -> tuple[dict[str, float],
                                                   dict[str, set[str]],
                                                   dict[str, set[str]]]:
    """Re-aggregate the edge graph at HOST granularity.

    Returns (host_score, host_classes, reg_classes). Uses the exact weighting
    extract_outlinks applies for `domain_authority` (seed-class trust ×
    min(pages, 5)), so a registrable's host scores sum back to its old
    registrable-level endorse_score (top-5000 truncation aside)."""
    host_score: dict[str, float] = defaultdict(float)
    host_classes: dict[str, set[str]] = defaultdict(set)
    reg_classes: dict[str, set[str]] = defaultdict(set)
    for _seed, info in (endorse.get("edges") or {}).items():
        sc = info.get("class")
        w = SEED_TRUST.get(sc, SEED_TRUST_DEFAULT)
        seed_host: dict[str, float] = defaultdict(float)   # this endorser only
        for tgt, cnt in (info.get("targets") or {}).items():
            try:
                h = norm_host(urlparse(tgt if "//" in tgt else "http://" + tgt).netloc)
            except Exception:  # noqa: BLE001
                continue
            if not h or "." not in h:
                continue
            try:
                strength = min(float(cnt), SEED_VOTE_CAP)
            except (TypeError, ValueError):
                strength = 1.0
            seed_host[h] += strength
            if sc:
                host_classes[h].add(sc)
                reg_classes[reg_domain(h)].add(sc)
        for h, s in seed_host.items():                     # ENDORSER_HOST_CAP
            host_score[h] += w * min(s, ENDORSER_HOST_CAP)
    return host_score, host_classes, reg_classes


def build_queue(cands: list[dict], endorse: dict,
                live_hosts: set[str], live_regs: set[str],
                granularity: str = "host"):
    """Core prep (pure, offline). Returns (kept, dropped_noise, dropped_indexed).

    host mode: candidates = per-host scores re-derived from the edge graph,
    PLUS registrable-level fallback entries for --candidates rows whose
    registrable has no edge provenance at all (other-source candidate lists —
    HN/wiki — fold in unchanged). Dedup drops a candidate iff its normalized
    host is itself live (exact-host carve-out): fedlex.admin.ch stays excluded,
    estv.admin.ch becomes eligible.

    registrable mode: the legacy behaviour (kept for A/B comparison)."""
    host_score, host_classes, reg_classes = host_scores_from_edges(endorse)
    host_cls_regs, _reg_cls_regs = endorser_regs_by_class(endorse)

    # assemble the candidate list at the requested granularity
    if granularity == "host":
        edge_regs = {reg_domain(h) for h in host_score}
        pool = [{"domain": h, "endorse_score": s, "_reg": reg_domain(h)}
                for h, s in host_score.items()]
        # fallback: candidate rows with no edge provenance keep reg granularity
        for c in cands:
            dom = norm_host(c.get("domain", ""))
            if not dom:
                continue
            if reg_domain(dom) not in edge_regs:
                pool.append({"domain": dom,
                             "endorse_score": c.get("endorse_score", 0.0),
                             "_reg": reg_domain(dom)})
    else:
        pool = [{"domain": norm_host(c.get("domain", "")),
                 "endorse_score": c.get("endorse_score", 0.0),
                 "_reg": reg_domain(norm_host(c.get("domain", "")))}
                for c in cands]

    kept, dropped_noise, dropped_indexed = [], [], []
    seen: set[str] = set()
    for c in sorted(pool, key=lambda x: -(x.get("endorse_score") or 0.0)):
        dom, rd = c["domain"], c["_reg"]
        es = c.get("endorse_score", 0.0)
        key = dom if granularity == "host" else rd
        if not dom or key in seen:
            continue
        seen.add(key)
        # dedup vs live index
        if granularity == "host":
            if dom in live_hosts:               # exact-host carve-out
                dropped_indexed.append(dom)
                continue
        else:                                    # legacy: whole registrable
            if rd in live_regs or dom in live_regs or dom in live_hosts:
                dropped_indexed.append(dom)
                continue
        # NOISE_SUB is substring-matched on the full host (which contains its
        # registrable), so a single host check covers both levels; granularity
        # governs the PLATFORM_DOC_CLOUDS tenant carve-out.
        nz = is_noise(dom, granularity)
        if nz:
            dropped_noise.append((dom, nz))
            continue
        bc = (host_classes.get(dom, set()) | reg_classes.get(rd, set())
              | reg_classes.get(dom, set()))
        cls, why = classify(dom, bc)
        basis = class_basis(why)
        # task 440: how many INDEPENDENT registrables endorse this host for this
        # class. Emitted so discovery_ingest can price hearsay below credentials
        # instead of stamping CLASS_TRUST[cls] for both.
        n_regs = n_corroborating_regs(dom, cls, host_cls_regs) if basis == "endorsed" else 0
        swiss = bool(SWISS_TLD.search(dom) or SWISS_TLD.search(rd))
        authority = CLASS_W[cls] * (SWISS_BOOST if swiss else 1.0)
        score = es * authority
        kept.append({
            "domain": dom, "reg": rd, "class": cls, "score": round(score, 2),
            "endorse_score": round(es, 2), "authority": round(authority, 3),
            "swiss": swiss, "by_classes": sorted(bc), "why": why,
            "class_basis": basis, "n_endorser_regs": n_regs,
        })

    kept.sort(key=lambda x: -x["score"])
    return kept, dropped_noise, dropped_indexed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--endorse", required=True)
    ap.add_argument("--live-domains", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=0, help="cap queue size (0=all)")
    ap.add_argument("--granularity", choices=("host", "registrable"),
                    default="host",
                    help="candidate/dedup granularity. 'host' (default; the "
                         "2026-07-03 fix): per-host candidates from the edge "
                         "graph, exact-host live dedup — agency subdomains of "
                         "an indexed registrable stay eligible. 'registrable': "
                         "legacy behaviour, for comparison only.")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="drop hosts below this endorse_score (host mode "
                         "surfaces the full host tail; 0 = keep all)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="proceed even if the endorse file has status=error "
                         "seeds (CDX outage during the run — see "
                         "extract_outlinks.py error-vs-empty contract)")
    args = ap.parse_args()

    cands = json.load(open(args.candidates))
    endorse = json.load(open(args.endorse))
    # Error-vs-empty contract (extract_outlinks.py): seeds with status=error hit
    # a CDX outage — their endorsements are MISSING, not zero. A queue built from
    # a partial run silently under-ranks everything those seeds endorse.
    err_seeds = endorse.get("errors") or [
        d for d, i in endorse.get("edges", {}).items()
        if i.get("status") == "error"]
    if err_seeds and not args.allow_partial:
        sys.exit(f"ABORT: endorse file {args.endorse} is PARTIAL — "
                 f"{len(err_seeds)} seeds errored (CDX outage), e.g. "
                 f"{err_seeds[:5]}. Re-run extract_outlinks for those seeds, "
                 f"or pass --allow-partial to proceed anyway.")
    if err_seeds:
        print(f"⚠ proceeding on PARTIAL endorse data ({len(err_seeds)} errored "
              f"seeds) per --allow-partial", file=sys.stderr)
    live_hosts, live_regs = load_live_hosts(args.live_domains)

    kept, dropped_noise, dropped_indexed = build_queue(
        cands, endorse, live_hosts, live_regs, args.granularity)
    if args.min_score:
        kept = [k for k in kept if k["endorse_score"] >= args.min_score]
    if args.top:
        kept = kept[: args.top]

    json.dump(kept, open(args.out, "w"), ensure_ascii=False, indent=0)

    # --- report -------------------------------------------------------------
    from collections import Counter
    cls_ct = Counter(k["class"] for k in kept)
    swiss = [k for k in kept if k["swiss"]]
    print(f"granularity: {args.granularity}")
    print(f"candidates in: {len(cands)} (file) "
          f"+ edge-derived hosts in host mode")
    print(f"  dropped (already indexed): {len(dropped_indexed)} -> {dropped_indexed}")
    print(f"  dropped (noise): {len(dropped_noise)}")
    print(f"  SURVIVE: {len(kept)}")
    print(f"  by class: {dict(cls_ct)}")
    print(f"  Swiss (.ch/.swiss/.li): {len(swiss)}")
    if args.granularity == "host":
        multi = Counter(k["reg"] for k in kept)
        m = {r: n for r, n in multi.most_common(8) if n > 1}
        print(f"  multi-host registrables surfaced (top): {m}")
    print("\nTop 25 overall (score = endorse × authority):")
    for k in kept[:25]:
        sw = " 🇨🇭" if k["swiss"] else ""
        print(f"  {k['score']:9.1f}  [{k['class']:6}] {k['domain']}{sw}  ({k['why']})")
    print("\nTop 25 Swiss:")
    for k in swiss[:25]:
        print(f"  {k['score']:9.1f}  [{k['class']:6}] {k['domain']}  ({k['why']})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
