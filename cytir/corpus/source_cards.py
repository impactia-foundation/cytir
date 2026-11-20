"""The source-card BUILDERS: topic derivation, navigation templates, the deep-index-vs-card
rubric, trust and freshness classes, and card assembly (task 683, S7 part 2).

Moved from the former `build_source_cards` pipeline script (design: notes/adr-source-authority-layer.md);
since task 1235 the `build` / `embed` / `search` commands live here too, behind `cyt ops source-cards`.
`cytir.consent.gate`
needs the card-schema helpers (`card_text`, `derive_topics`, `navigation_for`, `CLASS_*`,
`KNOWN_NAV`) to write the entity-card fallback for consent-declined hosts, and the package may not
import `pipeline/`. Behaviour is pinned byte for byte by `tests/test_aw_s7b_characterization.py`.

A card: { domain, authoritative_for, endorsement, jurisdiction, navigation, license,
freshness_class, trust_tier, disposition (card | deep-index | both) }.

The curated tables below (KNOWN_NAV, PAYWALLED, HN_HEAD, ...) are knowledge held as code; they
move to a data file with a schema when the card lane is ported (engineering-doctrine #4).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re

from cytir.core.country import country_for_domain

# ===========================================================================
# 1. TOPIC DERIVATION
#    Cheap, deterministic, offline. A card's authoritative_for is the embedding
#    surface (config B embeds topics + description). Topics are derived from
#    (a) the discovery class, (b) keyword hits in the domain label, (c) the
#    endorsement-graph `by_classes` provenance (which seed classes link it), and
#    (d) the curated head's hand-written topics.
# ===========================================================================

CLASS_TOPICS = {
    "gov": ["government", "public administration", "official records", "regulation",
            "public services", "statistics"],
    "edu": ["university", "research", "academic", "higher education", "science",
            "publications"],
    "health": ["health", "medicine", "hospital", "clinical", "patient care",
                "public health"],
    "ref": ["reference", "library", "archive", "catalogue", "museum",
            "cultural heritage"],
    "news": ["news", "journalism", "current events", "reporting", "media"],
    "tech": ["technology", "software", "engineering", "developer documentation",
              "standards"],
    "other": [],
}

# Keyword -> topic enrichment from the domain label (the registrable-domain text).
# Multilingual (DE/FR/IT/EN) because the Swiss set is multilingual.
LABEL_TOPICS = [
    (r"(spital|hopital|hopital|ospedale|clinic|klinik|insel|chuv|hug)",
     ["hospital", "clinical care"]),
    (r"(uni|epfl|eth|hochschule|haute.?ecole|scuola|fachhochschule|hslu|zhaw|supsi)",
     ["university", "higher education"]),
    (r"(biblio|library|bibliothek|libr|swisscovery|slsp|lib4ri)",
     ["library", "catalogue"]),
    (r"(museum|landesmuseum|archiv|archive|chateau|chateau|heritage)",
     ["museum", "archive", "cultural heritage"]),
    (r"(geo|swisstopo|geoportal|carto|map)", ["geodata", "maps", "cartography"]),
    (r"(recht|law|droit|legge|justiz|justice|tribunal|gericht)",
     ["law", "justice", "courts"]),
    (r"(steuer|impot|impot|tax|fisc)", ["taxation", "tax administration"]),
    (r"(ahv|iv|sozial|social|assurance|versicherung|prevoyance|pension)",
     ["social insurance", "pensions", "welfare"]),
    (r"(beruf|metier|career|orientation|berufsberatung)",
     ["careers", "vocational guidance"]),
    (r"(meteo|wetter|slf|lawine|avalanche|climat|klima)",
     ["weather", "climate", "environment"]),
    (r"(bank|finanz|finance|borse|bourse)", ["finance", "banking"]),
    (r"(srf|rts|rsi|news|zeitung|journal|tribune|nzz|tagi|blick)",
     ["news", "journalism"]),
]

# The 26 canton ccTLD-style abbreviations -> a cantonal-authority topic.
CANTON_ABBR = {
    "zh", "be", "lu", "ur", "sz", "ow", "nw", "gl", "zg", "fr", "so", "bs", "bl",
    "sh", "ar", "ai", "sg", "gr", "ag", "tg", "ti", "vd", "vs", "ne", "ge", "ju",
}


def derive_topics(domain: str, disc_class: str, by_classes=None) -> list[str]:
    topics: list[str] = []
    seen = set()

    def add(ts):
        for t in ts:
            if t and t not in seen:
                seen.add(t)
                topics.append(t)

    add(CLASS_TOPICS.get(disc_class, []))
    label = domain.lower()
    # cantonal authority? (e.g. vs.ch, zh.ch, ge.ch)
    head = label.split(".")[0]
    if head in CANTON_ABBR and label.endswith(".ch"):
        add([f"canton {head.upper()}", "cantonal authority", "regional government"])
    for pat, ts in LABEL_TOPICS:
        if re.search(pat, label):
            add(ts)
    # endorsement-graph provenance: which seed CLASSES link this domain.
    prov_map = {
        "federal-gov": "federal government", "health": "health",
        "reference-cultural": "cultural reference",
        "tech-standards": "technical standards",
        "universities-research": "research", "news": "news",
    }
    for c in (by_classes or []):
        if c in prov_map:
            add([prov_map[c]])
    return topics


# ===========================================================================
# 2. NAVIGATION TEMPLATES
#    How an agent queries the source. Priority:
#      (1) a curated KNOWN_NAV table for top sources (hand-verified),
#      (2) a per-class / per-platform heuristic default,
#      (3) the universal fallback: the conventional ?q= / ?s= search path.
#    The card carries {search_url_template, query_format, method, opensearch,
#    sitemap, api}. `{q}` is the URL-encoded query slot the agent fills.
# ===========================================================================

# Curated, hand-verified navigation for the global news / paywalled head and a
# few top Swiss sources. These are the "manual for top sources" tier of the
# template-derivation strategy (the others are sitemap / OpenSearch / pattern).
KNOWN_NAV = {
    "nytimes.com": dict(
        search_url_template="https://www.nytimes.com/search?query={q}",
        query_format="?query=<urlencoded terms>",
        sitemap="https://www.nytimes.com/sitemaps/new/news.xml.gz",
        api="https://developer.nytimes.com/apis (Article Search API, key required)",
    ),
    "theguardian.com": dict(
        search_url_template="https://www.theguardian.com/search?q={q}",
        query_format="?q=<urlencoded terms>",
        sitemap="https://www.theguardian.com/sitemaps/news.xml",
        api="https://open-platform.theguardian.com/ (Content API, free key)",
    ),
    "bloomberg.com": dict(
        search_url_template="https://www.bloomberg.com/search?query={q}",
        query_format="?query=<urlencoded terms>",
        api=None,
    ),
    "techcrunch.com": dict(
        search_url_template="https://techcrunch.com/?s={q}",
        query_format="?s=<urlencoded terms> (WordPress search)",
        sitemap="https://techcrunch.com/sitemap.xml",
    ),
    "ft.com": dict(
        search_url_template="https://www.ft.com/search?q={q}",
        query_format="?q=<urlencoded terms>",
    ),
    "wsj.com": dict(
        search_url_template="https://www.wsj.com/search?query={q}",
        query_format="?query=<urlencoded terms>",
    ),
    "economist.com": dict(
        search_url_template="https://www.economist.com/search?q={q}",
        query_format="?q=<urlencoded terms>",
    ),
    "reuters.com": dict(
        search_url_template="https://www.reuters.com/site-search/?query={q}",
        query_format="?query=<urlencoded terms>",
    ),
    "washingtonpost.com": dict(
        search_url_template="https://www.washingtonpost.com/search/?query={q}",
        query_format="?query=<urlencoded terms>",
    ),
    "wired.com": dict(
        search_url_template="https://www.wired.com/search/?q={q}",
        query_format="?q=<urlencoded terms>",
    ),
    "arstechnica.com": dict(
        search_url_template="https://arstechnica.com/search/?q={q}",
        query_format="?q=<urlencoded terms>",
    ),
    "nature.com": dict(
        search_url_template="https://www.nature.com/search?q={q}",
        query_format="?q=<urlencoded terms>",
        api="https://dev.springernature.com/ (Springer Nature API, key required)",
    ),
    # Swiss head with known search shapes.
    "srf.ch": dict(
        search_url_template="https://www.srf.ch/suche?q={q}",
        query_format="?q=<urlencoded terms>",
    ),
    "nzz.ch": dict(
        search_url_template="https://www.nzz.ch/suche?q={q}",
        query_format="?q=<urlencoded terms>",
    ),
    "swisscovery.ch": dict(
        search_url_template="https://swisscovery.slsp.ch/discovery/search?query=any,contains,{q}&vid=41SLSP_NETWORK:VU1_UNION",
        query_format="Primo discovery: query=any,contains,<terms>",
    ),
    "doi.org": dict(
        search_url_template="https://doi.org/{q}",
        query_format="resolver: append a DOI (10.xxxx/...) directly; not full-text search",
        api="https://api.crossref.org/works?query=<terms> (Crossref metadata search)",
    ),
}


def navigation_for(domain: str, disc_class: str) -> dict:
    host = "www." + domain if not domain.startswith("www.") else domain
    base = {
        "search_url_template": None,
        "query_format": None,
        "method": "GET",
        "opensearch": None,   # filled by a live probe (probe_opensearch) when run
        "sitemap": None,
        "api": None,
        "derivation": None,
    }
    if domain in KNOWN_NAV:
        base.update(KNOWN_NAV[domain])
        base["derivation"] = "curated"
        # default sitemap guess if the curated entry didn't set one
        if not base.get("sitemap"):
            base["sitemap"] = f"https://{domain}/sitemap.xml"
        return base
    # Universal fallback: the conventional ?q= search path + the conventional
    # sitemap location. This is a *candidate* the agent can try; a live probe
    # (OpenSearch description / sitemap HEAD) upgrades it to verified at ingest.
    base["search_url_template"] = f"https://{host}/?q={{q}}"
    base["query_format"] = "?q=<urlencoded terms> (conventional; verify per site)"
    base["sitemap"] = f"https://{domain}/sitemap.xml"
    base["derivation"] = "pattern-default"
    return base


# ===========================================================================
# 3. THE RUBRIC — deep-index vs card (machine-readable decision table)
#    deep-index a source  <=>  open-licensed AND fetchable AND stable AND
#                              passage-dense.  Else it is a CARD.
#    A source can be BOTH (index the open core; card the paywalled/live part).
#    The card is the universal floor; deep-indexing is the earned upgrade.
#    See the decision table in notes/adr-source-authority-layer.md.
# ===========================================================================

# Domains we KNOW are paywalled or hostile-to-index -> card-only. This is a
# CURATED, hand-verified signal (hard paywalls / bot-hostile heads), NEVER a
# class prior. Paywall membership must not be inferred from `disc_class`: per
# ADR-049 ruling #2 (Tim, 2026-07-17) the operative question is "did the owner
# serve the content to the crawler", already resolved live per source as
# `search_ok` (ADR-024), not guessed from a class at build time. A live
# paywall-intent probe (schema.org isAccessibleForFree, Piano/Tinypass/Poool
# fingerprints) is the future upgrade for the long tail; this curated set is the
# interim floor. (nzz.ch verified 2026-07-17: root returns HTTP 402 Payment
# Required — a literal wall.)
PAYWALLED = {
    # --- global hard / soft-metered heads (curated) ---
    "nytimes.com", "ft.com", "wsj.com", "economist.com", "bloomberg.com",
    "washingtonpost.com",
    "reuters.com",           # metered since Nov-2021 (blocked our fetch; known meter)
    # --- Swiss subscription / metered papers (per-ARTICLE probe, 2026-07-17) ---
    "nzz.ch",                # HTTP 402 Payment Required at root
    "letemps.ch",            # hard subscription (60 paywall/s-abonner markers)
    "tagesanzeiger.ch",      # Tamedia metered (login/paywall redirect-loop)
    "bazonline.ch",          # Tamedia metered (redirect-loop)
    "landbote.ch",           # Tamedia metered
    "handelszeitung.ch",     # Ringier; article redirects to bilanz.ch Abo wall
    # CH Media regional family — freemium meter ("Exklusiv für Abonnenten"):
    # luzernerzeitung + tagblatt article-probed WALLED; bzbasel + badenertagblatt
    # carry abo badges; the whole platform meters, so the family is paywalled.
    "aargauerzeitung.ch", "luzernerzeitung.ch", "tagblatt.ch", "bzbasel.ch",
    "badenertagblatt.ch", "solothurnerzeitung.ch",
    # --- UNVERIFIED (our crawler got HTTP 403; NO article evidence) — held
    #     paywalled conservatively per "fail toward the flag" (a false "free"
    #     misleads the agent; a false "paywalled" is merely cautious). blick.ch
    #     runs a Blick+ tier; avisdexperts/onlinereports are LIKELY free but
    #     unconfirmed. The live paywall-intent probe should re-test and free the
    #     genuinely-open ones. ---
    "blick.ch", "avisdexperts.ch", "onlinereports.ch",
}

# Public-service broadcasters / public media — FREE at point of use (licence-fee
# or tax funded), openly readable, welcoming search indexing. They are NOT
# paywalled and must NEVER be swept into the paywalled set by a news-class prior.
# This carve-out is the fix for the 2026-07-17 defect: the old
# `paywalled = ... or cls == "news"` force-carded EVERY news source as paywalled,
# mis-flagging Swiss public broadcasters SRF/RTS/RSI (and would mis-flag BBC,
# ARD, ZDF, France TV, ORF, ...). Membership is an explicit, hand-verified
# allowlist (free-to-read confirmed from source: srf.ch/rsi.ch fetched free;
# rts.ch is the SRG SSR French sibling, licence-fee funded / free). Extensible to
# any national public broadcaster.
# NB: at P0 these stay disposition="card" (news is not yet in `stable`); a
# license/robots probe promotes them to deep-index per
# specs/adr-source-authority-layer.md §6.3 (still open) — this fix corrects the
# paywalled FLAG, deliberately not that open policy question.
PUBLIC_BROADCASTER = {
    # Switzerland (SRG SSR — Serafe licence-fee funded, free to read)
    "srf.ch", "rts.ch", "rsi.ch", "rtr.ch", "swissinfo.ch",
    # Europe (public-service media)
    "bbc.com", "bbc.co.uk", "ard.de", "zdf.de", "tagesschau.de", "dw.com",
    "francetvinfo.fr", "france24.com", "radiofrance.fr", "orf.at", "rai.it",
    "rte.ie", "nrk.no", "yle.fi", "svt.se", "dr.dk", "rtve.es", "npo.nl",
    # Anglosphere public media
    "npr.org", "pbs.org", "cbc.ca", "abc.net.au",
}

# Sources with an open, fetchable, dense core already being deep-indexed
# elsewhere in AW (or trivially so) -> BOTH (index core, card live/extra).
KNOWN_OPEN_CORE = {"doi.org", "nature.com", "theguardian.com"}


def disposition_for(card: dict) -> dict:
    """Apply the rubric. Returns {disposition, signals, why}.

    Signals (heuristic at P0; an ingest pipeline replaces these with real probes):
      open_license  : license is permissive / open enough to mirror passages
      fetchable     : robots + consent allow CC/dump fetch (policy-(a))
      stable        : URLs are durable (not session/JS-only)
      passage_dense : pages carry substantive extractable text (not app shell)
    """
    domain = card["domain"]
    cls = card.get("disc_class", "other")

    # Paywalled is a CURATED signal, never a class prior (ADR-049 ruling #2).
    # Public broadcasters are free even if a stray denylist entry said otherwise,
    # so the explicit allowlist carve-out always wins.
    paywalled = (domain in PAYWALLED) and (domain not in PUBLIC_BROADCASTER)
    open_core = domain in KNOWN_OPEN_CORE

    # Heuristic signal estimates for the starter set. The real pipeline sets these
    # from: license probe, robots/consent (ADR-009), URL-shape sample, and a
    # text-density sample over a few fetched pages.
    open_license = not paywalled  # gov/edu/health Swiss pages are largely OGD/open
    fetchable = not paywalled     # paywalled => not policy-(a)-fetchable
    stable = cls in ("gov", "edu", "ref", "health", "tech") or open_core
    passage_dense = cls in ("gov", "edu", "ref", "health", "tech", "news")

    signals = dict(open_license=open_license, fetchable=fetchable,
                   stable=stable, passage_dense=passage_dense)

    deep_ok = open_license and fetchable and stable and passage_dense
    if open_core and not deep_ok:
        deep_ok = True  # known open core overrides (e.g. Guardian open API)

    if deep_ok and paywalled:
        disp, why = "both", "open/fetchable core, but paywalled/live part is card-only"
    elif deep_ok:
        disp, why = "deep-index", "open + fetchable + stable + passage-dense (earned upgrade)"
    elif paywalled:
        disp, why = "card", "paywalled/live: cannot mirror — index the source + how to query it"
    else:
        disp, why = "card", "not all of {open,fetchable,stable,dense} met — card is the floor"
    return dict(disposition=disp, signals=signals, why=why)


# ===========================================================================
# 4. ENDORSEMENT / TRUST
# ===========================================================================

# class-level trust floor (mirrors the discovery engine's class-trust weights).
CLASS_TRUST = {"gov": 0.85, "edu": 0.75, "health": 0.75, "ref": 0.70,
               "tech": 0.70, "news": 0.55, "other": 0.45}
# freshness class: how fast the authoritative content changes (a navigation hint).
CLASS_FRESHNESS = {"news": "live", "gov": "slow", "edu": "slow", "health": "medium",
                   "ref": "static", "tech": "medium", "other": "medium"}


def trust_tier(card: dict) -> float:
    return round(CLASS_TRUST.get(card.get("disc_class", "other"), 0.45), 2)


# ===========================================================================
# CARD ASSEMBLY
# ===========================================================================

def card_text(card: dict) -> str:
    """The embedding surface: authoritative_for topics + a short description.
    NOT the page content — the card embeds what the source is an authority FOR.
    A curated card (task 2659) also carries `names` (the host's name per language); they lead the
    text. Discovery cards have none, so their text is byte-identical to before."""
    topics = ", ".join(card.get("authoritative_for", []))
    juris = card.get("jurisdiction") or "global"
    names = " / ".join(v for v in (card.get("names") or {}).values() if v)
    head = f"{card['domain']} ({names})" if names else card["domain"]
    return (f"{head} — authoritative source for: {topics}. "
            f"Jurisdiction: {juris}. {card.get('description', '')}").strip()


def build_swiss_cards(discovery_path: str) -> list[dict]:
    data = json.load(open(os.path.expanduser(discovery_path)))
    cards = []
    for row in data:
        if not row.get("swiss"):
            continue
        domain = row["domain"]
        cls = row.get("class", "other")
        topics = derive_topics(domain, cls, row.get("by_classes"))
        card = {
            "domain": domain,
            "disc_class": cls,
            "authoritative_for": topics,
            "endorsement": {
                "source": "discovery-graph",
                "score": round(row.get("score", 0.0), 2),
                "endorse_score": round(row.get("endorse_score", 0.0), 2),
                "by_classes": row.get("by_classes", []),
                "why": row.get("why", ""),
            },
            "jurisdiction": country_for_domain(domain) or "CH",
            "license": "OGD/open (Swiss public-sector default; verify per source)",
            "freshness_class": CLASS_FRESHNESS.get(cls, "medium"),
            "set": "swiss-discovery",
            "description": "",
        }
        card["navigation"] = navigation_for(domain, cls)
        card["trust_tier"] = trust_tier(card)
        card.update(disposition_for(card))
        cards.append(card)
    return cards


# Curated HN-endorsed global news / paywalled head. endorsement.score is an
# illustrative HN-front-page weight (the real pipeline pulls it from
# ~/aw-data/hn_candidates.json domain aggregates). topics are hand-written.
HN_HEAD = [
    ("nytimes.com", "news", ["US national news", "world news", "politics",
        "technology", "investigative journalism"], 9800, "paywalled US daily"),
    ("theguardian.com", "news", ["UK news", "world news", "politics", "environment",
        "technology", "open journalism"], 8700, "UK daily, open Content API"),
    ("bloomberg.com", "news", ["financial markets", "business", "economics",
        "technology", "company data"], 6400, "financial news, paywalled"),
    ("techcrunch.com", "tech", ["startups", "venture capital", "technology news",
        "product launches"], 5200, "startup/tech news"),
    ("ft.com", "news", ["financial markets", "economics", "business", "world news"],
        4100, "Financial Times, paywalled"),
    ("wsj.com", "news", ["business", "finance", "US news", "markets"], 3900,
        "Wall Street Journal, paywalled"),
    ("economist.com", "news", ["world affairs", "economics", "politics", "analysis"],
        3600, "weekly, paywalled"),
    ("reuters.com", "news", ["world news", "breaking news", "markets", "business"],
        3500, "wire service"),
    ("washingtonpost.com", "news", ["US politics", "national news", "investigations"],
        3300, "US daily, paywalled"),
    ("wired.com", "tech", ["technology", "science", "culture", "security"], 3100,
        "technology magazine"),
    ("arstechnica.com", "tech", ["technology", "science", "computing", "policy"],
        2900, "deep tech reporting"),
    ("nature.com", "edu", ["science", "research", "peer-reviewed journals", "biology",
        "physics"], 4800, "scientific journals, mixed open/paywall"),
    ("doi.org", "ref", ["scholarly articles", "DOI resolution", "research metadata",
        "citations"], 2533, "DOI resolver (global registry)"),
]

US_HEAD = {"nytimes.com", "wsj.com", "washingtonpost.com", "bloomberg.com",
           "wired.com", "arstechnica.com"}


def build_hn_head_cards() -> list[dict]:
    cards = []
    for domain, cls, topics, score, desc in HN_HEAD:
        if domain in US_HEAD:
            juris = "US"
        elif domain == "doi.org":
            juris = "INTL"
        else:
            juris = country_for_domain(domain)
        card = {
            "domain": domain,
            "disc_class": cls,
            "authoritative_for": topics,
            "endorsement": {"source": "hn-endorsed", "score": float(score),
                            "endorse_score": float(score), "by_classes": [cls],
                            "why": "HN front-page endorsement (illustrative weight)"},
            "jurisdiction": juris,
            "license": "proprietary / all-rights-reserved (card-only)"
                       if domain in PAYWALLED else "mixed (verify per source)",
            "freshness_class": CLASS_FRESHNESS.get(cls, "medium"),
            "set": "hn-head",
            "description": desc,
        }
        card["navigation"] = navigation_for(domain, cls)
        card["trust_tier"] = trust_tier(card)
        card.update(disposition_for(card))
        cards.append(card)
    return cards


# ===========================================================================
# COMMANDS: build / embed / search (ported from the former `build_source_cards` pipeline
# script, task 1235; `cyt ops source-cards {build,embed,search}`)
# ===========================================================================

CLI_USAGE = """Build / embed / search the SOURCE-AUTHORITY + NAVIGATION-AFFORDANCE layer ("source cards").

  1. build  — assemble cards for the P0 starter set (the Swiss discovery domains plus a curated
              head of HN-endorsed news / paywalled global sources) into a cards.jsonl; pure,
              offline, reproducible. Each card gets topics, an endorsement weight, a jurisdiction,
              a navigation template and an index_disposition from the RUBRIC (`disposition_for`).
  2. embed  — embed the card text into a SEPARATE Qdrant collection `source_cards` (config B:
              arctic-l-v2 -> MRL-128 + scalar int8 + bm25, on_disk). NEVER blended into the
              grounding passage pool (ADR-021 / ADR-017).
  3. search — demo retrieval: a query -> the most relevant source cards with navigation hints.

USAGE
  cyt ops source-cards build  --discovery ~/aw-data/discovery_queue.json --out ~/aw-data/source_cards/cards.jsonl
  cyt ops source-cards merge  --base ~/aw-data/source_cards/cards.jsonl --out merged.jsonl   # + curated authority cards
  cyt ops source-cards embed  --in ~/aw-data/source_cards/cards.jsonl --collection source_cards --qdrant-url http://localhost:6333
  cyt ops source-cards reembed --from-collection source_cards --recreate   # same cards, served width
  cyt ops source-cards search --collection source_cards --qdrant-url http://localhost:6333 --query "..."
"""

DENSE_VEC = "dense"
SPARSE_BM25 = "bm25"

# ---- curated authority cards (task 2659; ADR-033/041 register-first) ----------------------------
CURATED_PATH = pathlib.Path(__file__).parent / "data" / "authority_cards_curated.yaml"
from cytir.core.common import TIER_TRUST          # the ONE home of the allowlist tier -> trust map
CURATED_REQUIRED = ("domain", "tier", "jurisdiction", "name", "authoritative_for", "description")
CURATED_LANGS = ("de", "fr", "it", "en")


def load_curated(path=None) -> list[dict]:
    """Parse + validate the curated file. Raises ValueError naming the first bad card."""
    import yaml
    doc = yaml.safe_load(open(path or CURATED_PATH, encoding="utf-8"))
    cards, seen = doc.get("cards") or [], set()
    for c in cards:
        dom = c.get("domain", "?")
        miss = [k for k in CURATED_REQUIRED if not c.get(k)]
        if miss:
            raise ValueError(f"curated card {dom}: missing {miss}")
        if c["tier"] not in TIER_TRUST:
            raise ValueError(f"curated card {dom}: tier {c['tier']!r} not in {sorted(TIER_TRUST)}")
        if set(c["name"]) != set(CURATED_LANGS) or not all(c["name"].values()):
            raise ValueError(f"curated card {dom}: name needs {CURATED_LANGS}")
        if dom != _norm_domain(dom) or dom in seen:
            raise ValueError(f"curated card {dom}: domain not bare/lowercase, or duplicated")
        seen.add(dom)
    return cards


def _norm_domain(d: str) -> str:
    d = (d or "").strip().lower()
    return d[4:] if d.startswith("www.") else d


def curated_card(entry: dict) -> dict:
    """One curated entry -> a full card. Disposition is read from the consent registry, never written
    in the file: a recorded decline -> `consent-declined`, else `card` (the floor)."""
    from cytir.consent import block
    dom = entry["domain"]
    juris, _, region = entry["jurisdiction"].partition("-")
    cls = "tech" if juris == "INTL" else "gov"
    hit = block.lookup(dom)
    declined = bool(hit) and hit[1].get("state") in ("declined", "unknown")
    nav = {"search_url_template": None, "query_format": None, "method": "GET", "opensearch": None,
           "sitemap": None, "api": None, "home": entry.get("home") or f"https://{dom}/",
           "derivation": "curated", "confidence": "home-only"}
    if (entry.get("navigation") or {}).get("search_url_template"):
        nav.update(entry["navigation"])
        nav.pop("confidence")                       # a curated search template is "verified"
    en = entry["name"]["en"]
    card = {
        "domain": dom, "name": en, "names": dict(entry["name"]), "disc_class": cls,
        "authoritative_for": list(entry["authoritative_for"]),
        "endorsement": {"source": "curated-register", "score": 1.0, "endorse_score": 1.0,
                        "by_classes": [cls], "why": f"curated authority register, allowlist tier {entry['tier']}"},
        "jurisdiction": juris, "license": "unverified (curated card; not mirrored)",
        "freshness_class": "medium" if cls == "tech" else "slow",
        "set": "curated-authority", "source": "curated", "tier": entry["tier"],
        "description": entry["description"], "navigation": nav,
        "trust_tier": TIER_TRUST[entry["tier"]],
        "disposition": "consent-declined" if declined else "card",
        "signals": {"open_license": False, "fetchable": not declined, "stable": True, "passage_dense": True},
        "why": ("consent-declined (registry): " + hit[1].get("basis", "")[:200]) if declined
               else f"curated authority (tier {entry['tier']}): {entry['description']}"[:500],
    }
    if region:
        card["region"] = region
    if entry.get("train_ok") is False and not declined:   # train-only decline (ADR-024 am.5): never blocks
        card["consent"] = {"search_ok": True, "train_ok": False, "checked_at": entry.get("train_ok_checked_at")}
    if declined:
        card["consent"] = {"search_ok": hit[1].get("search_ok", False),
                           "train_ok": hit[1].get("train_ok", False), "checked_at": hit[1].get("decided_on")}
    return card


def merge_cards(discovery: list[dict], curated: list[dict]) -> list[dict]:
    """Discovery cards + curated cards; a curated card WINS a host collision (www. folded). A discovery
    card that is `consent-declined` keeps that disposition and its consent record (stricter wins)."""
    by_host = {_norm_domain(c["domain"]): c for c in curated}
    out, taken = [], set()
    for d in discovery:
        k = _norm_domain(d["domain"])
        if k not in by_host:
            out.append(d)
            continue
        c = dict(by_host[k])
        # stricter wins, but a TRAIN-ONLY decline (search_ok true) never blocks (ADR-024 am.5)
        if (d.get("disposition") == "consent-declined" and c["disposition"] != "consent-declined"
                and (d.get("consent") or {}).get("search_ok") is not True):
            for f in ("disposition", "consent", "why", "signals"):
                if f in d:
                    c[f] = d[f]
        out.append(c)
        taken.add(k)
    out += [c for k, c in by_host.items() if k not in taken]
    return out


def cmd_merge(args):
    base = [json.loads(l) for l in open(os.path.expanduser(args.base)) if l.strip()]
    cur = [curated_card(e) for e in load_curated(args.curated)]
    merged = merge_cards(base, cur)
    out = pathlib.Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for c in merged:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    won = len(base) + len(cur) - len(merged)
    print(f"WROTE {len(merged)} cards -> {out}  (base {len(base)}, curated {len(cur)}, "
          f"host collisions won by curated {won}, new hosts {len(cur) - won})")


def cards_dim() -> int:
    """The dense width of the cards: the SEALED served profile's (`SERVED.dense_dim`), never the env
    default (task 2641: a hard-coded 128 left /find_authority at 503 after the aw_e10 flip to 256).
    `AW_MRL_DIM`, when set, must agree: the server embeds queries at that width, so a differing
    value means a build that would not match its own queries -> refuse, naming both."""
    from cytir.core.serving_profile import SERVED
    env = os.environ.get("AW_MRL_DIM")
    if env and int(env) != SERVED.dense_dim:
        raise SystemExit(f"source-cards: CYTIR_MRL_DIM={env} differs from the served profile "
                         f"SERVED.dense_dim={SERVED.dense_dim}; refusing to build cards at either "
                         "until they agree")
    return SERVED.dense_dim

def _source_cards_profile():
    from cytir.core import serving_profile as SP
    return SP.SOURCE_CARDS


def cmd_build(args):
    cards = build_swiss_cards(args.discovery) + build_hn_head_cards()
    out = pathlib.Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    # summary
    from collections import Counter
    by_set = Counter(c["set"] for c in cards)
    by_disp = Counter(c["disposition"] for c in cards)
    by_juris = Counter(c["jurisdiction"] or "None" for c in cards)
    print(f"WROTE {len(cards)} source cards -> {out}")
    print("  by set        :", dict(by_set))
    print("  by disposition:", dict(by_disp))
    print("  by jurisdiction:", dict(by_juris))
    deep = [c["domain"] for c in cards if c["disposition"] in ("deep-index", "both")][:8]
    print("  sample deep-index/both:", deep)


# ---- embed / search reuse the locked config-B stack (index_bakeoff) ----------



def _load_stack():
    # No import fallback for the tokenizer (it was a silent second tokenizer; the
    # incident ledger names it): the package's `bm25_tokens` or a loud ImportError.
    from qdrant_client import models  # noqa
    from cytir.core.qdrant import qdrant_connect
    from cytir.core.bm25_tokenizer import bm25_tokens
    from cytir.core.ids import record_point_id as point_id
    from cytir.core.serving_config import mrl_trunc
    from cytir.embed.model import load_embedder
    return qdrant_connect, models, load_embedder, mrl_trunc, point_id, bm25_tokens


def _bm25_vector(tokens, models):
    from collections import Counter
    c = Counter(tokens)
    if not c:
        return None
    h = lambda t: (hash(t) & 0x7FFFFFFF)
    return models.SparseVector(indices=[h(t) for t in c],
                               values=[float(v) for v in c.values()])


def ensure_source_cards_collection(client, models, name, dim, recreate):
    """Create the `source_cards` collection from `_source_cards_profile()` (the one creator,
    `cytir.core.serving_profile.create_collection`; `models` is kept for the old signature)."""
    from cytir.core import serving_profile as SP
    SP.create_collection(client, name, _source_cards_profile().with_(dense_dim=dim),
                         on_exists="recreate" if recreate else "keep", index_failure="warn")


def cmd_embed(args):
    cards = [json.loads(l) for l in open(os.path.expanduser(args.input)) if l.strip()]
    _embed_cards(args, cards)


def _existing_count(client, name):
    try:
        return client.count(name, exact=True).count
    except Exception:  # collection absent
        return None


def _embed_cards(args, cards):
    connect, models, load_embedder, mrl_trunc, point_id, bm25_tokens = _load_stack()
    client = connect(args.qdrant_url, timeout=600)
    have = _existing_count(client, args.collection) if args.recreate else None
    if have is not None and have != len(cards) and not getattr(args, "allow_count_change", False):
        raise SystemExit(f"source-cards: --recreate would replace {have} cards in {args.collection} "
                         f"with {len(cards)}; pass --allow-count-change if that is intended")
    ensure_source_cards_collection(client, models, args.collection, cards_dim(), args.recreate)

    encode_docs, _, _ = load_embedder("arctic-l-v2", use_fp16=True, max_seq=512)
    texts = [card_text(c) for c in cards]
    points = []
    B = args.batch
    for i in range(0, len(texts), B):
        chunk = texts[i:i + B]
        dense, _ = encode_docs(chunk, batch=B)
        dense = mrl_trunc(dense, cards_dim())
        for j, c in enumerate(cards[i:i + B]):
            payload = {k: v for k, v in c.items() if k != "text"}
            vec = {DENSE_VEC: dense[j]}
            sv = _bm25_vector(bm25_tokens(texts[i + j]), models)
            if sv is not None:
                vec[SPARSE_BM25] = sv
            points.append(models.PointStruct(
                id=point_id({"url": "card:" + c["domain"]}),
                vector=vec, payload=payload))
        print(f"  embedded {min(i+B, len(texts))}/{len(texts)}")
    client.upsert(collection_name=args.collection, points=points, wait=True)
    cnt = client.count(args.collection, exact=True).count
    print(f"UPSERT done. {args.collection} now has {cnt} source cards.")


MAX_CARDS = 100_000  # cards are thousands; the main index is millions


def _scroll_payloads(client, name):
    out, off = [], None
    while True:
        pts, off = client.scroll(name, limit=500, offset=off, with_payload=True, with_vectors=False)
        out += [p.payload for p in pts]
        if off is None:
            return out


def cmd_reembed(args):
    """Re-embed a collection from its OWN payloads (the card text derives from payload fields), at
    `cards_dim()`. Fixes a width drift with the card set unchanged. Snapshot the collection first
    (`POST /collections/NAME/snapshots`); `--recreate` drops and rebuilds it under the same name."""
    connect = _load_stack()[0]
    name = args.from_collection
    cards_name = os.environ.get("AW_CARDS_COLLECTION", "source_cards")
    if name != cards_name and not name.startswith("source_cards"):
        raise SystemExit(f"source-cards: reembed reads only the cards collection ({cards_name} or "
                         f"source_cards*), not {name!r}; the main index is never scrolled here")
    client = connect(args.qdrant_url, timeout=600)
    n = _existing_count(client, name)
    if n is not None and n > MAX_CARDS:
        raise SystemExit(f"source-cards: {name} holds {n} points, over the {MAX_CARDS} cap for a cards "
                         "collection; refusing to scroll it")
    cards = _scroll_payloads(client, name)
    if not cards:
        raise SystemExit(f"source-cards: {args.from_collection} holds no cards")
    args.collection = args.collection or args.from_collection
    _embed_cards(args, cards)


def cmd_search(args):
    connect, models, load_embedder, mrl_trunc, point_id, bm25_tokens = _load_stack()
    client = connect(args.qdrant_url, timeout=600)
    _, encode_queries, _ = load_embedder("arctic-l-v2", use_fp16=True, max_seq=512)
    qv, _ = encode_queries([args.query], batch=1)
    qv = mrl_trunc(qv, cards_dim())[0]
    flt = None
    if args.jurisdiction:
        flt = models.Filter(must=[models.FieldCondition(
            key="jurisdiction", match=models.MatchValue(value=args.jurisdiction))])
    res = client.query_points(
        collection_name=args.collection, query=qv, using=DENSE_VEC,
        limit=args.k, with_payload=True, query_filter=flt,
        search_params=models.SearchParams(quantization=models.QuantizationSearchParams(
            rescore=True, oversampling=3.0))).points
    out = []
    for p in res:
        c = p.payload
        nav = c.get("navigation", {})
        out.append({
            "score": round(p.score, 4),
            "authority": c["domain"],
            "authoritative_for": c.get("authoritative_for", [])[:6],
            "jurisdiction": c.get("jurisdiction"),
            "trust_tier": c.get("trust_tier"),
            "disposition": c.get("disposition"),
            "freshness": c.get("freshness_class"),
            "navigate": {
                "search_url": (nav.get("search_url_template") or "").replace(
                    "{q}", args.query.replace(" ", "+")),
                "query_format": nav.get("query_format"),
                "sitemap": nav.get("sitemap"),
                "api": nav.get("api"),
            },
        })
    print(json.dumps({"query": args.query, "results": out}, ensure_ascii=False, indent=2))

def main(argv=None, prog="cyt ops source-cards"):
    ap = argparse.ArgumentParser(prog=prog, description=CLI_USAGE,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="assemble source cards (offline)")
    b.add_argument("--discovery", default="~/aw-data/discovery_queue.json")
    b.add_argument("--out", default="~/aw-data/source_cards/cards.jsonl")
    b.set_defaults(func=cmd_build)

    m = sub.add_parser("merge", help="merge the curated authority cards into a cards.jsonl (offline)")
    m.add_argument("--base", required=True, help="discovery-built cards.jsonl")
    m.add_argument("--out", required=True)
    m.add_argument("--curated", default=None, help=f"default: {CURATED_PATH}")
    m.set_defaults(func=cmd_merge)

    e = sub.add_parser("embed", help="embed cards into the source_cards collection")
    e.add_argument("--in", dest="input", required=True)
    e.add_argument("--collection", default="source_cards")
    e.add_argument("--qdrant-url", default="http://localhost:6333")
    e.add_argument("--batch", type=int, default=128)
    e.add_argument("--recreate", action="store_true")
    e.add_argument("--allow-count-change", action="store_true",
                   help="allow --recreate to change the number of cards in the collection")
    e.set_defaults(func=cmd_embed)

    r = sub.add_parser("reembed", help="re-embed a collection from its own payloads at the served width")
    r.add_argument("--from-collection", required=True)
    r.add_argument("--collection", default=None, help="target (default: same as --from-collection)")
    r.add_argument("--qdrant-url", default="http://localhost:6333")
    r.add_argument("--batch", type=int, default=128)
    r.add_argument("--recreate", action="store_true")
    r.set_defaults(func=cmd_reembed)

    s = sub.add_parser("search", help="demo retrieval of source cards")
    s.add_argument("--query", required=True)
    s.add_argument("--collection", default="source_cards")
    s.add_argument("--qdrant-url", default="http://localhost:6333")
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--jurisdiction", default=None, help="filter by CH/FR/US/INTL")
    s.set_defaults(func=cmd_search)

    args = ap.parse_args(argv)
    # task 864 (review 869 S1): from an aw_e10 T0 to the flip only the drain writes the twin
    from cytir.index import twin_write_hold as _TWH
    _TWH.refuse_cli_write(args, tool='cyt ops source-cards')
    args.func(args)

    return 0
