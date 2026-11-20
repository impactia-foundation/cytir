# Moved from the retired `aw_country` module (task 677, task 643 S5); the old path is an alias shim.
"""Country/jurisdiction stamping (the `country` payload facet).

Used by:
  - the country backfill / re-stamp (`backfill_country_facet.py`) that sets the
    `country` payload field per (domain, source_type) group — never a wide write,
  - the go-forward ingest/rebuild stamp (`rebuild_v4.py`),
  - card jurisdiction tagging (`build_source_cards.py`, `consent_gate.py`),
  - (indirectly) serving: the live off-country penalty + country soft-boost read
    the stamped `country` field; they do NOT recompute it, so the stamp is the
    single source of truth.

POLICY (Tim, 2026-07-17) — content-conditioned, not domain-only
---------------------------------------------------------------
Earlier the stamp was a function of the DOMAIN alone (`country_for_domain`): the
`.ch` and `.gov` ccTLD fallbacks stamped a jurisdiction on ANY page under them.
That was harmless while `country` was only a soft *boost*. Once an off-country
*penalty* was added (task 240/254), a demotable stamp on non-location-bound
content became actively harmful: a CH-scoped medical/research query DEMOTED the
28M PubMed corpus (`pubmed.ncbi.nlm.nih.gov` → `*.gov` → US) even though PubMed
is authoritative everywhere.

New rule: **only stamp a jurisdiction on content that is really relevant only to
that location** — local businesses, local administration, and law. Everything
that is authoritative across jurisdictions is `INTL` (never demoted) or `None`
(unclassified; neither boosted nor demoted).

  STAMP a jurisdiction (CH/FR/CA/DE/US …) ONLY for LOCATION-BOUND content:
    - local administration / statistics / tax / geo / central bank / law:
      source_type in official, gov, gov-stat, gov-tax, gov-geo, central-bank,
      case_law, registry_record  ->  the source's national jurisdiction.
    - a registered local business card (source_type == entity_card WITH a
      register_id or LEI)  ->  its country of REGISTRATION. That country is
      captured from the register at ingest and lives in the payload; it is NOT
      derivable from the domain (a Swiss SME on a `.com`), so this module never
      overwrites it — it only decides that a country BELONGS (see KEEP_EXISTING).

  INTL (known cross-jurisdiction, EXEMPT from the off-country penalty):
    - standards bodies: W3C, IETF/RFC, IANA.
    - Wikipedia (all language editions — a French Wikipedia article is not a
      French *jurisdiction* source).
    - global research / reference authoritative everywhere: PubMed, arXiv, and
      general `source_type == reference` not tied to one jurisdiction.

  None (unclassified) for everything else general we have not classified:
    news, docs, tech, health, blog, community-qa, hn-discussion, edu, misc —
    and an entity_card WITHOUT a register_id (YC/category/author cards, not a
    registered local business). None is left UNSTAMPED: a country selection
    neither boosts nor demotes it.

Net effect: the off-country penalty then only demotes genuinely-LOCAL off-country
content (a foreign local business / gov / law page on a CH query), never global
research or reference.

PRECEDENCE
----------
0. Explicit per-publisher declarations transcribed from a ratified source pack
   (`_PUBLISHER_COUNTRY`) win over every heuristic below: a curated jurisdiction
   is a fact about the publisher, not an inference from its domain/source_type.
1. INTL domains (standards bodies, Wikipedia, PubMed, arXiv, and the task-261
   `.int`/IGO treaty bodies + clinicaltrials.gov) win over everything, regardless
   of source_type or TLD.
2. `source_type == reference` -> INTL (authoritative everywhere).
3. LOCATION_BOUND source_types -> national jurisdiction, resolved in the task-261
   signal order (spec country-stamping-coverage.md §, ADR-055):
     supra EU (europa.eu) -> explicit national hosts -> Wikidata P17 map
     (exact, then registrable-parent) -> restricted-2LD (gov.uk/…) ->
     nexus-TLD (.swiss/.ca/.no/.ie/.au) -> `.ch`/`.fr`/`.de` ccTLD fallback -> None.
   These fallbacks apply ONLY to location-bound types, never to general content.
4. entity_card -> KEEP_EXISTING (register country) iff register_id, else None.
5. everything else -> None.

The P17 map + restricted-2LD + nexus-TLD + supra tables (below) are the task-261
convergence signal stack ported from the standalone apply_country_convergence.py
so the next rebuild stamps them for free instead of losing the convergence.

The functions are pure and deterministic so the backfill is idempotent: a
re-stamp produces exactly the same assignment and can be diffed against the
already-present `country` values to find what still needs changing.
"""
from __future__ import annotations

__all__ = [
    "country_for_record", "country_for_domain", "country_for_url",
    "is_location_bound", "COUNTRIES", "LOCATION_BOUND_TYPES",
    "LOCATION_BOUND_PREFIXES", "KEEP_EXISTING",
    "P17_COUNTRY", "RESTRICTED_2LD", "NEXUS_TLD", "SUPRA_EU", "MUST_EXCLUDE",
]

# The selectable jurisdictions (UI dropdown + validation). INTL is a real value
# in the index but is not a user-selectable jurisdiction — it is the
# "never demote these" class. <none> is the absence of the param.
COUNTRIES = ["CH", "FR", "CA", "DE", "US"]

# Source types whose content is inherently location-bound: local administration
# / statistics / tax / geo, the national central bank, and law / case law /
# register records. These (and ONLY these) get a national jurisdiction stamp
# from the domain, including via the ccTLD/.gov fallbacks.
LOCATION_BOUND_TYPES = frozenset({
    "official", "gov", "gov-stat", "gov-tax", "gov-geo",
    "central-bank", "case_law", "registry_record",
})

# Prefix families that are location-bound BY CONSTRUCTION: every `gov-*` /
# `official-*` source_type is national administration / law / statistics content.
#
# WHY A PREFIX RULE, NOT AN ENUMERATION (task 259, defect 1 — 2026-07-17):
# the enumeration above was written against the TWIN's 18 source_types, but PROD
# carries 25. The extra prod-only types include `gov-law`, `gov-court`,
# `gov-stats` (note: PLURAL — the set above has the singular `gov-stat`) and
# `gov-health`. Membership-testing the enumeration silently classified those as
# "general" and would have DELETED the jurisdiction stamp from 62 location-bound
# gov/law records — the exact inverse of the policy. The twin dry-run could not
# catch it: those source_types do not exist on the twin at all.
#
# A prefix rule fails SAFE: a newly-introduced `gov-*`/`official-*` type is
# location-bound by default instead of silently becoming un-stamped. The INTL
# domain check still runs FIRST in country_for_record, so a standards body typed
# `official` (w3.org/ietf.org) is still INTL and is never downgraded.
LOCATION_BOUND_PREFIXES = ("gov-", "official-")

# Sentinel returned for a registered entity_card: the country of registration is
# owned by the ingest-time register join and lives in the payload; this module
# must NOT overwrite it from the domain. The backfill treats KEEP_EXISTING as a
# no-op (leave whatever the record already carries).
KEEP_EXISTING = "__KEEP_EXISTING__"


def is_location_bound(source_type: str | None) -> bool:
    """True if `source_type` is location-bound content (national administration,
    statistics, tax, geo, central bank, law/case law, register records) and so
    earns a national jurisdiction stamp from its domain.

    Membership in LOCATION_BOUND_TYPES OR a LOCATION_BOUND_PREFIXES family
    (`gov-*` / `official-*`). Callers MUST use this rather than testing
    `st in LOCATION_BOUND_TYPES` — see LOCATION_BOUND_PREFIXES for why.
    """
    st = (source_type or "").strip().lower()
    if not st:
        return False
    return st in LOCATION_BOUND_TYPES or st.startswith(LOCATION_BOUND_PREFIXES)


def _norm(domain: str | None) -> str:
    if not domain:
        return ""
    d = domain.strip().lower()
    # tolerate a full host with scheme/path or a leading www. — callers should
    # pass the registrable `domain` payload field, but be forgiving.
    if "://" in d:
        d = d.split("://", 1)[1]
    d = d.split("/", 1)[0]
    if d.startswith("www."):
        d = d[4:]
    return d.rstrip(".")


def _host_or_sub(d: str, suffix: str) -> bool:
    # match the domain itself or any subdomain of it
    return d == suffix or d.endswith("." + suffix)


# ---------------------------------------------------------------------------
# EXPLICIT PUBLISHER DECLARATIONS (task 448, 2026-08-27)
# ---------------------------------------------------------------------------
# A news article is not a location-bound source_type: the news lane types its
# passages `reference`, and rule (2) of country_for_record sends every
# `reference` record to INTL. That is correct for PubMed/arXiv and WRONG for a
# national broadcaster — it stamped all 6,357 rows of the 2026-08-27 news pack
# INTL, including four Swiss publishers, which silently disables CH-first
# ranking for the whole news layer.
#
# These four entries are NOT a new judgement. Each is the `country:` field of an
# already-ratified source pack, transcribed:
#
#   swissinfo.ch          -> sources/ch/swissinfo.yaml   `country: CH`
#   srf.ch/rts.ch/rsi.ch  -> sources/ch/ch-media.yaml    `country: CH`
#
# swissinfo.yaml's own notes asked for exactly this ("country override:
# `aw_country.country_for_record()` classifies swissinfo.ch as INTL by default
# ...; consider making this a durable aw_country rule and still not promoted to
# a durable rule as of this pack's authoring"). This is that promotion.
#
# The table is checked BEFORE the INTL/reference heuristics: an explicit,
# board-ratified per-publisher declaration outranks a source_type default.
# Nothing else changes — a publisher with no entry here keeps its old stamp, so
# HN rows and their linked third-party pages stay INTL exactly as before.
_PUBLISHER_COUNTRY = {
    "swissinfo.ch": "CH",   # SWI (SRG SSR international service)
    "srf.ch": "CH",         # SRF  (SRG SSR, DE)
    "rts.ch": "CH",         # RTS  (SRG SSR, FR)
    "rsi.ch": "CH",         # RSI  (SRG SSR, IT)
    "watson.ch": "CH",      # watson (CH Media, DE/FR): sources/ch/watson.yaml `country: CH`
}


def _publisher_country(d: str) -> str | None:
    """Explicit per-publisher jurisdiction from a ratified source pack, or None."""
    for host, country in _PUBLISHER_COUNTRY.items():
        if _host_or_sub(d, host):
            return country
    return None


def _intl_domain(d: str) -> bool:
    """True for domains that are authoritative ACROSS jurisdictions and must be
    INTL regardless of source_type or TLD: standards bodies, Wikipedia, the
    global research/reference corpora (PubMed, arXiv), and the intergovernmental
    treaty organisations (the `.int` registry + a few `.org`-hosted IGOs).

    The `.int` / IGO / `clinicaltrials.gov` additions are the task-261 `intl`
    signal ported to build time (worldmap.json provenance=`intl`): treaty bodies
    (Council of Europe/ECHR/HUDOC, WMO, ICAO, EPPO, EUROCONTROL, WIPO, ECMWF), the
    ILO, and the global trial registry `clinicaltrials.gov` (a `.gov` host whose
    content fails axis-2 — P17 declines to guess where the `.gov`->US rule would
    wrongly say US; spec §4.1). INTL runs FIRST, so listing it here is what keeps
    `clinicaltrials.gov` from being stamped US (ADR-046)."""
    return (
        # standards bodies
        _host_or_sub(d, "w3.org")
        or _host_or_sub(d, "ietf.org")
        or _host_or_sub(d, "rfc-editor.org")
        or _host_or_sub(d, "iana.org")
        # Wikipedia (all language editions)
        or _host_or_sub(d, "wikipedia.org")
        # global research / reference — authoritative everywhere (the whole point
        # of this refactor: PubMed is NOT a US-jurisdiction source)
        or _host_or_sub(d, "ncbi.nlm.nih.gov")   # PubMed / PMC / NCBI
        or _host_or_sub(d, "pubmed.gov")
        or _host_or_sub(d, "arxiv.org")
        # intergovernmental treaty organisations (task-261 `intl` signal): the
        # `.int` registry is reserved for treaty IGOs — coe.int (ECHR/HUDOC),
        # wmo.int, icao.int, eppo.int, eurocontrol.int, wipo.int, ecmwf.int, …
        or d == "int" or d.endswith(".int")
        or _host_or_sub(d, "ilo.org")            # ILO / ILOSTAT
        or _host_or_sub(d, "clinicaltrials.gov")  # global trial registry (.gov exception)
        # IGOs and global authority bodies on non-.int domains (task 399).
        # `ipcc.ch` is the load-bearing case and the reason this block exists:
        # the IPCC is a UN/WMO treaty body that merely happens to be hosted in
        # Geneva, so the `.ch` fallback stamped it CH. That is a RANKING defect,
        # not a cosmetic one — a CH stamp boosts it on Swiss-scoped queries and
        # subjects the world's primary climate-assessment source to the
        # off-country penalty (AW_OFF_COUNTRY_W=-2.0) on every other query.
        # Same class as the PubMed-is-not-US fix this module already documents.
        or _host_or_sub(d, "ipcc.ch")            # IPCC (UN/WMO treaty body)
        or _host_or_sub(d, "iea.org")            # International Energy Agency
        or _host_or_sub(d, "irena.org")          # IRENA
        or _host_or_sub(d, "iaea.org")           # IAEA
        or _host_or_sub(d, "unep.org")           # UN Environment Programme
        or _host_or_sub(d, "unesco.org")         # UNESCO + IIEP/UIS institutes
        or _host_or_sub(d, "unhabitat.org")      # UN-Habitat
        or _host_or_sub(d, "oecd.org")           # OECD
        or _host_or_sub(d, "cochrane.org")       # Cochrane (global evidence synthesis)
        or _host_or_sub(d, "cochranelibrary.com")
        or _host_or_sub(d, "iclei.org")          # global local-government network
        or _host_or_sub(d, "citiesalliance.org")
    )


# ---------------------------------------------------------------------------
# TASK-261 SIGNAL STACK, ported to BUILD TIME (2026-07-19)
# ---------------------------------------------------------------------------
# The 261 country-stamping convergence (spec country-stamping-coverage.md,
# ADR-055) was folded into the next rebuild instead of being paced on the live
# 30GB box. For the fold to survive a rebuild, the standalone signal stack that
# lived ONLY in the retired `apply_country_convergence` script (which just read
# task 261's `worldmap.json`) had to move into this build-time module,
# else the rebuild would re-stamp with the old .ch/.fr-only rules and silently
# LOSE the convergence. These four tables + the checks in _national_for_domain /
# _intl_domain reproduce worldmap.json exactly on the location-bound path.
#
# PRECEDENCE (spec §, worldmap generation order):
#   INTL/supra -> explicit national hosts -> Wikidata P17 -> restricted-2LD /
#   nexus-TLD -> .ch/.fr fallback -> None.
# ADR-046 still holds: this path is reached ONLY for location-bound source_types
# (is_location_bound); reference/PubMed/arXiv/Wikipedia -> INTL, general -> None.
# So P17 never re-introduces the domain-only stamp that tagged all of PubMed US.

# (1) Wikidata P17 (country-of-origin via P856 official-website exact-IRI join).
# Snapshot of the 521 domains 261 resolved live on WDQS (worldmap provenance
# =p17). DATA, not a rule: keyed on the registrable `domain`, EXACT match only
# (no subdomain fan-out — mirrors apply_country_convergence.py). Multi-country
# P17 conflicts were already declined to None upstream, so every value is single.
# Regenerate at future rebuilds by re-running the P17 pull (spec §4.1).
P17_COUNTRY = {
    # AT (4)
    'osce.org': 'AT', 'statistik.at': 'AT', 'vorarlberg.at': 'AT', 'whatchado.com': 'AT',
    # AU (7)
    'abs.gov.au': 'AU', 'legislation.gov.au': 'AU', 'nla.gov.au': 'AU', 'nsw.gov.au': 'AU',
    'qld.gov.au': 'AU', 'vic.gov.au': 'AU', 'wa.gov.au': 'AU',
    # BE (24)
    'bakermckenzie.com': 'BE', 'belgium.be': 'BE', 'cedr.eu': 'BE',
    'composeralliance.org': 'BE', 'const-court.be': 'BE', 'cost.eu': 'BE',
    'edf-feph.org': 'BE', 'enqa.eu': 'BE', 'eufic.org': 'BE', 'eurodesk.eu': 'BE',
    'fin.belgium.be': 'BE', 'frs-fnrs.be': 'BE', 'health.belgium.be': 'BE',
    'inami.fgov.be': 'BE', 'iweps.be': 'BE', 'myria.be': 'BE', 'nbb.be': 'BE', 'onem.be': 'BE',
    'raadvst-consetat.be': 'BE', 'statbel.fgov.be': 'BE', 'uliege.be': 'BE', 'unia.be': 'BE',
    'wallonie.be': 'BE', 'wcoomd.org': 'BE',
    # CA (5)
    'bankofcanada.ca': 'CA', 'canada.ca': 'CA', 'cifar.ca': 'CA', 'iata.org': 'CA',
    'umontreal.ca': 'CA',
    # CH (282)
    'aaeb.ch': 'CH', 'aarreha.ch': 'CH', 'abc-culture.ch': 'CH', 'actioninnocence.org': 'CH',
    'ahv-iv.ch': 'CH', 'airportbuochs.ch': 'CH', 'alpwirtschaft.ch': 'CH', 'altdorf.ch': 'CH',
    'altekanti.ch': 'CH', 'annagoeldi.ch': 'CH', 'arbeitgeber.ch': 'CH', 'arth.ch': 'CH',
    'at-schweiz.ch': 'CH', 'avry.ch': 'CH', 'barmelweid.ch': 'CH', 'bbbaden.ch': 'CH',
    'bbzp.ch': 'CH', 'bcv.ch': 'CH', 'berufsberatung.ch': 'CH', 'bezirk-march.ch': 'CH',
    'bezirk-schwyz.ch': 'CH', 'bfu.ch': 'CH', 'bger.ch': 'CH', 'bhm.ch': 'CH',
    'bibliomedia.ch': 'CH', 'bibliothequesonore.ch': 'CH', 'bienen.ch': 'CH',
    'biketowork.ch': 'CH', 'bis.org': 'CH', 'bosco-gurin.ch': 'CH', 'breitenbach.ch': 'CH',
    'bstger.ch': 'CH', 'burgerbib.ch': 'CH', 'burgzug.ch': 'CH', 'bvger.ch': 'CH',
    'canal9.ch': 'CH', 'canalalpha.ch': 'CH', 'cdt.ch': 'CH', 'ch-open.ch': 'CH',
    'chalais.ch': 'CH', 'changins.ch': 'CH', 'chaux-de-fonds.ch': 'CH', 'chgemeinden.ch': 'CH',
    'chillon.ch': 'CH', 'conservatorio.ch': 'CH', 'contrechamps.ch': 'CH',
    'croix-rouge-ne.ch': 'CH', 'cscs.ch': 'CH', 'curling.ch': 'CH', 'datascience.ch': 'CH',
    'designwerk.com': 'CH', 'digitale-verwaltung-schweiz.ch': 'CH', 'dodis.ch': 'CH',
    'eawag.ch': 'CH', 'ech.ch': 'CH', 'ecoinvent.org': 'CH', 'economiesuisse.ch': 'CH',
    'editore.ch': 'CH', 'edk.ch': 'CH', 'einsiedeln.ch': 'CH', 'elternnotruf.ch': 'CH',
    'eoren.ch': 'CH', 'epfl-ecal-lab.ch': 'CH', 'erstfeld.ch': 'CH',
    'evolvinglanguage.ch': 'CH', 'familienberatung-sg.ch': 'CH', 'femtoprint.ch': 'CH',
    'fhnw.ch': 'CH', 'filmmusikwettbewerb.ch': 'CH', 'finma.ch': 'CH',
    'fluechtlingshilfe.ch': 'CH', 'forscenter.ch': 'CH', 'forschung3r.ch': 'CH', 'fr.ch': 'CH',
    'frc.ch': 'CH', 'friedberg.ch': 'CH', 'gantrisch.ch': 'CH', 'garedunord.ch': 'CH',
    'gde-engelberg.ch': 'CH', 'gdk-cds.ch': 'CH', 'ge.ch': 'CH', 'gemeinde-andermatt.ch': 'CH',
    'gemeindeschwyz.ch': 'CH', 'geneve.ch': 'CH', 'gersau.ch': 'CH', 'ghgo.ch': 'CH',
    'giswil.ch': 'CH', 'glarus-nord.ch': 'CH', 'glarus-sued.ch': 'CH', 'glarus.ch': 'CH',
    'goldach.ch': 'CH', 'grenchen.ch': 'CH', 'gruene-gl.ch': 'CH',
    'gruyerepaysdenhaut.ch': 'CH', 'guido-fluri-stiftung.ch': 'CH', 'gurtnellen.ch': 'CH',
    'hagel.ch': 'CH', 'heiden.ch': 'CH', 'heimatschutz.ch': 'CH', 'helvetas.org': 'CH',
    'hep-verlag.ch': 'CH', 'herisau.ch': 'CH', 'holzbildhauerei.ch': 'CH',
    'holzenergie.ch': 'CH', 'horriwil.ch': 'CH', 'hslu.ch': 'CH', 'hueniken.ch': 'CH',
    'hvschwyz.ch': 'CH', 'ige.ch': 'CH', 'infoclio.ch': 'CH', 'infofauna.ch': 'CH',
    'insieme-ausserschwyz.ch': 'CH', 'ipcc.ch': 'CH', 'iru.org': 'CH', 'isdc.ch': 'CH',
    'isenthal.ch': 'CH', 'iwb.ch': 'CH', 'jacobsfoundation.org': 'CH', 'jagdschweiz.ch': 'CH',
    'jardinsmusicaux.ch': 'CH', 'jbneuchatel.ch': 'CH', 'jmp-ch.org': 'CH',
    'jodlerverband.ch': 'CH', 'jugendundsport.ch': 'CH', 'jungegrunliberale.ch': 'CH',
    'kantisargans.ch': 'CH', 'kantiwattwil.ch': 'CH', 'kbow.ch': 'CH', 'kdk.ch': 'CH',
    'kerns.ch': 'CH', 'kgk-cgc.ch': 'CH', 'klett.ch': 'CH', 'kovive.ch': 'CH',
    'ksb-sg.ch': 'CH', 'ksbg.ch': 'CH', 'kuessnacht.ch': 'CH', 'kunsthauszug.ch': 'CH',
    'kuppel-basel.ch': 'CH', 'laregione.ch': 'CH', 'laupersdorf.ch': 'CH',
    'lausanne-morges.ch': 'CH', 'lausanne.ch': 'CH', 'leb.ch': 'CH', 'liarumantscha.ch': 'CH',
    'loro.ch': 'CH', 'luesslingen-nennigkofen.ch': 'CH', 'lugano.ch': 'CH', 'mcah.ch': 'CH',
    'mediatheque.ch': 'CH', 'mieterverband.ch': 'CH', 'milvignes.ch': 'CH',
    'mobiliar.ch': 'CH', 'mobility.ch': 'CH', 'monteverita.org': 'CH',
    'musee-gruerien.ch': 'CH', 'museocentovallipedemonte.ch': 'CH', 'mx3.ch': 'CH',
    'myswitzerland.com': 'CH', 'nant-de-drance.ch': 'CH', 'nationalpark.ch': 'CH',
    'ne.ch': 'CH', 'nebelspalter.ch': 'CH', 'neuchatelville.ch': 'CH',
    'neue-stadtschulen.ch': 'CH', 'neustark.com': 'CH', 'nfp76.ch': 'CH',
    'observatoire-culture.ch': 'CH', 'opendata.ch': 'CH', 'orientation.ch': 'CH',
    'osar.ch': 'CH', 'osi.swiss': 'CH', 'ostwind.ch': 'CH', 'ow.ch': 'CH', 'pa-ch.ch': 'CH',
    'paediatrieschweiz.ch': 'CH', 'parks.swiss': 'CH', 'parlament.ch': 'CH',
    'patrinum.ch': 'CH', 'pdag.ch': 'CH', 'phlu.ch': 'CH', 'phsz.ch': 'CH', 'pmodwrc.ch': 'CH',
    'postauto.ch': 'CH', 'pro-velo.ch': 'CH', 'proinfirmis.ch': 'CH', 'projuventute.ch': 'CH',
    'publica.ch': 'CH', 'rabe.ch': 'CH', 'radio.ch': 'CH', 'radiochablais.ch': 'CH',
    'rbs.ch': 'CH', 'realp.ch': 'CH', 'redcross.ch': 'CH', 'rheumaliga.ch': 'CH',
    'rigi.ch': 'CH', 'ruetimattli.ch': 'CH', 'sak.ch': 'CH', 'sans-violence.ch': 'CH',
    'sarnen.ch': 'CH', 'sbb.ch': 'CH', 'schattdorf.ch': 'CH', 'search.ch': 'CH',
    'seedorf-uri.ch': 'CH', 'seelisberg.ch': 'CH', 'seges.ch': 'CH', 'sfgz.ch': 'CH',
    'sg.ch': 'CH', 'sgb.ch': 'CH', 'sgg-ssh.ch': 'CH', 'sirm-institute.ch': 'CH',
    'six-group.com': 'CH', 'skyguide.ch': 'CH', 'slf.ch': 'CH', 'snb.ch': 'CH', 'snf.ch': 'CH',
    'so.ch': 'CH', 'sogi.ch': 'CH', 'solaxess.ch': 'CH', 'solothurnerfilmtage.ch': 'CH',
    'sporthilfe.ch': 'CH', 'spv.ch': 'CH', 'srk-bern.ch': 'CH', 'srk-schwyz.ch': 'CH',
    'stadlerrail.com': 'CH', 'stadt-zuerich.ch': 'CH', 'stadt.winterthur.ch': 'CH',
    'stadtzug.ch': 'CH', 'stapferhaus.ch': 'CH', 'staur.ch': 'CH', 'stein-ar.ch': 'CH',
    'stiftung-habitat.ch': 'CH', 'stv-fst.ch': 'CH', 'suchtschweiz.ch': 'CH',
    'suedostschweiz.ch': 'CH', 'suva.ch': 'CH', 'svp.ch': 'CH', 'svv.ch': 'CH',
    'swissid.ch': 'CH', 'swisslos.ch': 'CH', 'swissmedic.ch': 'CH', 'swissolympic.ch': 'CH',
    'swissparalympic.ch': 'CH', 'swissubase.ch': 'CH', 'szgerzensee.ch': 'CH', 'szh.ch': 'CH',
    't-l.ch': 'CH', 'tcs.ch': 'CH', 'teatrodimitri.ch': 'CH', 'tele1.ch': 'CH',
    'telem1.ch': 'CH', 'theatercasino.ch': 'CH', 'tierschutz.com': 'CH', 'titlis.ch': 'CH',
    'tposcht.ch': 'CH', 'tvo-online.ch': 'CH', 'udc.ch': 'CH', 'unaids.org': 'CH',
    'unesco.ch': 'CH', 'unifr.ch': 'CH', 'unige.ch': 'CH', 'unilu.ch': 'CH', 'ur.ch': 'CH',
    'vd.ch': 'CH', 'verkehrsclub.ch': 'CH', 'villars-sur-glane.ch': 'CH',
    'vogelwarte.ch': 'CH', 'vs.ch': 'CH', 'vss.ch': 'CH', 'vsuzh.ch': 'CH', 'wsl.ch': 'CH',
    'wwf.ch': 'CH', 'www.sz.ch': 'CH', 'www.zh.ch': 'CH', 'xund.ch': 'CH',
    'zbsolothurn.ch': 'CH', 'zentralgut.ch': 'CH', 'zentrumbildung.ch': 'CH', 'zg.ch': 'CH',
    'zkb.ch': 'CH', 'zpk.org': 'CH', 'zurzachcare.ch': 'CH', 'zvv.ch': 'CH',
    # DE (99)
    'agroforst-info.de': 'DE', 'allianz-der-wissenschaftsorganisationen.de': 'DE',
    'antidiskriminierungsstelle.de': 'DE', 'aok.de': 'DE', 'arbeitsagentur.de': 'DE',
    'atmospheric-chemistry-and-physics.net': 'DE', 'awi.de': 'DE', 'bayern.de': 'DE',
    'bfn.de': 'DE', 'bibb.de': 'DE', 'bitkom.org': 'DE', 'bmas.de': 'DE',
    'bmbfsfj.bund.de': 'DE', 'bmv.de': 'DE', 'bmvg.de': 'DE',
    'bundesfreiwilligendienst.de': 'DE', 'bundesnetzagentur.de': 'DE',
    'bundesregierung.de': 'DE', 'bundestag.de': 'DE', 'bundesumweltministerium.de': 'DE',
    'bundesverfassungsgericht.de': 'DE', 'bundeswirtschaftsministerium.de': 'DE',
    'cargobeamer.de': 'DE', 'charta-der-vielfalt.de': 'DE', 'cube.eu': 'DE', 'daad.de': 'DE',
    'deginvest.de': 'DE', 'destatis.de': 'DE', 'deutschepost.de': 'DE', 'dguv.de': 'DE',
    'dgvn.de': 'DE', 'dhl.de': 'DE', 'dihk.de': 'DE', 'dkrz.de': 'DE', 'dresden.de': 'DE',
    'elster.de': 'DE', 'epo.org': 'DE', 'evs-eu.org': 'DE', 'finanzverwaltung.nrw.de': 'DE',
    'fli.de': 'DE', 'gema.de': 'DE', 'geoportal-bw.de': 'DE',
    'geoscientific-model-development.net': 'DE', 'gkv-spitzenverband.de': 'DE',
    'goldingcapital.com': 'DE', 'gvl.de': 'DE', 'hamburg.de': 'DE', 'hdba.de': 'DE',
    'helmholtz.de': 'DE', 'hochschulstart.de': 'DE', 'hrk.de': 'DE', 'iab.de': 'DE',
    'id.bund.de': 'DE', 'ihk.de': 'DE', 'ikw.org': 'DE', 'init.de': 'DE',
    'initiative-musik.de': 'DE', 'institut-fuer-menschenrechte.de': 'DE', 'jazzahead.de': 'DE',
    'kbs.de': 'DE', 'kfw.de': 'DE', 'kmk.org': 'DE', 'kreis-tir.de': 'DE',
    'kuenstlersozialkasse.de': 'DE', 'landscape-online.org': 'DE',
    'leibniz-gemeinschaft.de': 'DE', 'lrz.de': 'DE', 'lwk-niedersachsen.de': 'DE',
    'mags.nrw': 'DE', 'minijob-zentrale.de': 'DE', 'monopol-magazin.de': 'DE',
    'mpi-inf.mpg.de': 'DE', 'musikrat.de': 'DE', 'netzwerk-iq.de': 'DE', 'neustadt.de': 'DE',
    'podstars.de': 'DE', 'pretix.eu': 'DE', 'regierung-mv.de': 'DE', 'rki.de': 'DE',
    'rowenta.com': 'DE', 'schleswig-holstein.de': 'DE', 'science-on-stage.de': 'DE',
    'seethesound.de': 'DE', 'siemens.com': 'DE', 'soundtrackcologne.de': 'DE',
    'standesbeamte.de': 'DE', 'tu-braunschweig.de': 'DE', 'txlogistik.eu': 'DE',
    'uni-flensburg.de': 'DE', 'uni-giessen.de': 'DE', 'uni-jena.de': 'DE',
    'uni-stuttgart.de': 'DE', 'verbraucherzentrale-bawue.de': 'DE',
    'verbraucherzentrale.nrw': 'DE', 'verbraucherzentrale.sh': 'DE', 'vzhh.de': 'DE',
    'weiden.de': 'DE', 'wissenschaftsrat.de': 'DE', 'zbfs.bayern.de': 'DE',
    # DK (5)
    'dtu.dk': 'DK', 'motorst.dk': 'DK', 'skat.dk': 'DK', 'toldst.dk': 'DK', 'um.dk': 'DK',
    # ES (3)
    'aena.es': 'ES', 'csif.es': 'ES', 'datos.gob.es': 'ES',
    # FI (2)
    'klamicompetition.fi': 'FI', 'oph.fi': 'FI',
    # FR (18)
    'agroforesterie.fr': 'FR', 'ameli.fr': 'FR', 'blast-info.fr': 'FR', 'calameo.com': 'FR',
    'ccr-zkr.org': 'FR', 'cesni.eu': 'FR', 'ciec1.org': 'FR', 'cncdh.fr': 'FR',
    'cnil.fr': 'FR', 'defenseurdesdroits.fr': 'FR', 'editionsladecouverte.fr': 'FR',
    'france-education-international.fr': 'FR', 'francecompetences.fr': 'FR',
    'has-sante.fr': 'FR', 'ircam.fr': 'FR', 'monde-diplomatique.fr': 'FR', 'ngfs.net': 'FR',
    'piarc.org': 'FR',
    # GB (19)
    'brandwatch.com': 'GB', 'brew.sh': 'GB', 'caa.co.uk': 'GB', 'cabi.org': 'GB',
    'chambers.com': 'GB', 'cloudsecurityalliance.org': 'GB', 'cps.gov.uk': 'GB',
    'data.gov.uk': 'GB', 'gov.uk': 'GB', 'legal500.com': 'GB', 'legislation.gov.uk': 'GB',
    'metoffice.gov.uk': 'GB', 'nationalarchives.gov.uk': 'GB', 'nc3rs.org.uk': 'GB',
    'nhs.uk': 'GB', 'nice.org.uk': 'GB', 'ons.gov.uk': 'GB', 'openstreetmap.org': 'GB',
    'www.gov.uk': 'GB',
    # GE (1)
    'eqe.ge': 'GE',
    # IE (3)
    'accenture.com': 'IE', 'qqi.ie': 'IE', 'revenue.ie': 'IE',
    # IL (2)
    'cloudinary.com': 'IL', 'myheritage.com': 'IL',
    # IN (1)
    'postman.com': 'IN',
    # IT (7)
    'fao.org': 'IT', 'gazzettaufficiale.it': 'IT', 'istitutosvizzero.it': 'IT',
    'iubenda.com': 'IT', 'labiennale.org': 'IT', 'opac.sbn.it': 'IT', 'salute.gov.it': 'IT',
    # JP (1)
    'jstage.jst.go.jp': 'JP',
    # LV (4)
    'aic.lv': 'LV', 'infogram.com': 'LV', 'iub.gov.lv': 'LV', 'izm.gov.lv': 'LV',
    # NO (3)
    'hkdir.no': 'NO', 'nokut.no': 'NO', 'norlights.com': 'NO',
    # PE (2)
    'elor.com.pe': 'PE', 'gob.pe': 'PE',
    # PL (1)
    'nawa.gov.pl': 'PL',
    # PT (2)
    'a3es.pt': 'PT', 'dges.gov.pt': 'PT',
    # SE (4)
    'awa.com': 'SE', 'eurosurveillance.org': 'SE', 'libera.chat': 'SE', 'uhr.se': 'SE',
    # SI (1)
    'gov.si': 'SI',
    # US (20)
    'aad.org': 'US', 'addgene.org': 'US', 'anthropic.com': 'US', 'cancer.org': 'US',
    'chanzuckerberg.com': 'US', 'creativecommons.org': 'US', 'darpa.mil': 'US',
    'elastic.co': 'US', 'familysearch.org': 'US', 'genealogybank.com': 'US',
    'gutenberg.org': 'US', 'ifc.org': 'US', 'imf.org': 'US', 'oliverwyman.com': 'US',
    'opencollective.com': 'US', 'openlibrary.org': 'US', 'prnewswire.com': 'US',
    'reverbnation.com': 'US', 'stlouisfed.org': 'US', 'stroke.org': 'US',
    # VA (1)
    'educatio.va': 'VA',
}

# (2) Restricted second-level domains: sub-namespaces gated by a named public
# body that verifies the registrant (spec §4.2) — the real country signal that
# the flat ccTLD is NOT. gov.uk (CDDO), nhs.uk (NHS England), govt.nz, gov.au,
# gov.it, gob.es, gv.at, gov.pt, gov.co. Matched as the host or any subdomain.
RESTRICTED_2LD = {
    "gov.uk": "GB", "nhs.uk": "GB", "govt.nz": "NZ", "gov.au": "AU",
    "gov.it": "IT", "gob.es": "ES", "gv.at": "AT", "gov.pt": "PT", "gov.co": "CO",
}

# (3) The ONLY TLDs the research found enforce a verified national nexus (spec
# §4.2 — it falsified the "~50-entry ccTLD list" premise). .gov->US already lives
# in the explicit block below; the rest are the enforced-nexus backstop.
NEXUS_TLD = {"swiss": "CH", "ca": "CA", "no": "NO", "ie": "IE", "au": "AU"}

# (4) Supranational EU institutions -> EU (not a country; stamped explicitly and,
# like INTL, EXEMPT from the off-country penalty — ADR-055 §8.1). europa.eu plus
# the standalone EU clinical-trials portal.
SUPRA_EU = ("europa.eu", "euclinicaltrials.eu")

# (5) TASK-263 must-exclude: 619 domains typed `official` in the index that are
# NOT location-bound national authority (global research / standards docs /
# medical publishers / vendor-SaaS: journals.plos.org, rust-lang.github.io,
# cmake.org, owasp.org, stripe.com, cision.fr, …). The primary 263 go-forward fix
# re-types these OUT of `official` at ingest (discovery_ingest.py health->reference),
# after which is_location_bound gates them out on its own. This set is the
# belt-and-braces guard for records that still carry source_type=official at
# rebuild — the build-time equivalent of apply_country_convergence.py's --exclude
# NOT-filter (ADR-054/055): country_for_record short-circuits them to None so a P17
# / nexus / ccTLD signal can never make globally-authoritative content demotable.
# bag.admin.ch (a genuine Swiss federal health authority mislabeled `health`) is
# deliberately NOT in this set. Disjoint from the 261 worldmap (0 overlap).
MUST_EXCLUDE = frozenset({
    '143.ch', '4gamer.net', '5barricas.valenciaplaza.com', 'ab.wikipedia.org',
    'abkuerzungen.de', 'accessiblebooksconsortium.org', 'ace.wikipedia.org', 'acta.asso.fr',
    'activepages.com.au', 'addiction-neuchatel.ch', 'addictohug.ch', 'adipositasbern.ch',
    'adr.org', 'af.wikipedia.org', 'afro.who.int', 'aha.ch', 'airtable.com',
    'akupunktur-tcm.ch', 'albankaldawli.org', 'alberta.ca', 'allendowney.github.io',
    'allocine.fr', 'als.wikipedia.org', 'altair-viz.github.io', 'am.wikipedia.org',
    'ameblo.jp', 'an.wikipedia.org', 'ang.wikipedia.org', 'anq.ch', 'ansm.sante.fr',
    'antifraudcentre-centreantifraude.ca', 'app.hellothematic.com', 'aquaetgas.ch', 'ar.ch',
    'ar.wikipedia.org', 'ar.wikiquote.org', 'arc.wikipedia.org', 'ary.wikipedia.org',
    'arz.wikipedia.org', 'as.wikipedia.org', 'ast.wikipedia.org', 'auswaertiges-amt.de',
    'av.wikipedia.org', 'avk.wikipedia.org', 'awa.wikipedia.org', 'ay.wikipedia.org',
    'az.wikipedia.org', 'az.wikiquote.org', 'ba.wikipedia.org', 'badw.de', 'bancomundial.org',
    'banqueducanada.ca', 'bar.wikipedia.org', 'bauchzentrum-bern.ch', 'bcl.wikipedia.org',
    'be-tarask.wikipedia.org', 'be.wikipedia.org', 'bernmobil.ch', 'bew.wikipedia.org',
    'bew.wiktionary.org', 'bfarm.de', 'bfh.ch', 'bh.wikipedia.org', 'bi.wikipedia.org',
    'bigenc.ru', 'bjn.wikipedia.org', 'blender.community', 'blog.suisa.ch',
    'blogs.biomedcentral.com', 'blogs.nvidia.com.tw', 'bmz.de', 'board.com', 'boost.org',
    'br.wikipedia.org', 'brc.ch', 'bs.ch', 'bs.wikipedia.org', 'bs.wikiquote.org',
    'btm.wikipedia.org', 'bundesamtsozialesicherung.de', 'bundesgesundheitsministerium.de',
    'businesses.uber.com', 'buymeacoffee.com', 'bzgbs.ch', 'ca.wikiquote.org',
    'careers.philips.com', 'careers.smartrecruiters.com', 'caritas.ch',
    'catalogo.share-cat.unina.it', 'cbk-zam.wikipedia.org', 'cdo.wikipedia.org',
    'ce.wikipedia.org', 'ceb.wikipedia.org', 'cerclair.ch', 'ch.nicovideo.jp', 'change.org',
    'chase.com', 'circuit.li', 'cisco.com', 'cision.fr', 'clang.llvm.org', 'clarunis.ch',
    'cmake.org', 'cmswire.com', 'cnrtl.fr', 'co.wikipedia.org', 'codezine.jp',
    'colaus-psycolaus.ch', 'commonchemistry.cas.org', 'community.m5stack.com',
    'company.sbb.ch', 'computerweekly.com', 'conet.de', 'contributor-covenant.org', 'coub.com',
    'covid-vaccine.canada.ca', 'cplusplus.github.io', 'crh.wikipedia.org', 'cs.ubc.ca',
    'cs.wikipedia.org', 'cs.wiktionary.org', 'csb.wikipedia.org', 'csrs.ch', 'ctan.org',
    'cu.wikipedia.org', 'customerservices.philips.com', 'cy.wikipedia.org', 'd.hatena.ne.jp',
    'da.wikipedia.org', 'dag.wikipedia.org', 'daisy.org', 'data.worldbank.org',
    'datahelpdesk.worldbank.org', 'de.pons.com', 'de.wikisource.org', 'de.wiktionary.org',
    'deutsche-biographie.de', 'deutsches-kinderschmerzzentrum.de', 'dexonline.ro',
    'dictation.philips.com', 'diq.wikipedia.org', 'download.moodle.org', 'drupal.org',
    'dsb.wikipedia.org', 'dtp.wikipedia.org', 'dty.wikipedia.org', 'dv.wikipedia.org',
    'dv.wiktionary.org', 'dz.wikipedia.org', 'e-health-suisse.ch', 'ebmt.org',
    'ecolelasource.ch', 'emf-health-cluster.eu', 'eml.wikipedia.org', 'en.wikibooks.org',
    'en.wiktionary.org', 'ensembl.org', 'eo.wikinews.org', 'eo.wikipedia.org', 'eoas.ubc.ca',
    'eoc.ch', 'epic.com', 'epic.inselgruppe.ch', 'epicresearch.org', 'equinenow.com',
    'es.wikinews.org', 'es.wikipedia.org', 'es.wiktionary.org', 'escardio.org', 'esiee.fr',
    'eu.wikipedia.org', 'eur01.safelinks.protection.outlook.com',
    'eur02.safelinks.protection.outlook.com', 'eur03.safelinks.protection.outlook.com',
    'eurotopics.net', 'expanseproject.eu', 'exposoins26.ch', 'ext.wikipedia.org', 'famh.ch',
    'felixplatter.ch', 'fi.wikipedia.org', 'fit-care.ch', 'fiu-vro.wikipedia.org',
    'fj.wikipedia.org', 'flic.kr', 'fo.wikipedia.org', 'fondationhug.org', 'foxbusiness.com',
    'foxnews.com', 'fr.wiktionary.org', 'freizeit.sbb.ch', 'frontiersin.org',
    'frontierspartnerships.org', 'frp.wikipedia.org', 'frr.wikipedia.org', 'fundable.com',
    'fur.wikipedia.org', 'fy.wikipedia.org', 'ga.wikipedia.org', 'gag.wikipedia.org',
    'gateway.one', 'gavi.org', 'gbr01.safelinks.protection.outlook.com', 'gcr.wikipedia.org',
    'gd.wikipedia.org', 'geigystiftung.ch', 'geni.com', 'gesundheitsfoerderung-uri.ch',
    'gigazine.net', 'giz.de', 'glk.wikipedia.org', 'globalcause.co.uk', 'gn.wikipedia.org',
    'golosknig.com', 'gom.wikipedia.org', 'gor.wikipedia.org', 'gov.bc.ca', 'government.nl',
    'guidle.com', 'gv.wikipedia.org', 'ha.wikipedia.org', 'hak.wikipedia.org',
    'hands4health.dev', 'hanson.net', 'haw.wikipedia.org', 'hdr.undp.org', 'he-arc.ch',
    'healthytravel.ch', 'hearingsolutions.philips.com', 'heds-fr.ch', 'help.rescue-uk.org',
    'help.uber.com', 'hif.wikipedia.org', 'hogrefe.com', 'hogrefe.epaper-publishing-one.de',
    'home-appliances.philips', 'home.cern', 'homepage.ninja', 'hr.wikipedia.org', 'hrw.org',
    'hsb.wikipedia.org', 'huawei.com', 'humboldt-foundation.de', 'ia.wikipedia.org',
    'iba.wikipedia.org', 'ichspendeblut.ch', 'icrc.org', 'id.wikipedia.org', 'identi.ca',
    'ie.wikipedia.org', 'ig.wikipedia.org', 'igalia.com', 'ilo.wikipedia.org', 'ine.es',
    'infomaniak.events', 'infovac.ch', 'inh.wikipedia.org', 'initiative-qualitaetsmedizin.de',
    'insel-link.ch', 'intensedebate.com', 'internals.rust-lang.org',
    'international-climate-initiative.com', 'internezzo.ch', 'interpharma.ch', 'invelos.com',
    'invivomagazine.ch', 'io.wikipedia.org', 'is.wikipedia.org', 'isglobal.org',
    'iu.wikipedia.org', 'iupac.org', 'iwi.unisg.ch', 'ja.wikipedia.org', 'jam.wikipedia.org',
    'journals.plos.org', 'jv.wikipedia.org', 'k4d.ch', 'ka.wikipedia.org', 'kaa.wikipedia.org',
    'kab.wikipedia.org', 'kaj.wikipedia.org', 'kcg.wikipedia.org', 'kge.wikipedia.org',
    'ki.wikipedia.org', 'kidshealth.org', 'kindergesundheit.de', 'king-cr.jp',
    'km.wikipedia.org', 'kn.wikipedia.org', 'knc.wikipedia.org', 'knowyourmeme.com',
    'ko.wikipedia.org', 'korpus.cz', 'kotava.org', 'krc.wikipedia.org', 'krebsliga.ch',
    'ks.wikipedia.org', 'ksh.wikipedia.org', 'ku.wikipedia.org', 'kubernetes-csi.github.io',
    'kuehne-nagel.com', 'kununu.com', 'kv.wikipedia.org', 'kvg.org', 'kw.wikipedia.org',
    'ky.wikipedia.org', 'la.wikipedia.org', 'lad.wikipedia.org', 'lareunion.ars.sante.fr',
    'lausanneuniversityhospital.com', 'lb.wikipedia.org', 'leblogtvnews.com', 'leenaards.ch',
    'legacytree.com', 'levoyaging.fr', 'lfn.wikipedia.org', 'li.wikipedia.org',
    'lighting.philips.com', 'liguecancer.ch', 'lij.wikipedia.org', 'live.worldbank.org',
    'lld.wikipedia.org', 'llvm.org', 'lmo.wikipedia.org', 'ln.wikipedia.org', 'lnk.to',
    'lo.wikipedia.org', 'loop.frontiersin.org', 'lt.wikipedia.org', 'lt.wikiquote.org',
    'lv.wikipedia.org', 'mad.wikipedia.org', 'maelstrom-research.org', 'map-bms.wikipedia.org',
    'mapleprimes.com', 'matahari.global', 'mathworld.wolfram.com', 'mdf.wikipedia.org',
    'mea.lighting.philips.com', 'medicalxpress.com', 'medici.tv', 'medix.ch', 'menshealth.de',
    'mentalhealtheurope.org', 'meteoblue.com', 'metoomvmt.org', 'mg.wikipedia.org',
    'mi.wikipedia.org', 'min.wikipedia.org', 'mk.wikipedia.org', 'ml.wikipedia.org',
    'mn.wikipedia.org', 'monenfantestmalade.ch', 'moodle.org', 'ms.wikipedia.org',
    'munzinger.de', 'museedelamain.ch', 'my-insel.ch', 'my.wikipedia.org', 'myebr.org',
    'mzn.wikipedia.org', 'na01.safelinks.protection.outlook.com', 'nah.wikipedia.org',
    'nam10.safelinks.protection.outlook.com', 'nam11.safelinks.protection.outlook.com',
    'nam12.safelinks.protection.outlook.com', 'nap.wikipedia.org', 'nb.no',
    'nds-nl.wikipedia.org', 'newadvent.org', 'nextgem.eu', 'nfb.ca', 'nl.wikipedia.org',
    'nn.wikipedia.org', 'nn.wiktionary.org', 'nndb.com', 'no.wikipedia.org', 'novartis.com',
    'npb.jp', 'nrm.wikipedia.org', 'nso.wikipedia.org', 'nv.wikipedia.org', 'oanda.com',
    'oc.wikipedia.org', 'old.bigenc.ru', 'om.wikipedia.org', 'omr.com', 'one-piece.com',
    'onf.ca', 'online.flippingbook.com', 'optout.aboutads.info', 'or.wikipedia.org',
    'os.wikipedia.org', 'ost.ch', 'owasp.org', 'pa.wikipedia.org', 'palliativakademie-bern.ch',
    'pam.wikipedia.org', 'pap.wikipedia.org', 'parasite-diagnosis.ch',
    'parkleitsystem-basel.ch', 'partnerships.nature.com', 'passes.com', 'pasteur.fr',
    'path.org', 'pcd.wikipedia.org', 'pdc.wikipedia.org', 'petdiagnostik.ch',
    'pfl.wikipedia.org', 'phatwalletforums.com', 'phhelv.ch', 'philips.ch', 'philips.com.sg',
    'philips.de', 'philips.fr', 'pillarcatholic.com', 'pizolcare.ch', 'pl.wikipedia.org',
    'pl.wikiquote.org', 'planetesante.ch', 'plos.org', 'pms.wikipedia.org', 'podcast.de',
    'podcaster.de', 'policyalternatives.ca', 'polioeradication.org', 'polosbn.bnnonline.it',
    'potofu.me', 'primomedico.com', 'production.nfbonf.ca', 'profile.hatena.ne.jp',
    'projectgoliat.eu', 'promedmail.org', 'ps.wikipedia.org', 'pt.wikipedia.org',
    'pt.wiktionary.org', 'qna.habr.com', 'qu.wikipedia.org', 'qualtrics.com', 'r4d.ch',
    'radiolac.ch', 'ranas.ch', 'rc2nb.ch', 'rcmp.ca', 'reha-rheinfelden.ch', 'rescue.org',
    'retrainpain.org', 'rm.wikipedia.org', 'rnib.org.uk', 'ro.wikipedia.org',
    'roa-tara.wikipedia.org', 'ru.wikipedia.org', 'rue.wikipedia.org', 'rug.nl',
    'rust-lang.github.io', 'rust-random.github.io', 'rustacean-principles.netlify.app',
    'rw.wikipedia.org', 'sa.wikipedia.org', 'sages.ch', 'sah.wikipedia.org',
    'sante-sexuelle.ch', 'santepubliquefrance.fr', 'save-a-life.ch', 'sc.wikipedia.org',
    'scholarlykitchen.sspnet.org', 'scn.wikipedia.org', 'sco.wikipedia.org',
    'sd.wikipedia.org', 'sdgs.un.org', 'se.wikipedia.org', 'secondstreet.ru',
    'service-allergie.ch', 'sf-encyclopedia.com', 'sg.wikipedia.org', 'sgad.ch',
    'sh.wikipedia.org', 'shi.wikipedia.org', 'shihang.org', 'shonenjump.com',
    'shonenjumpplus.com', 'shopify.com', 'signify.com', 'sigplan.org', 'silgeneve.ch',
    'simple.wikipedia.org', 'sk.wikiquote.org', 'sketchfab.com', 'sl.wikipedia.org',
    'sm.wikipedia.org', 'smallcultfollowing.com', 'smartermedicine.ch', 'sn.wikipedia.org',
    'snaccooperative.org', 'so.wikipedia.org', 'software.intel.com', 'solidarmed.ch',
    'soundcloud.com', 'soundtrackzurich.com', 'sourceware.org', 'sphn.ch', 'spitalaarberg.ch',
    'spitalbelp.ch', 'spitalriggisberg.ch', 'spitalverbund.ch', 'spo.ch', 'sq.wikipedia.org',
    'sr.wikipedia.org', 'srn.wikipedia.org', 'ssph-journal.org', 'ssphplus.ch', 'stitcher.com',
    'stm-assoc.org', 'stonybrookchildrens.org', 'stonybrookmedicine.edu', 'stop-piracy.ch',
    'stpaulcenter.com', 'stq.wikipedia.org', 'streamingmedia.com', 'streamingmediaglobal.com',
    'stripe.com', 'su.wikipedia.org', 'suisa.ch', 'survio.com', 'sv.wikipedia.org', 'svri.org',
    'sw.wikipedia.org', 'swatch.com', 'swissbiobanking.ch', 'swisscardio.ch',
    'swisshealthcarestartups.com', 'swissheart.ch', 'swissmicrobiology.ch', 'swissnex.org',
    'swisstransplant.org', 'syl.wikipedia.org', 'szl.wikipedia.org', 'te.wikipedia.org',
    'testzentrale.ch', 'tg.wikipedia.org', 'tk.wikipedia.org', 'tl.wikipedia.org',
    'tok.wikipedia.org', 'tpg.ch', 'tpi.wikipedia.org', 'tr.wikipedia.org', 'tr.wikiquote.org',
    'transplantnet.ch', 'trapego.ch', 'travelhealthpro.org.uk', 'treffinsel.ch',
    'triple-underscore.github.io', 'troped.org', 'tt.wikipedia.org', 'tuigroup.com',
    'tunein.com', 'tvpsolar.com', 'tw.wikipedia.org', 'typo3.com', 'tyv.wikipedia.org',
    'uberhealth.com', 'ug.wiktionary.org', 'ukbb.ch', 'un.org', 'undp.org', 'unfpa.org',
    'unglobalcompact.org', 'uni-muenster.de', 'unicode-org.github.io', 'unine.ch',
    'unisante.ch', 'unisg.ch', 'unon.org', 'unstats.un.org', 'upov.int', 'ur.wikipedia.org',
    'usa.philips.com', 'uz.wikipedia.org', 'vec.wikipedia.org', 'vep.wikipedia.org',
    'versuni.com', 'vi.wikipedia.org', 'viuz.com', 'vls.wikipedia.org', 'vo.wikipedia.org',
    'vontobel.com', 'w3c.github.io', 'wa.wikipedia.org', 'wa.wiktionary.org', 'wantedly.com',
    'war.wikipedia.org', 'wbcsd.org', 'webassembly.github.io', 'webcitation.org',
    'wicg.github.io', 'wikimediafoundation.org', 'wikipedia.org', 'wipo.int',
    'wo.wikipedia.org', 'woah.org', 'writexo.com', 'wufoo.com', 'xal.wikipedia.org', 'xe.com',
    'xh.wikipedia.org', 'xmf.wikipedia.org', 'yi.wikipedia.org', 'yo.wikipedia.org',
    'youronlinechoices.eu', 'za.wikipedia.org', 'zapier.com', 'zea.wikipedia.org',
    'zgh.wikipedia.org', 'zh-min-nan.wikipedia.org', 'zh-yue.wikipedia.org',
    'zh.wikipedia.org', 'zh.wiktionary.org', 'zhaw.ch', 'zu.wikipedia.org',
})


def _p17_parent(d: str) -> str | None:
    """P17 country via the registrable PARENT of a subdomain.

    The payload `domain` is the full host (`fraser.stlouisfed.org`), but
    P17_COUNTRY is keyed on the registrable domain (`stlouisfed.org`), so the
    exact lookup in _national_for_domain step (2) misses subdomains. Registrant
    country is shared across a domain's subdomains, so we try each parent suffix
    of >=2 labels, most-specific first, and return the first P17 hit. Returns
    None for a 2-label host (step 2 already handled the exact case) or no hit.
    Used only on the location-bound path (gated by is_location_bound upstream).
    """
    parts = d.split(".")
    for i in range(1, len(parts) - 1):
        suffix = ".".join(parts[i:])
        if suffix in P17_COUNTRY:
            return P17_COUNTRY[suffix]
    return None


def _national_for_domain(d: str) -> str | None:
    """National jurisdiction for a LOCATION-BOUND source's domain, resolved in the
    task-261 signal order (spec country-stamping-coverage.md §, ADR-055):

      supra EU -> explicit national hosts -> Wikidata P17 map (exact, then
      registrable-parent) -> restricted-2LD -> nexus-TLD ->
      `.ch`/`.fr`/`.de` ccTLD fallback -> None.

    Returns a jurisdiction code (or "EU"), or None. This is the ONLY place the
    ccTLD/.gov fallbacks and the ported 261 tables live now — callers gate it
    behind a location-bound source_type. INTL / supra-INTL are handled earlier by
    _intl_domain, so this function never returns "INTL"."""
    if not d:
        return None

    # (0) supranational EU institutions -> EU (exempt from demotion, like INTL).
    if any(_host_or_sub(d, s) for s in SUPRA_EU):
        return "EU"

    # (1) explicit national authorities (the pre-261 hardcodes; "explicit hosts"
    #     in the spec precedence, ahead of P17).
    # --- CH (Switzerland) ---
    if (_host_or_sub(d, "admin.ch") or _host_or_sub(d, "swisstopo.ch")
            or _host_or_sub(d, "swisstopo.admin.ch")
            or "swisstopo" in d.split(".")[0] or _host_or_sub(d, "bfs.ch")):
        return "CH"
    # --- FR (France) ---
    if (_host_or_sub(d, "gouv.fr") or _host_or_sub(d, "insee.fr")
            or _host_or_sub(d, "bnf.fr")):
        return "FR"
    # --- CA (Canada) ---
    if _host_or_sub(d, "gc.ca") or _host_or_sub(d, "statcan.gc.ca"):
        return "CA"
    # --- DE (Germany) ---
    if _host_or_sub(d, "gesetze-im-internet.de"):
        return "DE"
    # --- US (.gov is US-only by registry policy; id.loc.gov etc.) ---
    if _host_or_sub(d, "loc.gov") or d == "gov" or d.endswith(".gov"):
        return "US"

    # (2) Wikidata P17 (primary 261 signal) — EXACT registrable-domain match,
    #     mirroring apply_country_convergence.py's domain-keyed lookup.
    if d in P17_COUNTRY:
        return P17_COUNTRY[d]

    # (2b) P17 by REGISTRABLE PARENT. The payload `domain` is the full host, but
    #      the P17 map is keyed on the registrable domain, so an exact lookup
    #      misses a subdomain (audit 2026-07-19: fraser.stlouisfed.org 1,963
    #      central-bank records unstamped though stlouisfed.org->US is in P17).
    #      Registrant country is shared across a domain's subdomains, so walking
    #      the parent suffixes (>=2 labels) is safe on the location-bound path.
    #      Exact keys still win (step 2), so the worldmap is reproduced verbatim.
    par = _p17_parent(d)
    if par:
        return par

    # (3) restricted second-level domains (gov.uk / nhs.uk / govt.nz / …).
    for suffix, cty in RESTRICTED_2LD.items():
        if _host_or_sub(d, suffix):
            return cty

    # (4) enforced-nexus TLDs (.swiss / .ca / .no / .ie / .au); .gov->US is in
    #     the explicit block above.
    tld = d.rsplit(".", 1)[-1] if "." in d else d
    if tld in NEXUS_TLD:
        return NEXUS_TLD[tld]

    # (5) broad ccTLD fallbacks (national, lowest precedence). `.de` joins the
    #     pre-261 `.ch`/`.fr` pair: DE is a scope country (COUNTRIES) yet had no
    #     ccTLD backstop, so location-bound German administration/law on a `.de`
    #     host with no P17 entry fell through to None. Audit 2026-07-19 found 6
    #     German federal/state law portals (rechtsprechung-im-internet.de,
    #     recht.nrw.de, landesrecht-bw.de, revosax.sachsen.de,
    #     gesetze-rechtsprechung.sh.juris.de, landesrecht.sachsen-anhalt.de =
    #     49,087 records) correctly stamped DE in the index that the build would
    #     otherwise DELETE at rebuild. Location-bound-only (this function is gated
    #     behind is_location_bound) + MUST_EXCLUDE runs first, so global-authority
    #     `.de` domains stay None; general `.de` content never reaches here.
    if d == "ch" or d.endswith(".ch"):
        return "CH"
    if d == "fr" or d.endswith(".fr"):
        return "FR"
    if d == "de" or d.endswith(".de"):
        return "DE"
    return None

def country_for_record(domain: str | None, source_type: str | None,
                       register_id: str | None = None,
                       lei: str | None = None) -> str | None:
    """Content-conditioned jurisdiction stamp (Tim's 2026-07-17 policy).

    Returns one of:
      - a jurisdiction code (CH/FR/CA/DE/US) for LOCATION-BOUND content,
      - "INTL" for known cross-jurisdiction content (standards, Wikipedia,
        PubMed, arXiv, general reference),
      - KEEP_EXISTING for a registered entity_card (register country is
        payload-owned; do not overwrite from the domain),
      - None for everything else general (unstamped; neither boosted nor demoted).

    See the module docstring for the full policy + precedence.
    """
    d = _norm(domain)
    st = (source_type or "").strip().lower()

    # 0) An explicit, pack-ratified publisher declaration outranks every
    #    heuristic below (see _PUBLISHER_COUNTRY). Curated beats inferred.
    declared = _publisher_country(d)
    if declared:
        return declared

    # 1) INTL domains win over everything (standards / Wikipedia / research).
    if _intl_domain(d):
        return "INTL"

    # 2) General research/reference is authoritative everywhere -> INTL.
    if st == "reference":
        return "INTL"

    # 3) Location-bound content -> its national jurisdiction (ccTLD/.gov fallback
    #    applies ONLY here now). Prefix-aware: covers gov-law/gov-court/gov-stats/
    #    gov-health and any future gov-*/official-* type (task 259 defect 1).
    if is_location_bound(st):
        # task-263 belt-and-braces: a domain that is really global-authority
        # (research / standards docs / medical publisher / vendor) but still
        # carries source_type=official is NEVER country-stamped — the P17/nexus/
        # ccTLD signal must not make it demotable (ADR-054/055). Post-263 these
        # are re-typed reference/docs at ingest and never reach here; this guard
        # catches any that still slip through at rebuild.
        if d in MUST_EXCLUDE:
            return None
        return _national_for_domain(d)

    # 4) Registered local business card -> keep the payload's register country;
    #    a card with neither a register_id nor a LEI is not a registered
    #    business (YC/category/author card) -> None.
    if st == "entity_card":
        return KEEP_EXISTING if (register_id or lei) else None

    # 5) Everything else general (news, docs, tech, health, blog, edu, ...) -> None.
    return None


def country_for_domain(domain: str | None) -> str | None:
    """LEGACY domain-only resolver — kept for card jurisdiction tagging
    (`build_source_cards.py`, `consent_gate.py`) which want a best-effort
    national/INTL guess from the domain alone. NOTE: this is NOT the content
    stamp anymore — use `country_for_record` for the payload `country` facet.

    Returns INTL for cross-jurisdiction domains (standards / Wikipedia / PubMed /
    arXiv), else the national jurisdiction (incl. ccTLD/.gov fallback), else None.
    """
    d = _norm(domain)
    if not d:
        return None
    if _intl_domain(d):
        return "INTL"
    return _national_for_domain(d)


def country_for_url(url: str | None) -> str | None:
    """Convenience: derive the host from a URL then map it. Prefer
    country_for_domain / country_for_record on the stored `domain` field."""
    if not url:
        return None
    return country_for_domain(url)
