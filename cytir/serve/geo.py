"""Jurisdiction and country scoping, split out of `retrieval.py` (S10 step 4, task 1095).

Query side: the canton / country keyword detector and the sub-national gazetteer. Candidate
side: the scope penalties, the off-country demotion gate, the `AW_COUNTRY_MAP` domain fallback and
the learned-scope conflict penalty. Bodies are verbatim from `retrieval.py`; `retrieval` imports
every name back, so `retrieval.<name>` still answers. The location-bound test is
`cytir.core.country.is_location_bound` (D7). This module must not import `retrieval`.
"""
from __future__ import annotations

import json
import re

from cytir.core.country import is_location_bound
from .settings import SETTINGS
from .signals import _domain

_S = SETTINGS   # typed config, loaded once at import (S10 step 2, task 1071)

# --------------------------------------------------------------------------- #
#  Jurisdiction scoping (task 123, spec §scoped-authority).
#  Government authority is scoped to its jurisdiction: a canton's law
#  collection is authoritative-for-{that canton}, a national government's
#  pages are authoritative-for-{that country} — not for legal questions
#  elsewhere. Without this, the 19 cantonal LexWork collections (trust 1.0,
#  self-similar statute text) flooded federal fedlex/admin.ch golds out of
#  the top-10, and French service-public.gouv.fr fiches topped Swiss queries
#  (measured: CH slice dom@10 0.44 -> 0.39 after the 2026-07-08 merges).
#
#  Two penalties (weights <= 0, default 0.0 = off), both QUERY-CONDITIONED:
#    p_canton      cantonal-law passage (payload `canton` stamp, ingest_lexwork)
#                  whose canton the query does NOT name — including queries
#                  naming no canton: without a canton in the query, no single
#                  canton's statute is the authoritative answer.
#    p_foreign_gov query names a jurisdiction (country marker, or a canton =>
#                  CH; an explicit `country=` arg wins over text detection)
#                  AND the candidate is `source_type=official` stamped with a
#                  DIFFERENT known country. INTL / unstamped are exempt —
#                  Wikipedia/RFCs are never touched.
#  False positives are cheap by design: a wrongly-detected canton/country only
#  EXEMPTS content from a penalty or demotes foreign GOV content — it never
#  boosts and never filters. Queries with no jurisdiction signal take the
#  exact pre-task-123 path (byte-identical scoring and stage-1 fast path).
#
#  Task 144 (residual cantonal flood, 141 M2): 123's penalties only fired when
#  the query named a country/canton, but MANY federal/national law queries name
#  neither (they reference a statute by concept or abbreviation — "ZGB
#  Stockwerkeigentum", "BGB Miete mindern"), so `scoped` was False and cantonal
#  collections flooded freely (measured: ch-law-017 top-10 = 8/10 cantonal). Fix
#  = the p_canton penalty is jurisdiction-INDEPENDENT and now applied to EVERY
#  query at STAGE-2 (a cantonal statute is authoritative only for a query naming
#  that canton — 123's own stated design intent). It stays STAGE-2-ONLY for
#  unscoped queries: the full stage-2 payload already carries `canton`, so this
#  is free and the payload-free stage-1 fast path (and its latency) is UNTOUCHED;
#  scoped queries keep 123's stage-1+2 behaviour unchanged. p_foreign_gov stays
#  query-conditioned (it must know the query's country).
# --------------------------------------------------------------------------- #
_CANTON_NAMES = {
    "ZH": ["zürich", "zuerich", "zurich", "zürcher"],
    "BE": ["bern", "berne", "berner"],
    "LU": ["luzern", "lucerne", "lucerna", "luzerner"],
    # "uri" bare collides with the software term URI (and would imply CH
    # country intent on software queries) — qualified forms only.
    "UR": ["kanton uri", "canton uri", "canton d'uri", "urner"],
    "SZ": ["schwyz", "schwyzer"],
    "OW": ["obwalden", "obwald"],
    "NW": ["nidwalden", "nidwald"],
    "GL": ["glarus", "glaris", "glarner"],
    "ZG": ["zug", "zoug", "zugo", "zuger"],
    "FR": ["fribourg", "freiburg", "friburgo", "freiburger", "fribourgeois"],
    "SO": ["solothurn", "soleure", "soletta", "solothurner"],
    "BS": ["basel-stadt", "basel", "bâle", "basilea", "basler"],
    "BL": ["basel-landschaft", "baselland", "basel-land", "basel", "baselbieter"],
    "SH": ["schaffhausen", "schaffhouse", "sciaffusa", "schaffhauser"],
    "AR": ["appenzell ausserrhoden", "appenzell"],
    "AI": ["appenzell innerrhoden", "appenzell"],
    "SG": ["st. gallen", "st.gallen", "sankt gallen", "saint-gall", "san gallo",
           "st. galler", "st.galler"],
    "GR": ["graubünden", "graubuenden", "grisons", "grigioni", "bündner"],
    "AG": ["aargau", "argovie", "argovia", "aargauer"],
    "TG": ["thurgau", "thurgovie", "turgovia", "thurgauer"],
    "TI": ["ticino", "tessin", "ticinese"],
    "VD": ["vaud", "waadt", "vaudois", "vaudoise"],
    "VS": ["valais", "wallis", "vallese", "valaisan", "walliser"],
    "NE": ["neuchâtel", "neuchatel", "neuenburg", "neuchâtelois"],
    "GE": ["genève", "geneve", "genf", "ginevra", "geneva", "genevois",
           "genevoise"],
    "JU": ["jura", "jurassien"],
}

_COUNTRY_MARKERS = {  # first match wins (order below); canton match => CH
    "CH": ["switzerland", "swiss", "schweiz", "schweizer", "schweizerisch",
           "schweizerische", "schweizerischen", "suisse", "svizzera",
           "svizzero", "svizzeri", "eidgenössisch", "eidgenössische",
           "eidg", "fedlex", "estv", "ahv", "suva"],
    "FR": ["france", "frankreich", "francia"],
    "DE": ["germany", "deutschland", "allemagne", "germania"],
    "CA": ["canada", "kanada", "canadian", "kanadisch", "canadien",
           "canadienne"],
    "US": ["united states", "usa", "u.s.", "amerikanisch", "états-unis"],
}

# candidate-side countries the foreign-gov penalty may bite on (aw_country
# stamp values); INTL and None/unstamped are exempt by omission.
_SCOPE_COUNTRIES = frozenset({"CH", "FR", "DE", "CA", "US"})

# task 261 (ADR-055): stamps a country selection NEVER demotes, listed EXPLICITLY
# so a widened demotable set can never accidentally bite them. INTL/EU are
# cross-jurisdiction; "" is the empty-string stamp (~41k `tech` records, ADR-055
# §8.1) — falsy, but named here so `AW_OFF_COUNTRY_ANY` cannot make it demotable.
_EXEMPT_STAMPS = frozenset({"INTL", "EU", ""})

# task 261 (ADR-055 §3.3, the reported "current VAT rates"+country=CH bug): with
# the demotable set hardcoded to the five _SCOPE_COUNTRIES, stamping revenue.ie->IE
# is a no-op — IE is not in the set, so the off-country penalty still returns 0 and
# revenue.ie stays at rank 3 on a CH-scoped query. AW_OFF_COUNTRY_ANY widens the
# set to ANY real stamped country (except _EXEMPT_STAMPS) so the agent's `country`
# filter actually narrows. Default OFF => byte-identical to today (ADR-054: a
# self-hoster pulling this commit gets today's behaviour, never a widened default).
_OFF_COUNTRY_ANY = _S.country.off_country_any

def _demotable_scope(cty):
    """True if a geo-scoped query may demote an off-country candidate stamped
    `cty`. Default = the five _SCOPE_COUNTRIES (byte-identical to `cty in
    _SCOPE_COUNTRIES`, since that set holds none of _EXEMPT_STAMPS). Under
    AW_OFF_COUNTRY_ANY = any real 2-letter ISO country code that is not an exempt
    stamp — so a stamped IE/BE/NZ/… is demotable on a foreign-scoped query."""
    if not cty or cty in _EXEMPT_STAMPS:
        return False
    if _OFF_COUNTRY_ANY:
        return len(cty) == 2 and cty.isalpha()
    return cty in _SCOPE_COUNTRIES

# task 164: EU member states. EU/supranational law (EUR-Lex, ec.europa.eu, the
# whole europa.eu family) is authoritative FOR every member state, so it must
# NOT be treated as `foreign_gov` when the query's jurisdiction is an EU member.
# (Switzerland is deliberately NOT a member — EU law is *relevant* to CH but not
# domestic; we leave CH out of the exemption and rely on EU content being
# None/`EU`-stamped, i.e. already outside _SCOPE_COUNTRIES, to keep it un-demoted
# for CH too — a SOFT non-demotion, per the task recommendation.)
_EU_MEMBERS = frozenset({
    "FR", "DE", "IT", "ES", "NL", "BE", "LU", "AT", "IE", "PT", "FI", "SE",
    "DK", "PL", "CZ", "SK", "HU", "RO", "BG", "GR", "EL", "HR", "SI", "EE",
    "LV", "LT", "CY", "MT",
})

def _is_eu_supra(url_or_domain: str | None) -> bool:
    """True for EU/supranational institution content (EUR-Lex, ec.europa.eu,
    digital-strategy.ec.europa.eu, curia/edps/… — the whole europa.eu family).
    Authoritative FOR EU member states; exempt from foreign_gov on member-state
    queries (task 164). Robust to the stamp: catches these pages whether they
    are stamped None/`EU`/a member code."""
    d = _domain(url_or_domain)
    if d.startswith("www."):
        d = d[4:]
    return d == "europa.eu" or d.endswith(".europa.eu")

def _terms_re(names) -> re.Pattern:
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(n) for n in names)
                      + r")(?!\w)", re.UNICODE)

_CANTON_RES = {c: _terms_re(ns) for c, ns in _CANTON_NAMES.items()}

_COUNTRY_RES = [(c, _terms_re(ns)) for c, ns in _COUNTRY_MARKERS.items()]

def _query_jurisdiction(q: str):
    """-> (frozenset of canton codes named in the query, country code or None)."""
    ql = (q or "").lower()
    cantons = frozenset(c for c, rx in _CANTON_RES.items() if rx.search(ql))
    q_country = None
    for c, rx in _COUNTRY_RES:
        if rx.search(ql):
            q_country = c
            break
    if cantons and q_country is None:
        q_country = "CH"
    return cantons, q_country

# --------------------------------------------------------------------------- #
#  Query-side sub-national GAZETTEER (ADR-062 Phase-2b, task 281 / spec §1 knob-13).
#  Deterministic name→country rollup for provinces/states/cantons/Länder/regions +
#  major cities that `_query_jurisdiction` does NOT enumerate (the keyword detector
#  owns cantons + country markers and takes precedence). Prominence-ordered
#  first-match. Homographs are ABSTAINED (country-vs-subnational + balanced city
#  collisions): they never roll up AND act as a HARD abstain override on the learned
#  scope head — "a wrong scope is worse than no scope" (ADR-054). Country-level
#  scope only; sub-national codes stay deferred (ADR-045). Cold-start prior + input
#  feature + fallback for the learned head (Q5), never the sole driver.
# --------------------------------------------------------------------------- #
_GAZ_PATH = _S.runtime.gazetteer_path

def _load_gazetteer():
    """Compile the gazetteer into (prominence-ordered rollup regexes, abstain
    regex). Fail-safe: any error -> empty (no sub-national rollup, byte-identical
    to the keyword-only detector)."""
    try:
        with open(_GAZ_PATH) as f:
            doc = json.load(f)
        entries = sorted(doc.get("entries", []),
                         key=lambda e: -int(e.get("prominence", 0)))
        rollup = [(e["country"], _terms_re(e["names"])) for e in entries if e.get("names")]
        anames = [a["name"] for a in doc.get("abstain", []) if a.get("name")]
        abstain_re = _terms_re(anames) if anames else None
        return {"rollup": rollup, "abstain_re": abstain_re,
                "n": len(rollup), "n_abstain": len(anames)}
    except Exception as e:  # noqa: BLE001 — a bad gazetteer must not break search
        import logging
        logging.getLogger("aw_server.retrieval").warning(
            "gazetteer load failed (%s: %s) — sub-national scope disabled",
            type(e).__name__, e)
        return {"rollup": [], "abstain_re": None, "n": 0, "n_abstain": 0}

_GAZETTEER = _load_gazetteer()

def _gazetteer_scope(q):
    """-> (country|None, hit_abstain). Deterministic prominence-ordered rollup of
    a sub-national name to its COUNTRY. `hit_abstain` is True when an ambiguous
    homograph matched (used to hard-abstain the learned head)."""
    ql = (q or "").lower()
    hit_abstain = bool(_GAZETTEER["abstain_re"] and _GAZETTEER["abstain_re"].search(ql))
    for country, rx in _GAZETTEER["rollup"]:
        if rx.search(ql):
            return country, hit_abstain
    return None, hit_abstain

def _scope_penalties(q_cantons, q_country, canton, cty, source_type,
                     is_eu_supra=False):
    """(p_canton, p_foreign_gov) in {0,1} — see block comment above.

    task 164: `is_eu_supra` content is EXEMPT from foreign_gov when the query's
    jurisdiction is an EU member (EU law is authoritative FOR France/Germany/…,
    not foreign to them). Independent of this flag, EU content is normally
    None/`EU`-stamped => outside _SCOPE_COUNTRIES => already un-demoted; the flag
    makes the exemption robust to a member-code stamp."""
    # task 163/161: entity cards carry a canton stamp (SME register cards) but are
    # identity locators, not cantonal LAW — never canton-penalize them.
    if source_type == "entity_card":
        canton = None
    p_c = 1.0 if (canton and canton not in q_cantons) else 0.0
    eu_exempt = is_eu_supra and (q_country in _EU_MEMBERS)
    # task 261: _demotable_scope() == `cty in _SCOPE_COUNTRIES` by default, and
    # widens to any real stamped country under AW_OFF_COUNTRY_ANY (so a foreign
    # `official`/gov page stamped IE/BE/… — e.g. revenue.ie — is demotable on a
    # CH-scoped query, which is the reported bug's foreign_gov lane).
    p_f = 1.0 if (q_country and source_type == "official" and not eu_exempt
                  and _demotable_scope(cty) and cty != q_country) else 0.0
    return p_c, p_f


# --------------------------------------------------------------------------- #
#  OFF-COUNTRY demotion (task 240) — the geo-scoped country gate, generalized.
#  The task-123 foreign_gov penalty only bit source_type=official candidates, so
#  an OFF-country ENTITY CARD or commercial/content page rode the flat trust
#  floor to #1 on a jurisdiction-scoped query (live 2026-07-15:
#  "best online shop electronics switzerland" → shoptiques.com, a US clothing
#  marketplace ENTITY CARD, #1 on trust 0.9 — wrong country AND category). This
#  generalizes the off-country demotion to the classes foreign_gov MISSES: on a
#  geo-scoped query (q_country from a country term in the text OR an explicit
#  country= arg) demote any candidate stamped with a CONFLICTING _SCOPE_COUNTRY
#  (US/FR/DE/CA/CH). INTL (Wikipedia/RFCs/W3C) and None/unstamped are EXEMPT by
#  omission — cross-border references still surface, recall preserved. EU-supra
#  content is exempt when q_country is an EU member (reuses task 164). We leave
#  source_type=official to foreign_gov (task 123) so the two compose with NO
#  double penalty; off_country owns entity_card / reference / content / unspecified.
#  Additive penalty (weight <= 0), default 0.0 = INERT (byte-identical scoring);
#  stage-2 only (a demotion never needs to widen the pool). rollback = 0.0.
#  Env override AW_OFF_COUNTRY_W stages the weight without editing the json.
# --------------------------------------------------------------------------- #
#  LOCATION-BOUND gate on the off-country demotion (task 259, ADR-046/052).
#  ---------------------------------------------------------------------------
#  The demotion above assumes a stamped scope-country MEANS "this content is
#  bound to that jurisdiction". It doesn't: `country` was stamped from the DOMAIN
#  alone, so the `.gov` fallback tagged the entire PubMed corpus US
#  (28,004,137 records ≈ 59% of the index). Every CH-scoped query therefore
#  applied -2.0 to globally-authoritative medical abstracts. Tim's 2026-07-17
#  directive: "only store country for pages whose content is only really relevant
#  to that location."
#
#  The fix is to demote only LOCATION-BOUND content (national administration /
#  statistics / tax / geo / central bank / law / case law / register records, and
#  registered entity_cards). Doing that HERE rather than by re-stamping 28M
#  payloads is deliberate:
#    * this is exactly what the ADR-043 gate MEASURED — its `core` column (zero
#      loss on authoritative-478/names/jurisdiction, chmed01 3→1, chmed05 None→1)
#      was produced by monkeypatching this very function; the re-stamp's
#      equivalence to it was only ever INFERRED;
#    * the re-stamp measures ~20h/leg (137-398 rec/s on both legs, NVMe — an
#      intrinsic qdrant payload-write rate, not a disk artifact);
#    * it is self-maintaining — it reads the POLICY, so it cannot rot the way 28M
#      payload values can as new content lands with an old stamp.
#  The re-stamp still runs (background, ADR-039 dump / data correctness) but is
#  no longer what the ranking depends on.
#
#  Location-bound classification is `cytir.core.country.is_location_bound` (S10 D7, task 1095):
#  the server's own copy of that test was deleted; the package image ships `cytir/`
#  (server/Dockerfile COPYs it), so serving imports the one definition.
#
#
#  *** DEFAULT OFF — BLOCKED BY ADR-054. DO NOT FLIP WITHOUT A PASSING GATE. ***
#  This was shipped 2026-07-17 and REVERTED the same hour: a live before/after
#  gold gate measured a REGRESSION, not the zero-loss the ADR-043 gate had
#  reported. authoritative-478 h@1 182→171, h@10 268→253, MRR 0.438→0.412 — a
#  coherent cluster (8 CA gov/tax golds 1→None, 6 FR gov/stats/health golds
#  1-2→None, ch-law-025 10→None). Exempting foreign `reference` stops demoting
#  US content on a CA-/FR-/CH-scoped query, so it FLOODS the top-10 and buries
#  the national authority — precisely what task 240's penalty exists to prevent.
#  The PubMed win is real (chmed r2 2→1, r5 6→4) but is PAID FOR by that burial:
#  ADR-046 framed a trade-off as a free correction.
#  Why the held ADR-043 gate passed is UNRESOLVED (its script was never
#  persisted). Settle that before any retry — an unexplained PASSING gate is
#  worse than a failing one.
_OFF_COUNTRY_LOCBOUND_ONLY = _S.country.off_country_locbound_only   # default OFF — ADR-054   # default OFF — ADR-054


def _country_is_binding(source_type, has_register=False):
    """True when this candidate's stamped `country` genuinely means "bound to
    that jurisdiction", i.e. when country_for_record() would assign a scope code:

      * location-bound types (gov*/official*/central-bank/case_law/
        registry_record)          -> the domain-derived national jurisdiction;
      * entity_card WITH a register_id/lei -> its country OF REGISTRATION;
      * everything else (reference/news/tech/docs/health/blog/... and a
        register-LESS card)       -> not binding.

    The entity_card split is load-bearing and must NOT be simplified to "exempt
    all cards": a REGISTERED foreign business card is genuinely off-country and
    must stay demotable — that is the task-240 shoptiques fix. The ADR-043 gate
    measured both: `core` (this rule) = zero loss; `full` (exempt ALL cards) lost
    ch-law-025 10→None and eu-fr-01 8→9. This reproduces `core`.
    """
    if is_location_bound(source_type):
        return True
    if (source_type or "").strip().lower() == "entity_card":
        return bool(has_register)
    return False


def _off_country_hit(q_country, cty, source_type, is_eu_supra=False,
                     has_register=False, include_official=False):
    """1.0 when a geo-scoped query should demote this OFF-country candidate
    (a non-official page/card stamped a conflicting scope-country), else 0.0."""
    if not q_country or (source_type == "official" and not include_official):
        return 0.0            # official is normally foreign_gov's lane (task 123)
    if not cty or cty in _EXEMPT_STAMPS or cty == q_country:
        return 0.0            # in-country, INTL/EU, "" or None/unstamped → never demote
    if not _demotable_scope(cty):
        return 0.0            # task 261: default = the 5 _SCOPE_COUNTRIES; under
                              # AW_OFF_COUNTRY_ANY = any real stamped country
    if _OFF_COUNTRY_LOCBOUND_ONLY and not _country_is_binding(source_type,
                                                             has_register):
        return 0.0            # task 259: the stamp is domain-derived noise on
                              # non-location-bound content (PubMed → US) →
                              # authoritative everywhere → never demote it
    if is_eu_supra and q_country in _EU_MEMBERS:
        return 0.0            # EU law is authoritative FOR member states (task 164)
    return 1.0


# task 261 (ADR-055 §7.1): serving-side domain->country map. Lets the country-
# stamping CONVERGENCE be GATED (and, if wanted, deployed) with ZERO qdrant
# writes: when a record's payload `country` is absent/empty, fall back to this
# map keyed on the INDEXED `domain`. Default UNSET => _effective_country(pl) is
# exactly pl.get("country"), byte-identical to reading the payload. The map only
# FILLS MISSING stamps (payload always wins), so it reproduces the
# unstamped->stamped convergence without ever overriding a register/P17 country
# already in the payload. (The ~91 wrong-stamp CORRECTIONS — landesbibliothek.li
# CH->INTL — are payload-present, so the map does not model them; the real
# set_payload apply owns those. Gate the narrowing benefit, which is the
# missing-stamp case, here.)
_COUNTRY_MAP_PATH = _S.runtime.country_map.strip()


_COUNTRY_MAP: dict[str, str] = {}


if _COUNTRY_MAP_PATH:
    try:
        with open(_COUNTRY_MAP_PATH) as _cmf:
            _COUNTRY_MAP = {str(k).lower().lstrip("."): v
                            for k, v in json.load(_cmf).items()
                            if not str(k).startswith("_") and v}
        import logging as _lg
        _lg.getLogger("aw_server.retrieval").info(
            "AW_COUNTRY_MAP: loaded %d domains from %s",
            len(_COUNTRY_MAP), _COUNTRY_MAP_PATH)
    except Exception as _e:   # never let a bad map break serving — fall back to payload
        import logging as _lg
        _lg.getLogger("aw_server.retrieval").warning(
            "AW_COUNTRY_MAP load failed (%s): %s", _COUNTRY_MAP_PATH, _e)
        _COUNTRY_MAP = {}


def _effective_country(pl):
    """The `country` used for GEO SCORING: the payload stamp if present, else the
    AW_COUNTRY_MAP fallback keyed on the indexed `domain`. When the map is unset
    this is exactly pl.get('country'). Used at every geo-scoring read (stage-1
    meta, foreign_gov, off-country, the country soft-boost) so the map reproduces
    the post-convergence ranking."""
    c = pl.get("country")
    if c:
        return c
    if _COUNTRY_MAP:
        d = (pl.get("domain") or _domain(pl.get("url")) or "").lower()
        if d.startswith("www."):
            d = d[4:]
        return _COUNTRY_MAP.get(d) or c
    return c


# task 260: GRADED off-country weight for NON-location-bound content
# (reference/news/tech/docs/health/blog + register-less entity_card — the class
# `_country_is_binding` calls not-binding; ~99% of it by volume is the 28M PubMed
# abstracts stamped US). ADR-054 measured only the two ENDPOINTS of this knob:
#   -2.0 (== _OFF_COUNTRY_W, demote non-bound content exactly like bound content)
#        = live baseline, and
#    0.0 (== AW_OFF_COUNTRY_LOCBOUND_ONLY=1, exempt it entirely)
#        = REGRESSES the gold set (-11 h@1 / -15 h@10): foreign reference floods
#          geo-scoped queries and buries the national gov/tax authority.
# The middle was never tested. Unset => None => INERT: non-bound content keeps
# _OFF_COUNTRY_W, i.e. byte-identical to today's live behaviour. Default MUST
# stay unset (ADR-054: a measured-regressive policy must never be the
# out-of-the-box default for self-hosters).
_OFF_COUNTRY_NONBOUND_W_RAW = _S.country.off_country_nonbound_w.strip()


_OFF_COUNTRY_NONBOUND_W = (float(_OFF_COUNTRY_NONBOUND_W_RAW)
                           if _OFF_COUNTRY_NONBOUND_W_RAW else None)


# task 262: which SUB-POPULATION of the non-bound class the graded weight applies
# to. AW_OFF_COUNTRY_LOCBOUND_ONLY=1 (the arm ADR-054 measured) exempted THREE
# populations at once and attributed the result to the largest by record count
# without isolating it:
#   ref      source_type=reference        ~28.0M (PubMed US 28.00M + 8.6k general)
#   card     register-less entity_card    ~170k  (ADR-021: cards are in-pool by
#                                                 construction -> the a-priori
#                                                 flood suspect, not PubMed)
#   general  news/tech/docs/health/blog/…  ~130k
# Default "ref,card,general" == the full non-bound class == the ADR-054 bundle,
# so the knob's endpoints stay comparable to it. Narrow it to isolate a component.
_OFF_COUNTRY_NONBOUND_CLASSES = frozenset(
    c.strip().lower() for c in
    _S.country.off_country_nonbound_classes.split(",")
    if c.strip())


def _nonbound_class(source_type, has_register=False):
    """Sub-population of the NON-location-bound class. Only meaningful for
    candidates `_country_is_binding` already called not-binding."""
    st = (source_type or "").strip().lower()
    if st == "reference":
        return "ref"
    if st == "entity_card":       # register-less by construction here
        return "card"
    return "general"


def _learned_scope_conflict_penalty(q_country, scope_conf, pl, default_weight):
    """Confidence-scaled learned-only country-conflict penalty for either stage.

    Unlike the legacy lane split this deliberately includes exact ``official``
    candidates, because no deterministic country exists to activate
    ``foreign_gov``.  `_off_country_hit` remains the shared safety boundary for
    in-country, INTL/EU, unstamped and EU-supranational candidates.
    """
    has_reg = bool(pl.get("register_id") or pl.get("lei"))
    hit = _off_country_hit(
        q_country, _effective_country(pl), pl.get("source_type"),
        _is_eu_supra(pl.get("domain") or pl.get("url")), has_reg,
        include_official=True)
    if not hit:
        return 0.0
    weight = default_weight * scope_conf
    if (_OFF_COUNTRY_NONBOUND_W is not None
            and not _country_is_binding(pl.get("source_type"), has_reg)
            and _nonbound_class(pl.get("source_type"), has_reg)
                in _OFF_COUNTRY_NONBOUND_CLASSES):
        # The override is a static magnitude, so apply the classifier confidence
        # here just as `k_off_country` already did for the default magnitude.
        weight = _OFF_COUNTRY_NONBOUND_W * scope_conf
    return weight * hit
