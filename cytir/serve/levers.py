"""Post-row levers of the name and reference families (S10 step 3, task 1083; moved verbatim
from `retrieval.py`, which imports every name back).

`AW_NAMEGATE` (task 404/406/411: damp wrong-person bios on a bare person-name query), the
name-token coverage that feeds it, and `AW_REFCANON` (task 404/406/578: demote `reference` rows
above a competitive canonical row). The constants are read from `settings` once at import, as
`retrieval.py` read them. No import of `retrieval`. The location-bound test is the package's
`cytir.core.country.is_location_bound`, whose body equals the server copy it replaces here (the
server copy in `retrieval.py` goes in step 4).
"""
from __future__ import annotations

from cytir.core.country import is_location_bound as _is_location_bound

from .fusion import _NAMEGATE_COV, _url_tie_len
from .query import _card_alias_in_query
from .settings import SETTINGS as _S
from . import settings as _settings
from .article_tokens import (article_exact_narrowed as _article_exact_narrowed,
                             query_article_ids as _query_article_ids, query_statutes as _query_statutes)
from .signals import _registrable_domain
from .textmatch import _WORD, _title_centrality
# S10 step 8b (task 1125): stage 2, levers and respond moved out of retrieval.search()
from .schema import COLLECTION, SPARSE_BM25, bm25_token_groups
from .fusion import (WEIGHTS, _NAMECOV_SURFACE, _NAMEGATE_TRUSTKEEP, _PRIMACY_KEEP, _TRUSTFLOOR,
    _TRUSTFLOOR_MINBETA, _TRUSTFLOOR_POOL_MAX, _TRUST_MODE, _TRUST_RELGATE, _TRUST_RELGATE_BETA, _W_BM25_EFF,
    _W_DENSE_EFF, _is_primary_class)
from .trace import (_DEBUG_SCORES, _LAST_QUERY_PLAN_TRACE, _QHEAD_COUNTRY_TRACE, _dbgc)
from .heads import (_LAST_V2_DUMP, _LEARNED_ONLY_SCOPE, _RANKHEAD, _RANKHEAD_MODEL, _RANKHEAD_V2_DUMP,
    _RANKHEAD_V2_MODEL)
from .geo import (_OFF_COUNTRY_NONBOUND_CLASSES, _OFF_COUNTRY_NONBOUND_W, _country_is_binding,
    _effective_country, _is_eu_supra, _learned_scope_conflict_penalty, _nonbound_class, _off_country_hit,
    _scope_penalties)
from .legs import _PER_TOKEN_LIMIT, _PER_TOKEN_MAXQ
from .textmatch import (_STOPWORDS, _card_name_match, _exact_v, _title_centrality_v, _title_versions_match)
from .signals import (_academic_penalty, _lf_bulk, _machine_url, _recency, _trust_tier)
from .tables import _authority, _hubness, _parasite
from .query import _name_token_primaries, _required_word_groups
from . import models
from . import qdrant_io
from . import schema
from qdrant_client import models as qm

_NAMEGATE_W = _S.fusion.namegate_w            # <= 0

_REFCANON = _S.fusion.refcanon
_REFCANON_W = _S.fusion.refcanon_w            # <= 0
_REFCANON_BETA = _S.fusion.refcanon_beta
_REFCANON_MAXBETA = _S.fusion.refcanon_maxbeta
_REFCANON_EXTRA = frozenset(
    t.strip().lower()
    for t in _S.fusion.refcanon_extra_classes.split(",")
    if t.strip())
# TASK 578 — the DEMOTED class set, parameterised. 404 hard-coded `reference`
# because the class it was designed to (a wiki article above the statute) is a
# reference page. Task 578's population says the displacer is `reference` for
# 226 of the 237 addressable pairs, so widening this is worth AT MOST 11 pairs
# — it is measured (E578-1 arm A4) rather than assumed, and the default is
# exactly 404's own single class, i.e. byte-identical.
_REFCANON_TARGETS = frozenset(
    t.strip().lower()
    for t in _S.fusion.refcanon_targets.split(",")
    if t.strip())
# TASK 406 (H4) — the REF-CANON collateral cure, designed to the MEASURED cases
# rather than to the "encyclopedic-intent exemption" 404 §5.5 guessed at.
# Reading the 404 refcanon-only leg (`task 404 2026-08-12/
# gate-e3-after-404rc.json`) against its OFF leg shows what actually fired the
# lever on every collateral:
#     dom_entity/en01  (Nestlé)        canonical = bgs.so.ch/texts_of_law/624.512
#     dom_reference/wk28 (Swiss Alps)  canonical = be.clex.ch/texts_of_law/935.111
#     dom_reference/wk21 (Wilhelm Tell) canonical = ur.ch/publikationen/15181
# — an OFF-TOPIC cantonal statute clears the loosened β=0.35 competitiveness bar
# and triggers a class-wide reference demotion, so the correct encyclopedic gold
# drops below pages that do not answer the query at all. The defect is not
# intent classification: it is that the pool condition stopped requiring the
# canonical candidate to be a PLAUSIBLE ANSWER once β was lowered from 0.7 to
# 0.35 (which it had to be — the target class sits at ratio 0.385–0.58).
#
# Cure: the canonical candidate must also OWN THE QUERY'S DISTINCTIVE TOKENS in
# its title (`_title_centrality` >= MINCENT) — the same distinctiveness signal
# the hub-floor uses, applied to the trigger instead of the target. A statute
# whose title is unrelated to the query no longer authorises demoting the
# encyclopedia. Generic (no host list, no source special-case), and 0.0 = inert
# == today's 404 behaviour.
_REFCANON_MINCENT = _S.fusion.refcanon_mincent


def _name_token_coverage(name_tokens: set[str], pl: dict) -> float:
    """Fraction of the queried NAME's tokens (from _person_name_tokens —
    never interrogative filler) present in the candidate's visible text
    (title + snippet — the only text we hold), by exact TOKEN equality after
    apostrophe-stripping. Token equality, not substring (_exact's basis),
    because 'tim' must not be satisfied by 'estimation'; and not stem-prefix,
    because names are not morphological variants (mirrors _card_name_match's
    rationale). 1.0 == the full person is named; a half-match flood page
    scores 1/n per missing name part."""
    if not name_tokens:
        return 0.0
    toks = {m.replace("'", "").replace("’", "") for m in _WORD.findall(
        ((str(pl.get("title") or "")) + " "
         + (str(pl.get("snippet") or ""))).lower())}
    if not toks:
        return 0.0
    return sum(1 for t in name_tokens if t in toks) / len(name_tokens)


def _sparse_name_coverage(name_tokens, primaries: dict, pid,
                          presence: dict) -> float:
    """Fraction of the queried NAME's tokens present in the candidate's STORED
    BM25 vector (the task-083 presence oracle), the surface an ADR-060 entity
    card actually carries its name/alias/register-people tokens on."""
    if not name_tokens:
        return 0.0
    return sum(1 for t in name_tokens
               if pid in presence.get(primaries.get(t), ())) / len(name_tokens)


def _apply_namegate(rows, name_cov: dict, active: bool) -> list:
    """AW_NAMEGATE demotion over the scored rows (see block comment). Mutates
    r[3] (penalties). Returns the demoted pids (empty == inert) so the call
    site can defer the authority boost and tests can assert exactly when and
    on whom it acted.

    v3 (first re-gate diagnostic, 2026-08-12): sub-full coverage INCLUDING
    ZERO is demoted. v2 exempted zero-coverage candidates ("a dense/alias
    match claims no name-token evidence"), and the measured result on aw_e3
    was that demoting the six exact-half-match bios revealed the next layer —
    a reservoir of ZERO-coverage near-name bios (Mike_O'Hearn, Timothy_Keller,
    Timmie_Rogers… fused 1.7–2.1 via the dense person-bio cluster + trust)
    that re-filled the top-10 and kept the n01 gold out. On a bare person-name
    query where the pool PROVES the full name is matchable, a candidate that
    does not visibly name the person is noise for this query shape — the
    full-coverage pool condition is the safety, not the zero exemption."""
    if not (active and _NAMEGATE_W and rows):
        return []
    if not any(name_cov.get(r[0], 0.0) >= _NAMEGATE_COV for r in rows):
        return []          # nobody in-pool proves the full name — stand down
    out = []
    for r in rows:
        if name_cov.get(r[0], 0.0) < _NAMEGATE_COV:
            r[3] += _NAMEGATE_W
            out.append(r[0])
    return out


def _refcanon_canonical(source_type) -> bool:
    """Is this source_type a canonical/primary class for AW_REFCANON's pool
    condition? Location-bound classes (incl. the gov-*/official-* prefix rule,
    so an unenumerated gov class defaults IN, not out) + the env extra-set."""
    st = (source_type or "").strip().lower()
    return _is_location_bound(st) or st in _REFCANON_EXTRA


def _apply_refcanon(rows, q_content, lang, head_on, rh_beta,
                    official_pref_fired) -> list:
    """AW_REFCANON demotion over the scored rows (see block comment). Mutates
    r[3] (penalties). Returns the demoted pids (empty == inert) so the call
    site can defer the authority boost (measured on aw_e3: wiki rows carry an
    earned-authority boost up to +0.75, which must not partially refund the
    demotion — same deferral rule as hubfloor/namegate) and tests can assert
    exactly when it acted.

    On AW_REFCANON_BETA (first re-gate diagnostic, 2026-08-12): official_pref's
    0.7 never fires on the measured eviction-by-worse class — epoch-3's wiki
    body chunks inflate the pool-best rel_core, leaving the canonical source
    at a LOW rel ratio (SCC decision 0.385, estg §32a 0.43, nbb.be 0.58 on the
    403-gate regressions). The gate arm runs 0.35 via env; the code default
    stays 0.7 until a gate ratifies otherwise."""
    if not (_REFCANON and _REFCANON_W and rows and head_on
            and rh_beta < _REFCANON_MAXBETA and not official_pref_fired):
        return []
    best_rel = max(r[2] for r in rows)
    if best_rel <= 0:
        return []
    # task 406 H4: the trigger must be relevance-competitive AND own the query's
    # distinctive tokens in its title. MINCENT 0.0 => the plain 404 condition.
    if not any(_refcanon_canonical(r[1].get("source_type"))
               and r[2] >= _REFCANON_BETA * best_rel
               and (_REFCANON_MINCENT <= 0.0
                    or _title_centrality(q_content, r[1].get("title", ""), lang)
                    >= _REFCANON_MINCENT)
               for r in rows):
        return []                      # no competitive canonical in-pool
    for r in rows:
        if (r[1].get("source_type") == "entity_card"
                and _card_alias_in_query(q_content, r[1].get("title", ""),
                                         r[1].get("names", ""), lang)):
            return []                  # entity query — the card lane owns it
    out = []
    for r in rows:
        if (r[1].get("source_type") or "").strip().lower() in _REFCANON_TARGETS:
            r[3] += _REFCANON_W
            out.append(r[0])
    return out


def _card_key(e) -> tuple:
    """Entity-card eligibility row `(name_match, sitelinks, rel_core, row)`
    best-first: the three measures desc, then stage 2's tie order (the shorter
    URL, then point id). A total key, so `rows` order never picks the card."""
    nm, sl, rel, r = e
    return (-nm, -sl, -rel, _url_tie_len(r[1].get("url")), str(r[0]))


def _card_winners(elig: list) -> list:
    """Collapse eligible cards by registrable domain (the most canonical card
    per domain), then order them best-first; the caller boosts the first
    `_ENTITY_CARD_CAP`. Ties no longer fall to `rows` order (review 779 M1)."""
    best_by_dom: dict = {}
    for e in elig:
        d = _registrable_domain(e[3][1].get("url", ""))
        if d not in best_by_dom or _card_key(e) < _card_key(best_by_dom[d]):
            best_by_dom[d] = e
    return sorted(best_by_dom.values(), key=_card_key)


# --- S10 step 8b (task 1125): constants that travelled with their readers ---

# --------------------------------------------------------------------------- #
#  TASK 665 — AW_VERSION_MATCH: dotted version tokens become first-class terms
#  in the stage-2 lexical features (title-centrality + exact), plus an optional
#  bounded bonus for a candidate whose TITLE names the queried version.
#
#  Mechanism (task 665 §1, measured on aw_e8): "GLM 5.3"
#  reaches bm25/exact/title as the single token `glm` (schema.version_tokens
#  block), so the HF page titled "GLM-5, GLM-5.1, GLM-5.2" gets title 2.5 +
#  exact 0.8 and the HN page titled "GLM-5.3 is now open-weight" gets 0.625 +
#  0.8 — the version the user asked about is the one fact every feature drops.
#
#  With the flag on, the query's version tokens join q_content/q_terms for the
#  rel_core title and exact terms ONLY (every other consumer of q_content —
#  hub-floor, authority min-centrality, refcanon, primacy — stays version-blind,
#  so this lever has one place of effect). Version tokens match by
#  schema.version_match (equal or dot-refinement), never by `_stem_match`'s
#  plain prefix and never by `_exact`'s substring test ("3.1" in "3.12").
#
#  POOL CONDITION (as task 411's required-term lever): the lever stands down
#  unless at least one stage-2 candidate's TITLE names a queried version. A
#  query whose version no candidate carries ("PostgreSQL 15.4 release notes"
#  against a generic "Release Notes" page) is scored exactly as today —
#  appending an unmatchable term would only rescale everyone's coverage.
#
#  AW_VERSION_MATCH_W (>= 0, default 0 => inert even with the flag on) is the
#  bonus added to rel_core for a candidate whose title names the version.
#  Flag-gated, default OFF => byte-identical scoring. No host, family or query
#  is special-cased (engineering-doctrine 9).
# --------------------------------------------------------------------------- #
_VERSION_MATCH = _S.query.version_match

_VERSION_MATCH_W = max(0.0, _S.query.version_match_w)

_TRUST_GATE_PCTL = _S.trust.trust_gate_pctl

# Official-source preference (task 144, class-2 Wikipedia-over-official flood).
# On a jurisdiction-scoped grounding query, a candidate whose rel_core clears
# this fraction of the pool's best rel_core counts as a "competitive official
# competitor" — only then is the encyclopedic reference tier demoted (so the
# penalty can NEVER fire on a query whose best answer is Wikipedia and no real
# official page exists, e.g. backbone/encyclopedic queries). Env-overridable.
_OFFICIAL_PREF_BETA = _S.trust.official_pref_beta

# task 165: legifrance bulk-legal demotion -- the URL test `_lf_bulk` and its why live in
# `signals`; this is its relevance gate (rel_core < gate * best_rel_core => demote).
_LFBULK_RELGATE = _S.trust.legifrance_bulk_relgate

# Entity-card return-band boost (task 163, Phase-1 entity cards LIVE). The mirror
# image of official_pref: on an ENTITY-NAME/identity query the entity's OWN
# official card (source_type=entity_card) should generally beat the encyclopedia
# article (grounding mission = return the authoritative source, not Wikipedia).
# Measured: cards do NOT flood category queries (the identity-centric embedding
# self-regulates → substantive queries return 0 cards); they UNDER-surface on
# name queries. A POSITIVE, DOUBLE-GATED boost (weight >= 0, default 0.0 = inert):
#   (a) POOL-CONDITIONED: card rel_core >= beta * best rel_core (a non-competitive
#       card is never lifted → substantive queries keep 0 cards).
#   (b) NAME-CONDITIONED: card_name_match >= thr — the query must genuinely NAME
#       the entity (category/roster queries name-match no card → never boosted →
#       no flood).
# DISAMBIGUATION (v2): when several same-name entities pass the gate ("Roche" ->
# Hoffmann-La Roche AND the La-Roche commune AND Roche-VD), the CANONICAL / most-
# NOTABLE one must win — else a trivial namesake takes #1. Eligible cards are
# collapsed by REGISTRABLE DOMAIN (fixes the two roche.com cards) and ordered by
# (name_match, PROMINENCE=`sitelinks` payload, rel_core); only the top `cap`
# (default 1) are boosted, so the less-notable namesakes stay at their relevance
# rank (below Wikipedia). Requires the `sitelinks` payload field
# (enrich_card_prominence.py); missing => 0 => falls back to rel_core
# order. Config-driven + env-overridable; rollback = entity_card=0.0.
_ENTITY_CARD_BETA = _S.trust.entity_card_beta

_ENTITY_CARD_NAME_THR = _S.trust.entity_card_name_thr

_ENTITY_CARD_CAP = _S.trust.entity_card_cap

# task 144: apply the cantonal-scope penalty to EVERY query at stage-2 (not just
# jurisdiction-detected ones). Default on and FIXED (constant _CANTON_UNCONDITIONAL below;
# no env setter since S10 D5). To revert, edit that constant and redeploy.
# Reverting to exact task-123 behaviour (canton penalty only on scoped queries) needs a
# constant edit and redeploy, not a config flip; it would keep 123 + foreign_gov.
_CANTON_UNCONDITIONAL = True  # fixed: was env AW_CANTON_UNCONDITIONAL (S10 D5: no product setter)

# task 240: off-country demotion weight (see _off_country_hit). Env-overridable
# so the gate can be A/B-staged without editing fusion_weights.json. Default
# falls back to the json weight (0.0 = inert / not shipped).
_OFF_COUNTRY_W = _S.country.off_country_w

_COMMERCIAL_REF_W = _S.query.commercial_ref_w

_HUBFLOOR_W = _S.trust.hubfloor_w            # penalty <= 0

# Hubness (host-df) floor: a table domain fires only if its hubness >= this. 0.0 =>
# membership is the gate (any table hub); raise to restrict to the biggest hubs.
_HUBFLOOR_DF_MIN = 0.0  # fixed 2026-10-23 (S10 step 3): was AW_HUBFLOOR_DF_MIN, default 0.0, no setter since #968

# Distinctiveness ceiling: fire only when the candidate's title-centrality
# (ADR-031, in [0,1]) is <= this — i.e. its TITLE is not central to the query
# (a generic-token match). A genuine authority hub (query topic in-title) is above.
_HUBFLOOR_DISTINCT_MAX = _S.trust.hubfloor_distinct_max

# Never fire above this trust ceiling (set < 1.0 to protect statute-tier hubs).
_HUBFLOOR_MAXTRUST = 1.0  # fixed 2026-10-23 (S10 step 3): was AW_HUBFLOOR_MAXTRUST, default 1.0, no setter since #968

# Dense-top-hit EXEMPTION: never fire on a candidate whose within-pool dense_norm
# is >= this (i.e. it IS the/near the strongest SEMANTIC match — genuinely the
# answer, not a trust-rider). This is the guard that protects a hub which
# legitimately OWNS a general query (a civic-service portal answering an on-remit
# how-to with a synonym title: dense-strong but low title-centrality). Distinct
# from the abandoned hub-gate (228: dense-ABSENCE as the TRIGGER + strips trust —
# both wrong); here it is an EXEMPTION on an additive penalty, encoding the class-C
# mechanism (rides TRUST, not the top relevance). Default 1.0 = no exemption
# (fires regardless of dense_norm); set < 1.0 to exempt the pool's top dense hits.
_HUBFLOOR_DENSE_MAX = _S.trust.hubfloor_dense_max

# --------------------------------------------------------------------------- #
#  ADR-099 RULING 3 — the rel-competitiveness exemption gains an ABSOLUTE term.
#
#  Task 446 measured the exemption above blocking the trust attenuation on the
#  rank-1 candidate in 43 of 43 queries: `AW_TRUSTFLOOR` is structurally
#  incapable of firing where it matters. The reason is that `rel_core` is built
#  from MIN-MAX NORMALIZED legs, so whichever candidate leads the BM25 leg sits
#  at ratio ~1.0 BY CONSTRUCTION — including a civic page leading a pool that
#  holds nothing relevant. The exemption's own calibration note assumed the
#  flood pages sit at ratio ~0.4; measured, they sit at 0.80-1.00. No value of
#  AW_TRUSTFLOOR_REL_BETA fixes that (below 1.0 still exempts them; 1.0 removes
#  the exemption and regresses the names class it exists for).
#
#  The fix is not a different ratio but a SECOND, ABSOLUTE condition. The
#  pool-relative test stays NECESSARY; it is no longer SUFFICIENT. A candidate
#  is exempted only if it ALSO shows absolute evidence that it owns the query:
#
#    (a) its own raw dense cosine clears AW_TRUSTFLOOR_ABS_DENSE, or
#    (b) it CONTAINS at least one surviving high-IDF query term (ADR-099
#        Ruling 2's classification, read off the stored BM25 vector — the only
#        term-presence oracle we hold; no passage text exists).
#
#  (b) is what keeps the class the exemption was built for: the dense-absent
#  golds are dense-absent because they win on a RARE term (`ohear`, `rfc9103`,
#  `meta-llama`) — they satisfy (b) trivially. The flood pages satisfy neither:
#  dense_raw = 0.000 and their entire BM25 mass is function words.
#
#  Flag-gated AW_TRUSTFLOOR_ABS, default OFF => byte-identical (abs_ok == True).
# --------------------------------------------------------------------------- #
_TRUSTFLOOR_ABS = _S.trust.trustfloor_abs

# Fan-out bound on arm (b): the pool-filtered term-presence probe costs one
# sub-request per term inside the batch the two-stage path already issues.
_TRUSTFLOOR_ABS_MAXTOK = 12  # fixed: was env AW_TRUSTFLOOR_ABS_MAXTOK (S10 D5: no product setter)

# --------------------------------------------------------------------------- #
#  TASK 404 — two epoch-3 displacement levers (403 §8 gate FAIL follow-ups).
#  Both default OFF (W=0.0 => byte-identical scoring). Both act on the existing
#  payload taxonomy (source_type), never on host lists — displacement-framework
#  rule: cure the CLASS, never special-case a domain.
#
#  (a) NAME-GATE (AW_NAMEGATE) — the names/n01 eviction-by-volume class,
#  designed to the ROOT-CAUSE PROBE (coordinator, 2026-08-12 evening, verified
#  against aw_e3 sparse vectors): NONE of the six pages that evicted the n01
#  gold contains BOTH name parts — "Tim O'Hear" tokenizes to {tim, hear,
#  ohear} and the flood is five half-matches on `tim` + one on `ohear`
#  (Anthony_O'Hear {tim:0, ohear:1.48}; Tim_Duquette/Tim_O'Connor/… {tim:>0,
#  ohear:0}), burying the one page holding both. (The `hear` junk token from
#  the apostrophe split is a documented AGGRAVATOR; tokenization is
#  corpus-epoch scope and is NOT changed here — coverage is enforced at the
#  ranking layer with the tokens as they are.)
#
#  Mechanism, SOURCE-AGNOSTIC (works for non-wiki person floods too — the
#  displacement-framework fit): on a PERSON-NAME-SHAPED query (2..MAXTERMS
#  capitalized content terms, ZERO lowercase content terms, no digits —
#  deterministic and conservative), a candidate whose visible text
#  (title+snippet, token-equality — no substring inflation) carries a STRICT
#  NON-EMPTY SUBSET of the name's tokens is a half-match flood: bounded
#  additive penalty (<= 0). Two safeties: (1) POOL-CONDITIONED — fires only
#  when >=1 in-pool candidate carries the FULL name (else nobody proves the
#  name is matchable and the lever is inert, mirroring official_pref);
#  (2) zero-coverage candidates are EXEMPT (a dense/alias match claims no
#  name-token evidence and is not the measured flood class). Full-coverage
#  candidates are exempt by construction, so "who was Ada Lovelace" keeps its
#  wiki-bio gold — the lever DISAMBIGUATES persons, it does not suppress any
#  source class.
#
#  (b) REF-CANON (AW_REFCANON) — the eviction-by-worse class (a wiki article
#  ABOVE the SCC decision / the statute / eur-lex, 403 §8.4). Task-144's
#  official_pref already encodes the cure but fires only on jurisdiction-SCOPED
#  queries; the 403 gate showed the same failure on UNSCOPED authority lookups.
#  Same pool-conditioned reference demotion, gated instead on the ADR-036
#  intent head (rh_beta < MAXBETA == the authority-lookup band; head OFF or
#  absent => lever INERT — it must never degrade to a blanket wikipedia
#  demotion), and EXEMPT when the query name-matches an in-pool entity card
#  (entity/company queries are the card lane's business: the dom_entity wiki
#  golds sit directly above official homepages and must not be flipped by this
#  lever). Canonical classes = the location-bound set (_is_location_bound:
#  official / gov* / official-* / case_law / central-bank / registry_record)
#  plus AW_REFCANON_EXTRA_CLASSES (comma-set, e.g. "docs" for spec/API-lookup
#  wins; ships EMPTY — widening it is a separate, separately-gated change).
#  Defers to official_pref: a query where 144 already demoted reference rows
#  takes no second penalty (no double-count, mirrors hubfloor/authority defer).
# --------------------------------------------------------------------------- #
_NAMEGATE = _S.fusion.namegate

# --------------------------------------------------------------------------- #
#  TASK 578 — PRIMACY-KEEP (AW_PRIMACY). The authority signal the measured
#  population actually asks for, and it is a REFUND, not a boost.
#
#  WHAT THE POPULATION SAYS (task 578 population.v1.json, 283 rows
#  joined from tasks 577/568/567/576):
#    * 237 of 683 displacement pairs are `primary gold -> secondary displacer`;
#      226 of those 237 have `reference` as the displacer — the class REF-CANON
#      ALREADY demotes. Widening what is demoted is worth at most 11 pairs.
#    * The curated trust register is ALREADY RIGHT on **234 of the 237** pairs:
#      the primary page carries the HIGHER stored trust_tier and loses anyway
#      (`trust-pricing.v1.json`). So this is not a missing prior — it is a
#      correct prior being switched off.
#    * What switches it off is the ADR-036 relgate, by construction: on
#      general intent `rh_beta` is large, and every candidate whose
#      `rel_core / best_rel_core` falls below it has its trust term scaled by
#      `ratio / eff_beta`. The reference page that tops relevance keeps 100% of
#      its trust; the primary source a few points behind keeps a fraction of a
#      LARGER prior. 63 of the 126 addressable golds sit in that band
#      (`rhbeta.v1.json`), and on those same queries REF-CANON is locked out by
#      the SAME signal (`rh_beta < MAXBETA` is false).
#
#  THE LEVER. A candidate that is (i) of a PRIMARY source class, (ii) genuinely
#  relevance-COMPETITIVE (`rel_core >= BETA * best_rel_core`) and (iii)
#  TITLE-CENTRAL (it owns the query's distinctive tokens — the same 406-H4
#  distinctiveness test, applied to the beneficiary) keeps the trust the
#  register already gave it: the general-intent relgate ramp and the
#  trust-floor attenuation are refunded by `AW_PRIMACY_KEEP` in [0, 1].
#
#  WHY THIS IS NOT THE TRUST FLOOD COMING BACK. Task 205/210/289 built those
#  attenuations against a WEAKLY-RELEVANT high-trust portal riding its prior
#  onto a general query (`info.gouv.fr` on "psychédéliques", the antirez class).
#  Conditions (ii) and (iii) are exactly the negation of that case, and the
#  HUB-GATE — which withholds trust from a candidate with NO dense support at
#  all — is deliberately NOT refunded: a pure-lexical float stays floated.
#  The flood tripwires (`rel_vs_trust`, `general_flood`) are scored in the
#  battery, and task 576 showed the committed flood proxy is SATURATED, so the
#  non-saturated readings (mean top-10 trust, high-trust slot counts) travel
#  with the verdict rather than a structural PASS.
#
#  NO HOST LISTS, NO SOURCE SPECIAL-CASE (memory `aw-eval-displacement-
#  framework`): the lever acts on the existing payload taxonomy and cures the
#  class. `entity_card` is NOT in the default primary set — the card lane has
#  its own pool-conditioned boost and cap (task 163 v2) and refunding trust
#  here as well would double-count it.
#
#  Double-inert by default (`AW_PRIMACY=0` AND `AW_PRIMACY_KEEP=0.0` =>
#  byte-identical scoring); rollback is the env, not a redeploy.
# --------------------------------------------------------------------------- #
_PRIMACY = _S.fusion.primacy

# relevance-competitiveness bar, as a fraction of the pool-best rel_core.
_PRIMACY_BETA = _S.fusion.primacy_beta

# distinctiveness bar on the beneficiary's OWN title (406 H4's signal).
_PRIMACY_MINCENT = _S.fusion.primacy_mincent

# --------------------------------------------------------------------------- #
#  TASK 406 — INTENT-CONDITIONED PERSON-NAME LEGS (Tim's design ruling,
#  2026-08-13, verbatim): "the way I see this happening is that our 'intent'
#  head either at search time or re-ranking reduces the relevance of dense and
#  increases the importance of all terms being in the sparse."
#
#  MEASURED MECHANISM this is designed to (405 dispute §4 + the orchestrator's
#  live prod probe, `task 405 diag-n01-2026-08-13.txt`): on
#  q="Tim O'Hear" with NO lang the DENSE leg is not merely unhelpful, it is
#  ACTIVELY MISLEADING — dense top-200 is 100% wikipedia person/name pages
#  (Tim O'Connor .556, Bret Loehr .516, Tim_Tim .513, floor .395) and the gold
#  is ABSENT from it entirely. `Tim_Tim` is an Austrian REGGAE BAND: at 128-d
#  MRL the dense leg is matching title-token dominance, not person identity.
#  The gold DOES reach the candidate pool (BM25 rank 13 of 300), so the pool is
#  fine and FUSION is where the gold loses: Tim_Tim scores two legs (rel 2.307)
#  against the gold's one (rel 1.289), and its non-relevance contribution is
#  −0.75 (+1.75 trust, −2.50 namegate) against the gold's 0.00 — i.e. the gold
#  wins as soon as its relevance comes within 0.75 of Tim_Tim's.
#
#  (H1) NAME-DENSE (AW_NAMEDENSE / _SCALE) — Tim's half (a). On person-name
#  intent, SCALE THE DENSE LEG DOWN in rel_core. This lowers a two-leg flooder's
#  relevance while leaving a BM25-only candidate's UNCHANGED (its dense norm is
#  0), which is exactly the asymmetry the measurement calls for. Second-order
#  and measured: it also lowers `best_rel_core`, which RAISES every candidate's
#  rel-ratio and can restore a trust floor the ratio exemption had stripped.
#
#  (H2) NAME-COV (AW_NAMECOV_W) — Tim's half (b), the POSITIVE complement of the
#  404 namegate. AW_NAMEGATE only PENALISES sub-full coverage; "increases the
#  importance of all terms being in the sparse" is a REQUIREMENT raised on the
#  full-coverage candidate's side: a bounded relevance bonus (>= 0) added to
#  rel_core when the candidate's visible text carries EVERY name token. It lands
#  in rel_core, not penalties, on purpose — rel_core is what the trust relgate,
#  the trustfloor exemption and the authority relgate all read, so the page that
#  provably names the person stops being treated as a weak-relevance trust-rider.
#
#  PLACEMENT — RERANK, not search (Tim left it open; §the task doc records the
#  measurement). The pool already holds the gold, so retrieval needs no change;
#  a search-time dense de-quota can only REMOVE candidates (irreversible), and
#  the pool condition below is not even computable before retrieval.
#
#  BOTH HALVES SHARE THE 404 POOL CONDITION: they arm only when the query is
#  person-name-shaped AND >= 1 in-pool candidate carries the FULL name. If
#  nobody proves the name is matchable, down-weighting dense would strip the
#  only signal the query has. This is also the ENCYCLOPEDIC-PERSON GUARD: on
#  "Marie Curie" / "Alan Turing" the wiki bio has FULL name coverage, so it
#  keeps its dense-independent rel and takes the H2 bonus, while the dense
#  down-weight strips the half-match noise floor around it.
#
#  (H3) TRUSTFLOOR x NAMEGATE (404 §5.5 / 405 §4 fix candidates i+ii), two
#  independently-flagged mechanisms:
#    AW_NAMEGATE_TRUSTKEEP — a FULL-COVERAGE candidate is exempt from the
#      trustfloor's dense-cosine attenuation. The page proven to name the person
#      must not be trust-stripped for lacking a dense hit; that is the very
#      signal the query class cannot rely on.
#    AW_TRUSTFLOOR_UNDEMOTED — the trustfloor's rel-ratio exemption measures
#      against the best NON-DEMOTED rel_core. A row a lever has already demoted
#      must not set the bar that strips another row's trust (measured: the
#      wiki-inflated `best_rel_core` 2.307 pushed the gold's ratio to 0.559).
#
#  (H5) SHAPE v2 (AW_NAMEGATE_SHAPE_V2) — the measured German-noun-capitalisation
#  false positives of the 404 detector (404 §5.2: bb-008 4->8, dom-de-01 5->6).
#  German capitalises every noun, so "all content tokens capitalised" carries no
#  person-name information there. Three deterministic, language-generic
#  rejections restore it: an ALL-CAPS token (BGB, RFC, CO) is a statute/spec
#  reference; a token the German de-compounder splits (Kuendigungsfrist,
#  Arbeitsvertrag) is a compound noun; and a person's name is a CONTIGUOUS
#  nominal phrase, so capitalised tokens separated by other words ("Inflation in
#  der Volkswirtschaft") are sentence nouns, not a name.
#
#  Every flag defaults to inert (scale 1.0 / weight 0.0 / off) => byte-identical
#  scoring; rollback is the env, not a redeploy.
# --------------------------------------------------------------------------- #
_NAMEDENSE = _S.fusion.namedense

# multiplicative scale on the dense leg under person-name intent; 1.0 == inert.
_NAMEDENSE_SCALE = _S.fusion.namedense_scale

# additive rel_core bonus for FULL name-token coverage (>= 0); 0.0 == inert.
_NAMECOV_W = _S.fusion.namecov_w

_TRUSTFLOOR_UNDEMOTED = _S.trust.trustfloor_undemoted

# Any of the person-name levers arms the (cheap, deterministic) shape detector.
_NAME_LEVERS_ARMED = bool((_NAMEGATE and _NAMEGATE_W)
                          or (_NAMEDENSE and _NAMEDENSE_SCALE != 1.0)
                          or _NAMECOV_W or _NAMEGATE_TRUSTKEEP)

# The record class whose DISPLAY text is provably not its indexing surface
# (ADR-048/060 + `compose_index_text`'s own exemption). Data, not prose: the
# set is env-overridable so a future producer with the same property is a
# config change, not a code change.
_NAMECOV_SPARSE_CLASSES = frozenset({"entity_card"})  # fixed: was env AW_NAMECOV_SPARSE_CLASSES (S10 D5: no product setter)

_REQTERM_W = _S.fusion.reqterm_w       # >= 0; 0 == inert

_REQTERM_MODE = "all"  # fixed 2026-10-23 (S10 step 3): was AW_REQTERM_MODE, default all, no setter since #968

# widen the df classification window used for REQUIRED-TERM selection only.
# 0 == inherit AW_BM25_PER_TOKEN_LIMIT (no extra call, no behaviour change).
# This never merges ids into the candidate pool: it is classification only, so
# retrieval is untouched and the experiment stays one-variable.
_REQTERM_DFMAX = 0  # fixed 2026-10-23 (S10 step 3): was AW_REQTERM_DFMAX, default 0, no setter since #968

# max source words the required-term machinery will consider (fan-out bound;
# 0 == inherit AW_BM25_PER_TOKEN_MAXQ).
_REQTERM_MAXQ = 0  # fixed 2026-10-23 (S10 step 3): was AW_REQTERM_MAXQ, default 0, no setter since #968

# below this many required words the bonus is inert: one required term is what
# the BM25 leg already ranks on, so "all of them match" says nothing new.
_REQTERM_MINGROUPS = 2  # fixed: was env AW_REQTERM_MINGROUPS (S10 D5: no product setter)

_AUTHORITY_W = _S.trust.authority_w            # boost >= 0

# Ceiling on the additive boost (a hub at authority~1.0 still lifts by at most this).
# Default large so the natural bound is AW_AUTHORITY_W * authority(<=1); set lower
# to cap the top hubs without shrinking the mid-tail's lift.
_AUTHORITY_MAXBOOST = _S.trust.authority_maxboost

# Only credit a host whose stored authority >= this floor (raise to restrict the
# boost to the strongly-endorsed head). 0.0 => any table host.
_AUTHORITY_MIN = _S.trust.authority_min

# RELEVANCE gate (the zero-regression guard): boost only a candidate whose rel_core
# is >= this fraction of the pool-best rel_core — a weakly-relevant authority host
# can't ride its endorsement past a strongly-relevant gold (mirrors official_pref /
# country / entity_card betas). 1.0 => only the top-relevance candidate; 0.0 => no
# gate (any pool member). This is the lever that keeps the boost pure-upside.
_AUTHORITY_RELGATE = _S.trust.authority_relgate

# Distinctiveness gate (mirror of the hub-floor guard, inverted): earn the boost
# only when the candidate's title-centrality (ADR-031, [0,1]) is >= this — i.e. the
# page genuinely matches the query, not a generic-token ride. 0.0 => no gate
# (byte-identical to ungated); raise to make authority a tie-breaker among genuine
# matches only. Kept OFF by default; the gate arbitrates whether it is needed.
_AUTHORITY_MIN_CENTRALITY = _S.trust.authority_min_centrality


def _reciprocal_competition_rank(scores):
    return {pid: 1.0 / (1 + sum(other > score for other in scores.values()))
            for pid, score in scores.items() if score > 0.0}


def coverage_probe(*, best_dense_raw, bm, lang, namegate_tokens, payloads, q, qfilter, rare_toks, raw,
        rh_beta):
    """Stage 2a: per-token term-coverage probe, required-term groups and trust-floor token hits (S10 step 8b)."""

    # per-token stage-2 term-coverage oracle (see flag note): one batched
    # sparse call filtered to the pool ids — tiny responses (ids only).
    # RARE tokens only: coverage over all tokens boosted verbose pages that
    # merely contain every function word of a long question (+1.8 to noise;
    # measured -0.14 MRR on the 29q eval). Rare tokens (df < limit, known
    # exactly because their window wasn't full) are the distinctive terms —
    # queries without any (most natural-language questions) keep baseline
    # stage-2 scoring untouched.
    #
    # task 411: the probe is generalised from "count the rare tokens" to
    # "{token -> the pool ids whose STORED bm25 vector contains it}", because
    # the name-coverage surface (directive a) and the required-term test
    # (directive c) need PER-TOKEN membership, not a count — and they need it
    # for tokens the df<50 test calls common (`maret`, `ohear`, `turing` are all
    # "common" at the shipped limit; measured, task 411 df-census).
    # With every 411 flag off the probe token list is exactly `rare_toks`, `cov`
    # is recomputed from the same memberships, and the call is byte-identical.
    #
    # task 411 (c): the DISTINCTIVENESS classification for required-term
    # selection, optionally re-measured at a wider window. Classification only —
    # these ids are NEVER merged into the candidate pool, so retrieval is
    # unchanged and the arm stays one-variable.
    distinctive = set(rare_toks)
    _reqterm_groups = []
    if _REQTERM_W and payloads:
        _rq_maxq = _REQTERM_MAXQ or _PER_TOKEN_MAXQ
        if _REQTERM_DFMAX > _PER_TOKEN_LIMIT:
            cand = []
            for _w, _p, _hs in bm25_token_groups(q or "", lang):
                if _w in _STOPWORDS:
                    continue
                cand.extend(h for h in _hs if h not in distinctive)
                if len(cand) >= 4 * _rq_maxq:
                    break
            cand = list(dict.fromkeys(cand))
            if cand:
                dreqs = [qm.QueryRequest(
                    query=qm.SparseVector(indices=[int(i)], values=[1.0]),
                    using=SPARSE_BM25, limit=_REQTERM_DFMAX,
                    filter=qfilter, with_payload=False) for i in cand]
                for _h, _resp in zip(cand, qdrant_io.client().query_batch_points(
                        collection_name=COLLECTION, requests=dreqs)):
                    _n = sum(1 for p in _resp.points if p.score > 0)
                    if 0 < _n < _REQTERM_DFMAX:   # window not full => df < DFMAX
                        distinctive.add(_h)
        _reqterm_groups = _required_word_groups(q, lang, distinctive, _rq_maxq)

    # task 411 (a): the name tokens' primary hashes, when the coverage surface
    # is the sparse channel rather than the display text.
    _name_primaries = (_name_token_primaries(namegate_tokens, lang)
                       if (namegate_tokens and _NAMECOV_SURFACE != "display")
                       else {})

    # ADR-099 Ruling 3 arm (b): the HIGH-IDF query terms whose presence in a
    # candidate is the absolute evidence the trust-floor exemption now requires.
    # Armed only where the trust floor can actually fire (flag + general intent +
    # weak pool), so authority-lookup and strong-pool queries pay nothing and
    # stay byte-identical; the terms ride the batched probe below, adding no
    # extra round trip. Highest-IDF first (an unmeasured token is high by
    # definition), capped for fan-out.
    _tf_eff_beta = rh_beta if rh_beta > 0 else (
        _TRUST_RELGATE_BETA if _TRUST_RELGATE else 0.0)
    _tf_abs_armed = bool(
        _TRUSTFLOOR and _TRUSTFLOOR_ABS and payloads
        and _tf_eff_beta >= _TRUSTFLOOR_MINBETA and _tf_eff_beta > 0
        and best_dense_raw < _TRUSTFLOOR_POOL_MAX)
    _tf_abs_toks: list[int] = []
    if _tf_abs_armed:
        _tf_abs_toks = [h for h in sorted(bm, key=lambda t: (-schema.token_idf(t), t))
                        if schema.token_idf(h) >= schema._QSTOP_IDF
                        ][:_TRUSTFLOOR_ABS_MAXTOK]

    presence = {}
    _probe_toks = list(dict.fromkeys(
        list(rare_toks)
        + [h for h in _name_primaries.values() if h is not None]
        + [h for _w, hs in _reqterm_groups for h in hs]
        + _tf_abs_toks))
    cov = {}
    if _probe_toks and payloads:
        pool_ids = list(payloads.keys())
        fl = qm.Filter(must=[qm.HasIdCondition(has_id=pool_ids)])
        reqs = [qm.QueryRequest(query=qm.SparseVector(indices=[int(i)], values=[1.0]),
                                using=SPARSE_BM25, limit=len(pool_ids),
                                filter=fl, with_payload=False)
                for i in _probe_toks]
        for tok_h, resp in zip(_probe_toks,
                               qdrant_io.client().query_batch_points(
                                   collection_name=COLLECTION, requests=reqs)):
            # id-filtered queries return non-matches at score 0 (task 336)
            presence[tok_h] = {p.id for p in resp.points if p.score > 0}
        for i in rare_toks:
            for pid_ in presence.get(i, ()):
                cov[pid_] = cov.get(pid_, 0) + 1
        if _QHEAD_COUNTRY_TRACE:
            _LAST_QUERY_PLAN_TRACE["qdrant_coverage_batch_calls"] += 1
            _LAST_QUERY_PLAN_TRACE["qdrant_total_batch_calls"] += 1

    # ADR-099 Ruling 3 arm (b), materialized: the pool ids whose stored BM25
    # vector contains at least one surviving high-IDF query term. Empty (and
    # therefore inert) whenever the lever is not armed.
    _tf_abs_hit: set = set()
    for _h in _tf_abs_toks:
        _tf_abs_hit |= presence.get(_h, set())

    # task 411 (c): the required-term bonus map. Pool-conditioned exactly like
    # the 404/406 name levers — if nobody in the pool matches every required
    # word, nothing distinguishes and the lever stands down.
    _reqterm = {}
    _reqterm_n = len(_reqterm_groups)
    if _REQTERM_W and _reqterm_n >= _REQTERM_MINGROUPS and payloads:
        _hits = {pid: sum(1 for _w, hs in _reqterm_groups
                          if any(pid in presence.get(h, ()) for h in hs))
                 for pid in payloads}
        if any(v == _reqterm_n for v in _hits.values()):
            for pid_, v in _hits.items():
                if _REQTERM_MODE == "frac":
                    _reqterm[pid_] = _REQTERM_W * v / _reqterm_n
                elif v == _reqterm_n:
                    _reqterm[pid_] = _REQTERM_W
            if _REQTERM_MODE == "hard":
                # measured-only arm: prove what a HARD requirement costs.
                payloads = {pid_: pl for pid_, pl in payloads.items()
                            if _hits.get(pid_) == _reqterm_n}

    # The common candidate head consumes retrieval evidence from this same
    # serving pass.  Do not reconstruct it in the miner and do not substitute
    # title/snippet text: these are scores/ranks/presence checks over the stored
    # dense and BM25 vectors. Competition rank gives tied scores the same value,
    # avoiding process-order noise for UUID point ids.
    _rh_dense_raw = raw.get("dense", {})
    _rh_bm25_raw = raw.get("bm25", {})
    _rh_dense_rank = _reciprocal_competition_rank(_rh_dense_raw)
    _rh_bm25_rank = _reciprocal_competition_rank(_rh_bm25_raw)
    _rh_bm25_best = max((s for s in _rh_bm25_raw.values() if s > 0.0), default=0.0)
    _rh_coverage_available = bool(rare_toks and payloads)
    return _name_primaries, _reqterm, _reqterm_groups, _rh_bm25_best, _rh_bm25_rank, _rh_bm25_raw, _rh_coverage_available, _rh_dense_rank, _rh_dense_raw, _tf_abs_armed, _tf_abs_hit, cov, payloads, presence


def build_rows(*, _name_primaries, _reqterm, authority_on, base_off_country, commercial, cov, hubfloor_on,
        k_bm25_scale, k_dense_scale, k_exact, k_machine, k_off_country, k_recency, k_title, lang,
        namegate_tokens, norm, payloads, presence, q, q_academic, q_cantons, q_content, q_country,
        q_country_soft, q_terms, rare_toks, scope_conf, scoped, target_bm25_weight, translated_bm25_weight,
        w_canton, w_foreign):
    """Stage 2b: relevance / penalty rows for the pool (S10 step 8b)."""
    # penalties = the additive-in-every-mode signals (recency/academic/machine).
    if _QHEAD_COUNTRY_TRACE:
        # Point ids, rather than URLs alone, are the join key: many chunks share
        # a URL.  The gate links a returned gold URL -> point id -> the earlier
        # country_rescue_ids evidence without changing the public response.
        _LAST_QUERY_PLAN_TRACE["candidate_urls"] = {
            str(pid): pl.get("url") for pid, pl in payloads.items()}
    rows = []  # (pid, pl, rel_core, penalties, trust_tier)
    # task 2792: exact-article-in-title bonus (weight 0 = inert; no article reference = empty set).
    _art_w = max(0.0, _settings.current("AW_ARTICLE_EXACT_W"))  # a negative setting is never a penalty
    _art_ids = _query_article_ids(q) if _art_w else frozenset()
    # task 2797: a statute named in the query keeps the boost only on candidates carrying it (removes, never adds).
    _art_statutes = (_query_statutes(q) if _art_ids and _settings.current("AW_ARTICLE_EXACT_STATUTE")
                     else frozenset())
    auth_defer = set()  # AUTHORITY: pids a demotion lever owns (no boost) — ADR-019
    _ng_cov = {}        # task 404 (a): per-pid name-token coverage (person queries)
    # task 406: the name-token coverage PRE-PASS. 404 computed coverage inside
    # the scoring loop because its only consumer (_apply_namegate) ran after it;
    # the 406 halves act INSIDE rel_core and need the POOL CONDITION — "some
    # in-pool candidate carries the full name" — before the first row is scored.
    # Payload-only and identical to the 404 in-loop computation, so the coverage
    # map (and therefore the 404 penalty) is unchanged.
    # task 411 (a): AW_NAMECOV_SURFACE picks the surface the coverage test reads.
    # "display" (default) is byte-identical to what ships; "sparse"/"union" add
    # the STORED BM25 vector, which is the ONLY surface an ADR-060 entity card
    # carries its name/alias/register-people tokens on (see the 411 block note).
    if namegate_tokens:
        for _pid, _pl in payloads.items():
            _d = _name_token_coverage(namegate_tokens, _pl)
            if (_NAMECOV_SURFACE == "display"
                    or (_NAMECOV_SURFACE == "card_sparse"
                        and (_pl.get("source_type") or "").lower()
                        not in _NAMECOV_SPARSE_CLASSES)):
                _ng_cov[_pid] = _d
            else:
                _s = _sparse_name_coverage(namegate_tokens, _name_primaries,
                                           _pid, presence)
                _ng_cov[_pid] = _s if _NAMECOV_SURFACE == "sparse" else max(_d, _s)
    # PERSON-NAME INTENT, armed: the query is name-shaped AND the pool proves the
    # full name is matchable. Both 406 halves hang off this one condition (see
    # the block comment) — it is what keeps "Marie Curie" safe and what stops a
    # dense down-weight from stripping the only signal on an unprovable name.
    _name_intent = bool(namegate_tokens
                        and any(c >= _NAMEGATE_COV for c in _ng_cov.values()))
    # H1 (Tim's half a): the per-query dense scale under person-name intent.
    # Equals k_dense_scale when the lever is inert => byte-identical rel_core.
    _k_dense_name = (k_dense_scale * _NAMEDENSE_SCALE
                     if (_name_intent and _NAMEDENSE) else k_dense_scale)
    # task 665 (AW_VERSION_MATCH): the query's dotted version tokens, armed only
    # when some pool candidate's title names one (pool condition, see block).
    _q_versions = schema.version_tokens(q) if _VERSION_MATCH else frozenset()
    _v_title_hit = ({pid for pid, pl in payloads.items()
                     if _title_versions_match(_q_versions, pl.get("title", ""))}
                    if _q_versions else set())
    if not _v_title_hit:
        _q_versions = frozenset()
    for pid, pl in payloads.items():
        exact = _exact_v(q_terms, _q_versions, pl)
        if cov:  # rare-token full-doc coverage backs up the title-only exact
            exact = max(exact, cov.get(pid, 0) / len(rare_toks))
        _dn = norm.get("dense", {}).get(pid, 0.0)
        _title_c = _title_centrality_v(q_content, _q_versions,
                                       pl.get("title", ""), lang)
        _v_bonus = _VERSION_MATCH_W if pid in _v_title_hit else 0.0
        # ADR-062: knob values resolve to today's static default when the head is
        # off / cold-start (alpha=0) -> byte-identical. dense/bm25 are SCALES (1/2),
        # exact/title/recency/machine are the weight itself (5/6/4/9).
        rel_core = (_W_DENSE_EFF * _k_dense_name * _dn
                    + WEIGHTS["lexical"] * norm.get("lexical", {}).get(pid, 0.0)
                    + _W_BM25_EFF * k_bm25_scale * norm.get("bm25", {}).get(pid, 0.0)
                    + target_bm25_weight * norm.get("target_bm25", {}).get(pid, 0.0)
                    + translated_bm25_weight * norm.get("translated_bm25", {}).get(pid, 0.0)
                    + k_exact * exact
                    + k_title * _title_c
                    + _v_bonus)          # task 665 (0.0 unless armed)
        penalties = (k_recency * _recency(pl.get("last_seen"))
                     + WEIGHTS.get("academic", 0.0) * _academic_penalty(pl.get("url", ""), q_academic)
                     + k_machine * _machine_url(pl.get("url", ""))
                     + WEIGHTS.get("parasite", 0.0) * _parasite(pl.get("url", "")))
        if _DEBUG_SCORES:    # task 583 — split both buckets into named terms
            _dbgc(pid, "rel.dense", _W_DENSE_EFF * _k_dense_name * _dn)
            _dbgc(pid, "rel.lexical",
                  WEIGHTS["lexical"] * norm.get("lexical", {}).get(pid, 0.0))
            _dbgc(pid, "rel.bm25",
                  _W_BM25_EFF * k_bm25_scale * norm.get("bm25", {}).get(pid, 0.0))
            _dbgc(pid, "rel.target_bm25",
                  target_bm25_weight * norm.get("target_bm25", {}).get(pid, 0.0))
            _dbgc(pid, "rel.translated_bm25",
                  translated_bm25_weight
                  * norm.get("translated_bm25", {}).get(pid, 0.0))
            _dbgc(pid, "rel.exact", k_exact * exact)
            _dbgc(pid, "rel.title", k_title * _title_c)
            _dbgc(pid, "rel.version", _v_bonus)
            _dbgc(pid, "pen.recency", k_recency * _recency(pl.get("last_seen")))
            _dbgc(pid, "pen.academic", WEIGHTS.get("academic", 0.0)
                  * _academic_penalty(pl.get("url", ""), q_academic))
            _dbgc(pid, "pen.machine", k_machine * _machine_url(pl.get("url", "")))
            _dbgc(pid, "pen.parasite", WEIGHTS.get("parasite", 0.0)
                  * _parasite(pl.get("url", "")))
        # task 144: the cantonal-scope penalty is jurisdiction-INDEPENDENT and
        # applied to EVERY query (a cantonal statute is authoritative only for a
        # query naming that canton). Stage-2 only for unscoped queries — the
        # fast payload-free stage-1 path is untouched; for scoped queries the
        # stage-1 canton demotion (123) still runs via `scope1`. foreign_gov
        # stays query-conditioned (task 123): only when the query names a
        # jurisdiction.
        if (w_canton and (_CANTON_UNCONDITIONAL or scoped)
                and pl.get("canton") and pl.get("canton") not in q_cantons):
            penalties += w_canton
            if _DEBUG_SCORES:
                _dbgc(pid, "pen.canton", w_canton)
        # task 282: track whether a COUNTRY lever demoted this candidate, so the
        # hub-floor DEFERS to it (§2.4 — no double penalty on a geo-scoped,
        # stamped-off-country hub; the country_scope head owns that partition).
        geo_penalized = False
        if scoped:  # foreign_gov — additive in every trust mode, like the others
            _, p_f = _scope_penalties(q_cantons, q_country, pl.get("canton"),
                                      _effective_country(pl), pl.get("source_type"),
                                      _is_eu_supra(pl.get("domain") or pl.get("url")))
            penalties += w_foreign * p_f
            geo_penalized = geo_penalized or bool(p_f and w_foreign)
            if _DEBUG_SCORES:
                _dbgc(pid, "pen.foreign_gov", w_foreign * p_f)
        # task 240: off-country demotion (geo-scoped country gate). Independent
        # of `scoped`/foreign_gov — fires on ANY geo-scoped query for the non-official
        # classes foreign_gov misses (entity_card/content). Additive in every trust
        # mode; inert when _OFF_COUNTRY_W == 0. ADR-062 knob 13: the effective scope
        # is the DETERMINISTIC q_country OR the LEARNED head-inferred soft scope —
        # off-country is the ONLY lever a head-inferred scope drives (knob 12), and
        # since it can only demote a CONFLICTING-country stamped candidate (never an
        # INTL/in-country gold), a mis-inference cannot regress a gold's rank.
        _eff_scope = q_country or q_country_soft
        if (_OFF_COUNTRY_W or _OFF_COUNTRY_NONBOUND_W
                or (_LEARNED_ONLY_SCOPE and base_off_country)) and _eff_scope:
            # task 259: a REGISTERED business card keeps its register country
            # and stays demotable (task 240); a register-less card does not.
            _has_reg = bool(pl.get("register_id") or pl.get("lei"))
            if _LEARNED_ONLY_SCOPE and q_country is None and q_country_soft:
                # Learned-only scope owns one confidence-scaled conflict signal
                # across source classes, including exact `official`.  The legacy
                # split left those candidates untouched because foreign_gov sees
                # only deterministic q_country.  All existing safety exits in
                # _off_country_hit (same-country, EU/INTL/unstamped, EU-supra)
                # remain in force.
                _learned_penalty = _learned_scope_conflict_penalty(
                    _eff_scope, scope_conf, pl, base_off_country)
                _hit = None
            else:
                _hit = _off_country_hit(
                    _eff_scope, _effective_country(pl), pl.get("source_type"),
                    _is_eu_supra(pl.get("domain") or pl.get("url")), _has_reg)
            if _hit is None:
                penalties += _learned_penalty
                geo_penalized = geo_penalized or bool(_learned_penalty)
                if _DEBUG_SCORES:
                    _dbgc(pid, "pen.scope_conflict", _learned_penalty)
            elif _hit:
                # task 260: non-location-bound content may be demoted by a
                # DIFFERENT (graded) weight. Inert unless AW_OFF_COUNTRY_NONBOUND_W
                # is set, in which case this is byte-identical to task 240.
                # ADR-062 knob 12: the base off-country penalty MAGNITUDE is the
                # head-driven knob (scope stays deterministic, Q2). k_off_country
                # == _OFF_COUNTRY_W at cold-start (byte-identical).
                _w = k_off_country
                if (_OFF_COUNTRY_NONBOUND_W is not None
                        and not _country_is_binding(pl.get("source_type"), _has_reg)
                        and _nonbound_class(pl.get("source_type"), _has_reg)
                            in _OFF_COUNTRY_NONBOUND_CLASSES):
                    _w = _OFF_COUNTRY_NONBOUND_W
                penalties += _w * _hit
                geo_penalized = geo_penalized or bool(_w)
                if _DEBUG_SCORES:
                    _dbgc(pid, "pen.off_country", _w * _hit)
        # task 254: on a commercial query, demote the encyclopedia (reference)
        # so actual providers/entity-cards outrank Wikipedia. Additive (<= 0),
        # composes with every trust mode; inert unless AW_COMMERCIAL fired.
        if commercial and pl.get("source_type") == "reference":
            penalties += _COMMERCIAL_REF_W
            if _DEBUG_SCORES:
                _dbgc(pid, "pen.commercial_ref", _COMMERCIAL_REF_W)
        # task 404 (a): name-token coverage is now captured in the PRE-PASS above
        # (task 406 needs the pool condition before scoring); the 404 demotion
        # itself is still POOL-CONDITIONED and applied after the loop.
        # task 406 H2 (Tim's half b): "increases the importance of all terms
        # being in the sparse" — a bounded rel_core BONUS for the candidate whose
        # visible text carries EVERY name token. In rel_core (not penalties) so
        # the trust relgate / trustfloor exemption / authority relgate all see
        # the page that provably names the person as genuinely relevant.
        if _NAMECOV_W and _name_intent and _ng_cov.get(pid, 0.0) >= _NAMEGATE_COV:
            rel_core += _NAMECOV_W
            if _DEBUG_SCORES:
                _dbgc(pid, "rel.namecov", _NAMECOV_W)
        # task 411 (c): Tim's generalisation of the same idea past person names
        # — "extract terms from the query that must match bm25, and score higher
        # if all match". Same seam as H2 (rel_core, so the trust relgate and the
        # trustfloor exemption both see it), same pool condition, 0.0 == inert.
        if _reqterm:
            rel_core += _reqterm.get(pid, 0.0)
            if _DEBUG_SCORES:
                _dbgc(pid, "rel.reqterm", _reqterm.get(pid, 0.0))
        if _art_ids and _article_exact_narrowed(_art_ids, _art_statutes, pl.get("title", ""), pl.get("url", "")):
            rel_core += _art_w
            if _DEBUG_SCORES:
                _dbgc(pid, "rel.article_exact", _art_w)
        # HUB-FLOOR (task 282): the relevance-riding-hub flood (class C). On a
        # general-intent query (hubfloor_on), a large high-trust AUTHORITY hub (the
        # serving hub table) whose TITLE is NOT central to the query — a generic-token
        # match (low title-centrality = low distinctiveness) — takes a bounded
        # additive penalty on its TOTAL (acts on rel_core+trust together, unlike the
        # trust head/hub-gate). DEFERS to the country levers (geo_penalized) so no
        # candidate takes both. NEVER fires on authority intent (rh_beta->0 excluded
        # upstream) nor on a hub whose title owns the query's distinctive tokens ->
        # legal/authority hold preserved. Weight <= 0, default 0.0 = inert.
        hubfloored = False
        if (hubfloor_on and not geo_penalized
                and _dn <= _HUBFLOOR_DENSE_MAX
                and _trust_tier(pl) <= _HUBFLOOR_MAXTRUST):
            _hubn = _hubness(pl)
            if _hubn > 0 and _hubn >= _HUBFLOOR_DF_MIN:
                _distinct = _title_centrality(q_content, pl.get("title", ""), lang)
                if _distinct <= _HUBFLOOR_DISTINCT_MAX:
                    penalties += _HUBFLOOR_W * _hubn
                    hubfloored = True
                    if _DEBUG_SCORES:
                        _dbgc(pid, "pen.hubfloor", _HUBFLOOR_W * _hubn)
        # AUTHORITY (ADR-019 product-1): the endorsement boost is a SECOND pass
        # (below), because it is RELEVANCE-GATED against the pool-best rel_core —
        # unknown until every row is scored (same pattern as official_pref /
        # entity_card / country). Here we only record whether this candidate must
        # be EXCLUDED from the boost because a demotion lever already owns it:
        # authority DEFERS to the hub-floor (hubfloored) and to the COUNTRY levers
        # (geo_penalized), so the two never fight / double-count.
        if authority_on and (hubfloored or geo_penalized):
            auth_defer.add(pid)
        rows.append([pid, pl, rel_core, penalties, _trust_tier(pl)])
    return _k_dense_name, _name_intent, _ng_cov, auth_defer, rows


def apply_levers(*, _ng_cov, _rh_bm25_best, _rh_bm25_rank, _rh_bm25_raw, _rh_coverage_available,
        _rh_dense_rank, _rh_dense_raw, auth_defer, authority_on, cov, lang, namegate_tokens, norm, q_cantons,
        q_content, q_country, q_country_soft, qk, rare_toks, rh_beta, rows, scope_conf,
        host_label=None):
    """Stage 2c: pool-conditioned levers on the rows, rank heads, trust gate, primacy (S10 step 8b)."""
    # task 404 (a): NAME-GATE — pool-conditioned half-match demotion on a
    # person-name-shaped query (see the _NAMEGATE block comment). Demoted
    # candidates join auth_defer so the authority boost cannot fight the
    # demotion (mirrors hubfloor / country-lever deferral).
    # task 406 H3(ii): the set of rows a demotion lever owns, so the trustfloor's
    # rel-ratio exemption can measure against the best NON-DEMOTED rel_core.
    _demoted_pids: set = set()
    # `active` keeps the 404 semantics exactly: the 406 halves can arm the shape
    # detector on their own, but only AW_NAMEGATE turns on the 404 PENALTY.
    for _ng_pid in _apply_namegate(rows, _ng_cov,
                                   bool(namegate_tokens) and _NAMEGATE):
        _demoted_pids.add(_ng_pid)
        if authority_on:
            auth_defer.add(_ng_pid)

    # Official-source preference (task 144, class-2 Wikipedia-over-official).
    # On a jurisdiction-scoped grounding query where the pool holds a COMPETITIVE
    # official candidate (rel_core >= beta * best rel_core), demote encyclopedic
    # reference-tier candidates so the official page outranks Wikipedia. The
    # pool condition is what keeps this from being a blanket wikipedia demotion:
    # on a backbone/encyclopedic query (no competitive official in the pool) it
    # never fires and Wikipedia — a legitimate gold there — is untouched.
    # Weight <= 0, default 0.0 = inert (byte-identical scoring); rollback = 0.0.
    w_official = WEIGHTS.get("official_pref", 0.0)
    _opref_fired = False
    if w_official and bool(q_cantons or q_country) and rows:
        best_rel = max(r[2] for r in rows)
        has_official = any(r[1].get("source_type") == "official"
                           and r[2] >= _OFFICIAL_PREF_BETA * best_rel for r in rows)
        if has_official:
            _opref_fired = True
            for r in rows:
                if r[1].get("source_type") == "reference":
                    r[3] += w_official
                    if _DEBUG_SCORES:
                        _dbgc(r[0], "pen.official_pref", w_official)

    # task 404 (b): REF-CANON — official_pref's cure extended to UNSCOPED
    # authority-lookup queries (403 §8.4 eviction-by-worse), intent-gated on
    # the ADR-036 head (inert when the head is off) and exempt on card-named
    # entity queries. Defers to official_pref (no double demotion); demoted
    # pids defer the authority boost (no partial refund). Inert at the
    # default W=0.0; see the _REFCANON block comment.
    for _rc_pid in _apply_refcanon(
            rows, q_content, lang,
            head_on=bool(_RANKHEAD and _RANKHEAD_MODEL is not None),
            rh_beta=rh_beta, official_pref_fired=_opref_fired):
        _demoted_pids.add(_rc_pid)
        if _DEBUG_SCORES:
            # task 583: recorded HERE and not inside `_apply_refcanon`, which
            # adds exactly `_REFCANON_W` to each pid it returns. The task-404 /
            # 406 / 578 lever tests exec that function's source in a namespace
            # of their own, so a module global referenced inside it is a
            # NameError there — the ledger belongs at the call site.
            _dbgc(_rc_pid, "pen.refcanon", _REFCANON_W)
        if authority_on:
            auth_defer.add(_rc_pid)

    # task 164: pool-best rel_core, to relevance-gate the country soft-boost.
    best_rel_core = max((r[2] for r in rows), default=0.0)
    # task 406 H3(ii) — AW_TRUSTFLOOR_UNDEMOTED: the trustfloor's rel-ratio
    # exemption measures against the best rel_core among rows NO lever demoted.
    # Measured motivation (405 §4): the n01 gold's ratio was 1.289/2.307 = 0.559
    # only because a namegate-DEMOTED wiki body chunk set `best_rel_core` — a row
    # the system had already ruled against was setting the bar that stripped the
    # gold's trust. Falls back to best_rel_core when every row is demoted.
    best_rel_undemoted = best_rel_core
    if _TRUSTFLOOR_UNDEMOTED and _demoted_pids:
        best_rel_undemoted = max((r[2] for r in rows if r[0] not in _demoted_pids),
                                 default=best_rel_core)

    # AUTHORITY (ADR-019 product-1) — the endorsement boost, applied here so it can
    # be RELEVANCE-GATED on the pool-best rel_core (mirrors official_pref / country /
    # legifrance_bulk). A bounded additive lift (>=0) on r[3] (penalties bucket —
    # additive in every trust mode) for a genuinely-endorsed host, ONLY when the
    # candidate is relevance-COMPETITIVE (rel_core >= beta * best) — so a weakly-
    # relevant authority host can NOT ride its endorsement past a strongly-relevant
    # gold (the failure mode ADR-019 Phase-1 flagged: authority displacing low-
    # citation golds). Excludes pids a demotion lever already owns (auth_defer:
    # hub-floored / country-demoted) — authority DEFERS, never double-counts. This
    # COMPLEMENTS trust_tier (curated prior) with the EARNED citation axis, bounded
    # so it sharpens order within a band rather than restacking it. Optional
    # distinctiveness gate (MIN_CENTRALITY) keeps it to in-title matches. Weight >=0,
    # default 0.0 = inert (byte-identical); rollback = AW_AUTHORITY=0 / W=0.0.
    if authority_on and best_rel_core > 0:
        for r in rows:
            if r[0] in auth_defer:
                continue
            if r[2] < _AUTHORITY_RELGATE * best_rel_core:   # not rel-competitive
                continue
            _auth = _authority(r[1])
            if _auth < _AUTHORITY_MIN or _auth <= 0.0:
                continue
            if _AUTHORITY_MIN_CENTRALITY > 0.0 and _title_centrality(
                    q_content, r[1].get("title", ""), lang) < _AUTHORITY_MIN_CENTRALITY:
                continue
            r[3] += min(_AUTHORITY_W * _auth, _AUTHORITY_MAXBOOST)
            if _DEBUG_SCORES:
                _dbgc(r[0], "pen.authority",
                      min(_AUTHORITY_W * _auth, _AUTHORITY_MAXBOOST))

    # task 165: legifrance bulk-legal demotion, rel-gated on best_rel_core (a bulk
    # article that is NOT competitive with the pool's best relevance is flood, not
    # the answer -> demote so it can't ride the trust floor above the real source).
    w_lfbulk = WEIGHTS.get("legifrance_bulk", 0.0)
    if w_lfbulk and rows and best_rel_core > 0:
        for r in rows:
            if _lf_bulk(r[1].get("url", "")) and r[2] < _LFBULK_RELGATE * best_rel_core:
                r[3] += w_lfbulk
                if _DEBUG_SCORES:
                    _dbgc(r[0], "pen.legifrance_bulk", w_lfbulk)

    # Entity-card return-band boost (task 163 v2). Positive, double-gated mirror
    # of official_pref, with same-name DISAMBIGUATION by prominence. Eligible
    # cards (pool-cond + name-match) are collapsed by registrable domain and
    # ordered by (name_match, sitelinks prominence, rel_core); only the top `cap`
    # are boosted, so a query surfaces the CANONICAL card and less-notable
    # namesakes stay at their relevance rank. Added to r[3] (additive in every
    # trust mode). Weight >= 0, default 0.0 = inert; rollback = entity_card=0.0.
    w_ec = qk("entity_card", WEIGHTS.get("entity_card", 0.0))  # ADR-062 knob 7
    if w_ec and rows:
        best_rel = max(r[2] for r in rows)
        elig = []
        for r in rows:
            pl = r[1]
            if pl.get("source_type") != "entity_card":
                continue
            if r[2] < _ENTITY_CARD_BETA * best_rel:      # not a competitive match
                continue
            if host_label:   # AW_HOST_QUERY_CARD (task 2050); None with the flag off
                nm = _card_name_match(q_content, pl.get("title", ""), pl.get("names", ""), lang,
                                      host_label=host_label, card_url=pl.get("url", ""))
            else:
                nm = _card_name_match(q_content, pl.get("title", ""),
                                      pl.get("names", ""), lang)
            if nm >= _ENTITY_CARD_NAME_THR:
                elig.append((nm, int(pl.get("sitelinks") or 0), r[2], r))
        winners = _card_winners(elig)
        for _, _, _, r in winners[:_ENTITY_CARD_CAP]:
            r[3] += w_ec
            if _DEBUG_SCORES:
                _dbgc(r[0], "pen.entity_card", w_ec)

    # RANKHEAD V2 (task 350) — the LAST additive stage-2 adjustment, so every
    # lever above (official_pref/authority/legifrance_bulk/entity_card) saw
    # unchanged inputs and in additive mode final = base_fused + delta exactly
    # (the trust term reads rel_core/best_rel_core, never penalties). Dump mode
    # records the identical feature vectors the head would consume — the miner
    # trains on the serving code path itself (no re-implementation, no skew).
    if _RANKHEAD_V2_MODEL is not None or _RANKHEAD_V2_DUMP:
        from . import rankhead_v2 as _rh2
        _v2_scope = q_country or q_country_soft
        _v2_conf = 1.0 if q_country else (scope_conf if q_country_soft else 0.0)
        _v2_qw = None
        if _RANKHEAD_V2_MODEL is not None:
            _v2_qfull = models.last_full_query_vec()
            if _v2_qfull is not None:
                _v2_qw = _rh2.query_weights(_RANKHEAD_V2_MODEL, _v2_qfull)
        if _RANKHEAD_V2_DUMP:
            _LAST_V2_DUMP.clear()
        for r in rows:
            _pl2 = r[1]
            _f2 = _rh2.doc_features(
                _pl2,
                authority=_authority(_pl2),
                country=_effective_country(_pl2),
                eu_supra=_is_eu_supra(_pl2.get("domain") or _pl2.get("url")),
                scope=_v2_scope, scope_conf=_v2_conf,
                title_centrality=_title_centrality(
                    q_content, _pl2.get("title", ""), lang),
                rel_ratio=(r[2] / best_rel_core) if best_rel_core > 0 else 0.0,
                dense_norm=norm.get("dense", {}).get(r[0], 0.0),
                dense_raw=_rh_dense_raw.get(r[0], 0.0),
                dense_rank=_rh_dense_rank.get(r[0], 0.0),
                bm25_raw=(_rh_bm25_raw.get(r[0], 0.0) / _rh_bm25_best
                          if _rh_bm25_best else 0.0),
                bm25_norm=norm.get("bm25", {}).get(r[0], 0.0),
                bm25_rank=_rh_bm25_rank.get(r[0], 0.0),
                sparse_coverage=(cov.get(r[0], 0) / len(rare_toks)
                                 if rare_toks else 0.0),
                coverage_available=_rh_coverage_available)
            _d2 = 0.0
            if _v2_qw is not None:
                _d2 = _rh2.delta(_RANKHEAD_V2_MODEL, _v2_qw, _f2)
                r[3] += _d2
                if _DEBUG_SCORES:
                    _dbgc(r[0], "pen.rankhead_v2", _d2)
            if _RANKHEAD_V2_DUMP:
                _LAST_V2_DUMP.append({
                    "pid": str(r[0]), "url": _pl2.get("url"),
                    "feats": [round(float(v), 5) for v in _f2],
                    "rel_core": round(float(r[2]), 5),
                    "trust_tier": float(r[4]), "v2_delta": round(_d2, 5)})

    # gate/rank_scaled need the pool-wide rel_core distribution.
    gate_thr = None
    rel_rank = {}
    if _TRUST_MODE in ("gated", "rank_scaled") and rows:
        rels = sorted((r[2] for r in rows))
        if _TRUST_MODE == "gated":
            gate_thr = rels[min(len(rels) - 1, int(_TRUST_GATE_PCTL * len(rels)))]
        else:  # rank_scaled: 0 = best rel_core
            for i, r in enumerate(sorted(rows, key=lambda r: r[2], reverse=True)):
                rel_rank[r[0]] = i

    # TASK 578 — PRIMACY-KEEP's beneficiary set, computed ONCE per query over
    # the same `rows` every lever above has already seen. Empty (and so
    # byte-identical) unless AW_PRIMACY=1 with a non-zero AW_PRIMACY_KEEP.
    # `_title_centrality` is the expensive half, so it is evaluated last and
    # only for rows that already cleared the class and competitiveness bars.
    _primacy_pids: set = set()
    if _PRIMACY and _PRIMACY_KEEP > 0.0 and rows and best_rel_core > 0:
        for r in rows:
            if not _is_primary_class(r[1].get("source_type")):
                continue
            if r[2] < _PRIMACY_BETA * best_rel_core:
                continue
            if (_PRIMACY_MINCENT > 0.0
                    and _title_centrality(q_content, r[1].get("title", ""),
                                          lang) < _PRIMACY_MINCENT):
                continue
            _primacy_pids.add(r[0])
    return _primacy_pids, best_rel_core, best_rel_undemoted, gate_thr, rel_rank
