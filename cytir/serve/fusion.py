"""Fusion helpers, trust-floor and primacy gates, and the total tie order (S10 step 7, task 1111;
moved verbatim from `retrieval.py`, which imports every name back).

The stage-2 trust-floor decision (`AW_TRUSTFLOOR_ABS`, ADR-099), the primacy refund, the dense
boost, and the deterministic tie order of tasks 758/841 (`_order_ids`, `_order_scored`,
`_url_tie_len`) with the authority-leg pool helpers. Constants travel with their only readers;
`_PRIMACY_KEEP` is also read by `search()`, which imports it from here. No import of `retrieval`.
"""
from __future__ import annotations

import math

from cytir.core.country import is_location_bound

from .settings import SETTINGS as _S
# S10 step 8b (task 1125): stage 2, levers and respond moved out of retrieval.search()
from .trace import (_DEBUG_SCORES, _LAST_COMPONENTS, _LAST_EXPERIMENT_TRACE, _LAST_SCORE_DEBUG, _dbgc)
from .geo import _effective_country
from .signals import _trust_tier

# Absolute raw-dense-cosine floor: below TAU the trust term is ramped down. 0.0 =>
# inert (nothing is below it). ~0.6 aligns with the _RELFLOOR_DENSE_MAX=0.68
# "no strong semantic match" band; calibrate per the twin before shipping.
_TRUSTFLOOR_TAU = _S.trust.trustfloor_tau
# Residual floor on the ramp: the minimum fraction of the trust term that survives
# even at cos→0 (0.0 = can fully strip; raise to soften). Keeps a dense-absent
# candidate from being fully trust-stripped if that ever over-fires.
_TRUSTFLOOR_FLOOR = 0.0  # fixed: was env AW_TRUSTFLOOR_FLOOR (S10 D5: no product setter)
# fraction of the attenuation refunded; 0.0 == inert, 1.0 == full exemption.
_PRIMACY_KEEP = _S.fusion.primacy_keep
# classes that count as primary ON TOP of the location-bound set
# (_is_location_bound: official*/gov*/case_law/central-bank/registry_record).
# `docs` and `model-hub` are the two families the population names that the
# location-bound rule does not reach (16 + 4 addressable pairs).
_PRIMACY_EXTRA = frozenset(
    t.strip().lower()
    for t in _S.fusion.primacy_extra_classes.split(",")
    if t.strip())


def _trustfloor_exempt(rel_core: float, best_rel_undemoted: float,
                       rel_beta: float, *, abs_on: bool, dense_raw: float,
                       abs_dense_floor: float, abs_term_hit: bool) -> bool:
    """Does this candidate keep its FULL trust term (i.e. escape attenuation)?

    Pure function of the scores, so the ADR-099 Ruling 3 decision is unit-testable
    without a qdrant (tests/test_serve_adr099_trustfloor_abs.py). With `abs_on=False`
    it is the pre-ADR-099 condition, term for term."""
    rel_ok = bool(best_rel_undemoted > 0
                  and rel_core >= rel_beta * best_rel_undemoted)
    if not rel_ok:
        return False                      # pool-relative test: still NECESSARY
    if not abs_on:
        return True                       # ... and, before ADR-099, sufficient
    return bool(dense_raw >= abs_dense_floor or abs_term_hit)


def _trustfloor_scale(dense_raw: float) -> float:
    """Multiplicative attenuation of the trust term from the candidate's ABSOLUTE
    raw dense cosine. 1.0 (no change) at/above TAU; a smooth linear ramp down to
    FLOOR at cos=0. Caller gates on intent (rh_beta) + MAXTRUST before applying."""
    if _TRUSTFLOOR_TAU <= 0.0:
        return 1.0
    if dense_raw >= _TRUSTFLOOR_TAU:
        return 1.0
    return max(_TRUSTFLOOR_FLOOR, dense_raw / _TRUSTFLOOR_TAU)


def _is_primary_class(source_type) -> bool:
    """AW_PRIMACY's beneficiary set: the location-bound classes plus
    AW_PRIMACY_EXTRA_CLASSES. Shares `_is_location_bound`'s prefix rule so an
    unenumerated `gov-*`/`official-*` family defaults IN rather than silently
    losing the refund (task 259 defect 1, same reasoning)."""
    st = (source_type or "").strip().lower()
    return bool(st) and (is_location_bound(st) or st in _PRIMACY_EXTRA)


def _primacy_refund(scale: float) -> float:
    """Refund `_PRIMACY_KEEP` of an attenuation. `scale` is the multiplier a
    gate wants to apply to the trust term (<= 1.0); the refunded multiplier
    moves it back toward 1.0. KEEP=0 returns `scale` unchanged (inert),
    KEEP=1 returns 1.0 (full exemption)."""
    k = min(1.0, max(0.0, _PRIMACY_KEEP))
    return scale + k * (1.0 - scale)


# --------------------------------------------------------------------------- #
#  TASK 758 — a TOTAL order for the ranking sorts (finding D6 of
#  specs/s10-serve-split.md, plan T11).
#
#  `search()` ranks a SET of string point ids. Sorting by score alone left
#  exact ties in set-iteration order, and Python salts string hashes per
#  process, so every container restart re-drew them: identical code under
#  another PYTHONHASHSEED answered 213 of 2,272 gate requests differently
#  (task 758, twin aw_e8). Exact ties are common. Most are the same text at
#  two URLs (language copies, tracking/parameter variants, version siblings),
#  which carries the same vectors and the same payload signals; some are
#  unrelated pages that score alike (21 of 170 measured pairs were two
#  entity cards of different companies).
#
#  Stage 2 breaks a tie by the SHORTER URL (scheme not counted), then by point
#  id; stage 1 (no payload yet) by point id. Shorter-first was chosen on the
#  measured ties (task 758, review 779): of the 21 first-differing tied pairs
#  where exactly one URL carries a query string, it puts the parameter-free URL
#  first in 18 (point id alone: a coin); the 3 exceptions include a CMS
#  dynamic-id URL beating its pretty slug. Between unrelated pages it is only a
#  short-domain preference: neutral. Every candidate key (none / published /
#  trust / last_seen / URL length) held the same golds (629 chooser + held-out
#  battery, 0 lost against all three incumbent seeds). Version siblings are
#  neutral under every key; a version the query names is AW_VERSION_MATCH's
#  job. Scores are compared exactly: a tie is equality, not a tolerance.
# --------------------------------------------------------------------------- #
def _url_tie_len(url) -> int:
    """Length of `url` without its scheme: the stage-2 tie key (shorter first).

    Raw characters after the first `://`, so it counts `www.`, a port, a
    trailing slash, the query, the fragment, and each percent-escape as 3
    characters; a URL without a scheme is counted whole. Consequences, all
    deliberate or harmless: a bare host beats `www.`, `/page` beats `/page/`,
    an ASCII slug beats a percent-encoded one, `#anx_XI` beats `#anx_XIII`."""
    return len((url or "").split("://", 1)[-1])


def _order_ids(ids, score) -> list:
    """Stage 1: point ids best-first by `score(pid)`; exact ties by point id.

    Stage 1 has no payload, so it cannot use stage 2's URL key: when a group of
    exactly tied rows straddles the margin cut, the copy stage 2 would prefer
    can be cut before stage 2 sees it. Rare (the cut is at >= 200 rows) and
    deterministic, but not stage 2's preference."""
    return sorted(ids, key=lambda pid: (-score(pid), str(pid)))


def _auth_admit(points, beta: float) -> dict:
    """The authority leg's admitted ids, as an ORDERED set (dict keys) in the
    leg's own ANN order: every point within `beta` of the leg's best score.
    Never a Python set: the ids join the stage-2 pool in this order, and a set
    of string ids iterates in hash-salt order (review 779 M1)."""
    if not points:
        return {}
    amax = max(p.score for p in points)
    return {p.id: None for p in points if p.score >= beta * amax}


def _extend_pool(pre: list, admit) -> list:
    """`pre` then the `admit` ids it lacks, each once, in `admit`'s order."""
    return list(dict.fromkeys(list(pre) + list(admit)))


def _order_scored(scored: list) -> None:
    """Stage 2, in place: rows `(fused, pid, payload)` by fused desc, then the
    shorter URL, then point id. A total order: no process state decides it."""
    scored.sort(key=lambda x: (-x[0], _url_tie_len((x[2] or {}).get("url")), str(x[1])))


# --------------------------------------------------------------------------- #
#  Stage-1 pool cut (S10 step 8a, task 1116): the closures `search()` carried, and the
#  leg-weight and country-boost constants they read. `WEIGHTS` is the same dict `retrieval` holds.
# --------------------------------------------------------------------------- #
WEIGHTS = _S.weights

# --------------------------------------------------------------------------- #
#  VOCABULARY / LANGUAGE ROBUSTNESS — dense-vs-BM25 fusion re-weighting (task 270).
#  When the query vocabulary/language does NOT lexically match the authoritative
#  page (synonym/register-shift/cross-language: "débarras/encombrants" vs a page
#  titled "déchets"; a DE statute cite vs an FR/IT/EN phrasing), the page earns
#  ZERO title-centrality (2.5) and ~0 BM25, while entity/same-name pages win on
#  title+BM25+trust — so the fusion buries a page the DENSE leg ranks high. Leg-
#  localization (task 270, twin aw_v4_sat): the addressable failures have the gold
#  DENSE-strong and inside the 100-candidate pool (dardagny "débarras" dense-rank5;
#  "Art.335c OR" dense-rank1) but fused #27/#4; the UN-addressable ones have the
#  gold BELOW the dense pool floor (dardagny "déchets volumineux" dense-rank234) or
#  BM25-present-but-dense-absent (335c FR dense-rank200/bm25-rank19) — no dense
#  weight recovers those (a pool/recall problem, ADR/253 Cembra finding: at 47.8M
#  the absolute dense cosine of a relevant OOD page can sit below noise-floor
#  pages, so it never enters the pool).
#
#  Lever = a DENSE-CORROBORATION additive boost: reward a candidate proportional
#  to how strongly the DENSE leg ranks it, so a semantically-matched page is not
#  buried for lacking the query's surface tokens. Smooth ramp above tau (no cliff),
#  applied in BOTH stage-1 (_s1rel, so a dense-strong candidate is admitted to the
#  margin pool) and stage-2 (rel_core). It ADDS relevance only — never demotes,
#  never touches penalties/trust — so a genuinely-lexical BM25 win is untouched;
#  it just lets the dense signal compete with the title(2.5)+trust(2.5) floor.
#  It is intent-AGNOSTIC on purpose (unlike the ADR-036 head): the buried golds
#  are BOTH gov-service (general) and statute (authority-lookup), so scoping to
#  general-intent would exclude the fedlex case. TRIPWIRE (measured, see the gate):
#  precise-lexical queries where BM25 SHOULD win (statute cites, exact signatures,
#  exact entity names) — the boost also lifts the exact gold (it is dense-strong
#  too) so it must not INVERT the exact answer to a dense-adjacent wrong page.
#  Optional weight overrides (AW_FUSION_DENSE_W / _BM25_W) stage a plain leg-weight
#  change without editing fusion_weights.json (the bm25 arm probes the complementary
#  lever for the BM25-present-dense-absent fedlex FR/IT variants). Flag-gated
#  NaN-default overrides => default OFF, byte-identical.
# --------------------------------------------------------------------------- #
# leg-weight overrides (NaN => use fusion_weights.json, byte-identical)
_FUSION_DENSE_W = float("nan")  # fixed: was env AW_FUSION_DENSE_W (S10 D5: no product setter)

_FUSION_BM25_W = float("nan")  # fixed: was env AW_FUSION_BM25_W (S10 D5: no product setter)

_W_DENSE_EFF = WEIGHTS["dense"] if math.isnan(_FUSION_DENSE_W) else _FUSION_DENSE_W

_W_BM25_EFF = WEIGHTS["bm25"] if math.isnan(_FUSION_BM25_W) else _FUSION_BM25_W

# Country/jurisdiction soft-boost (spec 2026-06-29, PR #20 ported to two-stage).
# When the caller selects a country, candidates whose stamped `country` payload
# matches get an additive boost -- a SOFT nudge, not a filter, so INTL sources
# (RFCs, Wikipedia, W3C) still rank. The boost is applied in BOTH stages: in
# stage-1 pool selection (else matches outside the top-margin pool never
# surface) and in the stage-2 full re-score. To feed stage-1, country queries
# fetch legs with a payload SELECTOR (only the small `country` field) instead
# of with_payload=False -- country queries pay a modest payload-read cost;
# no-country queries keep the fast payload-free path, byte-identical to today.
_COUNTRY_BOOST = 0.6  # fixed 2026-10-23 (S10 step 3): was AW_COUNTRY_BOOST, default 0.6, no setter since #968

# task 164: relevance-GATE the country soft-boost. The flat +0.6 was a
# relevance-independent floor (exactly the ADR-030 flat-trust failure mode):
# in the WEAK-relevance regime (a novel/tangential query where the whole pool
# scores low) it became the de-facto sort key and catapulted a weakly-relevant
# DOMESTIC page above a strongly-relevant supranational/foreign authority
# (live: "BCG McKinsey Bain AI regulatory playbook" + country=FR floated a
# barely-relevant Légifrance LODA article over the EU AI Act). We now apply the
# boost ONLY to a country-matched candidate whose relevance is COMPETITIVE with
# the pool's best (rel >= beta * best_rel) — pool-conditioned, like task 144's
# official_pref. A genuinely-domestic query (its best pool result IS the domestic
# source) still boosts and still wins; a strong foreign/EU authority can no
# longer be leapfrogged by a weak domestic result. Applied in BOTH stages
# (stage-1 gate keeps the strong foreign candidate inside the margin pool).
# Default ON; AW_COUNTRY_GATE=0 reverts to the flat boost (pre-164, byte-
# identical). beta env-overridable. The boost NEVER demotes (gate only withholds
# the additive bonus), so a gated candidate still ranks on its own merits.
_COUNTRY_GATE = True  # fixed: was env AW_COUNTRY_GATE (S10 D5: no product setter)

# beta=0.5: a domestic candidate must be at least half as relevant as the pool's
# best relevance to earn the boost. Measured (task 164): at 0.5 the BCG/AI-Act
# case is fixed (weak FR LODA rel_core-ratio ~0.23-0.37 < 0.5 => no boost) while
# genuine FR-law queries keep the FULL boost (legifrance ratio ~0.54-0.73 >= 0.5)
# — byte-identical top-k + scores to the pre-164 flat boost on those. The pool
# max is often an INTL reference (Wikipedia/arXiv) which normalizes the per-query
# relevance scale; the gate withholds the bonus only, it never demotes.
_COUNTRY_GATE_BETA = _S.country.country_gate_beta


def s1_rel(pid, *, norm, k_dense_scale, k_bm25_scale, target_bm25_weight, translated_bm25_weight):  # stage-1 leg-relevance proxy (no payload signals yet)
    dn = norm.get("dense", {}).get(pid, 0.0)
    # ADR-062 knobs 1/2: leg-weight SCALES applied in stage-1 too, so the
    # margin pool stays consistent with stage-2 if they ever leave 1.0.
    return (_W_DENSE_EFF * k_dense_scale * dn
            + WEIGHTS["lexical"] * norm.get("lexical", {}).get(pid, 0.0)
            + _W_BM25_EFF * k_bm25_scale * norm.get("bm25", {}).get(pid, 0.0)
            + target_bm25_weight * norm.get("target_bm25", {}).get(pid, 0.0)
            + translated_bm25_weight * norm.get("translated_bm25", {}).get(pid, 0.0))


def s1_boost(pid, *, country, cty1, s1_rel, best_s1):
    if country is None or cty1.get(pid) != country:
        return 0.0
    if _COUNTRY_GATE and s1_rel(pid) < _COUNTRY_GATE_BETA * best_s1:
        return 0.0
    return _COUNTRY_BOOST


def s1_key(pid, *, s1_rel, s1_boost, scope1):
    return (s1_rel(pid)
            + scope1.get(pid, 0.0)   # task 123 (<= 0, scoped only)
            + s1_boost(pid))


def page_of(pid, *, pg):
    return pg.get(pid) or f"__pid__:{pid}"


# --- S10 step 8b (task 1125): constants that travelled with their readers ---

_NAMEGATE_COV = _S.fusion.namegate_cov

_RELFLOOR = _S.trust.relfloor

# Source types whose trust term the commercial-intent lever scales (task 254; the intent
# test itself is `query._query_commercial_intent`).
_GOV_SOURCE_TYPES = frozenset({"official", "gov-stat", "central-bank"})

# --------------------------------------------------------------------------- #
#  Trust mode (task 103). The additive trust term (score += w_trust*trust_tier)
#  is a FLAT, relevance-independent floor: at w_trust=2.5 a trust-0.85 domain
#  contributes 2.125 BEFORE relevance matters, and because per-leg min-max
#  compresses the relevance band, trust became the de-facto sort key — ranking
#  by domain prestige, not relevance (canonical article buried under tangential
#  pages on .edu/.gov). These flag-gated modes make trust CONDITIONAL on
#  relevance so prestige can't override it. `additive` = unchanged legacy.
#    additive       : rel + penalties + w_trust*trust_tier          (default)
#    multiplicative : rel_core*(1 + alpha*trust_tier) + penalties
#                     trust SCALES relevance -> off-topic (low rel) can't
#                     inherit the boost. alpha is the trust strength.
#    gated          : rel + penalties + (w_trust*trust_tier IF the candidate's
#                     rel_core is >= the pool's `gate_pctl` percentile ELSE 0)
#                     -> trust breaks ties AMONG relevant results only.
#    rank_scaled    : rel + penalties + w_trust*trust_tier*rank_weight, where
#                     rank_weight decays linearly with the candidate's rel_core
#                     rank in the pool (rank 0 -> 1.0, tail -> 0).
#  rel_core = w_dense*dense + w_lexical*lex + w_bm25*bm25 + w_exact*exact (>=0);
#  penalties = w_recency*recency + w_academic*acad + w_machine*machine (recency
#  >=0, academic/machine <=0) stay ADDITIVE in every mode (a penalty must bite
#  regardless of trust). Country boost stays additive at the end (unchanged).
#  Config precedence: env AW_TRUST_* overrides the fusion_weights.json keys
#  (trust_mode / trust_alpha / trust_gate_pctl), which override the defaults.
# --------------------------------------------------------------------------- #
_TRUST_MODE = (_S.trust.trust_mode).lower()

_TRUST_ALPHA = _S.trust.trust_alpha

_COMMERCIAL_GOV_SCALE = _S.query.commercial_gov_scale

# --------------------------------------------------------------------------- #
#  CHANGE B — make-dense-earn-it: relevance-gate the BASE trust term (task 202).
#  The additive trust term (score += w_trust * trust_tier) is a FLAT, relevance-
#  independent floor (the ADR-030 failure mode): a high-trust page with weak
#  pool-relative relevance still gets the FULL bonus, so a coincidental keyword
#  collision floats it up (live 2026-07-13: "startup management tips" ->
#  nbb.be TARGET-"TIPS" payments regulation #8, purely because trust 0.95
#  amplifies "tips" the word ↔ "TIPS" the acronym). Fix (pool-conditioned, the
#  164/144 pattern): the FULL trust bonus is earned only when the candidate is
#  relevance-competitive with the pool's best (rel_core >= BETA * best_rel_core);
#  below that it is scaled DOWN by a smooth linear ramp (scale = ratio/BETA, so
#  scale is continuous = 1.0 at the threshold, 0.0 at zero relevance — NO cliff).
#  It only ATTENUATES a bonus (never demotes, never touches penalties or the
#  relevance legs), and it is purely arithmetic + stage-2 (trust is not a stage-1
#  signal, so the pool is unchanged). Applies ONLY to the additive mode (the
#  live one). TRIPWIRE: legal answers are legitimately high-trust-modest-dense —
#  a genuine statute is usually the pool's own best relevance (ratio ~1 => full
#  bonus), but this must be GATED on the legal slice before shipping. Flag-gated
#  AW_TRUST_RELGATE (default OFF, byte-identical); rollback = AW_TRUST_RELGATE=0.
# --------------------------------------------------------------------------- #
_TRUST_RELGATE = _S.trust.trust_relgate

_TRUST_RELGATE_BETA = _S.trust.trust_relgate_beta

_RELFLOOR_BETA = 0.5  # fixed: was env AW_RELFLOOR_BETA (S10 D5: no product setter)

#  HUB-GATE (task 210) — the BM25-hubness fix the trust relgate can't reach.
#  Diagnosis (task 210): the antirez.com flood on vague/general queries is NOT
#  dense hubness — antirez is dense host-rank #244 (51 top-100 appearances over
#  3.6k queries; 0 on the named floods). It is BM25 hubness: antirez is the #1
#  BM25 host (38,492 bm25 top-100 appearances; ahead of legifrance/canada.ca)
#  because its long discursive raw-tf essays + the per-token sub-legs (task 083,
#  weights are UNSATURATED tf*idf) match common query tokens. Every antirez result
#  in the fused top-50 enters BM25-ONLY (in_dense=False, dense_norm=0) with modest
#  rel_core and is floated purely by the FLAT additive trust term (2.5*0.8=2.0).
#  This is a CLASS: iep.utm.edu, earlychristianwritings.com, aic.lv,
#  ccat.sas.upenn.edu, oec.uzh.ch all top the raw-tf bm25 leg the same way. The
#  206 head/beta relgate attenuates the trust term proportionally but cannot fully
#  clear the high-bm25-rel members (they ride bm25 relevance, not trust). The
#  lever: on a GENERAL-intent query (the head has fired, eff_beta>0) a candidate
#  with NO dense (semantic) support — absent from the dense pool — should not
#  INHERIT the trust floor on a bare lexical coincidence. Withhold (scale, default
#  0.0) the trust term for dense-less candidates on general queries. It never
#  demotes below the candidate's own rel_core (a genuinely-lexical page still ranks
#  on relevance), is inert on authority-lookup queries (eff_beta=0 → statutes/RMF
#  keep the full floor) and on dense-supported results. Flag-gated AW_HUBGATE
#  (default OFF, byte-identical); rollback = AW_HUBGATE=0. Composes with the head:
#  the head attenuates dense-weak, the hub-gate removes the dense-absent bm25 float.
_HUBGATE = _S.trust.hubgate

_HUBGATE_SCALE = 0.0  # fixed: was env AW_HUBGATE_SCALE (S10 D5: no product setter)

#  Only fire on CONFIDENTLY-general queries (eff_beta >= min): the trust-floated
#  BM25-only pattern is shared by legitimate authority/name lookups the head
#  routes weakly-general (entity names, "latest Go release", "git rebase --onto"
#  — dense-less exact-term floats). Gating on eff_beta>0 (soft head ⇒ ~always)
#  decapitated those (names h@10 1.00→0.43). A min-beta protects the low-P tail.
_HUBGATE_MINBETA = 0.0  # fixed: was env AW_HUBGATE_MINBETA (S10 D5: no product setter)

#  Trust ceiling: never strip the floor from top-tier official sources (statutes,
#  gov) even when misrouted general; only blog/mid-tier hubs (antirez 0.8) gate.
_HUBGATE_MAXTRUST = 1.0  # fixed: was env AW_HUBGATE_MAXTRUST (S10 D5: no product setter)

# --------------------------------------------------------------------------- #
#  TRUST-FLOOR (task 289 / ADR-036 §trust-must-be-earned) — the ABSOLUTE-relevance
#  gate on the trust term that the pool-relative levers structurally cannot reach.
#
#  Diagnosis (task 289): on a WEAK query — one where the index holds NO strongly-
#  relevant document ("psychéliques, caprices et détresse mentale") — generic high-
#  trust civic/gov portals (info.gouv.fr/composition-du-gouvernement,
#  ch.ch/fr/votations-et-elections) flood the top on the FLAT additive trust term
#  (w_trust·trust_tier = 2.5·~0.95 ≈ 2.4), which dwarfs the whole pool's weak
#  rel_core. The three existing levers all MISS this regime:
#    * the ADR-036 rankhead relgate attenuates trust only when a candidate's
#      rel_core is a small FRACTION of the POOL-BEST (ratio < eff_beta). But every
#      leg is min-max normalized, so best_rel_core ≈ a per-query constant and the
#      civic page — which can BE the pool-best when nothing better exists — has
#      ratio ≈ 1 → NO attenuation. Pool-relative relevance cannot see "the whole
#      pool is weak."
#    * AW_HUBFLOOR fires only for the 17 enumerated law/stat mega-hubs (df≥25000);
#      info.gouv.fr / ch.ch are not table members → hubness 0 → floor never fires.
#    * AW_HUBGATE keys on dense-ABSENCE (a binary), but these pages ARE dense-present
#      (weakly) — they enter the pool via a low but non-zero cosine.
#
#  The one ABSOLUTE-relevance signal in the pipeline is the RAW dense cosine
#  (best_dense_raw is already computed for the zero-signal df-guard; per-candidate
#  raw["dense"][pid] is the same quantity). This lever attenuates a candidate's
#  trust term by a smooth ramp of its OWN raw dense cosine below TAU:
#      scale = clip(cos / TAU, FLOOR, 1.0)      (cos>=TAU → 1.0, no change)
#  so a page riding trust on a weak absolute semantic match loses the floor while a
#  page that genuinely OWNS the query (high cosine — the right civic portal on an
#  on-remit how-to, or the actually-relevant content here) keeps it. Attenuate-ONLY
#  (never demotes, never touches rel_core/penalties), stage-2, arithmetic → pool and
#  latency unchanged. Composes MULTIPLICATIVELY with the rankhead relgate/hubgate
#  (both scale the same trust_term) — it is the ABSOLUTE complement to their
#  POOL-RELATIVE gate, not a redundant re-lever.
#
#  INTENT-GATED exactly like AW_HUBFLOOR (rh_beta >= MINBETA): on AUTHORITY-lookup
#  intent (rh_beta→0: statutes, specs, entity/name lookups) the floor NEVER fires,
#  so the legitimate low-cosine-high-trust answers task 207 flagged (terse statutes
#  at dense-cos ~0.04, RMF) keep the full trust floor BY CONSTRUCTION — the same
#  tripwire protection the hub-floor relies on. MAXTRUST is an OPTIONAL extra ceiling
#  (default 1.0 = off; the civic targets are trust ~0.95, so the intent gate, not a
#  trust ceiling, is the primary discriminator). Flag-gated AW_TRUSTFLOOR (default
#  OFF, byte-identical when off); rollback = AW_TRUSTFLOOR=0 (or TAU=0.0).
# --------------------------------------------------------------------------- #
_TRUSTFLOOR = _S.trust.trustfloor

# Intent gate (head beta >= min). The head NEVER emits exactly 0 (sigmoid), so this
# is the real authority/general discriminator: authority-lookup queries (statutes,
# specs, names) get a SMALL beta, general queries a larger one. LOWER than
# AW_HUBFLOOR's 0.5 because the reported flood ("psychéliques…") routes only
# WEAKLY-general (beta ~0.30) — the pool-weakness gate below is what makes a low
# MINBETA safe (a strong-pool name/authority lookup is exempted by that gate, not
# by beta). Calibrate against the legal/names rh_beta band before shipping.
_TRUSTFLOOR_MINBETA = _S.trust.trustfloor_minbeta

# POOL-WEAKNESS gate (Tim's hypothesis, task 289): engage ONLY when the pool's BEST
# raw dense cosine is below this — i.e. the index holds NO strongly-relevant doc, so
# the flat trust floor is the de-facto sort key. This is the guard that lets MINBETA
# drop safely: a name/authority/general query with a genuine dense answer has a HIGH
# best-cosine pool => lever inert (no name/exact-float regression, the AW_HUBGATE
# failure mode); only genuinely weak-pool queries engage. ~0.68 aligns with the
# _RELFLOOR_DENSE_MAX "no strong semantic match" band. 1.0 => always engage (off).
_TRUSTFLOOR_POOL_MAX = _S.trust.trustfloor_pool_max

# REL-COMPETITIVENESS exemption — the discriminator that separates a trust-RIDING
# civic page from a legitimately-relevant DENSE-ABSENT answer. A rare-name gold
# ("Tim O'Hear" -> impactia.org) or an exact-title gold ("Universal Declaration of
# Human Rights" -> ohchr.org) has dcos=0 yet is the POOL-BEST rel_core (it wins on
# EXACT/title/rare-token BM25, the AW_HUBGATE names-regression class). The flood
# civic pages instead sit at rel_core ratio ~0.4. So: NEVER strip trust from a
# candidate whose rel_core is >= this fraction of the pool-best rel_core — it has
# EARNED the floor on the FULL relevance signal (exact+title+bm25), not just dense.
# This is what makes the lever safe on names/foundation. Mirrors the official_pref/
# country/entity_card rel-gates. 0.0 => no exemption (dense-only, regresses names).
_TRUSTFLOOR_REL_BETA = _S.trust.trustfloor_rel_beta

# Optional trust ceiling: never attenuate above this trust_tier (set < 1.0 to spare
# statute-tier). Default 1.0 = no ceiling (the intent gate is the discriminator).
_TRUSTFLOOR_MAXTRUST = 1.0  # fixed: was env AW_TRUSTFLOOR_MAXTRUST (S10 D5: no product setter)

# Absolute raw-cosine floor for arm (a). Deliberately BELOW _TRUSTFLOOR_TAU: at
# or above TAU the ramp is 1.0 and the exemption is moot, so a floor at TAU
# would make this arm dead code. Calibrated in task 461 against the measured
# rank-1 cosine distribution (flood candidates 0.000; genuine dense-supported
# rank-1s 0.50-0.81).
_TRUSTFLOOR_ABS_DENSE = 0.45  # fixed: was env AW_TRUSTFLOOR_ABS_DENSE (S10 D5: no product setter)

_NAMEGATE_TRUSTKEEP = _S.fusion.namegate_trustkeep

# --------------------------------------------------------------------------- #
#  TASK 411 — the COVERAGE SURFACE, and its generalisation to required terms.
#  Every flag below defaults to today's shipped behaviour (surface "display",
#  weight 0.0) => byte-identical scoring; rollback is the env, not a redeploy.
#
#  (a) THE COVERAGE-SURFACE DEFECT. `_name_token_coverage` reads title+snippet
#  ("the only text we hold"). For ADR-060 entity cards that is FALSE and the
#  divergence is deliberate: a card's DENSE channel is prose and its SPARSE
#  channel is name + aliases + domain + facet terms (407 §1.2 P8), and
#  `compose_index_text` is explicitly EXEMPT for source_type == entity_card. For
#  the register-people facet it is also POLICY: the published privacy notice
#  (task 390 §4.9.1) commits that a register-published person name is carried as
#  a SPARSE MATCHING TOKEN AND NOTHING ELSE — not in the card's text, no person
#  record. Measured on the live card (task 390's own prod receipt): `marc` and
#  `maret` are both present in the Vector8 AG sparse vector; the literal string
#  "Maret" appears nowhere in the payload.
#
#  So on `Marc Maret` the ONE record built to answer the query scores coverage
#  0.0, the pool condition ">= 1 candidate carries the full name" is never met,
#  `_name_intent` stays False, and BOTH shipped halves stand down — the −2.5
#  namegate on the half-match flood and the +0.5 bonus on the answer. The right
#  answer is indexed and invisible to the lever.
#
#  The cure MUST be read-time. Writing the name into title/snippet would fix
#  coverage by breaking the privacy commitment; that is a POLICY change and is
#  not proposed here. AW_NAMECOV_SURFACE selects the surface:
#    display (default) title+snippet — what ships today
#    sparse            the candidate's STORED BM25 vector, via the task-083
#                      id-filtered presence probe (the name token's PRIMARY
#                      hash only — never a de-compounded fragment)
#    union             max(display, sparse) — strictly more permissive
#    card_sparse       display, EXCEPT for records whose display text is not
#                      their indexing surface — see below
#
#  WHY `card_sparse` EXISTS, measured rather than assumed. The blanket `sparse`
#  surface fixes `Marc Maret` (None -> 1) but REGRESSES `Tim O'Hear` lang=fr
#  (1 -> 2) and `Laura Tocmacov` (2 -> 6) — measured, task 411/
#  diag411.json. Mechanism, named: for an ORDINARY content page the sparse
#  channel is the whole chunk body, so every impactia.org page that mentions the
#  person anywhere reaches coverage 1.0, takes the H2 bonus and keeps its trust
#  floor — the site's other pages then outrank the one page the query is about.
#  The display text is a genuinely tighter signal there and must be kept.
#
#  The two cases differ by a documented PRODUCER fork, not by host or query:
#  `compose_index_text` is exempt for `source_type == "entity_card"`, and a
#  card's sparse channel is `sparse_hooks(...)` (name/aliases/facets) while its
#  display text is the entity name + prose snippet (407 §1.2 P8). For that class
#  ALONE the display text is provably not the indexing surface. `card_sparse`
#  is therefore a class rule over the taxonomy the payload already carries —
#  the same shape as `_is_location_bound` or the ADR-048/060 exemption itself —
#  and not a special case for a domain (displacement-framework rule).
#
#  (b/c) REQUIRED TERMS. AW_REQTERM_W adds a bounded rel_core bonus to
#  candidates that match the query's DISTINCTIVE source words on the sparse leg.
#  Three things are deliberate:
#    * the unit is the SOURCE WORD (bm25_token_groups), not the emitted token —
#      otherwise `O'Hear`'s junk `hear` token and `Arbeitsvertrag`'s three
#      de-compounded pieces make "all terms match" unsatisfiable or trivial;
#    * only DISTINCTIVE words are required. "All query terms" was already
#      MEASURED AND LOST (task 083, the AW_BM25_PER_TOKEN block comment: it
#      "boosted verbose pages that merely contain every function word of a long
#      question" — 29q MRR 0.400->0.366, 117q hit@1 39->32). Distinctiveness is
#      the deployed df test: a single-token window that comes back NOT FULL
#      proves df < AW_BM25_PER_TOKEN_LIMIT. AW_REQTERM_DFMAX re-measures that
#      classification at a WIDER window when the shipped 50 is too strict (it
#      classifies `maret`, `ohear`, `turing`, `leyen` as "common" — measured,
#      task 411 df-census.json);
#    * it is a SOFT bonus, never a filter. Requiring every term would kill
#      recall on verbose queries; the arm `AW_REQTERM_MODE=hard` exists only to
#      measure that, not to ship.
#  Pool condition, mirroring the 404/406 safety: the bonus arms only when >= 1
#  in-pool candidate matches ALL required words. A mis-selected requirement can
#  then only fail to fire, never demote.
# --------------------------------------------------------------------------- #
_NAMECOV_SURFACES = ("display", "sparse", "union", "card_sparse")

_NAMECOV_SURFACE = _S.fusion.namecov_surface.strip().lower()

if _NAMECOV_SURFACE not in _NAMECOV_SURFACES:
    # fail loud: a typo'd surface must not silently degrade to today's behaviour
    raise ValueError(f"CYTIR_NAMECOV_SURFACE must be one of {_NAMECOV_SURFACES}, "
                     f"got {_NAMECOV_SURFACE!r}")


def score_rows(*, _k_dense_name, _name_intent, _ng_cov, _primacy_pids, _reqterm, _reqterm_groups,
        _tf_abs_armed, _tf_abs_hit, best_dense_raw, best_rel_core, best_rel_undemoted, commercial, country,
        country_leg_provenance, experimental, gate_thr, global_raw, k_country_boost, k_trust_scale, norm,
        query_plan, raw, rel_rank, rh_beta, rows):
    """Stage 2d: fused score per row, ordered (S10 step 8b)."""
    scored = []
    relmeta = {}   # task 207: advisory confidence per candidate (AW_RELFLOOR)
    n_rows = max(len(rows), 1)
    for pid, pl, rel_core, penalties, trust_tier in rows:
        if _TRUST_MODE == "multiplicative":
            # trust SCALES relevance; penalties stay additive so a machine/data
            # endpoint on a trust-0.95 domain is still demoted, not amplified.
            fused = rel_core * (1.0 + _TRUST_ALPHA * trust_tier) + penalties
        elif _TRUST_MODE == "gated":
            trust_term = WEIGHTS["trust"] * trust_tier if rel_core >= gate_thr else 0.0
            fused = rel_core + penalties + trust_term
        elif _TRUST_MODE == "rank_scaled":
            rw = max(0.0, 1.0 - rel_rank.get(pid, n_rows) / n_rows)
            fused = rel_core + penalties + WEIGHTS["trust"] * trust_tier * rw
        else:  # additive (legacy default)
            # ADR-062 knob 3b (trust_scale): head-driven whole-trust multiplier.
            # k_trust_scale == rh_trust_scale (ADR-036 constant 1.0) at cold-start.
            trust_term = WEIGHTS["trust"] * k_trust_scale * trust_tier
            # RANKING HEAD (task 205) takes precedence: a PER-QUERY relgate beta
            # from the intent head (rh_beta>0 only on general-intent queries; 0 on
            # authority-lookup → statutes keep the full floor). Same attenuate-only
            # ramp as CHANGE B, just per-query. When the head is off (rh_beta==0)
            # the global CHANGE B flag still applies (unchanged).
            eff_beta = rh_beta if rh_beta > 0 else (
                _TRUST_RELGATE_BETA if _TRUST_RELGATE else 0.0)
            if eff_beta > 0 and best_rel_core > 0:
                ratio = rel_core / best_rel_core
                if ratio < eff_beta:
                    _ramp = max(0.0, ratio) / eff_beta
                    # TASK 578 — a relevance-competitive, title-central PRIMARY
                    # source keeps the trust the register already gave it.
                    if pid in _primacy_pids:
                        _ramp = _primacy_refund(_ramp)
                    trust_term *= _ramp
            # HUB-GATE (task 210): on a general-intent query, a candidate with no
            # dense (semantic) support is a pure-lexical BM25 float (the antirez
            # class) — withhold its trust floor. Composes after the ramp above.
            if (_HUBGATE and eff_beta >= _HUBGATE_MINBETA and eff_beta > 0
                    and trust_tier <= _HUBGATE_MAXTRUST
                    and pid not in norm.get("dense", {})):
                trust_term *= _HUBGATE_SCALE
            # TRUST-FLOOR (task 289): the ABSOLUTE-relevance complement to the
            # pool-relative relgate above. On a general-intent query (eff_beta >=
            # MINBETA — statutes/specs/names have rh_beta→0 => exempt) attenuate the
            # trust term by the candidate's OWN raw dense cosine below TAU, so a
            # generic high-trust civic/gov portal riding trust on a weak absolute
            # semantic match (info.gouv.fr/ch.ch on "psychéliques…") loses the floor
            # while a page that genuinely owns the query keeps it. Uses the raw
            # (pre-minmax) cosine — the ONLY absolute-relevance signal; best_rel_core
            # is min-max normalized and near-constant per query, which is exactly why
            # the relgate misses the weak-pool regime. Attenuate-only, composes
            # multiplicatively. Inert (flag off / TAU 0) => byte-identical.
            # task 406 H3: (i) AW_NAMEGATE_TRUSTKEEP — a FULL-COVERAGE candidate
            # under person-name intent is EXEMPT (the page proven to name the
            # person must not be trust-stripped for lacking a dense hit — dense
            # is precisely the signal this query class cannot rely on); (ii)
            # AW_TRUSTFLOOR_UNDEMOTED — the ratio exemption reads
            # `best_rel_undemoted` (== best_rel_core when the flag is off).
            _tf_namekeep = (_NAMEGATE_TRUSTKEEP and _name_intent
                            and _ng_cov.get(pid, 0.0) >= _NAMEGATE_COV)
            # ADR-099 Ruling 3: the exemption is now pool-relative AND absolute
            # (`_trustfloor_exempt`); with AW_TRUSTFLOOR_ABS off it is the
            # pre-099 condition exactly.
            _tf_dense_raw = raw.get("dense", {}).get(pid, 0.0)
            if (_TRUSTFLOOR and eff_beta >= _TRUSTFLOOR_MINBETA and eff_beta > 0
                    and best_dense_raw < _TRUSTFLOOR_POOL_MAX
                    and trust_tier <= _TRUSTFLOOR_MAXTRUST
                    and not _tf_namekeep
                    and not _trustfloor_exempt(
                        rel_core, best_rel_undemoted, _TRUSTFLOOR_REL_BETA,
                        abs_on=_tf_abs_armed, dense_raw=_tf_dense_raw,
                        abs_dense_floor=_TRUSTFLOOR_ABS_DENSE,
                        abs_term_hit=pid in _tf_abs_hit)):
                _tf_scale = _trustfloor_scale(_tf_dense_raw)
                # TASK 578 — same refund on the ABSOLUTE-relevance half of the
                # attenuation. The HUB-GATE above is deliberately NOT refunded:
                # a candidate with no dense support at all is the pure-lexical
                # float task 210 was written for, primary class or not.
                if pid in _primacy_pids:
                    _tf_scale = _primacy_refund(_tf_scale)
                trust_term *= _tf_scale
            # task 254: attenuate the gov/regulator trust floor on a commercial
            # query (finma/seco/kmu.admin.ch/europa.eu no longer dominate a
            # "personal loan" lookup). Multiplies AFTER the head relgate/hubgate
            # so it composes; inert (scale 1.0) unless AW_COMMERCIAL fired.
            if commercial and pl.get("source_type") in _GOV_SOURCE_TYPES:
                trust_term *= _COMMERCIAL_GOV_SCALE
            if _DEBUG_SCORES:
                # task 583 — the trust term is a PRODUCT, not a sum: the
                # register's prior times a chain of attenuations. Record the
                # prior and the realised term; the difference is the whole of
                # what the relgate / hub-gate / trustfloor / commercial scale
                # took away, which is what a "is it trust?" attribution needs.
                _dbgc(pid, "trust.possible",
                      WEIGHTS["trust"] * k_trust_scale * trust_tier)
                _dbgc(pid, "trust.realised", trust_term)
                _dbgc(pid, "meta.eff_beta", eff_beta)   # a state, not a term
            fused = rel_core + penalties + trust_term
        # Country soft-boost: only when a jurisdiction is selected AND this
        # candidate is stamped authoritative for it. No country => term never
        # added => identical to today. task 164: RELEVANCE-GATED — a domestic
        # candidate must be competitive with the pool's best relevance
        # (rel_core >= beta * best_rel_core) to earn the boost, so a weak
        # domestic page can't leapfrog a strongly-relevant foreign/EU authority.
        if country is not None and _effective_country(pl) == country:
            if (not _COUNTRY_GATE) or rel_core >= _COUNTRY_GATE_BETA * best_rel_core:
                fused += k_country_boost   # ADR-062 knob 14 (== _COUNTRY_BOOST cold-start)
                if _DEBUG_SCORES:
                    # `bonus.` and not `pen.`: this one lands on `fused`, not on
                    # the penalties bucket, so the reconciliation identity is
                    # rel_core = SUM(rel.*), penalties = SUM(pen.*),
                    # trust_contrib = trust.realised + SUM(bonus.*).
                    _dbgc(pid, "bonus.country", k_country_boost)
        # task 207: advisory confidence marker — pool-relative relevance +
        # trust-share of THIS result. Non-destructive: computed only, attached
        # to the output item below when AW_RELFLOOR=1; no effect on `fused`.
        if _RELFLOOR:
            rr = (rel_core / best_rel_core) if best_rel_core > 0 else 0.0
            relmeta[id(pl)] = {
                "rel_ratio": round(float(rr), 3),
                "trust_frac": round(float((fused - rel_core - penalties) / fused), 3)
                              if fused else 0.0,
                "low_relevance": bool(rr < _RELFLOOR_BETA),
                "general_intent": bool(rh_beta > 0),
            }
        if _DEBUG_SCORES:
            _LAST_SCORE_DEBUG.append({
                # task 350: pid included so the rankhead_v2 miner can join the
                # feature dump to the fused score EXACTLY (url alone collides on
                # multi-chunk pages). Additive key — dbg consumers name fields.
                "pid": str(pid),
                "url": pl.get("url"), "domain": pl.get("domain"),
                "source_type": pl.get("source_type"),
                "trust_tier": _trust_tier(pl),
                "dense_raw": round(float(global_raw["dense"].get(pid, 0.0)), 4),
                "task381_sparse_scores": {
                    leg: round(float(raw.get(leg, {}).get(pid, 0.0)), 4)
                    for leg in ("bm25", "target_bm25", "translated_bm25")},
                "country_leg_scores": {
                    leg: round(float(raw[leg][pid]), 4)
                    for leg in country_leg_provenance.get(pid, ())},
                "country_plan": {
                    "country": query_plan.country,
                    "confidence": round(query_plan.confidence, 4),
                    "source": query_plan.source,
                },
                "rel_core": round(float(rel_core), 4),
                "penalties": round(float(penalties), 4),
                "trust_contrib": round(float(fused - rel_core - penalties), 4),
                "fused": round(float(fused), 4),
                "rel_ratio": round(float(rel_core / best_rel_core), 3) if best_rel_core > 0 else 0.0,
                # task 406: the person-name lever state, so per-half attribution
                # reads off the artifact instead of off inference.
                "name_intent": bool(_name_intent),
                "name_cov": round(float(_ng_cov.get(pid, 0.0)), 3),
                "name_dense_scale": round(float(_k_dense_name), 4),
                # task 411: the coverage surface actually used + the
                # required-term state, so per-mechanism attribution reads off
                # the artifact instead of off inference.
                "namecov_surface": _NAMECOV_SURFACE,
                "reqterm_words": [w for w, _hs in _reqterm_groups],
                "reqterm_bonus": round(float(_reqterm.get(pid, 0.0)), 4),
                "rel_ratio_undemoted": (round(float(rel_core / best_rel_undemoted), 3)
                                        if best_rel_undemoted > 0 else 0.0),
                # task 583: the named terms behind `rel_core` / `penalties` /
                # `trust_contrib`. Keys present only for levers that actually
                # fired on THIS row, so an absent key means "contributed 0",
                # never "not measured".
                "components": {kk: round(float(vv), 6) for kk, vv
                               in _LAST_COMPONENTS.get(pid, {}).items()},
                # the raw BM25 leg score and its pool rank, so a BM25-LEG
                # attribution reads the leg, not the fused bucket.
                "bm25_raw": round(float(raw.get("bm25", {}).get(pid, 0.0)), 4),
                "dense_norm": round(float(norm.get("dense", {}).get(pid, 0.0)), 4),
                "bm25_norm": round(float(norm.get("bm25", {}).get(pid, 0.0)), 4),
            })
        scored.append((fused, pid, pl))
    _order_scored(scored)
    if experimental:
        _LAST_EXPERIMENT_TRACE["stage2_scored"] = [
            {"id": str(pid), "url": pl.get("url"), "score": float(score)}
            for score, pid, pl in scored]
    return relmeta, scored
