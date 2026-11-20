"""Hybrid retrieval + fusion re-ranking (spec §5.2, ADR-012).

Three legs (BGE-M3 dense + learned-sparse + in-engine BM25) are queried
SEPARATELY so we have each leg's score per candidate, then re-ranked by a
weighted fusion of the normalized leg scores + signals (source trust, recency).
No passage text is used — copyright/size-safe. Weights default to sensible
values and can be overridden by `fusion_weights.json` (learned offline by
train_fusion.py from the eval labels).
"""
from __future__ import annotations

import functools

from . import fusion, levers, models, plan, qdrant_io, respond
from . import settings as _settings
from .settings import SETTINGS, _DEFAULT_WEIGHTS, _load_weights, _weights_path  # noqa: F401 (facade)
from .schema import (COLLECTION, DENSE_VEC, SPARSE_LEXICAL, SPARSE_BM25,
                     bm25_tokens, bm25_token_groups, _decompound)
# S10 slice 1 (task 752): pure helpers live in their own modules and are imported back under
# their old names, so `retrieval.<name>` keeps working for every caller.
from .query import (_AC_PAT, _AC_PHRASES, _AC_WORDS, _ID_LEI_RE,  # noqa: F401
                    _ID_SCHEME_SUFFIX, _ID_SIREN_RE, _ID_SIRET_RE, _ID_UID_RE, _ID_VAT_RE,
                    _detect_identifier, _query_academic_intent)
from .query import _COMM_LEGAL_GUARD, _COMM_PHRASES, _COMM_SUBSTR, _COMM_TOKENS  # noqa: F401
from .query import _query_commercial_intent  # noqa: F401
# S10 step 1 (task 1061): lexical matching shared by the query and the candidate side.
from .textmatch import (_ALPHA_RE, _STOPWORDS, _WORD, _card_aliases,  # noqa: F401
                        _card_name_match, _content_terms, _exact_v, _query_terms,
                        _stem_match, _title_centrality, _title_centrality_v,
                        _title_versions_match)
# S10 step 3 (task 1083): the person-name / required-word query side and the namegate/refcanon levers.
from .query import (_NAMEGATE_MAXTERMS, _NAMEGATE_SHAPE_V2, _NOBILIARY,  # noqa: F401
                    _card_alias_in_query, _name_token_primaries, _person_name_query,
                    _person_name_tokens, _required_word_groups)
from .levers import (_NAMEGATE_COV, _NAMEGATE_W, _REFCANON, _REFCANON_BETA,  # noqa: F401
                     _REFCANON_EXTRA, _REFCANON_MAXBETA, _REFCANON_MINCENT, _REFCANON_TARGETS,
                     _REFCANON_W, _apply_namegate, _apply_refcanon, _name_token_coverage,
                     _refcanon_canonical, _sparse_name_coverage)
from .signals import (_ACADEMIC_DOMAINS, _LF_BULK_RE, _LF_HOST_RE,  # noqa: F401
                      _MACHINE_URL_RE, _academic_penalty, _domain, _exact, _lf_bulk,
                      _machine_url, _minmax, _recency, _registrable_domain, _trust_tier)

# S10 step 4 (task 1095): jurisdiction/country scoping and the curated tables; the location-bound
# test is `cytir.core.country.is_location_bound` (D7).
from cytir.core.country import is_location_bound  # noqa: F401  (side-effect import, task 1808)
from .geo import (_CANTON_NAMES, _COUNTRY_MARKERS, _SCOPE_COUNTRIES, _EXEMPT_STAMPS,
    _OFF_COUNTRY_ANY, _demotable_scope, _EU_MEMBERS, _is_eu_supra, _terms_re, _CANTON_RES,
    _COUNTRY_RES, _query_jurisdiction, _GAZ_PATH, _load_gazetteer, _GAZETTEER, _gazetteer_scope,
    _scope_penalties, _OFF_COUNTRY_LOCBOUND_ONLY, _country_is_binding, _off_country_hit,
    _COUNTRY_MAP_PATH, _COUNTRY_MAP, _effective_country, _OFF_COUNTRY_NONBOUND_W_RAW,
    _OFF_COUNTRY_NONBOUND_W, _OFF_COUNTRY_NONBOUND_CLASSES, _nonbound_class,
    _learned_scope_conflict_penalty)  # noqa: F401
from .tables import (_load_parasites, _PARASITE_HOSTS, _PARASITE_RES, _parasite,
    weights_fingerprint, _HUBFLOOR, _HUBFLOOR_TABLE, _load_hub_hosts, _HUB_HOSTS, _hubness,
    _AUTHORITY, _AUTHORITY_TABLE, _load_authority_hosts, _AUTH_HOSTS, _authority)  # noqa: F401

from .heads import (_QHEAD_COUNTRY_LEGS, _RANKHEAD, _RANKHEAD_PATH, _RANKHEAD_V2, _RANKHEAD_V2_PATH,
    _RANKHEAD_V2_DUMP, _LAST_V2_DUMP, _load_rankhead, _RANKHEAD_MODEL, _load_rankhead_v2,
    _RANKHEAD_V2_MODEL, _head_beta, _QHEAD, _QHEAD_PATH, _QHEAD_KNOB_ALIASES, _QHEAD_KNOBS,
    _SCOPE_ACTIVE, _parse_country_scope_mode, _COUNTRY_SCOPE_MODE, _LEARNED_ONLY_SCOPE,
    _POLICY_ONLY_SCOPE, _QHEAD_DRIVEN, _load_qhead, _QHEAD_MODEL, _qhead_vector, _SCOPE_KNOB,
    _load_scope_head, _SCOPE_HEAD, _load_query_policy_heads, _QUERY_POLICY_HEADS,
    _query_policy_vector, _scope_head_predict, _resolve_scope, _effective_free_text_country_scope,
    _country_scope_for_ranking, _head_artifacts)  # noqa: F401 (facade)

# S10 step 6 (task 1104): the Qdrant I/O seam, the query plan, result shaping and the component
# ledger. `search()` reaches the client as `qdrant_io.client()` (a module attribute read at call time),
# so a test patches `qdrant_io.client`; the names below are the facade for readers only.
from .plan import QueryPlan, _country_partition_filter, _country_query_plan, candidate_pool, fuse_margin  # noqa: F401
from .qdrant_io import _identifier_lookup, _leg, client, health_client, indexed_fields  # noqa: F401
from .respond import _canonical_key, _is_tracking_param  # noqa: F401
from .trace import _LAST_COMPONENTS, _dbgc  # noqa: F401
from .fusion import (_PRIMACY_EXTRA,  # noqa: F401
                     _PRIMACY_KEEP, _TRUSTFLOOR_FLOOR, _TRUSTFLOOR_TAU, _auth_admit,
                     _extend_pool, _is_primary_class, _order_ids, _order_scored, _primacy_refund,
                     _trustfloor_exempt, _trustfloor_scale, _url_tie_len)
from .levers import _card_key, _card_winners  # noqa: F401
from .legs import (Stage1, Stage1Request, host_query_label as _host_label, stage1)  # noqa: F401
from .legs import (_AUTHLEG, _AUTHLEG_BETA, _AUTHLEG_N, _AUTHLEG_THRESH, _AUTH_LEG_KEY, _BATCH_LEGS, _BM25_ZERO_GUARD, _CARD_LANG_EXEMPT, _DENSE_HNSW_EF, _DENSE_IGNORE_QUANT, _DENSE_OVERSAMPLING, _DENSE_RESCORE, _DENSE_SP, _PAGECAP, _PAGECAP_MODE, _PAGECAP_OVERSAMPLE, _PER_TOKEN, _PER_TOKEN_K1, _PER_TOKEN_LIMIT, _PER_TOKEN_MAXQ, _PER_TOKEN_SUB_W, _RELFLOOR, _RELFLOOR_DENSE_MAX, _SPARSE_SCORE_LEGS, _TWO_STAGE)  # noqa: F401
from .fusion import (_COUNTRY_BOOST, _COUNTRY_GATE, _COUNTRY_GATE_BETA, _FUSION_BM25_W, _FUSION_DENSE_W, _W_BM25_EFF, _W_DENSE_EFF)  # noqa: F401
from .trace import (_LAST_EXPERIMENT_TRACE, _LAST_QUERY_PLAN_TRACE, _QHEAD_COUNTRY_TRACE)  # noqa: F401

_S = SETTINGS   # typed config, loaded once at import (S10 step 2, task 1071)





# --------------------------------------------------------------------------- #
#  Identifier fast-path (task 170). A bare UID/register-id/LEI/VAT query does
#  NOT resolve its entity card on the normal path: BM25 fragments
#  "CHE-115.140.892" into `che` + numbers that collide with protein names and
#  statute article numbers, so the card (whose register_id IS in the payload +
#  `names`) never surfaces. We detect an identifier-DOMINANT query (the id is
#  essentially the whole query; a trailing scheme word MWST/UID/VAT is allowed)
#  and do an EXACT payload-filter lookup on `register_id`/`lei`, returning the
#  card as #1. Everything else — prose containing a number, "Art. 335c" — does
#  not match and falls through to normal search. Skips the embedder entirely:
#  ~1 ms and embedder-independent. Requires a keyword index on register_id/lei
#  (task 170 §B.2, a prod op); the lookup fails safe to normal search on error.
# --------------------------------------------------------------------------- #
_ID_FASTPATH = True  # fixed: was env AW_ID_FASTPATH (S10 D5: no product setter)


WEIGHTS = _S.weights   # the file `settings` loaded (one read; precedence unchanged)










# task 254: commercial/product-intent down-weighting (see _query_commercial_intent).
# Flag-gated (default OFF => byte-identical). On a commercial query:
#   commercial_ref       additive penalty (<= 0) on source_type=reference (Wikipedia)
#   commercial_gov_scale multiplies the gov/regulator trust term (in [0,1]; 1.0 = inert)
# Env overrides stage the values without editing fusion_weights.json.
_COMMERCIAL = _S.query.commercial





# Fire only on CONFIDENTLY-general queries (head beta >= min): mirrors
# AW_HUBGATE_MINBETA. rh_beta==0 on authority lookups => floor never fires there.
_HUBFLOOR_MINBETA = _S.trust.hubfloor_minbeta

from .fusion import (_COMMERCIAL_GOV_SCALE, _GOV_SOURCE_TYPES, _HUBGATE, _HUBGATE_MAXTRUST,
    _HUBGATE_MINBETA, _HUBGATE_SCALE, _NAMECOV_SURFACE, _NAMECOV_SURFACES, _NAMEGATE_TRUSTKEEP,
    _RELFLOOR_BETA, _TRUSTFLOOR, _TRUSTFLOOR_ABS_DENSE, _TRUSTFLOOR_MAXTRUST, _TRUSTFLOOR_MINBETA,
    _TRUSTFLOOR_POOL_MAX, _TRUSTFLOOR_REL_BETA, _TRUST_ALPHA, _TRUST_MODE, _TRUST_RELGATE,
    _TRUST_RELGATE_BETA)  # noqa: F401
from .trace import (_DEBUG_SCORES, _LAST_SCORE_DEBUG)  # noqa: F401
from .respond import (_RELFLOOR_ZS_K)  # noqa: F401
from .levers import (_AUTHORITY_MAXBOOST, _AUTHORITY_MIN, _AUTHORITY_MIN_CENTRALITY,
    _AUTHORITY_RELGATE, _AUTHORITY_W, _CANTON_UNCONDITIONAL, _COMMERCIAL_REF_W, _ENTITY_CARD_BETA,
    _ENTITY_CARD_CAP, _ENTITY_CARD_NAME_THR, _HUBFLOOR_DENSE_MAX, _HUBFLOOR_DF_MIN,
    _HUBFLOOR_DISTINCT_MAX, _HUBFLOOR_MAXTRUST, _HUBFLOOR_W, _LFBULK_RELGATE,
    _NAMECOV_SPARSE_CLASSES, _NAMECOV_W, _NAMEDENSE, _NAMEDENSE_SCALE, _NAMEGATE,
    _NAME_LEVERS_ARMED, _OFFICIAL_PREF_BETA, _OFF_COUNTRY_W, _PRIMACY, _PRIMACY_BETA,
    _PRIMACY_MINCENT, _REQTERM_DFMAX, _REQTERM_MAXQ, _REQTERM_MINGROUPS, _REQTERM_MODE, _REQTERM_W,
    _TRUSTFLOOR_ABS, _TRUSTFLOOR_ABS_MAXTOK, _TRUSTFLOOR_UNDEMOTED, _TRUST_GATE_PCTL,
    _VERSION_MATCH, _VERSION_MATCH_W)  # noqa: F401


def search(q: str, k: int = 8, lang: str | None = None,
           candidates: int | None = None, country: str | None = None,
           *, target_bm25_language: str | None = None,
           target_bm25_quota: int = 0,
           target_bm25_weight: float = 0.0,
           translated_bm25_query: str | None = None,
           translated_bm25_quota: int = 0,
           translated_bm25_weight: float = 0.0,
           trace_experiment: bool = False,
           op_filter=None) -> list[dict]:
    """Search, optionally adding bounded sparse-only experimental legs.

    These keyword-only parameters are intentionally inert in normal serving.
    They let Task381 measure B2 (same original terms scoped only on an added
    target-language BM25 leg) and C (a separately scored translated BM25 leg)
    while retaining global dense and original BM25 exactly.  No parameter can
    replace the original sparse leg or affect filtering of dense retrieval.
    """
    experimental = bool(trace_experiment or target_bm25_language or translated_bm25_query)
    if (target_bm25_quota < 0 or translated_bm25_quota < 0
            or target_bm25_weight < 0 or translated_bm25_weight < 0):
        raise ValueError("Task381 sparse experiment quotas/weight must be non-negative")
    if translated_bm25_query and (not translated_bm25_quota or not translated_bm25_weight):
        raise ValueError("translated BM25 needs a positive bounded quota and fusion weight")
    if (target_bm25_language and not translated_bm25_query
            and (not target_bm25_quota or not target_bm25_weight)):
        raise ValueError("target-language BM25 needs positive bounded quota and fusion weight")
    if experimental:
        _LAST_EXPERIMENT_TRACE.clear()
    if _DEBUG_SCORES:
        _LAST_SCORE_DEBUG.clear()
        _LAST_COMPONENTS.clear()
    if _QHEAD_COUNTRY_TRACE:
        _LAST_QUERY_PLAN_TRACE.clear()
    # Identifier fast-path (task 170): a bare UID/register-id/LEI/VAT query is
    # resolved by an EXACT payload-filter lookup (no embedding), returning the
    # entity card as #1. Non-identifier queries fall straight through.
    if _ID_FASTPATH:
        ident = _detect_identifier(q)
        if ident:
            hits = (_identifier_lookup(ident[0], ident[1], k) if op_filter is None
                    else _identifier_lookup(ident[0], ident[1], k, op_filter))
            if hits:
                return hits
    # Per-leg candidate pool. FLOORED at 100 (task 199 k-stability — see the
    # _KSTAB block): every k <= 50 fetches the same leg pools, so small-k
    # results are a strict prefix of the k=50 ordering. Still scales with k
    # for large-k calls (agents may ask up to 100 results, which need enough
    # distinct URLs after per-URL dedup); an explicit argument overrides.
    if candidates is None:
        candidates = candidate_pool(k)
    if _PAGECAP and _PAGECAP_OVERSAMPLE > 1.0:
        # task 582 -- widen the LEG limit so the cap frees slots for distinct
        # pages instead of merely shortening the pool. Scaling `candidates`
        # here (not inside candidate_pool) keeps the k-stability formula in its
        # one home; the widened value is echoed by effective_config().
        candidates = int(round(candidates * _PAGECAP_OVERSAMPLE))
    # Normalize the selected country. Empty / "none" / unset => no boost and
    # the exact payload-free fast path (byte-identical to no-country). INTL is
    # the never-demote class, not a selectable jurisdiction.
    country = (country or "").strip().upper() or None
    if country in ("NONE", "ANY", "ALL", "INTL"):
        country = None

    # Jurisdiction scoping (task 123). An explicit country arg is a stronger
    # signal than text detection and wins. `scoped` False => the whole feature
    # is inert (no selector fetch, no penalty terms) — byte-identical scoring.
    q_cantons, q_country_kw = _query_jurisdiction(q)
    # Canton recognition remains independent.  The authoritative country scope
    # is resolved after the shared embedding, at the single caller/free-text
    # boundary below; a policy-head country never enters this variable.
    q_country = None
    # ADR-062 knob 13 (country_scope): scope_confidence scales the off-country
    # penalty magnitude (knob 12) — 1.0 for deterministic (keyword/gazetteer/arg)
    # scope, the learned head's calibrated P for a head-inferred scope. An explicit
    # arg replaces the head's competing output entirely (spec §3 precedence).
    scope_conf = 1.0
    q_country_soft = None   # LEARNED head-inferred scope — feeds ONLY the
                            # confidence-scaled off-country penalty (knob 12), never
                            # the hard scoped levers (official_pref/foreign_gov).
    w_canton = WEIGHTS.get("canton_scope", 0.0)
    w_foreign = WEIGHTS.get("foreign_gov", 0.0)

    dense, lexical = models.embed_query(q)
    # RANKING HEAD (task 205): read the full 1024-dim query vector the embedder
    # just captured (same embed, no extra call) and compute the per-query relgate
    # beta. 0.0 => head inert (== current prod). Head off => stays 0.0.
    rh_beta = 0.0
    rh_trust_scale = 1.0
    if _RANKHEAD and _RANKHEAD_MODEL is not None:
        rh_beta = _head_beta(models.last_full_query_vec())
        rh_trust_scale = _RANKHEAD_MODEL.get("_trust_scale", 1.0)
    # UNIFIED QUERY-UNDERSTANDING HEAD (ADR-062): per-query DRIVING values for the
    # head-driven knobs off the SAME frozen 1024-d vector (no extra embed). Empty
    # when the head is off / unavailable / cold-start (alpha=0) — every knob then
    # resolves to its static default via qk() below, byte-identical to today.
    qh = _qhead_vector(models.last_full_query_vec()) if _QHEAD else {}
    # Candidate-only query-policy artifact.  It is intentionally separate from
    # scoring knobs: an absent/malformed action block returns {}, which leaves
    # the retrieval graph at global dense + BM25 even when its flag is enabled.
    policy_outputs = _query_policy_vector(models.last_full_query_vec())

    # Legacy/learned-only modes resolve their country scope here after the one
    # shared embedding.  Policy-only instead returns no inferred ranking scope;
    # its typed policy output stays separate and can only route candidate legs.
    q_country, q_country_soft, scope_conf = _country_scope_for_ranking(
        country, q, q_country_kw, models.last_full_query_vec())
    # A deterministic legacy/caller scope activates the full country machinery.
    # A learned soft scope does not; it owns only the conflict signal below.
    scoped = bool(q_cantons or q_country) and bool(w_canton or w_foreign)

    # The common query policy converts typed qhead outputs into bounded retrieval
    # routing.  Its complete action block is required: old ranking-only qhead
    # artifacts are inert here by construction.  An explicit agent country wins
    # over the head's country label, but still benefits from its learned routing
    # budgets when the candidate-only artifact is present.
    query_plan = _country_query_plan(
        country, policy_outputs.get("country"),
        policy_outputs.get("country_confidence", 0.0), candidates,
        enabled=bool(policy_outputs),
        jurisdiction_binding=policy_outputs.get("jurisdiction_binding", 0.0),
        foreign_authority_needed=policy_outputs.get("foreign_authority_needed", 1.0),
        multi_scope=policy_outputs.get("multi_scope", 1.0),
        jurisdiction_threshold=policy_outputs.get(
            "jurisdiction_binding_threshold", 0.5),
        foreign_authority_threshold=policy_outputs.get(
            "foreign_authority_needed_threshold", 0.5),
        multi_scope_threshold=policy_outputs.get("multi_scope_threshold", 0.5),
        dense_quota=policy_outputs.get("country_dense_quota"),
        bm25_quota=policy_outputs.get("country_bm25_quota"),
    )

    qk = functools.partial(plan.qk, qh=qh)   # S10 step 8b: the closure's one free name is now a parameter

    # Effective per-query knob values. Each default is EXACTLY today's static read,
    # so cold-start (qh empty) is byte-identical. Leg-weight knobs are SCALES
    # (spec §1 knobs 1/2, ×[0.5,1.5]); the rest are the weight/magnitude itself.
    k_dense_scale = qk("dense_scale", 1.0)
    k_bm25_scale = qk("bm25_scale", 1.0)
    k_trust_scale = qk("trust_scale", rh_trust_scale)
    k_recency = qk("recency", WEIGHTS["recency"])
    k_title = qk("title", WEIGHTS.get("title", 0.0))
    k_exact = qk("exact", WEIGHTS.get("exact", 0.0))
    k_machine = qk("machine", WEIGHTS.get("machine", 0.0))
    # ADR-062 knob 12 × knob 13 fold: the off-country penalty magnitude is scaled
    # by the scope confidence, so a low-confidence (head-inferred) scope demotes
    # SOFTLY (ADR-046 — a mis-scope demotes, never excludes). scope_conf == 1.0 for
    # deterministic keyword/gazetteer/explicit-arg scope => byte-identical to today.
    base_off_country = qk("off_country", _OFF_COUNTRY_W)
    k_off_country = base_off_country * scope_conf
    k_country_boost = qk("country_boost", _COUNTRY_BOOST)
    q_terms = _query_terms(q)
    q_content = _content_terms(q, lang)   # for title-centrality (task 104)
    q_academic = _query_academic_intent(q)
    # task 254: commercial/product intent (down-weight reference + gov trust so
    # providers surface). Flag-off => always False => byte-identical scoring.
    commercial = _COMMERCIAL and _query_commercial_intent(q, lang)
    # task 404 (a): person-name-shaped query (NAME-GATE trigger). Flag-off or
    # zero weight => None => byte-identical scoring. When set, these are the
    # NAME tokens (capitalized content terms) coverage is measured against.
    # task 406 widens the arming condition to ANY person-name lever (H1 dense
    # down-weight / H2 coverage bonus / H3 trustkeep), so the detector runs when
    # the 404 penalty is off but a 406 half is on. All-off => None => unchanged.
    namegate_tokens = (_person_name_tokens(q, q_content)
                       if _NAME_LEVERS_ARMED else None)
    # HUB-FLOOR intent gate (task 282): general-intent from the ADR-036 head
    # (rh_beta = the per-query general beta; ==0 on authority lookups -> the floor
    # never fires -> legal/authority hold preserved by construction). Requires the
    # head ON; flag-off / head-off / no hub table => inert (byte-identical).
    hubfloor_on = bool(_HUBFLOOR and _HUB_HOSTS
                       and rh_beta >= _HUBFLOOR_MINBETA and rh_beta > 0)
    # AUTHORITY (ADR-019): the endorsement boost is live whenever the flag + a
    # non-zero weight + a loaded table are all present. Unlike the hub-floor it is
    # NOT intent-gated — the authority-lift target includes authority-lookup golds
    # (legal/names) where rh_beta==0; per-candidate deferral (hubfloored /
    # geo_penalized) is what keeps it from fighting the demotion levers.
    authority_on = bool(_AUTHORITY and _AUTHORITY_W and _AUTH_HOSTS)
    s1 = stage1(Stage1Request(
        q=q, lang=lang, k=k, candidates=candidates, country=country, dense=dense, lexical=lexical,
        query_plan=query_plan, scoped=scoped, q_cantons=q_cantons, q_country=q_country,
        q_country_soft=q_country_soft, scope_conf=scope_conf, base_off_country=base_off_country,
        w_canton=w_canton, w_foreign=w_foreign, k_dense_scale=k_dense_scale,
        k_bm25_scale=k_bm25_scale, experimental=experimental,
        target_bm25_language=target_bm25_language, target_bm25_quota=target_bm25_quota,
        target_bm25_weight=target_bm25_weight, translated_bm25_query=translated_bm25_query,
        translated_bm25_quota=translated_bm25_quota,
        translated_bm25_weight=translated_bm25_weight, op_filter=op_filter))
    if s1 is None:
        return []
    bm, qfilter, payloads, raw, norm = s1.bm, s1.qfilter, s1.payloads, s1.raw, s1.norm
    global_raw, country_leg_provenance = s1.global_raw, s1.country_leg_provenance
    best_dense_raw, rare_toks, zero_signal = s1.best_dense_raw, s1.rare_toks, s1.zero_signal
    _name_primaries, _reqterm, _reqterm_groups, _rh_bm25_best, _rh_bm25_rank, _rh_bm25_raw, _rh_coverage_available, _rh_dense_rank, _rh_dense_raw, _tf_abs_armed, _tf_abs_hit, cov, payloads, presence = levers.coverage_probe(
        best_dense_raw=best_dense_raw, bm=bm, lang=lang, namegate_tokens=namegate_tokens,
        payloads=payloads, q=q, qfilter=qfilter, rare_toks=rare_toks, raw=raw, rh_beta=rh_beta)
    _k_dense_name, _name_intent, _ng_cov, auth_defer, rows = levers.build_rows(
        _name_primaries=_name_primaries, _reqterm=_reqterm, authority_on=authority_on,
        base_off_country=base_off_country, commercial=commercial, cov=cov, hubfloor_on=hubfloor_on,
        k_bm25_scale=k_bm25_scale, k_dense_scale=k_dense_scale, k_exact=k_exact,
        k_machine=k_machine, k_off_country=k_off_country, k_recency=k_recency, k_title=k_title,
        lang=lang, namegate_tokens=namegate_tokens, norm=norm, payloads=payloads, presence=presence,
        q=q, q_academic=q_academic, q_cantons=q_cantons, q_content=q_content, q_country=q_country,
        q_country_soft=q_country_soft, q_terms=q_terms, rare_toks=rare_toks, scope_conf=scope_conf,
        scoped=scoped, target_bm25_weight=target_bm25_weight,
        translated_bm25_weight=translated_bm25_weight, w_canton=w_canton, w_foreign=w_foreign)
    _primacy_pids, best_rel_core, best_rel_undemoted, gate_thr, rel_rank = levers.apply_levers(
        _ng_cov=_ng_cov, _rh_bm25_best=_rh_bm25_best, _rh_bm25_rank=_rh_bm25_rank,
        _rh_bm25_raw=_rh_bm25_raw, _rh_coverage_available=_rh_coverage_available,
        _rh_dense_rank=_rh_dense_rank, _rh_dense_raw=_rh_dense_raw, auth_defer=auth_defer,
        authority_on=authority_on, cov=cov, lang=lang, namegate_tokens=namegate_tokens, norm=norm,
        q_cantons=q_cantons, q_content=q_content, q_country=q_country,
        q_country_soft=q_country_soft, qk=qk, rare_toks=rare_toks, rh_beta=rh_beta, rows=rows,
        scope_conf=scope_conf, host_label=_host_label(q))
    relmeta, scored = fusion.score_rows(
        _k_dense_name=_k_dense_name, _name_intent=_name_intent, _ng_cov=_ng_cov,
        _primacy_pids=_primacy_pids, _reqterm=_reqterm, _reqterm_groups=_reqterm_groups,
        _tf_abs_armed=_tf_abs_armed, _tf_abs_hit=_tf_abs_hit, best_dense_raw=best_dense_raw,
        best_rel_core=best_rel_core, best_rel_undemoted=best_rel_undemoted, commercial=commercial,
        country=country, country_leg_provenance=country_leg_provenance, experimental=experimental,
        gate_thr=gate_thr, global_raw=global_raw, k_country_boost=k_country_boost,
        k_trust_scale=k_trust_scale, norm=norm, query_plan=query_plan, raw=raw, rel_rank=rel_rank,
        rh_beta=rh_beta, rows=rows)
    return respond.finish(
        country=country, k=k, relmeta=relmeta, scored=scored, zero_signal=zero_signal)


def effective_config() -> dict:
    """`/health`'s `config` block: `settings.effective_config` (a projection of `SETTINGS`) with
    this module's two runtime facts, the v2 head's load state and the weights file read now."""
    return _settings.effective_config(_S, rankhead_v2_loaded=_RANKHEAD_V2_MODEL is not None,
                                      fusion_weights=weights_fingerprint())
