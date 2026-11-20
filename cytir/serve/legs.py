"""Stage 1 of `search()`: the leg queries, the per-leg score maps and the stage-1 pool (S10 step 8a,
task 1116; moved verbatim from `retrieval.py`, which imports every constant back).

`stage1(Stage1Request)` is the one entry. It runs three steps, each a top-level function whose
inputs are named in its signature and whose outputs are a frozen dataclass; nothing is read from a
caller's frame and nothing but the typed settings is read from the environment.

    run_legs       tokens, filters, payload selector, leg dispatch (one batch or sequential),
                   the authority-partition admit list, the per-token sub-leg batch      -> LegResults
    assemble_pool  raw score maps per leg, stage-1 country / scope terms, the country-partition
                   merge, one min-max per modality, the per-token merge, the zero-signal flag
                                                                                        -> Pool | None
    cut_pool       the k-stability margin cut (`fuse_margin`), the page cap, the authority admit,
                   and the one batched payload fetch                                    -> payloads

An empty candidate set is `None` (the caller returns `[]`). The three stage-1 closures that
`search()` carried (`_stage1_meta`, `_trace_points`, and the `_s1*`/`_page` helpers of the cut) are
top-level (`stage1_meta`, `trace.trace_points`, `fusion.s1_*`, `fusion.page_of`); each block binds
them back to the old local name with `functools.partial`, so the block text is unchanged.
No import of `retrieval`.
"""
from __future__ import annotations

import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from qdrant_client import models as qm
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from . import qdrant_io
from .article_tokens import glued_article_variants
from .fused_split_terms import fused_split_variants
from .fusion import (_RELFLOOR, _auth_admit, _extend_pool, _order_ids, page_of, s1_boost, s1_key, s1_rel)
from .geo import _effective_country, _is_eu_supra, _learned_scope_conflict_penalty, _scope_penalties
from .heads import _LEARNED_ONLY_SCOPE
from .operators import and_filter
from .plan import _country_partition_filter, fuse_margin
from .qdrant_io import _leg
from .respond import _canonical_key
from .schema import host_query_parts
from . import settings as _settings
from .schema import COLLECTION, DENSE_VEC, SPARSE_BM25, SPARSE_LEXICAL, bm25_tokens
from .settings import SETTINGS as _S
from .signals import _minmax
from .trace import _LAST_EXPERIMENT_TRACE, _LAST_QUERY_PLAN_TRACE, _QHEAD_COUNTRY_TRACE, trace_points

log = logging.getLogger(__name__)

# task 183: exempt entity cards from the hard per-leg `lang` filter (nested
# should: lang==query_lang OR source_type==entity_card). Default ON — this IS
# the fix for the lang=en card invisibility (task 182). Set
# The lang exemption is FIXED (constant _CARD_LANG_EXEMPT below; no env setter since
# S10 D5). The plain must-filter needs a constant edit plus redeploy (no instant rollback). REQUIRES the `source_type` keyword payload index (built
# 2026-07-12 on aw_prod + aw_v4_sat); without it the should-filter is a
# measured ~12x dense-leg regression.
_CARD_LANG_EXEMPT = True  # fixed: was env AW_CARD_LANG_EXEMPT (S10 D5: no product setter)

# Binary-quant rescore (ADR-011): oversample then rescore against full-precision
# vectors so dense recall isn't lost to quantization.
# Eval-backed (2026-07): rescore=off matches oversmp=3.0 recall (hit@3 55%, MRR 0.45)
# at ~6x lower latency on this box (full-vector rescore = pure cost). See log.md.
#
# task 329: env-gated dense-leg hnsw_ef. Incident: aw_b1's bulk-built HNSW
# graphs (m=0->16 build + post-build delta upserts 322/326, optimizer-merged)
# have a catastrophic recall hole under the task-183 card-exempt should-filter
# — 0/57 legifrance in the dense top-100 for fr-law-002's vector vs 57 under
# must=[lang], exact=true, or explicit hnsw_ef>=150; raising the leg LIMIT
# alone does NOT close it (ef does not follow limit on this path); the twin
# aw_v4_sat is unaffected. See task 328 ("recall hole")
# + notes/qdrant-should-filter-dense-recall.md. Unset/0 => hnsw_ef=None, the
# server-side default — byte-identical to pre-329 behavior; set (e.g.
# AW_DENSE_HNSW_EF=150) => explicit graph beam width on the dense leg only.
_DENSE_HNSW_EF = _S.legs.dense_hnsw_ef or None

# task 514: the two lines below were LITERALS, and the measurement that
# justified them was made in 2026-07 against a BINARY-quantized 128-dim index.
# The flip target is `256-pq8` -- product quantization at 8x, whose ratified
# serving config (cytir/core/serving_config.py, ADR-011 amendment) declares
# `rescore=True, oversampling=3.0`.
#
# ~~A product-quantized search with rescore OFF ranks on the 128-byte codes
# alone, and the codes are much lossier than binary was at 128 dims. Measured on
# the built collection: hit@1 276 -> 106 and hit@10 390 -> 290 against the prod
# twin ... i.e. the whole gap was the query path not honouring the quantizer the
# collection was built with.~~
#
# FALSIFIED, task 515/516/517. That 106 was the serving-metadata defect (title,
# trust_tier and source_type absent on 36.5% of the collection, no BM25 posting
# on 63.5%), not the quantizer. With the payload fixed, the SAME collection at
# rescore OFF scores 277/393 and at rescore ON x3 scores 276/392 on the 629-query
# gate (task 516 sec.7) -- one query apart. The knob is real and stays; the
# mechanism the comment claimed for it was wrong, and it is left struck through
# rather than deleted because the next person to see a big number move will
# reach for the same explanation.
#
# Defaults are exactly today's behaviour, so prod is byte-identical until the
# compose file sets them -- the values now live where the rest of the serving
# configuration lives, as data, instead of as a constant whose evidence names a
# different index.
_DENSE_RESCORE = _S.legs.dense_rescore

_DENSE_OVERSAMPLING = _S.legs.dense_oversampling

# task 517: search the ORIGINAL vectors and ignore the quantizer entirely.
# Needed because the 1024-rebuild's gate arms differ from the twin in TWO ways
# at once -- the collection AND the serving width/quantizer (twin 128-dim int8,
# build 256-dim product-x8) -- and task 516 sec.9.2 could not separate them.
# `ignore=True` makes the build's own collection answer from its 256-dim f16
# originals, which turns the quantizer into one flag instead of a confound.
# Default 0 = today's behaviour, byte-identical.
_DENSE_IGNORE_QUANT = _S.legs.dense_ignore_quant

_DENSE_SP = qm.SearchParams(
    hnsw_ef=_DENSE_HNSW_EF,
    quantization=qm.QuantizationSearchParams(
        ignore=_DENSE_IGNORE_QUANT,
        rescore=_DENSE_RESCORE, oversampling=_DENSE_OVERSAMPLING))

# Two-stage payload fetch (task 072): legs run with_payload=False, a first-pass
# fusion on leg scores alone picks the top AW_FUSE_MARGIN ids, and ONE batched
# retrieve() fetches just those payloads (vs up to 2x`candidates` random RocksDB
# payload reads). Full fused score (trust/recency/exact) is then recomputed
# within that pool. Flag-gated: AW_TWO_STAGE=1 enables; unset/0 = legacy path.
_TWO_STAGE = _S.legs.two_stage

# --------------------------------------------------------------------------- #
#  TASK 582 -- AW_PAGECAP: a per-page cap in the STAGE-1 pool (experiment
#  E582-1, the cure task 580 §7 prescribed and never tested).
#
#  The served list is already one-row-per-page: the canonical-URL dedup at the
#  bottom of search() keeps each page's highest-scoring chunk and drops the
#  rest, and task 582 measured 0 duplicate pages across 3,396 sealed gold
#  results on both collections and all three legs. What is NOT deduplicated is
#  everything upstream: a leg returns `candidate_pool(k)` ROWS, so a page cut
#  into 12 windows can hold 12 of the 100 slots, and the stage-1 margin cut
#  ranks rows, not pages.
#
#  Two modes, both applied at the ONE point a row-vs-page decision is made --
#  immediately before the margin cut, which is why this is "a per-page cap in
#  the pool" and not another dedup:
#
#    cap  keep at most AW_PAGECAP rows per canonical page, best `_s1rel` first,
#         then cut at the margin. AW_PAGECAP=1 is the literal reading.
#    agg  max-over-page aggregation: a page's stage-1 relevance IS its best
#         row's, pages are admitted in that order, and a page's rows enter
#         together (best first) until the margin is full.
#
#  AW_PAGECAP_OVERSAMPLE > 1 widens each leg's own limit by that factor first,
#  so the cap can convert freed slots into DISTINCT pages instead of merely
#  shortening the pool -- without it the cap is inert whenever |ids| <= margin,
#  which task 582 measured to be the normal case on the leg-isolated arms
#  (pool 147-190 rows against a margin of 200).
#
#  COST, named because it is a serving change and not a scoring tweak: page
#  identity lives in the payload and the two-stage path fetches no payloads in
#  stage 1 (that is the whole point of task 072). The cap therefore adds ONE
#  batched `retrieve` over the stage-1 ids with a `url`-only payload selector,
#  before the margin cut. Default OFF => byte-identical; rollback AW_PAGECAP=0.
# --------------------------------------------------------------------------- #
_PAGECAP = _S.legs.pagecap

_PAGECAP_MODE = _S.legs.pagecap_mode

_PAGECAP_OVERSAMPLE = _S.legs.pagecap_oversample

# --------------------------------------------------------------------------- #
#  CHANGE A — inject-only authority leg (task 202, from the 201 experiment GO).
#  A buried-but-authoritative page (a CH/DE/FR gov/legal source that global
#  relevance ranks far below the pool cut, e.g. fr-gov-024 beyond unfiltered
#  dense rank 3000) can sit at the TOP of the small authority partition
#  (trust_tier >= THRESH). We run ONE extra dense query filtered to that
#  partition (REUSING the query vector — no extra embed), admit its top-N into
#  the stage-1 pool, and FORCE-INCLUDE them into the stage-2 candidate set so
#  the UNCHANGED two-stage fusion (trust + title-centrality + exact) can lift
#  them. This is a pool-ADMISSION widener, NOT a scorer: there is deliberately
#  NO authority fusion weight (201 proved AUTH_W>0 net-negative — it regresses
#  software/pricing/news). Admission is PARTITION-RELATIVE: top-N of the
#  authority ANN with only a LIGHT floor vs the partition's OWN best
#  (score >= BETA * max_authority_score), never vs the global pool best (which
#  would re-exclude the low-global-relevance target this whole leg exists to
#  reach). 201: +5 golds / ZERO per-query loss on 478q, +3.8 ms qdrant-side.
#  Flag-gated AW_AUTHLEG (default OFF, byte-identical). THRESH=0.95 (tier-A),
#  N=10 (identical to N=20 at W=0, cheaper), BETA=0.3. Env-tunable; rollback =
#  AW_AUTHLEG=0. trust_tier is payload-indexed across all 47M so the filtered
#  ANN is a cheap indexed scan (tier-A is ~6% of the index).
# --------------------------------------------------------------------------- #
_AUTHLEG = _S.legs.authleg

_AUTHLEG_THRESH = 0.95  # fixed: was env AW_AUTHLEG_THRESH (S10 D5: no product setter)

_AUTHLEG_N = 10  # fixed: was env AW_AUTHLEG_N (S10 D5: no product setter)

_AUTHLEG_BETA = 0.3  # fixed: was env AW_AUTHLEG_BETA (S10 D5: no product setter)

# task 2665: a FILTERED authority leg (lang= / operator filter ANDed with trust_tier>=0.95) is the
# one stage-1 search that outlives the API's Qdrant budget on the 30 GB on-disk box: measured alone
# on prod 10.3 s (lang=fr) and 19.2 s (lang=it) against 2.1-2.5 s unfiltered or lang=de, while the
# batch it rides in has ONE AW_QDRANT_TIMEOUT (8 s) budget -> every lang=fr/it search was a 503.
# N > 0 issues that leg (and only that leg, only when a filter is set) as its own request, in
# parallel with the stage-1 batch, with Qdrant's SERVER-side `timeout=N` s (which also stops the
# search work on the box — measured on the twin). A timeout drops the admission widener for that
# request and keeps the request. 0 (default) = the unchanged single-batch dispatch.
_AUTHLEG_FILTERED_TIMEOUT = _S.legs.authleg_filtered_timeout
_AUTHLEG_POOL = None
# task 2686: a COLD filtered stage-1 batch (lang=fr|it, page cache evicted on the 30 GB box) outlives the
# AW_QDRANT_TIMEOUT client budget (5 s) and the whole request is a 503, although the same batch answers
# in about a second once warm. N > 0 gives the stage-1 batch and the per-token batch their own per-call
# timeout of N s. qdrant-client sends it as Qdrant's SERVER-side `timeout` AND as the HTTP read timeout
# of that call, so a loser is cancelled on the box and the client gives up at the same moment: a slow
# 200 instead of a 503, and no search work left running after an abandoned request. 0 (default) = no
# per-call timeout, the client-wide AW_QDRANT_TIMEOUT, byte-identical calls.
_STAGE1_TIMEOUT = _S.legs.stage1_timeout


def _stage1_kw() -> dict:
    return {"timeout": _STAGE1_TIMEOUT} if _STAGE1_TIMEOUT > 0 else {}


#: task 2665 (review 2671 F1): this request's bounded authority leg outcome, read by the API into
#: the response's `degraded` field: None (ran, or not split), "timed_out" (Qdrant's server-side
#: timeout) or "error" (any other Qdrant error answer / transport failure of that one call).
AUTHLEG_DEGRADED: ContextVar = ContextVar("aw_authleg_degraded", default=None)

# task 342: issue the independent stage-1 leg searches as ONE query_batch_points
# instead of N sequential round trips. Latency-only — the requests, and the
# (id, score) lists they return, are unchanged (see the LEG DISPATCH block in
# search()). Default OFF => byte-identical dispatch; rollback = AW_BATCH_LEGS=0.
_BATCH_LEGS = _S.legs.batch_legs

# internal spec key for the authority-partition leg — it is NOT a scoring leg
# and must never enter `legs`/`raw`/`norm` (it has no fusion weight, ADR: CHANGE
# A is pool ADMISSION only). Leading "_" so it cannot collide with a real leg.
_AUTH_LEG_KEY = "_authleg"

# --------------------------------------------------------------------------- #
#  RELEVANCE-FLOOR confidence marker (task 207) — NON-DESTRUCTIVE, advisory.
#  The 206 intent head gates TRUST-driven flood but not RELEVANCE-driven flood
#  (a prolific mid-trust blog filling the tail of a vague query). The task-207
#  Phase-1 separability analysis (task 207) proved a per-QUERY
#  relevance floor is NOT thresholdable: within general-intent queries the flood
#  winning-relevance distribution [1.00..2.04] is entirely NESTED inside the
#  answerable one [1.00..2.73] (best Youden J ~0.29-0.40; catching 50% of floods
#  trims 27% of answerable), and the legitimate low-relevance-high-authority
#  answers (RMF at dense-cos 0.044, terse statutes) sit at the BOTTOM of the
#  relevance band — an aggressive floor decapitates them first. So we do NOT trim
#  and do NOT silently drop results (would violate the zero-answerable-trim gate).
#  Instead — per Tim's parked note "agents/MCP can USE the signal; expose it,
#  don't silently drop" — when AW_RELFLOOR=1 we ATTACH an advisory per-result
#  `confidence` object (pool-relative relevance `rel_ratio`, trust-share
#  `trust_frac`, `low_relevance` flag vs beta, and whether the query is general-
#  intent). It changes NO score and NO ordering and TRIMS NOTHING — flag-off is
#  byte-identical; the agent decides. Kill-switch: AW_RELFLOOR=0 (default).
# --------------------------------------------------------------------------- #

_RELFLOOR_DENSE_MAX = 0.68  # fixed: was env AW_RELFLOOR_DENSE_MAX (S10 D5: no product setter)

# Per-token bm25 sub-legs (task 083, interim entity-query fix). The STORED
# bm25 sparse weights are RAW tf*idf, not BM25-saturated (e.g.
# en.wikipedia/Laura stores laura=36.0 where real BM25 would saturate ~2.2).
# For multi-token queries the high-tf pages of the COMMON token therefore
# flood the entire main bm25 leg top-N ("Laura" list pages score 145-249 vs
# ~29 for pages containing the rare "Tocmacov"), and if dense doesn't carry
# the entity pages either, they enter NO leg — no downstream weight can
# rescue them. Interim fix (real fix = re-store BM25-saturated weights);
# for >=2-token bm25 queries, all flag-gated on AW_BM25_PER_TOKEN:
#  (1) run one cheap sparse sub-query PER TOKEN (ONE batched qdrant call,
#      candidate generation only, with_payload as the main legs) and union
#      the ids into the stage-1 candidate set with
#        bm25_norm_eff = max(main-leg minmax-norm, sub-leg norm * 0.9).
#      Sub-leg norm: RARE tokens (window not full => df < limit, enormous
#      idf — real BM25 would rank every one of these docs at the very top
#      of the bm25 leg) get saturated-tf tf̂/(tf̂+k1) with tf̂ = s/min(s)
#      (min ≈ the tf=1 score, s ∝ tf within a single-token leg). Plain
#      minmax would zero-floor exactly the tf=1 entity pages (measured:
#      pool cutoff 0.221, target page norm ~0 for "Tim O'Hear"). COMMON
#      tokens (full window) keep minmax — their top docs are the same
#      flood the main leg already ranks, a no-op under max().
#  (2) stage-2: ONE more batched sparse call, filtered to the pool ids,
#      gives full-document term-presence per pool doc ("which pool docs
#      contain token t" — the stored bm25 vector is the oracle; we hold no
#      passage text). exact := max(title/snippet exact, RARE-token
#      coverage). This is the exact-match feature it was always meant to
#      be: snippets are None across the index, so `_exact` alone is
#      title-only and the entity page ("Conseil de fondation - impactIA",
#      contains tim+hear+ohear each tf=1, title matches neither term) can
#      NEVER outscore the dense name-flood (its ceiling 0.6*bm25 +
#      0.7*trust = 1.24 vs the measured top-10 floor 1.87). Coverage
#      rewards the one thing the flood pages lack: the distinctive (rare)
#      query terms actually appearing in the document. Rare tokens only —
#      all-token coverage boosted verbose pages containing every function
#      word of long questions (measured -0.14 MRR on the 29q eval);
#      queries with no rare token keep baseline stage-2 scoring.
# SHORT queries only (2..MAXQ bm25 tokens): in a short entity-style query
# the rare token IS the user's intent ("Tocmacov", "O'Hear", an error code)
# and coverage aligns with relevance. In a long natural-language question a
# rare token is usually incidental (an inflected form, a decompounded
# half), the target page need not contain that exact surface form, and the
# coverage boost promoted whichever verbose page did (measured on the
# question evals: 29q MRR 0.400->0.366, 117q-sample hit@1 39->32 with the
# machinery unrestricted; both fully recovered by this gate). MAXQ also
# bounds the sub-query fan-out (latency).
# Requires AW_TWO_STAGE=1 (sub-leg hits need the batched stage-2 payload
# fetch; the legacy path only scores docs whose payload arrived with a
# leg). Flag off => byte-identical current behavior (verified same-process
# on 10 queries, 2026-07-03).
_PER_TOKEN = _S.legs.bm25_per_token

_PER_TOKEN_LIMIT = _S.legs.bm25_per_token_limit

_PER_TOKEN_MAXQ = 4  # fixed: was env AW_BM25_PER_TOKEN_MAXQ (S10 D5: no product setter)

_PER_TOKEN_SUB_W = 0.9  # fixed: was env AW_BM25_PER_TOKEN_SUB_W (S10 D5: no product setter)

# k1 for the query-side tf saturation of rare-token sub-legs. Lower than the
# classic 1.2 because we have no doc-length normalization and rare-token
# tf=1 docs (df<50 in a 44M index) should still clear the stage-1 pool
# cutoff (~0.24): tf1 -> 0.67, tf2 -> 0.8, tf29 -> 0.98.
_PER_TOKEN_K1 = 0.5  # fixed: was env AW_BM25_PER_TOKEN_K1 (S10 D5: no product setter)

# --------------------------------------------------------------------------- #
#  TASK 606 — the SPARSE-leg zero guard, the main-leg half of task 336's fix.
#
#  A *filtered* sparse query against a **plain (unindexed) segment** returns
#  filter-matching points that do NOT contain any query term, scored 0.0. Task
#  336 found this the hard way ("a 90-point upsert on aw_b1 broke 4 of 663 gate
#  golds") and guarded the per-token sub-legs; the stage-2 coverage oracle
#  carries the same guard inline (`if p.score > 0`). The MAIN legs never got it,
#  and they are the ones whose min-max sets the scale of `rel.bm25`.
#
#  What it costs when it fires: `lo` collapses to 0.0, so every real match's
#  bm25_norm changes from (v-lo)/(hi-lo) to v/hi — on a measured pool
#  (lo 28.7, hi 64.9) a row at 35 moves 0.18 -> 0.55, a +0.37 swing in rel_core
#  against a median gate margin of 0.31. That is not a tie-break; it re-scales
#  the leg.
#
#  Why it is latent rather than live: a leg that returns its full `candidates`
#  window of positive scores never admits a 0.0 point, because qdrant merges
#  segment results by score. Measured on the 1,418-query battery against aw_e8
#  (task 606): 38 of 1,394 BM25 legs (2.7%) come back short of 100 — the narrow,
#  rare-term queries, which is exactly the legal-lookup population the gate
#  protects. Qdrant keeps a segment below `indexing_threshold` PLAIN while still
#  reporting `status: green` / `optimizer_status: ok`, so a routine content pack
#  leaves one behind and `wait_green` does not wait it away.
#
#  A non-positive sparse score means "this point does not contain the term", so
#  dropping it from the normalisation is what an INDEXED collection would have
#  returned. The point keeps its place in the candidate set and reads
#  `bm25_norm = 0.0` through `norm.get(pid, 0.0)` — it is not admitted, and it
#  is not silently deleted either.
#
#  Flag-gated and OFF by default: this is a ranking change and meets the ADR-043
#  per-gold zero-loss bar before it is armed. Measured inert on today's twin
#  (task 606: 0 golds moved, h@1 219 unchanged) — because today's aw_e8 carries
#  no plain segment. Rollback = AW_BM25_ZERO_GUARD=0.
#  NEVER applied to the dense leg: a cosine of 0.0 (or below) is a real score.
# --------------------------------------------------------------------------- #
_BM25_ZERO_GUARD = _S.legs.bm25_zero_guard

#: legs whose scores come from a sparse (term) vector, where a non-positive
#: score is a non-match rather than a weak match.
_SPARSE_SCORE_LEGS = frozenset({"bm25", "target_bm25", "translated_bm25",
                                "lexical"})


@dataclass(frozen=True)
class Stage1Request:
    """Everything stage 1 reads from the request and from query understanding."""
    q: str
    lang: str | None
    k: int
    candidates: int
    country: str | None
    dense: Any                      # the query's dense vector
    lexical: Any                    # the embedder's learned-sparse dict, or empty
    query_plan: Any                 # plan.QueryPlan
    scoped: bool
    q_cantons: Any
    q_country: str | None
    q_country_soft: str | None
    scope_conf: float
    base_off_country: float
    w_canton: float
    w_foreign: float
    k_dense_scale: float
    k_bm25_scale: float
    experimental: bool
    target_bm25_language: str | None
    target_bm25_quota: int
    target_bm25_weight: float
    translated_bm25_query: str | None
    translated_bm25_quota: int
    translated_bm25_weight: float
    op_filter: Any = None           # task 2640: site:/-site:/source: operator filter, ANDed into every leg


@dataclass(frozen=True)
class LegResults:
    """The leg dispatch: query tokens, language filter, payload selector, leg hits."""
    bm: dict
    qfilter: Any
    wp: Any
    soft_scoped: bool
    specs: list
    legs: dict                      # leg name -> list of ScoredPoint (authority leg excluded)
    auth_admit: dict
    tok_legs: list
    pertoken_batch_calls: int
    stage1_request_ms: float | None
    tld_hashes: frozenset = frozenset()   # AW_HOST_QUERY_CARD: TLD token hash of a host-shaped query
    auth_split: bool = False       # task 2665: the filtered authority leg ran beside the batch
    auth_leg_status: str | None = None   # task 2665: ... None, "timed_out" or "error" (no admits)


@dataclass(frozen=True)
class Pool:
    """The assembled candidate pool: raw and normalized per-leg scores (stage 2 reads them)."""
    payloads: dict
    raw: dict
    norm: dict
    ids: set
    cty1: dict
    scope1: dict
    global_raw: dict
    country_leg_provenance: dict
    best_dense_raw: float
    rare_toks: list
    zero_signal: bool


@dataclass(frozen=True)
class Stage1:
    """What stage 2 takes over."""
    bm: dict
    qfilter: Any
    payloads: dict
    raw: dict
    norm: dict
    global_raw: dict
    country_leg_provenance: dict
    best_dense_raw: float
    rare_toks: list
    zero_signal: bool


def stage1_meta(pid, pl, *, cty1, scope1, scoped, q_cantons, q_country, w_canton, w_foreign,
                soft_scoped, q_country_soft, scope_conf, base_off_country):
    c = _effective_country(pl)   # task 261: payload stamp, else AW_COUNTRY_MAP
    if c:
        cty1[pid] = c
    penalty = 0.0
    if scoped:
        p_c, p_f = _scope_penalties(q_cantons, q_country, pl.get("canton"),
                                    c, pl.get("source_type"),
                                    _is_eu_supra(pl.get("domain")))
        if p_c or p_f:
            penalty += w_canton * p_c + w_foreign * p_f
    if soft_scoped:
        penalty += _learned_scope_conflict_penalty(
            q_country_soft, scope_conf, pl, base_off_country)
    if penalty:
        scope1[pid] = penalty


def _authleg_executor():
    global _AUTHLEG_POOL
    if _AUTHLEG_POOL is None:
        _AUTHLEG_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="aw-authleg")
    return _AUTHLEG_POOL


def _is_timeout(e: Exception) -> bool:
    """Qdrant's server-side timeout answers HTTP 500 with "Timeout error: ... timed out" in the
    body; a client-side read timeout arrives as ResponseHandlingException(httpx timeout)."""
    if isinstance(e, UnexpectedResponse):
        body = (e.content or b"").decode("utf-8", "replace").lower()
        return "timeout" in body or "timed out" in body
    return "timeout" in type(getattr(e, "source", None)).__name__.lower()


def _bounded_authority_leg(dense, a_filter) -> tuple[list, str | None]:
    """task 2665: the filtered authority leg alone, under Qdrant's server-side timeout.
    Returns (points, status) with status None, "timed_out" or "error". Only a Qdrant error answer
    or a transport failure of THIS call is absorbed; the stage-1 batch still carries its own
    failure, so a Qdrant outage still surfaces as the request's error."""
    try:
        return qdrant_io.client().query_points(
            collection_name=COLLECTION, query=dense, using=DENSE_VEC, limit=_AUTHLEG_N,
            query_filter=a_filter, search_params=_DENSE_SP, with_payload=False,
            timeout=_AUTHLEG_FILTERED_TIMEOUT).points, None
    except (UnexpectedResponse, ResponseHandlingException) as e:
        status = "timed_out" if _is_timeout(e) else "error"
        log.warning("authority leg dropped (filtered, %s, budget %ss): %s", status,
                    _AUTHLEG_FILTERED_TIMEOUT, str(e)[:160])
        return [], status


def host_query_label(q: str) -> str | None:
    """AW_HOST_QUERY_CARD (task 2050): the label of a host-shaped query ("ethz" for "ethz.ch"),
    else None. Always None with the flag off, so the card leg takes its unchanged branch."""
    if not _S.query.host_query_card:
        return None
    parts = host_query_parts(q)
    return parts[0] if parts else None


def host_query_tld_hashes(q: str, lang: str | None) -> frozenset:
    """AW_HOST_QUERY_CARD (task 2050): the BM25 hash(es) of a host-shaped query's TLD token, so
    the df=0 count skips it. Empty with the flag off or a non-host query: nothing changes."""
    if not _S.query.host_query_card:
        return frozenset()
    parts = host_query_parts(q)
    return frozenset(bm25_tokens(parts[1], lang)) if parts else frozenset()


def run_legs(*, q, lang, dense, lexical, candidates, country, query_plan, scoped, q_country_soft,
             experimental, target_bm25_language, target_bm25_quota, target_bm25_weight,
             translated_bm25_query, translated_bm25_quota, translated_bm25_weight,
             op_filter=None) -> LegResults:
    """Stage 1, step 1: build the legs and dispatch them."""
    bm_q = glued_article_variants(q) if _settings.current("AW_QUERY_GLUED_ARTICLE") else q
    if _settings.current("AW_QUERY_FUSED_SPLIT"):  # task 2856: generic superset of the article lever
        bm_q = fused_split_variants(bm_q)
    bm = bm25_tokens(bm_q, lang)
    bm25 = qm.SparseVector(indices=list(bm.keys()), values=list(bm.values()))
    translated_bm = None
    if translated_bm25_query:
        translated_tokens = bm25_tokens(translated_bm25_query, target_bm25_language)
        translated_bm = qm.SparseVector(indices=list(translated_tokens.keys()),
                                        values=list(translated_tokens.values()))

    qfilter = None
    if lang:
        if _CARD_LANG_EXEMPT:
            # task 183 (root cause task 182): entity cards are language-neutral
            # IDENTITY anchors — a lang=en query must still see the (lang=de)
            # Swisscom/Nestlé card, else the 163 boost has nothing to boost and
            # same-name-prefix sub-articles flood the top-k. Nested should =
            # lang==query_lang OR source_type==entity_card. Both fields carry
            # keyword payload indexes (source_type index added 2026-07-12,
            # prod + twin); measured latency-neutral vs the plain must filter
            # (twin: dense leg 3.4→3.6 ms p50, bm25 4.0→4.0 ms). Content
            # language hygiene (166) is untouched — only cards are exempt.
            qfilter = qm.Filter(should=[
                qm.FieldCondition(key="lang", match=qm.MatchValue(value=lang)),
                qm.FieldCondition(key="source_type",
                                  match=qm.MatchValue(value="entity_card")),
            ])
        else:
            qfilter = qm.Filter(must=[qm.FieldCondition(key="lang", match=qm.MatchValue(value=lang))])

    target_bm25_filter = None
    if target_bm25_language:
        target_bm25_filter = qm.Filter(should=[
            qm.FieldCondition(key="lang", match=qm.MatchValue(value=target_bm25_language)),
            qm.FieldCondition(key="source_type", match=qm.MatchValue(value="entity_card")),
        ])
    # task 2640: the operator filter joins `qfilter` HERE, so every consumer below (main legs,
    # authority leg, country-partition legs, per-token sub-legs, coverage probe via Stage1) sees it.
    # None (no operator) leaves every filter object untouched.
    if op_filter is not None:
        qfilter = and_filter(qfilter, op_filter)
        target_bm25_filter = and_filter(target_bm25_filter, op_filter)

    # ADR-018: dense + BM25 (learned-sparse dropped — redundant with BM25). The
    # learned-sparse leg is added back only if the embedder still emits it (BGE-M3).
    # two-stage: legs skip payloads (fetched batched below) -- except when a
    # country is selected (legs fetch ONLY the small `country` field so the
    # boost can shape stage-1 pool selection) or the query is jurisdiction-
    # scoped (task 123: also `canton`/`source_type`, so the scope penalties
    # can shape stage-1 too — a stage-2-only penalty cannot rescue a federal
    # gold that the cantonal flood pushed below the margin cut; measured:
    # ch-law-011 gold at leg-rank 46 > margin 40).
    soft_scoped = bool(_LEARNED_ONLY_SCOPE and q_country_soft)
    if not _TWO_STAGE:
        wp = True
    elif soft_scoped:
        # Stage-1 must see the same fields as final scoring; otherwise a foreign
        # official flood can evict the domestic gold before the soft penalty is
        # ever evaluated.  This selector exists only in the experimental arm.
        wp = ["country", "canton", "source_type", "domain", "register_id", "lei"]
    elif scoped:
        wp = ["country", "canton", "source_type", "domain"]  # +domain: task 164 EU exemption
    elif country or query_plan.country_legs_enabled:
        wp = ["country"]
    else:
        wp = False
    # CHANGE A (task 202): inject-only authority leg. Same query vector + a
    # trust_tier>=THRESH payload filter, so a buried-but-authoritative page
    # competes against the small authority population, not the whole index.
    # Admission is PARTITION-RELATIVE (top-N with a light floor vs the
    # partition's OWN best), NOT vs the global best (which would re-exclude the
    # low-global-relevance target). We only collect the ids to ADMIT; there is
    # no authority score term (inject-only — the unchanged stage-2 fusion
    # decides). Reuses the query vector (no extra embed); the lang/card-exempt
    # `qfilter` is preserved so admission respects the same language scoping as
    # the main legs.
    a_filter = None
    if _AUTHLEG:
        a_must = [qm.FieldCondition(key="trust_tier",
                                    range=qm.Range(gte=_AUTHLEG_THRESH))]
        if qfilter is not None:
            a_must.append(qfilter)
        a_filter = qm.Filter(must=a_must)
    # LEG DISPATCH (task 342). The stage-1 searches are mutually INDEPENDENT —
    # dense, bm25, (lexical), and the authority partition all read the same
    # collection and none consumes another's output — but they were issued as
    # separate, strictly SEQUENTIAL HTTP round trips, so the block cost their
    # SUM. On a 132 GB on-disk index with ~17 GB of page cache that sum is
    # dominated by cold random NVMe reads (measured in situ, 10 novel queries:
    # dense p50 1.27 s, bm25 0.13 s, authority-partition 0.51 s — the last one
    # is a SECOND full HNSW dense search, budgeted at "+3.8 ms" in task 201 on
    # the 44M/in-cache index and now three orders of magnitude more expensive:
    # rule-vs-data drift, not a code change). AW_BATCH_LEGS=1 issues exactly the
    # same requests as ONE query_batch_points, which qdrant services with a
    # shared search runtime, so the block costs ~max() instead of sum().
    #   Result-identity is the gate, not the intent: sequential vs batched
    #   returns bit-identical (id, score) lists for every leg — verified on 20
    #   queries x 4 filter/payload shapes (task 342 ident342.out).
    # Flag OFF (default) issues the identical calls in the identical order.
    # NOTE: batching makes AW_QDRANT_TIMEOUT one budget for the whole block
    # instead of one per leg. Measured block p90 3.5 s / max 3.9 s against a
    # 20 s budget, but a deep-cold box is where that margin gets spent.
    _specs = [("dense", dense, DENSE_VEC, candidates, qfilter, wp, _DENSE_SP),
              ("bm25", bm25, SPARSE_BM25, candidates, qfilter, wp, None)]
    if target_bm25_filter is not None and target_bm25_quota:
        _specs.append(("target_bm25", bm25, SPARSE_BM25, target_bm25_quota,
                       target_bm25_filter, wp, None))
    if translated_bm is not None:
        _specs.append(("translated_bm25", translated_bm, SPARSE_BM25,
                       translated_bm25_quota, target_bm25_filter, wp, None))
    if lexical:
        lex = qm.SparseVector(indices=[int(i) for i in lexical.keys()],
                              values=[float(v) for v in lexical.values()])
        _specs.append(("lexical", lex, SPARSE_LEXICAL, candidates, qfilter,
                       wp, None))
    # task 2665: with a filter set and the knob on, the authority leg leaves the batch and runs
    # beside it under its own server-side timeout; unfiltered requests never take this branch.
    auth_future = None
    if a_filter is not None and qfilter is not None and _AUTHLEG_FILTERED_TIMEOUT > 0:
        auth_future = _authleg_executor().submit(_bounded_authority_leg, dense, a_filter)
    elif a_filter is not None:
        _specs.append((_AUTH_LEG_KEY, dense, DENSE_VEC, _AUTHLEG_N, a_filter,
                       False, _DENSE_SP))
    if query_plan.country_legs_enabled:
        country_filter = _country_partition_filter(qfilter, query_plan.country)
        if query_plan.country_dense_limit:
            _specs.append(("country_dense", dense, DENSE_VEC,
                           query_plan.country_dense_limit, country_filter,
                           wp, _DENSE_SP))
        if query_plan.country_bm25_limit:
            _specs.append(("country_bm25", bm25, SPARSE_BM25,
                           query_plan.country_bm25_limit, country_filter,
                           wp, None))
    _stage1_started = time.perf_counter() if _QHEAD_COUNTRY_TRACE else None
    if _BATCH_LEGS:
        _pts = [r.points for r in qdrant_io.client().query_batch_points(
            collection_name=COLLECTION,
            requests=[qm.QueryRequest(query=v, using=u, limit=lim, filter=f,
                                      params=sp, with_payload=p)
                      for _, v, u, lim, f, p, sp in _specs],
            **_stage1_kw())]
    else:
        _pts = [_leg(v, u, lim, f, sp, with_payload=p)
                for _, v, u, lim, f, p, sp in _specs]
    _stage1_request_ms = ((time.perf_counter() - _stage1_started) * 1000.0
                          if _stage1_started is not None else None)
    legs = {s[0]: pts for s, pts in zip(_specs, _pts)
            if s[0] != _AUTH_LEG_KEY}
    if experimental:
        _LAST_EXPERIMENT_TRACE.update({
            "q": q, "lang": lang, "target_bm25_language": target_bm25_language,
            "target_bm25_weight": target_bm25_weight,
            "translated_bm25_query": translated_bm25_query,
            "translated_bm25_weight": translated_bm25_weight,
            "original_bm25_tokens": sorted((int(i), float(v)) for i, v in bm.items()),
            "translated_bm25_tokens": (sorted((int(i), float(v)) for i, v in translated_tokens.items())
                                      if translated_bm is not None else None),
            "stage1_legs": {name: [{"id": str(p.id), "score": float(p.score)} for p in points]
                            for name, points in legs.items()},
        })

    # An ORDERED set (dict keys) in the leg's own ANN order, never a Python set:
    # these ids are appended to the stage-2 pool in this order, the pool order
    # becomes `rows` order (Qdrant `retrieve` answers in request order), and
    # `rows` order decided which of two tied entity cards got the card boost
    # (review 779 M1; a set of string ids iterates in hash-salt order).
    auth_admit: dict = {}
    auth_leg_status = None
    if auth_future is not None:
        a_pts, auth_leg_status = auth_future.result()
        if auth_leg_status is not None:
            AUTHLEG_DEGRADED.set(auth_leg_status)
        auth_admit = _auth_admit(a_pts, _AUTHLEG_BETA)
    elif a_filter is not None:
        # Country partition legs may follow the authority leg in the shared
        # batch.  Resolve by its reserved key, never by positional convention.
        a_pts = next((pts for spec, pts in zip(_specs, _pts)
                      if spec[0] == _AUTH_LEG_KEY), [])
        auth_admit = _auth_admit(a_pts, _AUTHLEG_BETA)

    # per-token bm25 sub-legs (task 083): ONE batched call, candidate-gen only.
    # Same filter as the main legs; payload selector mirrors wp so country
    # stage-1 boosting still sees sub-leg candidates. Single-token queries and
    # the lexical/main legs are untouched.
    tok_legs = []
    _pertoken_batch_calls = 0
    if _PER_TOKEN and _TWO_STAGE and 2 <= len(bm) <= _PER_TOKEN_MAXQ:
        reqs = [qm.QueryRequest(query=qm.SparseVector(indices=[i], values=[1.0]),
                                using=SPARSE_BM25, limit=_PER_TOKEN_LIMIT,
                                filter=qfilter, with_payload=wp)
                for i in bm]
        tok_legs = qdrant_io.client().query_batch_points(collection_name=COLLECTION,
                                               requests=reqs, **_stage1_kw())
        _pertoken_batch_calls = 1
    return LegResults(bm=bm, qfilter=qfilter, wp=wp, soft_scoped=soft_scoped, specs=_specs, legs=legs,
                      auth_admit=auth_admit, tok_legs=tok_legs,
                      pertoken_batch_calls=_pertoken_batch_calls, stage1_request_ms=_stage1_request_ms,
                      tld_hashes=host_query_tld_hashes(q, lang),
                      auth_split=auth_future is not None, auth_leg_status=auth_leg_status)


def assemble_pool(res: LegResults, *, scoped, q_cantons, q_country, q_country_soft, scope_conf,
                  base_off_country, w_canton, w_foreign, query_plan) -> Pool | None:
    """Stage 1, step 2: raw and normalized score maps; None when no leg returned a candidate."""
    bm, wp, soft_scoped, legs, auth_admit, tok_legs = (
        res.bm, res.wp, res.soft_scoped, res.legs, res.auth_admit, res.tok_legs)
    _specs, _pertoken_batch_calls, _stage1_request_ms = (
        res.specs, res.pertoken_batch_calls, res.stage1_request_ms)
    # assemble candidates + raw per-leg scores, then normalize per leg per query
    payloads, raw = {}, {leg: {} for leg in legs}
    # Retain country-leg provenance separately from the modality score maps so
    # Task 374 can prove both global-leg identity and country-leg rescues.
    country_leg_provenance = {}
    cty1 = {}   # stage-1 country per id (two-stage country/scoped queries only)
    scope1 = {}  # stage-1 scope-penalty term per id (task 123, scoped only)

    _stage1_meta = functools.partial(stage1_meta, cty1=cty1, scope1=scope1, scoped=scoped, q_cantons=q_cantons, q_country=q_country, w_canton=w_canton, w_foreign=w_foreign, soft_scoped=soft_scoped, q_country_soft=q_country_soft, scope_conf=scope_conf, base_off_country=base_off_country)

    for leg, pts in legs.items():
        for p in pts:
            if wp is True:
                payloads[p.id] = p.payload
            elif wp and p.payload:  # small-selector path (country/scoped)
                _stage1_meta(p.id, p.payload)
            raw[leg][p.id] = p.score
            if leg in {"country_dense", "country_bm25"}:
                country_leg_provenance.setdefault(p.id, []).append(leg)
    # Dense cosine and stored BM25 scores use the identical query/vector and
    # collection statistics under a Qdrant payload filter, so they are directly
    # comparable.  Merge raw scores BEFORE one normalisation per modality; never
    # max separately min-max-normalized partition ranks (which would smuggle an
    # uncalibrated country boost into fusion).  Keep immutable global copies for
    # trace evidence below.
    global_raw = {name: dict(raw[name]) for name in ("dense", "bm25")}
    for modality in ("dense", "bm25"):
        for pid, score in raw.get(f"country_{modality}", {}).items():
            raw[modality][pid] = max(raw[modality].get(pid, float("-inf")), score)
    if _QHEAD_COUNTRY_TRACE:
        _trace_points = functools.partial(trace_points, legs=legs)

        global_ids = set(global_raw["dense"]) | set(global_raw["bm25"])
        country_ids = (set(raw.get("country_dense", {}))
                       | set(raw.get("country_bm25", {})))
        _LAST_QUERY_PLAN_TRACE.update({
            "plan": {
                "country": query_plan.country,
                "confidence": round(query_plan.confidence, 8),
                "source": query_plan.source,
                "country_dense_limit": query_plan.country_dense_limit,
                "country_bm25_limit": query_plan.country_bm25_limit,
            },
            # Gate compares these two legs verbatim across arms.  The country
            # lists are separate so a rescue is auditable, never an inferred
            # side-effect of final rank.
            "legs": {name: _trace_points(name) for name in (
                "dense", "bm25", "country_dense", "country_bm25")},
            "global_candidate_count": len(global_ids),
            "country_candidate_count": len(country_ids),
            "country_rescue_ids": sorted(str(pid) for pid in country_ids - global_ids),
            # `main` is the batch that must contain global+country legs.  The
            # total is updated below for pre-existing per-token/coverage calls;
            # do not misreport those as part of country-routing latency.
            "qdrant_main_batch_calls": 1 if _BATCH_LEGS else 0,
            "qdrant_pertoken_batch_calls": _pertoken_batch_calls,
            "qdrant_coverage_batch_calls": 0,
            "qdrant_total_batch_calls": ((1 if _BATCH_LEGS else 0)
                                          + _pertoken_batch_calls),
            "qdrant_stage1_calls": (1 if _BATCH_LEGS else len(_specs)) + int(res.auth_split),
            "stage1_request_ms": round(_stage1_request_ms, 4),
            # Filled after payload retrieval/final scoring.  Kept present on
            # empty legs so a gate cannot mistake missing instrumentation for a
            # clean no-rescue result.
            "candidate_urls": {},
            "final_ranked": [],
        })
    ids = set().union(*(raw[leg].keys() for leg in legs))
    if _AUTHLEG and auth_admit:  # CHANGE A: widen the pool with the authority partition
        ids.update(auth_admit)
    if not ids:
        return None
    # Country legs have already been merged into their global modality's raw
    # score map.  Do not normalize a second partition-relative score space.
    norm = {leg: _minmax(raw[leg],
                         drop_nonpositive=(_BM25_ZERO_GUARD
                                           and leg in _SPARSE_SCORE_LEGS))
            for leg in legs if leg not in {"country_dense", "country_bm25"}}
    # task 246: best RAW dense cosine over the pool — the semantic-support signal
    # for the zero-signal df-guard (a strong dense match spares a query whose
    # lexical terms happen to be absent). Raw (pre-minmax) so it is an absolute
    # similarity, not a within-pool rank.
    best_dense_raw = max(raw.get("dense", {}).values(), default=0.0)

    # merge per-token sub-legs into the bm25 leg norm (max, damped by 0.9) and
    # into the candidate id set — additive only, never lowers an existing norm.
    rare_toks = []  # bm25 token hashes with df < limit (window not full)
    df0_toks = 0    # task 246: query tokens ABSENT from the index (df=0)
    for tok_i, resp in zip(bm, tok_legs):
        sub = {}
        for p in resp.points:
            sub[p.id] = p.score
            if wp is not True and wp and p.payload:  # selector parity with main legs
                _stage1_meta(p.id, p.payload)
        if not sub:
            # df=0: this token is genuinely ABSENT from the index (empty sub-leg).
            # A distinctive term with no matches is the zero-signal precondition
            # (task 246); a common near-stopword token below is NOT counted here.
            if tok_i in res.tld_hashes:   # host-shaped query's TLD (task 2050); empty set when flag off
                continue
            df0_toks += 1
            continue
        if len(sub) >= _PER_TOKEN_LIMIT:
            # common token (full window): the main bm25 leg already ranks the
            # multi-token sum for these; merging the single-token flood gave
            # "wie"/"tim"-stuffed pages bm25_norm 0.9 ABOVE genuine
            # multi-token main-leg matches (measured -0.11 MRR on the 29q
            # eval). Only rare tokens need the rescue — skip.
            continue
        # ZERO-GUARD (task 336, fault found in task 335 §3). The normalizer
        # below divides by min(sub), so a single 0.0-score point makes it
        # raise ZeroDivisionError and the whole query 500s. Such points are
        # real: a *filtered* sparse query against a **plain (unindexed)
        # segment** returns filter-matching points that do NOT contain the
        # token, scored 0.0 — and any collection that has taken a small live
        # write (delta-promote!) carries such a segment until it is optimized.
        # A 90-point upsert on aw_b1 broke 4 of 663 gate golds.
        #
        # Fix = drop the non-positive points instead of dividing by them. On an
        # INDEXED collection every point in a sparse window genuinely contains
        # the token and scores > 0 (qdrant's idf modifier is strictly positive,
        # and these are rare tokens), so `pos` == `sub.values()` and both `mn`
        # and `subn` are bit-for-bit what they were before this guard: INERT.
        # Only the plain-segment leak is changed, and it is changed to what the
        # indexed collection would have returned (those points simply are not
        # in the window there — so they must not enter `bnorm`/`ids` either).
        pos = [s for s in sub.values() if s > 0.0]
        if not pos:
            # Every point scored 0.0 => the token is absent from the index and
            # we are only seeing plain-segment filter matches. That is exactly
            # the `not sub` case above (df=0) as an indexed collection would
            # report it, so classify it the same way for the task-246 guard.
            df0_toks += 1
            continue
        rare_toks.append(tok_i)
        # rare token (df < limit): saturated tf, no minmax (see flag note)
        mn = min(pos)
        subn = {pid: (s / mn) / (s / mn + _PER_TOKEN_K1)
                for pid, s in sub.items() if s > 0.0}
        bnorm = norm.setdefault("bm25", {})
        for pid, v in subn.items():
            bnorm[pid] = max(bnorm.get(pid, 0.0), _PER_TOKEN_SUB_W * v)
            ids.add(pid)

    # task 246: ZERO-SIGNAL df-guard classification (see _RELFLOOR_ZS_K block).
    # Gated on AW_RELFLOOR so flag-off is byte-identical (zero_signal stays False,
    # eff_k stays k). Requires the per-token sub-legs to have run (df known), >=1
    # ABSENT distinctive term (df=0), NO present distinctive term (rare_toks
    # empty), AND no strong dense/semantic match — i.e. neither a lexical nor a
    # semantic handle exists, so the whole result set is a stopword coincidence.
    zero_signal = bool(
        _RELFLOOR and tok_legs
        and df0_toks > 0
        and not rare_toks
        and best_dense_raw < _RELFLOOR_DENSE_MAX)
    return Pool(payloads=payloads, raw=raw, norm=norm, ids=ids, cty1=cty1, scope1=scope1,
                global_raw=global_raw, country_leg_provenance=country_leg_provenance,
                best_dense_raw=best_dense_raw, rare_toks=rare_toks, zero_signal=zero_signal)


def cut_pool(pool: Pool, *, auth_admit, country, k, k_dense_scale, k_bm25_scale,
             target_bm25_weight, translated_bm25_weight) -> dict:
    """Stage 1, step 3: the margin cut and the one payload fetch; returns the pool's payloads."""
    payloads, ids, norm, cty1, scope1 = pool.payloads, pool.ids, pool.norm, pool.cty1, pool.scope1
    if _TWO_STAGE:
        # First pass: fuse on leg scores ONLY, keep a margin of top ids, then
        # ONE batched payload fetch. FLOORED at 200 (task 199 k-stability):
        # stage-2 signals (trust/title/exact) are NOT small for authoritative
        # pages (trust 2.5*tier + title 2.5*centrality can add >4 to an s1rel
        # of 0.04) — a margin of 32 cut the eventual #1 out of the pool at k=8.
        margin = fuse_margin(k)

        _s1rel = functools.partial(s1_rel, norm=norm, k_dense_scale=k_dense_scale, k_bm25_scale=k_bm25_scale, target_bm25_weight=target_bm25_weight, translated_bm25_weight=translated_bm25_weight)

        # task 164: gate the stage-1 country boost by leg-relevance so a weak
        # domestic candidate cannot flood the margin pool and evict the strongly-
        # relevant foreign/EU authority (the k=8 BCG failure was pool eviction:
        # eur-lex fell below margin-32 once ~30 FR pages each took +0.6).
        _best_s1 = max((_s1rel(pid) for pid in ids), default=0.0)

        _s1boost = functools.partial(s1_boost, country=country, cty1=cty1, s1_rel=_s1rel, best_s1=_best_s1)

        _s1key = functools.partial(s1_key, s1_rel=_s1rel, s1_boost=_s1boost, scope1=scope1)

        if not _PAGECAP:
            pre = _order_ids(ids, _s1key)[:margin]
        else:
            # task 582 -- page identity before the margin cut. ONE extra
            # batched retrieve, `url` only; a pid whose payload does not come
            # back is its OWN page (it is never silently merged into another
            # one, and never silently capped away).
            _ordered = _order_ids(ids, _s1key)
            _pg = {}
            try:
                for _r in qdrant_io.client().retrieve(collection_name=COLLECTION,
                                            ids=_ordered, with_payload=["url"]):
                    _u = (_r.payload or {}).get("url")
                    if _u:                   # NOT `_canonical_key(url or "")`:
                        _pg[_r.id] = _canonical_key(_u)   # that maps every
                    # payload-less row to ONE key and caps them all away as if
                    # they were the same page (caught by tests/test_serve_pagecap.py
                    # `test_a_missing_payload_is_its_own_page_never_merged`).
            except Exception:                # an error is not an empty:
                _pg = {}                     # fail OPEN to the shipped cut
            _page = functools.partial(page_of, pg=_pg)

            if _PAGECAP_MODE == "agg":
                # max-over-page: a page's stage-1 relevance is its best row's;
                # pages enter in that order, each page's rows best-first.
                _by_page: dict = {}
                for pid in _ordered:         # already best-first
                    _by_page.setdefault(_page(pid), []).append(pid)
                pre = []
                for _rows in _by_page.values():   # dicts keep insertion order,
                    pre.extend(_rows[:_PAGECAP])  # which IS page-best order
                    if len(pre) >= margin:
                        break
                pre = pre[:margin]
            else:
                _seen: dict = {}
                pre = []
                for pid in _ordered:
                    key_ = _page(pid)
                    n = _seen.get(key_, 0)
                    if n >= _PAGECAP:
                        continue
                    _seen[key_] = n + 1
                    pre.append(pid)
                    if len(pre) >= margin:
                        break
        # CHANGE A (task 202): force-include admitted authority candidates into
        # the stage-2 pool even if their leg-relevance sits below the margin cut
        # — that IS the point: the authority page's stage-2 trust/title/exact is
        # what lifts it (its dense/bm25 leg-relevance is by construction weak).
        if _AUTHLEG and auth_admit:
            pre = _extend_pool(pre, auth_admit)
        payloads = {r.id: r.payload
                    for r in qdrant_io.client().retrieve(collection_name=COLLECTION,
                                               ids=pre, with_payload=True)}
    return payloads


def stage1(req: Stage1Request) -> Stage1 | None:
    """The whole of stage 1; None means no candidates (search returns [])."""
    res = run_legs(
        q=req.q, lang=req.lang, dense=req.dense, lexical=req.lexical, candidates=req.candidates,
        country=req.country, query_plan=req.query_plan, scoped=req.scoped,
        q_country_soft=req.q_country_soft, experimental=req.experimental,
        target_bm25_language=req.target_bm25_language, target_bm25_quota=req.target_bm25_quota,
        target_bm25_weight=req.target_bm25_weight, translated_bm25_query=req.translated_bm25_query,
        translated_bm25_quota=req.translated_bm25_quota,
        translated_bm25_weight=req.translated_bm25_weight, op_filter=req.op_filter)
    pool = assemble_pool(
        res, scoped=req.scoped, q_cantons=req.q_cantons, q_country=req.q_country,
        q_country_soft=req.q_country_soft, scope_conf=req.scope_conf,
        base_off_country=req.base_off_country, w_canton=req.w_canton, w_foreign=req.w_foreign,
        query_plan=req.query_plan)
    if pool is None:
        return None
    payloads = cut_pool(
        pool, auth_admit=res.auth_admit, country=req.country, k=req.k,
        k_dense_scale=req.k_dense_scale, k_bm25_scale=req.k_bm25_scale,
        target_bm25_weight=req.target_bm25_weight,
        translated_bm25_weight=req.translated_bm25_weight)
    return Stage1(bm=res.bm, qfilter=res.qfilter, payloads=payloads, raw=pool.raw, norm=pool.norm,
                  global_raw=pool.global_raw, country_leg_provenance=pool.country_leg_provenance,
                  best_dense_raw=pool.best_dense_raw, rare_toks=pool.rare_toks,
                  zero_signal=pool.zero_signal)
