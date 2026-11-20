"""S10 step 6 (task 1104): the query plan: pool sizes and the country-partition legs.

`candidate_pool` / `fuse_margin` (one home for the k-stability floors, task 199) and the typed
`QueryPlan` (ADR-084 / task 369) moved out of `retrieval.py` unchanged. Pure arithmetic over frozen
constants from `settings` and `heads`; no client, no state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from qdrant_client import models as qm

from .heads import _QHEAD_COUNTRY_LEGS
from .settings import SETTINGS as _S

_FUSE_MARGIN = 0  # fixed: was env AW_FUSE_MARGIN (S10 D5: no product setter); 0 => max(_MARGIN_FLOOR, 4*k)

# --------------------------------------------------------------------------- #
#  k-stability floors (task 199). Both pool sizes used to scale with k
#  (candidates = max(80, 2k); margin = max(4k, 32)), so the RESULT SET was a
#  function of the k the caller asked for: at k=8/10 (the MCP/API default!) the
#  legs fetched 80 and the fusion pool was cut at 32/40, while at k=50 the legs
#  fetched 100 and the pool was 200. A candidate with modest leg relevance but
#  strong stage-2 signals (trust + title-centrality + exact) could therefore be
#  the #1 result at k=50 yet ABSENT ENTIRELY at k=10 — measured live 2026-07-13:
#  "enterprise AI risk management playbook" ranked nist.gov/itl/
#  ai-risk-management-framework #1 (fused 4.07) at k=50; at k<=40 the page was
#  in NO leg at all (dense-leg rank ~81-146 > 80), and even inside the legs its
#  stage-1 leg-fused rank (146) was far outside margin 32/40. This class
#  invisibly degrades every small-k consumer.
#
#  Fix: FLOOR both sizes at the k=50 values — candidates >= 100, margin >= 200 —
#  so every k <= 50 runs the IDENTICAL retrieval + fusion pipeline and the
#  returned top-k is a strict PREFIX of the top-50 ordering (k-stability by
#  construction; k > 50 still scales up). Floor choice is evidence-based (twin
#  aw_v4_sat, 26q x 3 warm): qdrant-side p50 25.9 -> 35.9 ms ((80,32) ->
#  (100,200)), ~+7% of the ~150 ms end-to-end warm p50 — within budget; the
#  next step up (150,300) costs +18 ms (~+12%) — rejected. The growing term is
#  the stage-2 batched retrieve() (3.4 -> 10.5 ms p50), still ONE call (072).
#  Rollback: AW_KSTAB=0 restores the exact legacy formulas; the floors are also
#  individually tunable (AW_CAND_FLOOR / AW_MARGIN_FLOOR).
# --------------------------------------------------------------------------- #
_KSTAB = _S.legs.kstab
_CAND_FLOOR = 100  # fixed: was env AW_CAND_FLOOR (S10 D5: no product setter)
_MARGIN_FLOOR = 200  # fixed: was env AW_MARGIN_FLOOR (S10 D5: no product setter)


# --------------------------------------------------------------------------- #
# QUERY-PLAN COUNTRY PARTITIONS (follow-up to ADR-084 / task 369).
#
# The calibrated country classifier is query-side evidence, not a document
# scorer.  When this experiment is enabled, translate that evidence once into a
# typed plan and add two *candidate-admission* legs: dense and BM25 within the
# predicted (or explicitly requested) country.  Global dense + BM25 are never
# removed; country raw scores are merged into their matching modality BEFORE its
# one normalisation (never as an independent fusion weight), so a country leg can
# recover a domestic candidate that global retrieval missed without making
# foreign/INTL material ineligible.  All legs remain entries in the existing single
# `query_batch_points` request when AW_BATCH_LEGS=1.
#
# The candidate qhead artifact exposes calibrated country confidence plus typed
# routing heads.  The learned quota outputs cap the two small retrieval legs. A
# future artifact may add more QueryPlan outputs without changing the dispatch
# contract below.  Default OFF is byte-identical.
# --------------------------------------------------------------------------- #
_COUNTRY_LEG_MAX_LIMIT = 100


_COUNTRY_LEG_DENSE_LIMIT = 32  # fixed: was env AW_QHEAD_COUNTRY_DENSE_LIMIT (S10 D5: no product setter)
_COUNTRY_LEG_BM25_LIMIT = 32  # fixed: was env AW_QHEAD_COUNTRY_BM25_LIMIT (S10 D5: no product setter)
_COUNTRY_LEG_MIN_CONF = 0.60  # fixed: was env AW_QHEAD_COUNTRY_LEG_MIN_CONF (S10 D5: no product setter)


@dataclass(frozen=True)
class QueryPlan:
    """Typed, bounded retrieval controls emitted from query-side evidence.

    ``country`` is either an explicit agent instruction or the calibrated
    country-head prediction.  ``source`` is retained for gate/debug provenance;
    it never changes ranking.  A disabled/abstaining plan has zero quotas.
    """

    country: str | None = None
    confidence: float = 0.0
    source: str = "none"
    country_dense_limit: int = 0
    country_bm25_limit: int = 0

    @property
    def country_legs_enabled(self) -> bool:
        return bool(self.country and (self.country_dense_limit
                                      or self.country_bm25_limit))


def _country_query_plan(explicit_country: str | None,
                        head_country: str | None,
                        head_confidence: float,
                        candidates: int, *, enabled: bool = _QHEAD_COUNTRY_LEGS,
                        dense_limit: int = _COUNTRY_LEG_DENSE_LIMIT,
                        bm25_limit: int = _COUNTRY_LEG_BM25_LIMIT,
                        min_confidence: float = _COUNTRY_LEG_MIN_CONF,
                        jurisdiction_binding: float = 1.0,
                        foreign_authority_needed: float = 0.0,
                        multi_scope: float = 0.0,
                        jurisdiction_threshold: float = 0.5,
                        foreign_authority_threshold: float = 0.5,
                        multi_scope_threshold: float = 0.5,
                        dense_quota: float | None = None,
                        bm25_quota: float | None = None) -> QueryPlan:
    """Resolve country retrieval routing with explicit control taking precedence.

    The head's calibrated abstention threshold is enforced again here because
    retrieval widens the candidate pool, which is a separate decision from
    ranking.  Quotas are confidence-proportional and bounded by both operator
    limits and the caller's global candidate budget.
    """
    if not enabled:
        return QueryPlan()
    if candidates <= 0:
        raise ValueError(f"candidates must be positive, got {candidates}")
    thresholds = (jurisdiction_threshold, foreign_authority_threshold,
                  multi_scope_threshold)
    if any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError(f"action thresholds must be in [0, 1], got {thresholds}")
    def _quota(limit, value):
        # Scalar policy outputs are normalized fractions.  Clamp here as a
        # malformed model artifact must never widen a retrieval partition.  Zero
        # is a real learned action: it suppresses that modality rather than
        # becoming a one-result leg.
        scale = max(0.0, min(1.0, float(value)))
        if scale == 0.0:
            return 0
        return min(candidates, limit, math.ceil(limit * scale))

    if explicit_country:
        return QueryPlan(country=explicit_country, confidence=1.0,
                         source="explicit",
                         country_dense_limit=_quota(
                             dense_limit, 1.0 if dense_quota is None else dense_quota),
                         country_bm25_limit=_quota(
                             bm25_limit, 1.0 if bm25_quota is None else bm25_quota))
    confidence = max(0.0, min(1.0, float(head_confidence)))
    if (not head_country or confidence < min_confidence
            or jurisdiction_binding < jurisdiction_threshold
            or foreign_authority_needed >= foreign_authority_threshold
            or multi_scope >= multi_scope_threshold):
        return QueryPlan()
    # Confidence must never expand the configured partition budget.  ceil keeps
    # a just-cleared calibrated prediction useful while the 100-point hard cap
    # bounds latency and protects global-pool diversity.
    quota_scale = max(confidence, min_confidence)
    return QueryPlan(
        country=head_country, confidence=confidence, source="head",
        country_dense_limit=_quota(
            dense_limit, quota_scale if dense_quota is None else dense_quota),
        country_bm25_limit=_quota(
            bm25_limit, quota_scale if bm25_quota is None else bm25_quota),
    )


def _country_partition_filter(base_filter, country: str):
    """Intersect the normal query filter with the indexed country partition."""
    condition = qm.FieldCondition(key="country", match=qm.MatchValue(value=country))
    if base_filter is None:
        return qm.Filter(must=[condition])
    # Nested Qdrant filters preserve the normal language/card-exemption OR
    # exactly; flattening it would accidentally turn that `should` into AND.
    return qm.Filter(must=[base_filter, condition])



def candidate_pool(k: int) -> int:
    """Per-leg candidate limit for a given k. THE definition — search() calls
    it and the retired offline tuner imported it (task 387: that tuner had
    its own hardcoded 80, so prod's task-199 floor of 100 never reached the
    simulator and a query whose prod #1 sat at bm25 rank 81 was unreachable
    offline). Never re-derive this formula; import it."""
    return max(_CAND_FLOOR if _KSTAB else 80, 2 * k)


def fuse_margin(k: int) -> int:
    """Stage-1 pool cut for a given k — the ids whose payloads get fetched and
    re-scored in stage 2. Same one-home rule as candidate_pool() above (the
    tuner's selftest hardcoded max(4k,32) = 32 while prod ran 200)."""
    return _FUSE_MARGIN or (max(_MARGIN_FLOOR, 4 * k) if _KSTAB
                            else max(4 * k, 32))


def qk(name, default, *, qh):
    """Resolve a knob to the head value if it is driving it (spec §3
    precedence), else today's static default. Operator env pins and explicit
    args are applied at their own sites and still win (they are the defaults
    passed in here)."""
    return qh.get(name, default)
