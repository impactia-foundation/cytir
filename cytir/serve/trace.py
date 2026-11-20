"""S10 step 6 (task 1104): the debug instruments. The per-component ledger (task 583) lives here.

`_LAST_COMPONENTS` is mutated (`setdefault`, `clear`), never re-bound, so `retrieval._LAST_COMPONENTS`
is the same object. The query-plan and experiment traces and the candidate-id trace flag moved here in step 8a (task 1116); they are
mutated (`clear`, `update`), never re-bound, so `retrieval` holds the same objects. `_LAST_SCORE_DEBUG` moves with stage 2.
"""
from __future__ import annotations

from .settings import SETTINGS as _S

# task 583 — the PER-COMPONENT ledger under the same flag.  `rel_core` and
# `penalties` are buckets: a dozen named levers add into them and the sealed
# breakdown could only ever say "penalties = -1.4", never WHICH lever.  An
# aggregate cannot acquit a component (verification-doctrine 11), and the
# question this serves — what does production do differently, per gold, per
# term — is not answerable from the buckets.  Written ONLY when _DEBUG_SCORES
# is on (off in prod, off in every gate arm), so the shipped scoring path is
# byte-identical; `tests/test_serve_task583_components.py` pins that.
_LAST_COMPONENTS: dict = {}


def _dbgc(pid, name: str, value) -> None:
    """Record one named contribution to a scoring bucket. Additive: a lever
    that fires twice for one candidate accumulates, exactly as the bucket
    does."""
    v = float(value or 0.0)
    if v == 0.0:
        return
    d = _LAST_COMPONENTS.setdefault(pid, {})
    d[name] = d.get(name, 0.0) + v


# intentionally opt-in so normal serving allocates no candidate-id trace.
_QHEAD_COUNTRY_TRACE = _S.runtime.qhead_country_trace

_LAST_QUERY_PLAN_TRACE: dict = {}

# Task381 uses it to bind every extra sparse leg's raw candidate list to a
# fixed-arm artifact instead of reconstructing a different fusion externally.
_LAST_EXPERIMENT_TRACE: dict = {}


def trace_points(name, *, legs):
    # Preserve the exact Python float returned by Qdrant.  Task374
    # compares global legs across arms; display rounding here would
    # conceal small retrieval drift and falsely certify isolation.
    return [{"id": str(point.id), "score": float(point.score)}
            for point in legs.get(name, ())]


# --- S10 step 8b (task 1125): constants that travelled with their readers ---

# Diagnostic side-channel (task 289) — env-gated, default OFF, inert in prod. When
# AW_DEBUG_SCORES=1, search() records a per-candidate score breakdown (rel_core vs
# trust contribution vs penalties vs raw dense cosine) into _LAST_SCORE_DEBUG so an
# offline harness can inspect WHY a candidate ranked where it did. No effect on
# scoring or ordering; byte-identical when off.
_DEBUG_SCORES = _S.runtime.debug_scores

_LAST_SCORE_DEBUG: list = []
