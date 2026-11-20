"""RANKHEAD V2 (task 350, plan 1a-rank part iii) — the DOC-SIDE feature head.

Why this exists (the 333 architecture finding): the ADR-036 rankhead is
QUERY-side-only — a linear probe on the frozen 1024-d arctic query embedding
emitting one scalar (the trust-relgate beta). It accepts no doc-side features,
so no amount of labels can teach it fedlex-over-cantonal: every cantonal
displacer carries the same trust_tier 1.0 / source_type "official" as the
federal gold, and the only discriminating signals (v9 authority, canton stamp,
country-vs-scope, recency) live on the CANDIDATE. V2 closes that gap:

    delta(q, d) = cap * tanh( (W x_q + b) . f(d) / cap )

  x_q  = the frozen 1024-d arctic query vector (already computed for the
         dense leg — one F x 1024 matmul per query, no extra embed)
  f(d) = a small per-candidate feature vector (payload classes + query-
         conditioned scalars, see FEATURES below)
  W, b = trained (the rank-head trainer) pairwise on the task-350
         family set + the 332 adjudication labels
  cap  = hard bound (head file) — tanh keeps the adjustment smooth and an
         OOD query cannot emit a pathological delta.

The delta is a bounded ADDITIVE stage-2 adjustment (positive or negative),
added to the penalties bucket AFTER every existing lever (authority boost,
hub-floor, entity-card, official_pref) so those levers see unchanged inputs.
In additive trust mode final = base_fused + delta EXACTLY (the trust term
depends only on rel_core/best_rel_core, never on penalties) — which is what
makes offline training on dumped base scores faithful to serving.

ADR-080 lives here as a training objective: the country one-hots + scope_match
features let the query side learn "statute-name marker => jurisdiction" and
demote foreign OFFICIAL law (the class the blanket official-exemption made
un-demotable by construction, 332 §4.3). The freshness-intent objective (Tim
2026-07-28) rides the has_published/recency features: a "latest/current/
demnächst" query learns positive recency weight, trained on dated slices.

ONE TRUTH: this module is imported by BOTH serving (retrieval.py) and the
trainer/miner. Feature semantics change => retrain; the head file carries the
feature list and the loader refuses a mismatched contract (fail loud).

Flag-gated AW_RANKHEAD_V2 (default OFF, byte-identical); rollback =
AW_RANKHEAD_V2=0. Head file: rankhead_v2.json beside this module
(AW_RANKHEAD_V2_PATH overrides).
"""
from __future__ import annotations

import datetime as _dt
import math

# The feature contract. Order is load-bearing (indexes into W columns).
# Every feature is in [0, 1] except scope_match in [-1, 1]. A head file whose
# "features" list differs from this is REFUSED by load_head (fail loud, not a
# silent misalignment).
FEATURES = [
    "trust_tier",       # payload trust_tier (curated prior, 0..1)
    "authority",        # ADR-019 v9 table score for the host (earned, 0..1)
    "src_official",     # source_type one-hots (class-level — the head cannot
    "src_reference",    #   memorize URLs; identity is invisible by design)
    "src_docs",
    "src_registry",     # registry_record | gov-stat | gov-geo
    "src_entity_card",
    "src_community",    # community-qa | hn-discussion | blog
    "vendor_tier",      # 1.0 iff trust_tier <= 0.60 (POLICY v2 vendor class)
    "has_canton",       # cantonal stamp present (the cantonal-vs-federal axis)
    "cty_ch", "cty_de", "cty_fr", "cty_ca", "cty_eu",  # country one-hots
    "cty_other", "cty_none",
    "scope_match",      # +conf when candidate country == effective scope,
                        # -conf when it conflicts, 0 unknown. Applies to
                        # official too (ADR-080 — the exemption is retired
                        # inside this head's lane).
    "has_published",    # a parseable `published` date exists
    "recency_pub",      # exp(-age_days/365) of `published` (0 if absent)
    "recency_seen",     # exp(-age_days/365) of `last_seen` (crawl date)
    "title_centrality", # query-terms-in-title distinctiveness (task 104 impl)
    "rel_ratio",        # rel_core / pool-best rel_core (pool-relative)
    "dense_norm",       # min-max dense leg score (0 = not in dense pool)
    # Retrieval evidence is derived exclusively from the already-returned dense
    # and sparse vectors.  It deliberately has no snippet/body/payload-text
    # feature: snippets are presentation pointers, not ranking evidence.
    "dense_raw",        # raw dense cosine, bounded to the valid [0,1] range
    "dense_rank",       # reciprocal competition rank in the dense leg (0 absent)
    "bm25_raw",         # raw BM25 score / best positive BM25 score in this pool
    "bm25_norm",        # existing min-max BM25 contribution (0 absent)
    "bm25_rank",        # reciprocal competition rank in the BM25 leg (0 absent)
    "leg_agreement",    # min(dense_norm, bm25_norm): both legs independently agree
    "sparse_coverage",  # bounded full-vector rare-term coverage, 0 if unavailable
    "sparse_coverage_available",  # coverage oracle ran for this query
]
N_FEATURES = len(FEATURES)

_EU_SET = frozenset({"EU"})
_REGISTRY_TYPES = frozenset({"registry_record", "gov-stat", "gov-geo"})
_COMMUNITY_TYPES = frozenset({"community-qa", "hn-discussion", "blog"})


def _decay(datestr, today) -> float:
    """exp(-age/365) freshness, 0.0 when absent/unparseable (same shape as
    retrieval._recency; duplicated here to avoid a server->trainer import
    cycle — the formula is 3 lines and covered by test_rankhead_v2)."""
    if not datestr:
        return 0.0
    try:
        d = _dt.date.fromisoformat(str(datestr)[:10])
    except (ValueError, TypeError):
        return 0.0
    return math.exp(-max((today - d).days, 0) / 365.0)


def retrieval_features(*, dense_raw: float, dense_norm: float, dense_rank: float,
                       bm25_raw: float, bm25_norm: float, bm25_rank: float,
                       sparse_coverage: float, coverage_available: bool) -> list[float]:
    """Generic query×candidate evidence from vector scores only.

    The serving scorer and dump/miner share this contract.  ``bm25_raw`` is
    made query-comparable as a ratio to the best positive BM25 score in the
    already-admitted pool; it is intentionally distinct from min-max score and
    rank.  Coverage is term *presence* in the stored sparse vector, never text
    retained or returned from the document.
    """
    dn = max(0.0, min(1.0, float(dense_norm)))
    bn = max(0.0, min(1.0, float(bm25_norm)))
    return [
        max(0.0, min(1.0, float(dense_raw))),
        max(0.0, min(1.0, float(dense_rank))),
        max(0.0, min(1.0, float(bm25_raw))),
        bn,
        max(0.0, min(1.0, float(bm25_rank))),
        min(dn, bn),
        max(0.0, min(1.0, float(sparse_coverage))),
        1.0 if coverage_available else 0.0,
    ]


def doc_features(pl, *, authority: float, country: str | None, eu_supra: bool,
                 scope: str | None, scope_conf: float,
                 title_centrality: float, rel_ratio: float, dense_norm: float,
                 dense_raw: float = 0.0, dense_rank: float = 0.0,
                 bm25_raw: float = 0.0, bm25_norm: float = 0.0,
                 bm25_rank: float = 0.0, sparse_coverage: float = 0.0,
                 coverage_available: bool = False,
                 today: _dt.date | None = None) -> list[float]:
    """Build f(d) for one candidate. `country` is retrieval._effective_country
    (payload stamp else AW_COUNTRY_MAP); `scope` is the query's effective
    jurisdiction (deterministic q_country, else the qhead soft scope) with its
    confidence; the three query-conditioned scalars are computed by the caller
    from the same values the rest of stage-2 uses (one truth)."""
    if today is None:
        today = _dt.date.today()
    st = pl.get("source_type") or ""
    trust = float(pl.get("trust_tier", 0.6))
    c = (country or "").upper() or None
    f = [0.0] * N_FEATURES
    f[0] = trust
    f[1] = float(authority)
    f[2] = 1.0 if st == "official" else 0.0
    f[3] = 1.0 if st == "reference" else 0.0
    f[4] = 1.0 if st == "docs" else 0.0
    f[5] = 1.0 if st in _REGISTRY_TYPES else 0.0
    f[6] = 1.0 if st == "entity_card" else 0.0
    f[7] = 1.0 if st in _COMMUNITY_TYPES else 0.0
    f[8] = 1.0 if trust <= 0.60 else 0.0
    f[9] = 1.0 if pl.get("canton") else 0.0
    if eu_supra or c in _EU_SET:
        f[14] = 1.0
    elif c == "CH":
        f[10] = 1.0
    elif c == "DE":
        f[11] = 1.0
    elif c == "FR":
        f[12] = 1.0
    elif c == "CA":
        f[13] = 1.0
    elif c is None:
        f[16] = 1.0
    else:
        f[15] = 1.0
    if scope and c:
        # EU-supra counts as in-scope for EU-member scopes (mirrors the
        # _is_eu_supra exemption in the off-country lever, task 164).
        in_scope = (c == scope) or (f[14] == 1.0 and scope in
                                    ("CH", "DE", "FR", "IT", "AT", "BE", "EU"))
        f[17] = scope_conf if in_scope else -scope_conf
    pub = pl.get("published")
    f[18] = 1.0 if _decay(pub, today) > 0.0 or pub else 0.0
    f[19] = _decay(pub, today)
    f[20] = _decay(pl.get("last_seen"), today)
    f[21] = max(0.0, min(1.0, float(title_centrality)))
    f[22] = max(0.0, min(1.0, float(rel_ratio)))
    f[23] = max(0.0, min(1.0, float(dense_norm)))
    f[24:] = retrieval_features(
        dense_raw=dense_raw, dense_norm=dense_norm, dense_rank=dense_rank,
        bm25_raw=bm25_raw, bm25_norm=bm25_norm, bm25_rank=bm25_rank,
        sparse_coverage=sparse_coverage,
        coverage_available=coverage_available)
    return f


def load_head(path):
    """Load + validate a head file. Returns None on any failure — the CALLER
    logs and serves without the head (same degrade contract as the v1 loader:
    a broken head file must not 500 the search path, but it must be VISIBLE)."""
    import json
    with open(path) as fh:
        h = json.load(fh)
    if h.get("features") != FEATURES:
        raise ValueError(
            f"rankhead_v2 feature contract mismatch: head file has "
            f"{h.get('features')}, serving expects {FEATURES}")
    W = h["W"]
    if len(W) != N_FEATURES or any(len(row) != len(W[0]) for row in W):
        raise ValueError("rankhead_v2 W shape invalid")
    if len(h["b"]) != N_FEATURES:
        raise ValueError("rankhead_v2 b shape invalid")
    cap = float(h["cap"])
    if not (0.0 < cap <= 4.0):
        raise ValueError(f"rankhead_v2 cap {cap} outside sane bound (0,4]")
    return {"W": [[float(v) for v in row] for row in W],
            "b": [float(v) for v in h["b"]], "cap": cap,
            "dim": len(W[0]), "_meta": h.get("_comment", "")}


def query_weights(head, qvec) -> list[float]:
    """w(q) = W x_q + b — once per query (F x dim matmul, pure python-free
    via numpy when available; the serving path always has numpy)."""
    import numpy as np
    x = np.asarray(qvec, dtype="float32")[: head["dim"]]
    W = np.asarray(head["W"], dtype="float32")
    b = np.asarray(head["b"], dtype="float32")
    return (W @ x + b).tolist()


def delta(head, qw, feats) -> float:
    """Bounded per-candidate adjustment. cap*tanh(raw/cap): smooth, |d|<cap."""
    cap = head["cap"]
    raw = 0.0
    for w, f in zip(qw, feats):
        if f:
            raw += w * f
    return cap * math.tanh(raw / cap)
