"""The learned heads, split out of `retrieval.py` (S10 step 5, task 1099).

Four artifacts, each loaded once at import and failing safe to None (a bad head never breaks
search): the ADR-036 rankhead (v1/v3 relgate beta), the doc-side rankhead v2, the unified
query head (`qhead`, ADR-062) with its learned `country_scope` classifier and the candidate-only
`policy_heads` block, and the scope resolvers that sit on top of them. Bodies are verbatim from
`retrieval.py`, which imports every name back so `retrieval.<name>` still answers; the loaders
keep their `aw_server.retrieval` logger name so log lines do not change. `_LAST_V2_DUMP` lives
here and is only mutated, never rebound. This module must not import `retrieval`.
"""
from __future__ import annotations

import json
import math
import re

from . import settings as _settings
from .geo import _gazetteer_scope
from .rankhead_schema import validate_head
from .settings import SETTINGS

_S = SETTINGS   # typed config, loaded once at import (S10 step 2, task 1071)
_QHEAD_COUNTRY_LEGS = _S.legs.qhead_country_legs   # also read by retrieval's QueryPlan dispatch

# --------------------------------------------------------------------------- #
#  RANKING HEAD (task 205) — the LEARNED, intent-conditioned trust relgate.
#  CHANGE B (above) failed as a GLOBAL gate: the beta that moves the off-topic
#  high-trust flood ("startup management tips"/"march madness" → opensource.org,
#  canada.ca, iep.utm.edu/foucault riding trust 0.9-0.95) is the SAME beta that
#  drops legitimate modest-dense statutes out of the legal slice (202-B: +0/-4
#  at beta 0.5; the legal tripwire killed it). The distinguishing signal is
#  INTENT/TOPIC, which a pool-relative relevance gate cannot see. This head
#  supplies it: a small linear probe on the FROZEN full 1024-dim arctic query
#  embedding (already computed for the dense leg — a matmul, no extra embed)
#  predicts a PER-QUERY relgate beta. On AUTHORITY-LOOKUP intents (legal_statute,
#  statistic, entity_lookup, api_reference, current_info) the head emits beta≈0
#  → relgate OFF → statutes keep the full trust floor (tripwire held by
#  construction). On GENERAL intents (factual/definition/howto) it emits the
#  tuned general beta → the off-topic trust-floats are attenuated. The relgate
#  arithmetic is IDENTICAL to CHANGE B (attenuate only, never demote, penalties/
#  relevance untouched, stage-2 only → pool unchanged, no added latency); the
#  ONLY change is beta is per-query from the head instead of a global env.
#
#  Head file (AW_RANKHEAD_PATH, default rankhead.json beside this module):
#    {"variant":"v1|v2|v3", "beta_gen":<f>, "beta_max":<f>,
#     "w":[1024],"b":<f>,                # v1/v2 binary logit P(general/relgate)
#     "w_beta":[1024],"b_beta":<f>,      # v3 direct beta regressor (bounded)
#     "trust_scale":<f?>}                # optional global trust multiplier
#  v1: beta = beta_gen if sigmoid(w·x+b) >= 0.5 else 0   (discrete preset)
#  v2: beta = beta_gen * sigmoid(w·x+b)                  (soft interpolation)
#  v3: beta = clip(w_beta·x+b_beta, 0, beta_max)         (distilled regressor)
#  Bounded to [0, beta_max] in every variant (an OOD arena query cannot get a
#  pathological weight — the convex hull of the vetted {off, general} presets).
#  Flag-gated AW_RANKHEAD (default OFF, byte-identical); rollback = AW_RANKHEAD=0.
# --------------------------------------------------------------------------- #
_RANKHEAD = _S.query.rankhead
_RANKHEAD_PATH = _S.runtime.rankhead_path

# --------------------------------------------------------------------------- #
#  RANKHEAD V2 (task 350, plan 1a-rank iii) — the DOC-SIDE feature head.
#  The ADR-036 head above is query-side-only, so labels alone can never teach
#  fedlex-over-cantonal: displacer and gold carry identical trust/source_type
#  and the discriminating signals (v9 authority, canton stamp, country-vs-
#  scope, recency) are candidate-side (task 333 §0, 332 §5). V2 is a small
#  bilinear head — per-query feature weights w(q) = W·x_q + b off the SAME
#  frozen 1024-d query vector, dotted with a per-candidate feature vector
#  (cytir/serve/rankhead_v2.py = the one-truth contract), emitting a
#  BOUNDED additive stage-2 delta (cap·tanh). Composes with (does not replace)
#  the v1 relgate head; applied after every existing lever so authority/
#  hubfloor/entity_card see unchanged inputs. ADR-080 (inferred scope demotes
#  officialdom) and the freshness-intent objective (Tim 2026-07-28) are
#  TRAINING objectives riding the country/scope and published/recency
#  features. AW_RANKHEAD_V2_DUMP exposes the per-candidate features + base
#  scores for the offline miner (`mine_rankfam_pools.py`, deleted; last at git-blob:51ff8e3da1a107339e0e5645c822a198c1e069a4:mine_rankfam_pools.py) so training
#  reads features from the serving code path itself — zero train/serve skew.
#  Flag-gated AW_RANKHEAD_V2 (default OFF, byte-identical);
#  rollback = AW_RANKHEAD_V2=0.
# --------------------------------------------------------------------------- #
_RANKHEAD_V2 = _S.query.rankhead_v2
_RANKHEAD_V2_PATH = _S.runtime.rankhead_v2_path
_RANKHEAD_V2_DUMP = _S.runtime.rankhead_v2_dump
_LAST_V2_DUMP: list[dict] = []   # per-search feature dump (miner-only)


def _load_rankhead():
    if not _RANKHEAD:
        return None
    try:
        import numpy as _np
        with open(_RANKHEAD_PATH) as f:
            h = validate_head(json.load(f))   # ADR-036 schema check: a malformed head disables itself, loudly
        h["_variant"] = h.get("variant", "v1")
        h["_beta_gen"] = float(h.get("beta_gen", 0.6))
        h["_beta_max"] = float(h.get("beta_max", 0.9))
        h["_trust_scale"] = float(h.get("trust_scale", 1.0))
        if "w" in h:
            h["_w"] = _np.asarray(h["w"], dtype="float32")
            h["_b"] = float(h.get("b", 0.0))
        if "w_beta" in h:
            h["_wb"] = _np.asarray(h["w_beta"], dtype="float32")
            h["_bb"] = float(h.get("b_beta", 0.0))
        return h
    except Exception as e:  # noqa: BLE001 — never let a bad head break search
        import logging
        logging.getLogger("aw_server.retrieval").warning(
            "rankhead load failed (%s: %s) — head disabled", type(e).__name__, e)
        return None


_RANKHEAD_MODEL = _load_rankhead()


def _load_rankhead_v2():
    if not (_RANKHEAD_V2 or _RANKHEAD_V2_DUMP):
        return None
    try:
        from . import rankhead_v2 as _rh2
        if not _RANKHEAD_V2:      # dump-only mode needs the module, not a head
            return None
        return _rh2.load_head(_RANKHEAD_V2_PATH)
    except Exception as e:  # noqa: BLE001 — never let a bad head break search
        import logging
        logging.getLogger("aw_server.retrieval").warning(
            "rankhead_v2 load failed (%s: %s) — head disabled",
            type(e).__name__, e)
        return None


_RANKHEAD_V2_MODEL = _load_rankhead_v2()


def _head_beta(qfull) -> float:
    """Per-query relgate beta from the head, given the full 1024-dim normalized
    query vector. Returns 0.0 (relgate off) when the head is unavailable or the
    vector is missing — fail-safe to the current prod behaviour."""
    h = _RANKHEAD_MODEL
    if h is None or qfull is None:
        return 0.0
    import numpy as _np
    x = _np.asarray(qfull, dtype="float32")
    if x.shape[0] != (h.get("_w", h.get("_wb")).shape[0]):
        return 0.0
    v = h["_variant"]
    if v == "v3":
        beta = float(x @ h["_wb"] + h["_bb"])
        return max(0.0, min(h["_beta_max"], beta))
    p = 1.0 / (1.0 + math.exp(-float(x @ h["_w"] + h["_b"])))
    if v == "v2":
        return max(0.0, min(h["_beta_max"], h["_beta_gen"] * p))
    return h["_beta_gen"] if p >= 0.5 else 0.0   # v1 discrete preset


# --------------------------------------------------------------------------- #
#  UNIFIED QUERY-UNDERSTANDING HEAD (ADR-062 / specs/unified-query-head.md).
#  Phase 1 = the COLD-START self-distillation head: a do-no-harm generalisation
#  of the ADR-036 `_head_beta` scalar (above) into the FULL ~14-knob per-query
#  vector. It reuses the SAME frozen 1024-d arctic query vector (no extra embed)
#  and is Q1-ratified as ONE shared feature + N INDEPENDENT linear output heads
#  (not a monolithic tensor), so every knob is independently gateable/revertible.
#
#  Precedence (spec §3): operator env pin > explicit agent arg > head inference >
#  static fusion_weights default. The head only ever fills a knob the layers above
#  left unset; every knob resolves through `qh.get(name, <today's static value>)`
#  at its scoring site, so an absent/abstaining head is BYTE-IDENTICAL to today.
#
#  Per-knob v2 soft-interpolation (Q3): the value fed to the scorer is
#      value = default + alpha * (clip(w·x + b, lo, hi) - default)
#  where `default` is the knob's inert value (spec §1) and `alpha` in [0,1] is the
#  per-knob blend toward the learned target. COLD-START ships alpha=0 for EVERY
#  head-driven knob -> the head emits NOTHING into the driving dict -> the resolver
#  falls straight through to today's static value. That is what makes the ON path
#  reproduce the OFF path byte-for-byte. Phase 2+ raises alpha per knob ONLY where
#  the ADR-043 per-class zero-regression gate clears (improvement is opt-in).
#
#  Non-head drivers (kept from today per Q5 — lexicons as feature + fallback):
#    trust_beta      -> ADR-036 `_head_beta` (output #3, unchanged; not regressed)
#    academic        -> `_query_academic_intent` lexicon (+ optional shadow head)
#    commercial_*    -> `_query_commercial_intent` lexicon (+ optional shadow head)
#    country_scope   -> deterministic `_query_jurisdiction` (Q2: scope stays
#                       deterministic; the head learns only the penalty MAGNITUDE
#                       = the `off_country` knob, which IS head-driven).
#
#  Fail-safe (spec §5, mirrors `_head_beta` failing to 0.0): head file missing /
#  bad JSON / vector missing / shape mismatch -> `_qhead_vector` returns {} and
#  every knob uses its static default. A single bad knob never breaks search.
#  Flag-gated AW_QHEAD (default OFF, byte-identical); per-knob allowlist
#  AW_QHEAD_KNOBS (default "all"); rollback = AW_QHEAD=0.
# --------------------------------------------------------------------------- #
_QHEAD = _S.query.qhead
_QHEAD_PATH = _S.runtime.qhead_path
# Knobs the head is ALLOWED to drive this deploy. "all" = every head-driven knob;
# else a comma list (e.g. "trust_scale,off_country,country_boost"). A knob not in
# the allowlist always falls back to its static default (spec §6.2 per-knob rollout).
#
# ADR-062 Phase-2 (country): the off-country penalty MAGNITUDE (spec §1 knob 12)
# is exposed to operators under the descriptive sub-flag name `off_country_mag`
# (the knob is wired internally as `off_country`, the weight on `_off_country_hit`).
# The alias keeps the flag self-documenting ("magnitude", not the raw penalty) while
# the code + artifact keep the single wired name. `AW_QHEAD_KNOBS=off_country_mag`
# therefore activates the `off_country` head; `off_country` still works too.
#
# ADR-062 Phase-2b (task 281): `scope` is the operator alias for the learned
# `country_scope` head (knob 13). Unlike the soft knobs, country_scope is NOT
# covered by the "all" wildcard — it must be named EXPLICITLY (so the Phase-1/2
# "all" byte-identical gates are preserved and enabling sub-national scope is a
# conscious op). `AW_QHEAD_KNOBS=country_scope` (or `scope`) activates it.
_QHEAD_KNOB_ALIASES = {"off_country_mag": "off_country", "scope": "country_scope"}
_QHEAD_KNOBS = frozenset(
    _QHEAD_KNOB_ALIASES.get(k.strip(), k.strip())
    for k in _S.query.qhead_knobs.split(",") if k.strip())
# country_scope is active ONLY when explicitly allowlisted (never via "all") and
# the head is on — so no deploy (AW_QHEAD=0) and no "all" gate is perturbed.
_SCOPE_ACTIVE = _QHEAD and "country_scope" in _QHEAD_KNOBS
# Experimental successors to the keyword-first scope policy.  `legacy` is the
# literal pre-experiment path; `learned_only` changes only free-text country
# inference.  `policy_only` is Task374's stronger retrieval-isolation arm: free
# text cannot create a ranking country scope at all.  Its typed candidate policy
# may still add bounded country dense/BM25 legs, but it cannot enter a country
# boost or demotion.  The caller's explicit `country=` argument remains a
# deterministic hard override in every mode.  This is deliberately a mode rather
# than another truthy flag so gate artifacts name the complete policy unambiguously.
def _parse_country_scope_mode(raw):
    mode = (raw or "legacy").strip().lower()
    if mode not in {"legacy", "learned_only", "policy_only"}:
        raise ValueError(
            "AW_COUNTRY_SCOPE_MODE must be 'legacy', 'learned_only', or "
            "'policy_only', "
            f"got {raw!r}")
    return mode


_COUNTRY_SCOPE_MODE = _parse_country_scope_mode(
    _S.query.country_scope_mode)
_LEARNED_ONLY_SCOPE = _COUNTRY_SCOPE_MODE == "learned_only"
_POLICY_ONLY_SCOPE = _COUNTRY_SCOPE_MODE == "policy_only"
# Head-driven knob names (soft weights/magnitudes). scope/beta/lexicon knobs are
# handled by their existing deterministic/learned paths, NOT emitted here.
_QHEAD_DRIVEN = frozenset({
    "dense_scale", "bm25_scale", "trust_scale", "recency", "title", "exact",
    "entity_card", "machine", "off_country", "country_boost",
})


def _load_qhead():
    """Load the unified-head artifact into per-knob numpy linear heads. Returns a
    dict {knob: {alpha, default, lo, hi, w, b}} for head-driven knobs, or None on
    any failure (search then uses static defaults — byte-identical to today)."""
    if not _QHEAD:
        return None
    try:
        import numpy as _np
        with open(_QHEAD_PATH) as f:
            spec = json.load(f)
        dim = int(spec.get("input_dim", 1024))
        knobs = {}
        for name, cfg in (spec.get("knobs") or {}).items():
            if cfg.get("driver") != "head" or name not in _QHEAD_DRIVEN:
                continue                       # non-head knobs keep their own path
            alpha = float(cfg.get("alpha", 0.0))
            entry = {
                "alpha": alpha,
                "default": float(cfg.get("default", 0.0)),
                "lo": float(cfg.get("lo", -1e9)),
                "hi": float(cfg.get("hi", 1e9)),
            }
            if "w" in cfg:
                w = _np.asarray(cfg["w"], dtype="float32")
                if w.shape[0] != dim:          # fail SAFE: drop this knob
                    continue
                entry["w"] = w
                entry["b"] = float(cfg.get("b", 0.0))
            knobs[name] = entry
        return {"_dim": dim, "_knobs": knobs}
    except Exception as e:  # noqa: BLE001 — never let a bad head break search
        import logging
        logging.getLogger("aw_server.retrieval").warning(
            "qhead load failed (%s: %s) — unified head disabled", type(e).__name__, e)
        return None


_QHEAD_MODEL = _load_qhead()


def _qhead_vector(qfull) -> dict:
    """Per-query DRIVING values for the head-driven knobs, given the frozen
    1024-d query vector. A knob is emitted ONLY when it is (a) in the active
    allowlist and (b) has alpha > 0 — so the COLD-START artifact (alpha=0 for
    every knob) emits nothing and each knob resolves to its static default,
    byte-identical to today. Fail-safe: head/vector missing or shape mismatch
    -> {} (all knobs fall back)."""
    m = _QHEAD_MODEL
    if m is None or qfull is None:
        return {}
    try:
        import numpy as _np
        x = _np.asarray(qfull, dtype="float32")
        if x.shape[0] != m["_dim"]:
            return {}
        out = {}
        for name, e in m["_knobs"].items():
            if _QHEAD_KNOBS != frozenset({"all"}) and name not in _QHEAD_KNOBS:
                continue
            if e["alpha"] <= 0.0 or "w" not in e:
                continue                        # cold-start / no-op knob: skip
            pred = float(x @ e["w"] + e["b"])
            pred = max(e["lo"], min(e["hi"], pred))
            out[name] = e["default"] + e["alpha"] * (pred - e["default"])
        return out
    except Exception:  # noqa: BLE001 — a bad matmul must not break search
        return {}


# --------------------------------------------------------------------------- #
#  Learned country_scope head (ADR-062 knob 13, amended Q2 — task 281).
#  A CALIBRATED, ABSTAINING, SOFT classifier over the frozen 1024-d query vector:
#  softmax(P(country)) with temperature scaling. It fires ONLY above a confidence
#  threshold; below → abstain (None). Its confidence scales the off-country penalty
#  magnitude (knob 12) so a low-confidence scope demotes SOFTLY (ADR-046: a
#  mis-scope demotes, never excludes). Stored as the `country_scope` classifier
#  entry in qhead.json. Fail-safe (mirrors _head_beta→0.0): missing/bad head or a
#  vector shape mismatch -> (None, 0.0) => no scope => today's keyword-only path.
# --------------------------------------------------------------------------- #
_SCOPE_KNOB = "country_scope"


def _load_scope_head():
    """Load the country_scope classifier from qhead.json (W: C×dim, b: C, classes,
    temperature, threshold). Returns None on any failure (search abstains)."""
    # The learned-only experiment is independently selectable: it needs only the
    # classifier entry, not the other unified-qhead knobs.  Legacy behaviour still
    # observes AW_QHEAD exactly as before.
    if not (_QHEAD or _LEARNED_ONLY_SCOPE):
        return None
    try:
        import numpy as _np
        with open(_QHEAD_PATH) as f:
            spec = json.load(f)
        cfg = (spec.get("knobs") or {}).get(_SCOPE_KNOB)
        if not cfg or cfg.get("kind") != "classifier" or "W" not in cfg:
            return None
        if not cfg.get("enabled", True):
            return None
        classes = list(cfg["classes"])
        dim = int(spec.get("input_dim", 1024))
        W = _np.asarray(cfg["W"], dtype="float32")
        b = _np.asarray(cfg.get("b", [0.0] * len(classes)), dtype="float32")
        if W.shape != (len(classes), dim) or b.shape[0] != len(classes):
            return None
        return {"W": W, "b": b, "classes": classes, "dim": dim,
                "temp": float(cfg.get("temperature", 1.0)),
                "thr": float(cfg.get("threshold", 0.5)),
                "none": cfg.get("none_class", "_NONE_")}
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger("aw_server.retrieval").warning(
            "scope head load failed (%s: %s) — learned scope disabled",
            type(e).__name__, e)
        return None


_SCOPE_HEAD = _load_scope_head()


def _load_query_policy_heads():
    """Load the candidate-only typed action block from the qhead artifact.

    `knobs.country_scope` remains the legacy serving classifier.  Retrieval
    routing deliberately requires the newer `policy_heads` block: an older
    artifact therefore cannot silently turn a ranking experiment into a recall
    experiment.  The entire block is rejected on a malformed required head so
    the flag degrades visibly to no country legs rather than guessing defaults.
    """
    if not _QHEAD_COUNTRY_LEGS:
        return None
    try:
        import numpy as _np
        with open(_QHEAD_PATH) as f:
            spec = json.load(f)
        if spec.get("artifact_role") != "candidate_only":
            return None
        contract = spec.get("query_policy_candidate")
        if (not isinstance(contract, dict)
                or contract.get("schema") != "aw.query-policy.candidate.v1"
                or contract.get("artifact_role") != "candidate_only"):
            return None
        sha256_pattern = re.compile(r"^[0-9a-f]{64}$")
        provenance_hashes = (contract.get(name) for name in (
            "base_qhead_sha256", "train_sha256", "calibration_sha256"))
        if any(not isinstance(value, str) or not sha256_pattern.fullmatch(value)
               for value in provenance_hashes):
            return None
        heads = spec.get("policy_heads")
        if not isinstance(heads, dict):
            return None
        dim = spec.get("input_dim")
        if (isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                or contract.get("input_dim") != dim):
            return None
        required = {
            "country_scope", "jurisdiction_binding", "foreign_authority_needed",
            "multi_scope", "country_dense_quota", "country_bm25_quota",
        }
        if set(heads) < required:
            return None
        country = heads["country_scope"]
        classes = list(country["classes"])
        none_class = country.get("none_class")
        country_code = re.compile(r"^[A-Z]{2}$")
        if (not classes or any(not isinstance(value, str) for value in classes)
                or len(set(classes)) != len(classes)
                or not isinstance(none_class, str)
                or classes.count(none_class) != 1
                or any(value != none_class and not country_code.fullmatch(value)
                       for value in classes)):
            return None
        W = _np.asarray(country["W"], dtype="float32")
        b = _np.asarray(country["b"], dtype="float32")
        if (country.get("kind") != "classifier" or W.shape != (len(classes), dim)
                or b.shape != (len(classes),)
                or country.get("input_dim") != dim
                or not _np.isfinite(W).all() or not _np.isfinite(b).all()):
            return None
        country_temp = float(country.get("temperature", 1.0))
        country_threshold = float(country.get("threshold", 0.5))
        if (not math.isfinite(country_temp) or country_temp <= 0.0
                or not math.isfinite(country_threshold)
                or not 0.0 <= country_threshold <= 1.0):
            return None
        scalar = {}
        binary_heads = {
            "jurisdiction_binding", "foreign_authority_needed", "multi_scope"}
        quota_heads = {"country_dense_quota", "country_bm25_quota"}
        for name in required - {"country_scope"}:
            cfg = heads[name]
            w = _np.asarray(cfg["W"], dtype="float32")
            expected_kind = ("binary_logistic" if name in binary_heads
                             else "bounded_scalar")
            expected_activation = "sigmoid" if name in binary_heads else "identity"
            bias = float(cfg["b"])
            if (name not in binary_heads | quota_heads
                    or cfg.get("kind") != expected_kind
                    or cfg.get("activation") != expected_activation
                    or cfg.get("input_dim") != dim
                    or w.shape != (dim,) or not _np.isfinite(w).all()
                    or not math.isfinite(bias)):
                return None
            lo, hi = float(cfg.get("lo", 0.0)), float(cfg.get("hi", 1.0))
            temperature = float(cfg.get("temperature", 1.0))
            if (not math.isfinite(lo) or not math.isfinite(hi)
                    or not 0.0 <= lo <= hi <= 1.0
                    or not math.isfinite(temperature) or temperature <= 0.0):
                return None
            threshold = float(cfg.get("threshold", 0.5))
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                return None
            scalar[name] = {"W": w, "b": bias,
                            "activation": expected_activation, "lo": lo, "hi": hi,
                            "temperature": temperature, "threshold": threshold}
        return {"dim": dim, "country": {
            "W": W, "b": b, "classes": classes,
            "temp": country_temp, "threshold": country_threshold,
            "none": none_class}, "scalar": scalar}
    except Exception as e:  # noqa: BLE001 -- candidate policy must never break search
        import logging
        logging.getLogger("aw_server.retrieval").warning(
            "query-policy load failed (%s: %s) — country retrieval legs disabled",
            type(e).__name__, e)
        return None


_QUERY_POLICY_HEADS = _load_query_policy_heads()


def _query_policy_vector(qfull) -> dict:
    """Run all typed routing outputs from one frozen query-vector forward pass."""
    m = _QUERY_POLICY_HEADS
    if m is None or qfull is None:
        return {}
    try:
        import numpy as _np
        x = _np.asarray(qfull, dtype="float32")
        if x.shape != (m["dim"],) or not _np.isfinite(x).all():
            return {}
        c = m["country"]
        logits = (c["W"] @ x + c["b"]) / max(c["temp"], 1e-6)
        logits = logits - float(logits.max())
        prob = _np.exp(logits)
        prob = prob / float(prob.sum())
        idx = int(prob.argmax())
        country, confidence = c["classes"][idx], float(prob[idx])
        if country == c["none"] or confidence < c["threshold"]:
            country, confidence = None, 0.0
        out = {"country": country, "country_confidence": confidence}
        for name, cfg in m["scalar"].items():
            value = float(x @ cfg["W"] + cfg["b"])
            if cfg["activation"] == "sigmoid":
                value = value / cfg["temperature"]
                value = 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, value))))
            out[name] = max(cfg["lo"], min(cfg["hi"], value))
            out[f"{name}_threshold"] = cfg["threshold"]
        return out
    except Exception:  # noqa: BLE001 -- abstain, preserving global retrieval
        return {}

if _LEARNED_ONLY_SCOPE:
    import logging
    _scope_log = logging.getLogger("aw_server.retrieval")
    if _SCOPE_HEAD is None:
        # Fail visibly and abstain.  Falling back to keyword/gazetteer scope here
        # would make a broken learned-only gate silently measure the legacy arm.
        _scope_log.error(
            "AW_COUNTRY_SCOPE_MODE=learned_only but country_scope head is "
            "unavailable — free-text country scope will abstain; keyword and "
            "gazetteer fallback remain disabled")
    else:
        _scope_log.info(
            "AW_COUNTRY_SCOPE_MODE=learned_only active — free-text country scope "
            "uses the calibrated classifier; keyword/gazetteer scope disabled")


def _scope_head_predict(qfull):
    """-> (country|None, confidence in [0,1]). Calibrated softmax over the scope
    classes. ABSTAINS (None, 0.0) when argmax is the None class OR the calibrated
    top prob is below threshold (learned abstain-on-ambiguity)."""
    m = _SCOPE_HEAD
    if m is None or qfull is None:
        return None, 0.0
    try:
        import numpy as _np
        x = _np.asarray(qfull, dtype="float32")
        if x.shape[0] != m["dim"]:
            return None, 0.0
        logits = (m["W"] @ x + m["b"]) / max(m["temp"], 1e-6)
        logits = logits - float(logits.max())
        p = _np.exp(logits)
        p = p / float(p.sum())
        idx = int(p.argmax())
        cls, conf = m["classes"][idx], float(p[idx])
        if cls == m["none"] or conf < m["thr"]:
            return None, 0.0
        return cls, conf
    except Exception:  # noqa: BLE001 — a bad matmul must not break search
        return None, 0.0


def _resolve_scope(q, q_country_kw, qfull, learned_only=False):
    """-> (det_country, soft_country, scope_confidence). Precedence (spec §3):
    keyword detector > gazetteer rollup > learned head; each ADDS scope only where
    the higher tier found none.

    The split is load-bearing (ADR-054 "a wrong scope is worse than no scope"):
      * `det_country` — DETERMINISTIC scope (keyword or gazetteer, confidence 1.0).
        Drives the FULL scoped machinery exactly like today's keyword scope
        (foreign_gov, official_pref, canton) — byte-identical when keyword fires.
      * `soft_country` — a LEARNED head-inferred scope (only when the deterministic
        tiers found none AND no homograph abstained). It drives ONLY the
        confidence-scaled off-country penalty (knob 12 × knob 13), NEVER the hard
        scoped levers — so a confident-but-non-jurisdictional inference (e.g.
        "la Révolution française"→FR) cannot bury a reference/entity gold via
        official_pref; the off-country penalty can only demote a CONFLICTING-country
        stamped candidate, never an INTL/in-country gold.
    A gazetteer homograph HARD-abstains the head. Only reached when country_scope
    is active and no explicit country= arg was supplied.

    In the flag-gated `learned_only` experiment the keyword and gazetteer values
    are intentionally not consulted: the calibrated classifier either emits a
    soft country/confidence or abstains.  Explicit `country=` is resolved by the
    caller before this function and never reaches this branch."""
    if learned_only:
        hc, hconf = _scope_head_predict(qfull)
        if hc is not None:
            return None, hc, hconf
        return None, None, 0.0
    if q_country_kw is not None:
        return q_country_kw, None, 1.0          # keyword wins, full deterministic
    gaz_country, gaz_abstain = _gazetteer_scope(q)
    if gaz_country is not None:
        return gaz_country, None, 1.0           # deterministic rollup, full scope
    if gaz_abstain:
        return None, None, 1.0                  # homograph → abstain, skip the head
    hc, hconf = _scope_head_predict(qfull)
    if hc is not None:
        return None, hc, hconf                  # learned → knob-12 penalty ONLY
    return None, None, 1.0


def _effective_free_text_country_scope(q, q_country_kw, qfull):
    """Return ``(hard_country, soft_country, confidence)`` for free text.

    This is the single serve/gate diagnostic contract for mode semantics.  In
    legacy mode it preserves keyword-first, then optional gazetteer/head scope.
    In learned-only mode keyword and gazetteer values have no control effect: the
    classifier emits a calibrated soft scope or abstains.  Explicit ``country=``
    remains a caller-owned hard override and must bypass this helper.

    Task374's ``policy_only`` mode returns no free-text ranking scope without
    consulting the keyword, gazetteer, or legacy ``knobs.country_scope`` head.
    The separate typed ``policy_heads`` forward pass can still nominate a country
    for bounded candidate legs; it is never an input to this function.  Keeping
    this stop at the one scope-resolution boundary makes ``AW_QHEAD_COUNTRY_LEGS=0``
    a clean park: no inferred country action remains in ranking.
    """
    if _POLICY_ONLY_SCOPE:
        return None, None, 1.0
    if _LEARNED_ONLY_SCOPE:
        return _resolve_scope(q, q_country_kw, qfull, learned_only=True)
    if _SCOPE_ACTIVE and q_country_kw is None:
        return _resolve_scope(q, q_country_kw, qfull)
    return q_country_kw, None, 1.0


def _country_scope_for_ranking(explicit_country, q, q_country_kw, qfull):
    """Resolve the only country scope allowed to enter ranking.

    An explicit caller instruction is the long-standing override contract.  In
    every other case this delegates exclusively to the mode-gated free-text
    resolver above.  In particular, the typed Task374 policy-head country is
    intentionally absent from this signature: it is candidate-leg routing
    evidence, never a country-conditioned ranking signal.
    """
    if explicit_country is not None:
        return explicit_country, None, 1.0
    return _effective_free_text_country_scope(q, q_country_kw, qfull)


def _head_artifacts() -> dict:
    """The learned heads' entries of `GET /config`'s artifact listing, spliced into
    `tables.loaded_artifacts(heads)` (the heads live here until step 5)."""
    _present = _settings.present
    return {
        "rankhead": (_RANKHEAD_PATH, _present(_RANKHEAD_MODEL)),
        "rankhead_v2": (_RANKHEAD_V2_PATH, _present(_RANKHEAD_V2_MODEL)),
        "qhead": (_QHEAD_PATH, _present(_QHEAD_MODEL)),
        "scope_head": (_QHEAD_PATH, _present(_SCOPE_HEAD)),
        "query_policy_heads": (_QHEAD_PATH, _present(_QUERY_POLICY_HEADS)),
    }
