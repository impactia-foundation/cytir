"""Unit contract for the typed, additive country retrieval query plan.

These tests deliberately execute the pure planner and artifact reader without
importing the Qdrant-serving module.  The gate remains the authority for ranking
quality; this suite pins the mechanical safety properties needed before a gate:
an absent candidate action block is inert, global limits cannot be exceeded, and
explicit agent control has precedence over inferred country labels.
"""
import json
import math
import pathlib
from dataclasses import dataclass


def _settings_ns() -> dict:
    """The typed settings the sliced code reads (S10 step 2, task 1071), loaded NOW: a fresh
    `settings.py` (a leaf module) parses the env this test just set, as an import would."""
    import importlib.util
    import sys as _sys
    path = pathlib.Path(__file__).resolve().parent.parent / "cytir" / "serve" / "settings.py"
    spec = importlib.util.spec_from_file_location("_aw_settings_for_slice", path)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return {"_S": mod.SETTINGS, "_settings": mod}


RETRIEVAL = (pathlib.Path(__file__).resolve().parent.parent
             / "cytir" / "serve" / "retrieval.py")


PLAN = RETRIEVAL.parent / "plan.py"   # the planner moved out of retrieval.py (S10 step 6, task 1104)


def _planner_namespace():
    src = PLAN.read_text()
    start = src.index("_COUNTRY_LEG_MAX_LIMIT =")
    end = src.index("\n\ndef _country_partition_filter", start)
    ns = {**_settings_ns(), "os": __import__("os"), "math": math, "dataclass": dataclass,
          "_QHEAD_COUNTRY_LEGS": False}   # the flag moved to `heads` (task 1099); the plan only reads it
    exec(compile(src[start:end], str(PLAN), "exec"), ns)
    return ns


def _policy_namespace(artifact):
    """`heads._load_query_policy_heads` (S10 step 5, task 1099), rebuilt against a namespace whose
    country-legs flag is on and whose qhead path is the test artifact."""
    import sys
    import types
    root = pathlib.Path(__file__).resolve().parent.parent
    for d in (root, root / "server"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    from cytir.serve import heads
    ns = dict(vars(heads))
    ns.update({"_QHEAD_COUNTRY_LEGS": True, "_QHEAD_PATH": str(artifact)})
    f = heads._load_query_policy_heads
    ns["_load_query_policy_heads"] = types.FunctionType(f.__code__, ns, f.__name__)
    g = heads._query_policy_vector
    ns["_query_policy_vector"] = types.FunctionType(g.__code__, ns, g.__name__)
    ns["_QUERY_POLICY_HEADS"] = ns["_load_query_policy_heads"]()
    return ns


def _trainer_candidate(heads):
    """Small artifact with the exact schema emitted by cytir.retune.policy_train."""
    return {
        "input_dim": 2,
        "artifact_role": "candidate_only",
        "query_policy_candidate": {
            "schema": "aw.query-policy.candidate.v1",
            "artifact_role": "candidate_only",
            "input_dim": 2,
            "base_qhead_sha256": "a" * 64,
            "train_sha256": "b" * 64,
            "calibration_sha256": "c" * 64,
        },
        "policy_heads": json.loads(json.dumps(heads)),
    }


def test_disabled_or_abstaining_plan_is_inert():
    make = _planner_namespace()["_country_query_plan"]
    assert make("CH", "FR", 0.99, 100, enabled=False).country_legs_enabled is False
    # Task374's policy-only arm parks an otherwise confident typed inferred
    # country completely when AW_QHEAD_COUNTRY_LEGS=0.
    assert make(None, "CH", 0.99, 100, enabled=False).country_legs_enabled is False
    assert make(None, None, 0.0, 100, enabled=True).country_legs_enabled is False
    assert make(None, "CH", 0.59, 100, enabled=True,
                min_confidence=0.60).country_legs_enabled is False


def test_explicit_country_wins_and_all_limits_are_bounded():
    make = _planner_namespace()["_country_query_plan"]
    plan = make("CH", "FR", 0.99, 20, enabled=True,
                dense_limit=32, bm25_limit=32,
                jurisdiction_binding=0.0, foreign_authority_needed=1.0,
                multi_scope=1.0, dense_quota=0.5, bm25_quota=0.75)
    assert plan.country == "CH"
    assert plan.source == "explicit"
    assert (plan.country_dense_limit, plan.country_bm25_limit) == (16, 20)
    assert plan.country_dense_limit <= 20 and plan.country_bm25_limit <= 20


def test_inferred_country_requires_binding_and_rejects_foreign_or_multi_scope():
    make = _planner_namespace()["_country_query_plan"]
    common = dict(enabled=True, dense_limit=32, bm25_limit=32,
                  min_confidence=0.60, jurisdiction_binding=0.9,
                  foreign_authority_needed=0.1, multi_scope=0.1,
                  dense_quota=0.5, bm25_quota=0.75)
    plan = make(None, "DE", 0.8, 100, **common)
    assert plan.country == "DE" and plan.source == "head"
    assert (plan.country_dense_limit, plan.country_bm25_limit) == (16, 24)
    assert not make(None, "DE", 0.8, 100,
                    **dict(common, foreign_authority_needed=0.9)).country_legs_enabled
    assert not make(None, "DE", 0.8, 100,
                    **dict(common, multi_scope=0.9)).country_legs_enabled
    assert not make(None, "DE", 0.8, 100,
                    **dict(common, jurisdiction_binding=0.1)).country_legs_enabled


def test_zero_quota_suppresses_only_that_modality():
    make = _planner_namespace()["_country_query_plan"]
    common = dict(enabled=True, dense_limit=32, bm25_limit=32,
                  min_confidence=0.60, jurisdiction_binding=0.9,
                  foreign_authority_needed=0.1, multi_scope=0.1)
    dense_only = make(None, "CH", 0.9, 100,
                      **dict(common, dense_quota=0.5, bm25_quota=0.0))
    assert dense_only.country_legs_enabled
    assert (dense_only.country_dense_limit, dense_only.country_bm25_limit) == (16, 0)
    bm25_only = make(None, "CH", 0.9, 100,
                     **dict(common, dense_quota=0.0, bm25_quota=0.5))
    assert bm25_only.country_legs_enabled
    assert (bm25_only.country_dense_limit, bm25_only.country_bm25_limit) == (0, 16)
    none = make(None, "CH", 0.9, 100,
                **dict(common, dense_quota=0.0, bm25_quota=0.0))
    assert not none.country_legs_enabled


def test_candidate_artifact_runs_one_typed_forward_pass(tmp_path):
    heads = {
        "country_scope": {
            "kind": "classifier", "input_dim": 2,
            "classes": ["_NONE_", "CH"],
            "W": [[0.0, 0.0], [3.0, 0.0]], "b": [0.0, 0.0],
            "temperature": 1.0, "threshold": 0.55, "none_class": "_NONE_",
        },
        "jurisdiction_binding": {
            "kind": "binary_logistic", "input_dim": 2,
            "W": [4.0, 0.0], "b": 0.0, "activation": "sigmoid",
            "temperature": 2.0,
            "lo": 0.0, "hi": 1.0, "threshold": 0.5,
        },
        "foreign_authority_needed": {
            "kind": "binary_logistic", "input_dim": 2,
            "W": [-4.0, 0.0], "b": 0.0, "activation": "sigmoid",
            "temperature": 2.0,
            "lo": 0.0, "hi": 1.0, "threshold": 0.5,
        },
        "multi_scope": {
            "kind": "binary_logistic", "input_dim": 2,
            "W": [-4.0, 0.0], "b": 0.0, "activation": "sigmoid",
            "temperature": 2.0,
            "lo": 0.0, "hi": 1.0, "threshold": 0.5,
        },
        "country_dense_quota": {
            "kind": "bounded_scalar", "input_dim": 2,
            "W": [0.0, 0.0], "b": 0.5, "activation": "identity",
            "lo": 0.0, "hi": 1.0,
        },
        "country_bm25_quota": {
            "kind": "bounded_scalar", "input_dim": 2,
            "W": [0.0, 0.0], "b": 0.75, "activation": "identity",
            "lo": 0.0, "hi": 1.0,
        },
    }
    artifact = tmp_path / "qhead-policy.json"
    artifact.write_text(json.dumps(_trainer_candidate(heads)))
    out = _policy_namespace(artifact)["_query_policy_vector"]([1.0, 0.0])
    assert out["country"] == "CH"
    assert out["country_confidence"] > 0.9
    # Binary temperatures are calibration, not trainer metadata: serving must
    # exactly reproduce sigmoid((W·x+b)/T), not sigmoid(W·x+b).
    assert math.isclose(out["jurisdiction_binding"], 1.0 / (1.0 + math.exp(-2.0)))
    assert math.isclose(out["foreign_authority_needed"], 1.0 / (1.0 + math.exp(2.0)))
    assert math.isclose(out["multi_scope"], 1.0 / (1.0 + math.exp(2.0)))
    assert out["country_dense_quota"] == 0.5
    assert out["country_bm25_quota"] == 0.75
    heads["multi_scope"]["temperature"] = 0.0
    artifact.write_text(json.dumps(_trainer_candidate(heads)))
    assert _policy_namespace(artifact)["_query_policy_vector"]([1.0, 0.0]) == {}
    heads["multi_scope"]["temperature"] = 2.0

    malformed = _trainer_candidate(heads)
    malformed["query_policy_candidate"]["schema"] = "aw.query-policy.candidate.v0"
    artifact.write_text(json.dumps(malformed))
    assert _policy_namespace(artifact)["_query_policy_vector"]([1.0, 0.0]) == {}

    malformed = _trainer_candidate(heads)
    malformed["query_policy_candidate"]["input_dim"] = 3
    artifact.write_text(json.dumps(malformed))
    assert _policy_namespace(artifact)["_query_policy_vector"]([1.0, 0.0]) == {}

    malformed = _trainer_candidate(heads)
    malformed["policy_heads"]["country_scope"]["classes"] = ["_NONE_", "SWISS"]
    artifact.write_text(json.dumps(malformed))
    assert _policy_namespace(artifact)["_query_policy_vector"]([1.0, 0.0]) == {}

    malformed = _trainer_candidate(heads)
    malformed["policy_heads"]["country_scope"]["none_class"] = "MISSING"
    artifact.write_text(json.dumps(malformed))
    assert _policy_namespace(artifact)["_query_policy_vector"]([1.0, 0.0]) == {}

    malformed = _trainer_candidate(heads)
    del malformed["query_policy_candidate"]["train_sha256"]
    artifact.write_text(json.dumps(malformed))
    assert _policy_namespace(artifact)["_query_policy_vector"]([1.0, 0.0]) == {}

    malformed = _trainer_candidate(heads)
    malformed["query_policy_candidate"]["calibration_sha256"] = "A" * 64
    artifact.write_text(json.dumps(malformed))
    assert _policy_namespace(artifact)["_query_policy_vector"]([1.0, 0.0]) == {}

    malformed = _trainer_candidate(heads)
    malformed["policy_heads"]["country_dense_quota"]["kind"] = "binary_logistic"
    artifact.write_text(json.dumps(malformed))
    assert _policy_namespace(artifact)["_query_policy_vector"]([1.0, 0.0]) == {}

    # Python's JSON decoder accepts NaN by default. The serving contract must
    # still reject the entire block rather than let min/max clamp it to a quota.
    malformed = _trainer_candidate(heads)
    malformed["policy_heads"]["country_scope"]["W"][0][0] = float("nan")
    artifact.write_text(json.dumps(malformed))
    assert _policy_namespace(artifact)["_query_policy_vector"]([1.0, 0.0]) == {}

    malformed = _trainer_candidate(heads)
    malformed["policy_heads"]["jurisdiction_binding"]["b"] = float("nan")
    artifact.write_text(json.dumps(malformed))
    assert _policy_namespace(artifact)["_query_policy_vector"]([1.0, 0.0]) == {}


def test_dispatch_keeps_global_legs_and_uses_one_batched_request():
    src = (RETRIEVAL.parent / "legs.py").read_text()      # stage 1 moved there (task 1116)
    dispatch = src[src.index("_specs = [(\"dense\""):src.index("# per-token bm25 sub-legs")]
    assert '("dense", dense, DENSE_VEC, candidates' in dispatch
    assert '("bm25", bm25, SPARSE_BM25, candidates' in dispatch
    assert '"country_dense"' in dispatch and '"country_bm25"' in dispatch
    assert "if query_plan.country_dense_limit:" in dispatch
    assert "if query_plan.country_bm25_limit:" in dispatch
    assert "query_batch_points(" in dispatch
    assert "country_leg_provenance" not in dispatch  # provenance is post-dispatch


def test_gate_trace_is_opt_in_and_contains_global_invariance_evidence():
    # the trace flag lives in `trace`; its writers are stage 1 (`legs`) and stage 2 (`retrieval`)
    src = (RETRIEVAL.parent / "legs.py").read_text() + RETRIEVAL.read_text()
    assert "_QHEAD_COUNTRY_TRACE = _S.runtime.qhead_country_trace" in (RETRIEVAL.parent / "trace.py").read_text()
    settings_src = (RETRIEVAL.parent / "settings.py").read_text()
    assert 'qhead_country_trace: bool = knob("AW_QHEAD_COUNTRY_TRACE", FLAG, "0"' in settings_src
    for key in ("country_rescue_ids", "qdrant_main_batch_calls",
                "qdrant_total_batch_calls", "qdrant_pertoken_batch_calls",
                "qdrant_coverage_batch_calls", "stage1_request_ms",
                "global_candidate_count", "country_candidate_count",
                "candidate_urls", "final_ranked"):
        assert f'"{key}"' in src
