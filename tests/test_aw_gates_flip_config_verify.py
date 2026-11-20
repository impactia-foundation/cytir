"""Unit tests for `flip_config_verify.check()` — no network, synthetic
`/health` responses only. The live-fetch half (`fetch_health`) is a single
`urllib.request` call and is exercised for real on flip day, not here.

Run: python3 -m pytest tests/test_aw_gates_flip_config_verify.py -q
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

from cytir.core import serving_config as sc
from cytir.gates import flip_config_verify as vfc

_GOOD_CONFIG = {
    "mrl_dim": 256,
    "dense_rescore": True,
    "dense_oversampling": 1.0,
    "dense_ignore_quant": False,
    "rankhead": True,
    "fusion_weights": {
        "path": "/app/cytir/serve/fusion_weights.candidate519.json",
        "sha256": sc.FLIP_FUSION_WEIGHTS_SHA256,
    },
}


def test_ratified_health_response_passes_clean():
    health = {"status": "ok", "points": 30_895_664, "config": _GOOD_CONFIG}
    assert vfc.check(health) == []


def test_degraded_status_is_reported_and_config_not_consulted():
    problems = vfc.check({"status": "degraded", "error": "boom"})
    assert problems and "degraded" in problems[0]


def test_missing_config_key_on_an_older_api_build_is_reported():
    problems = vfc.check({"status": "ok", "points": 1})
    assert problems and "no 'config' key" in problems[0]


def test_rescore_off_is_caught():
    bad = dict(_GOOD_CONFIG, dense_rescore=False)
    problems = vfc.check({"status": "ok", "config": bad})
    assert any("dense_rescore" in p for p in problems)


def test_wrong_mrl_dim_is_caught():
    bad = dict(_GOOD_CONFIG, mrl_dim=128)
    problems = vfc.check({"status": "ok", "config": bad})
    assert any("mrl_dim" in p for p in problems)


def test_oversampling_drift_is_caught():
    bad = dict(_GOOD_CONFIG, dense_oversampling=3.0)
    problems = vfc.check({"status": "ok", "config": bad})
    assert any("dense_oversampling" in p for p in problems)


def test_unpromoted_fusion_weights_is_caught():
    bad = dict(_GOOD_CONFIG,
              fusion_weights={"path": "/app/cytir/serve/fusion_weights.json",
                             "sha256": "deadbeef" * 8})
    problems = vfc.check({"status": "ok", "config": bad})
    assert any("fusion_weights.sha256" in p for p in problems)


def test_multiple_problems_are_all_reported_not_just_the_first():
    bad = dict(_GOOD_CONFIG, mrl_dim=128, dense_rescore=False)
    problems = vfc.check({"status": "ok", "config": bad})
    assert len(problems) >= 2


# ------------------------------------------ the rollback readback (task 590)
_INCUMBENT_CONFIG = {
    "collection": "aw_prod",
    "mrl_dim": 128,
    "dense_rescore": False,
    "dense_oversampling": 1.0,
    "dense_ignore_quant": False,
    "fusion_weights": {
        "path": "/app/cytir/serve/fusion_weights.json",
        "sha256": vfc._incumbent_weights_sha256(),
    },
}


def test_the_incumbent_config_passes_the_rollback_check():
    health = {"status": "ok", "points": 34_464_401, "config": _INCUMBENT_CONFIG}
    assert vfc.check_incumbent(health) == []


def test_a_rollback_that_left_the_overlay_applied_is_caught():
    """The api at 256 over a 128-dim collection is the outage, in reverse."""
    bad = dict(_INCUMBENT_CONFIG, mrl_dim=256, dense_rescore=True)
    problems = vfc.check_incumbent({"status": "ok", "config": bad})
    assert any("mrl_dim" in p and "overlay is still applied" in p
               for p in problems)
    assert any("dense_rescore" in p for p in problems)


def test_a_rollback_that_kept_the_promoted_weights_is_caught():
    bad = dict(_INCUMBENT_CONFIG,
               fusion_weights={"path": "/app/cytir/serve/x.json",
                               "sha256": sc.FLIP_FUSION_WEIGHTS_SHA256})
    problems = vfc.check_incumbent({"status": "ok", "config": bad})
    assert any("fusion_weights.sha256" in p for p in problems)


def test_a_degraded_health_fails_the_rollback_check_too():
    problems = vfc.check_incumbent({"status": "degraded", "error": "boom"})
    assert problems and "degraded" in problems[0]


def test_the_incumbent_expectations_are_not_typed_in():
    """Both directions read their expectations from tracked sources."""
    assert vfc.INCUMBENT_SERVING_CONFIG in sc.CONFIGS
    assert sc.get(vfc.INCUMBENT_SERVING_CONFIG).dense_dim == 128
    weights = (pathlib.Path(vfc.__file__).resolve().parents[2]
               / "cytir" / "serve" / "fusion_weights.json")
    assert vfc._incumbent_weights_sha256() == hashlib.sha256(
        weights.read_bytes()).hexdigest()


# --------------------------- the alias-vs-physical-collection gap (task 590)
def _health_stub(config):
    return lambda url, timeout=10.0: {"status": "ok", "points": 1,
                                      "config": config}


def _record_collection(seen):
    def fake(verdict_path=None, *, collection=None, config_name=None):
        seen["collection"] = collection
        return []
    return fake


def test_the_gate_is_checked_against_the_physical_collection_not_the_alias(
        monkeypatch):
    """/health reports `aw_prod`; the sealed verdict is bound to `aw_e8`.

    Without `--gate-collection` the gate check compares the ALIAS name to the
    candidate the battery scored and refuses every time -- a flip-day blocker
    that reads like tampering. flip-release.sh passes the collection it has
    just proved the alias resolves to.
    """
    seen = {}
    monkeypatch.setattr(vfc, "fetch_health",
                        _health_stub(dict(_GOOD_CONFIG, collection="aw_prod")))
    monkeypatch.setattr(vfc, "gate_problems", _record_collection(seen))
    monkeypatch.setattr(sys, "argv",
                        ["verify_flip_config.py", "--url", "http://x",
                         "--gate-collection", "aw_e8"])
    assert vfc.main() == 0
    assert seen["collection"] == "aw_e8"


def test_without_the_flag_the_gate_still_sees_only_the_alias(monkeypatch):
    """Documents the fallback, so nobody 'fixes' it by dropping the flag."""
    seen = {}
    monkeypatch.setattr(vfc, "fetch_health",
                        _health_stub(dict(_GOOD_CONFIG, collection="aw_prod")))
    monkeypatch.setattr(vfc, "gate_problems", _record_collection(seen))
    monkeypatch.setattr(sys, "argv",
                        ["verify_flip_config.py", "--url", "http://x"])
    assert vfc.main() == 0
    assert seen["collection"] == "aw_prod"


def test_rollback_mode_does_not_consult_the_flip_gate(monkeypatch):
    """A rollback to what prod already served is always permitted."""
    called = []

    def refuse(*a, **k):
        called.append(1)
        return ["nope"]

    monkeypatch.setattr(vfc, "fetch_health", _health_stub(_INCUMBENT_CONFIG))
    monkeypatch.setattr(vfc, "gate_problems", refuse)
    monkeypatch.setattr(sys, "argv",
                        ["verify_flip_config.py", "--url", "http://x",
                         "--rollback"])
    assert vfc.main() == 0
    assert called == []


def test_there_is_no_flag_to_skip_the_gate():
    src = pathlib.Path(vfc.__file__).read_text()
    for token in ("--skip-gate", "--no-gate", "--force", "--waive"):
        assert token not in src, f"{token} appeared in verify_flip_config"
