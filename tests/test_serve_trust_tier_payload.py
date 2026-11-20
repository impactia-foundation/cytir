#!/usr/bin/env python3
"""Regression coverage for null historic trust payloads (task 364)."""
import os
import sys

os.environ.update({"AW_EMBEDDER": "arctic", "AW_COLLECTION": "test_trust_payload"})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cytir.serve import retrieval  # noqa: E402


def test_unknown_trust_defaults_without_rewriting_zero():
    assert retrieval._trust_tier({}) == 0.6
    assert retrieval._trust_tier({"trust_tier": None}) == 0.6
    assert retrieval._trust_tier({"trust_tier": 0.0}) == 0.0
    assert retrieval._trust_tier({"trust_tier": 0.82}) == 0.82
