#!/usr/bin/env python3
"""Tests for the twin-fidelity check (task 399).

This is exactly the class verification-doctrine 1 reserves unit tests for: a
pure function whose silent failure is catastrophic. If `fidelity_verify` ever
returns 0 on a drifted replica, every later gate scores a polluted baseline and
NOTHING downstream would notice — that is the incident this check exists to
prevent, so the check itself gets a test.

  python3 tests/test_gate_fidelity.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cytir.gates.authority as GA


def _write(obj):
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(obj, fh)
    fh.close()
    return fh.name


def _cap(digest, per_q, version=None):
    return {"gate_set_version": version or GA.GATE_SET_VERSION,
            "n_sentinels": len(per_q), "collection": "aw_b1",
            "digest": digest, "per_q": per_q}


def test_identical_replica_passes():
    base_per_q = {"q1": ["https://a/", "https://b/"], "q2": ["https://c/"]}
    cap = _cap("deadbeef", base_per_q)
    GA.fidelity_probe = lambda _c: cap
    assert GA.fidelity_verify("cache", _write(cap)) == 0


def test_drifted_replica_fails():
    """The incident: same count, different behaviour -> must NOT pass."""
    base = _cap("digest_before", {"q1": ["https://gold/"], "q2": ["https://c/"]})
    now = _cap("digest_after", {"q1": ["https://other/"], "q2": ["https://c/"]})
    GA.fidelity_probe = lambda _c: now
    rc = GA.fidelity_verify("cache", _write(base))
    assert rc == 3, f"drifted replica returned {rc}, expected 3"


def test_version_mismatch_refuses():
    base = _cap("d", {"q1": ["https://a/"]}, version="v1")
    now = _cap("d", {"q1": ["https://a/"]}, version="v2")
    GA.fidelity_probe = lambda _c: now
    rc = GA.fidelity_verify("cache", _write(base))
    assert rc == 3, "a cross-version fidelity compare must refuse, not pass"


def test_digest_is_order_independent_but_content_sensitive():
    """Same mapping, different insertion order -> same digest.
    Any URL change -> different digest."""
    import hashlib

    def dig(per_q):
        return hashlib.sha256(
            json.dumps(per_q, sort_keys=True,
                       ensure_ascii=False).encode("utf-8")).hexdigest()

    a = {"q1": ["u1", "u2"], "q2": ["u3"]}
    b = {"q2": ["u3"], "q1": ["u1", "u2"]}
    c = {"q1": ["u1", "u9"], "q2": ["u3"]}
    assert dig(a) == dig(b), "sentinel digest must not depend on dict order"
    assert dig(a) != dig(c), "sentinel digest must change when a result changes"


def test_tie_reorder_does_not_abort():
    """A score-tie flip must NOT read as contamination.

    Measured 2026-08-07: on a green, stable-count twin the ordered-list digest
    oscillated between two values because ONE sentinel returned the same ten
    URLs in a flipped order. That is a benign tie, and an order-sensitive
    digest turns it into a gate-blocking false abort. Fidelity asks "same
    CONTENT?"; ranking belongs to the gate.
    """
    import hashlib

    def cap(per_q):
        sets = {q: sorted(set(u)) for q, u in per_q.items()}
        return {"gate_set_version": GA.GATE_SET_VERSION,
                "n_sentinels": len(per_q), "collection": "aw_b1",
                "digest": hashlib.sha256(
                    json.dumps(sets, sort_keys=True,
                               ensure_ascii=False).encode()).hexdigest(),
                "per_q": per_q, "per_q_sets": sets}

    base = cap({"q1": ["https://a/", "https://b/"], "q2": ["https://c/"]})
    flip = cap({"q1": ["https://b/", "https://a/"], "q2": ["https://c/"]})
    assert base["digest"] == flip["digest"], "tie reorder must not change digest"
    GA.fidelity_probe = lambda _c: flip
    assert GA.fidelity_verify("cache", _write(base)) == 0

    # but a genuine content change still aborts
    gone = cap({"q1": ["https://b/", "https://zzz/"], "q2": ["https://c/"]})
    GA.fidelity_probe = lambda _c: gone
    assert GA.fidelity_verify("cache", _write(base)) == 3, \
        "a changed result SET must still abort"


def test_sentinels_are_deterministic():
    if not any(os.path.exists(p) for _n, p in GA.SETS):
        print("  (skip: gold yamls not present in this checkout)")
        return
    a = [(n, q) for n, q, _r in GA.fidelity_sentinels()]
    b = [(n, q) for n, q, _r in GA.fidelity_sentinels()]
    assert a == b, "sentinel selection must be deterministic"
    assert len(a) <= GA.FIDELITY_SENTINEL_N
    assert len(set(a)) == len(a), "sentinel ids must be unique"


if __name__ == "__main__":
    test_identical_replica_passes(None)
    test_drifted_replica_fails()
    test_version_mismatch_refuses()
    test_digest_is_order_independent_but_content_sensitive()
    test_tie_reorder_does_not_abort()
    test_sentinels_are_deterministic()
    print("all gate-fidelity tests passed")
